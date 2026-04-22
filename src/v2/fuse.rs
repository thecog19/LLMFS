//! FUSE driver exposing a [`Filesystem`] as a mountable POSIX tree.
//!
//! # Op model
//!
//! fuser 0.15's dispatch loop reads one kernel request at a time and
//! invokes our trait method synchronously. The driver works *with*
//! that single-threaded loop by handing each request's actual work to
//! a freshly spawned worker thread — every FUSE method is a tiny
//! wrapper that clones the shared state, captures the `Reply*`
//! object, spawns, and returns. The dispatch loop is then free to
//! pick up the next request while previous work is still in flight.
//!
//! Concurrency model:
//! - Many readers may run in parallel against [`Filesystem`]'s
//!   `&self` methods (`read_file`, `readdir`, `inode_at`, `exists`).
//! - A writer (`mkdir` / `rmdir` / `create_file` / `unlink`) takes
//!   the exclusive [`RwLock`] on the filesystem; readers wait until
//!   it commits. V2's CoW commit semantics already require this.
//! - The [`InodeMap`] and the per-inode write buffer map have their
//!   own locks; concurrent reads and concurrent writes to *different*
//!   inodes don't serialize on each other.
//!
//! Lock acquisition order, when a method needs more than one:
//! `inodes` → `buffers` → `fs`. Document any new method that holds
//! more than one lock at a time, or it's easy to introduce a deadlock.
//!
//! # Buffered writes
//!
//! FUSE writes are offset-based; V2 rewrites whole files per commit.
//! The bridge is a per-inode [`WriteBuffer`] seeded on first open /
//! create. Writes splice into the buffer; `release` / `flush` /
//! `fsync` commit via [`Filesystem::create_file`]. Unreleased buffers
//! live in memory until drop.
//!
//! # Attributes V2 doesn't track
//!
//! V2 inodes store length + pointers, nothing else. `mode`, `mtime`,
//! `atime`, `ctime`, `uid`, `gid` are fabricated per-session (sane
//! defaults + the session's mount time). `setattr` on mode/size is
//! partially honoured — mode is stored only in the write buffer
//! (lost on release if contents aren't dirty); size truncates the
//! buffer.
//!
//! # Mounting
//!
//! [`mount_foreground`] blocks the calling thread until the kernel
//! releases the mount (e.g. `fusermount3 -u`). [`spawn_background`]
//! returns a [`BackgroundSession`] guard that unmounts on drop —
//! useful for tests and for CLI drivers that want to manage the mount
//! lifetime explicitly.

use std::collections::HashMap;
use std::ffi::{OsStr, OsString};
use std::path::Path;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, SystemTime};

use fuser::{
    BackgroundSession, FileAttr, FileType, Filesystem as FuserFilesystem, MountOption, ReplyAttr,
    ReplyCreate, ReplyData, ReplyDirectory, ReplyEmpty, ReplyEntry, ReplyOpen, ReplyStatfs,
    ReplyWrite, Request, Session, TimeOrNow,
};

use crate::v2::directory::EntryKind;
use crate::v2::fs::{Filesystem, FsError};

/// Shared handle to the underlying [`Filesystem`]. Exposed so the
/// CLI can [`share`](LlmdbV2Fs::share) a handle before consuming the
/// driver in `spawn_background`, and recover the filesystem + cover
/// bytes once the mount releases.
pub type SharedFilesystem = Arc<RwLock<Filesystem>>;

/// Shared inode-map handle. Reads (`path_of`, `ino_for_path`) take
/// the read lock; mutations (`intern`, `forget`) take the write lock.
type SharedInodes = Arc<RwLock<InodeMap>>;

/// Per-inode write-buffer map. The outer `RwLock` guards the map's
/// structure; each buffer lives behind its own `Mutex` so concurrent
/// writes to *different* inodes don't serialize on each other.
type BufferMap = Arc<RwLock<HashMap<u64, Arc<Mutex<WriteBuffer>>>>>;

/// The kernel reserves inode 1 for the root directory.
pub const ROOT_INO: u64 = 1;

/// How long the kernel may cache entry + attr replies before asking
/// again. One second is conservative — V2 commits bump the tree
/// generation so cached attrs can go stale mid-mount.
const TTL: Duration = Duration::from_secs(1);

/// FUSE `generation` for every entry. We don't reuse inode numbers
/// across a session, so a constant zero is fine.
const GENERATION: u64 = 0;

/// Default file mode when V2 has no stored mode.
const DEFAULT_FILE_MODE: u16 = 0o644;

/// Default directory mode when V2 has no stored mode.
const DEFAULT_DIR_MODE: u16 = 0o755;

/// Block size used for `stat.st_blksize` / `statvfs.f_bsize`.
const BLOCK_SIZE: u32 = 4096;

/// Reported max filename length. V2's `MAX_NAME_LEN` is 255.
const MAX_FILENAME_BYTES: u32 = 255;

// ==================================================================
// Path ↔ inode map
// ==================================================================

/// Session-scoped translation between kernel `u64` inodes and V2
/// paths. Inodes are assigned on first sight and stay stable for the
/// life of the mount; remounting resets the map.
#[derive(Debug)]
struct InodeMap {
    next_ino: u64,
    path_to_ino: HashMap<String, u64>,
    ino_to_path: HashMap<u64, String>,
}

impl Default for InodeMap {
    fn default() -> Self {
        let mut map = Self {
            next_ino: ROOT_INO + 1,
            path_to_ino: HashMap::new(),
            ino_to_path: HashMap::new(),
        };
        map.path_to_ino.insert("/".to_owned(), ROOT_INO);
        map.ino_to_path.insert(ROOT_INO, "/".to_owned());
        map
    }
}

impl InodeMap {
    fn intern(&mut self, path: &str) -> u64 {
        if let Some(ino) = self.path_to_ino.get(path) {
            return *ino;
        }
        let ino = self.next_ino;
        self.next_ino = self
            .next_ino
            .checked_add(1)
            .expect("u64 inode overflow — one session can't go that long");
        self.path_to_ino.insert(path.to_owned(), ino);
        self.ino_to_path.insert(ino, path.to_owned());
        ino
    }

    fn path_of(&self, ino: u64) -> Option<&str> {
        self.ino_to_path.get(&ino).map(String::as_str)
    }

    fn ino_for_path(&self, path: &str) -> Option<u64> {
        self.path_to_ino.get(path).copied()
    }

    fn forget(&mut self, ino: u64) {
        if let Some(path) = self.ino_to_path.remove(&ino) {
            self.path_to_ino.remove(&path);
        }
    }
}

/// Join a parent path with a child name into a well-formed absolute
/// path. `join("/", "c")` = `/c`, `join("/a/b", "c")` = `/a/b/c`.
fn join_path(parent: &str, name: &str) -> String {
    if parent == "/" {
        format!("/{name}")
    } else {
        format!("{parent}/{name}")
    }
}

// ==================================================================
// Write buffer
// ==================================================================

#[derive(Debug)]
struct WriteBuffer {
    path: String,
    mode: u16,
    contents: Vec<u8>,
    dirty: bool,
}

// ==================================================================
// Driver
// ==================================================================

pub struct LlmdbV2Fs {
    fs: SharedFilesystem,
    inodes: SharedInodes,
    /// Per-inode pending write buffer. Offset writes stage here;
    /// `release` / `flush` / `fsync` commits via
    /// [`Filesystem::create_file`].
    buffers: BufferMap,
    /// Session start — used as a fabricated timestamp for every
    /// attribute reply until V2 starts tracking real mtime/ctime.
    mount_time: SystemTime,
}

impl LlmdbV2Fs {
    /// Wrap a V2 [`Filesystem`] for FUSE dispatch. The filesystem is
    /// moved behind an `Arc<RwLock<_>>` so the adapter can co-exist
    /// with an out-of-band CLI handle — see [`Self::share`].
    pub fn new(fs: Filesystem) -> Self {
        Self::with_shared(Arc::new(RwLock::new(fs)))
    }

    /// Build a driver around an already-shared filesystem handle.
    /// Use this when a caller (e.g. the CLI) needs to hold onto a
    /// handle that survives the FUSE session — after `drop(session)`
    /// returns, the caller's `Arc` is the last reference and can
    /// [`Arc::try_unwrap`] the filesystem back.
    pub fn with_shared(fs: SharedFilesystem) -> Self {
        Self {
            fs,
            inodes: Arc::new(RwLock::new(InodeMap::default())),
            buffers: Arc::new(RwLock::new(HashMap::new())),
            mount_time: SystemTime::now(),
        }
    }

    /// Clone the underlying shared filesystem handle. See
    /// [`Self::with_shared`].
    pub fn share(&self) -> SharedFilesystem {
        Arc::clone(&self.fs)
    }

    /// Consume the driver and return the underlying filesystem.
    /// Panics if another `Arc` to the shared filesystem is alive —
    /// callers using [`Self::share`] should drop their handle (or
    /// the [`BackgroundSession`]) first.
    pub fn into_inner(self) -> Filesystem {
        let lock = Arc::try_unwrap(self.fs)
            .expect("into_inner called while another Arc<RwLock<Filesystem>> is alive");
        lock.into_inner().expect("filesystem RwLock poisoned")
    }

    /// Take a snapshot of the shared state needed to dispatch a
    /// request on a worker thread. Cheap — clones three Arcs and one
    /// SystemTime.
    fn ctx(&self) -> Ctx {
        Ctx {
            fs: Arc::clone(&self.fs),
            inodes: Arc::clone(&self.inodes),
            buffers: Arc::clone(&self.buffers),
            mount_time: self.mount_time,
        }
    }
}

/// Per-request execution context. Cloned into a worker thread for
/// each FUSE call; outlives the synchronous trait method.
#[derive(Clone)]
struct Ctx {
    fs: SharedFilesystem,
    inodes: SharedInodes,
    buffers: BufferMap,
    mount_time: SystemTime,
}

impl Ctx {
    /// Get a clone of the per-inode write buffer Arc, if one exists.
    /// Releases the outer map's read lock immediately.
    fn buffer(&self, ino: u64) -> Option<Arc<Mutex<WriteBuffer>>> {
        self.buffers.read().unwrap().get(&ino).cloned()
    }

    fn dir_attr(&self, ino: u64, uid: u32, gid: u32) -> FileAttr {
        FileAttr {
            ino,
            size: 0,
            blocks: 0,
            atime: self.mount_time,
            mtime: self.mount_time,
            ctime: self.mount_time,
            crtime: self.mount_time,
            kind: FileType::Directory,
            perm: DEFAULT_DIR_MODE,
            nlink: 2,
            uid,
            gid,
            rdev: 0,
            blksize: BLOCK_SIZE,
            flags: 0,
        }
    }

    /// Build a file attr. Reads the inode under `fs` read lock and
    /// merges with the buffer (if present) for size and mode.
    fn file_attr(&self, ino: u64, path: &str, uid: u32, gid: u32) -> Option<FileAttr> {
        let inode = self.fs.read().unwrap().inode_at(path).ok()?;
        let (buf_size, buf_mode) = match self.buffer(ino) {
            Some(b) => {
                let b = b.lock().unwrap();
                let size = b.dirty.then_some(b.contents.len() as u64);
                (size, Some(b.mode))
            }
            None => (None, None),
        };
        let size = buf_size.unwrap_or(inode.length);
        let mode = buf_mode.unwrap_or(DEFAULT_FILE_MODE);
        Some(FileAttr {
            ino,
            size,
            blocks: size.div_ceil(BLOCK_SIZE as u64),
            atime: self.mount_time,
            mtime: self.mount_time,
            ctime: self.mount_time,
            crtime: self.mount_time,
            kind: FileType::RegularFile,
            perm: mode,
            nlink: 1,
            uid,
            gid,
            rdev: 0,
            blksize: BLOCK_SIZE,
            flags: 0,
        })
    }

    /// Read the live contents for `ino`/`path` — buffer if present,
    /// otherwise the on-disk file via the V2 fs read lock.
    fn read_contents(&self, ino: u64, path: &str) -> Result<Vec<u8>, FsError> {
        if let Some(buf) = self.buffer(ino) {
            let b = buf.lock().unwrap();
            return Ok(b.contents.clone());
        }
        self.fs.read().unwrap().read_file(path)
    }

    /// Commit `ino`'s pending write buffer to V2. No-op if no buffer
    /// exists or it's clean. Holds the V2 write lock during the
    /// commit; per-buffer Mutex is held only to snapshot contents.
    fn flush_buffer(&self, ino: u64) -> Result<(), FsError> {
        let buf = match self.buffer(ino) {
            Some(b) => b,
            None => return Ok(()),
        };
        let (path, contents) = {
            let b = buf.lock().unwrap();
            if !b.dirty {
                return Ok(());
            }
            (b.path.clone(), b.contents.clone())
        };
        self.fs.write().unwrap().create_file(&path, &contents)?;
        let mut b = buf.lock().unwrap();
        b.dirty = false;
        Ok(())
    }
}

// ==================================================================
// FsError → errno
// ==================================================================

fn fs_err_to_errno(err: &FsError) -> i32 {
    match err {
        FsError::PathNotFound(_) => libc::ENOENT,
        FsError::NotADirectory(_) => libc::ENOTDIR,
        FsError::IsADirectory(_) => libc::EISDIR,
        FsError::AlreadyExists(_) => libc::EEXIST,
        FsError::DirectoryNotEmpty(_) => libc::ENOTEMPTY,
        FsError::InvalidPath(_) => libc::EINVAL,
        FsError::PathCannotBeRoot => libc::EBUSY,
        FsError::FileTooLarge { .. } | FsError::OutOfSpace { .. } => libc::ENOSPC,
        FsError::Directory(_) => libc::EINVAL,
        _ => libc::EIO,
    }
}

// ==================================================================
// fuser::Filesystem impl — every method spawns its work
// ==================================================================

impl FuserFilesystem for LlmdbV2Fs {
    fn lookup(&mut self, req: &Request<'_>, parent: u64, name: &OsStr, reply: ReplyEntry) {
        let ctx = self.ctx();
        let uid = req.uid();
        let gid = req.gid();
        let name = name.to_owned();
        thread::spawn(move || do_lookup(ctx, parent, name, uid, gid, reply));
    }

    fn forget(&mut self, _req: &Request<'_>, ino: u64, _nlookup: u64) {
        // Trivial map-pop — no reply, no need to spawn.
        self.inodes.write().unwrap().forget(ino);
    }

    fn getattr(&mut self, req: &Request<'_>, ino: u64, _fh: Option<u64>, reply: ReplyAttr) {
        let ctx = self.ctx();
        let uid = req.uid();
        let gid = req.gid();
        thread::spawn(move || do_getattr(ctx, ino, uid, gid, reply));
    }

    fn opendir(&mut self, _req: &Request<'_>, _ino: u64, _flags: i32, reply: ReplyOpen) {
        reply.opened(0, 0);
    }

    fn readdir(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        _fh: u64,
        offset: i64,
        reply: ReplyDirectory,
    ) {
        let ctx = self.ctx();
        thread::spawn(move || do_readdir(ctx, ino, offset, reply));
    }

    fn open(&mut self, _req: &Request<'_>, ino: u64, _flags: i32, reply: ReplyOpen) {
        let ctx = self.ctx();
        thread::spawn(move || {
            if ctx.inodes.read().unwrap().path_of(ino).is_none() {
                reply.error(libc::ENOENT);
            } else {
                reply.opened(0, 0);
            }
        });
    }

    fn read(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        _fh: u64,
        offset: i64,
        size: u32,
        _flags: i32,
        _lock_owner: Option<u64>,
        reply: ReplyData,
    ) {
        let ctx = self.ctx();
        thread::spawn(move || do_read(ctx, ino, offset, size, reply));
    }

    fn write(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        _fh: u64,
        offset: i64,
        data: &[u8],
        _write_flags: u32,
        _flags: i32,
        _lock_owner: Option<u64>,
        reply: ReplyWrite,
    ) {
        let ctx = self.ctx();
        let data = data.to_vec();
        thread::spawn(move || do_write(ctx, ino, offset, data, reply));
    }

    fn flush(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        _fh: u64,
        _lock_owner: u64,
        reply: ReplyEmpty,
    ) {
        let ctx = self.ctx();
        thread::spawn(move || match ctx.flush_buffer(ino) {
            Ok(()) => reply.ok(),
            Err(err) => reply.error(fs_err_to_errno(&err)),
        });
    }

    fn fsync(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        _fh: u64,
        _datasync: bool,
        reply: ReplyEmpty,
    ) {
        let ctx = self.ctx();
        thread::spawn(move || match ctx.flush_buffer(ino) {
            Ok(()) => reply.ok(),
            Err(err) => reply.error(fs_err_to_errno(&err)),
        });
    }

    fn release(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        _fh: u64,
        _flags: i32,
        _lock_owner: Option<u64>,
        _flush: bool,
        reply: ReplyEmpty,
    ) {
        let ctx = self.ctx();
        thread::spawn(move || {
            if let Err(err) = ctx.flush_buffer(ino) {
                reply.error(fs_err_to_errno(&err));
                return;
            }
            ctx.buffers.write().unwrap().remove(&ino);
            reply.ok();
        });
    }

    fn create(
        &mut self,
        req: &Request<'_>,
        parent: u64,
        name: &OsStr,
        mode: u32,
        _umask: u32,
        _flags: i32,
        reply: ReplyCreate,
    ) {
        let ctx = self.ctx();
        let uid = req.uid();
        let gid = req.gid();
        let name = name.to_owned();
        thread::spawn(move || do_create(ctx, parent, name, mode, uid, gid, reply));
    }

    fn unlink(&mut self, _req: &Request<'_>, parent: u64, name: &OsStr, reply: ReplyEmpty) {
        let ctx = self.ctx();
        let name = name.to_owned();
        thread::spawn(move || do_unlink(ctx, parent, name, reply));
    }

    fn mkdir(
        &mut self,
        req: &Request<'_>,
        parent: u64,
        name: &OsStr,
        _mode: u32,
        _umask: u32,
        reply: ReplyEntry,
    ) {
        let ctx = self.ctx();
        let uid = req.uid();
        let gid = req.gid();
        let name = name.to_owned();
        thread::spawn(move || do_mkdir(ctx, parent, name, uid, gid, reply));
    }

    fn rmdir(&mut self, _req: &Request<'_>, parent: u64, name: &OsStr, reply: ReplyEmpty) {
        let ctx = self.ctx();
        let name = name.to_owned();
        thread::spawn(move || do_rmdir(ctx, parent, name, reply));
    }

    fn setattr(
        &mut self,
        req: &Request<'_>,
        ino: u64,
        mode: Option<u32>,
        _uid: Option<u32>,
        _gid: Option<u32>,
        size: Option<u64>,
        _atime: Option<TimeOrNow>,
        _mtime: Option<TimeOrNow>,
        _ctime: Option<SystemTime>,
        _fh: Option<u64>,
        _crtime: Option<SystemTime>,
        _chgtime: Option<SystemTime>,
        _bkuptime: Option<SystemTime>,
        _flags: Option<u32>,
        reply: ReplyAttr,
    ) {
        let ctx = self.ctx();
        let uid = req.uid();
        let gid = req.gid();
        thread::spawn(move || do_setattr(ctx, ino, mode, size, uid, gid, reply));
    }

    fn statfs(&mut self, _req: &Request<'_>, _ino: u64, reply: ReplyStatfs) {
        let ctx = self.ctx();
        thread::spawn(move || {
            let free_weights = ctx.fs.read().unwrap().allocator_free_weights();
            let free_blocks = free_weights.saturating_div(BLOCK_SIZE as u64);
            reply.statfs(
                free_blocks.saturating_mul(2),
                free_blocks,
                free_blocks,
                0,
                0,
                BLOCK_SIZE,
                MAX_FILENAME_BYTES,
                BLOCK_SIZE,
            );
        });
    }

    fn access(&mut self, _req: &Request<'_>, _ino: u64, _mask: i32, reply: ReplyEmpty) {
        reply.ok();
    }
}

// ==================================================================
// Per-method work bodies (run on worker threads)
// ==================================================================

fn do_lookup(
    ctx: Ctx,
    parent: u64,
    name: OsString,
    uid: u32,
    gid: u32,
    reply: ReplyEntry,
) {
    let parent_path = match ctx.inodes.read().unwrap().path_of(parent).map(str::to_owned) {
        Some(p) => p,
        None => {
            reply.error(libc::ENOENT);
            return;
        }
    };
    let Some(name_str) = name.to_str() else {
        reply.error(libc::EINVAL);
        return;
    };
    let path = join_path(&parent_path, name_str);

    let entries = match ctx.fs.read().unwrap().readdir(&parent_path) {
        Ok(e) => e,
        Err(_) => {
            reply.error(libc::ENOENT);
            return;
        }
    };
    let Some(entry) = entries.iter().find(|e| e.name == name_str) else {
        reply.error(libc::ENOENT);
        return;
    };
    let kind = entry.kind;

    let ino = ctx.inodes.write().unwrap().intern(&path);
    let attr = match kind {
        EntryKind::Directory => ctx.dir_attr(ino, uid, gid),
        EntryKind::File => match ctx.file_attr(ino, &path, uid, gid) {
            Some(a) => a,
            None => {
                reply.error(libc::EIO);
                return;
            }
        },
    };
    reply.entry(&TTL, &attr, GENERATION);
}

fn do_getattr(ctx: Ctx, ino: u64, uid: u32, gid: u32, reply: ReplyAttr) {
    if ino == ROOT_INO {
        let attr = ctx.dir_attr(ino, uid, gid);
        reply.attr(&TTL, &attr);
        return;
    }
    let path = match ctx.inodes.read().unwrap().path_of(ino).map(str::to_owned) {
        Some(p) => p,
        None => {
            reply.error(libc::ENOENT);
            return;
        }
    };
    let fs = ctx.fs.read().unwrap();
    if !fs.exists(&path) {
        reply.error(libc::ENOENT);
        return;
    }
    let inode = match fs.inode_at(&path) {
        Ok(i) => i,
        Err(_) => {
            reply.error(libc::ENOENT);
            return;
        }
    };
    // Heuristic: try readdir; if it succeeds, it's a directory.
    if fs.readdir(&path).is_ok() {
        drop(fs);
        let attr = ctx.dir_attr(ino, uid, gid);
        reply.attr(&TTL, &attr);
        return;
    }
    drop(fs);

    let (buf_size, buf_mode) = match ctx.buffer(ino) {
        Some(b) => {
            let b = b.lock().unwrap();
            (b.dirty.then_some(b.contents.len() as u64), Some(b.mode))
        }
        None => (None, None),
    };
    let size = buf_size.unwrap_or(inode.length);
    let mode = buf_mode.unwrap_or(DEFAULT_FILE_MODE);
    let attr = FileAttr {
        ino,
        size,
        blocks: size.div_ceil(BLOCK_SIZE as u64),
        atime: ctx.mount_time,
        mtime: ctx.mount_time,
        ctime: ctx.mount_time,
        crtime: ctx.mount_time,
        kind: FileType::RegularFile,
        perm: mode,
        nlink: 1,
        uid,
        gid,
        rdev: 0,
        blksize: BLOCK_SIZE,
        flags: 0,
    };
    reply.attr(&TTL, &attr);
}

fn do_readdir(ctx: Ctx, ino: u64, offset: i64, mut reply: ReplyDirectory) {
    let dir_path = match ctx.inodes.read().unwrap().path_of(ino).map(str::to_owned) {
        Some(p) => p,
        None => {
            reply.error(libc::ENOENT);
            return;
        }
    };
    let entries = match ctx.fs.read().unwrap().readdir(&dir_path) {
        Ok(e) => e,
        Err(err) => {
            reply.error(fs_err_to_errno(&err));
            return;
        }
    };

    let dot: [(&str, FileType); 2] = [(".", FileType::Directory), ("..", FileType::Directory)];

    let mut idx = 0_i64;
    let mut next_offset = 1_i64;
    for (name, kind) in dot.iter() {
        if idx >= offset && reply.add(ino, next_offset, *kind, *name) {
            reply.ok();
            return;
        }
        idx += 1;
        next_offset += 1;
    }
    for entry in &entries {
        if idx < offset {
            idx += 1;
            next_offset += 1;
            continue;
        }
        let child_path = join_path(&dir_path, &entry.name);
        let child_ino = ctx.inodes.write().unwrap().intern(&child_path);
        let kind = match entry.kind {
            EntryKind::File => FileType::RegularFile,
            EntryKind::Directory => FileType::Directory,
        };
        if reply.add(child_ino, next_offset, kind, &entry.name) {
            break;
        }
        next_offset += 1;
    }
    reply.ok();
}

fn do_read(ctx: Ctx, ino: u64, offset: i64, size: u32, reply: ReplyData) {
    let path = match ctx.inodes.read().unwrap().path_of(ino).map(str::to_owned) {
        Some(p) => p,
        None => {
            reply.error(libc::ENOENT);
            return;
        }
    };
    let contents = match ctx.read_contents(ino, &path) {
        Ok(c) => c,
        Err(err) => {
            reply.error(fs_err_to_errno(&err));
            return;
        }
    };
    let start = offset.max(0) as usize;
    if start >= contents.len() {
        reply.data(&[]);
        return;
    }
    let end = start.saturating_add(size as usize).min(contents.len());
    reply.data(&contents[start..end]);
}

fn do_write(ctx: Ctx, ino: u64, offset: i64, data: Vec<u8>, reply: ReplyWrite) {
    let path = match ctx.inodes.read().unwrap().path_of(ino).map(str::to_owned) {
        Some(p) => p,
        None => {
            reply.error(libc::ENOENT);
            return;
        }
    };
    let start = offset.max(0) as usize;

    // Get-or-insert the buffer. We acquire the write lock briefly to
    // insert; per-buffer Mutex handles the rest.
    let buf = {
        let mut map = ctx.buffers.write().unwrap();
        if let Some(b) = map.get(&ino) {
            Arc::clone(b)
        } else {
            // Seed the new buffer from on-disk contents (read fs lock).
            let existing = ctx.fs.read().unwrap().read_file(&path).unwrap_or_default();
            let buf = Arc::new(Mutex::new(WriteBuffer {
                path: path.clone(),
                mode: DEFAULT_FILE_MODE,
                contents: existing,
                dirty: false,
            }));
            map.insert(ino, Arc::clone(&buf));
            buf
        }
    };

    let mut b = buf.lock().unwrap();
    let end = start + data.len();
    if b.contents.len() < end {
        b.contents.resize(end, 0);
    }
    b.contents[start..end].copy_from_slice(&data);
    b.dirty = true;
    reply.written(data.len() as u32);
}

fn do_create(
    ctx: Ctx,
    parent: u64,
    name: OsString,
    mode: u32,
    uid: u32,
    gid: u32,
    reply: ReplyCreate,
) {
    let parent_path = match ctx.inodes.read().unwrap().path_of(parent).map(str::to_owned) {
        Some(p) => p,
        None => {
            reply.error(libc::EINVAL);
            return;
        }
    };
    let Some(name_str) = name.to_str() else {
        reply.error(libc::EINVAL);
        return;
    };
    let path = join_path(&parent_path, name_str);

    {
        let fs = ctx.fs.read().unwrap();
        if fs.exists(&path) {
            reply.error(libc::EEXIST);
            return;
        }
    }

    if let Err(err) = ctx.fs.write().unwrap().create_file(&path, &[]) {
        reply.error(fs_err_to_errno(&err));
        return;
    }

    let ino = ctx.inodes.write().unwrap().intern(&path);
    let buf_mode = (mode & 0o7777) as u16;
    ctx.buffers.write().unwrap().insert(
        ino,
        Arc::new(Mutex::new(WriteBuffer {
            path: path.clone(),
            mode: buf_mode,
            contents: Vec::new(),
            dirty: false,
        })),
    );
    let attr = FileAttr {
        ino,
        size: 0,
        blocks: 0,
        atime: ctx.mount_time,
        mtime: ctx.mount_time,
        ctime: ctx.mount_time,
        crtime: ctx.mount_time,
        kind: FileType::RegularFile,
        perm: buf_mode,
        nlink: 1,
        uid,
        gid,
        rdev: 0,
        blksize: BLOCK_SIZE,
        flags: 0,
    };
    reply.created(&TTL, &attr, GENERATION, 0, 0);
}

fn do_unlink(ctx: Ctx, parent: u64, name: OsString, reply: ReplyEmpty) {
    let parent_path = match ctx.inodes.read().unwrap().path_of(parent).map(str::to_owned) {
        Some(p) => p,
        None => {
            reply.error(libc::ENOENT);
            return;
        }
    };
    let Some(name_str) = name.to_str() else {
        reply.error(libc::ENOENT);
        return;
    };
    let path = join_path(&parent_path, name_str);

    let ino_opt = ctx.inodes.read().unwrap().ino_for_path(&path);
    if let Some(ino) = ino_opt {
        ctx.buffers.write().unwrap().remove(&ino);
        ctx.inodes.write().unwrap().forget(ino);
    }
    match ctx.fs.write().unwrap().unlink(&path) {
        Ok(()) => reply.ok(),
        Err(err) => reply.error(fs_err_to_errno(&err)),
    }
}

fn do_mkdir(
    ctx: Ctx,
    parent: u64,
    name: OsString,
    uid: u32,
    gid: u32,
    reply: ReplyEntry,
) {
    let parent_path = match ctx.inodes.read().unwrap().path_of(parent).map(str::to_owned) {
        Some(p) => p,
        None => {
            reply.error(libc::EINVAL);
            return;
        }
    };
    let Some(name_str) = name.to_str() else {
        reply.error(libc::EINVAL);
        return;
    };
    let path = join_path(&parent_path, name_str);

    if let Err(err) = ctx.fs.write().unwrap().mkdir(&path) {
        reply.error(fs_err_to_errno(&err));
        return;
    }
    let ino = ctx.inodes.write().unwrap().intern(&path);
    let attr = ctx.dir_attr(ino, uid, gid);
    reply.entry(&TTL, &attr, GENERATION);
}

fn do_rmdir(ctx: Ctx, parent: u64, name: OsString, reply: ReplyEmpty) {
    let parent_path = match ctx.inodes.read().unwrap().path_of(parent).map(str::to_owned) {
        Some(p) => p,
        None => {
            reply.error(libc::ENOENT);
            return;
        }
    };
    let Some(name_str) = name.to_str() else {
        reply.error(libc::ENOENT);
        return;
    };
    let path = join_path(&parent_path, name_str);

    match ctx.fs.write().unwrap().rmdir(&path) {
        Ok(()) => {
            let ino_opt = ctx.inodes.read().unwrap().ino_for_path(&path);
            if let Some(ino) = ino_opt {
                ctx.inodes.write().unwrap().forget(ino);
            }
            reply.ok();
        }
        Err(err) => reply.error(fs_err_to_errno(&err)),
    }
}

fn do_setattr(
    ctx: Ctx,
    ino: u64,
    mode: Option<u32>,
    size: Option<u64>,
    uid: u32,
    gid: u32,
    reply: ReplyAttr,
) {
    let path = match ctx.inodes.read().unwrap().path_of(ino).map(str::to_owned) {
        Some(p) => p,
        None => {
            reply.error(libc::ENOENT);
            return;
        }
    };

    if size.is_some() || mode.is_some() {
        // Get-or-insert the buffer (seeded from disk if absent).
        let buf = {
            let mut map = ctx.buffers.write().unwrap();
            if let Some(b) = map.get(&ino) {
                Arc::clone(b)
            } else {
                let existing = ctx.fs.read().unwrap().read_file(&path).unwrap_or_default();
                let buf = Arc::new(Mutex::new(WriteBuffer {
                    path: path.clone(),
                    mode: DEFAULT_FILE_MODE,
                    contents: existing,
                    dirty: false,
                }));
                map.insert(ino, Arc::clone(&buf));
                buf
            }
        };
        let mut b = buf.lock().unwrap();
        if let Some(new_size) = size {
            b.contents.resize(new_size as usize, 0);
            b.dirty = true;
        }
        if let Some(new_mode) = mode {
            b.mode = (new_mode & 0o7777) as u16;
            b.dirty = true;
        }
    }

    if let Some(attr) = ctx.file_attr(ino, &path, uid, gid) {
        reply.attr(&TTL, &attr);
    } else if ctx.fs.read().unwrap().readdir(&path).is_ok() {
        reply.attr(&TTL, &ctx.dir_attr(ino, uid, gid));
    } else {
        reply.error(libc::ENOENT);
    }
}

// ==================================================================
// Mount helpers
// ==================================================================

/// Mount options always applied. `FSName` tags the mount for
/// `findmnt` output; `DefaultPermissions` delegates permission
/// checks to the kernel; `NoAtime` avoids spurious metadata churn.
fn base_options() -> Vec<MountOption> {
    vec![
        MountOption::FSName("llmdb-v2".to_owned()),
        MountOption::DefaultPermissions,
        MountOption::NoAtime,
    ]
}

pub struct MountConfig {
    pub allow_other: bool,
}

impl MountConfig {
    pub fn options(&self) -> Vec<MountOption> {
        let mut opts = base_options();
        if self.allow_other {
            opts.push(MountOption::AllowOther);
        }
        opts
    }
}

/// Mount the driver at `mount_point` and block the calling thread
/// until the kernel releases the mount.
pub fn mount_foreground(
    fs: LlmdbV2Fs,
    mount_point: impl AsRef<Path>,
    config: &MountConfig,
) -> std::io::Result<()> {
    fuser::mount2(fs, mount_point, &config.options())
}

/// Mount the driver on a background thread. The returned session
/// unmounts on drop — callers can stash it and use the mount point
/// normally while it's alive.
pub fn spawn_background(
    fs: LlmdbV2Fs,
    mount_point: impl AsRef<Path>,
    config: &MountConfig,
) -> std::io::Result<BackgroundSession> {
    let session = Session::new(fs, mount_point.as_ref(), &config.options())?;
    session.spawn()
}

// ==================================================================
// Unit tests (path + inode map only; no real mount required)
// ==================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_maps_to_ino_1() {
        let map = InodeMap::default();
        assert_eq!(map.path_of(ROOT_INO), Some("/"));
    }

    #[test]
    fn intern_is_idempotent() {
        let mut map = InodeMap::default();
        let a = map.intern("/foo");
        let b = map.intern("/foo");
        let c = map.intern("/bar");
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, ROOT_INO);
    }

    #[test]
    fn forget_drops_both_directions() {
        let mut map = InodeMap::default();
        let ino = map.intern("/victim");
        assert!(map.path_of(ino).is_some());
        map.forget(ino);
        assert!(map.path_of(ino).is_none());
        assert!(!map.path_to_ino.contains_key("/victim"));
    }

    #[test]
    fn ino_for_path_round_trip() {
        let mut map = InodeMap::default();
        let ino = map.intern("/a/b");
        assert_eq!(map.ino_for_path("/a/b"), Some(ino));
        assert_eq!(map.ino_for_path("/missing"), None);
    }

    #[test]
    fn join_path_root_parent() {
        assert_eq!(join_path("/", "child"), "/child");
    }

    #[test]
    fn join_path_nested_parent() {
        assert_eq!(join_path("/a/b", "c"), "/a/b/c");
    }
}
