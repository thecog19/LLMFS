//! FUSE driver exposing a [`Filesystem`] as a mountable POSIX tree.
//!
//! # Op model
//!
//! Dispatch is single-threaded by design: [`Filesystem`] methods take
//! `&mut self`, and even read-paths traverse in-memory caches. Threaded
//! dispatch would force a global lock that serialises everything anyway.
//!
//! The driver's only persistent state is the V2 filesystem plus a
//! per-session [`InodeMap`] — the kernel hands us `u64` inodes across
//! ops and we translate them to V2 paths (absolute, starting with `/`).
//! The map is rebuilt from scratch at mount time (same as tmpfs).
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
//! buffer. Richer metadata is an orthogonal V2 follow-up.
//!
//! # Mounting
//!
//! [`mount_foreground`] blocks the calling thread until the kernel
//! releases the mount (e.g. `fusermount3 -u`). [`spawn_background`]
//! returns a [`BackgroundSession`] guard that unmounts on drop —
//! useful for tests and for CLI drivers that want to manage the mount
//! lifetime explicitly.

use std::collections::HashMap;
use std::ffi::OsStr;
use std::path::Path;
use std::sync::{Arc, Mutex};
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
pub type SharedFilesystem = Arc<Mutex<Filesystem>>;

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

/// Block size used for `stat.st_blksize` / `statvfs.f_bsize`. Doesn't
/// have to match the underlying storage — just has to be a sensible
/// unit for userspace tools.
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
    inodes: InodeMap,
    /// Per-inode pending write buffer. Offset writes stage here;
    /// `release` / `flush` / `fsync` commits via
    /// [`Filesystem::create_file`].
    buffers: HashMap<u64, WriteBuffer>,
    /// Session start — used as a fabricated timestamp for every
    /// attribute reply until V2 starts tracking real mtime/ctime.
    mount_time: SystemTime,
}

impl LlmdbV2Fs {
    /// Wrap a V2 [`Filesystem`] for FUSE dispatch. The filesystem is
    /// moved behind an `Arc<Mutex<_>>` so the adapter can co-exist
    /// with an out-of-band CLI handle — see [`Self::share`].
    pub fn new(fs: Filesystem) -> Self {
        Self::with_shared(Arc::new(Mutex::new(fs)))
    }

    /// Build a driver around an already-shared filesystem handle.
    /// Use this when a caller (e.g. the CLI) needs to hold onto a
    /// handle that survives the FUSE session — after `drop(session)`
    /// returns, the caller's `Arc` is the last reference and can
    /// [`Arc::try_unwrap`] the filesystem back.
    pub fn with_shared(fs: SharedFilesystem) -> Self {
        Self {
            fs,
            inodes: InodeMap::default(),
            buffers: HashMap::new(),
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
        let mutex = Arc::try_unwrap(self.fs)
            .expect("into_inner called while another Arc<Mutex<Filesystem>> is alive");
        mutex.into_inner().expect("filesystem mutex poisoned")
    }

    fn path_for_child(&self, parent: u64, name: &OsStr) -> Option<String> {
        let parent_path = self.inodes.path_of(parent)?.to_owned();
        let name = name.to_str()?;
        Some(join_path(&parent_path, name))
    }

    fn file_attr(&self, req: &Request<'_>, ino: u64, path: &str) -> Option<FileAttr> {
        let inode = self.fs.lock().unwrap().inode_at(path).ok()?;
        let size = self
            .buffers
            .get(&ino)
            .filter(|b| b.dirty)
            .map(|b| b.contents.len() as u64)
            .unwrap_or(inode.length);
        let mode = self
            .buffers
            .get(&ino)
            .map(|b| b.mode)
            .unwrap_or(DEFAULT_FILE_MODE);
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
            uid: req.uid(),
            gid: req.gid(),
            rdev: 0,
            blksize: BLOCK_SIZE,
            flags: 0,
        })
    }

    fn dir_attr(&self, req: &Request<'_>, ino: u64) -> FileAttr {
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
            uid: req.uid(),
            gid: req.gid(),
            rdev: 0,
            blksize: BLOCK_SIZE,
            flags: 0,
        }
    }

    /// Look the current on-disk bytes for `path` — merged with a
    /// dirty write buffer if one exists (otherwise readers see stale
    /// on-disk bytes for a just-written file).
    fn read_contents(&self, ino: u64, path: &str) -> Result<Vec<u8>, FsError> {
        if let Some(buf) = self.buffers.get(&ino) {
            return Ok(buf.contents.clone());
        }
        self.fs.lock().unwrap().read_file(path)
    }

    fn flush_buffer(&mut self, ino: u64) -> Result<(), FsError> {
        let Some(buf) = self.buffers.get(&ino) else {
            return Ok(());
        };
        if !buf.dirty {
            return Ok(());
        }
        let path = buf.path.clone();
        let contents = buf.contents.clone();
        self.fs.lock().unwrap().create_file(&path, &contents)?;
        if let Some(buf) = self.buffers.get_mut(&ino) {
            buf.dirty = false;
        }
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
// fuser::Filesystem impl
// ==================================================================

impl FuserFilesystem for LlmdbV2Fs {
    fn lookup(&mut self, req: &Request<'_>, parent: u64, name: &OsStr, reply: ReplyEntry) {
        let Some(path) = self.path_for_child(parent, name) else {
            reply.error(libc::ENOENT);
            return;
        };
        let Ok(entries) = self.fs.lock().unwrap().readdir(
            self.inodes
                .path_of(parent)
                .unwrap_or("/")
                .to_owned()
                .as_str(),
        ) else {
            reply.error(libc::ENOENT);
            return;
        };
        let Some(name_str) = name.to_str() else {
            reply.error(libc::EINVAL);
            return;
        };
        let Some(entry) = entries.iter().find(|e| e.name == name_str) else {
            reply.error(libc::ENOENT);
            return;
        };
        let ino = self.inodes.intern(&path);
        let attr = match entry.kind {
            EntryKind::Directory => self.dir_attr(req, ino),
            EntryKind::File => {
                let Some(attr) = self.file_attr(req, ino, &path) else {
                    reply.error(libc::EIO);
                    return;
                };
                attr
            }
        };
        reply.entry(&TTL, &attr, GENERATION);
    }

    fn forget(&mut self, _req: &Request<'_>, ino: u64, _nlookup: u64) {
        self.inodes.forget(ino);
    }

    fn getattr(&mut self, req: &Request<'_>, ino: u64, _fh: Option<u64>, reply: ReplyAttr) {
        if ino == ROOT_INO {
            let attr = self.dir_attr(req, ino);
            reply.attr(&TTL, &attr);
            return;
        }
        let Some(path) = self.inodes.path_of(ino).map(str::to_owned) else {
            reply.error(libc::ENOENT);
            return;
        };
        if !self.fs.lock().unwrap().exists(&path) {
            reply.error(libc::ENOENT);
            return;
        }
        // Ask V2 about the inode to discriminate file vs. directory
        // without re-scanning the parent.
        let Ok(inode) = self.fs.lock().unwrap().inode_at(&path) else {
            reply.error(libc::ENOENT);
            return;
        };
        // Directories are the ones whose serialized content
        // deserializes to a Directory. Rather than re-parse, we
        // heuristic: try readdir; if it succeeds, it's a directory.
        if self.fs.lock().unwrap().readdir(&path).is_ok() {
            let attr = self.dir_attr(req, ino);
            reply.attr(&TTL, &attr);
            return;
        }
        let size = self
            .buffers
            .get(&ino)
            .filter(|b| b.dirty)
            .map(|b| b.contents.len() as u64)
            .unwrap_or(inode.length);
        let mode = self
            .buffers
            .get(&ino)
            .map(|b| b.mode)
            .unwrap_or(DEFAULT_FILE_MODE);
        let attr = FileAttr {
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
            uid: req.uid(),
            gid: req.gid(),
            rdev: 0,
            blksize: BLOCK_SIZE,
            flags: 0,
        };
        reply.attr(&TTL, &attr);
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
        mut reply: ReplyDirectory,
    ) {
        let Some(dir_path) = self.inodes.path_of(ino).map(str::to_owned) else {
            reply.error(libc::ENOENT);
            return;
        };
        let entries = match self.fs.lock().unwrap().readdir(&dir_path) {
            Ok(e) => e,
            Err(err) => {
                reply.error(fs_err_to_errno(&err));
                return;
            }
        };

        let dot: [(&str, FileType); 2] =
            [(".", FileType::Directory), ("..", FileType::Directory)];

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
            let child_ino = self.inodes.intern(&child_path);
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

    fn open(&mut self, _req: &Request<'_>, ino: u64, _flags: i32, reply: ReplyOpen) {
        if self.inodes.path_of(ino).is_none() {
            reply.error(libc::ENOENT);
            return;
        }
        reply.opened(0, 0);
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
        let Some(path) = self.inodes.path_of(ino).map(str::to_owned) else {
            reply.error(libc::ENOENT);
            return;
        };
        let contents = match self.read_contents(ino, &path) {
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
        let Some(path) = self.inodes.path_of(ino).map(str::to_owned) else {
            reply.error(libc::ENOENT);
            return;
        };
        let start = offset.max(0) as usize;

        // Seed the buffer from on-disk contents on first write.
        if !self.buffers.contains_key(&ino) {
            let existing = self.fs.lock().unwrap().read_file(&path).unwrap_or_default();
            self.buffers.insert(
                ino,
                WriteBuffer {
                    path: path.clone(),
                    mode: DEFAULT_FILE_MODE,
                    contents: existing,
                    dirty: false,
                },
            );
        }

        let buf = self.buffers.get_mut(&ino).unwrap();
        let end = start + data.len();
        if buf.contents.len() < end {
            buf.contents.resize(end, 0);
        }
        buf.contents[start..end].copy_from_slice(data);
        buf.dirty = true;
        reply.written(data.len() as u32);
    }

    fn flush(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        _fh: u64,
        _lock_owner: u64,
        reply: ReplyEmpty,
    ) {
        match self.flush_buffer(ino) {
            Ok(()) => reply.ok(),
            Err(err) => reply.error(fs_err_to_errno(&err)),
        }
    }

    fn fsync(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        _fh: u64,
        _datasync: bool,
        reply: ReplyEmpty,
    ) {
        match self.flush_buffer(ino) {
            Ok(()) => reply.ok(),
            Err(err) => reply.error(fs_err_to_errno(&err)),
        }
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
        if let Err(err) = self.flush_buffer(ino) {
            reply.error(fs_err_to_errno(&err));
            return;
        }
        self.buffers.remove(&ino);
        reply.ok();
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
        let Some(path) = self.path_for_child(parent, name) else {
            reply.error(libc::EINVAL);
            return;
        };
        if self.fs.lock().unwrap().exists(&path) {
            reply.error(libc::EEXIST);
            return;
        }
        // Create an empty file on disk so lookup / getattr see it
        // immediately — otherwise a `touch foo` followed by `stat
        // foo` without a write in between would ENOENT.
        if let Err(err) = self.fs.lock().unwrap().create_file(&path, &[]) {
            reply.error(fs_err_to_errno(&err));
            return;
        }
        let ino = self.inodes.intern(&path);
        let buf_mode = (mode & 0o7777) as u16;
        self.buffers.insert(
            ino,
            WriteBuffer {
                path: path.clone(),
                mode: buf_mode,
                contents: Vec::new(),
                dirty: false,
            },
        );
        let attr = FileAttr {
            ino,
            size: 0,
            blocks: 0,
            atime: self.mount_time,
            mtime: self.mount_time,
            ctime: self.mount_time,
            crtime: self.mount_time,
            kind: FileType::RegularFile,
            perm: buf_mode,
            nlink: 1,
            uid: req.uid(),
            gid: req.gid(),
            rdev: 0,
            blksize: BLOCK_SIZE,
            flags: 0,
        };
        reply.created(&TTL, &attr, GENERATION, 0, 0);
    }

    fn unlink(&mut self, _req: &Request<'_>, parent: u64, name: &OsStr, reply: ReplyEmpty) {
        let Some(path) = self.path_for_child(parent, name) else {
            reply.error(libc::ENOENT);
            return;
        };
        if let Some(ino) = self
            .inodes
            .path_to_ino
            .get(&path)
            .copied()
        {
            self.buffers.remove(&ino);
            self.inodes.forget(ino);
        }
        match self.fs.lock().unwrap().unlink(&path) {
            Ok(()) => reply.ok(),
            Err(err) => reply.error(fs_err_to_errno(&err)),
        }
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
        let Some(path) = self.path_for_child(parent, name) else {
            reply.error(libc::EINVAL);
            return;
        };
        if let Err(err) = self.fs.lock().unwrap().mkdir(&path) {
            reply.error(fs_err_to_errno(&err));
            return;
        }
        let ino = self.inodes.intern(&path);
        let attr = self.dir_attr(req, ino);
        reply.entry(&TTL, &attr, GENERATION);
    }

    fn rmdir(&mut self, _req: &Request<'_>, parent: u64, name: &OsStr, reply: ReplyEmpty) {
        let Some(path) = self.path_for_child(parent, name) else {
            reply.error(libc::ENOENT);
            return;
        };
        match self.fs.lock().unwrap().rmdir(&path) {
            Ok(()) => {
                if let Some(ino) = self.inodes.path_to_ino.get(&path).copied() {
                    self.inodes.forget(ino);
                }
                reply.ok();
            }
            Err(err) => reply.error(fs_err_to_errno(&err)),
        }
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
        let Some(path) = self.inodes.path_of(ino).map(str::to_owned) else {
            reply.error(libc::ENOENT);
            return;
        };

        if size.is_some() || mode.is_some() {
            // Both truncate and chmod stage through the write buffer.
            // Seed the buffer with disk contents if we don't have one.
            if !self.buffers.contains_key(&ino) {
                let existing = self.fs.lock().unwrap().read_file(&path).unwrap_or_default();
                self.buffers.insert(
                    ino,
                    WriteBuffer {
                        path: path.clone(),
                        mode: DEFAULT_FILE_MODE,
                        contents: existing,
                        dirty: false,
                    },
                );
            }
            if let Some(new_size) = size
                && let Some(buf) = self.buffers.get_mut(&ino)
            {
                buf.contents.resize(new_size as usize, 0);
                buf.dirty = true;
            }
            if let Some(new_mode) = mode
                && let Some(buf) = self.buffers.get_mut(&ino)
            {
                buf.mode = (new_mode & 0o7777) as u16;
                buf.dirty = true;
            }
        }

        if let Some(attr) = self.file_attr(req, ino, &path) {
            reply.attr(&TTL, &attr);
        } else if self.fs.lock().unwrap().readdir(&path).is_ok() {
            reply.attr(&TTL, &self.dir_attr(req, ino));
        } else {
            reply.error(libc::ENOENT);
        }
    }

    fn statfs(&mut self, _req: &Request<'_>, _ino: u64, reply: ReplyStatfs) {
        // V2 doesn't expose a total-block count; report free space
        // in stealable-bit weights scaled to blocks so userspace
        // tooling shows something reasonable.
        let free_weights = self.fs.lock().unwrap().allocator_free_weights();
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
    }

    fn access(&mut self, _req: &Request<'_>, _ino: u64, _mask: i32, reply: ReplyEmpty) {
        reply.ok();
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
    fn join_path_root_parent() {
        assert_eq!(join_path("/", "child"), "/child");
    }

    #[test]
    fn join_path_nested_parent() {
        assert_eq!(join_path("/a/b", "c"), "/a/b/c");
    }
}
