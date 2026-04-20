//! FUSE `Filesystem` implementation routing kernel requests to the
//! underlying `StegoDevice` file table.
//!
//! Dispatch is single-threaded by design. `StegoDevice` methods take
//! `&mut self`; enabling fuser's threaded dispatch would require a
//! global lock that serializes everything anyway.

use std::collections::{HashMap, HashSet};
use std::ffi::OsStr;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use fuser::{
    FileAttr, FileType, Filesystem, ReplyAttr, ReplyCreate, ReplyData, ReplyDirectory, ReplyEmpty,
    ReplyEntry, ReplyOpen, ReplyStatfs, ReplyWrite, Request, TimeOrNow,
};

use crate::fs::file_ops::FsError;
use crate::fs::file_table::FileEntry;
use crate::fuse::dir::{Child, ChildKind, DirView};
use crate::fuse::inode::{InodeMap, ROOT_INO};
use crate::stego::device::StegoDevice;

const TTL: Duration = Duration::from_secs(1);
const GENERATION: u64 = 0;

pub struct LlmdbFs {
    device: StegoDevice,
    inodes: InodeMap,
    /// Per-inode pending write buffer. Writes stage here; `release` /
    /// `flush` / `fsync` commits via delete-then-store.
    buffers: HashMap<u64, WriteBuffer>,
    /// Virtual directories that were `mkdir`'d with no file stored under
    /// them yet. Tracked in-memory; lost on unmount.
    explicit_dirs: HashSet<String>,
}

#[derive(Debug)]
struct WriteBuffer {
    path: String,
    mode: u16,
    contents: Vec<u8>,
    dirty: bool,
}

impl LlmdbFs {
    pub fn new(device: StegoDevice) -> Self {
        Self {
            device,
            inodes: InodeMap::default(),
            buffers: HashMap::new(),
            explicit_dirs: HashSet::new(),
        }
    }

    /// Consume the driver and return the underlying device. Used by tests
    /// and by the mount CLI for graceful shutdown.
    #[allow(dead_code)]
    pub fn into_device(self) -> StegoDevice {
        self.device
    }

    fn snapshot_entries(&self) -> Vec<FileEntry> {
        self.device.list_files().unwrap_or_default()
    }

    fn path_for_child(&self, parent: u64, name: &OsStr) -> Option<String> {
        let parent_path = self.inodes.path_of(parent)?.to_owned();
        let name = name.to_str()?;
        if parent_path.is_empty() {
            Some(name.to_owned())
        } else {
            Some(format!("{parent_path}/{name}"))
        }
    }

    fn file_attr_for(&self, req: &Request<'_>, ino: u64, entry: &FileEntry) -> FileAttr {
        let mtime = epoch_seconds(entry.modified);
        let ctime = mtime;
        let crtime = epoch_seconds(entry.created);
        FileAttr {
            ino,
            size: entry.size_bytes,
            blocks: entry.block_count as u64,
            atime: mtime,
            mtime,
            ctime,
            crtime,
            kind: FileType::RegularFile,
            perm: entry.mode & 0o7777,
            nlink: 1,
            uid: req.uid(),
            gid: req.gid(),
            rdev: 0,
            blksize: crate::BLOCK_SIZE as u32,
            flags: 0,
        }
    }

    fn dir_attr_for(&self, req: &Request<'_>, ino: u64) -> FileAttr {
        let now = SystemTime::now();
        FileAttr {
            ino,
            size: 0,
            blocks: 0,
            atime: now,
            mtime: now,
            ctime: now,
            crtime: now,
            kind: FileType::Directory,
            perm: 0o755,
            nlink: 2,
            uid: req.uid(),
            gid: req.gid(),
            rdev: 0,
            blksize: crate::BLOCK_SIZE as u32,
            flags: 0,
        }
    }

    fn load_contents(&self, path: &str) -> Option<Vec<u8>> {
        self.device.read_file_bytes(path).ok()
    }

    fn find_entry(&self, path: &str) -> Option<FileEntry> {
        self.device
            .list_files()
            .ok()?
            .into_iter()
            .find(|e| e.filename == path)
    }

    fn flush_buffer(&mut self, ino: u64) -> Result<(), FsError> {
        let Some(buf) = self.buffers.get(&ino) else {
            return Ok(());
        };
        if !buf.dirty {
            return Ok(());
        }
        let path = buf.path.clone();
        let mode = buf.mode;
        let contents = buf.contents.clone();
        // Commit sequence: delete-if-exists, then store. The current CLI
        // has the same delete-then-store semantics, so a crash between
        // the two leaves the file gone. Matches V1 commit behavior;
        // atomic in-place overwrite is V2.
        if self.find_entry(&path).is_some() {
            self.device.delete_file(&path)?;
        }
        self.device.store_bytes(&contents, &path, mode)?;
        if let Some(buf) = self.buffers.get_mut(&ino) {
            buf.dirty = false;
        }
        Ok(())
    }
}

fn epoch_seconds(secs: u64) -> SystemTime {
    UNIX_EPOCH + Duration::from_secs(secs)
}

fn fs_err_to_errno(err: &FsError) -> i32 {
    match err {
        FsError::FileNotFound(_) => libc::ENOENT,
        FsError::DuplicateName(_) => libc::EEXIST,
        FsError::InvalidFilename { .. } => libc::EINVAL,
        FsError::FileTooLarge { .. } => libc::ENOSPC,
        FsError::Crc32Mismatch { .. } => libc::EIO,
        FsError::TableFull { .. } => libc::ENOSPC,
        FsError::Io(_) | FsError::Device(_) | FsError::Table(_) => libc::EIO,
    }
}

impl Filesystem for LlmdbFs {
    fn lookup(&mut self, req: &Request<'_>, parent: u64, name: &OsStr, reply: ReplyEntry) {
        let Some(path) = self.path_for_child(parent, name) else {
            reply.error(libc::ENOENT);
            return;
        };
        let entries = self.snapshot_entries();
        if let Some(entry) = entries.iter().find(|e| e.filename == path).cloned() {
            let ino = self.inodes.intern(&path);
            let attr = self.file_attr_for(req, ino, &entry);
            reply.entry(&TTL, &attr, GENERATION);
            return;
        }
        if DirView::path_is_directory(&path, &entries, &self.explicit_dirs) {
            let ino = self.inodes.intern(&path);
            let attr = self.dir_attr_for(req, ino);
            reply.entry(&TTL, &attr, GENERATION);
            return;
        }
        reply.error(libc::ENOENT);
    }

    fn getattr(&mut self, req: &Request<'_>, ino: u64, _fh: Option<u64>, reply: ReplyAttr) {
        let Some(path) = self.inodes.path_of(ino).map(str::to_owned) else {
            reply.error(libc::ENOENT);
            return;
        };
        if ino == ROOT_INO {
            let attr = self.dir_attr_for(req, ino);
            reply.attr(&TTL, &attr);
            return;
        }
        let entries = self.snapshot_entries();
        if let Some(entry) = entries.iter().find(|e| e.filename == path).cloned() {
            // If there's a dirty write buffer, report its size rather
            // than the stale on-disk size.
            let mut attr = self.file_attr_for(req, ino, &entry);
            if let Some(buf) = self.buffers.get(&ino)
                && buf.dirty
            {
                attr.size = buf.contents.len() as u64;
                attr.blocks = attr.size.div_ceil(crate::BLOCK_SIZE as u64);
            }
            reply.attr(&TTL, &attr);
            return;
        }
        if DirView::path_is_directory(&path, &entries, &self.explicit_dirs) {
            let attr = self.dir_attr_for(req, ino);
            reply.attr(&TTL, &attr);
            return;
        }
        reply.error(libc::ENOENT);
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
        let entries = self.snapshot_entries();
        if !DirView::path_is_directory(&dir_path, &entries, &self.explicit_dirs) {
            reply.error(libc::ENOTDIR);
            return;
        }

        let dot_entries: [(&str, FileType); 2] =
            [(".", FileType::Directory), ("..", FileType::Directory)];
        let view = DirView::new(&dir_path, &entries, &self.explicit_dirs);
        let children: Vec<Child> = view.children();

        let mut next_offset = 1_i64;
        let mut idx = 0_i64;
        for (name, kind) in dot_entries.iter() {
            if idx >= offset && reply.add(ino, next_offset, *kind, *name) {
                reply.ok();
                return;
            }
            idx += 1;
            next_offset += 1;
        }
        for child in &children {
            if idx < offset {
                idx += 1;
                next_offset += 1;
                continue;
            }
            let (kind, child_path) = match &child.kind {
                ChildKind::File(_) => (FileType::RegularFile, join_path(&dir_path, &child.name)),
                ChildKind::Directory => (FileType::Directory, join_path(&dir_path, &child.name)),
            };
            let child_ino = self.inodes.intern(&child_path);
            if reply.add(child_ino, next_offset, kind, &child.name) {
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
        // Prefer the dirty buffer if present — otherwise the caller would
        // see stale on-disk bytes for a file they just wrote.
        let contents = if let Some(buf) = self.buffers.get(&ino) {
            buf.contents.clone()
        } else {
            match self.load_contents(&path) {
                Some(c) => c,
                None => {
                    reply.error(libc::ENOENT);
                    return;
                }
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
            let (existing, mode) = match self.find_entry(&path) {
                Some(entry) => (
                    self.load_contents(&path).unwrap_or_default(),
                    entry.mode & 0o7777,
                ),
                None => (Vec::new(), 0o644),
            };
            self.buffers.insert(
                ino,
                WriteBuffer {
                    path: path.clone(),
                    mode,
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
        if let Err(err) = self.flush_buffer(ino) {
            reply.error(fs_err_to_errno(&err));
            return;
        }
        if let Err(err) = self.device.flush() {
            reply.error(fs_err_to_errno(&FsError::Device(err)));
            return;
        }
        reply.ok();
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
        if self.find_entry(&path).is_some() {
            reply.error(libc::EEXIST);
            return;
        }
        let ino = self.inodes.intern(&path);
        // Seed a zero-byte buffer marked dirty so `release` commits an
        // empty file even if no writes happen between create and close.
        self.buffers.insert(
            ino,
            WriteBuffer {
                path: path.clone(),
                mode: (mode & 0o7777) as u16,
                contents: Vec::new(),
                dirty: true,
            },
        );
        let now = SystemTime::now();
        let attr = FileAttr {
            ino,
            size: 0,
            blocks: 0,
            atime: now,
            mtime: now,
            ctime: now,
            crtime: now,
            kind: FileType::RegularFile,
            perm: (mode & 0o7777) as u16,
            nlink: 1,
            uid: req.uid(),
            gid: req.gid(),
            rdev: 0,
            blksize: crate::BLOCK_SIZE as u32,
            flags: 0,
        };
        reply.created(&TTL, &attr, GENERATION, 0, 0);
    }

    fn unlink(&mut self, _req: &Request<'_>, parent: u64, name: &OsStr, reply: ReplyEmpty) {
        let Some(path) = self.path_for_child(parent, name) else {
            reply.error(libc::ENOENT);
            return;
        };
        if let Some(ino) = self.inodes.ino_of(&path) {
            self.buffers.remove(&ino);
        }
        match self.device.delete_file(&path) {
            Ok(()) => {
                if let Some(ino) = self.inodes.ino_of(&path) {
                    self.inodes.forget(ino);
                }
                reply.ok();
            }
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
        let entries = self.snapshot_entries();
        if DirView::path_is_directory(&path, &entries, &self.explicit_dirs) {
            reply.error(libc::EEXIST);
            return;
        }
        if entries.iter().any(|e| e.filename == path) {
            reply.error(libc::EEXIST);
            return;
        }
        self.explicit_dirs.insert(path.clone());
        let ino = self.inodes.intern(&path);
        let attr = self.dir_attr_for(req, ino);
        reply.entry(&TTL, &attr, GENERATION);
    }

    fn rmdir(&mut self, _req: &Request<'_>, parent: u64, name: &OsStr, reply: ReplyEmpty) {
        let Some(path) = self.path_for_child(parent, name) else {
            reply.error(libc::ENOENT);
            return;
        };
        let entries = self.snapshot_entries();
        if !DirView::path_is_directory(&path, &entries, &self.explicit_dirs) {
            reply.error(libc::ENOENT);
            return;
        }
        if DirView::has_descendants(&path, &entries) {
            reply.error(libc::ENOTEMPTY);
            return;
        }
        if !self.explicit_dirs.remove(&path) {
            // Virtual dir with no live descendants and not explicitly
            // created — remove from inode map silently.
        }
        if let Some(ino) = self.inodes.ino_of(&path) {
            self.inodes.forget(ino);
        }
        reply.ok();
    }

    fn rename(
        &mut self,
        _req: &Request<'_>,
        parent: u64,
        name: &OsStr,
        newparent: u64,
        newname: &OsStr,
        _flags: u32,
        reply: ReplyEmpty,
    ) {
        let Some(old_path) = self.path_for_child(parent, name) else {
            reply.error(libc::ENOENT);
            return;
        };
        let Some(new_path) = self.path_for_child(newparent, newname) else {
            reply.error(libc::EINVAL);
            return;
        };
        if old_path == new_path {
            reply.ok();
            return;
        }
        let entries = self.snapshot_entries();
        let old_is_file = entries.iter().any(|e| e.filename == old_path);
        let old_is_dir =
            !old_is_file && DirView::path_is_directory(&old_path, &entries, &self.explicit_dirs);
        if !old_is_file && !old_is_dir {
            reply.error(libc::ENOENT);
            return;
        }

        // Refuse rename onto a live file under the target name (we'd need
        // to atomic-overwrite, which V1 can't). Directories can clash too.
        let new_is_live_file = entries.iter().any(|e| e.filename == new_path);
        if new_is_live_file {
            reply.error(libc::EEXIST);
            return;
        }

        if old_is_file {
            // Read → store-under-new-name → delete-old.
            let contents = match self.load_contents(&old_path) {
                Some(c) => c,
                None => {
                    reply.error(libc::ENOENT);
                    return;
                }
            };
            let mode = entries
                .iter()
                .find(|e| e.filename == old_path)
                .map(|e| e.mode)
                .unwrap_or(0o644);
            if let Err(err) = self.device.store_bytes(&contents, &new_path, mode) {
                reply.error(fs_err_to_errno(&err));
                return;
            }
            if let Err(err) = self.device.delete_file(&old_path) {
                // Try to roll back the newly-stored copy so we don't
                // leave a duplicate. Best-effort.
                let _ = self.device.delete_file(&new_path);
                reply.error(fs_err_to_errno(&err));
                return;
            }
            if let Some(ino) = self.inodes.ino_of(&old_path) {
                self.inodes.rename(ino, &new_path);
            }
            reply.ok();
            return;
        }

        // Directory rename: walk live descendants, rewrite names.
        let old_prefix = format!("{old_path}/");
        let movers: Vec<FileEntry> = entries
            .iter()
            .filter(|e| e.is_live() && e.filename.starts_with(&old_prefix))
            .cloned()
            .collect();
        for entry in &movers {
            let rel = &entry.filename[old_prefix.len()..];
            let target = format!("{new_path}/{rel}");
            let contents = match self.load_contents(&entry.filename) {
                Some(c) => c,
                None => {
                    reply.error(libc::EIO);
                    return;
                }
            };
            if let Err(err) = self.device.store_bytes(&contents, &target, entry.mode) {
                reply.error(fs_err_to_errno(&err));
                return;
            }
            if let Err(err) = self.device.delete_file(&entry.filename) {
                reply.error(fs_err_to_errno(&err));
                return;
            }
        }
        // Move the explicit-dir marker if present.
        if self.explicit_dirs.remove(&old_path) {
            self.explicit_dirs.insert(new_path.clone());
        }
        if let Some(ino) = self.inodes.ino_of(&old_path) {
            self.inodes.rename(ino, &new_path);
        }
        reply.ok();
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

        if let Some(new_size) = size {
            // Truncate / extend. Seed the buffer from disk if we don't
            // already have one, then resize.
            if !self.buffers.contains_key(&ino) {
                let (existing, file_mode) = match self.find_entry(&path) {
                    Some(entry) => (
                        self.load_contents(&path).unwrap_or_default(),
                        entry.mode & 0o7777,
                    ),
                    None => {
                        reply.error(libc::ENOENT);
                        return;
                    }
                };
                self.buffers.insert(
                    ino,
                    WriteBuffer {
                        path: path.clone(),
                        mode: file_mode,
                        contents: existing,
                        dirty: false,
                    },
                );
            }
            if let Some(buf) = self.buffers.get_mut(&ino) {
                buf.contents.resize(new_size as usize, 0);
                buf.dirty = true;
            }
        }

        if let Some(new_mode) = mode {
            // Mode change is cheap but currently requires a re-store to
            // persist. Buffer the file and mark dirty; release will commit.
            if !self.buffers.contains_key(&ino) {
                let existing = self.load_contents(&path).unwrap_or_default();
                self.buffers.insert(
                    ino,
                    WriteBuffer {
                        path: path.clone(),
                        mode: (new_mode & 0o7777) as u16,
                        contents: existing,
                        dirty: true,
                    },
                );
            } else if let Some(buf) = self.buffers.get_mut(&ino) {
                buf.mode = (new_mode & 0o7777) as u16;
                buf.dirty = true;
            }
        }

        // Return current attrs (post-setattr). If the file doesn't exist
        // on disk AND we have no buffer, that's ENOENT above. Otherwise
        // synthesize from the buffer or the on-disk entry.
        if let Some(buf) = self.buffers.get(&ino) {
            let now = SystemTime::now();
            let attr = FileAttr {
                ino,
                size: buf.contents.len() as u64,
                blocks: (buf.contents.len() as u64).div_ceil(crate::BLOCK_SIZE as u64),
                atime: now,
                mtime: now,
                ctime: now,
                crtime: now,
                kind: FileType::RegularFile,
                perm: buf.mode,
                nlink: 1,
                uid: req.uid(),
                gid: req.gid(),
                rdev: 0,
                blksize: crate::BLOCK_SIZE as u32,
                flags: 0,
            };
            reply.attr(&TTL, &attr);
            return;
        }

        if let Some(entry) = self.find_entry(&path) {
            let attr = self.file_attr_for(req, ino, &entry);
            reply.attr(&TTL, &attr);
            return;
        }

        reply.error(libc::ENOENT);
    }

    fn statfs(&mut self, _req: &Request<'_>, _ino: u64, reply: ReplyStatfs) {
        let total = self.device.total_blocks() as u64;
        let free = self.device.free_blocks().unwrap_or(0) as u64;
        // bsize = block size, namelen = max filename, frsize = fragment size.
        reply.statfs(
            total,
            free,
            free,
            self.snapshot_entries().len() as u64,
            0,
            crate::BLOCK_SIZE as u32,
            crate::fs::file_table::MAX_FILENAME_BYTES as u32,
            crate::BLOCK_SIZE as u32,
        );
    }

    fn access(&mut self, _req: &Request<'_>, _ino: u64, _mask: i32, reply: ReplyEmpty) {
        reply.ok();
    }
}

fn join_path(dir_path: &str, name: &str) -> String {
    if dir_path.is_empty() {
        name.to_owned()
    } else {
        format!("{dir_path}/{name}")
    }
}
