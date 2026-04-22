//! Storage backend for the V2 cover bytes.
//!
//! V2 originally took the cover by value as `Vec<u8>` — the whole
//! cover was loaded into heap RAM at mount time and written back to
//! disk at unmount. That works for ≲100 MB models but immediately
//! OOMs on real-world weights (the user's prompting case: a 280 GB
//! cover).
//!
//! [`CoverStorage`] abstracts the cover so the V2 [`Filesystem`]
//! can hold either:
//!
//! - [`Vec<u8>`] — convenient for tests; the whole cover lives in
//!   heap. No flush semantics needed.
//! - [`memmap2::MmapMut`] — file-backed mutable map. The OS pages
//!   bytes in/out as needed; RAM usage is bounded by the working set,
//!   not by the cover size. [`flush`](CoverStorage::flush) calls
//!   `msync(2)` to make dirty pages durable.
//!
//! The trait is intentionally tiny — `bytes` / `bytes_mut` / `flush`.
//! Everything else inside V2 already takes `&[u8]` or `&mut [u8]`
//! slices of the cover.
//!
//! [`Filesystem`]: crate::v2::fs::Filesystem

use memmap2::MmapMut;

/// A cover-bytes storage backend usable by [`crate::v2::fs::Filesystem`].
///
/// Implementors must be `Send + Sync + std::fmt::Debug` so the V2
/// filesystem itself can derive these bounds.
pub trait CoverStorage: Send + Sync + std::fmt::Debug {
    /// Borrow the cover as a read-only byte slice.
    fn bytes(&self) -> &[u8];

    /// Borrow the cover as a mutable byte slice. Direct writes to
    /// the slice immediately mutate the underlying storage —
    /// [`flush`](Self::flush) must be called before drop to make
    /// them durable for file-backed storage.
    fn bytes_mut(&mut self) -> &mut [u8];

    /// Make any pending writes durable. For [`Vec<u8>`] this is a
    /// no-op; for [`MmapMut`] it calls `msync(2)`.
    ///
    /// Always call before dropping a file-backed cover, or dirty
    /// pages may be discarded by the OS without ever reaching disk.
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }

    /// Total cover length, in bytes.
    fn len(&self) -> usize {
        self.bytes().len()
    }

    /// Whether the cover has zero length.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl CoverStorage for Vec<u8> {
    fn bytes(&self) -> &[u8] {
        self.as_slice()
    }
    fn bytes_mut(&mut self) -> &mut [u8] {
        self.as_mut_slice()
    }
}

impl CoverStorage for MmapMut {
    fn bytes(&self) -> &[u8] {
        self.as_ref()
    }
    fn bytes_mut(&mut self) -> &mut [u8] {
        self.as_mut()
    }
    fn flush(&mut self) -> std::io::Result<()> {
        // MmapMut::flush calls msync(2) and waits for completion.
        MmapMut::flush(self)
    }
}

/// Lets `Box<dyn CoverStorage>` (the type returned by
/// [`crate::v2::fs::Filesystem::unmount`]) flow straight back into
/// `Filesystem::mount` for unmount→remount round-trips. Forwards
/// every method to the boxed inner. One extra pointer dereference
/// per access; cheap enough that we don't bother specializing.
impl CoverStorage for Box<dyn CoverStorage> {
    fn bytes(&self) -> &[u8] {
        (**self).bytes()
    }
    fn bytes_mut(&mut self) -> &mut [u8] {
        (**self).bytes_mut()
    }
    fn flush(&mut self) -> std::io::Result<()> {
        (**self).flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec_cover_round_trip() {
        let mut cover: Box<dyn CoverStorage> = Box::new(vec![0u8; 32]);
        assert_eq!(cover.len(), 32);
        cover.bytes_mut()[0] = 0xAA;
        assert_eq!(cover.bytes()[0], 0xAA);
        cover.flush().expect("vec flush is a no-op");
    }

    #[test]
    fn mmap_cover_round_trip() {
        use std::io::Write;
        let mut tmp = tempfile::NamedTempFile::new().expect("tempfile");
        tmp.as_file_mut().write_all(&[0u8; 64]).unwrap();
        tmp.as_file_mut().flush().unwrap();
        let mmap = unsafe { MmapMut::map_mut(tmp.as_file()).expect("mmap") };

        let mut cover: Box<dyn CoverStorage> = Box::new(mmap);
        assert_eq!(cover.len(), 64);
        cover.bytes_mut()[7] = 0xBB;
        assert_eq!(cover.bytes()[7], 0xBB);
        cover.flush().expect("mmap flush");

        // Re-read the file: the mutation must be visible.
        let on_disk = std::fs::read(tmp.path()).expect("read file");
        assert_eq!(on_disk[7], 0xBB);
    }
}
