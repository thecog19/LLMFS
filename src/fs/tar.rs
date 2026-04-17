//! Minimal ustar writer for the `llmdb dump` subcommand. Streams one
//! regular-file entry at a time to any `Write`, then `finish()`
//! emits the two-block zero terminator.
//!
//! We pick ustar (POSIX.1-1988) specifically because GNU tar, BSD
//! tar, and `bsdtar` all accept it without prompting about unknown
//! extensions. Filenames are truncated at 100 bytes — none of our
//! stego filenames can exceed that (the file table caps them at
//! MAX_FILENAME_BYTES=64), so the truncation branch is unreachable
//! in practice but still enforced for safety.

use std::io::{self, Write};

pub const BLOCK: usize = 512;
const NAME_LIMIT: usize = 100;

pub struct TarWriter<W: Write> {
    inner: W,
    finished: bool,
}

impl<W: Write> TarWriter<W> {
    pub fn new(inner: W) -> Self {
        Self {
            inner,
            finished: false,
        }
    }

    /// Append one regular-file entry.
    pub fn append(
        &mut self,
        name: &str,
        mode: u16,
        mtime_unix: u64,
        payload: &[u8],
    ) -> io::Result<()> {
        if self.finished {
            return Err(io::Error::other("tar writer already finished"));
        }
        let header = build_header(name, mode, mtime_unix, payload.len() as u64)?;
        self.inner.write_all(&header)?;
        self.inner.write_all(payload)?;
        let padding = (BLOCK - payload.len() % BLOCK) % BLOCK;
        if padding > 0 {
            self.inner.write_all(&vec![0_u8; padding])?;
        }
        Ok(())
    }

    /// Emit the two zero blocks that end a tar archive. Must be
    /// called exactly once before the underlying writer is dropped,
    /// or tooling will flag the archive as truncated.
    pub fn finish(mut self) -> io::Result<W> {
        self.inner.write_all(&[0_u8; BLOCK * 2])?;
        self.finished = true;
        Ok(self.inner)
    }
}

fn build_header(name: &str, mode: u16, mtime_unix: u64, size: u64) -> io::Result<[u8; BLOCK]> {
    let name_bytes = name.as_bytes();
    if name_bytes.len() > NAME_LIMIT {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "tar entry name exceeds 100 bytes ({}): {}",
                name_bytes.len(),
                name
            ),
        ));
    }

    let mut header = [0_u8; BLOCK];
    header[0..name_bytes.len()].copy_from_slice(name_bytes);
    write_octal(&mut header[100..108], mode as u64, 7); // 7 digits + NUL
    write_octal(&mut header[108..116], 0, 7); // uid
    write_octal(&mut header[116..124], 0, 7); // gid
    write_octal(&mut header[124..136], size, 11); // 11 digits + NUL
    write_octal(&mut header[136..148], mtime_unix, 11);
    // Checksum field is 8 bytes. For checksum computation it is
    // treated as 8 spaces, so prime it before summing.
    for b in &mut header[148..156] {
        *b = b' ';
    }
    header[156] = b'0'; // typeflag: regular file
    // linkname 157..257 stays zero
    header[257..263].copy_from_slice(b"ustar\0");
    header[263..265].copy_from_slice(b"00");
    // uname/gname/devmajor/devminor/prefix all zero

    let checksum: u32 = header.iter().map(|b| *b as u32).sum();
    write_octal(&mut header[148..155], checksum as u64, 6); // 6 digits + NUL
    header[155] = b' ';

    Ok(header)
}

fn write_octal(dest: &mut [u8], value: u64, width: usize) {
    let formatted = format!("{value:0>width$o}", width = width);
    let bytes = formatted.as_bytes();
    let take = bytes.len().min(width);
    dest[..take].copy_from_slice(&bytes[bytes.len() - take..]);
    if dest.len() > width {
        dest[width] = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_entry_archive_parses_bytewise() {
        let mut buf = Vec::new();
        let mut w = TarWriter::new(&mut buf);
        w.append("hello.txt", 0o644, 123456, b"hi there").unwrap();
        w.finish().unwrap();

        // Header block.
        assert_eq!(&buf[0..9], b"hello.txt");
        assert_eq!(buf[156], b'0');
        assert_eq!(&buf[257..263], b"ustar\0");
        assert_eq!(&buf[263..265], b"00");

        // Payload + padding = one block.
        assert_eq!(&buf[512..520], b"hi there");
        assert!(buf[520..1024].iter().all(|&b| b == 0));

        // Two-block terminator.
        assert_eq!(buf.len(), 1024 + 1024);
        assert!(buf[1024..].iter().all(|&b| b == 0));
    }

    #[test]
    fn checksum_validates() {
        let mut buf = Vec::new();
        let mut w = TarWriter::new(&mut buf);
        w.append("a", 0o600, 0, b"x").unwrap();
        w.finish().unwrap();

        let header = &buf[0..512];
        // Checksum field is 6 octal digits, then NUL, then space.
        let recorded_str = std::str::from_utf8(&header[148..154]).unwrap();
        let recorded: u32 = u32::from_str_radix(recorded_str, 8).unwrap();
        assert_eq!(header[154], 0);
        assert_eq!(header[155], b' ');

        let mut fake_space = [0_u8; 512];
        fake_space.copy_from_slice(header);
        for b in &mut fake_space[148..156] {
            *b = b' ';
        }
        let computed: u32 = fake_space.iter().map(|b| *b as u32).sum();
        assert_eq!(recorded, computed, "ustar checksum mismatch");
    }

    #[test]
    fn name_longer_than_100_bytes_errors() {
        let mut buf = Vec::new();
        let mut w = TarWriter::new(&mut buf);
        let long_name = "a".repeat(101);
        assert!(w.append(&long_name, 0o644, 0, b"").is_err());
    }
}
