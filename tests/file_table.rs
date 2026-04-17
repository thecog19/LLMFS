use llmdb::fs::file_table::{
    ENTRIES_PER_BLOCK, ENTRY_BYTES, FLAG_DELETED, FileEntry, FileEntryType, FileTableBlock,
    FileTableError, MAX_FILENAME_BYTES, MAX_INLINE_BLOCKS, OVERFLOW_ENTRIES_PER_BLOCK,
    OverflowBlock,
};
use llmdb::stego::integrity::NO_BLOCK;

#[test]
fn free_entry_encodes_to_256_zero_bytes() {
    let entry = FileEntry::free();
    let bytes = entry.encode().expect("encode free");
    assert_eq!(bytes, [0_u8; ENTRY_BYTES]);

    let decoded = FileEntry::decode(&bytes).expect("decode zeros");
    assert_eq!(decoded, entry);
    assert!(decoded.is_free());
}

#[test]
fn regular_entry_roundtrips_small_inline_map() {
    let entry = FileEntry {
        entry_type: FileEntryType::Regular,
        flags: 0,
        mode: 0o644,
        uid: 1000,
        gid: 1000,
        size_bytes: 12_345,
        created: 1_700_000_000,
        modified: 1_700_000_001,
        crc32: 0xDEAD_BEEF,
        block_count: 3,
        overflow_block: NO_BLOCK,
        filename: "hello.txt".to_owned(),
        inline_blocks: vec![5, 6, 7],
    };
    let bytes = entry.encode().expect("encode");
    let decoded = FileEntry::decode(&bytes).expect("decode");
    assert_eq!(decoded, entry);
    assert!(decoded.is_live());
    assert!(!decoded.is_deleted());
}

#[test]
fn entry_with_28_inline_blocks_roundtrips_without_overflow() {
    let inline: Vec<u32> = (100..100 + MAX_INLINE_BLOCKS as u32).collect();
    let entry = FileEntry {
        entry_type: FileEntryType::Regular,
        flags: 0,
        mode: 0o600,
        uid: 0,
        gid: 0,
        size_bytes: (MAX_INLINE_BLOCKS * 4096) as u64,
        created: 0,
        modified: 0,
        crc32: 0,
        block_count: MAX_INLINE_BLOCKS as u32,
        overflow_block: NO_BLOCK,
        filename: "full_inline.bin".to_owned(),
        inline_blocks: inline,
    };
    let bytes = entry.encode().expect("encode full inline");
    let decoded = FileEntry::decode(&bytes).expect("decode full inline");
    assert_eq!(decoded, entry);
}

#[test]
fn entry_with_overflow_roundtrips_and_truncates_inline_at_28() {
    let inline: Vec<u32> = (200..228).collect();
    let entry = FileEntry {
        entry_type: FileEntryType::Regular,
        flags: 0,
        mode: 0o644,
        uid: 0,
        gid: 0,
        size_bytes: 500_000,
        created: 0,
        modified: 0,
        crc32: 42,
        block_count: 123,
        overflow_block: 900,
        filename: "big.bin".to_owned(),
        inline_blocks: inline,
    };
    let bytes = entry.encode().expect("encode overflow");
    let decoded = FileEntry::decode(&bytes).expect("decode overflow");
    assert_eq!(decoded.block_count, 123);
    assert_eq!(decoded.overflow_block, 900);
    assert_eq!(decoded.inline_blocks.len(), MAX_INLINE_BLOCKS);
    assert_eq!(decoded.inline_blocks[0], 200);
    assert_eq!(decoded.inline_blocks[27], 227);
}

#[test]
fn tombstoned_entry_is_not_live_but_not_free() {
    let entry = FileEntry {
        entry_type: FileEntryType::Regular,
        flags: FLAG_DELETED,
        mode: 0o644,
        uid: 0,
        gid: 0,
        size_bytes: 1024,
        created: 0,
        modified: 0,
        crc32: 0,
        block_count: 0,
        overflow_block: NO_BLOCK,
        filename: "gone.txt".to_owned(),
        inline_blocks: vec![],
    };
    let bytes = entry.encode().expect("encode");
    let decoded = FileEntry::decode(&bytes).expect("decode");
    assert!(decoded.is_deleted());
    assert!(!decoded.is_live());
    assert!(!decoded.is_free());
    assert_eq!(decoded.filename, "gone.txt");
}

#[test]
fn file_table_block_roundtrips_16_entries() {
    let mut entries: Vec<FileEntry> = (0..ENTRIES_PER_BLOCK).map(|_| FileEntry::free()).collect();
    entries[2] = FileEntry {
        entry_type: FileEntryType::Regular,
        flags: 0,
        mode: 0o644,
        uid: 0,
        gid: 0,
        size_bytes: 8,
        created: 0,
        modified: 0,
        crc32: 1,
        block_count: 1,
        overflow_block: NO_BLOCK,
        filename: "slot2.bin".to_owned(),
        inline_blocks: vec![42],
    };
    let block = FileTableBlock {
        entries: entries.clone(),
    };
    let bytes = block.encode().expect("encode block");
    assert_eq!(bytes.len(), llmdb::BLOCK_SIZE);

    let decoded = FileTableBlock::decode(&bytes).expect("decode block");
    assert_eq!(decoded.entries.len(), ENTRIES_PER_BLOCK);
    assert_eq!(decoded.entries[2], entries[2]);
    for i in 0..ENTRIES_PER_BLOCK {
        if i != 2 {
            assert!(decoded.entries[i].is_free(), "slot {} should be free", i);
        }
    }
}

#[test]
fn empty_file_table_block_encodes_to_zeros() {
    let block = FileTableBlock::empty();
    let bytes = block.encode().expect("encode empty");
    assert!(bytes.iter().all(|&b| b == 0));
}

#[test]
fn encode_rejects_filename_longer_than_95_bytes() {
    let entry = FileEntry {
        entry_type: FileEntryType::Regular,
        flags: 0,
        mode: 0o644,
        uid: 0,
        gid: 0,
        size_bytes: 0,
        created: 0,
        modified: 0,
        crc32: 0,
        block_count: 0,
        overflow_block: NO_BLOCK,
        filename: "x".repeat(MAX_FILENAME_BYTES + 1),
        inline_blocks: vec![],
    };
    let result = entry.encode();
    assert!(matches!(
        result,
        Err(FileTableError::FilenameTooLong { .. })
    ));
}

#[test]
fn encode_rejects_more_than_28_inline_blocks() {
    let entry = FileEntry {
        entry_type: FileEntryType::Regular,
        flags: 0,
        mode: 0o644,
        uid: 0,
        gid: 0,
        size_bytes: 0,
        created: 0,
        modified: 0,
        crc32: 0,
        block_count: 29,
        overflow_block: 100,
        filename: "a".into(),
        inline_blocks: (0..29).collect(),
    };
    let result = entry.encode();
    assert!(matches!(
        result,
        Err(FileTableError::TooManyInlineBlocks { .. })
    ));
}

#[test]
fn encode_rejects_missing_overflow_when_block_count_exceeds_inline_capacity() {
    let entry = FileEntry {
        entry_type: FileEntryType::Regular,
        flags: 0,
        mode: 0o644,
        uid: 0,
        gid: 0,
        size_bytes: 0,
        created: 0,
        modified: 0,
        crc32: 0,
        block_count: 30,
        overflow_block: NO_BLOCK,
        filename: "a".into(),
        inline_blocks: (0..MAX_INLINE_BLOCKS as u32).collect(),
    };
    let result = entry.encode();
    assert!(matches!(
        result,
        Err(FileTableError::MissingOverflowBlock { .. })
    ));
}

#[test]
fn decode_rejects_invalid_entry_type() {
    let mut bytes = [0_u8; ENTRY_BYTES];
    bytes[0] = 99;
    let result = FileEntry::decode(&bytes);
    assert!(matches!(result, Err(FileTableError::InvalidEntryType(99))));
}

#[test]
fn decode_rejects_non_null_terminated_filename() {
    let mut bytes = [0_u8; ENTRY_BYTES];
    bytes[0] = 1; // regular
    bytes[0x2C..0x30].copy_from_slice(&NO_BLOCK.to_le_bytes()); // overflow_block
    for i in 0x30..0x90 {
        bytes[i] = b'x'; // fill filename field with no null
    }
    let result = FileEntry::decode(&bytes);
    assert!(matches!(
        result,
        Err(FileTableError::FilenameNotNullTerminated)
    ));
}

#[test]
fn overflow_block_roundtrips_full_capacity() {
    let entries: Vec<u32> = (0..OVERFLOW_ENTRIES_PER_BLOCK as u32).collect();
    let block = OverflowBlock::new(NO_BLOCK, entries.clone()).expect("new");
    let bytes = block.encode().expect("encode");
    assert_eq!(bytes.len(), llmdb::BLOCK_SIZE);

    let decoded =
        OverflowBlock::decode(&bytes, OVERFLOW_ENTRIES_PER_BLOCK).expect("decode");
    assert_eq!(decoded.next, NO_BLOCK);
    assert_eq!(decoded.entries, entries);
}

#[test]
fn overflow_block_partial_entries_roundtrip_with_live_count() {
    let block = OverflowBlock::new(777, vec![1, 2, 3, 4]).expect("new");
    let bytes = block.encode().expect("encode");
    let decoded = OverflowBlock::decode(&bytes, 4).expect("decode");
    assert_eq!(decoded.next, 777);
    assert_eq!(decoded.entries, vec![1, 2, 3, 4]);
}

#[test]
fn overflow_block_rejects_too_many_entries() {
    let entries: Vec<u32> = (0..=OVERFLOW_ENTRIES_PER_BLOCK as u32).collect();
    let result = OverflowBlock::new(NO_BLOCK, entries);
    assert!(matches!(
        result,
        Err(FileTableError::TooManyOverflowEntries { .. })
    ));
}
