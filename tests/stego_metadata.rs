use llmdb::stego::integrity::{
    ENTRIES_PER_INTEGRITY_BLOCK, IntegrityBlock, IntegrityError, NO_BLOCK, Superblock,
    SuperblockFields,
};

#[test]
fn superblock_roundtrips_with_checksum_validation() {
    let fields = SuperblockFields {
        total_blocks: 100,
        free_list_head: 12,
        integrity_chain_head: 1,
        redirection_table_start: 2,
        redirection_table_length: 1,
        file_table_start: 3,
        file_table_length: 1,
        flags: 0x01,
        quant_profile: 0b0010_0001,
    };

    let encoded = Superblock::new(fields).encode();
    assert_eq!(encoded.len(), llmdb::BLOCK_SIZE);

    let decoded = Superblock::decode(&encoded).expect("decode superblock");

    assert_eq!(decoded.fields.total_blocks, 100);
    assert_eq!(decoded.fields.free_list_head, 12);
    assert_eq!(decoded.fields.integrity_chain_head, 1);
    assert_eq!(decoded.fields.redirection_table_start, 2);
    assert_eq!(decoded.fields.redirection_table_length, 1);
    assert_eq!(decoded.fields.file_table_start, 3);
    assert_eq!(decoded.fields.file_table_length, 1);
    assert_eq!(decoded.fields.flags, 0x01);
    assert_eq!(decoded.fields.quant_profile, 0b0010_0001);
}

#[test]
fn superblock_flags_roundtrip() {
    for flags in [0x00, 0x01, 0x02, 0x03] {
        let fields = SuperblockFields {
            total_blocks: 10,
            free_list_head: NO_BLOCK,
            integrity_chain_head: NO_BLOCK,
            redirection_table_start: NO_BLOCK,
            redirection_table_length: 0,
            file_table_start: NO_BLOCK,
            file_table_length: 0,
            flags,
            quant_profile: 0,
        };

        let encoded = Superblock::new(fields).encode();
        let decoded = Superblock::decode(&encoded).expect("decode flags");
        assert_eq!(decoded.fields.flags, flags, "flags {flags:#04x}");
        assert_eq!(decoded.is_lobotomy(), flags & 0x01 != 0);
        assert_eq!(decoded.is_dirty(), flags & 0x02 != 0);
    }
}

#[test]
fn superblock_rejects_invalid_magic_version_and_checksum() {
    let valid = Superblock::new(SuperblockFields {
        total_blocks: 10,
        free_list_head: NO_BLOCK,
        integrity_chain_head: NO_BLOCK,
        redirection_table_start: NO_BLOCK,
        redirection_table_length: 0,
        file_table_start: NO_BLOCK,
        file_table_length: 0,
        flags: 0,
        quant_profile: 0,
    })
    .encode();

    // Invalid magic
    let mut bad_magic = valid.clone();
    bad_magic[0] = 0xFF;
    assert!(matches!(
        Superblock::decode(&bad_magic),
        Err(IntegrityError::InvalidMagic { .. })
    ));

    // Invalid version (version 7 at offset 0x05)
    let mut bad_version = valid.clone();
    bad_version[0x05] = 7;
    // Must re-compute CRC to avoid a checksum error masking the version error.
    // Since CRC is checked after version, tamper the version only.
    assert!(matches!(
        Superblock::decode(&bad_version),
        Err(IntegrityError::UnsupportedVersion(7))
    ));

    // Invalid checksum
    let mut bad_crc = valid.clone();
    bad_crc[0x28] ^= 0xFF;
    assert!(matches!(
        Superblock::decode(&bad_crc),
        Err(IntegrityError::ChecksumMismatch { .. })
    ));

    // Wrong block length
    assert!(matches!(
        Superblock::decode(&[0_u8; 100]),
        Err(IntegrityError::InvalidBlockLength { .. })
    ));
}

#[test]
fn superblock_version_is_single_byte_at_offset_0x05() {
    let encoded = Superblock::new(SuperblockFields {
        total_blocks: 1,
        free_list_head: NO_BLOCK,
        integrity_chain_head: NO_BLOCK,
        redirection_table_start: NO_BLOCK,
        redirection_table_length: 0,
        file_table_start: NO_BLOCK,
        file_table_length: 0,
        flags: 0,
        quant_profile: 0,
    })
    .encode();

    assert_eq!(encoded[0x05], 1, "version byte at 0x05 should be 1");
    assert_eq!(
        u16::from_le_bytes([encoded[0x06], encoded[0x07]]),
        4096,
        "block size u16 at 0x06"
    );
}

#[test]
fn integrity_block_roundtrips_crc_entries() {
    let block = IntegrityBlock {
        first_data_block: 12,
        entry_count: 3,
        next_integrity_block: 18,
        crc32_entries: vec![0xAAAA_5555, 0xDEAD_BEEF, 0x1234_5678],
    };

    let encoded = block.encode().expect("encode integrity block");
    let decoded = IntegrityBlock::decode(&encoded).expect("decode integrity block");

    assert_eq!(decoded.first_data_block, 12);
    assert_eq!(decoded.entry_count, 3);
    assert_eq!(decoded.next_integrity_block, 18);
    assert_eq!(
        decoded.crc32_entries,
        vec![0xAAAA_5555, 0xDEAD_BEEF, 0x1234_5678]
    );
}

#[test]
fn integrity_block_capacity_matches_design() {
    assert_eq!(ENTRIES_PER_INTEGRITY_BLOCK, 1020);
}
