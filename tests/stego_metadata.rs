use llmdb::stego::integrity::{
    ENTRIES_PER_INTEGRITY_BLOCK, IntegrityBlock, IntegrityError, NO_BLOCK, Superblock,
    SuperblockFields,
};

fn baseline_fields() -> SuperblockFields {
    SuperblockFields {
        total_blocks: 100,
        free_list_head: 12,
        integrity_chain_head: 1,
        redirection_table_start: 2,
        redirection_table_length: 1,
        file_table_start: 3,
        file_table_length: 1,
        flags: 0x01,
        quant_profile: 0b0010_0001,
        generation: 0,
        shadow_block: NO_BLOCK,
        shadow_target: NO_BLOCK,
    }
}

#[test]
fn superblock_roundtrips_with_checksum_validation() {
    let mut fields = baseline_fields();
    fields.generation = 0xDEAD_BEEF_1234_5678;
    fields.shadow_block = 42;
    fields.shadow_target = 99;

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
    assert_eq!(decoded.fields.generation, 0xDEAD_BEEF_1234_5678);
    assert_eq!(decoded.fields.shadow_block, 42);
    assert_eq!(decoded.fields.shadow_target, 99);
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
            generation: 0,
            shadow_block: NO_BLOCK,
            shadow_target: NO_BLOCK,
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
    let valid = Superblock::new(baseline_fields()).encode();

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

    // Invalid checksum — CRC lives at 0x38 under the current layout.
    let mut bad_crc = valid.clone();
    bad_crc[0x38] ^= 0xFF;
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
        generation: 0,
        shadow_block: NO_BLOCK,
        shadow_target: NO_BLOCK,
    })
    .encode();

    assert_eq!(encoded[0x05], 1, "version byte at 0x05 should be 1");
    assert_eq!(
        u16::from_le_bytes([encoded[0x06], encoded[0x07]]),
        4096,
        "block size u16 at 0x06"
    );
}

/// DESIGN-NEW §5 pins the new fields at specific offsets so the bit-level
/// layout is stable across readers. This test asserts each offset byte-wise
/// against raw encoded output.
#[test]
fn superblock_new_fields_live_at_design_new_offsets() {
    let mut fields = baseline_fields();
    fields.generation = 0x0102_0304_0506_0708;
    fields.shadow_block = 0x1111_2222;
    fields.shadow_target = 0x3333_4444;
    let encoded = Superblock::new(fields).encode();

    // Generation is a little-endian u64 at 0x28..0x30.
    assert_eq!(
        &encoded[0x28..0x30],
        &0x0102_0304_0506_0708_u64.to_le_bytes(),
        "generation at 0x28"
    );

    // Shadow block is a little-endian u32 at 0x30..0x34.
    assert_eq!(
        &encoded[0x30..0x34],
        &0x1111_2222_u32.to_le_bytes(),
        "shadow_block at 0x30"
    );

    // Shadow target is a little-endian u32 at 0x34..0x38.
    assert_eq!(
        &encoded[0x34..0x38],
        &0x3333_4444_u32.to_le_bytes(),
        "shadow_target at 0x34"
    );

    // The CRC sits at 0x38 and covers bytes 0x00..0x38. Flipping the CRC
    // bytes must trip a checksum mismatch on decode.
    let mut tampered = encoded.clone();
    tampered[0x38] ^= 0xFF;
    assert!(matches!(
        Superblock::decode(&tampered),
        Err(IntegrityError::ChecksumMismatch { .. })
    ));
}

/// NO_BLOCK is the sentinel for "no shadow in flight" per DESIGN-NEW §5.
/// Fresh fields with no write in flight must round-trip that sentinel.
#[test]
fn superblock_no_block_sentinel_survives_roundtrip() {
    let fields = baseline_fields();
    assert_eq!(fields.shadow_block, NO_BLOCK);
    assert_eq!(fields.shadow_target, NO_BLOCK);
    assert_eq!(fields.generation, 0);

    let decoded = Superblock::decode(&Superblock::new(fields).encode()).expect("decode");
    assert_eq!(decoded.fields.shadow_block, NO_BLOCK);
    assert_eq!(decoded.fields.shadow_target, NO_BLOCK);
    assert_eq!(decoded.fields.generation, 0);
}

/// A superblock written in the pre-layout-change format (CRC at 0x28 covering
/// 0x00..0x28, reserved area where the new fields now live) must fail decode
/// loudly — specifically with a checksum mismatch — rather than silently
/// misreading the reserved bytes as generation/shadow fields.
#[test]
fn superblock_old_layout_fails_loudly_on_new_decoder() {
    // Hand-build a superblock in the pre-change format.
    let mut bytes = vec![0_u8; llmdb::BLOCK_SIZE];
    bytes[0x00..0x05].copy_from_slice(b"LLMDB");
    bytes[0x05] = 1;
    bytes[0x06..0x08].copy_from_slice(&(llmdb::BLOCK_SIZE as u16).to_le_bytes());
    bytes[0x08..0x0C].copy_from_slice(&100_u32.to_le_bytes()); // total_blocks
    bytes[0x0C..0x10].copy_from_slice(&12_u32.to_le_bytes()); // free_list_head
    // integrity/redirection/file-table fields at 0x10..0x24: leave zero
    bytes[0x24] = 0x00; // flags
    bytes[0x25] = 0x00; // quant_profile
    // 0x26..0x28 reserved, zero
    // Old CRC at 0x28 covering 0x00..0x28.
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(&bytes[..0x28]);
    let old_crc = hasher.finalize();
    bytes[0x28..0x2C].copy_from_slice(&old_crc.to_le_bytes());
    // Rest stays zero, including where the new CRC at 0x38 would live.

    // Decoding with the new layout: the decoder computes CRC over bytes
    // 0x00..0x38 (which now includes the pre-change CRC bytes at 0x28 +
    // zero bytes for gen/shadow) and compares against bytes[0x38..0x3C] = 0.
    // That mismatch is the loud failure mode we want.
    let err = Superblock::decode(&bytes).expect_err("old layout must not decode");
    assert!(
        matches!(err, IntegrityError::ChecksumMismatch { .. }),
        "expected ChecksumMismatch, got {err:?}"
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
