use llmdb::stego::integrity::{
    ENTRIES_PER_INTEGRITY_BLOCK, IntegrityBlock, NO_BLOCK, PendingMetadataOp, Superblock,
    SuperblockFields,
};

#[test]
fn superblock_roundtrips_with_checksum_validation() {
    let fields = SuperblockFields {
        total_blocks: 9,
        free_list_head: 3,
        table_directory_block: NO_BLOCK,
        integrity_chain_head: 7,
        wal_region_start: NO_BLOCK,
        wal_region_length: 0,
        shadow_block: 8,
        pending_target_block: 6,
        pending_target_crc32: 0xDEAD_BEEF,
        pending_metadata_op: PendingMetadataOp::FreeHeadPush,
        pending_metadata_block: 5,
        pending_metadata_aux: 3,
        generation: 42,
    };

    let encoded = Superblock::new(fields).encode();
    let decoded = Superblock::decode(&encoded).expect("decode superblock");

    assert_eq!(decoded.fields.total_blocks, 9);
    assert_eq!(decoded.fields.free_list_head, 3);
    assert_eq!(decoded.fields.integrity_chain_head, 7);
    assert_eq!(decoded.fields.shadow_block, 8);
    assert_eq!(decoded.fields.pending_target_block, 6);
    assert_eq!(decoded.fields.pending_target_crc32, 0xDEAD_BEEF);
    assert_eq!(
        decoded.fields.pending_metadata_op,
        PendingMetadataOp::FreeHeadPush
    );
    assert_eq!(decoded.fields.pending_metadata_block, 5);
    assert_eq!(decoded.fields.pending_metadata_aux, 3);
    assert_eq!(decoded.fields.generation, 42);
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
