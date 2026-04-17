use llmdb::stego::integrity::NO_BLOCK;
use llmdb::stego::redirection::{RedirectionError, RedirectionTable};

#[test]
fn empty_table_reports_every_logical_as_unmapped() {
    let table = RedirectionTable::empty(100);

    for logical in 0..100 {
        assert_eq!(
            table.logical_to_physical(logical),
            None,
            "fresh logical {logical} must be unmapped"
        );
        assert!(!table.is_mapped(logical));
    }
    assert_eq!(table.logical_to_physical(100), None); // out of range

    let encoded = table.encode();
    assert_eq!(encoded.len(), 1);
    assert_eq!(encoded[0].len(), llmdb::BLOCK_SIZE);

    let decoded = RedirectionTable::decode(&encoded).expect("decode empty table");
    for logical in 0..100 {
        assert_eq!(decoded.logical_to_physical(logical), None);
    }
}

#[test]
fn set_mapping_binds_logical_to_physical_and_survives_roundtrip() {
    let mut table = RedirectionTable::empty(20);

    table.set_mapping(5, 42);
    table.set_mapping(19, 0);

    assert_eq!(table.logical_to_physical(5), Some(42));
    assert_eq!(table.logical_to_physical(19), Some(0));
    assert_eq!(table.logical_to_physical(6), None);

    let encoded = table.encode();
    let decoded = RedirectionTable::decode(&encoded).expect("decode bound table");

    assert_eq!(decoded.logical_to_physical(5), Some(42));
    assert_eq!(decoded.logical_to_physical(19), Some(0));
    assert_eq!(decoded.logical_to_physical(6), None);
}

#[test]
fn clear_unbinds_logical() {
    let mut table = RedirectionTable::empty(10);
    table.set_mapping(3, 100);
    assert_eq!(table.logical_to_physical(3), Some(100));

    table.clear(3);
    assert_eq!(table.logical_to_physical(3), None);
    assert!(!table.is_mapped(3));
}

#[test]
fn cross_block_table_spans_multiple_blocks() {
    let total = 3000_u32;
    let table = RedirectionTable::empty(total);

    let encoded = table.encode();
    assert_eq!(
        encoded.len(),
        3,
        "3000 entries need 3 blocks (1022+1022+956)"
    );
    for block in &encoded {
        assert_eq!(block.len(), llmdb::BLOCK_SIZE);
    }

    let decoded = RedirectionTable::decode(&encoded).expect("decode 3-block table");

    assert_eq!(decoded.logical_to_physical(0), None);
    assert_eq!(decoded.logical_to_physical(1021), None);
    assert_eq!(decoded.logical_to_physical(1022), None);
    assert_eq!(decoded.logical_to_physical(2999), None);
    assert_eq!(decoded.logical_to_physical(3000), None); // out of range
}

#[test]
fn cross_block_set_mapping_targets_correct_block() {
    let mut table = RedirectionTable::empty(3000);

    table.set_mapping(1500, 9999);

    let encoded = table.encode();
    let decoded = RedirectionTable::decode(&encoded).expect("decode");

    assert_eq!(decoded.logical_to_physical(1500), Some(9999));
    assert_eq!(decoded.logical_to_physical(1499), None);
    assert_eq!(decoded.logical_to_physical(1501), None);
}

#[test]
fn decode_rejects_wrong_block_length() {
    let result = RedirectionTable::decode(&[vec![0_u8; 100]]);
    assert!(matches!(
        result,
        Err(RedirectionError::InvalidBlockLength { .. })
    ));
}

#[test]
fn decode_rejects_entry_count_exceeding_capacity() {
    let mut block = vec![0_u8; llmdb::BLOCK_SIZE];
    block[0..4].copy_from_slice(&0_u32.to_le_bytes());
    block[4..8].copy_from_slice(&2000_u32.to_le_bytes());

    let result = RedirectionTable::decode(&[block]);
    assert!(matches!(
        result,
        Err(RedirectionError::EntryCountOverflow { .. })
    ));
}

#[test]
fn empty_zero_blocks_produces_empty_table() {
    let table = RedirectionTable::empty(0);
    assert_eq!(table.logical_to_physical(0), None);
    let encoded = table.encode();
    assert!(encoded.is_empty());
}

#[test]
fn no_block_sentinel_round_trips_through_encoding() {
    // An entry with NO_BLOCK sentinel must round-trip as "unmapped".
    let mut table = RedirectionTable::empty(4);
    table.set_mapping(2, 7);
    table.clear(2);
    let encoded = table.encode();
    let decoded = RedirectionTable::decode(&encoded).expect("decode");
    assert_eq!(decoded.logical_to_physical(2), None);
    // Sanity: the on-disk byte representation is NO_BLOCK little-endian.
    let entry_bytes = &encoded[0][8 + 2 * 4..8 + 2 * 4 + 4];
    assert_eq!(
        u32::from_le_bytes(entry_bytes.try_into().unwrap()),
        NO_BLOCK
    );
}
