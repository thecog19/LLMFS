use llmdb::stego::redirection::{RedirectionError, RedirectionTable};

#[test]
fn identity_mapping_roundtrips_through_encode_decode() {
    let table = RedirectionTable::identity(100);

    for logical in 0..100 {
        assert_eq!(
            table.logical_to_physical(logical),
            Some(logical),
            "identity mapping at {logical}"
        );
    }
    assert_eq!(table.logical_to_physical(100), None);

    let encoded = table.encode();
    assert_eq!(encoded.len(), 1);
    assert_eq!(encoded[0].len(), llmdb::BLOCK_SIZE);

    let decoded = RedirectionTable::decode(&encoded).expect("decode identity table");
    for logical in 0..100 {
        assert_eq!(decoded.logical_to_physical(logical), Some(logical));
    }
}

#[test]
fn set_mapping_overrides_identity_and_survives_roundtrip() {
    let mut table = RedirectionTable::identity(20);

    table.set_mapping(5, 42);
    table.set_mapping(19, 0);

    assert_eq!(table.logical_to_physical(5), Some(42));
    assert_eq!(table.logical_to_physical(19), Some(0));
    assert_eq!(table.logical_to_physical(6), Some(6));

    let encoded = table.encode();
    let decoded = RedirectionTable::decode(&encoded).expect("decode updated table");

    assert_eq!(decoded.logical_to_physical(5), Some(42));
    assert_eq!(decoded.logical_to_physical(19), Some(0));
    assert_eq!(decoded.logical_to_physical(6), Some(6));
}

#[test]
fn cross_block_table_spans_multiple_blocks() {
    let total = 3000_u32;
    let table = RedirectionTable::identity(total);

    let encoded = table.encode();
    assert_eq!(encoded.len(), 3, "3000 entries need 3 blocks (1022+1022+956)");
    for block in &encoded {
        assert_eq!(block.len(), llmdb::BLOCK_SIZE);
    }

    let decoded = RedirectionTable::decode(&encoded).expect("decode 3-block table");

    assert_eq!(decoded.logical_to_physical(0), Some(0));
    assert_eq!(decoded.logical_to_physical(1021), Some(1021));
    assert_eq!(decoded.logical_to_physical(1022), Some(1022));
    assert_eq!(decoded.logical_to_physical(2043), Some(2043));
    assert_eq!(decoded.logical_to_physical(2044), Some(2044));
    assert_eq!(decoded.logical_to_physical(2999), Some(2999));
    assert_eq!(decoded.logical_to_physical(3000), None);
}

#[test]
fn cross_block_set_mapping_targets_correct_block() {
    let mut table = RedirectionTable::identity(3000);

    table.set_mapping(1500, 9999);

    let encoded = table.encode();
    let decoded = RedirectionTable::decode(&encoded).expect("decode");

    assert_eq!(decoded.logical_to_physical(1500), Some(9999));
    assert_eq!(decoded.logical_to_physical(1499), Some(1499));
    assert_eq!(decoded.logical_to_physical(1501), Some(1501));
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
    // first_logical_block = 0
    block[0..4].copy_from_slice(&0_u32.to_le_bytes());
    // entry_count = 2000 (exceeds 1022)
    block[4..8].copy_from_slice(&2000_u32.to_le_bytes());

    let result = RedirectionTable::decode(&[block]);
    assert!(matches!(
        result,
        Err(RedirectionError::EntryCountOverflow { .. })
    ));
}

#[test]
fn identity_zero_blocks_produces_empty_table() {
    let table = RedirectionTable::identity(0);
    assert_eq!(table.logical_to_physical(0), None);
    let encoded = table.encode();
    assert!(encoded.is_empty());
}
