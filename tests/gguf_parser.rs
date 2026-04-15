mod common;

use common::{
    SyntheticGgufVersion, SyntheticTensorSpec, write_custom_gguf_fixture, write_parser_fixture,
    write_synthetic_gguf_fixture,
};

use llmdb::gguf::parser::{GgufMetadataValue, ParseError, parse_path};
use llmdb::gguf::quant::{GGML_TYPE_Q4_0_ID, GGML_TYPE_Q8_0_ID, GgufQuantType};

#[test]
fn parses_minimal_v2_fixture_with_metadata_and_tensor_info() {
    let fixture = write_parser_fixture(SyntheticGgufVersion::V2, "parser_full_v2.gguf");

    let parsed = parse_path(&fixture.path).expect("parse v2 fixture");

    assert_eq!(parsed.header.version, 2);
    assert_eq!(parsed.header.tensor_count, 1);
    assert_eq!(parsed.header.metadata_count, 3);
    assert_eq!(parsed.alignment, 32);
    assert_eq!(parsed.tensor_data_offset % 32, 0);
    assert_eq!(
        parsed.find_metadata_value("general.architecture"),
        Some(&GgufMetadataValue::String("llama".to_owned()))
    );
    assert_eq!(parsed.tokenizer_metadata().len(), 1);

    let tensor = &parsed.tensors[0];
    assert_eq!(tensor.name, "blk.0.ffn_down.weight");
    assert_eq!(tensor.dimensions, vec![4, 8]);
    assert_eq!(tensor.raw_type_id, GGML_TYPE_Q8_0_ID);
    assert_eq!(tensor.element_count(), 32);
    assert_eq!(
        tensor.absolute_offset(parsed.tensor_data_offset),
        Some(parsed.tensor_data_offset as u64)
    );
}

#[test]
fn parses_minimal_v3_fixture_with_metadata_and_tensor_info() {
    let fixture = write_parser_fixture(SyntheticGgufVersion::V3, "parser_full_v3.gguf");

    let parsed = parse_path(&fixture.path).expect("parse v3 fixture");

    assert_eq!(parsed.header.version, 3);
    assert_eq!(
        parsed.find_metadata_value("general.alignment"),
        Some(&GgufMetadataValue::Uint32(32))
    );
}

#[test]
fn parses_tensor_with_newly_declared_q4_0_type_id() {
    let fixture = write_custom_gguf_fixture(
        SyntheticGgufVersion::V3,
        "parser_q4_0.gguf",
        &[SyntheticTensorSpec {
            name: "blk.0.ffn_down.weight".to_owned(),
            dimensions: vec![32],
            raw_type_id: GGML_TYPE_Q4_0_ID,
            data: vec![0_u8; 18],
        }],
    );

    let parsed = parse_path(&fixture.path).expect("parse q4_0 fixture");
    assert_eq!(parsed.tensors.len(), 1);
    assert_eq!(parsed.tensors[0].raw_type_id, GGML_TYPE_Q4_0_ID);
    assert_eq!(parsed.tensors[0].quant_type(), Some(GgufQuantType::Q4_0));
}

#[test]
fn rejects_unsupported_versions() {
    let fixture = write_synthetic_gguf_fixture(SyntheticGgufVersion::V3, "unsupported.gguf");
    let mut bytes = std::fs::read(&fixture.path).expect("read fixture");
    bytes[4..8].copy_from_slice(&7_u32.to_le_bytes());
    std::fs::write(&fixture.path, bytes).expect("rewrite fixture");

    let err = parse_path(&fixture.path).expect_err("unsupported version should fail");
    assert!(matches!(err, ParseError::UnsupportedVersion(7)));
}
