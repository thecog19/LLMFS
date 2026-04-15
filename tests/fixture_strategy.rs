mod common;

use common::{SyntheticGgufVersion, write_synthetic_gguf_fixture};

#[test]
fn synthetic_v2_fixture_is_created_inside_temp_space() {
    let fixture = write_synthetic_gguf_fixture(SyntheticGgufVersion::V2, "parser_v2.gguf");

    assert!(fixture.path.exists());
    assert!(fixture.path.starts_with(fixture.root()));

    let bytes = std::fs::read(&fixture.path).expect("read fixture");
    assert!(bytes.starts_with(b"GGUF"));
    assert_eq!(bytes.len(), 24);
}

#[test]
fn synthetic_v3_fixture_is_created_inside_temp_space() {
    let fixture = write_synthetic_gguf_fixture(SyntheticGgufVersion::V3, "parser_v3.gguf");

    assert!(fixture.path.exists());
    assert!(fixture.path.starts_with(fixture.root()));

    let bytes = std::fs::read(&fixture.path).expect("read fixture");
    assert_eq!(&bytes[4..8], &3_u32.to_le_bytes());
}
