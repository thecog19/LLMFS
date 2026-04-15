#[test]
fn crate_exports_bootstrap_contracts() {
    assert_eq!(llmdb::APP_NAME, "llmdb");
    assert_eq!(llmdb::BLOCK_SIZE, 4096);
    assert_eq!(llmdb::bootstrap_status(), "bootstrap");

    let parser = llmdb::gguf::parser::ParserBootstrap;
    assert_eq!(parser.supported_versions(), &[2, 3]);

    let device = llmdb::stego::device::DeviceBootstrap::default();
    assert_eq!(device.block_size, llmdb::BLOCK_SIZE);

    let vfs = llmdb::vfs::sqlite_vfs::SqliteVfsBootstrap::default();
    assert_eq!(vfs.page_size, llmdb::BLOCK_SIZE);

    let compression = llmdb::compress::bpe::CompressionBootstrap::default();
    assert!(compression.enabled_by_default);

    let diagnostics = llmdb::diagnostics::DiagnosticsBootstrap::default();
    assert!(diagnostics.tracks_tiers);

    let packers = llmdb::stego::packing::supported_packers();
    assert!(packers.contains(&"q8_0"));
    assert!(packers.contains(&"float"));

    assert_eq!(
        llmdb::nlq::consistency::classify_temperature(0.0),
        llmdb::nlq::consistency::ConsistencyMode::Strong
    );
}
