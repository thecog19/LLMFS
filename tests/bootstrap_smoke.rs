#[test]
fn crate_exports_bootstrap_contracts() {
    assert_eq!(llmdb::APP_NAME, "llmdb");
    assert_eq!(llmdb::BLOCK_SIZE, 4096);
    assert_eq!(llmdb::bootstrap_status(), "bootstrap");

    let parser = llmdb::gguf::parser::ParserBootstrap;
    assert_eq!(parser.supported_versions(), &[2, 3]);

    let device = llmdb::stego::device::DeviceBootstrap::default();
    assert_eq!(device.block_size, llmdb::BLOCK_SIZE);

    let redirection = llmdb::stego::redirection::RedirectionBootstrap::default();
    assert_eq!(redirection.entries_per_block, 1022);

    let freelist = llmdb::stego::freelist::FreeListBootstrap::default();
    assert_eq!(freelist.block_size, llmdb::BLOCK_SIZE);

    let diagnostics = llmdb::diagnostics::DiagnosticsBootstrap::default();
    assert!(diagnostics.tracks_tiers);

    let file_table = llmdb::fs::file_table::FileTableBootstrap::default();
    assert_eq!(file_table.entries_per_block, 16);

    let nbd = llmdb::nbd::protocol::NbdProtocolBootstrap::default();
    assert_eq!(nbd.block_size, llmdb::BLOCK_SIZE);

    let ask = llmdb::ask::bridge::AskBridgeBootstrap::default();
    assert_eq!(ask.tool_count, 3);

    let packers = llmdb::stego::packing::supported_packers();
    assert!(packers.contains(&"q8_0"));
    assert!(packers.contains(&"float"));
}
