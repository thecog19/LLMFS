use std::path::PathBuf;

fn repo_file(path: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path)
}

#[test]
fn readme_get_example_uses_output_flag() {
    let readme = std::fs::read_to_string(repo_file("README.md")).expect("read README.md");
    assert!(
        readme.contains("./target/release/llmdb get model.gguf /notes.txt --output ./notes.out"),
        "README quickstart should match the current V2 CLI get syntax"
    );
}

#[test]
fn docs_do_not_claim_a_stale_fixed_test_count() {
    for path in ["README.md", "CHANGELOG.md"] {
        let body = std::fs::read_to_string(repo_file(path)).unwrap_or_else(|e| {
            panic!("read {path}: {e}");
        });
        assert!(
            !body.contains("131 tests"),
            "{path} still claims an outdated fixed test count"
        );
    }
}
