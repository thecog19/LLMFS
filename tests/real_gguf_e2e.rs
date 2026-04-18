use std::io::Write;
use std::path::{Path, PathBuf};

use assert_cmd::Command;

fn llmdb() -> Command {
    Command::cargo_bin("llmdb").expect("cargo_bin(llmdb)")
}

fn locate_model() -> Option<PathBuf> {
    std::env::var("LLMDB_E2E_GGUF")
        .ok()
        .map(PathBuf::from)
        .filter(|path| path.exists())
}

fn file_bytes(path: &Path) -> Vec<u8> {
    std::fs::read(path).expect("read file")
}

#[test]
fn real_gguf_cli_roundtrip_survives_init_store_get_verify_rm() {
    let Some(model) = locate_model() else {
        eprintln!("LLMDB_E2E_GGUF unset; skipping real_gguf_e2e");
        return;
    };

    let dir = tempfile::tempdir().expect("tempdir");
    let stego = dir.path().join("real-e2e.gguf");
    std::fs::copy(&model, &stego).expect("copy model");

    let payload_path = dir.path().join("payload.txt");
    let mut payload = std::fs::File::create(&payload_path).expect("create payload");
    writeln!(payload, "real gguf e2e payload").expect("write payload line 1");
    writeln!(payload, "line two").expect("write payload line 2");

    llmdb().arg("init").arg(&stego).assert().success();
    llmdb()
        .arg("store")
        .arg(&stego)
        .arg(&payload_path)
        .arg("--name")
        .arg("payload.txt")
        .assert()
        .success();

    let ls = llmdb().arg("ls").arg(&stego).assert().success();
    let ls_stdout = String::from_utf8_lossy(&ls.get_output().stdout).into_owned();
    assert!(
        ls_stdout.contains("payload.txt"),
        "ls output was {ls_stdout:?}"
    );

    let output_path = dir.path().join("out.txt");
    llmdb()
        .arg("get")
        .arg(&stego)
        .arg("payload.txt")
        .arg("--output")
        .arg(&output_path)
        .assert()
        .success();

    assert_eq!(
        file_bytes(&payload_path),
        file_bytes(&output_path),
        "roundtrip through a real GGUF must preserve file bytes"
    );

    let verify = llmdb().arg("verify").arg(&stego).assert().success();
    let verify_stdout = String::from_utf8_lossy(&verify.get_output().stdout).into_owned();
    assert!(
        verify_stdout.contains("verify: OK"),
        "verify output was {verify_stdout:?}"
    );

    llmdb()
        .arg("rm")
        .arg(&stego)
        .arg("payload.txt")
        .arg("--yes")
        .assert()
        .success();

    let ls_after = llmdb().arg("ls").arg(&stego).assert().success();
    let ls_after_stdout = String::from_utf8_lossy(&ls_after.get_output().stdout).into_owned();
    assert!(
        !ls_after_stdout.contains("payload.txt"),
        "file should be gone after rm, ls output was {ls_after_stdout:?}"
    );
}
