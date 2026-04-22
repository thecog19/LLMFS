//! V2 CLI smoke tests — drive the binary end-to-end via assert_cmd.
//!
//! Builds a synthetic F16 GGUF fixture, runs `llmdb init` on it, then
//! exercises the V2 verbs: status, store, ls, get, rm, ask-help, etc.

mod common;

use std::io::Write;
use std::path::Path;

use assert_cmd::Command;
use common::{SyntheticGgufVersion, SyntheticTensorSpec, write_custom_gguf_fixture};
use llmdb::gguf::quant::GGML_TYPE_F16_ID;

const F16_BYTES_PER_WEIGHT: usize = 2;

fn f16_tensor(name: &str, weight_count: usize) -> SyntheticTensorSpec {
    SyntheticTensorSpec {
        name: name.to_owned(),
        dimensions: vec![weight_count as u64],
        raw_type_id: GGML_TYPE_F16_ID,
        // Bits don't matter for CLI smoke — anchor + bitmap + alloc
        // succeed against zero-init bytes (capacity is what matters).
        data: vec![0_u8; weight_count * F16_BYTES_PER_WEIGHT],
    }
}

fn make_f16_tensors(count: usize, weights_per_tensor: usize) -> Vec<SyntheticTensorSpec> {
    (0..count)
        .map(|i| f16_tensor(&format!("blk.{}.ffn_down.weight", count - 1 - i), weights_per_tensor))
        .collect()
}

fn fixture(name: &str) -> common::FixtureHandle {
    write_custom_gguf_fixture(
        SyntheticGgufVersion::V3,
        name,
        // 12 tensors × 32 768 weights = 393 216 F16 weights × 4 stealable bits
        // = 1 572 864 bits ≈ 192 KiB capacity. Comfortable headroom for
        // anchor + root dir + dirty bitmap + a few small files.
        &make_f16_tensors(12, 32_768),
    )
}

fn llmdb() -> Command {
    Command::cargo_bin("llmdb").expect("cargo_bin(llmdb)")
}

fn init(model: &Path) {
    llmdb().arg("init").arg(model).assert().success();
}

#[test]
fn init_reports_capacity_summary() {
    let fx = fixture("cli_init.gguf");
    let out = llmdb().arg("init").arg(&fx.path).assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).into_owned();

    for label in ["initialized", "cover size:", "eligible slots:", "eligible weights:", "generation:"] {
        assert!(
            stdout.contains(label),
            "init output missing `{label}`:\n{stdout}"
        );
    }
}

#[test]
fn status_after_init_lists_v2_native_fields() {
    let fx = fixture("cli_status.gguf");
    init(&fx.path);

    let out = llmdb().arg("status").arg(&fx.path).assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).into_owned();
    for label in [
        "device:",
        "generation:",
        "files:",
        "directories:",
        "stored:",
        "dirty weights:",
        "allocator:",
        "dedup entries:",
        "quant profile:",
    ] {
        assert!(
            stdout.contains(label),
            "status output missing `{label}`:\n{stdout}"
        );
    }
}

#[test]
fn store_ls_get_rm_roundtrip() {
    let fx = fixture("cli_round.gguf");
    init(&fx.path);

    // Write a known payload to a host file.
    let mut tmp = tempfile::NamedTempFile::new().unwrap();
    let payload = b"V2 round-trip payload";
    tmp.as_file_mut().write_all(payload).unwrap();
    tmp.as_file_mut().flush().unwrap();

    // store
    llmdb()
        .arg("store")
        .arg(&fx.path)
        .arg(tmp.path())
        .arg("--stego-path")
        .arg("/notes/round.txt")
        .assert()
        .success();

    // ls at root and at /notes
    let ls_root = llmdb().arg("ls").arg(&fx.path).assert().success();
    let root_out = String::from_utf8_lossy(&ls_root.get_output().stdout).into_owned();
    assert!(root_out.contains("notes/"), "root ls missing notes/: {root_out}");

    let ls_notes = llmdb()
        .arg("ls")
        .arg(&fx.path)
        .arg("/notes")
        .assert()
        .success();
    let notes_out = String::from_utf8_lossy(&ls_notes.get_output().stdout).into_owned();
    assert!(
        notes_out.contains("round.txt"),
        "notes ls missing round.txt: {notes_out}"
    );

    // get into a fresh tempfile
    let dest_dir = tempfile::tempdir().unwrap();
    let dest = dest_dir.path().join("round.out");
    llmdb()
        .arg("get")
        .arg(&fx.path)
        .arg("/notes/round.txt")
        .arg("--output")
        .arg(&dest)
        .assert()
        .success();
    let read_back = std::fs::read(&dest).unwrap();
    assert_eq!(read_back, payload, "payload round-trip mismatch");

    // rm with --yes
    llmdb()
        .arg("rm")
        .arg(&fx.path)
        .arg("/notes/round.txt")
        .arg("--yes")
        .assert()
        .success();

    let ls_after = llmdb()
        .arg("ls")
        .arg(&fx.path)
        .arg("/notes")
        .assert()
        .success();
    let after_out = String::from_utf8_lossy(&ls_after.get_output().stdout).into_owned();
    assert!(
        !after_out.contains("round.txt"),
        "round.txt should be gone after rm: {after_out}"
    );
}

#[test]
fn rm_on_empty_directory_succeeds() {
    let fx = fixture("cli_rmdir.gguf");
    init(&fx.path);

    // store creates parents; then we rm the file, leaving an empty dir.
    let tmp = tempfile::NamedTempFile::new().unwrap();
    llmdb()
        .arg("store")
        .arg(&fx.path)
        .arg(tmp.path())
        .arg("--stego-path")
        .arg("/d/leaf.bin")
        .assert()
        .success();
    llmdb()
        .arg("rm")
        .arg(&fx.path)
        .arg("/d/leaf.bin")
        .arg("--yes")
        .assert()
        .success();

    // /d is now empty. rm should fall through unlink → rmdir.
    llmdb()
        .arg("rm")
        .arg(&fx.path)
        .arg("/d")
        .arg("--yes")
        .assert()
        .success();

    let ls = llmdb().arg("ls").arg(&fx.path).assert().success();
    let stdout = String::from_utf8_lossy(&ls.get_output().stdout).into_owned();
    assert!(stdout.trim().is_empty(), "root should be empty: {stdout}");
}

#[test]
fn get_unknown_path_exits_with_user_error() {
    let fx = fixture("cli_missing.gguf");
    init(&fx.path);

    let out = llmdb()
        .arg("get")
        .arg(&fx.path)
        .arg("/no-such-file.txt")
        .arg("--output")
        .arg("/tmp/should-not-be-written")
        .assert()
        .failure();
    let code = out.get_output().status.code().unwrap_or(-1);
    assert_eq!(code, 1, "expected user-error exit code 1, got {code}");
}

#[test]
fn re_init_wipes_prior_state() {
    let fx = fixture("cli_reinit.gguf");
    init(&fx.path);

    let tmp = tempfile::NamedTempFile::new().unwrap();
    llmdb()
        .arg("store")
        .arg(&fx.path)
        .arg(tmp.path())
        .arg("--stego-path")
        .arg("/dying.txt")
        .assert()
        .success();

    // Re-init replaces the V2 anchor + root dir; prior file is gone.
    init(&fx.path);
    let ls = llmdb().arg("ls").arg(&fx.path).assert().success();
    let stdout = String::from_utf8_lossy(&ls.get_output().stdout).into_owned();
    assert!(stdout.trim().is_empty(), "re-init should leave 0 files: {stdout}");
}

#[test]
fn help_flag_lists_v2_subcommands() {
    let out = llmdb().arg("--help").assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).into_owned();
    for verb in ["init", "status", "mount", "unmount", "ls", "store", "get", "rm", "ask"] {
        assert!(
            stdout.contains(verb),
            "--help missing verb `{verb}`:\n{stdout}"
        );
    }
    // V1 verbs that should be gone.
    for gone in ["wipe", "verify", "dump-block"] {
        assert!(
            !stdout.contains(gone),
            "--help still mentions removed verb `{gone}`:\n{stdout}"
        );
    }
}

#[test]
fn store_subcommand_help_names_its_args() {
    let out = llmdb().arg("store").arg("--help").assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).into_owned();
    // clap renders required positionals in uppercase angle-brackets.
    for token in ["<MODEL>", "<HOST_PATH>", "--stego-path"] {
        assert!(
            stdout.contains(token),
            "store --help missing `{token}`:\n{stdout}"
        );
    }
}
