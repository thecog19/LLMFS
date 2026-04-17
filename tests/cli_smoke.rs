mod common;

use std::io::Write;
use std::path::Path;

use assert_cmd::Command;
use common::{SyntheticGgufVersion, SyntheticTensorSpec, write_custom_gguf_fixture};
use llmdb::gguf::quant::GGML_TYPE_Q8_0_ID;

fn q8_tensor(name: &str, weight_count: usize) -> SyntheticTensorSpec {
    let chunk_count = weight_count / 32;
    SyntheticTensorSpec {
        name: name.to_owned(),
        dimensions: vec![weight_count as u64],
        raw_type_id: GGML_TYPE_Q8_0_ID,
        data: vec![0_u8; chunk_count * 34],
    }
}

fn make_q8_tensors(count: usize) -> Vec<SyntheticTensorSpec> {
    (0..count)
        .map(|i| q8_tensor(&format!("blk.{}.ffn_down.weight", count - 1 - i), 8192))
        .collect()
}

fn fixture(name: &str, tensor_count: usize) -> common::FixtureHandle {
    write_custom_gguf_fixture(
        SyntheticGgufVersion::V3,
        name,
        &make_q8_tensors(tensor_count),
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
    let fx = fixture("cli_init.gguf", 12);
    let out = llmdb().arg("init").arg(&fx.path).assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).into_owned();

    assert!(
        stdout.contains("total blocks"),
        "missing total blocks: {stdout}"
    );
    assert!(
        stdout.contains("data blocks"),
        "missing data blocks: {stdout}"
    );
    assert!(
        stdout.contains("quant profile"),
        "missing quant profile: {stdout}"
    );
}

#[test]
fn store_ls_get_rm_roundtrip_via_cli() {
    let fx = fixture("cli_roundtrip.gguf", 12);
    init(&fx.path);

    // Write a source file to host tempdir.
    let payload: Vec<u8> = (0..2048).map(|i| (i % 251) as u8).collect();
    let tmp = tempfile::NamedTempFile::new().unwrap();
    tmp.as_file().write_all(&payload).unwrap();

    llmdb()
        .arg("store")
        .arg(&fx.path)
        .arg(tmp.path())
        .arg("--name")
        .arg("roundtrip.bin")
        .assert()
        .success();

    let ls = llmdb().arg("ls").arg(&fx.path).assert().success();
    let ls_stdout = String::from_utf8_lossy(&ls.get_output().stdout).into_owned();
    assert!(
        ls_stdout.contains("roundtrip.bin"),
        "ls missing file: {ls_stdout}"
    );

    let out = tempfile::NamedTempFile::new().unwrap();
    llmdb()
        .arg("get")
        .arg(&fx.path)
        .arg("roundtrip.bin")
        .arg("--output")
        .arg(out.path())
        .assert()
        .success();

    let readback = std::fs::read(out.path()).unwrap();
    assert_eq!(readback, payload, "CLI roundtrip must preserve bytes");

    llmdb()
        .arg("rm")
        .arg(&fx.path)
        .arg("roundtrip.bin")
        .arg("--yes")
        .assert()
        .success();

    let ls2 = llmdb().arg("ls").arg(&fx.path).assert().success();
    let ls2_stdout = String::from_utf8_lossy(&ls2.get_output().stdout).into_owned();
    assert!(
        !ls2_stdout.contains("roundtrip.bin"),
        "ls after rm should not list the file: {ls2_stdout}"
    );
}

#[test]
fn wipe_yes_leaves_device_fresh_with_zero_files() {
    let fx = fixture("cli_wipe.gguf", 12);
    init(&fx.path);

    let tmp = tempfile::NamedTempFile::new().unwrap();
    tmp.as_file().write_all(b"placeholder").unwrap();
    llmdb()
        .arg("store")
        .arg(&fx.path)
        .arg(tmp.path())
        .arg("--name")
        .arg("dying.txt")
        .assert()
        .success();

    llmdb()
        .arg("wipe")
        .arg(&fx.path)
        .arg("--yes")
        .assert()
        .success();

    let status = llmdb().arg("status").arg(&fx.path).assert().success();
    let stdout = String::from_utf8_lossy(&status.get_output().stdout).into_owned();
    assert!(
        stdout.contains("files:"),
        "status missing files line: {stdout}"
    );
    assert!(
        stdout.lines().any(|line| line.trim() == "files:       0"),
        "wipe should leave 0 files visible, status was:\n{stdout}"
    );
}

#[test]
fn verify_on_clean_device_returns_ok() {
    let fx = fixture("cli_verify_ok.gguf", 12);
    init(&fx.path);

    let tmp = tempfile::NamedTempFile::new().unwrap();
    tmp.as_file().write_all(b"healthy").unwrap();
    llmdb()
        .arg("store")
        .arg(&fx.path)
        .arg(tmp.path())
        .arg("--name")
        .arg("good.txt")
        .assert()
        .success();

    let out = llmdb().arg("verify").arg(&fx.path).assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).into_owned();
    assert!(stdout.contains("OK"), "expected OK, got: {stdout}");
}

#[test]
fn verify_on_corrupted_fixture_reports_failure() {
    use llmdb::gguf::parser::parse_path;
    use llmdb::stego::planner::{AllocationMode, build_allocation_plan};

    let fx = fixture("cli_verify_bad.gguf", 12);
    init(&fx.path);

    let tmp = tempfile::NamedTempFile::new().unwrap();
    tmp.as_file().write_all(&vec![0x42_u8; 4096]).unwrap();
    llmdb()
        .arg("store")
        .arg(&fx.path)
        .arg(tmp.path())
        .arg("--name")
        .arg("victim.bin")
        .assert()
        .success();

    // Corrupt a non-scale byte inside the tensor that backs the first data
    // block (which is where `alloc_block` placed victim.bin). We parse the
    // fixture to find the right physical offset — picking the middle of the
    // gguf file as a "random" spot is not reliable on a 12-tensor fixture.
    let parsed = parse_path(&fx.path).expect("parse fixture");
    let plan = build_allocation_plan(&parsed.tensors, AllocationMode::Standard);
    let data_start = 4_usize; // 1 super + 1 integrity + 1 redir + 1 file_table
    let planned = &plan.tensors[data_start];
    let target_tensor = parsed
        .tensors
        .iter()
        .find(|t| t.name == planned.name)
        .expect("find tensor for first data block");
    let tensor_offset = target_tensor
        .absolute_offset(parsed.tensor_data_offset)
        .expect("tensor offset") as usize;
    // Offset +2 lands inside a Q8_0 block's weight payload (byte 0..1 are
    // the fp16 scale; byte 2+ are the quantized weights whose low nibble
    // we steal for stego).
    let corrupt_offset = tensor_offset + 2;

    let mut bytes = std::fs::read(&fx.path).unwrap();
    bytes[corrupt_offset] ^= 0x01;
    std::fs::write(&fx.path, bytes).unwrap();

    let out = llmdb().arg("verify").arg(&fx.path).assert().failure();
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.get_output().stdout),
        String::from_utf8_lossy(&out.get_output().stderr)
    );
    assert!(
        combined.contains("corrupted") || combined.contains("integrity"),
        "verify output should mention corruption; got: {combined}"
    );
}

#[test]
fn get_unknown_file_exits_with_user_error() {
    let fx = fixture("cli_notfound.gguf", 12);
    init(&fx.path);

    let out_path = tempfile::NamedTempFile::new().unwrap();
    let assert = llmdb()
        .arg("get")
        .arg(&fx.path)
        .arg("ghost.txt")
        .arg("--output")
        .arg(out_path.path())
        .assert()
        .failure();
    let code = assert.get_output().status.code().unwrap_or(0);
    assert_eq!(code, 1, "file-not-found should be a user error (exit 1)");
}

#[test]
fn dump_emits_ustar_archive_of_stored_files() {
    let fx = fixture("cli_dump.gguf", 12);
    init(&fx.path);

    for (name, payload) in [
        ("alpha.txt", b"alpha bytes" as &[u8]),
        ("beta.bin", &[0xAA; 2048]),
    ] {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.as_file().write_all(payload).unwrap();
        llmdb()
            .arg("store")
            .arg(&fx.path)
            .arg(tmp.path())
            .arg("--name")
            .arg(name)
            .assert()
            .success();
    }

    let out = llmdb().arg("dump").arg(&fx.path).assert().success();
    let stdout = out.get_output().stdout.clone();

    // Must be a multiple of 512 (tar blocks).
    assert!(
        stdout.len() >= 512 * 4,
        "dump too short: {} bytes",
        stdout.len()
    );
    assert_eq!(stdout.len() % 512, 0, "dump must be multiple of 512 bytes");

    // Both filenames must appear at the start of a 512-byte block (header).
    let alpha_hit = stdout
        .chunks_exact(512)
        .any(|block| block.starts_with(b"alpha.txt\0"));
    let beta_hit = stdout
        .chunks_exact(512)
        .any(|block| block.starts_with(b"beta.bin\0"));
    assert!(alpha_hit, "alpha.txt header missing from dump");
    assert!(beta_hit, "beta.bin header missing from dump");

    // Last two blocks are the zero terminator.
    let tail_start = stdout.len() - 1024;
    assert!(
        stdout[tail_start..].iter().all(|&b| b == 0),
        "dump must end with two zero blocks (ustar terminator)"
    );

    // `alpha bytes` payload must follow its header block at 512-byte alignment.
    let alpha_block = stdout
        .chunks_exact(512)
        .position(|block| block.starts_with(b"alpha.txt\0"))
        .unwrap();
    let alpha_payload_start = (alpha_block + 1) * 512;
    assert_eq!(
        &stdout[alpha_payload_start..alpha_payload_start + b"alpha bytes".len()],
        b"alpha bytes"
    );
}

#[test]
fn help_flag_lists_subcommands() {
    let out = llmdb().arg("--help").assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).into_owned();
    for sub in [
        "init", "status", "store", "get", "ls", "rm", "verify", "wipe",
    ] {
        assert!(
            stdout.contains(sub),
            "top-level --help missing subcommand {sub}:\n{stdout}"
        );
    }
}

#[test]
fn subcommand_help_names_its_args() {
    let out = llmdb().arg("store").arg("--help").assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).into_owned();
    let lowered = stdout.to_lowercase();
    assert!(
        lowered.contains("model"),
        "store --help missing model arg: {stdout}"
    );
    assert!(
        lowered.contains("host_path"),
        "store --help missing host_path arg: {stdout}"
    );
}
