use std::path::PathBuf;
use std::process::Command;

fn helper_script() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("scripts")
        .join("llmdb-e2e-root.sh")
}

#[test]
fn root_helper_script_exists_and_parses() {
    let script = helper_script();
    assert!(
        script.exists(),
        "expected helper script at {}",
        script.display()
    );

    let status = Command::new("bash")
        .arg("-n")
        .arg(&script)
        .status()
        .expect("run bash -n");
    assert!(status.success(), "bash -n failed for {}", script.display());
}

#[test]
fn root_helper_script_help_mentions_supported_commands() {
    let script = helper_script();
    let output = Command::new("bash")
        .arg(&script)
        .arg("--help")
        .output()
        .expect("run helper --help");
    assert!(
        output.status.success(),
        "--help failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    for command in [
        "install-sudoers",
        "prepare",
        "pick-free",
        "attach",
        "copy-file",
        "format",
        "mountfs",
        "unmountfs",
        "detach",
        "cleanup",
        "status",
    ] {
        assert!(
            stdout.contains(command),
            "missing command {command} in help output:\n{stdout}"
        );
    }
}
