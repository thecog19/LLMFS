use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use assert_cmd::Command as AssertCommand;

fn gated() -> bool {
    std::env::var("LLMDB_E2E_MOUNT").ok().as_deref() == Some("1")
}

fn helper_path() -> PathBuf {
    PathBuf::from(
        std::env::var("LLMDB_ROOT_HELPER")
            .expect("LLMDB_ROOT_HELPER must point at llmdb-e2e-root.sh"),
    )
}

fn locate_model() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("LLMDB_E2E_GGUF") {
        let path = PathBuf::from(path);
        if path.exists() {
            return Some(path);
        }
    }

    let fallback = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models")
        .join("stories15M-q8_0.gguf");
    fallback.exists().then_some(fallback)
}

fn llmdb() -> AssertCommand {
    AssertCommand::cargo_bin("llmdb").expect("cargo_bin(llmdb)")
}

fn run_helper(helper: &Path, args: &[&str]) {
    let status = Command::new("sudo")
        .arg("-n")
        .arg(helper)
        .args(args)
        .status()
        .expect("run helper");
    assert!(status.success(), "helper {:?} failed with {status:?}", args);
}

fn helper_output(helper: &Path, args: &[&str]) -> String {
    let output = Command::new("sudo")
        .arg("-n")
        .arg(helper)
        .args(args)
        .output()
        .expect("run helper");
    assert!(
        output.status.success(),
        "helper {:?} failed: {}",
        args,
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).trim().to_owned()
}

fn mount_visible(mount_point: &Path) -> bool {
    let mounts = std::fs::read_to_string("/proc/mounts").expect("read /proc/mounts");
    let needle = format!(" {}", mount_point.display());
    mounts.lines().any(|line| line.contains(&needle))
}

fn spawn_mount(
    helper: &Path,
    model: &Path,
    mount_point: &Path,
    log_path: &Path,
    format: bool,
) -> Child {
    let stdout = File::create(log_path).expect("create mount log");
    let stderr = stdout.try_clone().expect("clone mount log");

    let mut cmd =
        Command::new(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/debug/llmdb"));
    cmd.env("LLMDB_ROOT_HELPER", helper)
        .arg("mount")
        .arg(model)
        .arg(mount_point);
    if format {
        cmd.arg("--format").arg("--yes");
    }
    cmd.stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .spawn()
        .expect("spawn llmdb mount")
}

fn wait_for_mount(child: &mut Child, mount_point: &Path, log_path: &Path) {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if mount_visible(mount_point) {
            return;
        }
        if let Some(status) = child.try_wait().expect("poll mount child") {
            let log = std::fs::read_to_string(log_path).unwrap_or_default();
            panic!("mount child exited early with {status:?}:\n{log}");
        }
        if Instant::now() >= deadline {
            let log = std::fs::read_to_string(log_path).unwrap_or_default();
            panic!("mount did not appear within timeout:\n{log}");
        }
        thread::sleep(Duration::from_millis(100));
    }
}

fn unmount(helper: &Path, mount_point: &Path, log_path: &Path) {
    let output = Command::new(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/debug/llmdb"))
        .env("LLMDB_ROOT_HELPER", helper)
        .arg("unmount")
        .arg(mount_point)
        .output()
        .expect("run llmdb unmount");
    std::fs::write(log_path, [&output.stdout[..], &output.stderr[..]].concat())
        .expect("write unmount log");
    assert!(
        output.status.success(),
        "llmdb unmount failed:\n{}",
        String::from_utf8_lossy(&[&output.stdout[..], &output.stderr[..]].concat())
    );
}

#[test]
fn mount_unmount_persists_file_via_real_kernel_path() {
    if !gated() {
        eprintln!("LLMDB_E2E_MOUNT!=1; skipping mount_e2e");
        return;
    }
    let helper = helper_path();
    let model = locate_model().expect("no GGUF found for mount_e2e");
    let dir = tempfile::tempdir().expect("tempdir");

    let copied_model = dir.path().join("mount-e2e.gguf");
    std::fs::copy(&model, &copied_model).expect("copy model");

    let mount_point = dir.path().join("mnt");
    let payload_path = dir.path().join("payload.txt");
    let mut payload = File::create(&payload_path).expect("create payload");
    writeln!(payload, "kernel-backed e2e payload").expect("write payload line 1");
    writeln!(payload, "second line").expect("write payload line 2");

    let nbd_device = helper_output(&helper, &["pick-free"]);
    llmdb().arg("init").arg(&copied_model).assert().success();

    let mount1_log = dir.path().join("mount1.log");
    let unmount1_log = dir.path().join("unmount1.log");
    let mount2_log = dir.path().join("mount2.log");
    let unmount2_log = dir.path().join("unmount2.log");

    let child = spawn_mount(&helper, &copied_model, &mount_point, &mount1_log, true);
    struct Cleanup<'a> {
        helper: &'a Path,
        mount_point: &'a Path,
        nbd_device: String,
        child: Option<Child>,
    }
    impl Cleanup<'_> {
        fn disarm(mut self) {
            self.child.take();
        }
    }
    impl Drop for Cleanup<'_> {
        fn drop(&mut self) {
            let _ =
                Command::new(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/debug/llmdb"))
                    .env("LLMDB_ROOT_HELPER", self.helper)
                    .arg("unmount")
                    .arg(self.mount_point)
                    .output();
            let _ = Command::new("sudo")
                .arg("-n")
                .arg(self.helper)
                .arg("detach")
                .arg(&self.nbd_device)
                .output();
            if let Some(mut child) = self.child.take() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
    let cleanup = Cleanup {
        helper: &helper,
        mount_point: &mount_point,
        nbd_device: nbd_device.clone(),
        child: Some(child),
    };
    let mut cleanup = cleanup;

    let child_ref = cleanup.child.as_mut().expect("mount child present");
    wait_for_mount(child_ref, &mount_point, &mount1_log);
    run_helper(
        &helper,
        &[
            "copy-file",
            payload_path.to_str().unwrap(),
            mount_point.join("persist.txt").to_str().unwrap(),
        ],
    );

    unmount(&helper, &mount_point, &unmount1_log);
    let mut child = cleanup.child.take().expect("child for first mount");
    assert!(child.wait().expect("wait first mount").success());
    assert!(
        !mount_visible(&mount_point),
        "mount still visible after first unmount:\n{}",
        std::fs::read_to_string(&unmount1_log).unwrap_or_default()
    );

    cleanup.child = Some(spawn_mount(
        &helper,
        &copied_model,
        &mount_point,
        &mount2_log,
        false,
    ));
    let child_ref = cleanup.child.as_mut().expect("second mount child present");
    wait_for_mount(child_ref, &mount_point, &mount2_log);

    let persisted = std::fs::read(mount_point.join("persist.txt")).expect("read persisted file");
    let original = std::fs::read(&payload_path).expect("read original payload");
    assert_eq!(persisted, original, "persisted file contents changed");

    unmount(&helper, &mount_point, &unmount2_log);
    let mut child = cleanup.child.take().expect("child for second mount");
    assert!(child.wait().expect("wait second mount").success());
    assert!(
        !mount_visible(&mount_point),
        "mount still visible after second unmount:\n{}",
        std::fs::read_to_string(&unmount2_log).unwrap_or_default()
    );
    cleanup.disarm();
}
