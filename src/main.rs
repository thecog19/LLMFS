use std::error::Error;
use std::fmt;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use clap::{ArgAction, CommandFactory, Parser, Subcommand};

use llmdb::ask::AskError;
use llmdb::ask::bridge::{AskSession, HttpChatClient};
use llmdb::ask::server::LlamaServer;
use llmdb::fs::file_ops::FsError;
use llmdb::nbd::server::{NbdError, NbdServer, default_socket_path};
use llmdb::stego::device::{DeviceError, DeviceOptions, StegoDevice};
use llmdb::stego::integrity::decode_quant_profile;
use llmdb::stego::planner::AllocationMode;

#[derive(Debug, Parser)]
#[command(
    name = llmdb::APP_NAME,
    version,
    about = "Steganographic file storage backed by GGUF model weights"
)]
struct Cli {
    #[arg(short, long, global = true, action = ArgAction::SetTrue)]
    verbose: bool,

    /// Open the device in lobotomy mode (includes embedding / norm / output
    /// tensors in the stego plan). Must match the mode used at `init` time.
    #[arg(long, global = true, action = ArgAction::SetTrue)]
    lobotomy: bool,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    Init {
        model: PathBuf,
    },
    Status {
        model: PathBuf,
    },
    Store {
        model: PathBuf,
        host_path: PathBuf,
        #[arg(long)]
        name: Option<String>,
        #[arg(long, default_value = "644")]
        mode: String,
    },
    Get {
        model: PathBuf,
        stego_name: String,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    Ls {
        model: PathBuf,
        #[arg(short = 'l', long = "long", action = ArgAction::SetTrue)]
        long_format: bool,
    },
    Rm {
        model: PathBuf,
        stego_name: String,
        #[arg(long, action = ArgAction::SetTrue)]
        yes: bool,
    },
    Verify {
        model: PathBuf,
    },
    /// Mount the stego device as an ext4 filesystem. Runs the full NBD
    /// stack: serve → nbd-client → (optional mkfs) → mount. Blocks until
    /// Ctrl-C or until `llmdb unmount` is run in another shell. Requires
    /// root (invoke via `sudo`).
    Mount {
        model: PathBuf,
        mount_point: PathBuf,
        /// Run `mkfs.ext4` on the device before mounting. DESTRUCTIVE —
        /// any existing filesystem on the stego device is wiped.
        #[arg(long, action = ArgAction::SetTrue)]
        format: bool,
        /// Skip the mkfs confirmation prompt.
        #[arg(long, action = ArgAction::SetTrue)]
        yes: bool,
        /// Override auto-selected `/dev/nbdN`.
        #[arg(long)]
        nbd: Option<PathBuf>,
        /// Override the default socket path.
        #[arg(long)]
        socket: Option<PathBuf>,
    },
    /// Unmount the ext4 filesystem and tear down the NBD stack. Paired
    /// with `mount`. Requires root.
    Unmount {
        mount_point: PathBuf,
    },
    /// Bind the NBD server on a Unix socket and wait for a client to
    /// connect. Companion command for the manual mount flow —
    /// `nbd-client -unix <socket> /dev/nbdN` connects, then the user
    /// runs `mkfs.ext4` and `mount` in another shell. Exits when the
    /// NBD client disconnects.
    Serve {
        model: PathBuf,
        /// Override the default socket path (`/tmp/llmdb-<pid>.sock`).
        #[arg(long)]
        socket: Option<PathBuf>,
    },
    /// Diagnostic: hexdump a physical block as the stego layer decodes it.
    /// Useful for chasing free-list / redirection corruption.
    DumpBlock {
        model: PathBuf,
        block: u32,
    },
    Ask {
        model: PathBuf,
    },
    Dump {
        model: PathBuf,
    },
    Wipe {
        model: PathBuf,
        #[arg(long, action = ArgAction::SetTrue)]
        yes: bool,
    },
}

fn main() {
    let cli = Cli::parse();
    let verbose = cli.verbose;
    let mode = if cli.lobotomy {
        AllocationMode::Lobotomy
    } else {
        AllocationMode::Standard
    };
    let options = DeviceOptions { verbose };

    let exit_code = match cli.command {
        None => {
            let _ = Cli::command().print_help();
            println!();
            0
        }
        Some(cmd) => match dispatch(cmd, mode, options) {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("error: {e}");
                e.exit_code()
            }
        },
    };
    std::process::exit(exit_code);
}

fn dispatch(cmd: Command, mode: AllocationMode, options: DeviceOptions) -> Result<(), CliError> {
    match cmd {
        Command::Init { model } => cmd_init(&model, mode, options),
        Command::Status { model } => cmd_status(&model, mode, options),
        Command::Store {
            model,
            host_path,
            name,
            mode: perm,
        } => cmd_store(&model, &host_path, name, &perm, mode, options),
        Command::Get {
            model,
            stego_name,
            output,
        } => cmd_get(&model, &stego_name, output, mode, options),
        Command::Ls { model, long_format } => cmd_ls(&model, long_format, mode, options),
        Command::Rm {
            model,
            stego_name,
            yes,
        } => cmd_rm(&model, &stego_name, yes, mode, options),
        Command::Verify { model } => cmd_verify(&model, mode, options),
        Command::Wipe { model, yes } => cmd_wipe(&model, yes, mode, options),
        Command::Serve { model, socket } => cmd_serve(&model, socket, mode, options),
        Command::DumpBlock { model, block } => cmd_dump_block(&model, block, mode, options),
        Command::Mount {
            model,
            mount_point,
            format,
            yes,
            nbd,
            socket,
        } => cmd_mount(&model, &mount_point, format, yes, nbd, socket, mode, options),
        Command::Unmount { mount_point } => cmd_unmount(&mount_point),
        Command::Ask { model } => cmd_ask(&model, mode, options),
        Command::Dump { .. } => Err(CliError::internal(format!(
            "command not implemented yet: {}",
            command_name(&cmd)
        ))),
    }
}

fn command_name(cmd: &Command) -> &'static str {
    match cmd {
        Command::Init { .. } => "init",
        Command::Status { .. } => "status",
        Command::Store { .. } => "store",
        Command::Get { .. } => "get",
        Command::Ls { .. } => "ls",
        Command::Rm { .. } => "rm",
        Command::Verify { .. } => "verify",
        Command::Mount { .. } => "mount",
        Command::Unmount { .. } => "unmount",
        Command::Ask { .. } => "ask",
        Command::Dump { .. } => "dump",
        Command::Wipe { .. } => "wipe",
        Command::Serve { .. } => "serve",
        Command::DumpBlock { .. } => "dump-block",
    }
}

// ─── subcommands ──────────────────────────────────────────────────────────

fn cmd_init(model: &Path, mode: AllocationMode, options: DeviceOptions) -> Result<(), CliError> {
    let device = StegoDevice::initialize_with_options(model, mode, options).map_err(dev_err)?;
    let sb = device.superblock().clone();
    let total = sb.fields.total_blocks;
    let data_start = device.data_region_start();
    let data_blocks = total.saturating_sub(data_start);
    let metadata = data_start;
    let profile = decode_quant_profile(sb.fields.quant_profile);
    let profile_str = if profile.is_empty() {
        "(none)".to_owned()
    } else {
        profile
            .iter()
            .map(|q| format!("{q:?}"))
            .collect::<Vec<_>>()
            .join(", ")
    };

    println!("initialized {}", model.display());
    println!("  total blocks:     {total}");
    println!("  metadata blocks:  {metadata}");
    println!("  data blocks:      {data_blocks}");
    println!("  total capacity:   {} bytes", device.total_capacity_bytes());
    println!("  quant profile:    {profile_str}");
    println!(
        "  lobotomy:         {}",
        if sb.is_lobotomy() { "yes" } else { "no" }
    );

    device.close().map_err(dev_err)?;
    Ok(())
}

fn cmd_status(
    model: &Path,
    mode: AllocationMode,
    options: DeviceOptions,
) -> Result<(), CliError> {
    let device = StegoDevice::open_with_options(model, mode, options).map_err(open_err)?;
    let sb = device.superblock().clone();
    let total = device.total_blocks();
    let free = device.free_blocks().map_err(dev_err)?;
    let used = total.saturating_sub(free);
    let pct = if total == 0 {
        0.0
    } else {
        (used as f64 / total as f64) * 100.0
    };
    let files = device.list_files().map_err(fs_err)?.len();
    let profile = decode_quant_profile(sb.fields.quant_profile);

    println!("device:      {}", model.display());
    println!("total:       {total} blocks");
    println!("used:        {used} blocks");
    println!("free:        {free} blocks");
    println!("utilization: {pct:.1}%");
    println!("files:       {files}");
    println!("quant:       {profile:?}");
    println!(
        "lobotomy:    {}",
        if sb.is_lobotomy() { "yes" } else { "no" }
    );
    println!(
        "dirty:       {}",
        if sb.is_dirty() { "yes" } else { "no" }
    );

    device.close().map_err(dev_err)?;
    Ok(())
}

fn cmd_store(
    model: &Path,
    host_path: &Path,
    name: Option<String>,
    mode_str: &str,
    alloc_mode: AllocationMode,
    options: DeviceOptions,
) -> Result<(), CliError> {
    let stego_name = match name {
        Some(n) => n,
        None => host_path
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or_else(|| CliError::user("could not derive name from host path"))?
            .to_owned(),
    };
    let perm = parse_octal_mode(mode_str)?;

    let mut device =
        StegoDevice::open_with_options(model, alloc_mode, options).map_err(open_err)?;
    let entry = device
        .store_file(host_path, &stego_name, perm)
        .map_err(fs_err)?;

    println!(
        "stored {} ({} bytes, {} block{})",
        entry.filename,
        entry.size_bytes,
        entry.block_count,
        if entry.block_count == 1 { "" } else { "s" }
    );
    device.close().map_err(dev_err)?;
    Ok(())
}

fn cmd_get(
    model: &Path,
    stego_name: &str,
    output: Option<PathBuf>,
    alloc_mode: AllocationMode,
    options: DeviceOptions,
) -> Result<(), CliError> {
    let out_path = output.unwrap_or_else(|| PathBuf::from(stego_name));
    let device = StegoDevice::open_with_options(model, alloc_mode, options).map_err(open_err)?;
    device.get_file(stego_name, &out_path).map_err(fs_err)?;
    println!("wrote {}", out_path.display());
    device.close().map_err(dev_err)?;
    Ok(())
}

fn cmd_ls(
    model: &Path,
    long_format: bool,
    alloc_mode: AllocationMode,
    options: DeviceOptions,
) -> Result<(), CliError> {
    let device = StegoDevice::open_with_options(model, alloc_mode, options).map_err(open_err)?;
    let entries = device.list_files().map_err(fs_err)?;

    if long_format {
        for entry in &entries {
            println!(
                "{:o} {:>10} {:>10} {}",
                entry.mode, entry.size_bytes, entry.modified, entry.filename
            );
        }
    } else {
        for entry in &entries {
            println!("{}", entry.filename);
        }
    }

    device.close().map_err(dev_err)?;
    Ok(())
}

fn cmd_rm(
    model: &Path,
    stego_name: &str,
    yes: bool,
    alloc_mode: AllocationMode,
    options: DeviceOptions,
) -> Result<(), CliError> {
    if !yes && !confirm(&format!("delete {stego_name}?"))? {
        println!("aborted");
        return Ok(());
    }

    let mut device =
        StegoDevice::open_with_options(model, alloc_mode, options).map_err(open_err)?;
    device.delete_file(stego_name).map_err(fs_err)?;
    println!("deleted {stego_name}");
    device.close().map_err(dev_err)?;
    Ok(())
}

fn cmd_verify(
    model: &Path,
    alloc_mode: AllocationMode,
    options: DeviceOptions,
) -> Result<(), CliError> {
    let device = StegoDevice::open_with_options(model, alloc_mode, options).map_err(open_err)?;
    let corrupted = device.verify_integrity().map_err(dev_err)?;

    if corrupted.is_empty() {
        println!("verify: OK ({} live blocks)", device.used_blocks().map_err(dev_err)?);
    } else {
        println!("verify: {} corrupted block(s)", corrupted.len());
        for block in &corrupted {
            println!("  block {block}");
        }
    }

    device.close().map_err(dev_err)?;
    if corrupted.is_empty() {
        Ok(())
    } else {
        Err(CliError::user(format!(
            "{} block(s) failed integrity check",
            corrupted.len()
        )))
    }
}

fn cmd_serve(
    model: &Path,
    socket: Option<PathBuf>,
    alloc_mode: AllocationMode,
    options: DeviceOptions,
) -> Result<(), CliError> {
    let device = StegoDevice::open_with_options(model, alloc_mode, options).map_err(open_err)?;
    let server = NbdServer::new(device);
    let export = server.export_bytes();
    let data_blocks = export / llmdb::BLOCK_SIZE as u64;
    let pid = std::process::id();
    let socket_path = socket.unwrap_or_else(|| default_socket_path(pid));

    println!("llmdb serve: export {} bytes ({} data blocks)", export, data_blocks);
    println!("socket: {}", socket_path.display());
    println!();
    println!("In another shell (root required):");
    println!(
        "  sudo modprobe nbd && sudo nbd-client -unix {} /dev/nbd0",
        socket_path.display()
    );
    println!("  sudo mkfs.ext4 /dev/nbd0          # first mount only");
    println!("  sudo mount /dev/nbd0 /mnt/llmdb");
    println!();
    println!("To stop: sudo umount /mnt/llmdb && sudo nbd-client -d /dev/nbd0");
    println!("waiting for nbd client to connect...");

    server
        .serve_on_unix_socket(&socket_path)
        .map_err(nbd_err)?;
    println!("nbd client disconnected; server exiting");
    Ok(())
}

fn cmd_mount(
    model: &Path,
    mount_point: &Path,
    format: bool,
    yes: bool,
    nbd_override: Option<PathBuf>,
    socket_override: Option<PathBuf>,
    alloc_mode: AllocationMode,
    options: DeviceOptions,
) -> Result<(), CliError> {
    let nbd_dev = match nbd_override {
        Some(p) => p,
        None => find_free_nbd_device()?,
    };
    eprintln!("using {}", nbd_dev.display());

    if format && !yes {
        let prompt = format!(
            "`mkfs.ext4 {}` will wipe any existing filesystem on the stego \
             device. Proceed?",
            nbd_dev.display()
        );
        if !confirm(&prompt)? {
            return Err(CliError::user("aborted"));
        }
    }

    let device = StegoDevice::open_with_options(model, alloc_mode, options).map_err(open_err)?;
    let server = Arc::new(NbdServer::new(device));
    let export = server.export_bytes();
    let sock_path = socket_override.unwrap_or_else(|| default_socket_path(std::process::id()));

    // Spawn server thread before invoking nbd-client.
    let server_thread = {
        let server = Arc::clone(&server);
        let sock = sock_path.clone();
        thread::spawn(move || server.serve_on_unix_socket(&sock))
    };

    // Wait briefly for the socket file to appear.
    for _ in 0..40 {
        if sock_path.exists() {
            break;
        }
        thread::sleep(Duration::from_millis(25));
    }
    if !sock_path.exists() {
        return Err(CliError::internal(format!(
            "nbd socket {} never appeared",
            sock_path.display()
        )));
    }

    println!(
        "llmdb mount: export {} bytes on {}",
        export,
        nbd_dev.display()
    );

    // 1. nbd-client -unix <sock> /dev/nbdN
    run_cmd(&[
        "nbd-client",
        "-unix",
        sock_path.to_str().unwrap(),
        nbd_dev.to_str().unwrap(),
    ])?;

    // 2. optional mkfs.ext4
    if format {
        if let Err(e) = run_cmd(&["mkfs.ext4", "-F", nbd_dev.to_str().unwrap()]) {
            run_cmd(&["nbd-client", "-d", nbd_dev.to_str().unwrap()]).ok();
            return Err(e);
        }
    }

    // 3. mkdir -p mount_point
    if let Err(e) = std::fs::create_dir_all(mount_point) {
        run_cmd(&["nbd-client", "-d", nbd_dev.to_str().unwrap()]).ok();
        return Err(CliError::internal(format!(
            "could not create mount point {}: {e}",
            mount_point.display()
        )));
    }

    // 4. mount /dev/nbdN mount_point
    if let Err(e) = run_cmd(&[
        "mount",
        nbd_dev.to_str().unwrap(),
        mount_point.to_str().unwrap(),
    ]) {
        run_cmd(&["nbd-client", "-d", nbd_dev.to_str().unwrap()]).ok();
        return Err(e);
    }

    // 5. Write the sidecar state file so `llmdb unmount` can find us.
    let sidecar = MountState {
        mount_point: mount_point.to_path_buf(),
        nbd_device: nbd_dev.clone(),
        socket_path: sock_path.clone(),
        mount_pid: std::process::id(),
    };
    if let Err(e) = sidecar.write() {
        eprintln!(
            "warning: could not write mount sidecar ({e}); \
             `llmdb unmount` may not find this session — use Ctrl-C to stop"
        );
    }

    println!("mounted at {}", mount_point.display());
    println!(
        "stop with:  sudo llmdb unmount {}   (or Ctrl-C here)",
        mount_point.display()
    );

    // 6. Install Ctrl-C handler + wait.
    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_cc = Arc::clone(&shutdown);
    ctrlc::set_handler(move || shutdown_cc.store(true, Ordering::SeqCst))
        .map_err(|e| CliError::internal(format!("ctrlc handler install failed: {e}")))?;

    while !shutdown.load(Ordering::SeqCst) {
        if server_thread.is_finished() {
            // Client disconnected (e.g. via `llmdb unmount` running
            // `nbd-client -d`). Flow into the cleanup path below.
            break;
        }
        thread::sleep(Duration::from_millis(200));
    }

    println!("unmounting {} …", mount_point.display());
    // Tear down in the reverse order.
    run_cmd(&["umount", mount_point.to_str().unwrap()]).ok();
    run_cmd(&["nbd-client", "-d", nbd_dev.to_str().unwrap()]).ok();
    let _ = server_thread.join();
    let _ = MountState::remove_for(mount_point);
    println!("done");
    Ok(())
}

fn cmd_unmount(mount_point: &Path) -> Result<(), CliError> {
    let canonical = mount_point
        .canonicalize()
        .unwrap_or_else(|_| mount_point.to_path_buf());
    let state = MountState::find_for(&canonical)
        .or_else(|| MountState::find_for(mount_point))
        .ok_or_else(|| {
            CliError::user(format!(
                "no active llmdb mount found for {}",
                mount_point.display()
            ))
        })?;

    println!(
        "unmounting {} (nbd: {}, mount_pid: {}) …",
        state.mount_point.display(),
        state.nbd_device.display(),
        state.mount_pid
    );
    run_cmd(&["umount", state.mount_point.to_str().unwrap()]).ok();
    run_cmd(&["nbd-client", "-d", state.nbd_device.to_str().unwrap()]).ok();
    // The mount command's server thread should now finish as nbd-client
    // disconnects; the mount process cleans its own sidecar on exit.
    // Give it a moment, then force-remove the sidecar if it lingers.
    thread::sleep(Duration::from_millis(500));
    let _ = state.remove_file();
    println!("done — the `llmdb mount` shell should now return");
    Ok(())
}

fn find_free_nbd_device() -> Result<PathBuf, CliError> {
    for i in 0..16 {
        let dev = PathBuf::from(format!("/dev/nbd{i}"));
        if !dev.exists() {
            continue;
        }
        let pid_file = PathBuf::from(format!("/sys/block/nbd{i}/pid"));
        if !pid_file.exists() {
            return Ok(dev);
        }
        // pid file exists → device is in use by an nbd-client. Check anyway
        // in case the client died without cleanup (empty pid file).
        match std::fs::read_to_string(&pid_file) {
            Ok(s) if s.trim().is_empty() => return Ok(dev),
            _ => continue,
        }
    }
    Err(CliError::user(
        "no free /dev/nbdN device found; run `sudo modprobe nbd` if the \
         module isn't loaded"
            .to_owned(),
    ))
}

fn run_cmd(argv: &[&str]) -> Result<(), CliError> {
    let mut cmd = ProcessCommand::new(argv[0]);
    cmd.args(&argv[1..]);
    let status = cmd
        .status()
        .map_err(|e| CliError::internal(format!("spawn {}: {e}", argv[0])))?;
    if !status.success() {
        return Err(CliError::user(format!(
            "{} exited with status {:?}",
            argv[0],
            status.code()
        )));
    }
    Ok(())
}

// -- Mount sidecar state (for `unmount` to find a running mount) --

#[derive(Debug)]
struct MountState {
    mount_point: PathBuf,
    nbd_device: PathBuf,
    socket_path: PathBuf,
    mount_pid: u32,
}

const SIDECAR_DIR: &str = "/tmp/llmdb-mounts";

impl MountState {
    fn sidecar_path(&self) -> PathBuf {
        Self::path_for(&self.mount_point)
    }

    fn path_for(mount_point: &Path) -> PathBuf {
        // Flatten the mount point path into a filename: /mnt/foo → mnt-foo.
        let flat: String = mount_point
            .to_string_lossy()
            .trim_start_matches('/')
            .replace('/', "-");
        PathBuf::from(SIDECAR_DIR).join(format!("{flat}.state"))
    }

    fn write(&self) -> io::Result<()> {
        std::fs::create_dir_all(SIDECAR_DIR)?;
        let body = format!(
            "mount_point={}\nnbd_device={}\nsocket_path={}\nmount_pid={}\n",
            self.mount_point.display(),
            self.nbd_device.display(),
            self.socket_path.display(),
            self.mount_pid
        );
        std::fs::write(self.sidecar_path(), body)
    }

    fn find_for(mount_point: &Path) -> Option<Self> {
        let path = Self::path_for(mount_point);
        if !path.exists() {
            // Fall back to scan — in case the flattening differs.
            return Self::scan_for(mount_point);
        }
        Self::parse(&std::fs::read_to_string(&path).ok()?)
    }

    fn scan_for(mount_point: &Path) -> Option<Self> {
        let entries = std::fs::read_dir(SIDECAR_DIR).ok()?;
        for e in entries.flatten() {
            let body = std::fs::read_to_string(e.path()).ok()?;
            let state = Self::parse(&body)?;
            if state.mount_point == mount_point {
                return Some(state);
            }
        }
        None
    }

    fn parse(body: &str) -> Option<Self> {
        let mut mp = None;
        let mut nd = None;
        let mut sp = None;
        let mut pid = None;
        for line in body.lines() {
            let (k, v) = line.split_once('=')?;
            match k {
                "mount_point" => mp = Some(PathBuf::from(v)),
                "nbd_device" => nd = Some(PathBuf::from(v)),
                "socket_path" => sp = Some(PathBuf::from(v)),
                "mount_pid" => pid = v.parse().ok(),
                _ => {}
            }
        }
        Some(Self {
            mount_point: mp?,
            nbd_device: nd?,
            socket_path: sp?,
            mount_pid: pid?,
        })
    }

    fn remove_for(mount_point: &Path) -> io::Result<()> {
        let path = Self::path_for(mount_point);
        if path.exists() {
            std::fs::remove_file(path)
        } else {
            Ok(())
        }
    }

    fn remove_file(&self) -> io::Result<()> {
        let path = self.sidecar_path();
        if path.exists() {
            std::fs::remove_file(path)
        } else {
            Ok(())
        }
    }
}

fn cmd_ask(
    model: &Path,
    alloc_mode: AllocationMode,
    options: DeviceOptions,
) -> Result<(), CliError> {
    // We need the device for tool-call dispatch AND the model path for
    // llama-server. The device opens normally, then we spawn the server
    // on a free port.
    let mut device = StegoDevice::open_with_options(model, alloc_mode, options).map_err(open_err)?;

    let port = pick_free_port()?;
    println!("spawning llama-server on port {port} …");
    let server = LlamaServer::spawn(model, port).map_err(ask_err)?;
    println!("llama-server ready at {}", server.base_url());

    let client = HttpChatClient::new(server.base_url());
    let mut session = AskSession::new(client, &mut device, "llmdb-ask");

    println!("\n`ask` session ready. Type a question, or Ctrl-D / `exit` to quit.\n");

    let stdin = io::stdin();
    let mut line = String::new();
    loop {
        print!("> ");
        io::stdout().flush().ok();
        line.clear();
        match stdin.read_line(&mut line) {
            Ok(0) => break, // Ctrl-D
            Ok(_) => {}
            Err(e) => return Err(CliError::internal(format!("read_line: {e}"))),
        }
        let q = line.trim();
        if q.is_empty() {
            continue;
        }
        if matches!(q, "exit" | "quit") {
            break;
        }
        match session.ask(q) {
            Ok(answer) => println!("{answer}\n"),
            Err(e) => {
                eprintln!("ask error: {e}");
                // non-fatal — next prompt
            }
        }
    }

    println!("goodbye");
    // `server`'s Drop kills the subprocess.
    drop(session);
    drop(server);
    Ok(())
}

fn pick_free_port() -> Result<u16, CliError> {
    use std::net::TcpListener;
    let listener = TcpListener::bind("127.0.0.1:0")
        .map_err(|e| CliError::internal(format!("bind(0): {e}")))?;
    let port = listener
        .local_addr()
        .map_err(|e| CliError::internal(format!("local_addr: {e}")))?
        .port();
    drop(listener);
    Ok(port)
}

fn ask_err(e: AskError) -> CliError {
    match e {
        AskError::SpawnFailed(_)
        | AskError::HealthTimeout { .. }
        | AskError::InvalidToolCall(_)
        | AskError::ToolCallLimitExceeded { .. } => CliError::user(e.to_string()),
        _ => CliError::internal(e.to_string()),
    }
}

fn cmd_dump_block(
    model: &Path,
    block: u32,
    alloc_mode: AllocationMode,
    options: DeviceOptions,
) -> Result<(), CliError> {
    let device = StegoDevice::open_with_options(model, alloc_mode, options).map_err(open_err)?;
    let bytes = device.read_physical_block_for_diag(block).map_err(dev_err)?;
    let written = device.is_logical_written(block);
    println!(
        "physical block {block} (logical written? {written}); first 256 bytes:"
    );
    for (i, chunk) in bytes.chunks(16).take(16).enumerate() {
        let hex: Vec<String> = chunk.iter().map(|b| format!("{b:02x}")).collect();
        println!("  {:04x}: {}", i * 16, hex.join(" "));
    }
    let nonzero = bytes.iter().filter(|&&b| b != 0).count();
    println!("(total nonzero bytes in this block: {nonzero}/4096)");
    Ok(())
}

fn cmd_wipe(
    model: &Path,
    yes: bool,
    alloc_mode: AllocationMode,
    options: DeviceOptions,
) -> Result<(), CliError> {
    if !yes
        && !confirm(&format!(
            "WIPE all stego data in {}? This destroys every stored file.",
            model.display()
        ))?
    {
        println!("aborted");
        return Ok(());
    }

    let mut device =
        StegoDevice::open_with_options(model, alloc_mode, options).map_err(open_err)?;
    device.wipe().map_err(dev_err)?;
    println!("wiped {} (device re-initialized, 0 files)", model.display());
    device.close().map_err(dev_err)?;
    Ok(())
}

// ─── helpers ──────────────────────────────────────────────────────────────

fn parse_octal_mode(raw: &str) -> Result<u16, CliError> {
    let trimmed = raw.trim_start_matches("0o").trim_start_matches('0');
    let source = if trimmed.is_empty() { "0" } else { trimmed };
    u16::from_str_radix(source, 8).map_err(|_| CliError::user(format!("invalid octal mode: {raw}")))
}

fn confirm(prompt: &str) -> Result<bool, CliError> {
    print!("{prompt} [y/N]: ");
    io::stdout().flush().map_err(|e| CliError::internal(e.to_string()))?;
    let mut line = String::new();
    io::stdin()
        .read_line(&mut line)
        .map_err(|e| CliError::internal(e.to_string()))?;
    Ok(matches!(line.trim(), "y" | "Y" | "yes" | "YES"))
}

fn dev_err(e: DeviceError) -> CliError {
    match e {
        DeviceError::OutOfSpace => CliError::user(e.to_string()),
        DeviceError::BlockOutOfRange { .. }
        | DeviceError::ReservedMetadataBlock { .. }
        | DeviceError::InvalidBlockWriteLength { .. } => CliError::user(e.to_string()),
        other => CliError::internal(other.to_string()),
    }
}

fn fs_err(e: FsError) -> CliError {
    match e {
        FsError::FileNotFound(_)
        | FsError::DuplicateName(_)
        | FsError::InvalidFilename { .. }
        | FsError::FileTooLarge { .. }
        | FsError::TableFull { .. }
        | FsError::Crc32Mismatch { .. } => CliError::user(e.to_string()),
        other => CliError::internal(other.to_string()),
    }
}

fn open_err(e: DeviceError) -> CliError {
    // Present a clearer message for "device not initialized" errors so the
    // user knows to run `init` first.
    match e {
        DeviceError::Integrity(inner) => {
            CliError::user(format!("device not initialized or corrupt: {inner}"))
        }
        other => dev_err(other),
    }
}

fn nbd_err(e: NbdError) -> CliError {
    match e {
        NbdError::OutOfRange { .. } => CliError::user(e.to_string()),
        NbdError::Io(inner) => CliError::internal(inner.to_string()),
        NbdError::Device(inner) => dev_err(inner),
        NbdError::Protocol(inner) => CliError::internal(inner.to_string()),
        NbdError::LockPoisoned => CliError::internal("nbd device lock poisoned".to_owned()),
    }
}

// ─── CliError ─────────────────────────────────────────────────────────────

#[derive(Debug)]
enum CliError {
    User(String),
    Internal(String),
}

impl CliError {
    fn user(msg: impl Into<String>) -> Self {
        Self::User(msg.into())
    }
    fn internal(msg: impl Into<String>) -> Self {
        Self::Internal(msg.into())
    }
    fn exit_code(&self) -> i32 {
        match self {
            Self::User(_) => 1,
            Self::Internal(_) => 2,
        }
    }
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::User(m) | Self::Internal(m) => f.write_str(m),
        }
    }
}

impl Error for CliError {}
