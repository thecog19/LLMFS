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
    /// Mount the stego device as a FUSE filesystem. Blocks until Ctrl-C
    /// or until `llmdb unmount <mount_point>` is run in another shell.
    /// Unprivileged — no root or kernel helpers required, just
    /// `fusermount3` (or `fusermount`) on PATH.
    ///
    /// Put the command in the background with `&` / `nohup` / `disown`
    /// if you want cross-shell lifetime; the mount dies with the process
    /// that owns it.
    Mount {
        model: PathBuf,
        mount_point: PathBuf,
        /// Allow users other than the mounter to access the mount.
        /// Requires `user_allow_other` in /etc/fuse.conf.
        #[arg(long, action = ArgAction::SetTrue)]
        allow_other: bool,
    },
    /// Unmount a FUSE-mounted stego device. Shells out to
    /// `fusermount3 -u` (falling back to `fusermount -u`).
    Unmount {
        mount_point: PathBuf,
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

    /// V2: initialise a fresh inode + CoW filesystem on the cover
    /// (per DESIGN-NEW §15). Writes the anchor, an empty root
    /// directory, and an empty dirty bitmap, then writes the cover
    /// bytes back. Global `--lobotomy` is ignored — V2 placement is
    /// driven by ceiling magnitude over every eligible weight.
    V2Init {
        model: PathBuf,
    },

    /// V2: mount a V2-initialised cover as a FUSE filesystem. Loads
    /// the cover into memory, serves kernel ops through
    /// `v2::Filesystem`, and writes the cover back on unmount.
    V2Mount {
        model: PathBuf,
        mount_point: PathBuf,
        /// Allow users other than the mounter to access the mount.
        /// Requires `user_allow_other` in /etc/fuse.conf.
        #[arg(long, action = ArgAction::SetTrue)]
        allow_other: bool,
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
        Command::DumpBlock { model, block } => cmd_dump_block(&model, block, mode, options),
        Command::Mount {
            model,
            mount_point,
            allow_other,
        } => cmd_mount(&model, &mount_point, allow_other, mode, options),
        Command::Unmount { mount_point } => cmd_unmount(&mount_point),
        Command::Ask { model } => cmd_ask(&model),
        Command::Dump { model } => cmd_dump(&model, mode, options),
        Command::V2Init { model } => cmd_v2_init(&model),
        Command::V2Mount {
            model,
            mount_point,
            allow_other,
        } => cmd_v2_mount(&model, &mount_point, allow_other),
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
    println!(
        "  total capacity:   {} bytes",
        device.total_capacity_bytes()
    );
    println!("  quant profile:    {profile_str}");
    println!(
        "  lobotomy:         {}",
        if sb.is_lobotomy() { "yes" } else { "no" }
    );

    device.close().map_err(dev_err)?;
    Ok(())
}

fn cmd_status(model: &Path, mode: AllocationMode, options: DeviceOptions) -> Result<(), CliError> {
    let device = StegoDevice::open_with_options(model, mode, options).map_err(open_err)?;
    println!("device:      {}", model.display());
    let status = llmdb::diagnostics::gather(&device).map_err(fs_err)?;
    print!("{}", llmdb::diagnostics::format_human(&status));
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
        println!(
            "verify: OK ({} live blocks)",
            device.used_blocks().map_err(dev_err)?
        );
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

fn cmd_mount(
    model: &Path,
    mount_point: &Path,
    allow_other: bool,
    alloc_mode: AllocationMode,
    options: DeviceOptions,
) -> Result<(), CliError> {
    use llmdb::fuse::{LlmdbFs, MountConfig, spawn_background};

    std::fs::create_dir_all(mount_point).map_err(|e| {
        CliError::internal(format!(
            "could not create mount point {}: {e}",
            mount_point.display()
        ))
    })?;

    let device = StegoDevice::open_with_options(model, alloc_mode, options).map_err(open_err)?;
    let fs = LlmdbFs::new(device);
    let config = MountConfig { allow_other };

    let session = spawn_background(fs, mount_point, &config)
        .map_err(|e| CliError::internal(format!("fuse mount failed: {e}")))?;

    println!("mounted at {}", mount_point.display());
    println!(
        "stop with: llmdb unmount {}   (or Ctrl-C here)",
        mount_point.display()
    );

    // Exit on either Ctrl-C here or an external unmount (fusermount3 -u
    // from `llmdb unmount` in another shell). An external unmount ends
    // the kernel-side FUSE session; fuser notices and the session's
    // background thread finishes. Poll both signals.
    let shutdown = Arc::new(AtomicBool::new(false));
    let sc = Arc::clone(&shutdown);
    ctrlc::set_handler(move || sc.store(true, Ordering::SeqCst))
        .map_err(|e| CliError::internal(format!("ctrlc install failed: {e}")))?;
    loop {
        if shutdown.load(Ordering::SeqCst) {
            println!("unmounting {} …", mount_point.display());
            drop(session); // triggers fusermount3 -u
            break;
        }
        if session.guard.is_finished() {
            println!("mount released externally");
            drop(session);
            break;
        }
        thread::sleep(Duration::from_millis(200));
    }
    println!("done");
    Ok(())
}

fn cmd_unmount(mount_point: &Path) -> Result<(), CliError> {
    // Try fusermount3 first (Linux 5.x+); fall back to fusermount (older
    // systems / containers that ship only the v2 binary).
    for prog in ["fusermount3", "fusermount"] {
        let out = ProcessCommand::new(prog)
            .arg("-u")
            .arg(mount_point)
            .output();
        match out {
            Ok(o) if o.status.success() => {
                println!("unmounted {}", mount_point.display());
                return Ok(());
            }
            Ok(o) => {
                return Err(CliError::user(format!(
                    "{prog} -u {} failed: {}",
                    mount_point.display(),
                    String::from_utf8_lossy(&o.stderr).trim()
                )));
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                continue;
            }
            Err(e) => {
                return Err(CliError::internal(format!("spawning {prog}: {e}")));
            }
        }
    }
    Err(CliError::user(
        "neither fusermount3 nor fusermount is on PATH — install fuse3 or fuse".to_owned(),
    ))
}

fn cmd_ask(model: &Path) -> Result<(), CliError> {
    use llmdb::gguf::parser::parse_path as parse_gguf;
    use llmdb::stego::planner::build_allocation_plan;
    use llmdb::stego::tensor_map::TensorMap;
    use llmdb::v2::fs::Filesystem as V2Filesystem;

    // Load the cover as a V2 filesystem. The model bytes serve double
    // duty: llama-server reads the file directly from disk, and the
    // V2 fs reads a snapshot into memory for tool-call dispatch. We
    // write any modifications back at session end.
    let parsed = parse_gguf(model)
        .map_err(|e| CliError::user(format!("parse {}: {e}", model.display())))?;
    let plan = build_allocation_plan(&parsed.tensors, AllocationMode::Standard);
    let map = TensorMap::from_allocation_plan_with_base(&plan, parsed.tensor_data_offset as u64);
    let cover = std::fs::read(model)
        .map_err(|e| CliError::internal(format!("reading {}: {e}", model.display())))?;
    let mut fs = V2Filesystem::mount(cover, map).map_err(|e| {
        CliError::user(format!(
            "V2 mount (no anchor? run `llmdb v2-init` first): {e}"
        ))
    })?;

    let port = pick_free_port()?;
    println!("spawning llama-server on port {port} …");
    let server = LlamaServer::spawn(model, port).map_err(ask_err)?;
    println!("llama-server ready at {}", server.base_url());

    let client = HttpChatClient::new(server.base_url());
    let mut session = AskSession::new(client, &mut fs, "llmdb-ask");

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

    // Persist any state changes the model drove via tool calls. The
    // current 4-tool set is read-only, but `unmount` is still the
    // canonical way to close out the cover mmap → cover-bytes path.
    let cover_after = fs.unmount();
    std::fs::write(model, &cover_after).map_err(|e| {
        CliError::internal(format!("writing {}: {e}", model.display()))
    })?;
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
    let bytes = device
        .read_physical_block_for_diag(block)
        .map_err(dev_err)?;
    let written = device.is_logical_written(block);
    println!("physical block {block} (logical written? {written}); first 256 bytes:");
    for (i, chunk) in bytes.chunks(16).take(16).enumerate() {
        let hex: Vec<String> = chunk.iter().map(|b| format!("{b:02x}")).collect();
        println!("  {:04x}: {}", i * 16, hex.join(" "));
    }
    let nonzero = bytes.iter().filter(|&&b| b != 0).count();
    println!("(total nonzero bytes in this block: {nonzero}/4096)");
    Ok(())
}

fn cmd_dump(
    model: &Path,
    alloc_mode: AllocationMode,
    options: DeviceOptions,
) -> Result<(), CliError> {
    let device = StegoDevice::open_with_options(model, alloc_mode, options).map_err(open_err)?;
    let entries = device.list_files().map_err(fs_err)?;

    let stdout = io::stdout();
    let mut guard = stdout.lock();
    let mut writer = llmdb::fs::tar::TarWriter::new(&mut guard);
    for entry in entries {
        let bytes = device.read_file_bytes(&entry.filename).map_err(fs_err)?;
        writer
            .append(&entry.filename, entry.mode, entry.modified, &bytes)
            .map_err(|e| CliError::internal(format!("tar write: {e}")))?;
    }
    writer
        .finish()
        .map_err(|e| CliError::internal(format!("tar finalize: {e}")))?;
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

// ─── V2 (DESIGN-NEW §15) ──────────────────────────────────────────────────

fn cmd_v2_init(model: &Path) -> Result<(), CliError> {
    use llmdb::gguf::parser::parse_path as parse_gguf;
    use llmdb::stego::planner::build_allocation_plan;
    use llmdb::stego::tensor_map::TensorMap;
    use llmdb::v2::fs::Filesystem as V2Filesystem;

    let parsed = parse_gguf(model)
        .map_err(|e| CliError::user(format!("parse {}: {e}", model.display())))?;
    let plan = build_allocation_plan(&parsed.tensors, AllocationMode::Standard);
    let map =
        TensorMap::from_allocation_plan_with_base(&plan, parsed.tensor_data_offset as u64);

    let cover = std::fs::read(model).map_err(|e| {
        CliError::internal(format!("reading {}: {e}", model.display()))
    })?;

    let total_slots = map.slots.len();
    let total_weights: u64 = map.slots.iter().map(|s| s.weight_count).sum();
    let cover_len = cover.len();

    let fs = V2Filesystem::init(cover, map)
        .map_err(|e| CliError::user(format!("V2 init failed: {e}")))?;
    let generation = fs.generation();
    let cover_after = fs.unmount();

    std::fs::write(model, &cover_after).map_err(|e| {
        CliError::internal(format!("writing {}: {e}", model.display()))
    })?;

    println!("initialized (V2) {}", model.display());
    println!("  cover size:       {cover_len} bytes");
    println!("  eligible slots:   {total_slots}");
    println!("  eligible weights: {total_weights}");
    println!("  generation:       {generation}");
    Ok(())
}

fn cmd_v2_mount(
    model: &Path,
    mount_point: &Path,
    allow_other: bool,
) -> Result<(), CliError> {
    use llmdb::gguf::parser::parse_path as parse_gguf;
    use llmdb::stego::planner::build_allocation_plan;
    use llmdb::stego::tensor_map::TensorMap;
    use llmdb::v2::fs::Filesystem as V2Filesystem;
    use llmdb::v2::fuse::{LlmdbV2Fs, MountConfig, spawn_background};

    std::fs::create_dir_all(mount_point).map_err(|e| {
        CliError::internal(format!(
            "could not create mount point {}: {e}",
            mount_point.display()
        ))
    })?;

    let parsed = parse_gguf(model)
        .map_err(|e| CliError::user(format!("parse {}: {e}", model.display())))?;
    let plan = build_allocation_plan(&parsed.tensors, AllocationMode::Standard);
    let map =
        TensorMap::from_allocation_plan_with_base(&plan, parsed.tensor_data_offset as u64);
    let cover = std::fs::read(model).map_err(|e| {
        CliError::internal(format!("reading {}: {e}", model.display()))
    })?;

    let fs = V2Filesystem::mount(cover, map).map_err(|e| {
        CliError::user(format!("V2 mount (no anchor? run `llmdb v2-init` first): {e}"))
    })?;
    let driver = LlmdbV2Fs::new(fs);
    let shared = driver.share();
    let config = MountConfig { allow_other };

    let session = spawn_background(driver, mount_point, &config)
        .map_err(|e| CliError::internal(format!("fuse mount failed: {e}")))?;

    println!("mounted V2 filesystem at {}", mount_point.display());
    println!(
        "stop with: llmdb unmount {}   (or Ctrl-C here)",
        mount_point.display()
    );

    let shutdown = Arc::new(AtomicBool::new(false));
    let sc = Arc::clone(&shutdown);
    ctrlc::set_handler(move || sc.store(true, Ordering::SeqCst))
        .map_err(|e| CliError::internal(format!("ctrlc install failed: {e}")))?;
    loop {
        if shutdown.load(Ordering::SeqCst) {
            println!("unmounting {} …", mount_point.display());
            drop(session);
            break;
        }
        if session.guard.is_finished() {
            println!("mount released externally");
            drop(session);
            break;
        }
        thread::sleep(Duration::from_millis(200));
    }

    // Recover the filesystem, pull the cover bytes out, write them
    // back to the GGUF. `Arc::try_unwrap` should succeed because the
    // BackgroundSession's Arc was dropped with the driver above.
    let mutex = std::sync::Arc::try_unwrap(shared).map_err(|_| {
        CliError::internal(
            "could not recover V2 filesystem after unmount — a background handle is still alive"
                .to_owned(),
        )
    })?;
    let fs = mutex.into_inner().map_err(|_| {
        CliError::internal("V2 filesystem mutex poisoned (a FUSE op panicked)".to_owned())
    })?;
    let cover_after = fs.unmount();
    std::fs::write(model, &cover_after).map_err(|e| {
        CliError::internal(format!("writing {}: {e}", model.display()))
    })?;
    println!("wrote {} bytes back to {}", cover_after.len(), model.display());
    println!("done");
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
    io::stdout()
        .flush()
        .map_err(|e| CliError::internal(e.to_string()))?;
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
