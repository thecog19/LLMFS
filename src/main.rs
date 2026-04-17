use std::error::Error;
use std::fmt;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use clap::{ArgAction, CommandFactory, Parser, Subcommand};

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
    Mount {
        model: PathBuf,
        mount_point: PathBuf,
        #[arg(long, action = ArgAction::SetTrue)]
        format: bool,
    },
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
        Command::Dump { .. } | Command::Mount { .. } | Command::Unmount { .. } | Command::Ask { .. } => {
            Err(CliError::internal(format!(
                "command not implemented yet: {}",
                command_name(&cmd)
            )))
        }
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
