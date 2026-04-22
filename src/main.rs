use std::error::Error;
use std::fmt;
use std::fs::OpenOptions;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use clap::{ArgAction, CommandFactory, Parser, Subcommand};
use memmap2::MmapMut;

use llmdb::ask::AskError;
use llmdb::ask::bridge::{AskSession, HttpChatClient};
use llmdb::ask::server::LlamaServer;
use llmdb::gguf::parser::parse_path as parse_gguf;
use llmdb::stego::planner::{AllocationMode, build_allocation_plan};
use llmdb::stego::tensor_map::TensorMap;
use llmdb::v2::directory::EntryKind;
use llmdb::v2::fs::{Filesystem as V2Filesystem, FsError as V2FsError};

#[derive(Debug, Parser)]
#[command(
    name = llmdb::APP_NAME,
    version,
    about = "Steganographic file storage backed by GGUF model weights"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Initialise a fresh inode + CoW filesystem on the cover (per
    /// DESIGN-NEW §15). Writes the anchor, an empty root directory,
    /// and an empty dirty bitmap, then writes the cover bytes back.
    /// Re-running on an existing cover discards prior state.
    Init { model: PathBuf },

    /// Print a structured status report: file/dir counts, bytes
    /// stored, allocator usage, dirty-bit balance, dedup table size.
    Status { model: PathBuf },

    /// Mount the cover as a FUSE filesystem. Blocks until Ctrl-C
    /// or until `llmdb unmount <mount_point>` is run in another
    /// shell. Unprivileged — no root or kernel helpers required,
    /// just `fusermount3` (or `fusermount`) on PATH.
    ///
    /// Put the command in the background with `&` / `nohup` /
    /// `disown` if you want cross-shell lifetime; the mount dies
    /// with the process that owns it.
    Mount {
        model: PathBuf,
        mount_point: PathBuf,
        /// Allow users other than the mounter to access the mount.
        /// Requires `user_allow_other` in /etc/fuse.conf.
        #[arg(long, action = ArgAction::SetTrue)]
        allow_other: bool,
    },

    /// Unmount a FUSE-mounted cover. Shells out to `fusermount3 -u`
    /// (falling back to `fusermount -u`).
    Unmount { mount_point: PathBuf },

    /// List entries at an absolute path inside the filesystem
    /// (defaults to `/`). Each line: `<size>\t<name>` for files,
    /// `<dir>/` for directories.
    Ls {
        model: PathBuf,
        #[arg(default_value = "/")]
        path: String,
    },

    /// Copy a host file into the filesystem at `<stego_path>`
    /// (must be absolute, e.g. `/notes.txt`). Creates parent
    /// directories on the fly.
    Store {
        model: PathBuf,
        host_path: PathBuf,
        #[arg(long)]
        stego_path: String,
    },

    /// Copy a file out of the filesystem to the host. `--output`
    /// defaults to the leaf name of `<stego_path>` in the current
    /// working directory.
    Get {
        model: PathBuf,
        stego_path: String,
        #[arg(long)]
        output: Option<PathBuf>,
    },

    /// Remove a file or empty directory at `<stego_path>`.
    Rm {
        model: PathBuf,
        stego_path: String,
        #[arg(long, action = ArgAction::SetTrue)]
        yes: bool,
    },

    /// Spawn `llama-server` against the cover, expose 4 fs tools
    /// (ls / read / stat / list_all_files), and run an interactive
    /// REPL where the model can answer questions about its own
    /// stored files.
    Ask { model: PathBuf },
}

fn main() {
    let cli = Cli::parse();

    let exit_code = match cli.command {
        None => {
            let _ = Cli::command().print_help();
            println!();
            0
        }
        Some(cmd) => match dispatch(cmd) {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("error: {e}");
                e.exit_code()
            }
        },
    };
    std::process::exit(exit_code);
}

fn dispatch(cmd: Command) -> Result<(), CliError> {
    match cmd {
        Command::Init { model } => cmd_init(&model),
        Command::Status { model } => cmd_status(&model),
        Command::Mount {
            model,
            mount_point,
            allow_other,
        } => cmd_mount(&model, &mount_point, allow_other),
        Command::Unmount { mount_point } => cmd_unmount(&mount_point),
        Command::Ls { model, path } => cmd_ls(&model, &path),
        Command::Store {
            model,
            host_path,
            stego_path,
        } => cmd_store(&model, &host_path, &stego_path),
        Command::Get {
            model,
            stego_path,
            output,
        } => cmd_get(&model, &stego_path, output),
        Command::Rm {
            model,
            stego_path,
            yes,
        } => cmd_rm(&model, &stego_path, yes),
        Command::Ask { model } => cmd_ask(&model),
    }
}

// ─── shared open/save helpers ─────────────────────────────────────────────

/// Parse a GGUF and build the V2 tensor map without touching the
/// filesystem state. The returned map is fed to `Filesystem::init`
/// or `Filesystem::mount` together with the cover bytes.
fn build_tensor_map(model: &Path) -> Result<TensorMap, CliError> {
    let parsed = parse_gguf(model)
        .map_err(|e| CliError::user(format!("parse {}: {e}", model.display())))?;
    let plan = build_allocation_plan(&parsed.tensors, AllocationMode::Standard);
    Ok(TensorMap::from_allocation_plan_with_base(
        &plan,
        parsed.tensor_data_offset as u64,
    ))
}

/// Open the cover RW and memory-map it. Mutations to the resulting
/// `MmapMut` go through the page cache; durability requires
/// `flush()` (msync) before drop, which `Filesystem::unmount` handles.
///
/// The file handle is dropped after `map_mut` returns; the mapping
/// keeps the underlying file open at the kernel level.
///
/// SAFETY of `MmapMut::map_mut`: the caller asserts no other process
/// is writing to the same file concurrently. We rely on this — V2
/// has no cross-process locking. Concurrent in-process access is
/// gated by `Arc<RwLock<Filesystem>>` upstream.
fn open_cover_mmap(model: &Path) -> Result<MmapMut, CliError> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(model)
        .map_err(|e| CliError::user(format!("opening {}: {e}", model.display())))?;
    unsafe { MmapMut::map_mut(&file) }
        .map_err(|e| CliError::internal(format!("mmap {}: {e}", model.display())))
}

fn mount_v2(model: &Path) -> Result<V2Filesystem, CliError> {
    let map = build_tensor_map(model)?;
    let cover = open_cover_mmap(model)?;
    V2Filesystem::mount(cover, map).map_err(|e| {
        CliError::user(format!(
            "V2 mount (no anchor? run `llmdb init` first): {e}"
        ))
    })
}

/// Consume `fs` and flush its mmap-backed cover to disk. Wraps the
/// io::Error from msync into a `CliError::internal`.
fn flush_and_close(model: &Path, fs: V2Filesystem) -> Result<(), CliError> {
    let _cover = fs
        .unmount()
        .map_err(|e| CliError::internal(format!("flushing {}: {e}", model.display())))?;
    Ok(())
}

fn fs_err(e: V2FsError) -> CliError {
    use V2FsError::*;
    match e {
        InvalidPath(_)
        | PathNotFound(_)
        | NotADirectory(_)
        | IsADirectory(_)
        | AlreadyExists(_)
        | DirectoryNotEmpty(_)
        | PathCannotBeRoot
        | OutOfSpace { .. }
        | FileTooLarge { .. } => CliError::user(e.to_string()),
        _ => CliError::internal(e.to_string()),
    }
}

// ─── subcommands ──────────────────────────────────────────────────────────

fn cmd_init(model: &Path) -> Result<(), CliError> {
    let map = build_tensor_map(model)?;
    let total_slots = map.slots.len();
    let total_weights: u64 = map.slots.iter().map(|s| s.weight_count).sum();
    let cover = open_cover_mmap(model)?;
    let cover_len = cover.len();

    let fs = V2Filesystem::init(cover, map)
        .map_err(|e| CliError::user(format!("V2 init failed: {e}")))?;
    let generation = fs.generation();
    flush_and_close(model, fs)?;

    println!("initialized {}", model.display());
    println!("  cover size:       {cover_len} bytes");
    println!("  eligible slots:   {total_slots}");
    println!("  eligible weights: {total_weights}");
    println!("  generation:       {generation}");
    Ok(())
}

fn cmd_status(model: &Path) -> Result<(), CliError> {
    let map = build_tensor_map(model)?;
    let cover = open_cover_mmap(model)?;
    let fs = V2Filesystem::mount(cover, map.clone()).map_err(|e| {
        CliError::user(format!(
            "V2 mount (no anchor? run `llmdb init` first): {e}"
        ))
    })?;

    println!("device:             {}", model.display());
    let status = llmdb::diagnostics::gather(&fs, &map)
        .map_err(|e| CliError::user(format!("gather: {e}")))?;
    print!("{}", llmdb::diagnostics::format_human(&status));
    Ok(())
}

fn cmd_mount(model: &Path, mount_point: &Path, allow_other: bool) -> Result<(), CliError> {
    use llmdb::v2::fuse::{LlmdbV2Fs, MountConfig, spawn_background};

    std::fs::create_dir_all(mount_point).map_err(|e| {
        CliError::internal(format!(
            "could not create mount point {}: {e}",
            mount_point.display()
        ))
    })?;

    let fs = mount_v2(model)?;
    let driver = LlmdbV2Fs::new(fs);
    let shared = driver.share();
    let config = MountConfig { allow_other };

    let session = spawn_background(driver, mount_point, &config)
        .map_err(|e| CliError::internal(format!("fuse mount failed: {e}")))?;

    println!("mounted at {}", mount_point.display());
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

    // Recover the filesystem, write the cover bytes back. `try_unwrap`
    // succeeds because the BackgroundSession's Arc was dropped above.
    let lock = std::sync::Arc::try_unwrap(shared).map_err(|_| {
        CliError::internal(
            "could not recover V2 filesystem after unmount — a background handle is still alive"
                .to_owned(),
        )
    })?;
    let fs = lock
        .into_inner()
        .map_err(|_| CliError::internal("V2 filesystem RwLock poisoned (a FUSE op panicked)".to_owned()))?;
    flush_and_close(model, fs)?;
    println!("flushed cover to {}", model.display());
    println!("done");
    Ok(())
}

fn cmd_unmount(mount_point: &Path) -> Result<(), CliError> {
    // Try fusermount3 first (Linux 5.x+); fall back to fusermount
    // (older systems / containers that ship only the v2 binary).
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

fn cmd_ls(model: &Path, path: &str) -> Result<(), CliError> {
    let fs = mount_v2(model)?;
    let entries = fs.readdir(path).map_err(fs_err)?;
    for entry in &entries {
        match entry.kind {
            EntryKind::Directory => println!("{}/", entry.name),
            EntryKind::File => {
                let leaf_path = if path == "/" {
                    format!("/{}", entry.name)
                } else {
                    format!("{}/{}", path.trim_end_matches('/'), entry.name)
                };
                let inode = fs.inode_at(&leaf_path).map_err(fs_err)?;
                println!("{:>10}\t{}", inode.length, entry.name);
            }
        }
    }
    Ok(())
}

fn cmd_store(model: &Path, host_path: &Path, stego_path: &str) -> Result<(), CliError> {
    let bytes = std::fs::read(host_path).map_err(|e| {
        CliError::user(format!("reading {}: {e}", host_path.display()))
    })?;

    let mut fs = mount_v2(model)?;
    ensure_parent_dirs(&mut fs, stego_path)?;
    fs.create_file(stego_path, &bytes).map_err(fs_err)?;
    flush_and_close(model, fs)?;

    println!("stored {} ({} bytes)", stego_path, bytes.len());
    Ok(())
}

fn cmd_get(model: &Path, stego_path: &str, output: Option<PathBuf>) -> Result<(), CliError> {
    let fs = mount_v2(model)?;
    let bytes = fs.read_file(stego_path).map_err(fs_err)?;

    let out_path = output.unwrap_or_else(|| {
        let leaf = stego_path.rsplit('/').next().unwrap_or(stego_path);
        PathBuf::from(leaf)
    });
    std::fs::write(&out_path, &bytes).map_err(|e| {
        CliError::internal(format!("writing {}: {e}", out_path.display()))
    })?;
    println!("wrote {}", out_path.display());
    Ok(())
}

fn cmd_rm(model: &Path, stego_path: &str, yes: bool) -> Result<(), CliError> {
    if !yes && !confirm(&format!("delete {stego_path}?"))? {
        println!("aborted");
        return Ok(());
    }

    let mut fs = mount_v2(model)?;
    // Try unlink first; if the path is a directory, fall back to rmdir.
    match fs.unlink(stego_path) {
        Ok(()) => {}
        Err(V2FsError::IsADirectory(_)) => fs.rmdir(stego_path).map_err(fs_err)?,
        Err(e) => return Err(fs_err(e)),
    }
    flush_and_close(model, fs)?;
    println!("deleted {stego_path}");
    Ok(())
}

fn cmd_ask(model: &Path) -> Result<(), CliError> {
    // Load the cover as a V2 filesystem. The model bytes serve double
    // duty: llama-server reads the file directly from disk, and the
    // V2 fs reads a snapshot into memory for tool-call dispatch. We
    // write any modifications back at session end.
    let mut fs = mount_v2(model)?;

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
    drop(session);
    drop(server);

    flush_and_close(model, fs)?;
    Ok(())
}

// ─── helpers ──────────────────────────────────────────────────────────────

/// Ensure every parent directory of `path` exists. Idempotent on
/// already-present parents.
fn ensure_parent_dirs(fs: &mut V2Filesystem, path: &str) -> Result<(), CliError> {
    let mut accum = String::new();
    let parts: Vec<&str> = path.trim_start_matches('/').split('/').collect();
    if parts.len() <= 1 {
        return Ok(()); // Top-level file; no parents needed.
    }
    for component in &parts[..parts.len() - 1] {
        accum.push('/');
        accum.push_str(component);
        match fs.mkdir(&accum) {
            Ok(()) | Err(V2FsError::AlreadyExists(_)) => {}
            Err(e) => return Err(fs_err(e)),
        }
    }
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
