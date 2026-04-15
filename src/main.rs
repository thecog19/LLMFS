use std::path::PathBuf;

use clap::{ArgAction, Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(
    name = llmdb::APP_NAME,
    version,
    about = "Steganographic file storage backed by GGUF model weights"
)]
struct Cli {
    #[arg(short, long, global = true, action = ArgAction::SetTrue)]
    verbose: bool,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    Init {
        model: PathBuf,
        #[arg(long, action = ArgAction::SetTrue)]
        lobotomy: bool,
    },
    Status {
        model: PathBuf,
    },
    Store {
        model: PathBuf,
        host_path: PathBuf,
        #[arg(long)]
        name: Option<String>,
        #[arg(long)]
        mode: Option<String>,
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

impl Command {
    fn name(&self) -> &'static str {
        match self {
            Self::Init { .. } => "init",
            Self::Status { .. } => "status",
            Self::Store { .. } => "store",
            Self::Get { .. } => "get",
            Self::Ls { .. } => "ls",
            Self::Rm { .. } => "rm",
            Self::Verify { .. } => "verify",
            Self::Mount { .. } => "mount",
            Self::Unmount { .. } => "unmount",
            Self::Ask { .. } => "ask",
            Self::Dump { .. } => "dump",
            Self::Wipe { .. } => "wipe",
        }
    }
}

fn main() {
    let cli = Cli::parse();

    if cli.verbose {
        eprintln!("[llmdb] verbose mode enabled");
    }

    match cli.command {
        None => println!("llmdb bootstrap: CLI skeleton ready"),
        Some(command) => {
            eprintln!("command '{}' is not implemented yet", command.name());
            std::process::exit(2);
        }
    }
}
