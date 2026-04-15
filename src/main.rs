use std::path::PathBuf;

use clap::{ArgAction, Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(
    name = llmdb::APP_NAME,
    version,
    about = "Stego-backed SQLite experiments on GGUF files"
)]
struct Cli {
    #[arg(short, long, global = true, action = ArgAction::SetTrue)]
    verbose: bool,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    Init { model: PathBuf },
    Status { model: PathBuf },
    Query { model: PathBuf, sql: String },
    Dump { model: PathBuf },
    Load { model: PathBuf },
    Wipe { model: PathBuf },
}

impl Command {
    fn name(&self) -> &'static str {
        match self {
            Self::Init { .. } => "init",
            Self::Status { .. } => "status",
            Self::Query { .. } => "query",
            Self::Dump { .. } => "dump",
            Self::Load { .. } => "load",
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
