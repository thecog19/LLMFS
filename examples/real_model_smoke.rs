use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use rusqlite::params;

use llmdb::stego::device::{DeviceOptions, StegoDevice};
use llmdb::stego::planner::AllocationMode;
use llmdb::vfs::sqlite_vfs::{initialize_model, open_connection};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (source_path, verbose) = parse_args()?;
    let exercise_path = exercise_copy_path(&source_path);

    fs::copy(&source_path, &exercise_path)?;
    println!("source model: {}", source_path.display());
    println!("exercise copy: {}", exercise_path.display());

    initialize_model(&exercise_path, DeviceOptions { verbose })?;

    {
        let device = StegoDevice::open_with_options(
            &exercise_path,
            AllocationMode::Standard,
            DeviceOptions { verbose },
        )?;
        println!(
            "formatted device: total_blocks={} integrity_blocks={} data_region_start={} shadow_block={} used_blocks={}",
            device.total_blocks(),
            device.integrity_block_count(),
            device.data_region_start(),
            device.shadow_block(),
            device.used_blocks()?
        );
    }

    {
        let connection = open_connection(&exercise_path)?;
        let journal_mode: String =
            connection.query_row("PRAGMA journal_mode;", [], |row| row.get(0))?;
        println!("sqlite journal_mode: {journal_mode}");

        connection.execute_batch(
            "CREATE TABLE notes (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                body TEXT NOT NULL
             );
             CREATE INDEX notes_title_idx ON notes(title);",
        )?;

        for (id, title, body) in [
            (
                1_i64,
                "first",
                "this row was written into a real gguf-backed sqlite file",
            ),
            (
                2_i64,
                "second",
                "llmdb is exercising its current sqlite vfs slice on a model copy",
            ),
            (
                3_i64,
                "third",
                "the model weights now carry a tiny database payload",
            ),
        ] {
            connection.execute(
                "INSERT INTO notes(id, title, body) VALUES (?1, ?2, ?3)",
                params![id, title, body],
            )?;
        }

        let row_count: i64 =
            connection.query_row("SELECT COUNT(*) FROM notes", [], |row| row.get(0))?;
        let sqlite_integrity: String =
            connection.query_row("PRAGMA integrity_check;", [], |row| row.get(0))?;

        println!("inserted rows: {row_count}");
        println!("sqlite integrity_check: {sqlite_integrity}");

        connection.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")?;
    }

    {
        let reopened = open_connection(&exercise_path)?;
        let titles: Vec<String> = {
            let mut statement = reopened.prepare("SELECT title FROM notes ORDER BY id")?;
            statement
                .query_map([], |row| row.get(0))?
                .collect::<Result<Vec<String>, _>>()?
        };
        println!("reopened titles: {:?}", titles);
    }

    {
        let device = StegoDevice::open_with_options(
            &exercise_path,
            AllocationMode::Standard,
            DeviceOptions { verbose },
        )?;
        println!(
            "used blocks after sqlite workload: {}",
            device.used_blocks()?
        );
    }

    Ok(())
}

fn parse_args() -> Result<(PathBuf, bool), Box<dyn std::error::Error>> {
    let mut verbose = false;
    let mut source = None;

    for arg in env::args_os().skip(1) {
        if arg == "-v" || arg == "--verbose" {
            verbose = true;
        } else if source.is_none() {
            source = Some(PathBuf::from(arg));
        } else {
            return Err(
                "usage: cargo run --example real_model_smoke -- <model.gguf> [--verbose]".into(),
            );
        }
    }

    let source =
        source.ok_or("usage: cargo run --example real_model_smoke -- <model.gguf> [--verbose]")?;
    Ok((source, verbose))
}

fn exercise_copy_path(source: &Path) -> PathBuf {
    let parent = source.parent().unwrap_or_else(|| Path::new("."));
    let stem = source
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("model");
    let extension = source
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or("gguf");
    parent.join(format!("{stem}.exercise.{extension}"))
}
