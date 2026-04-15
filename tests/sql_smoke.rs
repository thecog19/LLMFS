mod common;

use common::{SyntheticGgufVersion, SyntheticTensorSpec, write_custom_gguf_fixture};

use rusqlite::params;

use llmdb::gguf::quant::GGML_TYPE_Q8_0_ID;
use llmdb::stego::device::DeviceOptions;
use llmdb::vfs::sqlite_vfs::{initialize_model, open_connection};

#[test]
fn sqlite_vfs_creates_tables_inserts_rows_and_reopens() {
    let fixture = write_custom_gguf_fixture(
        SyntheticGgufVersion::V3,
        "sqlite_smoke.gguf",
        &sqlite_q8_tensors(24),
    );

    initialize_model(&fixture.path, DeviceOptions { verbose: true })
        .expect("initialize sqlite model");

    {
        let connection = open_connection(&fixture.path).expect("open sqlite connection");
        let mode: String = connection
            .query_row("PRAGMA journal_mode;", [], |row| row.get(0))
            .expect("read journal mode");
        assert_eq!(mode.to_lowercase(), "wal");

        connection
            .execute_batch(
                "CREATE TABLE users (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    age INTEGER NOT NULL
                 );
                 CREATE INDEX users_name_idx ON users(name);",
            )
            .expect("create schema");

        for (id, name, age) in [(1_i64, "Ada", 37_i64), (2, "Linus", 55), (3, "Grace", 47)] {
            connection
                .execute(
                    "INSERT INTO users(id, name, age) VALUES (?1, ?2, ?3)",
                    params![id, name, age],
                )
                .expect("insert row");
        }

        let count: i64 = connection
            .query_row("SELECT COUNT(*) FROM users", [], |row| row.get(0))
            .expect("count rows");
        assert_eq!(count, 3);

        let oldest_name: String = connection
            .query_row(
                "SELECT name FROM users ORDER BY age DESC LIMIT 1",
                [],
                |row| row.get(0),
            )
            .expect("query oldest row");
        assert_eq!(oldest_name, "Linus");

        connection
            .execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")
            .expect("checkpoint wal");
    }

    let reopened = open_connection(&fixture.path).expect("reopen sqlite connection");
    let names: Vec<String> = {
        let mut statement = reopened
            .prepare("SELECT name FROM users ORDER BY id")
            .expect("prepare select");
        statement
            .query_map([], |row| row.get(0))
            .expect("query users")
            .map(|row| row.expect("row value"))
            .collect()
    };

    assert_eq!(
        names,
        vec!["Ada".to_string(), "Linus".to_string(), "Grace".to_string()]
    );
}

fn sqlite_q8_tensors(count: usize) -> Vec<SyntheticTensorSpec> {
    (0..count)
        .map(|index| SyntheticTensorSpec {
            name: format!("blk.{index}.ffn_down.weight"),
            dimensions: vec![8_192],
            raw_type_id: GGML_TYPE_Q8_0_ID,
            data: vec![0_u8; (8_192 / 32) * 34],
        })
        .collect()
}
