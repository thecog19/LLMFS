//! `status` reporting for the V2 filesystem: gather a structured
//! view from a `v2::Filesystem` and format it for humans.
//!
//! V2 has no concept of POSIX-mode tiers, lobotomy, or per-write
//! perplexity heuristics — placement is uniformly ceiling-magnitude-
//! ranked across every eligible weight. The status report reflects
//! that: file/dir counts, bytes stored, generation, dedup table size,
//! dirty-bitmap usage, and allocator free-weight balance, plus the
//! distinct quant types the cover contributes.

use std::collections::BTreeSet;
use std::fmt::Write as _;

use crate::gguf::quant::GgufQuantType;
use crate::stego::tensor_map::TensorMap;
use crate::v2::directory::EntryKind;
use crate::v2::fs::{Filesystem, FsError};

#[derive(Debug, Clone)]
pub struct DeviceStatus {
    /// Anchor / super-root generation. Bumps once per commit.
    pub generation: u64,
    /// Number of regular files in the tree (recursive count).
    pub file_count: u32,
    /// Number of directories in the tree (recursive count, excluding `/`).
    pub directory_count: u32,
    /// Sum of file lengths (bytes), recursive.
    pub total_stored_bytes: u64,
    /// Total weights the allocator can place chunks into. Equals the
    /// sum of `weight_count` over slots with stealable bits.
    pub allocator_total_capacity_weights: u64,
    /// Weights currently free (not part of any live chunk).
    pub allocator_free_weights: u64,
    /// Number of unique content-hash entries currently in the dedup
    /// index (one per live file chunk that's been seen).
    pub dedup_entries: u64,
    /// Bits set in the persistent dirty bitmap (= weights ever
    /// written to since init).
    pub dirty_bits_set: u64,
    /// Total bits in the dirty bitmap (= total stealable weights).
    pub dirty_bits_total: u64,
    /// Distinct quant types contributed by stealable slots.
    pub quant_profile: Vec<GgufQuantType>,
}

/// Walk the V2 filesystem + accessors and produce a [`DeviceStatus`].
pub fn gather(fs: &Filesystem, map: &TensorMap) -> Result<DeviceStatus, FsError> {
    let mut file_count = 0_u32;
    let mut directory_count = 0_u32;
    let mut total_stored_bytes = 0_u64;
    walk(fs, "/", &mut |path, kind| -> Result<(), FsError> {
        match kind {
            EntryKind::File => {
                file_count = file_count.saturating_add(1);
                let inode = fs.inode_at(path)?;
                total_stored_bytes = total_stored_bytes.saturating_add(inode.length);
            }
            EntryKind::Directory => {
                directory_count = directory_count.saturating_add(1);
            }
        }
        Ok(())
    })?;

    let dirty = fs.dirty_bitmap();
    let allocator_total_capacity_weights: u64 = map
        .slots
        .iter()
        .filter(|s| s.stealable_bits_per_weight > 0)
        .map(|s| s.weight_count)
        .sum();

    let mut quant_set: BTreeSet<u32> = BTreeSet::new();
    let mut quant_profile: Vec<GgufQuantType> = Vec::new();
    for slot in &map.slots {
        if slot.stealable_bits_per_weight == 0 {
            continue;
        }
        let key = slot.quant_type as u32;
        if quant_set.insert(key) {
            quant_profile.push(slot.quant_type);
        }
    }

    Ok(DeviceStatus {
        generation: fs.generation(),
        file_count,
        directory_count,
        total_stored_bytes,
        allocator_total_capacity_weights,
        allocator_free_weights: fs.allocator_free_weights(),
        dedup_entries: fs.dedup_index().len() as u64,
        dirty_bits_set: dirty.set_count(),
        dirty_bits_total: dirty.total_bits(),
        quant_profile,
    })
}

/// Recursively visit every entry under `dir`. The callback receives
/// the absolute path and kind of each entry.
fn walk<F>(fs: &Filesystem, dir: &str, cb: &mut F) -> Result<(), FsError>
where
    F: FnMut(&str, EntryKind) -> Result<(), FsError>,
{
    for entry in fs.readdir(dir)? {
        let child_path = if dir == "/" {
            format!("/{}", entry.name)
        } else {
            format!("{}/{}", dir.trim_end_matches('/'), entry.name)
        };
        cb(&child_path, entry.kind)?;
        if entry.kind == EntryKind::Directory {
            walk(fs, &child_path, cb)?;
        }
    }
    Ok(())
}

pub fn format_human(status: &DeviceStatus) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "generation:         {}", status.generation);
    let _ = writeln!(out, "files:              {}", status.file_count);
    let _ = writeln!(out, "directories:        {}", status.directory_count);
    let _ = writeln!(out, "stored:             {} bytes", status.total_stored_bytes);

    let dirty_pct = if status.dirty_bits_total == 0 {
        0.0
    } else {
        (status.dirty_bits_set as f64 / status.dirty_bits_total as f64) * 100.0
    };
    let _ = writeln!(
        out,
        "dirty weights:      {} / {} ({:.3}%)",
        status.dirty_bits_set, status.dirty_bits_total, dirty_pct
    );

    let used_weights = status
        .allocator_total_capacity_weights
        .saturating_sub(status.allocator_free_weights);
    let alloc_pct = if status.allocator_total_capacity_weights == 0 {
        0.0
    } else {
        (used_weights as f64 / status.allocator_total_capacity_weights as f64) * 100.0
    };
    let _ = writeln!(
        out,
        "allocator:          {} / {} weights used ({:.3}%)",
        used_weights, status.allocator_total_capacity_weights, alloc_pct
    );

    let _ = writeln!(out, "dedup entries:      {}", status.dedup_entries);

    let quant_str = if status.quant_profile.is_empty() {
        "(none)".to_owned()
    } else {
        status
            .quant_profile
            .iter()
            .map(|q| format!("{q:?}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let _ = writeln!(out, "quant profile:      {quant_str}");
    out
}
