//! V2 anchor — the rule-derived bootstrap region (DESIGN-NEW §15.2).
//!
//! An anchor is a small fixed-size record (64 bytes) that lives at
//! the N lowest-ceiling-magnitude stealable-bit positions of a
//! cover, sorted by `WeightRef`. The anchor holds magic + version +
//! two generational "root slots," each of which encodes a generation
//! counter + a `Pointer` to a super-root inode + a CRC32.
//!
//! Mount picks the slot with the higher valid generation; commits
//! always write to the inactive slot, so readers only ever see a
//! committed state. The slot alternation + generation counter is V2's
//! atomic-commit primitive.
//!
//! Tests cover:
//! 1. **Placement determinism.** `find_anchor_placement(cover, map)`
//!    returns the same positions on repeated calls, and the positions
//!    are chosen by ceiling magnitude + `WeightRef` ordering.
//! 2. **Round-trip.** `init_anchor` then `read_anchor` returns the
//!    super-root pointer written at init.
//! 3. **Commit alternation.** Successive commits alternate between
//!    the two slots; reader always sees the latest generation.
//! 4. **Corrupt-slot fallback.** If one slot has a bad CRC, the other
//!    (valid) slot is used.
//! 5. **Both-slots-corrupt error.** Neither slot valid → explicit
//!    error on read.
//! 6. **Bad magic.** A cover with no V2 anchor → explicit error.

use llmdb::gguf::quant::GgufQuantType;
use llmdb::stego::calibration::bit_io::write_bit;
use llmdb::stego::calibration::magnitude::read_weight_ceiling_abs;
use llmdb::stego::planner::TensorTier;
use llmdb::stego::tensor_map::{TensorMap, TensorSlot};
use llmdb::v2::anchor::{
    ANCHOR_BITS, AnchorError, SlotIndex, commit_anchor, find_anchor_placement,
    init_anchor, read_anchor,
};
use llmdb::v2::pointer::Pointer;

// ------------------------------------------------------------------
// Fixtures
// ------------------------------------------------------------------

fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 31) & 0x1) as u16;
    let exp32 = ((bits >> 23) & 0xFF) as i32;
    let mantissa32 = bits & 0x7FFFFF;
    if exp32 == 0 {
        return sign << 15;
    }
    let exp16 = exp32 - 127 + 15;
    if exp16 <= 0 {
        return sign << 15;
    }
    if exp16 >= 31 {
        return (sign << 15) | (0x1F << 10);
    }
    let mantissa16 = (mantissa32 >> 13) as u16;
    (sign << 15) | ((exp16 as u16) << 10) | mantissa16
}

fn f16_slot(weight_count: u64, data_offset: u64) -> TensorSlot {
    let bits = GgufQuantType::F16.stealable_bits_hint() as u64;
    TensorSlot {
        name: format!("anchor.f16.{weight_count}"),
        quant_type: GgufQuantType::F16,
        tier: TensorTier::Tier1,
        data_offset,
        weight_count,
        stealable_bits_per_weight: bits as usize,
        capacity_bits: weight_count * bits,
        bit_start: 0,
        bit_end: weight_count * bits,
    }
}

/// Big-enough cover for the anchor: 512 bits / 4 bits-per-F16 = 128
/// weights, plus slack. Use 512 weights for a meaningful ceiling-
/// magnitude range so bottom-N ranking has real work to do.
fn synthesize_cover() -> (Vec<u8>, TensorMap) {
    // Log-uniform magnitudes so the bottom-N is well-separated from
    // the rest; helps ceiling-magnitude ranking be stable across
    // minor test-rewrite perturbations.
    let values: Vec<f32> = (0..512)
        .map(|i| {
            // spread magnitudes from ~1e-4 to ~1.0
            let sign = if i % 3 == 0 { -1.0 } else { 1.0 };
            sign * (i as f32 + 1.0) * 0.0001
        })
        .collect();
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for v in &values {
        bytes.extend_from_slice(&f32_to_f16_bits(*v).to_le_bytes());
    }
    let slot = f16_slot(values.len() as u64, 0);
    let map = TensorMap {
        slots: vec![slot.clone()],
        total_capacity_bits: slot.capacity_bits,
        total_capacity_bytes: slot.capacity_bits / 8,
    };
    (bytes, map)
}

fn sample_pointer(seed: u64) -> Pointer {
    Pointer {
        slot: (seed & 0xFFFF) as u16,
        start_weight: ((seed >> 16) & 0xFFFF_FFFF) as u32,
        length_in_bits: 32768,
        flags: 0,
        reserved: 0,
    }
}

// ------------------------------------------------------------------
// Placement
// ------------------------------------------------------------------

#[test]
fn placement_covers_at_least_anchor_bits() {
    let (bytes, map) = synthesize_cover();
    let placement = find_anchor_placement(&bytes, &map);
    assert!(
        placement.positions.len() as u64 >= ANCHOR_BITS,
        "placement should cover at least {} bits, got {}",
        ANCHOR_BITS,
        placement.positions.len(),
    );
}

#[test]
fn placement_is_deterministic() {
    let (bytes, map) = synthesize_cover();
    let a = find_anchor_placement(&bytes, &map);
    let b = find_anchor_placement(&bytes, &map);
    let c = find_anchor_placement(&bytes, &map);
    assert_eq!(a, b);
    assert_eq!(b, c);
}

#[test]
fn placement_picks_lowest_ceiling_weights() {
    // Every weight in the selection should have ceiling <= every
    // weight outside it (with WeightRef tiebreak for ties).
    let (bytes, map) = synthesize_cover();
    let placement = find_anchor_placement(&bytes, &map);
    let selected: std::collections::HashSet<(u32, u64)> = placement
        .positions
        .iter()
        .map(|p| (p.slot_index, p.weight_index))
        .collect();
    let slot = &map.slots[0];
    let cutoff = selected
        .iter()
        .map(|(_, w)| read_weight_ceiling_abs(&bytes, slot, *w))
        .fold(0.0_f32, f32::max);
    for w in 0..slot.weight_count {
        if selected.contains(&(0, w)) {
            continue;
        }
        let c = read_weight_ceiling_abs(&bytes, slot, w);
        assert!(
            c >= cutoff - cutoff * 1e-6 - 1e-12,
            "weight {w} (ceiling {c}) excluded but is smaller than selected cutoff {cutoff}",
        );
    }
}

#[test]
fn placement_is_weightref_sorted() {
    let (bytes, map) = synthesize_cover();
    let placement = find_anchor_placement(&bytes, &map);
    // positions are generated per-weight in weight order; within each
    // weight, bit indices 0..stealable go in order. So globally the
    // sequence is ascending by (slot_index, weight_index, bit_index).
    for w in placement.positions.windows(2) {
        let a = (w[0].slot_index, w[0].weight_index, w[0].bit_index);
        let b = (w[1].slot_index, w[1].weight_index, w[1].bit_index);
        assert!(a <= b, "placement not sorted: {a:?} > {b:?}");
    }
}

// ------------------------------------------------------------------
// init + read
// ------------------------------------------------------------------

#[test]
fn init_then_read_returns_written_pointer() {
    let (bytes, map) = synthesize_cover();
    let mut cover = bytes;
    let sr = sample_pointer(0xBEEF);

    init_anchor(&mut cover, &map, sr).expect("init");
    let outcome = read_anchor(&cover, &map).expect("read");
    assert_eq!(outcome.active.super_root, sr);
    // After init, slot 1 has the higher generation (1 vs 0).
    assert_eq!(outcome.active_slot, SlotIndex::Slot1);
    assert_eq!(outcome.active.generation, 1);
}

#[test]
fn read_without_init_errors() {
    // Synthetic cover with no anchor written — anchor positions hold
    // whatever bits the cover had.
    let (bytes, map) = synthesize_cover();
    match read_anchor(&bytes, &map) {
        Err(AnchorError::NoValidAnchor) | Err(AnchorError::BadMagic { .. }) => {}
        other => panic!("expected NoValidAnchor or BadMagic, got {other:?}"),
    }
}

// ------------------------------------------------------------------
// commit alternation
// ------------------------------------------------------------------

#[test]
fn commit_bumps_generation_and_alternates_slot() {
    let (bytes, map) = synthesize_cover();
    let mut cover = bytes;
    let sr0 = sample_pointer(0x0000_00AA);
    let sr1 = sample_pointer(0x0000_00BB);
    let sr2 = sample_pointer(0x0000_00CC);

    init_anchor(&mut cover, &map, sr0).expect("init");
    let after_init = read_anchor(&cover, &map).expect("read after init");
    assert_eq!(after_init.active.generation, 1);
    assert_eq!(after_init.active_slot, SlotIndex::Slot1);

    let gen_1 = commit_anchor(&mut cover, &map, sr1, after_init.active.generation).expect("commit 1");
    assert_eq!(gen_1, after_init.active.generation + 1);
    let after_1 = read_anchor(&cover, &map).expect("read after commit 1");
    assert_eq!(after_1.active.super_root, sr1);
    assert_eq!(after_1.active.generation, gen_1);
    // Commit went to the inactive slot.
    assert_ne!(after_1.active_slot, after_init.active_slot);

    let gen_2 = commit_anchor(&mut cover, &map, sr2, after_1.active.generation).expect("commit 2");
    assert_eq!(gen_2, after_1.active.generation + 1);
    let after_2 = read_anchor(&cover, &map).expect("read after commit 2");
    assert_eq!(after_2.active.super_root, sr2);
    assert_eq!(after_2.active.generation, gen_2);
    // Second commit alternated back to the original init slot.
    assert_ne!(after_2.active_slot, after_1.active_slot);
    assert_eq!(after_2.active_slot, after_init.active_slot);
}

// ------------------------------------------------------------------
// Corrupt slot handling
// ------------------------------------------------------------------

#[test]
fn corrupt_slot0_falls_back_to_slot1() {
    // After init, slot 1 is active (gen=1), slot 0 holds gen=0. Commit
    // once so slot 0 gets the highest gen; then corrupt slot 0. Reader
    // should fall back to slot 1 (which still holds a valid gen=1).
    let (bytes, map) = synthesize_cover();
    let mut cover = bytes;
    let sr_a = sample_pointer(0xAAAA);
    let sr_b = sample_pointer(0xBBBB);

    init_anchor(&mut cover, &map, sr_a).expect("init");
    let after_init = read_anchor(&cover, &map).unwrap();
    assert_eq!(after_init.active_slot, SlotIndex::Slot1);

    // Commit — writes to slot 0 with gen=2.
    commit_anchor(&mut cover, &map, sr_b, after_init.active.generation).expect("commit");
    let after_commit = read_anchor(&cover, &map).unwrap();
    assert_eq!(after_commit.active_slot, SlotIndex::Slot0);
    assert_eq!(after_commit.active.generation, 2);
    assert_eq!(after_commit.active.super_root, sr_b);

    // Corrupt slot 0 by flipping a bit in its region. Reader should
    // fall back to slot 1 (which has sr_a at gen=1).
    flip_bits_in_anchor_slot(&mut cover, &map, SlotIndex::Slot0);
    let after_corrupt = read_anchor(&cover, &map).expect("fallback to slot 1");
    assert_eq!(after_corrupt.active_slot, SlotIndex::Slot1);
    assert_eq!(after_corrupt.active.generation, 1);
    assert_eq!(after_corrupt.active.super_root, sr_a);
}

#[test]
fn both_slots_corrupt_returns_error() {
    let (bytes, map) = synthesize_cover();
    let mut cover = bytes;
    let sr = sample_pointer(0xCAFE);
    init_anchor(&mut cover, &map, sr).expect("init");

    flip_bits_in_anchor_slot(&mut cover, &map, SlotIndex::Slot0);
    flip_bits_in_anchor_slot(&mut cover, &map, SlotIndex::Slot1);

    match read_anchor(&cover, &map) {
        Err(AnchorError::NoValidAnchor) => {}
        other => panic!("expected NoValidAnchor, got {other:?}"),
    }
}

/// Flip a bit inside a specific anchor slot's region. Uses the
/// documented layout: magic (4) + version (1) + reserved (3) + slot0
/// (28) + slot1 (28). Flips the first bit of the slot.
fn flip_bits_in_anchor_slot(cover: &mut [u8], map: &TensorMap, slot: SlotIndex) {
    let placement = find_anchor_placement(cover, map);
    // Offset in the anchor record (in bytes) where the slot starts.
    let slot_byte_offset = 8 + match slot {
        SlotIndex::Slot0 => 0,
        SlotIndex::Slot1 => 28,
    };
    // Flip all 8 bits of the first slot byte so CRC fails even if
    // one flip happens to land on a bit that was already flipped.
    let bit_base = slot_byte_offset * 8;
    for k in 0..8 {
        let pos = placement.positions[bit_base + k];
        let cover_slot = &map.slots[pos.slot_index as usize];
        let current = llmdb::stego::calibration::bit_io::read_bit(cover, cover_slot, pos);
        write_bit(cover, cover_slot, pos, !current);
    }
}

// ------------------------------------------------------------------
// Anchor record codec — implicit via the public API, but we can
// sanity-check the size.
// ------------------------------------------------------------------

#[test]
fn anchor_bits_constant_matches_expected_size() {
    // 8-byte header + 2 slots × 28 bytes = 64 bytes = 512 bits.
    assert_eq!(ANCHOR_BITS, 64 * 8);
}
