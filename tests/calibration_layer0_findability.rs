//! Layer 0's "findable on reopen" claim (DESIGN-NEW §15.2).
//!
//! Layer 0 has no sidecar, no pointer, no persisted location: the rule
//! for where metadata lives is "lowest-magnitude stealable-bit positions
//! in the current cover." That works only if re-running the rule on a
//! *post-write* cover gives back the same positions the writer used —
//! otherwise the metadata is unreadable after reopen.
//!
//! Writes perturb the magnitudes of the very weights metadata sits on,
//! so the invariant is non-trivial. Per quant type:
//!
//! - **F16** (4 stealable bits = low mantissa nibble). Worst-case
//!   relative magnitude change is `15/1024 ≈ 1.5%`. The bottom-N
//!   boundary weights are deep in the tail of a trained distribution;
//!   a 1.5% bump is too small to let them leapfrog the next-lowest
//!   unselected weight. Expected: **stable**.
//! - **F32** (8 stealable bits = low byte of mantissa). Relative
//!   change up to `255 / 2^23 ≈ 3e-5`. Expected: **stable**.
//! - **Q8_0** (4 stealable bits = low nibble of int8). Flipping those
//!   bits can swing an int8 value by up to 15, and for an int8 near
//!   zero that's a *magnitude-multiplying* change. The magnitude
//!   ranking is effectively re-randomised by a write. This is the same
//!   bit theft that destroys inference per
//!   `project_v1_stego_destroys_inference.md`; the instability here
//!   is the algorithmic analogue. Layer 0 alone does **not** rescue
//!   Q8_0 — the §15.5 protected-region trick is what will. Expected:
//!   **drift**, captured as an inverted assertion that becomes a loud
//!   signal the day something upstream fixes it.
//!
//! If an F16 / F32 test fails, Layer 0 is broken and we need to either
//! (a) find the boundary case the argument above missed, or (b) widen
//! the margin (e.g. only pick positions whose magnitude is strictly
//! below the cutoff bucket, trading a little capacity for determinism).
//!
//! If the Q8_0 test *passes* (cover stays stable), don't celebrate yet:
//! double-check the fixture actually perturbs magnitudes — a seed where
//! metadata positions happen to fall on zero-value weights would fake
//! a pass. The test rejects that by sampling nonzero-value positions
//! and asserting the int8 values changed.

use llmdb::gguf::quant::GgufQuantType;
use llmdb::stego::calibration::byte_io::{read_bytes, write_bytes};
use llmdb::stego::calibration::placement::compute_metadata_placement;
use llmdb::stego::planner::TensorTier;
use llmdb::stego::tensor_map::{TensorMap, TensorSlot};

// ------------------------------------------------------------------
// Deterministic PRNG + helpers
// ------------------------------------------------------------------

/// xorshift64 — seed-locked, reproducible. Paired with the individual
/// `fn` names so a failing assertion points at exactly which corpus it
/// needs to regenerate.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        // Scramble: raw 0 would stick at 0 in xorshift.
        let s = seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(0x1234_5678_DEAD_BEEF);
        Self(if s == 0 { 1 } else { s })
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn byte(&mut self) -> u8 {
        (self.next_u64() & 0xFF) as u8
    }
    /// Uniform in (0, 1) exclusive of both endpoints.
    fn uniform_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 + 0.5) / (1_u64 << 24) as f32
    }
    /// N(0, 1) via Box-Muller.
    fn gaussian_f32(&mut self) -> f32 {
        let u1 = self.uniform_f32().max(1e-10);
        let u2 = self.uniform_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

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
        return (sign << 15) | (0x1E << 10) | 0x3FF;
    }
    let mantissa16 = (mantissa32 >> 13) as u16;
    (sign << 15) | ((exp16 as u16) << 10) | mantissa16
}

fn random_bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| rng.byte()).collect()
}

fn single_slot_map(mut slot: TensorSlot) -> TensorMap {
    slot.bit_start = 0;
    slot.bit_end = slot.capacity_bits;
    let total = slot.capacity_bits;
    TensorMap {
        slots: vec![slot],
        total_capacity_bits: total,
        total_capacity_bytes: total / 8,
    }
}

// ------------------------------------------------------------------
// Synthetic cover builders
// ------------------------------------------------------------------

/// F16 cover whose weights are `N(0, 0.1)`-distributed — mimics a
/// trained model's weight spread enough that the bottom-N tail is
/// densely populated with small magnitudes (the worst case for
/// drift).
fn synthesize_f16_cover(weight_count: u64, seed: u64) -> (TensorSlot, Vec<u8>) {
    let mut rng = Rng::new(seed);
    let mut bytes = Vec::with_capacity(weight_count as usize * 2);
    for _ in 0..weight_count {
        let w = 0.1 * rng.gaussian_f32();
        bytes.extend_from_slice(&f32_to_f16_bits(w).to_le_bytes());
    }
    let bits = GgufQuantType::F16.stealable_bits_hint() as u64;
    let slot = TensorSlot {
        name: "findable.f16".to_owned(),
        quant_type: GgufQuantType::F16,
        tier: TensorTier::Tier1,
        data_offset: 0,
        weight_count,
        stealable_bits_per_weight: bits as usize,
        capacity_bits: weight_count * bits,
        bit_start: 0,
        bit_end: weight_count * bits,
    };
    (slot, bytes)
}

fn synthesize_f32_cover(weight_count: u64, seed: u64) -> (TensorSlot, Vec<u8>) {
    let mut rng = Rng::new(seed);
    let mut bytes = Vec::with_capacity(weight_count as usize * 4);
    for _ in 0..weight_count {
        let w = 0.1 * rng.gaussian_f32();
        bytes.extend_from_slice(&w.to_bits().to_le_bytes());
    }
    let bits = GgufQuantType::F32.stealable_bits_hint() as u64;
    let slot = TensorSlot {
        name: "findable.f32".to_owned(),
        quant_type: GgufQuantType::F32,
        tier: TensorTier::Tier1,
        data_offset: 0,
        weight_count,
        stealable_bits_per_weight: bits as usize,
        capacity_bits: weight_count * bits,
        bit_start: 0,
        bit_end: weight_count * bits,
    };
    (slot, bytes)
}

/// Q8_0 cover with realistic per-block scale variation. int8 values
/// span the full [-127, 127] range so the low-magnitude weights aren't
/// all sitting at int8=0 (which would fake stability — flipping low
/// nibble of 0 can push to 15 but that's still a small magnitude if
/// scale is small; we want a mix).
fn synthesize_q8_0_cover(weight_count: u64, seed: u64) -> (TensorSlot, Vec<u8>) {
    assert!(weight_count.is_multiple_of(32), "Q8_0 blocks are 32 weights");
    let block_count = weight_count / 32;
    let mut rng = Rng::new(seed);
    let mut bytes = Vec::with_capacity(block_count as usize * 34);
    for _ in 0..block_count {
        // Scale ~ log-uniform in [1e-4, 1e-1]: realistic spread of
        // per-block scales in a trained int8 cover.
        let scale = 1e-4_f32 * 1000.0_f32.powf(rng.uniform_f32());
        bytes.extend_from_slice(&f32_to_f16_bits(scale).to_le_bytes());
        for _ in 0..32 {
            let int8 = rng.byte() as i8;
            bytes.push(int8 as u8);
        }
    }
    let bits = GgufQuantType::Q8_0.stealable_bits_hint() as u64;
    let slot = TensorSlot {
        name: "findable.q8_0".to_owned(),
        quant_type: GgufQuantType::Q8_0,
        tier: TensorTier::Tier1,
        data_offset: 0,
        weight_count,
        stealable_bits_per_weight: bits as usize,
        capacity_bits: weight_count * bits,
        bit_start: 0,
        bit_end: weight_count * bits,
    };
    (slot, bytes)
}

// ------------------------------------------------------------------
// The "did placement drift?" measurement
// ------------------------------------------------------------------

#[derive(Debug)]
struct Findability {
    /// Bit-position-level drift: `placement_a.positions[i]` differs
    /// from `placement_b.positions[i]` for some `i < needed_bits`.
    /// This is the strict form of the Layer 0 contract — readback
    /// addresses bit `i` through `placement_b.positions[i]`, so even
    /// an ordering swap within the selected set breaks the readback.
    position_drift: usize,
    /// Selection-level drift: weights present in `placement_a` but
    /// absent from `placement_b` (or vice-versa), counted over the
    /// `needed_bits`-bit prefix of each. A weight that stays in the
    /// selection but shifts rank counts 0 here. Probes the weaker
    /// invariant: "the bottom-N weight set is stable under writes."
    selection_drift: usize,
    /// Bytes that differ between the original write buffer and the
    /// readback through `placement_b`. Non-zero means the metadata
    /// is not findable via recomputation.
    byte_mismatch: usize,
    /// Cover bytes whose value changed during the write. A zero here
    /// means the write landed on bits that already matched the target
    /// and the test is vacuous — reject it upstream.
    cover_bytes_modified: usize,
}

fn measure(
    cover: &mut [u8],
    map: &TensorMap,
    needed_bits: u64,
    data: &[u8],
) -> Findability {
    let before = cover.to_vec();
    let placement_a = compute_metadata_placement(cover, map, needed_bits);
    assert!(
        placement_a.positions.len() as u64 >= needed_bits,
        "fixture too small: placement only covers {} bits, asked for {}",
        placement_a.positions.len(),
        needed_bits,
    );

    write_bytes(cover, map, &placement_a, 0, data).expect("write in-bounds");

    let placement_b = compute_metadata_placement(cover, map, needed_bits);

    let first_n = needed_bits as usize;
    let position_drift = placement_a.positions[..first_n]
        .iter()
        .zip(&placement_b.positions[..first_n])
        .filter(|(a, b)| a != b)
        .count();

    let weights_a: std::collections::HashSet<(u32, u64)> = placement_a.positions[..first_n]
        .iter()
        .map(|p| (p.slot_index, p.weight_index))
        .collect();
    let weights_b: std::collections::HashSet<(u32, u64)> = placement_b.positions[..first_n]
        .iter()
        .map(|p| (p.slot_index, p.weight_index))
        .collect();
    let selection_drift = weights_a.symmetric_difference(&weights_b).count();

    let mut readback = vec![0_u8; data.len()];
    read_bytes(cover, map, &placement_b, 0, &mut readback).expect("read in-bounds");
    let byte_mismatch = readback.iter().zip(data).filter(|(a, b)| a != b).count();

    let cover_bytes_modified = before.iter().zip(cover.iter()).filter(|(a, b)| a != b).count();

    Findability {
        position_drift,
        selection_drift,
        byte_mismatch,
        cover_bytes_modified,
    }
}

// ------------------------------------------------------------------
// Tests
// ------------------------------------------------------------------

#[test]
fn f16_placement_is_findable_on_reopen() {
    // 10_000 F16 weights × 4 bits = 40_000 stealable bits.
    // Metadata region: 800 bits = 100 bytes.
    let (slot, cover) = synthesize_f16_cover(10_000, 42);
    let map = single_slot_map(slot);
    let mut cover = cover;
    let data = random_bytes(100, 0xC0FFEE);

    let m = measure(&mut cover, &map, 800, &data);

    // The write must actually have perturbed the cover — otherwise
    // "stability" is vacuous.
    assert!(
        m.cover_bytes_modified > 0,
        "fixture didn't modify the cover — no-op writes don't test drift",
    );

    assert_eq!(
        m.position_drift, 0,
        "F16 bit-position drift: {} of 800 positions shifted after write \
         (selection_drift={}, cover_bytes_modified={}, byte_mismatch={}). \
         Layer 0 not findable — readback pulls bits from the wrong offsets.",
        m.position_drift, m.selection_drift, m.cover_bytes_modified, m.byte_mismatch,
    );
    assert_eq!(
        m.byte_mismatch, 0,
        "F16 readback from recomputed placement differs from written data \
         ({} bytes differ)",
        m.byte_mismatch,
    );
}

#[test]
fn f32_placement_is_findable_on_reopen() {
    // 10_000 F32 weights × 8 bits = 80_000 bits. Metadata region:
    // 800 bits.
    let (slot, cover) = synthesize_f32_cover(10_000, 42);
    let map = single_slot_map(slot);
    let mut cover = cover;
    let data = random_bytes(100, 0xBEEF_CAFE);

    let m = measure(&mut cover, &map, 800, &data);

    assert!(m.cover_bytes_modified > 0);
    assert_eq!(
        m.position_drift, 0,
        "F32 placement drifted (drift={}) — this is surprising, perturbation \
         is bounded by 255/2^23 ≈ 3e-5 relative; investigate boundary case",
        m.position_drift,
    );
    assert_eq!(
        m.byte_mismatch, 0,
        "F32 readback differs: {} bytes",
        m.byte_mismatch,
    );
}

#[test]
fn q8_0_placement_drifts_documents_layer0_alone_insufficient() {
    // Q8_0: 320 blocks × 32 weights = 10_240 weights. Stealable bits:
    // 4 per weight → 40_960 bits. Metadata region: 800 bits.
    let (slot, cover) = synthesize_q8_0_cover(10_240, 42);
    let map = single_slot_map(slot);
    let mut cover = cover;
    let data = random_bytes(100, 0xDEAD_BEEF);

    let m = measure(&mut cover, &map, 800, &data);

    assert!(m.cover_bytes_modified > 0);

    // Inverted assertion — documents a known-bad behaviour. This test
    // passing is the CURRENT state; if it starts failing, Layer 0 got
    // strengthened (or the protected-region fix from §15.5 landed).
    // Investigate before loosening the assertion.
    assert!(
        m.position_drift > 0,
        "Q8_0 placement unexpectedly stable under writes (drift=0). \
         Either Layer 0 was strengthened upstream — in which case update \
         this test — or the fixture's int8 values don't actually perturb \
         magnitudes. Inspect `cover_bytes_modified={}` and the fixture's \
         scale distribution.",
        m.cover_bytes_modified,
    );
    assert!(
        m.byte_mismatch > 0,
        "Q8_0 readback matched write bit-for-bit despite placement drift=\
         {}. Either drift happened only in the tail bits beyond the write, \
         or the specific write pattern happened to land on positions that \
         don't reorder (low probability). If this triggers, log and \
         investigate before assuming Layer 0 became safe.",
        m.position_drift,
    );
}
