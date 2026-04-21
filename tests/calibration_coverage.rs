//! Coverage assertion for the magnitude dispatch in
//! `calibration::magnitude::read_weight_abs`. Every `GgufQuantType`
//! variant is explicitly classified here — if a new variant lands
//! without being listed, this test file fails to compile because of
//! the exhaustive `classify()` match. That forces a deliberate
//! decision: wire up a decoder or mark the variant as a stub.
//!
//! Three invariants checked:
//!
//! 1. **Decoded variants return nonzero magnitude for nonzero
//!    weights.** The failure mode this guards against: a decoder
//!    returning 0.0 for every weight (e.g. the TensorMap-offset bug
//!    we hit at 5555a13 where every K-quant read landed in metadata
//!    padding). Pristine-smoke catches it for the types the smoke
//!    fixture uses; this test covers all decoded types uniformly.
//!
//! 2. **Stub variants return 0.0.** Stubs are documented as not yet
//!    implemented; covers using them will silently mis-rank (the
//!    magnitude estimator treats them as minimum salience). This
//!    test makes the stub status explicit and reviewable.
//!
//! 3. **Eligible stubs are an enumerated known-bug set.** "Eligible"
//!    = `stealable_bits_for(t) > 0`. An eligible stub is a latent
//!    bug: the planner will allocate stego capacity in that tensor,
//!    but the estimator can't rank its weights properly. We expect
//!    exactly one eligible stub today (Q3_K); any new one is a
//!    regression unless consciously added.

use llmdb::gguf::quant::GgufQuantType;
use llmdb::stego::calibration::magnitude::read_weight_abs;
use llmdb::stego::calibration::stealable_bits_for;
use llmdb::stego::packing::{q4_k, q5_k, q6_k};
use llmdb::stego::planner::TensorTier;
use llmdb::stego::tensor_map::TensorSlot;

/// Enumerate every variant of `GgufQuantType`. If a new variant is
/// added to the enum but forgotten here, the exhaustive match in
/// `classify()` won't compile — the change forces a conscious update.
const ALL_VARIANTS: &[GgufQuantType] = &[
    GgufQuantType::F32,
    GgufQuantType::F16,
    GgufQuantType::Q4_0,
    GgufQuantType::Q4_1,
    GgufQuantType::Q5_0,
    GgufQuantType::Q5_1,
    GgufQuantType::Q8_0,
    GgufQuantType::Q8_1,
    GgufQuantType::Q2K,
    GgufQuantType::Q3K,
    GgufQuantType::Q4K,
    GgufQuantType::Q5K,
    GgufQuantType::Q6K,
    GgufQuantType::Q8K,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// `read_weight_abs` decodes this type; weights produce real
    /// magnitudes.
    Decoded,
    /// `read_weight_abs` hard-codes 0.0 for this type; it has no
    /// decoder yet. Re-classify to `Decoded` when one lands.
    Stub,
}

/// The authoritative classification. Kept in-sync by hand with the
/// dispatch arms of `calibration::magnitude::read_weight_abs` — if
/// the dispatch and this function drift, one of the assertions below
/// will flag it.
fn classify(t: GgufQuantType) -> Kind {
    match t {
        GgufQuantType::F32
        | GgufQuantType::F16
        | GgufQuantType::Q8_0
        | GgufQuantType::Q4K
        | GgufQuantType::Q5K
        | GgufQuantType::Q6K => Kind::Decoded,
        GgufQuantType::Q3K
        | GgufQuantType::Q2K
        | GgufQuantType::Q4_0
        | GgufQuantType::Q4_1
        | GgufQuantType::Q5_0
        | GgufQuantType::Q5_1
        | GgufQuantType::Q8_1
        | GgufQuantType::Q8K => Kind::Stub,
    }
}

fn slot_of(quant_type: GgufQuantType, weight_count: u64, data_offset: u64) -> TensorSlot {
    let bits = quant_type.stealable_bits_hint() as u64;
    TensorSlot {
        name: format!("coverage.{quant_type:?}").to_lowercase(),
        quant_type,
        tier: TensorTier::Tier1,
        data_offset,
        weight_count,
        stealable_bits_per_weight: bits as usize,
        capacity_bits: weight_count * bits,
        bit_start: 0,
        bit_end: weight_count * bits,
    }
}

/// Synthesize a byte buffer where weight 0 of the quant type has a
/// known nonzero magnitude. For stubs we don't need a real fixture —
/// `read_weight_abs` short-circuits to 0.0 without touching the mmap.
fn nonzero_fixture(quant_type: GgufQuantType) -> Vec<u8> {
    match quant_type {
        // F16 bits 0x3C00 = 1.0
        GgufQuantType::F16 => vec![0x00, 0x3C],
        // F32 bits 0x3F800000 = 1.0 (LE: 00 00 80 3F)
        GgufQuantType::F32 => vec![0x00, 0x00, 0x80, 0x3F],
        GgufQuantType::Q8_0 => {
            // 34-byte block: fp16 scale = 1.0, first int8 quant = 7
            // → weight 0 = |1 * 7| = 7.0.
            let mut b = vec![0_u8; 34];
            b[0..2].copy_from_slice(&0x3C00_u16.to_le_bytes());
            b[2] = 7;
            b
        }
        GgufQuantType::Q4K => {
            // d=1.0, dmin=0, scales[0]=1, scales[4]=0 → sub-block 0
            // has (sc=1, m=0). qs[0] = 0x07 → low nibble 7. Weight 0
            // = 1 * 1 * 7 - 0 * 0 = 7.
            let mut b = vec![0_u8; q4_k::BLOCK_BYTES];
            b[0..2].copy_from_slice(&0x3C00_u16.to_le_bytes()); // d
            b[2..4].copy_from_slice(&0x0000_u16.to_le_bytes()); // dmin
            b[4] = 1; // scales[0] holds sc for sub-block 0
            b[16] = 0x07; // qs[0] low nibble = 7
            b
        }
        GgufQuantType::Q5K => {
            let mut b = vec![0_u8; q5_k::BLOCK_BYTES];
            b[0..2].copy_from_slice(&0x3C00_u16.to_le_bytes()); // d
            b[2..4].copy_from_slice(&0x0000_u16.to_le_bytes()); // dmin
            b[4] = 1; // scales[0]
            b[48] = 0x07; // qs[0] low nibble = 7; qh[0] already 0 → no high bit
            b
        }
        GgufQuantType::Q6K => {
            // d=1.0, scales[0]=1, ql[0] low nibble = 5, qh[0] = 0 →
            // weight 0 q6 = (5 | 0) - 32 = -27, value = 1 * 1 * -27 =
            // -27 → |w| = 27.
            let mut b = vec![0_u8; q6_k::BLOCK_BYTES];
            b[0] = 0x05; // ql[0] low nibble 5
            b[208..210].copy_from_slice(&0x3C00_u16.to_le_bytes()); // d
            b[q6_k::BLOCK_BYTES - 18] = 1; // scales[0] (scales array at 128+64=192)
            b
        }
        // Stubs — buffers never touched; zero bytes suffice.
        GgufQuantType::Q3K => vec![0_u8; 110],
        GgufQuantType::Q2K
        | GgufQuantType::Q4_0
        | GgufQuantType::Q4_1
        | GgufQuantType::Q5_0
        | GgufQuantType::Q5_1
        | GgufQuantType::Q8_1
        | GgufQuantType::Q8K => Vec::new(),
    }
}

#[test]
fn decoded_types_return_nonzero_magnitude_for_nonzero_weights() {
    for t in ALL_VARIANTS.iter().copied() {
        if classify(t) != Kind::Decoded {
            continue;
        }
        let bytes = nonzero_fixture(t);
        let slot = slot_of(t, 1, 0);
        let mag = read_weight_abs(&bytes, &slot, 0);
        assert!(
            mag > 0.0,
            "decoded type {t:?} returned magnitude 0.0 for a known-nonzero weight — \
             this is the silent-mis-rank failure mode (see pristine_smoke.rs's \
             comment about the TensorMap-offset bug)"
        );
    }
}

#[test]
fn stub_types_return_zero_magnitude() {
    // Stubs short-circuit in `read_weight_abs`. The inputs don't
    // matter — any byte buffer should yield 0.0. Verifying this
    // anchors the "stub" side of the classification.
    for t in ALL_VARIANTS.iter().copied() {
        if classify(t) != Kind::Stub {
            continue;
        }
        let bytes = nonzero_fixture(t);
        let slot = slot_of(t, 1, 0);
        let mag = read_weight_abs(&bytes, &slot, 0);
        assert_eq!(
            mag, 0.0,
            "type {t:?} classified as stub but returned {mag} — either a decoder \
             was added (reclassify to Decoded) or someone injected a nonzero return"
        );
    }
}

#[test]
fn eligible_stubs_are_the_known_bug_set() {
    // An "eligible stub" is a quant type that the planner would give
    // stealable capacity to but the estimator can't actually rank.
    // Today Q3K is the only one — the rest of the stubs (Q2K, Q4_0,
    // Q4_1, Q5_0, Q5_1, Q8_1, Q8K) advertise 0 stealable bits, so
    // the planner skips them anyway.
    //
    // If this list grows without a matching decoder, covers using
    // that quant type will have their low-magnitude ranking
    // dominated by stub-type weights (all 0.0), producing subtly
    // wrong metadata placements. The fix is a real decoder, not
    // reclassification.
    let mut eligible_stubs: Vec<GgufQuantType> = ALL_VARIANTS
        .iter()
        .copied()
        .filter(|t| classify(*t) == Kind::Stub && stealable_bits_for(*t) > 0)
        .collect();
    eligible_stubs.sort_by_key(|t| format!("{t:?}"));
    assert_eq!(
        eligible_stubs,
        vec![GgufQuantType::Q3K],
        "eligible-stub set changed — add a decoder (preferred) or confirm the \
         stealable-bit budget in stealable_bits_hint() is deliberate",
    );
}

#[test]
fn every_variant_is_classified() {
    // Defence in depth: assertion that ALL_VARIANTS covers every
    // variant of the enum. If a new variant is added to the enum,
    // `classify()` won't compile without an arm for it — but
    // ALL_VARIANTS is a static slice and could drift. This test
    // catches the drift by counting against the enum's own coverage
    // via `from_raw_ggml_type`.
    // GgufQuantType doesn't implement Hash, so use a Vec and the
    // derived PartialEq. The list is tiny; O(N²) dedup is fine.
    let mut seen: Vec<GgufQuantType> = Vec::new();
    for t in ALL_VARIANTS {
        assert!(
            !seen.contains(t),
            "duplicate variant in ALL_VARIANTS: {t:?}"
        );
        seen.push(*t);
    }
    // Round-trip every raw GGML id in the documented set, make sure
    // every mapped variant appears in seen. If from_raw_ggml_type
    // starts returning a new variant, this fires.
    let raw_ids: &[u32] = &[0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    for id in raw_ids {
        let t = GgufQuantType::from_raw_ggml_type(*id)
            .unwrap_or_else(|| panic!("raw id {id} no longer maps to a variant"));
        assert!(
            seen.contains(&t),
            "raw id {id} maps to {t:?}, which isn't in ALL_VARIANTS",
        );
    }
}
