use llmdb::gguf::quant::GgufQuantType;
use llmdb::stego::packing::packer_for;

#[test]
fn packer_for_supported_types_reports_design_spec_shapes() {
    let cases: &[(GgufQuantType, u32, usize, usize)] = &[
        (GgufQuantType::Q8_0, 4, 34, 32),
        (GgufQuantType::Q6K, 2, 210, 256),
        (GgufQuantType::Q5K, 1, 176, 256),
        (GgufQuantType::Q4K, 1, 144, 256),
        (GgufQuantType::Q3K, 1, 110, 256),
        (GgufQuantType::F16, 4, 2, 1),
        (GgufQuantType::F32, 8, 4, 1),
    ];

    for (quant, bits, block, weights) in cases.iter().copied() {
        let packer = packer_for(quant);
        assert_eq!(packer.bits_per_weight(), bits, "{quant:?} bits");
        assert_eq!(packer.block_size_bytes(), block, "{quant:?} block size");
        assert_eq!(packer.weights_per_block(), weights, "{quant:?} weights");
        assert!(
            !packer.stealable_byte_offsets().is_empty(),
            "{quant:?} must expose at least one stealable byte offset"
        );
    }
}

#[test]
fn packer_for_unsupported_types_reports_zero_bits() {
    for quant in [
        GgufQuantType::Q2K,
        GgufQuantType::Q4_0,
        GgufQuantType::Q4_1,
        GgufQuantType::Q5_0,
        GgufQuantType::Q5_1,
        GgufQuantType::Q8_1,
        GgufQuantType::Q8K,
    ] {
        let packer = packer_for(quant);
        assert_eq!(
            packer.bits_per_weight(),
            0,
            "{quant:?} must report zero stealable bits"
        );
        assert!(
            packer.stealable_byte_offsets().is_empty(),
            "{quant:?} must expose no stealable byte offsets"
        );
    }
}

#[test]
fn q8_0_trait_extract_and_embed_roundtrip_via_trait_methods() {
    let packer = packer_for(GgufQuantType::Q8_0);
    let mut block = vec![0_u8; packer.block_size_bytes()];
    block[0] = 0xAB;
    block[1] = 0xCD;
    for (index, slot) in block[2..].iter_mut().enumerate() {
        *slot = 0xA0 | ((index as u8 * 5) & 0x0F);
    }
    let original = block.clone();
    let payload: Vec<u8> = (0..16_u8).map(|i| i * 17).collect();

    let embedded = packer.embed(&block, &payload);
    assert_eq!(embedded.len(), packer.block_size_bytes());
    let roundtrip = packer.extract(&embedded);
    assert_eq!(roundtrip, payload);

    // Non-stolen bits preserved: scale bytes + high nibble of each quant.
    assert_eq!(embedded[0], 0xAB);
    assert_eq!(embedded[1], 0xCD);
    for index in 0..32 {
        assert_eq!(embedded[2 + index] & 0xF0, original[2 + index] & 0xF0);
    }
}

#[test]
fn f16_trait_extract_and_embed_preserve_upper_twelve_bits() {
    let packer = packer_for(GgufQuantType::F16);
    let original: [u8; 2] = [0x3A, 0x12];
    let embedded = packer.embed(&original, &[0x0B]);
    let payload = packer.extract(&embedded);
    assert_eq!(payload, vec![0x0B]);

    let before = u16::from_le_bytes(original);
    let after = u16::from_le_bytes([embedded[0], embedded[1]]);
    assert_eq!(before & 0xFFF0, after & 0xFFF0);
}

#[test]
fn f32_trait_extract_and_embed_preserve_upper_twenty_four_bits() {
    let packer = packer_for(GgufQuantType::F32);
    let original: [u8; 4] = [0x7F, 0x56, 0x78, 0x12];
    let embedded = packer.embed(&original, &[0xC3]);
    let payload = packer.extract(&embedded);
    assert_eq!(payload, vec![0xC3]);

    let before = u32::from_le_bytes(original);
    let after = u32::from_le_bytes([embedded[0], embedded[1], embedded[2], embedded[3]]);
    assert_eq!(before & 0xFFFF_FF00, after & 0xFFFF_FF00);
}

#[test]
#[should_panic]
fn unsupported_packer_panics_on_extract() {
    let packer = packer_for(GgufQuantType::Q2K);
    let _ = packer.extract(&[]);
}

#[test]
#[should_panic]
fn unsupported_packer_panics_on_embed() {
    let packer = packer_for(GgufQuantType::Q2K);
    let _ = packer.embed(&[], &[]);
}
