pub const GGML_TYPE_F32_ID: u32 = 0;
pub const GGML_TYPE_F16_ID: u32 = 1;
pub const GGML_TYPE_Q4_0_ID: u32 = 2;
pub const GGML_TYPE_Q4_1_ID: u32 = 3;
// IDs 4 and 5 are deprecated ggml quant types (Q4_2, Q4_3) and are never present in modern GGUF files.
pub const GGML_TYPE_Q5_0_ID: u32 = 6;
pub const GGML_TYPE_Q5_1_ID: u32 = 7;
pub const GGML_TYPE_Q8_0_ID: u32 = 8;
pub const GGML_TYPE_Q8_1_ID: u32 = 9;
pub const GGML_TYPE_Q2_K_ID: u32 = 10;
pub const GGML_TYPE_Q3_K_ID: u32 = 11;
pub const GGML_TYPE_Q4_K_ID: u32 = 12;
pub const GGML_TYPE_Q5_K_ID: u32 = 13;
pub const GGML_TYPE_Q6_K_ID: u32 = 14;
pub const GGML_TYPE_Q8_K_ID: u32 = 15;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum GgufQuantType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
}

impl GgufQuantType {
    pub fn from_raw_ggml_type(raw_type_id: u32) -> Option<Self> {
        match raw_type_id {
            GGML_TYPE_F32_ID => Some(Self::F32),
            GGML_TYPE_F16_ID => Some(Self::F16),
            GGML_TYPE_Q4_0_ID => Some(Self::Q4_0),
            GGML_TYPE_Q4_1_ID => Some(Self::Q4_1),
            GGML_TYPE_Q5_0_ID => Some(Self::Q5_0),
            GGML_TYPE_Q5_1_ID => Some(Self::Q5_1),
            GGML_TYPE_Q8_0_ID => Some(Self::Q8_0),
            GGML_TYPE_Q8_1_ID => Some(Self::Q8_1),
            GGML_TYPE_Q2_K_ID => Some(Self::Q2K),
            GGML_TYPE_Q3_K_ID => Some(Self::Q3K),
            GGML_TYPE_Q4_K_ID => Some(Self::Q4K),
            GGML_TYPE_Q5_K_ID => Some(Self::Q5K),
            GGML_TYPE_Q6_K_ID => Some(Self::Q6K),
            GGML_TYPE_Q8_K_ID => Some(Self::Q8K),
            _ => None,
        }
    }

    pub fn stealable_bits_hint(self) -> usize {
        match self {
            Self::Q8_0 => 4,
            Self::Q6K => 2,
            Self::Q5K | Self::Q4K | Self::Q3K => 1,
            Self::F16 => 4,
            Self::F32 => 8,
            Self::Q2K
            | Self::Q4_0
            | Self::Q4_1
            | Self::Q5_0
            | Self::Q5_1
            | Self::Q8_1
            | Self::Q8K => 0,
        }
    }
}
