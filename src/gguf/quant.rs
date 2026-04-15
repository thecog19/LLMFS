pub const GGML_TYPE_F32_ID: u32 = 0;
pub const GGML_TYPE_F16_ID: u32 = 1;
pub const GGML_TYPE_Q8_0_ID: u32 = 8;
pub const GGML_TYPE_Q2_K_ID: u32 = 10;
pub const GGML_TYPE_Q3_K_ID: u32 = 11;
pub const GGML_TYPE_Q4_K_ID: u32 = 12;
pub const GGML_TYPE_Q5_K_ID: u32 = 13;
pub const GGML_TYPE_Q6_K_ID: u32 = 14;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufQuantType {
    Q8_0,
    Q6K,
    Q5K,
    Q4K,
    Q3K,
    Q2K,
    F16,
    F32,
}

impl GgufQuantType {
    pub fn from_raw_ggml_type(raw_type_id: u32) -> Option<Self> {
        match raw_type_id {
            GGML_TYPE_F32_ID => Some(Self::F32),
            GGML_TYPE_F16_ID => Some(Self::F16),
            GGML_TYPE_Q8_0_ID => Some(Self::Q8_0),
            GGML_TYPE_Q2_K_ID => Some(Self::Q2K),
            GGML_TYPE_Q3_K_ID => Some(Self::Q3K),
            GGML_TYPE_Q4_K_ID => Some(Self::Q4K),
            GGML_TYPE_Q5_K_ID => Some(Self::Q5K),
            GGML_TYPE_Q6_K_ID => Some(Self::Q6K),
            _ => None,
        }
    }

    pub fn stealable_bits_hint(self) -> usize {
        match self {
            Self::Q8_0 => 4,
            Self::Q6K => 2,
            Self::Q5K | Self::Q4K | Self::Q3K => 1,
            Self::Q2K => 0,
            Self::F16 => 4,
            Self::F32 => 8,
        }
    }
}
