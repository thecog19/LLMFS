pub mod float;
pub mod q3_k;
pub mod q4_k;
pub mod q5_k;
pub mod q6_k;
pub mod q8_0;

use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum PackingError {
    #[error("invalid storage length for {context}: expected multiple of {unit}, got {actual}")]
    InvalidStorageLength {
        context: &'static str,
        unit: usize,
        actual: usize,
    },
    #[error("payload too large for {context}: max {max_payload_bytes} bytes, got {actual}")]
    PayloadTooLarge {
        context: &'static str,
        max_payload_bytes: usize,
        actual: usize,
    },
    #[error("index {index} out of range for {context}: len {len}")]
    IndexOutOfRange {
        context: &'static str,
        index: usize,
        len: usize,
    },
}

pub trait StegoPacker {
    const NAME: &'static str;
    const STEALABLE_BITS_PER_WEIGHT: usize;
}

pub fn supported_packers() -> &'static [&'static str] {
    &[
        q8_0::NAME,
        q6_k::NAME,
        q5_k::NAME,
        q4_k::NAME,
        q3_k::NAME,
        float::NAME,
    ]
}
