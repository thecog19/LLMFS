use crate::stego::integrity::{FreeListBlock, IntegrityError, NO_BLOCK};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FreeListBootstrap {
    pub block_size: usize,
}

impl Default for FreeListBootstrap {
    fn default() -> Self {
        Self {
            block_size: crate::BLOCK_SIZE,
        }
    }
}

/// Build the initial free list chain as raw encoded blocks.
/// Returns a Vec of (block_index, encoded_block) pairs ready for
/// `write_logical_block_raw`.
pub fn build_free_chain(start: u32, end: u32) -> Vec<(u32, Vec<u8>)> {
    let mut chain = Vec::new();
    for block_index in start..end {
        let next = if block_index + 1 < end {
            block_index + 1
        } else {
            NO_BLOCK
        };
        let bytes = FreeListBlock {
            next_free_block: next,
        }
        .encode();
        chain.push((block_index, bytes));
    }
    chain
}

/// Decode a free-list block to extract the next pointer.
pub fn decode_next(bytes: &[u8]) -> Result<u32, IntegrityError> {
    let block = FreeListBlock::decode(bytes)?;
    Ok(block.next_free_block)
}

/// Encode a free-list block with the given next pointer.
pub fn encode_head(next: u32) -> Vec<u8> {
    FreeListBlock {
        next_free_block: next,
    }
    .encode()
}
