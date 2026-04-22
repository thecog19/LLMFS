use crate::gguf::quant::GgufQuantType;
use crate::stego::planner::{AllocationPlan, TensorTier};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSlot {
    pub name: String,
    pub quant_type: GgufQuantType,
    pub tier: TensorTier,
    pub data_offset: u64,
    pub weight_count: u64,
    pub stealable_bits_per_weight: usize,
    pub capacity_bits: u64,
    pub bit_start: u64,
    pub bit_end: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalBitSegment {
    pub tensor_name: String,
    pub quant_type: GgufQuantType,
    pub tier: TensorTier,
    pub data_offset: u64,
    pub bit_offset_in_tensor: u64,
    pub bit_len: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogicalByteMapping {
    pub byte_index: u64,
    pub segments: Vec<PhysicalBitSegment>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorMap {
    pub slots: Vec<TensorSlot>,
    pub total_capacity_bits: u64,
    pub total_capacity_bytes: u64,
}

impl TensorMap {
    /// Build a map whose `TensorSlot::data_offset` values are still
    /// relative to the GGUF tensor-data region. Used by code paths
    /// (tests, benches, address-translation diagnostics) that don't
    /// touch the real mmap and only care about logical→physical bit
    /// arithmetic.
    pub fn from_allocation_plan(plan: &AllocationPlan) -> Self {
        Self::from_allocation_plan_with_base(plan, 0)
    }

    /// Build a map whose `TensorSlot::data_offset` values are
    /// **absolute** mmap offsets (i.e. each tensor's relative
    /// `data_offset` plus the GGUF's `tensor_data_offset`).
    /// Calibration and any other code that reads weight bytes from
    /// the cover file must use this constructor — `data_offset`
    /// otherwise points to the wrong bytes.
    pub fn from_allocation_plan_with_base(plan: &AllocationPlan, base_offset: u64) -> Self {
        let mut slots = Vec::with_capacity(plan.tensors.len());
        let mut bit_cursor = 0_u64;

        for tensor in &plan.tensors {
            let bit_start = bit_cursor;
            bit_cursor = bit_cursor.saturating_add(tensor.capacity_bits);

            slots.push(TensorSlot {
                name: tensor.name.clone(),
                quant_type: tensor.quant_type,
                tier: tensor.tier,
                data_offset: tensor.data_offset.saturating_add(base_offset),
                weight_count: tensor.weight_count,
                stealable_bits_per_weight: tensor.stealable_bits_per_weight,
                capacity_bits: tensor.capacity_bits,
                bit_start,
                bit_end: bit_cursor,
            });
        }

        Self {
            slots,
            total_capacity_bits: bit_cursor,
            total_capacity_bytes: bit_cursor / 8,
        }
    }

    pub fn map_logical_byte(&self, byte_index: u64) -> Option<LogicalByteMapping> {
        if byte_index >= self.total_capacity_bytes {
            return None;
        }

        let mut segments = Vec::new();
        let mut remaining_bits = 8_u64;
        let mut current_bit = byte_index * 8;

        while remaining_bits > 0 {
            let slot = self.slot_for_bit(current_bit)?;
            let bit_offset_in_tensor = current_bit - slot.bit_start;
            let available_bits = slot.bit_end - current_bit;
            let take_bits = remaining_bits.min(available_bits);

            segments.push(PhysicalBitSegment {
                tensor_name: slot.name.clone(),
                quant_type: slot.quant_type,
                tier: slot.tier,
                data_offset: slot.data_offset,
                bit_offset_in_tensor,
                bit_len: take_bits as u8,
            });

            current_bit += take_bits;
            remaining_bits -= take_bits;
        }

        Some(LogicalByteMapping {
            byte_index,
            segments,
        })
    }

    fn slot_for_bit(&self, bit_index: u64) -> Option<&TensorSlot> {
        self.slots
            .iter()
            .find(|slot| bit_index >= slot.bit_start && bit_index < slot.bit_end)
    }
}
