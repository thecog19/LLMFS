use std::collections::{HashMap, HashSet};
use std::fs::OpenOptions;
use std::path::Path;

use crc32fast::Hasher;
use memmap2::MmapMut;
use thiserror::Error;

use crate::gguf::parser::{self, GgufFile};
use crate::gguf::quant::GgufQuantType;
use crate::stego::freelist;
use crate::stego::integrity::{
    ENTRIES_PER_INTEGRITY_BLOCK, IntegrityBlock, IntegrityError, NO_BLOCK, Superblock,
    SuperblockFields, encode_quant_profile,
};
use crate::stego::packing::{self, PackingError};
use crate::stego::planner::{AllocationMode, AllocationPlan, build_allocation_plan};
use crate::stego::redirection::{RedirectionError, RedirectionTable};

const SUPERBLOCK_BLOCK: u32 = 0;
const FILE_TABLE_INITIAL_BLOCKS: u32 = 1;

#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrashPoint {
    AfterShadowFlush,
    AfterRedirectionFlush,
}

impl CrashPoint {
    fn as_stop_phase(self) -> u8 {
        match self {
            Self::AfterShadowFlush => 1,
            Self::AfterRedirectionFlush => 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceBootstrap {
    pub block_size: usize,
}

impl Default for DeviceBootstrap {
    fn default() -> Self {
        Self {
            block_size: crate::BLOCK_SIZE,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DeviceOptions {
    pub verbose: bool,
}

#[derive(Debug)]
pub struct StegoDevice {
    mmap: MmapMut,
    plan: AllocationPlan,
    slots: Vec<TensorByteSlot>,
    superblock: Superblock,
    redirection: RedirectionTable,
    /// Logicals reserved by `alloc_block` that haven't yet had their first
    /// write. In-memory only — a crash before the first write loses the
    /// reservation, which is fine: on reopen the logical reads as
    /// unmapped (zeros) and re-allocates cleanly.
    reserved_logicals: HashSet<u32>,
    verbose: bool,
}

impl StegoDevice {
    pub fn initialize(path: impl AsRef<Path>, mode: AllocationMode) -> Result<Self, DeviceError> {
        Self::initialize_with_options(path, mode, DeviceOptions::default())
    }

    pub fn initialize_with_options(
        path: impl AsRef<Path>,
        mode: AllocationMode,
        options: DeviceOptions,
    ) -> Result<Self, DeviceError> {
        let mut device = Self::open_internal(path, mode, options)?;
        device.format()?;
        device.superblock.set_dirty(true);
        device.persist_superblock()?;
        Ok(device)
    }

    pub fn open(path: impl AsRef<Path>, mode: AllocationMode) -> Result<Self, DeviceError> {
        Self::open_with_options(path, mode, DeviceOptions::default())
    }

    pub fn open_with_options(
        path: impl AsRef<Path>,
        mode: AllocationMode,
        options: DeviceOptions,
    ) -> Result<Self, DeviceError> {
        let mut device = Self::open_internal(path, mode, options)?;
        device.superblock = device.load_superblock()?;
        device.redirection = device.load_redirection_table()?;

        if device.superblock.is_dirty() {
            device.log_verbose("unclean shutdown detected — running recovery");
            device.recover()?;
        }

        device.superblock.set_dirty(true);
        device.persist_superblock()?;

        device.log_verbose(format!(
            "opened device: total_blocks={}, data_region_start={}, free_list_head={}",
            device.superblock.fields.total_blocks,
            device.data_region_start(),
            device.superblock.fields.free_list_head,
        ));
        Ok(device)
    }

    pub fn close(mut self) -> Result<(), DeviceError> {
        self.superblock.set_dirty(false);
        self.persist_superblock()?;
        self.log_verbose("device closed cleanly");
        // Prevent Drop from running (we already cleared dirty)
        std::mem::forget(self);
        Ok(())
    }

    pub fn superblock(&self) -> &Superblock {
        &self.superblock
    }

    pub fn total_blocks(&self) -> u32 {
        self.superblock.fields.total_blocks
    }

    pub fn total_capacity_bytes(&self) -> u64 {
        self.plan.total_capacity_bytes
    }

    pub fn integrity_block_count(&self) -> u32 {
        let sb = &self.superblock.fields;
        if sb.integrity_chain_head == NO_BLOCK {
            0
        } else {
            let redir_start = sb.redirection_table_start;
            if redir_start == NO_BLOCK {
                self.total_blocks().saturating_sub(1)
            } else {
                redir_start - sb.integrity_chain_head
            }
        }
    }

    pub fn redirection_block_count(&self) -> u32 {
        self.superblock.fields.redirection_table_length
    }

    pub fn file_table_block_count(&self) -> u32 {
        self.superblock.fields.file_table_length
    }

    pub fn data_region_start(&self) -> u32 {
        let sb = &self.superblock.fields;
        if sb.file_table_start != NO_BLOCK {
            sb.file_table_start + sb.file_table_length
        } else if sb.redirection_table_start != NO_BLOCK {
            sb.redirection_table_start + sb.redirection_table_length
        } else if sb.integrity_chain_head != NO_BLOCK {
            sb.integrity_chain_head + compute_integrity_block_count(self.total_blocks())
        } else {
            1
        }
    }

    pub fn read_block(&self, block_index: u32) -> Result<Vec<u8>, DeviceError> {
        self.ensure_data_block(block_index)?;

        // Unmapped logicals present as zeros — they have no live data
        // (never written, or freed). NBD/ext4 and the file layer both
        // rely on this; the underlying physical doesn't exist as far as
        // this logical is concerned.
        let Some(physical) = self.redirection.logical_to_physical(block_index) else {
            self.log_verbose(format!("read unmapped logical={} → zeros", block_index));
            return Ok(vec![0_u8; crate::BLOCK_SIZE]);
        };

        let bytes = self.read_physical_block_raw(physical)?;
        self.verify_block_crc(block_index, &bytes)?;
        self.log_verbose(format!(
            "read block logical={} physical={}",
            block_index, physical
        ));
        Ok(bytes)
    }

    pub fn write_block(&mut self, block_index: u32, data: &[u8]) -> Result<(), DeviceError> {
        self.write_block_inner(block_index, data, 0)
    }

    #[doc(hidden)]
    pub fn write_block_with_crash_after(
        &mut self,
        block_index: u32,
        data: &[u8],
        crash_point: CrashPoint,
    ) -> Result<(), DeviceError> {
        self.write_block_inner(block_index, data, crash_point.as_stop_phase())
    }

    /// The `stop_after_phase` parameter controls early exit for crash testing:
    ///   0 = run to completion (normal write)
    ///   1 = stop after flush 1 (shadow data durable, redirection not flipped)
    ///   2 = stop after flush 2 (redirection flipped, old physical not freed)
    ///
    /// Writes split on whether `block_index` is already mapped:
    ///   - Unmapped (first write): pop a fresh physical from the free list,
    ///     write data, bind the logical to that physical. No old data to
    ///     preserve; orphan-scan reclaims if we crash before binding.
    ///   - Mapped (overwrite): full shadow-copy. Logical and physical
    ///     namespaces are separate, so the popped shadow physical can never
    ///     collide with the logical's address — phase 3 always reclaims the
    ///     old physical without conditionals.
    fn write_block_inner(
        &mut self,
        block_index: u32,
        data: &[u8],
        stop_after_phase: u8,
    ) -> Result<(), DeviceError> {
        self.ensure_data_block(block_index)?;
        if data.len() != crate::BLOCK_SIZE {
            return Err(DeviceError::InvalidBlockWriteLength {
                expected: crate::BLOCK_SIZE,
                actual: data.len(),
            });
        }

        let Some(old_physical) = self.redirection.logical_to_physical(block_index) else {
            // First write to an unmapped logical. Allocate a physical from
            // the free list, write data, bind. `stop_after_phase` is ignored
            // — crash tests target the shadow path only.
            let physical = self.pop_free_block().ok_or(DeviceError::OutOfSpace)?;
            self.write_physical_block_raw(physical, data)?;
            self.update_block_crc(block_index, data)?;
            self.redirection.set_mapping(block_index, physical);
            self.persist_redirection_block_for(block_index)?;
            self.persist_superblock()?;
            self.flush()?;
            self.log_verbose(format!(
                "first write: block={} → physical={}",
                block_index, physical
            ));
            return Ok(());
        };

        let shadow = self.pop_free_block().ok_or(DeviceError::OutOfSpace)?;

        // Phase 1: shadow data written + free-list pop durable. CRC for
        // `block_index` NOT updated — the integrity table still reflects the
        // old data at `old_physical`. If we crash here, orphan-scan reclaims
        // the shadow and the old block remains authoritative.
        self.write_physical_block_raw(shadow, data)?;
        self.persist_superblock()?;
        self.flush()?;
        self.log_verbose(format!(
            "shadow flush 1: block={} shadow_physical={}",
            block_index, shadow
        ));

        if stop_after_phase == 1 {
            return Ok(());
        }

        // Phase 2: flip redirection + update CRC atomically.
        self.redirection.set_mapping(block_index, shadow);
        self.update_block_crc(block_index, data)?;
        self.persist_redirection_block_for(block_index)?;
        self.flush()?;
        self.log_verbose(format!(
            "shadow flush 2: redirection flipped block={} old_physical={} new_physical={}",
            block_index, old_physical, shadow
        ));

        if stop_after_phase == 2 {
            return Ok(());
        }

        // Phase 3: reclaim the old physical. Always — there's no aliasing
        // risk with split namespaces, the freed physical is just storage.
        self.push_free_block(old_physical)?;
        self.persist_superblock()?;
        self.flush()?;

        Ok(())
    }

    /// Reserve an unused logical block index. Returns `OutOfSpace` if every
    /// data-region logical is already mapped or pending. Reservation is
    /// in-memory only — write_block transitions the logical from "reserved"
    /// to "mapped" on first write by allocating a physical from the free
    /// list. A crash between alloc and first write loses the reservation
    /// (the logical is unmapped on reopen, so it's re-issuable cleanly).
    pub fn alloc_block(&mut self) -> Result<u32, DeviceError> {
        let data_start = self.data_region_start();
        let total = self.total_blocks();
        for logical in data_start..total {
            if self.redirection.is_mapped(logical) || self.reserved_logicals.contains(&logical) {
                continue;
            }
            self.reserved_logicals.insert(logical);
            self.log_verbose(format!("alloc_block reserved logical {}", logical));
            return Ok(logical);
        }
        Err(DeviceError::OutOfSpace)
    }

    pub fn free_block(&mut self, block_index: u32) -> Result<(), DeviceError> {
        self.ensure_data_block(block_index)?;

        let physical = self.redirection.logical_to_physical(block_index);

        // Drop the in-memory reservation if any.
        self.reserved_logicals.remove(&block_index);

        // Return the backing physical to the free list (if mapped). Logical
        // L itself is *not* a physical, so we only push what's actually
        // backing the logical.
        if let Some(phys) = physical {
            self.push_free_block(phys)?;
            self.redirection.clear(block_index);
            self.persist_redirection_block_for(block_index)?;
            self.persist_superblock()?;
            self.flush()?;
            self.log_verbose(format!(
                "freed block logical={} physical={} new_free_head={}",
                block_index, phys, self.superblock.fields.free_list_head
            ));
        } else {
            self.log_verbose(format!(
                "freed unmapped logical={} (no physical to reclaim)",
                block_index
            ));
        }
        Ok(())
    }

    pub fn used_blocks(&self) -> Result<u32, DeviceError> {
        let free = self.free_blocks()?;
        Ok(self.total_blocks().saturating_sub(free))
    }

    /// True iff `logical` currently maps to a physical (i.e. has live data).
    /// Used by the NBD server to log per-write "first write vs. overwrite"
    /// information for diagnosis.
    pub fn is_logical_written(&self, logical: u32) -> bool {
        self.redirection.is_mapped(logical)
    }

    /// Diagnostic accessor: read a physical block as the stego layer
    /// decodes it (no redirection, no CRC check). Used by `llmdb dump-block`.
    pub fn read_physical_block_for_diag(
        &self,
        block_index: u32,
    ) -> Result<Vec<u8>, DeviceError> {
        self.read_physical_block_raw(block_index)
    }

    pub fn free_blocks(&self) -> Result<u32, DeviceError> {
        let mut free_count = 0_u32;
        let mut current = self.superblock.fields.free_list_head;

        while current != NO_BLOCK {
            let bytes = self.read_physical_block_raw(current)?;
            let next = freelist::decode_next(&bytes)?;
            current = next;
            free_count = free_count.saturating_add(1);
        }

        Ok(free_count)
    }

    pub fn verify_integrity(&self) -> Result<Vec<u32>, DeviceError> {
        let mut corrupted = Vec::new();
        for block_index in self.data_region_start()..self.total_blocks() {
            // Unmapped logicals have no CRC contract — they read as zeros.
            let Some(physical) = self.redirection.logical_to_physical(block_index) else {
                continue;
            };
            let bytes = self.read_physical_block_raw(physical)?;
            match self.verify_block_crc(block_index, &bytes) {
                Ok(()) => {}
                Err(DeviceError::IntegrityMismatch { .. }) => corrupted.push(block_index),
                Err(error) => return Err(error),
            }
        }
        Ok(corrupted)
    }

    pub fn flush(&mut self) -> Result<(), DeviceError> {
        self.log_verbose("flushing mmap");
        self.mmap.flush()?;
        Ok(())
    }

    /// Zero every stego block (destroying any residual user data) and then
    /// re-initialize the device in place. After `wipe`, the handle is valid
    /// and reflects a freshly formatted device — equivalent to `init` with
    /// the same allocation mode, but with the guarantee that no old stego
    /// bits survive in the non-metadata remainder of any block.
    pub fn wipe(&mut self) -> Result<(), DeviceError> {
        let total = self.total_blocks();
        let zero = vec![0_u8; crate::BLOCK_SIZE];
        for block_index in 0..total {
            self.write_physical_block_raw(block_index, &zero)?;
        }
        self.flush()?;
        self.format()?;
        self.superblock.set_dirty(true);
        self.persist_superblock()?;
        self.log_verbose("device wiped and re-initialized");
        Ok(())
    }

    fn open_internal(
        path: impl AsRef<Path>,
        mode: AllocationMode,
        options: DeviceOptions,
    ) -> Result<Self, DeviceError> {
        let parsed = parser::parse_path(&path)?;
        let plan = build_allocation_plan(&parsed.tensors, mode);
        let slots = build_tensor_byte_slots(&parsed, &plan)?;

        if plan.total_capacity_bytes < crate::BLOCK_SIZE as u64 {
            return Err(DeviceError::InsufficientCapacity(plan.total_capacity_bytes));
        }

        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let mmap = unsafe { MmapMut::map_mut(&file)? };

        let total_blocks =
            u32::try_from(plan.total_capacity_bytes / crate::BLOCK_SIZE as u64).unwrap_or(0);

        let device = Self {
            mmap,
            plan,
            slots,
            superblock: Superblock::new(SuperblockFields {
                total_blocks: 0,
                free_list_head: NO_BLOCK,
                integrity_chain_head: NO_BLOCK,
                redirection_table_start: NO_BLOCK,
                redirection_table_length: 0,
                file_table_start: NO_BLOCK,
                file_table_length: 0,
                flags: 0,
                quant_profile: 0,
            }),
            redirection: RedirectionTable::empty(total_blocks),
            reserved_logicals: HashSet::new(),
            verbose: options.verbose,
        };

        device.log_verbose(format!(
            "parsed gguf: tensors={}, usable_capacity_bytes={}",
            device.plan.tensors.len(),
            device.plan.total_capacity_bytes
        ));

        Ok(device)
    }

    fn format(&mut self) -> Result<(), DeviceError> {
        let total_blocks = u32::try_from(self.plan.total_capacity_bytes / crate::BLOCK_SIZE as u64)
            .map_err(|_| DeviceError::CapacityOverflow(self.plan.total_capacity_bytes))?;

        let layout = compute_layout(total_blocks)?;

        let quant_types: Vec<_> = self.plan.tensors.iter().map(|t| t.quant_type).collect();

        self.superblock = Superblock::new(SuperblockFields {
            total_blocks,
            free_list_head: layout.data_region_start,
            integrity_chain_head: if layout.integrity_count > 0 {
                layout.integrity_start
            } else {
                NO_BLOCK
            },
            redirection_table_start: layout.redirection_start,
            redirection_table_length: layout.redirection_count,
            file_table_start: layout.file_table_start,
            file_table_length: layout.file_table_count,
            flags: if self.plan.mode == AllocationMode::Lobotomy {
                0x01
            } else {
                0x00
            },
            quant_profile: encode_quant_profile(&quant_types),
        });

        self.redirection = RedirectionTable::empty(total_blocks);
        self.reserved_logicals.clear();

        self.log_verbose(format!(
            "formatting device: total_blocks={}, integrity={}, redir={}, filetable={}, \
             data_region_start={}",
            total_blocks,
            layout.integrity_count,
            layout.redirection_count,
            layout.file_table_count,
            layout.data_region_start,
        ));

        // Write free list chain for data blocks
        let chain = freelist::build_free_chain(layout.data_region_start, total_blocks);
        for (block_index, bytes) in &chain {
            self.write_physical_block_raw(*block_index, bytes)?;
        }

        // Initialize integrity blocks
        self.initialize_integrity_blocks(&layout, total_blocks)?;

        // Write redirection table (identity mapping)
        let redir_blocks = self.redirection.encode();
        for (offset, raw) in redir_blocks.iter().enumerate() {
            self.write_physical_block_raw(layout.redirection_start + offset as u32, raw)?;
        }

        // Write empty file table block(s)
        for offset in 0..layout.file_table_count {
            let empty = vec![0_u8; crate::BLOCK_SIZE];
            self.write_physical_block_raw(layout.file_table_start + offset, &empty)?;
        }

        self.persist_superblock()?;
        Ok(())
    }

    fn initialize_integrity_blocks(
        &mut self,
        layout: &DeviceLayout,
        total_blocks: u32,
    ) -> Result<(), DeviceError> {
        let data_block_count = total_blocks.saturating_sub(layout.data_region_start);

        for integrity_offset in 0..layout.integrity_count {
            let first_data_block =
                layout.data_region_start + integrity_offset * ENTRIES_PER_INTEGRITY_BLOCK as u32;
            let remaining = data_block_count
                .saturating_sub(integrity_offset * ENTRIES_PER_INTEGRITY_BLOCK as u32);
            let entry_count = remaining.min(ENTRIES_PER_INTEGRITY_BLOCK as u32);
            let block = IntegrityBlock {
                first_data_block,
                entry_count,
                next_integrity_block: if integrity_offset + 1 < layout.integrity_count {
                    layout.integrity_start + integrity_offset + 1
                } else {
                    NO_BLOCK
                },
                crc32_entries: vec![0_u32; entry_count as usize],
            };

            let encoded = block.encode()?;
            self.write_physical_block_raw(layout.integrity_start + integrity_offset, &encoded)?;
        }

        // Compute and store CRCs for all data blocks
        for block_index in layout.data_region_start..total_blocks {
            let bytes = self.read_physical_block_raw(block_index)?;
            self.update_block_crc_for_data(block_index, layout.data_region_start, &bytes)?;
        }

        Ok(())
    }

    fn persist_superblock(&mut self) -> Result<(), DeviceError> {
        let bytes = self.superblock.encode();
        self.log_verbose("persisting superblock");
        self.write_physical_block_raw(SUPERBLOCK_BLOCK, &bytes)?;
        self.flush()?;
        Ok(())
    }

    fn load_superblock(&self) -> Result<Superblock, DeviceError> {
        let bytes = self.read_stego_bytes(0, crate::BLOCK_SIZE)?;
        Superblock::decode(&bytes).map_err(DeviceError::Integrity)
    }

    fn load_redirection_table(&self) -> Result<RedirectionTable, DeviceError> {
        let sb = &self.superblock.fields;
        if sb.redirection_table_start == NO_BLOCK || sb.redirection_table_length == 0 {
            return Ok(RedirectionTable::empty(sb.total_blocks));
        }

        let mut raw_blocks = Vec::with_capacity(sb.redirection_table_length as usize);
        for offset in 0..sb.redirection_table_length {
            let block_index = sb.redirection_table_start + offset;
            let bytes = self.read_physical_block_raw(block_index)?;
            raw_blocks.push(bytes);
        }

        Ok(RedirectionTable::decode(&raw_blocks)?)
    }

    fn ensure_valid_block(&self, block_index: u32) -> Result<(), DeviceError> {
        if block_index >= self.total_blocks() {
            Err(DeviceError::BlockOutOfRange {
                index: block_index,
                total_blocks: self.total_blocks(),
            })
        } else {
            Ok(())
        }
    }

    fn ensure_data_block(&self, block_index: u32) -> Result<(), DeviceError> {
        self.ensure_valid_block(block_index)?;
        if block_index < self.data_region_start() {
            return Err(DeviceError::ReservedMetadataBlock {
                index: block_index,
                data_region_start: self.data_region_start(),
            });
        }
        Ok(())
    }

    pub(crate) fn read_physical_block_raw(
        &self,
        block_index: u32,
    ) -> Result<Vec<u8>, DeviceError> {
        self.read_stego_bytes(block_index as usize * crate::BLOCK_SIZE, crate::BLOCK_SIZE)
    }

    pub(crate) fn write_physical_block_raw(
        &mut self,
        block_index: u32,
        data: &[u8],
    ) -> Result<(), DeviceError> {
        if data.len() != crate::BLOCK_SIZE {
            return Err(DeviceError::InvalidBlockWriteLength {
                expected: crate::BLOCK_SIZE,
                actual: data.len(),
            });
        }
        self.write_stego_bytes(block_index as usize * crate::BLOCK_SIZE, data)
    }

    fn read_stego_bytes(&self, start: usize, len: usize) -> Result<Vec<u8>, DeviceError> {
        let end = start
            .checked_add(len)
            .ok_or(DeviceError::ByteRangeOverflow)?;
        if end > self.plan.total_capacity_bytes as usize {
            return Err(DeviceError::ByteRangeOutOfBounds {
                start,
                len,
                capacity: self.plan.total_capacity_bytes as usize,
            });
        }

        let mut out = vec![0_u8; len];
        for slot in &self.slots {
            if end <= slot.byte_start || start >= slot.byte_end {
                continue;
            }
            let overlap_start = start.max(slot.byte_start);
            let overlap_end = end.min(slot.byte_end);
            let within_slot_start = overlap_start - slot.byte_start;
            let within_slot_len = overlap_end - overlap_start;
            let dst_start = overlap_start - start;
            let dst_end = overlap_end - start;
            let piece = self.read_slot_range(slot, within_slot_start, within_slot_len)?;
            out[dst_start..dst_end].copy_from_slice(&piece);
        }

        Ok(out)
    }

    fn write_stego_bytes(&mut self, start: usize, data: &[u8]) -> Result<(), DeviceError> {
        let end = start
            .checked_add(data.len())
            .ok_or(DeviceError::ByteRangeOverflow)?;
        if end > self.plan.total_capacity_bytes as usize {
            return Err(DeviceError::ByteRangeOutOfBounds {
                start,
                len: data.len(),
                capacity: self.plan.total_capacity_bytes as usize,
            });
        }

        for slot_index in 0..self.slots.len() {
            let slot = self.slots[slot_index].clone();
            if end <= slot.byte_start || start >= slot.byte_end {
                continue;
            }
            let overlap_start = start.max(slot.byte_start);
            let overlap_end = end.min(slot.byte_end);
            let within_slot_start = overlap_start - slot.byte_start;
            let src_start = overlap_start - start;
            let src_end = overlap_end - start;
            self.write_slot_range(&slot, within_slot_start, &data[src_start..src_end])?;
        }

        Ok(())
    }

    /// Update only the quant blocks in `slot` that back the payload byte
    /// range `[start_in_slot, start_in_slot + data.len())`. Replaces the
    /// former decode-whole-tensor / splice / encode-whole-tensor cycle
    /// — see `src/stego/packing/mod.rs::blockwise_write_range`.
    fn write_slot_range(
        &mut self,
        slot: &TensorByteSlot,
        start_in_slot: usize,
        data: &[u8],
    ) -> Result<(), DeviceError> {
        let storage =
            &mut self.mmap[slot.file_offset..slot.file_offset + slot.storage_len];
        match slot.quant_type {
            GgufQuantType::Q8_0 => {
                packing::q8_0::write_stego_range(storage, start_in_slot, data)?
            }
            GgufQuantType::Q6K => {
                packing::q6_k::write_stego_range(storage, start_in_slot, data)?
            }
            GgufQuantType::Q5K => {
                packing::q5_k::write_stego_range(storage, start_in_slot, data)?
            }
            GgufQuantType::Q4K => {
                packing::q4_k::write_stego_range(storage, start_in_slot, data)?
            }
            GgufQuantType::Q3K => {
                packing::q3_k::write_stego_range(storage, start_in_slot, data)?
            }
            GgufQuantType::F16 => {
                packing::float::write_f16_range(storage, start_in_slot, data)?
            }
            GgufQuantType::F32 => {
                packing::float::write_f32_range(storage, start_in_slot, data)?
            }
            _ => return Err(DeviceError::UnsupportedQuantType(slot.quant_type)),
        }
        Ok(())
    }

    fn read_slot_range(
        &self,
        slot: &TensorByteSlot,
        start_in_slot: usize,
        len: usize,
    ) -> Result<Vec<u8>, DeviceError> {
        let storage = &self.mmap[slot.file_offset..slot.file_offset + slot.storage_len];
        Ok(match slot.quant_type {
            GgufQuantType::Q8_0 => packing::q8_0::read_stego_range(storage, start_in_slot, len)?,
            GgufQuantType::Q6K => packing::q6_k::read_stego_range(storage, start_in_slot, len)?,
            GgufQuantType::Q5K => packing::q5_k::read_stego_range(storage, start_in_slot, len)?,
            GgufQuantType::Q4K => packing::q4_k::read_stego_range(storage, start_in_slot, len)?,
            GgufQuantType::Q3K => packing::q3_k::read_stego_range(storage, start_in_slot, len)?,
            GgufQuantType::F16 => {
                packing::float::read_f16_range(storage, start_in_slot, len)?
            }
            GgufQuantType::F32 => {
                packing::float::read_f32_range(storage, start_in_slot, len)?
            }
            _ => return Err(DeviceError::UnsupportedQuantType(slot.quant_type)),
        })
    }

    fn verify_block_crc(&self, block_index: u32, data: &[u8]) -> Result<(), DeviceError> {
        let Some(expected) = self.read_crc_entry(block_index)? else {
            return Ok(());
        };
        let actual = crc32(data);
        if expected != actual {
            return Err(DeviceError::IntegrityMismatch {
                block_index,
                expected,
                actual,
            });
        }
        Ok(())
    }

    fn update_block_crc(&mut self, block_index: u32, data: &[u8]) -> Result<(), DeviceError> {
        self.update_block_crc_for_data(block_index, self.data_region_start(), data)
    }

    fn update_block_crc_for_data(
        &mut self,
        block_index: u32,
        data_region_start: u32,
        data: &[u8],
    ) -> Result<(), DeviceError> {
        if block_index < data_region_start || block_index >= self.total_blocks() {
            return Ok(());
        }

        let crc = crc32(data);
        let zero_based = block_index - data_region_start;
        let integrity_offset = zero_based / ENTRIES_PER_INTEGRITY_BLOCK as u32;
        let entry_index = (zero_based % ENTRIES_PER_INTEGRITY_BLOCK as u32) as usize;

        let integrity_block_index = self.superblock.fields.integrity_chain_head + integrity_offset;
        let integrity_bytes = self.read_physical_block_raw(integrity_block_index)?;
        let mut integrity_block = IntegrityBlock::decode(&integrity_bytes)?;
        if entry_index >= integrity_block.crc32_entries.len() {
            return Err(DeviceError::MissingIntegrityEntry {
                block_index,
                integrity_block_index,
            });
        }
        integrity_block.crc32_entries[entry_index] = crc;
        let encoded = integrity_block.encode()?;
        self.write_physical_block_raw(integrity_block_index, &encoded)?;
        Ok(())
    }

    fn read_crc_entry(&self, block_index: u32) -> Result<Option<u32>, DeviceError> {
        let data_start = self.data_region_start();
        if block_index < data_start || block_index >= self.total_blocks() {
            return Ok(None);
        }

        let zero_based = block_index - data_start;
        let integrity_offset = zero_based / ENTRIES_PER_INTEGRITY_BLOCK as u32;
        let entry_index = (zero_based % ENTRIES_PER_INTEGRITY_BLOCK as u32) as usize;

        let integrity_block_index = self.superblock.fields.integrity_chain_head + integrity_offset;
        let integrity_bytes = self.read_physical_block_raw(integrity_block_index)?;
        let integrity_block = IntegrityBlock::decode(&integrity_bytes)?;
        integrity_block
            .crc32_entries
            .get(entry_index)
            .copied()
            .map(Some)
            .ok_or(DeviceError::MissingIntegrityEntry {
                block_index,
                integrity_block_index,
            })
    }

    /// Pop the head of the free list. The free list stores PHYSICAL block
    /// indices — reads/writes go directly to physical stego space, bypassing
    /// the redirection table.
    fn pop_free_block(&mut self) -> Option<u32> {
        let head = self.superblock.fields.free_list_head;
        if head == NO_BLOCK {
            return None;
        }

        let bytes = match self.read_physical_block_raw(head) {
            Ok(bytes) => bytes,
            Err(_) => return None,
        };
        let next = match freelist::decode_next(&bytes) {
            Ok(next) => next,
            Err(_) => return None,
        };
        self.superblock.fields.free_list_head = next;
        Some(head)
    }

    /// Push a PHYSICAL block index onto the free list. Writes directly to
    /// the physical stego space, bypassing the redirection table. With the
    /// split namespace, pushing physical X never touches `redirection[X]` —
    /// physicals and logicals are independent, and any logical that may
    /// happen to share the index `X` is unaffected by reclaiming the
    /// underlying storage slot.
    fn push_free_block(&mut self, physical_block: u32) -> Result<(), DeviceError> {
        let previous_head = self.superblock.fields.free_list_head;
        let bytes = freelist::encode_head(previous_head);
        self.write_physical_block_raw(physical_block, &bytes)?;
        self.superblock.fields.free_list_head = physical_block;
        Ok(())
    }

    fn persist_redirection_block_for(&mut self, block_index: u32) -> Result<(), DeviceError> {
        let redir_start = self.superblock.fields.redirection_table_start;
        if redir_start == NO_BLOCK {
            return Ok(());
        }

        let redir_block_offset =
            block_index as usize / crate::stego::redirection::ENTRIES_PER_BLOCK;
        let encoded = self.redirection.encode();
        if let Some(raw) = encoded.get(redir_block_offset) {
            self.write_physical_block_raw(redir_start + redir_block_offset as u32, raw)?;
        }
        Ok(())
    }

    fn recover(&mut self) -> Result<(), DeviceError> {
        let data_start = self.data_region_start();
        let total = self.total_blocks();

        // In-use physicals = those that some logical's redirection points to.
        // With the split namespace, "logical L is alive" and "physical X is
        // in use" are independent — we only care about physicals here for
        // orphan detection.
        let mut in_use = HashSet::new();
        for logical in data_start..total {
            if let Some(physical) = self.redirection.logical_to_physical(logical) {
                in_use.insert(physical);
            }
        }

        // Physicals currently on the free list.
        let mut free_set = HashSet::new();
        let mut current = self.superblock.fields.free_list_head;
        while current != NO_BLOCK {
            if current < data_start || current >= total {
                break;
            }
            if !free_set.insert(current) {
                break; // cycle
            }
            let bytes = self.read_physical_block_raw(current)?;
            current = freelist::decode_next(&bytes)?;
        }

        // Orphan physicals: in the data region, not referenced by any
        // logical, not on the free list. These are shadows from interrupted
        // writes or first-write physicals popped before the redirection
        // bind landed.
        let mut orphans = Vec::new();
        for block in data_start..total {
            if !in_use.contains(&block) && !free_set.contains(&block) {
                orphans.push(block);
            }
        }

        if !orphans.is_empty() {
            self.log_verbose(format!(
                "recovery: reclaiming {} orphan block(s)",
                orphans.len()
            ));
            for orphan in orphans {
                self.push_free_block(orphan)?;
            }
            self.persist_superblock()?;
            self.flush()?;
        }

        self.superblock.set_dirty(false);
        self.persist_superblock()?;

        self.log_verbose("recovery complete");
        Ok(())
    }

    fn log_verbose(&self, message: impl AsRef<str>) {
        if self.verbose {
            eprintln!("[llmdb:device] {}", message.as_ref());
        }
    }
}

impl Drop for StegoDevice {
    fn drop(&mut self) {
        self.superblock.set_dirty(false);
        let _ = self.persist_superblock();
    }
}

// -- Layout computation --

struct DeviceLayout {
    integrity_start: u32,
    integrity_count: u32,
    redirection_start: u32,
    redirection_count: u32,
    file_table_start: u32,
    file_table_count: u32,
    data_region_start: u32,
}

fn compute_layout(total_blocks: u32) -> Result<DeviceLayout, DeviceError> {
    // Iteratively solve: metadata = 1 (super) + I + R + F, where I and R
    // depend on data_blocks = total - metadata.
    let file_table_count = FILE_TABLE_INITIAL_BLOCKS;
    let mut integrity_count = 0_u32;
    let mut redirection_count = 0_u32;

    loop {
        let metadata = 1 + integrity_count + redirection_count + file_table_count;
        let data_blocks = total_blocks.saturating_sub(metadata);

        let needed_integrity = if data_blocks == 0 {
            0
        } else {
            (data_blocks as usize).div_ceil(ENTRIES_PER_INTEGRITY_BLOCK) as u32
        };
        let needed_redirection = if total_blocks == 0 {
            0
        } else {
            (total_blocks as usize).div_ceil(crate::stego::redirection::ENTRIES_PER_BLOCK) as u32
        };

        if integrity_count >= needed_integrity && redirection_count >= needed_redirection {
            break;
        }
        integrity_count = needed_integrity;
        redirection_count = needed_redirection;
    }

    let integrity_start = 1;
    let redirection_start = integrity_start + integrity_count;
    let file_table_start = redirection_start + redirection_count;
    let data_region_start = file_table_start + file_table_count;

    if data_region_start >= total_blocks {
        return Err(DeviceError::InsufficientCapacityForMetadata {
            total_blocks,
            integrity_blocks: integrity_count,
        });
    }

    Ok(DeviceLayout {
        integrity_start,
        integrity_count,
        redirection_start,
        redirection_count,
        file_table_start,
        file_table_count,
        data_region_start,
    })
}

fn compute_integrity_block_count(total_blocks: u32) -> u32 {
    let mut integrity_count = 0_u32;
    loop {
        let data_blocks = total_blocks.saturating_sub(1 + integrity_count);
        let needed = if data_blocks == 0 {
            0
        } else {
            (data_blocks as usize).div_ceil(ENTRIES_PER_INTEGRITY_BLOCK) as u32
        };
        if integrity_count >= needed {
            return integrity_count;
        }
        integrity_count = needed;
    }
}

// -- Tensor byte slots --

#[derive(Debug, Clone, PartialEq, Eq)]
struct TensorByteSlot {
    name: String,
    quant_type: GgufQuantType,
    file_offset: usize,
    storage_len: usize,
    capacity_bytes: usize,
    byte_start: usize,
    byte_end: usize,
}

fn build_tensor_byte_slots(
    parsed: &GgufFile,
    plan: &AllocationPlan,
) -> Result<Vec<TensorByteSlot>, DeviceError> {
    let parsed_by_name: HashMap<_, _> = parsed
        .tensors
        .iter()
        .map(|tensor| (tensor.name.as_str(), tensor))
        .collect();

    let mut byte_cursor = 0_usize;
    let mut slots = Vec::with_capacity(plan.tensors.len());

    for planned in &plan.tensors {
        let parsed_tensor = parsed_by_name
            .get(planned.name.as_str())
            .ok_or_else(|| DeviceError::MissingTensor(planned.name.clone()))?;
        let file_offset = parsed_tensor
            .absolute_offset(parsed.tensor_data_offset)
            .ok_or_else(|| DeviceError::OffsetOverflow(planned.name.clone()))?;
        let layout = tensor_storage_layout(planned.quant_type, planned.weight_count)?;
        let capacity_bytes = usize::try_from(planned.capacity_bytes_floor)
            .map_err(|_| DeviceError::CapacityOverflow(planned.capacity_bytes_floor))?;

        slots.push(TensorByteSlot {
            name: planned.name.clone(),
            quant_type: planned.quant_type,
            file_offset: usize::try_from(file_offset)
                .map_err(|_| DeviceError::OffsetOverflow(planned.name.clone()))?,
            storage_len: layout.storage_len,
            capacity_bytes,
            byte_start: byte_cursor,
            byte_end: byte_cursor + capacity_bytes,
        });
        byte_cursor += capacity_bytes;
    }

    Ok(slots)
}

struct TensorStorageLayout {
    storage_len: usize,
}

fn tensor_storage_layout(
    quant_type: GgufQuantType,
    weight_count: u64,
) -> Result<TensorStorageLayout, DeviceError> {
    let weight_count =
        usize::try_from(weight_count).map_err(|_| DeviceError::CapacityOverflow(weight_count))?;
    let storage_len = match quant_type {
        GgufQuantType::Q8_0 => chunk_storage_len(weight_count, 32, packing::q8_0::BLOCK_BYTES)?,
        GgufQuantType::Q6K => chunk_storage_len(weight_count, 256, packing::q6_k::BLOCK_BYTES)?,
        GgufQuantType::Q5K => chunk_storage_len(weight_count, 256, packing::q5_k::BLOCK_BYTES)?,
        GgufQuantType::Q4K => chunk_storage_len(weight_count, 256, packing::q4_k::BLOCK_BYTES)?,
        GgufQuantType::Q3K => chunk_storage_len(weight_count, 256, packing::q3_k::BLOCK_BYTES)?,
        GgufQuantType::F16 => weight_count * 2,
        GgufQuantType::F32 => weight_count * 4,
        _ => return Err(DeviceError::UnsupportedQuantType(quant_type)),
    };

    Ok(TensorStorageLayout { storage_len })
}

fn chunk_storage_len(
    weight_count: usize,
    weights_per_chunk: usize,
    bytes_per_chunk: usize,
) -> Result<usize, DeviceError> {
    if weight_count % weights_per_chunk != 0 {
        return Err(DeviceError::InvalidTensorWeightCount {
            weight_count,
            weights_per_chunk,
        });
    }

    Ok((weight_count / weights_per_chunk) * bytes_per_chunk)
}

fn crc32(data: &[u8]) -> u32 {
    let mut hasher = Hasher::new();
    hasher.update(data);
    hasher.finalize()
}

#[derive(Debug, Error)]
pub enum DeviceError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("gguf parse error: {0}")]
    Parse(#[from] parser::ParseError),
    #[error("metadata error: {0}")]
    Integrity(#[from] IntegrityError),
    #[error("redirection error: {0}")]
    Redirection(#[from] RedirectionError),
    #[error("packing error: {0}")]
    Packing(#[from] PackingError),
    #[error("insufficient capacity: {0} bytes")]
    InsufficientCapacity(u64),
    #[error(
        "insufficient capacity for metadata: total_blocks={total_blocks}, integrity_blocks={integrity_blocks}"
    )]
    InsufficientCapacityForMetadata {
        total_blocks: u32,
        integrity_blocks: u32,
    },
    #[error("capacity overflow: {0}")]
    CapacityOverflow(u64),
    #[error("missing tensor in parsed gguf: {0}")]
    MissingTensor(String),
    #[error("offset overflow for tensor {0}")]
    OffsetOverflow(String),
    #[error("unsupported quant type: {0:?}")]
    UnsupportedQuantType(GgufQuantType),
    #[error("invalid tensor weight count {weight_count} for chunk size {weights_per_chunk}")]
    InvalidTensorWeightCount {
        weight_count: usize,
        weights_per_chunk: usize,
    },
    #[error("block index {index} out of range, total blocks {total_blocks}")]
    BlockOutOfRange { index: u32, total_blocks: u32 },
    #[error("block {index} is reserved metadata, first data block is {data_region_start}")]
    ReservedMetadataBlock { index: u32, data_region_start: u32 },
    #[error("invalid block write length: expected {expected}, got {actual}")]
    InvalidBlockWriteLength { expected: usize, actual: usize },
    #[error("byte range overflow")]
    ByteRangeOverflow,
    #[error("byte range out of bounds: start {start}, len {len}, capacity {capacity}")]
    ByteRangeOutOfBounds {
        start: usize,
        len: usize,
        capacity: usize,
    },
    #[error("device is out of free blocks")]
    OutOfSpace,
    #[error("integrity mismatch on block {block_index}: expected {expected:#x}, got {actual:#x}")]
    IntegrityMismatch {
        block_index: u32,
        expected: u32,
        actual: u32,
    },
    #[error(
        "missing integrity entry for block {block_index} in integrity block {integrity_block_index}"
    )]
    MissingIntegrityEntry {
        block_index: u32,
        integrity_block_index: u32,
    },
}
