use std::collections::HashMap;
use std::fs::OpenOptions;
use std::path::Path;

use crc32fast::Hasher;
use memmap2::MmapMut;
use thiserror::Error;

use crate::gguf::parser::{self, GgufFile};
use crate::gguf::quant::GgufQuantType;
use crate::stego::integrity::{
    ENTRIES_PER_INTEGRITY_BLOCK, FreeListBlock, IntegrityBlock, IntegrityError, NO_BLOCK,
    PendingMetadataOp, Superblock, SuperblockFields,
};
use crate::stego::packing::{self, PackingError};
use crate::stego::planner::{AllocationMode, AllocationPlan, build_allocation_plan};

const PRIMARY_SUPERBLOCK_BLOCK: u32 = 0;
const BACKUP_SUPERBLOCK_BLOCK: u32 = 1;

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
        device.recover_if_needed()?;
        device.log_verbose(format!(
            "opened device: total_blocks={}, integrity_head={}, free_list_head={}, shadow_block={}, generation={}",
            device.superblock.fields.total_blocks,
            device.superblock.fields.integrity_chain_head,
            device.superblock.fields.free_list_head,
            device.superblock.fields.shadow_block,
            device.superblock.fields.generation
        ));
        Ok(device)
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
        integrity_block_count(self.total_blocks())
    }

    pub fn data_region_start(&self) -> u32 {
        2 + self.integrity_block_count()
    }

    pub fn shadow_block(&self) -> u32 {
        self.superblock.fields.shadow_block
    }

    pub fn read_block(&self, block_index: u32) -> Result<Vec<u8>, DeviceError> {
        self.ensure_data_block(block_index)?;
        let bytes = self.read_logical_block_raw(block_index)?;
        self.verify_block_crc(block_index, &bytes)?;
        self.log_verbose(format!("read data block {}", block_index));
        Ok(bytes)
    }

    pub fn write_block(&mut self, block_index: u32, data: &[u8]) -> Result<(), DeviceError> {
        self.ensure_data_block(block_index)?;
        if data.len() != crate::BLOCK_SIZE {
            return Err(DeviceError::InvalidBlockWriteLength {
                expected: crate::BLOCK_SIZE,
                actual: data.len(),
            });
        }

        let shadow_block = self.shadow_block_index()?;
        let pending_crc = self.stage_pending_write(block_index, data)?;
        self.write_logical_block_raw(block_index, data)?;
        self.update_block_crc(block_index, data)?;
        self.flush()?;
        self.clear_pending_write()?;
        self.log_verbose(format!("wrote data block {}", block_index));
        self.log_verbose(format!(
            "committed shadow block {} into data block {} with crc {pending_crc:#x}",
            shadow_block, block_index
        ));
        Ok(())
    }

    pub fn alloc_block(&mut self) -> Result<u32, DeviceError> {
        let head = self.superblock.fields.free_list_head;
        if head == NO_BLOCK {
            return Err(DeviceError::OutOfSpace);
        }

        let head_block = self.read_logical_block_raw(head)?;
        self.verify_block_crc(head, &head_block)?;
        let free_block = FreeListBlock::decode(&head_block)?;
        self.stage_pending_metadata(
            PendingMetadataOp::AllocHeadAdvance,
            head,
            free_block.next_free_block,
        )?;
        self.superblock.fields.free_list_head = free_block.next_free_block;
        self.clear_pending_metadata_fields();
        self.persist_superblocks()?;
        self.log_verbose(format!(
            "allocated block {} next_free={}",
            head, self.superblock.fields.free_list_head
        ));
        Ok(head)
    }

    pub fn free_block(&mut self, block_index: u32) -> Result<(), DeviceError> {
        self.ensure_data_block(block_index)?;

        let previous_head = self.superblock.fields.free_list_head;
        self.stage_pending_metadata(PendingMetadataOp::FreeHeadPush, block_index, previous_head)?;
        let free_block = FreeListBlock {
            next_free_block: previous_head,
        };
        let bytes = free_block.encode();
        self.write_block(block_index, &bytes)?;
        self.superblock.fields.free_list_head = block_index;
        self.clear_pending_metadata_fields();
        self.persist_superblocks()?;
        self.log_verbose(format!(
            "freed block {} new_free_head={}",
            block_index, self.superblock.fields.free_list_head
        ));
        Ok(())
    }

    pub fn used_blocks(&self) -> Result<u32, DeviceError> {
        let mut free_count = 0_u32;
        let mut current = self.superblock.fields.free_list_head;

        while current != NO_BLOCK {
            self.ensure_data_block(current)?;
            let bytes = self.read_logical_block_raw(current)?;
            self.verify_block_crc(current, &bytes)?;
            let free_block = FreeListBlock::decode(&bytes)?;
            current = free_block.next_free_block;
            free_count = free_count.saturating_add(1);
        }

        Ok(self.total_blocks().saturating_sub(free_count))
    }

    pub fn verify_integrity(&self) -> Result<Vec<u32>, DeviceError> {
        let mut corrupted = Vec::new();
        for block_index in self.data_region_start()..self.shadow_block_index()? {
            let bytes = self.read_logical_block_raw(block_index)?;
            match self.verify_block_crc(block_index, &bytes) {
                Ok(()) => {}
                Err(DeviceError::IntegrityMismatch { .. }) => corrupted.push(block_index),
                Err(error) => return Err(error),
            }
        }
        Ok(corrupted)
    }

    pub fn reserve_all_data_blocks(&mut self) -> Result<(), DeviceError> {
        self.superblock.fields.free_list_head = NO_BLOCK;
        self.clear_pending_metadata_fields();
        self.persist_superblocks()?;
        self.log_verbose("reserved entire data region for linear ownership");
        Ok(())
    }

    pub fn flush(&mut self) -> Result<(), DeviceError> {
        self.log_verbose("flushing mmap");
        self.mmap.flush()?;
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

        let device = Self {
            mmap,
            plan,
            slots,
            superblock: Superblock::new(SuperblockFields {
                total_blocks: 0,
                free_list_head: NO_BLOCK,
                table_directory_block: NO_BLOCK,
                integrity_chain_head: NO_BLOCK,
                wal_region_start: NO_BLOCK,
                wal_region_length: 0,
                shadow_block: NO_BLOCK,
                pending_target_block: NO_BLOCK,
                pending_target_crc32: 0,
                pending_metadata_op: PendingMetadataOp::None,
                pending_metadata_block: NO_BLOCK,
                pending_metadata_aux: NO_BLOCK,
                generation: 0,
            }),
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
        let integrity_count = integrity_block_count(total_blocks);
        let data_region_start = 2 + integrity_count;
        let shadow_block = total_blocks.saturating_sub(1);
        if shadow_block <= data_region_start {
            return Err(DeviceError::InsufficientCapacityForShadow {
                total_blocks,
                integrity_blocks: integrity_count,
            });
        }
        let free_list_head = if data_region_start < shadow_block {
            data_region_start
        } else {
            NO_BLOCK
        };

        self.superblock = Superblock::new(SuperblockFields {
            total_blocks,
            free_list_head,
            table_directory_block: NO_BLOCK,
            integrity_chain_head: if integrity_count > 0 {
                BACKUP_SUPERBLOCK_BLOCK + 1
            } else {
                NO_BLOCK
            },
            wal_region_start: NO_BLOCK,
            wal_region_length: 0,
            shadow_block,
            pending_target_block: NO_BLOCK,
            pending_target_crc32: 0,
            pending_metadata_op: PendingMetadataOp::None,
            pending_metadata_block: NO_BLOCK,
            pending_metadata_aux: NO_BLOCK,
            generation: 0,
        });

        self.log_verbose(format!(
            "formatting device: total_blocks={}, integrity_blocks={}, data_region_start={}, shadow_block={}",
            total_blocks, integrity_count, data_region_start, shadow_block
        ));

        for block_index in data_region_start..shadow_block {
            let next = if block_index + 1 < shadow_block {
                block_index + 1
            } else {
                NO_BLOCK
            };
            let bytes = FreeListBlock {
                next_free_block: next,
            }
            .encode();
            self.write_logical_block_raw(block_index, &bytes)?;
        }

        self.initialize_integrity_blocks(integrity_count, data_region_start, shadow_block)?;
        self.persist_superblocks()?;
        Ok(())
    }

    fn initialize_integrity_blocks(
        &mut self,
        integrity_count: u32,
        data_region_start: u32,
        shadow_block: u32,
    ) -> Result<(), DeviceError> {
        let data_block_count = shadow_block.saturating_sub(data_region_start);

        for integrity_offset in 0..integrity_count {
            let first_data_block =
                data_region_start + integrity_offset * ENTRIES_PER_INTEGRITY_BLOCK as u32;
            let remaining = data_block_count
                .saturating_sub(integrity_offset * ENTRIES_PER_INTEGRITY_BLOCK as u32);
            let entry_count = remaining.min(ENTRIES_PER_INTEGRITY_BLOCK as u32);
            let mut crc32_entries = Vec::with_capacity(entry_count as usize);
            for entry_offset in 0..entry_count {
                let block_index = first_data_block + entry_offset;
                let next = if block_index + 1 < shadow_block {
                    block_index + 1
                } else {
                    NO_BLOCK
                };
                let bytes = FreeListBlock {
                    next_free_block: next,
                }
                .encode();
                crc32_entries.push(crc32(&bytes));
            }
            let block = IntegrityBlock {
                first_data_block,
                entry_count,
                next_integrity_block: if integrity_offset + 1 < integrity_count {
                    BACKUP_SUPERBLOCK_BLOCK + 1 + integrity_offset + 1
                } else {
                    NO_BLOCK
                },
                crc32_entries,
            };

            let encoded = block.encode()?;
            self.write_logical_block_raw(BACKUP_SUPERBLOCK_BLOCK + 1 + integrity_offset, &encoded)?;
        }

        Ok(())
    }

    fn persist_superblocks(&mut self) -> Result<(), DeviceError> {
        self.superblock.fields.generation = self.superblock.fields.generation.saturating_add(1);
        let bytes = self.superblock.encode();
        self.log_verbose(format!(
            "persisting mirrored superblocks generation={}",
            self.superblock.fields.generation
        ));
        self.write_logical_block_raw(PRIMARY_SUPERBLOCK_BLOCK, &bytes)?;
        self.flush()?;
        self.write_logical_block_raw(BACKUP_SUPERBLOCK_BLOCK, &bytes)?;
        self.flush()?;
        Ok(())
    }

    fn load_superblock(&self) -> Result<Superblock, DeviceError> {
        let primary_bytes = self.read_stego_bytes(0, crate::BLOCK_SIZE)?;
        let backup_bytes = self.read_stego_bytes(crate::BLOCK_SIZE, crate::BLOCK_SIZE)?;
        let primary = Superblock::decode(&primary_bytes);
        let backup = Superblock::decode(&backup_bytes);

        match (primary, backup) {
            (Ok(primary), Ok(backup)) => {
                if primary.fields.generation >= backup.fields.generation {
                    self.log_verbose(format!(
                        "loaded primary superblock generation={} backup_generation={}",
                        primary.fields.generation, backup.fields.generation
                    ));
                    Ok(primary)
                } else {
                    self.log_verbose(format!(
                        "loaded backup superblock generation={} primary_generation={}",
                        backup.fields.generation, primary.fields.generation
                    ));
                    Ok(backup)
                }
            }
            (Ok(primary), Err(error)) => {
                self.log_verbose(format!("backup superblock invalid, using primary: {error}"));
                Ok(primary)
            }
            (Err(error), Ok(backup)) => {
                self.log_verbose(format!("primary superblock invalid, using backup: {error}"));
                Ok(backup)
            }
            (Err(primary_error), Err(backup_error)) => {
                Err(DeviceError::AllSuperblockMirrorsCorrupt {
                    primary: primary_error,
                    backup: backup_error,
                })
            }
        }
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
        let shadow_block = self.shadow_block_index()?;
        if block_index >= shadow_block {
            return Err(DeviceError::ReservedShadowBlock {
                index: block_index,
                shadow_block,
            });
        }
        Ok(())
    }

    fn shadow_block_index(&self) -> Result<u32, DeviceError> {
        let shadow_block = self.superblock.fields.shadow_block;
        if shadow_block == NO_BLOCK {
            return Err(DeviceError::MissingShadowBlock);
        }
        Ok(shadow_block)
    }

    fn stage_pending_write(&mut self, block_index: u32, data: &[u8]) -> Result<u32, DeviceError> {
        let shadow_block = self.shadow_block_index()?;
        let pending_crc = crc32(data);

        self.write_logical_block_raw(shadow_block, data)?;
        self.flush()?;

        self.superblock.fields.pending_target_block = block_index;
        self.superblock.fields.pending_target_crc32 = pending_crc;
        self.persist_superblocks()?;

        self.log_verbose(format!(
            "staged pending write for block {} via shadow block {} crc={pending_crc:#x}",
            block_index, shadow_block
        ));

        Ok(pending_crc)
    }

    fn clear_pending_write(&mut self) -> Result<(), DeviceError> {
        self.superblock.fields.pending_target_block = NO_BLOCK;
        self.superblock.fields.pending_target_crc32 = 0;
        self.persist_superblocks()?;
        self.log_verbose("cleared pending write marker");
        Ok(())
    }

    fn recover_if_needed(&mut self) -> Result<(), DeviceError> {
        self.recover_pending_data_write()?;
        self.recover_pending_metadata()?;
        Ok(())
    }

    fn recover_pending_data_write(&mut self) -> Result<(), DeviceError> {
        let target_block = self.superblock.fields.pending_target_block;
        if target_block == NO_BLOCK {
            return Ok(());
        }

        self.ensure_data_block(target_block)?;
        let shadow_block = self.shadow_block_index()?;
        let shadow_bytes = self.read_logical_block_raw(shadow_block)?;
        let actual_crc = crc32(&shadow_bytes);
        let expected_crc = self.superblock.fields.pending_target_crc32;

        if actual_crc != expected_crc {
            return Err(DeviceError::PendingWriteCrcMismatch {
                target_block,
                expected: expected_crc,
                actual: actual_crc,
            });
        }

        self.log_verbose(format!(
            "replaying pending write from shadow block {} into block {}",
            shadow_block, target_block
        ));
        self.write_logical_block_raw(target_block, &shadow_bytes)?;
        self.update_block_crc(target_block, &shadow_bytes)?;
        self.flush()?;
        self.clear_pending_write()?;
        Ok(())
    }

    fn recover_pending_metadata(&mut self) -> Result<(), DeviceError> {
        match self.superblock.fields.pending_metadata_op {
            PendingMetadataOp::None => Ok(()),
            PendingMetadataOp::AllocHeadAdvance => {
                let block_index = self.superblock.fields.pending_metadata_block;
                let next_head = self.superblock.fields.pending_metadata_aux;
                self.log_verbose(format!(
                    "rolling back pending alloc: block={} next_head={}",
                    block_index, next_head
                ));
                self.superblock.fields.free_list_head = block_index;
                self.clear_pending_metadata_fields();
                self.persist_superblocks()?;
                Ok(())
            }
            PendingMetadataOp::FreeHeadPush => {
                let block_index = self.superblock.fields.pending_metadata_block;
                let previous_head = self.superblock.fields.pending_metadata_aux;
                self.ensure_data_block(block_index)?;

                let free_bytes = FreeListBlock {
                    next_free_block: previous_head,
                }
                .encode();
                self.log_verbose(format!(
                    "finalizing pending free: block={} previous_head={}",
                    block_index, previous_head
                ));
                self.write_logical_block_raw(block_index, &free_bytes)?;
                self.update_block_crc(block_index, &free_bytes)?;
                self.flush()?;

                self.superblock.fields.free_list_head = block_index;
                self.clear_pending_metadata_fields();
                self.persist_superblocks()?;
                Ok(())
            }
        }
    }

    fn stage_pending_metadata(
        &mut self,
        operation: PendingMetadataOp,
        block_index: u32,
        auxiliary: u32,
    ) -> Result<(), DeviceError> {
        self.superblock.fields.pending_metadata_op = operation;
        self.superblock.fields.pending_metadata_block = block_index;
        self.superblock.fields.pending_metadata_aux = auxiliary;
        self.persist_superblocks()?;
        self.log_verbose(format!(
            "staged pending metadata op {:?} block={} aux={}",
            operation, block_index, auxiliary
        ));
        Ok(())
    }

    fn clear_pending_metadata_fields(&mut self) {
        self.superblock.fields.pending_metadata_op = PendingMetadataOp::None;
        self.superblock.fields.pending_metadata_block = NO_BLOCK;
        self.superblock.fields.pending_metadata_aux = NO_BLOCK;
    }

    #[doc(hidden)]
    pub fn stage_pending_write_for_test(
        &mut self,
        block_index: u32,
        data: &[u8],
    ) -> Result<(), DeviceError> {
        self.ensure_data_block(block_index)?;
        if data.len() != crate::BLOCK_SIZE {
            return Err(DeviceError::InvalidBlockWriteLength {
                expected: crate::BLOCK_SIZE,
                actual: data.len(),
            });
        }

        self.stage_pending_write(block_index, data)?;
        Ok(())
    }

    #[doc(hidden)]
    pub fn stage_pending_alloc_for_test(&mut self) -> Result<u32, DeviceError> {
        let head = self.superblock.fields.free_list_head;
        if head == NO_BLOCK {
            return Err(DeviceError::OutOfSpace);
        }

        let head_block = self.read_logical_block_raw(head)?;
        self.verify_block_crc(head, &head_block)?;
        let free_block = FreeListBlock::decode(&head_block)?;
        self.stage_pending_metadata(
            PendingMetadataOp::AllocHeadAdvance,
            head,
            free_block.next_free_block,
        )?;
        Ok(head)
    }

    #[doc(hidden)]
    pub fn stage_pending_free_for_test(&mut self, block_index: u32) -> Result<(), DeviceError> {
        self.ensure_data_block(block_index)?;
        let previous_head = self.superblock.fields.free_list_head;
        self.stage_pending_metadata(PendingMetadataOp::FreeHeadPush, block_index, previous_head)?;
        let free_bytes = FreeListBlock {
            next_free_block: previous_head,
        }
        .encode();
        self.write_block(block_index, &free_bytes)?;
        Ok(())
    }

    fn read_logical_block_raw(&self, block_index: u32) -> Result<Vec<u8>, DeviceError> {
        self.ensure_valid_block(block_index)?;
        self.read_stego_bytes(block_index as usize * crate::BLOCK_SIZE, crate::BLOCK_SIZE)
    }

    fn write_logical_block_raw(
        &mut self,
        block_index: u32,
        data: &[u8],
    ) -> Result<(), DeviceError> {
        self.ensure_valid_block(block_index)?;
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

            let decoded = self.decode_slot(slot)?;
            let overlap_start = start.max(slot.byte_start);
            let overlap_end = end.min(slot.byte_end);
            let src_start = overlap_start - slot.byte_start;
            let src_end = overlap_end - slot.byte_start;
            let dst_start = overlap_start - start;
            let dst_end = overlap_end - start;
            out[dst_start..dst_end].copy_from_slice(&decoded[src_start..src_end]);
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

            let mut decoded = self.decode_slot(&slot)?;
            let overlap_start = start.max(slot.byte_start);
            let overlap_end = end.min(slot.byte_end);
            let src_start = overlap_start - start;
            let src_end = overlap_end - start;
            let dst_start = overlap_start - slot.byte_start;
            let dst_end = overlap_end - slot.byte_start;
            decoded[dst_start..dst_end].copy_from_slice(&data[src_start..src_end]);
            self.encode_slot(&slot, &decoded)?;
        }

        Ok(())
    }

    fn decode_slot(&self, slot: &TensorByteSlot) -> Result<Vec<u8>, DeviceError> {
        let storage = &self.mmap[slot.file_offset..slot.file_offset + slot.storage_len];
        let payload = match slot.quant_type {
            GgufQuantType::Q8_0 => decode_blockwise(
                storage,
                packing::q8_0::BLOCK_BYTES,
                packing::q8_0::read_payload_block,
            )?,
            GgufQuantType::Q6K => decode_blockwise(
                storage,
                packing::q6_k::BLOCK_BYTES,
                packing::q6_k::read_payload_block,
            )?,
            GgufQuantType::Q5K => decode_blockwise(
                storage,
                packing::q5_k::BLOCK_BYTES,
                packing::q5_k::read_payload_block,
            )?,
            GgufQuantType::Q4K => decode_blockwise(
                storage,
                packing::q4_k::BLOCK_BYTES,
                packing::q4_k::read_payload_block,
            )?,
            GgufQuantType::Q3K => decode_blockwise(
                storage,
                packing::q3_k::BLOCK_BYTES,
                packing::q3_k::read_payload_block,
            )?,
            GgufQuantType::F16 => {
                let mut payload = packing::float::read_f16_payload(storage)?;
                payload.truncate(slot.capacity_bytes);
                payload
            }
            GgufQuantType::F32 => packing::float::read_f32_payload(storage)?,
            GgufQuantType::Q2K
            | GgufQuantType::Q4_0
            | GgufQuantType::Q4_1
            | GgufQuantType::Q5_0
            | GgufQuantType::Q5_1
            | GgufQuantType::Q8_1
            | GgufQuantType::Q8K => {
                return Err(DeviceError::UnsupportedQuantType(slot.quant_type));
            }
        };

        Ok(payload)
    }

    fn encode_slot(&mut self, slot: &TensorByteSlot, payload: &[u8]) -> Result<(), DeviceError> {
        let storage = &mut self.mmap[slot.file_offset..slot.file_offset + slot.storage_len];
        match slot.quant_type {
            GgufQuantType::Q8_0 => encode_blockwise(
                storage,
                packing::q8_0::BLOCK_BYTES,
                packing::q8_0::PAYLOAD_BYTES_PER_BLOCK,
                payload,
                packing::q8_0::write_payload_block,
            )?,
            GgufQuantType::Q6K => encode_blockwise(
                storage,
                packing::q6_k::BLOCK_BYTES,
                packing::q6_k::PAYLOAD_BYTES_PER_BLOCK,
                payload,
                packing::q6_k::write_payload_block,
            )?,
            GgufQuantType::Q5K => encode_blockwise(
                storage,
                packing::q5_k::BLOCK_BYTES,
                packing::q5_k::PAYLOAD_BYTES_PER_BLOCK,
                payload,
                packing::q5_k::write_payload_block,
            )?,
            GgufQuantType::Q4K => encode_blockwise(
                storage,
                packing::q4_k::BLOCK_BYTES,
                packing::q4_k::PAYLOAD_BYTES_PER_BLOCK,
                payload,
                packing::q4_k::write_payload_block,
            )?,
            GgufQuantType::Q3K => encode_blockwise(
                storage,
                packing::q3_k::BLOCK_BYTES,
                packing::q3_k::PAYLOAD_BYTES_PER_BLOCK,
                payload,
                packing::q3_k::write_payload_block,
            )?,
            GgufQuantType::F16 => packing::float::write_f16_payload(storage, payload)?,
            GgufQuantType::F32 => packing::float::write_f32_payload(storage, payload)?,
            GgufQuantType::Q2K
            | GgufQuantType::Q4_0
            | GgufQuantType::Q4_1
            | GgufQuantType::Q5_0
            | GgufQuantType::Q5_1
            | GgufQuantType::Q8_1
            | GgufQuantType::Q8K => {
                return Err(DeviceError::UnsupportedQuantType(slot.quant_type));
            }
        }

        Ok(())
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
        if !self.tracks_crc_for_block(block_index)? {
            return Ok(());
        }

        let crc = crc32(data);
        let (integrity_block_index, entry_index) = self.integrity_location(block_index)?;
        let integrity_bytes = self.read_logical_block_raw(integrity_block_index)?;
        let mut integrity_block = IntegrityBlock::decode(&integrity_bytes)?;
        if entry_index >= integrity_block.crc32_entries.len() {
            return Err(DeviceError::MissingIntegrityEntry {
                block_index,
                integrity_block_index,
            });
        }
        integrity_block.crc32_entries[entry_index] = crc;
        let encoded = integrity_block.encode()?;
        self.write_logical_block_raw(integrity_block_index, &encoded)?;
        self.log_verbose(format!(
            "updated crc for block {} in integrity block {} entry {}",
            block_index, integrity_block_index, entry_index
        ));
        Ok(())
    }

    fn read_crc_entry(&self, block_index: u32) -> Result<Option<u32>, DeviceError> {
        if !self.tracks_crc_for_block(block_index)? {
            return Ok(None);
        }

        let (integrity_block_index, entry_index) = self.integrity_location(block_index)?;
        let integrity_bytes = self.read_logical_block_raw(integrity_block_index)?;
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

    fn integrity_location(&self, block_index: u32) -> Result<(u32, usize), DeviceError> {
        self.ensure_data_block(block_index)?;

        let zero_based = block_index - self.data_region_start();
        let integrity_offset = zero_based / ENTRIES_PER_INTEGRITY_BLOCK as u32;
        let integrity_block_index = BACKUP_SUPERBLOCK_BLOCK + 1 + integrity_offset;
        let entry_index = (zero_based % ENTRIES_PER_INTEGRITY_BLOCK as u32) as usize;
        Ok((integrity_block_index, entry_index))
    }

    fn tracks_crc_for_block(&self, block_index: u32) -> Result<bool, DeviceError> {
        let shadow_block = self.shadow_block_index()?;
        Ok(block_index >= self.data_region_start() && block_index < shadow_block)
    }

    fn log_verbose(&self, message: impl AsRef<str>) {
        if self.verbose {
            eprintln!("[llmdb:device] {}", message.as_ref());
        }
    }
}

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
        GgufQuantType::Q2K
        | GgufQuantType::Q4_0
        | GgufQuantType::Q4_1
        | GgufQuantType::Q5_0
        | GgufQuantType::Q5_1
        | GgufQuantType::Q8_1
        | GgufQuantType::Q8K => return Err(DeviceError::UnsupportedQuantType(quant_type)),
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

fn decode_blockwise<const PAYLOAD_BYTES: usize>(
    storage: &[u8],
    block_bytes: usize,
    reader: fn(&[u8]) -> Result<[u8; PAYLOAD_BYTES], PackingError>,
) -> Result<Vec<u8>, DeviceError> {
    if storage.len() % block_bytes != 0 {
        return Err(DeviceError::InvalidPhysicalStorageLength {
            len: storage.len(),
            block_bytes,
        });
    }

    let mut payload = Vec::new();
    for chunk in storage.chunks_exact(block_bytes) {
        payload.extend_from_slice(&reader(chunk)?);
    }
    Ok(payload)
}

fn encode_blockwise(
    storage: &mut [u8],
    block_bytes: usize,
    payload_bytes_per_block: usize,
    payload: &[u8],
    writer: fn(&mut [u8], &[u8]) -> Result<(), PackingError>,
) -> Result<(), DeviceError> {
    if storage.len() % block_bytes != 0 {
        return Err(DeviceError::InvalidPhysicalStorageLength {
            len: storage.len(),
            block_bytes,
        });
    }

    if payload.len() != (storage.len() / block_bytes) * payload_bytes_per_block {
        return Err(DeviceError::PayloadLengthMismatch {
            expected: (storage.len() / block_bytes) * payload_bytes_per_block,
            actual: payload.len(),
        });
    }

    for (chunk, payload_chunk) in storage
        .chunks_exact_mut(block_bytes)
        .zip(payload.chunks_exact(payload_bytes_per_block))
    {
        writer(chunk, payload_chunk)?;
    }

    Ok(())
}

fn integrity_block_count(total_blocks: u32) -> u32 {
    let mut integrity_count = 0_u32;
    loop {
        let data_blocks = total_blocks.saturating_sub(3 + integrity_count);
        let needed = if data_blocks == 0 {
            0
        } else {
            ((data_blocks as usize).div_ceil(ENTRIES_PER_INTEGRITY_BLOCK)) as u32
        };
        if integrity_count >= needed {
            return integrity_count;
        }
        integrity_count = needed;
    }
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
    #[error("no valid mirrored superblock found: primary={primary}, backup={backup}")]
    AllSuperblockMirrorsCorrupt {
        primary: IntegrityError,
        backup: IntegrityError,
    },
    #[error("packing error: {0}")]
    Packing(#[from] PackingError),
    #[error("insufficient capacity: {0} bytes")]
    InsufficientCapacity(u64),
    #[error(
        "insufficient capacity for shadow layout: total_blocks={total_blocks}, integrity_blocks={integrity_blocks}"
    )]
    InsufficientCapacityForShadow {
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
    #[error("invalid physical storage length {len} for block size {block_bytes}")]
    InvalidPhysicalStorageLength { len: usize, block_bytes: usize },
    #[error("payload length mismatch: expected {expected}, got {actual}")]
    PayloadLengthMismatch { expected: usize, actual: usize },
    #[error("block index {index} out of range, total blocks {total_blocks}")]
    BlockOutOfRange { index: u32, total_blocks: u32 },
    #[error("block {index} is reserved metadata, first data block is {data_region_start}")]
    ReservedMetadataBlock { index: u32, data_region_start: u32 },
    #[error("block {index} is the reserved shadow block {shadow_block}")]
    ReservedShadowBlock { index: u32, shadow_block: u32 },
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
    #[error("device superblock is missing a shadow block")]
    MissingShadowBlock,
    #[error(
        "pending shadow write crc mismatch for block {target_block}: expected {expected:#x}, got {actual:#x}"
    )]
    PendingWriteCrcMismatch {
        target_block: u32,
        expected: u32,
        actual: u32,
    },
}
