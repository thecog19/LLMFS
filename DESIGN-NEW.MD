# LLMDB Implementation Specification

## 1. Project Overview

LLMDB is a steganographic block storage device backed by the least-significant bits of quantized model weights in GGUF files. It exposes a block device that can be formatted with a standard filesystem (ext4 via NBD) or used directly through the CLI for file storage. A built-in inference bridge allows the model to read and reason about files stored in its own weights.

### Target Model

SmolLM3-3B (`ggml-org/SmolLM3-3B-GGUF`). Apache 2.0 license, tool-calling support, GGUF available in Q2_K through F16. Q8_0 GGUF is 3.28 GB. The implementation is model-agnostic — any GGUF file works.

### Language

Rust. Dependencies: `memmap2` (mmap), `clap` (CLI), `crc32fast` (integrity), `nbd` or custom NBD protocol implementation, `llama-cpp-rs` or subprocess shelling to `llama.cpp` for inference.

## 2. Version Roadmap

### V1 — Naive Stego + Filesystem

- GGUF parser (v2, v3)
- All K-quant packing modules (Q8_0, Q6_K, Q5_K, Q4_K, Q3_K, F16, F32)
- Sequential block allocation with fixed eligibility tiers
- Lobotomy mode (removes eligibility restrictions)
- Integrity blocks (separate from data blocks)
- Shadow-copy-then-swap atomic writes
- Flat file table with POSIX-style metadata
- NBD server for ext4 mounting
- Direct CLI file operations (store, get, ls)
- `ask` command (model queries its own files)
- Diagnostics (utilization, degradation estimate)

**Cover-file viability (empirical, 2026-04-17):** V1's dual-use property
— the GGUF remains a functional model while carrying stego data — holds
for **F16 and F32** cover files only. Those packers steal bits strictly
from the low mantissa (mask `0xFFF0` / `0xFFFF_FF00`), preserving sign
and exponent, so weight magnitudes survive at reduced precision only.
SmolLM2-135M-f16 at 97% utilization still produces fluent, on-topic
completions; baseline and post-stego outputs diverge only in minor
word-choice details on some prompts.

The **int-quantized** packers (Q8_0, Q6_K, Q5_K, Q4_K, Q3_K) cannot
offer the dual-use property in V1: Q8_0 masks with `0xF0`, destroying
half of every int8 weight's precision including its magnitude. Both
stories15M-q8_0 and smollm2-135m-q8_0 collapse into degenerate
repetition (`ushushush…`, `bianbian…`) after `llmdb init` alone —
before any user data is stored. For Q8_0 / K-quant cover files V1 is a
**storage-only** device; the model is a write-once casualty. Restoring
dual-use on int-quantized covers is explicitly V2's job (sensitivity-
aware allocation), not a V1 tuning fix.

### V2 — Inode + CoW Storage

- Ceiling-magnitude anchor at rule-derived positions (invariant under writes; replaces V1's magnitude-ranked bootstrap which drifted for Q8_0)
- Unix-style inode substrate (per-file inodes with direct + single/double/triple-indirect pointers, replaces V1's block-indexed redirection table)
- Hierarchical directories (directory-is-a-file, FUSE-native)
- Variable-length chunks with content-defined chunking (FastCDC)
- Allocation priority: content-match dedup (BLAKE3-128) → dirty-first → pristine by max-ceiling-magnitude
- Copy-on-write writes with two-slot generational root; atomic commit via slot alternation (replaces V1's shadow-copy-via-redirection)
- Multi-reader / single-writer concurrency
- Progressive calibration (AWQ fast / Hessian full; Fisher deferred to V3)
- GPTQ-style error compensation (future; layers on top of the inode substrate)
- Damage-accumulation remap (lossless remap deferred to V3's VRAM shadow)

### V3 — GPU Acceleration + Lossless Remap

- GPU-accelerated forward pass (supplants V2's CPU calibration path; unlocks large models)
- Fisher-information calibration (tractable on GPU hardware)
- Scatter map in VRAM / unused KV-cache slots (moves V2's sensitivity index off-disk)
- VRAM-resident dirty block cache with lazy flush to GGUF
- VRAM shadow store → **lossless remap** (the V2-deferred decision)
- Tiered degradation (GPU hot → CPU warm → disk cold)

V2 details: §15. V3 details: §16. This document specifies V1 (§3-14); the V2 and V3 sections specify how those later versions modify or extend V1's data structures and algorithms.

## 3. GGUF Parser

### File Structure

```
Offset    Content
0x00      Magic: 0x46554747 ("GGUF" little-endian)
0x04      Version: uint32 (2 or 3)
0x08      Tensor count: uint64
0x10      KV metadata count: uint64
0x18      KV metadata entries (variable length)
...       Tensor info entries (variable length)
...       Alignment padding to ALIGNMENT boundary (default 32 bytes)
...       Tensor data (contiguous)
```

### Parsing Requirements

1. Read and validate magic and version. Reject versions other than 2 or 3.
2. Parse all KV metadata entries. Store as a HashMap<String, GGUFValue>. Required keys: `general.architecture`, `general.name`. Optional but useful: tokenizer vocabulary (for V2 compression).
3. Parse all tensor info entries. For each tensor, record:
   - `name: String`
   - `n_dims: u32`
   - `shape: Vec<u64>`
   - `dtype: GGUFType` (enum, see below)
   - `offset: u64` (relative to start of tensor data section)
4. Compute the absolute file offset of the tensor data section: after all metadata and tensor info entries, aligned to ALIGNMENT boundary.
5. For each tensor, compute its absolute file offset: `tensor_data_start + tensor.offset`.

### GGUFType Enum

```rust
#[repr(u32)]
enum GGUFType {
    F32     = 0,
    F16     = 1,
    Q4_0    = 2,
    Q4_1    = 3,
    // 4, 5 deprecated
    Q5_0    = 6,
    Q5_1    = 7,
    Q8_0    = 8,
    Q8_1    = 9,
    Q2_K    = 10,
    Q3_K    = 11,
    Q4_K    = 12,
    Q5_K    = 13,
    Q6_K    = 14,
    Q8_K    = 15,
    // others exist but are uncommon
}
```

### GGUF KV Metadata Types

Version 2 and version 3 differ in string length encoding (uint32 vs uint64). The parser must dispatch on version for all variable-length fields.

```rust
enum GGUFValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GGUFValue>),
}
```

## 4. Quantization Block Layouts

Each quantization type has a fixed block structure. A "block" here refers to the GGUF quantization block (a group of weights sharing scale factors), not an LLMDB storage block.

### Q8_0

```
Block size: 32 weights
Layout:     2 bytes (f16 scale) + 32 bytes (int8 quants)
Total:      34 bytes per block
Steal:      4 LSBs per int8 quant
Capacity:   4 bits per weight = 16 bytes per block
```

Bit operations:
```rust
fn steal_q8_0(quant: i8) -> u8 {
    (quant as u8) & 0x0F
}

fn embed_q8_0(quant: i8, data: u8) -> i8 {
    ((quant as u8) & 0xF0 | (data & 0x0F)) as i8
}
```

File offset for weight at `index` within a tensor stored as Q8_0:
```rust
fn q8_0_file_offset(tensor_start: u64, weight_index: usize) -> u64 {
    let block = weight_index / 32;
    let within_block = weight_index % 32;
    tensor_start + (block as u64 * 34) + 2 + within_block as u64
}
```

### Q6_K

```
Block size: 256 weights (16 sub-blocks of 16)
Layout:
    ql[128]     low 4 bits of each 6-bit quant, packed as nibbles
    qh[64]      high 2 bits of each 6-bit quant, packed as bit pairs
    scales[16]  int8 per sub-block
    d[2]        f16 super-scale
Total:      210 bytes per block
Steal:      2 LSBs from each nibble in ql
Capacity:   2 bits per weight
```

Bit operations for ql array:
```rust
fn steal_q6k(ql_byte: u8) -> u8 {
    // Each byte has two nibbles, steal bottom bit of each
    let lo = ql_byte & 0x01;         // bit 0 of low nibble
    let hi = (ql_byte >> 4) & 0x01;  // bit 0 of high nibble
    (hi << 1) | lo                   // 2 stolen bits packed
}

fn embed_q6k(ql_byte: u8, data: u8) -> u8 {
    let lo_cleared = ql_byte & 0xFE;           // clear bit 0 of low nibble
    let hi_cleared = lo_cleared & 0xEF;         // clear bit 4 (bit 0 of high nibble)
    hi_cleared | (data & 0x01) | ((data & 0x02) << 3)
}
```

File offset for the ql byte containing weight pair at `weight_index`:
```rust
fn q6k_ql_offset(tensor_start: u64, weight_index: usize) -> u64 {
    let block = weight_index / 256;
    let within_block = weight_index % 256;
    let ql_byte = within_block / 2;
    tensor_start + (block as u64 * 210) + ql_byte as u64
}
```

### Q5_K

```
Block size: 256 weights (8 sub-blocks of 32)
Layout (Q5_K_M):
    qs[128]     low 4 bits of each 5-bit quant, packed as nibbles
    qh[32]      high 1 bit of each 5-bit quant
    scales[12]  packed 6-bit scales and mins
    d[2]        f16 super-scale
    dmin[2]     f16 super-min
Total:      176 bytes per block
Steal:      1 LSB per nibble in qs
Capacity:   1 bit per weight
```

Bit operations:
```rust
fn steal_q5k(qs_byte: u8) -> u8 {
    let lo = qs_byte & 0x01;
    let hi = (qs_byte >> 4) & 0x01;
    (hi << 1) | lo
}

fn embed_q5k(qs_byte: u8, data: u8) -> u8 {
    let cleared = qs_byte & 0xEE;  // clear bit 0 of each nibble
    cleared | (data & 0x01) | ((data & 0x02) << 3)
}
```

### Q4_K

```
Block size: 256 weights (8 sub-blocks of 32)
Layout (Q4_K_M):
    qs[128]     4-bit quants packed as nibbles
    scales[12]  packed 6-bit scales and mins
    d[2]        f16 super-scale
    dmin[2]     f16 super-min
Total:      144 bytes per block
Steal:      1 LSB per nibble in qs
Capacity:   1 bit per weight
```

Same steal/embed as Q5_K — the qs array has the same nibble structure. The degradation is worse (Q4→Q3 effective precision) but the bit operations are identical.

### Q3_K

```
Block size: 256 weights
Layout:
    qs[64]      low 2 bits of each 3-bit quant, packed as 2-bit pairs
    hmask[32]   high 1 bit of each 3-bit quant
    scales[12]  packed 6-bit scales
    d[2]        f16 super-scale
Total:      110 bytes per block
Steal:      1 bit per 2-bit pair in qs
Capacity:   1 bit per weight
```

Bit operations for qs (4 quants per byte, 2 bits each):
```rust
fn steal_q3k(qs_byte: u8) -> u8 {
    // 4 two-bit values per byte, steal LSB of each
    let b0 = qs_byte & 0x01;
    let b1 = (qs_byte >> 2) & 0x01;
    let b2 = (qs_byte >> 4) & 0x01;
    let b3 = (qs_byte >> 6) & 0x01;
    (b3 << 3) | (b2 << 2) | (b1 << 1) | b0
}

fn embed_q3k(qs_byte: u8, data: u8) -> u8 {
    let cleared = qs_byte & 0b10101010;  // clear bit 0 of each 2-bit pair
    cleared
        | (data & 0x01)
        | ((data & 0x02) << 1)
        | ((data & 0x04) << 2)
        | ((data & 0x08) << 3)
}
```

### F16

```
Layout:     2 bytes per weight (IEEE 754 half-precision)
            1 sign bit, 5 exponent bits, 10 mantissa bits
Steal:      4 LSBs of mantissa
Capacity:   4 bits per weight
```

```rust
fn steal_f16(bytes: [u8; 2]) -> u8 {
    bytes[0] & 0x0F  // bottom 4 bits of low byte = bottom 4 mantissa bits
}

fn embed_f16(bytes: [u8; 2], data: u8) -> [u8; 2] {
    [(bytes[0] & 0xF0) | (data & 0x0F), bytes[1]]
}
```

### F32

```
Layout:     4 bytes per weight (IEEE 754 single-precision)
            1 sign bit, 8 exponent bits, 23 mantissa bits
Steal:      8 LSBs of mantissa
Capacity:   8 bits per weight (1 byte per weight)
```

```rust
fn steal_f32(bytes: [u8; 4]) -> u8 {
    bytes[0]  // bottom byte = bottom 8 mantissa bits
}

fn embed_f32(bytes: [u8; 4], data: u8) -> [u8; 4] {
    [data, bytes[1], bytes[2], bytes[3]]
}
```

### Q2_K

Not eligible. Zero stealable bits. Skip during tensor map construction.

### Packing Trait

All packing modules implement a common trait:

```rust
trait QuantPacker {
    /// Number of stealable bits per weight for this quant type.
    fn bits_per_weight(&self) -> u32;

    /// Size of one quantization block in bytes.
    fn block_size_bytes(&self) -> usize;

    /// Number of weights per quantization block.
    fn weights_per_block(&self) -> usize;

    /// Extract stolen bits from a quantization block.
    /// Returns a Vec of nibbles/bits extracted from this block.
    fn extract(&self, block_bytes: &[u8]) -> Vec<u8>;

    /// Embed data bits into a quantization block.
    /// Returns the modified block bytes.
    fn embed(&self, block_bytes: &[u8], data: &[u8]) -> Vec<u8>;

    /// File offset of the stealable bytes within a quant block,
    /// relative to block start. Used for partial reads.
    fn stealable_byte_offsets(&self) -> Vec<usize>;
}
```

## 5. Stego Device

### Tensor Eligibility

Tensors are classified by name pattern matching against the GGUF tensor names.

```rust
enum TensorTier {
    Skip,           // Never use for storage
    Tier1,          // FFN weights — preferred
    Tier2,          // Attention projection weights
    Lobotomy,       // Embeddings, norms, head — only with --lobotomy
}

fn classify_tensor(name: &str, lobotomy: bool) -> TensorTier {
    if name.contains("token_embd") || name.contains("output.weight") {
        if lobotomy { TensorTier::Lobotomy } else { TensorTier::Skip }
    } else if name.contains("_norm") {
        if lobotomy { TensorTier::Lobotomy } else { TensorTier::Skip }
    } else if name.contains("ffn_gate") || name.contains("ffn_up") || name.contains("ffn_down") {
        TensorTier::Tier1
    } else if name.contains("attn_q") || name.contains("attn_k")
           || name.contains("attn_v") || name.contains("attn_output") {
        TensorTier::Tier2
    } else {
        // Unknown tensor type — skip by default
        TensorTier::Skip
    }
}
```

### Tensor Map

The tensor map provides a flat address space over all eligible weight slots, ordered by allocation priority.

```rust
struct TensorSlot {
    tensor_name: String,
    quant_type: GGUFType,
    file_offset: u64,      // absolute byte offset in GGUF file
    n_weights: usize,
    bits_per_weight: u32,
    tier: TensorTier,
    layer_index: u32,       // extracted from tensor name (e.g., blk.31 → 31)
}

struct TensorMap {
    slots: Vec<TensorSlot>,  // ordered: Tier1 deep→shallow, then Tier2 deep→shallow, then Lobotomy
    total_capacity_bytes: u64,
}
```

Construction:

1. Parse all tensor info from GGUF.
2. Classify each tensor by eligibility and tier.
3. For each eligible tensor, determine its quant type and compute bits_per_weight. Skip Q2_K.
4. Extract layer index from tensor name (regex: `blk\.(\d+)`).
5. Sort eligible tensors: Tier1 before Tier2 before Lobotomy. Within each tier, sort by layer index descending (deepest first).
6. Compute cumulative capacity.

### Address Translation

Given a logical byte offset into the stego address space, return the physical GGUF file location and quant type:

```rust
struct PhysicalLocation {
    file_offset: u64,
    quant_type: GGUFType,
    // For sub-byte quant types, which bits within the byte
    bit_offset: u32,
    bits_available: u32,
}

impl TensorMap {
    fn translate(&self, byte_offset: u64) -> PhysicalLocation {
        let mut remaining = byte_offset;
        for slot in &self.slots {
            let slot_capacity = (slot.n_weights as u64 * slot.bits_per_weight as u64) / 8;
            if remaining < slot_capacity {
                // This byte falls within this tensor slot
                let bit_offset_in_slot = remaining * 8;
                let weight_index = bit_offset_in_slot / slot.bits_per_weight as u64;
                // Dispatch to quant-type-specific offset calculation
                return compute_physical_offset(slot, weight_index as usize);
            }
            remaining -= slot_capacity;
        }
        panic!("byte_offset exceeds total capacity");
    }
}
```

The `compute_physical_offset` function dispatches to the per-quant-type offset calculation (e.g., `q8_0_file_offset`, `q6k_ql_offset`, etc.).

### Block Device Interface

```rust
const BLOCK_SIZE: usize = 4096;

struct StegoDevice {
    mmap: MmapMut,            // mmap'd GGUF file
    tensor_map: TensorMap,
    superblock: Superblock,
    lobotomy: bool,
}

impl StegoDevice {
    fn new(path: &Path, lobotomy: bool) -> Result<Self>;

    /// Read a full 4096-byte block from the stego address space.
    fn read_block(&self, block_index: u32) -> Result<[u8; BLOCK_SIZE]>;

    /// Write a full 4096-byte block via shadow-copy-then-swap.
    fn write_block(&mut self, block_index: u32, data: &[u8; BLOCK_SIZE]) -> Result<()>;

    /// Allocate a block from the free list.
    fn alloc_block(&mut self) -> Result<u32>;

    /// Return a block to the free list.
    fn free_block(&mut self, block_index: u32) -> Result<()>;

    /// Verify CRC32 of a block against the integrity table.
    fn verify_block(&self, block_index: u32) -> Result<bool>;

    /// Full integrity scan. Returns indices of corrupted blocks.
    fn verify_all(&self) -> Vec<u32>;

    /// Flush all pending writes to disk.
    fn flush(&self) -> Result<()>;

    /// Total number of blocks in the device.
    fn total_blocks(&self) -> u32;

    /// Number of currently allocated blocks.
    fn used_blocks(&self) -> u32;

    /// Percentage of eligible weight capacity used.
    fn utilization_pct(&self) -> f32;
}
```

### Block Read Implementation

```
fn read_block(&self, block_index: u32) -> Result<[u8; BLOCK_SIZE]> {
    1. Compute stego byte range: [block_index * BLOCK_SIZE, (block_index + 1) * BLOCK_SIZE)
    2. For each byte in range:
       a. Call tensor_map.translate(byte_offset) → PhysicalLocation
       b. Read the byte(s) at physical_location.file_offset from the mmap
       c. Dispatch to the appropriate QuantPacker::extract
       d. Accumulate extracted bits into the output byte
    3. Verify CRC32 against integrity table
    4. Return the 4096-byte block
}
```

This is the hot path in V1. It performs up to 8192 translated reads per block (2 nibble reads per byte for Q8_0 4-bit stealing). Optimization opportunities exist (batch reads within the same quant block, prefetch) but correctness first.

### Block Write Implementation (Shadow-Copy-Then-Swap)

Every write is bracketed by superblock updates that record the in-flight operation.
The recovery protocol (section 13) uses these fields to clean up after crashes.

```
fn write_block(&mut self, block_index: u32, data: &[u8; BLOCK_SIZE]) -> Result<()> {
    1. Allocate a shadow block from the free list.
    2. Record intent in superblock:
       - shadow_block = shadow physical index
       - shadow_target = block_index (logical)
       - Increment generation counter
       - Flush superblock
    3. Write data to the shadow block's physical locations:
       a. For each byte in data:
          - Call tensor_map.translate(shadow_byte_offset) → PhysicalLocation
          - Read current byte from mmap at that offset
          - Call QuantPacker::embed to splice data bits into the weight byte
          - Write modified byte back to mmap
    4. Compute CRC32 of data.
    5. Update the integrity block: store CRC32 for the shadow block. Flush.
    6. Update the redirection table: logical block_index → shadow physical index. Flush.
       (This is the commit point. After this, the new data is canonical.)
    7. Return the old physical block to the free list.
    8. Clear intent in superblock:
       - shadow_block = 0xFFFFFFFF
       - shadow_target = 0xFFFFFFFF
       - Increment generation counter
       - Flush superblock
}
```

The redirection table provides the indirection needed for shadow-copy. Every logical
block index maps to a physical block index. Initially this is an identity mapping.
After a shadow-copy write, the logical block points to the shadow's physical location.
The table is ~1.2 MB (4 bytes × ~300K blocks), stored in dedicated stego blocks.

### Superblock (Block 0)

```
Offset  Size    Field
0x00    5       Magic: "LLMDB" (0x4C4C4D4442)
0x05    1       Version: 1
0x06    2       Block size: 4096 (fixed, stored for validation)
0x08    4       Total blocks (computed from tensor map capacity)
0x0C    4       Free list head (physical block index, 0xFFFFFFFF = empty)
0x10    4       Integrity chain head (block index of first integrity block)
0x14    4       Redirection table start (block index)
0x18    4       Redirection table length (number of blocks used)
0x1C    4       File table start (block index)
0x20    4       File table length (number of blocks used)
0x24    1       Flags: bit 0 = lobotomy mode, bit 1 = dirty (unclean shutdown)
0x25    1       Quant profile: summary of quant types present (for status display)
0x26    2       Reserved
0x28    8       Generation counter (monotonically increasing, incremented on every write operation)
0x30    4       Shadow block index (0xFFFFFFFF = no shadow in flight)
0x34    4       Shadow target (logical block the shadow is replacing, valid only if shadow block != 0xFFFFFFFF)
0x38    4       CRC32 of bytes 0x00-0x37
0x3C    4036    Reserved / padding
```

Total: one block (4096 bytes).

### Free List

Linked list of free physical blocks. Each free block contains:

```
Offset  Size    Field
0x00    4       Next free block index (0xFFFFFFFF = end of list)
0x04    4092    Unused
```

On first init, all blocks except superblock, integrity blocks, redirection table, and file table are chained into the free list.

### Integrity Blocks

Separate blocks storing CRC32 checksums for data blocks.

```
Offset  Size    Field
0x00    4       Magic: "ICHK" (0x4943484B)
0x04    4       First data block index covered
0x08    4       Number of entries in this block
0x0C    4       Next integrity block index (0xFFFFFFFF = last)
0x10    4080    CRC32 entries (1020 entries × 4 bytes)
```

One integrity block covers 1020 data blocks. Number of integrity blocks needed = ceil(total_blocks / 1020).

On read: verify the block's CRC32 against the stored value. On write: update the CRC32 entry and flush.

### Redirection Table

Stored in dedicated blocks. Maps logical block index → physical block index.

```
Offset  Size    Field
0x00    4       First logical block index covered by this table block
0x04    4       Entry count
0x08    4088    Entries: uint32 physical block index × 1022 entries
```

Initially populated as identity mapping (logical == physical). Updated during shadow-copy writes.

## 6. File Storage Layer

V1 implements a flat file table with POSIX-style metadata. No directory hierarchy in V1 (can be added later; the file table format supports it via a parent field).

### File Table Entry

```
Offset  Size    Field
0x00    1       Entry type: 0=free, 1=regular file, 2=directory, 3=symlink
0x01    1       Flags: bit 0 = deleted (tombstone)
0x02    2       Mode: POSIX permission bits (e.g., 0o644)
0x04    4       UID
0x08    4       GID
0x0C    8       Size in bytes
0x14    8       Created timestamp (unix epoch, seconds)
0x1C    8       Modified timestamp
0x24    4       CRC32 of file contents (end-to-end, not per-block)
0x28    4       Block count
0x2C    4       Block map overflow block index (0xFFFFFFFF = inline only)
0x30    96      Filename (null-terminated UTF-8, max 95 chars)
0x90    112     Inline block map (28 block indices × 4 bytes)
```

Total: 256 bytes per entry. 16 entries per file table block.

The inline block map holds up to 28 block indices directly in the file table entry. For files up to 28 × 4096 = 114,688 bytes (~112 KB), no additional metadata blocks are needed.

For larger files, the overflow field points to a block map overflow block:

```
Block Map Overflow Block:
Offset  Size    Field
0x00    4       Next overflow block index (0xFFFFFFFF = last)
0x04    4092    Block indices (1023 entries × 4 bytes)
```

One overflow block extends capacity by 1023 blocks = ~4 MB. Overflow blocks chain for arbitrarily large files. A 100 MB file uses 28 inline + ceil((25000 - 28) / 1023) = 25 overflow blocks — negligible overhead.

### File Data Blocks

Data blocks are clean 4096 bytes. No internal metadata, no chain pointers, no headers. The block's logical position within a file is determined entirely by its position in the file table entry's block map.

```
Offset  Size    Field
0x00    4096    File data
```

This means:
- Every data block is a full 4096-byte page.
- NBD and ext4 see exactly the block size they expect with no shearing.
- Data blocks are interchangeable — a block has no knowledge of which file it belongs to or where in that file it sits. This is determined by the block map.
- Reads and writes are block-aligned with no read-modify-write for chain pointer maintenance.

### File Operations

```rust
impl StegoDevice {
    /// Store a file from the host filesystem into the stego device.
    fn store_file(&mut self, host_path: &Path, stego_name: &str, mode: u16) -> Result<()> {
        // 1. Read file from host
        // 2. Allocate blocks: ceil(file_size / 4096) data blocks
        // 3. If block_count > 28, allocate overflow block map blocks
        // 4. Write file data into blocks (clean 4096-byte writes)
        // 5. Populate block map (inline entries + overflow blocks)
        // 6. Create file table entry
        // 7. Flush
    }

    /// Retrieve a file from the stego device to the host filesystem.
    fn get_file(&self, stego_name: &str, host_path: &Path) -> Result<()> {
        // 1. Find file table entry by name
        // 2. Read block indices from inline map + overflow blocks
        // 3. Read each data block in order, concatenate
        // 4. Truncate to file size (last block may be partial)
        // 5. Verify end-to-end CRC32
        // 6. Write to host path
    }

    /// List all files.
    fn list_files(&self) -> Result<Vec<FileEntry>> {
        // Read file table blocks, return non-free entries
    }

    /// Delete a file.
    fn delete_file(&mut self, stego_name: &str) -> Result<()> {
        // 1. Find file table entry
        // 2. Walk block chain, freeing each block
        // 3. Mark entry as deleted (tombstone)
        // 4. Flush
    }
}
```

## 7. FUSE Filesystem

The mount path is a FUSE driver (`fuser` crate) that exposes the file
table as a real filesystem. No root required (unprivileged mount via
`fusermount3`), no kernel modules, no block-device round-trip. Virtual
directories are synthesized from `/`-separated filenames at the FUSE
layer; on-disk storage stays flat per §6.

Earlier drafts of this document specified an NBD-backed ext4 mount.
That path was deleted: it required root plus the nbd kernel module,
produced an ext4 image independent of the file table (so CLI and mount
saw incompatible data), and its test coverage was gated behind root
helpers that ran in practice ~never. FUSE is a smaller surface with
better coverage and a single source of truth.

### Operations

| FUSE op | Action |
|---|---|
| `lookup`, `getattr` | Resolve a path to a live file or a virtual directory (any entry whose name starts with `<path>/`). |
| `readdir` | Prefix-match live entries at the current directory; dedup nested components into virtual-directory entries. |
| `open`, `read` | Load full file contents via `read_file_bytes` (or the dirty write buffer), slice to `(offset, size)`. |
| `write` | Stage into a per-inode buffer; no on-disk mutation until `release` / `flush` / `fsync`. |
| `create` | Allocate an inode, seed an empty dirty buffer; `release` commits via `store_bytes`. |
| `release` / `flush` / `fsync` | Commit: `delete_file` (if exists) then `store_bytes`. |
| `unlink` | Tombstone via `delete_file`. |
| `mkdir` | Record the path in an in-memory `explicit_dirs` set. Nothing persisted — empty dirs vanish on unmount unless a file is stored under them. |
| `rmdir` | ENOTEMPTY if any live entry has the directory's prefix; else drop from `explicit_dirs`. |
| `rename` | For files: read + store-under-new + delete-old. For directories: walk matching entries, rename each. |
| `setattr` | `mode` change → re-store with new mode. Truncate → resize the write buffer. atime ignored. |
| `statfs` | Device block count / free count; block size 4096; namelen 95. |

### Concurrency

Dispatch is single-threaded. `StegoDevice` methods take `&mut self`;
enabling fuser's threaded dispatch would require a global lock that
serializes everything anyway. Documented at the top of
`src/fuse/fs.rs`.

### Filename limits

The file-table entry's filename field is 96 bytes (95 usable + null
terminator, per §6). FUSE full paths share that limit, not just the
basename. Returns `ENAMETOOLONG` if exceeded. A FUSE client doing
`create /mnt/x/a/very/deep/path/…` hits this at ~3–4 directory
levels of reasonable-length names; growing the filename field is a
V2-or-later format change.

### Lifetime

`llmdb mount` blocks until `Ctrl-C` or until `llmdb unmount` runs
`fusermount3 -u` from another shell. Put it in the background with
`&` / `nohup` / `disown` for cross-shell lifetime; the mount dies
with the process that owns it.

## 8. `ask` Command

The `ask` command loads the model for inference, gives it tool-use access to the stego filesystem, and lets it answer questions about files stored in its own weights.

### Architecture

```
User question
    → llmdb ask process
        → load model via llama.cpp (subprocess or llama-cpp-rs)
        → construct system prompt with tool definitions
        → send user question
        → model generates tool calls
        → llmdb executes tool calls against StegoDevice
        → results fed back to model
        → model generates final answer
```

### Tool Definitions

The model receives these tool definitions in its system prompt:

```json
[
    {
        "name": "list_files",
        "description": "List all files stored in the model's weights",
        "parameters": {}
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file stored in the model's weights",
        "parameters": {
            "filename": {"type": "string", "description": "Name of file to read"}
        }
    },
    {
        "name": "file_info",
        "description": "Get metadata about a file (size, permissions, timestamps)",
        "parameters": {
            "filename": {"type": "string", "description": "Name of file"}
        }
    }
]
```

### System Prompt

```
You are a language model with files stored in your own weights.
You can use the provided tools to list and read these files.
When asked about your files, use the tools to look up the information
rather than guessing. The files are stored steganographically in your
weight parameters — you are literally reading data from your own neural network.
```

### Inference Backend

V1 shells out to `llama-server` (from llama.cpp) running locally, communicating via the OpenAI-compatible API it exposes. This avoids linking llama.cpp's C++ into our Rust binary.

Startup sequence:
1. `llmdb ask model.gguf` starts `llama-server --model model.gguf --port <random>` as a subprocess.
2. Wait for the server to be ready (poll the `/health` endpoint).
3. Send chat completion requests with tools to `http://localhost:<port>/v1/chat/completions`.
4. Parse tool-call responses, execute against StegoDevice, feed results back.
5. On exit, kill the subprocess.

## 9. CLI

```
llmdb init <model.gguf> [--lobotomy]
    Parse GGUF, build tensor map, write superblock.
    Reports: total capacity, quant type distribution, eligibility summary.
    If --lobotomy: all tensors eligible, flag stored in superblock.

llmdb status <model.gguf>
    Read superblock and file table.
    Reports: utilization, file count, total stored bytes,
    estimated degradation (based on tier utilization), quant breakdown.

llmdb store <model.gguf> <host_path> [--name <stego_name>] [--mode <octal>]
    Store a file from the host filesystem into the stego device.
    Default stego_name is the host filename.
    Default mode is 0644.

llmdb get <model.gguf> <stego_name> [--output <host_path>]
    Retrieve a file from the stego device to the host filesystem.
    Default output is ./<stego_name>.

llmdb ls <model.gguf> [-l]
    List stored files. -l for long format (permissions, size, timestamps).

llmdb rm <model.gguf> <stego_name>
    Delete a file from the stego device. Blocks returned to free list.

llmdb verify <model.gguf>
    Full integrity scan. CRC32-check all allocated blocks.
    Reports any corrupted blocks.

llmdb mount <model.gguf> <mount_point> [--format]
    Start NBD server, attach block device, mount ext4.
    --format: create ext4 filesystem (destructive, prompts for confirmation).

llmdb unmount <mount_point>
    Unmount, disconnect NBD, flush stego device.

llmdb ask <model.gguf>
    Interactive session. Model reads and reasons about its own files.

llmdb dump <model.gguf>
    Export all stored files to a tar archive on stdout.

llmdb zero <model.gguf>
    Set all stego-eligible LSBs to zero. Destroys all stored data.
    Does NOT restore the original model weights — the original LSB values
    were overwritten on first init and are not recoverable. The zeroed
    model will have slightly different characteristics than the original
    (systematic zero bias in LSBs vs. the original stochastic values).
    To get the original model back, re-download the GGUF.
    Prompts for confirmation.
```

## 10. Crate Structure

```
llmdb/
├── Cargo.toml
├── src/
│   ├── main.rs                  # CLI entry point (clap)
│   ├── gguf/
│   │   ├── mod.rs
│   │   ├── parser.rs            # GGUF header, KV metadata, tensor info
│   │   ├── types.rs             # GGUFType, GGUFValue enums
│   │   └── quant.rs             # Quant type metadata (block sizes, bits per weight)
│   ├── packing/
│   │   ├── mod.rs               # QuantPacker trait
│   │   ├── q8_0.rs
│   │   ├── q6_k.rs
│   │   ├── q5_k.rs
│   │   ├── q4_k.rs
│   │   ├── q3_k.rs
│   │   ├── f16.rs
│   │   └── f32.rs
│   ├── stego/
│   │   ├── mod.rs
│   │   ├── device.rs            # StegoDevice: block I/O, alloc, free
│   │   ├── tensor_map.rs        # TensorMap: eligibility, ordering, address translation
│   │   ├── superblock.rs        # Superblock read/write/validate
│   │   ├── integrity.rs         # Integrity blocks: CRC32 storage and verification
│   │   ├── freelist.rs          # Free list management
│   │   └── redirection.rs       # Block redirection table for shadow-copy
│   ├── fs/
│   │   ├── mod.rs
│   │   ├── file_table.rs        # File table entries, CRUD operations
│   │   └── file_ops.rs          # store, get, ls, rm implementations
│   ├── nbd/
│   │   ├── mod.rs
│   │   ├── protocol.rs          # NBD protocol constants, request/reply structs
│   │   └── server.rs            # NBD server: socket handling, request dispatch
│   ├── ask/
│   │   ├── mod.rs
│   │   ├── server.rs            # llama-server subprocess management
│   │   └── bridge.rs            # Tool-call handling: parse, execute, respond
│   └── diagnostics.rs           # Status reporting, utilization, degradation estimates
├── tests/
│   ├── gguf_parse.rs            # Parse real GGUF files, validate tensor info
│   ├── packing_roundtrip.rs     # Per-quant-type: embed then extract, assert equality
│   ├── stego_block_io.rs        # Write block, read block, verify identical
│   ├── shadow_copy.rs           # Write with simulated crash mid-write, verify old block intact
│   ├── integrity.rs             # Corrupt a block, verify detection
│   ├── file_ops.rs              # Store file, get file, verify identical
│   ├── free_list.rs             # Alloc/free cycles, verify no leaks
│   └── nbd_smoke.rs             # Start server, read/write via nbd-client, verify
└── benches/
    ├── block_read.rs            # Block read throughput (criterion)
    └── block_write.rs           # Block write throughput (criterion)
```

## 11. Testing Strategy

### Unit Tests

Each packing module gets a roundtrip test: generate random data, embed into a random quant block, extract, assert equality. Run for all possible 4-bit values (0x0-0xF) and for blocks with extreme weight values (near min/max of the signed range).

### Integration Tests

Use a real SmolLM3-3B Q8_0 GGUF file (downloaded in CI or from a local cache):

1. **Roundtrip**: Write 100 random blocks, read them back, assert byte-for-byte equality.
2. **Inference survival**: Write blocks, then run a short inference pass through the model (via llama.cpp), verify the model produces coherent text. This is a smoke test, not a perplexity benchmark.
3. **Shadow-copy atomicity**: Begin a block write, simulate a crash (abort the process after step 2 of the shadow-copy sequence), reopen the device, verify the original block is intact.
4. **Integrity detection**: Write a block, flip a random bit in the underlying mmap, attempt to read — verify that CRC32 check fails.
5. **File roundtrip**: Store a file, get it back, compare SHA256.
6. **Fill and free**: Allocate blocks until the device is full, free all of them, verify total_blocks == free blocks.

### Benchmarks

Criterion benchmarks for:
- Block read throughput (blocks/sec, MB/sec)
- Block write throughput (blocks/sec, MB/sec)
- Address translation throughput (translations/sec) — this is the hot inner loop

## 12. Init Sequence

When `llmdb init model.gguf` is run for the first time on a GGUF file:

```
1.  Parse GGUF header, validate magic and version.
2.  Parse KV metadata.
3.  Parse tensor info entries.
4.  Build tensor map:
    a. Classify each tensor (tier, eligibility based on --lobotomy flag).
    b. Skip Q2_K tensors.
    c. Sort by allocation priority.
    d. Compute total stego capacity in bytes and blocks.
5.  Compute layout:
    a. Block 0: superblock
    b. Blocks 1..N: integrity blocks (N = ceil(total_blocks / 1020))
    c. Blocks N+1..M: redirection table (M = ceil(total_blocks / 1022))
    d. Blocks M+1..P: file table (initial: 1 block = 16 file slots)
    e. Blocks P+1..end: data blocks (initially all free)
6.  Write superblock to block 0.
7.  Initialize integrity blocks (all CRC32s set to 0).
8.  Initialize redirection table (identity mapping).
9.  Initialize file table (all entries type=free).
10. Chain all data blocks into the free list.
11. Flush (msync).
12. Report:
    - Total capacity: X MB (Y blocks)
    - Metadata overhead: Z blocks
    - Available for data: W blocks
    - Quant types present: [list with per-type capacity contribution]
    - Lobotomy: yes/no
```

## 13. Crash Recovery

### Write Protocol

Every write operation follows a strict sequence designed so that a crash at any point leaves the device in a recoverable state. The superblock records the in-flight operation so recovery can identify and clean up partial writes.

**Block write sequence:**

```
1.  Allocate shadow block from free list.
2.  Write superblock: set shadow_block = shadow index, shadow_target = logical block index.
    Increment generation counter. Flush.
    (Crash here: superblock says a shadow is in flight. Recovery frees it.)
3.  Write data to shadow block's physical locations. Flush.
    (Crash here: shadow has partial data. Superblock identifies it. Recovery frees it.)
4.  Update integrity block: store CRC32 for shadow block. Flush.
    (Crash here: shadow complete but not yet mapped. Recovery frees it.)
5.  Update redirection table: logical block → shadow physical block. Flush.
    (Crash here: redirection updated but old block not freed. Recovery frees old block.)
6.  Return old physical block to free list.
7.  Write superblock: clear shadow_block and shadow_target (set to 0xFFFFFFFF).
    Increment generation counter. Flush.
```

The key invariant: **the superblock always knows whether a shadow operation is in flight.** If `shadow_block != 0xFFFFFFFF` on device open, the last write did not complete cleanly.

**File store sequence:**

```
1.  Allocate all needed blocks (data blocks + overflow block map blocks if needed).
2.  Write all data blocks. Flush.
3.  Write overflow block map blocks (if any). Flush.
4.  Write file table entry. Flush.
5.  Increment superblock generation counter. Flush.
```

Crash during file store: the file table entry is the commit point. If the entry was not written, the allocated blocks are orphans. If the entry was written, the file is complete.

### Recovery Protocol

On device open, if the dirty flag is set:

```
1.  Read superblock. Validate magic and CRC32.
    If CRC32 fails: superblock itself is corrupt. Unrecoverable without backup.

2.  Check shadow_block field.
    If shadow_block != 0xFFFFFFFF:
        a. The shadow block contains partial or complete data from an interrupted write.
        b. The redirection table may or may not have been updated.
        c. Check the redirection table entry for shadow_target:
           - If it points to shadow_block: the mapping completed. Free the old block
             (determine the old block by scanning — it's the physical block that was
             previously mapped to shadow_target, now orphaned).
           - If it still points to the old block: the mapping did not complete.
             The old data is intact. Free the shadow block.
        d. Clear shadow_block and shadow_target in superblock.

3.  Orphan scan.
    Build a bitmap of all physical blocks that are referenced by:
        - Superblock (block 0)
        - Integrity blocks (chained from superblock)
        - Redirection table blocks (range from superblock)
        - File table blocks (range from superblock)
        - Data blocks referenced by the redirection table
        - Overflow block map blocks referenced by file table entries
    Any allocated block NOT in this bitmap is an orphan.
    Return all orphans to the free list.

4.  Integrity scan.
    For each allocated data block, read it and verify CRC32 against the
    integrity table. Report any mismatches. Do not attempt repair —
    report and let the user decide.

5.  Clear dirty flag. Write superblock. Flush.

6.  Report recovery results:
    - Orphaned blocks recovered: N
    - Integrity failures: M
    - Generation counter: G
```

The orphan scan is O(total_blocks) — walk every metadata structure, mark referenced blocks, free the rest. For ~300K blocks this is fast.

### Generation Counter Usage

The generation counter serves two purposes:

1. **Ordering**: If two metadata structures disagree (e.g., the redirection table says one thing, the free list says another), the structure written at the higher generation is canonical.

2. **Stale detection**: External tools can record the generation counter at a point in time. If the counter has changed, the device has been modified. This is useful for the `ask` command's cache invalidation: if the generation counter hasn't changed since the last session, cached file listings are still valid.

### Unclean Shutdown Detection

The dirty flag (bit 1 of superblock flags) is set on device open and cleared on clean close. If the flag is set on open, run the recovery protocol above. If the flag is clear on open, no recovery needed — skip directly to normal operation.

### Foreign Modification

If the GGUF file is modified externally (fine-tuning, re-quantization, merging), stego data is destroyed. Detected by: superblock magic check fails, or superblock CRC32 fails, or widespread integrity failures during scan. Report and refuse to mount.

## 14. Open Items

- SmolLM3-3B specifically (tensor names, hidden dim, layer count) is
  unverified. Development and benchmarking to date use SmolLM2-135M;
  the code is model-agnostic at the parser level, but the §1 target
  has not been exercised end-to-end. Downloading the 3B variants and
  re-running the perplexity sweep is the next step if
  SmolLM3-specific behavior becomes load-bearing.
- ~~`llama-server` tool-calling chat template for SmolLM3.~~
  Exercised end-to-end for SmolLM2 via `tests/ask_e2e.rs`
  (`LLMDB_E2E_ASK=1`). SmolLM3 itself remains untested pending the
  tensor-name work above.
- ~~NBD vs FUSE decision.~~ **Resolved.** Mount path is FUSE-only;
  NBD was deleted. See §7.
- ~~Address translation hot-path profile.~~ Measured: `map_logical_byte`
  at ~66 ns/byte (~15 M bytes/s) on this workstation via
  `cargo bench --bench address_translation`. Per-byte cost is real
  but not the current bottleneck — the range-update packer path
  (commit 2f51e2d) batches writes so the per-byte translation sits
  inside a loop that's already dominated by mantissa encode work.
  If V2's allocation changes re-introduce whole-tensor scans, this
  becomes the top profile target; fix is per-quant-block batching
  as originally noted.

## 15. V2 Architecture — Inode + CoW Storage

### 15.1 Motivation

V1 is architecturally uniform: the planner picks tensors by role
(FFN before attention) and treats every weight within a chosen
tensor as equally eligible for bit theft. Empirical measurement on
`wikitext-2` (`benches/perplexity-results/`) confirmed what DESIGN
§2 anticipated:

- **F16 / F32 covers:** perplexity flat across 0→95% utilization
  for both SmolLM2-135M and Qwen2.5-0.5B. Mantissa-only theft
  preserves sign and exponent; the model survives full saturation.
- **Q8_0 covers:** catastrophic collapse on `llmdb init` alone.
  SmolLM2-135M: pristine 18.12 → post-init 34M (~1.9M× degradation).
  Qwen2.5-0.5B: pristine 13.50 → post-init 5.6M. The 4-bit theft
  from each int8 weight randomizes magnitudes; architecture-
  independent.

V2's goal is to preserve F16's flat curve while lifting the Q8_0
curve from "dead model" to "within a few PPL of pristine." The
V1 findability drama (see
`tests/calibration_layer0_findability.rs` for the bug and
`commit 60cd5f5` for the partial fix) showed that the way V1
addresses metadata — magnitude-ranking recomputed at reopen —
conflates two requirements that pull in opposite directions:

- **Finding old writes** demands an invariant rule: the ranking must
  not shift when stealable bits change, or readers compute a
  different location than writers used.
- **Planning new writes** wants to use current weight state for
  safety, which is exactly the quantity that shifts under writes.

V2 separates them cleanly. A tiny bootstrap anchor uses an
invariant rule; every other position is persisted. The persisted
positions live in a **Unix-style inode substrate** with
**copy-on-write** writes and **atomic root-pointer commits**.

### 15.2 Layer 0 — Anchor placement

V1 places metadata at stego-block 0, which lands in whatever tensor
the 4-tier sort picks first. V2 replaces this with a small **anchor
region** at positions computable from the cover's structure, using
a rule invariant under any stealable-bit write:

> Rank every eligible stealable-bit position by the **ceiling
> magnitude** of its containing weight — `|w|` computed assuming
> the stealable bits are at their worst-case value — ascending. The
> anchor occupies the N lowest-ceiling positions, sorted by
> `(slot_index, weight_index)` for canonical order. N is the
> anchor payload in bits (~1600, so a two-slot generational root
> fits).

**Why ceiling magnitude, not real magnitude.** Real magnitude shifts
under writes — a mantissa or low-nibble flip changes `|w|`, which
reorders the ranking. Ceiling magnitude depends only on the
*non-stealable* bits of each weight, so the ranking is invariant
under any write V2 makes. The anchor is tiny (~200 bytes) so the
safety trade — picking by worst-case `|w|` rather than current
`|w|` — is negligible in absolute terms.

Per-quant-type ceiling formulas: §15.10.

**Anchor contents — two-slot generational root:**
```
slot0: [ generation u64 | super_root_pointer (16 B) | crc32 u32 ]
slot1: [ generation u64 | super_root_pointer (16 B) | crc32 u32 ]
magic + version header prefix the two slots.
```
Readers read both slots, verify CRC, and pick the slot with the
higher valid generation. Writers always write the inactive slot;
the write becomes the active state on the next mount or
subsequent write. The slot alternation + generation counter is
V2's atomic-commit primitive (§15.9).

**Mount.**
- Both slots fail magic/CRC → `"no V2 filesystem found; run
  llmdb init <cover>"`. Fresh-cover vs corrupted-cover distinction
  is collapsed into "not a V2 filesystem" for the first cut; fsck
  is V3 territory.
- One slot valid → use it.
- Both valid → use the higher generation.

**Explicit init.** `llmdb init <cover>` creates the anchor and the
initial super-root; mounts never auto-init. This keeps "I
accidentally mounted the wrong file" from silently overwriting
non-V2 state.

### 15.3 Layer 1 — Inode substrate

V2 replaces V1's block-indexed superblock / integrity /
redirection / file-table layout with a Unix-style inode substrate.

**Inode.** The fundamental unit. Represents a file, a directory, or
any V2-internal metadata structure:
```
length               u64
direct[12]           12 × Pointer
single_indirect      Pointer    // → chunk of pointers
double_indirect      Pointer    // → chunk of chunks of pointers
triple_indirect      Pointer    // → chunk of chunks of chunks of pointers
```
Direct pointers cover the first 12 chunks of the file; indirect
blocks extend the reach to any practical file size.

**Pointer.** 16 bytes:
```
[ slot u16 | start_weight u32 | length_in_bits u32 | flags u16 | reserved u48 ]
```
`(slot, start_weight)` addresses the first stealable-bit position;
`length_in_bits` gives the chunk's extent. Chunks are contiguous
runs of stealable bits within a single slot.

**Chunk.** A variable-length run of stealable bits in the cover —
not a fixed 4 KB block. Chunk boundaries are picked by
content-defined chunking (§15.6); a "chunk" is whatever run of
bits a `Pointer` refers to.

**Self-similar structure.** The same inode type holds:
- User files (content = raw bytes of the file)
- Directories (content = list of `(name, child_inode_pointer, kind)` entries)
- The dedup hash index (content = BLAKE3-128-keyed map of chunk positions)
- The dirty bitmap (content = one bit per potential chunk-start position)
- The free-run state (content = serialized `BTreeMap` of free runs)
- The ceiling-magnitude bucket summary (content = max-ceiling per 256-weight bucket)

V2-internal structures are addressed by the super-root; user files
are addressed through the root directory.

**Super-root.** The inode referenced by the anchor's
`super_root_pointer`. Contents:
```
magic u32
version u8
root_dir_inode       Pointer
dedup_index_inode    Pointer
dirty_bitmap_inode   Pointer
free_run_state_inode Pointer
ceiling_summary_inode Pointer
generation u64  // mirrors anchor, for sanity-check
```

**Copy-on-write writes.** Every write allocates fresh chunks:
1. Write new data into newly-allocated chunk(s).
2. Write a fresh inode containing pointers to the new chunks.
3. Walk up the containing-inode tree, rewriting each ancestor with
   the updated child pointer.
4. Rewrite the super-root with the updated metadata-inode
   pointers.
5. Write the super-root's new pointer into the anchor's inactive
   slot; bump the generation. The slot flip is the commit point.

Readers snapshot the active root-slot pointer at op-start and
traverse from there; writers serialize on the commit. §15.9
covers the crash-safety invariants; the concurrency model is at
the end of this section.

**Findability.** Every position beyond the anchor is persisted in
an inode's `Pointer`. No ranking recomputation at reopen, no
magnitude-based address derivation for anything larger than the
anchor — all of that drama collapses into "read the super-root,
read the inodes, follow the pointers." The anchor (rule-derived)
is the only thing the invariance argument has to carry.

**Reed-Solomon redundancy is removed.** V1's concern was "what if
fine-tuning drift corrupts metadata?" V2's answer is CoW +
two-slot root + CRC per chunk: corruption is detectable (CRC
fails) and recoverable from the previous root generation. The 40%
expansion cost of RS framing is avoided.

**Concurrency.** V2 is multi-reader / single-writer. Readers
traverse from a snapshot of the active root-slot taken at
op-start; they're immune to concurrent writers because the CoW
tree they see is immutable until the next commit. Writers
serialize on the root-slot commit via a single `RwLock` at the
`Filesystem` boundary.

### 15.4 Layer 2 — Progressive calibration

**Stage A — AWQ (fast).** Single forward pass with a baked-in
calibration corpus (first ~512 tokens of `wiki.test.raw`, stored in
the `llmdb` binary). Produces per-channel salience score
`s_c = mean_batch(|x_c|) · |w_c|`. Implementation is one forward
pass that sums absolute activations per channel; much cheaper than
full inference. Runs during `llmdb init`. Expected cost: seconds
on SmolLM-scale, tens of seconds on Qwen, minutes on 3B models.

**Stage B — Hessian (full).** Accumulate the second-moment
statistic `H = X^T X` across a larger calibration corpus (e.g.,
1024-8192 tokens). Per-column diagonal `diag(H)` gives refined
per-weight salience. Invoked explicitly via `llmdb calibrate
--full <model>` or triggered when the user wants best-possible
placement. Expected cost: minutes to tens of minutes on CPU for
models up to 3B.

**Stage C — Fisher information.** Deferred to V3. Requires second-
order gradient statistics over a data distribution; tractable only
with GPU backends.

**Calibration infrastructure — the implementation tent pole.** V2
needs transformer forward-pass math in Rust. Three paths:

1. **Patched llama.cpp** — add an activation-dumping hook to the
   existing C++ library, pipe stats back to Rust. Fastest to
   implement, ugliest to maintain; requires fork/patch management.
2. **Third-party Rust inference crate** (candle-core, burn) — adds
   ~100K LOC of dependency weight; partial GGUF coverage; would
   likely require upstream contributions to handle all quant types.
3. **Hand-rolled minimal forward pass in Rust** — ~1000-2000 LOC
   covering just the operations needed for activation statistics
   (matmul + the activation functions used by transformers).
   Self-contained; no external deps; no full inference fidelity
   required since we only need first-moment or second-moment
   statistics.

Option 3 is preferred. Decision deferred pending a PoC. This choice
dominates V2's implementation timeline — everything else in V2 is
weeks of engineering; calibration is the one multi-week commitment.

### 15.5 Layer 3 — Allocation

V1's `src/stego/planner.rs` 4-tier ordering is replaced by a
**per-allocation** decision. Every time the filesystem needs to
place a new chunk, the allocator walks this priority:

1. **Content-match.** Hash the desired chunk content (BLAKE3
   truncated to 128 bits); look up in the dedup index. On hit,
   share the existing pointer — zero new cover write.
2. **Dirty-free chunk.** If no content match, pick a free chunk
   whose position has been written before (tracked in the dirty
   bitmap). Writes to already-perturbed weights cause less
   incremental damage than writes to pristine weights.
3. **Pristine free chunk.** As a last resort, pick a pristine free
   chunk from a per-slot priority queue keyed by **max
   ceiling-magnitude over the run's weights**, smallest first.
   Lowest-max-ceiling minimizes worst-case inference damage.

**The protected-region-S% concept of §15.5-original is absorbed.**
A weight with high ceiling magnitude (e.g., a Q8_0 int8 far from
zero) is automatically deprioritized; it only gets used when no
safer chunk is available. There's no hard cutoff — allocation
degrades gracefully as the cover fills.

**Per-slot free-run structure.**
`BTreeMap<(max_ceiling, length, run_id), start_weight>`:
- Keyed so "smallest max_ceiling with length ≥ L" is a range
  query — O(log F) per alloc, F = free-run count.
- Invariant: splitting a free run with `max = M` yields two
  halves, each with `max ≤ M`. So splits only reduce max.
- On merge (free adjacent): new max = `max(left.max, right.max)`.

Alloc cost: O(log F + L) where L = run length (scan to compute
exact max over the new range). Free cost: O(log F + adjacency
merge).

**Ceiling-magnitude bucket summary.** Max ceiling per **256-weight
bucket**, persisted inside V2's filesystem as a regular inode.
~5 MB for a 135M-weight cover, proportional to cover size. Bucket
queries are slightly pessimistic (consider one bucket more than
strictly needed) but precision-adequate for ranking. Computed
once at `llmdb init`; read into memory at mount.

### 15.6 Layer 4 — Content-defined chunking + dedup

**Content-defined chunking (CDC).** Chunk boundaries are picked by
a rolling hash over the data (FastCDC), not by fixed offset.
Parameters: min 1 KB / avg 4 KB / max 16 KB. Benefit: inserting a
byte at the start of a file doesn't shift all downstream chunks —
most stay identical, keeping dedup effective for edited data.

**Content-hash dedup.** On chunk allocation, hash the desired
content (BLAKE3 → 128-bit digest); look up in the dedup index. On
hit, reuse the existing chunk's pointer — the new inode's pointer
references the same chunk. On chunk free, hash the content and
insert `(hash → position)` into the dedup index.

Replaces §15.6-original's bit-match-pool idea: CDC + content-hash
gives the same effect at chunk granularity, with a concrete
persisted index.

**Dedup + cover preservation.** A content-match costs zero cover
writes — strictly better than dirty-preference. All-zero regions
common in user data (file padding, sparse-file holes, uninit
space) collapse to a single stored chunk, reducing cover
perturbation for structured data significantly.

**Pointer sharing and CoW.** A chunk with multiple inode
references is shared. A write to one file's shared chunk triggers
CoW at the chunk level: allocate a fresh chunk with the new
content, update only the writing file's pointer, leave the other
file pointing at the original. Standard dedup-filesystem
semantics.

### 15.7 Layer 5 — Error compensation (GPTQ-style)

After perturbing weight `W[i,j]` by `δW`, the forward operation
`y[i] = Σ_k W[i,k]·x[k]` acquires delta `δy[i] = δW · x[j]`. GPTQ
compensates by distributing counter-adjustments across the other
weights in row `i` such that `Σ Δ[i,k]·x[k] ≈ -δy[i]`, computed
via the Cholesky factorization of the Hessian `H` already
accumulated during Stage B.

Compensating updates:

- Touch **only eligible weights** in the same row/column (avoid
  writing to the protected set).
- Are small individually (delta is distributed); stay within FP
  precision bounds so they don't introduce their own artifacts.
- Accumulate — block N's compensation depends on writes 0..N-1.

GPTQ compensation is most valuable for int-quantized covers where
per-weight perturbation is large. For F16 it's optional — the
uncompensated perturbation is already below measurement threshold.

### 15.8 Layer 6 — Compression

- **Binary files:** `zstd` at default level. ~2-3× reduction on
  typical payloads; mature Rust crate (`zstd`).
- **Text files:** optional BPE tokenization using the cover's own
  tokenizer before compression. Same tokenizer, same vocabulary,
  typically 3-5× reduction on natural text.

Compression composes multiplicatively with the upstream layers:
fewer bits to store → fewer positions to touch → less work for
matching, compensation, and placement.

### 15.9 Layer 7 — Atomic commit and remap

V1 uses shadow-copy-then-swap via a block-level redirection table
for atomic writes. V2's CoW substrate makes atomicity a property
of every write, not a separate mechanism:

**Every write is atomic.** CoW allocates fresh chunks, builds a
fresh inode tree up to a fresh super-root, then atomically swaps
the anchor's active generational slot to point at the new
super-root. The slot-swap is the single commit point. Readers see
either the pre-commit tree or the post-commit tree; no partial
state is observable.

**Slot alternation + generation counter.** Two anchor slots. Each
carries `(generation, super_root_pointer, crc32)`. Reader picks
the slot with higher valid generation. Writer always writes to
the inactive slot (current gen + 1); the write becomes active on
completion.

**Crash safety properties:**
- Crash before slot write → both slots unchanged, reader sees
  pre-commit state.
- Crash mid-slot-write (partial pointer) → CRC fails on that
  slot, reader falls back to the other valid slot.
- Crash after slot write, before fsync of dependent chunk bytes →
  super-root points at chunks that are partially written. Avoided
  by write-barrier ordering: every dependent chunk (file bytes,
  new inode, fresh super-root) is fsynced before the anchor's
  slot-write; the anchor's slot-write is the last operation in
  the commit sequence.

**Remap for calibration upgrades.** On Layer 2 calibration
changes, sensitivity ordering shifts. A remap:
1. Reads all live data from old chunk positions.
2. Allocates fresh chunks via the new ranking + priority order
   (dedup → dirty → pristine-by-new-max-ceiling).
3. Writes data to new chunks, builds a fresh inode tree.
4. Commits via the standard atomic slot swap.
5. Abandoned positions are marked dirty in the bitmap, preferring
   them for future writes — compensating for accumulated damage
   at the abandoned sites.

Damage accumulation bound: each remap's "newly-damaged" set is
`new ∖ old`, typically small because sensitivity ranking is
stable across calibration stages. Over many remaps the
accumulated damage is capped by how different successive rankings
can be.

**Lossless remap (restore pristine values at abandoned positions)
is deferred** pending a design decision on where to host the
shadow store. See §16.4 for the V3 resolution.

### 15.10 Per-quant-type adaptations

**Ceiling-magnitude formulas** — used by Layer 0 anchor placement
and Layer 3 pristine allocation. Defined as
`ceiling(w) = max over stealable-bit values of |w|`; invariant
under writes by construction.

- **F16** (4 stealable = low mantissa nibble). Monotonic in
  mantissa for fixed sign/exp; max when low nibble is all ones:
  ```
  ceiling = |decode_f16(sign, exp, high_6_mantissa | 0xF)|
  ```
- **F32** (8 stealable = low mantissa byte). Same reasoning:
  ```
  ceiling = |decode_f32(sign, exp, high_15_mantissa | 0xFF)|
  ```
- **Q8_0** (4 stealable = low int8 nibble). Two's complement
  requires the two-argument max:
  ```
  H = sign_extend(int8 & 0xF0)       // high nibble as signed byte
  ceiling = max(|H|, |H + 15|) * |scale|
  ```
  - `int8 = 0x08` (H=0): max(0, 15) = 15.
  - `int8 = 0xFF` (H=-16): max(16, 1) = 16.
  - `int8 = 0x80` (H=-128): max(128, 113) = 128.
  - `int8 = 0x7F` (H=112): max(112, 127) = 127.
- **Q3_K** (1 stealable = LSB of 2-bit qs; hmask bit is **not**
  stealable):
  ```
  q_base = ((q_lo_msb << 1) | (hmask << 2)) - 4
  ceiling = |d * scale| * max(|q_base|, |q_base + 1|)
  ```
- **Q4_K** (1 stealable = LSB of qs nibble):
  ```
  q_base = q_high_3 << 1
  ceiling = max(|d*sc*q_base - dmin*m|, |d*sc*(q_base+1) - dmin*m|)
  ```
- **Q5_K** (1 stealable = LSB of qs nibble; qh bit is
  non-stealable):
  ```
  q_base = (qh << 4) | (q_high_3 << 1)
  ceiling = max(|d*sc*q_base - dmin*m|, |d*sc*(q_base+1) - dmin*m|)
  ```
- **Q6_K** (2 stealable = 2 LSBs of ql):
  ```
  q_base = (qh << 4) | (q_high_2 << 2)
  ceiling = max over k ∈ {0,1,2,3} of |d * scale * (q_base + k)|
  ```

Implemented as `read_weight_ceiling_abs(mmap, slot, weight_index)`
in `src/stego/calibration/magnitude.rs`, dispatching on the
slot's quant type. Per-weight cost is O(1) (closed form for
F16/F32/Q8_0; 2-4-value enumeration for K-quants).

**Per-weight allocation plumbing.** Each packer's write path must
respect the allocated-runs bitmap so two chunks never overlap. The
bitmap is consulted at allocation; during a write, the packer
touches only the bits inside its allocated run.

### 15.11 Migration from V1

**No auto-migration.** A V1 cover mounted as V2 will have no valid
anchor (V1 metadata sits at the start of the first-FFN tensor
whose positions aren't ceiling-magnitude-ranked). V2 refuses to
mount such a cover with `"no V2 filesystem found"`. To preserve
V1 data, copy it out via V1 tooling before running `llmdb init`
on the cover.

**Format identifier.** V2's super-root carries `version = 2`. The
anchor's magic bytes distinguish V2 from any other (past or
future) format attempting to live at the same rule-derived
positions.

**V2 CLI additions.**
- `llmdb init <cover>` — creates the anchor and initial super-root.
- `llmdb mount <cover> <mount_point>` — mounts via FUSE;
  multi-reader / single-writer.
- `llmdb status <cover>` — reports filesystem state (free space,
  file count, dedup hit rate, calibration stage).
- `llmdb calibrate <cover> [--full]` — runs Layer 2 calibration
  (future; see §15.4).

### 15.12 Out of scope for V2

- Lossless remap — V3 (§16.4).
- Fisher calibration — V3.
- Training-time integration ("co-train cover to absorb payload")
  — future research.
- Null-space perturbation (store data in directions the forward
  pass doesn't activate) — research.
- Content-addressable cover (payload stored as offsets into
  existing weight bit-patterns) — research.

## 16. V3 Architecture — GPU Acceleration + Lossless Remap

### 16.1 Scope

V3 layers on top of V2 without changing its data model. The stego
device, placement rules, and on-disk format are V2's. V3 adds:

- GPU-accelerated forward pass (replaces V2's CPU calibration).
- Fisher-information calibration (tractable only on GPU).
- In-VRAM scatter map / sensitivity index.
- VRAM dirty-block cache with lazy flush.
- VRAM shadow store that makes lossless remap tractable.
- Tiered hot/warm/cold storage hierarchy.

### 16.2 GPU forward pass

V2 defines WHAT calibration computes (per-channel AWQ salience,
per-column Hessian). V2's implementation choice (option 3:
hand-rolled Rust forward pass) is CPU-bound by design. V3 binds
the same logical calibration to a GPU backend — reuse `ggml-cuda`
or equivalent; the intermediate statistics are the same, the
execution is 10-100× faster.

Consequence: calibration cost for 3B+ models drops from "minutes
on CPU" to "seconds on GPU." `llmdb calibrate --full` becomes
usable as a routine operation rather than a once-a-week chore.

### 16.3 Scatter map in VRAM

V2 keeps the inode tree + free-run state + ceiling-magnitude
bucket summary on disk, read into RAM on mount. V3 moves this
working set into VRAM, exploiting the typically-underutilized
KV-cache slots allocated at model load. Stego chunk I/O becomes a
GPU kernel: a single gather/scatter operation dispatches thousands
of stego-bit reads or writes in parallel. Expected 100× throughput
improvement on chunk-intensive operations (ext4-style metadata
bursts; large-file mount reads).

### 16.4 VRAM shadow store → lossless remap

The V2-deferred decision resolves naturally at V3. VRAM is dense
and cheap to address; hosting a ~50 MB-to-~1 GB shadow of pristine
weight bits for the stego-eligible region is a small fraction of
VRAM on any GPU that can host the model at all.

**Mechanism:**

- At GPU model load, the pristine cover bits of stego-eligible
  weights are copied into a VRAM shadow buffer. (This requires
  either: (a) a pristine reference of the cover file that V3 can
  read — e.g., `models/pristine/` — or (b) the shadow is
  reconstructed from the live cover at load time before any
  additional writes, which is fine if the load is the FIRST V3
  open of a V2-format device).
- Live stego writes update the cover file (mmap) but do not touch
  the VRAM shadow.
- On remap: data is read from old positions, written to new
  positions. Then for each old position no longer referenced,
  the VRAM shadow's original bit is written back — a true restore.
- The shadow is rebuilt from the cover + stored delta on every GPU
  load; it's not persisted to disk.

**Design dependency:** requires either a pristine cover reference
(dual-cover deployment pattern) or accepting that the first V3
load captures the current cover as "pristine," which is only
lossless-remap-correct if no earlier stego damage is present.
Operationally cleaner: document that V3 deployments SHOULD keep a
pristine cover reference alongside the stego cover, same way
`models/pristine/` already does for benchmarking.

### 16.5 VRAM dirty-block cache

Writes land in a VRAM-resident cache, flushed to the GGUF cover on
quantized schedules (threshold blocks / explicit fsync / unmount).
Reduces mantissa re-encoding overhead — the hot write path (ext4
journal, log-heavy workloads) amortizes encode cost across many
writes instead of paying per-write.

Matches standard block-cache semantics: dirty bits tracked,
eviction on cache pressure, write-back on flush.

### 16.6 Tiered degradation

- **GPU hot:** active working set — current read/write blocks +
  shadow + scatter index. VRAM-resident.
- **CPU warm:** larger working set — recently-touched blocks,
  mapped from the GGUF via mmap. Evicted to cold when not accessed.
- **Disk cold:** the GGUF cover file itself; accessed via mmap
  when VRAM and CPU caches miss.

Policy: LRU-ish, with a preference for keeping metadata regions
VRAM-resident since they're consulted on every I/O.

### 16.7 Fisher calibration

Fisher information requires second-order gradient statistics over
a data distribution — requires autodiff-capable forward passes.
Infeasible as a routine CPU operation; tractable as a GPU
operation via standard ML stacks. V3's `llmdb calibrate --fisher`
runs it; results replace the Hessian table in metadata and
trigger a remap.

### 16.8 Rolled forward from V2

Three V2 deferrals close at V3:

1. **Lossless remap** — VRAM shadow makes it cheap (§16.4).
2. **Fisher calibration** — GPU makes it tractable (§16.7).
3. **Large-model calibration** — V2's CPU path becomes impractical
   past ~3B params; V3's GPU path extends the supported range.

### 16.9 Out of scope for V3

- Training-time integration (model learns to absorb its payload).
- Federated / multi-host deployments.
- Content-addressable cover (payload encoded as offsets into
  existing bit-patterns in the cover).
- Novel radical directions: null-space perturbation, co-trained
  covers, implicit-entropy storage. Tracked as research agenda.
