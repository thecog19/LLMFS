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

### V2 — Optimal Embedding + Compression

- Progressive background calibration (AWQ fast, Hessian full)
- Sensitivity-ordered allocation
- Content-aware matching with bucket index
- Error compensation via GPTQ column update
- Scatter map with bootstrap region
- BPE tokenizer transparent compression
- Cross-model comparison reporting (`report --compare`)
- Recalibration support

### V3 — GPU Acceleration

- Scatter map in unused KV cache slots
- GPU gather/scatter for block I/O on weight tensors in VRAM
- VRAM dirty block cache with lazy flush to GGUF
- Tiered degradation (GPU hot → CPU warm → disk cold)

This document specifies V1. V2 and V3 are listed for context; they will be specced separately.

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

## 7. NBD Server

The Network Block Device protocol allows exposing the stego device as a Linux block device. The kernel sends read/write requests; we respond with stego block operations.

### Protocol

NBD uses a simple request/response protocol over a Unix socket or TCP:

```
Client → Server:  NBD_REQUEST { magic, type, handle, offset, length }
Server → Client:  NBD_REPLY { magic, error, handle, data (for reads) }
```

Request types we must handle:
- `NBD_CMD_READ`: Read `length` bytes at `offset`. Map to stego block reads.
- `NBD_CMD_WRITE`: Write `length` bytes at `offset`. Map to stego block writes.
- `NBD_CMD_FLUSH`: Call `StegoDevice::flush()`.
- `NBD_CMD_DISC`: Disconnect. Flush and clean up.

### Implementation

```rust
struct NbdServer {
    device: StegoDevice,
    socket_path: PathBuf,
}

impl NbdServer {
    fn new(device: StegoDevice) -> Self;

    /// Start the NBD server and connect it to /dev/nbdN.
    /// This forks a background process or spawns a thread.
    fn start(&mut self) -> Result<PathBuf>; // returns /dev/nbdN path

    /// Handle one request from the kernel.
    fn handle_request(&mut self, req: NbdRequest) -> NbdReply;

    /// Disconnect and clean up.
    fn stop(&mut self) -> Result<()>;
}
```

The `llmdb mount` command:
1. Opens the GGUF file as a StegoDevice.
2. Starts the NBD server.
3. Runs `nbd-client` to attach `/dev/nbdN`.
4. If the device is unformatted (no ext4 signature), prompts to format.
5. Mounts `/dev/nbdN` at the specified mount point.
6. Waits for unmount signal (SIGINT or `llmdb unmount`).
7. Unmounts, disconnects NBD, flushes stego device.

### Alignment

NBD requests can be for arbitrary byte ranges, not aligned to our 4096-byte block size. The handler must:
- For reads: read the enclosing block(s), return the requested sub-range.
- For writes: read-modify-write for partial block writes. Read the enclosing block, splice in the new data, write back the full block.

ext4 with 4096-byte block size will naturally issue 4096-aligned requests in most cases, but we must handle the general case.

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

- Determine SmolLM3-3B's exact tensor names (need to inspect an actual GGUF file) for the eligibility classifier.
- Determine whether `llama-server` from llama.cpp supports SmolLM3's tool-calling chat template correctly, or whether we need to handle the chat template ourselves.
- NBD requires root or `CAP_SYS_ADMIN` for `nbd-client`. Determine whether FUSE (via `fuser` crate) is a better option for unprivileged use. V1 can support both: NBD for full ext4 support, FUSE as a fallback.
- Determine the actual hidden dimension and layer count of SmolLM3-3B by inspecting the GGUF metadata, to validate capacity estimates.
- Profile the address translation hot path. If per-byte translation is too slow, batch translations per quant block (translate once per 32-weight block rather than per weight).
