# LLMDB Stego Layer: Design Document

## Overview

The stego layer provides a block storage engine backed by the least-significant bits of quantized model weights in a GGUF file. It exposes a flat address space of readable/writable blocks that higher layers (table storage, SQL engine) consume. The model remains functional for inference after data is written — degraded from Q8 toward an effective 4-bit precision regime. This is not identical to a proper Q4_0 quantization (which optimizes scale factors to minimize reconstruction error), but the degradation is in a comparable range. Actual quality impact must be benchmarked empirically for each model.

The punchline: your model gets dumber as you store more data in it.

## GGUF Internals (What We're Working With)

A GGUF file has this structure:

```
┌──────────────────────────┐
│ Header                   │  magic, version, tensor_count, kv_count
├──────────────────────────┤
│ KV Metadata              │  model name, architecture, hyperparams, tokenizer
├──────────────────────────┤
│ Tensor Info Table         │  name, n_dims, shape, dtype, offset (per tensor)
├──────────────────────────┤
│ Alignment Padding         │
├──────────────────────────┤
│ Tensor Data               │  raw quantized weight bytes, contiguous
└──────────────────────────┘
```

For Q8_0 quantization, tensor data is organized in **blocks of 32 values**:

```
┌─────────────────────────────────────────┐
│ Q8_0 Block (34 bytes)                   │
├──────────┬──────────────────────────────┤
│ scale    │ quants[32]                   │
│ (f16,    │ (int8 × 32,                 │
│  2 bytes)│  32 bytes)                   │
└──────────┴──────────────────────────────┘
```

Each `quants[i]` is a signed int8 (-128 to 127). The reconstructed weight is `quants[i] * scale`. We steal the bottom N bits of each `quants[i]` value.

## Bit Stealing Strategy

### Target: 4 LSBs per Q8_0 weight

Stealing 4 bits from an int8 quantized weight forces it into a 4-bit effective precision range. This is *not* identical to a proper Q4_0 quantization — Q4 pipelines optimize scale factors to minimize reconstruction error across the block, while we're replacing LSBs with arbitrary data. The degradation is in a comparable regime (both retain 4 bits of signal per weight), but the error characteristics differ. Proper Q4 concentrates error where it's least damaging; we distribute it uniformly. The actual perplexity impact needs to be benchmarked per-model, not assumed from Q4_0 literature.

**Masking**: For a given int8 value `q`:
```
stored_data = q & 0x0F            # extract bottom 4 bits
degraded_q  = q & 0xF0            # zero out bottom 4 bits (or reconstruct from top 4)
```

To write data bit `d` (4 bits) into weight `q`:
```
q_new = (q & 0xF0) | (d & 0x0F)
```

On read, we extract `q & 0x0F`.

### Tensor Eligibility

Not all tensors are equally robust to bit-stealing. Layer sensitivity roughly follows:

| Tensor Type | Name Pattern | Sensitivity | Policy |
|---|---|---|---|
| Token embedding | `token_embd.weight` | **Very high** | SKIP |
| LM head / output | `output.weight` | **Very high** | SKIP |
| Layer norms | `blk.*.attn_norm.weight`, `blk.*.ffn_norm.weight` | **Very high** | SKIP |
| Attention QKV | `blk.*.attn_q.weight`, `attn_k`, `attn_v` | Moderate | ELIGIBLE |
| Attention output | `blk.*.attn_output.weight` | Moderate | ELIGIBLE |
| FFN gate/up/down | `blk.*.ffn_gate.weight`, `ffn_up`, `ffn_down` | **Low** | ELIGIBLE (preferred) |

**Rationale**: Embedding and LM head layers map directly between token space and weight space — bit errors here cause vocabulary-level corruption. Layer norms are tiny tensors with high per-parameter sensitivity. FFN weights are the bulk of the model and the most noise-tolerant; attention weights are a middle ground.

### Allocation Order

When writing data, we fill tensors in this priority:
1. FFN weights (gate, up, down) — most robust, largest
2. Attention projection weights (Q, K, V, output)

Within each category, we fill layer-by-layer from the deepest layer toward the shallowest. Deep layers are generally more robust to perturbation than early layers, which carry more of the representational load.

### `--lobotomy` Mode

Default mode respects tensor eligibility. `--lobotomy` mode does not.

```
$ llmdb init model.gguf --lobotomy
⚠️  LOBOTOMY MODE: All tensors eligible for storage, including embeddings,
    LM head, and layer norms. Model degradation will be severe and
    entertaining. Proceed? [y/N]
```

In lobotomy mode, the SKIP tensors become eligible and are appended as the lowest-priority allocation tiers:

```
...
12. Attn weights, Q3_K        (standard, last resort)
 — LOBOTOMY BOUNDARY —
13. Token embedding            (vocabulary will start drifting)
14. LM head / output           (output distribution goes weird)
15. Layer norms                 (the model forgets how to normalize)
```

What actually happens as you fill these:

**Embedding layer corruption**: The model's mapping from tokens to vectors develops noise. Common tokens (the, is, a) are high-magnitude and survive well. Rare tokens start mapping to garbage vectors. The model begins avoiding words it can no longer represent — it develops a *shrinking vocabulary*. At high storage utilization, it speaks in simple, common words. It is losing its language.

**LM head corruption**: The inverse mapping — vectors back to token probabilities — gets noisy. The model starts making confident predictions for wrong tokens. Temperature sampling helps mask this. At high corruption, the model becomes a confident idiot — it speaks fluently but says nothing.

**Layer norm corruption**: LayerNorm has very few parameters (hidden_dim × 2 per layer) but they control the scale and bias of every activation. Corrupting these is like adjusting the gain on every channel of a mixing board simultaneously. The model's internal representations lose calibration. Layers that expect inputs in [-1, 1] start seeing [-50, 50]. Activations explode or collapse. This is the fastest path to complete model death.

The intelligence gauge in lobotomy mode adds a new region:

```
Model Intelligence: 12% [██░░░░░░░░░░░░░░░░░░]
  Storage: 1.8 GB / 2.1 GB used
  ⚠️  LOBOTOMY MODE ACTIVE
  Embedding integrity: 34% — vocabulary shrinkage detected
  LM head integrity: 67% — output distribution skewed
  LayerNorm integrity: 89% — activations still calibrated (for now)
  
  Last inference sample:
  > "The the important thing is is that we have the the good data
  >  in the the system and the the results are are good"
```

**Why ship this**: It's a live demonstration of what each component of the transformer actually does. Watching the model degrade as you write data into specific weight categories is a more visceral explanation of transformer architecture than any blog post. The embedding corruption → vocabulary shrinkage pipeline alone is worth a paper (or at least a Twitter thread).

It's also a built-in benchmark. Compare the model's perplexity with 0% lobotomy storage vs. 50% vs. 100%. Plot the curve. The shape of that curve *is* the answer to "how important are embeddings vs. FFN weights vs. layer norms" for that specific architecture.

## Capacity Math

For a concrete target — **Qwen 2.5 3B Q8_0**:

```
                            Standard Mode       Lobotomy Mode
Total parameters:           ~3.09B              ~3.09B
Eligible parameters:        ~2.8B               ~3.09B
Bits per parameter:         4                   4
Raw bit capacity:           ~11.2 Gbit          ~12.4 Gbit
                            = ~1.4 GB           = ~1.55 GB
After metadata overhead:    ~1.3 GB             ~1.45 GB
```

The lobotomy bonus is only ~150 MB because embeddings and norms are small relative to FFN/attention weights. You're trading catastrophic model degradation for an extra 10% capacity. This is a bad trade and we should let people make it.

After reserving space for the superblock, free list, and metadata (see below), usable capacity is approximately **1.3 GB**.

For reference, the Q8_0 GGUF file itself is ~3.3 GB. You get 1.3 GB of storage for free inside a file you were already downloading. This is an absurd amount of space.

## Block Device Abstraction

### Address Space

The stego layer presents a flat array of **blocks** to higher layers. Block size is 4096 bytes (4 KB), matching filesystem conventions and making the SQLite integration trivial (SQLite's default page size is 4096).

```
Block 0        → bytes [0, 4096)
Block 1        → bytes [4096, 8192)
...
Block N-1      → bytes [(N-1)*4096, N*4096)
```

### Physical Layout

Each logical byte maps to 2 Q8_0 weight values (4 bits per weight, 2 weights per byte). The mapping from block address to physical weight offset:

```
byte_offset = block_index * BLOCK_SIZE + offset_within_block
weight_index = byte_offset * 2     # 2 weights per stored byte
```

The `weight_index` then maps into the eligible tensor list. We precompute a **tensor map** at init time: a flat array of (tensor_name, offset_within_tensor) pairs that maps global weight indices to physical file positions, skipping ineligible tensors.

```
TensorMap:
  global_weight[0]      → (blk.31.ffn_down.weight, 0)
  global_weight[1]      → (blk.31.ffn_down.weight, 1)
  ...
  global_weight[K]      → (blk.31.ffn_gate.weight, 0)
  ...
```

### Physical I/O

Each Q8_0 block in the GGUF file is 34 bytes (2 byte scale + 32 byte quants). To read/write weight at `offset_within_tensor`:

```python
block_idx   = offset_within_tensor // 32
byte_in_blk = offset_within_tensor % 32
file_offset = tensor_data_start + (block_idx * 34) + 2 + byte_in_blk
#                                                   ^^^ skip scale bytes
```

We read/write directly in the GGUF file via `mmap` or `seek`+`read`/`write`. No full file rewrite needed.

## On-Disk Metadata

### Superblock (Block 0)

Block 0 is always the superblock. It is never allocated to user data.

```
Offset  Size    Field
0x00    5       Magic: "LLMDB" (0x4C4C4D4442)
0x05    2       Version: 1
0x07    2       Block size: 4096
0x09    4       Total blocks
0x0D    4       Free list head (block index)
0x11    4       Table directory block (block index)
0x15    4       Integrity chain head (first integrity block index)
0x19    4       WAL region start (block index, 0xFFFFFFFF = no WAL)
0x1D    4       WAL region length (blocks)
0x21    4       Shadow block (reserved for atomic writes)
0x25    4       Checksum (CRC32 of superblock)
0x29    ...     Reserved / padding to block boundary
```

### Free List

Simple linked-list free list. Each free block contains:

```
Offset  Size    Field
0x00    4       Next free block index (0xFFFFFFFF = end)
0x04    4092    Unused
```

On init (first use of a model), all blocks except block 0 are chained into the free list.

### Table Directory (Block 1)

Stores the schema for all tables. Simple fixed-size entries:

```
Offset  Size    Field
0x00    64      Table name (null-terminated UTF-8)
0x40    4       Root data block index
0x44    4       Row count
0x48    2       Column count
0x4A    2       Row size (bytes, fixed-width)
0x4C    ...     Column definitions (repeated):
                  32 bytes: column name
                  1 byte:   type (0=INT32, 1=INT64, 2=TEXT64, 3=FLOAT64)
                  1 byte:   flags (nullable, etc.)
```

We support a maximum of ~50 tables in block 1. If you need more tables inside your language model, something has gone wrong in your life, but this limit is soft and can be extended by chaining directory blocks.

### Data Blocks

Rows are stored packed in data blocks. Each data block has a small header:

```
Offset  Size    Field
0x00    4       Next data block for this table (0xFFFFFFFF = last)
0x04    2       Row count in this block
0x06    4090    Row data (packed, fixed-width rows)
```

Rows do not span block boundaries. If a row doesn't fit in the remaining space, it goes in the next block.

## Integrity

### Checksums

Logical blocks are **full 4096 bytes** with no inline metadata. This is non-negotiable — SQLite expects clean pages at the size we declare in the VFS, and shaving 4 bytes per page would silently corrupt every B-tree node.

Integrity metadata lives in **dedicated integrity blocks**, allocated separately from data blocks. Each integrity block stores CRC32 checksums for a range of data blocks:

```
Integrity Block Layout (4096 bytes):
┌──────────────────────────────────────────────────┐
│ Offset  Size    Field                            │
├──────────────────────────────────────────────────┤
│ 0x00    4       Magic: "ICHK"                    │
│ 0x04    4       First data block index covered   │
│ 0x08    4       Count of entries in this block    │
│ 0x0C    4       Next integrity block (chain)     │
│ 0x10    4080    CRC32 entries (1020 × 4 bytes)   │
└──────────────────────────────────────────────────┘
```

One integrity block covers up to 1020 data blocks (~4 MB of logical storage). For a 1.3 GB capacity device, we need ~330 integrity blocks, consuming ~1.3 MB — negligible overhead.

On read, the stego device verifies the CRC32 of the returned data block against the integrity table. On write, it updates the corresponding CRC32 entry and flushes both the data block and its integrity block. The integrity blocks themselves are stored in the same stego bit space as everything else — checksums all the way down.

This is not optional. We are storing data in the noise floor of a neural network. One bad bit-packing routine and your table is silently wrong. We need to know.

### The "Oops I Ran Inference" Problem

If someone loads the GGUF file in llama.cpp and runs inference, **the data is fine**. Inference reads weights but does not write them. The stego data survives any number of inference passes.

If someone **fine-tunes** or **re-quantizes** the model, the data is destroyed. This is the equivalent of reformatting your hard drive. We detect this via the superblock magic — if the magic is gone, we report that the database has been reformatted and refuse to mount.

### The "Model Quality" Gauge

On init, we report what percentage of eligible weight capacity is currently used for storage. This directly correlates with model quality degradation:

```
Storage used:    0%  → Model at full Q8_0 quality
Storage used:   50%  → Half the weights degraded to Q4
Storage used:  100%  → All eligible weights at Q4 quality
```

This should be displayed to the user as a fuel gauge or health bar. "Estimated Model Degradation: 27%".

## Failure and Recovery Semantics

This is the section where we confront the fact that our storage medium is insane.

### The Atomicity Problem

A single 4096-byte logical block write touches **8192 weight values** (at 4 bits per weight in Q8_0). In mixed-quant files, those weight values may be scattered across multiple tensors with different block layouts. A crash, SIGKILL, or power loss mid-write leaves a **torn block**: some weights carry new data, others carry old data, and the logical block is corrupted.

This is not a theoretical concern. The write fan-out is large enough that a single page write takes non-trivial wall time, and the window for interruption is real.

### Write Strategy: Shadow-Copy-Then-Swap

We do not write data blocks in place. Instead:

1. **Read** the current logical block (assembling it from weight LSBs).
2. **Compute** the new block contents in memory.
3. **Write** the new block to a **shadow block** (a reserved spare block from the free list).
4. **Update** the integrity block to point the logical address at the shadow block's CRC32.
5. **Flush** (`msync`).
6. **Update** the block mapping to swap the shadow block in as the canonical location.
7. **Flush** again.
8. **Return** the old block to the free list.

The key invariant: **at no point does a crash leave the canonical block in a partially-written state**. Either the swap completed (new data visible) or it didn't (old data still valid). The shadow block may contain garbage after a crash, but it's not referenced by anything — it's a free block with junk in it, which is fine.

This costs one extra block of capacity for the shadow reservation and doubles the write amplification. Both are acceptable. Correctness in this medium is not negotiable.

### SQLite WAL Mode

SQLite's WAL (Write-Ahead Logging) mode is **required**, not optional. In WAL mode, SQLite writes new pages to a separate WAL file before committing them to the main database. This gives us:

- **Atomic commits**: a transaction either fully commits or fully rolls back
- **Readers don't block writers**: inference can read the GGUF while we write stego data
- **Crash recovery**: on restart, SQLite replays or discards the WAL

In our VFS, the "WAL file" is a second region of stego blocks, logically separate from the main database region. The superblock tracks both regions.

We **do not support** SQLite's legacy rollback journal mode. Rollback journaling requires atomic sector writes at the VFS level, which we cannot guarantee without the shadow-copy mechanism, and the interaction between rollback journals and our write strategy would be nightmarish to verify. WAL mode only.

### Crash Recovery Procedure

On `llmdb init` or `llmdb status`, if the superblock is valid but the WAL region is non-empty:

1. Report that a dirty shutdown was detected.
2. Let SQLite's recovery logic replay or discard the WAL.
3. Run a full integrity scan (CRC32 check on all allocated blocks).
4. Report any blocks that fail integrity checks.
5. If the superblock itself is corrupt, refuse to mount and suggest `llmdb dump` from a backup.

### What We Cannot Recover From

- **Fine-tuning or re-quantization**: Destroys all stego data. Detected via superblock magic check. Not recoverable.
- **External modification of weight values**: Any tool that writes to the weight tensors (merging, pruning, editing) will corrupt stego data. Detected via CRC32 failures. Not recoverable without a dump.
- **Bit-packing bugs**: If the packing code for a quant type has a bug, data written through that code path is silently wrong. This is why the per-format test suite is critical — every quant packer gets a roundtrip test that writes random bytes, reads them back, and asserts equality across the full block address space.

## API Surface

The stego layer exposes a minimal block device interface to higher layers:

```python
class StegoDevice:
    def __init__(self, gguf_path: str, lobotomy: bool = False):
        """Open GGUF file, parse headers, build tensor map, read superblock.
        
        If lobotomy=True, all tensors including embeddings, LM head, and
        layer norms are eligible for storage. The model will suffer.
        """

    def read_block(self, block_index: int) -> bytes:
        """Read a 4096-byte block from weight LSBs. Verifies CRC32 against
        integrity table; raises CorruptionError on mismatch."""

    def write_block(self, block_index: int, data: bytes) -> None:
        """Write a 4096-byte block via shadow-copy-then-swap.
        Atomic: a crash at any point leaves the previous block intact."""

    def alloc_block(self) -> int:
        """Pop a block from the free list. Raises if full."""

    def free_block(self, block_index: int) -> None:
        """Push a block back onto the free list."""

    def total_blocks(self) -> int:
        """Total available blocks."""

    def used_blocks(self) -> int:
        """Currently allocated blocks."""

    def verify_integrity(self) -> list[int]:
        """Full scan: CRC32-check every allocated block. Returns list of
        block indices that fail verification. Empty list = clean."""

    def intelligence_pct(self) -> float:
        """Model intelligence remaining (100% = no storage used)."""

    def diagnostics(self) -> dict:
        """Per-component integrity breakdown. In lobotomy mode, includes
        embedding_integrity, lm_head_integrity, layernorm_integrity.
        Always includes per-tier utilization and estimated perplexity impact."""

    def flush(self) -> None:
        """Ensure all writes are persisted to the GGUF file."""
```

Higher layers (table manager, SQL engine) never touch weight bits directly. They see blocks.

## SQLite Integration (Layer Above)

The cleanest path to real SQL: implement a **SQLite VFS (Virtual File System)** backed by `StegoDevice`. SQLite already knows how to do everything — B-trees, query planning, transactions, WAL — and its VFS layer is specifically designed for exactly this kind of cursed backing store.

This means:
- We get real SQL for free (not a toy parser)
- Transactions work (SQLite handles rollback journaling)
- The query optimizer is battle-tested
- We can throw away the custom table directory / row format above and let SQLite manage its own page format

The custom block format described above becomes a **fallback** for environments where bundling SQLite is undesirable, or for the README flex of "we wrote our own storage engine."

**Recommended approach**: Ship with SQLite VFS as the default. Keep the custom format as `--raw` mode for purists.

## The Full Primitive Stack

The stego storage engine is the foundation, but LLMDB's thesis is that *every* database primitive maps to a transformer primitive. Each mapping below is a real, implemented feature — not a README joke.

### Compression Layer: BPE Tokenizer

The model's BPE tokenizer is a compression algorithm. We use it as one.

Between the SQLite VFS and the stego device, we insert a compression layer. Before a 4096-byte page is written to weight LSBs, it's tokenized through the model's own BPE vocabulary. Each byte sequence that matches a learned token gets replaced by a (shorter) token ID. On read, we detokenize back to raw bytes.

**Why this actually works**: BPE was designed to find and exploit statistical regularities in byte sequences. Text-heavy SQL data — string columns, repeated keywords, structured values — contains exactly the kind of regularities BPE is optimized for. For a table of user records, the tokenizer has almost certainly seen patterns like `"name": "`, `INSERT INTO`, and common English names during training. These compress well.

**Why this is also terrible**: The compression ratio is a function of how much your data resembles the model's training distribution. English text compresses well. JSON compresses decently. Binary blobs, UUIDs, and base64 get *longer* after tokenization because the tokenizer fragments them into single-byte tokens with multi-byte IDs. The compression ratio is a direct, measurable proxy for "how much does your data look like internet text."

```
$ llmdb insert benchmark model.gguf --data english_text.csv
  Compression ratio: 1.7x (BPE tokenizer)
  Equivalent gzip:   2.1x
  Verdict: worse than gzip, but our compressor is also a language model

$ llmdb insert benchmark model.gguf --data random_uuids.csv
  Compression ratio: 0.6x (expansion!)
  Verdict: the tokenizer has never seen your data and is confused
```

Implementation: we read the tokenizer vocabulary from the GGUF file's metadata (it's embedded in the KV section). We use the same BPE merge table the model uses for inference. The page-level compression adds a 2-byte header storing the compressed length, so we know where the real data ends and padding begins.

Compression is on by default, with `--no-compress` to disable it when your data is adversarial to BPE (or for benchmarking purposes).

### Cache Layer: KV Cache

The KV cache is a key-value store. We use it as one.

During an `llmdb ask` session, the model stays loaded in memory. The KV cache — the cached key and value projection matrices from previous tokens — persists across tool-use turns. When the model queries the database, the results enter the context window and become part of the KV cache. Subsequent queries about the same data benefit from the cached representations.

This is a **read cache with semantic locality**. The cache doesn't store raw SQL results; it stores the model's *processed understanding* of those results, encoded as key-value vectors. A follow-up question that's semantically related to a previous query gets a "cache hit" in the sense that the attention mechanism can attend to the cached representations without re-reading from the stego layer.

```
$ llmdb ask model.gguf
> How many users do we have?
  [SQL: SELECT COUNT(*) FROM users]  →  42 users
  KV cache: 847 tokens cached

> What's the average age?
  [SQL: SELECT AVG(age) FROM users]  →  31.4
  KV cache: 1203 tokens cached (356 new, 847 warm)

> Who's the oldest?
  [SQL: SELECT name, age FROM users ORDER BY age DESC LIMIT 1]
  Cache status: schema + prior results warm — model skipped re-reading table structure
```

Cache properties:
- **Capacity**: context window length (model-dependent, typically 4K-128K tokens)
- **Eviction policy**: oldest tokens fall out of the context window. This is LRU by construction.
- **Invalidation**: write operations (INSERT, UPDATE, DELETE) append to the context, so the model sees the mutation and knows cached results may be stale. The model can (and does) choose to re-query after a write. We don't force invalidation — the model's own reasoning serves as the invalidation logic.
- **Persistence**: none. Cache dies when you exit the `ask` session. This is memcached semantics.

### Index Layer: Attention

Self-attention computes pairwise similarity between all positions via Q/K dot products. This is a learned, approximate similarity lookup — which is what a database index does.

During `llmdb ask`, when the model has table schema and sample data in its context, the attention heads are performing index-like operations: identifying which parts of the stored context are relevant to the current query. The model doesn't iterate through every row; it attends to the rows that match.

We make this concrete with a **semantic index mode**. For tables with a designated index column, we maintain a compact representation of the indexed values in the system prompt:

```sql
-- System prompt includes:
-- Table 'users' semantic index on 'name':
-- Entries: Alice(28), Bob(35), Charlie(42), Dave(35), Eve(31) ...
```

When the user asks "tell me about Dave," the attention mechanism performs a similarity lookup over the index entries — the Q/K dot product between "Dave" in the query and "Dave" in the index is high, directing the model's attention to the right entry. The model then formulates a targeted SQL query rather than a full table scan.

This is RAG. We are calling it a database index. Because it is one.

For large tables where the full index doesn't fit in context, we implement **index pagination**: the system prompt contains the first N entries, and if the model's attention doesn't find a match, it can request the next page. This is a B-tree with a branching factor of "however many rows fit in the context window." It's worse than a real B-tree on every axis except that it handles typos and synonyms.

### Write-Ahead Log: The Prompt

The conversation context in `llmdb ask` is an append-only, causally ordered log of every operation performed in the session. Queries and their results are appended sequentially. This is a WAL.

We don't need to implement this — it's how chat inference already works. We just name it correctly.

### Transaction Model: Autoregressive Decoding

Each token is generated sequentially, conditioned on all previous tokens. Causal masking ensures that token N cannot see token N+1. This gives us serializable isolation for free — each "transaction" (tool-use turn) sees a consistent snapshot of everything that came before it and nothing that comes after.

Again, this is just how autoregressive decoding works. We name it.

### Consistency Model: Temperature

```
$ llmdb ask model.gguf --temperature 0
  Consistency mode: STRONG (deterministic replicas)

$ llmdb ask model.gguf --temperature 0.7
  Consistency mode: EVENTUAL (probabilistic, creative)

$ llmdb ask model.gguf --temperature 2.0
  Consistency mode: BYZANTINE (the model is hallucinating and so are you)
```

Temperature 0 gives deterministic, reproducible query results. Temperature > 0 introduces probabilistic variation — the model might rephrase results, infer unstated relationships, or hallucinate data that doesn't exist. This is a tunable consistency knob, and it maps cleanly onto the CAP theorem:

- **Local model, temperature 0**: CP (consistent, partition-tolerant, not always available if your GPU is busy)
- **Hosted API, temperature 0**: CA (consistent, available, not partition-tolerant if the network is down)
- **Local model, temperature > 0**: AP (available, partition-tolerant, not consistent because it's making things up)

This is not a joke. This is a correct CAP analysis. That's the problem.

## K-Quant Support

If it's not complex, is it worth doing. V1 ships with support for every GGUF quantization type worth stealing from.

### K-Quant Block Layouts

The K-quant formats use a **super-block** structure of 256 weights, subdivided into sub-blocks of 16 or 32, with per-sub-block scales and minimums stored at reduced precision. This is where the GGUF spec gets genuinely hairy.

#### Q8_0 (baseline, simplest)

```
Super-block: 32 weights
┌──────────┬──────────────────────────────┐
│ scale    │ quants[32]                   │
│ f16 (2B) │ int8 × 32 (32B)             │
└──────────┴──────────────────────────────┘
Total: 34 bytes per 32 weights
Bits per weight: 8
Stealable bits: 4 → effective Q4
```

Straightforward. Each quant is a full int8. Mask the bottom 4 bits.

#### Q6_K

```
Super-block: 256 weights (16 sub-blocks of 16)
┌───────────────────────────────────────────────┐
│ ql[128]     low 4 bits of 6-bit quants        │  128 bytes
│ qh[64]      high 2 bits of 6-bit quants       │   64 bytes
│ scales[16]  int8 per sub-block                │   16 bytes
│ d           f16 super-scale                   │    2 bytes
└───────────────────────────────────────────────┘
Total: 210 bytes per 256 weights
Bits per weight: 6.5625 effective
Stealable bits: 2 from ql (bottom 2 of the low nibble)
```

The 6-bit quant is split across two arrays: `ql` holds the bottom 4 bits (packed as nibbles), `qh` holds the top 2 bits (packed as bit pairs). We steal from `ql` only — the low bits of the low bits. This degrades the effective precision to ~4 bits, which is the Q4 territory we know is survivable.

**Bit extraction from ql**: Each byte of `ql` contains two 4-bit values (for weights 2i and 2i+1). We steal the bottom bit of each nibble:

```python
byte = ql[i]
lo_nibble = byte & 0x0F
hi_nibble = (byte >> 4) & 0x0F
stolen_lo = lo_nibble & 0x03  # 2 bits from low weight
stolen_hi = hi_nibble & 0x03  # 2 bits from high weight
# Pack: 4 stolen bits per ql byte
```

#### Q5_K_S / Q5_K_M

```
Super-block: 256 weights (8 sub-blocks of 32)
┌───────────────────────────────────────────────┐
│ qs[128]     low 4 bits of 5-bit quants        │  128 bytes
│ qh[32]      high 1 bit of 5-bit quants        │   32 bytes
│ scales[12]  6-bit scales + mins, packed       │   12 bytes  (Q5_K_S)
│  — or —                                       │
│ scales[12] + d(f16) + dmin(f16)               │   16 bytes  (Q5_K_M)
└───────────────────────────────────────────────┘
Total: 176 bytes (M) per 256 weights
Bits per weight: 5.5 effective
Stealable bits: 1 from ql → effective Q4
```

Same split structure as Q6_K but with a 5th bit in `qh` instead of 2 bits. We steal 1 bit per nibble from `qs`, giving us 2 stolen bits per `qs` byte. Degradation is Q5→Q4, which is nearly invisible in benchmarks.

#### Q4_K_S / Q4_K_M

```
Super-block: 256 weights (8 sub-blocks of 32)
┌───────────────────────────────────────────────┐
│ qs[128]     4-bit quants packed as nibbles     │  128 bytes
│ scales[12]  6-bit scales + mins, packed       │   12 bytes  (Q4_K_S)
│  — or —                                       │
│ scales[12] + d(f16) + dmin(f16)               │   16 bytes  (Q4_K_M)
└───────────────────────────────────────────────┘
Total: 144 bytes (M) per 256 weights
Bits per weight: 4.5 effective
Stealable bits: 1 per nibble → effective Q3
```

Now we're in dangerous territory. Q4→Q3 is a real degradation. Perplexity impact is measurable and non-trivial. But it *works* — Q3 models exist and people use them on memory-constrained hardware.

We steal the LSB of each nibble:

```python
byte = qs[i]
stolen_lo = byte & 0x01
stolen_hi = (byte >> 4) & 0x01
# 2 stolen bits per byte
```

#### Q3_K_S / Q3_K_M / Q3_K_L

```
Super-block: 256 weights
┌───────────────────────────────────────────────┐
│ qs[64]      low 2 bits of 3-bit quants        │   64 bytes
│ hmask[32]   high 1 bit of 3-bit quants        │   32 bytes
│ scales[12]  packed 6-bit scales               │   12 bytes
│ d           f16 super-scale                   │    2 bytes
└───────────────────────────────────────────────┘
Total: 110 bytes per 256 weights
Bits per weight: 3.4375 effective
Stealable bits: 1 per 2-bit pair from qs → effective Q2
```

Q3→Q2 is pain. The model will noticeably degrade. But this is LLMDB — we're not here to be responsible. We steal 1 bit per weight from `qs`, giving us 1 stolen bit per quant pair. Capacity is low but nonzero.

Mark these tensors as LAST RESORT in allocation priority.

#### Q2_K

```
Stealable bits: 0
```

There is nothing left to steal. Even we have limits.

### Bit Budget Summary

| Format | Bits/Weight | Stealable | Effective Post-Steal | Capacity (3B model) | Degradation |
|--------|-------------|-----------|---------------------|---------------------|-------------|
| Q8_0   | 8           | 4         | ~Q4 regime          | ~1.4 GB             | Moderate, needs benchmarking |
| Q6_K   | 6.56        | 2         | ~Q4                 | ~700 MB             | Mild to moderate |
| Q5_K   | 5.5         | 1         | ~Q4                 | ~350 MB             | Mild |
| Q4_K   | 4.5         | 1         | ~Q3                 | ~350 MB             | Significant |
| Q3_K   | 3.44        | 1         | ~Q2                 | ~350 MB             | Severe |
| Q2_K   | 2           | 0         | —                   | 0                   | — |
| F16    | 16          | 4         | ~F12                | ~1.4 GB             | Negligible |
| F32    | 32          | 8         | ~F24                | ~2.8 GB             | Negligible |

### Mixed-Quantization Files

Real GGUF files from llama.cpp use **mixed quantization** — different layers get different quant types. A typical Q4_K_M file might have:

- Attention Q/K: Q4_K_S
- Attention V/O: Q6_K
- FFN gate/up: Q4_K_S
- FFN down: Q6_K
- Embedding/head: F16 or Q6_K

The tensor map must handle this. Each tensor entry includes its quant type, and the bit-packing code dispatches on type:

```python
@dataclass
class TensorSlot:
    name: str
    quant_type: GGUFQuantType
    offset: int          # byte offset in file
    n_weights: int
    stealable_bits: int  # per weight, depends on quant_type
    capacity_bytes: int  # total stealable capacity in this tensor

class TensorMap:
    slots: list[TensorSlot]   # ordered by allocation priority
    total_capacity: int       # sum of all slot capacities

    def global_byte_to_physical(self, byte_index: int) -> PhysicalLocation:
        """Map a logical byte offset to a (file_offset, quant_type, bit_position)."""
        # Walk the slot list, accumulating capacity, until we find which
        # tensor and which weight within that tensor this byte maps to.
        # Dispatch to the appropriate bit-packing routine.
```

The complexity is in `global_byte_to_physical` — it's doing a variable-rate address translation across tensors with different bits-per-weight. This is equivalent to a scatter-gather DMA map, which is a fun sentence to put in a README.

### Allocation Priority (Updated)

Within each tensor eligibility tier (FFN → Attention), prefer higher-capacity quant types:

```
1. FFN weights, Q8_0     (4 bits/weight — most capacity, moderate degradation)
2. FFN weights, F16/F32  (4-8 bits/weight — if present, negligible degradation)
3. FFN weights, Q6_K     (2 bits/weight — mild degradation)
4. FFN weights, Q5_K     (1 bit/weight — mild degradation)
5. Attn weights, Q8_0
6. Attn weights, F16/F32
7. Attn weights, Q6_K
8. Attn weights, Q5_K
9. FFN weights, Q4_K     (1 bit/weight — significant degradation, here be dragons)
10. Attn weights, Q4_K
11. FFN weights, Q3_K    (1 bit/weight — severe degradation, last resort)
12. Attn weights, Q3_K
```

The intelligence gauge should reflect not just how many blocks are used, but *which tiers* they're in. A model with 30% storage used entirely in Q8_0 FFN weights is in much better shape than one with 30% used across Q3_K attention weights.

### Intelligence Gauge v2

```
Model Intelligence: 81% [████████████████░░░░]
  Storage: 247 MB / 1.3 GB used
  Degradation: Tier 1-3 only (FFN, Q8/Q6)
  Estimated perplexity impact: +0.3
```

vs.

```
Model Intelligence: 34% [██████░░░░░░░░░░░░░░]
  Storage: 890 MB / 1.3 GB used
  Degradation: Tier 1-11 (including Q3_K attn — god help you)
  Estimated perplexity impact: +4.7
```

The perplexity estimate is a lookup table, not a live measurement. We benchmark each tier combination once and ship the table.

## Resolved Decisions

1. **mmap vs seek/read**: mmap via `memmap2`. Random access at known offsets is the entire workload. Sub-byte bit operations happen in userspace after mapping.

2. **Concurrency**: `flock` on write. Inference-only readers are safe without locking (inference reads weights, never writes them). Document this.

3. **GGUF version compatibility**: Parse version field, support v2 and v3, reject others. Tensor data layout is identical across versions.

4. **K-Quant support**: All formats ship in V1. See K-Quant Support section above.

5. **Data export**: `llmdb dump > backup.sql` and `llmdb load < backup.sql`. Non-negotiable — your data lives in a neural network and you need an ejection seat.

6. **Endianness**: GGUF is little-endian. x86-64 is little-endian. ARM Linux is little-endian. Document the assumption, don't abstract it.

## Implementation Plan

### Language: Rust

This is a portfolio piece, not a prototype. Rust gets us:
- Zero-cost abstractions over the bit-packing math (the `StegoDevice` trait impls for each quant type are monomorphized away)
- `memmap2` for mmap with safe lifetime management
- `clap` for the CLI
- The README says "written in Rust" and a certain type of person immediately takes it seriously

### SQL Engine: SQLite via Custom VFS

We implement a [SQLite VFS](https://www.sqlite.org/vfs.html) backed by `StegoDevice`. The VFS interface is a C API — we go through `rusqlite` with a custom VFS registration via its FFI layer.

SQLite VFS requires us to implement:
```
xOpen, xDelete, xAccess, xFullPathname    — file management
xRead, xWrite, xTruncate, xSync, xFileSize — block I/O
xLock, xUnlock, xCheckReservedLock         — locking
```

Our implementation:
- `xRead` / `xWrite` → `StegoDevice::read_block` / `write_block` (SQLite's default page size is 4096, matching our block size — not a coincidence)
- `xSync` → `StegoDevice::flush` → `msync` on the mmap
- `xLock` → `flock` on the GGUF file
- `xFileSize` → `StegoDevice::total_blocks * BLOCK_SIZE`
- `xDelete` → zero out all stego bits (this is "reformatting your neural network")
- `xFullPathname` → return the GGUF file path with a `.llmdb` suffix (purely cosmetic)

SQLite then handles B-trees, WAL journaling, query planning, and everything else. We get `JOIN`s. We get `CREATE INDEX`. We get `EXPLAIN QUERY PLAN`. All of it running on bits stolen from a language model's weights.

### Crate Structure

```
llmdb/
├── Cargo.toml
├── README.md
├── src/
│   ├── main.rs              # CLI entry point (clap)
│   ├── gguf/
│   │   ├── mod.rs
│   │   ├── parser.rs         # GGUF header/metadata/tensor info parsing
│   │   ├── quant.rs          # Quant type definitions, block layouts
│   │   └── tokenizer.rs      # BPE tokenizer extracted from GGUF metadata
│   ├── stego/
│   │   ├── mod.rs
│   │   ├── device.rs         # StegoDevice: block read/write/alloc/free
│   │   ├── tensor_map.rs     # TensorMap: global_byte_to_physical mapping
│   │   ├── packing/
│   │   │   ├── mod.rs
│   │   │   ├── q8_0.rs       # Q8_0 bit steal/restore
│   │   │   ├── q6_k.rs       # Q6_K bit steal/restore (ql nibble extraction)
│   │   │   ├── q5_k.rs       # Q5_K bit steal/restore
│   │   │   ├── q4_k.rs       # Q4_K bit steal/restore
│   │   │   ├── q3_k.rs       # Q3_K bit steal/restore
│   │   │   └── float.rs      # F16/F32 bit steal/restore
│   │   └── integrity.rs      # CRC32 integrity blocks, superblock magic
│   ├── compress/
│   │   ├── mod.rs
│   │   └── bpe.rs            # BPE tokenizer as page compression layer
│   ├── vfs/
│   │   ├── mod.rs
│   │   └── sqlite_vfs.rs     # SQLite VFS impl over StegoDevice
│   ├── nlq/
│   │   ├── mod.rs
│   │   ├── bridge.rs         # Natural language query → SQL via the model
│   │   ├── cache.rs          # KV cache management: session state, hit tracking
│   │   ├── index.rs          # Semantic index: schema/values in prompt, attention as lookup
│   │   └── consistency.rs    # Temperature → consistency mode mapping, CAP reporting
│   └── diagnostics.rs        # Intelligence gauge, per-tier stats, compression ratios
├── tests/
│   ├── stego_roundtrip.rs    # Write bytes → read bytes, verify identical
│   ├── quant_packing.rs      # Per-format bit packing correctness
│   ├── sql_smoke.rs          # CREATE TABLE / INSERT / SELECT through the VFS
│   ├── compression.rs        # BPE compression ratios: text vs binary vs adversarial
│   ├── lobotomy.rs           # Verify lobotomy mode unlocks restricted tensors
│   └── cap_theorem.rs        # Yes, we have a test for the CAP theorem
└── benches/
    ├── throughput.rs          # Block read/write throughput via criterion
    └── compression.rs         # BPE vs gzip vs uncompressed on various data types
```

### CLI

```
llmdb init <model.gguf> [--lobotomy] [--no-compress]
    Initialize stego superblock in a GGUF file. Prints capacity and intelligence gauge.
    --no-compress disables BPE compression layer (for benchmarking or binary data).

llmdb status <model.gguf>
    Print intelligence gauge, storage utilization, per-tier breakdown,
    compression ratio, and cache stats (if in a session).

llmdb query <model.gguf> "<SQL>"
    Execute a SQL statement against the stego-backed SQLite database.
    Supports full SQLite SQL: DDL, DML, queries, joins, indexes.

llmdb ask <model.gguf> [--temperature N] [--index <column>]
    Interactive session. The model queries data stored in its own weights.
    --temperature 0:   STRONG consistency (deterministic)
    --temperature 0.7: EVENTUAL consistency (probabilistic)
    --temperature 2.0: BYZANTINE consistency (hallucinations)
    --index <column>:  Load indexed column values into prompt for semantic lookup.
    Session displays live KV cache utilization and hit rate.

llmdb dump <model.gguf>
    Export all stored data as SQL statements to stdout.

llmdb load <model.gguf>
    Import SQL statements from stdin into the stego database.

llmdb bench <model.gguf>
    Run the full benchmark suite:
    - Stego throughput (block read/write)
    - BPE compression ratio vs gzip on stored data
    - Perplexity before/after at current storage utilization
    - Inference latency with warm vs cold KV cache

llmdb defrag <model.gguf>
    Compact storage, moving data into higher-priority (lower-degradation) tiers
    if space has been freed. The model gets smarter when you delete data.

llmdb wipe <model.gguf>
    Zero all stego bits. Restore model to original quality.
    "Format C: drive" but for a neural network.
```

### The `ask` Command: How Recursion Works

`llmdb ask` loads the model for inference via `llama.cpp` (we shell out or use `llama-cpp-rs`), constructs a tool-use prompt giving the model access to `llmdb query` as a tool, and lets it reason about the question.

```
$ llmdb ask model.gguf --temperature 0 --index users.name

LLMDB Interactive Session
  Consistency: STRONG (temperature 0)
  Semantic index: users.name (42 entries loaded)
  KV cache: empty

> Who is our oldest user?
  [Attention over index: "oldest" → scanning name entries...]
  [Tool call]: llmdb query "SELECT name, age FROM users ORDER BY age DESC LIMIT 1"
  [Result]: Dave, 35
  Dave is your oldest user at 35.
  KV cache: 847 tokens | Compression: 1.6x BPE

> How about the youngest?
  [KV cache hit: schema + users context warm]
  [Tool call]: llmdb query "SELECT name, age FROM users ORDER BY age ASC LIMIT 1"
  [Result]: Alice, 22
  Alice, at 22.
  KV cache: 1203 tokens (356 new) | Cache hit rate: 71%

> Add a new user: Eve, age 19
  [Tool call]: llmdb query "INSERT INTO users (name, age) VALUES ('Eve', 19)"
  [Cache invalidation: users table mutated, model aware via context]
  [Index update: Eve added to semantic index]
  Done. Eve (19) added. She's now your youngest user.
  Storage: 248 MB / 1.3 GB | Model degradation: 27%
```

The model is querying data stored in its own weights. It is introspecting in the most literal possible sense. The data it retrieves was encoded by corrupting the very parameters it's using to reason about the query. The cache that accelerates its lookups is its own KV cache. The index it uses to find relevant rows is its own attention mechanism. The compression that packed the data was its own tokenizer. The fact that this works at all is the thesis statement of the entire project.

### Distribution

Clone and run:

```
$ git clone https://github.com/thecog19/llmdb
$ cd llmdb
$ cargo build --release
$ ./target/release/llmdb init ~/models/qwen2.5-3b-q8_0.gguf
```

No package registry. No installer. If you can't run `cargo build`, this project isn't for you, and that's fine.

## Non-Goals (V1)

- Encryption of stored data (your data is hidden in a neural network; security through obscurity is the entire point)
- Compression of stored data before stego embedding (the irony of compressing data before hiding it in a model is noted)
- Support for non-GGUF formats (safetensors is a stretch goal — but honestly, GGUF is harder and more interesting)
- Distributed storage across multiple model files (though "sharding" your database across a model zoo is an incredible future punchline)