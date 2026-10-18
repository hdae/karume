# 0090: Shard spec v3 — tensor pieces, a single 256 MiB limit measured by file length

- Status: accepted (2026-09-02); superseded by [0108](0108-container-format.md) except decision 4:
  decision 2 on 2026-09-24 (0108 step 3d); decisions 1, 3 and 5 marked superseded on 2026-10-18 (the
  shard layout they govern was retired by the same step). Decision 4 (runtime consumption of pieces)
  stands, restated in 0108 decision 9 and [container-v1](../container-v1.md) §5. Feasibility study:
  [research/2026-09-02-tensor-piece-split.md](../research/2026-09-02-tensor-piece-split.md)
- Date: 2026-09-02
- Related: [0081](0081-shard-spec-v2.md) (this ADR replaced its decision 2, the 1 GiB limit, and the
  writer target of its 2026-09-02 addendum; decisions 1 / 3 / 4 / 5 unchanged) /
  [0070](0070-shard-loading-admission.md) (co-shard rule of decision 1 extended to pieces; decision 3
  unchanged) / [0089](0089-memory-limits-preflight.md) (reader-side shard-limit check) /
  [0063](0063-safetensors-physical-layout.md) (a piece is an ordinary tensor in a shard) /
  [0069](0069-packed-w4-storage.md) (the i4 group scale is cut along rows like pieces)

## Context

ADR 0081's addendum (2026-09-02) set the writer target to 256 MiB, but **tensors could not be
split**, so a component holding a unit larger than the target raised its target to that unit
(effective target = max(target, largest unit)). Two values coexisted: the acceptance limit (1 GiB)
and the target (256 MiB). Three embedding tables were affected (gemma4 `lm_head` 384 MiB, irodori
`tok_embeddings` 300 MiB, anima `embed_tokens` 297 MiB), raising their components' shards to 385 /
300 / 297 MiB. The host RAM peak at load is "constant + one largest shard" (ADR 0070 addendum), so the
raise went straight into the peak.

The two sides also measured the limit differently: the exporter counted the data section, the hub the
manifest `size` (file length including the header). With a single limit, "data section exactly at the
limit + header" would fail in the hub.

## Decision

### 1. Reader contract: pieces (a tensor split along row ranges of its leading dimension)

(The key spelling, the placement rules and the 4-byte rule were superseded by ADR 0108 on 2026-09-24:
pieces are now the `pieces[]` row ranges of the binding table — 0108 decision 18 and container-v1 §5;
the 4-byte rule for middle pieces is generalised to every block by 0108 decision 7. Only the
migration CLI still parses the old key, `parse_piece_key` in `tools/exporter/src/karume/legacy.py`.)

- **Key**: `<parent>#NNNNN-of-NNNNN` (five digits, index from 1, count ≥ 2), read as a piece only when
  the parent is declared (`initializer.tensor`). Tensor names written by Karume never contain `#`.
- **Shape**: a contiguous range of the parent's leading dimension (rows); same dtype, shape
  `[rows_i, *parent.shape[1:]]` with `rows_i ≥ 1`; pieces 1..n cover rows `[0, parent.shape[0])` in
  order without gaps. Each shard stays a self-consistent safetensors file.
- **Placement**: one piece per **consecutive shard**, index increasing with shard order; other tensors
  may share the shards of piece 1 and piece n. Two pieces of one parent in one shard, skipped or
  reversed indices, count mismatches and mixing whole tensor and pieces are violations.
- **Scale**: the companion scale (`storage.scale`) is not split and sits **in the shard of piece 1**.
- **Alignment**: every piece but the last MUST have a byte length that is a multiple of 4 (the runtime
  writes at an offset with `queue.writeBuffer`). The last piece may have any length; the runtime's
  tail padding (f16 / i8) aligns it.
- **Completeness**: an incomplete piece sequence is listed as missing at end of read.

### 2. Reader contract: one limit `SHARD_BYTE_LIMIT` = 256 MiB, measured by **file length**

(Superseded by ADR 0108 on 2026-09-24: the ceiling for host RAM peak and for Chromium's single
ArrayBuffer moved to the 32 MiB block, decision 5; file and request counts moved to the part,
decision 6.)

- Every shard's file length (header included) ≤ 256 MiB; the hub checks the manifest `size`, the
  exporter's verify the real file length.
- One constant (`karume.shards.SHARD_BYTE_LIMIT` = hub `MAX_SHARD_BYTES` = 2^28). The
  `SHARD_TARGET_BYTES` and the effective-target formula of ADR 0081 are abolished. The only derived
  restriction is that a tensor whose **single row** exceeds the capacity cannot be distributed.
- Rationale: the limit is exactly what it adds to the host RAM peak; 256 MiB is the low end that keeps
  file and request counts within the hub's four parallel fetches and the reader's 1024-file cap.

### 3. Writer policy (`karume.shards`)

(Superseded by ADR 0108 on 2026-09-24: `shards.py` no longer exists; the writer packs 32 MiB blocks,
`BLOCK_MAX_BYTES` in `tools/exporter/src/karume/container.py`, into parts — 0108 decisions 5 and 6.)

- Packing capacity `SHARD_DATA_CAPACITY` = limit − header allowance `SHARD_HEADER_ALLOWANCE` (1 MiB).
  A header that exceeds the allowance fails the file-length check of verify; the writer does not
  shrink silently.
- Only units above the capacity (a tensor, or a weight + scale pair) are cut into **1 MiB row blocks**
  (`SPLIT_BLOCK_BYTES`, rounded up to a row count that keeps 4-byte alignment) and fed to the existing
  packer; consecutive blocks in one shard fold into one piece. Units within capacity are never split.
- **Chain pulling**: a pair moves as "scale + all blocks", a contiguous run placed where either half
  is first reached. Pulling only the scale scattered pieces over non-consecutive shards in the spike.
- Deterministic (same input → same split → same bytes), decided from declarations without reading data.

### 4. Runtime (reader-side consumption)

- Pieces are written as they arrive into **one GPU buffer for the parent**, at row offset × row byte
  length, with `queue.writeBuffer`. Buffer allocation, residency bookkeeping and the scale upload
  happen at piece 1; tail padding only at the last piece. GPU layout, kernels and diagnostics are
  unchanged; output is bit-identical to the unsplit tensor.
- Weights expanded to f32 on the CPU (compressed storage consumed outside a weight slot) are expanded
  per piece (the piece shape plus the scale cut along the leading dimension; the expansion function is
  unchanged). They are not re-joined on the host, which would add the whole raw tensor to the peak.
- The per-shard errorScope / fence / release contract (ADR 0070 decision 3) was unchanged at the time.
  ADR 0108 decision 9 later split it into an errorScope per block and a fence per part, and its
  addendum of 2026-09-24 freed the CPU-side release from waiting on the fence; shards no longer exist.

### 5. Deriving `requiredLimits`

(Superseded by ADR 0108 on 2026-09-24: the demand comes from the merged supply plan,
`component_demand_bytes` in `tools/exporter/src/karume/dist.py`, under container-v1 §5 rule ④; a
piece sequence still counts as one parent buffer.)

`karume.limits.max_tensor_payload` sums pieces by parent name (one tensor = one GPU buffer = one
binding).

## Consequences

- Of the real assets only the four components above the limit changed (gemma4 model, anima
  text_encoder, irodori v4 and v4.1 backbone): gemma4 6 → 7 shards (largest 385 → 254 MiB), anima
  text_encoder 6 shards (297 → 233 MiB), irodori backbone 7 → 6 shards (300 → 252 MiB).
- Older runtimes reject pieces loudly; no compatibility layer (unreleased).
- Recipe-side readers that read distributions by key (irodori reference pipeline, vowel_detector
  round-trip check) fold pieces back into the parent.
- ADR 0081 §5 (preferred cut positions) stays open; layer-wise and MoE-wise splits remain writer
  policy extensions.

## Addendum (2026-09-22) — ADR 0108 overrides decision 2

ADR 0108 (accepted 2026-09-22) overrides decision 2; it took effect on 2026-09-24 when step 3d
(`9e905d9f`) removed the old reader. The acceptance limit `SHARD_BYTE_LIMIT` disappeared with that
reader; the limits are now the 32 MiB block (0108 decision 5) and the part length (decision 6). Pieces
remain as `pieces[]` row ranges of the binding table; only the migration CLI's `parse_piece_key`
(`tools/exporter/src/karume/legacy.py`) reads the old `<name>#NNNNN-of-NNNNN` spelling.
