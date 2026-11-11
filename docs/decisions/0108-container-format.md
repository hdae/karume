# 0108: Karume container format — `krm` / `krg`, blocks and parts, codec ledger

- Status: accepted (2026-09-22 — filed as proposed as the output of step 0 and accepted the same day
  when the maintainer approved the start of step 1; the four codec names of decision 13 were fixed
  at the same time); steps 0–3 implemented (2026-09-24), steps 4–6 open. Partly superseded:
  decision 20 by [0109](0109-manifest-v5-container.md) (container granularity, 2026-09-22),
  decision 12 by addendum 12 (codec owners, 2026-10-11), the bf16 statements of decisions 12 and 19
  and of the addendum 12 table by [0131](0131-bf16-weight-execution.md) (bf16 executes, 2026-10-17). On
  2026-10-18 decision 15 was re-graded to SHOULD, the retention rule of decision 18 to MAY, and the
  "no second implementation in Python" rule of decision 17 was withdrawn. This ADR and
  [container-v1](../container-v1.md) are the canonical sources.
- Date: 2026-09-22
- Scope: spec [container-v1](../container-v1.md) (new, canonical physical format) and
  [ir-v2](../ir-v2.md); runtime `packages/runtime/src/format/` and `src/runtime/`
  (`session-build.ts`, `plan.ts`, `weight-residency.ts`, `executor.ts`); hub `manifest.ts`
  (`karume/5`), `fetch.ts`, `sources/`, `container.ts`; models (the `components` option, QAT,
  speculative decoding, PLE readers); exporter `tools/exporter/src/karume/` with the migration CLI
  and `tools/export-recipes/`; operations (`docs/release-runbook.md`, `tools/release/hf-upload.zsh`,
  `karume dist`, the gate tests).
- **Supersedes** (existing decisions this ADR overrides; the older ADRs carry addenda):
  - ADR [0003](0003-ir-v1.md): the container convention (one safetensors file, graph JSON in
    `__metadata__.karume_ir`), and "change 1 from v0" (per-initializer `storage` in the graph JSON —
    moves to the binding table's `encoding`, decision 17)
  - ADR [0037](0037-karume-monorepo.md) §3 ("no custom `.krm` extension", "one graph = one
    safetensors")
  - ADR [0063](0063-safetensors-physical-layout.md) (safetensors physical layout: no gaps, element
    alignment, fixed write order `F32 → I32 → I4 → even-length F16 → odd-length F16 → I8`)
  - ADR [0081](0081-shard-spec-v2.md) decisions 1 / 3 (shard 0 holds only the graph; always split)
  - ADR [0090](0090-shard-spec-v3-tensor-pieces.md) decision 2 (one acceptance limit
    `SHARD_BYTE_LIMIT` = 256 MiB measured on the file length)
  - ADR [0006](0006-quantization.md) "quantization is confined to the initializer's storage
    declaration (0003)" — now confined to the binding table's `encoding` (decision 17)
  - ADR [0010](0010-symbolic-constant-folding.md): storage dtype `i32` (IR v1 `storage` vocabulary)
    — now the `i32` entry of the codec ledger (decision 12)
  - ADR [0019](0019-i8-weight-execution.md): the scale shape and declaration (keepdim broadcast of
    the weight rank, `storage.scale`, rejected with `group_size`) and the I8 position in the write
    order — replaced by the rank 2 group shape `[shape[rowAxis], rowLength / groupSize]`,
    `encoding.scale` and a declared `rowAxis` (per-channel codecs MUST have `groupSize` = row
    length) of decision 11, and by the retirement of the safetensors layout rules (decision 18)
  - The last four override only how storage is declared. The numeric contracts (i8 is ±127
    symmetric per channel, scale is applied per element at dequantization, bias is f32, semantic
    dtype and storage are separate) do not change.
- Related: [0038](0038-manifest-v1.md) §3 / §4 (execution defaults belong to the manifest — **not**
  changed) / [0041](0041-manifest-v2.md) / [0071](0071-manifest-v3-shards.md) /
  [0075](0075-quant-presentation.md) (`karume/4`, followed by `karume/5`) /
  [0069](0069-packed-w4-storage.md) (packed 4-bit, moved verbatim into the codec ledger) /
  [0097](0097-gemma4-qat-integration.md) (INT2 and mixed QAT storage) /
  [0089](0089-memory-limits-preflight.md) / [0105](0105-packed-static-quantize-activations.md) /
  [0106](0106-device-keyed-references.md) / [0058](0058-numerics-opt-in-contract.md) /
  [assets-layout](../assets-layout.md) / [release-runbook](../release-runbook.md)

## Context

Before this ADR the distribution was "one safetensors file + graph JSON in
`__metadata__.karume_ir`", and the quantized storage dtypes `I4` / `I2` were a safetensors
**dialect** (the official library 0.8.0 rejects them; [limitations](../limitations.md) already said
a custom format would be considered at the next manifest format change). The original benefit,
"opens in any safetensors tool", was **already lost** for quantized presets.

The benefit that remained cost five things, counted on the assets:

1. **Graph / weight separation was a physical-layout hack.** Shard 0 was a graph-only file with an
   empty data section (ADR 0081 decision 1). Of 447 mirror files, 77 carried `karume_ir`, all with
   zero tensors. The IR totalled 14.2 MB, **0.019 %** of the 73.8 GB data sections.
2. **Tensor names carried physical concerns.** A split tensor used a piece key
   `<name>#00002-of-00003`, leaking the layout into the name space.
3. **Auxiliary assets depended on the physical layout.** The PLE sidecar extracted the safetensors
   `byteOffset` itself and read 9,100 B per token in two reads.
4. **Tail zero padding was scattered over the reader.** `writeBuffer` rejects sizes that are not a
   multiple of 4, so the transfer side had three padding branches (whole tensor / last piece).
   Padding is a runtime concern, not a byte-level one, yet every reader rebuilt it.
5. **The fetch unit was one file**, so host RAM peaked at "constant + largest shard" (ADR 0090
   decision 2). The hub issued no HTTP Range requests; range reads existed only inside a warmed
   cache.

Three new needs appeared: **reading unmodified upstream safetensors** (upstream weights for a
supported architecture), **applying LoRA at load time**, and **replacing a single component**
(for example a transformer from another source). All three combine a "graph contract" and a "weight
supply" as separate things; the old format fused both in one file.

Quantization added a fourth need. For gemma4 e2b, ternary 2-bit instead of i4 g32 saves **−40 %**
storage, f16 scales **−10 %**, group 32 → 128 **−15 %** (estimates from real headers). The
declaration vocabulary enumerated `storage.dtype` (`f32 | f16 | bf16 | i8 | i4 | i2 | i32`), so each
new packing grew branches through every layer: a declaration touched **3 files**, execution **22
non-test files and 10–21 WGSL snapshot variants**.

## Decision

### 1. Two container kinds, one physical format

- **`krm`** (model) = graph + trained weights + scales + assets.
- **`krg`** (graph) = graph **without weight blocks**. The definition is "no weight blocks", not
  "weight-independent": QAT bakes activation scales into the `static_quantize` attributes (ADR 0097
  addendum 2), so a `krg` can carry numbers derived from weights.
- The physical format, graph representation and descriptor schema are **identical** for both. Only
  **one character of the 4-byte magic** (`KRMC` / `KRGC`) distinguishes them; the descriptor has no
  `kind` field (the same fact is not stored twice). A reader can reject the wrong kind before
  reading any JSON.
- A `krg` has **no weight binding table** (weight requirements derive from the declared shape and
  dtype in `values`). The **graph descriptor** does hold the table that binds const-region blocks to
  initializers (`const.constants[]` = `{ graph, initializer, block, encoding }`) and `const.length`.
  This is the only way a `krg` alone supplies `const.*`, and the only way it knows the size of
  part 1 (CPU spike ① — "no binding table at all" would leave the const `encoding` without a home).
  Provenance is not inherited; the only assets a `krg` can carry are weight-independent ones such
  as `rope_base`.
- Bundling `krg` files in `@karume/models` is **allowed by the format but not the standard
  distribution**: all graphs after deduplication total 9,444,431 B (9.0 MiB) against a published
  `@karume/models` of 1,408,706 B (1.34 MiB), **6.7×** (alternative 2).

### 2. Header and a **two-document descriptor**

```text
[magic 4 B]   "KRMC" (krm) / "KRGC" (krg)
[u32 LE container version]
[u64 LE graph descriptor length]
[u64 LE model descriptor length]     ※ MUST be 0 in a krg
```

The descriptor is split into a **graph descriptor** and a **model descriptor**.

| Document         | Holds                                                                       |
| ---------------- | --------------------------------------------------------------------------- |
| Graph descriptor | `graphs` (IR v2), the **const-region index**, `capabilities`                |
| Model descriptor | the **binding table**, weight / asset index, part layout, `provenance`      |

The split makes decision 4 ("extract a `krg` by byte copy") possible. With one document, every
`krg` extraction would regenerate JSON, and a `krg` taken from the same graph could differ by
generator version.

The descriptor does **not** hold `session` / `gpuFeatures` / `requiredLimits` / `label` /
`description` (ADR 0038 §3 — execution defaults belong to the manifest; with the bytes input of
`openContainer` the caller passes them).

### 3. Physical layout — single and split forms share one order

Split form:

| Part     | Content                                                                         |
| -------- | ------------------------------------------------------------------------------- |
| 0        | `[header][graph descriptor][model descriptor]`                                  |
| 1        | **const region** (owned by the graph; block offsets are **relative to the region start**) |
| 2 and up | weight / asset blocks                                                           |

The single form **concatenates the same order** (boundaries 64 B aligned). The two forms differ only
in whether the byte stream has cuts, and one descriptor reader serves both.

Const data gets its own part because it is not always small (gemma4 e2b i4: 0.08 %; birefnet 2048
f32: 29.3 MB = 2.5 %; sbv2 F1 front i4: 2.1 MB = **29.7 %**). Preparation fetches part 0 only;
const is fetched when needed.

Part 0 alone closes admission (graph acceptance, execution plan, estimate). This keeps the order
"graph first → admission → weights" as light as before.

### 4. A `krg` is extracted by **byte copy**

`krg` = `[header'][graph descriptor][const region]`. Extraction rewrites **no offset** (const block
offsets are relative to the region — decision 3).

A `krg` extracted from a `krm` and a `krg` written directly from the same graph are byte-identical,
so `krg` identity is decided by **content hash** (not by name or "same architecture"). The graph
descriptor holds the const binding table (decision 1), so an extracted `krg` supplies its consts on
its own.

Byte identity has a second precondition: **the graph descriptor is byte-identical across
re-exports**. This needs fixed IR v2 **serialization rules** (key order, number spelling); they
were a step 1 work item (rules in addendum 1, item 2).

The design draft cited "sharing the RoPE table" in the gemma4 distribution recipe as a precedent;
that does not match the code — the gemma4 RoPE cos / sin tables are no longer distributed and the
host builds them from the `pipelineConfig.rope` declaration
(`tools/export-recipes/gemma4/distribution.py`). Content-hash identity is a new decision here.

### 5. **Block** = unit of fetch, verification and release; limit **≤ 32 MiB** (independent constant)

- A block is fetched, sha256-verified and released after GPU upload independently.
- A block does **not** cross a part.
- **The limit is ≤ 32 MiB** (`length ≤ 33,554,432 B`, inclusive). It is an **independent
  constant**, not derived from the part length (`BLOCK_MAX_BYTES` in
  `packages/runtime/src/format/container/limits.ts` and `tools/exporter/src/karume/container.py`).
  Three reasons (CPU spikes ① / ②, 2026-09-22):
  1. **Splitting does not raise the total digest cost.** One 256 MiB digest runs at 923 MiB/s;
     8 × 32 MiB at 938 MiB/s (ratio 0.984).
  2. **The largest anima transformer weight (2048×8192 f16) is exactly 33,554,432 B, and 56 tensors
     have that size.** With "≤" no tensor is split into pieces; with "<" the same 56 are split.
  3. **A const block cannot be split into pieces** (only the weight binding table can express a
     piece list, and a `krg` has none — decision 1), so a const larger than the limit cannot be
     represented. The largest today is birefnet 2048's f32 constant at **29.3 MB**, about 2.7 MB
     under 32 MiB.
- Digest takes a **`subarray`** (a `slice` copy drops 943 → 641 MiB/s).
- Concurrent digests help (16 × 16 MiB with `Promise.all`: **3,702 MiB/s** — 2.07× sequential
  16 MiB, 4.0× one 256 MiB digest). One whole-file digest cannot get this.
- Browsers were not measured. Chrome copies the digest input into Blink in full (recorded with
  `BYTE_BUDGET` in `packages/hub/src/fetch.ts`), so blocks should help most there (**speculation** —
  the copy is capped at the block length).
- **Possible reduction to 16 MiB**: interleaved runs gave 1,784 MiB/s for 16 MiB against 938 MiB/s
  for 32 MiB. Side effects: ① **the const limit breaks first** (birefnet's 29.3 MB no longer fits;
  the escape hatch is open in container-v1 §10), ② **block count and descriptor size double**. It is
  one constant; the review moved to step 6 (addendum 4, item 14). The Deno RAM-peak harness of steps
  2–3 (`tools/ram-peak/`) is not on the Node toolchain (ADR [0127](0127-node-dawn-toolchain.md)
  decision 5).
- History (2026-09-22): the draft's "cliff at 32 MiB" was an artefact of measuring sizes in
  ascending order; digest size sweeps are run interleaved.

### 6. **Part** = delivery granularity; length **chosen by the writer by model size**

- Part length is the writer's choice of **`{256, 512, 768, 1024} MiB`** (default **256 MiB**,
  kept after the step 3 measurement — addendum 5, item 1).
- The manifest and the descriptor **both** declare every part length, so a reader estimates host
  RAM from declarations alone (ADR 0089 — the size of every buffer is known before the first byte).
- Of ADR 0090 decision 2's three reasons, R1 (RAM peak) and R3 (Chromium's single-ArrayBuffer limit
  2,145,386,496 B) **move to the block limit**. Only R2 stays with the part (file and request count
  within the hub's 4 concurrent fetches — `CONCURRENCY` in `packages/hub/src/fetch.ts` — and
  `MAX_PARTS` = 1024).
- Assets at the time: all 447 files ≤ 256 MiB, largest 254.3 MiB, median 224.0 MiB, total
  68.8 GiB; the 256 MiB exemption for `assets` / `extras` was unused (ADR 0109 retired it).

### 7. Tail padding is **baked by the writer**

`writeBuffer` rejects sizes that are not a multiple of 4. The **writer pads each block to a multiple
of 4**.

- Block starts are 64 B aligned (so offsets are multiples of 4); **block lengths are multiples
  of 4**.
- The padding byte is **fixed at 0x00 (MUST)**. Two hashes cover it: **padding at a block's tail is
  inside the block's sha256** (the writer bakes it, so it is deterministic and bit-identical
  repacking includes padding); **padding between blocks or parts is covered only by the part's
  sha256**.
- Padding is applied only to **a whole block and the last piece of a piece list**, **never to a
  middle piece** (it would overwrite the next piece's start). A middle piece's byte length **MUST be
  a multiple of 4 already**; the writer rounds the row step to `4/gcd(rowBytes,4)` and fails loudly
  when it cannot.
- **ADR 0063's write-order rule retires** (order is no longer needed for alignment). The plan was
  that the reader's three padding branches retire as well; addendum 10 corrected this — two remain
  (`alignF16Payload` in `packages/runtime/src/format/f16.ts`, `alignI8Payload` in
  `packages/runtime/src/format/i8.ts`, called from `session-build.ts`).
- Binding rule ② (formerly "a non-final piece's block length is a multiple of 4") generalizes to
  "**every block length is a multiple of 4**". It applies to **every block, const and assets
  included**; data whose length is not a multiple of 4 (bool tables, odd-length f16) is declared with
  its padding. **Consumers derive the logical length from the declared shape.**

### 8. Integrity — three roles for hashes

- **The descriptor is verified first against an outer expected hash and length** (the manifest's
  FileRef, or the pin the caller of `openContainer` passes). A descriptor cannot vouch for itself.
- **Runtime integrity is the sha256 of the descriptor and of each block.** Warm reads compare the
  recorded hash only: **zero digests** (no gigabyte digest on a cache hit — the existing
  `BYTE_BUDGET` discipline of `packages/hub/src/fetch.ts`). The cold path digests blocks only for
  unverified sources (refined by addendum 2, item 3).
- **The whole-file sha256 is for publishing and repack comparison** (not used at runtime).
- The spec states three hash roles: ① **expected hash of a fetched object** (held outside),
  ② **derived cache key** (input digest + recipe version + conversion settings — known before
  generation), ③ **content hash of a derived object** (computed after generation). One field for all
  three would demand the content hash of something that does not exist yet when importing upstream
  weights.

### 9. GPU transfer and the memory contract

- **No `mappedAtCreation`** (alternative 6). It cannot be unmapped until every block is present,
  which defeats per-block release. `queue.writeBuffer(dstOffset)` is called per block.
- wgpu (Deno) keeps `writeBuffer` staging **until the submit completes** (measured: 2,311 MiB after a
  2 GiB `createBuffer` → 4,359 MiB after `writeBuffer` —
  `docs/research/2026-08-08-vram-oom-misreport.md`). The existing "empty submit +
  `onSubmittedWorkDone` per shard" (ADR 0070 decision 3) is therefore kept as a **fence per part
  (or per budget of in-flight bytes)** (`packages/runtime/src/runtime/session-build.ts`).
- **errorScope per block, fence per part** (measured 2026-09-22, Arc B570 / Deno 2.9.6, 256 MiB,
  median of 5): a push/pop pair costs **1.81 µs** — practically free; an empty submit +
  `onSubmittedWorkDone` costs **13.0 ms**. Pop and fence per block (32 MiB × 8): 127.4 ms (1.73× the
  73.8 ms of the old shape); pop per block with one fence per 256 MiB: **56.5 ms (0.77×)**. Failures
  are attributed per block without a wall-clock penalty.
  **Corrected by addendum 13 (2026-10-27):** the penalty-free claim holds for the push/pop issue
  cost on Deno / wgpu only. Awaiting each block's pop before reading the next one serialises the supply
  on Dawn / Chromium, so the pops are now awaited at the part fence, in block order.
- Overlap is limited by **both count and total bytes** (for example "1 in transfer + 1 being
  received / verified"). Expansion (1 bit → f32 is 32×) has its own processing unit.
- Limits are **kept separately** (none derived from another): descriptor length / nesting depth /
  block count / part-length set / block length / in-flight bytes / expansion area / GPU device
  limits.
- **The whole-ArrayBuffer input stays** (`openContainer` with `{ kind: "bytes" }`, named
  `fromContainer(bytes)` in the draft — addendum 8). Chromium's 2,145,386,496 B limit
  **remains only on this input** (`MAX_SINGLE_CONTAINER_BYTES`); the HF distribution is split and
  never hits it.
- **Addendum (2026-09-24, step 3e) — CPU-side release does not wait for the fence.** The fence
  governs only wgpu staging; WebGPU copies `dataContents` at the `writeBuffer` call, so CPU bytes
  may be dropped when it returns. Session construction reads, uploads and drops one block at a time
  (lazy `WeightBatch.items`): the live JS weight bytes shrink from one part to one item (counting in
  container-v1 §11). For scan-type sources a block is a view of the hub's holding slot, so the bound
  stays one slot + buffers awaiting GC. The fence stays once per part.
  `packages/runtime/tests/gpu_write_buffer_copy_test.ts` pins it (bytes poisoned right after the
  call still give bit-identical output); ADR 0070 decision 3 was amended the same day.
  **Extended by addendum 14 (2026-10-27):** the supply now reads up to 2 blocks ahead of the item
  being uploaded, across part boundaries; the live JS weight bytes are one item + those 2 blocks.

### 10. `encoding` vocabulary — bit width is derived

Each binding table entry carries a supply description:

```jsonc
"encoding": {
  "codec": "<ledger name>",
  "packing": { "blockElements": 8, "blockBytes": 4, "alignBytes": 4 },
  "rowAxis": 0,
  "groupSize": 32,
  "scale": { "block": "<block id>", "dtype": "f32" },
  "zeroPoint": { "block": "<block id>" }
}
```

- **Bit width derives from `blockElements` / `blockBytes`.** A declared `bits` cannot express
  non-integer bits per weight (bit-plane decomposition, block quantization of external formats) and
  does not fix the payload length. The old reader required exactly `numel × bits / 8`; going
  through `packing` **generalizes without changing the accepted set**.
- **Alignment is a codec attribute** (`alignBytes`; the old `DTYPE_ALIGN` was I8 = 1 / I4 = 4 /
  I2 = 4). It depends on whether the GPU binds the data as `array<u32>` and cannot be derived from
  the declaration.
- `groupSize` equal to the row length means per-channel.

### 11. `scale` is unified to the rank 2 group shape, `rowAxis` is declared (**part of IR v2**)

Before, scale had **two shapes** (i8 keepdim broadcast vs i4 / i2 rank 2 group — disjoint accepted
sets), and the i8 **channel axis was not declared but derived from the consuming op**
(`weightChannelAxes` in `packages/runtime/src/runtime/plan.ts`, which fails when consumers disagree).

- An i8 per-channel scale, read as "row = first dimension, group length = row length", **is** the
  rank 2 group shape `[shape[0], 1]` (i2 already used `[N,1]`).
- The only exception is **i8 `conv_transpose1d`** (weight `[Cin,Cout,K]`, channel axis 1); it
  **declares** `rowAxis: 1`.
- The declaration alone fixes the meaning of the scale. (Corrected by addendum 1, item 4:
  `weightChannelAxes` stays as the cross-check against the consuming ops.)
- **Part of IR v2 in step 1.** Acceptance: **the old and new eligibility predicates agree on all 77
  graphs**.

### 12. Codecs live in an **immutable ledger in the repository**

**(Partly withdrawn by addendum 12 on 2026-10-11:** the entry fields `decodeCpu` / `executableOps` /
`wgsl?` and folding the residency predicates into `executableOps` are withdrawn; storage format,
CPU decoding and executability have separate owners.**)**

- **No registration at import time** (it would break the "no module side effects" invariant). The
  ledger is data in the repository (`CODEC_LEDGER` in
  `packages/runtime/src/format/container/codecs.ts`).
- **Three axes are reported separately**: ① readable as a declaration, ② executable, ③ which ops
  keep it compressed and resident. The code already separated them when this ADR was written (i2 is
  ①②③, with ③ only for linear / embedding), so **"accept a new codec as a declaration first" was an
  existing path**. The example of the time, bf16 as ① only, no longer holds **(superseded by ADR
  0131 on 2026-10-17: bf16 executes like f16; `RUNTIME_SUPPORT.storage` lists it —
  `packages/runtime/src/ops/contracts.ts`)**.
- **An unknown codec is rejected before any weight is fetched** (the same shape as the old rejection
  of storage dtypes outside the vocabulary).
- The ledger's purpose was to stop a predicate growing per codec (`eligibleCompressedInitializers` /
  `i2EligibleInitializers` / `i4EligibleInitializers`; adding i2 nearly copied the i4 predicate).
  Addendum 12 accepts the three predicates as the cost of keeping executability out of the ledger.

### 13. The first codecs are four — i8 / i4 / i2 / **ternary**

The ledger names are `int8-sym`, `int4-sym-g`, `int2-off` and `ternary`. The packing, levels, scale
and reconstruction formula of the existing three are **moved verbatim** to
[container-v1 §6](../container-v1.md) (no new value is created).

The fourth, **ternary**:

- The range `{−1, 0, +1}` is a **subset of the i2 range `[−2, +1]`**, so **packing and
  reconstruction are identical to i2** (`u = q + 2` ∈ `{1,2,3}`, code 0 unused, reconstruction
  `fround((u − 2) · s)`).
- **The runtime adds zero lines**; WGSL and CPU decoding reuse i2.
- The exporter adds **one absmean quantizer**.
- It is still **declared as a separate codec**, so assets stay identifiable; the converse does not
  hold. The existing i2 assets are **not** ternary (counting 2-bit codes in the first 4 MiB of
  gemma4-qat e2b's I2 tensors, `q = −2` occurs **5.8–7.4 %**). Reading i2 assets as ternary would
  change about 6 % of all elements.

### 14. Odd bit widths (3 / 5 / 6 / 7) — **bit-plane decomposition is reserved only**

- Method: 32 elements are stored as `bits` u32 words in "planes" (bit j of word b = bit b of
  element j). **Storage is exactly `bits` per element** (no wasted bits) and the flat index keeps a
  power-of-two granularity (32 elements per plane).
- Only **the ledger entry shape** is reserved; implementation starts from the width that is needed.
- **"n elements per u32 word, discard the remainder" is not adopted** (alternative 5).
- **base-243 (GGUF TQ1_0, 5 trits / byte) is not adopted** (alternative 4).

### 15. Low-bit codecs are **not justified by speed**

Low-bit codecs SHOULD NOT be introduced for speed; a speed claim is measured on the target model
family and machine first **(re-graded 2026-10-18 to SHOULD: the evidence is one model family on one
machine — gemma4 LLM decode on an RTX 3080 Ti; DiT models and narrow-bandwidth devices were not
measured)**. On the RTX 3080 Ti (Chrome, 2026-09-19) decode was **not** bandwidth-bound (bandwidth
utilisation 11.5 / 25.5 %; source, recomputation after ADR 0105 and unmeasured machines in
addendum 7). Introducing a low-bit codec for speed would trim the side that is not limiting.
Low-bit codecs **presuppose packed int8 activations (ADR 0105)**, and what they save is
**memory** (the −40 % / −10 % / −15 % of the Context).

### 16. Generalizing the exporter

- The bit-width dependence of RTN / GPTQ (`max_level`) is made a parameter (today a parameter of
  `tools/exporter/src/karume/quant_methods.py`).
- Withdrawn by addendum 9 on 2026-09-25: "ternary absmean gets a rewritten idempotency argument".
  absmean has no fixed point; the rule became "one scale ledger per weight, never recomputed".
- **Codebook schemes are a separate decision** (a two-level scale breaks the bit-identity argument
  "reconstruction rounds once, one f32 multiply" in `packages/runtime/src/format/i4.ts`, so it
  belongs to the numerics opt-in of ADR 0058).

### 17. IR v2 = IR v1 − `storage` (+ decision 11)

- `storage` moves from `initializers[name]` to the **binding table**. The aim is "**one
  architecture, one quantization scheme, one graph**": the graph becomes independent of the PTQ
  preset (storage dtype, group, session options) and of weight replacement with the same config
  (fine-tuning).
- **QAT stays a separate graph**, because the graph is part of the scheme (485 `static_quantize`
  nodes and 403 baked activation scales; `requires.ops`, `values` and `outputs` differ from the
  ordinary preset).
- The merge layer is not thin. **Before any weight is fetched, the graph and the selected bindings
  are merged, and the full set of old storage rules** (i8 / i4 require a scale; i4 `group_size` is a
  power of two ≥ 16; i2 has no group; the row length divides) **plus the admission checks, residency
  plan and estimate run on the merged representation.** It lives in one place in runtime `format/`
  (`packages/runtime/src/format/container/bind.ts` — addendum 4, item 7).
- Withdrawn 2026-10-18: "Python does not implement the rules twice". Python is not present at
  runtime, so the exporter mirrors the merge rules (`bind_graphs` and the `bind.ts` mirror in
  `tools/exporter/src/karume/verify.py`; `container.py`), and cross-language fixtures keep the two
  implementations in agreement.

### 18. Compatibility with the old safetensors format is **cut completely**

- New packages read only `karume/5` and the new container. **No dual reading** (alternative 3).
- Reading the old format is confined to the **migration CLI** (Python): old single file / old shard
  list / old manifest → new `krm` / `krg`. Old inputs MAY be kept, and the migration CLI
  (`karume migrate`, `tools/exporter/src/karume/legacy.py`) MAY stay in the core **(re-graded
  2026-10-18 to MAY: the migration is complete — acceptance ① closed in step 2, and 0.13.0 and 0.14.0
  shipped `karume/5` — so keeping them is a maintenance-cost choice)**.
- **The safetensors reader for upstream checkpoints is kept for its own purpose** (reading
  unmodified upstream weights — not a distribution format).
- Old packages pin old revisions (40-character SHA) and **keep working**. **HF repository history is
  kept** (the runbook's "delete and recreate the repository" is not used).
- History (2026-09-22): the migration covered 11 mirrors / 447 files / 69 GB, 32 golden models,
  test helpers, gates, upload scripts and exporter modules; the 10 pinned repositories were
  re-uploaded at the release (addendum 4, item 13).

### 19. Replacing single components

- Family entry points take `fromPretrained(source, { components: { <role>: ComponentSource } })`.
  The seam is the existing **`ComponentOpener`** (`packages/models/src/hub/components.ts`;
  `AnimaPipeline.#build` already receives `open: ComponentOpener`).
- The checks (other-source weights against the graph declaration, preset vs execution settings,
  `pipelineConfig` overrides) sit **in admission, before any weight byte is fetched**.
- **The runtime does not know the source.** The hub only handles N sources.
- Only three things may be inferred: "which component (key fingerprint), conversion table version,
  storage dtype". **Required declarations**: "base `krg`, `pipelineConfig`, preset, exclusions for a
  superset".
- **diffusers' `strict=False` is not adopted.** Missing and surplus tensors are **all listed and
  rejected**. A single-file checkpoint is a set of "component name → weight supply"; unselected
  components are declared as **exclusions with a reason**.
- Evidence (CPU spike ②, two variants anima-wai-v1.0 / anima-copycat-20260610): the anima
  transformer recipe is **23 renames + 8 deletions** (diffusers 0.39.0 Cosmos 2.0 table) **+ 3 prefix
  rules, 0 numeric conversions** (`model.` is prefixed for IR keys). The **685** checkpoint tensors
  split 1:1 into **DiT 567 + conditioner 118**: **0 missing, 0 surplus, 0 shape mismatches** — no
  reshape, transpose, concatenation or split.
- **The 118 tensors prefixed `llm_adapter.` are a declared routing (`routedTo`, to the
  text_conditioner), not "surplus".**
- **The rename table is an array**: order matters (`adaln_modulation_self_attn.1` before
  `self_attn`), and a JSON object would drop the order contract.
- Upstream weights are bf16, so a **bf16 decoder is required**. (At the time the runtime did not
  execute bf16 — **superseded by ADR 0131 on 2026-10-17**: `RUNTIME_SUPPORT.storage` lists `bf16`,
  `packages/runtime/src/ops/contracts.ts`, decoded by `decodeBf16` in
  `packages/runtime/src/format/bf16.ts`.) Rounding **bf16 → exact f32 → `Math.f16round`** matches the
  distributed f16 weights **567/567 bytes**. The 113 tensors stored as f32 **already hold f16-rounded
  values** (`round_weights_to_f16` in `tools/export-recipes/anima/export.py` runs over the whole
  wrapper, and the emitter chooses the storage dtype separately). Assuming "f32 preset = plain bf16
  → f32" makes all 113 look mismatched.
- The weight recipe is **declarative data** (versioned, read by Python and TS). **It is not a
  language for arbitrary preprocessing** (alternative 7).
- **Open**: nothing detects a change of the upstream table (Python is not present at runtime).
  Whether a `krg` carries a "hash of the supported upstream key set" or the recipe version in the
  derived cache key suffices is decided when there is a real need.

### 20. Split of responsibilities with manifest `karume/5`

**(Corrected by ADR [0109](0109-manifest-v5-container.md) decision 2 and addendum 2, item 1 on
2026-09-22:** the container granularity is component × dtype — `weights.<component>.<dtype>.container`
— not one container per preset, and `karume/5` has no shared-`krg` `graph: FileRef`.**)**

| Owner                | Canonical for                                                                                                                                                                                                                                           |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| manifest `karume/5`  | model / quant lists, default selection, `label` / `description`, `session` defaults, `gpuFeatures`, `requiredLimits`, the **entry FileRefs** (descriptor expectations and the part FileRefs of each container), cross-repository references (per container: descriptor and every part from one repo + revision), `pipeline` / `pipelineConfig` |
| `krm` descriptor     | `graphs` (or a content reference to the graph), binding table, blocks, part layout, assets, provenance, capabilities                                                                                                                                   |
| `krg`                | the compute contract, weight-independent constants and their conditions (config values)                                                                                                                                                                |
| models               | semantic checks of `pipelineConfig`, explicit execution settings take precedence, interpretation of assets                                                                                                                                             |
| distribution         | LICENSE / NOTICE texts (repository root — ADR 0071). For a single `krm` taken elsewhere, the descriptor's `provenance` carries **the license identifier and a NOTICE reference** (not the text)                                                         |

The hub needs each file's `size` / `sha256` before the first byte (cache key, in-flight budget,
buffer size), so **the part FileRefs stay in the manifest** (inside the descriptor only, fetching
the descriptor would be a second round trip).

## Alternatives considered

1. **Only a single `krm` (no `krg`)** — rejected. Upstream weights, LoRA and component replacement
   all combine the graph contract and the weight supply separately. A `krm` with a replacement slot
   is effectively a `krg`, and distributing "just a graph" would ship a `krm` with empty weight
   fields — the accepted set expressed by empty fields, which weakens fail-loudly.
2. **Bundling graphs in `@karume/models` by default** — rejected. Known models get easier, but
   **package and asset versions couple** (a graph update = a JSR release; unknown models wait for a
   package update). Size: 9.0 MiB against a 1.34 MiB package, 6.7×. Bundling is **not forbidden**
   (decision 1), only not the default.
3. **Keeping dual reading of old and new formats in a transition version** — rejected (the first
   draft and design v2 §9 proposed it; **retracted**). Pins are 40-character revisions, so old
   packages keep working; dual reading only helps a new package read old assets. A **one-time
   conversion in the migration CLI** covers that, while dual reading in the core would keep every
   rule to retire (old checks, piece keys, physical-layout hacks, verbatim `karume_ir`) alive in the
   reader. The old format is distinguishable by magic, so the migration CLI can read it when needed.
4. **base-243 ternary (like GGUF TQ1_0, 5 trits / byte) at 1.6875 bits per weight** — rejected. It
   saves **only −15.6 %** over 2-bit packing and loses much. A 5-element granularity is not a power
   of two, so **both** invariants "the word position comes from the flat index by shift" and "a row
   starts on a word boundary" break. The row length `K` needs a `% 5` divisibility condition, and
   none of gemma4's real shapes (1536 / 2048 / 6144 / 12288) is divisible by 5. (The draft's further
   reason "worse given that decode is ALU-bound" was withdrawn by addendum 6 on 2026-09-25 as a
   misquotation; the rejection stands on the two broken invariants and the divisibility condition.)
5. **n elements per u32 word, discarding the remaining bits** (3 bit: 10 elements + 2 bits wasted) —
   rejected. Effective bits per element grow to 3.200 / 5.333 / 6.400 / 8.000 (+6.7 % for 3 / 5 / 6
   bits, +14.3 % for 7), and **elements per word stop being a power of two**. The GEMV family unrolls
   each word fully, so no division appears in the loop, but **the scale group index**
   (`(unit·step) >> shift` in `packages/runtime/src/kernels/linear-gemv.ts`) assumes powers of two,
   and shifts would become multiplications and divisions. `i4GroupShift`
   (`packages/runtime/src/kernels/weight-storage.ts`) requires a power-of-two group for the same
   reason.
6. **`mappedAtCreation` to reduce staging copies** — rejected; it works against the goal. A mapped
   buffer **cannot be released until every block is present and it is unmapped**, which breaks the
   core of this ADR (fetch, verify, release per block). `mappedAtCreation` is used **nowhere** in the
   repository (weights, inputs and resident tensors all use `queue.writeBuffer`).
7. **A declarative language for arbitrary weight preprocessing** — rejected. Mapping upstream FQN ↔
   IR key needs a finite set of operations (slice / split / transpose / alias / constant supply /
   exclusion with reason); the anima transformer needed **23 renames + 8 deletions + 3 prefix rules,
   0 numeric conversions** (decision 19). An arbitrary language would ① need a guarantee that the
   Python and TS implementations agree, ② grow an interpreter inside the distribution like
   `karume_ir`, ③ make checks "unknown until run". **Numeric conversions are a separately defined
   finite set** (dtype, reduction axis, operation order, rounding and non-finite handling fixed;
   comparison fixtures of the converted tensors in both Python and TS).

## Consequences

- **This is a breaking change.** The distribution format, the manifest (`karume/4` → `karume/5`),
  the IR (v1 → v2) and the public reading API move together; the CHANGELOG lists it under Breaking.
  The project is released, so this is handled as a **version split**: old pins keep working, new
  pins use the new format (decision 18).
- **Retired**: ADR 0063's write order; `repack.py` invariant ② (verbatim `karume_ir` — IR v2 is
  re-serialized; invariant ①, raw bytes identical, **is inherited**); the piece key spelling
  `<name>#00002-of-00003`; the graph-only shard 0. (The plan also listed the reader's three tail
  padding branches and `weightChannelAxes`; addendum 10 keeps two padding branches and addendum 1,
  item 4 keeps `weightChannelAxes` as a cross-check.)
- **Lost**: the HF safetensors preview (already lost for i4 / i2 presets), "opens in generic
  safetensors tools", a re-upload of the 10 published repositories with 10 new pins, and the
  `.safetensors`-based operation scripts (`hf-upload.zsh`, runbook, gate tests).
- **Gained**: one entry point; the graph becomes independent of PTQ presets and fine-tunes (shared
  `krg`); a block index and verification base (partial fetch once Range is added); PLE and other
  assets become first-class without layout hacks; unmodified safetensors, LoRA and component
  replacement compose through **one binding table**.
- **A `krg` can carry only weight-independent assets**, so "ship a `krg` and reproduce anything"
  does not hold. QAT activation scales living in the graph (decision 1) are one example; `krg`
  identity is decided by **content hash and dependency conditions** (not by name or "same
  architecture").
- **The step 0 open item is closed**: splitting the synchronous `pushErrorScope('validation')`
  section per block is practically free (decision 9 — push/pop 1.81 µs). The cost is the fence
  (13.0 ms), so the fence stays per part (or per budget).
- **Speed** of low-bit codecs is not a claim of this ADR (decision 15, SHOULD since 2026-10-18).
  Ternary is not expected to decode faster; it saves memory. A speed claim needs measurements on
  the side of ADR 0105 (packed int8 activations) first.

## Staging and acceptance

Steps 0–3 are done (2026-09-22 to 2026-09-24; the details are in addenda 1–5). Steps 4–6 are open;
[backlog](../backlog.md) tracks them and this table is canonical for what they build and how they
are accepted.

| Step  | Builds                                                                                                                                                                      | Acceptance                                                                                                                                                                                                                                                                                         |
| ----- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **0** | This ADR + container-v1; four CPU spikes (container read / write, upstream anima weights → binding table, LoRA → rewritten graph, codec ledger) — done 2026-09-22           | Round trip byte-identical; extracted `krg` = independently written `krg`; all 685 upstream tensors bound; ledger decoding bit-identical to `decodeI8` / `decodeI4` / `decodeI2`; errorScope cost measured (decision 9)                                                                              |
| **1** | IR v2 (with serialization rules), TS reader, Python writer and migration CLI — done 2026-09-22 (addendum 1)                                                               | CPU verbatim comparison old ↔ new; eligibility predicates agree on 77 graphs; output bytes compared; existing sha256 reference values and goldens kept (general rule: ADR 0106 and ADR 0110 decision 7); re-export gives a byte-identical graph descriptor                                            |
| **2** | manifest `karume/5`, block-level hub reads, PLE as assets, component replacement, one re-upload — done 2026-09-23 (addenda 2–3)                                             | `fromPretrained` green on a new pin; RAM-peak harness; zero warm digests; replacement rejected in admission with every missing / surplus tensor listed                                                                                                                                             |
| **3** | Migration of all assets, gate tests / runbook / `hf-upload.zsh` / `karume dist`, part-length review — done 2026-09-24 (addenda 4–5; pins moved at the release)             | All lanes green, CHANGELOG Breaking; gate tests read manifest contents; part length re-measured (256 MiB kept)                                                                                                                                                                                      |
| **4** | Import of the upstream anima transformer as f16 (bf16 decoder, comparison fixtures, `index.json`, streamed fetch of large files)                                             | Upstream weights → TS path produce tensors that match the distributed bytes **without exception** (f16 preset). The RTN path is checked against a tolerance                                                                                                                                          |
| **5** | LoRA A/B at load time (PEFT → kohya)                                                                                                                                         | ① adapter with scale 0 **identical** to the base graph ② matches a CPU reference with the same reconstructed base and A/B formula ③ one unconsumed tensor throws ④ added nodes, lost fusions, dispatch count and wall time recorded (quality against PEFT evaluated separately)                     |
| **6** | New bit widths / ternary kernels, Range fetch, composition at load, re-fusion                                                                                               | Measured case by case                                                                                                                                                                                                                                                                              |

Step 5 estimate (CPU spike ③, 2026-09-22): the anima transformer IR has 2,603 nodes, 454 `linear`
and 0 `static_quantize`; inserting LoRA into all 454 adds 3 nodes each (2,603 → 3,965), reuses the
6 existing zero biases, passes the IR parser, and loses only the `linearStaticQuantize` fusion (0 in
anima).

## Addendum 1 (2026-09-22) — points settled while implementing step 1

The spec lives in [container-v1](../container-v1.md) and [ir-v2](../ir-v2.md); this addendum keeps
the reasons.

1. **Initializer name = the key of the data** (IR v2). In v1 the initializer name was the
   torch.export placeholder and the upstream key (FQN) sat in a separate `tensor` field. Moving
   `tensor` to the binding table would lose the key for upstream import (decision 19) and for LoRA
   target resolution (step 5). Naming by FQN (constants `const.<hash>`) needs no extra table (77
   mirror graphs: all names ASCII, longest 87 characters, 0 collisions). A shared initializer's
   **borrower name = the lender's initializer name** (`shared.tensor` is gone).
2. **Canonical serialization** (the condition of decision 4): no whitespace; keys in schema order;
   maps keyed by name in code point order; numbers spelled as ECMAScript `Number::toString` (JS
   `JSON.stringify` is canonical as is; Python implements it). Descriptor arrays have a fixed order
   (`const.blocks` by offset ascending, MUST; `blocks` by (part, offset); `constants` by (graph,
   initializer)).
3. **Five spec corrections** (container-v1 revision history): ① `capabilities.codecs` moves to the
   model descriptor's `codecs` (the graph descriptor MUST be byte-identical between `krm` and `krg`,
   so it cannot hold fields that depend on the binding table); ② the binding table has no borrowing
   form (shared declarations live only in the IR; the binding table's key set equals exactly "the
   initializers that are neither shared nor const-supplied"); ③ `rowAxis` / `groupSize` / `scale`
   only for quantized codecs; ④ `int8-sym` packing is 1 element / 1 byte / alignment 4 (4 / 4 / 1
   would add a `numel % 4 == 0` constraint absent in v1); ⑤ fixed array order (item 2).
4. **`rowAxis` is canonical in the declaration, but the cross-check against the consuming op's axis
   stays** (corrects "`weightChannelAxes` disappears" of decision 11 / container-v1 §13.2). A
   declaration that disagrees with the consumer would make the GPU-resident path apply the scale to
   another axis — a silent wrong value — so `planWeightResidency` keeps the cross-check. For the
   group codec (`int4-sym-g`) `rowAxis` is 0 only (the expansion kernel and `decodeI4` treat the
   first dimension as the row).
5. **Zero-element degenerate shapes** (`in_features = 0` and the like): per-channel `groupSize`
   cannot satisfy "≥ 1" with row length 0, so it is **1** and the group count is **1** (one
   per-channel scale per row, matching the old `[rows, 1]`). The `scaleShape` formula of §6.1
   includes this case.
6. **Shared-initializer check**: borrowers declare no storage, so "declared storage dtype equals the
   lender's" is gone; instead the **expected residency is derived from the lender's codec and the
   borrower's consumption (eligibility)** and compared with the lender's actual residency (for i8 /
   i2 down to the row axis). Semantic dtype equality was added by ADR
   [0096](0096-speculative-decoding.md) (addendum 2026-10-10).
7. **Temporary glue until step 3** (not dual reading): old readers converted to the merged form at
   read time; consumers saw only the merged vocabulary. Closed in step 3d (addendum 4, item 6).
8. **errorScope per block, fence per part** (implementation of decision 9): one empty submit + wait
   per supply unit `WeightBatch` (one part), errorScope per item (block).
9. **A `krg` alone cannot build a Session** (no weight supply) — by design. Graphs supplied only by
   consts and shared initializers are the exception.
10. **Public surface**: `openContainer` / `prepareContainer` / `createSessionFromContainer` /
    `codecLayout` (codec → expansion path; the only place where models branch on storage) /
    `ContainerFormatError`.
11. History: the miniature of acceptance ③ (`gpu_container_session_test.ts`, `krm` vs the old
    path) was deleted with the old path; successor in addendum 4, item 10.
12. **The Python writer writes one graph per container** (`write_model_container(...,
    graph_name=…)`, bindings as a flat map by tensor key; one component = one graph; several graphs
    per container remain a format capability). Part length enforces only the ceiling (1024 MiB).
    (Assets: addendum 2, item 4; `provenance.writer`: addendum 4, item 4.)
13. **`__proto__` keys are blocked on both sides** (container-v1 §0).
14. **Migration CLI surface**: per component (1 component = 1 graph = 1 container); `--license`
    required; `.partial` → check → replace (container-v1 §12). (Graph name and `provenance.writer`
    defaults corrected by addendum 4, items 3 and 4.)
15. History: acceptance at step 1 — 3 real components compared verbatim, predicates agreed on all
    77 graphs (18,515 initializers, 0 mismatches); full comparison closed in addendum 3, item 7.

## Addendum 2 (2026-09-22) — rulings for step 2

The manifest shape is canonical in ADR [0109](0109-manifest-v5-container.md); this addendum holds
only corrections and additions to this ADR.

1. **Correction of decision 20 — container granularity is component × dtype, not preset.**
   `quants[].container` becomes `weights.<component>.<dtype>.container`; presets keep the
   `karume/4` mapping. Basis: on the 11 real mirrors, one physical file per preset would turn
   65.9 GiB into 197 GiB (×2.99) — a preset is not a weight unit (gemma4's three presets share the
   same weights and differ only in execution options). "Several graphs per container" remains a
   format capability, unused in `karume/5`. Model-level `assets` (tokenizers and the like,
   cross-repository allowed) stay in the manifest. The shared-`krg` `graph: FileRef` is not in
   `karume/5` (added at step 5 when needed).
2. **The fetch unit of step 2 is the part** (as decision 6 says). The hub returns one `BlockSource`
   per container; the cold path warms the part with phase 1 of the fetch layer (stream the whole
   file, verify sha256, record the hash) and then opens range reads. Reads branch on cost type —
   seek (browser Blob, local positional reads) reads per block, scan (Deno and Node via HF) reads
   the part once and cuts it. The fetch layer (`@hdae/fetch-cache`) is not changed. **HTTP Range
   stays at step 6**; the trigger to bring it forward is "the RAM-peak harness shows a cold peak above
   'part length + overlap'".
3. **Addition to decision 8 — block sha256 applies only to unverified sources.** Bytes whose whole
   file the fetch layer verified (via HF) are declared verified by the `BlockSource`, and
   `readBlock` does not digest them (no double digest on cold reads). The bytes input of
   `openContainer` and local directories (ADR 0086 decision 2 — the hub does not check whole-file
   sha256 there) are digested per block. Warm stays at zero. Block sha256 remains the contract for
   "verify what arrived" under the Range fetch of step 6 (container-v1 §7 corrected).
4. **Correction of addendum 1, item 12 — assets are taken in step 2.** PLE becomes assets (role
   `ple-values` / `ple-scales` block lists — for range reads one block = one part — and role
   `ple-index`, index schema 3); the `rope_base` of `extras` becomes an asset (role `rope-base`,
   66 KB × 2 copies). Assets declare `assets[].length` (payload length — ADR 0109 decision 4).
5. **In step 2 `karume/5` is written by the whole-repository mode of the migration CLI**
   (`karume migrate --manifest`). It reads shard lists from the old manifest, which closes the
   cross-directory shard lists and the full comparison left open in addendum 1. Recipes write `krm`
   directly from step 3; `hf-upload.zsh` followed `*.krm` already in step 2 for the re-upload.
6. History: coexistence during step 2 — see addendum 3, item 8.
7. History: the re-uploaded repository was irodori-v4.1-small (widest component × preset mapping
   without cross-repository references).
8. **The RAM-peak harness (acceptance ②) is automated on Deno** (`--expose-gc` +
   `Deno.memoryUsage().external`), with manual Chrome steps; fetches (part 0 / const / block) are
   counted separately. (The harness was not ported to Node — ADR 0127 decision 5.)
9. History: sub-steps 2a spec → 2b hub → 2c runtime → 2d exporter → 2e models → 2f acceptance.

## Addendum 3 (2026-09-23) — points settled while implementing step 2

The manifest shape is canonical in ADR [0109](0109-manifest-v5-container.md), PLE in ADR
[0085](0085-ple-host-gather.md) (addendum 2026-09-22).

1. **Assets declare their logical length** (`assets[].length` — container-v1 §2.2). Assets have no
   shape, so the payload length cannot be recovered from the descriptor and consumers had to guess
   the trailing 0x00 (the PLE index JSON was the case). Parsing checks "block length = logical
   length rounded up to a multiple of 4" (MUST).
2. **Range-read assets: one block = one part; whole-read assets may share parts with each other**
   (never with weight blocks). The writer's `AssetInput.dedicated_part` tells them apart. gemma4
   E2B's `model` container has 83 parts (8 weight + 74 asset + descriptor + const).
3. **The hub's container surface does not warm** (`openContainerSource` is synchronous and reads
   ranges only). The caller warms with `prefetchAssets`, in the order descriptor (part 0) →
   admission → weight parts → assets. Warming every part inside the surface would break "no weight
   byte is fetched for a model that cannot run". The phase-1 capability check of `runPrefetchPhase`
   became per ref, fixing a bug where a local session plus HF cross-repository references skipped
   warming.
4. **The models seam** (`packages/models/src/hub/components.ts`): `ModelComponent` carries, besides
   `graph` / `createSession`, the container's asset declarations (`assets`: name → role) and a
   reader (`asset(name)`). Admission decides from each component's graph and asset declarations;
   assets that share no part with weight blocks (index, `rope_base`) may be read there. The
   whole-file surface `assetComponentOpener` opens every component first (before admission) to
   return a synchronous supply.
5. **Component replacement**: `ComponentSource = { source, model?, quant? }` (the same role from
   another `karume/5` repository). Admission = ① the graph descriptor sha256 equals the manifest's
   declaration ② no missing / surplus bindings (`openContainer`) ③ the family's admission check.
   ① needs only the two manifests and fails before any descriptor is fetched. For families that bake
   activation scales into the graph (QAT), the graph descriptor differs per checkpoint and
   replacement is rejected (until step 5 declares a graph-identity policy).
6. **Three facts found by the whole-repository migration of real mirrors**: `pipeline` is the string
   `"<name>/<major>"`; `gemma4-qat` also owns PLE; the PLE index of non-QAT karume-gemma4 is schema 1
   (I8, no `storage` field) — schema 3 always spells `storage` (normalized to `i8`).
7. History: acceptance ① (CPU verbatim comparison) closed on 2026-09-23 — 108 unique containers,
   40,939 blocks, 31,882 initializers, 66.8 GiB, no missing or surplus binding.
8. History: during step 2 the old `karume/4` mirrors lived outside the repository
   (`~/workspace/karume-models-v4/`), and only irodori-v4.1-small was re-pinned.
9. **PLE discipline**: the default residency limit is "two largest blocks" (about 64 MiB; formerly
   two largest shards ≈ 506 MiB). The reader has no cost type, so the seek / scan policy table is
   gone; the floor for falling back to a whole read is `min(32, rows per block)`. A single read
   cannot be cut short (`AssetReader.read` takes no signal), so cancellation happens only between
   gather steps.
10. History: acceptance ②③ — warm runs did zero payload digests and zero cache writes on three
    real configurations; seek-type sources stayed within "one part + one block", Deno via HF (scan)
    about twice that ([research](../research/2026-09-23-container-ram-peak.md)). The harness
    `tools/ram-peak/matrix.ts` is not on the Node toolchain (ADR 0127 decision 5).

## Addendum 4 (2026-09-24) — points settled in steps 3a–3d

One writer (3a), series outputs moved (3b), tests and tools followed (3c), old readers removed (3d).
Part length and RAM peak are in addendum 5. Format: [container-v1](../container-v1.md); manifest:
ADR [0109](0109-manifest-v5-container.md).

1. **One writer** (3a, `8086ed37`). The recipes' `publish_model` / `export_to_file` and
   `karume migrate` pass the same three publishing steps (write → re-read and verify → replace —
   `karume.publish.publish_container`); two writers would make migrated mirrors and re-exported
   series different things. `provenance` and `graph_name` are required (no default that misstates
   the origin; no working name `<component>.staging` picked up as the graph name). Identity is pinned
   by "old shards → `karume migrate`" and "direct write" giving byte-identical part lists (f32 / f16
   / i8 / i4, `rope_base`, PLE dedicated parts). `karume repack` and the old shard writer retired;
   only the migration module `karume.legacy` reads the old format. Recipes write `krm` directly
   (`d2630521`).
2. **`karume dist` builds `karume/5` and checks the files against the declarations** (3a, 3b). It
   writes `weights.<component>.<dtype>.container` (the two descriptor expectations + part FileRefs)
   and fails before assembly when the set of graph names differs from the weights keys
   (`DistError`). The IR acceptance rules (runtime support + op contracts — `assert_runtime_support`
   / `assert_op_contracts`) run with the same functions in export, `karume dist` and `karume verify`.
   dist and verify rebuild the IR from the stored descriptors (`verify.ir_graph_from_container`);
   export checks the graph before writing.
3. **Correction of addendum 1, item 14 — container graph name = component name = the weights key of
   `karume.json`** MUST (3b, `2bbd013b` / `0b86439e`). Rule text: container-v1 §2.1; reason: ADR
   0109 addendum 1. Directory names are not part of the rule (directories may carry the series name,
   and spellings differ: irodori's `caption-proj` has key `caption_proj`, deberta's `full-24layer`
   has key `text_encoder`). Recipes name the graph by a constant; the AST check
   `tools/export-recipes/tests/test_graph_names.py` verifies that it is a constant and that it maps
   to a weights key of the distribution plan (and owns the exceptions). On the file side, the
   `karume dist` check of item 2 compares container graph names with weights keys.
4. **Correction of addendum 1, item 14 — `provenance.writer` is not written by default** (3b,
   `2bbd013b`). A default generator tag would make part 0 of migrated and directly written containers
   differ forever by version, and the same asset could not be claimed byte-identical. The tool
   version lives once, in the `generator` field of `karume.json`.
5. History: all assets migrated (54 series / 135 components / 70.9 GiB via `python -m
   migrate_series`, values bit-identical — `0b86439e`); the 32 golden fixtures rebuilt with
   `karume.goldens` (`2bde1baf`); migration stays in the recipes, not the core (ADR 0065).
6. **Addendum 1, item 7 (temporary glue) closed** (3d, `9e905d9f`, decision 18). The runtime lost
   the old safetensors-dialect reader, the shard progression checks and the IR v1 loader; Session
   supply has one container kind (`BoundContainer`). `parseSafetensors` stays for plain safetensors
   assets and rejects only the dialect dtypes `I4` / `I2`. The hub dropped the sequential shard
   surface (`streamAssets`) and the reused buffers (`into` / `readFileInto`). Tests and tools moved to
   the container surface in 3c (real-asset e2e opens part lists by positional reads —
   `packages/runtime/tests/helpers/container-files.ts`; the series → component → graph-name table
   shared by the gate tests and e2e is `packages/models/tests/helpers/series-graphs.ts`).
7. **The merge layer of decision 17 is `packages/runtime/src/format/container/bind.ts`** (3d). Two
   stages: `bindDeclarations`, independent of the descriptor documents, and the wrapper
   `bindGraphs` that builds lookups from the two documents; `krm` and in-memory containers (item 8)
   both pass `bindDeclarations`. The old storage rules are split in two: the presence of scale /
   `groupSize` / `rowAxis` (required for quantized codecs, forbidden otherwise) is checked by the
   descriptor reader (`parseEncoding` in `descriptor.ts`) and again in `memory.ts` for the
   in-memory input; values (codec × semantic dtype, `groupSize` range, row-length divisibility, i2
   shapes, scale length), piece coverage, middle pieces as multiples of 4 and integer row bounds are
   in `bind.ts`. The IR reader reads one IR v2 declaration without storage (`parseIrDeclaration` /
   `parseIrDeclarationValue`). Names in the decisions and the staging table (`parseIrGraph`,
   `prepareModel`, `format/container.ts`) are step 0 names.
8. **In-memory containers** (3d, `9e905d9f`) — `openMemoryContainer`
   (`packages/runtime/src/format/container/memory.ts`) puts bytes at hand on the same merge,
   admission and construction path as a `krm` **without writing a container file**, returning the
   same `BoundContainer` as `openContainer`. It serves graphs that models build on the host (irodori
   CFG composition / Euler update, gemma argmax selection, PLE gather) and the tools' synthetic
   models; it replaces the old `createSession(gpu, model)` without adding a format.
   - **Input**: `graphs` (graph name → IR v2 declaration) and `tensors` (graph name → initializer
     name → supply). Every non-shared initializer MUST be supplied exactly (missing and surplus are
     all listed and rejected).
   - **Two supply forms**, mutually exclusive: `bytes` (one whole buffer — returned as passed, not
     copied) and `pieces` (two or more first-dimension row ranges with a `read()` — no bytes held;
     each `readBlock` calls `read()` again). Quantized codecs carry an f32 scale (rank 2, 4 B
     aligned) and `groupSize` in `encoding`.
   - **Part split**: whole supplies are stacked in declaration order and move to a new part past the
     default part length (256 MiB) (a companion scale stays with its data); each piece is one part,
     with the scale in piece 1's part (rule ③). Parts are the fence unit of Session construction, so
     the split bounds construction-time staging (host RAM does not depend on it — addendum 5,
     item 3).
   - **No sha256**: the bytes passed in crossed neither network nor disk, so there is nothing to
     verify the declaration against (block digests are for unverified sources — decision 8).
   - **Checks**: binding rules run in the same merge layer as for `krm` (item 7). This layer adds the
     presence re-check of item 7 and five checks of its own — correspondence with declarations
     (missing / surplus, undeclared graph names), `pieces` count ≥ 2, collisions of synthesized block
     ids, 4 B alignment of scale byte positions, and `readBlock` length equal to the synthesized
     declared length. Host-built declarations always pass the public `parseIrDeclarationValue` (the
     same check `openContainer` applies). The descriptor reader checks non-finite numbers and nesting
     depth for `krm`; for in-memory containers the caller does (doc of `packages/runtime/mod.ts`).
9. **Public surface after 3d** (snapshot): runtime removed `openModel`, `KarumeModel`,
   `ModelShard`, `createSession`, `createSessionFromShards`, `prepareModel`,
   `estimateSessionMemory`, `ContainerError` and added the in-memory container types
   (`openMemoryContainer`, `BoundContainer`, `MemoryContainerInput`, `MemoryEncoding`, `MemoryPiece`,
   `MemoryTensor`), `parseIrDeclarationValue`, `IrDeclaration`, `RuntimeSupportError`; hub removed
   `streamAssets` / `StreamedAsset` / `StreamAssetsOptions`. The five names of addendum 1, item 10
   remain. Missing capabilities raise `RuntimeSupportError`, container rule violations
   `ContainerFormatError`; the estimate is `prepareContainer(...).estimate()`.
10. **Successor of the miniature test of addendum 1, item 11** (3d).
    `packages/runtime/tests/gpu_memory_container_test.ts` compares `krm` with an in-memory container
    on the same synthetic model (linear i4 + group scale → add f16 → linear i8 + per-channel scale +
    bias f32, with piece splits, split differently on purpose). The two paths share everything from
    the merge layer to Session construction, so the test also compares against a CPU reference chain
    (`decodeI4` / `decodeF16` / `decodeI8` + `applyReferenceOp`) as an independent oracle (restored
    in `bafe1316`, 2026-09-24). The estimate A/B (`runtime_prepare_container_test.ts`) has the same
    shared-path shape.
11. History: acceptance ② passed (3c, `dcd6fb1c`) — `distribution_gate_test.ts` reads `karume.json`
    with the hub parser and checks part existence and length for the default model / quant;
    `packages/e2e/tests/e2e_assets_gate_test.ts` opens containers of series without a manifest. Other
    presets' parts are checked only when an e2e test opens them.
12. History: acceptance ① passed conditionally (CHANGELOG Breaking in `994bdad8`); on the Arc B570
    development machine 12 of 14 lanes were green, the two red lanes being environment issues listed
    in [known-issues](../known-issues.md) ("Intel Arc B570").
13. **Moving all pins goes from step 3 to the release** (corrects step 3 of the staging table and
    the "until step 3" of addenda 2 / 3). In step 3 no pin moved; only irodori-v4.1-small pointed at
    `karume/5`. The other 9 repositories were re-uploaded and re-pinned at the release. The order of
    [release-runbook](../release-runbook.md) §0 (version bump → rebuild → upload → pin → JSR
    publish) is a non-commutative MUST: the manifest's `generator` field copies the package version,
    so uploading before the bump would publish a distribution naming an old version. A re-upload
    deletes old files the new manifest no longer references from the same repository and pins the
    `main` after the deletion (done for irodori-v4.1-small in `1c952948`: 65 old safetensors,
    5.84 GiB; old pins resolve from history). Deleting and recreating a repository is not used for
    repositories with published pins (decision 18, "history is kept"). `hf-upload.zsh` followed
    `*.krm` in step 2 (`f33f7be2`). The manifest side is ADR 0109 addendum 1.
14. **The block-limit review (32 → 16 MiB) was not measured in step 3** (step 3e measured only part
    length). It moves to step 6; decision 5's prerequisites apply (an escape hatch for oversized
    constants — container-v1 §10 — and an assessment of doubled block count and descriptor size).

## Addendum 5 (2026-09-24) — points settled by the step 3e measurements

The part-length review of step 3 and the three RAM-peak candidates carried over from step 2. The
numbers are canonical in the [research record](../research/2026-09-24-part-length-ram-peak.md)
(3 series × part length {256, 512, 1024} MiB × cold / warm / local × median of 3).

1. **The default part length stays 256 MiB** (the answer to decision 6). After items 2 / 3, seek-type
   peaks (local positional reads, browser Blob) no longer depend on part length; scan-type peaks
   (Deno via HF) roughly grow with it because the hub's holding slot keeps a part (cold anima run
   1,139 / 1,741 / 2,245 MiB). A larger default barely reduces the part count (gemma4 E2B's `model`
   container: 83 / 79 / 78 parts, since range-read assets have dedicated parts).
2. **Candidate 3 adopted — the hub's scan-type slices are views of the holding slot's buffer**
   (`slice()` → `subarray()`). The double copy of a part is gone. It relies on alignment: the buffer
   is tight (byteOffset 0) and block starts are 64 B aligned (decision 7), so the 4 B alignment a
   `Float32Array` scale view needs holds. Callers that keep a range longer than the read copy it (a
   MUST of `AssetReader.read`).
3. **Candidate 2(c) adopted — Session construction reads, uploads and drops one block at a time**
   (lazy `WeightBatch.items`; the fence stays once per part). The basis is the WebGPU spec of
   `writeBuffer` (copied at the call — the addendum in decision 9, ADR 0070 decision 3 addendum);
   `packages/runtime/tests/gpu_write_buffer_copy_test.ts` was green in the `test:core` lane on
   Arc B570 / Deno 2.9.6.
4. **Candidate 1 (lending a buffer to the fetch layer's `readFile`) is not adopted.** Alone it saves
   no more than candidate 3; with 3, the buffer view would leave the hub and turn the lifetime
   contract of the public `BlockSource` into a loan (a breaking change). The fetch layer also refuses
   concurrent use of one buffer, serializing PLE's 16 parallel row reads, and a largest-part buffer
   would stay resident (about +224 MiB during gemma4 host-PLE decode — **speculation**).
5. History: acceptance ③ — seek-type peaks were 71–139 MiB independent of part length, within the
   estimate of one item (block + scale + f32 expansion; formula in container-v1 §11); scan-type peaks
   are "one slot + buffers awaiting GC" and are not bounded by declarations (up to about 4.4 largest
   parts on anima). Warm stayed at zero payload digests on all 27 configurations.
6. History: the Range trigger (addendum 2, item 2) still formally holds for scan-type cold reads
   (part 256: irodori 582, gemma4 1,038, anima 1,139 MiB), now caused by whole-part reads; whether to
   bring Range forward was left undecided.

## Addendum 6 (2026-09-25) — alternative 4's "ALU-bound" was a misquotation

Withdrawn: the reason "worse given that decode is ALU-bound" in alternative 4. Decision 15 says only
"not bandwidth-bound" (the source, [research 2026-09-19](../research/2026-09-19-qat-speed-recon.md),
found decode neither bandwidth- nor compute-bound); the rejection stands on its other reasons.

## Addendum 7 (2026-09-25) — source and recomputation of decision 15's bandwidth utilisation

Source: [research 2026-09-19](../research/2026-09-19-qat-speed-recon.md) §4 (RTX 3080 Ti, 912 GB/s,
Chrome pass timestamps). Recomputed after ADR 0105: QAT ≈ 13.6 %, regular ≈ 27.8 % — still not
bandwidth-bound on that machine; Arc B570 and Apple M2 were not measured.

## Addendum 8 (2026-09-25) — the real name of `fromContainer`

`fromContainer(bytes)` in the draft is the `{ kind: "bytes" }` input of `openContainer`
(`ContainerInput` in `packages/runtime/src/format/container/open.ts`); the decisions above use the
real name.

## Addendum 9 (2026-09-25) — decision 16's absmean idempotency argument is withdrawn; scale-ledger uniqueness instead

Decision 16's "ternary absmean gets a rewritten idempotency argument" is withdrawn: no such argument
exists. absmean (`γ = mean|W|`, `W̃ = RoundClip(W/γ, −1, 1)·γ`) has no fixed point. One rounding
maps elements to `{−γ, 0, +γ}`. Applying it again gives `γ' = p·γ` (`p` = share of non-zero
elements). If `p < 1`, the non-zero elements map to `±1/p`, clip to ±1 and move to `±p·γ`. The only
fixed point is `p = 1` (no zeros).

The following are preconditions of step 6 (absmean implementation):

1. **The scale ledger is fixed together with the rounding and never recomputed afterwards** MUST.
   Stored values are quantized with the fixed scale passed in (the same hand-off as i8's
   `quantize_to_int8(weight, scale)` — `tools/exporter/src/karume/quantize.py`).
2. **Paths that rely on idempotency are not usable with absmean.** Rounding the same weight twice
   (double application to tied weights — the form that the `fake_quant_int8` docstring makes i8 / i4
   mutually exclusive, MUST) is detected and fails loudly. Re-deriving the scale on re-export, and
   checks that rely on "re-deriving the scale gives the same values", are not used.
3. The discipline is renamed from "idempotency" to "**scale-ledger uniqueness**" (one scale ledger
   per weight, never re-derived).

Moving to an idempotent variant (for example a `max|W|`-based scale) is not adopted: even with a
ternary range it is a different quantizer from BitNet-style absmean, with different quality. No
current code is affected (absmean is not implemented — step 6).

## Addendum 10 (2026-09-25) — correction of decision 7 / Consequences: two reader-side tail-padding branches remain

Decision 7 said that writer-baked block lengths retire the reader's three tail zero-padding
branches, and the Consequences listed the same. The writer does bake 0x00 at block tails as
decided, but **two branches remain** on the reader side (code unchanged — the maintainer's ruling of
2026-09-25):

- f16: an odd-length payload is padded with 2 trailing zero bytes to 4-byte alignment
  (`alignF16Payload`, `packages/runtime/src/format/f16.ts`).
- i8 / i2: a payload whose element count is not a multiple of 4 is zero-padded to 4-byte alignment
  (`alignI8Payload`, `packages/runtime/src/format/i8.ts`).

Both are called from `session-build.ts`. They remain because of the supply shape: `containerBatches`
(`packages/runtime/src/runtime/executor.ts`) cuts `readBlock(...)` to the block's payload length
before passing it on, so the writer's padding never reaches the reader. In-memory container blocks
(`openMemoryContainer`) carry no padding at all (`format/container/memory.ts` — the synthesized
index length = payload length). One shape serves both sources, so the reader re-pads.

The cost is one copy of the payload into a new buffer per odd-length f16 initializer and per i8 / i2
initializer whose element count is not a multiple of 4 — mostly small tensors such as biases.
Passing padded lengths from the supply side to remove the two branches would loosen Session
construction's length check (equality with `seat.payloadBytes`) into a range check, so it is not
adopted here.

## Addendum 11 (2026-10-10) — duplicate JSON keys are accepted, last one wins

In the graph descriptor, the model descriptor and the IR JSON, duplicate keys within one object are
**accepted, last one wins**. Readers follow the defaults of `JSON.parse` (TS) and `json` (Python):
only the later value is seen, the earlier one never reaches the checks. Not writing duplicates is the
writer's responsibility; canonical serialization cannot write them. This matches the manifest (ADR
0038 — accepted last-wins, detection is the exporter's job) and keeps the interpretation equal to
validators in other languages. Rejecting would need a tokenizer in front of `JSON.parse`, which is
not adopted here. Spec: [container-v1](../container-v1.md) §0 and [ir-v2](../ir-v2.md), "container"
section.

## Addendum 12 (2026-10-11) — decision 12 is partly withdrawn; codecs have three owners

The part of decision 12 that puts storage format, CPU decoding and "executable compressed on the
GPU" together in one ledger is withdrawn (the maintainer's ruling of 2026-10-04). This changes the
spec to match the implementation; the code does not change.

Withdrawn:

1. `decodeCpu` / `executableOps` / `wgsl?` as ledger entry fields. `levels` and `name` are not
   ledger fields either (the registered name is the key of the ledger map; value set, pack order and
   reconstruction formula belong to the codec spec, implemented by CPU decoding and WGSL).
2. Folding the compressed-residency predicates (`eligibleCompressedInitializers` /
   `i2EligibleInitializers` / `i4EligibleInitializers`) into one `executableOps` field.

The rest of decision 12 stays: the ledger is immutable repository data, nothing registers at import
time, an unknown codec is rejected before weights are fetched, and the three axes (① readable as a
declaration, ② executable, ③ which ops keep it compressed and resident) are reported separately.

Owners of each axis:

| Holds                                | Canonical source (implementation)                                                                                                                       | Content                                                                                                                                                                                                                               |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Storage format (axis ①)              | `CODEC_LEDGER` in `packages/runtime/src/format/container/codecs.ts` (Python mirror: `CODEC_LEDGER` in `tools/exporter/src/karume/container.py`)          | Registered name, `layout` (kind of expansion path), `packing`, `scale`, `grouping`, `zeroPoint`, `rowAxes`. Declaration rules in `descriptor.ts` (unknown-codec rejection, field presence), `memory.ts` (field presence), `bind.ts` (ranges, divisibility, scale length, `rowAxes`) |
| CPU decoding                         | `decodeF16` (`packages/runtime/src/format/f16.ts`), `decodeBf16` (`bf16.ts`, since ADR 0131), `decodeI8` (`i8.ts`), `decodeI4` (`i4.ts`), `decodeI2` (`i2.ts`) | One per expansion path. Called from `packages/runtime/src/runtime/session-build.ts` (weights that axis ③ rules ineligible are expanded to f32)                                                                                        |
| Executable (axis ②)                  | `RUNTIME_SUPPORT.storage` in `packages/runtime/src/ops/contracts.ts` and `assertRuntimeSupport` in `packages/runtime/src/ops/support.ts`                | The set of executable expansion paths. The original row said "`bf16` is in the ledger but not here (declaration accepted, execution rejected as a missing capability before weights are fetched)" — **superseded by ADR 0131 on 2026-10-17**: `bf16` is in the set |
| Compressed-resident ops (axis ③)     | the three predicates in `packages/runtime/src/runtime/plan.ts` and `planWeightResidency` in `packages/runtime/src/runtime/weight-residency.ts`           | Eligible ops per expansion path; ineligible weights go to CPU decoding                                                                                                                                                                |
| GPU decoding (WGSL)                  | `packages/runtime/src/kernels/weight-storage.ts` (generators shared by the five fused kernels) and dedicated kernels (`linear-gemv.ts`, `linear-i8a8.ts`, …) | Per storage form: reading and scale multiplication                                                                                                                                                                                    |

- **Dependency direction MUST**: the format layer (ledger, declaration rules, CPU decoding) knows
  nothing about execution. `format` does not import `ops/` / `runtime/` / `kernels/`; the execution
  side reads the ledger's `layout` and branches.
- **The canonical source of axis ② is `RUNTIME_SUPPORT.storage`.** Putting an execution field back
  into the ledger and deriving ② from it is not adopted.
- **Reason**: with `executableOps` and `wgsl` in one entry as decision 12 had it, the lower format
  layer would hold execution-side judgements ("which ops can execute it", "is there WGSL"), reversing
  the dependency. Declaring the storage format and the backend's execution capability separately is
  common practice (for example, ONNX Runtime execution providers declare the nodes they take via
  `GetCapability()`). Decision 12's motive, "stop predicates growing per codec", is given up; the
  predicates stay unfolded as an accepted cost.
- **Places to touch when adding a codec** (as implemented): the ledger entry (TS and Python), one CPU
  decoder, `RUNTIME_SUPPORT.storage`, the residency predicates and the per-path branch of
  `planWeightResidency` (including `i4Eligible` / `i2Eligible` for shared initializers), and the WGSL
  variants. A codec that shares an existing expansion path (`ternary` → `i2`) needs only the ledger
  entry on the runtime side (decision 13 — the exporter's quantizer is separate).
- The spec was corrected at the same time: [container-v1](../container-v1.md) §6.2 / §6.3 / §13.5,
  [ir-v2](../ir-v2.md) "values and types", [glossary](../glossary.md) "codec ledger".

## Addendum 13 (2026-10-27) — the error scopes are awaited at the part fence (perf-ledger H-38)

**What changes.** Session construction still pushes and pops one error-scope pair per block (item)
inside the block's synchronous section, so a failure is still caught by the scope of the block that
caused it. What moves is the **await**: the pop pair's promise is kept and settled at the part
fence, in item order, instead of being awaited before the next block is read
(`packages/runtime/src/runtime/session-build.ts` — `settleScopes` / `firstScopeFailure` and the
part loop). The fence itself is unchanged (one empty submit + `onSubmittedWorkDone` per part). The
wait for the scopes at the end of a part is counted in `SessionBuildStats.uploadFenceMs` together
with the fence (both are waits for the GPU).

**Early stop.** The part loop does not upload further blocks once a failure is known: as soon as
one block's scopes settle as a failure, the loop stops before uploading the next block and the
failure is reported (in item order — invariant 1). This bounds the extra work after an OOM (the
allocations and writes of the rest of the part) without bringing back the per-block round trip;
the success path never takes it.

**Why decision 9's "no wall-clock penalty" does not carry over.** Decision 9 measured the cost of
issuing a push/pop pair (1.81 µs) on Deno / wgpu, where the pop's promise settles almost at once. It
did not measure what the per-block **await** costs when the next block's read waits behind it. On
Dawn in Chromium the pop's result comes back from the GPU process, so each block waits one round
trip before its successor's read even starts: supply and upload run strictly in series, and
`SessionBuildStats` sees none of the wait (it falls between the counters). Measured:

| Environment | Per Wan2.2 request (the builds each row covers) | Source |
| --- | --- | --- |
| Chromium-Linux, RTX 3080 Ti | −3.0 to −3.2 s over all three builds (umT5 −1.2 to −1.3 s, DiT −1.5 to −1.6 s, the two VAE builds −0.24 to −0.30 s; three A/B runs) | [research 2026-10-25 (3080 Ti)](../research/2026-10-25-speed-recon-3080ti.md) §9.2, HB-1 (the total); `outputs/diag/perf-recon-2026-10-24/ledger-host/ROUND1-digest.txt`, HB-1 (the split per build; not tracked) |
| Chrome 155 / D3D12, RTX 5070 Ti | −1.52 s over the DiT + umT5 builds (DiT −0.79 s, umT5 −0.73 s) | [research 2026-10-25 (5070 Ti)](../research/2026-10-25-speed-recon-5070ti.md) §10 |
| Node + Dawn (Vulkan), RTX 3080 Ti, local mirror | Over the DiT + umT5 builds: −0.37 s with per-block sha256 (the Node local-directory path: DiT −0.06 s, within the noise — 5 of 8 pairs faster; umT5 −0.31 s, 7 of 8); −0.74 s without it (the browser's shape: DiT −0.17 s, 7 of 8; umT5 −0.57 s, 8 of 8) | `outputs/diag/session-build-2026-10-27/` (not tracked) |

The Node row is the **median of the paired deltas** (8 pairs per component: per-block await and
part-fence await interleaved in one process on one device, the leading arm alternating per pair),
summed over the two components. Single pairs range from −2.1 to +1.5 s per component and from
−2.5 to +1.4 s per request (the two components' deltas of one pair summed), both in the sha256 run;
the run without sha256 stays inside both ranges. Its harness reads through its own positional-read
source with one persistent file handle per part (`measure.ts`), not
through the hub's `nodeDirectory`, so its absolute build times are not comparable with addendum
14's Node row.

The bytes on the GPU do not change (bit-identical): the same buffers receive the same
`writeBuffer` calls in the same order; only the point where the build waits for the scopes moves.

**Invariant (MUST).** The per-block attribution decision 9 bought stays exactly as it was:

1. When one or more blocks of a part fail, the build fails with the error of the **first failing
   block in item order** — the same error type (`GpuOutOfMemoryError` / `GpuValidationError`, or a
   rejected pop as is) and the same message (`<part> weight upload for initializer '<name>': …`)
   the per-block await reported. A later block that settles first does not win.
2. A scope failure of an earlier block also wins over a host-side failure or a block-read failure
   of a later block (the per-block await stopped before reaching that later block).
3. No failure is dropped: every block's scopes are settled before the part's fence and before the
   part counts as uploaded, so a failure that settles after the last block still fails the build.
   Failures after the first are discarded on purpose — the per-block await never reached them.
4. No failure is reported twice and none escapes as an unhandled rejection: each pop's promise gets
   its handlers when it is issued.
5. Device loss and validation keep their classification: a block's scopes catch what they caught
   before, and the fence still races `onSubmittedWorkDone` against the loss. This rests on the
   WebGPU spec's ordering — which scope an error lands in is fixed when the call is issued, and a
   pop resolves on the device timeline before the work issued after it — so awaiting the pop later
   moves no error to another scope. It was not observed on Dawn: no test raises a real GPU error
   there, and the fake device of the tests reproduces this ordering rather than checking it. The
   early stop changes only how soon the loop ends, not which error is reported.

The transaction boundary of decision 9 is unchanged: a failed part destroys the weights arena and
publishes no partial Session. Pinned by `packages/runtime/tests/runtime_weight_upload_scope_test.ts`
(a fake device injects per-block scope failures, settle delays and read delays: the order of the
reported failure, the deferral itself — every block's pops are issued before the first block's
scopes settle — and the early stop); the order of push → write → pop per block and of the fence
per part stays pinned by `runtime_weight_supply_test.ts`.

## Addendum 14 (2026-10-27) — the supply reads 2 blocks ahead, across part boundaries (perf-ledger L-26)

**What changes.** Session construction no longer reads a block only when its item is pulled. The
container supply (`containerBatches` in `packages/runtime/src/runtime/executor.ts`) walks one fixed
read order over the whole supply — parts ascending, then each item's payload block and, on piece 1
or a whole initializer, its scale block — and keeps up to **2 reads in flight beyond the last read
of the item being uploaded** (`READ_AHEAD_BLOCKS`). The order is global, so the read-ahead crosses
part boundaries: the next part's first blocks are read while the current part's error scopes and
fence are awaited (addendum 13). Uploads are unchanged: one item at a time, the same `writeBuffer`
calls with the same bytes in the same order, one fence per part (bit-identical).

**Why across parts.** A read-ahead confined to a part has nothing in flight while the part's fence
is awaited, which is where most of the remaining supply wait sits once the scopes are awaited at the
fence. Measured beyond H-38:

| Environment | Per Wan2.2 request (DiT + umT5 builds) | Source |
| --- | --- | --- |
| Chromium-Linux, RTX 3080 Ti | −1.0 s across parts; −0.6 s if confined to a part | [research 2026-10-25 (3080 Ti)](../research/2026-10-25-speed-recon-3080ti.md) §9.2, SP-1 |
| Chrome 155 / D3D12, RTX 5070 Ti | −0.59 s across parts; −0.50 s confined to a part | [research 2026-10-25 (5070 Ti)](../research/2026-10-25-speed-recon-5070ti.md) §10 |
| Node + Dawn (Vulkan), RTX 3080 Ti, local mirror through `nodeDirectory` | −3.09 s with per-block sha256 (the Node local-directory path: DiT −1.24 s, umT5 −1.85 s); −1.61 s without it (the browser's shape: DiT −0.55 s, umT5 −1.07 s); every pair faster (8 of 8 per component) | `outputs/diag/session-build-2026-10-27/` (not tracked) |

The Node row uses the statistic of addendum 13's Node row — the **median of the paired deltas** (8
pairs per component: depth 0, the read on pull, and depth 2 interleaved in one process on one
device, the leading arm alternating per pair; H-38 in both arms), summed over the two components.
Its harness reads through the hub's `nodeDirectory`, which opens one file handle per range read,
while addendum 13's harness keeps one handle per part, so the absolute build times of the two rows
are not comparable.

On the sha256 path the read-ahead also overlaps the per-block digests (the same digests as before —
one per block, no extra digest work), which is why the gain there is larger.

**Invariants (MUST).**

1. **Order and bytes.** Items, their bytes and their order are the same as with one read per pull;
   the cursor fails loudly if the takes ever drift from the planned read order.
2. **Attribution.** A read that fails ahead of its turn is held (a handler is attached when it is
   issued, so it is never an unhandled rejection) and surfaces only when its block is taken. A read
   failure is therefore reported for the block it belongs to and only after every earlier block —
   an earlier block's read failure, host-side failure or (addendum 13) scope failure still wins.
3. **Bound.** At most the item being uploaded (its block, plus its scale block on piece 1 or a whole
   initializer) plus 2 blocks are alive on the JS side — 3 or 4 blocks. For a `krm` (blocks within
   the 32 MiB limit of decision 5) on a seek-type source that is at most 128 MiB, whatever the part
   length. The in-memory container does not check its pieces' lengths against the block limit, so
   there the bound is 3 or 4 of the caller's pieces (and scales), whatever their length. On a
   scan-type source each block is a view of its whole part, so near a part boundary up to 3 parts
   can be alive (when parts hold one block each). A source sees up to 3 concurrent reads, and what
   they hold depends on the source — two separate limits: on a local directory each read opens its
   own file handle, so up to 3 file handles are open at once; on the hub's network path a read that
   misses the cache downloads its part, and those network fetches stay within the hub's own cap of
   4 concurrent fetches (decision 6), which the read-ahead depth does not change.
   `BlockSource.read` and `MemoryPiece.read` MUST take concurrent calls and MUST NOT overwrite bytes
   they returned earlier (stated on both types).
   **Amended by addendum 15 (2026-11-03):** for an initializer whose scale is split, the riding
   scale is the joined scale (the whole scale, up to 120 MiB for Gemma 4 12B's embedding), not one
   block.
4. **Stop.** When the build fails or the consumer stops early, the supply's iterator is returned:
   no read is issued after that, and the build rejects only once the reads in flight have settled
   (`readBlock` has no abort, so they are waited for, not cancelled; their bytes and errors are
   dropped). The cost: the time to reject is bounded by the slowest read in flight, which is
   unbounded when a fetch stalls — `BlockSource.read` takes no signal, and on the hub's scan path
   `openFile` may download the part on a cache miss. The success path already had the same
   exposure (a stalled read stalls the build), so no abort is added here; that would be an API
   change of `BlockSource`.

**Residual (an analysis, not observed).** The hub's scan path keeps one holding slot and decides it
after an asynchronous `openFile` per part (`packages/hub/src/container.ts`). If the plans of two
parts the read-ahead reads at once settle out of order, a part of at most 2 blocks can take the slot
back from the next part, whose later blocks then read it in full again — correct bytes, but one
extra part read, and the next part is briefly held twice (the evicted buffer stays alive while the
views already cut from it are being uploaded), so host RAM rises by one part for that moment. The
affected route is HF on Deno / Node, where `openFile` returns a scan-type source. Wan's parts hold
many blocks; ordering the slot decisions by call order would close it on the hub side.

Pinned by `packages/runtime/tests/runtime_weight_supply_test.ts` (crossing a part boundary, the bound
of 2 blocks and 3 concurrent reads — also with a riding scale, where 4 blocks are alive — failure
order and attribution, a scope failure that wins over a slow read failure of a later part, stopping
on failure and on an early return, no pull after close); the per-block scope attribution of
addendum 13 stays pinned by `runtime_weight_upload_scope_test.ts`, whose fake device attributes
calls by the scope pushes, because with the read-ahead a read no longer identifies the item being
uploaded.

## Addendum 15 (2026-11-03) — a quantized binding's scale may be split on the data's pieces

**What broke.** Decision 5 caps a block at 32 MiB, and rule ③ of [container-v1](../container-v1.md)
§5 put a quantized binding's companion scale in exactly one block, so a scale could not exceed
32 MiB. An `int4-sym-g` scale holds one f32 per `groupSize` elements; with `groupSize` 32 the scale
alone passes 32 MiB beyond 268,435,456 elements (8,388,608 scales × 32). The Gemma 4 embeddings
imported as i4 g32 from the QAT grids need 48 MiB (E2B, `[262144, 1536]`), 80 MiB (E4B,
`[262144, 2560]`) and 120 MiB (12B, `[262144, 3840]`) of scale. The writer planned the scale as one
block without checking its length (const and asset blocks were checked, scale blocks were not), so
the export ran to the end and failed only at the read-back
(`modelDescriptor.blocks[1]: length 125829120 exceeds the block limit 33554432`).

**Decision.**

1. **Encoding.** `encoding.scale` takes `pieces` in place of `block`: the element of the supply
   form's `pieces` (§5), `{ "block": <id>, "rows": [begin, end) }`, 2 or more, exclusive with
   `block`:

   ```jsonc
   "scale": { "pieces": [{ "block": "s0017", "rows": [0, 17476] }, /* … */], "dtype": "f32" }
   ```

   A scale that fits one block MUST be written with `block`: the writer splits a scale only when
   it exceeds the block limit, the same rule that splits the data.
2. **Alignment (MUST).** A split scale goes with a data piece sequence only, and scale piece k
   covers exactly the rows of data piece k along `rowAxis` (0 for every piece sequence — rule ④).
   Its length is `rows × (rowLength / groupSize) × 4` bytes — for `int4-sym-g` g32,
   `rows × rowLength / 8`, a quarter of the data piece. **Every scale piece is in the part of data
   piece 1** (rule ③ extended — Session construction reads the whole scale with piece 1).
3. **Reader rules** (TS: `parseEncoding` / `validateModelDescriptor` in `descriptor.ts`,
   `planSplitScale` in `bind.ts`; Python mirror: `_parse_encoding` / `_validate_model_descriptor` in
   `karume/container.py`, `_plan_split_scale` in `karume/verify.py`):
   - structure, when parsing: 2 or more entries, exclusive with `block` (neither is rejected too),
     each row range non-empty, continuing from row 0 (a gap or an overlap is rejected — the same
     parser as the data's `pieces`);
   - placement, in the descriptor check: each piece's block has role `"scale"`, is bound once
     (rule ①), sits in the part of data piece 1 (rule ③) and is within the block limit (§4.1 —
     the rule every block passes);
   - alignment, in the merge layer: the data has a piece sequence, the piece counts are equal, the
     rows are equal piece by piece, and each piece's length is its rows' bytes (a middle piece
     exactly, the last one with less than 4 of padding). The total then equals the scale's payload
     length, so it has no check of its own;
   - a const cannot have a split scale (a const has no pieces — §3).
4. **Writer** (`_cut_scale` / `_plan_data_parts` / `_build_const_region` in `karume/container.py`).
   It fails loudly **while planning the layout, before any byte is written**, when no split fits:
   the data is one block (no piece boundaries) or a scale piece would still exceed the limit (both
   only for `int8-sym` rows shorter than 4 elements, the one ledger shape whose scale is larger than
   its data), a const's scale exceeds the limit, or piece 1 and the whole scale do not fit one part
   (the existing part check). This
   also closes the missing up-front check for scale blocks. `zeroPoint` is not extended: no ledger
   codec allows one; a codec that does would take the same `pieces` spelling.
5. **Hand-off to Session construction: one contiguous scale.** The merge layer presents a split
   scale as one scale (`ScaleSupply` in `bind.ts`, `ScaleSupply` in `karume/verify.py`): `id` is the
   **joined id** (the pieces' block ids joined with `+`, which is outside the block-id vocabulary of
   §0 and so never names a declared block), `payloadBytes` is the whole scale's, and `pieces` lists
   the declared blocks. `readBlock` of the joined id (`open.ts`; Python `ReadContainer.block`)
   reads the pieces one at a time, each verified like a block, into one new buffer. The
   construction code (`containerBatches` in `executor.ts`, `session-build.ts`) and the kernels are
   unchanged: they receive the scale with piece 1, byte for byte what an unsplit scale gives.

**Why this encoding is the minimal one.**

- **One new spelling, and it is an existing one.** The only addition is `pieces` inside
  `encoding.scale`, and its element is the supply form's piece element. No new element shape, no
  new key anywhere else, no version change.
- **No second spelling of the unsplit case.** A scale that fits MUST stay `{ block, dtype }`, so a
  container that needs no split is byte-identical to what the writer wrote before this addendum
  (checked against the committed cross-language fixtures and against a sha256 of the previous
  writer's output — `test_a_container_that_needs_no_split_is_the_bytes_the_writer_wrote_before` in
  `tools/exporter/tests/test_container.py`), and readers read it exactly as before. A reader that
  predates this addendum rejects a container with a split scale (missing `block`, unknown key
  `pieces` — §0), so nothing is misread silently.
- **The rows are written although they repeat the data's.** The alternatives: (a) a bare list of
  block ids with the rows implied by position — a new element shape found nowhere else, which a
  reader can check against nothing but its count; (b) a `scale` field in each element of the data's
  `pieces` — it moves the scale out of `encoding`, which the const region and the in-memory
  container share, and changes the piece element of every piece sequence. The repeated rows are a
  copy the reader checks, like `packing` (§6.1), not an independently updated field: a writer that
  cut the scale elsewhere fails loudly instead of drifting.
- **One scale piece per data piece**, not a cut of its own at the limit: no new row boundaries, the
  CPU decode of data piece k reads exactly the rows of scale piece k, and a later per-piece upload
  can pair them without slicing.
- **All in the part of piece 1**, not each next to its data piece: construction reads the whole
  scale with piece 1, so pieces in later parts would make that read pull later parts early on the
  scan path (the hub's one holding slot — addendum 14, "Residual"); and laying scale pieces between
  data pieces would make the writer fetch the converted tensor again for every block (`_Payloads`
  keeps one tensor only while the same key repeats).

**Why not bind per piece.** Uploading scale piece k with data piece k would keep the riding scale
inside the block limit, but it changes Session construction: `containerBatches` rides the scale only
on piece 1, and `session-build.ts` writes it to the GPU whole (compressed residency) or copies it
whole for the pieces that follow (CPU decoding) at piece 1. The joined read needs no change there and
none in the kernels.

**Consequence: the bound of addendum 14 item 3 and container-v1 §11.** For an initializer with a
split scale the riding scale is the joined scale, not one block. The item at piece 1 holds piece 1 +
the whole scale, the join briefly holds one scale piece on top, and a read ahead can be a joined
scale; under CPU decoding the carried-over copy is the whole scale as before, now not bounded by the
block limit either. The writer's plan for the Gemma 4 embeddings (`_cut_rows` / `_cut_scale` at the
32 MiB limit):

| Model | Embedding | Data pieces (rows each, last) | Scale | Scale pieces (bytes each, last) | Piece 1 + scale in part 2 |
| --- | --- | --- | ---: | --- | ---: |
| E2B | `[262144, 1536]` | 7 (43,690, last 4) | 48 MiB | 7 (8,388,480 B, last 768 B) | 80.0 MiB |
| E4B | `[262144, 2560]` | 11 (26,214, last 4) | 80 MiB | 11 (8,388,480 B, last 1,280 B) | 112.0 MiB |
| 12B | `[262144, 3840]` | 16 (17,476, last 4) | 120 MiB | 16 (8,388,480 B, last 1,920 B) | 152.0 MiB |

So the seek-type bound of "at most 128 MiB for a `krm`" becomes three blocks + the largest joined
scale (about 216 MiB for 12B); on the scan path the joined scale is a copy on top of the holding
slot (the pieces are views of the part, the join is not). Writing a split
scale's pieces to the GPU one at a time at piece 1 (they are all in that part) would restore the
block bound; it is a construction change and is not part of this addendum.

Pinned by `tools/exporter/tests/test_container.py` (`TestASplitScale`,
`TestTheSplitScaleFailsLoudlyAtPlanning`, `TestTheReaderRejectsABrokenSplitScale`, the split-scale
fixture), `tools/exporter/tests/test_verify_container.py` (`TestASplitScale` — the merge-layer
rules on hand-built descriptors), `packages/runtime/tests/container_split_scale_test.ts` (writer →
reader, `containerBatches`, every reader rejection), `container_fixture_test.ts` (the
Python-written `synthetic-split-scale.krm` read by the TS reader, canonical serialization) and
`gpu_split_scale_test.ts` (the same output bytes as a one-block scale under compressed residency and
under CPU decoding). Spec: [container-v1](../container-v1.md) §3, §5 rules ① / ③, §6.1, §11, §13.3.
