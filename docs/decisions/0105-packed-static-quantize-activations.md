# 0105: Pass fixed-SRQ activations to the parallel GEMV as packed int8

- Status: accepted (2026-09-19 — stage 1a of K-45, in the scope the maintainer ruled "a first");
  promoted to the manifest vocabulary with the `i4-fast` declaration by addendum 2 (2026-09-20).
  Partly superseded: the "no manifest field" clause of decision 1 by addendum decision 3 (2026-09-20);
  the manifest format of addendum decision 3 by [0109](0109-manifest-v5-container.md) (2026-09-22);
  "E4B stays `i4`" of addendum decision 4 by the [0104](0104-gemma-fast-quant.md) addendum of
  2026-09-26. The M2 re-test procedure on `tools/llm-speed` was withdrawn on 2026-10-18.
- Related: [0097](0097-gemma4-qat-integration.md), [0098](0098-linear-gemv-parallel.md),
  [0103](0103-linear-static-quantize-fusion.md), [0104](0104-gemma-fast-quant.md),
  [0040](0040-fusion-pass.md), [0058](0058-numerics-opt-in-contract.md),
  [0022](0022-gemm-register-blocking.md)
- Evidence: [attribution of QAT decode speed](../research/2026-09-19-qat-speed-recon.md) §14 (the
  stage 0 isolation)

## Context

QAT E2B decode spends its time on **the number of activation loads**, not on arithmetic. The stage 0
single-kernel A/B (research §14) measured that cutting the activation loads per weight word from 16 to 1
makes the down shape 6× and the i4 g512 shape 2.3× faster, while cutting the ALU per word by 83% moves
the down shape by only 0.3%. The parallel GEMV reads the activations as `vec4<f32>`: 16 / 8 / 4 loads
per weight word (i2 = 64 elements / i4 = 32 / i8 = 16).

The activations come from the fixed SRQ (`static_quantize`), whose values take only the 256 int8 levels.
Passing them as f32 makes the number of words read four times larger than needed.

## Decision

### 1. The option

`SessionOptions.packedStaticQuantize?: boolean` (default false). true is accepted only together with
`linearGemvReduce: "parallel"` and `linearCompute: "f32"`; an unsupported combination and a non-boolean
value are rejected at Session construction (the same approach as ADR 0103: no silent fallback to the
f32 path). The common Gemma pipeline option of `@karume/models` passes the same value to the target and
drafter Sessions.

The manifest-owned vocabulary (the hub's `SessionSpec`) gets no field for it; until adoption it enters
**only by the caller's explicit option**. (Superseded by addendum decision 3 on 2026-09-20: the field
was added to `SessionSpec`.)

### 2. Pairing (decided at plan time; the IR does not change)

When the option is on, a `static_quantize` node is passed packed only if all of the following hold:

1. It **remains a plain node** after fusion. An SRQ absorbed by the linear→SRQ fusion (ADR 0103) is an
   output-side epilogue, and its packed output is outside this stage.
2. Its output is not a graph output, input and output are f32, the last dimension k is a multiple of
   16, and the scale is positive and finite.
3. It has at least one consumer, and **every** consumer is the **activation slot** (`ins[0]`) of a
   linear that falls into the parallel GEMV (`linear_gemv_parallel:*`, with or without the SRQ fusion
   epilogue). One GEMM, sequential GEMV, subgroup variant or other op among them keeps it f32.
4. No fusion step other than the linear→SRQ fusion consumes the value. Only that fusion has a spelling
   that reads packed words, so the moment another rule takes `linear` into its window, its fused kernel
   would read packed words as f32. None of the current 7 rules has `linear` in its window, so this
   never fires today; without the check on the accepting side, **only the newly added rule** would
   break silently.

Whether a linear falls into the parallel GEMV is decided by one predicate, `linearGemvParallelEligible`
(storage, `gemmUsesVec4`, `k % step`, the i4 group length, the lane table of measured shapes); the
pairing (`fusion.ts`) and the recipe-builder check MUST read the same predicate (addendum decision 2
narrows it to `linearGemvPackedEligible`). Keeping two copies lets one be wider than the other, and f32 words
then flow into a `vec4<u32>` binding: a silent wrong value with no exception.

The decision needs to know whether the node remained plain, so `planFusions` **runs its scan twice**
when the option is on: the first scan provides the input to the pairing, the second is the result. The
packed flag enters no rule's match condition (only the key, WGSL and params of the dispatch built by the
linear→SRQ fusion change), so both scans see the same rule take the same window.

The fusion counter (ADR 0040) gains `packedStaticQuantize`. The QAT E2B decode plan had **210** (70
after addendum decision 2), the prefill plan (M ≥ 32, GEMM) 0, with no overlap with the 275 of the
linear→SRQ fusion (whose output-side consumers include non-linear ops). Fusion on / off does not move
this number.

### 3. Packed layout and restoration (the basis of bit identity)

One u32 word holds four int8 codes; the code of element i is byte `i & 3` of word `i >> 2` (the
little-endian order of `pack4xI8` / `unpack4xI8`). The code is **exactly the level / sign that the
current `quantize` determines**; the boundary table and the binary search are shared verbatim with the
f32 path. The output tensor is allocated with its declared shape (only a quarter of it is written), so
the arena buckets do not move.

The consumer restores **`vec4<f32>(unpack4xI8(word)) * x_scale`** (one f32 multiplication per element).
This is u32-identical, element by element, to the producer's output table (`params[129 + level]` of
`staticQuantizeParams` = `Math.fround(level * scale)`):

- `level` is 0..128 and exact in f32. `scale` is exact in f32 (checked on the params side).
- Their product has at most 31 mantissa bits, so it is **exact in f64**. The table side (f64 product
  rounded to f32) and the WGSL side (f32 multiplication = correctly rounded product) both yield "the
  exact product, correctly rounded to f32".
- The sign is the sign of the product, equal to the table lookup's `| sign` (level ≥ 1).

Therefore **the table (129 uniform words) is not passed to the consumer**. A lookup is bit-identical by
construction, but it would add a second 65×vec4 to the parallel GEMV's Dims and take more than one
instruction; when the proof reduces it to one multiplication, that is cheaper.

`x_scale` is one word added as the **last member of Dims** (word 3 for unfused packed, word 264 after the
existing 264 words for packed with the SRQ fusion epilogue). The params layout of existing keys does not
move by one word.

**Only the two values that have no int8 code behave differently from the f32 path** (accepted within the
numerics opt-in of ADR 0058):

- `-0.0` becomes code 0 = `+0.0`. The GEMV accumulator starts at `+0.0`, and adding `±0.0` does not move
  it (`+0.0 + (-0.0) = +0.0`; a non-zero accumulator is unchanged), so **the output does not move**. This
  is treated as a measured claim, not an argument: the check is a full u32 match on activations that
  contain `-0.0`.
- `NaN` falls outside the boundary table and **saturates** to ±127 / -128 (the f32 path passes NaN
  through). `±Inf` saturates on the f32 path too, with the same table, so they agree.

### 4. The packed-activation variant of GEMV

The parallel GEMV (i2 / i4 / i8 × lanes 2 to 32 × with / without the SRQ fusion epilogue) gains a variant
whose activation binding is `array<vec4<u32>>`. **The geometry, the lane table, the cyclic K
distribution, the workgroup addition tree, the bias addition order and the output-side epilogue do not
change by one byte.** Only two things change:

- the quad index at the start of a row becomes `dims.k / 16u` instead of `dims.k / 4u`;
- a word is read once per four quads, and each quad is restored to f32 by the formula above.

The activation loads per weight word drop to **i2 16 → 4, i4 8 → 2, i8 4 → 1**. The key is the existing
key with `:packed-x-i8` appended, so the position of the storage discriminator that diagnostics and the
census read does not move. The sequential GEMV, row-block GEMV, subgroup variants and GEMM do not change.

## Alternatives considered

- **Integer dot product (`dot4I8Packed`) from the start** (stage 1b). Stage 0 measured arithmetic as
  free (down shape 0.3% / i4 g512 shape 8%), so most of the gain comes from the activation loads alone.
  The integer dot product bets on the portability of `dot4I8Packed` (whether M2 Metal lowers it
  natively), and its i32 accumulation is **no longer bit-identical to the current path**. It pays off
  only for the lm_head shape (③, arithmetic 36%), so it was deferred to a separate stage limited to that
  shape (rejected by addendum 2).
- **Pass the output table to the consumer as well and look up `table[|code|] | sign`.** Bit-identical by
  construction, but decision 3 proves it equals one multiplication, so it only costs 1,040 uniform bytes
  and extra instructions.
- **Write the output-side SRQ fusion epilogue packed too.** Its consumers are not GEMVs (they expect
  f32), so it is outside this stage.
- **Bind the activations as `array<u32>` (four elements per word).** The bytes drop to a quarter but
  **the number of loads does not** (i2 stays at 16), so it misses the bottleneck stage 0 measured.

## Consequences

- Seven new keys: `static_quantize:v1:packed-i8:wg128` and the packed variants of the parallel GEMV ×
  {unfused, SRQ fusion}, with WGSL snapshots. **The output of existing keys does not change by one
  byte.**
- Prefill plans (M ≥ 32) do not fall into the parallel GEMV and are naturally out of scope.
- The IR, weights, manifest, published source pins and the reference path do not change. The default
  false keeps the former behaviour.
- With `linearGemvParallelEligible`, the acceptance of the linear→SRQ fusion (ADR 0103) goes through the
  same predicate. Its acceptance set does not change (every shape in `PARALLEL_SHAPES` meets the added
  conditions).

## Acceptance

Completed 2026-09-19 (CPU: option handling, pairing conditions, fusion counter, WGSL snapshots, load
counts; GPU: restoration u32-identical to the current SRQ for all 256 codes, 540 GEMV cases u32-identical,
QAT E2B 64-token greedy ids identical with the option on / off). This is not a specification guarantee of
floating-point rounding on every GPU (the same position as ADR 0103 — agreement on real hardware is an
acceptance result).

## Addendum 1 (2026-09-19) — adoption per shape and a redesigned SRQ kernel

Measured as decided, the option made QAT E2B decode GPU time **+0.25 ms/token worse** (Chrome). The
per-key measurement (`outputs/bench/karume/2026-09-19_21-25-39_k45-1a-packed-aa2db539/per-key-on-off.json`)
showed that packed pays off only for long-K shapes with 32 lanes, and that the packed SRQ kernel itself
was 19% slower (too few threads); details in research §16.

### Addendum decision 1: the packed SRQ kernel uses the geometry of the f32 path

`STATIC_QUANTIZE_PACKED_WGSL` returns to **one thread per element**. Each thread computes its element's
code and stores it in workgroup shared memory (`array<u32, 128>`); after `workgroupBarrier()`, the lower
32 threads pack four codes each into one word and write it. The dispatch count equals the f32 version's
`ceil(count / 128)`. No atomic OR is used (each word is written by one thread).

- **Grid-stride edge**: the tile (128 elements) is chosen from `workgroup_id` only. Even when `count` is
  not a multiple of 128, the barrier is passed uniformly by the whole workgroup (looping on
  `local_invocation_index` would make the barrier non-uniform on the edge tile and fail WGSL's
  uniformity analysis). An out-of-range element stores code 0 in shared memory and **writes no word** —
  `count` is a multiple of 4 (checked on the params), so a word is either entirely in range or entirely
  out of range.
- **Code determination** (boundary table, binary search, level / sign) does not change by one byte. Only
  the work assignment and the write path changed; the bit-identity argument of decision 3 stands.
- Each tile passes the barrier twice (the second is the write-after-read barrier before the next tile
  overwrites shared memory).

### Addendum decision 2: adoption per shape lives in the measured-shape table

Every row of `PARALLEL_SHAPES` (the "measured shapes only" approach of ADR 0098 / 0022) gains
`packedActivations: boolean`; only the four rows that measured faster are true.

| Row                    | lanes | per-key        | Adopted |
| ---------------------- | ----: | -------------- | ------- |
| i2 n=1536 k=12288      |    32 | `wi2:l32`      | true    |
| i4 g2048 n=1536 k=2048 |    32 | `wi4g2048:l32` | true    |
| i4 g2048 n=1536 k=6144 |    32 | `wi4g2048:l32` | true    |
| i4 g4096 n=1536 k=4096 |    32 | `wi4g4096:l32` | true    |
| every other E2B row    |     — | —              | false   |

The E4B rows added later are the exception: ADR 0098 addendum (2026-09-26) placed them unmeasured,
copying `packedActivations` from the E2B row of the same role (three QAT E4B rows are true —
`packages/runtime/src/kernels/linear-gemv.ts`).

The consumer condition of the pairing (decision 2) includes this flag: a `static_quantize` is packed only
when **every consumer falls into a true row**. The decision is one predicate,
`linearGemvPackedEligible` (`linearGemvParallelEligible` plus the row's flag), and the MUST of decision 2
that the pairing in `fusion.ts` and the recipe-builder check read the same predicate stands.

Rows with false can still generate the packed WGSL and params (tests and snapshots keep every shape),
but **production plans do not select them**.

The fusion counter `packedStaticQuantize` of the QAT E2B decode plan goes **210 → 70** (20 + 43 + 7, the
per-key counts); the prefill plan stays 0. Acceptance (2026-09-19): the new kernel's codes match the f32
path, the gating per row, decode 70 / prefill 0, and fault injection turning three GPU checks red.

## Addendum 2 (2026-09-20) — promotion to the manifest vocabulary, the `i4-fast` declaration and the rejection of stage 1b

The "no manifest field" clause of decision 1 is lifted. The A/B of research §16 (Chrome +8.3% / GPU
−0.57 ms/token, Deno +1.5 to 2.9%, identical 64-token greedy ids) decided adoption, so the option becomes
vocabulary that a distribution can declare.

### Addendum decision 3: the hub's `SessionSpec` gains `packedStaticQuantize?: boolean`

- Only true / false are accepted; null, numbers and strings are rejected. false is not folded into an
  omission, and the common mapping in `@karume/models` (`WRITERS`) forwards it explicitly to the same
  field (the approach of the two fusion fields of [0104](0104-gemma-fast-quant.md)). The manifest stays
  `karume/4` (superseded by ADR 0109 decision 2 on 2026-09-22: the manifest is `karume/5` and the field
  carries over unchanged). An old reader rejects it as an unknown key, so a distribution that declares
  it needs the matching hub / models.
- Gemma accepts it from `quant.session`. Precedence is **explicit option → quant declaration → unset
  (runtime default false)**; a correctly typed explicit false beats the quant's true. The rejection of
  decision 1 ("`linearGemvReduce: "parallel"` required") applies to quant-declared values too.

### Addendum decision 4: QAT E2B `i4-fast` declares it

- The recipe (`qat_quants` in `gemma4_qat/distribution.py`) adds `packedStaticQuantize: true` to E2B
  `i4-fast`. E4B stays `i4`. (Superseded by the ADR 0104 addendum on 2026-09-26: QAT E4B also has
  `i4-fast` as its default, with `packedStaticQuantize: true` —
  `tools/export-recipes/gemma4_qat/distribution.py`.)
- Regular Gemma 4 has no `static_quantize` nodes, so nothing pairs and it does not declare the field
  (declaring it would be a no-op, but a distribution does not carry a meaningless field).
- Default numerics: the "`-0.0` becomes code 0, NaN saturates" of decision 3 becomes the behaviour of the
  QAT default preset (within the numerics opt-in of ADR 0058). The reference golden, the reference path
  and `i4` / `i4-gemvpar` do not change.
- Withdrawn 2026-10-18: the M2 re-test through an off / on axis in the browser measurement page
  (`tools/llm-speed/browser/`) — the tool was removed by bb35fe2a; which measurement tools return as a
  browser app is an open point of ADR [0127](0127-node-dawn-toolchain.md).

### Stage 1b (integer dot product) is rejected

The integer dot product that "Alternatives considered" deferred to a stage limited to the lm_head shape
(③, arithmetic 36%) is not implemented. The rejection covers the QAT E2B lm_head only; it does not forbid
integer dot products in the parallel GEMV in general.

- "③ benefits from arithmetic" (research §14) was a reading of the single-kernel harness; the real
  model's lm_head has **no int8 activation**. The official checkpoint's lm_head has SRQ scale 0 on input
  and output (uncalibrated = identity), and the recipe inserts no identity SRQ into the IR (the shared
  head check in `packages/models/src/gemma/qat.ts`, ADR 0097). Creating the int8 activation would add a
  dynamic quantization that the official model does not have either, a numerics contract bit-identical
  to neither the official mobile path nor the current one.
- The ceiling is 36% arithmetic of the lm_head dispatch's 0.24 ms/token (research §13.4) ≈ 0.09 ms/token
  (1.3% of 6.63 ms GPU). It does not fit K-45's goal ("the same computation form as the official mobile
  path") and the gain is small, so it is rejected by the same judgement as perf-ledger K-52.

## Addendum 3 (2026-09-20) — Metal divergence and a rounding barrier on the restored activation

On the maintainer's M2 (Chrome, apple / metal-3) the English prompt's ids diverged between off / on at
position 56 of 64; addendum decision 5 put a rounding barrier on the restored quad. Withdrawn by
addendum 4 on the same day ([research §16.1](../research/2026-09-19-qat-speed-recon.md)).

## Addendum 4 (2026-09-20) — addendum 3 withdrawn; the parallel GEMV family spells its multiply-add with an explicit `fma()`

The barrier of addendum 3 did not help on M2; reading the cause as re-association of the expression was
wrong, so it is withdrawn and the barrier and the `rounding_mask` word were removed. A diagnostic script
(the former `tools/diag/packed-activation-probe.ts`, removed by bb35fe2a) run on M2 and RTX showed that
the RTX f32 path is the compiler contracting the multiply-add into fma, while Metal contracts differently
per expression form, even for identities; only an explicit `fma()` made the f32 and packed paths agree on
M2 in all four combinations.

### Addendum decision 6: the parallel family's multiply-add is `acc = fma(x, d, acc)`

- It applies to every variant generated by `parallelWgsl` (f32 / packed × without / with the linear→SRQ
  fusion). The sequential GEMV and the row block keep their former spelling so as not to move the
  numerics of the reference path. The subgroup32 variant was aligned to the same spelling on 2026-09-25
  (ADR 0101 addendum — it produces the same sum as parallel, so the reference-path exclusion does not
  apply).
- On RTX, numerics and ids do not change (540-case sweep; the QAT E2B 64 / 256-token outputs are
  identical to before).
- **On Metal, the output of the parallel path (`i4-fast` / `i4-gemvpar`) changes from its former
  values.** Those values were set by the compiler's contraction habits; afterwards packed on / off and
  fusion on / off agree. The reference preset `i4` (sequential) does not change.
- "One multiplication is a correctly rounded product" of decision 3 stays true. What diverged was the
  contraction of the multiply-add, not the restoration.

Acceptance (2026-09-20): green on RTX; on the maintainer's M2 the u32 checks were green and the ids of
English and Japanese prompts matched across off / on at a neutral speed (35.8 / 35.9 tok/s —
[research §16.1](../research/2026-09-19-qat-speed-recon.md)). The `i4-fast` declaration stays.

## Addendum 5 (2026-09-25) — the manifest after the container

The `packedStaticQuantize` field of `quants[].session` carries over unchanged into `karume/5` (ADR
[0109](0109-manifest-v5-container.md) decision 2). Storage is defined by codec / layout (ADR
[0108](0108-container-format.md) decisions 12 / 13, [container-v1](../container-v1.md) §6).
