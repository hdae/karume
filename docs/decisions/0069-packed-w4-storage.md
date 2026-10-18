# 0069: Packed 4-bit storage (reopening ADR 0019's rejection of w4)

- Status: accepted (2026-08-17 — maintainer rulings B (no zero-point field, with a reservation) and
  C (f32 scale first, f16 as a later goal) plus a delegated check; Codex review rounds 3 to 5 folded
  in, go at round 6); partly superseded by [0108](0108-container-format.md) decisions 10–13
  (2026-09-22: the declaration of the storage — dtype label, safetensors reader, zero-point field
  name — moved to the container's codec ledger; addendum 11) and by
  [0082](0082-linear-gemv-decode.md) (2026-08-31: the GEMV family unpacks in registers, not at tile
  load); decision 5's linear-only start was widened by addenda 6 and 7 (2026-08-20); decision 6's
  Phase 0 gate was withdrawn on 2026-10-18. The semantics (15 levels, offset 8, pack order, group
  size, group scale shape, f16-scale admission) stay canonical here.
- Date: 2026-08-17
- Related: ADR [0019](0019-i8-weight-execution.md) (i8 path — this ADR supersedes its "do not
  measure w4 again" and extends its implementation discipline) / [0006](0006-quantization.md)
  (storage-only quantization, fake-quant is the reference) / [0063](0063-safetensors-physical-layout.md)
  (physical layout, revised together with this ADR) / [0058](0058-numerics-opt-in-contract.md)
  (division of quality control) / [0066](0066-generation-context-state-slots.md) (the LLM work this
  serves)
- Evidence:
  [research/2026-08-17-autoregressive-references.md](../research/2026-08-17-autoregressive-references.md)
  §4 ("survey §n" below)

## Context

Distributing and keeping resident an acceptance-class model (1–2B parameters) costs 2–4 GiB even in
f16, so 4-bit storage is a practical requirement. ADR 0019 said "per-tensor / w4 group quantization:
rejected, do not measure again", but that scope was too wide (survey §4.3): the old measurement was
**one int4 group measurement of one SBV2 voice family at one point in time** (2026-08-03), and the
parent ADR 0006 only listed group-wise quantization as a low-priority backlog item. One reason for
the rejection, "systematically shorter utterances", **also appears with the adopted w8** (ADR 0029),
so it is no w4-specific criterion. 4-bit on WebGPU is in production in ORT MatMulNBits, MLC q4f16_1
and llama.cpp WebGPU (survey §4.1).

## Decision

### 1. What of ADR 0019 is superseded (the exact reopening)

- "Do not measure again" is withdrawn. The old measurement applies only to "SBV2 voice audio SNR /
  utterance length, per-tensor RTN and one int4 group measurement" and **is not generalised to text
  generation**. The attribution is corrected too: −1.5 to +5.1 dB is the int4 group measurement (ADR
  0019 is right; ADR 0006, which attributed it to naive RTN, is not — survey §7 LB-6).
- The rest of ADR 0019 (i8 path, ±127 symmetric, flat indexing, dequantization at tile load,
  fake-quant discipline) **stays valid as the base of this ADR**.
- The derived documents (the IR spec's "group_size has no execution path", the w4 entry in
  limitations, the container's groupSize rejection) were updated in the implementation wave.

### 2. Representation: generalised bit width (shape stays logical)

A storage dtype `i4` is added. **The tensor shape stays logical**; only the byte count is derived
from the bit width (the same layout as GGUF / safetensors upstream — survey §4.2). The ORT form
"physical shape + logical attributes" is not adopted: it would break every consuming op contract
(linear `W[out,in]` and others).

- **(superseded by ADR 0108 decisions 10–13 on 2026-09-22: the declaration `storage.dtype: "i4"` /
  `storage.group_size`, the three revised surfaces of the safetensors reader (bit-based size table,
  byte-based alignment table, raw `Uint8Array` view plus logical element count) and the ADR 0063
  write order moved to the container's `encoding` binding (codec `int4-sym-g`, `groupSize`,
  `scale`) and the codec ledger `CODEC_LEDGER` in `packages/runtime/src/format/container/codecs.ts`;
  see [docs/container-v1.md](../container-v1.md) §6 and addendum 11.)**
- **No remainders, so no alignment problem:** `i4` MUST have the element count of the quantized axis
  divisible by `group_size`, and `group_size` is **a power of two and ≥ 16** (the same constraint as
  ORT — survey §4.1; no single-scale exception). Row and group boundaries are then always
  byte-aligned and the total byte count is a multiple of 4 (natural u32 alignment, no tail padding).

### 3. Quantization: groups along K, symmetric, 15 levels, f32 scale

- **Symmetric group quantization:** `scale = clamp(amax / 7, f32 tiny)` per group (the clamp closes
  division by zero on all-zero groups; same wording as ADR 0019's `clamp(amax/127, f32 tiny)`);
  `q = clamp(round(w/scale), −7, +7)` stored as an **unsigned nibble with offset 8** (range [1,15],
  0 unused = 15 levels). Fake-quant is idempotent — the 4-bit version of ADR 0019's ±127 argument
  (same idea as MLC q4f16_1 — survey §4.1; addendum 3 refines which half holds exactly).
- **Scale shape:** the scale has **the weight's rank with the quantized axis replaced by the group
  count** (linear `W[O,I]` → scale `[O, I/group_size]`; ADR 0019's same-rank keepdim broadcast is
  per-channel only). The container's scale check and the WGSL scale binding have a separate group
  branch (index `(row, k / group_size)`); the CPU expansion uses the same index formula, so the two
  stay bit-identical. Addendum 7 generalised the shape to any rank.
- **No zero-point field** (ruled 2026-08-17 — symmetric only). Reservations so it can be added
  later: ① the field is a companion of the scale's shape `[O, I/group_size]` and an added field,
  not affecting existing assets or checks — **(superseded by ADR 0108 on 2026-09-22: the field name
  is `encoding.zeroPoint`, not `storage.zero_point`; it is enabled by changing the ledger's
  `zeroPoint: "forbidden"` to `"allowed"` — addendum 11)**; ② the dequantization formula
  `(u − 8) · scale` reads as "zero_point defaults to 8 when omitted" (the ORT convention; going
  asymmetric only replaces the constant 8 with a zero-point read — pack format, pack order and the
  15-level range stay).
- **The scale starts as an f32 companion; accepting f16 is a later goal** (ruled 2026-08-17 —
  eventually both; the code accepts f32 only today, `SCALE_DTYPES` in `codecs.ts`). An f16 scale
  keeps bit identity: f16 → f32 is exact, so CPU and GPU use the same f32 value of a scale rounded
  to f16 once at emit. When f16 is accepted, the scale companion's dtype accepts F32 | F16 and emit
  takes the fake-quant reference with the f16-rounded scale. **Extra condition for f16:** f32 → f16
  rounding is not exact — `f32 tiny` becomes 0 and a large scale can become inf. Emit admission
  checks that every rounded scale is **finite and non-zero** (same family as the existing f16 emit's
  inf rejection), and the clamp floor for all-zero groups becomes **the smallest f16 normal**. The
  size cost of an f32 scale can also be tuned by group size (overhead per bit width: group 32 —
  f32 +25% / f16 +12.5%; group 128 — f32 +6.25%).

### 4. Pack order fixed in writing, with detectors

**"Element 2i in the low nibble, element 2i+1 in the high nibble of one byte"** (ORT style — pairs
with the unpack order of two `unpack4xU8` calls). llama.cpp Q4_0's split-half order (byte j holds
element j low and element j+16 high) is not adopted. Pack order is a free parameter on which
upstreams disagree, and a mistake is **a silent wrong value with matching shape and type** (survey
§4.1). Detectors MUST exist: ① a unit with an asymmetric pattern where all neighbours differ; ②
shapes whose group and row lengths do not align with word boundaries (the flat-indexing trap — the
4-bit version of ADR 0019's detector); ③ a round-trip bit-identity check right after exporter emit
(`dequant(pack(w)) == fake_quant(w)`).

### 5. Execution path

- Eligible ops: `linear`, `embedding` and `conv1d` with `groups == 1` (`I4_WEIGHT_OPS` in
  `tools/exporter/src/karume/emit.py` and `packages/runtime/src/runtime/plan.ts`). **(superseded by
  addenda 6 and 7 on 2026-08-20: the record started with linear weight slots only.)** An i4
  declaration outside the eligible set is expanded on the CPU at load (f32, bit-identical to the
  GPU — the same fallback as ADR 0019).
- WGSL uses core builtins only (`unpack4xU8` + mask / shift — survey §4.1); words and lanes use flat
  indexing (MUST). Unpacking happened once at shared-memory tile load (ADR 0019 style; ORT's
  unpacking inside the dot product was not adopted) **(superseded by ADR 0082 on 2026-08-31: the
  GEMV family drops shared memory and unpacks each weight word in registers inside the dot product;
  the invariant kept is that GPU restoration and CPU expansion are bit-identical)**.
- The scale is applied per element (`q·s` — as ADR 0019; moving it outside the reduction has ADR
  0019's revision conditions — addendum 8).

### 6. Phase 0 = Python fake-quant sweep as the implementation gate

Withdrawn 2026-10-18: the gate was passed on 2026-08-18 (addendum 1). The division of quality
control (machine checks = fake-quant idempotence and round-trip bit identity; quality = human review
plus advisory numbers) is canonical in ADR 0058 decision 5.

## Size estimate (Gemma 4 E2B class, about 2B weight parameters)

| Form                                    | bpw  | Estimate  |
| --------------------------------------- | ---- | --------- |
| f16                                     | 16   | ~4.0 GiB  |
| i8 + per-channel f32 scale              | ~8.1 | ~2.0 GiB  |
| i4 group 128 + f32 scale                | 4.25 | ~1.06 GiB |
| i4 group 32 + f32 scale                 | 5.0  | ~1.25 GiB |
| (for reference) i4 group 32 + f16 scale | 4.5  | ~1.13 GiB |

## Consequences

- ADR 0019 carries a supersede note (limited scope); ADR 0006's attribution carries a correction.
- Acceptance checks of the implementation: existing f32 / f16 / i8 assets read and write unchanged;
  round-trip bit identity; the pack-order detectors of decision 4; GPU dequantization and CPU
  expansion bit-identical; diagnostics (`residentCompressedBytes` includes i4 + scale).

## Addendum 1 (2026-08-18) — Phase 0 done: default group size and ruling B confirmed

The sweep of decision 6 ran (canonical numbers:
[research/2026-08-18-w4-fake-quant-sweep.md](../research/2026-08-18-w4-fake-quant-sweep.md); script
`tools/export-recipes/minicpm5/sweep_w4.py`). Two points confirmed (maintainer ruling 2026-08-18):

1. **Default group size = 32 (symmetric).** Weight relative RMSE is monotone 32 < 64 < 128, and only
   g32 symmetric keeps free-running greedy reasonably (23/48 — others 3–8/48). The size cost is +18%
   over g128 (5.0 bpw, ~1.25 GiB at 2B), which does not defeat residency. `group_size` is a storage
   field; an asset that favours size may pick 64 / 128 (accepted set as decision 2: power of two,
   ≥ 16).
2. **Ruling B (no zero-point field) confirmed:** even an upper-bound measurement with a continuous
   zero point improves weight space (−11 to −15%) without improving model quality, and one finer
   group step gives the same gain. The reservations of decision 3 stay for a future addition.

Side finding: w4 RTN clearly costs quality on a 1B checkpoint (teacher-forced agreement 31–37/48).
This does not affect the acceptance method (agreement with fake-quantized weights — ADR 0006), but
w4 is a trade of quality for size. AWQ / GPTQ-class weight adjustment was not measured (a separate
ADR if needed).

## Addendum 2 (2026-08-18) — write order corrected

Decision 2's "append I4 after I8" is **corrected**: the order is **descending alignment unit** =
`F32 → I32 → I4 → even-length F16 → odd-length F16 → I8` (ADR 0063 revised at the same time).
Reasons: the I8 section has arbitrary length, so I4 after it cannot meet the 4-byte start alignment,
and the data section MUST be covered without gaps (ADR 0063); putting I4 earlier keeps later
alignment (the element count is a multiple of 16, so the byte length is a multiple of 8); existing
assets' bytes do not change. (The write order itself now belongs to the container — ADR 0108.)

## Addendum 3 (2026-08-18) — "amax restored exactly" refined (measured)

Of decision 3's claim, what always holds in f32 is **idempotence**. Exact restoration of amax,
`fl(7·fl(amax/7)) == amax`, can be off by 1 ulp (about 9.5% of 200,000 random groups of 16 — the
NOTE in `tools/exporter/tests/test_quantize.py`). Idempotence holds because the largest-magnitude
element of each group always lands on `|q| = 7`, so the scale re-derived from the rounded weights,
`fl(amax(|q·s|)/7) = s`, is a fixed point; ADR 0019's ±127 argument has the same structure (i8 NOTE
in `tests/test_emit.py`). The checks (round-trip bit identity, idempotence) are unchanged.

## Addendum 4 (2026-08-19) — acceptance on a real model and the mixed-storage API

Real-model w4 acceptance completed on Gemma 4 E2B (linear in i4 g32, embeddings in i8, the rest
f32; greedy K = 16 on a real GPU exactly matched the torch golden). Three implementation decisions:

1. **Mixed-storage API:** `export_to_file` / `publish_model` take `weight_dtype_overrides` (tensor
   key → storage dtype, taking precedence over `weight_dtype`), and `fake_quant_int8` / `int4` take
   `include` (a predicate on module FQNs). Eligibility rules, scale shapes and checks of each dtype
   stay as decided in this ADR and ADRs 0018 / 0019; only the assignment gained an expression. An
   explicit override that cannot be met **fails loudly** (unknown key, ineligible slot, i4 on an op
   outside the i4 set); the default keeps leaving ineligible weights in f32 silently (f32 is the more
   faithful side, not an approximation). An explicit `"f32"` excludes a tensor from the compressed
   default.
2. **Tied lm_head assignment:** a tied tensor consumed by both the embedding and the linear weight
   slot was i4-ineligible under decision 5 and i8-eligible; the fake-quant `include` **excludes it
   from the i4 side and keeps it in i8** (passing it through both rounds it twice and the scale
   ledger disagrees with the values). Storage is specified as "default i8 + linear explicitly i4".
   Addendum 6 later made such a tensor i4-eligible.
3. **RoPE position tables are not quantized:** with table lookup in decode, the cos / sin tables
   enter the embedding weight slots (the i8-eligible set), so an explicit `"f32"` exclusion is
   required. Rounding a position table, unlike rounding weights, **accumulates angle error along the
   position**. The census pins "f32 embeddings = exactly the number of tables the series declares".
   Gemma 4 dropped the tables from the distribution (ADR 0091, 2026-09-03); the series that still
   bakes them (minicpm5 `export_decode`) has two tables, `cos_table` / `sin_table`.

## Addendum 5 (2026-08-19) — measurement side: op opt-in and method screening

An extension of **fake-quant only**; the accepted storage set (decision 2) and emit eligibility do
not change.

1. **`fake_quant_int4` gains an explicit `op_types` opt-in** (default `(nn.Linear,)` — existing
   calls unchanged). It can widen to the i8 op types (`QUANT_MODULE_TYPES`) to measure non-linear w4
   without touching the runtime. The group axis generalises "the in axis" (`quantize.channel_rows`:
   conv flattens the receptive field `Cin·K` / `Cin·Kh·Kw` per output channel, `ConvTranspose1d` uses
   axis 1 of its transposed layout, embedding uses D per vocabulary entry). A conv scale is rank 2
   after flattening and **cannot reach emit structurally** (misuse is not silent).
2. **Measurement-only rounding methods live in `quant_methods.py`** (FP4 e2m1 / NF4 / MXFP4 /
   k-means codebook at three granularities). A separate module keeps "what `quantize.py` holds =
   what can ship". They return counts only, **never a scale / codebook ledger** (no path into emit).
   All are deterministic; target selection shares `iter_quant_targets` with the storage path.
3. Method comparison ran at g = 32 (maintainer ruling 2026-08-19); the group-size re-evaluation is
   addendum 9.

## Addendum 6 (2026-08-20) — first extension of decision 5: i4 embedding

**Embedding became i4-eligible** (maintainer approval, `0d8c6f6`).

1. **Eligible set:** `I4_WEIGHT_OPS = {linear, embedding}` at the time (mirrored in `emit.py` and
   `plan.ts`; eligibility, executor and estimate share one implementation per side).
2. **Storage shape unchanged:** the vocabulary table `[V,D]` has identity `channel_rows`, and the
   group scale `[V, D/g]` follows decisions 2 / 3.
3. **Execution:** the gather kernel reads the group scale per thread (one thread per output element,
   `wscale[pick·(D/g) + col/g]` — separate from linear's per-quad read). The group length is baked
   into WGSL as a shift, so the key carries it (`:wi4g32`). GPU dequantization and CPU expansion
   (`decodeI4`) are checked bit-identical over all 15 nibbles × 8 positions per word
   (`packages/runtime/tests/gpu_i4_weights_test.ts`).
4. **Tied lm_head reversed:** a tensor consumed by both linear and embedding slots is now
   **eligible** (both ops have an unpacking path).
5. Applied to DeBERTa (vocabulary table 22012×1024; 202.45 → 193.91 MB, 2026-08-20).

## Addendum 7 (2026-08-20) — second extension of decision 5: i4 conv1d and rank-independent scale

**conv1d with `groups == 1` became i4-eligible** (maintainer approval, `eac9d43`).

1. **Predicate:** op is conv1d ∧ `groups == 1` ∧ stored row length `Cin·K % group_size == 0`.
   `groups > 1` (depthwise) goes to the direct kernel with no unpacking path; `conv_transpose1d`
   has its row axis not first (`[Cin,Cout,K]`), so the pack order does not fit (no permuted pack for
   2.6 MiB — maintainer ruling the same day). Emit (`I4_WEIGHT_OPS` + `_has_i4_kernel`) and
   `plan.ts` (`I4_WEIGHT_OPS` + `i4Executable`) mirror each other in content and structure.
2. **Rank-independent scale shape** (revises decision 3): quantized axis = row length with the first
   dimension as rows and the rest flattened (exporter `quantize.storage_rows` / runtime
   `groupScaleShape` in `format/container/codecs.ts` — one implementation per side). Scale = rank 2
   `[rows, (numel/rows)/g]`; for rank-2 weights this equals the earlier rule. **Narrowing:** a rank-1
   i4 weight is no longer expressible (no eligible op has one). Pack bytes follow the weight's
   contiguous memory order regardless of rank.
3. **Execution:** conv1d's implicit GEMM (A side = `W[Cout, Cin·K]`) reads the group scale per quad
   while filling tiles (`fillAConv` in `gemm.ts`; an aligned quad never crosses a group ≥ 16). i4 on
   the direct kernel or conv2d is a `CodegenError` at the codegen entry. A conv1d missing attributes
   fails loudly at eligibility time (no default filling).
4. **Census and graph agree:** fake-quant counts from the module tree, so **a wrapper holding unused
   submodules** disagrees with the torch.export graph and emit's override spelling check (against
   silent drops) fails. SBV2 front's `SdpReverseNoiseIn` was restructured so that ownership equals
   use (`tools/export-recipes/sbv2/patch.py` — same objects, same call order, numerically unchanged).
5. Applied to SBV2 (2026-08-20, all 14 models re-exported; w4 download 237.5 MB, −36.3% against w8).

## Addendum 8 (2026-08-21) — decision 5's scale application: w4a8 reduces per group

Decision 5 applied the scale per element and left moving it outside the reduction to ADR 0019's
revision conditions. ADR [0076](0076-w4a8-linear-execution.md) (w4a8) made that change, and meets
the conditions here.

1. **Only the w4a8 path.** The default `linearCompute` (f32 compute) keeps per-element application.
2. **ADR 0019's reason for rejecting `(Σ x·q)·s` disappears in w4a8** — the same argument as ADR
   [0025](0025-w8a8-linear-execution.md) decision 4. The rejection was about order dependence of
   f32 reduction; w4a8 **reduces exactly in i32 within a group**, so order does not matter. The
   flush is fixed at group boundaries and rounding happens `k/g + 1` times.
3. **atol = 0 holds.** The CPU reference (`packages/runtime/src/reference/i8a8.ts`) rounds at the
   same positions the same number of times; the real-GPU comparison uses `assertExact` (`===` over
   all elements).

Tile geometry stays outside the numeric contract (the flush is at group boundaries, not tile
boundaries).

## Addendum 9 (2026-08-22) — group size re-evaluated: SBV2 stays at g32

Done as deferred in addendum 5 (canonical measurements and listening:
[research/2026-08-22-sbv2-g-axis.md](../research/2026-08-22-sbv2-g-axis.md)).

1. **Ruling:** SBV2 net_g keeps **g32** (`DEFAULT_GROUP_SIZE = 32` unchanged, storage shape
   unchanged). Of g16 / g32 / g64 (g128 rejected by estimate), g32 is closest to f32; by ear g16 moved
   the tension the **opposite way** (higher). The slightly low tension of addendum 7's application is
   **not reduced by the group axis**.
2. **Lesson (method):** weight-level fidelity is clearly better at g16 (nested-split guarantee plus
   −15% relRMS), yet waveform identity metrics come out in reverse order — **chaotic amplification
   in the flow / decoder means waveform metrics do not carry the order of g**. Group-size rulings
   need a listening check (the first numeric stage cannot even narrow the field).
3. **Scope:** this ruling covers SBV2 net_g (RTN, all roles). It does not generalise to anima. The
   rig's `--w4-group-size` argument (`e813b8c`) remains.

## Addendum 10 (2026-09-24) — where the scale-shape formula lives

The scale shape comes from `groupScaleShape` in `packages/runtime/src/format/container/codecs.ts`
(with `rowAxis`) for every consumer, including the CPU expansion `decodeI4` (`e034a136`).

## Addendum 11 (2026-09-25) — the declaration of decisions 2 / 3 moved to the container's codec ledger

- **Only the declaration moved:** the storage dtype `i4` and `group_size`, the three safetensors
  reader surfaces (size, alignment, view) and the ADR 0063 write order moved to ADR
  [0108](0108-container-format.md) decisions 10–13 and the `encoding` binding in
  [docs/container-v1.md](../container-v1.md) §6 (codec `int4-sym-g` / `groupSize` / `scale`) with
  the codec ledger (`CODEC_LEDGER` in `packages/runtime/src/format/container/codecs.ts`). The
  safetensors dtype `I4` is not accepted (ADR 0063, addendum of 2026-09-24).
- **The semantics stay canonical here:** range (15 levels, unsigned nibble with offset 8), pack order
  (decision 4), group scale shape (addendum 10), dequantization `(u − 8) · scale`, group-size
  constraint (power of two, ≥ 16) and the f16-scale admission conditions.
- **The zero-point field's real name:** not `storage.zero_point` but `encoding.zeroPoint` (a block
  reference — container-v1 §6.1), enabled by changing the ledger's `zeroPoint: "forbidden"` to
  `"allowed"`. The reading "offset 8 = default when the zero point is omitted" stays reserved.
