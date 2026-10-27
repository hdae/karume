# 0135: Vocabulary extension: gated_delta_scan (the Gated DeltaNet recurrence, an extended composite op)

- Status: proposed (2026-10-27). Acceptance waits for the export measurements of entry condition ①
  (open point 1). The kernel-side numbers below are measured on one GPU; what is still unmeasured on
  the kernel side is listed in open point 10.
- Date: 2026-10-27
- Scope: the new op `gated_delta_scan` (the value form, for prefill), its kernel, its CPU reference,
  the custom op `karume::gated_delta_scan` and one exporter handler. The states form for decode is
  named here (decision 4) and decided with the first Qwen3.5-class language model.
- Related: [0056](0056-gru-scan.md) (`gru_scan`, the precedent this ADR follows and departs from) /
  [0059](0059-op-vocabulary-entry-doors.md), [0064](0064-entry-door-generality-axes.md) (entry doors,
  semantic surface and implemented subset, execution-model invariants) /
  [0012](0012-attrs-and-fused-ops.md) (empty attrs) / [0067](0067-autoregressive-attention-vocabulary.md)
  (divisible head broadcast; the states form) / [0066](0066-generation-context-state-slots.md) (state
  slots) / [0068](0068-decode-exit-multi-output.md) (multi-output; `operator.getitem`) /
  [0058](0058-numerics-opt-in-contract.md) (no automatic numerics-changing choice) /
  [0110](0110-practical-tier-numerics-contract.md) (contract classes, in-device determinism) /
  [0115](0115-geometry-profiles.md) (geometry chosen per adapter only when bit-identical) /
  [0005](0005-verification.md) addendum 2026-10-11 (upstream parity) /
  [0065](0065-exporter-core-recipe-split.md) (exporter core and recipes) / research
  [2026-10-27-gdn-scan-spike](../research/2026-10-27-gdn-scan-spike.md) (the measurements behind
  every number here; section numbers below are that record's).
- Demand: the text encoder of MOSS-Image (Qwen3.5-9B: 32 layers, **24 of them Gated DeltaNet**,
  prefill only, T up to 4096 tokens, no cache). Later, Qwen3.5-class language models (prefill plus
  one-token decode with a carried state).

## Context

A Gated DeltaNet (GDN) layer (transformers 5.14.1, `Qwen3_5GatedDeltaNet` in
`models/qwen3_5/modeling_qwen3_5.py`) projects the hidden state with four `nn.Linear` (`in_proj_qkv`,
`in_proj_z`, `in_proj_b`, `in_proj_a`), runs a causal depthwise conv1d (kernel 4) and SiLU over q / k
/ v, splits them into heads, computes `beta = sigmoid(b)` in the model dtype (`modeling_qwen3_5.py`
:520) and the log decay `g = −exp(A_log)·softplus(a + dt_bias)` in f32 (`a.float()`, :522), repeats
q / k from the key heads to the value heads (`repeat_interleave`, :523-525), runs **the
recurrence**, applies a gated RMSNorm and `out_proj`. The MOSS layer has 16 key heads and 32 value
heads of dimension 128, conv kernel 4, and `mamba_ssm_dtype` float32 (the state is f32).

The recurrence has two upstream implementations, both called with `use_qk_l2norm_in_kernel=True`:
`torch_chunk_gated_delta_rule` (prefill; 64-token chunks with a WY / UT transform) and
`torch_recurrent_gated_delta_rule` (cached one-token decode; a token loop). Both l2-normalise q and k
in the input dtype (`x·rsqrt(Σx² + 1e-6)`), then cast q / k / v / β / g to f32 (:261-266,
:333-337), scale q by `dk^-½`, start from a zero state (prefill) or the carried state (decode), and
loop in Python over chunks or tokens. They agree in exact arithmetic but
not in f32: against an f64 oracle the chunked function's output error reaches o rel_l2 5.6e-7 in the
fast-decay regime where the recurrent one stays at 8.5e-8 (research §5.1).

Nothing in Karume implements the recurrence. Everything around it maps onto existing ops (`linear`,
`conv1d` with `groups`, `sigmoid` + `mul` fused to SiLU, `slice` / `reshape`, the softplus
decomposition, the `rms_norm` fold), except two export gaps outside the recurrence found by reading
the code (not measured — open points 2 and 3).

## Decision

### 1. The op is the recurrence only (candidate A): `gated_delta_scan(q, k, v, g, beta) → o`

**Fixed arity 5, uniform f32, empty attrs.** Notation: B batch, T tokens, Hk key heads, Hv value
heads, dk key dim, dv value dim, `r = Hv / Hk`.

| Tensor | Shape          | Contract                                                                    |
| ------ | -------------- | --------------------------------------------------------------------------- |
| `q`    | `[B,T,Hk,dk]`  | queries, **already l2-normalised, not scaled**                              |
| `k`    | `[B,T,Hk,dk]`  | keys, already l2-normalised                                                 |
| `v`    | `[B,T,Hv,dv]`  | values                                                                      |
| `g`    | `[B,T,Hv]`     | log decay per value head and token (≤ 0 in GDN; not enforced — decision 6) |
| `beta` | `[B,T,Hv]`     | write strength per value head and token                                     |
| output | `[B,T,Hv,dv]`  | `o`                                                                         |

Per batch element, value head h (reading key head `⌊h/r⌋`) and token t, with the state
`S ∈ ℝ^{dk×dv}` (row = key dim, column = value dim):

```
S_0   = 0
α_t   = exp(g_t)
S'    = α_t · S_{t−1}                 # decay first
m     = S'ᵀ k_t                       # what the state predicts for k_t
δ     = β_t · (v_t − m)
S_t   = S' + k_t δᵀ                   # rank-1 delta-rule update
o_t   = S_tᵀ (q_t · dk^-½)            # read after the update: token t is included
```

- **Layout** = upstream's interface layout `[B,T,H,d]` (what the projections give after a reshape),
  contiguous; upstream's internal head-first transpose is not exposed. T is the symbolic length axis.
- **Head broadcast is derived from the shapes**, as attention's GQA (ADR 0067 decisions 1–2):
  `Hv % Hk == 0` is required (otherwise the contract check fails loudly) and value head h reads key
  head `⌊h/r⌋`, exactly `repeat_interleave(r, dim=2)`. `Hk = Hv` (q / k already repeated) is the same
  contract with r = 1. Measured: r = 1 and r = 2 give bit-identical o and final state on 90 / 90 runs
  (research §5.2).
- **The q scale `dk^-½` is inside and derived from dk** (no attr): it sits inside both upstream
  functions unconditionally. A `scale` attr MAY be added when a model needs another value.
- **g enters as the log decay**, not as `α`: both upstream functions take it so, and a log-space
  form is what any chunked or blocked kernel needs.
- **dtype**: f32 for all five inputs and the output. Upstream computes g and the state in f32 (β and
  the l2-normalisation in the model dtype, cast to f32 before the recurrence) and casts the output
  back to the model dtype; in an f32 export the casts are no-ops.

**Fields left out** (a missing field states that the form is not in the vocabulary — 0056 decision 7):

| Axis                    | Treatment                                                                                                                     |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| q / k l2-normalisation  | Outside the op (candidate B below). The recipe computes it with existing ops                                                 |
| initial state           | No slot: the value form starts from zero. MOSS prefill passes none. A carried state is the states form (decision 4)         |
| final state output      | Not returned: MOSS discards it, and value-form multi-output stops at `operator.getitem` in export (ADR 0068 addendum 2, item 3) |
| q scale                 | Derived (`dk^-½`), no attr                                                                                                   |
| head repeat             | Derived broadcast, no attr                                                                                                    |
| variants (bool attrs)   | None. A variant becomes a separate op name, as `gru_scan_reverse` (0056 decision 2)                                         |

**Semantic surface and implemented subset** (0064 axis A): the semantics hold for any B, T ≥ 1, dk,
dv and `Hk | Hv`; the CPU reference implements all of it. The GPU kernel implements **dk = dv = 128**
(the only measured head dims) and fails loudly on others in the contract check and in its params, as
`GRU_SCAN_MAX_HIDDEN` does. B > 1 runs through the dispatch grid but was measured only at B = 1; the
landing test covers B = 2. Generalising the dims needs a measured geometry per dim.

### 2. Door: extended composite layer — entry condition ③ applies; ① is expected and gates acceptance

`karume::gated_delta_scan` is Karume's own name (outside aten), and the recurrence **can** be written
exactly as a composition of existing ops (per token: `exp`, `mul`, `sum`, `sub`, `add`; T times), so
it is composite: the extended composite layer, which needs a measurement for at least one of entry
conditions ① to ④ (each is addressed below).

- **① Semantic information is lost when restoring from the decomposition — expected to apply as for
  `gru_scan`, not measured yet.** Both upstream functions loop in Python, over T tokens or over T/64
  chunks with a T-dependent padding, so `torch.export` is expected to specialise the length axis and
  unroll; "this is a delta-rule scan" would have to be restored from a node sequence that differs per
  T. The spike did not run the export measurements (research §9). MUST before acceptance: the
  specialisation under `Dim("T")`, node counts after `run_decompositions(curated_decompositions())`
  at fixed T, and one node with a symbolic T through the custom op (open point 1).
- **② Materialising the intermediate — the op side is measured, the decomposed side is not.** The
  kernel keeps the state in registers and allocates no scratch (`scratchBytes` 0); one layer at
  T = 4096 takes 2.56 ms in one dispatch (research §4.1, §4.5). The decomposed recurrent form, by
  construction, materialises the full per-layer state (2 MiB at the MOSS shape) at every token step;
  its traffic and dispatch count on the GPU come with the measurements of ①.
- **③ The recurrence rate justifies the commitment — applies.** One form serves 24 of the 32 layers
  of the MOSS encoder, and the GDN layers of Qwen3.5-class language models (prefill here, decode
  through decision 4).
- **④ Attr variants — does not apply now.** Candidate B, if it is ever needed, is a separate name.

**Axis B** (0064): the value form breaks none of single output, full-write (checked by NaN-poisoned
outputs — research §3), 4-byte storage and full-range write of resident tensors; its symbolic T is
the same kind as `gru_scan`'s. The states form of decision 4 is a new kind of state effect
(read-modify-write of a slot), so its ADR states the impact up front.

### 3. Kernel: the sequential form with an 8-way key split, for every T

Measured on the RTX 3080 Ti (Node + Dawn / Vulkan, f32, no subgroups), one MOSS GDN layer, median µs
per layer (research §4.1, §4.2, §4.5):

| Form                                                       | Dispatches | Fits defaults | T = 64 | T = 4096 | 24 layers, T = 4096 | Scratch, T = 4096 |
| ---------------------------------------------------------- | ---------- | ------------- | ------ | -------- | ------------------- | ----------------- |
| (a1) sequential, one thread per value column               | 1          | yes           | 50     | 3563     | 85.94 ms            | none              |
| **(a2) sequential, key split 8** (`s8:c32:tb8`)           | 1          | yes (13.6 KiB) | 41    | 2556     | 62.01 ms            | none              |
| (a2) sequential, key split 8 (`s8:c16:tb16`)              | 1          | no (19.6 KiB)  | 39    | 2357     | 57.56 ms            | none              |
| (a3) blocked, 8-token blocks (`s4:c16:b8`)                | 2          | yes (14.8 KiB) | 41    | 2368     | 57.01 ms            | 13.25 MiB         |
| (a3) blocked, 8-token blocks (`s4:c32:b8`)                | 2          | no (20.8 KiB)  | 38    | 2131     | 51.39 ms            | 13.25 MiB         |
| (b) chunked WY / UT, 64-token chunks as upstream (`bv32`) | 3          | no (24.5 KiB)  | 143   | 3411     | 82.46 ms            | 320.5 MiB         |

For scale: the layer's `in_proj_qkv` with the product's f16-weight linear takes 22.1 ms at T = 4096
(the yardstick varies by about 15 % between processes), so the fastest scan is 9.6 % of that one
projection (research §4.6).

- **The op's kernel is (a2)**: a sequential token loop in one dispatch; one value column is split
  over **8 threads by key slices** (16 key rows each); per token, the partial sums of `m` are
  exchanged through workgroup memory in a fixed order and reduced with one barrier; the token's k is
  held in registers; q / k / v / g / β of a block of tokens are staged in workgroup memory. The
  default geometry `s8:c32:tb8` (256 invocations, 13.6 KiB of workgroup storage) fits the WebGPU
  defaults, and its time under a device requested with the defaults is within 2.2 % of the adapter
  limits (research §4.4). Why this form: it is within 1.2× of the fastest form at T = 4096 and within
  9 % at T = 64 (41.5 against 38.1 µs for `blk:s4:c32:b8`), needs one dispatch and no scratch, has
  the smallest error of the forms in the fast-decay regime (o rel_l2 7.0e-8 at T = 4096 — research
  §5.1), and is the same kernel family as decode (decision 4).
- **The key split is part of the numerics** (it fixes the order of the partial sums of `m` and `o`)
  and lives in the key; it does not vary by device. **Columns per workgroup and the token block are
  geometry.** A larger geometry such as `s8:c16:tb16` (−7.8 % at T = 4096, 19.6 KiB) MAY be chosen
  per adapter in the manner of ADR 0115 **only after a Uint32 check on that adapter shows it
  bit-identical** to the default geometry. On this GPU the independent check's re-run compared the
  raw output words: `s8:c32:tb8` and `s8:c16:tb16` (both with the accurate exp) are bit-identical in
  every word of o and S on `init_T4096` and `long_T4096` (0 differing words — research §6.1); other
  cases, GPUs and backends are not measured, so the per-adapter check stays (open point 6).
- **MUST: `α = exp(g)` uses an accurate range-reduced exp, not WGSL's built-in `exp`.** On this GPU
  the built-in `exp` is biased on the long-memory range (mean −0.34 ULP on [−0.1, 0)), and the
  per-token product of decays carries the bias across the memory: with it, (a2) fails all four
  long-memory prefill cases on o and S together (2.455× the bound at worst, on S's rel_l2 at
  T = 512). On o alone — the only output of the value form — it fails at T = 512, 2048 and 4096 (o
  rel_l2 up to 9.4e-7 against a bound of 5.1e-7 at T = 4096; 1.83× the bound at worst) and passes at
  T = 64. With the accurate exp it uses at most 0.500 of the bound (0.434 on o alone — research §5.4,
  §6). The accurate exp is computed once per token at staging (no measurable time) and is a shared
  WGSL string, so other kernels that multiply per-step `exp` outputs can use the same body. **Its
  contract**, stated with the body: it is defined for `x ≤ 0` (what this op needs). Below −87 it
  returns 0, which also flushes the normal results on (−87.34, −87) to 0 (negligible). Outside
  `x ≤ 0` the result is unspecified: above about 88.4 the true result overflows f32
  (`n = round(x·log2 e) ≥ 128`), and WGSL's `ldexp` gives an indeterminate value once the exponent
  exceeds bias + 1 = 128 (a reading of the WGSL spec, not tried on the GPU), so +inf is not
  guaranteed. Its truncation error is at most r⁸/8! ≈ 5.2e-9 for |r| ≤ ln 2 / 2, about 7e-9 relative
  to e^r — well under 1 ULP (research §8, finding 1).
- **No switch of form by T or by workgroup-storage limit.** (a3) is faster at long T but rounds
  differently (its block-wise UT algebra; 8- and 16-token blocks give different metrics on 8 of 10
  cases — research §6). Choosing it by T would make the encoder's output depend on the prompt length,
  and choosing it by limit would make it depend on the device: the automatic numerics-changing choice
  ADR 0058 decision 2 forbids. The gain does not argue for an opt-in now either: 62.01 → 51.39 ms over
  24 layers at T = 4096 (−10.6 ms) against **[estimate]** about 530 ms (24 × 22.1 ms) for
  `in_proj_qkv` alone. (a3) is recorded as a performance candidate (perf ledger K-92); if it is
  ever wanted, it enters as a named option (ADR 0058 decision 1) that states its ADR 0110 class (it
  changes more than the addition order).
- **The chunked form (b) is rejected on this GPU.** It is upstream's prefill algorithm, but it is
  slower at every T (3.41–3.46 ms at T = 4096, 126–128 µs at T = 1 because it pads to one chunk),
  needs 3 dispatches and 320.5 MiB of scratch per layer at T = 4096, its chunk-start states reach
  exactly the default `maxStorageBufferBindingSize` at T = 4096, and it is the least accurate form in
  the fast-decay regime (o rel_l2 3.5e-7 against 7.0e-8 for (a2) at T = 4096). Its time is spread
  over its three kernels (prep 1.37 / state pass 1.09 / output 0.96 ms for `chunk:bv32` at T = 4096 —
  research §4.3). Running its matmul-shaped parts on the product's bmm alone would take 3.57 ms, more
  than the whole form, so the product's matmul does not rescue it (research §7); an ablation of the
  chunk kernels was discarded because it ran in the machine's slow mode (research §8, finding 4).
  Revisit if matrix units become available to the product.
- **(a1) is rejected**: slower at every T (3.56 ms at T = 4096) with 128 state floats per thread.
- The sequential form is bound by the latency of one workgroup's token step (about 960 cycles), not
  by throughput (research §7); a dispatch at T = 4096 lasts 2.56 ms, so no split of T into several
  dispatches is needed for the MOSS range. Splitting T with a state round trip through storage is the
  way out if a longer T ever needs it (not measured).

### 4. Decode is the states form of the same op (named here; decided with the first language model)

The language models' one-token decode carries S across steps. Following ADR 0067 decision 4 ("the
presence of the field decides the form"), the states form adds a `states: { state }` field that names
a generation-context slot `[B,Hv,dk,dv]` f32, which the op reads as `S_0` and overwrites with `S_T`.
The kernel is (a2) with the state read and written in place. Measured at T = 1: 4.2 µs per layer
(key split 4, `s4:c16:tb2`), 4.8 µs with the prefill geometry in place (a tuning run with the
built-in exp, not the accurate one — research §7), 0.15 ms for 24 distinct layers in one pass; (a3)
in place 13 µs; the chunked form 126 µs. Both T = 1 cases pass the acceptance (research §4.1, §4.2,
§5.1, §7). Left to that ADR (open point 5): the in-place state effect (full write of the slot,
ordering by ADR 0067 decision 5b, rewind and fork of ADR 0066), the conv-state slot (the last 3 conv
inputs per channel), the decode geometry (key split 4 and 8 round differently — research §6), and
prefill with a carried state (upstream passes the cached state to the chunked function too).

### 5. Export capture: a `karume::` custom op that keeps T symbolic

As ADR 0056 decision 6:

- `karume::gated_delta_scan` is a `torch.library.custom_op` with `register_fake` in
  `tools/exporter/src/karume/custom_ops.py`. The fake returns a tensor shaped like `v` with T taken
  from the input, so T stays symbolic, and `karume::` is not in the Core ATen decomposition table, so
  `run_decompositions` does not unroll it.
- **MUST: the eager body is a verbatim copy of upstream's prefill function**
  `torch_chunk_gated_delta_rule` (transformers 5.14.1), called with `use_qk_l2norm_in_kernel=False`
  (q / k arrive normalised), `initial_state=None` and `output_final_state=False`, after the
  `repeat_interleave(r, dim=2)` of q / k that upstream's `forward` does before the call. Goldens come
  from it. It is a copy, not an import, because the exporter core depends only on torch, torchvision,
  numpy and safetensors (ADR 0065); the copy carries its upstream version.
- **The model port's patch** (`tools/export-recipes`) swaps the instance attributes
  `chunk_gated_delta_rule` / `recurrent_gated_delta_rule` of each `Qwen3_5GatedDeltaNet` (set in its
  `__init__`, :423-424) for a wrapper that applies upstream's `l2norm` with existing ops and calls the
  custom op, so `forward` is not copied. The wrapper receives q / k after `forward`'s
  `repeat_interleave` (:523-525), so the swap gives r = 1 for free. r = 2 (reading half the q / k
  data) needs either a change to `forward` or a strided slice in the wrapper that undoes the repeat,
  and with the slice the repeat itself still stays in the export graph (open point 4); both layouts
  are this contract.
- **MUST: the recipe makes sure the stock `forward` runs** before relying on the attribute swap (or
  disables hub kernels): `Qwen3_5GatedDeltaNet` carries `@use_kernel_forward_from_hub` (:373), so
  where the `kernels` package is active `forward` itself may be replaced (not verified).
- **Upstream's own kernel choice**: where `flash-linear-attention` is installed, upstream binds its
  Triton kernels instead of the torch functions (`chunk_gated_delta_rule or
  torch_chunk_gated_delta_rule`, :73-77, :423-424; likewise `FusedRMSNormGated` and
  `causal_conv1d`). The swap makes export independent of this, but an upstream reference run is not:
  the goldens and the upstream errors of the band come from the torch functions, so a recipe's
  reference run MUST record which path ran (the exporter's venv has neither package installed today).
- The handler maps the custom op to the IR op of the same name and narrows rank (4 / 4 / 4 / 3 / 3),
  dtype (f32) and arity (5). One contract set: TS `OP_CONTRACTS`, `karume/ops.py`, the shape rule in
  `shapes.py` and the TS shapes, `fixtures/op-contracts.json`, the CPU reference, and a golden
  `gated_delta_scan_block` with a symbolic T.

### 6. Reference and tolerance: an f64 oracle with a band set by upstream's own f32 error

- **Bit identity is not the proof** (unlike `gru_scan`): there is no single upstream rounding
  sequence. Upstream's two functions round differently, the GPU's `exp` is not torch's, and every GPU
  form differs from both functions by up to their own error ((a2) against `torch_chunk_gated_delta_rule`
  in f32: o max_abs 7.19e-7 at T = 4096 in the fast-decay regime, where the upstream function itself
  is 7.2e-7 from the oracle — research §6). Correctness is shown another way (ADR 0005 addendum
  2026-10-11).
- **The CPU reference evaluates the recurrence in f64** (`Float64Array`), not with `Math.fround` per
  step as `referenceGruScan` does. With `‖k‖ = 1` and `β ∈ (0, 1)`, `I − β k kᵀ` has eigenvalue 1 on
  the directions orthogonal to k (and `1 − β` along k), so the transition's norm is exactly α: the
  recurrence is non-expansive for `α ≤ 1`, and strictly contractive only when `α < 1` (g < 0, as in
  GDN). The measured error does not run away with T: (a2) o rel_l2 7.0e-8 at T = 64 and at
  T = 4096 (fast decay), 1.0e-7 and 1.7e-7 (long memory). The spike's TS f64 reference matches the
  Python f64 oracle to o rel_l2 4.8e-16 (research §1).
- **The band** (the spike's acceptance, adopted): the outputs are finite, and max_abs and rel_l2 of o
  (and of the final state where a form exposes it) are each at most **2 × the larger of the two upstream
  f32 functions' errors** against the same oracle on the same inputs. Measured: (a2) uses at most
  0.500 of it, (a3) 0.634, (b) 0.751, and the built-in-exp regression lands at 2.455 (research §6).
  That 2.455 is the final state's rel_l2 at long_T512; the value form exposes only o, so the band the
  product test can apply is the o band. On o alone the built-in-exp regression's share (max_abs /
  rel_l2) is 0.937 / 0.747 at long_T64 (passes), 1.157 / 1.658 at long_T512, 1.366 / 1.805 at
  long_T2048 and 1.120 / 1.834 at long_T4096 (these three fail): the o band catches it with a
  margin of 1.83× at worst, and not at all at T = 64 (research §6). In the fast-decay regime the
  band is set by the upstream chunked function's large error, and (a2) uses only about 0.07 of the
  o band there (0.068 at init_T4096 — research §6.1), so a numeric regression of roughly 14× passes
  there; coarse faults (share ≥ 1e6 — research §5.3) are always caught wherever they can act. **In
  the fast-decay regime the band guards coarse errors only.** The fixed atol / rtol of the product's
  GPU-vs-reference test are derived from these records at landing, with the derivation in the test
  constant's docstring (open point 7).
- **MUST (landing tests): the default lane, not an opt-in one, catches a built-in-exp regression.**
  Two tests: a direct test of the accurate-exp body — its bias and ULP error on [−0.1, 0) against an
  f64 exp, which the built-in exp fails (mean signed −0.337 against −0.091 ULP, max 3.00 against
  1.00 ULP — research §5.4) — and a long-memory case with T ≥ 512 in the default lane. That fixture
  MAY reduce the head count (the recurrence is per head) so the tracked fixture stays small.
- **Fixtures**: generated in Python as in the spike (seeded inputs, both upstream f32 outputs, the f64
  oracle, and the upstream errors), in two decay regimes (fast decay; long memory with g × 0.001).
- **Non-tautology (MUST, kept as tests)**: the five fault injections of the spike each fail the band
  wherever they can act — no decay 10 / 10, no β 10 / 10, key head `h % Hk` 10 / 10, output read
  before the update 10 / 10, missing initial state on both T = 1 cases (it cannot act in prefill)
  (research §5.3). Outputs are pre-filled with a NaN pattern (full-write).
- **Upstream parity**: a golden against the eager body runs in the opt-in parity lane; its band is at
  least upstream's chunked function's own error.
- **In-device determinism** (ADR 0110 decision 6): the kernel has a fixed reduction order and no
  atomics. Measured on this GPU by the independent check's re-run: two runs each of
  `seq:s8:c32:tb8:u:kreg:xexp`, `seq:s8:c16:tb16:u:kreg:xexp` and the built-in-exp
  `seq:s8:c32:tb8:u:kreg` at long_T4096 are Uint32-identical in o and the final state (research
  §6.1); other cases, GPUs and backends are not measured. A repeated-run Uint32 check stays part of
  the landing tests.
- **Numerically hazardous class** (the MUST of `docs/op-vocabulary.md`: the op contains `exp`). In
  GDN, `g = −exp(A_log)·softplus(·) ≤ 0`, so `α ∈ [0, 1]`; strongly negative g underflows α toward 0,
  which is the true value to f32 precision (the fast-decay inputs reach g = −77 at T = 4096 and pass).
  The contract does not enforce `g ≤ 0`: a positive g makes the transition expand and is outside the
  accurate exp's domain, so its result is unspecified (decision 3: not guaranteed to be +inf); the op
  does not guard it, and its doc says so. The GPU tests keep canaries for g = 0 and −0 (α exactly 1),
  the fast-decay band, g = −inf (α = 0) and NaN in g and v; their exact thresholds are derived at
  landing.

## Consequences

- One new op name and one contract set, one kernel (one geometry at first), one exporter handler and
  the custom op. **No effect on existing IR or assets.**
- A shared accurate-exp WGSL body enters the codegen numerics, with its domain contract (defined for
  `x ≤ 0`, unspecified outside — decision 3). Any later kernel that multiplies per-step `exp` outputs
  over long sequences inherits the bias of the built-in one and SHOULD use it.
- MOSS at T = 4096 on the RTX 3080 Ti: about 62 ms of scan over the 24 GDN layers with the default
  geometry. **[estimate]** By multiply-accumulate count, a GDN layer's projections — `in_proj_qkv`,
  `in_proj_z`, `in_proj_b` / `in_proj_a` and `out_proj` (4096 · 16448 MAC per token) — are about
  2.0 × `in_proj_qkv` (the four `nn.Linear` of the Context alone are about 1.51 ×). Against them the
  default geometry's scan is about 6 % (2556 / (2 × 22099) ≈ 5.8 %); about 5 % holds only for the
  fastest form (a3).
- Performance candidates in the [perf ledger](../perf-ledger.md): K-92 (a3) as a named option
  (−10.6 ms per 24 layers at T = 4096); K-93 `s8:c16:tb16` per adapter after the Uint32 check on
  that adapter (−7.8 % per layer); K-94 the runtime-bounded loop trips in `gru_scan`'s kernel (about
  60–67 cycles per trip measured on such loops here — research §8, finding 2).

## Alternatives considered

- **Candidate B — the recurrence plus the q / k l2-normalisation** (`gated_delta_scan_l2norm`,
  attrs `{ eps }`): the eager body would be upstream's function with `use_qk_l2norm_in_kernel=True`
  verbatim and the `rsqrt` gap would disappear, but the permanent contract grows (an attr; the
  normalisation's rounding becomes part of the op), and a GDN model without in-kernel normalisation
  would need A anyway under a second name (④). Its saving is unmeasured. Kept as the fallback if the
  decomposed l2-normalisation proves material or the `rsqrt` rewrite fails (open point 2).
- **Candidate C — the whole mixer** (projections, conv, gates, recurrence, gated norm, `out_proj`):
  rejected for the reasons option G was rejected in ADR 0056 decision 1. The projections would become
  op-internal weight slots and lose `linear`'s f16 / i8 / a8 storage, while they are almost all of the
  layer's cost (the scan is 9.6 % of `in_proj_qkv` alone); the contract would carry the conv, the
  softplus threshold, the gates and the norm; decode would need two states in one op. A partial
  variant that absorbs only the gates adds two weight slots and the softplus threshold to the
  contract to save a few small elementwise dispatches.
- **A required initial-state input and a final-state output in the value form** (the `gru_scan`
  shape): a zero initial state would fold into a 2 MiB initializer per layer (48 MiB over the 24
  layers) in the distribution, and the output would need the `operator.getitem` wiring that export
  does not have; the carried state belongs to the states form.
- **Bit identity against a T-fixed decomposed graph** (ADR 0056 decision 3): the comparison partner
  there is Karume's own decomposition into existing ops on the same GPU, not torch, and the IR `exp`
  op lowers to WGSL's built-in `exp` (`packages/runtime/src/codegen/elementwise.ts:292`). That graph
  multiplies the same biased per-token decays, so it would fail the band itself in the long-memory
  regime (by the mechanism of decision 3; the decomposed graph was not run), and being bit-identical
  to it would pin the very `exp` that the MUST of decision 3 forbids.
- **A `Math.fround`-per-step reference**: a non-expansive recurrence (strictly contractive for
  g < 0) is better judged against f64; the band is defined against f64.
- **Choosing (a2) or (a3) by T** (for example (a3) from T = 64): rejected in decision 3.
- **The chunked form (b) and (a1)**: rejected in decision 3. An associative (parallel-prefix) scan
  over chunk transition matrices was rejected without measuring in the design note (composing two
  128 × 128 transitions costs more than a chunk's whole work).

## Open points

1. **Export measurements of entry condition ①** (acceptance gate): the length-axis specialisation of
   both upstream functions under `Dim("T")`, node counts at fixed T after `run_decompositions`, the
   custom op as one node with a symbolic T, and a 1-layer `Qwen3_5GatedDeltaNet` through the
   exporter. The decomposed form's GPU dispatch count and traffic (②) come with them.
2. **`rsqrt` in the l2-normalisation**: `rsqrt` is not in the IR vocabulary (it is only folded into
   `rms_norm`). Either a normalize rewrite `rsqrt(x)` → `div(1, sqrt(x))` checked against torch
   (`rsqrt` is in Core ATen, so a ledger NOTE), or candidate B.
3. **The causal conv1d's trailing `[..., :T]`** is a `slice` on the symbolic axis `T + 3`, which
   `slice` rejects (static axes only). A recipe-side rewrite (left `pad` with no conv padding — its
   symbolic-axis support is unverified) or a contract change, decided outside this ADR.
4. **The recipe's head layout**: r = 1 is what the attribute swap gives for free (the wrapper
   receives the repeated q / k). r = 2 reads half the q / k data but needs one of two routes: a
   change to `forward`, or a strided slice in the wrapper that undoes the repeat (the repeat then
   stays in the export graph). Same contract, same bits (decision 5).
5. **The states form for decode** (decision 4), with the first Qwen3.5-class language model.
6. **Bit identity between geometries of (a2)** (a Uint32 check) on each adapter before its
   per-adapter geometry. On the RTX 3080 Ti `s8:c16:tb16` matched `s8:c32:tb8` word for word on two
   cases (research §6.1).
7. **The product's band constants and fixture sizes** (the spike's fixed inputs are 1.4 GB in all):
   tracked fixtures stay small — the default lane's long-memory case with T ≥ 512 keeps its T and
   reduces the head count (decision 6) — and the full MOSS shape at large T goes to an opt-in lane;
   and the hazard canaries' thresholds.
8. **(a3) as a named option** and its ADR 0110 class, if the 10.6 ms is ever wanted.
9. **bf16 parity**: upstream MOSS runs in bf16 (the l2-normalisation and β in bf16 before the f32
   recurrence); Karume's f32 path is checked against f32 upstream runs and the f64 oracle only.
10. **Not measured**: the real `A_log` / `dt_bias` of MOSS, B > 1, head dims other than 128, other
    GPUs and backends (Chrome on D3D12, Metal), the hazard canaries (g = 0 / −0 / −inf / NaN), and
    in-device determinism and geometry identity beyond the cases measured on this GPU (research
    §6.1).
