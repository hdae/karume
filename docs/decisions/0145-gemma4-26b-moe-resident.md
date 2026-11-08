# 0145: Gemma 4 26B A4B with every expert resident

- Status: proposed (2026-11-08 — the maintainer approved the plan; not implemented)
- Date: 2026-11-08
- Related: ADR [0059](0059-op-vocabulary-entry-doors.md) (vocabulary doors) / [0061](0061-index-oob-semantics.md) (index
  semantics) / [0068](0068-decode-exit-multi-output.md) (static-k `topk`) / [0069](0069-packed-w4-storage.md) (packed i4, f32
  scale) / [0082](0082-linear-gemv-decode.md) (the GEMV family of `linear`) / [0128](0128-default-preset-original-weights.md) (the
  q4_0 grid import) / [0135](0135-gated-delta-rule-scan.md) (a `karume::` custom op captured at export) /
  [0143](0143-f16-full-state-storage.md) (f16 full-layer KV)

## Context

Gemma 4 26B A4B is a mixture-of-experts model of the `gemma4` class (the class of E2B / E4B, not 12B's `gemma4_unified`). Every
one of its 30 layers runs a dense GeGLU MLP (intermediate 2112) **and** a block of 128 experts of which a router picks 8 per token
(expert intermediate 704). Upstream (transformers 5.14.1, `modeling_gemma4.py:1299-1461`), from one residual `r` per layer:

- dense branch: `post_feedforward_layernorm_1(mlp(pre_feedforward_layernorm(r)))`;
- router: RMSNorm without scale of `r` → × `router.scale` → × `hidden_size^-0.5` → Linear 2816→128 → softmax in f32 → top-8 →
  divided by the sum of the 8 → × `per_expert_scale[index]`;
- expert branch: `post_feedforward_layernorm_2(experts(pre_feedforward_layernorm_2(r)))`, with 3-D parameters
  `gate_up_proj [128, 1408, 2816]` (gate rows then up rows) and `down_proj [128, 2816, 704]`: `gelu_tanh(gate) · up`, then down,
  × the routing weight, summed over the 8;
- out = `r` + `post_feedforward_layernorm`(dense + expert), then × `layer_scalar`.

The three norms read the same `r` (a shared mean square, a fusion candidate). Attention is 12B's form: sliding layers 8 KV heads ×
256, full layers 5 / 11 / 17 / 23 / 29 with K = V (one projection, two slots) and 2 KV heads × 512; no PLE, no KV sharing.

Text parameters are 25.23 B, of which the experts are 22.84 B; 3.82 B are active per token. The source is Google's
`google/gemma-4-26B-A4B-it-qat-q4_0-unquantized` (two shards, with a vision tower this decision leaves out).

What is missing:

- `linear` takes a rank-2 weight only, so nothing reads one expert out of a stacked weight by an index held on the GPU.
- The tables that keep a weight compressed are rank-2 and per op (`WEIGHT_SLOTS` / `WEIGHT_CHANNEL_AXES` in `ops/names.ts`, the
  i4 eligibility of `runtime/plan.ts`, and their mirrors `I4_WEIGHT_OPS` / `_plan_fixed_weights` in the exporter's `emit.py`). A
  weight outside them is planned expanded to f32 on the host, which for the experts is about 91 GB.
- The exporter's `gemma4` path loads the checkpoint in F32 (about 101 GB, over the 61 GiB of the development machine). 12B's
  meta trace and row-block import do not, but they accept `gemma4_unified` only, read one file, and import rank-2 weights.
- The router's export meets three ops the handlers refuse: `softmax(dtype=f32)`, `sum(keepdim=True)` and
  `per_expert_scale[index]` (`aten.index.Tensor` with a runtime index). `topk` has no aten handler (deferred by ADR 0068).
- The i4 scale is f32 (ADR 0069 decision 3), and its dtype stops at the container: the IR storage, the bound scale, the byte
  counts and every i4 kernel assume f32. With f32 scales the text weights come to 14.72 GiB (router and norms included), against
  about 14.2–14.8 GiB that Chrome can use on the 16 GB RTX 5070 Ti (derived; the idle VRAM of that machine is not measured).

The measurement of 2026-11-08 (`outputs/diag/gemma4-26b-2026-11-08/`, not tracked) imported 12B's grid rule on every text weight
of the checkpoint, and on 12B's checkpoint for comparison:

| Weights | Groups off the grid | Groups reproduced bit for bit, f32 step / f16 step | Worst error in steps, f32 / f16 |
| ------- | ------------------- | ------------------------------------------------ | ------------------------------- |
| 26B experts (gate_up, down) | 0 of 713.7 M | 87.7% / 87.2% | 0.037 / 0.035 |
| 26B router.proj | 0 of 0.34 M | 87.1% / 86.4% | 0.035 / 0.035 |
| 26B dense MLP, attention, embedding | 0 of 74.5 M | 82–86% / 81–86% | 0.040 / 0.041 |
| 12B (shipped with f32 steps) | 0 of 372.1 M | 82% / 81% | 0.053 / 0.052 |

"Reproduced" means codes × step rounded to bf16 gives the checkpoint's bf16 value for all 32 elements. Every expert step lies in
f16's normal range (1.0e-3 to 1.3e-1); attention has steps down to 5.3e-5, below it.

## Decision

1. **Scope.** Model `26b` of the q4_0 QAT line, in the distribution `karume-gemma4-qat` beside E2B / E4B / 12B, text only, with
   every expert resident in VRAM (the constraint "MoE assumes every expert is resident" of [limitations](../limitations.md)
   holds). `speculative` is refused on 26B until its drafter lands. The vision tower, the drafter and an expert cache are later
   work. The distribution declares the `maxStorageBufferBindingSize` its largest weight needs (one layer's `gate_up`, 253.8 MB),
   so an adapter below it fails before the download.

2. **One new op, `expert_linear(x, index, weight) → y`** (Karume's own name: the extended composite layer of ADR 0059).
   - `weight` is `[E, out, in]`, `index` is `[T, k]` i32, `x` is `[T, in]` (shared by the k picks) or `[T, k, in]` (one row per
     pick), and `y[t, j] = weight[index[t, j]] · x[t (, j)]`, shape `[T, k, out]`.
   - The expert FFN is two of these with existing ops between them (`slice`, `gelu_tanh`, `mul`); the weighted sum over k is
     `mul` + `sum` along k in pick order. The op names no activation, so a SiLU model uses it too.
   - Accepted weight storages: f32 (fixtures, the CPU reference, cut-model tests) and the two i4 codecs of decision 3. Others are
     refused. The op enters every table of a weight slot on both sides (`WEIGHT_SLOTS` with slot 2, the i4 eligibility, emit's
     fixed weights for rank 3); its i8 row axis is not defined, because the op refuses i8.
   - **The index is in range by construction, checked at plan time** (the host-boundary rule of ADR 0061's addendum of
     2026-09-25). An `expert_linear` index MUST be the index output of a `topk` whose reduced dimension equals the weight's E;
     the planner refuses any other producer. `topk` returns indices in `[0, dim)` (ADR 0068), so no routing id reaches the host
     and no GPU NaN serves as a check. Out of range otherwise follows ADR 0061 (quiet NaN on the GPU, a throw in the CPU
     reference) as a broken invariant.
   - Entry condition ② of ADR 0059 applies: written with existing ops, the gathered weight is `[T, k, out, in]` — about 127 MB
     of temporary per layer for one decode token, orders of magnitude more for a prefill chunk. Entry condition ③ applies: every
     layer of 26B, and the routed experts of other MoE language models.
   - The sum over k runs in pick order (top-k order, ties to the smaller expert id — ADR 0068). Upstream's eager adds in
     expert-id order; its default `grouped_mm` and `batched_mm` sum in pick order. Parity is at the level of the numbers, not the
     bits (ADR [0005](0005-verification.md) addendum 2026-10-11).

3. **Storage: the grid import for every text linear, and an f16 step for `expert_linear` only.**
   - The experts, the router, the dense MLP, attention and the tied embedding are imported from the q4_0 grid as 12B is (ADR 0128
     addendum 2026-11-03); the router is on the grid too. A rank-3 weight imports as `[E · out, in]`, the same bytes.
   - **A sibling codec `int4-sym-g-f16`** (the bytes of `int4-sym-g`, an f16 scale) with its own layout, so the scale dtype
     travels in the codec name and every consumer that switches on the layout fails to compile until it decides. Only
     `expert_linear` accepts it; `linear`, the GEMV kernels, `embedding`, `conv1d`, the estimate and the tuner refuse it.
   - It meets ADR 0069 decision 3's conditions for f16 (an addendum to 0069 records the binding to one op): the step is the f32
     refit rounded to f16 once at emit; emit refuses a rounded step that is zero or not finite; the reference and the goldens use
     the rounded step (ADR 0006). The kernel reads it with `unpack2x16float`, so no `shader-f16` feature is needed.
   - The measurement above shows the f16 step as faithful to the checkpoint as the f32 one; it saves 1.43 GB of the experts' 2.85
     GB of f32 scales. E2B / E4B / 12B and their sha rows do not change.
   - The 26B export fails loudly when any parameter of rank ≥ 2 is not in the fixed (imported) weights, so a 3-D expert can never
     be written as F32.

4. **Kernel: the GEMV form, with the index read on the GPU.**
   - One thread owns one output column of one pick, as in the GEMV family of `linear` (ADR 0082), and reads `index[t, j]` from its
     buffer, so routing stays on the GPU and decode keeps one fence per run.
   - Its own numeric contract (there is no default path to be bit-identical to): per output, a sequential f32 sum over `in` by
     groups, each element decoded as `f32(code) · f32(step)`; step `g` of column `c` of expert `e` sits at
     `e · (out · in / G) + c · (in / G) + g`. MUST: the result does not depend on how rows are split across workgroups.
   - The T × k rows grid-stride (they can exceed `maxComputeWorkgroupsPerDimension`).
   - A prefill chunk first runs the same kernel over its T × k rows: correct and slow, because rows of different experts share no
     weight read. A grouped form for prefill (rows sorted by expert, deterministic, no atomics) is a performance item.

5. **Export.**
   - **Experts:** the recipe registers an implementation named `karume` in transformers' `ExpertsInterface` and sets
     `_experts_implementation` to it for the meta trace and the reference alike, so upstream's `forward` is not copied (as ADR
     0135 decision 5 swaps a function, not a class). It calls `karume::expert_linear`, whose eager body is upstream's per-expert
     math for each pick, dequantising only the picked experts, and whose fake keeps T symbolic. The handler narrows x (f32, rank
     2 or 3), index (i64 → i32, rank 2) and weight (rank 3).
   - **Router:** upstream's module. `softmax` with a `dtype` equal to its input's dtype is a plain `softmax` (another dtype is
     refused, in the handler); two normalize passes, which add no vocabulary, rewrite `sum(keepdim=True)` as `sum` + `reshape`
     and `aten.index.Tensor` with one runtime index on axis 0 of a rank-1 parameter or buffer as `embedding` over the table as
     `[E, 1]`, reshaped (other forms stay refused).
     `topk` gets its aten handler and `getitem` wiring, leaves `NON_EMITTABLE_OPS`, and maps its i64 indices to i32 (ADR 0009).
   - **The text path:** 12B's product wrapper and graph interface, soft inputs included (`soft_embeds` / `soft_mask`, unused
     until 26B's vision lands, so that work reuses this container instead of re-exporting it), on 12B's meta trace and
     row-block import, generalised to the `gemma4` class, to sharded checkpoints, to rank-3 weights, and to the vision keys this
     decision ignores. The container and the reference share one grid import. The K = V wiring check of 12B runs on its graph.
     The front end (tokenizer, chat template, generation settings) is pinned from `google/gemma-4-26B-A4B-it`, as 12B pins
     12B-it; the chat template is 12B's byte for byte, so the dialect is `gemma4-unified`.

6. **Checks.**
   - This machine's GPU (12 GB) cannot hold the model, so its tests run a model cut after a few layers, against layer goldens from
     upstream computed a layer at a time.
   - Goldens record which upstream experts implementation ran (as ADR 0135 requires). Routing is compared first: ids must match,
     and a row whose 8th / 9th probabilities are closer than a floor is reported apart instead of inside a numeric tolerance (a
     near-tie can swap one whole expert).
   - The full model runs only in Chrome on the RTX 5070 Ti: greedy text against the golden, and the maintainer reads dedicated and
     shared GPU memory in the Task Manager. Dawn on D3D12 pages heaps over PCIe past its budget instead of failing an allocation
     (from Dawn's code — not observed), so a model that does not fit gets slow instead of stopping, and `capacity: "fit"` and the
     headroom check, which rest on a failing trial allocation, may report a fit there. 26B keeps the manifest capacity 4096 and
     does not rely on them on D3D12 (a limitation until measured).
   - `/kernel-check` gets a case for `expert_linear`. The run can dump its routing ids, as data for the cache design.

## Consequences

- The op vocabulary gains `expert_linear` (contract set: TS `OP_CONTRACTS`, `karume/ops.py`, shapes on both sides, fixtures, the
  CPU reference, a golden, the weight-slot tables); the exporter gains the `topk`, `softmax`-dtype, keepdim `sum` and
  index-on-a-table handlers.
- The codec ledger gains `int4-sym-g-f16`, bound to one op.
- The 26B text weights are about 14.3 GB (13.4 GiB; experts at 0.5625 bytes per parameter, the rest at 0.625), against
  7.4 GB for 12B; a browser needs that much CacheStorage quota for one origin.
- 26B's dense parts start without the RMS→add fusion and the parallel-GEMV rows, whose tables list measured widths only
  (2816 is not among them) — a performance item.

## Alternatives considered

- **The whole expert block as one op** (router to weighted sum): one node to capture, but it fixes the activation and the
  combination inside the op, so it serves one family; the decomposed form costs a few small dispatches more per layer.
- **`batched_mm` or `grouped_mm` of transformers through the exporter**: `batched_mm` exports, but its gather is the materialised
  `[T·k, out, in]` weight that decision 2 avoids; `grouped_mm` exports as a transformers custom op with `sort` / `histc` /
  `index_put_` around it, none of which the runtime has.
- **An f16 scale on the existing codec, read by every i4 op**: changes the numbers and sha rows of E2B / E4B / 12B for no need of
  theirs, puts attention's smallest steps below f16's normal range, and lets a consumer read f16 bytes as f32 unless every one is
  checked by hand.
- **f32 scale everywhere**: leaves no margin on the 16 GB card, where running out does not fail but pages.
- **A host check of the routing ids**: a readback per run, against one fence per run; the plan-time producer check gives the same
  guarantee.

## Open points

1. The grouped prefill form and its speed (decision 4).
2. The measured idle VRAM of the RTX 5070 Ti machine in Chrome, whether the model with capacity 4096 stays in dedicated memory,
   and what `capacity: "fit"` reports there.
3. The reference's RAM peak and time per forward with only the picked experts dequantised (decision 5), against 61 GiB.
4. The drafter (`…-unquantized-assistant`, 0.42 B, shares the target's KV), vision (an encoder tower of 0.57 B) and the expert cache.
