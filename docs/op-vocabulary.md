# Op vocabulary ledger — Core ATen 160 (carried over from the prototype)

Origin: carried over from `docs/op-vocabulary.md` of the preceding experimental project (hereafter
the prototype), as of 2026-08-01. At the user's instruction, Karume continues the "study of the ops
we need" that was in progress in the prototype. From here on this file is the source of truth.

NOTE: The individual lists of the 25 and the 15 are included in this document. No per-op assignment
of the 75 / 38 / 7 boxes exists anywhere; it is fixed by export measurements from M1-P2 onward
(listing every unsupported op).

## Fixing the population (conclusions carried over)

- The population is **the 160 ops of Core ATen IR** (`torch.Tag.core`). Out of 2879 aten overloads,
  1302 function names and 943 decomp table entries, torch.export's `run_decompositions` reduces
  everything down to these. **This is an upper bound**, not a TODO list. Filling the 160 is not a
  goal in itself.
- Conversely, for 12 ops that would hurt if decomposed too far ("12 ops whose decomposition is
  forbidden" below) the exporter stops the decomposition — because one fused kernel is faster. This
  asymmetry is the correct state, and the prototype's `attention` / `rms_norm` / `silu` being absent
  from the 160 has the same reason (`attention` entered Karume's vocabulary in ADR 0023 —
  preservation is per target).

## Five classes (assignment of all 160)

| Class                   | Count | Contents                                                                    |
| ----------------------- | ----- | --------------------------------------------------------------------------- |
| Need not be implemented | 25    | backward / random (host responsibility) / shape meta / 3D ops               |
| Required primitives     | 75    | ≈45 implemented in the prototype. The rest is the bulk of the work          |
| Covered by composition  | 38    | 0 new kernels. Absorbed by decomposition / normalisation into existing ops  |
| Nice to have            | 7     | Low priority                                                                |
| Hard / slow             | 15    | The three groups below (each presumes its own ADR)                          |

The implementation target is 120 (= 75 + 38 + 7).

### Breakdown of the 25 that need not be implemented

- **Training only (7)**: the `*_backward` ops. They do not appear in a forward-only export.
- **Random (4)**: `rand` / `randn` / `randperm` / `native_dropout`. For diffusion, **generating the
  initial noise on the host is correct** (it is the calling application's responsibility, outside
  the inference runtime's).
- **Shape meta / aliases (9)**: `sym_size` / `sym_numel` / `sym_stride` / `sym_storage_offset` emit
  no node, because the exporter evaluates SymInt expressions and folds them into symbolic dimensions
  (`T`, `8T+2` and so on). `alias` is handled as a skipped vocabulary entry on the prototype side.
  `_local_scalar_dense` / `as_strided` / `empty` / `empty_strided` are outside the accepted set —
  failing loudly when they appear is correct (as the cross-cutting invariant "never approximate
  silently" says).
- **3D / video only (5)**: can be added in the same shape once the 2D versions are settled.

## The three groups of the 15 hard / slow ops — listed one by one

1. **Scatter (scatter_add / index_put / scatter / scatter_reduce / masked_scatter / col2im)** —
   WGSL has no `atomic<f32>`, and a CAS implementation makes the summation order nondeterministic,
   silently breaking the determinism of codegen / execution (same input → same output). When the
   indices are static at export time, **inverting into a gather** (making the inverse mapping a
   constant) fits better.
2. **Data-dependent output shapes (nonzero / bool-mask index / _embedding_bag)** — the executor
   assumes static shapes and fixes the output shape of every node before execution, and symbolic
   dimensions are bound only from the element counts of inputs. **No path exists that reads GPU
   results back to decide a shape.** Supporting them needs either "fixed upper-bound shape + mask"
   or a design ruling on the execution model itself.
3. **Value-dependent reordering and others (sort / topk / grid_sampler_2d / _cdist_forward /
   _pdist_forward / _fft_r2c / _fft_c2r)** — on hold. `topk` can be replaced by `argmax` when k=1.
   **Correction (2026-08-17)**: `topk` is **off hold** (ADR 0068 decision 3 — static-k, two
   outputs; see the `topk` NOTE below). Two conditions unlocked it: ① k is a **plan-time constant**
   in attrs, so the output shape is static (only the contents depend on values) ② "lane-local top-k
   → tournament merge" avoids an argsort over the whole vocabulary. `sort` (reordering the whole
   vocabulary) and `grid_sampler_2d` / the fft ops remain on hold.

NOTE: The individual lists add up to 16 (6+3+7), one more than the box total of 15. The mismatch
comes from the prototype's original (the original does not record which op is double-counted
either). It will be resolved from M1-P2 on, when the per-op assignment is fixed by measurement.

## Design findings carried over

- **The biggest hole = the value max-reduce family**: amax / amin / max / min / argmax / argmin were
  entirely missing. Without them safe-softmax, max-pool, logsumexp and greedy decoding cannot be
  built, and Transformers models cannot be handled generically. The implementation is a ≈20-line
  copy of the row kernel for sum, so it is done first. **Correction (2026-08-17)**: "a ≈20-line
  copy" held for the two that fold values only (amax / amin), but **not for argmax** — because it
  carries (value, index) pairs, its identity, comparison expression and scratch are different, and
  riding on the existing reduce family would move the keys of the default path (the implementation
  is a separate family = src/kernels/argmax.ts; see the argmax NOTE below).
- **12 ops whose decomposition is forbidden (fusion kept)**: linear, layer_norm, rms_norm, softmax,
  gelu, leaky_relu, conv1d, conv2d, conv_transpose1d, embedding, masked_fill,
  **scaled_dot_product_attention** (`leaky_relu` took the count from 9 to 10 in ADR 0015. Its
  decomposition `gt_scalar + mul + where` swells the intermediate buffers 1.5–2×. `rms_norm` took it
  from 10 to 11 in ADR 0017 — here **preservation alone is not enough**, and it is supplied by two
  routes, together with a folding pass that picks up the hand-written decomposition.
  `scaled_dot_product_attention` took it from 11 to 12 in ADR 0023).
  **Only SDPA (the 12th) is not on the default preservation list** (`PRESERVED_OP_PREFIXES` holds
  the 12 above without SDPA (the earlier 11, `leaky_relu` among them) plus `conv3d` of the
  extended composite layer(ADR 0118) and `group_norm` of the Core ATen layer (ADR NNNN — the NOTE
  of 2026-10-27 below), 13 in all) —
  SDPA can express mask / causal through its arguments, so preserving it globally would make shapes
  outside the contract (a bool mask passed by kwargs, `[B,1,M,N]` and so on) hit the fail-loudly of
  `_h_attention`, and the export would no longer work. Only an additive f32 `[1,1,M,N]` mask entered
  the vocabulary, with the revision of 2026-08-11 (ADR 0023 addendum). Preservation is a
  **per-target opt-in** through `curated_decompositions(preserved=…)`
  (`PRESERVED_OP_PREFIXES_WITH_ATTENTION` — first Anima's transformer / vae_decoder and
  EmbeddingGemma; the current users are the recipes under `tools/export-recipes/` that pass it, the
  Gemma 4, MiniCPM5 and Wan recipes among them).
  The exporter stops the decomposition — the judgement is that one fused kernel is faster than the
  several dispatches after decomposition, and this asymmetry is the correct state. The prototype's
  attention / rms_norm / silu being absent from Core ATen 160 has the same root. The
  no-decomposition list maps directly onto speed, and adding to it later means redesigning the graph
  layer.
- **Counter-examples to "it can be decomposed"** (found by the prototype's adversarial verification;
  a decomposition proposal requires a check against torch):

  | Claim                                          | Verdict     | Reason                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
  | ---------------------------------------------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
  | `log` = `log1p(x−1)`                           | Wrong       | `log1p` is not a WGSL built-in and generates `log(1+a)`, so the rounding of `x−1` comes straight back. At x=1e-6 it is off by about 6%. A codegen branch for unary log already exists; the executor needs only one more branch                                                                                                                                                                                                                                                                                                                                                                                         |
  | `_log_softmax` = `log(softmax(x))`             | Wrong       | Hits the error above in the y≪1 region. Underflowed elements become `log(0)=−inf`, disagreeing with torch's finite values. **It erases the op's reason to exist**                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
  | `pow` is a one-line WGSL built-in              | Wrong       | WGSL `pow` is **undefined for a negative base**. Disagrees with `torch.pow(-2,3) = -8`. Integer exponents go to repeated `mul` (implemented)                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
  | `maximum`/`amax` expose the internal `max`     | Wrong       | "WGSL `max` does not guarantee NaN propagation", so `leaky_relu` uses the select form (ADR 0015, implemented in M1-P3 wave 3). **Measured (2026-08-02)**: even the select form is folded by the compiler into the max idiom, and the driver's `max` swallows NaN (`clamp(NaN,-1,1) = -1` / `clamp_min(NaN,0) = 0` / `relu(NaN) = 0`. A comparison alone yields false, as the spec says). **Fixed at the root (2026-08-03)**: `clamp` / `clamp_min` / `relu` / `amax` / `amin` propagate NaN through an outer shell of **bit-pattern NaN detection** (`(bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u`) — an integer comparison is not subject to that folding. Only `leaky_relu` keeps the select form, because x appears in both branches |
  | `mean`/`var`/`group_norm` = plain `sum`        | Wrong       | The reference implementation of sum scans sequentially, one thread = one row. group_norm has the opposite shape — 32 rows, each millions of elements long — so 32 threads loop millions of times. **A two-stage reduction works**                                                                                                                                                                                                                                                                                                                                                                                   |
  | `var` = `sum((x−mean)²)/N`                     | Wrong       | `torch.var` defaults to `correction=1` (N−1). Division by N is only for the internal variance of the norm ops                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
  | `group_norm` → delegate to `layer_norm`        | Partly      | The normalisation matches but **the affine does not** — layer_norm's weight length assumes the last dimension, so huge resolution-dependent ones/zeros would be needed. A separate two-pass `[1,C,1,1]` broadcast is required                                                                                                                                                                                                                                                                                                                                                                                         |
  | tensor comparison = `sub` + `*_scalar` compare | Wrong       | With `a=b=+inf`, `a−b=NaN` and `eq` is false (torch says true). This really occurs in attention's −inf guard                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
  | `floor`/`ceil`/`round`/`trunc` = cast round trip | Wrong     | WGSL has built-ins, so it is **a one-line codegen addition**. The cast route breaks at `\|x\| ≥ 2^31`. The rounding rule is ties-to-even, matching torch                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
  | `fmod` needs hand-written code                 | Wrong       | WGSL `%` is `e1 − e2·trunc(e1/e2)` = `torch.fmod`. **`remainder`** is the one that needs a sign correction for floor division. Both are sealed off for integer types                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
  | `expm1` = `exp(x)−1` is exact                  | Wrong       | For x≲1e-8, `exp(x)` rounds to 1.0 and the result is 0. torch returns x, so the relative error is 100%                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
  | non-unit-axis `repeat` = unsqueeze+expand+view | Wrong       | With a rank-4 input the intermediate is rank 5, exceeding the rank limit (4) of the strided representation. Usable only up to rank≤3                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
  | `select_scatter`/`slice_scatter` = `cat`       | Unconfirmed | At the boundaries (index 0 / end) a zero-length slice appears, and since WebGPU storage bindings require size>0 it may fail (not measured). A branch that drops empty operands is required                                                                                                                                                                                                                                                                                                                                                                                                                             |
  | 6-step expansion of `batch_norm`               | Holds       | Numerically equivalent. But without elementwise fusion, one node = one dispatch + one full-size intermediate. For a 1024px VAE, `[1,128,1024,1024]` = 537MB, so capacity becomes the problem                                                                                                                                                                                                                                                                                                                                                                                                                          |

  NOTE (2026-08-11): The **−inf guard of attention** that the "tensor comparison" row refers to was
  settled with **one op, `safe_softmax`**, while the proposal to turn the guard subtree into ops
  (`eq(−inf)` and the like in the vocabulary) stayed rejected — ADR
  [0044](decisions/0044-runtime-attention-mask.md). Only the guard's **semantics** (an all −inf row
  → 0) is held, as a variant of `softmax`, so no comparison op that handles non-finite values enters
  the vocabulary.

## Things better not done

- **Do not build a generic reduce framework first.** For amax, a copy of about 20 lines of the sum
  row kernel is the right answer. With only two implementations (sum / cumsum) to abstract from,
  generalising first means abstracting before the shapes have settled, and the rework is large.
- **Do not port torch's decomposition table.** It has 943 entries, but the decompositions actually
  written cover 38 ops and most are a few lines. Add them one by one, at the same granularity as the
  prototype's precedents (select / split_with_sizes / the equivalent of pow(2)).
- **Do not add ops absent from Core ATen "for symmetry".**
- **Do not make filling all 160 a goal.**
- **Do not casually add `scatter_add` with an `atomic<f32>` CAS loop** — the summation order becomes
  nondeterministic and silently breaks "same input → same output".

## Procedure for adding an op (entry-door model — named layers)

DECIDED: [0059](decisions/0059-op-vocabulary-entry-doors.md) (entry-door model, layer numbers
retired. The old definitions of layers 0 to 3 are in [0043](decisions/0043-op-addition-layers.md) —
a historical record) + [0064](decisions/0064-entry-door-generality-axes.md) (adds two axes: semantic
scope and execution-model invariants). When an unsupported op shows up in
`UnsupportedAtenOpsError`, drop it to the **cheapest door**, trying from the top down. At every door,
what actually executes is a kernel.
**The axis that decides whether an ADR is needed is one test, "Core ATen membership"** (the entry
test = a measurement of `torch.Tag.core in op.tags`) **+ one exception** (an addition that breaks an
execution-model invariant needs an ADR even in core — 0064 axis B). Atomic / composite is not an
axis that splits doors but the type of proof inside the ADR (proof that it cannot be written /
measured grounds for keeping it).
`UnsupportedAtenOpsError` reports, for each op, this entry test (`[Core ATen]` / `[outside Core ATen]` /
`[outside aten]` — the labels as the exporter prints them today),
the originating module (the innermost class and the first path of `nn_module_stack`) and the
occurrence count (the test is taken from the `Tag.core` of the torch in use; no list is kept).

| Door (old number)                            | Location                                                                                                  | Procedure / commitment                                                                                                       |
| -------------------------------------------- | --------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **Export-eliminated layer** (old 0)          | `FOLDABLE_OPS` constant folding / the decomposition passes of normalize / convert                         | None (never reaches the runtime). The top-priority way out                                                                          |
| **Core ATen layer** (old 1 + part of old 2)  | IR vocabulary. `Tag.core` ops and their attr variants under separate op names (gelu_tanh)                 | **Ledger NOTE only (no ADR)** — the NOTE records the measurements that rejected steps 1/2. Ceiling = 160 + the attr surface   |
| **Extended atomic layer** (old 1')           | IR vocabulary. Non-core, and **cannot** be written as an exact equivalent by composition in the vocabulary | **ADR required** = proof that it cannot be avoided (the deform_conv2d pattern). The requester (IR mechanism / model) does not split doors |
| **Extended composite layer** (rest of old 2) | IR vocabulary. Non-core, can be written but is **kept** for performance or meaning                        | **ADR required** = a measurement for at least one of entry conditions ① to ④ (the gru_scan pattern)                         |
| **Fusion layer** (old 3)                     | Adjacent matchers in `fusion.ts` (the current inventory lives in `FUSION_RULES` and the fusion tests — 0064) | None (free to withdraw). When the exact match misses, things get slower silently — **observation points are required**     |

The test (from the top down):

1. **Is the value an export-time constant?** (the leaves of its dependencies are only arange / lifted
   constants) → export-eliminated layer, FOLDABLE. Conditions: it does not consume a symbol in a
   value position, does not refer to a position on a symbolic axis, and dynamic views are limited to
   adding or removing unit axes (structural check — the 2026-08-13 / 2026-09-10 / 2026-10-10
   addenda of ADR 0010); it passes the two-point evaluation of `_check_prefix_commutes`; only f32/i32
   can become an initializer.
2. **Is it an exact equivalent of a composition of existing primitives?** → the decomposition pass
   of the export-eliminated layer. **A check against torch is required** (as the counter-examples
   above show, many decompositions break on rounding, NaN or boundaries). Whether to escape by
   decomposition or to add an op is judged by a **precedent guide**, not a binding threshold:
   rewrites whose composition barely adds intermediates or FLOPs (reciprocal → division form, the
   softplus decomposition(the guard against intermediate breakdown is that torch's decomposition
   carries its threshold=20 where into the IR as is — select does not propagate NaN from the
   unselected arm), the real-valued RoPE type) → decomposition is the default / when intermediates
   grow, **there is already a precedent for keeping the op at 1.5–2×** (leaky_relu — put on the
   preservation list in ADR 0015) / compositions that grow by orders of magnitude
   (upsample_bilinear2d FLOPs ×184, batch_norm 537MB — the counter-examples above) are **grounds to
   reject the decomposition**, and whether to add the op is decided by demand. Either way, the next
   door test (whether an ADR is needed) does not change.
3. **Measure Core ATen membership** — `uv run python -c "import torch;
   print(torch.Tag.core in torch.ops.aten.<name>.<overload>.tags)"` (in tools/exporter).
   Core (+ its attr variants under separate op names) → **Core ATen layer** = ledger NOTE only. One
   contract set (`OP_CONTRACTS` + `karume/ops.py` + `shapes.py` + fixtures/op-contracts.json + CPU
   reference + golden COVERAGE). **Exception (0064 axis B)**: an addition that breaks an
   execution-model invariant (single output(superseded by ADR 0068 decision 1 on 2026-08-17 —
   node-level multi-output is allowed) / static shapes / full-write / 4-byte storage + three semantic
   dtypes / full-range write of resident tensors) **needs an ADR** even in core — stating the impact
   (planner / executor / recipe / IR …) up front is a required section (the two outputs of `topk` /
   `sort` / `argmax` are the typical case — **`topk` has gone through this route**: ADR
   [0068](decisions/0068-decode-exit-multi-output.md) stated the impact of multi-output up front on
   eight surfaces, and it moved in with decision 3. `argmax` has a single output, so this exception
   does not apply to it and a ledger NOTE alone suffices).
4. **Non-core** (non-core aten, torchvision, Karume's own names) → together with an ADR:
   **cannot be written** as an exact equivalent by composition in the vocabulary → **extended atomic
   layer** (ADR = proof that it cannot be avoided).
   **Can be written but should be kept** → **extended composite layer** (ADR = a measurement for one
   of the entry conditions: ① runtime reconstruction from the decomposed form is fragile / semantic
   information is lost(attention's finite mask value, rms_norm's weight length, gru_scan's runtime
   length axis) ② materialising the intermediate is unacceptable in capacity or bandwidth ③ the
   recurrence rate justifies a permanent commitment ④ attr variants are split into separate ops to
   keep the empty-attrs contract (ADR 0012)(when non-core)). **A non-core op (decided by the
   measured `Tag.core` — not by the spelling of the IR name; aliases derived from core ops such as
   cast / ge_scalar are in the Core ATen layer) appears in the IR only when Karume chose it on
   purpose** (kept or invented) — the choice comes with a record; that is how the door works
   (ADR 0059 decision 5). ADRs / NOTEs **write the semantic surface and the implemented subset
   separately**, and when an observed subset is fixed as the semantics, they leave one line saying
   so + the condition for generalising (0064 axis A).
   When a missing mechanism (bool attrs and the like) is covered by splitting the op name, this is
   stated together with the condition for merging.
5. **A correctly working decomposed form already exists and only performance is to be recovered** →
   a fusion-layer rule. Having observation points (`lastRunFusions` /
   `packages/e2e/tests/e2e_*_fusion_counts_test.ts`) is the entry condition (ADR 0040).

"Things better not done" (the section above) takes precedence over every door. The no-decomposition
list is an axis independent of the doors (the exporter's folding policy — ADR 0043 addendum decision
3, 0059 decision 7). The dated NOTEs below are **snapshots** (they keep the old numbers — read the
mapping from the (old number) column of the door table above).

NOTE: 2026-08-11 added `gelu_tanh` to layer 2 (the first application of ADR 0043 — Gemma's
`approximate="tanh"`. Its use is confirmed in the configs of EmbeddingGemma / Gemma 4 E2B).

NOTE: 2026-08-11 added `sin` (f32 unary) to **layer 1**. Grounds = DACVAE's Snake activation
`x + (α+1e-9)⁻¹·sin²(αx)` (the codec of Irodori-TTS v4). It does not stop at step 1 because
**x is a runtime value** — constant RoPE tables are folded by `aten.sin.default` in `FOLDABLE_OPS`
as before, and both routes hold at once (removing the folding would add runtime dispatches). The
composition of step 2 is not possible either (no expression represents `sin` exactly with existing
primitives). `aten.sin.default` is inside Core ATen, so layer 1. **`cos` is not added** — no
measured case takes `cos` of a runtime value, and "no additions for symmetry" takes precedence. No
ADR is written (layer 1 is ledger only).

NOTE: 2026-08-12 the decision axis was unified into **atomic / composite** (ADR 0043 addendum).
Boundaries fixed: `gelu` is atomic (no erf → layer 1 — its membership in the no-decomposition list is
unchanged; no-decomposition is an axis independent of the layers) / `leaky_relu` is composite
(layer 2, grounds ADR 0015) / `sym_prefix_slice` is the first case of **layer 1'**.
Ops already introduced are not reassigned (the layers govern only the procedure for future
additions).

NOTE: 2026-08-13 added `upsample_bilinear2d` to **layer 1**. Grounds = `F.interpolate(...,
size=(H,W), mode="bilinear", align_corners=True)`, used in common by BiRefNet_HR / Lucida / Depth
Anything V2 (22 in the inference graph of BiRefNet_HR alone; 0 uses of `scale_factor` on the active
path; the scale factors are non-integer). It does not stop at step 1 because **size and scale are
static but the values are runtime values**. The composition of step 2 **can be written but does not
hold up** — ① the matmul of two constant matrices (separable form) is exactly equivalent but **about
184× the FLOPs of a direct kernel** (at the last stage of BiRefNet_HR,
`[1,192,512,512] → [1,192,2048,2048]`: 1.03 TFLOP against 5.6 GFLOP) ② the gather form cannot
broadcast the index under the contract of `gather` (the leading dimensions of index match src
exactly), and needs an index tensor of **805MB–3.22GB** (the same shape as the `batch_norm` 537MB row
of the counter-examples, one order of magnitude up).
`aten.upsample_bilinear2d.vec` is inside Core ATen, so layer 1 = **ledger only, no ADR**
(the interim handling of the layer classification is in the known-issues section that the NOTE
below points to, user ruling 2026-08-13
〔addendum 2026-08-14: the interim handling was resolved by the entry-door model, ADR 0059 — this
op's outcome, the Core ATen layer, is unchanged〕).
**`align_corners=False` is not accepted** — both the coordinate formula (`scale·(i+0.5) − 0.5`) and
the edge handling differ, and not creating a field for it in attrs is itself the fail-loudly (the
discipline of ADR 0023 decision 4). When demand appears
(SAM / RMBG-1.4 / IS-Net / MODNet), it is added as a **separate op**, the same way as `gelu` /
`gelu_tanh`. **Correction (2026-08-14)**: the HF port of Depth Anything V2 that was on this list
**is not a source of demand** — measured during the port, all five `upsample_bilinear2d` in the
graph are `align_corners=True`, and the residual resize branch of the fusion layers, which
uses `False`, was structurally unreachable. `upsample_nearest2d` is not added
(Anima's nearest-exact ×2 is already covered by the layer-3 fusion rule `upsample2x`, and "no
additions for symmetry" takes precedence).

NOTE: 2026-08-13 added `deform_conv2d` (DCNv2) to **layer 1'**. ADR
[0055](decisions/0055-deform-conv2d.md) is the source of truth. Grounds = the whole BiRefNet family
(8 official + Lucida) uses `dec_att='ASPPDeformable'`, calling `torchvision.ops.deform_conv2d`
**20 times** per forward. It does not stop at step 1 because the offset is the **runtime output** of
a trained conv (and clamping is disabled). The composition of step 2 is not possible either — `floor`
is not in the vocabulary (`cast` truncates), and `gather` is fixed to the last dimension and cannot
express a value-dependent gather over two axes. `torchvision::deform_conv2d` is **outside Core ATen**
(measured: it also remains one node after `run_decompositions(curated_decompositions())`), so under
the interim handling below (an atomic op outside Core ATen = layer 1' 〔addendum 2026-08-14: the
current door = extended atomic layer, ADR 0059 — outcome unchanged〕) **an ADR is required**. The
rejected alternative is a decomposition via `grid_sampler_2d`
(exactly equivalent, with the procedural advantage of being inside Core ATen, but its intermediate is
im2col itself, **3.29GB at k=7 / 512²** — on the level of the 3.62GB for which ADR 0024 rejected
im2col). **The hold on `grid_sampler_2d` is not lifted** (the box of group 3 in the section on the 15
hard / slow ops stays as it is — it is decided independently when demand appears).

NOTE: 2026-08-14 added `gru_scan` / `gru_scan_reverse` to **layer 2**. ADR
[0056](decisions/0056-gru-scan.md) is the source of truth. Grounds = in vowel-detector (a 2-layer
BiGRU), decomposition **fully unrolls** `aten.gru.input` **along time** (874/1,714/8,434
call_function nodes at T10 = 20/40/200), and `torch.export` specialises the length axis — through
this op it becomes **19 nodes independent of T**. It does not stop at step 2 because "**it can be
written, but semantic information is lost**": the decomposed form is written as an **exact
equivalent** with existing ops of the vocabulary (= composite, not atomic), but recovering "this is a
GRU" with a layer-3 matcher from the node sequence unrolled T times does not work, and what is lost
is the very fact that "**the time axis is decided at run time**". It meets entry conditions ①
(semantic information is lost) and ③ (recurrence rate) of layer 2. **The input-side GEMM is not
included** (left to the caller's `linear` — the existing f16 / i8 storage applies as is, and the
target of the bit-identity proof shrinks to the hidden side only). **The reverse direction is a
separate op name, not a bool in attrs** (the same approach as `gelu` / `gelu_tanh` — every existing
attr validation function explicitly rejects bools, and accepting one would need a new validation
mechanism). **LSTM is not added** ("no additions for symmetry" — the points for an extension are in
ADR 0056 decision 8).

NOTE: 2026-08-17 added `argmax` to the **Core ATen layer**. The contract's source of truth is ADR
[0068](decisions/0068-decode-exit-multi-output.md) decision 2 (it sits in the same ADR as the
unlocking of multi-output, but **argmax itself has a single output**, so it does not fall under the
exception of 0064 axis B(an addition that breaks an execution-model invariant) — as a door, a ledger
NOTE alone suffices). Grounds = the exit of greedy decoding. Reading the logits of the whole
vocabulary back to the host is already a minority form among the reference implementations (survey
§2), and putting greedy on the GPU side needs this op.
Measurement of the entry test (2026-08-17 / torch 2.13, run in tools/exporter):

```sh
uv run python -c "import torch; print(torch.Tag.core in torch.ops.aten.argmax.default.tags)"
# → True
```

It does not stop at step 1 because **the logits are runtime values**. The composition of step 2 is
not possible either — the vocabulary has neither "a reduction that folds value and index together"
nor runtime index generation equivalent to `arange`, and an attempt with `amax` + `ge` + `where`
and the like becomes circular: ① in a row where several elements equal the maximum, several indices
remain, and one more reduction stage that picks "the smallest index" is needed ② that stage is
argmax itself (no index-min reduce exists in the vocabulary).
A counting form with `sum` does not give an index either, since a row sum of bools returns only a
count.

The contract is **last dimension fixed, empty attrs, rank preserved** (no `keepdim` field is created
— the absence of the field is itself the declaration that "a rank-reducing form is not in the
vocabulary"; the discipline of ADR 0023 decision 4). **It is not put in the reduce family** — the
attrs (none), the output dtype (i32), the rank handling (preserved) and the kernel family (identity is
−inf and it carries (value, index) pairs) differ on all four counts, and making it ride along would
move the key versions of the existing three and move the WAV / PNG checks wholesale. Three fixed
behaviours, measured to follow torch (same as above): **tie-break = smallest index** /
**NaN is the maximum** (smallest index if several — the side consistent within the family with the
NaN propagation of `amax`) / **an all −inf row gives index 0**. **`argmin` is not added** ("no
additions for symmetry" — greedy does not need it).
`max` / `min` (two outputs: value and index) are still not implemented either, but **the
multi-output mechanism has landed** (the `topk` NOTE below), so the material for the decision is
there — whether to add them waits until a measured need appears.

NOTE: 2026-08-17 added `topk` to the **Core ATen layer**. The contract's source of truth is ADR
[0068](decisions/0068-decode-exit-multi-output.md) decision 3 (**the first occupant of node-level
multi-output** — it breaks the execution-model invariant "single output", so it falls under the
exception of 0064 axis B, on the **ADR required** side. The impact stated up front is in ADR 0068
decision 1). Grounds = the exit of top-k sampling (temperature, top-p and random numbers stay on the
host, so the GPU side only needs to go as far as topk — decision 4). Measurement of the entry test
(2026-08-17 / torch 2.13, run in tools/exporter):

```sh
uv run python -c "import torch; print(torch.Tag.core in torch.ops.aten.topk.default.tags)"
# → True (not in the core decomposition table either = it remains one node after the curated decomp)
```

It does not stop at step 1 because **the logits are runtime values**. The composition of step 2 is
not possible either — the vocabulary has no "reduction that folds value and index together", and an
attempt by repeating `argmax` ① needs a mechanism to exclude the chosen element and find the next
(a runtime mask update equivalent to masked_fill) ② that repetition is k row reduces = reading the
row k times, exactly the high-cost side the ADR avoids.

The contract is **last dimension fixed, attrs `k` required to declare (static-k), two outputs**
(values f32 in descending order + indices i32, both `[…, k]`). The accepted range is
**`1 ≤ k ≤ last dimension`**; k=0 (which torch accepts), k > last dimension and a symbolic k all
fail loudly. No `largest` / `sorted` field is created (only the one form, the largest side in
descending order — the same discipline as argmax, which got no `keepdim` field). The
**implementation limit** is fixed statically by the device limit of workgroup storage (32 lanes per
row × `8·W·(k+1)` bytes → with the WebGPU default of 16384, **k ≤ 63**); beyond it, it fails with
the limit value instead of falling back.

Fixed behaviour: **tie-break = smallest index** (the same predicate family as argmax) / **NaN is the
maximum** (smallest index if several) / an all −inf row also gives k entries from the smallest index.
**The value column matches torch bit for bit** (descending order, same multiplicity), but **the index
column is Karume's own specification, not torch-conformant**, which is where it differs from argmax —
measured (2026-08-17 / torch 2.13.0+cpu), torch's `topk` does not guarantee the order of equal
elements: `topk([5,5,5,5], 1)` returns index **2** while `argmax([5,5,5,5])` returns **0** (a
self-contradiction within the same repository. In a 7-row tie table, the 3 rows with k=1 disagree).
Karume specifies the smallest index, in line with argmax, and a check verifies **that topk with k=1
matches argmax**.

**`sort` is not added** (reordering the whole vocabulary cannot be narrowed to a width of k, so the
reason for the hold remains as it is).
**There is no aten handler on the exporter side** — the tuple meta of a multi-output aten op and the
slot wiring of `operator.getitem` are a new mechanism, deferred until sampling is actually needed
(ADR 0068 addendum). What the measurement stops at is **`operator.getitem`**, not
`aten.topk.default`, so adding only a handler does not open the way.
Today topk nodes appear only from **hand-written IR on the runtime side / a future decode graph
script**.

NOTE: 2026-08-18 added `state_append` to the **extended atomic layer** (outside Core ATen), and added
the **states form** to `attention` (an optional `states` field — a contract extension under the same
op name, where the presence of the field identifies the form).
The contract's source of truth is ADR [0067](decisions/0067-autoregressive-attention-vocabulary.md)
decisions 4 / 5.
`state_append` is an effect op with **no counterpart in aten** (the first occupant with **zero
outputs** — the list form of ADR 0068 decision 1 accommodates the zero-output case), and in principle
never comes out of the torch.export path
(`NON_EMITTABLE_OPS` — emitting it is the job of the decode graph script). Why it is not built into
attention, the `window` attrs and the effect ordering by the order of the nodes array are in ADR
decisions 5 / 5b.

NOTE: 2026-09-08 added the **readonly form** to the states form of `attention` (optional attrs
`readonly` — `true` only). The contract's source of truth is ADR
[0096](decisions/0096-speculative-decoding.md) stage 2 §1.2. **Neither op names nor the vocabulary
grow** (it is the second step of the same "the presence of a field identifies the form" extension as
the states form), but it differs from the existing two forms in one point only, that `ins` is
**exactly one q**: it has no k / v of the current step, and reads only the rows `[P−min(P,W), P)` of
the borrowed slot (`[0, P)` for full). q is `[B,H,1,D]` with **M = 1 MUST**, and the referenced slots
are **external slots only** (`states[].external` — the writer sits on the lending context's side).
It does not come out of the torch.export path (the decomposition of `scaled_dot_product_attention`
has no form "without the current step's k/v") — emitting it is the job of the exporter's surgery
(`states.to_external_states_form`).

NOTE: The two holes of the layer table filed on 2026-08-13 (no place for a model-derived atomic op
outside Core ATen / non-equivalent lines drawn for "does not hold up in capacity or performance") were
**resolved by the revision to the entry-door model on 2026-08-14
([ADR 0059](decisions/0059-op-vocabulary-entry-doors.md))** — by retiring the requester axis and by
structuring things so that "a does-not-hold-up judgement does not affect whether an ADR is needed".
The old numbers in the dated NOTEs above (layer 1, layer 1', layer 2) remain as recorded at the time
— read them through the (old number) column of the door table.

NOTE: 2026-09-11 added `static_quantize` as an extended composite op for fixed activation quantization ([ADR 0097](decisions/0097-gemma4-qat-integration.md#addendum-2-2026-09-11--fixed-srq-op-contract)).
To preserve QAT mobile's fixed scale and its f32 intermediate rounding, it is not lowered to a plain decomposition of division, rounding and multiplication.
We measured that the tolerance of GPU division crosses integer rounding boundaries, and reproduced it with tables of integer boundaries and outputs.
It is supplied as `karume::static_quantize` → the IR op of the same name. It is not inserted automatically into existing graphs.

NOTE: 2026-10-02 added `conv3d` to the **extended composite layer** (outside Core ATen — measured
with `torch.Tag.core`). ADR [0118](decisions/0118-wan21-video-generation.md) decision 1 is the source
of truth (that ADR doubles as the ADR of this door).
Grounds = the causal 3D VAE of Wan2.1 (CausalConv3d with a time kernel of 3). The composition of
step 2 (`Σ_kt conv2d`, putting the time axis into the batch of conv2d) **can be written**, but the
output-sized intermediate becomes kt = 3 times larger, and the materialising copies of slices and the
adds increase too — beyond the precedent guide's "precedent for keeping at 1.5–2×" (entry condition
② materialising the intermediate). Partial sums per kt change the reduction order from flat K, so
adding the op later would not give the same bits. On the demand side (③ recurrence rate): the VAE of
Wan2.2 and the causal 3D VAEs of other video models (speculation).
**Semantic scope and implemented subset** (0064 axis A): the contract is **unbatched**
`x[Cin,T,H,W]`, weights `[Cout,Cin/groups,Kt,Kh,Kw]`, attrs `stride` / `padding` / `dilation` with
three components `[T,H,W]` and `groups` a scalar (all four required to declare), arity fixed at 3,
padding symmetric zeros per axis. The CPU reference holds the full semantics (general groups), while
the GPU has only the **groups == 1 implicit GEMM** (sharing the skeleton of ADR 0024 — inheriting the
bit-identity basis of ascending flat K, K tile 16, bias first and 0 out of range), and groups > 1
fails loudly at plan time (when the Session is built). Conditions for generalising: batched rank 5 is
added, distinguished by rank, when the rank limit of the strided family is raised or demand for
B > 1 appears / groups > 1 gets a direct kernel or an extension of the skeleton when demand such as
depthwise appears / causal (asymmetric) padding is not put in the vocabulary (the recipe expresses it
by cat-ing the previous chunk's cache onto the front of the time axis). Weight storage is f32 / f16 /
i8 (filling the A tile is shared with conv1d / conv2d, so i8 has an execution path too — the eligible
preset is decided by the recipe's quant presets); i4 / i2 are rejected at the entry of generation.
No geometry profile is looked up (the same default geometry as conv1d).
**Axis B** (0064): it breaks none of single output, static shapes, full-write, 4-byte storage and
full-range write of resident tensors (a rank-5 initializer is a first, but the rank limit of the
strided family applies to values, not to initializers).

NOTE: 2026-10-27 added `group_norm` to the **Core ATen layer** (`aten.native_group_norm`; ledger
entry — the ADR of the family that needs it holds the design: ADR NNNN (MOSS-Image's DiT and VAE)
decision 7). Grounds = the FLUX.2 small decoder of MOSS-Image runs **untiled** (its upstream
default — tiling would change every GroupNorm's statistics), with 30 GroupNorms (32 groups) over
symbolic spatial axes. It does not stop at step 2: the group reshape `[N, G, C/G·h·w]` is a
product of symbols (no IR form — one symbol per dimension), and the decomposed form streams every
full-size intermediate several times (the `batch_norm` counter-example above). The exporter takes
the **preserved** `aten.group_norm` (decomposed it is `native_group_norm`'s three outputs with
`HxW` = the product of the spatial symbols), added to `PRESERVED_OP_PREFIXES` (12 → 13 — no
export changes by it: a GroupNorm failed before as an unsupported `native_group_norm`).
**Contract**: `x[N,C,…]` (rank ≥ 2; C static — a symbolic C fails in the shape rule and in the
handler; any trailing extent, symbolic allowed), `weight[C]` / `bias[C]` (arity fixed at 3 — the
exporter synthesises ones / zeros for a module without affine; the bias-only form fails), attrs
`num_groups` (C divisible — a shape-rule check) and `eps` (the normalisation ops' rule),
population variance, f32 only, not in `WEIGHT_SLOTS`. **Kernel**: five
dispatches over two node temporaries — per 1,024-element chunk partial sums, a fixed-tree total per
group, in **two passes** (mean, then squared deviations — the layer_norm discipline), then the
grid-stride affine; the tree depends on the shape only (determinism, geometry invariance). Its
four-point check (ADR 0110 decision 3) and the large-group check at 16.8M elements per group are
`packages/runtime/tests/gpu_group_norm_test.ts`. The `group_norm` row of the counter-examples
("delegate to layer_norm — partly") stays true: the A/B check of the kernel uses exactly that
decomposition on small static shapes.

NOTE: 2026-10-31 added `gated_delta_scan` (the value form) to the **extended composite layer**
(outside aten — Karume's own `karume::gated_delta_scan`). ADR
[0135](decisions/0135-gated-delta-rule-scan.md) is the source of truth (accepted 2026-10-31, after the
export measurements of entry condition ① were recorded). Grounds = the Gated DeltaNet layers of the
MOSS-Image text encoder (24 of 32 layers) and of Qwen3.5-class language models. The recurrence can be
written exactly with existing ops (per token `exp` / `mul` / `sum` / `sub` / `add`, T times — composite,
not atomic), but both upstream functions loop in Python, so `torch.export` specialises the length
axis: entry condition ③ (recurrence rate) applies and ① (the symbolic length is lost) is expected. The
captured form is one node with a symbolic T; the projections, the conv, the gates and the q / k
l2-normalisation stay existing ops around it.
**Semantic scope and implemented subset** (0064 axis A): the contract is arity 5 (q / k already
l2-normalised `[B,T,Hk,dk]`, v `[B,T,Hv,dv]`, the log decay g and β `[B,T,Hv]`), uniform f32, empty
attrs; the head broadcast `⌊h/(Hv/Hk)⌋` and the q scale `dk^-½` are derived from the shapes. The CPU
reference (f64) holds the semantics for any B, T ≥ 1, dk, dv and `Hk | Hv`; the GPU kernel is the
measured sequential form with an 8-way key split for **dk = dv = 128 only**, and other head dims fail
loudly when the Session is built (`validateGraphContracts` — the conv3d split). Condition for
generalising: a model with another head dim, with a measured geometry for it. The initial / final
state is not a field (the states form for decode is named in ADR 0135 decision 4 and decided with the
first Qwen3.5-class language model).
**Axis B** (0064): the value form breaks none of single output, static shapes, full-write, 4-byte
storage and full-range write of resident tensors (its symbolic T is the same kind as `gru_scan`'s).
**Numerically hazardous class** (the rule below — the op contains `exp`): the only transcendental at
run time is `α = exp(g)`, computed with the accurate exp of `codegen/numerics-wgsl.ts`
(`EXP_ACCURATE_WGSL`), not the built-in, whose bias accumulates across the memory (ADR 0135 decision 3).
Its overflow range `x ≳ 88.4` is outside the GDN domain `g = −exp(A_log)·softplus(·) ≤ 0`; there α ∈
[0, 1], a strongly negative g underflows α toward 0 (the true value to f32 precision; the body returns 0
below −87), and the recurrence is non-expansive (`‖k‖ = 1`, `β ∈ (0, 1)`). A positive g is not guarded
and its result is unspecified (docs/limitations.md). The canaries are permanent in
`tests/gpu_gated_delta_scan_test.ts`: g = +0 / −0 decay by exactly 1 and g = −∞ by exactly 0 (inputs
on which the whole scan is exact, so the expected words are known), NaN in g and in v propagates to
exactly the reference's positions, and the accurate exp's body is measured directly (ULP and bias on
[−0.1, 0), and its special values). The acceptance is the upstream-error band against the f64 oracle
(ADR 0135 decision 6), with a long-memory case at T = 512 in the default lane that the built-in exp
fails.

## Check for numerically hazardous classes (mandatory when adding an op — made a rule on 2026-08-31)

The CPU reference of the f64 oracle evaluates the whole expression in f64 and rounds only once at the
end, so **comparison against the reference structurally cannot detect the class "in f32 the
intermediate overflows / underflows, but the mathematical final value is finite"** (the mechanism by
which gelu_tanh's silent NaN on Metal slipped through the op vocabulary sweep and survived until it
shipped — [research/2026-08-31-op-numerics-review.md](research/2026-08-31-op-numerics-review.md)).
The general inputs of the sweep (around ±4.5) do not reach the hazard band either. Therefore:

- MUST: **when adding an op that contains transcendental functions (exp / log / sqrt / pow / tanh /
  erf family), division or reciprocal square root, derive the f32 breakdown range of its intermediate
  expressions in the doc, and keep dedicated canary inputs that step into that range (saturation
  range, ±Inf, NaN-propagation pairs) permanently in the GPU tests**. Precedent =
  the tanh / gelu_tanh saturation-range canaries in tests/gpu_ops_test.ts (with exact ±1.0 / x /
  −0.0 checks).
- Three types of guard (strongest first): ① eliminate the hazard range structurally by rewriting the
  expression (sigmoid_stable's fixed −|x|, softmax's max subtraction) ② a cutoff that does not change
  any non-NaN value bit for bit (tanh_stable — the boundary must be derived) ③ carry torch's
  decomposition into the IR branch by branch (softplus's threshold-where).
  If ① can be used, do not choose ②. Passing a bare built-in function through is allowed only when
  the doc can carry a derivation that the hazard range is unreachable.

## Implementation order (carried over as the prototype ruled — a Karume re-ruling goes in an ADR)

1. Row reduce family (amax/amin/max/min/argmax/argmin) — **partly done (updated 2026-08-17)**:
   implemented are sum / amax / amin (what safe-softmax needs) + **argmax** (the exit of greedy
   decoding — ADR 0068 decision 2; revisited in the LLM wave, as announced), four in all. max / min
   (two outputs: value and index) and argmin are not implemented — the former two are a separate
   judgement after the multi-output mechanism lands, and for argmin "no additions for symmetry"
   takes precedence. **argmax is not a member of the reduce family** (a separate kind, a separate
   kernel family — the NOTE above).
2. Bulk elementwise (WGSL built-in passthrough)
3. Generalise tensor-tensor comparison
4. Generalise axes (slice step, pad on any axis, ~~sum/amax dim~~ → **implemented (2026-08-04)**:
   the reduce family (sum/amax/amin) carries attrs `dim` as **required to declare**; the last
   dimension runs on the row kernel and the others on the axis variant. The reduction order matches
   exactly between the two variants =
   bit-identical output (research/2026-08-04-vae-axis-reduce-recon.md))
5. Concentrated exporter decomposition passes (≈25 ops, 0 new kernels)
6. Dedicated native_group_norm kernel — **done (2026-10-27)**: `group_norm` (the NOTE of
   2026-10-27 above — ADR NNNN decision 7)
7. Extend the conv contract (relax B=1 and the like)
8. pooling / upsample
9. scatter_add (presumes an ADR)
10. On hold (sort/fft/data-dependent shapes) — **`topk` done on 2026-08-17** (ADR 0068 decision 3,
    a static-k multi-output op)

## Open and unverified (settle before starting)

- ~~Whether WGSL NaN comparisons yield false as IEEE754 says~~ → **measured (2026-08-02)**: a
  comparison alone yields false, as the spec says. But `select(x, m, x < m)` is folded into the max
  idiom and the driver's `max` swallows NaN, so NaN detection is done on the bit pattern (fixed at
  the root — the counter-examples above)
- The gap between WGSL `sign(NaN)` and torch (which returns NaN) — remains with both the
  decomposition and the built-in
- Whether the zero-length slice of the select_scatter decomposition fails on WebGPU's size>0 binding
  requirement
- ~~(prototype leftover) silu is in UnaryOp but is suspected to be a dead branch unreachable from the
  executor~~ →
  **resolved (2026-08-08)**: no public op was added; instead it became an executor-internal fusion
  rule that folds the adjacent two nodes `sigmoid → mul` at run time only (`src/runtime/fusion.ts`)

## Adjacent observations (on the prototype side)

- The prototype's exporter had a processing path dedicated to constant folding (the FOLDABLE path),
  and `arange` / `full` / `sin` / `cos` / `sign` / `ceil` / `reciprocal` and others were already
  handled there. As a **third way out** after "write a new kernel" and "decompose into existing ops",
  always check this path (in Karume, whether a constant-folding pass exists) before adding a new op.
- The prototype kept the design in which cumsum and sum scan sequentially, one thread = one row,
  with zero parallelism along dim. Ruling once, **before** growing the row reduce family, whether the
  template may be copied while staying sequential reduces the rewriting when switching to parallel
  later (the prototype went on implementing with this ruling deferred).

NOTE: 2026-08-19 in wave H (Gemma 4 E2B), layer 2 extended acceptance to three forms with **zero op
additions** (all equivalent rewrites in normalize — ledger NOTE only, the step-2 side of ADR 0043):
`pow(x,-0.5)` → folded to rsqrt (the pow form of Gemma4RMSNorm), folding of RMSNorm without weight
(`v_norm` = with_scale=False, the ones composed on the handler side), static `select.int` →
`slice+squeeze` composition (per-layer slicing of PLE). The runtime row selection of the token-only
exit (ADR 0068 addendum 4) also works by composition with the existing `embedding` (no new op). The
token-only exit itself was removed on 2026-10-17 (the same row selection is taken over by
`last_row[R]` of the product graph).
