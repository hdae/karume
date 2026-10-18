# 0068: Decode exit — node-level multi-output, argmax and static-k topk

- Status: accepted (2026-08-17 — delegated check, no global ruling; Codex review rounds 3 to 5
  folded in, go at round 6); partly superseded by [0083](0083-generation-api-surface.md) decision 6
  (2026-08-31) and addendum 7 (2026-09-08) (the default readback is the logits and hidden state of
  the selected rows, not a token or topk) and by [0129](0129-test-layout-and-lanes.md) (2026-10-17:
  the token-only series is retired and the logits series is no longer an acceptance fixture by
  default); on 2026-10-18 the byte-identity MUST of decision 1 and the token-only rules of addenda 4
  and 5 were withdrawn, and the ban on GPU-side argmax at the target exit was re-graded to SHOULD
- Date: 2026-08-17
- Related: ADR [0066](0066-generation-context-state-slots.md) (the reserved data-dependent-shape
  extent needs multiple outputs) / [0059](0059-op-vocabulary-entry-doors.md) (entry doors for a new
  op) / [0058](0058-numerics-opt-in-contract.md) (style of the three-part check)
- Evidence:
  [research/2026-08-17-autoregressive-references.md](../research/2026-08-17-autoregressive-references.md)
  §2 ("survey §n" below)

## Context

Reading the full-vocabulary logits back to the host at the end of each decode step is already the
minority among reference implementations (llama.cpp issues no readback at all; web-llm reads one
int32 — survey §2). Putting greedy / top-k on the GPU in Karume needs **node-level multi-output**.
Graph-level multi-output already worked, but six surfaces assumed one output per node (contracts /
plan / recipe / recipe builder / executor / exporter — survey §2). Opening them is also a
prerequisite for Kokoro-82M (LSTM `h_n`) and for the reserved data-dependent-shape (DDS) extent
(payload + extent — ADR 0066 decision 3).

## Decision

### 1. Node-level multi-output (no change to the IR schema; implementation surfaces revised)

The IR schema already allows `outs` of length one or more, so **multiple outputs need no spec
change**. The implementation surfaces are revised (eight in the code, see addendum 1):

- Contract tables (runtime `ops/contracts.ts` / exporter `ops.py`) gain a **per-output-slot dtype /
  shape mapping**, generalising "slot 0 → output, identity". The `outs.length !== 1` check becomes
  "equal to the output count the contract declares".
- **The domain includes zero outputs.** An effect op that defines no value (`state_append`, ADR
  0067) declares zero outputs in its contract. The IR parser accepts empty `outs` only for ops whose
  contract declares an effect; execution order comes from the node-array order contract (ADR 0067
  decision 5b), not from data edges. Output allocation, retain and release bookkeeping in the recipe
  are generalised to zero and to many outputs. Allowing zero outputs did change the IR spec (now
  [docs/ir-v2.md](../ir-v2.md), which permits length 0 for effect ops).
- Plan / recipe / recipe builder / executor generalise the `outs[0]` assumption (the single
  `outputName` / `output` / `uses` of `StepRecipe`) to an output list.
- The contract layer can express outputs with different dtypes (topk = values f32 + indices i32).
- Withdrawn 2026-10-18: the MUST that recipes, generated code and diagnostics of single-output nodes
  stay byte-identical. It was the no-regression condition of the introduction; codegen determinism
  is a separate invariant of the project.

### 2. `argmax` (greedy MVP — single output, introduced first)

- Contract: argmax over the last dimension; input f32, output **i32**; rank preserved (no `keepdim`
  field — the last dimension is fixed to 1).
- MUST: **ties break to the smallest index.** This is Karume's rule. It matches what torch `argmax`
  returns, but torch itself does not guarantee tie order (addendum 2). llama.cpp disagrees with
  itself (GPU = largest index, CPU sampler = smallest — survey §2); without a written rule greedy
  reproducibility splits between implementations.
- MUST: the row maximum starts at **−inf** (no finite sentinel). Together with index tracking, an
  all-−inf row deterministically yields index 0 (same reason as ADR 0067 decision 6).
- The kernel follows the existing row reduce (grid-stride local max + index → shared-memory tree
  reduction); only carrying the index is new.

### 3. `topk` (first multi-output op)

- Contract: top-k over the last dimension; **k is an attribute (required, a plan-time constant =
  static-k)**; two outputs = values f32 (**sorted descending**, as torch) + indices i32.
- MUST: no full-vocabulary argsort (MLC's WebGPU path is the expensive example — survey §2). The
  implementation is block-local top-k → merge (the llama.cpp WebGPU argsort + merge skeleton,
  narrowed to width k). Scratch, if needed, is a temporary buffer in the recipe, never co-located in
  the output buffer (that would hide the allocation).
- Ties prefer the smallest index, as in argmax. The value sequence matches torch; the index sequence
  is Karume's rule (addendum 2).
- **Accepted domain:** `1 ≤ k ≤ last dimension`. k = 0, k above the last dimension and a symbolic k
  fail loudly. A k above the implementation limit (fixed by device limits such as workgroup
  storage) also **fails loudly** with the limit in the message — no fallback, the same treatment as
  an unimplemented combination in ADR 0058 decision 3. The runtime and exporter derive their
  accepted sets from the one contract table, so the surfaces cannot disagree.

### 4. Exit shape of the decode graph

- lm_head runs **only at the last position** (vLLM style — survey §2). In Karume this is not an op
  but part of the exporter's decode graph script (decode has queryLength = 1, so nothing extra is
  needed; prefill chunks do not output intermediate logits).
- The default readback is the sampled token (argmax output, i32) or topk. Full-vocabulary logits
  remain an opt-in when the graph declares logits as an output (logprobs — the field stays).
  **(superseded by ADR 0083 decision 6 on 2026-08-31 and addendum 7 of this ADR on 2026-09-08: the
  product graph returns the logits `[1,R,V]` and hidden state `[1,R,H]` of the selected rows;
  general sampling and penalties need the full vocabulary.)**
- **Sampling / RNG stay on the host** (op-vocabulary ruling, restated as a MUST in ADR 0083
  decision 6 — the GPU side does argmax / topk only; temperature, top-p and random numbers run on the
  host after readback).

## Alternatives considered

- Full sampling on the GPU (gumbel-max / top-p renormalisation — vLLM, web-llm): cuts readback to one
  int32, but widens the RNG reproducibility contract (seed management) and the verification surface
  at once. A topk readback of k values is small enough for host sampling. Rejected (to revisit with
  measurements).
- Splitting multi-output into single-output nodes (`topk_values` / `topk_indices`): simpler
  contracts, but either the reduction runs twice or the ops share hidden state; both are worse.
  LSTM `h_n` and the DDS extent cannot be split at all. Rejected.

## Consequences

- The prerequisites of Kokoro-82M (LSTM `h_n` output) and the DDS extent (ADR 0066) are in place.
- Every op's contract carries the output mapping (existing ops make "one output, identity"
  explicit — behaviour unchanged). The exporter's output-count check reads the contract
  (`output_count` in `ops.py`).
- Checks of the implementation: argmax / topk agree with the CPU reference oracle, with direct
  tie-break checks (equal elements, all-−inf rows); fault injection (index shift, merge boundary,
  k remainder); lifetime of multi-output (a graph that consumes only one output keeps uses / release
  correct).

## Addendum 1 (2026-08-17) — implementation scout: eight surfaces, not six

Decision 1's six surfaces are eight in the code: fusion (`fusion.ts`, single-output fused steps) and
the contract fixture `packages/runtime/tests/fixtures/op-contracts.json` (the one schema both the
TS and Python implementations read). In the exporter, tuple metadata plus `operator.getitem` slot
wiring is the new mechanism (argmax, being single-output, does not need it).

## Addendum 2 (2026-08-17) — three corrections from wave B measurements

1. **Decision 3's "tie order also matches torch" does not hold** (torch 2.13.0+cpu): torch `topk`
   does not guarantee tie order — `topk([5,5,5,5],1)` returns index 2 while `argmax` returns 0. The
   rule adopted: **the value sequence equals torch numerically** (descending, same multiplicities;
   bit-identical when the equivalence class is bit-identical, true for all 200×4 measured cases;
   **the sign of ±0.0 and NaN payloads depend on the selected index** — Karume writes the bits of the
   smallest-index element and may differ from torch's unspecified choice), and **the index sequence
   is Karume's smallest-index rule** (same family as argmax; a check compares k = 1 with argmax).
2. **Implementation limit:** scratch lives in workgroup storage (no temporaries), so the limit is
   `8·W·(k+1) ≤ maxComputeWorkgroupStorageSize` (W = 32; `topkMaxK` in
   `packages/runtime/src/kernels/topk.ts`). With the WebGPU core default of 16384 bytes this is
   **k ≤ 63** (`TOPK_CORE_LIMIT_MAX_K`); the runtime requests the adapter's
   `maxComputeWorkgroupStorageSize` (`REQUIRED_LIMIT_KEYS` in `src/gpu/acquire.ts`), so a larger
   adapter allows a larger k. Exceeding the limit fails loudly with the limit and required bytes.
   The device-dependent step is recorded in [limitations](../limitations.md).
3. **Multi-output export stops at `operator.getitem`** (measured): torch.export inserts getitem on
   tuple returns, confirming addendum 1. The topk aten handler is deferred until sampling needs it
   (2026-08-17 ruling); the contract table, kernel and checks were implemented first.

## Addendum 3 (2026-08-18) — the default exit of decision 4 deferred

Wave E's decode script (minicpm5 `export_decode.py`) shipped the opt-in form (lm_head on all M rows,
logits + token outputs), because the last valid row of a prefill chunk is a runtime scalar. The
token-only form followed in addendum 4. The rule that the logits series coexists as a diagnostic
series is superseded by ADR 0129 on 2026-10-17 (see addendum 6).

## Addendum 4 (2026-08-19) — the token-only exit of decision 4

Implemented on Gemma 4 E2B (`tools/export-recipes/gemma4/export_token.py`):

- **Wiring:** a graph input `last_row[1]` i32 (last valid row = `queryLength − 1`, supplied by the
  models pipeline) → the hidden state after the final norm `[1,M,H]` → `F.embedding(last_row,
  hidden[0])` selects one row → lm_head + softcap on that row only. **No new op** (row selection is
  the existing `embedding`; the upstream form that passes a tensor to `logits_to_keep` is advanced
  indexing, outside the vocabulary). IR, runtime and exporter core unchanged. The product graph
  uses the same row selection through `ProductChunkWrapper`
  (`tools/export-recipes/gemma4/export_product.py`), which mirrors `TokenOnlyChunkWrapper` without
  the argmax (addenda 6 and 7).
- Withdrawn 2026-10-18: the MUST that the token-only exit is argmax-direct, the cross-series parity
  acceptance and the MiniCPM5 follow-up. The token-only series is not a product feature and its test
  was retired (`466b7196`, ADR 0129).

## Addendum 5 (2026-08-19) — provenance binding of the token-only series

Withdrawn 2026-10-18 with the token-only series (`466b7196`): the provenance check (a missing or
mismatching record fails instead of skipping) and the whole-directory placement rule. The
`reference.json` provenance record (`tools/export-recipes/gemma4/provenance.py`) remains in use for
the drafter series; ADR 0096 is canonical for it.

## Addendum 6 (2026-08-31) — the product graph's default exit is the last-row logits

The generation-API ruling of 2026-08-31 made the opt-in of decision 4 the default for the gemma4
product graph, but only for **the last row selected by `last_row`**: `logits[1,1,V]` (canonical: ADR
[0083](0083-generation-api-surface.md) decision 6).

- **Wiring:** the implementation of addendum 4 without the argmax (`ProductChunkWrapper` in
  `export_product.py`, which mirrors `TokenOnlyChunkWrapper` of `export_token.py`). No new op; IR, runtime and exporter core unchanged.
- **The topk exit stays deferred** (as does `operator.getitem` wiring). The implementation limit
  k ≤ 63 at the core default falls one short of `top_k: 64` in gemma-4-E2B-it's
  `generation_config.json`; repetition penalty, logit bias and full-vocabulary nucleus need the whole
  vocabulary anyway.
- **Sampling / RNG stay on the host** (full GPU sampling stays rejected; adopting it needs a new
  ruling). The host sampler's place and contract: ADR 0083 decision 7.
- The rule that the logits opt-in and token-only series coexist as acceptance fixtures **(superseded
  by ADR 0129 on 2026-10-17: the token-only series is retired; upstream parity of the logits series
  runs only with `KARUME_UPSTREAM_PARITY=1`)**.

## Addendum 7 (2026-09-08) — the product exit is the logits + hidden state of R selected rows

Generalises addendum 6 (ADR 0096 decision 5). `last_row` is `[R]` (a second symbol R; its element
count is the only source binding how many rows a run selects). The graph outputs are **output 0 =
logits `[1,R,V]`, output 1 = hidden state after the final norm `[1,R,H]`** (the order is the
contract; the runtime reads by slot number). Ordinary prefill / decode use R = 1, bit-identical in
values and tokens to the one-row exit. Speculative verification scores R = k+1 rows in one run and
feeds the accepted rows' hidden state to the drafter. R is not shared with the chunk row count M
(prefill would read back `[1,768,V]`). The structural check that row selection precedes lm_head takes
the row count as an argument (`assert_row_selected_lm_head` in
`tools/export-recipes/gemma4/export_decode.py`).

The target graph's exit SHOULD NOT reduce to argmax on the GPU; a reduction there is adopted only
after measurement (re-graded 2026-10-18 to SHOULD: the measurement it waited for was made on
2026-09-12 — ADR 0083 picks the token of temperature-0, non-speculative decode with a GPU topk(k=1)
in a separate Session while the saved graph's exit stays logits + hidden, because general sampling
and penalties need the full vocabulary).

## Addendum 8 (2026-09-09) — argmax over long rows uses two dispatches

Decision 2's argmax used one workgroup per row; at vocabulary length (gemma4 262,144) 256 threads
folding 1,024 elements each sequentially was the bottleneck (1.5–2.1 ms per drafter cycle —
[research/2026-09-09-mtp-stage4.md](../research/2026-09-09-mtp-stage4.md) §4). Rows of length
≥ 16,384 (four or more spans of 4,096 elements) run **partial (per-span maximum to a temporary
buffer) → merge (fold the spans per row)**. The result is **bit-identical** to the one-dispatch form
(the lexicographic maximum of (value descending, index ascending) does not depend on association;
spans start from the same −inf / sentinel `dim`; spans are never empty). The path is a pure function
of the row length (`argmaxSplitGroups`), shared by execution and the estimate (temporary
`[rows, groups] × 8 B`). Short rows keep their key and WGSL. The tie / NaN / all-−inf rules of
addendum 2 are pinned for the two-phase form with the same literals
(`packages/runtime/tests/gpu_ops_test.ts`). This changes only the speed of the drafter's existing
argmax exit, not the target exit (addendum 7).

## Addendum 9 (2026-09-11) — topk k=1 over long rows also uses two dispatches

- An experiment to cut output transfer found that the existing topk k = 1 path costs more on large
  vocabularies. It shares argmax's partial maximum and merge and splits into two dispatches when
  k = 1, the row count is positive and the row length is ≥ 16,384. Adopted on
  unit and whole-model measurements
  ([research/2026-09-10-codex-mtp-optimization.md](../research/2026-09-10-codex-mtp-optimization.md), section of
  2026-09-11 on the topk k=1 split).
- The comparison order is unchanged (NaN first, value descending, smallest index). The value output
  copies the selected input's u32, keeping ±0 and NaN payloads; it is never converted from an
  intermediate f32 maximum. argmax's generated code is unchanged; topk with k > 1 and short rows are
  unchanged.
- The selection function `topkOneSplitGroups` serves execution and the estimate. Rows needing more
  than 65,535 spans stay on the grid-stride one-dispatch form, so no input is rejected. The partial
  temporary is `rows × groups × 8 B`; the input and partial live until the merge, and values and
  indices go to separate output regions. Execution and the estimate agree on lifetimes.
- No change to the public API, IR or saved format; this is an internal optimisation when a Session
  runs an existing topk node. Gemma's product assets and the CLI still exit with logits + hidden, so
  this alone does not reduce transfer.
- Checked: boundaries, NaN payloads, ties, all −inf, signed zero, last index, intermediate outputs of
  a preceding node, backing reuse and the memory estimate.

## Addendum 10 (2026-09-12) — maximum and top-candidate comparison that keeps subnormals

- `argmax` and `topk` compare by an integer rank key. From the input f32 bits, positive values flip
  the sign bit and negative values flip all bits. ±0 share one key; every NaN gets the largest key.
  Equal keys still pick the smallest index.
- This is a fix, not a relaxation of the numeric contract: the implementation could not tell
  subnormals from zero. [WGSL §15.7.2](https://www.w3.org/TR/WGSL/#floating-point-evaluation) allows
  flushing subnormals in floating-point comparison, so a comparison can change even when the value
  read back is intact. Reproduction, acceptance and performance are recorded in
  [research/2026-09-10-codex-mtp-optimization.md](../research/2026-09-10-codex-mtp-optimization.md) (section of 2026-09-12
  on subnormals in maximum selection).
- The conversion happens once while scanning the input. Input, scratch and partial results stay u32,
  with no intermediate f32 bitcast; the two words of a partial result are the rank key and the index.
  topk re-reads the original bits at the selected index, keeping NaN payloads and signed zero.
- The keys of plain and split argmax / topk move to v2. Dispatch shape, uniform layout and the k
  limit from workgroup storage are unchanged, as are the public API, IR, saved assets and output
  dtypes. Rounding and tolerances of matrix and other ops are unchanged.
- Converting on every comparison was measured too; converting at input time won in back-to-back unit
  runs and was adopted. Regression tests use the CPU's ordinary numeric comparison as the reference
  and do not duplicate the GPU rank conversion. Checked: subnormals, NaN, ±0, Infinity, ties, split
  boundaries, large vocabularies and general k. Fault injection on tie direction, finite sentinels
  and NaN priority is kept.
