/**
 * The three kernels of states-form attention (ADR 0067 decisions 4 / 6 / 7). 1 node = 3 dispatches:
 *
 * | Step | Key                                      | Role                                                    |
 * | ---- | ---------------------------------------- | ------------------------------------------------------- |
 * | ①    | `attention_state_qk:v1:f32:wg16x4`       | **materialises** `S` over logical cols in row-block windows |
 * | ①'   | `attention_state_qk:v1:f32:wg16x16:par`  | **D-parallel reduction** variant of ① (opt-in — below)  |
 * | ①ₜ   | `attention_state_qk:v1:f32:reg<geometry>` | **K row-tile sharing** variant of ① (default — below)   |
 * | ②    | `attention_state_stats:v2:f32:wg256`     | per-row `m = amax S` and `inv = 1/Σexp(S−m)`            |
 * | ③    | `attention_state_pv:v1:f32:wg16x4`       | `O = P @ V` (`P = exp(S−m)·inv` is **not materialised**) |
 * | ③'   | `attention_state_pv:v1:f32:wg16x16:par`  | **KV-parallel reduction** variant of ③ (opt-in — below) |
 * | ③ₜ   | `attention_state_pv:v1:f32:reg<geometry>` | **V row-tile sharing** variant of ③ (default — below)   |
 *
 * A **separate family** from the existing fused attention (src/kernels/attention.ts + the GEMM
 * skeleton), with its own key family. ①ₜ / ③ₜ do share the row tiles of the GEMM skeleton
 * (src/kernels/gemm.ts) and plug in only the fragments that read states. There are two reasons for
 * the separate family, both because the semantics differ, not to avoid touching the existing side:
 *
 * 1. **K / V come from two sources** (logical col < pastLength from the state slot, the rest from
 *    this step's `ins`), while the tile fill of the existing fused attention assumes "one
 *    contiguous buffer".
 * 2. **The scan range is a run-time value** (the live column count read from the logical-length
 *    uniform), not proportional to the capacity `C` (the workload acceptance condition of ADR 0066
 *    decision 3).
 *
 * The identity −inf of the row statistics and the empty-row guard (the states form **legitimately**
 * produces padding rows and empty-context rows, so empty row → output 0 is guaranteed by
 * construction — ADR 0067 decision 6; substituting a finite sentinel is MUST NOT) are no longer a
 * reason for the separate family, since the existing ② took the same form in v2.
 *
 * The QK path splits three ways by M (physical chunk rows): ①' for M ≤ 8 when the option is
 * parallel ({@link stateQkParallelEligible}), ①ₜ for M ≥ 16 regardless of the option
 * ({@link stateQkTiledEligible}), and ① otherwise (9 to 15, and M ≤ 8 with sequential).
 *
 * MUST: **the bit-identity contract (agreement with the decomposed path) does not apply to the
 * states form** (ADR 0067 decision 4 has no decomposed path — GQA models must preserve SDPA).
 * Instead **determinism** applies as before: same key → byte-identical WGSL, same input → same
 * output (the reduction is fixed as a sequential pass in ascending col).
 *
 * ## ③' KV-parallel reduction variant (`SessionOptions.stateAttentionReduce: "parallel"` — perf-ledger K-12)
 *
 * `"parallel-fused"` selects ③' as the same option value (on eligible plans it replaces ② and ③'
 * with the fused stats-PV — ADR 0102, state-attention-stats-pv.ts).
 *
 * In ③ one invocation accumulates one element of O in a **sequential loop** over the live columns,
 * so in decode (M=1) the active invocations are fixed at `D × B·H` (4,096 on the full layers of
 * Gemma 4 E2B), and as the KV length grows only one thread's sequential length grows (at P=16K
 * attention is 72% of decode GPU time — docs/research/2026-09-03-gemma4-context-length-sweep.md).
 * ③' rearranges the workgroup in two dimensions, `TILE_X (D direction) × KV_LANES (KV direction)`;
 * lane `l` accumulates the columns with `cl ≡ l (mod KV_LANES)` in ascending order, and the partial
 * sums are folded in workgroup shared memory by a **fixed-order tree reduction** (stride 8 → 4 → 2
 * → 1). The dispatch count and intermediate buffers are the same as ③ (only the lanes inside the
 * workgroup increase).
 *
 * MUST: the reduction order differs from ③, so it is **not bit-identical** (determinism holds —
 * same input → same output). It is an opt-in option value under ADR 0058, and the default is ③
 * (the reference path). The checks come as a set of three (the reference path's checks unchanged,
 * the A/B tolerance check of ③' = tests/gpu_state_attention_parallel_test.ts, the census check =
 * tests/gpu_state_execution_test.ts).
 * SHOULD: **③' is selected only for plans with `M < 16`** (the condition is the other side of
 * {@link statePvTiledEligible}) (re-graded 2026-10-18 — ADR 0067, geometry-table addendum of
 * 2026-09-06). `M ≥ 16` is taken by ③ₜ regardless of the option (below, bit-identical to ③) — even
 * with the option `"parallel"`, prefill plans produce the reference path's values. In the K-12
 * measurements the gain of ③' was confined to decode (prefill within noise —
 * docs/research/2026-09-03-gemma4-chunklength-k12-sweep.md), so preferring ③ₜ, which cuts
 * traffic, loses nothing.
 * MUST: the pad-row branch is **workgroup-uniform** (the local row comes from `workgroup_id.y`), so
 * returning before the barrier is allowed, but lanes with `d ≥ D` take part in the barrier (they do
 * not return — they run an empty scan and contribute `0.0`). A WGSL barrier cannot sit outside
 * uniform control flow.
 *
 * ## ①' D-parallel reduction variant (**the same option value as ③'**, `stateAttentionReduce: "parallel"` — perf-ledger K-14)
 *
 * `"parallel-fused"` selects ①' as the same option value (the fusion is on the ③ side only — ADR
 * 0102, state-attention-stats-pv.ts).
 *
 * In ① one invocation owns one element of S (`(local row, live column)`) and the dot product is a
 * **sequential loop** over `D` (one thread does D multiply-adds). In decode (M=1) the invocation
 * count grows with the live columns, but the latency of one invocation stays the long D-sequential
 * one, and at P=16K ①QK is 17% of decode GPU time (breakdown after K-12 —
 * docs/research/2026-09-03-gemma4-chunklength-k12-sweep.md). ①' rearranges the workgroup
 * **symmetrically** to ③' into `TILE_X (column direction = S's cl) × D_LANES (D direction)`; one
 * workgroup = one local row (`workgroup_id.y`) × `TILE_X` columns. Lane `l` accumulates `d = l,
 * l + D_LANES, …` in ascending order, the partial sums are folded in workgroup shared memory by a
 * **fixed-order tree reduction** (stride 8 → 4 → 2 → 1), and lane 0 writes the one word of S. The
 * dispatch count, bindings, params and intermediate buffers are the same as ① (only the lanes
 * inside the workgroup increase, and only the row axis changes from `⌈valid rows / TILE_M⌉` to
 * "valid rows").
 *
 * MUST: the reduction order differs from ①, so it is **not bit-identical** (determinism holds).
 * **The option value is the same single one as ③'** (specifying `"parallel"` selects ①' and ③'
 * together — the ruling of 2026-09-06; no new knob).
 * MUST: **①' is selected only for plans with `M ≤ 8`** (the condition is
 * {@link stateQkParallelEligible} — decode's M=1 and speculative verify with M ≤ 8). Plans with
 * `M ≥ 9` run ① even with the option `"parallel"` — in prefill ① is already filled by rows ×
 * columns, and ①' was measured slower by the narrower row tile and the added barriers (2026-09-06
 * — details in that function's WHY). On the ③' side only plans with `M < 16` take it (`M ≥ 16`
 * takes ③ₜ regardless of the option), so **one option value covers two steps with different
 * ranges** (for decode / verify with `M ≤ 8`, ①' and ③' are selected together).
 * MUST: not one of ①'s contracts moves — outside the predicate (causal + sliding lower bound) but
 * inside the live range, **−inf is always written** (so ② does not consume leftovers); columns with
 * `cl ≥ live` and pad rows are not written; the scaling (half scale on both the q side and the k
 * side) and the bits of −inf (`params.neg_inf`) are the same as ①.
 * MUST: the pad-row branch is **workgroup-uniform** (from `workgroup_id.y`), so returning before
 * the barrier is allowed, but columns with `cl ≥ live` and lanes with `d ≥ D` do not return and
 * **contribute 0 through an empty run** (the same reason as ③' — a WGSL barrier cannot sit outside
 * uniform control flow).
 *
 * ## ①ₜ K row-tile sharing variant (**the default path regardless of the option** — perf-ledger K-13)
 *
 * In ① one invocation = one element of S, so one row of K is re-read **once per row**. In prefill
 * (M = 768), on a 14.7K-token chunk, ①QK is a traffic-bound 40% of the chunk's GPU time (breakdown
 * in docs/research/2026-09-06-state-qk-parallel-k14.md). ①ₜ is the **GEMM skeleton** that fused
 * attention's `attention_qk` already has (shared-memory tiles, register blocking, M-bucket geometry
 * — src/kernels/gemm.ts) with the states fragments plugged in; it loads K into the row tile once
 * and reuses it across the q of `tileM` rows. The matrix mapping is A = the q of this row block
 * (`[rows_block × depth]`), B = Kᵀ (`[depth × live]`, columns = the two sources of logical col),
 * output = S (the same indexing as ①).
 *
 * MUST: **bit-identical to ①**. The summation order per output element is fixed by the skeleton as
 * a sequential pass in ascending `d` (K tiles of 16 in ascending order — ADR 0022 decision 3),
 * exactly matching the sequential pass of ①'s `stateScoreFn`. The half scale is also matched in the
 * one-term expression (`(q·scale) · (k·scale)`) (q side in the A loader, k side in the B loader).
 * That is why it can be selected as the **default path** regardless of the option
 * (`stateAttentionReduce`) — unlike ①', which changes the reduction order.
 * MUST: the effective `N` is **the run-time live**. The skeleton's `Dims.n` gets the static upper
 * bound `col_cap`, but the B loader reads only columns with `cl < live` and the store writes only
 * `cl < live` (the leftovers in `[live, col_cap)` are left untouched, as in ①). Rows go up to
 * `local_row < effective_rows(query)` (pad rows get not a single word).
 * MUST: the predicate is the same `in_window(col, past + row)` as ①. Inside the live range **−inf
 * is always written** even outside the predicate (so ② does not consume leftovers). Running the
 * multiply-adds of columns outside the predicate is allowed (the K rows read are in range, so it is
 * safe, and the value is discarded by the `select` at the store).
 * SHOULD: the condition is {@link stateQkTiledEligible} (`M ≥ 16`) (re-graded 2026-10-18 — ADR
 * 0067, geometry-table addendum of 2026-09-06). The priority is `M ≤ 8` (decode's M=1 and verify)
 * with the option `"parallel"` / `"parallel-fused"` → ①' / `M ≥ 16` → ①ₜ / otherwise → ① (decided
 * in one place on the runtime side, `buildStateAttention`).
 *
 * ## ③ₜ V row-tile sharing variant (**the default path regardless of the option** — perf-ledger K-13 step 2)
 *
 * In ③ one invocation = one element of O, so one row of V is re-read **once per row** (the same
 * traffic bound that ①ₜ removed on the K side is still there on the V side in prefill). ③ₜ is the
 * same GEMM skeleton as ①ₜ (src/kernels/gemm.ts) with the states fragments plugged in; it loads V
 * into the row tile once and reuses it across the P of `tileM` rows. The matrix mapping is A = P
 * (`[rows_block × live]`, **computed from S and the row statistics while filling**, not
 * materialised), B = V (`[live × depth]`, rows = the two sources of logical col), output = O (the
 * same indexing as ③).
 *
 * MUST: **bit-identical to ③**. The A loader builds the same one-term expression as ③,
 * `p = exp(S − m) · inv` (the same form as `probability` in fused attention's ③PV), and the
 * reduction is fixed by the skeleton as a **sequential pass in ascending col** (K tiles of 16 in
 * ascending order — ADR 0022 decision 3), so the summation order exactly matches ③'s sequential
 * ascending-col pass. That is why it can be selected as the **default path** regardless of the
 * option (`stateAttentionReduce`).
 * MUST: the effective `K` (reduction length) is **the run-time live**. The skeleton's `Dims.k`
 * gets S's column stride `col_cap`, but the K tile loop bound is built from live (looping up to
 * `col_cap` makes the work proportional to the capacity and fails the acceptance condition of ADR
 * 0066 decision 3). The A loader reads only columns with `cl < live`.
 * MUST: **all** `rows_block` rows are written (the same full-write invariant as ③ — the dispatch's
 * row axis is `rowsBlock` too). But pad rows (`local_row ≥ valid rows`) get **an exact `0.0`, not
 * `acc`** (equivalent to ③'s "pad rows scan no live column and are 0". Merely zero-filling A would,
 * with a non-finite V mixed in, turn a pad row into NaN through `0 · NaN` and differ from ③ by one
 * word).
 * MUST: if a row tile contains no valid row, **the K loop does not run even once** (a mirror of ③'s
 * pad rows not scanning live — the mechanism that keeps the work proportional to Q). The test comes
 * from `workgroup_id.y` and is workgroup-uniform, so it does not break the uniformity requirement
 * of the inner `workgroupBarrier`.
 * SHOULD: the condition is {@link statePvTiledEligible} (`M ≥ 16`) (re-graded 2026-10-18 — ADR
 * 0067, geometry-table addendum of 2026-09-06). The priority is `M ≥ 16` → ③ₜ (regardless of the
 * option) / `M < 16` with the option `"parallel"` → ③' / otherwise → ③ (decided in one place on the
 * runtime side, `buildStateAttention`).
 *
 * ## Symbols (source of truth = ADR 0067 decision 4)
 *
 * | Symbol   | Meaning                                    | Source                               |
 * | -------- | ------------------------------------------ | ------------------------------------ |
 * | `P`      | pastLength (logical length of committed KV) | **run time** — logical-length uniform |
 * | `Q`      | queryLength (real tokens of this step)     | **run time** — logical-length uniform |
 * | `M`      | physical chunk rows (`chunk_rows`)         | static (declared shape)              |
 * | `W`      | sliding window (`0` = full)                | static (attrs `window`)              |
 * | `C`      | row capacity of the slot (`capacity`)      | static (states declaration + binding) |
 * | `r`      | GQA repeat count `H / Hkv`                 | static (derived value)               |
 * | `D`      | head dimension                             | static                               |
 *
 * ## Logical col space and the live range
 *
 * Logical col is `[0, P+Q)`. `col < P` is past (read from the **slot**), `col ≥ P` is current
 * (read from row `col − P` of **ins**). S materialises only the resident range: from
 * `colBase = full: 0 / sliding: P − min(P, W−1)` ({@link stateColumnBase}),
 * `n_live = (P − colBase) + Q` columns ({@link stateLiveColumns}).
 *
 * S is stored row-major as `[B·H, rows_block, colCap]`, where `colCap` is a **static** upper bound
 * on the column stride.
 * MUST: kernels use `colCap` **only as a stride** and cut the scan at `n_live` (running up to
 * `colCap` makes the work proportional to the capacity and fails the acceptance condition of ADR
 * 0066 decision 3).
 * MUST: the leftovers in `[n_live, colCap)` are made harmless by **readers not touching them** (②③
 * both cut at `n_live`). Conversely `[0, n_live)` **always gets −inf** even outside the predicate
 * (S is a temporary buffer holding leftovers from the previous run — without the write ②'s amax
 * consumes them).
 *
 * ## The predicate is MUST on both sides (ADR 0067 decision 4, a high finding of the fifth review round)
 *
 * For output row `row` (= `row_offset` + local row): `col ≤ P + row` (causal) AND, with sliding,
 * `col ≥ max(0, P + row − W + 1)`. **An implementation with only the upper bound is forbidden** —
 * with W=4, row 1 would silently mix in 5 keys. The lower bound is held as `(P + row) − col < W`
 * ({@link stateWindowFn} — equivalent forms: `col ≥ limit−W+1 ⟺ limit−col ≤ W−1 ⟺ limit−col < W`.
 * The subtraction is evaluated **only after** the causal upper bound `col ≤ limit` short-circuits,
 * so it does not wrap around in u32).
 *
 * ## Valid rows and pad rows (work proportional to Q — ADR 0066 decision 3 / the corrected formula of addendum item 1)
 *
 * The physical chunk has `M` rows, but valid data is the compact prefix of the first `Q` rows (ADR
 * 0066 addendum 6). The **valid row count** of row block `[row_offset, row_offset + rows_block)` is
 * `clamp(Q − row_offset, 0, rows_block)` ({@link stateEffectiveRows}), and
 *
 * - **①QK / ②stats cover valid rows only** (the dispatch count and the kernel's row range come from
 *   the same single formula — the host side is a pure function of the geometry, the WGSL side is
 *   its mirror {@link stateEffectiveRowsWgsl}). For a block with 0 valid rows the host computation
 *   returns 0 and the dispatch itself is not recorded.
 * - **③PV writes all M rows of the declared shape** (the full-write invariant is unchanged), but
 *   **pad rows scan no live column and write an exact `0.0` to all of D, then return**. Empty rows
 *   (rows where no col satisfies the predicate) are **a subset of pad rows** (a valid row always
 *   contains the causal self-reference `col = P + row`, so it is non-empty), so this zero write
 *   **contains empty row → 0 (ADR 0067 decision 6) by construction**, and also closes the remaining
 *   hole where "a non-finite V turns an empty row's output into NaN through `0 · NaN`". ②'s
 *   empty-row guard **stays as a defence**.
 *
 * ## How empty row → 0 is built (ADR 0067 decision 6)
 *
 * ②'s row max has the identity **−inf**. If `amax == −inf` (an empty row) it writes `(m, inv) =
 * (0.0, 0.0)`, otherwise `inv = 1/Σexp(S−amax)` (the largest element gives `exp(0)=1`, so the
 * denominator is ≥ 1). ③'s `p = exp(S − m)·inv` becomes
 *
 * - empty row: `exp(−inf − 0) · 0 = 0 · 0 = 0` (exact)
 * - outside the predicate on a non-empty row: `exp(−inf − finite) · inv = 0 · inv = 0` (exact)
 *
 * so **an exact 0** comes out by construction. MUST: ②'s Σ loop does not run on an empty row
 * (`exp(−inf − (−inf))` = `exp(NaN)` = NaN would enter the denominator).
 */

import { IS_NAN_BITS_WGSL, NAN_MAX_WGSL } from "../codegen/numerics-wgsl.ts";
import { CodegenError } from "../codegen/errors.ts";
import { gridStrideWorkgroups, tiledWorkgroups } from "../codegen/dispatch.ts";
import { assertU32Params } from "../codegen/params.ts";
import {
  gemmGeometryForRows,
  gemmGeometryTileKeyPart,
  gemmTileM,
  gemmTileN,
} from "./gemm-geometry.ts";

/** Workgroup width of ①QK / ③PV (columns for ①, D for ③). */
export const STATE_ATTENTION_TILE_X = 16;

/** Workgroup height of ①QK / ③PV (rows for both — the local rows of the row block). */
export const STATE_ATTENTION_TILE_M = 4;

/**
 * Workgroup width of ② (a 256-wide tree reduction with 1 row = 1 workgroup — the same shape as
 * the existing attention_stats).
 */
export const STATE_STATS_WORKGROUP_SIZE = 256;

/** Words per row of ②'s row statistics (`[0]` = row maximum / `[1]` = `1/Σexp(S−m)`). */
export const STATE_STATS_STRIDE = 2;

/**
 * f32 bits of −inf (carried in a params word).
 *
 * MUST: do not write it as a constant expression in WGSL (the same reason as argmax / topk /
 * safe_softmax — an implementation may reject the constant expression `bitcast<f32>(0xff800000u)`
 * with a shader-creation error as "a const-expression is inf").
 */
export const STATE_NEG_INF_BITS = 0xff800000;

/**
 * Variant discriminator (**only the two bits `:sliding` and `:gqa`**).
 *
 * MUST: the order is fixed here in one place (both can be set, so a scattered order gives the same
 * configuration two keys — the same rule as `maskKeyPart` / `gqaKeyPart` in attention.ts).
 * MUST: **the values of `W` / `r` / `C` themselves are not in the key** (they travel in the uniform —
 * putting them in would add a pipeline per distinct value).
 */
export const stateVariantKeyPart = (sliding: boolean, gqa: boolean): string =>
  `${sliding ? ":sliding" : ""}${gqa ? ":gqa" : ""}`;

/**
 * **The only test** for the sliding variant (`window = 0` = the attrs field is absent = the whole
 * context).
 *
 * MUST: key selection, WGSL generation and the params check all go through this one function. If
 * one side tested `window > 0` and another a different condition, full params (`window = 0`) would
 * reach sliding WGSL, and the u32 underflow of `window - 1u` plus the implementation-dependent value
 * of `col % 0u` would cause silent misreads.
 */
export const stateSliding = (window: number): boolean => window > 0;

export const stateQkKey = (sliding: boolean, gqa: boolean): string =>
  `attention_state_qk:v1:f32:wg${STATE_ATTENTION_TILE_X}x${STATE_ATTENTION_TILE_M}${stateVariantKeyPart(
    sliding,
    gqa,
  )}`;

/**
 * D-direction lane count of ①' (workgroup = `TILE_X × D_LANES` = 256 threads — fits the same
 * portability floor as ③', the spec default 256 of `maxComputeInvocationsPerWorkgroup`).
 */
export const STATE_QK_D_LANES = 16;

/** Key of ①' (`:par` is the marker for the census check — followed by the same variant bits as ①). */
export const stateQkParallelKey = (sliding: boolean, gqa: boolean): string =>
  `attention_state_qk:v1:f32:wg${STATE_ATTENTION_TILE_X}x${STATE_QK_D_LANES}:par${stateVariantKeyPart(
    sliding,
    gqa,
  )}`;

/**
 * **Eligibility** of ①' — only plans whose `M` (physical chunk rows) is 8 or less. Even with the
 * option (`stateAttentionReduce`) set to `"parallel"`, plans that fail this condition run ① (the
 * option itself is checked on the runtime side by `buildStateAttention`; this is a pure function
 * that looks only at the shape of the plan).
 *
 * WHY: what ①' shortens is "the latency of one invocation running `D` multiply-adds
 * sequentially", and that is the bottleneck only in decode (M=1), where the active invocations are
 * just `live columns × 1 row`. In prefill (M=768) ① is already filled by rows × columns, and ①'
 * only narrows the row tile from `TILE_M` = 4 to 1, adding 4× the workgroups and barriers (1
 * workgroup = 1 row, so the reuse of K rows across rows is lost as well). The measurements
 * (2026-09-06) showed decode improving 6.3 → 3.7–4.1 ms/token (×1.55–1.72), while prefill's ①QK
 * GPU time grew 1.5–1.9× and prefill as a whole regressed by +30–60%.
 * For speculative verify (M = k+1 ≤ 8, MTP stage 4-B ⑤, measured 2026-09-09 — research 2026-09-09
 * §6.2) it was neutral: RTX, P≈4.8K, M=4, ①QK full 3.32 → 3.87 ms, sliding 1.87 → 1.09 ms (−0.2 ms
 * in total), wall −2% (within noise). It is widened even though neutral because with the default
 * option value (parallel) verify's row 0 and decode then share **the same reduction order** (u32
 * match), so row 0's logits do not move wherever speculation switches on or off.
 * SHOULD: keep the condition **inside the measured range** (the same rule as ADR 0082 decision 4 —
 * do not widen it by extrapolating "a small M is probably a gain". M=1 (a gain) and M=4 (neutral)
 * were measured, and `M <= 8` is the range matched to verify's bucket limit = the sliding slack 8.
 * M ≥ 9 is unmeasured, so it stays on ①) (re-graded 2026-10-18 — ADR 0082 decision 4). That ③'
 * (PV) has no such check may look asymmetric, but the 2026-09-03 measurements confirmed that ③'
 * does not regress in prefill either (it reduces along the KV length, and the sequential length per
 * thread grows in prefill too).
 */
export const stateQkParallelEligible = (chunkRows: number): boolean => chunkRows <= 8;

/**
 * Key of ①ₜ (the tiled path of the GEMM skeleton — perf-ledger K-13). The geometry is spelled with
 * the same fragment as on the gemm side ({@link gemmGeometryTileKeyPart}), so the `M` bucket goes in
 * as is.
 *
 * MUST: the geometry goes in (in the position of ①'s `wg16x4`). Without it, two plans in different
 * `M` buckets would ask for different WGSL under the same key, breaking "same key → byte-identical
 * WGSL".
 * MUST: `chunkRows` is **the same value** passed to generation (`attentionStateQkTiledWgsl` in
 * gemm.ts) (the same wording as the MUST of `gemmKeyPart` in gemm.ts — forgetting to pass it on one
 * side lets cached WGSL of another geometry disagree with the dispatch count and drop output tiles).
 */
export const stateQkTiledKey = (sliding: boolean, gqa: boolean, chunkRows: number): string =>
  `attention_state_qk:v1:f32:${gemmGeometryTileKeyPart(gemmGeometryForRows(chunkRows))}${stateVariantKeyPart(
    sliding,
    gqa,
  )}`;

/**
 * **Eligibility** of ①ₜ — only plans whose `M` (physical chunk rows) is at least the smallest row
 * tile edge, 16.
 *
 * WHY: ① is one invocation = one element of S, traffic-bound because it re-reads one row of K
 * **once per row**. ①ₜ loads K into the shared-memory tile of the GEMM skeleton once and reuses it
 * across the q rows of that row tile (`tileM` of the `M` bucket). Re-reads drop only when the row
 * tile holds 2 or more rows, and the smallest bucket M16N16 has `tileM` = 16, so with `M < 16` most
 * of the tile runs empty: the same traffic as ① plus the cost of barriers and shared memory.
 * MUST: **bit-identical to ①** (the summation order per output element matches as a sequential
 * pass in ascending d — the numeric contract of ADR 0022 decision 3 is an invariant of the
 * skeleton). That is why it is the **default path** regardless of the option
 * (`stateAttentionReduce`), unlike ①', which changes the reduction order.
 * MUST: for plans with `M ≤ 8` (decode's M=1 and verify) it does not overlap ①'s condition
 * ({@link stateQkParallelEligible}) (8 < 16). The priority is held in one place on the runtime
 * side, `buildStateAttention`.
 */
export const stateQkTiledEligible = (chunkRows: number): boolean => chunkRows >= 16;

// v2: the row max uses nan_max (closes the hole where an all-NaN row passed as an empty row and got
// stats (0,0) — ADR 0020. Bit-identical for non-NaN input)
export const stateStatsKey = (sliding: boolean): string =>
  `attention_state_stats:v2:f32:wg${STATE_STATS_WORKGROUP_SIZE}${stateVariantKeyPart(
    sliding,
    false,
  )}`;

export const statePvKey = (sliding: boolean, gqa: boolean): string =>
  `attention_state_pv:v1:f32:wg${STATE_ATTENTION_TILE_X}x${STATE_ATTENTION_TILE_M}${stateVariantKeyPart(
    sliding,
    gqa,
  )}`;

/**
 * KV-direction lane count of ③' (workgroup = `TILE_X × KV_LANES` = 256 threads — fits the
 * portability floor, the spec default 256 of `maxComputeInvocationsPerWorkgroup`).
 */
export const STATE_PV_KV_LANES = 16;

/** Key of ③' (`:par` is the marker for the census check — followed by the same variant bits as ③). */
export const statePvParallelKey = (sliding: boolean, gqa: boolean): string =>
  `attention_state_pv:v1:f32:wg${STATE_ATTENTION_TILE_X}x${STATE_PV_KV_LANES}:par${stateVariantKeyPart(
    sliding,
    gqa,
  )}`;

/**
 * Key of ③ₜ (the tiled path of the GEMM skeleton — perf-ledger K-13 step 2). The spelling rule is
 * the same as ①ₜ's {@link stateQkTiledKey}: the `M` bucket goes into the geometry fragment
 * ({@link gemmGeometryTileKeyPart}) as is.
 *
 * MUST: the geometry goes in (in the position of ③'s `wg16x4`). Without it, two plans in different
 * `M` buckets would ask for different WGSL under the same key, breaking "same key → byte-identical
 * WGSL".
 * MUST: `chunkRows` is **the same value** passed to generation (`attentionStatePvTiledWgsl` in
 * gemm.ts) and to the dispatch ({@link statePvTiledWorkgroups}).
 */
export const statePvTiledKey = (sliding: boolean, gqa: boolean, chunkRows: number): string =>
  `attention_state_pv:v1:f32:${gemmGeometryTileKeyPart(gemmGeometryForRows(chunkRows))}${stateVariantKeyPart(
    sliding,
    gqa,
  )}`;

/**
 * **Eligibility** of ③ₜ — only plans whose `M` (physical chunk rows) is at least the smallest row
 * tile edge, 16.
 *
 * WHY: ③ is one invocation = one element of O, traffic-bound because it re-reads one row of V
 * **once per row** (the same shape that ①ₜ removed on the K side remains on the V side). ③ₜ loads V
 * into the shared-memory tile of the GEMM skeleton once and reuses it across the P rows of that row
 * tile (`tileM` of the `M` bucket). Re-reads drop only when the row tile holds 2 or more rows, and
 * the smallest bucket M16N16 has `tileM` = 16, so with `M < 16` most of the tile runs empty: the
 * same traffic as ③ plus the cost of barriers and shared memory.
 * MUST: **bit-identical to ③** (the one-term expression is the same `exp(S − m)·inv`, and the
 * summation order matches as a sequential pass in ascending col — the numeric contract of ADR 0022
 * decision 3 is an invariant of the skeleton). That is why it is the **default path** regardless of
 * the option (`stateAttentionReduce`), unlike ③', which changes the reduction order.
 * MUST: the threshold is **the same 16** as ①ₜ ({@link stateQkTiledEligible}), but the function is
 * kept separate (①ₜ and ③ₜ are separate kernels, so the range of one can move on its own). Decode
 * with `M = 1` meets neither condition, so ①' / ③' are selected per the option.
 */
export const statePvTiledEligible = (chunkRows: number): boolean => chunkRows >= 16;

/**
 * Static params shared by ①③ (**eligible for the content-addressed cache** — no value that changes
 * every step).
 *
 * Word order (**this table is the source of truth**; on the host side only
 * {@link stateAttentionParams} builds it):
 *
 * | Word | Field        | Meaning                                                  | ① | ③ |
 * | ---- | ------------ | -------------------------------------------------------- | - | - |
 * | 0    | `rows_block` | rows this dispatch handles (rows of S)                   | ✓ | ✓ |
 * | 1    | `row_offset` | first row in the chunk (global row = offset + local)     | ✓ | ✓ |
 * | 2    | `chunk_rows` | `M` (row stride of q / ins / O)                          | ✓ | ✓ |
 * | 3    | `depth`      | `D`                                                      | ✓ | ✓ |
 * | 4    | `kv_repeat`  | `r = H / Hkv`                                            | ✓ | ✓ |
 * | 5    | `window`     | `W` (`0` = full)                                         | ✓ | ✓ |
 * | 6    | `capacity`   | `C` (row capacity of the slot)                           | ✓ | ✓ |
 * | 7    | `col_cap`    | column stride of S                                       | ✓ | ✓ |
 * | 8    | `neg_inf`    | f32 bits of −inf                                         | ✓ |   |
 * | 9    | `scale`      | half scale (**f32**)                                     | ✓ |   |
 *
 * MUST: ①③ use the same struct (= the same word order). Holding the same geometry in two tables
 * lets a dispatch whose table was updated on one side only read another buffer region without an
 * exception. Having the word order in one place outweighs the waste of the two words ③ does not
 * read (`neg_inf` / `scale`).
 */
export const STATE_PARAMS_STRUCT = `struct Params {
  rows_block: u32,
  row_offset: u32,
  chunk_rows: u32,
  depth: u32,
  kv_repeat: u32,
  window: u32,
  capacity: u32,
  col_cap: u32,
  neg_inf: u32,
  scale: f32,
}`;

/**
 * Logical-length uniform (the carrier of ADR 0066 addendum 4 — 8 mutable bytes owned by the
 * `GenerationContext`).
 *
 * MUST: `P` / `Q` are not put into the static params (putting values that change every step into
 * the content-addressed cache hits both "unbounded cache growth" and "cannot be updated on a
 * PreparedPlan hit").
 * MUST: it goes in **the last binding** (the four kernels share the order "the end of the binding
 * table = the context-owned surface"; `lengths` of src/runtime/generation-context.ts goes in as is).
 */
export const STATE_LENGTHS_STRUCT = `struct Lengths {
  past: u32,
  query: u32,
}`;

/**
 * Default name of the uniform variable the predicate and geometry fragments read (`params` of ①
 * ①' ② ③ ③' and `state_append`).
 *
 * Only ①ₜ (the tiled path of the GEMM skeleton) puts the skeleton's `dims` at binding 0, so it
 * passes {@link STATE_TILED_UNIFORM} to the fragments and generates from **the same single
 * expression** (copying the expression would let reads and writes drift silently when only one copy
 * is fixed).
 */
const STATE_UNIFORM = "params";

/**
 * Mapping from logical col → physical slot row (**same formula for reads and writes, MUST** — ADR
 * 0067 decision 4).
 *
 * The **reads** of ①QK / ①ₜ / ③PV and the **writes** of `state_append` share this one string
 * (src/kernels/state-append.ts imports it). A different formula on the read side alone silently
 * points every read at another row once the ring has wrapped (a silent misread with neither an
 * exception nor NaN).
 *
 * WHY the modulus is **`capacity`**, not `window`: the physical row count of a sliding slot is not
 * exactly the window but "window + slack", and the slack holds **rows written ahead of the logical
 * length** (speculative-decoding drafts — rows that verification may reject). The logical column a
 * rejected row `P+i` (`0 ≤ i < slack`) overwrites is `P+i − capacity`, which is always below the
 * lower edge of the next run's live window `P+a − (W−1)` (`a` = accepted rows `≥ 0`)
 * (`i + W − 1 − a ≤ W + slack − 2 < capacity`). So as long as there is slack, the writes of rejected
 * rows break not a single row of live past KV. Going back to `window` as the modulus lets rejected
 * rows overwrite columns inside the window, and the next step reads another token's KV as its past
 * **with neither an exception nor NaN**. `window` stays with `column_base` / `live_columns` /
 * `in_window` (the window over logical columns) — with this change the physical row count and the
 * logical window width became separate quantities.
 */
export const stateSlotRowWgsl = (sliding: boolean, uniform = STATE_UNIFORM): string =>
  `fn slot_row(col: u32) -> u32 {
  return ${sliding ? `col % ${uniform}.capacity` : "col"};
}`;

/**
 * The two functions that decide the live range (the window of logical cols that S materialises).
 *
 * `column_base` = full: `0` / sliding: `P − min(P, W−1)` (before the append, so every row up to
 * row 0's window is resident — ADR 0067 decision 4). `live_columns` = `(P − column_base) + Q`.
 */
export const stateLiveWgsl = (sliding: boolean, uniform = STATE_UNIFORM): string =>
  `fn column_base(past: u32) -> u32 {
  return ${sliding ? `past - min(past, ${uniform}.window - 1u)` : "0u"};
}

fn live_columns(past: u32, query: u32) -> u32 {
  return ${sliding ? `min(past, ${uniform}.window - 1u)` : "past"} + query;
}`;

/**
 * The predicate (`limit = P + row`). The sliding lower bound **MUST be ANDed** with the causal upper
 * bound.
 *
 * The lower bound is `limit − col < W` (the u32-safe form of `col ≥ limit − W + 1`).
 *
 * MUST: **the subtraction is on the `limit − col` side** (not `col + W > limit`). WGSL's `&&`
 * short-circuits, so it is evaluated only when the preceding `col <= limit` is true, and
 * `limit − col` never wraps around. Conversely `col + W` **wraps around in a generation where P has
 * advanced near the u32 maximum**, judging even the causal diagonal (`col = limit`) outside the
 * window and silently emptying the row. Computing `limit − W + 1` directly is not used either,
 * since it underflows when `limit < W−1`.
 */
const stateWindowFn = (sliding: boolean, uniform = STATE_UNIFORM): string =>
  `fn in_window(col: u32, limit: u32) -> bool {
  return col <= limit${sliding ? ` && (limit - col) < ${uniform}.window` : ""};
}`;

/**
 * Derives the **valid row count** in a row block from the logical lengths (the WGSL-side mirror of
 * {@link stateEffectiveRows}).
 *
 * MUST: the host (dispatch count) and the WGSL (row range) come from **the same single formula**.
 * If they drift, the rows ①② write and the rows ③ treats as pad disagree, and unwritten S / stats
 * are read as a silent wrong value (neither an exception nor NaN).
 * MUST: the subtraction is guarded by a branch to avoid underflow (`query - row_offset` wraps
 * around in u32).
 * NOTE: all three kernels name the params fields `rows_block` / `row_offset`, so this one string can
 * be shared as is. Only ①ₜ has its row count in the skeleton's `dims.m` (the first 3 words of Dims
 * are fixed as `m` / `n` / `k`), so the field name is a parameter too.
 */
export const stateEffectiveRowsWgsl = (uniform = STATE_UNIFORM, rows = "rows_block"): string =>
  `fn effective_rows(query: u32) -> u32 {
  if (query <= ${uniform}.row_offset) {
    return 0u;
  }
  return min(${uniform}.${rows}, query - ${uniform}.row_offset);
}`;

/**
 * Dot product of ①QK (**the half-scale contract** — `scale` is applied to **both** the q side and
 * the k side before multiplying).
 *
 * MUST: generation has a single body (this function). K comes from two sources, the slot and ins,
 * so two function bodies are generated, but with the expression written in two places, fixing only
 * one into "multiply once after the dot product" would make columns from the slot and columns from
 * ins round differently.
 * MUST: ①''s **partial sums** (with `lanes` — `d = lane, lane + lanes, …`) come from the same
 * single place too. Writing the product expression separately for ① and ①' would, when only one is
 * changed to fold the half scale, silently break the basis of the A/B tolerance ("the same products
 * summed in a different order"). Only the **start and step** of the scan change.
 */
const stateScoreFn = (name: string, array: string, lanes?: number): string =>
  `fn ${name}(q_base: u32, k_base: u32${lanes === undefined ? "" : ", lane: u32"}) -> f32 {
  var acc = 0.0;
  for (var d = ${lanes === undefined ? "0u" : "lane"}; d < params.depth; d = d + ${
    lanes === undefined ? "1u" : `${lanes}u`
  }) {
    acc = acc + (q[q_base + d] * params.scale) * (${array}[k_base + d] * params.scale);
  }
  return acc;
}`;

/**
 * Mapping to the kv plane (for `z = b·H + h`, `z / r = b·Hkv + h/r` holds exactly under integer
 * division — the same identity as ADR 0067 decision 2).
 *
 * SHOULD: non-GQA variants do not generate the division at all (no integer division is left on the
 * `r = 1` path) (re-graded 2026-10-18 — ADR 0067 decision 2). The range check
 * ({@link assertStateGeometry}) guarantees `r ≥ 1`, so the GQA variant never divides by zero.
 */
export const kvPlaneWgsl = (gqa: boolean, uniform = STATE_UNIFORM): string =>
  gqa ? `z / ${uniform}.kv_repeat` : "z";

/**
 * Name of the uniform variable read by ①ₜ (the tiled path of the GEMM skeleton —
 * `attentionStateQkTiledWgsl` in src/kernels/gemm.ts). The skeleton puts `dims: Dims` at binding 0,
 * so only the name differs from ①'s `params`.
 */
const STATE_TILED_UNIFORM = "dims";

/**
 * The set of predicate and geometry functions ①ₜ uses (`slot_row` / `column_base` /
 * `live_columns` / `in_window` / `effective_rows`).
 *
 * MUST: generation uses **the same four fragments** as ① (this function only swaps the uniform name
 * and the field name of the row count). Copying the expressions for ①ₜ would, when the ring
 * mapping, the window's lower bound or the live cut is fixed on one side only, let reads and writes
 * drift **with neither an exception nor NaN**.
 * NOTE: the row count of `effective_rows` is the skeleton's `dims.m` (= `rows_block`). The first 3
 * words of Dims are fixed as `m` / `n` / `k`, so this is the one field whose states name cannot be
 * used as is.
 */
export const stateTiledGeometryWgsl = (sliding: boolean): string =>
  `${stateSlotRowWgsl(sliding, STATE_TILED_UNIFORM)}

${stateLiveWgsl(sliding, STATE_TILED_UNIFORM)}

${stateWindowFn(sliding, STATE_TILED_UNIFORM)}

${stateEffectiveRowsWgsl(STATE_TILED_UNIFORM, "m")}`;

/**
 * The set of predicate and geometry functions ③ₜ uses (`slot_row` / `column_base` /
 * `live_columns` / `effective_rows`). It is ①ₜ's {@link stateTiledGeometryWgsl} **without
 * `in_window`**, generated from the same fragments.
 *
 * WHY: the ③ family does not evaluate the predicate itself — ① has written −inf into S for the
 * columns outside the predicate, and they drop out because `p = exp(−inf − m)·inv = 0` comes out
 * exactly (ADR 0067 decision 6). Leaving an unused `in_window` in the output would make readers
 * think "③ₜ also cuts the window".
 */
export const stateTiledPvGeometryWgsl = (sliding: boolean): string =>
  `${stateSlotRowWgsl(sliding, STATE_TILED_UNIFORM)}

${stateLiveWgsl(sliding, STATE_TILED_UNIFORM)}

${stateEffectiveRowsWgsl(STATE_TILED_UNIFORM, "m")}`;

/** kv plane of ①ₜ / ③ₜ ({@link kvPlaneWgsl} with the uniform swapped — `z` is an alias of `wid.z`). */
export const stateTiledKvPlaneWgsl = (gqa: boolean): string =>
  kvPlaneWgsl(gqa, STATE_TILED_UNIFORM);

/**
 * The 7 words ①ₜ appends to the skeleton's `Dims` (`{m, n, k}`). The first 3 words are
 * **`m` = `rows_block` / `n` = `col_cap` / `k` = `depth`**, exactly the M / N / K of the matrix
 * product (the effective N is the run-time live, so `col_cap` serves only as **the static upper
 * bound of the stride**).
 *
 * MUST: the order pairs with {@link stateQkTiledParams} (these two are the only pair — the same
 * rule as `ROW_WINDOW_DIMS_EXTRA` / `GQA_DIMS_EXTRA` on the gemm.ts side).
 * MUST: it may use **a different order** from {@link STATE_PARAMS_STRUCT}, which ① ③ share (①ₜ is a
 * swappable kernel whose binding numbers alone match ①, and it has params of its own). Conversely
 * the word order of `STATE_PARAMS_STRUCT` does not move — ③ and the estimate read the same table.
 * MUST: no field comes or goes with the variant (sliding / GQA). Using one struct for every variant
 * keeps the host-side params assembly to one function (with variant-dependent fields, writing
 * params whose word positions are shifted for one variant only would raise no exception).
 */
export const STATE_QK_TILED_DIMS_EXTRA = `  row_offset: u32,
  chunk_rows: u32,
  kv_repeat: u32,
  window: u32,
  capacity: u32,
  neg_inf: u32,
  scale: f32,
`;

/**
 * The 5 words ③ₜ appends to the skeleton's `Dims` (`{m, n, k}`). The first 3 words are
 * **`m` = `rows_block` / `n` = `depth` / `k` = `col_cap`**, the M / N / K of the matrix product (the
 * effective K is the run-time live, so `col_cap` serves only as **S's row stride and the static
 * upper bound of K**).
 *
 * MUST: the order pairs with {@link statePvTiledParams} (these two are the only pair).
 * MUST: **drop** `neg_inf` / `scale` from ①ₜ's {@link STATE_QK_TILED_DIMS_EXTRA} (the ③ family reads
 * neither — `exp` merely turns the −inf ① baked into S into 0). Keeping words that are not read
 * would grow the misreading "③ₜ also has a predicate and a half scale" on the params side too.
 * MUST: no field comes or goes with the variant (sliding / GQA) (the same wording as ①ₜ — the
 * host-side params assembly stays one function).
 */
export const STATE_PV_TILED_DIMS_EXTRA = `  row_offset: u32,
  chunk_rows: u32,
  kv_repeat: u32,
  window: u32,
  capacity: u32,
`;

/**
 * ①QK. 1 invocation = one element of S (`(local row, live column)`); the dot product is a
 * sequential loop over D.
 *
 * Bindings (**binding 0 = static params, last = logical lengths**):
 *
 * | binding | Resource                                  |
 * | ------- | ----------------------------------------- |
 * | 0       | `Params` (uniform)                        |
 * | 1       | `q` `[B,H,M,D]`                           |
 * | 2       | `ins_k` `[B,Hkv,M,D]`                     |
 * | 3       | `slot_k` `[B,Hkv,C,D]`                    |
 * | 4       | `s` `[B·H, rows_block, col_cap]` (write)  |
 * | 5       | `Lengths` (uniform — owned by context)    |
 */
export const stateQkWgsl = (sliding: boolean, gqa: boolean): string =>
  `// karume attention_state_qk (states-form S materialisation, f32${sliding ? ", sliding window" : ""}${
    gqa ? ", GQA" : ""
  })
${STATE_PARAMS_STRUCT}
${STATE_LENGTHS_STRUCT}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> ins_k: array<f32>;
@group(0) @binding(3) var<storage, read> slot_k: array<f32>;
@group(0) @binding(4) var<storage, read_write> s: array<f32>;
@group(0) @binding(5) var<uniform> lengths: Lengths;

${stateSlotRowWgsl(sliding)}

${stateLiveWgsl(sliding)}

${stateWindowFn(sliding)}

${stateEffectiveRowsWgsl()}

${stateScoreFn("score_slot", "slot_k")}

${stateScoreFn("score_ins", "ins_k")}

@compute @workgroup_size(${STATE_ATTENTION_TILE_X}, ${STATE_ATTENTION_TILE_M})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let past = lengths.past;
  let live = live_columns(past, lengths.query);
  let local_row = gid.y;
  let cl = gid.x;
  // Empty run of the edge tile. Leaving the leftovers right of live ([live, col_cap)) in place is
  // correct and pairs with the readers (②③) cutting at live. Rows go **up to the valid rows** (no
  // one reads the S of pad rows — ③ writes 0 and returns), so the work is proportional to Q
  if (local_row >= effective_rows(lengths.query) || cl >= live) {
    return;
  }
  let z = gid.z;
  let col = column_base(past) + cl;
  let row = params.row_offset + local_row;
  let q_base = (z * params.chunk_rows + row) * params.depth;
  let kv_plane = ${kvPlaneWgsl(gqa)};
  // -inf outside the predicate. The live range is **always written, even outside the predicate**
  // (otherwise ② consumes the previous leftovers)
  var value = bitcast<f32>(params.neg_inf);
  if (in_window(col, past + row)) {
    if (col < past) {
      // past (col < P) comes from the slot. The physical row is slot_row (same formula for reads
      // and writes)
      value = score_slot(q_base, (kv_plane * params.capacity + slot_row(col)) * params.depth);
    } else {
      // current (col ≥ P) comes from row col − P of this step's ins
      value = score_ins(q_base, (kv_plane * params.chunk_rows + (col - past)) * params.depth);
    }
  }
  s[(z * params.rows_block + local_row) * params.col_cap + cl] = value;
}
`;

/**
 * ①' D-parallel reduction variant (the section "①' D-parallel reduction variant" at the top of the
 * file). Bindings and params are **the same** as ① (swappable — the caller switches only the key,
 * the WGSL and the workgroup computation).
 *
 * 1 workgroup = 1 local row × `TILE_X` columns `cl`. Lane `lane` accumulates `d = lane, lane +
 * D_LANES, …` in ascending order, puts the result in `scratch[lane][x]`, and after the fixed-order
 * tree fold **lane 0 writes S**.
 *
 * MUST: the write conditions are the same as ① — with `cl < live`, `-inf` is written even outside
 * the predicate, and for `cl ≥ live` and pad rows not a single word is written. `cl ≥ live` is not
 * workgroup-uniform (the edge tile), so those lanes **do not return**; they run the dot product
 * empty and take part in the barrier.
 */
export const stateQkParallelWgsl = (sliding: boolean, gqa: boolean): string =>
  `// karume attention_state_qk (states-form S materialisation, f32, D-parallel reduction${
    sliding ? ", sliding window" : ""
  }${gqa ? ", GQA" : ""})
${STATE_PARAMS_STRUCT}
${STATE_LENGTHS_STRUCT}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> ins_k: array<f32>;
@group(0) @binding(3) var<storage, read> slot_k: array<f32>;
@group(0) @binding(4) var<storage, read_write> s: array<f32>;
@group(0) @binding(5) var<uniform> lengths: Lengths;

${stateSlotRowWgsl(sliding)}

${stateLiveWgsl(sliding)}

${stateWindowFn(sliding)}

${stateEffectiveRowsWgsl()}

${stateScoreFn("score_slot", "slot_k", STATE_QK_D_LANES)}

${stateScoreFn("score_ins", "ins_k", STATE_QK_D_LANES)}

var<workgroup> scratch: array<f32, ${STATE_ATTENTION_TILE_X * STATE_QK_D_LANES}>;

@compute @workgroup_size(${STATE_ATTENTION_TILE_X}, ${STATE_QK_D_LANES})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let local_row = wid.y;
  // No one reads the S of pad rows (③ writes 0 and returns), so return without writing a word. The
  // local row is workgroup-uniform, so returning before the barrier is allowed
  if (local_row >= effective_rows(lengths.query)) {
    return;
  }
  let past = lengths.past;
  let live = live_columns(past, lengths.query);
  let cl = wid.x * ${STATE_ATTENTION_TILE_X}u + lid.x;
  let lane = lid.y;
  let z = wid.z;
  let col = column_base(past) + cl;
  let row = params.row_offset + local_row;
  let q_base = (z * params.chunk_rows + row) * params.depth;
  let kv_plane = ${kvPlaneWgsl(gqa)};
  // Edge-tile columns (cl ≥ live) and columns outside the predicate do not run the dot product.
  // **No return** — this branch is not workgroup-uniform, and the barrier below can sit only inside
  // uniform control flow (contributes 0.0)
  let inside = cl < live && in_window(col, past + row);
  var acc = 0.0;
  if (inside) {
    if (col < past) {
      // past (col < P) comes from the slot. The physical row is slot_row (same formula for reads
      // and writes)
      acc = score_slot(q_base, (kv_plane * params.capacity + slot_row(col)) * params.depth, lane);
    } else {
      // current (col ≥ P) comes from row col − P of this step's ins
      acc = score_ins(q_base, (kv_plane * params.chunk_rows + (col - past)) * params.depth, lane);
    }
  }
  scratch[lane * ${STATE_ATTENTION_TILE_X}u + lid.x] = acc;
  workgroupBarrier();
  // Fixed-order tree reduction (stride 8 → 4 → 2 → 1) — the basis of determinism
  var stride = ${STATE_QK_D_LANES / 2}u;
  while (stride > 0u) {
    if (lane < stride) {
      let mine = lane * ${STATE_ATTENTION_TILE_X}u + lid.x;
      scratch[mine] = scratch[mine] + scratch[(lane + stride) * ${STATE_ATTENTION_TILE_X}u + lid.x];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }
  // -inf outside the predicate. The live range is **always written, even outside the predicate**
  // (otherwise ② consumes the previous leftovers)
  if (lane == 0u && cl < live) {
    var value = bitcast<f32>(params.neg_inf);
    if (inside) {
      value = scratch[lid.x];
    }
    s[(z * params.rows_block + local_row) * params.col_cap + cl] = value;
  }
}
`;

/**
 * The fragments swapped by the two entry points of ② row statistics (states form
 * {@link stateStatsWgsl} / readonly {@link stateStatsReadonlyWgsl}). Bindings, params, reduction
 * order and barrier positions do not differ by a single byte between the two; the only differences
 * are **the header note, the live-column prologue and the WGSL comments**, so only those are
 * parameters.
 *
 * MUST: the generated WGSL is **frozen by snapshots, comments included** (`attention_state_stats*.wgsl`
 * of `codegen_wgsl_test.ts`). Notes are passed with their leading indentation and trailing newline
 * (an empty string removes the whole line).
 */
type StateStatsShape = {
  /** Note appended right after `f32` on the header line (readonly: a note that only past is read). */
  readonly headerNote: string;
  /** Definitions of `column_base` / `live_columns` (states form: past+query, readonly: past only). */
  readonly live: string;
  /** Right before `let rows` — valid rows are packed at the front, and why no division by 0 occurs. */
  readonly rowsNote: string;
  /** Right before the `workgroupBarrier()` after reading the row max — why reusing scratch syncs. */
  readonly amaxBarrierNote: string;
  /** Heading of the Σ exp stage — why empty rows never run it. */
  readonly sumNote: string;
  /** Right before writing `stats` — why an empty row becomes (0.0, 0.0). */
  readonly emptyStatsNote: string;
  /** Right before the `workgroupBarrier()` closing the row loop — why sync before the next row. */
  readonly tailBarrierNote: string;
};

/**
 * ② row statistics. 1 row = 1 workgroup (256) + grid-stride over rows (the same skeleton as the
 * existing attention_stats; the one difference is that **dim is a run-time value** (the live column
 * count). The identity −inf and the empty-row guard have the same form in the existing ②'s v2).
 *
 * Bindings:
 *
 * | binding | Resource                                              |
 * | ------- | ----------------------------------------------------- |
 * | 0       | `Params` (uniform — a separate struct of the 6 words below) |
 * | 1       | `s` (read)                                            |
 * | 2       | `stats` `[B·H·rows_block, 2]` (write)                 |
 * | 3       | `Lengths` (uniform)                                   |
 *
 * MUST: it covers **valid rows only** (`B·H × effective_rows`). Statistics are written to the same
 * `z · rows_block + local row` as S, and the entries of pad rows stay unwritten (they have no
 * reader — ③ writes 0 for pad rows without reading a single word of `stats`).
 * MUST: `col_cap` is **a stride**, and the scan goes up to live.
 * MUST: the row max is `nan_max` (a bit-pattern NaN check — ADR 0020). The plain `max` drops NaN at
 * the spec level, turning an all-NaN row into an empty row (stats (0,0)). With nan_max, m = NaN
 * reaches ③ and the NaN class of the output is preserved. Bit-identical for non-NaN input (v2).
 */
const stateStatsBodyWgsl = (sliding: boolean, shape: StateStatsShape): string =>
  `// karume attention_state_stats (states-form row statistics m = amax(S) and inv = 1/Σexp(S - m), f32${shape.headerNote}${
    sliding ? ", sliding window" : ""
  })
struct Params {
  batch_heads: u32,
  rows_block: u32,
  row_offset: u32,
  col_cap: u32,
  window: u32,
  neg_inf: u32,
}
${STATE_LENGTHS_STRUCT}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> s: array<f32>;
@group(0) @binding(2) var<storage, read_write> stats: array<f32>;
@group(0) @binding(3) var<uniform> lengths: Lengths;

${shape.live}

${stateEffectiveRowsWgsl()}

${IS_NAN_BITS_WGSL}

${NAN_MAX_WGSL}

var<workgroup> scratch: array<f32, ${STATE_STATS_WORKGROUP_SIZE}>;

@compute @workgroup_size(${STATE_STATS_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid3: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let lid = lid3.x;
  let neg_inf = bitcast<f32>(params.neg_inf);
  let live = live_columns(lengths.past, lengths.query);
${shape.rowsNote}  let rows = effective_rows(lengths.query);
  let total = params.batch_heads * rows;
  var index = wid.x;
  while (index < total) {
    let row = (index / rows) * params.rows_block + index % rows;
    let base = row * params.col_cap;

    // ① Row maximum. The identity is **-inf** (a finite sentinel is MUST NOT — ADR 0067 decision 6)
    var hi = neg_inf;
    var i = lid;
    while (i < live) {
      hi = nan_max(hi, s[base + i]);
      i = i + ${STATE_STATS_WORKGROUP_SIZE}u;
    }
    scratch[lid] = hi;
    workgroupBarrier();
    var stride = ${STATE_STATS_WORKGROUP_SIZE / 2}u;
    while (stride > 0u) {
      if (lid < stride) {
        scratch[lid] = nan_max(scratch[lid], scratch[lid + stride]);
      }
      workgroupBarrier();
      stride = stride / 2u;
    }
    let amax = scratch[0u];
${shape.amaxBarrierNote}    workgroupBarrier();

${shape.sumNote}    let empty = amax == neg_inf;
    var acc = 0.0;
    if (!empty) {
      var j = lid;
      while (j < live) {
        acc = acc + exp(s[base + j] - amax);
        j = j + ${STATE_STATS_WORKGROUP_SIZE}u;
      }
    }
    scratch[lid] = acc;
    workgroupBarrier();
    var stride2 = ${STATE_STATS_WORKGROUP_SIZE / 2}u;
    while (stride2 > 0u) {
      if (lid < stride2) {
        scratch[lid] = scratch[lid] + scratch[lid + stride2];
      }
      workgroupBarrier();
      stride2 = stride2 / 2u;
    }
    if (lid == 0u) {
${shape.emptyStatsNote}      var m = 0.0;
      var inv = 0.0;
      if (!empty) {
        m = amax;
        inv = 1.0 / scratch[0u];
      }
      stats[row * ${STATE_STATS_STRIDE}u] = m;
      stats[row * ${STATE_STATS_STRIDE}u + 1u] = inv;
    }
${shape.tailBarrierNote}    workgroupBarrier();
    index = index + nwg.x;
  }
}
`;

/** ② row statistics (states form — live is past+query). Contract: {@link stateStatsBodyWgsl}. */
export const stateStatsWgsl = (sliding: boolean): string =>
  stateStatsBodyWgsl(sliding, {
    headerNote: "",
    live: stateLiveWgsl(sliding),
    rowsNote: `  // The valid rows are the first rows of each z plane (**packed at the front**). With
  // total = 0 the loop is not entered, so dividing by rows or taking the remainder never divides
  // by 0 (the host records no dispatch for 0 either)
`,
    amaxBarrierNote: `    // Make sure scratch has been read before ② overwrites it
`,
    sumNote: `    // ② Σ exp(S - amax). **An empty row (amax == -inf) never runs it** —
    // exp(-inf - (-inf)) = exp(NaN) = NaN would enter the denominator
`,
    emptyStatsNote: `      // An empty row is (0.0, 0.0). ③'s exp(-inf - 0) * 0 = 0 makes the output
      // **exactly 0**
`,
    tailBarrierNote: `    // Make sure scratch[0] has been read before the next row overwrites scratch[lid]
`,
  });

/**
 * ③PV. 1 invocation = one element of O (`(local row, D)`); the reduction is **a sequential pass in
 * ascending live-column order** (determinism).
 *
 * Bindings:
 *
 * | binding | Resource                                   |
 * | ------- | ------------------------------------------ |
 * | 0       | `Params` (uniform — the same struct as ①)  |
 * | 1       | `s` (read)                                 |
 * | 2       | `stats` (read)                             |
 * | 3       | `ins_v` `[B,Hkv,M,D]`                      |
 * | 4       | `slot_v` `[B,Hkv,C,D]`                     |
 * | 5       | `out` `[B,H,M,D]` (write)                  |
 * | 6       | `Lengths` (uniform)                        |
 *
 * MUST: the output writes **all** `rows_block` rows from `row_offset` (pad rows (`row ≥ Q`) are
 * written as ordinary output too — ADR 0066 addendum 6's "undefined = the value is meaningless by
 * contract", not unwritten). But pad rows get **an exact `0.0` without scanning a single live
 * column** (① ② cover valid rows only, so there is no S / stats for them + the work stays
 * proportional to Q + empty row → 0 is contained by construction).
 */
export const statePvWgsl = (sliding: boolean, gqa: boolean): string =>
  `// karume attention_state_pv (states-form O = P @ V, f32${sliding ? ", sliding window" : ""}${
    gqa ? ", GQA" : ""
  })
${STATE_PARAMS_STRUCT}
${STATE_LENGTHS_STRUCT}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> s: array<f32>;
@group(0) @binding(2) var<storage, read> stats: array<f32>;
@group(0) @binding(3) var<storage, read> ins_v: array<f32>;
@group(0) @binding(4) var<storage, read> slot_v: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;
@group(0) @binding(6) var<uniform> lengths: Lengths;

${stateSlotRowWgsl(sliding)}

${stateLiveWgsl(sliding)}

${stateEffectiveRowsWgsl()}

@compute @workgroup_size(${STATE_ATTENTION_TILE_X}, ${STATE_ATTENTION_TILE_M})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let d = gid.x;
  let local_row = gid.y;
  if (d >= params.depth || local_row >= params.rows_block) {
    return;
  }
  let z = gid.z;
  let at = (z * params.chunk_rows + params.row_offset + local_row) * params.depth + d;
  // Pad rows (row ≥ Q): write **an exact 0** without scanning live and return (full-write holds =
  // ADR 0066 addendum 6's "the value is meaningless by contract" is pinned to 0). Empty rows ⊂ pad
  // rows, so this branch contains ADR 0067 decision 6's "empty row → exact 0" by construction and
  // also closes the 0·NaN hole of a non-finite V
  if (local_row >= effective_rows(lengths.query)) {
    out[at] = 0.0;
    return;
  }
  let past = lengths.past;
  let live = live_columns(past, lengths.query);
  let base_col = column_base(past);
  let kv_plane = ${kvPlaneWgsl(gqa)};
  let s_row = z * params.rows_block + local_row;
  let s_base = s_row * params.col_cap;
  let amax = stats[s_row * ${STATE_STATS_STRIDE}u];
  let inv = stats[s_row * ${STATE_STATS_STRIDE}u + 1u];
  // The reduction is a sequential pass in ascending col (determinism). Outside the predicate S is
  // -inf, so p = 0 comes out **exactly**
  var acc = 0.0;
  for (var cl = 0u; cl < live; cl = cl + 1u) {
    let col = base_col + cl;
    let p = exp(s[s_base + cl] - amax) * inv;
    var value = 0.0;
    if (col < past) {
      value = slot_v[(kv_plane * params.capacity + slot_row(col)) * params.depth + d];
    } else {
      value = ins_v[(kv_plane * params.chunk_rows + (col - past)) * params.depth + d];
    }
    acc = acc + p * value;
  }
  out[at] = acc;
}
`;

/**
 * ③' KV-parallel reduction variant (the section "③' KV-parallel reduction variant" at the top of
 * the file). Bindings and params are **the same** as ③ (swappable — the caller switches only the
 * key and the workgroup computation).
 *
 * 1 workgroup = 1 local row × `TILE_X` values of `d`. Lane `lane` accumulates `cl = lane, lane +
 * KV_LANES, …` in ascending order, puts the result in `scratch[lane][x]`, and folds it with the
 * fixed-order tree.
 */
export const statePvParallelWgsl = (sliding: boolean, gqa: boolean): string =>
  `// karume attention_state_pv (states-form O = P @ V, f32, KV-parallel reduction${
    sliding ? ", sliding window" : ""
  }${gqa ? ", GQA" : ""})
${STATE_PARAMS_STRUCT}
${STATE_LENGTHS_STRUCT}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> s: array<f32>;
@group(0) @binding(2) var<storage, read> stats: array<f32>;
@group(0) @binding(3) var<storage, read> ins_v: array<f32>;
@group(0) @binding(4) var<storage, read> slot_v: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;
@group(0) @binding(6) var<uniform> lengths: Lengths;

${stateSlotRowWgsl(sliding)}

${stateLiveWgsl(sliding)}

${stateEffectiveRowsWgsl()}

var<workgroup> scratch: array<f32, ${STATE_ATTENTION_TILE_X * STATE_PV_KV_LANES}>;

@compute @workgroup_size(${STATE_ATTENTION_TILE_X}, ${STATE_PV_KV_LANES})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let d = wid.x * ${STATE_ATTENTION_TILE_X}u + lid.x;
  let local_row = wid.y;
  let lane = lid.y;
  let z = wid.z;
  let in_depth = d < params.depth;
  let at = (z * params.chunk_rows + params.row_offset + local_row) * params.depth + d;
  // Pad rows (row ≥ Q) do not scan live and are exactly 0 (the same contract as ③). The local row is
  // workgroup-uniform, so returning before the barrier is allowed. KV_LANES lanes point at the same
  // out[at], so as on ordinary rows only lane 0 writes (unsynchronised writes to the same element are
  // a WGSL data race even with equal values)
  if (local_row >= effective_rows(lengths.query)) {
    if (lane == 0u && in_depth) {
      out[at] = 0.0;
    }
    return;
  }
  let past = lengths.past;
  let live = live_columns(past, lengths.query);
  let base_col = column_base(past);
  let kv_plane = ${kvPlaneWgsl(gqa)};
  let s_row = z * params.rows_block + local_row;
  let s_base = s_row * params.col_cap;
  let amax = stats[s_row * ${STATE_STATS_STRIDE}u];
  let inv = stats[s_row * ${STATE_STATS_STRIDE}u + 1u];
  // Per-lane partial sum (ascending col, stride KV_LANES). A lane whose d is out of range does not
  // scan and contributes 0 (it does not return, so that it takes part in the barrier)
  var acc = 0.0;
  if (in_depth) {
    for (var cl = lane; cl < live; cl = cl + ${STATE_PV_KV_LANES}u) {
      let col = base_col + cl;
      let p = exp(s[s_base + cl] - amax) * inv;
      var value = 0.0;
      if (col < past) {
        value = slot_v[(kv_plane * params.capacity + slot_row(col)) * params.depth + d];
      } else {
        value = ins_v[(kv_plane * params.chunk_rows + (col - past)) * params.depth + d];
      }
      acc = acc + p * value;
    }
  }
  scratch[lane * ${STATE_ATTENTION_TILE_X}u + lid.x] = acc;
  workgroupBarrier();
  // Fixed-order tree reduction (stride 8 → 4 → 2 → 1) — the basis of determinism
  var stride = ${STATE_PV_KV_LANES / 2}u;
  while (stride > 0u) {
    if (lane < stride) {
      let mine = lane * ${STATE_ATTENTION_TILE_X}u + lid.x;
      scratch[mine] = scratch[mine] + scratch[(lane + stride) * ${STATE_ATTENTION_TILE_X}u + lid.x];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }
  if (lane == 0u && in_depth) {
    out[at] = scratch[lid.x];
  }
}
`;

/** Static geometry shared by ①③ (the type side of the word order of {@link STATE_PARAMS_STRUCT}). */
export type StateAttentionGeometry = {
  /** Rows this dispatch handles (rows of S — `rows_block`). */
  readonly rowsBlock: number;
  /** First row in the chunk. Global row = `rowOffset` + local row. */
  readonly rowOffset: number;
  /** `M` (row stride of q / ins / O = physical chunk row count). */
  readonly chunkRows: number;
  /** `D`. */
  readonly depth: number;
  /** `r = H / Hkv` (the GQA repeat count). */
  readonly kvRepeat: number;
  /** `W` (`0` = full — {@link stateSliding}). */
  readonly window: number;
  /** `C` (row capacity of the slot). */
  readonly capacity: number;
  /** Upper bound on the column stride of S. */
  readonly colCap: number;
  /** Half scale (the value applied to both the q side and the k side). */
  readonly scale: number;
};

/**
 * Range check of the static geometry.
 *
 * MUST: `kvRepeat ≥ 1` (WGSL u32 division by zero does not trap and returns an
 * implementation-dependent value, so `r = 0` corrupts the kv plane with neither an exception nor
 * NaN — the same shape as `assertKvRepeat` in attention.ts).
 * MUST: sliding requires `1 ≤ W ≤ C` (ADR 0067 decision 4 ③). Dispatching sliding WGSL with
 * `W = 0` underflows `window - 1u` and makes `col % 0u` an implementation-dependent value.
 * MUST: `colCap` is at least **the static upper bound** of the live columns (`C` for full, from the
 * context-side check `P + Q ≤ C`; `(W−1) + M` for sliding). If it is short, ①'s write
 * `s[(z·rows_block + local_row)·col_cap + cl]` for columns with `cl ≥ col_cap` **silently overwrites
 * the S of the next row** for every row except the last (only the last row falls out of range and
 * is discarded). ②③ misread the next row as their own columns through the same index. There is
 * neither an exception nor NaN, so this check is MUST.
 * MUST: `rowOffset + rowsBlock ≤ M` (the row block does not run past the chunk).
 */
const assertStateGeometry = (where: string, geometry: StateAttentionGeometry): void => {
  const { rowsBlock, rowOffset, chunkRows, depth, kvRepeat, window, capacity, colCap } = geometry;
  assertU32Params(where, {
    rows_block: rowsBlock,
    row_offset: rowOffset,
    chunk_rows: chunkRows,
    depth,
    kv_repeat: kvRepeat,
    window,
    capacity,
    col_cap: colCap,
  });
  if (rowsBlock < 1 || chunkRows < 1 || depth < 1 || capacity < 1 || colCap < 1) {
    throw new CodegenError(
      `${where}: rows_block / chunk_rows / depth / capacity / col_cap は正整数` +
        `（${rowsBlock} / ${chunkRows} / ${depth} / ${capacity} / ${colCap}）`,
    );
  }
  if (kvRepeat < 1) {
    throw new CodegenError(`${where}: kv_repeat は正整数（${kvRepeat}）`);
  }
  if (rowOffset + rowsBlock > chunkRows) {
    throw new CodegenError(
      `${where}: 行ブロック [${rowOffset}, ${
        rowOffset + rowsBlock
      }) が chunk 行数 ${chunkRows} を超える`,
    );
  }
  if (stateSliding(window)) {
    if (window > capacity) {
      throw new CodegenError(
        `${where}: window ${window} が容量 ${capacity} を超える（ADR 0067 決定 4 ③）`,
      );
    }
    const need = window - 1 + chunkRows;
    if (colCap < need) {
      throw new CodegenError(
        `${where}: col_cap ${colCap} が sliding の live 上限 ${need}（= W−1 + M）に足りない`,
      );
    }
  } else if (colCap < capacity) {
    throw new CodegenError(
      `${where}: col_cap ${colCap} が full の live 上限 ${capacity}（= C）に足りない`,
    );
  }
  // MUST: check finiteness **as f32** (the same check as the attention_qk params — `1e39`, finite in
  // f64, becomes `+Inf` in an f32 word, and `0 * Inf = NaN` silently breaks the scores).
  if (!Number.isFinite(Math.fround(geometry.scale))) {
    throw new CodegenError(`${where}: scale は f32 として有限の数値（${geometry.scale}）`);
  }
};

/**
 * Uniform shared by ①③ (the 10 words of {@link STATE_PARAMS_STRUCT} rounded up to 16-byte units,
 * 48 bytes allocated — allocating in 16-byte units is a project convention (the fusion pass fixes
 * the same form in ADR 0040), not a requirement of the spec).
 */
export const stateAttentionParams = (
  geometry: StateAttentionGeometry,
): Uint32Array<ArrayBuffer> => {
  assertStateGeometry("attention_state params", geometry);
  const params = new Uint32Array(12);
  params[0] = geometry.rowsBlock;
  params[1] = geometry.rowOffset;
  params[2] = geometry.chunkRows;
  params[3] = geometry.depth;
  params[4] = geometry.kvRepeat;
  params[5] = geometry.window;
  params[6] = geometry.capacity;
  params[7] = geometry.colCap;
  params[8] = STATE_NEG_INF_BITS;
  new Float32Array(params.buffer)[9] = geometry.scale;
  return params;
};

/**
 * Uniform of ①ₜ (10 words: {@link STATE_QK_TILED_DIMS_EXTRA} plus the skeleton's `{m, n, k}` —
 * rounded up to 16-byte units, 48 bytes allocated — the reason is at {@link stateAttentionParams}).
 *
 * MUST: the geometry range check is **the same single one** as ① ({@link assertStateGeometry}). ①ₜ
 * comes from the same static geometry as ①, so a separate check would allow a shape that passes
 * only one of them.
 * MUST: the order pairs with {@link STATE_QK_TILED_DIMS_EXTRA} (these two are the only pair).
 */
export const stateQkTiledParams = (geometry: StateAttentionGeometry): Uint32Array<ArrayBuffer> => {
  assertStateGeometry("attention_state_qk (tiled) params", geometry);
  const params = new Uint32Array(12);
  params[0] = geometry.rowsBlock;
  params[1] = geometry.colCap;
  params[2] = geometry.depth;
  params[3] = geometry.rowOffset;
  params[4] = geometry.chunkRows;
  params[5] = geometry.kvRepeat;
  params[6] = geometry.window;
  params[7] = geometry.capacity;
  params[8] = STATE_NEG_INF_BITS;
  new Float32Array(params.buffer)[9] = geometry.scale;
  return params;
};

/**
 * Uniform of ③ₜ (8 words: {@link STATE_PV_TILED_DIMS_EXTRA} plus the skeleton's `{m, n, k}` =
 * exactly 32 bytes — the reason for 16-byte units is at {@link stateAttentionParams}).
 *
 * MUST: the geometry range check is **the same single one** as ③ ({@link assertStateGeometry}). ③ₜ
 * comes from the same static geometry as ③, so a separate check would allow a shape that passes
 * only one of them.
 * MUST: the order pairs with {@link STATE_PV_TILED_DIMS_EXTRA} (these two are the only pair).
 */
export const statePvTiledParams = (geometry: StateAttentionGeometry): Uint32Array<ArrayBuffer> => {
  assertStateGeometry("attention_state_pv (tiled) params", geometry);
  const params = new Uint32Array(8);
  params[0] = geometry.rowsBlock;
  params[1] = geometry.depth;
  params[2] = geometry.colCap;
  params[3] = geometry.rowOffset;
  params[4] = geometry.chunkRows;
  params[5] = geometry.kvRepeat;
  params[6] = geometry.window;
  params[7] = geometry.capacity;
  return params;
};

/**
 * Uniform of ② (the 6 words `{batch_heads, rows_block, row_offset, col_cap, window, neg_inf}` —
 * rounded up to 16-byte units, 32 bytes allocated — the reason is at {@link stateAttentionParams}).
 *
 * MUST: `B·H` and `rows_block` are taken **without folding them** (the same two axes as ① ③). A
 * single folded word cannot be turned back into `z · rows_block + local row`, so the rows of S /
 * stats could not be found from the front-packed valid rows.
 * MUST: `window` also goes through the range check (②'s side of "key selection, WGSL generation and
 * the params check all agree", which {@link stateSliding} requires). ② does not take `capacity` /
 * `M`, so it cannot write the bound `W−1 + M` ({@link assertStateGeometry}) itself, but it can write
 * the necessary condition `W ≤ col_cap` that follows from `M ≥ 1`. If that breaks, `live_columns`
 * crosses `col_cap` and mixes the S of the adjacent row into this row's amax (a silent wrong value
 * with neither an exception nor NaN).
 */
export const stateStatsParams = (
  batchHeads: number,
  rowsBlock: number,
  rowOffset: number,
  colCap: number,
  window: number,
): Uint32Array<ArrayBuffer> => {
  assertU32Params("attention_state_stats params", {
    batch_heads: batchHeads,
    rows_block: rowsBlock,
    row_offset: rowOffset,
    col_cap: colCap,
    window,
  });
  if (batchHeads < 1 || rowsBlock < 1 || colCap < 1) {
    throw new CodegenError(
      `attention_state_stats params: batch_heads / rows_block / col_cap は正整数` +
        `（${batchHeads} / ${rowsBlock} / ${colCap}）`,
    );
  }
  if (stateSliding(window) && colCap < window) {
    throw new CodegenError(
      `attention_state_stats params: col_cap ${colCap} が sliding の live 上限 ${window}` +
        `（= W−1 + M の最小 M = 1）に足りない`,
    );
  }
  const params = new Uint32Array(8);
  params[0] = batchHeads;
  params[1] = rowsBlock;
  params[2] = rowOffset;
  params[3] = colCap;
  params[4] = window;
  params[5] = STATE_NEG_INF_BITS;
  return params;
};

/**
 * Start of the resident range of a sliding slot (logical col). Always 0 for full.
 *
 * MUST: the host (dispatch count) and the WGSL ({@link stateLiveWgsl}) use **the same formula**. If
 * they drift, the range ① writes and the range ②③ read disagree, and a wider reader side consumes
 * leftovers.
 */
export const stateColumnBase = (window: number, past: number): number =>
  stateSliding(window) ? past - Math.min(past, window - 1) : 0;

/** Live column count `n_live = (P − colBase) + Q` (columns S materialises = the scan length of ②③). */
export const stateLiveColumns = (window: number, past: number, query: number): number =>
  past - stateColumnBase(window, past) + query;

/**
 * **Valid row count** of row block `[rowOffset, rowOffset + rowsBlock)` (the number of local rows
 * with `row < Q`).
 *
 * MUST: the dispatch counts of ①QK / ②stats come from this one function (the workload acceptance
 * condition of ADR 0066 decision 3 — making them proportional to the physical chunk rows `M` would
 * make decode with `Q = 1` run `M` times the rows).
 * MUST: **the same formula** as the WGSL side ({@link stateEffectiveRowsWgsl}). The host is the pure
 * function of the geometry, and the kernel is its mirror.
 */
export const stateEffectiveRows = (rowsBlock: number, rowOffset: number, query: number): number =>
  Math.min(rowsBlock, Math.max(0, query - rowOffset));

/** Range check of `queryLength` (carried as u32; `Q ≥ 1` as in the `GenerationContext` contract). */
const assertQueryLength = (where: string, query: number): void => {
  assertU32Params(where, { query });
  if (query < 1) {
    throw new CodegenError(`${where}: queryLength は 1 以上（${query}）`);
  }
};

/** Range check of the logical lengths (carried as u32; `Q ≥ 1` per the `GenerationContext` contract). */
const assertLengths = (where: string, past: number, query: number): void => {
  assertU32Params(where, { past });
  assertQueryLength(where, query);
};

/**
 * Dispatch geometry (**the side computed from the logical lengths** — ADR 0066 decision 3 / ADR 0067
 * decision 4).
 *
 * MUST: it does **not hold** `capacity` / `colCap`. Holding them would make "a workgroup count
 * proportional to the capacity" writable, silently going back to the form ADR 0066 decision 3
 * rejects: three orders of magnitude of waste with a 131K capacity × a short sequence.
 */
export type StateDispatchGeometry = {
  /** `B·H` (the z axis — 1 workgroup = 1 (b, h)). */
  readonly batchHeads: number;
  readonly rowsBlock: number;
  /** First row in the chunk (needed for the valid row count `clamp(Q − rowOffset, 0, rowsBlock)`). */
  readonly rowOffset: number;
  readonly depth: number;
  /** `W` (`0` = full — the live column count is capped at `min(P, W−1) + Q`). */
  readonly window: number;
};

/** Range check of the dispatch geometry (all three axes positive integers; `rowOffset` may be 0). */
const assertDispatchGeometry = (where: string, geometry: StateDispatchGeometry): void => {
  const { batchHeads, rowsBlock, rowOffset, depth, window } = geometry;
  assertU32Params(where, {
    batch_heads: batchHeads,
    rows_block: rowsBlock,
    row_offset: rowOffset,
    depth,
    window,
  });
  if (batchHeads < 1 || rowsBlock < 1 || depth < 1) {
    throw new CodegenError(
      `${where}: batchHeads / rowsBlock / depth は正整数（${batchHeads} / ${rowsBlock} / ${depth}）`,
    );
  }
};

/**
 * Workgroup count of ①QK `[live columns, valid rows, B·H]`.
 *
 * MUST: the column axis is **the live column count** (neither `colCap` nor `C`), and the row axis is
 * **the valid row count** (neither `rowsBlock` nor `M`). These two axes are the mechanism of the
 * workload acceptance condition (∝ `Q × (valid past + Q)`), and the capacity is left out of
 * {@link StateDispatchGeometry} so that it cannot be passed.
 * MUST: exceeding the limit fails loudly (tiled — clamping would drop tiles of S, and ②③ would read
 * leftovers and carry on without an exception).
 * NOTE: a block with 0 valid rows (`rowOffset ≥ Q`) gets a row axis of 0, and the caller records no
 * dispatch at all (the doc of {@link stateEffectiveRows}).
 */
export const stateQkWorkgroups = (
  geometry: StateDispatchGeometry,
  past: number,
  query: number,
  limit: number,
  where: string,
): [number, number, number] => {
  assertDispatchGeometry(`${where} ①QK`, geometry);
  assertLengths(`${where} ①QK`, past, query);
  const live = stateLiveColumns(geometry.window, past, query);
  const rows = stateEffectiveRows(geometry.rowsBlock, geometry.rowOffset, query);
  return [
    tiledWorkgroups(live, STATE_ATTENTION_TILE_X, limit, `${where} ①QK`),
    tiledWorkgroups(rows, STATE_ATTENTION_TILE_M, limit, `${where} ①QK`),
    tiledWorkgroups(geometry.batchHeads, 1, limit, `${where} ①QK`),
  ];
};

/**
 * Workgroup count of ①' `[⌈live / TILE_X⌉, valid rows, B·H]` (the row axis is **1 row = 1
 * workgroup**).
 *
 * As in ①, the column axis is **the live column count** and the row axis **the valid row count**
 * (`colCap` / `rowsBlock` would fail the workload acceptance condition and write the S of pad rows).
 * Exceeding the limit fails loudly, also as in ①. Only the row tile width changes; the D direction
 * is folded into the lanes inside the workgroup, so it does not show in the dispatch.
 */
export const stateQkParallelWorkgroups = (
  geometry: StateDispatchGeometry,
  past: number,
  query: number,
  limit: number,
  where: string,
): [number, number, number] => {
  assertDispatchGeometry(`${where} ①'QK`, geometry);
  assertLengths(`${where} ①'QK`, past, query);
  const live = stateLiveColumns(geometry.window, past, query);
  const rows = stateEffectiveRows(geometry.rowsBlock, geometry.rowOffset, query);
  return [
    tiledWorkgroups(live, STATE_ATTENTION_TILE_X, limit, `${where} ①'QK`),
    tiledWorkgroups(rows, 1, limit, `${where} ①'QK`),
    tiledWorkgroups(geometry.batchHeads, 1, limit, `${where} ①'QK`),
  ];
};

/**
 * Workgroup count of ①ₜ `[⌈live / tileN⌉, ⌈valid rows / tileM⌉, B·H]` (1 workgroup = 1 output
 * tile).
 *
 * MUST: the tile edges are **derived from the geometry** (the same wording as the gemm skeleton's
 * MUST — carrying a constant around makes two values that can disagree with the geometry, and the
 * moment `ceil(dim / constant)` falls below the count for the real tile edge, **tiles are dropped, a
 * silent wrong value**). The key, generation and this function resolve the geometry, all three
 * through the pure function of the single value `chunkRows` ({@link gemmGeometryForRows}).
 * MUST: as in ①, the column axis is **the live column count** and the row axis **the valid row
 * count** (`colCap` / `rowsBlock` would fail the workload acceptance condition (ADR 0066 decision 3)
 * and write the S of pad rows).
 * MUST: exceeding the limit fails loudly (tiled — clamping would drop tiles of S, and ②③ would read
 * leftovers and carry on without an exception).
 */
export const stateQkTiledWorkgroups = (
  geometry: StateDispatchGeometry,
  chunkRows: number,
  past: number,
  query: number,
  limit: number,
  where: string,
): [number, number, number] => {
  assertDispatchGeometry(`${where} ①ₜQK`, geometry);
  assertLengths(`${where} ①ₜQK`, past, query);
  const tile = gemmGeometryForRows(chunkRows);
  const live = stateLiveColumns(geometry.window, past, query);
  const rows = stateEffectiveRows(geometry.rowsBlock, geometry.rowOffset, query);
  return [
    tiledWorkgroups(live, gemmTileN(tile), limit, `${where} ①ₜQK`),
    tiledWorkgroups(rows, gemmTileM(tile), limit, `${where} ①ₜQK`),
    tiledWorkgroups(geometry.batchHeads, 1, limit, `${where} ①ₜQK`),
  ];
};

/**
 * Workgroup count of ② `[B·H × valid rows, 1, 1]` (grid-stride over rows, so exceeding the limit is
 * **clamped**).
 *
 * It covers valid rows only (no one reads the statistics of pad rows). The live scan is inside the
 * row loop, so the total iteration count is `B·H × valid rows × live`, proportional to
 * `Q × (valid past + Q)` as the workload acceptance condition requires.
 */
export const stateStatsWorkgroups = (
  geometry: StateDispatchGeometry,
  query: number,
  limit: number,
  where: string,
): [number, number, number] => {
  assertDispatchGeometry(`${where} ②stats`, geometry);
  assertQueryLength(`${where} ②stats`, query);
  const rows = stateEffectiveRows(geometry.rowsBlock, geometry.rowOffset, query);
  return [gridStrideWorkgroups(geometry.batchHeads * rows, 1, limit), 1, 1];
};

/**
 * Workgroup count of ③PV `[D, rows, B·H]`.
 *
 * MUST: the row axis is **all of `rows_block`** (unlike ① ②, not cut at the valid rows — writing pad
 * rows too is the full-write invariant). Keeping the work proportional to Q happens in the kernel:
 * a pad-row invocation runs no live column, writes one word of `0.0` and returns.
 * MUST: exceeding the limit fails loudly (tiled — clamping would leave part of O unwritten and
 * silently break the full-write invariant).
 */
export const statePvWorkgroups = (
  geometry: StateDispatchGeometry,
  limit: number,
  where: string,
): [number, number, number] => {
  assertDispatchGeometry(`${where} ③PV`, geometry);
  return [
    tiledWorkgroups(geometry.depth, STATE_ATTENTION_TILE_X, limit, `${where} ③PV`),
    tiledWorkgroups(geometry.rowsBlock, STATE_ATTENTION_TILE_M, limit, `${where} ③PV`),
    tiledWorkgroups(geometry.batchHeads, 1, limit, `${where} ③PV`),
  ];
};

/**
 * Workgroup count of ③' `[D / TILE_X, rows_block, B·H]` (the row axis is **1 row = 1 workgroup**).
 *
 * As in ③, the row axis is all of `rows_block` (pad rows are full-write too), and exceeding the
 * limit fails loudly.
 */
export const statePvParallelWorkgroups = (
  geometry: StateDispatchGeometry,
  limit: number,
  where: string,
): [number, number, number] => {
  assertDispatchGeometry(`${where} ③'PV`, geometry);
  return [
    tiledWorkgroups(geometry.depth, STATE_ATTENTION_TILE_X, limit, `${where} ③'PV`),
    tiledWorkgroups(geometry.rowsBlock, 1, limit, `${where} ③'PV`),
    tiledWorkgroups(geometry.batchHeads, 1, limit, `${where} ③'PV`),
  ];
};

/**
 * Workgroup count of ③ₜ `[⌈D / tileN⌉, ⌈rows_block / tileM⌉, B·H]` (1 workgroup = 1 output tile).
 *
 * MUST: the tile edges are **derived from the geometry** (the same wording as ①ₜ's
 * {@link stateQkTiledWorkgroups} — carrying a constant around makes two values that can disagree
 * with the geometry, and the moment `ceil(dim / constant)` falls below the count for the real tile
 * edge, **tiles are dropped, a silent wrong value**). The key, generation and this function resolve
 * the geometry, all three through the pure function of the single value `chunkRows`
 * ({@link gemmGeometryForRows}).
 * MUST: the row axis is **all of `rows_block`** (unlike the ① family, not cut at the valid rows —
 * writing pad rows too is ③'s full-write invariant). Keeping the work proportional to Q happens in
 * the kernel: a row tile with no valid row runs the K loop zero times and ends by writing `0.0`.
 * MUST: it does not take the logical lengths (like ③ / ③', it is decided by the output shape
 * alone).
 * MUST: exceeding the limit fails loudly (tiled — clamping would leave part of O unwritten and
 * silently break the full-write invariant).
 */
export const statePvTiledWorkgroups = (
  geometry: StateDispatchGeometry,
  chunkRows: number,
  limit: number,
  where: string,
): [number, number, number] => {
  assertDispatchGeometry(`${where} ③ₜPV`, geometry);
  const tile = gemmGeometryForRows(chunkRows);
  return [
    tiledWorkgroups(geometry.depth, gemmTileN(tile), limit, `${where} ③ₜPV`),
    tiledWorkgroups(geometry.rowsBlock, gemmTileM(tile), limit, `${where} ③ₜPV`),
    tiledWorkgroups(geometry.batchHeads, 1, limit, `${where} ③ₜPV`),
  ];
};

// ---- readonly variants (read past only — ADR 0096 decision 1 / the stage 2 drafter) ----------
//
// The drafter (MTP head) has no K/V of its own and reads **only the slots** of the lending context.
// The query is one row at logical position `P` (the uncommitted frontier token), and it reads only
// the past columns `[column_base, P)` — its own column is not in the lender's ring (the ADR 0096
// addendum on stage 3: the drafter's inputs are frontier `b@P`, hidden `h@(P−1)` and RoPE position
// `P`). There is no ins for this step (and no append).
// The skeleton, reduction order and numeric contract of the three steps (①' ② ③') are the same as
// the states form; what changes is **the live-range formula and the predicate, and the absence of
// the ins branch and binding** only.
//
// MUST: the physical-row mapping shares {@link stateSlotRowWgsl} (the same formula for reads and
// writes). A formula different from the lender's append silently points every read at another row
// once the ring has wrapped.
// MUST: the predicate is `col < P` (rejected rows = rows written ahead of the logical length are
// **not read** — the slack of the ADR 0066 addendum of 2026-09-08 only guarantees that "rejected
// rows fall outside the live window"; with `col ≤ P` the unconfirmed row at position P would be
// consumed as past).

/**
 * The readonly live range (`column_base = P − min(P, W)` / `live = min(P, W)`; `0` / `P` for full).
 *
 * It differs from the states form ({@link stateLiveWgsl}) in two points, "no +Q" and "the window is
 * W, not W−1" — the query at position P reads only the past (its own column is not in the ring), so
 * the live range is exactly the W columns `[P−W, P)` (the states form includes its own Q columns in
 * the window, so its past side is W−1). That the oldest column is `P−W` is also the basis of the
 * deferred limit `queryLength ≤ slidingSlack` (the ADR 0096 addendum on stage 3).
 * The `query` parameter is taken only to keep the signature the same as the states form (it is not
 * read).
 */
const stateReadonlyLiveWgsl = (sliding: boolean, uniform = STATE_UNIFORM): string =>
  `fn column_base(past: u32) -> u32 {
  return ${sliding ? `past - min(past, ${uniform}.window)` : "0u"};
}

fn live_columns(past: u32, query: u32) -> u32 {
  return ${sliding ? `min(past, ${uniform}.window)` : "past"};
}`;

/**
 * The readonly predicate (`col < P` AND, with sliding, `(P−1−col) < W`).
 *
 * MUST: the subtraction is on the `past − 1u − col` side and is evaluated only when `col < past` is
 * true (WGSL's `&&` short-circuits — `past ≥ 1` and `col ≤ past − 1`, so it does not wrap around).
 * By construction every column in the live range is inside the predicate, but the predicate stays
 * to keep the same "write −inf outside the predicate" structure as ① (the boundary formula lives in
 * one place).
 */
const stateReadonlyWindowFn = (sliding: boolean, uniform = STATE_UNIFORM): string =>
  `fn in_window(col: u32, past: u32) -> bool {
  return col < past${sliding ? ` && (past - 1u - col) < ${uniform}.window` : ""};
}`;

/** Key of ①' readonly (`:par:ro` — it has one binding fewer, so a key separate from ①' is MUST). */
export const stateQkParallelReadonlyKey = (sliding: boolean, gqa: boolean): string =>
  `attention_state_qk:v1:f32:wg${STATE_ATTENTION_TILE_X}x${STATE_QK_D_LANES}:par:ro${stateVariantKeyPart(
    sliding,
    gqa,
  )}`;

/** Key of ② readonly (the live formula differs, so a key separate from ② is MUST). */
export const stateStatsReadonlyKey = (sliding: boolean): string =>
  `attention_state_stats:v2:f32:wg${STATE_STATS_WORKGROUP_SIZE}:ro${stateVariantKeyPart(
    sliding,
    false,
  )}`;

/** Key of ③' readonly (`:par:ro`). */
export const statePvParallelReadonlyKey = (sliding: boolean, gqa: boolean): string =>
  `attention_state_pv:v1:f32:wg${STATE_ATTENTION_TILE_X}x${STATE_PV_KV_LANES}:par:ro${stateVariantKeyPart(
    sliding,
    gqa,
  )}`;

/**
 * ①' readonly. Bindings (**there is no ins, so they shift up by one** — not the same numbers as
 * ①'):
 *
 * | binding | Resource                                                 |
 * | ------- | -------------------------------------------------------- |
 * | 0       | `Params` (uniform)                                       |
 * | 1       | `q` `[B,H,M,D]` (M = 1)                                  |
 * | 2       | `slot_k` `[B,Hkv,C,D]`                                   |
 * | 3       | `s` `[B·H, rows_block, colCap]` (write)                  |
 * | 4       | `Lengths` (uniform — reads only `past`; `query` is 1)    |
 *
 * The reduction is the same fixed-order tree over D as ①' (the partial sums of
 * {@link stateScoreFn} come from the same single place).
 */
export const stateQkParallelReadonlyWgsl = (sliding: boolean, gqa: boolean): string =>
  `// karume attention_state_qk (states-form S materialisation, f32, D-parallel reduction, readonly = past only${
    sliding ? ", sliding window" : ""
  }${gqa ? ", GQA" : ""})
${STATE_PARAMS_STRUCT}
${STATE_LENGTHS_STRUCT}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> slot_k: array<f32>;
@group(0) @binding(3) var<storage, read_write> s: array<f32>;
@group(0) @binding(4) var<uniform> lengths: Lengths;

${stateSlotRowWgsl(sliding)}

${stateReadonlyLiveWgsl(sliding)}

${stateReadonlyWindowFn(sliding)}

${stateEffectiveRowsWgsl()}

${stateScoreFn("score_slot", "slot_k", STATE_QK_D_LANES)}

var<workgroup> scratch: array<f32, ${STATE_ATTENTION_TILE_X * STATE_QK_D_LANES}>;

@compute @workgroup_size(${STATE_ATTENTION_TILE_X}, ${STATE_QK_D_LANES})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let local_row = wid.y;
  if (local_row >= effective_rows(lengths.query)) {
    return;
  }
  let past = lengths.past;
  let live = live_columns(past, lengths.query);
  let cl = wid.x * ${STATE_ATTENTION_TILE_X}u + lid.x;
  let lane = lid.y;
  let z = wid.z;
  let col = column_base(past) + cl;
  let row = params.row_offset + local_row;
  let q_base = (z * params.chunk_rows + row) * params.depth;
  let kv_plane = ${kvPlaneWgsl(gqa)};
  // Only the slot (past) is read. The edge tile (cl ≥ live) does not run the dot product and **does
  // not return either** (the barrier below sits only inside uniform control flow)
  let inside = cl < live && in_window(col, past);
  var acc = 0.0;
  if (inside) {
    acc = score_slot(q_base, (kv_plane * params.capacity + slot_row(col)) * params.depth, lane);
  }
  scratch[lane * ${STATE_ATTENTION_TILE_X}u + lid.x] = acc;
  workgroupBarrier();
  // Fixed-order tree reduction (stride 8 → 4 → 2 → 1) — the basis of determinism
  var stride = ${STATE_QK_D_LANES / 2}u;
  while (stride > 0u) {
    if (lane < stride) {
      let mine = lane * ${STATE_ATTENTION_TILE_X}u + lid.x;
      scratch[mine] = scratch[mine] + scratch[(lane + stride) * ${STATE_ATTENTION_TILE_X}u + lid.x];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }
  // -inf outside the predicate. The live range is **always written, even outside the predicate**
  // (otherwise ② consumes the previous leftovers)
  if (lane == 0u && cl < live) {
    var value = bitcast<f32>(params.neg_inf);
    if (inside) {
      value = scratch[lid.x];
    }
    s[(z * params.rows_block + local_row) * params.col_cap + cl] = value;
  }
}
`;

/**
 * ② readonly. The same skeleton, bindings and params as ②; only the live formula is the readonly one
 * ({@link stateReadonlyLiveWgsl}).
 */
export const stateStatsReadonlyWgsl = (sliding: boolean): string =>
  stateStatsBodyWgsl(sliding, {
    headerNote: ", readonly = past のみ",
    live: stateReadonlyLiveWgsl(sliding),
    rowsNote: "",
    amaxBarrierNote: "",
    sumNote: `    // ② Σ exp(S - amax). **An empty row (amax == -inf, including P = 0 with 0 columns)
    // never runs it**
`,
    emptyStatsNote: "",
    tailBarrierNote: "",
  });

/**
 * ③' readonly. Bindings (**there is no ins, so they shift up by one**):
 *
 * | binding | Resource                                   |
 * | ------- | ------------------------------------------ |
 * | 0       | `Params` (uniform — the same struct as ①)  |
 * | 1       | `s` (read)                                 |
 * | 2       | `stats` (read)                             |
 * | 3       | `slot_v` `[B,Hkv,C,D]`                     |
 * | 4       | `out` `[B,H,M,D]` (write)                  |
 * | 5       | `Lengths` (uniform)                        |
 *
 * The reduction is the same fixed-order tree over KV as ③'. Pad rows (`row ≥ Q`) are exactly 0 (the
 * same contract as ③). As in ③', only lane 0 writes the zeros of pad rows — KV_LANES lanes point at
 * the same `out[at]`, and unsynchronised writes to the same element are a WGSL data race even with
 * equal values.
 */
export const statePvParallelReadonlyWgsl = (sliding: boolean, gqa: boolean): string =>
  `// karume attention_state_pv (states-form O = P @ V, f32, KV-parallel reduction, readonly = past only${
    sliding ? ", sliding window" : ""
  }${gqa ? ", GQA" : ""})
${STATE_PARAMS_STRUCT}
${STATE_LENGTHS_STRUCT}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> s: array<f32>;
@group(0) @binding(2) var<storage, read> stats: array<f32>;
@group(0) @binding(3) var<storage, read> slot_v: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<uniform> lengths: Lengths;

${stateSlotRowWgsl(sliding)}

${stateReadonlyLiveWgsl(sliding)}

${stateEffectiveRowsWgsl()}

var<workgroup> scratch: array<f32, ${STATE_ATTENTION_TILE_X * STATE_PV_KV_LANES}>;

@compute @workgroup_size(${STATE_ATTENTION_TILE_X}, ${STATE_PV_KV_LANES})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let d = wid.x * ${STATE_ATTENTION_TILE_X}u + lid.x;
  let local_row = wid.y;
  let lane = lid.y;
  let z = wid.z;
  let in_depth = d < params.depth;
  let at = (z * params.chunk_rows + params.row_offset + local_row) * params.depth + d;
  if (local_row >= effective_rows(lengths.query)) {
    if (lane == 0u && in_depth) {
      out[at] = 0.0;
    }
    return;
  }
  let past = lengths.past;
  let live = live_columns(past, lengths.query);
  let base_col = column_base(past);
  let kv_plane = ${kvPlaneWgsl(gqa)};
  let s_row = z * params.rows_block + local_row;
  let s_base = s_row * params.col_cap;
  let amax = stats[s_row * ${STATE_STATS_STRIDE}u];
  let inv = stats[s_row * ${STATE_STATS_STRIDE}u + 1u];
  // Per-lane partial sum (ascending col, stride KV_LANES). Only the slot (past) is read
  var acc = 0.0;
  if (in_depth) {
    for (var cl = lane; cl < live; cl = cl + ${STATE_PV_KV_LANES}u) {
      let col = base_col + cl;
      let p = exp(s[s_base + cl] - amax) * inv;
      acc = acc + p * slot_v[(kv_plane * params.capacity + slot_row(col)) * params.depth + d];
    }
  }
  scratch[lane * ${STATE_ATTENTION_TILE_X}u + lid.x] = acc;
  workgroupBarrier();
  var stride = ${STATE_PV_KV_LANES / 2}u;
  while (stride > 0u) {
    if (lane < stride) {
      let mine = lane * ${STATE_ATTENTION_TILE_X}u + lid.x;
      scratch[mine] = scratch[mine] + scratch[(lane + stride) * ${STATE_ATTENTION_TILE_X}u + lid.x];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }
  if (lane == 0u && in_depth) {
    out[at] = scratch[lid.x];
  }
}
`;

/**
 * Start of the readonly resident range (logical col) — the same formula as the WGSL
 * {@link stateReadonlyLiveWgsl}, MUST.
 */
export const stateColumnBaseReadonly = (window: number, past: number): number =>
  stateSliding(window) ? past - Math.min(past, window) : 0;

/** Readonly live column count `min(P, W)` (`P` for full) — the same formula as the WGSL, MUST. */
const stateLiveColumnsReadonly = (window: number, past: number): number =>
  stateSliding(window) ? Math.min(past, window) : past;

/**
 * Workgroup count of ①' readonly `[⌈live / TILE_X⌉, valid rows, B·H]`.
 *
 * `query` is fixed at 1 (the borrowing context's contract), so it is not a parameter. `P = 0` gives
 * live 0 → a column axis of 0, the dispatch is empty and not a single word of S is written (② sees
 * an empty row with live 0 → ③' writes an exact 0).
 */
export const stateQkParallelReadonlyWorkgroups = (
  geometry: StateDispatchGeometry,
  past: number,
  limit: number,
  where: string,
): [number, number, number] => {
  assertDispatchGeometry(`${where} ①'QK(ro)`, geometry);
  assertU32Params(`${where} ①'QK(ro)`, { past });
  const live = stateLiveColumnsReadonly(geometry.window, past);
  const rows = stateEffectiveRows(geometry.rowsBlock, geometry.rowOffset, 1);
  return [
    tiledWorkgroups(live, STATE_ATTENTION_TILE_X, limit, `${where} ①'QK(ro)`),
    tiledWorkgroups(rows, 1, limit, `${where} ①'QK(ro)`),
    tiledWorkgroups(geometry.batchHeads, 1, limit, `${where} ①'QK(ro)`),
  ];
};

/** Workgroup count of ② readonly `[B·H × valid rows, 1, 1]` (`query` is fixed at 1). */
export const stateStatsReadonlyWorkgroups = (
  geometry: StateDispatchGeometry,
  limit: number,
  where: string,
): [number, number, number] => {
  assertDispatchGeometry(`${where} ②stats(ro)`, geometry);
  const rows = stateEffectiveRows(geometry.rowsBlock, geometry.rowOffset, 1);
  return [gridStrideWorkgroups(geometry.batchHeads * rows, 1, limit), 1, 1];
};

/** Workgroup count of ③' readonly (as ③' — the row axis is all of `rows_block`; no logical lengths). */
export const statePvParallelReadonlyWorkgroups = (
  geometry: StateDispatchGeometry,
  limit: number,
  where: string,
): [number, number, number] => statePvParallelWorkgroups(geometry, limit, `${where}(ro)`);
