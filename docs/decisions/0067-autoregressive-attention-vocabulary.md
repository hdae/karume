# 0067: Autoregressive attention vocabulary — divisible GQA broadcast and a contract with state references

- Status: accepted (2026-08-17 — the maintainer's ruling: GQA × i8a8 starts as fail loudly, with support
  to follow); partly superseded: decision 4's "one form, capacity = window, no slack" and decision 5's
  "exactly one append per slot" by [0096](0096-speculative-decoding.md) decisions 1 / 4 (2026-09-08),
  the M = 1 condition of ①′ by 0096 (2026-09-09), the two-value option by
  [0102](0102-state-attention-stats-pv-fusion.md) (2026-09-15), the `Gemma4Pipeline` default by
  [0104](0104-gemma-fast-quant.md) (addendum of 2026-09-26). Re-graded 2026-10-18: decision 2's r = 1
  byte identity and the geometry thresholds to SHOULD; subgroup reductions reopened as MAY.
- Date: 2026-08-17
- Related: [0023](0023-fused-attention.md) (decision 1 supersedes **part** of its decision 4) /
  [0044](0044-runtime-attention-mask.md) (safe_softmax) / [0060](0060-row-block-attention.md) (row
  blocks — an acceptance condition) / [0066](0066-generation-context-state-slots.md) (state slots,
  logical-length scalars) / [0058](0058-numerics-opt-in-contract.md) (unimplemented combinations fail
  loudly instead of falling back)
- Basis: [research/2026-08-17-autoregressive-references.md](../research/2026-08-17-autoregressive-references.md)
  §3, §6 ("research §n" below)

## Context

Neither target model fit the contract: Gemma 4 E2B is 8:1 MQA (`num_key_value_heads: 1`), MiniCPM5-1B
is 16:2 GQA, and `attention` required the same H for q/k/v (ADR 0023 decision 4,
`packages/runtime/src/ops/shapes.ts`). There was also no vocabulary for "read KV from a state slot",
"express causal / sliding" or "cut work by the logical length". Reference facts are in research §3
(GQA by integer division in the kernel is the norm; masks are attrs + values or computed predicates;
how an empty row becomes 0 depends on the implementation).

## Decision

### 1. GQA = the H match of `attention` relaxed to a divisible broadcast (G3 option A)

`q[B,H,M,D]` / `k[B,Hkv,N,D]` / `v[B,Hkv,N,D]`, condition **`H % Hkv == 0`**, output `[B,H,M,D]`.
`r = H / Hkv` is **derived** (no attrs field). The implementation also requires **`H ≥ Hkv ≥ 1`**
(`0 % Hkv == 0` would let H = 0 through, the equality short-circuit H = Hkv = 0 — fix after
independent review, 2026-08-17).

- Only "**exact H match of q/k/v**" of ADR 0023 decision 4 is superseded. Exact B match, equal Hkv of
  k and v, equal D and the N = 0 rejection **stay as detectors of mix-ups**. MQA (Hkv = 1) uses the
  same formula.
- No divisible batch broadcast for `bmm` (option B): its exact batch match is a deliberate detector
  (ADR 0022 / 0023). GQA on the decomposed path is not rescued: **GQA models MUST preserve SDPA**. The
  exporter no longer reads `enable_gqa` (`_h_attention` in `tools/exporter/src/karume/aten_handlers.py`);
  the shape layer checks the GQA conditions.

### 2. Implementation: integer division in the kernel + a uniform

Head mapping `wid.z / r` (for `wid.z = b*H + h` and `H = Hkv·r`, integer division gives exactly
`b*Hkv + h/r` — as ORT WebGPU, research §3.2).

- The f32 fused path changes the B-operand batch base in `gemm.ts` (the branch shared by attention_qk /
  attention_pv; an op branch is added since the latter lives with linear). The codegen key gets one GQA
  bit; r is a uniform.
- The r = 1 path SHOULD generate WGSL without the GQA terms, so non-GQA shapes do not pay for the
  uniform (precedent: ORT JSEP's `nReps === 1` specialisation) (re-graded 2026-10-18 to SHOULD: byte
  identity with the pre-GQA WGSL was the migration condition at introduction; codegen determinism
  stays a separate invariant).
- Acceptance: **Uint32 parity against a materialised repeat_kv** (r ∈ {1, 2, 8}, B > 1, fault injection:
  wrong r, missing PV-side mapping). An ADR 0060 decision 3-style "only base arithmetic differs"
  argument is not claimed until measured (research §7, attn LB-6).

### 3. GQA × i8a8 starts as fail loudly (ruled 2026-08-17 — support to follow)

The i8a8 attention is a separate WGSL with five head bases (`attention-i8a8.ts`: only K/scale and
V/scale map to the kv-head), and the recipe builder quantizes and allocates K/V for `B*H` (research
§3.2). **The first implementation rejects `attentionCompute: "i8a8"` × a GQA shape with fail loudly**
(ADR 0058 decision 3; a silent drop to f32 would change performance silently) — in
`packages/runtime/src/runtime/recipe-builders/attention.ts`.

**The rejection is provisional; support is expected** (the maintainer's ruling). Surfaces: ① of the
five bases only kbase / ksbase (and the V/scale bases on the PV side) take the kv-head mapping
(`wid.z / r`); qbase / qsbase / sbase stay on the q-head; ② the recipe builder quantizes and allocates
K/V in the Hkv shape. The f32 GQA parity assets (decision 2) are built to carry over to i8a8.

### 4. Attention with state references (same op, extended contract; the field decides the form)

Nodes get an **optional `states` field** (ADR 0066 decision 2): `{ "k": <slot name>, "v": <slot name> }`.
Without it the old contract holds (existing assets and checks unaffected — the same extension pattern
as the fourth mask input). With it:

- **One form only**: **ins k/v = this step's new k/v; slots = the past only**. The declared ins shape is
  **`[B,Hkv,M,D]` (M = physical chunk dimension — chunkLength for prefill, 1 for decode)**; valid data
  is the compact prefix of queryLength rows (ADR 0066 addendum 6). KV-sharing layers (the last 20 of
  Gemma 4 E2B) compute no projection and **wire the owner layer's k/v value tensors into ins** (same
  slot name; resolution as vLLM's kv_sharing_target_layer_name, "nearest non-shared layer of the same
  kind" — refs/vllm gemma4.py:462-488). Every reader reads the past before the append (decision 5b), so
  the ring capacity stays the window, without staging or slack. (Superseded by ADR 0096 decisions 1 / 4
  on 2026-09-08: a readonly form (`ins: [q]`) reads external slots, and sliding capacity is window +
  slack — the addendum of 2026-09-08 and [docs/ir-v2.md](../ir-v2.md).)
- **Physical slot shape and checks**: slots are **`[B, Hkv, C, D]`** (C = capacity, f32; f16 is the
  reserved option of ADR 0066 addendum 5). Fail loudly on ① k/v slots of different shapes ② B / Hkv / D
  mismatch with ins ③ `window ≤ C` violated (sliding) ④ `pastLength + queryLength > C` at run time for
  full slots (checked by the context) — the state extension of the shape check, stopping slot mix-ups
  before OOB or silent misreads.
- **Causal only** (no field — no demand for non-causal + state; bidirectional prefill uses the form
  without states). Predicate `col ≤ pastLength + row` (logical coordinates, TVM style — research §3.1;
  no mask tensor). **With sliding, the lower bound MUST be ANDed**: `max(0, pastLength + row − window +
  1) ≤ col ≤ pastLength + row`. With the upper bound alone, row > 0 **silently mixes in row-many
  out-of-window keys** (W = 4: row 1 sees 5 keys). Reference: vLLM's `(q_abs − key_pos) < W` AND
  (triton_attention_helpers.py:197-229 @7ea4b40).
- **The sliding window is the optional attrs `window`** (positive int; absent = whole context);
  per-layer mixes (Gemma 4 E2B 28 / 7) use per-node attrs, no separate op or kernel (research §3.4).
  **Logical col → physical row MUST use the same formula for reads and writes**: `col % window`, as
  `state_append` writes (superseded by the addendum of 2026-09-08 on 2026-09-08: the modulus is
  capacity; the same-formula MUST stays). The referenced past is resident in `[pastLength −
  min(pastLength, window − 1), pastLength)`; the current part (`col ≥ pastLength`) comes from ins. The
  kernel evaluates the predicate in logical coordinates, then maps.
- The logical lengths are **runtime scalars** in the **context-owned mutable uniform** (ADR 0066
  addendum 4), and **the host computes dispatch counts from them** (no graph capture, every run is
  encoded, so no indirect dispatch is needed to meet ADR 0066's workload condition).
- **RoPE stays outside the attention op.** Per-layer RoPE (Gemma 4 E2B — research §6.1) was baked into
  the graph by the exporter (replaced by ADR [0091](0091-gemma4-host-rope-variable-capacity.md)
  decision 1 on 2026-09-03: the host passes cos / sin per chunk as derived inputs). **"The attention op
  does not know RoPE" MUST stays.**

### 5. KV writes are a separate op, `state_append`

The **single-purpose op `state_append`** writes this step's k/v into the slot (slot name + input
`[B,Hkv,M,D]` (same physical chunk dimension as attention's ins) + a logical position scalar; **it writes
only the first queryLength rows** (slots are outside full-write — ADR 0066 addendum 6); it also owns the
ring write of sliding slots).

- why-not (append inside attention, TVM style): one more dispatch, but ① verification of full-write /
  no-op pad rows stays in a single-purpose op ② attention stays read-only, so bit-identity checks stay
  simple ③ KV-sharing layers are just "no `state_append` node" (ORT's kv_empty by the absence of an op).
  **Absence is per layer; per slot there is always exactly one terminal `state_append`** (shared layers
  split it among readers), so the check works per slot (decision 5b, `assertStateOrder` in
  `runtime/plan.ts`). (Superseded by ADR 0096 decision 1 on 2026-09-08: true for slots the context
  owns; external slots read by a readonly attention have no append.)
- why-not (staging / ring slack, ORT WindowedKvCache style): appending before readers needs `window + Q
  − 1` slack or a staging buffer; with every reader taking current from ins the append is always last.
- **Zero outputs** (an effect op). The IR parser no longer rejects empty `outs`; the contract layer
  allows 0 outputs only for ops whose contract declares an effect (`format/ir.ts`). The output-count
  generalisation over six surfaces is ADR 0068 decision 1.

### 5b. Order of state effects = `nodes` array order (independent of data edges)

State reads and writes **draw no data edges**, so topological order does not fix them. Contract:
**nodes touching the same slot MUST execute in `nodes` array order** (plan / recipe never reorder
state-touching steps; fusion matchers never reorder across state). Exporter emission rules:

1. Emit **every reader of the slot (owner + shared attention layers) → `state_append`**: the append is
   always the last, single write, so a ring wrap cannot overwrite this step's readers' past rows.
2. Checks: the plan fails loudly on "more than one append to a slot per step" and "a reader of the slot
   after its append".

### 6. Empty row → 0 built into the states form (safe_softmax family)

The states form legitimately produces pad rows and empty-context rows, so ADR 0044's "an all −inf row
into fused attention is a contract violation" **does not apply**. Its row statistics use **"row max
initialised to −inf + a zero-denominator guard"**, giving empty row → output 0 by construction
(llama.cpp WebGPU style — research §3.4).

MUST: **−inf is never replaced by a finite sentinel (−5e4 and the like)** — TVM's construction makes an
empty row "the weight-1 average of V", breaking safe_softmax (row max −inf → all 0) (research §7, attn
LB-4).

### 7. S is materialised in row blocks (ADR 0060 on the preserved path — acceptance condition)

① QK materialises S in row-block windows (`[B·H, block, N]` — ADR 0060 moved to the preserved
`attention` path); decode has S `[B·H, 1, N]`. Prefill chunk × long context legitimately exceeds the 128
MiB `maxStorageBufferBindingSize` (research §3.1), so **a states-form implementation without row blocks
is not acceptable**. **A state slot's own binding can also exceed the limit** (131K × f32 on Gemma 4
E2B's full layers) — that contract is ADR 0066 addendum 5.

## Consequences

- Existing assets and checks are unaffected (no states field, r = 1 unchanged, mask contract unchanged).
- ADR 0023 carries a supersede note on its H clause; ADR 0044 the contrast "empty rows are legitimate in
  the states form". The IR gained the states field, `state_append` and attrs `window`; the exporter emits
  them in the decode graph scripts.
- Acceptance conditions of the implementation wave: ① r = 1 snapshots byte-identical ② repeat_kv parity
  + fault injection ③ census (GQA took effect — 0058 decision 4) ④ row blocks (with forced splits) ⑤ a
  direct check of empty row → 0 ⑥ all existing sha256 checks green.

## Addendum (2026-09-03) — KV-parallel reduction variant ③′ of ③ PV (perf-ledger K-12)

③ PV (`O = P @ V`) gets **a variant that splits the KV length over 16 lanes of a workgroup**
(`attention_state_pv:v1:f32:wg16x16:par`). With one invocation per O element, decode (M = 1) has a fixed
`D × B·H` invocations and KV length only lengthens one thread's loop; at P = 16K attention took 72% of
decode GPU time ([research](../research/2026-09-03-gemma4-context-length-sweep.md) §3).

- **Contract**: bindings, params and the **number** of dispatches equal ③ (row axis `rowsBlock` instead
  of `rowsBlock / TILE_M`: 4× workgroups in prefill, equal in decode). Lane `l` accumulates `cl ≡ l (mod
  16)` ascending and folds along a fixed tree (stride 8 → 4 → 2 → 1). Determinism, independence of
  capacity and row blocks, and pad rows → exact 0 are held by bit-exact checks; **not bit-identical to
  ③** (③ reduces sequentially by ascending col; ③′ owns "lane partial sums → fixed tree").
- **Option**: the ADR 0058 opt-in `SessionOptions.stateAttentionReduce: "sequential" | "parallel"`
  (default `"sequential"` = reference path) (superseded by ADR 0102 on 2026-09-15: a third value
  `"parallel-fused"` for states forms with M ≤ 8 and a column bound ≤ 1024; still one option). Checks:
  ③ unchanged; A/B tolerance check (`tests/gpu_state_attention_parallel_test.ts`, tolerance 5e-6);
  census check (`tests/gpu_state_execution_test.ts`).
- **Measured** ([research](../research/2026-09-03-gemma4-chunklength-k12-sweep.md) §3): full PV 35.0 →
  3.6 ms, decode at P = 16K 81.3 → 41.0 ms/token, tokens unchanged. The promotion to the
  `Gemma4Pipeline` default (2026-09-03) is superseded by ADR 0104 addendum of 2026-09-26: the default
  is the preset's `session` declaration in the manifest; without one the reference path `"sequential"`
  runs (no family default in code — `packages/models/src/gemma/session-options.ts`).
- why-not (split-KV + merge / online softmax): the in-workgroup split adds no dispatch or buffer and can
  become stage A of a split-KV; the online form removes the S temporary but does not fix decode
  parallelism. Subgroup reductions MAY be reconsidered on measurement (re-graded 2026-10-18 to MAY: the
  2026-08-10 reason — the adapter did not advertise the feature — no longer holds; `acquireGpu`
  requests subgroups and ADR [0100](0100-rms-subgroup-reduction.md) /
  [0101](0101-linear-gemv-subgroup.md) use them).

## Addendum (2026-09-06) — D-parallel reduction variant ①′ of ① QK (perf-ledger K-14)

① QK gets, by the technique of ③′, **a variant that splits D over 16 lanes**
(`attention_state_qk:v1:f32:wg16x16:par`); at P = 16K ① QK took 17% of decode GPU time
([research](../research/2026-09-06-state-qk-parallel-k14.md)).

- **Contract**: bindings, params and dispatch count equal ①. One workgroup = one row × 16 columns; lane
  `l` accumulates `d ≡ l (mod 16)` and folds along the fixed tree. Write conditions (−inf outside the
  predicate in the live range; nothing for `cl ≥ live` or pad rows) and half-scaling equal ①; **not
  bit-identical to ①**; determinism and the other properties are held by bit-exact checks.
- **Option**: the **same** `stateAttentionReduce: "parallel"` (one knob — the maintainer's ruling of
  2026-09-06). **①′ was selected only for M (chunkRows) = 1** (`stateQkParallelEligible`) (superseded
  by ADR 0096 on 2026-09-09: M ≤ 8, so verify rows and decode share the reduction order; `chunkRows <=
  8` in `packages/runtime/src/kernels/state-attention.ts`). In prefill (M = 768) ①′ shrinks the row tile
  4 → 1 and was 1.5–1.9× slower. Selection uses only plan-time static values (branching on the runtime
  logical length would make one plan key point at different pipelines).
- **Checks**: ① unchanged; A/B tolerance check (`tests/gpu_state_attention_parallel_test.ts`, 5e-6, two
  fault injections fail it); census check (`tests/gpu_state_execution_test.ts`); a non-GPU truth table
  of the eligibility conditions (`tests/kernel_state_attention_test.ts`).
- **Measured**: decode ① QK 6.3 → 3.7–4.1 ms/token, decode wall at P = 16K −9 to −15%. The
  `Gemma4Pipeline` default is superseded as above (ADR 0104, 2026-09-26). Prefill is traffic-bound, which
  D lanes do not help; the geometry table below holds both in one family and one option.

## Addendum (2026-09-06) — geometry table: prefill plans (M ≥ 16) take the tiled path ①ₜ / ③ₜ (perf-ledger K-13)

Prefill plans (M = 768) were traffic-bound, rereading K / V rows M times (attention 79% of GPU time for
a 14.7K-token chunk — [research](../research/2026-09-06-state-attention-tiled-k13.md)). ①ₜ / ③ₜ plug
states fragments into the GEMM skeleton of fused attention (shared tiles, register blocking, ascending
K tiles of 16 — ADR 0022 decision 3) and are **selected for M ≥ 16 regardless of the option**.

- **Contract (bit-identical)**: ①ₜ tiles A = q·scale / B = k·scale and sums `acc = acc + a * b` over
  ascending d — ①'s formula and order; ③ₜ runs A = P (`exp(S − m)·inv`, not materialised) / B = V over
  ascending col — ③'s. S fills the live range with −inf / values and writes neither `[live, col_cap)`
  nor pad rows (as ①); O is full-write with pad rows an exact +0.0 via `select` (as ③). Row tiles
  without valid rows run 0 K iterations (work ∝ Q — ADR 0066 decision 3). Checks: u32 equality of every
  word of S / O (17 / 19 cases) + fault injection + exact Gemma 4 golden.
- **Geometry table** (current; chosen by M like `gemmGeometryForRows`):

  | M of the plan        | ① QK              | ③ PV              | Numerics                                    |
  | -------------------- | ----------------- | ----------------- | ------------------------------------------- |
  | 1–8 (decode, verify) | ①′ (parallel) / ① | ③′ (parallel) / ③ | by the option (parallel: A/B tolerance)     |
  | 9–15                 | ①                 | ③′ (parallel) / ③ | same                                        |
  | ≥ 16 (prefill)       | ①ₜ                | ③ₜ                | bit-identical to the reference path         |

  (The row "2–15 → ①" of 2026-09-06 is superseded by ADR 0096 on 2026-09-09; `"parallel-fused"` of ADR
  0102 applies on the M ≤ 8 row.) ③′ narrows to M < 16. Eligibility is a pure function of plan-time
  values (`stateQkTiledEligible` / `statePvTiledEligible` = `chunkRows >= 16`). The thresholds SHOULD
  stay as listed until re-measured (re-graded 2026-10-18 to SHOULD: ①ₜ / ③ₜ are bit-identical to the
  reference path, so the thresholds are a performance choice measured on RTX 3080 Ti / Vulkan (wgpu) /
  Deno 2.9.6, open to change after measurement on Dawn and browsers).
- **Measured**: prefill at P = 16K −64% wall, decode and tokens unchanged.
- why-not (online softmax): removing the materialised S is a separate axis; the bound was traffic.

## Addendum (2026-09-08) — the ring modulus is `capacity`, not `window` (ADR 0096 decision 4)

- `slot_row(col) = col % window` of decision 4 becomes **`col % capacity`** (the same-formula MUST
  holds — one `stateSlotRowWgsl` string is shared by ①/①′/①ₜ/③/③′/③ₜ/append and the reference
  implementation). `column_base` / `live_columns` / `in_window` stay window-based. The distribution bakes
  sliding capacity as `window + slack` (8 for Gemma 4), the slack holding rejected verify rows (ADR 0066
  addendum of 2026-09-08). The `window ≤ capacity` check is unchanged.
- The de-duplication guard of `state_append` (with Q > capacity only the last logical row per physical
  row writes) is capacity-based too; a window-based guard would silently drop rows when capacity >
  window.
- Decision 5 was unchanged in step 1; step 2 added read-only external slots (ADR 0096 decision 1).

## Addendum (2026-10-27) — ①′ folded over the query heads of one kv plane; the unrolled KV loop of ③′ and the fused stats + PV (perf-ledger K-55 / K-91)

In ①′ and ③′ one workgroup serves one query head, so with GQA every head of a kv plane re-reads the same K / V
rows (for Gemma 4 E2B 8 × 32.8 MB (P = 16,000) of K per full layer, beyond the 6 MB L2 of the RTX 3080 Ti). Two
bit-identical rewrites ([research](../research/2026-10-27-k55-head-fold-spike.md)):

- **Fold of ①′ (K-55)**: on plans that take ①′ (option `"parallel"` / `"parallel-fused"`, M ≤ 8) and are **full
  attention of a selected class (r, D, M)** — Gemma 4 E2B (r = 8, D = 512) at M = 1 / 4 / 8 and MiniCPM5 (r = 8,
  D = 128) at M = 1 — one workgroup serves the F = r query heads of one kv plane over 8 columns and loads each K
  element once (`attention_state_qk:v1:f32:wg8x16:par:fold<F>:gqa`, dispatch `[⌈live / 8⌉, valid rows, B·H / F]`).
  The rule is `stateQkFoldHeads`, a pure function of plan-time values: a class is selected only where the fold was
  measured not slower than ①′ at every measured P (1,024 / 4,000 / 16,000) **and** observed bit-identical to ①′ on
  every backend checked (Vulkan / RTX 3080 Ti and D3D12 / RTX 5070 Ti). It stays inside the measured range (the
  discipline of ADR 0082 decision 4): other r, D or M, non-GQA plans, sliding layers (the fold measured ≤ ×1.07
  there, F ≥ 4 slower) and the readonly drafter kernels keep ①′. Gemma 4 E4B (r = 4, F = 4, D = 512) was measured
  faster at M = 1 / 4 / 8 but keeps ①′: its F = 4 kernel differs from ①′ on D3D12 (below), so selecting it would be
  an automatic choice that moves numbers there (ADR 0058 decision 2); making it identical is perf-ledger K-95. Being
  bit-identical, the plans of one model may take different
  ones of the two (verify and decode) without moving a word of the verify-vs-decode u32 check. The 8-column tile
  halves the live length at which the column axis hits `maxComputeWorkgroupsPerDimension` (524,280 columns at
  65,535, against 1,048,560 for ①′); past it the dispatch fails loudly, and today's models stay far below. F is in
  the key (r travels in the uniform and stays out), and the host asserts F | r before selecting
  (`assertStateQkFold` — a group straddling two kv planes would read another plane's K with neither an exception
  nor NaN). The selection stays at the single site of `buildStateAttention`, from plan-time static values; the M 1–8
  row of the geometry table reads "①′ (folded on the measured full-attention classes) / ①".
- **Unroll of the PV loop (K-91)**: every live-form ③′ variant (the readonly `:ro` excluded) and the fused stats +
  PV (ADR 0102) scan each lane's columns in batches of 8, issuing the batch's V loads before its multiply-adds,
  then a single-column tail; the workgroup and the dispatch are unchanged. The keys moved to
  `attention_state_pv:v2:…:par…` and `attention_state_stats_pv:v2:…` (one key MUST map to byte-identical WGSL); ③,
  ③ₜ and the readonly ③′ keep v1. Folding the PV heads measured at most ×1.07 over the unroll and is not done.
- **Contract — bit identity, not a band**: per output word the expression and the order of operations are those of
  the kernel replaced. QK: `acc + (q·scale)·kd` with `kd = k·scale`, d ascending per lane, the stride 8 → 4 → 2 → 1
  tree. PV: `acc + (exp(s − amax)·inv)·v`, col ascending per lane, the same tree; the fused statistics unchanged.
  The −inf / NaN / empty-row guards and the write predicates are unchanged; no `fma()` and no reassociation. Being
  bit-identical to the kernels they replace, neither needs an option (ADR 0058 governs paths that change numbers);
  the A/B tolerance checks of ①′ / ③′, the speculative verify-vs-decode u32 check, the e2e goldens and the Gemma
  sha256 rows do not move.
- **The identity is the backend's, not WGSL's**: it rests on the backend compiling the same expression the same way
  (for example contracting `acc + a·b` into an FMA in both kernels) — the class of caveat of ADR
  [0082](0082-linear-gemv-decode.md) addendum 1. It is checked word for word per backend by
  `packages/runtime/tests/gpu_state_attention_fold_test.ts` (the folded ①′ against ①′; ③′ / fused against their v1
  WGSL kept in `packages/runtime/tests/fixtures/wgsl-baseline/`), with order-only mutants that must mismatch;
  the browser runbook `outputs/diag/k55-fold-2026-10-27/runbook/` of the main checkout (not tracked) runs the same
  case set in Chrome. Observed 2026-10-27: on Dawn / Vulkan / RTX 3080 Ti every case matched (F = 8 and F = 4, ③′,
  fused); on Chrome 156 / D3D12 / RTX 5070 Ti (Blackwell) the E2B (F = 8) and MiniCPM5 (F = 8) folds, ③′ (546 cases)
  and the fused kernel (490 cases) matched word for word, while the E4B fold (F = 4) differed on all 84 of its cases
  (M = 1 / 4 / 8; 2,722,774 S words; per-case ULP distances up to 2²⁰; the same WGSL bytes as on the dev machine,
  every mutant detected — [research](../research/2026-10-27-k55-head-fold-spike.md) §6.1). So the identity is a
  per-backend, per-kernel observation, and only classes observed identical on every checked backend are selected.
  A new backend runs that check (the GPU test, or the runbook in a browser) before it relies on the identity.
- **Measured** (RTX 3080 Ti, Node + Dawn, M = 1, per dispatch at P = 1K / 4K / 16K): ①′ fold E2B ×1.81 / ×3.24 /
  ×6.12, E4B ×1.38 / ×1.90 / ×3.65; ③′ unroll E2B ×1.94 / ×2.46 / ×2.51; sliding fused ×1.60–1.69. The fold at
  verify M = 4 / 8 (P = 1,024 / 4,000 / 16,000): E2B ×3.42–7.28, E4B ×2.42–4.04; MiniCPM5 at M = 1 ×1.70 / ×2.67 /
  ×4.68. About 0.49 / 1.15 / 4.74 ms per E2B decode token (an estimate from per-dispatch medians).
