# 0044: Runtime attention mask — bool input + safe_softmax (a constructive replacement for the guard proof)

- Status: accepted (2026-08-11); decision 3 narrowed by the addendum of 2026-08-31 (an all-masked
  row is a valid input of fused attention)
- Date: 2026-08-11
- Related: [0016](0016-anima-chain-export.md) (guard removal by evaluating the mask's actual values)
  / [0023](0023-fused-attention.md) (fused-attention mask contract, static `[1,1,M,N]`) /
  [0009](0009-dtype-i32-bool.md) (bool = u32 0/1, allowed as I/O) /
  [0043](0043-op-addition-layers.md) (op-addition tiers, superseded by 0059; safe_softmax was tier 2)

## Context

- The CFG of Irodori-TTS v4 builds uncond as "condition state = 0 + **all-zero mask**" (the
  maintainer's ruling of 2026-08-11: allow a runtime bool mask input, option a; recon:
  [research/2026-08-11-irodori-source-recon.md](../research/2026-08-11-irodori-source-recon.md)).
  To remove the safe-softmax guard of the decomposed SDPA path ("an all −inf row returns 0"),
  `_assert_guard_inactive` (`tools/exporter/src/karume/normalize.py`) evaluates the mask's
  **actual values**, and `_eval_static` fails loudly on a runtime-input placeholder. With a runtime
  mask the proof is impossible in principle.
- Irodori's masks (upstream `irodori_tts/model.py`, HEAD `8224daf`) are all **key-only
  `[B,1,1,K]`** (model.py:403-404, 189-191). The joint-attention cat starts with `self_mask`
  (default `torch.ones(bool)`, model.py:357-358), so **no row can become all −inf** even when the
  condition-side mask is all zero.
- EmbeddingGemma's missing runtime `attention_mask` (`docs/limitations.md`) has the same root; this
  ADR solves the mechanism, the wiring is separate work.
- torch's SDPA decomposition turns a bool mask into an additive one with `where(mask, 0, −inf)`, then
  emits `add → softmax → guard`. In the Karume vocabulary `where` exists with the slot contract
  `[BOOL, F32, F32]` (`packages/runtime/src/ops/contracts.ts`), `scalar_tensor` folds 0 and −inf into
  f32 **binary** initializers (no non-finite value in the IR JSON layer), and `cat` is f32-only.

## Decision

### 1. The mask's IR boundary is a bool input (pre-concatenated, made additive inside the graph)

- The graph input takes the **bool mask (u32 0/1, ADR 0009)** as is; torch's `where(mask, 0, −inf)
  → add(scores, ...)` is **carried into the IR unchanged** (CSE folds the conversions of one mask
  into one).
- **No bool cat is added to the vocabulary.** The export recipe normalizes Irodori's cat of segment
  masks so the host passes one pre-concatenated `[1,1,1,K]` mask (the caller already knows the
  segment lengths).
- The mask contract of fused attention (ADR 0023) **does not change**; SDPA with a runtime mask runs
  on the decomposed path. Runtime masks in fused attention get their own ADR once a performance need
  is measured.

### 2. Guard handling: prove from actual values, otherwise rewrite to safe_softmax (a tier-2 op)

- `_drop_safe_softmax_guard` works in two steps:
  1. Try `_assert_guard_inactive` as before; on success **remove** the guard (output bytes of
     existing assets do not change).
  2. Otherwise do not fail: **rewrite** the guard subtree `where(¬any(¬eq(src,−inf)), 0,
     softmax(src))` into one `safe_softmax(src)` node (the matcher reuses `_guard_parts`).
- Contract of `safe_softmax`: softmax plus "**a row whose max is −inf writes all zeros**", in f32.
  Rows with a finite element follow the softmax computation exactly (bit identity pinned by
  snapshots and parity). Under "scores are finite" (non-finite inputs are outside the contract),
  all −inf ⇔ row max = −inf, so this matches torch's guard exactly.
- Adding the guard's pieces (`eq(−inf)` and the like) to the vocabulary stays rejected as in ADR
  0016 (a scalar −inf collides with the IR JSON layer's rejection of non-finite values).

### 3. Input contract

Additive mask values {0, −inf} from `where` are canonical. An all −inf row (= an all-zero bool row)
is **canonical only on the safe_softmax path** (output row 0, as in torch). An all −inf row in plain
softmax violates the contract and contaminates with NaN (like an out-of-range gather — no check).
(The same rule for fused attention was withdrawn on 2026-08-31; see the addendum.)

## Alternatives considered

- **Host-built additive f32 mask input**: one `add` in the graph, but it needs a host 0/−inf
  conversion and a parity test against Python (like the SBV2 relative-attention tables), and the IR
  input's meaning no longer shows in its dtype. Not worth saving one `where` dispatch. Rejected.
- **Constructive inactivity prover** (prove "the first cat segment is constant all-ones" on the
  graph): fragile against how each model builds masks (the "silently stops matching" fragility of
  fusion matchers, moved to the correctness side) and blind to models where empty rows are
  canonical (batch padding). safe_softmax is closed by semantics. Rejected.

## Addendum (2026-08-11) — clarification during implementation

"No proof possible" also covers a proof that the guard **is active** (a static mask with a real all
−inf row): the guard becomes safe_softmax instead of being removed, so no NaN flows downstream. This
shape, which failed loudly under ADR 0016, is now accepted
(`test_an_all_masked_row_rewrites_the_guard_to_safe_softmax` and two more,
`tools/exporter/tests/test_normalize.py`).

## Consequences

- Irodori DiT export (wave 2) is unblocked by one new op, `safe_softmax` (tier 2: two contract
  implementations, WGSL, CPU reference, conformance table, golden, bit-identical parity with the
  decomposed path).
- Existing assets and IR files do not change; safe_softmax appears only on the "no proof" path.
- A path to a runtime `attention_mask` for EmbeddingGemma exists: a bool pad mask added to the band
  constant (adding 0/−inf is order-independent and exact). As of 2026-10-18 it is not wired
  (`docs/limitations.md`, "EmbeddingGemma: runtime attention_mask (in-batch padding) not supported").
- Out of scope, left to Irodori wave 2: upstream `_safe_attention_mask` (model.py:429) becomes a
  tier-0 patch under "B=1 and a non-empty mask when a reference is present"; the duration
  `masked_mean` (0/0 for an all-zero caption) is handled in the export recipe.

## Addendum (2026-08-18) — contrast: empty rows are canonical in the states form

Decision 3 **does not apply to the states form** (ADR
[0067](0067-autoregressive-attention-vocabulary.md) decision 4, attention with a `states` field).
There, padding rows that fall out of the window produce empty rows **canonically**, and empty row →
output exactly 0 is guaranteed by construction (0067 decision 6: a finite sentinel MUST NOT be
substituted; the guarantee is the zero write of pad rows — ADR 0066 addendum 8). The two contracts
coexist as two forms of one op; the presence of the field selects the form.

## Addendum (2026-08-31) — op numerics review W-2 / W-3, ruling a

Decision 3 is **withdrawn for fused attention**: attention_stats v2
(`packages/runtime/src/kernels/attention.ts`) ports safe_softmax's three guard points (identity
−inf, empty-row test, empty-row stats `(0,0)`), so an all-masked row is a **valid input with output
0** (`referenceAttention` has the same branch). Rows with a finite element keep their bits, so
parity with plain softmax still holds. An all −inf row in plain softmax remains a violation.

The row max of softmax, safe_softmax, attention_stats and attention_state_stats now uses `nan_max`
(ADR 0020). Plain `max` drops NaN in WGSL, so **an all-NaN row looked empty to the safe forms and
became exactly 0, silently losing the NaN**. Now it stays NaN at the output. Non-NaN inputs keep
identical bits in every variant (golden, WAV and PNG checks unchanged).

## Addendum (2026-09-10) — guard for symbol-dependent masks

A mask non-empty at symbol lengths 5 and 9 may be empty elsewhere: `arange(T) >= 4` masks every
element for T = 2..4. The inactivity test of decision 2 never treats a symbol-dependent mask as
proven; the rewrite to `safe_softmax` keeps the semantics. Removal for static masks and for paths
without a source of non-finite values is kept.
