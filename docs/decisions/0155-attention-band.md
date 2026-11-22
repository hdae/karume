# 0155: The band of plain attention is an attr checked inside ①QK, not a mask tensor

- Status: accepted (2026-11-22 — the maintainer approved the design as stage 4a of the EmbeddingGemma 2 context work;
  built on the development machine, unreleased)
- Date: 2026-11-22
- Revisits: ADR [0044](0044-runtime-attention-mask.md) decision 1 ("the mask contract of fused attention does not change")
  for one form only — a key band declared on the node, with no mask tensor. The additive mask of ADR
  [0023](0023-fused-attention.md) and every other decision of ADR 0044 stand.
- Related: [0067](0067-autoregressive-attention-vocabulary.md) decision 4 (the states form's causal rule and window are a
  predicate over positions, the precedent) / [0010](0010-symbolic-constant-folding.md) (the folded constant the band
  replaces) / [0060](0060-row-block-attention.md), [0124](0124-attention-row-block-budget.md) (row blocks of S) /
  [0147](0147-numerics-and-kernel-adoption.md) decision 2 (contract class E) and decision 9 (per-backend evidence) /
  [0144](0144-gemma4-12b-vision.md) (the bidirectional block, a QK predicate with its own kernel check) /
  [0065](0065-exporter-core-recipe-split.md) (exporter core and recipes) / the [backlog](../backlog.md) item
  "EmbeddingGemma 2", step 2

## Context

The sliding layers of EmbeddingGemma 2 attend to the keys with `|q − kv| ≤ 512` (inclusive and symmetric — upstream's
`sliding_window_bidirectional_overlay`). The recipe exports that band the way torch runs it: an additive mask of 0 and −inf,
folded by ADR 0010 into one `[1, 1, Tmax, Tmax]` f32 constant that `sym_prefix_slice` cuts to `T × T` and that the 20
sliding layers share. Three things bound it:

- the constant is `Tmax² × 4` bytes, so it fits the container's 32 MiB block only up to Tmax ≈ 2,896 (the model reads
  8,192);
- the sliced `[1, 1, T, T]` value is a run buffer that passes the default 128 MiB binding near T = 5,792;
- the mask is bound whole to every row block of S, which reads its rows of the full-M tensor.

The states form already has the mechanism: its causal rule and its window are checked inside the score kernel from the
logical positions, and no mask tensor exists (ADR 0067 decision 4). The plain form had only the mask.

## Decision

1. **IR: plain `attention` takes an optional integer attr `band` (≥ 0).** Key column `n` is visible to query row `m` iff
   `|m − n| ≤ band` — inclusive, symmetric, `m` and `n` being the node's own row and column indices (`[0, M)`, `[0, N)`).
   Absence = every column visible; a node without the field keeps its kernels, keys, uniforms and WGSL byte for byte. No
   existing name means this: `window` is the states form's one-sided width over logical positions, and it stays refused on
   the plain form, so one attr does not mean two things by form. A row with no visible column (only when M > N) is the empty
   row of ADR 0044's addendum of 2026-08-31: output 0.
2. **One mechanism per node.** `band` is refused on the states form (the readonly form included) and together with the
   fourth input, both at the contract layer and in the shape rule. The contract table carries it as a new field, the
   optional attrs of the form without a states field (`plainOptionalAttrs` / `plain_optional_attrs`, mirrored by the
   conformance table `fixtures/op-contracts.json`).
3. **Kernel.** ①QK writes the score of a column outside the band as −inf, at the place of the mask's add:
   `select(−inf, acc, in_band(row, col))`, with the absolute row (the row-block offset added) and the column the store
   writes. The predicate has no subtraction — `row + band ≥ col && col + band ≥ row` — and the host writes the band word
   `min(band, max(M, N))`, which hides the same columns and keeps both sums in u32 (a shape where they would not fit fails
   loudly). The band word and the bits of −inf travel in the uniform, after every existing word; the pipeline key gains the
   one word `:band`, after every existing word. ② and ③ are unchanged. Every dense ①QK form carries the predicate: scalar and
   vec4, f16 compute, f16 score storage, GQA, the row window, every geometry. `attentionCompute: "a8"` is refused for a band
   node exactly as for a masked one, on the three a8 dispatches and on the online kernel alike. No work is skipped: S is still
   `B·H × M × N` and every tile is computed.
4. **Contract class E**, the mirror being the same node fed the band as the equivalent additive mask, on finite scores (the
   analysis below). The evidence is the twin check `checkAttentionBand` of `@karume/kernel-checks` (every O word, both sides'
   ①QK keys), run by the runtime's node GPU test and by the workbench's `kernel-check` entry on each backend, with a band off
   by one as the fault injection (node test). The CPU reference implements the band as a distance, and a torch golden
   (`band_attention`) compares the node with torch's masked SDPA.
5. **Exporter: a recipe declares the band.** `karume.band.to_band_form(graph, [BandAttentionSpec(output, band), ...])`
   rewrites the named plain attentions that take a mask: it drops the fourth input, writes `attrs.band` and prunes every
   node and declaration no live value reaches (the folded constant and its `sym_prefix_slice` once no attention reads them).
   It never reads the mask's values or how it was built; that the declared band is the mask the model applies is the
   recipe's claim, held by the recipe's own checks against upstream. A mask that depends on a graph input (a run-time mask)
   fails loudly, also when other nodes read that input: the surgery walks back from each dropped mask through its producers
   before pruning. No recipe calls it yet; every existing export is unchanged.

## The sign of a zero score

Inside the band the mask plan writes `fl(acc + 0)` and the band plan writes `acc`. For a finite acc these are the same word
except for `acc = −0`, which the add turns into `+0`. Outside the band both write −inf.

- **S is not −0 under IEEE arithmetic.** acc is a sum seeded with `+0.0`; under round to nearest a sum is −0 only when both
  addends are −0, so the accumulator never becomes −0. Measured on the RTX 3080 Ti (Node + Dawn / Vulkan, 2026-11-22): q rows
  of −0 against k of both signs (every product a zero of either sign) gave +0 scores in both kernels, scalar and vec4, and
  every S word of the band kernel equalled the mask kernel's.
- **A backend may ignore the sign of a zero** (WGSL allows it). If S is −0 there, only the sign of a zero can differ, and S
  reaches O only through the row maximum (`nan_max`) and `exp(S − m)`. For a finite nonzero S, `S − (±0) = S` exactly; a zero
  S gives a zero difference. So the sign can reach O only through `exp(±0)`, which is 1 in IEEE arithmetic but is not pinned
  by WGSL's accuracy bound for `exp`. That is the one backend assumption of the class E claim, and the twin check with −0 rows
  in q and k on each backend is its evidence. The row statistics themselves may then hold −0 against +0; they are internal.
- **Non-finite scores are outside the claim.** A NaN or +inf score outside the band becomes NaN under the mask (`NaN + −inf`,
  `+inf + −inf`) and −inf under the band: the band hides a column it never reads, as the states form's predicate does.

## Alternatives considered

- **A mask computed in the graph** from positions (iota, compare, where): no constant to store, but still a `[T, T]` value
  per run, bound to every row block and read by ①QK. It lifts the 2,896 limit and keeps the 5,792 one. Not taken.
- **A banded score layout** (S holding only the `2·band + 1` columns of the band): S's size grows with T instead of T², but
  ② and ③ change and the row blocks with them — a larger change, left for later.
- **An f32 online kernel** (no S at all): another reduction order, so a new class with its own evidence (R at best) — left for
  later.
- **Recognising the band from the folded constant** in the exporter: a matcher over constants silently stops matching when a
  model builds its mask another way, and the result would be a wrong value, not an error (the reason ADR 0044 rejected a
  constructive prover). Not taken; the recipe declares.
- **Accepting the band beside the mask** (a band plus a padding mask): two mechanisms on one node, where which one the
  kernel applies first is decided by the kernel and not the IR. Refused; a padding mask on top of the band is later work.

## Consequences

- No existing key, uniform, WGSL or graph moves (the snapshots of every attention kernel are unchanged; ten new snapshots
  pin the band variants, among them band × GQA and band × GQA × row window).
- The workbench's `kernel-check` entry runs a fourteenth check, `attention-band`.
- The memory estimate of a band node is the plain node's: the same S and row statistics per row block, and no mask input.
- Not built: skipping the tiles outside the band (①QK and ③PV still do `M × N` work, and S keeps its size), the band with a8,
  the band beside a mask, and the EmbeddingGemma 2 graph's use of it (its recipe still feeds the folded mask — the next
  stage of the backlog item).
