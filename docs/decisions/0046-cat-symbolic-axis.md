# 0046: The cat axis may be a linear sum of one symbol

- Status: accepted (2026-08-11; the maintainer's ruling, option c)
- Date: 2026-08-11
- Related: [0014](0014-layout-ops-full-write.md) (its "the cat axis is static" is revised here) /
  [0010](0010-symbolic-constant-folding.md) (dimension grammar `coeff·sym+offset`) /
  [0044](0044-runtime-attention-mask.md) (runtime mask — with this ADR it unblocks the DiT) /
  measurements: [research/2026-08-11-dit-export-recon.md](../research/2026-08-11-dit-export-recon.md)

## Context

- The Irodori DiT's joint attention builds K/V as `cat([self | text | speaker | caption], dim=1)`
  with a symbolic self length S. ADR 0014's static cat axis blocks the export of one DiT step
  (recon §1-2: only the two K/V cats per layer are symbolic; the RoPE pairs and the head
  re-concatenation are static).
- ADR 0014 kept the axis static because a sum of symbolic lengths does not fit the dimension grammar
  in general. The DiT shape is **one symbol plus a constant** (`S + 1519`), which fits
  `coeff·sym+offset` as is. Sums of several symbols (`S+Tt+Ts+Tc`) are rejected by torch.export
  itself (recon §1-3), so nothing inexpressible gets in.
- Measured (recon §1-5): with a two-line relaxation a 12-layer export through `verify_model` and TS
  `bindSymbols` / `planGraph` was green. **Only the declaration-level guard rejected it**; the
  post-binding shape rules, the strided copy kernel and the layout rules already handle it.

## Decision

- The `cat` axis is accepted **only when every input's extent on it is a constant or a linear
  expression of one and the same symbol, and the sum fits `coeff·sym+offset`** (`S` + constant →
  `S+1519`, `S` + `S` → `2S`). **Mixing different symbols fails loudly**, as before.
- The change is three declaration-level checks plus the conformance cases: the exporter's
  `_cat_axis` (called from `_h_cat`, `tools/exporter/src/karume/aten_handlers.py`), the shape rule
  `_cat` (`tools/exporter/src/karume/shapes.py`), the TS check `assertCatAxis`
  (`packages/runtime/src/runtime/plan.ts`), and the shapes section of
  `packages/runtime/tests/fixtures/op-contracts.json`. `dim-grammar.json` is unchanged.
- **Kernels and layout invariants do not change**; ADR 0014's full-write invariant (output extent =
  sum of input extents) holds after binding as before.
- `slice` / `flip` **stay static-only** (no measured need; `assertStaticLayoutAxis` in `plan.ts`).
- Fixed in the same change: the exporter's `range_constraints` scan (`convert.py`) raised
  `AttributeError` on **derived dimensions** (a sympy `Add` such as `S+1519` in an input shape); only
  plain symbols are taken as keys now.

## Alternatives considered

- **Split self/context plus a flash-style softmax merge (option b)**: passes today without touching
  the vocabulary, but puts a hand-written amax/exp/sum merge into the core of attention, bypasses
  the `safe_softmax` path of ADR 0044, and adds 22% nodes (1498 vs 1223). Kept only as a fallback.
- **Fixed S = 750 (option d)**: 3.1 × the compute for a 10 s utterance (S = 250). Rejected.
- **All four axes symbolic (option 2 of the recon)**: impossible, torch.export rejects such sums.

## Consequences

- The Irodori DiT (G5) exports with torch's SDPA decomposition plus `safe_softmax` (execution form:
  ADR [0047](0047-irodori-dit-execution.md)).
- The expected reuse for KV-cache appends in Gemma 4 E2B decode did not happen: its KV cache uses
  the named state slots of ADR [0066](0066-generation-context-state-slots.md), not `cat`.
- Both contract implementations (TS / Python), the conformance cases and fault injection (a
  mixed-symbol cat really fails) move as one set.
