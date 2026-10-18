# 0057: Binding symbols from derived dimensions of input shapes

- Status: accepted (2026-08-14); states shapes became a second binding source later (ADR
  [0066](0066-generation-context-state-slots.md), addendum of 2026-08-17, item 7).
- Date: 2026-08-14
- Related: [0010](0010-symbolic-constant-folding.md) (dimension language `coeff·sym+offset`) /
  [0014](0014-layout-ops-full-write.md) (no symbolic axes before measured demand) /
  [0046](0046-cat-symbolic-axis.md) (relaxed only `cat`; `slice` / `flip` stay static) /
  [0056](0056-gru-scan.md) (`gru_scan`, which needed this). Specification: "shape and dimension
  language" in [ir-v2.md](../ir-v2.md).
- Demand: the vowel-detector CRNN (`tools/export-recipes/vowel_detector/export.py`) starts with
  `Conv1d(kernel=5, stride=2, padding=2)`. A plain `Dim("T")` length gives the extent
  **`((T−1)//2)+1`** (floor division), outside the dimension language; `2*Dim("T")` makes the conv
  output exactly `T`, but **`T` never appears plain in an input**.

## Context

IR v1 accepted a declared symbol only if it appeared **plain (coefficient 1, offset 0)** in some input
shape: a binding is taken from a dimension position of an input, never solved from an element count,
because **that solution is not unique** (the element count of `[T, S]` does not determine T and S).

The vowel-detector CRNN hits this rule. With plain `Dim("T")` **`convert` fails** (floor division);
with `2*Dim("T")` the only input is `[1, 2T, 83]` and the exporter's IR verification **fails because
'T' does not appear plain in an input shape** (measured). Both workarounds are poor: a rank-4 input
`[1, T, 2, 83]` reshaped to `2T` adds a meaningless axis to the public API; binding only through an
explicit seed to `session.run` leaves the graph alone unable to bind, failing the IR acceptance rule.

## Decision

### 1. Binding sources widen from "plain" to "a linear expression at a dimension position"

At a **dimension position** holding `coeff·sym+offset` (one symbol, coeff ≥ 1, offset ≥ 0), the size
gives **exactly one** solution

    sym = (size − offset) / coeff

The ban on solving from element counts concerns several symbols in one dimension (`[T·S]`); its reason,
non-uniqueness, is absent for a linear expression per dimension position. The rule becomes
"**appears at a dimension position of an input shape**". The static `slice` / `flip` axes stay (ADR
0046); only how bindings are taken changes, not the dimension language or any op contract.

### 2. A size that does not divide fails loudly (no rounding)

If `(size − offset)` is negative or not divisible by `coeff`, the input does not have the declared
form and no binding exists. **No rounding or truncation**: the run fails with an `ExecutionError` naming
the input, the dimension, the measured size, the declared form and the requirement (for example 285
against `2T` needs a length ≥ 0 divisible by 2 — `bindSymbols` in `packages/runtime/src/runtime/plan.ts`).
Rounding would let an odd length run through a `2T` graph with a last element that means something
else, while the caller believes the graph accepted it. For vowel-detector this is what the runtime
contract "the number of 10 ms frames is even" rests on.

### 3. Implementation: three places and one inverse function

- `packages/runtime/src/format/dims.ts` — `solveDim(expr, size)`, the inverse of `evalDim`
  (`undefined` when there is no solution).
- `packages/runtime/src/runtime/plan.ts` — `bindSymbols` solves with `solveDim`. The existing checks
  stay: conflicting bindings of one symbol fail, and the second pass re-evaluates every declared
  dimension (derived ones included) against the actual inputs.
- `packages/runtime/src/format/ir.ts` / `tools/exporter/src/karume/verify.py` — the acceptance rule
  becomes "appears at a dimension position". **A symbol that appears only in values is still rejected**
  (no binding source at run time). States-shape dimension positions are now a second binding source
  (`_check_symbol_bindability` in `verify.py` — ADR 0066).

An explicit seed (`session.run(inputs, bindings)`) still takes precedence and fails if it disagrees
with the solved value. The distribution pipeline (`VowelDetectorPipeline`) passes the host-counted
20 ms frame count, and the real-weight E2E (`packages/e2e/tests/e2e_vowel_detector_chain_test.ts`)
**passes none and lets the runtime solve it** — both routes are checked.

## Consequences

- The vowel-detector CRNN runs **any length with one graph** (with ADR 0056, the goal of that work);
  the distribution went from 4 length buckets, 34,088,454 B, to one graph, 2,759,461 B (−91.9%).
- An odd graph input length **had no check anywhere** before (the bucket era padded zeros on the
  right); it now fails at binding.
- **Graphs that were rejected now pass** (for example an input with only `2T`); graphs with a plain
  form behave as before, so there is no compatibility problem.
- Unverified: large `coeff` (`16T` and the like) in practice; only `2T` is measured.
