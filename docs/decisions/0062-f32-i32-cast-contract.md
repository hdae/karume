# 0062: f32 → i32 cast guarantees truncation in range; out-of-range values and NaN are undefined

- Status: accepted (2026-08-15 — records a ruling previously kept in `docs/limitations.md`)
- Date: 2026-08-15
- Related: [0009](0009-dtype-i32-bool.md) (fixed truncation, left out-of-range values and NaN open)

## Context

- For out-of-range or NaN input, both WGSL `i32(f32)` and torch `.to(int)` give
  implementation-defined results; the specifications do not guarantee that they agree.
- The fail-loudly invariant would call for a per-element range check, but a cast is a single
  memory-bound pass; a check would make it compute-bound and every cast would pay a constant cost.

## Decision

- **In range, the cast truncates toward zero as torch does, and agreement is guaranteed** (ADR
  0009's contract, restated; the exporter states it in `_assert_cast_target`,
  `tools/exporter/src/karume/ops.py`).
- **Out-of-range values and NaN are undefined** (no agreement claimed among GPU, CPU and torch).
- **There is deliberately no per-element range check** (it would change the performance character —
  an explicit exception to the fail-loudly invariant). The contract is that **the caller or the
  model clamps** values that may fall out of range first.

## Consequences

- This is the only cast path with a written exception to fail loudly. A new cast variant is ruled
  on individually along the same axis (does it change what bounds the pass); i32 → f32 is out of
  scope (every value is representable). [limitations](../limitations.md) points here.
