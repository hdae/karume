# 0014: Layout op extension — slice / cat / pad / flip and the full-write invariant

- Status: accepted (2026-08-02); the static `cat` axis relaxed by [0046](0046-cat-symbolic-axis.md) (2026-08-11)
- Date: 2026-08-02
- Related: [recon](../research/2026-08-02-sbv2-chain-recon.md) / [0004](0004-execution-model.md) /
  [0011](0011-layout-strategy.md)

## Context

ADR 0011 deferred `slice` because its only measured use folded away at Tmax. SBV2 breaks that premise
(the m_p / logs_p split of enc_p, the 96 / 96 coupling split, the spline, the Flip of flow / sdp).

## Decision

- Add the IR ops **`slice` (static axis and range) / `cat` (static axis) / `pad` (constant zero on the
  last axis) / `flip` (static axis)**. Symbolic slicing stays with `sym_prefix_slice`; `slice` is static
  only. Revised 2026-08-11 (ADR 0046): the `cat` axis may be a linear sum of one symbol.
- **New execution-model invariant: every kernel writes every byte of its output buffer (full-write).**
  The prototype's cat / pad relied on WebGPU zero-initialized buffers plus `copy_region`, which fails
  once a pool reuses buffers (reused buffers are not zeroed). Karume has the pool (ADR 0004), so no
  kernel may depend on zero initialization:
  - `slice`: the existing strided **read** copy family (its params already carry the offset ADR 0011
    foresaw).
  - `cat`: a new **strided write** copy family (contiguous read, output written through offset +
    strides — the dual of the read family), one write per input; together the inputs cover the whole
    output, so full-write holds **per node**.
  - `pad`: one kernel writes the whole output (copy inside, constant 0 outside), with no reliance on
    zero initialization or a no-reuse exception.
  - `flip`: a small dedicated kernel (index reversal on a static axis).
- The CPU reference implements each op independently.

## Alternatives considered

- **flip as a negative stride**: no new kernel, but it revises the params type, the determinism keys
  and every snapshot reviewed in M0.1. Rejected for the asymmetric blast radius.
- **flip as slice + cat**: reaches only axis length 2 (sdp), not the 192-channel flip of flow. Rejected.
- **cat / pad on zero initialization plus `copy_region`**: incompatible with pool reuse. A fresh
  no-reuse buffer for pad outputs would make the arena know op kinds. Rejected.

## Consequences

- Full-write is invariant 5 of the ADR 0004 arena, pinned by fault injection: reused pool buffers are
  poisoned before cat / pad / flip / conv_transpose1d run, and no poison may remain
  (`packages/runtime/tests/gpu_full_write_test.ts`).
- The strided write family follows the read family's determinism-key rule (element type, rank and
  workgroup size are in the key — `stridedWriteKey`, `packages/runtime/src/codegen/strided.ts`).
