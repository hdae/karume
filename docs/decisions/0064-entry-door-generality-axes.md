# 0064: Two more axes at the op entry door — semantic scope and execution-model invariants

- Status: accepted (2026-08-15 — ruling on an external generality audit; amends
  [0059](0059-op-vocabulary-entry-doors.md) by addition, its single Core ATen test unchanged); "single
  output" in decision 2 superseded by [0068](0068-decode-exit-multi-output.md) (2026-08-17); decision
  1's no-retroactivity clause withdrawn 2026-10-18
- Date: 2026-08-15
- Related: [0055](0055-deform-conv2d.md) / [0056](0056-gru-scan.md) (axis A examples) /
  [0060](0060-row-block-attention.md) / [0012](0012-attrs-and-fused-ops.md) (empty-attrs contract)

## Context

- The door of ADR 0059 decides whether an ADR is needed, but once an ADR exists **the subset one model
  uses can enter the public vocabulary as the op's semantics**. `deform_conv2d` fixes stride /
  dilation / groups / offset_groups to 1 by leaving the fields out (ADR 0055); `gru_scan_reverse`
  expresses direction by **a split op name** because no mechanism validates a bool attr (ADR 0056).
  Both fail loudly, but if narrowing the implemented capability and narrowing the semantics become
  one habit, new models bring a variant explosion and debt in the public contract.
- Core ATen ops enter with a ledger NOTE only, yet **some break execution-model invariants**: `topk` /
  `sort` (`Tag.core=True` on torch 2.13.0) return values + indices, which broke "every op has a
  single output" (then checked in `ops.ts`, one output per NodePlan) and need changes across
  planner, dtype, shape, recipe and executor. The door let this through with one NOTE.
- ADR 0059 decision 6 ("exactly four" fusion rules) went stale with ADR 0060: inventories in ADRs rot.

## Decision

1. **Axis A — state the semantic scope.** The ADR or ledger NOTE of a new op (or of a contract
   extension of an existing op) **separates the semantic surface (what the op promises) from the
   implemented subset (what the current kernels accept)**. When an observed subset is fixed as the
   semantics, it says so and gives in one line the condition for generalising (which demand adds the
   field); "field absent = not in the vocabulary" is never implicit.
   Withdrawn 2026-10-18: "not applied retroactively; a re-ruling before publication is a release item"
   — the packages are published; the re-ruling stays open in `docs/backlog.md` (release section).
2. **Axis B — an addition that breaks an execution-model invariant MUST have an ADR, even when it
   belongs to Core ATen.** The invariants: single output (superseded by ADR 0068 decision 1 on
   2026-08-17: node-level multi-output is allowed through the six surfaces of 0068 — eight in the code, ADR 0068
   addendum 1) / static shapes (no
   data-dependent output shape) / full-write / 4-byte storage with three semantic dtypes / full-range
   write of resident tensors. The ADR states the impact (planner / executor / recipe / IR …) up front
   as a required section. This is an exception to step 3 of the procedure (core → ledger NOTE only).
3. **The current inventory of vocabulary and rules lives in the ledger and the source.** An ADR does
   not record one (or marks it as a snapshot). Fusion rules: `FUSION_RULES` in
   `packages/runtime/src/runtime/fusion.ts` and the fusion tests; ADR 0059 decision 6 is a snapshot.
4. **Splitting an op name to cover a missing mechanism is stated** (SHOULD). When a missing generic
   attr mechanism (bool attrs and the like) is the reason for a separate op name, the ADR says so and
   gives the condition for merging once the mechanism exists — a duty to record so that
   `gru_scan_reverse` is not repeated, not a ban on splitting.

## Consequences

- [op-vocabulary.md](../op-vocabulary.md) carries axes A and B (exception in step 3, scope statement in
  step 4) and points at the source for fusions; ADR 0059 carries an amendment note (body unchanged).
- `topk` and runtime multi-output were the first application of axis B (ADR 0068, 2026-08-17).
