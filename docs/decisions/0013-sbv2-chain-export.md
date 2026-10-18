# 0013: Export strategy for the SBV2 chain

- Status: accepted (2026-08-02); golden-only targets re-graded to MAY on 2026-10-18
- Date: 2026-08-02
- Related: [recon](../research/2026-08-02-sbv2-chain-recon.md) / [0129](0129-test-layout-and-lanes.md) /
  recipe `tools/export-recipes/sbv2/export.py`

## Decision

- **Five emit targets.** `front` (enc_p + dp + sdp reverse in one graph, dynamic P, Pmax = 512) and
  `voice` (flow + dec) are the distribution parts; fusing them removes intermediate readbacks
  (measured in the prototype). Goldens are produced for every target. `dp` (the early E2E scaffold),
  `flow` and `dec` are golden-only targets and MAY be kept (re-graded 2026-10-18 to MAY: their only
  remaining use is the opt-in upstream-parity check, `KARUME_UPSTREAM_PARITY=1`, ADR 0129 addendum
  2026-10-17 (2)).
- **Random numbers and runtime knobs stay out of the graph.** The sdp reverse noise is the input
  `z_noise`; `noise_scale`, the sdp_ratio mix and durations are host work (same seed → deterministic goldens).
- **Patch layer in the recipe** (the model package is unchanged): (1) a branch-free equivalent spline;
  (2) relative-position attention as a gather (removes shape expressions such as 2P−1 — required for
  export); (3) the FFN's explicit pad folded into the conv padding (equal only for an odd, non-causal
  kernel — asserted). Patches replace class attributes process-wide, so **the pre-patch reference is
  taken once per process**: the CLI rejects `--verify` with emit, `--verify` takes one target, and the
  order is "fix every reference value → patch → compare".
- **Relative-position tables, two methods:**
  - front (P ≤ 512): baked at Tmax plus `sym_prefix_slice` (about 2 MB).
  - flow (T ≤ 4096): baking costs O(Tmax²) ≈ 134 MB, so `idx_k` / `valid` are **graph inputs**. The host
    TS builds them (`packages/models/src/sbv2/relattn-tables.ts`), pinned by a **byte-parity test**
    against the Python generator (`packages/e2e/tests/e2e_sbv2_relattn_parity_test.ts`). Tables are
    clamped and in range, so the gather OOB rule (NaN poisoning, ADR 0009) is not hit.
  - `window_size == 4` is asserted at load (`_assert_window_size`); a mismatch reads wrong embeddings
    without a shape error.
- **`sym_max` per target**: front and dp 512 (phonemes), flow / dec / voice 4096 (frames). A wrong value
  is silent, so the CLI holds the defaults (`TARGET_SYM_MAX`) and a deviation needs an explicit flag.
- **weight_norm is only in dec** (measured). `remove_weight_norm` (idempotent) runs before export, and
  f16 / int8 rounding applies to the **effective weights after removal** (`_fake_quant`). An assert pins
  that flow has no derived parameters.

## Alternatives considered

- Per-module export only: readback round trips and split dispatches fragment the chain.
- One graph for the whole chain: impossible — host length work (durations → y_mask) sits in between.
- Baking the flow tables too: a 134 MB constant and an artificial utterance cap. Rejected.

## Consequences

- The host TS gains a table generator, a new verification duty (Python parity, the `window_size` assert).
- The exporter CLI checks the exclusivity of verify options (a vacuous comparison is a false PASS).
