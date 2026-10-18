# 0036: Arbitrary and non-square resolutions (`WxH`, S-shaped DiT plus tiled VAE)

- Status: accepted (2026-08-05 — the maintainer accepted the four-pattern visual check: "all four
  images are fine"); partly superseded by [0038](0038-manifest-v1.md) §4 (decision 2, 2026-08-05 —
  the distribution keeps only S-shaped graphs and always tiles, so the two opt-in knobs are gone)
- Date: 2026-08-05
- Related: [0033](0033-vae-fixed-tile-decode.md) (tiled VAE decode) and
  [0034](0034-dit-dynamic-tokens.md) (S-shaped DiT) — this ADR is where they meet: only with both
  do the assets become independent of the resolution. It also closes detection limit 1 of ADR 0034
  (the h↔w swap of the rope base table; see Findings). The maintainer asked on 2026-08-04 for tests
  of 16:9 and 3:4 in both orientations.

## Context

Before this ADR the Anima pipeline generated only square images (512 or 1024), and the image side
was recovered from the pixel count. With an S-shaped DiT graph (ADR 0034) and tiled VAE decode
(ADR 0033) one asset set can serve any size within the limits of the rope table and the tile
decoder.

## Decision

1. **The resolution is spelled `WxH`** (`1344x768`; a square keeps the short form `512`). The
   entry point rejects a size outside the acceptance set and names the condition it breaks. The
   acceptance set is: each side a multiple of 16 (spatial compression 8 × patch 2); each side 512 to
   2048 px (lower bound: one VAE decoder tile of latent 64; upper bound: the rope base table of 128
   rows = latent 256); S = (W/16)(H/16) ≤ 16,384 (`Dim("S")` at export). The acceptance set lives in
   `packages/models/src/anima/resolution.ts` (`parseResolution`, `assertAcceptableResolution`),
   exported from `@karume/models/anima`. The CLI demo that first carried `--resolution` was removed
   in `bb35fe2a`; the export recipes keep the same spelling for `--resolution`.
2. **A non-standard resolution required both `--dit-graph dyn` and `--vae-tiling`**, and a missing
   knob was rejected by name instead of surfacing as a missing asset in `readModel`. The default path
   (static assets, no knobs) stayed byte-identical. (Superseded by ADR 0038 §4 on 2026-08-05: the
   distribution has only S-shaped graphs and always tiles, so both knobs no longer exist.)
3. **Dimensions travel with the value.** `squareSideOf`, which recovered a square side from the pixel
   count, is deleted. Without tiling the graph output shape is the source of truth; with tiling the
   tile geometry is. A final check that the decoded image size equals the requested resolution closes
   any swap (`packages/models/src/anima/pipeline.ts`, the size check after decode).
4. **The spelling is shared with the Python reference scripts**
   (`tools/export-recipes/anima/resolution.py`, used by `pipeline_ref.py` and `tiling.py`). The
   acceptance set is defined only on the TypeScript side; Python checks only the structural
   conditions. Rationale: writing the same set on both sides lets one side drift silently.

## Findings

1. Detection limit 1 of ADR 0034 is solved by non-square pairs. In a square, cos_h equals cos_w, so
   a position swap between the axes is invisible; with H' ≠ W' the two tables differ and a swap
   fails. A detection limit that depends on the model configuration can sometimes be recovered by
   test data with a different configuration.
2. The model declares `rope.max_size` h/w = 120 tokens (1920 px), below the accepted upper bound of
   2048 px (128 table rows, set by the t axis). Requests between 1920 and 2048 px use positions
   outside the trained range; they are accepted and their quality is not measured. Recorded as a
   by-design limitation in [limitations.md](../limitations.md) (ruling of 2026-09-04: the acceptance
   set is not narrowed for this reason); the code note is on `MAX_LATENT_SIDE` in
   `packages/models/src/anima/resolution.ts`.

## Consequences

- One asset set (S-shaped DiT, 512 VAE tile decoder, two text encoders) generates any aspect ratio
  from 512 to 2048 px.

## Addendum (2026-08-05) — verification and measurements

- New checks: rope tables for four non-square geometries match torch bit for bit (fault injection
  turns them red), a non-square end-to-end reference at 1344×768 with tolerances derived from the
  measured error, and unit tests of the acceptance set
  (`packages/models/tests/anima_resolution_test.ts`). The default-path PNG checks stayed
  byte-identical.
- All four patterns run at S = 4,032 with DiT time per step at the 1024 px square level. The final
  benchmark of the time stays 1024×1024 square
  ([research](../research/2026-08-05-final-perf-bench.md)).
