# 0033: Fixed-tile VAE decode (host glue only)

- Status: accepted (2026-08-04; the seam visual check was accepted by the maintainer); partly
  superseded by [0038](0038-manifest-v1.md) §4, which is still canonical (decisions 1 and 6: the
  distribution has one VAE decoder file and always tiles), by this ADR's addendum of 2026-09-02
  (P-3: decision 2 replaced) and, for decision 5, by its addendum of 2026-09-05 (ruling 9a) and
  [0129](0129-test-layout-and-lanes.md) addendum 2026-10-17 (2) (upstream parity is opt-in).
  Decisions 3 and 4 and the detection limits are in force.
  The tiled decode is a declared approximation of the reference level ([0147](0147-numerics-and-kernel-adoption.md)
  decision 10, 2026-11-10); oracle, distance and reason are recorded there.
- Date: 2026-08-04
- Related: recon: [2026-08-03-dynres-vae-tiling.md](../research/2026-08-03-dynres-vae-tiling.md)
  (§1.2 structural invariance of the graph, §4 tiling design, §4.2 the announced deliberate
  deviation in the scan) / [0016](0016-anima-chain-export.md) (resolution-baked static graphs — this
  ADR crosses resolutions **outside** them) / [0004](0004-execution-model.md) (per-run arena — the
  reason a per-tile peak can be measured on its own)

## Context

The VRAM peak of the 1024 px chain was set by the VAE decode (transient 1,536 MiB, VRAM 6,725 MiB),
not the DiT, and at 2048 px a non-tiled decode cannot hold the attention buffer `[1,65536,65536]` at
all. The recon measured that the VAE decoder graph is structurally invariant in resolution: the
512 px and 1024 px `model.safetensors` match in node list and weight bytes, and differ only in the
536 bytes of declared intermediate shapes. **The 512 px asset can be used as is as a tile decoder
for 64 × 64 latents.**

## Decision

1. **Host glue only.** Nothing is added to the runtime (`packages/runtime/src/`) or the IR; the
   tiling lives in `packages/models/src/anima/tiling.ts`. It was introduced as an opt-in demo knob
   (default off, so the default path stayed byte-identical). (Superseded by ADR 0038 §4 on
   2026-08-05: the distribution and the models pipeline always tile; there is no knob.)
2. ~~**Tile geometry: evenly spaced, snapped placement.** Starts are evenly spaced from 0 and the
   last start snaps to `extent − tile`. The smallest tile count meeting "overlap ≥ 8 latents
   (64 sample px)" is chosen and the stride derived from it (`(extent − tile) % stride == 0` holds by
   construction). 1024 px (latent 128) = 3 × 3 = 9 tiles, stride 32, blend 256 px; 512 px
   degenerates to one tile. diffusers' `range(0, H, stride)` scan (short last tile) cannot be fed
   to a fixed-shape graph, so this **deviates on purpose** (as announced in recon §4.2).~~
   (Replaced by the addendum of 2026-09-02, P-3: the divisibility constraint is removed; the reason
   for the deviation and the snap of the last tile carry over.)
3. **The blend is the linear ramp of diffusers' `blend_v` / `blend_h`** (sample space, vertical then
   horizontal, in place, matching f32 rounding positions). **Pasting is by region assignment**
   (tile i owns `[starts[i], starts[i+1])`, the last one up to the end). Carrying upstream's
   "truncate to stride width, then crop the whole" into snapped placement **loses the last 256 px**
   (fault injection turned red; the implementation leg caught the error in its instructions).
4. **Degenerate check: with one tile (latent 64) the tiled path MUST be Uint32-identical to the
   non-tiled path.** The identity is fixed by structure, not by a tolerance (a tolerance would miss
   one ULP). Current check: `packages/models/tests/anima_tiling_test.ts:269-277`.
5. **The reference is a torch implementation of the same geometry** (now
   `tools/export-recipes/anima/tiling.py`; `vae.enable_tiling()` is not used and `use_tiling`
   being true fails loudly; the blend formula is a verbatim port, and pytest fixes its bit match
   with the real one). The geometry metadata (starts / blend widths) is written to `tiling.json`,
   and the TS side was to check the geometry itself, not only the numbers. (Superseded by the
   addendum below and ADR 0129 addendum 2026-10-17 (2): numeric comparison with upstream is
   opt-in.)

   > **Addendum (2026-09-05, ruling 9a).** The TS-side geometry check is not implemented and will
   > not be. TS re-derives the geometry from the asset's input / output shapes (`planTileAxis` /
   > `blendExtentAt` in `packages/models/src/anima/tiling.ts`), and agreement is held by a double
   > freeze: Python's `MIRRORED_STARTS` (`tools/export-recipes/anima/tests/test_tiling.py`) and TS's
   > literal table of the 97 accepted sides (`AXIS_STARTS` in
   > `packages/models/tests/anima_tiling_test.ts`) (review item 1-13 of 2026-09-03). The Python
   > module moved from `tools/exporter/anima_tiling.py` to `tools/export-recipes/anima/tiling.py`
   > with the core / recipe split of ADR [0065](0065-exporter-core-recipe-split.md).
6. When tiling, the **512 px `vae_decoder` asset (f16 series) is opened** whatever the resolution;
   the demo chooses the asset, and the tile shape comes from the opened asset's input / output
   shapes — no literal 8 or 64 in the demo. (Superseded by ADR 0038 §4 on 2026-08-05: there is one
   VAE decoder file and no per-resolution or per-dtype series to choose. Still in force: the tile
   shape is read from the asset's input / output shapes, `planVaeTiling`.)

## Measurements (2026-08-04, 1024 px, production latents)

Decode time 1,451.2 → 3,467.8 ms (2.39 ×, slower); intermediate-buffer peak 1,536.0 → 384.0 MiB
(1/4) with 9 tiles (still 3 × 3 under the P-3 rule). **A memory lever, not a time lever** (the
2.25 × pixel redundancy shows up almost entirely as time); above 2048 px it is the only way.
Tiling itself is an approximation (attention's receptive field ends at the tile, as in upstream
`tiled_decode`): torch tiled vs non-tiled maxAbs 5.07e-2, three orders above the implementation
error against torch (maxAbs 1.642e-5). The default-path sha256 checks were unchanged. Full record:
commit `99a3c649`, this file.

## Detection limits

1. **Making the pasted region too large (always the full tile width) cannot be detected
   numerically.** In row-major order the following tile overwrites the excess and the final image
   does not change by one bit (injection I5 stayed green). The harmlessness depends on the scan
   order, so **pasting in parallel or in reverse turns it into silent wrong values**; the region
   comment of `assembleTiles` fixes it as a MUST. Too small (missing edge) is detected (I4 red).
2. **The degenerate check detects identity, not geometry.** Geometry and blend errors (I1–I4) are
   trivially identical with one tile, so it does not reach them; it catches only errors that break
   the identity of cutting and pasting (I6 / I7; shown non-vacuous by injection). The geometry
   detectors are the analytic tests (a synthetic decoder that depends on the position inside the
   tile) and the double freeze of decision 5's addendum.
3. Blend test data MUST give each tile different values in the overlap — with identical overlapping
   values a reversed ramp stays green (fixed in the test docs).

## Consequences

- The 1024 px decode VRAM peak drops to the 512 px level (transient 1,536 → 384 MiB), opening the
  way to 2048 px; the DiT side was made resolution-independent by
  [ADR 0034](0034-dit-dynamic-tokens.md).
- Time costs about +2.0 s at 1024 px; the final benchmark reported tiled and non-tiled separately
  ([2026-08-05-final-perf-bench.md](../research/2026-08-05-final-perf-bench.md)).
- Images differ from non-tiled (approximation), but the **visual check was accepted by the
  maintainer** (2026-08-04, "I don't notice any seams"; 1024 px production form, same seed), in line
  with the torch-to-torch difference of maxAbs 5.07e-2 not being visible.

## Addendum (2026-09-02) — P-3: decision 2 replaced, no divisibility constraint

Old decision 2 required strictly even spacing, so the count could only come from divisors of
`span = extent − tile`; sides with few divisors jumped to tens of tiles (1680 px: 74) and eight
resolutions had to be rejected by a "10 tiles per side" rule.

**New decision 2 (rounded evenly spaced, snapped placement):**

- Count = `ceil(span / (tile − minOverlap)) + 1` (span ≤ 0 degenerates to one tile). The only
  constraint is **the lower bound on overlap**; the spacing need not divide evenly.
- Start i = `round(i · span / (count − 1))` for i = 0 … count − 1; the last is exactly
  `extent − tile`.
- The blend width is **per pair** (`blendExtentAt(axis, scale, index)`); `TileAxis` has no `stride`
  field — folding a per-pair quantity that varies by up to one latent into one number would silently
  disagree with the placement.
- Uniformity means "**every pair overlaps at least the lower bound and spacings differ by at most
  one latent**". Rounding error stays within ±0.5, so the spacing takes only the `floor` and `ceil`
  values, and `ceil(span / (count − 1)) ≤ tile − minOverlap` satisfies the bound by construction;
  the implementation asserts the bound anyway (`planTileAxis`).

Sides whose count does not change keep their placement (1024², 512², 2048² and 67.3% of accepted
pairs are byte-identical); the eight rejected resolutions are accepted again
(`MAX_TILES_PER_AXIS` removed); decisions 3 and 4 still hold; `tiling.json` drops `stride` and
`blend_sample` becomes per pair. Measured 2026-09-03: VAE decode −22% at 1824 × 1248 (12 vs 15
tiles), visual check OK — [perf-ledger](../perf-ledger.md) P-3,
[2026-09-03-perf-spikes-p1-p3.md](../research/2026-09-03-perf-spikes-p1-p3.md); full addendum in
commit `95917831`.
