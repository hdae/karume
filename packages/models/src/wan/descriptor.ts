/**
 * Wan's generation descriptors — the values that differ between generations and cannot be derived
 * from the graph declarations, gathered in one place per generation (ADR 0121 decision 10). Values
 * derivable from the graph declarations, such as the latent channel count, the tile edge and the
 * spatial ratio of a graph's input to its output, do not live here (they are read from the asset
 * declarations).
 *
 * NOTE: the `export`s are for the pipeline internals, the tests and gpu-lab (removed in 0.15) (not
 * exposed on `mod.ts` / the subpath surface — ADR 0008).
 */

import {
  WAN22_LATENTS_MEAN,
  WAN22_LATENTS_STD,
  WAN_LATENTS_MEAN,
  WAN_LATENTS_STD,
  type WanLatentStats,
} from "./latents.ts";

/**
 * The DiT's graph input form (ADR 0121 decision 3). `"t2v"` = Wan2.1's 5 inputs (`tokens`,
 * `timesteps_proj`, `encoder_hidden_states`, `rope_cos`, `rope_sin`), `"ti2v"` = Wan2.2 TI2V's 7
 * inputs (the 5 + `timesteps_proj_condition`, `condition_mask`).
 */
export type WanDitInputForm = "t2v" | "ti2v";

export type WanGenerationDescriptor = {
  /**
   * The statistics for denormalization before the VAE (they exist only in the VAE config — in neither
   * the IR nor the distribution assets; the tables are in `latents.ts`).
   */
  readonly latents: WanLatentStats;
  /**
   * The accepted sizes (used by the request check and by the family admission check that verifies
   * every size can be covered by tiles).
   */
  readonly acceptedSizes: readonly { readonly width: number; readonly height: number }[];
  /** The accepted frame-count range (inclusive, 4n+1 only — the VAE's temporal compression). */
  readonly minFrames: number;
  readonly maxFrames: number;
  /**
   * The size and frame count used when the request omits them (inside the accepted set). The
   * distribution does not declare them — the defaults for step count, guidance and shift are the
   * manifest's `pipelineConfig` (`config.ts`).
   */
  readonly defaults: { readonly frames: number; readonly width: number; readonly height: number };
  /**
   * The output frame rate (a fact of the upstream generation, not a knob — not put in the manifest;
   * ADR 0121 decision 8).
   */
  readonly fps: number;
  /**
   * The upstream VAE's `patch_size` (the factor of the host unpatchify; 1 when absent — ADR 0121
   * decision 6). A factor of the spatial compression (spatial ratio of the graph's input to output ×
   * this factor). Assuming an RGB exit it could also be derived from the VAE exit's channel count
   * (3·p²), but it is kept as an upstream fact and matched against the declaration at admission
   * (`assertWanVaeMatchesGeneration` in `tile-decode.ts`).
   */
  readonly vaePatchSize: number;
  /**
   * The upstream DiT's input form (the TI2V DiT has a condition-side time and a condition mask). Like
   * `vaePatchSize` it is kept as an upstream fact and matched against the graph declaration at
   * admission (`ditContract` in `dit-loop.ts`).
   */
  readonly ditInputForm: WanDitInputForm;
};

/**
 * The descriptor of Wan2.1 (T2V 1.3B).
 *
 * Accepted set (ADR 0118 decision 7 — 832×480 / 480×832 × 4n+1 frames, 5–81). Acceptance-tested:
 * 33 frames at 832×480 (steps 3 / 5 / 6), 81 frames (step 8), and the VAE at 480×832 (step 5).
 *
 * Whether 81 frames fit is decided by the VRAM of the DiT stage alone — as measured in the NOTE at
 * the top of `pipeline.ts`, the allocation comes back right after the DiT Session is folded, and the
 * DiT and VAE stages do not overlap (peak at the switch = peak of the DiT stage). Step 8's
 * measurement (B570, a full 81-frame 50-step run, fdinfo) gave a DiT stage peak of 7.31 GiB and a VAE
 * stage peak of 3.78 GiB (ADR 0118 "Step 8 results"). MUST: the rejection message states only the
 * limit (81) (step numbers mean nothing to users).
 *
 * The default is the first target, 832×480 with 33 frames (a fact of the accepted set). fps 16 and no
 * `patch_size` (1) are upstream Wan2.1's values.
 *
 * MUST: when changing the accepted set, change the model card (`WAN_ACCEPTED_SIZES` / `WAN_FRAMES` in
 * `tools/export-recipes/wan/card.py`) and `tests/fixtures/wan-card-limits.json` to the same values —
 * the card keeps a copy of this fact, which is not in the manifest, so the tests on both sides of the
 * fixture (wan_pipeline_test.ts and the recipe's test_distribution.py) turn a one-sided update red.
 */
export const WAN21_GENERATION: WanGenerationDescriptor = {
  latents: { mean: WAN_LATENTS_MEAN, std: WAN_LATENTS_STD },
  acceptedSizes: [
    { width: 832, height: 480 },
    { width: 480, height: 832 },
  ],
  minFrames: 5,
  maxFrames: 81,
  defaults: { frames: 33, width: 832, height: 480 },
  fps: 16,
  vaePatchSize: 1,
  ditInputForm: "t2v",
};

/**
 * The descriptor of Wan2.2 (TI2V 5B) (only 1280×704 / 704×1280 × 4n+1 frames, 5–121 — ADR 0121
 * "Addendum (2026-10-05) — accepted sizes moved to the two official sizes").
 *
 * The accepted sizes are only the sizes the official implementation supports (Wan2.2's
 * `SUPPORTED_SIZES` — the two orientations the official code calls 720P). The pinned Diffusers
 * pipeline's default 832×480 is outside them, and when actually generated and viewed all three runs
 * (practical preset shift 3, reference preset shift 3, practical preset shift 5) broke (thick
 * outlines and flat fills). 1280×704 came out photorealistic with the practical preset at shift 5
 * (same seed and prompt). So 832×480 / 480×832 are not accepted. Stage 5's tile references at
 * 832×480 remain as numeric comparisons that do not depend on the accepted sizes (the tile-planning
 * entry `planWanGenerationTiles` does not look at the accepted set).
 *
 * The limit of 121 frames is the default of the official implementation (the sampler of the official
 * Wan2.2 repository — ADR 0121's upstream study §4.1; distinct from the default 81 of Diffusers'
 * `Pipeline`). On the development machine (RTX 3080 Ti, 12 GiB, Deno) 1280×704 with 50 steps
 * completed with the practical preset (VRAM peak of the DiT stage 11,361 MiB (nvidia-smi, whole GPU),
 * about 0.9 GiB below the physical capacity), and it also completed in Chrome on the RTX 5070 Ti
 * (ADR 0121's addenda "RTX 3080 Ti lanes and full verify" and "RTX 5070 Ti Chrome"). The development
 * machine's reference values (sha rows) come from the case of 2 steps × 121 frames (the opt-in in
 * `e2e_wan_ti2v_pipeline_test.ts`). On the B570 class (Deno's total allocation ceiling about 9.4 GiB
 * (9,600 MiB)) the verified length goes up to 57 frames (49 frames with both presets, 57 also
 * completed with the practical preset), and longer lengths are unsupported (docs/limitations.md).
 * They are unsupported because no length above 57 was run on the B570, and whether they fit is
 * speculation: ADR 0121's capacity table estimated only two points — at 81 frames the practical
 * preset does not fit, and at 121 frames the practical preset's DiT diag value of 10.64 GiB (measured
 * on the 3080 Ti) exceeds the ceiling, and the reference preset also exceeds it once the non-DiT
 * allocations are added to the extrapolated DiT diag value of 9.16–9.50 GiB. 61–77 and 85–117 have
 * no estimate at all. The accepted set is not per machine, so such a machine accepts them too, and if
 * they do not fit the run fails midway.
 *
 * The default 1280×704 with 33 frames is the maintainer's ruling (2026-10-09 — ADR 0121 addendum
 * "stage 7 results"). It runs with the defaults on a B570-class machine (up to 57 frames) too, and
 * takes about 13 minutes per video on this development machine. fps 24 and `patch_size` 2 are
 * upstream Wan2.2's values (the tables of denormalization statistics are in `latents.ts`).
 *
 * MUST: when changing the accepted set, change the model card (`WAN22_ACCEPTED_SIZES` /
 * `WAN22_FRAMES` in `tools/export-recipes/wan/card.py`) and `tests/fixtures/wan-ti2v-card-limits.json`
 * to the same values — the card keeps a copy of this fact, which is not in the manifest, so the tests
 * on both sides of the fixture (wan_pipeline_test.ts and the recipe's test_distribution.py) turn a
 * one-sided update red.
 */
export const WAN22_TI2V_GENERATION: WanGenerationDescriptor = {
  latents: { mean: WAN22_LATENTS_MEAN, std: WAN22_LATENTS_STD },
  acceptedSizes: [
    { width: 1280, height: 704 },
    { width: 704, height: 1280 },
  ],
  minFrames: 5,
  maxFrames: 121,
  defaults: { frames: 33, width: 1280, height: 704 },
  fps: 24,
  vaePatchSize: 2,
  ditInputForm: "ti2v",
};
