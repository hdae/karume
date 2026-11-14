/**
 * End-to-end comparison of Wan2.2 TI2V-5B (`Ti2vPipeline`'s T2V — the wiring of ADR 0121 stage 6 and the distribution
 * form of stage 8) (real GPU).
 *
 * The distribution mirror `models/karume-wan2.2/` (assembled by `dist.py --pipeline wan-ti2v` — three `krm` files and
 * the two text assets of the 2.1 series) is passed to `Ti2vPipeline.fromPretrained` through `nodeDirectory`, and
 * `generate` runs on it (the public fetch surface — manifest resolution, family admission, sequential part reads; the
 * same entry as 2.1's stage 7). The references (`pipeline_steps.*`) are goldens no
 * distribution carries, read from the test-asset root (ADR 0139 decision 2).
 *
 * On a machine without the distribution, this e2e SKIPs explicitly (it prints the reason and the assembly command).
 * Turning the resulting all-SKIP into a FAIL is the gate test `packages/runtime/tests/distribution_gate_test.ts` (to
 * pass it on purpose, `KARUME_ALLOW_NO_DISTRIBUTION=1` — the same split as 2.1's end-to-end e2e).
 *
 * The comparison target is the recipe's two-step reference (`tools/export-recipes/wan/ti2v_few_step_ref.py` —
 * `goldens/wan2.2-ti2v-5b-i8-dyn/pipeline_steps.<case>.safetensors` of the test-asset root): the latents of diffusers'
 * `Pipeline(expand_timesteps=False)` run plainly for two steps in CPU f32 (with CFG, guidance 5.0, shift 5.0), and the
 * frames from passing them through stage 5's tiled decode → unpatchify → clamp (the weights: the DiT is the i8 RTN
 * fake-quant and the VAE holds f16-representable values = the same values as the reference preset `f16+dit8`). The
 * shape is 1280×704 with 17 frames (latents `[48,5,44,80]`, S = 4,400), and the two deciding cases and the one
 * acceptance case share it (the acceptance case's shape is not changed to push it out of the band — the maintainer's
 * ruling 2026-10-05). The initial noise is the reference's `latents_init` (torch's `randn`), injected through
 * `latents`.
 *
 * ## Checks (default lane)
 *
 * What this e2e holds are the **wiring** checks (swapped prompts, a guidance drift, the shift wiring, the step order,
 * the hand-off between stages, the direction of the 48 statistics), not the numeric check of one DiT forward (that is
 * the r check of stage 2's `e2e_wan_ti2v_dit_test.ts`).
 *
 * - **Host self-consistency** (no GPU): from the DiT outputs (cond / uncond) the reference recorded, the host's CFG +
 *   UniPC (shift 5.0) rebuilds the reference latents **bit for bit**. The metadata's timesteps equal
 *   `wanUniPcSchedule(2, 5.0)` exactly and the sigmas match in f32. The reference's `latents_mean` / `latents_std` are
 *   bit-identical to `WAN22_LATENTS_*` (pinning the direction of the denormalization statistics).
 * - **Band (opt-in upstream parity)** (ratio = max absolute difference ÷ the reference's max absolute value — the
 *   same metric as 2.1): the observation points are the latents after each step (`latents.0` / `latents.1` — the
 *   `denoise-step` event's `copyLatents`) and the clamped frames. The band is the worst of the two deciding cases
 *   (`band-boxing-cats` / `band-cat-dog-baking`) per observation point × 5 (rounded up to two significant digits), and
 *   the acceptance case (`accept-ferret` — seed and prompt differ from the deciding cases) must fall inside it. While a
 *   band is underived ({@link LATENT_RATIO_BANDS} / {@link FRAME_RATIO_BAND} `undefined`), each case records its
 *   ratios and fails, and the "band candidate" case prints the deciding worst × 5 (no placeholder value that passes
 *   and so has no detection power — the step 2 precedent). The judgement and the candidate run only with
 *   `KARUME_UPSTREAM_PARITY=1` ("Default run vs opt-in upstream parity" below); the three cases themselves run by
 *   default for their sha256 rows.
 *   MUST: do not change the band or the cases after seeing the acceptance result. If the acceptance case leaves the
 *   band, investigate the cause instead of widening the band.
 * - **Fault injection** (the acceptance case's initial noise, compared at `latents.1`, floor = {@link FAULT_MARGIN} ×
 *   the band): swapped positive / negative prompts, guidance 5.0 → 5.05, shift 5.0 → 3.0 (the shift wiring). Each one
 *   stops at the `end` event of the DiT stage and does not run the VAE. In T2V the condition mask and the conditioning
 *   timestep cannot show in the values by construction (both are the same value, the mask is all false — the same
 *   reason as stage 2), so they are not injected.
 * - **sha256 per-environment rows** (ADR 0106 — `fixtures/references/wan-ti2v.json`; 2.1's `wan.json` is not touched):
 *   the output frames (the uint8 RGB of all frames concatenated — the rule of `frameToRgba`). The default lane has 8
 *   rows = the reference preset's 6 (the 3 band cases, the seed path's 1280×704 and 704×1280, and the GPU path (umT5 on
 *   the GPU — a machine without the umT5 mirror `models/karume-umt5-xxl/` SKIPs it explicitly)) and one seed-path
 *   1280×704 row for each of the two practical presets (s16 and online — ADR 0132) ({@link PRACTICAL_SEED_CASES} — the
 *   same request as the reference preset's seed-path 1280×704, only the preset differs). The rows of the opt-in 50
 *   steps and 121 frames are in the sections below. An ID carries the preset, path, step count, case, size, frame count
 *   and shift ({@link caseIdOf} — ruling F12). A machine without the row skips that case's sha comparison (the artifact
 *   and the measured sha are kept), and the reference guard turns red (the existing rule). Rows are written after the
 *   band is green by running `KARUME_REFERENCE=write` on this file alone (ruling F8). The reference preset's rows are
 *   the values written in stage 6 by the version that **read the series directly** (`fromAssets`) (the RTX 3080 Ti rows
 *   were written by the frozen copy `128b511e` — the maintainer's ruling 2026-10-06), and this e2e, which goes through
 *   the distribution (`fromPretrained`), is required to match the same rows at HEAD (stage 8 of ADR 0121's acceptance)
 *   — the distribution is an independent copy of the series' `krm` files and text assets, and the preset and every
 *   request value are explicit, so a single differing bit is a regression of the fetch surface or of preset resolution.
 *   The practical presets' rows (added in stage 7) are written by this e2e through the distribution.
 * - **Row classes** (ADR 0110 decision 7 — decided by whether the manifest preset's `session` is empty): the rows of
 *   the reference preset `f16+dit8` are **reference rows** (frozen — not touched by commits that change the practical
 *   tier), the rows of the practical preset `f16+dit8-a8-attn8-s16` are **practical rows** (the detector of
 *   practical-tier regressions and of determinism — `rewrite` in the same commit when a commit changes practical-tier
 *   numerics on purpose). The CPU reference was taken only with the reference preset's values, so the practical presets
 *   carry no band and are pinned by the sha256 per-environment rows alone. Before the comparison and the write, a
 *   practical row of the default lane is checked to differ by at least 1 bit from the reference preset's artifact for
 *   the same request (the floor — ADR 0110 decision 5), so that a regression in which the preset's `session` does not
 *   arrive does not freeze the reference row's sha as a practical row. The numeric check of the practical presets
 *   (against the reference preset) is the self A/B check `e2e_wan_ti2v_ab_test.ts` (ADR 0121 stage 7 — 1280×704 with 33
 *   frames and the opt-in 121 frames).
 * - **The preset is explicit in every case** ({@link loadPipeline} requires it): the manifest's `defaultQuant` (the
 *   practical preset — the visual-check ruling 2026-10-06) is a value that can move with a quality ruling, and riding
 *   on the default preset would run the same ID with another preset's values.
 *
 * ## Default run vs opt-in upstream parity
 *
 * The band judgement (each reference case's latents and frames inside the band of the diffusers CPU f32 reference) and
 * the band candidate are upstream parity, so they are opt-in behind `KARUME_UPSTREAM_PARITY=1`
 * (`helpers/upstream-parity.ts` — ADR 0129 decision 4 and its addendum 2026-10-17 (2)). The judgement is gated at the
 * assertion, not by skipping the cases: the three reference cases run by default because their initial noise comes
 * from the fixtures and their outputs carry sha256 rows. Without the opt-in they log the ratios but neither assert the
 * band nor record it as a measurement; with the opt-in the band is asserted before the row is settled, so
 * `KARUME_REFERENCE=write` run together with the opt-in still refuses to write a row for an output outside the band.
 * Everything else runs by default: the GPU-free host self-consistency (bit identity, no tolerance), the sha256 rows
 * (the reference seat's band cases and seed path in both sizes, the practical seat with its floor against the
 * reference seat, the GPU text path) and the fault injections. The fault injections read the accept fixture and use
 * the band only as the margin a wiring fault must exceed (ratio > {@link FAULT_MARGIN} × band) — a check on the
 * wiring, not on parity.
 *
 * ## The 50-step run (env opt-in `KARUME_WAN_TI2V_FULL_PIPELINE=1` — not in the default lane)
 *
 * The precomputed path, `boxing-cats`, seed 42, shift 5.0, 1280×704, 33 frames and the manifest's default 50 steps run
 * once with the reference preset and once with each practical preset, writing the PNG of every frame, a contact sheet
 * (4×8 cells, 1/4 scale), the RGB artifact, the time per stage and the VRAM peaks at the stage boundaries to
 * `outputs/verify/<environment key>/<date>_wan-ti2v-pipeline-full/`, and checking 0 non-finite values and no device
 * lost (material for the maintainer's visual check — it also shows that the bytes equal the PNG of the generation
 * script for the same request). Both kinds of sha256 per-environment rows are kept (reference preset = reference row,
 * practical preset = practical row — "Row classes" above. Stage 6 only observed, so as not to freeze an output before
 * its numerics were confirmed, and stage 7 gave it rows together with the practical presets' numeric check (the self
 * A/B check) — ADR 0121 stage 7). Before the comparison and the write, a practical preset is checked to differ by at
 * least 1 bit from the reference preset's artifact of the same run (the floor — the same as the default lane's
 * practical rows, ADR 0110 decision 5) (the order reference preset → practical presets also produces this floor's
 * partner first). The cases register with the reference guard only under the opt-in (they do not run in the default
 * lane, so registering them would turn the reference guard red on a machine without the rows — the same as the 121
 * frames). The rows are written by adding `KARUME_REFERENCE=write` to the same command
 * (`KARUME_WAN_TI2V_FULL_PIPELINE=1` and `-t "50 steps"`). The results directory is kept apart from the default
 * lane's (`<date>_wan-ti2v-pipeline/`) because `results.json` is rewritten whole on every run.
 *
 * ## The 121-frame reference values (env opt-in `KARUME_WAN_TI2V_121F=1` — not in the default lane)
 *
 * The development machine's reference values for the accepted maximum of 121 frames (upstream's default — the accepted
 * range was widened after a 50-step run completed on the development machine's RTX 3080 Ti, ADR 0121 stage 10). The
 * request is the same as the seed path's 1280×704 case (the precomputed path, `boxing-cats`, seed 42, 2 steps,
 * guidance 5.0, shift 5.0) with only the frame count set to 121, and it checks the output shape, 0 non-finite values,
 * no device lost, the number of denoise steps and the sha256 per-environment row (a machine without the row skips the
 * comparison and keeps the artifact and the measured sha — the same as the seed cases). The preset is the reference
 * preset `f16+dit8` only (the practical presets' 121 frames run as the opt-in point of the self A/B check —
 * `e2e_wan_ti2v_ab_test.ts`). It takes about 15–20 minutes (about 12 minutes for the VAE alone), so like the 50-step
 * run it runs only under the env opt-in, and it registers with the reference guard only under the opt-in (it does not
 * turn the default lane's reference guard red). The results directory is `<date>_wan-ti2v-pipeline-121f/` (kept apart
 * from the default lane for the same reason as the 50-step run). The row is written by adding `KARUME_REFERENCE=write`
 * to the same command:
 *
 * ```
 * KARUME_WAN_TI2V_121F=1 vp test packages/e2e/tests/e2e_wan_ti2v_pipeline_test.ts -t "121 frames"
 * ```
 *
 * ## Device use (B570's `destroy()` does not return VRAM — docs/known-issues.md)
 *
 * The one `describe` of the default lane takes a single device (in its `beforeAll`) and runs every case as an `it`
 * (with `settleReleases` between cases). The opt-in 50 steps and 121 frames likewise use one device and run as a
 * **separate process** with `-t "50 steps"` / `-t "121 frames"` (so they do not carry what the default lane's device
 * left behind). Building (`fromPretrained`) only reads the distribution's parts and does not build Sessions (each
 * stage builds and disposes its own).
 *
 * ## Observations (not checks)
 *
 * The time per stage (wall clock — between `stage` events) and the VRAM at each stage boundary (`drm-total-*` of
 * `/proc/self/fdinfo` — `helpers/drm-usage.ts`). GPU time is not taken — a measuring device cannot open the VAE's
 * batch (building refuses it).
 *
 * Environments without the distribution SKIP explicitly (the host self-consistency check does not depend on the
 * distribution and still runs). Environments without the references (`pipeline_steps.*`) SKIP only the band, the
 * fault injection and the host self-consistency check explicitly (the seed path and the GPU path do not depend on the
 * CPU reference and still run). An environment with only some of the references FAILs (the same rule as 2.1's
 * end-to-end e2e).
 */

import { readFile, writeFile } from "@karume/test-support/fs";
import { afterAll, beforeAll, describe, it, test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { type DistributionSource, parseManifest, resolveSelection } from "@karume/hub";
import { nodeDirectory } from "@karume/hub/node";
import {
  type GpuContext,
  parseSafetensors,
  type SafetensorsFile,
  type SessionDiagnostics,
} from "@karume/runtime";
import { encodePng } from "../../models/mod.ts";
import {
  type GeneratedVideo,
  frameToRgba,
  type GenerateEvent,
  type GenerateRequest,
  type PipelineOptions,
  type Prompt,
  type RunComponent,
  Ti2vPipeline,
} from "../../models/wan.ts";
import { WAN22_LATENTS_MEAN, WAN22_LATENTS_STD } from "../../models/src/wan/latents.ts";
import {
  WAN_UNIPC_CONFIG,
  wanClassifierFreeGuidance,
  WanUniPcSampler,
  wanUniPcSchedule,
} from "../../models/src/wan/scheduler.ts";
import { acquireTestGpu, GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import { settleReleases } from "../../models/tests/helpers/settle-releases.ts";
import {
  type DrmTimeline,
  formatDrmUsage,
  monitorDrmUsage,
} from "../../models/tests/helpers/drm-usage.ts";
import {
  formatNvidiaTimeline,
  monitorNvidiaSmi,
  type NvidiaTimeline,
} from "../../models/tests/helpers/nvidia-smi.ts";
import {
  filePresent,
  TEST_ASSETS_FETCH,
  WAN_TI2V_SERIES_NAME,
} from "../../models/tests/helpers/wan-ti2v-dit.ts";
import { mirrorPresent, readMirrorAsset } from "../../models/tests/helpers/mirror.ts";
import { goldenDir, TEST_ASSETS, testAssetsFetched } from "@karume/test-support/test-assets";
import {
  readWanTi2vDistributionManifestText,
  WAN_TI2V_ASSEMBLE_COMMAND,
  WAN_TI2V_DIST_ROOT,
  WAN_TI2V_CONV8_QUANT,
  WAN_TI2V_ONLINE_QUANT,
  WAN_TI2V_PRACTICAL_QUANT,
  WAN_TI2V_REFERENCE_QUANT,
} from "../../models/tests/helpers/wan-ti2v-pipeline.ts";
import { contactSheet, writeWanFrames } from "../../models/tests/helpers/wan-video-artifacts.ts";
import { verifyRunningGpu } from "../../runtime/tests/helpers/environment.ts";
import {
  openReferences,
  referenceMismatchMessage,
  type ReferenceSettlement,
  registerReferenceGate,
  settleOrObserve,
  sha256Hex,
} from "../../runtime/tests/helpers/reference.ts";
import { openResults, runRecordedCase } from "../../runtime/tests/helpers/results.ts";
import { UPSTREAM_PARITY, UPSTREAM_PARITY_OPT_IN } from "./helpers/upstream-parity.ts";

/**
 * The band of the two-step run's latents (after each step — `[48,5,44,80]`) (ratio = max absolute difference ÷ the
 * reference's max absolute value). The index is the step. `undefined` = not derived yet (module doc "Checks" — each
 * case records its ratios and stops red, and the "band candidate" step prints a candidate).
 *
 * The derivation rule: per observation point, the worst of the two deciding cases × 5 (rounded up to 2 significant
 * digits). The candidate the first real run printed is written here together with the table of measurements (the two
 * deciding cases, the band, the acceptance case) (the same form as the doc of the constant of the same name in 2.1's
 * `e2e_wan_pipeline_test.ts`).
 *
 * Derivation (2026-10-05, B570, reference preset):
 *
 * | Point     | band-boxing-cats | band-cat-dog-baking | worst × 5 | band   | acceptance accept-ferret |
 * | --------- | ---------------: | ------------------: | --------: | -----: | -----------------------: |
 * | latents.0 |         7.185e-6 |            3.315e-6 |  3.593e-5 | 3.6e-5 |                 2.968e-6 |
 * | latents.1 |         1.035e-4 |            1.885e-4 |  9.424e-4 | 9.5e-4 |                 1.199e-4 |
 * | frames    |         2.276e-3 |            1.839e-4 |  1.138e-2 | 1.2e-2 |                 1.781e-3 |
 *
 * Fault-injection ratios (latents.1, floor = 2 × the band = 1.9e-3): swapped positive / negative prompts 1.321,
 * guidance 5.0 → 5.05 8.181e-2, shift 5.0 → 3.0 5.512e-1.
 *
 * MUST: do not change this value or the deciding cases after seeing the acceptance (`accept-ferret`) result (that
 * would break the independence of setting the band and accepting it). If the acceptance case leaves the band,
 * investigate the cause instead of widening the band.
 */
const LATENT_RATIO_BANDS: readonly [number | undefined, number | undefined] = [3.6e-5, 9.5e-4];

/**
 * The band of the two-step run's output frames (`[3,17,704,1280]` after the clamp — the reference is written after
 * the clamp too). `undefined` = not derived yet (the same rule as {@link LATENT_RATIO_BANDS}).
 *
 * MUST: do not change this value after seeing the acceptance result (the same as {@link LATENT_RATIO_BANDS}).
 */
const FRAME_RATIO_BAND: number | undefined = 1.2e-2;

/**
 * The floor of the factor by which a fault (swapped positive / negative prompts, a 1% guidance drift, a wrong shift)
 * must stay away from the band (2, the same as 2.1 and stage 2 — an injection that does not turn red is a sign that
 * the band is too wide).
 */
const FAULT_MARGIN = 2;

/**
 * The two-step run's conditions (`STEPS` / `GUIDANCE_SCALE` / `FLOW_SHIFT` / `LATENT_SHAPE` of the reference's
 * `ti2v_few_step_ref.py`).
 */
const STEPS = 2;
const GUIDANCE = 5;
const SHIFT = 5;
const WIDTH = 1280;
const HEIGHT = 704;
const FRAMES = 17;
const LATENT_SHAPE: readonly number[] = [48, 5, 44, 80];

/**
 * The references' place (goldens no distribution carries): the test-asset root's folder of the int8 DiT's series (ADR
 * 0139 decision 2; ADR 0121 stage 6 ruling 10).
 */
const STEPS_ROOT = goldenDir(WAN_TI2V_SERIES_NAME);

/**
 * The reference cases (the source of truth is `FIXTURE_CASES` in the recipe `wan/ti2v_few_step_ref.py`). Two
 * deciding (`band`) cases + one acceptance (`accept`) case.
 * MUST: do not change the deciding cases after seeing the acceptance result.
 */
const CASES: readonly { readonly name: string; readonly role: "band" | "accept" }[] = [
  { name: "band-boxing-cats", role: "band" },
  { name: "band-cat-dog-baking", role: "band" },
  { name: "accept-ferret", role: "accept" },
];
const ACCEPT_CASE = "accept-ferret";

/**
 * The case ID of a sha row (it carries the preset, path, step count, case, size, frame count and shift — ruling F12).
 * `head` is `<preset>[-gpu-text]-<step count>-<case>`.
 *
 * MUST: when the request (prompt, seed, step count, size, frame count, shift, preset, path) changes, change the ID too
 * (otherwise the row's value is compared with the sha of other conditions). This function spells the size, frame
 * count and shift from the same values as the request.
 */
const caseIdOf = (head: string, width: number, height: number, frames: number): string =>
  `${head}-${width}x${height}-${frames}f-shift${SHIFT}`;

/** The band cases' ID (reference preset, precomputed path). */
const bandCaseId = (name: string): string =>
  caseIdOf(`${WAN_TI2V_REFERENCE_QUANT}-2step-${name}`, WIDTH, HEIGHT, FRAMES);

/** The seed path's fixed request (the original text of the asset's row, seed 42). */
const SEED_PROMPT = "boxing-cats";
const SEED = 42;

/**
 * The seed path's two steps (no `latents` injected; the initial noise comes from the seed). There is no CPU
 * reference, so it is pinned only by completion, 0 non-finite values and the sha256 per-environment row. One case per
 * size (landscape and portrait — the two of the accepted set).
 */
const SEED_CASES: readonly {
  readonly id: string;
  readonly width: number;
  readonly height: number;
}[] = [
  { width: WIDTH, height: HEIGHT },
  { width: HEIGHT, height: WIDTH },
].map(({ width, height }) => ({
  id: caseIdOf(
    `${WAN_TI2V_REFERENCE_QUANT}-2step-${SEED_PROMPT}-seed${SEED}`,
    width,
    height,
    FRAMES,
  ),
  width,
  height,
}));

/**
 * The practical preset's seed-path two steps (the same request as the 1280×704 of {@link SEED_CASES}, only the preset
 * is the practical one — a practical row of the default lane; module doc "Row classes"). It carries no band (the
 * practical preset's numeric check is the self A/B check). It builds the preset's own pipeline, separate from the
 * reference preset's.
 */
const PRACTICAL_SEED_CASE_ID = caseIdOf(
  `${WAN_TI2V_PRACTICAL_QUANT}-2step-${SEED_PROMPT}-seed${SEED}`,
  WIDTH,
  HEIGHT,
  FRAMES,
);

/**
 * The online-attention preset's seed case (ADR 0132 decision 6): the same request as {@link PRACTICAL_SEED_CASE_ID} with
 * the preset `f16+dit8-a8-attn8-online` — a practical row of its own, written fresh (the reference rows do not move),
 * under the same floor (it differs from the reference preset's output for the same request by at least one bit).
 */
const ONLINE_SEED_CASE_ID = caseIdOf(
  `${WAN_TI2V_ONLINE_QUANT}-2step-${SEED_PROMPT}-seed${SEED}`,
  WIDTH,
  HEIGHT,
  FRAMES,
);

/**
 * The conv8 preset's seed case (ADR 0133 decision 6): the online bundle with the int8 VAE decoder on int8 activations
 * — a practical row of its own under the same floor against the reference preset (its decoder differs in storage and
 * compute, so the floor holds by construction; the same-weights floor of ADR 0110 §5, against the `i8` preset's
 * decoder, is the decoder's own A/B in `e2e_wan_ti2v_vae_a8_test.ts`).
 */
const CONV8_SEED_CASE_ID = caseIdOf(
  `${WAN_TI2V_CONV8_QUANT}-2step-${SEED_PROMPT}-seed${SEED}`,
  WIDTH,
  HEIGHT,
  FRAMES,
);

/**
 * The conv8 preset's seed case decoded in 32-latent tiles (ADR 0134 — `vaeTile: 32`, the decoder pair that
 * borrows the 16-latent pair's weights): the same request as {@link CONV8_SEED_CASE_ID} with the tile knob — a
 * practical row of its own. Its output must differ from the tile-16 case's (the knob reached the decoder: the
 * seams blend at other places). Its VAE stage seconds and the nvidia-smi VRAM peak are recorded next to the tile-16
 * case's ({@link Conv8VaeRecord} — ADR 0134 decision 7-4, the kill line is the orchestrator's reading, not a gate).
 */
const CONV8_VAE32_SEED_CASE_ID = caseIdOf(
  `${WAN_TI2V_CONV8_QUANT}-vae32-2step-${SEED_PROMPT}-seed${SEED}`,
  WIDTH,
  HEIGHT,
  FRAMES,
);

/** The practical presets' seed cases, run in this order after the reference preset's (the s16 preset first). */
const PRACTICAL_SEED_CASES: readonly {
  readonly id: string;
  readonly quant: string;
  /** `GenerateRequest.vaeTile` (omitted = the default 16). */
  readonly vaeTile?: 32;
}[] = [
  { id: PRACTICAL_SEED_CASE_ID, quant: WAN_TI2V_PRACTICAL_QUANT },
  { id: ONLINE_SEED_CASE_ID, quant: WAN_TI2V_ONLINE_QUANT },
  { id: CONV8_SEED_CASE_ID, quant: WAN_TI2V_CONV8_QUANT },
  { id: CONV8_VAE32_SEED_CASE_ID, quant: WAN_TI2V_CONV8_QUANT, vaeTile: 32 },
];

/** The VAE stage of one conv8 case at one tile (the record of ADR 0134 decision 7-4). */
type Conv8VaeRecord = {
  readonly vaeSeconds: number;
  /** nvidia-smi's GPU-wide peak during the `vae_decoder` stage (undefined without nvidia-smi or samples). */
  readonly vaePeakMiB: number | undefined;
  /** nvidia-smi's GPU-wide peak before the first stage (the baseline other processes hold). */
  readonly baselineMiB: number | undefined;
};

/**
 * The GPU-path case (umT5 i8 on the GPU — the fixed prompt's original text, seed 42, 2 steps, 1280×704). It has no CPU
 * reference (umT5's numeric check is `e2e_wan_umt5_test.ts`). The same string as the asset path's seed case; only the
 * path differs.
 */
const GPU_TEXT_CASE_ID = caseIdOf(
  `${WAN_TI2V_REFERENCE_QUANT}-gpu-text-2step-${SEED_PROMPT}-seed${SEED}`,
  WIDTH,
  HEIGHT,
  FRAMES,
);

/** The opt-in of the 50-step run (module doc). */
const FULL_PIPELINE = process.env.KARUME_WAN_TI2V_FULL_PIPELINE === "1";
/** The 50-step run's frame count (the default frame count). */
const FULL_FRAMES = 33;
/**
 * The 50-step run's step count. The request does not set it and leaves it to the manifest's default, so it is pinned
 * by the observed number of denoise steps (if the manifest's default changes, it disagrees with the ID's `50step`).
 */
const FULL_STEPS = 50;
/**
 * The 50-step run's presets. Ordered by allocation size, largest first: at S = 7,920 (1280×704, 33 frames) the DiT's
 * diag is 7.36 GiB for the reference preset > 7.25 GiB for the practical preset (the capacity table of the addendum to
 * ADR 0121 stage 2). What an earlier allocation leaves behind can make a later large allocation OOM (B570's delayed
 * `destroy()`), so the larger one goes first.
 */
const FULL_CASES: readonly { readonly id: string; readonly quant: string }[] = [
  WAN_TI2V_REFERENCE_QUANT,
  WAN_TI2V_PRACTICAL_QUANT,
  WAN_TI2V_ONLINE_QUANT,
].map((quant) => ({
  id: caseIdOf(`${quant}-50step-${SEED_PROMPT}-seed${SEED}`, WIDTH, HEIGHT, FULL_FRAMES),
  quant,
}));

/** The opt-in of the 121-frame reference values (module doc). */
const LONG_CLIP = process.env.KARUME_WAN_TI2V_121F === "1";
/**
 * The 121-frame case's frame count (the accepted maximum = upstream's default). Held as a value, not read from the
 * descriptor — the ID's `121f` is the sha row's key, and the same ID must not run with another request's values if the
 * descriptor's maximum moves.
 */
const LONG_CLIP_FRAMES = 121;
/**
 * The 121-frame case (reference preset, precomputed path; it differs from the seed path's 1280×704 case only in the
 * frame count).
 */
const LONG_CLIP_CASE_ID = caseIdOf(
  `${WAN_TI2V_REFERENCE_QUANT}-2step-${SEED_PROMPT}-seed${SEED}`,
  WIDTH,
  HEIGHT,
  LONG_CLIP_FRAMES,
);

const ASSEMBLE_UMT5_COMMAND = "cd tools/export-recipes && uv run python dist.py --pipeline umt5";

/**
 * The text-embedding asset the references were made with (ADR 0121 decision 9): the mirror's `text_embeds` (ADR 0139
 * decision 1 — the distribution carries it, the same bytes as the series file the references were made from). The host
 * self-consistency check compares its sha with the references' meta `text_embeds_sha256`, so a copy that differs from
 * the asset the references were made with is caught here as well as by the GPU band cases (which take their prompts
 * from the same asset).
 */
const TEXT_EMBEDS = "text_embeds";

/**
 * umT5's distribution mirror (the target that `text_encoder` of the Wan2.2 manifest references across repositories —
 * the same mirror as 2.1).
 */
const UMT5_ROOT = new URL("../../../models/karume-umt5-xxl/", import.meta.url);
/** The manifest's component name (`TEXT_ENCODER` of `src/wan/pipeline.ts`). */
const TEXT_ENCODER = "text_encoder";

const fixtureUrl = (name: string): URL => new URL(`pipeline_steps.${name}.safetensors`, STEPS_ROOT);

/** The distribution's manifest (`undefined` when absent — the end-to-end comparison SKIPs). */
const DIST_MANIFEST_TEXT = await readWanTi2vDistributionManifestText();
const DIST_MANIFEST =
  DIST_MANIFEST_TEXT === undefined ? undefined : parseManifest(DIST_MANIFEST_TEXT);
const DIST_PRESENT = DIST_MANIFEST !== undefined;
if (!DIST_PRESENT) {
  console.warn(
    `[karume] the distribution mirror ${WAN_TI2V_DIST_ROOT.pathname} is absent, so Wan2.2's end-to-end comparison is SKIPPED. ` +
      `Assemble it: ${WAN_TI2V_ASSEMBLE_COMMAND} (the gate test distribution_gate_test.ts turns an all-SKIP into a FAIL)`,
  );
}
/**
 * The repo of umT5's cross-repository target (declared by part 0 of `text_encoder` in the distribution's default
 * selection — the key is not copied by hand). A distribution without umT5, or with umT5 in its own repo, is outside
 * the 2.2 distribution's declaration, so building the GPU path fails loudly ({@link umt5CrossRepo}).
 */
const UMT5_REPO =
  DIST_MANIFEST === undefined
    ? undefined
    : resolveSelection(DIST_MANIFEST).containers[TEXT_ENCODER]?.parts[0]?.repo;
const UMT5_PRESENT = filePresent(new URL("karume.json", UMT5_ROOT));
/** Whether the GPU path can be built (the distribution and the umT5 mirror are present). */
const GPU_TEXT_PRESENT = DIST_PRESENT && UMT5_PRESENT;
if (DIST_PRESENT && !GPU_TEXT_PRESENT) {
  console.warn(
    `[karume] umT5's distribution mirror ${UMT5_ROOT.pathname} (the cross-repository target of ${
      UMT5_REPO ?? TEXT_ENCODER
    }) is absent, so ` + `Wan2.2's GPU-path case is SKIPPED. Assemble it: ${ASSEMBLE_UMT5_COMMAND}`,
  );
}

/** The cross-repository mapping (the repo of umT5's container → the local mirror). */
const umt5CrossRepo = (): Record<string, DistributionSource> => {
  if (UMT5_REPO === undefined) {
    throw new Error(
      `in the default selection of the distribution ${WAN_TI2V_DIST_ROOT.pathname}, ${TEXT_ENCODER} is not a cross-repository reference ` +
        `(the 2.2 distribution references umT5 from karume-umt5-xxl). Reassemble it: ${WAN_TI2V_ASSEMBLE_COMMAND}`,
    );
  }
  return { [UMT5_REPO]: nodeDirectory(UMT5_ROOT) };
};
const FIXTURES_PRESENT = CASES.map(({ name }) => filePresent(fixtureUrl(name)));
const ANY_FIXTURE = FIXTURES_PRESENT.some(Boolean);
/**
 * A complete fetch of the pinned test-asset dataset makes a missing reference a FAIL (the completeness test below): the
 * dataset at the pin lacks a file the cases read. Without one, the cases SKIP (the assets gate is red then).
 */
const FETCHED = testAssetsFetched();
const MISSING_GOLDENS = FETCHED
  ? `the test-asset dataset at revision ${TEST_ASSETS.revision} lacks them`
  : `fetch: ${TEST_ASSETS_FETCH}`;
if (!ANY_FIXTURE) {
  console.warn(
    `[karume] no two-step reference (pipeline_steps.*) under ${STEPS_ROOT.pathname}, so Wan2.2's end-to-end band, ` +
      `fault injection and host self-consistency checks are SKIPPED (${MISSING_GOLDENS})`,
  );
}
/**
 * The host self-consistency check reads the mirror's `text_embeds` through its manifest, which needs the mirror and
 * every sibling it references (the umT5 mirror) — otherwise it SKIPs.
 */
const TEXT_EMBEDS_READABLE = mirrorPresent(WAN_TI2V_DIST_ROOT);
if (ANY_FIXTURE && !TEXT_EMBEDS_READABLE) {
  console.warn(
    `[karume] the mirror ${WAN_TI2V_DIST_ROOT.pathname} is absent (or not karume/5, or a sibling it references is ` +
      `absent), so Wan2.2's host self-consistency check (it reads the mirror's ${TEXT_EMBEDS}) is SKIPPED`,
  );
}

const references = openReferences(
  new URL("../../models/tests/fixtures/references/wan-ti2v.json", import.meta.url),
);
const results = openResults("wan-ti2v-pipeline");
/** The 50-step run's results directory (kept apart from the default lane — module doc). */
const fullResults = openResults("wan-ti2v-pipeline-full");
/** The 121-frame results directory (likewise kept apart from the default lane). */
const longClipResults = openResults("wan-ti2v-pipeline-121f");

/** A value a `beforeAll` assigns, read inside an `it` (fails loudly when the hook did not assign it). */
const assigned = <T>(value: T | undefined, what: string): T => {
  assert(value !== undefined, `${what} is not assigned (its beforeAll did not complete)`);
  return value;
};

test.skipIf(!ANY_FIXTURE && !FETCHED)(
  "Wan2.2 end-to-end references: all three two-step references are present",
  () => {
    assertEquals(
      FIXTURES_PRESENT,
      FIXTURES_PRESENT.map(() => true),
      `missing: ${CASES.filter((_, index) => !FIXTURES_PRESENT[index])
        .map(({ name }) => fixtureUrl(name).pathname)
        .join(", ")} (${MISSING_GOLDENS})`,
    );
  },
);

/** One reference (F32 tensors, shapes, metadata). */
type Fixture = {
  readonly tensor: (key: string) => Float32Array<ArrayBuffer>;
  readonly shape: (key: string) => readonly number[];
  readonly meta: (key: string) => string;
};

const readFixture = async (url: URL): Promise<Fixture> => {
  const bytes = await readFile(url);
  const file: SafetensorsFile = parseSafetensors(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
  const view = (key: string) => {
    const found = file.tensors.get(key);
    assert(found !== undefined && found.dtype === "F32", `${url.pathname}: '${key}' is not F32`);
    return found;
  };
  return {
    tensor: (key) => {
      const found = view(key);
      return new Float32Array(file.buffer, found.byteOffset, found.byteLength / 4);
    },
    shape: (key) => view(key).shape,
    meta: (key) => {
      const value = file.metadata.get(key);
      assert(value !== undefined, `${url.pathname}: metadata '${key}' is missing`);
      return value;
    },
  };
};

/** A summary of the difference (ratio = max absolute difference ÷ the reference's max absolute value). */
type Difference = {
  readonly maxAbs: number;
  /** The maximum per-element relative difference (elements where the reference is 0 are excluded — for the record). */
  readonly maxRel: number;
  readonly referenceMax: number;
  readonly ratio: number;
  readonly nonFinite: number;
};

const difference = (got: Float32Array, want: Float32Array): Difference => {
  assertEquals(got.length, want.length, "element count");
  let maxAbs = 0;
  let maxRel = 0;
  let referenceMax = 0;
  let nonFinite = 0;
  for (let index = 0; index < want.length; index += 1) {
    if (!Number.isFinite(got[index])) {
      nonFinite += 1;
      continue;
    }
    const absolute = Math.abs(got[index] - want[index]);
    maxAbs = Math.max(maxAbs, absolute);
    if (want[index] !== 0) maxRel = Math.max(maxRel, absolute / Math.abs(want[index]));
    referenceMax = Math.max(referenceMax, Math.abs(want[index]));
  }
  return { maxAbs, maxRel, referenceMax, ratio: maxAbs / referenceMax, nonFinite };
};

const formatDifference = (label: string, diff: Difference): string =>
  `${label} ratio ${diff.ratio.toExponential(3)} (maxAbs ${diff.maxAbs.toExponential(3)} / reference ${diff.referenceMax.toFixed(
    3,
  )})`;

/** Bit identity of two f32 arrays (the sign of NaN and -0 are told apart too). */
const bitsEqual = (got: Float32Array, want: Float32Array): boolean => {
  if (got.length !== want.length) return false;
  const gotBits = new Uint32Array(got.buffer, got.byteOffset, got.length);
  const wantBits = new Uint32Array(want.buffer, want.byteOffset, want.length);
  return gotBits.every((value, index) => value === wantBits[index]);
};

/** The uint8 RGB of all output frames concatenated (the sha256 artifact). */
const rgbBytes = (video: GeneratedVideo): Uint8Array<ArrayBuffer> => {
  const plane = video.width * video.height;
  const out = new Uint8Array(video.frames * plane * 3);
  for (let frame = 0; frame < video.frames; frame += 1) {
    const rgba = frameToRgba(video, frame);
    for (let index = 0; index < plane; index += 1) {
      out.set(rgba.subarray(index * 4, index * 4 + 3), (frame * plane + index) * 3);
    }
  }
  return out;
};

const countNonFinite = (values: Float32Array): number =>
  values.reduce((count, value) => count + (Number.isFinite(value) ? 0 : 1), 0);

/**
 * The observations of one generate (from the events — the latents after each step, the time per stage, the VRAM at the
 * stage boundaries).
 */
type Observed = {
  readonly video: GeneratedVideo;
  readonly latents: readonly Float32Array[];
  /** Stage name → wall-clock ms (from `start` to `end`). */
  readonly stageMs: Readonly<Record<string, number>>;
  readonly wallMs: number;
  readonly drm: DrmTimeline;
  /** GPU-wide nvidia-smi samples per stage (only when `observe` was asked for them). */
  readonly nvidia: NvidiaTimeline;
  readonly diagnostics: ReadonlyMap<RunComponent, SessionDiagnostics>;
};

/**
 * Runs generate once and collects observations from the events. With `abortAfterDenoise` it throws at the DiT stage's
 * `end` (after its Session is disposed) and skips the VAE stage (fault injection looks only at the latents — it does
 * not pay for the 28-tile VAE).
 */
const observe = async (
  pipeline: Ti2vPipeline,
  diagnostics: Map<RunComponent, SessionDiagnostics>,
  request: Omit<GenerateRequest, "onEvent">,
  options: { readonly abortAfterDenoise?: boolean; readonly nvidiaSmi?: boolean } = {},
): Promise<Observed | { readonly latents: readonly Float32Array[] }> => {
  const latents: Float32Array[] = [];
  const stageStarted = new Map<string, number>();
  const stageMs: Record<string, number> = {};
  const monitor = monitorDrmUsage();
  // GPU-wide samples per stage (the phase is the stage between its start and end events, "between"
  // after an end). Started before the generate and given one sample so the baseline is recorded.
  const nvidia = options.nvidiaSmi === true ? monitorNvidiaSmi() : undefined;
  await nvidia?.waitForSample(2000);
  const abort = new Error("stop after the DiT stage (fault injection)");
  diagnostics.clear();
  const started = performance.now();
  const onEvent = (event: GenerateEvent): void => {
    if (event.kind === "denoise-step") latents.push(event.copyLatents().data);
    if (event.kind !== "stage") return;
    const label = `${event.component} ${event.at}`;
    monitor.mark(label);
    monitor.enter(label);
    nvidia?.enter(event.at === "start" ? event.component : `after ${event.component}`);
    if (event.at === "start") stageStarted.set(event.component, performance.now());
    else stageMs[event.component] = performance.now() - (stageStarted.get(event.component) ?? 0);
    if (
      options.abortAfterDenoise === true &&
      event.component === "transformer" &&
      event.at === "end"
    ) {
      throw abort;
    }
  };
  try {
    const video = await pipeline.generate({ ...request, onEvent });
    const wallMs = performance.now() - started;
    return {
      video,
      latents,
      stageMs,
      wallMs,
      drm: monitor.stop(),
      nvidia: await nvidia?.stop(),
      diagnostics: new Map(diagnostics),
    };
  } catch (error) {
    monitor.stop();
    await nvidia?.stop();
    if (error === abort) return { latents };
    throw error;
  }
};

/**
 * The observation lines (time per stage, the VRAM peak per interval and the value at each boundary, the diagnostics
 * breakdown).
 */
const formatObserved = (observed: Observed): string[] => {
  const gib = (bytes: number) => `${(bytes / 2 ** 30).toFixed(3)} GiB`;
  const lines = [
    `wall ${(observed.wallMs / 1000).toFixed(1)} s, stages ${Object.entries(observed.stageMs)
      .map(([stage, ms]) => `${stage} ${(ms / 1000).toFixed(1)} s`)
      .join(", ")}`,
    ...[...observed.drm.peaks].map(
      ([phase, peak]) => `VRAM peak [${phase}] ${formatDrmUsage(peak)}`,
    ),
    ...observed.drm.marks.map(({ label, usage }) => `VRAM at [${label}] ${formatDrmUsage(usage)}`),
    ...(observed.nvidia === undefined
      ? []
      : [`nvidia-smi ${formatNvidiaTimeline(observed.nvidia)}`]),
  ];
  for (const [component, diagnostics] of observed.diagnostics) {
    lines.push(
      `${component}: weights ${gib(diagnostics.weights.allocatedBytes)}, backing ${gib(
        diagnostics.planBacking.residentBytes + diagnostics.planBacking.inputBytes,
      )}, geometry ${diagnostics.geometryProfile}, ${diagnostics.submit.submitCount} submits, max window mean ${
        diagnostics.submit.chunkBudget.maxWindowMeanMs?.toFixed(1) ?? "—"
      } ms, ${diagnostics.submit.chunkBudget.overBudgetChunks} over budget`,
    );
  }
  return lines;
};

/**
 * The record line of the conv8 VAE stage by tile (ADR 0134 decision 7-4: at 1280×704×17 the tile-32 stage is
 * rejected unless under 0.8× of tile 16; the VRAM peak is GPU-wide — other processes' allocations included, the
 * baseline before the first stage says how much).
 */
const conv8VaeLines = (records: ReadonlyMap<16 | 32, Conv8VaeRecord>): string[] => {
  const mib = (value: number | undefined) => (value === undefined ? "—" : `${value} MiB`);
  const lines = [...records].map(
    ([tile, record]) =>
      `conv8 VAE tile ${tile}: ${record.vaeSeconds.toFixed(2)} s · nvidia-smi peak [vae_decoder] ${mib(
        record.vaePeakMiB,
      )} (baseline ${mib(record.baselineMiB)})`,
  );
  const tile16 = records.get(16);
  const tile32 = records.get(32);
  if (tile16 !== undefined && tile32 !== undefined) {
    const ratio = tile32.vaeSeconds / tile16.vaeSeconds;
    lines.push(
      `conv8 VAE tile 32 / tile 16 = ${ratio.toFixed(3)} (kill line 0.8 — ${ratio < 0.8 ? "under" : "NOT under"})`,
    );
  }
  return lines;
};

const textOf = (
  prompts: readonly Prompt[],
  name: string,
  form: "prompt" | "normalized",
): string => {
  const entry = prompts.find((candidate) => candidate.name === name);
  assert(entry !== undefined, `the embedding asset has no '${name}'`);
  return entry[form];
};

/**
 * Builds the reference / practical preset's pipeline by reading the distribution through a source handle (neither the
 * network nor CacheStorage is involved). The caller always names the path and the preset (it never rides silently on
 * the default `"gpu"` or on the manifest's default preset — the same rule as 2.1's e2e). `"gpu"` passes umT5's
 * cross-repository target through the `crossRepo` mapping.
 */
const loadPipeline = (
  gpu: GpuContext,
  diagnostics: Map<RunComponent, SessionDiagnostics>,
  textEncoder: NonNullable<PipelineOptions["textEncoder"]>,
  quant: string,
): Promise<Ti2vPipeline> =>
  Ti2vPipeline.fromPretrained(
    textEncoder === "gpu"
      ? nodeDirectory(WAN_TI2V_DIST_ROOT, { crossRepo: umt5CrossRepo() })
      : nodeDirectory(WAN_TI2V_DIST_ROOT),
    {
      gpu,
      textEncoder,
      quant,
      onRunDiagnostics: (component, diagnosed) => diagnostics.set(component, diagnosed),
    },
  );

/** The common part of the two-step run's requests (each case adds the size, prompt and initial noise). */
const TWO_STEP = { steps: STEPS, guidance: GUIDANCE, shift: SHIFT, frames: FRAMES } as const;

test.skipIf(!ANY_FIXTURE || !TEXT_EMBEDS_READABLE)(
  "Wan2.2 end-to-end (host, no GPU): the reference metadata (embedding asset, σ / timestep, shift 5.0) and the 48 " +
    "statistics match TS, and CFG + UniPC rebuild the reference latents bit for bit from the reference DiT outputs",
  async () => {
    const embedsSha = await sha256Hex(await readMirrorAsset(WAN_TI2V_DIST_ROOT, TEXT_EMBEDS));
    const schedule = wanUniPcSchedule(STEPS, SHIFT, WAN_UNIPC_CONFIG.numTrainTimesteps);
    const mean = Float32Array.from(WAN22_LATENTS_MEAN);
    const std = Float32Array.from(WAN22_LATENTS_STD);
    for (const { name, role } of CASES) {
      const fixture = await readFixture(fixtureUrl(name));
      assertEquals(fixture.meta("role"), role, `${name}: role`);
      assertEquals(
        fixture.meta("text_embeds_sha256"),
        embedsSha,
        `${name}: the embedding asset the reference was made with`,
      );
      assertEquals(
        [
          fixture.meta("steps"),
          fixture.meta("guidance_scale"),
          fixture.meta("flow_shift"),
          fixture.meta("expand_timesteps"),
        ],
        [String(STEPS), "5.0", "5.0", "false"],
        `${name}: step count, guidance, shift, timestep path`,
      );
      assertEquals(fixture.meta("negative"), "negative", `${name}: negative row name`);
      assertEquals(fixture.shape("latents_init"), LATENT_SHAPE, `${name}: latent shape`);
      assertEquals(fixture.shape("frames"), [3, FRAMES, HEIGHT, WIDTH], `${name}: frame shape`);
      // The 48 statistics (direction of the denormalization) — the values the reference wrote from upstream's VAE
      // config are bit-identical to the TS constants.
      assert(
        bitsEqual(fixture.tensor("latents_mean"), mean),
        `${name}: latents_mean differs from WAN22`,
      );
      assert(
        bitsEqual(fixture.tensor("latents_std"), std),
        `${name}: latents_std differs from WAN22`,
      );
      const timesteps: unknown = JSON.parse(fixture.meta("timesteps"));
      const sigmas: unknown = JSON.parse(fixture.meta("sigmas"));
      assertEquals(timesteps, schedule.timesteps, `${name}: timesteps`);
      assert(
        Array.isArray(sigmas) && sigmas.every((value) => typeof value === "number"),
        `${name}: sigmas is not an array of numbers`,
      );
      assert(
        bitsEqual(Float32Array.from(sigmas), schedule.sigmas),
        `${name}: sigmas ${JSON.stringify(sigmas)} do not match ${[...schedule.sigmas]} in f32`,
      );
      // Rebuild the reference latents with the host's CFG + UniPC from the DiT outputs the reference recorded (forward
      // hook).
      const sampler = new WanUniPcSampler(schedule, WAN_UNIPC_CONFIG);
      let latents: Float32Array<ArrayBuffer> = fixture.tensor("latents_init");
      for (let step = 0; step < STEPS; step += 1) {
        assertEquals(
          fixture.shape(`latents.${step}`),
          LATENT_SHAPE,
          `${name}: latents.${step} shape`,
        );
        const velocity = wanClassifierFreeGuidance(
          fixture.tensor(`noise_cond.${step}`),
          fixture.tensor(`noise_uncond.${step}`),
          GUIDANCE,
        );
        latents = sampler.step(velocity, latents);
        const want = fixture.tensor(`latents.${step}`);
        assert(
          bitsEqual(latents, want),
          `${name}: step ${step} latents are not bit-identical (${formatDifference(
            "latents",
            difference(latents, want),
          )}）`,
        );
      }
    }
  },
);

/** The observation points (outputs with a band — the latents after each step and the clamped frames). */
const OBSERVATION_POINTS = ["latents.0", "latents.1", "frames"] as const;
type ObservationPoint = (typeof OBSERVATION_POINTS)[number];

/** The band per observation point (`undefined` when not derived). */
const bandOf = (point: ObservationPoint): number | undefined =>
  point === "frames" ? FRAME_RATIO_BAND : LATENT_RATIO_BANDS[point === "latents.0" ? 0 : 1];

const BANDS_DERIVED = OBSERVATION_POINTS.every((point) => bandOf(point) !== undefined);

/**
 * One case's ratios per observation point (the material for the band candidate). Stored after the non-finite and
 * device-lost checks pass and before the band verdict (a run outside the band is kept, and ratios computed by skipping
 * non-finite elements or after a device lost are not mixed into the candidate).
 */
type CaseRatios = {
  readonly name: string;
  readonly role: "band" | "accept";
  readonly ratios: Readonly<Record<ObservationPoint, number>>;
};

/**
 * The band candidate (worst × 5 rounded up to 2 significant digits — the same rule as stage 2's `bandCandidate`). So
 * that a division error (e.g. 1.1e-6 / 1e-7 slightly exceeding 11) does not round an exactly-2-digit value up by one
 * unit, the quotient is rounded to 12 digits before rounding up.
 */
const roundUpTwoDigits = (value: number): number => {
  const unit = 10 ** (Math.floor(Math.log10(value)) - 1);
  return Number((Math.ceil(Number((value / unit).toPrecision(12))) * unit).toPrecision(2));
};

describe.skipIf(!DIST_PRESENT || !GPU_AVAILABLE)(
  "Wan2.2 end-to-end 2 steps (real GPU): 1280×704, 17 frames, fault injection outside the band, " +
    "sha256 per-environment rows (reference preset's band cases, seed path in 2 sizes, practical presets' seed path, GPU path) — the latents and " +
    "frames inside the band of the diffusers reference only with KARUME_UPSTREAM_PARITY=1",
  () => {
    let deviceLost: string | undefined;
    // One device shared by the its below (acquired before them, destroyed after them all), and the
    // reference-seat pipeline the band, fault and seed its run on.
    let gpuContext: GpuContext | undefined;
    let referencePipeline: Ti2vPipeline | undefined;
    const diagnostics = new Map<RunComponent, SessionDiagnostics>();
    /**
     * The reference preset's seed-path 1280×704 artifact (RGB bytes) — the partner of the floor of the practical
     * presets' cases for the same request (`beforeSettle` of {@link runSeedCase}).
     */
    let referenceSeedRgb: Uint8Array<ArrayBuffer> | undefined;
    /** The practical seats' outputs seen so far (the online seat must also differ from the s16 seat). */
    const practicalSeedRgb: Uint8Array<ArrayBuffer>[] = [];
    /**
     * One seed-path two-step case (completion, 0 non-finite values, the sha256 per-environment row). The preset is the
     * one `seatPipeline` was built with; the reference preset's two sizes and the practical presets' 1280×704 go
     * through the same body (so the checks do not diverge per preset). `beforeSettle` receives the artifact before the
     * sha comparison and the row write (`settleOrObserve`) (the place for checks that must apply before a row is
     * written, even in a `KARUME_REFERENCE=write` run).
     */
    /** The conv8 cases' VAE stage by tile (filled in run order — tile 16 first). */
    const conv8Vae = new Map<16 | 32, Conv8VaeRecord>();
    const runSeedCase = async (
      seatPipeline: Ti2vPipeline,
      id: string,
      width: number,
      height: number,
      beforeSettle: (bytes: Uint8Array<ArrayBuffer>) => void,
      knobs: {
        readonly vaeTile?: 32;
        /** Records the VAE stage under this tile in {@link conv8Vae} (with nvidia-smi samples). */
        readonly conv8Record?: 16 | 32;
      } = {},
    ): Promise<void> => {
      const gpu = assigned(gpuContext, "gpuContext");
      let settlement: ReferenceSettlement | undefined;
      try {
        await runRecordedCase(results, { id }, async () => {
          const observed = await observe(
            seatPipeline,
            diagnostics,
            {
              ...TWO_STEP,
              prompt: textOf(seatPipeline.prompts, SEED_PROMPT, "prompt"),
              seed: SEED,
              width,
              height,
              ...(knobs.vaeTile === undefined ? {} : { vaeTile: knobs.vaeTile }),
            },
            { nvidiaSmi: knobs.conv8Record !== undefined },
          );
          assert("video" in observed, "generate did not run to the end");
          assertEquals(observed.latents.length, STEPS, "denoise-step count");
          const { video } = observed;
          assertEquals([video.frames, video.height, video.width], [FRAMES, height, width]);
          const nonFinite = countNonFinite(video.data);
          const notes = [`non-finite ${nonFinite}`, ...formatObserved(observed)];
          if (knobs.conv8Record !== undefined) {
            const vaeMs = observed.stageMs.vae_decoder;
            assert(vaeMs !== undefined, `${id}: the vae_decoder stage was not timed`);
            const peak = (phase: string): number | undefined => {
              const sampled = observed.nvidia?.phases.get(phase);
              return sampled === undefined || sampled.samples === 0
                ? undefined
                : sampled.peakMemoryMiB;
            };
            conv8Vae.set(knobs.conv8Record, {
              vaeSeconds: vaeMs / 1000,
              vaePeakMiB: peak("vae_decoder"),
              baselineMiB: peak("start"),
            });
            notes.push(...conv8VaeLines(conv8Vae));
          }
          console.log(`[wan-ti2v-pipeline] ${id}:\n  ${notes.join("\n  ")}`);
          assertEquals(nonFinite, 0, `${id}: non-finite values`);
          assertEquals(deviceLost, undefined, "device lost");
          const bytes = rgbBytes(video);
          beforeSettle(bytes);
          const outcome = await settleOrObserve(references, results, {
            id,
            artifact: `${id}.rgb`,
            bytes,
          });
          settlement = outcome.settlement;
          return { ...outcome.fields, note: notes.join(" / ") };
        });
      } finally {
        await settleReleases(gpu);
      }
      if (settlement?.check.status === "fail") {
        throw new Error(referenceMismatchMessage(id, settlement, references));
      }
    };
    beforeAll(async () => {
      gpuContext = verifyRunningGpu(
        await acquireTestGpu({
          onDeviceLost: (info) => {
            deviceLost = `${info.reason}: ${info.message}`;
          },
        }),
      );
      referencePipeline = await loadPipeline(
        gpuContext,
        diagnostics,
        "precomputed",
        WAN_TI2V_REFERENCE_QUANT,
      );
    });
    afterAll(async () => {
      if (gpuContext === undefined) return;
      try {
        await referencePipeline?.dispose();
      } finally {
        // MUST: wait for the releases before discarding the device (B570's delayed destroy — leave budget for the
        // following tests).
        await settleReleases(gpuContext);
        gpuContext.destroy();
      }
    });
    const caseRatios: CaseRatios[] = [];

    for (const { name, role } of CASES) {
      const id = bandCaseId(name);
      it.skipIf(!ANY_FIXTURE)(`${id}（${role}）`, async () => {
        const gpu = assigned(gpuContext, "gpuContext");
        const pipeline = assigned(referencePipeline, "referencePipeline");
        const { prompts } = pipeline;
        const fixture = await readFixture(fixtureUrl(name));
        assertEquals(fixture.meta("role"), role, `${name}: role`);
        // band looks up the original text, accept the normalized string (both spellings of the accepted set are used).
        const prompt = textOf(
          prompts,
          fixture.meta("prompt"),
          role === "band" ? "prompt" : "normalized",
        );
        let settlement: ReferenceSettlement | undefined;
        let caseNote = "";
        try {
          await runRecordedCase(
            results,
            { id, failureNote: () => caseNote },
            async ({ measurements }) => {
              const observed = await observe(pipeline, diagnostics, {
                ...TWO_STEP,
                prompt,
                latents: fixture.tensor("latents_init"),
                width: WIDTH,
                height: HEIGHT,
              });
              assert("video" in observed, "generate did not run to the end");
              assertEquals(observed.latents.length, STEPS, "denoise-step count");
              const { video } = observed;
              assertEquals([video.frames, video.height, video.width], [FRAMES, HEIGHT, WIDTH]);
              const diffs = OBSERVATION_POINTS.map((point) => ({
                point,
                band: bandOf(point),
                // The reference frames are written after the clamp (the clamp of `ti2v_few_step_ref.decode_frames`).
                diff:
                  point === "frames"
                    ? difference(video.data, fixture.tensor("frames"))
                    : difference(
                        observed.latents[point === "latents.0" ? 0 : 1],
                        fixture.tensor(point),
                      ),
              }));
              // A measurement records the band it was judged against, so it exists only when the band is
              // judged (upstream parity is opt-in — module doc).
              if (UPSTREAM_PARITY) {
                for (const { point, band, diff } of diffs) {
                  measurements.push({
                    output: point,
                    maxAbs: diff.maxAbs,
                    maxRel: diff.maxRel,
                    // An underived band is recorded as an infinite band (the verdict below fails — the same as
                    // stage 2's `toleranceOf`).
                    tolerance:
                      band === undefined
                        ? { atol: Number.POSITIVE_INFINITY, rtol: 0 }
                        : { atol: band * diff.referenceMax, rtol: 0 },
                    stage: "karume",
                  });
                }
              }
              const notes = [
                ...diffs.map(({ point, diff }) => formatDifference(point, diff)),
                ...(UPSTREAM_PARITY
                  ? []
                  : ["band not judged (upstream parity is opt-in: KARUME_UPSTREAM_PARITY=1)"]),
                ...formatObserved(observed),
              ];
              caseNote = notes.join(" / ");
              console.log(`[wan-ti2v-pipeline] ${id}:\n  ${notes.join("\n  ")}`);
              for (const { point, diff } of diffs) {
                assertEquals(diff.nonFinite, 0, `${id}: ${point} non-finite values`);
              }
              assertEquals(deviceLost, undefined, "device lost");
              caseRatios.push({
                name,
                role,
                ratios: {
                  "latents.0": diffs[0].diff.ratio,
                  "latents.1": diffs[1].diff.ratio,
                  frames: diffs[2].diff.ratio,
                },
              });
              // MUST stay before the row is settled: with the opt-in, an output outside the band is never
              // written as a row under KARUME_REFERENCE=write.
              if (UPSTREAM_PARITY) {
                for (const { point, band, diff } of diffs) {
                  if (band === undefined) {
                    throw new Error(
                      `${id}: ${point} band not derived yet (ratio ${diff.ratio.toExponential(3)}) — ` +
                        'the "band candidate" step prints the candidate',
                    );
                  }
                  assert(
                    diff.ratio <= band,
                    `${id}: ${formatDifference(point, diff)} is outside the band ${band}`,
                  );
                }
              }
              const outcome = await settleOrObserve(references, results, {
                id,
                artifact: `${id}.rgb`,
                bytes: rgbBytes(video),
              });
              settlement = outcome.settlement;
              return { ...outcome.fields, note: caseNote };
            },
          );
        } finally {
          // Before the next case allocates, wait for the release of this case's stage allocations (B570's delayed
          // `destroy()`).
          await settleReleases(gpu);
        }
        if (settlement?.check.status === "fail") {
          throw new Error(referenceMismatchMessage(id, settlement, references));
        }
      });
    }

    // The candidate for an underived band (built from the deciding cases only — the acceptance ratios are listed but
    // not used; when one of the deciding cases failed before reporting its ratios, no candidate is printed). When the
    // bands are derived, no step is created.
    if (!BANDS_DERIVED) {
      it.skipIf(!ANY_FIXTURE || !UPSTREAM_PARITY)(
        `band candidate (not derived yet)${UPSTREAM_PARITY_OPT_IN}`,
        () => {
          const expected = CASES.filter(({ role }) => role === "band").length;
          const decided = caseRatios.filter(({ role }) => role === "band");
          const candidate =
            decided.length === expected
              ? OBSERVATION_POINTS.map((point) => {
                  const worst = Math.max(...decided.map(({ ratios }) => ratios[point]));
                  return `${point}: worst of the ${expected} deciding cases ${worst.toExponential(3)} × 5 = ${(
                    worst * 5
                  ).toExponential(
                    3,
                  )} (rounded up to 2 significant digits: ${roundUpTwoDigits(worst * 5)})`;
                }).join(" / ")
              : `of the ${expected} deciding cases, only ${decided.length} reported ratios — no candidate (investigate the failed cases first)`;
          const accepted = caseRatios
            .filter(({ role }) => role === "accept")
            .map(
              ({ name, ratios }) =>
                `${name} ${OBSERVATION_POINTS.map(
                  (point) => `${point} ${ratios[point].toExponential(3)}`,
                ).join(", ")}`,
            );
          throw new Error(
            `band not derived yet — ${candidate}. Acceptance ratios (not used to set the band): ${
              accepted.join(" / ") || "none"
            }`,
          );
        },
      );
    }

    // Fault injection (the acceptance case's initial noise — only the latents are checked, so it stops before the VAE
    // stage).
    let acceptFixture: Fixture | undefined;
    const faults: readonly {
      readonly label: string;
      readonly request: (
        ferret: string,
        negative: string,
      ) => Omit<GenerateRequest, "onEvent" | "latents">;
    }[] = [
      {
        label: "swapped positive / negative prompts",
        request: (ferret, negative) => ({
          ...TWO_STEP,
          prompt: negative,
          negativePrompt: ferret,
        }),
      },
      {
        label: "guidance 5.0 → 5.05",
        request: (ferret) => ({ ...TWO_STEP, prompt: ferret, guidance: 5.05 }),
      },
      {
        label: "shift 5.0 → 3.0",
        request: (ferret) => ({ ...TWO_STEP, prompt: ferret, shift: 3 }),
      },
    ];
    const lastBand = LATENT_RATIO_BANDS[STEPS - 1];
    for (const { label, request } of faults) {
      it.skipIf(!ANY_FIXTURE)(
        `fault injection: ${label} → ${ACCEPT_CASE} latents outside the band`,
        async () => {
          const gpu = assigned(gpuContext, "gpuContext");
          const pipeline = assigned(referencePipeline, "referencePipeline");
          const { prompts } = pipeline;
          try {
            acceptFixture ??= await readFixture(fixtureUrl(ACCEPT_CASE));
            const accept = acceptFixture;
            const ferret = textOf(prompts, accept.meta("prompt"), "prompt");
            const negative = textOf(prompts, accept.meta("negative"), "prompt");
            const observed = await observe(
              pipeline,
              diagnostics,
              {
                ...request(ferret, negative),
                latents: accept.tensor("latents_init"),
                width: WIDTH,
                height: HEIGHT,
              },
              { abortAfterDenoise: true },
            );
            // If a stage event name changes and the abort stops working, the 28-tile VAE is paid silently (it does not
            // show in the value verdict).
            assert(!("video" in observed), "did not stop after the DiT stage");
            assertEquals(observed.latents.length, STEPS, "denoise-step count");
            const diff = difference(
              observed.latents[STEPS - 1],
              accept.tensor(`latents.${STEPS - 1}`),
            );
            console.log(
              `[wan-ti2v-pipeline] fault ${label}: ${formatDifference(
                `latents.${STEPS - 1}`,
                diff,
              )}（${
                lastBand === undefined
                  ? "band not derived yet"
                  : `${(diff.ratio / lastBand).toFixed(1)} × the band`
              }）`,
            );
            if (lastBand === undefined) {
              throw new Error(
                `${label}: band not derived yet (fault-injection ratio ${diff.ratio.toExponential(3)})`,
              );
            }
            assert(
              diff.ratio > FAULT_MARGIN * lastBand,
              `${label}: ratio ${diff.ratio} does not exceed the band ${lastBand} × ${FAULT_MARGIN}`,
            );
          } finally {
            await settleReleases(gpu);
          }
        },
      );
    }

    // The seed path's two steps (2 sizes — completion, 0 non-finite values, the sha256 per-environment row).
    for (const { id, width, height } of SEED_CASES) {
      it(`${id} (seed path, sha256 per-environment row)`, () =>
        runSeedCase(
          assigned(referencePipeline, "referencePipeline"),
          id,
          width,
          height,
          (bytes) => {
            // Keep as the floor's partner only the artifact of the practical presets' request (1280×704).
            if (width === WIDTH && height === HEIGHT) referenceSeedRgb = bytes;
          },
        ));
    }

    // The practical presets' seed path (the reference preset's pipeline is disposed before this preset's is built —
    // building creates no Session).
    for (const { id: seedCaseId, quant, vaeTile } of PRACTICAL_SEED_CASES) {
      it(`${seedCaseId} (practical preset, seed path, sha256 per-environment row)`, async () => {
        // The reference-seat its above are done: dispose that pipeline before building this seat's
        // (afterAll disposes it again, idempotently, for runs where this it does not run).
        await referencePipeline?.dispose();
        await using practical = await loadPipeline(
          assigned(gpuContext, "gpuContext"),
          diagnostics,
          "precomputed",
          quant,
        );
        await runSeedCase(
          practical,
          seedCaseId,
          WIDTH,
          HEIGHT,
          (bytes) => {
            // The floor (ADR 0110 decision 5 — a practical preset differs by at least 1 bit from the reference preset
            // for the same request): in a regression where the preset's `session` does not reach the Session on the
            // distribution path, the practical preset produces the same bytes as the reference preset. Writing that
            // artifact as a practical row would freeze the reference row's sha, and from then on this row would pin
            // neither practical-tier regressions nor determinism. Fail before writing the row (before
            // `settleOrObserve`).
            if (referenceSeedRgb === undefined) {
              throw new Error(
                `${seedCaseId}: no reference-preset artifact for the same request (the reference preset's seed-path 1280×704 did not run — the floor cannot be checked)`,
              );
            }
            const reference = referenceSeedRgb;
            const differs = (other: Uint8Array<ArrayBuffer>): boolean =>
              bytes.length !== other.length || bytes.some((value, index) => value !== other[index]);
            assert(
              differs(reference),
              `${seedCaseId}: the practical preset's output is byte-identical to the reference preset's for the same request (the preset's session did not arrive)`,
            );
            // The practical seats differ from each other too (the online seat replaces the three
            // dispatches; a route that silently kept them would reproduce the s16 bytes).
            // The vae32 case differs from every earlier case too: the conv8 tile-16 case among them, so a tile
            // knob that does not reach the decoder fails here.
            for (const earlier of practicalSeedRgb) {
              assert(differs(earlier), `${seedCaseId}: the output equals another practical seat's`);
            }
            practicalSeedRgb.push(bytes);
          },
          {
            ...(vaeTile === undefined ? {} : { vaeTile }),
            ...(quant === WAN_TI2V_CONV8_QUANT ? { conv8Record: vaeTile ?? 16 } : {}),
          },
        );
      });
    }

    // The GPU path (the umT5 text stage is disposed before the DiT and VAE stages are built).
    it.skipIf(!GPU_TEXT_PRESENT)(
      `${GPU_TEXT_CASE_ID} (GPU path, sha256 per-environment row)`,
      async () => {
        const gpu = assigned(gpuContext, "gpuContext");
        let settlement: ReferenceSettlement | undefined;
        try {
          // Building creates no Session (umT5's weights are read by the text stage on every generate — ADR 0119
          // decision 11).
          await using gpuPipeline = await loadPipeline(
            gpu,
            diagnostics,
            "gpu",
            WAN_TI2V_REFERENCE_QUANT,
          );
          const prompt = textOf(gpuPipeline.prompts, SEED_PROMPT, "prompt");
          await runRecordedCase(results, { id: GPU_TEXT_CASE_ID }, async () => {
            const observed = await observe(gpuPipeline, diagnostics, {
              ...TWO_STEP,
              prompt,
              seed: SEED,
              width: WIDTH,
              height: HEIGHT,
            });
            assert("video" in observed, "generate did not run to the end");
            assertEquals(observed.latents.length, STEPS, "denoise-step count");
            // The text stage is disposed before the DiT stage is built (the order of the stage boundaries — the VRAM
            // points are taken in the same order).
            assertEquals(
              observed.drm.marks.slice(0, 3).map(({ label }) => label),
              ["text_encoder start", "text_encoder end", "transformer start"],
              "text stage → DiT stage order",
            );
            assert(
              observed.diagnostics.has("text_encoder"),
              "the diagnostics of the umT5 run did not arrive",
            );
            const { video } = observed;
            assertEquals([video.frames, video.height, video.width], [FRAMES, HEIGHT, WIDTH]);
            const nonFinite = countNonFinite(video.data);
            const notes = [
              "textEncoder: gpu (umT5 i8, f32 activations)",
              `non-finite ${nonFinite}`,
              ...formatObserved(observed),
            ];
            console.log(`[wan-ti2v-pipeline] ${GPU_TEXT_CASE_ID}:\n  ${notes.join("\n  ")}`);
            assertEquals(nonFinite, 0, `${GPU_TEXT_CASE_ID}: non-finite values`);
            assertEquals(deviceLost, undefined, "device lost");
            const outcome = await settleOrObserve(references, results, {
              id: GPU_TEXT_CASE_ID,
              artifact: `${GPU_TEXT_CASE_ID}.rgb`,
              bytes: rgbBytes(video),
            });
            settlement = outcome.settlement;
            return { ...outcome.fields, note: notes.join(" / ") };
          });
        } finally {
          await settleReleases(gpu);
        }
        if (settlement?.check.status === "fail") {
          throw new Error(referenceMismatchMessage(GPU_TEXT_CASE_ID, settlement, references));
        }
      },
    );
  },
);

// NOTE: run with the opt-in env and without `-t`, one process takes two devices in turn together with the default
// lane's `describe` (hitting B570's `destroy()` not returning VRAM). Run it as a separate process with
// `-t "50 steps"` (module doc "Device use").
describe.skipIf(!FULL_PIPELINE || !DIST_PRESENT || !GPU_AVAILABLE)(
  "Wan2.2 end-to-end 50 steps (real GPU, opt-in KARUME_WAN_TI2V_FULL_PIPELINE=1): 1280×704, 33 frames (reference preset, practical s16, online)" +
    " complete with 0 non-finite values; time and VRAM at stage switches, PNG of every frame + contact sheet, sha256 per-environment rows",
  () => {
    let deviceLost: string | undefined;
    // One device shared by the its below (acquired before them, destroyed after them all).
    let gpuContext: GpuContext | undefined;
    const diagnostics = new Map<RunComponent, SessionDiagnostics>();
    beforeAll(async () => {
      gpuContext = verifyRunningGpu(
        await acquireTestGpu({
          onDeviceLost: (info) => {
            deviceLost = `${info.reason}: ${info.message}`;
          },
        }),
      );
    });
    afterAll(async () => {
      if (gpuContext === undefined) return;
      await settleReleases(gpuContext);
      gpuContext.destroy();
    });
    /**
     * The sha256 of the reference preset's artifact (RGB bytes) — the partner of the practical presets' floor for the
     * same request (the same floor as the default lane's practical rows).
     */
    let referenceRgbSha: string | undefined;
    for (const { id, quant } of FULL_CASES) {
      it(id, async () => {
        const gpu = assigned(gpuContext, "gpuContext");
        let settlement: ReferenceSettlement | undefined;
        try {
          await using pipeline = await loadPipeline(gpu, diagnostics, "precomputed", quant);
          await runRecordedCase(fullResults, { id }, async () => {
            const observed = await observe(pipeline, diagnostics, {
              prompt: textOf(pipeline.prompts, SEED_PROMPT, "prompt"),
              seed: SEED,
              shift: SHIFT,
              width: WIDTH,
              height: HEIGHT,
              frames: FULL_FRAMES,
            });
            assert("video" in observed, "generate did not run to the end");
            assertEquals(
              observed.latents.length,
              FULL_STEPS,
              "denoise-step count (the manifest's default)",
            );
            const { video } = observed;
            assertEquals([video.frames, video.height, video.width], [FULL_FRAMES, HEIGHT, WIDTH]);
            const nonFinite = countNonFinite(video.data);
            const notes = [
              `preset ${quant}`,
              `non-finite ${nonFinite}`,
              ...formatObserved(observed),
            ];
            console.log(`[wan-ti2v-pipeline] ${id}:\n  ${notes.join("\n  ")}`);
            assertEquals(nonFinite, 0, "non-finite values");
            assertEquals(deviceLost, undefined, "device lost");
            await writeWanFrames(fullResults, id, video);
            const sheet = contactSheet(video, 4, 8, 4);
            await writeFile(
              fullResults.artifact(`${id}-sheet.png`),
              await encodePng(sheet.rgba, sheet.width, sheet.height),
            );
            console.log(`[wan-ti2v-pipeline] PNG: ${fullResults.dir.pathname}`);
            const bytes = rgbBytes(video);
            const sha = await sha256Hex(bytes);
            if (quant === WAN_TI2V_REFERENCE_QUANT) {
              referenceRgbSha = sha;
            } else {
              // The floor (ADR 0110 decision 5 — a practical preset differs by at least 1 bit from the reference preset
              // for the same request): in a regression where the preset's `session` does not reach the Session, the
              // practical preset produces the same bytes as the reference preset. Fail before the comparison and the
              // write, so that artifact is not frozen as a practical row.
              if (referenceRgbSha === undefined) {
                throw new Error(
                  `${id}: no reference-preset artifact for the same request (the reference preset's 50 steps did not run — the floor cannot be checked)`,
                );
              }
              assert(
                sha !== referenceRgbSha,
                `${id}: the practical preset's output is byte-identical to the reference preset's for the same request (the preset's session did not arrive)`,
              );
            }
            const outcome = await settleOrObserve(references, fullResults, {
              id,
              artifact: `${id}.rgb`,
              bytes,
            });
            settlement = outcome.settlement;
            return { ...outcome.fields, note: notes.join(" / ") };
          });
        } finally {
          // Before the next preset allocates, wait for the release of this preset's stage allocations (B570's delayed
          // `destroy()`).
          await settleReleases(gpu);
        }
        if (settlement?.check.status === "fail") {
          throw new Error(referenceMismatchMessage(id, settlement, references));
        }
      });
    }
  },
);

// NOTE: like the 50-step run, run it as a separate process with `-t "121 frames"` (module doc "Device use").
test.skipIf(!LONG_CLIP || !DIST_PRESENT || !GPU_AVAILABLE)(
  "Wan2.2 end-to-end 121 frames (real GPU, opt-in KARUME_WAN_TI2V_121F=1): 1280×704, 2 steps (reference preset, seed path) complete" +
    " with 0 non-finite values; sha256 per-environment row, time and VRAM at stage switches",
  async () => {
    let deviceLost: string | undefined;
    const gpu = verifyRunningGpu(
      await acquireTestGpu({
        onDeviceLost: (info) => {
          deviceLost = `${info.reason}: ${info.message}`;
        },
      }),
    );
    const diagnostics = new Map<RunComponent, SessionDiagnostics>();
    let settlement: ReferenceSettlement | undefined;
    try {
      await using pipeline = await loadPipeline(
        gpu,
        diagnostics,
        "precomputed",
        WAN_TI2V_REFERENCE_QUANT,
      );
      await runRecordedCase(longClipResults, { id: LONG_CLIP_CASE_ID }, async () => {
        const observed = await observe(pipeline, diagnostics, {
          ...TWO_STEP,
          frames: LONG_CLIP_FRAMES,
          prompt: textOf(pipeline.prompts, SEED_PROMPT, "prompt"),
          seed: SEED,
          width: WIDTH,
          height: HEIGHT,
        });
        assert("video" in observed, "generate did not run to the end");
        assertEquals(observed.latents.length, STEPS, "denoise-step count");
        const { video } = observed;
        assertEquals([video.frames, video.height, video.width], [LONG_CLIP_FRAMES, HEIGHT, WIDTH]);
        const nonFinite = countNonFinite(video.data);
        const notes = [`non-finite ${nonFinite}`, ...formatObserved(observed)];
        console.log(`[wan-ti2v-pipeline] ${LONG_CLIP_CASE_ID}:\n  ${notes.join("\n  ")}`);
        assertEquals(nonFinite, 0, `${LONG_CLIP_CASE_ID}: non-finite values`);
        assertEquals(deviceLost, undefined, "device lost");
        const outcome = await settleOrObserve(references, longClipResults, {
          id: LONG_CLIP_CASE_ID,
          artifact: `${LONG_CLIP_CASE_ID}.rgb`,
          bytes: rgbBytes(video),
        });
        settlement = outcome.settlement;
        return { ...outcome.fields, note: notes.join(" / ") };
      });
    } finally {
      await settleReleases(gpu);
      gpu.destroy();
    }
    if (settlement?.check.status === "fail") {
      throw new Error(referenceMismatchMessage(LONG_CLIP_CASE_ID, settlement, references));
    }
  },
);

/**
 * The cases registered with the reference guard (only those that can run — the band cases need the references and the
 * distribution, the seed cases the distribution, and the GPU path also the umT5 mirror). The 50-step run and the
 * 121-frame case register only under the opt-in (they do not run in the default lane, so registering them would turn
 * the reference guard red on a machine without the rows).
 */
const CASE_IDS = [
  ...(DIST_PRESENT && ANY_FIXTURE ? CASES.map(({ name }) => bandCaseId(name)) : []),
  ...(DIST_PRESENT
    ? [...SEED_CASES.map(({ id }) => id), ...PRACTICAL_SEED_CASES.map(({ id }) => id)]
    : []),
  ...(GPU_TEXT_PRESENT ? [GPU_TEXT_CASE_ID] : []),
  ...(DIST_PRESENT && FULL_PIPELINE ? FULL_CASES.map(({ id }) => id) : []),
  ...(DIST_PRESENT && LONG_CLIP ? [LONG_CLIP_CASE_ID] : []),
];
const RUNNABLE = DIST_PRESENT && GPU_AVAILABLE;
if (RUNNABLE) references.warnMissing(CASE_IDS);
registerReferenceGate(references, { runnable: RUNNABLE, caseIds: CASE_IDS });
