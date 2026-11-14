/**
 * End-to-end comparison of Wan2.2 TI2V-5B's I2V (`Ti2vPipeline` given a condition image — ADR 0121 decision 11, stage
 * 9b) (real GPU).
 *
 * The pipeline is built from the distribution mirror `models/karume-wan2.2/` (with the encoder's three components —
 * assembled by `dist.py --pipeline wan-ti2v`) through `nodeDirectory` and runs generation from a condition image. The
 * references (`pipeline_steps_i2v.*`) are goldens no distribution carries, read from the test-asset root (ADR 0139
 * decision 2). A machine without the mirror SKIPs explicitly (with the reason and the assembly command — an all-SKIP is
 * a FAIL of the gate `distribution_gate_test.ts`).
 *
 * The comparison target is the recipe's two-step reference (`tools/export-recipes/wan/ti2v_i2v_few_step_ref.py` —
 * `goldens/wan2.2-ti2v-5b-i8-dyn/pipeline_steps_i2v.<case>.safetensors` of the test-asset root): the latents of the
 * `expand_timesteps` branch of diffusers' `WanImageToVideoPipeline` run for two steps in CPU f32 (with CFG, guidance
 * 5.0, shift 5.0, the timestep MLP with M = 1 per value — decision 4), and the frames from passing them through stage
 * 5's tiled decode → unpatchify → clamp (the weights hold the same values as the reference preset `f16+dit8`). The
 * shape is 1280×704 with 9 frames (latents `[48,3,44,80]`, S = 2,640, 880 condition tokens), the same for all three
 * cases. The initial noise is the reference's `latents_init`, injected through `latents`, and the condition image is
 * the RGB8 of the source image embedded in the reference (`source` 832×480 — relying neither on PNG decoding nor on
 * another asset), passed as `image` (the product preprocessing chooses the size and crops).
 *
 * ## Checks (default lane)
 *
 * - **Host self-consistency** (no GPU — all three cases): the reference metadata (σ / timestep, shift, timestep path,
 *   condition token count, image provenance, embedding asset) and the 48 statistics match TS, and passing the embedded
 *   `source` through the TS crop (`fitWanI2vImage`) gives the reference's `rgb8` bit for bit. From the DiT outputs the
 *   reference recorded, the host's CFG + UniPC (without the replacement — the same as diffusers' callback) rebuilds the
 *   reference's `latents.<i>` bit for bit, and the product's replacement function (`withWanFirstFrameCondition`)
 *   rebuilds the reference's `latents_final` bit for bit from `latents.1` and the condition latents.
 * - **Band (opt-in upstream parity)** (ratio = max absolute difference ÷ the reference's max absolute value — the
 *   same metric as T2V): the observation points are the scheduler latents after each step (`latents.0` / `latents.1`
 *   — the `denoise-step` event's `copyLatents`, before the replacement), the last latents (`latents_final` — after the
 *   replacement, the value handed to the VAE; "Observing the last latents" below) and the clamped frames. The band is
 *   the worst of the two deciding cases (`band-boxing-cats` / `band-cat-dog-baking`) per observation point × 5 (rounded
 *   up to two
 *   significant digits), and the acceptance case (`accept-ferret`) must fall inside it. While a band is underived
 *   (`undefined`), each case records its ratios and fails, and the "band candidate" case prints the deciding worst × 5
 *   (the second deciding case is behind `KARUME_WAN_TI2V_I2V_FULL=1`, so the candidate needs that opt-in as well). The
 *   judgement and the candidate run only with `KARUME_UPSTREAM_PARITY=1` ("Default run vs opt-in upstream parity"
 *   below); the cases themselves run for their wiring checks and sha256 rows. MUST: do not change the band or the cases
 *   after seeing the acceptance result. If the acceptance case leaves the band, investigate the cause instead of
 *   widening the band.
 * - **DiT input wiring** (per band case — the run's own inputs): the DiT runs are the 2 CFG passes × 2 steps. In every
 *   run the condition mask is true only for the first 880 tokens, the conditioning timestep is bit-identical to the
 *   proj at t = 0 (`timestepsProj(0)`), the first latent frame of the model input has the same value in every run (=
 *   the condition latents), and the remaining frames match that step's scheduler state (step 0 = the injected initial
 *   noise, step 1 = `latents.0`). The stage order is encoder → DiT (the precomputed path has no text stage), and the
 *   diagnostics of the encoder's three graphs arrive.
 * - **Condition latents (opt-in upstream parity, per band case)**: the max absolute difference between the first frame
 *   the DiT received (the exit of the encoder stage) and the reference's `condition_latents` is inside the band of the
 *   encoder's GPU check (`WAN_I2V_ENCODER_BAND` — the encoder check keeps the deciding / acceptance split), so an encoder
 *   regression is named at the encoder stage instead of being left to the latent band. It compares against diffusers
 *   under a tolerance, so it runs only with `KARUME_UPSTREAM_PARITY=1` like the encoder check itself; the non-finite
 *   count of the condition latents is checked by default.
 * - **VAE input wiring** (per band case): "Observing the last latents" below.
 * - **Fault injection** (the initial noise of the default deciding case `band-boxing-cats`, stopped at the DiT stage
 *   and compared at `latents.1`, floor = {@link FAULT_MARGIN} × the band): the condition mask all false, the condition
 *   mask shifted by one token, the two timestep inputs swapped, the first frame of the model input not replaced, the
 *   condition latents not normalized. They go in through the shape of "Hooks for observation and fault injection"
 *   below, without adding a hook to the product code. Besides the band, each injection also checks that the named check
 *   (the DiT input wiring or the condition-latent check) turns red on the same run's record. That the VAE input wiring
 *   check turns red is checked on the band case's record by replacing the expectation with the faulty value (0 GPU
 *   time).
 * - **sha256 per-environment rows** (ADR 0106 — `fixtures/references/wan-ti2v.json`, shared with T2V under different
 *   case IDs): the output frames (the uint8 RGB of all frames concatenated — the rule of `frameToRgba`). The default
 *   lane has 2 rows = the reference preset's default case ({@link DEFAULT_CASE}) and the practical preset's same
 *   request ({@link PRACTICAL_CASE_ID}). The practical preset's row is a **practical row** (ADR 0110 decision 7 —
 *   `rewrite` in the same commit when a commit changes practical-tier numerics on purpose), checked before the
 *   comparison and the write to differ by at least 1 bit from the reference preset's artifact for the same request
 *   (the floor — ADR 0110 decision 5). The reference preset's row is a **reference row** (frozen). An ID carries the
 *   preset, I2V and fit, step count, case, size, frame count and shift ({@link caseIdOf} — ruling F12). A machine
 *   without the row skips that case's sha comparison (the artifact and the measured sha are kept), and the reference
 *   guard turns red. Rows are written after the band is green by running `KARUME_REFERENCE=write` on this file alone.
 * - **The preset is explicit in every case**: the manifest's `defaultQuant` is a value that can move with a quality
 *   ruling, and riding on the default preset would run the same ID with another preset's values.
 *
 * ## Default run vs opt-in upstream parity
 *
 * The band judgement (each reference case's latents, last latents and frames inside the band of the diffusers CPU f32
 * reference), the condition-latent check against the reference's `condition_latents` and the band candidate are
 * upstream parity, so they are opt-in behind `KARUME_UPSTREAM_PARITY=1` (`helpers/upstream-parity.ts` — ADR 0129
 * decision 4 and its addendum 2026-10-17 (2)). They are gated at the assertion, not by skipping the cases: the reference
 * cases run as before (the default case by default, the other two with `KARUME_WAN_TI2V_I2V_FULL=1`) because their
 * initial noise and condition image come from the fixtures, their DiT / VAE input wiring is checked on the run's own
 * inputs, and their outputs carry sha256 rows. Without the opt-in they log the ratios but neither assert the bands nor
 * record them as measurements; with the opt-in the bands are asserted before the row is settled, so
 * `KARUME_REFERENCE=write` run together with the opt-in still refuses to write a row for an output outside the band.
 * Everything else runs by default: the GPU-free host self-consistency (bit identity, no tolerance), the wiring checks,
 * the sha256 rows (with the practical seat's floor against the reference seat) and the fault injections. The fault
 * injections read the default case's fixture and use the bands only as the margin a fault must exceed (ratio >
 * {@link FAULT_MARGIN} × band, and the condition-latent check must turn red for the unnormalised condition) — checks on
 * the wiring, not on parity.
 *
 * ## Opt-in (one env, `KARUME_WAN_TI2V_I2V_FULL=1` — not in the default lane)
 *
 * - The remaining two two-step reference cases (`band-cat-dog-baking` — the second deciding case the band derivation
 *   needs, `accept-ferret` — the acceptance case). Both carry reference-preset rows.
 * - Portrait 704×1280 (reference preset, `boxing-cats`'s landscape source image cropped to an explicit portrait size,
 *   seed 42, no CPU reference — completion, 0 non-finite values, the sha256 per-environment row).
 * - The 50-step run (a separate `describe` — the section below; sha256 per-environment rows for the reference and the
 *   practical preset).
 *
 * The opt-in rows register with the reference guard only under the opt-in (they do not run in the default lane, so
 * registering them would turn the reference guard red on a machine without the rows — the same as T2V's 121 frames and
 * 50 steps). There is one opt-in variable for the remaining two-step cases and the 50 steps (T2V has a separate
 * variable `KARUME_WAN_TI2V_FULL_PIPELINE` for its 50 steps — a different shape), so the two 50-step rows are
 * registered even when only the two-step opt-in is run: on a machine without those rows, the load-time warning names
 * the two 50-step IDs, and the reference guard turns red even when the two-step rows are all present (the known
 * behaviour until the 50-step rows are made). The command that runs the two-step opt-in (without putting the 50 steps
 * in the same process):
 *
 * ```
 * KARUME_WAN_TI2V_I2V_FULL=1 vp test packages/e2e/tests/e2e_wan_ti2v_i2v_pipeline_test.ts -t "I2V end-to-end 2 steps"
 * ```
 *
 * ## The 50-step run (the same opt-in, a separate process with `-t "50 steps"`)
 *
 * The precomputed path, `boxing-cats`'s image and prompt, seed 42, shift 5.0, 1280×704 (chosen by the image's aspect
 * ratio), 33 frames and the manifest's default 50 steps run once each with the reference and the practical preset,
 * writing the PNG of every frame, a contact sheet, the RGB artifact, the time per stage and the VRAM at the stage
 * boundaries to `outputs/verify/<environment key>/<date>_wan-ti2v-i2v-pipeline-full/`, and checking 0 non-finite values
 * and no device lost (material for the maintainer's visual check). The difference between the first frame and the
 * condition image is only recorded (not a check). Both kinds of sha256 per-environment rows are kept (reference preset
 * = reference row, practical preset = practical row — the same as T2V's 50 steps. Stage 9b only recorded, and stage 9d
 * gave it rows together with the practical preset's numeric check (the I2V case of the self A/B check —
 * `e2e_wan_ti2v_ab_test.ts`) — ADR 0121 stage 9d). Before the comparison and the write, the practical preset is
 * checked to differ by at least 1 bit from the reference preset's artifact of the same run (the floor — the same as
 * the two-step practical row, ADR 0110 decision 5).
 * The cases register with the reference guard only under the opt-in (the "Opt-in" section above). On a machine without
 * the rows, this test skips that case's sha comparison (keeping the artifact and the measured sha) and ends green; what
 * turns red is the reference-guard test (registered by `registerReferenceGate` — the `-t` of the command below
 * does not run it; run it by its name with the same env, or run without a filter). The rows are written by adding
 * `KARUME_REFERENCE=write` to the same command:
 *
 * ```
 * KARUME_WAN_TI2V_I2V_FULL=1 vp test packages/e2e/tests/e2e_wan_ti2v_i2v_pipeline_test.ts -t "50 steps"
 * ```
 *
 * ## Hooks for observation and fault injection (no hook added to the product code)
 *
 * The DiT inputs (model input, condition mask, conditioning timestep) and the VAE stage's inputs are outside the public
 * surface, so the runs that look at the DiT inputs (the reference preset's band cases and the fault injections) run
 * **one layer inside**: building with `loadWanFromPretrained`, generation with `generateWanVideo` (the very bodies the
 * public class's `fromPretrained` / `generate` put on their serialization chain — the fetch surface, family admission
 * and stage order go through the same single path). Only the `transformer` / `vae_decoder_first` / `vae_decoder_next`
 * components of the state are replaced by copies that wrap the Session in a Proxy ({@link withSessionHook}), which copy
 * the inputs of `run` / `enqueue` (an observing run passes the input objects through as they are — the values are
 * not changed). A fault injection either rewrites the DiT inputs through the same hook (mask, timestep inputs, model
 * input) or sets the statistics of the family values passed to `generateWanVideo` to mean 0 and std 1 (the
 * normalization `(x − mean)·(1/std)` then returns mu as is — a value only the encoder stage reads; it stops before the
 * VAE stage). The practical preset, the portrait case and the 50 steps run through the public surface
 * (`Ti2vPipeline.fromPretrained` + `generate`) (the real-GPU end-to-end run of the public entry that takes a
 * condition image).
 *
 * ### Observing the last latents
 *
 * `copyLatents` is the scheduler state (before the replacement), so the last latents are assembled from two things:
 * the condition latents read from the DiT's step-0 input (the first frame of the model input — the patchify undone with
 * the permutation of the product's `patchifyLatents`) and the observed `latents.1`. That the assembled value, passed
 * through the product's replacement function, is what reached the VAE is pinned by the VAE stage's inputs (the
 * denormalized tiles — 28 in the first chunk, 56 in the following chunks) matching by value an expectation assembled
 * without the replacement function — the first frame from the condition latents, frames 1 / 2 from those frames of
 * `latents.1` — denormalized and cut by the reference's tile windows (the metadata's `rows_starts` × `cols_starts`)
 * (if the replacement after the loop is lost, not a single window of the first frame matches; if a copy of the last
 * model input is passed, not a single window of frames 1 / 2 matches). The windows cover the whole latent, so the band
 * of `latents_final` is also the band of what the VAE actually received.
 *
 * ## Device use
 *
 * The one two-step `describe` takes a single device (in its `beforeAll`) and runs every case as an `it` (with
 * `settleReleases` between cases). Building only reads the distribution's parts and creates no Session (each stage builds and disposes
 * its own), so the one-layer-inside state and the public pipeline use the same device in turn. The 50 steps are a
 * separate `describe` (one device) and run as a **separate process** with `-t "50 steps"`.
 *
 * ## Observations (not checks)
 *
 * The time per stage (wall clock — between `stage` events), the VRAM at the stage boundaries
 * (`helpers/drm-usage.ts`), the condition-latent difference as a multiple of the encoder's band, the difference
 * between the first frame and the condition image (`[-1, 1]`, a VAE round trip), the per-frame max absolute difference
 * of `latents.<i>`, and the per-frame max absolute difference between step 0's DiT outputs (uncond / cond) and the
 * reference's `noise_uncond.0` / `noise_cond.0` (step 0's model input differs from the reference only by the
 * condition-latent difference, so this is the clue that tells whether the difference gathers in the condition tokens or
 * shows on the generated side — the note on the derivation of {@link LATENT_RATIO_BANDS}).
 *
 * Environments without the references (`pipeline_steps_i2v.*`) SKIP every GPU case (the condition image is also read
 * from the references) and the host self-consistency check explicitly. An environment with only some of the references
 * FAILs (the same rule as T2V's end-to-end e2e).
 */

import { readFile, writeFile } from "@karume/test-support/fs";
import { afterAll, beforeAll, describe, it, test } from "vite-plus/test";
import { assert, assertEquals, assertThrows } from "@std/assert";
import { nodeDirectory } from "@karume/hub/node";
import {
  type GpuContext,
  parseSafetensors,
  type RunInputs,
  type RunOutputs,
  type SafetensorsFile,
  type Session,
  type SessionDiagnostics,
  type Tensor,
} from "@karume/runtime";
import { encodePng } from "../../models/mod.ts";
import {
  type GeneratedVideo,
  type Rgb8Image,
  frameToRgba,
  type GenerateEvent,
  type Prompt,
  type RunComponent,
  type Ti2vGenerateRequest,
  Ti2vPipeline,
} from "../../models/wan.ts";
import type { ModelComponent } from "../../models/src/hub/components.ts";
import {
  generateWanVideo,
  loadWanFromPretrained,
  ownWanRequestLatents,
  WAN22_TI2V_FAMILY,
  type WanFamilySpec,
  type WanState,
} from "../../models/src/wan/family.ts";
import {
  wanConditionMask,
  wanDitPatch,
  withWanFirstFrameCondition,
} from "../../models/src/wan/dit-loop.ts";
import {
  patchifyLatents,
  unpatchifyTokens,
  type WanPatchGeometry,
  wanTokenGrid,
} from "../../models/src/wan/dit-tokens.ts";
import { timestepsProj } from "../../models/src/wan/dit-timestep.ts";
import { fitWanI2vImage, preprocessWanI2vImage } from "../../models/src/wan/i2v-preprocess.ts";
import {
  denormalizeWanLatents,
  WAN22_LATENTS_MEAN,
  WAN22_LATENTS_STD,
} from "../../models/src/wan/latents.ts";
import {
  WAN_UNIPC_CONFIG,
  wanClassifierFreeGuidance,
  WanUniPcSampler,
  wanUniPcSchedule,
} from "../../models/src/wan/scheduler.ts";
import { WAN_VAE_LATENT_INPUT } from "../../models/src/wan/vae-chunks.ts";
import { acquireTestGpu, GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import { settleReleases } from "../../models/tests/helpers/settle-releases.ts";
import {
  type DrmTimeline,
  formatDrmUsage,
  monitorDrmUsage,
} from "../../models/tests/helpers/drm-usage.ts";
import { filePresent, TEST_ASSETS_FETCH } from "../../models/tests/helpers/wan-ti2v-dit.ts";
import { mirrorPresent, readMirrorAsset } from "../../models/tests/helpers/mirror.ts";
import { TEST_ASSETS, testAssetsFetched } from "@karume/test-support/test-assets";
import {
  WAN_I2V_SOURCE_SHAPE,
  WAN_I2V_STEPS_ROOT,
  wanI2vStepsFixtureUrl,
} from "../../models/tests/helpers/wan-i2v-fixture.ts";
import {
  WAN_I2V_ENCODER_BAND,
  WAN_I2V_IMAGE_SHA256,
} from "../../models/tests/helpers/wan-i2v-image.ts";
import {
  readWanTi2vDistributionManifestText,
  WAN_TI2V_ASSEMBLE_COMMAND,
  WAN_TI2V_DIST_ROOT,
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
 * The band of the two-step run's scheduler latents (after each step — `[48,3,44,80]`) (ratio = max absolute difference
 * ÷ the reference's max absolute value). The index is the step. `undefined` = not derived yet (module doc "Checks" —
 * each case records its ratios and stops red, and the "band candidate" step prints a candidate).
 *
 * The derivation rule: per observation point, the worst of the two deciding cases × 5 (rounded up to 2 significant
 * digits). The candidate the first real run printed is written here together with the table of measurements (the two
 * deciding cases, the band, the acceptance case) (the same form as the doc of the constant of the same name in T2V's
 * `e2e_wan_ti2v_pipeline_test.ts`).
 *
 * Derivation (2026-10-09, RTX 3080 Ti, Deno, reference preset, all three under the opt-in — run twice on the same day,
 * every ratio identical):
 *
 * | Point         | band-boxing-cats | band-cat-dog-baking | worst × 5 | band   | acceptance accept-ferret |
 * | ------------- | ---------------: | ------------------: | --------: | -----: | -----------------------: |
 * | latents.0     |         9.564e-5 |            5.786e-4 |  2.893e-3 | 2.9e-3 |                 5.517e-6 |
 * | latents.1     |         7.108e-5 |            6.985e-4 |  3.492e-3 | 3.5e-3 |                 1.160e-4 |
 * | latents_final |         1.117e-4 |            7.716e-4 |  3.858e-3 | 3.9e-3 |                 4.390e-5 |
 * | frames        |         1.412e-3 |            1.766e-3 |  8.832e-3 | 8.9e-3 |                 7.367e-4 |
 *
 * The latent band is about 4 times wider than T2V's (latents.1 9.5e-4). cat-dog-baking decides it, and the difference
 * gathers in the DiT output of the condition tokens (the first frame, t = 0, the latents of a clean image) (per-frame
 * maxAbs of step 0's cond pass 3.21e-3 / 2.81e-4 / 2.42e-4). Its source is not the precision of the DiT forward but
 * the DiT amplifying the encoder-output difference (condition-latent maxAbs 1.26e-5 — inside the encoder band 7.8e-5)
 * about 250 times in the condition tokens: replacing the first frame of the model input with the reference's
 * `condition_latents` and running the same step 0 (a throwaway diagnostic, 2026-10-09) brings the cond pass's frame-0
 * difference down to 2.41e-4 and the `latents.1` difference to 4.12e-4, back to the same order as T2V (boxing-cats
 * stays at `latents.1` 4.72e-4 with the replacement — the size of the amplification depends on the image). The golden
 * of stage 2's DiT check makes its latents from random numbers, so this sensitivity first shows in the end-to-end run.
 *
 * Fault-injection ratios (latents.1, floor = 2 × the band = 7.0e-3): condition mask all false 1.220, shifted by one
 * token 4.844e-1, the two timestep inputs swapped 1.252, the first frame of the model input not replaced 1.043, the
 * condition latents not normalized 1.098 (at least 138 × the band — the timestep swap was measured on the second run
 * of the same day). The first version, "make the conditioning timestep the same as the generated side's", gave 1.220,
 * the same value as the all-false condition mask (both are the same input, in which every token receives the
 * generated side's timestep), so it was replaced with the timestep swap.
 *
 * The default lane's detection power (a record — the rule is unchanged): the band is decided by the opt-in
 * cat-dog-baking, and the margin of the band as seen from boxing-cats, which runs in the default lane, is 30 × at
 * latents.0, 49 × at latents.1, 35 × at latents_final and 6.3 × at frames (the fault-injection floor is about 98 ×
 * boxing-cats' latents.1). A DiT or UniPC regression that multiplies boxing-cats' latent error by a few tens stays
 * inside the band in the default lane and shows under the opt-in. An encoder regression is named by the
 * condition-latent check (the encoder band — the condition-latent maxAbs is 1.150e-5 for boxing-cats, 1.264e-5 for
 * cat-dog-baking and 5.394e-6 for ferret, the same values as the 1280×704 ones of the encoder's GPU check).
 *
 * MUST: do not change this value or the deciding cases after seeing the acceptance (`accept-ferret`) result (that
 * would break the independence of setting the band and accepting it). If the acceptance case leaves the band,
 * investigate the cause instead of widening the band.
 */
const LATENT_RATIO_BANDS: readonly [number | undefined, number | undefined] = [2.9e-3, 3.5e-3];

/**
 * The band of the last latents (after the replacement — the value handed to the VAE, `[48,3,44,80]`). `undefined` =
 * not derived yet (the same rule as {@link LATENT_RATIO_BANDS}).
 *
 * MUST: do not change this value after seeing the acceptance result (the same as {@link LATENT_RATIO_BANDS}).
 */
const FINAL_RATIO_BAND: number | undefined = 3.9e-3;

/**
 * The band of the output frames (`[3,9,704,1280]` after the clamp — the reference is written after the clamp too).
 * `undefined` = not derived yet (the same rule as {@link LATENT_RATIO_BANDS}).
 *
 * MUST: do not change this value after seeing the acceptance result (the same as {@link LATENT_RATIO_BANDS}).
 */
const FRAME_RATIO_BAND: number | undefined = 8.9e-3;

/**
 * The floor of the factor by which a fault must stay away from the band (2, the same as T2V, 2.1 and stage 2 — an
 * injection that does not turn red is a sign that the band is too wide).
 */
const FAULT_MARGIN = 2;

/**
 * The two-step run's conditions (`STEPS` / `GUIDANCE_SCALE` / `FLOW_SHIFT` / `LATENT_SHAPE` of the reference's
 * `ti2v_i2v_few_step_ref.py`).
 */
const STEPS = 2;
const GUIDANCE = 5;
const SHIFT = 5;
const WIDTH = 1280;
const HEIGHT = 704;
const FRAMES = 9;
const LATENT_SHAPE: readonly [number, number, number, number] = [48, 3, 44, 80];
/** The shape of the condition latents (one latent frame, the first). */
const CONDITION_SHAPE: readonly number[] = [48, 1, 44, 80];
/**
 * The condition token count (the tokens of the first latent frame `(44/2)·(80/2)` — the reference metadata
 * `condition_tokens`).
 */
const CONDITION_TOKENS = 880;
/** The reference's resize mode (upstream's — omitted from the request to use the product default; in the ID). */
const FIT = "crop";

/** The opt-in (module doc "Opt-in" — the remaining two-step reference cases, the portrait case, the 50 steps). */
const I2V_FULL = process.env.KARUME_WAN_TI2V_I2V_FULL === "1";

/**
 * The reference cases (the source of truth is `FIXTURE_CASES` in the recipe `wan/ti2v_i2v_few_step_ref.py` — the
 * prompt and the image share a name). Two deciding (`band`) cases + one acceptance (`accept`) case. The default lane
 * runs only the first deciding case (ruling 8 = b); the rest are opt-in.
 * MUST: do not change the deciding cases after seeing the acceptance result.
 */
const CASES: readonly {
  readonly name: string;
  readonly role: "band" | "accept";
  readonly optIn: boolean;
}[] = [
  { name: "band-boxing-cats", role: "band", optIn: false },
  { name: "band-cat-dog-baking", role: "band", optIn: true },
  { name: "accept-ferret", role: "accept", optIn: true },
];
/** The default lane's case (fault injection and the practical preset use this case's request too). */
const DEFAULT_CASE = "band-boxing-cats";

/**
 * The case ID of a sha row (it carries the preset, I2V and fit, step count, case, size, frame count and shift — ruling
 * F12). {@link i2vHead} spells `head`.
 *
 * MUST: when the request (image, fit, prompt, seed / initial noise, step count, size, frame count, shift, preset)
 * changes, change the ID too (otherwise the row's value is compared with the sha of other conditions).
 */
const caseIdOf = (head: string, width: number, height: number, frames: number): string =>
  `${head}-${width}x${height}-${frames}f-shift${SHIFT}`;

/** The ID head (`<preset>-i2v-<fit>-<step count>-<case>` — `-i2v-` sets it apart from the T2V IDs). */
const i2vHead = (quant: string, steps: string, name: string): string =>
  `${quant}-i2v-${FIT}-${steps}-${name}`;

/** The band cases' ID (reference preset, precomputed path). */
const bandCaseId = (name: string): string =>
  caseIdOf(i2vHead(WAN_TI2V_REFERENCE_QUANT, "2step", name), WIDTH, HEIGHT, FRAMES);

/**
 * The practical preset's case (the same request as {@link DEFAULT_CASE}, only the preset is the practical one — a
 * practical row of the default lane). It carries no band (the practical preset's numeric check is the self A/B check —
 * the I2V A/B is stage 9d). It runs through the public surface.
 */
const PRACTICAL_CASE_ID = caseIdOf(
  i2vHead(WAN_TI2V_PRACTICAL_QUANT, "2step", DEFAULT_CASE),
  WIDTH,
  HEIGHT,
  FRAMES,
);

/**
 * The seed path's image and prompt (same name — the source image embedded in the reference of {@link DEFAULT_CASE}) and
 * seed.
 */
const SEED_IMAGE = "boxing-cats";
const SEED = 42;

/**
 * The portrait case (opt-in, reference preset, public surface). The landscape source image is cropped to an explicit
 * 704×1280 (the same as the portrait case of stage 9a's encoder golden). There is no CPU reference, so it is pinned
 * only by completion, 0 non-finite values and the sha256 per-environment row.
 */
const PORTRAIT_CASE_ID = caseIdOf(
  i2vHead(WAN_TI2V_REFERENCE_QUANT, "2step", `${SEED_IMAGE}-seed${SEED}`),
  HEIGHT,
  WIDTH,
  FRAMES,
);

/** The 50-step run's frame count (the default frame count). */
const FULL_FRAMES = 33;
/**
 * The 50-step run's step count. The request does not set it and leaves it to the manifest's default, so it is pinned
 * by the observed number of denoise steps (if the manifest's default changes, it disagrees with the ID's `50step`).
 */
const FULL_STEPS = 50;
/**
 * The 50-step run's presets (the reference preset, with the larger allocation, first — the same reason as T2V's 50
 * steps).
 */
const FULL_CASES: readonly { readonly id: string; readonly quant: string }[] = [
  WAN_TI2V_REFERENCE_QUANT,
  WAN_TI2V_PRACTICAL_QUANT,
].map((quant) => ({
  id: caseIdOf(i2vHead(quant, "50step", `${SEED_IMAGE}-seed${SEED}`), WIDTH, HEIGHT, FULL_FRAMES),
  quant,
}));

/**
 * The text-embedding asset the references were made with (ADR 0121 decision 9): the mirror's `text_embeds` (ADR 0139
 * decision 1 — the same bytes as the series file the references were made from). The host self-consistency check
 * compares its sha with the references' meta `text_embeds_sha256` (as the T2V end-to-end test does).
 */
const TEXT_EMBEDS = "text_embeds";

/** The distribution's manifest (when absent the end-to-end comparison SKIPs — this e2e does not read its content). */
const DIST_PRESENT = (await readWanTi2vDistributionManifestText()) !== undefined;
if (!DIST_PRESENT) {
  console.warn(
    `[karume] the distribution mirror ${WAN_TI2V_DIST_ROOT.pathname} is absent, so Wan2.2's I2V end-to-end comparison is SKIPPED. ` +
      `Assemble it: ${WAN_TI2V_ASSEMBLE_COMMAND} (the gate test distribution_gate_test.ts turns an all-SKIP into a FAIL)`,
  );
}
const FIXTURES_PRESENT = CASES.map(({ name }) => filePresent(wanI2vStepsFixtureUrl(name)));
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
    `[karume] no I2V two-step reference (pipeline_steps_i2v.*) under ${WAN_I2V_STEPS_ROOT.pathname}, so Wan2.2's ` +
      `I2V end-to-end check (the condition image is also read from the references) and the host self-consistency ` +
      `check are SKIPPED (${MISSING_GOLDENS})`,
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
      `absent), so Wan2.2's I2V host self-consistency check (it reads the mirror's ${TEXT_EMBEDS}) is SKIPPED`,
  );
}

const references = openReferences(
  new URL("../../models/tests/fixtures/references/wan-ti2v.json", import.meta.url),
);
const results = openResults("wan-ti2v-i2v-pipeline");
/**
 * The 50-step run's results directory (`results.json` is rewritten whole on every run, so it is kept apart from the
 * runs in another process).
 */
const fullResults = openResults("wan-ti2v-i2v-pipeline-full");

/** A value a `beforeAll` assigns, read inside an `it` (fails loudly when the hook did not assign it). */
const assigned = <T>(value: T | undefined, what: string): T => {
  assert(value !== undefined, `${what} is not assigned (its beforeAll did not complete)`);
  return value;
};

test.skipIf(!ANY_FIXTURE && !FETCHED)(
  "Wan2.2 I2V end-to-end references: all three two-step references are present",
  () => {
    assertEquals(
      FIXTURES_PRESENT,
      FIXTURES_PRESENT.map(() => true),
      `missing: ${CASES.filter((_, index) => !FIXTURES_PRESENT[index])
        .map(({ name }) => wanI2vStepsFixtureUrl(name).pathname)
        .join(", ")} (${MISSING_GOLDENS})`,
    );
  },
);

/** One reference (F32 / U8 tensors, shapes, metadata). */
type Fixture = {
  readonly tensor: (key: string) => Float32Array<ArrayBuffer>;
  readonly bytes: (key: string) => Uint8Array<ArrayBuffer>;
  readonly shape: (key: string) => readonly number[];
  readonly meta: (key: string) => string;
};

const readFixture = async (url: URL): Promise<Fixture> => {
  const bytes = await readFile(url);
  const file: SafetensorsFile = parseSafetensors(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
  const view = (key: string, dtype: "F32" | "U8") => {
    const found = file.tensors.get(key);
    assert(
      found !== undefined && found.dtype === dtype,
      `${url.pathname}: '${key}' is not ${dtype}`,
    );
    return found;
  };
  return {
    tensor: (key) => {
      const found = view(key, "F32");
      return new Float32Array(file.buffer, found.byteOffset, found.byteLength / 4);
    },
    bytes: (key) => {
      const found = view(key, "U8");
      return new Uint8Array(file.buffer, found.byteOffset, found.byteLength);
    },
    shape: (key) => {
      const found = file.tensors.get(key);
      assert(found !== undefined, `${url.pathname}: '${key}' is missing`);
      return found.shape;
    },
    meta: (key) => {
      const value = file.metadata.get(key);
      assert(value !== undefined, `${url.pathname}: metadata '${key}' is missing`);
      return value;
    },
  };
};

/** The source image embedded in the reference (RGB8, HWC — the form passed to the product's `image`). */
const sourceImageOf = (fixture: Fixture): Rgb8Image => {
  assertEquals(fixture.shape("source"), WAN_I2V_SOURCE_SHAPE, "source image shape");
  return {
    width: WAN_I2V_SOURCE_SHAPE[1],
    height: WAN_I2V_SOURCE_SHAPE[0],
    data: fixture.bytes("source"),
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

/**
 * The first index where two arrays differ by value (`===` — -0 and +0 count as equal) (-1 when they match, 0 when the
 * lengths differ). The wiring checks look at "the same value was passed", so a replacement formula
 * `(1 − m)·cond + m·latents` that only moves signed zeros is not counted as a mismatch.
 */
const firstValueMismatch = (
  got: Float32Array | Uint32Array,
  want: Float32Array | Uint32Array,
): number => {
  if (got.length !== want.length) return 0;
  return got.findIndex((value, index) => value !== want[index]);
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
 * The difference between the output's first frame and the condition image (the RGB8 after the crop mapped to `[-1, 1]`)
 * (an observation — it includes the error of the VAE round trip; not a check). `video.data` is `[3, F, H, W]`, `rgb8`
 * is `[H, W, 3]`.
 */
const firstFrameDifference = (
  video: GeneratedVideo,
  rgb8: Uint8Array,
): { readonly maxAbs: number; readonly meanAbs: number } => {
  const plane = video.width * video.height;
  assertEquals(rgb8.length, plane * 3, "condition image element count");
  let maxAbs = 0;
  let sum = 0;
  for (let channel = 0; channel < 3; channel += 1) {
    const base = channel * video.frames * plane;
    for (let index = 0; index < plane; index += 1) {
      const want = Math.fround(2 * Math.fround(rgb8[index * 3 + channel] / 255) - 1);
      const absolute = Math.abs(video.data[base + index] - want);
      maxAbs = Math.max(maxAbs, absolute);
      sum += absolute;
    }
  }
  return { maxAbs, meanAbs: sum / (plane * 3) };
};

const formatFirstFrame = (diff: { readonly maxAbs: number; readonly meanAbs: number }): string =>
  `first frame vs condition image ([-1,1], observation) maxAbs ${diff.maxAbs.toFixed(3)}, meanAbs ${diff.meanAbs.toFixed(
    4,
  )}`;

/** Cuts one frame of the latents `[C, F, H, W]` (`LATENT_SHAPE`) out as `[C, 1, H, W]`. */
const latentFrame = (latents: Float32Array, frame: number): Float32Array<ArrayBuffer> => {
  const [channels, frames, height, width] = LATENT_SHAPE;
  const plane = height * width;
  assertEquals(latents.length, channels * frames * plane, "latent element count");
  const out = new Float32Array(channels * plane);
  for (let channel = 0; channel < channels; channel += 1) {
    const from = (channel * frames + frame) * plane;
    out.set(latents.subarray(from, from + plane), channel * plane);
  }
  return out;
};

/**
 * The per-frame max absolute difference of the latents (an observation — the first frame is a value the scheduler state
 * does not replace, advanced from the DiT output of the condition tokens. Recording which frame the difference sits in
 * is a clue for reading the band).
 */
const frameMaxAbs = (got: Float32Array, want: Float32Array): string =>
  Array.from({ length: LATENT_SHAPE[1] }, (_, frame) =>
    difference(latentFrame(got, frame), latentFrame(want, frame)).maxAbs.toExponential(2),
  ).join(", ");

const textOf = (
  prompts: readonly Prompt[],
  name: string,
  form: "prompt" | "normalized",
): string => {
  const entry = prompts.find((candidate) => candidate.name === name);
  assert(entry !== undefined, `the embedding asset has no '${name}'`);
  return entry[form];
};

// ---- hooks for observation and fault injection (module doc "Hooks for observation and fault injection") ----------

/**
 * The input names of the DiT's S-shaped graph (the forward argument names of the recipe's `wan/export_dit.py` /
 * `wan/ti2v_export_dit.py` — the same spelling as the private constants of the product's `dit-loop.ts`. If they
 * differ, {@link f32Input} / {@link boolInput} fail naming them).
 */
const DIT_TOKENS = "tokens";
const DIT_TIMESTEPS_PROJ = "timesteps_proj";
const DIT_TIMESTEPS_PROJ_CONDITION = "timesteps_proj_condition";
const DIT_CONDITION_MASK = "condition_mask";

/** A run input's f32 host tensor (missing, resident or a different dtype fails loudly). */
const f32Input = (inputs: RunInputs, name: string): Extract<Tensor, { readonly dtype: "f32" }> => {
  const input = inputs[name];
  if (input === undefined || !("data" in input) || input.dtype !== "f32") {
    throw new Error(
      `input '${name}' is not an f32 host tensor (inputs: ${Object.keys(inputs).join(", ")})`,
    );
  }
  return input;
};

/** A run input's bool host tensor (the same rule as {@link f32Input}). */
const boolInput = (
  inputs: RunInputs,
  name: string,
): Extract<Tensor, { readonly dtype: "bool" }> => {
  const input = inputs[name];
  if (input === undefined || !("data" in input) || input.dtype !== "bool") {
    throw new Error(
      `input '${name}' is not a bool host tensor (inputs: ${Object.keys(inputs).join(", ")})`,
    );
  }
  return input;
};

/**
 * A copy of a component whose Session is wrapped in a Proxy that passes `method`'s inputs through `hook` (the hook for
 * observation and fault injection — no hook is added to the product code). `run`'s outputs go to `onOutputs` (an
 * observation — the values are returned as they are). Other methods are bound to the original Session (the Session
 * has `#` private fields, so calling them with the Proxy as `this` fails).
 */
const withSessionHook = (
  component: ModelComponent,
  method: "run" | "enqueue",
  hook: (inputs: RunInputs) => RunInputs,
  onOutputs?: (outputs: RunOutputs) => void,
): ModelComponent => ({
  ...component,
  createSession: async (gpu, options) => {
    const session = await component.createSession(gpu, options);
    return new Proxy<Session>(session, {
      get: (target, property) => {
        if (property === "run" && method === "run") {
          return (...args: Parameters<Session["run"]>) => {
            const [inputs, ...rest] = args;
            return target.run(hook(inputs), ...rest).then((outputs) => {
              onOutputs?.(outputs);
              return outputs;
            });
          };
        }
        if (property === "enqueue" && method === "enqueue") {
          return (...args: Parameters<Session["enqueue"]>) => {
            const [inputs, ...rest] = args;
            return target.enqueue(hook(inputs), ...rest);
          };
        }
        const value: unknown = Reflect.get(target, property, target);
        return typeof value === "function" ? value.bind(target) : value;
      },
    });
  },
});

/**
 * The patchify permutation (which latent index each element of the token order comes from). Built by passing an index
 * array through the product's `patchifyLatents` (the latent element count 506,880 is exact in f32). It also checks
 * here that it is a permutation (every index exactly once).
 */
const tokenSourcesOf = (patch: WanPatchGeometry): Int32Array<ArrayBuffer> => {
  const count = LATENT_SHAPE.reduce((product, dim) => product * dim, 1);
  const indices = new Float32Array(count);
  for (let index = 0; index < count; index += 1) indices[index] = index;
  const sources = Int32Array.from(patchifyLatents(indices, LATENT_SHAPE, patch));
  const seen = new Uint8Array(count);
  for (const source of sources) seen[source] += 1;
  assert(
    seen.every((times) => times === 1),
    "patchify is not a permutation of the latent indices",
  );
  return sources;
};

/**
 * The inputs of one DiT run (an observation — `modelInput` is the model input's latents with the patchify undone,
 * `[C, F, H, W]`).
 */
type DitRunInputs = {
  readonly modelInput: Float32Array<ArrayBuffer>;
  readonly proj: Float32Array<ArrayBuffer>;
  readonly conditionProj: Float32Array<ArrayBuffer>;
  readonly mask: Uint32Array<ArrayBuffer>;
};

/** The record of the inputs passed to the Sessions in one generate. */
type SessionRecord = {
  /** Per DiT run (after the rewrite — the values the DiT actually received). */
  readonly dit: DitRunInputs[];
  /**
   * The output of each DiT run (`[C, F, H, W]` after unpatchify — the same order as the reference's `noise_uncond.<i>`
   * / `noise_cond.<i>`).
   */
  readonly ditOutputs: Float32Array<ArrayBuffer>[];
  /** The `latent` of each enqueue of the VAE's first chunk (`vae_decoder_first`) (denormalized tile `[C, 1, t, t]`). */
  readonly vaeFirstLatents: Float32Array<ArrayBuffer>[];
  /** The `latent` of each enqueue of the following VAE chunks (`vae_decoder_next` — latent frames 1 on; same shape). */
  readonly vaeNextLatents: Float32Array<ArrayBuffer>[];
};

const emptyRecord = (): SessionRecord => ({
  dit: [],
  ditOutputs: [],
  vaeFirstLatents: [],
  vaeNextLatents: [],
});

/**
 * A copy of the state with the Sessions of `transformer` and `vae_decoder_first` / `vae_decoder_next` wrapped (their
 * inputs are copied to `record`). `alter` rewrites the inputs of the DiT runs (fault injection — when omitted, the
 * input objects are passed through as they are).
 */
const probedState = (
  state: WanState,
  record: SessionRecord,
  sources: Int32Array,
  alter?: (inputs: RunInputs) => RunInputs,
): WanState => ({
  ...state,
  transformer: withSessionHook(
    state.transformer,
    "run",
    (inputs) => {
      const altered = alter === undefined ? inputs : alter(inputs);
      const tokens = f32Input(altered, DIT_TOKENS).data;
      assertEquals(tokens.length, sources.length, "DiT tokens element count");
      const modelInput = new Float32Array(sources.length);
      for (let index = 0; index < sources.length; index += 1) {
        modelInput[sources[index]] = tokens[index];
      }
      record.dit.push({
        modelInput,
        proj: Float32Array.from(f32Input(altered, DIT_TIMESTEPS_PROJ).data),
        conditionProj: Float32Array.from(f32Input(altered, DIT_TIMESTEPS_PROJ_CONDITION).data),
        mask: Uint32Array.from(boolInput(altered, DIT_CONDITION_MASK).data),
      });
      return altered;
    },
    (outputs) => {
      record.ditOutputs.push(
        unpatchifyTokens(f32Input(outputs, state.dit.output).data, LATENT_SHAPE, state.dit.patch),
      );
    },
  ),
  vaeFirst: withSessionHook(state.vaeFirst, "enqueue", (inputs) => {
    record.vaeFirstLatents.push(Float32Array.from(f32Input(inputs, WAN_VAE_LATENT_INPUT).data));
    return inputs;
  }),
  vaeNext: withSessionHook(state.vaeNext, "enqueue", (inputs) => {
    record.vaeNextLatents.push(Float32Array.from(f32Input(inputs, WAN_VAE_LATENT_INPUT).data));
    return inputs;
  }),
});

/**
 * Checks the DiT input wiring (module doc "Checks") on the run's record and returns the condition latents (the first
 * frame of the model input). `stepLatents` are the observed scheduler latents after each step.
 */
const assertDitWiring = (
  runs: readonly DitRunInputs[],
  latentsInit: Float32Array,
  stepLatents: readonly Float32Array[],
  where: string,
): Float32Array<ArrayBuffer> => {
  assertEquals(runs.length, STEPS * 2, `${where}: DiT run count (2 CFG passes × ${STEPS} steps)`);
  const [, frames] = LATENT_SHAPE;
  const condition = latentFrame(runs[0].modelInput, 0);
  const expectedMask = new Uint32Array(runs[0].mask.length);
  expectedMask.fill(1, 0, CONDITION_TOKENS);
  // The expected conditioning timestep (upstream's `first_frame_mask · t` — the condition tokens get t = 0; the width
  // is the same as the generated side's proj).
  const zeroProj = timestepsProj(0, runs[0].proj.length);
  runs.forEach((run, index) => {
    const step = Math.floor(index / 2);
    const label = `${where}: step ${step} run ${index}`;
    assertEquals(
      firstValueMismatch(run.mask, expectedMask),
      -1,
      `${label}: condition mask is not true for exactly the first ${CONDITION_TOKENS} tokens`,
    );
    // The proj at t = 0 in every run, so this also covers not depending on the step and differing from the generated
    // side (t > 0).
    assert(
      bitsEqual(run.conditionProj, zeroProj),
      `${label}: conditioning timestep is not bit-identical to the proj at t = 0 (timestepsProj(0))`,
    );
    assertEquals(
      firstValueMismatch(latentFrame(run.modelInput, 0), condition),
      -1,
      `${label}: model input's first frame differs from run 0 (not replaced by the condition latents)`,
    );
    const current = step === 0 ? latentsInit : stepLatents[step - 1];
    for (let frame = 1; frame < frames; frame += 1) {
      assertEquals(
        firstValueMismatch(latentFrame(run.modelInput, frame), latentFrame(current, frame)),
        -1,
        `${label}: model input frame ${frame} differs from that step's scheduler state`,
      );
    }
  });
  return condition;
};

/**
 * The VAE stage's inputs (denormalized tiles) match by value an expectation assembled without the replacement function
 * (module doc "Observing the last latents"): the first chunk (latent frame 0) is the condition latents, the following
 * chunks (frames 1 on) are those frames of the last scheduler state `schedulerLatents` — both denormalized and cut by
 * the reference's tile windows. This pins, on the values the VAE actually received, that the replacement after the
 * loop reached the VAE and that the frames not replaced were passed as the last state. It does not depend on the
 * window order (each chunk is matched as a multiset).
 */
const assertVaeReceivesLatents = (
  record: Pick<SessionRecord, "vaeFirstLatents" | "vaeNextLatents">,
  condition: Float32Array<ArrayBuffer>,
  schedulerLatents: Float32Array,
  fixture: Fixture,
  where: string,
): void => {
  const tile = Number(fixture.meta("tile"));
  const rows = fixture.meta("rows_starts").split(",").map(Number);
  const cols = fixture.meta("cols_starts").split(",").map(Number);
  const [channels, frames, height, width] = LATENT_SHAPE;
  const stats = WAN22_TI2V_FAMILY.generation.latents;
  const expected = Array.from({ length: frames }, (_, frame) =>
    denormalizeWanLatents(frame === 0 ? condition : latentFrame(schedulerLatents, frame), stats),
  );
  const chunks = [
    {
      label: "first chunk (vae_decoder_first)",
      recorded: record.vaeFirstLatents,
      frames: [0],
      cause: "the replacement after the loop did not reach the VAE",
    },
    {
      label: "following chunks (vae_decoder_next)",
      recorded: record.vaeNextLatents,
      frames: Array.from({ length: frames - 1 }, (_, index) => index + 1),
      cause:
        "a frame that is not replaced is not the last scheduler state (e.g. a copy of the last model input)",
    },
  ];
  for (const chunk of chunks) {
    assertEquals(
      chunk.recorded.length,
      rows.length * cols.length * chunk.frames.length,
      `${where}: VAE ${chunk.label} enqueue count (tiles × frames)`,
    );
    const unmatched = [...chunk.recorded];
    for (const frame of chunk.frames) {
      for (const top of rows) {
        for (const left of cols) {
          assert(
            top + tile <= height && left + tile <= width,
            `${where}: window (${top}, ${left}) is outside the latents`,
          );
          const window = new Float32Array(channels * tile * tile);
          for (let channel = 0; channel < channels; channel += 1) {
            for (let y = 0; y < tile; y += 1) {
              const from = (channel * height + top + y) * width + left;
              window.set(expected[frame].subarray(from, from + tile), (channel * tile + y) * tile);
            }
          }
          const at = unmatched.findIndex((latent) => firstValueMismatch(latent, window) === -1);
          assert(
            at !== -1,
            `${where}: the VAE ${chunk.label} inputs lack the expected denormalization of latent frame ${frame}, window (row ${top}, col ${left})` +
              ` (${chunk.cause}, or the tile windows differ from the reference's plan)`,
          );
          unmatched.splice(at, 1);
        }
      }
    }
  }
};

/**
 * The condition-latent check (the encoder stage's exit — the band of the encoder's GPU check, {@link
 * WAN_I2V_ENCODER_BAND}, applied to the max absolute difference between the first frame the DiT received in the
 * end-to-end run and the reference's `condition_latents`). An encoder regression (swapped bindings, mixed-up
 * normalization statistics, f16 rounding) is self-consistent as "the condition" inside the end-to-end run, so leaving
 * it to the latent band cannot name the stage at fault (in the default case, boxing-cats, the DiT barely amplifies the
 * encoder's difference — the note on the derivation of {@link LATENT_RATIO_BANDS}).
 */
const assertConditionInBand = (diff: Difference, where: string): void => {
  assertEquals(diff.nonFinite, 0, `${where}: condition latents non-finite values`);
  assert(
    diff.maxAbs <= WAN_I2V_ENCODER_BAND,
    `${where}: maxAbs between the condition latents (the VAE encoder stage's exit, after normalization) and the reference's condition_latents ` +
      `${diff.maxAbs.toExponential(3)} is outside the encoder band ${WAN_I2V_ENCODER_BAND}`,
  );
};

// ---- observing one generation ---------------------------------------------------------------------------

/** The generation entry (the one-layer-inside `generateWanVideo` or the public `Ti2vPipeline.generate`). */
type Generate = (request: Ti2vGenerateRequest) => Promise<GeneratedVideo>;

const viaState =
  (state: WanState, spec: WanFamilySpec = WAN22_TI2V_FAMILY): Generate =>
  (request) =>
    generateWanVideo(spec, state, ownWanRequestLatents(request));

const viaPipeline =
  (pipeline: Ti2vPipeline): Generate =>
  (request) =>
    pipeline.generate(request);

/**
 * The observations of one generate (from the events — the latents after each step, the time per stage, the VRAM at the
 * stage boundaries).
 */
type Observed = {
  readonly video: GeneratedVideo;
  readonly latents: readonly Float32Array<ArrayBuffer>[];
  /** Stage name → wall-clock ms (from `start` to `end`). */
  readonly stageMs: Readonly<Record<string, number>>;
  readonly wallMs: number;
  readonly drm: DrmTimeline;
  readonly diagnostics: ReadonlyMap<RunComponent, SessionDiagnostics>;
};

/**
 * Runs generate once and collects observations from the events. With `abortAfterDenoise` it throws at the DiT stage's
 * `end` (after its Session is disposed) and skips the VAE stage (fault injection looks only at the latents — it does
 * not pay for the 28-tile VAE). `onLatents` receives the scheduler latents after each step (the "not replaced" fault
 * injection builds the next step's model input from them).
 */
const observe = async (
  generate: Generate,
  diagnostics: Map<RunComponent, SessionDiagnostics>,
  request: Omit<Ti2vGenerateRequest, "onEvent">,
  options: {
    readonly abortAfterDenoise?: boolean;
    readonly onLatents?: (latents: Float32Array<ArrayBuffer>) => void;
  } = {},
): Promise<Observed | { readonly latents: readonly Float32Array<ArrayBuffer>[] }> => {
  const latents: Float32Array<ArrayBuffer>[] = [];
  const stageStarted = new Map<string, number>();
  const stageMs: Record<string, number> = {};
  const monitor = monitorDrmUsage();
  const abort = new Error("stop after the DiT stage (fault injection)");
  diagnostics.clear();
  const started = performance.now();
  const onEvent = (event: GenerateEvent): void => {
    if (event.kind === "denoise-step") {
      const data = event.copyLatents().data;
      latents.push(data);
      options.onLatents?.(data);
    }
    if (event.kind !== "stage") return;
    const label = `${event.component} ${event.at}`;
    monitor.mark(label);
    monitor.enter(label);
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
    const video = await generate({ ...request, onEvent });
    const wallMs = performance.now() - started;
    return {
      video,
      latents,
      stageMs,
      wallMs,
      drm: monitor.stop(),
      diagnostics: new Map(diagnostics),
    };
  } catch (error) {
    monitor.stop();
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
  ];
  for (const [component, diagnostics] of observed.diagnostics) {
    lines.push(
      `${component}: weights ${gib(diagnostics.weights.allocatedBytes)}, backing ${gib(
        diagnostics.planBacking.residentBytes + diagnostics.planBacking.inputBytes,
      )}, geometry ${diagnostics.geometryProfile}, ${diagnostics.submit.submitCount} submits`,
    );
  }
  return lines;
};

/**
 * The stage order (encoder → DiT — the precomputed path has no text stage) and the diagnostics of the encoder's three
 * graphs (the wiring of a real run of a request with a condition image).
 */
const assertEncoderStage = (observed: Observed, where: string): void => {
  assertEquals(
    observed.drm.marks.slice(0, 3).map(({ label }) => label),
    ["vae_encoder start", "vae_encoder end", "transformer start"],
    `${where}: encoder stage → DiT stage order`,
  );
  for (const component of ["vae_encoder_pre", "vae_encoder_attn", "vae_encoder_post"] as const) {
    assert(
      observed.diagnostics.has(component),
      `${where}: the diagnostics of the ${component} run did not arrive`,
    );
  }
};

/** The common part of the two-step run's requests (each case adds the prompt, image and initial noise). */
const TWO_STEP = { steps: STEPS, guidance: GUIDANCE, shift: SHIFT, frames: FRAMES } as const;

test.skipIf(!ANY_FIXTURE || !TEXT_EMBEDS_READABLE)(
  "Wan2.2 I2V end-to-end (host, no GPU): the reference metadata, image provenance and crop, and the 48 statistics " +
    "match TS; from the reference DiT outputs, CFG + UniPC rebuild the reference latents and the replacement function the last latents bit for bit",
  async () => {
    const embedsSha = await sha256Hex(await readMirrorAsset(WAN_TI2V_DIST_ROOT, TEXT_EMBEDS));
    const schedule = wanUniPcSchedule(STEPS, SHIFT, WAN_UNIPC_CONFIG.numTrainTimesteps);
    const mean = Float32Array.from(WAN22_LATENTS_MEAN);
    const std = Float32Array.from(WAN22_LATENTS_STD);
    const patch = wanDitPatch(LATENT_SHAPE[0]);
    const mask = wanConditionMask("ti2v", wanTokenGrid(LATENT_SHAPE, patch), patch, true, "test");
    assert(mask !== undefined, "no ti2v condition mask");
    // The number of true tokens of the product's condition mask (matched against the reference metadata
    // `condition_tokens` below).
    assertEquals(
      mask.reduce((count, value) => count + value, 0),
      CONDITION_TOKENS,
      "the product's condition mask",
    );
    for (const { name, role } of CASES) {
      const fixture = await readFixture(wanI2vStepsFixtureUrl(name));
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
          fixture.meta("timestep_mlp"),
          fixture.meta("condition_tokens"),
          fixture.meta("condition_timestep"),
        ],
        [String(STEPS), "5.0", "5.0", "true", "per-value-m1", String(CONDITION_TOKENS), "0"],
        `${name}: step count, guidance, shift, timestep path, condition token count and timestep`,
      );
      assertEquals(fixture.meta("negative"), "negative", `${name}: negative row name`);
      // Image provenance: the prompt and the image share a name, and the PNG's sha is the same value as the provenance
      // of stage 9a's golden.
      const image = fixture.meta("image");
      assertEquals(
        fixture.meta("prompt"),
        image,
        `${name}: the prompt and the image do not share a name`,
      );
      assertEquals(
        fixture.meta("image_sha256"),
        WAN_I2V_IMAGE_SHA256[image],
        `${name}: image sha256`,
      );
      assertEquals(
        [
          fixture.meta("width"),
          fixture.meta("height"),
          fixture.meta("fit"),
          fixture.meta("resample"),
        ],
        [String(WIDTH), String(HEIGHT), FIT, "LANCZOS"],
        `${name}: size, fit, resample`,
      );
      const source = sourceImageOf(fixture);
      const rgb8 = fixture.bytes("rgb8");
      assertEquals(fixture.shape("rgb8"), [HEIGHT, WIDTH, 3], `${name}: cropped RGB8 shape`);
      assertEquals(
        await sha256Hex(fixture.bytes("source")),
        fixture.meta("source_rgb8_sha256"),
        `${name}: source image RGB8 sha256`,
      );
      assertEquals(
        await sha256Hex(rgb8),
        fixture.meta("rgb8_sha256"),
        `${name}: cropped RGB8 sha256`,
      );
      // The product preprocessing: the size is 1280×704 from the image's aspect ratio, and the crop is bit-identical to
      // the reference's crop.
      const selected = preprocessWanI2vImage(source, {}, WAN22_TI2V_FAMILY.generation);
      assertEquals([selected.width, selected.height], [WIDTH, HEIGHT], `${name}: size selection`);
      const cropped = fitWanI2vImage(source, { width: WIDTH, height: HEIGHT }, "crop");
      const mismatch = cropped.data.findIndex((value, index) => value !== rgb8[index]);
      assertEquals(
        mismatch,
        -1,
        `${name}: the TS crop differs from the reference's rgb8 at pixel ${mismatch}`,
      );
      assertEquals(fixture.shape("latents_init"), LATENT_SHAPE, `${name}: latent shape`);
      assertEquals(
        fixture.shape("condition_latents"),
        CONDITION_SHAPE,
        `${name}: condition latent shape`,
      );
      assertEquals(fixture.shape("frames"), [3, FRAMES, HEIGHT, WIDTH], `${name}: frame shape`);
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
      // Rebuild the reference latents with the host's CFG + UniPC (without the replacement) from the DiT outputs the
      // reference recorded.
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
      // The last latents = the last scheduler latents with the first frame replaced by the condition latents (rebuilt
      // with the product's function).
      const final = withWanFirstFrameCondition(
        fixture.tensor(`latents.${STEPS - 1}`),
        fixture.tensor("condition_latents"),
        LATENT_SHAPE,
      );
      assert(
        bitsEqual(final, fixture.tensor("latents_final")),
        `${name}: the last latents are not bit-identical (${formatDifference(
          "latents_final",
          difference(final, fixture.tensor("latents_final")),
        )}）`,
      );
    }
  },
);

/**
 * The observation points (outputs with a band — the scheduler latents after each step, the last latents, the clamped
 * frames).
 */
const OBSERVATION_POINTS = ["latents.0", "latents.1", "latents_final", "frames"] as const;
type ObservationPoint = (typeof OBSERVATION_POINTS)[number];

/** The band per observation point (`undefined` when not derived). */
const bandOf = (point: ObservationPoint): number | undefined => {
  switch (point) {
    case "latents.0":
      return LATENT_RATIO_BANDS[0];
    case "latents.1":
      return LATENT_RATIO_BANDS[1];
    case "latents_final":
      return FINAL_RATIO_BAND;
    case "frames":
      return FRAME_RATIO_BAND;
  }
};

const BANDS_DERIVED = OBSERVATION_POINTS.every((point) => bandOf(point) !== undefined);

/**
 * One case's ratios per observation point (the material for the band candidate — stored after the non-finite and
 * device-lost checks pass and before the band verdict).
 */
type CaseRatios = {
  readonly name: string;
  readonly role: "band" | "accept";
  readonly ratios: Readonly<Record<ObservationPoint, number>>;
};

/**
 * The band candidate (worst × 5 rounded up to 2 significant digits — the same rule as the function of the same name in
 * T2V's end-to-end e2e). So that a division error does not round an exactly-2-digit value up by one unit, the quotient
 * is rounded to 12 digits before rounding up.
 */
const roundUpTwoDigits = (value: number): number => {
  const unit = 10 ** (Math.floor(Math.log10(value)) - 1);
  return Number((Math.ceil(Number((value / unit).toPrecision(12))) * unit).toPrecision(2));
};

describe.skipIf(!DIST_PRESENT || !GPU_AVAILABLE || !ANY_FIXTURE)(
  "Wan2.2 I2V end-to-end 2 steps (real GPU): 1280×704, 9 frames, DiT input wiring, " +
    "fault injection outside the band, sha256 per-environment rows (reference preset's band cases, practical preset, opt-in portrait) — the latents, last latents, " +
    "frames and condition latents inside the bands of the diffusers reference only with KARUME_UPSTREAM_PARITY=1",
  () => {
    let deviceLost: string | undefined;
    // One device and one reference-quant inner state shared by the its below (built before them, released after them all).
    let gpuContext: GpuContext | undefined;
    let innerState: WanState | undefined;
    let statePrompts: readonly Prompt[] | undefined;
    let tokenSources: Int32Array<ArrayBuffer> | undefined;
    const diagnostics = new Map<RunComponent, SessionDiagnostics>();
    const onRunDiagnostics = (component: RunComponent, diagnosed: SessionDiagnostics): void => {
      diagnostics.set(component, diagnosed);
    };
    /**
     * The artifact of the reference preset's default case (RGB bytes) — the partner of the floor of the practical
     * preset's case for the same request.
     */
    let referenceRgb: Uint8Array<ArrayBuffer> | undefined;
    /** The default case's reference (read by fault injection, the practical preset and the portrait case). */
    let defaultFixture: Fixture | undefined;
    const defaultFixtureOf = async (): Promise<Fixture> =>
      (defaultFixture ??= await readFixture(wanI2vStepsFixtureUrl(DEFAULT_CASE)));
    /** The default case's request (the practical preset uses the same request — only the preset differs). */
    const defaultRequest = async (
      prompts: readonly Prompt[],
    ): Promise<Omit<Ti2vGenerateRequest, "onEvent">> => {
      const fixture = await defaultFixtureOf();
      return {
        ...TWO_STEP,
        prompt: textOf(prompts, fixture.meta("prompt"), "prompt"),
        negativePrompt: textOf(prompts, fixture.meta("negative"), "prompt"),
        latents: fixture.tensor("latents_init"),
        image: sourceImageOf(fixture),
      };
    };
    /**
     * The end point that compares (or writes) one sha256 per-environment row. `beforeSettle` receives the artifact
     * before the comparison and the write (the place for checks that must apply before a row is written, even in a
     * `KARUME_REFERENCE=write` run — the practical preset's floor).
     */
    const settleCase = async (
      id: string,
      video: GeneratedVideo,
      beforeSettle?: (bytes: Uint8Array<ArrayBuffer>) => void,
    ) => {
      const bytes = rgbBytes(video);
      beforeSettle?.(bytes);
      return await settleOrObserve(references, results, { id, artifact: `${id}.rgb`, bytes });
    };
    const caseRatios: CaseRatios[] = [];
    beforeAll(async () => {
      gpuContext = verifyRunningGpu(
        await acquireTestGpu({
          onDeviceLost: (info) => {
            deviceLost = `${info.reason}: ${info.message}`;
          },
        }),
      );
      // The reference preset's one-layer-inside state (for the runs that observe the DiT inputs — module doc "Hooks for
      // observation and fault injection").
      innerState = await loadWanFromPretrained(
        WAN22_TI2V_FAMILY,
        nodeDirectory(WAN_TI2V_DIST_ROOT),
        {
          gpu: gpuContext,
          textEncoder: "precomputed",
          quant: WAN_TI2V_REFERENCE_QUANT,
          onRunDiagnostics,
        },
      );
      statePrompts = innerState.textEmbeds.entries;
      tokenSources = tokenSourcesOf(innerState.dit.patch);
    });
    afterAll(async () => {
      const gpu = gpuContext;
      if (gpu === undefined) return;
      // MUST: wait for the releases before discarding the device (B570's delayed destroy — leave budget for the
      // following tests). The one-layer-inside state did not acquire the GPU itself (`gpu` was passed in), so it holds
      // no resources to dispose.
      await settleReleases(gpu);
      gpu.destroy();
    });

    for (const { name, role, optIn } of CASES) {
      const id = bandCaseId(name);
      it.skipIf(optIn && !I2V_FULL)(`${id}（${role}${optIn ? ", opt-in" : ""}）`, async () => {
        const gpu = assigned(gpuContext, "gpu");
        const state = assigned(innerState, "state");
        const prompts = assigned(statePrompts, "prompts");
        const sources = assigned(tokenSources, "sources");
        const fixture = await readFixture(wanI2vStepsFixtureUrl(name));
        assertEquals(fixture.meta("role"), role, `${name}: role`);
        // band looks up the original text, accept the normalized string (both spellings of the accepted set are used —
        // the same as T2V).
        const prompt = textOf(
          prompts,
          fixture.meta("prompt"),
          role === "band" ? "prompt" : "normalized",
        );
        const latentsInit = fixture.tensor("latents_init");
        let settlement: ReferenceSettlement | undefined;
        let caseNote = "";
        try {
          await runRecordedCase(
            results,
            { id, failureNote: () => caseNote },
            async ({ measurements }) => {
              const record = emptyRecord();
              const observed = await observe(
                viaState(probedState(state, record, sources)),
                diagnostics,
                {
                  ...TWO_STEP,
                  prompt,
                  negativePrompt: textOf(prompts, fixture.meta("negative"), "prompt"),
                  latents: latentsInit,
                  // The size is omitted (1280×704 is chosen from the image's aspect ratio — the default use), and so is
                  // fit (the default crop).
                  image: sourceImageOf(fixture),
                },
              );
              assert("video" in observed, "generate did not run to the end");
              assertEquals(observed.latents.length, STEPS, "denoise-step count");
              const { video } = observed;
              assertEquals([video.frames, video.height, video.width], [FRAMES, HEIGHT, WIDTH]);
              assertEncoderStage(observed, id);
              const condition = assertDitWiring(record.dit, latentsInit, observed.latents, id);
              const last = observed.latents[STEPS - 1];
              assertVaeReceivesLatents(record, condition, last, fixture, id);
              // The VAE wiring check turns red on real data (0 GPU time). The matching compares values, so turning
              // red when the expectation is replaced with the value the VAE would have received under a fault is the
              // same as turning red when the VAE receives that value: the shape where the replacement after the loop is
              // lost (the first frame stays the scheduler state) and the shape that passes a copy of the last model
              // input (the frames not replaced are one state behind).
              assertThrows(
                () => assertVaeReceivesLatents(record, latentFrame(last, 0), last, fixture, id),
                Error,
                "first chunk (vae_decoder_first)",
              );
              assertThrows(
                () =>
                  assertVaeReceivesLatents(
                    record,
                    condition,
                    observed.latents[STEPS - 2],
                    fixture,
                    id,
                  ),
                Error,
                "following chunks (vae_decoder_next)",
              );
              const final = withWanFirstFrameCondition(
                observed.latents[STEPS - 1],
                condition,
                LATENT_SHAPE,
              );
              const diffs = OBSERVATION_POINTS.map((point) => ({
                point,
                band: bandOf(point),
                diff:
                  point === "frames"
                    ? // The reference frames are written after the clamp (the clamp of `ti2v_few_step_ref.decode_frames`).
                      difference(video.data, fixture.tensor("frames"))
                    : point === "latents_final"
                      ? difference(final, fixture.tensor("latents_final"))
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
                    // An underived band is recorded as an infinite band (the verdict below fails — the same as T2V).
                    tolerance:
                      band === undefined
                        ? { atol: Number.POSITIVE_INFINITY, rtol: 0 }
                        : { atol: band * diff.referenceMax, rtol: 0 },
                    stage: "karume",
                  });
                }
              }
              const conditionDiff = difference(condition, fixture.tensor("condition_latents"));
              const notes = [
                ...diffs.map(({ point, diff }) => formatDifference(point, diff)),
                ...(["latents.0", "latents.1"] as const).map(
                  (point, step) =>
                    `${point} per-frame maxAbs (observation) [${frameMaxAbs(
                      observed.latents[step],
                      fixture.tensor(point),
                    )}]`,
                ),
                `${formatDifference(
                  "condition latents",
                  conditionDiff,
                )}, encoder band ${WAN_I2V_ENCODER_BAND} × ${(
                  conditionDiff.maxAbs / WAN_I2V_ENCODER_BAND
                ).toFixed(2)}`,
                // Step 0's model input differs from the reference only by the condition-latent difference, so the DiT
                // forward's difference can be isolated here.
                ...(["noise_uncond.0", "noise_cond.0"] as const).map(
                  (key, pass) =>
                    `DiT output ${key} (observation, per-frame maxAbs [${frameMaxAbs(
                      record.ditOutputs[pass],
                      fixture.tensor(key),
                    )}]) ${formatDifference(
                      "",
                      difference(record.ditOutputs[pass], fixture.tensor(key)),
                    )}`,
                ),
                formatFirstFrame(firstFrameDifference(video, fixture.bytes("rgb8"))),
                ...(UPSTREAM_PARITY
                  ? []
                  : [
                      "bands and condition latents not judged (upstream parity is opt-in: " +
                        "KARUME_UPSTREAM_PARITY=1)",
                    ]),
                ...formatObserved(observed),
              ];
              caseNote = notes.join(" / ");
              console.log(`[wan-ti2v-i2v-pipeline] ${id}:\n  ${notes.join("\n  ")}`);
              for (const { point, diff } of diffs) {
                assertEquals(diff.nonFinite, 0, `${id}: ${point} non-finite values`);
              }
              // The condition check comes before the band-candidate material is stored (a band is not derived from the
              // ratios of a run with a broken encoder).
              if (UPSTREAM_PARITY) {
                assertConditionInBand(conditionDiff, id);
              } else {
                assertEquals(
                  conditionDiff.nonFinite,
                  0,
                  `${id}: condition latents non-finite values`,
                );
              }
              assertEquals(deviceLost, undefined, "device lost");
              caseRatios.push({
                name,
                role,
                ratios: {
                  "latents.0": diffs[0].diff.ratio,
                  "latents.1": diffs[1].diff.ratio,
                  latents_final: diffs[2].diff.ratio,
                  frames: diffs[3].diff.ratio,
                },
              });
              // MUST stay before the row is settled: with the opt-in, an output outside the band is never
              // written as a row under KARUME_REFERENCE=write.
              if (UPSTREAM_PARITY) {
                for (const { point, band, diff } of diffs) {
                  if (band === undefined) {
                    throw new Error(
                      `${id}: ${point} band not derived yet (ratio ${diff.ratio.toExponential(3)}) — ` +
                        'the "band candidate" step prints the candidate (run both deciding cases with the opt-in)',
                    );
                  }
                  assert(
                    diff.ratio <= band,
                    `${id}: ${formatDifference(point, diff)} is outside the band ${band}`,
                  );
                }
              }
              const outcome = await settleCase(id, video, (bytes) => {
                if (name === DEFAULT_CASE) referenceRgb = bytes;
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
    // not used; no candidate is printed when one of the deciding cases failed before reporting its ratios, or when the
    // second deciding case did not run without the opt-in). When the bands are derived, no step is created.
    if (!BANDS_DERIVED) {
      it.skipIf(!UPSTREAM_PARITY)(
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
              : `of the ${expected} deciding cases, only ${decided.length} reported ratios — no candidate` +
                " (run every deciding case with the opt-in KARUME_WAN_TI2V_I2V_FULL=1; investigate failed cases first)";
          const accepted = caseRatios
            .filter(({ role }) => role === "accept")
            .map(
              ({ name, ratios }) =>
                `${name} ${OBSERVATION_POINTS.map(
                  (point) => `${point} ${ratios[point].toExponential(3)}`,
                ).join(", ")}`,
            );
          const lines = caseRatios
            .filter(({ role }) => role === "band")
            .map(
              ({ name, ratios }) =>
                `${name} ${OBSERVATION_POINTS.map(
                  (point) => `${point} ${ratios[point].toExponential(3)}`,
                ).join(", ")}`,
            );
          throw new Error(
            `band not derived yet — ${candidate}. Deciding ratios: ${
              lines.join(" / ") || "none"
            }. Acceptance ratios (not used to set the band): ${accepted.join(" / ") || "none"}`,
          );
        },
      );
    }

    // Fault injection (the default case's initial noise — only the latents are checked, so it stops before the VAE
    // stage).
    const channels = WAN22_TI2V_FAMILY.generation.latents.mean.length;
    /**
     * The family values with the statistics set to mean 0 and std 1 (the normalization then returns mu as is — the
     * "not normalized" injection).
     */
    const unnormalized: WanFamilySpec = {
      ...WAN22_TI2V_FAMILY,
      generation: {
        ...WAN22_TI2V_FAMILY.generation,
        latents: {
          mean: Array.from({ length: channels }, () => 0),
          std: Array.from({ length: channels }, () => 1),
        },
      },
    };
    const faults: readonly {
      readonly label: string;
      /** Rewrites a DiT run's inputs (`current` is the current step's scheduler state — before the replacement). */
      readonly alter?: (inputs: RunInputs, current: Float32Array<ArrayBuffer>) => RunInputs;
      /** The family values (the product's when omitted). */
      readonly spec?: WanFamilySpec;
      /**
       * The check that must turn red by name besides the band (the DiT input wiring — a fragment of its message / the
       * condition-latent check). It shows on the same run's record, without adding GPU time, that the wiring and
       * condition checks turn red on real data.
       */
      readonly named:
        | { readonly kind: "wiring"; readonly fragment: string }
        | {
            readonly kind: "condition";
          };
    }[] = [
      {
        label: "condition mask all false",
        alter: (inputs) => {
          const mask = boolInput(inputs, DIT_CONDITION_MASK);
          return {
            ...inputs,
            [DIT_CONDITION_MASK]: { ...mask, data: new Uint32Array(mask.data.length) },
          };
        },
        named: { kind: "wiring", fragment: ": condition mask is not" },
      },
      {
        label: "condition mask shifted by one token",
        alter: (inputs) => {
          const mask = boolInput(inputs, DIT_CONDITION_MASK);
          const shifted = new Uint32Array(mask.data.length);
          shifted.set(mask.data.subarray(0, mask.data.length - 1), 1);
          return { ...inputs, [DIT_CONDITION_MASK]: { ...mask, data: shifted } };
        },
        named: { kind: "wiring", fragment: ": condition mask is not" },
      },
      {
        // The two timestep inputs bound the wrong way round (the generated tokens receive t = 0 and the condition
        // tokens the generated side's t). "Make the conditioning timestep the same as the generated side's" would give
        // the DiT the same values as "condition mask all false" (every token receives the generated side's timestep)
        // and repeat the same run on the band, so it is not included (the t = 0 comparison of the wiring check names
        // that fault).
        label: "the two timestep inputs swapped",
        alter: (inputs) => ({
          ...inputs,
          [DIT_TIMESTEPS_PROJ]: f32Input(inputs, DIT_TIMESTEPS_PROJ_CONDITION),
          [DIT_TIMESTEPS_PROJ_CONDITION]: f32Input(inputs, DIT_TIMESTEPS_PROJ),
        }),
        named: {
          kind: "wiring",
          fragment: ": conditioning timestep is not bit-identical to the proj at t = 0",
        },
      },
      {
        label: "first frame of the model input not replaced",
        alter: (inputs, current) => {
          const tokens = f32Input(inputs, DIT_TOKENS);
          return {
            ...inputs,
            [DIT_TOKENS]: {
              ...tokens,
              data: patchifyLatents(current, LATENT_SHAPE, assigned(innerState, "state").dit.patch),
            },
          };
        },
        named: { kind: "wiring", fragment: ": model input's first frame differs from run 0" },
      },
      {
        label: "condition latents not normalized",
        spec: unnormalized,
        named: { kind: "condition" },
      },
    ];
    const lastBand = LATENT_RATIO_BANDS[STEPS - 1];
    for (const { label, alter, spec, named } of faults) {
      it(`fault injection: ${label} → ${DEFAULT_CASE} latents.${STEPS - 1} outside the band`, async () => {
        const gpu = assigned(gpuContext, "gpu");
        const state = assigned(innerState, "state");
        const prompts = assigned(statePrompts, "prompts");
        const sources = assigned(tokenSources, "sources");
        try {
          const fixture = await defaultFixtureOf();
          const record = emptyRecord();
          let current: Float32Array<ArrayBuffer> = fixture.tensor("latents_init");
          const altered =
            alter === undefined
              ? undefined
              : (inputs: RunInputs): RunInputs => alter(inputs, current);
          const observed = await observe(
            viaState(probedState(state, record, sources, altered), spec),
            diagnostics,
            await defaultRequest(prompts),
            {
              abortAfterDenoise: true,
              onLatents: (latents) => {
                current = latents;
              },
            },
          );
          // If a stage event name changes and the abort stops working, the 28-tile VAE is paid silently (it does not
          // show in the value verdict).
          assert(!("video" in observed), "did not stop after the DiT stage");
          assertEquals(observed.latents.length, STEPS, "denoise-step count");
          // The rewrite applied to every run (the hook is not vacuous).
          assertEquals(record.dit.length, STEPS * 2, "DiT run count");
          const diff = difference(
            observed.latents[STEPS - 1],
            fixture.tensor(`latents.${STEPS - 1}`),
          );
          console.log(
            `[wan-ti2v-i2v-pipeline] fault ${label}: ${formatDifference(
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
          // The named check also turns red on the same record (the hook that points at the stage or wiring at fault
          // before the band works).
          if (named.kind === "wiring") {
            assertThrows(
              () =>
                assertDitWiring(
                  record.dit,
                  fixture.tensor("latents_init"),
                  observed.latents,
                  label,
                ),
              Error,
              named.fragment,
            );
          } else {
            assertThrows(
              () =>
                assertConditionInBand(
                  difference(
                    latentFrame(record.dit[0].modelInput, 0),
                    fixture.tensor("condition_latents"),
                  ),
                  label,
                ),
              Error,
              "is outside the encoder band",
            );
          }
        } finally {
          await settleReleases(gpu);
        }
      });
    }

    // Portrait (opt-in, reference preset, public surface — the landscape source image cropped to an explicit 704×1280,
    // seed path).
    it.skipIf(!I2V_FULL)(
      `${PORTRAIT_CASE_ID} (portrait, public surface, sha256 per-environment row, opt-in)`,
      async () => {
        const gpu = assigned(gpuContext, "gpu");
        let settlement: ReferenceSettlement | undefined;
        try {
          await using pipeline = await Ti2vPipeline.fromPretrained(
            nodeDirectory(WAN_TI2V_DIST_ROOT),
            {
              gpu,
              textEncoder: "precomputed",
              quant: WAN_TI2V_REFERENCE_QUANT,
              onRunDiagnostics,
            },
          );
          const fixture = await defaultFixtureOf();
          await runRecordedCase(results, { id: PORTRAIT_CASE_ID }, async () => {
            const observed = await observe(viaPipeline(pipeline), diagnostics, {
              ...TWO_STEP,
              prompt: textOf(pipeline.prompts, SEED_IMAGE, "prompt"),
              seed: SEED,
              image: sourceImageOf(fixture),
              width: HEIGHT,
              height: WIDTH,
              fit: "crop",
            });
            assert("video" in observed, "generate did not run to the end");
            assertEquals(observed.latents.length, STEPS, "denoise-step count");
            const { video } = observed;
            assertEquals([video.frames, video.height, video.width], [FRAMES, WIDTH, HEIGHT]);
            assertEncoderStage(observed, PORTRAIT_CASE_ID);
            const nonFinite = countNonFinite(video.data);
            const notes = [`non-finite ${nonFinite}`, ...formatObserved(observed)];
            console.log(`[wan-ti2v-i2v-pipeline] ${PORTRAIT_CASE_ID}:\n  ${notes.join("\n  ")}`);
            assertEquals(nonFinite, 0, `${PORTRAIT_CASE_ID}: non-finite values`);
            assertEquals(deviceLost, undefined, "device lost");
            const outcome = await settleCase(PORTRAIT_CASE_ID, video);
            settlement = outcome.settlement;
            return { ...outcome.fields, note: notes.join(" / ") };
          });
        } finally {
          await settleReleases(gpu);
        }
        if (settlement?.check.status === "fail") {
          throw new Error(referenceMismatchMessage(PORTRAIT_CASE_ID, settlement, references));
        }
      },
    );

    // The practical preset (public surface — the same request as the default case, only the preset differs).
    it(`${PRACTICAL_CASE_ID} (practical preset, public surface, sha256 per-environment row)`, async () => {
      const gpu = assigned(gpuContext, "gpu");
      let settlement: ReferenceSettlement | undefined;
      try {
        await using practical = await Ti2vPipeline.fromPretrained(
          nodeDirectory(WAN_TI2V_DIST_ROOT),
          {
            gpu,
            textEncoder: "precomputed",
            quant: WAN_TI2V_PRACTICAL_QUANT,
            onRunDiagnostics,
          },
        );
        const request = await defaultRequest(practical.prompts);
        await runRecordedCase(results, { id: PRACTICAL_CASE_ID }, async () => {
          const observed = await observe(viaPipeline(practical), diagnostics, request);
          assert("video" in observed, "generate did not run to the end");
          assertEquals(observed.latents.length, STEPS, "denoise-step count");
          const { video } = observed;
          assertEquals([video.frames, video.height, video.width], [FRAMES, HEIGHT, WIDTH]);
          assertEncoderStage(observed, PRACTICAL_CASE_ID);
          const nonFinite = countNonFinite(video.data);
          const fixture = await defaultFixtureOf();
          const notes = [
            `non-finite ${nonFinite}`,
            formatFirstFrame(firstFrameDifference(video, fixture.bytes("rgb8"))),
            ...formatObserved(observed),
          ];
          console.log(`[wan-ti2v-i2v-pipeline] ${PRACTICAL_CASE_ID}:\n  ${notes.join("\n  ")}`);
          assertEquals(nonFinite, 0, `${PRACTICAL_CASE_ID}: non-finite values`);
          assertEquals(deviceLost, undefined, "device lost");
          const outcome = await settleCase(PRACTICAL_CASE_ID, video, (bytes) => {
            // The floor (ADR 0110 decision 5 — a practical preset differs by at least 1 bit from the reference preset
            // for the same request): in a regression where the preset's `session` does not reach the Session, the
            // practical preset produces the same bytes as the reference preset. Fail before writing the row, so that
            // artifact is not frozen as a practical row.
            if (referenceRgb === undefined) {
              throw new Error(
                `${PRACTICAL_CASE_ID}: no reference-preset artifact for the same request (the reference preset's ${DEFAULT_CASE} did not run — ` +
                  "the floor cannot be checked)",
              );
            }
            const reference = referenceRgb;
            assert(
              bytes.length !== reference.length ||
                bytes.some((value, index) => value !== reference[index]),
              `${PRACTICAL_CASE_ID}: the practical preset's output is byte-identical to the reference preset's for the same request (the preset's session did not arrive)`,
            );
          });
          settlement = outcome.settlement;
          return { ...outcome.fields, note: notes.join(" / ") };
        });
      } finally {
        await settleReleases(gpu);
      }
      if (settlement?.check.status === "fail") {
        throw new Error(referenceMismatchMessage(PRACTICAL_CASE_ID, settlement, references));
      }
    });
  },
);

// NOTE: run with the opt-in env and without `-t`, one process takes two devices in turn together with the
// two-step `describe` (hitting B570's `destroy()` not returning VRAM). Run it as a separate process with
// `-t "50 steps"`.
describe.skipIf(!I2V_FULL || !DIST_PRESENT || !GPU_AVAILABLE || !ANY_FIXTURE)(
  "Wan2.2 I2V end-to-end 50 steps (real GPU, opt-in KARUME_WAN_TI2V_I2V_FULL=1): 1280×704, 33 frames (reference and practical preset)" +
    " complete with 0 non-finite values; time and VRAM at stage switches, PNG of every frame + contact sheet, sha256 per-environment rows",
  () => {
    let deviceLost: string | undefined;
    // One device and the default case's fixture shared by the its below (released after them all).
    let gpuContext: GpuContext | undefined;
    let defaultFixture: Fixture | undefined;
    const diagnostics = new Map<RunComponent, SessionDiagnostics>();
    beforeAll(async () => {
      gpuContext = verifyRunningGpu(
        await acquireTestGpu({
          onDeviceLost: (info) => {
            deviceLost = `${info.reason}: ${info.message}`;
          },
        }),
      );
      defaultFixture = await readFixture(wanI2vStepsFixtureUrl(DEFAULT_CASE));
    });
    afterAll(async () => {
      const gpu = gpuContext;
      if (gpu === undefined) return;
      await settleReleases(gpu);
      gpu.destroy();
    });
    /**
     * The sha256 of the reference preset's artifact (RGB bytes) — the partner of the practical preset's floor for the
     * same request (the same floor as the two-step practical row).
     */
    let referenceRgbSha: string | undefined;
    for (const { id, quant } of FULL_CASES) {
      it(id, async () => {
        const gpu = assigned(gpuContext, "gpu");
        const fixture = assigned(defaultFixture, "fixture");
        let settlement: ReferenceSettlement | undefined;
        try {
          await using pipeline = await Ti2vPipeline.fromPretrained(
            nodeDirectory(WAN_TI2V_DIST_ROOT),
            {
              gpu,
              textEncoder: "precomputed",
              quant,
              onRunDiagnostics: (component, diagnosed) => diagnostics.set(component, diagnosed),
            },
          );
          await runRecordedCase(fullResults, { id }, async () => {
            const observed = await observe(viaPipeline(pipeline), diagnostics, {
              prompt: textOf(pipeline.prompts, SEED_IMAGE, "prompt"),
              seed: SEED,
              shift: SHIFT,
              frames: FULL_FRAMES,
              image: sourceImageOf(fixture),
            });
            assert("video" in observed, "generate did not run to the end");
            assertEquals(
              observed.latents.length,
              FULL_STEPS,
              "denoise-step count (the manifest's default)",
            );
            const { video } = observed;
            assertEquals([video.frames, video.height, video.width], [FULL_FRAMES, HEIGHT, WIDTH]);
            assertEncoderStage(observed, id);
            const nonFinite = countNonFinite(video.data);
            const notes = [
              `preset ${quant}`,
              `non-finite ${nonFinite}`,
              formatFirstFrame(firstFrameDifference(video, fixture.bytes("rgb8"))),
              ...formatObserved(observed),
            ];
            console.log(`[wan-ti2v-i2v-pipeline] ${id}:\n  ${notes.join("\n  ")}`);
            assertEquals(nonFinite, 0, "non-finite values");
            assertEquals(deviceLost, undefined, "device lost");
            await writeWanFrames(fullResults, id, video);
            const sheet = contactSheet(video, 4, 8, 4);
            await writeFile(
              fullResults.artifact(`${id}-sheet.png`),
              await encodePng(sheet.rgba, sheet.width, sheet.height),
            );
            console.log(`[wan-ti2v-i2v-pipeline] PNG: ${fullResults.dir.pathname}`);
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

/**
 * The cases registered with the reference guard (only those that can run — every case needs the references (the
 * condition image is also read from them) and the distribution). The opt-in cases (the remaining two-step cases, the
 * portrait case, the two presets of the 50 steps) register only under the opt-in (they do not run in the default lane,
 * so registering them would turn the reference guard red on a machine without the rows).
 */
const CASE_IDS =
  DIST_PRESENT && ANY_FIXTURE
    ? [
        ...CASES.filter(({ optIn }) => !optIn || I2V_FULL).map(({ name }) => bandCaseId(name)),
        PRACTICAL_CASE_ID,
        ...(I2V_FULL ? [PORTRAIT_CASE_ID, ...FULL_CASES.map(({ id }) => id)] : []),
      ]
    : [];
const RUNNABLE = DIST_PRESENT && GPU_AVAILABLE && ANY_FIXTURE;
if (RUNNABLE) references.warnMissing(CASE_IDS);
registerReferenceGate(references, { runnable: RUNNABLE, caseIds: CASE_IDS });
