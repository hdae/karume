/**
 * Wan2.2 TI2V-5B **self A/B check** (real GPU; ADR 0121 step 7 — the rules keep the shape of ADR 0110 decisions 4–7 and
 * ADR 0120 decision 4). It compares the practical preset `f16+dit8-a8-attn8-s16` (the default preset — a preset whose
 * `session` bundles linear a8 / attention a8 / f16 storage of S), **on the same machine with the same i8 weights**,
 * against the reference partner whose `session` is empty (derived from the manifest — `referencePartnerOf`; `f16+dit8`
 * in the current distribution). The comparison partner is the same machine's reference path; no torch golden is used
 * (ADR 0110 decision 5 ②). Two practical presets, s16 and online (`f16+dit8-a8-attn8-online` — ADR 0132), are checked
 * under the same band (a describe for each of the 2 presets per T2V case; the fault injections and I2V run for s16
 * only).
 *
 * The steps, the verdict and the fault injections are the one implementation shared with 2.1 (`e2e_wan_ab_test.ts`)
 * ({@link registerWanAbCase} of `helpers/wan-ab-gate.ts`): capture the step-1 latent (step 1 of a 2-step run) with the
 * reference partner and abort → do the same twice independently with the practical preset → determinism (the 2 runs
 * are bit-identical), census (in every pass of the practical preset's DiT exactly 310 linears are i8a8 and 0 linears
 * are on the reference path — the `wan-ti2v` row of the bundle's census table; at least 1 for each of the 2 attention
 * knobs), band (`judgeAb`).
 *
 * The input is `boxing-cats`, seed 42, 2 steps (step 1 = the 2 CFG passes), 1280×704, guidance 5, shift 5, precomputed
 * embeddings (`"precomputed"`) — the same request as the sha row of the seed path of the end-to-end e2e (`TWO_STEP` of
 * `e2e_wan_ti2v_pipeline_test.ts`). Dimensions and guidance / shift are explicit (not silently left to the generation
 * defaults — so that the band's premises (the σ of step 1 and the number of CFG passes) do not move when the
 * manifest defaults move).
 *
 * ## Cases (the user's ruling, 2026-10-09)
 *
 * In the shape of ADR 0120 decision 4 (two points, the default length and the longest — the attention a8 error grows
 * with the row length, so the band of the shorter one is not extrapolated to the longer one), the 2 points of 2.2 are:
 *
 * - **33 frames** (the default frame count, S = 7,920 — default lanes). The 5 fault injections also run in this case
 *   (showing a check that misses needs no length).
 * - **121 frames** (the accepted maximum, S = 27,280 — only with the env opt-in `KARUME_WAN_TI2V_121F=1`; the same
 *   variable as the 121-frame case of the end-to-end e2e). It stays out of the default lanes so as not to lengthen the
 *   full verify (a countermeasure is to be considered separately).
 *
 * Command that runs the 121 frames (as a **process of its own** through `--filter` — "Device use" below):
 *
 * ```
 * KARUME_WAN_TI2V_121F=1 deno test -A --v8-flags=--expose-gc packages/models/tests/e2e_wan_ti2v_ab_test.ts --filter "121f"
 * ```
 *
 * ## The I2V case (ADR 0121 step 9d; the user's rulings, 2026-10-09 to 10)
 *
 * The numeric check of the practical preset for generation from a condition image (I2V). The request is the 33-frame
 * case plus a condition image and a fit (1280×704, 33 frames, guidance 5, shift 5, seed 42, `boxing-cats`, precomputed
 * embeddings). The condition image is the `boxing-cats` source image of the default case of the I2V end-to-end e2e
 * (`e2e_wan_ti2v_i2v_pipeline_test.ts`): the RGB8 (`source` 832×480 — relying neither on PNG decoding nor on another
 * asset) embedded in the CPU reference (`pipeline_steps_i2v.band-boxing-cats.safetensors`) is read from that same
 * reference. The fit is an explicit `"crop"` (not silently left to the product default — so that the band's premises do
 * not move when the default moves). The dimensions are explicit, so the choice is not left to the image's aspect ratio.
 * Environments without the reference **SKIP explicitly** with the reason and the generation command.
 *
 * It runs only with the env opt-in `KARUME_WAN_TI2V_I2V_FULL=1` (the same variable as the opt-in of the I2V end-to-end
 * e2e) (kept out of the default lanes — the user's ruling). No fault injections run (the T2V 33 frames show that the
 * check is not blind — the steps are the same implementation). Census and determinism are those of
 * {@link registerWanAbCase}. In addition, {@link registerWanAbCase} checks in every observation of a case given a
 * condition image that the runs of the encoder's 3 graphs arrived (if the image fell out of the request and both
 * presets ran as T2V, relRMS would take the T2V value and stay inside the band, and the census looks only at the DiT,
 * so neither the band nor the census would catch it). Command that runs it:
 *
 * ```
 * KARUME_WAN_TI2V_I2V_FULL=1 deno test -A --v8-flags=--expose-gc packages/models/tests/e2e_wan_ti2v_ab_test.ts --filter "i2v"
 * ```
 *
 * There is no per-knob record (2.1's `knobRungs`) — it is not in the acceptance conditions of step 7, and it would
 * lengthen the default lanes. The final output (frames) is not produced here — the sha rows
 * (`e2e_wan_ti2v_pipeline_test.ts` / `e2e_wan_ti2v_i2v_pipeline_test.ts`) own it, and it gets no band (ADR 0110
 * decision 5-4). The results directories are `wan-ti2v-ab` for the 33 frames of the default lanes, `wan-ti2v-ab-121f`
 * for the opt-in 121 frames and `wan-ti2v-ab-i2v` for the opt-in I2V (`results.json` is rewritten whole on every run,
 * so they are separated to keep an opt-in case run in another process from erasing the same day's 33-frame record —
 * the same treatment as the 121 frames of the end-to-end e2e).
 *
 * ## Device use
 *
 * One device is acquired per describe, and the releases are awaited with `settleReleases` between observations (as in
 * 2.1). Run with the opt-in env and without `--filter`, one process acquires devices in turn (121 → 33 → I2V). On a
 * machine where `destroy()` does not return VRAM at once (B570 — ADR 0120 risk 6), what is left of the previous case's
 * allocations can overlap the next case's, so the opt-in cases are run alone with the commands above (the same
 * discipline as the opt-ins of the end-to-end e2e). The 121 frames come first so that, even when run without
 * `--filter`, the large allocations are made without carrying what other cases left on the device.
 *
 * MUST: the assets are `models/karume-wan2.2/` (untracked; local assets of a real-GPU machine). Environments without
 * them or without a GPU **SKIP explicitly** with the reason (the assembly command) (ADR 0005). The fault-injection
 * presets are built from a temporary manifest that borrows the same distribution through symlinks; the original
 * distribution is not changed by a single byte.
 */

import { readFile } from "@karume/test-support/fs";
import { afterAll, beforeAll, describe } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import { type GpuContext, parseSafetensors } from "@karume/runtime";
import { type Rgb8Image, Ti2vPipeline } from "../../models/wan.ts";
import { assertAdapterMatchesEnvironment } from "../../runtime/tests/helpers/environment.ts";
import { sha256Hex } from "../../runtime/tests/helpers/reference.ts";
import { openResults, type Results } from "../../runtime/tests/helpers/results.ts";
import type { AbBand } from "./helpers/ab-gate.ts";
import { acquireTestGpu, GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import { settleReleases } from "../../models/tests/helpers/settle-releases.ts";
import {
  assigned,
  registerWanAbCase,
  type WanAbScope,
  type WanAbSubject,
} from "./helpers/wan-ab-gate.ts";
import { WAN_I2V_IMAGE_SHA256 } from "../../models/tests/helpers/wan-i2v-image.ts";
import {
  WAN_I2V_SOURCE_SHAPE,
  wanI2vStepsFixtureUrl,
} from "../../models/tests/helpers/wan-i2v-fixture.ts";
import { filePresent, TEST_ASSETS_FETCH } from "../../models/tests/helpers/wan-ti2v-dit.ts";
import {
  readWanTi2vDistributionManifestText,
  WAN_TI2V_ASSEMBLE_COMMAND,
  WAN_TI2V_DIST_MODEL,
  WAN_TI2V_DIST_ROOT,
  WAN_TI2V_ONLINE_QUANT,
  WAN_TI2V_PRACTICAL_QUANT,
} from "../../models/tests/helpers/wan-ti2v-pipeline.ts";

/** Where measurements and outcomes go (`outputs/verify/<environment key>/<date>_wan-ti2v-ab/` — safe to delete). */
const results = openResults("wan-ti2v-ab");
/**
 * Where the opt-in 121 frames go (`<date>_wan-ti2v-ab-121f/` — why it is separate from the default lanes: module doc).
 */
const longClipResults = openResults("wan-ti2v-ab-121f");
/** Where the opt-in I2V goes (`<date>_wan-ti2v-ab-i2v/` — why it is separate from the default lanes: module doc). */
const i2vResults = openResults("wan-ti2v-ab-i2v");

/**
 * Fixed request values (the landscape shape of the accepted set; guidance / shift are the same values as `TWO_STEP` of
 * the end-to-end e2e — the values equal to the current manifest defaults are also given explicitly).
 */
const REQUEST = { width: 1280, height: 704, guidance: 5, shift: 5 } as const;

/** Opt-in for the 121-frame case (the same variable as the 121-frame case of `e2e_wan_ti2v_pipeline_test.ts`). */
const LONG_CLIP = process.env.KARUME_WAN_TI2V_121F === "1";

/** Opt-in for the I2V case (the same variable as the opt-in of `e2e_wan_ti2v_i2v_pipeline_test.ts`). */
const I2V_FULL = process.env.KARUME_WAN_TI2V_I2V_FULL === "1";

/**
 * Name of the I2V case's condition image (the same name as the prompt — the source image of the default case of the I2V
 * end-to-end e2e).
 */
const I2V_IMAGE = "boxing-cats";

/**
 * The CPU reference that embeds the I2V condition image (the default case `band-boxing-cats` of the I2V end-to-end e2e
 * — the location, the generation command and the source-image shape are in `helpers/wan-i2v-fixture.ts`, shared with
 * that e2e).
 */
const I2V_FIXTURE_URL = wanI2vStepsFixtureUrl(`band-${I2V_IMAGE}`);

/**
 * The band of the I2V case (derived independently under the same rule as {@link CASES} — condition tokens are added, so
 * the 33-frame T2V band is not reused; the measurement and the derivation are the `i2v-33f` row of the table of
 * {@link CASES}).
 *
 * MUST: `undefined` before derivation (the same as the MUST of {@link CASES}).
 */
const I2V_BAND: AbBand | undefined = { floor: 0, ceiling: 9.3e-2 };

/** Frame count of the I2V case (the request of the 33-frame case plus a condition image and a fit). */
const I2V_FRAMES = 33;

/**
 * Cases (frame counts) and bands. The 121 frames, which allocate the most, come first (module doc).
 *
 * The bands are **declarations**: they are not rows per environment key and are not rewritten by `KARUME_REFERENCE`
 * (ADR 0110 decision 4). Derivation rule (decision 5): ceiling = about 2 × the measured relRMS at the observation point
 * (rounded up to 2 significant digits), derived independently per case (the 33-frame value is not extrapolated to 121
 * frames — the attention a8 error grows with the row length S; ADR 0120 decision 4). Floor = at least 1 bit different
 * (MUST — `judgeAb` always checks it, regardless of the band); `floor` is set above 0 only when the measurement with
 * one knob removed is shown to be clearly smaller than with all of them. The 2.1 bands (33 frames 2.2e-1 / 81 frames
 * 4.2e-1) are no basis for 2.2 (the model, the row length and the latent shape differ).
 *
 * MUST: keep it `undefined` before derivation — `judgeAb` prints the measurement and stops red (ADR 0120 decision 4).
 * Do not add a check without detection power by putting in a value that passes (the MUST of `ab-gate.ts`). While the
 * band is `undefined`, fault injection ⑤ (another seed's latent is red at the ceiling) is also red with "band not yet
 * derived" (it becomes effective once the band is declared — the same behaviour as 2.1 before derivation).
 *
 * | Case    | S      | relRMS    | maxAbs    | floor | ceiling              | Measured on                    |
 * | ------- | ------ | --------- | --------- | ----- | -------------------- | ------------------------------ |
 * | 121f    | 27,280 | 4.9151e-2 | 2.1226e-1 | 0     | 9.9e-2 (× 2, ceiled) | 2026-10-09, RTX 3080 Ti (Deno) |
 * | 33f     | 7,920  | 4.3881e-2 | 1.9291e-1 | 0     | 8.8e-2 (× 2, ceiled) | 2026-10-09, RTX 3080 Ti (Deno) |
 * | i2v-33f | 7,920  | 4.6146e-2 | 4.9261e-1 | 0     | 9.3e-2 (× 2, ceiled) | 2026-10-10, RTX 3080 Ti (Deno) |
 *
 * `i2v-33f` is the I2V case ({@link I2V_BAND} — the 33-frame request plus the condition image `boxing-cats` and the fit
 * `"crop"`; opt-in). Compared with the T2V 33 frames, relRMS is 1.05× (nearly the same) and maxAbs 2.6×. The maxAbs
 * sits in latent frame 0 (the observation point is the state before the condition replaces it, so a frame that never
 * reaches the output): frame 0 has relRMS 6.6217e-2 and maxAbs 4.9261e-1; frames 1–8 together have relRMS 4.1754e-2
 * and a per-frame maxAbs of at most 2.7906e-1 (2026-10-10, the same machine, a one-off diagnostic measurement — not
 * kept in the test). The band is a declaration on relRMS, so its ceiling is of the same order as T2V.
 *
 * 33-frame fault injections (recorded): ③ (the preset without attention a8) has relRMS 4.0983e-2, just over 90% of all
 * knobs — at 2.2's 33 frames the contribution of attention a8 is small (the rest is linear a8 and the f16 storage of S
 * — the breakdown was not measured; in 2.1, conversely, attention a8 is most of it — at 33 frames linear a8 alone is
 * 2.9991e-2 → all knobs 1.0676e-1).
 * ③ stays inside the band, which demonstrates that the census is needed. ⑤ (another seed) is 1.3782e+0 (a stand-in for
 * collapse).
 */
const CASES: readonly {
  readonly frames: number;
  readonly band: AbBand | undefined;
  /** Whether the case runs only with the env opt-in (`KARUME_WAN_TI2V_121F=1`). */
  readonly optIn: boolean;
}[] = [
  { frames: 121, band: { floor: 0, ceiling: 9.9e-2 }, optIn: true },
  { frames: 33, band: { floor: 0, ceiling: 8.8e-2 }, optIn: false },
];

/**
 * Frame count the fault injections run at (the 33 of the default lanes — showing a check that misses needs no length).
 */
const FAULT_FRAMES = 33;

/**
 * The practical presets the T2V cases run, each against the same reference preset and under the same band: the s16
 * preset, and the online-attention preset (ADR 0132 decision 6 — it must stay inside the band derived for the a8
 * preset, so the band is not re-derived for it). The case id carries the preset name, so the two presets never share
 * a row of the results.
 *
 * The fault injections run for the s16 preset only ({@link FAULT_PRESET}): they show that the gate's steps are not
 * blind, which the shared steps of {@link registerWanAbCase} establish once per machine whatever the preset; and
 * fault ③ (dropping `attentionCompute`) has no meaning for the online preset — `attentionScoreStorage: "none"`
 * without `attentionCompute: "a8"` is rejected at Session construction (ADR 0132 decision 1), so it would fail before
 * the census it is meant to exercise.
 */
const T2V_PRESETS: readonly string[] = [WAN_TI2V_PRACTICAL_QUANT, WAN_TI2V_ONLINE_QUANT];

/** The preset the fault injections run for (see {@link T2V_PRESETS}). */
const FAULT_PRESET = WAN_TI2V_PRACTICAL_QUANT;

const manifestText = await readWanTi2vDistributionManifestText();
if (manifestText === undefined) {
  console.warn(
    `[karume] ${WAN_TI2V_DIST_ROOT.pathname} has no karume.json, so the Wan2.2 self A/B check is SKIPPED` +
      ` (assemble: ${WAN_TI2V_ASSEMBLE_COMMAND})`,
  );
}
const RUNNABLE = GPU_AVAILABLE && manifestText !== undefined;

/**
 * The subject of the 2.2 check (passed to the shared {@link registerWanAbCase}). `caseResults` is the case's results
 * directory (the opt-in cases are kept apart from the default lanes — module doc), `request` the fixed request values
 * (the I2V case adds a condition image and a fit).
 *
 * MUST: call it only inside the tests of {@link RUNNABLE} (reaching it without a distribution means the skip decision
 * broke, so it fails).
 */
const subjectOf = (
  practical: string,
  caseResults: Results,
  request: NonNullable<WanAbSubject["request"]> = REQUEST,
): WanAbSubject => {
  if (manifestText === undefined) {
    throw new Error(
      `${WAN_TI2V_DIST_ROOT.pathname} has no karume.json, yet the A/B check ran (the skip decision broke)`,
    );
  }
  return {
    label: "wan-ti2v",
    family: "wan-ti2v",
    model: WAN_TI2V_DIST_MODEL,
    practical,
    assetsRoot: WAN_TI2V_DIST_ROOT,
    manifestText,
    results: caseResults,
    fromPretrained: (source, options) => Ti2vPipeline.fromPretrained(source, options),
    request,
  };
};

const I2V_FIXTURE_PRESENT = filePresent(I2V_FIXTURE_URL);
if (RUNNABLE && I2V_FULL && !I2V_FIXTURE_PRESENT) {
  console.warn(
    `[karume] the golden ${I2V_FIXTURE_URL.pathname} is absent, so Wan2.2's I2V A/B gate on this machine is ` +
      `SKIPPED (the condition image is read from this reference). Fetch it: ${TEST_ASSETS_FETCH}`,
  );
}

/**
 * The I2V condition image (RGB8 HWC of the source image embedded in the reference — the same value as `sourceImageOf`
 * of the I2V end-to-end e2e). Its provenance (image name, PNG sha256, RGB8 sha256) is compared with the reference's
 * metadata before it is passed on (so that the band is never measured on another image).
 */
const readI2vImage = async (): Promise<Rgb8Image> => {
  const bytes = await readFile(I2V_FIXTURE_URL);
  const file = parseSafetensors(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
  const where = I2V_FIXTURE_URL.pathname;
  const meta = (key: string): string | undefined => file.metadata.get(key);
  assertEquals(meta("image"), I2V_IMAGE, `${where}: image name`);
  assertEquals(meta("image_sha256"), WAN_I2V_IMAGE_SHA256[I2V_IMAGE], `${where}: image sha256`);
  const source = file.tensors.get("source");
  if (source === undefined || source.dtype !== "U8") {
    throw new Error(`${where}: 'source' is not U8`);
  }
  assertEquals(source.shape, WAN_I2V_SOURCE_SHAPE, `${where}: source image shape`);
  const data = new Uint8Array(file.buffer, source.byteOffset, source.byteLength);
  assertEquals(
    await sha256Hex(data),
    meta("source_rgb8_sha256"),
    `${where}: source image RGB8 sha256`,
  );
  return { width: WAN_I2V_SOURCE_SHAPE[1], height: WAN_I2V_SOURCE_SHAPE[0], data };
};

for (const { frames, band, optIn } of CASES) {
  for (const practical of T2V_PRESETS) {
    const id = `${practical}-${frames}f`;
    const faults = frames === FAULT_FRAMES && practical === FAULT_PRESET;
    describe.skipIf(!RUNNABLE || (optIn && !LONG_CLIP))(
      `e2e (real GPU): Wan2.2 ${id} step-1 latent agrees with the reference partner in determinism, census and band (self A/B)` +
        (faults ? ", 5 fault injections are red" : "") +
        (optIn ? "（opt-in KARUME_WAN_TI2V_121F=1）" : ""),
      () => {
        // One device per case, shared by the case's its (acquired before, released after them all).
        let gpu: GpuContext | undefined;
        let subject: WanAbSubject | undefined;
        beforeAll(async () => {
          gpu = await acquireTestGpu();
          assertAdapterMatchesEnvironment(gpu);
          subject = subjectOf(practical, optIn ? longClipResults : results);
        });
        afterAll(async () => {
          if (gpu === undefined) return;
          await settleReleases(gpu);
          gpu.destroy();
        });
        const scope: WanAbScope = {
          gpu: () => assigned(gpu, "gpu"),
          subject: () => assigned(subject, "subject"),
        };
        registerWanAbCase(scope, { id, frames, band, faults });
      },
    );
  }
}

// The I2V case runs for the s16 preset only: the online preset's acceptance (ADR 0132 decision 6) is measured on
// text to video, and the I2V band was derived for the s16 preset.
{
  const id = `${WAN_TI2V_PRACTICAL_QUANT}-i2v-crop-${I2V_FRAMES}f`;
  describe.skipIf(!RUNNABLE || !I2V_FULL || !I2V_FIXTURE_PRESENT)(
    `e2e (real GPU): Wan2.2 ${id} step-1 latent agrees with the reference partner in determinism, census and band` +
      " (self A/B, condition image boxing-cats, opt-in KARUME_WAN_TI2V_I2V_FULL=1)",
    () => {
      let gpu: GpuContext | undefined;
      let subject: WanAbSubject | undefined;
      beforeAll(async () => {
        const image = await readI2vImage();
        gpu = await acquireTestGpu();
        assertAdapterMatchesEnvironment(gpu);
        subject = subjectOf(WAN_TI2V_PRACTICAL_QUANT, i2vResults, {
          ...REQUEST,
          image,
          fit: "crop",
        });
      });
      afterAll(async () => {
        if (gpu === undefined) return;
        await settleReleases(gpu);
        gpu.destroy();
      });
      const scope: WanAbScope = {
        gpu: () => assigned(gpu, "gpu"),
        subject: () => assigned(subject, "subject"),
      };
      registerWanAbCase(scope, { id, frames: I2V_FRAMES, band: I2V_BAND, faults: false });
    },
  );
}
