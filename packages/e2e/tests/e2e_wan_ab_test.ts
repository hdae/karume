/**
 * Wan2.1 **self A/B check** (real GPU; ADR 0110 decision 5 / ADR 0120 decisions 4 / 5; the per-knob
 * record of steps 4 and 3). It compares the practical preset `f16+dit8-a8-attn8-s16` (a preset
 * whose `session` bundles linear a8 / attention a8 / f16 storage of S), **on the same machine with
 * the same i8 weights**, against the reference partner whose `session` is empty (derived from the
 * manifest — `referencePartnerOf`; `f16+dit8` in the current distribution). The comparison partner
 * is the same machine's reference path; no torch golden is used (the `f16` preset has different
 * weights, so the floor would be trivially true — ADR 0110 decision 5 ②).
 *
 * Steps of one case (= one frame count) (the same shape as `e2e_anima_ab_test.ts`; the body is
 * {@link registerWanAbCase} of `helpers/wan-ab-gate.ts`, shared with 2.2):
 *
 * 1. With the reference partner's pipeline, capture the **step-1 latent** (step 1 of a 2-step run —
 *    right after the preset first takes effect; decision 5-4) and stop the generation by throwing
 *    from the subscriber (`captureThenAbort`). Dispose it and wait for the release.
 * 2. Do the same **twice** with the practical preset (2 independent pipelines).
 * 3. Apply 3 checks:
 *    - **Determinism** (the MUST of decision 6) — the practical preset's 2 latents are
 *      bit-identical (`assertBitIdentical`).
 *    - **census** (decision 5-1) — in every pass of the practical preset's DiT exactly 307 linears
 *      are i8a8 and 0 linears are on the reference path (the bundle's row of the census table —
 *      `assertRowCensus`). The 2 attention knobs are by convention not written in the table (the
 *      number of row blocks changes with the device's binding limit), so at least 1
 *      (`assertSeatsApplied`).
 *    - **Band** (decisions 5-2 / 5-3) — relRMS within the ceiling of {@link CASES}, no non-finite
 *      values, at least 1 bit different from the reference partner (`judgeAb`).
 *
 * The 33-frame case then runs 5 fault injections (each asserts that "the check turns red" —
 * decision 5-5) and the per-knob record (step 3, {@link knobRungs}). The 81-frame per-knob record
 * is opt-in (`KARUME_WAN_FULL_PIPELINE=1`).
 *
 * The final output (frames) is not produced here — the sha rows (the reference / practical rows of
 * `e2e_wan_pipeline_test.ts`) own it, and it gets no band (decision 5-4). The results directory is
 * `wan-ab`.
 *
 * VRAM: pipelines are built one at a time (never at once — construction builds no Session, and the
 * DiT Session is torn down by the step-1 abort). On the B570 the release of `destroy()` lags (the
 * result of ADR 0118 step 8; ADR 0120 risk 6), so each observation waits for the release with
 * {@link settleReleases}, and the 81 frames, which allocate the most, run first.
 *
 * MUST: the assets are `models/karume-wan2.1/` (untracked; local assets of a real-GPU machine).
 * Environments without them or without a GPU **SKIP explicitly** with the reason (ADR 0005). The
 * fault injections and the per-knob presets are built from a temporary manifest that borrows the
 * same distribution through symlinks (`withManifestOverride`); the original distribution is not
 * changed by a single byte.
 */

import { afterAll, beforeAll, describe, it } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import type { SessionSpec } from "@karume/hub";
import type { GpuContext } from "@karume/runtime";
import { Pipeline } from "../../models/wan.ts";
import { assertSeatsApplied, mergeCensus } from "../../runtime/tests/helpers/pipeline-census.ts";
import { assertAdapterMatchesEnvironment } from "../../runtime/tests/helpers/environment.ts";
import { openResults, runRecordedCase } from "../../runtime/tests/helpers/results.ts";
import {
  type AbBand,
  comparisonOf,
  describeAb,
  describeBundleCensus,
  measureAb,
} from "./helpers/ab-gate.ts";
import { acquireTestGpu, GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import { WAN_ASSEMBLE_COMMAND } from "../../models/tests/helpers/wan-distribution.ts";
import { readTextIfPresent } from "../../models/tests/helpers/read-if-present.ts";
import { settleReleases } from "../../models/tests/helpers/settle-releases.ts";
import {
  assigned,
  declaredSessionOf,
  effectiveSessionOf,
  errorText,
  observeOverridden,
  registerWanAbCase,
  WAN_AB_OBSERVATION,
  type WanAbRun,
  type WanAbScope,
  type WanAbSubject,
} from "./helpers/wan-ab-gate.ts";

/** Location of the distribution (output directory of `dist.py --pipeline wan`). */
const ASSETS_DIR = new URL("../../../models/karume-wan2.1/", import.meta.url);
/**
 * Where measurements and outcomes go (`outputs/verify/<environment key>/<date>_wan-ab/` — safe to
 * delete).
 */
const results = openResults("wan-ab");

/**
 * The model under test (the band is declared per model × preset × frame count, so it is explicit).
 */
const MODEL = "t2v-1.3b";
/** The practical preset under test (ADR 0120 decision 1). */
const PRACTICAL = "f16+dit8-a8-attn8-s16";

/**
 * Opt-in for the heavy 50-step-class records (the 81-frame per-knob record — same variable as
 * `e2e_wan_pipeline_test.ts`).
 */
const FULL_PIPELINE = process.env.KARUME_WAN_FULL_PIPELINE === "1";

/**
 * Cases (frame counts) and bands. The 81 frames, which allocate the most, come first (the lag of
 * the B570's `destroy()` — module doc).
 *
 * The bands are **declarations**: they are not rows per environment key and are not rewritten by
 * `KARUME_REFERENCE` (ADR 0110 decision 4). Derivation rule (decision 5): ceiling = about 2 × the
 * measured relRMS at the observation point, derived independently per case (the 33-frame value is
 * not extrapolated to 81 frames — the attention a8 error grows with the row length N; ADR 0120
 * decision 4). Floor = at least 1 bit different (MUST — {@link judgeAb} always checks it,
 * regardless of the band); `floor` is set above 0 only when the measurement with one knob removed
 * is shown to be clearly smaller than with all of them.
 *
 * MUST: `undefined` before derivation — {@link judgeAb} prints the measurement and stops red (ADR
 * 0120 decision 4). Do not add a check without detection power by putting in a value that passes
 * (the MUST of `ab-gate.ts`). The B570 measurement of step 4 × 2 is written here together with the
 * derivation table. The anima 1024² value (3.0e-2) is a reference only, not a basis for Wan (the
 * attention a8 error grows with the row length N — decision 4).
 *
 * | Case | Measured relRMS | Measured maxAbs | floor | ceiling              | Date, machine    |
 * | ---- | --------------- | --------------- | ----- | -------------------- | ---------------- |
 * | 81f  | 2.0956e-1       | 5.3956e-1       | 0     | 4.2e-1 (× 2, ceiled) | 2026-10-03, B570 |
 * | 33f  | 1.0676e-1       | 2.8391e-1       | 0     | 2.2e-1 (× 2, ceiled) | 2026-10-03, B570 |
 *
 * Per knob (33 frames, recorded): linear a8 alone 2.9991e-2 → + attention a8 1.04e-1 → + s16
 * 1.0676e-1. Attention a8 is most of the error, and at 81 frames (N = 32,760) it is about twice
 * that of 33 frames (N = 14,040) (the measurement of ADR 0120 risk 1). Fault injection ③ (the
 * preset without attention a8) gives 3.0778e-2, ⑤ (another seed) 1.3828e+0.
 */
const CASES: readonly { readonly frames: number; readonly band: AbBand | undefined }[] = [
  { frames: 81, band: { floor: 0, ceiling: 4.2e-1 } },
  { frames: 33, band: { floor: 0, ceiling: 2.2e-1 } },
];

/**
 * Frame count the fault injections run at (the short 33 — showing a check that misses needs no
 * length).
 */
const FAULT_FRAMES = 33;

/**
 * Steps of the per-knob record (ADR 0120 decision 5, step 3) — the same i8 weights with knobs added
 * one at a time (`a8` → `+ attn8` → `+ s16`). The values come from the manifest's declaration of
 * the practical preset (the bundle's values are not copied into the test). The last step = the
 * practical preset's declaration itself.
 *
 * MUST: fail unless the declaration has exactly 3 knobs (do not silently build a step order for a
 * bundle whose knobs changed — decide the order again).
 */
const knobRungs = (
  declared: SessionSpec,
  where: string,
): readonly { readonly label: string; readonly session: SessionSpec }[] => {
  const { linearCompute, attentionCompute, attentionScoreStorage, ...rest } = declared;
  if (
    linearCompute === undefined ||
    attentionCompute === undefined ||
    attentionScoreStorage === undefined ||
    Object.keys(rest).length > 0
  ) {
    throw new Error(
      `${where}: the practical preset's declaration ${JSON.stringify(
        declared,
      )} is not exactly the 3 knobs linear / attention / S storage` +
        " (decide the step order again)",
    );
  }
  return [
    { label: `linearCompute=${linearCompute}`, session: { linearCompute } },
    {
      label: `+ attentionCompute=${attentionCompute}`,
      session: { linearCompute, attentionCompute },
    },
    { label: `+ attentionScoreStorage=${attentionScoreStorage}`, session: declared },
  ];
};

const manifestText = await readTextIfPresent(new URL("karume.json", ASSETS_DIR));
if (manifestText === undefined) {
  console.warn(
    `[karume] ${ASSETS_DIR.pathname} has no karume.json, so the Wan self A/B check is SKIPPED` +
      ` (assemble: ${WAN_ASSEMBLE_COMMAND})`,
  );
}
const RUNNABLE = GPU_AVAILABLE && manifestText !== undefined;

/**
 * The subject of the 2.1 check (passed to the shared {@link registerWanAbCase}). The fixed request
 * values (`request`) are left out so the generation defaults apply (832×480; guidance / shift are
 * the manifest defaults) — the same request as the sha row of the seed path of
 * `e2e_wan_pipeline_test.ts`.
 *
 * MUST: call it only inside the tests of {@link RUNNABLE} (reaching it without a distribution
 * means the skip decision broke, so it fails).
 */
const subjectOf = (): WanAbSubject => {
  if (manifestText === undefined) {
    throw new Error(
      `${ASSETS_DIR.pathname} has no karume.json, yet the A/B check ran (the skip decision broke)`,
    );
  }
  return {
    label: "wan",
    family: "wan",
    model: MODEL,
    practical: PRACTICAL,
    assetsRoot: ASSETS_DIR,
    manifestText,
    results,
    fromPretrained: (source, options) => Pipeline.fromPretrained(source, options),
  };
};

/**
 * Per-knob record (ADR 0120 decision 5, step 3 — **a record only, not a check**). Records in the
 * note and the comparisons of results.json the step-1 latent relRMS of the presets that add knobs
 * one at a time to the same i8 weights in the order of {@link knobRungs} (against the reference
 * partner and against the previous step). The last step (all knobs) reuses the first run of the
 * A/B check's practical preset.
 *
 * Only the shapes that make the record meaningless turn red: the knob not reaching the step's
 * preset (`assertSeatsApplied`) and non-finite values.
 *
 * NOTE: per-layer differences are not taken — there is no measurement graph with per-layer outputs
 * for the i8 weights yet (`…-i8-dyn-probe`) (`--layers` of the export script covers only the f16
 * series).
 */
const registerKnobLadder = (scope: WanAbScope, frames: number, run: () => WanAbRun): void => {
  it(`per-knob record (${frames} frames, not a check)`, async () => {
    const gpu = scope.gpu();
    const subject = scope.subject();
    const { reference: referenceName, referenceRun, first: practical } = run();
    const reference = { name: referenceName, observation: referenceRun };
    const id = `knobs-${frames}f`;
    await runRecordedCase(results, { id, failureNote: errorText }, async (recorded) => {
      const rungs = knobRungs(declaredSessionOf(subject), id);
      const notes: string[] = [];
      let previous = reference.observation;
      for (const [index, { label, session }] of rungs.entries()) {
        const where = `${id} ${label}`;
        const observed =
          index === rungs.length - 1
            ? practical
            : await observeOverridden(gpu, subject, session, frames);
        const effective = effectiveSessionOf(subject, session, where);
        const census = mergeCensus(observed.passes, where);
        // If the knob does not reach the step's preset, that step's relRMS does not measure what it
        // claims to.
        assertSeatsApplied(census, effective, where);
        const toReference = measureAb(observed.latent, reference.observation.latent);
        const toPrevious = measureAb(observed.latent, previous.latent);
        recorded.comparisons.push(
          comparisonOf(
            {
              output: WAN_AB_OBSERVATION,
              reference: reference.name,
              practical: `${PRACTICAL} (up to ${label})`,
            },
            toReference,
            undefined,
          ),
        );
        notes.push(
          `${label}: vs the reference partner ${describeAb(toReference)}, vs the previous step ${describeAb(
            toPrevious,
          )}, census ${describeBundleCensus(census, effective)}`,
        );
        assertEquals(Number.isFinite(toReference.relRms), true, `${where}: non-finite`);
        previous = observed;
      }
      const note = notes.join(" / ");
      console.log(`[e2e] wan per knob, ${frames} frames: ${note}`);
      return { status: "pass", note };
    });
  });
};

for (const { frames, band } of CASES) {
  const id = `${PRACTICAL}-${frames}f`;
  const faults = frames === FAULT_FRAMES;
  const knobs = frames === FAULT_FRAMES || FULL_PIPELINE;
  describe.skipIf(!RUNNABLE)(
    `e2e (real GPU): Wan ${id} step-1 latent agrees with the reference partner in determinism, census and band (self A/B)` +
      (faults ? ", 5 fault injections are red" : "") +
      (knobs ? ", per-knob relRMS recorded" : ""),
    () => {
      // One device per case, shared by the case's its (acquired before, released after them all).
      let gpu: GpuContext | undefined;
      let subject: WanAbSubject | undefined;
      beforeAll(async () => {
        gpu = await acquireTestGpu();
        assertAdapterMatchesEnvironment(gpu);
        subject = subjectOf();
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
      const run = registerWanAbCase(scope, { id, frames, band, faults });
      if (knobs) registerKnobLadder(scope, frames, run);
    },
  );
}
