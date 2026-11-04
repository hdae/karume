/**
 * Anima's **self A/B check** (real GPU, ADR 0110 decision 5). It compares the default preset
 * `f16+dit8-a8-attn8-s16` (practical tier — a preset whose `session` bundles linear a8, attention
 * a8 and f16 storage of S) with the reference partner whose `session` is empty, **on the same
 * machine with the same weights** (derived from the manifest — {@link referencePartnerOf};
 * `f16+dit8` in the current distribution). The comparison partner is the reference tier on the same
 * machine; no torch golden is used.
 *
 * The steps of one case (= one resolution):
 *
 * 1. Take the **step 1 latent** with the reference partner's pipeline (right after the preset first
 *    takes effect — decision 5-4) and stop the generation with a throw on the subscriber side
 *    (`copyLatents()` in `onEvent`, {@link captureThenAbort}). Dispose.
 * 2. Do the same with the practical preset **twice** (two independent pipelines).
 * 3. Apply three checks (if any is red the outcome is `fail`):
 *    - **Determinism** (the MUST of decision 6) — the two latents of the practical preset are
 *      bit-identical ({@link assertBitIdentical}). The fault injection that turns it red: ② (a
 *      different seed for the second run only).
 *    - **Census** (the MUST of decision 5-1) — in the practical preset's transformer run, the
 *      variant of every knob of the bundle runs at least once (`assertSeatsApplied` — the
 *      expectation is the declaration of **the original distribution**). The counts stay in the
 *      outcome's note. Attention a8 / s16 took effect on every dispatch
 *      (`assertAttentionSeatsComplete` — the resolutions of {@link CASES} have both sides a
 *      multiple of 32 px and a self-attention key length that is a multiple of 4, so the declared
 *      route can be expected). The fault injection that turns it red: ③ (a preset with only
 *      `attentionCompute` removed).
 *    - **Band** (decisions 5-2 / 5-3) — relRMS falls in the band of {@link CASES}, there is no
 *      non-finite value, and the output differs from the reference partner by at least 1 bit
 *      ({@link judgeAb}). The fault injection that turns it red: ① (a manifest with the practical
 *      preset's `session` emptied → floor).
 *
 * The final output (PNG) is not produced here — the sha check of the practical-preset rows
 * (`e2e_anima_test.ts`) owns it, without a band (decision 5-4). The results directory is `anima-ab`
 * (separate from `anima` of `e2e_anima_test.ts` — two files with the same series name overwrite
 * each other's `results.json`).
 *
 * MUST: the assets are `models/karume-anima/` (untracked; local assets of the real-GPU machine).
 * Without them, and without a GPU, the cases **SKIP explicitly** with a reason (ADR 0005). The
 * local distribution is passed as the source handle (`nodeDirectory`) — a fault injection uses a
 * temporary manifest that borrows the same distribution through symlinks
 * ({@link withManifestOverride}) and does not change a single byte of the original distribution.
 */

import { test } from "vite-plus/test";
import { assertThrows } from "@std/assert";
import { type DistributionSource, type Manifest, parseManifest } from "@karume/hub";
import { nodeDirectory } from "@karume/hub/node";
import type { SessionDiagnostics } from "@karume/runtime";
import { Pipeline, formatResolution, type ImageSize } from "../../models/anima.ts";
import {
  assertAttentionSeatsComplete,
  assertSeatsApplied,
  type CensusRow,
  mergeCensus,
} from "../../runtime/tests/helpers/pipeline-census.ts";
import { openResults, runRecordedCase } from "../../runtime/tests/helpers/results.ts";
import {
  type AbBand,
  assertBitIdentical,
  captureThenAbort,
  comparisonOf,
  describeAb,
  describeBundleCensus,
  judgeAb,
  measureAb,
  overrideQuantSession,
  quantOf,
  referencePartnerOf,
} from "./helpers/ab-gate.ts";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import { withManifestOverride } from "../../models/tests/helpers/manifest-override.ts";
import { readTextIfPresent } from "../../models/tests/helpers/read-if-present.ts";

/** Where the assets live (`models/karume-anima/` at the repository root). */
const ASSETS_DIR = new URL("../../../models/karume-anima/", import.meta.url);
/**
 * Where the measurements and outcomes go (`outputs/verify/<environment key>/<date>_anima-ab/` —
 * safe to delete).
 */
const results = openResults("anima-ab");

/**
 * The model under test (the default model — a band is declared per model × preset × resolution, so
 * it is stated explicitly).
 */
const MODEL = "anima-turbo-v1.1";
/** The practical preset under test (the default preset). */
const PRACTICAL = "f16+dit8-a8-attn8-s16";
/** The name of the observation point (`comparisons[].output`). */
const OBSERVATION = "step1-latent";

/**
 * The inputs (the same as the sha check of `e2e_anima_test.ts` — the step 1 sigma depends on steps,
 * so they are kept equal).
 */
const PROMPT =
  "1girl, solo, long hair, blue eyes, school uniform, cherry blossoms, outdoors, " +
  "smile, upper body, masterpiece, best quality";
const STEPS = 8;
const SEED = 42;

/**
 * The cases (resolutions) and their bands. `band: undefined` = not derived yet — the check prints
 * the measurement and stops red (the form of ADR 0050 decision 4: no value that merely passes is
 * filled in to add a check without detection power).
 *
 * A band is a **declaration**: it is not a per-environment-key row and `KARUME_REFERENCE` does not
 * rewrite it (ADR 0110 decision 4). The derivation rule (decision 5 — the development machine is a
 * single one today, so the band comes from the measurement of one machine):
 *
 * - **Ceiling** = about 2 × the measured relRMS at the observation point (decision 5-3 —
 *   precedents: 1.6 × in ADR 0028, 2.0 × in irodori). "5 to 10 × the measurement" is not used (the
 *   MUST NOT of decision 4). These are quantised presets, so the agreement with the theoretical
 *   value (the a8 rounding error ≈ the per-layer theoretical value, matching at a median of 1.8% —
 *   ADR 0025) is kept in the table's remarks.
 * - **Floor** = differs by at least 1 bit (the MUST of decision 5-2 — {@link judgeAb} always checks
 *   it, independently of the band). `floor` is set above that only when the measurement of a
 *   temporary manifest with one knob removed can be shown to be **clearly smaller than the
 *   measurement with everything on** (refutation verify/A2 §4-4 — the records only support a
 *   coefficient range of 0.35 to 0.67). If that cannot be shown, `floor: 0`, and the per-knob
 *   fallback is left to the census (decision 5-1).
 *
 * Derivation table (2026-09-26, Intel Arc B570 / Deno, reference partner `f16+dit8`):
 *
 * | Case | Measured relRMS | Measured maxAbs | relRMS of ③ (attn removed) | floor | ceiling | Taken on        |
 * | ---- | --------------- | --------------- | -------------------------- | ----- | ------- | --------------- |
 * | 1024 | 1.4294e-2       | 8.5310e-2       | — (measured at 512 only)   | 0     | 3.0e-2  | 2026-09-26 B570 |
 * | 512  | 1.1899e-2       | 7.2329e-2       | 1.0758e-2 (×0.90)          | 0     | 2.4e-2  | 2026-09-26 B570 |
 *
 * Ceiling = the measurement × 2 rounded to two significant digits. ③ (attentionCompute removed) is
 * 0.90 × the measurement with everything on, which is not "clearly smaller", so the floor is 0 (the
 * census owns the per-knob fallback). It is of the same order as the a8 theoretical value (ADR
 * 0025: a median per-layer rounding error of 1.8%) and consistent with the w8a8 step 1
 * discrimination band (1.6 × the measurement, ADR 0025 decision 6).
 */
const CASES: readonly { readonly resolution: ImageSize; readonly band: AbBand | undefined }[] = [
  { resolution: { width: 1024, height: 1024 }, band: { floor: 0, ceiling: 3.0e-2 } },
  { resolution: { width: 512, height: 512 }, band: { floor: 0, ceiling: 2.4e-2 } },
];

/**
 * The resolution that runs the fault injections (the shortest, 512 — showing that a check is not
 * vacuous needs no particular resolution).
 */
const FAULT_CASE = CASES[1];

const manifestText = await readTextIfPresent(new URL("karume.json", ASSETS_DIR));
if (manifestText === undefined) {
  console.warn(
    `[karume] ${ASSETS_DIR.pathname} has no karume.json, so Anima's self A/B check SKIPs` +
      " (bake it with the exporter's dist.py)",
  );
}
const RUNNABLE = GPU_AVAILABLE && manifestText !== undefined;

const readManifest = (): Manifest => parseManifest(manifestText as string);

/** The observation of one run (the step 1 latent and the transformer census up to it). */
type Observation = {
  readonly latent: Float32Array<ArrayBuffer>;
  readonly census: readonly CensusRow[];
};

/**
 * Builds the pipeline of `quant`, takes the step 1 latent and stops (through dispose — the pipeline
 * also acquires and releases the GPU itself). The census covers only the transformer runs (the
 * quant's `session` applies only to the DiT's Session — `sessionOptions` in `anima/pipeline.ts`).
 */
const observeStep1 = async (
  source: DistributionSource,
  quant: string,
  resolution: ImageSize,
  seed: number,
): Promise<Observation> => {
  const where = `${quant} / ${formatResolution(resolution)} / seed ${seed}`;
  const runs: SessionDiagnostics["lastRunPipelines"][] = [];
  await using pipeline = await Pipeline.fromPretrained(source, {
    model: MODEL,
    quant,
    onRunDiagnostics: (component, diagnostics) => {
      if (component === "transformer") runs.push(diagnostics.lastRunPipelines);
    },
  });
  const latent = await captureThenAbort<Float32Array<ArrayBuffer>>(where, (capture) =>
    pipeline.generate({
      prompt: PROMPT,
      resolution,
      steps: STEPS,
      seed,
      onEvent: (event) => {
        if (event.kind === "denoise-step" && event.step === 1) capture(event.copyLatents().data);
      },
    }),
  );
  return { latent, census: mergeCensus(runs, `${where}: transformer`) };
};

const errorText = (cause: unknown): string =>
  cause instanceof Error ? cause.message : String(cause);

for (const { resolution, band } of CASES) {
  const id = `${PRACTICAL}-${formatResolution(resolution)}`;
  test.skipIf(!RUNNABLE)(
    `e2e(real GPU): the step 1 latent of ${id} balances the reference partner on determinism, census and band (self A/B)`,
    async () => {
      await runRecordedCase(results, { id, failureNote: errorText }, async (recorded) => {
        const manifest = readManifest();
        const reference = referencePartnerOf(manifest, MODEL, PRACTICAL);
        const declared = quantOf(manifest, MODEL, PRACTICAL).session;
        const source = nodeDirectory(ASSETS_DIR);
        const referenceRun = await observeStep1(source, reference, resolution, SEED);
        const first = await observeStep1(source, PRACTICAL, resolution, SEED);
        const second = await observeStep1(source, PRACTICAL, resolution, SEED);
        const measured = measureAb(first.latent, referenceRun.latent);
        recorded.comparisons.push(
          comparisonOf({ output: OBSERVATION, reference, practical: PRACTICAL }, measured, band),
        );
        const census = describeBundleCensus(first.census, declared);
        console.log(
          `[e2e] anima A/B ${id}: ${reference} → ${PRACTICAL} ${describeAb(measured)} ` +
            `/ census ${census}`,
        );
        assertBitIdentical(first.latent, second.latent, `${id}: the two practical-preset runs`);
        assertSeatsApplied(first.census, declared, `${id}: the practical preset's transformer`);
        assertAttentionSeatsComplete(
          first.census,
          declared,
          `${id}: the practical preset's transformer`,
        );
        judgeAb(measured, band, id);
        return { status: "pass", note: `census ${census}` };
      });
    },
  );
}

// --- Fault injections (evidence that no check is vacuous — ADR 0110 decision 5-5)
// ------------------
//
// Each one asserts that a check turns red (green = the check missed the fault). The temporary
// manifest borrows the same distribution through symlinks and replaces only the practical preset's
// `session`. Only the one resolution of {@link FAULT_CASE} runs.

test.skipIf(!RUNNABLE)(
  "e2e(real GPU): fault injection ① a manifest with the practical preset's session emptied is red on the floor",
  async () => {
    const { resolution, band } = FAULT_CASE;
    const id = `fault1-empty-session-${formatResolution(resolution)}`;
    await runRecordedCase(results, { id, failureNote: errorText }, async (recorded) => {
      const manifest = readManifest();
      const reference = referencePartnerOf(manifest, MODEL, PRACTICAL);
      const injected = overrideQuantSession(
        JSON.parse(manifestText as string),
        MODEL,
        PRACTICAL,
        {},
      );
      const referenceRun = await observeStep1(
        nodeDirectory(ASSETS_DIR),
        reference,
        resolution,
        SEED,
      );
      const faulty = await withManifestOverride(ASSETS_DIR, injected, (source) =>
        observeStep1(source, PRACTICAL, resolution, SEED),
      );
      const measured = measureAb(faulty.latent, referenceRun.latent);
      recorded.comparisons.push(
        comparisonOf(
          { output: OBSERVATION, reference, practical: `${PRACTICAL} (empty session)` },
          measured,
          band,
        ),
      );
      console.log(`[e2e] anima fault injection ① ${describeAb(measured)}`);
      assertThrows(() => judgeAb(measured, band, id), Error, "floor failure");
    });
  },
);

test.skipIf(!RUNNABLE)(
  "e2e(real GPU): fault injection ② a different seed on the practical preset's second run turns the determinism check red",
  async () => {
    const { resolution } = FAULT_CASE;
    const source = nodeDirectory(ASSETS_DIR);
    const first = await observeStep1(source, PRACTICAL, resolution, SEED);
    const second = await observeStep1(source, PRACTICAL, resolution, SEED + 1);
    assertThrows(
      () => assertBitIdentical(first.latent, second.latent, "fault injection ②"),
      Error,
      "are not bit-identical",
    );
  },
);

/**
 * Fault injection ③ — a preset with only `attentionCompute` removed. Asserts that the census (whose
 * expectation is the original distribution's declaration) turns red. The band verdict is **only
 * recorded** (a rerun of ADR 0030 decision 5 — the loss of a later stage of stacked presets is hard
 * to catch numerically. If the band stays green it shows the census is needed; if red, it is
 * material for setting the floor coefficient k above 0).
 */
test.skipIf(!RUNNABLE)(
  "e2e(real GPU): fault injection ③ a preset with only attentionCompute removed is red on the census (the band is recorded)",
  async () => {
    const { resolution, band } = FAULT_CASE;
    const id = `fault3-no-attention-a8-${formatResolution(resolution)}`;
    await runRecordedCase(results, { id, failureNote: errorText }, async (recorded) => {
      const manifest = readManifest();
      const reference = referencePartnerOf(manifest, MODEL, PRACTICAL);
      const declared = quantOf(manifest, MODEL, PRACTICAL).session;
      const { attentionCompute: _dropped, ...withoutAttention } = declared;
      const injected = overrideQuantSession(
        JSON.parse(manifestText as string),
        MODEL,
        PRACTICAL,
        withoutAttention,
      );
      const referenceRun = await observeStep1(
        nodeDirectory(ASSETS_DIR),
        reference,
        resolution,
        SEED,
      );
      const faulty = await withManifestOverride(ASSETS_DIR, injected, (source) =>
        observeStep1(source, PRACTICAL, resolution, SEED),
      );
      const measured = measureAb(faulty.latent, referenceRun.latent);
      recorded.comparisons.push(
        comparisonOf(
          { output: OBSERVATION, reference, practical: `${PRACTICAL} (attentionCompute removed)` },
          measured,
          band,
        ),
      );
      let bandVerdict = "green";
      try {
        judgeAb(measured, band, id);
      } catch (cause) {
        bandVerdict = `red (${errorText(cause)})`;
      }
      console.log(`[e2e] anima fault injection ③ ${describeAb(measured)} / band ${bandVerdict}`);
      assertThrows(
        () => assertSeatsApplied(faulty.census, declared, id),
        Error,
        "attentionCompute",
      );
      return { status: "pass", note: `band ${bandVerdict}` };
    });
  },
);
