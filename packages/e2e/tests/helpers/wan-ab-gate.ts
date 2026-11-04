/**
 * One case of Wan's **self A/B check** (ADR 0110 decision 5 / ADR 0120 decision 4) — shared by 2.1 (`e2e_wan_ab_test.ts`)
 * and 2.2 (`e2e_wan_ti2v_ab_test.ts`). The differences between the two generations (the pipeline class, the census table's
 * series, the model, the preset, where the distribution lives, the fixed request values, the log prefix) come in through
 * {@link WanAbSubject}; the procedure, the order of the verdicts, the shape of the fault injections and the recording into the
 * results are one implementation (so the check's claim does not diverge between generations). The verdict of "what turns red"
 * itself is the pure functions of `ab-gate.ts`.
 *
 * The procedure of one case (= one frame count) ({@link registerWanAbCase}):
 *
 * 1. With the pipeline of the reference partner (derived from the manifest — `referencePartnerOf`), capture the **step 1
 *    latent** (step 1 of a 2-step run — right after the preset first takes effect, decision 5-4) and abort generation with a
 *    throw in the subscriber (`captureThenAbort`). Dispose and wait for the release.
 * 2. Do the same **twice** with the practical preset (two independent pipelines).
 * 3. Apply three checks: determinism (the practical preset's two runs are bit-identical — the MUST of decision 6), census
 *    (each DiT pass of the practical preset is compared with the bundle's census table row at the exact count, and the two
 *    attention knobs need at least one dispatch — decision 5-1), and the band (`judgeAb` — decisions 5-2 / 5-3).
 * 4. Only in the cases that ask for it, 5 fault injections (each asserts that "the check turns red" — decision 5-5).
 *
 * VRAM: pipelines are built one at a time in sequence (never at the same time — construction builds no Session, and the DiT
 * Session is torn down by the abort at step 1). After every observation `settleReleases` waits for the release (the delayed
 * release of `destroy()` on the B570 — ADR 0120 risk 6).
 *
 * MUST: the fault-injection presets are built from a temporary manifest that borrows the same distribution through symlinks
 * (`withManifestOverride`); not one byte of the original distribution is rewritten.
 */

import { beforeAll, it } from "vite-plus/test";
import { assert, assertEquals, assertThrows } from "@std/assert";
import { type DistributionSource, parseManifest, type SessionSpec } from "@karume/hub";
import { nodeDirectory } from "@karume/hub/node";
import type { GpuContext, SessionDiagnostics, SessionOptions } from "@karume/runtime";
import type {
  GeneratedVideo,
  FromPretrainedOptions,
  Prompt,
  Ti2vGenerateRequest,
} from "../../../models/wan.ts";
import {
  assertAttentionSeatsComplete,
  assertSeatsApplied,
  mergeCensus,
} from "../../../runtime/tests/helpers/pipeline-census.ts";
import { type Results, runRecordedCase } from "../../../runtime/tests/helpers/results.ts";
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
} from "./ab-gate.ts";
import {
  assertRowCensus,
  type BundleCensusRow,
  censusRowOf,
  effectiveSessionOptions,
} from "../../../models/tests/helpers/census-table.ts";
import { withManifestOverride } from "../../../models/tests/helpers/manifest-override.ts";
import { settleReleases } from "../../../models/tests/helpers/settle-releases.ts";

/** The name of the observation point (`comparisons[].output`). */
export const WAN_AB_OBSERVATION = "step1-latent";

/**
 * The inputs (the same prompt and seed as the sha rows of each generation's seed path; the size / guidance / shift and the I2V
 * conditioning image are the `request` of {@link WanAbSubject} — a generation that omits it uses the manifest's defaults). The σ
 * of step 1 is determined by steps and shift, so steps is fixed at 2 (step 1 of a 2-step run).
 */
const PROMPT = "boxing-cats";
const SEED = 42;
const STEPS = 2;
/**
 * The number of DiT passes up to step 1 (CFG runs 2 passes per step, uncond → cond — this assumes guidance > 1. If the
 * assumption breaks, the pass-count check in {@link observeStep1} fails).
 */
const PASSES_TO_STEP1 = 2;

/**
 * The pipeline surface the A/B needs (`Pipeline` and `Ti2vPipeline` satisfy it structurally). The request is taken as the
 * wider type (`Ti2vGenerateRequest`, with 2.2's I2V fields) — 2.1's `generate` takes the narrower type, but parameters are
 * contravariant, so it satisfies the surface.
 */
export type WanAbPipeline = AsyncDisposable & {
  readonly prompts: readonly Prompt[];
  generate(request: Ti2vGenerateRequest): Promise<GeneratedVideo>;
};

/** What differs per generation (the check's procedure and verdicts are the single {@link registerWanAbCase}). */
export type WanAbSubject = {
  /** The log prefix (`[e2e] <label> A/B …`). */
  readonly label: string;
  /** The family of the census table and the acceptance table (the manifest's pipeline name). */
  readonly family: "wan" | "wan-ti2v";
  /** The model under test (explicit, because a band is declared per model × preset × frame count). */
  readonly model: string;
  /** The practical preset under test. */
  readonly practical: string;
  /** Where the distribution lives (`karume.json` and the containers). */
  readonly assetsRoot: URL;
  /** The raw text of the distribution's `karume.json` (the preset declarations and the base of the temporary manifest overrides). */
  readonly manifestText: string;
  /** Where the measurements and outcomes are recorded. */
  readonly results: Results;
  /** The generation's pipeline constructor (`Pipeline.fromPretrained` / `Ti2vPipeline.fromPretrained`). */
  readonly fromPretrained: (
    source: DistributionSource,
    options: FromPretrainedOptions,
  ) => Promise<WanAbPipeline>;
  /**
   * The fixed request values (the size, and the guidance / shift that determine the step 1 latent — plus the conditioning
   * image for I2V). Omitted, the generation's defaults apply (the manifest's `pipelineConfig` and the default size) — 2.1 uses
   * the defaults, like its sha rows. A generation that sets them sets all of them (setting only the size and leaving σ and the
   * CFG pass count at the defaults would silently move the band's assumptions when a default moves). The I2V step 1 latent is
   * the scheduler state of `copyLatents` (before the first latent frame is replaced by the condition — the doc of
   * `GenerateEvent`).
   */
  readonly request?: {
    readonly width: number;
    readonly height: number;
    readonly guidance: number;
    readonly shift: number;
    /**
     * The I2V conditioning image (Wan2.2 only — omitted means T2V). The size is set explicitly by `width` / `height` above
     * (not left to the selection by the image's aspect ratio — so the size the band assumes does not silently move with the
     * image). Passed to a 2.1 pipeline, `generate` rejects it with `ModelInputError` (fail loudly — one type shared by both
     * generations). When passed, each observation checks that runs of the encoder's 3 graphs arrived (catching, outside the band
     * and the census, a regression where the image is silently dropped and the run goes T2V).
     */
    readonly image?: Ti2vGenerateRequest["image"];
    /** How the conditioning image is fitted to the size (only with `image` — omitted means the official `"crop"`). */
    readonly fit?: Ti2vGenerateRequest["fit"];
  };
};

/** The observation of one run (the step 1 latent and the census of each DiT pass up to it). */
export type WanStep1Observation = {
  readonly latent: Float32Array<ArrayBuffer>;
  /** One element per DiT pass ({@link PASSES_TO_STEP1} of them). */
  readonly passes: readonly SessionDiagnostics["lastRunPipelines"][];
};

export const errorText = (cause: unknown): string =>
  cause instanceof Error ? cause.message : String(cause);

/**
 * Builds the pipeline of `quant` on `gpu`, captures the step 1 latent and aborts (up to disposing and waiting for the
 * release). The census covers only the DiT runs (the quant's `session` affects only the DiT Session — the VAE stage of
 * `wan/tile-decode.ts` gets `{}`).
 */
const observeStep1 = async (
  gpu: GpuContext,
  subject: WanAbSubject,
  source: DistributionSource,
  quant: string,
  frames: number,
  seed: number,
): Promise<WanStep1Observation> => {
  const where = `${quant} / ${frames} frames / seed ${seed}`;
  const passes: SessionDiagnostics["lastRunPipelines"][] = [];
  /** The components whose run diagnostics arrived (checks that the I2V conditioning image reached the encoder stage). */
  const ran = new Set<string>();
  try {
    await using pipeline = await subject.fromPretrained(source, {
      gpu,
      model: subject.model,
      quant,
      // The A/B is about the DiT preset — request the precomputed embeddings explicitly (the default "gpu" adds the umT5
      // stage and gives the prompt context different values — breaking the assumptions the band was measured under).
      textEncoder: "precomputed",
      onRunDiagnostics: (component, diagnostics) => {
        ran.add(component);
        if (component === "transformer") passes.push(diagnostics.lastRunPipelines);
      },
    });
    const prompt = pipeline.prompts.find(({ name }) => name === PROMPT)?.prompt;
    if (prompt === undefined) throw new Error(`${where}: the embedding assets have no '${PROMPT}'`);
    const latent = await captureThenAbort<Float32Array<ArrayBuffer>>(where, (capture) =>
      pipeline.generate({
        prompt,
        seed,
        steps: STEPS,
        frames,
        ...subject.request,
        onEvent: (event) => {
          if (event.kind === "denoise-step" && event.step === 1) {
            capture(event.copyLatents().data);
          }
        },
      }),
    );
    assertEquals(passes.length, PASSES_TO_STEP1, `${where}: DiT passes up to step 1`);
    if (subject.request?.image !== undefined) {
      // In a regression where the conditioning image drops out of the request and the run takes the T2V path, both presets
      // produce T2V latents, the band (relRMS) stays green with T2V values, and the census sees only the DiT, so neither catches
      // it. Requiring runs of the encoder's 3 graphs pins that the I2V path was taken.
      for (const component of ["vae_encoder_pre", "vae_encoder_attn", "vae_encoder_post"]) {
        assert(
          ran.has(component),
          `${where}: a conditioning image was passed but no ${component} run arrived`,
        );
      }
    }
    return { latent, passes };
  } finally {
    // Wait for this observation's allocations to be released before the next observation's pipeline allocates (B570 —
    // settleReleases).
    await settleReleases(gpu);
  }
};

/** The distribution's declaration of the practical preset (the manifest's literal `session` — the base of the temporary manifest overrides). */
export const declaredSessionOf = (subject: WanAbSubject): SessionSpec =>
  quantOf(parseManifest(subject.manifestText), subject.model, subject.practical).session;

/** The effective options after passing the declaration through the family's composition (acceptance table — `WAN_SESSION_POLICY`) (the census table's key and the census expectations). */
export const effectiveSessionOf = (
  subject: WanAbSubject,
  session: SessionSpec,
  where: string,
): SessionOptions => effectiveSessionOptions(subject.family, session, {}, where);

/** The census table row of the practical preset's bundle (fails loudly when absent — this check is about the practical preset itself, so a missing row is a gap in the table). */
const practicalRowOf = (
  subject: WanAbSubject,
  effective: SessionOptions,
  where: string,
): BundleCensusRow => {
  const row = censusRowOf(subject.family, subject.model, effective);
  if (row === undefined) {
    throw new Error(`${where}: no row in the bundle census table (helpers/census-table.ts)`);
  }
  return row;
};

/**
 * The practical preset's census (decision 5-1): each pass is compared with the table row at the exact count (linear), and
 * each bundle knob's variant must have run at least once (attention — by convention not in the table: the row-block count
 * changes with the device's binding limit). For attention it further checks that a8 / s16 took effect on every dispatch
 * (fully eligible — a ratio, so independent of the device). The census covers only the DiT Session's runs
 * ({@link observeStep1}), so f32 attention of another Session does not mix in.
 */
const assertPracticalCensus = (
  row: BundleCensusRow,
  observation: WanStep1Observation,
  effective: SessionOptions,
  where: string,
): void => {
  const checked = assertRowCensus(
    row,
    new Map([["pass", observation.passes]]),
    () => "transformer",
    where,
  );
  assertEquals(
    checked,
    observation.passes.length,
    `${where}: passes compared with the table's expectations`,
  );
  const census = mergeCensus(observation.passes, where);
  assertSeatsApplied(census, effective, where);
  assertAttentionSeatsComplete(census, effective, where);
};

/**
 * A copy of a census table row with the expected linear count shifted by `delta` (fault injection ④ — the "exact" check
 * catches one dispatch too many or too few on the real census).
 */
const shiftedLinearRow = (row: BundleCensusRow, delta: number): BundleCensusRow => {
  if (row.family !== "wan" && row.family !== "wan-ti2v") {
    throw new Error(`the census table row is not of a wan family (${row.family})`);
  }
  const pass = row.census.transformer?.pass;
  const linear = pass?.linearCompute;
  if (pass === undefined || linear === undefined) {
    throw new Error(
      `the ${row.family} row of the census table has no linearCompute for transformer/pass`,
    );
  }
  return {
    ...row,
    census: {
      transformer: {
        pass: { ...pass, linearCompute: { ...linear, variant: linear.variant + delta } },
      },
    },
  };
};

/**
 * Builds the practical preset with only its `session` replaced, through a temporary manifest, and captures step 1 (fault
 * injection and per-knob presets — the weights and the other presets are untouched).
 */
export const observeOverridden = (
  gpu: GpuContext,
  subject: WanAbSubject,
  session: Readonly<Record<string, unknown>>,
  frames: number,
): Promise<WanStep1Observation> =>
  withManifestOverride(
    subject.assetsRoot,
    overrideQuantSession(
      JSON.parse(subject.manifestText),
      subject.model,
      subject.practical,
      session,
    ),
    (source) => observeStep1(gpu, subject, source, subject.practical, frames, SEED),
  );

/** The observations {@link registerWanAbCase} returns to the caller (2.1's per-knob records reuse the A/B observations). */
export type WanAbRun = {
  /** The reference partner's name (derived from the manifest). */
  readonly reference: string;
  readonly referenceRun: WanStep1Observation;
  /** The practical preset's first run. */
  readonly first: WanStep1Observation;
};

/** The specification of one case. */
export type WanAbCase = {
  /** The `cases[].id` in the results (the caller builds the same value as the test's name). */
  readonly id: string;
  readonly frames: number;
  /** The declared band (`undefined` before it is derived — `judgeAb` prints the measurement and stops red). */
  readonly band: AbBand | undefined;
  /** Whether to run the 5 fault injections (showing that a check is not vacuous needs no length, so only the short cases). */
  readonly faults: boolean;
};

/**
 * Read access to the values the caller's `beforeAll` assigns at describe scope (the device and the
 * subject). The accessors are only called from hooks and its, never at collection time.
 */
export type WanAbScope = {
  readonly gpu: () => GpuContext;
  readonly subject: () => WanAbSubject;
};

/** The value a `beforeAll` assigned, or a loud failure when the hook did not get that far. */
export const assigned = <T>(value: T | undefined, what: string): T => {
  if (value === undefined) throw new Error(`${what} was not assigned by beforeAll`);
  return value;
};

/** What the case's `beforeAll` derives and observes before its its run. */
type WanAbCaseState = WanAbRun & {
  readonly effective: SessionOptions;
  readonly row: BundleCensusRow;
  /** The practical preset's second run. */
  readonly second: WanStep1Observation;
};

/**
 * Registers one case (one frame count) of the self A/B gate inside the caller's describe (the steps
 * in the module doc): a `beforeAll` takes the three observations, then one it per gate step. The
 * caller owns the device (acquiring it, checking the adapter, releasing it) in its own
 * `beforeAll` / `afterAll`, registered before this call — Vitest runs a suite's `beforeAll` hooks
 * sequentially in registration order (`sequence.hooks: "stack"`, the default), so the device is
 * there when the observations start. Returns an accessor for the observations, valid inside its
 * registered after this call.
 */
export const registerWanAbCase = (
  scope: WanAbScope,
  { id, frames, band, faults }: WanAbCase,
): (() => WanAbRun) => {
  let state: WanAbCaseState | undefined;
  beforeAll(async () => {
    const gpu = scope.gpu();
    const subject = scope.subject();
    const { practical } = subject;
    const manifest = parseManifest(subject.manifestText);
    const reference = referencePartnerOf(manifest, subject.model, practical);
    const effective = effectiveSessionOf(
      subject,
      quantOf(manifest, subject.model, practical).session,
      id,
    );
    const row = practicalRowOf(subject, effective, id);
    const source = nodeDirectory(subject.assetsRoot);
    const referenceRun = await observeStep1(gpu, subject, source, reference, frames, SEED);
    const first = await observeStep1(gpu, subject, source, practical, frames, SEED);
    const second = await observeStep1(gpu, subject, source, practical, frames, SEED);
    state = { reference, referenceRun, first, second, effective, row };
  });
  const current = (): WanAbCaseState => assigned(state, `${id}: the observations`);

  it(`${id}: determinism, census, band`, async () => {
    const { label, practical, results } = scope.subject();
    const { reference, referenceRun, first, second, effective, row } = current();
    await runRecordedCase(results, { id, failureNote: errorText }, (recorded) => {
      const measured = measureAb(first.latent, referenceRun.latent);
      recorded.comparisons.push(
        comparisonOf({ output: WAN_AB_OBSERVATION, reference, practical }, measured, band),
      );
      const census = describeBundleCensus(mergeCensus(first.passes, id), effective);
      console.log(
        `[e2e] ${label} A/B ${id}: ${reference} → ${practical} ${describeAb(
          measured,
        )} / census ${census}`,
      );
      assertBitIdentical(first.latent, second.latent, `${id}: the practical preset's two runs`);
      assertPracticalCensus(row, first, effective, `${id}: the practical preset's transformer`);
      judgeAb(measured, band, id);
      return Promise.resolve({ status: "pass", note: `census ${census}` });
    });
  });

  if (!faults) return current;

  // --- Fault injection (showing that each check is not vacuous — ADR 0110 decision 5-5) -----------
  // Each asserts that "the check turns red" (green = the check missed the fault). ①–③ are the 3 of ADR 0120 decision 4;
  // ④⑤ confirm that the census's "exact" and the ceiling work on real values.

  it("fault injection ① a manifest with the practical preset's session emptied (= the reference partner itself) is red at the floor", async () => {
    const gpu = scope.gpu();
    const subject = scope.subject();
    const { label, practical, results } = subject;
    const { reference, referenceRun } = current();
    const faultId = `fault1-empty-session-${frames}f`;
    await runRecordedCase(results, { id: faultId, failureNote: errorText }, async (recorded) => {
      const faulty = await observeOverridden(gpu, subject, {}, frames);
      const measured = measureAb(faulty.latent, referenceRun.latent);
      recorded.comparisons.push(
        comparisonOf(
          { output: WAN_AB_OBSERVATION, reference, practical: `${practical} (empty session)` },
          measured,
          band,
        ),
      );
      console.log(`[e2e] ${label} fault injection ① ${describeAb(measured)}`);
      assertThrows(() => judgeAb(measured, band, faultId), Error, "floor failure");
    });
  });

  // NOTE: ⑤ checks that the band's ceiling turns red, so while the band is not derived yet (`undefined`) `judgeAb` throws
  // "band not derived yet" and this step is red too (it becomes effective once the band is declared — the same as 2.1's steps
  // 3 / 4; ② works in the meantime).
  it("fault injection ② / ⑤ running the practical preset with another seed — determinism against the first run and the ceiling against the reference partner are red", async () => {
    const gpu = scope.gpu();
    const subject = scope.subject();
    const { label, practical, results } = subject;
    const { reference, referenceRun, first } = current();
    const faultId = `fault2-5-other-seed-${frames}f`;
    await runRecordedCase(results, { id: faultId, failureNote: errorText }, async (recorded) => {
      const shifted = await observeStep1(
        gpu,
        subject,
        nodeDirectory(subject.assetsRoot),
        practical,
        frames,
        SEED + 1,
      );
      // ② determinism: only the second run of the same preset has a different input (the bit-identity check catches the
      // difference on real latents).
      assertThrows(
        () => assertBitIdentical(first.latent, shifted.latent, "fault injection ②"),
        Error,
        "are not bit-identical",
      );
      // ⑤ ceiling: the observation point became a latent unrelated to the reference (a stand-in for collapse — step 1 from
      // different initial noise).
      const measured = measureAb(shifted.latent, referenceRun.latent);
      recorded.comparisons.push(
        comparisonOf(
          {
            output: WAN_AB_OBSERVATION,
            reference,
            practical: `${practical}（seed ${SEED + 1}）`,
          },
          measured,
          band,
        ),
      );
      console.log(`[e2e] ${label} fault injection ⑤ ${describeAb(measured)}`);
      assertThrows(() => judgeAb(measured, band, faultId), Error, "ceiling failure");
    });
  });

  // ③ the band verdict is only recorded (a repeat of ADR 0030 decision 5 — a later knob of a stacked preset dropping out is
  // hard to catch numerically. If the band stays green, that shows the census is needed; if red, it is material for setting
  // the floor's coefficient above 0).
  it("fault injection ③ a preset without attentionCompute is red in the census (the band is recorded)", async () => {
    const gpu = scope.gpu();
    const subject = scope.subject();
    const { label, practical, results } = subject;
    const { reference, referenceRun, effective, row } = current();
    const faultId = `fault3-no-attention-a8-${frames}f`;
    await runRecordedCase(results, { id: faultId, failureNote: errorText }, async (recorded) => {
      const { attentionCompute: _dropped, ...withoutAttention } = declaredSessionOf(subject);
      const faulty = await observeOverridden(gpu, subject, withoutAttention, frames);
      const measured = measureAb(faulty.latent, referenceRun.latent);
      recorded.comparisons.push(
        comparisonOf(
          {
            output: WAN_AB_OBSERVATION,
            reference,
            practical: `${practical} (without attentionCompute)`,
          },
          measured,
          band,
        ),
      );
      let bandVerdict = "green";
      try {
        judgeAb(measured, band, faultId);
      } catch (cause) {
        bandVerdict = `red (${errorText(cause)})`;
      }
      console.log(`[e2e] ${label} fault injection ③ ${describeAb(measured)} / band ${bandVerdict}`);
      assertThrows(
        () => assertPracticalCensus(row, faulty, effective, faultId),
        Error,
        "attentionCompute",
      );
      return { status: "pass", note: `band ${bandVerdict}` };
    });
  });

  it("fault injection ④ shifting the census table's linear count by 1 is red on the real census (no extra GPU work)", () => {
    const { first, effective, row } = current();
    for (const delta of [-1, 1]) {
      assertThrows(
        () =>
          assertPracticalCensus(
            shiftedLinearRow(row, delta),
            first,
            effective,
            `fault injection ④ (${delta > 0 ? "+" : ""}${delta})`,
          ),
        Error,
        "variant of linearCompute=a8",
      );
    }
  });

  return current;
};
