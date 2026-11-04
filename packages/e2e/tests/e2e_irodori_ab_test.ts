/**
 * Irodori's **self A/B check** (real GPU, ADR 0110 decision 5). It compares the practical seat
 * `i8-a8` (the default seat — `linearCompute: "a8"`, which runs `dit`'s linears with integer dot
 * products down to the activations) with the reference seat whose `session` is empty (derived from
 * the manifest — {@link referencePartnerOf}; `i8` in today's distribution) **on the same machine
 * with the same i8 weights**.
 *
 * The shape is the same as `e2e_anima_ab_test.ts` (the procedure, the checks and the fault
 * injection mapping are in that module's doc). The differences:
 *
 * - The observation point is **the step-1 latent** (`copyLatents()` of `denoise-step` — the latent
 *   `[frames, latentDim]` after one step completes, not a forward count inside CFG).
 *   `durationSeconds` fixes S and the `duration` graph does not run (the quant's `session` reaches
 *   only `dit`, so deciding S does not depend on the seat).
 * - The bundle has a single knob, `linearCompute`, so there are 2 fault injections: ① (a manifest
 *   with `session` emptied → floor) and ② (a different seed only for the second run →
 *   determinism). ③ "remove just one knob" takes the same shape as ① on a one-knob seat, so it is
 *   not included.
 * - The exact key census of the practical seat lives here too (the "key census" test below): over
 *   one full generation on the resident DiT path, every `dit` / `dit_context` run dispatches
 *   exactly the census-table number of i8a8 GEMMs and `quantize_rows`, zero f32 linear kernels,
 *   and `dit_context` runs once per generation. There is no band against the `i8` torch golden:
 *   activation quantization cannot be a numeric-parity gate (ADR 0025 decision 6 / ADR 0026), and
 *   the silent-f32-fallback floor that band guarded is covered by the census and fault injection ①.
 *
 * The band derivation rule is the same as the doc of `CASES` in `e2e_anima_ab_test.ts` (ceiling =
 * about measured × 2, floor = MUST differ by at least 1 bit + a coefficient only when it can be
 * shown).
 *
 * MUST: the assets are `models/karume-irodori-v4.1-small/` (untracked, local assets of the
 * real-GPU machine — the distribution the examples' scripts default to). An environment without
 * them or without a GPU **SKIPs explicitly** with the reason (ADR 0005).
 */

import { test } from "vite-plus/test";
import { assert, assertEquals, assertThrows } from "@std/assert";
import {
  type DistributionSource,
  type Manifest,
  parseManifest,
  resolveSelection,
  selectionRefs,
} from "@karume/hub";
import { nodeDirectory } from "@karume/hub/node";
import type { SessionDiagnostics } from "@karume/runtime";
import { Pipeline, type RunComponent, SOURCES } from "../../models/irodori.ts";
import {
  assertSeatsApplied,
  type CensusRow,
  mergeCensus,
  requireCensus,
  SEAT_SIGNATURES,
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
import {
  assertRowCensus,
  censusRowOf,
  effectiveSessionOptions,
} from "../../models/tests/helpers/census-table.ts";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import { withManifestOverride } from "../../models/tests/helpers/manifest-override.ts";
import { readTextIfPresent } from "../../models/tests/helpers/read-if-present.ts";

/** Where the distribution lives (the v4.1 output of `karume dist --pipeline irodori`). */
const ASSETS_DIR = new URL("../../../models/karume-irodori-v4.1-small/", import.meta.url);
/**
 * v4.1-small references the components it shares with v4-small in v4-small's repository (ADR 0038
 * §7 addendum 2026-10-28); a local source reads them from v4-small's mirror through an explicit
 * mapping. A mirror built before the references declares none, and the mapping goes unused.
 */
const V4_REPO = SOURCES["irodori-v4-small"].repo;
const V4_ASSETS_DIR = new URL("../../../models/karume-irodori-v4-small/", import.meta.url);
const CROSS_REPO = { crossRepo: { [V4_REPO]: nodeDirectory(V4_ASSETS_DIR) } };
/** The v4.1-small distribution with its cross-repository mapping. */
const assetsSource = (): DistributionSource => nodeDirectory(ASSETS_DIR, CROSS_REPO);
/**
 * Where the measurements and settlements go (`outputs/verify/<environment key>/<date>_irodori-ab/`
 * — safe to delete).
 */
const results = openResults("irodori-ab");

const MODEL = "v4.1-small";
const PRACTICAL = "i8-a8";
const OBSERVATION = "step1-latent";
/** Utterance length (seconds). S is fixed so the `duration` graph does not run. */
const DURATION_SECONDS = 3;
const SEED = 1235;

/** One case's input and band. */
type AbCase = {
  readonly name: string;
  readonly text: string;
  readonly caption?: string;
  readonly band: AbBand | undefined;
};

/**
 * Cases and bands (`band: undefined` = not derived yet — the check prints the measurement and
 * stops red). The text and caption are the same sentences as `e2e_irodori_wav_test.ts` (no
 * reference audio — the speaker path does not depend on the seat).
 *
 * Derivation table (2026-09-26, Intel Arc B570 / Deno, reference seat `i8`, ceiling = measured × 2
 * to 2 significant digits):
 *
 * | Case    | relRMS      | maxAbs      | floor | ceiling | Date, machine   |
 * | ------- | ----------- | ----------- | ----- | ------- | --------------- |
 * | no-ref  | 3.5879e-3   | 1.4885e-2   | 0     | 7.2e-3  | 2026-09-26 B570 |
 * | caption | 5.8457e-3   | 2.2464e-2   | 0     | 1.2e-2  | 2026-09-26 B570 |
 */
const CASES: readonly AbCase[] = [
  {
    name: "no-ref",
    text: "本日はお越しいただき、誠にありがとうございます。",
    band: { floor: 0, ceiling: 7.2e-3 },
  },
  {
    name: "caption",
    text: "今日は近くの店まで歩いて行きました。とても良い天気でしたね。",
    caption:
      "若く元気な女性の声。カフェの店員のように、明るくハキハキとした少し高めのトーンで話している。",
    band: { floor: 0, ceiling: 1.2e-2 },
  },
];

/** The case the fault injection runs (no-ref, the minimal setup). */
const FAULT_CASE = CASES[0];

const manifestText = await readTextIfPresent(new URL("karume.json", ASSETS_DIR));
if (manifestText === undefined) {
  console.warn(
    `[karume] ${ASSETS_DIR.pathname} has no karume.json, so Irodori's self A/B check SKIPs` +
      "（cd tools/export-recipes && uv run python dist.py --pipeline irodori --model v4.1-small --resolve-refs hf）",
  );
}
/**
 * A rebuilt v4.1-small mirror reads its borrowed files from v4-small's mirror: without that mirror
 * the gate SKIPs explicitly instead of failing on the first file it cannot read (a missing asset,
 * not a defect).
 */
const borrowsFromV4 = (text: string): boolean => {
  const manifest = parseManifest(text);
  return Object.keys(manifest.models[MODEL].quants).some((quant) =>
    selectionRefs(resolveSelection(manifest, { model: MODEL, quant })).some(
      (ref) => ref.repo === V4_REPO,
    ),
  );
};
const missingV4 =
  manifestText !== undefined &&
  borrowsFromV4(manifestText) &&
  (await readTextIfPresent(new URL("karume.json", V4_ASSETS_DIR))) === undefined;
if (missingV4) {
  console.warn(
    `[karume] ${ASSETS_DIR.pathname} references ${V4_REPO}, but ${V4_ASSETS_DIR.pathname} has no` +
      " karume.json — SKIP the Irodori A/B gate (cd tools/export-recipes && uv run python dist.py" +
      " --pipeline irodori)",
  );
}
const RUNNABLE = GPU_AVAILABLE && manifestText !== undefined && !missingV4;

const readManifest = (): Manifest => parseManifest(manifestText as string);

type Observation = {
  readonly latent: Float32Array<ArrayBuffer>;
  readonly census: readonly CensusRow[];
};

/**
 * Builds the `quant` pipeline, takes the step-1 latent and aborts. The census covers `dit`'s runs
 * only (passing `onEvent` makes the DiT loop take the host path, delivering to the observer per
 * forward).
 */
const observeStep1 = async (
  source: DistributionSource,
  quant: string,
  input: AbCase,
  seed: number,
): Promise<Observation> => {
  const where = `${quant} / ${input.name} / seed ${seed}`;
  const runs: SessionDiagnostics["lastRunPipelines"][] = [];
  await using pipeline = await Pipeline.fromPretrained(source, {
    model: MODEL,
    quant,
    onRunDiagnostics: (component, diagnostics) => {
      if (component === "dit") runs.push(diagnostics.lastRunPipelines);
    },
  });
  const latent = await captureThenAbort<Float32Array<ArrayBuffer>>(where, (capture) =>
    pipeline.generateLatent({
      text: input.text,
      ...(input.caption === undefined ? {} : { caption: input.caption }),
      seed,
      durationSeconds: DURATION_SECONDS,
      onEvent: (event) => {
        if (event.kind === "denoise-step" && event.step === 1) capture(event.copyLatents().data);
      },
    }),
  );
  return { latent, census: mergeCensus(runs, `dit of ${where}`) };
};

const errorText = (cause: unknown): string =>
  cause instanceof Error ? cause.message : String(cause);

for (const input of CASES) {
  const id = `${PRACTICAL}-${input.name}`;
  test.skipIf(!RUNNABLE)(
    `e2e (real GPU): ${id}'s step-1 latent agrees with the reference seat on determinism, census and band (self A/B)`,
    async () => {
      await runRecordedCase(results, { id, failureNote: errorText }, async (recorded) => {
        const manifest = readManifest();
        const reference = referencePartnerOf(manifest, MODEL, PRACTICAL);
        const declared = quantOf(manifest, MODEL, PRACTICAL).session;
        const source = assetsSource();
        const referenceRun = await observeStep1(source, reference, input, SEED);
        const first = await observeStep1(source, PRACTICAL, input, SEED);
        const second = await observeStep1(source, PRACTICAL, input, SEED);
        const measured = measureAb(first.latent, referenceRun.latent);
        recorded.comparisons.push(
          comparisonOf(
            { output: OBSERVATION, reference, practical: PRACTICAL },
            measured,
            input.band,
          ),
        );
        const census = describeBundleCensus(first.census, declared);
        console.log(
          `[e2e] irodori A/B ${id}: ${reference} → ${PRACTICAL} ${describeAb(measured)} ` +
            `/ census ${census}`,
        );
        assertBitIdentical(first.latent, second.latent, `${id}: two runs of the practical seat`);
        assertSeatsApplied(first.census, declared, `${id}: dit of the practical seat`);
        judgeAb(measured, input.band, id);
        return { status: "pass", note: `census ${census}` };
      });
    },
  );
}

// --- Key census of the practical seat (exact dispatch counts per run) -----------------

/**
 * The utterance length (seconds) of the census generation. S is fixed small without running
 * `duration`.
 */
const CENSUS_SECONDS = 1;

/**
 * The components the census is taken from (the 2 the quant's `session` reaches — spelled as the
 * observer names them).
 */
type CensusComponent = Extract<RunComponent, "dit" | "dit-context">;
const CENSUS_COMPONENTS: readonly CensusComponent[] = ["dit", "dit-context"];
const isCensusComponent = (component: RunComponent): component is CensusComponent =>
  (CENSUS_COMPONENTS as readonly RunComponent[]).includes(component);
/**
 * The census table's phases (`dit` per denoise step, `dit_context` one run per generation).
 */
const PHASE_OF: Readonly<Record<CensusComponent, string>> = { dit: "step", "dit-context": "run" };

/**
 * The `quantize_rows` dispatch count per run of each component (`dit` = one forward,
 * `dit_context` = the single run of one generation).
 *
 * The DiT has 317 linears in the single graph before the split (k ∈ {32, 192, 512, 768, 1280,
 * 3680} — all multiples of 4, so i8-a8 eligible; the sizeBreakdown of the quantization recon). ADR
 * 0114 split the 72 condition-side K/V projections (12 blocks × text / speaker / caption × K / V)
 * out into `dit_context`, giving `dit` 245 + `dit_context` 72 — **an expected value changed
 * because the graph changed**, not a loosened count. No check on the total of 317 is kept (pinning
 * each component exactly is enough, and a total could cancel out a mix-up on one side). An i8a8
 * linear runs as the pair "`quantize_rows`, which drops the activations to per-token i8 → the
 * integer dot-product GEMM", so the expectation can be written as a function of the count. The
 * GEMM side's count (the variant key of the seat `linearCompute: "a8"`) is held by the bundle
 * census table (`helpers/census-table.ts`) — to keep one source of truth for the seat's counts.
 *
 * To re-derive after a graph change, read the measured counts from the census log line this test
 * prints (the source is `lastRunPipelines`, which does not depend on timing).
 */
const QUANTIZE_ROWS_PER_RUN: Readonly<Record<CensusComponent, number>> = {
  dit: 245,
  "dit-context": 72,
};

/**
 * Classification of pipeline keys (the spelling's source of truth is
 * `src/kernels/linear{,-i8a8}.ts` / `quantize-rows.ts`).
 */
const isLinearKey = (key: string): boolean => key.startsWith("linear:");
const isI8a8 = SEAT_SIGNATURES.linearCompute.a8;
const isQuantizeRows = (key: string): boolean => key.startsWith("quantize_rows:");

/** The classified dispatch counts of one run. */
type KeyCensus = {
  readonly i8a8Linear: number;
  readonly quantizeRows: number;
  readonly plainLinear: number;
};

const censusOf = (diagnostics: SessionDiagnostics, component: CensusComponent): KeyCensus => {
  // MUST: do not count a run without a census as "0" — the key check would silently pass vacuously
  // (requireCensus fails on undefined / empty).
  const entries = requireCensus(diagnostics.lastRunPipelines, component);
  let i8a8Linear = 0;
  let quantizeRows = 0;
  let plainLinear = 0;
  for (const entry of entries) {
    if (isLinearKey(entry.key)) {
      if (isI8a8(entry.key)) i8a8Linear += entry.dispatchCount;
      else plainLinear += entry.dispatchCount;
    } else if (isQuantizeRows(entry.key)) {
      quantizeRows += entry.dispatchCount;
    }
  }
  return { i8a8Linear, quantizeRows, plainLinear };
};

test.skipIf(!RUNNABLE)(
  `e2e (real GPU): the dit / dit_context runs use only i8a8 GEMMs and quantize_rows (key census)`,
  async () => {
    const manifest = readManifest();
    // The bundle census table's row (the key is the effective setting: the quant seat's
    // declaration passed through the family's composition). This check is the check of the `i8-a8`
    // seat itself, so a missing row fails as a gap in the table.
    const where = `irodori ${MODEL}/${PRACTICAL}`;
    const row = censusRowOf(
      "irodori",
      MODEL,
      effectiveSessionOptions("irodori", quantOf(manifest, MODEL, PRACTICAL).session, {}, where),
    );
    assert(
      row !== undefined,
      `${where}: no row in the bundle census table (helpers/census-table.ts)`,
    );
    const observed: Record<CensusComponent, KeyCensus[]> = { dit: [], "dit-context": [] };
    const runs: Record<CensusComponent, SessionDiagnostics["lastRunPipelines"][]> = {
      dit: [],
      "dit-context": [],
    };
    // No `onEvent` and no timing device: the DiT loop takes the resident path (the production
    // path), whose diagnostics arrive once per `enqueue` — the A/B cases above observe the host
    // path instead.
    await using pipeline = await Pipeline.fromPretrained(assetsSource(), {
      model: MODEL,
      quant: PRACTICAL,
      onRunDiagnostics: (component, diagnostics) => {
        if (!isCensusComponent(component)) return;
        observed[component].push(censusOf(diagnostics, component));
        runs[component].push(diagnostics.lastRunPipelines);
      },
    });
    // S is fixed with `durationSeconds` to keep it short without running `duration` (what matters
    // is "what ran", not the length). The minimal setup without a reference or a caption is enough.
    await pipeline.generateLatent({
      text: FAULT_CASE.text,
      seed: SEED,
      durationSeconds: CENSUS_SECONDS,
    });

    if (observed.dit.length === 0) throw new Error("no dit run was observed");
    // The condition-side K/V is one run per generation (if it ran per forward, ADR 0114's hoist
    // has come off).
    assertEquals(
      observed["dit-context"].length,
      1,
      "dit_context does not run exactly once per generation",
    );
    for (const component of CENSUS_COMPONENTS) {
      const first = observed[component][0];
      const count = observed[component].length;
      console.log(
        `[e2e] irodori ${MODEL}/${PRACTICAL} census: ${component} runs ${count} / ` +
          `i8a8 linear ${first.i8a8Linear} / quantize_rows ${first.quantizeRows} / ` +
          `f32 linear ${first.plainLinear}`,
      );
      // Every forward runs the same plan (uncond is a mask reduction — ADR 0047 decision 1), so
      // the counts do not vary per run. A variation is direct evidence that "only some runs fell to
      // another path".
      for (const [index, census] of observed[component].entries()) {
        assertEquals(
          census,
          first,
          `${component}: only run ${index} has a different dispatch breakdown`,
        );
      }
      assertEquals(
        first.plainLinear,
        0,
        `${component}: f32-skeleton linears are running (eligibility came off and some silently fell to the f32 path)`,
      );
    }
    // Checks the seat's variant (i8a8 GEMM) and the linears left on the reference path at the
    // exact count per run (phase → component is spelled as the census table's axis — `dit` is
    // step, `dit_context` is run).
    const checked = assertRowCensus(
      row,
      new Map(CENSUS_COMPONENTS.map((component) => [PHASE_OF[component], runs[component]])),
      (phase) => (phase === PHASE_OF.dit ? "dit" : "dit-context"),
      where,
    );
    assert(checked > 0, `${where}: no run was compared against the table's expectation`);
    for (const component of CENSUS_COMPONENTS) {
      assertEquals(
        observed[component][0].quantizeRows,
        QUANTIZE_ROWS_PER_RUN[component],
        `${component}'s quantize_rows dispatch count differs from the expectation (the DiT's linear count or fusion moved)`,
      );
    }
  },
);

// --- Fault injection (ADR 0110 decision 5-5 — asserts that the checks go red) -------------------

test.skipIf(!RUNNABLE)(
  "e2e (real GPU): fault injection ① a manifest with the practical seat's session emptied goes red on the floor",
  async () => {
    const id = `fault1-empty-session-${FAULT_CASE.name}`;
    await runRecordedCase(results, { id, failureNote: errorText }, async (recorded) => {
      const manifest = readManifest();
      const reference = referencePartnerOf(manifest, MODEL, PRACTICAL);
      const injected = overrideQuantSession(
        JSON.parse(manifestText as string),
        MODEL,
        PRACTICAL,
        {},
      );
      const referenceRun = await observeStep1(assetsSource(), reference, FAULT_CASE, SEED);
      const faulty = await withManifestOverride(
        ASSETS_DIR,
        injected,
        (source) => observeStep1(source, PRACTICAL, FAULT_CASE, SEED),
        CROSS_REPO,
      );
      const measured = measureAb(faulty.latent, referenceRun.latent);
      recorded.comparisons.push(
        comparisonOf(
          { output: OBSERVATION, reference, practical: `${PRACTICAL} (empty session)` },
          measured,
          FAULT_CASE.band,
        ),
      );
      console.log(`[e2e] irodori fault injection ① ${describeAb(measured)}`);
      assertThrows(() => judgeAb(measured, FAULT_CASE.band, id), Error, "floor failure");
    });
  },
);

test.skipIf(!RUNNABLE)(
  "e2e (real GPU): fault injection ② a different seed for only the practical seat's second run turns the determinism check red",
  async () => {
    const source = assetsSource();
    const first = await observeStep1(source, PRACTICAL, FAULT_CASE, SEED);
    const second = await observeStep1(source, PRACTICAL, FAULT_CASE, SEED + 1);
    assertThrows(
      () => assertBitIdentical(first.latent, second.latent, "fault injection ②"),
      Error,
      "not bit-identical",
    );
  },
);
