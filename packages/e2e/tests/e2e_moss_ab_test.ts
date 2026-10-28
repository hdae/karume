/**
 * The self A/B check of MOSS-Image's practical preset `f32+dit8-a8-attn8-online` (ADR 0110 decision
 * 5, ADR 0120 decision 4; ADR NNNN decisions 11 and 13, plan step 5) against the reference preset
 * `f32+dit8` — the same i8 weights with no session knob, on the same machine (no torch golden).
 *
 * ## Observations (one per preset and seed — {@link observe})
 *
 * The product's stages (`runMossTextStage` → `withMossDitPair` → one `predict`) on upstream's
 * default request (`fridge-note` with the empty negative, seed 42, 50 steps, guidance 3, shift 5).
 * The observation points sit right after the preset first takes effect (decision 5-4), one per
 * branch and graph:
 *
 * - `dit` — the positive branch's DiT output (the head rows `[128, H/16, W/16]`) of step 1;
 * - `dit_skip22` — the negative branch's (block 22 skipped) of step 1;
 * - `latent` — the latent after step 1 (upstream's CFG of the two in f32, then the Euler step —
 *   decision 5-4's DiT point; the sum of both branches). At 50 steps it is a dependent point: the
 *   initial noise is the same in both presets, so its difference is the CFG of the branches'
 *   differences (× 3 and × 2) scaled by |dt| = 0.004: 0.004 · √(9 · 1.85e-2² + 4 · 1.57e-2²) =
 *   2.5e-4, taking both branch outputs at about the noise's RMS and their differences as
 *   uncorrelated, against 2.4677e-4 measured. It carries no information the two branch points do
 *   not; it is kept because decision 5-4 names it.
 *
 * `dit_text` runs with f32 compute in every preset (plan item 15-11), so its two outputs must be
 * **bit-identical** between the presets and its census must hold no variant of a knob.
 *
 * ## The checks (per case = per size — {@link CASES})
 *
 * 1. **Determinism** (decision 6 MUST): two independent practical observations are bit-identical at
 *    every point.
 * 2. **Census** (decision 5-1 MUST), per Session of the pair: every knob's variant ran
 *    (`assertSeatsApplied`), the attention ran the online kernel at every node and the three
 *    dispatches at none (`assertAttentionSeatsComplete` — complete eligibility holds at every MOSS
 *    shape since the online kernel's live key length, ADR 0132's addendum of 2026-10-27), exactly
 *    {@link LINEAR_A8} i8a8 linear dispatches, exactly {@link ATTENTION_NODES} × the node's query-row
 *    windows online-kernel dispatches (`planAttentionOnlineWindows` — a pure function of the shape),
 *    and the online kernel in its padded form (`:kpad`) at every dispatch when the branch's
 *    S = N + T is not a multiple of 4, at none otherwise.
 * 3. **Band** (decisions 5-2 / 5-3): per observation point, at least one bit different from the
 *    reference (MUST), finite, and relRMS under the ceiling = the measured value × 2, rounded up to
 *    two significant digits, derived per case and point ({@link CASES} has the derivation and the
 *    agreement with the a8 error model). The floor is 0, although one single knob does give clearly
 *    less difference than the bundle: without `linearCompute` (a8 attention and the online kernel
 *    kept) the difference is 54 % (`dit`), 28 % (`dit_skip22`) and 53 % (`latent`) of the bundle's
 *    (1024², 2026-10-28, plan step-5 record — 1.0032e-2 / 4.3333e-3 / 1.3191e-4). ADR 0120
 *    decision 4 allows a floor above 0 then; it does not require one. The census sees that fault
 *    exactly (the same run: `linearCompute="a8"` has no variant in either Session — red), while a
 *    floor would be one more per-machine value derived from a fault configuration, so single knobs
 *    stay with the census (as Anima and Wan2.2). The other direction — `linearCompute` alone, the
 *    attention in f32 over the three dispatches — gives 89 % / 97 % / 89 % (1.6467e-2 / 1.5177e-2 /
 *    2.1866e-4): the bundle's difference is mostly the linears' a8, in quadrature (√(1.6467² +
 *    1.0032²) = 1.93 against the bundle's 1.85, unit e-2 at `dit`), and the positive branch's
 *    attention share is not negligible.
 *
 * ## Fault injections (decision 5-5 — at 1024², each must turn its check red)
 *
 * ① the practical preset with no knob (= the reference preset) → the floor at every point;
 * ② the second practical observation with another seed → determinism; ⑤ the same observation
 * against the reference → the ceiling; ③ `attentionScoreStorage` dropped (a8 attention over the
 * three dispatches) → the census (its band is recorded: whether the band alone sees it); ④ the
 * expected i8a8 linear count ± 1, and the expected attention node count ± 1 → the census (no GPU). (`attentionCompute` alone cannot be dropped:
 * `"none"` without `"a8"` is refused at Session construction — ADR 0132 decision 1.)
 *
 * ## The area cap (opt-in `KARUME_MOSS_AB_CAP=1`)
 *
 * 1696×1648 (the area cap of a 2 GiB binding — `vae-contract.ts`): S₊ = 10,981 and S₋ = 10,928 (the
 * negative branch's S is a multiple of 4 here, so its online kernel takes the whole form). The
 * practical observations run through the product's headroom check. The reference preset's pair
 * needs 12,519,585,664 B there, which a 12 GiB device refuses (plan step-5 record), although what
 * it holds fits (10,915 MiB measured): its observation runs as a measurement harness with the
 * pair's two estimates replaced by 0 (the stage's own check then probes the text contexts, the
 * part and the margin alone — the form of `e2e_moss_vae_test.ts`'s cap case). MUST: only the
 * partner's observation is taken this way; the practical preset's are the product's.
 *
 * Without a GPU adapter, the series or the stage-0 embeddings the file skips explicitly (ADR 0005).
 * Results: `outputs/verify/<environment key>/<date>_moss-ab/` (safe to delete).
 */

import { afterAll, beforeAll, describe, it } from "vite-plus/test";
import { assert, assertEquals, assertThrows } from "@std/assert";
import { parseSafetensors, type GpuContext, type SessionDiagnostics } from "@karume/runtime";
import type { ModelComponent } from "../../models/src/hub/components.ts";
import {
  mossDitContract,
  type MossDitContract,
  mossDitGraphContract,
  readMossDitAssets,
} from "../../models/src/moss/dit-contract.ts";
import {
  type MossDitComponents,
  mossClassifierFreeGuidance,
  planMossDit,
  runMossTextStage,
  withMossDitPair,
} from "../../models/src/moss/dit-stage.ts";
import { mossImageTokens } from "../../models/src/moss/dit-tokens.ts";
import { mossInitialLatents } from "../../models/src/moss/generate.ts";
import {
  MOSS_PRACTICAL_PRESET,
  MOSS_REFERENCE_PRESET,
  mossAcquireGpuOptions,
  type MossPreset,
  mossPreset,
} from "../../models/src/moss/preset.ts";
import type { MossTextEmbeddings } from "../../models/src/moss/request.ts";
import { mossEulerStep, mossSchedule } from "../../models/src/moss/sampler.ts";
import { acquireTestGpu, GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import {
  filePresent,
  floatsOf,
  MOSS_GENERATE_EMBEDS,
  MOSS_GENERATE_SERIES,
  MOSS_TEXT_EMBEDS,
  mossSeriesPresent,
  openMossSeries,
  readBuffer,
} from "../../models/tests/helpers/moss-dit.ts";
import { settleReleases } from "../../models/tests/helpers/settle-releases.ts";
import {
  assertAttentionSeatsComplete,
  assertSeatsApplied,
  type CensusRow,
  countDispatches,
  requireCensus,
  SEAT_SIGNATURES,
} from "../../runtime/tests/helpers/pipeline-census.ts";
import { planAttentionOnlineWindows } from "../../runtime/src/runtime/recipe-builders/attention.ts";
import { openResults, runRecordedCase } from "../../runtime/tests/helpers/results.ts";
import {
  type AbBand,
  type AbMeasurement,
  assertBitIdentical,
  comparisonOf,
  describeAb,
  judgeAb,
  measureAb,
} from "./helpers/ab-gate.ts";

/** Artifacts and verdicts (`outputs/verify/<environment key>/<date>_moss-ab/` — safe to delete). */
const results = openResults("moss-ab");

const CAP = process.env.KARUME_MOSS_AB_CAP === "1";

const SERIES_PRESENT = mossSeriesPresent();
const EMBEDS_PRESENT = filePresent(MOSS_TEXT_EMBEDS);

const skipped = (ok: boolean, what: string, command: string): string =>
  ok ? "" : ` (SKIP: no ${what} — ${command})`;

const REFERENCE = mossPreset(MOSS_REFERENCE_PRESET);
const PRACTICAL = mossPreset(MOSS_PRACTICAL_PRESET);

/** The request of the observations (upstream's default request — module doc). */
const PROMPT = "fridge-note";
const NEGATIVE = "negative";
const SEED = 42;
/**
 * Upstream's default step count. Step 1 runs at σ = 1 (training-grid index 0) at every step count,
 * so the branches' outputs do not depend on it; the latent after step 1 does. At 2 steps (the sha
 * rows') step 1 goes from σ 1 to 0.024 and the latent is about upstream's x₀ prediction,
 * which another seed barely moves — measured 2026-10-28 at 1024²: relRMS 1.5423e-1 against the
 * other seed, under the ceiling 1.9e-1 that the bundle's 9.0852e-2 derives, so fault ⑤ would not
 * be red there. At 50 steps step 1 goes from σ 1 to 0.996 — a short step from the noise, and another
 * seed's noise is far: fault ⑤'s `latent` value there (1.4137e+0) is the two noises' own distance
 * (√2 for two independent unit normals), not a DiT error — the branch points carry ⑤ on the DiT.
 *
 * Recorded, not checked (material for the maintainer's visual review): at 2 steps the latent after
 * step 1 is about upstream's x₀ prediction, and there the practical preset moves it 9.0852e-2 from
 * the reference — 59 % of what another seed moves it (1.5423e-1; 1024², 2026-10-28). At the cap the
 * same 2-step point gave 9.4005e-2 (no other seed measured there).
 */
const STEPS = 50;
const GUIDANCE = 3;
const SHIFT = 5;

/** The observation points (module doc). */
const POINTS = ["dit", "dit_skip22", "latent"] as const;
type Point = (typeof POINTS)[number];

/**
 * The i8a8 linear dispatches of one forward per graph (the census's exact count — decision 5-1):
 * every linear of the graph, the four at M = 1 (the time MLP's two and the two modulations)
 * included. `dit`: 42 blocks × 6 + `video_embedder`, the time MLP's two, the two modulations and
 * `final_linear_video` = 258 (the recipe's census of the graph — plan step 1); `dit_skip22`:
 * 41 × 6 + 6 = 252. Measured 2026-10-28 (RTX 3080 Ti): 258 / 252.
 */
const LINEAR_A8: Readonly<Record<"dit" | "dit_skip22", number>> = { dit: 258, dit_skip22: 252 };

/**
 * The attention nodes per graph (the recipe's census of the graph — `EXPECTED` in
 * `tools/export-recipes/moss/export_dit.py`: one per block, 42 / 41). Each node runs one
 * online-kernel dispatch per query-row window (`planAttentionOnlineWindows` — 1 at every size up
 * to the area cap with these texts, more for a long S at about 2⁴⁰ MACs per window).
 */
const ATTENTION_NODES: Readonly<Record<"dit" | "dit_skip22", number>> = { dit: 42, dit_skip22: 41 };

/** The expected i8a8 linear dispatches and attention nodes per graph (the census's exact counts). */
type CensusCounts = {
  readonly linear: Readonly<Record<"dit" | "dit_skip22", number>>;
  readonly attention: Readonly<Record<"dit" | "dit_skip22", number>>;
};

const COUNTS: CensusCounts = { linear: LINEAR_A8, attention: ATTENTION_NODES };

type Size = { readonly width: number; readonly height: number };

/** One case: a size, its bands per observation point, whether the faults run, its opt-in. */
type AbCase = {
  readonly size: Size;
  readonly bands: Readonly<Record<Point, AbBand | undefined>>;
  readonly faults: boolean;
  readonly optIn: boolean;
};

/**
 * The cases and their bands (declarations — not per-environment rows; `KARUME_REFERENCE` does not
 * rewrite them — ADR 0110 decision 4). Ceiling = the measured relRMS × 2, rounded up to two
 * significant digits, per case and point (derived independently — the 1024² values are not carried
 * to the cap); floor 0. MUST: `undefined` before the derivation (the check prints the measurement
 * and fails — `judgeAb`).
 *
 * Derivation (2026-10-28, RTX 3080 Ti, Node + Dawn, the online kernel's `sg32` form):
 *
 * | Case      | Point        | relRMS    | maxAbs    | Ceiling (× 2, up) |
 * | --------- | ------------ | --------- | --------- | ----------------- |
 * | 1024²     | `dit`        | 1.8482e-2 | 1.1560e-1 | 3.7e-2            |
 * | 1024²     | `dit_skip22` | 1.5683e-2 | 8.5594e-2 | 3.2e-2            |
 * | 1024²     | `latent`     | 2.4677e-4 | 1.3812e-3 | 5.0e-4            |
 * | 1696×1648 | `dit`        | 1.7865e-2 | 9.6400e-2 | 3.6e-2            |
 * | 1696×1648 | `dit_skip22` | 1.6198e-2 | 9.7922e-2 | 3.3e-2            |
 * | 1696×1648 | `latent`     | 2.3550e-4 | 1.1912e-3 | 4.8e-4            |
 *
 * `latent` is small against the branches: step 1 of 50 moves the noise by dt = σ₁ − 1 = −0.004, so
 * the branches' differences reach it scaled by |dt| after CFG (× 3 and × 2).
 *
 * Agreement with the theoretical value (decision 5-3 — in the precedent's form, Anima's
 * `e2e_anima_ab_test.ts`: an order-of-magnitude agreement, not a MOSS measurement). The a8 rounding
 * of a row is relRMS ≈ (peak / rms) / (127 · √12) = (peak / rms) / 440, the model whose per-layer
 * values the measured w8a8 activations matched at a median of 1.8 % (ADR 0025's quality gate —
 * docs/research/2026-08-04-attention-a8-design.md, its quality consequence), with per-layer relRMS of the 1e-2
 * order. The branch outputs' 1.6–1.8e-2 here are of that order, as Anima's 1.2–1.4e-2. The
 * attention's part alone (`linearCompute` removed — module doc, "The checks" 3: 1.0e-2 at `dit`,
 * 4.3e-3 at `dit_skip22`) lies inside the same model's range for P̃ (0.23 % for a flat row to
 * 14.5 % for a one-key row — the research note's table). No per-layer peak / rms of MOSS was
 * measured, so this checks the order only; a tighter agreement would need that measurement.
 *
 * The faults at 1024² (the same run): ① no knob — 0 at every point (bit-identical to the reference
 * preset: the floor); ⑤ seed 43 — `dit` 1.3566e+0, `dit_skip22` 1.3910e+0, `latent` 1.4137e+0
 * (37–2,827 × the ceilings); ③ `attentionScoreStorage` dropped (a8 attention over the three
 * dispatches, ③PV and S in f32 at these S) — `dit` 1.6550e-2, `dit_skip22` 1.5294e-2, `latent`
 * 2.1969e-4: 89–98 % of the bundle's and inside every band, so the band does not see the online
 * kernel taken away and the census is what does (as Wan2.2's ③ — most of the difference is the
 * linears' a8).
 */
const CASES: readonly AbCase[] = [
  {
    size: { width: 1024, height: 1024 },
    bands: {
      dit: { floor: 0, ceiling: 3.7e-2 },
      dit_skip22: { floor: 0, ceiling: 3.2e-2 },
      latent: { floor: 0, ceiling: 5.0e-4 },
    },
    faults: true,
    optIn: false,
  },
  {
    size: { width: 1696, height: 1648 },
    bands: {
      dit: { floor: 0, ceiling: 3.6e-2 },
      dit_skip22: { floor: 0, ceiling: 3.3e-2 },
      latent: { floor: 0, ceiling: 4.8e-4 },
    },
    faults: false,
    optIn: true,
  },
];

const sizeKey = ({ width, height }: Size): string => `${width}x${height}`;

/** One observation (module doc "Observations"). */
type Observation = {
  readonly points: Readonly<Record<Point, Float32Array<ArrayBuffer>>>;
  /** The refined texts of `dit_text` (positive, negative). */
  readonly text: readonly Float32Array<ArrayBuffer>[];
  readonly textCensus: readonly SessionDiagnostics["lastRunPipelines"][];
  readonly census: Readonly<Record<"dit" | "dit_skip22", readonly CensusRow[]>>;
  /** S of each branch. */
  readonly sequence: Readonly<Record<"dit" | "dit_skip22", number>>;
};

/** The stage-0 embeddings by prompt name (`moss.text_embeds`). */
const readEmbeddings = async (): Promise<ReadonlyMap<string, MossTextEmbeddings>> => {
  const file = parseSafetensors(await readBuffer(MOSS_TEXT_EMBEDS));
  return new Map(
    [...file.tensors.values()].map((view) => [
      view.name,
      { data: floatsOf(file, view.name, "text embeds").slice(), shape: view.shape },
    ]),
  );
};

/** A component whose estimate counts 0 (the cap's measurement harness for the partner — module doc). */
const unprobed = (component: ModelComponent): ModelComponent => ({
  ...component,
  estimate: (options) => ({ ...component.estimate(options), peakAccountedBytes: 0 }),
});

const errorText = (cause: unknown): string =>
  cause instanceof Error ? cause.message : String(cause);

describe.skipIf(!GPU_AVAILABLE || !SERIES_PRESENT || !EMBEDS_PRESENT)(
  `MOSS-Image self A/B check of ${MOSS_PRACTICAL_PRESET}${skipped(GPU_AVAILABLE, "GPU adapter", "a WebGPU device")}${skipped(SERIES_PRESENT, "series", MOSS_GENERATE_SERIES)}${skipped(EMBEDS_PRESENT, "stage-0 embeddings", MOSS_GENERATE_EMBEDS)}`,
  () => {
    let gpu: GpuContext | undefined;
    let components: MossDitComponents | undefined;
    let contract: MossDitContract | undefined;
    let embeds: ReadonlyMap<string, MossTextEmbeddings> | undefined;
    beforeAll(async () => {
      const series = await openMossSeries();
      const assets = await readMossDitAssets(series.components.dit);
      components = series.components;
      contract = mossDitContract(
        mossDitGraphContract(series.components, "MOSS-Image"),
        assets.ropeBase,
        assets.trainingGrid,
        "MOSS-Image",
      );
      embeds = await readEmbeddings();
      // Subgroups when available: the online kernel's subgroup form (the reference preset reads none).
      gpu = await acquireTestGpu(mossAcquireGpuOptions(PRACTICAL));
    });
    afterAll(async () => {
      if (gpu === undefined) return;
      await settleReleases(gpu);
      gpu.destroy();
    });

    const embedding = (name: string): MossTextEmbeddings => {
      assert(embeds !== undefined, "beforeAll did not finish");
      const found = embeds.get(name);
      if (found === undefined) throw new Error(`no stage-0 embedding '${name}'`);
      return found;
    };

    /** One observation of `preset` (module doc); `harness` replaces the pair's estimates by 0. */
    const observe = async (
      preset: MossPreset,
      size: Size,
      seed: number,
      harness = false,
    ): Promise<Observation> => {
      assert(gpu !== undefined && components !== undefined && contract !== undefined);
      const device = gpu;
      const state = {
        gpu: device,
        contract,
        preset,
        components: harness
          ? {
              ...components,
              dit: unprobed(components.dit),
              ditSkip22: unprobed(components.ditSkip22),
            }
          : components,
      };
      const plan = planMossDit({
        ...size,
        positive: embedding(PROMPT),
        negative: embedding(NEGATIVE),
      });
      const schedule = mossSchedule(contract.trainingGrid, STEPS, SHIFT);
      const initial = mossInitialLatents(
        { dit: plan, initial: { kind: "seed", seed } },
        contract.latentChannels,
      );
      try {
        const textCensus: SessionDiagnostics["lastRunPipelines"][] = [];
        const refined = await runMossTextStage(state, plan, (diagnostics) =>
          textCensus.push(diagnostics.lastRunPipelines),
        );
        assert(refined.negative !== undefined, "the negative branch is planned");
        const { outputs, diagnostics } = await withMossDitPair(
          state,
          plan,
          refined,
          async (pair) => ({
            outputs: await pair.predict(initial, schedule.gridIndices[0]),
            diagnostics: pair.diagnostics(),
          }),
        );
        assert(outputs.negative !== undefined && diagnostics.negative !== undefined);
        const latent = mossEulerStep(
          initial,
          mossClassifierFreeGuidance(outputs.positive, outputs.negative, GUIDANCE),
          schedule.sigmas[0],
          schedule.sigmas[1],
        );
        const image = mossImageTokens(plan.cells);
        return {
          points: { dit: outputs.positive, dit_skip22: outputs.negative, latent },
          text: [refined.positive.data, refined.negative.data],
          textCensus,
          census: {
            dit: requireCensus(diagnostics.positive.lastRunPipelines, "dit"),
            dit_skip22: requireCensus(diagnostics.negative.lastRunPipelines, "dit_skip22"),
          },
          sequence: {
            dit: image + refined.positive.tokens,
            dit_skip22: image + refined.negative.tokens,
          },
        };
      } finally {
        // The next observation's probes must not see this one's allocation.
        await settleReleases(device);
      }
    };

    /** The online kernel's dispatches, and those in its padded form (`:kpad`). */
    const onlineDispatches = (census: readonly CensusRow[]) => ({
      total: countDispatches(census, SEAT_SIGNATURES.attentionScoreStorage.none),
      padded: countDispatches(
        census,
        (key) => SEAT_SIGNATURES.attentionScoreStorage.none(key) && `${key}:`.includes(":kpad:"),
      ),
    });

    /** The census of the practical pair (module doc "The checks" 2), per Session. */
    const assertPracticalCensus = (
      observation: Observation,
      knobs: MossPreset["dit"],
      counts: CensusCounts,
      where: string,
    ): void => {
      assert(contract !== undefined);
      const depth = contract.ropeWidth;
      const heads = contract.width / depth;
      for (const graph of ["dit", "dit_skip22"] as const) {
        const census = observation.census[graph];
        const at = `${where} ${graph}`;
        assertSeatsApplied(census, knobs, at);
        assertAttentionSeatsComplete(census, knobs, at);
        const a8 = countDispatches(census, SEAT_SIGNATURES.linearCompute.a8);
        assertEquals(a8, counts.linear[graph], `${at}: linearCompute=a8 dispatches`);
        const online = onlineDispatches(census);
        const sequence = observation.sequence[graph];
        const windows = planAttentionOnlineWindows(sequence, {
          batch: heads,
          cols: sequence,
          depth,
        }).length;
        assertEquals(
          online.total,
          counts.attention[graph] * windows,
          `${at}: online-kernel dispatches (${counts.attention[graph]} nodes × ${windows} windows)`,
        );
        const padded = sequence % 4 !== 0;
        assertEquals(
          online.padded,
          padded ? online.total : 0,
          `${at}: online dispatches in the padded form at S = ${sequence}`,
        );
      }
    };

    /** `dit_text` alike in both presets: bit-identical outputs, no knob's variant in its census. */
    const assertTextUntouched = (
      practical: Observation,
      reference: Observation,
      where: string,
    ): void => {
      practical.text.forEach((values, branch) =>
        assertBitIdentical(values, reference.text[branch], `${where}: dit_text branch ${branch}`),
      );
      for (const rows of practical.textCensus) {
        assertSeatsApplied(
          rows,
          { linearCompute: "f32", attentionCompute: "f32", attentionScoreStorage: "f32" },
          `${where}: dit_text`,
        );
      }
    };

    for (const testCase of CASES) {
      const key = sizeKey(testCase.size);
      describe.skipIf(testCase.optIn && !CAP)(
        `${key}${testCase.optIn ? " (opt-in KARUME_MOSS_AB_CAP=1)" : ""}`,
        () => {
          let reference: Observation | undefined;
          let first: Observation | undefined;
          let second: Observation | undefined;
          beforeAll(async () => {
            reference = await observe(REFERENCE, testCase.size, SEED, testCase.optIn);
            first = await observe(PRACTICAL, testCase.size, SEED);
            second = await observe(PRACTICAL, testCase.size, SEED);
          });
          const observed = () => {
            assert(reference !== undefined && first !== undefined && second !== undefined);
            return { reference, first, second };
          };

          it(`${key}: determinism, census, dit_text untouched, band per branch and graph`, async () => {
            const id = `${PRACTICAL.name}-${key}`;
            await runRecordedCase(results, { id, failureNote: errorText }, (recorded) => {
              const { reference: partner, first: one, second: two } = observed();
              const measured = POINTS.map((point) => {
                const result = measureAb(one.points[point], partner.points[point]);
                recorded.comparisons.push(
                  comparisonOf(
                    { output: point, reference: REFERENCE.name, practical: PRACTICAL.name },
                    result,
                    testCase.bands[point],
                  ),
                );
                console.log(`[moss-ab] ${key} ${point}: ${describeAb(result)}`);
                return [point, result] as const;
              });
              const counts = (["dit", "dit_skip22"] as const).map((graph) => {
                const online = onlineDispatches(one.census[graph]);
                return (
                  `${graph} S = ${one.sequence[graph]}: linear a8 ×` +
                  `${countDispatches(one.census[graph], SEAT_SIGNATURES.linearCompute.a8)},` +
                  ` online ×${online.total} (padded ×${online.padded})`
                );
              });
              console.log(`[moss-ab] ${key} census: ${counts.join(" / ")}`);
              for (const point of POINTS) {
                assertBitIdentical(
                  one.points[point],
                  two.points[point],
                  `${key} ${point}: two runs`,
                );
              }
              assertPracticalCensus(one, PRACTICAL.dit, COUNTS, key);
              assertTextUntouched(one, partner, key);
              const failures: string[] = [];
              for (const [point, result] of measured) {
                try {
                  judgeAb(result, testCase.bands[point], `${key} ${point}`);
                } catch (cause) {
                  failures.push(errorText(cause));
                }
              }
              assert(failures.length === 0, failures.join("\n"));
              return Promise.resolve({ status: "pass", note: counts.join(" / ") });
            });
          });

          if (!testCase.faults) return;

          /** Records a fault's measured points (results `comparisons`) and runs its assertions. */
          const recordFault = (
            id: string,
            practical: string,
            body: (record: (point: Point, faulty: Observation) => AbMeasurement) => Promise<void>,
          ): Promise<void> =>
            runRecordedCase(results, { id: `${key}-${id}`, failureNote: errorText }, (recorded) =>
              body((point, faulty) => {
                const { reference: partner } = observed();
                const result = measureAb(faulty.points[point], partner.points[point]);
                recorded.comparisons.push(
                  comparisonOf(
                    { output: point, reference: REFERENCE.name, practical },
                    result,
                    testCase.bands[point],
                  ),
                );
                console.log(`[moss-ab] ${key} ${id} ${point}: ${describeAb(result)}`);
                return result;
              }).then(() => ({ status: "pass" as const })),
            );

          it(`${key} fault ①: the practical preset without a knob is red at the floor of every point`, async () => {
            await recordFault("fault1-no-knob", `${PRACTICAL.name} (no knob)`, async (record) => {
              const faulty = await observe({ ...PRACTICAL, dit: {} }, testCase.size, SEED);
              for (const point of POINTS) {
                const result = record(point, faulty);
                assertThrows(
                  () => judgeAb(result, testCase.bands[point], `${key} fault ① ${point}`),
                  Error,
                  "床の失敗",
                );
              }
            });
          });

          it(`${key} fault ② / ⑤: another seed breaks determinism and leaves the ceiling of every point`, async () => {
            const { first: one } = observed();
            await recordFault(
              "fault2-5-other-seed",
              `${PRACTICAL.name} (seed ${SEED + 1})`,
              async (record) => {
                const shifted = await observe(PRACTICAL, testCase.size, SEED + 1);
                for (const point of POINTS) {
                  assertThrows(
                    () => assertBitIdentical(one.points[point], shifted.points[point], "fault ②"),
                    Error,
                    "ビット一致しない",
                  );
                  const result = record(point, shifted);
                  assertThrows(
                    () => judgeAb(result, testCase.bands[point], `${key} fault ⑤ ${point}`),
                    Error,
                    "上限の失敗",
                  );
                }
              },
            );
          });

          it(`${key} fault ③: without attentionScoreStorage the census is red (the band is recorded)`, async () => {
            const { attentionScoreStorage: _dropped, ...withoutOnline } = PRACTICAL.dit;
            await recordFault(
              "fault3-no-online",
              `${PRACTICAL.name} (attentionScoreStorage dropped)`,
              async (record) => {
                const faulty = await observe(
                  { ...PRACTICAL, dit: withoutOnline },
                  testCase.size,
                  SEED,
                );
                for (const point of POINTS) {
                  const result = record(point, faulty);
                  let verdict = "inside";
                  try {
                    judgeAb(result, testCase.bands[point], `${key} fault ③ ${point}`);
                  } catch (cause) {
                    verdict = `red (${errorText(cause)})`;
                  }
                  console.log(`[moss-ab] ${key} fault ③ ${point}: band ${verdict}`);
                }
                assertThrows(
                  () => assertPracticalCensus(faulty, PRACTICAL.dit, COUNTS, `${key} fault ③`),
                  Error,
                  'attentionScoreStorage="none"',
                );
              },
            );
          });

          it(`${key} fault ④: the expected i8a8 linear or attention node count ± 1 is red on the real census`, () => {
            const { first: one } = observed();
            for (const delta of [-1, 1]) {
              for (const graph of ["dit", "dit_skip22"] as const) {
                const linear = { ...LINEAR_A8, [graph]: LINEAR_A8[graph] + delta };
                assertThrows(
                  () =>
                    assertPracticalCensus(
                      one,
                      PRACTICAL.dit,
                      { ...COUNTS, linear },
                      `${key} fault ④`,
                    ),
                  Error,
                  "linearCompute=a8 dispatches",
                );
                const attention = { ...ATTENTION_NODES, [graph]: ATTENTION_NODES[graph] + delta };
                assertThrows(
                  () =>
                    assertPracticalCensus(
                      one,
                      PRACTICAL.dit,
                      { ...COUNTS, attention },
                      `${key} fault ④`,
                    ),
                  Error,
                  "online-kernel dispatches",
                );
              }
            }
          });
        },
      );
    }
  },
);
