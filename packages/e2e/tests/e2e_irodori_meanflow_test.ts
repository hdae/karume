/**
 * Real-GPU checks of the **MeanFlow** distribution `models/karume-irodori-v4.1-small-mf/` (ADR
 * 0150) — upstream's MeanFlow student of v4.1-small: one conditional `dit` forward per step that
 * also takes the interval embedding `delta_embed`, 4 steps, no CFG. It follows the structure of the
 * v4-small checks, for what this distribution makes new:
 *
 * - **per-graph parity** (`e2e_irodori_test.ts`'s form) for the two graphs whose bytes are new,
 *   `dit` (8 cases — the RF case table with an interval each, plus `dit-cond-1s-wide`, which differs
 *   from `dit-cond-1s` in the interval alone) and `dit-context`, against the torch goldens of
 *   `goldens/irodori-v4.1-small-mf/<target>/` (written by `tools/export-recipes/irodori/export.py`).
 *   The other seven graphs are the published v4-small / v4.1-small containers this distribution
 *   borrows, checked by those distributions' own suites;
 * - **latent parity** (`e2e_irodori_latent_test.ts`'s form) per storage series (`f32` / `f16` /
 *   `i8` / `i8+dit4`) against the full-loop goldens `goldens/irodori-v4.1-small-mf*\/pipeline/`
 *   (written by `irodori/pipeline_ref.py` only after its comparison with upstream's
 *   `sample_euler_meanflow`): S and the forward count exactly, z within a tolerance derived per
 *   series from measurement;
 * - the **premise** that the distribution's declaration and the golden's generation conditions agree
 *   (`flowParameterization`, `steps`, and the t / interval sequences **bit for bit** against the TS
 *   host's — the f32 values of upstream's `linspace`);
 * - **determinism** and **path identity**: two generations with the same seed are byte-identical,
 *   and the host path (`onEvent`) gives the same bytes as the resident path, with one `denoise-step`
 *   event per step (the MUST of `runDitLoopResident`, on the MeanFlow loop).
 *
 * The parity cases are upstream parity, so they are opt-in (`KARUME_UPSTREAM_PARITY=1`); the premise,
 * determinism and path cases run by default. The WAV sha256 rows are `e2e_irodori_meanflow_wav_test.ts`.
 *
 * MUST: the mirror and the goldens are local assets of a real-GPU machine. An environment without
 * the mirror (or a mirror it borrows from), without the goldens, or without a GPU SKIPs explicitly
 * with the missing path (ADR 0005); a golden absent from a complete fetch of the pinned test assets
 * FAILs (ADR 0139).
 */

import { readFile } from "@karume/test-support/fs";
import { goldenDir, testAssetsFetched } from "@karume/test-support/test-assets";
import { test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { parseManifest } from "@karume/hub";
import { acquireGpu, parseSafetensors, prepareContainer, type Tensor } from "../../runtime/mod.ts";
import {
  compareTensors,
  formatAllclose,
  type Tolerance,
} from "../../runtime/src/reference/allclose.ts";
import { assertAdapterMatchesEnvironment } from "../../runtime/tests/helpers/environment.ts";
import { ioTensor } from "../../runtime/tests/helpers/golden-io.ts";
import { openResults, runRecordedCase } from "../../runtime/tests/helpers/results.ts";
import { Pipeline } from "../../models/irodori.ts";
import { parseIrodoriPipelineConfig } from "../../models/src/irodori/config.ts";
import { meanflowIntervals, meanflowSchedule } from "../../models/src/irodori/host/sampler.ts";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import {
  mirrorDirectory,
  mirrorHasModel,
  mirrorRoot,
  openSelectedContainer,
  selectMirrorComponent,
} from "../../models/tests/helpers/mirror.ts";
import { readTextIfPresent } from "../../models/tests/helpers/read-if-present.ts";
import {
  type GoldenCase,
  goldenPresent,
  pipelineGoldenDir,
  readCase,
  worstDifference,
} from "./helpers/irodori-assets.ts";
import { UPSTREAM_PARITY, UPSTREAM_PARITY_OPT_IN } from "./helpers/upstream-parity.ts";

/** The MeanFlow distribution (it borrows its other graphs from the v4-small / v4.1-small mirrors). */
const MIRROR = mirrorRoot("karume-irodori-v4.1-small-mf");
const MODEL = "v4.1-small-mf";
const MIRROR_READY = mirrorHasModel(MIRROR, MODEL);
const FETCHED = testAssetsFetched();
const FETCH_HINT =
  "fetch: node tools/test-assets/fetch.ts (goldens/irodori-v4.1-small-mf*/ — until they are in the " +
  "pinned dataset, stage them from outputs/series/ as tools/test-assets/README.md describes)";

if (!MIRROR_READY) {
  console.warn(
    `[karume] Irodori MeanFlow e2e SKIPs: the mirror ${MIRROR.pathname} with model '${MODEL}' is ` +
      "absent (or a mirror it borrows from is) — cd tools/export-recipes && uv run python dist.py " +
      "--pipeline irodori --model v4.1-small-mf --resolve-refs hf",
  );
}

const lostAsset = (paths: readonly string[]): Error =>
  new Error(
    `${paths.join(" / ")} absent, but outputs/test-assets/ holds a complete fetch of the pinned ` +
      "revision: the test-asset dataset at the pin lacks it",
  );

// --- per-graph parity (dit / dit-context) --------------------------------------------------

/**
 * The tolerance of the MeanFlow `dit` (f32 container) against the torch CPU goldens.
 *
 * Measured (a plain comparison with `atol=rtol=0`, 2026-11-14, RTX 3080 Ti / Node):
 *
 * | Case               | S   | delta | maxAbs  | maxRel  |
 * | ------------------ | --- | ----- | ------- | ------- |
 * | dit-cond-min       | 2   | 0.25  | 2.55e-5 | 1.26e-3 |
 * | dit-cond-1s        | 25  | 0.25  | 1.53e-5 | 7.08e-4 |
 * | dit-cond-1s-wide   | 25  | 0.9   | 1.40e-5 | 3.10e-3 |
 * | dit-cond-late      | 25  | 0.3   | 6.20e-6 | 2.71e-3 |
 * | dit-uncond-text    | 25  | 0.25  | 6.91e-6 | 6.64e-4 |
 * | dit-uncond-speaker | 25  | 0.25  | 2.16e-5 | 3.29e-3 |
 * | dit-uncond-caption | 25  | 0.25  | 1.22e-5 | 1.92e-2 |
 * | dit-cond-max       | 750 | 0.25  | 2.24e-5 | 1.41e-1 |
 *
 * atol 2e-4 is 7.8 times the measured worst 2.55e-5 (the RF `dit`'s 5e-5 is 6.0 times its own
 * worst, measured on another machine — not reused, ADR 0027). The interval embedding changes the
 * output by 2.6 between `dit-cond-1s` and `dit-cond-1s-wide` (`dit_delta_divergence` of the export),
 * four orders of magnitude above this.
 */
const MEANFLOW_DIT_TOLERANCE: Tolerance | undefined = { atol: 2e-4, rtol: 1e-6 };

/**
 * The tolerance of the MeanFlow `dit-context` (24 outputs, f32 container). Measured (the same run):
 * maxAbs 4.58e-5 over the 24 outputs of `dit-cond-min` (a range of O(10–180), so maxRel up to 3.5 at
 * near-zero elements is not the deciding measure). atol 4e-4 is 8.7 times that.
 */
const MEANFLOW_DIT_CONTEXT_TOLERANCE: Tolerance | undefined = { atol: 4e-4, rtol: 1e-6 };

/** The golden root of the per-graph goldens (the f32 series). */
const GRAPH_GOLDENS = goldenDir("irodori-v4.1-small-mf");

/** Target directory → the manifest's weights key, and the cases the export writes. */
const TARGETS: Readonly<
  Record<string, { readonly graph: string; readonly cases: readonly string[] }>
> = {
  dit: {
    graph: "dit",
    cases: [
      "dit-cond-1s",
      "dit-cond-1s-wide",
      "dit-cond-late",
      "dit-cond-max",
      "dit-cond-min",
      "dit-uncond-caption",
      "dit-uncond-speaker",
      "dit-uncond-text",
    ],
  },
  "dit-context": { graph: "dit_context", cases: ["dit-cond-min"] },
};

const toleranceOf = (target: string): Tolerance | undefined =>
  target === "dit" ? MEANFLOW_DIT_TOLERANCE : MEANFLOW_DIT_CONTEXT_TOLERANCE;

const ioUrl = (target: string, name: string): URL =>
  new URL(`${target}/io.${name}.safetensors`, GRAPH_GOLDENS);

const graphResults = openResults("irodori-meanflow-golden");

for (const [target, { graph, cases }] of Object.entries(TARGETS)) {
  for (const name of cases) {
    const url = ioUrl(target, name);
    const present = goldenPresent(url);
    if (!present && !FETCHED) {
      console.warn(
        `[karume] Irodori MeanFlow ${target}/${name} SKIPs: missing ${url.pathname} (${FETCH_HINT})`,
      );
    }
    test.skipIf(!GPU_AVAILABLE || !MIRROR_READY || (!present && !FETCHED) || !UPSTREAM_PARITY)(
      `Irodori MeanFlow golden comparison: ${target} / ${name} (real GPU / torch CPU expected values)${UPSTREAM_PARITY_OPT_IN}`,
      async () => {
        if (!present) throw lostAsset([url.pathname]);
        await runRecordedCase(
          graphResults,
          { id: `${target}/${name}` },
          async ({ measurements }) => {
            const opened = await openSelectedContainer(
              await selectMirrorComponent(MIRROR, graph, { model: MODEL, quant: "f32" }),
            );
            const parsed = prepareContainer(opened, graph);
            const bytes = await readFile(url);
            const io = parseSafetensors(
              bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
            );
            const expectedKeys = [
              ...parsed.graph.inputs.map((spec) => `input.${spec.name}`),
              ...parsed.graph.outputs.map((_, index) => `output.${index}`),
            ].sort();
            assertEquals(
              [...io.tensors.keys()].sort(),
              expectedKeys,
              `tensor keys of ${url.pathname}`,
            );
            const inputs: Record<string, Tensor> = {};
            for (const spec of parsed.graph.inputs) {
              const view = io.tensors.get(`input.${spec.name}`);
              assert(view !== undefined, `input.${spec.name} is not in the golden`);
              inputs[spec.name] = ioTensor(io, view, spec.dtype);
            }
            const tolerance = toleranceOf(target);
            const gpu = await acquireGpu();
            try {
              assertAdapterMatchesEnvironment(gpu);
              const session = await parsed.createContainerSession(gpu);
              try {
                const outputs = await session.run(inputs);
                const failures: string[] = [];
                parsed.graph.outputs.forEach((output, index) => {
                  const view = io.tensors.get(`output.${index}`);
                  assert(view !== undefined, `output.${index} is not in the golden`);
                  const declared = parsed.graph.values[output].dtype;
                  assertEquals(outputs[output].shape, view.shape, `${output}: shape`);
                  const report = compareTensors(
                    outputs[output],
                    ioTensor(io, view, declared),
                    tolerance ?? { atol: 0, rtol: 0 },
                  );
                  measurements.push({
                    output,
                    maxAbs: report.maxAbsError,
                    maxRel: report.maxRelError,
                    tolerance: tolerance ?? { atol: 0, rtol: 0 },
                    stage: "karume",
                  });
                  if (tolerance !== undefined && !report.pass) {
                    failures.push(`${output}: ${formatAllclose(report)}`);
                  }
                });
                const worst = Math.max(...measurements.map((entry) => entry.maxAbs));
                console.log(
                  `[e2e] irodori ${MODEL} ${target}/${name}: maxAbs ${worst.toExponential(3)}`,
                );
                if (tolerance === undefined) {
                  throw new Error(
                    `${target}: the tolerance is not derived yet (measured maxAbs ${worst.toExponential(3)}) — ` +
                      "fill in the table of the target's tolerance constant from every case's log",
                  );
                }
                assert(failures.length === 0, `${target}/${name}: ${failures.join(" / ")}`);
              } finally {
                await session.dispose();
              }
            } finally {
              gpu.destroy();
            }
          },
        );
      },
    );
  }
}

// --- latent parity, premise, determinism, paths --------------------------------------------

/** The fields of a MeanFlow golden `meta.json` this file reads (the exporter owns the shape). */
type MeanFlowMeta = {
  readonly flowParameterization: string;
  readonly steps: number;
  readonly frameRate: number;
  readonly tSchedule: readonly number[];
  readonly deltaSchedule: readonly number[];
  readonly tEmbedDim: number;
  readonly caps: { readonly text: number; readonly speaker: number; readonly caption: number };
  readonly cases: Readonly<Record<string, GoldenCase>>;
};

/** One storage series (the quant preset × its full-loop golden series — one to one). */
type MeanFlowSeries = {
  readonly name: string;
  readonly seriesRoot: string;
  /** The tolerance of `z` (derived per series from measurement — `undefined` = not derived yet). */
  readonly zAtol: number | undefined;
};

/**
 * The `z` tolerances per series. Measured (a plain comparison with `atol = rtol = 0` — the values
 * this check logs every time; 2026-11-14, RTX 3080 Ti / Node; S 152 / 115, 4 forwards each):
 *
 * | Series  | full z maxAbs | no-ref z maxAbs | \|z\| max | atol   | Ratio to worst |
 * | ------- | ------------- | --------------- | ---------- | ------ | -------------- |
 * | f32     | 1.3989e-4     | 1.1706e-4       | 4.22       | 1e-3   | 7.1            |
 * | f16     | 3.6764e-4     | 5.3048e-5       | 4.22       | 2.5e-3 | 6.8            |
 * | i8      | 8.2552e-5     | 8.7261e-5       | 4.22       | 6e-4   | 6.9            |
 * | i8+dit4 | 2.7826e-4     | 7.6175e-5       | 4.22       | 2e-3   | 7.2            |
 *
 * The four steps leave little room for the differences to accumulate (the RF loop's 40 steps took
 * them to 1e-4–2e-3), and the golden is taken with each series' own rounded weights, so this is
 * the implementation difference only. A wrong formula (the schedule, the interval, the update)
 * shows at O(1): dropping the interval alone moves z by 3.4–4.5 (`deltaEffectMaxAbs` in the
 * golden's `meta.json`).
 *
 * MUST: derived per series from the raw measurement (the pattern of ADR 0027), never reused from the
 * RF series or from another MeanFlow series. Re-derive by setting the constant back to `undefined`,
 * running with `KARUME_UPSTREAM_PARITY=1` and taking 5–10 times the logged worst.
 */
const SERIES: readonly MeanFlowSeries[] = [
  { name: "f32", seriesRoot: "irodori-v4.1-small-mf", zAtol: 1e-3 },
  { name: "f16", seriesRoot: "irodori-v4.1-small-mf-f16", zAtol: 2.5e-3 },
  { name: "i8", seriesRoot: "irodori-v4.1-small-mf-i8", zAtol: 6e-4 },
  { name: "i8+dit4", seriesRoot: "irodori-v4.1-small-mf-i4", zAtol: 2e-3 },
];

/** The utterance length (seconds) of the determinism and path checks (`duration` does not run). */
const SHORT_SECONDS = 1;

const manifestText = MIRROR_READY
  ? await readTextIfPresent(new URL("karume.json", MIRROR))
  : undefined;

const openPipeline = async (quant: string): Promise<Pipeline> =>
  await Pipeline.fromPretrained(mirrorDirectory(MIRROR), { model: MODEL, quant });

const f32Bits = (values: readonly number[] | Float32Array<ArrayBuffer>): number[] =>
  Array.from(new Uint32Array(Float32Array.from(values).buffer));

for (const series of SERIES) {
  const directory = pipelineGoldenDir(series.seriesRoot);
  const metaUrl = new URL("meta.json", directory);
  const metaText = await readTextIfPresent(metaUrl);
  const readMeta = (): MeanFlowMeta => {
    if (metaText === undefined) throw lostAsset([metaUrl.pathname]);
    return JSON.parse(metaText) as MeanFlowMeta;
  };
  const seated =
    manifestText !== undefined &&
    Object.hasOwn(parseManifest(manifestText).models[MODEL].quants, series.name);
  const available = seated && (metaText !== undefined || FETCHED);
  if (seated && metaText === undefined && !FETCHED) {
    console.warn(
      `[karume] Irodori MeanFlow latent (${series.name}) SKIPs: missing ${metaUrl.pathname} (${FETCH_HINT})`,
    );
  }
  const runnable = GPU_AVAILABLE && seated;

  for (const caseName of ["full", "no-ref"] as const) {
    test.skipIf(!(GPU_AVAILABLE && available && UPSTREAM_PARITY))(
      `e2e (real GPU): the MeanFlow latent matches the full-loop golden (${series.name} / ${caseName})${UPSTREAM_PARITY_OPT_IN}`,
      async () => {
        const expected = readMeta().cases[caseName];
        const caseUrl = new URL(`case.${caseName}.safetensors`, directory);
        if (!goldenPresent(caseUrl)) {
          throw new Error(
            `inconsistent golden set: ${metaUrl.pathname} is present but ${caseUrl.pathname} is absent`,
          );
        }
        const golden = await readCase(directory, caseName);
        await using pipeline = await openPipeline(series.name);
        const started = performance.now();
        const latent = await pipeline.generateLatent({
          text: expected.text,
          ...(expected.caption === "" ? {} : { caption: expected.caption }),
          ...(expected.reference === null
            ? {}
            : { speaker: { latent: golden("reference_latent") } }),
          initialNoise: golden("noise"),
        });
        const z = golden("z");
        const { maxAbs, at } = worstDifference(latent.data, z);
        console.log(
          `[e2e] irodori ${MODEL}/${series.name} ${caseName}: ${((performance.now() - started) / 1000).toFixed(1)}s / ` +
            `S ${latent.frames} / forwards ${latent.forwards} / z maxAbs ${maxAbs.toExponential(4)} ` +
            `(atol ${series.zAtol ?? "not derived"} / |z| max ${expected.zAbsMax})`,
        );
        assertEquals(latent.frames, expected.S, `${caseName}: S differs from the golden`);
        assertEquals(
          latent.forwards,
          expected.forwards,
          `${caseName}: dit's forward count differs`,
        );
        if (series.zAtol === undefined) {
          throw new Error(
            `the z tolerance of ${series.name} is not derived yet (measured maxAbs ${maxAbs.toExponential(4)}) — ` +
              "fill in this series' zAtol from the log of every case (do not reuse another series' value)",
          );
        }
        if (maxAbs > series.zAtol) {
          throw new Error(
            `${caseName}: the latent differs from the golden by ${maxAbs.toExponential(4)} (tolerance ${series.zAtol}); ` +
              `worst at frame ${Math.floor(at / latent.latentDim)}: golden ${z[at]} → measured ${latent.data[at]} — ` +
              "check the schedule / interval embedding / update before the tolerance",
          );
        }
      },
    );
  }

  test.skipIf(!available)(
    `e2e (assets): the MeanFlow declaration matches the golden's generation conditions bit for bit (${series.name})`,
    () => {
      const meta = readMeta();
      const config = parseIrodoriPipelineConfig(
        parseManifest(manifestText as string).models[MODEL].pipelineConfig,
      );
      assertEquals(config.flowParameterization, "meanflow");
      assertEquals(meta.flowParameterization, "meanflow");
      assertEquals(config.steps, meta.steps, "steps");
      assertEquals(config.frameRate, meta.frameRate, "frameRate");
      assertEquals(config.timestepEmbedDim, meta.tEmbedDim, "timestepEmbedDim");
      assertEquals(
        [config.maxTextLen, config.speakerRows, config.maxCaptionLen],
        [meta.caps.text, meta.caps.speaker, meta.caps.caption],
        "condition caps",
      );
      // The host's t sequence and intervals are upstream's f32 values themselves (not within a
      // tolerance — the doc of `meanflowSchedule`).
      const schedule = meanflowSchedule(config.steps);
      assertEquals(f32Bits(schedule), f32Bits(meta.tSchedule), "t schedule");
      assertEquals(f32Bits(meanflowIntervals(schedule)), f32Bits(meta.deltaSchedule), "intervals");
      for (const [name, entry] of Object.entries(meta.cases)) {
        assertEquals(entry.forwards, config.steps, `${name}: one forward per step`);
      }
    },
  );

  test.skipIf(!runnable)(
    `e2e (real GPU): MeanFlow generations are deterministic and both loop paths give the same bytes (${series.name})`,
    async () => {
      await using pipeline = await openPipeline(series.name);
      const request = {
        text: "本日はお越しいただき、誠にありがとうございます。",
        seed: 7,
        durationSeconds: SHORT_SECONDS,
      };
      const first = await pipeline.generateLatent(request);
      const second = await pipeline.generateLatent(request);
      const steps: number[] = [];
      const observed = await pipeline.generateLatent({
        ...request,
        onEvent: (event) => {
          if (event.kind === "denoise-step") steps.push(event.step);
        },
      });
      const config = parseIrodoriPipelineConfig(
        parseManifest(manifestText as string).models[MODEL].pipelineConfig,
      );
      assertEquals(first.forwards, config.steps, "one conditional forward per step (no CFG)");
      assertEquals(
        new Uint8Array(second.data.buffer),
        new Uint8Array(first.data.buffer),
        "the latent changed with the same seed and the same request",
      );
      assertEquals(
        steps,
        Array.from({ length: config.steps }, (_, index) => index + 1),
      );
      assertEquals(observed.forwards, first.forwards);
      assertEquals(
        new Uint8Array(observed.data.buffer),
        new Uint8Array(first.data.buffer),
        "the host path (onEvent) and the resident path differ (the MUST of runDitLoopResident)",
      );
    },
  );
}
