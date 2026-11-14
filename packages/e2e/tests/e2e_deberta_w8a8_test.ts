// The **w8a8 mirror check** of the real-weight DeBERTa-v2 (real GPU, ADR 0026 decision 3).
//
// Where `e2e_deberta_test.ts` compares w8 (i8 storage, f32 compute) with the normal torch CPU
// goldens `io.<case>`, this one compares w8a8 (`SessionOptions.linearCompute: "a8"` — the
// activations are quantised to per-token i8 and computed with integer dot products) with **the
// torch mirror goldens `io-i8a8.<case>`** taken on the same inputs (`--act-quant` — the numeric
// source of truth is the exporter's `karume.act_quant`). The two are split into separate files
// because their expected values are different series; reading one's goldens in the other's check
// contaminates them with the activation quantisation as a whole (the same reason the mirror side
// has its own prefix — `ACT_IO_PREFIX` of `deberta/export.py`).
//
// The target is `outputs/series/deberta-i8/full-24layer/` (outside the repository — `outputs/` in
// `.gitignore`). The mirror io is written into the same directory by `--act-quant` of the same
// export.
//
// ## Where the detection power sits (the "detection limit" of ADR 0026)
//
// Activation rounding is a discontinuous function: an upstream difference at the 1e-5 level causes
// ±1-step jumps at rounding boundaries, which saturate within a few layers. In the deep layers the
// GPU and the torch mirror become "different samples of the same distribution", and at the last
// layer they cannot be told apart from the f32 route vs the mirror (historical values: i8a8 1.46 /
// f32 route 1.33). Applying a plain "5 to 10 × the measurement" to every output would let the f32
// route through as well and make it tautological. So the detection power is split into three:
//
// 1. **A strict tolerance on output.0 / output.1** ({@link STRICT_TOLERANCE}) — the only place a
//    discrimination band remains is the exit of the first layer (historical values: i8a8 5.72e-6 vs
//    f32 route 5.76e-2 = 10,058 ×). If `linearCompute` silently falls back to the f32 route, the
//    difference **opens up** to the 5.76e-2 level and it fails here.
// 2. **The pipeline-key census** ({@link EXPECTED_CENSUS}) — the run actually drives the i8a8 GEMM
//    and `quantize_rows` once per linear and never runs any other linear kernel. Where 1 is about
//    distributions, this one is a direct observation of the execution itself.
// 3. **The collapse ceilings from output.2 on** ({@link COLLAPSE_CEILINGS}, verdict by
//    {@link judgeCollapse}) — not numeric parity; it only detects collapse. The metric is the
//    per-output relative RMS error ‖gpu − golden‖₂ / ‖golden‖₂ ≤ that output's ceiling (0.020 for
//    output.2 to 0.33 for output.24), plus zero non-finite values (both sides, every element).
//    - What it catches (fault injections that corrupted real-GPU dumps and ran them through the same
//      verdict — red on every one of the 92 outputs of 4 cases × output.2–24): all elements 0
//      (1.0), a sign flip (≈ 2.0), a NaN / +Inf in a single element, all elements × 2 (1.00–1.02),
//      all elements × 1.5 (0.50–0.53).
//    - What it does not catch: a × 1.25 scaling of the deep outputs (11 of the 12 outputs of
//      output.22–24 are green — 81 of the 92 outputs are red), a jump of a single element in an
//      intermediate layer, a silent fallback to the f32 route (the dump of the GPU f32 route stays
//      within the ceilings on every one of the 92 outputs — 0.016–0.120). These are held by 1 and 2
//      and by the atol=0 numeric contract of `tests/gpu_i8a8_test.ts` (the 24 layers go through the
//      same pipelines of the same shape, so a defect in a shallow layer shows on output.1 of 1 as
//      well).
//    - How the ceilings are derived: per output, the worst of the legitimate samples (GPU a8, 6
//      different samples of the torch mirror retaken on this machine's CPU, and a perturbation
//      ensemble of 8 per case with relative ±2^-23 noise injected at the mirror's embedding exit) ×
//      2, rounded up to 2 significant digits. The breakdown of the samples and the table of each
//      output's worst and ceiling are in the addendum of ADR 0026 (2026-10-07).
//    - Why the maxAbs atol 3 was dropped: atol 3 is about 2 × the historical value 1.46 of ADR 0026
//      for **the last layer (output.24) only**, and it was being applied as maxAbs down to the
//      intermediate layers. In the intermediate layers, at the sign branching point of an outlier
//      channel (channel 686, \|ref\| max 28.8), a flipped step of the activation quantisation moves
//      one element a lot, and the same element moves by 7.17 even between different samples of the
//      CPU torch mirror (output.19 of case2 on the RTX 3080 Ti turning red at 3.54 was this shape —
//      not a GPU defect).
//
// ## Where the values come from
//
// The numbers of 1 are **historical values** of ADR 0026 (2026-08-03, RTX 3080 Ti, a plain
// comparison with the torch mirror), not re-derived in this environment. The numbers of 3 were
// re-derived from measurement on 2026-10-07 in this environment (RTX 3080 Ti, Ryzen 5 7600). This
// check keeps the maxAbs of output.0 / 1 (`measurements`) and the relative RMS error and maxAbs
// from output.2 on (`comparisons`) in results.json under `outputs/verify/` (even a failing run
// measures every output before failing), so a re-derivation reads those measurements.
//
// ## The mirror goldens are not retaken
//
// The mirror goldens (`io-i8a8.<case>`) were taken on 2026-09-05 on the CPU of the machine before
// the hardware swap. The strict tolerance of 1 depends on the rounding direction of the
// quantisation steps of layer 0 on the CPU that took the goldens: retaking them on this machine's
// CPU moves output.1 of case0 / padded by 1.285e-3 from the saved version (the GPU is on the same
// side as the old machine's CPU), and 1 with atol 5e-5 turns red. Retaking is not a fix, so the
// mirror goldens are not rebuilt on another machine's CPU.
//
// ## Variants that are not declared
//
// `deberta-i8/sbv2-22layer` outputs only the last layer, and its output lies only in the saturated
// region (outside the discrimination band of 1) — only the collapse ceilings of 3 and the census of
// 2 could be applied, and the collapse ceilings are derived only from full-24layer samples.
// `deberta-i8/dev-2layer` has no measurement either. To add one, measure with a plain comparison
// with `atol=rtol=0` and different samples first, then declare it (no values are invented).
//
// Where the assets are absent it SKIPs (independently of ADR 0005's gate, which looks at whether a
// GPU adapter exists). When the mirror is present **only in part** it is a FAIL rather than a SKIP
// (the "asset completeness" test below).
//
// ## An implementation-time series check (ADR 0139 decision 6)
//
// No distribution ships the full-24layer graph, and the distribution never runs DeBERTa in a8 (the
// SBV2 pipeline builds its text encoder Session with the default execution), so every case of this
// file reads the series and runs only with `KARUME_SERIES=1`; without it each case SKIPs
// explicitly and the series is not touched. The mirror goldens stay in the series (they are not
// part of the test-asset dataset).

import { type DirEntry, isNotFound, readDirSync, readFile } from "@karume/test-support/fs";
import { assert, assertEquals } from "@std/assert";
import { test } from "vite-plus/test";
import {
  acquireGpu,
  type GpuContext,
  parseSafetensors,
  prepareContainer,
  type SessionDiagnostics,
  type Tensor,
} from "../../runtime/mod.ts";
import {
  compareTensors,
  formatAllclose,
  type Tolerance,
} from "../../runtime/src/reference/allclose.ts";
import { assertAdapterMatchesEnvironment } from "../../runtime/tests/helpers/environment.ts";
import { ioTensor } from "../../runtime/tests/helpers/golden-io.ts";
import { GPU_AVAILABLE } from "../../runtime/tests/helpers/gpu.ts";
import { requireCensus, SEAT_SIGNATURES } from "../../runtime/tests/helpers/pipeline-census.ts";
import { openResults, runRecordedCase } from "../../runtime/tests/helpers/results.ts";
import { modelPresent, openSeriesContainer } from "../../runtime/tests/helpers/container-files.ts";
import { seriesGraph } from "../../models/tests/helpers/series-graphs.ts";
import {
  collapseCeilingOf,
  GRAPH_OUTPUTS,
  judgeCollapse,
  STRICT_OUTPUTS,
} from "../../models/tests/helpers/collapse-ceilings.ts";
import { SERIES_CHECKS, SERIES_CHECKS_OPT_IN } from "../../models/tests/helpers/series-checks.ts";
import { UPSTREAM_PARITY, UPSTREAM_PARITY_OPT_IN } from "./helpers/upstream-parity.ts";

/**
 * The strict tolerance applied to output.0 / output.1 (**historical value** — ADR 0026 decision 3).
 *
 * ADR 0026's measurement (full-24layer, GPU i8a8 vs torch mirror): maxAbs of output.1 5.72e-6. The
 * f32 route vs the mirror gives 5.76e-2 on the same output.1. atol 5e-5 is about 8.7 × the former
 * and 1/1,150 of the latter, so a run in which the activation quantisation did not run (a silent
 * fallback to the f32 route) fails here.
 *
 * output.0 is the exit of the embedding layer and does not go through a linear (the same route as
 * the measured 9.54e-7 of the w8 check), so the same value suffices.
 *
 * **rtol is 0** (the same reason as the w8 check — the \|ref\| smallest nonzero thins out down to
 * the 1e-7 level, and rtol is dragged by the appearance of near-zero elements).
 */
const STRICT_TOLERANCE: Tolerance = { atol: 5e-5, rtol: 0 };

/**
 * The expected breakdown of the dispatches of one run (**values derived from the graph and the
 * implementation** — they agree with the 192 diagnostic keys of ADR 0026).
 *
 * - `i8a8Linear` 192 = 24 layers × 8. DeBERTa-v2 uses query_proj / key_proj for the relative
 *   position projection too (`share_att_key`), so against 6 modules per layer the graph has 8
 *   linear nodes (confirmed by reading the IR of full-24layer: 192 linears, every weight in i8
 *   storage, k ∈ {1024, 4096}, all a8-eligible, k % 4 == 0).
 * - `quantizeRows` 192 = the same count. An a8 linear lowers to the pair "`quantize_rows`, which
 *   quantises the activations to per-token i8 → the integer dot-product GEMM" (`buildLinearI8a8` of
 *   `recipe-builders/linear.ts`). Even linears that share an input do not share the quantisation.
 *   NOTE: since the erf gelu joined the linear prologue (perf-ledger F-4, 2026-10-25), 24 of the
 *   192 are its prologue form (`quantize_rows:…:gelu:reg4` — the 24 intermediate → output linears,
 *   K 4096, whose input is the gelu), which the `quantize_rows:` prefix counts as well: the total
 *   stays 192, and the 24 gelu dispatches leave the run.
 * - `otherLinear` 0 = linear kernels other than i8a8 (any of the GEMM / GEMV families). If even one
 *   runs, the eligibility check missed and part of it silently fell back to the f32 route.
 */
const EXPECTED_CENSUS: KeyCensus = { i8a8Linear: 192, quantizeRows: 192, otherLinear: 0 };

const MODEL_FILE = "model.krm";
/**
 * The prefix of the mirror io (MUST be the same spelling as `ACT_IO_PREFIX` of
 * `deberta/export.py`).
 */
const ACT_IO_PREFIX = "io-i8a8.";
const IO_SUFFIX = ".safetensors";

const ROOT = new URL("../../../outputs/series/deberta-i8/full-24layer/", import.meta.url);
const GRAPH = seriesGraph("deberta-i8", "full-24layer");
const VARIANT = "i8/full-24layer";
/** The generation command pasted as is on SKIP (tools/export-recipes/deberta/README.md). */
const GENERATE =
  "cd tools/export-recipes && uv run --with 'transformers==5.14.1' " +
  "python -m deberta.export --dtype i8 --layers 24 --act-quant";

/**
 * The cases that should have been generated. **They are pinned here, not enumerated** — so that
 * tests do not silently vanish in an environment where only part of the generation ran (the source
 * of truth is `GOLDEN_SENTENCES` + `padded` of `deberta/export.py`).
 */
const EXPECTED_CASES = ["case0", "case1", "case2", "padded"] as const;

/** The classified dispatch counts of one run. */
type KeyCensus = {
  readonly i8a8Linear: number;
  readonly quantizeRows: number;
  readonly otherLinear: number;
};

/**
 * The classification of pipeline keys (the source of truth of the spellings is
 * `src/kernels/linear{,-i8a8,-gemv}.ts` / `quantize-rows.ts`).
 *
 * MUST: the linear side is counted by a prefix match on `"linear"` (not `"linear:"`). This check's
 * T is 11 to 35, and a linear that fell back to the f32 route lowers not to the GEMM (`linear:`)
 * but to the row-block GEMV (`linear_gemv:` — M ≤ 64). Counting with `"linear:"` would keep that
 * fallback out of `otherLinear`, which would pass at 0.
 */
const isLinearKey = (key: string): boolean => key.startsWith("linear");
const isI8a8 = SEAT_SIGNATURES.linearCompute.a8;
const isQuantizeRows = (key: string): boolean => key.startsWith("quantize_rows:");

const censusOf = (diagnostics: SessionDiagnostics): KeyCensus => {
  // MUST: a run without a census is not counted as "0 dispatches" — the `otherLinear: 0` check
  // would silently pass vacuously (requireCensus fails on undefined / empty).
  const entries = requireCensus(diagnostics.lastRunPipelines, "deberta w8a8");
  let i8a8Linear = 0;
  let quantizeRows = 0;
  let otherLinear = 0;
  for (const entry of entries) {
    if (isLinearKey(entry.key)) {
      if (isI8a8(entry.key)) i8a8Linear += entry.dispatchCount;
      else otherLinear += entry.dispatchCount;
    } else if (isQuantizeRows(entry.key)) {
      quantizeRows += entry.dispatchCount;
    }
  }
  return { i8a8Linear, quantizeRows, otherLinear };
};

/**
 * Lists the mirror io. It degrades to empty only when the directory does not exist.
 * MUST: anything other than NotFound propagates — reading a permission error and the like as "no
 * assets" would turn a verification that did not run into a quietly green SKIP.
 */
const mirrorFiles = (root: URL): readonly string[] => {
  let entries: readonly DirEntry[];
  try {
    entries = [...readDirSync(root)];
  } catch (cause) {
    if (isNotFound(cause)) return [];
    throw cause;
  }
  return entries
    .filter(
      (entry) =>
        entry.isFile && entry.name.startsWith(ACT_IO_PREFIX) && entry.name.endsWith(IO_SUFFIX),
    )
    .map((entry) => entry.name)
    .sort();
};

const caseNameOf = (file: string): string =>
  file.slice(ACT_IO_PREFIX.length, file.length - IO_SUFFIX.length);

const readBuffer = async (file: string): Promise<ArrayBuffer> => {
  const bytes = await readFile(new URL(file, ROOT));
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
};

/**
 * Listed synchronously because it is needed at registration (the same reason as the
 * `test.skipIf` decision).
 * MUST: the series is listed only under SERIES_CHECKS (the default lanes do not read it).
 */
const FILES = SERIES_CHECKS ? mirrorFiles(ROOT) : [];
/**
 * Whether the mirror exists. None at all = an environment generated without `--act-quant` (or not
 * generated), so every case SKIPs. The export replaces the output directory as a whole
 * (`karume.artifacts.staged_publication`), so an old mirror left next to a new model is never
 * produced — its absence can be read as a legitimate state.
 */
const AVAILABLE = FILES.length > 0;
const RUNNABLE = AVAILABLE && GPU_AVAILABLE;
/**
 * The cases registered: the files on disk under SERIES_CHECKS (the completeness case compares them
 * with {@link EXPECTED_CASES}), the expected set otherwise, so each case SKIPs by name.
 */
const CASE_FILES: readonly string[] = SERIES_CHECKS
  ? FILES
  : EXPECTED_CASES.map((name) => `${ACT_IO_PREFIX}${name}${IO_SUFFIX}`);

if (SERIES_CHECKS && !AVAILABLE) {
  console.warn(
    `[karume] ${ROOT.pathname} has no w8a8 mirror io (${ACT_IO_PREFIX}<case>), so ` +
      `DeBERTa's w8a8 mirror check (${VARIANT}) SKIPs. Generate: ${GENERATE}`,
  );
}

test.skipIf(!AVAILABLE)(
  `DeBERTa w8a8 assets (${VARIANT}): every mirror io case and the model are present${SERIES_CHECKS_OPT_IN}`,
  () => {
    assertEquals(FILES.map(caseNameOf), [...EXPECTED_CASES], `mirror cases of ${ROOT.pathname}`);
    assert(modelPresent(new URL(MODEL_FILE, ROOT)), `${MODEL_FILE} is missing`);
  },
);

/**
 * Builds the Session in a8 and runs it once on the mirror io's inputs (shared by the numeric check
 * and the census).
 */
const openCase = async (file: string) => {
  const [opened, ioBytes] = await Promise.all([
    openSeriesContainer(new URL(MODEL_FILE, ROOT)),
    readBuffer(file),
  ]);
  const parsed = prepareContainer(opened, GRAPH);
  const io = parseSafetensors(ioBytes);
  // Check first that it outputs all layers (25). With the distribution's (one-output) assets mixed
  // up, output.1, where the strict tolerance applies, does not even exist, and the check shrinks to
  // the collapse ceilings alone.
  assertEquals(parsed.graph.outputs.length, GRAPH_OUTPUTS, `${VARIANT}: count of graph.outputs`);
  // Every tensor of io corresponds exactly to the graph's inputs and outputs (nothing extra,
  // nothing missing).
  const expectedKeys = [
    ...parsed.graph.inputs.map((spec) => `input.${spec.name}`),
    ...parsed.graph.outputs.map((_, index) => `output.${index}`),
  ].sort();
  assertEquals([...io.tensors.keys()].sort(), expectedKeys, `tensor keys of ${file}`);
  const inputs: Record<string, Tensor> = {};
  for (const spec of parsed.graph.inputs) {
    const view = io.tensors.get(`input.${spec.name}`);
    assert(view !== undefined, `input.${spec.name} is missing from ${file}`);
    inputs[spec.name] = ioTensor(io, view, spec.dtype);
  }
  return { parsed, io, inputs };
};

/**
 * Where the outcomes and measurements go (`outputs/verify/<environment key>/<date>_deberta-w8a8/` —
 * safe to delete). It is a separate directory from the w8 check's (`deberta-golden`) because two
 * modules writing to the same directory overwrite each other's `cases`.
 */
const results = openResults("deberta-w8a8");

/** One output of an a8 run, already checked for shape and dtype against the mirror io. */
type CheckedOutput = {
  readonly index: number;
  readonly name: string;
  readonly ioName: string;
  readonly where: string;
  readonly actual: Tensor;
  readonly expected: Tensor;
};

/**
 * Runs one mirror case in a8 on `gpu` and checks the run's structure: the output set, and each
 * output's shape and dtype against the mirror io. The values go to `inspect` (the opt-in parity
 * case compares them with the torch mirror golden).
 */
const runCheckedCase = async (
  gpu: GpuContext,
  file: string,
  inspect: (output: CheckedOutput) => void = () => {},
): Promise<void> => {
  const caseName = caseNameOf(file);
  const { parsed, io, inputs } = await openCase(file);
  // The device is released even when building the Session fails (putting the build failure outside
  // the try would leak it).
  const session = await parsed.createContainerSession(gpu, { linearCompute: "a8" });
  try {
    const outputs = await session.run(inputs);
    assertEquals(Object.keys(outputs).sort(), [...parsed.graph.outputs].sort());
    parsed.graph.outputs.forEach((name, index) => {
      // MUST: the io lookup and the collapse-ceiling lookup use the same name (no applying another
      // output's ceiling through a mixed-up key).
      const ioName = `output.${index}`;
      const view = io.tensors.get(ioName);
      assert(view !== undefined, `${ioName} is missing from ${file}`);
      const where = `${VARIANT}/${caseName} ${ioName} ('${name}')`;
      const declared = parsed.graph.values[name].dtype;
      assertEquals(outputs[name].shape, view.shape, `${where}: shape`);
      assertEquals(outputs[name].dtype, declared, `${where}: dtype`);
      inspect({
        index,
        name,
        ioName,
        where,
        actual: outputs[name],
        expected: ioTensor(io, view, declared),
      });
    });
  } finally {
    await session.dispose();
  }
};

for (const file of CASE_FILES) {
  const caseName = caseNameOf(file);
  test.skipIf(!RUNNABLE)(
    `DeBERTa w8a8: ${VARIANT} / ${caseName} runs in a8 with the graph's outputs, shapes and dtypes (real GPU)${SERIES_CHECKS_OPT_IN}`,
    async () => {
      const gpu = await acquireGpu();
      try {
        await runCheckedCase(gpu, file);
      } finally {
        gpu.destroy();
      }
    },
  );
  test.skipIf(!RUNNABLE || !UPSTREAM_PARITY)(
    `DeBERTa w8a8 mirror parity: ${VARIANT} / ${caseName} (real GPU / torch mirror golden)${UPSTREAM_PARITY_OPT_IN}${SERIES_CHECKS_OPT_IN}`,
    async () => {
      await runRecordedCase(results, { id: `${VARIANT}/${caseName}` }, async (recorded) => {
        const { measurements, comparisons } = recorded;
        const gpu = await acquireGpu();
        /**
         * The outputs outside their band (failed after every output is measured — a re-derivation
         * needs the measurements of every output).
         */
        const failures: string[] = [];
        try {
          // This route keeps results as this machine's rows, so check that the adapter the key was
          // taken from is the running adapter (mixing them up on a multi-GPU machine would measure
          // against another machine's band).
          assertAdapterMatchesEnvironment(gpu);
          await runCheckedCase(gpu, file, ({ index, name, ioName, where, actual, expected }) => {
            if (index < STRICT_OUTPUTS) {
              const report = compareTensors(actual, expected, STRICT_TOLERANCE);
              measurements.push({
                output: name,
                maxAbs: report.maxAbsError,
                maxRel: report.maxRelError,
                tolerance: STRICT_TOLERANCE,
                stage: "karume",
              });
              if (!report.pass) failures.push(`${where}: ${formatAllclose(report)}`);
              return;
            }
            // The outputs under a collapse ceiling go into `comparisons` (the existing form that
            // holds the relative RMS and maxAbs in one entry — reference = the torch mirror golden,
            // practical = a8. The floor is 0 = no floor).
            const ceiling = collapseCeilingOf(ioName);
            const report = judgeCollapse(actual.data, expected.data, ceiling);
            comparisons.push({
              output: name,
              reference: `${ACT_IO_PREFIX}${caseName}`,
              practical: "a8",
              relRms: report.relRms,
              maxAbs: report.maxAbs,
              band: { metric: "relRms", floor: 0, ceiling },
            });
            if (!report.pass) {
              failures.push(
                `${where}: relRms=${report.relRms} (ceiling ${ceiling}) ` +
                  `nonFinite=${report.nonFiniteCount} maxAbs=${report.maxAbs}`,
              );
            }
          });
        } finally {
          gpu.destroy();
        }
        assertEquals(
          failures,
          [],
          "outputs outside their band (strict: output.0/1, collapse ceiling: output.2 on)",
        );
      });
    },
  );
}

test.skipIf(!RUNNABLE)(
  `DeBERTa w8a8 (${VARIANT}): the run drives its linears with i8a8 GEMMs and quantize_rows only (key census)${SERIES_CHECKS_OPT_IN}`,
  async () => {
    // The breakdown does not depend on T (the plan only changes shape with the binding of the
    // symbolic dimensions; the order of the dispatches is the same), so one case suffices. `padded`
    // is used so that the mask path rides on the same run.
    const file = `${ACT_IO_PREFIX}padded${IO_SUFFIX}`;
    const { parsed, inputs } = await openCase(file);
    const gpu = await acquireGpu();
    try {
      const session = await parsed.createContainerSession(gpu, { linearCompute: "a8" });
      try {
        await session.run(inputs);
        const census = censusOf(session.diagnostics());
        console.log(
          `[e2e] deberta ${VARIANT} w8a8 census: i8a8 linear ${census.i8a8Linear} / ` +
            `quantize_rows ${census.quantizeRows} / other linear ${census.otherLinear}`,
        );
        assertEquals(
          census,
          EXPECTED_CENSUS,
          "the dispatch breakdown differs from the expectation (otherLinear > 0 is a silent fallback to the f32 route; " +
            "a different count means the number of linears or how they are lowered moved)",
        );
      } finally {
        await session.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
);
