// Real-GPU comparison against torch-derived golden fixtures (stage 3 of ADR 0005). The standing
// test that proves the path from export to inference end to end, and the source of truth for the
// justification of the tolerances.
//
// The targets are decided by **enumerating every directory** under tests/fixtures/golden/. A model
// that is added becomes a target automatically (hard-coding the list would allow a "silent apparent
// success" where a golden is added but no test appears).
//
// The runtime is called through the public API (mod.ts) only. Reading io.safetensors is a test-side
// fixture read, not a distribution, so it uses src/format/safetensors.ts directly.

import { readDirSync, readFile } from "@karume/test-support/fs";
import { test } from "vite-plus/test";
import { assert, assertEquals, assertThrows } from "@std/assert";
import {
  acquireGpu,
  capabilities,
  ExecutionError,
  type IrDeclaration,
  type MemoryTensor,
  type OpenedContainer,
  openMemoryContainer,
  parseIrDeclarationValue,
  type PreparedModel,
  prepareContainer,
  type Tensor,
} from "../mod.ts";
import { parseSafetensors } from "../src/format/safetensors.ts";
import {
  compareTensors,
  EXACT_TOLERANCE,
  formatAllclose,
  type Tolerance,
} from "../src/reference/allclose.ts";
import { assertAdapterMatchesEnvironment, ENVIRONMENT } from "./helpers/environment.ts";
import { ioTensor } from "./helpers/golden-io.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { openResults, runRecordedCase } from "./helpers/results.ts";
import { openSeriesContainer } from "./helpers/container-files.ts";

/**
 * The tolerance for the comparison with the torch CPU expected values = **the first stage of the
 * verdict** (a Karume-specific criterion shared by all outputs). Exceeding it alone does not go
 * red; the second stage, {@link OUTPUT_TOLERANCE} (the WGSL spec band), takes it and records a
 * warning.
 *
 * Measured (of **76 outputs across all 32 models**, 67 are f32. The 9 i32 / bool outputs match
 * exactly with a difference of 0 and are outside this value — ADR 0009. The 2 outputs of
 * `argmax_pick` are indices, so they are on that side), the worst values are
 * **maxAbs 4.77e-7** (deform_conv2d_block output.0) /
 * **maxRel 1.77e-5** (batch_matmul output.0). The former is **only because that element's value,
 * 4.82, is larger than in other goldens; measured in ulp it is exactly 1 ulp** (ulp(4.82) = 2² ·
 * 2⁻²³ = 4.77e-7). The latter is relative error growing on an element near zero; that element's
 * absolute error is only 1.19e-7 (= 1 ulp). The source is reordering of operations, not
 * approximation: ① fma fusion on the GPU side ② a reduction order of matmul / row reduce that
 * differs from torch ③ implementation differences in transcendental functions (the WGSL spec
 * allows a few ulp) — each of the order of a few ulp per output.
 *
 * **atol drives the verdict** (`|x−y| ≤ atol + rtol·|ref|`): elements whose maxRel exceeds rtol
 * also pass because their absolute error stays within 1/8 of atol. atol 1e-6 is about 2.1 times
 * the measured worst 4.77e-7, and that is where the headroom for longer reductions and chains is
 * concentrated. **A golden whose value range exceeds O(10) should be absorbed on the rtol side, not
 * atol** (absolute error scales with the value, so raising atol alone would loosen the net for
 * implementation bugs in small outputs at the same time). The error of an implementation bug (a
 * wrong op, a shifted index) is O(1), six orders of magnitude above this threshold.
 *
 * **Goldens with i8 storage are checked with this same value** (`i8_weights` — ADR 0019).
 * Fake-quant is the ground truth, so the expected values are also computed with the rounded
 * weights, and what is measured here is **implementation error only** (measured maxAbs 1.19e-7 /
 * 8.94e-8 / 2.38e-7, maxRel at most 1.83e-6 — the same order as the other f32 models, updating
 * neither worst value). "Quantization quality" is a separate axis outside this net.
 */
const GOLDEN_TOLERANCE: Tolerance = { atol: 1e-6, rtol: 1e-5 };

/**
 * The per-output **WGSL spec band** = the second stage of the verdict. The outer key is
 * `<model>/<output name>` and the inner key is the **environment key** (`<runtime>-<adapter name
 * slug>` — ADR 0106 decision 2). **Only outputs that have a row for the machine running the test**
 * can exceed {@link GOLDEN_TOLERANCE} without going red.
 *
 * Rows are kept per environment key so that **a loosening does not spread beyond the machine it was
 * added for**. A spec band is needed only for outputs where measurement showed that "this GPU's
 * implementation deviates that far", and some machines round the same op almost correctly as the
 * spec intends. Sharing a row across all machines would loosen the regression net on those
 * machines by the same amount. **On a machine without a row there is no second stage at all**: it
 * is measured against the first-stage criterion alone and goes red when it exceeds it.
 *
 * The two-stage verdict (ruling of 2026-09-20):
 *
 * 1. **pass** if it passes the Karume-specific criterion ({@link GOLDEN_TOLERANCE}, shared by all
 *    outputs).
 * 2. If a failing output has a `spec` for this machine, it is measured again with it and, if it
 *    passes, it is **pass + warning** — the `note` of the results JSON
 *    (`outputs/verify/<environment key>/<date>_golden/results.json`) records "which outputs were
 *    accepted by the spec band", and `measurements` records their measured values and the band used
 *    to accept them.
 * 3. **fail** if this machine has no `spec` or it fails the `spec` too (the message is unchanged).
 *
 * The Karume-specific criterion is a guide that keeps "this value used to pass" in view; it has no
 * basis in the spec and **may readily be removed** (the net that catches implementation bugs is
 * the spec band — the error of a wrong op or a shifted index is O(1), four orders of magnitude
 * above the spec band too).
 *
 * MUST: add a row here only **per op, within the WGSL spec's accuracy guarantee, and when there is
 * no real harm**, and write the basis (the relevant spec section and the measured value) for each
 * row. Do not loosen everything at once.
 *
 * - `activations/sin` on `deno-intel-graphics-bmg-g21`: the WGSL spec's `sin(x)` allows **an
 *   absolute error of up to 2⁻¹¹** for |x| ≤ π (§ Accuracy of Concrete Floating Point
 *   Expressions). The golden inputs are [−1.28, 1.89], inside this interval. Intel Arc B570 (Mesa
 *   ANV) gives 2.68e-5 at x = 1.5908 (measured 2026-09-20 — within the spec, four orders of
 *   magnitude below the O(1) of an implementation bug). NVIDIA (RTX 3080 Ti) rounds almost
 *   correctly, so it passes at 1e-6 and has no row.
 */
const OUTPUT_TOLERANCE: Readonly<
  Record<string, Readonly<Record<string, { readonly spec: Tolerance }>>>
> = {
  "activations/sin": {
    "deno-intel-graphics-bmg-g21": { spec: { atol: 2 ** -11, rtol: 0 } },
  },
};

/**
 * The spec band for the machine of this run (`undefined` when there is none = no second stage).
 *
 * Only machines that cannot acquire a GPU adapter have no environment key, and there the golden
 * tests themselves are skipped at registration (the row still cannot be looked up, so it returns
 * `undefined`).
 */
const specTolerance = (model: string, output: string): Tolerance | undefined => {
  const environment = ENVIRONMENT.key;
  if (environment === undefined) return undefined;
  const key = `${model}/${output}`;
  if (!Object.hasOwn(OUTPUT_TOLERANCE, key)) return undefined;
  const byEnvironment = OUTPUT_TOLERANCE[key];
  if (!Object.hasOwn(byEnvironment, environment)) return undefined;
  return byEnvironment[environment].spec;
};

/**
 * Where verdicts and warnings go (`outputs/verify/<environment key>/<date>_golden/` — safe to
 * delete).
 */
const results = openResults("golden");

const GOLDEN_ROOT = new URL("./fixtures/golden/", import.meta.url);

/**
 * Enumerated synchronously because it is needed at registration (the same reason as Deno.test's
 * ignore decision).
 */
const goldenModels = (): readonly string[] =>
  [...readDirSync(GOLDEN_ROOT)]
    .filter((entry) => entry.isDirectory)
    .map((entry) => entry.name)
    .sort();

const MODELS = goldenModels();

const readBuffer = async (model: string, file: string): Promise<ArrayBuffer> => {
  const bytes = await readFile(new URL(`${model}/${file}`, GOLDEN_ROOT));
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
};

/** The representative path of one golden's container (`MODEL_FILE` in `goldens.py`). */
const MODEL_FILE = "model.krm";

/**
 * Opens one golden's container. Every golden is stored in split form (when const is empty, part 1
 * is a zero-byte file — container-v1 §8). Files are found with `resolveParts`.
 *
 * **The graph name is the name of its directory** (`graph_name=spec.name` in `goldens.py`).
 */
const openGolden = async (model: string): Promise<OpenedContainer> =>
  await openSeriesContainer(new URL(`${model}/${MODEL_FILE}`, GOLDEN_ROOT));

/**
 * Goldens whose exported graph **no Session accepts, by design**, with the input that makes it so
 * and the refusal it MUST meet (a list with reasons, MUST).
 *
 * `expert_linear_block`: an `expert_linear` index MUST be output 1 of a `topk` over the
 * weight's E, checked when the model is prepared (ADR 0145 decision 2), and `topk` does not export
 * yet (ADR 0068 addendum), so the golden takes the index as a graph input. The comparison runs the
 * exported graph and weights with a router spliced in front ({@link spliceRouter}): the index input
 * becomes output 1 of `topk(scores)` over a scores input that ranks the golden's ids first, in
 * their order.
 * MUST: the refusal is asserted on the exported graph. The day the router exports (ADR 0145 decision
 * 5) and the golden is re-exported with it, the refusal stops and this entry goes red — remove it.
 */
const ROUTED_GOLDENS: Readonly<
  Record<string, { readonly index: string; readonly refusal: string }>
> = {
  expert_linear_block: {
    index: "index",
    refusal: "an expert_linear index MUST be output 1 of a topk over the weight's E",
  },
};

/**
 * The exported graph of a routed golden with its index input replaced by output 1 of a
 * `topk(scores, k)` node (see {@link ROUTED_GOLDENS}), on the golden's own f32 weights, and the
 * scores that make the topk return `index` row by row (pick j of a token scores k − j at its id,
 * every other id 0 — the ids of a token are distinct, as a topk's are).
 */
const spliceRouter = async (
  opened: OpenedContainer,
  model: string,
  indexName: string,
  index: Tensor,
): Promise<{ readonly prepared: PreparedModel; readonly scores: Tensor }> => {
  const bound = opened.graphs[model];
  const declaration: IrDeclaration = bound.declaration;
  const spec = declaration.inputs.find((input) => input.name === indexName);
  const reader = declaration.nodes.find(
    (node) => node.op === "expert_linear" && node.ins[1] === indexName,
  );
  if (spec === undefined || reader === undefined || index.dtype !== "i32") {
    throw new Error(`${model}: no i32 input '${indexName}' read as an expert_linear index`);
  }
  const experts = declaration.values[reader.ins[2]].shape[0];
  const picks = spec.shape[1];
  if (typeof experts !== "number" || typeof picks !== "number") {
    throw new Error(`${model}: E and k of the routed index must be numbers`);
  }
  const scoresName = `${indexName}_scores`;
  const valuesName = `${indexName}_values`;
  const spliced = parseIrDeclarationValue({
    format: declaration.format,
    version: declaration.version,
    requires: { ops: [...new Set([...declaration.requires.ops, "topk"])].sort() },
    symbols: declaration.symbols,
    inputs: declaration.inputs.map((input) =>
      input.name === indexName
        ? { name: scoresName, dtype: "f32", shape: [input.shape[0], experts] }
        : input,
    ),
    outputs: declaration.outputs,
    initializers: Object.fromEntries(
      Object.entries(declaration.initializers).map(([name, { shared }]) => [
        name,
        shared ? { shared: true } : {},
      ]),
    ),
    values: {
      ...declaration.values,
      [indexName]: { dtype: "i32", shape: spec.shape },
      [valuesName]: { dtype: "f32", shape: spec.shape },
    },
    nodes: [
      { op: "topk", ins: [scoresName], outs: [valuesName, indexName], attrs: { k: picks } },
      ...declaration.nodes.map(({ op, ins, outs, attrs }) => ({ op, ins, outs, attrs })),
    ],
  });
  const tensors: Record<string, MemoryTensor> = {};
  for (const [name, supply] of bound.supplies) {
    if (supply.encoding.codec !== "f32" || supply.blocks.length !== 1) {
      throw new Error(`${model}: initializer '${name}' is not one whole f32 block`);
    }
    const [block] = supply.blocks;
    const bytes = (await opened.readBlock(block.id)).slice(0, block.payloadBytes);
    tensors[name] = { bytes, encoding: { codec: "f32" } };
  }
  const prepared = prepareContainer(
    openMemoryContainer({ graphs: { [model]: spliced }, tensors: { [model]: tensors } }),
    model,
  );
  const [tokens] = index.shape;
  const data = new Float32Array(tokens * experts);
  for (let t = 0; t < tokens; t += 1) {
    for (let j = 0; j < picks; j += 1) data[t * experts + index.data[t * picks + j]] = picks - j;
  }
  return { prepared, scores: { dtype: "f32", shape: [tokens, experts], data } };
};

test("there is at least one golden fixture, and every one is registered as a test", () => {
  // An empty enumeration would only be "0 tests and green", so the lower bound is pinned here
  // (ADR 0005).
  assert(MODELS.length > 0, `no golden model in ${GOLDEN_ROOT.pathname}`);
});

// A coverage check exists on the generating side too (goldens.py in tools/exporter), but that one
// compares against **the ops the exporter can emit** and runs only when the fixtures are
// regenerated. A runtime that adds an op without adding a golden does not touch the fixtures, so
// the generating side cannot detect it. This pins, by reading the distributions themselves, that
// **no executable op (capabilities()) is missing from every golden** (independent of the GPU
// adapter — a set operation on declarations, not a comparison).
/**
 * Ops that cannot have a golden (the mirror of `NON_EMITTABLE_OPS` on the Python side — a list with
 * reasons, MUST).
 *
 * `topk` is in the contract table but **cannot be exported from torch** (ADR 0068 addendum): there
 * is no aten handler, and the tuple meta of multi-output aten plus the slot wiring of
 * `operator.getitem` remain to be built as a new mechanism (deferred until sampling actually needs
 * it — what stops in measurement is getitem, not `aten.topk.default`).
 * `state_append` cannot be exported from torch either (ADR 0067 decision 5): it is an effect op
 * **with no aten counterpart**, emitted only by the decode graph script (an implementation step).
 * In addition, the state slots themselves are owned by GenerationContext, so it cannot run at all
 * in a one-shot golden distribution.
 * MUST: the comparison below works **in both directions** (this list matches the actual uncovered
 * set exactly). The day a golden starts to exercise topk, this goes red too and forces the entry to
 * be removed.
 */
const OPS_WITHOUT_GOLDEN: readonly string[] = ["state_append", "topk"];

test("the requires.ops of all goldens cover the set of executable ops", async () => {
  const covered = new Set<string>();
  for (const model of MODELS) {
    const opened = await openGolden(model);
    // A routed golden is refused by prepare by design (ROUTED_GOLDENS); its ops are declared.
    const graph = Object.hasOwn(ROUTED_GOLDENS, model)
      ? opened.graphs[model].declaration
      : prepareContainer(opened, model).graph;
    for (const op of graph.requires.ops) covered.add(op);
  }
  const uncovered = capabilities().ops.filter((op) => !covered.has(op));
  assertEquals(
    uncovered,
    OPS_WITHOUT_GOLDEN,
    "ops that no golden exercises (when adding an op, add a golden too / an op that cannot be exported goes to OPS_WITHOUT_GOLDEN with a reason)",
  );
});

/**
 * The spelling of an environment key (the form `environmentKey` builds = `<runtime>-<slug of
 * lower-case letter / digit words joined by single hyphens>`).
 */
const ENVIRONMENT_KEY_PATTERN = /^(deno|chrome)-[a-z0-9]+(-[a-z0-9]+)*$/;

// A spec-band row is looked up only **for outputs that failed the first stage on that machine**, so
// an orphaned row (a deleted golden, an output renamed by a re-export) or a misspelled environment
// key is treated like "a machine without a row" and stays silently. This pins, independently of the
// GPU adapter, that the rows point at real things (keeps the ledger of ADR 0106 addendum decision 1
// from drifting from reality).
test("each OUTPUT_TOLERANCE row points at an existing golden output and a correctly spelled environment key", async () => {
  for (const [key, byEnvironment] of Object.entries(OUTPUT_TOLERANCE)) {
    const slash = key.indexOf("/");
    assert(slash > 0, `${key}: the key is not of the form <model>/<output name>`);
    const model = key.slice(0, slash);
    const output = key.slice(slash + 1);
    assert(MODELS.includes(model), `${key}: golden '${model}' is not in ${GOLDEN_ROOT.pathname}`);
    const outputs = prepareContainer(await openGolden(model), model).graph.outputs;
    const listed = outputs.join(", ");
    assert(
      outputs.includes(output),
      `${key}: output '${output}' is not in graph.outputs of golden '${model}' (${listed})`,
    );
    for (const [environment, { spec }] of Object.entries(byEnvironment)) {
      assert(
        ENVIRONMENT_KEY_PATTERN.test(environment),
        `${key}: environment key '${environment}' does not match ${ENVIRONMENT_KEY_PATTERN}`,
      );
      for (const [name, value] of Object.entries(spec)) {
        assert(
          Number.isFinite(value) && value >= 0,
          `${key} / ${environment}: ${name} = ${value} is not a finite non-negative number`,
        );
      }
    }
  }
});

for (const model of MODELS) {
  test.skipIf(!GPU_AVAILABLE)(
    `golden comparison: ${model} (real GPU / torch CPU expected values)`,
    async () => {
      /**
       * Output names that exceeded the Karume-specific criterion but were accepted by the spec band
       * (they become the note of the results JSON).
       */
      const accepted: string[] = [];
      /** Messages of outputs that failed even the spec band (the first one fails the test). */
      const failures: string[] = [];
      /**
       * The note listing output names accepted by the spec band (absent when there are none — the
       * numbers live in measurements).
       */
      const acceptedNote = (): string | undefined =>
        accepted.length === 0 ? undefined : `accepted by the spec band: ${accepted.join(", ")}`;
      await runRecordedCase(
        results,
        {
          id: model,
          // Failures other than the tolerance (reading assets, createSession, output keys, shape /
          // dtype, exceptions from run) are recorded in the results too. Even if recording fails,
          // it does not replace the original exception (the only diagnostic that says what broke).
          failureNote: (cause) =>
            [acceptedNote(), `exception: ${cause instanceof Error ? cause.message : String(cause)}`]
              .filter((part) => part !== undefined)
              .join("; "),
        },
        async ({ measurements }) => {
          const [opened, ioBytes] = await Promise.all([
            openGolden(model),
            readBuffer(model, "io.safetensors"),
          ]);
          const routed = Object.hasOwn(ROUTED_GOLDENS, model) ? ROUTED_GOLDENS[model] : undefined;
          if (routed !== undefined) {
            // MUST: the exported graph itself is refused (see ROUTED_GOLDENS).
            assertThrows(() => prepareContainer(opened, model), ExecutionError, routed.refusal);
          }
          const plain = routed === undefined ? prepareContainer(opened, model) : undefined;
          const exported = plain?.graph ?? opened.graphs[model].declaration;
          const io = parseSafetensors(ioBytes);

          // Every tensor in io corresponds exactly to the graph's inputs and outputs (nothing
          // extra, nothing missing). The naming convention's source of truth is "golden layout" in
          // tools/exporter/README.md.
          const expectedKeys = [
            ...exported.inputs.map((spec) => `input.${spec.name}`),
            ...exported.outputs.map((_, index) => `output.${index}`),
          ].sort();
          assertEquals(
            [...io.tensors.keys()].sort(),
            expectedKeys,
            "tensor keys of io.safetensors",
          );

          // The symbolic dimension T is bound from the actual length of the golden input shape (no
          // explicit bindings are passed).
          const inputs: Record<string, Tensor> = {};
          for (const spec of exported.inputs) {
            const view = io.tensors.get(`input.${spec.name}`);
            assert(view !== undefined, `input.${spec.name} is not in io.safetensors`);
            inputs[spec.name] = ioTensor(io, view, spec.dtype);
          }
          // A routed golden runs with its router spliced in (ROUTED_GOLDENS), every other one as
          // exported.
          const spliced =
            routed === undefined
              ? undefined
              : await spliceRouter(opened, model, routed.index, inputs[routed.index]);
          const parsed = plain ?? spliced?.prepared;
          assert(parsed !== undefined, `${model}: neither prepared nor spliced`);
          if (routed !== undefined && spliced !== undefined) {
            delete inputs[routed.index];
            inputs[`${routed.index}_scores`] = spliced.scores;
          }

          const gpu = await acquireGpu();
          // MUST: device destruction also covers a failing adapter check and a failing
          // `createSession`. If it is missed, the rest of the run carries an undestroyed device and
          // later tests go red with OOM (deferred release in known-issues).
          try {
            // This path records reference values and results as this machine's row, so it checks
            // here that the adapter the key was taken from and the adapter that runs are the same
            // (mixing them up on a multi-GPU machine would measure against another machine's band).
            assertAdapterMatchesEnvironment(gpu);
            const session = await parsed.createContainerSession(gpu);
            try {
              const outputs = await session.run(inputs);
              assertEquals(Object.keys(outputs).sort(), [...parsed.graph.outputs].sort());

              parsed.graph.outputs.forEach((name, index) => {
                const view = io.tensors.get(`output.${index}`);
                assert(view !== undefined, `output.${index} is not in io.safetensors`);
                const where = `${model} output.${index} ('${name}')`;
                const declared = parsed.graph.values[name].dtype;
                assertEquals(outputs[name].shape, view.shape, `${where}: shape`);
                assertEquals(outputs[name].dtype, declared, `${where}: dtype`);
                const expected = ioTensor(io, view, declared);
                // f32 uses allclose, i32 / bool use exact equality (integer arithmetic leaves no
                // room for approximation).
                // MUST: `compareTensors` drops the band argument for anything other than f32 and
                // measures with {@link EXACT_TOLERANCE}, so **the band put in the record is chosen
                // the same way**, from this one place (the `tolerance` of `measurements` is "the
                // band used for acceptance" — it does not show the reader headroom that does not
                // exist).
                const band = (declaredBand: Tolerance): Tolerance =>
                  declared === "f32" ? declaredBand : EXACT_TOLERANCE;
                const karumeBand = band(GOLDEN_TOLERANCE);
                const karume = compareTensors(outputs[name], expected, karumeBand);
                // Only outputs that failed the first stage reach the second stage (the WGSL spec
                // band). If they pass, it is pass + warning.
                const spec = karume.pass ? undefined : specTolerance(model, name);
                if (spec === undefined) {
                  // No second stage (passed the first stage / this machine has no row) = the first
                  // stage is the deciding stage.
                  measurements.push({
                    output: name,
                    maxAbs: karume.maxAbsError,
                    maxRel: karume.maxRelError,
                    tolerance: karumeBand,
                    stage: "karume",
                  });
                  if (!karume.pass) failures.push(`${where}: ${formatAllclose(karume)}`);
                  return;
                }
                const specBand = band(spec);
                const report = compareTensors(outputs[name], expected, specBand);
                // The measured values (maxAbs / maxRel) do not depend on the band, so the
                // second-stage report is taken as is.
                measurements.push({
                  output: name,
                  maxAbs: report.maxAbsError,
                  maxRel: report.maxRelError,
                  tolerance: specBand,
                  stage: "spec",
                });
                if (!report.pass) {
                  failures.push(`${where}: ${formatAllclose(report)}`);
                  return;
                }
                // measurements holds the numbers, so the note lists only the accepted output names
                // (not kept twice).
                accepted.push(name);
              });
            } finally {
              await session.dispose();
            }
          } finally {
            gpu.destroy();
          }
          // The verdict is recorded before throwing (the note of a run that ended red is needed
          // too).
          return { status: failures.length === 0 ? "pass" : "fail", note: acceptedNote() };
        },
      );
      assert(failures.length === 0, failures[0]);
    },
  );
}
