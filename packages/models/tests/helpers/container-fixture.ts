/**
 * Tooling for tests to build **`krm` containers** and the `karume/5` distribution form (the one
 * helper the models tests share).
 *
 * NOTE: the container **writer** is borrowed from
 * `packages/runtime/tests/helpers/container-write.ts`. It is not "test-only code of runtime" but
 * **format tooling** (the same standing as Python's `karume.container`); a copy on the models side
 * would let the two implementations drift. The rule of `helpers/memory-cache.ts` (do not bring the
 * other side's test-only code over here) still holds.
 *
 * What this file adds is only the "container → distribution form" side:
 *
 * - {@link writeContainer} … writes a `ModelInput` and returns the **expected values** the
 *   manifest needs (length + sha256 of the two documents) together with the parts
 * - {@link partAssets} … folds a written container into full-surface keys (`<component>[i]`)
 * - {@link declaredContainer} … a **declaration-only** container for the manifest's
 *   `weights.<component>.<dtype>`
 * - {@link linearComponent} / {@link tensorlessContainer} … the smallest runnable component (a
 *   single `linear`) and a declaration-only component (a vessel for reaching the cross-check)
 * - {@link runLinearProbe} … runs the Session of `linearComponent` once and returns the output
 *   (the observation point of whether the weights reached the GPU)
 */

import {
  type AssetInput,
  type ModelInput,
  type TensorInput,
  writeModelContainer,
  type WriteOptions,
} from "../../../runtime/tests/helpers/container-write.ts";
import { type IrDeclaration, parseIrDeclarationValue, type Session } from "@karume/runtime";

export { parseIrDeclarationValue };

/**
 * The expected value of one document inside a container (the shape of the manifest's
 * `container.descriptor`).
 */
type DocumentExpectation = { readonly length: number; readonly sha256: string };

/** One written container (single form, split form, and the expected values the manifest needs). */
export type TestContainer = {
  /** The split form (index 0 is part 0 = header + the two documents). */
  readonly parts: readonly Uint8Array<ArrayBuffer>[];
  /** The single form (the whole container in one piece). */
  readonly single: Uint8Array<ArrayBuffer>;
  readonly descriptor: {
    readonly graph: DocumentExpectation;
    readonly model: DocumentExpectation;
  };
};

export const sha256Hex = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
  [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");

const documentOf = async (bytes: Uint8Array<ArrayBuffer>): Promise<DocumentExpectation> => ({
  length: bytes.byteLength,
  sha256: await sha256Hex(bytes),
});

/** Writes one container (by default it cuts small parts to make a split form). */
export const writeContainer = async (
  input: ModelInput,
  options: WriteOptions = { partBytes: 4096, blockBytes: 512 },
): Promise<TestContainer> => {
  const written = await writeModelContainer(input, options);
  return {
    parts: written.parts,
    single: written.single,
    descriptor: {
      graph: await documentOf(written.graphDescriptorBytes),
      model: await documentOf(written.modelDescriptorBytes),
    },
  };
};

/** A declaration-only input (no values flow, so what the operation computes does not matter). */
export type TensorlessGraphSpec = {
  /**
   * The names of symbolic dimensions (a static graph when omitted). Each must appear in some input
   * shape.
   */
  readonly symbols?: readonly string[];
  /** The input declarations (the surface `assertStaticDim` reads). */
  readonly inputs: readonly {
    readonly name: string;
    readonly shape: readonly (number | string)[];
  }[];
  /**
   * The declaration of the one output (the surface `assertOutputScale` / `assertOutputDim` reads).
   */
  readonly output: {
    readonly name: string;
    readonly shape: readonly (number | string)[];
  };
};

/**
 * Builds an IR v2 graph declaration from the spec.
 *
 * The only node is "one `mul` that takes the first input on both of its inputs" — the purpose is
 * the cross-check of declarations, so it is not run (with no initializer at all, the binding table
 * is empty too).
 */
const declarationOf = (spec: TensorlessGraphSpec): IrDeclaration =>
  parseIrDeclarationValue({
    format: "karume-ir",
    version: 2,
    requires: { ops: ["mul"] },
    symbols: spec.symbols ?? [],
    inputs: spec.inputs.map((input) => ({ name: input.name, dtype: "f32", shape: input.shape })),
    outputs: [spec.output.name],
    initializers: {},
    values: { [spec.output.name]: { dtype: "f32", shape: spec.output.shape } },
    states: {},
    nodes: [
      {
        op: "mul",
        ins: [spec.inputs[0].name, spec.inputs[0].name],
        outs: [spec.output.name],
        attrs: {},
      },
    ],
  });

/**
 * The **pre-write input** of one declaration-only container (the shape passed to `serveContainer`,
 * which serves it from the fake HF). Graph name = the weights key of the distribution manifest —
 * container-v1 §2.1.
 */
export const tensorlessInput = (
  graph: string,
  spec: TensorlessGraphSpec,
  assets: readonly AssetInput[] = [],
): ModelInput => ({
  graphs: { [graph]: declarationOf(spec) },
  consts: [],
  weights: [],
  assets: [...assets],
  provenance: { license: "test" },
});

/**
 * One declaration-only container (graph name = the weights key of the distribution manifest —
 * container-v1 §2.1).
 */
export const tensorlessContainer = async (
  graph: string,
  spec: TensorlessGraphSpec,
  assets: readonly AssetInput[] = [],
): Promise<TestContainer> => await writeContainer(tensorlessInput(graph, spec, assets));

const f32Bytes = (values: readonly number[]): Uint8Array<ArrayBuffer> =>
  new Uint8Array(Float32Array.from(values).buffer);

/**
 * The smallest runnable component (a single `linear(x, w, b)`, static 2×2). Replacing `op` makes it
 * "a graph that cannot run" (the IR parser does not look at the op name's spelling, so what fails
 * is the capability check = admission).
 */
export const linearComponent = (
  graph: string,
  values: { readonly op?: string; readonly w?: number; readonly b?: number } = {},
): ModelInput => {
  const op = values.op ?? "linear";
  const declaration = parseIrDeclarationValue({
    format: "karume-ir",
    version: 2,
    requires: { ops: [op] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [2, 2] }],
    outputs: ["y"],
    initializers: { w: {}, b: {} },
    values: {
      w: { dtype: "f32", shape: [2, 2] },
      b: { dtype: "f32", shape: [2] },
      y: { dtype: "f32", shape: [2, 2] },
    },
    states: {},
    nodes: [{ op, ins: ["x", "w", "b"], outs: ["y"], attrs: {} }],
  });
  const weights: readonly TensorInput[] = [
    {
      graph,
      initializer: "w",
      bytes: f32Bytes(Array(4).fill(values.w ?? 0.5)),
      encoding: { codec: "f32" },
    },
    {
      graph,
      initializer: "b",
      bytes: f32Bytes(Array(2).fill(values.b ?? 0.25)),
      encoding: { codec: "f32" },
    },
  ];
  return {
    graphs: { [graph]: declaration },
    consts: [],
    weights,
    assets: [],
    provenance: { license: "test" },
  };
};

/**
 * The expected value of {@link runLinearProbe} (with the defaults of {@link linearComponent}, w =
 * 0.5 and b = 0.25).
 *
 * `linear` is `y[m, n] = Σ_k x[m, k] · w[n, k] + b[n]` (runtime's CPU reference
 * `reference/ops.ts`), so `x = [[1, 2], [3, 4]]` gives `[[1.75, 1.75], [3.75, 3.75]]`. Every value
 * is exact in binary, so f32 matches exactly. A build that uploads a zero-filled buffer without
 * reading the weight block outputs 0.
 */
export const LINEAR_PROBE_Y: readonly number[] = [1.75, 1.75, 3.75, 3.75];

/**
 * Runs the Session of {@link linearComponent} once with `x = [[1, 2], [3, 4]]` and returns `y`
 * flattened.
 */
export const runLinearProbe = async (session: Session): Promise<number[]> => {
  const outputs = await session.run({
    x: { dtype: "f32", shape: [2, 2], data: Float32Array.from([1, 2, 3, 4]) },
  });
  return Array.from(outputs["y"].data);
};

/** The component keys passed to the full surface (`from*Assets`), in index order from part 0. */
export const partAssets = (
  key: string,
  container: TestContainer,
): Record<string, Uint8Array<ArrayBuffer>> =>
  Object.fromEntries(container.parts.map((part, index) => [`${key}[${index}]`, part]));

/**
 * The container header length (container-v1 §1 — part 0 is "exactly the header + the two
 * documents").
 */
const HEADER_BYTES = 24;

/**
 * A **declaration-only** container for the manifest's `weights.<component>.<dtype>` (the checks of
 * `fromAssets` do not read the parts' contents, so the path / size / sha256 triple only has to be
 * well-formed). The size of part 0 is kept consistent with the lengths of the two declared
 * documents — hub's parse checks that (container-v1 §8).
 */
export const declaredContainer = (stem: string): Record<string, unknown> => {
  const graph = { length: 16, sha256: "b".repeat(64) };
  const model = { length: 8, sha256: "c".repeat(64) };
  return {
    container: {
      descriptor: { graph, model },
      parts: [
        {
          path: `${stem}-00001-of-00002.krm`,
          size: HEADER_BYTES + graph.length + model.length,
          sha256: "a".repeat(64),
        },
        { path: `${stem}-00002-of-00002.krm`, size: 64, sha256: "d".repeat(64) },
      ],
    },
  };
};
