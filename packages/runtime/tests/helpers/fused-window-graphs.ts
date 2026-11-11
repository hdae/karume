/**
 * One graph per fusion rule with **exactly one window** of that rule — graph inputs in, the window's
 * output out — for the memory estimate's post-fusion accounting (estimate_fused_steps_test.ts pins
 * each one's `workspaceBytes` to a literal derived by hand; gpu_estimate_test.ts compares it with the
 * slot table of a real Session). The ten built-in rules (adaLN in both of its forms, rmsNormRope also
 * in the Wan form whose window defers k's nodes), two external rules (the single-dispatch
 * `test/gelu_mul` and `test/gelu_mul_staged`, which stages the gelu value in a step temporary; the
 * staged one also after a dead region, and the single-dispatch one also binding an intermediate), and
 * the i8a8 linear with and without a folded producer.
 *
 * The expected numbers stay in the tests (moving hand-derived constants here would make the checks
 * tautological — the rule of estimate-graphs.ts).
 */

import type { SessionOptions } from "../../src/runtime/executor.ts";
import {
  defineRule,
  type FusionMatch,
  fusedBinaryWgsl,
  type NodePlan,
  prologueUnaryWgsl,
  sameShape,
} from "../../kernel.ts";
import type { TensorInput } from "./container-write.ts";
import { decomposedAttentionGraph, weight } from "./estimate-graphs.ts";
import { EXTERNAL_GELU_MUL_RULE } from "./external-gelu-mul-rule.ts";
import {
  linearStaticQuantizeGraph,
  linearStaticQuantizeTensors,
} from "./linear-static-quantize-graph.ts";
import { type DeclarationJson, f32Bytes, fill, type FilledTensor } from "./model-fixture.ts";
import { quantizeI8 } from "./i8.ts";
import { rmsNormRopeGraph } from "./rms-norm-rope-graph.ts";

/** One case: the graph, its supplies, the inputs of a run and the options of both sides. */
export type FusedWindowCase = {
  readonly name: string;
  /** The counter the window moves in `lastRunFusions` / the plan's counts. */
  readonly counter: string;
  /** How many windows the counter counts (2 for the Wan form of rmsNormRope). */
  readonly windows: number;
  readonly graph: DeclarationJson;
  readonly tensors: readonly TensorInput[];
  readonly inputs: Readonly<Record<string, FilledTensor>>;
  /** The Session options the window needs, given to the estimate too. */
  readonly options: SessionOptions;
  /** Symbol bindings of the estimate (the run binds them from the inputs). */
  readonly bindings?: Readonly<Record<string, number>>;
};

const f32 = (shape: readonly number[]) => ({ dtype: "f32", shape: [...shape] });

const declaration = (parts: {
  readonly inputs: DeclarationJson["inputs"];
  readonly outputs: readonly string[];
  readonly values: DeclarationJson["values"];
  readonly nodes: DeclarationJson["nodes"];
  readonly initializers?: DeclarationJson["initializers"];
}): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: [...new Set(parts.nodes.map((node) => node.op))] },
  symbols: [],
  inputs: parts.inputs,
  outputs: [...parts.outputs],
  initializers: parts.initializers ?? {},
  values: parts.values,
  nodes: parts.nodes,
});

/** Deterministic inputs for every graph input (bool inputs alternate 0 / 1 by row). */
const inputsOf = (
  graph: DeclarationJson,
  bindings: Readonly<Record<string, number>> = {},
): Record<string, FilledTensor> =>
  Object.fromEntries(
    graph.inputs.map((spec, slot) => {
      const shape = spec.shape.map((dim) => (typeof dim === "number" ? dim : bindings[dim]));
      return [
        spec.name,
        spec.dtype === "bool"
          ? fill(shape, (i) => Math.floor(i / shape[shape.length - 1]) % 2, "bool")
          : fill(shape, (i) => Math.sin(i * 0.37 + slot) * 1.5),
      ];
    }),
  );

const ROWS = 4;
const DIM = 8;

// ------------------------------------------------------------------------------------- silu

/** `mul(x, sigmoid(x))` on `[4, 8]`. */
const silu = (): FusedWindowCase => {
  const graph = declaration({
    inputs: [{ name: "x", ...f32([4, 8]) }],
    outputs: ["y"],
    values: { s: f32([4, 8]), y: f32([4, 8]) },
    nodes: [
      { op: "sigmoid", ins: ["x"], outs: ["s"], attrs: {} },
      { op: "mul", ins: ["x", "s"], outs: ["y"], attrs: {} },
    ],
  });
  return {
    name: "silu",
    counter: "silu",
    windows: 1,
    graph,
    tensors: [],
    inputs: inputsOf(graph),
    options: {},
  };
};

// -------------------------------------------------------------------------------- upsample2x

/** The nearest 2× upsample of `[2, 3, 5, 7]` as reshape / expand / reshape × 2 → `[2, 3, 10, 14]`. */
const upsample2x = (): FusedWindowCase => {
  const graph = declaration({
    inputs: [{ name: "x", ...f32([2, 3, 5, 7]) }],
    outputs: ["y"],
    values: {
      a: f32([30, 7, 1]),
      wide: f32([30, 7, 2]),
      b: f32([6, 5, 14]),
      c: f32([6, 5, 1, 14]),
      tall: f32([6, 5, 2, 14]),
      y: f32([2, 3, 10, 14]),
    },
    nodes: [
      { op: "reshape", ins: ["x"], outs: ["a"], attrs: {} },
      { op: "expand", ins: ["a"], outs: ["wide"], attrs: {} },
      { op: "reshape", ins: ["wide"], outs: ["b"], attrs: {} },
      { op: "reshape", ins: ["b"], outs: ["c"], attrs: {} },
      { op: "expand", ins: ["c"], outs: ["tall"], attrs: {} },
      { op: "reshape", ins: ["tall"], outs: ["y"], attrs: {} },
    ],
  });
  return {
    name: "upsample2x",
    counter: "upsample2x",
    windows: 1,
    graph,
    tensors: [],
    inputs: inputsOf(graph),
    options: {},
  };
};

// -------------------------------------------------------------------------------------- rope

/** The half-split RoPE (slice-first) of `x [1, 1, 5, 128]` with `cos` / `sin [1, 1, 5, 128]`. */
const rope = (): FusedWindowCase => {
  const full = [1, 1, 5, 128];
  const half = [1, 1, 5, 64];
  const graph = declaration({
    inputs: [
      { name: "x", ...f32(full) },
      { name: "cos", ...f32(full) },
      { name: "sin", ...f32(full) },
    ],
    outputs: ["y"],
    values: {
      first: f32(half),
      second: f32(half),
      negative: f32(half),
      rotated: f32(full),
      direct: f32(full),
      cross: f32(full),
      y: f32(full),
    },
    nodes: [
      { op: "slice", ins: ["x"], outs: ["first"], attrs: { dim: 3, start: 0, end: 64 } },
      { op: "slice", ins: ["x"], outs: ["second"], attrs: { dim: 3, start: 64, end: 128 } },
      { op: "neg", ins: ["second"], outs: ["negative"], attrs: {} },
      { op: "cat", ins: ["negative", "first"], outs: ["rotated"], attrs: { dim: 3 } },
      { op: "mul", ins: ["x", "cos"], outs: ["direct"], attrs: {} },
      { op: "mul", ins: ["rotated", "sin"], outs: ["cross"], attrs: {} },
      { op: "add", ins: ["direct", "cross"], outs: ["y"], attrs: {} },
    ],
  });
  return {
    name: "rope",
    counter: "rope",
    windows: 1,
    graph,
    tensors: [],
    inputs: inputsOf(graph),
    options: {},
  };
};

// -------------------------------------------------------------------------------- rmsNormAdd

/** `add(r, rms_norm(x, w))` on `[4, 256]` (fuseRmsNormAdd). */
const rmsNormAdd = (): FusedWindowCase => {
  const graph = declaration({
    inputs: [
      { name: "x", ...f32([4, 256]) },
      { name: "w", ...f32([256]) },
      { name: "r", ...f32([4, 256]) },
    ],
    outputs: ["y"],
    values: { n: f32([4, 256]), y: f32([4, 256]) },
    nodes: [
      { op: "rms_norm", ins: ["x", "w"], outs: ["n"], attrs: { eps: 1e-6 } },
      { op: "add", ins: ["r", "n"], outs: ["y"], attrs: {} },
    ],
  });
  return {
    name: "rmsNormAdd",
    counter: "rmsNormAdd",
    windows: 1,
    graph,
    tensors: [],
    inputs: inputsOf(graph),
    options: { fuseRmsNormAdd: true },
  };
};

// ------------------------------------------------------------------------------- rmsNormRope

/** q's chain alone (`[1, S, 256]`, two heads of 128) at S = 4. */
const rmsNormRope = (): FusedWindowCase => {
  const graph = rmsNormRopeGraph({ form: "single" });
  const bindings = { S: 4 };
  return {
    name: "rmsNormRope",
    counter: "rmsNormRope",
    windows: 1,
    graph,
    tensors: [],
    inputs: inputsOf(graph, bindings),
    options: {},
    bindings,
  };
};

/** The Wan form at S = 4: q's window defers k's rms_norm and reshapes, which k's window then takes. */
const rmsNormRopeDeferred = (): FusedWindowCase => {
  const graph = rmsNormRopeGraph();
  const bindings = { S: 4 };
  return {
    name: "rmsNormRope (Wan form, deferred nodes)",
    counter: "rmsNormRope",
    windows: 2,
    graph,
    tensors: [],
    inputs: inputsOf(graph, bindings),
    options: {},
    bindings,
  };
};

// ------------------------------------------------------------------------------------- adaln

/**
 * The broadcast form: `add(mul(layer_norm(x), add(scale, one)), shift)` on `[1, 4, 8]`, with the
 * two modulation vectors' reshapes between `layer_norm` and the chain (the in-window passthroughs
 * the step list puts before the fused step — aliases of graph inputs, no slot).
 */
const adaln = (): FusedWindowCase => {
  const row = [1, ROWS, DIM];
  const vector = [1, 1, DIM];
  const graph = declaration({
    inputs: [
      { name: "x", ...f32(row) },
      { name: "ln_weight", ...f32([DIM]) },
      { name: "ln_bias", ...f32([DIM]) },
      { name: "one", ...f32([1]) },
      { name: "shift_src", ...f32([1, DIM]) },
      { name: "scale_src", ...f32([1, DIM]) },
    ],
    outputs: ["y"],
    values: {
      t: f32(row),
      shift: f32(vector),
      scale: f32(vector),
      s: f32(vector),
      p: f32(row),
      y: f32(row),
    },
    nodes: [
      {
        op: "layer_norm",
        ins: ["x", "ln_weight", "ln_bias"],
        outs: ["t"],
        attrs: { normalized_shape: [DIM], eps: 0.000001 },
      },
      { op: "reshape", ins: ["shift_src"], outs: ["shift"], attrs: {} },
      { op: "reshape", ins: ["scale_src"], outs: ["scale"], attrs: {} },
      { op: "add", ins: ["scale", "one"], outs: ["s"], attrs: {} },
      { op: "mul", ins: ["t", "s"], outs: ["p"], attrs: {} },
      { op: "add", ins: ["p", "shift"], outs: ["y"], attrs: {} },
    ],
  });
  return {
    name: "adaln",
    counter: "adaln",
    windows: 1,
    graph,
    tensors: [],
    inputs: inputsOf(graph),
    options: {},
  };
};

/**
 * The per-token form (Wan2.2): the modulation vectors picked per token by `where(mask, c, g)`, two
 * dispatches with the pack of the four vectors in a step temporary.
 */
const adalnToken = (): FusedWindowCase => {
  const row = [1, ROWS, DIM];
  const vector = [1, 1, DIM];
  const graph = declaration({
    inputs: [
      { name: "x", ...f32(row) },
      { name: "ln_weight", ...f32([DIM]) },
      { name: "ln_bias", ...f32([DIM]) },
      { name: "mask", dtype: "bool", shape: [1, ROWS, 1] },
      { name: "one", ...f32([1]) },
      { name: "scale_c", ...f32(vector) },
      { name: "scale_g", ...f32(vector) },
      { name: "shift_c", ...f32(vector) },
      { name: "shift_g", ...f32(vector) },
    ],
    outputs: ["y"],
    values: {
      t: f32(row),
      sel_scale: f32(row),
      s: f32(row),
      p: f32(row),
      sel_shift: f32(row),
      y: f32(row),
    },
    nodes: [
      {
        op: "layer_norm",
        ins: ["x", "ln_weight", "ln_bias"],
        outs: ["t"],
        attrs: { normalized_shape: [DIM], eps: 0.000001 },
      },
      { op: "where", ins: ["mask", "scale_c", "scale_g"], outs: ["sel_scale"], attrs: {} },
      { op: "add", ins: ["sel_scale", "one"], outs: ["s"], attrs: {} },
      { op: "mul", ins: ["t", "s"], outs: ["p"], attrs: {} },
      { op: "where", ins: ["mask", "shift_c", "shift_g"], outs: ["sel_shift"], attrs: {} },
      { op: "add", ins: ["p", "sel_shift"], outs: ["y"], attrs: {} },
    ],
  });
  return {
    name: "adaln (per-token form)",
    counter: "adaln",
    windows: 1,
    graph,
    tensors: [],
    inputs: inputsOf(graph),
    options: {},
  };
};

// ----------------------------------------------------------------------------- gatedResidual

/** `add(residual, mul(y, where(mask, gate_c, gate_g)))` on `[1, 4, 8]`. */
const gatedResidual = (): FusedWindowCase => {
  const row = [1, ROWS, DIM];
  const vector = [1, 1, DIM];
  const graph = declaration({
    inputs: [
      { name: "mask", dtype: "bool", shape: [1, ROWS, 1] },
      { name: "gate_c", ...f32(vector) },
      { name: "gate_g", ...f32(vector) },
      { name: "y", ...f32(row) },
      { name: "residual", ...f32(row) },
    ],
    outputs: ["out"],
    values: { g: f32(row), p: f32(row), out: f32(row) },
    nodes: [
      { op: "where", ins: ["mask", "gate_c", "gate_g"], outs: ["g"], attrs: {} },
      { op: "mul", ins: ["y", "g"], outs: ["p"], attrs: {} },
      { op: "add", ins: ["residual", "p"], outs: ["out"], attrs: {} },
    ],
  });
  return {
    name: "gatedResidual",
    counter: "gatedResidual",
    windows: 1,
    graph,
    tensors: [],
    inputs: inputsOf(graph),
    options: {},
  };
};

// ---------------------------------------------------------------------- linearStaticQuantize

/**
 * `static_quantize(linear(x [4, 1536], w i8 [256, 1536], b))` (fuseLinearStaticQuantize with the
 * parallel GEMV) — the step binds the weight's resident scale (`weightScale`).
 */
const linearStaticQuantize = (): FusedWindowCase => {
  const base = linearStaticQuantizeGraph();
  // The helper's `neg` of x (its value check of the release) is a second node; this case keeps the
  // window alone.
  const graph: DeclarationJson = {
    ...base,
    requires: { ops: ["linear", "static_quantize"] },
    outputs: ["y"],
    values: Object.fromEntries(Object.entries(base.values).filter(([name]) => name !== "copy")),
    nodes: base.nodes.filter((node) => node.op !== "neg"),
  };
  return {
    name: "linearStaticQuantize",
    counter: "linearStaticQuantize",
    windows: 1,
    graph,
    tensors: linearStaticQuantizeTensors(),
    inputs: inputsOf(graph),
    options: { fuseLinearStaticQuantize: true, linearGemvReduce: "parallel" },
  };
};

// --------------------------------------------------------------------- geluMulStaticQuantize

/** `static_quantize(mul(gelu_tanh(gate), other))` on `[1, 4, 256]` (fuseGeluMulStaticQuantize). */
const geluMulStaticQuantize = (): FusedWindowCase => {
  const row = [1, 4, 256];
  const graph = declaration({
    inputs: [
      { name: "gate", ...f32(row) },
      { name: "other", ...f32(row) },
    ],
    outputs: ["y"],
    values: { g: f32(row), m: f32(row), y: f32(row) },
    nodes: [
      { op: "gelu_tanh", ins: ["gate"], outs: ["g"], attrs: {} },
      { op: "mul", ins: ["g", "other"], outs: ["m"], attrs: {} },
      { op: "static_quantize", ins: ["m"], outs: ["y"], attrs: { scale: Math.fround(0.0371) } },
    ],
  });
  return {
    name: "geluMulStaticQuantize",
    counter: "geluMulStaticQuantize",
    windows: 1,
    graph,
    tensors: [],
    inputs: inputsOf(graph),
    options: { fuseGeluMulStaticQuantize: true },
  };
};

// ------------------------------------------------------------------------- rowBlockAttention

/** The decomposed attention of estimate-graphs.ts at 2 heads × 64 queries × 64 keys × 8. */
const rowBlockAttention = (): FusedWindowCase => {
  const graph = decomposedAttentionGraph({ heads: 2, queries: 64, keys: 64, headDim: 8 });
  const inputs = inputsOf(graph);
  return {
    name: "rowBlockAttention",
    counter: "rowBlockAttention",
    windows: 1,
    graph,
    tensors: [],
    // A mask of 0 and −1 (an additive mask, as the exporter emits).
    inputs: { ...inputs, mask: fill([1, 1, 1, 64], (i) => (i % 4 === 3 ? -1 : 0)) },
    options: {},
  };
};

// ---------------------------------------------------------------------------- external rules

/** `mul(gelu(gate), other)` on `[1, 4, 64]` — the window both external rules take. */
export const geluMulGraph = (): DeclarationJson => {
  const row = [1, 4, 64];
  return declaration({
    inputs: [
      { name: "gate", ...f32(row) },
      { name: "other", ...f32(row) },
    ],
    outputs: ["y"],
    values: { g: f32(row), y: f32(row) },
    nodes: [
      { op: "gelu", ins: ["gate"], outs: ["g"], attrs: {} },
      { op: "mul", ins: ["g", "other"], outs: ["y"], attrs: {} },
    ],
  });
};

/** The single-dispatch external rule (`test/gelu_mul` — operands in the default order). */
const externalGeluMul = (): FusedWindowCase => {
  const graph = geluMulGraph();
  return {
    name: "external test/gelu_mul",
    counter: "test/gelu_mul",
    windows: 1,
    graph,
    tensors: [],
    inputs: inputsOf(graph),
    options: { fusionRules: [EXTERNAL_GELU_MUL_RULE] },
  };
};

const STAGED_WORKGROUP_SIZE = 64;

/** The grid-stride loop of the staged rule's two kernels over `params.n` elements. */
const stagedKernel = (title: string, bindings: string, helpers: readonly string[], body: string) =>
  `// karume runtime test rule test/gelu_mul_staged: ${title}
struct Params { n: u32, reserved0: u32, reserved1: u32, reserved2: u32 }
@group(0) @binding(0) var<uniform> params: Params;
${bindings}

${helpers.join("\n\n")}

@compute @workgroup_size(${STAGED_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
  for (var i = gid.x; i < params.n; i += groups.x * ${STAGED_WORKGROUP_SIZE}u) {
    ${body}
  }
}`;

type StagedMatch = FusionMatch & {
  readonly gate: string;
  readonly other: string;
  readonly count: number;
};

/**
 * `mul(gelu(gate), other)` in **two** dispatches with the gelu value in a step temporary: ① writes
 * `act = gelu(gate)` into temporary 0, ② reads it with `other` and writes the output. Its operands
 * are declared (temporary, binds, output), so the estimate maps the rule's own schedule (the same
 * values as the unfused pair: each kernel is the plain elementwise expression of its node).
 */
export const EXTERNAL_STAGED_GELU_MUL_RULE = defineRule<StagedMatch, "test/gelu_mul_staged">({
  name: "test/gelu_mul_staged",
  heads: ["gelu"],
  contract: "E",
  internals: "private",
  match: (nodes: readonly NodePlan[], index: number): StagedMatch | undefined => {
    const [gelu, mul] = [nodes[index], nodes[index + 1]];
    if (gelu?.node.op !== "gelu" || mul?.node.op !== "mul") return undefined;
    if (mul.node.ins[0] !== gelu.outputs[0].name || gelu.node.ins.length !== 1) return undefined;
    const gate = gelu.node.ins[0];
    const other = mul.node.ins[1];
    const shape = gelu.inputShapes[0];
    const shapes = [gelu.outputs[0].shape, ...mul.inputShapes, mul.outputs[0].shape];
    if (other === gate || !shapes.every((each) => sameShape(each, shape))) return undefined;
    const count = shape.reduce((product, dim) => product * dim, 1);
    if (count < 1) return undefined;
    return { window: [gelu, mul], chain: [gelu, mul], gate, other, count };
  },
  build: (matched) => {
    const product = matched.chain[1].outputs[0];
    const gelu = prologueUnaryWgsl("gelu", "gate[i]");
    const params = new Uint32Array([matched.count, 0, 0, 0]);
    const workgroups = {
      kind: "gridStride",
      items: matched.count,
      size: STAGED_WORKGROUP_SIZE,
    } as const;
    return {
      binds: [matched.gate, matched.other],
      outputName: product.name,
      outputShape: product.shape,
      temps: [{ byteLength: matched.count * 4, allocBefore: 0, releaseAfter: 1 }],
      dispatches: [
        {
          key: `ext:test:gelu_mul_staged:v1:gelu:wg${STAGED_WORKGROUP_SIZE}`,
          wgsl: () =>
            stagedKernel(
              "gelu(gate) into the step temporary",
              "@group(0) @binding(1) var<storage, read> gate: array<f32>;\n" +
                "@group(0) @binding(2) var<storage, read_write> act: array<f32>;",
              gelu.helpers,
              `act[i] = ${gelu.value};`,
            ),
          params,
          operands: [
            { kind: "bind", index: 0 },
            { kind: "temp", id: 0 },
          ],
          workgroups,
        },
        {
          key: `ext:test:gelu_mul_staged:v1:mul:wg${STAGED_WORKGROUP_SIZE}`,
          wgsl: () =>
            stagedKernel(
              "the step temporary times other",
              "@group(0) @binding(1) var<storage, read> act: array<f32>;\n" +
                "@group(0) @binding(2) var<storage, read> other: array<f32>;\n" +
                "@group(0) @binding(3) var<storage, read_write> out: array<f32>;",
              [],
              `out[i] = ${fusedBinaryWgsl("mul", "act[i]", "other[i]")};`,
            ),
          params,
          operands: [{ kind: "temp", id: 0 }, { kind: "bind", index: 1 }, { kind: "output" }],
          workgroups,
        },
      ],
    };
  },
});

/** The two-dispatch external rule with a step temporary. */
const externalStagedGeluMul = (): FusedWindowCase => {
  const graph = geluMulGraph();
  return {
    name: "external test/gelu_mul_staged",
    counter: "test/gelu_mul_staged",
    windows: 1,
    graph,
    tensors: [],
    inputs: inputsOf(graph),
    options: { fusionRules: [EXTERNAL_STAGED_GELU_MUL_RULE] },
  };
};

/**
 * `graph` behind a region its window can reuse: `dead = neg(z)` (`[1, 2048]`, 8,192 B) and the
 * one-element `probe = slice(dead)` (a graph output) run first, so `dead` is released before the
 * window's step allocates anything. Its 8,192 B region then has room for the window's output and a
 * temporary together, and only the read / write separation of the window's dispatches (a value read
 * and a value written by one dispatch go to different regions) keeps the temporary out of it — the
 * graphs where a dispatch's read set moves the total.
 */
const withDeadRegion = (graph: DeclarationJson): DeclarationJson => ({
  ...graph,
  requires: { ops: [...new Set(["neg", "slice", ...graph.requires.ops])] },
  inputs: [{ name: "z", ...f32([1, 2048]) }, ...graph.inputs],
  outputs: ["probe", ...graph.outputs],
  values: { dead: f32([1, 2048]), probe: f32([1, 1]), ...graph.values },
  nodes: [
    { op: "neg", ins: ["z"], outs: ["dead"], attrs: {} },
    { op: "slice", ins: ["dead"], outs: ["probe"], attrs: { dim: 1, start: 0, end: 1 } },
    ...graph.nodes,
  ],
});

/** The staged external rule's window after a dead region ({@link withDeadRegion}). */
const externalStagedGeluMulAfterDeadRegion = (): FusedWindowCase => {
  const graph = withDeadRegion(geluMulGraph());
  return {
    name: "external test/gelu_mul_staged after a dead region",
    counter: "test/gelu_mul_staged",
    windows: 1,
    graph,
    tensors: [],
    inputs: inputsOf(graph),
    options: { fusionRules: [EXTERNAL_STAGED_GELU_MUL_RULE] },
  };
};

/**
 * The single-dispatch external rule's window binding an intermediate: `gate = neg(x)` runs after a
 * dead region ({@link withDeadRegion}), so the window's step binds a value an earlier step produced
 * (a slot), not a graph input. The region `dead` leaves has room for `gate` and the window's output
 * together; only the step's read of `gate` (its dispatch reads `gate` and writes the output) keeps
 * the output out of it — the graph where a fused step's read of a bind moves the total.
 */
const externalGeluMulOfIntermediate = (): FusedWindowCase => {
  const base = geluMulGraph();
  const row = [1, 4, 64];
  const graph = withDeadRegion({
    ...base,
    requires: { ops: [...new Set(["neg", ...base.requires.ops])] },
    inputs: [{ name: "x", ...f32(row) }, ...base.inputs.filter((spec) => spec.name !== "gate")],
    values: { gate: f32(row), ...base.values },
    nodes: [{ op: "neg", ins: ["x"], outs: ["gate"], attrs: {} }, ...base.nodes],
  });
  return {
    name: "external test/gelu_mul binding an intermediate after a dead region",
    counter: "test/gelu_mul",
    windows: 1,
    graph,
    tensors: [],
    inputs: inputsOf(graph),
    options: { fusionRules: [EXTERNAL_GELU_MUL_RULE] },
  };
};

/** Every rule's case, in the order of the built-in list, then the external rules. */
export const FUSED_WINDOW_CASES: readonly (() => FusedWindowCase)[] = [
  linearStaticQuantize,
  geluMulStaticQuantize,
  rmsNormAdd,
  silu,
  upsample2x,
  rope,
  rmsNormRope,
  rmsNormRopeDeferred,
  adaln,
  adalnToken,
  gatedResidual,
  rowBlockAttention,
  externalGeluMul,
  externalStagedGeluMul,
  externalStagedGeluMulAfterDeadRegion,
  externalGeluMulOfIntermediate,
];

/**
 * `linear(x, w, b)` (`prologue` false) or `linear(gelu_tanh(x), w, b)` with `w [16, 64]` resident as
 * i8 and `x [1, 4, 64]` — the a8 linear's quantize → GEMM pair with its two node temporaries, and the
 * same with the producer folded into its step ① (perf-ledger F-4).
 */
export const a8LinearCase = (prologue: boolean): FusedWindowCase => {
  const [rows, k, n] = [4, 64, 16];
  const row = [1, rows, k];
  const quantized = quantizeI8(fill([n, k], (i) => Math.sin(i * 0.61)).data, [n, k], 0);
  const graph = declaration({
    inputs: [{ name: "x", ...f32(row) }],
    outputs: ["y"],
    initializers: { w: {}, b: {} },
    values: {
      w: f32([n, k]),
      b: f32([n]),
      ...(prologue ? { g: f32(row) } : {}),
      y: f32([1, rows, n]),
    },
    nodes: [
      ...(prologue ? [{ op: "gelu_tanh", ins: ["x"], outs: ["g"], attrs: {} }] : []),
      { op: "linear", ins: [prologue ? "g" : "x", "w", "b"], outs: ["y"], attrs: {} },
    ],
  });
  return {
    name: prologue ? "a8 linear with a folded gelu_tanh" : "a8 linear",
    counter: "linearPrologue",
    windows: prologue ? 1 : 0,
    graph,
    tensors: [
      weight("b", f32Bytes(new Array(n).fill(0.5))),
      weight("w", quantized.bytes, {
        codec: "int8-sym",
        groupSize: k,
        scale: { bytes: f32Bytes([...quantized.scale]), dtype: "f32" },
      }),
    ],
    inputs: inputsOf(graph),
    options: { linearCompute: "a8" },
  };
};

/** The a8 linear (no prologue) after a dead region ({@link withDeadRegion}). */
export const a8LinearAfterDeadRegion = (): FusedWindowCase => {
  const linear = a8LinearCase(false);
  const graph = withDeadRegion(linear.graph);
  return {
    ...linear,
    name: "a8 linear after a dead region",
    graph,
    inputs: inputsOf(graph),
  };
};
