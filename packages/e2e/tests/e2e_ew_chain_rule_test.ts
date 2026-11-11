// The candidate rule `candidate/ew_chain` (`@karume/candidate-rules`,
// packages/candidate-rules/src/ew-chain.ts — a run of consecutive f32 elementwise nodes in one
// dispatch, written against `@karume/runtime/kernel` alone) on the GPU, against its unfused plan:
// the class E evidence of docs/kernel-rules.md (d). Each graph runs on a Session given the rule
// through `SessionOptions.fusionRules` and on one built from the same options with the rule
// disabled by the test-only `DISABLED_FUSION_RULES` (the plain side, the same graph without the
// rule). Every output is compared as u32 words (`wordIdentity`, NaN by class); the rule's counter
// in `lastRunFusions` reads 1 on the fused side and 0 on the plain side, and the fused side ran the
// whole graph as the rule's one dispatch, so the two sides ran the plans they stand for.
//
// The graphs are small and synthetic (no model asset): the four forms the MOSS-Image DiT has
// (asserted by their dispatch keys — the nine-node gate with two `[1]` constants over a rank-3
// activation, and the three `mul → add` tails at their real ranks and broadcast patterns), the
// same tails with the broadcast operand in the other slot or of rank 1, an internal value read
// twice, a chain of every op of the rule, an element count that is not a multiple of the rule's
// 128 threads and one below 128. The first elements of every input hold ±0, subnormals, ±Inf, NaN
// payloads, the largest finite values and values that saturate tanh and sigmoid; a block after
// them holds exact rounding ties for the products (1 + 2⁻¹² squared lies halfway between two f32,
// and the add of −1 then keeps or loses that half ulp), where a product contracted into the add
// would differ.
//
// Fault injection: a test rule built here from the same pieces (`elementwiseValueWgsl`,
// `stagedLoopWgsl`), identical to the rule's `mul → add` kernel except that the add reads the
// product inline instead of from the staged word, is run on the full-shape `mul → add` graphs;
// it must differ from the plain side, or the check could not see a missing barrier. It lives here,
// not in the runtime, because the runtime's tests import nothing outside the runtime. No adapter =
// explicit SKIP.

import { describe, it } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import {
  acquireGpu,
  createSessionFromContainer,
  type GpuContext,
  type RunOutputs,
  type Session,
  type SessionOptions,
  type Tensor,
} from "@karume/runtime";
import {
  allF32,
  defineRule,
  elementwiseValueWgsl,
  type FusionMatch,
  type NodePlan,
  sameShape,
  stagedLoopWgsl,
} from "@karume/runtime/kernel";
import { wordIdentity } from "@karume/runtime/tune";
import { DISABLED_FUSION_RULES } from "../../runtime/src/runtime/session-types.ts";
import type { TensorInput } from "../../runtime/tests/helpers/container-write.ts";
import {
  type DeclarationJson,
  f32Bytes,
  fill,
  GRAPH_NAME,
  openModelBytes,
} from "../../runtime/tests/helpers/model-fixture.ts";
import { EW_CHAIN_RULE, EW_CHAIN_RULE_NAME } from "@karume/candidate-rules";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";

/** Rows × width: 771 elements leave a partial last workgroup of the rule's 128 threads. */
const ROWS = 3;
const WIDTH = 257;
const FULL = [1, ROWS, WIDTH];
const CHANNEL = [1, 1, WIDTH];
/** The nine-node gate's activation: 900 elements, 4 in the last workgroup. */
const GATE = [1, 3, 300];
/** The RoPE tail's `[1, S, H, D]` and its `[1, S, 1, D]` table: 360 elements. */
const ROPE = [1, 5, 3, 24];
const ROPE_TABLE = [1, 5, 1, 24];
/** Fewer elements than one workgroup has threads. */
const TINY = [1, 2, 45];

/** The words written into the first elements of every input (rotated per input). */
const SPECIAL_WORDS = [
  0x00000000, // +0
  0x80000000, // -0
  0x00000001, // the smallest subnormal
  0x807fffff, // the largest negative subnormal
  0x00400000, // a subnormal in the middle
  0x7f800000, // +Inf
  0xff800000, // -Inf
  0x7fc01234, // a quiet NaN with a payload
  0xff812345, // a negative NaN with a payload
  0x7f7fffff, // the largest finite
  0xff7fffff, // the most negative finite
  0x42c80000, // 100: tanh and sigmoid saturate
  0xc2c80000, // -100
  0x41200000, // 10
  0xc1180000, // -9.5
  0x3f800000, // 1
];

/** Where the tie block starts, and its length (inside the first row of every broadcast input). */
const TIE_AT = 16;
const TIE_LENGTH = 8;
/** 1 + 2⁻¹²: its square is a tie between two f32. */
const TIE_FACTOR = 0x3f800800;
/** −1: added to the rounded square it leaves 2⁻¹¹, added to the exact square 2⁻¹¹ + 2⁻²⁴. */
const TIE_ADDEND = 0xbf800000;

type InputSpec = {
  readonly name: string;
  readonly shape: readonly number[];
  /** An f32 initializer of one element with this value (a weight, as the DiT's constants are). */
  readonly constant?: number;
  /** The word written over the tie block. */
  readonly tie?: number;
};

type NodeSpec = readonly [op: string, ins: readonly string[], out: string];

type Case = {
  readonly name: string;
  readonly inputs: readonly InputSpec[];
  /** In order; the last one writes `y`, the graph's only output. */
  readonly nodes: readonly NodeSpec[];
  /** The shape every node writes. */
  readonly shape: readonly number[];
  /** The scale of the generated values. */
  readonly scale: number;
  /** The rule's dispatch key, for the forms the MOSS-Image DiT has. */
  readonly key?: string;
};

const input = (name: string, shape: readonly number[], tie?: number): InputSpec => ({
  name,
  shape,
  ...(tie === undefined ? {} : { tie }),
});

/** `mul(a, b) → add` reading the product in `slot`; both products tie and the addend is −1. */
const mulAdd = (
  name: string,
  shape: readonly number[],
  [a, b, c]: readonly (readonly number[])[],
  slot: 0 | 1,
  key?: string,
): Case => ({
  name,
  inputs: [input("a", a, TIE_FACTOR), input("b", b, TIE_FACTOR), input("c", c, TIE_ADDEND)],
  nodes: [
    ["mul", ["a", "b"], "t"],
    ["add", slot === 0 ? ["t", "c"] : ["c", "t"], "y"],
  ],
  shape,
  scale: 3.7,
  ...(key === undefined ? {} : { key }),
});

const keyOf = (form: string): string => `ext:candidate:ew_chain:v1:${form}:f32:wg128`;

const CASES: readonly Case[] = [
  {
    // The DiT's FFN gate: two softcapped branches, `c·tanh(g/c)`, one of them times sigmoid(g).
    name: "nine-node gate",
    inputs: [
      input("g", GATE),
      { name: "c4", shape: [1], constant: 4 },
      input("h", GATE),
      { name: "c25", shape: [1], constant: 25 },
    ],
    nodes: [
      ["div", ["g", "c4"], "a0"],
      ["tanh", ["a0"], "a1"],
      ["mul", ["a1", "c4"], "a2"],
      ["sigmoid", ["g"], "a3"],
      ["mul", ["a2", "a3"], "a4"],
      ["div", ["h", "c25"], "a5"],
      ["tanh", ["a5"], "a6"],
      ["mul", ["a6", "c25"], "a7"],
      ["mul", ["a4", "a7"], "y"],
    ],
    shape: GATE,
    scale: 120,
    key: keyOf(
      "f.s.f.s:div.x0.x1-tanh.t0-mul.t1.x1-sigmoid.x0-mul.t2.t3-div.x2.x3-tanh.t5-mul.t6.x3-mul.t4.t7",
    ),
  },
  mulAdd("RoPE tail", ROPE, [ROPE, ROPE_TABLE, ROPE], 1, keyOf("f.b0101.f:mul.x0.x1-add.x2.t0")),
  mulAdd("gated residual", FULL, [FULL, CHANNEL, FULL], 1, keyOf("f.b001.f:mul.x0.x1-add.x2.t0")),
  mulAdd("adaLN tail", FULL, [FULL, CHANNEL, CHANNEL], 0, keyOf("f.b001.b001:mul.x0.x1-add.t0.x2")),
  mulAdd("RoPE tail, table first", ROPE, [ROPE_TABLE, ROPE, ROPE], 0),
  mulAdd("gated residual, gate first", FULL, [CHANNEL, FULL, FULL], 0),
  mulAdd("adaLN tail, scale first", FULL, [CHANNEL, FULL, CHANNEL], 1),
  mulAdd("gated residual, rank-1 gate", FULL, [FULL, [WIDTH], FULL], 1),
  mulAdd("gated residual, below one workgroup", TINY, [TINY, [1, 1, 45], TINY], 1),
  {
    name: "a value read twice",
    inputs: [input("x", FULL), input("z", FULL)],
    nodes: [
      ["add", ["x", "z"], "t"],
      ["mul", ["t", "t"], "u"],
      ["sub", ["u", "x"], "y"],
    ],
    shape: FULL,
    scale: 6.1,
  },
  {
    // Every op of the rule, the longest run it takes (16 nodes), values read across rounds.
    name: "every op",
    inputs: [input("x", FULL), input("z", FULL)],
    nodes: [
      ["neg", ["x"], "t0"],
      ["abs", ["x"], "t1"],
      ["relu", ["t0"], "t2"],
      ["gelu", ["x"], "t3"],
      ["gelu_tanh", ["z"], "t4"],
      ["sin", ["z"], "t5"],
      ["exp", ["t5"], "t6"],
      ["log1p", ["t2"], "t7"],
      ["sqrt", ["t1"], "t8"],
      ["tanh", ["t3"], "t9"],
      ["sigmoid", ["t4"], "t10"],
      ["log", ["t6"], "t11"],
      ["add", ["t7", "t8"], "t12"],
      ["mul", ["t9", "t10"], "t13"],
      ["sub", ["t12", "t13"], "t14"],
      ["div", ["t14", "t11"], "y"],
    ],
    shape: FULL,
    scale: 12.5,
  },
];

const elements = (shape: readonly number[]): number => shape.reduce((count, dim) => count * dim, 1);

const graphOf = (testCase: Case): DeclarationJson => {
  const constants = testCase.inputs.filter((spec) => spec.constant !== undefined);
  const values: DeclarationJson["values"] = {};
  for (const spec of constants) values[spec.name] = { dtype: "f32", shape: [...spec.shape] };
  for (const [, , out] of testCase.nodes)
    values[out] = { dtype: "f32", shape: [...testCase.shape] };
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: [...new Set(testCase.nodes.map(([op]) => op))].sort() },
    symbols: [],
    inputs: testCase.inputs
      .filter((spec) => spec.constant === undefined)
      .map((spec) => ({ name: spec.name, dtype: "f32", shape: [...spec.shape] })),
    outputs: ["y"],
    initializers: Object.fromEntries(constants.map((spec) => [spec.name, {}])),
    values,
    nodes: testCase.nodes.map(([op, ins, out]) => ({ op, ins: [...ins], outs: [out], attrs: {} })),
  };
};

const weightsOf = (testCase: Case): readonly TensorInput[] =>
  testCase.inputs.flatMap((spec) =>
    spec.constant === undefined
      ? []
      : [
          {
            graph: GRAPH_NAME,
            initializer: spec.name,
            bytes: f32Bytes([spec.constant]),
            encoding: { codec: "f32" },
          },
        ],
  );

/** Generated values with the special words first (rotated by the input's index) and the ties. */
const feedsOf = (testCase: Case): Readonly<Record<string, Tensor>> =>
  Object.fromEntries(
    testCase.inputs.flatMap((spec, k) => {
      if (spec.constant !== undefined) return [];
      const tensor = fill(
        spec.shape,
        (i) => Math.sin(i * (0.37 + 0.11 * k) + k) * testCase.scale + (i % 7) / (13 + k),
      );
      const words = new Uint32Array(tensor.data.buffer);
      const rotated = [...SPECIAL_WORDS.slice(k), ...SPECIAL_WORDS.slice(0, k)];
      words.set(rotated.slice(0, Math.min(rotated.length, words.length)));
      if (spec.tie !== undefined) words.fill(spec.tie, TIE_AT, TIE_AT + TIE_LENGTH);
      return [[spec.name, tensor]];
    }),
  );

/** What one run of a side did. */
type Side = {
  readonly outputs: RunOutputs;
  readonly hits: Readonly<Record<string, number | undefined>>;
  readonly pipelines: readonly { readonly key: string; readonly dispatchCount: number }[];
};

/** One run of a Session opened on `testCase`'s graph with `options`. */
const runSide = async (gpu: GpuContext, testCase: Case, options: SessionOptions): Promise<Side> => {
  const session: Session = await createSessionFromContainer(
    gpu,
    await openModelBytes(graphOf(testCase), weightsOf(testCase)),
    GRAPH_NAME,
    options,
  );
  try {
    const outputs = await session.run(feedsOf(testCase));
    const diagnostics = session.diagnostics();
    return {
      outputs,
      hits: diagnostics.lastRunFusions ?? {},
      pipelines: diagnostics.lastRunPipelines ?? [],
    };
  } finally {
    await session.dispose();
  }
};

const wordsOf = (tensor: Tensor | undefined): Uint32Array<ArrayBuffer> => {
  if (tensor === undefined) throw new Error("the run returned no output 'y'");
  return new Uint32Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.length);
};

/** The fused side's options: the rule given. The plain side: the same, with the rule disabled. */
const FUSED: SessionOptions = { fusionRules: [EW_CHAIN_RULE] };
const PLAIN: SessionOptions = { ...FUSED, [DISABLED_FUSION_RULES]: [EW_CHAIN_RULE_NAME] };

// ---- The fault: the rule's `mul → add` kernel with the product read inline by the add ----

const UNSTAGED_RULE_NAME = "test/ew_chain_unstaged";
const WORKGROUP_SIZE = 128;

type UnstagedMatch = FusionMatch & {
  readonly binds: readonly string[];
  /** The add's slot of the product. */
  readonly slot: 0 | 1;
  readonly count: number;
};

/** `mul(x0, x1) → add` reading the product once, every value of one shape, three inputs. */
const unstagedMatch = (nodes: readonly NodePlan[], index: number): UnstagedMatch | undefined => {
  const [mul, add] = [nodes[index], nodes[index + 1]];
  if (mul?.node.op !== "mul" || add?.node.op !== "add") return undefined;
  const chain = [mul, add];
  const product = mul.outputs[0].name;
  const slot = add.node.ins.indexOf(product);
  if (!allF32(chain) || (slot !== 0 && slot !== 1) || add.node.ins[1 - slot] === product) {
    return undefined;
  }
  const shape = add.outputs[0].shape;
  const shapes = [...mul.inputShapes, ...add.inputShapes, mul.outputs[0].shape];
  const binds = [mul.node.ins[0], mul.node.ins[1], add.node.ins[1 - slot]];
  if (shapes.some((each) => !sameShape(each, shape)) || new Set(binds).size !== 3) {
    return undefined;
  }
  return { window: chain, chain, binds, slot, count: elements(shape) };
};

/** The rule's kernel for this form, but the add reads `(x0[i] * x1[i])`, not the staged word. */
const unstagedWgsl = (slot: 0 | 1): string => {
  const product = elementwiseValueWgsl("mul", ["x0[i]", "x1[i]"]).value;
  const inline = `(${product})`;
  const sum = elementwiseValueWgsl("add", slot === 0 ? [inline, "x2[i]"] : ["x2[i]", inline]);
  // The product is still staged as in the rule's kernel (one round), but the add does not read it.
  return `// karume test rule ${UNSTAGED_RULE_NAME} (slot ${slot})
struct Params { n: u32, reserved0: u32, reserved1: u32, reserved2: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x0: array<f32>;
@group(0) @binding(2) var<storage, read> x1: array<f32>;
@group(0) @binding(3) var<storage, read> x2: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

${stagedLoopWgsl({
  workgroupSize: WORKGROUP_SIZE,
  count: "params.n",
  rounds: [[{ name: "t0", value: product }]],
  finish: [`out[i] = ${sum.value};`],
})}`;
};

const UNSTAGED_RULE = defineRule<UnstagedMatch, typeof UNSTAGED_RULE_NAME>({
  name: UNSTAGED_RULE_NAME,
  heads: ["mul"],
  contract: "E",
  internals: "stayInChain",
  match: unstagedMatch,
  build: (matched) => {
    const product = matched.chain[1].outputs[0];
    return {
      binds: matched.binds,
      outputName: product.name,
      outputShape: product.shape,
      temps: [],
      dispatches: [
        {
          key: `ext:test:ew_chain_unstaged:v1:${matched.slot}:f32:wg${WORKGROUP_SIZE}`,
          wgsl: () => unstagedWgsl(matched.slot),
          params: new Uint32Array([matched.count, 0, 0, 0]),
          workgroups: { kind: "gridStride", items: matched.count, size: WORKGROUP_SIZE },
        },
      ],
    };
  },
});

/** The `mul → add` tails with every operand of the activation's shape (the fault's form). */
const FAULT_CASES: readonly Case[] = [
  mulAdd("full-shape mul → add(·, product)", FULL, [FULL, FULL, FULL], 1),
  mulAdd("full-shape mul → add(product, ·)", FULL, [FULL, FULL, FULL], 0),
];

describe.skipIf(!GPU_AVAILABLE)(
  "the candidate rule candidate/ew_chain against its unfused plan (real GPU)",
  () => {
    it("is u32-identical to the plain nodes in every form, runs each graph as its one dispatch and counts once", async () => {
      const gpu = await acquireGpu();
      try {
        for (const testCase of [...CASES, ...FAULT_CASES]) {
          const where = testCase.name;
          const fused = await runSide(gpu, testCase, FUSED);
          const plain = await runSide(gpu, testCase, PLAIN);
          assertEquals(
            [fused.hits[EW_CHAIN_RULE_NAME], plain.hits[EW_CHAIN_RULE_NAME]],
            [1, 0],
            `${where}: the rule's counter`,
          );
          assertEquals(fused.pipelines.length, 1, `${where}: the fused side's pipelines`);
          const [{ key, dispatchCount }] = fused.pipelines;
          assert(key.startsWith(`ext:candidate:ew_chain:`), `${where}: ${key}`);
          assertEquals(dispatchCount, 1, `${where}: the fused side's dispatches`);
          if (testCase.key !== undefined) assertEquals(key, testCase.key, `${where}: the form`);
          assertEquals(
            plain.pipelines.reduce((sum, pipeline) => sum + pipeline.dispatchCount, 0),
            testCase.nodes.length,
            `${where}: the plain side runs every node`,
          );
          assertEquals(Object.keys(fused.outputs), ["y"], where);
          assertEquals(
            wordIdentity(wordsOf(fused.outputs.y), wordsOf(plain.outputs.y), "class"),
            { words: elements(testCase.shape), failures: 0 },
            `${where}: the fused words against the plain words`,
          );
        }
      } finally {
        gpu.destroy();
      }
    });

    it("fault injection: the same kernel with the product read inline by the add differs from the plain nodes", async () => {
      const gpu = await acquireGpu();
      try {
        let failures = 0;
        for (const testCase of FAULT_CASES) {
          const where = testCase.name;
          const unstaged = await runSide(gpu, testCase, { fusionRules: [UNSTAGED_RULE] });
          const plain = await runSide(gpu, testCase, {});
          assertEquals(unstaged.hits[UNSTAGED_RULE_NAME], 1, `${where}: the fault's counter`);
          const identity = wordIdentity(
            wordsOf(unstaged.outputs.y),
            wordsOf(plain.outputs.y),
            "class",
          );
          console.log(
            `[ew_chain] fault ${where}: ${identity.failures} of ${identity.words} words differ`,
          );
          failures += identity.failures;
        }
        assert(failures > 0, "the unstaged kernel matched the plain nodes on every word");
      } finally {
        gpu.destroy();
      }
    });
  },
);
