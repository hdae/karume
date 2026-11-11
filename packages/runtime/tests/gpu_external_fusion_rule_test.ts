// A fusion rule written outside the runtime, on the GPU (ADR 0148): the test-support rule
// `test/gelu_mul` (packages/models/tests/helpers/gelu-mul-rule.ts — `gelu_tanh` then `mul` in one
// dispatch, written against `@karume/runtime/kernel` alone) passed through
// `SessionOptions.fusionRules`, against the same Session options with the rule disabled by the
// test-only `DISABLED_FUSION_RULES` (the plain side). `checkFusionTwin` compares every output as u32
// words (NaN by class) and reads the rule's own counter: its hits on the fused side, 0 on the plain
// side. Small synthetic graphs: the two windows the rule takes (the nodes adjacent in both mul
// orders, and one independent linear between them), a partial last workgroup, and inputs past the
// tanh saturation with ±0, a subnormal, ±Inf and a NaN payload. No adapter = explicit SKIP.

import { describe, it } from "vite-plus/test";
import { type CheckCase, checkFusionTwin } from "@karume/kernel-checks";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import {
  createSessionFromContainer,
  DISABLED_FUSION_RULES,
  type Session,
  type SessionOptions,
  type Tensor,
} from "../src/runtime/executor.ts";
import { GELU_MUL_RULE, GELU_MUL_RULE_NAME } from "../../models/tests/helpers/gelu-mul-rule.ts";
import type { TensorInput } from "./helpers/container-write.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { assertCheckCase } from "./helpers/kernel-check-assert.ts";
import { type DeclarationJson, fill, GRAPH_NAME, openModelBytes } from "./helpers/model-fixture.ts";

/** Rows × width: 771 elements leave a partial last workgroup of the rule's 128 threads. */
const ROWS = 3;
const WIDTH = 257;
/** The input width of the linear between the two nodes. */
const K = 64;

type Form = "gelu-other" | "other-gelu" | "over-linear";

/**
 * `gelu_tanh(gate)` and a `mul` with `other`: adjacent in either operand order, or with
 * `other = linear(x)` computed between them (the dense MLP's order).
 */
const graphOf = (form: Form): DeclarationJson => {
  const row = [1, ROWS, WIDTH];
  const values: DeclarationJson["values"] = {
    h: { dtype: "f32", shape: row },
    y: { dtype: "f32", shape: row },
  };
  const inputs: DeclarationJson["inputs"] = [{ name: "gate", dtype: "f32", shape: row }];
  const nodes: DeclarationJson["nodes"] = [
    { op: "gelu_tanh", ins: ["gate"], outs: ["h"], attrs: {} },
  ];
  const initializers: DeclarationJson["initializers"] = {};
  if (form === "over-linear") {
    inputs.push({ name: "x", dtype: "f32", shape: [1, ROWS, K] });
    values.w = { dtype: "f32", shape: [WIDTH, K] };
    values.b = { dtype: "f32", shape: [WIDTH] };
    values.other = { dtype: "f32", shape: row };
    initializers.w = {};
    initializers.b = {};
    nodes.push({ op: "linear", ins: ["x", "w", "b"], outs: ["other"], attrs: {} });
  } else {
    inputs.push({ name: "other", dtype: "f32", shape: row });
  }
  nodes.push({
    op: "mul",
    ins: form === "other-gelu" ? ["other", "h"] : ["h", "other"],
    outs: ["y"],
    attrs: {},
  });
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: [...new Set(nodes.map((node) => node.op))].sort() },
    symbols: [],
    inputs,
    outputs: ["y"],
    initializers,
    values,
    nodes,
  };
};

/** The linear's f32 weight and bias (the over-linear form only). */
const weightsOf = (form: Form): readonly TensorInput[] => {
  if (form !== "over-linear") return [];
  const weight = fill([WIDTH, K], (i) => Math.sin(i * 0.61) * 0.25);
  const bias = fill([WIDTH], (i) => Math.cos(i * 0.7) * 0.5);
  return [
    {
      graph: GRAPH_NAME,
      initializer: "w",
      bytes: new Uint8Array(weight.data.buffer),
      encoding: { codec: "f32" },
    },
    {
      graph: GRAPH_NAME,
      initializer: "b",
      bytes: new Uint8Array(bias.data.buffer),
      encoding: { codec: "f32" },
    },
  ];
};

/** The words written into the first elements of `gate` and `other`. */
const SPECIAL_WORDS = [
  0x00000000, // +0
  0x80000000, // -0
  0x00000001, // the smallest subnormal
  0x7f800000, // +Inf
  0xff800000, // -Inf
  0x7fc01234, // a NaN payload
  0x7f7fffff, // the largest finite
];

/** Values past the tanh saturation (|x| ≥ 9.5) both ways, and small ones, then the special words. */
const feedsOf = (form: Form): Readonly<Record<string, Tensor>> => {
  const shape = [1, ROWS, WIDTH];
  const gate = fill(shape, (i) => Math.sin(i * 0.37) * 14.5 + i / 997);
  new Uint32Array(gate.data.buffer).set(SPECIAL_WORDS);
  if (form === "over-linear") {
    return { gate, x: fill([1, ROWS, K], (i) => Math.cos(i * 0.29) * 1.7) };
  }
  const other = fill(shape, (i) => Math.cos(i * 0.53) * 3.3 - i / 1201);
  new Uint32Array(other.data.buffer).set(SPECIAL_WORDS.toReversed(), SPECIAL_WORDS.length);
  return { gate, other };
};

const open = async (gpu: GpuContext, form: Form, options: SessionOptions): Promise<Session> =>
  await createSessionFromContainer(
    gpu,
    await openModelBytes(graphOf(form), weightsOf(form)),
    GRAPH_NAME,
    options,
  );

/** The fused side's options: the rule given. The plain side: the same, with the rule disabled. */
const FUSED: SessionOptions = { fusionRules: [GELU_MUL_RULE] };
const PLAIN: SessionOptions = { ...FUSED, [DISABLED_FUSION_RULES]: [GELU_MUL_RULE_NAME] };

const twin = async (gpu: GpuContext, form: Form): Promise<CheckCase> => {
  const fused = await open(gpu, form, FUSED);
  try {
    const plain = await open(gpu, form, PLAIN);
    try {
      return await checkFusionTwin(gpu, {
        label: `test/gelu_mul ${form}`,
        fused,
        plain,
        feeds: feedsOf(form),
        counter: GELU_MUL_RULE_NAME,
        hits: 1,
        nan: "class",
        // gelu_tanh + mul unfused, one dispatch fused; the linear's own dispatches are not counted.
        ...(form === "over-linear" ? {} : { dispatches: { fused: 1, plain: 2 } }),
      });
    } finally {
      await plain.dispose();
    }
  } finally {
    await fused.dispose();
  }
};

describe.skipIf(!GPU_AVAILABLE)(
  "an external fusion rule against its unfused plan (real GPU)",
  () => {
    it("test/gelu_mul is u32-identical to the plain gelu_tanh and mul in each window it takes, and counts once", async () => {
      const gpu = await acquireGpu();
      try {
        for (const form of ["gelu-other", "other-gelu", "over-linear"] as const) {
          assertCheckCase(await twin(gpu, form));
        }
      } finally {
        gpu.destroy();
      }
    });
  },
);
