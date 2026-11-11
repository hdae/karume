// A fusion rule written outside the runtime's source, on the GPU (ADR 0148): the test rule
// `test/gelu_mul` of the runtime's own test tree (tests/helpers/external-gelu-mul-rule.ts — `gelu`
// then `mul` in one dispatch, written against the kernel entry alone) passed through
// `SessionOptions.fusionRules`, against the same Session options with the rule disabled by the
// test-only `DISABLED_FUSION_RULES` (the plain side). `checkFusionTwin` compares every output as u32
// words (NaN by class) and reads the rule's own counter: one hit on the fused side, 0 on the plain
// side, with one dispatch against the plain pair's two. Small synthetic graphs: both mul operand
// orders, a partial last workgroup, and inputs with ±0, a subnormal, ±Inf and a NaN payload. The
// models package's `gelu_tanh` rule is checked the same way by the e2e package
// (packages/e2e/tests/e2e_external_fusion_rule_test.ts). No adapter = explicit SKIP.

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
import {
  EXTERNAL_GELU_MUL_RULE,
  EXTERNAL_GELU_MUL_RULE_NAME,
} from "./helpers/external-gelu-mul-rule.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { assertCheckCase } from "./helpers/kernel-check-assert.ts";
import { type DeclarationJson, fill, GRAPH_NAME, openModelBytes } from "./helpers/model-fixture.ts";

/** Rows × width: 771 elements leave a partial last workgroup of the rule's 128 threads. */
const ROWS = 3;
const WIDTH = 257;

type Order = "gelu-other" | "other-gelu";

/** `y = mul(gelu(gate), other)`, the gelu value in the mul's slot `order` names. */
const graphOf = (order: Order): DeclarationJson => {
  const row = [1, ROWS, WIDTH];
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["gelu", "mul"] },
    symbols: [],
    inputs: [
      { name: "gate", dtype: "f32", shape: row },
      { name: "other", dtype: "f32", shape: row },
    ],
    outputs: ["y"],
    initializers: {},
    values: { h: { dtype: "f32", shape: row }, y: { dtype: "f32", shape: row } },
    nodes: [
      { op: "gelu", ins: ["gate"], outs: ["h"], attrs: {} },
      {
        op: "mul",
        ins: order === "other-gelu" ? ["other", "h"] : ["h", "other"],
        outs: ["y"],
        attrs: {},
      },
    ],
  };
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

const feeds = (): Readonly<Record<string, Tensor>> => {
  const shape = [1, ROWS, WIDTH];
  const gate = fill(shape, (i) => Math.sin(i * 0.37) * 9.5 + i / 997);
  new Uint32Array(gate.data.buffer).set(SPECIAL_WORDS);
  const other = fill(shape, (i) => Math.cos(i * 0.53) * 3.3 - i / 1201);
  new Uint32Array(other.data.buffer).set(SPECIAL_WORDS.toReversed(), SPECIAL_WORDS.length);
  return { gate, other };
};

const open = async (gpu: GpuContext, order: Order, options: SessionOptions): Promise<Session> =>
  await createSessionFromContainer(
    gpu,
    await openModelBytes(graphOf(order), []),
    GRAPH_NAME,
    options,
  );

/** The fused side's options: the rule given. The plain side: the same, with the rule disabled. */
const FUSED: SessionOptions = { fusionRules: [EXTERNAL_GELU_MUL_RULE] };
const PLAIN: SessionOptions = { ...FUSED, [DISABLED_FUSION_RULES]: [EXTERNAL_GELU_MUL_RULE_NAME] };

const twin = async (gpu: GpuContext, order: Order): Promise<CheckCase> => {
  const fused = await open(gpu, order, FUSED);
  try {
    const plain = await open(gpu, order, PLAIN);
    try {
      return await checkFusionTwin(gpu, {
        label: `test/gelu_mul ${order}`,
        fused,
        plain,
        feeds: feeds(),
        counter: EXTERNAL_GELU_MUL_RULE_NAME,
        hits: 1,
        nan: "class",
        dispatches: { fused: 1, plain: 2 },
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
    it("test/gelu_mul is u32-identical to the plain gelu and mul in both operand orders, and counts once", async () => {
      const gpu = await acquireGpu();
      try {
        for (const order of ["gelu-other", "other-gelu"] as const) {
          assertCheckCase(await twin(gpu, order));
        }
      } finally {
        gpu.destroy();
      }
    });
  },
);
