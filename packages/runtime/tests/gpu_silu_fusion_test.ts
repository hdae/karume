// The strict peephole for SiLU = x * sigmoid(x) (fusion rule silu — src/runtime/fusion.ts).
// Only the two adjacent nodes the exporter emits are folded into one dispatch; any shape the
// matcher does not catch falls back to the plain sequence.

import { test } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import { acquireGpu } from "../src/gpu/device.ts";
import { siluKey } from "../src/kernels/silu.ts";
import { createSessionFromContainer, type Tensor } from "../src/runtime/executor.ts";
import { type DeclarationJson, fill, GRAPH_NAME, openModelBytes } from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { requireCensus } from "./helpers/pipeline-census.ts";

// A length that leaves a partial 256-thread workgroup.
const SHAPE = [257] as const;

type MulOrder = "xs" | "sx";

type SiluGraphOptions = {
  readonly interpose?: boolean;
  readonly sigmoidOutput?: boolean;
  readonly extraConsumer?: boolean;
  readonly xExtraConsumer?: boolean;
  readonly otherMul?: boolean;
};

/**
 * The real exported form of `sigmoid(x) → mul(x, sigmoid(x))`.
 * `interpose` only inserts one zero-dispatch alias, which breaks the strict matcher's adjacency
 * condition without changing the values or the physical dispatch count (= it builds the
 * reference on the same backend).
 */
const siluGraph = (order: MulOrder, options: SiluGraphOptions = {}): DeclarationJson => {
  const sigmoidName = "sigmoid";
  const mulSigmoidName = options.interpose ? "sigmoid_alias" : sigmoidName;
  const values: DeclarationJson["values"] = {
    sigmoid: { dtype: "f32", shape: [...SHAPE] },
    y: { dtype: "f32", shape: [...SHAPE] },
  };
  const inputs: DeclarationJson["inputs"] = [{ name: "x", dtype: "f32", shape: [...SHAPE] }];
  if (options.interpose) values.sigmoid_alias = { dtype: "f32", shape: [...SHAPE] };
  if (options.extraConsumer) values.sigmoid_copy = { dtype: "f32", shape: [...SHAPE] };
  if (options.xExtraConsumer) values.x_copy = { dtype: "f32", shape: [...SHAPE] };
  if (options.otherMul) {
    inputs.push({ name: "z", dtype: "f32", shape: [...SHAPE] });
  }

  const nodes: DeclarationJson["nodes"] = [
    {
      op: "sigmoid",
      ins: ["x"],
      outs: [sigmoidName],
      attrs: {},
    },
  ];
  if (options.interpose) {
    nodes.push({ op: "reshape", ins: [sigmoidName], outs: [mulSigmoidName], attrs: {} });
  }
  const mulInputs = options.otherMul
    ? [mulSigmoidName, "z"]
    : order === "xs"
      ? ["x", mulSigmoidName]
      : [mulSigmoidName, "x"];
  nodes.push({ op: "mul", ins: mulInputs, outs: ["y"], attrs: {} });
  if (options.extraConsumer) {
    nodes.push({ op: "neg", ins: [sigmoidName], outs: ["sigmoid_copy"], attrs: {} });
  }
  if (options.xExtraConsumer) {
    nodes.push({ op: "neg", ins: ["x"], outs: ["x_copy"], attrs: {} });
  }

  const outputs = [
    ...(options.sigmoidOutput ? [sigmoidName] : []),
    "y",
    ...(options.extraConsumer ? ["sigmoid_copy"] : []),
    ...(options.xExtraConsumer ? ["x_copy"] : []),
  ];
  return {
    format: "karume-ir",
    version: 2,
    requires: {
      ops: [
        "sigmoid",
        "mul",
        ...(options.interpose ? ["reshape"] : []),
        ...(options.extraConsumer || options.xExtraConsumer ? ["neg"] : []),
      ],
    },
    symbols: [],
    inputs,
    outputs,
    initializers: {},
    values,
    nodes,
  };
};

const words = (tensor: Tensor): Uint32Array =>
  new Uint32Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.length);

const isNanWord = (word: number): boolean => (word & 0x7fffffff) > 0x7f800000;

/**
 * Non-NaN values must be bit-identical; NaN only has to match in class, allowing backend
 * differences.
 */
const assertFloatParity = (actual: Uint32Array, expected: Uint32Array, label: string): void => {
  assertEquals(actual.length, expected.length, label + ": length");
  for (let i = 0; i < actual.length; i++) {
    if (isNanWord(expected[i])) {
      assertEquals(isNanWord(actual[i]), true, `${label}: NaN classification index=${i}`);
    } else {
      assertEquals(actual[i], expected[i], `${label}: word index=${i}`);
    }
  }
};

const testInput = (): Tensor => {
  const x = fill(SHAPE, (i) => Math.sin(i * 0.37) * 3.1 + i / 997);
  // Mix finite values, signed zeros, subnormals, the smallest normal, the overflow boundary, ±Inf
  // and NaN.
  words(x).set([
    0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007fffff, 0x807fffff, 0x00800000, 0x80800000,
    0x7f800000, 0xff800000, 0x7fc01234, 0xff801234,
  ]);
  x.data.set([80, -80, 88, -88, 100, -100], 12);
  return x;
};

test.skipIf(!GPU_AVAILABLE)(
  "SiLU fusion matches the primitives in finite bits and NaN class in both xs / sx orders and folds 2→1 dispatch (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const x = testInput();
    try {
      for (const order of ["xs", "sx"] as const) {
        const fused = await createSessionFromContainer(
          gpu,
          await openModelBytes(siluGraph(order), []),
          GRAPH_NAME,
        );
        const primitive = await createSessionFromContainer(
          gpu,
          await openModelBytes(siluGraph(order, { interpose: true }), []),
          GRAPH_NAME,
        );
        try {
          const fusedOut = (await fused.run({ x })).y;
          const primitiveOut = (await primitive.run({ x })).y;
          assertFloatParity(words(fusedOut), words(primitiveOut), `${order}: fused vs primitive`);
          assertEquals(fused.diagnostics().submit.dispatchCount, 1, `${order}: fused 1 dispatch`);
          assertEquals(
            primitive.diagnostics().submit.dispatchCount,
            2,
            `${order}: sigmoid + mul primitive`,
          );
          // If fusion silently stopped applying, the values would stay correct and only the
          // dispatch count would grow. Pin it directly with the counter.
          assertEquals(fused.diagnostics().lastRunFusions?.silu, 1, `${order}: fusion counter`);
          assertEquals(
            primitive.diagnostics().lastRunFusions?.silu,
            0,
            `${order}: the counter of the counterexample is 0`,
          );
          assertEquals(
            requireCensus(fused.diagnostics().lastRunPipelines, order).map((row) => row.key),
            [siluKey(order === "xs" ? "x-sigmoid" : "sigmoid-x")],
            `${order}: census key`,
          );
        } finally {
          await fused.dispose();
          await primitive.dispose();
        }
      }
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "SiLU keeps the input buffer alive until the later consumer even when x is released twice (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const x = testInput();
    const fused = await createSessionFromContainer(
      gpu,
      await openModelBytes(siluGraph("xs", { xExtraConsumer: true }), []),
      GRAPH_NAME,
    );
    const primitive = await createSessionFromContainer(
      gpu,
      await openModelBytes(siluGraph("xs", { interpose: true, xExtraConsumer: true }), []),
      GRAPH_NAME,
    );
    try {
      const fusedOut = await fused.run({ x });
      const primitiveOut = await primitive.run({ x });
      assertFloatParity(words(fusedOut.y), words(primitiveOut.y), "SiLU output");
      assertFloatParity(words(fusedOut.x_copy), words(primitiveOut.x_copy), "late x consumer");
      assertEquals(fused.diagnostics().submit.dispatchCount, 2, "SiLU + late consumer");
      assertEquals(primitive.diagnostics().submit.dispatchCount, 3, "primitive + late consumer");
    } finally {
      await fused.dispose();
      await primitive.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "SiLU with the inner sigmoid as an output / another consumer / an alias / another mul input falls back from the strict matcher (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const x = testInput();
    const z = fill(SHAPE, (i) => Math.cos(i * 0.19) * 2.7 - i / 911);
    const cases: readonly {
      readonly name: string;
      readonly graph: DeclarationJson;
      readonly inputs: Readonly<Record<string, Tensor>>;
      readonly dispatches: number;
    }[] = [
      {
        name: "internal output",
        graph: siluGraph("xs", { sigmoidOutput: true }),
        inputs: { x },
        dispatches: 2,
      },
      {
        name: "extra consumer",
        graph: siluGraph("sx", { extraConsumer: true }),
        inputs: { x },
        dispatches: 3,
      },
      {
        name: "interposed alias",
        graph: siluGraph("xs", { interpose: true }),
        inputs: { x },
        dispatches: 2,
      },
      {
        name: "mul(s,z)",
        graph: siluGraph("xs", { otherMul: true }),
        inputs: { x, z },
        dispatches: 2,
      },
    ];
    try {
      for (const testCase of cases) {
        const session = await createSessionFromContainer(
          gpu,
          await openModelBytes(testCase.graph, []),
          GRAPH_NAME,
        );
        try {
          await session.run(testCase.inputs);
          assertEquals(
            session.diagnostics().submit.dispatchCount,
            testCase.dispatches,
            `${testCase.name}: fallback dispatch count`,
          );
          assertEquals(
            session.diagnostics().lastRunFusions?.silu,
            0,
            `${testCase.name}: fusion counter 0`,
          );
        } finally {
          await session.dispose();
        }
      }
    } finally {
      gpu.destroy();
    }
  },
);
