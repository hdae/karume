/**
 * The strict peephole for SiLU = x * sigmoid(x) (fusion rule silu — the runtime's
 * src/runtime/fusion.ts) against the primitive sequence on the same backend: u32 identity with NaN
 * by class in both mul orders, 2 dispatches folded into 1, and the input kept alive for a later
 * consumer of x.
 */

import type { GpuContext } from "../../../runtime/src/gpu/device.ts";
import {
  type DeclarationJson,
  fill,
  type FilledTensor,
  openModelBytes,
} from "../../../runtime/tests/helpers/model-fixture.ts";
import { checkFusionTwin } from "../fusion-twin.ts";
import { type CheckOptions, type CheckPart, type ExpectedCases, wordsOf } from "../result.ts";
import { twinCases, withTwinSessions } from "./session.ts";

/** A length that leaves a partial 256-thread workgroup. */
export const SILU_SHAPE = [257] as const;

export type SiluMulOrder = "xs" | "sx";

export type SiluGraphOptions = {
  readonly interpose?: boolean;
  readonly sigmoidOutput?: boolean;
  readonly extraConsumer?: boolean;
  readonly xExtraConsumer?: boolean;
  readonly otherMul?: boolean;
};

/**
 * The real exported form of `sigmoid(x) → mul(x, sigmoid(x))`.
 * `interpose` only inserts one zero-dispatch alias, which breaks the strict matcher's adjacency
 * condition without changing the values or the physical dispatch count (= it builds the reference
 * on the same backend).
 */
export const siluGraph = (order: SiluMulOrder, options: SiluGraphOptions = {}): DeclarationJson => {
  const sigmoidName = "sigmoid";
  const mulSigmoidName = options.interpose ? "sigmoid_alias" : sigmoidName;
  const values: DeclarationJson["values"] = {
    sigmoid: { dtype: "f32", shape: [...SILU_SHAPE] },
    y: { dtype: "f32", shape: [...SILU_SHAPE] },
  };
  const inputs: DeclarationJson["inputs"] = [{ name: "x", dtype: "f32", shape: [...SILU_SHAPE] }];
  if (options.interpose) values.sigmoid_alias = { dtype: "f32", shape: [...SILU_SHAPE] };
  if (options.extraConsumer) values.sigmoid_copy = { dtype: "f32", shape: [...SILU_SHAPE] };
  if (options.xExtraConsumer) values.x_copy = { dtype: "f32", shape: [...SILU_SHAPE] };
  if (options.otherMul) {
    inputs.push({ name: "z", dtype: "f32", shape: [...SILU_SHAPE] });
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

export const siluInput = (): FilledTensor => {
  const x = fill(SILU_SHAPE, (i) => Math.sin(i * 0.37) * 3.1 + i / 997);
  // Mix finite values, signed zeros, subnormals, the smallest normal, the overflow boundary, ±Inf
  // and NaN.
  wordsOf(x.data).set([
    0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007fffff, 0x807fffff, 0x00800000, 0x80800000,
    0x7f800000, 0xff800000, 0x7fc01234, 0xff801234,
  ]);
  x.data.set([80, -80, 88, -88, 100, -100], 12);
  return x;
};

export const SILU_ORDERS: readonly SiluMulOrder[] = ["xs", "sx"];

const PART = "fusion-silu";

/** The cases of {@link checkSiluTwins}: both mul orders, and the late x consumer. */
export const SILU_TWIN_CASES = { [PART]: 3 } as const satisfies ExpectedCases;

/**
 * Both mul orders (one dispatch against sigmoid + mul) and the SiLU with a later consumer of x (both
 * outputs; two dispatches against three): the fused Session against the interposed one
 * (checkFusionTwin, NaN by class), silu at 1 and 0.
 */
export const checkSiluTwins = async (
  gpu: GpuContext,
  options: CheckOptions = {},
): Promise<CheckPart> => {
  const sink = twinCases(PART, SILU_TWIN_CASES[PART], options);
  const x = siluInput();
  for (const order of SILU_ORDERS) {
    sink.next();
    await withTwinSessions(
      gpu,
      `SiLU ${order}`,
      { model: await openModelBytes(siluGraph(order), []) },
      { model: await openModelBytes(siluGraph(order, { interpose: true }), []) },
      async ({ fused, plain }) => {
        // If fusion silently stopped applying, the values would stay correct and only the dispatch
        // count would grow. Pin it directly with the counter.
        sink.add(
          await checkFusionTwin(gpu, {
            label: `${order}: fused vs primitive`,
            fused,
            plain,
            feeds: { x },
            counter: "silu",
            hits: 1,
            nan: "class",
            // One dispatch against sigmoid + mul.
            dispatches: { fused: 1, plain: 2 },
          }),
        );
      },
    );
  }
  sink.next();
  await withTwinSessions(
    gpu,
    "SiLU with a late x consumer",
    { model: await openModelBytes(siluGraph("xs", { xExtraConsumer: true }), []) },
    {
      model: await openModelBytes(siluGraph("xs", { interpose: true, xExtraConsumer: true }), []),
    },
    async ({ fused, plain }) => {
      // Both outputs: the SiLU output and the late x consumer.
      sink.add(
        await checkFusionTwin(gpu, {
          label: "SiLU and the late x consumer",
          fused,
          plain,
          feeds: { x },
          counter: "silu",
          hits: 1,
          nan: "class",
          // SiLU + late consumer against the primitives + late consumer.
          dispatches: { fused: 2, plain: 3 },
        }),
      );
    },
  );
  return {
    id: PART,
    title: "silu: fused vs sigmoid → mul",
    cases: sink.cases,
    requirements: [],
  };
};
