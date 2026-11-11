// The fused-vs-unfused identity check `checkFusionTwin` (packages/kernel-checks/src/fusion-twin.ts)
// on the silu rule: its plain side made by the test-only `DISABLED_FUSION_RULES` key or by a twin
// graph, and the three ways a twin comparison can be wrong, kept as standing cases — a fused side
// that computes something else, a plain side that fuses too, a fused side whose rule did not fire —
// each failing its own criterion and no other (and a wrong dispatch expectation failing the dispatch
// criteria). No adapter = explicit SKIP.

import { describe, it } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import { type CheckCase, checkFusionTwin } from "@karume/kernel-checks";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import {
  createSessionFromContainer,
  DISABLED_FUSION_RULES,
  type Session,
  type SessionOptions,
  type Tensor,
} from "../src/runtime/executor.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { assertCheckCase } from "./helpers/kernel-check-assert.ts";
import { type DeclarationJson, fill, GRAPH_NAME, openModelBytes } from "./helpers/model-fixture.ts";

// A length that leaves a partial 256-thread workgroup.
const SHAPE = [257];

type SiluOptions = {
  /** A zero-dispatch reshape between sigmoid and mul: the matcher stops matching. */
  readonly interpose?: boolean;
  /** A neg after the mul: the graph computes `-silu(x)` (and still folds the silu). */
  readonly negate?: boolean;
};

/** `y = x * sigmoid(x)`, the silu rule's form. */
const siluGraph = (options: SiluOptions = {}): DeclarationJson => {
  const values: DeclarationJson["values"] = {
    s: { dtype: "f32", shape: SHAPE },
    y: { dtype: "f32", shape: SHAPE },
  };
  const nodes: DeclarationJson["nodes"] = [{ op: "sigmoid", ins: ["x"], outs: ["s"], attrs: {} }];
  if (options.interpose) {
    values.s_alias = { dtype: "f32", shape: SHAPE };
    nodes.push({ op: "reshape", ins: ["s"], outs: ["s_alias"], attrs: {} });
  }
  const product = options.negate ? "p" : "y";
  if (options.negate) values.p = { dtype: "f32", shape: SHAPE };
  nodes.push({
    op: "mul",
    ins: ["x", options.interpose ? "s_alias" : "s"],
    outs: [product],
    attrs: {},
  });
  if (options.negate) nodes.push({ op: "neg", ins: ["p"], outs: ["y"], attrs: {} });
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: [...new Set(nodes.map((node) => node.op))] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: SHAPE }],
    outputs: ["y"],
    initializers: {},
    values,
    nodes,
  };
};

/** Finite values with ±0, a subnormal, ±Inf and a NaN payload in the first words. */
const feeds = (): Readonly<Record<string, Tensor>> => {
  const x = fill(SHAPE, (i) => Math.sin(i * 0.37) * 3.1 + i / 997);
  new Uint32Array(x.data.buffer).set([
    0x00000000, 0x80000000, 0x00000001, 0x7f800000, 0xff800000, 0x7fc01234,
  ]);
  return { x };
};

const open = async (
  gpu: GpuContext,
  graph: DeclarationJson,
  options: SessionOptions = {},
): Promise<Session> =>
  await createSessionFromContainer(gpu, await openModelBytes(graph, []), GRAPH_NAME, options);

/** sigmoid + mul unfused, one dispatch fused. */
const SILU_DISPATCHES = { fused: 1, plain: 2 } as const;

/** Runs `checkFusionTwin` on two Sessions opened here, and disposes them. */
const twin = async (
  gpu: GpuContext,
  fused: () => Promise<Session>,
  plain: () => Promise<Session>,
  label: string,
  dispatches?: { readonly fused: number; readonly plain: number },
): Promise<CheckCase> => {
  const fusedSession = await fused();
  const plainSession = await plain();
  try {
    return await checkFusionTwin(gpu, {
      label,
      fused: fusedSession,
      plain: plainSession,
      feeds: feeds(),
      counter: "silu",
      hits: 1,
      nan: "class",
      ...(dispatches === undefined ? {} : { dispatches }),
    });
  } finally {
    await fusedSession.dispose();
    await plainSession.dispose();
  }
};

const failing = (checkCase: CheckCase): readonly string[] =>
  checkCase.criteria
    .filter((criterion) => criterion.failures > 0)
    .map((criterion) => criterion.name);

describe.skipIf(!GPU_AVAILABLE)("checkFusionTwin (real GPU)", () => {
  it("passes every criterion with the plain side made by DISABLED_FUSION_RULES ('all' or the rule's name) on the same graph, or by a twin graph", async () => {
    const gpu = await acquireGpu();
    try {
      for (const [label, plain] of [
        [
          "DISABLED_FUSION_RULES 'all'",
          () => open(gpu, siluGraph(), { [DISABLED_FUSION_RULES]: "all" }),
        ],
        [
          "DISABLED_FUSION_RULES ['silu']",
          () => open(gpu, siluGraph(), { [DISABLED_FUSION_RULES]: ["silu"] }),
        ],
        ["interposed reshape", () => open(gpu, siluGraph({ interpose: true }))],
      ] as const) {
        const checkCase = await twin(
          gpu,
          () => open(gpu, siluGraph()),
          plain,
          label,
          SILU_DISPATCHES,
        );
        assertEquals(
          checkCase.criteria.map((criterion) => criterion.name),
          [
            "output y (NaN by class)",
            "fused silu",
            "plain silu",
            "fused dispatches",
            "plain dispatches",
          ],
          label,
        );
        assertCheckCase(checkCase);
      }
    } finally {
      gpu.destroy();
    }
  });

  it("fails only the output's words, with a failing word, when the fused side computes something else", async () => {
    const gpu = await acquireGpu();
    try {
      const checkCase = await twin(
        gpu,
        () => open(gpu, siluGraph({ negate: true })),
        () => open(gpu, siluGraph(), { [DISABLED_FUSION_RULES]: "all" }),
        "fused side negated",
      );
      assertEquals(failing(checkCase), ["output y (NaN by class)"]);
      assertEquals(checkCase.criteria[0].failure?.at, "first", "a failing word is reported");
    } finally {
      gpu.destroy();
    }
  });

  it("fails only the plain counter when the plain side fuses too (the words then agree)", async () => {
    const gpu = await acquireGpu();
    try {
      const checkCase = await twin(
        gpu,
        () => open(gpu, siluGraph()),
        () => open(gpu, siluGraph()),
        "plain side fused",
      );
      assertEquals(failing(checkCase), ["plain silu"]);
    } finally {
      gpu.destroy();
    }
  });

  it("fails only the fused counter when the fused side's rule did not fire (the words then agree)", async () => {
    const gpu = await acquireGpu();
    try {
      const checkCase = await twin(
        gpu,
        () => open(gpu, siluGraph({ interpose: true })),
        () => open(gpu, siluGraph(), { [DISABLED_FUSION_RULES]: "all" }),
        "fused side unfused",
      );
      assertEquals(failing(checkCase), ["fused silu"]);
    } finally {
      gpu.destroy();
    }
  });

  it("fails only the dispatch criteria when a side records other dispatches than expected", async () => {
    const gpu = await acquireGpu();
    try {
      const checkCase = await twin(
        gpu,
        () => open(gpu, siluGraph()),
        () => open(gpu, siluGraph({ interpose: true })),
        "dispatches swapped",
        { fused: SILU_DISPATCHES.plain, plain: SILU_DISPATCHES.fused },
      );
      assertEquals(failing(checkCase), ["fused dispatches", "plain dispatches"]);
    } finally {
      gpu.destroy();
    }
  });
});
