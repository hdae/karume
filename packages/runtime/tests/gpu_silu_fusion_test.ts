// The strict peephole for SiLU = x * sigmoid(x) (fusion rule silu — src/runtime/fusion.ts).
// Only the two adjacent nodes the exporter emits are folded into one dispatch; any shape the
// matcher does not catch falls back to the plain sequence.
//
// The twin cases live in @karume/kernel-checks (checkSiluTwins), which the playground's
// /kernel-check page runs on the browser's backend too; the census and the fallback cases stay here.

import { test } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import {
  checkSiluTwins,
  SILU_ORDERS,
  SILU_SHAPE,
  SILU_TWIN_CASES,
  siluGraph,
  siluInput,
} from "@karume/kernel-checks";
import { acquireGpu } from "../src/gpu/device.ts";
import { siluKey } from "../src/kernels/silu.ts";
import { createSessionFromContainer, type Tensor } from "../src/runtime/executor.ts";
import { type DeclarationJson, fill, GRAPH_NAME, openModelBytes } from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { assertCheckPart } from "./helpers/kernel-check-assert.ts";
import { requireCensus } from "./helpers/pipeline-census.ts";

test.skipIf(!GPU_AVAILABLE)(
  "SiLU fusion matches the primitives in finite bits and NaN class in both xs / sx orders and folds 2→1 dispatch, and keeps x alive for a later consumer (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      const part = await checkSiluTwins(gpu);
      assertEquals([part.id, part.cases.length], Object.entries(SILU_TWIN_CASES)[0]);
      assertCheckPart(part);
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "SiLU fusion runs only the SiLU kernel of its order (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const x = siluInput();
    try {
      for (const order of SILU_ORDERS) {
        const fused = await createSessionFromContainer(
          gpu,
          await openModelBytes(siluGraph(order), []),
          GRAPH_NAME,
        );
        try {
          await fused.run({ x });
          assertEquals(
            requireCensus(fused.diagnostics().lastRunPipelines, order).map((row) => row.key),
            [siluKey(order === "xs" ? "x-sigmoid" : "sigmoid-x")],
            `${order}: census key`,
          );
        } finally {
          await fused.dispose();
        }
      }
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "SiLU with the inner sigmoid as an output / another consumer / an alias / another mul input falls back from the strict matcher (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const x = siluInput();
    const z = fill(SILU_SHAPE, (i) => Math.cos(i * 0.19) * 2.7 - i / 911);
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
