/**
 * Parallel GEMV keys planned for every linear of the real QAT E4B graph (no GPU). It lives in e2e
 * because it reads the distribution mirror `models/karume-gemma4-qat-mobile/`; the asset-free census of the
 * same kernel is in `packages/runtime/tests/gpu_linear_gemv_parallel_test.ts`.
 */
import { isNotFound, readTextFile, stat } from "@karume/test-support/fs";
import { assertEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import type { IrGraph } from "../../runtime/src/format/ir.ts";
import {
  linearGemvPackedEligible,
  linearGemvParallelEligible,
  linearGemvParallelKey,
} from "../../runtime/src/kernels/linear-gemv.ts";
import { bindSymbols, planGraph } from "../../runtime/src/runtime/plan.ts";
import { planWeightResidency } from "../../runtime/src/runtime/weight-residency.ts";
import {
  type ContainerManifest,
  containerPart0,
  readContainerGraph,
} from "../../runtime/tests/helpers/container-graph.ts";

// ---------------------------------------------------------------------------
// The QAT E4B product graph (added 2026-09-26, no GPU needed)
// ---------------------------------------------------------------------------

/**
 * The parallel GEMV measurement table (`PARALLEL_SHAPES`) is an allowlist of shapes; a shape not
 * on it silently falls back to the sequential GEMV even when `parallel` is requested (ADR 0058
 * decision 3). E4B rows were added, so on the distribution mirror's **actual** QAT E4B graph
 * (part 0 = only the graph description and the binding table; the weight blocks are not read)
 * this pins which key each linear lands on in the plan (the same predicate as recipe-builder).
 * The regular E4B has no distribution mirror yet, so it is not covered.
 */
const QAT_ROOT = new URL("../../../models/karume-gemma4-qat-mobile/", import.meta.url);
const QAT_MANIFEST = new URL("karume.json", QAT_ROOT);

/**
 * Whether the asset exists (anything but NotFound propagates — a mount fault does not become a
 * SKIP).
 */
const exists = async (url: URL): Promise<boolean> => {
  try {
    await stat(url);
    return true;
  } catch (cause) {
    if (isNotFound(cause)) return false;
    throw cause;
  }
};

const QAT_AVAILABLE = await exists(QAT_MANIFEST);
if (!QAT_AVAILABLE) {
  console.warn(
    `[karume] ${QAT_MANIFEST.pathname} is missing, so the QAT E4B parallel GEMV product key check is SKIPPED`,
  );
}

/**
 * A linear that does not land on the parallel GEMV (sequential GEMV / row block / prefill — not
 * told apart here).
 */
const NOT_PARALLEL = "(not parallel)";

/** Folds the linears of a decode plan (`rows` physical rows) into parallel GEMV key → count. */
const planParallelKeys = (
  graph: IrGraph,
  rows: number,
): { readonly keys: Record<string, number>; readonly packed: number } => {
  const residency = planWeightResidency(graph);
  const inputShapes = Object.fromEntries(
    graph.inputs.map((spec) => [
      spec.name,
      spec.shape.map((dim) => (typeof dim === "number" ? dim : rows)),
    ]),
  );
  const stateShapes = new Map(
    Object.entries(graph.states).map(([name, slot]) => [
      name,
      slot.shape.map((dim) => (typeof dim === "number" ? dim : 640)),
    ]),
  );
  const plan = planGraph(graph, bindSymbols(graph, inputShapes), stateShapes);
  const keys = new Map<string, number>();
  let packed = 0;
  for (const step of plan.nodes) {
    if (step.node.op !== "linear") continue;
    const [x, weight] = step.inputShapes;
    const m = x.slice(0, -1).reduce((product, dim) => product * dim, 1);
    const seat = residency.get(step.node.ins[1]);
    const storage =
      seat?.seat === "i2" || seat?.seat === "i4" || seat?.seat === "i8" ? seat.seat : undefined;
    const group = seat?.seat === "i4" ? seat.groupSize : undefined;
    const lanes =
      storage === undefined
        ? undefined
        : linearGemvParallelEligible(storage, m, weight[0], weight[1], group);
    const key =
      storage === undefined || lanes === undefined
        ? NOT_PARALLEL
        : linearGemvParallelKey(storage, group, lanes);
    keys.set(key, (keys.get(key) ?? 0) + 1);
    if (
      storage !== undefined &&
      linearGemvPackedEligible(storage, m, weight[0], weight[1], group) !== undefined
    )
      packed += 1;
  }
  return { keys: Object.fromEntries([...keys].toSorted(([a], [b]) => (a < b ? -1 : 1))), packed };
};

describe.skipIf(!QAT_AVAILABLE)(
  "parallel GEMV on the QAT E4B product graph (plan, no GPU needed)",
  () => {
    it("every linear except the i2 lm_head and the f32 per_layer_model_projection is on an E4B row", async () => {
      const manifest: ContainerManifest = JSON.parse(await readTextFile(QAT_MANIFEST));
      const head = containerPart0(manifest, "e4b", "model", "i4");
      const graph = await readContainerGraph(new URL(head.path, QAT_ROOT), "model");
      // Breakdown of the counts (42 layers, 7 of them full): i4 g512 l4 = q 35 + q(full) 7 +
      // gate/up 84 / i4 g512 l32 = k/v 40 + k/v(full) 8 / i4 g2048 l32 = o 35 + down 42 / i4 g4096
      // l32 = o(full) 7 / i8 l32 = per_layer_input_gate 42 / i8 l4 = per_layer_projection 42 / not
      // covered 2.
      const expected = Object.fromEntries(
        (
          [
            [linearGemvParallelKey("i4", 512, 4), 126],
            [linearGemvParallelKey("i4", 512, 32), 48],
            [linearGemvParallelKey("i4", 2048, 32), 77],
            [linearGemvParallelKey("i4", 4096, 32), 7],
            [linearGemvParallelKey("i8", undefined, 32), 42],
            [linearGemvParallelKey("i8", undefined, 4), 42],
            [NOT_PARALLEL, 2],
          ] as const
        ).toSorted(([a], [b]) => (a < b ? -1 : 1)),
      );
      for (const rows of [1, 4, 8]) {
        const { keys, packed } = planParallelKeys(graph, rows);
        assertEquals(keys, expected, `M=${rows}`);
        // Packed int8 activations (ADR 0105) cover the packedActivations rows (l32 of g2048 /
        // g4096).
        assertEquals(packed, 84, `M=${rows} packed`);
      }
      // M>8 is outside the parallel GEMV (ADR 0098 decision 2).
      assertEquals(planParallelKeys(graph, 9), { keys: { [NOT_PARALLEL]: 344 }, packed: 0 });
    });
  },
);
