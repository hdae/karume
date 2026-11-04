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
// QAT E4B の製品グラフ（2026-09-26 追記・GPU 不要）
// ---------------------------------------------------------------------------

/**
 * 並列 GEMV の実測表（`PARALLEL_SHAPES`）は形の allowlist で、載っていない形は `parallel` を
 * 指定しても逐次 GEMV へ黙って縮退する（ADR 0058 決定 3）。E4B の行を足したので、配布ミラーの
 * **実際の** QAT E4B グラフ（part 0 = グラフ記述と束縛表だけ・重みの block は読まない）で、
 * 各 linear が計画上どのキーへ落ちるか（recipe-builder と同じ述語）を固定する。
 * 通常 E4B は配布ミラーがまだ無いので対象外。
 */
const QAT_ROOT = new URL("../../../models/karume-gemma4-qat-mobile/", import.meta.url);
const QAT_MANIFEST = new URL("karume.json", QAT_ROOT);

/** 資産の有無（NotFound 以外は伝播 — マウント異常を SKIP に化かさない）。 */
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
    `[karume] ${QAT_MANIFEST.pathname} が無いため QAT E4B の並列 GEMV 製品キー検査を SKIP する`,
  );
}

/** 並列 GEMV へ落ちない linear（逐次 GEMV / 行ブロック / prefill — ここでは区別しない）。 */
const NOT_PARALLEL = "(not parallel)";

/** decode 計画（物理行数 `rows`）の linear を、並列 GEMV のキー → 本数に畳む。 */
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
  "QAT E4B の製品グラフでの並列 GEMV の適用（計画・GPU 不要）",
  () => {
    it("i2 の lm_head と f32 の per_layer_model_projection を除く全 linear が E4B の行に載る", async () => {
      const manifest: ContainerManifest = JSON.parse(await readTextFile(QAT_MANIFEST));
      const head = containerPart0(manifest, "e4b", "model", "i4");
      const graph = await readContainerGraph(new URL(head.path, QAT_ROOT), "model");
      // 本数の内訳（42 層・full 7 層）: i4 g512 l4 = q 35 + q(full) 7 + gate/up 84 /
      // i4 g512 l32 = k/v 40 + k/v(full) 8 / i4 g2048 l32 = o 35 + down 42 / i4 g4096 l32 = o(full) 7 /
      // i8 l32 = per_layer_input_gate 42 / i8 l4 = per_layer_projection 42 / 対象外 2。
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
        // packed int8 活性（ADR 0105）の対象 = packedActivations の行（g2048 / g4096 の l32）。
        assertEquals(packed, 84, `M=${rows} packed`);
      }
      // M>8 は並列 GEMV の対象外（ADR 0098 決定 2）。
      assertEquals(planParallelKeys(graph, 9), { keys: { [NOT_PARALLEL]: 344 }, packed: 0 });
    });
  },
);
