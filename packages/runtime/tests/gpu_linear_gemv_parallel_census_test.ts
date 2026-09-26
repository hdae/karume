import { assert, assertEquals } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import type { CodecName } from "../src/format/container/codecs.ts";
import type { IrGraph } from "../src/format/ir.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import {
  linearGemvPackedEligible,
  linearGemvParallelEligible,
  linearGemvParallelKey,
} from "../src/kernels/linear-gemv.ts";
import { createSessionFromContainer } from "../src/runtime/executor.ts";
import { bindSymbols, planGraph } from "../src/runtime/plan.ts";
import { planWeightResidency } from "../src/runtime/weight-residency.ts";
import {
  type ContainerManifest,
  containerPart0,
  readContainerGraph,
} from "./helpers/container-graph.ts";
import type { TensorInput } from "./helpers/container-write.ts";
import {
  type DeclarationJson,
  f32Bytes,
  GRAPH_NAME,
  openModelBytes,
} from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { countDispatches, requireCensus, SEAT_SIGNATURES } from "./helpers/pipeline-census.ts";

/** 格納の呼び名 → codec 台帳の登録名（i2 は per-channel の `int2-off`）。 */
const CODEC: Readonly<Record<"i2" | "i4" | "i8", CodecName>> = {
  i2: "int2-off",
  i4: "int4-sym-g",
  i8: "int8-sym",
};

describe({
  name: "GEMV並列加算の適用箇所（実GPU）",
  ignore: !GPU_AVAILABLE,
  fn: () => {
    it("明示指定の対象だけに適用し、M1/4/8で全出力列のu32一致を保つ", async () => {
      const gpu = await acquireGpu();
      try {
        for (
          const shape of [
            { storage: "i2", bits: 2, n: 12288, k: 1536, group: undefined },
            { storage: "i4", bits: 4, n: 256, k: 1536, group: 32 },
            { storage: "i4", bits: 4, n: 256, k: 1536, group: 512 },
            { storage: "i4", bits: 4, n: 1536, k: 2048, group: 2048 },
            { storage: "i4", bits: 4, n: 1536, k: 4096, group: 4096 },
            { storage: "i8", bits: 8, n: 256, k: 1536, group: undefined },
            // E4B の行（2026-09-26・lanes は E2B から写した未実測）の代表 — l32 / l4 × g32 / g512 /
            // g2048 / i8。実寸の n で全出力列を見る。
            { storage: "i4", bits: 4, n: 512, k: 2560, group: 32 },
            { storage: "i4", bits: 4, n: 10240, k: 2560, group: 512 },
            { storage: "i4", bits: 4, n: 2560, k: 10240, group: 2048 },
            { storage: "i8", bits: 8, n: 2560, k: 256, group: undefined },
          ] as const
        ) {
          const { storage, n, k, bits, group } = shape;
          const w = Uint8Array.from(
            { length: n * k * bits / 8 },
            (_, i) => (Math.imul(i + 1, 0x9e3779b9) >>> 24),
          );
          // per-channel（i2 / i8）は groups = 1、group 量子化（i4）は 行長 / group。
          const groups = group === undefined ? 1 : k / group;
          const tensors: readonly TensorInput[] = [
            {
              graph: GRAPH_NAME,
              initializer: "w",
              bytes: w,
              encoding: {
                codec: CODEC[storage],
                groupSize: group ?? k,
                scale: {
                  dtype: "f32",
                  bytes: f32Bytes(
                    Array.from({ length: n * groups }, (_, i) => (i % 17 + 1) * 0.00017),
                  ),
                },
              },
            },
            {
              graph: GRAPH_NAME,
              initializer: "b",
              bytes: f32Bytes(Array.from({ length: n }, (_, i) => (i % 7 - 3) * 0.11)),
              encoding: { codec: "f32" },
            },
          ];
          let expected: Uint32Array<ArrayBuffer> | undefined;
          for (const m of [1, 4, 8, 9]) {
            for (const parallel of m === 1 ? [false, true] : [true]) {
              const graph: DeclarationJson = {
                format: "karume-ir",
                version: 2,
                requires: { ops: ["linear"] },
                symbols: [],
                inputs: [{ name: "x", dtype: "f32", shape: [m, k] }],
                outputs: ["y"],
                initializers: { w: {}, b: {} },
                values: {
                  w: { dtype: "f32", shape: [n, k] },
                  b: { dtype: "f32", shape: [n] },
                  y: { dtype: "f32", shape: [m, n] },
                },
                nodes: [{ op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} }],
              };
              const session = await createSessionFromContainer(
                gpu,
                await openModelBytes(graph, tensors),
                GRAPH_NAME,
                parallel ? { linearGemvReduce: "parallel" } : {},
              );
              try {
                const input = Float32Array.from(
                  { length: m * k },
                  (_, i) => Math.sin(i * 0.037) * 0.75,
                );
                const output =
                  (await session.run({ x: { dtype: "f32", shape: [m, k], data: input } })).y;
                assertEquals(output.shape, [m, n]);
                for (const value of output.data) assert(Number.isFinite(value));
                const where = `${storage} M${m}`;
                const census = requireCensus(session.diagnostics().lastRunPipelines, where);
                assertEquals(
                  countDispatches(census, SEAT_SIGNATURES.linearGemvReduce.parallel) > 0,
                  parallel && m <= 8,
                  `${where} ${census.map((row) => row.key)}`,
                );
                if (parallel && m <= 8) {
                  const first = new Uint32Array(output.data.buffer, output.data.byteOffset, n)
                    .slice();
                  expected ??= first;
                  assertEquals(first, expected);
                }
              } finally {
                await session.dispose();
              }
            }
          }
        }
      } finally {
        gpu.destroy();
      }
    });
  },
});

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
const QAT_ROOT = new URL("../../../models/karume-gemma4-qat/", import.meta.url);
const QAT_MANIFEST = new URL("karume.json", QAT_ROOT);

/** 資産の有無（NotFound 以外は伝播 — マウント異常を SKIP に化かさない）。 */
const exists = async (url: URL): Promise<boolean> => {
  try {
    await Deno.stat(url);
    return true;
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return false;
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
    const storage = seat?.seat === "i2" || seat?.seat === "i4" || seat?.seat === "i8"
      ? seat.seat
      : undefined;
    const group = seat?.seat === "i4" ? seat.groupSize : undefined;
    const lanes = storage === undefined
      ? undefined
      : linearGemvParallelEligible(storage, m, weight[0], weight[1], group);
    const key = storage === undefined || lanes === undefined
      ? NOT_PARALLEL
      : linearGemvParallelKey(storage, group, lanes);
    keys.set(key, (keys.get(key) ?? 0) + 1);
    if (
      storage !== undefined &&
      linearGemvPackedEligible(storage, m, weight[0], weight[1], group) !== undefined
    ) packed += 1;
  }
  return { keys: Object.fromEntries([...keys].toSorted(([a], [b]) => a < b ? -1 : 1)), packed };
};

describe({
  name: "QAT E4B の製品グラフでの並列 GEMV の適用（計画・GPU 不要）",
  ignore: !QAT_AVAILABLE,
  fn: () => {
    it("i2 の lm_head と f32 の per_layer_model_projection を除く全 linear が E4B の行に載る", async () => {
      const manifest: ContainerManifest = JSON.parse(await Deno.readTextFile(QAT_MANIFEST));
      const head = containerPart0(manifest, "e4b", "model", "i4");
      const graph = await readContainerGraph(new URL(head.path, QAT_ROOT), "model");
      // 本数の内訳（42 層・full 7 層）: i4 g512 l4 = q 35 + q(full) 7 + gate/up 84 /
      // i4 g512 l32 = k/v 40 + k/v(full) 8 / i4 g2048 l32 = o 35 + down 42 / i4 g4096 l32 = o(full) 7 /
      // i8 l32 = per_layer_input_gate 42 / i8 l4 = per_layer_projection 42 / 対象外 2。
      const expected = Object.fromEntries(
        ([
          [linearGemvParallelKey("i4", 512, 4), 126],
          [linearGemvParallelKey("i4", 512, 32), 48],
          [linearGemvParallelKey("i4", 2048, 32), 77],
          [linearGemvParallelKey("i4", 4096, 32), 7],
          [linearGemvParallelKey("i8", undefined, 32), 42],
          [linearGemvParallelKey("i8", undefined, 4), 42],
          [NOT_PARALLEL, 2],
        ] as const).toSorted(([a], [b]) => a < b ? -1 : 1),
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
});
