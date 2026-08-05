// GPU 時間診断（timestamp-query・op 別内訳 — ADR 0021）の検証。
//
// 「計測が結果を変えない」ことと「内訳が実行そのものと一致する」ことを、実行済みの他の統計
// （SubmitStats.dispatchCount / pipelineCount）との**相互検算**で固定する。内訳だけを単体で
// 見ると、集計が空でも 0 件でも緑になってしまう（計測が黙って死んでいる形が素通りする）。

import { assert, assertEquals, assertGreater, assertThrows } from "@std/assert";
import { elementwiseKey } from "../src/codegen/elementwise.ts";
import { acquireGpu, GpuFeatureError, planTimestampFeature } from "../src/gpu/device.ts";
import type { GpuTimingStats } from "../src/gpu/submit.ts";
import { matmulKey } from "../src/kernels/matmul.ts";
import { openModel } from "../src/format/container.ts";
import { createSession, type Tensor } from "../src/runtime/executor.ts";
import { f32Bytes, type GraphJson } from "./helpers/format.ts";
import { fill, graphModelBuffer } from "./helpers/graph.ts";
import { GPU_AVAILABLE, TIMESTAMP_QUERY_AVAILABLE } from "./helpers/gpu.ts";

/**
 * y = relu(x·w + b) + b。**add を 2 回**通すのは、集計がキー単位でまとまること
 * （dispatch 4 件 → 内訳 3 キー）を、件数の一致だけで確かめられるようにするため。
 */
const GRAPH: GraphJson = {
  format: "karume-ir",
  version: 1,
  requires: { ops: ["matmul", "add", "relu"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [6, 4] }],
  outputs: ["y"],
  initializers: {
    w: { tensor: "enc.w", storage: { dtype: "f32" } },
    b: { tensor: "enc.b", storage: { dtype: "f32" } },
  },
  values: {
    w: { dtype: "f32", shape: [4, 3] },
    b: { dtype: "f32", shape: [3] },
    h: { dtype: "f32", shape: [6, 3] },
    g: { dtype: "f32", shape: [6, 3] },
    r: { dtype: "f32", shape: [6, 3] },
    y: { dtype: "f32", shape: [6, 3] },
  },
  nodes: [
    { op: "matmul", ins: ["x", "w"], outs: ["h"], attrs: {} },
    { op: "add", ins: ["h", "b"], outs: ["g"], attrs: {} },
    { op: "relu", ins: ["g"], outs: ["r"], attrs: {} },
    { op: "add", ins: ["r", "b"], outs: ["y"], attrs: {} },
  ],
};

const W = [0.5, -1, 0.25, 2, 0.125, -0.5, -3, 1.5, 0.75, 1, -0.25, 0.5];
const B = [1, -2, 0.5];

/** グラフが実際に踏むパイプラインキー（executor と同じ生成関数から引く）。 */
const EXPECTED_KEYS: ReadonlySet<string> = new Set([
  // x[6,4] × w[4,3]: n = 3 は 4 の倍数でないのでスカラ変種を踏む
  matmulKey(false),
  elementwiseKey({ op: "add", rank: 2, dtype: "f32" }),
  elementwiseKey({ op: "relu", rank: 2, dtype: "f32" }),
]);

/** 出力の**バイト列**（計測モードが結果を変えないことは値ではなくビットで見る）。 */
const outputBytes = (tensor: Tensor): Uint8Array =>
  new Uint8Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.byteLength).slice();

/** グラフを 1 回だけ走らせ、出力バイト列と診断を返す。 */
const runOnce = async (
  gpuTiming: boolean | undefined,
): Promise<{
  readonly timingEnabled: boolean;
  readonly bytes: Uint8Array;
  readonly features: ReadonlySet<string>;
  readonly dispatchCount: number;
  readonly pipelineCount: number;
  readonly timing: GpuTimingStats | undefined;
}> => {
  const gpu = await acquireGpu(gpuTiming === undefined ? {} : { gpuTiming });
  const model = openModel(graphModelBuffer(GRAPH, [
    { name: "enc.w", dtype: "F32", shape: [4, 3], data: f32Bytes(W) },
    { name: "enc.b", dtype: "F32", shape: [3], data: f32Bytes(B) },
  ]));
  const session = await createSession(gpu, model);
  try {
    const outputs = await session.run({ x: fill([6, 4], (i) => ((i % 11) - 5) * 0.5) });
    const diagnostics = session.diagnostics();
    return {
      timingEnabled: gpu.gpuTimingEnabled,
      bytes: outputBytes(outputs["y"]),
      features: new Set(gpu.device.features),
      dispatchCount: diagnostics.submit.dispatchCount,
      pipelineCount: diagnostics.pipelineCount,
      timing: diagnostics.lastRunTiming,
    };
  } finally {
    await session.dispose();
    gpu.destroy();
  }
};

// 三値のうち「true × feature 不在」だけは実機で作れない（アダプタの feature は消せない）ため、
// 判定を切り出した純関数で全 6 通りを固定する。
Deno.test("planTimestampFeature が gpuTiming の三値を判定し、必須指定の不足は fail loudly", () => {
  const present = new Set(["shader-f16", "timestamp-query"]);
  const absent = new Set(["shader-f16"]);

  // MUST: `undefined` は「要求しない」（shaderF16 と同じ規律）。ここを「持っていれば要求」に
  // 戻すと、計測の代価（1 dispatch = 1 pass）がアダプタ次第で既定経路に乗る。
  assertEquals(planTimestampFeature(present, undefined), false, "既定は持っていても要求しない");
  assertEquals(planTimestampFeature(absent, undefined), false);
  assertEquals(planTimestampFeature(present, true), true);
  assertEquals(planTimestampFeature(present, false), false, "false は持っていても要求しない");
  assertEquals(planTimestampFeature(absent, false), false);

  const error = assertThrows(
    () => planTimestampFeature(absent, true),
    GpuFeatureError,
    "timestamp-query",
  );
  // 不足内容が読めること（環境で何が使えるかまで書く）。
  assert(error.message.includes("shader-f16"), error.message);
});

Deno.test({
  name: "gpuTiming: true の内訳が dispatch 数・キー集合と相互に一致する（実 GPU）",
  ignore: !GPU_AVAILABLE || !TIMESTAMP_QUERY_AVAILABLE,
  fn: async () => {
    const result = await runOnce(true);
    assert(result.timingEnabled, "true を要求して有効にならないのは device 側の矛盾");
    const timing = result.timing;
    assert(timing !== undefined, "計測が有効なら内訳は必ず出る");

    assertEquals(result.dispatchCount, 4, "matmul + add + relu + add");
    assertEquals(timing.dispatchCount, result.dispatchCount, "内訳の dispatch 総数の相互検算");
    assertEquals(
      new Set(timing.entries.map((entry) => entry.key)),
      EXPECTED_KEYS,
      "キー集合は実行された pipeline と一致する",
    );
    assertEquals(timing.entries.length, result.pipelineCount, "生成した pipeline は全て走った");

    for (const entry of timing.entries) {
      assert(entry.ns >= 0, `${entry.key}: ns が負（${entry.ns}）`);
      assertGreater(entry.workgroupCount, 0, `${entry.key}: workgroup 数`);
    }
    assertEquals(
      timing.entries.find((entry) => entry.key.includes(":add:"))?.dispatchCount,
      2,
      "同じキーの dispatch はまとまる",
    );
    assertEquals(
      timing.totalNs,
      timing.entries.reduce((total, entry) => total + entry.ns, 0),
      "合計は内訳の総和",
    );
    assertEquals(
      [...timing.entries].sort((a, b) => b.ns - a.ns).map((entry) => entry.key),
      timing.entries.map((entry) => entry.key),
      "表は ns の降順",
    );
    // 0 に丸めた負の差分（ドライバの非単調）は握り潰さず件数で観測する。
    assertEquals(timing.clampedNegativeSamples, 0, "非単調な timestamp は観測されない");
    assertGreater(timing.totalNs, 0, "実 GPU の計測が全て 0 になるのは計測が死んでいる形");
  },
});

// 既定（undefined）と明示 false が**同じ**であることの門。ここが割れると、計測の代価が
// アダプタ次第で既定経路に乗る（1 dispatch = 1 pass — ADR 0021 / 案 3）。
Deno.test({
  name: "gpuTiming の既定と false は feature を要求せず、内訳も出さない（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    for (const requested of [undefined, false] as const) {
      const result = await runOnce(requested);
      const where = `gpuTiming: ${String(requested)}`;
      assertEquals(result.features.has("timestamp-query"), false, `${where}: feature を持たない`);
      assertEquals(result.timingEnabled, false, where);
      assertEquals(result.timing, undefined, where);
    }
  },
});

Deno.test({
  name: "計測モードは実行結果を変えない（同一グラフの出力がバイト一致・実 GPU）",
  ignore: !GPU_AVAILABLE || !TIMESTAMP_QUERY_AVAILABLE,
  fn: async () => {
    const timed = await runOnce(true);
    const plain = await runOnce(false);
    assert(timed.timingEnabled, "対照が成立していない（計測側が有効になっていない）");
    assertEquals(timed.bytes, plain.bytes, "1 dispatch = 1 pass に開いても出力は同一バイト");
    assertEquals(timed.dispatchCount, plain.dispatchCount, "dispatch 数も変わらない");
  },
});
