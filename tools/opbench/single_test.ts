// opbench single のテスト: 加重行 → 1 ノード × 反復のグラフ組み立て（GPU 不要）と、実 GPU で
// K-11 の形（M=1 の i4 linear）が GEMV 族のキーで測れること。

import { assert, assertEquals, assertThrows } from "@std/assert";
import { acquireGpu, openModel } from "../../packages/runtime/mod.ts";
import type { CensusSummary, WeightRow } from "./census.ts";
import {
  buildCaseModel,
  MAX_OUTPUT_BYTES,
  measureCase,
  selectCases,
  sessionOptionsOf,
} from "./single.ts";

/** K-11 の常設門（gpu_linear_gemv_test.ts）と同じ「端がどこにも無い」形: k128 n64 g32・M=1。 */
const GEMV_ROW: WeightRow = {
  component: "unit",
  op: "linear",
  in_shapes: [[1, 128], [64, 128], [64]],
  out_shapes: [[1, 64]],
  in_dtypes: ["f32", "f32", "f32"],
  out_dtypes: ["f32"],
  attrs: {},
  storage: [null, { dtype: "i4", group_size: 32 }, { dtype: "f32" }],
  storage_signature: "f32+i4g32",
  fused_by: null,
  aliases_input: false,
  count: 3,
  out_elements: 64,
};

const row = (overrides: Partial<WeightRow>): WeightRow => ({ ...GEMV_ROW, ...overrides });

const summaryOf = (weights: readonly WeightRow[], scenario = "unit"): CensusSummary => ({
  generated_at: "2026-09-04T00:00:00.000Z",
  source: "synthetic",
  family: "unit",
  model: "unit",
  quant: "i4",
  session: {},
  scenarios: [{
    scenario,
    bindings: {},
    binding_source: "default",
    provenance: "synthetic",
    unused_bindings: [],
    node_count: weights.length,
    components: { unit: weights.length },
    by_op: {},
    by_storage: {},
    by_fusion: { absorbed: {}, hits: {}, plain: weights.length, aliased: 0 },
    weights,
  }],
});

Deno.test("buildCaseModel: 加重行 1 本から反復ぶんのノードを持つ配布形が組める（openModel が読む）", () => {
  const model = buildCaseModel(GEMV_ROW, 4);
  assertEquals(model.reps, 4);
  const { graph, file } = openModel(model.bytes);
  assertEquals(graph.nodes.length, 4);
  assertEquals(graph.outputs.length, 4);
  // 全ノードが同じ入力を読む（活性 x0 と初期化子 w1 / w2）。
  for (const node of graph.nodes) assertEquals([...node.ins], ["x0", "w1", "w2"]);
  assertEquals(Object.keys(model.inputs), ["x0"]);
  const x0 = model.inputs["x0"];
  assert("data" in x0, "x0 はホスト配列");
  assertEquals(x0.shape, [1, 128]);
  // i4 の scale は rank 2 の group 形 [rows, 行長 / group]、i8 なら keepdim 形になる（container の門が検査）。
  assertEquals(file.tensors.get("m.s1")?.shape, [64, 4]);
  assertEquals(file.tensors.get("m.w1")?.dtype, "I4");
  assertEquals(file.tensors.get("m.w2")?.dtype, "F32");
});

Deno.test("buildCaseModel: i8 の scale は keepdim broadcast 形（先頭次元がチャネル）", () => {
  const model = buildCaseModel(
    row({
      in_shapes: [[1, 8, 16], [4, 8, 3]],
      out_shapes: [[1, 4, 14]],
      in_dtypes: ["f32", "f32"],
      storage: [null, { dtype: "i8" }],
      storage_signature: "i8",
      op: "conv1d",
      attrs: { stride: 1, padding: 0, dilation: 1, groups: 1 },
    }),
    1,
  );
  const { file } = openModel(model.bytes);
  assertEquals(file.tensors.get("m.s1")?.shape, [4, 1, 1]);
  assertEquals(file.tensors.get("m.w1")?.dtype, "I8");
});

Deno.test("buildCaseModel: 反復は出力 readback の上限で抑えられる", () => {
  const big = row({
    in_shapes: [[4096, 4096]],
    out_shapes: [[4096, 4096]],
    in_dtypes: ["f32"],
    storage: [null],
    op: "relu",
    storage_signature: "none",
  });
  const model = buildCaseModel(big, 1024);
  assertEquals(model.reps, Math.floor(MAX_OUTPUT_BYTES / (4096 * 4096 * 4)));
});

Deno.test("selectCases: 融合済み・別名化・f32 でない活性・符号化器の無い格納は理由つきで除外する", () => {
  const summary = summaryOf([
    GEMV_ROW,
    row({ fused_by: "rope" }),
    row({ aliases_input: true }),
    row({ in_dtypes: ["i32", "f32", "f32"] }),
    row({ storage: [null, { dtype: "bf16" }, { dtype: "f32" }] }),
    row({
      op: "state_append",
      in_shapes: [[1, 1, 1, 256]],
      out_shapes: [],
      in_dtypes: ["f32"],
      out_dtypes: [],
      storage: [null],
      storage_signature: "none",
    }),
    row({ op: "attention", attrs: { window: 512 } }),
  ]);
  const selection = selectCases(summary);
  assertEquals(selection.cases.length, 1);
  assertEquals(selection.excluded, {
    "fused (rope)": 1,
    "aliases_input (0 dispatch)": 1,
    "activation dtype i32 (f32 only)": 1,
    "storage dtype bf16 (no encoder)": 1,
    "state slot op (needs GenerationContext)": 2,
  });
});

Deno.test("selectCases: 加重（count × 出力要素）の降順で並び、--limit と --op が効く", () => {
  const summary = summaryOf([
    row({ op: "relu", count: 1, out_elements: 64 }),
    row({ op: "gelu", count: 10, out_elements: 64 }),
    row({ op: "silu", count: 5, out_elements: 64 }),
  ]);
  assertEquals(selectCases(summary).cases.map((c) => c.row.op), ["gelu", "silu", "relu"]);
  assertEquals(selectCases(summary, { limit: 2 }).cases.map((c) => c.row.op), ["gelu", "silu"]);
  assertEquals(selectCases(summary, { ops: new Set(["relu"]) }).cases.map((c) => c.row.op), [
    "relu",
  ]);
});

Deno.test("sessionOptionsOf: 宣言を写し、上書きが勝ち、未知のノブと不正な値は落ちる", () => {
  assertEquals(sessionOptionsOf(null), {});
  assertEquals(sessionOptionsOf({}), {});
  assertEquals(sessionOptionsOf({ linearCompute: "a8", attentionScoreStorage: "f16" }), {
    linearCompute: "a8",
    attentionScoreStorage: "f16",
  });
  assertEquals(sessionOptionsOf({ linearCompute: "a8" }, { linearCompute: "f32" }), {
    linearCompute: "f32",
  });
  assertThrows(() => sessionOptionsOf({ linearCompute: "i8a8" }), Error, "linearCompute");
  assertThrows(() => sessionOptionsOf({ submitPolicy: "x" }), Error, "未知のノブ");
});

const adapter = navigator.gpu === undefined ? null : await navigator.gpu.requestAdapter();
const timestampQuery = adapter !== null && adapter.features.has("timestamp-query");

Deno.test({
  name: "実 GPU: M=1 の i4 linear を timing モードで測ると GEMV 族のキーが立ち ns が正になる",
  ignore: !timestampQuery,
  fn: async () => {
    const gpu = await acquireGpu({ gpuTiming: true });
    try {
      const record = await measureCase({ scenario: "unit", row: GEMV_ROW }, {
        gpu,
        session: {},
        mode: "timing",
        rounds: 2,
      });
      assertEquals(record.mode, "timing");
      assert(record.reps >= 1);
      assert((record.ns_per_node_min ?? 0) > 0);
      assertEquals(record.dispatches_per_node, 1);
      assert(record.keys.some((key) => key.startsWith("linear_gemv:")), record.keys.join(" "));
      assertEquals(record.weighted_ms, (record.ns_per_node_min ?? 0) * 3 / 1e6);
      assertEquals(record.wall_ms_per_rep_min, null);
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "実 GPU: wall モードは計測無効の device でフェンス 1 本の区間を測り、正の ms/rep を返す",
  ignore: adapter === null,
  fn: async () => {
    const gpu = await acquireGpu({ gpuTiming: false });
    try {
      const record = await measureCase({ scenario: "unit", row: GEMV_ROW }, {
        gpu,
        session: {},
        mode: "wall",
        rounds: 2,
      });
      assertEquals(record.mode, "wall");
      assert((record.wall_ms_per_rep_min ?? 0) > 0);
      assertEquals(record.ns_per_node_min, null);
      assertEquals(record.keys, []);
    } finally {
      gpu.destroy();
    }
  },
});
