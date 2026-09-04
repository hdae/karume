// opbench graph のテスト（GPU 不要）: パイプラインキー → op の写像と、census との突合表。

import { assertEquals, assertThrows } from "@std/assert";
import type { CensusSummary, WeightRow } from "./census.ts";
import type { SingleSummary } from "./single.ts";
import { compareWithCensus, opOfKey, type RunRecord } from "./graph.ts";

Deno.test("opOfKey: 先頭語を op に写す（変種名は表で・表に無ければ先頭語そのまま）", () => {
  assertEquals(opOfKey("linear_gemv:v1:f32:c32u4:wi4g32"), "linear");
  assertEquals(opOfKey("linear:v4:i8a8:tile128x64r8x8w8x16k16v4:dp4a"), "linear");
  assertEquals(opOfKey("quantize_rows:v1:f32>i8:pertoken:wg256"), "aux");
  assertEquals(opOfKey("attention_state_qk:v2"), "attention");
  assertEquals(opOfKey("rms_norm:v1:f32"), "rms_norm");
  assertEquals(opOfKey("ew:v3:gelu_tanh:f32>f32:r3:wg256"), "gelu_tanh");
  assertEquals(opOfKey("strided_write:v1:f32:r4:wg256"), "strided");
  assertEquals(opOfKey("rope:v1:half:f32:wg256"), "fused");
  assertEquals(opOfKey("silu:v1:x-sigmoid:f32:wg256"), "fused");
  assertEquals(opOfKey("something_new:v1"), "something_new");
});

const weight = (op: string, count: number, extra: Partial<WeightRow> = {}): WeightRow => ({
  component: "model",
  op,
  in_shapes: [[1, 8]],
  out_shapes: [[1, 8]],
  in_dtypes: ["f32"],
  out_dtypes: ["f32"],
  attrs: {},
  storage: [null],
  storage_signature: "none",
  fused_by: null,
  aliases_input: false,
  count,
  out_elements: 8,
  ...extra,
});

const census: CensusSummary = {
  generated_at: "2026-09-04T00:00:00.000Z",
  source: "synthetic",
  family: "unit",
  model: "unit",
  quant: "i4",
  session: {},
  scenarios: [{
    scenario: "decode",
    bindings: {},
    binding_source: "default",
    provenance: "synthetic",
    unused_bindings: [],
    node_count: 0,
    components: {},
    by_op: {},
    by_storage: {},
    by_fusion: { absorbed: {}, hits: {}, plain: 0, aliased: 0 },
    weights: [
      weight("linear", 10),
      weight("linear", 2, { storage_signature: "f32+i8" }),
      weight("rms_norm", 4),
      weight("rope", 3, { fused_by: "rope" }), // 融合で消えた行は素のノードに数えない
      weight("reshape", 5, { aliases_input: true }), // 0 dispatch も数えない
      weight("linear", 99, { component: "other" }), // run 群が触らないコンポーネントは数えない
    ],
  }],
};

const run = (index: number, label: string, entries: RunRecord["entries"]): RunRecord => ({
  index,
  component: "model",
  label,
  dispatch_count: entries.reduce((total, entry) => total + entry.dispatch_count, 0),
  total_ns: entries.reduce((total, entry) => total + entry.ns, 0),
  entries,
  fusions: null,
  clamped_negative_samples: 0,
});

Deno.test("compareWithCensus: op 別に census の素ノード本数と実測 dispatch を並べ、run 群は平均する", () => {
  const records = [
    run(0, "decode-1", [
      { key: "linear_gemv:v1", ns: 8_000_000, dispatch_count: 10 },
      { key: "linear:v4:i8a8", ns: 2_000_000, dispatch_count: 2 },
      { key: "quantize_rows:v1", ns: 500_000, dispatch_count: 2 },
      { key: "rms_norm:v1", ns: 1_000_000, dispatch_count: 4 },
      { key: "rope:v1:half", ns: 300_000, dispatch_count: 3 },
      { key: "mystery:v1", ns: 1_000, dispatch_count: 1 },
    ]),
    run(1, "decode-2", [
      { key: "linear_gemv:v1", ns: 12_000_000, dispatch_count: 10 },
      { key: "linear:v4:i8a8", ns: 2_000_000, dispatch_count: 2 },
      { key: "quantize_rows:v1", ns: 500_000, dispatch_count: 2 },
      { key: "rms_norm:v1", ns: 1_000_000, dispatch_count: 4 },
      { key: "rope:v1:half", ns: 300_000, dispatch_count: 3 },
      { key: "mystery:v1", ns: 1_000, dispatch_count: 1 },
    ]),
  ];
  const comparison = compareWithCensus(records, census, "decode");
  assertEquals(comparison.runs, 2);
  assertEquals(comparison.census_plain_nodes, 16);
  assertEquals(comparison.components, ["model"]);
  const byOp = Object.fromEntries(comparison.rows.map((row) => [row.op, row]));
  assertEquals(byOp.linear.census_nodes, 12);
  assertEquals(byOp.linear.measured_dispatches, 12);
  assertEquals(byOp.linear.measured_ms, 12); // (10 + 14) / 2
  assertEquals(byOp.rms_norm.census_nodes, 4);
  assertEquals(byOp.aux.census_nodes, null);
  assertEquals(byOp.aux.measured_dispatches, 2);
  // 融合ルールのキーは fused バケツ、表にも census にも無い語だけが unmapped に残る（黙って捨てない）。
  assertEquals(byOp.fused.measured_dispatches, 3);
  assertEquals(comparison.unmapped_keys, ["mystery:v1"]);
  assertEquals(comparison.rows[0].op, "linear"); // ms 降順
});

Deno.test("compareWithCensus: single の加重合計を op 別に足して single / graph の比を出す", () => {
  const single: SingleSummary = {
    generated_at: "2026-09-04T00:00:00.000Z",
    census: "synthetic",
    family: "unit",
    model: "unit",
    quant: "i4",
    session: {},
    mode: "timing",
    rig: {
      vendor: "",
      architecture: "",
      device: "",
      description: "",
      deno: "",
      target_pass_ms: 80,
      rounds: 5,
    },
    measured: 2,
    excluded: {},
    failed: [],
    weighted_ms_by_op_storage: { "linear/f32+i4g32": 6, "linear/f32+i8": 3, "rms_norm/f32": 2 },
  };
  const records = [run(0, "decode-1", [
    { key: "linear_gemv:v1", ns: 10_000_000, dispatch_count: 12 },
    { key: "rms_norm:v1", ns: 1_000_000, dispatch_count: 4 },
  ])];
  const byOp = Object.fromEntries(
    compareWithCensus(records, census, "decode", single).rows.map((row) => [row.op, row]),
  );
  assertEquals(byOp.linear.single_weighted_ms, 9);
  assertEquals(byOp.linear.single_over_graph, 0.9);
  assertEquals(byOp.rms_norm.single_over_graph, 2);
});

Deno.test("compareWithCensus: 無いシナリオ名は既知の一覧つきで落ちる", () => {
  assertThrows(() => compareWithCensus([], census, "prefill"), Error, "既知: decode");
});
