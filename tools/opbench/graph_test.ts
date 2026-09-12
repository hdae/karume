// opbench graph のテスト（GPU 不要）: パイプラインキー → op の写像と、census との突合表。

import { assert, assertEquals, assertThrows } from "@std/assert";
import type { IrodoriRunComponent } from "../../packages/models/mod.ts";
import type { CensusSummary, WeightRow } from "./census.ts";
import type { SingleSummary } from "./single.ts";
import {
  compareWithCensus,
  defaultRunsPrefix,
  DRIVE_FAMILIES,
  gemma4RunLabel,
  irodoriCensusComponent,
  opOfKey,
  type RunRecord,
} from "./graph.ts";

Deno.test("opOfKey: 先頭語を op に写す（変種名は表で・表に無ければ先頭語そのまま）", () => {
  assertEquals(opOfKey("linear_gemv:v1:f32:c32u4:wi4g32"), "linear");
  assertEquals(opOfKey("linear_gemv_parallel:wi4g512:l4"), "linear");
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

Deno.test("irodoriCensusComponent: 観測席のハイフン綴りを census のアンダースコア綴りへ写す", () => {
  // 8 名の全部を固定する（写し漏れが 1 つでもあると、その段だけ census と当たらない）。
  const components: readonly IrodoriRunComponent[] = [
    "backbone",
    "text-proj",
    "caption-proj",
    "speaker",
    "duration",
    "dit",
    "codec-encoder",
    "codec-decoder",
  ];
  assertEquals(components.map(irodoriCensusComponent), [
    "backbone",
    "text_proj",
    "caption_proj",
    "speaker",
    "duration",
    "dit",
    "codec_encoder",
    "codec_decoder",
  ]);
});

Deno.test("gemma4RunLabel: label は観測席の phase から作る（回数からではない）", () => {
  // 単一 chunk のターン: prefill-1 → decode-1 → decode-2。
  assertEquals(gemma4RunLabel({ kind: "prefill", chunk: 1, chunks: 1 }), "prefill-1");
  assertEquals(gemma4RunLabel({ kind: "decode", step: 1 }), "decode-1");
  assertEquals(gemma4RunLabel({ kind: "decode", step: 2 }), "decode-2");
  // 複数 chunk のターン: 2 通目も prefill（回数で決めると decode-1 に化ける — F-01）。
  assertEquals(gemma4RunLabel({ kind: "prefill", chunk: 2, chunks: 2 }), "prefill-2");
  // 投機のターン: 1 cycle = draft 1 本 + verify 1 本で、**同じ cycle 番号**を名乗る（label が
  // cycle でなく通し番号で振られると、draft と verify の対応が突合表から読めなくなる）。
  assertEquals(gemma4RunLabel({ kind: "draft", cycle: 1 }), "draft-1");
  assertEquals(gemma4RunLabel({ kind: "verify", cycle: 1, rows: 4, accepted: 3 }), "verify-1");
  assertEquals(gemma4RunLabel({ kind: "draft", cycle: 12 }), "draft-12");
  // `k' = 0` の cycle（予算末尾）は draft を採らず verify 1 行だけ — 行数や受理数は label に
  // 出さない（同じ形の run が別名になると census との突合が cycle ごとに割れる）。
  assertEquals(gemma4RunLabel({ kind: "verify", cycle: 12, rows: 1, accepted: 0 }), "verify-12");
  // 突合は接頭辞一致（main.ts の `--runs`）: gemma4 の既定接頭辞は decode 群だけを拾い、prefill 群は
  // chunk が何本でも `prefill` で全部拾える（複数 chunk の 2 通目が decode 側へ混ざらない）。
  const decodePrefix = defaultRunsPrefix("gemma4");
  assert(gemma4RunLabel({ kind: "decode", step: 7 }).startsWith(decodePrefix));
  assert(!gemma4RunLabel({ kind: "prefill", chunk: 2, chunks: 2 }).startsWith(decodePrefix));
  assert(gemma4RunLabel({ kind: "prefill", chunk: 3, chunks: 4 }).startsWith("prefill"));
  // 投機の run は decode 群に混ざらない（既定接頭辞は非投機の decode だけを拾う）。
  assert(!gemma4RunLabel({ kind: "draft", cycle: 1 }).startsWith(decodePrefix));
  assert(
    !gemma4RunLabel({ kind: "verify", cycle: 1, rows: 4, accepted: 0 }).startsWith(
      decodePrefix,
    ),
  );
});

Deno.test("defaultRunsPrefix: 家族ごとに突合する run の接頭辞が決まる", () => {
  assertEquals(DRIVE_FAMILIES.map(defaultRunsPrefix), [
    "decode",
    "transformer",
    "vision",
    "dit",
  ]);
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
