// 2 段境界（ADR 0070 決定 5 / graph-first）の prepare 相 — グラフ shard 1 本で admission が
// 完結すること。GPU も重み shard も要らない層だけをここで見る（実 GPU の A/B 門は
// gpu_prepared_model_test.ts）。
//
// 検出したいのは 2 つ:
// ①見積りが「重み shard を 1 本も持たないまま」全量面と同じ数を出すこと（= 重み DL 前に
//   必要側が分かるという 2 段境界の存在理由そのもの）
// ②実行できないモデルは prepareModel の時点で落ちること — 使うグラフ shard は**重みテンソルを
//   1 本も含まない**ので、落ちた時点で重み側に触れていないことが構成から言える。

import { assert, assertEquals, assertThrows } from "@std/assert";
import { ContainerError, openModel } from "../src/format/container.ts";
import { IrError } from "../src/format/ir.ts";
import { SafetensorsError } from "../src/format/safetensors.ts";
import { OpContractError } from "../src/ops.ts";
import { estimateSessionMemory } from "../src/runtime/estimate.ts";
import { prepareModel } from "../src/runtime/executor.ts";
import { baseGraph, buildSafetensors, type GraphJson } from "./helpers/format.ts";
import { graphModelBuffer } from "./helpers/graph.ts";
import { buildFixture } from "./helpers/shard-fixture.ts";

const GRAPH_SHARD_ID = "fixture-v1.0/model-00001.safetensors";

const graphShard = (buffer: ArrayBuffer) => ({
  id: GRAPH_SHARD_ID,
  bytes: new Uint8Array(buffer),
});

/** 重みテンソルを 1 本も持たないグラフ shard（karume_ir だけ）。 */
const graphOnlyShard = (graph: GraphJson) => graphShard(graphModelBuffer(graph));

Deno.test("prepareModel の見積りは重み shard 抜きで全量面と一致する", () => {
  const fixture = buildFixture();
  // 既定の 3 分割の先頭 = karume_ir + bias 群（w1 / w2 / w3 は後続の重み shard 側）
  const prepared = prepareModel(graphShard(fixture.shards()[0]));
  const estimate = prepared.estimate();

  assertEquals(estimate, estimateSessionMemory(openModel(fixture.fullBuffer())));
  // 恒真化の防波堤: 4 格納混在の fixture なので圧縮常駐と非圧縮常駐がどちらも 0 でない
  // （全欄 0 どうしの一致で通ってしまう形を塞ぐ）。
  assert(estimate.compressedWeightBytes > 0, "圧縮常駐が 0（fixture が壊れている）");
  assert(estimate.uncompressedWeightBytes > 0, "非圧縮常駐が 0（fixture が壊れている）");
  assert(estimate.totalBytes > estimate.compressedWeightBytes, "合計が重みだけになっている");
  // 同じ PreparedModel から何度呼んでも同じ（prepare 相は消費されない）
  assertEquals(prepared.estimate(), estimate);
});

Deno.test("prepareModel は capability 不足・契約違反をグラフ shard の時点で落とす", () => {
  const unsupportedStorage = baseGraph();
  // bf16 は IR の語彙にはあるが実行経路が無い（ADR 0069 の隣 — capability 不足で列挙）
  unsupportedStorage.initializers.w.storage = { dtype: "bf16" };
  assertThrows(
    () => prepareModel(graphOnlyShard(unsupportedStorage)),
    ContainerError,
    "bf16",
  );

  const badArity = baseGraph();
  // add に 3 本目の入力（capability 表は slot dtype しか見ないのでアリティは契約検査の担当）
  badArity.nodes[1].ins = ["h", "b", "b"];
  assertThrows(() => prepareModel(graphOnlyShard(badArity)), OpContractError);
});

Deno.test("prepareModel の失敗は資産名（shard [0] 'id'）を名乗る", () => {
  // safetensors ですらないバイト列（パーサ門のクラスは保つ — 包み直すと呼び手の分岐が壊れる）
  const broken = assertThrows(
    () => prepareModel(graphShard(new Uint8Array(64).buffer)),
    SafetensorsError,
  );
  assert(broken.message.includes(GRAPH_SHARD_ID), broken.message);
  assert(broken.message.includes("shard [0]"), broken.message);

  // safetensors ではあるが karume_ir が無い（= 先頭に重み shard を置いた取り違え）
  const notGraph = assertThrows(
    () => prepareModel(graphShard(buildSafetensors([], undefined))),
    ContainerError,
    "karume_ir",
  );
  assert(notGraph.message.includes(GRAPH_SHARD_ID), notGraph.message);

  // karume_ir はあるが IR として壊れている
  const brokenIr = assertThrows(
    () => prepareModel(graphShard(buildSafetensors([], { karume_ir: "{" }))),
    IrError,
  );
  assert(brokenIr.message.includes(GRAPH_SHARD_ID), brokenIr.message);
});

Deno.test("prepareModel は bytes が buffer 全体を占めない view を受けない", () => {
  const loose = new Uint8Array(new ArrayBuffer(16), 4, 8) as Uint8Array<ArrayBuffer>;
  assertThrows(
    () => prepareModel({ id: GRAPH_SHARD_ID, bytes: loose }),
    Error,
    "buffer 全体",
  );
});
