// 2 段境界（ADR 0070 決定 5 / graph-first）の prepare 相 — 開いた容器の**宣言だけ**で admission が
// 完結すること。GPU も重みの block も要らない層だけをここで見る（実 GPU の門は
// gpu_prepared_model_test.ts）。
//
// 検出したいのは 2 つ:
// ①見積りが「重みの block を 1 つも取らないまま」出ること（= 重み DL 前に必要側が分かるという
//   2 段境界の存在理由そのもの）。取得の回数を数えて言う。
// ②実行できないモデルは prepareContainer の時点で落ちること — 落ちた時点で重みの block に
//   1 つも触れていないことを、同じ取得回数で言う。

import { assert, assertEquals, assertThrows } from "@std/assert";
import { OpContractError } from "../src/ops.ts";
import { RuntimeSupportError } from "../src/ops/support.ts";
import { prepareContainer } from "../src/runtime/executor.ts";
import { buildFixture, countingContainer } from "./helpers/mixed-codec-fixture.ts";
import {
  baseDeclaration,
  baseTensors,
  GRAPH_NAME,
  memoryModel,
  openModelBytes,
} from "./helpers/model-fixture.ts";

Deno.test("prepareContainer の見積りは重みの block を 1 つも取らずに出る", async () => {
  const fixture = buildFixture();
  const counted = countingContainer(await openModelBytes(fixture.declaration, fixture.tensors));
  const prepared = prepareContainer(counted.container, GRAPH_NAME);
  const estimate = prepared.estimate();
  assertEquals(counted.reads(), 0, "見積りが重みの block を取っている");

  // 供給元が違っても同じ数（krm 経路 / メモリ内容器 — 見積りは宣言だけで決まる）
  assertEquals(
    estimate,
    prepareContainer(memoryModel(fixture.declaration, fixture.tensors), GRAPH_NAME).estimate(),
  );
  // 恒真化の防波堤: 4 codec 混在の fixture なので圧縮常駐と非圧縮常駐がどちらも 0 でない
  // （全欄 0 どうしの一致で通ってしまう形を塞ぐ）。
  const { weights } = estimate.resident;
  assert(weights.compressedBytes > 0, "圧縮常駐が 0（fixture が壊れている）");
  assert(weights.uncompressedBytes > 0, "非圧縮常駐が 0（fixture が壊れている）");
  assert(
    estimate.peakAccountedBytes > weights.totalBytes,
    "ピークが重み常駐だけになっている（シナリオ側が乗っていない）",
  );
  // 同じ PreparedModel から何度呼んでも同じ（prepare 相は消費されない）
  assertEquals(prepared.estimate(), estimate);
});

Deno.test("prepareContainer は capability 不足・契約違反を重みの block を取る前に落とす", async () => {
  // bf16 は容器の codec 台帳にはあるが実行経路が無い（ADR 0069 の隣 — capability 不足で列挙）
  const unsupportedCodec = countingContainer(
    await openModelBytes(baseDeclaration(), [
      {
        graph: GRAPH_NAME,
        initializer: "w",
        bytes: new Uint8Array(new ArrayBuffer(24)),
        encoding: { codec: "bf16" },
      },
      baseTensors()[1],
    ]),
  );
  assertThrows(
    () => prepareContainer(unsupportedCodec.container, GRAPH_NAME),
    RuntimeSupportError,
    "bf16",
  );
  assertEquals(unsupportedCodec.reads(), 0, "capability 門の前に重みの block を取っている");

  const badArity = baseDeclaration();
  // add に 3 本目の入力（capability 表は slot dtype しか見ないのでアリティは契約検査の担当）
  badArity.nodes[1].ins = ["h", "b", "b"];
  const contractViolation = countingContainer(await openModelBytes(badArity, baseTensors()));
  assertThrows(
    () => prepareContainer(contractViolation.container, GRAPH_NAME),
    OpContractError,
  );
  assertEquals(contractViolation.reads(), 0, "契約検査の前に重みの block を取っている");
});
