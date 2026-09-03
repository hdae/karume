// barrel（`mod.ts`）の公開面の門。GPU も実資産も要らない。
//
// ADR [0083](../../../docs/decisions/0083-generation-api-surface.md) 決定 9 で `generateGreedy` /
// `GreedySpec` を**削除した**（破壊的変更 — 消費側の doc は
// [limitations](../../../docs/limitations.md)）。実装は `src/generation/greedy.ts` に parity 検収用の
// 内部ヘルパとして残っているので、`mod.ts` へ 1 行足せば公開面がいつでも戻ってしまう。値の再輸出は
// 型検査では捕まらない（**増えた** export は誰も咎めない）ので、門はここにしか置けない。
//
// NOTE: 型 export（`GreedySpec`）は実行時に観測できないので、ここで見るのは値 export だけ。
// 型だけを戻した場合は「型はあるのに実装が無い」形になり、消費側の `deno check` が落ちる。

import { assert, assertEquals } from "@std/assert";
import * as models from "../mod.ts";

Deno.test("barrel: 生成ループは公開面に無い（ADR 0083 決定 9 の格下げ）", () => {
  const surface = Object.keys(models);
  // 陽性対照 — この門が「空の名前空間を見て緑になっている」形でないことを先に確かめる。
  assert(surface.includes("encodePng"), `barrel の import が壊れている（${surface.length} 件）`);
  assertEquals(
    surface.filter((name) => name.toLowerCase().includes("greedy")),
    [],
    "barrel に greedy 系の export が戻っている",
  );
});

Deno.test("barrel: 内部ヘルパとしての generateGreedy は残っている（消えたのは公開面だけ）", async () => {
  const greedy = await import("../src/generation/greedy.ts");
  assertEquals(typeof greedy.generateGreedy, "function");
  assertEquals(typeof greedy.planPrefillChunks, "function");
});

// ---- gemma（生成 API 波の段 4）--------------------------------------------
//
// 値 export の集合だけを見る（型は実行時に観測できない）。ここで縛るのは 2 点:
//
// ① 出すべきものが出ていること — 面が痩せると消費者は内部パスへ直接 import する
// ② **組み立ての入口を出さないこと** — `createGenerationProgram` / `createGenerationSequence` /
//    `createGemma4Ple` は静的配線の検証と資産の突合を通す前の半端な実体を作れる面で、
//    入口はパイプラインだけにする（ADR 0008）。増えた export は型検査では咎められない

/** barrel と `./gemma` の両方が出す gemma / 生成 API の**値** export。 */
const GEMMA_VALUES = [
  "Gemma4Pipeline",
  "gemma4ChatPrompt",
  // 多ターンを `sequence` で回す消費者の入口。無いと「全体を描いて先頭の `<bos>` を剥がす」
  // 当て推量を書かされる（turn-local 契約が公開面から読めない）。
  "gemma4ChatTurn",
  "GenerationCapacityError",
  // `fromAssets` を使う消費者が hub の `Record<string, unknown>` から config を組む口。
  // 出していないと `as` で被せるか、3 つの数を配布形と消費側で二重持ちすることになる。
  "parseGemma4PipelineConfig",
  // RoPE の cos / sin はグラフに焼かれておらずホストが chunk ごとに作る（ADR 0091 決定 1）。
  // 低レベル面（`fromAssets` / 自前の生成ループ）の消費者はこれが無いと表を自前で組む羽目になる。
  "gemma4RopeInputs",
  // 多ターンの会話を KV 継続で回す高レベル面（ADR 0083 追記 2026-09-02）。`mod.ts` の 1 行が
  // 消えても両建て門（`./gemma` ⊆ barrel）は緑のままなので、名指しで縛るのはここだけ。
  "Gemma4ChatSession",
  // 溢れたときの既定ポリシー。消費者が**名指しで差し替える**口なので公開面に要る
  // （出していないと「既定と同じものを自分で書く」しかなくなる）。
  "dropOldestTurns",
];

/** 公開面に出してはならない綴り（内部の組み立て口）。 */
const INTERNAL_VALUES = [
  "createGenerationProgram",
  "createGenerationSequence",
  "createSampler",
  "createGemma4Ple",
  "parseGemma4PleIndex",
  "parseGemmaTokenizerAsset",
  "renderGemma4Chat",
  "renderGemma4ChatTurn",
  "gemma4StopTokens",
  "GemmaTokenizer",
  // chat 層の内部実装（`gemma_chat_test.ts` が `src/` から直に掴んでいる面）。公開面は
  // `Gemma4Pipeline.chat` / `Gemma4ChatSession` だけで、逐次復号の部品は出さない
  // — 出すと「停止文字列フィルタと detokenizer を自分で繋ぐ」形が消費者の正道になってしまう。
  // `StreamingDetokenizer` は `export type` の再輸出なので**値としては**出ていない（型だけを
  // 戻した場合は消費側の `deno check` が落ちる — 冒頭の NOTE）。
  "createStopStringFilter",
  "decodeChatChunks",
  "chatStreamOf",
  "StreamingDetokenizer",
];

Deno.test("barrel: gemma の公開面（薄い面 — 組み立ての入口は出さない）", async () => {
  const gemma = await import("../gemma.ts");
  const barrel = Object.keys(models);
  const subpath = Object.keys(gemma);

  for (const name of GEMMA_VALUES) {
    assert(barrel.includes(name), `barrel に ${name} が無い`);
    assert(subpath.includes(name), `./gemma に ${name} が無い`);
  }
  for (const name of INTERNAL_VALUES) {
    assertEquals(barrel.includes(name), false, `barrel に内部の ${name} が出ている`);
    assertEquals(subpath.includes(name), false, `./gemma に内部の ${name} が出ている`);
  }
  // 両建て（ADR 0037）の食い違いを塞ぐ — サブパスの値 export は barrel にも全部載ること。
  assertEquals(
    subpath.filter((name) => !barrel.includes(name)),
    [],
    "./gemma にあって barrel に無い値 export",
  );
});
