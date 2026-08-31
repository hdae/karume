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
const GEMMA_VALUES = ["Gemma4Pipeline", "gemma4ChatPrompt", "GenerationCapacityError"];

/** 公開面に出してはならない綴り（内部の組み立て口）。 */
const INTERNAL_VALUES = [
  "createGenerationProgram",
  "createGenerationSequence",
  "createSampler",
  "createGemma4Ple",
  "parseGemma4PleIndex",
  "parseGemmaTokenizerAsset",
  "renderGemma4Chat",
  "gemma4StopTokens",
  "GemmaTokenizer",
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
