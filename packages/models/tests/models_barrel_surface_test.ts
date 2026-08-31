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
