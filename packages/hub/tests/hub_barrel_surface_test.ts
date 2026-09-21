// barrel（`mod.ts`）の**値非公開**の門（models の `models_barrel_surface_test.ts` と同型）。
// ネットワークも実資産も要らない。
//
// `DistributionSource` は**中身が不透明**なハンドルで、判別は `isDistributionSource` に閉じる
// （`mod.ts` の doc が理由の正本 — 構造で見分けようとすると綴り間違いが取得元として通る）。
// その不透明さは `export type` でしか保てないのに、`export` へ変えても型検査は通り、公開面
// スナップショット門も差分を出さない（`deno doc --json` は type-only 再輸出と値再輸出を
// 区別しない — 2026-09-21 実測）。値として出ていないことは実行時にしか観測できない。

import { assert, assertEquals } from "@std/assert";
import * as hub from "../mod.ts";

/** 型としてのみ公開する綴り（不透明ハンドルの面）。 */
const TYPE_ONLY = ["DistributionSource"];

Deno.test("barrel: 不透明ハンドルの型は値として出ていない", () => {
  const surface = Object.keys(hub);
  // 陽性対照 — 空の名前空間を見て緑になっている形でないことを先に確かめる。
  assert(
    surface.includes("isDistributionSource"),
    `barrel の import が壊れている（${surface.length} 件）`,
  );
  assertEquals(
    surface.filter((name) => TYPE_ONLY.includes(name)),
    [],
    "型としてのみ公開する綴りが値として出ている（不透明ハンドルの判別が構造依存になる）",
  );
});
