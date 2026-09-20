// `@karume/runtime` の公開面のスナップショット門（ADR
// [0008](../../../docs/decisions/0008-public-api.md) 追記 2026-09-20）。GPU も実資産も要らない。
//
// 見るのは `deno.json` の `exports`（= `./mod.ts` 1 面）が出すシンボルの増減だけで、「何を出す
// べきか」は言わない。ADR 0008 の「薄い面」は人のレビューで守るものだが、内部モジュールの
// 再輸出を 1 行足すと面は一気に太るし、削除は消費者にとって破壊的変更になる。どちらも型検査
// では赤くならないので、差分を見る門をここに置く。
//
// 面を変えたときの更新手順は失敗メッセージが持つ（`KARUME_SURFACE=write`）。

import { assertPublicSurface } from "./public-surface.ts";

Deno.test("公開面: runtime の exports がスナップショットと一致する", async () => {
  await assertPublicSurface(new URL("../", import.meta.url));
});
