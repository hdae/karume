// `@karume/models` の公開面のスナップショット門（ADR
// [0008](../../../docs/decisions/0008-public-api.md) 追記 2026-09-20）。GPU も実資産も要らない。
//
// 同じディレクトリの `models_barrel_surface_test.ts` とは役割が違う:
//
// - 名指しの門（あちら）= **意図の宣言**。「`Gemma4Pipeline` は出す」「`createGemma4Ple` は
//   出さない」を人が書く。書いていない綴りの増減は素通りする。
// - この門 = **増減の検出**。何が正しいかは言わず、barrel（`.`）と 9 サブパスそれぞれについて
//   前回との差だけを見る。型 export も採るので、`Object.keys` では観測できない
//   `export type` の再輸出もここで縛れる（あちらの冒頭 NOTE の穴）。
//
// 両建て（ADR 0037）の面が 10 もあるので、1 面だけが痩せた・太った形は目視では拾えない。
//
// 門の実体は runtime 側に 1 本だけ置く（`shard-files.ts` 等と同じパッケージ跨ぎ相対 import）。

import { assertPublicSurface } from "../../runtime/tests/helpers/public-surface.ts";

Deno.test("公開面: models の exports（barrel + 9 サブパス）がスナップショットと一致する", async () => {
  await assertPublicSurface(new URL("../", import.meta.url));
});
