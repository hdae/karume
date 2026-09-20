// `@karume/hub` の公開面のスナップショット門（ADR
// [0008](../../../docs/decisions/0008-public-api.md) 追記 2026-09-20）。ネットワークも実資産も
// 要らない。
//
// hub は `.`（Web 標準 API だけの面）と `./deno`（Deno 専用アダプタ）の 2 面を出す（ADR 0080 /
// 0086）。2 面の**振り分け**が崩れる形 —— Deno 依存の綴りが `.` 側へ漏れる、あるいは `./deno`
// が痩せて消費者が内部パスを直に掴む —— は、どちらも型検査では赤くならない。面ごとに増減を
// 見ることでその移動を差分として出す。
//
// 門の実体は runtime 側に 1 本だけ置く（`shard-files.ts` 等と同じパッケージ跨ぎ相対 import）。

import { assertPublicSurface } from "../../runtime/tests/public-surface.ts";

Deno.test("公開面: hub の exports（. と ./deno）がスナップショットと一致する", async () => {
  await assertPublicSurface(new URL("../", import.meta.url));
});
