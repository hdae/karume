// barrel（`mod.ts`）の**値非公開**の門（models の `models_barrel_surface_test.ts` と同型）。
// GPU も実資産も要らない。
//
// `GpuContext` / `BatchScope` / `ResidentTensor` は「構築の入口を 1 本に絞る」ために
// `export type` でだけ出している（`mod.ts` の doc が理由の正本 — 直接構築すると
// planRequiredLimits / assertLimitsGranted や errorScope の門を迂回した実体が作れる）。
// `export type` を `export` に変えても**型検査は通る**し、公開面スナップショット門も差分を
// 出さない（`deno doc --json` は type-only 再輸出と値再輸出を `kind` でも `declarationKind`
// でも区別しない — 2026-09-21 実測）。値として出ていないことは実行時にしか観測できないので、
// 名指しの門をここに置く。

import { assert, assertEquals } from "@std/assert";
import * as runtime from "../mod.ts";

/** 型としてのみ公開する綴り（構築口を塞ぐために値を出していない面）。 */
const TYPE_ONLY = ["BatchScope", "GpuContext", "ResidentTensor"];

Deno.test("barrel: 構築口を塞いでいる面は値として出ていない", () => {
  const surface = Object.keys(runtime);
  // 陽性対照 — 空の名前空間を見て緑になっている形でないことを先に確かめる。
  assert(surface.includes("acquireGpu"), `barrel の import が壊れている（${surface.length} 件）`);
  assertEquals(
    surface.filter((name) => TYPE_ONLY.includes(name)),
    [],
    "型としてのみ公開する綴りが値として出ている（構築の入口が 1 本でなくなる）",
  );
});
