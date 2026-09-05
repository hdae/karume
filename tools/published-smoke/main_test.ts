// published-smoke の純ロジックの門（ネットワークも公開物も要らない部分だけ）。
//
// 本体（main.ts）は起動と同時に registry から `jsr:@karume/*` を取りに行くので、門は cli.ts に
// 切り出してある。

import { assertEquals, assertThrows } from "@std/assert";
import { isPinRef, parseSmokeArgs } from "./cli.ts";

Deno.test("isPinRef: repo と revision が揃った値だけを pin と認める", () => {
  assertEquals(isPinRef({ repo: "a", revision: "b" }), true);
  assertEquals(isPinRef({ repo: "a" }), false);
  assertEquals(isPinRef({ revision: "b" }), false);
  assertEquals(isPinRef({ repo: "a", revision: 1 }), false);
  assertEquals(isPinRef(null), false);
  assertEquals(isPinRef("x"), false);
  assertEquals(isPinRef(undefined), false);
});

Deno.test("parseSmokeArgs: 引数なしは完全経路・--manifests-only は manifest だけ", () => {
  assertEquals(parseSmokeArgs([]), { manifestsOnly: false });
  assertEquals(parseSmokeArgs(["--manifests-only"]), { manifestsOnly: true });
});

Deno.test("parseSmokeArgs: 打ち間違いは黙って完全経路へ落ちず、使い方つきで落ちる", () => {
  assertThrows(
    () => parseSmokeArgs(["--manifest-only"]),
    Error,
    "未知の引数 --manifest-only",
  );
  assertThrows(() => parseSmokeArgs(["--manifests_only"]), Error, "未知の引数");
  assertThrows(() => parseSmokeArgs(["extra"]), Error, "未知の引数 extra");
});
