// 掃引の CLI（main.ts）の引数のうち候補集合（--set / --quick）の門。GPU を使わない。

import { assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { parseFlags } from "./main.ts";

describe("parseFlags: 候補集合", () => {
  it("指定しなければ quick+（以前の「省略 = full」ではない）", () => {
    assertEquals(parseFlags([]).candidateSet, "quick+");
  });

  it("--set で quick / quick+ / full を選び、--quick は --set quick の別名", () => {
    for (const set of ["quick", "quick+", "full"] as const) {
      assertEquals(parseFlags(["--set", set]).candidateSet, set);
    }
    assertEquals(parseFlags(["--quick"]).candidateSet, "quick");
  });

  it("知らない集合と 2 回の指定は落ちる（どちらが効いたかを推測させない）", () => {
    assertThrows(() => parseFlags(["--set", "quik"]), Error, "のどれでもない");
    assertThrows(() => parseFlags(["--quick", "--set", "full"]), Error, "2 回");
    assertThrows(() => parseFlags(["--set", "quick", "--quick"]), Error, "2 回");
  });
});
