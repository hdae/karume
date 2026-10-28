// Checks of published-smoke's pure logic (only the part that needs neither the network nor the
// published packages).
//
// The main script (main.ts) fetches `jsr:@karume/*` from the registry as soon as it starts, so the
// checks are split out into cli.ts.

import { test } from "vite-plus/test";
import { assertEquals, assertThrows } from "@std/assert";
import { isPinRef, isSbv2Module, parseSmokeArgs } from "../cli.ts";

test("isPinRef: accepts as a pin only a value with both repo and revision", () => {
  assertEquals(isPinRef({ repo: "a", revision: "b" }), true);
  assertEquals(isPinRef({ repo: "a" }), false);
  assertEquals(isPinRef({ revision: "b" }), false);
  assertEquals(isPinRef({ repo: "a", revision: 1 }), false);
  assertEquals(isPinRef(null), false);
  assertEquals(isPinRef("x"), false);
  assertEquals(isPinRef(undefined), false);
});

test("isSbv2Module: accepts the 0.15 surface of the sbv2 subpath and nothing else", () => {
  const dispose = (): Promise<void> => Promise.resolve();
  class Pipeline {
    static fromPretrained = (): Promise<{ dispose: () => Promise<void> }> =>
      Promise.resolve({ dispose });
  }
  assertEquals(isSbv2Module({ Pipeline, SOURCES: {} }), true);
  // The 0.14 spelling of the same module.
  assertEquals(isSbv2Module({ Sbv2Pipeline: Pipeline, SBV2_SOURCES: {} }), false);
  assertEquals(isSbv2Module({ Pipeline: class {}, SOURCES: {} }), false);
  assertEquals(isSbv2Module({ Pipeline, SOURCES: null }), false);
  assertEquals(isSbv2Module(null), false);
});

test("parseSmokeArgs: no argument is the full path, --manifests-only is manifests only", () => {
  assertEquals(parseSmokeArgs([]), { manifestsOnly: false });
  assertEquals(parseSmokeArgs(["--manifests-only"]), { manifestsOnly: true });
});

test("parseSmokeArgs: a typo does not fall silently through to the full path but fails with the usage", () => {
  assertThrows(
    () => parseSmokeArgs(["--manifest-only"]),
    Error,
    "unknown argument --manifest-only",
  );
  assertThrows(() => parseSmokeArgs(["--manifests_only"]), Error, "unknown argument");
  assertThrows(() => parseSmokeArgs(["extra"]), Error, "unknown argument extra");
});
