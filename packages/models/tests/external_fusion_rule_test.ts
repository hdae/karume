// The fusion rule of tests/helpers/gelu-mul-rule.ts is written the way a package outside the runtime
// writes one (ADR 0148): its only import is `@karume/runtime/kernel` — no runtime-internal module, no
// load hook (that would need `node:module`) — and its code has no type assertion, `any` or
// TypeScript directive to get around the entry's types. No GPU needed; what the rule computes is
// checked in the runtime (tests/gpu_external_fusion_rule_test.ts) and on Gemma 4 26B's graph in the
// e2e package (e2e_gemma4_26b_external_rule_test.ts).

import { importSpecifiersOf } from "@karume/test-support/module-graph";
import { readTextFile } from "@karume/test-support/fs";
import { describe, it } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import { GELU_MUL_RULE, GELU_MUL_RULE_NAME } from "./helpers/gelu-mul-rule.ts";

const RULE_FILE = new URL("./helpers/gelu-mul-rule.ts", import.meta.url);

/** The source without its comments (a comment may say "as" or "any" in prose). */
const codeOf = (source: string): string =>
  source.replaceAll(/\/\*[\s\S]*?\*\//g, "").replaceAll(/\/\/.*$/gm, "");

describe("the external gelu -> mul rule", () => {
  it("imports @karume/runtime/kernel and nothing else", async () => {
    const source = await readTextFile(RULE_FILE);
    assertEquals(importSpecifiersOf(RULE_FILE.href, source), ["@karume/runtime/kernel"]);
  });

  it("has no type assertion, no any and no TypeScript directive in its code", async () => {
    const code = codeOf(await readTextFile(RULE_FILE));
    // The control: the stripped code still holds the rule.
    assertEquals(code.includes("defineRule<GeluMulMatch"), true, "the code was stripped away");
    assertEquals(code.match(/\bas\s+(?!const\b)\w+/g), null, "a type assertion");
    assertEquals(code.match(/\b(?:any|never)\b/g), null, "any or never");
    assertEquals(code.match(/@ts-/g), null, "a TypeScript directive");
  });

  it("is a defined external rule of class E under its namespaced name", () => {
    assertEquals(
      [GELU_MUL_RULE.name, GELU_MUL_RULE.contract, GELU_MUL_RULE.heads],
      [GELU_MUL_RULE_NAME, "E", ["gelu_tanh"]],
    );
  });
});
