// The fusion rule of tests/helpers/gelu-mul-rule.ts is written the way a package outside the runtime
// writes one (ADR 0148): its only import is `@karume/runtime/kernel` — no runtime-internal module, no
// load hook (that would need `node:module`) — and its code has no type assertion, `any` or
// TypeScript directive to get around the entry's types; and a Session takes it through the same
// construction check that refuses anything defineRule did not build. No GPU needed; what the rule
// computes is checked in the runtime (tests/gpu_external_fusion_rule_test.ts) and on Gemma 4 26B's
// graph in the e2e package (e2e_gemma4_26b_external_rule_test.ts).

import { importSpecifiersOf } from "@karume/test-support/module-graph";
import { readTextFile } from "@karume/test-support/fs";
import { describe, it } from "vite-plus/test";
import { assertEquals, assertRejects } from "@std/assert";
import { createSessionFromContainer, ExecutionError, type SessionOptions } from "@karume/runtime";
import { fakeDevice, fakeGpuContext } from "../../runtime/tests/helpers/fake-gpu.ts";
import {
  type DeclarationJson,
  GRAPH_NAME,
  openModelBytes,
} from "../../runtime/tests/helpers/model-fixture.ts";
import { GELU_MUL_RULE, GELU_MUL_RULE_NAME } from "./helpers/gelu-mul-rule.ts";

const RULE_FILE = new URL("./helpers/gelu-mul-rule.ts", import.meta.url);

/** `y = mul(gelu_tanh(gate), other)`, the rule's adjacent window. */
const GELU_MUL_GRAPH: DeclarationJson = {
  format: "karume-ir",
  version: 2,
  requires: { ops: ["gelu_tanh", "mul"] },
  symbols: [],
  inputs: [
    { name: "gate", dtype: "f32", shape: [8] },
    { name: "other", dtype: "f32", shape: [8] },
  ],
  outputs: ["y"],
  initializers: {},
  values: { h: { dtype: "f32", shape: [8] }, y: { dtype: "f32", shape: [8] } },
  nodes: [
    { op: "gelu_tanh", ins: ["gate"], outs: ["h"], attrs: {} },
    { op: "mul", ins: ["h", "other"], outs: ["y"], attrs: {} },
  ],
};

/** Session construction on a fake device: it checks `fusionRules` before any GPU work. */
const construct = async (fusionRules: SessionOptions["fusionRules"]) =>
  await createSessionFromContainer(
    fakeGpuContext(
      Object.assign(fakeDevice(), { limits: { minStorageBufferOffsetAlignment: 256 } }),
    ),
    await openModelBytes(GELU_MUL_GRAPH, []),
    GRAPH_NAME,
    { fusionRules },
  );

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

  it("is a defined external rule of class E under its namespaced name", async () => {
    assertEquals(
      [GELU_MUL_RULE.name, GELU_MUL_RULE.contract, GELU_MUL_RULE.heads],
      [GELU_MUL_RULE_NAME, "E", ["gelu_tanh"]],
    );
    // A Session takes it through the construction check that refuses anything defineRule did not
    // build; a spread copy of it, which carries no brand, is refused by the same check.
    await (await construct([GELU_MUL_RULE])).dispose();
    await assertRejects(
      () => construct([{ ...GELU_MUL_RULE }]),
      ExecutionError,
      "options.fusionRules[0] is not a rule built by defineRule",
    );
  });
});
