// The candidate fusion rule of `@karume/candidate-rules` (packages/candidate-rules/src/gelu-mul.ts)
// is written the way a package outside the runtime writes one (ADR 0148): its only import is
// `@karume/runtime/kernel` — no runtime-internal module, no load hook (that would need
// `node:module`) — and its code has no type assertion, `any` or TypeScript directive to get around
// the entry's types; every other file of the package imports nothing else of the runtime and
// nothing outside the package; and a Session takes the rule through the same construction check
// that refuses anything defineRule did not build. No GPU needed; what the rule
// computes is checked against its unfused plan on the GPU (e2e_external_fusion_rule_test.ts) and
// planned on Gemma 4 26B's graph (e2e_gemma4_26b_external_rule_test.ts), both in the e2e package.
//
// The adopted rules of this package are held to the same entry: `models/ew_chain`
// (packages/models/src/moss/ew-chain.ts, which the MOSS-Image pipeline passes by default) and
// `models/vae_rms_silu` (packages/models/src/vae/vae-rms-silu.ts, which the Wan and anima pipelines
// pass by default to their VAE decoder Sessions). Each one's only import is
// `@karume/runtime/kernel`, its code has no type assertion, `any` or directive, and a Session takes
// it. Their fused-against-unfused checks are e2e_ew_chain_rule_test.ts and
// e2e_vae_rms_silu_rule_test.ts.

import { importSpecifiersOf } from "@karume/test-support/module-graph";
import { readDir, readTextFile } from "@karume/test-support/fs";
import { describe, it } from "vite-plus/test";
import { assertEquals, assertRejects } from "@std/assert";
import { createSessionFromContainer, ExecutionError, type SessionOptions } from "@karume/runtime";
import { fakeDevice, fakeGpuContext } from "../../runtime/tests/helpers/fake-gpu.ts";
import {
  type DeclarationJson,
  GRAPH_NAME,
  openModelBytes,
} from "../../runtime/tests/helpers/model-fixture.ts";
import { GELU_MUL_RULE, GELU_MUL_RULE_NAME } from "@karume/candidate-rules";
import { VAE_RMS_SILU_RULE, VAE_RMS_SILU_RULE_NAME } from "../anima.ts";
import { EW_CHAIN_RULE, EW_CHAIN_RULE_NAME } from "../moss.ts";

/** The candidate package (every file of it is held to the entry a package outside the runtime has). */
const PACKAGE = new URL("../../candidate-rules/", import.meta.url);
const RULE_FILE = new URL("src/gelu-mul.ts", PACKAGE);

/** The adopted rules of this package: the file, the rule and the start of its definition. */
const ADOPTED_RULES = [
  {
    file: new URL("../src/moss/ew-chain.ts", import.meta.url),
    rule: EW_CHAIN_RULE,
    name: EW_CHAIN_RULE_NAME,
    definition: /defineRule<\s*EwChainMatch/,
  },
  {
    file: new URL("../src/vae/vae-rms-silu.ts", import.meta.url),
    rule: VAE_RMS_SILU_RULE,
    name: VAE_RMS_SILU_RULE_NAME,
    definition: /defineRule<\s*VaeRmsSiluMatch/,
  },
] as const;

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

/** Every TypeScript file under `directory` (its `node_modules` left out), as `file:` hrefs. */
const packageSources = async (directory: URL): Promise<string[]> => {
  const files: string[] = [];
  for await (const entry of readDir(directory)) {
    const url = new URL(entry.isDirectory ? `${entry.name}/` : entry.name, directory);
    if (entry.isDirectory && entry.name !== "node_modules")
      files.push(...(await packageSources(url)));
    else if (entry.isFile && /\.(ts|tsx|mts|cts)$/.test(entry.name)) files.push(url.href);
  }
  return files.sort();
};

/** The source without its comments (a comment may say "as" or "any" in prose). */
const codeOf = (source: string): string =>
  source.replaceAll(/\/\*[\s\S]*?\*\//g, "").replaceAll(/\/\/.*$/gm, "");

describe("the external gelu -> mul rule", () => {
  it("imports @karume/runtime/kernel and nothing else", async () => {
    const source = await readTextFile(RULE_FILE);
    assertEquals(importSpecifiersOf(RULE_FILE.href, source), ["@karume/runtime/kernel"]);
  });

  it("sits in a package whose every file imports @karume/runtime/kernel or its own files, nothing else", async () => {
    const files = await packageSources(PACKAGE);
    // The control: the walk found the entry and the rule.
    assertEquals(
      ["mod.ts", "src/gelu-mul.ts"].every((path) => files.includes(new URL(path, PACKAGE).href)),
      true,
      `the package's files: ${files.join(", ")}`,
    );
    for (const file of files) {
      const outside = importSpecifiersOf(file, await readTextFile(new URL(file))).filter(
        (specifier) =>
          specifier !== "@karume/runtime/kernel" &&
          !(specifier.startsWith(".") && new URL(specifier, file).href.startsWith(PACKAGE.href)),
      );
      assertEquals(outside, [], file);
    }
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

for (const adopted of ADOPTED_RULES) {
  describe(`the adopted rule ${adopted.name}`, () => {
    it("imports @karume/runtime/kernel and nothing else", async () => {
      const source = await readTextFile(adopted.file);
      assertEquals(importSpecifiersOf(adopted.file.href, source), ["@karume/runtime/kernel"]);
    });

    it("has no type assertion, no any and no TypeScript directive in its code", async () => {
      const code = codeOf(await readTextFile(adopted.file));
      // The control: the stripped code still holds the rule.
      assertEquals(adopted.definition.test(code), true, "the code was stripped away");
      assertEquals(code.match(/\bas\s+(?!const\b)\w+/g), null, "a type assertion");
      assertEquals(code.match(/\b(?:any|never)\b/g), null, "any or never");
      assertEquals(code.match(/@ts-/g), null, "a TypeScript directive");
    });

    it("is a defined external rule of class E under its models namespace, which a Session takes", async () => {
      assertEquals(
        [adopted.rule.name, adopted.rule.contract, adopted.name.startsWith("models/")],
        [adopted.name, "E", true],
      );
      await (await construct([adopted.rule])).dispose();
    });
  });
}
