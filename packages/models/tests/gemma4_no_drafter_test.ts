// A gemma4 distribution whose model ships no drafter — 26B A4B of the q4_0 QAT line, until its
// drafter lands (ADR 0145 decision 1). `speculative` is refused through the missing role: the
// weights resolution has no `drafter` to select, so the load fails before any weight part is fetched,
// with nothing in the pipeline named after the model. No GPU: a fake hub serving one small container.

import { assertEquals, assertFalse, assertRejects } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { ManifestReferenceError } from "@karume/hub";
import type { ModelInput } from "../../runtime/tests/helpers/container-write.ts";
import { Pipeline } from "../src/gemma/pipeline.ts";
import { parseIrDeclarationValue } from "./helpers/container-fixture.ts";
import {
  fileRef,
  HUB_URL,
  REPO,
  serveContainer,
  serveRepo,
  SHA,
} from "./helpers/container-loading-fixture.ts";

/**
 * One small valid graph (not a gemma4 one): the refusal comes before any container is opened, and
 * without `speculative` the family admission refuses the placeholder declaration first.
 */
const component = (): ModelInput => ({
  graphs: {
    model: parseIrDeclarationValue({
      format: "karume-ir",
      version: 2,
      requires: { ops: ["linear"] },
      symbols: ["M"],
      inputs: [{ name: "x", dtype: "f32", shape: ["M", 4] }],
      outputs: ["y"],
      initializers: { w: {}, b: {} },
      values: {
        w: { dtype: "f32", shape: [4, 4] },
        b: { dtype: "f32", shape: [4] },
        y: { dtype: "f32", shape: ["M", 4] },
      },
      nodes: [{ op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} }],
    }),
  },
  consts: [],
  weights: [
    {
      graph: "model",
      initializer: "w",
      bytes: new Uint8Array(new Float32Array(16).buffer),
      encoding: { codec: "f32" as const },
    },
    {
      graph: "model",
      initializer: "b",
      bytes: new Uint8Array(new Float32Array(4).buffer),
      encoding: { codec: "f32" as const },
    },
  ],
  assets: [],
  provenance: { license: "test" },
});

/** A distribution with the `model` role alone, as 26B ships. */
const drafterless = async () => {
  const model = await serveContainer("model/model.i4", component());
  const tokenizer = Uint8Array.from([0x7b, 0x7d]);
  const rig = await serveRepo(
    {
      test: {
        pipeline: "gemma4/1",
        weights: { model: { i4: model.entry } },
        assets: { tokenizer: await fileRef("tokenizer.json", tokenizer) },
        quants: { i4: { weights: { model: "i4" }, session: {} } },
        defaultQuant: "i4",
        pipelineConfig: {},
      },
    },
    new Map([...model.files, ["tokenizer.json", tokenizer]]),
    { model: model.parts },
  );
  const open = (options: { readonly speculative?: Record<string, never> }) =>
    Pipeline.fromPretrained(
      { repo: REPO, revision: SHA, hubUrl: HUB_URL },
      { ...options, fetch: rig.mock.fetch, caches: rig.hubOptions.caches },
    );
  const parts = model.parts.map((part) => part.path);
  return { open, fetchedParts: () => rig.mock.paths.filter((path) => parts.includes(path)) };
};

describe("gemma.Pipeline over a distribution whose model ships no drafter", () => {
  it("refuses speculative through the missing drafter role, before any weight part is fetched", async () => {
    const rig = await drafterless();
    await assertRejects(
      () => rig.open({ speculative: {} }),
      ManifestReferenceError,
      "weights 'drafter' is not in the manifest (available: model)",
    );
    assertEquals(rig.fetchedParts(), [], "parts fetched before the refusal");
  });

  it("without speculative, the same distribution gets past the weights resolution (the refusal is speculation's)", async () => {
    const rig = await drafterless();
    // The load goes on to the family admission, which refuses this placeholder declaration.
    const error = await assertRejects(() => rig.open({}), Error, "pipelineConfig");
    assertFalse(error instanceof ManifestReferenceError, error.message);
  });
});
