// A Gemma 4 product graph with or without PLE (per-layer inputs) — the admission decision and the
// options it refuses. No GPU and no real assets.
//
// gemma-4-12B-it has no per-layer inputs: its graph has no `per_layer_inputs` input and its
// container no PLE assets. Admission settles "PLE present / absent" from both declarations and
// fails loudly when they disagree (one side was baked for another model), and a PLE-free model
// refuses the PLE placement options instead of silently ignoring them. E2B / E4B (graph input +
// assets) keep the PLE path.

import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { ModelInputError } from "../src/errors.ts";
import { admitGemma4Ple, gemmaEntryName } from "../src/gemma/admission.ts";
import { assertGemma4PleFreeOptions, type PipelineOptions } from "../src/gemma/pipeline.ts";
import { PLE_INDEX_ASSET, readGemma4PleIndex } from "../src/gemma/ple-index.ts";
import type { GraphOwner, ModelComponent } from "../src/hub/components.ts";
import { pleComponentOf, pleFixture } from "./helpers/ple-fixture.ts";
import { stubModel } from "./helpers/stub-model.ts";
import { fakeDevice, fakeGpuContext } from "../../runtime/tests/helpers/fake-gpu.ts";
import type { ModelInput } from "../../runtime/tests/helpers/container-write.ts";
import { Pipeline } from "../gemma.ts";
import { parseIrDeclarationValue } from "./helpers/container-fixture.ts";
import {
  HUB_URL,
  REPO,
  serveContainer,
  serveRepo,
  SHA,
} from "./helpers/container-loading-fixture.ts";

const ENTRY = gemmaEntryName("gemma4");

/** The declarations admission reads: the graph inputs and the container's asset table. */
type Declared = GraphOwner & Pick<ModelComponent, "assets">;

/**
 * A product graph declaration with or without the `per_layer_inputs` input (the rest is the shape
 * of the shipped graph; admission's PLE decision reads only the input names).
 */
const graphOf = (withPle: boolean): GraphOwner =>
  stubModel({
    symbols: ["C", "M", "R"],
    inputs: [
      { name: "input_ids", shape: [1, "M"], dtype: "i32" },
      ...(withPle ? [{ name: "per_layer_inputs", shape: [1, "M", 35, 256] }] : []),
      { name: "rope_sliding_attention_cos", shape: [1, "M", 256] },
      { name: "rope_sliding_attention_sin", shape: [1, "M", 256] },
      { name: "rope_full_attention_cos", shape: [1, "M", 512] },
      { name: "rope_full_attention_sin", shape: [1, "M", 512] },
      { name: "last_row", shape: ["R"], dtype: "i32" },
    ],
    outputs: ["logits", "hidden"],
    values: { logits: [1, "R", 262144], hidden: [1, "R", 3840] },
  });

/** The asset table of a container that carries PLE (index + blocks, as the exporter writes them). */
const pleAssets = (): Readonly<Record<string, string>> =>
  pleComponentOf(pleFixture({ tokens: 4, layers: 2, dim: 8, embedScale: 4 })).assets;

const declared = (withPle: boolean, assets: Readonly<Record<string, string>>): Declared => ({
  ...graphOf(withPle),
  assets,
});

describe("gemma4 admission: PLE present or absent is settled from the graph and the container", () => {
  it("graph input and PLE assets together: a model with PLE (E2B / E4B)", () => {
    assertEquals(admitGemma4Ple(declared(true, pleAssets()), ENTRY), true);
  });

  it("neither the input nor any PLE asset: a PLE-free model (12B)", () => {
    assertEquals(admitGemma4Ple(declared(false, {}), ENTRY), false);
    // Assets of other roles do not count as PLE.
    assertEquals(admitGemma4Ple(declared(false, { notes: "license" }), ENTRY), false);
  });

  it("the graph takes per-layer inputs but the container has no PLE assets: fails loudly", () => {
    const error = assertThrows(
      () => admitGemma4Ple(declared(true, {}), ENTRY),
      Error,
      "the graph takes 'per_layer_inputs' but the container declares no PLE assets",
    );
    assert(error.message.startsWith(`${ENTRY}: `), error.message);
    // A mismatch of the assets, not of the caller's request (ADR 0107 decision 2).
    assert(!(error instanceof ModelInputError));
  });

  it("the container has PLE assets but the graph has no per-layer input: fails naming them", () => {
    const error = assertThrows(
      () => admitGemma4Ple(declared(false, pleAssets()), ENTRY),
      Error,
      "but the graph has no 'per_layer_inputs' input",
    );
    assert(error.message.includes(`${PLE_INDEX_ASSET} (ple-index)`), error.message);
    assert(!(error instanceof ModelInputError));
  });

  it("each PLE spelling alone is enough to be a mix (the index name, or a block role)", () => {
    const mixes: readonly Readonly<Record<string, string>>[] = [
      { [PLE_INDEX_ASSET]: "misc" },
      { "ple.values.0": "ple-values" },
      { "ple.scales.0": "ple-scales" },
      { index: "ple-index" },
    ];
    for (const assets of mixes) {
      assertThrows(
        () => admitGemma4Ple(declared(false, assets), ENTRY),
        Error,
        "declares PLE assets",
      );
    }
  });

  it("the family's entry name prefixes the message", () => {
    const qat = gemmaEntryName("gemma4-qat");
    const error = assertThrows(() => admitGemma4Ple(declared(true, {}), qat), Error);
    assert(error.message.startsWith(`${qat}: `), error.message);
  });

  it("a model with PLE still goes on to the index check (the decision does not replace it)", async () => {
    const component = pleComponentOf(pleFixture({ tokens: 4, layers: 2, dim: 8, embedScale: 4 }));
    assertEquals(admitGemma4Ple({ ...graphOf(true), assets: component.assets }, ENTRY), true);
    const index = await readGemma4PleIndex("test", component);
    assertEquals(index.tokens, 4);
  });
});

describe("gemma4 PLE-free model: the PLE placement options are refused", () => {
  const refused: readonly [string, PipelineOptions][] = [
    ["pleResidency: host", { pleResidency: "host" }],
    ["pleResidency: gpu", { pleResidency: "gpu" }],
    ["maxResidentPleBytes", { maxResidentPleBytes: 0 }],
  ];
  for (const [label, options] of refused) {
    it(`${label} is an input error naming the option`, () => {
      const error = assertThrows(
        () => assertGemma4PleFreeOptions("gemma.Pipeline.fromPretrained", options),
        ModelInputError,
        "cannot be used with this model",
      );
      const [key] = Object.keys(options);
      assert(error.message.includes(key), error.message);
      assert(error.message.startsWith("gemma.Pipeline.fromPretrained: "), error.message);
    });
  }

  it("both options at once are named together", () => {
    assertThrows(
      () => assertGemma4PleFreeOptions("T", { pleResidency: "host", maxResidentPleBytes: 1024 }),
      ModelInputError,
      "pleResidency / maxResidentPleBytes",
    );
  });

  it("options that do not concern PLE pass (negative control)", () => {
    assertGemma4PleFreeOptions("T", {});
    assertGemma4PleFreeOptions("T", { chunkLength: 64, stateAttentionReduce: "parallel" });
  });
});

// ---- fromPretrained: the refusal comes before any weight part is fetched ---------------------
//
// A PLE-free distribution on a fake Hugging Face (`helpers/container-loading-fixture.ts`, which
// records every fetched path): one `model` container whose weight block lives in a weight part
// (part 1 onward), and no tokenizer asset. With a PLE option the load fails at admission: the
// descriptor (part 0) was read, no weight part was. Without it (the control) the same load fetches
// the weight part and fails later, at the missing tokenizer — so "no weight part" is a property
// of where the refusal sits, not of a distribution that has none.

describe("gemma4 PLE-free fromPretrained: the PLE options fail before the weight parts", () => {
  const MODEL = "model";
  const rope = (name: string, width: number) => ({ name, dtype: "f32", shape: [1, "M", width] });
  /** The admission-passing declaration of `gemma4_gpu_features_test.ts`, plus one weight `w`. */
  const component = (): ModelInput => ({
    graphs: {
      [MODEL]: parseIrDeclarationValue({
        format: "karume-ir",
        version: 2,
        requires: { ops: ["mul", "state_append"] },
        symbols: ["C", "M"],
        inputs: [
          rope("rope_sliding_attention_cos", 256),
          rope("rope_sliding_attention_sin", 256),
          rope("rope_full_attention_cos", 512),
          rope("rope_full_attention_sin", 512),
        ],
        outputs: ["logits", "hidden"],
        initializers: { w: {} },
        values: {
          w: { dtype: "f32", shape: [256] },
          logits: { dtype: "f32", shape: [1, "M", 8] },
          hidden: { dtype: "f32", shape: [1, "M", 4] },
        },
        states: { kv: { dtype: "f32", shape: [1, "C", 4] } },
        nodes: [
          { op: "mul", ins: ["rope_sliding_attention_cos", "w"], outs: ["logits"], attrs: {} },
          {
            op: "mul",
            ins: ["rope_full_attention_cos", "rope_full_attention_sin"],
            outs: ["hidden"],
            attrs: {},
          },
          {
            op: "state_append",
            ins: ["rope_sliding_attention_cos"],
            outs: [],
            attrs: {},
            states: { slot: "kv" },
          },
        ],
      }),
    },
    consts: [],
    weights: [
      {
        graph: MODEL,
        initializer: "w",
        bytes: new Uint8Array(new Float32Array(256).fill(0.5).buffer),
        encoding: { codec: "f32" },
      },
    ],
    assets: [],
    provenance: { license: "test" },
  });

  const load = async (options: PipelineOptions) => {
    const served = await serveContainer(`${MODEL}/model.f32`, component());
    const rig = await serveRepo(
      {
        test: {
          pipeline: "gemma4/1",
          weights: { [MODEL]: { f32: served.entry } },
          assets: {},
          quants: { f32: { weights: { [MODEL]: "f32" }, session: {} } },
          defaultQuant: "f32",
          pipelineConfig: {
            chunkLength: 32,
            maxChunkLength: 128,
            maxPosition: 1024,
            capacity: 640,
            rope: {
              sliding_attention: { theta: 10000, headDim: 256, rotaryDim: 256 },
              full_attention: { theta: 1000000, headDim: 512, rotaryDim: 128 },
            },
          },
        },
      },
      new Map(served.files),
      { [MODEL]: served.parts },
    );
    const [descriptor, ...weightParts] = served.parts.map((part) => part.path);
    const fetchedWeightParts = (): string[] =>
      rig.mock.paths.filter((path) => weightParts.includes(path));
    const result = Pipeline.fromPretrained(
      { repo: REPO, revision: SHA, hubUrl: HUB_URL },
      {
        ...options,
        fetch: rig.mock.fetch,
        caches: rig.hubOptions.caches,
        gpu: fakeGpuContext(fakeDevice()),
      },
    );
    return { result, descriptor, weightParts, fetchedWeightParts, paths: rig.mock.paths };
  };

  for (const [label, options] of [
    ["pleResidency: host", { pleResidency: "host" }],
    ["pleResidency: gpu", { pleResidency: "gpu" }],
    ["maxResidentPleBytes", { maxResidentPleBytes: 0 }],
  ] as const) {
    it(`${label}: an input error after the descriptor, with no weight part fetched`, async () => {
      const loaded = await load(options);
      assert(loaded.weightParts.length > 0, "the fixture has no weight part (the case is vacuous)");
      await assertRejects(() => loaded.result, ModelInputError, "cannot be used with this model");
      assert(loaded.paths.includes(loaded.descriptor), "admission did not read the descriptor");
      assertEquals(loaded.fetchedWeightParts(), [], "weight parts fetched before the refusal");
    });
  }

  it("control: without a PLE option the load fetches the weight part and fails later", async () => {
    const loaded = await load({});
    const error = await assertRejects(() => loaded.result, Error);
    assert(error.message.includes("asset 'tokenizer' is missing"), error.message);
    assert(loaded.fetchedWeightParts().length > 0, "the control fetched no weight part");
  });
});
