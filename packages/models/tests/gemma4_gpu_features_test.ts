// Checks of the wiring of the **GPU features a gemma quant declares** (`gpuFeatures` — ADR 0038 §3) (no GPU needed).
//
// gemma used not to read `quant.gpuFeatures` (the other families checked the shared GPU's capabilities in the
// admission step). What is pinned here is that the admission closure of `fromPretrained` goes through **the same
// single path** (`sessionGpuFeatures` → `assertGpuFeaturesGranted` in `session/gpu-features.ts`): given a
// distribution that declares shader-f16 and a shared GPU without shader-f16, it fails with the preset's name before
// any weight part is fetched.
// A distribution without the declaration passes this gate on the same shared GPU and goes on to a
// later stage (the control that the gate is not always-failing).
//
// The distribution is a **declaration-only container** on a mock HF (`helpers/container-loading-fixture.ts`), the
// minimal shape that passes `admitGemma4`'s structural checks (the widths of the 4 RoPE-derived inputs, 2 outputs,
// 1 capacity symbol).
// It has neither a per_layer_inputs input nor PLE assets, so admission takes it as a PLE-free model
// (gemma-4-12B-it's form), and the manifest declares no tokenizer asset: the control that passes
// the gate fails at the tokenizer stage, after admission and the weight fetch.
//
// NOTE: the shared GPU is a real `GpuContext` from `runtime/tests/helpers/fake-gpu.ts` (its feature set is read
// from the device's `features` — an empty set = no shader-f16). For the same reason as the NOTE in
// container_loading_test.ts, only this file borrows a runtime test tool.

import { assertEquals, assertRejects, assertStringIncludes } from "@std/assert";
import { describe, it } from "vite-plus/test";
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

/** gemma's weights component name (`MODEL` in `src/gemma/pipeline.ts`) = the graph name inside the container. */
const MODEL = "model";

/** The RoPE width per layer type (the same as `ROPE` in `gemma4_config_test.ts` — the real distribution's sliding 256 / full 512). */
const ROPE = {
  sliding_attention: { theta: 10000, headDim: 256, rotaryDim: 256 },
  full_attention: { theta: 1000000, headDim: 512, rotaryDim: 128 },
} as const;

/** The minimal declaration that passes `parsePipelineConfig` (`MINIMAL` in `gemma4_config_test.ts`). */
const PIPELINE_CONFIG = {
  chunkLength: 32,
  maxChunkLength: 128,
  maxPosition: 1024,
  capacity: 640,
  rope: ROPE,
} as const;

/**
 * The minimal declaration that passes `admitGemma4`'s structural checks (never runs).
 *
 * MUST: write it in a shape that **passes** the structural checks — failing here would not exercise the feature
 * check that sits after them. Three things are checked: `[1, M, headDim]` for the 4 RoPE-derived inputs, 2 outputs
 * (logits `[1,R,V]` / hidden `[1,R,H]`), and exactly 1 symbol not determined by the inputs (the capacity `C`).
 */
const gemmaComponent = (): ModelInput => {
  const rope = (name: string, width: number) => ({
    name,
    dtype: "f32",
    shape: [1, "M", width],
  });
  return {
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
        initializers: {},
        values: {
          logits: { dtype: "f32", shape: [1, "M", 8] },
          hidden: { dtype: "f32", shape: [1, "M", 4] },
        },
        // The capacity symbol `C` appears only in the states (a symbol not determined by the input shapes = the
        // capacity — admission's check).
        states: { kv: { dtype: "f32", shape: [1, "C", 4] } },
        nodes: [
          {
            op: "mul",
            ins: ["rope_sliding_attention_cos", "rope_sliding_attention_sin"],
            outs: ["logits"],
            attrs: {},
          },
          {
            op: "mul",
            ins: ["rope_full_attention_cos", "rope_full_attention_sin"],
            outs: ["hidden"],
            attrs: {},
          },
          // A declared slot must be referenced (the IR's referential completeness).
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
    weights: [],
    assets: [],
    provenance: { license: "test" },
  };
};

/** Puts a gemma4 distribution (one container) on a mock HF and swaps only the `quant` field. */
const loadWithQuant = async (quant: Record<string, unknown>): Promise<Pipeline> => {
  const served = await serveContainer(`${MODEL}/model.f32`, gemmaComponent());
  const rig = await serveRepo(
    {
      test: {
        pipeline: "gemma4/1",
        weights: { [MODEL]: { f32: served.entry } },
        assets: {},
        quants: { f32: { weights: { [MODEL]: "f32" }, session: {}, ...quant } },
        defaultQuant: "f32",
        pipelineConfig: PIPELINE_CONFIG,
      },
    },
    new Map(served.files),
    { [MODEL]: served.parts },
  );
  return await Pipeline.fromPretrained(
    { repo: REPO, revision: SHA, hubUrl: HUB_URL },
    {
      fetch: rig.mock.fetch,
      caches: rig.hubOptions.caches,
      // A shared GPU without shader-f16 (the device's features are an empty set).
      gpu: fakeGpuContext(fakeDevice()),
    },
  );
};

describe("gemma.Pipeline.fromPretrained gpuFeatures (shared GPU)", () => {
  it("a quant that declares shader-f16 fails in admission with the quant name when the shared GPU lacks it", async () => {
    const error = await assertRejects(
      () => loadWithQuant({ gpuFeatures: { shaderF16: true } }),
      Error,
    );
    assertStringIncludes(error.message, "gemma.Pipeline: quant 'f32' requires shader-f16, but");
  });

  it("a quant without the declaration passes this gate on the same shared GPU and reaches the tokenizer stage (control)", async () => {
    const error = await assertRejects(() => loadWithQuant({}), Error);
    // An always-failing gate would fail with the same message. Past feature → limits → admission
    // (a PLE-free model) the next failure is the missing tokenizer asset — the positive case above
    // would reach it too if the gate were absent.
    assertEquals(error.message.includes("shader-f16"), false, error.message);
    assertStringIncludes(error.message, "asset 'tokenizer' is missing");
  });
});
