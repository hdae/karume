// gemma の **quant が宣言した GPU feature**（`gpuFeatures` — ADR 0038 §3）の結線の門（GPU 不要）。
//
// gemma は以前 `quant.gpuFeatures` を読んでいなかった（他家族は admission 席で共有 GPU の能力を
// 見ていた）。ここで縛るのは `fromPretrained` の admission 閉包が**同じ 1 本**
// （`session/gpu-features.ts` の `sessionGpuFeatures` → `assertGpuFeaturesGranted`）を通ること:
// shader-f16 を宣言した配布形と shader-f16 の無い共有 GPU を渡すと、重みの part を取る前に
// 席の名前つきで落ちる。宣言しない配布形は同じ共有 GPU でこの門を抜け、後段（PLE 索引の読み）
// まで進む（門が恒真でないことの対照）。
//
// 配布形は疑似 HF（`helpers/container-loading-fixture.ts`）に載せた**宣言だけの容器**で、
// `admitGemma4` の構造検査（RoPE 派生入力 4 本の幅・出力 2 本・容量記号 1 本）を通る最小形に
// してある。PLE 索引も重みも持たないので、門を抜けた対照は PLE 索引の段で落ちる。
//
// NOTE: 共有 GPU は `runtime/tests/helpers/fake-gpu.ts` の実物の `GpuContext`（feature 集合は
// device の `features` から読まれる — 空集合 = shader-f16 なし）。container_loading_test.ts の
// NOTE と同じ理由で、ここだけ runtime のテスト道具を借りる。

import { assertEquals, assertRejects, assertStringIncludes } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { fakeDevice, fakeGpuContext } from "../../runtime/tests/helpers/fake-gpu.ts";
import type { ModelInput } from "../../runtime/tests/helpers/container-write.ts";
import { Gemma4Pipeline } from "../gemma.ts";
import { parseIrDeclarationValue } from "./helpers/container-fixture.ts";
import {
  HUB_URL,
  REPO,
  serveContainer,
  serveRepo,
  SHA,
} from "./helpers/container-loading-fixture.ts";

/** gemma の weights 部品名（`src/gemma/pipeline.ts` の `MODEL`）= 容器の中のグラフ名。 */
const MODEL = "model";

/** RoPE の層種別の幅（`gemma4_config_test.ts` の `ROPE` と同じ — 実配布形の sliding 256 / full 512）。 */
const ROPE = {
  sliding_attention: { theta: 10000, headDim: 256, rotaryDim: 256 },
  full_attention: { theta: 1000000, headDim: 512, rotaryDim: 128 },
} as const;

/** `parseGemma4PipelineConfig` を通る最小の宣言（`gemma4_config_test.ts` の `MINIMAL`）。 */
const PIPELINE_CONFIG = {
  chunkLength: 32,
  maxChunkLength: 128,
  maxPosition: 1024,
  capacity: 640,
  rope: ROPE,
} as const;

/**
 * `admitGemma4` の構造検査を通る最小の宣言（実行はしない）。
 *
 * MUST: 構造検査は**通る**形で綴る — ここで落ちると、その後段に居る feature の門を踏んだことに
 * ならない。見られるのは RoPE 派生入力 4 本の `[1, M, headDim]`・出力 2 本（logits `[1,R,V]` /
 * hidden `[1,R,H]`）・入力から決まらない記号がちょうど 1 本（容量 `C`）の 3 点。
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
        // 容量記号 `C` は states にしか現れない（入力 shape から決まらない記号 = 容量 — admission の
        // 検査）。
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
          // 宣言したスロットは参照されていなければならない（IR の参照完全性）。
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

/** gemma4 の配布形（容器 1 本）を疑似 HF に載せ、`quant` 欄だけを差し替える。 */
const loadWithQuant = async (quant: Record<string, unknown>): Promise<Gemma4Pipeline> => {
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
  return await Gemma4Pipeline.fromPretrained(
    { repo: REPO, revision: SHA, hubUrl: HUB_URL },
    {
      fetch: rig.mock.fetch,
      caches: rig.hubOptions.caches,
      // shader-f16 を持たない共有 GPU（device の features が空集合）。
      gpu: fakeGpuContext(fakeDevice()),
    },
  );
};

describe("Gemma4Pipeline.fromPretrained の gpuFeatures（共有 GPU）", () => {
  it("shader-f16 を宣言した quant は、共有 GPU に無ければ admission で quant 名つきで落ちる", async () => {
    const error = await assertRejects(
      () => loadWithQuant({ gpuFeatures: { shaderF16: true } }),
      Error,
    );
    assertStringIncludes(error.message, "Gemma4Pipeline: quant 'f32' は shader-f16 を要求するが");
  });

  it("宣言しない quant は同じ共有 GPU でこの門を抜け、PLE 索引の段まで進む（対照）", async () => {
    const error = await assertRejects(() => loadWithQuant({}), Error);
    // 門が恒真なら同じ文言で落ちる。抜けた先は feature → limits の後の PLE 索引の段（この配布形は
    // 索引を持たない）— 上の陽性ケースも、門が無ければここへ来る。
    assertEquals(error.message.includes("shader-f16"), false, error.message);
    assertStringIncludes(error.message, "'ple_index'");
  });
});
