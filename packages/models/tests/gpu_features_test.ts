// manifest の `gpuFeatures`（device 生成時にしか要求できない feature）→ `acquireGpu` の要求への
// 写像。ADR 0038 §3 の綴りの契約で、抜けは**沈黙劣化**（配布形が要求した能力を持たない device で
// 実行が進む）になる。8 家族全部が同じ 1 本を使うので、門もここ 1 本に集約する（元は 7 家族が
// `quant.gpuFeatures?.shaderF16 === true` を 1 行ずつ独立に読んでいた — レビュー M1-V9）。
//
// 取得後の検査（`assertGpuFeaturesGranted`）も GPU 無しで踏む。`GpuContext` は runtime が値と
// して公開しない（`acquireGpu` が唯一の入口 — ADR 0008）ので、向こうの helper
// `fake-gpu.ts` の `fakeGpuContext`（= **実物の `GpuContext`** に fake device を包む）を使う
// （`container_loading_test.ts` の⑧が先例）。自前に偽物を組むと検査対象そのものが偽物になるが、
// 実物を包む限り検査対象は本物のままである。
//
// 有効側の対照（feature を持つ device では通る）だけは helper に features を渡す口が要るので
// まだ書けない（runtime テスト担当へ依頼済み — `fakeGpuContext` の第 3 引数）。

import { assertEquals, assertStringIncludes, assertThrows } from "@std/assert";
import type { GpuFeaturesSpec } from "@karume/hub";
import { fakeDevice, fakeGpuContext } from "../../runtime/tests/helpers/fake-gpu.ts";
import {
  assertGpuFeaturesGranted,
  sessionGpuFeatures,
  toAcquireGpuOptions,
} from "../src/session/gpu-features.ts";

Deno.test("toAcquireGpuOptions: 宣言された feature だけを要求する", () => {
  // 宣言が無い / 明示的に false の配布形で feature を要求しにいくと、持たないアダプタでは
  // `acquireGpu` 自体が落ちる（= 動くはずの環境で動かなくなる）。
  assertEquals(toAcquireGpuOptions(undefined), {});
  assertEquals(toAcquireGpuOptions({}), {});
  assertEquals(toAcquireGpuOptions({ shaderF16: false }), {});
  assertEquals(toAcquireGpuOptions({ shaderF16: true }), { shaderF16: true });
});

Deno.test("toAcquireGpuOptions: GpuFeaturesSpec の全キーを写す（キー追加の取り残しを検出）", () => {
  // 写像の網羅は `src/session/gpu-features.ts` の `FEATURES`（`Required<GpuFeaturesSpec>` の
  // 網羅表）が型で固定しており、キーが増えれば**コンパイルエラー**になる。ここはその型門が
  // 生きていることを実行時からも見る対（型を緩めた改変は型検査を通ってしまうため）—
  // 全キーを true で埋めた spec を渡し、出てくるキー集合が入力と一致することを確かめる。
  const full: Required<GpuFeaturesSpec> = { shaderF16: true };
  const requested = toAcquireGpuOptions(full) as Record<string, unknown>;
  assertEquals(Object.keys(requested).sort(), Object.keys(full).sort());
});

// ---- 取得済み GpuContext での検査（共有 GPU 経路の唯一の門）--------------------

const WHERE = "TestPipeline: quant 'f16'";

/** shader-f16 を持たない device（`fakeGpuContext` は features を空 `Set` で作る）。 */
const noFeatureGpu = () => fakeGpuContext(fakeDevice());

Deno.test("assertGpuFeaturesGranted: 要求された feature が無効な共有 GPU は名指しで落ちる", () => {
  // 共有 GPU（`options.gpu`）を渡された経路では device 生成後なので feature を取り直せない
  // （ADR 0028）。ここを通すと Session 構築まで進んでから落ちるか、黙って別経路へ縮退する。
  const error = assertThrows(
    () => assertGpuFeaturesGranted({ shaderF16: true }, noFeatureGpu(), WHERE),
    Error,
  );
  // 診断の 3 点セット: どの席か / どの feature か / どう取り直すか。
  assertStringIncludes(error.message, WHERE);
  assertStringIncludes(error.message, "shader-f16");
  assertStringIncludes(error.message, "acquireGpu({ shaderF16: true })");
});

Deno.test("assertGpuFeaturesGranted: 要求していない配布形は no-op（門が恒真でないことの対）", () => {
  const gpu = noFeatureGpu();
  assertGpuFeaturesGranted(undefined, gpu, WHERE);
  assertGpuFeaturesGranted({}, gpu, WHERE);
  assertGpuFeaturesGranted({ shaderF16: false }, gpu, WHERE);
});

// ---- 実効 SessionOptions から導く要求（上書き口を持つ家族の席）--------------------

Deno.test("sessionGpuFeatures: f16 計算を実効設定が選べば shader-f16 を要求に足す", () => {
  // 明示の `linearCompute: "f16"` を quant 宣言（shader-f16 を要求しない）に重ねた形。足さないと
  // 自前で取る device が feature を持たず、重みを全部上げた後の Session 構築で落ちる。
  assertEquals(sessionGpuFeatures(undefined, { linearCompute: "f16" }), { shaderF16: true });
  assertEquals(sessionGpuFeatures({ shaderF16: false }, { attentionCompute: "f16" }), {
    shaderF16: true,
  });
  // 共有 GPU の経路では、足した要求がそのまま admission 席の検査に掛かる（重みを取る前に落ちる）。
  assertThrows(
    () =>
      assertGpuFeaturesGranted(
        sessionGpuFeatures(undefined, { linearCompute: "f16" }),
        noFeatureGpu(),
        WHERE,
      ),
    Error,
    "shader-f16",
  );
});

Deno.test("sessionGpuFeatures: 要らない・宣言済みなら宣言をそのまま返す（要求を増やさない）", () => {
  // s16（`attentionScoreStorage: "f16"`）は shader-f16 に依らない格納形（ADR 0030 決定 1）。
  assertEquals(sessionGpuFeatures(undefined, {}), undefined);
  assertEquals(sessionGpuFeatures(undefined, { attentionScoreStorage: "f16" }), undefined);
  assertEquals(sessionGpuFeatures(undefined, { linearCompute: "a8" }), undefined);
  const declared = { shaderF16: true };
  assertEquals(sessionGpuFeatures(declared, { linearCompute: "f16" }) === declared, true);
});
