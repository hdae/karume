// Host-side tests of `src/session/gpu-features.ts`: the GPU prerequisites a `Quant` declares
// (`gpuFeatures` and `requiredLimits`), mapped to `acquireGpu` requests and checked against a GPU.
// No adapter is needed: the feature checks wrap a fake device in the real `GpuContext`, and the
// limits checks are pure functions.

import { test } from "vite-plus/test";
import { assertEquals, assertStringIncludes, assertThrows } from "@std/assert";
import type { GpuFeaturesSpec, RequiredLimitsSpec } from "@karume/hub";
import type { RequiredLimits } from "@karume/runtime";
import { fakeDevice, fakeGpuContext } from "../../runtime/tests/helpers/fake-gpu.ts";
import {
  assertGpuFeaturesGranted,
  assertRequiredLimitsSatisfied,
  sessionGpuFeatures,
  toAcquireGpuOptions,
} from "../src/session/gpu-features.ts";

// ==== gpuFeatures ====================================================================
//
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

test("toAcquireGpuOptions: 宣言された feature だけを要求する", () => {
  // 宣言が無い / 明示的に false の配布形で feature を要求しにいくと、持たないアダプタでは
  // `acquireGpu` 自体が落ちる（= 動くはずの環境で動かなくなる）。
  assertEquals(toAcquireGpuOptions(undefined), {});
  assertEquals(toAcquireGpuOptions({}), {});
  assertEquals(toAcquireGpuOptions({ shaderF16: false }), {});
  assertEquals(toAcquireGpuOptions({ shaderF16: true }), { shaderF16: true });
});

test("toAcquireGpuOptions: GpuFeaturesSpec の全キーを写す（キー追加の取り残しを検出）", () => {
  // 写像の網羅は `src/session/gpu-features.ts` の `FEATURES`（`Required<GpuFeaturesSpec>` の
  // 網羅表）が型で固定しており、キーが増えれば**コンパイルエラー**になる。ここはその型門が
  // 生きていることを実行時からも見る対（型を緩めた改変は型検査を通ってしまうため）—
  // 全キーを true で埋めた spec を渡し、出てくるキー集合が入力と一致することを確かめる。
  const full: Required<GpuFeaturesSpec> = { shaderF16: true };
  const requested = toAcquireGpuOptions(full) as Record<string, unknown>;
  assertEquals(Object.keys(requested).sort(), Object.keys(full).sort());
});

// ---- 取得済み GpuContext での検査（共有 GPU 経路の唯一の門）--------------------

const FEATURES_WHERE = "TestPipeline: quant 'f16'";

/** shader-f16 を持たない device（`fakeGpuContext` は features を空 `Set` で作る）。 */
const noFeatureGpu = () => fakeGpuContext(fakeDevice());

test("assertGpuFeaturesGranted: 要求された feature が無効な共有 GPU は名指しで落ちる", () => {
  // 共有 GPU（`options.gpu`）を渡された経路では device 生成後なので feature を取り直せない
  // （ADR 0028）。ここを通すと Session 構築まで進んでから落ちるか、黙って別経路へ縮退する。
  const error = assertThrows(
    () => assertGpuFeaturesGranted({ shaderF16: true }, noFeatureGpu(), FEATURES_WHERE),
    Error,
  );
  // 診断の 3 点セット: どの席か / どの feature か / どう取り直すか。
  assertStringIncludes(error.message, FEATURES_WHERE);
  assertStringIncludes(error.message, "shader-f16");
  assertStringIncludes(error.message, "acquireGpu({ shaderF16: true })");
});

test("assertGpuFeaturesGranted: 要求していない配布形は no-op（門が恒真でないことの対）", () => {
  const gpu = noFeatureGpu();
  assertGpuFeaturesGranted(undefined, gpu, FEATURES_WHERE);
  assertGpuFeaturesGranted({}, gpu, FEATURES_WHERE);
  assertGpuFeaturesGranted({ shaderF16: false }, gpu, FEATURES_WHERE);
});

// ---- 実効 SessionOptions から導く要求（上書き口を持つ家族の席）--------------------

test("sessionGpuFeatures: f16 計算を実効設定が選べば shader-f16 を要求に足す", () => {
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
        FEATURES_WHERE,
      ),
    Error,
    "shader-f16",
  );
});

test("sessionGpuFeatures: 要らない・宣言済みなら宣言をそのまま返す（要求を増やさない）", () => {
  // s16（`attentionScoreStorage: "f16"`）は shader-f16 に依らない格納形（ADR 0030 決定 1）。
  assertEquals(sessionGpuFeatures(undefined, {}), undefined);
  assertEquals(sessionGpuFeatures(undefined, { attentionScoreStorage: "f16" }), undefined);
  assertEquals(sessionGpuFeatures(undefined, { linearCompute: "a8" }), undefined);
  const declared = { shaderF16: true };
  assertEquals(sessionGpuFeatures(declared, { linearCompute: "f16" }) === declared, true);
});

// ==== requiredLimits =================================================================
//
// manifest の `requiredLimits`（quant が宣言する device limit の最小値 — ADR 0038 §7）と GPU
// 側の limits を突き合わせる検査。ADR 0089 決定 5 の**重みを 1 バイトも落とす前**の門なので、
// 見落とし（宣言したのに誰も見ない = 数 GiB を落としてから確保で落ちる）も過剰（書かれて
// いない limit まで見る = 動くはずの環境を誤拒否する）も、どちらも実害が出る側にある。
//
// 突き合わせ相手を用意する面（`assertRequiredLimitsBeforeDownload` — 共有 GPU の limits か
// アダプタ実測値か）は GPU / ロード経路が要るので `container_loading_test.ts` の⑧と
// `container_loading_test.ts` の⑨が縛る。

const LIMITS_WHERE = "TestPipeline: quant 'i4'";

/**
 * 突き合わせ相手の実測値。`maxBufferSize` 以外を 0 にしてあるのは、**spec に書かれていない
 * limit を見ていない**ことが「通る」で示せるようにするため。
 */
const LIMITS: RequiredLimits = {
  maxBufferSize: 268435456,
  maxStorageBufferBindingSize: 0,
  maxUniformBufferBindingSize: 0,
  maxStorageBuffersPerShaderStage: 0,
  maxUniformBuffersPerShaderStage: 0,
  maxComputeWorkgroupStorageSize: 0,
  maxComputeInvocationsPerWorkgroup: 0,
  maxComputeWorkgroupSizeX: 0,
  maxComputeWorkgroupSizeY: 0,
  maxComputeWorkgroupSizeZ: 0,
  maxComputeWorkgroupsPerDimension: 0,
};

test("assertRequiredLimitsSatisfied: 要求と実測が等しい配布形は通る", () => {
  // 境界は「不足 = 実測 < 要求」— ここを `<=` で書くと、要求ちょうどの device（exporter が
  // 焼く値は現物の最大テンソルそのもの）が全て誤拒否される。
  const spec: RequiredLimitsSpec = { maxBufferSize: LIMITS.maxBufferSize };
  assertRequiredLimitsSatisfied(spec, LIMITS, LIMITS_WHERE);
});

test("assertRequiredLimitsSatisfied: 1 でも下回る配布形は落ちる", () => {
  const spec: RequiredLimitsSpec = { maxBufferSize: LIMITS.maxBufferSize + 1 };
  assertThrows(
    () => assertRequiredLimitsSatisfied(spec, LIMITS, LIMITS_WHERE),
    Error,
    "maxBufferSize",
  );
});

test("assertRequiredLimitsSatisfied: spec に書かれていない limit は見ない（部分写像）", () => {
  // 実測は 10 本が 0 だが、配布形が主張しているのは `maxBufferSize` だけ。書かれていない
  // limit まで見ると「欄なし = WebGPU 保証既定で動く」（ADR 0089 決定 3）の意味論が壊れる。
  const spec: RequiredLimitsSpec = { maxBufferSize: 1 };
  assertRequiredLimitsSatisfied(spec, LIMITS, LIMITS_WHERE);
});

test("assertRequiredLimitsSatisfied: 宣言の無い配布形は no-op", () => {
  assertRequiredLimitsSatisfied(undefined, LIMITS, LIMITS_WHERE);
});

test("assertRequiredLimitsSatisfied: 診断に where・limit 名・要求値・実測値を全部載せる", () => {
  // 落ちた利用者が次に打つ手（別の quant / 別の機械）を決めるのに、4 つとも要る。不足は
  // **全件を 1 回で**出す（1 本ずつ潰させると、環境が足りないことの全体像が出ない）。
  const spec: RequiredLimitsSpec = {
    maxBufferSize: 1073741824,
    maxStorageBufferBindingSize: 134217728,
  };
  const error = assertThrows(
    () => assertRequiredLimitsSatisfied(spec, LIMITS, LIMITS_WHERE),
    Error,
  );
  for (const fragment of [
    LIMITS_WHERE,
    "maxBufferSize",
    "1073741824",
    "268435456",
    "maxStorageBufferBindingSize",
    "134217728",
    "0",
  ]) {
    if (!error.message.includes(fragment)) {
      throw new Error(`診断に '${fragment}' が無い: ${error.message}`);
    }
  }
});

test("assertRequiredLimitsSatisfied: runtime が持たない limit 名は語彙のずれとして落ちる", () => {
  // hub が受理した名前を runtime が持たない形（防御の二段目）。黙って読み飛ばすと「宣言された
  // limit が誰にも見られない」= 静かな頭打ちに戻る。`limit_vocabulary_test.ts` が緑である限り
  // 構造的には到達しないが、その門が外れたときにここが受け止める。
  //
  // 未知キーは index signature 経由で組む（`as unknown as` で型を殺すと、この検査が
  // 「型が緩んだこと」ではなく「実行時に何を言うか」を見ていることが読み取れなくなる）。
  const spec: Record<string, number> = { maxBufferSize: 1, karumeUnknownLimit: 1 };
  const error = assertThrows(
    () => assertRequiredLimitsSatisfied(spec, LIMITS, LIMITS_WHERE),
    Error,
  );
  for (const fragment of [LIMITS_WHERE, "karumeUnknownLimit", "tests/limit_vocabulary_test.ts"]) {
    if (!error.message.includes(fragment)) {
      throw new Error(`診断に '${fragment}' が無い: ${error.message}`);
    }
  }
});
