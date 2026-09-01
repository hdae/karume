// manifest の `requiredLimits`（quant が宣言する device limit の最小値 — ADR 0038 §7）と GPU
// 側の limits を突き合わせる検査。ADR 0089 決定 5 の**重みを 1 バイトも落とす前**の門なので、
// 見落とし（宣言したのに誰も見ない = 数 GiB を落としてから確保で落ちる）も過剰（書かれて
// いない limit まで見る = 動くはずの環境を誤拒否する）も、どちらも実害が出る側にある。
//
// 突き合わせ相手を用意する面（`assertRequiredLimitsBeforeDownload` — 共有 GPU の limits か
// アダプタ実測値か）は GPU / ロード経路が要るので `shard_loading_test.ts` の⑧⑨が縛る。

import { assertThrows } from "@std/assert";
import type { RequiredLimitsSpec } from "@karume/hub";
import type { RequiredLimits } from "@karume/runtime";
import { assertRequiredLimitsSatisfied } from "../src/session/gpu-features.ts";

const WHERE = "TestPipeline: quant 'i4'";

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

Deno.test("assertRequiredLimitsSatisfied: 要求と実測が等しい配布形は通る", () => {
  // 境界は「不足 = 実測 < 要求」— ここを `<=` で書くと、要求ちょうどの device（exporter が
  // 焼く値は現物の最大テンソルそのもの）が全て誤拒否される。
  const spec: RequiredLimitsSpec = { maxBufferSize: LIMITS.maxBufferSize };
  assertRequiredLimitsSatisfied(spec, LIMITS, WHERE);
});

Deno.test("assertRequiredLimitsSatisfied: 1 でも下回る配布形は落ちる", () => {
  const spec: RequiredLimitsSpec = { maxBufferSize: LIMITS.maxBufferSize + 1 };
  assertThrows(
    () => assertRequiredLimitsSatisfied(spec, LIMITS, WHERE),
    Error,
    "maxBufferSize",
  );
});

Deno.test("assertRequiredLimitsSatisfied: spec に書かれていない limit は見ない（部分写像）", () => {
  // 実測は 10 本が 0 だが、配布形が主張しているのは `maxBufferSize` だけ。書かれていない
  // limit まで見ると「欄なし = WebGPU 保証既定で動く」（ADR 0089 決定 3）の意味論が壊れる。
  const spec: RequiredLimitsSpec = { maxBufferSize: 1 };
  assertRequiredLimitsSatisfied(spec, LIMITS, WHERE);
});

Deno.test("assertRequiredLimitsSatisfied: 宣言の無い配布形は no-op", () => {
  assertRequiredLimitsSatisfied(undefined, LIMITS, WHERE);
});

Deno.test("assertRequiredLimitsSatisfied: 診断に where・limit 名・要求値・実測値を全部載せる", () => {
  // 落ちた利用者が次に打つ手（別の quant / 別の機械）を決めるのに、4 つとも要る。不足は
  // **全件を 1 回で**出す（1 本ずつ潰させると、環境が足りないことの全体像が出ない）。
  const spec: RequiredLimitsSpec = {
    maxBufferSize: 1073741824,
    maxStorageBufferBindingSize: 134217728,
  };
  const error = assertThrows(
    () => assertRequiredLimitsSatisfied(spec, LIMITS, WHERE),
    Error,
  );
  for (
    const fragment of [
      WHERE,
      "maxBufferSize",
      "1073741824",
      "268435456",
      "maxStorageBufferBindingSize",
      "134217728",
      "0",
    ]
  ) {
    if (!error.message.includes(fragment)) {
      throw new Error(`診断に '${fragment}' が無い: ${error.message}`);
    }
  }
});
