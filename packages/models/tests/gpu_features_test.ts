// manifest の `gpuFeatures`（device 生成時にしか要求できない feature）→ `acquireGpu` の要求への
// 写像。ADR 0038 §3 の綴りの契約で、抜けは**沈黙劣化**（配布形が要求した能力を持たない device で
// 実行が進む）になる。7 家族が同じ 1 本を使うので、門もここ 1 本に集約する（元は 7 家族が
// `quant.gpuFeatures?.shaderF16 === true` を 1 行ずつ独立に読んでいた — レビュー M1-V9）。
//
// 取得後の検査（`assertGpuFeaturesGranted`）は `GpuContext` の実体が要るので、実 GPU の
// e2e 側に委ねる（型としてしか公開されておらず、偽物を組むと検査対象が偽物になる）。

import { assertEquals } from "@std/assert";
import type { GpuFeaturesSpec } from "@karume/hub";
import { toAcquireGpuOptions } from "../src/session/gpu-features.ts";

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
