/**
 * 実 GPU テストの実行可否判定（models 版）。
 *
 * MUST: アダプタが無い環境は `Deno.test({ ignore })` で**明示 SKIP** する（テストを消して
 * 無音で緑にしない — ADR 0005）。判定はテスト登録時点で要るため、モジュール評価時に 1 回だけ
 * 行う。`requestAdapter()` が例外を投げる環境は「壊れた環境」なので握り潰さず伝播させる。
 *
 * NOTE: runtime 側の同名ヘルパ（`packages/runtime/tests/helpers/gpu.ts`）は import しない —
 * パッケージのテストが他パッケージのテスト内部に依存すると、向こうの門（`gpu_gate_test`）や
 * 環境変数の都合がこちらへ漏れる。流儀（判定 1 回・警告文・明示 SKIP）だけ合わせる。
 */

import { acquireGpu, type AcquireGpuOptions, type GpuContext } from "@karume/runtime";
import { collectDestroyedDevices } from "./collect-destroyed-devices.ts";

const detectAdapter = async (): Promise<boolean> => {
  const gpu: GPU | undefined = navigator.gpu;
  if (gpu === undefined) return false;
  return (await gpu.requestAdapter()) !== null;
};

export const GPU_AVAILABLE: boolean = await detectAdapter();

/**
 * テストの GPU の取得口: 破棄済みの device の回収を促してから `acquireGpu` する。
 *
 * NOTE: 対症療法（{@link collectDestroyedDevices} の注記 — Deno の `destroy()` が VRAM を返さない件）。
 * 今これを通すのは Wan のレーン（`test:models:wan` / `test:models:wan-ti2v`）の e2e だけで、他のテスト
 * と、パイプラインが内部で取る device（`gpu` を渡さない読み込み）は通らない。Wan の e2e の中でも、
 * `assertRunningAdapter`（runtime のテストの helpers/environment.ts — 走行に 1 度の検査用の取得）は
 * 通らない（`gc` は process 全体に効くので、それも次にここを通る取得の前にまとめて回収される）。
 */
export const acquireTestGpu = async (options?: AcquireGpuOptions): Promise<GpuContext> => {
  await collectDestroyedDevices();
  return await acquireGpu(options);
};

if (!GPU_AVAILABLE) {
  console.warn(
    "[karume] GPUAdapter が無いため models の実 GPU テストを SKIP する" +
      "（リリース判定は実 GPU 緑が必須 — ADR 0005）",
  );
}
