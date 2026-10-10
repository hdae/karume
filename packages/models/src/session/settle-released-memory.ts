/**
 * 破棄したバッファの解放を待つ道具（**パイプライン非依存の共通処理** — 段を畳んだ直後に次の段が
 * 大きな確保をする家族が使う）。
 *
 * MUST: barrel には出さない（同居する `options.ts` / `dispose-steps.ts` と同じ理由 — 利用者が触る面
 * ではない内部機構）。
 */

import type { GpuContext } from "@karume/runtime";

/**
 * 空の `queue.submit([])` を 1 本出してから `onSubmittedWorkDone` を待つ（device 消失とは競わせる）。
 *
 * WHY: Intel / wgpu（B570）は `destroy()` の解放が次の device poll まで遅れ、畳んだ直後に確保し直すと
 * OOM を踏みうる（docs/known-issues.md「Intel Arc B570」節）。素の WebGPU の probe では、この待ちだけで
 * 確保し直しが通った（docs/research/2026-09-26-anima-residency-bench.md）。固定の sleep は足さない。
 * 空の submit は、wgpu の保留中の destroy を流すのと、致命的な OOM で無効化された device の消失を
 * ここで表面化させる（Deno は `device.lost` を有効性を検査する呼び出しで初めて解決する）ためのもの。
 *
 * MUST: device 消失と競わせる — 消失後の `onSubmittedWorkDone` が解決しない実装がありうる（runtime の
 * `raceDeviceLost` の doc）。消失したら待たずに戻り、次の段が消失の例外で fail loudly になる。
 */
export const settleReleasedMemory = async (gpu: GpuContext): Promise<void> => {
  let unsubscribe: () => void = () => {};
  const lost = new Promise<void>((resolve) => {
    unsubscribe = gpu.onLost(() => resolve());
  });
  try {
    gpu.device.queue.submit([]);
    await Promise.race([gpu.device.queue.onSubmittedWorkDone(), lost]);
  } finally {
    unsubscribe();
  }
};
