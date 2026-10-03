/**
 * 破棄したバッファの解放を待つ（実 GPU の e2e の後始末 — 空の `submit` → `onSubmittedWorkDone`・device
 * 消失とは競わせる）。
 *
 * WHY: B570（Linux xe / wgpu）は `destroy()` の解放が次の device poll まで遅れる（docs/known-issues.md
 * 「Intel Arc B570」節）。2026-10-02 の実測（素の WebGPU）: 7 GiB を破棄して同じ device で取り直すと OOM で、
 * この待ちの後なら通る。待たずに device を捨てると別の device から 7 GiB が OOM（解放されないまま予算を
 * 食い続ける）、待ってから捨てると通る。なので GB 級の確保を持つ e2e は、run の間・Session の間と、device を
 * 捨てる前（後続のテストの予算を残すため）に通す。
 */

import type { GpuContext } from "@karume/runtime";

export const settleReleases = async (gpu: GpuContext): Promise<void> => {
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
