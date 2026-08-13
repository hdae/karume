// フェンス数の機械検査（`queue.onSubmittedWorkDone` の呼び出し回数）。
//
// 「待ちを減らす」最適化は値が正しいままでも黙って退化する（内部で flush へ退避しても例外も
// 警告も出ない）ので、フェンスが実際に減っていることを数える門が要る。実装内部に観測点を
// 足さずに済むよう、テスト側でキューの面をラップする。

import type { GpuContext } from "../../src/gpu/device.ts";

export type FenceCounter = {
  /** ラップしてからの `onSubmittedWorkDone` の呼び出し回数。 */
  readonly count: () => number;
  /** 元の面へ戻す（MUST: finally で必ず呼ぶ — 他のテストへ計数が漏れる）。 */
  restore: () => void;
};

export const countFences = (gpu: GpuContext): FenceCounter => {
  const queue = gpu.device.queue;
  const original = queue.onSubmittedWorkDone.bind(queue);
  let fences = 0;
  Object.defineProperty(queue, "onSubmittedWorkDone", {
    configurable: true,
    writable: true,
    value: () => {
      fences += 1;
      return original();
    },
  });
  return {
    count: () => fences,
    restore: () => {
      Object.defineProperty(queue, "onSubmittedWorkDone", {
        configurable: true,
        writable: true,
        value: original,
      });
    },
  };
};
