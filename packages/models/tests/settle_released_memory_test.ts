// 解放待ち（`src/session/settle-released-memory.ts` の `settleReleasedMemory`）の門 — 段を畳んだ
// 直後に次の段が大きな確保をする家族が共有する 1 本（利用者は anima の退避と Wan2.2 の I2V の
// encoder の段）。偽の GpuContext で縛る（GPU も資産も要らない）。
//
// WHY ここに置く: 共通層の検査は、利用者のどのレーンを回しても走るべきである。ファイル名に家族名を
// 含めないので `test:core` に入る（家族の名前を持つファイルに置くと、core と別系列のレーンでは
// 走らない — ADR 0005 追記のレーン規約）。

import { assert, assertEquals } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { settleReleasedMemory } from "../src/session/settle-released-memory.ts";
import { losableGpuContext } from "../../runtime/tests/helpers/fake-gpu.ts";

describe("settleReleasedMemory（退避の後の解放待ち）", () => {
  it("空の submit を出してから onSubmittedWorkDone を待ち、消失の購読を残さない", async () => {
    // 空の submit が「無効化された device の消失を表面化させる呼び出し」（settle-released-memory.ts の doc）。
    const calls: string[] = [];
    const { gpu } = losableGpuContext(undefined, {
      submit: (commandBuffers) => void calls.push(`submit:${commandBuffers.length}`),
      onSubmittedWorkDone: () => {
        calls.push("onSubmittedWorkDone");
        return Promise.resolve();
      },
    });
    const baseline = gpu.pendingLostListeners;
    await settleReleasedMemory(gpu);
    assertEquals(calls, ["submit:0", "onSubmittedWorkDone"]);
    assertEquals(gpu.pendingLostListeners, baseline, "消失の購読が積み残っている");
  });

  it("空の submit が消失を表面化させたら、onSubmittedWorkDone が解決しなくても待たずに戻る", async () => {
    // Deno は無効化された device の消失を次の有効性検査（ここでは submit）で初めて解決する。
    const { gpu, lose } = losableGpuContext(undefined, {
      submit: () => lose(),
      onSubmittedWorkDone: () => new Promise<void>(() => {}),
    });
    const baseline = gpu.pendingLostListeners;
    await settleReleasedMemory(gpu);
    assertEquals(gpu.pendingLostListeners, baseline);
  });

  it("onSubmittedWorkDone が解決しなくても、device が失われたら待たずに戻る", async () => {
    // 消失後の onSubmittedWorkDone が解決しない実装への備え（MUST）。競わせないとここで固まる。
    const { gpu, lose } = losableGpuContext(undefined, {
      onSubmittedWorkDone: () => new Promise<void>(() => {}),
    });
    const baseline = gpu.pendingLostListeners;
    const settled = settleReleasedMemory(gpu);
    lose();
    await settled;
    assertEquals(gpu.pendingLostListeners, baseline);
  });

  it("既に失われた device では待たずに戻る", async () => {
    const { gpu, lose } = losableGpuContext(undefined, {
      onSubmittedWorkDone: () => new Promise<void>(() => {}),
    });
    lose();
    // 消失の通知は device.lost の解決（マイクロタスク）越しに届く。
    await Promise.resolve();
    assert(gpu.lost !== undefined, "前提: 消失を観測済み");
    await settleReleasedMemory(gpu);
  });
});
