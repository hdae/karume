/**
 * 空き VRAM の**試し確保** — 「今この瞬間に `bytes` ぶんの確保が通るか」を、非致命な唯一の確保経路
 * （`createBuffer`）で 1 度だけ踏んで確かめる。
 *
 * ## なぜ試し確保か
 *
 * WebGPU が「余力切れでも device を失わない」と保証している確保は `createBuffer` / `createTexture`
 * の out-of-memory だけ（仕様: "If the allocation fails without side-effects, generate an
 * out-of-memory error"）。実装もその線で作られている — wgpu-core（Deno）は `create_buffer` の
 * OOM だけを非致命扱いにし、`queue.writeBuffer` が内部で作る staging・submit・bind group 生成の
 * OOM は device を無効化する（`handle_hal_error` → `lose()`）。Dawn（Chrome）も同じ規則
 * （`Device::HandleError` — OOM を errorScope へ通す呼び出しは createBuffer / createTexture /
 * createQuerySet だけ）。しかも Deno はその OOM を**普通の** `GPUOutOfMemoryError` として
 * errorScope へ届け、`device.lost` は次の呼び出しまで解決しない。つまり「OOM を踏んでから退避する」
 * 設計は、踏んだ OOM が staging 側だった時点で手遅れになる（docs/research/2026-09-27-h35-oom-device-lost.md）。
 * 大きな確保（重みのアップロード）を伴う段の前に、その量が入るかをここで先に見る。
 *
 * ## 何を測っているか
 *
 * wgpu の OOM は実際の確保失敗ではなく、`VK_EXT_memory_budget` の予算に対する事前チェック
 * （使用量 + 要求 ≥ 予算の 97% で OOM）なので、この試し確保は「device-local heap の予算残」を
 * 同じ検査で直接読む形になる。予算は他プロセスの使用量でも動く（Mesa は `使用量 + 空き × 0.9`）ので、
 * 結果は**その瞬間の事実**であって予約ではない。呼び手は後続の確保が別の理由で落ちる可能性を消せない
 * （最終門は従来どおり out-of-memory errorScope）。
 *
 * MUST: 確保するのは STORAGE（device-local heap の検査）。host-visible な種類（MAP_*）にすると
 * システム RAM 側の heap まで検査対象に入り、聞きたい heap の答えにならない。
 * MUST: 確保したバッファは即 destroy し、`onSubmittedWorkDone`（= device poll）で解放を確定させてから
 * 返る。wgpu の poll は解放（maintain）→ 99% 線の判定の順なので、この待ちは device を危険にしない。
 * `queue.submit([])` は出さない — submit 側の 99% 線判定は解放より先に走るため、天井近くでは
 * 試し確保ぶんの使用量を抱えたまま判定を踏む。
 * MUST NOT: 結果を「空き容量」として公開しない（二分探索で天井を測る用途は診断ツールに留める —
 * 常に動く値を API の顔にすると当て推量の温床になる: ADR 0070 決定 5）。
 */

import type { GpuContext } from "./context.ts";
import { RUNTIME_INTERNAL } from "./context.ts";
import {
  discardFailureScopes,
  GpuOutOfMemoryError,
  popFailureScopes,
  pushFailureScopes,
} from "./error-scope.ts";
import { BUFFER_USAGE } from "./webgpu-constants.ts";

/**
 * `bytes` を `maxBufferSize` 以下の**等分**に割る（各片は 4 バイト整列・最後の片も他と同じ大きさ ± 4）。
 *
 * 等分にするのは、割り当て器（gpu-allocator）がブロック（既定 256 MiB）未満の要求をブロックから
 * 切り出すため — 端数の小片を 1 本作ると、事前チェックが見た量より大きい実確保（新規ブロック）が
 * 起きうる。等分なら片はどれも `bytes / 片数` で、`bytes` が GiB 級なら全片が専用確保になる。
 *
 * NOTE: `export` は GPU 無しで割り方を縛るテストのため（`mod.ts` には出さない）。
 */
export const probePieces = (bytes: number, maxBufferSize: number): number[] => {
  if (!Number.isSafeInteger(bytes) || bytes < 0) {
    throw new RangeError(`試し確保のバイト数は 0 以上の安全な整数（受け取った値: ${bytes}）`);
  }
  if (!Number.isSafeInteger(maxBufferSize) || maxBufferSize < 4) {
    throw new RangeError(`maxBufferSize が 4 未満か整数でない（受け取った値: ${maxBufferSize}）`);
  }
  if (bytes === 0) return [];
  let count = Math.ceil(bytes / maxBufferSize);
  let each = Math.ceil(bytes / count / 4) * 4;
  // 4 バイト整列の切り上げで片が maxBufferSize を超える形（bytes が maxBufferSize の倍数の直上）は
  // 片を 1 本増やす。
  if (each > maxBufferSize) {
    count += 1;
    each = Math.ceil(bytes / count / 4) * 4;
  }
  return Array.from({ length: count }, () => each);
};

/**
 * `bytes` ぶんの確保が今通るかを試す（詳細はモジュール冒頭）。
 *
 * - `true`: 確保は通り、全部 destroy して解放まで待った（次の確保に同じ量を期待してよい — 予約ではない）
 * - `false`: out-of-memory（device は生きている — 呼び手はここで退避なり縮退なりを決める）
 * - throw: validation（バグ）/ `GpuDeviceLostError`（既に無効な device — 解放待ちの中で表面化する）
 *
 * MUST: push から pop の**発行**までを 1 つの同期区間に保つ（この間に await を挟むと、他所の errorScope
 * と LIFO が交錯する — `gpu/context.ts` 冒頭「errorScope 区間の不変条件」）。
 */
export const fitsHeadroom = async (gpu: GpuContext, bytes: number): Promise<boolean> => {
  const pieces = probePieces(bytes, gpu.limits.maxBufferSize);
  if (pieces.length === 0) return true;
  const device = gpu.device;
  const probes: GPUBuffer[] = [];
  pushFailureScopes(device);
  try {
    for (const size of pieces) {
      probes.push(
        device.createBuffer({ label: "karume-headroom-probe", size, usage: BUFFER_USAGE.STORAGE }),
      );
    }
  } catch (cause) {
    // 同期例外（実装が仕様外で投げる場合）でも 2 本のスコープを積み残さない。
    await discardFailureScopes(device);
    for (const probe of probes) probe.destroy();
    throw cause;
  }
  const failure = await popFailureScopes(device, "空きの試し確保");
  for (const probe of probes) probe.destroy();
  // 解放の確定（poll）。無効化済みの device ならここで GpuDeviceLostError になる。
  await gpu[RUNTIME_INTERNAL].raceDeviceLost(
    device.queue.onSubmittedWorkDone(),
    "空きの試し確保の後始末",
  );
  if (failure === undefined) return true;
  if (failure instanceof GpuOutOfMemoryError) return false;
  throw failure;
};
