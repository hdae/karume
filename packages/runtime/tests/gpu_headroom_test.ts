/**
 * 空き VRAM の試し確保（`fitsHeadroom` — `src/gpu/headroom.ts`）。
 *
 * 実 GPU で縛るのは 3 つ: ①入る量は true ②入らない量は false で **device は生きたまま**（続けて小さい
 * 確保が通り、`device.lost` は解決しない）③true の後に同じ量を本当に確保できる（試し確保の解放が
 * 返る前に戻っていない）。②は「createBuffer の OOM は非致命」という WebGPU 仕様の保証そのものを、
 * この機の実装が守ることの確認でもある（wgpu は writeBuffer 側の OOM で device を失う —
 * docs/research/2026-09-27-h35-oom-device-lost.md）。
 */

import { assert, assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { acquireGpu } from "../src/gpu/device.ts";
import { fitsHeadroom, probePieces } from "../src/gpu/headroom.ts";
import { BUFFER_USAGE } from "../src/gpu/webgpu-constants.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const MIB = 1024 ** 2;

describe("probePieces（試し確保の割り方）", () => {
  it("0 バイトは片を作らない", () => {
    assertEquals(probePieces(0, 1024), []);
  });
  it("maxBufferSize 以下は 1 片で、4 バイトに切り上げる", () => {
    assertEquals(probePieces(7, 1024), [8]);
    assertEquals(probePieces(1024, 1024), [1024]);
  });
  it("maxBufferSize を超える量は等分し、どの片も maxBufferSize 以下", () => {
    const pieces = probePieces(5 * 1024 + 1, 1024);
    assertEquals(pieces.length, 6);
    for (const piece of pieces) {
      assert(piece <= 1024 && piece % 4 === 0, `piece ${piece}`);
    }
    assert(pieces.reduce((sum, piece) => sum + piece, 0) >= 5 * 1024 + 1);
  });
  it("maxBufferSize が 4 の倍数でなく切り上げが上限を越える形は片を 1 本増やす", () => {
    // Deno の maxBufferSize は 2^31 - 1 でこの形になる: 1 片 1023 → 4 整列で 1024 > 1023。
    const pieces = probePieces(1023, 1023);
    assertEquals(pieces, [512, 512]);
    const wide = probePieces(2 ** 31 - 1, 2 ** 31 - 1);
    assertEquals(wide, [2 ** 30, 2 ** 30]);
  });
  it("負数・非整数は RangeError", () => {
    assertThrows(() => probePieces(-4, 1024), RangeError);
    assertThrows(() => probePieces(1.5, 1024), RangeError);
    assertThrows(() => probePieces(4, 2), RangeError);
  });
});

describe("fitsHeadroom（実 GPU）", () => {
  it("小さい量は入り、天井を越える量は device を生かしたまま false", {
    ignore: !GPU_AVAILABLE,
  }, async () => {
    const gpu = await acquireGpu();
    try {
      let lost = false;
      gpu.onLost(() => {
        lost = true;
      });
      assertEquals(await fitsHeadroom(gpu, 4), true);
      // 1 TiB — どの device でも入らない（maxBufferSize ごとの片に割って順に確保し、途中で OOM する）。
      assertEquals(await fitsHeadroom(gpu, 2 ** 40), false);
      // 非致命の確認: 続けて小さい量が入り、本当の確保 + 書き込み + フェンスも通る。
      assertEquals(await fitsHeadroom(gpu, 4), true);
      gpu.device.pushErrorScope("out-of-memory");
      const buffer = gpu.device.createBuffer({
        size: 16 * MIB,
        usage: BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_DST,
      });
      gpu.device.queue.writeBuffer(buffer, 0, new Uint8Array(16 * MIB));
      assertEquals(await gpu.device.popErrorScope(), null);
      await gpu.device.queue.onSubmittedWorkDone();
      buffer.destroy();
      assertEquals(lost, false);
      assertEquals(gpu.lost, undefined);
    } finally {
      gpu.destroy();
    }
  });

  it("true の後は同じ量を本当に確保できる（解放を待ってから返る）", {
    ignore: !GPU_AVAILABLE,
  }, async () => {
    const gpu = await acquireGpu();
    try {
      const bytes = 512 * MIB;
      if (!(await fitsHeadroom(gpu, bytes))) {
        // 512 MiB も入らない機は前提が崩れている（テストの対象外 — 明示して落とす）。
        throw new Error("512 MiB の試し確保が通らない device では検証できない");
      }
      const pieces = probePieces(bytes, gpu.limits.maxBufferSize);
      gpu.device.pushErrorScope("out-of-memory");
      const buffers = pieces.map((size) =>
        gpu.device.createBuffer({ size, usage: BUFFER_USAGE.STORAGE })
      );
      assertEquals(await gpu.device.popErrorScope(), null);
      for (const buffer of buffers) buffer.destroy();
      await gpu.device.queue.onSubmittedWorkDone();
    } finally {
      gpu.destroy();
    }
  });
});
