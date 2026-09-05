import { assertEquals, assertRejects, assertStrictEquals, assertThrows } from "@std/assert";
import { ArenaError, RunArena, STORAGE_USAGE, toSizeClass } from "../src/gpu/arena.ts";

/** RunArena が触る面だけを持つフェイク。DOM 型全体は再現しないため cast で渡す。 */
type FakeGpu = {
  readonly device: GPUDevice;
  readonly created: { readonly size: number; readonly usage: number }[];
  destroyedCount(): number;
};

const createFakeGpu = (): FakeGpu => {
  const created: { size: number; usage: number }[] = [];
  let destroyedCount = 0;
  const device = {
    createBuffer: (descriptor: { readonly size: number; readonly usage: number }) => {
      created.push({ size: descriptor.size, usage: descriptor.usage });
      return {
        size: descriptor.size,
        usage: descriptor.usage,
        destroy: (): void => {
          destroyedCount += 1;
        },
      };
    },
  };
  return {
    device: device as unknown as GPUDevice,
    created,
    destroyedCount: () => destroyedCount,
  };
};

const noopFlush = (): Promise<void> => Promise.resolve();

/**
 * 非安全整数のバイト数は切り上げが端数を落とすため、サイズクラスにできない。
 * `Number.isInteger` は 2^53 以上でも真になるので、そこを通すと「要求より小さいバッファを
 * 切り上げた顔で配る」形になる（`toSizeClass` の MUST が例外なしに破れる）。
 */
Deno.test("toSizeClass は安全整数を超えるバイト数を拒否する（切り上げで丸めない）", () => {
  assertThrows(() => toSizeClass(2 ** 53), ArenaError, "0 以上の安全整数");
  assertThrows(() => toSizeClass(2 ** 53 + 4), ArenaError, "0 以上の安全整数");
  assertThrows(() => toSizeClass(-4), ArenaError, "0 以上の安全整数");
  assertThrows(() => toSizeClass(1.5), ArenaError, "0 以上の安全整数");
  // 対照: 安全整数の上限直下は 4 バイト整列へ切り上げて通る。
  assertEquals(toSizeClass(Number.MAX_SAFE_INTEGER - 6), 2 ** 53 - 4);
  assertEquals(toSizeClass(0), 4);
});

Deno.test("RunArena.allocRegion は計画の領域を STORAGE_USAGE で確保し、破棄まで所有する", async () => {
  const gpu = createFakeGpu();
  const arena = new RunArena(gpu.device, noopFlush);

  const region = arena.allocRegion(1000);
  assertEquals(region.size, 1000, "4 バイト整列済みの実サイズちょうどで確保する");
  assertEquals(region.usage, STORAGE_USAGE, "中間バッファの usage は 1 定数");
  const odd = arena.allocRegion(9);
  assertEquals(odd.size, 12);
  assertEquals(arena.stats.allocCount, 2);
  assertEquals(arena.stats.allocatedBytes, 1012);
  // 計画から写す 3 欄は recordTransients が載せるまで 0（領域の確保では動かない）。
  assertEquals(arena.stats.reuseCount, 0);
  assertEquals(arena.stats.peakTransientBytes, 0);

  await arena.destroy();
  assertEquals(gpu.destroyedCount(), 2, "領域の破棄はアリーナが持つ");
});

Deno.test("RunArena.recordTransients は計画の診断 3 欄を stats へ写す（確保の実数は動かない）", () => {
  const gpu = createFakeGpu();
  const arena = new RunArena(gpu.device, noopFlush);
  arena.allocRegion(64);
  arena.recordTransients({ reuseCount: 3, transientBytes: 16, peakTransientBytes: 48 });
  assertEquals(arena.stats, {
    allocCount: 1,
    allocatedBytes: 64,
    reuseCount: 3,
    transientBytes: 16,
    peakTransientBytes: 48,
  });
});

Deno.test("RunArena.allocHostRead は staging を領域外で確保し、破棄まで所有する", async () => {
  const gpu = createFakeGpu();
  const arena = new RunArena(gpu.device, noopFlush);

  const staging = arena.allocHostRead(9);
  assertEquals(staging.size, 12, "4 バイト整列した実サイズちょうどで確保する");
  assertEquals(
    staging.usage,
    GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    "MAP_READ は STORAGE と併用できない",
  );
  assertEquals(arena.stats.allocatedBytes, 12);

  await arena.destroy();
  assertEquals(gpu.destroyedCount(), 1, "staging の破棄もアリーナが持つ");
});

Deno.test("RunArena.allocHostWritten は COPY_DST の無い usage を拒否する", () => {
  const gpu = createFakeGpu();
  const arena = new RunArena(gpu.device, noopFlush);

  const uniform = arena.allocHostWritten(64, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  assertEquals(uniform.usage, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  assertEquals(arena.stats.allocatedBytes, 64);

  assertThrows(
    () => arena.allocHostWritten(64, GPUBufferUsage.UNIFORM),
    ArenaError,
    "COPY_DST",
  );
});

Deno.test("RunArena.destroy は flush 完了後にのみバッファを破棄する", async () => {
  const gpu = createFakeGpu();
  let releaseFlush: () => void = () => {};
  const flushed = new Promise<void>((resolve) => {
    releaseFlush = resolve;
  });
  const arena = new RunArena(gpu.device, () => flushed);
  arena.allocRegion(32);
  arena.allocHostWritten(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

  const destroying = arena.destroy();
  assertEquals(gpu.destroyedCount(), 0, "flush 未完了の間は 1 つも破棄しない");
  assertThrows(
    () => arena.allocRegion(32),
    ArenaError,
    "破棄済み",
    "flush 待ちの間に確保されたバッファは誰にも破棄されない",
  );

  // 2 度目の destroy も同じ完了を待つ（早すぎる「破棄済み」応答を返さない）
  const second = arena.destroy();
  releaseFlush();
  await Promise.all([destroying, second]);
  assertEquals(gpu.destroyedCount(), 2);

  assertThrows(() => arena.allocRegion(32), ArenaError, "破棄済み");
});

Deno.test("RunArena.destroy は flush が失敗してもバッファを破棄し、その失敗を伝える", async () => {
  const gpu = createFakeGpu();
  // 実運用の主因は device 消失（flush が GpuDeviceLostError で落ちる）
  const lost = new Error("flush 中に device が失われた");
  const arena = new RunArena(gpu.device, () => Promise.reject(lost));
  arena.allocRegion(32);
  arena.allocHostWritten(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

  const caught = await assertRejects(() => arena.destroy());
  assertStrictEquals(caught, lost, "flush の失敗は握り潰さない");
  assertEquals(gpu.destroyedCount(), 2, "flush が失敗しても所有バッファは 1 本残らず破棄する");

  // 破棄済みの状態は確定している（後始末が済んでいるので再確保は受け付けない）
  assertThrows(() => arena.allocRegion(32), ArenaError, "破棄済み");
  const again = await assertRejects(() => arena.destroy());
  assertStrictEquals(again, lost);
  assertEquals(gpu.destroyedCount(), 2, "2 度目の destroy が二重破棄を起こさない");
});
