import {
  assertEquals,
  assertNotStrictEquals,
  assertRejects,
  assertStrictEquals,
  assertThrows,
} from "@std/assert";
import { ArenaError, RunArena } from "../src/gpu/arena.ts";

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

Deno.test("RunArena は最終消費で解放したバッファを同一サイズクラスで再利用する", () => {
  const gpu = createFakeGpu();
  const arena = new RunArena(gpu.device, noopFlush);

  const first = arena.allocStorage(64);
  arena.retain(first, 1);
  arena.release(first); // 消費 1 回ぶん
  arena.release(first); // 定義ぶん（定義したノードの境界）
  const second = arena.allocStorage(64);

  assertStrictEquals(second, first);
  assertEquals(arena.stats.allocCount, 1);
  assertEquals(arena.stats.reuseCount, 1);
  assertEquals(gpu.created.length, 1);
});

Deno.test("RunArena はサイズクラスが違えば再利用しない（要求より大きいバッファを配らない）", () => {
  const gpu = createFakeGpu();
  const arena = new RunArena(gpu.device, noopFlush);

  const small = arena.allocStorage(1);
  assertEquals(small.size, 4, "4 バイト整列した実サイズちょうどで確保する");
  arena.retain(small, 1);
  arena.release(small);
  arena.release(small);

  const larger = arena.allocStorage(13);
  assertNotStrictEquals(larger, small);
  assertEquals(larger.size, 16);
  assertEquals(arena.stats.reuseCount, 0);
});

Deno.test("RunArena は消費者ゼロの値も定義ぶんの解放でプールへ返す", () => {
  const gpu = createFakeGpu();
  const arena = new RunArena(gpu.device, noopFlush);

  // 誰にも読まれない中間出力（グラフ出力でもない）。release が 1 度も来ないと居座る。
  const dead = arena.allocStorage(48);
  arena.retain(dead, 0);
  assertEquals(arena.stats.peakTransientBytes, 48);

  arena.release(dead);
  assertEquals(arena.stats.transientBytes, 0, "消費者ゼロでも生存バイト数が落ちる");
  assertStrictEquals(arena.allocStorage(48), dead, "再利用の対象に戻る");
  assertEquals(arena.stats.allocCount, 1);
});

Deno.test("RunArena.allocHostRead は staging をプール外で確保し、破棄まで所有する", async () => {
  const gpu = createFakeGpu();
  const arena = new RunArena(gpu.device, noopFlush);

  const staging = arena.allocHostRead(9);
  assertEquals(staging.size, 12, "4 バイト整列した実サイズちょうどで確保する");
  assertEquals(
    staging.usage,
    GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    "MAP_READ は STORAGE と併用できない",
  );
  assertEquals(arena.isReadable(staging), true, "readback の宛先そのもの");
  assertEquals(arena.stats.transientBytes, 0, "プール管理下には入らない");
  assertEquals(arena.stats.allocatedBytes, 12);

  await arena.destroy();
  assertEquals(gpu.destroyedCount(), 1, "staging の破棄もアリーナが持つ");
});

Deno.test("RunArena は writeBuffer 対象のバッファをプールに入れない", () => {
  const gpu = createFakeGpu();
  const arena = new RunArena(gpu.device, noopFlush);

  const uniform = arena.allocHostWritten(64, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  arena.retain(uniform, 1);
  arena.release(uniform);

  const storage = arena.allocStorage(64);
  assertNotStrictEquals(storage, uniform, "解放してもプールに戻らない");
  assertEquals(arena.stats.reuseCount, 0);
  assertEquals(arena.isReadable(uniform), true);

  assertThrows(
    () => arena.allocHostWritten(64, GPUBufferUsage.UNIFORM),
    ArenaError,
    "COPY_DST",
  );
});

Deno.test("RunArena は延べ消費回数ぶん解放されて初めてプールに返す", () => {
  const gpu = createFakeGpu();
  const arena = new RunArena(gpu.device, noopFlush);

  const shared = arena.allocStorage(32);
  arena.retain(shared, 2);

  arena.release(shared);
  assertNotStrictEquals(arena.allocStorage(32), shared, "消費が残る間は再利用されない");

  arena.release(shared);
  assertNotStrictEquals(arena.allocStorage(32), shared, "定義ぶんの 1 が残る間も再利用されない");

  arena.release(shared);
  assertStrictEquals(arena.allocStorage(32), shared);

  // 再確保後は retain し直すのが契約。計数が合わない解放は黙って進めず落とす。
  assertThrows(() => arena.release(shared), ArenaError, "参照カウントが負");
  assertThrows(() => arena.retain(shared, -1), ArenaError);
});

Deno.test("RunArena はピン留めしたグラフ出力をプールに返さず readback 可能に保つ", () => {
  const gpu = createFakeGpu();
  const arena = new RunArena(gpu.device, noopFlush);

  const output = arena.allocStorage(32);
  arena.retain(output, 1, { pinned: true });
  const intermediate = arena.allocStorage(64);
  arena.retain(intermediate, 1);

  assertEquals(arena.isReadable(output), true);
  assertEquals(arena.isReadable(intermediate), false, "中間値の readback は拒否する");

  arena.release(output);
  arena.release(output);
  arena.release(intermediate);
  arena.release(intermediate);

  assertNotStrictEquals(arena.allocStorage(32), output, "ピン留めは再利用対象にならない");
  assertStrictEquals(arena.allocStorage(64), intermediate);
});

// ADR 0011 の reshape は「1 実バッファ = 複数論理値」。アリーナは論理値を知らないので、
// 別名の寿命は retain の**加算**だけで成立していなければならない。
Deno.test("RunArena は別名（1 実バッファ = 複数論理値）の参照を加算して数える", () => {
  const gpu = createFakeGpu();
  const arena = new RunArena(gpu.device, noopFlush);

  // h = neg(x): 消費 1 回（reshape が読む）
  const buffer = arena.allocStorage(48);
  arena.retain(buffer, 1);
  arena.release(buffer); // h の定義ぶん（ノード境界）

  // r = reshape(h): 実体は同じバッファ。r 自身の消費 2 回ぶんを同じバッファに積む
  arena.retain(buffer, 2);
  arena.release(buffer); // reshape が h を消費した 1 回ぶん
  arena.release(buffer); // r の定義ぶん
  assertEquals(arena.stats.transientBytes, 48, "別名の消費が残る間はプールに返さない");

  arena.release(buffer);
  assertEquals(arena.stats.transientBytes, 48, "r の消費が 1 回残っている");
  arena.release(buffer);
  assertEquals(arena.stats.transientBytes, 0);
  assertStrictEquals(arena.allocStorage(48), buffer, "別名の最終消費でプールへ戻る");
  assertEquals(arena.stats.allocCount, 1, "別名は確保を増やさない");
});

// MUST: 別名は alloc 経路を通らないので、破棄（= discard-or-flush before destroy の対象）は
// 論理値の本数ではなく**実バッファの本数**でちょうど 1 回。
Deno.test("RunArena は別名を張っても実バッファを 1 度だけ破棄する", async () => {
  const gpu = createFakeGpu();
  const arena = new RunArena(gpu.device, noopFlush);

  const buffer = arena.allocStorage(32);
  arena.retain(buffer, 1);
  arena.retain(buffer, 0, { pinned: true }); // 別名がグラフ出力（readback のためピン留め）
  assertEquals(arena.isReadable(buffer), true, "別名がグラフ出力なら実体も読み戻せる");

  await arena.destroy();
  assertEquals(gpu.created.length, 1);
  assertEquals(gpu.destroyedCount(), 1, "論理値 2 本でも実バッファは 1 本ぶんだけ破棄する");
});

Deno.test("RunArena.assertDrained は未解放の参照を検出する", () => {
  const gpu = createFakeGpu();
  const arena = new RunArena(gpu.device, noopFlush);

  const leaked = arena.allocStorage(32);
  arena.retain(leaked, 1);
  assertThrows(() => arena.assertDrained(), ArenaError, "未解放の参照");

  arena.release(leaked);
  assertThrows(() => arena.assertDrained(), ArenaError, "未解放の参照", "定義ぶんが残っている");
  arena.release(leaked);
  arena.assertDrained();

  const pinned = arena.allocStorage(16);
  arena.retain(pinned, 1, { pinned: true });
  arena.assertDrained();
});

Deno.test("RunArena.destroy は flush 完了後にのみバッファを破棄する", async () => {
  const gpu = createFakeGpu();
  let releaseFlush: () => void = () => {};
  const flushed = new Promise<void>((resolve) => {
    releaseFlush = resolve;
  });
  const arena = new RunArena(gpu.device, () => flushed);
  arena.allocStorage(32);
  arena.allocHostWritten(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

  const destroying = arena.destroy();
  assertEquals(gpu.destroyedCount(), 0, "flush 未完了の間は 1 つも破棄しない");
  assertThrows(
    () => arena.allocStorage(32),
    ArenaError,
    "破棄済み",
    "flush 待ちの間に確保されたバッファは誰にも破棄されない",
  );

  // 2 度目の destroy も同じ完了を待つ（早すぎる「破棄済み」応答を返さない）
  const second = arena.destroy();
  releaseFlush();
  await Promise.all([destroying, second]);
  assertEquals(gpu.destroyedCount(), 2);

  assertThrows(() => arena.allocStorage(32), ArenaError, "破棄済み");
});

Deno.test("RunArena.destroy は flush が失敗してもバッファを破棄し、その失敗を伝える", async () => {
  const gpu = createFakeGpu();
  // 実運用の主因は device 消失（flush が GpuDeviceLostError で落ちる）
  const lost = new Error("flush 中に device が失われた");
  const arena = new RunArena(gpu.device, () => Promise.reject(lost));
  arena.allocStorage(32);
  arena.allocHostWritten(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

  const caught = await assertRejects(() => arena.destroy());
  assertStrictEquals(caught, lost, "flush の失敗は握り潰さない");
  assertEquals(gpu.destroyedCount(), 2, "flush が失敗しても所有バッファは 1 本残らず破棄する");
  assertEquals(arena.stats.transientBytes, 0, "プールと参照計数も片付いている");

  // 破棄済みの状態は確定している（後始末が済んでいるので再確保は受け付けない）
  assertThrows(() => arena.allocStorage(32), ArenaError, "破棄済み");
  const again = await assertRejects(() => arena.destroy());
  assertStrictEquals(again, lost);
  assertEquals(gpu.destroyedCount(), 2, "2 度目の destroy が二重破棄を起こさない");
});
