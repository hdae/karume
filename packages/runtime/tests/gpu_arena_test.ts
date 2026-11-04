import { test } from "vite-plus/test";
import { assertEquals, assertRejects, assertStrictEquals, assertThrows } from "@std/assert";
import { ArenaError, RunArena, STORAGE_USAGE, toSizeClass } from "../src/gpu/arena.ts";

/**
 * A fake holding only the surface RunArena touches. The full DOM types are not reproduced, so it is
 * passed with a cast.
 */
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
 * A byte count that is not a safe integer cannot become a size class, because the round-up drops
 * the fraction. `Number.isInteger` is true at 2^53 and above too, so letting it through would hand
 * out "a buffer smaller than requested, posing as rounded up" (breaking the MUST of `toSizeClass`
 * without an exception).
 */
test("toSizeClass rejects byte counts beyond the safe integers (no rounding by the round-up)", () => {
  assertThrows(() => toSizeClass(2 ** 53), ArenaError, "safe integer >= 0");
  assertThrows(() => toSizeClass(2 ** 53 + 4), ArenaError, "safe integer >= 0");
  assertThrows(() => toSizeClass(-4), ArenaError, "safe integer >= 0");
  assertThrows(() => toSizeClass(1.5), ArenaError, "safe integer >= 0");
  // Control: just below the safe-integer maximum rounds up to 4-byte alignment and passes.
  assertEquals(toSizeClass(Number.MAX_SAFE_INTEGER - 6), 2 ** 53 - 4);
  assertEquals(toSizeClass(0), 4);
});

test("RunArena.allocRegion allocates a planned region with STORAGE_USAGE and owns it until destroy", async () => {
  const gpu = createFakeGpu();
  const arena = new RunArena(gpu.device, noopFlush);

  const region = arena.allocRegion(1000);
  assertEquals(region.size, 1000, "allocates exactly the 4-byte-aligned actual size");
  assertEquals(region.usage, STORAGE_USAGE, "the usage of intermediate buffers is one constant");
  const odd = arena.allocRegion(9);
  assertEquals(odd.size, 12);
  assertEquals(arena.stats.allocCount, 2);
  assertEquals(arena.stats.allocatedBytes, 1012);
  // The 3 fields copied from the plan stay 0 until recordTransients sets them (allocating regions
  // does not move them).
  assertEquals(arena.stats.reuseCount, 0);
  assertEquals(arena.stats.peakTransientBytes, 0);

  await arena.destroy();
  assertEquals(gpu.destroyedCount(), 2, "destroying the regions is the arena's job");
});

test("RunArena.recordTransients copies the plan's 3 diagnostic fields into stats (the actual allocation counts do not move)", () => {
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

test("RunArena.allocHostRead allocates staging outside the regions and owns it until destroy", async () => {
  const gpu = createFakeGpu();
  const arena = new RunArena(gpu.device, noopFlush);

  const staging = arena.allocHostRead(9);
  assertEquals(staging.size, 12, "allocates exactly the 4-byte-aligned actual size");
  assertEquals(
    staging.usage,
    GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    "MAP_READ cannot be combined with STORAGE",
  );
  assertEquals(arena.stats.allocatedBytes, 12);

  await arena.destroy();
  assertEquals(gpu.destroyedCount(), 1, "destroying staging is the arena's job too");
});

test("RunArena.allocHostWritten rejects a usage without COPY_DST", () => {
  const gpu = createFakeGpu();
  const arena = new RunArena(gpu.device, noopFlush);

  const uniform = arena.allocHostWritten(64, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  assertEquals(uniform.usage, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  assertEquals(arena.stats.allocatedBytes, 64);

  assertThrows(() => arena.allocHostWritten(64, GPUBufferUsage.UNIFORM), ArenaError, "COPY_DST");
});

test("RunArena.destroy destroys buffers only after the flush completes", async () => {
  const gpu = createFakeGpu();
  let releaseFlush: () => void = () => {};
  const flushed = new Promise<void>((resolve) => {
    releaseFlush = resolve;
  });
  const arena = new RunArena(gpu.device, () => flushed);
  arena.allocRegion(32);
  arena.allocHostWritten(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

  const destroying = arena.destroy();
  assertEquals(gpu.destroyedCount(), 0, "destroys nothing while the flush is incomplete");
  assertThrows(
    () => arena.allocRegion(32),
    ArenaError,
    "destroyed",
    "a buffer allocated while waiting for the flush would be destroyed by no one",
  );

  // A second destroy waits for the same completion (no premature "already destroyed" answer)
  const second = arena.destroy();
  releaseFlush();
  await Promise.all([destroying, second]);
  assertEquals(gpu.destroyedCount(), 2);

  assertThrows(() => arena.allocRegion(32), ArenaError, "destroyed");
});

test("RunArena.destroy destroys buffers even when the flush fails, and reports that failure", async () => {
  const gpu = createFakeGpu();
  // The main cause in practice is device loss (the flush fails with GpuDeviceLostError)
  const lost = new Error("device was lost during flush");
  const arena = new RunArena(gpu.device, () => Promise.reject(lost));
  arena.allocRegion(32);
  arena.allocHostWritten(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

  const caught = await assertRejects(() => arena.destroy());
  assertStrictEquals(caught, lost, "the flush failure is not swallowed");
  assertEquals(
    gpu.destroyedCount(),
    2,
    "every owned buffer is destroyed even when the flush fails",
  );

  // The destroyed state is final (cleanup is done, so allocating again is refused)
  assertThrows(() => arena.allocRegion(32), ArenaError, "destroyed");
  const again = await assertRejects(() => arena.destroy());
  assertStrictEquals(again, lost);
  assertEquals(gpu.destroyedCount(), 2, "a second destroy does not destroy twice");
});
