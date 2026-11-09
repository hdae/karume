// The staging ring of the exact expert cache without a GPU (ADR 0146 decision 5): its size, a load's
// rounds and moves (each load paired with the eviction of the slot it takes), and a slot's life —
// written only while mapped, mapped again only after the submit, waited for until its mapping
// resolves (which WebGPU resolves only after the copies that read the slot completed), a mapping
// that settled before the wait not missed, and every broken slot failing loudly instead of being
// written. The ring through a load is in expert_cache_test.ts (fake device) and
// gpu_expert_cache_test.ts (bit identity, rounds on a real GPU).

import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { GpuContext, GpuDeviceLostError, readAdapterInfo } from "../src/gpu/device.ts";
import {
  EXPERT_RING_BYTES,
  EXPERT_RING_SLOTS,
  expertLoadMoves,
  ringRounds,
  stagingRingSlots,
} from "../src/runtime/expert-plan.ts";
import { ExpertStagingRing } from "../src/runtime/expert-staging-ring.ts";
import { ExecutionError } from "../src/runtime/plan.ts";

describe("the staging ring's size", () => {
  it("is 32 slots of 26B experts (3,345,408 bytes, a load of up to 30 · 64)", () => {
    assertEquals([EXPERT_RING_SLOTS, EXPERT_RING_BYTES], [32, 134217728]);
    assertEquals(stagingRingSlots(3345408, 30 * 64), 32);
  });

  it("takes fewer slots for experts so large that 32 would pass 128 MiB, and at least one", () => {
    assertEquals(stagingRingSlots(8 * 2 ** 20, 1920), 16);
    assertEquals(stagingRingSlots(200 * 2 ** 20, 1920), 1);
  });

  it("never takes more slots than one load can need (layers · K)", () => {
    assertEquals(stagingRingSlots(3456, 2 * 4), 8);
  });
});

describe("a load's rounds through the ring", () => {
  it("cuts the moves into full rounds in order, the remainder last", () => {
    assertEquals(ringRounds(0, 32), []);
    assertEquals(ringRounds(5, 32), [{ first: 0, count: 5 }]);
    assertEquals(ringRounds(64, 32), [
      { first: 0, count: 32 },
      { first: 32, count: 32 },
    ]);
    assertEquals(ringRounds(70, 32), [
      { first: 0, count: 32 },
      { first: 32, count: 32 },
      { first: 64, count: 6 },
    ]);
  });

  it("refuses a ring without slots", () => {
    assertThrows(() => ringRounds(3, 0), ExecutionError, "broken bookkeeping");
  });
});

describe("a load's moves", () => {
  it("pairs each load with the eviction of the slot it takes, layer by layer in load order", () => {
    const moves = expertLoadMoves([
      {
        load: [
          { expert: 7, slot: 2 },
          { expert: 4, slot: 1 },
          { expert: 9, slot: 3 },
        ],
        evict: [
          { expert: 6, slot: 1 },
          { expert: 5, slot: 2 },
        ],
      },
      { load: [], evict: [] },
      { load: [{ expert: 0, slot: 0 }], evict: [] },
    ]);
    assertEquals(moves, [
      { layer: 0, expert: 7, slot: 2, victim: 5 },
      { layer: 0, expert: 4, slot: 1, victim: 6 },
      { layer: 0, expert: 9, slot: 3, victim: undefined },
      { layer: 2, expert: 0, slot: 0, victim: undefined },
    ]);
  });
});

/**
 * A slot buffer whose `mapAsync` the test settles by hand (the GPU finishing the copies that read
 * it), and which refuses its mapped range while unmapped — what WebGPU does.
 */
const fakeSlot = (size: number) => {
  const bytes = new Uint8Array(size);
  const state = {
    mapped: true,
    log: [] as string[],
    settle: undefined as ((failure?: Error) => void) | undefined,
  };
  const buffer = {
    size,
    getMappedRange: (offset = 0, length = size): ArrayBuffer => {
      if (!state.mapped) throw new Error("fake: getMappedRange of an unmapped buffer");
      assertEquals([offset, length], [0, size]);
      return bytes.buffer;
    },
    unmap: (): void => {
      state.mapped = false;
      state.log.push("unmap");
    },
    mapAsync: (): Promise<void> => {
      state.log.push("mapAsync");
      return new Promise<void>((resolve, reject) => {
        state.settle = (failure) => {
          if (failure !== undefined) return reject(failure);
          state.mapped = true;
          resolve();
        };
      });
    },
  };
  return { buffer: buffer as unknown as GPUBuffer, bytes, state };
};

/** A GpuContext over a device that never draws, whose loss the test triggers. */
const losableGpu = () => {
  let lose: (info: GPUDeviceLostInfo) => void = () => undefined;
  const device = {
    lost: new Promise<GPUDeviceLostInfo>((resolve) => {
      lose = resolve;
    }),
    features: new Set<string>(),
    limits: {},
    destroy: (): void => undefined,
  } as unknown as GPUDevice;
  const gpu = new GpuContext(device, readAdapterInfo({}), {} as never, new Set());
  return {
    gpu,
    lose: () => lose({ reason: "unknown", message: "injected" } as GPUDeviceLostInfo),
  };
};

/** Lets every pending reaction run (the ring's promises settle in microtasks). */
const drain = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 0));

describe("a staging-ring slot's life", () => {
  it("writes each entry into its slot while mapped, then unmaps it", async () => {
    const slots = [fakeSlot(8), fakeSlot(8)];
    const ring = new ExpertStagingRing(
      slots.map((slot) => slot.buffer),
      8,
    );
    const { gpu } = losableGpu();
    await ring.acquire(2, gpu);
    const staged = ring.stage([
      Uint8Array.of(1, 2, 3, 4),
      Uint8Array.of(5, 6, 7, 8, 9, 10, 11, 12),
    ]);
    assertEquals(staged, [slots[0].buffer, slots[1].buffer]);
    assertEquals([...slots[0].bytes], [1, 2, 3, 4, 0, 0, 0, 0]);
    assertEquals([...slots[1].bytes], [5, 6, 7, 8, 9, 10, 11, 12]);
    assertEquals(
      slots.map((slot) => [slot.state.mapped, slot.state.log]),
      [
        [false, ["unmap"]],
        [false, ["unmap"]],
      ],
    );
  });

  it("does not hand a slot out again until the mapping asked after its submit resolves", async () => {
    const slot = fakeSlot(4);
    const ring = new ExpertStagingRing([slot.buffer], 4);
    const { gpu } = losableGpu();
    ring.stage([Uint8Array.of(1, 2, 3, 4)]);
    ring.remap(1);
    let acquired = false;
    const waiting = ring.acquire(1, gpu).then(() => {
      acquired = true;
    });
    await drain();
    // The copies that read the slot are still running (the mapping has not resolved).
    assert(!acquired, "the slot was handed out while its mapping was pending");
    slot.state.settle?.();
    await waiting;
    assert(acquired && slot.state.mapped);
    assertEquals(slot.state.log, ["unmap", "mapAsync"]);
  });

  it("does not miss a mapping that resolved before the next load waited for it", async () => {
    const slot = fakeSlot(4);
    const ring = new ExpertStagingRing([slot.buffer], 4);
    const { gpu } = losableGpu();
    ring.stage([Uint8Array.of(1, 2, 3, 4)]);
    ring.remap(1);
    slot.state.settle?.();
    await drain();
    await ring.acquire(1, gpu);
    ring.stage([Uint8Array.of(9, 9, 9, 9)]);
    assertEquals([...slot.bytes], [9, 9, 9, 9]);
  });

  it("fails a wait for a slot that was staged but not mapped again (a round broken in between)", async () => {
    const ring = new ExpertStagingRing([fakeSlot(4).buffer], 4);
    const { gpu } = losableGpu();
    ring.stage([Uint8Array.of(1, 2, 3, 4)]);
    await assertRejects(() => ring.acquire(1, gpu), ExecutionError, "staged but not mapped again");
  });

  it("fails a wait for a slot whose mapping failed, and holds a failure nobody waits for", async () => {
    const slots = [fakeSlot(4), fakeSlot(4)];
    const ring = new ExpertStagingRing(
      slots.map((slot) => slot.buffer),
      4,
    );
    const { gpu } = losableGpu();
    ring.stage([Uint8Array.of(1, 2, 3, 4), Uint8Array.of(5, 6, 7, 8)]);
    ring.remap(2);
    // Slot 1's failure is never waited for (as after a dispose): it must not surface as an unhandled
    // rejection, which the runner reports as a failure of this file.
    slots[1].state.settle?.(new Error("aborted"));
    slots[0].state.settle?.(new Error("operation failed"));
    await assertRejects(() => ring.acquire(1, gpu), ExecutionError, "operation failed");
    await drain();
  });

  it("fails a wait whose mapping never resolves once the device is lost", async () => {
    const slot = fakeSlot(4);
    const ring = new ExpertStagingRing([slot.buffer], 4);
    const { gpu, lose } = losableGpu();
    ring.stage([Uint8Array.of(1, 2, 3, 4)]);
    ring.remap(1);
    const waiting = ring.acquire(1, gpu);
    lose();
    await assertRejects(() => waiting, GpuDeviceLostError, "staging ring");
  });

  it("refuses a round larger than the ring and an entry larger than a slot", async () => {
    const ring = new ExpertStagingRing([fakeSlot(4).buffer], 4);
    const { gpu } = losableGpu();
    await assertRejects(
      () => ring.acquire(2, gpu),
      ExecutionError,
      "a round of 2 experts in 1 slots",
    );
    assertThrows(
      () => ring.stage([Uint8Array.of(1, 2, 3, 4, 5, 6, 7, 8)]),
      ExecutionError,
      "does not fit a 4-byte slot",
    );
  });
});
