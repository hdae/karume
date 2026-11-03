import { assertEquals, assertRejects, assertStrictEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { fakeDevice, fakeGpuContext } from "../../runtime/tests/helpers/fake-gpu.ts";
import {
  GEMV_INTEGER_CASES,
  GEMV_PLANES_CASES,
  GEMV_SUBGROUP_CASES,
  RMS_SUBGROUP_CASES,
  STATE_ATTENTION_CASES,
  STATE_ATTENTION_MUTANTS,
} from "../mod.ts";
import { scopedGpuWork } from "../src/result.ts";

/**
 * A real GpuContext around a fake device that counts the error scopes pushed and popped and keeps
 * the event listeners still attached.
 */
const recordingGpu = (
  popErrorScope: () => Promise<GPUError | null> = () => Promise.resolve(null),
) => {
  let pushed = 0;
  let popped = 0;
  const listeners = new Map<unknown, string>();
  const device = Object.assign(
    fakeDevice({
      popErrorScope: () => {
        popped += 1;
        return popErrorScope();
      },
    }),
    {
      pushErrorScope: (): void => {
        pushed += 1;
      },
      addEventListener: (type: string, listener: unknown): void => {
        listeners.set(listener, type);
      },
      removeEventListener: (_type: string, listener: unknown): void => {
        listeners.delete(listener);
      },
    },
  );
  return {
    gpu: fakeGpuContext(device),
    left: () => ({ pushed, popped, listeners: [...listeners.values()] }),
  };
};

describe("scopedGpuWork", () => {
  describe("when the work succeeds and no scope caught an error", () => {
    it("returns the work's value with every scope popped and its listener removed", async () => {
      const { gpu, left } = recordingGpu();
      assertEquals(await scopedGpuWork(gpu, "case", () => Promise.resolve(7)), 7);
      assertEquals(left(), { pushed: 3, popped: 3, listeners: [] });
    });
  });

  describe("when the work throws synchronously instead of returning a promise", () => {
    it("rejects with the work's error after popping the three scopes and removing its listener", async () => {
      const { gpu, left } = recordingGpu();
      const thrown = new Error("buffer size is not a multiple of 4");
      const error = await assertRejects(() =>
        scopedGpuWork(gpu, "case", () => {
          throw thrown;
        }),
      );
      assertStrictEquals(error, thrown);
      assertEquals(left(), { pushed: 3, popped: 3, listeners: [] });
    });
  });

  describe("when a pop rejects", () => {
    const rejectingPop = () => Promise.reject(new Error("no error scope on the stack"));

    it("carries the work's own error as the cause and still removes its listener", async () => {
      const { gpu, left } = recordingGpu(rejectingPop);
      const workError = new Error("mapAsync aborted");
      const error = await assertRejects(
        () => scopedGpuWork(gpu, "e2b-full P=0", () => Promise.reject(workError)),
        Error,
        "e2b-full P=0: popErrorScope rejected: no error scope on the stack",
      );
      assertStrictEquals(error.cause, workError);
      assertEquals(left().listeners, []);
    });

    it("rejects with the pop's error after a successful work and still removes its listener", async () => {
      const { gpu, left } = recordingGpu(rejectingPop);
      await assertRejects(
        () => scopedGpuWork(gpu, "case", () => Promise.resolve(1)),
        Error,
        "no error scope on the stack",
      );
      assertEquals(left().listeners, []);
    });
  });
});

describe("the expected case counts each check exports", () => {
  // The node GPU tests and the playground hold a run to these counts; a changed grid changes them
  // here as well, on purpose
  it("are the counts of the grids the checks were measured at", () => {
    assertEquals(GEMV_SUBGROUP_CASES, { "gemv-subgroup32": 180 });
    assertEquals(RMS_SUBGROUP_CASES, { "rms-rows": 48, "rms-fusion": 12, "rms-grid-stride": 1 });
    assertEquals(STATE_ATTENTION_CASES, {
      "state-qk-fold": 252,
      "state-pv-unroll": 546,
      "state-stats-pv-unroll": 490,
    });
    assertEquals(STATE_ATTENTION_MUTANTS, 5);
    assertEquals(GEMV_INTEGER_CASES, { "gemv-integer": 692 });
    assertEquals(GEMV_PLANES_CASES, { "gemv-planes": 240 });
  });
});
