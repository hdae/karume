/**
 * The execution-scope contract of `withSession` (`src/session/with-session.ts`):
 *
 * - Whether the body returns or throws, the Session it built is disposed **exactly once** (if the
 *   VRAM of a stage that failed midway stays, a later stage fails its allocation "somewhere other
 *   than the first failure").
 * - The body's return value and exception reach the caller as they are, unwrapped. Only when
 *   dispose fails as well are both carried in an `AggregateError` with the body's exception first
 *   (a dispose failure does not overwrite the first failure).
 * - `run` returns the Session's outputs, and `observe` is called every time **after** that run
 *   and **before** control returns to the caller.
 * - If building the Session fails, the body is not called (the branch that acquires no GPU — this
 *   file's last describe, run without an adapter).
 *
 * NOTE: `Session` is a class with `#` private fields and cannot be replaced by a fake, so a real
 * Session is built from a small weightless graph (y = x + x) in an in-memory container. The dispose
 * count is taken by wrapping the real Session's `dispose`, and the release itself is observed as
 * "later runs fail".
 */

import { assert, assertEquals, assertRejects, assertStrictEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import {
  acquireGpu,
  ExecutionError,
  type GpuContext,
  openMemoryContainer,
  parseIrDeclarationValue,
  prepareContainer,
  type Session,
  type SessionDiagnostics,
  type SessionOptions,
  type Tensor,
} from "@karume/runtime";
import { fakeDevice, fakeGpuContext } from "../../runtime/tests/helpers/fake-gpu.ts";
import type { ModelComponent } from "../src/hub/components.ts";
import { withSession } from "../src/session/with-session.ts";
import { openerOf, single } from "./helpers/asset-container-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const GRAPH = "double";
const LENGTH = 2;

/**
 * A graph without a single weight (y = x + x). The smallest form that builds a Session from
 * declarations alone.
 */
const doubleGraph = () =>
  parseIrDeclarationValue({
    format: "karume-ir",
    version: 2,
    requires: { ops: ["add"] },
    symbols: [],
    initializers: {},
    inputs: [{ name: "x", dtype: "f32", shape: [LENGTH] }],
    outputs: ["y"],
    values: { y: { dtype: "f32", shape: [LENGTH] } },
    nodes: [{ op: "add", ins: ["x", "x"], outs: ["y"], attrs: {} }],
  });

const inputOf = (values: readonly number[]): Record<string, Tensor> => ({
  x: { dtype: "f32", shape: [LENGTH], data: Float32Array.from(values) },
});

const valuesOf = (outputs: Record<string, Tensor>): number[] => Array.from(outputs["y"].data);

/** A record of what withSession did to the Session. */
type SessionLog = {
  readonly sessions: Session[];
  readonly options: (SessionOptions | undefined)[];
  disposeCalls: number;
};

/**
 * A {@link ModelComponent} that builds a real Session. It wraps the built Session's `dispose` to
 * count the calls (the release itself is delegated to the original `dispose`).
 */
const spiedComponent = (log: SessionLog): ModelComponent => {
  const prepared = prepareContainer(
    openMemoryContainer({ graphs: { [GRAPH]: doubleGraph() }, tensors: {} }),
    GRAPH,
  );
  return {
    graph: prepared.graph,
    assets: {},
    asset: (name) => {
      throw new Error(`this fixture has no assets: ${name}`);
    },
    estimate: (options) => prepared.estimate(options),
    maxPartBytes: 0,
    createSession: async (gpu, options) => {
      log.options.push(options);
      const session = await prepared.createContainerSession(gpu, options);
      const dispose = session.dispose.bind(session);
      session.dispose = () => {
        log.disposeCalls += 1;
        return dispose();
      };
      log.sessions.push(session);
      return session;
    },
  };
};

const emptyLog = (): SessionLog => ({ sessions: [], options: [], disposeCalls: 0 });

/**
 * A released Session fails loudly on any later run (the observation point that dispose really took
 * effect).
 */
const assertDisposed = async (session: Session): Promise<void> => {
  await assertRejects(() => session.run(inputOf([1, 2])), ExecutionError, "disposed Session");
};

const withGpu = async (fn: (gpu: GpuContext) => Promise<void>): Promise<void> => {
  const gpu = await acquireGpu();
  try {
    await fn(gpu);
  } finally {
    gpu.destroy();
  }
};

describe.skipIf(!GPU_AVAILABLE)("withSession (real GPU)", () => {
  describe("when the body returns a value", () => {
    it("returns that value unwrapped and disposes the Session built with the given options exactly once", async () => {
      await withGpu(async (gpu) => {
        const log = emptyLog();
        const options: SessionOptions = { planBackingBudgetBytes: 0 };
        const value = { marker: "the body's return value" };
        let received: Session | undefined;

        const result = await withSession(gpu, spiedComponent(log), options, undefined, (_, s) => {
          received = s;
          return Promise.resolve(value);
        });

        assertStrictEquals(result, value);
        assertEquals(log.sessions.length, 1);
        assertStrictEquals(log.options[0], options);
        assertStrictEquals(
          received,
          log.sessions[0],
          "the Session passed to the body is not the one built",
        );
        assertEquals(log.disposeCalls, 1);
        await assertDisposed(log.sessions[0]);
      });
    });
  });

  describe("when the body throws", () => {
    it("disposes the Session exactly once and passes the body's exception through as is", async () => {
      await withGpu(async (gpu) => {
        const log = emptyLog();
        const thrown = new Error("failed in the middle of the body");

        const error = await assertRejects(() =>
          withSession(gpu, spiedComponent(log), {}, undefined, async (run) => {
            await run(inputOf([1, 2]));
            throw thrown;
          }),
        );

        assertStrictEquals(error, thrown);
        assertEquals(log.sessions.length, 1);
        assertEquals(log.disposeCalls, 1);
        await assertDisposed(log.sessions[0]);
      });
    });
  });

  describe("when both the body and dispose throw", () => {
    it("does not let the dispose failure overwrite and passes both in an AggregateError, the body's exception first", async () => {
      await withGpu(async (gpu) => {
        const log = emptyLog();
        const thrown = new Error("failed in the middle of the body");
        const disposeFailure = new Error("flush failed");
        const spied = spiedComponent(log);
        const component: ModelComponent = {
          ...spied,
          createSession: async (gpuContext, options) => {
            const session = await spied.createSession(gpuContext, options);
            const dispose = session.dispose.bind(session);
            session.dispose = async () => {
              await dispose();
              throw disposeFailure;
            };
            return session;
          },
        };

        const error = await assertRejects(
          () =>
            withSession(gpu, component, {}, undefined, () => {
              throw thrown;
            }),
          AggregateError,
        );

        assertEquals(error.errors.length, 2);
        assertStrictEquals(error.errors[0], thrown);
        assertStrictEquals(error.errors[1], disposeFailure);
        assertEquals(log.disposeCalls, 1);
        await assertDisposed(log.sessions[0]);
      });
    });
  });

  describe("run", () => {
    it("returns the Session's outputs and calls observe once per run, after the run and before returning to the caller", async () => {
      await withGpu(async (gpu) => {
        const log = emptyLog();
        const events: string[] = [];
        const observed: SessionDiagnostics[] = [];
        const observe = (diagnostics: SessionDiagnostics) => {
          events.push("observe");
          observed.push(diagnostics);
        };

        const outputs = await withSession(gpu, spiedComponent(log), {}, observe, async (run) => {
          const first = await run(inputOf([1, 2]));
          events.push("run 1 returned");
          const second = await run(inputOf([3, -4]));
          events.push("run 2 returned");
          return [valuesOf(first), valuesOf(second)];
        });

        assertEquals(outputs, [
          [2, 4],
          [6, -8],
        ]);
        assertEquals(events, ["observe", "run 1 returned", "observe", "run 2 returned"]);
        // Diagnostics read before the run would still have an empty latest-run field.
        assert(observed.every((diagnostics) => diagnostics.lastRun !== undefined));
        assertEquals(log.disposeCalls, 1);
      });
    });

    it("returns the Session's outputs without observe as well", async () => {
      await withGpu(async (gpu) => {
        const log = emptyLog();

        const values = await withSession(gpu, spiedComponent(log), {}, undefined, async (run) =>
          valuesOf(await run(inputOf([0.5, -1.5]))),
        );

        assertEquals(values, [1, -3]);
        assertEquals(log.disposeCalls, 1);
      });
    });
  });
});

/*
 * NOTE: `withSession` starts by calling `createSession(gpu, …)` and does not touch the gpu before
 * that. A component whose build rejects immediately needs no real adapter, so a fake `GpuContext`
 * (`fake-gpu.ts`) is passed.
 */
describe("withSession", () => {
  describe("when building the Session fails", () => {
    it("does not call the body and passes the build's exception through as is", async () => {
      const failure = new Error("could not build the Session");
      const open = await openerOf({ dit: single() });
      const component: ModelComponent = {
        ...open("dit"),
        createSession: () => Promise.reject(failure),
      };
      let bodyCalled = false;

      const error = await assertRejects(() =>
        withSession(fakeGpuContext(fakeDevice()), component, {}, undefined, () => {
          bodyCalled = true;
          return Promise.resolve();
        }),
      );

      assertStrictEquals(error, failure);
      assertEquals(bodyCalled, false);
    });
  });
});
