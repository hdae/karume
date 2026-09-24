/**
 * `withSession`（`src/session/with-session.ts`）の実行スコープ契約:
 *
 * - body が返しても throw しても、張った Session を**ちょうど 1 回** dispose する（途中で落ちた
 *   段の VRAM が残ると、後続の段が「最初の失敗とは別の場所」で確保に落ちる）。
 * - body の返り値と例外は包まずにそのまま呼び手へ渡る。
 * - `run` は Session の出力を返し、`observe` はその run の**後**・呼び手へ戻る**前**に毎回呼ばれる。
 * - Session の構築が落ちたら body は呼ばれない。
 *
 * NOTE: `Session` は `#` 付き private field を持つクラスで fake に置き換えられないため、重みを
 * 持たない小さなグラフ（y = x + x）をメモリ内容器で組んで実 Session を張る。dispose の回数は
 * 実 Session の `dispose` を包んで数え、解放そのものは「以後の run が落ちる」ことで見る。
 */

import { assert, assertEquals, assertRejects, assertStrictEquals } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
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
import type { ModelComponent } from "../src/hub/components.ts";
import { withSession } from "../src/session/with-session.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const GRAPH = "double";
const LENGTH = 2;

/** 重みを 1 本も持たないグラフ（y = x + x）。宣言だけで Session が張れる最小形。 */
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

/** withSession が Session に対して何をしたかの記録。 */
type SessionLog = {
  readonly sessions: Session[];
  readonly options: (SessionOptions | undefined)[];
  disposeCalls: number;
};

/**
 * 実 Session を張る {@link ModelComponent}。張った Session の `dispose` を包んで呼び出し回数を
 * 数える（解放は元の `dispose` へそのまま委ねる）。
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
      throw new Error(`この fixture は資産を持たない: ${name}`);
    },
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

/** 解放済みの Session は以後の run を fail loudly で拒む（dispose が本当に効いたことの観測点）。 */
const assertDisposed = async (session: Session): Promise<void> => {
  await assertRejects(
    () => session.run(inputOf([1, 2])),
    ExecutionError,
    "dispose 済み",
  );
};

const withGpu = async (fn: (gpu: GpuContext) => Promise<void>): Promise<void> => {
  const gpu = await acquireGpu();
  try {
    await fn(gpu);
  } finally {
    gpu.destroy();
  }
};

describe({
  name: "withSession（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: () => {
    describe("body が値を返したとき", () => {
      it("その値を包まずに返し、渡した options で張った Session を 1 回だけ dispose する", async () => {
        await withGpu(async (gpu) => {
          const log = emptyLog();
          const options: SessionOptions = { planBackingBudgetBytes: 0 };
          const value = { marker: "body の返り値" };
          let received: Session | undefined;

          const result = await withSession(gpu, spiedComponent(log), options, undefined, (_, s) => {
            received = s;
            return Promise.resolve(value);
          });

          assertStrictEquals(result, value);
          assertEquals(log.sessions.length, 1);
          assertStrictEquals(log.options[0], options);
          assertStrictEquals(received, log.sessions[0], "body に渡る Session が張ったものと違う");
          assertEquals(log.disposeCalls, 1);
          await assertDisposed(log.sessions[0]);
        });
      });
    });

    describe("body が throw したとき", () => {
      it("Session を 1 回だけ dispose し、body の例外をそのまま伝える", async () => {
        await withGpu(async (gpu) => {
          const log = emptyLog();
          const thrown = new Error("body の途中で落ちた");

          const error = await assertRejects(() =>
            withSession(gpu, spiedComponent(log), {}, undefined, async (run) => {
              await run(inputOf([1, 2]));
              throw thrown;
            })
          );

          assertStrictEquals(error, thrown);
          assertEquals(log.sessions.length, 1);
          assertEquals(log.disposeCalls, 1);
          await assertDisposed(log.sessions[0]);
        });
      });
    });

    describe("run", () => {
      it("Session の出力を返し、observe を run の後・呼び手へ戻る前に 1 回ずつ呼ぶ", async () => {
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
            events.push("run 1 が戻った");
            const second = await run(inputOf([3, -4]));
            events.push("run 2 が戻った");
            return [valuesOf(first), valuesOf(second)];
          });

          assertEquals(outputs, [[2, 4], [6, -8]]);
          assertEquals(events, ["observe", "run 1 が戻った", "observe", "run 2 が戻った"]);
          // run の前に読んだ診断なら直近 run の欄は空のまま。
          assert(observed.every((diagnostics) => diagnostics.lastRun !== undefined));
          assertEquals(log.disposeCalls, 1);
        });
      });

      it("observe が無くても Session の出力を返す", async () => {
        await withGpu(async (gpu) => {
          const log = emptyLog();

          const values = await withSession(
            gpu,
            spiedComponent(log),
            {},
            undefined,
            async (run) => valuesOf(await run(inputOf([0.5, -1.5]))),
          );

          assertEquals(values, [1, -3]);
          assertEquals(log.disposeCalls, 1);
        });
      });
    });

    describe("Session の構築が落ちたとき", () => {
      it("body を呼ばず、構築の例外をそのまま伝える", async () => {
        await withGpu(async (gpu) => {
          const failure = new Error("Session を張れなかった");
          const component: ModelComponent = {
            ...spiedComponent(emptyLog()),
            createSession: () => Promise.reject(failure),
          };
          let bodyCalled = false;

          const error = await assertRejects(() =>
            withSession(gpu, component, {}, undefined, () => {
              bodyCalled = true;
              return Promise.resolve();
            })
          );

          assertStrictEquals(error, failure);
          assertEquals(bodyCalled, false);
        });
      });
    });
  },
});
