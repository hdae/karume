// 重みの供給は block を 1 本ずつ読んで上げる（lazy — `WeightBatch.items`）。
//
// Session 構築（`createSessionFromContainer` → `containerBatches` → `buildSessionState`）は part を
// 読み切ってから上げるのではなく、「block を 1 本読む → writeBuffer → 次の block を読む」と交互に
// 進み、フェンス（空 submit + 完了待ち）は part ごとに 1 回だけ張る。交互であることが、JS 側に
// 生きる重みのバイト列を part 1 本から block 1 本へ下げる成立条件そのもの（GC で実際に何バイト
// 解放されるかは単体では観測できないので、ここは順序で固定する — 実測は tools/ram-peak）。
//
// 併せて固定するもの:
// - `SessionBuildStats.shardWaitMs` が block の読みを待った時間と、列の終わりを知るまでの待ちを
//   含む（読みが batch の本体へ移っても、供給側の費用が統計から消えない）
// - part の途中で読みが落ちたら、上げ済みの重みごとアリーナを破棄し、以後の block を 1 本も
//   読まない（transaction 境界 — lazy で初めて生じる「part の前半だけ上げた」状態）
//
// 実 GPU は使わない。順序と回数は、確保・書き込み・フェンスを記録するフェイク device で
// 決定論的に固定できる。writeBuffer が「呼んだ時点で写す」こと自体は実機でしか固定できないので
// tests/gpu_write_buffer_copy_test.ts が持つ。

import { assert, assertEquals, assertRejects, assertStrictEquals } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import type { BoundContainer } from "../src/format/container/bind.ts";
import { openMemoryContainer } from "../src/format/container/memory.ts";
import { parseIrDeclarationValue } from "../src/format/ir.ts";
import { GpuContext, readAdapterInfo, type RequiredLimits } from "../src/gpu/device.ts";
import { createSessionFromContainer, Session } from "../src/runtime/executor.ts";
import type { ReadyInitializer } from "../src/runtime/session-build.ts";
import { mergeGraph } from "./helpers/merged-graph.ts";
import { f32Bytes, GRAPH_NAME } from "./helpers/model-fixture.ts";

// ---------------------------------------------------------------------------
// 材料（part 3 本のモデル・記録するフェイク device）
// ---------------------------------------------------------------------------

/**
 * part 3 本に割れる最小のグラフ: y = x·w + b + c。
 *
 * メモリ内容器は丸ごと供給（`b` / `c`）を同じ part に積み、piece は 1 本 1 part に置く
 * （`openMemoryContainer` の part 割り）ので、part は `{b, c}` → `w` piece 1 → `w` piece 2 の
 * 3 本になる。先頭の part に block が 2 本あるので「part の中の交互」も見える。
 */
const declaration = () =>
  parseIrDeclarationValue({
    format: "karume-ir",
    version: 2,
    requires: { ops: ["add", "matmul"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [2, 4] }],
    outputs: ["y"],
    initializers: { w: {}, b: {}, c: {} },
    values: {
      w: { dtype: "f32", shape: [4, 3] },
      b: { dtype: "f32", shape: [3] },
      c: { dtype: "f32", shape: [3] },
      h: { dtype: "f32", shape: [2, 3] },
      t: { dtype: "f32", shape: [2, 3] },
      y: { dtype: "f32", shape: [2, 3] },
    },
    nodes: [
      { op: "matmul", ins: ["x", "w"], outs: ["h"], attrs: {} },
      { op: "add", ins: ["h", "b"], outs: ["t"], attrs: {} },
      { op: "add", ins: ["t", "c"], outs: ["y"], attrs: {} },
    ],
  });

/** `w` の 1 行ぶんのバイト長（f32 × 3 列）。 */
const W_ROW_BYTES = 3 * 4;

const container = (): BoundContainer => {
  const w = f32Bytes(Array.from({ length: 12 }, (_, i) => i * 0.25));
  return openMemoryContainer({
    graphs: { [GRAPH_NAME]: declaration() },
    tensors: {
      [GRAPH_NAME]: {
        w: {
          encoding: { codec: "f32" },
          // piece は読むたびに別の器（写し）を返す — 書き込みの記録が器で block を見分けるため。
          pieces: [
            { rows: [0, 2], read: () => Promise.resolve(w.slice(0, 2 * W_ROW_BYTES)) },
            {
              rows: [2, 4],
              read: () => Promise.resolve(w.slice(2 * W_ROW_BYTES, 4 * W_ROW_BYTES)),
            },
          ],
        },
        b: { encoding: { codec: "f32" }, bytes: f32Bytes([1, 2, 3]) },
        c: { encoding: { codec: "f32" }, bytes: f32Bytes([4, 5, 6]) },
      },
    },
  });
};

/** 上限は門に掛からない値でよい（見たいのは順序で、確保の寸法ではない）。 */
const LIMITS: RequiredLimits = {
  maxBufferSize: 1 << 20,
  maxStorageBufferBindingSize: 1 << 20,
  maxUniformBufferBindingSize: 65536,
  maxStorageBuffersPerShaderStage: 8,
  maxUniformBuffersPerShaderStage: 12,
  maxComputeWorkgroupStorageSize: 16384,
  maxComputeInvocationsPerWorkgroup: 256,
  maxComputeWorkgroupSizeX: 256,
  maxComputeWorkgroupSizeY: 256,
  maxComputeWorkgroupSizeZ: 64,
  maxComputeWorkgroupsPerDimension: 65535,
};

/** 取得元の読みの前に差し込む処理（遅延・失敗の注入口）。 */
type BeforeRead = (id: string) => Promise<void>;

/**
 * 記録するフェイク device と、取得元を包む口。
 *
 * 記録は 1 本の列 `events` に時系列で積む: `read <block>`（取得元の読みの開始）/
 * `write <block>`（`queue.writeBuffer` — 渡された器から、どの block の読みが返した器かを引く）/
 * `submit`（空 submit）/ `fence`（`onSubmittedWorkDone` の完了待ち）。
 *
 * DOM 型全体は再現しないので cast で渡す（tests/helpers/fake-gpu.ts と同じテスト専用の境界）。
 * GpuContext 自体は本物を使う（消失購読と errorScope の経路は本番実装のまま通す）。
 */
const recorder = () => {
  const events: string[] = [];
  const blockOf = new Map<ArrayBufferLike, string>();
  const counts = { created: 0, destroyed: 0, openScopes: 0 };
  const device = {
    lost: new Promise<GPUDeviceLostInfo>(() => {}),
    features: new Set<string>(),
    limits: { minStorageBufferOffsetAlignment: 256 },
    pushErrorScope: (): void => {
      counts.openScopes += 1;
    },
    popErrorScope: (): Promise<GPUError | null> => {
      counts.openScopes -= 1;
      return Promise.resolve(null);
    },
    createBuffer: (descriptor: GPUBufferDescriptor): GPUBuffer => {
      counts.created += 1;
      return {
        size: descriptor.size,
        destroy: (): void => {
          counts.destroyed += 1;
        },
      } as unknown as GPUBuffer;
    },
    queue: {
      writeBuffer: (_buffer: GPUBuffer, _offset: number, data: ArrayBufferView): void => {
        events.push(`write ${blockOf.get(data.buffer) ?? "（読んでいない器）"}`);
      },
      submit: (): void => {
        events.push("submit");
      },
      onSubmittedWorkDone: (): Promise<void> => {
        events.push("fence");
        return Promise.resolve();
      },
    },
  } as unknown as GPUDevice;
  const gpu = new GpuContext(device, readAdapterInfo({}), LIMITS, new Set());
  const trace = (inner: BoundContainer, before?: BeforeRead): BoundContainer => ({
    graphs: inner.graphs,
    readBlock: async (id) => {
      events.push(`read ${id}`);
      await before?.(id);
      const bytes = await inner.readBlock(id);
      blockOf.set(bytes.buffer, id);
      return bytes;
    },
  });
  return { events, counts, gpu, trace };
};

const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

// ---------------------------------------------------------------------------

describe("重みの供給は block を 1 本ずつ読んで上げる", () => {
  it("読みと writeBuffer が block ごとに交互に並び、フェンスは part ごとに 1 回だけ張る", async () => {
    const { events, gpu, trace } = recorder();
    const session = await createSessionFromContainer(gpu, trace(container()), GRAPH_NAME);
    // 破棄の flush も完了待ちを 1 回呼ぶので、構築が終わった時点の列を写しておく。
    const built = [...events];
    const { shardCount } = session.diagnostics().buildStats;
    await session.dispose();

    assertEquals(built, [
      // part 1 本目（丸ごと供給 b / c）: part を読み切らずに 1 本ずつ上げる。
      "read model/b",
      "write model/b",
      "read model/c",
      "write model/c",
      "submit",
      "fence",
      // 次の part の読みはフェンスの後。
      "read model/w#piece1",
      "write model/w#piece1",
      "submit",
      "fence",
      "read model/w#piece2",
      "write model/w#piece2",
      "submit",
      "fence",
    ]);
    // フェンスの回数は上の事象列の完全一致が固定する。ここが数えるのは消費した batch の本数。
    assertEquals(shardCount, 3, "消費した batch の本数 = part の本数");
  });

  it("shardWaitMs は block の読みを待った時間を含む（読みの遅い取得元）", async () => {
    const DELAY_MS = 20;
    const { gpu, trace } = recorder();
    let readMs = 0;
    let reads = 0;
    const slow = trace(container(), async () => {
      const start = performance.now();
      await sleep(DELAY_MS);
      readMs += performance.now() - start;
      reads += 1;
    });
    const session = await createSessionFromContainer(gpu, slow, GRAPH_NAME);
    const { shardWaitMs } = session.diagnostics().buildStats;
    await session.dispose();

    assertEquals(reads, 4, "block 4 本（b / c / w の piece 2 本）を 1 回ずつ読む");
    assert(readMs >= DELAY_MS, `読みの遅延が効いていない（${readMs} ms — 検出器が空振りする）`);
    // 読み 1 回の区間は、同じ時計で測る反復待ちの区間の内側に入れ子で収まる。したがって下限は
    // 読みの時間の総和そのもの（許すのは浮動小数の足し算の順序差だけ）。
    const SUM_ORDER_EPSILON_MS = 1e-6;
    assert(
      shardWaitMs >= readMs - SUM_ORDER_EPSILON_MS,
      `shardWaitMs ${shardWaitMs} ms が読みを待った時間 ${readMs} ms を含んでいない`,
    );
  });

  it("shardWaitMs は列の終わりを知るまでの待ちも含む（最後の item の後で遅れる列）", async () => {
    // 容器経路（`containerBatches`）の列の終わりは読みを伴わず、取得元の読みに遅延を入れても
    // ここには届かない。そこで構築の入口（`Session.build`）へ、終わりを告げる前に遅れる列を
    // 直接渡す — item 0 本なので、測られる待ちは列の終わりの待ちだけになる。
    const DELAY_MS = 20;
    const { gpu } = recorder();
    let tailMs = 0;
    const lateEnd: AsyncIterable<ReadyInitializer> = {
      [Symbol.asyncIterator]: () => ({
        next: async (): Promise<IteratorResult<ReadyInitializer>> => {
          const start = performance.now();
          await sleep(DELAY_MS);
          tailMs += performance.now() - start;
          return { done: true, value: undefined };
        },
      }),
    };
    const emptyGraph = mergeGraph({
      format: "karume-ir",
      version: 2,
      requires: { ops: [] },
      symbols: [],
      inputs: [],
      outputs: [],
      initializers: {},
      values: {},
      nodes: [],
    });
    async function* oneBatch() {
      yield { origin: "part 1", items: lateEnd };
    }
    const session = await Session.build(gpu, emptyGraph, new Map(), oneBatch(), {});
    const { shardWaitMs, shardCount } = session.diagnostics().buildStats;
    await session.dispose();

    assertEquals(shardCount, 1);
    assert(
      tailMs >= DELAY_MS,
      `列の終わりの遅延が効いていない（${tailMs} ms — 検出器が空振りする）`,
    );
    const SUM_ORDER_EPSILON_MS = 1e-6;
    assert(
      shardWaitMs >= tailMs - SUM_ORDER_EPSILON_MS,
      `shardWaitMs ${shardWaitMs} ms が列の終わりの待ち ${tailMs} ms を含んでいない`,
    );
  });

  it("part の途中で読みが落ちたら、上げ済みの重みごとアリーナを破棄し、以後の block を読まない", async () => {
    const { events, counts, gpu, trace } = recorder();
    const injected = new Error("注入: block の取得に失敗");
    // part 1 本目の 2 本目（b を上げた後・フェンスの前）で落とす。
    const failing = trace(
      container(),
      (id) => id === "model/c" ? Promise.reject(injected) : Promise.resolve(),
    );
    const error = await assertRejects(() => createSessionFromContainer(gpu, failing, GRAPH_NAME));

    assertStrictEquals(error, injected, "読みの失敗は包み直さずにそのまま抜ける");
    assertEquals(
      events.filter((event) => event.startsWith("read ") || event.startsWith("write ")),
      ["read model/b", "write model/b", "read model/c"],
      "失敗の後に次の block を読んでいる（供給の列が閉じていない）",
    );
    assertEquals(events.includes("submit"), false, "失敗した part のフェンスを張っている");
    assert(counts.created > 0, "上げ済みの重みが無い（part の前半を上げた状態を作れていない）");
    assertEquals(counts.destroyed, counts.created, "上げ済みの重みのバッファが破棄されていない");
    assertEquals(counts.openScopes, 0, "errorScope を積み残している");
  });
});
