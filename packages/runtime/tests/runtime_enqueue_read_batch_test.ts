// `Session.enqueue` / `Session.enqueueRead` が受け取る `options.batch` は、**利用者のオブジェクト
// のプロパティ**なので getter を仕込める。読むたびに別の値を返す getter を通すと、受け口の検査が
// 見た区間・in-flight リースを取った区間・`#readBatch` に覚えた区間が食い違い、
//
// ① 「未決着 run がある Session には enqueue できない」の検査を素通りして自己デッドロックになる
// ② getter の throw が戻り Promise の reject ではなく **同期 throw** として漏れる
// ③ `enqueueRead` の「決着まで後続 enqueue を拒む」門が別の区間に掛かり、読む slot を後続 enqueue に
//    上書きされる（沈黙誤値）
//
// が起きる。ここが固定するのは「`options.batch` は発行の同期区間で 1 度だけ読み、以後はその値だけを
// 使う」— 上の 3 つはその系。
//
// 実 GPU は使わない。検査もリース取得も dispatch を 1 本も出さない同期区間の話なので、フェイク
// device（`helpers/fake-gpu.ts`）で決定論的に固定できる。

import { assertEquals, assertInstanceOf, assertNotEquals, assertStringIncludes } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import type { IrGraph } from "../src/format/ir.ts";
import {
  type BatchScope,
  BatchScopeError,
  GpuContext,
  readAdapterInfo,
  type RequiredLimits,
} from "../src/gpu/device.ts";
import { Session } from "../src/runtime/executor.ts";
import { fakeDevice } from "./helpers/fake-gpu.ts";
import { mergeGraph } from "./helpers/merged-graph.ts";

// ---------------------------------------------------------------------------
// 材料（空グラフ・フェイク device・決着の観測）
// ---------------------------------------------------------------------------

/** ノード 0 本のグラフ。受け口の検査は積む中身を 1 バイトも見ないので、これで足りる。 */
const emptyGraph = (): IrGraph =>
  mergeGraph({
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

/** 上限は門に掛からない値でよい（検査したいのは受け口で、確保の寸法ではない）。 */
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

/**
 * `helpers/fake-gpu.ts` の最小フェイクに `limits` を足した device。
 * 構築相（`buildSessionState`）が `minStorageBufferOffsetAlignment` を 1 回読むため。
 */
const deviceWithLimits = (): GPUDevice =>
  Object.assign(fakeDevice(), { limits: { minStorageBufferOffsetAlignment: 256 } });

const gpuContext = (): GpuContext =>
  new GpuContext(deviceWithLimits(), readAdapterInfo({}), LIMITS, new Set());

const emptySession = async (gpu: GpuContext): Promise<Session> => {
  async function* noShards(): AsyncGenerator<never> {}
  return await Session.build(gpu, emptyGraph(), new Map(), noShards(), {});
};

/**
 * 決着の観測（`"resolved"` / `"rejected"` / `"pending"`）。
 *
 * 退行すると対象は**永久に pending** になる（自己デッドロックは例外も診断も出ない）ので、
 * 待ち切らずに打ち切って「pending のままだった」を assert できる形にする。
 */
const outcomeOf = async (
  awaited: Promise<unknown>,
): Promise<{ readonly state: string; readonly cause?: unknown }> => {
  const cutoff = Promise.withResolvers<{ readonly state: string }>();
  const timer = setTimeout(() => cutoff.resolve({ state: "pending" }), 500);
  try {
    return await Promise.race([
      awaited.then(() => ({ state: "resolved" }), (cause) => ({ state: "rejected", cause })),
      cutoff.promise,
    ]);
  } finally {
    clearTimeout(timer);
  }
};

const rejectionOf = async (awaited: Promise<unknown>): Promise<unknown> => {
  const outcome = await outcomeOf(awaited);
  assertEquals(outcome.state, "rejected");
  return outcome.cause;
};

// ---------------------------------------------------------------------------

describe("enqueue の受け口は options.batch を 1 度だけ読む", () => {
  it("getter を仕込まない通常形では 1 度しか読まず、受理も読み戻しも区間の決着も通る", async () => {
    const gpu = gpuContext();
    const session = await emptySession(gpu);
    const batch = await gpu.beginBatch();
    let reads = 0;
    const read = session.enqueueRead({}, {
      get batch(): BatchScope {
        reads += 1;
        return batch;
      },
    });
    const finished = outcomeOf(batch.finish());
    assertEquals(await outcomeOf(read.admitted), { state: "resolved" });
    assertEquals(await outcomeOf(read.outputs), { state: "resolved" });
    assertEquals(await finished, { state: "resolved" });
    // 読みが 1 度きりであること自体が、以下 3 本の退行が起きえない根拠になる。
    assertEquals(reads, 1);
    await session.dispose();
  });

  it("2 度目以降に別の区間を返す getter でも、後続 enqueue を拒む対象は入った区間のまま", async () => {
    const gpu = gpuContext();
    const session = await emptySession(gpu);
    // 区間は device 単位で排他なので、すり替え先は別 GpuContext で開く。
    const other = gpuContext();
    const entered = await gpu.beginBatch();
    const decoy = await other.beginBatch();
    let reads = 0;
    const read = session.enqueueRead({}, {
      get batch(): BatchScope {
        reads += 1;
        return reads === 1 ? entered : decoy;
      },
    });
    // 入った区間が `entered` であることの観測: `decoy` で enter していれば「別の GpuContext で
    // 開いた batch には enqueue できない」で受理そのものが落ちる。
    assertEquals(await outcomeOf(read.admitted), { state: "resolved" });
    // `#readBatch` も `entered` のまま = 決着前の後続 enqueue は拒まれる。
    const refused = await rejectionOf(session.enqueue({}, { batch: entered }));
    assertInstanceOf(refused, BatchScopeError);
    assertStringIncludes(refused.message, "enqueueRead を積んだ batch には");
    assertEquals(await outcomeOf(entered.finish()), { state: "resolved" });
    // 決着後は同じ区間ではなくなる（門が解除されることの逆側）。
    assertEquals(await outcomeOf(decoy.finish()), { state: "resolved" });
    await session.dispose();
  });

  it("2 度目以降で throw する getter でも、enqueueRead は同期 throw しない", async () => {
    const gpu = gpuContext();
    const session = await emptySession(gpu);
    const batch = await gpu.beginBatch();
    let reads = 0;
    // 同期 throw が漏れると、この呼び出し自体がここで落ちる（戻り値を受け取れない）。
    const read = session.enqueueRead({}, {
      get batch(): BatchScope {
        reads += 1;
        if (reads >= 2) throw new Error("2 度目の読み");
        return batch;
      },
    });
    assertEquals(await outcomeOf(read.admitted), { state: "resolved" });
    assertEquals(await outcomeOf(batch.finish()), { state: "resolved" });
    await session.dispose();
  });

  it("最初の読みで throw する getter は、同期 throw ではなく戻り Promise の reject になる", async () => {
    const gpu = gpuContext();
    const session = await emptySession(gpu);
    const batch = await gpu.beginBatch();
    const read = session.enqueueRead({}, {
      get batch(): BatchScope {
        throw new Error("最初の読み");
      },
    });
    assertInstanceOf(await rejectionOf(read.admitted), Error);
    assertInstanceOf(await rejectionOf(read.outputs), Error);
    // 受理に落ちた enqueue は区間へ 1 本も入っていない（リースが残らず finish は決着する）。
    assertEquals(await outcomeOf(batch.finish()), { state: "resolved" });
    await session.dispose();
  });

  it("getter の中から同じ Session の run を発行しても、受理と区間は決着する", async () => {
    const gpu = gpuContext();
    const session = await emptySession(gpu);
    const batch = await gpu.beginBatch();
    const issued: Promise<unknown>[] = [];
    const read = session.enqueueRead({}, {
      get batch(): BatchScope {
        if (issued.length === 0) issued.push(session.run({}));
        return batch;
      },
    });
    // 未決着 run を持つ Session の enqueue は自己デッドロックなので、受け口が型付き例外へ変換する。
    const refused = await rejectionOf(read.admitted);
    assertInstanceOf(refused, BatchScopeError);
    assertStringIncludes(refused.message, "未決着の run が");
    // リースを取る前に落ちているので区間は決着する。
    assertEquals(await outcomeOf(batch.finish()), { state: "resolved" });
    // run も区間ロックを取れるところまで進む（本体はフェイク device にコマンド列を作る口が
    // 無いので落ちる — ここで固定したいのは「ロック待ちのまま永久に pending にならない」こと）。
    assertEquals(issued.length, 1);
    assertNotEquals((await outcomeOf(issued[0])).state, "pending");
    await session.dispose();
  });
});
