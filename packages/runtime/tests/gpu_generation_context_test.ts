// GenerationContext（1 生成ぶんの可変 state の所有者 — ADR 0066）の門。
//
// 波 C の時点では **state を読み書きする dispatch がまだ無い**（ノードの states 欄・
// state_append・attention 統合は波 D）。したがってここが固定するのは値ではなく契約:
// ①所有権と寿命（確保 → dispose・二重 dispose・Session との順序独立）②受け口のゲート
// （states 無しグラフ・chunkLength・記号容量・容量の束縛上限）③確保失敗が errorScope で
// fail loudly になること ④論理長の進行と巻き戻しの境界 ⑤汚染と device 消失の拒否
// ⑥論理長 uniform が**実際に GPU 上へ載っている**こと ⑦context が計画鍵に一切効かないこと。
//
// ⑥ が無いと writeBuffer の no-op（無効バッファ・整列違反では警告すら出ない）を検出できない。
// ⑦ が無いと、波 D で context を鍵に混ぜる実装が「値は正しいまま decode が毎 step 再導出」と
// いう形で通ってしまう（例外も警告も出ない）。

import { assert, assertEquals, assertRejects, assertStrictEquals, assertThrows } from "@std/assert";
import { openModel } from "../src/format/container.ts";
import type { IrGraph } from "../src/format/ir.ts";
import {
  acquireGpu,
  type GpuContext,
  GpuDeviceLostError,
  GpuOutOfMemoryError,
  LIMIT_CAPS,
  RUNTIME_INTERNAL,
} from "../src/gpu/device.ts";
import { createSession, type Session, type Tensor } from "../src/runtime/executor.ts";
import {
  GenerationContext,
  type GenerationContextHost,
} from "../src/runtime/generation-context.ts";
import { OpContractError } from "../src/ops.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import { baseGraph, f32Bytes, type GraphJson } from "./helpers/format.ts";
import { graphModelBuffer } from "./helpers/graph.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

/** WebGPU core 既定のストレージ束縛上限（容量ゲートの門はここを再現する）。 */
const CORE_STORAGE_BINDING_LIMIT = 128 * 1024 * 1024;

/** 論理長 uniform のバイト数（pastLength / queryLength の 2 語 — ADR 0066 追記 4）。 */
const LENGTHS_BYTES = 8;

/** 数値容量の state スロット 2 本（k / v = [1,2,C,4]）を持つグラフ。y = x·w。 */
const STATE_CAPACITY = 16;
const SLOT_BYTES = 1 * 2 * STATE_CAPACITY * 4 * 4;

/**
 * `state_append` の入力（今 step の chunk — 宣言 shape は `[B,Hkv,M,D]` の rank-4 MUST）。
 * スロット `[1,2,C,4]` と B / Hkv / D が一致する形（ADR 0067 決定 4 の②）。
 */
const CHUNK_SHAPE: readonly number[] = [1, 2, 4, 4];

/**
 * 宣言したスロットは**必ずノードから参照される**（参照完全性 — ADR 0067 決定 4 / 5）ので、
 * スロット 1 本につき最小の `state_append` を 1 本置く。波 C の時点では「宣言だけのグラフ」で
 * 足りていたが、参照側の欄が入った今それは孤立宣言として拒否される。
 */
const stateGraph = (
  states: GraphJson["states"],
  extra: Partial<GraphJson> = {},
): GraphJson => {
  const slots = Object.keys(states ?? {});
  return {
    format: "karume-ir",
    version: 1,
    requires: { ops: slots.length > 0 ? ["matmul", "state_append"] : ["matmul"] },
    symbols: ["T"],
    inputs: [{ name: "x", dtype: "f32", shape: ["T", 4] }],
    outputs: ["y"],
    initializers: {
      w: { tensor: "proj.weight", storage: { dtype: "f32" } },
      chunk: { tensor: "kv.chunk", storage: { dtype: "f32" } },
    },
    values: {
      w: { dtype: "f32", shape: [4, 3] },
      y: { dtype: "f32", shape: ["T", 3] },
      chunk: { dtype: "f32", shape: [...CHUNK_SHAPE] },
    },
    states,
    nodes: [
      { op: "matmul", ins: ["x", "w"], outs: ["y"], attrs: {} },
      ...slots.map((slot) => ({
        op: "state_append",
        ins: ["chunk"],
        outs: [],
        attrs: {},
        states: { slot },
      })),
    ],
    ...extra,
  };
};

const NUMERIC_STATES: GraphJson["states"] = {
  k: { dtype: "f32", shape: [1, 2, STATE_CAPACITY, 4] },
  v: { dtype: "f32", shape: [1, 2, STATE_CAPACITY, 4] },
};

/**
 * 記号容量のグラフ。容量記号 `C` は**どの入力にも現れない** — 束縛点は
 * `createGenerationContext(spec.bindings)` の側で（ADR 0066 追記 7 の束縛点 2）、値 shape の
 * 解決に使われることも無い。波 C では入力 `cap` を置いて束縛可能性を満たしていたが、波 D-1 で
 * 検査が「入力 shape ∪ states shape」へ緩んだので**素の states 専用記号**として書ける。
 */
const symbolicGraph = (): GraphJson =>
  stateGraph({ k: { dtype: "f32", shape: [1, 2, "C", 4] } }, { symbols: ["T", "C"] });

const modelBytes = (graph: GraphJson): ArrayBuffer =>
  graphModelBuffer(graph, [
    {
      name: "proj.weight",
      dtype: "F32",
      shape: [4, 3],
      data: f32Bytes([0.5, -1.5, 2, 0.25, -0.75, 1, 0.125, -0.25, 1.5, 2, -1, 0.75]),
    },
    {
      name: "kv.chunk",
      dtype: "F32",
      shape: [...CHUNK_SHAPE],
      data: f32Bytes(new Array(1 * 2 * 4 * 4).fill(0)),
    },
  ]);

const stateSession = (
  gpu: GpuContext,
  graph: GraphJson = stateGraph(NUMERIC_STATES),
): Promise<Session> => createSession(gpu, openModel(modelBytes(graph)));

/** 内部面（波 D の実行統合が呼ぶ進行・汚染・搬送路）をテストから駆動する。 */
const internals = (context: GenerationContext) => context[RUNTIME_INTERNAL];

/** 論理長の上限（搬送先が Uint32Array / WGSL u32 — ADR 0066 追記 4）。 */
const U32_MAX = 0xffffffff;

/** 注入する故障（確保途中の同期 throw / pop の reject / pop が決着しない窓）。 */
type Fault = {
  readonly failCreateAt?: number;
  readonly popError?: Error;
  readonly hangPop?: boolean;
};

/**
 * 実 device に proxy を被せて確保経路へ故障を注入し、**返したバッファの destroy 回数を確保順に
 * 数える**（漏れ = 0 回・二重 destroy = 2 回として同じ観測点で見える）。
 *
 * 注入面を生産コードに開けないための代替。OOM 門（下の 64GiB）は「余力を使い切って取れなくなる」
 * ことでしか漏れを検出できず、1GiB 未満の機では丸ごと SKIP される上に、途中の N 本目・pop の
 * reject・消失を撃ち分けられない。
 */
const injectFaults = (
  gpu: GpuContext,
  fault: Fault,
): { readonly gpu: GpuContext; readonly destroyCounts: number[] } => {
  const destroyCounts: number[] = [];
  let creations = 0;
  const device = new Proxy(gpu.device, {
    get(target, prop): unknown {
      if (prop === "createBuffer") {
        return (descriptor: GPUBufferDescriptor): GPUBuffer => {
          creations += 1;
          if (creations === fault.failCreateAt) {
            throw new Error(`注入: ${creations} 本目の createBuffer が同期 throw`);
          }
          const buffer = target.createBuffer(descriptor);
          const index = destroyCounts.push(0) - 1;
          return new Proxy(buffer, {
            get(inner, innerProp): unknown {
              if (innerProp === "destroy") {
                return (): void => {
                  destroyCounts[index] += 1;
                  inner.destroy();
                };
              }
              const value: unknown = Reflect.get(inner, innerProp);
              return typeof value === "function" ? value.bind(inner) : value;
            },
          });
        };
      }
      if (prop === "popErrorScope" && (fault.popError !== undefined || fault.hangPop === true)) {
        return (): Promise<GPUError | null> => {
          if (fault.hangPop === true) {
            // MUST: 決着しない窓は**実 pop を出さずに**作る。この case は直後に device を
            // 破棄するので LIFO の均衡は問われない（スタックごと消える）一方、消失後に
            // 決着しない実 op を残すと待ちの正体がテスト側の op になる。
            return new Promise<GPUError | null>(() => {});
          }
          // MUST: reject を注入する側は**実 pop を必ず発行する**。握り潰すと push した 2 本が
          // device の LIFO に残り、以後のテストの失敗が誤ったスコープに吸われる。
          void target.popErrorScope().catch(() => null);
          return Promise.reject(fault.popError);
        };
      }
      const value: unknown = Reflect.get(target, prop);
      return typeof value === "function" ? value.bind(target) : value;
    },
  });
  const proxied = new Proxy(gpu, {
    get(target, prop): unknown {
      if (prop === "device") return device;
      const value: unknown = Reflect.get(target, prop);
      return typeof value === "function" ? value.bind(target) : value;
    },
  });
  return { gpu: proxied, destroyCounts };
};

/** 注入した gpu の上へ context を直に建てる面（Session を通さない — 借りるのは数欄だけ）。 */
const injectedHost = (gpu: GpuContext, graph: IrGraph): GenerationContextHost => ({
  gpu,
  graph,
  flush: () => Promise.resolve(),
  serialize: <T>(body: () => Promise<T>): Promise<T> => body(),
  forget: () => {},
});

Deno.test({
  name: "createGenerationContext はスロットごとに物理確保し、dispose で返す（診断つき・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu);
    try {
      assertEquals(
        session.diagnostics().stateBacking,
        { residentBytes: 0, contextCount: 0, rebindCount: 0 },
        "context を作っていない Session は state バイトを 1 つも払わない",
      );

      const context = await session.createGenerationContext({ chunkLength: 4 });
      assertEquals(context.chunkLength, 4);
      assertEquals(context.pastLength, 0, "生成直後の論理長は 0");
      assertEquals(
        session.diagnostics().stateBacking,
        {
          residentBytes: SLOT_BYTES * 2 + LENGTHS_BYTES,
          contextCount: 1,
          rebindCount: 0,
        },
        "スロット 2 本 + 論理長 uniform ぶんが常駐する",
      );
      // 宣言 shape が束縛解決済みの容量として実体に載っていること（サイズは GPUBuffer 側で確認 —
      // 4 バイト床や切り上げが混ざると arrayLength() が黙って変わる）。
      assertEquals([...internals(context).slots.keys()].sort(), ["k", "v"]);
      for (const [name, slot] of internals(context).slots) {
        assertEquals(slot.shape, [1, 2, STATE_CAPACITY, 4], name);
        assertEquals(slot.byteLength, SLOT_BYTES, name);
        assertEquals(slot.buffer.size, SLOT_BYTES, name);
      }
      assertEquals(internals(context).lengths.size, LENGTHS_BYTES);

      // 2 本目の context は独立の実体を持つ（1 本目の実体を共有しない）。
      const second = await session.createGenerationContext({ chunkLength: 1 });
      assertEquals(session.diagnostics().stateBacking.contextCount, 2);
      assertEquals(
        session.diagnostics().stateBacking.residentBytes,
        (SLOT_BYTES * 2 + LENGTHS_BYTES) * 2,
      );
      assert(
        internals(second).slots.get("k")?.buffer !== internals(context).slots.get("k")?.buffer,
        "context ごとに別のバッファでなければ 2 本目が 1 本目の KV を上書きする",
      );

      // 二重 dispose は無害（**同一の Promise** を返す — Session.dispose と同じ規律。別の
      // Promise を返す実装は「2 度目だけが flush の決着前に resolve する」形で退行しうる）。
      const disposal = context.dispose();
      const again = context.dispose();
      assertStrictEquals(again, disposal, "2 度目以降は最初の完了そのものを返す");
      assertEquals(await Promise.all([disposal, again]), [undefined, undefined]);
      assertEquals(
        session.diagnostics().stateBacking,
        {
          residentBytes: SLOT_BYTES * 2 + LENGTHS_BYTES,
          contextCount: 2,
          rebindCount: 0,
        },
        "常駐バイト数は生存集合から導出（累計本数は減らない）",
      );

      // dispose 済み context の操作は fail loudly（読みも含む — 実体はもう無い）。
      assertThrows(() => context.pastLength, ExecutionError, "dispose 済み");
      assertThrows(() => context.rewind(0), ExecutionError, "dispose 済み");
      // 内部面は dispose の**2 段目**（破棄本体が走った後）で閉じる — ここは await 済みなので
      // 2 段目まで済んでいる。
      assertThrows(() => internals(context).advance(0, 1), ExecutionError, "dispose 済み");
      assertThrows(() => internals(context).pastLength(), ExecutionError, "dispose 済み");

      // context は Session より長生きできる（dispose の順序に依存を作らない）。
      await session.dispose();
      await second.dispose();
      assertEquals(session.diagnostics().stateBacking.residentBytes, 0);
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "states 宣言を持たないグラフでは GenerationContext を作れない（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await createSession(
      gpu,
      openModel(graphModelBuffer(baseGraph(), [
        { name: "enc.w", dtype: "F32", shape: [4, 3], data: f32Bytes(new Array(12).fill(0.5)) },
        { name: "enc.b", dtype: "F32", shape: [3], data: f32Bytes([1, 2, 3]) },
      ])),
    );
    try {
      const error = await assertRejects(
        () => session.createGenerationContext({ chunkLength: 1 }),
        ExecutionError,
      );
      assert(error.message.includes("states 宣言を持たない"), error.message);
      assertEquals(session.diagnostics().stateBacking.contextCount, 0);
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "spec の検査: chunkLength は 1 以上の整数・記号容量は bindings だけで決まる（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu);
    const symbolic = await stateSession(gpu, symbolicGraph());
    try {
      for (const chunkLength of [0, -1, 1.5, Number.NaN]) {
        const error = await assertRejects(
          () => session.createGenerationContext({ chunkLength }),
          ExecutionError,
        );
        assert(error.message.includes("chunkLength"), error.message);
      }

      // 未束縛の記号容量は fail loudly（0 や 1 で埋めない）。
      const unbound = await assertRejects(
        () => symbolic.createGenerationContext({ chunkLength: 2 }),
        ExecutionError,
      );
      assert(unbound.message.includes("'C' が束縛されていない"), unbound.message);

      // symbols に無い束縛も fail loudly（綴り違いが黙って無視されない）。
      const unknown = await assertRejects(
        () => symbolic.createGenerationContext({ chunkLength: 2, bindings: { D: 8 } }),
        ExecutionError,
      );
      assert(unknown.message.includes("束縛 'D'"), unknown.message);

      // 容量 0 の束縛は「実体を持てないスロット」なので拒否。
      const empty = await assertRejects(
        () => symbolic.createGenerationContext({ chunkLength: 2, bindings: { C: 0 } }),
        ExecutionError,
      );
      assert(empty.message.includes("容量 0"), empty.message);

      // 解決できた記号容量はそのまま物理容量になる（入力 `cap` は 1 度も渡していない）。
      const context = await symbolic.createGenerationContext({
        chunkLength: 2,
        bindings: { C: 8 },
      });
      try {
        const slot = internals(context).slots.get("k");
        assertEquals(slot?.shape, [1, 2, 8, 4]);
        assertEquals(slot?.byteLength, 1 * 2 * 8 * 4 * 4);
      } finally {
        await context.dispose();
      }
    } finally {
      await session.dispose();
      await symbolic.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name:
    "スロット単体が maxStorageBufferBindingSize を超える容量は fail loudly（絞った device・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // Gemma 4 E2B の full 層（131K 容量 × f32）と同型の 1GiB スロット。core 既定 128MiB の機では
    // 束縛できない = ADR 0066 追記 5 が拒否を要求する形。
    const capacity = 131072;
    const heads = 8;
    const headDim = 256;
    const slotBytes = heads * capacity * headDim * 4;
    assert(
      slotBytes > CORE_STORAGE_BINDING_LIMIT,
      `合成スロットの ${slotBytes}B が core 既定の上限を超えていない（門が空振りする）`,
    );
    const gpu = await acquireGpu({
      [LIMIT_CAPS]: { maxStorageBufferBindingSize: CORE_STORAGE_BINDING_LIMIT },
    });
    try {
      assertEquals(
        gpu.limits.maxStorageBufferBindingSize,
        CORE_STORAGE_BINDING_LIMIT,
        "requiredLimits が絞られていない（絞れていなければ門は何も見ていない）",
      );
      const session = await stateSession(
        gpu,
        stateGraph({ big: { dtype: "f32", shape: [1, heads, capacity, headDim] } }),
      );
      try {
        const error = await assertRejects(
          () => session.createGenerationContext({ chunkLength: 1 }),
          ExecutionError,
        );
        assert(error.message.includes(`${slotBytes} バイト`), error.message);
        assert(
          error.message.includes(`${CORE_STORAGE_BINDING_LIMIT} バイト`),
          error.message,
        );
        assertEquals(
          session.diagnostics().stateBacking.contextCount,
          0,
          "拒否された context は数えない（診断が実体の無いぶんを主張しない）",
        );
      } finally {
        await session.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
});

/**
 * OOM 門の前提: スロット 1 本（1GiB）を束縛できるアダプタで、**合計 64GiB** の要求が device の
 * 余力を超えること。前者は絞れない（`LIMIT_CAPS` は上限を下げる向きだけ）ので、束縛上限が
 * 1GiB 未満のアダプタでは容量ゲート側が先に落ちて OOM 経路を見られない = 明示 SKIP する。
 * 後者を満たさない機（64GiB 超の VRAM）ではこの門は**赤くなる**（黙って通らない）。
 */
const OOM_SLOT_BYTES = 1024 * 1024 * 1024;
const OOM_SLOT_COUNT = 64;
const detectBindingLimit = async (): Promise<number> => {
  const adapter = await navigator.gpu?.requestAdapter();
  return adapter === null || adapter === undefined ? 0 : adapter.limits.maxStorageBufferBindingSize;
};
const BINDING_LIMIT: number = GPU_AVAILABLE ? await detectBindingLimit() : 0;
if (GPU_AVAILABLE && BINDING_LIMIT < OOM_SLOT_BYTES) {
  console.warn(
    "[karume] maxStorageBufferBindingSize が 1GiB 未満のアダプタでは GenerationContext の " +
      "OOM 門を SKIP する（容量ゲートが先に落ちるため確保失敗の経路を通れない）",
  );
}

Deno.test({
  name: "state 確保の失敗は out-of-memory errorScope で fail loudly（実 GPU）",
  ignore: !GPU_AVAILABLE || BINDING_LIMIT < OOM_SLOT_BYTES,
  fn: async () => {
    const elements = OOM_SLOT_BYTES / 4;
    const states: GraphJson["states"] = {};
    for (let i = 0; i < OOM_SLOT_COUNT; i += 1) {
      states[`kv${i}`] = { dtype: "f32", shape: [1, 1, elements / 1024, 1024] };
    }
    const gpu = await acquireGpu();
    const session = await stateSession(gpu, stateGraph(states));
    // 失敗経路の後始末の検出器: OOM は「余力を使い切った」状態なので、確保済みを返し損ねていれば
    // この 1GiB は取れない（漏れた実体は dispose からも到達できず、量はこの Session で最大になる）。
    const survivor = await stateSession(
      gpu,
      stateGraph({ kv: { dtype: "f32", shape: [1, 1, elements / 1024, 1024] } }),
    );
    try {
      const error = await assertRejects(
        () => session.createGenerationContext({ chunkLength: 1 }),
        GpuOutOfMemoryError,
      );
      assert(error.message.startsWith("GenerationContext の state 確保: "), error.message);
      assertEquals(session.diagnostics().stateBacking.contextCount, 0);
      const context = await survivor.createGenerationContext({ chunkLength: 1 });
      assertEquals(context.pastLength, 0);
      await context.dispose();
    } finally {
      await session.dispose();
      await survivor.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "確保途中の失敗は確保済みバッファをちょうど 1 回ずつ返す（故障注入・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const graph = openModel(modelBytes(stateGraph(NUMERIC_STATES))).graph;
    const create = (host: GenerationContextHost): Promise<GenerationContext> =>
      GenerationContext.create(host, { chunkLength: 1 });
    try {
      // 成功経路は 1 本も destroy しない（所有者は返した context — 返るのは dispose だけ）。
      const ok = injectFaults(gpu, {});
      const context = await create(injectedHost(ok.gpu, graph));
      assertEquals(ok.destroyCounts, [0, 0, 0], "スロット 2 本 + 論理長 uniform を確保した");
      await context.dispose();
      assertEquals(ok.destroyCounts, [1, 1, 1], "dispose でちょうど 1 回ずつ返る");

      // ① 途中の同期 throw: 2 本目で落ちても 1 本目がちょうど 1 回返る。
      const midway = injectFaults(gpu, { failCreateAt: 2 });
      await assertRejects(
        () => create(injectedHost(midway.gpu, graph)),
        Error,
        "注入: 2 本目の createBuffer",
      );
      assertEquals(midway.destroyCounts, [1], "確保済みの 1 本が漏れない");

      // ② pop の reject: 全確保物がちょうど 1 回返る（pop を try/finally の外に置いた形では
      // ここで 3 本とも到達不能になり、dispose からも返せない）。
      const rejected = injectFaults(gpu, { popError: new Error("注入: popErrorScope の reject") });
      await assertRejects(
        () => create(injectedHost(rejected.gpu, graph)),
        Error,
        "注入: popErrorScope の reject",
      );
      assertEquals(rejected.destroyCounts, [1, 1, 1]);
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "確保中の device 消失は GpuDeviceLostError にして全確保物を返す（故障注入・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // この case は device を壊すので専用の GpuContext を取る。
    const gpu = await acquireGpu();
    const graph = openModel(modelBytes(stateGraph(NUMERIC_STATES))).graph;
    // pop が決着しない窓を作る。待機を raceDeviceLost に通していなければ、この case は
    // 「失敗」ではなく**ハング**になる（消失後の popErrorScope が解決しない実装がありうる）。
    const injected = injectFaults(gpu, { hangPop: true });
    const pending = GenerationContext.create(injectedHost(injected.gpu, graph), { chunkLength: 1 });
    assertEquals(injected.destroyCounts, [0, 0, 0], "確保は create の同期区間で終わっている");

    gpu.destroy();
    const error = await assertRejects(() => pending, GpuDeviceLostError);
    assert(error.message.includes("GenerationContext の state 確保"), error.message);
    assertEquals(injected.destroyCounts, [1, 1, 1], "消失でも確保物は 1 本残らず返る");
  },
});

Deno.test({
  name: "論理長は run の成功で進み、rewind は 0..pastLength の整数だけを受ける（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu);
    const context = await session.createGenerationContext({ chunkLength: 4 });
    try {
      // prefill chunk（queryLength ≤ chunkLength）と decode（1）の両方で進む。
      // 第 1 引数は run が頭で捕捉した pastLength（context の現在値と照合される）。
      internals(context).advance(0, 4);
      internals(context).advance(4, 1);
      assertEquals(context.pastLength, 5);

      for (const queryLength of [0, -1, 1.5, 5]) {
        assertThrows(
          () => internals(context).advance(5, queryLength),
          ExecutionError,
          "queryLength",
        );
      }
      assertEquals(context.pastLength, 5, "拒否された進行は論理長を動かさない");

      // 捕捉 P が現在値と割れた進行は fail loudly（リースがあれば起きない = 内部の不変条件破れ）。
      for (const captured of [4, 6]) {
        const split = assertThrows(
          () => internals(context).advance(captured, 1),
          ExecutionError,
        );
        assert(split.message.includes("食い違う"), split.message);
      }
      assertThrows(() => internals(context).writeLengths(4, 1), ExecutionError, "食い違う");
      assertEquals(context.pastLength, 5, "拒否された進行は論理長を動かさない");

      context.rewind(2);
      assertEquals(context.pastLength, 2);
      context.rewind(2);
      assertEquals(context.pastLength, 2, "同じ位置への rewind は no-op");
      context.rewind(0);
      assertEquals(context.pastLength, 0);

      internals(context).advance(0, 3);
      for (const position of [-1, 4, 1.5, Number.NaN]) {
        assertThrows(() => context.rewind(position), ExecutionError, "rewind");
      }
      assertEquals(context.pastLength, 3, "拒否された rewind は論理長を動かさない");
    } finally {
      await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "論理長は u32 の上限で fail loudly（Uint32Array への沈黙切り詰めの遮断・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu);
    try {
      // 上限ちょうどは受理（chunkLength は計画時定数で、物理確保はスロット容量だけが決める）。
      const context = await session.createGenerationContext({ chunkLength: U32_MAX });
      try {
        assertEquals(context.chunkLength, U32_MAX);
        // u32 を超える queryLength は chunkLength の門で落ちる（0 に切り詰めて書かない）。
        assertThrows(
          () => internals(context).writeLengths(0, U32_MAX + 1),
          ExecutionError,
          "chunkLength",
        );
        // 加算後の overflow はここでしか見られない（両項が u32 以下でも和は溢れる）。
        internals(context).advance(0, U32_MAX);
        assertEquals(context.pastLength, U32_MAX);
        const overflow = assertThrows(
          () => internals(context).advance(U32_MAX, 1),
          ExecutionError,
        );
        assert(overflow.message.includes("u32 の上限"), overflow.message);
        assertEquals(context.pastLength, U32_MAX, "拒否された進行は論理長を動かさない");
        // rewind の位置も pastLength を上限に持つので u32 を超えられない。
        assertThrows(() => context.rewind(U32_MAX + 1), ExecutionError, "rewind");
      } finally {
        await context.dispose();
      }

      // 上限 +1 の chunkLength は拒否（受理すると queryLength の門が u32 を超えて開く）。
      const rejected = await assertRejects(
        () => session.createGenerationContext({ chunkLength: U32_MAX + 1 }),
        ExecutionError,
      );
      assert(rejected.message.includes("chunkLength"), rejected.message);
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "汚染後は dispose 以外の全操作を拒否する（読みも含む・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu);
    const context = await session.createGenerationContext({ chunkLength: 2 });
    try {
      internals(context).advance(0, 2);
      internals(context).poison("state 変更 dispatch を含む run が失敗した");
      // 2 度目の汚染は真因を上書きしない。
      internals(context).poison("後続の別の失敗");

      // ADR 0066 追記 3 の「以後の全操作 fail loudly」— `pastLength` の読みも遮断面に入る
      // （復旧不能な context の論理長を正常値として返すと、ホストは必ず「ここから再開できる」
      // と読む）。
      for (
        const operation of [
          () => context.pastLength,
          () => context.rewind(0),
          () => internals(context).pastLength(),
          () => internals(context).acquireRun(),
          () => internals(context).advance(2, 1),
          () => internals(context).writeLengths(2, 1),
        ]
      ) {
        const error = assertThrows(operation, ExecutionError);
        assert(error.message.includes("汚染された"), error.message);
        assert(
          error.message.includes("state 変更 dispatch を含む run が失敗した"),
          `真因が残っていない: ${error.message}`,
        );
      }
    } finally {
      // 汚染された context も dispose は通る（物理バッファは返せる）。
      await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "device 消失後の GenerationContext は全操作が GpuDeviceLostError（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // この case は device を壊すので専用の GpuContext を取る。
    const gpu = await acquireGpu();
    const session = await stateSession(gpu);
    const context = await session.createGenerationContext({ chunkLength: 2 });
    internals(context).advance(0, 1);

    gpu.destroy();
    await gpu.device.lost;
    assert(gpu.lost !== undefined, "消失が記録されていない（門が空振りする）");

    assertThrows(() => context.pastLength, GpuDeviceLostError, "device が失われた");
    assertThrows(() => context.rewind(0), GpuDeviceLostError);
    assertThrows(() => internals(context).acquireRun(), GpuDeviceLostError);
    assertThrows(() => internals(context).advance(1, 1), GpuDeviceLostError);
    assertThrows(() => internals(context).writeLengths(1, 1), GpuDeviceLostError);

    // dispose は「バッファを返してから flush の失敗を伝播させる」（RunArena と同じ規律）。
    const failure = await assertRejects(() => context.dispose(), GpuDeviceLostError);
    const again = await assertRejects(() => context.dispose(), GpuDeviceLostError);
    assertStrictEquals(again, failure, "2 度目以降も同じ完了（同じ失敗）を返す");
    await session.dispose().catch(() => undefined);
  },
});

Deno.test({
  name: "device.destroy() 直後（lost の反応前）から全操作を拒否する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // この case は device を壊すので専用の GpuContext を取る。
    const gpu = await acquireGpu();
    const session = await stateSession(gpu);
    const context = await session.createGenerationContext({ chunkLength: 2 });
    internals(context).advance(0, 1);

    gpu.destroy();
    // MUST: ここで lost が**未記録**であること。記録済みなら既存の lost 判定だけで通ってしまい、
    // 「destroy の同期フラグを見ている」ことの証拠にならない（門が恒真になる）。
    assertStrictEquals(gpu.lost, undefined, "destroy 直後に lost の reaction が走っている");
    assert(gpu.destroyRequested, "destroy 要求が同期に立っていない");

    assertThrows(() => context.pastLength, GpuDeviceLostError, "device が失われた");
    assertThrows(() => context.rewind(0), GpuDeviceLostError);
    assertThrows(() => internals(context).advance(1, 1), GpuDeviceLostError);
    // writeLengths が最も危険な穴（破棄済みバッファへの writeBuffer は警告すら出ない no-op で、
    // ホストの論理長と GPU が見る値が黙って分裂する）。
    assertThrows(() => internals(context).writeLengths(1, 1), GpuDeviceLostError);
    // 生成の入口も同じ判定（空の KV を持つ context を作らせない）。
    await assertRejects(
      () => session.createGenerationContext({ chunkLength: 1 }),
      GpuDeviceLostError,
    );

    await session.dispose().catch(() => undefined);
  },
});

Deno.test({
  name: "論理長 uniform は毎回全域が書き直され、書いた値が GPU 上に載る（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu);
    const context = await session.createGenerationContext({ chunkLength: 4 });
    /** 論理長 uniform を staging 経由で読み戻す（writeBuffer の no-op を検出する唯一の観測点）。 */
    const readLengths = async (): Promise<readonly number[]> => {
      const staging = gpu.device.createBuffer({
        size: LENGTHS_BYTES,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      try {
        const encoder = gpu.device.createCommandEncoder();
        encoder.copyBufferToBuffer(internals(context).lengths, 0, staging, 0, LENGTHS_BYTES);
        gpu.device.queue.submit([encoder.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        const words = Array.from(new Uint32Array(staging.getMappedRange().slice(0)));
        staging.unmap();
        return words;
      } finally {
        staging.destroy();
      }
    };
    try {
      internals(context).writeLengths(0, 4);
      assertEquals(await readLengths(), [0, 4], "prefill 1 本目は past 0 / query 4");

      internals(context).advance(0, 4);
      internals(context).writeLengths(4, 3);
      assertEquals(await readLengths(), [4, 3], "past と query は独立のスカラ");

      // decode 形（queryLength=1）へ移ると query 語だけが変わる = 全域を書き直している証拠。
      internals(context).advance(4, 3);
      internals(context).writeLengths(7, 1);
      assertEquals(await readLengths(), [7, 1]);

      // rewind は次の writeLengths からそのまま効く（論理長は context が所有する）。
      context.rewind(2);
      internals(context).writeLengths(2, 1);
      assertEquals(await readLengths(), [2, 1]);

      for (const queryLength of [0, 5, 1.5]) {
        assertThrows(() => internals(context).writeLengths(2, queryLength), ExecutionError);
      }
      assertEquals(await readLengths(), [2, 1], "拒否された書き出しは uniform を触らない");
    } finally {
      await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
});

// 波 C ではここが「context を 2 本作っても 2 回目の run が同じ計画鍵に当たる」を見ていた。
// 波 D-1 で参照完全性が入り、**states を宣言するグラフは必ず state 参照ノードを持つ**ように
// なったため、1-shot の `run` はもう成立しない（スロットの実体を持つのは GenerationContext で、
// 実行経路の結線は波 D-3）。ここが固定するのは 2 点へ組み替えた:
// ①context 生成そのものが計画導出を 1 回も走らせない（鍵に context が載っていれば導出が動く）
// ②state 参照ノードを持つグラフの 1-shot 実行は fail loudly（黙って state 抜きで走らない）
// NOTE: 「同じ鍵に当たる（hit: true）」= ADR 0066 受入条件③の run 側は波 D-3 で結線され、
// tests/gpu_state_execution_test.ts が「別 context・同容量なら 2 本目が hit」の形で持っている。
Deno.test({
  name: "context 生成は計画導出を走らせず、state 参照グラフの 1-shot 実行は拒否される（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu);
    const input: Tensor = {
      dtype: "f32",
      shape: [2, 4],
      data: Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8]),
    };
    try {
      const baseline = session.diagnostics();
      assertEquals(baseline.lastRunPrepared, undefined, "run 前は計画の実績が無い");

      const contexts: GenerationContext[] = [
        await session.createGenerationContext({ chunkLength: 4 }),
        await session.createGenerationContext({ chunkLength: 1 }),
      ];
      try {
        const after = session.diagnostics();
        assertEquals(after.lastRunPrepared, undefined, "context 生成は計画導出を走らせない");
        assertEquals(after.planBacking.buildCount, 0, "slot backing も焼かれない");
        assertEquals(after.stateBacking.contextCount, 2, "診断だけが context を数える");

        // 落ちるのは shape 計算層（OpContractError）— スロットの解決済み shape が無い run は
        // そこまで進めない。
        const error = await assertRejects(() => session.run({ x: input }), OpContractError);
        assert(error.message.includes("GenerationContext"), error.message);
        assertEquals(
          session.diagnostics().lastRunPrepared,
          undefined,
          "拒否された run は計画キャッシュを汚さない",
        );
      } finally {
        for (const context of contexts) await context.dispose();
      }
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});
