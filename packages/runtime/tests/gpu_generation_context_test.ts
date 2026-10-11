// GenerationContext（1 生成ぶんの可変 state の所有者 — ADR 0066）の門。
//
// 波 C の時点では **state を読み書きする dispatch がまだ無い**（ノードの states 欄・
// state_append・attention 統合は波 D）。したがってここが固定するのは値ではなく契約:
// ①所有権と寿命（確保 → dispose・二重 dispose・Session との順序独立）②受け口のゲート
// （states 無しグラフ・chunkLength・記号容量・容量の束縛上限）③確保失敗が errorScope で
// fail loudly になること ④論理長の進行と巻き戻しの境界 ⑤汚染と device 消失の拒否
// ⑥論理長 uniform が**実際に GPU 上へ載っている**こと ⑦context が計画鍵に一切効かないこと
// ⑧`Session.run` の第 3 引数の寿命契約（発行時 snapshot / 同一 context への並行発行の拒否）。
// ⑨`fork`（会話状態の分岐 — ADR 0066 追記〈2026-10-11〉）: 写す中身・分岐の独立・同期区間の
// 契約・拒否の行列・診断（節の冒頭に詳細）。
//
// ⑥ が無いと writeBuffer の no-op（無効バッファ・整列違反では警告すら出ない）を検出できない。
// ⑦ が無いと、波 D で context を鍵に混ぜる実装が「値は正しいまま decode が毎 step 再導出」と
// いう形で通ってしまう（例外も警告も出ない）。

import {
  assert,
  assertEquals,
  assertNotEquals,
  assertRejects,
  assertStrictEquals,
  assertThrows,
} from "@std/assert";
import type { IrGraph } from "../src/format/ir.ts";
import {
  acquireGpu,
  type GpuContext,
  GpuDeviceLostError,
  LIMIT_CAPS,
  RUNTIME_INTERNAL,
} from "../src/gpu/device.ts";
import { GpuOutOfMemoryError } from "../src/gpu/error-scope.ts";
import {
  createSessionFromContainer,
  prepareContainer,
  type Session,
  type Tensor,
} from "../src/runtime/executor.ts";
import {
  GenerationContext,
  type GenerationContextHost,
} from "../src/runtime/generation-context.ts";
import { OpContractError } from "../src/ops.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import type { OpenedContainer } from "../src/format/container/open.ts";
import type { TensorInput } from "./helpers/container-write.ts";
import {
  baseDeclaration,
  baseTensors,
  type DeclarationJson,
  f32Bytes,
  GRAPH_NAME,
  openGraphModel,
  openModelBytes,
} from "./helpers/model-fixture.ts";
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
  states: DeclarationJson["states"],
  extra: Partial<DeclarationJson> = {},
  /** `state_append` ノードの attrs（sliding にするときだけ `{ window }` を渡す）。 */
  appendAttrs: Record<string, number> = {},
): DeclarationJson => {
  const slots = Object.keys(states ?? {});
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: slots.length > 0 ? ["matmul", "state_append"] : ["matmul"] },
    symbols: ["T"],
    inputs: [{ name: "x", dtype: "f32", shape: ["T", 4] }],
    outputs: ["y"],
    initializers: { w: {}, chunk: {} },
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
        attrs: { ...appendAttrs },
        states: { slot },
      })),
    ],
    ...extra,
  };
};

const NUMERIC_STATES: DeclarationJson["states"] = {
  k: { dtype: "f32", shape: [1, 2, STATE_CAPACITY, 4] },
  v: { dtype: "f32", shape: [1, 2, STATE_CAPACITY, 4] },
};

/**
 * 記号容量のグラフ。容量記号 `C` は**どの入力にも現れない** — 束縛点は
 * `createGenerationContext(spec.bindings)` の側で（ADR 0066 追記 7 の束縛点 2）、値 shape の
 * 解決に使われることも無い。波 C では入力 `cap` を置いて束縛可能性を満たしていたが、波 D-1 で
 * 検査が「入力 shape ∪ states shape」へ緩んだので**素の states 専用記号**として書ける。
 */
const symbolicGraph = (): DeclarationJson =>
  stateGraph({ k: { dtype: "f32", shape: [1, 2, "C", 4] } }, { symbols: ["T", "C"] });

/** 宣言した 2 本の initializer の実体（v2 では initializer 名がそのまま実体の鍵）。 */
const stateTensors = (): readonly TensorInput[] => [
  {
    graph: GRAPH_NAME,
    initializer: "w",
    bytes: f32Bytes([0.5, -1.5, 2, 0.25, -0.75, 1, 0.125, -0.25, 1.5, 2, -1, 0.75]),
    encoding: { codec: "f32" },
  },
  {
    graph: GRAPH_NAME,
    initializer: "chunk",
    bytes: f32Bytes(new Array(1 * 2 * 4 * 4).fill(0)),
    encoding: { codec: "f32" },
  },
];

const openStateModel = (graph: DeclarationJson): Promise<OpenedContainer> =>
  openModelBytes(graph, stateTensors());

const stateSession = async (
  gpu: GpuContext,
  graph: DeclarationJson = stateGraph(NUMERIC_STATES),
): Promise<Session> =>
  await createSessionFromContainer(gpu, await openStateModel(graph), GRAPH_NAME);

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

/**
 * 注入した gpu の上へ context を直に建てる面（Session を通さない — 借りるのは数欄だけ）。
 *
 * `adopt` は登録の簿記を持たない素通し（`Session.#adopt` の MUST どおり `allocate` を同期に呼ぶ）。
 * fork の故障注入だけが `adopt` を差し替える（{@link delayedAdoptHost}）。
 */
const injectedHost = (gpu: GpuContext, graph: IrGraph): GenerationContextHost => ({
  gpu,
  graph,
  flush: () => Promise.resolve(),
  serialize: <T>(body: () => Promise<T>): Promise<T> => body(),
  forget: () => {},
  adopt: (allocate) => allocate(),
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
  name: "確保の await を跨いで dispose された Session は context を握らせない（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu);
    try {
      // createGenerationContext は直列化チェーンに乗らない（乗せると batch 区間の未決着 run と
      // 同型の閉路に巻き込まれる）ので、呼び出し時点の dispose 判定を確保の await が跨ぐ。
      // 確保後に再判定しないと、dispose 済み Session の生存集合へ context が登録され、診断が
      // 実体の残る常駐を主張し続ける（バッファは利用者が context.dispose() を呼ぶまで残る）。
      const pending = session.createGenerationContext({ chunkLength: 4 });
      const disposal = session.dispose();
      await disposal;
      const error = await assertRejects(() => pending, ExecutionError);
      assert(error.message.includes("dispose 済みの Session"), error.message);
      assertEquals(
        session.diagnostics().stateBacking,
        { residentBytes: 0, contextCount: 0, rebindCount: 0 },
        "確保できたぶんは破棄され、生存集合にも累計にも載らない",
      );
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
    const session = await createSessionFromContainer(
      gpu,
      await openModelBytes(baseDeclaration(), baseTensors()),
      GRAPH_NAME,
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
    const states: DeclarationJson["states"] = {};
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
    const graph = prepareContainer(
      await openStateModel(stateGraph(NUMERIC_STATES)),
      GRAPH_NAME,
    ).graph;
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
    const graph = prepareContainer(
      await openStateModel(stateGraph(NUMERIC_STATES)),
      GRAPH_NAME,
    ).graph;
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
  name: "spec の chunkBuckets は凍結コピーで公開され、許可集合を組む（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu);
    try {
      // 呼び出し側の配列は**写す**（後から書き換えても許可集合と公開面が割れない）。
      const buckets = [2, 4];
      const context = await session.createGenerationContext({
        chunkLength: 8,
        chunkBuckets: buckets,
      });
      try {
        assertEquals(context.chunkBuckets, [2, 4]);
        assert(Object.isFrozen(context.chunkBuckets), "公開面は凍結コピー");
        buckets.push(6);
        assertEquals(context.chunkBuckets, [2, 4], "後からの書き換えは context に届かない");

        // 許可集合 = {1} ∪ chunkBuckets ∪ {chunkLength}（run 前検査が読む唯一の形）。
        assertEquals([...internals(context).allowedRows], [1, 2, 4, 8]);
        // chunkLength は最大値のまま（queryLength の上限も動かない）。
        assertEquals(context.chunkLength, 8);
        assertThrows(() => internals(context).writeLengths(0, 9), ExecutionError, "chunkLength");
      } finally {
        await context.dispose();
      }

      // 省略した context は従来どおり prefill 形 1 本 + decode 形。
      const plain = await session.createGenerationContext({ chunkLength: 8 });
      try {
        assertEquals(plain.chunkBuckets, []);
        assertEquals([...internals(plain).allowedRows], [1, 8]);
      } finally {
        await plain.dispose();
      }

      // 値域・順序の違反は確保の前に落ちる（受理集合の正本は assertChunkBuckets）。
      for (const chunkBuckets of [[1], [8], [4, 3]]) {
        const error = await assertRejects(
          () => session.createGenerationContext({ chunkLength: 8, chunkBuckets }),
          ExecutionError,
        );
        assert(error.message.includes("chunkBuckets"), error.message);
      }
      // 累計は上の 2 本のまま（拒否された spec は 1 本も確保していない）。
      assertEquals(session.diagnostics().stateBacking.contextCount, 2);
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "確保の await を跨いだ chunkBuckets の push は許可集合に届かない（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu);
    try {
      // 検査点（assertChunkBuckets）と constructor 渡しの間には state 確保の await があるので、
      // spec から読み直す形だと**この窓**の書き換えが未検査のまま許可集合へ載る。写しを持ち回る
      // 実装であることを、window の中で push して固定する（同期の push だけを見る上のテストとは
      // 別の窓 — こちらは create が中断している最中）。
      const buckets = [2, 4];
      const pending = session.createGenerationContext({ chunkLength: 8, chunkBuckets: buckets });
      buckets.push(6);
      const context = await pending;
      try {
        assertEquals(context.chunkBuckets, [2, 4], "確保の await を跨いだ push は届かない");
        assertEquals([...internals(context).allowedRows], [1, 2, 4, 8]);
      } finally {
        await context.dispose();
      }
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "確保の await を跨いだ chunkLength の書き換えは context に届かない（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu);
    try {
      // 配列と同じ窓を長さについても閉じる。`spec` から読み直す形だと、検査は 8 で通ったのに
      // context の上限が 1 になる（許可集合も {1} に縮む）— 例外も警告も出ないまま、以後の
      // prefill 全部が queryLength の門で落ちる形になる。
      const spec = { chunkLength: 8 };
      const pending = session.createGenerationContext(spec);
      spec.chunkLength = 1;
      const context = await pending;
      try {
        assertEquals(context.chunkLength, 8, "検査を通った値が context の上限");
        assertEquals([...internals(context).allowedRows], [1, 8]);
        // 上限が書き換え後の 1 に締まっていないこと（8 行の run は通る）。
        internals(context).writeLengths(0, 8);
      } finally {
        await context.dispose();
      }
    } finally {
      await session.dispose();
      gpu.destroy();
    }
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
          () => context.pendingCommit,
          () => context.rewind(0),
          () => context.commit(0),
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

// `Session.run` の第 3 引数の寿命契約（R6-1 / CG5-2）。run 本体は `#serialize` 越し
// （1 マイクロタスク以降）に走り、`context` / `queryLength` を await を跨いだ複数の時点で読むので、
// ここが固定するのは 2 点:
//   ①**発行時 snapshot**（`inputs` / `bindings` と同じ規律）— 発行後に引数オブジェクトを
//     書き換えても、この run は発行時点の 2 欄で走る（リースも同じ写しから立つ）。
//   ②**同一 context への未決着 run は 1 本まで**（2 本目は発行の同期区間で fail loudly）。
// どちらも破れは例外も警告も出ない位置ずれ / 論理長の分裂になるので、値ではなく契約で縛る。

/** 発行時 snapshot 用の可変な第 3 引数（`GenerationRun` の 2 欄は readonly なので写しを持つ）。 */
type MutableGenerationRun = { context: GenerationContext; queryLength: number };

const RUN_INPUT: Tensor = {
  dtype: "f32",
  shape: [2, 4],
  data: Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8]),
};

Deno.test({
  name: "run の queryLength は発行時に固定される（発行後の書き換えを読まない・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu);
    const context = await session.createGenerationContext({ chunkLength: 4 });
    try {
      const generation: MutableGenerationRun = { context, queryLength: 4 };
      const running = session.run({ x: RUN_INPUT }, {}, generation);
      // 本体が走る前（同じ tick）に書き換える。参照のまま持ち回る実装は、dispatch 数を 4 行ぶん
      // 撃ったまま uniform と進行だけ 1 になる（= GPU の走査範囲とホストの簿記が分裂する）。
      generation.queryLength = 1;
      await running;
      assertEquals(context.pastLength, 4, "発行時の queryLength で走り、その値だけ進む");
    } finally {
      await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "run の context は発行時に固定され、リースも同じ写しに立つ（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu);
    const first = await session.createGenerationContext({ chunkLength: 4 });
    const second = await session.createGenerationContext({ chunkLength: 4 });
    try {
      const generation: MutableGenerationRun = { context: first, queryLength: 1 };
      const running = session.run({ x: RUN_INPUT }, {}, generation);
      generation.context = second;
      // リースは発行時の写し（= first）に立つ。書き換え後もこちらが「進行中 run あり」で落ちる。
      const blocked = assertThrows(() => first.rewind(0), ExecutionError);
      assert(blocked.message.includes("進行中の generation run"), blocked.message);
      // 差し替え先は run に一切関与しないので、巻き戻しは素通りする（リースが漏れていない証拠）。
      second.rewind(0);
      await running;
      assertEquals(first.pastLength, 1, "KV を書いたのは発行時の context");
      assertEquals(second.pastLength, 0, "差し替え先は 1 行も進まない");
    } finally {
      await first.dispose();
      await second.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "同一 context への未 await 並行発行は 2 本目を拒否する（別 context は通る・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu);
    const context = await session.createGenerationContext({ chunkLength: 4 });
    const other = await session.createGenerationContext({ chunkLength: 4 });
    try {
      // MUST: 2 本まとめて settle させる（拒否は発行の同期区間で立つので、await を挟むと
      // 未処理拒否として観測される前にテストが落ちる）。
      const [first, second] = await Promise.allSettled([
        session.run({ x: RUN_INPUT }, {}, { context, queryLength: 1 }),
        session.run({ x: RUN_INPUT }, {}, { context, queryLength: 1 }),
      ]);
      assertEquals(first.status, "fulfilled", "1 本目は従来どおり通る");
      assertEquals(second.status, "rejected");
      assert(second.status === "rejected");
      assert(
        second.reason instanceof ExecutionError,
        `拒否の型が違う: ${String(second.reason)}`,
      );
      assert(second.reason.message.includes("並行発行"), second.reason.message);
      assertEquals(context.pastLength, 1, "進んだのは 1 本目のぶんだけ");

      // 別 context への並行発行は従来どおり成立する（塞ぐのは同一 context だけ）。
      const parallel = await Promise.allSettled([
        session.run({ x: RUN_INPUT }, {}, { context, queryLength: 1 }),
        session.run({ x: RUN_INPUT }, {}, { context: other, queryLength: 1 }),
      ]);
      assertEquals(parallel.map((result) => result.status), ["fulfilled", "fulfilled"]);
      assertEquals(context.pastLength, 2);
      assertEquals(other.pastLength, 1);
    } finally {
      await context.dispose();
      await other.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
});

// ---------------------------------------------------------------------------
// deferred commit（投機デコードの verify 形 — 論理長を run の成功では進めず、受理行数で確定する）
// ---------------------------------------------------------------------------

Deno.test({
  name: "deferred run は論理長を保留し、commit(rows) が受理行数だけ進める（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu);
    const context = await session.createGenerationContext({ chunkLength: 4 });
    /** 論理長 uniform を読み戻す（「GPU が見た past」が commit に追随することの観測点）。 */
    const readPast = async (): Promise<number> => {
      const staging = gpu.device.createBuffer({
        size: LENGTHS_BYTES,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      try {
        const encoder = gpu.device.createCommandEncoder();
        encoder.copyBufferToBuffer(internals(context).lengths, 0, staging, 0, LENGTHS_BYTES);
        gpu.device.queue.submit([encoder.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        const words = new Uint32Array(staging.getMappedRange().slice(0));
        staging.unmap();
        return words[0];
      } finally {
        staging.destroy();
      }
    };
    try {
      // 既定（commit 欄なし）は従来どおり run の成功で進む。
      await session.run({ x: RUN_INPUT }, {}, { context, queryLength: 1 });
      assertEquals(context.pastLength, 1);
      assertEquals(context.pendingCommit, undefined, "immediate な run は保留を作らない");

      // deferred は 3 行書いて論理長を止める。
      await session.run({ x: RUN_INPUT }, {}, { context, queryLength: 3, commit: "deferred" });
      assertEquals(context.pastLength, 1, "deferred は run の成功では進まない");
      assertEquals(context.pendingCommit, { pastLength: 1, queryLength: 3 });

      // 保留がある間は次の run も rewind も通さない（論理長を動かす経路は 1 本のまま）。
      const rejected = await assertRejects(
        () => session.run({ x: RUN_INPUT }, {}, { context, queryLength: 1 }),
        ExecutionError,
      );
      assert(rejected.message.includes("commit 待ち"), rejected.message);
      const blocked = assertThrows(() => context.rewind(0), ExecutionError);
      assert(blocked.message.includes("commit 待ち"), blocked.message);

      // 値域: 非負整数 / 書いた行数まで。拒否は保留も論理長も動かさない。
      for (const rows of [-1, 1.5, 4]) {
        assertThrows(() => context.commit(rows), ExecutionError, "commit");
      }
      assertEquals(context.pendingCommit, { pastLength: 1, queryLength: 3 });
      assertEquals(context.pastLength, 1);

      // 2 行受理 = 3 行目は棄却（物理 ring には書かれたまま論理長に載らない）。
      context.commit(2);
      assertEquals(context.pastLength, 3);
      assertEquals(context.pendingCommit, undefined, "commit は保留を畳む");

      // 保留が無い commit は fail loudly（黙って no-op にすると受理行数が二重に効く）。
      const orphan = assertThrows(() => context.commit(1), ExecutionError);
      assert(orphan.message.includes("確定待ちの generation run が無い"), orphan.message);

      // 次の run が載せる past は commit 後の値（ホストの簿記と GPU が見る値が一致している）。
      await session.run({ x: RUN_INPUT }, {}, { context, queryLength: 1 });
      assertEquals(await readPast(), 3, "GPU が見た past が commit を反映していない");
      assertEquals(context.pastLength, 4);

      // commit(0) は「1 行も受理しない」— 論理長は動かず保留だけが畳まれる。
      await session.run({ x: RUN_INPUT }, {}, { context, queryLength: 2, commit: "deferred" });
      context.commit(0);
      assertEquals(context.pastLength, 4, "commit(0) は論理長を動かさない");
      assertEquals(context.pendingCommit, undefined);

      // 型の外から来た綴り違いは既定へ倒さず落とす。
      const spelling = await assertRejects(
        () =>
          session.run(
            { x: RUN_INPUT },
            {},
            { context, queryLength: 1, commit: "defered" as "deferred" },
          ),
        ExecutionError,
      );
      assert(spelling.message.includes("'immediate' か 'deferred'"), spelling.message);
      assertEquals(context.pastLength, 4, "拒否された run は論理長も保留も動かさない");
      assertEquals(context.pendingCommit, undefined);
    } finally {
      await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "pendingCommit は凍結して返り、受理行数の上限は外から書き換えられない（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu);
    const context = await session.createGenerationContext({ chunkLength: 4 });
    try {
      await session.run({ x: RUN_INPUT }, {}, { context, queryLength: 2, commit: "deferred" });
      const pending = context.pendingCommit;
      assert(pending !== undefined, "deferred run が保留を立てていない");
      assert(Object.isFrozen(pending), "commit の上限を運ぶ object は凍結 MUST");
      // TS の readonly は代入互換性に効かないので、公開面はこの向きで**型検査なしに**書き換え
      // られる（`commit` はこの `queryLength` を「書いた行数」として信頼する）。凍結だけが
      // 検出線で、strict mode では代入そのものが TypeError になる。
      const writable: { pastLength: number; queryLength: number } = pending;
      assertThrows(() => {
        writable.queryLength = 4;
      }, TypeError);
      assertEquals(context.pendingCommit, { pastLength: 0, queryLength: 2 }, "上限は動いていない");

      // 書いていない行の確定は依然拒否（上限が 4 に開いていたらここが通ってしまう）。
      const over = assertThrows(() => context.commit(3), ExecutionError);
      assert(over.message.includes("queryLength 2"), over.message);
      context.commit(2);
      assertEquals(context.pastLength, 2);
    } finally {
      await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "失敗した deferred run は保留を作らない / dispose は保留ごと畳む（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu);
    const context = await session.createGenerationContext({ chunkLength: 4 });
    try {
      // queryLength の門で落ちる run（dispatch を 1 本も積まない = 物理 ring は無傷）。
      await assertRejects(
        () => session.run({ x: RUN_INPUT }, {}, { context, queryLength: 5, commit: "deferred" }),
        ExecutionError,
      );
      assertEquals(context.pendingCommit, undefined, "失敗 run が保留を作った");
      assertEquals(context.pastLength, 0);

      // 保留を残したまま dispose しても畳める（確定させる相手が居ないので捨てる）。
      await session.run({ x: RUN_INPUT }, {}, { context, queryLength: 2, commit: "deferred" });
      assertEquals(context.pendingCommit, { pastLength: 0, queryLength: 2 });
      await context.dispose();
      assertThrows(() => context.commit(1), ExecutionError, "dispose 済み");
      assertThrows(() => context.pendingCommit, ExecutionError, "dispose 済み");
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

/** sliding スロット（窓 8）を 1 本持つグラフ。容量は記号 `C` で context 側から与える。 */
const slidingGraph = (): DeclarationJson =>
  stateGraph(
    { k: { dtype: "f32", shape: [1, 2, "C", 4] } },
    { symbols: ["T", "C"] },
    { window: 8 },
  );

Deno.test({
  name: "slidingSlack は sliding スロットの capacity − window の最小（sliding 無しは undefined）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu);
    const sliding = await stateSession(gpu, slidingGraph());
    try {
      // full だけの context は余裕という概念を持たない（undefined と 0 を混同させない）。
      const full = await session.createGenerationContext({ chunkLength: 1 });
      try {
        assertEquals(full.slidingSlack, undefined);
      } finally {
        await full.dispose();
      }

      // 容量 16 / 窓 8 → 余裕 8。容量ちょうど（8）なら 0。
      for (const [capacity, slack] of [[16, 8], [8, 0]] as const) {
        const context = await sliding.createGenerationContext({
          chunkLength: 1,
          bindings: { C: capacity },
        });
        try {
          assertEquals(context.slidingSlack, slack, `C=${capacity}`);
        } finally {
          await context.dispose();
        }
      }

      // 窓 > 容量 は確保の時点で fail loudly（ADR 0067 決定 4 ③ を run より前で落とす）。
      const tooNarrow = await assertRejects(
        () => sliding.createGenerationContext({ chunkLength: 1, bindings: { C: 4 } }),
        ExecutionError,
      );
      assert(tooNarrow.message.includes("window 8"), tooNarrow.message);
    } finally {
      await sliding.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "deferred run の queryLength は sliding の余裕まで（超える発行は同期区間で拒否・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const sliding = await stateSession(gpu, slidingGraph());
    // 容量 9 / 窓 8 → 余裕 1。deferred は「余裕」= 1 行まで（借り手 = readonly 読者の窓の
    // 下端は貸し手より 1 列低い `P+m−W` なので、条件は `Q ≤ 余裕`）。
    const context = await sliding.createGenerationContext({ chunkLength: 4, bindings: { C: 9 } });
    try {
      assertEquals(context.slidingSlack, 1);
      // 2 行の deferred は発行の同期区間で落ち、リースも保留も残らない（受理 0 行でも棄却行が
      // 借り手の live 窓の外に落ちる、という余裕の条件を超える）。
      const rejected = await assertRejects(
        () => sliding.run({ x: RUN_INPUT }, {}, { context, queryLength: 2, commit: "deferred" }),
        ExecutionError,
      );
      assert(rejected.message.includes("余裕 1"), rejected.message);
      assertEquals(context.pendingCommit, undefined);
      assertEquals(context.pastLength, 0);
      // immediate は全行を確定させるので上限は chunkLength のまま。
      await sliding.run({ x: RUN_INPUT }, {}, { context, queryLength: 3 });
      assertEquals(context.pastLength, 3);
      // 余裕ちょうどは通る。
      await sliding.run({ x: RUN_INPUT }, {}, { context, queryLength: 1, commit: "deferred" });
      assertEquals(context.pendingCommit, { pastLength: 3, queryLength: 1 });
      context.commit(0);
      assertEquals(context.pastLength, 3);
    } finally {
      await context.dispose();
      await sliding.dispose();
      gpu.destroy();
    }
  },
});

// ---------------------------------------------------------------------------
// fork（会話状態の分岐 — ADR 0066 追記〈2026-10-11〉）
// ---------------------------------------------------------------------------
//
// 縛るのは 4 つ:
//   ①写す中身: 全スロットの全バイト（u32 で読み戻す）と pastLength・不変の値。焼いた束は写さない
//   ②分岐の独立: 同じ入力なら元と複製の出力が u32 で一致し、複製だけを進めても元のバイトは動かない
//   ③同期区間の契約: `fork()` を await せず同じ tick に元へ発行した run / rewind / dispose /
//     2 本目の fork はどれも通り、複製は呼んだ時点の状態を持つ
//   ④拒否の行列（(g) 借り手は gpu_borrowed_context_test.ts）と診断
// ②③ の門が空振りでないことは故障注入 2 種（複製の dispatch が元のスロットを束ねる形・複製の前に
// await を挟む host）で示す。

/** fork の門のスロット容量・深さ（full attention 1 head — 値がスロットの中身に依存する最小形）。 */
const FORK_CAPACITY = 16;
const FORK_DEPTH = 4;
const FORK_SLOT_BYTES = FORK_CAPACITY * FORK_DEPTH * 4;

/**
 * 書いた KV を読むグラフ（full attention + `state_append` 2 本）。出力 `o` が過去の KV の中身に
 * 依存するので、「同じ入力なら同じ出力」が状態の同一性の観測になる。
 */
const forkGraph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["attention", "state_append"] },
  symbols: ["M"],
  inputs: ["q", "k", "v"].map((name) => ({
    name,
    dtype: "f32",
    shape: [1, 1, "M", FORK_DEPTH],
  })),
  outputs: ["o"],
  initializers: {},
  values: { o: { dtype: "f32", shape: [1, 1, "M", FORK_DEPTH] } },
  states: {
    kslot: { dtype: "f32", shape: [1, 1, FORK_CAPACITY, FORK_DEPTH] },
    vslot: { dtype: "f32", shape: [1, 1, FORK_CAPACITY, FORK_DEPTH] },
  },
  nodes: [
    {
      op: "attention",
      ins: ["q", "k", "v"],
      outs: ["o"],
      attrs: { scale: 0.5 },
      states: { k: "kslot", v: "vslot" },
    },
    ...[["k", "kslot"], ["v", "vslot"]].map(([name, slot]) => ({
      op: "state_append",
      ins: [name],
      outs: [],
      attrs: {},
      states: { slot },
    })),
  ],
});

const forkSession = async (gpu: GpuContext): Promise<Session> =>
  await createSessionFromContainer(gpu, await openGraphModel(forkGraph()), GRAPH_NAME);

/** `salt` で中身の変わる決定的な入力（`rows` 行 — pad 行は作らない = 有効行 = 物理行）。 */
const forkInputs = (rows: number, salt: number): Record<string, Tensor> =>
  Object.fromEntries(
    ["q", "k", "v"].map((name, n) => [
      name,
      {
        dtype: "f32",
        shape: [1, 1, rows, FORK_DEPTH],
        data: Float32Array.from(
          { length: rows * FORK_DEPTH },
          (_, i) => (((i * 7 + n * 3 + salt * 5) % 17) - 8) * 0.13,
        ),
      } satisfies Tensor,
    ]),
  );

/**
 * 1 step（`rows` 行を全部有効行として流す）。
 *
 * MUST: async にしない — `session.run` を**呼んだ同期区間**で発行させる（同じ tick に並べる門が
 * この性質に依る）。
 */
const forkStep = (
  session: Session,
  context: GenerationContext,
  rows: number,
  salt: number,
): Promise<Record<string, Tensor>> =>
  session.run(forkInputs(rows, salt), {}, { context, queryLength: rows });

/** 出力のビット列（u32 で比べる — 値の近さではなく同一性を見る）。 */
const outputWords = (outputs: Record<string, Tensor>): number[] => {
  const data = outputs["o"].data;
  return Array.from(new Uint32Array(data.buffer, data.byteOffset, data.length));
};

/** GPU バッファ全域を u32 列として読み戻す。 */
const readWords = async (
  gpu: GpuContext,
  buffer: GPUBuffer,
  byteLength: number,
): Promise<number[]> => {
  const staging = gpu.device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const encoder = gpu.device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, staging, 0, byteLength);
    gpu.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const words = Array.from(new Uint32Array(staging.getMappedRange().slice(0)));
    staging.unmap();
    return words;
  } finally {
    staging.destroy();
  }
};

/** context の全スロットを読み戻す（スロット名 → 全バイトの u32 列）。 */
const readSlotWords = async (
  gpu: GpuContext,
  context: GenerationContext,
): Promise<Record<string, number[]>> => {
  const words: Record<string, number[]> = {};
  for (const [name, slot] of internals(context).slots) {
    words[name] = await readWords(gpu, slot.buffer, slot.byteLength);
  }
  return words;
};

Deno.test({
  name: "fork は全スロットの全バイトと論理長を写し、分岐は独立に進む（u32・診断つき・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await forkSession(gpu);
    const source = await session.createGenerationContext({ chunkLength: 4, chunkBuckets: [2] });
    let forked: GenerationContext | undefined;
    try {
      // prefill 1 本 + decode 2 本。2 本目の decode は導出済み計画に当たって backing を作るので、
      // 元の context は decode 形の backing に対して 1 回焼いている（焼き直しの勘定の起点）。
      await forkStep(session, source, 4, 0);
      await forkStep(session, source, 1, 1);
      await forkStep(session, source, 1, 2);
      assertEquals(source.pastLength, 6);
      const before = session.diagnostics().stateBacking;
      const words = await readSlotWords(gpu, source);
      // 期待値が自明でないこと（書いた行が 0 のままだと、写し漏れも一致してしまう）。
      for (const [name, slot] of Object.entries(words)) {
        assert(slot.some((word) => word !== 0), `${name}: 元のスロットが全 0（門が空振りする）`);
      }

      forked = await source.fork();
      // ① 写す中身: 論理長と不変の値、全スロットの全バイト。実体は別のバッファ。
      assertEquals(forked.pastLength, 6);
      assertEquals(forked.chunkLength, 4);
      assertEquals(forked.chunkBuckets, [2]);
      assertEquals([...internals(forked).allowedRows], [1, 2, 4]);
      assertEquals(forked.slidingSlack, undefined);
      assertEquals(forked.pendingCommit, undefined);
      for (const [name, slot] of internals(forked).slots) {
        assert(
          slot.buffer !== internals(source).slots.get(name)?.buffer,
          `${name}: 複製が元のバッファを共有している`,
        );
      }
      assertEquals(await readSlotWords(gpu, forked), words, "全スロットの全バイトが一致する");
      // 診断: 常駐は元の 1 本ぶん増え、累計は 1 増える。複製そのものは焼かない。
      assertEquals(
        session.diagnostics().stateBacking,
        {
          residentBytes: before.residentBytes + FORK_SLOT_BYTES * 2 + LENGTHS_BYTES,
          contextCount: before.contextCount + 1,
          rebindCount: before.rebindCount,
        },
        "複製はスロット 2 本 + 論理長 uniform ぶんを常駐させ、焼き直しは run まで起きない",
      );

      // ② 同じ decode の入力で両方を進めると、出力は u32 で一致する。焼き直しは複製の最初の
      // run だけ（使った backing 1 本につき 1 回）で、元の続きの run では増えない。
      const fromSource = await forkStep(session, source, 1, 3);
      assertEquals(
        session.diagnostics().stateBacking.rebindCount,
        before.rebindCount,
        "元の続きの run が焼き直した（元の束を捨てている）",
      );
      const fromFork = await forkStep(session, forked, 1, 3);
      assertEquals(
        session.diagnostics().stateBacking.rebindCount,
        before.rebindCount + 1,
        "複製の最初の run は使った backing 1 本につき 1 回だけ焼く",
      );
      assertEquals(outputWords(fromFork), outputWords(fromSource), "同じ入力の出力が u32 で一致");
      const sourceWords = await readSlotWords(gpu, source);
      assertEquals(
        await readSlotWords(gpu, forked),
        sourceWords,
        "同じ入力で進めた 2 本は同じバイト",
      );

      // 複製だけを進めても元のバイトは動かない（陽性対照: 複製の run は自分のスロットへ書いた）。
      await forkStep(session, forked, 1, 9);
      assertEquals(session.diagnostics().stateBacking.rebindCount, before.rebindCount + 1);
      assertEquals(
        await readSlotWords(gpu, source),
        sourceWords,
        "複製の run が元のスロットへ書いた",
      );
      assertNotEquals(await readSlotWords(gpu, forked), sourceWords, "複製の run が書いていない");
      assertEquals([source.pastLength, forked.pastLength], [7, 8]);

      // 元を先に dispose してよい（親子の依存を持たない）。
      await source.dispose();
      await forkStep(session, forked, 1, 10);
      assertEquals(forked.pastLength, 9);
      assertEquals(
        session.diagnostics().stateBacking.residentBytes,
        FORK_SLOT_BYTES * 2 + LENGTHS_BYTES,
        "元を返した後に残るのは複製のぶんだけ",
      );
    } finally {
      await forked?.dispose();
      await source.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name:
    "fork() を await せず同じ tick に元へ run を発行しても、複製は呼んだ時点の状態を持つ（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await forkSession(gpu);
    const source = await session.createGenerationContext({ chunkLength: 4 });
    let forked: GenerationContext | undefined;
    try {
      await forkStep(session, source, 4, 0);
      const words = await readSlotWords(gpu, source);
      // 同じ同期区間で fork → run の順に発行する。fork は元に予約を立てないので run は拒否されず、
      // 複製は fork を呼んだ時点（run の書き込みより前）の内容を持つ。
      const forking = source.fork();
      const running = forkStep(session, source, 1, 5);
      [forked] = await Promise.all([forking, running]);
      assertEquals(await readSlotWords(gpu, forked), words, "複製が後続の run の書き込みを写した");
      assertEquals(forked.pastLength, 4);
      assertEquals(source.pastLength, 5);
      // 陽性対照: 同じ tick の run は確かに元のスロットへ書いている（書いていなければ上の一致は
      // 順序を何も見ていない）。
      assertNotEquals(await readSlotWords(gpu, source), words, "同じ tick の run が書いていない");
    } finally {
      await forked?.dispose();
      await source.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "fork() の直後（await 前）の rewind / 2 本目の fork / dispose は通る（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await forkSession(gpu);
    const source = await session.createGenerationContext({ chunkLength: 4 });
    const forks: GenerationContext[] = [];
    try {
      await forkStep(session, source, 4, 0);
      const words = await readSlotWords(gpu, source);
      // どれも同じ同期区間で発行する（fork は予約を取らないので、元の後続の操作を塞がない）。
      const first = source.fork();
      source.rewind(2);
      const second = source.fork();
      const disposal = source.dispose();
      // dispose 要求の後の fork は拒否される（拒否の (a)）。
      const late = assertRejects(() => source.fork(), ExecutionError, "dispose 済み");
      forks.push(...await Promise.all([first, second]));
      await disposal;
      await late;

      const [atFour, atTwo] = forks;
      assertEquals([atFour.pastLength, atTwo.pastLength], [4, 2], "論理長は呼んだ時点の値");
      // rewind は論理長だけを動かすので、2 本ともバイトは元の prefill 後の内容。
      assertEquals(await readSlotWords(gpu, atFour), words);
      assertEquals(await readSlotWords(gpu, atTwo), words);
      // 元を返した後も、複製はどちらも独立に進む。
      await forkStep(session, atFour, 1, 6);
      await forkStep(session, atTwo, 1, 6);
      assertEquals([atFour.pastLength, atTwo.pastLength], [5, 3]);
      assertEquals(session.diagnostics().stateBacking.contextCount, 3);
    } finally {
      for (const forked of forks) await forked.dispose();
      await source.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name:
    "fork の拒否: 汚染・進行中 run・batch の予約中・commit 待ち・dispose 済み Session（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await forkSession(gpu);
    const sink = await gpu.createResident(FORK_DEPTH * 4);
    const contexts: GenerationContext[] = [];
    const fresh = async (): Promise<GenerationContext> => {
      const context = await session.createGenerationContext({ chunkLength: 4 });
      contexts.push(context);
      return context;
    };
    try {
      // (c) 汚染の後（背後の物理 state が失われた context を写させない）。
      const poisoned = await fresh();
      internals(poisoned).poison("テストの故障注入");
      await assertRejects(() => poisoned.fork(), ExecutionError, "汚染された");

      // (d) 決着していない run（予約は発行の同期区間で立つので、await しない並びで踏める）。
      const busy = await fresh();
      const running = forkStep(session, busy, 1, 0);
      await assertRejects(() => busy.fork(), ExecutionError, "進行中の generation run");
      await running;
      contexts.push(await busy.fork());

      // (e) batch の使用予約中（enqueue の Promise は返っても、区間の決着までは予約が残る）。
      const batched = await fresh();
      const batch = await gpu.beginBatch();
      try {
        await session.enqueue(forkInputs(1, 0), {
          batch,
          generation: { context: batched, queryLength: 1 },
          copyOutputs: { o: sink },
        });
        await assertRejects(() => batched.fork(), ExecutionError, "batch の最終決着待ち");
      } finally {
        await batch.finish();
      }
      contexts.push(await batched.fork());

      // (f) deferred commit の確定待ち（物理 ring に書いた行のどこまでが確定か決まっていない）。
      const deferred = await fresh();
      await session.run(forkInputs(1, 0), {}, {
        context: deferred,
        queryLength: 1,
        commit: "deferred",
      });
      await assertRejects(() => deferred.fork(), ExecutionError, "commit 待ち");
      deferred.commit(1);
      contexts.push(await deferred.fork());

      // 拒否された fork は 1 本も確保していない（累計は作った 4 本 + 通った fork 3 本だけ）。
      assertEquals(session.diagnostics().stateBacking.contextCount, 7);

      // (h) Session の dispose が確保の await を跨いだ場合: 確保した分は返し、生存集合にも累計にも
      // 載せない（元は残る）。
      const target = await fresh();
      const before = session.diagnostics().stateBacking;
      const pending = target.fork();
      await session.dispose();
      const crossed = await assertRejects(() => pending, ExecutionError);
      // 文言は fork の文脈つき（create の文言と読み分ける）。
      assert(crossed.message.includes("fork: dispose 済みの Session"), crossed.message);
      assertEquals(session.diagnostics().stateBacking, before, "確保した複製が生存集合に残った");
      // (h) 呼んだ時点で dispose 済みの Session。
      await assertRejects(() => target.fork(), ExecutionError, "fork: dispose 済みの Session");
      assertEquals(target.pastLength, 0, "拒否された fork は元を汚さない");
    } finally {
      for (const context of contexts) await context.dispose();
      await session.dispose();
      sink.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "device 消失後の fork は GpuDeviceLostError（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // この case は device を壊すので専用の GpuContext を取る。
    const gpu = await acquireGpu();
    const session = await forkSession(gpu);
    const context = await session.createGenerationContext({ chunkLength: 4 });
    gpu.destroy();
    // destroy 直後（lost の反応前）から拒否する — 失われた device 上の確保は無効なバッファを
    // 返すだけなので、通すと空の KV を持つ複製ができる。
    await assertRejects(() => context.fork(), GpuDeviceLostError, "device が失われた");
    await session.dispose().catch(() => undefined);
  },
});

/**
 * fork の故障注入用の device（確保の N 本目の同期 throw・pop の reject を**後から**仕掛ける）。
 *
 * {@link injectFaults} と違い、返すバッファは**素の GPUBuffer**（`destroy` だけを自身の
 * プロパティで包む）。fork は確保したバッファを `copyBufferToBuffer` へ渡すので、Proxy を
 * WebGPU の実装へ渡す形を避ける。仕掛けは元の context を作った後に `arm` で入れる（元の確保を
 * 巻き込まない）。
 */
const armableFaults = (
  gpu: GpuContext,
): {
  readonly gpu: GpuContext;
  readonly destroyCounts: number[];
  arm(fault: Omit<Fault, "hangPop">): void;
} => {
  const destroyCounts: number[] = [];
  let creations = 0;
  let fault: Omit<Fault, "hangPop"> = {};
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
          const destroy = buffer.destroy.bind(buffer);
          Object.defineProperty(buffer, "destroy", {
            value: (): void => {
              destroyCounts[index] += 1;
              destroy();
            },
          });
          return buffer;
        };
      }
      if (prop === "popErrorScope" && fault.popError !== undefined) {
        const error = fault.popError;
        return (): Promise<GPUError | null> => {
          // MUST: 実 pop は必ず発行する（{@link injectFaults} と同じ理由 — LIFO を崩さない）。
          void target.popErrorScope().catch(() => null);
          return Promise.reject(error);
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
  return {
    gpu: proxied,
    destroyCounts,
    arm: (next): void => {
      fault = next;
      creations = 0;
    },
  };
};

Deno.test({
  name: "fork の確保途中の失敗は確保物をちょうど 1 回ずつ返し、元は無傷（故障注入・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const graph = prepareContainer(
      await openStateModel(stateGraph(NUMERIC_STATES)),
      GRAPH_NAME,
    ).graph;
    const injected = armableFaults(gpu);
    try {
      const source = await GenerationContext.create(injectedHost(injected.gpu, graph), {
        chunkLength: 1,
      });
      internals(source).advance(0, 1);
      assertEquals(injected.destroyCounts, [0, 0, 0], "元はスロット 2 本 + 論理長 uniform");

      // ① 途中の同期 throw: 複製の 2 本目で落ちても 1 本目がちょうど 1 回返る。
      injected.arm({ failCreateAt: 2 });
      await assertRejects(() => source.fork(), Error, "注入: 2 本目の createBuffer");
      assertEquals(injected.destroyCounts, [0, 0, 0, 1], "確保済みの 1 本が漏れない");

      // ② pop の reject: 複製の全確保物（スロット 2 本 + 論理長 uniform）がちょうど 1 回返る。
      injected.arm({ popError: new Error("注入: popErrorScope の reject") });
      await assertRejects(() => source.fork(), Error, "注入: popErrorScope の reject");
      assertEquals(injected.destroyCounts, [0, 0, 0, 1, 1, 1, 1]);

      // 元は無傷: 論理長も読め、仕掛けを外せば fork も通る。
      injected.arm({});
      assertEquals(source.pastLength, 1);
      const forked = await source.fork();
      assertEquals(forked.pastLength, 1);
      await forked.dispose();
      await source.dispose();
      assertEquals(injected.destroyCounts, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name:
    "分岐の独立の門は空振りでない — 複製の dispatch が元のスロットを束ねると元が動く（故障注入・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await forkSession(gpu);
    const source = await session.createGenerationContext({ chunkLength: 4 });
    let forked: GenerationContext | undefined;
    try {
      // 元が decode 形の backing に対して焼いた束を持つところまで進める。
      await forkStep(session, source, 4, 0);
      await forkStep(session, source, 1, 1);
      await forkStep(session, source, 1, 2);
      forked = await source.fork();
      const sourceWords = await readSlotWords(gpu, source);
      const forkedWords = await readSlotWords(gpu, forked);

      // 注入: 元が焼いた束を複製へ写す（`#baked` を写す実装の再現。バッファを共有する実装も
      // 「複製の dispatch が元のスロットを束ねる」点で同じ観測になる）。backing の世代は
      // 1 始まりの単調増加（`planBacking.buildCount` が最大値）。
      let injected = 0;
      for (let build = 1; build <= session.diagnostics().planBacking.buildCount; build += 1) {
        const groups = internals(source).bakedGroups(build);
        if (groups === undefined) continue;
        internals(forked).setBakedGroups(build, groups);
        injected += 1;
      }
      assert(injected > 0, "写す束が無い（故障注入が空振りする）");
      const rebinds = session.diagnostics().stateBacking.rebindCount;
      await forkStep(session, forked, 1, 9);

      // 独立の門の 3 つの観測がどれも赤になる: 元のバイトが動き、複製のバイトは動かず、
      // 複製の最初の run が焼き直さない。
      assertNotEquals(
        await readSlotWords(gpu, source),
        sourceWords,
        "注入しても元のバイトが動かない（独立の門が空振りする）",
      );
      assertEquals(await readSlotWords(gpu, forked), forkedWords);
      assertEquals(session.diagnostics().stateBacking.rebindCount, rebinds);
    } finally {
      await forked?.dispose();
      await source.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
});

/**
 * `adopt` に await を挟む host（同期区間の契約を破る将来の改修の再現 — 故障注入）。
 * `allocate` は待ちの後で呼ぶので、複製は待ちの間に積まれた書き込みの後の状態を写す。
 */
const delayedAdoptHost = (
  gpu: GpuContext,
  graph: IrGraph,
  delay: () => Promise<void>,
): GenerationContextHost => ({
  ...injectedHost(gpu, graph),
  adopt: async (allocate) => {
    await delay();
    return await allocate();
  },
});

Deno.test({
  name:
    "同期区間の門は空振りでない — 複製の前に await を挟む host では後続の書き込みを写す（故障注入・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const graph = prepareContainer(
      await openStateModel(stateGraph(NUMERIC_STATES)),
      GRAPH_NAME,
    ).graph;
    const earlier = new Uint32Array(SLOT_BYTES / 4).fill(0x3f800000);
    const later = new Uint32Array(SLOT_BYTES / 4).fill(0x40000000);
    /**
     * 元の `k` を `earlier` で埋め、`fork()` を呼んだ**同じ同期区間**で `later` に上書きしてから、
     * 複製の `k` を読む（queue の順序は writeBuffer と submit の発行順）。
     */
    const forkThenOverwrite = async (host: GenerationContextHost): Promise<number[]> => {
      const source = await GenerationContext.create(host, { chunkLength: 1 });
      try {
        const slot = internals(source).slots.get("k");
        assert(slot !== undefined, "スロット k が引けない");
        gpu.device.queue.writeBuffer(slot.buffer, 0, earlier);
        const pending = source.fork();
        gpu.device.queue.writeBuffer(slot.buffer, 0, later);
        const forked = await pending;
        try {
          const copied = internals(forked).slots.get("k");
          assert(copied !== undefined, "複製のスロット k が引けない");
          return await readWords(gpu, copied.buffer, copied.byteLength);
        } finally {
          await forked.dispose();
        }
      } finally {
        await source.dispose();
      }
    };
    try {
      assertEquals(
        await forkThenOverwrite(injectedHost(gpu, graph)),
        [...earlier],
        "契約どおりの host: 複製は fork() を呼んだ時点の内容",
      );
      // 待ちを挟むと、同じ同期区間の後続の書き込みが複製に入る（= 上の門が赤になる）。microtask
      // 1 段でもここは捕まる（書き込みが同期区間の中にあるため）。run 経由の書き込みは本体が
      // マイクロタスクの先で積まれるので、そちらで microtask のずれを捕まえる保証は無い（推測）。
      for (
        const [label, delay] of [
          ["microtask", () => Promise.resolve()],
          ["macrotask", () => new Promise<void>((resolve) => setTimeout(resolve, 0))],
        ] as const
      ) {
        assertEquals(
          await forkThenOverwrite(delayedAdoptHost(gpu, graph, delay)),
          [...later],
          `${label} の await を挟んでも呼んだ時点の内容が残る（門が空振りする）`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name:
    "実 Session の fork も同期区間で複製する — 同じ区間の後続の書き込みは複製に入らない（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // 上の門は host を自作するので、`Session.#adopt`（判定 → `allocate` を同期に呼ぶ）の側に
    // await が入る退行は捕まらない。ここは同じ形を実 Session の登録経路で縛る。退行の再現は
    // `#adopt` の `allocate()` の前に `await Promise.resolve()` を 1 段挟む形（`allocate` が
    // 後続の writeBuffer の後に走るので、複製は `later` を読む）。
    const gpu = await acquireGpu();
    const session = await stateSession(gpu);
    const earlier = new Uint32Array(SLOT_BYTES / 4).fill(0x3f800000);
    const later = new Uint32Array(SLOT_BYTES / 4).fill(0x40000000);
    const source = await session.createGenerationContext({ chunkLength: 1 });
    let forked: GenerationContext | undefined;
    try {
      const slot = internals(source).slots.get("k");
      assert(slot !== undefined, "スロット k が引けない");
      gpu.device.queue.writeBuffer(slot.buffer, 0, earlier);
      const pending = source.fork();
      gpu.device.queue.writeBuffer(slot.buffer, 0, later);
      forked = await pending;
      const copied = internals(forked).slots.get("k");
      assert(copied !== undefined, "複製のスロット k が引けない");
      assertEquals(
        await readWords(gpu, copied.buffer, copied.byteLength),
        [...earlier],
        "複製が fork() を呼んだ後の書き込みを写した（登録経路に await が入った）",
      );
    } finally {
      await forked?.dispose();
      await source.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "fork の所要時間を記録する（スロット 2 本 × 64 MiB・記録のみ・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // 複製は scheduler の外の自前 submit なので、巨大な複製の所要は他に観測点が無い。値の門は
    // 置かない（device と負荷で動く）— 弱い GPU で TDR 域に近づくかを読む材料として残す。
    // MUST: 区間の末尾で queue の完了を待つ。fork が待つ popErrorScope の解決はコピーの実行
    // 完了を待たない（128 MiB で 0.1 ms という観測）ので、それだけでは複製の所要を測れない。
    const rows = 64 * 1024 * 1024 / (2 * 4 * 4);
    const gpu = await acquireGpu();
    const session = await stateSession(
      gpu,
      stateGraph({
        k: { dtype: "f32", shape: [1, 2, rows, 4] },
        v: { dtype: "f32", shape: [1, 2, rows, 4] },
      }),
    );
    const source = await session.createGenerationContext({ chunkLength: 1 });
    try {
      const started = performance.now();
      const forked = await source.fork();
      await gpu.device.queue.onSubmittedWorkDone();
      const elapsed = performance.now() - started;
      try {
        assertEquals(internals(forked).bytes, internals(source).bytes);
        console.log(
          `[gpu] GenerationContext.fork: ${
            (internals(source).bytes / (1024 * 1024)).toFixed(0)
          } MiB を ${elapsed.toFixed(1)} ms（確保 + 複製の実行完了まで）`,
        );
      } finally {
        await forked.dispose();
      }
    } finally {
      await source.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
});
