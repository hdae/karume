// params バッファの内容アドレスキャッシュ（Session 常駐）。params の全バイトは
// グラフ・node.attrs・解決済み shape の純関数なので、同じ内容の params は run 内・run 跨ぎの
// どちらでも 1 本しか作られないのが不変条件。
//
// NOTE: 同一 bindings の 2 run 目は導出済み計画（see the prepared-plan section below）に当たり、導出相
// ごと飛ぶので #writeParams に到達しない（lastRunParams は {0,0}）。したがってこの門は
// ①同一 run 内の重複 params（同形ノード 2 本）と、②導出済み計画から**追い出された後**の
// 再導出 run（prepared ミス・params は全ヒット）の 2 点で再利用を観測する — 2 つのキャッシュの
// 寿命が独立であること（params は無上限・prepared は LRU 上限あり）もここで固定される。
//
// この門が見るのは「新規生成 0 で再利用 > 0」と「値が完全一致」の 2 つ。前者だけだと
// キャッシュが誤ったバッファを配っても緑になり、後者だけだとキャッシュが黙って外れて
// 毎 dispatch 確保に戻っても緑になる（どちらも例外は出ない）。

import { assertEquals, assertRejects } from "@std/assert";
import { describe, it, test } from "vite-plus/test";
import { DispatchLimitError } from "../src/codegen/errors.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import { GpuOutOfMemoryError } from "../src/gpu/error-scope.ts";
import { defaultGemmGeometry, gemmTileN } from "../src/kernels/gemm-geometry.ts";
import {
  createSessionFromContainer,
  PREPARED_PLAN_CAPACITY,
  type Tensor,
} from "../src/runtime/executor.ts";
import type { OpenedContainer } from "../src/format/container/open.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import { type DeclarationJson, f32Bytes, fill, openGraphModel } from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { injectOutOfMemory } from "./helpers/oom-injection.ts";

/**
 * y[T] = Σ_c ((x[T,3] · w[3,2] + b[2]) + b[2])。params の形が異なる 3 ノード（matmul / add /
 * sum）に加えて、**同形の add をもう 1 本**置く — 2 本目の add は params の全バイトが
 * 1 本目と一致するので、1 run 目の中で再利用が 1 件観測できる。
 */
const GRAPH: DeclarationJson = {
  format: "karume-ir",
  version: 2,
  requires: { ops: ["matmul", "add", "sum"] },
  symbols: ["T"],
  inputs: [{ name: "x", dtype: "f32", shape: ["T", 3] }],
  outputs: ["y"],
  initializers: { w: {}, b: {} },
  values: {
    w: { dtype: "f32", shape: [3, 2] },
    b: { dtype: "f32", shape: [2] },
    h: { dtype: "f32", shape: ["T", 2] },
    g: { dtype: "f32", shape: ["T", 2] },
    g2: { dtype: "f32", shape: ["T", 2] },
    y: { dtype: "f32", shape: ["T"] },
  },
  nodes: [
    { op: "matmul", ins: ["x", "w"], outs: ["h"], attrs: {} },
    { op: "add", ins: ["h", "b"], outs: ["g"], attrs: {} },
    { op: "add", ins: ["g", "b"], outs: ["g2"], attrs: {} },
    { op: "sum", ins: ["g2"], outs: ["y"], attrs: { dim: 1 } },
  ],
};

const openedModel = () =>
  openGraphModel(GRAPH, [
    {
      graph: "model",
      initializer: "w",
      bytes: f32Bytes([0.5, -1.5, 2, 0.25, -0.75, 1]),
      encoding: { codec: "f32" },
    },
    {
      graph: "model",
      initializer: "b",
      bytes: f32Bytes([0.125, -0.5]),
      encoding: { codec: "f32" },
    },
  ]);

const input = (rows: number): Tensor => ({
  dtype: "f32",
  shape: [rows, 3],
  data: Float32Array.from({ length: rows * 3 }, (_, i) => ((i % 9) - 4) * 0.5),
});

test.skipIf(!GPU_AVAILABLE)(
  "同じ内容の params は run 内で 1 本に畳まれ、ヒット run は 1 本も触らない（実 GPU）",
  async () => {
    const gpu = await acquireGpu();
    const session = await createSessionFromContainer(gpu, await openedModel(), "model");
    try {
      assertEquals(session.diagnostics().lastRunParams, undefined, "未実行なら診断は undefined");

      const first = await session.run({ x: input(4) });
      const firstParams = session.diagnostics().lastRunParams;
      assertEquals(firstParams?.reuseCount, 1, "同形 add の 2 本目は 1 run 目の中で再利用");
      assertEquals((firstParams?.allocCount ?? 0) > 0, true, "1 run 目は新規生成がある");
      // 重みアリーナが params の実体も所有する（破棄は weights の dispose に相乗り）。
      const weightsAfterFirst = session.diagnostics().weights.allocCount;

      // 2 run 目は導出済み計画に当たり導出相ごと飛ぶ — params の GPU 操作もゼロ
      // （キャッシュが外れたのではなく、レシピが params 実体を直参照している）。
      const second = await session.run({ x: input(4) });
      assertEquals(session.diagnostics().lastRunPrepared?.hit, true);
      assertEquals(session.diagnostics().lastRunParams, { allocCount: 0, reuseCount: 0 });
      assertEquals(
        session.diagnostics().weights.allocCount,
        weightsAfterFirst,
        "ヒット run はアリーナに 1 本も足さない",
      );

      // 使い回した params で値が変わらない（キャッシュが別の内容のバッファを配っていない）
      assertEquals(second["y"].shape, first["y"].shape);
      assertEquals(Array.from(second["y"].data), Array.from(first["y"].data));
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "導出済み計画から追い出されても params キャッシュは生き残り全ヒットする（実 GPU）",
  async () => {
    const gpu = await acquireGpu();
    const session = await createSessionFromContainer(gpu, await openedModel(), "model");
    try {
      await session.run({ x: input(4) });
      const narrow = session.diagnostics().lastRunParams;
      const narrowAlloc = narrow?.allocCount ?? 0;

      // 記号次元が変われば解決済み shape が変わり、params の内容も変わる（= 別のキー）。
      await session.run({ x: input(9) }, { T: 9 });
      assertEquals(
        session.diagnostics().lastRunParams?.allocCount,
        narrowAlloc,
        "別 shape は作り直す",
      );

      // T=4 を導出済み計画（LRU 上限）から確実に追い出す — 上限 +1 種類の別 shape を通せば、
      // 最古の T=4 は必ず落ちる。params キャッシュは無上限なのでここで作られた params は残る。
      // MUST: 本数は定数から出す（写しを置くと上限を動かしたときに追い出しが起きなくなり、
      // 「prepared ミスなのに params 全ヒット」という観測点そのものが消える）。
      for (let rows = 5; rows <= 5 + PREPARED_PLAN_CAPACITY; rows += 1) {
        await session.run({ x: input(rows) });
      }

      // 再導出 run（prepared はミス）だが params は 1 本も作り直さない — 1 run 目の実体が
      // Session 常駐のまま残っている（+1 は同形 add の run 内再利用）。
      await session.run({ x: input(4) });
      assertEquals(session.diagnostics().lastRunPrepared?.hit, false, "計画は追い出し済み");
      assertEquals(session.diagnostics().lastRunParams, {
        allocCount: 0,
        reuseCount: narrowAlloc + 1,
      });
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

/**
 * params キャッシュが **Session 寿命で無界**であること（by-design — docs/limitations.md）を
 * 固定する。追い出し機構が入れば赤くなるので、入れるときはこの門ごと裁定し直すこと。
 *
 * 同範囲の他キャッシュ（`SubmitScheduler.MEASURED_HISTORY` / `PREPARED_PLAN_CAPACITY`）は
 * 「MUST: 有界」を明示しており、ここだけが規律の外にある。数値で固定しておかないと、
 * 「可変 shape を長時間回すと GPU バッファが単調増加する」という制約が挙動からは見えない
 * （例外も警告も出ず、`diagnostics().weights.allocCount` が伸びるだけ）。
 */
test.skipIf(!GPU_AVAILABLE)(
  "params キャッシュは Session 寿命で無界（可変 shape の run ごとに単調増加・実 GPU）",
  async () => {
    const gpu = await acquireGpu();
    const session = await createSessionFromContainer(gpu, await openedModel(), "model");
    try {
      // initializer 2 本ぶんだけが載った状態から始まる。
      assertEquals(session.diagnostics().weights.allocCount, 2);

      // 記号次元 T を毎回変える = params の内容が毎回変わる = 1 度も当たらない。
      // 導出済み計画（LRU 上限）からは追い出されるが、params の実体は 1 本も返らない。
      const RUNS = PREPARED_PLAN_CAPACITY + 1;
      for (let i = 0; i < RUNS; i += 1) {
        await session.run({ x: input(2 + i) });
        assertEquals(
          session.diagnostics().lastRunParams,
          { allocCount: 3, reuseCount: 1 },
          `run ${i}: 3 種の params を新規生成（同形 add の 2 本目だけ run 内再利用）`,
        );
      }

      assertEquals(
        session.diagnostics().weights.allocCount,
        2 + RUNS * 3,
        "追い出しは 1 つも無い（initializer 2 本 + run 数 × ノード種 3）",
      );
      assertEquals(
        session.diagnostics().lastRunPrepared?.cachedPlans,
        PREPARED_PLAN_CAPACITY,
        "導出済み計画だけが上限で頭打ちになる（params キャッシュとは寿命が独立）",
      );
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

/** 新しい Session で 1 run だけ回した出力（巻き戻し後の値の照合先）。 */
const freshOutput = async (
  gpu: Awaited<ReturnType<typeof acquireGpu>>,
  rows: number,
): Promise<readonly number[]> => {
  const session = await createSessionFromContainer(gpu, await openedModel(), "model");
  try {
    return Array.from((await session.run({ x: input(rows) }))["y"].data);
  } finally {
    await session.dispose();
  }
};

// 非致命 OOM（`createBuffer` が無効なバッファを返すだけで例外にならない）は、params と計画を
// キャッシュへ載せた**後**の errorScope で初めて判明する。巻き戻さないと、メモリが空いた後の
// 再試行でも同じ無効バッファを直参照する計画に当たり、validation エラーが Session の寿命いっぱい
// 続く。巻き戻すのは「その実行が新しく足したもの」だけで、既存の正常なエントリは残す。
describe("params の確保が非致命 OOM で失敗した実行の後（注入・実 GPU）", () => {
  it.skipIf(!GPU_AVAILABLE)(
    "失敗した run が足した params と計画だけを捨てて次の run で作り直し、既存の計画は残す",
    async () => {
      const gpu = await acquireGpu();
      const session = await createSessionFromContainer(gpu, await openedModel(), "model");
      const oom = injectOutOfMemory(gpu.device);
      try {
        await session.run({ x: input(4) });

        oom.arm();
        await assertRejects(
          () => session.run({ x: input(9) }, { T: 9 }),
          GpuOutOfMemoryError,
          "注入: 非致命 OOM",
        );
        assertEquals(oom.fired(), 1, "注入が run の pop に当たっていない（門が空振る）");

        await session.run({ x: input(4) });
        assertEquals(
          session.diagnostics().lastRunPrepared?.hit,
          true,
          "失敗と無関係な既存の計画まで捨てている",
        );

        const retried = await session.run({ x: input(9) }, { T: 9 });
        assertEquals(
          session.diagnostics().lastRunPrepared?.hit,
          false,
          "失敗した run が登録した計画が残っている",
        );
        assertEquals(
          session.diagnostics().lastRunParams,
          { allocCount: 3, reuseCount: 1 },
          "失敗した run が確保した params をキャッシュから再利用している",
        );
        assertEquals(Array.from(retried["y"].data), await freshOutput(gpu, 9));
      } finally {
        oom.restore();
        await session.dispose();
        gpu.destroy();
      }
    },
  );

  it.skipIf(!GPU_AVAILABLE)(
    "失敗で決着した batch の enqueue が足した params と計画を捨て、次の batch で作り直す",
    async () => {
      const gpu = await acquireGpu();
      const session = await createSessionFromContainer(gpu, await openedModel(), "model");
      const oom = injectOutOfMemory(gpu.device);
      try {
        const failed = await gpu.beginBatch();
        await session.enqueue({ x: input(4) }, { batch: failed });
        assertEquals(session.diagnostics().lastRunParams?.allocCount, 3, "導出が走っていない");
        oom.arm();
        await assertRejects(() => failed.finish(), GpuOutOfMemoryError, "注入: 非致命 OOM");
        assertEquals(oom.fired(), 1, "注入が finish の pop に当たっていない（門が空振る）");

        const retry = await gpu.beginBatch();
        await session.enqueue({ x: input(4) }, { batch: retry });
        await retry.finish();
        assertEquals(
          session.diagnostics().lastRunPrepared?.hit,
          false,
          "失敗した batch の enqueue が登録した計画が残っている",
        );
        assertEquals(
          session.diagnostics().lastRunParams,
          { allocCount: 3, reuseCount: 1 },
          "失敗した batch の enqueue が確保した params をキャッシュから再利用している",
        );
        assertEquals(
          Array.from((await session.run({ x: input(4) }))["y"].data),
          await freshOutput(gpu, 4),
        );
      } finally {
        oom.restore();
        await session.dispose();
        gpu.destroy();
      }
    },
  );
});

// GPU 側の失敗でなければ、作れた params と計画は有効なので外さない。外すと、何度でも同じく落ちる
// 失敗（入力の誤り）のたびに params を新しく確保し、キャッシュから外すだけで `dispose` まで
// 積もる（params は破棄せず weights アリーナが持つ — `RecipeBuilder.forgetParams`）。
// enqueue は導出が入力の検査より前にあるので、入力の誤りでも params を載せた後に落ちる。
describe("ホスト側の失敗（入力の要素数の誤り）で落ちる enqueue を繰り返したとき（実 GPU）", () => {
  it.skipIf(!GPU_AVAILABLE)(
    "1 回目に作れた params と計画を残し、繰り返しても params を新しく確保しない",
    async () => {
      const gpu = await acquireGpu();
      const session = await createSessionFromContainer(gpu, await openedModel(), "model");
      // 要素数の誤りは束縛の解決を通り、導出の後の入力検査で落ちる。
      const wrongCount: Tensor = { dtype: "f32", shape: [4, 3], data: new Float32Array(11) };
      const failOnce = async (): Promise<void> => {
        const batch = await gpu.beginBatch();
        await assertRejects(() => session.enqueue({ x: wrongCount }, { batch }), ExecutionError);
        await assertRejects(() => batch.finish(), ExecutionError);
      };
      try {
        const initial = session.diagnostics().weights.allocCount;
        await failOnce();
        const derived = session.diagnostics().weights.allocCount;
        assertEquals(derived - initial, 3, "1 回目の enqueue が導出していない（門が空振る）");

        await failOnce();
        await failOnce();
        assertEquals(
          session.diagnostics().weights.allocCount,
          derived,
          "同じホスト側の失敗のたびに params を新しく確保している",
        );

        const batch = await gpu.beginBatch();
        await session.enqueue({ x: input(4) }, { batch });
        await batch.finish();
        assertEquals(
          session.diagnostics().lastRunPrepared?.hit,
          true,
          "ホスト側の失敗で、作れた計画を捨てている",
        );
        assertEquals(
          Array.from((await session.run({ x: input(4) }))["y"].data),
          await freshOutput(gpu, 4),
        );
      } finally {
        await session.dispose();
        gpu.destroy();
      }
    },
  );
});

// --- Prepared-plan cache ---
//
// 導出済み実行計画の Session 常駐キャッシュ（キー = 解決済み bindings）。計画・融合判定・
// レシピ導出は graph と bindings だけの純関数なので、同一 bindings の 2 run 目以降は導出相を
// 丸ごと飛ばせる（src/runtime/executor.ts の #preparedKey）。
//
// この門が見るのは「ヒット / ミスの判定」と「ヒット run の出力がミス run とビット単位で
// 一致する」の 2 つ。前者だけだと誤った計画を配っても緑になり、後者だけだとキャッシュが
// 黙って外れて毎 run 導出に戻っても緑になる（どちらも例外は出ない）。

/**
 * y = silu(x·w + b)（x: [T,3] → y: [T,2]）。末尾 2 ノードが silu 融合に掴まれるので、
 * 融合カウンタ（ADR 0040 §3 の常設診断）をヒット run 側でも見られる。
 */
const SILU_GRAPH: DeclarationJson = {
  format: "karume-ir",
  version: 2,
  requires: { ops: ["matmul", "add", "sigmoid", "mul"] },
  symbols: ["T"],
  inputs: [{ name: "x", dtype: "f32", shape: ["T", 3] }],
  outputs: ["y"],
  initializers: { w: {}, b: {} },
  values: {
    w: { dtype: "f32", shape: [3, 2] },
    b: { dtype: "f32", shape: [2] },
    h: { dtype: "f32", shape: ["T", 2] },
    g: { dtype: "f32", shape: ["T", 2] },
    s: { dtype: "f32", shape: ["T", 2] },
    y: { dtype: "f32", shape: ["T", 2] },
  },
  nodes: [
    { op: "matmul", ins: ["x", "w"], outs: ["h"], attrs: {} },
    { op: "add", ins: ["h", "b"], outs: ["g"], attrs: {} },
    { op: "sigmoid", ins: ["g"], outs: ["s"], attrs: {} },
    { op: "mul", ins: ["g", "s"], outs: ["y"], attrs: {} },
  ],
};

const openProjection = (): Promise<OpenedContainer> =>
  openGraphModel(SILU_GRAPH, [
    {
      graph: "model",
      initializer: "w",
      bytes: f32Bytes([0.5, -1.5, 2, 0.25, -0.75, 1]),
      encoding: { codec: "f32" },
    },
    {
      graph: "model",
      initializer: "b",
      bytes: f32Bytes([0.125, -0.5]),
      encoding: { codec: "f32" },
    },
  ]);

/** 出力のビット列。値の一致は「バイト同値」で見る（丸めの取り違えを許容しない）。 */
const bits = (tensor: Tensor): readonly number[] =>
  Array.from(new Uint32Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.length));

test.skipIf(!GPU_AVAILABLE)(
  "同一 bindings の 2 run 目は導出済み計画に当たり、出力はビット単位で一致する（実 GPU）",
  async () => {
    const gpu = await acquireGpu();
    const session = await createSessionFromContainer(gpu, await openProjection(), "model");
    try {
      assertEquals(session.diagnostics().lastRunPrepared, undefined, "未実行なら診断は undefined");

      const first = await session.run({ x: input(4) });
      assertEquals(
        session.diagnostics().lastRunPrepared,
        { hit: false, cachedPlans: 1 },
        "1 run 目は導出してから載せる",
      );

      const second = await session.run({ x: input(4) });
      assertEquals(
        session.diagnostics().lastRunPrepared,
        { hit: true, cachedPlans: 1 },
        "2 run 目は導出相を飛ばす",
      );
      // 飛ばしたのは導出だけで、GPU へ積むコマンド列は同一 — 出力が唯一の検出器。
      assertEquals(second["y"].shape, first["y"].shape);
      assertEquals(bits(second["y"]), bits(first["y"]));

      // 導出相ごと飛ぶので params の GPU 操作もゼロになる（値の意味は従来どおり）。
      assertEquals(session.diagnostics().lastRunParams, { allocCount: 0, reuseCount: 0 });
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "bindings が変わる run は導出し直し、元へ戻すと再びヒットする（実 GPU）",
  async () => {
    const gpu = await acquireGpu();
    const session = await createSessionFromContainer(gpu, await openProjection(), "model");
    try {
      const narrow = await session.run({ x: input(4) });
      assertEquals(session.diagnostics().lastRunPrepared, { hit: false, cachedPlans: 1 });

      // 記号次元が変われば解決済み bindings が変わる（= 別のキー）。
      await session.run({ x: input(9) });
      assertEquals(
        session.diagnostics().lastRunPrepared,
        { hit: false, cachedPlans: 2 },
        "別 bindings は導出し直して 2 本目として載る",
      );

      const again = await session.run({ x: input(4) });
      assertEquals(
        session.diagnostics().lastRunPrepared,
        { hit: true, cachedPlans: 2 },
        "先の bindings のぶんは Session 常駐のまま残っている",
      );
      assertEquals(bits(again["y"]), bits(narrow["y"]));
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "導出済み計画は上限本数で頭打ちになり、最古のものから追い出される（実 GPU）",
  async () => {
    const gpu = await acquireGpu();
    const session = await createSessionFromContainer(gpu, await openProjection(), "model");
    const capacity = PREPARED_PLAN_CAPACITY;
    try {
      for (let rows = 1; rows <= capacity; rows += 1) {
        await session.run({ x: input(rows) });
        assertEquals(
          session.diagnostics().lastRunPrepared,
          { hit: false, cachedPlans: rows },
          `T=${rows} は初出`,
        );
      }

      // 上限 +1 種類目で頭打ちに当たる。載せてから最古（T=1）を落とすので本数は変わらない。
      await session.run({ x: input(capacity + 1) });
      assertEquals(session.diagnostics().lastRunPrepared, { hit: false, cachedPlans: capacity });

      // 追い出しの証明: 一度当たっていた T=1 が再びミスになる。
      await session.run({ x: input(1) });
      assertEquals(
        session.diagnostics().lastRunPrepared,
        { hit: false, cachedPlans: capacity },
        "最古の bindings は落ちている",
      );
      // 残りは健在（追い出しが 1 本ずつであることの裏）。
      await session.run({ x: input(capacity) });
      assertEquals(session.diagnostics().lastRunPrepared, { hit: true, cachedPlans: capacity });
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "ヒット run も融合回数を報告する（キャッシュで常設診断が消えない・実 GPU）",
  async () => {
    const gpu = await acquireGpu();
    const session = await createSessionFromContainer(gpu, await openProjection(), "model");
    try {
      await session.run({ x: input(4) });
      const derived = session.diagnostics().lastRunFusions;
      assertEquals(derived?.silu, 1, "1 run 目で silu 融合が掴めている（掴めないと門が空振る）");

      await session.run({ x: input(4) });
      assertEquals(session.diagnostics().lastRunPrepared?.hit, true);
      const cached = session.diagnostics().lastRunFusions;
      assertEquals(cached === undefined, false, "ヒット run でも undefined にならない");
      assertEquals(cached, derived, "計画時に決まった回数がそのまま報告される");
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "導出相で落ちた run の部分レシピはキャッシュに載らない（実 GPU）",
  async () => {
    const gpu = await acquireGpu();
    // matmul の**出力タイル辺**で上限超過が決まる（`ceil(n / tileN) > 上限`）。辺は定数では
    // なく既定幾何から導く — 直接書くと辺を変えた瞬間に throw が起きず、assertRejects だけが
    // 静かに落ちる（runtime_executor_test.ts と同じ理由）。
    const tileN = gemmTileN(defaultGemmGeometry());
    const huge = gpu.limits.maxComputeWorkgroupsPerDimension * tileN + tileN;
    const graph: DeclarationJson = {
      format: "karume-ir",
      version: 2,
      requires: { ops: ["relu", "matmul"] },
      symbols: ["N"],
      inputs: [
        { name: "x0", dtype: "f32", shape: [1, 1] },
        { name: "x1", dtype: "f32", shape: [1, "N"] },
      ],
      outputs: ["y"],
      initializers: {},
      values: {
        t: { dtype: "f32", shape: [1, 1] },
        y: { dtype: "f32", shape: [1, "N"] },
      },
      nodes: [
        { op: "relu", ins: ["x0"], outs: ["t"], attrs: {} },
        { op: "matmul", ins: ["t", "x1"], outs: ["y"], attrs: {} },
      ],
    };
    const session = await createSessionFromContainer(gpu, await openGraphModel(graph), "model");
    const small = () => ({ x0: fill([1, 1], () => 1), x1: fill([1, 4], () => 1) });
    try {
      await session.run(small());
      assertEquals(session.diagnostics().lastRunPrepared, { hit: false, cachedPlans: 1 });

      // 2 ノード目（matmul）が導出相で上限超過に落ちる。レシピ列は 1 ノード目のぶんまで
      // 出来ているが、登録は #buildRecipes が完走した後だけ。
      await assertRejects(
        () => session.run({ x0: fill([1, 1], () => 1), x1: fill([1, huge], () => 1) }),
        DispatchLimitError,
      );
      assertEquals(
        session.diagnostics().lastRunPrepared,
        undefined,
        "導出相で決着しなかった run は実績を残さない",
      );

      // 本数が増えていない = 失敗 run の部分レシピが載っていない。
      await session.run(small());
      assertEquals(session.diagnostics().lastRunPrepared, { hit: true, cachedPlans: 1 });
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);
