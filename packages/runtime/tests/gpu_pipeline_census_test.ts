// `SessionDiagnostics.lastRunPipelines`（直近 run の計画上の dispatch 本数 — パイプラインキー別）を
// 実 GPU で固定する門。純関数の振る舞いは runtime_pipeline_census_test.ts。
//
// 見るのは 3 つ:
// ① 計測ありの device で、実際に発行された dispatch の内訳（`lastRunTiming`）とキー集合・本数が
//    一致する（ミス run とヒット run の両方 — ヒット run は導出相を飛ばすので、census の源が
//    Session 累積のキー集合だと空振りする）。
// ② 計測の無い常駐経路（enqueue — ADR 0054）でも埋まる（この欄の存在理由）。
// ③ レシピ生成で落ちた run の直後は undefined（1 本前の成功 run の表を残さない）。

import { assert, assertEquals, assertRejects } from "@std/assert";
import { DispatchLimitError } from "../src/codegen/errors.ts";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import type { GpuTimingStats } from "../src/gpu/submit.ts";
import { defaultGemmGeometry, gemmTileN } from "../src/kernels/gemm-geometry.ts";
import { createSessionFromContainer, type Session } from "../src/runtime/executor.ts";
import { GPU_AVAILABLE, TIMESTAMP_QUERY_AVAILABLE } from "./helpers/gpu.ts";
import { type DeclarationJson, fill, GRAPH_NAME, openGraphModel } from "./helpers/model-fixture.ts";

type Census = readonly { readonly key: string; readonly dispatchCount: number }[];

/**
 * y = relu(neg(relu(x))) + relu(x)。relu を 2 回使うので、同じキーに 2 本載る行ができる
 * （elementwise のキーは op / rank / dtype で決まる — 本数の集計が空振りしていないことの材料）。
 * MUST: states を持たない（= workgroup 数が全て静的な 3 つ組）グラフにする。論理長から算出する
 * states 形の dispatch は仕事量ゼロの step で実際には積まれず、計画上の本数（census）が実発行
 * （timing）より多く出る — ここでは完全一致を断言したいので、その差が生じ得ない形に限る。
 */
const GRAPH: DeclarationJson = {
  format: "karume-ir",
  version: 2,
  requires: { ops: ["relu", "neg", "add"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [4, 4] }],
  outputs: ["y"],
  initializers: {},
  values: {
    a: { dtype: "f32", shape: [4, 4] },
    b: { dtype: "f32", shape: [4, 4] },
    c: { dtype: "f32", shape: [4, 4] },
    y: { dtype: "f32", shape: [4, 4] },
  },
  nodes: [
    { op: "relu", ins: ["x"], outs: ["a"], attrs: {} },
    { op: "neg", ins: ["a"], outs: ["b"], attrs: {} },
    { op: "relu", ins: ["b"], outs: ["c"], attrs: {} },
    { op: "add", ins: ["c", "a"], outs: ["y"], attrs: {} },
  ],
};

const input = () => ({ x: fill([4, 4], (i) => (i % 7) - 3) });

const openSession = async (gpu: GpuContext): Promise<Session> =>
  await createSessionFromContainer(gpu, await openGraphModel(GRAPH), GRAPH_NAME);

const byKey = (a: { readonly key: string }, b: { readonly key: string }): number =>
  a.key < b.key ? -1 : a.key > b.key ? 1 : 0;

/** 実発行の内訳を census と同じ形（キーの辞書順）へ写す（`entries` は ns の降順）。 */
const issued = (timing: GpuTimingStats): Census =>
  timing.entries.map(({ key, dispatchCount }) => ({ key, dispatchCount })).sort(byKey);

/** census が空振りしていないこと（空の表同士の一致で緑にしない）。 */
const assertNonTrivial = (census: Census | undefined): Census => {
  assert(census !== undefined, "lastRunPipelines が埋まっていない");
  assert(census.length > 0, "census が空（数え損ねを一致で隠す）");
  assert(
    census.some((entry) => entry.dispatchCount >= 2),
    "同じキーに 2 本以上載る行が無い（本数の集計が検査されていない）",
  );
  return census;
};

Deno.test({
  name:
    "lastRunPipelines は計測の実発行内訳とキー集合・本数が一致する（ミス / ヒット run・実 GPU）",
  // 計測（timestamp-query）が無い device では突合の相手が居ない — 明示 SKIP。
  ignore: !GPU_AVAILABLE || !TIMESTAMP_QUERY_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu({ gpuTiming: true });
    const session = await openSession(gpu);
    try {
      assertEquals(session.diagnostics().lastRunPipelines, undefined, "未実行なら undefined");
      for (const hit of [false, true]) {
        await session.run(input());
        const diagnostics = session.diagnostics();
        assertEquals(diagnostics.lastRunPrepared?.hit, hit);
        const timing = diagnostics.lastRunTiming;
        assert(timing !== undefined, "gpuTiming: true なのに内訳が無い");
        const census = assertNonTrivial(diagnostics.lastRunPipelines);
        assertEquals(census, issued(timing), `hit=${hit}: 計画上の本数と実発行が食い違う`);
        assertEquals(
          census.reduce((total, entry) => total + entry.dispatchCount, 0),
          timing.dispatchCount,
        );
      }
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "lastRunPipelines は計測の無い常駐経路（enqueue）でも埋まり、run と同じ表になる（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // 計測を要求しない device（計測有効だと beginBatch が拒否される — ADR 0054）。
    const gpu = await acquireGpu();
    const session = await openSession(gpu);
    try {
      const batch = await gpu.beginBatch();
      // 1 本目は enqueue 側で導出（ミス）、2 本目はヒット。どちらの直後も埋まっていること。
      // `#prepareInvocation` が毎回 undefined へ倒すので、2 本目の直後に埋まっているのは
      // 2 本目自身が書いた値（1 本目の残りではない）。
      const seen: Census[] = [];
      for (const hit of [false, true]) {
        await session.enqueue(input(), { batch });
        const diagnostics = session.diagnostics();
        assertEquals(diagnostics.lastRunPrepared?.hit, hit);
        assertEquals(diagnostics.lastRunTiming, undefined, "常駐経路は計測窓を作らない");
        seen.push(assertNonTrivial(diagnostics.lastRunPipelines));
      }
      await batch.finish();
      assertEquals(seen[1], seen[0], "同じ計画のミス / ヒットで表が変わる");

      // 対照: 同じ入力の run（同じ導出済み計画に当たる）と同じ表。
      await session.run(input());
      assertEquals(session.diagnostics().lastRunPrepared?.hit, true);
      assertEquals(session.diagnostics().lastRunPipelines, seen[0]);
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "レシピ生成で落ちた run の直後は lastRunPipelines が undefined（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    // 故障注入の位置は**レシピ生成**（`RecipeBuilder.buildRecipes` の matmul が dispatch 上限超過で
    // throw — gpu_prepared_plan_test.ts と同じ仕掛け）。出力タイル辺は既定幾何から導く（定数で
    // 書くと辺を変えた瞬間に throw が起きず、assertRejects だけが静かに落ちる）。
    // NOTE: `bindSymbols`（入力 shape の検証）と `planSteps`（計画・融合判定）の失敗は対象外 —
    // どちらも `#prepareInvocation` で「直近 run」の席を倒すより前に throw する。
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
    const session = await createSessionFromContainer(
      gpu,
      await openGraphModel(graph),
      GRAPH_NAME,
    );
    const small = () => ({ x0: fill([1, 1], () => 1), x1: fill([1, 4], () => 1) });
    try {
      // 対照: 成功 run の直後は埋まっている（失敗 run の undefined が「最初から空」ではない証明）。
      await session.run(small());
      const before = session.diagnostics().lastRunPipelines;
      assert(before !== undefined && before.length > 0);

      await assertRejects(
        () => session.run({ x0: fill([1, 1], () => 1), x1: fill([1, huge], () => 1) }),
        DispatchLimitError,
      );
      const failed = session.diagnostics();
      assertEquals(failed.lastRunPipelines, undefined, "落ちた run が 1 本前の表を語る");
      // 融合回数は落ちた run 自身の計画結果なので残る（倒れたのが「レシピ列が確定した後の席」だけ）。
      assertEquals(failed.lastRunFusions !== undefined, true);

      // 復帰: 同じ bindings の run は元の計画に当たり、同じ表を返す。
      await session.run(small());
      assertEquals(session.diagnostics().lastRunPrepared?.hit, true);
      assertEquals(session.diagnostics().lastRunPipelines, before);
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});
