// 導出済み実行計画の Session 常駐キャッシュ（キー = 解決済み bindings）。計画・融合判定・
// レシピ導出は graph と bindings だけの純関数なので、同一 bindings の 2 run 目以降は導出相を
// 丸ごと飛ばせる（src/runtime/executor.ts の #preparedKey）。
//
// この門が見るのは「ヒット / ミスの判定」と「ヒット run の出力がミス run とビット単位で
// 一致する」の 2 つ。前者だけだと誤った計画を配っても緑になり、後者だけだとキャッシュが
// 黙って外れて毎 run 導出に戻っても緑になる（どちらも例外は出ない）。

import { assertEquals, assertRejects } from "@std/assert";
import { DispatchLimitError } from "../src/codegen/errors.ts";
import { openModel } from "../src/format/container.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import { defaultGemmGeometry, gemmTileN } from "../src/kernels/gemm-geometry.ts";
import { createSession, type Tensor } from "../src/runtime/executor.ts";
import { f32Bytes, type GraphJson } from "./helpers/format.ts";
import { fill, graphModelBuffer } from "./helpers/graph.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

/**
 * y = silu(x·w + b)（x: [T,3] → y: [T,2]）。末尾 2 ノードが silu 融合に掴まれるので、
 * 融合カウンタ（ADR 0040 §3 の常設診断）をヒット run 側でも見られる。
 */
const GRAPH: GraphJson = {
  format: "karume-ir",
  version: 1,
  requires: { ops: ["matmul", "add", "sigmoid", "mul"] },
  symbols: ["T"],
  inputs: [{ name: "x", dtype: "f32", shape: ["T", 3] }],
  outputs: ["y"],
  initializers: {
    w: { tensor: "proj.weight", storage: { dtype: "f32" } },
    b: { tensor: "proj.bias", storage: { dtype: "f32" } },
  },
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

const modelBytes = (): ArrayBuffer =>
  graphModelBuffer(GRAPH, [
    {
      name: "proj.weight",
      dtype: "F32",
      shape: [3, 2],
      data: f32Bytes([0.5, -1.5, 2, 0.25, -0.75, 1]),
    },
    { name: "proj.bias", dtype: "F32", shape: [2], data: f32Bytes([0.125, -0.5]) },
  ]);

const input = (rows: number): Tensor => ({
  dtype: "f32",
  shape: [rows, 3],
  data: Float32Array.from({ length: rows * 3 }, (_, i) => ((i % 9) - 4) * 0.5),
});

/** 出力のビット列。値の一致は「バイト同値」で見る（丸めの取り違えを許容しない）。 */
const bits = (tensor: Tensor): readonly number[] =>
  Array.from(new Uint32Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.length));

Deno.test({
  name: "同一 bindings の 2 run 目は導出済み計画に当たり、出力はビット単位で一致する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await createSession(gpu, openModel(modelBytes()));
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
});

Deno.test({
  name: "bindings が変わる run は導出し直し、元へ戻すと再びヒットする（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await createSession(gpu, openModel(modelBytes()));
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
});

Deno.test({
  name: "導出済み計画は上限 4 本で頭打ちになり、最古のものから追い出される（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await createSession(gpu, openModel(modelBytes()));
    try {
      for (const rows of [1, 2, 3, 4]) {
        await session.run({ x: input(rows) });
        assertEquals(
          session.diagnostics().lastRunPrepared,
          { hit: false, cachedPlans: rows },
          `T=${rows} は初出`,
        );
      }

      // 5 種類目で上限に当たる。載せてから最古（T=1）を落とすので本数は 4 のまま。
      await session.run({ x: input(5) });
      assertEquals(session.diagnostics().lastRunPrepared, { hit: false, cachedPlans: 4 });

      // 追い出しの証明: 一度当たっていた T=1 が再びミスになる。
      await session.run({ x: input(1) });
      assertEquals(
        session.diagnostics().lastRunPrepared,
        { hit: false, cachedPlans: 4 },
        "最古の bindings は落ちている",
      );
      // 残り 3 本は健在（追い出しが 1 本ずつであることの裏）。
      await session.run({ x: input(4) });
      assertEquals(session.diagnostics().lastRunPrepared, { hit: true, cachedPlans: 4 });
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "ヒット run も融合回数を報告する（キャッシュで常設診断が消えない・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await createSession(gpu, openModel(modelBytes()));
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
});

Deno.test({
  name: "導出相で落ちた run の部分レシピはキャッシュに載らない（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    // matmul の**出力タイル辺**で上限超過が決まる（`ceil(n / tileN) > 上限`）。辺は定数では
    // なく既定幾何から導く — 直接書くと辺を変えた瞬間に throw が起きず、assertRejects だけが
    // 静かに落ちる（runtime_executor_test.ts と同じ理由）。
    const tileN = gemmTileN(defaultGemmGeometry());
    const huge = gpu.limits.maxComputeWorkgroupsPerDimension * tileN + tileN;
    const graph: GraphJson = {
      format: "karume-ir",
      version: 1,
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
    const session = await createSession(gpu, openModel(graphModelBuffer(graph)));
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
});
