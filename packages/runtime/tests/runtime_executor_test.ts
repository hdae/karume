// Session のライフサイクルと fail loudly（実 GPU — createSession が device を要求するため）。

import { assertEquals, assertRejects, assertThrows } from "@std/assert";
import { DispatchLimitError } from "../src/codegen/errors.ts";
import { ContainerError, openModel } from "../src/format/container.ts";
import { IrError } from "../src/format/ir.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import { GEMM_TILE } from "../src/kernels/gemm.ts";
import { DEFAULT_SUBMIT_POLICY } from "../src/gpu/submit.ts";
import { allclose, compareTensors, formatAllclose } from "../src/reference/allclose.ts";
import { applyReferenceOp, refTensor } from "../src/reference/ops.ts";
import { createSession, type Tensor } from "../src/runtime/executor.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import { f32Bytes, type GraphJson } from "./helpers/format.ts";
import { fill, graphModelBuffer } from "./helpers/graph.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

/** y = relu(x·w + b)（x: [T,4] → y: [T,3]）— 中間値 2 本を持つ最小の鎖。 */
const chainGraph = (): GraphJson => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["matmul", "add", "relu"] },
  symbols: ["T"],
  inputs: [{ name: "x", dtype: "f32", shape: ["T", 4] }],
  outputs: ["y"],
  initializers: {
    w: { tensor: "enc.w", storage: { dtype: "f32" } },
    b: { tensor: "enc.b", storage: { dtype: "f32" } },
  },
  values: {
    w: { dtype: "f32", shape: [4, 3] },
    b: { dtype: "f32", shape: [3] },
    h: { dtype: "f32", shape: ["T", 3] },
    g: { dtype: "f32", shape: ["T", 3] },
    y: { dtype: "f32", shape: ["T", 3] },
  },
  nodes: [
    { op: "matmul", ins: ["x", "w"], outs: ["h"], attrs: {} },
    { op: "add", ins: ["h", "b"], outs: ["g"], attrs: {} },
    { op: "relu", ins: ["g"], outs: ["y"], attrs: {} },
  ],
});

const W = Float32Array.from([0.5, -1, 0.25, 2, 0.125, -0.5, -3, 1.5, 0.75, 1, -0.25, 0.5]);
const B = Float32Array.from([1, -2, 0.5]);

const chainModelBuffer = (graph: GraphJson = chainGraph()): ArrayBuffer =>
  graphModelBuffer(graph, [
    { name: "enc.w", dtype: "F32", shape: [4, 3], data: f32Bytes([...W]) },
    { name: "enc.b", dtype: "F32", shape: [3], data: f32Bytes([...B]) },
  ]);

const expectedChain = (x: Tensor): ArrayLike<number> => {
  const h = applyReferenceOp("matmul", [x, refTensor([4, 3], W)]);
  const g = applyReferenceOp("add", [h, refTensor([3], B)]);
  return applyReferenceOp("relu", [g]).data;
};

Deno.test({
  name: "initializer を使う鎖グラフが CPU 参照と一致し、出力はグラフ出力だけ返る（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await createSession(gpu, openModel(chainModelBuffer()));
    try {
      const x = fill([5, 4], (i) => ((i % 11) - 5) * 0.5);
      const outputs = await session.run({ x });
      assertEquals(Object.keys(outputs), ["y"]);
      assertEquals(outputs["y"].shape, [5, 3]);
      const report = allclose(outputs["y"].data, expectedChain(x));
      assertEquals(report.pass, true, formatAllclose(report));

      // 同じ Session で別の T を束縛し直せる
      const x2 = fill([2, 4], (i) => i - 3);
      const outputs2 = await session.run({ x: x2 });
      assertEquals(outputs2["y"].shape, [2, 3]);
      assertEquals(allclose(outputs2["y"].data, expectedChain(x2)).pass, true);
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

/**
 * "__proto__" を名前に使う最小グラフ。入力名 / シンボル名（前者）と出力名（後者）で分ける
 * — IR は 1 つの名前をちょうど 1 箇所でしか宣言できず、1 本のグラフに同居させられない。
 *
 * MUST: `values` に "__proto__" を置くときは計算キー `["__proto__"]` で書く。リテラルの
 * `__proto__:` は own key ではなく [[Prototype]] 指定になり、JSON.stringify に載らないまま
 * テストが検査対象を外す。
 */
const protoInputGraph = (): GraphJson => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["relu"] },
  symbols: ["__proto__"],
  inputs: [{ name: "__proto__", dtype: "f32", shape: ["__proto__", 3] }],
  outputs: ["y"],
  initializers: {},
  values: { y: { dtype: "f32", shape: ["__proto__", 3] } },
  nodes: [{ op: "relu", ins: ["__proto__"], outs: ["y"], attrs: {} }],
});

const protoOutputGraph = (): GraphJson => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["relu"] },
  symbols: ["T"],
  inputs: [{ name: "x", dtype: "f32", shape: ["T", 3] }],
  outputs: ["__proto__"],
  initializers: {},
  values: { ["__proto__"]: { dtype: "f32", shape: ["T", 3] } },
  nodes: [{ op: "relu", ins: ["x"], outs: ["__proto__"], attrs: {} }],
});

Deno.test({
  name: "'__proto__' という入力名 / シンボル名 / 出力名が黙って消えない（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const x = fill([4, 3], (i) => ((i % 7) - 3) * 0.5);
    const expected = applyReferenceOp("relu", [x]).data;
    try {
      const byInput = await createSession(gpu, openModel(graphModelBuffer(protoInputGraph())));
      try {
        // 計算キー MUST（上のフィクスチャと同じ理由 — リテラルでは run に名前が届かない）
        const outputs = await byInput.run({ ["__proto__"]: x });
        assertEquals(allclose(outputs["y"].data, expected).pass, true);
      } finally {
        await byInput.dispose();
      }

      const byOutput = await createSession(gpu, openModel(graphModelBuffer(protoOutputGraph())));
      try {
        const outputs = await byOutput.run({ x });
        // 素の `{}` に戻すと、ブラウザでは Tensor が [[Prototype]] に化けてこの出力だけ
        // 消える。Deno では own property が作られてしまい hasOwn / keys では退行を検出
        // できないため、器が null プロトタイプであることを併せて固定する。
        assertEquals(Object.getPrototypeOf(outputs), null);
        assertEquals(Object.hasOwn(outputs, "__proto__"), true);
        assertEquals(Object.keys(outputs), ["__proto__"]);
        assertEquals(allclose(outputs["__proto__"].data, expected).pass, true);
      } finally {
        await byOutput.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
});

/**
 * reshape / 恒等 expand の別名（ADR 0011）を踏む鎖。
 *
 * - `r` は中間値 `h` の実バッファをそのまま指すグラフ出力（ピン留めが別名越しに効くこと）
 * - `g` の確保は `h` のバッファを掴んではいけない（別名 `r` がまだ生きている）
 * - `q` は別名のまま消費だけされる（消費計数が別名越しに正しく減ること）
 *
 * 計数を 1 でも取り違えると、`g` の確保がプールから `h` のバッファを受け取り、`r` の中身が
 * 静かに exp(x) に化ける（例外は出ない）。
 */
const aliasGraph = (aliasOp: "reshape" | "expand"): GraphJson => {
  // 恒等 expand は複製軸を持たないので入出力 shape が完全一致する形でしか別名化されない。
  const aliasShape = aliasOp === "reshape" ? ["2T", 2] : ["T", 4];
  return {
    format: "karume-ir",
    version: 1,
    requires: { ops: ["neg", aliasOp, "exp", "add"] },
    symbols: ["T"],
    inputs: [{ name: "x", dtype: "f32", shape: ["T", 4] }],
    outputs: ["r", "s"],
    initializers: {},
    values: {
      h: { dtype: "f32", shape: ["T", 4] },
      r: { dtype: "f32", shape: aliasShape },
      g: { dtype: "f32", shape: ["T", 4] },
      q: { dtype: "f32", shape: aliasShape },
      s: { dtype: "f32", shape: aliasShape },
    },
    nodes: [
      { op: "neg", ins: ["x"], outs: ["h"], attrs: {} },
      { op: aliasOp, ins: ["h"], outs: ["r"], attrs: {} },
      { op: "exp", ins: ["x"], outs: ["g"], attrs: {} },
      { op: aliasOp, ins: ["g"], outs: ["q"], attrs: {} },
      { op: "add", ins: ["r", "q"], outs: ["s"], attrs: {} },
    ],
  };
};

for (
  const [aliasOp, expectedShape, identityExpands] of [
    ["reshape", [6, 2], 0],
    ["expand", [3, 4], 2],
  ] as const
) {
  Deno.test({
    name: `${aliasOp} の別名がコピー無しで、生きている別名を後続の確保に配り直さない（実 GPU）`,
    ignore: !GPU_AVAILABLE,
    fn: async () => {
      const gpu = await acquireGpu();
      const session = await createSession(gpu, openModel(graphModelBuffer(aliasGraph(aliasOp))));
      try {
        const x = fill([3, 4], (i) => ((i % 7) - 3) * 0.5);
        const outputs = await session.run({ x });
        assertEquals(Object.keys(outputs).sort(), ["r", "s"]);
        assertEquals(outputs["r"].shape, expectedShape);
        assertEquals(outputs["s"].shape, expectedShape);

        const negated = applyReferenceOp("neg", [x]);
        const exponent = applyReferenceOp("exp", [x]);
        // 別名でもグラフ出力の readback は元の値のまま（reshape なら形だけ [6,2] になる）
        assertEquals(allclose(outputs["r"].data, negated.data).pass, true, "別名出力の readback");
        const sum = applyReferenceOp("add", [negated, exponent]);
        assertEquals(allclose(outputs["s"].data, sum.data).pass, true, "別名を消費した先の値");

        // 実体化コピーが出ていないことを実績で固定する。プール管理下（dispatch が書く出力
        // ストレージ）の生存ピークは h / g / s の 3 本ぶん = 48×3。別名側が確保していれば
        // r / q のぶんが上乗せされる。
        assertEquals(session.diagnostics().lastRun?.peakTransientBytes, 144);
        // dispatch は neg / exp / add の 3 本だけ（reshape / 恒等 expand はどちらも 0 本）
        assertEquals(session.diagnostics().submit.dispatchCount, 3);
        assertEquals(
          session.diagnostics().lastRunFusions?.identityExpand,
          identityExpands,
          "別名化カウンタ",
        );
      } finally {
        await session.dispose();
        gpu.destroy();
      }
    },
  });
}

Deno.test({
  name: "同一 Session への並行 run は直列化され、全件が CPU 参照と一致する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await createSession(gpu, openModel(chainModelBuffer()));
    try {
      const batch = [
        fill([5, 4], (i) => ((i % 11) - 5) * 0.5),
        fill([2, 4], (i) => i - 3),
        fill([7, 4], (i) => ((i % 7) - 3) * 0.25),
        fill([1, 4], (i) => i + 1),
      ];
      // await せずまとめて発行する。直列化されていないと device 単位 LIFO の errorScope が
      // run 同士で交錯し、自分の validation 失敗を取り逃がすか無関係な run が落ちる。
      const outputs = await Promise.all(batch.map((x) => session.run({ x })));
      for (const [index, x] of batch.entries()) {
        assertEquals(outputs[index]["y"].shape, [x.shape[0], 3], `run ${index} の shape`);
        const report = allclose(outputs[index]["y"].data, expectedChain(x));
        assertEquals(report.pass, true, `run ${index}: ${formatAllclose(report)}`);
      }
      assertEquals(session.diagnostics().submit.dispatchCount, 3 * batch.length);
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "実行中の run がある間の dispose は run の完了後に重みを破棄する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await createSession(gpu, openModel(chainModelBuffer()));
    try {
      const x = fill([6, 4], (i) => ((i % 9) - 4) * 0.5);
      const running = session.run({ x });
      // await せずに dispose する。直列化されていないと実行中の run の下から重みバッファが
      // destroy され、submit がコマンドバッファ丸ごと失敗して誤った値が静かに残る。
      const disposing = session.dispose();
      const outputs = await running;
      await disposing;
      assertEquals(allclose(outputs["y"].data, expectedChain(x)).pass, true);
      // dispose の受理は呼び出し時点で確定する（以後の run は待たずに落ちる）
      await assertRejects(() => session.run({ x }), ExecutionError);
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "submitPolicy でチャンクを 1 に絞ると dispatch ごとに submit される（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    // 既定の initialChunkSize=16 では 3 dispatch のグラフがチャンク途中 submit を通らない
    const session = await createSession(gpu, openModel(chainModelBuffer()), {
      submitPolicy: {
        timeBudgetMs: DEFAULT_SUBMIT_POLICY.timeBudgetMs,
        initialChunkSize: 1,
        minChunkSize: 1,
        maxChunkSize: 1,
      },
    });
    try {
      const x = fill([6, 4], (i) => ((i % 7) - 3) * 0.5);
      const outputs = await session.run({ x });
      const submit = session.diagnostics().submit;
      assertEquals(submit.dispatchCount, 3);
      assertEquals(submit.submitCount, 3, "チャンク上限 1 なので dispatch ごとに submit される");
      const report = allclose(outputs["y"].data, expectedChain(x));
      assertEquals(report.pass, true, formatAllclose(report));
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "診断がパイプライン数と直近 run の中間バッファ再利用を報告する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await createSession(gpu, openModel(chainModelBuffer()));
    try {
      assertEquals(session.diagnostics().lastRun, undefined);
      // 重み 2 本は明示 async ステージでアップロード済み
      assertEquals(session.diagnostics().weights.allocCount, 2);

      await session.run({ x: fill([8, 4], (i) => i * 0.25) });
      const stats = session.diagnostics();
      assertEquals(stats.pipelineCount, 3);
      assertEquals(stats.submit.dispatchCount, 3);
      // h は add のエンコード後に解放され、relu の出力確保で配り直される
      assertEquals((stats.lastRun?.reuseCount ?? 0) >= 1, true);
      assertEquals((stats.lastRun?.peakTransientBytes ?? 0) > 0, true);
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

/**
 * `d` は誰にも読まれずグラフ出力でもない中間出力（IR は到達不能な値を禁じていない）。
 * 消費者ゼロの値はノード境界で解放されないとプール再利用から外れ、確保が 1 本増えて
 * peakTransientBytes が実際より大きく出る。
 */
const deadValueGraph = (): GraphJson => ({
  ...chainGraph(),
  values: {
    w: { dtype: "f32", shape: [4, 3] },
    b: { dtype: "f32", shape: [3] },
    h: { dtype: "f32", shape: ["T", 3] },
    d: { dtype: "f32", shape: ["T", 3] },
    g: { dtype: "f32", shape: ["T", 3] },
    y: { dtype: "f32", shape: ["T", 3] },
  },
  nodes: [
    { op: "matmul", ins: ["x", "w"], outs: ["h"], attrs: {} },
    { op: "relu", ins: ["h"], outs: ["d"], attrs: {} },
    { op: "add", ins: ["h", "b"], outs: ["g"], attrs: {} },
    { op: "relu", ins: ["g"], outs: ["y"], attrs: {} },
  ],
});

Deno.test({
  name: "消費者ゼロの中間出力もノード境界でプールへ戻る（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await createSession(gpu, openModel(chainModelBuffer(deadValueGraph())));
    try {
      const x = fill([8, 4], (i) => ((i % 11) - 5) * 0.5);
      const outputs = await session.run({ x });
      // 到達不能な d が挟まっても最終結果は鎖グラフと同じ
      const report = allclose(outputs["y"].data, expectedChain(x));
      assertEquals(report.pass, true, formatAllclose(report));

      // 中間 4 本（h/d/g/y）は 2 本の実確保で回る。d が居座ると reuse は 1・peak は 3 本分。
      const lastRun = session.diagnostics().lastRun;
      assertEquals(lastRun?.reuseCount, 2);
      assertEquals(lastRun?.peakTransientBytes, 2 * 8 * 3 * 4);
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "capability 不足（非対応 op / 非対応 格納 dtype）は createSession で落ちる（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      // conv_transpose2d は語彙にすら無い op（ADR 0007 の allowlist 凍結）— 契約表に
      // 載っていないので、出たら capability 不足として列挙で落ちる。
      // NOTE: conv2d は ADR 0017 で契約表へ入った（Anima の VAE decoder で実測に出た）ので、
      // 「保存はする / 実行はできない」の代表はこちらが引き継ぐ。
      const foreign = chainGraph();
      foreign.requires.ops = ["matmul", "add", "conv_transpose2d"];
      foreign.nodes[2] = { op: "conv_transpose2d", ins: ["g"], outs: ["y"], attrs: {} };
      await assertRejects(
        () => createSession(gpu, openModel(chainModelBuffer(foreign))),
        ContainerError,
      );

      // NOTE: 非対応格納の代表は **bf16**（f16 は ADR 0018 で実行経路が入った — 適格判定に
      // 関わらず実行できるので、もう capability 不足にはならない）。
      const quantized = chainGraph();
      quantized.initializers.w = { tensor: "enc.w", storage: { dtype: "bf16" } };
      const buffer = graphModelBuffer(quantized, [
        { name: "enc.w", dtype: "BF16", shape: [4, 3], data: new Uint8Array(24) },
        { name: "enc.b", dtype: "F32", shape: [3], data: f32Bytes([...B]) },
      ]);
      await assertRejects(() => createSession(gpu, openModel(buffer)), ContainerError);
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "契約外のグラフ（非空 attrs / op が受理しない dtype）は createSession で落ちる（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      // どちらも capability 突合（op 名 + dtype + attrs）が最初の門で捕まえる
      const withAttrs = chainGraph();
      withAttrs.nodes[2].attrs = { approximate: "tanh" };
      await assertRejects(
        () => createSession(gpu, openModel(chainModelBuffer(withAttrs))),
        ContainerError,
        "未実装 attrs (1): nodes[2] (relu): approximate",
      );

      // i32 の転送自体は解禁済み（ADR 0009）だが、matmul は f32 専業なので op 契約で落ちる
      const intInput = chainGraph();
      intInput.inputs = [{ name: "x", dtype: "i32", shape: ["T", 4] }];
      await assertRejects(
        () => createSession(gpu, openModel(chainModelBuffer(intInput))),
        ContainerError,
        "非対応 意味論 dtype (1): 値 'x': i32",
      );
    } finally {
      gpu.destroy();
    }
  },
});

/**
 * 実測グラフ（DeBERTa front）の mask 経路をそのまま縮めたもの:
 * mask 外積 mul(i32) → 真偽化 cast → bitwise_not → 重み化 cast → f32 の重み掛け。
 * 出力に bool を 1 本置いて readback の非 f32 経路も踏む。
 */
const maskGraph = (): GraphJson => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["mul", "cast", "bitwise_not"] },
  symbols: ["T"],
  inputs: [
    { name: "mcol", dtype: "i32", shape: ["T", 1] },
    { name: "mrow", dtype: "i32", shape: [1, "T"] },
    { name: "scores", dtype: "f32", shape: ["T", "T"] },
  ],
  outputs: ["masked", "drop"],
  initializers: {},
  values: {
    pair: { dtype: "i32", shape: ["T", "T"] },
    keep: { dtype: "bool", shape: ["T", "T"] },
    drop: { dtype: "bool", shape: ["T", "T"] },
    weight: { dtype: "f32", shape: ["T", "T"] },
    masked: { dtype: "f32", shape: ["T", "T"] },
  },
  nodes: [
    { op: "mul", ins: ["mcol", "mrow"], outs: ["pair"], attrs: {} },
    { op: "cast", ins: ["pair"], outs: ["keep"], attrs: { to: "bool" } },
    { op: "bitwise_not", ins: ["keep"], outs: ["drop"], attrs: {} },
    { op: "cast", ins: ["keep"], outs: ["weight"], attrs: { to: "f32" } },
    { op: "mul", ins: ["scores", "weight"], outs: ["masked"], attrs: {} },
  ],
});

Deno.test({
  name: "i32 入力 → bool 中間 → 非 f32 出力の鎖が CPU 参照と厳密一致する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await createSession(gpu, openModel(graphModelBuffer(maskGraph())));
    try {
      const mcol = fill([4, 1], (i) => (i === 2 ? 0 : 1), "i32");
      const mrow = fill([1, 4], (i) => (i === 3 ? 0 : 1), "i32");
      const scores = fill([4, 4], (i) => (i % 7) - 3);
      const outputs = await session.run({ mcol, mrow, scores });

      const pair = applyReferenceOp("mul", [mcol, mrow]);
      const keep = applyReferenceOp("cast", [pair], { to: "bool" });
      const drop = applyReferenceOp("bitwise_not", [keep]);
      const weight = applyReferenceOp("cast", [keep], { to: "f32" });
      const masked = applyReferenceOp("mul", [scores, weight]);

      // bool 出力は Uint32Array（u32 の 0/1）で返る — f32 固定の readback では通らない形
      assertEquals(outputs["drop"].dtype, "bool");
      assertEquals(outputs["drop"].data.constructor.name, "Uint32Array");
      assertEquals(compareTensors(outputs["drop"], drop).pass, true);
      assertEquals(outputs["masked"].dtype, "f32");
      assertEquals(compareTensors(outputs["masked"], masked).pass, true);
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

// MUST: 要素は全型 4 バイトなので、TypedArray の取り違えは writeBuffer を素通りして
// ビット列の読み替えになる（例外は出ない）。宣言 dtype との突合が唯一の防波堤。
Deno.test({
  name: "宣言 dtype と食い違う入力テンソルは転送前に落ちる（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await createSession(gpu, openModel(graphModelBuffer(maskGraph())));
    try {
      const mrow = fill([1, 4], () => 1, "i32");
      const scores = fill([4, 4], () => 1);
      // i32 宣言の入力に f32 の Tensor（判別子も配列型も違う）
      await assertRejects(
        () => session.run({ mcol: fill([4, 1], () => 1), mrow, scores }),
        ExecutionError,
        "宣言 dtype 'i32'",
      );
      // 判別子だけ合わせて配列型を偽った形（TS では作れないので明示的に組む）
      const spoofed = {
        dtype: "i32",
        shape: [4, 1],
        data: Float32Array.from([1, 1, 1, 1]),
      } as unknown as Tensor;
      await assertRejects(
        () => session.run({ mcol: spoofed, mrow, scores }),
        ExecutionError,
        "Float32Array",
      );
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "run の入力不整合と dispose 後の実行は fail loudly（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await createSession(gpu, openModel(chainModelBuffer()));
    try {
      // 宣言 [T,4] に対する rank / 数値次元の不一致
      await assertRejects(() => session.run({ x: fill([5], () => 1) }), ExecutionError);
      await assertRejects(() => session.run({ x: fill([5, 3], () => 1) }), ExecutionError);
      // shape と要素数の食い違い（shape は宣言に合うが data が足りない）
      await assertRejects(
        () =>
          session.run({ x: { dtype: "f32", shape: [5, 4], data: Float32Array.from([1, 2, 3]) } }),
        ExecutionError,
      );
      // 明示束縛が入力 shape 由来の束縛と衝突
      await assertRejects(
        () => session.run({ x: fill([5, 4], () => 1) }, { T: 4 }),
        ExecutionError,
      );
      // 入力の欠落
      await assertRejects(() => session.run({}), ExecutionError);
    } finally {
      await session.dispose();
    }
    await assertRejects(() => session.run({ x: fill([5, 4], () => 1) }), ExecutionError);
    // dispose は冪等
    await session.dispose();
    gpu.destroy();
  },
});

Deno.test({
  name: "宣言 shape と計算 shape の食い違いは 1 dispatch も出さずに落ちる（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    // IR は shape を計算しないので、宣言の誤りは実行計画の段で初めて見つかる
    const wrong = chainGraph();
    wrong.values.h = { dtype: "f32", shape: ["T", 4] };
    wrong.values.g = { dtype: "f32", shape: ["T", 4] };
    wrong.values.y = { dtype: "f32", shape: ["T", 4] };
    const session = await createSession(gpu, openModel(chainModelBuffer(wrong)));
    try {
      await assertRejects(() => session.run({ x: fill([3, 4], () => 1) }), ExecutionError);
      assertEquals(session.diagnostics().submit.dispatchCount, 0);
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "エンコード途中で落ちた run は残 pending dispatch を submit せず捨てる（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    // 1 ノード目（relu）は dispatch が積まれ、2 ノード目（matmul）が上限超過で throw する。
    // 既定チャンクサイズ 16 なので relu のぶんは未 submit のまま残る。
    // MUST: 上限超過の閾値は matmul の**出力タイル辺**で決まる（`ceil(n / GEMM_TILE) > 上限`）。
    // タイル辺を定数から引かないと、辺を変えた瞬間に throw が起きず assertRejects だけが
    // 静かに落ちる（この波で 16 → 64 になった）。
    const n = gpu.limits.maxComputeWorkgroupsPerDimension * GEMM_TILE + GEMM_TILE;
    const graph: GraphJson = {
      format: "karume-ir",
      version: 1,
      requires: { ops: ["relu", "matmul"] },
      symbols: [],
      inputs: [
        { name: "x0", dtype: "f32", shape: [1, 1] },
        { name: "x1", dtype: "f32", shape: [1, n] },
      ],
      outputs: ["y"],
      initializers: {},
      values: {
        t: { dtype: "f32", shape: [1, 1] },
        y: { dtype: "f32", shape: [1, n] },
      },
      nodes: [
        { op: "relu", ins: ["x0"], outs: ["t"], attrs: {} },
        { op: "matmul", ins: ["t", "x1"], outs: ["y"], attrs: {} },
      ],
    };
    const session = await createSession(gpu, openModel(graphModelBuffer(graph)));
    try {
      await assertRejects(
        () => session.run({ x0: fill([1, 1], () => 1), x1: fill([1, n], () => 1) }),
        DispatchLimitError,
      );

      const stats = session.diagnostics().submit;
      assertEquals(stats.dispatchCount, 1, "relu のぶんは実際に積まれていた");
      // 残骸を後始末（arena.destroy → flush）で出すと、ロック解放後・errorScope ゼロ本の
      // submit になり、その validation エラーが並行する別 Session の区間に帰属する。
      assertEquals(stats.submitCount, 0, "失敗した run の残骸は 1 度も submit されない");
      assertEquals(stats.discardedDispatches, 1, "捨てた件数が診断に出る");
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "束縛が取れないシンボルは IR パーサが受理しない（入力 shape に素の形で現れない）",
  fn: () => {
    const unbindable = chainGraph();
    unbindable.symbols = ["T", "S"];
    unbindable.values.h = { dtype: "f32", shape: ["S", 3] };
    assertThrows(() => openModel(chainModelBuffer(unbindable)), IrError, "束縛が取れない");
  },
});
