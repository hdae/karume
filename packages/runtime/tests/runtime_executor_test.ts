// Session のライフサイクルと fail loudly（実 GPU — createSession が device を要求するため）。

import { assertEquals, assertRejects, assertThrows } from "@std/assert";
import { DispatchLimitError } from "../src/codegen/errors.ts";
import { ContainerError, openModel } from "../src/format/container.ts";
import { IrError } from "../src/format/ir.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import { defaultGemmGeometry, gemmTileN } from "../src/kernels/gemm-geometry.ts";
import { DEFAULT_SUBMIT_POLICY } from "../src/gpu/submit.ts";
import { allclose, compareTensors, formatAllclose } from "../src/reference/allclose.ts";
import { applyReferenceOp, applyReferenceOpOutputs, refTensor } from "../src/reference/ops.ts";
import { createSession, type SessionOptions, type Tensor } from "../src/runtime/executor.ts";
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

/**
 * 多出力ノードの寿命（ADR 0068 受入条件 ④ — `topk` が最初の入居者）。
 *
 * `topk` の 2 本は**寿命が違う**:
 *
 * - `v`（値）は `neg` に 1 度だけ消費される中間値 → そのステップの後でプールへ返る
 * - `i`（添字）はグラフ出力 → **ピン留めされてプールへ返らない**
 * - `e = exp(neg(v))` の確保は `v` の実体を掴んでよい（`v` は消費済み）が、`i` の実体を
 *   掴んではいけない
 *
 * slot ごとの `uses` / `pinned` を 1 でも取り違えると、`e` の確保がプールから `i` の
 * バッファを受け取り、**添字の readback が静かに exp の f32 ビット列に化ける**（例外は
 * 出ない — 別名テストと同じ形の沈黙故障）。逆に `v` を解放し損ねれば `e` が新しい実体を
 * 掴み、生存ピークが 1 本ぶん増える。
 *
 * 2 本目のグラフは**片方の出力が誰にも消費されない**形（`uses = 0` かつグラフ出力でもない
 * 到達不能な値）。定義ぶんの retain が解放されないと run 末尾の `assertDrained` が落ちる。
 *
 * MUST: どちらの形も**同一 Session で 2 回**走らせる。slot backing はヒット run でしか活性化
 * しないので（`Session.#activateBacking`）、1 回ずつでは `bakeBindGroups` の slot 1 束縛・
 * pin 済み出力写像・backed readback の欠落や入れ替わりがアリーナ経路の緑の裏に隠れる。
 */
const topkLifetimeGraph = (pinIndices: boolean): GraphJson => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["topk", "neg", "exp"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [3, 8] }],
  outputs: pinIndices ? ["e", "i"] : ["e"],
  initializers: {},
  values: {
    v: { dtype: "f32", shape: [3, 2] },
    i: { dtype: "i32", shape: [3, 2] },
    nv: { dtype: "f32", shape: [3, 2] },
    e: { dtype: "f32", shape: [3, 2] },
  },
  nodes: [
    { op: "topk", ins: ["x"], outs: ["v", "i"], attrs: { k: 2 } },
    { op: "neg", ins: ["v"], outs: ["nv"], attrs: {} },
    { op: "exp", ins: ["nv"], outs: ["e"], attrs: {} },
  ],
});

Deno.test({
  name:
    "多出力ノードの slot ごとの uses / pin が正しく、片方だけ消費する形でも値が壊れない（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      const x = fill([3, 8], (i) => ((i % 7) - 3) * 0.5 + (i % 3) * 0.25);
      const expected = applyReferenceOpOutputs("topk", [x], { k: 2 });
      const negated = applyReferenceOp("neg", [expected[0]]);
      const exponent = applyReferenceOp("exp", [negated]);

      // ① 添字をグラフ出力にする形（ピン留めが多出力の slot 1 に効くこと）
      const session = await createSession(
        gpu,
        openModel(graphModelBuffer(topkLifetimeGraph(true))),
      );
      try {
        const outputs = await session.run({ x });
        assertEquals(Object.keys(outputs).sort(), ["e", "i"]);
        assertEquals(outputs["i"].dtype, "i32");
        // 添字は 1 語も化けていない（プールから配り直されていれば f32 のビット列が出る）
        assertEquals([...outputs["i"].data], [...expected[1].data], "添字の readback");
        assertEquals(allclose(outputs["e"].data, exponent.data).pass, true, "値側を消費した先");
        // プール管理下の生存ピークは v / i / nv の 3 本ぶん（各 24 バイト）= 72。
        // `e` は消費済みの `v` の実体を再利用するので上乗せは無く、`i` を掴むこともない。
        assertEquals(session.diagnostics().lastRun?.peakTransientBytes, 72);
        // dispatch は topk / neg / exp の 3 本（topk は 1 dispatch で 2 本書く）
        assertEquals(session.diagnostics().submit.dispatchCount, 3);

        // 同じ Session の 2 回目 = 導出済み計画にヒットして **slot backing 経路**へ入る
        // （bind group は構築時に焼き込み済みで、読み戻し先も構築時に確定した写像）。多出力の
        // slot 1 の束縛・pin 済み出力写像・backed readback のどれが欠けても・入れ替わっても、
        // アリーナ経路の 1 回目だけなら緑のまま通る。
        const again = await session.run({ x });
        assertEquals(session.diagnostics().lastRunPrepared?.hit, true, "2 run 目は導出済み計画");
        assertEquals(session.diagnostics().planBacking.buildCount, 1, "backing を構築した");
        assertEquals(Object.keys(again).sort(), ["e", "i"]);
        assertEquals(again["i"].dtype, "i32");
        assertEquals([...again["i"].data], [...expected[1].data], "backed 経路の添字 readback");
        assertEquals(allclose(again["e"].data, exponent.data).pass, true, "backed 経路の値側");
        // 積むコマンド列はアリーナ経路と同一（3 本 → 累計 6 本）
        assertEquals(session.diagnostics().submit.dispatchCount, 6);
      } finally {
        await session.dispose();
      }

      // ② 添字が誰にも消費されない形（到達不能な値 — 定義ぶんの解放が閉じることの確認。
      // 閉じていなければ run 末尾の assertDrained が落ちる）
      const orphan = await createSession(
        gpu,
        openModel(graphModelBuffer(topkLifetimeGraph(false))),
      );
      try {
        const outputs = await orphan.run({ x });
        assertEquals(Object.keys(outputs), ["e"]);
        assertEquals(allclose(outputs["e"].data, exponent.data).pass, true, "値側は不変");
        // 消費者ゼロの添字はそのステップの末尾でプールへ戻るので、`nv` がその実体を掴む
        // （生存ピークは v / i の 2 本ぶん = 48）。
        assertEquals(orphan.diagnostics().lastRun?.peakTransientBytes, 48);

        // orphan 形の 2 回目も backing 経路。読み戻し先の写像が slot を取り違えれば、`e` が
        // pin されていない実体を掴む（`Session.#isReadable` が pin 済み slot だけを許すので
        // 多くは例外で落ち、通ってしまう形はここの値突合が拾う）。
        const again = await orphan.run({ x });
        assertEquals(orphan.diagnostics().lastRunPrepared?.hit, true, "2 run 目は導出済み計画");
        assertEquals(orphan.diagnostics().planBacking.buildCount, 1, "backing を構築した");
        assertEquals(Object.keys(again), ["e"]);
        assertEquals(allclose(again["e"].data, exponent.data).pass, true, "backed 経路でも不変");
      } finally {
        await orphan.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
});

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
      // 単一フェンス経路（gpuTiming OFF・出力 1 本）では読み戻しの copy も**同じコマンド列**へ
      // 積まれ、チャンク上限 1 の下では 1 件で 1 チャンクを占める（H-1）。したがって
      // submit は「3 dispatch + copy 1 件」の 4 本。dispatch 数は 3 のまま。
      assertEquals(submit.submitCount, 4, "チャンク上限 1 なのでコマンドごとに submit される");
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
 * **型の外から来る呼び手**（JS の消費者・改名前の綴りを残した古いビルド）を作る。TS の型は
 * この経路を 1 つも守らないので、受理集合の検査はここでしか撃てない。
 */
const jsCallerOptions = (options: Readonly<Record<string, string>>): SessionOptions =>
  options as unknown as SessionOptions;

Deno.test({
  name: "実行形ノブの union 外の綴りは Session 構築で全件列挙して落ちる（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      // `"i8a8"` は 0.5.0 で `"a8"` へ改名した綴り（ADR 0074 決定 3・互換シム無し）。検査が
      // 無いと下流の等値比較（`=== "a8"`）が全て外れ、opt-in が適用されないまま既定の f32 で
      // 走る = 「a8 を測った」と読める沈黙。
      const error = await assertRejects(
        () =>
          createSession(
            gpu,
            openModel(chainModelBuffer()),
            jsCallerOptions({ linearCompute: "i8a8", attentionScoreStorage: "s16" }),
          ),
        ExecutionError,
        "実行形ノブ 2 本",
      );
      assertEquals(error.message.includes(`linearCompute: "i8a8"`), true, error.message);
      assertEquals(error.message.includes(`attentionScoreStorage: "s16"`), true, error.message);
      // 受理集合そのものを文言に出す（何なら通るのかが診断だけで分かる）。
      assertEquals(error.message.includes("'f32' / 'a8' / 'f16'"), true, error.message);

      // 対照: union 内の綴りは従来どおり構築が通る（上が「何を渡しても落ちる」ではない証明）。
      const session = await createSession(gpu, openModel(chainModelBuffer()), {
        linearCompute: "a8",
        attentionScoreStorage: "f16",
      });
      await session.dispose();
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "失敗した run の直後の診断は 1 本前の成功 run の実績を残さない（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await createSession(gpu, openModel(chainModelBuffer()));
    try {
      // 対照: 成功 run の直後は「直近 run」の 4 席が揃って埋まる。
      await session.run({ x: fill([8, 4], (i) => i * 0.25) });
      const success = session.diagnostics();
      assertEquals(success.lastRun !== undefined, true);
      assertEquals(success.lastRunParams !== undefined, true);
      assertEquals(success.lastRunFusions !== undefined, true);
      assertEquals(success.lastRunPrepared !== undefined, true);

      // 導出（融合判定まで）は通り、入力検査（宣言 f32 に i32 を渡す）で落ちる run。
      await assertRejects(
        () => session.run({ x: { dtype: "i32", shape: [8, 4], data: new Int32Array(32) } }),
        ExecutionError,
        "宣言 dtype",
      );

      // 4 席は同じ 1 本の run（= 落ちた run）を映す。融合回数だけは落ちた run 自身の導出結果
      // なので残り、成功経路でしか埋まらない 3 席は undefined へ倒れる — ここが割れると、
      // 1 つのスナップショットが「落ちた run の融合回数」と「1 本前の成功 run のアリーナ /
      // params 実績」を混ぜて語る（失敗の切り分けで最も読みたい瞬間の診断）。
      const failed = session.diagnostics();
      assertEquals(failed.lastRun, undefined);
      assertEquals(failed.lastRunParams, undefined);
      assertEquals(failed.lastRunPrepared, undefined);
      assertEquals(failed.lastRunFusions !== undefined, true);
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
  name: "導出相で落ちた run は GPU コマンドを 1 件も積まない（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    // 2 ノード目（matmul）が上限超過で throw する。dispatch の発行は**レシピ実行相**にしか
    // 無く、そこへ入る前に導出相が落ちるので、1 ノード目（relu）のぶんも積まれない
    // （src/runtime/recipe.ts の 2 相分離 — 失敗が確定した run は GPU の仕事をゼロにする）。
    // MUST: 上限超過の閾値は matmul の**出力タイル辺**で決まる（`ceil(n / tileN) > 上限`）。
    // タイル辺は定数ではなく**既定幾何から導く**（辺を変えた瞬間に throw が起きず
    // assertRejects だけが静かに落ちるドリフトが 16 → 64 → 128 と 2 度起きた形）。
    const tileN = gemmTileN(defaultGemmGeometry());
    const n = gpu.limits.maxComputeWorkgroupsPerDimension * tileN + tileN;
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
      assertEquals(stats.dispatchCount, 0, "先行ノードのぶんも積まれていない");
      // 残骸を後始末（arena.destroy → flush）で出すと、ロック解放後・errorScope ゼロ本の
      // submit になり、その validation エラーが並行する別 Session の区間に帰属する。
      assertEquals(stats.submitCount, 0, "失敗した run の残骸は 1 度も submit されない");
      assertEquals(stats.discardedDispatches, 0, "積んでいないので捨てるものも無い");
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
