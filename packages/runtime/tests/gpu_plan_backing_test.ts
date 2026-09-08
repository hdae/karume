// transient slot の GPU backing（導出済み計画にヒットした run が使う Session 常駐バッファ群）の門。
//
// backed run はアリーナの確保・参照計数を通らず、**前 run の残骸が載ったバッファ**へそのまま
// 書き込む。正しさの根拠は full-write（ADR 0014）だけなので、検出器は「値を変えながら同じ
// bindings で回した出力が、非 backed 実行とビット単位で一致するか」— stale slot が 1 本でも
// 混じれば前 run の値が残り、例外は 1 つも出ない。
//
// 併せて ①単発 run は slot メモリを一切払わない ②**バイト予算つきの LRU 集合**として持つ
// （ADR 0095 決定 1 — 予算内の signature は切替で作り直さず保持・超過分は古い順に退役・
// 新規 1 本だけで超える形は全退役してその 1 本だけ・予算 0 は従来の容量 1。予算が勘定するのは
// **領域 + backing が所有する入力バッファ**＝診断の `residentBytes + inputBytes`）③計画の LRU
// 追い出しで返る ④slot の総バイト数が現行 run のプール確保と一致する（footprint 不変 —
// **backing 1 本ぶん**の命題）を固定する。④が崩れると VRAM の前提（常駐化しても新しい
// ピークは生まれない）が崩れる。

import { assert, assertEquals, assertRejects } from "@std/assert";
import { openModel } from "../src/format/container.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import { BUFFER_USAGE } from "../src/gpu/webgpu-constants.ts";
import {
  createSession,
  PREPARED_PLAN_CAPACITY,
  type Session,
  type Tensor,
} from "../src/runtime/executor.ts";
import type { PlanBackingStats } from "../src/runtime/session-types.ts";
import { f32Bytes, type GraphJson } from "./helpers/format.ts";
import { fill, graphModelBuffer } from "./helpers/graph.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

/** y = silu(x·w + b)（x: [T,3] → y: [T,2]）。末尾 2 ノードが silu 融合に掴まれる。 */
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

/** `phase` ごとに値が変わる入力（同じ値を配ると stale slot が検出できない）。 */
const input = (rows: number, phase = 0): Tensor => ({
  dtype: "f32",
  shape: [rows, 3],
  data: Float32Array.from({ length: rows * 3 }, (_, i) => ((i + phase * 3) % 9 - 4) * 0.5),
});

/** 出力のビット列。値の一致は「バイト同値」で見る（丸めの取り違えを許容しない）。 */
const bits = (tensor: Tensor): readonly number[] =>
  Array.from(new Uint32Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.length));

/** Session を作り直して 1 run だけ回す = 必ず非 backed（backing はヒット run でしか作らない）。 */
const runFresh = async (
  gpu: Awaited<ReturnType<typeof acquireGpu>>,
  inputs: Parameters<Session["run"]>[0],
  model: ArrayBuffer = modelBytes(),
): Promise<Tensor> => {
  const session = await createSession(gpu, openModel(model));
  try {
    const outputs = await session.run(inputs);
    assertEquals(
      session.diagnostics().planBacking,
      { residentBytes: 0, inputBytes: 0, retainedCount: 0, buildCount: 0 },
      "単発 run は slot メモリを一切払わない",
    );
    return outputs["y"];
  } finally {
    await session.dispose();
  }
};

Deno.test({
  name: "backed run の出力は非 backed 実行とビット単位で一致する（stale slot の門・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await createSession(gpu, openModel(modelBytes()));
    try {
      // 1 run 目 = ミス（アリーナ経路）/ 2 run 目 = backing 構築 / 3 run 目 = backed 高速路。
      const phases = [0, 1, 2];
      const actual: Tensor[] = [];
      for (const phase of phases) actual.push((await session.run({ x: input(4, phase) }))["y"]);
      assertEquals(session.diagnostics().lastRunPrepared, { hit: true, cachedPlans: 1 });
      assertEquals(session.diagnostics().planBacking.buildCount, 1, "構築は 1 度だけ");

      // 恒真化の門: 入力を変えても出力が同じなら、stale slot が残っていても緑になる。
      assert(
        JSON.stringify(bits(actual[0])) !== JSON.stringify(bits(actual[1])),
        "phase ごとに出力が変わっていない（検出器として空振る）",
      );

      for (const phase of phases) {
        const expected = await runFresh(gpu, { x: input(4, phase) });
        assertEquals(actual[phase].shape, expected.shape);
        assertEquals(bits(actual[phase]), bits(expected), `phase ${phase} のビット一致`);
      }
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "backed run のアリーナ確保は readback staging だけになる（入力固定の門・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await createSession(gpu, openModel(modelBytes()));
    try {
      await session.run({ x: input(4, 0) });
      const miss = session.diagnostics().lastRun;
      await session.run({ x: input(4, 1) });
      const built = session.diagnostics().lastRun;
      await session.run({ x: input(4, 2) });
      const backed = session.diagnostics().lastRun;
      assert(miss !== undefined && built !== undefined && backed !== undefined);

      // グラフ出力は 1 本なので staging も 1 本。入力アップロードが 1 本でもアリーナに
      // 残っていればここが 2 以上になる（中間は既に slot 常駐へ移っている）。
      assertEquals(backed.allocCount, 1, "backed run の確保は readback staging の 1 本だけ");
      assertEquals(built.allocCount, 1, "初ヒット（構築）run も同じ");
      assertEquals(backed.transientBytes, 0, "中間はアリーナを通らない");
      assert(
        miss.allocCount > backed.allocCount,
        `ミス run の確保が減っていない（${miss.allocCount} → ${backed.allocCount}）`,
      );
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "同一 signature の並行 run は共有入力バッファでも取り違えない（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await createSession(gpu, openModel(modelBytes()));
    try {
      // backed run の入力は backing 所有の**1 本**へ上書きされる。await せず並行発行して、
      // 直列化（#chain）と「run は flush 完了後にしか返らない」が崩れたときに落ちる形にする
      // — 崩れれば後続 run の writeBuffer が先行 run の未 submit dispatch を追い越し、
      // 例外なしで前後の phase の値が混ざる。
      const phases = [0, 1, 2, 3, 4];
      const actual = await Promise.all(phases.map((phase) => session.run({ x: input(4, phase) })));
      assertEquals(session.diagnostics().planBacking.buildCount, 1);
      for (const phase of phases) {
        const expected = await runFresh(gpu, { x: input(4, phase) });
        assertEquals(bits(actual[phase]["y"]), bits(expected), `phase ${phase} のビット一致`);
      }
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

/** グラフ出力が入力の別名になる形（reshape）+ 実 dispatch を 1 本持つ形。 */
const ALIAS_GRAPH: GraphJson = {
  format: "karume-ir",
  version: 1,
  requires: { ops: ["reshape", "sigmoid"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [4, 3] }],
  outputs: ["y", "s"],
  initializers: {},
  values: {
    y: { dtype: "f32", shape: [12] },
    s: { dtype: "f32", shape: [4, 3] },
  },
  nodes: [
    { op: "reshape", ins: ["x"], outs: ["y"], attrs: {} },
    { op: "sigmoid", ins: ["x"], outs: ["s"], attrs: {} },
  ],
};

Deno.test({
  name: "グラフ出力が入力の別名でも backed run は読み戻せる（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const model = graphModelBuffer(ALIAS_GRAPH);
    const session = await createSession(gpu, openModel(model));
    try {
      for (const phase of [0, 1, 2]) {
        const x = input(4, phase);
        const outputs = await session.run({ x });
        // 別名出力は入力バッファそのものの読み戻し。値が phase ごとに変わることが、
        // 「常駐入力バッファに毎 run 書けている」ことの検出器になる。
        assertEquals(outputs["y"].shape, [12]);
        assertEquals(bits(outputs["y"]), bits(x), `phase ${phase} の別名出力`);

        const reference = await createSession(gpu, openModel(model));
        try {
          assertEquals(bits(outputs["s"]), bits((await reference.run({ x }))["s"]));
        } finally {
          await reference.dispose();
        }
      }
      assertEquals(session.diagnostics().lastRunPrepared?.hit, true);
      assertEquals(session.diagnostics().planBacking.buildCount, 1);
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

/** backing 1 本ぶんの実測（領域 = `residentBytes` / 所有する入力バッファ = `inputBytes`）。 */
type Measured = { readonly residentBytes: number; readonly inputBytes: number };

/**
 * 予算が勘定する量（`#evictBackingsFor` と同じ算式 — 領域と所有入力バッファの両方）。
 *
 * MUST: 予算の門はこれで組む。領域だけで組むと、入力バッファのぶんだけ実際の勘定が予算を
 * 上回り、「予算ちょうど」のつもりの Session が最初から超過側（= 常に全退役）で回る。
 */
const accountedBytes = (measured: readonly Measured[]): number =>
  measured.reduce((total, one) => total + one.residentBytes + one.inputBytes, 0);

/** 保持集合が `retained` のときの診断（**全欄** — 部分一致に逃げない）。 */
const expectStats = (retained: readonly Measured[], buildCount: number): PlanBackingStats => ({
  residentBytes: retained.reduce((total, one) => total + one.residentBytes, 0),
  inputBytes: retained.reduce((total, one) => total + one.inputBytes, 0),
  retainedCount: retained.length,
  buildCount,
});

/**
 * signature（`T = rows`）1 本ぶんの backing が常駐させるバイト数を、**その形だけを回した
 * 別 Session** で測る。
 *
 * 予算の門（下の 3 本）はこの実測値から組む。定数で書くと、slot 表の詰め方（ADR 0093）が
 * 変わったときに「予算ちょうど」の意味が黙ってずれて、LRU の順を見ているつもりの門が
 * 「全部入る」形か「1 本も入らない」形に化ける。
 */
const backingBytes = async (
  gpu: Awaited<ReturnType<typeof acquireGpu>>,
  rows: number,
): Promise<Measured> => {
  const session = await createSession(gpu, openModel(modelBytes()));
  try {
    // 1 run 目 = ミス（計画の導出）/ 2 run 目 = ヒット（backing の構築）。
    await session.run({ x: input(rows) });
    await session.run({ x: input(rows) });
    const stats = session.diagnostics().planBacking;
    assertEquals(stats.retainedCount, 1, `T=${rows} を 1 形だけ回した Session の保持が 1 本でない`);
    assert(stats.residentBytes > 0, `T=${rows} の backing が 0 バイト（門が空振る）`);
    // 通常入力（常駐でない `Tensor`）のバッファは backing 所有 = 予算の勘定に入る。0 なら
    // 「入力を勘定していない」形なので、下の予算の門は領域だけで組まれたことになる。
    assert(stats.inputBytes > 0, `T=${rows} の所有入力バッファが 0 バイト（予算が勘定していない）`);
    return { residentBytes: stats.residentBytes, inputBytes: stats.inputBytes };
  } finally {
    await session.dispose();
  }
};

Deno.test({
  name: "予算内の 2 signature は交互に回しても保持されたまま作り直されない（既定予算・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      const narrow = await backingBytes(gpu, 4);
      const wide = await backingBytes(gpu, 9);
      assert(
        accountedBytes([wide]) > accountedBytes([narrow]),
        `T=9 の backing が T=4 より大きくない（${JSON.stringify(narrow)} / ${
          JSON.stringify(wide)
        }）`,
      );

      // 既定予算（256 MiB）— この 2 形はどちらも数十バイトなので両方が収まる。
      const session = await createSession(gpu, openModel(modelBytes()));
      try {
        await session.run({ x: input(4) });
        assertEquals(
          session.diagnostics().planBacking,
          expectStats([], 0),
          "ミス run は backing を作らない",
        );

        await session.run({ x: input(4) });
        assertEquals(
          session.diagnostics().planBacking,
          expectStats([narrow], 1),
          "初ヒットで slot が 1 本常駐する",
        );

        // 別 signature はミス run では作らず、ヒットして初めて**足される**（置き換えではない）。
        await session.run({ x: input(9) });
        assertEquals(
          session.diagnostics().planBacking,
          expectStats([narrow], 1),
          "別 signature のミス run では不変",
        );
        await session.run({ x: input(9) });
        assertEquals(
          session.diagnostics().planBacking,
          expectStats([narrow, wide], 2),
          "2 本目が旧 backing を退役させている（総和が単独実測の和にならない）",
        );

        // 交互に 3 往復しても作り直さない = 切替 1 回ぶんの再構築（perf-ledger H-15 の ≈ 40 ms）が
        // 消えたことの観測点。値も非 backed 実行とビット一致する（保持した束が別の形の実体を
        // 掴んでいれば、例外なしでここが割れる）。
        for (const phase of [1, 2, 3]) {
          const narrowOut = (await session.run({ x: input(4, phase) }))["y"];
          const wideOut = (await session.run({ x: input(9, phase) }))["y"];
          assertEquals(
            session.diagnostics().planBacking,
            expectStats([narrow, wide], 2),
            `phase ${phase}: 切替で backing を作り直している`,
          );
          assertEquals(bits(narrowOut), bits(await runFresh(gpu, { x: input(4, phase) })));
          assertEquals(bits(wideOut), bits(await runFresh(gpu, { x: input(9, phase) })));
        }
      } finally {
        await session.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "予算 0 の Session は切替のたびに作り直す（従来の容量 1 の再現・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      const narrow = await backingBytes(gpu, 4);
      const wide = await backingBytes(gpu, 9);

      const session = await createSession(gpu, openModel(modelBytes()), {
        planBackingBudgetBytes: 0,
      });
      try {
        await session.run({ x: input(4) });
        await session.run({ x: input(4) });
        assertEquals(session.diagnostics().planBacking, expectStats([narrow], 1));

        // 予算 0 では新規 1 本が必ず超過側になるので、保持中は全て退役する。
        await session.run({ x: input(9) });
        await session.run({ x: input(9) });
        assertEquals(
          session.diagnostics().planBacking,
          expectStats([wide], 2),
          "予算 0 で旧 backing が残っている（容量 1 の挙動が再現できていない）",
        );

        // 計画は LRU に残っているので、戻った run はヒットして**作り直し**になる。
        const narrowOut = (await session.run({ x: input(4, 1) }))["y"];
        assertEquals(session.diagnostics().lastRunPrepared?.hit, true);
        assertEquals(
          session.diagnostics().planBacking,
          expectStats([narrow], 3),
          "切替で作り直していない（予算 0 が効いていない）",
        );
        assertEquals(bits(narrowOut), bits(await runFresh(gpu, { x: input(4, 1) })));
      } finally {
        await session.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "新規 1 本だけで予算を超える形は保持中を全て退役させて 1 本だけ持つ（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      const narrow = await backingBytes(gpu, 4);
      const wide = await backingBytes(gpu, 9);
      // どの 1 本も単体で超える予算（= 常駐は `max(予算, 最大 1 本)`（ADR 0095 決定 1）の
      // 「最大 1 本」側だけになる形）。勘定は領域 + 所有入力なので、引くのもその和から。
      const budget = accountedBytes([narrow]) - 1;
      assert(budget >= 0, `narrow の実測 ${JSON.stringify(narrow)} が小さすぎて予算を作れない`);

      const session = await createSession(gpu, openModel(modelBytes()), {
        planBackingBudgetBytes: budget,
      });
      try {
        await session.run({ x: input(4) });
        await session.run({ x: input(4) });
        assertEquals(
          session.diagnostics().planBacking,
          expectStats([narrow], 1),
          "予算を超える 1 本目が確保されていない（backed 経路が消えている）",
        );

        await session.run({ x: input(9) });
        await session.run({ x: input(9) });
        assertEquals(
          session.diagnostics().planBacking,
          expectStats([wide], 2),
          "予算超過の新規が旧 backing を残している（常駐が max(予算, 最大 1 本) を超える）",
        );
      } finally {
        await session.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "backed 計画が LRU から追い出されると backing も返り、後続 run は完走する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await createSession(gpu, openModel(modelBytes()));
    try {
      await session.run({ x: input(4) });
      await session.run({ x: input(4) });
      assert(session.diagnostics().planBacking.residentBytes > 0);

      // T=4 は最古。別 shape を**上限ちょうど**通すと、追い出されるのは T=4 の 1 本だけになる
      // （下で T=5 のヒットを見るので、余分に追い出すと門が別の理由で赤くなる）。本数は定数から
      // 出す — 写しを置くと上限を動かしたときに追い出しが起きず、この門が空振りする。
      for (let rows = 5; rows < 5 + PREPARED_PLAN_CAPACITY; rows += 1) {
        await session.run({ x: input(rows) });
      }
      assertEquals(
        session.diagnostics().planBacking,
        expectStats([], 1),
        "追い出された計画の backing は返る（作り直しは起きていない）",
      );

      // 追い出し + 破棄の後も、残った計画のヒット run は正しく完走する。
      const hit = (await session.run({ x: input(5) }))["y"];
      assertEquals(session.diagnostics().lastRunPrepared?.hit, true);
      assertEquals(session.diagnostics().planBacking.buildCount, 2);
      assertEquals(bits(hit), bits(await runFresh(gpu, { x: input(5) })));
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

/**
 * attention（ノード内一時 2 本）+ reshape（別名・グラフ出力）。slot 導出の 3 経路
 * （出力確保 / 一時の入れ子寿命 / 別名越しのピン留め）を 1 本のグラフで踏む。
 */
const HALF_SCALE = Math.fround(Math.sqrt(1 / Math.sqrt(4)));

const TEMP_GRAPH: GraphJson = {
  format: "karume-ir",
  version: 1,
  requires: { ops: ["attention", "reshape"] },
  symbols: ["M"],
  inputs: [
    { name: "q", dtype: "f32", shape: [1, 1, "M", 4] },
    { name: "k", dtype: "f32", shape: [1, 1, 4, 4] },
    { name: "v", dtype: "f32", shape: [1, 1, 4, 4] },
  ],
  outputs: ["y"],
  initializers: {},
  values: {
    a: { dtype: "f32", shape: [1, 1, "M", 4] },
    y: { dtype: "f32", shape: ["M", 4] },
  },
  nodes: [
    { op: "attention", ins: ["q", "k", "v"], outs: ["a"], attrs: { scale: HALF_SCALE } },
    { op: "reshape", ins: ["a"], outs: ["y"], attrs: {} },
  ],
};

const ROWS = 6;
const tempInputs = (phase: number) => ({
  q: fill([1, 1, ROWS, 4], (i) => ((i + phase) % 7 - 3) * 0.25),
  k: fill([1, 1, 4, 4], (i) => ((i % 5) - 2) * 0.5),
  v: fill([1, 1, 4, 4], (i) => ((i % 3) - 1) * 0.75),
});

/** アリーナが 1 本のバッファに配る大きさ（最小 4 バイト + 4 バイト整列）。 */
const bufferBytes = (count: number): number => Math.max(4, count * 4);

Deno.test({
  name: "slot の総バイト数は非 backed run のプール確保と一致する（footprint 不変の門・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    // グラフ入力 3 本のアップロード（q: [1,1,ROWS,4] / k, v: [1,1,4,4]）。backed run では同じ
    // 大きさのバッファを backing が所有する = 診断の `inputBytes` の期待値でもある。
    const inputHostBytes = bufferBytes(ROWS * 4) + bufferBytes(16) * 2;
    // プール外（入力アップロードと readback staging）のぶん。アリーナの allocatedBytes から
    // これを引いた残りが「dispatch が書く出力ストレージの実確保」= slot 表の総バイト数。
    const hostBytes = inputHostBytes + bufferBytes(ROWS * 4);
    const reference = await createSession(gpu, openModel(graphModelBuffer(TEMP_GRAPH)));
    let pooledBytes = 0;
    try {
      await reference.run(tempInputs(0));
      const stats = reference.diagnostics().lastRun;
      assert(stats !== undefined, "1 run 目のアリーナ実績が無い");
      pooledBytes = stats.allocatedBytes - hostBytes;
      assert(pooledBytes > 0, `プール確保が 0（門が空振る）: ${stats.allocatedBytes}`);
    } finally {
      await reference.dispose();
    }

    const session = await createSession(gpu, openModel(graphModelBuffer(TEMP_GRAPH)));
    try {
      const first = (await session.run(tempInputs(1)))["y"];
      const second = (await session.run(tempInputs(1)))["y"];
      // MUST: 保持 1 本の状態で測る（診断の `residentBytes` は保持集合の総和なので、2 形を
      // 回した Session で測ると「1 run のプール確保と一致する」という命題そのものが変わる）。
      assertEquals(
        session.diagnostics().planBacking.retainedCount,
        1,
        "footprint 不変は backing 1 本ぶんの命題（保持が 1 本でない状態で測っている）",
      );
      assertEquals(
        session.diagnostics().planBacking.residentBytes,
        pooledBytes,
        "slot 表が現行のプール確保と食い違う（VRAM の前提が崩れる）",
      );
      // 入力側も同じ命題 — backing が所有する入力バッファは、アリーナ経路が run ごとに確保して
      // いたアップロード先そのもの（予算が勘定するのはこの和なので、ここがずれると予算の意味が
      // 実際の VRAM から外れる）。
      assertEquals(
        session.diagnostics().planBacking.inputBytes,
        inputHostBytes,
        "backing の所有入力がアリーナ経路のアップロード先と食い違う",
      );
      // 一時と別名を踏む形でも backed 出力は非 backed と一致する（別名越しのピン留めも門）。
      assertEquals(bits(second), bits(first));
      assertEquals(second.shape, [ROWS, 4]);
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

/** storage バッファ 1 本ぶんの記録（確保の順に 1 要素 = その実体の destroy 回数）。 */
type BufferRecord = { readonly destroyCounts: number[] };

/**
 * `#activateBacking` の確保途中失敗を撃つ注入。
 *
 * 実 GPU では「N 本目の createBuffer だけ落とす」形も「確保済みがちょうど 1 回返ったか」も
 * 作れないので、device の `createBuffer` を包む（生産コードに注入面を開けないための代替 —
 * `gpu_generation_context_test.ts` の `injectFaults` と同型）。
 * MUST: 返す実体は**本物の GPUBuffer のまま**にする（`destroy` だけを包む）。差し替えると
 * bind group 生成に渡す実体がテスト側の作り物になり、門が実装を検査しなくなる。
 */
const injectStorageFault = (device: GPUDevice): {
  /** 以後の storage 確保を新しい記録へ載せる（`failAt` 本目で同期 throw）。 */
  readonly record: (failAt?: number) => BufferRecord;
  readonly restore: () => void;
} => {
  const original = device.createBuffer.bind(device);
  let current: BufferRecord = { destroyCounts: [] };
  let creations = 0;
  let failAt: number | undefined;
  device.createBuffer = ((descriptor: GPUBufferDescriptor): GPUBuffer => {
    // 数えるのは slot / 入力の storage だけ（readback staging は MAP_READ で別経路）。
    if ((descriptor.usage & BUFFER_USAGE.STORAGE) === 0) return original(descriptor);
    creations += 1;
    if (creations === failAt) {
      throw new Error(`注入: storage ${creations} 本目の createBuffer が同期 throw`);
    }
    const buffer = original(descriptor);
    const counts = current.destroyCounts;
    const index = counts.push(0) - 1;
    const destroy = buffer.destroy.bind(buffer);
    buffer.destroy = (): undefined => {
      counts[index] += 1;
      return destroy();
    };
    return buffer;
  }) as typeof device.createBuffer;
  return {
    record: (next?: number): BufferRecord => {
      creations = 0;
      failAt = next;
      current = { destroyCounts: [] };
      return current;
    },
    restore: (): void => {
      device.createBuffer = original;
    },
  };
};

// executor.ts の `#activateBacking` は「確保 → 保持集合（`#backings`）への登録（所有権の確立）」
// までを try/catch で囲む。この窓で漏れた実体は `#retireBacking()` からも `dispose()` からも
// 到達できず、しかも量はこの Session で最大（slot 表の総バイト）になる。同型の門は
// `GenerationContext.create` 側にあるが executor 側には無かった。
//
// 予算で分かれるのは**旧 backing の扱い**だけ（ADR 0095 決定 2 — 失敗経路が退役させるのは
// 「この run が新規構築した 1 本」）: 予算内なら旧 backing は保持されたまま、予算 0 なら確保の
// 前の退役が先に効いているので旧 backing も返る。確保途中で漏れた実体がちょうど 1 回返ること
// は、どちらの予算でも同じ。
const assertPartialBackingFault = async (
  planBackingBudgetBytes: number | undefined,
  expected: {
    /** 失敗 run の後に旧 backing（T=4）が保持されたままか。 */
    readonly retainsOld: boolean;
  },
): Promise<void> => {
  const gpu = await acquireGpu();
  const session = await createSession(
    gpu,
    openModel(modelBytes()),
    planBackingBudgetBytes === undefined ? {} : { planBackingBudgetBytes },
  );
  // 常駐入力（焼き込み参照が retain / release される側）。backing は入力バッファを所有
  // しないので、storage の記録は slot 表そのものになる。
  const narrow = await gpu.createResident(4 * 3 * 4, "narrow");
  const wide = await gpu.createResident(9 * 3 * 4, "wide");
  const fault = injectStorageFault(gpu.device);
  try {
    narrow.write(input(4, 0).data);
    wide.write(input(9, 0).data);

    // ミス run（アリーナ経路）の確保は別の記録へ逃がす。以後 backing の構築だけを数える。
    // MUST: 記号 T は明示で束縛する（常駐入力は shape を持たないので束縛源にならない）。
    fault.record();
    await session.run({ x: narrow }, { T: 4 });

    // T=4 のヒット run = backing の構築。この窓の storage 確保は slot 表そのもの
    // （常駐入力は backing 所有ではなく、readback staging は MAP_READ で別経路）。
    const built = fault.record();
    await session.run({ x: narrow }, { T: 4 });
    assertEquals(session.diagnostics().planBacking.buildCount, 1);
    const old: Measured = {
      residentBytes: session.diagnostics().planBacking.residentBytes,
      inputBytes: session.diagnostics().planBacking.inputBytes,
    };
    // 常駐入力は GpuContext 所有 = backing は所有しない。ここが 0 でなくなると、下の
    // 「storage の記録は slot 表そのもの」という前提（= 本数の数え方）が崩れる。
    assertEquals(old.inputBytes, 0, "常駐入力を backing 所有として勘定している");
    assertEquals(narrow.bakedReferences, 1, "焼き込み参照が立っていない（門が空振りする）");
    const slots = built.destroyCounts.length;
    assert(slots >= 2, `slot が ${slots} 本しかなく「途中で落とす」形にならない`);

    // T=9 はミス run では backing を作らない。次のヒット run が構築の窓。
    fault.record();
    await session.run({ x: wide }, { T: 9 });

    const partial = fault.record(2);
    await assertRejects(
      () => session.run({ x: wide }, { T: 9 }),
      Error,
      "注入: storage 2 本目",
    );

    assertEquals(partial.destroyCounts, [1], "確保済みの 1 本がちょうど 1 回だけ返る");
    assertEquals(
      built.destroyCounts,
      Array(slots).fill(expected.retainsOld ? 0 : 1),
      expected.retainsOld
        ? "無関係な失敗で旧 backing が退役している（再構築のスラッシングの入り口）"
        : "退役した旧 backing もちょうど 1 回ずつ返る",
    );
    assertEquals(
      session.diagnostics().planBacking,
      expected.retainsOld ? expectStats([old], 1) : expectStats([], 1),
      "失敗した構築が backing として据わっている",
    );
    assertEquals(
      narrow.bakedReferences,
      expected.retainsOld ? 1 : 0,
      expected.retainsOld
        ? "保持されたままの backing の焼き込み参照が返っている"
        : "退役で焼き込み参照が返っていない",
    );
    assertEquals(wide.bakedReferences, 0, "失敗した構築が焼き込み参照を残している");

    // 同一 signature の次の run は作り直して完走し、非 backed 実行とビット一致する。
    fault.restore();
    const actual = (await session.run({ x: wide }, { T: 9 }))["y"];
    assertEquals(session.diagnostics().planBacking.buildCount, 2, "作り直していない");
    assertEquals(bits(actual), bits(await runFresh(gpu, { x: input(9, 0) })));
  } finally {
    fault.restore();
    await session.dispose();
    narrow.dispose();
    wide.dispose();
    gpu.destroy();
  }
};

Deno.test({
  name:
    "backing 確保の途中失敗は確保済みだけを 1 回返し、旧 backing は保持したまま（既定予算・故障注入・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: () => assertPartialBackingFault(undefined, { retainsOld: true }),
});

Deno.test({
  name:
    "予算 0 では確保の前に旧 backing が退役し、確保済みと合わせてちょうど 1 回ずつ返る（故障注入・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: () => assertPartialBackingFault(0, { retainsOld: false }),
});

/**
 * 予算ちょうどの Session に 3 つ目の signature を入れて、退役するのが**最も古く使われた方**
 * （直前に触った方ではない）ことを見る。
 *
 * `touched` は 3 つ目の前に触り直す形（= LRU の最新へ動く形）。2 通りとも置くのは、片方だけだと
 * 「常に先に構築した方を退役させる」実装（挿入順のみ・ヒットで動かさない）が緑のまま通るため。
 */
const assertLruOrder = async (touched: 4 | 9): Promise<void> => {
  const gpu = await acquireGpu();
  try {
    const narrow = await backingBytes(gpu, 4);
    const wide = await backingBytes(gpu, 9);
    const small = await backingBytes(gpu, 2);
    // 3 つ目は「1 本退役させれば必ず収まる」大きさでなければ、退役が 2 本に及んで LRU の順が
    // 観測できない（門が空振る）。比べるのは予算が勘定する量（領域 + 所有入力）。
    assert(
      accountedBytes([small]) <= accountedBytes([narrow]) &&
        accountedBytes([small]) <= accountedBytes([wide]),
      `T=2 の実測 ${JSON.stringify(small)} が他の形以下でない` +
        `（${JSON.stringify(narrow)} / ${JSON.stringify(wide)}）`,
    );

    const session = await createSession(gpu, openModel(modelBytes()), {
      planBackingBudgetBytes: accountedBytes([narrow, wide]),
    });
    const fault = injectStorageFault(gpu.device);
    try {
      // ミス run のぶんは捨てる記録へ逃がし、構築の窓だけを形ごとに数える。
      fault.record();
      await session.run({ x: input(4) });
      const narrowRecord = fault.record();
      await session.run({ x: input(4) });
      fault.record();
      await session.run({ x: input(9) });
      const wideRecord = fault.record();
      await session.run({ x: input(9) });
      assertEquals(
        session.diagnostics().planBacking,
        expectStats([narrow, wide], 2),
        "予算ちょうどの 2 本が両方とも保持されていない",
      );

      // 触り直し（LRU の最新へ動かす）。ヒットなので作り直しは起きない。
      await session.run({ x: input(touched) });
      assertEquals(
        session.diagnostics().planBacking.buildCount,
        2,
        "触り直しで backing を作り直している",
      );

      // 3 つ目。予算ちょうどなので、最も古く使われた 1 本だけが退役する。
      fault.record();
      await session.run({ x: input(2) });
      const smallRecord = fault.record();
      await session.run({ x: input(2) });

      const kept = touched === 4
        ? { record: narrowRecord, measured: narrow, rows: 4 }
        : { record: wideRecord, measured: wide, rows: 9 };
      const evicted = touched === 4
        ? { record: wideRecord, rows: 9 }
        : { record: narrowRecord, rows: 4 };
      assertEquals(
        session.diagnostics().planBacking,
        expectStats([kept.measured, small], 3),
        `直前に触った T=${touched} ではない方が退役している（LRU の順が挿入順のまま）`,
      );
      assert(evicted.record.destroyCounts.length > 0, "退役側の記録が空（門が空振る）");
      assertEquals(
        evicted.record.destroyCounts,
        Array(evicted.record.destroyCounts.length).fill(1),
        `退役した T=${evicted.rows} のバッファがちょうど 1 回ずつ返っていない`,
      );
      assertEquals(
        kept.record.destroyCounts,
        Array(kept.record.destroyCounts.length).fill(0),
        `保持中の T=${kept.rows} のバッファが返っている（使用中の実体を破棄している）`,
      );
      assertEquals(
        smallRecord.destroyCounts,
        Array(smallRecord.destroyCounts.length).fill(0),
        "新規に確保した backing がその場で返っている",
      );

      // 保持している形は作り直さずに回り、値は非 backed 実行と一致する。
      const hit = (await session.run({ x: input(kept.rows, 1) }))["y"];
      assertEquals(session.diagnostics().planBacking.buildCount, 3, "保持中の形を作り直した");
      assertEquals(bits(hit), bits(await runFresh(gpu, { x: input(kept.rows, 1) })));
    } finally {
      fault.restore();
      await session.dispose();
    }
  } finally {
    gpu.destroy();
  }
};

Deno.test({
  name: "予算超過の退役は最も古く使われた 1 本（直前に触った T=4 が残る・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: () => assertLruOrder(4),
});

Deno.test({
  name: "予算超過の退役は最も古く使われた 1 本（直前に触った T=9 が残る・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: () => assertLruOrder(9),
});
