// shard 進行検証（ADR 0070 決定 1）— shard を 1 本ずつ受けて即座に決まる違反を落とし、
// 全 shard 読了後に完全性（欠け）を見る。単一ファイル面（openModel）と同じ検査経路に載って
// いることが前提なので、ここは「shard に割れたときにだけ現れる形」を主に固定する。

import { assertEquals, assertThrows } from "@std/assert";
import {
  ContainerError,
  createShardValidator,
  extractIrGraph,
  IR_METADATA_KEY,
  parsePieceKey,
} from "../src/format/container.ts";
import { parseSafetensors, type SafetensorsFile } from "../src/format/safetensors.ts";
import {
  baseGraph,
  buildSafetensors,
  f32Bytes,
  type GraphJson,
  type TensorSpec,
} from "./helpers/format.ts";

/** 重み shard（metadata を持たない — グラフ shard 以外は karume_ir を載せない）。 */
const weightShard = (tensors: readonly TensorSpec[]): SafetensorsFile =>
  parseSafetensors(buildSafetensors(tensors));

/** グラフ shard（karume_ir + 小テンソル）。 */
const graphShard = (graph: GraphJson, tensors: readonly TensorSpec[] = []): SafetensorsFile =>
  parseSafetensors(buildSafetensors(tensors, { karume_ir: JSON.stringify(graph) }));

const W_F32: TensorSpec = {
  name: "enc.w",
  dtype: "F32",
  shape: [4, 3],
  data: f32Bytes(new Array(12).fill(0.5)),
};
const B_F32: TensorSpec = { name: "enc.b", dtype: "F32", shape: [3], data: f32Bytes([1, 2, 3]) };

/** 格納 i8 + per-channel（keepdim broadcast 形）scale の最小グラフ。 */
const i8Graph = (): GraphJson => {
  const graph = baseGraph();
  graph.initializers["w"].storage = { dtype: "i8", scale: "enc.w.scale" };
  return graph;
};
const W_I8: TensorSpec = { name: "enc.w", dtype: "I8", shape: [4, 3], data: new Uint8Array(12) };
const W_I8_SCALE: TensorSpec = {
  name: "enc.w.scale",
  dtype: "F32",
  shape: [4, 1],
  data: f32Bytes([1, 1, 1, 1]),
};

/** 格納 i4 + group 形 scale（重み `[4,64]` / group_size 32 → scale `[4,2]`）。 */
const i4Graph = (): GraphJson => {
  const graph = baseGraph();
  graph.values["w"] = { dtype: "f32", shape: [4, 64] };
  graph.initializers["w"].storage = { dtype: "i4", scale: "enc.w.scale", group_size: 32 };
  return graph;
};
const W_I4: TensorSpec = { name: "enc.w", dtype: "I4", shape: [4, 64], data: new Uint8Array(128) };
const W_I4_SCALE: TensorSpec = {
  name: "enc.w.scale",
  dtype: "F32",
  shape: [4, 2],
  data: f32Bytes(new Array(8).fill(1)),
};

// テンソル分割（piece）の素材。単独で 1 shard に収まらない大テンソルを
// 先頭次元（行）の連続範囲へ割り、連続する shard へ 1 本ずつ配る形を最小サイズで写す。

/** `enc.w`（f32 [4,3]）を 2 行ずつ 2 本へ割った piece（1 行 = 12 バイト = 常に 4 の倍数）。 */
const pieceSpec = (
  index: number,
  count: number,
  rows: number,
  overrides: Partial<TensorSpec> = {},
): TensorSpec => ({
  name: `enc.w#${String(index).padStart(5, "0")}-of-${String(count).padStart(5, "0")}`,
  dtype: "F32",
  shape: [rows, 3],
  data: f32Bytes(new Array(rows * 3).fill(0.5)),
  ...overrides,
});
const W_PIECE_1 = pieceSpec(1, 2, 2);
const W_PIECE_2 = pieceSpec(2, 2, 2);

/**
 * piece + companion scale 用の最小 i8 グラフ（`w` は [4,4] — 1 行 = 4 バイトなので、どの行
 * 境界で割っても「末尾以外の piece は 4 の倍数」を満たす）。
 */
const i8PieceGraph = (): GraphJson => {
  const graph = baseGraph();
  graph.values["w"] = { dtype: "f32", shape: [4, 4] };
  graph.values["b"] = { dtype: "f32", shape: [4] };
  graph.values["h"] = { dtype: "f32", shape: ["T", 4] };
  graph.values["y"] = { dtype: "f32", shape: ["T", 4] };
  graph.initializers["w"].storage = { dtype: "i8", scale: "enc.w.scale" };
  return graph;
};
const B4_F32: TensorSpec = {
  name: "enc.b",
  dtype: "F32",
  shape: [4],
  data: f32Bytes([1, 2, 3, 4]),
};
const W_I8_WIDE_SCALE: TensorSpec = {
  name: "enc.w.scale",
  dtype: "F32",
  shape: [4, 1],
  data: f32Bytes([1, 1, 1, 1]),
};
const W_I8_PIECE_1: TensorSpec = {
  name: "enc.w#00001-of-00002",
  dtype: "I8",
  shape: [2, 4],
  data: new Uint8Array(8),
};
const W_I8_PIECE_2: TensorSpec = { ...W_I8_PIECE_1, name: "enc.w#00002-of-00002" };

Deno.test("extractIrGraph: グラフ shard から IR を取り出す", () => {
  const graph = extractIrGraph(graphShard(baseGraph(), [B_F32]));
  assertEquals(graph.nodes.length, 2);
  assertEquals(graph.initializers["w"].tensor, "enc.w");
});

// 重み shard に karume_ir は載らない（グラフ shard は先頭の 1 本だけ — ADR 0070 決定 3）。
Deno.test("extractIrGraph: karume_ir を持たない shard を拒否する", () => {
  assertThrows(() => extractIrGraph(weightShard([W_F32])), ContainerError, IR_METADATA_KEY);
});

Deno.test("createShardValidator: グラフ shard + 重み shard を逐次受理して読了できる", () => {
  const graph = extractIrGraph(graphShard(baseGraph(), [B_F32]));
  const validator = createShardValidator(graph);

  // グラフ shard に載った小テンソルもこの経路で実体が確定する（別扱いにしない）
  const fromGraphShard = validator.intake(graphShard(baseGraph(), [B_F32]));
  assertEquals(fromGraphShard.map((ready) => ready.name), ["b"]);
  assertEquals(fromGraphShard[0].view.shape, [3]);
  assertEquals(fromGraphShard[0].scale, undefined);

  const weights = weightShard([W_F32]);
  const fromWeightShard = validator.intake(weights);
  assertEquals(fromWeightShard.map((ready) => ready.name), ["w"]);
  assertEquals(fromWeightShard[0].view.shape, [4, 3]);
  // 逐次消費側は転送後にこの参照を手放すので、どの shard の実体かが要る
  assertEquals(fromWeightShard[0].file, weights);

  validator.finish();
});

// 転送順が shard のヘッダ並びに依存すると、同一資産でも配布形の詰め方でアリーナ配置が変わる。
Deno.test("createShardValidator: intake の戻りは shard のヘッダ順ではなく宣言順", () => {
  const graph = extractIrGraph(graphShard(baseGraph()));
  const validator = createShardValidator(graph);

  // ヘッダは b → w の順（宣言は w → b）
  const ready = validator.intake(weightShard([B_F32, W_F32]));
  assertEquals(ready.map((item) => item.name), ["w", "b"]);
  validator.finish();
});

Deno.test("createShardValidator: 量子化 initializer は payload と scale の両方を返す", () => {
  const graph = extractIrGraph(graphShard(i4Graph()));
  const validator = createShardValidator(graph);

  const ready = validator.intake(weightShard([W_I4, W_I4_SCALE, B_F32]));
  assertEquals(ready.map((item) => item.name), ["w", "b"]);
  assertEquals(ready[0].view.byteLength, 128);
  assertEquals(ready[0].scale?.shape, [4, 2]);
  assertEquals(ready[1].scale, undefined);
  validator.finish();
});

// 欠けは shard 単体では決まらない（後続 shard で来るかもしれない）ので、読了まで持ち越す。
Deno.test("createShardValidator: 読了時に欠けている payload と scale を全件列挙する", () => {
  const graph = extractIrGraph(graphShard(i8Graph()));
  const validator = createShardValidator(graph);
  validator.intake(graphShard(i8Graph(), [B_F32]));

  const error = assertThrows(() => validator.finish(), ContainerError, "不足するテンソル (2)");
  assertEquals(error.message.includes("テンソル 'enc.w' がファイルに無い"), true, error.message);
  assertEquals(
    error.message.includes("scale テンソル 'enc.w.scale' がファイルに無い"),
    true,
    error.message,
  );
});

// 欠けが 1 本も無ければ読了は通る（欠け検査が常に鳴る退行の裏取り）。
Deno.test("createShardValidator: 全 shard が揃えば読了は通る", () => {
  const graph = extractIrGraph(graphShard(i8Graph()));
  const validator = createShardValidator(graph);
  validator.intake(weightShard([W_I8, W_I8_SCALE]));
  validator.intake(weightShard([B_F32]));
  validator.finish();
});

Deno.test("createShardValidator: 突合集合に無いテンソルを持つ shard を intake で落とす", () => {
  const graph = extractIrGraph(graphShard(baseGraph()));
  const validator = createShardValidator(graph);

  const error = assertThrows(
    () => validator.intake(weightShard([W_F32, { ...W_F32, name: "enc.dead" }])),
    ContainerError,
    "参照されないテンソル (1)",
  );
  assertEquals(error.message.includes("enc.dead"), true, error.message);
});

// shard 横断でしか見えない違反。同名が 2 本の shard にあると「どちらが勝つか」が転送順で
// 決まる沈黙誤値になる。
Deno.test("createShardValidator: 別の shard で定義済みのテンソルの再登場を落とす", () => {
  const graph = extractIrGraph(graphShard(baseGraph()));
  const validator = createShardValidator(graph);
  validator.intake(weightShard([W_F32, B_F32]));

  const error = assertThrows(
    () => validator.intake(weightShard([W_F32])),
    ContainerError,
    "別の shard で既に定義されたテンソル (1)",
  );
  assertEquals(error.message.includes("enc.w"), true, error.message);
});

// co-shard MUST（ADR 0070 決定 1）: 逐次消費は weight と scale を同時に要するので、shard を
// 跨ぐと「転送したら参照を手放す」契約と両立しない。両向きとも intake で落ちる。
Deno.test("createShardValidator: payload だけを載せた shard を co-shard 違反で落とす", () => {
  const graph = extractIrGraph(graphShard(i8Graph()));
  const validator = createShardValidator(graph);

  const error = assertThrows(
    () => validator.intake(weightShard([W_I8])),
    ContainerError,
    "実体 'enc.w' と同じ shard に置く MUST",
  );
  assertEquals(error.message.includes("scale テンソル 'enc.w.scale'"), true, error.message);
});

Deno.test("createShardValidator: scale だけを載せた shard を co-shard 違反で落とす", () => {
  const graph = extractIrGraph(graphShard(i8Graph()));
  const validator = createShardValidator(graph);

  // scale 名は突合集合に入っているので余剰では拾えない — co-shard として帰属を言う
  const error = assertThrows(
    () => validator.intake(weightShard([W_I8_SCALE])),
    ContainerError,
    "だけが shard にあり実体 'enc.w' が無い",
  );
  assertEquals(error.message.includes("enc.w.scale"), true, error.message);
});

Deno.test("createShardValidator: 格納 dtype と実テンソル dtype の不一致を intake で落とす", () => {
  const graph = extractIrGraph(graphShard(baseGraph()));
  const validator = createShardValidator(graph);

  assertThrows(
    () =>
      validator.intake(
        weightShard([{ name: "enc.w", dtype: "BF16", shape: [4, 3], data: new Uint8Array(24) }]),
      ),
    ContainerError,
    "F32 が必要",
  );
});

Deno.test("createShardValidator: 宣言 shape と実テンソル shape の不一致を intake で落とす", () => {
  const graph = extractIrGraph(graphShard(baseGraph()));
  const validator = createShardValidator(graph);

  assertThrows(
    () => validator.intake(weightShard([{ ...W_F32, shape: [3, 4] }])),
    ContainerError,
    "宣言 shape",
  );
});

// ADR 0069 決定 3: group 形（i4）と keepdim broadcast 形（i8）は受理集合が交わらない別分岐。
// shard 経由でも同じ分岐が掛かる（形検査が単一ファイル面にだけ残る退行を落とす）。
Deno.test("createShardValidator: 格納 i4 の scale が group 形でない shard を落とす", () => {
  const graph = extractIrGraph(graphShard(i4Graph()));
  const validator = createShardValidator(graph);

  // keepdim broadcast 形（i8 なら通る形）は group 形として落ちる
  assertThrows(
    () =>
      validator.intake(
        weightShard([W_I4, { ...W_I4_SCALE, shape: [4, 1], data: f32Bytes([1, 1, 1, 1]) }]),
      ),
    ContainerError,
    "group 形",
  );
  // F32 以外（別 dtype のビット列として読むと全 group が桁違いの値になる）
  assertThrows(
    () =>
      validator.intake(
        weightShard([W_I4, { ...W_I4_SCALE, dtype: "F16", data: new Uint8Array(16) }]),
      ),
    ContainerError,
    "F32 が必要",
  );
});

Deno.test("createShardValidator: 格納 i8 の scale が broadcast できない shard を落とす", () => {
  const graph = extractIrGraph(graphShard(i8Graph()));
  const validator = createShardValidator(graph);

  assertThrows(
    () =>
      validator.intake(
        weightShard([W_I8, { ...W_I8_SCALE, shape: [2, 1], data: f32Bytes([1, 1]) }]),
      ),
    ContainerError,
    "broadcast できない",
  );
});

// scale キーの衝突はグラフ単体で決まる — 衝突相手が別 shard にいる配布形でも検出が
// 「たまたま同居したときだけ」にならないよう、構築時に 1 回で落とす。
Deno.test("createShardValidator: scale キーが他 initializer の実体と衝突するグラフを構築時に落とす", () => {
  const json = baseGraph();
  json.initializers["w"].storage = { dtype: "i8", scale: "enc.b" };
  const graph = extractIrGraph(graphShard(json));

  assertThrows(
    () => createShardValidator(graph),
    ContainerError,
    "initializer 'b' の実体と同じキー",
  );
});

// 共有は衝突と同じ機序（後発の重みが先発の scale で逆量子化される沈黙誤値）で、チャネル数さえ
// 揃えば形検査・余剰・欠け・co-shard の全てを素通りする。IR v1 は重み tying の語彙を持たない
// ので、共有形は取り違えだけを意味する。
const sharedScaleGraph = (
  storage: (name: string) => Record<string, unknown>,
  shapes: { readonly wa: (number | string)[]; readonly wb: (number | string)[] },
): GraphJson => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["matmul"] },
  symbols: ["T"],
  inputs: [{ name: "x", dtype: "f32", shape: ["T", 3] }],
  outputs: ["ha", "hb"],
  initializers: {
    wa: { tensor: "m.wa", storage: storage("m.wa") },
    wb: { tensor: "m.wb", storage: storage("m.wb") },
  },
  values: {
    wa: { dtype: "f32", shape: shapes.wa },
    wb: { dtype: "f32", shape: shapes.wb },
    ha: { dtype: "f32", shape: ["T", shapes.wa[1]] },
    hb: { dtype: "f32", shape: ["T", shapes.wb[1]] },
  },
  nodes: [
    { op: "matmul", ins: ["x", "wa"], outs: ["ha"], attrs: {} },
    { op: "matmul", ins: ["x", "wb"], outs: ["hb"], attrs: {} },
  ],
});

Deno.test("createShardValidator: 2 本の i8 initializer が scale を共有するグラフを構築時に落とす", () => {
  // per-channel scale はどちらも [3,1] になるので、形検査は両方を通してしまう
  const graph = extractIrGraph(graphShard(
    sharedScaleGraph(() => ({ dtype: "i8", scale: "m.s" }), { wa: [3, 4], wb: [3, 8] }),
  ));

  const error = assertThrows(() => createShardValidator(graph), ContainerError, "共有されている");
  // 帰属が分かる診断 MUST — 名前が出ないと直す側はどちらを改名するか決められない
  assertEquals(error.message.includes("'m.s'"), true, error.message);
  assertEquals(error.message.includes("'wa'"), true, error.message);
  assertEquals(error.message.includes("'wb'"), true, error.message);
});

Deno.test("createShardValidator: 行数と group 数が一致する 2 本の i4 initializer の scale 共有も落とす", () => {
  // wa [4,64] group 32 と wb [4,128] group 64 は scale がどちらも [4,2]
  const graph = extractIrGraph(graphShard(
    sharedScaleGraph(
      (tensor) => ({ dtype: "i4", scale: "m.s", group_size: tensor === "m.wa" ? 32 : 64 }),
      { wa: [4, 64], wb: [4, 128] },
    ),
  ));

  assertThrows(() => createShardValidator(graph), ContainerError, "共有されている");
});

// 共有検査が常に鳴る退行の裏取り（scale が別々なら従来どおり構築でき、intake も 2 件返す）。
Deno.test("createShardValidator: scale キーが別々の 2 本は従来どおり構築・受理できる", () => {
  const graph = extractIrGraph(graphShard(
    sharedScaleGraph(
      (tensor) => ({ dtype: "i8", scale: `${tensor}.scale` }),
      { wa: [3, 4], wb: [3, 8] },
    ),
  ));
  const validator = createShardValidator(graph);

  const ready = validator.intake(weightShard([
    { name: "m.wa", dtype: "I8", shape: [3, 4], data: new Uint8Array(12) },
    { name: "m.wa.scale", dtype: "F32", shape: [3, 1], data: f32Bytes([1, 1, 1]) },
    { name: "m.wb", dtype: "I8", shape: [3, 8], data: new Uint8Array(24) },
    { name: "m.wb.scale", dtype: "F32", shape: [3, 1], data: f32Bytes([1, 1, 1]) },
  ]));
  assertEquals(ready.map((item) => item.name), ["wa", "wb"]);
  assertEquals(ready[0].scale?.shape, [3, 1]);
  assertEquals(ready[1].scale?.shape, [3, 1]);
  validator.finish();
});

// ここからテンソル分割（piece）。読み手契約は「キーが `<親名>#NNNNN-of-NNNNN`・親の先頭次元の
// 連続範囲・連続する shard に 1 本ずつ・scale は piece 1 と同居・末尾以外は 4 の倍数バイト」。

Deno.test("parsePieceKey: 5 桁ゼロ詰めで count ≥ 2 の形だけを piece と解釈する", () => {
  assertEquals(parsePieceKey("enc.w#00002-of-00003"), { name: "enc.w", index: 2, count: 3 });
  // 親名に `#` を含む形も末尾の綴りで割れる（貪欲一致 = 最後の `#` が区切り）
  assertEquals(parsePieceKey("a#00001-of-00002#00002-of-00002"), {
    name: "a#00001-of-00002",
    index: 2,
    count: 2,
  });
  // 素のテンソル名・桁数違い・範囲外はどれも piece ではない（= 突合集合の外 → 余剰）
  assertEquals(parsePieceKey("enc.w"), undefined);
  assertEquals(parsePieceKey("enc.w#1-of-2"), undefined);
  assertEquals(parsePieceKey("enc.w#00001-of-00001"), undefined);
  assertEquals(parsePieceKey("enc.w#00003-of-00002"), undefined);
  assertEquals(parsePieceKey("enc.w#00000-of-00002"), undefined);
});

Deno.test("createShardValidator: piece 列を宣言順に受理して rowOffset / first / last を返す", () => {
  const graph = extractIrGraph(graphShard(baseGraph()));
  const validator = createShardValidator(graph);

  const first = validator.intake(weightShard([W_PIECE_1, B_F32]));
  // 戻りは宣言順（w → b）で、piece でも位置は 1 件ぶん
  assertEquals(first.map((item) => item.name), ["w", "b"]);
  assertEquals(first[0].view.shape, [2, 3]);
  assertEquals(first[0].piece, { rowOffset: 0, first: true, last: false });
  // 分割されていないテンソルは従来どおり piece 欄を持たない
  assertEquals(first[1].piece, undefined);

  const second = validator.intake(weightShard([W_PIECE_2]));
  assertEquals(second.map((item) => item.name), ["w"]);
  assertEquals(second[0].piece, { rowOffset: 2, first: false, last: true });
  validator.finish();
});

Deno.test("createShardValidator: piece 列の companion scale は piece 1 とだけ同居する", () => {
  const graph = extractIrGraph(graphShard(i8PieceGraph()));
  const validator = createShardValidator(graph);

  const first = validator.intake(weightShard([B4_F32, W_I8_WIDE_SCALE, W_I8_PIECE_1]));
  assertEquals(first[0].scale?.shape, [4, 1]);
  assertEquals(first[0].piece?.first, true);
  // 2 本目以降の shard に scale は来ない（消費側は piece 1 で読んだ値を持ち越す）
  const second = validator.intake(weightShard([W_I8_PIECE_2]));
  assertEquals(second[0].scale, undefined);
  assertEquals(second[0].piece, { rowOffset: 2, first: false, last: true });
  validator.finish();
});

Deno.test("createShardValidator: piece 2 の shard に置かれた scale を落とす", () => {
  const graph = extractIrGraph(graphShard(i8PieceGraph()));
  const validator = createShardValidator(graph);
  validator.intake(weightShard([W_I8_WIDE_SCALE, W_I8_PIECE_1]));

  const error = assertThrows(
    () => validator.intake(weightShard([W_I8_WIDE_SCALE, W_I8_PIECE_2])),
    ContainerError,
    "別の shard で既に定義されたテンソル (1)",
  );
  assertEquals(error.message.includes("enc.w.scale"), true, error.message);
});

// index は shard 順に 1 ずつ増える。飛び・逆行・総数の食い違いはどれも「どの行が欠けたか」が
// 転送後には分からなくなる形なので、その shard で落とす。
Deno.test("createShardValidator: piece の index の飛び・逆行・count の食い違いを落とす", () => {
  const skipped = createShardValidator(extractIrGraph(graphShard(baseGraph())));
  skipped.intake(weightShard([pieceSpec(1, 3, 1)]));
  assertThrows(
    () => skipped.intake(weightShard([pieceSpec(3, 3, 1)])),
    ContainerError,
    "index 3 が期待 2 と違う",
  );

  const rewound = createShardValidator(extractIrGraph(graphShard(baseGraph())));
  rewound.intake(weightShard([pieceSpec(1, 2, 2)]));
  assertThrows(
    () => rewound.intake(weightShard([pieceSpec(1, 2, 2)])),
    ContainerError,
    "index 1 が期待 2 と違う",
  );

  const miscounted = createShardValidator(extractIrGraph(graphShard(baseGraph())));
  miscounted.intake(weightShard([pieceSpec(1, 2, 2)]));
  assertThrows(
    () => miscounted.intake(weightShard([pieceSpec(2, 3, 2)])),
    ContainerError,
    "総数 3 が先行 piece の 2 と違う",
  );
});

Deno.test("createShardValidator: 同じ shard に同じ親の piece が 2 本ある形を落とす", () => {
  const validator = createShardValidator(extractIrGraph(graphShard(baseGraph())));

  const error = assertThrows(
    () => validator.intake(weightShard([W_PIECE_1, W_PIECE_2])),
    ContainerError,
    "同じ shard に piece が 2 本ある",
  );
  assertEquals(error.message.includes("enc.w#00002-of-00002"), true, error.message);
});

// 「途中まで来て次の shard に続きが無い」は欠けとして読了まで持ち越さない — どの shard から
// 崩れたのかが失われるため、次の shard の intake で落とす。
Deno.test("createShardValidator: piece 列が途切れた shard を落とす", () => {
  const validator = createShardValidator(extractIrGraph(graphShard(baseGraph())));
  validator.intake(weightShard([pieceSpec(1, 3, 1)]));

  assertThrows(
    () => validator.intake(weightShard([B_F32])),
    ContainerError,
    "piece 列がこの shard で途切れた",
  );
});

Deno.test("createShardValidator: piece の dtype と残り次元の不一致を落とす", () => {
  const wrongDtype = createShardValidator(extractIrGraph(graphShard(baseGraph())));
  assertThrows(
    () =>
      wrongDtype.intake(
        weightShard([pieceSpec(1, 2, 2, { dtype: "BF16", data: new Uint8Array(12) })]),
      ),
    ContainerError,
    "F32 が必要",
  );

  // 先頭次元だけが piece ごとに変わる — 残りの次元は宣言と同値 MUST
  const wrongTail = createShardValidator(extractIrGraph(graphShard(baseGraph())));
  assertThrows(
    () =>
      wrongTail.intake(
        weightShard([
          pieceSpec(1, 2, 2, { shape: [2, 4], data: f32Bytes(new Array(8).fill(0.5)) }),
        ]),
      ),
    ContainerError,
    "の行範囲でない",
  );
});

Deno.test("createShardValidator: piece の累積行数が宣言の先頭次元と合わない形を落とす", () => {
  // 超過は即座に（残りの piece を待たずに決まる）
  const over = createShardValidator(extractIrGraph(graphShard(baseGraph())));
  over.intake(weightShard([pieceSpec(1, 2, 3)]));
  assertThrows(
    () => over.intake(weightShard([pieceSpec(2, 2, 3)])),
    ContainerError,
    "累積行数 6 が宣言 shape の先頭次元 4 を超える",
  );

  // 不足は最後の piece で決まる（そこまでは後続で埋まりうる）
  const under = createShardValidator(extractIrGraph(graphShard(baseGraph())));
  under.intake(weightShard([pieceSpec(1, 2, 1)]));
  assertThrows(
    () => under.intake(weightShard([pieceSpec(2, 2, 1)])),
    ContainerError,
    "累積行数が 2 行で宣言 shape の先頭次元 4 行に届かない",
  );
});

// 続きの piece は「行オフセット由来のバイト位置」へ書かれるので、末尾以外の piece が 4 の倍数
// でないと writeBuffer が validation で no-op になる（= 重みが欠けたまま走り出す）。
Deno.test("createShardValidator: 末尾でない piece の非整列バイト長を落とす", () => {
  const graph = extractIrGraph(graphShard(i8Graph()));
  const validator = createShardValidator(graph);

  // i8 [4,3] は 1 行 3 バイト — 2 行の piece は 6 バイトで 4 の倍数にならない
  assertThrows(
    () =>
      validator.intake(weightShard([
        W_I8_SCALE,
        { name: "enc.w#00001-of-00002", dtype: "I8", shape: [2, 3], data: new Uint8Array(6) },
      ])),
    ContainerError,
    "が 6 バイト（4 の倍数が必要",
  );
});

// 1 テンソルは「丸ごと」か「piece 列」のどちらか一方。混在はどちらのバイトが勝つかが転送順で
// 決まる沈黙誤値なので、3 通りとも落とす。
Deno.test("createShardValidator: 丸ごとと piece の混在を落とす", () => {
  const sameShard = createShardValidator(extractIrGraph(graphShard(baseGraph())));
  assertThrows(
    () => sameShard.intake(weightShard([W_F32, W_PIECE_1])),
    ContainerError,
    "の両方でこの shard に在る",
  );

  const wholeFirst = createShardValidator(extractIrGraph(graphShard(baseGraph())));
  wholeFirst.intake(weightShard([W_F32]));
  assertThrows(
    () => wholeFirst.intake(weightShard([W_PIECE_1])),
    ContainerError,
    "別の shard で実体が確定しているのに piece",
  );

  const piecesFirst = createShardValidator(extractIrGraph(graphShard(baseGraph())));
  piecesFirst.intake(weightShard([W_PIECE_1]));
  piecesFirst.intake(weightShard([W_PIECE_2]));
  assertThrows(
    () => piecesFirst.intake(weightShard([W_F32])),
    ContainerError,
    "別の shard で既に定義されたテンソル (1)",
  );
});

// 親が宣言に無い piece キーは piece と解釈しない（= 突合集合の外）。綴りだけで受理集合が
// 広がると、消えた重みの置き土産が「分割の途中」として黙って通る。
Deno.test("createShardValidator: 親が宣言に無い piece キーは余剰として落ちる", () => {
  const validator = createShardValidator(extractIrGraph(graphShard(baseGraph())));

  const error = assertThrows(
    () => validator.intake(weightShard([W_F32, { ...W_PIECE_1, name: "enc.dead#00001-of-00002" }])),
    ContainerError,
    "参照されないテンソル (1)",
  );
  assertEquals(error.message.includes("enc.dead#00001-of-00002"), true, error.message);
});

Deno.test("createShardValidator: 読了時に未完の piece 列を欠けとして列挙する", () => {
  const validator = createShardValidator(extractIrGraph(graphShard(baseGraph(), [B_F32])));
  validator.intake(graphShard(baseGraph(), [B_F32]));
  validator.intake(weightShard([pieceSpec(1, 3, 1)]));

  const error = assertThrows(() => validator.finish(), ContainerError, "不足するテンソル (1)");
  // どこまで来て何行残っているか — 配布形を組み直す側はこの 2 つで作り直す piece を決める
  assertEquals(
    error.message.includes("piece 列が未完（piece 1/3 まで受理・残り 3 行）"),
    true,
    error.message,
  );
});
