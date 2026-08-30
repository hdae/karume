// shard 進行検証（ADR 0070 決定 1）— shard を 1 本ずつ受けて即座に決まる違反を落とし、
// 全 shard 読了後に完全性（欠け）を見る。単一ファイル面（openModel）と同じ検査経路に載って
// いることが前提なので、ここは「shard に割れたときにだけ現れる形」を主に固定する。

import { assertEquals, assertThrows } from "@std/assert";
import {
  ContainerError,
  createShardValidator,
  extractIrGraph,
  IR_METADATA_KEY,
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
