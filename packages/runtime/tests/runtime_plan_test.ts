import { assert, assertEquals, assertThrows } from "@std/assert";
import { DimError } from "../src/format/dims.ts";
import { type IrGraph, parseIrGraph } from "../src/format/ir.ts";
import { OpContractError } from "../src/ops.ts";
import {
  bindSymbols,
  countUses,
  ExecutionError,
  planGraph,
  resolveShape,
  validateGraphContracts,
} from "../src/runtime/plan.ts";
import type { GraphJson } from "./helpers/format.ts";

/** x: [T,4] → matmul(w[4,3]) → h[T,3] → add(b[3]) → y[T,3] */
const linearGraph = (): GraphJson => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["matmul", "add"] },
  symbols: ["T"],
  inputs: [{ name: "x", dtype: "f32", shape: ["T", 4] }],
  outputs: ["y"],
  initializers: {
    w: { tensor: "w", storage: { dtype: "f32" } },
    b: { tensor: "b", storage: { dtype: "f32" } },
  },
  values: {
    w: { dtype: "f32", shape: [4, 3] },
    b: { dtype: "f32", shape: [3] },
    h: { dtype: "f32", shape: ["T", 3] },
    y: { dtype: "f32", shape: ["T", 3] },
  },
  nodes: [
    { op: "matmul", ins: ["x", "w"], outs: ["h"], attrs: {} },
    { op: "add", ins: ["h", "b"], outs: ["y"], attrs: {} },
  ],
});

const parse = (graph: GraphJson): IrGraph => parseIrGraph(JSON.stringify(graph));

Deno.test("シンボル束縛は入力 shape の次元位置から取る", () => {
  const graph = parse(linearGraph());
  assertEquals(bindSymbols(graph, { x: [7, 4] }), { T: 7 });
  assertEquals(bindSymbols(graph, { x: [0, 4] }), { T: 0 });
});

/** 入力の時間軸が派生形 `2T` だけのグラフ（母音検出 CRNN の形 — ADR 0057）。 */
const derivedGraph = (): GraphJson => {
  const graph = linearGraph();
  graph.inputs = [{ name: "x", dtype: "f32", shape: ["2T", 4] }];
  graph.values.h = { dtype: "f32", shape: ["2T", 3] };
  graph.values.y = { dtype: "f32", shape: ["2T", 3] };
  return graph;
};

Deno.test("派生次元だけの入力からもシンボルを解く（解は一意 — ADR 0057）", () => {
  const graph = parse(derivedGraph());
  assertEquals(bindSymbols(graph, { x: [284, 4] }), { T: 142 });
  assertEquals(bindSymbols(graph, { x: [0, 4] }), { T: 0 });
  // 明示 seed は従来どおり優先され、解と食い違えば落ちる。
  assertEquals(bindSymbols(graph, { x: [284, 4] }, { T: 142 }), { T: 142 });
  assertThrows(() => bindSymbols(graph, { x: [284, 4] }, { T: 284 }), ExecutionError);
});

Deno.test("宣言の形をしていない実寸は fail loudly（丸めない）", () => {
  // 母音検出 CRNN の「10ms フレーム数は偶数」がここで落ちる（黙って 142 や 143 に丸めない）。
  const graph = parse(derivedGraph());
  const error = assertThrows(
    () => bindSymbols(graph, { x: [285, 4] }),
    ExecutionError,
    "宣言 '2T' の形をしていない",
  );
  assert(error.message.includes("実測 285"), error.message);
});

Deno.test("束縛と実 shape・宣言の食い違いは全て fail loudly", () => {
  const graph = parse(linearGraph());
  // 数値次元の不一致
  assertThrows(() => bindSymbols(graph, { x: [7, 5] }), ExecutionError);
  // rank の不一致
  assertThrows(() => bindSymbols(graph, { x: [7] }), ExecutionError);
  // 入力の欠落・余剰
  assertThrows(() => bindSymbols(graph, {}), ExecutionError);
  assertThrows(() => bindSymbols(graph, { x: [7, 4], z: [1] }), ExecutionError);
  // 非負整数でない次元
  assertThrows(() => bindSymbols(graph, { x: [-1, 4] }), ExecutionError);
  // 明示束縛が入力 shape 由来の束縛と衝突
  assertThrows(() => bindSymbols(graph, { x: [7, 4] }, { T: 8 }), ExecutionError);
  // 宣言に無いシンボルの明示束縛
  assertThrows(() => bindSymbols(graph, { x: [7, 4] }, { S: 8 }), ExecutionError);
});

Deno.test("明示束縛が入力 shape と一致していれば受理する", () => {
  const graph = parse(linearGraph());
  assertEquals(bindSymbols(graph, { x: [7, 4] }, { T: 7 }), { T: 7 });
});

Deno.test("束縛の有無は Object.hasOwn で見る（prototype 由来の名前は束縛済みにしない）", () => {
  const source = linearGraph();
  source.symbols = ["toString"];
  source.inputs = [{ name: "x", dtype: "f32", shape: ["toString", 4] }];
  source.values.h = { dtype: "f32", shape: ["toString", 3] };
  source.values.y = { dtype: "f32", shape: ["toString", 3] };
  const graph = parse(source);
  assertEquals(bindSymbols(graph, { x: [6, 4] }), { toString: 6 });
});

Deno.test("bindSymbols は '__proto__' というシンボル名の束縛を own property として保全する", () => {
  const source = linearGraph();
  source.symbols = ["__proto__"];
  source.inputs = [{ name: "x", dtype: "f32", shape: ["__proto__", 4] }];
  source.values.h = { dtype: "f32", shape: ["__proto__", 3] };
  source.values.y = { dtype: "f32", shape: ["__proto__", 3] };
  const graph = parse(source);

  const bindings = bindSymbols(graph, { x: [6, 4] });
  // 実行系によっては `o["__proto__"] = v` でも own property が作られるため、hasOwn だけでは
  // 素の `{}` への退行を検出できない。器が null プロトタイプであること（＝ 受理集合に
  // エンジン差を持ち込まないこと）を併せて固定する。
  assertEquals(Object.getPrototypeOf(bindings), null);
  assertEquals(Object.hasOwn(bindings, "__proto__"), true);
  assertEquals(bindings["__proto__"], 6);
  // 束縛が後段まで生きている（shape が数値に落ちる）
  assertEquals(resolveShape(graph.values["y"].shape, bindings), [6, 3]);

  // 明示束縛（seed）側も同じ。MUST: 計算キーで作る — リテラルの `__proto__:` は own key では
  // なく [[Prototype]] 指定になり、Object.entries に載らずに検査対象を外す。
  const seeded = bindSymbols(graph, { x: [6, 4] }, { ["__proto__"]: 6 });
  assertEquals(Object.getPrototypeOf(seeded), null);
  assertEquals(seeded["__proto__"], 6);
  assertThrows(() => bindSymbols(graph, { x: [6, 4] }, { ["__proto__"]: 5 }), ExecutionError);
});

Deno.test("素の形と派生形が同居しても束縛は 1 つ（食い違いは fail loudly）", () => {
  const source = linearGraph();
  // 入力は inputs[] だけで宣言される（values{} に置くと二重宣言で IR が落ちる）
  source.inputs = [
    { name: "x", dtype: "f32", shape: ["T", 4] },
    { name: "m", dtype: "f32", shape: ["2T+1"] },
  ];
  const graph = parse(source);
  assertEquals(bindSymbols(graph, { x: [3, 4], m: [7] }), { T: 3 });
  // 派生側が別の T を指す（9 → T=4）: 衝突として落ちる。
  assertThrows(() => bindSymbols(graph, { x: [3, 4], m: [9] }), ExecutionError);
  // 派生側がそもそも `2T+1` の形をしていない（6 は奇数長にならない）。
  assertThrows(() => bindSymbols(graph, { x: [3, 4], m: [6] }), ExecutionError);
});

Deno.test("planGraph が全値の shape を解決し、ノード出力を契約から照合する", () => {
  const graph = parse(linearGraph());
  const plan = planGraph(graph, { T: 5 });
  assertEquals(plan.shapes.get("x"), [5, 4]);
  assertEquals(plan.shapes.get("w"), [4, 3]);
  assertEquals(plan.shapes.get("h"), [5, 3]);
  assertEquals(plan.nodes.length, 2);
  assertEquals(plan.nodes[0].outputs, [{ name: "h", shape: [5, 3], dtype: "f32" }]);
  assertEquals(plan.nodes[1].contract.kind, "binary");
});

Deno.test("宣言 shape と計算 shape の不一致・未束縛シンボルは fail loudly", () => {
  const source = linearGraph();
  source.values.h = { dtype: "f32", shape: ["T", 4] };
  assertThrows(() => planGraph(parse(source), { T: 5 }), ExecutionError);
  assertThrows(() => planGraph(parse(linearGraph()), {}), DimError);
  assertThrows(() => resolveShape(["T"], {}), DimError);
  assertEquals(resolveShape([2, "T+1"], { T: 3 }), [2, 4]);
});

Deno.test("useCounts は node.ins の厳密な延べ計数（同じ値を 2 回取れば 2）", () => {
  const source = linearGraph();
  source.requires.ops = ["matmul", "add", "mul"];
  source.nodes.push({ op: "mul", ins: ["y", "y"], outs: ["z"], attrs: {} });
  source.values.z = { dtype: "f32", shape: ["T", 3] };
  source.outputs = ["z"];
  const counts = countUses(parse(source));
  assertEquals(counts.get("y"), 2);
  assertEquals(counts.get("x"), 1);
  assertEquals(counts.get("z"), undefined);
});

Deno.test("契約検査は未対応 op・非空 attrs・非 f32 dtype を構築時に落とす", () => {
  assertEquals(validateGraphContracts(parse(linearGraph())), undefined);

  const foreignOp = linearGraph();
  foreignOp.requires.ops = ["matmul", "softmax"];
  foreignOp.nodes[1] = { op: "softmax", ins: ["h"], outs: ["y"], attrs: {} };
  assertThrows(() => validateGraphContracts(parse(foreignOp)), OpContractError);

  const withAttrs = linearGraph();
  withAttrs.nodes[1].attrs = { alpha: 1 };
  assertThrows(() => validateGraphContracts(parse(withAttrs)), OpContractError);

  const intInput = linearGraph();
  intInput.inputs = [{ name: "x", dtype: "i32", shape: ["T", 4] }];
  assertThrows(() => validateGraphContracts(parse(intInput)), OpContractError);
});

/** table[6,5]（Tmax 形の i32 定数）→ sym_prefix_slice → y[T,T]。 */
const prefixSliceGraph = (overrides: Partial<GraphJson> = {}): GraphJson => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["sym_prefix_slice"] },
  symbols: ["T"],
  inputs: [{ name: "bind", dtype: "f32", shape: ["T"] }],
  outputs: ["y"],
  initializers: { table: { tensor: "table", storage: { dtype: "i32" } } },
  values: {
    table: { dtype: "i32", shape: [6, 5] },
    y: { dtype: "i32", shape: ["T", "T"] },
  },
  nodes: [{
    op: "sym_prefix_slice",
    ins: ["table"],
    outs: ["y"],
    attrs: {
      sym: "T",
      slices: [{ dim: 0, coeff: 1, offset: 0 }, { dim: 1, coeff: 1, offset: 0 }],
    },
  }],
  ...overrides,
});

// ADR 0010: sym_prefix_slice の「グラフ全体を見ないと判定できない」契約は束縛前に落とす。
Deno.test("sym_prefix_slice のグラフ文脈の契約は Session 構築時に落ちる", () => {
  validateGraphContracts(parse(prefixSliceGraph()));

  // sym が graph.symbols に無い（実行時に束縛が取れず prefix 長が決まらない）
  const foreignSym = prefixSliceGraph();
  foreignSym.nodes[0].attrs = {
    sym: "U",
    slices: [{ dim: 0, coeff: 1, offset: 0 }],
  };
  assertThrows(() => validateGraphContracts(parse(foreignSym)), ExecutionError);

  // dim が入力 rank の外
  const badDim = prefixSliceGraph();
  badDim.nodes[0].attrs = { sym: "T", slices: [{ dim: 2, coeff: 1, offset: 0 }] };
  assertThrows(() => validateGraphContracts(parse(badDim)), ExecutionError);

  // MUST: 入力が記号 shape の形は束縛後の数値からは見分けが付かない（T = Tmax の run では
  // 一致してしまう）。宣言の形を見られるここでしか検出できない。
  const symbolicSource = prefixSliceGraph();
  symbolicSource.initializers = {};
  symbolicSource.inputs = [
    { name: "bind", dtype: "f32", shape: ["T"] },
    { name: "table", dtype: "i32", shape: ["T", 5] },
  ];
  delete symbolicSource.values["table"];
  assertThrows(() => validateGraphContracts(parse(symbolicSource)), ExecutionError);
});

// prefix 長は attrs（+ 束縛）から計算し、宣言と照合する（宣言が目標形の reshape とは違う）。
Deno.test("sym_prefix_slice の出力 shape は束縛から計算されて宣言と突合される", () => {
  const graph = parse(prefixSliceGraph());
  const plan = planGraph(graph, bindSymbols(graph, { bind: [3] }));
  assertEquals(plan.nodes[0].outputs[0].shape, [3, 3]);

  // Tmax 超過（定数バッファの範囲外読み出しになる）は束縛時点で落ちる
  assertThrows(() => planGraph(graph, bindSymbols(graph, { bind: [6] })), OpContractError);
});

// ADR 0014: slice / flip の対象軸は**宣言レベルで静的**（記号軸の切り出しは sym_prefix_slice
// の担当）。cat の連結軸だけは ADR 0046 が「同一シンボルの一次和」まで緩めた。どちらも
// 束縛後の数値 shape では見分けが付かない（別シンボルどうしも同じ値に解決されうる）ので、
// 宣言の形を見られる validateGraphContracts でしか検出できない。
const layoutAxisGraph = (
  op: string,
  attrs: Record<string, unknown>,
  ins: readonly { name: string; shape: (number | string)[] }[],
  outShape: (number | string)[],
): GraphJson => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: [op] },
  // 素の形（'T' 等）で現れる入力次元だけが束縛源になる（派生形しか無いシンボルは宣言できない）
  symbols: [...new Set(ins.flatMap((spec) => spec.shape.filter((dim) => typeof dim === "string")))],
  inputs: ins.map((spec) => ({ name: spec.name, dtype: "f32", shape: [...spec.shape] })),
  outputs: ["y"],
  initializers: {},
  values: { y: { dtype: "f32", shape: [...outShape] } },
  nodes: [{ op, ins: ins.map((spec) => spec.name), outs: ["y"], attrs }],
});

Deno.test("slice / flip の記号軸は Session 構築時に落ちる（静的軸のみ）", () => {
  // 受理: 対象軸は静的で、他の軸が記号なのは構わない
  validateGraphContracts(
    parse(layoutAxisGraph("slice", { dim: 1, start: 1, end: 3 }, [{
      name: "x",
      shape: ["T", 4],
    }], ["T", 2])),
  );
  validateGraphContracts(
    parse(layoutAxisGraph("flip", { dim: 1 }, [{ name: "x", shape: ["T", 4] }], ["T", 4])),
  );

  // 拒否: 対象軸そのものが記号
  assertThrows(
    () =>
      validateGraphContracts(
        parse(layoutAxisGraph("slice", { dim: 0, start: 0, end: 2 }, [{
          name: "x",
          shape: ["T", 4],
        }], [2, 4])),
      ),
    ExecutionError,
  );
  assertThrows(
    () =>
      validateGraphContracts(
        parse(layoutAxisGraph("flip", { dim: 0 }, [{ name: "x", shape: ["T", 4] }], ["T", 4])),
      ),
    ExecutionError,
  );
});

// ADR 0046 が ADR 0014 の「連結軸は静的」を改訂した。DiT の joint attention は
// `cat([self | context], dim=1)` で self 側の軸長が記号（`S+1519`）— 記号 1 本 + 定数は
// 次元言語 `coeff·sym+offset` にそのまま載るので、旧規則の拒否理由（和が載らない）が
// 成り立たない。残る拒否は**異なるシンボルの混在**だけ。
Deno.test("cat の連結軸は同一シンボルの一次和まで受理し、異シンボルの混在で落ちる", () => {
  // 受理: 連結軸が静的（他の軸が記号なのは構わない）
  validateGraphContracts(
    parse(layoutAxisGraph("cat", { dim: 1 }, [
      { name: "a", shape: ["T", 2] },
      { name: "b", shape: ["T", 3] },
    ], ["T", 5])),
  );
  // 受理: 記号 + 定数（`S+1519` と同型）
  validateGraphContracts(
    parse(layoutAxisGraph("cat", { dim: 0 }, [
      { name: "a", shape: ["T", 2] },
      { name: "b", shape: [3, 2] },
    ], ["T+3", 2])),
  );
  // 受理: 同一シンボルどうし（係数が積み上がる）
  validateGraphContracts(
    parse(layoutAxisGraph("cat", { dim: 0 }, [
      { name: "a", shape: ["T", 2] },
      { name: "b", shape: ["T", 2] },
    ], ["2T", 2])),
  );

  // 拒否: 異なるシンボルの混在（和が 1 次元 1 シンボルの文法に載らない）
  assertThrows(
    () =>
      validateGraphContracts(
        parse(layoutAxisGraph("cat", { dim: 0 }, [
          { name: "a", shape: ["T", 2] },
          { name: "b", shape: ["U", 2] },
        ], ["T", 2])),
      ),
    ExecutionError,
    "異なるシンボル",
  );
});

// cat のアリティは**下限** 2（可変アリティ — ADR 0014）。1 本は恒等コピーで語彙に無い。
Deno.test("cat の入力は 2 本以上で、3 本以上も契約検査を通る", () => {
  validateGraphContracts(
    parse(layoutAxisGraph("cat", { dim: 1 }, [
      { name: "a", shape: [2, 1] },
      { name: "b", shape: [2, 2] },
      { name: "c", shape: [2, 3] },
    ], [2, 6])),
  );
  assertThrows(
    () =>
      validateGraphContracts(
        parse(layoutAxisGraph("cat", { dim: 1 }, [{ name: "a", shape: [2, 1] }], [2, 1])),
      ),
    OpContractError,
  );
});

// ---- state 参照ノード（ADR 0067 決定 4 / 5 / 5b） --------------------------

/**
 * `kv` スロットを読む attention 1 本と書く `state_append` 1 本（発行順は決定 5b の
 * 「全読者 → append」）。`extra` でノード列だけを差し替えて順序検査の変異を作る。
 */
const stateGraph = (
  nodes?: GraphJson["nodes"],
  states?: GraphJson["states"],
): GraphJson => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["attention", "state_append"] },
  symbols: ["M", "C"],
  inputs: [
    { name: "q", dtype: "f32", shape: [1, 8, "M", 4] },
    { name: "k", dtype: "f32", shape: [1, 2, "M", 4] },
    { name: "v", dtype: "f32", shape: [1, 2, "M", 4] },
  ],
  outputs: ["o"],
  initializers: {},
  values: { o: { dtype: "f32", shape: [1, 8, "M", 4] } },
  states: states ?? {
    "kv.k": { dtype: "f32", shape: [1, 2, "C", 4] },
    "kv.v": { dtype: "f32", shape: [1, 2, "C", 4] },
  },
  nodes: nodes ?? [
    {
      op: "attention",
      ins: ["q", "k", "v"],
      outs: ["o"],
      attrs: { scale: 0.5 },
      states: { k: "kv.k", v: "kv.v" },
    },
    { op: "state_append", ins: ["k"], outs: [], attrs: {}, states: { slot: "kv.k" } },
    { op: "state_append", ins: ["v"], outs: [], attrs: {}, states: { slot: "kv.v" } },
  ],
});

/** 容量記号 `C` は入力に現れない（束縛点は createGenerationContext — ADR 0066 追記 7）。 */
const stateShapes = (capacity = 16): ReadonlyMap<string, readonly number[]> =>
  new Map([["kv.k", [1, 2, capacity, 4]], ["kv.v", [1, 2, capacity, 4]]]);

Deno.test("state 参照グラフは契約検査を通り、スロット shape を渡せば計画できる", () => {
  const graph = parse(stateGraph());
  validateGraphContracts(graph);

  const plan = planGraph(graph, { M: 3, C: 16 }, stateShapes());
  assertEquals(plan.nodes[0].outputs.map((out) => out.name), ["o"]);
  assertEquals(plan.nodes[0].outputs[0].shape, [1, 8, 3, 4]);
  // effect op は出力列が空（`outs` が 0 本であることが計画にもそのまま出る）
  assertEquals(plan.nodes[1].outputs, []);
  assertEquals(plan.nodes[2].outputs, []);
});

Deno.test("スロット shape を渡さない計画は state 参照ノードで fail loudly", () => {
  const graph = parse(stateGraph());
  const error = assertThrows(
    () => planGraph(graph, { M: 3, C: 16 }),
    OpContractError,
    "GenerationContext",
  );
  assert(error.message.includes("kv.k"), error.message);
});

// ADR 0067 決定 5b: state 参照はデータ辺を張らないので、順序の契約は nodes 配列順そのもの。
Deno.test("同一スロットへの state_append は 1 step に 1 本まで", () => {
  const nodes = stateGraph().nodes;
  assertThrows(
    () =>
      validateGraphContracts(parse(stateGraph([
        nodes[0],
        nodes[1],
        { op: "state_append", ins: ["v"], outs: [], attrs: {}, states: { slot: "kv.k" } },
        nodes[2],
      ]))),
    ExecutionError,
    "state_append が 2 本",
  );
});

Deno.test("state_append より後に同じスロットの読者を置けない", () => {
  const nodes = stateGraph().nodes;
  const error = assertThrows(
    () => validateGraphContracts(parse(stateGraph([nodes[1], nodes[2], nodes[0]]))),
    ExecutionError,
    "より後に読者",
  );
  // 「append より後に読者が居る」は**そのスロットに触れる並び**だけの話（他スロットの
  // ノードが間に挟まっても順序は変わらない）
  assert(error.message.includes("kv.k") || error.message.includes("kv.v"), error.message);
});

Deno.test("同一スロットに触れるノードの window は存在有無も値も一致する", () => {
  const nodes = stateGraph().nodes;
  const windowed = (window: number | undefined): GraphJson["nodes"] => [
    {
      ...nodes[0],
      attrs: window === undefined ? { scale: 0.5 } : { scale: 0.5, window },
    },
    { op: "state_append", ins: ["k"], outs: [], attrs: { window: 8 }, states: { slot: "kv.k" } },
    { op: "state_append", ins: ["v"], outs: [], attrs: { window: 8 }, states: { slot: "kv.v" } },
  ];
  // 読み側だけ window 宣言が無い（= 全 context を走査する）形は沈黙誤読になる
  assertThrows(
    () => validateGraphContracts(parse(stateGraph(windowed(undefined)))),
    ExecutionError,
    "attrs.window が食い違う",
  );
  // 値が違う形も同じ
  assertThrows(
    () => validateGraphContracts(parse(stateGraph(windowed(4)))),
    ExecutionError,
    "attrs.window が食い違う",
  );
  // 一致していれば通る
  validateGraphContracts(parse(stateGraph(windowed(8))));
});

Deno.test("sliding の window はスロット容量を超えられない（読み書きの両側）", () => {
  const nodes = stateGraph().nodes;
  const windowed: GraphJson["nodes"] = [
    { ...nodes[0], attrs: { scale: 0.5, window: 32 } },
    { op: "state_append", ins: ["k"], outs: [], attrs: { window: 32 }, states: { slot: "kv.k" } },
    { op: "state_append", ins: ["v"], outs: [], attrs: { window: 32 }, states: { slot: "kv.v" } },
  ];
  const graph = parse(stateGraph(windowed));
  validateGraphContracts(graph);
  // 容量 32 なら通り、16 では落ちる（宣言だけでは決まらない = 容量は context が決める）
  planGraph(graph, { M: 3, C: 32 }, stateShapes(32));
  assertThrows(
    () => planGraph(graph, { M: 3, C: 16 }, stateShapes(16)),
    OpContractError,
    "スロット容量 16 を超える",
  );
});
