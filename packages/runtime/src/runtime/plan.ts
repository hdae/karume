/**
 * 実行計画（純関数）: シンボル束縛 → 全値の shape 解決 → ノードごとの出力 shape 評価と
 * 宣言との照合。GPU に触れないのでアダプタ無し環境でも検証できる。
 *
 * 静的形状（ADR 0004）: 全ノードの出力 shape は 1 dispatch も出す前に確定する。ここで
 * 落ちなかったグラフだけが executor のエンコード段に進む。
 */

import { evalDim, parseDim } from "../format/dims.ts";
import type { IrDim, IrDtype, IrGraph, IrNode } from "../format/ir.ts";
import {
  assertNodeContract,
  catDim,
  computeOutputShape,
  flipDim,
  IO_DTYPES,
  type OpContract,
  resolveNodeDtypes,
  sliceAttrs,
  symPrefixSliceAttrs,
  WEIGHT_CHANNEL_AXES,
  WEIGHT_SLOTS,
} from "../ops.ts";

/** 実行時に契約・束縛・宣言のいずれかが破れた（黙って近似せず必ずこれを投げる）。 */
export class ExecutionError extends Error {
  override readonly name = "ExecutionError";
}

/** 記号次元 → 実行時の具体値。 */
export type SymbolBindings = Readonly<Record<string, number>>;

export type NodePlan = {
  readonly node: IrNode;
  readonly contract: OpContract;
  readonly inputShapes: readonly (readonly number[])[];
  /** 入力の意味論 dtype（宣言由来）。カーネルの要素型はここから決まる。 */
  readonly inputDtypes: readonly IrDtype[];
  readonly outputName: string;
  readonly outputShape: readonly number[];
  /** 出力の意味論 dtype（契約から導き宣言と照合済み — {@link resolveNodeDtypes}）。 */
  readonly outputDtype: IrDtype;
};

export type GraphPlan = {
  readonly bindings: SymbolBindings;
  /** 入力・initializer・全ノード出力の解決済み shape。 */
  readonly shapes: ReadonlyMap<string, readonly number[]>;
  readonly nodes: readonly NodePlan[];
};

/** 宣言 shape を束縛で具体化する。未束縛シンボルは dims.ts が DimError で落とす。 */
export const resolveShape = (
  shape: readonly IrDim[],
  bindings: SymbolBindings,
): number[] =>
  shape.map((dim) => (typeof dim === "number" ? dim : evalDim(parseDim(dim), bindings)));

/** 値名 → 宣言（入力は inputs[]、それ以外は values{} — IR がちょうど 1 箇所を保証する）。 */
const declarationOf = (
  graph: IrGraph,
  name: string,
): { readonly dtype: IrDtype; readonly shape: readonly IrDim[] } => {
  const input = graph.inputs.find((spec) => spec.name === name);
  if (input !== undefined) return { dtype: input.dtype, shape: input.shape };
  if (!Object.hasOwn(graph.values, name)) {
    throw new ExecutionError(`値 '${name}' の宣言が無い`);
  }
  const value = graph.values[name];
  return { dtype: value.dtype, shape: value.shape };
};

/** ノードの入出力 dtype を宣言から集めて契約と照合し、出力 dtype を返す。 */
const nodeDtypes = (
  graph: IrGraph,
  node: IrNode,
  contract: OpContract,
  where: string,
): { readonly inputs: readonly IrDtype[]; readonly output: IrDtype } => {
  const inputs = node.ins.map((name) => declarationOf(graph, name).dtype);
  const declaredOutput = declarationOf(graph, node.outs[0]).dtype;
  return { inputs, output: resolveNodeDtypes(contract, node, inputs, declaredOutput, where) };
};

/**
 * 束縛に依存しない契約検査（Session 構築時に 1 回）。アリティ・attrs・dtype 規則を見る。
 * MUST: 未対応 dtype はここで落とす — カーネルは op ごとに解禁した要素型しか持たず、
 * 別の型の値を dispatch すると例外なしにビット列を読み替えた誤値が出る。
 */
export const validateGraphContracts = (graph: IrGraph): void => {
  graph.nodes.forEach((node, index) => {
    const where = `nodes[${index}] (${node.op})`;
    const contract = assertNodeContract(node, where);
    nodeDtypes(graph, node, contract, where);
    if (contract.kind === "symPrefixSlice") assertSymPrefixSlice(graph, node, where);
    if (contract.kind === "slice" || contract.kind === "flip") {
      assertStaticLayoutAxis(graph, node, contract.kind, where);
    }
    if (contract.kind === "cat") assertCatAxis(graph, node, where);
  });
  for (const spec of graph.inputs) {
    if (!IO_DTYPES.includes(spec.dtype)) {
      throw new ExecutionError(
        `入力 '${spec.name}' の意味論 dtype '${spec.dtype}' は転送できない（${
          IO_DTYPES.join(" / ")
        } のみ）`,
      );
    }
  }
};

/**
 * sym_prefix_slice の**グラフ文脈が要る**契約（ADR 0010）。attrs の形は契約表が済ませており、
 * ここは「グラフ全体を見ないと判定できない」3 点だけを見る。
 *
 * 1. `sym` が `graph.symbols` にある（無ければ実行時に束縛が取れず、prefix 長が決まらない）
 * 2. `dim` が入力 rank の内側
 * 3. 入力の宣言 shape が**記号を含まない静的形**（= Tmax 形）。ここが記号だと、読み出し
 *    stride が実行のたびに縮む値から組まれ、T < Tmax で必ず別の行を読む
 *
 * MUST: 束縛前に落とす。3 は束縛後の数値 shape からは見分けが付かない（T = Tmax の run では
 * 数値が一致してしまう）ので、宣言の形を見られるここでしか検出できない。
 */
const assertSymPrefixSlice = (graph: IrGraph, node: IrNode, where: string): void => {
  const { sym, slices } = symPrefixSliceAttrs(node.attrs, where);
  if (!graph.symbols.includes(sym)) {
    throw new ExecutionError(
      `${where}: sym_prefix_slice の sym '${sym}' が graph.symbols [${
        graph.symbols.join(", ")
      }] に無い`,
    );
  }
  const source = declarationOf(graph, node.ins[0]).shape;
  if (source.some((dim) => typeof dim !== "number")) {
    throw new ExecutionError(
      `${where}: sym_prefix_slice の入力 '${node.ins[0]}' の宣言 shape [${
        source.join(",")
      }] に記号次元がある（入力は Tmax で焼いた静的形でなければならない）`,
    );
  }
  for (const slice of slices) {
    if (slice.dim >= source.length) {
      throw new ExecutionError(
        `${where}: sym_prefix_slice の dim ${slice.dim} が入力 rank ${source.length} の外`,
      );
    }
  }
};

/**
 * slice / flip の対象軸が**宣言レベルで静的**であることを見る（ADR 0014）。
 *
 * MUST: 束縛前に落とす。束縛後の数値 shape では記号かどうかが見分けられない（T = 実長の
 * run では数値が一致してしまう）ので、宣言の形を見られるここでしか検出できない —
 * assertSymPrefixSlice が「入力は Tmax 形」を見るのと同じ層。
 *
 * 軸ごとの理由:
 * - `slice` — 記号軸の切り出しは `sym_prefix_slice` の担当で、こちらは静的専業
 *   （ADR 0014 が「重複させない」と決めた分担）。
 * - `flip` — 実測は全て静的軸（flow の 192ch / sdp の 2ch）。動的軸の反転はカーネル上は
 *   書けるが、要求実測が出るまで語彙を広げない（softmax の dim と同じ絞り方）。
 *
 * `cat` の連結軸だけは ADR 0046 で緩めた（{@link assertCatAxis}）。エクスポータ側の shape 層も
 * 同じ規則を持つ（受理集合を両側で揃える）。
 */
const assertStaticLayoutAxis = (
  graph: IrGraph,
  node: IrNode,
  kind: "slice" | "flip",
  where: string,
): void => {
  const dim = kind === "slice" ? sliceAttrs(node.attrs, where).dim : flipDim(node.attrs, where);
  for (const name of node.ins) {
    const shape = declarationOf(graph, name).shape;
    if (dim < shape.length && typeof shape[dim] !== "number") {
      throw new ExecutionError(
        `${where}: ${node.op} の軸 ${dim} が入力 '${name}' の宣言 [${
          shape.join(",")
        }] で記号次元（記号軸の切り出しは sym_prefix_slice の担当）`,
      );
    }
  }
};

/**
 * cat の連結軸は〈定数〉または〈**同一**シンボルの一次式〉（ADR 0046 が ADR 0014 を改訂）。
 *
 * 総和 `Σ(coeff_i·sym + offset_i)` は同一シンボルなら次元言語 `coeff·sym+offset` にそのまま
 * 載る（`S`+1519 → `S+1519`、`S`+`S` → `2S`）。**異なるシンボルの混在**だけが表現できない。
 *
 * MUST: 束縛前に落とす。束縛後の数値 shape では別シンボルどうしが同じ値に解決されうるので、
 * 宣言の形を見られるここでしか検出できない（slice / flip の静的軸検査と同じ層）。
 * 出力の軸長が総和と一致することは planGraph が束縛後の数値で照合する — ここは「宣言だけで
 * 表現不能な形」を止める側だけを持つ。
 */
const assertCatAxis = (graph: IrGraph, node: IrNode, where: string): void => {
  const dim = catDim(node.attrs, where);
  let sym: string | undefined;
  for (const name of node.ins) {
    const shape = declarationOf(graph, name).shape;
    // rank 違い（dim が外）は束縛後に computeOutputShape が見る（ここの担当ではない）。
    if (dim >= shape.length) continue;
    const extent = shape[dim];
    if (typeof extent === "number") continue;
    const parsed = parseDim(extent);
    if (sym !== undefined && sym !== parsed.sym) {
      throw new ExecutionError(
        `${where}: ${node.op} の連結軸 ${dim} に異なるシンボル（${sym} と ${parsed.sym}）が` +
          `混ざる（入力 '${name}' の宣言 [${shape.join(",")}]）` +
          " — 和が次元言語 coeff·sym+offset に載らない",
      );
    }
    sym = parsed.sym;
  }
};

/**
 * 値名 → 宣言 dtype（入力・initializer・全ノード出力）。束縛に依存しないので Session 構築時に
 * 1 回だけ引けばよい。転送（upload）と readback の TypedArray はこれで決まる。
 */
export const declaredDtypes = (graph: IrGraph): ReadonlyMap<string, IrDtype> => {
  const dtypes = new Map<string, IrDtype>();
  for (const spec of graph.inputs) dtypes.set(spec.name, spec.dtype);
  for (const [name, value] of Object.entries(graph.values)) dtypes.set(name, value.dtype);
  return dtypes;
};

/**
 * 重みスロットで消費される initializer → per-channel scale の**チャネル軸**（ADR 0019）。
 *
 * 適格な initializer（{@link eligibleCompressedInitializers}）は消費が重みスロットだけなので、
 * 消費側 op から軸が一意に決まる。同じ重みを軸の違う 2 op（例: linear と
 * conv_transpose1d）が食う形は scale の意味が割れるので fail loudly。
 *
 * MUST: 判定は {@link WEIGHT_SLOTS} と対の {@link WEIGHT_CHANNEL_AXES} から引く。軸を
 * カーネル側の定数から推測すると、scale の平坦添字がずれても shape 検査を素通りする。
 */
export const weightChannelAxes = (graph: IrGraph): ReadonlyMap<string, number> => {
  const axes = new Map<string, number>();
  for (const node of graph.nodes) {
    const weightSlot = WEIGHT_SLOTS.get(node.op);
    if (weightSlot === undefined) continue;
    const name = node.ins[weightSlot];
    if (name === undefined || !Object.hasOwn(graph.initializers, name)) continue;
    const axis = WEIGHT_CHANNEL_AXES.get(node.op);
    if (axis === undefined) {
      throw new ExecutionError(`op '${node.op}' に per-channel scale のチャネル軸の定義が無い`);
    }
    const known = axes.get(name);
    if (known !== undefined && known !== axis) {
      throw new ExecutionError(
        `initializer '${name}': チャネル軸が消費側で食い違う（${known} と ${axis}）`,
      );
    }
    axes.set(name, axis);
  }
  return axes;
};

/**
 * 圧縮格納のまま GPU 常駐**できる** initializer（ADR 0018 / 0019 の適格判定 — f16 と i8 で
 * 共用する。判定はグラフ構造だけで決まり、格納 dtype を見ないため分ける理由が無い）。
 *
 * 適格 = 「その initializer の消費が、融合 5 op の重みスロット（{@link WEIGHT_SLOTS}）**だけ**」。
 * 束縛にも GPU にも依存しない。
 *
 * MUST: 消費が 1 つでも重みスロット以外にあれば適格外。同じテンソルを linear の重みと
 * elementwise の被演算子の両方で使うグラフでは、後者のカーネルが f32 として読むため、
 * 圧縮のまま上げるとビット列の読み替えになる（例外は出ない）。
 * MUST: 消費ゼロの initializer も適格外にする。実行に使われないバイトを「GPU 常駐圧縮」として
 * 数えると診断（ADR 0006）が実態からずれる。
 */
export const eligibleCompressedInitializers = (graph: IrGraph): ReadonlySet<string> => {
  const initializers = new Set(Object.keys(graph.initializers));
  const eligible = new Set<string>();
  const disqualified = new Set<string>();
  for (const node of graph.nodes) {
    const weightSlot = WEIGHT_SLOTS.get(node.op);
    node.ins.forEach((name, slot) => {
      if (!initializers.has(name)) return;
      if (slot === weightSlot) eligible.add(name);
      else disqualified.add(name);
    });
  }
  for (const name of disqualified) eligible.delete(name);
  return eligible;
};

/** 値名 → グラフ内での消費回数。MUST: `node.ins` の厳密な延べ計数（同じ値を 2 回取れば 2）。 */
export const countUses = (graph: IrGraph): ReadonlyMap<string, number> => {
  const counts = new Map<string, number>();
  for (const node of graph.nodes) {
    for (const name of node.ins) counts.set(name, (counts.get(name) ?? 0) + 1);
  }
  return counts;
};

const assertExtent = (size: number, where: string): number => {
  if (!Number.isSafeInteger(size) || size < 0) {
    throw new ExecutionError(`${where}: 次元 ${String(size)} が非負整数でない`);
  }
  return size;
};

/**
 * 実入力の shape からシンボルを束縛する。
 *
 * MUST: 束縛は**入力 shape の次元位置**から取る（要素数からの逆算はしない — 複数シンボルや
 * 係数付き次元があると解が一意でなく、静かに誤った束縛が通る）。派生形（`2T+1` 等）の
 * 位置は束縛源にせず、束縛確定後に評価して実 shape と照合する。
 * MUST: 束縛の有無は `Object.hasOwn` のみで見る。`bindings[sym] !== undefined` は
 * Object.prototype 由来の `toString` 等が素通りし、以後の算術が黙って NaN 化する。
 *
 * @param deferredInputs shape を伴わない入力（常駐テンソル — バイト列と大きさしか持たない
 *   ので、ホスト側に shape が無い）。これらは**束縛源にならず、照合もされない**: 宣言 shape は
 *   他の入力か `seed` で決まった束縛から `planGraph` が解決し、実体の大きさとの突合は
 *   executor が解決済み shape に対して行う。決まらないシンボルは従来どおり fail loudly。
 */
export const bindSymbols = (
  graph: IrGraph,
  inputShapes: Readonly<Record<string, readonly number[]>>,
  seed: SymbolBindings = {},
  deferredInputs?: ReadonlySet<string>,
): SymbolBindings => {
  const symbols = new Set(graph.symbols);
  // MUST: シンボル名を蓄積する器は null プロトタイプ。シンボルの文法（dims.ts の
  // `[A-Za-z_][A-Za-z0-9_]*`）は "__proto__" にマッチし、素の `{}` では代入が
  // [[Prototype]] 設定に化けて own property が作られないため、束縛できたはずのシンボルが
  // 直後の hasOwn 検査で「束縛できなかった」として落ちる（受理集合がエンジンで割れる）。
  // Deno はこの setter を無効化しているため手元では再現しないが、ブラウザ（対象実行系の
  // 一方）では起きる。
  const bindings: Record<string, number> = Object.create(null);
  for (const [sym, value] of Object.entries(seed)) {
    if (!symbols.has(sym)) {
      throw new ExecutionError(
        `束縛 '${sym}' はグラフの symbols [${graph.symbols.join(", ")}] に無い`,
      );
    }
    bindings[sym] = assertExtent(value, `束縛 '${sym}'`);
  }

  const declared = new Set(graph.inputs.map((spec) => spec.name));
  // MUST: 遅延入力も「グラフの入力か」の検査を通す（素通りさせると、名前を間違えた常駐
  // テンソルが黙って無視され、本来の入力は「渡されていない」で落ちる）。
  for (const name of [...Object.keys(inputShapes), ...(deferredInputs ?? [])]) {
    if (!declared.has(name)) throw new ExecutionError(`'${name}' はグラフの入力ではない`);
  }

  for (const spec of graph.inputs) {
    if (deferredInputs?.has(spec.name) === true) continue;
    if (!Object.hasOwn(inputShapes, spec.name)) {
      throw new ExecutionError(`入力 '${spec.name}' が渡されていない`);
    }
    const actual = inputShapes[spec.name];
    if (actual.length !== spec.shape.length) {
      throw new ExecutionError(
        `入力 '${spec.name}': rank ${actual.length} が宣言 [${spec.shape.join(",")}] と違う`,
      );
    }
    spec.shape.forEach((dim, index) => {
      const where = `入力 '${spec.name}' の次元 ${index}`;
      const size = assertExtent(actual[index], where);
      if (typeof dim === "number") return;
      const expr = parseDim(dim);
      if (expr.coeff !== 1 || expr.offset !== 0) return;
      if (!Object.hasOwn(bindings, expr.sym)) {
        bindings[expr.sym] = size;
        return;
      }
      if (bindings[expr.sym] !== size) {
        throw new ExecutionError(
          `${where}: シンボル '${expr.sym}' の束縛が衝突（${bindings[expr.sym]} と ${size}）`,
        );
      }
    });
  }

  for (const sym of graph.symbols) {
    if (!Object.hasOwn(bindings, sym)) {
      throw new ExecutionError(`シンボル '${sym}' を入力 shape から束縛できなかった`);
    }
  }

  // 2 巡目: 数値次元も派生形も含めて宣言 shape を評価し直し、実入力と全一致を要求する。
  for (const spec of graph.inputs) {
    if (deferredInputs?.has(spec.name) === true) continue;
    const resolved = resolveShape(spec.shape, bindings);
    const actual = inputShapes[spec.name];
    if (resolved.some((dim, index) => dim !== actual[index])) {
      throw new ExecutionError(
        `入力 '${spec.name}': shape [${actual.join(",")}] が宣言 [${
          spec.shape.join(",")
        }]（束縛後 [${resolved.join(",")}]）と一致しない`,
      );
    }
  }
  return bindings;
};

/** 束縛済みの shape を全値に配り、各ノードの出力 shape を契約から計算して宣言と照合する。 */
export const planGraph = (graph: IrGraph, bindings: SymbolBindings): GraphPlan => {
  const shapes = new Map<string, readonly number[]>();
  for (const spec of graph.inputs) shapes.set(spec.name, resolveShape(spec.shape, bindings));
  for (const [name, value] of Object.entries(graph.values)) {
    shapes.set(name, resolveShape(value.shape, bindings));
  }

  const nodes = graph.nodes.map((node, index): NodePlan => {
    const where = `nodes[${index}] (${node.op})`;
    const contract = assertNodeContract(node, where);
    const inputShapes = node.ins.map((name) => {
      const shape = shapes.get(name);
      if (shape === undefined) {
        throw new ExecutionError(`${where}: 入力 '${name}' の shape が未解決`);
      }
      return shape;
    });
    const outputName = node.outs[0];
    const declaredShape = shapes.get(outputName);
    if (declaredShape === undefined) {
      throw new ExecutionError(`${where}: 出力 '${outputName}' の宣言が無い`);
    }
    // reshape / expand は「宣言 shape が目標形」、permute は attrs が要る（ADR 0011）。
    // 宣言を渡しても照合は下で必ず行う — 目標形として使う op だけが自明に一致するだけで、
    // 他の op は従来どおり計算 shape と宣言の食い違いで落ちる。
    const computed = computeOutputShape(contract, inputShapes, where, {
      declared: declaredShape,
      attrs: node.attrs,
      bindings,
    });
    if (
      declaredShape.length !== computed.length ||
      declaredShape.some((dim, i) => dim !== computed[i])
    ) {
      throw new ExecutionError(
        `${where}: 出力 '${outputName}' の計算 shape [${computed.join(",")}] が宣言 [${
          declaredShape.join(",")
        }] と一致しない`,
      );
    }
    const dtypes = nodeDtypes(graph, node, contract, where);
    return {
      node,
      contract,
      inputShapes,
      inputDtypes: dtypes.inputs,
      outputName,
      outputShape: computed,
      outputDtype: dtypes.output,
    };
  });

  return { bindings, shapes, nodes };
};
