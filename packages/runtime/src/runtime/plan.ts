/**
 * 実行計画（純関数）: シンボル束縛 → 全値の shape 解決 → ノードごとの出力 shape 評価と
 * 宣言との照合。GPU に触れないのでアダプタ無し環境でも検証できる。
 *
 * 静的形状（ADR 0004）: 全ノードの出力 shape は 1 dispatch も出す前に確定する。ここで
 * 落ちなかったグラフだけが executor のエンコード段に進む。
 */

import { evalDim, parseDim, solveDim } from "../format/dims.ts";
import type { IrDim, IrDtype, IrGraph, IrNode } from "../format/ir.ts";
import {
  assertNodeContract,
  catDim,
  computeOutputShape,
  CONV1D_OP,
  conv1dAttrs,
  EMBEDDING_OP,
  flipDim,
  IO_DTYPES,
  LINEAR_OP,
  type OpContract,
  resolveNodeDtypes,
  sliceAttrs,
  STATE_APPEND_OP,
  stateWindow,
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

/** ノード出力 1 本ぶんの計画（{@link NodePlan.outputs} に**出力 slot 昇順**で並ぶ）。 */
export type NodeOutputPlan = {
  readonly name: string;
  readonly shape: readonly number[];
  /** 出力の意味論 dtype（契約から導き宣言と照合済み — {@link resolveNodeDtypes}）。 */
  readonly dtype: IrDtype;
};

export type NodePlan = {
  readonly node: IrNode;
  readonly contract: OpContract;
  readonly inputShapes: readonly (readonly number[])[];
  /** 入力の意味論 dtype（宣言由来）。カーネルの要素型はここから決まる。 */
  readonly inputDtypes: readonly IrDtype[];
  /**
   * 出力 slot 昇順の計画（`node.outs` と同順・同長 — ADR 0068 決定 1）。
   *
   * MUST: 名前・shape・dtype を別々の列で持たない。3 本の並列配列は「長さが揃っている」
   * という不変条件を型で表せず、slot をずらして読む誤りが shape 検査を素通りする。
   */
  readonly outputs: readonly NodeOutputPlan[];
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

/** ノードの入出力 dtype を宣言から集めて契約と照合し、**出力 slot 順の dtype 列**を返す。 */
const nodeDtypes = (
  graph: IrGraph,
  node: IrNode,
  contract: OpContract,
  where: string,
): { readonly inputs: readonly IrDtype[]; readonly outputs: readonly IrDtype[] } => {
  const inputs = node.ins.map((name) => declarationOf(graph, name).dtype);
  const declaredOutputs = node.outs.map((name) => declarationOf(graph, name).dtype);
  return {
    inputs,
    outputs: resolveNodeDtypes(contract, node, inputs, declaredOutputs, where),
  };
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
  assertStateOrder(graph);
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

/** 1 スロットに触れたノード 1 本ぶんの記録（{@link assertStateOrder}）。 */
type StateTouch = {
  readonly index: number;
  readonly op: string;
  readonly appends: boolean;
  readonly window: number | undefined;
};

/**
 * state effect の順序（ADR 0067 決定 5b の②）。state 参照は**テンソルのデータ辺を張らない**
 * ため DAG のトポロジ順では順序が決まらず、契約は `nodes` **配列順**そのもの。束縛に依存
 * しないので Session 構築時に 1 回、スロットごとに 3 点を見る:
 *
 * 1. `state_append` は 1 スロットにつき**ちょうど 1 本**。2 本以上は 1 step に 2 回書く形で、
 *    ring の位置式が二重に進み読者が見る過去が step の途中で変わる。0 本は誰もスロットへ
 *    書かないまま `advance` で論理長だけ進む形で、読者が未初期化（ゼロ）行を過去として読み、
 *    2 step 目以降が沈黙で誤値を返す。KV 共有層（ADR 0067 決定 5 の「`state_append` ノードが
 *    無い層」）は**スロット単位で 1 本の append を複数の読者が共有する**形なので、スロット
 *    単位のこの検査と両立する — 弾かれるのはスロット全体に書き手が居ない形だけ
 * 2. append が在るなら**そのスロットに触れる最後のノード**（append より後に読者が居ると、
 *    その読者は「今 step の k/v を過去として二重に読む」— 第 3 / 4 巡 high 指摘の閉鎖）
 * 3. 同一スロットに触れる全ノードの `window` は**存在有無も値も一致**（論理 col → 物理 row の
 *    写像は読み書き同式 MUST — ADR 0067 決定 4。読み側だけ別式にすると沈黙誤読）
 *
 * MUST: fail loudly。3 点とも「順序 / 宣言の誤り」が例外ではなく**別の値**として出る種類の
 * 破れなので、実行前のここでしか止められない。
 */
const assertStateOrder = (graph: IrGraph): void => {
  const touches = new Map<string, StateTouch[]>();
  graph.nodes.forEach((node, index) => {
    const slots = Object.values(node.states);
    if (slots.length === 0) return;
    // attrs の値域検査は assertNodeContract が済ませている（ここは引き直すだけ）。
    const window = stateWindow(node.attrs, `nodes[${index}] (${node.op})`);
    for (const slot of slots) {
      const list = touches.get(slot) ?? [];
      list.push({ index, op: node.op, appends: node.op === STATE_APPEND_OP, window });
      touches.set(slot, list);
    }
  });
  for (const [slot, list] of touches) {
    const appends = list.filter((touch) => touch.appends);
    if (appends.length > 1) {
      throw new ExecutionError(
        `state スロット '${slot}': ${STATE_APPEND_OP} が ${appends.length} 本（nodes[${
          appends.map((touch) => touch.index).join("], nodes[")
        }]）— 1 step に 1 回まで（ADR 0067 決定 5b）`,
      );
    }
    if (appends.length === 0) {
      throw new ExecutionError(
        `state スロット '${slot}': ${STATE_APPEND_OP} が 1 本も無い（読者 nodes[${
          list.map((touch) => touch.index).join("], nodes[")
        }] だけ）— 読者が居るスロットには終端 ${STATE_APPEND_OP} がちょうど 1 本 MUST` +
          `（ADR 0067 決定 5b。書き手が居ないと未初期化の過去を読んで論理長だけ進む）`,
      );
    }
    // ここに来た時点で append はちょうど 1 本（上の 2 枝が 0 本と 2 本以上を落としている）。
    const last = list[list.length - 1];
    if (!last.appends) {
      throw new ExecutionError(
        `state スロット '${slot}': ${STATE_APPEND_OP}（nodes[${
          appends[0].index
        }]）より後に読者 nodes[${last.index}] (${last.op}) が居る` +
          `（append は当該スロットに触れる最後のノード MUST — ADR 0067 決定 5b）`,
      );
    }
    const first = list[0];
    const mismatch = list.find((touch) => touch.window !== first.window);
    if (mismatch !== undefined) {
      const show = (touch: StateTouch): string =>
        `nodes[${touch.index}] (${touch.op}) は ${touch.window ?? "宣言なし"}`;
      throw new ExecutionError(
        `state スロット '${slot}': attrs.window が食い違う（${show(first)} / ${show(mismatch)}）` +
          ` — 論理 col → 物理 row の写像は読み書き同式 MUST（ADR 0067 決定 4）`,
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
 * MUST: `graph.outputs` に載った initializer も適格外にする（IR は initializer 名をそのまま
 * グラフ出力に書くことを許す）。readback は宣言 dtype の semantic f32（4 バイト / 要素）を
 * 仮定して重みバッファから写すため、圧縮のまま常駐すると大半のサイズで copy が実バッファを
 * はみ出して validation で落ち（原因を指さない誤誘導）、極小サイズではゼロ詰めまで収まって
 * ビット列の読み替えが黙って返る。
 */
export const eligibleCompressedInitializers = (graph: IrGraph): ReadonlySet<string> => {
  const initializers = new Set(Object.keys(graph.initializers));
  const eligible = new Set<string>();
  const disqualified = new Set<string>(graph.outputs);
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

/**
 * i4 の**展開経路を持つ** op（ADR 0069 決定 5 と embedding / conv1d 追補）。conv1d は
 * **groups == 1 のときに限り**適格で、その絞り込みは attrs を見ないと決まらないので
 * {@link i4Executable} が担う。エクスポータ側 `karume/emit.py: I4_WEIGHT_OPS` +
 * `_has_i4_kernel` の鏡像 — **集合の中身も絞りの構造も対**（片側だけ集合へ足すと、鏡像を
 * 名乗る 2 定数が別物になり将来の追補で片側だけが更新される）。
 *
 * MUST: conv2d / conv_transpose1d を入れない（展開経路そのものが無い — 生成側でも
 * src/kernels/gemm.ts の conv2d 門と src/kernels/conv1d.ts の direct 門が落とす）。
 */
const I4_WEIGHT_OPS: ReadonlySet<string> = new Set([LINEAR_OP, EMBEDDING_OP, CONV1D_OP]);

/**
 * このノードの重みスロットに i4 の展開経路があるか（ADR 0069 決定 5 と その追補の述語）。
 *
 * conv1d は **`groups == 1` の implicit GEMM だけ**が展開経路を持つ（A タイル充填の group
 * scale — gemm.ts の `fillAConv`）。`groups > 1` は直接カーネルへ流れ、そちらに展開経路は
 * 無い（src/kernels/conv1d.ts）。conv_transpose1d は転置レイアウトの pack が要るので対象外
 * （2026-08-20 ユーザー裁定）。
 *
 * MUST: `groups` は**既定値で補完しない**（`conv1dAttrs` が欠落を落とす — ADR 0015 の
 * 「欄を作った後は既定値補完をしないことだけが担保する」）。ここが黙って 1 を仮定すると、
 * depthwise の重みが i4 で常駐して直接カーネルが packed バイトを f32 として読む。
 */
const i4Executable = (node: IrNode): boolean =>
  I4_WEIGHT_OPS.has(node.op) &&
  (node.op !== CONV1D_OP || conv1dAttrs(node.attrs, `nodes (${node.op})`).groups === 1);

/**
 * 重みスロットでの消費が {@link i4Executable} を満たす op **だけ**の initializer（i4 の適格集合の
 * 狭め — ADR 0069 決定 5。エクスポータ側 `karume/emit.py: i4_eligible_initializers` の鏡像）。
 *
 * MUST: {@link eligibleCompressedInitializers} との**積**で使う — ここは「重みスロットの中で
 * 展開経路の無い op（conv2d / conv_transpose1d / groups > 1 の conv1d）にも食われていないか」
 * だけを見る。共有された重みを常駐させるとそちらのカーネルが packed バイトを f32 として読む
 * （例外は出ない）。
 */
export const i4EligibleInitializers = (graph: IrGraph): ReadonlySet<string> => {
  const executable = new Set<string>();
  const other = new Set<string>();
  for (const node of graph.nodes) {
    const weightSlot = WEIGHT_SLOTS.get(node.op);
    if (weightSlot === undefined) continue;
    const name = node.ins[weightSlot];
    if (name === undefined || !Object.hasOwn(graph.initializers, name)) continue;
    (i4Executable(node) ? executable : other).add(name);
  }
  for (const name of other) executable.delete(name);
  return executable;
};

/**
 * 値名 → グラフ内での消費回数。MUST: `node.ins` の厳密な延べ計数（同じ値を 2 回取れば 2）。
 *
 * NOTE: 多出力ノード（ADR 0068 決定 1）でも数え方は 1 文字も変わらない — 数えるのは**消費側**
 * だけで、定義側が 1 本か複数本かに依らない。出力ごとの `uses` は「その値名の消費回数」を
 * この表から引くだけ（recipe-builder の `#buildStep`）。
 */
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
 * **states 専用記号**（states の shape にしか現れない記号 — KV 容量 `C` 等）の集合。
 *
 * 束縛点は `createGenerationContext(spec.bindings)` の側（ADR 0066 追記 7 の束縛点 2）で、
 * run / enqueue の入力からは原理的に解けない。valid な IR では「入力から解けない記号 =
 * states に現れる記号」が成立する（format/ir.ts の checkSymbolBindability が入力 ∪ states の
 * どちらかへの出現を要求し、states 専用記号の値 shape 出現を拒否している）ので、判定は
 * 「入力 shape の次元位置に現れない」だけでよい。
 */
export const statesOnlySymbols = (graph: IrGraph): ReadonlySet<string> => {
  const fromInputs = new Set<string>();
  for (const spec of graph.inputs) {
    for (const dim of spec.shape) {
      if (typeof dim !== "string") continue;
      fromInputs.add(parseDim(dim).sym);
    }
  }
  return new Set(graph.symbols.filter((sym) => !fromInputs.has(sym)));
};

/**
 * 実入力の shape からシンボルを束縛する。
 *
 * MUST: 束縛は**入力 shape の次元位置**から取る（要素数からの逆算はしない — 複数シンボルが
 * 混ざると解が一意でなく、静かに誤った束縛が通る）。1 次元 1 シンボルの派生形（`2T` /
 * `T+8` 等）は位置ごとに解が一意なので束縛源になる（{@link solveDim} — ADR 0057）。
 * MUST: **states 専用記号は要求しない・seed でも受けない**（ADR 0066 追記 7 の効く範囲の
 * 分担 — 値 shape の解決に効くのは入力由来の束縛だけで、容量の正本は context が持つ）。
 * 受けると「context 生成時の容量」と「run に渡した値」の二重簿記になり、食い違いは値には
 * 出ず計画キャッシュの鍵だけが静かに割れる（同じ計画が別鍵で重複導出される）。
 * MUST: 割り切れない実寸は fail loudly。丸めて受けると、宣言 `2T` に奇数長を渡した run が
 * 「末尾 1 要素だけ意味の違う入力」として最後まで通る。
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
  const statesOnly = statesOnlySymbols(graph);
  for (const [sym, value] of Object.entries(seed)) {
    if (!symbols.has(sym)) {
      throw new ExecutionError(
        `束縛 '${sym}' はグラフの symbols [${graph.symbols.join(", ")}] に無い`,
      );
    }
    if (statesOnly.has(sym)) {
      throw new ExecutionError(
        `束縛 '${sym}' は states 専用記号 — 束縛点は createGenerationContext(spec.bindings)` +
          "（ADR 0066 追記 7）。run / enqueue の bindings では与えられない",
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
      const solved = solveDim(expr, size);
      if (solved === undefined) {
        throw new ExecutionError(
          `${where}: 実測 ${size} が宣言 '${dim}' の形をしていない` +
            `（${expr.coeff} で割り切れる ${expr.offset} 以上の長さが要る）`,
        );
      }
      if (!Object.hasOwn(bindings, expr.sym)) {
        bindings[expr.sym] = solved;
        return;
      }
      if (bindings[expr.sym] !== solved) {
        throw new ExecutionError(
          `${where}: シンボル '${expr.sym}' の束縛が衝突（${bindings[expr.sym]} と ${solved}）`,
        );
      }
    });
  }

  for (const sym of graph.symbols) {
    // states 専用記号は入力から解けないのが正規（束縛点は context 生成 — 上の MUST）。
    if (statesOnly.has(sym)) continue;
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

/**
 * 束縛済みの shape を全値に配り、各ノードの出力 shape を契約から計算して宣言と照合する。
 *
 * @param stateShapes 束縛解決済みの state スロット shape（スロット名 → 容量込みの具体形）。
 *   実体を確保する GenerationContext（ADR 0066 決定 1）が渡す — 省略した計画では state 参照
 *   ノードが fail loudly する（黙って従来形として計算しない）。
 */
export const planGraph = (
  graph: IrGraph,
  bindings: SymbolBindings,
  stateShapes?: ReadonlyMap<string, readonly number[]>,
): GraphPlan => {
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
    const declaredShapes = node.outs.map((name) => {
      const shape = shapes.get(name);
      if (shape === undefined) {
        throw new ExecutionError(`${where}: 出力 '${name}' の宣言が無い`);
      }
      return shape;
    });
    // reshape / expand は「宣言 shape が目標形」、permute は attrs が要る（ADR 0011）。
    // 宣言を渡しても照合は下で必ず行う — 目標形として使う op だけが自明に一致するだけで、
    // 他の op は従来どおり計算 shape と宣言の食い違いで落ちる。
    // NOTE: 目標形を要求する 2 op（reshape / expand）は単一出力なので、{@link ShapeContext} は
    // 宣言 shape を 1 本しか受けない（slot 0）。多出力 op の shape は入力から導く。
    const computed = computeOutputShape(contract, inputShapes, where, {
      declared: declaredShapes[0],
      attrs: node.attrs,
      bindings,
      states: node.states,
      stateShapes,
    });
    // MUST: 列長の一致をここで見る。契約が宣言する出力数（`node.outs` の本数 —
    // assertNodeContract が済ませている）と shape 計算が返した列の長さがずれると、以下の
    // 照合が undefined を触って TypeError になり、どの op のどの slot が欠けたのか出ない。
    if (computed.length !== declaredShapes.length) {
      throw new ExecutionError(
        `${where}: 出力 shape の計算が ${computed.length} 本（宣言は ${declaredShapes.length} 本）`,
      );
    }
    declaredShapes.forEach((declaredShape, slot) => {
      const shape = computed[slot];
      if (
        declaredShape.length !== shape.length ||
        declaredShape.some((dim, i) => dim !== shape[i])
      ) {
        throw new ExecutionError(
          `${where}: 出力 '${node.outs[slot]}' の計算 shape [${shape.join(",")}] が宣言 [${
            declaredShape.join(",")
          }] と一致しない`,
        );
      }
    });
    const dtypes = nodeDtypes(graph, node, contract, where);
    return {
      node,
      contract,
      inputShapes,
      inputDtypes: dtypes.inputs,
      outputs: node.outs.map((name, slot): NodeOutputPlan => ({
        name,
        shape: computed[slot],
        dtype: dtypes.outputs[slot],
      })),
    };
  });

  return { bindings, shapes, nodes };
};
