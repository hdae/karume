/**
 * ランタイム対応表（実行できる op / 格納 / 転送 dtype）と、グラフとの突合門。
 *
 * 表の実値は契約表（{@link "./contracts.ts"} の `RUNTIME_SUPPORT`）が持ち、ここは**型と突合の
 * 手続きだけ**を置く。グラフ単体で決まる規則（宣言・SSA・語彙）は `format/ir.ts`、宣言 shape と
 * 供給の突合は `format/container/bind.ts` が持つので、この層が見るのは「このランタイムで実行
 * できるか」の 1 点に閉じる。
 */

import { type CodecLayout, codecLayout } from "../format/container/codecs.ts";
import type { IrDtype, IrGraph } from "../format/ir.ts";

/** capability 不足（実行できない op / 意味論 dtype / attrs / 格納）。 */
export class RuntimeSupportError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "RuntimeSupportError";
  }
}

/** op ごとの実行可能条件。op 名だけでは dtype と attrs の差が表せない。 */
export type OpSupport = {
  /**
   * 実行できる意味論 dtype の**和**（op ごとに違う — 契約表 src/ops/contracts.ts が正本）。
   * スロット別契約の op ではこの和が実際の受理より広いので、入力の突合には使わない。
   */
  readonly dtypes: ReadonlySet<IrDtype>;
  /**
   * **入力スロット別**の受理集合（並びは契約のアリティぶん）。uniform 契約では全スロットが
   * 同じ集合になる。
   *
   * MUST: 入力の突合はこちらで行う。和だけで見ると gather / embedding / masked_fill の
   * スロット取り違え（`gather(index, src)` のように値と添字を逆に渡した形）が
   * 「どちらの dtype も和には入っている」として列挙門を素通りする — 契約検査（plan.ts）まで
   * 落ちて初めて 1 件ずつ止まるので、「非対応は全件列挙」の意図が壊れる。
   */
  readonly slotDtypes: readonly ReadonlySet<IrDtype>[];
  /**
   * **出力 slot 別**に現れうる意味論 dtype（契約表の出力 dtype 写像の値域 — 並びは契約が
   * 宣言する出力数ぶん。ADR 0068 決定 1）。
   *
   * MUST: 入力スロット 0 の受理集合で代用しない。比較（f32 → bool）・bool の `sum`（→ i32）・
   * `where`（bool → f32）は入力と出力の dtype が違うため、スロット 0 で突き合わせると
   * **正しいグラフが列挙門で落ちる**。逆に恒等な op では両者が一致するので、専用の欄を
   * 持たせても既存の判定は変わらない。
   */
  readonly outDtypes: readonly ReadonlySet<IrDtype>[];
  /**
   * 実装済みの attr キー（契約表が宣言する必須 attrs と省略可能 attrs〈`optionalAttrs`〉の
   * キーの和）。
   */
  readonly attrKeys: ReadonlySet<string>;
};

/** ランタイムが実行できる op（dtype / attrs 込み）と格納 dtype の対応表。 */
export type RuntimeSupport = {
  readonly ops: ReadonlyMap<string, OpSupport>;
  /**
   * 実行できる格納 dtype。
   *
   * NOTE: f16 は「適格な重みスロットなら圧縮のまま GPU 常駐、それ以外はロード時に CPU で
   * f32 展開」（ADR 0018）で、どちらの経路でも実行できるためこの集合に入る。適格判定の
   * 結果は実行の可否ではなく VRAM の効き方を変えるだけなので、ここでは分岐しない
   * （適格 0MB を沈黙させないのは Session の診断の役目 — ADR 0006）。
   */
  readonly storage: ReadonlySet<CodecLayout>;
  /**
   * グラフ入力として転送できる意味論 dtype。
   *
   * MUST: op 起点の突合とは別軸で要る。実行器はどのノードも消費しない入力も含めて
   * `graph.inputs` を全件転送するため、転送層の制約は「その入力を使うノードがあるか」と
   * 無関係に実在する。この軸が無いと、未使用入力の dtype 違反だけが列挙門を素通りする。
   */
  readonly io: ReadonlySet<IrDtype>;
};

/**
 * ノードが触る値の宣言 dtype。ins / outs が inputs[] か values{} のちょうど 1 箇所で
 * 宣言されることは IR パーサが保証済み（checkDefinitions / checkDeclarations）。
 */
const declaredDtype = (graph: IrGraph, name: string): IrDtype => {
  const input = graph.inputs.find((spec) => spec.name === name);
  return input !== undefined ? input.dtype : graph.values[name].dtype;
};

/**
 * ランタイム対応表と突合する。
 *
 * MUST: op 名だけでなく**意味論 dtype と attrs まで**見る。名前だけの突合は「対応表には
 * あるのに実行時に落ちる」を作る（docs/research の recon §3-9 が名指しした形 — 先行実験で
 * 実バグとして観測されている）。
 * 非対応は**全件列挙して** fail loudly する（1 件ずつ落とすと、対応表を埋める側が
 * 何本足りないのか分からない）。
 *
 * NOTE: 同じ規則を plan.ts の `validateGraphContracts` も見るが層が違う — こちらは
 * 「モデル作者へ capability 不足を一度に列挙する門」、あちらは「GPU 非依存に毎回通る
 * 契約検査」。両者とも src/ops.ts の契約表由来なので規則が割れることはない。
 */
export const assertRuntimeSupport = (graph: IrGraph, support: RuntimeSupport): void => {
  const missingOps = new Set<string>();
  // MUST: dtype 違反は**宣言（値名）単位に重複除去**する。エイリアス入力（`add(h,h)`）や、
  // 定義ノードの outs と消費ノードの ins の両方に現れる値は同じ宣言を何度も踏むため、素朴に
  // 積むと件数 N が「直すべき宣言の本数」より多く出て、列挙の指標としての意味が薄れる。
  const badDtypes = new Map<string, IrDtype>();
  const badAttrs: string[] = [];
  // 転送層の軸。ノード起点の突合より先に見る（宣言順 = inputs が先）。
  for (const spec of graph.inputs) {
    if (!support.io.has(spec.dtype)) badDtypes.set(spec.name, spec.dtype);
  }
  graph.nodes.forEach((node, index) => {
    const op = support.ops.get(node.op);
    if (op === undefined) {
      missingOps.add(node.op);
      return;
    }
    const where = `nodes[${index}] (${node.op})`;
    node.ins.forEach((name, slot) => {
      // 契約より入力が多い形（アリティ違反）は契約検査の担当。ここは列挙門なので、
      // 対応するスロットが無いぶんは和で見て 1 件でも多く拾う。
      const accept = op.slotDtypes[slot] ?? op.dtypes;
      const dtype = declaredDtype(graph, name);
      if (!accept.has(dtype)) badDtypes.set(name, dtype);
    });
    node.outs.forEach((name, slot) => {
      // 出力は契約表の写像の値域を**出力 slot 別に**見る（cast は attrs.to で決まるので
      // 語彙全体）。入力側の集合で代用すると、比較や bool の sum のように dtype が変わる op で
      // 正しいグラフが落ちる。契約より出力が多い形（出力数違反）は入力側と同様に契約検査の
      // 担当で、余ったぶんは全 slot の和で見て 1 件でも多く拾う。
      const accept = op.outDtypes[slot] ??
        new Set(op.outDtypes.flatMap((slotAccept) => [...slotAccept]));
      const dtype = declaredDtype(graph, name);
      if (!accept.has(dtype)) badDtypes.set(name, dtype);
    });
    const unknown = Object.keys(node.attrs).filter((key) => !op.attrKeys.has(key)).sort();
    if (unknown.length > 0) badAttrs.push(`${where}: ${unknown.join(", ")}`);
  });

  // 格納は展開経路（layout）で見る — `ternary` は `int2-off` と同じ i2 経路なので個別の対応は要らない。
  // 共有 initializer は格納を持たない（実行可否は貸し手の常駐重みが決める — 借り手構築時の門）。
  const missingStorage = new Map<CodecLayout, string[]>();
  for (const [name, initializer] of Object.entries(graph.initializers)) {
    if (initializer.shared !== undefined) continue;
    const layout = codecLayout(initializer.storage.codec);
    if (support.storage.has(layout)) continue;
    const users = missingStorage.get(layout) ?? [];
    users.push(name);
    missingStorage.set(layout, users);
  }

  if (
    missingOps.size === 0 && badDtypes.size === 0 && badAttrs.length === 0 &&
    missingStorage.size === 0
  ) return;

  const diagnostics: string[] = [];
  if (missingOps.size > 0) {
    diagnostics.push(`非対応 op (${missingOps.size}): ${[...missingOps].sort().join(", ")}`);
  }
  if (badDtypes.size > 0) {
    const listed = [...badDtypes].map(([name, dtype]) => `値 '${name}': ${dtype}`);
    diagnostics.push(`非対応 意味論 dtype (${badDtypes.size}): ${listed.join(", ")}`);
  }
  if (badAttrs.length > 0) {
    diagnostics.push(`未実装 attrs (${badAttrs.length}): ${badAttrs.join(", ")}`);
  }
  for (const [layout, users] of [...missingStorage].sort((a, b) => a[0].localeCompare(b[0]))) {
    diagnostics.push(`非対応 格納 '${layout}' (${users.length}): ${users.sort().join(", ")}`);
  }
  throw new RuntimeSupportError(`ランタイムの capability 不足 — ${diagnostics.join(" / ")}`);
};
