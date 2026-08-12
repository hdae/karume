// 配布形（safetensors 1 ファイル + __metadata__ 埋め込みのグラフ JSON）の結合検証。
// グラフ単体の規則は ir.ts が済ませている前提で、ここは宣言と実テンソルの突合、および
// ランタイム対応表との突合だけを持つ。

import { type IrDtype, type IrGraph, type IrStorageDtype, parseIrGraph } from "./ir.ts";
import { parseSafetensors, type SafetensorsDtype, type SafetensorsFile } from "./safetensors.ts";

/** グラフ JSON を載せる __metadata__ のキー。 */
export const IR_METADATA_KEY = "karume_ir";

export class ContainerError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ContainerError";
  }
}

export type KarumeModel = {
  readonly graph: IrGraph;
  readonly file: SafetensorsFile;
};

/** op ごとの実行可能条件。op 名だけでは dtype と attrs の差が表せない。 */
export type OpSupport = {
  /**
   * 実行できる意味論 dtype の**和**（op ごとに違う — 契約表 src/ops.ts が正本）。
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
   * **出力**として現れうる意味論 dtype（契約表の出力 dtype 写像の値域）。
   *
   * MUST: 入力スロット 0 の受理集合で代用しない。比較（f32 → bool）・bool の `sum`（→ i32）・
   * `where`（bool → f32）は入力と出力の dtype が違うため、スロット 0 で突き合わせると
   * **正しいグラフが列挙門で落ちる**。逆に恒等な op では両者が一致するので、専用の欄を
   * 持たせても既存の判定は変わらない。
   */
  readonly outDtypes: ReadonlySet<IrDtype>;
  /** 実装済みの attr キー（契約表の attrs スキーマが宣言するキーそのもの）。 */
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
  readonly storage: ReadonlySet<IrStorageDtype>;
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
 * 格納 dtype → safetensors dtype。f32 / f16 / bf16 / i8 は意味論 f32 の符号化で、
 * `i32` だけが生の int32（ADR 0010 の明示的な例外）。
 *
 * NOTE: この表は「宣言と実テンソルの突合」専用で、実行できるかどうかとは別軸
 * （実行可否の正本は {@link RuntimeSupport.storage} の実値 = src/ops.ts の `RUNTIME_SUPPORT`。
 * ここで数え上げると同じ事実を 2 箇所で持つことになり、増えたときに片方だけ腐る）。
 */
const STORAGE_ENCODING: Readonly<Record<IrStorageDtype, SafetensorsDtype>> = {
  f32: "F32",
  f16: "F16",
  bf16: "BF16",
  i8: "I8",
  i32: "I32",
};

export const openModel = (buffer: ArrayBuffer): KarumeModel =>
  openModelFile(parseSafetensors(buffer));

/**
 * 解析済み safetensors からモデルを開く。宣言（グラフ JSON）と実テンソル（ヘッダ）の
 * 突合はこの 1 経路に集約する — 読み方が増えても検証が分岐しないようにするため。
 */
const openModelFile = (file: SafetensorsFile): KarumeModel => {
  const json = file.metadata.get(IR_METADATA_KEY);
  if (json === undefined) {
    throw new ContainerError(`__metadata__.${IR_METADATA_KEY} が無い（Karume モデルではない）`);
  }
  const graph = parseIrGraph(json);

  for (const [name, initializer] of Object.entries(graph.initializers)) {
    const where = `initializer '${name}'`;
    const view = file.tensors.get(initializer.tensor);
    if (view === undefined) {
      throw new ContainerError(`${where}: テンソル '${initializer.tensor}' がファイルに無い`);
    }
    // 意味論 dtype と格納 dtype の組（f32 の符号化語彙 / i32 は生の int32）と数値 shape は
    // parseIrGraph が保証済み（グラフ単体で決まる規則はパーサに一本化 — docs/ir-v1.md）。
    // ここは実テンソルとの突合だけを見る。
    const declared = graph.values[name];
    const expected = STORAGE_ENCODING[initializer.storage.dtype];
    if (view.dtype !== expected) {
      throw new ContainerError(
        `${where}: 格納 dtype '${initializer.storage.dtype}' に対し safetensors 側が ${view.dtype}（${expected} が必要）`,
      );
    }
    if (
      declared.shape.length !== view.shape.length ||
      declared.shape.some((dim, index) => dim !== view.shape[index])
    ) {
      throw new ContainerError(
        `${where}: 宣言 shape [${declared.shape.join(",")}] ≠ 実テンソル [${view.shape.join(",")}]`,
      );
    }
    const scaleKey = initializer.storage.scale;
    if (scaleKey !== undefined) assertScaleTensor(graph, file, name, scaleKey, declared.shape);
  }

  return { graph, file };
};

/**
 * 量子化格納の scale テンソル（ADR 0019）を実ファイルと突き合わせる。
 *
 * MUST: 4 点すべてを見る。scale は IR の値ではなく safetensors の**素のテンソル**なので、
 * 宣言完全性の検査（parseIrGraph）が 1 つも掛からない — ここだけが門になる。
 *
 * 1. 実在（無ければ束縛するバッファが無い）
 * 2. **F32**（scale を f16 のビット列として読むと全チャネルが桁違いの値になる）
 * 3. 重みと**同 rank の keepdim broadcast 形**（各軸は 1 か重みと同値。1 軸だけが
 *    チャネル軸として残る形 — `torch.amax(..., keepdim=True)` の出力そのもの）
 * 4. **実テンソルとの名前衝突が無い**（別の initializer の実体を scale として読むと、
 *    dtype も shape も偶然合う組で沈黙誤値になる）
 *
 * NOTE: 「非 1 の軸が消費側 op のチャネル軸と一致するか」は op を知らないと決まらないので
 * ここでは見ない（GPU 常駐経路の平坦添字が掛かる条件 — src/runtime/executor.ts が見る）。
 */
const assertScaleTensor = (
  graph: IrGraph,
  file: SafetensorsFile,
  name: string,
  scaleKey: string,
  weightShape: readonly (number | string)[],
): void => {
  const where = `initializer '${name}'`;
  const view = file.tensors.get(scaleKey);
  if (view === undefined) {
    throw new ContainerError(`${where}: scale テンソル '${scaleKey}' がファイルに無い`);
  }
  for (const [other, initializer] of Object.entries(graph.initializers)) {
    if (initializer.tensor === scaleKey) {
      throw new ContainerError(
        `${where}: scale テンソル '${scaleKey}' が initializer '${other}' の実体と同じキー`,
      );
    }
  }
  if (view.dtype !== "F32") {
    throw new ContainerError(
      `${where}: scale テンソル '${scaleKey}' が ${view.dtype}（F32 が必要）`,
    );
  }
  if (view.shape.length !== weightShape.length) {
    throw new ContainerError(
      `${where}: scale [${view.shape.join(",")}] の rank が重み [${
        weightShape.join(",")
      }] と違う（keepdim broadcast 形が必要）`,
    );
  }
  const broadcastable = view.shape.every((dim, axis) => dim === 1 || dim === weightShape[axis]);
  if (!broadcastable) {
    throw new ContainerError(
      `${where}: scale [${view.shape.join(",")}] が重み [${
        weightShape.join(",")
      }] へ broadcast できない`,
    );
  }
};

/**
 * ノードが触る値の宣言 dtype。ins / outs が inputs[] か values{} のちょうど 1 箇所で
 * 宣言されることは parseIrGraph が保証済み（checkDefinitions / checkDeclarations）。
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
    for (const name of node.outs) {
      // 出力は契約表の写像の値域で見る（cast は attrs.to で決まるので語彙全体）。
      // 入力側の集合で代用すると、比較や bool の sum のように dtype が変わる op で
      // 正しいグラフが落ちる。
      const dtype = declaredDtype(graph, name);
      if (!op.outDtypes.has(dtype)) badDtypes.set(name, dtype);
    }
    const unknown = Object.keys(node.attrs).filter((key) => !op.attrKeys.has(key)).sort();
    if (unknown.length > 0) badAttrs.push(`${where}: ${unknown.join(", ")}`);
  });

  const missingStorage = new Map<IrStorageDtype, string[]>();
  // group 量子化（w4 の語彙 — ADR 0019 で不採用確定）は実行経路が無い。黙って無視すると
  // group ごとの scale を per-channel として読む沈黙誤値になるので、capability 不足で落とす。
  const groupQuantized: string[] = [];
  for (const [name, initializer] of Object.entries(graph.initializers)) {
    const dtype = initializer.storage.dtype;
    if (!support.storage.has(dtype)) {
      const users = missingStorage.get(dtype) ?? [];
      users.push(name);
      missingStorage.set(dtype, users);
      continue;
    }
    if (initializer.storage.groupSize !== undefined) groupQuantized.push(name);
  }

  if (
    missingOps.size === 0 && badDtypes.size === 0 && badAttrs.length === 0 &&
    missingStorage.size === 0 && groupQuantized.length === 0
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
  for (const [dtype, users] of [...missingStorage].sort((a, b) => a[0].localeCompare(b[0]))) {
    diagnostics.push(`非対応 格納 dtype '${dtype}' (${users.length}): ${users.sort().join(", ")}`);
  }
  if (groupQuantized.length > 0) {
    diagnostics.push(
      `非対応 group 量子化 (${groupQuantized.length}): ${groupQuantized.sort().join(", ")}`,
    );
  }
  throw new ContainerError(`ランタイムの capability 不足 — ${diagnostics.join(" / ")}`);
};
