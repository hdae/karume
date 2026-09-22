/**
 * model / quant → **選択結果の構造型**（ADR 0109 決定 6）。`karume/4` の「取得キー → FileRef」の
 * 平坦な表（`<weights>[i]` / `<weights>.<extra>` の綴り規約）は退役し、選択は
 *
 * - 部品名 → コンテナ参照（{@link ResolvedSelection.containers}）
 * - 資産名 → FileRef（{@link ResolvedSelection.assets}）
 *
 * の 2 欄で表す。取得・在庫・進捗が要る平坦な FileRef 列は {@link selectionRefs} が**そこから
 * 導く** — 独立に更新される 2 本目の表を持たない。
 */

import { ManifestReferenceError } from "./errors.ts";
import type { ContainerRef, FileRef, Manifest, ModelEntry } from "./manifest.ts";
import { fileRefKey } from "./manifest.ts";

/**
 * 選択 1 つぶんの結果。`model` / `quant` は**既定を解決した後の実名**なので、呼び手は
 * 「実際にどの席が選ばれたか」を表示や診断にそのまま使える。
 */
export type ResolvedSelection = {
  readonly model: string;
  readonly quant: string;
  /**
   * 部品名 → 容器（manifest の `weights` の宣言順）。{@link ResolveOptions.weights} で
   * 部分集合を頼んだときはその分だけ。
   */
  readonly containers: Readonly<Record<string, ContainerRef>>;
  /** assets 名 → FileRef（quant にも weights の絞り込みにも依らず**常に全数**）。 */
  readonly assets: Readonly<Record<string, FileRef>>;
};

/** {@link resolveSelection} の選択軸。どれも省略すると manifest の既定が使われる。 */
export type ResolveOptions = {
  /** モデル名（省略時は `defaultModel`）。 */
  readonly model?: string;
  /** quant 名（省略時はそのモデルの `defaultQuant`）。 */
  readonly quant?: string;
  /**
   * 取得する weights の**部分集合**（省略時はそのモデルの weights 全数）。
   *
   * 1 つのモデルが「本体だけでも動き、追加の役割を足すこともできる」形（gemma4 の
   * `model` + 投機の `drafter`）を配布形の分割なしに扱うための軸である。使わない役割の
   * コンテナは表に**現れない**ので、DL も進捗の総量も在庫の勘定もその役割ぶんだけ減る。
   *
   * MUST: 未知の weights 名は {@link ManifestReferenceError}（綴り間違いを黙って
   * 「取らない」に畳まない）。同じ名前を 2 度並べるのも拒否する — 部分集合の指定として
   * 意味が無い綴りだから。
   *
   * NOTE: 表の並びは**宣言順**（manifest の `weights` の順）で、この配列の順ではない —
   * 呼び手の並べ方で取得の送出順が動くと、同じ manifest が呼び手ごとに別の順序を見ることになる。
   * NOTE: 空配列は「weights を 1 本も取らない（assets だけ）」— 拒否はしない（部分集合として
   * 定義された値で、要求どおりコンテナの無い選択が返る）。
   */
  readonly weights?: readonly string[];
};

/**
 * 取得する weights 名を**宣言順**で決める（省略時は全数）。
 *
 * MUST: 実在検査はここ 1 本（`quants` の完全写像は parse が保証済みなので、実在しさえすれば
 * dtype ラベルは必ず引ける）。
 */
const selectWeightNames = (
  entry: ModelEntry,
  weights: readonly string[] | undefined,
): readonly string[] => {
  const declared = Object.keys(entry.weights);
  if (weights === undefined) return declared;
  const chosen = new Set<string>();
  for (const name of weights) {
    if (!Object.hasOwn(entry.weights, name)) {
      throw new ManifestReferenceError(
        `weights '${name}' は manifest に無い（利用可能: ${declared.join(" / ")}）`,
        { available: entry.available },
      );
    }
    if (chosen.has(name)) {
      throw new ManifestReferenceError(
        `weights '${name}' が 2 度指定された（取得する weights の部分集合は重複なしで並べる）`,
        { available: entry.available },
      );
    }
    chosen.add(name);
  }
  return declared.filter((name) => chosen.has(name));
};

/**
 * モデルを選ぶ。未知のモデル名は**利用可能な一覧**を添えて落とす（v2 で初めて列挙が機械可読に
 * なった — ADR 0041 §8）。
 *
 * MUST: 既定の解決（`defaultModel` への落ち）はここ 1 か所で、**名前ごと返す** — 呼び手が
 * 同じ式をもう 1 度書くと、既定の決め方を変えたときに「返り値の model 名」と「実際に引いた
 * ModelEntry」が別のモデルを指し得る。
 */
const selectModel = (
  manifest: Manifest,
  model?: string,
): { readonly name: string; readonly entry: ModelEntry } => {
  const name = model ?? manifest.defaultModel;
  if (!Object.hasOwn(manifest.models, name)) {
    throw new ManifestReferenceError(
      `model '${name}' は manifest に無い（利用可能: ${manifest.available.models.join(" / ")}）`,
      { available: manifest.available },
    );
  }
  return { name, entry: manifest.models[name] };
};

/**
 * model と quant を選んで**選択結果**を作る。省略時は `defaultModel` / `defaultQuant`。
 * {@link ResolveOptions.weights} を渡すとその部分集合だけを並べる（assets は常に全数）。
 *
 * 解決可能性（weights 写像の完全性・dtype ラベルの実在）は parse 時に検査済みなので、ここで
 * 新たに落ちるのは「存在しない model / quant / weights 名を指定した」場合だけ。部品名と
 * assets 名は**別の欄**に入るので、同名衝突という失敗形はもう無い（`karume/4` の取得キー表は
 * 3 つの名前空間を 1 枚へ畳んでいたのでその門が要った）。
 */
export const resolveSelection = (
  manifest: Manifest,
  options: ResolveOptions = {},
): ResolvedSelection => {
  const { name: model, entry } = selectModel(manifest, options.model);
  const quant = options.quant ?? entry.defaultQuant;
  if (!Object.hasOwn(entry.quants, quant)) {
    throw new ManifestReferenceError(
      `quant '${quant}' は manifest に無い（利用可能: ${entry.available.quants.join(" / ")}）`,
      { available: entry.available },
    );
  }
  const labels = entry.quants[quant].weights;
  let containers: Readonly<Record<string, ContainerRef>> = {};
  for (const name of selectWeightNames(entry, options.weights)) {
    containers = { ...containers, [name]: entry.weights[name][labels[name]].container };
  }
  // MUST: assets も**新しい表**を組む（parse 済み manifest の表そのものを返さない）— 返り値の
  // 欄へ代入されると、以後の全選択・在庫勘定・evict が汚染された表を見ることになる。
  // 2 欄で寿命の扱いを揃える意味もある（containers は毎回組み直している）。
  return { model, quant, containers, assets: { ...entry.assets } };
};

/**
 * 選択を**取得・在庫・進捗が扱う平坦な FileRef 列**へ畳む。並びは宣言順（コンテナは
 * `weights` の順・その中は part の添字順・最後に assets の順）。
 *
 * MUST: **長さ 0 の part は落とす**（ADR 0109 決定 3）。0 バイトのファイルは取りに行っても
 * 1 バイトも来ず、読み手も `partLength` が 0 を答えた時点で読みに行かない — 列に残すと
 * 進捗の総量とファイル数だけが水増しされる。
 * MUST: 同一性（{@link fileRefKey}）で**一意化する**。同じ実体を 2 つの名前が指す形（別名の
 * assets・複数の部品が共有する資産）は manifest として正当だが、この列を受ける面（
 * `prefetchAssets` / `streamAssets`）は重複を呼び手の誤りとして拒否する。畳むのは表を作る
 * ここ 1 か所で、面ごとに書くと片方だけ畳み忘れる。
 */
export const selectionRefs = (selection: ResolvedSelection): readonly FileRef[] => {
  const unique = new Map<string, FileRef>();
  const put = (ref: FileRef): void => {
    const key = fileRefKey(ref);
    if (!unique.has(key)) unique.set(key, ref);
  };
  for (const name of Object.keys(selection.containers)) {
    for (const ref of selection.containers[name].parts) {
      if (ref.size > 0) put(ref);
    }
  }
  for (const name of Object.keys(selection.assets)) put(selection.assets[name]);
  return [...unique.values()];
};
