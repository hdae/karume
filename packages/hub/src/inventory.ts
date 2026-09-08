/**
 * キャッシュ在庫の照会（{@link listCachedAssets}）と、**選択単位**の削除
 * （{@link evictCachedAssets}）。`clearHubCache`（名前空間まるごと）の隣に置く細粒度の面で、
 * 単位は取得と同じ「model / quant の 1 組」。
 *
 * 削除が単なる「対象の参照を消す」で済まないのは、同じファイルを複数の選択が共有するため
 * （quant を替えても tokenizer や f16 の text_encoder は同じ 1 本）。素朴に消すと、消していない
 * 選択が黙って部分在庫に落ち、次の起動でその選択だけ再 DL が走る。そこで**参照勘定**を挟む:
 * 他の選択が在庫として成立している（＝全参照が揃っている）なら、その選択が使うファイルは
 * 消さずに残す。
 *
 * ただし**対象と参照集合がまったく同じ選択は守る側に数えない**（例: 重みは同じで session の
 * 計算ノブだけ違う `f16` と `f16-c16`）。キャッシュの粒度では両者を区別できず「片方だけ消す」は
 * 定義上できないので、守る側に数えると 1 本も消えないのに「守った」と名乗る嘘の答えになる。
 * どの選択が巻き添えで部分在庫に落ちたかは {@link EvictedAssets.alsoEvicted} が名乗る。
 * 守る側をアプリが決めたいときは {@link CacheInventoryOptions.protect} で明示する。
 *
 * MUST: 在庫の問い合わせは**取得元へ委ねる**（`source.ts` ⑥⑦）— キャッシュキーの綴りは取得層の
 * 所有物で、hub が組み立てると取得層の版が上がるたびに「消したつもりで残る」形が生まれる。
 * MUST: 参照の同一性は {@link fileRefKey}（越境参照は別リポの同名 path を別の 1 本として数える）。
 *
 * NOTE: manifest 本体（`karume.json`）のエントリは対象外 — URL キーで小さく、選択の所有物でも
 * ない（丸ごと消すのは `clearHubCache`）。
 *
 * ## weights の絞り込み（{@link ResolveOptions.weights}）が勘定に効く形
 *
 * 選択は {@link ResolveOptions} なので、対象にも `protect` にも「weights の部分集合」を渡せる。
 * 効き方は 1 つ — **参照の集合が絞ったぶんだけ小さくなる**（`resolveFiles` の結果がそのまま
 * 勘定の材料）:
 *
 * - 照会（{@link listCachedAssets}）… `cached` / `missing` は絞った参照だけを数える。ある役割
 *   （gemma4 の `drafter`）を落とし済みかを、本体の在庫と独立に問える。
 * - 削除（{@link evictCachedAssets}）… 消す候補が絞った参照だけになる。**同じ (model, quant) の
 *   残りは守る側に数えない**（守る側の候補は「対象と label の違う選択」で、同じ label の
 *   別の部分集合は候補に上がらない）ので、「本体は残して drafter だけ消す」が書ける。
 * - 巻き添え（{@link EvictedAssets.alsoEvicted}）… 数えるのは**他の (model, quant)** だけで、
 *   同じ選択の残りは載らない。同じ label の中で部分在庫になったかどうかは、必要なら
 *   {@link listCachedAssets} をもう一度引いて見る（label は `"<model>/<quant>"` のままで、
 *   部分集合を名乗る欄は持たない）。
 * - ラベル（{@link KeptAsset.sharedWith} / {@link EvictedAssets.alsoEvicted}）は絞っても
 *   `"<model>/<quant>"` のまま。`protect` に同じ label の部分集合を 2 つ並べると**両方が守る**
 *   （同じ label が 2 度出ないよう `sharedWith` は一意化する）。
 */

import { HubError } from "./errors.ts";
import { crossRefOf, type FileRef, fileRefKey, type Manifest } from "./manifest.ts";
import { resolveFiles, type ResolveOptions } from "./resolve.ts";
import { type LoadedManifest, pinnedSourceOf } from "./session.ts";
import { type PinnedSource, sourceForRef } from "./source.ts";

/** 在庫の照会・削除の作法（取得の面と同じ差し替え点）。 */
export type CacheInventoryOptions = {
  /** `CacheStorage` の差し替え（テスト用）。無指定は取得層の既定 = `globalThis.caches`。 */
  readonly caches?: CacheStorage;
  /**
   * 守る側の選択を明示する（**{@link evictCachedAssets} だけが読む** — 在庫の照会に守る側は
   * 無い）。無指定なら manifest の全選択が候補（ただし対象と参照集合が同じものは除く）で、
   * 指定するとこの一覧だけが候補になる。「全参照が在庫にあるものだけが実際に守る」のは
   * どちらも同じ。
   *
   * 指定時は同一集合の除外を**しない** — 兄弟の選択を名指しで守れば `kept: "shared"` になる。
   * 「同一集合をどう扱うか」の方針をアプリ側に残すための席。対象自身が入っていても無視する
   * （= 同じ (model, quant) は `weights` で絞っていても候補にならない）。存在しない
   * model / quant / weights は `ManifestReferenceError`。
   */
  readonly protect?: readonly ResolveOptions[];
};

/** {@link listCachedAssets} の結果。合わせると選択の全参照（`fileRefKey` で一意化済み）になる。 */
export type CachedAssets = {
  /** 在庫にある参照（`resolveFiles` の順）。 */
  readonly cached: readonly FileRef[];
  /** 在庫に無い参照（`resolveFiles` の順）。 */
  readonly missing: readonly FileRef[];
};

/** 削除を見送った参照 1 本と、その理由。 */
export type KeptAsset = {
  readonly ref: FileRef;
  /**
   * `"shared"` = 守る側の選択（全ファイルが在庫にあるもの）が参照している /
   * `"cross-repo"` = 越境参照で、実体の持ち主は参照先 repo。
   *
   * 守る側の候補は既定では「同じ manifest の他の選択のうち、対象と参照集合が同一でないもの」、
   * {@link CacheInventoryOptions.protect} 指定時はその一覧だけ。
   */
  readonly reason: "shared" | "cross-repo";
  /** `reason` が `"shared"` のとき、守っている選択のラベル `"<model>/<quant>"` の一覧。 */
  readonly sharedWith: readonly string[];
};

/** {@link evictCachedAssets} の結果。もともと在庫に無い参照はどちらにも載らない。 */
export type EvictedAssets = {
  /** 実際に在庫から消えた参照（`resolveFiles` の順）。 */
  readonly evicted: readonly FileRef[];
  /** 消さなかった参照と理由（`resolveFiles` の順）。 */
  readonly kept: readonly KeptAsset[];
  /**
   * この削除で**巻き添えに部分在庫へ落ちた選択**のラベル `"<model>/<quant>"`（manifest の
   * 宣言順）。対象を除く manifest の全選択のうち、呼び出し前は全参照が在庫にあり、かつ
   * {@link EvictedAssets.evicted} の 1 本以上を使っていたもの。
   *
   * 候補ではなく**実際に消えた参照**で決まる（`evicted` と同じ契約 — 取得元が消したと名乗った
   * ものだけ）。既定では対象と同一集合の兄弟がここに載り、`protect` 指定時は守らなかった
   * 全在庫の選択が載る。アプリは「落とし済み」表示をこの一覧ぶん取り下げればよい。
   */
  readonly alsoEvicted: readonly string[];
};

/** (model, quant) の実名 1 組（既定を解決した後の名前）。 */
type Selection = { readonly model: string; readonly quant: string };

const labelOf = (selection: Selection): string => `${selection.model}/${selection.quant}`;

/**
 * `fileRefKey` で一意化する（`fetchAssets` と同じ規則・同じ順）。同じ path を指すキーが複数
 * あっても、在庫の 1 本は 1 本。
 */
const uniqueRefs = (refs: Iterable<FileRef>): readonly FileRef[] => {
  const unique = new Map<string, FileRef>();
  for (const ref of refs) {
    const key = fileRefKey(ref);
    if (!unique.has(key)) unique.set(key, ref);
  }
  return [...unique.values()];
};

/** 選択 1 つぶんの参照列。存在しない model / quant はここで `ManifestReferenceError`。 */
const refsOf = (manifest: Manifest, selection: ResolveOptions): readonly FileRef[] =>
  uniqueRefs(Object.values(resolveFiles(manifest, selection)));

/** manifest の全 (model, quant) の組。 */
const allSelections = (manifest: Manifest): readonly Selection[] =>
  Object.entries(manifest.models).flatMap(([model, entry]) =>
    Object.keys(entry.quants).map((quant): Selection => ({ model, quant }))
  );

/**
 * 選択を実名へ正規化する。実在検査は {@link resolveFiles} が持つので、**resolveFiles を通した
 * 後にだけ**呼ぶ（診断の正本を 2 か所に増やさない）。
 */
const namedSelection = (manifest: Manifest, selection: ResolveOptions): Selection => {
  const model = selection.model ?? manifest.defaultModel;
  if (!Object.hasOwn(manifest.models, model)) {
    throw new Error(`hub: model '${model}' が resolveFiles 通過後に引けない（不変条件破れ）`);
  }
  return { model, quant: selection.quant ?? manifest.models[model].defaultQuant };
};

/** 選択 1 つと、その参照列（在庫の突合と守る側の判定で 1 組にして持ち回る）。 */
type SelectionRefs = { readonly selection: Selection; readonly refs: readonly FileRef[] };

/**
 * 2 つの参照列が**同じ集合**か（順は問わない）。どちらも {@link uniqueRefs} 済みなので、本数が
 * 同じで片側が全部含まれていれば集合として等しい。
 */
const sameRefSet = (left: readonly FileRef[], right: readonly FileRef[]): boolean => {
  if (left.length !== right.length) return false;
  const keys = new Set(left.map(fileRefKey));
  return right.every((ref) => keys.has(fileRefKey(ref)));
};

/**
 * `protect` の一覧を守る側の候補へ正規化する。`resolveFiles` を通してから実名化するので、
 * 存在しない model / quant / weights はここで `ManifestReferenceError`。重複は落とし、対象自身も
 * 落とす（対象を自分から守ることはできない）。
 *
 * MUST: 一意化の鍵は label ではなく**label + 参照集合**にする。`weights` で絞った選択は同じ
 * label のまま別の集合を守るので、label だけで畳むと後勝ちで片方の指定が黙って消える
 * （「守ったはずのファイルが消えている」= 診断の出ない取り違え）。
 */
const protectorsOf = (
  manifest: Manifest,
  entries: readonly ResolveOptions[],
  target: Selection,
): readonly SelectionRefs[] => {
  const byKey = new Map<string, SelectionRefs>();
  for (const entry of entries) {
    const refs = refsOf(manifest, entry);
    const selection = namedSelection(manifest, entry);
    if (selection.model === target.model && selection.quant === target.quant) continue;
    // 参照の並びは `resolveFiles` の宣言順なので、同じ部分集合は同じ鍵になる。
    byKey.set(`${labelOf(selection)} ${refs.map(fileRefKey).join(" ")}`, {
      selection,
      refs,
    });
  }
  return [...byKey.values()];
};

/**
 * 参照を origin ごとにまとめて在庫を問い合わせ、在庫にあるものの `fileRefKey` を返す。
 *
 * MUST: origin の同一性は `origin.label`（診断のための表示文字列）ではなく
 * {@link crossRefOf} の (repo, revision) の組で見る — ラベルは取得元ごとに語彙が違う表示欄で、
 * 同じ座標が別ラベルを名乗ることも別の座標が同じラベルを名乗ることもある。
 */
const queryInventory = async (
  source: PinnedSource,
  refs: readonly FileRef[],
): Promise<ReadonlySet<string>> => {
  const groups = new Map<string, FileRef[]>();
  for (const ref of refs) {
    const cross = crossRefOf(ref);
    // 空文字はセッションの取得元（越境の座標は必ず `<owner/name>@<sha>` の形になる）。
    const origin = cross === undefined ? "" : `${cross.repo}@${cross.revision}`;
    const bucket = groups.get(origin);
    if (bucket === undefined) groups.set(origin, [ref]);
    else bucket.push(ref);
  }
  const cached = new Set<string>();
  for (const bucket of groups.values()) {
    const originSource = sourceForRef(source, bucket[0]);
    const inventory = originSource.inventory;
    if (inventory === undefined) {
      // 組み込みの 2 取得元は両方持つので、ここに来るのは取得元契約の破れ（利用者の入力起因
      // ではない）— `HubError` ではなく素の `Error` で落とす。
      throw new Error(
        `hub: 取得元 ${originSource.origin.label} が在庫の照会を持たない（取得元契約の不変条件破れ）`,
      );
    }
    for (const key of await inventory(bucket)) cached.add(key);
  }
  return cached;
};

/**
 * 選択 1 つぶんの参照が「network に出ずに読めるか」を照会する。**取りには行かない**
 * （在庫の問い合わせだけで、足りない分の DL も検証も起こさない）。
 *
 * 使いどころは「この quant は落とし済みか」の表示と、`prefetchAssets` を出す前の判断。
 * **キャッシュを持つ取得元（HF）では** `CacheStorage` が無い環境で全て `missing` になる
 * （取得層の契約どおり — キャッシュは正しさの要件ではなく最適化なので、hub 側で特別扱いしない）。
 * 手元のディレクトリはキャッシュを介さずに読むので、`CacheStorage` の有無に関わらず
 * 「渡された全部が在庫にある」と答える。
 *
 * `Cache.keys()` 未実装のランタイム（Deno 2.8 以前）では取得層が fail loud に throw する
 * （{@link HubError} ではなく素の `Error`）。
 */
export const listCachedAssets = async (
  loaded: LoadedManifest,
  selection: ResolveOptions = {},
  options: CacheInventoryOptions = {},
): Promise<CachedAssets> => {
  const source = pinnedSourceOf(loaded, options);
  const refs = refsOf(loaded.manifest, selection);
  const stock = await queryInventory(source, refs);
  return {
    cached: refs.filter((ref) => stock.has(fileRefKey(ref))),
    missing: refs.filter((ref) => !stock.has(fileRefKey(ref))),
  };
};

/**
 * 選択 1 つぶんの在庫を消して容量を空ける。**他の選択が壊れない範囲でだけ**消す:
 *
 * - 守る側の選択が**全参照を在庫に持っている**なら、その選択が使うファイルは残す
 *   （`kept: "shared"`）。部分在庫の選択は守らない — どのみち次に使うとき残りを取りに行くので、
 *   守らせると「消せないのに使えないファイル」だけが残る。
 * - 守る側の候補は既定では「同じ manifest の他の選択」から**対象と参照集合が同一のものを
 *   除いた**もの。同一集合の選択（重みは同じで session の設定だけ違う quant）はキャッシュの
 *   粒度で区別できず、守る側に数えると 1 本も消えない。`protect` を渡すとその一覧だけが候補に
 *   なり、同一集合の除外もしない（守り方の方針をアプリが決める席）。
 * - 巻き添えで部分在庫に落ちた選択は `alsoEvicted` が名乗る（消えた参照を 1 本以上使っていた、
 *   呼び出し前は全在庫だった選択）。次のロードで足りない分だけ取り直せば復旧する。
 * - 越境参照は残す（`kept: "cross-repo"`）。実体の持ち主は参照先 repo で、参照元の manifest
 *   から消すのは「他人のリポの在庫を、たまたま参照している側の都合で消す」ことになる。
 *   消したいときは参照先 repo の manifest を開いて消す。
 * - もともと在庫に無い参照は結果に載らない（消すも守るも無い）。
 *
 * キャッシュを持たない取得元（手元のディレクトリ）は {@link HubError} で断る — 中身は取得物
 * ではなく利用者の資産なので、hub が消してよいものが 1 つも無い。名前空間ごと消すのは
 * `clearHubCache`。
 *
 * `Cache.keys()` 未実装のランタイム（Deno 2.8 以前）では取得層が fail loud に throw する
 * （{@link HubError} ではなく素の `Error`）。
 */
export const evictCachedAssets = async (
  loaded: LoadedManifest,
  selection: ResolveOptions = {},
  options: CacheInventoryOptions = {},
): Promise<EvictedAssets> => {
  const { manifest } = loaded;
  const source = pinnedSourceOf(loaded, options);
  const targets = refsOf(manifest, selection);
  const evictRefs = source.evict;
  if (evictRefs === undefined) {
    throw new HubError(
      `evictCachedAssets: 取得元 ${source.origin.label} はキャッシュを持たない` +
        `（手元のディレクトリは取得物ではなく利用者の資産 — 消さない）`,
      { available: manifest.available },
    );
  }

  const target = namedSelection(manifest, selection);
  const others: readonly SelectionRefs[] = allSelections(manifest)
    .filter((other) => other.model !== target.model || other.quant !== target.quant)
    .map((other) => ({ selection: other, refs: refsOf(manifest, other) }));
  // 守る側の候補。既定で同一集合の選択を外すのは、キャッシュの粒度では対象と区別できず
  // 「片方だけ消す」が定義上できないため — 守る側に数えると必ず全ファイルが `shared` になり、
  // 1 本も消えないのに「守った」と名乗る嘘になる。真部分集合・上位集合は従来どおり守る。
  const protectors = options.protect === undefined
    ? others.filter((other) => !sameRefSet(other.refs, targets))
    : protectorsOf(manifest, options.protect, target);
  // 在庫の問い合わせは**全選択の和集合に対して 1 回**（origin ごと）。選択ごとに引くと、
  // 同じ repo の全列挙を選択の数だけ繰り返すことになる。守る側は manifest の選択なので、
  // その参照はこの和集合に必ず含まれる。
  const stock = await queryInventory(
    source,
    uniqueRefs([...targets, ...others.flatMap(({ refs }) => refs)]),
  );
  const isCached = (ref: FileRef): boolean => stock.has(fileRefKey(ref));

  const sharedWith = new Map<string, string[]>();
  for (const other of protectors) {
    if (!other.refs.every(isCached)) continue;
    const label = labelOf(other.selection);
    for (const ref of other.refs) {
      const key = fileRefKey(ref);
      const labels = sharedWith.get(key);
      if (labels === undefined) sharedWith.set(key, [label]);
      // 同じ label の部分集合が 2 つ守っていても、名乗る label は 1 つ（`protect` に
      // `weights` 違いを並べたときだけ起きる — 同じ名前を 2 度出しても読み手に何も足さない）。
      else if (!labels.includes(label)) labels.push(label);
    }
  }

  const kept: KeptAsset[] = [];
  const evictable: FileRef[] = [];
  for (const ref of targets) {
    if (!isCached(ref)) continue;
    if (crossRefOf(ref) !== undefined) {
      kept.push({ ref, reason: "cross-repo", sharedWith: [] });
      continue;
    }
    const labels = sharedWith.get(fileRefKey(ref));
    if (labels !== undefined) {
      kept.push({ ref, reason: "shared", sharedWith: labels });
      continue;
    }
    evictable.push(ref);
  }

  // 消えた事実は取得元が名乗る（件数 0 = 誰かが先に消していた）。順は対象の列に揃える。
  const removed = new Set((await evictRefs(evictable)).map(fileRefKey));
  const evicted = evictable.filter((ref) => removed.has(fileRefKey(ref)));
  // 巻き添えの判定も**実際に消えた集合**（= `evicted` そのもの）で行う。候補で数えると、
  // 取得元が消せなかった参照の利用者まで「部分在庫に落ちた」と名乗ることになる。在庫の有無は
  // 呼び出し前の観測（`stock`）で見る。
  const evictedKeys = new Set(evicted.map(fileRefKey));
  const alsoEvicted = others
    .filter((other) =>
      other.refs.every(isCached) && other.refs.some((ref) => evictedKeys.has(fileRefKey(ref)))
    )
    .map((other) => labelOf(other.selection));
  return { evicted, kept, alsoEvicted };
};
