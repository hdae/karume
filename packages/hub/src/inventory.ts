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
 * MUST: 在庫の問い合わせは**取得元へ委ねる**（`source.ts` ⑥⑦）— キャッシュキーの綴りは取得層の
 * 所有物で、hub が組み立てると取得層の版が上がるたびに「消したつもりで残る」形が生まれる。
 * MUST: 参照の同一性は {@link fileRefKey}（越境参照は別リポの同名 path を別の 1 本として数える）。
 *
 * NOTE: manifest 本体（`karume.json`）のエントリは対象外 — URL キーで小さく、選択の所有物でも
 * ない（丸ごと消すのは `clearHubCache`）。
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
   * `"shared"` = 同じ manifest の他の選択（全ファイルが在庫にあるもの）が参照している /
   * `"cross-repo"` = 越境参照で、実体の持ち主は参照先 repo。
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
 * - 他の選択（同じ manifest の別 model / 別 quant）が**全参照を在庫に持っている**なら、その
 *   選択が使うファイルは残す（`kept: "shared"`）。部分在庫の選択は守らない — どのみち次に
 *   使うとき残りを取りに行くので、守らせると「消せないのに使えないファイル」だけが残る。
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
  const others = allSelections(manifest)
    .filter((other) => other.model !== target.model || other.quant !== target.quant)
    .map((other) => ({ selection: other, refs: refsOf(manifest, other) }));
  // 在庫の問い合わせは**全選択の和集合に対して 1 回**（origin ごと）。選択ごとに引くと、
  // 同じ repo の全列挙を選択の数だけ繰り返すことになる。
  const stock = await queryInventory(
    source,
    uniqueRefs([...targets, ...others.flatMap(({ refs }) => refs)]),
  );
  const isCached = (ref: FileRef): boolean => stock.has(fileRefKey(ref));

  const sharedWith = new Map<string, string[]>();
  for (const other of others) {
    if (!other.refs.every(isCached)) continue;
    for (const ref of other.refs) {
      const key = fileRefKey(ref);
      const labels = sharedWith.get(key);
      if (labels === undefined) sharedWith.set(key, [labelOf(other.selection)]);
      else labels.push(labelOf(other.selection));
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
  return { evicted: evictable.filter((ref) => removed.has(fileRefKey(ref))), kept };
};
