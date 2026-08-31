/**
 * manifest の weights コンポーネントを **2 段境界 + shard 逐次面**へ載せる内部機構
 * （7 家族の `fromPretrained` が共有する 1 本 — ADR 0070 決定 3 / 決定 5）。
 *
 * MUST: barrel には出さない。ロード経路の綴りを揃えるための機構で、利用者が触る面ではない。
 *
 * ## なぜ経路を変えるのか
 *
 * 従来のロードは `fetchAssets` でモデル**全量**をホスト RAM に載せてから `openModel` して
 * いた。そのため ①実行できないモデル（非対応 op / 契約違反）でも重みを全部落とすまで分からず
 * ②ホスト RAM のピークがモデル全量だった。ここでは各コンポーネントの**先頭 shard（= グラフ
 * shard）だけ**を先に取って {@link prepareModel} に通し（capability 門と契約検査はこの時点で
 * 落ちる）、続けて家族側の門（{@link FamilyAdmission}）を通し、重み shard は admission を
 * 通った直後にキャッシュへ落としておき、Session を組むその瞬間にキャッシュから 1 本ずつ流す
 * （ホスト RAM に載るのは常に「今の 1 本」だけ）。
 *
 * MUST: admission を通したら `PreparedModel` は**その場で捨てる**（グラフ shard のバイト列を
 * 握り続けない）。shard 仕様 v2（ADR 0081 決定 1）でグラフ shard はデータ節 0 テンソルに
 * なったので握るコストは MB 級に下がったが、握って得るものも無い — Session はグラフ shard も
 * 含めた列を毎回流し直すので、残すのは `IrGraph`（JSON 由来の純データ）だけ（v2 以前は
 * グラフ shard が実重みを最大 1GiB 含み、anima 4 コンポーネントで 2.4GiB 常駐の実測が
 * この MUST の起点 — ADR 0070 追記 CX-4.2）。
 *
 * ## 全量面（`from*Assets`）は温存する
 *
 * 取得済みバイト列から組む入口は {@link wholeComponent} で同じ姿（{@link ModelComponent}）に
 * 畳む。Session の構築は全量面 `createSession` のままなので、失敗の帰属も文言も 1 文字も
 * 変わらない（ADR 0070 受入①の契約 — 全量面は `origin` を名乗らない）。
 *
 * ただし取得キーが `<役割>[i]` の **shard 分割形**（`resolveFiles` の規約 — 1GiB 超の
 * コンポーネント）で届いた役割だけは、全量面でも shard 逐次面へ流す（{@link
 * assetComponentOpener}）— 「`fromPretrained` で読める配布形は `fromAssets` でも読める」が
 * 全量面の契約（X2-101）。バイト列の連結はしない（shard は独立ヘッダの safetensors 1 本ずつで、
 * 連結しても単一コンテナにはならない）。
 *
 * ## 常駐させない資産（{@link LoadShardOptions.eagerAssets}）
 *
 * weights 以外の資産は既定で全量常駐だが、それが成立しない配布形がある（gemma4 の PLE sidecar =
 * 1 本 758MB × 3・ADR 0085 決定 3 の「触った shard だけ遅延ロード」）。`eagerAssets` を渡した
 * 家族は、並べなかった資産を**参照のまま**（{@link ShardComponents.deferred}）受け取り、要る
 * 1 本を {@link readCachedAsset} でキャッシュから読み直す。取得（prefetch）は重み shard と
 * 同じ 1 回に載るので、進捗の総量も DL の順序も変わらない。
 */

import {
  createSession,
  createSessionFromShards,
  type GpuContext,
  type KarumeModel,
  type ModelShard,
  openModel,
  prepareModel,
  type Session,
  type SessionOptions,
} from "@karume/runtime";
import {
  type AssetProgress,
  fetchAssets,
  type FileRef,
  type LoadedManifest,
  prefetchAssets,
  type ResolvedFiles,
  streamAssets,
  type StreamAssetsOptions,
} from "@karume/hub";

/**
 * グラフ**宣言**を持つもの（`KarumeModel` と {@link ModelComponent} の共通面）。
 *
 * 宣言との突合（入出力の本数・静的次元と `pipelineConfig` の一致）は shard 面でも全量面でも
 * 同じ 1 本で書きたいので、検査 helper はこの面だけを受ける。
 */
export type GraphOwner = {
  readonly graph: KarumeModel["graph"];
};

/** コンポーネント 1 本の「実行前の姿」= グラフ宣言 + Session の入口。 */
export type ModelComponent = GraphOwner & {
  /**
   * Session を 1 本張る。shard 面では**呼ぶたびに** shard 列（先頭のグラフ shard を含む）を
   * 新しく流す（使い切った列を再利用できないため。全量面は `KarumeModel` からそのまま組む）。
   */
  readonly createSession: (gpu: GpuContext, options?: SessionOptions) => Promise<Session>;
};

/**
 * 取得キー → コンポーネントの供給口。全量面は「手元のバイト列を `openModel`」、shard 面は
 * 「prepare 済みの引き当て」で、パイプライン本体はどちらか知らずに同じ順序で組み立てる。
 */
export type ComponentOpener = (key: string) => ModelComponent;

/** 全量面（取得済み 1 本）のコンポーネント。Session は全量面 `createSession` のまま。 */
export const wholeComponent = (model: KarumeModel): ModelComponent => ({
  graph: model.graph,
  createSession: (gpu, options = {}) => createSession(gpu, model, options),
});

/** `<役割>[<添字>]` の添字部分（10 進整数のみ）。 */
const SHARD_INDEX = /^\d+$/;

/**
 * 取得済み資産の中で `<役割>[i]` を `[0]` から**連続する範囲だけ**拾う。1 本も無ければ空
 * （= 素の 1 本の配布形）。
 */
const assetShardKeys = (
  assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>,
  componentKey: string,
): readonly string[] => {
  const keys: string[] = [];
  for (let index = 0; Object.hasOwn(assets, `${componentKey}[${index}]`); index += 1) {
    keys.push(`${componentKey}[${index}]`);
  }
  return keys;
};

/** `<役割>[<添字>]` の形をした取得キーの総数（{@link assetShardKeys} との差が欠番の本数）。 */
const indexedKeyCount = (
  assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>,
  componentKey: string,
): number => {
  const prefix = `${componentKey}[`;
  let count = 0;
  for (const key of Object.keys(assets)) {
    if (!key.startsWith(prefix) || !key.endsWith("]")) continue;
    if (SHARD_INDEX.test(key.slice(prefix.length, -1))) count += 1;
  }
  return count;
};

/**
 * 手元の shard 列を Session 構築のたびに新しい iterator で流す。
 *
 * MUST: 呼ぶたびに新しい iterator を返す（{@link componentShardStream} と同じ理由 — 使い切った
 * 列を使い回すと 2 本目の Session 構築が空の列を受ける）。バイト列は呼び手の Record が持って
 * いるので、ここは参照を並べ直すだけ（全量面はホスト RAM に全量が載っている面 — shard 面の
 * 「今の 1 本だけ」の性質はここでは得られない）。
 */
const assetShardStream = (shards: readonly ModelShard[]): AsyncIterable<ModelShard> => ({
  [Symbol.asyncIterator]: async function* () {
    for (const shard of shards) yield shard;
  },
});

/**
 * 全量面（`from*Assets`）のコンポーネント供給口 — 7 家族が共有する 1 本。
 *
 * 取得キーの形は `resolveFiles` の規約そのままで、2 形とも受ける:
 *
 * - 素の 1 本（`transformer`）… 従来どおり {@link wholeComponent}（全量面 `createSession`・
 *   失敗は `origin` を名乗らない）。
 * - **shard 分割形**（`transformer[0]` / `transformer[1]` / …）… 宣言順（= 添字順）の shard 列を
 *   そのまま {@link createSessionFromShards} へ流す（`fromPretrained` と同じ逐次面）。バイト列は
 *   連結しない — shard は独立ヘッダの safetensors 1 本ずつで、連結しても単一コンテナにならない。
 *   失敗とフェンスは shard 面の綴り（`shard [n] 'transformer[0]'`）で帰属する（帰属先が複数
 *   あるので名乗るのが正しい）。
 *
 * MUST: 添字は `[0]` から欠番なく連続していること・素キーと `[i]` を混ぜないこと。どちらも
 * 取得キーの作り方が壊れている印で、黙って読み飛ばすと遠くの層から「重みが足りない」の形で
 * 落ちる（未対応・想定外は fail loudly）。
 *
 * `buffer` は家族側の資産アクセサ（`assetBuffer`）— 「資産が無い」「bytes が buffer 全体を
 * 占めていない」の文言を家族側に残すため、shard 1 本ずつも同じ門を通す。
 */
export const assetComponentOpener = (
  where: string,
  assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>,
  buffer: (key: string) => ArrayBuffer,
): ComponentOpener =>
(key) => {
  const shardKeys = assetShardKeys(assets, key);
  // 素の 1 本（キーごと無い場合も含む — 「資産 X が無い」は家族側が揃っているキーつきで言う）。
  if (shardKeys.length === 0) return wholeComponent(openModel(buffer(key)));
  if (Object.hasOwn(assets, key)) {
    throw new Error(
      `${where}: 資産 '${key}' が素のキーと shard 分割キー（'${key}[0]'）の両方で届いている` +
        `（どちらか一方 MUST — 揃っているキー: ${Object.keys(assets).join(" / ")}）`,
    );
  }
  const indexed = indexedKeyCount(assets, key);
  if (indexed !== shardKeys.length) {
    throw new Error(
      `${where}: 資産 '${key}' の shard 添字が [0] から連続していない` +
        `（連続しているのは ${shardKeys.length} 本 / 添字つきのキーは ${indexed} 本）` +
        `（揃っているキー: ${Object.keys(assets).join(" / ")}）`,
    );
  }
  const shards = shardKeys.map((shardKey) => {
    // 家族の門（bytes が buffer 全体を占めるか）を shard 1 本ずつにも通す。返る buffer は
    // その view の buffer そのものなので、view を作り直しても写しは 1 バイトも起きない。
    const bytes = buffer(shardKey);
    return { id: shardKey, bytes: new Uint8Array(bytes) } satisfies ModelShard;
  });
  return {
    // MUST: `PreparedModel` は握らず `IrGraph` だけ残す（モジュール doc の MUST と同じ規律 —
    // Session はグラフ shard も含めた列を毎回流し直す）。
    graph: prepareModel(shards[0]).graph,
    createSession: (gpu, options = {}) =>
      createSessionFromShards(gpu, assetShardStream(shards), options),
  };
};

/**
 * 家族側の admission — 「この manifest / このグラフでは、この家族として実行できない」を
 * **重み shard を 1 バイトも取る前に**落とすための席（ADR 0070 決定 5 / CG3-1）。
 *
 * グラフ admission（{@link prepareModel}）の直後・`prefetchAssets` の**前**に 1 度だけ呼ばれ、
 * 手元にあるのは各コンポーネントの `IrGraph`（{@link ComponentOpener} 経由）と、家族が閉包で
 * 持ち込む manifest / 構築オプションだけ。extras / assets のバイト列はまだ無い（あれを待つと
 * 重み prefetch より前という位置が保てない）ので、資産を要る検査は後段に残る。
 *
 * MUST: 家族はここで**自分の門を全部**通し、戻り値（parse 済み config / quant / 開いた
 * コンポーネント）を後段の状態構築へそのまま渡す — 同じ検査を前段と後段に 2 実装持つと、
 * 片方だけ更新された瞬間に「前は通るが後で落ちる」形へ戻る。
 *
 * NOTE: この席は将来「admission の前倒しで extras の取得を並行に始める」拡張（DL スロット
 * 改善）が同居する予定の場所でもある。
 */
export type FamilyAdmission<Admitted> = (open: ComponentOpener) => Admitted | Promise<Admitted>;

/** {@link loadShardComponents} の戻り。 */
export type ShardComponents<Admitted> = {
  /** weights コンポーネントの供給口（渡した `componentKeys` 以外は fail loudly）。 */
  readonly open: ComponentOpener;
  /** コンポーネント以外の資産（extras / assets）のバイト列 — 従来どおり全量で受け取る。 */
  readonly assets: Record<string, Uint8Array<ArrayBuffer>>;
  /**
   * 常駐させずに**参照だけ**返した資産（{@link LoadShardOptions.eagerAssets} で絞った残り）。
   *
   * バイト列は永続キャッシュに落ちている（下の prefetch）ので、家族側は要る 1 本だけを
   * {@link readCachedAsset} で読み直す。
   */
  readonly deferred: ResolvedFiles;
  /** 家族 admission（{@link FamilyAdmission}）が確定させた材料。 */
  readonly admitted: Admitted;
};

/**
 * {@link loadShardComponents} の追加オプション（取得層のオプションはそのまま透過する）。
 */
export type LoadShardOptions = StreamAssetsOptions & {
  /**
   * **全量で受け取る**資産キーの allowlist（省略時は残り全部 = 従来どおり）。
   *
   * MUST: 「常駐させない側」ではなく「常駐させる側」を並べる — 資産の一覧は manifest 次第で
   * 増えるので、除外リストで書くと**新しく増えた資産が黙って全量常駐へ落ちる**。gemma4 の PLE
   * sidecar は 1 本 758MB × 3 で、全量常駐は ADR 0085 決定 3（触った shard だけ遅延ロード）
   * そのものを壊す。
   */
  readonly eagerAssets?: readonly string[];
};

/**
 * Session 構築のたびにコンポーネントの shard 列**全部**（先頭 = グラフ shard）を流す列。
 * ロード時に {@link streamAssets}（グラフ shard）と {@link prefetchAssets}（重み shard）で
 * 全 shard を永続キャッシュへ落としてあるので、この列は network に出ずキャッシュから 1 本ずつ
 * 読み直す。列は空になりえない（{@link componentShards} が 0 本を拒否する）。
 */
const componentShardStream = (
  loaded: LoadedManifest,
  refs: readonly FileRef[],
  options: StreamAssetsOptions,
): AsyncIterable<ModelShard> => ({
  // MUST: `streamAssets` の呼び出しは iterator を取る**その時**に置く（生成器を 1 本作って
  // 使い回すと 2 本目の Session 構築が空の列を受ける）。
  [Symbol.asyncIterator]: () => streamAssets(loaded, refs, options),
});

/**
 * コンポーネント 1 本の shard 列（宣言順 — 先頭がグラフ shard・ADR 0071）を取得キーの表から
 * 引く。キーの綴りは `resolveFiles` の規約そのもの（1 shard なら weights 名・複数なら
 * `<weights>[i]`）。
 */
const componentShards = (
  where: string,
  files: ResolvedFiles,
  key: string,
  consumed: Set<string>,
): readonly FileRef[] => {
  if (Object.hasOwn(files, key)) {
    consumed.add(key);
    return [files[key]];
  }
  const shards: FileRef[] = [];
  for (let index = 0; Object.hasOwn(files, `${key}[${index}]`); index += 1) {
    const shardKey = `${key}[${index}]`;
    consumed.add(shardKey);
    shards.push(files[shardKey]);
  }
  if (shards.length === 0) {
    throw new Error(
      `${where}: コンポーネント '${key}' のファイルが manifest に無い` +
        `（取得キー: ${Object.keys(files).join(" / ")}）`,
    );
  }
  return shards;
};

/**
 * モデル全体で**1 本**の進捗ストリームにする（MUST — 消費者から見える契約）。
 *
 * 取得が「グラフ shard の逐次面 + 残り資産の全量面 + 重み shard の逐次面」へ割れても、呼び手が
 * 見る `total` は `resolveFiles` の size 合計のままで、`loaded` は全ファイルの受信済み合計。
 * per-file の欄（`fileLoaded` / `fileTotal`）と `phase` は取得層のものを素通しする。
 *
 * NOTE: 引き当てのキーは `path` 1 本 — 進捗イベントが運ぶ識別子がそれしかないため（越境参照の
 * 同名 path を区別できないのは公開イベント側の既知の穴 — `docs/backlog.md`）。
 */
const aggregateProgress = (
  files: ResolvedFiles,
  onProgress: ((progress: AssetProgress) => void) | undefined,
): ((progress: AssetProgress) => void) | undefined => {
  if (onProgress === undefined) return undefined;
  const sizes = new Map<string, number>();
  for (const key of Object.keys(files)) sizes.set(files[key].path, files[key].size);
  let total = 0;
  for (const size of sizes.values()) total += size;
  const received = new Map<string, number>();
  return ({ phase, path, fileLoaded, fileTotal }) => {
    received.set(path, fileLoaded);
    let loaded = 0;
    for (const bytes of received.values()) loaded += bytes;
    onProgress({ phase, path, loaded, total, fileLoaded, fileTotal });
  };
};

/**
 * グラフ shard だけを取って admission（グラフ + 家族）を通し、通った後に重み shard を永続
 * キャッシュへ落とし、残りの資産（extras / assets）を全量で取る。
 *
 * MUST: グラフ shard は**全コンポーネントぶんを 1 回の `streamAssets`** で流す — 家族ごとに
 * 呼び分けると取得層の同時取得が効かず、直列 DL に落ちる。同じ shard を 2 つのコンポーネントが
 * 共有する manifest は逐次面が重複として落とす（現行の配布形には存在しない）。
 *
 * MUST: `admit`（{@link FamilyAdmission}）は省略できない席にする — 「実行できないモデルの
 * 重みは 1 バイトも落とさない」（決定 5）は runtime の capability 門だけでは満たせず、家族の
 * 門（pipeline 名 / major・`pipelineConfig` の schema・グラフと config の突合・共有 GPU の
 * feature）まで前段に揃って初めて文面どおりになる。
 *
 * NOTE: 逐次面は相 1 で渡した ref を全部キャッシュへ落としてから相 2 で 1 本ずつ引き渡すので、
 * 1 本目の admission が走るのは「全コンポーネントのグラフ shard が揃った後」になる。
 * 「実行できないモデルの**重み**を落とさない」という決定 5 の目的は満たす。
 */
export const loadShardComponents = async <Admitted>(
  where: string,
  loaded: LoadedManifest,
  files: ResolvedFiles,
  componentKeys: readonly string[],
  admit: FamilyAdmission<Admitted>,
  options: LoadShardOptions = {},
): Promise<ShardComponents<Admitted>> => {
  const { eagerAssets, ...streamOptions } = options;
  const aggregated = aggregateProgress(files, streamOptions.onProgress);
  const hubOptions: StreamAssetsOptions = {
    ...streamOptions,
    ...(aggregated === undefined ? {} : { onProgress: aggregated }),
  };

  const consumed = new Set<string>();
  const shards = componentKeys.map((key) => componentShards(where, files, key, consumed));

  // グラフ shard は宣言順に届く（相 2 は渡した `refs` の順）ので、位置で引き当てる。
  //
  // MUST: 取り出すのは `PreparedModel.graph`（`IrGraph` = JSON 由来の純データ）だけで、
  // `PreparedModel` 自体はこの式を抜けた時点で到達不能にする — 束縛に残すとグラフ shard の
  // バイト列がパイプラインの寿命いっぱい常駐する（モジュール doc の MUST）。
  const graphs: GraphOwner["graph"][] = [];
  for await (const asset of streamAssets(loaded, shards.map(([graph]) => graph), hubOptions)) {
    // hub の `StreamedAsset` が runtime の `ModelShard` を**構造的に満たす**ことの門
    // （両パッケージに同時に依存できるのは models だけなので、境界の構造互換はここでしか
    // 型に固定できない）。構造が割れたらこの代入が赤くなる。
    const graphShard: ModelShard = asset;
    graphs.push(prepareModel(graphShard).graph);
  }
  if (graphs.length !== componentKeys.length) {
    // 位置で引き当てるので、本数が合わないまま進むと**別のコンポーネントのグラフ**を配る。
    throw new Error(
      `${where}: グラフ shard が ${graphs.length} 本しか届いていない` +
        `（期待 ${componentKeys.length} 本 — 逐次面の不変条件破れ）`,
    );
  }

  // Session 構築時の相 2 は prefetch 済みキャッシュの読み直しなので、**進捗は流さない** —
  // 流すと `complete` がロード完了の後にもう一度出て、集約 `loaded` が二重計上になる
  // （「ロードが終わったのに進捗が動く」列になる）。
  //
  // MUST: `signal` も落とす — 呼び手が渡す signal は「このロード 1 回」の寿命を表す値で、
  // Session 構築面へ持ち越すと `AbortSignal.timeout(120_000)` やアンマウント時の `abort()` が
  // 「ロードは成功したのに以後の生成が全部落ちる」形になる（相 2 は shard ごとに
  // `throwIfAborted()` を踏むので、キャッシュ完備でも確実に落ちる）。`headers` / `fetch` /
  // `caches` / `onCacheError` は「取得の道具」なので寿命いっぱい持つのが正しく、`signal` だけが
  // 別種の値。ロード**中**の中断は `hubOptions` 側が従来どおり担う。
  const { onProgress: _loadProgress, signal: _loadSignal, ...sessionStreamOptions } = hubOptions;

  const components = new Map<string, ModelComponent>();
  componentKeys.forEach((key, index) => {
    const componentRefs = shards[index];
    components.set(key, {
      graph: graphs[index],
      createSession: (gpu, sessionOptions = {}) =>
        // MUST: 流すのは shard 列**全部**（先頭のグラフ shard を含む）— admission 済みの
        // `PreparedModel` を握らない代わりに、構築のたびにグラフ shard から組み直す。
        createSessionFromShards(
          gpu,
          componentShardStream(loaded, componentRefs, sessionStreamOptions),
          sessionOptions,
        ),
    });
  });

  const open: ComponentOpener = (key) => {
    const component = components.get(key);
    if (component === undefined) {
      throw new Error(
        `${where}: コンポーネント '${key}' は取得していない` +
          `（取得済み: ${[...components.keys()].join(" / ")}）`,
      );
    }
    return component;
  };

  // 家族 admission — グラフ admission の直後・重み prefetch の前（{@link FamilyAdmission}）。
  // 供給口は下で返すものと**同じ 1 本**を渡す（前段だけ別の開き方をすると、後段が握るのと
  // 別のコンポーネントを検査したことになる）。
  const admitted = await admit(open);

  // 取得を 2 群へ割る。extras（`<weights>.<extra>`）と assets は IR コンテナとは限らないので
  // 全量面のままで、`eagerAssets` を渡した家族だけが並べなかった資産を**参照のまま**受け取る
  // （バイト列は下の prefetch でキャッシュに入る）。
  let rest: ResolvedFiles = {};
  let deferred: ResolvedFiles = {};
  for (const key of Object.keys(files)) {
    if (consumed.has(key)) continue;
    if (eagerAssets !== undefined && !eagerAssets.includes(key)) {
      deferred = { ...deferred, [key]: files[key] };
      continue;
    }
    rest = { ...rest, [key]: files[key] };
  }

  // MUST: 重み shard の prefetch は admission **2 つとも**（グラフ = `prepareModel` / 家族 =
  // 上の `admit`）の後に置く（決定 5 — 実行できないモデルの重みは 1 バイトも落とさない。
  // 文面が無限定なので、家族の門が後段に残っていると実装がこの MUST より狭くなる — CG3-1）。
  // ここで全コンポーネントぶんを 1 回で落とすのは、Session を遅延構築する家族で
  // 「重みの DL が初回実行まで遅れ、ロード進捗にも現れない」形を無くすため
  // （進捗の `total` は元から全ファイルの合計なので、集約は追加の細工なしで整合する）。
  // グラフ shard は上の `streamAssets` の相 1 が既にキャッシュへ落としているので、ここで
  // 落とすのは 2 本目以降だけでよい。**遅延資産も同じ 1 回に載せる** — 常駐させないだけで
  // 「いつか必ず要るバイト列」なので、後回しにすると生成の途中で無進捗の DL が始まる。
  const prefetched = [
    ...shards.flatMap((componentRefs) => componentRefs.slice(1)),
    ...Object.keys(deferred).map((key) => deferred[key]),
  ];
  if (prefetched.length > 0) await prefetchAssets(loaded, prefetched, hubOptions);

  const assets = Object.keys(rest).length === 0 ? {} : await fetchAssets(loaded, rest, hubOptions);

  return { open, assets, deferred, admitted };
};

/**
 * 遅延資産 1 本を**永続キャッシュから**読み直す（{@link ShardComponents.deferred} の相方）。
 *
 * `streamAssets` の相 1 は prefetch 済みなのでキャッシュヒットで済み、ホスト RAM に載るのは
 * その 1 本だけ。呼ぶたびに新しい列を作るのは、家族側の LRU が同じ shard を何度でも読み直す
 * ため（使い切った iterator を持ち回さない）。
 *
 * MUST: 返す `ArrayBuffer` は view が buffer 全体を占めていることを確かめてから渡す
 * （取得層の契約 — 崩れていたら `slice` で写さず落とす。1 本 758MB 級の資産で RAM ピークを
 * 倍にしない）。
 */
export const readCachedAsset = async (
  where: string,
  loaded: LoadedManifest,
  ref: FileRef,
  options: StreamAssetsOptions = {},
): Promise<ArrayBuffer> => {
  for await (const asset of streamAssets(loaded, [ref], options)) {
    const { bytes } = asset;
    if (bytes.byteOffset !== 0 || bytes.byteLength !== bytes.buffer.byteLength) {
      throw new Error(
        `${where}: 資産 '${ref.path}' の bytes が buffer 全体を占めていない` +
          `（byteOffset ${bytes.byteOffset} / byteLength ${bytes.byteLength} /` +
          ` buffer ${bytes.buffer.byteLength}）`,
      );
    }
    return bytes.buffer;
  }
  throw new Error(`${where}: 資産 '${ref.path}' が逐次面から 1 本も届かなかった`);
};
