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
 * 落ちる）、重み shard は Session を組むその瞬間に 1 本ずつ流す。
 *
 * ## 全量面（`from*Assets`）は温存する
 *
 * 取得済みバイト列から組む入口は {@link wholeComponent} で同じ姿（{@link ModelComponent}）に
 * 畳む。Session の構築は全量面 `createSession` のままなので、失敗の帰属も文言も 1 文字も
 * 変わらない（ADR 0070 受入①の契約 — 全量面は `origin` を名乗らない）。
 */

import {
  createSession,
  type GpuContext,
  type KarumeModel,
  type ModelShard,
  type PreparedModel,
  prepareModel,
  type Session,
  type SessionOptions,
} from "@karume/runtime";
import {
  type AssetProgress,
  fetchAssets,
  type FileRef,
  type LoadedManifest,
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
   * Session を 1 本張る。shard 面では**呼ぶたびに**重み shard 列を新しく流す
   * （使い切った列を再利用できないため。全量面は `KarumeModel` からそのまま組む）。
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

/** {@link loadShardComponents} の戻り。 */
export type ShardComponents = {
  /** weights コンポーネントの供給口（渡した `componentKeys` 以外は fail loudly）。 */
  readonly open: ComponentOpener;
  /** コンポーネント以外の資産（extras / assets）のバイト列 — 従来どおり全量で受け取る。 */
  readonly assets: Record<string, Uint8Array<ArrayBuffer>>;
};

/**
 * 重み shard が 1 本も無い列（現行の配布形は全コンポーネントが 1 shard = グラフ shard だけ）。
 *
 * MUST: 呼ぶたびに新しい iterator を返す（使い切った列を使い回すと、2 本目の Session 構築が
 * 「既に done」なのか空列なのか区別できない）。空の `refs` で `streamAssets` を呼ばないのは、
 * 逐次面が空の取得対象を呼び出し側の誤りとして拒否するため。
 */
const noWeightShards: AsyncIterable<ModelShard> = {
  [Symbol.asyncIterator]: () => ({
    next: (): Promise<IteratorResult<ModelShard, undefined>> =>
      Promise.resolve({ done: true, value: undefined }),
  }),
};

/**
 * Session 構築のたびに重み shard を流す列。取得層は相 1 で全 shard を永続キャッシュへ落として
 * あるので、2 回目以降の構築は network に出ずキャッシュから 1 本ずつ読み直す。
 */
const weightShardStream = (
  loaded: LoadedManifest,
  refs: readonly FileRef[],
  options: StreamAssetsOptions,
): AsyncIterable<ModelShard> =>
  refs.length === 0 ? noWeightShards : {
    // MUST: `streamAssets` の呼び出しは iterator を取る**その時**に置く（生成器を 1 本作って
    // 使い回すと 2 本目の Session 構築が空の列を受ける）。
    [Symbol.asyncIterator]: () => streamAssets(loaded, refs, options),
  };

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
 * グラフ shard だけを取って admission を通し、残りの資産（extras / assets）を全量で取る。
 *
 * MUST: グラフ shard は**全コンポーネントぶんを 1 回の `streamAssets`** で流す — 家族ごとに
 * 呼び分けると取得層の同時取得が効かず、直列 DL に落ちる。同じ shard を 2 つのコンポーネントが
 * 共有する manifest は逐次面が重複として落とす（現行の配布形には存在しない）。
 *
 * NOTE: 逐次面は相 1 で渡した ref を全部キャッシュへ落としてから相 2 で 1 本ずつ引き渡すので、
 * 1 本目の admission が走るのは「全コンポーネントのグラフ shard が揃った後」になる。
 * 「実行できないモデルの**重み**を落とさない」という決定 5 の目的は満たす。
 */
export const loadShardComponents = async (
  where: string,
  loaded: LoadedManifest,
  files: ResolvedFiles,
  componentKeys: readonly string[],
  options: StreamAssetsOptions = {},
): Promise<ShardComponents> => {
  const aggregated = aggregateProgress(files, options.onProgress);
  const hubOptions: StreamAssetsOptions = {
    ...options,
    ...(aggregated === undefined ? {} : { onProgress: aggregated }),
  };

  const consumed = new Set<string>();
  const shards = componentKeys.map((key) => componentShards(where, files, key, consumed));

  // グラフ shard は宣言順に届く（相 2 は渡した `refs` の順）ので、位置で引き当てる。
  const prepared: PreparedModel[] = [];
  for await (const asset of streamAssets(loaded, shards.map(([graph]) => graph), hubOptions)) {
    // hub の `StreamedAsset` が runtime の `ModelShard` を**構造的に満たす**ことの門
    // （両パッケージに同時に依存できるのは models だけなので、境界の構造互換はここでしか
    // 型に固定できない）。構造が割れたらこの代入が赤くなる。
    const graphShard: ModelShard = asset;
    prepared.push(prepareModel(graphShard));
  }
  if (prepared.length !== componentKeys.length) {
    // 位置で引き当てるので、本数が合わないまま進むと**別のコンポーネントのグラフ**を配る。
    throw new Error(
      `${where}: グラフ shard が ${prepared.length} 本しか届いていない` +
        `（期待 ${componentKeys.length} 本 — 逐次面の不変条件破れ）`,
    );
  }

  const components = new Map<string, ModelComponent>();
  componentKeys.forEach((key, index) => {
    const model = prepared[index];
    const weights = shards[index].slice(1);
    components.set(key, {
      graph: model.graph,
      createSession: (gpu, sessionOptions = {}) =>
        model.createSession(gpu, weightShardStream(loaded, weights, hubOptions), sessionOptions),
    });
  });

  // 残りは extras（`<weights>.<extra>`）と assets — IR コンテナとは限らないので全量面のまま。
  let rest: ResolvedFiles = {};
  for (const key of Object.keys(files)) {
    if (!consumed.has(key)) rest = { ...rest, [key]: files[key] };
  }
  const assets = Object.keys(rest).length === 0 ? {} : await fetchAssets(loaded, rest, hubOptions);

  return {
    open: (key) => {
      const component = components.get(key);
      if (component === undefined) {
        throw new Error(
          `${where}: コンポーネント '${key}' は取得していない` +
            `（取得済み: ${[...components.keys()].join(" / ")}）`,
        );
      }
      return component;
    },
    assets,
  };
};
