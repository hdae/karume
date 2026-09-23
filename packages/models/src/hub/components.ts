/**
 * manifest の weights 部品を**コンテナ（`krm`）**から Session へ載せる内部機構
 * （8 系列の `fromPretrained` / `fromAssets` が共有する 1 本 — ADR 0109 決定 2 / 10・ADR 0108 決定 19）。
 *
 * MUST: barrel には出さない。ロード経路の綴りを揃えるための機構で、利用者が触る面ではない。
 *
 * ## 順序（「実行できないモデルの重みは 1 バイトも落とさない」— ADR 0070 決定 5 を継承）
 *
 * 1. 各部品の **part 0（descriptor）だけ**を温めて開く（`openContainer` — 2 文書を manifest の期待値で
 *    照合し、グラフ × 束縛表の合流で**不足 / 余剰 0** を宣言だけで確かめる）。`prepareContainer` の
 *    capability 門もここ。
 * 2. 家族の門（{@link FamilyAdmission}）を通す。手元にあるのは各部品のグラフ宣言と資産の**宣言**
 *    （名前 → 役割・論理長）。重み block と part を共有しない資産（索引・`rope_base` — 書き手の規約
 *    container-v1 §4.2）は `asset(name)` でここでも読んでよい（PLE の索引 × 資産の全件突合がそれ）。
 * 3. 通ったら重みの part（part 1 以降）を永続キャッシュへ落とす（`prefetchAssets` — 進捗・中断・
 *    4 並列）。Session を組むその瞬間に block が part 順に読まれる（hub の取得面 — ホスト RAM に
 *    載るのは seek 型の取得元で block の重ね合わせ、scan 型で part 1 本）。
 * 4. 残りの資産（tokenizer 等 — manifest の `assets`）を全量で取る。
 *
 * ## 部品差し替え席（{@link LoadContainerOptions.components} — ADR 0108 決定 19）
 *
 * 役割 → {@link ComponentSource}（別の `karume/5` リポの同じ役割）で部品を差し替える。admission は
 * **重みを 1 バイトも取る前**に ①グラフ記述の sha256 が manifest の宣言と一致（グラフ契約が同一の
 * 部品だけ）②束縛表の不足 / 余剰 0（`openContainer` が descriptor だけで出す）③家族の門、を通す。
 * 不足も余剰も全件列挙で拒否する（diffusers の `strict=False` は採らない）。
 *
 * ## 全量面（`from*Assets`）
 *
 * 取得済みバイト列から組む入口は {@link assetComponentOpener} で同じ姿（{@link ModelComponent}）に
 * 畳む。部品キーは**単一形 `krm` のバイト列**（`<役割>`）か**分割形の part 列**（`<役割>[i]` —
 * part 0 から添字順）。期待値（2 文書の sha256）は無い — 手元のバイト列は呼び手が保証する。
 * 同期の供給口を返すために**全部品を先に開く**ので、この面では容器を開くのが家族 admission より
 * 先になる（取得面と順序が逆）— 中断は家族の `fromAssets` 入口で先に見る。
 */

import {
  type AssetReader,
  type ContainerInput,
  type GpuContext,
  type KarumeModel,
  openContainer,
  type OpenedContainer,
  prepareContainer,
  type PreparedModel,
  type Session,
  type SessionOptions,
} from "@karume/runtime";
import {
  type AssetProgress,
  type ContainerRef,
  type DistributionSource,
  fetchAssets,
  type FileRef,
  type HubRepoRef,
  type LoadedManifest,
  loadManifest,
  openContainerSource,
  prefetchAssets,
  type ResolvedSelection,
  resolveSelection,
  type StreamAssetsOptions,
} from "@karume/hub";
import { toManifestSource } from "./repo-ref.ts";

/**
 * グラフ**宣言**を持つもの（`KarumeModel` と {@link ModelComponent} の共通面）。
 *
 * 宣言との突合（入出力の本数・静的次元と `pipelineConfig` の一致）は取得面でも全量面でも
 * 同じ 1 本で書きたいので、検査 helper はこの面だけを受ける。
 */
export type GraphOwner = {
  readonly graph: KarumeModel["graph"];
};

/** コンポーネント 1 本の「実行前の姿」= グラフ宣言 + Session の入口 + 容器の資産。 */
export type ModelComponent = GraphOwner & {
  /** Session を 1 本張る。呼ぶたびに block を part 順に読み直す（列は使い回さない）。 */
  readonly createSession: (gpu: GpuContext, options?: SessionOptions) => Promise<Session>;
  /**
   * 容器が宣言する資産名 → 役割（container-v1 §2.2 — runtime は役割を解釈しない）。家族が
   * 「この部品に PLE がある / 無い」を**宣言だけ**で見る口（admission で使える）。
   */
  readonly assets: Readonly<Record<string, string>>;
  /**
   * 資産 1 本の読み口（未宣言は fail loudly）。開くだけでは 1 バイトも取らない — 区間読みは
   * `read(offset, length)` のときだけ（PLE の行読み・`rope_base` の全量読みがこれ）。
   *
   * 呼ぶたびに新しい読み口を返す。未検証の取得元（全量面の part 列・ローカルディレクトリ）では
   * 読み口が block を**寿命ぶん保持**する（runtime の `AssetReader`）ので、家族は読み口を 1 回の
   * 読みより長く握らない（握ると家族側の常駐予算の外でホスト RAM が育つ）。
   */
  readonly asset: (name: string) => AssetReader;
};

/**
 * 役割 → コンポーネントの供給口。全量面は「手元のバイト列を `openContainer`」、取得面は
 * 「開いて admission 済みの引き当て」で、パイプライン本体はどちらか知らずに同じ順序で組み立てる。
 */
export type ComponentOpener = (key: string) => ModelComponent;

/**
 * 部品差し替えの出所（ADR 0108 決定 19 / ADR 0109 決定 10）— 別の `karume/5` リポの**同じ役割**。
 * `model` / `quant` は省略時にそのリポの既定。
 */
export type ComponentSource = {
  readonly source: string | HubRepoRef | DistributionSource;
  readonly model?: string;
  readonly quant?: string;
};

/**
 * 家族側の admission — 「この manifest / このグラフでは、この家族として実行できない」を
 * **重みの part を 1 バイトも取る前に**落とすための席（ADR 0070 決定 5 を継承）。
 *
 * `openContainer` / `prepareContainer`（合流・不足 / 余剰・capability）の直後・重み prefetch の**前**に
 * 1 度だけ呼ばれ、手元にあるのは各部品の `IrGraph` と資産の宣言（{@link ModelComponent.assets} /
 * `asset(name).length`）だけ。資産のバイト列はまだ無い（あれを待つと重み prefetch より前という
 * 位置が保てない）。
 *
 * MUST: 家族はここで**自分の門を全部**通し、戻り値（parse 済み config / quant）と**同じ供給口**を
 * 後段の状態構築へそのまま渡す — 同じ検査を前段と後段に 2 実装持つと、片方だけ更新された瞬間に
 * 「前は通るが後で落ちる」形へ戻る。
 */
export type FamilyAdmission<Admitted> = (open: ComponentOpener) => Admitted | Promise<Admitted>;

/** {@link loadContainerComponents} の戻り。 */
export type ContainerComponents<Admitted> = {
  /** weights 部品の供給口（渡した `componentKeys` 以外は fail loudly）。 */
  readonly open: ComponentOpener;
  /** manifest の `assets`（tokenizer 等）のバイト列 — 全量で受け取る。 */
  readonly assets: Record<string, Uint8Array<ArrayBuffer>>;
  /** 家族 admission（{@link FamilyAdmission}）が確定させた材料。 */
  readonly admitted: Admitted;
};

/** {@link loadContainerComponents} の追加オプション（取得層のオプションはそのまま透過する）。 */
export type LoadContainerOptions = StreamAssetsOptions & {
  /**
   * 部品差し替え席（役割 → 出所）。渡した役割は `componentKeys` の部分集合 MUST（未知の役割は
   * fail loudly — 綴り間違いを黙って「差し替えない」に畳まない）。
   */
  readonly components?: Readonly<Record<string, ComponentSource>>;
};

/** 開いた容器 1 本を {@link ModelComponent} に畳む（取得面 / 全量面が共有する 1 本）。 */
const containerComponent = (opened: OpenedContainer, prepared: PreparedModel): ModelComponent => ({
  graph: prepared.graph,
  createSession: (gpu, options = {}) => prepared.createContainerSession(gpu, options),
  assets: Object.fromEntries(
    Object.entries(opened.model?.assets ?? {}).map(([name, asset]) => [name, asset.role]),
  ),
  asset: (name) => opened.asset(name),
});

/** 開いた部品の表から供給口を作る（開いていない役割は fail loudly）。 */
const openerOf = (
  where: string,
  components: ReadonlyMap<string, ModelComponent>,
): ComponentOpener =>
(key) => {
  const component = components.get(key);
  if (component === undefined) {
    throw new Error(
      `${where}: 部品 '${key}' は開いていない（開いた部品: ${[...components.keys()].join(" / ")}）`,
    );
  }
  return component;
};

/** `<役割>[<添字>]` の添字部分（10 進整数のみ）。 */
const PART_INDEX = /^\d+$/;

/** 全量面の部品 1 本の読み方（単一形のキー 1 本 / 分割形の part キー列）。 */
type ComponentKeyPlan =
  | { readonly kind: "bytes"; readonly key: string }
  | { readonly kind: "parts"; readonly keys: readonly string[] };

/** `<役割>[<添字>]` の形をしたキーの総数（連続本数との差が欠番の本数）。 */
const indexedKeyCount = (keys: readonly string[], componentKey: string): number => {
  const prefix = `${componentKey}[`;
  let count = 0;
  for (const key of keys) {
    if (!key.startsWith(prefix) || !key.endsWith("]")) continue;
    if (PART_INDEX.test(key.slice(prefix.length, -1))) count += 1;
  }
  return count;
};

/**
 * 全量面の部品 1 本を「単一形 1 本」と「分割形の part 列」のどちらで読むか。
 *
 * MUST: 添字は `[0]` から欠番なく連続していること・素キーと `[i]` を混ぜないこと。どちらも
 * キーの作り方が壊れている印で、黙って読み飛ばすと遠くの層から「part が足りない」の形で落ちる。
 */
const planComponentKeys = (
  where: string,
  keys: readonly string[],
  componentKey: string,
): ComponentKeyPlan => {
  const has = (key: string): boolean => keys.includes(key);
  const partKeys: string[] = [];
  for (let index = 0; has(`${componentKey}[${index}]`); index += 1) {
    partKeys.push(`${componentKey}[${index}]`);
  }
  const indexed = indexedKeyCount(keys, componentKey);
  const available = `（揃っているキー: ${keys.join(" / ")}）`;
  if (has(componentKey)) {
    if (indexed > 0) {
      throw new Error(
        `${where}: 部品 '${componentKey}' が単一形のキーと分割形のキー` +
          `（'${componentKey}[0]' 等）の両方で届いている（どちらか一方 MUST — ` +
          `添字つきのキーは ${indexed} 本）${available}`,
      );
    }
    return { kind: "bytes", key: componentKey };
  }
  if (indexed === 0) {
    throw new Error(
      `${where}: 部品 '${componentKey}' の容器が無い` +
        `（単一形なら '${componentKey}'・分割形なら '${componentKey}[0]' から添字順）${available}`,
    );
  }
  if (partKeys.length === 0) {
    throw new Error(
      `${where}: 部品 '${componentKey}' の part 添字が [0] から始まっていない` +
        `（'${componentKey}[0]' が無い / 添字つきのキーは ${indexed} 本）${available}`,
    );
  }
  if (indexed !== partKeys.length) {
    throw new Error(
      `${where}: 部品 '${componentKey}' の part 添字が [0] から連続していない` +
        `（連続しているのは ${partKeys.length} 本 / 添字つきのキーは ${indexed} 本）${available}`,
    );
  }
  return { kind: "parts", keys: partKeys };
};

/**
 * 全量面（`from*Assets`）のコンポーネント供給口 — 8 系列が共有する 1 本。
 *
 * `componentKeys` の部品を**先に全部開く**（`openContainer` は非同期なので、同期の
 * {@link ComponentOpener} を返すには開いてから配るしかない）。`buffer` は家族側の資産アクセサ
 * （「資産が無い」「bytes が buffer 全体を占めていない」の文言を家族側に残すため、part 1 本ずつも
 * 同じ門を通す）。
 */
export const assetComponentOpener = async (
  where: string,
  assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>,
  buffer: (key: string) => ArrayBuffer,
  componentKeys: readonly string[],
): Promise<ComponentOpener> => {
  const keys = Object.keys(assets);
  const components = new Map<string, ModelComponent>();
  for (const key of componentKeys) {
    const plan = planComponentKeys(where, keys, key);
    const input: ContainerInput = plan.kind === "bytes"
      ? { kind: "bytes", bytes: new Uint8Array(buffer(plan.key)) }
      : { kind: "parts", parts: plan.keys.map((partKey) => new Uint8Array(buffer(partKey))) };
    const opened = await openContainer(input);
    components.set(key, containerComponent(opened, prepareContainer(opened, key)));
  }
  return openerOf(where, components);
};

/**
 * モデル全体で**1 本**の進捗ストリームにする（MUST — 消費者から見える契約）。
 *
 * 取得が「descriptor の温め + 重み part の温め + 資産の全量面」へ割れ、差し替え席では取得元も
 * 割れるが、呼び手が見る `total` は取る全ファイルの size 合計で、`loaded` は受信済み合計。
 * per-file の欄（`fileLoaded` / `fileTotal`）と `phase` は取得層のものを素通しする。
 *
 * NOTE: 引き当てのキーは `path` 1 本 — 進捗イベントが運ぶ識別子がそれしかないため（別リポの
 * 同名 path を区別できないのは公開イベント側の既知の穴 — `docs/backlog.md`）。
 */
const aggregateProgress = (
  refs: readonly FileRef[],
  onProgress: ((progress: AssetProgress) => void) | undefined,
): ((progress: AssetProgress) => void) | undefined => {
  if (onProgress === undefined) return undefined;
  const sizes = new Map<string, number>();
  for (const ref of refs) sizes.set(ref.path, ref.size);
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

/** 部品 1 本の出所（差し替え席なら別リポの manifest）。 */
type Seat = {
  readonly key: string;
  readonly loaded: LoadedManifest;
  readonly container: ContainerRef;
};

/** 実体を持つ part（長さ 0 の part は取得の対象外 — hub も取らない）。 */
const nonEmptyParts = (parts: readonly FileRef[]): readonly FileRef[] =>
  parts.filter((part) => part.size > 0);

/**
 * 席ごとに選んだ FileRef を、manifest（取得元）ごとにまとめて温める。同じ manifest の part は
 * 1 回の `prefetchAssets` に載せる — 家族ごと・部品ごとに呼び分けると取得層の同時取得が効かず、
 * 直列 DL に落ちる。
 */
const prefetchSeats = async (
  seats: readonly Seat[],
  select: (seat: Seat) => readonly FileRef[],
  options: StreamAssetsOptions,
): Promise<void> => {
  const bySource = new Map<LoadedManifest, FileRef[]>();
  for (const seat of seats) {
    const refs = bySource.get(seat.loaded) ?? [];
    refs.push(...select(seat));
    bySource.set(seat.loaded, refs);
  }
  for (const [loaded, refs] of bySource) {
    if (refs.length > 0) await prefetchAssets(loaded, refs, options);
  }
};

/**
 * 各部品の descriptor（part 0）だけを取って admission（合流 + capability + 家族）を通し、通った後に
 * 重みの part を永続キャッシュへ落とし、残りの資産（manifest の `assets`）を全量で取る。
 *
 * MUST: `admit`（{@link FamilyAdmission}）は省略できない席にする — 「実行できないモデルの
 * 重みは 1 バイトも落とさない」は runtime の capability 門だけでは満たせず、家族の門
 * （pipeline 名 / major・`pipelineConfig` の schema・グラフと config の突合・共有 GPU の
 * feature）まで前段に揃って初めて文面どおりになる。
 *
 * MUST: 差し替え席（{@link LoadContainerOptions.components}）の検査（役割の実在・グラフ記述の
 * 同一性）は descriptor を取る前に済ませる — 別リポの manifest を読むだけで判定できる。
 */
export const loadContainerComponents = async <Admitted>(
  where: string,
  loaded: LoadedManifest,
  selection: ResolvedSelection,
  componentKeys: readonly string[],
  admit: FamilyAdmission<Admitted>,
  options: LoadContainerOptions = {},
): Promise<ContainerComponents<Admitted>> => {
  const { components: replacements = {}, ...streamOptions } = options;
  for (const key of Object.keys(replacements)) {
    if (!componentKeys.includes(key)) {
      throw new Error(
        `${where}: components の '${key}' はこの系列の部品ではない` +
          `（差し替えられる役割: ${componentKeys.join(" / ")}）`,
      );
    }
  }

  // 席（部品ごとの出所）— 差し替えは別リポの manifest を読んで**同じ役割**の容器を引く。
  const seats: Seat[] = [];
  for (const key of componentKeys) {
    const base = selection.containers[key];
    if (base === undefined) {
      throw new Error(
        `${where}: 部品 '${key}' の容器が manifest に無い` +
          `（選択 ${selection.model} / ${selection.quant} が持つ部品: ${
            Object.keys(selection.containers).join(" / ")
          }）`,
      );
    }
    const replacement = replacements[key];
    if (replacement === undefined) {
      seats.push({ key, loaded, container: base });
      continue;
    }
    const seatWhere = `${where} components['${key}']`;
    const { onProgress: _progress, ...manifestOptions } = streamOptions;
    const other = await loadManifest(
      toManifestSource(replacement.source, seatWhere),
      manifestOptions,
    );
    const container = resolveSelection(other.manifest, {
      ...(replacement.model === undefined ? {} : { model: replacement.model }),
      ...(replacement.quant === undefined ? {} : { quant: replacement.quant }),
      // 未知の役割名は resolveSelection が `ManifestReferenceError`（利用可能な部品つき）で落とす。
      weights: [key],
    }).containers[key];
    if (container === undefined) {
      throw new Error(`${seatWhere}: 参照先の manifest に部品 '${key}' の容器が無い`);
    }
    // グラフ契約の同一性 — 重みを 1 バイトも取る前に、2 つの manifest の宣言だけで見る。
    if (container.descriptor.graph.sha256 !== base.descriptor.graph.sha256) {
      throw new Error(
        `${seatWhere}: グラフ記述が manifest の宣言と違う（差し替えられるのはグラフ記述の sha256 が` +
          ` 同一の部品だけ — 宣言 ${base.descriptor.graph.sha256} / 差し替え ${container.descriptor.graph.sha256}）`,
      );
    }
    seats.push({ key, loaded: other, container });
  }

  const aggregated = aggregateProgress(
    [
      ...seats.flatMap((seat) => nonEmptyParts(seat.container.parts)),
      ...Object.values(selection.assets),
    ],
    streamOptions.onProgress,
  );
  const hubOptions: StreamAssetsOptions = {
    ...streamOptions,
    ...(aggregated === undefined ? {} : { onProgress: aggregated }),
  };
  // Session 構築時の読みはキャッシュ済み part の区間読みなので、**進捗は流さない**（流すと
  // `complete` がロード完了の後にもう一度出て、集約 `loaded` が二重計上になる）。
  //
  // MUST: `signal` も落とす — 呼び手が渡す signal は「このロード 1 回」の寿命を表す値で、
  // 取得面へ持ち越すと `AbortSignal.timeout(120_000)` やアンマウント時の `abort()` が
  // 「ロードは成功したのに以後の生成が全部落ちる」形になる。`headers` / `fetch` / `caches` /
  // `onCacheError` は「取得の道具」なので寿命いっぱい持つのが正しく、`signal` だけが別種の値。
  const { onProgress: _loadProgress, signal: _loadSignal, ...sessionOptions } = hubOptions;

  // 1. descriptor（part 0）だけを温めて開く。合流（不足 / 余剰 0）と capability 門はここで落ちる。
  await prefetchSeats(seats, (seat) => [seat.container.parts[0]], hubOptions);
  const components = new Map<string, ModelComponent>();
  for (const seat of seats) {
    hubOptions.signal?.throwIfAborted();
    const source = openContainerSource(seat.loaded, seat.container, sessionOptions);
    const opened = await openContainer({ kind: "source", source }, seat.container.descriptor);
    // グラフ名 = 役割名（書き手の規約 — container-v1 §12。無ければ prepareContainer が在るグラフを
    // 列挙して落とす）。
    components.set(seat.key, containerComponent(opened, prepareContainer(opened, seat.key)));
  }
  const open = openerOf(where, components);

  // 2. 家族 admission — 合流 / capability の直後・重み prefetch の前。供給口は下で返すものと
  // **同じ 1 本**を渡す（前段だけ別の開き方をすると、後段が握るのと別の部品を検査したことになる）。
  const admitted = await admit(open);

  // 3. 重みの part を落とす（全部品ぶんを取得元ごとに 1 回で — Session を遅延構築する家族で
  // 「重みの DL が初回実行まで遅れ、ロード進捗にも現れない」形を無くす）。
  await prefetchSeats(
    seats,
    (seat) => nonEmptyParts(seat.container.parts.slice(1)),
    hubOptions,
  );

  // 4. 残りの資産（tokenizer 等）は全量面のまま。
  const assets = Object.keys(selection.assets).length === 0
    ? {}
    : await fetchAssets(loaded, selection.assets, hubOptions);

  return { open, assets, admitted };
};
