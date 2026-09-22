/**
 * Gemma 4 の PLE（per-layer embeddings）を**ホスト側で gather** する実装（ADR 0085）。
 *
 * 索引（容器の資産 `ple_index`）の codec と定数は `./ple-index.ts` にあり、このファイルが持つのは
 * **所有者**だけである — block / 行 / 予算の 3 本が 1 つの `residentBytes` を共有する関係は
 * 割れないので、同じ関数の中に置く。
 *
 * ## なぜホストが引くのか
 *
 * PLE は `input_ids` **だけ**を引数に取る純粋な行 lookup で、E2B では i8 35 表 = 2,240MiB。
 * ランタイムは全 initializer に Session 構築時の GPU 常駐席を与える契約なので、グラフに
 * 残す限り lazy にならない（ADR 0085 Context）。グラフから外して `per_layer_inputs[1,M,35,256]`
 * を**通常のグラフ入力**として供給すると、常駐が 3.70 → 1.51 GiB になる。ランタイムの契約は
 * 1 文字も変わらない（ADR 0085 決定 6 — 「pageable initializer」の席は作らない）。
 *
 * 前例は SBV2 の相対位置表（`sbv2/relattn-tables.ts`）と Anima の rope 素表 — どちらも
 * 「モデル固有の入力の作り方」は models 側の知識で、runtime の語彙ではない（ADR 0008）。
 *
 * ## 配布形（モデル容器の資産 — ADR 0109 決定 4）
 *
 * `values`（**token-major** の `[tokens, layers, dim]`）と `scales`（`[tokens, layers]` f32）を
 * それぞれ block 上限以下・**行の倍数**で切った block 列にし、役割 `ple-values` / `ple-scales` の
 * 資産として容器へ入れる。区間読みを要する block は専用 part に単独で置かれる
 * （container-v1 §4.2）ので、1 行の読みが隣の重みを引きずらない。2 表の block 境界は
 * **独立**である（1 行 = 値 8,960 B / scale 140 B なので、同じ 32MiB でも跨ぐ token 数が違う）。
 *
 * ## MUST: 逆量子化は GPU 側 `embedding` とビット一致する
 *
 * グラフに残していれば `embedding` が `f32(i8) × per-row scale` を計算し、その直後の `mul` が
 * `embed_scale`（= `hidden_size_per_layer_input ** 0.5` = 16.0）を掛けていた。ここも**同じ順序**
 * （`(q × scale) × embedScale`）で組む。`embedScale` は 2 冪なので f32 の乗算が厳密で、順序さえ
 * 揃えば一致する（ADR 0085 決定 4）。割れると token 列 parity が割れ、「機能不変であること」の
 * 証明が使えなくなる。門は `packages/models/tests/e2e_gemma4_product_test.ts` の
 * `ple.probe.safetensors` 突合（torch が 35 表経路で計算した値との**厳密一致**）。
 *
 * ## 読み方（block 全量 ↔ 行）と読み口の寿命
 *
 * 索引が指す block の読み口（runtime の `AssetReader`）は `open(MODEL).asset(name)` で開く。
 * 1 回の gather は表ごとに block で束ね、その block に触る**一意行数**で方針を決める:
 *
 * | 状態 | 方針 |
 * | --- | --- |
 * | 既に常駐 | hit（LRU の末尾へ） |
 * | 一意行数 ≥ {@link FULL_BLOCK_ROWS}（block の行数が下回ればその行数）かつ予算に**追い出し無しで**載る | 全量読み + LRU |
 * | それ以外 | 行読み（`read(行 offset, rowBytes)`） |
 *
 * MUST: **読み口を 1 回の読みより長く持たない**。検証済みでない取得元（手元のバイト列 /
 * ローカルディレクトリ）の `AssetReader` は block を 1 度検証して**読み口が生きている間**
 * 保持する契約（runtime の `AssetReader` の doc）なので、読み口を掴み続けると
 * {@link Gemma4PleOptions.maxResidentBytes} の外でホスト RAM が育つ。常駐は
 * 「読み終えた ArrayBuffer」1 本だけで表し、読み口はその場で捨てる。
 *
 * 読み終えた量子化行は最大 {@link ROW_CACHE_CAPACITY} 行まで LRU で再利用する。全量 block を
 * 優先し、既存のホスト RAM 予算の空きだけを使う。予算 0 は行も保持しない
 * （ADR 0085 追記 2026-09-12）。値は同じバイト列を同じ 2 段丸めに掛けるので、決定 4 の
 * ビット一致は経路に依らない。
 *
 * ## gather の順序（hit 先行）と重複 id
 *
 * 1 回の gather は**行キャッシュと常駐 block の hit を先に触る**。未常駐を先に読むと、その
 * 読みの LRU 追い出しが「この gather がまだ触っていない hit」を落とし、同じ gather の中で
 * 読み直しになる。hit を先に触ると LRU の末尾へ回るので、後続の miss の追い出し先が
 * 「この gather で用の済んだ block」側へ寄る。走行中に掴んだ実体はローカル変数が持つので、
 * 途中で追い出されても値は揃う。
 *
 * 同じ id が並ぶ列（prefill の pad 行 id 0 が典型 — 768 行のうち大半が同じ id）は、**最初の
 * 1 位置だけ逆量子化して残りへ f32 バイト列を複写**する。再計算ではなく複写なので 2 段丸めの
 * 結果とビット同一で、`(q × scale) × embedScale` の契約はそのまま保たれる。
 *
 * ## MUST: id 空間を相互照合する（ADR 0085 決定 5）
 *
 * 索引の行数 / 主 embedding の vocab 行数 / 実際に引く id を突き合わせる。ここがずれると
 * **OOB ではなく「別 token の有効な行」**を引く（例外なしで沈黙して壊れる）ので、fail loudly の
 * 門を置く場所はここしかない。索引と容器の資産の突合（block の実在・長さ）は入口の admission
 * （`./ple-index.ts` の `assertGemma4PleAssets`）が**重みを 1 バイトも取る前**に全件列挙する。
 */

import type { AssetReader, Tensor } from "@karume/runtime";
import { ModelInputError } from "../errors.ts";
import { readAssetRange } from "../hub/asset-readers.ts";
import {
  defaultGemma4PleResidentBytes,
  type Gemma4PleBlock,
  gemma4PleBlockBytes,
  gemma4PleBlockOf,
  type Gemma4PleIndex,
  type Gemma4PleTable,
  largestBlockBytes,
  packFactor,
} from "./ple-index.ts";

/**
 * gather 1 回へ透過するノブ。
 *
 * MUST: **best-effort** の契約である — 中断は**段の境目**（行読みの 1 本ごと・全量読みの
 * 直前）でしか見ない。runtime の `AssetReader.read` は signal を受けないので、走り出した
 * 読み 1 本を途中で畳む口は無い。生成側は run の発行前に自分で `signal` を見る
 * （`generation/sequence.ts`）ので、中断の正しさをここへ委ねていない。
 */
export type Gemma4PleReadOptions = {
  /** この gather の中断（生成 1 回ぶんの `signal` がそのまま降りてくる）。 */
  readonly signal?: AbortSignal;
};

export type Gemma4PleOptions = {
  readonly index: Gemma4PleIndex;
  /**
   * 索引が指す block 1 本の読み口を開く（`open(MODEL).asset` そのもの — 呼び手の責務）。
   *
   * MUST: 呼ばれるたびに**新しい読み口**を返してよい（むしろそれが既定である）。この実装は
   * 開いた読み口を 1 回の読みより長く持たない（モジュール doc の MUST）ので、読み口側で
   * block を保持する取得元でもホスト RAM は {@link Gemma4PleOptions.maxResidentBytes} に収まる。
   */
  readonly openBlock: (asset: string) => AssetReader;
  /**
   * 主 embedding の vocab 行数（id 空間の相互照合 — ADR 0085 決定 5）。
   *
   * MUST: 省略可能にしない。索引と主 embedding が別の語彙で焼かれた組み合わせは、
   * shape も dtype も合ったまま**別 token の行**を引く。
   */
  readonly vocabSize: number;
  /**
   * 常駐させる block と量子化行の**ホスト RAM 上限（バイト）**（LRU — ADR 0085 決定 3）。
   *
   * 省略時は {@link defaultGemma4PleResidentBytes}（= 最大 block 2 本ぶん）。`0` は「常駐させ
   * ない」= gather が使い終わった block を即座に落とす形で、正当な指定である（読み直しが毎回
   * 走るのと引き換えに、この索引のホスト RAM がピーク 1 本ぶんで収まる）。
   *
   * MUST: 本数ではなくバイトで受ける。block 幅は資産世代（書き手の block 上限）で変わるので、
   * 「2 本」は世代ごとに違う RAM を意味してしまう（ADR 0085 追記 2026-09-02）。
   */
  readonly maxResidentBytes?: number;
};

/** 遅延ロードの実測（門が「触った block だけ読んだ」を恒真でなく見るための欄）。 */
export type Gemma4PleStats = {
  /** block を**全量**取りに行った回数（キャッシュミスのうち行読みへ倒さなかった数）。 */
  readonly loads: number;
  /**
   * 行読みで引いた区間の本数（{@link Gemma4PleStats.loads} と対）。
   *
   * 「全量読みが起きていない」を恒真でなく見るための欄である — 全量読みが 0 なだけなら
   * 「そもそも引いていない」形でも成り立つので、引いた区間の本数を並べて初めて行読み経路を
   * 踏んだことが言える。1 行は `values` と `scales` の 2 表から引くので、両方を行読みで
   * 引いた行は 2 本数える。
   */
  readonly rowReads: number;
  /** 現在常駐している block 数（2 表の合計）。 */
  readonly resident: number;
  /**
   * 常駐 block と行キャッシュが占めるホスト RAM（{@link Gemma4PleOptions.maxResidentBytes} と
   * 同じ単位）。
   *
   * 全量 block は読み**始めた**時点で予約し、行は読み終えて保持した byte 数を加える。
   * block の予約時に行を追い出すため、両者の合計は既存予算を超えない。読取り中の一時バッファや
   * Map 等の管理用オブジェクトはこの量に含まない。
   */
  readonly residentBytes: number;
};

export type Gemma4Ple = {
  /**
   * token id 列 → `per_layer_inputs` の `[1, ids.length, layers, dim]` f32。
   *
   * prefill の pad 行も**そのまま id を渡す**（ホストは `input_ids` の pad を 0 で埋めるので、
   * ここにも 0 行の PLE が入る）— グラフ内で引いていたときと同じ値になり、pad 行の値契約
   * （ADR 0066 追記 6）が保たれる。
   */
  gather(ids: readonly number[], options?: Gemma4PleReadOptions): Promise<Tensor>;
  stats(): Gemma4PleStats;
  /**
   * 常駐 block を解放する（ホスト RAM で {@link Gemma4PleOptions.maxResidentBytes} ぶん）。
   *
   * MUST: 解放口を持つ。GPU 側の常駐は `Session.dispose` が返すが、ここは**ホスト RAM の
   * キャッシュ**なので、口が無いと「パイプラインを dispose しても常駐ぶんが返らない」
   * （実体を掴む参照を 1 つ残せばプロセス寿命まで残る）。
   *
   * 以後の {@link Gemma4Ple.gather} は fail loudly — 解放済みの実体が黙って読み直しを始めると、
   * 「dispose したのに RAM が戻らない」形が復活する。冪等。
   */
  dispose(): void;
};

/**
 * 全量読みへ倒す一意行数の下限（**実測から置いた暫定値**）。
 *
 * 区間読み 1 本は 0.1〜0.6 ms なので、**この gather 1 回**を見る限り行読みは何行でも全量読み
 * より安い。全量読みが得になるのは「以後の gather の hit のために予算の空きへ載せる」ときだけ
 * で、この閾値は「1 回でこれだけ触る block は以後も触る」という見込みの線である。低くしすぎると
 * 常駐が回り、高くしすぎると常駐が温まらない（ADR 0085 追記 2026-09-07 の `SEEK_FULL_ROWS` を
 * block 単位へ引き継いだもの）。
 *
 * 実際の下限は **block の行数と小さい方**を採る（{@link fullBlockRows}）— block の全行を触る
 * gather では、行読み `rows × 2` 回より全量 1 回の方が必ず安い。この規則が無いと、行数の少ない
 * block（末尾の端数・小さい配布形）は何度触っても常駐にならない。
 */
const FULL_BLOCK_ROWS = 32;

/** その block を全量読みへ倒す一意行数の下限（block が細いときは全行が下限）。 */
const fullBlockRows = (block: Gemma4PleBlock): number =>
  Math.min(FULL_BLOCK_ROWS, block.stop - block.start);

/**
 * 行読みの**同時発行の上限**（1 回の gather が 2 表を通して持つ in-flight の読み本数）。
 *
 * MUST: 上限を置く。取得元によっては **1 read = 1 fd** で（ローカル取得元は read のたびに
 * `Deno.open` する）、上限が無いと 1 回の gather が**引く区間の本数ぶん**の fd を同時に握る。
 * prefill の 768 行 chunk を全て行読みで流す形（常駐予算に空きが無い状態）が現実の上界で、
 * `ulimit -n 1024`（Linux の既定）の残りを丸ごと食う。実測（2026-09-07・上限無しの版）:
 * 768 行でピーク 768 本、1,200 行で `Too many open files (os error 24)`。
 *
 * MUST: 上限は表ごと・block ごとではなく **gather 全体**に掛ける — 分けて掛けると、複数の
 * block に散る prefill で上限が本数ぶん倍になる（fd の枯渇はそこで起きる）。
 */
const ROW_READ_CONCURRENCY = 16;

/** 同じ token の再読取りを省く上限。block 常駐後の予算の空きだけを使う（ADR 0085）。 */
const ROW_CACHE_CAPACITY = 256;

/** 読み終えた行 1 本（量子化バイト列のまま持ち、逆量子化の順序を変えない）。 */
type CachedRow = {
  readonly values: Int8Array<ArrayBuffer> | Uint8Array<ArrayBuffer>;
  readonly scales: Float32Array<ArrayBuffer>;
};

/**
 * 1 表ぶんの引き当て結果（全量 block の view か、行読みで得た 1 行）。
 *
 * `from` は**要素**単位の行の先頭（`values` はバイト・`scales` は f32 の添字）で、
 * {@link createGemma4Ple} の `writeRow` がそのまま受ける形である。
 */
type Supply<T> = {
  readonly array: T;
  readonly from: number;
  /** 行読みで得た（= 自分の buffer を丸ごと占める）なら true。行キャッシュはこれだけを保存する。 */
  readonly owned: boolean;
};

/** 1 表ぶんの計画（全量読みへ倒した block と、行読みへ倒した id）。 */
type TablePlan = {
  readonly table: Gemma4PleTable;
  /** block 添字 → その block を引く id 列。 */
  readonly grouped: Map<number, number[]>;
  /** block 添字 → 全量読み（開始済み）。 */
  readonly full: Map<number, Promise<ArrayBuffer>>;
  /** 行読みへ倒した id → block 添字。 */
  readonly rows: Map<number, number>;
  /** 行読みの結果（id → 自分の buffer を占める 1 行）。 */
  readonly read: Map<number, ArrayBuffer>;
};

/**
 * 索引の指す block を**触ったぶんだけ**読む gather を組む（ADR 0085 決定 3 + 0109 決定 4）。
 *
 * MUST: モジュール副作用ゼロ（この関数を呼ぶまで何も起きない）。
 */
export const createGemma4Ple = (options: Gemma4PleOptions): Gemma4Ple => {
  const { index, openBlock, vocabSize } = options;
  const budget = options.maxResidentBytes ?? defaultGemma4PleResidentBytes(index);
  // 予算は呼び手の `maxResidentPleBytes` そのものなので入力起因（ADR 0107 決定 2）。
  if (!Number.isSafeInteger(budget) || budget < 0) {
    throw new ModelInputError(`maxResidentBytes ${budget} が 0 以上の整数でない`);
  }
  const largest = largestBlockBytes(index);
  // MUST: 「1 本すら載らない予算」は fail loudly。黙って超過すれば予算が意味を失い、黙って
  // 守れば gather が引けない — どちらも呼び手の指定を裏切る。0 は例外で、「常駐させない」
  // という指定として正当（読み終えた block を即座に落とす形）。
  // 資産の寸法と突き合わせるが、拒否しているのは呼び手が渡した予算の値なので入力起因。
  if (budget > 0 && budget < largest) {
    throw new ModelInputError(
      `maxResidentBytes ${budget} が PLE の block 1 本ぶん ${largest} バイトに満たない` +
        `（この索引の block は values ${index.values.blocks.length} 本 /` +
        ` scales ${index.scales.blocks.length} 本 — 常駐させないなら 0 を渡す）`,
    );
  }
  // ① 索引の行数 と ② 主 embedding の vocab 行数（ADR 0085 決定 5 の相互照合）。
  // こちらは焼いた組み合わせの齟齬なので素の `Error` のまま（呼び手は指定を直せない）。
  if (index.tokens !== vocabSize) {
    throw new Error(
      `PLE の索引の行数 ${index.tokens} が主 embedding の vocab 行数 ${vocabSize} と違う` +
        `（別の語彙で焼かれた組み合わせ — 引ける id が食い違ったまま形は合う）`,
    );
  }
  const stride = index.layers * index.dim;
  const factor = packFactor(index);
  const rowValueBytes = index.values.rowBytes;
  /** 行 1 本ぶんの `scales`（f32 × 層数）の**要素数**。 */
  const scaleStride = index.layers;
  const rowBytes = rowValueBytes + index.scales.rowBytes;
  /** 行も量子化された byte 列のまま保持し、逆量子化の順序を変えない。挿入順 = LRU。 */
  const rowCache = new Map<number, CachedRow>();
  /** 挿入順 = LRU（触った block を末尾へ付け替え、予算超過分は先頭から落とす）。 */
  const resident = new Map<
    string,
    { readonly bytes: number; readonly data: Promise<ArrayBuffer> }
  >();
  /** 常駐（= 読みを始めたぶんを含む）の合計バイト。`resident` の増減と必ず対で動かす。 */
  let residentBytes = 0;
  let loads = 0;
  let rowReads = 0;
  let disposed = false;

  const trimRows = (): void => {
    const limit = Math.min(
      ROW_CACHE_CAPACITY,
      Math.floor(Math.max(0, budget - residentBytes) / rowBytes),
    );
    for (const oldest of rowCache.keys()) {
      if (rowCache.size <= limit) break;
      rowCache.delete(oldest);
    }
  };

  const release = (asset: string): void => {
    const entry = resident.get(asset);
    if (entry === undefined) return;
    resident.delete(asset);
    residentBytes -= entry.bytes;
  };

  /**
   * block 1 本を常駐キャッシュから取る（未常駐なら**全量**読む）。
   *
   * MUST: 同期のまま保つ（`resident` の照会と登録の間に await を挟まない）— 挟むと同じ block を
   * 同時に要求した 2 本が揃って miss と判定し、二重読みになる。
   */
  const acquire = (table: Gemma4PleTable, position: number): Promise<ArrayBuffer> => {
    const block = table.blocks[position];
    const cached = resident.get(block.asset);
    if (cached !== undefined) {
      // 参照したので末尾へ付け替える（Map の反復順 = 挿入順）。
      resident.delete(block.asset);
      resident.set(block.asset, cached);
      return cached.data;
    }
    const bytes = gemma4PleBlockBytes(table, block);
    loads += 1;
    // 読み口はこの読み 1 回ぶんだけ生かす（モジュール doc の MUST）。
    const data = readAssetRange(openBlock(block.asset), 0, bytes);
    // MUST: 失敗した取得を常駐させない（次の gather が同じ拒否済み Promise を掴み続ける）。
    data.catch(() => {
      if (resident.get(block.asset)?.data === data) release(block.asset);
    });
    // dispose 中も先行 gather は完了してよいが、所有者のキャッシュへは戻さない。
    if (disposed) return data;
    resident.set(block.asset, { bytes, data });
    residentBytes += bytes;
    // 予算はバイトで測る（本数ではない）。予算 0 では今入れたぶんもここで落ちるが、走行中の
    // gather は解決済みの実体を自分で掴んでいるので値は揃う。
    for (const oldest of resident.keys()) {
      if (residentBytes <= budget) break;
      release(oldest);
    }
    // 全量 block を優先し、行の常駐を既存予算へ上乗せしない。
    trimRows();
    return data;
  };

  /**
   * 行 1 本を `data` の位置 `target` へ逆量子化して書く。
   *
   * MUST: 全量経路と行読み経路が**この 1 実装**を共有する（式を 2 つ持つと、片方だけ丸め点が
   * ずれても値が「だいたい合う」ので気づけない）。
   * MUST: 逆量子化 → embed scale の **2 段**（GPU 側 `embedding` と直後の `mul` と同じ順序・
   * 同じ丸め点）。JS の算術は f64 なので、`Math.fround` で 1 段目を f32 へ落としてから掛ける —
   * f64 のまま 1 度に丸めると subnormal 域で 2 段丸めと結果が割れうる（ADR 0085 決定 4）。
   */
  const writeRow = (
    data: Float32Array,
    target: number,
    values: Int8Array<ArrayBuffer> | Uint8Array<ArrayBuffer>,
    valuesFrom: number,
    scales: Float32Array<ArrayBuffer>,
    scalesFrom: number,
  ): void => {
    let cursor = target;
    for (let layer = 0; layer < index.layers; layer += 1) {
      const scale = scales[scalesFrom + layer];
      const base = valuesFrom + layer * index.dim / factor;
      if (factor === 1) {
        for (let column = 0; column < index.dim; column += 1) {
          data[cursor + column] = Math.fround(values[base + column] * scale) * index.embedScale;
        }
      } else {
        const bits = 8 / factor;
        const mask = (1 << bits) - 1;
        const offset = 1 << (bits - 1);
        for (let column = 0; column < index.dim; column += 1) {
          const value =
            ((values[base + Math.floor(column / factor)] >>> (bits * (column % factor))) & mask) -
            offset;
          data[cursor + column] = Math.fround(value * scale) * index.embedScale;
        }
      }
      cursor += index.dim;
    }
  };

  /**
   * 同じ id が並ぶ残りの位置へ、**計算済みの f32 バイト列を複写**する。
   *
   * 再計算ではなく複写なので、2 段丸めの結果とビット同一であることが構造で保証される。
   */
  const copyDuplicates = (
    data: Float32Array,
    positions: readonly number[],
    first: number,
  ): void => {
    for (let rest = 1; rest < positions.length; rest += 1) {
      data.copyWithin(positions[rest] * stride, first, first + stride);
    }
  };

  /** 引く id を block へ束ねる（方針はまだ決めない）。 */
  const groupTable = (table: Gemma4PleTable, missing: readonly number[]): TablePlan => {
    const grouped = new Map<number, number[]>();
    for (const id of missing) {
      const position = gemma4PleBlockOf(table, id);
      const ids = grouped.get(position);
      if (ids === undefined) grouped.set(position, [id]);
      else ids.push(id);
    }
    return { table, grouped, full: new Map(), rows: new Map(), read: new Map() };
  };

  /**
   * 既に常駐している block を触って LRU の末尾へ回し、`planned` へ計上する。
   *
   * MUST: **2 表ぶんをまとめて先に**通す（後段の全量読みより前）— 表ごとに「触る → 載せる」を
   * 済ませると、先に走った表の全量読みが**もう一方の表の hit**を追い出す（追い出しは表を
   * またぐ 1 本の LRU で起きる）。
   */
  const touchHits = (plan: TablePlan, planned: { value: number }): void => {
    for (const position of plan.grouped.keys()) {
      const block = plan.table.blocks[position];
      if (!resident.has(block.asset)) continue;
      planned.value += gemma4PleBlockBytes(plan.table, block);
      plan.full.set(position, acquire(plan.table, position));
    }
  };

  /**
   * 未常駐の block を「全量読み」と「行読み」へ振り分ける。
   *
   * `planned` は**この gather が握る常駐量**（hit + これから載せるぶん）で、2 表を通して 1 つを
   * 共有する。予算を超える手前で止めるので、**1 回の gather が自分の hit を追い出すことは無い** —
   * 一方で前の gather が載せた block は追い出されるので、作業集合は LRU で入れ替わる。
   */
  const decideTable = (plan: TablePlan, planned: { value: number }): void => {
    for (const [position, ids] of plan.grouped) {
      if (plan.full.has(position)) continue;
      const block = plan.table.blocks[position];
      const bytes = gemma4PleBlockBytes(plan.table, block);
      if (ids.length >= fullBlockRows(block) && planned.value + bytes <= budget) {
        planned.value += bytes;
        plan.full.set(position, acquire(plan.table, position));
        continue;
      }
      for (const id of ids) plan.rows.set(id, position);
    }
  };

  /** 行読みの 1 本（`[行 offset, +rowBytes)` を引いて自分の buffer に持つ）。 */
  const readRow = async (plan: TablePlan, id: number): Promise<void> => {
    const position = plan.rows.get(id) as number;
    const block: Gemma4PleBlock = plan.table.blocks[position];
    rowReads += 1;
    plan.read.set(
      id,
      await readAssetRange(
        openBlock(block.asset),
        (id - block.start) * plan.table.rowBytes,
        plan.table.rowBytes,
      ),
    );
  };

  /**
   * 2 表ぶんの行読みを 1 本の列に畳み、{@link ROW_READ_CONCURRENCY} 本のプールで流す。
   */
  const gatherRows = async (
    plans: readonly TablePlan[],
    options: Gemma4PleReadOptions,
  ): Promise<void> => {
    const jobs: { readonly plan: TablePlan; readonly id: number }[] = [];
    for (const plan of plans) {
      for (const id of plan.rows.keys()) jobs.push({ plan, id });
    }
    if (jobs.length === 0) return;
    let next = 0;
    const worker = async (): Promise<void> => {
      // MUST: 取り出しと `next` の前進の間に await を挟まない（挟むと 2 本が同じ行を引く）。
      while (next < jobs.length) {
        const job = jobs[next];
        next += 1;
        // 中断は段の境目で見る（読み 1 本を途中で畳む口は `AssetReader` に無い）。
        options.signal?.throwIfAborted();
        await readRow(job.plan, job.id);
      }
    };
    await Promise.all(Array.from({ length: Math.min(ROW_READ_CONCURRENCY, jobs.length) }, worker));
  };

  /**
   * 表 1 本から id の行を引き当てる（行読みの結果 → 無ければ全量 block の view）。
   *
   * `rowUnits` は 1 行ぶんの**要素数**（`values` はバイト数・`scales` は層数）。
   */
  const supplyOf = <T>(
    plan: TablePlan,
    id: number,
    blocks: ReadonlyMap<number, ArrayBuffer>,
    view: (buffer: ArrayBuffer) => T,
    rowUnits: number,
  ): Supply<T> => {
    const row = plan.read.get(id);
    if (row !== undefined) return { array: view(row), from: 0, owned: true };
    const position = gemma4PleBlockOf(plan.table, id);
    const buffer = blocks.get(position);
    if (buffer === undefined) {
      // 起きるなら計画と引き当ての食い違い（= この実装のバグ）なので黙って 0 行を配らない。
      throw new Error(`PLE gather: token ${id} の block ${position} を取っていない`);
    }
    return {
      array: view(buffer),
      from: (id - plan.table.blocks[position].start) * rowUnits,
      owned: false,
    };
  };

  /** 全量 block の待ち合わせ（block 添字 → バイト列）。 */
  const awaitFull = async (plan: TablePlan): Promise<ReadonlyMap<number, ArrayBuffer>> =>
    new Map(
      await Promise.all(
        [...plan.full].map(async ([position, data]) => [position, await data] as const),
      ),
    );

  const valuesView = (buffer: ArrayBuffer): Int8Array<ArrayBuffer> | Uint8Array<ArrayBuffer> =>
    index.storage === "i8" ? new Int8Array(buffer) : new Uint8Array(buffer);
  const scalesView = (buffer: ArrayBuffer): Float32Array<ArrayBuffer> => new Float32Array(buffer);

  return {
    async gather(ids: readonly number[], options: Gemma4PleReadOptions = {}): Promise<Tensor> {
      if (disposed) {
        throw new Error("PLE gather: dispose 済みの索引は引けない（常駐は解放済み）");
      }
      if (ids.length < 1) throw new Error("PLE gather の token 列が空");
      // 中断済みで呼ばれたら資産に 1 バイトも触らずに返す（段の境目の 1 本目）。
      options.signal?.throwIfAborted();
      ids.forEach((id, position) => {
        // ③ 実際に引く id（tokenizer が生成しうる special id を含む）— 範囲外は OOB ではなく
        // 「別 token の有効な行」になるので、ここが唯一の fail loudly の位置。
        if (!Number.isSafeInteger(id) || id < 0 || id >= index.tokens) {
          throw new Error(
            `token id[${position}] ${id} が PLE の索引の 0..${index.tokens - 1} の外`,
          );
        }
      });
      const data = new Float32Array(ids.length * stride);
      // 同じ id の位置を束ねる（逆量子化は id ごとに 1 回で済む）。
      const rows = new Map<number, number[]>();
      ids.forEach((id, position) => {
        const positions = rows.get(id);
        if (positions === undefined) rows.set(id, [position]);
        else positions.push(position);
      });
      // ① 行キャッシュの hit を先に書く（後続の読みの追い出しで落ちる前に触る）。
      const missing: number[] = [];
      for (const [id, positions] of rows) {
        const cached = rowCache.get(id);
        if (cached === undefined) {
          missing.push(id);
          continue;
        }
        rowCache.delete(id);
        rowCache.set(id, cached);
        const first = positions[0] * stride;
        writeRow(data, first, cached.values, 0, cached.scales, 0);
        copyDuplicates(data, positions, first);
      }
      if (missing.length > 0) {
        // ② 表ごとに block へ束ね、**2 表の hit を先に触ってから**方針を決める。
        const planned = { value: 0 };
        const valuePlan = groupTable(index.values, missing);
        const scalePlan = groupTable(index.scales, missing);
        touchHits(valuePlan, planned);
        touchHits(scalePlan, planned);
        decideTable(valuePlan, planned);
        decideTable(scalePlan, planned);
        // ③ 行読みを先に流す（hit 先行の延長 — 全量読みを待ってから流すと、その追い出しが
        // この gather の行キャッシュ hit を落とす）。
        await gatherRows([valuePlan, scalePlan], options);
        const valueBlocks = await awaitFull(valuePlan);
        const scaleBlocks = await awaitFull(scalePlan);
        // ④ 2 表を突き合わせて書く。
        for (const id of missing) {
          const positions = rows.get(id) as number[];
          const values = supplyOf(valuePlan, id, valueBlocks, valuesView, rowValueBytes);
          const scales = supplyOf(scalePlan, id, scaleBlocks, scalesView, scaleStride);
          const first = positions[0] * stride;
          writeRow(data, first, values.array, values.from, scales.array, scales.from);
          copyDuplicates(data, positions, first);
          // 行キャッシュに載せるのは**両表とも行読みで引いた**行だけ（全量 block の view を
          // 保存すると block そのものが予算の外で生き残る）。中断・dispose 後には復活させない。
          if (
            values.owned && scales.owned && !disposed && options.signal?.aborted !== true &&
            budget - residentBytes >= rowBytes
          ) {
            rowCache.delete(id);
            rowCache.set(id, { values: values.array, scales: scales.array });
            trimRows();
          }
        }
      }
      return { dtype: "f32", shape: [1, ids.length, index.layers, index.dim], data };
    },
    stats(): Gemma4PleStats {
      return {
        loads,
        rowReads,
        resident: resident.size,
        residentBytes: residentBytes + rowCache.size * rowBytes,
      };
    },
    dispose(): void {
      disposed = true;
      // 走行中の読みまでは止めない（返ってきた buffer は誰も掴まないので回収される）。
      resident.clear();
      residentBytes = 0;
      rowCache.clear();
    },
  };
};
