/**
 * Gemma 4 の PLE（per-layer embeddings）を**ホスト側で gather** する実装（ADR 0085）。
 *
 * 索引（`ple.json`）の codec と定数は `./ple-index.ts`・shard の読み口と検査は `./ple-shard.ts`
 * にあり、このファイルが持つのは**所有者**だけである — shard / 行 / 読み口 / 位置の 4 本の
 * キャッシュと `residentBytes` が 1 つの予算を共有する関係は割れないので、同じ関数の中に置く。
 * 公開 barrel とテストは従来どおりこのファイルの綴りで名前を取るので、移した名前は末尾で
 * そのまま re-export する。
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
 * ## 配布形（sidecar）
 *
 * 書き手の正本は `tools/export-recipes/gemma4/export_product.py`:
 *
 * - `ple.json` — 索引（token 総数 / 層数 / 層当たり次元 / embed scale / shard の token 範囲）
 * - `ple-NNNNN-of-NNNNN.safetensors` — **token-major**。`values` `[rows, layers, dim]` と
 *   `scales` `[rows, layers]` f32 で、1 token の PLE が**連続 1 読み**になる（ADR 0085 決定 1）。
 *   vocab の範囲で shard し、上限は書き手の容量（ADR 0090 — 256MiB − ヘッダ余裕 = 1 回の読みの上限）。
 *
 * schema 1 は I8、schema 2 は storage で I2 / I4 を明示する（ADR 0097 追記 4）。
 * packed 値も shape は論理要素数で、読み取りと常駐予算には格納 byte 数を使う。
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
 * ## gather の順序（hit 先行）と重複 id
 *
 * 1 回の gather は触る shard を束ねて 1 本ずつ引くが、**常駐している shard を先に処理する**。
 * 未常駐を先に読むと、その読みの LRU 追い出しが「この gather がまだ触っていない hit」を落とし、
 * 同じ gather の中で読み直しになる（実測: 予算 = 2 本で shard 1/2 が常駐している状態から
 * `[0,1,2]` 順に処理すると 3 load・`[1,2,0]` 順なら 1 load）。hit を先に触ると LRU の末尾へ
 * 回るので、後続の miss の追い出し先が「この gather で用の済んだ shard」側へ寄る。
 * 走行中に掴んだ shard の実体はローカル変数が持つので、途中で追い出されても値は揃う。
 *
 * 同じ id が並ぶ列（prefill の pad 行 id 0 が典型 — 768 行のうち大半が同じ id）は、**最初の
 * 1 位置だけ逆量子化して残りへ f32 バイト列を複写**する。再計算ではなく複写なので 2 段丸めの
 * 結果とビット同一で、`(q × scale) × embedScale` の契約はそのまま保たれる。
 *
 * ## 行読み（区間読みできる読み口 — ADR 0085 追記 2026-09-07）
 *
 * 読み口（{@link Gemma4PleShardSource}）が区間読みを持つなら、shard 全量ではなく **行 2 区間**
 * （values 8,960 B + scales 140 B）だけを引ける。方針は shard ごとに、その gather が束ねた
 * 一意行数 `rows` で決める（{@link createGemma4Ple} の方針表）。decode の 1 token は 253MiB の
 * 全量読みではなく 9,100 B の 2 読みになる。
 *
 * 読み終えた量子化行は最大 256 行まで LRU で再利用する。全量 shard を優先し、既存の
 * ホスト RAM 予算の空きだけを使う。予算 0 は行も保持しない（ADR 0085 追記 2026-09-12）。
 * 行の位置は shard ごとに**ヘッダを 1 度だけ**解いて持つ（先頭 8 バイト → ヘッダ長 → ヘッダ
 * JSON の 2 段読み）。検査は全量経路と**同じ 1 実装**（`./ple-shard.ts` の
 * `gemma4PleShardViews`）を通すので、行読みのときだけ別形式の資産が通ることはない。
 * 値は同じバイト列を同じ 2 段丸めに掛けるので、決定 4 のビット一致は経路に依らない。
 *
 * ## MUST: id 空間を相互照合する（ADR 0085 決定 5）
 *
 * sidecar の行数 / 主 embedding の vocab 行数 / 実際に引く id を突き合わせる。ここがずれると
 * **OOB ではなく「別 token の有効な行」**を引く（例外なしで沈黙して壊れる）ので、fail loudly の
 * 門を置く場所はここしかない。
 */

import type { Tensor } from "@karume/runtime";
import { ModelInputError } from "../errors.ts";
import {
  defaultGemma4PleResidentBytes,
  type Gemma4PleIndex,
  gemma4PleShardBytes,
  largestShardBytes,
  packFactor,
  SCALE_BYTES,
} from "./ple-index.ts";
import {
  type Gemma4PleReadOptions,
  type Gemma4PleShardSource,
  readRange,
  readResidentShard,
  readShardLayout,
  type ResidentShard,
  type ShardLayout,
  type ShardRange,
} from "./ple-shard.ts";

/**
 * 互換 re-export（公開 barrel・テスト・他 family が従来どおり `./ple.ts` の綴りで取る面）。
 *
 * NOTE: 分割前の名前の出どころをここへ残すのは、`mod.ts` / サブパス面の綴り（ADR 0008）を
 * 分割の都合で動かさないためである。新しい呼び手は移動先（`./ple-index.ts` / `./ple-shard.ts`）
 * から直接取る。
 */
export {
  defaultGemma4PleResidentBytes,
  type Gemma4PleIndex,
  type Gemma4PleShard,
  gemma4PleShardBytes,
  parseGemma4PleIndex,
} from "./ple-index.ts";
export {
  type Gemma4PleReadOptions,
  type Gemma4PleShardSource,
  gemma4PleShardViews,
  readGemma4PleHeaderPrefix,
} from "./ple-shard.ts";

export type Gemma4PleOptions = {
  readonly index: Gemma4PleIndex;
  /**
   * shard 1 本の読み口を開く（ファイル / hub の `openAsset` — 呼び手の責務）。
   *
   * MUST NOT: 開いた口へ全量を載せない — 返すのは口だけである（`Gemma4Ple` は shard ごとに
   * 1 度だけ開いて handle を持つので、ここで全量を RAM へ持つと「触っていない shard まで
   * 常駐する」形になる）。取得元が在庫を作るための温め（hub ⑧ の HF 取得元は在庫の無い参照で
   * 相 1 の全量 DL を 1 度だけ挟む）は口を開く費用そのもので、これには当たらない。
   * MAY: open に渡った `options.signal` をその温めの中断に使う。
   * MUST NOT: 開いた読み口が open 時の `options.signal` を保持しない。handle は最初に触った
   * gather のものが以後ずっと使われるので、保持すると open に使った signal が以後の読みへも
   * 効き、最初の生成の中断に後続の読みまで道連れになる。
   */
  readonly openShard: (
    file: string,
    options?: Gemma4PleReadOptions,
  ) => Promise<Gemma4PleShardSource>;
  /**
   * 主 embedding の vocab 行数（id 空間の相互照合 — ADR 0085 決定 5）。
   *
   * MUST: 省略可能にしない。sidecar と主 embedding が別の語彙で焼かれた組み合わせは、
   * shape も dtype も合ったまま**別 token の行**を引く。
   */
  readonly vocabSize: number;
  /**
   * 常駐させる shard と量子化行の**ホスト RAM 上限（バイト）**（LRU — ADR 0085 決定 3）。
   *
   * 省略時は {@link defaultGemma4PleResidentBytes}（= 最大 shard 2 本ぶん）。`0` は「常駐させ
   * ない」= gather が使い終わった shard を即座に落とす形で、正当な指定である（読み直しが毎回
   * 走るのと引き換えに、この sidecar のホスト RAM がピーク 1 本ぶんで収まる）。
   *
   * MUST: 本数ではなくバイトで受ける。shard 幅は資産世代（書き手の shard 上限）で変わるので、
   * 「2 本」は世代ごとに違う RAM を意味してしまう（ADR 0085 追記 2026-09-02）。
   */
  readonly maxResidentBytes?: number;
};

/** 遅延ロードの実測（門が「触った shard だけ読んだ」を恒真でなく見るための欄）。 */
export type Gemma4PleStats = {
  /** shard を**全量**取りに行った回数（キャッシュミスのうち行読みへ倒さなかった数）。 */
  readonly loads: number;
  /**
   * 行読みで引いた行数の累計（{@link Gemma4PleStats.loads} と対）。
   *
   * 「全量読みが起きていない」を恒真でなく見るための欄である — 全量読みが 0 なだけなら
   * 「そもそも引いていない」形でも成り立つので、引いた行数を並べて初めて行読み経路を
   * 踏んだことが言える。
   */
  readonly rowReads: number;
  /** 現在常駐している shard 数。 */
  readonly resident: number;
  /**
   * shard と行キャッシュが占めるホスト RAM（{@link Gemma4PleOptions.maxResidentBytes} と同じ単位）。
   *
   * 全量 shard は読み**始めた**時点で予約し、行は読み終えて保持した byte 数を加える。
   * shard の予約時に行を追い出すため、両者の合計は既存予算を超えない。
   * 読取り中の一時バッファや Map 等の管理用オブジェクトは従来同様この量に含まない。
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
   * 常駐 shard を解放する（ホスト RAM で {@link Gemma4PleOptions.maxResidentBytes} ぶん）。
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
 * `cost: "scan"` で行読みへ倒す一意行数の上限（**実測から置いた暫定値** — ADR 0085 追記
 * 2026-09-07）。
 *
 * scan の 1 読みは offset に比例した読み飛ばしを伴い、実測 17〜76 ms／全量読みは 120〜300 ms
 * である。3 行以上を引くなら全量 1 回の方が安いので、行読みは 2 行までに留める。
 */
const SCAN_ROW_LIMIT = 2;

/**
 * `cost: "seek"` で全量読みへ倒す一意行数の下限（**実測から置いた暫定値**）。
 *
 * seek の 1 行は 2 読みで 0.1〜0.6 ms なので、**この gather 1 回**を見る限り行読みは何行でも
 * 全量読み（120〜300 ms）より安い。全量読みが得になるのは「以後の gather の hit のために
 * 予算の空きへ載せる」ときだけで、この閾値は「1 回でこれだけ触る shard は以後も触る」という
 * 見込みの線である。低くしすぎると常駐が回り、高くしすぎると常駐が温まらない。
 */
const SEEK_FULL_ROWS = 32;

/**
 * 行読みの**同時発行の上限**（1 回の gather が全 shard を通して持つ in-flight の読み本数）。
 *
 * MUST: 上限を置く。取得元によっては **1 read = 1 fd** で（`denoDirectory` は read のたびに
 * `Deno.open` する — hub の `AssetRangeReader` は fd も handle も保持しない契約）、上限が
 * 無いと 1 回の gather が**引く行数ぶん**の fd を同時に握る。prefill の 768 行 chunk を全て
 * 行読みで流す形（常駐予算に空きが無い状態）が現実の上界で、`ulimit -n 1024`（Linux の
 * 既定）の残りを丸ごと食う。実測（2026-09-07・上限無しの版）: 768 行でピーク 768 本、
 * 1,200 行で `Too many open files (os error 24)`。
 *
 * 16 に置いたのは、seek の 1 読みが 46 µs なので 768 行（= 行あたり 2 区間で 1,536 読み）でも
 * `1,536 / 16 × 46 µs ≈ 4 ms` に収まり、prefill 1 chunk の壁（数百 ms）に対して無視できる
 * ため（実測: 768 行の行読みが 53 ms 対 上限無し 73 ms — 上限があっても遅くならない）。
 * 上げても壁は縮まず（律速は GPU 側）、下げると prefill の gather だけが伸びる。
 */
const ROW_READ_CONCURRENCY = 16;

/** 同じ token の再読取りを省く上限。shard 常駐後の予算の空きだけを使う（ADR 0085）。 */
const ROW_CACHE_CAPACITY = 256;

type CachedRow = {
  readonly values: Int8Array<ArrayBuffer> | Uint8Array<ArrayBuffer>;
  readonly scales: Float32Array<ArrayBuffer>;
};

/** 行読みへ倒した shard 1 本ぶんの計画（方針表の結果 — `range` は絞り込み済み）。 */
type RowPlan = {
  readonly position: number;
  readonly rows: ReadonlyMap<number, number[]>;
  readonly source: Gemma4PleShardSource;
  readonly range: ShardRange;
};

/** 行読みの 1 単位（プールが 1 スロットで流す仕事）。 */
type RowJob = {
  readonly plan: RowPlan;
  readonly id: number;
  readonly positions: readonly number[];
};

/**
 * sidecar を**触ったぶんだけ**読む gather を組む（ADR 0085 決定 3 + 追記 2026-09-07）。
 *
 * shard ごとに、その gather が束ねた**一意行数 `rows`** で方針を決める:
 *
 * | 読み口 | 条件 | 方針 |
 * | --- | --- | --- |
 * | 常駐（読み中を含む） | — | hit（LRU の末尾へ） |
 * | 区間読み無し | — | 全量読み + LRU（追い出しあり） |
 * | `cost: "scan"` | `rows ≤ 2` | 行読み |
 * | `cost: "scan"` | `rows > 2` | 全量読み + LRU（追い出しあり） |
 * | `cost: "seek"` | `rows ≥ 32` かつ予算に**追い出し無しで**載る | 全量読み + LRU |
 * | `cost: "seek"` | それ以外 | 行読み |
 *
 * **区間読みを持つ shard は LRU の追い出しを起こさない**（seek の行）。空きの勘定には
 * **この gather で全量読みする全プラン**のバイトを足すので、取得元が混在していても
 * （区間読みを持たない shard の全量読みが同じ gather に居ても）seek の shard が追い出しの
 * 引き金になることはない。seek では行読みが全量
 * 読みの 1/1000 級なので、予算を超えてまで載せる価値が無い — 実測（2026-09-07・自然文 2 ターン
 * × 200 token・既定予算）では、9 本に散る生成 token に対して LRU が回り続け、shard の読み直しが
 * 137 回・約 42 s／壁 58 s だった。
 *
 * 処理順は **hit → 行読み → 全量読み**。hit を先に触るのは、未常駐の読みが起こす追い出しで
 * 「この gather がまだ触っていない hit」が落ちるのを避けるため（モジュール doc）。全量読みを
 * 最後に置くのはその延長で、同じ gather の中で自分の hit を捨てない。
 *
 * MUST: モジュール副作用ゼロ（この関数を呼ぶまで何も起きない）。
 */
export const createGemma4Ple = (options: Gemma4PleOptions): Gemma4Ple => {
  const { index, openShard, vocabSize } = options;
  const budget = options.maxResidentBytes ?? defaultGemma4PleResidentBytes(index);
  // 予算は呼び手の `maxResidentPleBytes` そのものなので入力起因（ADR 0107 決定 2）。
  if (!Number.isSafeInteger(budget) || budget < 0) {
    throw new ModelInputError(`maxResidentBytes ${budget} が 0 以上の整数でない`);
  }
  const shardBytes = index.shards.map((shard) => gemma4PleShardBytes(index, shard));
  const largest = largestShardBytes(index);
  // MUST: 「1 本すら載らない予算」は fail loudly。黙って超過すれば予算が意味を失い、黙って
  // 守れば gather が引けない — どちらも呼び手の指定を裏切る。0 は例外で、「常駐させない」
  // という指定として正当（読み終えた shard を即座に落とす形）。
  // 資産の寸法と突き合わせるが、拒否しているのは呼び手が渡した予算の値なので入力起因。
  if (budget > 0 && budget < largest) {
    throw new ModelInputError(
      `maxResidentBytes ${budget} が PLE shard 1 本ぶん ${largest} バイトに満たない` +
        `（この索引の shard は ${index.shards.length} 本 — 常駐させないなら 0 を渡す）`,
    );
  }
  // ① sidecar の行数 と ② 主 embedding の vocab 行数（ADR 0085 決定 5 の相互照合）。
  // こちらは焼いた組み合わせの齟齬なので素の `Error` のまま（呼び手は指定を直せない）。
  if (index.tokens !== vocabSize) {
    throw new Error(
      `PLE sidecar の行数 ${index.tokens} が主 embedding の vocab 行数 ${vocabSize} と違う` +
        `（別の語彙で焼かれた組み合わせ — 引ける id が食い違ったまま形は合う）`,
    );
  }
  const stride = index.layers * index.dim;
  const factor = packFactor(index);
  const rowValueBytes = stride / factor;
  /** 行 1 本ぶんの `scales`（f32 × 層数）のバイト数。 */
  const scaleStride = index.layers * SCALE_BYTES;
  const rowBytes = rowValueBytes + scaleStride;
  /** 行も量子化された byte 列のまま保持し、逆量子化の順序を変えない。挿入順 = LRU。 */
  const rowCache = new Map<number, CachedRow>();

  /** 挿入順 = LRU（触った shard を末尾へ付け替え、予算超過分は先頭から落とす）。 */
  const resident = new Map<number, Promise<ResidentShard>>();
  /**
   * shard ごとの読み口（**1 度だけ開く**）。閉じる面は無い
   * （{@link Gemma4PleShardSource} の doc — 支える読み口が資源を保持しない契約）。
   */
  const sources = new Map<number, Promise<Gemma4PleShardSource>>();
  /** 行読みの位置（ヘッダを shard ごとに 1 度だけ解いた結果）。 */
  const layouts = new Map<number, Promise<ShardLayout>>();
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

  const release = (position: number): void => {
    resident.delete(position);
    residentBytes -= shardBytes[position];
  };

  const shardOf = (id: number): number => {
    // 索引は昇順の隙間なし分割（`parseGemma4PleIndex` の MUST）なので二分探索でよい。
    let low = 0;
    let high = index.shards.length - 1;
    while (low < high) {
      const middle = (low + high) >> 1;
      if (id < index.shards[middle].stop) high = middle;
      else low = middle + 1;
    }
    return low;
  };

  /**
   * shard の読み口を開く（**1 度だけ** — 2 度目以降は同じ handle を返す）。
   *
   * MUST: 失敗した open を握らない（次の gather が同じ拒否済み Promise を掴み続ける）。
   */
  const openSource = (
    position: number,
    options: Gemma4PleReadOptions,
  ): Promise<Gemma4PleShardSource> => {
    const cached = sources.get(position);
    if (cached !== undefined) return cached;
    const pending = openShard(index.shards[position].file, options);
    pending.catch(() => {
      if (sources.get(position) === pending) sources.delete(position);
    });
    if (!disposed) sources.set(position, pending);
    return pending;
  };

  /** 行の位置（ヘッダの 2 段読み）を shard ごとに 1 度だけ解く。失敗は握らない（同上）。 */
  const layoutOf = (
    position: number,
    range: ShardRange,
    bytes: number,
    options: Gemma4PleReadOptions,
  ): Promise<ShardLayout> => {
    const cached = layouts.get(position);
    if (cached !== undefined) return cached;
    const pending = readShardLayout(range, bytes, index, index.shards[position], options);
    pending.catch(() => {
      if (layouts.get(position) === pending) layouts.delete(position);
    });
    if (!disposed) layouts.set(position, pending);
    return pending;
  };

  /**
   * shard 1 本を常駐キャッシュから取る（未常駐なら**全量**読む）。
   *
   * NOTE: `signal` が効くのは**その shard の読みを始めた gather** に対してだけである。同じ
   * shard を待つ後続の gather は先行の読みに相乗りするので、先行が中断されれば同じ拒否を
   * 受ける（中断は自分のものでないので失敗として上がる = 沈黙はしない）。読みを要求ごとに
   * 分けると 758MB の二重読みになるため、best-effort の側を取っている。
   *
   * MUST: 同期のまま保つ（`resident` の照会と登録の間に await を挟まない）— 挟むと同じ shard を
   * 同時に要求した 2 本が揃って miss と判定し、二重読みになる。
   */
  const acquire = (
    position: number,
    source: Gemma4PleShardSource,
    options: Gemma4PleReadOptions,
  ): Promise<ResidentShard> => {
    const cached = resident.get(position);
    if (cached !== undefined) {
      // 参照したので末尾へ付け替える（Map の反復順 = 挿入順）。
      resident.delete(position);
      resident.set(position, cached);
      return cached;
    }
    const shard = index.shards[position];
    loads += 1;
    const pending = source.readAll(options).then((bytes) => readResidentShard(bytes, index, shard));
    // MUST: 失敗した取得を常駐させない（次の gather が同じ拒否済み Promise を掴み続ける）。
    pending.catch(() => {
      if (resident.get(position) === pending) release(position);
    });
    // dispose 中も先行 gather は完了してよいが、所有者のキャッシュへは戻さない。
    if (disposed) return pending;
    resident.set(position, pending);
    residentBytes += shardBytes[position];
    // 予算はバイトで測る（本数ではない — shard 幅は資産世代で変わる）。予算 0 では今入れた
    // ぶんもここで落ちるが、走行中の gather は解決済みの実体を自分で掴んでいるので値は揃う。
    for (const oldest of resident.keys()) {
      if (residentBytes <= budget) break;
      release(oldest);
    }
    // 全量 shard を優先し、行の常駐を既存予算へ上乗せしない。
    trimRows();
    return pending;
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

  /** 常駐している shard から、束ねた行をまとめて書く。 */
  const fillResident = (
    data: Float32Array,
    loaded: ResidentShard,
    rows: ReadonlyMap<number, number[]>,
  ): void => {
    for (const [id, positions] of rows) {
      const row = id - loaded.start;
      const first = positions[0] * stride;
      writeRow(data, first, loaded.values, row * rowValueBytes, loaded.scales, row * index.layers);
      copyDuplicates(data, positions, first);
    }
  };

  /**
   * shard から**行 1 本だけ**を引いて書く（全量は読まない）。
   *
   * 1 行の値と scale は別の 2 区間にある。全行を同時に引いても
   * {@link ROW_READ_CONCURRENCY} を超えない小さい gather だけ、2 区間を並行して読む。
   * 書き込み先は行ごとに重ならず、値の復元順序も変わらない。
   */
  const readRow = async (
    job: RowJob,
    data: Float32Array,
    options: Gemma4PleReadOptions,
    parallelPair: boolean,
  ): Promise<void> => {
    const { plan, id, positions } = job;
    const { source, range } = plan;
    const { file } = index.shards[plan.position];
    const layout = await layoutOf(plan.position, range, source.bytes, options);
    const row = id - layout.start;
    const valuesRequest = readRange(
      range,
      source.bytes,
      file,
      layout.valuesOffset + row * rowValueBytes,
      rowValueBytes,
      options,
    );
    const readScales = (): Promise<ArrayBuffer> =>
      readRange(
        range,
        source.bytes,
        file,
        layout.scalesOffset + row * scaleStride,
        scaleStride,
        options,
      );
    const scalesRequest = parallelPair ? readScales() : valuesRequest.then(readScales);
    // 片方の拒否でも、発行済みのもう片方を待ち切る。値側の原因を先に返す。
    const [valuesResult, scalesResult] = await Promise.allSettled([valuesRequest, scalesRequest]);
    if (valuesResult.status === "rejected") throw valuesResult.reason;
    if (scalesResult.status === "rejected") throw scalesResult.reason;
    const values = valuesResult.value, scales = scalesResult.value;
    rowReads += 1;
    const first = positions[0] * stride;
    const cached: CachedRow = {
      values: index.storage === undefined ? new Int8Array(values) : new Uint8Array(values),
      scales: new Float32Array(scales),
    };
    writeRow(data, first, cached.values, 0, cached.scales, 0);
    copyDuplicates(data, positions, first);
    // 値と scale の成功後だけ保存し、中断・dispose 後にはキャッシュを復活させない。
    if (!disposed && !options.signal?.aborted && budget - residentBytes >= rowBytes) {
      rowCache.delete(id);
      rowCache.set(id, cached);
      trimRows();
    }
  };

  /**
   * 行読みへ倒した全 shard の行を**1 本の列**に畳み、{@link ROW_READ_CONCURRENCY} 本のプールで
   * 流す。
   *
   * MUST: 上限は shard ごとではなく gather 全体に掛ける — shard ごとに掛けると、9 本に散る
   * prefill で上限が本数ぶん倍になる（fd の枯渇はそこで起きる）。
   */
  const gatherRows = async (
    plans: readonly RowPlan[],
    data: Float32Array,
    options: Gemma4PleReadOptions,
  ): Promise<void> => {
    const jobs: RowJob[] = [];
    for (const plan of plans) {
      for (const [id, positions] of plan.rows) {
        const cached = rowCache.get(id);
        if (cached === undefined) {
          jobs.push({ plan, id, positions });
        } else {
          // hit を先に複写し、後続 miss の LRU 追い出しで読み直すことを避ける。
          rowCache.delete(id);
          rowCache.set(id, cached);
          const first = positions[0] * stride;
          writeRow(data, first, cached.values, 0, cached.scales, 0);
          copyDuplicates(data, positions, first);
        }
      }
    }
    const parallelPair = jobs.length * 2 <= ROW_READ_CONCURRENCY;
    let next = 0;
    const worker = async (): Promise<void> => {
      // MUST: 取り出しと `next` の前進の間に await を挟まない（挟むと 2 本が同じ行を引く）。
      while (next < jobs.length) {
        const job = jobs[next];
        next += 1;
        await readRow(job, data, options, parallelPair);
      }
    };
    await Promise.all(
      Array.from({ length: Math.min(ROW_READ_CONCURRENCY, jobs.length) }, worker),
    );
  };

  return {
    async gather(ids: readonly number[], options: Gemma4PleReadOptions = {}): Promise<Tensor> {
      if (disposed) {
        throw new Error("PLE gather: dispose 済みの sidecar は引けない（常駐は解放済み）");
      }
      if (ids.length < 1) throw new Error("PLE gather の token 列が空");
      ids.forEach((id, position) => {
        // ③ 実際に引く id（tokenizer が生成しうる special id を含む）— 範囲外は OOB ではなく
        // 「別 token の有効な行」になるので、ここが唯一の fail loudly の位置。
        if (!Number.isSafeInteger(id) || id < 0 || id >= index.tokens) {
          throw new Error(
            `token id[${position}] ${id} が PLE sidecar の 0..${index.tokens - 1} の外`,
          );
        }
      });
      const data = new Float32Array(ids.length * stride);
      // shard ごとに束ねて引き（同じ shard の行が散っていても取得は 1 回）、その中で同じ id の
      // 位置をさらに束ねる（逆量子化は id ごとに 1 回で済む）。
      const grouped = new Map<number, Map<number, number[]>>();
      ids.forEach((id, position) => {
        const shard = shardOf(id);
        let rows = grouped.get(shard);
        if (rows === undefined) {
          rows = new Map<number, number[]>();
          grouped.set(shard, rows);
        }
        const positions = rows.get(id);
        if (positions === undefined) rows.set(id, [position]);
        else positions.push(position);
      });
      // hit 先行（モジュール doc「gather の順序」）— 未常駐を先に読むと、その追い出しで
      // 「まだ触っていない hit」が落ち、同じ gather の中で読み直しになる。hit を先に触れば
      // LRU の末尾へ回るので、後続の miss に追い出されにくくなる。
      const hits: [number, Map<number, number[]>][] = [];
      const misses: [number, Map<number, number[]>][] = [];
      for (const entry of grouped) {
        if (resident.has(entry[0])) hits.push(entry);
        else misses.push(entry);
      }
      for (const [position, rows] of hits) {
        // 常駐している shard の読み口は必ず開いている（常駐になる唯一の道が全量読み）ので、
        // ここの `openSource` は掴んである handle を返すだけである。
        const source = await openSource(position, options);
        fillResident(data, await acquire(position, source, options), rows);
      }
      // 未常駐は読み口を開いてから方針を決める（区間読みの有無と費用の型が方針表の軸）。
      const opened = await Promise.all(misses.map(async ([position, rows]) => ({
        position,
        rows,
        source: await openSource(position, options),
      })));
      const rowPlans: RowPlan[] = [];
      const fullPlans: typeof opened = [];
      // 予算の空きは「この gather でこれから載せるぶん」を含めて見る — 1 本ずつなら載る 2 本が
      // 合わさって追い出す形を作らない（seek で追い出しを起こさないことが方針表の要）。
      // MUST: 全量読みする**全プラン**を足す（区間読みを持たない shard・scan の 3 行以上も）。
      // seek の分だけを数えると、取得元が混在した gather では先に走る他の全量読みが予算を
      // 埋め、seek の 1 本が「空きに載る」判定のまま追い出しを起こす。
      let planned = residentBytes;
      for (const plan of opened) {
        const bytes = shardBytes[plan.position];
        const { range } = plan.source;
        if (range === undefined) {
          planned += bytes;
          fullPlans.push(plan);
          continue;
        }
        if (range.cost === "scan") {
          if (plan.rows.size > SCAN_ROW_LIMIT) {
            planned += bytes;
            fullPlans.push(plan);
          } else rowPlans.push({ ...plan, range });
          continue;
        }
        if (plan.rows.size >= SEEK_FULL_ROWS && planned + bytes <= budget) {
          planned += bytes;
          fullPlans.push(plan);
        } else {
          rowPlans.push({ ...plan, range });
        }
      }
      // 行読みが先（hit 先行の延長）— 全量読みを先に流すと、その追い出しがこの gather の hit を
      // 落とす。行読みは常駐に触らないので shard をまたいで束ねてよく、発行数だけを
      // {@link ROW_READ_CONCURRENCY} で抑える。
      await gatherRows(rowPlans, data, options);
      for (const plan of fullPlans) {
        fillResident(data, await acquire(plan.position, plan.source, options), plan.rows);
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
      // 読み口と行の位置も落とす（gather は以後 fail loudly なので、掴み続ける理由が無い）。
      sources.clear();
      layouts.clear();
      rowCache.clear();
    },
  };
};
