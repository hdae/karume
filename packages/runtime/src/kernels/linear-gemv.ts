/**
 * linear の **GEMV 族**（重み i4 / i8 格納、および M=1 の f16 / f32 格納 — ADR 0082）。`linear` の 2 本目のカーネル族で、
 * 出力・束縛・uniform は既定経路（src/kernels/gemm.ts の linear）と同じまま、**担当割りだけ**が
 * 「1 スレッド = 1 出力列」へ変わる。M=1（decode）の変種と、小 M（2〜{@link LINEAR_GEMV_MAX_ROWS}・
 * 短い prefill / 投機検証）の**行ブロック変種**の 2 形を持つ（後者は ADR 0082 追記 5 /
 * perf-ledger K-21）。
 *
 * ## なぜ既定の GEMM 骨格では足りないのか（機序 — ADR 0082 / research 2026-08-30 §7）
 *
 * 既定は M のバケットで幾何を選ぶ（src/kernels/gemm-geometry.ts の `gemmGeometryForRows`）が、
 * どのバケットも「共有タイル + K タイル 16 ごとの二重 barrier」という 1 つの骨格を共有する。
 * M=1 に当たる `GEOMETRY_M16N16`（64 スレッド）では:
 *
 * - **出力を書くのは 64 スレッド中 4 本だけ**（`lid.y == 0` の行）。共有 A タイル（16 行 × K16）は
 *   1 行しか実体が無く、残り 15 行は 0 で埋めた死荷重になる。
 * - K タイルごとに `workgroupBarrier()` を 2 回通るため、**重み読みのレイテンシがタイル本数ぶん
 *   逐次に露出する**。実測は `k / 16 × ≈1.3µs` の k 比例で **n にほぼ非依存**、重みが L2 に
 *   収まる小形でも同じだけ遅い — つまり律速は帯域飢餓ではなく**発行の逐次化**。
 *
 * 本族は共有メモリと barrier を丸ごと落とし、重み語を先読みしてメモリ並列度を作る。
 * `n` 本の独立した縮約が同時に走るので、遅延は列方向の並列で隠れる。
 * この費用は M=2〜32 でも M に無関係にそのまま残る（同じ幾何 M16N16 が M ≤ 64 を受ける —
 * research 2026-09-07 §7.2 で GEMM 経路 65〜67 ms が M=2〜16 で不変）ので、行ブロック変種は
 * 同じ手で同じ費用を消す。
 *
 * ## 行ブロック（M ≥ 2）— 重み語 1 回読みを `rows` 行で共有し、行方向は y タイル
 *
 * 1 スレッドが 1 列 × `rows` 行の縮約を持ち、重み語（と i4 の group scale）は 1 回だけ読んで
 * 行ごとの x と積和する。M は `ceil(m / rows)` 枚の y タイルに割る（重みはタイルごとに読み直す）。
 * `rows` は {@link linearGemvRowsForShape} が **(格納, m, n, 並列度の目標) の純関数**で決める
 * （目標は Session 生成時に固定される静的なノブ — {@link ROWS_THREAD_TARGET}）。掃引（RTX 3080 Ti・
 * gemma4 E2B の census 13 形 × M ∈ {1..64} — docs/research/2026-09-07-gemv-rows-k21.md）で見えた機序:
 *
 * - スレッド数（= n × y タイル数）が **≈16K を下回る間は、行を 1 スレッドに畳むより y タイルで
 *   並列度を買う方が速い**（重みの読み直しは同じ dispatch 内で L2 が吸う）。n が小さい形
 *   （256 / 512 / 1536 列）は M=32 でも `rows = 1` が最良。
 * - 16K を超えた先は重みの転送量が効き、`rows` を増やして読み直しを減らす方が速い（lm_head
 *   262,144 列は `rows = M` が最良）。
 * - ただし 1 スレッドが 1 語あたりに抱える要素数（`rows × 刻み`）には天井が 2 つある。実行時間の
 *   天井は 512（i4 16 行 / i8 32 行 — その上はレジスタ圧で遅い）、**シェーダのコンパイル費**の
 *   天井がその手前の {@link ROWS_ELEMENTS_PER_WORD_CAP} = 256（生成テキストが行数に比例し、
 *   解析・検証の費用がテキスト量に超線形 — 定数の doc）。
 *
 * 目標値 {@link ROWS_THREAD_TARGET} と天井 1 つずつで両側を再現し、形ごとの最良（oracle）との差は
 * census 加重で M ≤ 32 は 1〜5%・M ≤ 64 は ≤ 13%。M=1 は行ブロック化しない（既存の M=1 変種の
 * 生成物と同じテキスト — 実測でも同等以上で、decode の生成物を 1 バイトも動かさない）。
 *
 * ## 格納の軸（i4 / i8）
 *
 * 機序（barrier による発行の逐次化）は格納 dtype に依らないので、族の内側では格納が**変種軸**に
 * なる。重み 1 語は**どちらも `vec4<u32>` = 16 B** で、運ぶ要素数（= 縮約の刻み
 * {@link linearGemvUnit}）だけが違う: i4 は 32 要素・i8 は 16 要素。scale の引き方も違い、
 * i4 は group ごと（k 依存なので語ごとに引き直す）・i8 は出力チャネルごと 1 本
 * （k 不変なので縮約の外で 1 度だけ束ねる — ADR 0019 のループ不変巻き上げ）。
 *
 * ## 数値契約（ビット同一 MUST）
 *
 * MUST: 1 出力要素あたりの縮約は **k 昇順の逐次**・積和の字面は `acc = acc + a * b`・bias は
 * 最後に 1 度だけ加算。重みの復元は既定経路（weight-storage.ts の `dequant4`）と**同じ字面**で、
 * i4 は `f32(i32(u) − 8) * scale`・i8 は `f32(q) * scale` の成分ごと f32 乗算
 * （scale を縮約の外へ括り出さない — ADR 0019）。変わるのは ADR 0022 決定 3 が自由と認めた
 * **担当割り**だけ = **既定経路とビット同一**。行ブロックでも 1 出力要素あたりの縮約順・積の
 * 対応・bias の足し順は M=1 と同一で、違いは復元した値 `f32(q) * scale` を `let` に置いて行間で
 * 共有すること（丸め点は同じ 1 個の f32 乗算 — {@link rowsMacsI4} の WHY）。
 * MUST: 先読み（`unroll`）は「語をまとめて読む」だけで、**積和の順序は語の昇順のまま**。
 * 語をまたいで積和を混ぜると 1 出力要素あたりの加算順が動き、契約が割れる。
 * NOTE: f32 の縮約に順序非依存の理論保証は無いので、ビット同一は gemm-geometry と同じく
 * **実測命題**（門 = tests/gpu_linear_gemv_test.ts の u32 完全一致）。
 *
 * MUST NOT: k を分割して部分和を足し直す（split-K）形は縮約順が変わるため、この族には
 * 入れない。実測でも ADR 0058 の opt-in 席を切る価値が無かった（ADR 0082 §不採用）。
 *
 * ## 幾何との関係（gemm-geometry の「唯一の選択点」MUST の射程）
 *
 * gemm-geometry は「担当割りの選択点は `gemmGeometryForRows` 1 箇所」と書くが、その射程は
 * **GEMM 骨格の中**にとどまる（ADR 0082 決定 2）。本族は骨格を共有しない別カーネルで、選択は
 * 2 段 — **族の選択**（GEMV 族に入るか）は `#buildLinear` の 1 箇所、**変種の選択**（M=1 変種か
 * 行ブロックか・行ブロックの `rows`）は {@link defaultLinearGemvRowsVariant} の 1 箇所 — で、
 * どちらも**プラン時 shape の純関数**。判別子はパイプラインキーの族名 `linear_gemv`（行ブロックは
 * 変種の `r<rows>`）に載る — 実行時オートチューン禁止と「同一キー → バイト同一 WGSL」は
 * どちらも保たれる。
 */

import { CodegenError } from "../codegen/errors.ts";
import { gemmParams } from "./gemm.ts";
import {
  i4GroupKeyPart,
  i4GroupShift,
  WEIGHT_SCALE_VAR,
  weightKeyPart,
  weightNote,
  weightScaleWgsl,
  type WeightStorage,
} from "./weight-storage.ts";

/**
 * 重み 1 語（`vec4<u32>` または `vec4<f32>` = 16 B）が運ぶ要素数 = **縮約の刻み**（格納ごと）。
 *
 * 適格判定（src/runtime/recipe-builder.ts の `#buildLinear`）が k と group 長へ課す整除の
 * 単位でもあるので、門とカーネルが格納ごとに同じ 1 個の導出点を読む。
 * f16 は 8 要素 / 語、f32 は 4 要素 / 語で、両者とも M=1 のみ（ADR 0082 追記 6・7）。
 */
export const linearGemvUnit = (storage: WeightStorage): number => {
  if (storage === "i4") return 32;
  if (storage === "i8") return 16;
  if (storage === "f16") return 8;
  if (storage === "f32") return 4;
  throw new CodegenError(
    `linear_gemv: 重み ${storage} 格納は本族に無い（f32 / f16 / i4 / i8 のみ）`,
  );
};

/** f16 / f32 は M=1 だけで検収する。行ブロック側へ暗黙に広げない。 */
const assertRowsStorage = (storage: WeightStorage): void => {
  if (storage === "f16" || storage === "f32") {
    throw new CodegenError(`linear_gemv: ${storage} 格納の行ブロックは未対応（M=1 変種のみ）`);
  }
};

/** WebGPU core が保証する 1 workgroup のスレッド数上限（`cols` の上界）。 */
const MAX_THREADS = 256;

/**
 * 生成パラメタ（幾何にあたるもの）。**{@link defaultLinearGemvVariant} が唯一の選択点**で、
 * 実行時に選び直さない（ADR 0022 の実行時オートチューン禁止は本族にも掛かる）。
 */
export type LinearGemvVariant = {
  /** 1 workgroup が担当する出力列数（= workgroup のスレッド数 — 1 スレッド 1 列）。 */
  readonly cols: number;
  /** 1 反復で先読みする重み語数（メモリ並列度）。積和の順序には影響しない。 */
  readonly unroll: number;
};

/**
 * 行ブロック変種（M ≥ 2）の生成パラメタ。`cols` / `unroll` は M=1 変種と同じ意味で、
 * `rows` = 1 スレッドが持つ行数（y タイルの高さ）。**{@link defaultLinearGemvRowsVariant} が
 * 唯一の選択点**。
 */
export type LinearGemvRowsVariant = LinearGemvVariant & {
  readonly rows: number;
};

/**
 * 既定の変種。**RTX 3080 Ti / gemma4 E2B decode の実 12 形 + 端数 4 形の掃引**で census 加重
 * 最良（`c32 u4`・対既定 ×8.45 — ADR 0082 / docs/research/2026-08-30-gemma4-decode-wallclock.md
 * §7）。MUST: 既定の変更はビット同一門（tests/gpu_linear_gemv_test.ts）の再実測とセット。
 */
export const defaultLinearGemvVariant = (): LinearGemvVariant => ({ cols: 32, unroll: 4 });

/**
 * 行ブロック変種が受ける M の上限 = **門の上限**（`#buildLinear` が本族へ入れる行数の範囲）。
 * 掃引した範囲（M ≤ 64 — 既定 GEMM 骨格の M16N16 バケットと同じ幅）そのもので、その外は既定の
 * GEMM 骨格のまま（ADR 0082 決定 4 の「実測した範囲に留める」）。
 */
export const LINEAR_GEMV_MAX_ROWS = 64;

/**
 * `rows` を決める並列度の目標（スレッド数 = 出力列 n × y タイル数 `ceil(m / rows)`）。
 * RTX 3080 Ti の掃引で、この値を下回る形は y タイルで並列度を買う方が速く、上回る形は
 * `rows` で重みの読み直しを減らす方が速かった（モジュール doc の機序）。
 * NOTE: **参照 device（RTX 3080 Ti・80 SM）の飽和点を焼いた値**で、可搬な最適値ではない。
 * 飽和点が 1 桁小さい GPU（内蔵 GPU・Apple M 系）では y タイルを買いすぎて重みの読み直しが
 * 最適より増える。他 device の値は `SessionOptions.linearGemvRowsThreadTarget` で差し替える
 * （Session 生成時に固定する**静的**なノブ = {@link linearGemvRowsForShape} の第 4 引数に流れる
 * 限界値で、選択は純関数のまま）。device を見て自動で選ぶことはしない（実行時オートチューン禁止
 * 〈ADR 0022〉— docs/limitations.md）。
 * NOTE: 目標を替えても `rows` はキーに載る（{@link linearGemvRowsKey} の `r<rows>`）ので、
 * 「同一キー → バイト同一 WGSL」は保たれる — 目標が違えば選ばれる `rows` が変わり、キーも変わる。
 * MUST: 変更は掃引の再実測とセット（tests/gpu_linear_gemv_test.ts の門は値を固定しない —
 * 選択が (格納, m, n, 目標) の純関数であることだけを見る）。
 */
const ROWS_THREAD_TARGET = 16384;

/**
 * 1 スレッドが重み語 1 本あたりに抱える要素数（`rows × ` {@link linearGemvUnit}）の天井 =
 * i4 8 行 / i8 16 行。実行時間だけなら 512（i4 16 行 / i8 32 行）まで伸びる（その上はレジスタ圧で
 * 遅くなる）が、**生成テキストが行数に比例してシェーダのコンパイル費になる**ので 256 で止める:
 * 512 では i4 16 行が 138 KB / ≈165 ms・i8 32 行が 137 KB / ≈0.9 s（naga の解析・検証がテキスト量に
 * 超線形）で、初回ターンに ≈1 s 乗る。256 なら選ばれうる最大形（i4 8 行 73 KB・i8 16 行 70 KB）で
 * 1 本 ≈55〜60 ms・初回ターン合計 ≈150 ms。失う実行時間は census 加重で M=32 / 64 の +3%（i4）と
 * lm_head +1.6 ms/run（i8）— research 2026-09-07-gemv-rows-k21 §6。
 * MUST: 変更は掃引（実行時間 + コンパイル時間）の再実測とセット。
 */
const ROWS_ELEMENTS_PER_WORD_CAP = 256;

/**
 * 行ブロックの高さ `rows` を **(格納, m, n, 目標) の純関数**で選ぶ: m 以下の最大の 2 冪
 * （天井 {@link ROWS_ELEMENTS_PER_WORD_CAP} / 刻み の内側）から始め、スレッド数
 * `n · ceil(m / rows)` が `threadTarget` に届くまで半分にする（届かなければ 1）。
 * n が小さい形は常に 1（= M=1 のカーネルを y に M 枚並べた形）、lm_head 級の n では天井まで伸びる。
 *
 * `threadTarget` は既定が {@link ROWS_THREAD_TARGET}（参照 device の飽和点）で、他 device では
 * `SessionOptions.linearGemvRowsThreadTarget` が Session 生成時に固定した値がここへ来る。
 * 値域の門は Session 側 1 箇所（同じ門を 2 実装持たない）。
 *
 * MUST: 返り値はキーに載る（{@link linearGemvRowsKey}）。純関数であることが
 * 「同一キー → バイト同一 WGSL」と実行時オートチューン禁止の両方を担保する。
 */
export const linearGemvRowsForShape = (
  storage: WeightStorage,
  m: number,
  n: number,
  threadTarget: number = ROWS_THREAD_TARGET,
): number => {
  assertRowsStorage(storage);
  if (!Number.isSafeInteger(m) || m < 1 || m > LINEAR_GEMV_MAX_ROWS) {
    throw new CodegenError(`linear_gemv: 行数 ${m} は 1..${LINEAR_GEMV_MAX_ROWS} の外`);
  }
  if (!Number.isSafeInteger(n) || n < 1) {
    throw new CodegenError(`linear_gemv: 列数 ${n} は正整数でない`);
  }
  const cap = ROWS_ELEMENTS_PER_WORD_CAP / linearGemvUnit(storage);
  let rows = 1;
  while (rows * 2 <= Math.min(m, cap)) rows *= 2;
  while (rows > 1 && n * Math.ceil(m / rows) < threadTarget) rows /= 2;
  return rows;
};

/**
 * 行ブロック変種の既定（`cols` / `unroll` は M=1 の既定と同じ・`rows` は格納と形と目標から）。
 * `threadTarget` の意味と既定は {@link linearGemvRowsForShape}。
 */
export const defaultLinearGemvRowsVariant = (
  storage: WeightStorage,
  m: number,
  n: number,
  threadTarget?: number,
): LinearGemvRowsVariant => ({
  ...defaultLinearGemvVariant(),
  rows: linearGemvRowsForShape(storage, m, n, threadTarget),
});

const assertVariant = (variant: LinearGemvVariant): void => {
  const { cols, unroll } = variant;
  if (!Number.isSafeInteger(cols) || cols < 1 || cols > MAX_THREADS) {
    throw new CodegenError(`linear_gemv: cols は 1..${MAX_THREADS} の整数（${cols}）`);
  }
  if (!Number.isSafeInteger(unroll) || unroll < 1) {
    throw new CodegenError(`linear_gemv: unroll は正整数（${unroll}）`);
  }
};

const assertRowsVariant = (variant: LinearGemvRowsVariant): void => {
  assertVariant(variant);
  const { rows } = variant;
  if (!Number.isSafeInteger(rows) || rows < 1 || rows > LINEAR_GEMV_MAX_ROWS) {
    throw new CodegenError(`linear_gemv: rows は 1..${LINEAR_GEMV_MAX_ROWS} の整数（${rows}）`);
  }
};

/**
 * group 長 → WGSL に焼く shift（i8 は group を持たないので `undefined`）。
 *
 * 2 冪 ≥ 16 と「格納と group 長は対」は {@link i4GroupShift}（宣言層と同じ導出点）が見る。
 * 本族はさらに **group ≥ {@link linearGemvUnit}** を要求する — 1 語ぶんの要素が group を跨ぐと
 * 語あたり 1 個の scale では足りず、黙って別の scale が掛かった値が出るため。
 */
const gemvGroupShift = (
  storage: WeightStorage,
  groupSize: number | undefined,
): number | undefined => {
  const unit = linearGemvUnit(storage);
  const shift = i4GroupShift("linear_gemv", storage, groupSize);
  // i8 は group を持たない（対の検査は i4GroupShift が済ませている）
  if (groupSize === undefined) return shift;
  if (shift === undefined || groupSize < unit) {
    throw new CodegenError(
      `linear_gemv: group_size ${groupSize} が ${unit} 以上の 2 冪でない` +
        `（1 語 = ${unit} 要素が group を跨ぐ）`,
    );
  }
  return shift;
};

/**
 * uniform の Dims（既定経路の `linearParams(m, n, k)` とバイト単位で同一 — 束縛レイアウトを
 * 分けない契約）。族固有なのは検査だけで、m / n / k の u32 域は {@link gemmParams} へ委譲する。
 *
 * MUST: k を **{@link linearGemvUnit} の倍数**に限る。WGSL の `units = dims.k / <刻み>u` は
 * 端数を切り捨てるので、外すと縮約が行の末尾を黙って落とした値を返す（例外は出ない）。
 * MUST: i4 では k を **group_size の倍数**にも限る（WGSL の `scale_base = col * (k >> shift)` が
 * 行あたりの scale 本数を割り算で導くため）。宣言層（ADR 0069 決定 2）と recipe-builder の
 * 適格判定が同じ条件を保証しているが、カーネル直呼びはそこを通らない。
 * MUST: m は 1..{@link LINEAR_GEMV_MAX_ROWS}（M=1 変種は 1 のみ・行ブロック変種は
 * `ceil(m / rows)` 枚の y タイルで受ける）。
 */
export const linearGemvParams = (
  storage: WeightStorage,
  m: number,
  n: number,
  k: number,
  groupSize?: number,
): Uint32Array<ArrayBuffer> => {
  const unit = linearGemvUnit(storage);
  gemvGroupShift(storage, groupSize);
  if ((storage === "f16" || storage === "f32") && m !== 1) {
    throw new CodegenError(`linear_gemv params: ${storage} 格納は m=1 のみ（${m}）`);
  }
  if (!Number.isSafeInteger(m) || m < 1 || m > LINEAR_GEMV_MAX_ROWS) {
    throw new CodegenError(
      `linear_gemv params: m は 1..${LINEAR_GEMV_MAX_ROWS} の整数（${m}）`,
    );
  }
  if (!Number.isSafeInteger(k) || k < 0 || k % unit !== 0) {
    throw new CodegenError(
      `linear_gemv params: k は ${unit} の倍数の非負整数（${k}）`,
    );
  }
  if (groupSize !== undefined && k % groupSize !== 0) {
    throw new CodegenError(
      `linear_gemv params: k=${k} が group_size ${groupSize} で割り切れない`,
    );
  }
  return gemmParams("linear", m, n, k);
};

/**
 * パイプラインキー（M=1 変種）。族名 `linear_gemv` が既定経路（`linear`）との判別子で、変種・格納・
 * group 長はどれも WGSL に焼かれるのでキーに載せる（同一キー → バイト同一 WGSL の codegen 決定性）。
 *
 * 格納判別子（`:wi4` / `:wi8`）と group 断片（`g32`）は weight-storage.ts の綴りをそのまま使う —
 * 診断・census が `:wi4g32` / `:wi8` で経路を識別する既存の読み方（ADR 0069 決定 5）に揃える。
 */
export const linearGemvKey = (
  storage: WeightStorage,
  groupSize?: number,
  variant: LinearGemvVariant = defaultLinearGemvVariant(),
): string => {
  assertVariant(variant);
  gemvGroupShift(storage, groupSize);
  return `linear_gemv:v1:f32:c${variant.cols}u${variant.unroll}${weightKeyPart(storage)}${
    i4GroupKeyPart(groupSize)
  }`;
};

/**
 * パイプラインキー（行ブロック変種）。M=1 変種のキーに `r<rows>` を足した形で、`rows` は
 * WGSL（アキュムレータ本数・y タイルの高さ）に焼かれるのでキーに載る。
 * NOTE: `r1` は M=1 変種とは別のテキスト（y タイル化された行ブロック）— 判別子の有無で区別する。
 */
export const linearGemvRowsKey = (
  storage: WeightStorage,
  groupSize: number | undefined,
  variant: LinearGemvRowsVariant,
): string => {
  assertRowsVariant(variant);
  assertRowsStorage(storage);
  gemvGroupShift(storage, groupSize);
  return `linear_gemv:v1:f32:c${variant.cols}u${variant.unroll}r${variant.rows}${
    weightKeyPart(storage)
  }${i4GroupKeyPart(groupSize)}`;
};

/**
 * 束縛。**既定経路の linear と同じ番号・同じ意味**（0 dims / 1 x / 2 w / 3 bias / 4 out /
 * 5 wscale）で、`#buildLinear` が組む束縛列をそのまま受ける。
 *
 * 変わるのは要素型 2 つだけ: 重みは `vec4<u32>`（16 B = i4 32 要素 / i8 16 要素を 1 度に読む）、
 * 出力は `f32`（1 スレッド 1 列のスカラ書き — 既定 v4 経路の `vec4<f32>` と違い n の整除を
 * 要らない）。
 */
const bindings = (unit: number, storage: WeightStorage): string =>
  `@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
// 行頭が 16 B 整列なのは k % ${unit} == 0 から（適格判定が保証する）
@group(0) @binding(2) var<storage, read> w: array<vec4<${storage === "f32" ? "f32" : "u32"}>>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
${
    storage === "f16" || storage === "f32"
      ? ""
      : "@group(0) @binding(5) var<storage, read> wscale: array<f32>;"
  }`;

/**
 * 語 1 本ぶんの読み（重み語 + x の quad 先頭 + i4 だけ group scale）。
 *
 * i8 の scale は出力チャネルごとで K ループ不変なので、ここではなく縮約の外で 1 度だけ束ねる
 * （{@link weightScaleWgsl} — ADR 0019 と同じ巻き上げ）。
 */
const unitLoads = (
  storage: WeightStorage,
  slot: string,
  unitExpr: string,
  shift: number | undefined,
): string => {
  const unit = linearGemvUnit(storage);
  const groupScale = storage === "i4"
    ? `
    let ws${slot} = wscale[scale_base + ((unit${slot} * ${unit}u) >> ${shift}u)];`
    : "";
  return `    let unit${slot} = ${unitExpr};
    let pw${slot} = w[row_base + unit${slot}];${groupScale}
    let xq${slot} = unit${slot} * ${unit / 4}u;`;
};

/**
 * 語 1 本（i4 32 要素）の積和展開（M=1 変種）。
 *
 * nibble の並びは weight-storage.ts の `dequant4` と同一（要素 2i = 下位 / 2i+1 = 上位・
 * 格納値 `u = q + 8` — 正本はエクスポータ `karume/emit.py: pack_int4`）。
 * MUST: 展開順は語内の要素昇順（成分 x→w × バイト x→w × 下位→上位 nibble）— これが
 * 「k 昇順の逐次」そのもので、崩すと既定経路とのビット同一が割れる。
 * MUST: x は `vec4<f32>` 束縛から**静的成分**で引く（動的成分添字は Metal でローカル領域へ
 * 落ちる — gemm.ts の `storeBTransposed` と同じ規律）。
 * MUST: このテキストは行ブロック化の前と 1 バイトも変えない（decode の生成物 —
 * tests/fixtures/wgsl/linear_gemv_*.wgsl が検出器）。
 */
const unitMacsI4 = (slot: string): string => {
  const lanes = ["x", "y", "z", "w"] as const;
  return lanes.map((component, quad) => {
    const bytes = `b${slot}_${quad}`;
    const xa = `xa${slot}_${quad}`;
    const xb = `xb${slot}_${quad}`;
    // 語の成分 `quad` は要素 8·quad..8·quad+7 = x の quad 2 本ぶん。
    const macs = lanes.flatMap((byte, lane) => {
      const source = lane < 2 ? xa : xb;
      const low = lanes[(lane * 2) % 4];
      const high = lanes[(lane * 2 + 1) % 4];
      return [
        `    acc = acc + ${source}.${low} * (f32(i32(${bytes}.${byte} & 0xFu) - 8) * ws${slot});`,
        `    acc = acc + ${source}.${high} * (f32(i32(${bytes}.${byte} >> 4u) - 8) * ws${slot});`,
      ];
    }).join("\n");
    return `    let ${bytes} = unpack4xU8(pw${slot}.${component});
    let ${xa} = x[xq${slot} + ${quad * 2}u];
    let ${xb} = x[xq${slot} + ${quad * 2 + 1}u];
${macs}`;
  }).join("\n");
};

/**
 * 語 1 本（i8 16 要素）の積和展開（M=1 変種）。
 *
 * レーンの並びは weight-storage.ts の `dequant4`（i8 quad 版 = `vec4<f32>(unpack4xI8(…)) * scale`）
 * と同一で、成分 `quad` の 4 要素がちょうど x の 1 quad に対応する（i4 の 2 quad と違う唯一の点）。
 * MUST: 展開順は語内の要素昇順（成分 x→w × レーン x→w）・字面は `f32(q) * ws` の要素ごと乗算。
 * MUST: x は `vec4<f32>` 束縛から**静的成分**で引く（i4 版と同じ Metal の規律）。
 */
const unitMacsI8 = (slot: string): string => {
  const lanes = ["x", "y", "z", "w"] as const;
  return lanes.map((component, quad) => {
    const bytes = `b${slot}_${quad}`;
    const xa = `xa${slot}_${quad}`;
    const macs = lanes.map((lane) =>
      `    acc = acc + ${xa}.${lane} * (f32(${bytes}.${lane}) * ${WEIGHT_SCALE_VAR});`
    ).join("\n");
    return `    let ${bytes} = unpack4xI8(pw${slot}.${component});
    let ${xa} = x[xq${slot} + ${quad}u];
${macs}`;
  }).join("\n");
};

/**
 * f16 8 要素 / 語。unpack2x16float は既定 GEMM と同じ復元で、scale は持たない。
 * MUST: 語の x→w、各対の下位→上位の順に積和する。活性の成分添字は静的に展開する。
 */
const unitMacsF16 = (slot: string): string => {
  const lanes = ["x", "y", "z", "w"] as const;
  return lanes.map((component, pair) => {
    const weights = `h${slot}_${pair}`;
    const activation = `x${slot}_${pair}`;
    return `    let ${weights} = unpack2x16float(pw${slot}.${component});
    let ${activation} = x[xq${slot} + ${Math.floor(pair / 2)}u];
    acc = acc + ${activation}.${lanes[(pair * 2) % 4]} * ${weights}.x;
    acc = acc + ${activation}.${lanes[(pair * 2 + 1) % 4]} * ${weights}.y;`;
  }).join("\n");
};

/** f32 4 要素 / 語。通常 GEMM と同じ K 昇順の積和を静的成分で展開する。 */
const unitMacsF32 = (slot: string): string => {
  const macs = ["x", "y", "z", "w"].map((lane) =>
    `    acc = acc + xf${slot}.${lane} * pw${slot}.${lane};`
  ).join("\n");
  return `    let xf${slot} = x[xq${slot}];
${macs}`;
};

const unitMacs = (storage: WeightStorage, slot: string): string =>
  storage === "f32"
    ? unitMacsF32(slot)
    : storage === "f16"
    ? unitMacsF16(slot)
    : storage === "i4"
    ? unitMacsI4(slot)
    : unitMacsI8(slot);

/**
 * 語 1 本の積和展開（行ブロック変種・`rows` 行）。
 *
 * 逆量子化した値 `d = f32(q) * scale`（i4 は `f32(i32(u) − 8) * ws`）を要素ごとに 1 度だけ `let` に
 * 置き、行ごとに `acc<r> = acc<r> + x * d` を語内の要素昇順で書く。1 出力要素あたりの縮約順・
 * 積の対応（x の要素 × その要素の逆量子化値）・bias の足し順は M=1 変種と同じで、違うのは
 * 「逆量子化の乗算を行間で共有する」ことだけ — 丸め点は同じ 1 個の f32 乗算で、既定経路との
 * u32 完全一致は実測（門 = tests/gpu_linear_gemv_test.ts・掃引 6,000 組超 — research
 * 2026-09-07-gemv-rows-k21）。
 *
 * WHY 巻き上げ（行ごとに M=1 と同じ字面を書かない）: 生成テキストの量がそのままシェーダの
 * コンパイル費になる（naga の解析・検証がテキスト量に超線形 — i4 8 行で 100 KB / ≈200 ms が
 * 73 KB / ≈55 ms、研究ノート §6）。速度は行ごとに書く形と同等（掃引で差はノイズ帯）。
 * MUST: x は `vec4<f32>` 束縛から**静的成分**で引く（M=1 変種と同じ Metal の規律）。行の添字
 * `xr<r>` は配列添字であって成分添字ではない。
 */
const rowsMacsI4 = (slot: string, rows: number): string => {
  const lanes = ["x", "y", "z", "w"] as const;
  return lanes.map((component, quad) => {
    const bytes = `b${slot}_${quad}`;
    // 要素 e = 2·byte + nibble（0..7）。d の添字は要素の並びそのもの。
    const dequant = lanes.flatMap((byte, lane) => [
      `    let d${slot}_${quad}_${lane * 2} = f32(i32(${bytes}.${byte} & 0xFu) - 8) * ws${slot};`,
      `    let d${slot}_${quad}_${
        lane * 2 + 1
      } = f32(i32(${bytes}.${byte} >> 4u) - 8) * ws${slot};`,
    ]).join("\n");
    const perRow = Array.from({ length: rows }, (_, row) => {
      const xa = `xa${slot}_${quad}_${row}`;
      const xb = `xb${slot}_${quad}_${row}`;
      // 語の成分 `quad` は要素 8·quad..8·quad+7 = x の quad 2 本ぶん（要素 e は quad 2 本の 8 成分に順に対応）。
      const macs = Array.from(
        { length: 8 },
        (_, element) =>
          `    acc${row} = acc${row} + ${element < 4 ? xa : xb}.${
            lanes[element % 4]
          } * d${slot}_${quad}_${element};`,
      ).join("\n");
      return `    let ${xa} = x[xr${row} + xq${slot} + ${quad * 2}u];
    let ${xb} = x[xr${row} + xq${slot} + ${quad * 2 + 1}u];
${macs}`;
    }).join("\n");
    return `    let ${bytes} = unpack4xU8(pw${slot}.${component});
${dequant}
${perRow}`;
  }).join("\n");
};

/** i8 版（成分 `quad` の 4 要素 = x の 1 quad・scale は列ごと 1 本 — {@link unitMacsI8} と同じ対応）。 */
const rowsMacsI8 = (slot: string, rows: number): string => {
  const lanes = ["x", "y", "z", "w"] as const;
  return lanes.map((component, quad) => {
    const bytes = `b${slot}_${quad}`;
    const dequant = lanes.map((lane) =>
      `    let d${slot}_${quad}_${lane} = f32(${bytes}.${lane}) * ${WEIGHT_SCALE_VAR};`
    ).join("\n");
    const perRow = Array.from({ length: rows }, (_, row) => {
      const xa = `xa${slot}_${quad}_${row}`;
      const macs = lanes.map((lane) =>
        `    acc${row} = acc${row} + ${xa}.${lane} * d${slot}_${quad}_${lane};`
      ).join("\n");
      return `    let ${xa} = x[xr${row} + xq${slot} + ${quad}u];
${macs}`;
    }).join("\n");
    return `    let ${bytes} = unpack4xI8(pw${slot}.${component});
${dequant}
${perRow}`;
  }).join("\n");
};

const rowsMacs = (storage: WeightStorage, slot: string, rows: number): string =>
  storage === "i4" ? rowsMacsI4(slot, rows) : rowsMacsI8(slot, rows);

/** i4 は行あたりの scale 本数から group の先頭を導く / i8 は出力チャネル 1 本を巻き上げる。 */
const scaleSetupWgsl = (storage: WeightStorage, shift: number | undefined): string =>
  storage === "i4"
    ? `
  let scale_base = col * (dims.k >> ${shift}u);`
    : weightScaleWgsl(storage, "col", "  ");

/**
 * GEMV の WGSL（`out[n] = x[k] · wᵀ[n,k] + bias[n]`・M=1・重み f16 / i4 / i8 格納）。
 *
 * 1 スレッドが 1 出力列の縮約を丸ごと持つので、並列度の上限は `n`。これはビット同一の代償
 * そのもので、k 方向へ割れば並列度は上がるが縮約順が動く（MUST NOT — モジュール doc）。
 */
export const linearGemvWgsl = (
  storage: WeightStorage,
  groupSize?: number,
  variant: LinearGemvVariant = defaultLinearGemvVariant(),
): string => {
  assertVariant(variant);
  const unit = linearGemvUnit(storage);
  const shift = gemvGroupShift(storage, groupSize);
  const { cols, unroll } = variant;
  const slots = Array.from({ length: unroll }, (_, slot) => `${slot}`);
  const loads = slots.map((slot) => unitLoads(storage, slot, `unit + ${slot}u`, shift)).join("\n");
  const macs = slots.map((slot) => unitMacs(storage, slot)).join("\n");
  return `// karume linear gemv (M=1: out[n] = x[k] · wᵀ[n,k] + bias[n], f32${
    weightNote(storage)
  }, ${cols} 列 / wg, 語 ${unroll} 本先読み)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
${bindings(unit, storage)}

@compute @workgroup_size(${cols})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let col = gid.x;
  // 共有メモリも barrier も持たないので、端の workgroup は早期 return してよい
  if (col >= dims.n) {
    return;
  }
  let units = dims.k / ${unit}u;
  let row_base = col * units;${scaleSetupWgsl(storage, shift)}
  var acc = 0.0;
  var unit = 0u;
  // 先読みぶんの重み語を**先に**全て発行してから積和へ入る（メモリ並列度）。語の処理順は
  // 昇順のままなので、1 出力要素あたりの加算順序は先読み本数によらず同じ
  for (; unit + ${unroll}u <= units; unit = unit + ${unroll}u) {
${loads}
${macs}
  }
  // 端数の語（units % ${unroll} 本）— 上と同じ順序を 1 語ずつ辿る
  for (; unit < units; unit = unit + 1u) {
${unitLoads(storage, "t", "unit", shift)}
${unitMacs(storage, "t")}
  }
  out[col] = acc + bias[col];
}
`;
};

/**
 * 行ブロック GEMV の WGSL（`out[m,n] = x[m,k] · wᵀ[n,k] + bias[n]`・M ≥ 2・重み i4 / i8 格納）。
 *
 * dispatch は `[ceil(n / cols), ceil(m / rows), 1]`。1 スレッドは列 `gid.x` × 行
 * `gid.y · rows .. +rows` を持ち、重み語と scale は 1 回だけ読んで行ごとの x と積和する。
 * MUST: m を超える行は **x の最終行を読み直す**（読みを束縛の範囲内に保つため）が、
 * **書き戻しは `row < m` の行だけ**。出力の形は `[m, n]` で、行 r の列 c は `out[r·n + c]`。
 * MUST: `dims.m ≥ 1` が前提（`min(row0 + r, dims.m - 1u)` は m = 0 で u32 が巻き戻る）。
 * 担保は {@link linearGemvParams} の m 域検査（1..{@link LINEAR_GEMV_MAX_ROWS}）— WGSL 側は検査しない。
 * MUST: 縮約の字面・順序は M=1 変種（{@link linearGemvWgsl}）と同一（モジュール doc の数値契約）。
 */
export const linearGemvRowsWgsl = (
  storage: WeightStorage,
  groupSize: number | undefined,
  variant: LinearGemvRowsVariant,
): string => {
  assertRowsVariant(variant);
  assertRowsStorage(storage);
  const unit = linearGemvUnit(storage);
  const shift = gemvGroupShift(storage, groupSize);
  const { cols, unroll, rows } = variant;
  const slots = Array.from({ length: unroll }, (_, slot) => `${slot}`);
  const rowList = Array.from({ length: rows }, (_, row) => row);
  const loads = slots.map((slot) => unitLoads(storage, slot, `unit + ${slot}u`, shift)).join("\n");
  const macs = slots.map((slot) => rowsMacs(storage, slot, rows)).join("\n");
  const rowHeads = rowList.map((row) => `  let xr${row} = min(row0 + ${row}u, dims.m - 1u) * kq;`)
    .join("\n");
  const accs = rowList.map((row) => `  var acc${row} = 0.0;`).join("\n");
  const stores = rowList.map((row) =>
    `  if (row0 + ${row}u < dims.m) {
    out[(row0 + ${row}u) * dims.n + col] = acc${row} + bias[col];
  }`
  ).join("\n");
  return `// karume linear gemv rows (M≥2: out[m,n] = x[m,k] · wᵀ[n,k] + bias[n], f32${
    weightNote(storage)
  }, ${cols} 列 / wg, 語 ${unroll} 本先読み, ${rows} 行 / スレッド)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
${bindings(unit, storage)}

@compute @workgroup_size(${cols})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let col = gid.x;
  // 共有メモリも barrier も持たないので、端の workgroup は早期 return してよい
  if (col >= dims.n) {
    return;
  }
  let units = dims.k / ${unit}u;
  let row_base = col * units;
  // 行ブロックの先頭行と、行ごとの x の先頭（quad 単位）。m を超える行は最終行を読み直す
  // （読みを範囲内に保つだけで、書き戻しは行数の内側だけ）
  let kq = dims.k / 4u;
  let row0 = gid.y * ${rows}u;
${rowHeads}${scaleSetupWgsl(storage, shift)}
${accs}
  var unit = 0u;
  // 先読みぶんの重み語を**先に**全て発行してから積和へ入る（メモリ並列度）。語の処理順は
  // 昇順のままなので、1 出力要素あたりの加算順序は先読み本数によらず同じ
  for (; unit + ${unroll}u <= units; unit = unit + ${unroll}u) {
${loads}
${macs}
  }
  // 端数の語（units % ${unroll} 本）— 上と同じ順序を 1 語ずつ辿る
  for (; unit < units; unit = unit + 1u) {
${unitLoads(storage, "t", "unit", shift)}
${rowsMacs(storage, "t", rows)}
  }
${stores}
}
`;
};
