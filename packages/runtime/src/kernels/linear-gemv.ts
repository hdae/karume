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
 * 対応・bias の足し順は M=1 と同一。行ブロックでは語ごとの関数を行別に呼ぶが、
 * 復元の丸め点は同じ 1 個の f32 乗算を保つ（ADR 0082 追記 8・9）。
 * MUST: 先読み（`unroll`）は「語をまとめて読む」だけで、**積和の順序は語の昇順のまま**。
 * 語をまたいで積和を混ぜると 1 出力要素あたりの加算順が動き、契約が割れる。
 * NOTE: f32 の縮約に順序非依存の理論保証は無いので、ビット同一は gemm-geometry と同じく
 * **実測命題**（門 = tests/gpu_linear_gemv_test.ts の u32 完全一致）。
 *
 * MUST NOT: k を分割して部分和を足し直す（split-K）形は縮約順が変わるため、この族には
 * 入れない。別の任意指定カーネル `linearGemvParallelWgsl` は ADR 0098 に従う。
 *
 * ## 幾何との関係（gemm-geometry の「唯一の選択点」MUST の射程）
 *
 * gemm-geometry は「担当割りの選択点は `gemmGeometryForRows` 1 箇所」と書くが、その射程は
 * **GEMM 骨格の中**にとどまる（ADR 0082 決定 2）。本族は骨格を共有しない別カーネルで、選択は
 * 2 段 — **族の選択**（GEMV 族に入るか）は `buildLinear` の 1 箇所、**変種の選択**（M=1 変種か
 * 行ブロックか・行ブロックの `rows`）は {@link defaultLinearGemvRowsVariant} の 1 箇所 — で、
 * どちらも**プラン時 shape の純関数**。判別子はパイプラインキーの族名 `linear_gemv`（行ブロックは
 * 変種の `r<rows>`）に載る — 実行時オートチューン禁止と「同一キー → バイト同一 WGSL」は
 * どちらも保たれる。
 */

import { CodegenError } from "../codegen/errors.ts";
import { gemmParams, gemmUsesVec4 } from "./gemm.ts";
import { staticQuantizeParams } from "./static-quantize.ts";
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
 * 適格判定（src/runtime/recipe-builders/linear.ts の `buildLinear`）が k と group 長へ課す整除の
 * 単位でもあるので、門とカーネルが格納ごとに同じ 1 個の導出点を読む。
 * f16 は 8 要素 / 語、f32 は 4 要素 / 語で、両者とも M=1 のみ（ADR 0082 追記 6・7）。
 */
export const linearGemvUnit = (storage: WeightStorage): number => {
  if (storage === "i2") return 64;
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
 * §7）。M=1の実測形を渡す場合は追記10の語彙INT8をc16へ選択する。
 * MUST: 既定の変更はビット同一門（tests/gpu_linear_gemv_test.ts）の再実測とセット。
 */
export const defaultLinearGemvVariant = (
  shape?: { readonly storage: WeightStorage; readonly n: number; readonly k: number },
): LinearGemvVariant => ({
  // DECIDED: 大語彙INT8の実測形だけ担当列数を減らす（ADR 0082 追記10）。
  // docs/decisions/0082-linear-gemv-decode.md
  cols: shape?.storage === "i8" && shape.n === 262144 && shape.k === 1536 ? 16 : 32,
  unroll: 4,
});

/**
 * 行ブロック変種が受ける M の上限 = **門の上限**（`buildLinear` が本族へ入れる行数の範囲）。
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
  return `linear_gemv:v2:f32:c${variant.cols}u${variant.unroll}r${variant.rows}${
    weightKeyPart(storage)
  }${i4GroupKeyPart(groupSize)}`;
};

/**
 * 束縛。**既定経路の linear と同じ番号・同じ意味**（0 dims / 1 x / 2 w / 3 bias / 4 out /
 * 5 wscale）で、`buildLinear` が組む束縛列をそのまま受ける。
 *
 * 変わるのは要素型 2 つだけ: 重みは `vec4<u32>`（16 B = i4 32 要素 / i8 16 要素を 1 度に読む）、
 * 出力は `f32`（1 スレッド 1 列のスカラ書き — 既定 v4 経路の `vec4<f32>` と違い n の整除を
 * 要らない）。
 */
const bindings = (
  unit: number,
  storage: WeightStorage,
  quantize = false,
  packed = false,
): string =>
  `@group(0) @binding(1) var<storage, read> x: array<vec4<${packed ? "u32" : "f32"}>>;
// 行頭が 16 B 整列なのは k % ${unit} == 0 から（適格判定が保証する）
@group(0) @binding(2) var<storage, read> w: array<vec4<${storage === "f32" ? "f32" : "u32"}>>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<${quantize ? "u32" : "f32"}>;
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
  inputBase = "",
  packed = false,
): string => {
  const unit = linearGemvUnit(storage);
  const groupScale = storage === "i4"
    ? `
    let ws${slot} = wscale[scale_base + ((unit${slot} * ${unit}u) >> ${shift}u)];`
    : "";
  // packed 活性は 1 束縛要素（`vec4<u32>`）が 16 要素を運ぶ（f32 の quad は 4 要素）。
  return `    let unit${slot} = ${unitExpr};
    let pw${slot} = w[row_base + unit${slot}];${groupScale}
    let xq${slot} = ${inputBase}unit${slot} * ${unit / (packed ? 16 : 4)}u;`;
};

/** 語内の成分名（`vec4` の静的添字 — Metal の動的添字を避ける規律）。 */
const LANES = ["x", "y", "z", "w"] as const;

/**
 * 活性 1 quad（4 要素）を読む 1 行。既定は `vec4<f32>` の quad をそのまま引く。
 *
 * packed 活性（ADR 0105）では束縛が `array<vec4<u32>>` になり、**16 要素 = 1 本**で読める。
 * 語は 4 quad に 1 度だけ読み、quad ごとに `unpack4xI8` で int8 コードへ戻してから
 * `f32(code) * x_scale` で f32 値を作る。これで 1 重み語あたりの活性ロードが
 * i2 は 16 → 4 本・i4 は 8 → 2 本・i8 は 4 → 1 本になる（research 2026-09-19 §14 の律速）。
 *
 * MUST: 復元の字面は `vec4<f32>(unpack4xI8(…)) * dims.x_scale` = **要素ごとの f32 1 乗算**。
 * 生産側 SRQ の出力値表（static-quantize.ts の `params[129 + level]` = `Math.fround(level*scale)`）
 * と要素ごとに u32 一致する — level ≤ 128 と f32 scale の積は f64 で厳密なので、表側も
 * この乗算も「厳密な積を正しく丸めた f32」に一致する（ADR 0105 の決定 3）。
 * MUST: 語の成分は静的添字（`.x` / `.y` / `.z` / `.w`）で引く（既定経路と同じ Metal の規律）。
 * NOTE: 復元値に丸め障壁は置かない。Metal で f32 経路と割れた原因は積和の fma 縮約の入れ方で、
 * 並列族の積和を明示 `fma()` で綴ることで両経路が揃う（{@link mac} — ADR 0105 追記 4）。
 */
const activationQuad = (packed: boolean, slot: string, name: string, quad: number): string => {
  if (!packed) return `    let ${name} = x[xq${slot} + ${quad}u];`;
  const word = quad >> 2;
  const load = quad % 4 === 0 ? `    let xp${slot}_${word} = x[xq${slot} + ${word}u];\n` : "";
  return `${load}    let ${name} = vec4<f32>(unpack4xI8(xp${slot}_${word}.${
    LANES[quad & 3]
  })) * dims.x_scale;`;
};

/**
 * 積和 1 行。並列族（`parallelWgsl` の f32 / packed × 融合なし / あり）は明示 `fma()` で綴る。
 *
 * MUST: 並列族は `fma`。`acc + x * d` の綴りは fma への縮約をコンパイラに委ねる形で、RTX / Vulkan
 * では常に縮約される（明示 fma と u32 同一 — 540 組の掃引）が、Metal（M2）は式形ごとに縮約の入れ方を
 * 変え、同じ数式のカーネル 2 本（f32 と packed）が u32 で割れた。明示 fma は縮約の自由度そのものを
 * 消すので、両経路が同じ丸めになる（ADR 0105 追記 4）。逐次 GEMV・行ブロック・subgroup 変種は
 * 参照経路の数値を動かさないため従来の綴りのまま。
 */
const mac = (fma: boolean, x: string, d: string): string =>
  fma ? `    acc = fma(${x}, ${d}, acc);` : `    acc = acc + ${x} * ${d};`;

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
const unitMacsI4 = (slot: string, packed = false, fma = false): string => {
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
        mac(fma, `${source}.${low}`, `(f32(i32(${bytes}.${byte} & 0xFu) - 8) * ws${slot})`),
        mac(fma, `${source}.${high}`, `(f32(i32(${bytes}.${byte} >> 4u) - 8) * ws${slot})`),
      ];
    }).join("\n");
    return `    let ${bytes} = unpack4xU8(pw${slot}.${component});
${activationQuad(packed, slot, xa, quad * 2)}
${activationQuad(packed, slot, xb, quad * 2 + 1)}
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
const unitMacsI8 = (slot: string, packed = false, fma = false): string => {
  const lanes = ["x", "y", "z", "w"] as const;
  return lanes.map((component, quad) => {
    const bytes = `b${slot}_${quad}`;
    const xa = `xa${slot}_${quad}`;
    const macs = lanes.map((lane) =>
      mac(fma, `${xa}.${lane}`, `(f32(${bytes}.${lane}) * ${WEIGHT_SCALE_VAR})`)
    ).join("\n");
    return `    let ${bytes} = unpack4xI8(pw${slot}.${component});
${activationQuad(packed, slot, xa, quad)}
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

/** INT2 の16 B語を K 昇順に積和する。成分添字を静的にして Metal の動的添字を避ける。 */
const unitMacsI2 = (slot: string, packed = false, fma = false): string => {
  const lanes = ["x", "y", "z", "w"] as const;
  return lanes.flatMap((component, word) =>
    lanes.map((_, byte) => {
      const index = word * 4 + byte;
      const quantized = `q${slot}_${index}`;
      const decoded = `d${slot}_${index}`;
      const activation = `x${slot}_${index}_0`;
      const products = `${activationQuad(packed, slot, activation, index)}
${lanes.map((lane) => mac(fma, `${activation}.${lane}`, `${decoded}.${lane}`)).join("\n")}`;
      return `    let ${quantized} = (pw${slot}.${component} >> ${byte * 8}u) & 255u;
    let ${decoded} = vec4<f32>(vec4<i32>(vec4<u32>(${quantized}, ${quantized} >> 2u, ${quantized} >> 4u, ${quantized} >> 6u) & vec4<u32>(3u)) - vec4<i32>(2)) * ${WEIGHT_SCALE_VAR};
${products}`;
    })
  ).join("\n");
};

/**
 * INT2 の行ブロックだけを関数にまとめ、語・行ごとの展開による初回の解析費を減らす。
 * 1 出力の K 昇順、成分の静的添字、復元と積和の丸め点は M=1 と同じ。
 * DECIDED: docs/decisions/0082-linear-gemv-decode.md#追記-82026-09-11-int2-行ブロックのシェーダーを縮小するk-30
 */
const i2WordWgsl = (): string => `
fn linear_i2_word(pwa: vec4<u32>, xqa: u32, wscale_v: f32, initial: f32) -> f32 {
  var acc = initial;
${unitMacsI2("a")}
  return acc;
}
`;

const rowsMacsI2 = (slot: string, rows: number): string =>
  Array.from(
    { length: rows },
    (_, row) =>
      `    acc${row} = linear_i2_word(pw${slot}, xr${row} + xq${slot}, wscale_v, acc${row});`,
  ).join("\n");

const unitMacs = (storage: WeightStorage, slot: string, packed = false, fma = false): string =>
  storage === "f32"
    ? unitMacsF32(slot)
    : storage === "f16"
    ? unitMacsF16(slot)
    : storage === "i4"
    ? unitMacsI4(slot, packed, fma)
    : storage === "i2"
    ? unitMacsI2(slot, packed, fma)
    : unitMacsI8(slot, packed, fma);

/**
 * I4/I8も行ごとの積和を小さな関数へまとめる。K昇順とf32の丸め点はM=1と同じ。
 * 逆量子化の式を行ごとに複製する代わりに関数を呼び、WGSLの解析・コンパイル量を減らす。
 * DECIDED: docs/decisions/0082-linear-gemv-decode.md#追記-92026-09-11-int4int8-行ブロックにも関数化を適用する
 */
const rowsWordCall = (storage: "i4" | "i8", slot: string, rows: number): string =>
  Array.from({ length: rows }, (_, row) => {
    const scale = storage === "i4" ? `ws${slot}` : "wscale_v";
    return `    acc${row} = linear_${storage}_word(pw${slot}, xr${row} + xq${slot}, ${scale}, acc${row});`;
  }).join("\n");

const wordHelperWgsl = (storage: WeightStorage): string => {
  if (storage === "i2") return i2WordWgsl();
  if (storage === "f16" || storage === "f32") return "";
  const scale = storage === "i4" ? "wsa" : "wscale_v";
  return `
fn linear_${storage}_word(pwa: vec4<u32>, xqa: u32, ${scale}: f32, initial: f32) -> f32 {
  var acc = initial;
${storage === "i4" ? unitMacsI4("a") : unitMacsI8("a")}
  return acc;
}
`;
};

const rowsMacs = (storage: WeightStorage, slot: string, rows: number): string =>
  storage === "i4"
    ? rowsWordCall("i4", slot, rows)
    : storage === "i2"
    ? rowsMacsI2(slot, rows)
    : rowsWordCall("i8", slot, rows);

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
 * 行ブロック GEMV の WGSL（`out[m,n] = x[m,k] · wᵀ[n,k] + bias[n]`・M ≥ 2・重み i2 / i4 / i8 格納）。
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
${bindings(unit, storage)}${wordHelperWgsl(storage)}

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

/** K を分担する lane 数。1 workgroup は常に128 thread（共有部分和512 byte）。 */
export type LinearGemvParallelLanes = 2 | 4 | 8 | 16 | 32;

type ParallelShape = {
  readonly storage: WeightStorage;
  readonly n: number;
  readonly k: number;
  readonly group?: number;
  readonly lanes: LinearGemvParallelLanes;
  /**
   * packed int8 活性（ADR 0105）で受け取る形か。**実測で速くなった行だけ true**。
   *
   * 効くのは「K が長く lanes 32」= 1 スレッドが 1 重み語あたりに読む活性が多い形だけで、
   * K=1536 の lanes 2 / 4 と i8 は追加の unpack / 変換 / 乗算が利得を食い潰す
   * （ADR 0105 追記 1 の per-key 実測）。
   */
  readonly packedActivations: boolean;
};

// DECIDED: 実測した形だけを任意指定の対象にする。GPU名による自動選択ではない。
// docs/decisions/0098-linear-gemv-parallel.md
const PARALLEL_SHAPES: readonly ParallelShape[] = [
  { storage: "i4", n: 8960, k: 1536, group: 32, lanes: 4, packedActivations: false },
  { storage: "i4", n: 2048, k: 1536, group: 32, lanes: 4, packedActivations: false },
  { storage: "i4", n: 256, k: 1536, group: 32, lanes: 32, packedActivations: false },
  { storage: "i4", n: 1536, k: 2048, group: 32, lanes: 32, packedActivations: false },
  { storage: "i4", n: 6144, k: 1536, group: 32, lanes: 4, packedActivations: false },
  { storage: "i4", n: 1536, k: 6144, group: 32, lanes: 32, packedActivations: false },
  { storage: "i4", n: 1536, k: 256, group: 32, lanes: 4, packedActivations: false },
  { storage: "i4", n: 4096, k: 1536, group: 32, lanes: 4, packedActivations: false },
  { storage: "i4", n: 512, k: 1536, group: 32, lanes: 32, packedActivations: false },
  { storage: "i4", n: 1536, k: 4096, group: 32, lanes: 32, packedActivations: false },
  { storage: "i4", n: 12288, k: 1536, group: 32, lanes: 4, packedActivations: false },
  { storage: "i4", n: 1536, k: 12288, group: 32, lanes: 32, packedActivations: false },
  { storage: "i8", n: 262144, k: 1536, lanes: 16, packedActivations: false },
  { storage: "i4", n: 2048, k: 1536, group: 512, lanes: 4, packedActivations: false },
  { storage: "i4", n: 256, k: 1536, group: 512, lanes: 32, packedActivations: false },
  { storage: "i4", n: 1536, k: 2048, group: 2048, lanes: 32, packedActivations: true },
  { storage: "i4", n: 6144, k: 1536, group: 512, lanes: 4, packedActivations: false },
  { storage: "i4", n: 1536, k: 6144, group: 2048, lanes: 32, packedActivations: true },
  { storage: "i8", n: 256, k: 1536, lanes: 32, packedActivations: false },
  { storage: "i8", n: 1536, k: 256, lanes: 4, packedActivations: false },
  { storage: "i4", n: 4096, k: 1536, group: 512, lanes: 4, packedActivations: false },
  { storage: "i4", n: 512, k: 1536, group: 512, lanes: 32, packedActivations: false },
  { storage: "i4", n: 1536, k: 4096, group: 4096, lanes: 32, packedActivations: true },
  { storage: "i2", n: 12288, k: 1536, lanes: 2, packedActivations: false },
  { storage: "i2", n: 1536, k: 12288, lanes: 32, packedActivations: true },
];

/** 実測表の行引き（M の範囲も含めて 1 箇所）。 */
const parallelShapeFor = (
  storage: WeightStorage,
  m: number,
  n: number,
  k: number,
  group?: number,
): ParallelShape | undefined => {
  if (m < 1 || m > 8) return undefined;
  return PARALLEL_SHAPES.find((shape) =>
    shape.storage === storage && shape.n === n && shape.k === k && shape.group === group
  );
};

/** 同じ形の M=1/4/8 は同じ加算順。M>8 の prefill は既存の行ブロックを維持する。 */
export const linearGemvParallelLanes = (
  storage: WeightStorage,
  m: number,
  n: number,
  k: number,
  group?: number,
): LinearGemvParallelLanes | undefined => parallelShapeFor(storage, m, n, k, group)?.lanes;

/**
 * 「この形が並列 GEMV（{@link linearGemvParallelWgsl} 族）へ落ちるか」を返す**唯一の純関数**。
 *
 * {@link linearGemvParallelLanes}（実測形の表引き）に、recipe-builders/linear.ts の `buildLinear` が
 * GEMV 族へ入れる条件（格納・`k % 刻み`・i4 の group 長・出力 vec4 の実測範囲）を重ねたもの。
 * packed 活性の対付け（ADR 0105）はこれをさらに行ごとの採否で絞った
 * {@link linearGemvPackedEligible} を読む。
 */
export const linearGemvParallelEligible = (
  storage: WeightStorage,
  m: number,
  n: number,
  k: number,
  group?: number,
): LinearGemvParallelLanes | undefined => {
  if (storage !== "i2" && storage !== "i4" && storage !== "i8") return undefined;
  if (!gemmUsesVec4(k, n)) return undefined;
  if (k <= 0 || k % linearGemvUnit(storage) !== 0) return undefined;
  if (storage === "i4") {
    if (group === undefined || group % linearGemvUnit("i4") !== 0) return undefined;
  } else if (group !== undefined) return undefined;
  return linearGemvParallelLanes(storage, m, n, k, group);
};

/**
 * 「この形を **packed int8 活性**（ADR 0105）で受け取るか」を返す**唯一の純関数**。
 *
 * {@link linearGemvParallelEligible}（並列 GEMV へ落ちるか）に、実測表の行の
 * `packedActivations` を重ねたもの。packed 変種の WGSL・params は全形ぶん生成できる
 * （テストとスナップショットは全形を維持する）が、**製品の plan が選ぶのはこの述語が
 * lane を返す形だけ**。
 *
 * MUST: 対付け（fusion.ts）と recipe-builder の門が同じ 1 本を読む。別に持つと、片方だけ
 * 広いときに `vec4<u32>` 束縛へ f32 の語を流す形（例外なしの沈黙誤値）が出る。
 */
export const linearGemvPackedEligible = (
  storage: WeightStorage,
  m: number,
  n: number,
  k: number,
  group?: number,
): LinearGemvParallelLanes | undefined => {
  const lanes = linearGemvParallelEligible(storage, m, n, k, group);
  if (lanes === undefined) return undefined;
  return parallelShapeFor(storage, m, n, k, group)?.packedActivations === true ? lanes : undefined;
};

export const linearGemvParallelKey = (
  storage: WeightStorage,
  group: number | undefined,
  lanes: LinearGemvParallelLanes,
): string => `linear_gemv_parallel${weightKeyPart(storage)}${i4GroupKeyPart(group)}:l${lanes}`;

/**
 * 圧縮語を K の lane に巡回配分する任意指定版。逆量子化と語内の積和は逐次版と共有するが、
 * 語をまたぐ加算順は異なる。128 thread 内の木縮約だけで完結し、追加 feature を要求しない。
 * M を shader に焼かず、decode と少数行の投機検証を同一キー・同一算術にする。
 * DECIDED: docs/decisions/0098-linear-gemv-parallel.md
 */
export const linearGemvParallelWgsl = (
  storage: WeightStorage,
  group: number | undefined,
  lanes: LinearGemvParallelLanes,
): string => parallelWgsl(storage, group, lanes, false);

/** 固定SRQまで融合する別キー。参照のparallel本文はバイト単位で維持する（ADR 0103）。 */
export const linearGemvStaticQuantizeKey = (
  storage: WeightStorage,
  group: number | undefined,
  lanes: LinearGemvParallelLanes,
): string => `${linearGemvParallelKey(storage, group, lanes)}:static-quantize:v1`;

export const linearGemvStaticQuantizeParams = (
  storage: WeightStorage,
  m: number,
  n: number,
  k: number,
  scale: number,
  group?: number,
): Uint32Array<ArrayBuffer> => {
  if (m < 1 || m > 8) throw new CodegenError("linear static_quantize: mは1..8のみ対応");
  assertRowsStorage(storage);
  const params = new Uint32Array(264);
  params.set(linearGemvParams(storage, m, n, k, group));
  // 第4語は実行時の丸め障壁。0とのXORを定数式へ置き換えない。
  params.set(staticQuantizeParams(m * n, scale), 4);
  return params;
};

export const linearGemvStaticQuantizeWgsl = (
  storage: WeightStorage,
  group: number | undefined,
  lanes: LinearGemvParallelLanes,
): string => parallelWgsl(storage, group, lanes, true);

/**
 * packed int8 活性の変種（ADR 0105）を表すキー断片。**既存キーの末尾に足す**ので、診断・census が
 * 読んでいる格納判別子（`:wi4g512` / `:l4` / `:static-quantize:v1`）の位置は 1 つも動かない。
 */
const PACKED_ACTIVATION_KEY_PART = ":packed-x-i8";

export const linearGemvParallelPackedKey = (
  storage: WeightStorage,
  group: number | undefined,
  lanes: LinearGemvParallelLanes,
): string => `${linearGemvParallelKey(storage, group, lanes)}${PACKED_ACTIVATION_KEY_PART}`;

/**
 * 並列 GEMV の packed 活性変種。幾何・lane 表・縮約木・bias の足し順は
 * {@link linearGemvParallelWgsl} と 1 バイトも変えず、**活性の読みと復元だけ**が違う。
 */
export const linearGemvParallelPackedWgsl = (
  storage: WeightStorage,
  group: number | undefined,
  lanes: LinearGemvParallelLanes,
): string => parallelWgsl(storage, group, lanes, false, true);

export const linearGemvStaticQuantizePackedKey = (
  storage: WeightStorage,
  group: number | undefined,
  lanes: LinearGemvParallelLanes,
): string => `${linearGemvStaticQuantizeKey(storage, group, lanes)}${PACKED_ACTIVATION_KEY_PART}`;

/** SRQ 融合エピローグ付き並列 GEMV の packed 活性変種（出力側 SRQ は f32 のまま）。 */
export const linearGemvStaticQuantizePackedWgsl = (
  storage: WeightStorage,
  group: number | undefined,
  lanes: LinearGemvParallelLanes,
): string => parallelWgsl(storage, group, lanes, true, true);

/**
 * packed 活性の復元 scale（= 生産側 SRQ の f32 scale）を u32 の bit 列にする。
 *
 * MUST: 正・有限・厳密に f32 表現できる値（生産側 {@link staticQuantizeParams} と同じ門）。
 * 0 は恒等 SRQ で packed int8 に落とせないので、ここでも拒否する（fail loudly）。
 */
const activationScaleBits = (scale: number): number => {
  if (!Number.isFinite(scale) || scale <= 0 || Math.fround(scale) !== scale) {
    throw new CodegenError(
      `linear_gemv packed: 活性 scale ${scale} は正・有限で厳密に f32 表現できる値が必要`,
    );
  }
  const scratch = new Float32Array(1);
  scratch[0] = scale;
  return new Uint32Array(scratch.buffer)[0];
};

/** packed 活性変種の params（Dims の最終メンバ `x_scale` に 1 語足すだけ）。 */
export const linearGemvParallelPackedParams = (
  storage: WeightStorage,
  m: number,
  n: number,
  k: number,
  xScale: number,
  group?: number,
): Uint32Array<ArrayBuffer> => {
  assertRowsStorage(storage);
  const params = linearGemvParams(storage, m, n, k, group);
  params[3] = activationScaleBits(xScale);
  return params;
};

/**
 * SRQ 融合エピローグ付き packed 活性変種の params。既存の並び（`m/n/k/rounding_mask` +
 * 出力側 SRQ の表 260 語）をそのまま保ち、**末尾**へ `x_scale` を 1 語足す
 * （`array<vec4<u32>>` の 16 B 整列で語 264 に落ちる — WGSL の Dims と同じ導出）。
 */
export const linearGemvStaticQuantizePackedParams = (
  storage: WeightStorage,
  m: number,
  n: number,
  k: number,
  xScale: number,
  scale: number,
  group?: number,
): Uint32Array<ArrayBuffer> => {
  const params = new Uint32Array(268);
  params.set(linearGemvStaticQuantizeParams(storage, m, n, k, scale, group));
  params[264] = activationScaleBits(xScale);
  return params;
};

/** SRQの整数表を共用し、128境界の上限探索だけを固定回数へ展開する。 */
const staticQuantizeEpilogue = (): string => `
fn srqWord(index: u32) -> u32 { return dims.srq[index >> 2u][index & 3u]; }
fn quantize(bits: u32) -> u32 {
  if (srqWord(258u) != 0u) { return bits; }
  let magnitude = bits & 0x7fffffffu;
  if (magnitude > 0x7f800000u) { return bits | 0x00400000u; }
  var lo = 0u;
${
  [64, 32, 16, 8, 4, 2, 1].map((step) =>
    `  lo += select(0u, ${step}u, magnitude >= srqWord(lo + ${step}u));`
  ).join("\n")
}
  lo += select(0u, 1u, magnitude >= srqWord(128u));
  let sign = bits & 0x80000000u;
  let level = min(lo, select(127u, 128u, sign != 0u));
  return srqWord(129u + level) | sign;
}
`;

const parallelWgsl = (
  storage: WeightStorage,
  group: number | undefined,
  lanes: LinearGemvParallelLanes,
  quantize: boolean,
  packed = false,
): string => {
  if (![2, 4, 8, 16, 32].includes(lanes)) {
    throw new CodegenError("linear_gemv_parallel: 不正なlane数");
  }
  assertRowsStorage(storage);
  const unit = linearGemvUnit(storage);
  const shift = gemvGroupShift(storage, group);
  // MUST: `x_scale` は Dims の**最終メンバ**（packed 変種の params はこの位置へ 1 語足す）。
  return `// karume linear gemv K parallel (${storage}, ${lanes} lanes/output${
    packed ? ", packed int8 活性" : ""
  })
struct Dims {
  m: u32,
  n: u32,
  k: u32,${quantize ? "\n  rounding_mask: u32,\n  srq: array<vec4<u32>, 65>," : ""}${
    packed ? "\n  x_scale: f32," : ""
  }
}
@group(0) @binding(0) var<uniform> dims: Dims;
${bindings(unit, storage, quantize, packed)}${quantize ? staticQuantizeEpilogue() : ""}
var<workgroup> partial: array<f32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let lane = lid % ${lanes}u;
  let col = wg.x * ${128 / lanes}u + lid / ${lanes}u;
  var acc = 0.0;
  // 端の列も部分和0を書き、全threadが同じbarrierを通る。
  if (col < dims.n) {
    let units = dims.k / ${unit}u;
    let row_base = col * units;${scaleSetupWgsl(storage, shift)}
    for (var unit = lane; unit < units; unit += ${lanes}u) {
${unitLoads(storage, "t", "unit", shift, `wg.y * (dims.k / ${packed ? 16 : 4}u) + `, packed)}
${unitMacs(storage, "t", packed, true)}
    }
  }
  partial[lid] = acc;
  workgroupBarrier();
  for (var width = ${lanes / 2}u; width > 0u; width /= 2u) {
    if (lane < width) {
      partial[lid] = partial[lid] + partial[lid + width];
    }
    workgroupBarrier();
  }
  if (col < dims.n && lane == 0u) {
    out[wg.y * dims.n + col] = ${
    quantize
      ? "quantize(bitcast<u32>(partial[lid] + bias[col]) ^ dims.rounding_mask)"
      : "partial[lid] + bias[col]"
  };
  }
}
`;
};

export const linearGemvSubgroupKey = (
  storage: WeightStorage,
  group: number | undefined,
  lanes: LinearGemvParallelLanes,
): string => `${linearGemvParallelKey(storage, group, lanes)}:subgroup32`;

/**
 * parallelと同じ入力配分・加算木をsubgroup内の値交換で実行する。
 * 既存のWGSLとキーは維持する。固定32レーンの機能を明示要求し、共有メモリを使わない。
 * DECIDED: docs/decisions/0101-linear-gemv-subgroup.md
 */
export const linearGemvSubgroupWgsl = (
  storage: WeightStorage,
  group: number | undefined,
  lanes: LinearGemvParallelLanes,
): string => {
  if (![2, 4, 8, 16, 32].includes(lanes)) {
    throw new CodegenError("linear_gemv_parallel: 不正なlane数");
  }
  assertRowsStorage(storage);
  const unit = linearGemvUnit(storage);
  const shift = gemvGroupShift(storage, group);
  return `enable subgroups, subgroup_size_control;
// karume linear gemv K subgroup32 (${storage}, ${lanes} lanes/output)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
${bindings(unit, storage)}

@compute @workgroup_size(128) @subgroup_size(32)
fn main(
  @builtin(subgroup_invocation_id) sub: u32,
  @builtin(subgroup_id) sg: u32,
  @builtin(workgroup_id) wg: vec3<u32>,
) {
  let lane = sub % ${lanes}u;
  let col = wg.x * ${128 / lanes}u + sg * ${32 / lanes}u + sub / ${lanes}u;
  var acc = 0.0;
  // 端の列も全レーンがshuffleへ参加する。local IDとsubgroup IDの配置を仮定しない。
  if (col < dims.n) {
    let units = dims.k / ${unit}u;
    let row_base = col * units;${scaleSetupWgsl(storage, shift)}
    for (var unit = lane; unit < units; unit += ${lanes}u) {
${unitLoads(storage, "t", "unit", shift, "wg.y * (dims.k / 4u) + ")}
${unitMacs(storage, "t")}
    }
  }
  for (var width = ${lanes / 2}u; width > 0u; width /= 2u) {
    let other = subgroupShuffleXor(acc, width);
    if (lane < width) { acc = acc + other; }
  }
  if (col < dims.n && lane == 0u) {
    out[wg.y * dims.n + col] = acc + bias[col];
  }
}
`;
};
