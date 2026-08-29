/**
 * 融合 attention の **i8a8 カーネル**（①QK / ③PV）が、この device で**既知解を返すか**を
 * 実走で確かめるカナリア。判定結果が `SessionState.attentionI8a8Dot` の既定になる。
 *
 * ## なぜ attention だけ実走の門が要るのか
 *
 * 「`dot4I8Packed` 版とエミュ版は同じ整数を返す」（src/kernels/linear-i8a8.ts）は整数演算の
 * 性質から導かれる主張で、linear では Metal を含めて実機で通っている。ところが**融合
 * attention だけ**は Apple M2 で dp4a / emu の一致が崩れる（docs/known-issues.md）。同じ
 * `idot` を使っているのだから故障は `idot` 単体ではなく、**attention カーネルの文脈**
 * （幾何・共有タイルの充填・書き出し）との相互作用にある。したがって:
 *
 * - MUST: カナリアは **production の幾何で最低 1 タイル全域**を埋める形で撃つ。極小形は
 *   タイル充填の担当割りも K タイルの巡回も踏まないので、故障を素通りする。
 * - MUST: 突合の**一次判定**は **既知解**（TS 参照 src/reference/i8a8.ts）と行う。腕同士の
 *   相互比較を一次判定にすると、emu の側が壊れている device で「両方同じ誤値 → 一致 →
 *   健全」と判定する（M2 でどちらが壊れているかは未確定なので、この可能性は排除できない）。
 *   相互比較を使うのは**両腕とも既知解の sanity 帯に収まった後**の腕選びだけ（下記 3）。
 * - MUST: 両腕とも sanity 帯を外したら {@link GpuFeatureError} で落とす。変種選択の問題では
 *   なく「この device の整数 attention が壊れている」ので、黙って片方を選ぶと ADR 0058 決定 2
 *   （アダプタ能力から数値を変える経路を自動選択しない）を守ったまま誤値を配ることになる。
 *
 * 縮退したことは診断に出る — emu を選べばパイプラインキーに `:dp4aEmu` が載る
 * （src/kernels/attention-i8a8.ts・ADR 0058 決定 3）。
 *
 * ## 判定則 v2 — なぜ atol=0 だけでは足りないか
 *
 * 初版は「既知解と atol=0 で一致した腕」しか採らなかった。ところが Apple M2 の実測
 * （2026-08-29 / Deno 2.9.6 — docs/research/2026-08-29-chatgpt-review-verification.md）は
 * **dp4a と emu が 1 ビットも違わない**（カナリア ①QK v4 の実測は両腕とも
 * 28.433290481567383）のに、既知解 28.43328857421875 とは**ちょうど 1 ULP** ずれる。整数段
 * （③PV の qP）は 62,088 要素で不一致 0% なので、ずれているのは整数内積ではなく**両変種が
 * 共有する f32 エピローグの丸め**（仮説: naga→MSL の FMA 契約差 — 未確定）。初版はこれを
 * 「両腕とも既知解を外した」と読んで落とし、M2 で `attentionCompute: "a8"` を丸ごと殺していた。
 *
 * そこで判定は次の順で行う（{@link decideAttentionI8a8Dot}）:
 *
 * 1. dp4a が既知解と厳密一致 → `"dp4a"`（健全機は従来どおり 1 submit で終わる）。
 * 2. emu が既知解と厳密一致 → `"emu"`（片腕だけ壊れた device）。
 * 3. どちらも厳密一致しないなら **sanity 帯**（{@link SANITY_RELATIVE_TOLERANCE}）で裁く。
 *    片腕だけ帯内ならその腕。両腕とも帯内なら**腕同士が全 6 変種の全要素でビット同一か**を
 *    見て、同一なら `"dp4a"`（= 変種選択が数値に無関係であることを機械検証できたので
 *    ADR 0058 決定 2 は満たされる）、相違なら既知解に近い腕（同点は dp4a）。
 * 4. 両腕とも帯外 → {@link GpuFeatureError}（本当に壊れている device だけがここへ来る）。
 *
 * 3 と 4 で決まった判定は「厳密一致ではなかった」事実を戻り値
 * （{@link AttentionI8a8Decision.exact}）に載せる。呼び手（src/runtime/executor.ts）は
 * device 単位に 1 度だけ警告を出す — 黙って通さないが、実行は止めない。
 *
 * ## 撃つ変種（生成物のキー軸の最小代表集合）
 *
 * ①QK は `v4`（出力列 `N % 4`）× S の格納形、③PV は `v4`（出力列 `D % 4`）× S の格納形で
 * WGSL が分岐する。実際に生成されうる組は 6 つで、それを 1 submit で撃つ:
 *
 * - ①QK: `v4/s=f32` / `スカラ/s=f32` / `v4/s=f16`
 * - ③PV: `v4/s=f32` / `スカラ/s=f32` / `v4/s=f16`
 *
 * `スカラ × s=f16` は生成できない（s16 は v4 経路専用 — src/kernels/score-storage.ts）ので
 * 直積にはならない。**バッチ軸（B·H ≥ 2）はカナリアの守備範囲外** — base の算術は dp4a /
 * emu で 1 文字も変わらない生成部分で、既存の GPU テストが検出器を持っている。
 *
 * ## 固定入力（凍結）
 *
 * 入力は全て決定的な純関数で作る（乱数は使わない — 失敗が再現しないため）。**量子化は
 * ホスト側で済ませて i8 ペイロードを直接上げる**ので、`quantize_rows` の除算（WGSL は
 * 2.5 ULP まで許される）はカナリアの経路に入らない = 既知解との一致は atol=0 で主張できる。
 * ③PV だけは `qP = round(127·exp(S−m))` を GPU が作るため `exp` の実装差が乗る。そこで
 * **`127·exp(S−m)` が丸め境界（半整数）から 0.30 以上離れる S しか使わない**
 * （WGSL の `exp` 誤差は数 ULP = 1e-5 のオーダーなので桁で安全）。この余裕は
 * tests/gpu_attention_dp4a_canary_test.ts が固定入力の全値で実測して門にする。
 */

import { defaultI8a8Geometry, i8a8TileM, i8a8TileN } from "../kernels/i8a8-geometry.ts";
import {
  ATTENTION_PV_V_SCALE_BINDING,
  ATTENTION_QK_K_SCALE_BINDING,
  ATTENTION_QK_Q_SCALE_BINDING,
  attentionPvI8a8Key,
  attentionPvI8a8Params,
  attentionPvI8a8Wgsl,
  attentionQkI8a8Key,
  attentionQkI8a8Params,
  attentionQkI8a8Wgsl,
} from "../kernels/attention-i8a8.ts";
import { tiledWorkgroups } from "../codegen/dispatch.ts";
import type { ScoreStorage } from "../kernels/score-storage.ts";
import { f16BitsToF32, roundToF16 } from "../format/f16.ts";
import {
  quantizeRowsReference,
  referenceAttentionPvI8a8Core,
  referenceAttentionPvQuant,
  referenceAttentionQkI8a8,
} from "../reference/i8a8.ts";
import type { I8a8Dot } from "../runtime/session-types.ts";
import {
  discardFailureScopes,
  type GpuContext,
  GpuFeatureError,
  popFailureScopes,
  pushFailureScopes,
  RUNTIME_INTERNAL,
  withPipelineScope,
} from "./device.ts";
import { BUFFER_USAGE, MAP_MODE } from "./webgpu-constants.ts";

/** 診断とエラー文言に使う名前（この 1 本で「どのカナリアが落ちたか」が読める）。 */
const WHERE = "attention i8a8 カナリア";

/** 生成 WGSL の差し替え口（**故障注入テスト専用**）。既定は production の生成物そのもの。 */
export type CanaryWgslPatch = (wgsl: string) => string;

const IDENTITY_PATCH: CanaryWgslPatch = (wgsl) => wgsl;

/**
 * sanity 帯の相対幅。要素ごとの許容は `max(|既知解| · この値, {@link SANITY_ABSOLUTE_FLOOR})`。
 *
 * 導出（M2 実測 2026-08-29 — docs/research/2026-08-29-chatgpt-review-verification.md）:
 *
 * - **許容したい側** = 両変種が共有する f32 エピローグの丸め差。実測は 1 ULP
 *   （28.433290481567383 vs 既知解 28.43328857421875 = 相対 6.7e-8）で、数 ULP を見ても
 *   1.2e-7 相対。帯の **2 桁下**に収まる。
 * - **弾きたい側** = 実際の整数故障。故障注入（dp4a の WGSL 破壊）の実測は 28.435983657836914
 *   = 相対 9.5e-5 で、帯の **1 桁上**。
 *
 * つまり 1e-5 は両者の間に 3 桁あいた谷の底に置いてある。MUST NOT: 緩める — 1e-3 まで広げると
 * f16 格納 1 ULP（相対 ~1e-3）級の実害まで帯に入り、s16 変種の検出器が死ぬ。
 */
const SANITY_RELATIVE_TOLERANCE = 1e-5;

/**
 * sanity 帯の絶対床。相対だけで裁くと `|既知解| → 0` で許容も 0 に潰れ、0 近傍の要素だけが
 * 事実上 atol=0 のまま残る（帯が要素ごとにまだらになる）。
 *
 * 1e-7 の根拠: 現在の固定入力の最小振幅は ③PV の |O| = 6.2e-4（f32 の ULP は ~6e-11）なので
 * 床が効く要素は今は無い — 相対側だけで既に ~100 ULP の余裕がある。効き始めるのは既知解を
 * 作り直して 0 近傍が現れたときで、そこでも 1e-7 は「数 ULP の丸め差は飲むが、整数故障は
 * 飲まない」位置にある（③PV の acc が 1 ずれると |O| は最小でも 5e-4 動く — prow·vs の下限）。
 */
const SANITY_ABSOLUTE_FLOOR = 1e-7;

// ---------------------------------------------------------------------------
// 固定入力（凍結 — 値を動かしたらテストの余裕実測をやり直すこと）
// ---------------------------------------------------------------------------

/** ①QK の q（f32・量子化前）。刻みが scale の格子と共約にならない形。 */
const QK_QUERY = (i: number): number => (((i * 7) % 29) - 14) * 0.3719 + 0.0417;
/** ①QK の k（f32・量子化前）。q とは別周期にして行 / 列 scale の取り違えを値に出す。 */
const QK_KEY = (i: number): number => (((i * 5) % 41) - 20) * 0.2917 - 0.0173;

/**
 * ③PV の S が行 max から下がる幅（1/64 刻み）。**16 本とも f16 ちょうどで表せて**、
 * `127·exp(-j/64)` が半整数から 0.30 以上離れる j だけを選んである（qP は
 * 127/121/112/102/93/86/77/68/60/52/44/34/26/18/9/1 の 16 段になる）。
 * MUST: 先頭は 0 — 行の max がその行に実在する（`exp(S−m) ≤ 1` が構造で保証される）形にする。
 */
const PV_SCORE_STEPS = [0, 3, 8, 14, 20, 25, 32, 40, 48, 57, 68, 84, 102, 125, 169, 310] as const;

/** ③PV の行 max。1/64 刻みの S が f16 ちょうどに収まる小さな値なら何でもよい。 */
const PV_ROW_MAX = 2;

/** ③PV の行 `1/Σexp` （f32 で厳密な 1/8 刻み — 行ごとに変える）。 */
const PV_ROW_INV = (row: number): number => (1 + (row % 5)) / 8;

/** ③PV の Vᵀ 量子化値（±127 の整数・N 連続）。 */
const PV_V_QUANT = (i: number): number => ((i * 53) % 251) - 125;

/** ③PV の V の per-column scale（f32 で厳密な 1/8 刻み）。 */
const PV_V_SCALE = (col: number): number => 0.5 + (col % 9) * 0.125;

/** 半スケール（ADR 0023 の `√scale_factor`）— 実形と同じ作り方で 1 つ求める。 */
const halfScale = (depth: number): number => Math.fround(Math.sqrt(1 / Math.sqrt(depth)));

const fillBy = (length: number, value: (i: number) => number): Float32Array<ArrayBuffer> => {
  const data = new Float32Array(length);
  for (let i = 0; i < length; i += 1) data[i] = value(i);
  return data;
};

/** ①QK の固定入力と既知解。 */
type QkCase = {
  readonly rows: number;
  readonly cols: number;
  readonly depth: number;
  readonly scale: number;
  /** 量子化済み q / k のペイロード（`[行, depth]` の平坦 i8 = GPU の 4 詰めと同じ並び）。 */
  readonly queryPayload: Int8Array<ArrayBuffer>;
  readonly keyPayload: Int8Array<ArrayBuffer>;
  readonly queryScale: Float32Array<ArrayBuffer>;
  readonly keyScale: Float32Array<ArrayBuffer>;
  /** 既知解 S（`[rows, cols]`）。 */
  readonly expected: Float32Array<ArrayBuffer>;
};

/** ③PV の固定入力と既知解。 */
type PvCase = {
  readonly rows: number;
  /** 縮約軸 N（= params の `k`）。 */
  readonly cols: number;
  /** 出力の列 D（= params の `n`）。 */
  readonly depth: number;
  readonly scores: Float32Array<ArrayBuffer>;
  /** 行統計 `[max, inv]`（②行統計の出力と同じ並び）。 */
  readonly stats: Float32Array<ArrayBuffer>;
  readonly valuePayload: Int8Array<ArrayBuffer>;
  readonly valueScale: Float32Array<ArrayBuffer>;
  /** 既知解 O（`[rows, depth]`）。 */
  readonly expected: Float32Array<ArrayBuffer>;
};

/**
 * ①QK の固定入力（**タイル辺は幾何から導く** — 既定の幾何を替えてもカナリアは 1 タイル
 * 全域を埋め続ける）。K は必ず 2 タイル以上（1 タイルだと巡回そのものが踏まれない）。
 */
export const buildQkCase = (): QkCase => {
  const geometry = defaultI8a8Geometry("attention_qk");
  const rows = i8a8TileM(geometry);
  const cols = i8a8TileN(geometry);
  const depth = geometry.tileK * 2;
  const query = fillBy(rows * depth, QK_QUERY);
  const key = fillBy(cols * depth, QK_KEY);
  const quantizedQuery = quantizeRowsReference(query, rows, depth);
  const quantizedKey = quantizeRowsReference(key, cols, depth);
  const scale = halfScale(depth);
  return {
    rows,
    cols,
    depth,
    scale,
    queryPayload: quantizedQuery.q,
    keyPayload: quantizedKey.q,
    queryScale: quantizedQuery.scale,
    keyScale: quantizedKey.scale,
    // 参照は q / k を自分で量子化する（同じ純関数なので上のペイロードと必ず一致する）
    expected: referenceAttentionQkI8a8({
      q: query,
      k: key,
      batch: 1,
      m: rows,
      n: cols,
      d: depth,
      scale,
    }),
  };
};

/** ③PV の固定入力（同上 — ③ は ① と別の幾何を既定に取るので導出元も別）。 */
export const buildPvCase = (): PvCase => {
  const geometry = defaultI8a8Geometry("attention_pv");
  const rows = i8a8TileM(geometry);
  const depth = i8a8TileN(geometry);
  const cols = geometry.tileK * 2;
  const scores = new Float32Array(rows * cols);
  const stats = new Float32Array(rows * 2);
  for (let row = 0; row < rows; row += 1) {
    // 行ごとに位相をずらす（行の取り違えが qP の並びに出る形）
    for (let i = 0; i < cols; i += 1) {
      const step = PV_SCORE_STEPS[(i + row) % PV_SCORE_STEPS.length];
      scores[row * cols + i] = PV_ROW_MAX - step / 64;
    }
    stats[row * 2] = PV_ROW_MAX;
    stats[row * 2 + 1] = PV_ROW_INV(row);
  }
  const valuePayload = new Int8Array(depth * cols);
  for (let i = 0; i < valuePayload.length; i += 1) valuePayload[i] = PV_V_QUANT(i);
  const valueScale = fillBy(depth, PV_V_SCALE);
  const rowMax = new Float32Array(rows).fill(PV_ROW_MAX);
  const rowInv = fillBy(rows, PV_ROW_INV);
  // A タイル充填が作る整数列（GPU 側の `round(127·exp(S−m))` に対応）と、
  // 整数を受け取ってからの純関数（atol=0 が立つ側）を分けて既知解を組む。
  const quantized = referenceAttentionPvQuant(scores, rowMax, rows, cols);
  return {
    rows,
    cols,
    depth,
    scores,
    stats,
    valuePayload,
    valueScale,
    expected: referenceAttentionPvI8a8Core({
      qp: quantized,
      vq: valuePayload,
      rowInv,
      vs: valueScale,
      batch: 1,
      m: rows,
      n: cols,
      d: depth,
    }),
  };
};

// ---------------------------------------------------------------------------
// S の f16 格納（固定入力を GPU と同じ並びで詰める / 読み戻す）
// ---------------------------------------------------------------------------

/**
 * f16 **ちょうどで表せる正規数**の 16bit パターン。
 *
 * MUST: 一般の f32 を渡さない — 丸めも subnormal も Inf も扱わず、仮数の下位 13bit を
 * 黙って捨てる。固定 S は 1/64 刻みで |v| ≤ 3 の正規数だけなので、指数の付け替えと仮数の
 * 切り出しで厳密に決まる（丸めの正本は {@link roundToF16} 側で、こちらは詰め直しの器）。
 * 往復の一致（{@link f16BitsToF32} で戻ること）は固定入力の全値でテストが門にする。
 */
const exactF16Bits = (view: DataView, value: number): number => {
  view.setFloat32(0, value);
  const bits = view.getUint32(0);
  const sign = (bits >>> 16) & 0x8000;
  const exponent = ((bits >>> 23) & 0xff) - 127 + 15;
  const mantissa = (bits & 0x7fffff) >>> 13;
  return sign | (exponent << 10) | mantissa;
};

/**
 * S を GPU と同じ格納形（`pack2x16float` の 2 要素／語）へ詰める。
 * MUST: 下位半分が偶数添字（`pack2x16float(vec2(x, y))` の x 側）。
 */
export const packScoresF16 = (scores: Float32Array<ArrayBuffer>): Uint32Array<ArrayBuffer> => {
  const words = new Uint32Array(scores.length / 2);
  const view = new DataView(new ArrayBuffer(4));
  for (let word = 0; word < words.length; word += 1) {
    const low = exactF16Bits(view, scores[word * 2]);
    const high = exactF16Bits(view, scores[word * 2 + 1]);
    words[word] = (low | (high << 16)) >>> 0;
  }
  return words;
};

/** f16 格納の S を f32 へ戻す（`unpack2x16float` は厳密なので丸めは起きない）。 */
const unpackScoresF16 = (words: Uint32Array<ArrayBuffer>): Float32Array<ArrayBuffer> => {
  const values = new Float32Array(words.length * 2);
  for (let word = 0; word < words.length; word += 1) {
    values[word * 2] = f16BitsToF32(words[word] & 0xffff);
    values[word * 2 + 1] = f16BitsToF32(words[word] >>> 16);
  }
  return values;
};

// ---------------------------------------------------------------------------
// 実走
// ---------------------------------------------------------------------------

const STORAGE_IN = BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_DST;
const STORAGE_OUT = BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_SRC;

/** 1 変種を既知解と突き合わせた結果（{@link compareToReference}）。 */
export type CanaryVariantOutcome = {
  /** production のパイプラインキーそのもの（どの生成物の話か診断で読める）。 */
  readonly key: string;
  /** 既知解と atol=0 で一致したか。 */
  readonly exact: boolean;
  /** 全要素が sanity 帯に収まったか。 */
  readonly withinBand: boolean;
  /** 既知解との最大絶対誤差。 */
  readonly maxAbsError: number;
  /** 帯余裕の最大（`|誤差| / 許容`）。1 以下が帯内。 */
  readonly maxBandRatio: number;
  /** 最初の不一致の診断（厳密一致なら undefined）。 */
  readonly mismatch: string | undefined;
  /** 読み戻した出力そのもの。**腕同士のビット同一比較に要る**ので保持する（数 KB）。 */
  readonly output: Uint8Array<ArrayBuffer>;
};

/**
 * 既知解との全数比較。厳密一致（atol=0）と sanity 帯の 2 つを 1 走査で採る。
 *
 * MUST: 帯・最大の更新は `!(diff <= x)` / {@link Math.max} の形で書く。`diff > x` で書くと
 * NaN が黙って「帯内」に落ちる（無効 dispatch や未初期化の読み戻しが健全と判定される）。
 */
const compareToReference = (
  key: string,
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  output: Uint8Array<ArrayBuffer>,
): CanaryVariantOutcome => {
  let exact = true;
  let withinBand = true;
  let maxAbsError = 0;
  let maxBandRatio = 0;
  let mismatch: string | undefined;
  for (let i = 0; i < expected.length; i += 1) {
    if (actual[i] !== expected[i]) {
      exact = false;
      mismatch ??= `${key}[${i}]: 実測 ${actual[i]} / 既知解 ${expected[i]}`;
    }
    const diff = Math.abs(actual[i] - expected[i]);
    const tolerance = Math.max(
      Math.abs(expected[i]) * SANITY_RELATIVE_TOLERANCE,
      SANITY_ABSOLUTE_FLOOR,
    );
    if (!(diff <= tolerance)) withinBand = false;
    maxAbsError = Math.max(maxAbsError, diff);
    maxBandRatio = Math.max(maxBandRatio, diff / tolerance);
  }
  return { key, exact, withinBand, maxAbsError, maxBandRatio, mismatch, output };
};

/** 1 変種ぶんの実走計画（パイプライン生成は済み・バッファ確保はこれから）。 */
type Variant = {
  /** production のパイプラインキーそのもの（診断でどの生成物が落ちたか読める形にする）。 */
  readonly key: string;
  readonly pipeline: GPUComputePipeline;
  /** 束縛（0 番の params は実走側が足す）。 */
  readonly bindings: readonly (readonly [number, GPUBuffer])[];
  readonly params: GPUBuffer;
  readonly output: GPUBuffer;
  readonly outputBytes: number;
  readonly workgroups: readonly [number, number, number];
  /** 読み戻したバイト列を既知解と突き合わせる。 */
  readonly compare: (bytes: ArrayBuffer) => CanaryVariantOutcome;
};

/** 片腕（dp4a か emu のどちらか）を 6 変種とも撃った結果（{@link probeAttentionI8a8Dot}）。 */
export type CanaryProbe = {
  readonly dot: I8a8Dot;
  /** 6 変種**全て**が既知解と atol=0 で一致したか。 */
  readonly exact: boolean;
  /** 6 変種**全て**が sanity 帯に収まったか。 */
  readonly withinBand: boolean;
  /** 6 変種を通した最大絶対誤差。 */
  readonly maxAbsError: number;
  /** 6 変種を通した最大の帯余裕（1 以下が帯内）。 */
  readonly maxBandRatio: number;
  /** 最初に既知解を外した変種の診断（{@link exact} なら undefined）。 */
  readonly mismatch: string | undefined;
  /** 変種ごとの内訳（順序は生成順で、腕を跨いで同じ）。 */
  readonly variants: readonly CanaryVariantOutcome[];
};

/**
 * 6 変種を 1 submit で撃ち、**全変種の突合結果**を返す。
 *
 * MUST: 途中で打ち切らない（初版は最初の不一致で return していた）。腕同士のビット同一比較
 * （{@link decideAttentionI8a8Dot} の 3）に全変種の出力が要り、帯判定にも全要素の走査が要る。
 * MUST: 呼び手は device 単位の errorScope 区間ロックの中で呼ぶ（{@link decideAttentionI8a8Dot}）。
 * `await` を跨いで errorScope を張るため、裸で走らせると並行 Session の run と交差して失敗が
 * 誤帰属する。
 */
export const probeAttentionI8a8Dot = async (
  gpu: GpuContext,
  dot: I8a8Dot,
  patch: CanaryWgslPatch = IDENTITY_PATCH,
): Promise<CanaryProbe> => {
  const device = gpu.device;
  const dp4a = dot === "dp4a";
  const qk = buildQkCase();
  const pv = buildPvCase();
  const qkGeometry = defaultI8a8Geometry("attention_qk");
  const pvGeometry = defaultI8a8Geometry("attention_pv");
  const limit = gpu.limits.maxComputeWorkgroupsPerDimension;

  const pipelineOf = async (key: string, wgsl: string): Promise<GPUComputePipeline> =>
    await gpu[RUNTIME_INTERNAL].raceDeviceLost(
      withPipelineScope(device, `${WHERE} '${key}'`, () =>
        device.createComputePipeline({
          layout: "auto",
          compute: { module: device.createShaderModule({ code: patch(wgsl) }), entryPoint: "main" },
        })),
      `${WHERE} '${key}' のパイプライン生成`,
    );

  // パイプラインは同期区間の**外**で作る（生成は await を伴う）。**スカラ × s=f16 は組まない**
  // — ①QK 側は生成そのものが拒む（s16 は v4 経路専用）ので、③PV だけ撃っても対にならない。
  const shapes = [
    { v4: true, score: "f32" },
    { v4: false, score: "f32" },
    { v4: true, score: "f16" },
  ] as const;
  type Resolved = {
    readonly key: string;
    readonly pipeline: GPUComputePipeline;
    readonly score: ScoreStorage;
  };
  const qkPipelines: Resolved[] = [];
  const pvPipelines: Resolved[] = [];
  for (const { v4, score } of shapes) {
    const qkKey = attentionQkI8a8Key(v4, dp4a, score, qkGeometry);
    qkPipelines.push({
      key: qkKey,
      score,
      pipeline: await pipelineOf(qkKey, attentionQkI8a8Wgsl(v4, dp4a, score, qkGeometry)),
    });
    const pvKey = attentionPvI8a8Key(v4, dp4a, score, pvGeometry);
    pvPipelines.push({
      key: pvKey,
      score,
      pipeline: await pipelineOf(pvKey, attentionPvI8a8Wgsl(v4, dp4a, score, pvGeometry)),
    });
  }

  const buffers: GPUBuffer[] = [];
  const upload = (data: ArrayBufferView<ArrayBuffer>, usage: number): GPUBuffer => {
    const buffer = device.createBuffer({ size: data.byteLength, usage });
    buffers.push(buffer);
    device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  };
  const allocate = (byteLength: number): GPUBuffer => {
    const buffer = device.createBuffer({ size: byteLength, usage: STORAGE_OUT });
    buffers.push(buffer);
    return buffer;
  };

  try {
    // MUST: push から pop の発行までに await を挟まない（device 単位 LIFO の交錯を防ぐ根拠 —
    // src/gpu/device.ts の「errorScope 区間の不変条件」）。確保・束縛・エンコード・submit は
    // 全て同期。
    pushFailureScopes(device);
    let variants: readonly Variant[];
    let staging: GPUBuffer;
    try {
      const uniform = BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST;
      const qkParams = upload(attentionQkI8a8Params(qk.rows, qk.cols, qk.depth, qk.scale), uniform);
      const queryPayload = upload(qk.queryPayload, STORAGE_IN);
      const keyPayload = upload(qk.keyPayload, STORAGE_IN);
      const queryScale = upload(qk.queryScale, STORAGE_IN);
      const keyScale = upload(qk.keyScale, STORAGE_IN);
      const pvParams = upload(attentionPvI8a8Params(pv.rows, pv.depth, pv.cols), uniform);
      const scoresF32 = upload(pv.scores, STORAGE_IN);
      const scoresF16 = upload(packScoresF16(pv.scores), STORAGE_IN);
      const stats = upload(pv.stats, STORAGE_IN);
      const valuePayload = upload(pv.valuePayload, STORAGE_IN);
      const valueScale = upload(pv.valueScale, STORAGE_IN);

      const qkVariants = qkPipelines.map(({ key, pipeline, score }): Variant => {
        const outputBytes = qk.rows * qk.cols * (score === "f16" ? 2 : 4);
        const output = allocate(outputBytes);
        return {
          key,
          pipeline,
          params: qkParams,
          bindings: [
            [1, queryPayload],
            [2, keyPayload],
            [3, output],
            [ATTENTION_QK_Q_SCALE_BINDING, queryScale],
            [ATTENTION_QK_K_SCALE_BINDING, keyScale],
          ],
          output,
          outputBytes,
          workgroups: [
            tiledWorkgroups(qk.cols, i8a8TileN(qkGeometry), limit, WHERE),
            tiledWorkgroups(qk.rows, i8a8TileM(qkGeometry), limit, WHERE),
            1,
          ],
          compare: (bytes) =>
            score === "f16"
              // s16 変種の出力 ≡ S をホストで f16 に丸めた f32 変種（丸めは格納の 1 回だけ）
              ? compareToReference(
                key,
                unpackScoresF16(new Uint32Array(bytes)),
                Float32Array.from(qk.expected, roundToF16),
                new Uint8Array(bytes),
              )
              : compareToReference(
                key,
                new Float32Array(bytes),
                qk.expected,
                new Uint8Array(bytes),
              ),
        };
      });
      const pvVariants = pvPipelines.map(({ key, pipeline, score }): Variant => {
        const outputBytes = pv.rows * pv.depth * 4;
        const output = allocate(outputBytes);
        return {
          key,
          pipeline,
          params: pvParams,
          bindings: [
            [1, score === "f16" ? scoresF16 : scoresF32],
            [2, valuePayload],
            [3, stats],
            [4, output],
            [ATTENTION_PV_V_SCALE_BINDING, valueScale],
          ],
          output,
          outputBytes,
          workgroups: [
            tiledWorkgroups(pv.depth, i8a8TileN(pvGeometry), limit, WHERE),
            tiledWorkgroups(pv.rows, i8a8TileM(pvGeometry), limit, WHERE),
            1,
          ],
          // S の格納形は入力側の話なので、③ の既知解は 3 変種とも同じ 1 本（固定 S は f16
          // ちょうどで表せる = 詰め直しで 1 ビットも動かない）。
          compare: (bytes) =>
            compareToReference(key, new Float32Array(bytes), pv.expected, new Uint8Array(bytes)),
        };
      });
      variants = [...qkVariants, ...pvVariants];

      staging = device.createBuffer({
        size: variants.reduce((sum, variant) => sum + variant.outputBytes, 0),
        usage: BUFFER_USAGE.COPY_DST | BUFFER_USAGE.MAP_READ,
      });
      buffers.push(staging);

      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      for (const variant of variants) {
        pass.setPipeline(variant.pipeline);
        pass.setBindGroup(
          0,
          device.createBindGroup({
            layout: variant.pipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: variant.params } },
              ...variant.bindings.map(([binding, buffer]) => ({
                binding,
                resource: { buffer },
              })),
            ],
          }),
        );
        pass.dispatchWorkgroups(...variant.workgroups);
      }
      pass.end();
      let offset = 0;
      for (const variant of variants) {
        encoder.copyBufferToBuffer(variant.output, 0, staging, offset, variant.outputBytes);
        offset += variant.outputBytes;
      }
      device.queue.submit([encoder.finish()]);
    } catch (cause) {
      // MUST: push した 2 本は必ず pop して積み残さない（積み残すと以後の検証結果が誤った
      // スコープに吸われ、エラーが恒久的に見えなくなる）。
      await discardFailureScopes(device);
      throw cause;
    }
    const failure = await gpu[RUNTIME_INTERNAL].raceDeviceLost(
      popFailureScopes(device, WHERE),
      `${WHERE}の検証`,
    );
    // MUST: validation / OOM は「既知解と違う」ではなく loud な失敗として上げる。無効な
    // dispatch は全 0 を返すので、不一致に潰すと「この device の整数 attention が壊れている」
    // と誤診断する。
    if (failure !== undefined) throw failure;

    await gpu[RUNTIME_INTERNAL].raceDeviceLost(
      staging.mapAsync(MAP_MODE.READ),
      `${WHERE}の読み戻し`,
    );
    const mapped = staging.getMappedRange().slice(0);
    staging.unmap();
    let offset = 0;
    const outcomes: CanaryVariantOutcome[] = [];
    for (const variant of variants) {
      outcomes.push(variant.compare(mapped.slice(offset, offset + variant.outputBytes)));
      offset += variant.outputBytes;
    }
    return {
      dot,
      exact: outcomes.every((outcome) => outcome.exact),
      withinBand: outcomes.every((outcome) => outcome.withinBand),
      maxAbsError: outcomes.reduce((worst, o) => Math.max(worst, o.maxAbsError), 0),
      maxBandRatio: outcomes.reduce((worst, o) => Math.max(worst, o.maxBandRatio), 0),
      mismatch: outcomes.find((outcome) => outcome.mismatch !== undefined)?.mismatch,
      variants: outcomes,
    };
  } finally {
    for (const buffer of buffers) buffer.destroy();
  }
};

/** {@link AttentionI8a8Decision} がどの分岐で決まったか（判定則 v2 — モジュール冒頭 §判定則 v2）。 */
export type CanaryBranch =
  | "dp4a-exact"
  | "emu-exact"
  | "band-single-arm"
  | "band-both-identical"
  | "band-both-differ";

/** 分岐ごとの日本語の根拠（警告文言の頭 — {@link formatAttentionI8a8Decision}）。 */
const BRANCH_NOTE: Record<CanaryBranch, string> = {
  "dp4a-exact": "dot4I8Packed 版が既知解と厳密一致した",
  "emu-exact": "dot4I8Packed 版だけが既知解を外し、エミュ版は厳密一致した",
  "band-single-arm": "厳密一致した腕が無く、sanity 帯に収まったのは 1 腕だけだった",
  "band-both-identical":
    "両腕とも厳密一致は外したが sanity 帯に収まり、腕同士は全 6 変種でビット同一だった",
  "band-both-differ":
    "両腕とも厳密一致は外したが sanity 帯に収まり、腕同士は相違したので既知解に近い側を採った",
};

/** カナリアの判定（{@link decideAttentionI8a8Dot}）。 */
export type AttentionI8a8Decision = {
  readonly dot: I8a8Dot;
  /**
   * 選んだ腕が既知解と **atol=0** で一致したか。false は「帯内だが厳密ではない」— 呼び手は
   * 実行を止めず、device 単位に 1 度だけ警告を出す（黙って通さない）。
   */
  readonly exact: boolean;
  readonly branch: CanaryBranch;
  /** 選んだ腕の既知解との最大絶対誤差（{@link exact} なら 0）。 */
  readonly maxAbsError: number;
  /** 選んだ腕の最大の帯余裕（`|誤差| / 許容` — 1 以下が帯内）。 */
  readonly maxBandRatio: number;
};

/** 判定の根拠 1 行（警告と診断の共通の言い回し）。 */
export const formatAttentionI8a8Decision = (decision: AttentionI8a8Decision): string =>
  `${BRANCH_NOTE[decision.branch]}ので '${decision.dot}' を選んだ` +
  `（既知解との最大絶対誤差 ${decision.maxAbsError} / 帯余裕 ${decision.maxBandRatio}）`;

const decide = (
  probe: CanaryProbe,
  branch: CanaryBranch,
): AttentionI8a8Decision => ({
  dot: probe.dot,
  exact: probe.exact,
  branch,
  maxAbsError: probe.maxAbsError,
  maxBandRatio: probe.maxBandRatio,
});

/**
 * 腕同士が**全 6 変種の全要素**でビット同一か。1 変種でも 1 バイトでも違えば「相違」。
 *
 * 読み戻したバイト列をそのまま比べる（f32 も f16 格納も同じ規律で見られる）。変種の並びは
 * 生成順で腕を跨いで同じなので、添字で対にできる。
 */
const armsBitIdentical = (a: CanaryProbe, b: CanaryProbe): boolean =>
  a.variants.length === b.variants.length &&
  a.variants.every((variant, index) => {
    const other = b.variants[index].output;
    return variant.output.length === other.length &&
      variant.output.every((byte, at) => byte === other[at]);
  });

/**
 * 融合 attention の整数内積変種を**この device の実走**で決める（device 単位に 1 度 —
 * メモ化は {@link GpuContext} 側）。手順はモジュール冒頭 §判定則 v2 の 1〜4 そのもの:
 *
 * - dp4a が既知解と厳密一致 → `"dp4a"`（**この経路では emu 側を撃たない** — 健全な機で払う
 *   コストを 1 submit に留める。emu が壊れている機は dp4a を選んだ時点で無関係）。
 * - dp4a だけ不一致 → `"emu"`（linear の席は別なので、linear は dp4a のまま維持される）。
 * - どちらも厳密一致しない → sanity 帯で裁く（片腕だけ帯内 / 両腕帯内でビット同一 /
 *   両腕帯内で相違）。
 * - 両腕とも帯外 → {@link GpuFeatureError}。変種の選び方の問題ではないので縮退先が無い。
 *
 * `patch` は**故障注入テスト専用**の生成物差し替え口（不一致経路と両不一致経路は健全な実機
 * では作れない — src/gpu/device.ts の shader-f16 カナリアと同じ事情）。
 */
export const decideAttentionI8a8Dot = (
  gpu: GpuContext,
  patch: CanaryWgslPatch = IDENTITY_PATCH,
): Promise<AttentionI8a8Decision> =>
  gpu[RUNTIME_INTERNAL].withScopeLock(async () => {
    const dp4a = await probeAttentionI8a8Dot(gpu, "dp4a", patch);
    if (dp4a.exact) return decide(dp4a, "dp4a-exact");
    const emu = await probeAttentionI8a8Dot(gpu, "emu", patch);
    if (emu.exact) return decide(emu, "emu-exact");
    if (dp4a.withinBand !== emu.withinBand) {
      return decide(dp4a.withinBand ? dp4a : emu, "band-single-arm");
    }
    if (dp4a.withinBand) {
      // 腕同士がビット同一 = 「変種選択は数値に無関係」（ADR 0058 決定 2）が**この device で
      // 機械検証できた**ということなので、既定の dp4a をそのまま採る。共有エピローグの丸めが
      // 既知解と数 ULP ずれているだけの M2 はここへ来る。
      if (armsBitIdentical(dp4a, emu)) return decide(dp4a, "band-both-identical");
      // 相違するなら変種選択が数値を動かしている。帯内なので実行は止めないが、既知解に近い
      // 腕を採る（同点は dp4a — 既定を動かす根拠が無い）。
      return decide(emu.maxAbsError < dp4a.maxAbsError ? emu : dp4a, "band-both-differ");
    }
    throw new GpuFeatureError(
      `融合 attention の i8a8 カーネルが dot4I8Packed 版・エミュ版とも既知解の sanity 帯` +
        `（相対 ${SANITY_RELATIVE_TOLERANCE} / 絶対床 ${SANITY_ABSOLUTE_FLOOR}）を外した` +
        `（dp4a: ${dp4a.mismatch} 帯余裕 ${dp4a.maxBandRatio} / ` +
        `emu: ${emu.mismatch} 帯余裕 ${emu.maxBandRatio}）。` +
        "共有 f32 エピローグの丸め差（帯内）は既に許容した上で外しているので、" +
        "整数内積の変種選択の問題ではなく、この device の整数 attention 自体が信用できない — " +
        "attentionCompute を 'f32' か 'f16' にして実行すること",
    );
  });
