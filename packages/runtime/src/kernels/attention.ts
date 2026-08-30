/**
 * 融合 attention（ADR 0023）の 3 カーネル。1 ノード = 3 dispatch:
 *
 * | 段 | キー                                        | 役割                                              |
 * | -- | ------------------------------------------- | ------------------------------------------------- |
 * | ①  | `attention_qk:v1:f32:reg64x64r8x4w16{v4}`   | `S = (q·scale) @ (k·scale)ᵀ` を**実体化**         |
 * | ②  | `attention_stats:v1:f32:lastdim:safe:wg256` | 行ごとの `m = amax S` と `inv = 1/Σexp(S−m)`      |
 * | ③  | `attention_pv:v1:f32:reg64x64r8x4w16{v4}`   | `O = P @ v`（`P = exp(S−m)·inv` は**非実体化**）  |
 *
 * ① と ③ は GEMM 骨格（src/kernels/gemm.ts）の断片共有で、内積ループの正本は 1 箇所のまま。
 * ② だけがここに実体を持つ。
 *
 * ## ビット同一が設計の核（MUST）
 *
 * 出力は分解経路（`mul → permute → bmm → softmax → expand → bmm`）と**ビット単位で一致する**。
 * 成立の根拠は 4 つで、どれか 1 つでも崩すと丸め列が変わる:
 *
 * 1. ① の縮約は K 昇順・K タイル 16（現行 bmm と同一の骨格）。
 * 2. `scale` は**タイル充填時に q 側と k 側の両方**へ掛ける（半スケール契約 — src/ops.ts）。
 *    内積の後に 1 度だけ掛ける形に変えると、`(q·s)·(k·s)` の各積が f32 へ丸まる位置が動く。
 * 3. ② は**現行 softmax.ts のパス①②を逐語で切り出したもの**（同じ 256 幅ツリー縮約・
 *    同じ走査順 `i += 256`・同じ identity `-F32_MAX`・同じ `inv = 1.0 / Σ`）。
 * 4. ③ の A 要素は `exp(S − m) · inv` を f32 で評価した値で、現行 softmax がパス③で
 *    書き出す値と同じ式・同じ演算順。以降の縮約も現行 bmm と同一。
 *
 * MUST: ② を「1 パスの online softmax」へ書き換えない。速いが縮約順序が変わり、上記 3 が
 * 崩れてビット同一の門（PNG sha256 / atol=0 の実測一致）を失う。別順序の変種を入れるなら
 * **別キー + tolerance 全面再導出**とセット（ADR 0022 決定 3 の規律）。
 *
 * ## 加算 mask は ① の epilogue だけの軸（ADR 0023 改訂）
 *
 * 省略可能な第 4 入力 `mask[1,1,M,N]`（src/ops.ts の契約）は、**①QK の書き出し直前で
 * `S' = fl(S + mask[m·N+n])` を 1 度足す**形で入る。分解経路が `bmm`（S を実体化）→
 * `add`（mask を足す）の 2 ノードで作る値と、丸めの位置も回数も同じなのでビット同一。
 * MUST: ②③ は mask を**一切見ない**（S は既に mask 済み）。生成物もキーも 1 バイト動かない
 * ことをスナップショットが固定する。
 *
 * ## S の格納形は独立した軸（案 γ 波 1）
 *
 * 3 カーネルが受け渡す S の**格納形**は、計算形（{@link "./gemm.ts"} の `GemmCompute}`）とは
 * **別のつまみ**（src/kernels/score-storage.ts）。`s16` は S を `array<u32>` に
 * `pack2x16float` で詰める形で、①が書き ②③ が読む以上 **3 カーネルが同時に切り替わる**
 * （段ごとの混成はあり得ない）。丸めが増えるのは格納の 1 点だけなので、上記 4 つの根拠は
 * 「S が f16 に丸まった f32 経路」としてそのまま生きる。
 */

import { CodegenError } from "../codegen/errors.ts";
import { assertU32Params } from "../codegen/params.ts";
import {
  assertGemmRowWindow,
  type GemmCompute,
  gemmComputeKeyPart,
  gemmKeyPart,
  gemmParams,
  type GemmRowWindowSpan,
  gemmWgsl,
} from "./gemm.ts";
import {
  assertScoreStorageSupported,
  scoreArrayType,
  scoreKeyPart,
  scoreNote,
  scoreReadAt,
  scoreScalarLoaderWgsl,
  type ScoreStorage,
} from "./score-storage.ts";

/** ② の workgroup 幅。**softmax と同じ 256**（ツリー縮約の段数が縮約順序そのもの）。 */
export const ATTENTION_STATS_WORKGROUP_SIZE = 256;

/** 行統計 1 行あたりの語数（`[0]` = 行の最大値 / `[1]` = `1/Σexp(S−m)`）。 */
export const ATTENTION_STATS_STRIDE = 2;

/**
 * ①QK の**加算 mask** 判別子（ADR 0023 改訂）。
 *
 * MUST: 無しは空文字（既存キーは 1 文字も動かない）。MUST: 語は s16 のさらに**後ろ**に置く —
 * 計算 `:c16` / 格納 `:s16` / mask の 3 語が同時に立ちうるので、並び順をここ 1 箇所で固定
 * しないと同一構成が 2 通りのキーを持つ。
 */
const maskKeyPart = (mask: boolean): string => mask ? ":mask" : "";

/**
 * **GQA**（整除 broadcast — ADR 0067 決定 1 / 2）の判別子。
 *
 * MUST: 無しは空文字（`r = 1` の既存キーは 1 文字も動かない — ADR 0067 決定 2 の
 * 「r=1 はバイト同一」がキー側にも掛かる）。MUST: 語は mask の**さらに後ろ**（並び順を
 * ここ 1 箇所で固定する — 同一構成が 2 通りのキーを持たないための規律）。
 * MUST: `r` の値そのものは載せない（uniform で運ぶ — 載せると r の種類ぶんパイプラインが
 * 増える。scale / row_offset と同じ規律）。
 */
const gqaKeyPart = (gqa: boolean): string => gqa ? ":gqa" : "";

/**
 * **行窓**（クエリ行のブロック実行 — {@link "./gemm.ts"} `GemmRowWindow`）の判別子。
 *
 * MUST: 無しは空文字（n = 1 の機は既存キーと 1 文字も変わらない — 行ブロック化の
 * 「分割が発火しない形は既存と完全一致」がキー側にも掛かる）。MUST: 語は GQA の**さらに
 * 後ろ**（並び順をここ 1 箇所で固定する）。MUST: 側（`a` / `c`）は op から決まるので綴りは
 * 固定 — bmm の `:rwa` / `:rwc` と同じ語彙にして、診断のキー一覧で並べて読める形に保つ。
 * MUST: 行オフセットと全 M は載せない（uniform で運ぶ — 載せるとブロックの本数だけ
 * パイプラインが増える）。
 */
const rowWindowKeyPart = (side: "a" | "c", rowWindow: boolean): string =>
  rowWindow ? `:rw${side}` : "";

export const attentionQkKey = (
  v4: boolean,
  compute: GemmCompute = "f32",
  score: ScoreStorage = "f32",
  mask = false,
  gqa = false,
  rowWindow = false,
): string =>
  `attention_qk:v1:f32:${gemmKeyPart(v4)}${gemmComputeKeyPart(compute)}${scoreKeyPart(score)}${
    maskKeyPart(mask)
  }${gqaKeyPart(gqa)}${rowWindowKeyPart("a", rowWindow)}`;

export const attentionPvKey = (
  v4: boolean,
  compute: GemmCompute = "f32",
  score: ScoreStorage = "f32",
  gqa = false,
  rowWindow = false,
): string =>
  `attention_pv:v1:f32:${gemmKeyPart(v4)}${gemmComputeKeyPart(compute)}${scoreKeyPart(score)}${
    gqaKeyPart(gqa)
  }${rowWindowKeyPart("c", rowWindow)}`;

/**
 * ② の **regcache 変種**（S を 1 回だけ読んでレジスタに残す）が受け持てる 1 スレッドあたりの
 * 要素数の上限。self（dim = 4096）は 16・cross（dim = 512）は 2 で、実測は ×1.672 / ×1.115
 * （docs/research/2026-08-10-kernel-variant-sweep.md §3.1）。
 *
 * 上限を置くのは、`epc` がそのままレジスタ本数になるから — dim が大きい形（32k のような
 * 長系列）で青天井にすると spill してレジスタ保持の意味が消える。超えた形は**現行の
 * 2 回読みループへ落ちる**（値はどちらもビット同一なので、これは速度変種の選択であって
 * 近似ではない）。
 */
export const ATTENTION_STATS_REG_CACHE_MAX = 32;

/**
 * dim から regcache 変種の 1 スレッドあたり要素数を決める**唯一の純関数**
 * （`undefined` = 2 回読みループのまま）。executor はこれをキーと WGSL の両方へ渡すので、
 * 「生成された `epc` と実際の dim が食い違う」状態が構造的に起こらない。
 */
export const attentionStatsRegCache = (dim: number): number | undefined => {
  const perThread = Math.ceil(dim / ATTENTION_STATS_WORKGROUP_SIZE);
  return perThread <= ATTENTION_STATS_REG_CACHE_MAX ? perThread : undefined;
};

export const attentionStatsKey = (
  compute: GemmCompute = "f32",
  score: ScoreStorage = "f32",
  regCache?: number,
): string =>
  `attention_stats:v1:f32:lastdim:safe:wg${ATTENTION_STATS_WORKGROUP_SIZE}${
    gemmComputeKeyPart(compute)
  }${scoreKeyPart(score)}${regCache === undefined ? "" : `:rc${regCache}`}`;

/** f32 変種のキー（既存の呼び出し面 — 正本は {@link attentionStatsKey}）。 */
export const ATTENTION_STATS_KEY = attentionStatsKey();

export const attentionQkWgsl = (
  v4: boolean,
  compute: GemmCompute = "f32",
  score: ScoreStorage = "f32",
  mask = false,
  gqa = false,
  rowWindow = false,
): string => gemmWgsl({ op: "attention_qk", v4, compute, score, mask, gqa, rowWindow });

export const attentionPvWgsl = (
  v4: boolean,
  compute: GemmCompute = "f32",
  score: ScoreStorage = "f32",
  gqa = false,
  rowWindow = false,
): string => gemmWgsl({ op: "attention_pv", v4, compute, score, gqa, rowWindow });

/** f32 の最大有限値。WGSL に無限大リテラルが無いため amax の identity にこれを使う（softmax と同じ）。 */
const F32_MAX = "3.402823466e38";

/**
 * ② 行統計。**現行 softmax.ts のパス①②をそのまま切り出したもの**で、書き出しだけが
 * 「行ごとの 2 語」に変わっている（`exp` の 2 度目の評価と行全体の書き戻しが消えた）。
 *
 * MUST: 走査順（`i = lid; i += 256`）とツリー縮約の段（`stride = 128, 64, …`）を
 * softmax と一致させる。1 行の総和は加算順序で最終 ulp が動くので、ここが違うと
 * 分解経路とのビット同一が壊れる。
 * NOTE: `max` は NaN 伝播を保証しない（softmax / reduce.ts と同じ既知の乖離）。
 */
export const attentionStatsWgsl = (
  compute: GemmCompute = "f32",
  score: ScoreStorage = "f32",
  regCache?: number,
): string => {
  const f16 = compute === "f16";
  // ② は S を**読むだけ**なので v4 の制約（書き側の RMW）は掛からない — 見るのは
  // 「`:c16` と同時に立っていないこと」だけ。
  assertScoreStorageSupported("attention_stats", score, true, f16);
  // MUST: S は f32 へ広げてから縮約する（拡幅は厳密なので、同じ S に対して f32 変種と
  // **ビット単位で同じ** m / inv が出る）。f16 のまま max / exp を取ると縮約の丸め列が
  // 変わり、ADR 0023 の「② は softmax のパス①② と逐語一致」が崩れる。
  const read = (index: string): string => f16 ? `f32(s[${index}])` : scoreReadAt("s", score, index);
  // regcache 変種は「要素 → スレッドの割当」も「縮約の順序」も 2 回読み版と**完全に同一**で、
  // 違いは S を 2 度読むか 1 度読んでレジスタに残すかだけ（範囲外はどちらの段でも触らない）。
  // したがって m / inv は 1 ビットも動かない — ADR 0023 のビット同一の根拠 3 を壊さない。
  const slots = regCache === undefined ? [] : Array.from({ length: regCache }, (_, at) => at);
  const indexPrologue = regCache === undefined ? "" : `${
    slots
      .map((at) =>
        `  let i${at} = lid${at === 0 ? "" : ` + ${at * ATTENTION_STATS_WORKGROUP_SIZE}u`};`
      )
      .join("\n")
  }\n`;
  const maxPass = regCache === undefined
    ? `    var i = lid;
    while (i < dim) {
      hi = max(hi, ${read("base + i")});
      i = i + ${ATTENTION_STATS_WORKGROUP_SIZE}u;
    }`
    : slots
      .map((at) =>
        `    var c${at} = 0.0;
    if (i${at} < dim) {
      c${at} = ${read(`base + i${at}`)};
      hi = max(hi, c${at});
    }`
      )
      .join("\n");
  const sumPass = regCache === undefined
    ? `    var j = lid;
    while (j < dim) {
      acc = acc + exp(${read("base + j")} - amax);
      j = j + ${ATTENTION_STATS_WORKGROUP_SIZE}u;
    }`
    : slots
      .map((at) =>
        `    if (i${at} < dim) {
      acc = acc + exp(c${at} - amax);
    }`
      )
      .join("\n");
  return `// karume attention_stats (行ごとの m = amax(S) と inv = 1/Σexp(S - m), f32${
    f16 ? ", S は f16 格納" : ""
  }${scoreNote(score)}${
    regCache === undefined ? "" : `, S は 1 回読みでレジスタ保持（1 スレッド ${regCache} 要素）`
  })${f16 ? "\nenable f16;" : ""}
struct Params {
  rows: u32,
  dim: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> s: array<${scoreArrayType(score, f16 ? "f16" : "f32")}>;
@group(0) @binding(2) var<storage, read_write> stats: array<f32>;
${scoreScalarLoaderWgsl("s", score)}
var<workgroup> scratch: array<f32, ${ATTENTION_STATS_WORKGROUP_SIZE}>;

@compute @workgroup_size(${ATTENTION_STATS_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid3: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let lid = lid3.x;
  let dim = params.dim;
${indexPrologue}  var row = wid.x;
  while (row < params.rows) {
    let base = row * dim;

    // ① 行の最大値（safe-softmax の減算項）
    var hi = -${F32_MAX};
${maxPass}
    scratch[lid] = hi;
    workgroupBarrier();
    var stride = ${ATTENTION_STATS_WORKGROUP_SIZE / 2}u;
    while (stride > 0u) {
      if (lid < stride) {
        scratch[lid] = max(scratch[lid], scratch[lid + stride]);
      }
      workgroupBarrier();
      stride = stride / 2u;
    }
    let amax = scratch[0u];
    // scratch の読み終わりを揃えてから ② で上書きする
    workgroupBarrier();

    // ② Σ exp(S - amax)（最大要素が exp(0) = 1 を出すので分母は必ず 1 以上）
    var acc = 0.0;
${sumPass}
    scratch[lid] = acc;
    workgroupBarrier();
    var stride2 = ${ATTENTION_STATS_WORKGROUP_SIZE / 2}u;
    while (stride2 > 0u) {
      if (lid < stride2) {
        scratch[lid] = scratch[lid] + scratch[lid + stride2];
      }
      workgroupBarrier();
      stride2 = stride2 / 2u;
    }
    // MUST: 逆数はここで作る（③ で割り算に戻すと softmax のパス③と演算が変わる）
    let inv = 1.0 / scratch[0u];
    if (lid == 0u) {
      stats[row * ${ATTENTION_STATS_STRIDE}u] = amax;
      stats[row * ${ATTENTION_STATS_STRIDE}u + 1u] = inv;
    }
    // 次の行が scratch[lid] を上書きする前に scratch[0] の読み終わりを揃える
    workgroupBarrier();
    row = row + nwg.x;
  }
}
`;
};

/** f32 変種の WGSL（既存の呼び出し面 — 正本は {@link attentionStatsWgsl}）。 */
export const ATTENTION_STATS_WGSL: string = attentionStatsWgsl();

/**
 * GQA の繰り返し数 `r = H / Hkv`（uniform に載る導出値 — ADR 0067 決定 2）の値域門。
 *
 * MUST: 0 を通さない。WGSL の u32 ゼロ除算は trap せず実装依存の値を返すので、`r = 0` は
 * 例外も NaN も出ないまま K / V の base が化ける（沈黙誤値）。
 */
const assertKvRepeat = (where: string, kvRepeat: number): void => {
  assertU32Params(where, { kv_repeat: kvRepeat });
  if (kvRepeat < 1) {
    throw new CodegenError(`${where}: kv_repeat は正整数（${kvRepeat}）`);
  }
};

/**
 * ① の uniform（`{m, n, k, scale}`）。4 語目が**半スケール**の f32 ビット列。
 *
 * MUST: scale を WGSL に焼かない（値の種類だけパイプラインが増える — masked_fill の
 * 埋め値と同じ理由）。
 *
 * `kvRepeat` は **GQA 変種だけ**が渡す 5 語目（ADR 0067 決定 2）。uniform struct の整列で
 * 32 バイトに伸びる（bmm の行窓変種と同じ形）。MUST: 省略時のバイト列は従来どおり 16 バイト
 * ちょうど — `r = 1` の既存経路が 1 バイトも動かないことがこの分岐の目的。
 *
 * `window` は **行窓変種だけ**が渡す末尾 2 語（{@link "./gemm.ts"} `ROW_WINDOW_DIMS_EXTRA` と対）。
 * MUST: 位置は `kv_repeat` の**後ろ**（GQA の語位置を行窓の有無で動かさない — 生成側の
 * dimsExtra が同じ順で綴る）。MUST: `m` はブロック行数（全 M ではない）。
 */
export const attentionQkParams = (
  m: number,
  n: number,
  k: number,
  scale: number,
  kvRepeat?: number,
  window?: GemmRowWindowSpan,
): Uint32Array<ArrayBuffer> => {
  // MUST: 有限判定は **f32 として**行う（載せ先が f32 語 — f64 で有限な `1e39` は `+Inf` に
  // なり、`0 * Inf = NaN` でスコアが黙って壊れる。契約層 `assertFiniteAttr` と同じ門）。
  if (!Number.isFinite(Math.fround(scale))) {
    throw new CodegenError(`attention_qk params: scale は f32 として有限の数値（${scale}）`);
  }
  if (kvRepeat !== undefined) assertKvRepeat("attention_qk params", kvRepeat);
  if (window !== undefined) assertGemmRowWindow("attention_qk 行窓 params", m, window);
  const base = gemmParams("attention_qk", m, n, k);
  if (kvRepeat === undefined && window === undefined) {
    new Float32Array(base.buffer)[3] = scale;
    return base;
  }
  const params = new Uint32Array(8);
  params.set(base.subarray(0, 3));
  new Float32Array(params.buffer)[3] = scale;
  if (kvRepeat !== undefined) params[4] = kvRepeat;
  if (window !== undefined) {
    const at = kvRepeat === undefined ? 4 : 5;
    params[at] = window.offset;
    params[at + 1] = window.rowsFull;
  }
  return params;
};

/**
 * ③ の uniform（`{m, n, k}` — bmm と同じ 3 語）。
 *
 * `kvRepeat` は **GQA 変種だけ**が渡す 4 語目（ADR 0067 決定 2）。16 バイト整列の余りに収まる
 * ので、GQA でもバイト数は変わらない（省略時は従来どおり 4 語目が 0）。
 *
 * `window` は **行窓変種だけ**が渡す末尾 2 語（①QK と同じ規律 — `kv_repeat` の後ろ）。ここで
 * だけ struct が 16 バイトを越えるので 32 バイト確保する。MUST: `m` はブロック行数。
 */
export const attentionPvParams = (
  m: number,
  n: number,
  k: number,
  kvRepeat?: number,
  window?: GemmRowWindowSpan,
): Uint32Array<ArrayBuffer> => {
  if (kvRepeat !== undefined) assertKvRepeat("attention_pv params", kvRepeat);
  if (window !== undefined) assertGemmRowWindow("attention_pv 行窓 params", m, window);
  const params = gemmParams("attention_pv", m, n, k);
  if (window === undefined) {
    if (kvRepeat !== undefined) params[3] = kvRepeat;
    return params;
  }
  const wide = new Uint32Array(8);
  wide.set(params.subarray(0, 3));
  if (kvRepeat !== undefined) wide[3] = kvRepeat;
  const at = kvRepeat === undefined ? 3 : 4;
  wide[at] = window.offset;
  wide[at + 1] = window.rowsFull;
  return wide;
};

/**
 * ② の uniform。WGSL の uniform アドレス空間では struct の整列が 16 バイトになるため、
 * 2 語ぶんの内容でも 16 バイト確保する MUST（softmax と同じ）。
 *
 * MUST: regcache 変種を使うなら、その `regCache` をここにも渡して幾何を見る（生成側と二重だが、
 * カーネル直呼びの経路も通る門 — gru_scan の hidden 上限と同型）。`epc` は生成時に焼かれる
 * 一方 `dim` は実行時値なので、`dim > epc · 256` になると余った要素が max にも Σ にも
 * 入らず、確率が黙ってずれる（例外も NaN も出ない沈黙誤値）。
 */
export const attentionStatsParams = (
  rows: number,
  dim: number,
  regCache?: number,
): Uint32Array<ArrayBuffer> => {
  assertU32Params("attention_stats params", { rows, dim });
  if (dim < 1) {
    throw new CodegenError(`attention_stats params: dim は正整数（${dim}）`);
  }
  if (regCache !== undefined) {
    // MUST: 整数であることを先に見る。生成側の `Array.from({ length: regCache })` は非整数長を
    // **切り捨てて**スロットを並べる一方、下の被覆計算は非整数のまま `regCache · 256` を主張する
    // ので、`1.5` は「スロット 1 本 = 256 要素ぶんしか展開していないのに 384 要素まで担当」と
    // 名乗って通り、余った要素が max にも Σ にも入らない沈黙誤値になる。
    if (!Number.isInteger(regCache)) {
      throw new CodegenError(`attention_stats params: regCache ${regCache} が整数でない`);
    }
    if (regCache > ATTENTION_STATS_REG_CACHE_MAX) {
      throw new CodegenError(
        `attention_stats params: regCache ${regCache} が上限 ${ATTENTION_STATS_REG_CACHE_MAX} を超える`,
      );
    }
    const covered = regCache * ATTENTION_STATS_WORKGROUP_SIZE;
    if (dim > covered) {
      throw new CodegenError(
        `attention_stats params: dim ${dim} が regcache 変種の担当範囲 ${covered}（${regCache} × ${ATTENTION_STATS_WORKGROUP_SIZE}）を超える`,
      );
    }
  }
  const params = new Uint32Array(4);
  params[0] = rows;
  params[1] = dim;
  return params;
};
