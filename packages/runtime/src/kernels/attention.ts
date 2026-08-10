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
 * ## S の格納形は独立した軸（案 γ 波 1）
 *
 * 3 カーネルが受け渡す S の**格納形**は、計算形（{@link "./gemm.ts"} の `GemmCompute}`）とは
 * **別のつまみ**（src/kernels/score-storage.ts）。`s16` は S を `array<u32>` に
 * `pack2x16float` で詰める形で、①が書き ②③ が読む以上 **3 カーネルが同時に切り替わる**
 * （段ごとの混成はあり得ない）。丸めが増えるのは格納の 1 点だけなので、上記 4 つの根拠は
 * 「S が f16 に丸まった f32 経路」としてそのまま生きる。
 */

import { CodegenError } from "../codegen/errors.ts";
import { type GemmCompute, gemmComputeKeyPart, gemmKeyPart, gemmParams, gemmWgsl } from "./gemm.ts";
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

export const attentionQkKey = (
  v4: boolean,
  compute: GemmCompute = "f32",
  score: ScoreStorage = "f32",
): string =>
  `attention_qk:v1:f32:${gemmKeyPart(v4)}${gemmComputeKeyPart(compute)}${scoreKeyPart(score)}`;

export const attentionPvKey = (
  v4: boolean,
  compute: GemmCompute = "f32",
  score: ScoreStorage = "f32",
): string =>
  `attention_pv:v1:f32:${gemmKeyPart(v4)}${gemmComputeKeyPart(compute)}${scoreKeyPart(score)}`;

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
): string => gemmWgsl({ op: "attention_qk", v4, compute, score });

export const attentionPvWgsl = (
  v4: boolean,
  compute: GemmCompute = "f32",
  score: ScoreStorage = "f32",
): string => gemmWgsl({ op: "attention_pv", v4, compute, score });

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
 * ① の uniform（`{m, n, k, scale}`）。4 語目が**半スケール**の f32 ビット列。
 *
 * MUST: scale を WGSL に焼かない（値の種類だけパイプラインが増える — masked_fill の
 * 埋め値と同じ理由）。
 */
export const attentionQkParams = (
  m: number,
  n: number,
  k: number,
  scale: number,
): Uint32Array<ArrayBuffer> => {
  if (!Number.isFinite(scale)) {
    throw new CodegenError(`attention_qk params: scale は有限の数値（${scale}）`);
  }
  const params = gemmParams("attention_qk", m, n, k);
  new Float32Array(params.buffer)[3] = scale;
  return params;
};

/** ③ の uniform（`{m, n, k}` — bmm と同じ 3 語）。 */
export const attentionPvParams = (
  m: number,
  n: number,
  k: number,
): Uint32Array<ArrayBuffer> => gemmParams("attention_pv", m, n, k);

/**
 * ② の uniform。WGSL の uniform アドレス空間では struct の整列が 16 バイトになるため、
 * 2 語ぶんの内容でも 16 バイト確保する MUST（softmax と同じ）。
 */
export const attentionStatsParams = (
  rows: number,
  dim: number,
): Uint32Array<ArrayBuffer> => {
  if (!Number.isSafeInteger(rows) || rows < 0 || !Number.isSafeInteger(dim) || dim < 1) {
    throw new CodegenError(
      `attention_stats params: rows は非負整数 / dim は正整数（${rows}, ${dim}）`,
    );
  }
  const params = new Uint32Array(4);
  params[0] = rows;
  params[1] = dim;
  return params;
};
