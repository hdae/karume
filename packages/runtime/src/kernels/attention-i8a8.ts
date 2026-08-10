/**
 * 融合 attention ①QK / ③PV の **i8a8 変種**（活性 i8 の**整数内積**）— 設計は
 * docs/research/2026-08-04-attention-a8-design.md §2.1 / §2.2 / §2.3 / §4.1 / §4.2、
 * opt-in は `SessionOptions.attentionCompute: "i8a8"`（既定 `"f32"`）。
 *
 * ```
 * ①QK: acc[i32] = Σ_d qq[b,m,d] · kq[b,n,d]          （厳密な整数・丸め 0 回）
 *       S        = f32(acc) · (qs'[b,m] · ks'[b,n])   （qs' = qs·scale / ks' = ks·scale）
 * ③PV: acc[i32] = Σ_n qP[b,m,n] · vq[b,d,n]          （qP は A タイル充填で作る非実体化の列）
 *       O        = f32(acc) · (prow[b,m] · vs[b,d])   （prow = inv·(1/127)）
 * ```
 *
 * ## linear の i8a8（src/kernels/linear-i8a8.ts）との関係
 *
 * `S[b,m,n] = Σ_d q[b,m,d]·k[b,n,d]` は **縮約軸 D が q / k の両方で最内連続**なので、
 * linear の `x[m,k]`（k 連続）× `w[n,k]`（k 連続）と添字の形が完全に一致する。したがって
 * 活性量子化は {@link "./quantize-rows.ts"} を**無改変で 2 回呼ぶだけ**で足り、内積ループ・
 * 共有タイルの `[pack][行]` 配置・K 端数の 0 埋めも linear の i8a8 と同型になる。
 *
 * 生成器を linear と共有しないのは、違いが**断片差し込みでは吸収できない位置**に散っている
 * ため（バッチ base 5 本 / dequant が両側 scale + 半スケール / bias 無し）。共有するのは
 * **タイル幾何の定数**（{@link GEMM_TILE} / {@link GEMM_WORKGROUP}）と**整数内積の実体**
 * （{@link i8a8IdotWgsl}）で、数値契約に効く部分の正本は 1 箇所に保つ。
 *
 * ## 数値契約（GPU vs TS 参照 atol=0 — src/reference/i8a8.ts）
 *
 * 整数縮約が順序非依存の厳密値なので、出力 1 要素あたりの浮動小数演算は
 * `qs = amax_q·(1/127)` / `ks = amax_k·(1/127)` / `qs·scale` / `ks·scale` /
 * `qs'·ks'` / `f32(acc)·(…)` の 6 つだけ。どれも round-to-nearest-even で決定的
 * （唯一のほころびは量子化の `x/s` = データ依存の除算で、WGSL の 2.5 ULP 許容ぶん
 * 丸め境界近傍の要素が ±1 段揺れうる — テストが `quantizeRowsTieMargin` で余裕を実測して門にする）。
 *
 * MUST: dequant の乗算順序を固定する — **`qs' · ks'` を先に 1 つの f32 へ畳んでから**
 * `f32(acc)` に掛ける。`f32(acc)·qs'·ks'` の逐次形は丸めの位置が動いて参照とのビット一致が
 * 崩れる（linear の `xs·wscale` と同型の MUST）。**bias が無いので `fma` は使わない**。
 * MUST: 半スケール（ADR 0023 の `√scale_factor`）は **dequant 側**へ q / k の両方に掛ける。
 * 量子化の**前**に掛ける形（`q·scale` を量子化する）は、amax も同じ倍率で動くので整数値は
 * ほぼ同じだが f32 の丸めが 1 段増えるだけの純損で、しかも `quantize_rows` を無改変で
 * 使えなくなる（設計 §2.1 の裁定）。attrs の契約（`scale` は √scale のまま）は不変。
 * MUST: 行 scale は `qs[batch·M + arow]`・列 scale は `ks[batch·N + wcol]`。取り違えは
 * 例外の出ない誤値で、**B·H ≥ 2 かつ head ごとに統計が違う形** × **m / n タイル 2 枚以上**
 * のテストだけが検出器になる（ADR 0024 の MUST ④ と同型の罠）。
 *
 * ## ③PV の構造（設計 §2.2 / §2.3 — ①QK と決定的に違う 3 点）
 *
 * 1. **A 側（P̃）は量子化カーネルを通らない**。`P̃ = exp(S−m)` は行内 max が構造的に 1.0
 *    （`exp(0)` は f32 で厳密）なので、per-token amax 量子化を掛けても scale は **1/127 に
 *    縮退する**。したがって amax を取る意味が無く、A タイル充填が `round(127·exp(S−m))` を
 *    その場で作って 4 詰めする（**除算ゼロ**・上限 127 は構造的で clamp も不要）。
 *    MUST: amax を取れる形に書かないこと — 「行ごとに適応している」と誤読され、
 *    ランタイムに実装できない楽観上界（設計 §5.1 の粒度表）へ静かに寄っていく。
 * 2. **V の scale は per-column（N 全体の amax）でなければならない**。縮約軸は n なので、
 *    `f32(acc)·s` 形が成立する条件は **s が n に依存しないこと**そのもの。per-token
 *    （V の行 n ごと）にすると scale が縮約軸上で変わり総和から括り出せない — 黙って使うと
 *    例外の出ない誤値になる（linear で `wcol` を行に流用する誤りと同型 — ADR 0024 の MUST④）。
 *    executor は既存の `strided`（permute）で Vᵀ`[B·H, D, N]` を作り、その上で
 *    `quantize_rows`（行 = `(b,h,d)`）を呼ぶ — **per-column 量子化と dp4a が要求する
 *    N 連続パックが同時に得られる**ので新カーネルは要らない。
 * 3. **行の `1/l` は dequant 側へ移る**。`prow = inv·(1/127)` は行ループ不変なので、
 *    `O = f32(acc)·(prow·vs[d])` が linear の `xs·wscale` と同型になる。②行統計は
 *    **f32 のまま 1 バイトも変えない**（分母を量子化後の総和にする案は波 Q0 の実測で
 *    明確悪化と出て不採用確定 — 設計 §4.3 に対する反証）。
 *
 * ## ③PV の数値契約（②とは違い O に atol=0 は立たない — 設計 §4.2）
 *
 * `qP = round(127·exp(S−m))` の `exp` は WGSL と JS でビット一致する保証が無く、
 * `127·exp` は [0,127] に密に分布するので丸め境界に落ちる要素が必ず出る。そこで契約を
 * **2 段に割る**:
 *
 * - **整数を受け取ってからの純関数**（`referenceAttentionPvI8a8Core`）とは **atol=0**。
 *   ここは `f32(acc)` / `prow·vs` / 積 の 3 演算しか無い。
 * - `qP` の生成は別の門: TS 参照との不一致は **必ず ±1 段**（±2 が 1 件でも出たら実装バグ）で、
 *   不一致率そのものを記録する（tests/gpu_attention_pv_i8a8_test.ts）。
 *
 * MUST: dequant の乗算順序を固定する — **`prow · vs[d]` を先に 1 つの f32 へ畳んでから**
 * `f32(acc)` に掛ける（①QK と同型）。MUST: `1/127` は**乗算**で作る（除算は 2.5 ULP まで
 * 許される — quantize-rows.ts の実測）。
 * NOTE（契約の外）: S に非有限値がある行の `qP` は**実装依存**（`vec4<i32>(NaN)` の値が
 * WGSL で未定義）。f32 変種は NaN をそのまま O へ伝播させるので、**そこだけ挙動が違う**。
 * quantize-rows.ts の「非有限行の量子化値は契約の外」と同じ扱いで、突合の対象にしない。
 *
 * ## 波の位置づけ（段階導入）
 *
 * 本ファイルが受け持つのは ①QK と ③PV で、②行統計は f32 のまま走る（設計 §7 の波 1 + 波 2）。
 * 適格判定は**段ごとに独立**（①は `D % 4 == 0`・③は `N % 4 == 0`）なので、
 * **片方だけ i8a8 の混成**が起こりうる（executor の `#encodeAttention`）。
 */

import { CodegenError } from "../codegen/errors.ts";
import { GEMM_TILE, GEMM_WORKGROUP, gemmKeyPart, gemmParams } from "./gemm.ts";
import { i8a8IdotWgsl, LINEAR_I8A8_MAX_K } from "./linear-i8a8.ts";
import {
  assertScoreStorageSupported,
  scoreArrayType,
  scoreKeyPart,
  scoreNote,
  scoreQuadLoaderWgsl,
  scoreReadQuad,
  type ScoreStorage,
  scoreStoreQuad,
  scoreStoreWgsl,
} from "./score-storage.ts";

/** q の per-token scale（`[B·H·M]`）の束縛（出力 S の次の番号）。 */
export const ATTENTION_QK_Q_SCALE_BINDING = 4;
/** k の per-token scale（`[B·H·N]`）の束縛。**出力列ごとの scale** に相当する。 */
export const ATTENTION_QK_K_SCALE_BINDING = 5;
/**
 * Vᵀ の per-row scale（`[B·H·D]`）の束縛（③PV の出力 O の次の番号）。Vᵀ の「行」= `(b,h,d)`
 * なので、これは **V の per-column（N 全体の amax）scale** そのもの（上の MUST 2）。
 */
export const ATTENTION_PV_V_SCALE_BINDING = 5;

/**
 * P̃ の量子化格子の端。**quantize-rows.ts の `QUANTIZE_ROWS_ABS_MAX` を輸入しない** —
 * あちらは「行 amax から作る適応 scale の格子」で、こちらは「行内 max が構造的に 1.0 だから
 * scale が 1/127 に縮退する」という**別の事実**（設計 §2.2）。値が一致するのは偶然に近く、
 * 束ねると片方の変更がもう片方へ黙って伝播する。
 */
const P_LATTICE_MAX = 127;
/** MUST: `1/127` は乗算で作る（WGSL の f32 除算は 2.5 ULP まで許される）。 */
const INV_P_LATTICE = 1 / P_LATTICE_MAX;

/** 1 スレッドが持つ出力の一辺。 */
const REG = 4;
/** workgroup の一辺（16×16 = 256 スレッド）。 */
const WG = GEMM_WORKGROUP;
/** K タイル幅（16 要素 = 4 パック）。 */
const TILE_K = 16;
/** K タイルあたりのパック数。q / k 充填のスレッド割当に使う。 */
const K_PACKS = TILE_K / REG;

/**
 * パイプラインキー。`linear:v2:f32:…` → `linear:v3:i8a8:…:dp4a` の前例に厳密に倣う
 * （**世代を上げ、dtype 欄を `i8a8` にし、内積変種を末尾に付ける**）。
 * MUST: 既存の f32 / `:c16` キーは 1 文字も動かさない（スナップショットが検出器）。
 */
export const attentionQkI8a8Key = (
  v4: boolean,
  dp4a: boolean,
  score: ScoreStorage = "f32",
): string =>
  `attention_qk:v2:i8a8:${gemmKeyPart(v4)}:${dp4a ? "dp4a" : "dp4aEmu"}${scoreKeyPart(score)}`;

/**
 * S を `vec4<f32>` で書けるか。**D 側の条件は適格判定（`D % 4 == 0`）が既に担っている**ので
 * ここは N（出力の列）だけを見る（linear の i8a8 と同じ規律）。
 */
export const attentionQkI8a8UsesVec4 = (n: number): boolean => n % 4 === 0;

/**
 * i32 accumulator の初期化（①QK / ③PV 共通）。**配列 1 本ではなく `acc0..acc3` の名前付き
 * 変数**にするのは、`acc[i]` の動的添字がアドレス可能な関数ローカル領域を要求し、レジスタに
 * 載らずローカルメモリへ落ちるため（linear の i8a8 で先に確立した展開 — 機序は
 * {@link "./linear-i8a8.ts"} の同名関数）。展開しても縮約は i32 の厳密加算のまま・行ごとに
 * 独立なので、**返る整数は 1 ビットも変わらない**（RTX 3080 Ti の変種スイープで
 * 幾何もキーも変えずに QK ×1.408 / PV ×1.451・出力はバイト同一 —
 * docs/research/2026-08-10-kernel-variant-sweep.md §3.2）。
 */
const accumulatorInit = (): string =>
  Array.from({ length: REG }, (_, i) => `  var acc${i} = vec4<i32>(0);`).join("\n");

/**
 * K パック内側の 4 行更新（①QK / ③PV 共通）。{@link accumulatorInit} と対で展開する。
 * 1 出力あたりの加算順序（K タイル昇順・パック内 4 語まとめ）は展開前と同一。
 */
const accumulatorUpdate = (): string =>
  Array.from(
    { length: REG },
    (_, i) =>
      `      let a${i} = sa[p * ${GEMM_TILE}u + lid.y * ${REG}u + ${i}u];
      acc${i} = acc${i} + vec4<i32>(idot(a${i}, b0), idot(a${i}, b1), idot(a${i}, b2), idot(a${i}, b3));`,
  ).join("\n");

/** 2 行目以降の行番号は展開時に確定するので、その場で let に落とす（0 行目は `orow0`）。 */
const rowDecl = (i: number, indent: string): string =>
  i === 0 ? "" : `${indent}let orow${i} = orow0 + ${i}u;\n`;

/**
 * ①QK の書き出し。行ループは codegen 時に展開して {@link accumulatorInit} の `acc0..acc3` を
 * 静的に読む（`acc[i]` の動的添字を残さないための展開 — 理由は同関数の docstring）。
 * ガードの構造・`qs'·ks'` の畳み順・半スケールの位置は展開前と同一。
 */
const store = (v4: boolean, score: ScoreStorage): string => {
  const rows = `  let orow0 = wid.y * ${GEMM_TILE}u + lid.y * ${REG}u;`;
  if (v4) {
    const rowStores = Array.from(
      { length: REG },
      (_, i) =>
        `${rowDecl(i, "    ")}    if (orow${i} < dims.m) {
      ${
          scoreStoreQuad(
            "s",
            score,
            `sbase + orow${i} * n4 + ocq`,
            `vec4<f32>(acc${i}) * ((qscale[qsbase + orow${i}] * dims.scale) * ks)`,
          )
        }
    }`,
    ).join("\n");
    return `  let ocq = wid.x * ${WG}u + lid.x;
${rows}
  if (ocq < n4) {
    // n % ${REG} == 0 かつ ocq < n4 なので oc + ${REG - 1} < n。
    // MUST: 半スケールは dequant 側で q / k の**両方**へ（列側はここで 1 度だけ）
    let oc = ocq * ${REG}u;
    let ks = vec4<f32>(
      kscale[ksbase + oc],
      kscale[ksbase + oc + 1u],
      kscale[ksbase + oc + 2u],
      kscale[ksbase + oc + 3u],
    ) * dims.scale;
    // MUST: qs'·ks' を先に 1 つの f32 へ畳んでから f32(acc) に掛ける（bias が無いので fma は無い）
${rowStores}
  }`;
  }
  const write = (i: number, component: string, offset: string): string =>
    `s[obase + ocol${offset}] = f32(acc${i}.${component}) * (qs * (kscale[ksbase + ocol${offset}] * dims.scale));`;
  const rowStores = Array.from(
    { length: REG },
    (_, i) =>
      `${rowDecl(i, "  ")}  if (orow${i} < dims.m) {
    let obase = sbase + orow${i} * dims.n;
    // MUST: qs'·ks' を先に 1 つの f32 へ畳んでから f32(acc) に掛ける（bias が無いので fma は無い）
    let qs = qscale[qsbase + orow${i}] * dims.scale;
    if (ocol < dims.n) {
      ${write(i, "x", "")}
    }
    if (ocol + 1u < dims.n) {
      ${write(i, "y", " + 1u")}
    }
    if (ocol + 2u < dims.n) {
      ${write(i, "z", " + 2u")}
    }
    if (ocol + 3u < dims.n) {
      ${write(i, "w", " + 3u")}
    }
  }`,
  ).join("\n");
  return `  let ocol = wid.x * ${GEMM_TILE}u + lid.x * ${REG}u;
${rows}
${rowStores}`;
};

export const attentionQkI8a8Wgsl = (
  v4: boolean,
  dp4a: boolean,
  score: ScoreStorage = "f32",
): string => {
  assertScoreStorageSupported("attention_qk i8a8", score, v4);
  return `// karume attention_qk (S[b,m,n] = q[b,m,d] · k[b,n,d]ᵀ, q/k とも per-token i8 の整数内積${
    dp4a ? "" : "・エミュ"
  }, 半スケールは dequant 側${scoreNote(score)}, レジスタ ${GEMM_TILE}x${GEMM_TILE} タイル${
    v4 ? " + vec4" : ""
  })
struct Dims {
  m: u32,
  n: u32,
  k: u32,
  scale: f32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> qq: array<u32>;
@group(0) @binding(2) var<storage, read> kq: array<u32>;
@group(0) @binding(3) var<storage, read_write> s: array<${
    scoreArrayType(score, v4 ? "vec4<f32>" : "f32")
  }>;
@group(0) @binding(${ATTENTION_QK_Q_SCALE_BINDING}) var<storage, read> qscale: array<f32>;
@group(0) @binding(${ATTENTION_QK_K_SCALE_BINDING}) var<storage, read> kscale: array<f32>;
${scoreStoreWgsl("s", score)}
${i8a8IdotWgsl(dp4a)}

// 共有タイルは **[pack][行] / [pack][列]** 配置（linear の i8a8 と同じ — 内側の読みが
// 語ストライド ${REG} になりバンク衝突が 8-way から 2-way に減る）
var<workgroup> sa: array<u32, ${K_PACKS * GEMM_TILE}>;
var<workgroup> sb: array<u32, ${K_PACKS * GEMM_TILE}>;

@compute @workgroup_size(${WG}, ${WG})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.y * ${WG}u + lid.x;
  // 適格条件で D % 4 == 0 なので、行頭は必ず語境界に来る（i8 ペイロードは平坦添字 4 詰め）
  let k4 = dims.k / ${REG}u;
${
    v4
      ? `  let n4 = dims.n / ${REG}u;\n`
      : ""
  }  // B と H を畳んだバッチ軸（workgroup 単位で一様 — 内側の workgroupBarrier が WGSL の
  // 一様性要件を満たすための前提）。MUST: q / k のペイロードは **pack 単位**、scale は行数
  // 単位、S は要素${v4 ? "（v4 では quad）" : ""}単位で数える — 単位を取り違えると B·H ≥ 2 で
  // 隣の head を読み書きする（例外なしの誤値で、B=H=1 のテストには出ない）
  let qbase = wid.z * dims.m * k4;
  let kbase = wid.z * dims.n * k4;
  let qsbase = wid.z * dims.m;
  let ksbase = wid.z * dims.n;
  let sbase = wid.z * dims.m * ${v4 ? "n4" : "dims.n"};
  // q タイルの担当（${GEMM_TILE} 行 × ${K_PACKS} pack = ${GEMM_TILE * K_PACKS} スレッド）
  let ar = tid / ${K_PACKS}u;
  let ap = tid % ${K_PACKS}u;
  let arow = wid.y * ${GEMM_TILE}u + ar;
  let arow_base = qbase + arow * k4;
  // k タイルの担当（${GEMM_TILE} 列（N）× ${K_PACKS} pack = ${GEMM_TILE * K_PACKS} スレッド）。
  // k は [N,D] のまま読む（連続方向が D = パック方向なので**転置が要らない**）
  let wc = tid / ${K_PACKS}u;
  let wp = tid % ${K_PACKS}u;
  let wcol = wid.x * ${GEMM_TILE}u + wc;
  let krow_base = kbase + wcol * k4;
  // ループ条件は uniform（dims は uniform バッファ）— 内側の workgroupBarrier が
  // WGSL の一様性要件を満たすために必要
  let tiles = (k4 + ${K_PACKS - 1}u) / ${K_PACKS}u;
${accumulatorInit()}
  for (var t = 0u; t < tiles; t = t + 1u) {
    // 範囲外は 0 で埋める（dot4I8Packed(0, x) == 0 なので K 端数でも結果は厳密）
    let apack = t * ${K_PACKS}u + ap;
    var av = 0u;
    if (arow < dims.m && apack < k4) {
      av = qq[arow_base + apack];
    }
    sa[ap * ${GEMM_TILE}u + ar] = av;
    let kpack = t * ${K_PACKS}u + wp;
    var kv = 0u;
    if (wcol < dims.n && kpack < k4) {
      kv = kq[krow_base + kpack];
    }
    sb[wp * ${GEMM_TILE}u + wc] = kv;
    workgroupBarrier();
    // 共有ロード 5 回（k の 4 語 + q の 1 語）で ${REG * REG} 個の整数内積 = ${
    REG * REG * REG
  } MAC。
    // 縮約は i32 の厳密加算なので**順序に依存しない**（f32 骨格と違い加算順は数値契約に無い）
    for (var p = 0u; p < ${K_PACKS}u; p = p + 1u) {
      let bcol = p * ${GEMM_TILE}u + lid.x * ${REG}u;
      let b0 = sb[bcol];
      let b1 = sb[bcol + 1u];
      let b2 = sb[bcol + 2u];
      let b3 = sb[bcol + 3u];
${accumulatorUpdate()}
    }
    workgroupBarrier();
  }
${store(v4, score)}
}
`;
};

/**
 * uniform の Dims（`{m,n,k}` + 4 語目が**半スケール**の f32 ビット列 — f32 変種の
 * {@link "./attention.ts"} `attentionQkParams` と同じ並び）。
 *
 * MUST: scale を WGSL に焼かない（値の種類だけパイプラインが増える）。
 */
export const attentionQkI8a8Params = (
  m: number,
  n: number,
  k: number,
  scale: number,
): Uint32Array<ArrayBuffer> => {
  if (!Number.isFinite(scale)) {
    throw new CodegenError(`attention_qk i8a8 params: scale は有限の数値（${scale}）`);
  }
  if (!Number.isSafeInteger(k) || k < 0 || k % 4 !== 0) {
    throw new CodegenError(`attention_qk i8a8 params: k は 4 の倍数の非負整数（${k}）`);
  }
  if (k > LINEAR_I8A8_MAX_K) {
    throw new CodegenError(
      `attention_qk i8a8 params: k=${k} が i32 縮約の門 ${LINEAR_I8A8_MAX_K} を超える`,
    );
  }
  const params = gemmParams("attention_qk", m, n, k);
  new Float32Array(params.buffer)[3] = scale;
  return params;
};

// ---------------------------------------------------------------------------
// ③PV（P̃ は非実体化のまま整数化 / V は per-column i8）
// ---------------------------------------------------------------------------

/** ③PV の i8a8 変種のキー（①QK と同じ規約 — 世代 v2・dtype 欄 i8a8・末尾に内積変種）。 */
export const attentionPvI8a8Key = (
  v4: boolean,
  dp4a: boolean,
  score: ScoreStorage = "f32",
): string =>
  `attention_pv:v2:i8a8:${gemmKeyPart(v4)}:${dp4a ? "dp4a" : "dp4aEmu"}${scoreKeyPart(score)}`;

/**
 * O を `vec4<f32>` で書けるか。**縮約側（N）の条件は適格判定（`N % 4 == 0`）が既に担っている**
 * ので、ここは出力の列 = D だけを見る（①QK / linear の i8a8 と同じ規律）。
 */
export const attentionPvI8a8UsesVec4 = (depth: number): boolean => depth % 4 === 0;

/**
 * A タイル（P̃）の充填。**S を読んで `round(127·exp(S−m))` をその場で作り 4 詰めする**ので、
 * P は 1 バイトも実体化されない（ADR 0023 の性質をそのまま保つ）。
 *
 * MUST: 変換を掛けるのは**範囲内の要素だけ**。範囲外に `exp(0−m)·127` を掛けると 0 でない
 * 整数が内積へ混ざり、K 端数タイルだけが静かに誤る（gemm.ts の `fillA` と同じ MUST）。
 * MUST: `exp` は**成分ごとのスカラ式**（`vec4` へまとめない — 超越関数のベクトル版は実装依存）。
 * NOTE: clamp を置かないのは、`exp(S−m) ≤ 1` が `m = amax` から構造的だから（設計 §2.2）。
 * 「格子を 128 にする」退行の検出器は clamp ではなく **この 127 と dequant 側の 1/127** で、
 * どちらも参照との突合が拾う。
 */
const fillPTile = (score: ScoreStorage): string => {
  const lane = (component: string): string =>
    `round(exp(raw.${component} - row_max) * ${P_LATTICE_MAX}.0)`;
  return `    // 範囲外は 0 で埋める（qP = 0 は内積に寄与しないので K 端数でも結果は厳密）
    let apack = t * ${K_PACKS}u + ap;
    var av = 0u;
    if (arow < dims.m && apack < k4) {
      let raw = ${scoreReadQuad("s", score, "arow_base + apack")};
      // P̃ の scale は **1/127 固定**（行内 max が exp(0) = 1 で構造的 — amax は取らない）。
      // 除算が 1 つも無いので WGSL の 2.5 ULP 問題の外側にいる
      let p = vec4<f32>(
        ${["x", "y", "z", "w"].map(lane).join(",\n        ")},
      );
      av = pack4xI8(vec4<i32>(p));
    }
    sa[ap * ${GEMM_TILE}u + ar] = av;`;
};

/**
 * ③PV の書き出し。①QK の {@link store} と同じく行ループは codegen 時に展開して
 * `acc0..acc3` を静的に読む（理由は {@link accumulatorInit} の docstring）。ガードの構造・
 * `prow·vs` の畳み順・`1/127` の乗算は展開前と同一。
 */
const pvStore = (v4: boolean): string => {
  const rows = `  let orow0 = wid.y * ${GEMM_TILE}u + lid.y * ${REG}u;`;
  // MUST: prow は**出力行**の統計（`rbase + orow`）。A タイル充填で引いた `arow` の側を
  // 流用すると、担当が違うスレッドの行 inv が乗る（B·H = M = 1 でしか一致しない）。
  const prow = (i: number, indent: string): string =>
    `${indent}let prow = stats[(rbase + orow${i}) * 2u + 1u] * ${INV_P_LATTICE};`;
  if (v4) {
    const rowStores = Array.from(
      { length: REG },
      (_, i) =>
        `${rowDecl(i, "    ")}    if (orow${i} < dims.m) {
${prow(i, "      ")}
      o[obase + orow${i} * n4 + ocq] = vec4<f32>(acc${i}) * (prow * vs);
    }`,
    ).join("\n");
    return `  let ocq = wid.x * ${WG}u + lid.x;
${rows}
  if (ocq < n4) {
    // D % ${REG} == 0 かつ ocq < n4 なので oc + ${REG - 1} < D。
    // MUST: 列 scale は Vᵀ の**行** = (b, h, d)（= V の per-column scale）。行 scale の側と
    // 取り違えても例外は出ない
    let oc = ocq * ${REG}u;
    let vs = vec4<f32>(
      vscale[vsbase + oc],
      vscale[vsbase + oc + 1u],
      vscale[vsbase + oc + 2u],
      vscale[vsbase + oc + 3u],
    );
    // MUST: prow·vs を先に 1 つの f32 へ畳んでから f32(acc) に掛ける（bias が無いので fma は無い）
${rowStores}
  }`;
  }
  const write = (i: number, component: string, offset: string): string =>
    `o[orow_base + ocol${offset}] = f32(acc${i}.${component}) * (prow * vscale[vsbase + ocol${offset}]);`;
  const rowStores = Array.from(
    { length: REG },
    (_, i) =>
      `${rowDecl(i, "  ")}  if (orow${i} < dims.m) {
    let orow_base = obase + orow${i} * dims.n;
    // MUST: prow·vs を先に 1 つの f32 へ畳んでから f32(acc) に掛ける（bias が無いので fma は無い）
${prow(i, "    ")}
    if (ocol < dims.n) {
      ${write(i, "x", "")}
    }
    if (ocol + 1u < dims.n) {
      ${write(i, "y", " + 1u")}
    }
    if (ocol + 2u < dims.n) {
      ${write(i, "z", " + 2u")}
    }
    if (ocol + 3u < dims.n) {
      ${write(i, "w", " + 3u")}
    }
  }`,
  ).join("\n");
  return `  let ocol = wid.x * ${GEMM_TILE}u + lid.x * ${REG}u;
${rows}
${rowStores}`;
};

export const attentionPvI8a8Wgsl = (
  v4: boolean,
  dp4a: boolean,
  score: ScoreStorage = "f32",
): string =>
  `// karume attention_pv (O[b,m,d] = P[b,m,n] · v[b,n,d], P̃ = round(127·exp(S − m)) は非実体化${
    dp4a ? "" : "・エミュ"
  }, v は per-column i8（Vᵀ 連続）${scoreNote(score)}, レジスタ ${GEMM_TILE}x${GEMM_TILE} タイル${
    v4 ? " + vec4" : ""
  })
struct Dims {
  m: u32,
  n: u32,
  k: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
// 適格条件で N % 4 == 0 なので、S の行頭は常に quad 境界に来る（P̃ の 4 詰めと同じ刻み）
@group(0) @binding(1) var<storage, read> s: array<${scoreArrayType(score, "vec4<f32>")}>;
@group(0) @binding(2) var<storage, read> vq: array<u32>;
@group(0) @binding(3) var<storage, read> stats: array<f32>;
@group(0) @binding(4) var<storage, read_write> o: array<${v4 ? "vec4<f32>" : "f32"}>;
@group(0) @binding(${ATTENTION_PV_V_SCALE_BINDING}) var<storage, read> vscale: array<f32>;
${scoreQuadLoaderWgsl("s", score)}
${i8a8IdotWgsl(dp4a)}

// 共有タイルは **[pack][行] / [pack][列]** 配置（①QK / linear の i8a8 と同じ — 内側の読みが
// 語ストライド ${REG} になりバンク衝突が 8-way から 2-way に減る）
var<workgroup> sa: array<u32, ${K_PACKS * GEMM_TILE}>;
var<workgroup> sb: array<u32, ${K_PACKS * GEMM_TILE}>;

@compute @workgroup_size(${WG}, ${WG})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.y * ${WG}u + lid.x;
  // 縮約軸 N は S 側では quad 単位・vq 側では pack 単位（同じ 4 要素の刻み）
  let k4 = dims.k / ${REG}u;
${
    v4
      ? `  let n4 = dims.n / ${REG}u;\n`
      : ""
  }  // B と H を畳んだバッチ軸（workgroup 単位で一様 — 内側の workgroupBarrier が WGSL の
  // 一様性要件を満たすための前提）。MUST: S は quad 単位、vq は pack 単位、行統計は行あたり
  // 2 語、vscale は D 本、O は要素${v4 ? "（v4 では quad）" : ""}単位で数える — 単位を
  // 取り違えると B·H ≥ 2 で隣の head を読み書きする（例外なしの誤値で B=H=1 では出ない）
  let sbase = wid.z * dims.m * k4;
  let vbase = wid.z * dims.n * k4;
  let rbase = wid.z * dims.m;
  let vsbase = wid.z * dims.n;
  let obase = wid.z * dims.m * ${v4 ? "n4" : "dims.n"};
  // P̃ タイルの担当（${GEMM_TILE} 行 × ${K_PACKS} pack = ${GEMM_TILE * K_PACKS} スレッド）
  let ar = tid / ${K_PACKS}u;
  let ap = tid % ${K_PACKS}u;
  let arow = wid.y * ${GEMM_TILE}u + ar;
  let arow_base = sbase + arow * k4;
  // 行の最大 m（K タイルループ不変なので 1 度だけ引く）。端タイルでは arow >= M がありうるので
  // 添字を 0 へ倒す（読んだ値は arow < M の枝でしか使われない）
  let stat_at = select(0u, (rbase + arow) * 2u, arow < dims.m);
  let row_max = stats[stat_at];
  // Vᵀ タイルの担当（${GEMM_TILE} 列（D）× ${K_PACKS} pack = ${GEMM_TILE * K_PACKS} スレッド）。
  // vq は [D,N] の N 連続（= パック方向）なので**転置が要らない**
  let wc = tid / ${K_PACKS}u;
  let wp = tid % ${K_PACKS}u;
  let wcol = wid.x * ${GEMM_TILE}u + wc;
  let vrow_base = vbase + wcol * k4;
  // ループ条件は uniform（dims は uniform バッファ）— 内側の workgroupBarrier が
  // WGSL の一様性要件を満たすために必要
  let tiles = (k4 + ${K_PACKS - 1}u) / ${K_PACKS}u;
${accumulatorInit()}
  for (var t = 0u; t < tiles; t = t + 1u) {
${fillPTile(score)}
    let vpack = t * ${K_PACKS}u + wp;
    var vv = 0u;
    if (wcol < dims.n && vpack < k4) {
      vv = vq[vrow_base + vpack];
    }
    sb[wp * ${GEMM_TILE}u + wc] = vv;
    workgroupBarrier();
    // 共有ロード 5 回（v の 4 語 + P̃ の 1 語）で ${REG * REG} 個の整数内積 = ${
    REG * REG * REG
  } MAC。
    // 縮約は i32 の厳密加算なので**順序に依存しない**（f32 骨格と違い加算順は数値契約に無い）
    for (var p = 0u; p < ${K_PACKS}u; p = p + 1u) {
      let bcol = p * ${GEMM_TILE}u + lid.x * ${REG}u;
      let b0 = sb[bcol];
      let b1 = sb[bcol + 1u];
      let b2 = sb[bcol + 2u];
      let b3 = sb[bcol + 3u];
${accumulatorUpdate()}
    }
    workgroupBarrier();
  }
${pvStore(v4)}
}
`;

/**
 * uniform の Dims（`{m, n, k}` = `{M, D, N}` — f32 変種の {@link "./attention.ts"}
 * `attentionPvParams` と同じ並び。③PV に半スケールは登場しない）。
 */
export const attentionPvI8a8Params = (
  m: number,
  n: number,
  k: number,
): Uint32Array<ArrayBuffer> => {
  if (!Number.isSafeInteger(k) || k < 0 || k % 4 !== 0) {
    throw new CodegenError(`attention_pv i8a8 params: k（= N）は 4 の倍数の非負整数（${k}）`);
  }
  if (k > LINEAR_I8A8_MAX_K) {
    throw new CodegenError(
      `attention_pv i8a8 params: k=${k} が i32 縮約の門 ${LINEAR_I8A8_MAX_K} を超える`,
    );
  }
  return gemmParams("attention_pv", m, n, k);
};
