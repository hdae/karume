/**
 * 融合 attention ①QK / ③PV の **i8a8 変種**（活性 i8 の**整数内積**）— 設計は
 * docs/research/2026-08-04-attention-a8-design.md §2.1 / §2.2 / §2.3 / §4.1 / §4.2、
 * opt-in は `SessionOptions.attentionCompute: "a8"`（既定 `"f32"`）。
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
 * **タイル幾何とその算術**（{@link "./i8a8-geometry.ts"}）と**整数内積の実体**
 * （{@link i8a8IdotWgsl}）で、数値契約と添字算術に効く部分の正本は 1 箇所に保つ。
 *
 * MUST: ①QK と ③PV は**別の幾何**を既定に取る（③ は N = D = 128 が 1 タイルに収まる
 * `M64N128` が勝つ — docs/research/2026-08-10-kernel-variant-sweep.md §3.2）。「1 本の最良幾何」
 * を仮定して 2 段を束ねると、どちらかが必ず劣後する。
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
import {
  assertGemmRowWindow,
  gemmParams,
  type GemmRowWindowSpan,
  ROW_WINDOW_DIMS_EXTRA,
} from "./gemm.ts";
import {
  assertI8a8Geometry,
  defaultI8a8Geometry,
  I8A8_PACK,
  i8a8AccumulatorInit,
  i8a8ASlots,
  i8a8BSlots,
  i8a8FillStride,
  type I8a8Geometry,
  i8a8GeometryKeyPart,
  i8a8GeometryNote,
  i8a8InnerProductLoop,
  i8a8KPacks,
  i8a8Threads,
  i8a8TileM,
  i8a8TileN,
} from "./i8a8-geometry.ts";
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

/** 添字の加算オフセット（0 は省く — 生成物を読める形に保つ）。 */
const at = (offset: number): string => offset === 0 ? "" : ` + ${offset}u`;

/** 出力の行番号（`orow0` を基点に展開）。 */
const rowDecls = (geometry: I8a8Geometry): string =>
  Array.from(
    { length: geometry.regM },
    (_, row) =>
      row === 0
        ? `  let orow0 = wid.y * ${i8a8TileM(geometry)}u + lid.y * ${geometry.regM}u;`
        : `  let orow${row} = orow0 + ${row}u;`,
  ).join("\n");

/** 出力の列 quad（v4 経路 — `ocq0` を基点に展開）。 */
const quadDecls = (geometry: I8a8Geometry): string => {
  const quads = geometry.regN / I8A8_PACK;
  return Array.from(
    { length: quads },
    (_, quad) =>
      quad === 0
        ? `  let ocq0 = wid.x * ${i8a8TileN(geometry) / I8A8_PACK}u + lid.x * ${quads}u;`
        : `  let ocq${quad} = ocq0 + ${quad}u;`,
  ).join("\n");
};

/** A（q / P̃）タイルの担当 — 行番号は K タイルループ不変なので prologue で 1 度だけ組む。 */
const prologueA = (geometry: I8a8Geometry, base: string, label: string): string => {
  const tileM = i8a8TileM(geometry);
  const stride = i8a8FillStride(geometry);
  const slots = Array.from(
    { length: i8a8ASlots(geometry) },
    (_, slot) =>
      `  let arow${slot} = ${slot === 0 ? `wid.y * ${tileM}u + ar` : `arow0 + ${slot * stride}u`};
  let arow_base${slot} = ${base} + arow${slot} * k4;`,
  ).join("\n");
  return `  // ${label}（${tileM} 行 × ${i8a8KPacks(geometry)} pack を ${
    i8a8Threads(geometry)
  } スレッドで ${i8a8ASlots(geometry)} 巡）
  let ar = tid / ${i8a8KPacks(geometry)}u;
  let ap = tid % ${i8a8KPacks(geometry)}u;
  let sa_at = ap * ${tileM}u + ar;
${slots}`;
};

/** B（k / Vᵀ）タイルの担当 — どちらもパック方向が最内連続なので転置が要らない。 */
const prologueB = (
  geometry: I8a8Geometry,
  base: string,
  baseName: string,
  note: string,
): string => {
  const tileN = i8a8TileN(geometry);
  const stride = i8a8FillStride(geometry);
  const slots = Array.from(
    { length: i8a8BSlots(geometry) },
    (_, slot) =>
      `  let wcol${slot} = ${slot === 0 ? `wid.x * ${tileN}u + wc` : `wcol0 + ${slot * stride}u`};
  let ${baseName}${slot} = ${base} + wcol${slot} * k4;`,
  ).join("\n");
  return `  // B タイルの担当（${tileN} 列 × ${i8a8KPacks(geometry)} pack を ${
    i8a8Threads(geometry)
  } スレッドで ${i8a8BSlots(geometry)} 巡）。
  // ${note}
  let wc = tid / ${i8a8KPacks(geometry)}u;
  let wp = tid % ${i8a8KPacks(geometry)}u;
  let sb_at = wp * ${tileN}u + wc;
${slots}`;
};

/** ①QK の A タイル充填（q をそのまま読む — linear の x と同型）。 */
const fillA = (geometry: I8a8Geometry, name: string): string => {
  const stride = i8a8FillStride(geometry);
  const slots = Array.from({ length: i8a8ASlots(geometry) }, (_, slot) =>
    `    var av${slot} = 0u;
    if (arow${slot} < dims.m && apack < k4) {
      av${slot} = ${name}[arow_base${slot} + apack];
    }
    sa[sa_at${at(slot * stride)}] = av${slot};`).join("\n");
  return `    let apack = t * ${i8a8KPacks(geometry)}u + ap;
${slots}`;
};

/** B タイルの充填（範囲外は 0 埋め — `idot(0, x) == 0` なので K 端数でも厳密）。 */
const fillB = (geometry: I8a8Geometry, name: string, baseName: string, value: string): string => {
  const stride = i8a8FillStride(geometry);
  const slots = Array.from(
    { length: i8a8BSlots(geometry) },
    (_, slot) =>
      `    var ${value}${slot} = 0u;
    if (wcol${slot} < dims.n && ${value}pack < k4) {
      ${value}${slot} = ${name}[${baseName}${slot} + ${value}pack];
    }
    sb[sb_at${at(slot * stride)}] = ${value}${slot};`,
  ).join("\n");
  return `    let ${value}pack = t * ${i8a8KPacks(geometry)}u + wp;
${slots}`;
};

/**
 * パイプラインキー。`linear:v4:i8a8:…` の前例に厳密に倣う（**世代を上げ、dtype 欄を `i8a8` に
 * し、幾何判別子を丸ごと載せ、内積変種を末尾に付ける**）。幾何がパラメタになったので
 * タイル辺だけのキーでは生成物が決まらない（{@link i8a8GeometryKeyPart}）。
 * MUST: 既存の f32 / `:c16` キーは 1 文字も動かさない（スナップショットが検出器）。
 */
export const attentionQkI8a8Key = (
  v4: boolean,
  dp4a: boolean,
  score: ScoreStorage = "f32",
  geometry: I8a8Geometry = defaultI8a8Geometry("attention_qk"),
  rowWindow = false,
): string =>
  `attention_qk:v3:i8a8:${i8a8GeometryKeyPart(geometry, v4)}:${dp4a ? "dp4a" : "dp4aEmu"}${
    scoreKeyPart(score)
  }${rowWindow ? ":rwa" : ""}`;

/**
 * S を `vec4<f32>` で書けるか。**D 側の条件は適格判定（`D % 4 == 0`）が既に担っている**ので
 * ここは N（出力の列）だけを見る（linear の i8a8 と同じ規律）。
 */
export const attentionQkI8a8UsesVec4 = (n: number): boolean => n % I8A8_PACK === 0;

/**
 * ①QK の書き出し。行・列 quad とも codegen 時に展開して `acc{行}_{列 quad}` を静的に読む
 * （`acc[i]` の動的添字を残さないための展開 — 理由は {@link i8a8AccumulatorInit}）。
 * ガードの構造・`qs'·ks'` の畳み順・半スケールの位置は幾何に依らず同一。
 */
const store = (geometry: I8a8Geometry, v4: boolean, score: ScoreStorage): string => {
  const { regM, regN } = geometry;
  const quads = regN / I8A8_PACK;
  const rows = rowDecls(geometry);
  if (v4) {
    const blocks = Array.from({ length: quads }, (_, quad) => {
      const rowStores = Array.from({ length: regM }, (_, row) =>
        `    if (orow${row} < dims.m) {
      ${
          scoreStoreQuad(
            "s",
            score,
            `sbase + orow${row} * n4 + ocq${quad}`,
            `vec4<f32>(acc${row}_${quad}) * ((qscale[qsbase + orow${row}] * dims.scale) * ks)`,
          )
        }
    }`).join("\n");
      return `  if (ocq${quad} < n4) {
    // n % ${I8A8_PACK} == 0 かつ ocq${quad} < n4 なので oc + ${I8A8_PACK - 1} < n。
    // MUST: 半スケールは dequant 側で q / k の**両方**へ（列側はここで 1 度だけ）
    let oc = ocq${quad} * ${I8A8_PACK}u;
    let ks = vec4<f32>(
      kscale[ksbase + oc],
      kscale[ksbase + oc + 1u],
      kscale[ksbase + oc + 2u],
      kscale[ksbase + oc + 3u],
    ) * dims.scale;
    // MUST: qs'·ks' を先に 1 つの f32 へ畳んでから f32(acc) に掛ける（bias が無いので fma は無い）
${rowStores}
  }`;
    }).join("\n");
    return `${quadDecls(geometry)}
${rows}
${blocks}`;
  }
  const lanes = ["x", "y", "z", "w"] as const;
  const rowStores = Array.from({ length: regM }, (_, row) => {
    const writes = Array.from({ length: regN }, (_, col) => {
      const offset = at(col);
      const acc = `acc${row}_${Math.floor(col / I8A8_PACK)}.${lanes[col % I8A8_PACK]}`;
      return `    if (ocol${offset} < dims.n) {
      s[obase + ocol${offset}] = f32(${acc}) * (qs * (kscale[ksbase + ocol${offset}] * dims.scale));
    }`;
    }).join("\n");
    return `  if (orow${row} < dims.m) {
    let obase = sbase + orow${row} * dims.n;
    // MUST: qs'·ks' を先に 1 つの f32 へ畳んでから f32(acc) に掛ける（bias が無いので fma は無い）
    let qs = qscale[qsbase + orow${row}] * dims.scale;
${writes}
  }`;
  }).join("\n");
  return `  let ocol = wid.x * ${i8a8TileN(geometry)}u + lid.x * ${regN}u;
${rows}
${rowStores}`;
};

export const attentionQkI8a8Wgsl = (
  v4: boolean,
  dp4a: boolean,
  score: ScoreStorage = "f32",
  geometry: I8a8Geometry = defaultI8a8Geometry("attention_qk"),
  rowWindow = false,
): string => {
  assertScoreStorageSupported("attention_qk i8a8", score, v4);
  assertI8a8Geometry(geometry, "attention_qk i8a8");
  const kPacks = i8a8KPacks(geometry);
  // 行窓（{@link "./gemm.ts"} `GemmRowWindow` の `"a"`）: **量子化済み q の base 2 本**だけが
  // 全 M ストライド + 行オフセットで数える。S / 行統計はブロック相対のままなので、内積ループ・
  // 共有タイル充填・dequant は 1 文字も動かない = 1 枚実行とビット同一。
  // MUST: ペイロードは pack 単位・scale は行数単位（単位を取り違えると隣の head を読む）。
  const qRows = rowWindow ? "dims.rows_full" : "dims.m";
  return `// karume attention_qk (S[b,m,n] = q[b,m,d] · k[b,n,d]ᵀ, q/k とも per-token i8 の整数内積${
    dp4a ? "" : "・エミュ"
  }, 半スケールは dequant 側${scoreNote(score)}${rowWindow ? ", 行窓 a" : ""}, ${
    i8a8GeometryNote(geometry)
  }${v4 ? " + vec4" : ""})
struct Dims {
  m: u32,
  n: u32,
  k: u32,
  scale: f32,
${rowWindow ? ROW_WINDOW_DIMS_EXTRA : ""}}
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
// 語ストライド ${I8A8_PACK} になりバンク衝突が 8-way から 2-way に減る）
var<workgroup> sa: array<u32, ${kPacks * i8a8TileM(geometry)}>;
var<workgroup> sb: array<u32, ${kPacks * i8a8TileN(geometry)}>;

@compute @workgroup_size(${geometry.wgX}, ${geometry.wgY})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.y * ${geometry.wgX}u + lid.x;
  // 適格条件で D % 4 == 0 なので、行頭は必ず語境界に来る（i8 ペイロードは平坦添字 4 詰め）
  let k4 = dims.k / ${I8A8_PACK}u;
${
    v4
      ? `  let n4 = dims.n / ${I8A8_PACK}u;\n`
      : ""
  }  // B と H を畳んだバッチ軸（workgroup 単位で一様 — 内側の workgroupBarrier が WGSL の
  // 一様性要件を満たすための前提）。MUST: q / k のペイロードは **pack 単位**、scale は行数
  // 単位、S は要素${v4 ? "（v4 では quad）" : ""}単位で数える — 単位を取り違えると B·H ≥ 2 で
  // 隣の head を読み書きする（例外なしの誤値で、B=H=1 のテストには出ない）
  let qbase = wid.z * ${qRows} * k4${rowWindow ? " + dims.row_offset * k4" : ""};
  let kbase = wid.z * dims.n * k4;
  let qsbase = wid.z * ${qRows}${rowWindow ? " + dims.row_offset" : ""};
  let ksbase = wid.z * dims.n;
  let sbase = wid.z * dims.m * ${v4 ? "n4" : "dims.n"};
${prologueA(geometry, "qbase", "q タイルの担当")}
${
    prologueB(
      geometry,
      "kbase",
      "krow_base",
      "k は [N,D] のまま読む（連続方向が D = パック方向なので**転置が要らない**）",
    )
  }
  // ループ条件は uniform（dims は uniform バッファ）— 内側の workgroupBarrier が
  // WGSL の一様性要件を満たすために必要
  let tiles = (k4 + ${kPacks - 1}u) / ${kPacks}u;
${i8a8AccumulatorInit(geometry)}
  for (var t = 0u; t < tiles; t = t + 1u) {
    // 範囲外は 0 で埋める（dot4I8Packed(0, x) == 0 なので K 端数でも結果は厳密）
${fillA(geometry, "qq")}
${fillB(geometry, "kq", "krow_base", "k")}
    workgroupBarrier();
${i8a8InnerProductLoop(geometry)}
    workgroupBarrier();
  }
${store(geometry, v4, score)}
}
`;
};

/**
 * uniform の Dims（`{m,n,k}` + 4 語目が**半スケール**の f32 ビット列 — f32 変種の
 * {@link "./attention.ts"} `attentionQkParams` と同じ並び）。
 *
 * MUST: scale を WGSL に焼かない（値の種類だけパイプラインが増える）。
 *
 * `window` は **行窓変種だけ**が渡す 5〜6 語目（f32 変種の `attentionQkParams` と同じ規律で、
 * struct は 32 バイトへ伸びる）。MUST: `m` はブロック行数（全 M ではない）。
 */
export const attentionQkI8a8Params = (
  m: number,
  n: number,
  k: number,
  scale: number,
  window?: GemmRowWindowSpan,
): Uint32Array<ArrayBuffer> => {
  // MUST: 有限判定は f32 語で見る（attention.ts / state-attention.ts と同じ — f64 では有限でも
  // f32 へ落とすと +Inf になる値が 0·Inf = NaN で S を汚染する。公開 API 側 attrs 門との二重化）。
  if (!Number.isFinite(Math.fround(scale))) {
    throw new CodegenError(`attention_qk i8a8 params: scale は f32 で有限の数値（${scale}）`);
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
  if (window === undefined) {
    new Float32Array(params.buffer)[3] = scale;
    return params;
  }
  assertGemmRowWindow("attention_qk i8a8 行窓 params", m, window);
  const wide = new Uint32Array(8);
  wide.set(params.subarray(0, 3));
  new Float32Array(wide.buffer)[3] = scale;
  wide[4] = window.offset;
  wide[5] = window.rowsFull;
  return wide;
};

// ---------------------------------------------------------------------------
// ③PV（P̃ は非実体化のまま整数化 / V は per-column i8）
// ---------------------------------------------------------------------------

/** ③PV の i8a8 変種のキー（①QK と同じ規約 — 世代 ++ / dtype 欄 i8a8 / 幾何 / 内積変種）。 */
export const attentionPvI8a8Key = (
  v4: boolean,
  dp4a: boolean,
  score: ScoreStorage = "f32",
  geometry: I8a8Geometry = defaultI8a8Geometry("attention_pv"),
  rowWindow = false,
): string =>
  `attention_pv:v3:i8a8:${i8a8GeometryKeyPart(geometry, v4)}:${dp4a ? "dp4a" : "dp4aEmu"}${
    scoreKeyPart(score)
  }${rowWindow ? ":rwc" : ""}`;

/**
 * O を `vec4<f32>` で書けるか。**縮約側（N）の条件は適格判定（`N % 4 == 0`）が既に担っている**
 * ので、ここは出力の列 = D だけを見る（①QK / linear の i8a8 と同じ規律）。
 */
export const attentionPvI8a8UsesVec4 = (depth: number): boolean => depth % I8A8_PACK === 0;

/**
 * A タイル（P̃）の充填。**S を読んで `round(127·exp(S−m))` をその場で作り 4 詰めする**ので、
 * P は 1 バイトも実体化されない（ADR 0023 の性質をそのまま保つ）。
 *
 * MUST: 変換を掛けるのは**範囲内の要素だけ**。範囲外に `exp(0−m)·127` を掛けると 0 でない
 * 整数が内積へ混ざり、K 端数タイルだけが静かに誤る（gemm.ts の `fillA` と同じ MUST）。
 * MUST: `exp` は**成分ごとのスカラ式**（`vec4` へまとめない — 超越関数のベクトル版は実装依存）。
 * MUST: 行の最大 `m` は**その充填スロットが担当する行**のもの（`row_max{slot}`）。1 本で
 * 使い回すと、1 スレッドが 2 行以上を埋める幾何（`tileM > threads / kPacks`）で
 * 隣の行の最大値が混ざる — 例外の出ない誤値。
 * NOTE: clamp を置かないのは、`exp(S−m) ≤ 1` が `m = amax` から構造的だから（設計 §2.2）。
 * 「格子を 128 にする」退行の検出器は clamp ではなく **この 127 と dequant 側の 1/127** で、
 * どちらも参照との突合が拾う。
 */
const fillPTile = (geometry: I8a8Geometry, score: ScoreStorage): string => {
  const stride = i8a8FillStride(geometry);
  const lane = (slot: number, component: string): string =>
    `round(exp(raw${slot}.${component} - row_max${slot}) * ${P_LATTICE_MAX}.0)`;
  const slots = Array.from({ length: i8a8ASlots(geometry) }, (_, slot) =>
    `    var av${slot} = 0u;
    if (arow${slot} < dims.m && apack < k4) {
      let raw${slot} = ${scoreReadQuad("s", score, `arow_base${slot} + apack`)};
      // P̃ の scale は **1/127 固定**（行内 max が exp(0) = 1 で構造的 — amax は取らない）。
      // 除算が 1 つも無いので WGSL の 2.5 ULP 問題の外側にいる
      let p${slot} = vec4<f32>(
        ${["x", "y", "z", "w"].map((component) => lane(slot, component)).join(",\n        ")},
      );
      av${slot} = pack4xI8(vec4<i32>(p${slot}));
    }
    sa[sa_at${at(slot * stride)}] = av${slot};`).join("\n");
  return `    // 範囲外は 0 で埋める（qP = 0 は内積に寄与しないので K 端数でも結果は厳密）
    let apack = t * ${i8a8KPacks(geometry)}u + ap;
${slots}`;
};

/** 各充填スロットが担当する行の最大値（K タイルループ不変なので 1 度だけ引く）。 */
const prologuePvStats = (geometry: I8a8Geometry): string =>
  Array.from(
    { length: i8a8ASlots(geometry) },
    (_, slot) =>
      `  let stat_at${slot} = select(0u, (rbase + arow${slot}) * 2u, arow${slot} < dims.m);
  let row_max${slot} = stats[stat_at${slot}];`,
  ).join("\n");

/**
 * ③PV の書き出し。①QK の {@link store} と同じく行・列 quad を codegen 時に展開して
 * `acc{行}_{列 quad}` を静的に読む。ガードの構造・`prow·vs` の畳み順・`1/127` の乗算は
 * 幾何に依らず同一。
 */
const pvStore = (geometry: I8a8Geometry, v4: boolean): string => {
  const { regM, regN } = geometry;
  const quads = regN / I8A8_PACK;
  const rows = rowDecls(geometry);
  // MUST: prow は**出力行**の統計（`rbase + orow`）。A タイル充填で引いた `arow` の側を
  // 流用すると、担当が違うスレッドの行 inv が乗る（B·H = M = 1 でしか一致しない）。
  const prow = (row: number, indent: string): string =>
    `${indent}let prow = stats[(rbase + orow${row}) * 2u + 1u] * ${INV_P_LATTICE};`;
  if (v4) {
    const blocks = Array.from({ length: quads }, (_, quad) => {
      const rowStores = Array.from({ length: regM }, (_, row) =>
        `    if (orow${row} < dims.m) {
${prow(row, "      ")}
      o[obase + orow${row} * n4 + ocq${quad}] = vec4<f32>(acc${row}_${quad}) * (prow * vs);
    }`).join("\n");
      return `  if (ocq${quad} < n4) {
    // D % ${I8A8_PACK} == 0 かつ ocq${quad} < n4 なので oc + ${I8A8_PACK - 1} < D。
    // MUST: 列 scale は Vᵀ の**行** = (b, h, d)（= V の per-column scale）。行 scale の側と
    // 取り違えても例外は出ない
    let oc = ocq${quad} * ${I8A8_PACK}u;
    let vs = vec4<f32>(
      vscale[vsbase + oc],
      vscale[vsbase + oc + 1u],
      vscale[vsbase + oc + 2u],
      vscale[vsbase + oc + 3u],
    );
    // MUST: prow·vs を先に 1 つの f32 へ畳んでから f32(acc) に掛ける（bias が無いので fma は無い）
${rowStores}
  }`;
    }).join("\n");
    return `${quadDecls(geometry)}
${rows}
${blocks}`;
  }
  const lanes = ["x", "y", "z", "w"] as const;
  const rowStores = Array.from({ length: regM }, (_, row) => {
    const writes = Array.from({ length: regN }, (_, col) => {
      const offset = at(col);
      const acc = `acc${row}_${Math.floor(col / I8A8_PACK)}.${lanes[col % I8A8_PACK]}`;
      return `    if (ocol${offset} < dims.n) {
      o[orow_base + ocol${offset}] = f32(${acc}) * (prow * vscale[vsbase + ocol${offset}]);
    }`;
    }).join("\n");
    return `  if (orow${row} < dims.m) {
    let orow_base = obase + orow${row} * dims.n;
    // MUST: prow·vs を先に 1 つの f32 へ畳んでから f32(acc) に掛ける（bias が無いので fma は無い）
${prow(row, "    ")}
${writes}
  }`;
  }).join("\n");
  return `  let ocol = wid.x * ${i8a8TileN(geometry)}u + lid.x * ${regN}u;
${rows}
${rowStores}`;
};

export const attentionPvI8a8Wgsl = (
  v4: boolean,
  dp4a: boolean,
  score: ScoreStorage = "f32",
  geometry: I8a8Geometry = defaultI8a8Geometry("attention_pv"),
  rowWindow = false,
): string => {
  assertI8a8Geometry(geometry, "attention_pv i8a8");
  const kPacks = i8a8KPacks(geometry);
  // 行窓（{@link "./gemm.ts"} `GemmRowWindow` の `"c"`）: **O の base 1 本**だけが全 M
  // ストライド + 行オフセットで数える。S / 行統計 / Vᵀ はブロック相対のままなので、
  // P̃ の生成も dequant も 1 文字も動かない = 1 枚実行とビット同一。
  const oUnit = v4 ? "n4" : "dims.n";
  return `// karume attention_pv (O[b,m,d] = P[b,m,n] · v[b,n,d], P̃ = round(127·exp(S − m)) は非実体化${
    dp4a ? "" : "・エミュ"
  }, v は per-column i8（Vᵀ 連続）${scoreNote(score)}${rowWindow ? ", 行窓 c" : ""}, ${
    i8a8GeometryNote(geometry)
  }${v4 ? " + vec4" : ""})
struct Dims {
  m: u32,
  n: u32,
  k: u32,
${rowWindow ? ROW_WINDOW_DIMS_EXTRA : ""}}
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
// 語ストライド ${I8A8_PACK} になりバンク衝突が 8-way から 2-way に減る）
var<workgroup> sa: array<u32, ${kPacks * i8a8TileM(geometry)}>;
var<workgroup> sb: array<u32, ${kPacks * i8a8TileN(geometry)}>;

@compute @workgroup_size(${geometry.wgX}, ${geometry.wgY})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.y * ${geometry.wgX}u + lid.x;
  // 縮約軸 N は S 側では quad 単位・vq 側では pack 単位（同じ 4 要素の刻み）
  let k4 = dims.k / ${I8A8_PACK}u;
${
    v4
      ? `  let n4 = dims.n / ${I8A8_PACK}u;\n`
      : ""
  }  // B と H を畳んだバッチ軸（workgroup 単位で一様 — 内側の workgroupBarrier が WGSL の
  // 一様性要件を満たすための前提）。MUST: S は quad 単位、vq は pack 単位、行統計は行あたり
  // 2 語、vscale は D 本、O は要素${v4 ? "（v4 では quad）" : ""}単位で数える — 単位を
  // 取り違えると B·H ≥ 2 で隣の head を読み書きする（例外なしの誤値で B=H=1 では出ない）
  let sbase = wid.z * dims.m * k4;
  let vbase = wid.z * dims.n * k4;
  let rbase = wid.z * dims.m;
  let vsbase = wid.z * dims.n;
  let obase = wid.z * ${rowWindow ? "dims.rows_full" : "dims.m"} * ${oUnit}${
    rowWindow ? ` + dims.row_offset * ${oUnit}` : ""
  };
${prologueA(geometry, "sbase", "P̃ タイルの担当")}
  // 行の最大 m（K タイルループ不変なので 1 度だけ引く）。端タイルでは arow >= M がありうるので
  // 添字を 0 へ倒す（読んだ値は arow < M の枝でしか使われない）
${prologuePvStats(geometry)}
${
    prologueB(
      geometry,
      "vbase",
      "vrow_base",
      "列は D。vq は [D,N] の N 連続（= パック方向）なので**転置が要らない**",
    )
  }
  // ループ条件は uniform（dims は uniform バッファ）— 内側の workgroupBarrier が
  // WGSL の一様性要件を満たすために必要
  let tiles = (k4 + ${kPacks - 1}u) / ${kPacks}u;
${i8a8AccumulatorInit(geometry)}
  for (var t = 0u; t < tiles; t = t + 1u) {
${fillPTile(geometry, score)}
${fillB(geometry, "vq", "vrow_base", "v")}
    workgroupBarrier();
${i8a8InnerProductLoop(geometry)}
    workgroupBarrier();
  }
${pvStore(geometry, v4)}
}
`;
};

/**
 * uniform の Dims（`{m, n, k}` = `{M, D, N}` — f32 変種の {@link "./attention.ts"}
 * `attentionPvParams` と同じ並び。③PV に半スケールは登場しない）。
 *
 * `window` は **行窓変種だけ**が渡す 4〜5 語目（struct は 32 バイトへ伸びる）。
 * MUST: `m` はブロック行数（全 M ではない）。
 */
export const attentionPvI8a8Params = (
  m: number,
  n: number,
  k: number,
  window?: GemmRowWindowSpan,
): Uint32Array<ArrayBuffer> => {
  if (!Number.isSafeInteger(k) || k < 0 || k % 4 !== 0) {
    throw new CodegenError(`attention_pv i8a8 params: k（= N）は 4 の倍数の非負整数（${k}）`);
  }
  if (k > LINEAR_I8A8_MAX_K) {
    throw new CodegenError(
      `attention_pv i8a8 params: k=${k} が i32 縮約の門 ${LINEAR_I8A8_MAX_K} を超える`,
    );
  }
  const params = gemmParams("attention_pv", m, n, k);
  if (window === undefined) return params;
  assertGemmRowWindow("attention_pv i8a8 行窓 params", m, window);
  const wide = new Uint32Array(8);
  wide.set(params.subarray(0, 3));
  wide[3] = window.offset;
  wide[4] = window.rowsFull;
  return wide;
};
