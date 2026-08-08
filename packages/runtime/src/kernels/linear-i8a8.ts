/**
 * linear の **w8a8 変種**（活性 per-token i8 × 重み per-channel i8 の**整数内積**）—
 * 設計は docs/research/2026-08-03-dp4a-w8a8-design.md §4.3、opt-in は
 * `SessionOptions.linearCompute: "i8a8"`（既定 `"f32"`）。
 *
 * ```
 * acc[i32] = Σ_k xq[m,k] · wq[n,k]                    （厳密な整数・丸め 0 回）
 * out      = f32(acc) · (xs[row] · wscale[col]) + bias[col]
 * ```
 *
 * ## 骨格を gemm.ts と共有しない理由（ADR 0022 の MUST との関係）
 *
 * ADR 0022 の「内積ループの正本 1 箇所」は **f32 の縮約順序を全 op で揃える**ための MUST で、
 * 本カーネルの縮約は `i32` の厳密加算（順序非依存）なので、その MUST が守ろうとしている
 * 性質は最初から成立している。共有 A/B タイルの要素型（`u32` の 4 詰め）・充填の割り当て・
 * 内積の形（`dot4I8Packed` の 4 要素まとめ）が f32 骨格と別物になるため、断片差し込みでは
 * かえって両方の可読性を落とす。**タイル幾何（64×64 出力 / 16×16 スレッド / 1 スレッド 4×4 /
 * K タイル 16）と `store` のガード構造は gemm.ts と揃える**（{@link GEMM_TILE} /
 * {@link GEMM_WORKGROUP} を輸入して定数の正本を 1 箇所に保つ）。
 *
 * ## 数値契約（本経路の核）
 *
 * - **整数縮約は順序非依存の厳密値**なので、`(Σ xq·wq)` はどんな実装でも同じ 1 つの整数に
 *   なる。ADR 0019 が `(Σ x·q)·s` 形を却下した理由（GPU の dequant と CPU 展開のビット
 *   不一致）は w8a8 では消え、**GPU と TS 参照が atol=0 で一致する**（src/reference/i8a8.ts）。
 * - MUST: dequant の乗算順序を固定する — **`xs[row] · wscale[col]` を先に 1 つの f32 へ
 *   畳んでから** `f32(acc)` に掛け、最後に bias を足す。`f32(acc)·xs·wscale` の逐次形は
 *   丸めの位置が動いて参照とのビット一致が崩れる。
 * - MUST: 最後の積和は **`fma`**（積を丸めずに 1 度だけ丸める）。素の `a * b + c` は
 *   ドライバが融合する**かもしれない**形で、数値契約が実装依存になる（本機 RTX 3080 Ti では
 *   実測で融合された — 素の式・`let` 経由・`bitcast` 固定の 3 形とも同じ融合値）。`fma` で
 *   書けば単一丸めが**ソースの性質**になり、TS 参照（倍精度で 1 度だけ丸める）と対になる。
 * - **K 端数は 0 埋めで厳密**（`dot4I8Packed(0, x) == 0`）。ADR 0022 の検出限界①（A/B 片側
 *   だけの 0 埋め退行が相殺で見えない）はそのまま引き継ぐ。
 *
 * ## 整数内積の 2 変種（数値完全一致）
 *
 * `dot4I8Packed`（WGSL 言語拡張 `packed_4x8_integer_dot_product`）と
 * `dot(unpack4xI8(a), unpack4xI8(b))`（core WGSL）は**同じ整数を返す** — 整数加算は結合的
 * かつ厳密で、中間値も最大 4·127² = 64,516 なので巻き戻りが起こりえない。したがって拡張の
 * 有無は**速度にしか効かず**、fail loudly も tolerance 分岐も要らない。選択は
 * `navigator.gpu.wgslLanguageFeatures` の列挙（{@link dp4aAvailable}）で、キーに `:dp4a` /
 * `:dp4aEmu` が載るので診断（ADR 0021）でどちらが走ったか見える。唯一の機械的検出器は
 * 「拡張のある機で両変種を実走して atol=0 で突合する」テスト（tests/gpu_i8a8_test.ts と、
 * 融合 attention の QK 側は tests/gpu_attention_i8a8_test.ts）。
 */

import { CodegenError } from "../codegen/errors.ts";
import {
  GEMM_TILE,
  GEMM_WORKGROUP,
  gemmKeyPart,
  gemmParams,
  LINEAR_SCALE_BINDING,
} from "./gemm.ts";

export { LINEAR_SCALE_BINDING } from "./gemm.ts";

/**
 * 活性 scale（per-token）の束縛。**重み格納の概念ではない**ので weight-storage.ts ではなく
 * ここに置く（linear の現行 0..5 の後ろ）。
 */
export const LINEAR_ACT_SCALE_BINDING = LINEAR_SCALE_BINDING + 1;

/**
 * i32 縮約のオーバフロー門。`|acc| ≤ k · 127² = k · 16129` なので i32 の上限
 * 2,147,483,647 からは `k ≤ 133,142`。安全側で 2^17 を門にする（DiT の最大 k = 8,192 に
 * 対し余裕 16 倍）。超える k は fail loudly — 黙って通すと i32 の巻き戻りで符号ごと化ける。
 */
export const LINEAR_I8A8_MAX_K = 131072;

/** 整数内積に使う WGSL 言語拡張の名前（device feature ではない）。 */
export const DP4A_WGSL_FEATURE = "packed_4x8_integer_dot_product";

/**
 * `dot4I8Packed` 変種を使えるか。
 *
 * MUST: これは**数値が同一な変種の選択**にだけ使う（src/gpu/device.ts の docstring と対）。
 * 列挙はコンパイルが通ることの保証であってハードウェア対応の保証ではないため、機能検出には
 * 使えない — 本経路では誤った選択でも結果が 1 ビットも変わらないので成立する。
 */
export const dp4aAvailable = (wgslLanguageFeatures: ReadonlySet<string>): boolean =>
  wgslLanguageFeatures.has(DP4A_WGSL_FEATURE);

/**
 * 出力を `vec4<f32>` で書けるか。**k 側の条件は適格判定（`k % 4 == 0`）が既に担っている**ので
 * ここは n だけを見る（f32 骨格の {@link gemmUsesVec4} が k と n の両方を見るのと対照的 —
 * i8a8 では k 条件が必須で、n 条件はスカラ書き出し変種へ落ちるだけ）。
 */
export const linearI8a8UsesVec4 = (n: number): boolean => n % 4 === 0;

/** 1 スレッドが持つ出力の一辺。 */
const REG = 4;
/** workgroup の一辺（16×16 = 256 スレッド）。 */
const WG = GEMM_WORKGROUP;
/** K タイル幅（16 要素 = 4 パック）。 */
const TILE_K = 16;
/** K タイルあたりのパック数。A / W 充填のスレッド割当に使う。 */
const K_PACKS = TILE_K / REG;

export const linearI8a8Key = (v4: boolean, dp4a: boolean): string =>
  `linear:v3:i8a8:${gemmKeyPart(v4)}:${dp4a ? "dp4a" : "dp4aEmu"}`;

/**
 * 整数内積の 2 変種。**返す整数は完全に同じ**で、違いは速度だけ（上の docstring）。
 *
 * MUST: i8a8 カーネルは全てこの 1 本を使う（融合 attention の QK も — src/kernels/
 * attention-i8a8.ts）。「エミュは数値同一」という主張は `idot` の実体が 1 箇所にあることに
 * 掛かっていて、書き写すとカーネルごとに別々にずれうる形になる。
 */
export const i8a8IdotWgsl = (dp4a: boolean): string =>
  dp4a
    ? `// 整数内積（WGSL 言語拡張 packed_4x8_integer_dot_product）。エミュ変種と**同じ整数**を返す
fn idot(a: u32, b: u32) -> i32 {
  return dot4I8Packed(a, b);
}`
    : `// 整数内積のエミュレーション（core WGSL のみ）。dot4I8Packed と**同じ整数**を返す —
// 整数加算は結合的かつ厳密で、中間値も最大 4·127² = 64,516 なので巻き戻らない
fn idot(a: u32, b: u32) -> i32 {
  return dot(unpack4xI8(a), unpack4xI8(b));
}`;

/**
 * 出力の書き出し。行ループは codegen 時に展開して {@link accumulatorInit} の `acc0..acc3` を
 * 静的に読む（`acc[i]` の動的添字を残さないための展開 — 理由は同関数の docstring）。
 * ガードの構造・`fma` の単一丸め・`xs·wscale` の畳み順は展開前と同一。
 */
const store = (v4: boolean): string => {
  const rows = `  let orow0 = wid.y * ${GEMM_TILE}u + lid.y * ${REG}u;`;
  // 2 行目以降の行番号は展開時に確定するので、その場で let に落とす（0 行目は上の rows）
  const rowDecl = (i: number, indent: string): string =>
    i === 0 ? "" : `${indent}let orow${i} = orow0 + ${i}u;\n`;
  if (v4) {
    const rowStores = Array.from(
      { length: REG },
      (_, i) =>
        `${rowDecl(i, "    ")}    if (orow${i} < dims.m) {
      out[orow${i} * n4 + ocq] = fma(vec4<f32>(acc${i}), xscale[orow${i}] * ws, biasv);
    }`,
    ).join("\n");
    return `  let n4 = dims.n / ${REG}u;
  let ocq = wid.x * ${WG}u + lid.x;
${rows}
  if (ocq < n4) {
    // n % ${REG} == 0 かつ ocq < n4 なので oc + ${REG - 1} < n（bias は常に f32 — ADR 0006）
    let oc = ocq * ${REG}u;
    let ws = vec4<f32>(wscale[oc], wscale[oc + 1u], wscale[oc + 2u], wscale[oc + 3u]);
    let biasv = vec4<f32>(bias[oc], bias[oc + 1u], bias[oc + 2u], bias[oc + 3u]);
    // MUST: xs·wscale を先に 1 つの f32 へ畳み、積和は fma（単一丸め）
${rowStores}
  }`;
  }
  const write = (i: number, component: string, offset: string): string =>
    `out[obase + ocol${offset}] = fma(f32(acc${i}.${component}), xs * wscale[ocol${offset}], bias[ocol${offset}]);`;
  const rowStores = Array.from(
    { length: REG },
    (_, i) =>
      `${rowDecl(i, "  ")}  if (orow${i} < dims.m) {
    let obase = orow${i} * dims.n;
    // MUST: xs·wscale を先に 1 つの f32 へ畳み、積和は fma（単一丸め）
    let xs = xscale[orow${i}];
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

/**
 * i32 accumulator の初期化。**配列 1 本ではなく `acc0..acc3` の名前付き変数**にするのは、
 * `acc[i]` の動的添字がアドレス可能な関数ローカル領域を要求し、レジスタに載らずローカル
 * メモリへ落ちるため（Metal で顕著）。展開しても縮約は i32 の厳密加算のまま・行ごとに
 * 独立なので、**返る整数は 1 ビットも変わらない**（実測でも M2 / RTX 3080 Ti の代表 6 形状が
 * 全て bit 一致し、node 重み付きのカーネル時間は M2 で 1.175 倍・RTX で 1.429 倍）。
 */
const accumulatorInit = (): string =>
  Array.from({ length: REG }, (_, i) => `  var acc${i} = vec4<i32>(0);`).join("\n");

/**
 * K パック内側の 4 行更新。{@link accumulatorInit} と対で展開する。
 * 1 出力あたりの加算順序（kk 昇順・パック内 4 語まとめ）は展開前と同一。
 */
const accumulatorUpdate = (): string =>
  Array.from(
    { length: REG },
    (_, i) =>
      `      let a${i} = sa[p * ${GEMM_TILE}u + lid.y * ${REG}u + ${i}u];
      acc${i} = acc${i} + vec4<i32>(idot(a${i}, b0), idot(a${i}, b1), idot(a${i}, b2), idot(a${i}, b3));`,
  )
    .join("\n");

export const linearI8a8Wgsl = (v4: boolean, dp4a: boolean): string =>
  `// karume linear (x[m,k] · wᵀ[k,n] + bias[n], 活性 per-token i8 × 重み i8 の整数内積${
    dp4a ? "" : "・エミュ"
  }, レジスタ ${GEMM_TILE}x${GEMM_TILE} タイル${v4 ? " + vec4" : ""})
struct Dims {
  m: u32,
  n: u32,
  k: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> xq: array<u32>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<${v4 ? "vec4<f32>" : "f32"}>;
@group(0) @binding(${LINEAR_SCALE_BINDING}) var<storage, read> wscale: array<f32>;
@group(0) @binding(${LINEAR_ACT_SCALE_BINDING}) var<storage, read> xscale: array<f32>;

${i8a8IdotWgsl(dp4a)}

// 共有タイルは **[pack][行] / [pack][列]** 配置（プロトタイプの [行][pack] からの組み替え —
// 内側の読みが語ストライド ${REG} になりバンク衝突が 8-way から 2-way に減る）
var<workgroup> sa: array<u32, ${K_PACKS * GEMM_TILE}>;
var<workgroup> sb: array<u32, ${K_PACKS * GEMM_TILE}>;

@compute @workgroup_size(${WG}, ${WG})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.y * ${WG}u + lid.x;
  // 適格条件で k % 4 == 0 なので、行頭は必ず語境界に来る（i8 ペイロードは平坦添字 4 詰め）
  let k4 = dims.k / ${REG}u;
  // x タイルの担当（${GEMM_TILE} 行 × ${K_PACKS} pack = ${GEMM_TILE * K_PACKS} スレッド）
  let ar = tid / ${K_PACKS}u;
  let ap = tid % ${K_PACKS}u;
  let arow = wid.y * ${GEMM_TILE}u + ar;
  let arow_base = arow * k4;
  // W タイルの担当（${GEMM_TILE} 出力チャネル × ${K_PACKS} pack = ${
    GEMM_TILE * K_PACKS
  } スレッド）。
  // 重みは [n,k] のまま読む（連続方向が k = パック方向なので**転置が要らない**）
  let wc = tid / ${K_PACKS}u;
  let wp = tid % ${K_PACKS}u;
  let wcol = wid.x * ${GEMM_TILE}u + wc;
  let wrow_base = wcol * k4;
  // ループ条件は uniform（dims は uniform バッファ）— 内側の workgroupBarrier が
  // WGSL の一様性要件を満たすために必要
  let tiles = (k4 + ${K_PACKS - 1}u) / ${K_PACKS}u;
${accumulatorInit()}
  for (var t = 0u; t < tiles; t = t + 1u) {
    // 範囲外は 0 で埋める（dot4I8Packed(0, x) == 0 なので K 端数でも結果は厳密）
    let apack = t * ${K_PACKS}u + ap;
    var av = 0u;
    if (arow < dims.m && apack < k4) {
      av = xq[arow_base + apack];
    }
    sa[ap * ${GEMM_TILE}u + ar] = av;
    let wpack = t * ${K_PACKS}u + wp;
    var wv = 0u;
    if (wcol < dims.n && wpack < k4) {
      wv = w[wrow_base + wpack];
    }
    sb[wp * ${GEMM_TILE}u + wc] = wv;
    workgroupBarrier();
    // 共有ロード 5 回（W の 4 語 + x の 1 語）で ${REG * REG} 個の整数内積 = ${
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
${store(v4)}
}
`;

/** uniform の Dims（`{m,n,k}` — f32 骨格と同じ 3 語 + 16 バイト整列）。 */
export const linearI8a8Params = (m: number, n: number, k: number): Uint32Array<ArrayBuffer> => {
  if (!Number.isSafeInteger(k) || k < 0 || k % 4 !== 0) {
    throw new CodegenError(`linear i8a8 params: k は 4 の倍数の非負整数（${k}）`);
  }
  if (k > LINEAR_I8A8_MAX_K) {
    throw new CodegenError(
      `linear i8a8 params: k=${k} が i32 縮約の門 ${LINEAR_I8A8_MAX_K} を超える`,
    );
  }
  return gemmParams("linear", m, n, k);
};
