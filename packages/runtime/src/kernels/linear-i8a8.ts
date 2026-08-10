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
 * かえって両方の可読性を落とす。**タイル幾何は f32 骨格から独立**（{@link
 * "./i8a8-geometry.ts"}）で、整数縮約が順序非依存だからこそ実測で選べる自由度になっている
 * — 幾何の算術と内積ループの正本は i8a8 側で 1 箇所（同ファイル）に保つ。
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
import { gemmParams, LINEAR_SCALE_BINDING } from "./gemm.ts";
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
export const linearI8a8UsesVec4 = (n: number): boolean => n % I8A8_PACK === 0;

/**
 * パイプラインキー。幾何がパラメタになったので**タイル辺だけでは生成物が決まらない**
 * （{@link i8a8GeometryKeyPart}）— 世代を `v4` へ上げ、幾何判別子を丸ごと載せる。
 */
export const linearI8a8Key = (
  v4: boolean,
  dp4a: boolean,
  geometry: I8a8Geometry = defaultI8a8Geometry("linear"),
): string => `linear:v4:i8a8:${i8a8GeometryKeyPart(geometry, v4)}:${dp4a ? "dp4a" : "dp4aEmu"}`;

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

/** 添字の加算オフセット（0 は省く — 生成物を読める形に保つ）。 */
const at = (offset: number): string => offset === 0 ? "" : ` + ${offset}u`;

/**
 * x タイルの担当（K タイルループ不変な行番号と行頭オフセット）。1 スレッドが
 * {@link i8a8ASlots} 本を充填ストライド刻みで受け持つ。
 */
const prologueA = (geometry: I8a8Geometry): string => {
  const tileM = i8a8TileM(geometry);
  const stride = i8a8FillStride(geometry);
  const slots = Array.from(
    { length: i8a8ASlots(geometry) },
    (_, slot) =>
      `  let arow${slot} = ${slot === 0 ? `wid.y * ${tileM}u + ar` : `arow0 + ${slot * stride}u`};
  let arow_base${slot} = arow${slot} * k4;`,
  ).join("\n");
  return `  // x タイルの担当（${tileM} 行 × ${i8a8KPacks(geometry)} pack を ${
    i8a8Threads(geometry)
  } スレッドで ${i8a8ASlots(geometry)} 巡）
  let ar = tid / ${i8a8KPacks(geometry)}u;
  let ap = tid % ${i8a8KPacks(geometry)}u;
  let sa_at = ap * ${tileM}u + ar;
${slots}`;
};

/** W タイルの担当。重みは `[n,k]` のまま読む（連続方向が k = パック方向なので転置が要らない）。 */
const prologueB = (geometry: I8a8Geometry): string => {
  const tileN = i8a8TileN(geometry);
  const stride = i8a8FillStride(geometry);
  const slots = Array.from(
    { length: i8a8BSlots(geometry) },
    (_, slot) =>
      `  let wcol${slot} = ${slot === 0 ? `wid.x * ${tileN}u + wc` : `wcol0 + ${slot * stride}u`};
  let wrow_base${slot} = wcol${slot} * k4;`,
  ).join("\n");
  return `  // W タイルの担当（${tileN} 出力チャネル × ${i8a8KPacks(geometry)} pack を ${
    i8a8Threads(geometry)
  } スレッドで ${i8a8BSlots(geometry)} 巡）。
  // 重みは [n,k] のまま読む（連続方向が k = パック方向なので**転置が要らない**）
  let wc = tid / ${i8a8KPacks(geometry)}u;
  let wp = tid % ${i8a8KPacks(geometry)}u;
  let sb_at = wp * ${tileN}u + wc;
${slots}`;
};

/** 共有タイルの充填（範囲外は 0 埋め — `idot(0, x) == 0` なので K 端数でも厳密）。 */
const fillTiles = (geometry: I8a8Geometry): string => {
  const kPacks = i8a8KPacks(geometry);
  const stride = i8a8FillStride(geometry);
  const a = Array.from({ length: i8a8ASlots(geometry) }, (_, slot) =>
    `    var av${slot} = 0u;
    if (arow${slot} < dims.m && apack < k4) {
      av${slot} = xq[arow_base${slot} + apack];
    }
    sa[sa_at${at(slot * stride)}] = av${slot};`).join("\n");
  const b = Array.from({ length: i8a8BSlots(geometry) }, (_, slot) =>
    `    var wv${slot} = 0u;
    if (wcol${slot} < dims.n && wpack < k4) {
      wv${slot} = w[wrow_base${slot} + wpack];
    }
    sb[sb_at${at(slot * stride)}] = wv${slot};`).join("\n");
  return `    // 範囲外は 0 で埋める（dot4I8Packed(0, x) == 0 なので K 端数でも結果は厳密）
    let apack = t * ${kPacks}u + ap;
${a}
    let wpack = t * ${kPacks}u + wp;
${b}`;
};

/**
 * 出力の書き出し。行・列 quad とも codegen 時に展開して `acc{行}_{列 quad}` を静的に読む
 * （`acc[i]` の動的添字を残さないための展開 — 理由は {@link i8a8AccumulatorInit}）。
 * ガードの構造・`fma` の単一丸め・`xs·wscale` の畳み順は幾何に依らず同一。
 */
const store = (geometry: I8a8Geometry, v4: boolean): string => {
  const { regM, regN } = geometry;
  const quads = regN / I8A8_PACK;
  const rows = Array.from(
    { length: regM },
    (_, row) =>
      row === 0
        ? `  let orow0 = wid.y * ${i8a8TileM(geometry)}u + lid.y * ${regM}u;`
        : `  let orow${row} = orow0 + ${row}u;`,
  ).join("\n");
  if (v4) {
    const quadDecls = Array.from(
      { length: quads },
      (_, quad) =>
        quad === 0
          ? `  let ocq0 = wid.x * ${i8a8TileN(geometry) / I8A8_PACK}u + lid.x * ${quads}u;`
          : `  let ocq${quad} = ocq0 + ${quad}u;`,
    ).join("\n");
    const blocks = Array.from({ length: quads }, (_, quad) => {
      const rowStores = Array.from({ length: regM }, (_, row) =>
        `    if (orow${row} < dims.m) {
      out[orow${row} * n4 + ocq${quad}] = fma(vec4<f32>(acc${row}_${quad}), xscale[orow${row}] * ws, biasv);
    }`).join("\n");
      return `  if (ocq${quad} < n4) {
    // n % ${I8A8_PACK} == 0 かつ ocq${quad} < n4 なので oc + ${
        I8A8_PACK - 1
      } < n（bias は常に f32 — ADR 0006）
    let oc = ocq${quad} * ${I8A8_PACK}u;
    let ws = vec4<f32>(wscale[oc], wscale[oc + 1u], wscale[oc + 2u], wscale[oc + 3u]);
    let biasv = vec4<f32>(bias[oc], bias[oc + 1u], bias[oc + 2u], bias[oc + 3u]);
    // MUST: xs·wscale を先に 1 つの f32 へ畳み、積和は fma（単一丸め）
${rowStores}
  }`;
    }).join("\n");
    return `  let n4 = dims.n / ${I8A8_PACK}u;
${quadDecls}
${rows}
${blocks}`;
  }
  const lanes = ["x", "y", "z", "w"] as const;
  const rowStores = Array.from({ length: regM }, (_, row) => {
    const writes = Array.from({ length: regN }, (_, col) => {
      const offset = at(col);
      const acc = `acc${row}_${Math.floor(col / I8A8_PACK)}.${lanes[col % I8A8_PACK]}`;
      return `    if (ocol${offset} < dims.n) {
      out[obase + ocol${offset}] = fma(f32(${acc}), xs * wscale[ocol${offset}], bias[ocol${offset}]);
    }`;
    }).join("\n");
    return `  if (orow${row} < dims.m) {
    let obase = orow${row} * dims.n;
    // MUST: xs·wscale を先に 1 つの f32 へ畳み、積和は fma（単一丸め）
    let xs = xscale[orow${row}];
${writes}
  }`;
  }).join("\n");
  return `  let ocol = wid.x * ${i8a8TileN(geometry)}u + lid.x * ${regN}u;
${rows}
${rowStores}`;
};

export const linearI8a8Wgsl = (
  v4: boolean,
  dp4a: boolean,
  geometry: I8a8Geometry = defaultI8a8Geometry("linear"),
): string => {
  assertI8a8Geometry(geometry, "linear i8a8");
  const kPacks = i8a8KPacks(geometry);
  return `// karume linear (x[m,k] · wᵀ[k,n] + bias[n], 活性 per-token i8 × 重み i8 の整数内積${
    dp4a ? "" : "・エミュ"
  }, ${i8a8GeometryNote(geometry)}${v4 ? " + vec4" : ""})
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
// 内側の読みが語ストライド ${I8A8_PACK} になりバンク衝突が 8-way から 2-way に減る）
var<workgroup> sa: array<u32, ${kPacks * i8a8TileM(geometry)}>;
var<workgroup> sb: array<u32, ${kPacks * i8a8TileN(geometry)}>;

@compute @workgroup_size(${geometry.wgX}, ${geometry.wgY})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.y * ${geometry.wgX}u + lid.x;
  // 適格条件で k % 4 == 0 なので、行頭は必ず語境界に来る（i8 ペイロードは平坦添字 4 詰め）
  let k4 = dims.k / ${I8A8_PACK}u;
${prologueA(geometry)}
${prologueB(geometry)}
  // ループ条件は uniform（dims は uniform バッファ）— 内側の workgroupBarrier が
  // WGSL の一様性要件を満たすために必要
  let tiles = (k4 + ${kPacks - 1}u) / ${kPacks}u;
${i8a8AccumulatorInit(geometry)}
  for (var t = 0u; t < tiles; t = t + 1u) {
${fillTiles(geometry)}
    workgroupBarrier();
${i8a8InnerProductLoop(geometry)}
    workgroupBarrier();
  }
${store(geometry, v4)}
}
`;
};

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
