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
 *
 * ## w4a8 変種（`weight: "i4"` — i4 常駐の重み × per-token i8 活性）
 *
 * i4 常駐は取得量と VRAM を取る代わりに、i8a8 の述語（i8 常駐）から外れて f32 計算経路へ
 * 流れ、dp4a の利得だけを失っていた（docs/research/2026-08-21-anima-i4-seat-speed.md）。
 * 本変種は **i4 のまま整数内積へ載せる**ための第 2 の数値契約を持つ:
 *
 * ```
 * accf = 0
 * for gi in 0..k/g-1:
 *     acci = Σ_{i∈group gi} xq[row,i] · wq[col,i]   // i32 厳密（|acci| ≤ g·127·8）
 *     accf = fma(f32(acci), wscale[col, gi], accf)  // group 境界でだけ flush
 * out = fma(accf, xs[row], bias[col])
 * ```
 *
 * - 丸めは厳密に **`k/g + 1` 回**で、flush は **group 境界ちょうど**（K タイル境界ではない）。
 *   これで「タイル幾何は数値契約の外」という i8a8 の自由度が w4a8 でもそのまま保たれる —
 *   タイルを何枚に割ろうが group の中は i32 の厳密加算で、flush の位置は g だけが決める。
 * - **MUST: `xs · wscale` を先に 1 つの f32 へ畳む i8 変種の MUST は w4a8 では成立しない。**
 *   wscale が group ごとに変わるので、畳もうとすると group ごとに `xs·wscale` を作ることに
 *   なり、丸めが `k/g` 回ぶん増えたうえ「xs は行の量」という構造も壊れる。xs は**最後の
 *   fma 1 回**へ回す（この非対称が w4a8 と i8a8 の唯一の数値契約の差）。
 * - group の中は i32 厳密なので、i8 変種と同じく **GPU と TS 参照が atol=0 で一致する**
 *   （{@link "../reference/i8a8.ts"} の `referenceLinearW4a8`）。
 * - K タイルが group 境界を跨ぐと 2 group の重みが 1 つの i32 に混ざるので、
 *   `groupSize % tileK == 0` を生成時に落とす（{@link "./i8a8-geometry.ts"} の
 *   `assertI8a8GroupGeometry`）。k % g == 0 と併せて **K 端数が原理的に出ない**。
 *
 * NOTE: w4a8 の K ループは「group 外側 × タイル内側」の 2 段なので、共有する断片
 * （{@link i8a8InnerProductLoop} / A 側充填）は 1 段ぶん浅い字下げのまま埋め込まれる。
 * 生成物の見た目より**断片の正本が 1 箇所であること**を優先した（幾何の添字算術を書き写すと
 * 片方だけ pack ストライドを取り違える形になる）。
 */

import { CodegenError } from "../codegen/errors.ts";
import { gemmParams, LINEAR_SCALE_BINDING } from "./gemm.ts";
import {
  i4GroupKeyPart,
  i4GroupShift,
  i4IntegerLanesWgsl,
  weightKeyPart,
  type WeightStorage,
} from "./weight-storage.ts";
import {
  assertI8a8Geometry,
  assertI8a8GroupGeometry,
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

/**
 * **w4a8 の** i32 縮約の門（group 長側）。i4 レーンは `|q| ≤ 8` なので 1 group ぶんの
 * `|acci| ≤ g · 127 · 8 = g · 1016` で、i32 の上限 2,147,483,647 からは `g ≤ 2,113,665`。
 * 門はそこから 1 つ内側の 2,113,664 を採る（安全側 — 等号ちょうどの形を通さない）。
 *
 * MUST: w4a8 に {@link LINEAR_I8A8_MAX_K} を適用しない。flush が group ごとなので i32 に
 * 載るのは 1 group ぶんだけで、k はいくら長くても巻き戻らない（k 側の量は f32 accumulator の
 * 丸め回数 `k/g + 1` という別の契約に移る）。k の門で代用すると、通るはずの長い k を
 * 落としつつ「g が異様に大きい」本当の危険を素通しする。
 */
export const LINEAR_W4A8_MAX_GROUP = 2113664;

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
 *
 * MUST: w4a8 は末尾に格納判別子 + group 長（`:wi4g32`）を足す。i8 変種は既定引数で
 * 従来どおり空になる（既存キーはバイト不変 — ブラウザの暗黙シェーダキャッシュを取り直さない）。
 * group 長を載せ忘れると **group 32 のパイプラインが group 64 の資産で走る**沈黙誤値になる
 * （g は WGSL に shift として焼かれる — ADR 0069）。
 */
export const linearI8a8Key = (
  v4: boolean,
  dp4a: boolean,
  geometry: I8a8Geometry = defaultI8a8Geometry("linear"),
  weight: WeightStorage = "i8",
  groupSize?: number,
): string =>
  `linear:v4:i8a8:${i8a8GeometryKeyPart(geometry, v4)}:${dp4a ? "dp4a" : "dp4aEmu"}` +
  (weight === "i8" ? "" : `${weightKeyPart(weight)}${i4GroupKeyPart(groupSize)}`);

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

/**
 * W タイルの担当。重みは `[n,k]` のまま読む（連続方向が k = パック方向なので転置が要らない）。
 *
 * `rowStride` は行頭の刻み。i8 は 4 詰めの**語数** `k4`、i4 は 1 語 8 要素で pack が語の
 * 半分になるため**平坦要素数** `dims.k`（語割りは {@link i4IntegerLanesWgsl} が持つ）。
 * MUST: 既定は `"k4"` — i8 変種の生成物を 1 バイトも動かさない。
 */
const prologueB = (geometry: I8a8Geometry, rowStride = "k4"): string => {
  const tileN = i8a8TileN(geometry);
  const stride = i8a8FillStride(geometry);
  const slots = Array.from(
    { length: i8a8BSlots(geometry) },
    (_, slot) =>
      `  let wcol${slot} = ${slot === 0 ? `wid.x * ${tileN}u + wc` : `wcol0 + ${slot * stride}u`};
  let wrow_base${slot} = wcol${slot} * ${rowStride};`,
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

/**
 * A（活性）側の共有タイル充填（範囲外は 0 埋め — `idot(0, x) == 0` なので K 端数でも厳密）。
 *
 * MUST: i8 / w4a8 の両変種がこの 1 本を使う。`t` は**通し番号の K タイル**なので、
 * w4a8 の 2 段ループ（group 外側 × タイル内側）でも `t` を束ねさえすれば式が同一になる。
 */
const fillTilesA = (geometry: I8a8Geometry): string => {
  const kPacks = i8a8KPacks(geometry);
  const stride = i8a8FillStride(geometry);
  const a = Array.from({ length: i8a8ASlots(geometry) }, (_, slot) =>
    `    var av${slot} = 0u;
    if (arow${slot} < dims.m && apack < k4) {
      av${slot} = xq[arow_base${slot} + apack];
    }
    sa[sa_at${at(slot * stride)}] = av${slot};`).join("\n");
  return `    // 範囲外は 0 で埋める（dot4I8Packed(0, x) == 0 なので K 端数でも結果は厳密）
    let apack = t * ${kPacks}u + ap;
${a}`;
};

/** B（重み i8）側の共有タイル充填。 */
const fillTilesB = (geometry: I8a8Geometry): string => {
  const kPacks = i8a8KPacks(geometry);
  const stride = i8a8FillStride(geometry);
  const b = Array.from({ length: i8a8BSlots(geometry) }, (_, slot) =>
    `    var wv${slot} = 0u;
    if (wcol${slot} < dims.n && wpack < k4) {
      wv${slot} = w[wrow_base${slot} + wpack];
    }
    sb[sb_at${at(slot * stride)}] = wv${slot};`).join("\n");
  return `    let wpack = t * ${kPacks}u + wp;
${b}`;
};

/** 共有タイルの充填（範囲外は 0 埋め — `idot(0, x) == 0` なので K 端数でも厳密）。 */
const fillTiles = (geometry: I8a8Geometry): string =>
  `${fillTilesA(geometry)}\n${fillTilesB(geometry)}`;

/**
 * B（重み i4）側の共有タイル充填。1 pack = 4 要素は語（8 要素）の半分なので、添字は
 * **平坦要素**で作り {@link i4IntegerLanesWgsl} が語割りと nibble 抽出を持つ。
 *
 * K 側のガードが無いのは、`k % g == 0`（配布形の不変条件）と `g % tileK == 0`
 * （{@link assertI8a8GroupGeometry}）から **K 端数が原理的に出ない**ため。A 側の
 * `apack < k4` は共有断片ゆえに残るが、w4a8 では常に真になる。
 */
const fillTilesBI4 = (geometry: I8a8Geometry): string => {
  const stride = i8a8FillStride(geometry);
  const b = Array.from({ length: i8a8BSlots(geometry) }, (_, slot) =>
    `    var wv${slot} = 0u;
    if (wcol${slot} < dims.n) {
      wv${slot} = i4lanes(wrow_base${slot} + welem);
    }
    sb[sb_at${at(slot * stride)}] = wv${slot};`).join("\n");
  return `    // i4 は 1 語 8 要素なので pack の添字は平坦要素（k % g == 0 ∧ g % tileK == 0 で K 端数なし）
    let welem = t * ${geometry.tileK}u + wp * ${I8A8_PACK}u;
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

/** 担当列（`ocol` 起点の相対 `col`）の WGSL 式。 */
const columnAt = (col: number): string => col === 0 ? "ocol" : `ocol + ${col}u`;

/**
 * w4a8 の**ループ前**の束縛 — 出力の担当行 / 列と、列ごとの group scale 行頭。
 *
 * i8 変種はこれらを store でだけ使うので末尾に置けるが、w4a8 は **group 境界の flush が
 * 列ごとの scale を引く**のでループの外へ巻き上げる（行内 group 数 `groups` はループ不変）。
 */
const prologueW4a8 = (geometry: I8a8Geometry, shift: number): string => {
  const { regM, regN } = geometry;
  const rows = Array.from(
    { length: regM },
    (_, row) =>
      row === 0
        ? `  let orow0 = wid.y * ${i8a8TileM(geometry)}u + lid.y * ${regM}u;`
        : `  let orow${row} = orow0 + ${row}u;`,
  ).join("\n");
  const bases = Array.from(
    { length: regN },
    (_, col) => `  let wsb${col} = ${col === 0 ? "ocol" : `(${columnAt(col)})`} * groups;`,
  ).join("\n");
  const accf = Array.from({ length: regM }, (_, row) =>
    Array.from(
      { length: regN / I8A8_PACK },
      (_, quad) => `  var accf${row}_${quad} = vec4<f32>(0.0);`,
    ).join("\n")).join("\n");
  return `  // 出力の担当は**ループの外**（group 境界の flush が列ごとの scale を引く）
  let ocol = wid.x * ${i8a8TileN(geometry)}u + lid.x * ${regN}u;
${rows}
  // group scale は [n, k/g] の平坦。行内 group 数と列ごとの行頭はループ不変なので巻き上げる
  let groups = dims.k >> ${shift}u;
${bases}
  // f32 accumulator（group 境界でだけ書かれる — 丸めは k/g 回）
${accf}`;
};

/**
 * group 境界の flush。**i32 accumulator を f32 へ 1 回だけ畳む**唯一の箇所で、
 * 丸めの位置（= 数値契約）そのもの。
 *
 * MUST: 積和は `fma`（単一丸め）。MUST: ここで `xs` を掛けない — wscale が group ごとに
 * 変わるので i8 変種の「`xs·wscale` を先に畳む」形は成立せず、掛けると丸めが `k/g` 回
 * ぶん増える（ファイル冒頭の w4a8 節）。
 */
const flushW4a8 = (geometry: I8a8Geometry): string => {
  const { regM, regN } = geometry;
  const quads = regN / I8A8_PACK;
  const blocks = Array.from({ length: quads }, (_, quad) => {
    const lanes = Array.from(
      { length: I8A8_PACK },
      (_, lane) => `wscale[wsb${quad * I8A8_PACK + lane} + gi]`,
    ).join(", ");
    const rows = Array.from(
      { length: regM },
      (_, row) =>
        `    accf${row}_${quad} = fma(vec4<f32>(acc${row}_${quad}), ws${quad}, accf${row}_${quad});`,
    ).join("\n");
    return `    let ws${quad} = vec4<f32>(${lanes});
${rows}`;
  }).join("\n");
  return `    // group 境界ちょうどで f32 へ flush（MUST: fma の単一丸め・xs は最後まで掛けない）
${blocks}`;
};

/**
 * w4a8 の K ループ（**group 外側 × タイル内側**の 2 段）。
 *
 * `t` を通し番号の K タイルとして束ね直すので、A 側充填と内積ループは i8 変種と**同一の
 * 断片**がそのまま入る。内側の上限は定数 `g / tileK` で、`groups` だけが uniform 実行時値
 * （内側の `workgroupBarrier` の一様性要件は 2 段とも満たされる）。
 */
const loopW4a8 = (geometry: I8a8Geometry, groupSize: number): string => {
  const tilesPerGroup = groupSize / geometry.tileK;
  const accInit = Array.from({ length: geometry.regM }, (_, row) =>
    Array.from(
      { length: geometry.regN / I8A8_PACK },
      (_, quad) => `    var acc${row}_${quad} = vec4<i32>(0);`,
    ).join("\n")).join("\n");
  return `  // ループ条件は uniform（dims は uniform バッファ）— 内側の workgroupBarrier が
  // WGSL の一様性要件を満たすために必要
  for (var gi = 0u; gi < groups; gi = gi + 1u) {
    // i32 accumulator は group ごとに初期化（|acci| ≤ g·127·8 — 門は LINEAR_W4A8_MAX_GROUP）
${accInit}
    // 1 group = K タイルちょうど ${tilesPerGroup} 枚（g % tileK == 0 の門）。t は通し番号
    let gbase = gi * ${tilesPerGroup}u;
    for (var gt = 0u; gt < ${tilesPerGroup}u; gt = gt + 1u) {
      let t = gbase + gt;
${fillTilesA(geometry)}
${fillTilesBI4(geometry)}
      workgroupBarrier();
${i8a8InnerProductLoop(geometry)}
      workgroupBarrier();
    }
${flushW4a8(geometry)}
  }`;
};

/**
 * w4a8 の書き出し。`out = fma(accf, xs[row], bias[col])` で、**xs はここで初めて掛かる**
 * （i8 変種の `xs·wscale` 畳みとの非対称 — ファイル冒頭の w4a8 節）。行・列 quad の展開と
 * ガードの構造は i8 変種と同じ。
 */
const storeW4a8 = (geometry: I8a8Geometry, v4: boolean): string => {
  const { regM, regN } = geometry;
  const quads = regN / I8A8_PACK;
  if (v4) {
    const quadDecls = Array.from(
      { length: quads },
      (_, quad) =>
        quad === 0 ? `  let ocq0 = ocol / ${I8A8_PACK}u;` : `  let ocq${quad} = ocq0 + ${quad}u;`,
    ).join("\n");
    const blocks = Array.from({ length: quads }, (_, quad) => {
      const rowStores = Array.from({ length: regM }, (_, row) =>
        `    if (orow${row} < dims.m) {
      out[orow${row} * n4 + ocq${quad}] = fma(accf${row}_${quad}, vec4<f32>(xscale[orow${row}]), biasv);
    }`).join("\n");
      return `  if (ocq${quad} < n4) {
    // n % ${I8A8_PACK} == 0 かつ ocq${quad} < n4 なので oc + ${
        I8A8_PACK - 1
      } < n（bias は常に f32 — ADR 0006）
    let oc = ocq${quad} * ${I8A8_PACK}u;
    let biasv = vec4<f32>(bias[oc], bias[oc + 1u], bias[oc + 2u], bias[oc + 3u]);
    // MUST: xs は最後の fma（単一丸め）へ回す — wscale は group 境界で畳み済み
${rowStores}
  }`;
    }).join("\n");
    return `  let n4 = dims.n / ${I8A8_PACK}u;
${quadDecls}
${blocks}`;
  }
  const lanes = ["x", "y", "z", "w"] as const;
  const rowStores = Array.from({ length: regM }, (_, row) => {
    const writes = Array.from({ length: regN }, (_, col) => {
      const column = columnAt(col);
      const acc = `accf${row}_${Math.floor(col / I8A8_PACK)}.${lanes[col % I8A8_PACK]}`;
      return `    if (${column} < dims.n) {
      out[obase + ${column}] = fma(${acc}, xs, bias[${column}]);
    }`;
    }).join("\n");
    return `  if (orow${row} < dims.m) {
    let obase = orow${row} * dims.n;
    // MUST: xs は最後の fma（単一丸め）へ回す — wscale は group 境界で畳み済み
    let xs = xscale[orow${row}];
${writes}
  }`;
  }).join("\n");
  return rowStores;
};

export const linearI8a8Wgsl = (
  v4: boolean,
  dp4a: boolean,
  geometry: I8a8Geometry = defaultI8a8Geometry("linear"),
  weight: WeightStorage = "i8",
  groupSize?: number,
): string => {
  assertI8a8Geometry(geometry, "linear i8a8");
  if (weight !== "i8" && weight !== "i4") {
    throw new CodegenError(`linear i8a8: 重み格納は i8 / i4 のみ（${weight}）`);
  }
  // MUST: i4 と group 長は対（欠け / 余りはどちらも結線バグ — weight-storage.ts）
  const shift = i4GroupShift("linear i8a8", weight, groupSize);
  if (shift !== undefined && groupSize !== undefined) {
    assertI8a8GroupGeometry(geometry, groupSize, "linear w4a8");
  }
  const kPacks = i8a8KPacks(geometry);
  return `// karume linear (x[m,k] · wᵀ[k,n] + bias[n], 活性 per-token i8 × 重み ${
    shift === undefined ? "i8" : `i4 群 ${groupSize}`
  } の整数内積${dp4a ? "" : "・エミュ"}, ${i8a8GeometryNote(geometry)}${v4 ? " + vec4" : ""})
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
${shift === undefined ? "" : i4IntegerLanesWgsl("w")}
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
${prologueB(geometry, shift === undefined ? "k4" : "dims.k")}
${
    shift === undefined || groupSize === undefined
      ? `  // ループ条件は uniform（dims は uniform バッファ）— 内側の workgroupBarrier が
  // WGSL の一様性要件を満たすために必要
  let tiles = (k4 + ${kPacks - 1}u) / ${kPacks}u;
${i8a8AccumulatorInit(geometry)}
  for (var t = 0u; t < tiles; t = t + 1u) {
${fillTiles(geometry)}
    workgroupBarrier();
${i8a8InnerProductLoop(geometry)}
    workgroupBarrier();
  }
${store(geometry, v4)}`
      : `${prologueW4a8(geometry, shift)}
${loopW4a8(geometry, groupSize)}
${storeW4a8(geometry, v4)}`
  }
}
`;
};

/**
 * uniform の Dims（`{m,n,k}` — f32 骨格と同じ 3 語 + 16 バイト整列）。
 *
 * `groupSize` を渡すと **w4a8 の門**（group 側のオーバフローと k の整除）へ切り替わる。
 * MUST: i8 変種の k 門（{@link LINEAR_I8A8_MAX_K}）は w4a8 に適用しない
 * （理由は {@link LINEAR_W4A8_MAX_GROUP}）。
 */
export const linearI8a8Params = (
  m: number,
  n: number,
  k: number,
  groupSize?: number,
): Uint32Array<ArrayBuffer> => {
  if (!Number.isSafeInteger(k) || k < 0 || k % 4 !== 0) {
    throw new CodegenError(`linear i8a8 params: k は 4 の倍数の非負整数（${k}）`);
  }
  if (groupSize === undefined) {
    if (k > LINEAR_I8A8_MAX_K) {
      throw new CodegenError(
        `linear i8a8 params: k=${k} が i32 縮約の門 ${LINEAR_I8A8_MAX_K} を超える`,
      );
    }
    return gemmParams("linear", m, n, k);
  }
  // group が K を割り切らないと最後の group が短くなり、flush の位置が WGSL の焼き込み
  // （tiles/group が定数）とずれる — 黙って通すと縮約が k を超えて隣の行を食う。
  if (k % groupSize !== 0) {
    throw new CodegenError(
      `linear w4a8 params: k=${k} が group_size ${groupSize} で割り切れない`,
    );
  }
  if (groupSize > LINEAR_W4A8_MAX_GROUP) {
    throw new CodegenError(
      `linear w4a8 params: group_size ${groupSize} が i32 縮約の門 ${LINEAR_W4A8_MAX_GROUP} を超える`,
    );
  }
  return gemmParams("linear", m, n, k);
};
