/**
 * i8a8 GEMM 族（linear / 融合 attention の ①QK・③PV）が共有する**タイル幾何のパラメタ面**。
 *
 * 3 本のカーネルは生成器を共有しない（違いが断片差し込みでは吸収できない位置に散っている —
 * src/kernels/linear-i8a8.ts の docstring）が、**幾何の算術と内積ループだけはここ 1 箇所**に
 * 置く。幾何を各ファイルの定数へ書き写すと、`sa` / `sb` の添字算術（pack ストライド・充填の
 * スレッド割当）が 3 通りに分岐して、片方だけ静かに壊れる形になる。
 *
 * ```
 * tileM = regM · wgY   （1 workgroup が持つ出力の行数）
 * tileN = regN · wgX   （                     列数）
 * threads = wgX · wgY  （1 スレッドが regM×regN の出力を持つ）
 * ```
 *
 * ## なぜパラメタにしてよいか（数値契約との関係）
 *
 * i8a8 の縮約は **i32 の厳密加算で順序非依存**なので、「どのスレッドがどの出力を担当するか」
 * を変えても返る整数は 1 ビットも動かない（RTX 3080 Ti で 100 変種以上を production カーネルと
 * バイト単位で突合し不一致 0 件 — docs/research/2026-08-10-kernel-variant-sweep.md）。
 * MUST: この自由度は **i8a8 経路だけのもの**。f32 / f16 の GEMM（src/kernels/gemm.ts）は
 * K 縮約順が数値契約そのもの（ADR 0022）で、同じ理屈は使えない。
 * MUST: 幾何は**パイプラインキーに載せる**（{@link i8a8GeometryKeyPart}）。載せないと
 * 「同一キー → バイト同一 WGSL」の決定性が崩れる（`gemmUsesVec4` の v4 フラグと同じ語彙）。
 *
 * ## 既定の出どころ
 *
 * {@link defaultI8a8Geometry} が唯一の選択点。実測（同上・§2 / §3.2）で
 * linear と ①QK は `M128N64 r8×8 wg8×16 K16`（3 形状で ×1.39〜1.47 / QK ×1.92）、
 * ③PV だけ `M64N128 r8×8 wg16×8 K16`（N = D = 128 が 1 タイルに収まる形 — ×1.74）が勝つ。
 * 律速は LDS 帯域 × 発行で、効くのは**レジスタブロック（regM·regN）が第一**・
 * threads ≤ 128 が第二・tileK が第三。
 */

import { CodegenError } from "../codegen/errors.ts";

/** i8 ペイロードの 4 詰め（u32 1 語 = 4 要素）。dp4a の粒度そのもの。 */
export const I8A8_PACK = 4;

/** 幾何を選ぶ単位。**キーの op 欄と一致させる**（診断で読み手が対応を取れる形にする）。 */
export type I8a8GemmOp = "linear" | "attention_qk" | "attention_pv";

/**
 * 生成パラメタ。出力タイル辺は `regM · wgY` × `regN · wgX` で**導出**する
 * （タイル辺を独立に持つと 3 つの値が食い違いうる — 導出なら食い違いようがない）。
 */
export type I8a8Geometry = {
  /** 1 スレッドが持つ出力の行数。 */
  readonly regM: number;
  /** 1 スレッドが持つ出力の列数（`acc` を vec4 に束ねる都合で 4 の倍数）。 */
  readonly regN: number;
  /** workgroup の x 辺（列方向）。 */
  readonly wgX: number;
  /** workgroup の y 辺（行方向）。 */
  readonly wgY: number;
  /** K タイル幅（要素数 — pack 数は `tileK / 4`）。 */
  readonly tileK: number;
};

/** 出力タイルの行数。dispatch の y は `ceil(m / tileM)`。 */
export const i8a8TileM = (geometry: I8a8Geometry): number => geometry.regM * geometry.wgY;

/** 出力タイルの列数。dispatch の x は `ceil(n / tileN)`。 */
export const i8a8TileN = (geometry: I8a8Geometry): number => geometry.regN * geometry.wgX;

/** 1 workgroup のスレッド数。 */
export const i8a8Threads = (geometry: I8a8Geometry): number => geometry.wgX * geometry.wgY;

/** K タイルあたりのパック数（共有タイルの pack ストライド）。 */
export const i8a8KPacks = (geometry: I8a8Geometry): number => geometry.tileK / I8A8_PACK;

/**
 * 共有タイル充填の担当ストライド。1 スレッドは `tid / kPacks` 行目から
 * このストライドで {@link i8a8ASlots} 本（B 側は {@link i8a8BSlots} 本）を埋める。
 */
export const i8a8FillStride = (geometry: I8a8Geometry): number =>
  i8a8Threads(geometry) / i8a8KPacks(geometry);

/** A（活性 / q / P̃）タイルを 1 スレッドが埋める本数。 */
export const i8a8ASlots = (geometry: I8a8Geometry): number =>
  i8a8TileM(geometry) / i8a8FillStride(geometry);

/** B（重み / k / Vᵀ）タイルを 1 スレッドが埋める本数。 */
export const i8a8BSlots = (geometry: I8a8Geometry): number =>
  i8a8TileN(geometry) / i8a8FillStride(geometry);

/**
 * 幾何の整合を生成時に落とす。**割り切れない組み合わせは共有タイルの穴になる**
 * （充填が届かない語が 0 のまま内積へ入る = 例外の出ない誤値）ので、fail loudly が唯一の門。
 */
export const assertI8a8Geometry = (geometry: I8a8Geometry, where: string): void => {
  const { regM, regN, wgX, wgY, tileK } = geometry;
  for (
    const [name, value] of [["regM", regM], ["regN", regN], ["wgX", wgX], ["wgY", wgY], [
      "tileK",
      tileK,
    ]] as const
  ) {
    if (!Number.isSafeInteger(value) || value < 1) {
      throw new CodegenError(`${where}: 幾何の ${name} は正整数（${value}）`);
    }
  }
  if (regN % I8A8_PACK !== 0) {
    throw new CodegenError(`${where}: regN は ${I8A8_PACK} の倍数（acc は vec4 束ね・${regN}）`);
  }
  if (tileK % I8A8_PACK !== 0) {
    throw new CodegenError(`${where}: tileK は ${I8A8_PACK} の倍数（i8 の 4 詰め・${tileK}）`);
  }
  const threads = i8a8Threads(geometry);
  const kPacks = i8a8KPacks(geometry);
  if (threads % kPacks !== 0) {
    throw new CodegenError(
      `${where}: スレッド数 ${threads} が K パック数 ${kPacks} で割り切れない（充填の担当が組めない）`,
    );
  }
  const stride = i8a8FillStride(geometry);
  if (i8a8TileM(geometry) % stride !== 0 || i8a8TileN(geometry) % stride !== 0) {
    throw new CodegenError(
      `${where}: タイル辺 ${i8a8TileM(geometry)}x${
        i8a8TileN(geometry)
      } が充填ストライド ${stride} で割り切れない`,
    );
  }
};

/**
 * パイプラインキーの幾何判別子。`reg64x64v4`（f32 骨格の {@link "./gemm.ts"} `gemmKeyPart`）の
 * 流儀を保ったまま、**タイル辺だけでは幾何が決まらなくなった**ぶんを足す
 * （`M128N64` は `r8×8 wg8×16` とも `r16×8 wg8×8` とも組めるので、タイル辺だけのキーは
 * 別物へ衝突する）。v4 フラグの位置と綴りは従来どおり末尾。
 */
export const i8a8GeometryKeyPart = (geometry: I8a8Geometry, v4: boolean): string =>
  `tile${i8a8TileM(geometry)}x${i8a8TileN(geometry)}r${geometry.regM}x${geometry.regN}` +
  `w${geometry.wgX}x${geometry.wgY}k${geometry.tileK}${v4 ? "v4" : ""}`;

/** linear / ①QK の既定（実測で 3 形状すべてが同じ最良点を指した唯一の変種）。 */
const GEOMETRY_M128N64: I8a8Geometry = { regM: 8, regN: 8, wgX: 8, wgY: 16, tileK: 16 };

/** ③PV の既定（N = D = 128 を 1 タイルで覆う形が self / cross とも勝つ）。 */
const GEOMETRY_M64N128: I8a8Geometry = { regM: 8, regN: 8, wgX: 16, wgY: 8, tileK: 16 };

/**
 * 既定の幾何を決める**唯一の純関数**。
 *
 * 現状は op ごとの定数で、形状バケットは切っていない — L2（4096×8192×2048）だけは
 * `r16×8 wg8×8` が僅差（~6%）で勝つが、境界を 1 本切るとキー本数と検証面が倍になる。
 * 将来「手動チューンで作ったプロファイルを渡して既定を上書きする」形はここを差し替え点に
 * する（呼び出し側は幾何を受け取って流すだけなので、差し替えの影響がこの関数に閉じる）。
 */
export const defaultI8a8Geometry = (op: I8a8GemmOp): I8a8Geometry =>
  op === "attention_pv" ? GEOMETRY_M64N128 : GEOMETRY_M128N64;

/**
 * i32 accumulator の初期化。**配列 1 本ではなく `acc{行}_{列 quad}` の名前付き変数**にするのは、
 * `acc[i]` の動的添字がアドレス可能な関数ローカル領域を要求し、レジスタに載らずローカル
 * メモリへ落ちるため（Metal で顕著）。展開しても縮約は i32 の厳密加算のまま・出力ごとに
 * 独立なので、**返る整数は 1 ビットも変わらない**。
 */
export const i8a8AccumulatorInit = (geometry: I8a8Geometry): string => {
  const quads = geometry.regN / I8A8_PACK;
  return Array.from(
    { length: geometry.regM },
    (_, row) =>
      Array.from({ length: quads }, (_, quad) => `  var acc${row}_${quad} = vec4<i32>(0);`).join(
        "\n",
      ),
  )
    .join("\n");
};

/**
 * K タイル内側の内積ループ。共有タイル（`sa` = `[pack][行]` / `sb` = `[pack][列]`）から
 * B を `regN` 語・A を `regM` 語読んで `regM · regN` 個の整数内積を回す。
 *
 * MUST: 3 本のカーネルはこの 1 本を使う（幾何の添字算術を書き写すと、片方だけ pack ストライドを
 * 取り違える形になる）。1 出力あたりの加算順序は「K タイル昇順・pack 内 4 語まとめ」で、
 * 幾何を変えても i32 の厳密加算なので結果は順序非依存。
 */
export const i8a8InnerProductLoop = (geometry: I8a8Geometry): string => {
  const { regM, regN } = geometry;
  const quads = regN / I8A8_PACK;
  const kPacks = i8a8KPacks(geometry);
  const at = (offset: number): string => offset === 0 ? "" : ` + ${offset}u`;
  const bLoads = Array.from(
    { length: regN },
    (_, col) => `      let b${col} = sb[bcol${at(col)}];`,
  ).join("\n");
  const rows = Array.from({ length: regM }, (_, row) => {
    const updates = Array.from({ length: quads }, (_, quad) => {
      const lanes = Array.from(
        { length: I8A8_PACK },
        (_, lane) => `idot(a${row}, b${quad * I8A8_PACK + lane})`,
      ).join(", ");
      return `      acc${row}_${quad} = acc${row}_${quad} + vec4<i32>(${lanes});`;
    }).join("\n");
    return `      let a${row} = sa[arow_at${at(row)}];\n${updates}`;
  }).join("\n");
  return `    // 共有ロード ${regN + regM} 回（B の ${regN} 語 + A の ${regM} 語）で ${
    regM * regN
  } 個の整数内積 = ${regM * regN * I8A8_PACK} MAC。
    // 縮約は i32 の厳密加算なので**順序に依存しない**（f32 骨格と違い加算順は数値契約に無い）
    for (var p = 0u; p < ${kPacks}u; p = p + 1u) {
      let bcol = p * ${i8a8TileN(geometry)}u + lid.x * ${regN}u;
${bLoads}
      let arow_at = p * ${i8a8TileM(geometry)}u + lid.y * ${regM}u;
${rows}
    }`;
};

/** 生成コメント用の幾何注記（ヘッダ 1 行に載せる形）。 */
export const i8a8GeometryNote = (geometry: I8a8Geometry): string =>
  `タイル ${i8a8TileM(geometry)}x${
    i8a8TileN(geometry)
  } / 1 スレッド ${geometry.regM}x${geometry.regN} / wg ${geometry.wgX}x${geometry.wgY} / K ${geometry.tileK}`;
