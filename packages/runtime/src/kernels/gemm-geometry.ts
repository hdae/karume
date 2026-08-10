/**
 * f32 / f16 GEMM 骨格（src/kernels/gemm.ts が持つ matmul / bmm / linear / 融合 attention の
 * ①QK・③PV / conv2d の implicit GEMM）が共有する**タイル幾何のパラメタ面**。
 *
 * ```
 * tileM = regM · wgY   （1 workgroup が持つ出力の行数 — conv2d だけが 32 の変種を取る）
 * tileN = regN · wgX   （                     列数）
 * threads = wgX · wgY  （1 スレッドが regM×regN の出力を持つ）
 * ```
 *
 * ## なぜパラメタにしてよいか（数値契約との関係）
 *
 * MUST: 幾何が決めてよいのは**どのスレッドがどの出力を担当するか**だけ。1 出力要素あたりの
 * K 縮約順（外側 `t` 昇順・内側 `kk` 昇順）と積和式の字面（`acc = acc + a * b`）は幾何に
 * よらず不変で、ここが ADR 0022 決定 3 の数値契約そのもの（i8a8 の i32 縮約と違い f32 の
 * 加算は順序依存なので、担当割り以外を触ると値が動く）。
 * MUST: 幾何は**パイプラインキーに載せる**（{@link gemmGeometryKeyPart}）。載せないと
 * 「同一キー → バイト同一 WGSL」の決定性が崩れる。
 * MUST: 概念が重なっても i8a8 側（src/kernels/i8a8-geometry.ts）と実装を共有しない。
 * あちらは整数縮約が順序非依存だからこそ幾何が自由で、こちらは f32 縮約の担当割りだけが
 * 自由 — 数値契約が別物なので、独立していること自体が不変条件になっている。
 *
 * ## 既定の出どころ
 *
 * {@link defaultGemmGeometry} が唯一の選択点。RTX 3080 Ti の E2E 実測（ABBA 回文・冷却
 * 規約つき — docs/research/2026-08-10-f32-geometry-probe.md）で、64×64 掃引の最良
 * `r8×4 wg16×8` に対しタイル軸 128 の `M128N128 r8×8 wg16×16`（256 スレッド・共有 16KB）が
 * さらに f16-1024 ×1.17（w8a8-s16-1024 は中立）。効いた順は「レジスタブロック第一
 * （regM·regN = 32 → 64）・タイル辺第二」で i8a8 側の掃引と同じだが、最良のタイル形は
 * i8a8（M128N64）と**一致しない** — 幾何の答えは経路ごとの実測でしか出ない。
 * いずれの幾何でも PNG / WAV の sha256 門はビット同一（実測命題 — ADR 0022 追記）。
 */

import { CodegenError } from "../codegen/errors.ts";

/**
 * conv2d の m タイルヒューリスティック（`conv2dIgemmMTile`）の基準行数。既定幾何の
 * tileM（128）とは別物 — conv2d は出力チャネル数が小さいので m タイルを 64 / 32 に抑える。
 *
 * MUST: **実タイル辺の正本は幾何**（{@link gemmTileM} / {@link gemmTileN}）。この定数は
 * ヒューリスティックの語彙で、生成・dispatch が辺として直接読むと幾何と食い違いうる値が
 * 2 つになる。
 */
export const GEMM_TILE = 64;

/**
 * K タイル幅。**幾何ではなく数値契約**（ADR 0022 の MUST）なので {@link GemmGeometry} には
 * 入れない — 変えると 1 出力要素あたりの加算の刻みが動く。
 */
export const GEMM_TILE_K = 16;

/** vec4 の成分数。共有 B タイルの束ね方と入出力の quad 読み書きの粒度で、**幾何とは別軸**。 */
export const GEMM_QUAD = 4;

/** K タイルあたりの quad 数（A / W タイル充填のスレッド割当に使う）。 */
export const GEMM_K_QUADS = GEMM_TILE_K / GEMM_QUAD;

/**
 * 生成パラメタ。出力タイル辺は `regM · wgY` × `regN · wgX` で**導出**する
 * （タイル辺を独立に持つと食い違いうる値が 2 つになる — i8a8 側と同じ規律）。
 */
export type GemmGeometry = {
  /** 1 スレッドが持つ出力の行数。 */
  readonly regM: number;
  /** 1 スレッドが持つ出力の列数（`acc` を vec4 に束ねる都合で {@link GEMM_QUAD} の倍数）。 */
  readonly regN: number;
  /** workgroup の x 辺（列方向）。 */
  readonly wgX: number;
  /** workgroup の y 辺（行方向）。 */
  readonly wgY: number;
};

/** 出力タイルの行数。dispatch の y は `ceil(m / tileM)`。 */
export const gemmTileM = (geometry: GemmGeometry): number => geometry.regM * geometry.wgY;

/** 出力タイルの列数。dispatch の x は `ceil(n / tileN)`。 */
export const gemmTileN = (geometry: GemmGeometry): number => geometry.regN * geometry.wgX;

/** 1 workgroup のスレッド数。 */
export const gemmThreads = (geometry: GemmGeometry): number => geometry.wgX * geometry.wgY;

/** 1 スレッドが持つ列 quad 数（`acc{行}_{列 quad}` の列側の本数）。 */
export const gemmQuadsPerThread = (geometry: GemmGeometry): number => geometry.regN / GEMM_QUAD;

/** 出力タイルの列 quad 数（共有 B タイルの列ストライド）。 */
export const gemmColumnQuads = (geometry: GemmGeometry): number => gemmTileN(geometry) / GEMM_QUAD;

/**
 * `[行][k quad]` 配置のタイル（A / linear の W / ①QK の k / conv2d の重み）を埋めるとき、
 * 1 スロットで進む行数。1 スレッドは自分の行から この刻みで {@link gemmRowSlots} 本
 * （W 側は {@link gemmColumnSlots} 本）を受け持つ。
 */
export const gemmRowFillStride = (geometry: GemmGeometry): number =>
  gemmThreads(geometry) / GEMM_K_QUADS;

/** A タイル（{@link gemmTileM} 行）を 1 スレッドが埋める本数。 */
export const gemmRowSlots = (geometry: GemmGeometry): number =>
  gemmTileM(geometry) / gemmRowFillStride(geometry);

/** W / k タイル（出力チャネル = 列を行として読むので {@link gemmTileN} 本）の担当本数。 */
export const gemmColumnSlots = (geometry: GemmGeometry): number =>
  gemmTileN(geometry) / gemmRowFillStride(geometry);

/** `[k 行][列 quad]` 配置の dense B タイルを埋めるとき、1 スロットで進む k 行数。 */
export const gemmQuadFillStride = (geometry: GemmGeometry): number =>
  gemmThreads(geometry) / gemmColumnQuads(geometry);

/** dense B タイル（K {@link GEMM_TILE_K} 行）を 1 スレッドが埋める本数。 */
export const gemmQuadSlots = (geometry: GemmGeometry): number =>
  GEMM_TILE_K / gemmQuadFillStride(geometry);

/**
 * 幾何の整合を生成時に落とす。**割り切れない組み合わせは共有タイルの穴になる**
 * （充填が届かない語が 0 のまま内積へ入る = 例外の出ない誤値）ので、fail loudly が唯一の門。
 */
export const assertGemmGeometry = (geometry: GemmGeometry, where: string): void => {
  const { regM, regN, wgX, wgY } = geometry;
  for (
    const [name, value] of [
      ["regM", regM],
      ["regN", regN],
      ["wgX", wgX],
      ["wgY", wgY],
    ] as const
  ) {
    if (!Number.isSafeInteger(value) || value < 1) {
      throw new CodegenError(`${where}: 幾何の ${name} は正整数（${value}）`);
    }
  }
  // regN が quad の倍数なら tileN も倍数なので、v4 経路の `n4` 換算はここで従う。
  if (regN % GEMM_QUAD !== 0) {
    throw new CodegenError(`${where}: regN は ${GEMM_QUAD} の倍数（acc は vec4 束ね・${regN}）`);
  }
  const threads = gemmThreads(geometry);
  for (
    const [name, unit] of [
      ["K quad", GEMM_K_QUADS],
      ["列 quad", gemmColumnQuads(geometry)],
    ] as const
  ) {
    if (threads % unit !== 0) {
      throw new CodegenError(
        `${where}: スレッド数 ${threads} が ${name} 数 ${unit} で割り切れない（充填の担当が組めない）`,
      );
    }
  }
  const tileM = gemmTileM(geometry);
  const tileN = gemmTileN(geometry);
  const rowStride = gemmRowFillStride(geometry);
  if (tileM % rowStride !== 0 || tileN % rowStride !== 0) {
    throw new CodegenError(
      `${where}: タイル辺 ${tileM}x${tileN} が行充填ストライド ${rowStride} で割り切れない`,
    );
  }
  const quadStride = gemmQuadFillStride(geometry);
  if (GEMM_TILE_K % quadStride !== 0) {
    throw new CodegenError(
      `${where}: K タイル ${GEMM_TILE_K} が quad 充填ストライド ${quadStride} で割り切れない`,
    );
  }
};

/**
 * パイプラインキーの幾何判別子（`reg64x64` の後ろに付く形）。
 *
 * **タイル辺だけでは幾何が決まらない**ぶんを足す（64×64 は `r4×4 wg16×16` とも
 * `r8×4 wg16×8` とも組めるので、辺だけのキーは別物へ衝突する）。wgY は載せない —
 * `tileM / regM` で決まるので、辺と regM を載せた時点で判別力は尽きている。conv2d の
 * implicit GEMM は自前のキー（`igemm{tileM}x{tileN}…:wg{x}x{y}`）が workgroup 形を
 * 載せているので、そちらはこの断片を使わずに同じ判別力を持つ。
 */
export const gemmGeometryKeyPart = (geometry: GemmGeometry): string =>
  `r${geometry.regM}x${geometry.regN}w${geometry.wgX}`;

/** 生成コメント用の幾何注記（WGSL ヘッダ 1 行に載せる形）。 */
export const gemmGeometryNote = (geometry: GemmGeometry): string =>
  `レジスタ ${gemmTileM(geometry)}x${
    gemmTileN(geometry)
  } タイル / 1 スレッド ${geometry.regM}x${geometry.regN} / wg ${geometry.wgX}x${geometry.wgY}`;

/**
 * 既定の幾何を決める**唯一の純関数**。全 op 共通の 1 点で、op 別の既定は持たない
 * （必要になったら i8a8 側（{@link "./i8a8-geometry.ts"} `defaultI8a8Geometry`）と同じ形に
 * 上げられるので、今は要らない機械を足さない）。
 *
 * `M128N128 r8×8 wg16×16` の出どころはモジュール冒頭の実測。
 * MUST: 既定の変更は PNG / WAV 門の再実測とセット（ADR 0022 追記 — ビット同一は実測命題）。
 */
export const defaultGemmGeometry = (): GemmGeometry => ({ regM: 8, regN: 8, wgX: 16, wgY: 16 });

/**
 * f32 accumulator の初期化。**配列 1 本ではなく `acc{行}_{列 quad}` の名前付き変数**にするのは、
 * `acc[i]` の動的添字がアドレス可能な関数ローカル領域を要求し、レジスタに載らずローカル
 * メモリへ落ちるため（Metal で顕著）。展開しても K タイル幅・`t` / `kk` の昇順・1 出力要素
 * あたりの加算順序は変わらない（出力ごとに独立な縮約を名前へ割り当て直すだけ）。
 *
 * `init` は 1 行ぶんの初期値式（既定は 0）。conv2d だけが **bias-first**（ADR 0024 の MUST）で
 * 行 = 出力チャネルの bias を差し込む。
 */
export const gemmAccumulatorInit = (
  geometry: GemmGeometry,
  init: (row: number) => string = () => "vec4<f32>(0.0)",
): string =>
  Array.from(
    { length: geometry.regM },
    (_, row) =>
      Array.from(
        { length: gemmQuadsPerThread(geometry) },
        (_, quad) => `  var acc${row}_${quad} = ${init(row)};`,
      ).join("\n"),
  ).join("\n");
