/**
 * f32 / f16 GEMM 骨格（src/kernels/gemm.ts が持つ matmul / bmm / linear / 融合 attention の
 * ①QK・③PV / conv2d の implicit GEMM）が共有する**タイル幾何のパラメタ面**。
 *
 * ```
 * 出力タイル = mTile × GEMM_TILE      （mTile は 64 か 32 — conv2d だけが 32 を取る）
 * tileN = regN · wgX = GEMM_TILE      （列は常に 1 タイル 64 を覆い切る）
 * wgY   = mTile / regM               （行は 1 スレッド regM 行）
 * threads = wgX · wgY                （1 スレッドが regM×regN の出力を持つ）
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
 * 規約つき）で `r8×4 wg16×8`（128 スレッド）が最良で、f16-1024 が ×1.287・w8a8-s16-1024 が
 * ×1.103（いずれも PNG / WAV の sha256 門はビット同一）。旧 `r4×4 wg16×16`（256 スレッド）
 * から効いたのは **1 スレッドあたりのレジスタブロック（regM·regN = 16 → 32）**で、
 * i8a8 側の掃引が示した「レジスタブロックが第一・threads ≤ 128 が第二」と同じ順序。
 */

import { CodegenError } from "../codegen/errors.ts";

/** 出力タイルの一辺（列は常にこれ、行は conv2d だけ 32 を取る）。dispatch は `ceil(dim / 64)`。 */
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

/** 出力タイルの列 quad 数（共有 B タイルの列ストライド）。 */
export const GEMM_N_QUADS = GEMM_TILE / GEMM_QUAD;

/**
 * 生成パラメタ。列側のタイル辺は `regN · wgX` で {@link GEMM_TILE} を覆い切る前提
 * （{@link assertGemmGeometry} が門）、行側は `regM · wgY = mTile` を wgY 側で吸収する
 * ので wgY は持たない（持つと mTile と食い違いうる値が 2 つになる）。
 */
export type GemmGeometry = {
  /** 1 スレッドが持つ出力の行数。 */
  readonly regM: number;
  /** 1 スレッドが持つ出力の列数（`acc` を vec4 に束ねる都合で {@link GEMM_QUAD} の倍数）。 */
  readonly regN: number;
  /** workgroup の x 辺（列方向）。 */
  readonly wgX: number;
};

/** workgroup の y 辺（1 スレッド regM 行なので `mTile / regM`）。 */
export const gemmWorkgroupRows = (geometry: GemmGeometry, mTile: number): number =>
  mTile / geometry.regM;

/** 1 workgroup のスレッド数。 */
export const gemmThreads = (geometry: GemmGeometry, mTile: number): number =>
  geometry.wgX * gemmWorkgroupRows(geometry, mTile);

/** 1 スレッドが持つ列 quad 数（`acc{行}_{列 quad}` の列側の本数）。 */
export const gemmQuadsPerThread = (geometry: GemmGeometry): number => geometry.regN / GEMM_QUAD;

/**
 * `[行][k quad]` 配置のタイル（A / linear の W / ①QK の k / conv2d の重み）を埋めるとき、
 * 1 スロットで進む行数。1 スレッドは自分の行から この刻みで {@link gemmRowSlots} 本
 * （W 側は {@link gemmColumnSlots} 本）を受け持つ。
 */
export const gemmRowFillStride = (geometry: GemmGeometry, mTile: number): number =>
  gemmThreads(geometry, mTile) / GEMM_K_QUADS;

/** A タイル（`mTile` 行）を 1 スレッドが埋める本数。 */
export const gemmRowSlots = (geometry: GemmGeometry, mTile: number): number =>
  mTile / gemmRowFillStride(geometry, mTile);

/** W / k タイル（出力チャネル = 列を行として読むので {@link GEMM_TILE} 本）の担当本数。 */
export const gemmColumnSlots = (geometry: GemmGeometry, mTile: number): number =>
  GEMM_TILE / gemmRowFillStride(geometry, mTile);

/** `[k 行][列 quad]` 配置の dense B タイルを埋めるとき、1 スロットで進む k 行数。 */
export const gemmQuadFillStride = (geometry: GemmGeometry, mTile: number): number =>
  gemmThreads(geometry, mTile) / GEMM_N_QUADS;

/** dense B タイル（K {@link GEMM_TILE_K} 行）を 1 スレッドが埋める本数。 */
export const gemmQuadSlots = (geometry: GemmGeometry, mTile: number): number =>
  GEMM_TILE_K / gemmQuadFillStride(geometry, mTile);

/**
 * 幾何の整合を生成時に落とす。**割り切れない組み合わせは共有タイルの穴になる**
 * （充填が届かない語が 0 のまま内積へ入る = 例外の出ない誤値）ので、fail loudly が唯一の門。
 */
export const assertGemmGeometry = (
  geometry: GemmGeometry,
  mTile: number,
  where: string,
): void => {
  const { regM, regN, wgX } = geometry;
  for (
    const [name, value] of [
      ["regM", regM],
      ["regN", regN],
      ["wgX", wgX],
      ["mTile", mTile],
    ] as const
  ) {
    if (!Number.isSafeInteger(value) || value < 1) {
      throw new CodegenError(`${where}: 幾何の ${name} は正整数（${value}）`);
    }
  }
  if (regN % GEMM_QUAD !== 0) {
    throw new CodegenError(`${where}: regN は ${GEMM_QUAD} の倍数（acc は vec4 束ね・${regN}）`);
  }
  if (regN * wgX !== GEMM_TILE) {
    throw new CodegenError(
      `${where}: regN·wgX が出力タイルの列 ${GEMM_TILE} と一致しない（${regN}·${wgX}）`,
    );
  }
  if (mTile % regM !== 0) {
    throw new CodegenError(`${where}: m タイル ${mTile} が regM ${regM} で割り切れない`);
  }
  const threads = gemmThreads(geometry, mTile);
  for (
    const [name, unit] of [["K quad", GEMM_K_QUADS], ["列 quad", GEMM_N_QUADS]] as const
  ) {
    if (threads % unit !== 0) {
      throw new CodegenError(
        `${where}: スレッド数 ${threads} が ${name} 数 ${unit} で割り切れない（充填の担当が組めない）`,
      );
    }
  }
  const rowStride = gemmRowFillStride(geometry, mTile);
  if (mTile % rowStride !== 0 || GEMM_TILE % rowStride !== 0) {
    throw new CodegenError(
      `${where}: タイル辺 ${mTile}x${GEMM_TILE} が行充填ストライド ${rowStride} で割り切れない`,
    );
  }
  const quadStride = gemmQuadFillStride(geometry, mTile);
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
 * `r8×4 wg16×8` とも組めるので、辺だけのキーは別物へ衝突する）。conv2d の implicit GEMM は
 * 自前のキー（`igemm{mTile}x64…:wg{x}x{y}`）が workgroup 形を載せているので、そちらは
 * この断片を使わずに同じ判別力を持つ。
 */
export const gemmGeometryKeyPart = (geometry: GemmGeometry): string =>
  `r${geometry.regM}x${geometry.regN}w${geometry.wgX}`;

/** 生成コメント用の幾何注記（WGSL ヘッダ 1 行に載せる形）。 */
export const gemmGeometryNote = (geometry: GemmGeometry, mTile: number): string =>
  `レジスタ ${mTile}x${GEMM_TILE} タイル / 1 スレッド ${geometry.regM}x${geometry.regN} / wg ${geometry.wgX}x${
    gemmWorkgroupRows(geometry, mTile)
  }`;

/**
 * 既定の幾何を決める**唯一の純関数**。全 op 共通の 1 点で、op 別の既定は持たない
 * （必要になったら i8a8 側（{@link "./i8a8-geometry.ts"} `defaultI8a8Geometry`）と同じ形に
 * 上げられるので、今は要らない機械を足さない）。
 *
 * `r8×4 wg16×8` の出どころはモジュール冒頭の実測。
 */
export const defaultGemmGeometry = (): GemmGeometry => ({ regM: 8, regN: 4, wgX: 16 });

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
