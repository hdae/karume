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
 *
 * ## 形状バケット
 *
 * 既定は M ≥ 128 を前提にした形なので、skinny-M（M < 128）だけ {@link gemmGeometryForRows} が
 * 別の幾何を返す。**選択は静的な shape の純関数**（実行時オートチューンは ADR 0022 で禁止）で、
 * 返り値はそのままキーに載るため「同一キー → バイト同一 WGSL」は保たれる。
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
 * 小 M バケットの候補（{@link gemmGeometryForRows} の表）。
 *
 * ## なぜ M で切るか
 *
 * 既定 `M128N128` は 1 workgroup が 128 行 × 128 列を持つので、**M < 128 では行タイルが 1 枚しか
 * 立たない**。dispatch 数は `ceil(n / 128)` だけになり、EmbeddingGemma（bare T=4）の実測形
 * `M=4, N ∈ {256, 768, 1152, 3072}` では workgroup が 2〜24 個 = SM を埋め切らない
 * （実測 170 dispatch で 41.3ms・242.6µs/本 — 重み転送量から見た所要の 2 桁上）。
 *
 * 行タイルが 1 枚のとき、重み `[n,k]` は**どの幾何でもちょうど 1 回ずつ**読まれる（タイルを
 * 跨いで重複しない）ので、幾何の効き所は「同じ転送量をどれだけ並列に流せるか」だけになる。
 * grid 全体のスレッド数は `n · wgY / regN`、workgroup 数は `n / tileN` なので、
 * **tileM を M の直上に抑えて wgY を稼ぎ、tileN を絞って workgroup を増やす**のが方向。
 *
 * ## 整除の根拠（{@link assertGemmGeometry} の 5 条件）
 *
 * | 幾何 | tileM×tileN | threads | rowFillStride | quadFillStride | 共有 |
 * | --- | --- | --- | --- | --- | --- |
 * | M16N16 `r1×4 wg4×16` | 16×16 | 64 | 16 | 4 | 2,048 B |
 * | 既定 M128N128 `r8×8 wg16×16` | 128×128 | 256 | 64 | 8 | 16,384 B |
 *
 * `regN % 4 == 0`・`threads % 4`（K quad 数）`== 0`・`threads % (tileN/4)`（列 quad 数）`== 0`・
 * `tileM % rowFillStride == 0 && tileN % rowFillStride == 0`・`16 % quadFillStride == 0` を
 * 満たす。**`tileM ≥ threads / 4` が実質の上限**（行充填ストライドがタイル辺を越えられない）で、
 * tileM=16 なら threads ≤ 64、tileM=64 なら threads ≤ 256。
 *
 * ## 値の出どころ（RTX 3080 Ti・2026-08-11 掃引 — docs/research/2026-08-11-skinny-m-geometry.md）
 *
 * 候補 11 種 × EmbeddingGemma の実 linear 形状 × M ∈ {1,4,64,318,512} の ABBA 対計測
 * （全候補・全形状でビット同一を同時確認）。実 run の形状構成で重み付けした結果:
 *
 * - M = 1 / 4 / 64 とも **M16N16 が対既定 ×3.0〜3.2** で、2 段目に置いていた
 *   `M64N32 r4×4 wg8×16`（×2.5）を **M=64 でも**上回った — tileM を M の直上に置いて重みの
 *   読み直しを避けるより、行タイルを 4 枚に割ってでも workgroup を増やす方が勝つ。
 *   よってバケットは **M ≤ 64 → M16N16 の 1 段だけ**。
 * - `M16N8`（threads 32）は M16N16 と 1% 未満差の同着（3 点とも僅差で上だが規約上は同値）。
 *   タイの側は採らず、実装済み・検査済みの M16N16 を保持。
 * - M = 318 / 512 は `M64N32 r4×4 wg8×16` が ×1.67 / ×1.28 で最良（M16N16 は ×1.35 へ落ちる —
 *   行タイル 20〜32 枚の読み直しが効き始める）。65〜512 のバケットは当初 Anima / SBV2 への
 *   波及を理由に見送ったが、**両モデルの E2E A/B（ABBA・PNG/WAV 門込み）とセットで採用**
 *   （2026-08-11 波① — 実測は docs/research/2026-08-11-skinny-m-geometry.md §4）。
 * - M ≥ 513 は既定のまま。掃引の実測点は 512 までで、DiT（S = 1024 / 4096）の領域は
 *   2026-08-10 のタイル軸掃引で既定が選ばれている — 実測の無い区間を補間で埋めない。
 */
const GEOMETRY_M16N16: GemmGeometry = { regM: 1, regN: 4, wgX: 4, wgY: 16 };

const GEOMETRY_M64N32: GemmGeometry = { regM: 4, regN: 4, wgX: 8, wgY: 16 };

/**
 * 行数 M から幾何を選ぶ**静的テーブル**（matmul / bmm / linear の 3 経路専用）。
 *
 * MUST: 純関数 = プラン時 shape だけの関数であること。実行時オートチューン（実測して選び直す）は
 * ADR 0022 で禁じている — f32 縮約は担当割りしか自由がなく、選択が実行ごとに揺れると
 * 「同一キー → バイト同一 WGSL」も PNG / WAV 門のビット同一もキーの意味も同時に崩れる。
 * MUST: **M ≥ 513 は既定をそのまま返す**。DiT（S = 1024 / 4096）の実測選定（2026-08-10
 * タイル軸掃引）を動かさない境界で、掃引の実測点（M ≤ 512）の外側を補間で埋めない。
 * MUST: 融合 attention（①QK / ③PV）はこの表を通さない。あちらの既定は Anima の実測で選ばれた
 * ものなので、M（= クエリ長）で勝手に振り替えると実測の前提が消える。
 *
 * バケット境界 64 / 512 は掃引の実測境界（M=64 まで M16N16 が最良・M=318/512 は M64N32・
 * それより上は未実測 = 既定）。65〜512 段の採用は Anima / SBV2 の E2E A/B（ABBA・門込み）で
 * 退行なしを確認した上でのもの（research doc §4）。
 */
export const gemmGeometryForRows = (rows: number): GemmGeometry => {
  if (!Number.isSafeInteger(rows) || rows < 0) {
    throw new CodegenError(`幾何の選択: 行数 M は非負整数（${rows}）`);
  }
  if (rows <= 64) return GEOMETRY_M16N16;
  if (rows <= 512) return GEOMETRY_M64N32;
  return defaultGemmGeometry();
};

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
