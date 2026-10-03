/**
 * conv3d（unbatched `x[Cin,T,H,W] * W[Cout,Cin/groups,Kt,Kh,Kw] + b[Cout]`、f32 — ADR 0118 決定 1）の
 * implicit GEMM カーネル。
 *
 * | 経路          | キー                                                       | 条件        |
 * | ------------- | ---------------------------------------------------------- | ----------- |
 * | implicit GEMM | `conv3d:v1:f32:igemm{tileM}x{tileN}{v4}:wg{x}x{y}{:w…}`    | groups == 1 |
 *
 * `C[Cout, N] = W[Cout, K] × Xcol[K, N]` を GEMM 骨格（src/kernels/gemm.ts）の断片共有で解く。
 * `M = Cout`・`N = Tout·Hout·Wout`（**時間軸は N に畳む**）・`K = Cin·Kt·Kh·Kw`。重み
 * `[Cout, Cin, Kt, Kh, Kw]` の平坦化が A タイル `[M, K]` そのもので、出力 `[Cout][Tout·Hout·Wout]`
 * の行優先が unbatched の出力テンソルそのもの（reshape ゼロ — ADR 0024 決定 1 と同じ論証）。
 *
 * ビット同一の土台は ADR 0024 決定 3 をそのまま継ぐ: 縮約は平坦 K 昇順（= `(ic, kt, kh, kw)` の
 * 四重昇順）・K タイル 16・bias は `acc` の初期値・範囲外の x は 0。この並びは Kt = 1 で conv2d の
 * `(ic, kh, kw)` と同じ列になるので、「Kt = 1 の conv3d ≡ フレームごとの conv2d」が Uint32 で
 * 一致する（tests/gpu_conv3d_parity_test.ts の恒等門 ①）。
 *
 * ## 実装済み subset（ADR 0064 軸 A — 意味論は CPU 参照が全体を持つ）
 *
 * - **groups == 1 だけ**。groups > 1 は Session の構築時に fail loudly（src/runtime/plan.ts）で、
 *   params 層（{@link conv3dIgemmParams}）でも二重に落とす。直接カーネルは持たない — 需要
 *   （depthwise 等）が出たら足す（ADR 0118 決定 1 の一般化の条件）。
 * - 重み格納は f32 / f16 / i8（A タイルの充填は 1D / 2D と共有なので、i8 の per-channel scale は
 *   行 = 出力チャネルで引かれる — ADR 0024 決定 6）。i4 / i2 は生成の入口（gemm.ts）が落とす。
 * - **幾何プロファイルは引かない**。既定の幾何（`gemmMTileGeometry(64 / 32)`）で走る（ADR 0118
 *   決定 1 — conv1d と同じ扱い。掃引ケースが無いまま conv2d 欄の幾何を 3D へ当てない）。
 *
 * MUST: dispatch は `[ceil(N/tileN), ceil(M/tileM), 1]` で、N のタイル数が上限を超える形は
 * `DispatchLimitError`（grid-stride で縮退できない — タイル欠落は沈黙誤値になる）。
 */

import { tiledWorkgroups } from "../codegen/dispatch.ts";
import { CodegenError } from "../codegen/errors.ts";
import { assertU32Params } from "../codegen/params.ts";
import { gemmMTileGeometry, gemmWgsl, LINEAR_SCALE_BINDING } from "./gemm.ts";
import { GEMM_TILE, gemmTileM, gemmTileN } from "./gemm-geometry.ts";
import { weightKeyPart, type WeightStorage } from "./weight-storage.ts";

/**
 * i8 変種の scale 束縛（出力の次の番号 — executor の bind entries と対で使う）。束縛配置は
 * linear / conv1d / conv2d の implicit GEMM と同じ（0 dims / 1 x / 2 重み / 3 bias / 4 出力）。
 */
export const CONV3D_SCALE_BINDING = LINEAR_SCALE_BINDING;

/** conv3d の幾何（params 関数の唯一の入力型）。 */
export type Conv3dDims = {
  readonly channelsIn: number;
  readonly channelsOut: number;
  readonly timeIn: number;
  readonly heightIn: number;
  readonly widthIn: number;
  readonly timeOut: number;
  readonly heightOut: number;
  readonly widthOut: number;
  readonly kernelT: number;
  readonly kernelH: number;
  readonly kernelW: number;
  readonly strideT: number;
  readonly strideH: number;
  readonly strideW: number;
  readonly paddingT: number;
  readonly paddingH: number;
  readonly paddingW: number;
  readonly dilationT: number;
  readonly dilationH: number;
  readonly dilationW: number;
  readonly groups: number;
};

/**
 * implicit GEMM の v4（vec4 読み書き）判定。
 *
 * MUST: 判定はここ 1 箇所。3 条件は conv2d（`conv2dUsesVec4`）と同じ理由で、最内軸は W:
 * - `kFlat % 4 == 0` … A 側の quad 読み（f16 の `dequant4` / i8 の `unpack4xI8` は平坦添字が
 *   4 の倍数であることに依存する）。**`Cin % 4` ではない**。
 * - `widthOut % 4 == 0` … B 側の列 quad が**出力行（とフレーム）をまたがない**こと。
 *   **`N % 4 == 0` では不十分**（Hout·Wout が 4 の倍数でも Wout が 4 の倍数でなければ quad が
 *   2 行に割れる）。store 側が要求する `n % 4 == 0` はこの条件から従う。
 * - `strideW === 1` … 4 列の x アドレスが連続すること。
 */
export const conv3dUsesVec4 = (kFlat: number, widthOut: number, strideW: number): boolean =>
  kFlat % 4 === 0 && widthOut % 4 === 0 && strideW === 1;

/**
 * implicit GEMM のパイプラインキー。v4 フラグは形状 → 1 ビットの写像（決定性は崩れない —
 * ADR 0022 決定 2 と同じ語彙）。m タイルの選択述語は `conv2dIgemmMTile`（src/kernels/conv2d.ts）を
 * 共有する（M = Cout の関数でしかなく次元に依らない）。
 *
 * MUST: キーの幾何は生成と**同じ解決点**（`gemmMTileGeometry`）から導く。mTile を直に埋めると、
 * 幾何を差し替えたときにキーだけが古い辺を名乗って別物の WGSL へ衝突する。
 */
export const conv3dIgemmKey = (
  weight: WeightStorage,
  v4: boolean,
  mTile: number = GEMM_TILE,
): string => {
  const geometry = gemmMTileGeometry(mTile);
  return `conv3d:v1:f32:igemm${gemmTileM(geometry)}x${gemmTileN(geometry)}${
    v4 ? "v4" : ""
  }:wg${geometry.wgX}x${geometry.wgY}${weightKeyPart(weight)}`;
};

export const conv3dIgemmWgsl = (
  weight: WeightStorage,
  v4: boolean,
  mTile: number = GEMM_TILE,
): string => gemmWgsl({ op: "conv3d", v4, weight, mTile });

/**
 * implicit GEMM の dispatch `[ceil(N/tileN), ceil(M/tileM), 1]`（unbatched — z は 1）。
 *
 * MUST: タイル辺は生成・キーと**同じ解決点**（`gemmMTileGeometry(mTile)`）から導く — 辺を定数で
 * 持ち回ると、幾何と食い違ったときに `ceil(dim / 定数)` が実タイル辺の枚数を下回って出力タイルが
 * 欠落する（例外の出ない誤値）。
 * MUST: 上限超過は `DispatchLimitError`（{@link tiledWorkgroups}）。1 workgroup = 1 出力タイルの
 * 骨格は grid-stride で縮退できない。既定の tileN 128 で N ≤ 8,388,480（上限 65,535 の機）。
 */
export const conv3dIgemmWorkgroups = (
  dims: Conv3dDims,
  mTile: number,
  limit: number,
  where: string,
): readonly [number, number, number] => {
  const geometry = gemmMTileGeometry(mTile);
  return [
    tiledWorkgroups(
      dims.timeOut * dims.heightOut * dims.widthOut,
      gemmTileN(geometry),
      limit,
      where,
    ),
    tiledWorkgroups(dims.channelsOut, gemmTileM(geometry), limit, where),
    1,
  ];
};

/**
 * implicit GEMM の uniform Dims（`{m, n, k}` + 幾何 18 語 = 21 語なので 24 語 = 96 バイト確保 —
 * uniform アドレス空間の struct は 16 バイト整列 MUST）。
 *
 * `m = Cout` / `n = Tout·Hout·Wout` / `k = Cin·Kt·Kh·Kw`。
 * MUST: 並びは gemm.ts の `CONV3D_DIMS_EXTRA` と対。
 * MUST: stride / dilation / 実寸は正整数（stride 0 はループが進まず GPU ハング・0 の辺は縮約を
 * 1 度も回さず出力が bias 一色になる — 契約検査と二重だが、カーネル直呼びの経路も塞ぐ）。
 * MUST: `groups == 1` 専用（縮約帯がグループごとに違うので 1 枚の m タイルが同じ B タイルを
 * 共有できない）。groups > 1 は fail loudly（ADR 0118 決定 1 の実装済み subset）。
 * MUST: 軸ごとに `入力長 + 2·padding ≤ 2^31 − 1`（WGSL の座標演算が i32 — 理由は本体の門）。
 */
export const conv3dIgemmParams = (dims: Conv3dDims): Uint32Array<ArrayBuffer> => {
  // 名前は WGSL の Dims 欄名と対。並びがそのまま uniform の語順になる。
  const geometry = {
    channels_in: dims.channelsIn,
    time_in: dims.timeIn,
    height_in: dims.heightIn,
    width_in: dims.widthIn,
    height_out: dims.heightOut,
    width_out: dims.widthOut,
    kernel_t: dims.kernelT,
    kernel_h: dims.kernelH,
    kernel_w: dims.kernelW,
    stride_t: dims.strideT,
    stride_h: dims.strideH,
    stride_w: dims.strideW,
    padding_t: dims.paddingT,
    padding_h: dims.paddingH,
    padding_w: dims.paddingW,
    dilation_t: dims.dilationT,
    dilation_h: dims.dilationH,
    dilation_w: dims.dilationW,
  };
  assertU32Params("conv3d igemm params", {
    ...geometry,
    channels_out: dims.channelsOut,
    time_out: dims.timeOut,
    groups: dims.groups,
  });
  const positive: readonly (readonly [string, number])[] = [
    ["channels_in", dims.channelsIn],
    ["channels_out", dims.channelsOut],
    ["time_in", dims.timeIn],
    ["height_in", dims.heightIn],
    ["width_in", dims.widthIn],
    ["time_out", dims.timeOut],
    ["height_out", dims.heightOut],
    ["width_out", dims.widthOut],
    ["kernel_t", dims.kernelT],
    ["kernel_h", dims.kernelH],
    ["kernel_w", dims.kernelW],
    ["stride_t", dims.strideT],
    ["stride_h", dims.strideH],
    ["stride_w", dims.strideW],
    ["dilation_t", dims.dilationT],
    ["dilation_h", dims.dilationH],
    ["dilation_w", dims.dilationW],
    ["groups", dims.groups],
  ];
  for (const [name, value] of positive) {
    if (value < 1) throw new CodegenError(`conv3d igemm params: ${name} は正整数（${value}）`);
  }
  if (dims.groups !== 1) {
    throw new CodegenError(
      `conv3d igemm params: groups は 1 専用（${dims.groups}）— groups > 1 の GPU 実行経路は無い（ADR 0118 決定 1）`,
    );
  }
  const n = dims.timeOut * dims.heightOut * dims.widthOut;
  const k = dims.channelsIn * dims.kernelT * dims.kernelH * dims.kernelW;
  assertU32Params("conv3d igemm params", { n, k });
  // MUST: 軸ごとに `入力長 + 2·padding ≤ 2^31 − 1`。WGSL は入力座標を
  // `i32(出力座標·stride) + i32(k·dilation) − i32(padding)` の 32 ビット整数で組むので、真の座標が
  // i32 の域を出ると折り返し、範囲外 0 の門を実在する別の画素として通り抜ける（例外の出ない誤値）。
  // 出力長の式（shapes.ts）から `(out − 1)·stride ≤ L + 2p − d(K − 1) − 1` と
  // `k·dilation ≤ d(K − 1) ≤ L + 2p − 1` が従い、座標は `[−p, L + p − 1]` に入るので、この 1 条件で
  // 中間値が全て i32 に収まる。CPU 参照の意味論は変えない（GPU 実行の subset — ADR 0064 軸 A）。
  const axes: readonly (readonly [string, number, number])[] = [
    ["t", dims.timeIn, dims.paddingT],
    ["h", dims.heightIn, dims.paddingH],
    ["w", dims.widthIn, dims.paddingW],
  ];
  for (const [axis, length, padding] of axes) {
    const extent = length + 2 * padding;
    if (extent > 0x7fff_ffff) {
      throw new CodegenError(
        `conv3d igemm params: 軸 ${axis} の入力長 + 2·padding ${extent} が i32 の上限 2^31 − 1 を超える` +
          "（GPU の座標演算は i32 — 折り返すと範囲外 0 の門を通り抜けて誤値になる）",
      );
    }
  }
  const params = new Uint32Array(24);
  params[0] = dims.channelsOut;
  params[1] = n;
  params[2] = k;
  Object.values(geometry).forEach((value, index) => {
    params[index + 3] = value;
  });
  return params;
};
