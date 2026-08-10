/**
 * conv2d（`x[B,Cin,H,W] * W[Cout,Cin/groups,Kh,Kw] + b[Cout]`、f32）の 2 カーネル。
 *
 * | 経路                | キー                                              | 条件        |
 * | ------------------- | ------------------------------------------------- | ----------- |
 * | implicit GEMM       | `conv2d:v2:f32:igemm{64|32}x64{v4}:wg16x{8|4}{:w…}` | groups == 1 |
 * | 直接畳み込み        | `conv2d:v1:f32:direct:wg256{:w…}`                 | groups > 1  |
 *
 * implicit GEMM（ADR 0024）は `C[Cout, N] = W[Cout, K] × Xcol[K, N]` を GEMM 骨格
 * （src/kernels/gemm.ts）の断片共有で解く。**縮約順序が直接畳み込みと厳密に一致する**ので
 * 出力はビット同一（唯一の例外 = 符号付きゼロ — {@link conv2dIgemmKey} の doc）。
 * 直接畳み込み（ADR 0017）は 1 スレッド = 1 出力要素で、groups > 1 を受け持つと同時に
 * **恒久の差分オラクル**（tests/gpu_conv2d_parity_test.ts）を兼ねる。
 *
 * ## 契約の狭さ（ADR 0007: IR ではなくランタイム capability 側に置く）
 *
 * - **stride / padding / dilation は H/W の 2 成分・groups はスカラで、4 つとも attrs 宣言必須**
 *   （既定値補完なし — src/ops.ts の CONV2D_ATTRS）。conv1d と同じ規律で、省略を許した
 *   瞬間に「非対称 padding の IR が対称として黙って実行される」経路ができる。
 * - **bias は常時あり（アリティ 3 固定）**。bias 無しの conv はエクスポータの**ゼロ bias
 *   合成**でアリティ 3 へ正規化する（ADR 0015）— カーネルに arity 分岐を持ち込まない。
 *
 * MUST: **重みは `[Cout, Cin/groups, Kh, Kw]`**。取り違え（`[Cin/groups, Cout, …]` /
 * `Kh` と `Kw` の入れ替え）は要素数が合う形が作れて shape 検査を素通りするため、
 * 添字は `oc` → `ic_rel` → `kh` → `kw` の順で組み、テスト側が非対称形
 * （Cin ≠ Cout / Kh ≠ Kw / stride・padding の H≠W）で固定する（conv_transpose1d の教訓）。
 * MUST: grid-stride ループ前提。1 次元の dispatch 上限（仕様既定 65535）は実モデルで超える
 * （VAE decoder は 512px で 1 層 128×512×512 = 3355 万要素）。
 *
 * 縮約は `(ic, kh, kw)` 昇順の逐次で、同じ入力なら常に同じ丸めになる（決定性）。padding 域は
 * 0 詰めなので**加算せずに読み飛ばす**（0 を足すと丸めの並びが変わるうえ、範囲外読みの
 * 分岐が消えて添字が負になる）。
 *
 * groups は conv1d と同じ形（出力チャネル `oc` が属するグループの入力チャネル帯だけを
 * 縮約し、重みの第 2 軸はグループ内の相対番号で引く）。
 *
 * 重みは格納の変種を持つ（`w=f32` / `w=f16` — ADR 0018）。差は縮約内の読み出し 1 行だけ
 * （src/kernels/weight-storage.ts）。
 */

import { CodegenError } from "../codegen/errors.ts";
import { GEMM_MTILE_SMALL, GEMM_TILE, gemmMTileGeometry, gemmWgsl } from "./gemm.ts";
import { gemmTileM, gemmTileN } from "./gemm-geometry.ts";
import {
  WEIGHT_SCALE_VAR,
  weightArrayType,
  weightKeyPart,
  weightLoaderWgsl,
  weightNote,
  weightRead,
  weightScaleWgsl,
  type WeightStorage,
} from "./weight-storage.ts";

export const CONV2D_WORKGROUP_SIZE = 256;

/**
 * i8 変種の scale 束縛（出力の次の番号 — executor の bind entries と対で使う）。
 * 直接カーネルと implicit GEMM は束縛配置が同じなので番号も共通（gemm.ts の
 * `LINEAR_SCALE_BINDING` と同値であることを codegen テストが固定する）。
 */
export const CONV2D_SCALE_BINDING = 5;

/** MUST: WGSL を変えたらキーも上げる（パイプラインキャッシュは本文を見ない）。 */
export const conv2dKey = (weight: WeightStorage): string =>
  `conv2d:v1:f32:direct:wg${CONV2D_WORKGROUP_SIZE}${weightKeyPart(weight)}`;

/**
 * implicit GEMM の v4（vec4 読み書き）判定。
 *
 * MUST: 判定はここ 1 箇所。3 条件はそれぞれ別の理由で必要:
 * - `kFlat % 4 == 0` … A 側の quad 読み（f16 の `dequant4` / i8 の `unpack4xI8` は**平坦添字が
 *   4 の倍数**であることに依存し、行頭 = `arow · kFlat` がその条件を満たす）。
 *   **`Cin % 4` ではない**（Cin=2 の 3×3 は kFlat=18）。
 * - `widthOut % 4 == 0` … B 側の列 quad が**出力行をまたがない**こと。**`N % 4 == 0` では
 *   不十分**（B=1・Hout=Wout=2 は N=4 で N%4==0 だが quad が 2 行に割れる）。実測形は Wout が
 *   全て 4 の倍数なので、この取り違えは**実モデルでは絶対に露見しない** —
 *   tests/gpu_ops_test.ts と tests/gpu_conv2d_parity_test.ts の Hout=Wout=2 ケースが唯一の検出器。
 *   store 側が要求する `n % 4 == 0` はこの条件から従う（n = B·Hout·Wout）。
 * - `strideW === 1` … 4 列の x アドレスが連続すること。
 */
export const conv2dUsesVec4 = (kFlat: number, widthOut: number, strideW: number): boolean =>
  kFlat % 4 === 0 && widthOut % 4 === 0 && strideW === 1;

/**
 * implicit GEMM のパイプラインキー（ADR 0024）。直接カーネルの `v1` とは別系統で、
 * v4 フラグは形状 → 1 ビットの写像（決定性は崩れない — ADR 0022 決定 2 と同じ語彙）。
 *
 * NOTE: 出力は直接カーネルと**ビット同一**（縮約順序が厳密一致）。唯一の例外は符号付きゼロ
 * で、部分和がちょうど `−0.0` の位置に padding 由来の `+0.0` を足すと `+0.0` に転ぶ
 * （直接カーネルは padding を加算しないので `−0.0` が残る）。bias が 0 でない限り到達しない。
 */
export const conv2dIgemmKey = (
  weight: WeightStorage,
  v4: boolean,
  mTile: number = GEMM_TILE,
): string => {
  // MUST: キーの幾何は生成と**同じ解決点**（`gemmMTileGeometry`）から導く。mTile を直に
  // 埋めると、幾何を差し替えたときにキーだけが古い辺を名乗って別物の WGSL へ衝突する。
  const geometry = gemmMTileGeometry(mTile);
  return `conv2d:v2:f32:igemm${gemmTileM(geometry)}x${gemmTileN(geometry)}${
    v4 ? "v4" : ""
  }:wg${geometry.wgX}x${geometry.wgY}${weightKeyPart(weight)}`;
};

export const conv2dIgemmWgsl = (
  weight: WeightStorage,
  v4: boolean,
  mTile: number = GEMM_TILE,
): string => gemmWgsl({ op: "conv2d", v4, weight, mTile });

/**
 * m タイルの行数の選択（ADR 0024 隣接の 32 行変種）。
 *
 * `M % 64` が 1〜32 のときだけ 32 行タイルを選ぶ。この帯だけが「64 行タイルの最後の 1 枚が
 * 半分以下しか埋まらない」形で、32 行 2 枚に割り直すと丸ごと無駄が消える（census の
 * Cout=96 は `2·64/96 = 1.33×` → `3·32/96 = 1.00×`・Cout=3 は 21.33× → 10.67×）。
 *
 * MUST: 境界は `<= 32`。`< 32` にすると M%64 == 32 ちょうど（= census の Cout=96・
 * **1024 系列 MAC の 32.4% を占める本命**）が 64 行のまま残る。`M % 64 == 0` の形
 * （Cout=192/384/1152）は 64 行が無駄ゼロで、32 行に割ると B タイル（x の暗黙 gather）を
 * 2 倍読むだけ損になる。33〜64 の帯も**タイル量子化の無駄は同じ**（どちらも 1 枚 64 行 ぶん）
 * なので、B タイル読みが少ない 64 行を選ぶ。
 */
export const conv2dIgemmMTile = (channelsOut: number): number => {
  const remainder = channelsOut % GEMM_TILE;
  return remainder >= 1 && remainder <= GEMM_MTILE_SMALL ? GEMM_MTILE_SMALL : GEMM_TILE;
};

export const conv2dWgsl = (weight: WeightStorage): string =>
  `// karume conv2d (x[B,Cin,H,W] * W[Cout,Cin/groups,Kh,Kw] + b[Cout], f32${
    weightNote(weight)
  }, 直接畳み込み)
struct Dims {
  n: u32,
  batch: u32,
  channels_in: u32,
  channels_out: u32,
  height_in: u32,
  width_in: u32,
  height_out: u32,
  width_out: u32,
  kernel_h: u32,
  kernel_w: u32,
  stride_h: u32,
  stride_w: u32,
  padding_h: u32,
  padding_w: u32,
  dilation_h: u32,
  dilation_w: u32,
  groups: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<${weightArrayType(weight)}>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
${weightLoaderWgsl("w", weight, CONV2D_SCALE_BINDING)}
@compute @workgroup_size(${CONV2D_WORKGROUP_SIZE})
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let step = nwg.x * ${CONV2D_WORKGROUP_SIZE}u;
  // 契約検査（src/ops.ts）で groups は Cin / Cout を割り切る — 除算は厳密。
  let in_per_group = dims.channels_in / dims.groups;
  let out_per_group = dims.channels_out / dims.groups;
  var i = gid.x;
  while (i < dims.n) {
    let plane = dims.height_out * dims.width_out;
    let chunk = dims.channels_out * plane;
    let b = i / chunk;
    let rest = i % chunk;
    let oc = rest / plane;
    let pixel = rest % plane;
    let oy = pixel / dims.width_out;
    let ox = pixel % dims.width_out;
    // 入力側の開始位置は符号付き（padding ぶん負に出る）
    let origin_y = i32(oy * dims.stride_h) - i32(dims.padding_h);
    let origin_x = i32(ox * dims.stride_w) - i32(dims.padding_w);
    // 重みの第 2 軸は Cin/groups — グループ内の相対番号で引く
    let group_index = oc / out_per_group;
    let ic_base = group_index * in_per_group;${weightScaleWgsl(weight, "oc", "    ")}
    var acc = bias[oc];
    for (var ic_rel = 0u; ic_rel < in_per_group; ic_rel = ic_rel + 1u) {
      let x_base = (b * dims.channels_in + ic_base + ic_rel) * dims.height_in * dims.width_in;
      let w_base = (oc * in_per_group + ic_rel) * dims.kernel_h * dims.kernel_w;
      for (var kh = 0u; kh < dims.kernel_h; kh = kh + 1u) {
        let iy = origin_y + i32(kh * dims.dilation_h);
        // padding 域は 0 詰め — 加算せずに読み飛ばす
        if (iy < 0 || u32(iy) >= dims.height_in) {
          continue;
        }
        let row_base = x_base + u32(iy) * dims.width_in;
        let w_row = w_base + kh * dims.kernel_w;
        for (var kw = 0u; kw < dims.kernel_w; kw = kw + 1u) {
          let ix = origin_x + i32(kw * dims.dilation_w);
          if (ix >= 0 && u32(ix) < dims.width_in) {
            acc = acc + x[row_base + u32(ix)] * ${
    weightRead("w", weight, "w_row + kw", WEIGHT_SCALE_VAR)
  };
          }
        }
      }
    }
    out[i] = acc;
    i = i + step;
  }
}
`;

/** conv2d の幾何（2 つの params 関数が共有する唯一の入力型）。 */
export type Conv2dDims = {
  readonly batch: number;
  readonly channelsIn: number;
  readonly channelsOut: number;
  readonly heightIn: number;
  readonly widthIn: number;
  readonly heightOut: number;
  readonly widthOut: number;
  readonly kernelH: number;
  readonly kernelW: number;
  readonly strideH: number;
  readonly strideW: number;
  readonly paddingH: number;
  readonly paddingW: number;
  readonly dilationH: number;
  readonly dilationW: number;
  readonly groups: number;
};

/**
 * 幾何の契約検査（両カーネル共通）。直接カーネルの Dims 並び順で値を返す。
 *
 * MUST: stride 0 はループが進まず GPU ハング（例外にならない）— 契約検査と二重だが、
 * カーネル直呼びの経路も塞ぐ（conv_transpose1d と同じ二重の門）。
 */
const checkConv2dDims = (dims: Conv2dDims): readonly number[] => {
  const values = [
    dims.batch,
    dims.channelsIn,
    dims.channelsOut,
    dims.heightIn,
    dims.widthIn,
    dims.heightOut,
    dims.widthOut,
    dims.kernelH,
    dims.kernelW,
    dims.strideH,
    dims.strideW,
    dims.paddingH,
    dims.paddingW,
    dims.dilationH,
    dims.dilationW,
    dims.groups,
  ];
  for (const value of values) {
    if (!Number.isSafeInteger(value) || value < 0) {
      throw new CodegenError(`conv2d params: 全ての次元は非負整数（${values.join(", ")}）`);
    }
  }
  const positive: readonly (readonly [string, number])[] = [
    ["stride_h", dims.strideH],
    ["stride_w", dims.strideW],
    ["kernel_h", dims.kernelH],
    ["kernel_w", dims.kernelW],
    ["dilation_h", dims.dilationH],
    ["dilation_w", dims.dilationW],
    ["groups", dims.groups],
  ];
  for (const [name, value] of positive) {
    if (value < 1) throw new CodegenError(`conv2d params: ${name} は正整数（${value}）`);
  }
  // MUST: グループの割り切れは params 層でも見る（シェーダの除算が切り捨てになり、
  // 読む入力チャネル帯が黙ってずれる）。
  if (dims.channelsIn % dims.groups !== 0 || dims.channelsOut % dims.groups !== 0) {
    throw new CodegenError(
      `conv2d params: groups ${dims.groups} が Cin ${dims.channelsIn} / Cout ${dims.channelsOut} を割り切らない`,
    );
  }
  return values;
};

/**
 * 直接カーネルの uniform Dims。17 語なので 16 バイト整列に合わせて 20 語（80 バイト）確保する
 * MUST（uniform アドレス空間の struct 整列。不足すると binding が validation で落ちる）。
 */
export const conv2dParams = (dims: Conv2dDims): Uint32Array<ArrayBuffer> => {
  const values = checkConv2dDims(dims);
  const params = new Uint32Array(20);
  params[0] = dims.batch * dims.channelsOut * dims.heightOut * dims.widthOut;
  values.forEach((value, index) => {
    params[index + 1] = value;
  });
  return params;
};

/**
 * implicit GEMM の uniform Dims（`{m, n, k}` + 幾何 12 語 = 15 語なので 16 語 = 64 バイト確保）。
 *
 * `m = Cout` / `n = Hout·Wout`（**1 バッチぶんの出力平面** — バッチは dispatch の z 軸）/
 * `k = Cin·Kh·Kw`。
 * MUST: 並びは gemm.ts の `CONV2D_DIMS_EXTRA` と対。`groups == 1` 専用で、それ以外は
 * fail loudly（**縮約帯がグループごとに違うので 64 行の m タイルが同じ B タイルを共有できない**
 * — groups > 1 は直接カーネルへ流す）。
 */
export const conv2dIgemmParams = (dims: Conv2dDims): Uint32Array<ArrayBuffer> => {
  checkConv2dDims(dims);
  if (dims.groups !== 1) {
    throw new CodegenError(`conv2d igemm params: groups は 1 専用（${dims.groups}）`);
  }
  const params = new Uint32Array(16);
  params[0] = dims.channelsOut;
  params[1] = dims.heightOut * dims.widthOut;
  params[2] = dims.channelsIn * dims.kernelH * dims.kernelW;
  [
    dims.channelsIn,
    dims.heightIn,
    dims.widthIn,
    dims.widthOut,
    dims.kernelH,
    dims.kernelW,
    dims.strideH,
    dims.strideW,
    dims.paddingH,
    dims.paddingW,
    dims.dilationH,
    dims.dilationW,
  ].forEach((value, index) => {
    params[index + 3] = value;
  });
  return params;
};
