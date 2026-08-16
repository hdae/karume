/**
 * conv1d（`x[B,Cin,L] * W[Cout,Cin/groups,K] + b[Cout]`、f32）の 2 カーネル。
 *
 * | 経路          | キー                                                    | 条件        |
 * | ------------- | ------------------------------------------------------- | ----------- |
 * | implicit GEMM | `conv1d:v3:f32:igemm{tileM}x{tileN}{v4}:wg{x}x{y}{:w…}`  | groups == 1 |
 * | 直接畳み込み  | `conv1d:v2:f32:direct:wg256{:w…}`                       | groups > 1  |
 *
 * implicit GEMM のキーの辺と workgroup 形は**幾何から導く**（{@link conv1dIgemmKey} —
 * conv2d と同じ規律で、具体値を書き写すと幾何を差し替えたとき doc とキーだけが古い辺を名乗る）。
 *
 * implicit GEMM（ADR 0024 の 1D 版）は `C[Cout, N] = W[Cout, K] × Xcol[K, N]` を GEMM 骨格
 * （src/kernels/gemm.ts）の断片共有で解く。**縮約順序が直接畳み込みと厳密に一致する**ので
 * 出力はビット同一（唯一の例外 = 符号付きゼロ — {@link conv1dIgemmKey} の doc）。
 * 直接畳み込み（ADR 0017）は 1 スレッド = 1 出力要素で、groups > 1 を受け持つと同時に
 * **恒久の差分オラクル**（tests/gpu_conv1d_parity_test.ts）を兼ねる。
 *
 * ## 契約の狭さ（ADR 0007: IR ではなくランタイム capability 側に置く）
 *
 * - **groups / dilation は attrs（ADR 0015）**。従来の「欄が無い = 1 固定」で担保していた
 *   「1 以外を黙って 1 で実行する経路が無い」性質は、欄を作った後は**既定値補完をしない**
 *   ことだけが担保する（src/ops.ts の CONV1D_ATTRS）。
 * - **bias は常時あり（アリティ 3 固定）**。bias 無しの conv はエクスポータが**ゼロ bias を
 *   合成**してアリティ 3 へ正規化する（ADR 0015）— カーネルに arity 分岐を持ち込まない。
 * - `B` / `Cin` / `Cout` / `K` / stride / padding / dilation / groups は params で運ぶ
 *   **通常のループ境界**なので、実測値へ固定はしない。固定しても実行できる形が減るだけで、
 *   誤りを検出する力は増えない（誤りが出るのは軸の取り違えで、それは非対称な shape の
 *   テストが捕まえる）。
 *
 * MUST: grid-stride ループ前提。1 次元の dispatch 上限（仕様既定 65535）は実モデルで超える
 * （実測は出力 1024×T 要素）。
 *
 * 縮約は `(ic, k)` 昇順の逐次で、同じ入力なら常に同じ丸めになる（決定性）。padding 域は
 * 0 詰めなので**加算せずに読み飛ばす**（0 を足すと丸めの並びが変わるうえ、範囲外読みの
 * 分岐が消えて添字が負になる）。
 *
 * groups は「出力チャネル oc が属するグループ g = oc / (Cout/groups) の入力チャネル帯
 * `[g·Cin/groups, (g+1)·Cin/groups)` だけを縮約する」形で載せる。重みの第 2 軸は
 * **Cin ではなく Cin/groups** なので、w の添字も帯の中の相対番号で組む（ここを Cin のまま
 * にすると depthwise で重みを読み飛ばす沈黙誤値になる）。
 *
 * 重みは格納の変種を持つ（`w=f32` / `w=f16` / `w=i8` — ADR 0018 / 0019）。差は縮約内の
 * 読み出し 1 行と、i8 の per-channel scale（`oc` 単位でループ不変なので縮約の外へ巻き上げる）
 * だけ（src/kernels/weight-storage.ts）。
 */

import { CodegenError } from "../codegen/errors.ts";
import { gemmMTileGeometry, gemmWgsl } from "./gemm.ts";
import { GEMM_TILE, gemmTileM, gemmTileN } from "./gemm-geometry.ts";
import { assertU32Params } from "./params.ts";
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

export const CONV1D_WORKGROUP_SIZE = 256;

/** i8 変種の scale 束縛（出力の次の番号 — executor の bind entries と対で使う）。 */
export const CONV1D_SCALE_BINDING = 5;

/** MUST: WGSL を変えたらキーも上げる（パイプラインキャッシュは本文を見ない）。 */
export const conv1dKey = (weight: WeightStorage): string =>
  `conv1d:v2:f32:direct:wg${CONV1D_WORKGROUP_SIZE}${weightKeyPart(weight)}`;

/**
 * implicit GEMM の v4（vec4 読み書き）判定。
 *
 * MUST: 判定はここ 1 箇所。3 つの条件は 2 つの理由に対応する:
 * - `kFlat % 4 == 0` … A 側の quad 読み（f16 の `dequant4` / i8 の `unpack4xI8` は**平坦添字が
 *   4 の倍数**であることに依存し、行頭 = `arow · kFlat` がその条件を満たす）。
 *   **`Cin % 4` ではない**（Cin=1 の K=7 は kFlat=7）。
 * - `lengthOut % 4 == 0 && stride == 1` … B 側の列 quad が x の連続 4 要素に落ちること、
 *   および store 側の quad 書きに端数が出ないこと。1D では出力平面が 1 行なので
 *   `n = Lout` で、conv2d が `Wout % 4` と `N % 4` を区別した理由（quad が出力行をまたぐ）は
 *   ここでは消える。
 */
export const conv1dUsesVec4 = (kFlat: number, lengthOut: number, stride: number): boolean =>
  kFlat % 4 === 0 && lengthOut % 4 === 0 && stride === 1;

/**
 * implicit GEMM のパイプラインキー（ADR 0024 の 1D 版）。直接カーネルの `v2` とは別系統で、
 * v4 フラグは形状 → 1 ビットの写像（決定性は崩れない — ADR 0022 決定 2 と同じ語彙）。
 *
 * NOTE: 出力は直接カーネルと**ビット同一**（縮約順序が厳密一致）。唯一の例外は符号付きゼロ
 * で、部分和がちょうど `−0.0` の位置に padding 由来の `+0.0` を足すと `+0.0` に転ぶ
 * （直接カーネルは padding を加算しないので `−0.0` が残る）。bias が 0 でない限り到達しない
 * （機序の固定は tests/gpu_conv1d_parity_test.ts の負ケース）。
 *
 * NOTE: m タイルの選択述語は `conv2dIgemmMTile`（src/kernels/conv2d.ts）を共有する。
 * **M = Cout の関数でしかない**（無駄 = `ceil(M/tile)·tile / M`）ので次元に依らず、
 * 1D 用に写すと同じ境界が 2 箇所に散る。
 */
export const conv1dIgemmKey = (
  weight: WeightStorage,
  v4: boolean,
  mTile: number = GEMM_TILE,
): string => {
  // MUST: キーの幾何は生成と**同じ解決点**（`gemmMTileGeometry`）から導く。mTile を直に
  // 埋めると、幾何を差し替えたときにキーだけが古い辺を名乗って別物の WGSL へ衝突する。
  const geometry = gemmMTileGeometry(mTile);
  return `conv1d:v3:f32:igemm${gemmTileM(geometry)}x${gemmTileN(geometry)}${
    v4 ? "v4" : ""
  }:wg${geometry.wgX}x${geometry.wgY}${weightKeyPart(weight)}`;
};

export const conv1dIgemmWgsl = (
  weight: WeightStorage,
  v4: boolean,
  mTile: number = GEMM_TILE,
): string => gemmWgsl({ op: "conv1d", v4, weight, mTile });

export const conv1dWgsl = (weight: WeightStorage): string =>
  `// karume conv1d (x[B,Cin,L] * W[Cout,Cin/groups,K] + b[Cout], f32${
    weightNote(weight)
  }, 直接畳み込み)
struct Dims {
  n: u32,
  batch: u32,
  channels_in: u32,
  channels_out: u32,
  length_in: u32,
  length_out: u32,
  kernel: u32,
  stride: u32,
  padding: u32,
  dilation: u32,
  groups: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<${weightArrayType(weight)}>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
${weightLoaderWgsl("w", weight, CONV1D_SCALE_BINDING)}
@compute @workgroup_size(${CONV1D_WORKGROUP_SIZE})
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let step = nwg.x * ${CONV1D_WORKGROUP_SIZE}u;
  // 契約検査（src/ops.ts）で groups は Cin / Cout を割り切る — 除算は厳密。
  let in_per_group = dims.channels_in / dims.groups;
  let out_per_group = dims.channels_out / dims.groups;
  var i = gid.x;
  while (i < dims.n) {
    let plane = dims.channels_out * dims.length_out;
    let b = i / plane;
    let rest = i % plane;
    let oc = rest / dims.length_out;
    let ox = rest % dims.length_out;
    // 入力側の開始位置は符号付き（padding ぶん負に出る）
    let origin = i32(ox * dims.stride) - i32(dims.padding);
    // 重みの第 2 軸は Cin/groups — グループ内の相対番号で引く
    let group_index = oc / out_per_group;
    let ic_base = group_index * in_per_group;${weightScaleWgsl(weight, "oc", "    ")}
    var acc = bias[oc];
    for (var ic_rel = 0u; ic_rel < in_per_group; ic_rel = ic_rel + 1u) {
      let x_base = (b * dims.channels_in + ic_base + ic_rel) * dims.length_in;
      let w_base = (oc * in_per_group + ic_rel) * dims.kernel;
      for (var k = 0u; k < dims.kernel; k = k + 1u) {
        let ix = origin + i32(k * dims.dilation);
        // padding 域は 0 詰め — 加算せずに読み飛ばす
        if (ix >= 0 && u32(ix) < dims.length_in) {
          acc = acc + x[x_base + u32(ix)] * ${
    weightRead("w", weight, "w_base + k", WEIGHT_SCALE_VAR)
  };
        }
      }
    }
    out[i] = acc;
    i = i + step;
  }
}
`;

/** conv1d の幾何（2 つの params 関数が共有する唯一の入力型）。 */
export type Conv1dDims = {
  readonly batch: number;
  readonly channelsIn: number;
  readonly channelsOut: number;
  readonly lengthIn: number;
  readonly lengthOut: number;
  readonly kernel: number;
  readonly stride: number;
  readonly padding: number;
  readonly dilation: number;
  readonly groups: number;
};

/**
 * 幾何の契約検査（両カーネル共通）。直接カーネルの Dims 並び順で値を返す。
 *
 * MUST: stride 0 はループが進まず GPU ハング（例外にならない）— 契約検査と二重だが、
 * カーネル直呼びの経路も塞ぐ。
 */
const checkConv1dDims = (dims: Conv1dDims): readonly number[] => {
  // 名前は WGSL の Dims 欄名（`n` の次から）と対。並びがそのまま uniform の語順になる。
  const dimensions = {
    batch: dims.batch,
    channels_in: dims.channelsIn,
    channels_out: dims.channelsOut,
    length_in: dims.lengthIn,
    length_out: dims.lengthOut,
    kernel: dims.kernel,
    stride: dims.stride,
    padding: dims.padding,
    dilation: dims.dilation,
    groups: dims.groups,
  };
  assertU32Params("conv1d params", dimensions);
  const values = Object.values(dimensions);
  if (dims.stride < 1 || dims.kernel < 1 || dims.dilation < 1 || dims.groups < 1) {
    throw new CodegenError(
      `conv1d params: stride / kernel / dilation / groups は正整数（${dims.stride}, ${dims.kernel}, ${dims.dilation}, ${dims.groups}）`,
    );
  }
  // MUST: グループの割り切れは params 層でも見る（シェーダの除算が切り捨てになり、
  // 読む入力チャネル帯が黙ってずれる — 契約検査と二重だが、カーネル直呼びの経路も塞ぐ）。
  if (dims.channelsIn % dims.groups !== 0 || dims.channelsOut % dims.groups !== 0) {
    throw new CodegenError(
      `conv1d params: groups ${dims.groups} が Cin ${dims.channelsIn} / Cout ${dims.channelsOut} を割り切らない`,
    );
  }
  return values;
};

/**
 * 直接カーネルの uniform Dims。11 語なので 16 バイト整列に合わせて 12 語（48 バイト）確保する
 * MUST（uniform アドレス空間の struct 整列。不足すると binding が validation で落ちる）。
 */
export const conv1dParams = (dims: Conv1dDims): Uint32Array<ArrayBuffer> => {
  const values = checkConv1dDims(dims);
  const params = new Uint32Array(12);
  params[0] = dims.batch * dims.channelsOut * dims.lengthOut;
  values.forEach((value, index) => {
    params[index + 1] = value;
  });
  return params;
};

/**
 * implicit GEMM の uniform Dims（`{m, n, k}` + 幾何 6 語 = 9 語なので 12 語 = 48 バイト確保）。
 *
 * `m = Cout` / `n = Lout`（**1 バッチぶんの出力平面** — バッチは dispatch の z 軸）/
 * `k = Cin·K`。
 * MUST: 並びは gemm.ts の `CONV1D_DIMS_EXTRA` と対。`groups == 1` 専用で、それ以外は
 * fail loudly（**縮約帯がグループごとに違うので 1 枚の m タイルが同じ B タイルを共有できない**
 * — groups > 1 は直接カーネルへ流す）。
 */
export const conv1dIgemmParams = (dims: Conv1dDims): Uint32Array<ArrayBuffer> => {
  checkConv1dDims(dims);
  if (dims.groups !== 1) {
    throw new CodegenError(`conv1d igemm params: groups は 1 専用（${dims.groups}）`);
  }
  const params = new Uint32Array(12);
  params[0] = dims.channelsOut;
  params[1] = dims.lengthOut;
  params[2] = dims.channelsIn * dims.kernel;
  [
    dims.channelsIn,
    dims.lengthIn,
    dims.kernel,
    dims.stride,
    dims.padding,
    dims.dilation,
  ].forEach((value, index) => {
    params[index + 3] = value;
  });
  return params;
};
