/**
 * conv1d（`x[B,Cin,L] * W[Cout,Cin/groups,K] + b[Cout]`、f32）の固定カーネル。直接畳み込みで
 * 1 スレッド = 1 出力要素（正しさ優先 — 性能マイルストーンでの置換対象）。
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

/**
 * uniform の Dims。11 語なので 16 バイト整列に合わせて 12 語（48 バイト）確保する MUST
 * （uniform アドレス空間の struct 整列。不足すると binding が validation で落ちる）。
 */
export const conv1dParams = (dims: {
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
}): Uint32Array<ArrayBuffer> => {
  const values = [
    dims.batch,
    dims.channelsIn,
    dims.channelsOut,
    dims.lengthIn,
    dims.lengthOut,
    dims.kernel,
    dims.stride,
    dims.padding,
    dims.dilation,
    dims.groups,
  ];
  for (const value of values) {
    if (!Number.isSafeInteger(value) || value < 0) {
      throw new CodegenError(`conv1d params: 全ての次元は非負整数（${values.join(", ")}）`);
    }
  }
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
  const params = new Uint32Array(12);
  params[0] = dims.batch * dims.channelsOut * dims.lengthOut;
  params[1] = dims.batch;
  params[2] = dims.channelsIn;
  params[3] = dims.channelsOut;
  params[4] = dims.lengthIn;
  params[5] = dims.lengthOut;
  params[6] = dims.kernel;
  params[7] = dims.stride;
  params[8] = dims.padding;
  params[9] = dims.dilation;
  params[10] = dims.groups;
  return params;
};
