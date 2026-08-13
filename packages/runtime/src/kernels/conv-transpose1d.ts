/**
 * conv_transpose1d（`x[B,Cin,L] ⊛ᵀ W[Cin,Cout,K] + b[Cout]`、f32）の固定カーネル。
 * 1 スレッド = 1 出力要素の **gather 形**（full-write 不変条件の側の要請 — 下記）。縮約の内側は
 * **residue grouping**（perf-ledger K-10 — 有効 tap だけを O(1) の範囲で回る）。
 *
 * ## gather 形である理由（full-write 不変条件 — ADR 0014）
 *
 * 転置畳み込みの素直な実装は「入力 1 点を出力の複数点へ**散らす**」scatter で、出力バッファを
 * ゼロ初期化してから read-modify-write する形になる。Karume のバッファはプール再利用
 * （ADR 0004）でゼロ初期化されないうえ、複数スレッドの加算が競合する。ここは
 * **出力 1 要素 = 寄与する入力の総和**へ裏返して、1 スレッドが 1 出力要素を**必ず 1 回書く**
 * 形にする。
 *
 * 出力座標 `ox` に寄与するのは `ox = ix·stride − padding + k` を満たす `(ix, k)` の組で、
 * `t = ox + padding − k` が **stride で割り切れて** `0 <= t/stride < L` のときだけ。
 * 割り切れない `k` は「そこに入力が無い」＝寄与ゼロなので、padding 域と同じく
 * **加算に現れない**（0 を足すと丸めの並びが変わる）。
 *
 * ## residue grouping（perf-ledger K-10）
 *
 * 割り切れ判定は `k ≡ shifted (mod stride)`（`shifted = ox + padding`）と同値なので、
 * `shifted = q·stride + r` へ**出力要素あたり 1 回だけ**分解すれば、有効 tap は
 * `k = r + j·stride` / `ix = q − j` と**数え上げ**で書ける。`j` の範囲は 3 本の不等式
 * （`ix <= L−1` / `ix >= 0` / `k <= K−1`）で O(1) に閉じるので、`k` 全数の走査と剰余判定が
 * 消える（dacvae の実測形 stride 8 / K 16 では tap の 7/8 が無効判定だった）。
 *
 * ## 契約の狭さ
 *
 * - 重みは **`[Cin, Cout, K]`**（conv1d の `[Cout, Cin, K]` と転置）。取り違えても要素数が
 *   合う形（Cin == Cout）が作れて shape 検査を素通りするため、テストは**非対称チャネル数**で
 *   固定する（ADR 0015 / recon §4）。
 * - **stride >= 1 MUST**。プロトタイプ実装は stride 0 でループが進まず **GPU ハング**（例外に
 *   ならない）だった。gather 形の本カーネルはループ境界に stride を使わないが、residue 分割の
 *   `shifted % stride` が要素あたり 1 回ゼロ除算になるので {@link convTranspose1dParams} で
 *   遮断する（契約検査と二重の門）。
 * - bias は常時あり（アリティ 3 固定）。bias 無しはエクスポータのゼロ bias 合成で正規化される。
 * - `output_padding` / `dilation` / `groups` は attrs に無い = 実測どおりの 0 / 1 / 1 固定。
 *
 * MUST: grid-stride ループ前提（dec の ups は出力 512×T·8 要素まで伸びる）。
 * 縮約は `(ic, k)` 昇順の逐次で、同じ入力なら常に同じ丸めになる（決定性）。
 *
 * 重みは格納の変種を持つ（`w=f32` / `w=f16` / `w=i8` — ADR 0018 / 0019）。差は縮約内の
 * 読み出し 1 行と、i8 の per-channel scale だけ（src/kernels/weight-storage.ts）。
 * MUST: i8 の scale の**チャネル軸は 1**（重み `[Cin, Cout, K]` の第 2 軸 = 出力チャネル）。
 * 他の 4 カーネルは軸 0 で、ここだけ転置レイアウトのぶんずれる。
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

export const CONV_TRANSPOSE1D_WORKGROUP_SIZE = 256;

/** i8 変種の scale 束縛（出力の次の番号 — executor の bind entries と対で使う）。 */
export const CONV_TRANSPOSE1D_SCALE_BINDING = 5;

export const convTranspose1dKey = (weight: WeightStorage): string =>
  `conv_transpose1d:v2:f32:gather:wg${CONV_TRANSPOSE1D_WORKGROUP_SIZE}${weightKeyPart(weight)}`;

export const convTranspose1dWgsl = (weight: WeightStorage): string =>
  `// karume conv_transpose1d (x[B,Cin,L] * W[Cin,Cout,K] + b[Cout], f32${
    weightNote(weight)
  }, gather 形)
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
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<${weightArrayType(weight)}>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
${weightLoaderWgsl("w", weight, CONV_TRANSPOSE1D_SCALE_BINDING)}
@compute @workgroup_size(${CONV_TRANSPOSE1D_WORKGROUP_SIZE})
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let step = nwg.x * ${CONV_TRANSPOSE1D_WORKGROUP_SIZE}u;
  var i = gid.x;
  while (i < dims.n) {
    let plane = dims.channels_out * dims.length_out;
    let b = i / plane;
    let rest = i % plane;
    let oc = rest / dims.length_out;
    let ox = rest % dims.length_out;
    // ox = ix*stride - padding + k  =>  ix*stride = ox + padding - k
    // 有効 tap は k ≡ shifted (mod stride) だけ — 出力要素あたり 1 回だけ商と剰余へ分解する
    let shifted = ox + dims.padding;
    let q = shifted / dims.stride;
    let r = shifted % dims.stride;
    // 有効 tap は k = r + j*stride / ix = q - j。j の範囲は 3 本の不等式で閉じる:
    // ix <= L-1 → j >= q+1-L / ix >= 0 → j <= q / k <= K-1 → j <= (K-1-r)/stride
    let j_start = max(0, i32(q) + 1 - i32(dims.length_in));
    // r > K-1（stride > K のときだけ到達）は有効 tap ゼロ。K-1-r の u32 桁借りを避けて
    // j_end = -1 < j_start（j_start >= 0）で空ループへ倒す
    var j_end = -1;
    if (r < dims.kernel) {
      j_end = min(i32((dims.kernel - 1u - r) / dims.stride), i32(q));
    }${weightScaleWgsl(weight, "oc", "    ")}
    var acc = bias[oc];
    for (var ic = 0u; ic < dims.channels_in; ic = ic + 1u) {
      let x_base = (b * dims.channels_in + ic) * dims.length_in;
      // 重みは [Cin, Cout, K] — conv1d と転置（第 1 軸が入力チャネル）
      let w_base = (ic * dims.channels_out + oc) * dims.kernel;
      // MUST: j 昇順 = k 昇順。tap 集合も (ic, k) 昇順の縮約順序も k 全数走査版と同一で、
      // 同じ被演算子に同じ f32 積和が同じ順で掛かる = ビット同一（並べ替えは丸めを変える）
      for (var j = j_start; j <= j_end; j = j + 1) {
        let ix = q - u32(j);
        let k = r + u32(j) * dims.stride;
        acc = acc + x[x_base + ix] * ${weightRead("w", weight, "w_base + k", WEIGHT_SCALE_VAR)};
      }
    }
    out[i] = acc;
    i = i + step;
  }
}
`;

/**
 * uniform の Dims。9 語なので 16 バイト整列に合わせて 12 語（48 バイト）確保する MUST
 * （uniform アドレス空間の struct 整列。不足すると binding が validation で落ちる）。
 */
export const convTranspose1dParams = (dims: {
  readonly batch: number;
  readonly channelsIn: number;
  readonly channelsOut: number;
  readonly lengthIn: number;
  readonly lengthOut: number;
  readonly kernel: number;
  readonly stride: number;
  readonly padding: number;
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
  ];
  for (const value of values) {
    if (!Number.isSafeInteger(value) || value < 0) {
      throw new CodegenError(
        `conv_transpose1d params: 全ての次元は非負整数（${values.join(", ")}）`,
      );
    }
  }
  // MUST: stride 0 は WGSL の `% dims.stride` がゼロ除算になる（結果は未定義で、実装に
  // よっては例外にならないまま沈黙誤値・ハングになる — recon §4 の前例）。
  if (dims.stride < 1 || dims.kernel < 1) {
    throw new CodegenError(
      `conv_transpose1d params: stride / kernel は正整数（${dims.stride}, ${dims.kernel}）`,
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
  return params;
};
