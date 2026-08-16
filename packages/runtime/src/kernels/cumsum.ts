/**
 * cumsum（最終次元の前縁和 `out[…, j] = Σ_{i ≤ j} x[…, i]`、f32）の固定カーネル。
 *
 * 入力は先行次元を平坦化した `[rows, dim]` の連続レイアウトで、出力も同形・連続。
 *
 * **1 invocation = 1 行の逐次走査**にしている（行 reduce の「1 workgroup = 1 行 + 木状の
 * 縮約」とは別の形）。理由は実測の行長: spline の cumwidths / cumheights は bins ≈ 10 で
 * （recon §2）、Hillis–Steele や Blelloch の並列スキャンを持ち込むと 1 行あたり
 * log2(dim) 回の workgroupBarrier を払って 10 要素を畳むことになり、逐次より遅いうえに
 * 実装の複雑さ（バリアの一様性・行の跨ぎ）だけが増える。行数の方が並列度の源なので、
 * **行方向を grid-stride** で回す。
 *
 * MUST: 行長が伸びる需要（数千要素の行）が実測に出たら、この選択は見直すこと。逐次走査は
 * 1 行あたり O(dim) を 1 スレッドで払う形で、dim が大きいと並列度が行数だけになる。
 * MUST: grid-stride ループ前提。1 次元の dispatch 上限（仕様既定 65535）は実モデルで超える。
 * MUST: 累積は**前から**（`out[j]` は `x[0..j]` の和）。逆向きに書いても shape も要素数も
 * 変わらないため、方向の取り違えは値でしか検出できない（tests/gpu_ops_test.ts が
 * 非対称な列で固定する）。
 */

import { assertU32Params } from "../codegen/params.ts";

export const CUMSUM_WORKGROUP_SIZE = 256;

export const CUMSUM_KEY = `cumsum:v1:f32:wg${CUMSUM_WORKGROUP_SIZE}`;

export const CUMSUM_WGSL: string = [
  "// karume cumsum (last dim, f32, sequential per row)",
  "struct Params {",
  "  rows: u32,",
  "  dim: u32,",
  "}",
  "@group(0) @binding(0) var<uniform> params: Params;",
  "@group(0) @binding(1) var<storage, read> x: array<f32>;",
  "@group(0) @binding(2) var<storage, read_write> out: array<f32>;",
  "",
  `@compute @workgroup_size(${CUMSUM_WORKGROUP_SIZE})`,
  "fn main(",
  "  @builtin(global_invocation_id) gid: vec3<u32>,",
  "  @builtin(num_workgroups) nwg: vec3<u32>,",
  ") {",
  `  let stride = nwg.x * ${CUMSUM_WORKGROUP_SIZE}u;`,
  "  var row = gid.x;",
  "  while (row < params.rows) {",
  "    let base = row * params.dim;",
  "    var acc = 0.0;",
  "    var j = 0u;",
  "    while (j < params.dim) {",
  "      acc = acc + x[base + j];",
  "      out[base + j] = acc;",
  "      j = j + 1u;",
  "    }",
  "    row = row + stride;",
  "  }",
  "}",
  "",
].join("\n");

/**
 * uniform の Params（行 op 族と同じ 2 語）。WGSL の uniform アドレス空間では struct の整列が
 * 16 バイトになるため、2 語ぶんの内容でも 16 バイト確保する MUST。
 */
export const cumsumParams = (rows: number, dim: number): Uint32Array<ArrayBuffer> => {
  assertU32Params("cumsum params", { rows, dim });
  const params = new Uint32Array(4);
  params[0] = rows;
  params[1] = dim;
  return params;
};
