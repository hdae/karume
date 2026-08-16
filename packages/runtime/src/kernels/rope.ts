/**
 * half-split RoPE を 1 pass で適用する f32 カーネル（融合ルール `rope` の実体 —
 * src/runtime/fusion.ts）。
 *
 * 入力 `x` は `[B,H,S,D]`、cos / sin は `[1,1,S,D]`。最後の次元を半分に分け、
 * `rotate_half(x) = cat(-x[..., D/2:], x[..., :D/2])` として
 * `x * cos + rotate_half(x) * sin` を出力する。
 *
 * MUST: cos / sin はホストが torch 由来の素表から組んだ f32 値をそのまま読む。カーネル内で
 * 三角関数や token の H/W 座標を再計算しない（非正方形でも token 順だけが契約）。
 * MUST: 2 本の積を 2KiB の workgroup `u32` 配列へ書き、barrier 後に加算へ戻す。素の
 * `x*cos + rot*sin` と書くと、置換元の primitive 列（mul → mul → add の 3 dispatch）に
 * あった f32 の丸め 2 回が縮約されうる。
 * **この「丸め障壁」は WGSL 仕様の保証ではない** — 仕様は浮動小数の fusion（fma への縮約）を
 * 許すだけで禁じてはおらず、workgroup memory 往復を境界として尊重するかは実装依存の実測事実。
 * したがって**バックエンド更新で PNG sha256 門が割れたら、まずここを疑うこと**（融合を外して
 * primitive 列に戻せば切り分けられる）。
 * MUST: grid-stride で出力の全要素を書く。barrier は workgroup 一様な loop 内だけ。optional
 * feature / subgroup / atomics は使わない。
 */

import { CodegenError } from "../codegen/errors.ts";
import { assertU32Params } from "../codegen/params.ts";

export const ROPE_WORKGROUP_SIZE = 256;

/** MUST: WGSL を変えたらキーも上げる（パイプラインキャッシュは本文を見ない）。 */
export const ROPE_KEY = `rope:v1:half:f32:wg${ROPE_WORKGROUP_SIZE}`;

export const ROPE_WGSL: string = `// karume RoPE (half split, f32, full-write)
struct Params {
  n: u32,
  sequence: u32,
  head_dim: u32,
  half_dim: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> cos_table: array<f32>;
@group(0) @binding(3) var<storage, read> sin_table: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
var<workgroup> products: array<vec2<u32>, ${ROPE_WORKGROUP_SIZE}>;

@compute @workgroup_size(${ROPE_WORKGROUP_SIZE})
fn main(
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let stride = nwg.x * ${ROPE_WORKGROUP_SIZE}u;
  var base = wid.x * ${ROPE_WORKGROUP_SIZE}u;
  while (base < params.n) {
    let i = base + lid;
    if (i < params.n) {
      let d = i % params.head_dim;
      let row = i / params.head_dim;
      let token = row % params.sequence;
      let row_base = i - d;
      var rotated_bits: u32;
      if (d < params.half_dim) {
        rotated_bits = bitcast<u32>(x[row_base + d + params.half_dim]) ^ 0x80000000u;
      } else {
        rotated_bits = bitcast<u32>(x[row_base + d - params.half_dim]);
      }
      let table_index = token * params.head_dim + d;
      let direct_bits = bitcast<u32>(x[i] * cos_table[table_index]);
      let cross_bits = bitcast<u32>(bitcast<f32>(rotated_bits) * sin_table[table_index]);
      products[lid] = vec2<u32>(direct_bits, cross_bits);
    } else {
      products[lid] = vec2<u32>(0u);
    }
    workgroupBarrier();
    if (i < params.n) {
      let pair = products[lid];
      out[i] = bitcast<f32>(pair.x) + bitcast<f32>(pair.y);
    }
    workgroupBarrier();
    base = base + stride;
  }
}
`;

/** 16-byte uniform params。`n` は `[B,H,S,D]` の全要素数。 */
export const ropeParams = (
  n: number,
  sequence: number,
  headDim: number,
): Uint32Array<ArrayBuffer> => {
  assertU32Params("rope params", { n, sequence, headDim });
  if (sequence === 0) throw new CodegenError("rope params: sequence は 1 以上");
  if (headDim === 0 || headDim % 2 !== 0) {
    throw new CodegenError(`rope params: headDim は正の偶数（${headDim}）`);
  }
  const rowSize = sequence * headDim;
  assertU32Params("rope params", { "sequence × headDim": rowSize });
  if (n % rowSize !== 0) {
    throw new CodegenError(
      `rope params: n ${n} が sequence ${sequence} × headDim ${headDim} の整数行でない`,
    );
  }
  return new Uint32Array([n, sequence, headDim, headDim / 2]);
};
