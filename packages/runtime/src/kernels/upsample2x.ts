/**
 * NCHW の最終 2 空間軸を nearest-neighbor で 2 倍にする f32 固定カーネル（融合ルール
 * `upsample2x` の実体 — src/runtime/fusion.ts）。
 *
 * 入力を連続 `[N*C*H, W]` として平坦に読み、1 invocation が入力 1 要素を出力の 2×2 block へ
 * 書く。入力要素ごとに出力先が重ならないので atomics も workgroup memory も要らない。f32 を
 * 演算せず `u32` のビット view で複製するため、NaN payload / subnormal / ±0 もバックエンドの
 * 浮動小数正規化を受けない（丸め障壁の議論が要らないのはこのルールだけ）。出力の全要素は
 * ちょうど 1 回書かれる（full-write 不変条件 — ADR 0014）。
 *
 * MUST: grid-stride を使う。NCHW の batch / channel / height は row に畳み、width だけを
 * params に残すので rank 固有の座標復元を持ち込まない。
 */

import { CodegenError } from "../codegen/errors.ts";
import { assertU32Params } from "../codegen/params.ts";

export const UPSAMPLE_2X_WORKGROUP_SIZE = 256;

/** MUST: WGSL を変えたらキーも上げる（パイプラインキャッシュは本文を見ない）。 */
export const UPSAMPLE_2X_KEY = `upsample2x:v1:nchw:f32:wg${UPSAMPLE_2X_WORKGROUP_SIZE}`;

export const UPSAMPLE_2X_WGSL: string = `// karume upsample2x (NCHW nearest, f32 bits, full-write)
struct Params {
  n: u32,
  width: u32,
  out_width: u32,
  reserved: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;

@compute @workgroup_size(${UPSAMPLE_2X_WORKGROUP_SIZE})
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let stride = nwg.x * ${UPSAMPLE_2X_WORKGROUP_SIZE}u;
  var i = gid.x;
  while (i < params.n) {
    let col = i % params.width;
    let row = i / params.width;
    let base = 2u * row * params.out_width + 2u * col;
    let value = x[i];
    out[base] = value;
    out[base + 1u] = value;
    out[base + params.out_width] = value;
    out[base + params.out_width + 1u] = value;
    i = i + stride;
  }
}
`;

/**
 * 16-byte uniform params。`n` は入力の全要素数、`width` は入力 NCHW の最終次元。
 *
 * MUST: `n % width` をここで拒否する。割れない値を通すと、最後の不完全な行の 2×2 書き込みが
 * 次行と重なり、全要素数だけでは検出できない沈黙誤値になる。
 */
export const upsample2xParams = (n: number, width: number): Uint32Array<ArrayBuffer> => {
  assertU32Params("upsample2x params", { n, width });
  if (width === 0) throw new CodegenError("upsample2x params: width は 1 以上");
  if (n % width !== 0) {
    throw new CodegenError(`upsample2x params: n ${n} が width ${width} の整数行でない`);
  }
  const outWidth = width * 2;
  const outN = n * 4;
  assertU32Params("upsample2x params", { "2 * width": outWidth, "4 * n": outN });
  return new Uint32Array([n, width, outWidth, 0]);
};
