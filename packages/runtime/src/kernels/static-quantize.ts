import { CodegenError } from "../codegen/errors.ts";
import { assertU32Params } from "../codegen/params.ts";

/** f32 の除算 → 偶数丸め → [-128,127] 飽和 → f32 乗算を整数表で再現する。 */
export const staticQuantizeParams = (count: number, scale: number): Uint32Array<ArrayBuffer> => {
  assertU32Params("static_quantize params", { count });
  if (!Number.isFinite(scale) || scale < 0 || Math.fround(scale) !== scale) {
    throw new CodegenError("static_quantize: scale は非負・有限で厳密に f32 表現できる値が必要");
  }
  // 16 byte 整列。count / 境界 128 語 / 出力 129 語 / 恒等フラグ / padding。
  const params = new Uint32Array(260);
  const scratch = new Float32Array(1);
  const bits = new Uint32Array(scratch.buffer);
  params[0] = count;
  params[258] = scale === 0 ? 1 : 0;
  if (scale === 0) return params;
  for (let j = 0; j < 128; j++) {
    const half = j + 0.5;
    const error = 2 ** (Math.floor(Math.log2(half)) - 24);
    const strict = j % 2 === 0;
    // f32 除算が half へ丸まる区間を含める。積は f64 で厳密（最大49bit）。
    const cut = (half + (strict ? error : -error)) * scale;
    scratch[0] = cut;
    if (scratch[0] < cut || (strict && scratch[0] === cut)) bits[0]++;
    params[1 + j] = bits[0];
  }
  for (let j = 0; j <= 128; j++) {
    scratch[0] = j * scale;
    params[129 + j] = bits[0];
  }
  return params;
};

export const STATIC_QUANTIZE_WORKGROUP_SIZE = 128;
export const STATIC_QUANTIZE_KEY = "static_quantize:v1:f32:wg128";
export const STATIC_QUANTIZE_WGSL: string =
  `// karume static_quantize: 固定 f32 scale の SRQ（ADR 0097）
@group(0) @binding(0) var<uniform> params: array<vec4<u32>, 65>;
@group(0) @binding(1) var<storage, read> input: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
fn word(index: u32) -> u32 { return params[index >> 2u][index & 3u]; }
fn quantize(bits: u32) -> u32 {
  if (word(258u) != 0u) { return bits; }
  let magnitude = bits & 0x7fffffffu;
  if (magnitude > 0x7f800000u) { return bits | 0x00400000u; }
  var lo = 0u;
  var hi = 128u;
  while (lo < hi) {
    let mid = (lo + hi) >> 1u;
    if (magnitude >= word(1u + mid)) { lo = mid + 1u; }
    else { hi = mid; }
  }
  let sign = bits & 0x80000000u;
  let level = min(lo, select(127u, 128u, sign != 0u));
  return word(129u + level) | sign;
}
@compute @workgroup_size(${STATIC_QUANTIZE_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
  let stride = groups.x * ${STATIC_QUANTIZE_WORKGROUP_SIZE}u;
  var i = gid.x;
  while (i < word(0u)) {
    output[i] = quantize(input[i]);
    i += stride;
  }
}
`;
