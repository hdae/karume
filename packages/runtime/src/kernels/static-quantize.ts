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

/**
 * packed int8 出力の SRQ（ADR 0105）の params。**表は f32 経路と同じもの**で、
 * packed で表せない 2 つの形だけを追加で拒否する:
 *
 * - `scale === 0`（恒等 — 入力 f32 語をそのまま流す形）は int8 コードに落とせない。
 * - 要素数が 4 の倍数でない形は 1 語 4 要素に詰め切れない。
 *
 * MUST: どちらも fail loudly（黙って f32 経路へ退避しない — 退避すると消費側の GEMV が
 * `vec4<u32>` 束縛のまま f32 の語を読み、例外なしの沈黙誤値になる）。
 */
export const staticQuantizePackedParams = (
  count: number,
  scale: number,
): Uint32Array<ArrayBuffer> => {
  if (!Number.isSafeInteger(count) || count < 0 || count % 4 !== 0) {
    throw new CodegenError(`static_quantize packed: 要素数 ${count} が 4 の倍数の非負整数でない`);
  }
  if (scale === 0) {
    throw new CodegenError("static_quantize packed: scale 0（恒等）は packed int8 で表せない");
  }
  return staticQuantizeParams(count, scale);
};

export const STATIC_QUANTIZE_PACKED_KEY = "static_quantize:v1:packed-i8:wg128";
/**
 * packed int8 出力の SRQ（ADR 0105）。出力は u32 1 語 = int8 コード 4 個で、要素 i のコードは
 * 語 `i >> 2` のバイト `i & 3`（`pack4xI8` / `unpack4xI8` と同じリトルエンディアン順）。
 *
 * MUST: 境界表と level / sign の確定は f32 経路（{@link STATIC_QUANTIZE_WGSL} の `quantize`）と
 * **同じ字面**。違うのは最後の 1 手だけ — 出力値表 `word(129 + level)` を引く代わりに
 * int8 コード（2 の補数）を返す。消費側は `f32(code) * scale` で戻し、表引きと要素ごとに
 * u32 一致する（根拠は ADR 0105）。
 * NOTE: int8 に席が無い 2 値だけは f32 経路と違う。**-0.0 はコード 0 = +0.0 へ落ち**
 * （GEMV の積和では `acc` が +0.0 始まりなので出力は動かない — ADR 0105）、**NaN は境界表の
 * 外側として ±127 / -128 へ飽和する**（f32 経路は NaN をそのまま流す）。
 */
export const STATIC_QUANTIZE_PACKED_WGSL: string =
  `// karume static_quantize packed: 固定 f32 scale の SRQ を int8 コード 4 個 / u32 語で書く（ADR 0105）
@group(0) @binding(0) var<uniform> params: array<vec4<u32>, 65>;
@group(0) @binding(1) var<storage, read> input: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
fn word(index: u32) -> u32 { return params[index >> 2u][index & 3u]; }
fn quantizeCode(bits: u32) -> u32 {
  let magnitude = bits & 0x7fffffffu;
  var lo = 0u;
  var hi = 128u;
  while (lo < hi) {
    let mid = (lo + hi) >> 1u;
    if (magnitude >= word(1u + mid)) { lo = mid + 1u; }
    else { hi = mid; }
  }
  let sign = bits & 0x80000000u;
  let level = min(lo, select(127u, 128u, sign != 0u));
  // 負は 2 の補数の低位 8 bit（level 0 の符号は席が無く +0 へ落ちる）
  return select(level, (0u - level) & 255u, sign != 0u);
}
@compute @workgroup_size(${STATIC_QUANTIZE_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
  let stride = groups.x * ${STATIC_QUANTIZE_WORKGROUP_SIZE}u;
  let words = word(0u) >> 2u;
  var i = gid.x;
  while (i < words) {
    let base = i << 2u;
    output[i] = quantizeCode(input[base]) | (quantizeCode(input[base + 1u]) << 8u) |
      (quantizeCode(input[base + 2u]) << 16u) | (quantizeCode(input[base + 3u]) << 24u);
    i += stride;
  }
}
`;
