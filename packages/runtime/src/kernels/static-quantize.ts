import { CodegenError } from "../codegen/errors.ts";
import { assertU32Params } from "../codegen/params.ts";

/**
 * Reproduces f32 division → round half to even → saturation to [-128,127] → f32 multiplication
 * with an integer table.
 */
export const staticQuantizeParams = (count: number, scale: number): Uint32Array<ArrayBuffer> => {
  assertU32Params("static_quantize params", { count });
  if (!Number.isFinite(scale) || scale < 0 || Math.fround(scale) !== scale) {
    throw new CodegenError(
      "static_quantize: scale must be non-negative, finite and exactly representable as f32",
    );
  }
  // 16-byte aligned. count / 128 boundary words / 129 output words / identity flag / padding.
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
    // Include the interval where the f32 division rounds to half. The product is exact in f64 (at
    // most 49 bits).
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

/**
 * The level guess of the SRQ search (v2 — perf-ledger K-97): one f32 division by the scale proposes
 * level `c`, and the two neighbouring boundaries confirm it, so the search issues one round of
 * independent table loads instead of the ~9 dependent ones of the binary search (that chain's
 * latency, not the table's address space, is what the standalone SRQ and the GEMV epilogue wait on
 * — 2026-10-31 attribution on the RTX 3080 Ti).
 *
 * `word(130)` is `fround(1 · scale)` = the scale itself ({@link staticQuantizeParams} — exact for
 * any f32 scale). The check reads boundary `c − 1` (`word(c)`) and boundary `c` (`word(c + 1)`):
 * `c` is the level exactly when the first is ≤ the magnitude and the second is not. `&` / `|` do
 * not short-circuit, so both loads issue together (`word(129)` at c = 128 is in range).
 * MUST: a guess that fails the check falls back to the v1 search text unchanged. The table is
 * monotone, so a confirmed guess equals the search's count, and the result is the v1 result for
 * every input **whatever the division rounds to** (its accuracy affects speed only; `min(…, 128)`
 * keeps any conversion result — inf, a saturated or implementation-defined value — in range).
 */
export const staticQuantizeGuessWgsl = (word: string, indent = "  "): string =>
  `${indent}let c_guess = min(u32(bitcast<f32>(magnitude) / bitcast<f32>(${word}(130u)) + 0.5), 128u);
${indent}let off_guess = ((c_guess > 0u) & (magnitude < ${word}(c_guess))) |
${indent}  ((c_guess < 128u) & (magnitude >= ${word}(c_guess + 1u)));`;

/** The binary search of v1 over the 128 boundaries — the fallback of a failed guess. */
const SEARCH_WGSL = `  var lo = 0u;
  var hi = 128u;
  while (lo < hi) {
    let mid = (lo + hi) >> 1u;
    if (magnitude >= word(1u + mid)) { lo = mid + 1u; }
    else { hi = mid; }
  }`;

/**
 * `fn quantize` of the f32 SRQ (v2): the bits of one f32 value → the bits of its SRQ output. It
 * reads the {@link staticQuantizeParams} table through `fn word(index: u32) -> u32`, which the
 * kernel that includes it defines (the standalone kernel reads its uniform array; a fused kernel
 * reads the table member of its own uniform).
 * MUST: a kernel that quantizes like the standalone SRQ includes this text, never a copy — the
 * fused result is the standalone one only while both compile the same function.
 */
export const STATIC_QUANTIZE_FN_WGSL: string = `fn quantize(bits: u32) -> u32 {
  if (word(258u) != 0u) { return bits; }
  let magnitude = bits & 0x7fffffffu;
  if (magnitude > 0x7f800000u) { return bits | 0x00400000u; }
  let sign = bits & 0x80000000u;
  let cap = select(127u, 128u, sign != 0u);
${staticQuantizeGuessWgsl("word")}
  // The output word of the guess loads with the check (it is discarded when the check fails)
  let guessed = word(129u + min(c_guess, cap));
  if (!off_guess) { return guessed | sign; }
${SEARCH_WGSL}
  let level = min(lo, cap);
  return word(129u + level) | sign;
}`;

export const STATIC_QUANTIZE_KEY = "static_quantize:v2:f32:wg128";
export const STATIC_QUANTIZE_WGSL: string = `// karume static_quantize: SRQ with a fixed f32 scale (ADR 0097)
@group(0) @binding(0) var<uniform> params: array<vec4<u32>, 65>;
@group(0) @binding(1) var<storage, read> input: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
fn word(index: u32) -> u32 { return params[index >> 2u][index & 3u]; }
${STATIC_QUANTIZE_FN_WGSL}
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
 * params of the SRQ with packed int8 output (ADR 0105). **The table is the same as the f32 path's**;
 * only the two forms that packed cannot represent are additionally rejected:
 *
 * - `scale === 0` (identity — the form that passes the input f32 words through) cannot be reduced
 *   to int8 codes.
 * - A form whose element count is not a multiple of 4 cannot be packed into 4 elements per word.
 *
 * MUST: both fail loudly (no silent fallback to the f32 path — with a fallback the consuming GEMV
 * would read f32 words through its `vec4<u32>` binding, a silent wrong value without an exception).
 */
export const staticQuantizePackedParams = (
  count: number,
  scale: number,
): Uint32Array<ArrayBuffer> => {
  if (!Number.isSafeInteger(count) || count < 0 || count % 4 !== 0) {
    throw new CodegenError(
      `static_quantize packed: element count ${count} is not a non-negative integer multiple of 4`,
    );
  }
  if (scale === 0) {
    throw new CodegenError(
      "static_quantize packed: scale 0 (identity) cannot be represented in packed int8",
    );
  }
  return staticQuantizeParams(count, scale);
};

export const STATIC_QUANTIZE_PACKED_KEY = "static_quantize:v2:packed-i8:wg128";
/**
 * SRQ with packed int8 output (ADR 0105). One u32 output word = 4 int8 codes; the code of element i
 * is byte `i & 3` of word `i >> 2` (the same little-endian order as `pack4xI8` / `unpack4xI8`).
 *
 * MUST: the boundary table, the level search (the guess of {@link staticQuantizeGuessWgsl}, then
 * the v1 search) and the resolution of level / sign are the f32 path's (`quantize` in
 * {@link STATIC_QUANTIZE_WGSL}). Only the last step differs — instead of
 * looking up the output value table `word(129 + level)` it returns the int8 code (two's
 * complement). The consumer restores it with `f32(code) * scale`, which matches the table lookup
 * per element as u32 (rationale: ADR 0105).
 * NOTE: only the two values int8 has no code for differ from the f32 path. **-0.0 falls to code
 * 0 = +0.0** (in the GEMV multiply-add `acc` starts at +0.0, so the output does not move — ADR
 * 0105), and **NaN saturates to ±127 / -128 as outside the boundary table** (the f32 path passes NaN
 * through).
 *
 * ## Geometry (the same "1 thread, 1 element" as the f32 path — ADR 0105 addendum 1)
 *
 * A form where one thread quantizes the 4 elements of a word serially drops the thread count to
 * 1/4 of the element count and lacks occupancy (k=1536 gives 384 threads = 3 workgroups). It also
 * measured 19% slower than the f32 path, so **only the work assignment** is aligned with the f32
 * path:
 *
 * - Each thread computes the code of its own element and puts it in workgroup shared memory (128
 *   elements = 1 tile).
 * - After `workgroupBarrier()`, the lower 32 threads pack 4 codes each into one word and write it.
 * - Tile selection is a grid-stride decided by `workgroup_id` alone, so the dispatch count is the
 *   same `ceil(count / 128)` as the f32 path, and the barrier is uniform across the whole
 *   workgroup even on the edge tile.
 *
 * MUST: branches that span a barrier derive only from `workgroup_id` (looping on
 * `local_invocation_index` makes the barrier non-uniform on the edge tile and fails WGSL
 * uniformity analysis).
 */
export const STATIC_QUANTIZE_PACKED_WGSL: string = `// karume static_quantize packed: SRQ with a fixed f32 scale written as 4 int8 codes per u32 word (ADR 0105)
@group(0) @binding(0) var<uniform> params: array<vec4<u32>, 65>;
@group(0) @binding(1) var<storage, read> input: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
fn word(index: u32) -> u32 { return params[index >> 2u][index & 3u]; }
fn quantizeCode(bits: u32) -> u32 {
  let magnitude = bits & 0x7fffffffu;
${staticQuantizeGuessWgsl("word")}
  var lo = c_guess;
  if (off_guess) {
${SEARCH_WGSL.replace("  var lo = 0u;", "  lo = 0u;").replace(/^/gm, "  ")}
  }
  let sign = bits & 0x80000000u;
  let level = min(lo, select(127u, 128u, sign != 0u));
  // Negatives are the low 8 bits of the two's complement (level 0 has no sign slot and falls to +0)
  return select(level, (0u - level) & 255u, sign != 0u);
}
// Code storage for one tile (${STATIC_QUANTIZE_WORKGROUP_SIZE} elements)
var<workgroup> codes: array<u32, ${STATIC_QUANTIZE_WORKGROUP_SIZE}>;
@compute @workgroup_size(${STATIC_QUANTIZE_WORKGROUP_SIZE})
fn main(
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(num_workgroups) groups: vec3<u32>,
) {
  let count = word(0u);
  let tiles = (count + ${STATIC_QUANTIZE_WORKGROUP_SIZE - 1}u) / ${STATIC_QUANTIZE_WORKGROUP_SIZE}u;
  // Tile selection depends on workgroup_id alone = the barrier below is uniform even at the edge
  for (var tile = wg.x; tile < tiles; tile += groups.x) {
    let index = tile * ${STATIC_QUANTIZE_WORKGROUP_SIZE}u + lid;
    var code = 0u;
    if (index < count) { code = quantizeCode(input[index]); }
    codes[lid] = code;
    workgroupBarrier();
    // The element count is a multiple of 4 (the params check), so a word is either entirely in range
    // or entirely out of range
    if (lid < ${STATIC_QUANTIZE_WORKGROUP_SIZE / 4}u) {
      let base = lid * 4u;
      if (tile * ${STATIC_QUANTIZE_WORKGROUP_SIZE}u + base < count) {
        output[tile * ${STATIC_QUANTIZE_WORKGROUP_SIZE / 4}u + lid] = codes[base] |
          (codes[base + 1u] << 8u) | (codes[base + 2u] << 16u) | (codes[base + 3u] << 24u);
      }
    }
    // Every thread finishes the reads above before the next tile overwrites codes (WAR)
    workgroupBarrier();
  }
}
`;
