// karume linear (x[m,k] · wᵀ[k,n] + bias[n], integer dot product of per-token i8 activations × i4 group 32 weights, tile 128x64 / 8x8 per thread / wg 8x16 / K 16)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> xq: array<u32>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<storage, read> wscale: array<f32>;
@group(0) @binding(6) var<storage, read> xscale: array<f32>;

// integer dot product (WGSL language extension packed_4x8_integer_dot_product); returns **the same integer** as the emulated variant
fn idot(a: u32, b: u32) -> i32 {
  return dot4I8Packed(a, b);
}

// integer-lane decode of i4 storage: flat elements i..i+3 (i a multiple of 4) → a u32 of four i8
// lanes. 1 word = 8 elements, so the 4 elements are the upper or lower 16 bits of the word
// ((i >> 2) & 1 picks). Restored as q = u − 8 **as an integer** (the scale is applied by the f32
// flush at the group boundary — ADR 0069 decision 4)
fn i4lanes(i: u32) -> u32 {
  let half = (w[i >> 3u] >> (((i >> 2u) & 1u) * 16u)) & 0xFFFFu;
  let u = vec4<u32>(half, half >> 4u, half >> 8u, half >> 12u) & vec4<u32>(0xFu);
  return pack4xI8(vec4<i32>(u) - vec4<i32>(8));
}

// Shared tiles are laid out **[pack][row] / [pack][column]** (rearranged from the prototype's
// [row][pack] — the inner reads get word stride 4 and bank conflicts drop from 8-way to
// 2-way)
var<workgroup> sa: array<u32, 512>;
var<workgroup> sb: array<u32, 256>;

@compute @workgroup_size(8, 16)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.y * 8u + lid.x;
  // Eligibility requires k % 4 == 0, so a row always starts on a word boundary (the i8 payload
  // packs 4 per flat index)
  let k4 = dims.k / 4u;
  // x tile assignment (128 rows × 4 packs over 128 threads in 4 rounds)
  let ar = tid / 4u;
  let ap = tid % 4u;
  let sa_at = ap * 128u + ar;
  let arow0 = wid.y * 128u + ar;
  let arow_base0 = arow0 * k4;
  let arow1 = arow0 + 32u;
  let arow_base1 = arow1 * k4;
  let arow2 = arow0 + 64u;
  let arow_base2 = arow2 * k4;
  let arow3 = arow0 + 96u;
  let arow_base3 = arow3 * k4;
  // W tile assignment (64 output channels × 4 packs over 128 threads in 2 rounds).
  // Weights are read as [n,k] (the contiguous direction is k = the pack direction, so **no
  // transpose is needed**)
  let wc = tid / 4u;
  let wp = tid % 4u;
  let sb_at = wp * 64u + wc;
  let wcol0 = wid.x * 64u + wc;
  let wrow_base0 = wcol0 * dims.k;
  let wcol1 = wcol0 + 32u;
  let wrow_base1 = wcol1 * dims.k;
  // Output assignment is **outside the loop** (the flush at a group boundary looks up
  // per-column scales)
  let ocol = wid.x * 64u + lid.x * 8u;
  let orow0 = wid.y * 128u + lid.y * 8u;
  let orow1 = orow0 + 1u;
  let orow2 = orow0 + 2u;
  let orow3 = orow0 + 3u;
  let orow4 = orow0 + 4u;
  let orow5 = orow0 + 5u;
  let orow6 = orow0 + 6u;
  let orow7 = orow0 + 7u;
  // group scales are a flat [n, k/g]. Invalid columns are clamped to the last valid column, so
  // nothing out of range is read. The row start is loop-invariant and hoisted (output of invalid
  // columns is discarded by the store guard)
  let groups = dims.k >> 5u;
  let wsb0 = min(ocol, dims.n - 1u) * groups;
  let wsb1 = min(ocol + 1u, dims.n - 1u) * groups;
  let wsb2 = min(ocol + 2u, dims.n - 1u) * groups;
  let wsb3 = min(ocol + 3u, dims.n - 1u) * groups;
  let wsb4 = min(ocol + 4u, dims.n - 1u) * groups;
  let wsb5 = min(ocol + 5u, dims.n - 1u) * groups;
  let wsb6 = min(ocol + 6u, dims.n - 1u) * groups;
  let wsb7 = min(ocol + 7u, dims.n - 1u) * groups;
  // f32 accumulator (written only at group boundaries — k/g roundings)
  var accf0_0 = vec4<f32>(0.0);
  var accf0_1 = vec4<f32>(0.0);
  var accf1_0 = vec4<f32>(0.0);
  var accf1_1 = vec4<f32>(0.0);
  var accf2_0 = vec4<f32>(0.0);
  var accf2_1 = vec4<f32>(0.0);
  var accf3_0 = vec4<f32>(0.0);
  var accf3_1 = vec4<f32>(0.0);
  var accf4_0 = vec4<f32>(0.0);
  var accf4_1 = vec4<f32>(0.0);
  var accf5_0 = vec4<f32>(0.0);
  var accf5_1 = vec4<f32>(0.0);
  var accf6_0 = vec4<f32>(0.0);
  var accf6_1 = vec4<f32>(0.0);
  var accf7_0 = vec4<f32>(0.0);
  var accf7_1 = vec4<f32>(0.0);
  // The loop condition is uniform (dims is a uniform buffer) — needed for the inner
  // workgroupBarrier to meet WGSL's uniformity requirement
  for (var gi = 0u; gi < groups; gi = gi + 1u) {
    // The i32 accumulator is reset per group (|acci| ≤ g·127·8 — limit: LINEAR_W4A8_MAX_GROUP)
    var acc0_0 = vec4<i32>(0);
    var acc0_1 = vec4<i32>(0);
    var acc1_0 = vec4<i32>(0);
    var acc1_1 = vec4<i32>(0);
    var acc2_0 = vec4<i32>(0);
    var acc2_1 = vec4<i32>(0);
    var acc3_0 = vec4<i32>(0);
    var acc3_1 = vec4<i32>(0);
    var acc4_0 = vec4<i32>(0);
    var acc4_1 = vec4<i32>(0);
    var acc5_0 = vec4<i32>(0);
    var acc5_1 = vec4<i32>(0);
    var acc6_0 = vec4<i32>(0);
    var acc6_1 = vec4<i32>(0);
    var acc7_0 = vec4<i32>(0);
    var acc7_1 = vec4<i32>(0);
    // 1 group = exactly 2 K tiles (the g % tileK == 0 check). t is the running number
    let gbase = gi * 2u;
    for (var gt = 0u; gt < 2u; gt = gt + 1u) {
      let t = gbase + gt;
    // Zero-fill out of range (dot4I8Packed(0, x) == 0, so the result is exact even with a K
    // remainder)
    let apack = t * 4u + ap;
    var av0 = 0u;
    if (arow0 < dims.m && apack < k4) {
      av0 = xq[arow_base0 + apack];
    }
    sa[sa_at] = av0;
    var av1 = 0u;
    if (arow1 < dims.m && apack < k4) {
      av1 = xq[arow_base1 + apack];
    }
    sa[sa_at + 32u] = av1;
    var av2 = 0u;
    if (arow2 < dims.m && apack < k4) {
      av2 = xq[arow_base2 + apack];
    }
    sa[sa_at + 64u] = av2;
    var av3 = 0u;
    if (arow3 < dims.m && apack < k4) {
      av3 = xq[arow_base3 + apack];
    }
    sa[sa_at + 96u] = av3;
    // i4 has 8 elements per word, so the pack index is in flat elements (k % g == 0 ∧
    // g % tileK == 0 leaves no K remainder)
    let welem = t * 16u + wp * 4u;
    var wv0 = 0u;
    if (wcol0 < dims.n) {
      wv0 = i4lanes(wrow_base0 + welem);
    }
    sb[sb_at] = wv0;
    var wv1 = 0u;
    if (wcol1 < dims.n) {
      wv1 = i4lanes(wrow_base1 + welem);
    }
    sb[sb_at + 32u] = wv1;
      workgroupBarrier();
    // 16 shared loads (8 words of B + 8 words of A) give 64 integer inner products = 256 MACs.
    // The reduction is exact i32 addition, so it **does not depend on order** (unlike the f32
    // skeleton, the addition order is not part of the numeric contract)
    for (var p = 0u; p < 4u; p = p + 1u) {
      let bcol = p * 64u + lid.x * 8u;
      let b0 = sb[bcol];
      let b1 = sb[bcol + 1u];
      let b2 = sb[bcol + 2u];
      let b3 = sb[bcol + 3u];
      let b4 = sb[bcol + 4u];
      let b5 = sb[bcol + 5u];
      let b6 = sb[bcol + 6u];
      let b7 = sb[bcol + 7u];
      let arow_at = p * 128u + lid.y * 8u;
      let a0 = sa[arow_at];
      acc0_0 = acc0_0 + vec4<i32>(idot(a0, b0), idot(a0, b1), idot(a0, b2), idot(a0, b3));
      acc0_1 = acc0_1 + vec4<i32>(idot(a0, b4), idot(a0, b5), idot(a0, b6), idot(a0, b7));
      let a1 = sa[arow_at + 1u];
      acc1_0 = acc1_0 + vec4<i32>(idot(a1, b0), idot(a1, b1), idot(a1, b2), idot(a1, b3));
      acc1_1 = acc1_1 + vec4<i32>(idot(a1, b4), idot(a1, b5), idot(a1, b6), idot(a1, b7));
      let a2 = sa[arow_at + 2u];
      acc2_0 = acc2_0 + vec4<i32>(idot(a2, b0), idot(a2, b1), idot(a2, b2), idot(a2, b3));
      acc2_1 = acc2_1 + vec4<i32>(idot(a2, b4), idot(a2, b5), idot(a2, b6), idot(a2, b7));
      let a3 = sa[arow_at + 3u];
      acc3_0 = acc3_0 + vec4<i32>(idot(a3, b0), idot(a3, b1), idot(a3, b2), idot(a3, b3));
      acc3_1 = acc3_1 + vec4<i32>(idot(a3, b4), idot(a3, b5), idot(a3, b6), idot(a3, b7));
      let a4 = sa[arow_at + 4u];
      acc4_0 = acc4_0 + vec4<i32>(idot(a4, b0), idot(a4, b1), idot(a4, b2), idot(a4, b3));
      acc4_1 = acc4_1 + vec4<i32>(idot(a4, b4), idot(a4, b5), idot(a4, b6), idot(a4, b7));
      let a5 = sa[arow_at + 5u];
      acc5_0 = acc5_0 + vec4<i32>(idot(a5, b0), idot(a5, b1), idot(a5, b2), idot(a5, b3));
      acc5_1 = acc5_1 + vec4<i32>(idot(a5, b4), idot(a5, b5), idot(a5, b6), idot(a5, b7));
      let a6 = sa[arow_at + 6u];
      acc6_0 = acc6_0 + vec4<i32>(idot(a6, b0), idot(a6, b1), idot(a6, b2), idot(a6, b3));
      acc6_1 = acc6_1 + vec4<i32>(idot(a6, b4), idot(a6, b5), idot(a6, b6), idot(a6, b7));
      let a7 = sa[arow_at + 7u];
      acc7_0 = acc7_0 + vec4<i32>(idot(a7, b0), idot(a7, b1), idot(a7, b2), idot(a7, b3));
      acc7_1 = acc7_1 + vec4<i32>(idot(a7, b4), idot(a7, b5), idot(a7, b6), idot(a7, b7));
    }
      workgroupBarrier();
    }
    // Flush to f32 exactly at the group boundary (MUST: single rounding of fma; xs is not
    // applied until the end)
    let ws0 = vec4<f32>(wscale[wsb0 + gi], wscale[wsb1 + gi], wscale[wsb2 + gi], wscale[wsb3 + gi]);
    accf0_0 = fma(vec4<f32>(acc0_0), ws0, accf0_0);
    accf1_0 = fma(vec4<f32>(acc1_0), ws0, accf1_0);
    accf2_0 = fma(vec4<f32>(acc2_0), ws0, accf2_0);
    accf3_0 = fma(vec4<f32>(acc3_0), ws0, accf3_0);
    accf4_0 = fma(vec4<f32>(acc4_0), ws0, accf4_0);
    accf5_0 = fma(vec4<f32>(acc5_0), ws0, accf5_0);
    accf6_0 = fma(vec4<f32>(acc6_0), ws0, accf6_0);
    accf7_0 = fma(vec4<f32>(acc7_0), ws0, accf7_0);
    let ws1 = vec4<f32>(wscale[wsb4 + gi], wscale[wsb5 + gi], wscale[wsb6 + gi], wscale[wsb7 + gi]);
    accf0_1 = fma(vec4<f32>(acc0_1), ws1, accf0_1);
    accf1_1 = fma(vec4<f32>(acc1_1), ws1, accf1_1);
    accf2_1 = fma(vec4<f32>(acc2_1), ws1, accf2_1);
    accf3_1 = fma(vec4<f32>(acc3_1), ws1, accf3_1);
    accf4_1 = fma(vec4<f32>(acc4_1), ws1, accf4_1);
    accf5_1 = fma(vec4<f32>(acc5_1), ws1, accf5_1);
    accf6_1 = fma(vec4<f32>(acc6_1), ws1, accf6_1);
    accf7_1 = fma(vec4<f32>(acc7_1), ws1, accf7_1);
  }
  if (orow0 < dims.m) {
    let obase = orow0 * dims.n;
    // MUST: xs goes to the final fma (single rounding) — wscale was folded at the group boundaries
    let xs = xscale[orow0];
    if (ocol < dims.n) {
      out[obase + ocol] = fma(accf0_0.x, xs, bias[ocol]);
    }
    if (ocol + 1u < dims.n) {
      out[obase + ocol + 1u] = fma(accf0_0.y, xs, bias[ocol + 1u]);
    }
    if (ocol + 2u < dims.n) {
      out[obase + ocol + 2u] = fma(accf0_0.z, xs, bias[ocol + 2u]);
    }
    if (ocol + 3u < dims.n) {
      out[obase + ocol + 3u] = fma(accf0_0.w, xs, bias[ocol + 3u]);
    }
    if (ocol + 4u < dims.n) {
      out[obase + ocol + 4u] = fma(accf0_1.x, xs, bias[ocol + 4u]);
    }
    if (ocol + 5u < dims.n) {
      out[obase + ocol + 5u] = fma(accf0_1.y, xs, bias[ocol + 5u]);
    }
    if (ocol + 6u < dims.n) {
      out[obase + ocol + 6u] = fma(accf0_1.z, xs, bias[ocol + 6u]);
    }
    if (ocol + 7u < dims.n) {
      out[obase + ocol + 7u] = fma(accf0_1.w, xs, bias[ocol + 7u]);
    }
  }
  if (orow1 < dims.m) {
    let obase = orow1 * dims.n;
    // MUST: xs goes to the final fma (single rounding) — wscale was folded at the group boundaries
    let xs = xscale[orow1];
    if (ocol < dims.n) {
      out[obase + ocol] = fma(accf1_0.x, xs, bias[ocol]);
    }
    if (ocol + 1u < dims.n) {
      out[obase + ocol + 1u] = fma(accf1_0.y, xs, bias[ocol + 1u]);
    }
    if (ocol + 2u < dims.n) {
      out[obase + ocol + 2u] = fma(accf1_0.z, xs, bias[ocol + 2u]);
    }
    if (ocol + 3u < dims.n) {
      out[obase + ocol + 3u] = fma(accf1_0.w, xs, bias[ocol + 3u]);
    }
    if (ocol + 4u < dims.n) {
      out[obase + ocol + 4u] = fma(accf1_1.x, xs, bias[ocol + 4u]);
    }
    if (ocol + 5u < dims.n) {
      out[obase + ocol + 5u] = fma(accf1_1.y, xs, bias[ocol + 5u]);
    }
    if (ocol + 6u < dims.n) {
      out[obase + ocol + 6u] = fma(accf1_1.z, xs, bias[ocol + 6u]);
    }
    if (ocol + 7u < dims.n) {
      out[obase + ocol + 7u] = fma(accf1_1.w, xs, bias[ocol + 7u]);
    }
  }
  if (orow2 < dims.m) {
    let obase = orow2 * dims.n;
    // MUST: xs goes to the final fma (single rounding) — wscale was folded at the group boundaries
    let xs = xscale[orow2];
    if (ocol < dims.n) {
      out[obase + ocol] = fma(accf2_0.x, xs, bias[ocol]);
    }
    if (ocol + 1u < dims.n) {
      out[obase + ocol + 1u] = fma(accf2_0.y, xs, bias[ocol + 1u]);
    }
    if (ocol + 2u < dims.n) {
      out[obase + ocol + 2u] = fma(accf2_0.z, xs, bias[ocol + 2u]);
    }
    if (ocol + 3u < dims.n) {
      out[obase + ocol + 3u] = fma(accf2_0.w, xs, bias[ocol + 3u]);
    }
    if (ocol + 4u < dims.n) {
      out[obase + ocol + 4u] = fma(accf2_1.x, xs, bias[ocol + 4u]);
    }
    if (ocol + 5u < dims.n) {
      out[obase + ocol + 5u] = fma(accf2_1.y, xs, bias[ocol + 5u]);
    }
    if (ocol + 6u < dims.n) {
      out[obase + ocol + 6u] = fma(accf2_1.z, xs, bias[ocol + 6u]);
    }
    if (ocol + 7u < dims.n) {
      out[obase + ocol + 7u] = fma(accf2_1.w, xs, bias[ocol + 7u]);
    }
  }
  if (orow3 < dims.m) {
    let obase = orow3 * dims.n;
    // MUST: xs goes to the final fma (single rounding) — wscale was folded at the group boundaries
    let xs = xscale[orow3];
    if (ocol < dims.n) {
      out[obase + ocol] = fma(accf3_0.x, xs, bias[ocol]);
    }
    if (ocol + 1u < dims.n) {
      out[obase + ocol + 1u] = fma(accf3_0.y, xs, bias[ocol + 1u]);
    }
    if (ocol + 2u < dims.n) {
      out[obase + ocol + 2u] = fma(accf3_0.z, xs, bias[ocol + 2u]);
    }
    if (ocol + 3u < dims.n) {
      out[obase + ocol + 3u] = fma(accf3_0.w, xs, bias[ocol + 3u]);
    }
    if (ocol + 4u < dims.n) {
      out[obase + ocol + 4u] = fma(accf3_1.x, xs, bias[ocol + 4u]);
    }
    if (ocol + 5u < dims.n) {
      out[obase + ocol + 5u] = fma(accf3_1.y, xs, bias[ocol + 5u]);
    }
    if (ocol + 6u < dims.n) {
      out[obase + ocol + 6u] = fma(accf3_1.z, xs, bias[ocol + 6u]);
    }
    if (ocol + 7u < dims.n) {
      out[obase + ocol + 7u] = fma(accf3_1.w, xs, bias[ocol + 7u]);
    }
  }
  if (orow4 < dims.m) {
    let obase = orow4 * dims.n;
    // MUST: xs goes to the final fma (single rounding) — wscale was folded at the group boundaries
    let xs = xscale[orow4];
    if (ocol < dims.n) {
      out[obase + ocol] = fma(accf4_0.x, xs, bias[ocol]);
    }
    if (ocol + 1u < dims.n) {
      out[obase + ocol + 1u] = fma(accf4_0.y, xs, bias[ocol + 1u]);
    }
    if (ocol + 2u < dims.n) {
      out[obase + ocol + 2u] = fma(accf4_0.z, xs, bias[ocol + 2u]);
    }
    if (ocol + 3u < dims.n) {
      out[obase + ocol + 3u] = fma(accf4_0.w, xs, bias[ocol + 3u]);
    }
    if (ocol + 4u < dims.n) {
      out[obase + ocol + 4u] = fma(accf4_1.x, xs, bias[ocol + 4u]);
    }
    if (ocol + 5u < dims.n) {
      out[obase + ocol + 5u] = fma(accf4_1.y, xs, bias[ocol + 5u]);
    }
    if (ocol + 6u < dims.n) {
      out[obase + ocol + 6u] = fma(accf4_1.z, xs, bias[ocol + 6u]);
    }
    if (ocol + 7u < dims.n) {
      out[obase + ocol + 7u] = fma(accf4_1.w, xs, bias[ocol + 7u]);
    }
  }
  if (orow5 < dims.m) {
    let obase = orow5 * dims.n;
    // MUST: xs goes to the final fma (single rounding) — wscale was folded at the group boundaries
    let xs = xscale[orow5];
    if (ocol < dims.n) {
      out[obase + ocol] = fma(accf5_0.x, xs, bias[ocol]);
    }
    if (ocol + 1u < dims.n) {
      out[obase + ocol + 1u] = fma(accf5_0.y, xs, bias[ocol + 1u]);
    }
    if (ocol + 2u < dims.n) {
      out[obase + ocol + 2u] = fma(accf5_0.z, xs, bias[ocol + 2u]);
    }
    if (ocol + 3u < dims.n) {
      out[obase + ocol + 3u] = fma(accf5_0.w, xs, bias[ocol + 3u]);
    }
    if (ocol + 4u < dims.n) {
      out[obase + ocol + 4u] = fma(accf5_1.x, xs, bias[ocol + 4u]);
    }
    if (ocol + 5u < dims.n) {
      out[obase + ocol + 5u] = fma(accf5_1.y, xs, bias[ocol + 5u]);
    }
    if (ocol + 6u < dims.n) {
      out[obase + ocol + 6u] = fma(accf5_1.z, xs, bias[ocol + 6u]);
    }
    if (ocol + 7u < dims.n) {
      out[obase + ocol + 7u] = fma(accf5_1.w, xs, bias[ocol + 7u]);
    }
  }
  if (orow6 < dims.m) {
    let obase = orow6 * dims.n;
    // MUST: xs goes to the final fma (single rounding) — wscale was folded at the group boundaries
    let xs = xscale[orow6];
    if (ocol < dims.n) {
      out[obase + ocol] = fma(accf6_0.x, xs, bias[ocol]);
    }
    if (ocol + 1u < dims.n) {
      out[obase + ocol + 1u] = fma(accf6_0.y, xs, bias[ocol + 1u]);
    }
    if (ocol + 2u < dims.n) {
      out[obase + ocol + 2u] = fma(accf6_0.z, xs, bias[ocol + 2u]);
    }
    if (ocol + 3u < dims.n) {
      out[obase + ocol + 3u] = fma(accf6_0.w, xs, bias[ocol + 3u]);
    }
    if (ocol + 4u < dims.n) {
      out[obase + ocol + 4u] = fma(accf6_1.x, xs, bias[ocol + 4u]);
    }
    if (ocol + 5u < dims.n) {
      out[obase + ocol + 5u] = fma(accf6_1.y, xs, bias[ocol + 5u]);
    }
    if (ocol + 6u < dims.n) {
      out[obase + ocol + 6u] = fma(accf6_1.z, xs, bias[ocol + 6u]);
    }
    if (ocol + 7u < dims.n) {
      out[obase + ocol + 7u] = fma(accf6_1.w, xs, bias[ocol + 7u]);
    }
  }
  if (orow7 < dims.m) {
    let obase = orow7 * dims.n;
    // MUST: xs goes to the final fma (single rounding) — wscale was folded at the group boundaries
    let xs = xscale[orow7];
    if (ocol < dims.n) {
      out[obase + ocol] = fma(accf7_0.x, xs, bias[ocol]);
    }
    if (ocol + 1u < dims.n) {
      out[obase + ocol + 1u] = fma(accf7_0.y, xs, bias[ocol + 1u]);
    }
    if (ocol + 2u < dims.n) {
      out[obase + ocol + 2u] = fma(accf7_0.z, xs, bias[ocol + 2u]);
    }
    if (ocol + 3u < dims.n) {
      out[obase + ocol + 3u] = fma(accf7_0.w, xs, bias[ocol + 3u]);
    }
    if (ocol + 4u < dims.n) {
      out[obase + ocol + 4u] = fma(accf7_1.x, xs, bias[ocol + 4u]);
    }
    if (ocol + 5u < dims.n) {
      out[obase + ocol + 5u] = fma(accf7_1.y, xs, bias[ocol + 5u]);
    }
    if (ocol + 6u < dims.n) {
      out[obase + ocol + 6u] = fma(accf7_1.z, xs, bias[ocol + 6u]);
    }
    if (ocol + 7u < dims.n) {
      out[obase + ocol + 7u] = fma(accf7_1.w, xs, bias[ocol + 7u]);
    }
  }
}
