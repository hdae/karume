enable subgroups, subgroup_size_control;
diagnostic(off, subgroup_uniformity);
// karume attention_online (O = softmax((q·scale)(k·scale)ᵀ) · v with S never stored — ADR 0132; q / k per-token i8, v per-column i8, P̃ = round(127·exp(S − m)) against the running max (emulated), D = 128, 64 rows × 64-key tiles, 256 threads, vec4 tiles, subgroup shuffle reductions)
struct Dims {
  m: u32,
  n: u32,
  row_offset: u32,
  scale: f32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> qq: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> kq: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> vq: array<u32>;
@group(0) @binding(4) var<storage, read_write> o: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> qs: array<f32>;
@group(0) @binding(6) var<storage, read> ks: array<f32>;
@group(0) @binding(7) var<storage, read> vs: array<vec4<f32>>;
// emulation of the integer dot product (core WGSL only); returns **the same integer** as dot4I8Packed —
// integer addition is associative and exact, and the intermediates stay at most 4·127·128 = 65,024, so nothing wraps
fn idot(a: u32, b: u32) -> i32 {
  return dot(unpack4xI8(a), unpack4xI8(b));
}
// q tile: 64 rows × 8 vec4 words, padded to 9 per row (loaded once per workgroup)
var<workgroup> qt: array<vec4<u32>, 576>;
// k tile: 64 keys × 8 vec4 words, padded to 9 per key
var<workgroup> kt: array<vec4<u32>, 576>;
// Vᵀ tile: [pack 0..15][column quad 0..31], padded to 33 per pack; word w of a column packs the keys {w, w+16, w+32, w+48}
var<workgroup> vt: array<vec4<u32>, 528>;
// P̃ tile, transposed: [word 0..15][row group 0..15] = that word of the rows ty·4..+3, padded to 17 per word
var<workgroup> pt: array<vec4<u32>, 272>;

const INV_P: f32 = 0.007874015748031496;
const NEG: f32 = -3.0e38;
@compute @workgroup_size(256) @subgroup_size(32)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_index) t: u32,
) {
  let bh = wid.y;
  let row0 = dims.row_offset + wid.x * 64u;
  // the 16 lanes of a row: one half of a subgroup under the measured linear lane mapping
  let tx = t % 16u;
  let ty = t / 16u;
  let n4 = dims.n / 4u;
  // this head's offsets: q / k in vec4 words, Vᵀ in words, the k scales in elements
  let q_bh = bh * dims.m * 8u;
  let k_bh = bh * dims.n * 8u;
  let v_bh = bh * 128u * n4;
  let ks_bh = bh * dims.n;
  let qrow0 = row0 + ty * 4u + 0u;
  // the half scale on the q side (the scale of a row beyond M is never used — clamp keeps the read in range)
  let qsc0 = qs[bh * dims.m + min(qrow0, dims.m - 1u)] * dims.scale;
  let qrow1 = row0 + ty * 4u + 1u;
  // the half scale on the q side (the scale of a row beyond M is never used — clamp keeps the read in range)
  let qsc1 = qs[bh * dims.m + min(qrow1, dims.m - 1u)] * dims.scale;
  let qrow2 = row0 + ty * 4u + 2u;
  // the half scale on the q side (the scale of a row beyond M is never used — clamp keeps the read in range)
  let qsc2 = qs[bh * dims.m + min(qrow2, dims.m - 1u)] * dims.scale;
  let qrow3 = row0 + ty * 4u + 3u;
  // the half scale on the q side (the scale of a row beyond M is never used — clamp keeps the read in range)
  let qsc3 = qs[bh * dims.m + min(qrow3, dims.m - 1u)] * dims.scale;
  for (var w = t; w < 512u; w = w + 256u) {
    let r = w / 8u;
    let pk = w % 8u;
    qt[r * 9u + pk] = select(vec4<u32>(0u), qq[q_bh + (row0 + r) * 8u + pk], row0 + r < dims.m);
  }
  var m0: f32 = NEG;
  var l0: f32 = 0.0;
  var m1: f32 = NEG;
  var l1: f32 = 0.0;
  var m2: f32 = NEG;
  var l2: f32 = 0.0;
  var m3: f32 = NEG;
  var l3: f32 = 0.0;
  var o0_0: vec4<f32> = vec4<f32>(0.0);
  var o0_1: vec4<f32> = vec4<f32>(0.0);
  var o1_0: vec4<f32> = vec4<f32>(0.0);
  var o1_1: vec4<f32> = vec4<f32>(0.0);
  var o2_0: vec4<f32> = vec4<f32>(0.0);
  var o2_1: vec4<f32> = vec4<f32>(0.0);
  var o3_0: vec4<f32> = vec4<f32>(0.0);
  var o3_1: vec4<f32> = vec4<f32>(0.0);
  var kr0: vec4<u32>;
  var kr1: vec4<u32>;
  var vr0: vec4<u32>;
  var vr1: vec4<u32>;
  let tiles = (dims.n + 63u) / 64u;
  for (var tile = 0u; tile < tiles; tile = tile + 1u) {
    let key0 = tile * 64u;
    // every load of the tile into registers first, then every store (the loads overlap)
    { let w = t + 0u; let r = w / 8u; let pk = w % 8u; kr0 = select(vec4<u32>(0u), kq[k_bh + (key0 + r) * 8u + pk], key0 + r < dims.n); }
    { let w = t + 256u; let r = w / 8u; let pk = w % 8u; kr1 = select(vec4<u32>(0u), kq[k_bh + (key0 + r) * 8u + pk], key0 + r < dims.n); }
    // Vᵀ owner (pk, d4): consecutive lanes take consecutive keys of the same four columns
    { let w = t + 0u; let pk = w % 16u; let d4 = w / 16u; let kg = key0 + pk; let shift = (kg % 4u) * 8u; let gw = kg / 4u;
      var vw0 = 0u;
      { let base = v_bh + (d4 * 4u + 0u) * n4 + gw; if (gw + 0u < n4) { vw0 = vw0 | (((vq[base + 0u] >> shift) & 0xffu) << 0u); } if (gw + 4u < n4) { vw0 = vw0 | (((vq[base + 4u] >> shift) & 0xffu) << 8u); } if (gw + 8u < n4) { vw0 = vw0 | (((vq[base + 8u] >> shift) & 0xffu) << 16u); } if (gw + 12u < n4) { vw0 = vw0 | (((vq[base + 12u] >> shift) & 0xffu) << 24u); } }
      var vw1 = 0u;
      { let base = v_bh + (d4 * 4u + 1u) * n4 + gw; if (gw + 0u < n4) { vw1 = vw1 | (((vq[base + 0u] >> shift) & 0xffu) << 0u); } if (gw + 4u < n4) { vw1 = vw1 | (((vq[base + 4u] >> shift) & 0xffu) << 8u); } if (gw + 8u < n4) { vw1 = vw1 | (((vq[base + 8u] >> shift) & 0xffu) << 16u); } if (gw + 12u < n4) { vw1 = vw1 | (((vq[base + 12u] >> shift) & 0xffu) << 24u); } }
      var vw2 = 0u;
      { let base = v_bh + (d4 * 4u + 2u) * n4 + gw; if (gw + 0u < n4) { vw2 = vw2 | (((vq[base + 0u] >> shift) & 0xffu) << 0u); } if (gw + 4u < n4) { vw2 = vw2 | (((vq[base + 4u] >> shift) & 0xffu) << 8u); } if (gw + 8u < n4) { vw2 = vw2 | (((vq[base + 8u] >> shift) & 0xffu) << 16u); } if (gw + 12u < n4) { vw2 = vw2 | (((vq[base + 12u] >> shift) & 0xffu) << 24u); } }
      var vw3 = 0u;
      { let base = v_bh + (d4 * 4u + 3u) * n4 + gw; if (gw + 0u < n4) { vw3 = vw3 | (((vq[base + 0u] >> shift) & 0xffu) << 0u); } if (gw + 4u < n4) { vw3 = vw3 | (((vq[base + 4u] >> shift) & 0xffu) << 8u); } if (gw + 8u < n4) { vw3 = vw3 | (((vq[base + 8u] >> shift) & 0xffu) << 16u); } if (gw + 12u < n4) { vw3 = vw3 | (((vq[base + 12u] >> shift) & 0xffu) << 24u); } }
      vr0 = vec4<u32>(vw0, vw1, vw2, vw3); }
    // Vᵀ owner (pk, d4): consecutive lanes take consecutive keys of the same four columns
    { let w = t + 256u; let pk = w % 16u; let d4 = w / 16u; let kg = key0 + pk; let shift = (kg % 4u) * 8u; let gw = kg / 4u;
      var vw0 = 0u;
      { let base = v_bh + (d4 * 4u + 0u) * n4 + gw; if (gw + 0u < n4) { vw0 = vw0 | (((vq[base + 0u] >> shift) & 0xffu) << 0u); } if (gw + 4u < n4) { vw0 = vw0 | (((vq[base + 4u] >> shift) & 0xffu) << 8u); } if (gw + 8u < n4) { vw0 = vw0 | (((vq[base + 8u] >> shift) & 0xffu) << 16u); } if (gw + 12u < n4) { vw0 = vw0 | (((vq[base + 12u] >> shift) & 0xffu) << 24u); } }
      var vw1 = 0u;
      { let base = v_bh + (d4 * 4u + 1u) * n4 + gw; if (gw + 0u < n4) { vw1 = vw1 | (((vq[base + 0u] >> shift) & 0xffu) << 0u); } if (gw + 4u < n4) { vw1 = vw1 | (((vq[base + 4u] >> shift) & 0xffu) << 8u); } if (gw + 8u < n4) { vw1 = vw1 | (((vq[base + 8u] >> shift) & 0xffu) << 16u); } if (gw + 12u < n4) { vw1 = vw1 | (((vq[base + 12u] >> shift) & 0xffu) << 24u); } }
      var vw2 = 0u;
      { let base = v_bh + (d4 * 4u + 2u) * n4 + gw; if (gw + 0u < n4) { vw2 = vw2 | (((vq[base + 0u] >> shift) & 0xffu) << 0u); } if (gw + 4u < n4) { vw2 = vw2 | (((vq[base + 4u] >> shift) & 0xffu) << 8u); } if (gw + 8u < n4) { vw2 = vw2 | (((vq[base + 8u] >> shift) & 0xffu) << 16u); } if (gw + 12u < n4) { vw2 = vw2 | (((vq[base + 12u] >> shift) & 0xffu) << 24u); } }
      var vw3 = 0u;
      { let base = v_bh + (d4 * 4u + 3u) * n4 + gw; if (gw + 0u < n4) { vw3 = vw3 | (((vq[base + 0u] >> shift) & 0xffu) << 0u); } if (gw + 4u < n4) { vw3 = vw3 | (((vq[base + 4u] >> shift) & 0xffu) << 8u); } if (gw + 8u < n4) { vw3 = vw3 | (((vq[base + 8u] >> shift) & 0xffu) << 16u); } if (gw + 12u < n4) { vw3 = vw3 | (((vq[base + 12u] >> shift) & 0xffu) << 24u); } }
      vr1 = vec4<u32>(vw0, vw1, vw2, vw3); }
    { let w = t + 0u; let r = w / 8u; let pk = w % 8u; kt[r * 9u + pk] = kr0; }
    { let w = t + 256u; let r = w / 8u; let pk = w % 8u; kt[r * 9u + pk] = kr1; }
    { let w = t + 0u; let pk = w % 16u; let d4 = w / 16u; vt[pk * 33u + d4] = vr0; }
    { let w = t + 256u; let pk = w % 16u; let d4 = w / 16u; vt[pk * 33u + d4] = vr1; }
    workgroupBarrier();
    // ---- scores: my 4 rows × the keys {tx, tx+16, tx+32, tx+48} ----
    var acc0: vec4<i32> = vec4<i32>(0);
    var acc1: vec4<i32> = vec4<i32>(0);
    var acc2: vec4<i32> = vec4<i32>(0);
    var acc3: vec4<i32> = vec4<i32>(0);
    for (var p4 = 0u; p4 < 8u; p4 = p4 + 1u) {
      let qv0 = qt[(ty * 4u + 0u) * 9u + p4];
      let qv1 = qt[(ty * 4u + 1u) * 9u + p4];
      let qv2 = qt[(ty * 4u + 2u) * 9u + p4];
      let qv3 = qt[(ty * 4u + 3u) * 9u + p4];
      let kv0 = kt[(tx + 0u) * 9u + p4];
      let kv1 = kt[(tx + 16u) * 9u + p4];
      let kv2 = kt[(tx + 32u) * 9u + p4];
      let kv3 = kt[(tx + 48u) * 9u + p4];
      acc0 = acc0 + vec4<i32>(idot(qv0.x, kv0.x) + idot(qv0.y, kv0.y) + idot(qv0.z, kv0.z) + idot(qv0.w, kv0.w), idot(qv0.x, kv1.x) + idot(qv0.y, kv1.y) + idot(qv0.z, kv1.z) + idot(qv0.w, kv1.w), idot(qv0.x, kv2.x) + idot(qv0.y, kv2.y) + idot(qv0.z, kv2.z) + idot(qv0.w, kv2.w), idot(qv0.x, kv3.x) + idot(qv0.y, kv3.y) + idot(qv0.z, kv3.z) + idot(qv0.w, kv3.w));
      acc1 = acc1 + vec4<i32>(idot(qv1.x, kv0.x) + idot(qv1.y, kv0.y) + idot(qv1.z, kv0.z) + idot(qv1.w, kv0.w), idot(qv1.x, kv1.x) + idot(qv1.y, kv1.y) + idot(qv1.z, kv1.z) + idot(qv1.w, kv1.w), idot(qv1.x, kv2.x) + idot(qv1.y, kv2.y) + idot(qv1.z, kv2.z) + idot(qv1.w, kv2.w), idot(qv1.x, kv3.x) + idot(qv1.y, kv3.y) + idot(qv1.z, kv3.z) + idot(qv1.w, kv3.w));
      acc2 = acc2 + vec4<i32>(idot(qv2.x, kv0.x) + idot(qv2.y, kv0.y) + idot(qv2.z, kv0.z) + idot(qv2.w, kv0.w), idot(qv2.x, kv1.x) + idot(qv2.y, kv1.y) + idot(qv2.z, kv1.z) + idot(qv2.w, kv1.w), idot(qv2.x, kv2.x) + idot(qv2.y, kv2.y) + idot(qv2.z, kv2.z) + idot(qv2.w, kv2.w), idot(qv2.x, kv3.x) + idot(qv2.y, kv3.y) + idot(qv2.z, kv3.z) + idot(qv2.w, kv3.w));
      acc3 = acc3 + vec4<i32>(idot(qv3.x, kv0.x) + idot(qv3.y, kv0.y) + idot(qv3.z, kv0.z) + idot(qv3.w, kv0.w), idot(qv3.x, kv1.x) + idot(qv3.y, kv1.y) + idot(qv3.z, kv1.z) + idot(qv3.w, kv1.w), idot(qv3.x, kv2.x) + idot(qv3.y, kv2.y) + idot(qv3.z, kv2.z) + idot(qv3.w, kv2.w), idot(qv3.x, kv3.x) + idot(qv3.y, kv3.y) + idot(qv3.z, kv3.z) + idot(qv3.w, kv3.w));
    }
    let kidx = key0 + tx;
    let kok = vec4<bool>(kidx < dims.n, kidx + 16u < dims.n, kidx + 32u < dims.n, kidx + 48u < dims.n);
    let ksc = vec4<f32>(
      ks[ks_bh + min(kidx, dims.n - 1u)],
      ks[ks_bh + min(kidx + 16u, dims.n - 1u)],
      ks[ks_bh + min(kidx + 32u, dims.n - 1u)],
      ks[ks_bh + min(kidx + 48u, dims.n - 1u)],
    ) * dims.scale;
    // MUST: fold (qs·scale)·(ks·scale) into one f32 first (the dequant order of ①QK)
    let sv0 = select(vec4<f32>(NEG), vec4<f32>(acc0) * (qsc0 * ksc), kok);
    var tm0 = max(max(sv0.x, sv0.y), max(sv0.z, sv0.w));
    // MUST: fold (qs·scale)·(ks·scale) into one f32 first (the dequant order of ①QK)
    let sv1 = select(vec4<f32>(NEG), vec4<f32>(acc1) * (qsc1 * ksc), kok);
    var tm1 = max(max(sv1.x, sv1.y), max(sv1.z, sv1.w));
    // MUST: fold (qs·scale)·(ks·scale) into one f32 first (the dequant order of ①QK)
    let sv2 = select(vec4<f32>(NEG), vec4<f32>(acc2) * (qsc2 * ksc), kok);
    var tm2 = max(max(sv2.x, sv2.y), max(sv2.z, sv2.w));
    // MUST: fold (qs·scale)·(ks·scale) into one f32 first (the dequant order of ①QK)
    let sv3 = select(vec4<f32>(NEG), vec4<f32>(acc3) * (qsc3 * ksc), kok);
    var tm3 = max(max(sv3.x, sv3.y), max(sv3.z, sv3.w));
    tm0 = max(tm0, subgroupShuffleXor(tm0, 8u));
    tm0 = max(tm0, subgroupShuffleXor(tm0, 4u));
    tm0 = max(tm0, subgroupShuffleXor(tm0, 2u));
    tm0 = max(tm0, subgroupShuffleXor(tm0, 1u));
    tm1 = max(tm1, subgroupShuffleXor(tm1, 8u));
    tm1 = max(tm1, subgroupShuffleXor(tm1, 4u));
    tm1 = max(tm1, subgroupShuffleXor(tm1, 2u));
    tm1 = max(tm1, subgroupShuffleXor(tm1, 1u));
    tm2 = max(tm2, subgroupShuffleXor(tm2, 8u));
    tm2 = max(tm2, subgroupShuffleXor(tm2, 4u));
    tm2 = max(tm2, subgroupShuffleXor(tm2, 2u));
    tm2 = max(tm2, subgroupShuffleXor(tm2, 1u));
    tm3 = max(tm3, subgroupShuffleXor(tm3, 8u));
    tm3 = max(tm3, subgroupShuffleXor(tm3, 4u));
    tm3 = max(tm3, subgroupShuffleXor(tm3, 2u));
    tm3 = max(tm3, subgroupShuffleXor(tm3, 1u));
    let mn0 = max(m0, tm0);
    let alpha0 = exp(m0 - mn0);
    m0 = mn0;
    l0 = l0 * alpha0;
    o0_0 = o0_0 * alpha0;
    o0_1 = o0_1 * alpha0;
    // MUST: exp per component (vector transcendentals are implementation-dependent)
    let pe0 = vec4<f32>(exp(sv0.x - mn0), exp(sv0.y - mn0), exp(sv0.z - mn0), exp(sv0.w - mn0));
    l0 = l0 + pe0.x + pe0.y + pe0.z + pe0.w;
    let pq0 = pack4xI8(vec4<i32>(round(pe0 * 127.0)));
    let mn1 = max(m1, tm1);
    let alpha1 = exp(m1 - mn1);
    m1 = mn1;
    l1 = l1 * alpha1;
    o1_0 = o1_0 * alpha1;
    o1_1 = o1_1 * alpha1;
    // MUST: exp per component (vector transcendentals are implementation-dependent)
    let pe1 = vec4<f32>(exp(sv1.x - mn1), exp(sv1.y - mn1), exp(sv1.z - mn1), exp(sv1.w - mn1));
    l1 = l1 + pe1.x + pe1.y + pe1.z + pe1.w;
    let pq1 = pack4xI8(vec4<i32>(round(pe1 * 127.0)));
    let mn2 = max(m2, tm2);
    let alpha2 = exp(m2 - mn2);
    m2 = mn2;
    l2 = l2 * alpha2;
    o2_0 = o2_0 * alpha2;
    o2_1 = o2_1 * alpha2;
    // MUST: exp per component (vector transcendentals are implementation-dependent)
    let pe2 = vec4<f32>(exp(sv2.x - mn2), exp(sv2.y - mn2), exp(sv2.z - mn2), exp(sv2.w - mn2));
    l2 = l2 + pe2.x + pe2.y + pe2.z + pe2.w;
    let pq2 = pack4xI8(vec4<i32>(round(pe2 * 127.0)));
    let mn3 = max(m3, tm3);
    let alpha3 = exp(m3 - mn3);
    m3 = mn3;
    l3 = l3 * alpha3;
    o3_0 = o3_0 * alpha3;
    o3_1 = o3_1 * alpha3;
    // MUST: exp per component (vector transcendentals are implementation-dependent)
    let pe3 = vec4<f32>(exp(sv3.x - mn3), exp(sv3.y - mn3), exp(sv3.z - mn3), exp(sv3.w - mn3));
    l3 = l3 + pe3.x + pe3.y + pe3.z + pe3.w;
    let pq3 = pack4xI8(vec4<i32>(round(pe3 * 127.0)));
    pt[tx * 17u + ty] = vec4<u32>(pq0, pq1, pq2, pq3);
    workgroupBarrier();
    // ---- PV: my 4 rows × the columns {64·g + 4·tx + c}, one column group at a time ----
    {
      var p0_0: vec4<i32> = vec4<i32>(0);
      var p1_0: vec4<i32> = vec4<i32>(0);
      var p2_0: vec4<i32> = vec4<i32>(0);
      var p3_0: vec4<i32> = vec4<i32>(0);
      for (var p4 = 0u; p4 < 4u; p4 = p4 + 1u) {
        {
          let pw = pt[(p4 * 4u + 0u) * 17u + ty];
          let vv = vt[(p4 * 4u + 0u) * 33u + tx];
          p0_0 = p0_0 + vec4<i32>(idot(pw.x, vv.x), idot(pw.x, vv.y), idot(pw.x, vv.z), idot(pw.x, vv.w));
          p1_0 = p1_0 + vec4<i32>(idot(pw.y, vv.x), idot(pw.y, vv.y), idot(pw.y, vv.z), idot(pw.y, vv.w));
          p2_0 = p2_0 + vec4<i32>(idot(pw.z, vv.x), idot(pw.z, vv.y), idot(pw.z, vv.z), idot(pw.z, vv.w));
          p3_0 = p3_0 + vec4<i32>(idot(pw.w, vv.x), idot(pw.w, vv.y), idot(pw.w, vv.z), idot(pw.w, vv.w));
        }
        {
          let pw = pt[(p4 * 4u + 1u) * 17u + ty];
          let vv = vt[(p4 * 4u + 1u) * 33u + tx];
          p0_0 = p0_0 + vec4<i32>(idot(pw.x, vv.x), idot(pw.x, vv.y), idot(pw.x, vv.z), idot(pw.x, vv.w));
          p1_0 = p1_0 + vec4<i32>(idot(pw.y, vv.x), idot(pw.y, vv.y), idot(pw.y, vv.z), idot(pw.y, vv.w));
          p2_0 = p2_0 + vec4<i32>(idot(pw.z, vv.x), idot(pw.z, vv.y), idot(pw.z, vv.z), idot(pw.z, vv.w));
          p3_0 = p3_0 + vec4<i32>(idot(pw.w, vv.x), idot(pw.w, vv.y), idot(pw.w, vv.z), idot(pw.w, vv.w));
        }
        {
          let pw = pt[(p4 * 4u + 2u) * 17u + ty];
          let vv = vt[(p4 * 4u + 2u) * 33u + tx];
          p0_0 = p0_0 + vec4<i32>(idot(pw.x, vv.x), idot(pw.x, vv.y), idot(pw.x, vv.z), idot(pw.x, vv.w));
          p1_0 = p1_0 + vec4<i32>(idot(pw.y, vv.x), idot(pw.y, vv.y), idot(pw.y, vv.z), idot(pw.y, vv.w));
          p2_0 = p2_0 + vec4<i32>(idot(pw.z, vv.x), idot(pw.z, vv.y), idot(pw.z, vv.z), idot(pw.z, vv.w));
          p3_0 = p3_0 + vec4<i32>(idot(pw.w, vv.x), idot(pw.w, vv.y), idot(pw.w, vv.z), idot(pw.w, vv.w));
        }
        {
          let pw = pt[(p4 * 4u + 3u) * 17u + ty];
          let vv = vt[(p4 * 4u + 3u) * 33u + tx];
          p0_0 = p0_0 + vec4<i32>(idot(pw.x, vv.x), idot(pw.x, vv.y), idot(pw.x, vv.z), idot(pw.x, vv.w));
          p1_0 = p1_0 + vec4<i32>(idot(pw.y, vv.x), idot(pw.y, vv.y), idot(pw.y, vv.z), idot(pw.y, vv.w));
          p2_0 = p2_0 + vec4<i32>(idot(pw.z, vv.x), idot(pw.z, vv.y), idot(pw.z, vv.z), idot(pw.z, vv.w));
          p3_0 = p3_0 + vec4<i32>(idot(pw.w, vv.x), idot(pw.w, vv.y), idot(pw.w, vv.z), idot(pw.w, vv.w));
        }
      }
      o0_0 = o0_0 + vec4<f32>(p0_0);
      o1_0 = o1_0 + vec4<f32>(p1_0);
      o2_0 = o2_0 + vec4<f32>(p2_0);
      o3_0 = o3_0 + vec4<f32>(p3_0);
    }
    {
      var p0_1: vec4<i32> = vec4<i32>(0);
      var p1_1: vec4<i32> = vec4<i32>(0);
      var p2_1: vec4<i32> = vec4<i32>(0);
      var p3_1: vec4<i32> = vec4<i32>(0);
      for (var p4 = 0u; p4 < 4u; p4 = p4 + 1u) {
        {
          let pw = pt[(p4 * 4u + 0u) * 17u + ty];
          let vv = vt[(p4 * 4u + 0u) * 33u + 16u + tx];
          p0_1 = p0_1 + vec4<i32>(idot(pw.x, vv.x), idot(pw.x, vv.y), idot(pw.x, vv.z), idot(pw.x, vv.w));
          p1_1 = p1_1 + vec4<i32>(idot(pw.y, vv.x), idot(pw.y, vv.y), idot(pw.y, vv.z), idot(pw.y, vv.w));
          p2_1 = p2_1 + vec4<i32>(idot(pw.z, vv.x), idot(pw.z, vv.y), idot(pw.z, vv.z), idot(pw.z, vv.w));
          p3_1 = p3_1 + vec4<i32>(idot(pw.w, vv.x), idot(pw.w, vv.y), idot(pw.w, vv.z), idot(pw.w, vv.w));
        }
        {
          let pw = pt[(p4 * 4u + 1u) * 17u + ty];
          let vv = vt[(p4 * 4u + 1u) * 33u + 16u + tx];
          p0_1 = p0_1 + vec4<i32>(idot(pw.x, vv.x), idot(pw.x, vv.y), idot(pw.x, vv.z), idot(pw.x, vv.w));
          p1_1 = p1_1 + vec4<i32>(idot(pw.y, vv.x), idot(pw.y, vv.y), idot(pw.y, vv.z), idot(pw.y, vv.w));
          p2_1 = p2_1 + vec4<i32>(idot(pw.z, vv.x), idot(pw.z, vv.y), idot(pw.z, vv.z), idot(pw.z, vv.w));
          p3_1 = p3_1 + vec4<i32>(idot(pw.w, vv.x), idot(pw.w, vv.y), idot(pw.w, vv.z), idot(pw.w, vv.w));
        }
        {
          let pw = pt[(p4 * 4u + 2u) * 17u + ty];
          let vv = vt[(p4 * 4u + 2u) * 33u + 16u + tx];
          p0_1 = p0_1 + vec4<i32>(idot(pw.x, vv.x), idot(pw.x, vv.y), idot(pw.x, vv.z), idot(pw.x, vv.w));
          p1_1 = p1_1 + vec4<i32>(idot(pw.y, vv.x), idot(pw.y, vv.y), idot(pw.y, vv.z), idot(pw.y, vv.w));
          p2_1 = p2_1 + vec4<i32>(idot(pw.z, vv.x), idot(pw.z, vv.y), idot(pw.z, vv.z), idot(pw.z, vv.w));
          p3_1 = p3_1 + vec4<i32>(idot(pw.w, vv.x), idot(pw.w, vv.y), idot(pw.w, vv.z), idot(pw.w, vv.w));
        }
        {
          let pw = pt[(p4 * 4u + 3u) * 17u + ty];
          let vv = vt[(p4 * 4u + 3u) * 33u + 16u + tx];
          p0_1 = p0_1 + vec4<i32>(idot(pw.x, vv.x), idot(pw.x, vv.y), idot(pw.x, vv.z), idot(pw.x, vv.w));
          p1_1 = p1_1 + vec4<i32>(idot(pw.y, vv.x), idot(pw.y, vv.y), idot(pw.y, vv.z), idot(pw.y, vv.w));
          p2_1 = p2_1 + vec4<i32>(idot(pw.z, vv.x), idot(pw.z, vv.y), idot(pw.z, vv.z), idot(pw.z, vv.w));
          p3_1 = p3_1 + vec4<i32>(idot(pw.w, vv.x), idot(pw.w, vv.y), idot(pw.w, vv.z), idot(pw.w, vv.w));
        }
      }
      o0_1 = o0_1 + vec4<f32>(p0_1);
      o1_1 = o1_1 + vec4<f32>(p1_1);
      o2_1 = o2_1 + vec4<f32>(p2_1);
      o3_1 = o3_1 + vec4<f32>(p3_1);
    }
    workgroupBarrier(); // the next tile overwrites kt / vt / pt
  }
  l0 = l0 + subgroupShuffleXor(l0, 8u);
  l0 = l0 + subgroupShuffleXor(l0, 4u);
  l0 = l0 + subgroupShuffleXor(l0, 2u);
  l0 = l0 + subgroupShuffleXor(l0, 1u);
  l1 = l1 + subgroupShuffleXor(l1, 8u);
  l1 = l1 + subgroupShuffleXor(l1, 4u);
  l1 = l1 + subgroupShuffleXor(l1, 2u);
  l1 = l1 + subgroupShuffleXor(l1, 1u);
  l2 = l2 + subgroupShuffleXor(l2, 8u);
  l2 = l2 + subgroupShuffleXor(l2, 4u);
  l2 = l2 + subgroupShuffleXor(l2, 2u);
  l2 = l2 + subgroupShuffleXor(l2, 1u);
  l3 = l3 + subgroupShuffleXor(l3, 8u);
  l3 = l3 + subgroupShuffleXor(l3, 4u);
  l3 = l3 + subgroupShuffleXor(l3, 2u);
  l3 = l3 + subgroupShuffleXor(l3, 1u);
  let vs0 = vs[bh * 32u + tx];
  let vs1 = vs[bh * 32u + 16u + tx];
  if (qrow0 < dims.m) {
    // MUST: 1/127 by multiplication; O · vs[d] first, then the row factor
    let f0 = INV_P / l0;
    let ob0 = (bh * dims.m + qrow0) * 32u;
    o[ob0 + tx] = (o0_0 * vs0) * f0;
    o[ob0 + 16u + tx] = (o0_1 * vs1) * f0;
  }
  if (qrow1 < dims.m) {
    // MUST: 1/127 by multiplication; O · vs[d] first, then the row factor
    let f1 = INV_P / l1;
    let ob1 = (bh * dims.m + qrow1) * 32u;
    o[ob1 + tx] = (o1_0 * vs0) * f1;
    o[ob1 + 16u + tx] = (o1_1 * vs1) * f1;
  }
  if (qrow2 < dims.m) {
    // MUST: 1/127 by multiplication; O · vs[d] first, then the row factor
    let f2 = INV_P / l2;
    let ob2 = (bh * dims.m + qrow2) * 32u;
    o[ob2 + tx] = (o2_0 * vs0) * f2;
    o[ob2 + 16u + tx] = (o2_1 * vs1) * f2;
  }
  if (qrow3 < dims.m) {
    // MUST: 1/127 by multiplication; O · vs[d] first, then the row factor
    let f3 = INV_P / l3;
    let ob3 = (bh * dims.m + qrow3) * 32u;
    o[ob3 + tx] = (o3_0 * vs0) * f3;
    o[ob3 + 16u + tx] = (o3_1 * vs1) * f3;
  }
}
