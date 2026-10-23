enable subgroups, subgroup_size_control;
diagnostic(off, subgroup_uniformity);
// karume attention_online (O = softmax((q·scale)(k·scale)ᵀ) · v with S never stored — ADR 0132; q / k per-token i8, v per-column i8, P̃ = round(127·exp(S − m)) against the running max (emulated), D = 128, 64 rows × 64-key tiles, 256 threads, subgroup shuffle reductions)
struct Dims {
  m: u32,
  n: u32,
  row_offset: u32,
  scale: f32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> qq: array<u32>;
@group(0) @binding(2) var<storage, read> kq: array<u32>;
@group(0) @binding(3) var<storage, read> vq: array<u32>;
@group(0) @binding(4) var<storage, read_write> o: array<f32>;
@group(0) @binding(5) var<storage, read> qs: array<f32>;
@group(0) @binding(6) var<storage, read> ks: array<f32>;
@group(0) @binding(7) var<storage, read> vs: array<f32>;
// emulation of the integer dot product (core WGSL only); returns **the same integer** as dot4I8Packed —
// integer addition is associative and exact, and the intermediates stay at most 4·127·128 = 65,024, so nothing wraps
fn idot(a: u32, b: u32) -> i32 {
  return dot(unpack4xI8(a), unpack4xI8(b));
}
// q tile: 64 rows × 32 words, padded to 33 per row (loaded once per workgroup)
var<workgroup> qt: array<u32, 2112>;
// k tile: 64 keys × 32 words, padded to 33 per key
var<workgroup> kt: array<u32, 2112>;
// Vᵀ tile: [pack 0..15][d 0..127]; word w of a column packs the keys {w, w+16, w+32, w+48}
var<workgroup> vt: array<u32, 2048>;
// P̃ tile: 64 rows × 16 words, padded to 17 per row
var<workgroup> pt: array<u32, 1088>;

const INV_P: f32 = 0.007874015748031496;
const NEG: f32 = -3.0e38;
@compute @workgroup_size(256) @subgroup_size(32)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_index) t: u32,
  @builtin(subgroup_invocation_id) lane: u32,
  @builtin(subgroup_id) sgid: u32,
) {
  let bh = wid.y;
  let row0 = dims.row_offset + wid.x * 64u;
  // MUST: the 16 lanes of a row share a subgroup — identified by the subgroup builtins, not by t
  let tx = lane % 16u;
  let ty = sgid * 2u + lane / 16u;
  let n4 = dims.n / 4u;
  let q_bh = bh * dims.m * 32u;
  let k_bh = bh * dims.n * 32u;
  let v_bh = bh * 128u * n4;
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
  for (var w = t; w < 2048u; w = w + 256u) {
    let r = w / 32u;
    let pk = w % 32u;
    qt[r * 33u + pk] = select(0u, qq[q_bh + (row0 + r) * 32u + pk], row0 + r < dims.m);
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
  let tiles = (dims.n + 63u) / 64u;
  for (var tile = 0u; tile < tiles; tile = tile + 1u) {
    let key0 = tile * 64u;
    for (var w = t; w < 2048u; w = w + 256u) {
      let r = w / 32u;
      let pk = w % 32u;
      kt[r * 33u + pk] = select(0u, kq[k_bh + (key0 + r) * 32u + pk], key0 + r < dims.n);
    }
    // Vᵀ: word (pk, d) = the bytes of keys key0 + pk + 16·i (i = 0..3), each in the global word
    // (key0 + pk)/4 + 4·i at byte (key0 + pk) % 4 (key0 is a multiple of 64, so the byte is pk % 4)
    for (var w = t; w < 2048u; w = w + 256u) {
      let pk = w / 128u;
      let d = w % 128u;
      let kg = key0 + pk;
      let shift = (kg % 4u) * 8u;
      let base = v_bh + d * n4 + kg / 4u;
      var word = 0u;
      if (kg / 4u + 0u < n4) { word = word | (((vq[base + 0u] >> shift) & 0xffu) << 0u); }
      if (kg / 4u + 4u < n4) { word = word | (((vq[base + 4u] >> shift) & 0xffu) << 8u); }
      if (kg / 4u + 8u < n4) { word = word | (((vq[base + 8u] >> shift) & 0xffu) << 16u); }
      if (kg / 4u + 12u < n4) { word = word | (((vq[base + 12u] >> shift) & 0xffu) << 24u); }
      vt[w] = word;
    }
    workgroupBarrier();
    // ---- scores: my 4 rows × the keys {tx, tx+16, tx+32, tx+48} ----
    var acc0: vec4<i32> = vec4<i32>(0);
    var acc1: vec4<i32> = vec4<i32>(0);
    var acc2: vec4<i32> = vec4<i32>(0);
    var acc3: vec4<i32> = vec4<i32>(0);
    for (var pk = 0u; pk < 32u; pk = pk + 1u) {
      let qw0 = qt[(ty * 4u + 0u) * 33u + pk];
      let qw1 = qt[(ty * 4u + 1u) * 33u + pk];
      let qw2 = qt[(ty * 4u + 2u) * 33u + pk];
      let qw3 = qt[(ty * 4u + 3u) * 33u + pk];
      let kw = vec4<u32>(
        kt[tx * 33u + pk],
        kt[(tx + 16u) * 33u + pk],
        kt[(tx + 32u) * 33u + pk],
        kt[(tx + 48u) * 33u + pk],
      );
      acc0 = acc0 + vec4<i32>(idot(qw0, kw.x), idot(qw0, kw.y), idot(qw0, kw.z), idot(qw0, kw.w));
      acc1 = acc1 + vec4<i32>(idot(qw1, kw.x), idot(qw1, kw.y), idot(qw1, kw.z), idot(qw1, kw.w));
      acc2 = acc2 + vec4<i32>(idot(qw2, kw.x), idot(qw2, kw.y), idot(qw2, kw.z), idot(qw2, kw.w));
      acc3 = acc3 + vec4<i32>(idot(qw3, kw.x), idot(qw3, kw.y), idot(qw3, kw.z), idot(qw3, kw.w));
    }
    let kidx = key0 + tx;
    let kok = vec4<bool>(kidx < dims.n, kidx + 16u < dims.n, kidx + 32u < dims.n, kidx + 48u < dims.n);
    let ksc = vec4<f32>(
      ks[k_bh / 32u + min(kidx, dims.n - 1u)],
      ks[k_bh / 32u + min(kidx + 16u, dims.n - 1u)],
      ks[k_bh / 32u + min(kidx + 32u, dims.n - 1u)],
      ks[k_bh / 32u + min(kidx + 48u, dims.n - 1u)],
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
    pt[(ty * 4u + 0u) * 17u + tx] = pack4xI8(vec4<i32>(round(pe0 * 127.0)));
    let mn1 = max(m1, tm1);
    let alpha1 = exp(m1 - mn1);
    m1 = mn1;
    l1 = l1 * alpha1;
    o1_0 = o1_0 * alpha1;
    o1_1 = o1_1 * alpha1;
    // MUST: exp per component (vector transcendentals are implementation-dependent)
    let pe1 = vec4<f32>(exp(sv1.x - mn1), exp(sv1.y - mn1), exp(sv1.z - mn1), exp(sv1.w - mn1));
    l1 = l1 + pe1.x + pe1.y + pe1.z + pe1.w;
    pt[(ty * 4u + 1u) * 17u + tx] = pack4xI8(vec4<i32>(round(pe1 * 127.0)));
    let mn2 = max(m2, tm2);
    let alpha2 = exp(m2 - mn2);
    m2 = mn2;
    l2 = l2 * alpha2;
    o2_0 = o2_0 * alpha2;
    o2_1 = o2_1 * alpha2;
    // MUST: exp per component (vector transcendentals are implementation-dependent)
    let pe2 = vec4<f32>(exp(sv2.x - mn2), exp(sv2.y - mn2), exp(sv2.z - mn2), exp(sv2.w - mn2));
    l2 = l2 + pe2.x + pe2.y + pe2.z + pe2.w;
    pt[(ty * 4u + 2u) * 17u + tx] = pack4xI8(vec4<i32>(round(pe2 * 127.0)));
    let mn3 = max(m3, tm3);
    let alpha3 = exp(m3 - mn3);
    m3 = mn3;
    l3 = l3 * alpha3;
    o3_0 = o3_0 * alpha3;
    o3_1 = o3_1 * alpha3;
    // MUST: exp per component (vector transcendentals are implementation-dependent)
    let pe3 = vec4<f32>(exp(sv3.x - mn3), exp(sv3.y - mn3), exp(sv3.z - mn3), exp(sv3.w - mn3));
    l3 = l3 + pe3.x + pe3.y + pe3.z + pe3.w;
    pt[(ty * 4u + 3u) * 17u + tx] = pack4xI8(vec4<i32>(round(pe3 * 127.0)));
    workgroupBarrier();
    // ---- PV: my 4 rows × the columns {tx + 16·j} ----
    var p0_0: vec4<i32> = vec4<i32>(0);
    var p0_1: vec4<i32> = vec4<i32>(0);
    var p1_0: vec4<i32> = vec4<i32>(0);
    var p1_1: vec4<i32> = vec4<i32>(0);
    var p2_0: vec4<i32> = vec4<i32>(0);
    var p2_1: vec4<i32> = vec4<i32>(0);
    var p3_0: vec4<i32> = vec4<i32>(0);
    var p3_1: vec4<i32> = vec4<i32>(0);
    for (var pk = 0u; pk < 16u; pk = pk + 1u) {
      let pw0 = pt[(ty * 4u + 0u) * 17u + pk];
      let pw1 = pt[(ty * 4u + 1u) * 17u + pk];
      let pw2 = pt[(ty * 4u + 2u) * 17u + pk];
      let pw3 = pt[(ty * 4u + 3u) * 17u + pk];
      let vb = pk * 128u;
      let vw0 = vec4<u32>(vt[vb + tx], vt[vb + tx + 16u], vt[vb + tx + 32u], vt[vb + tx + 48u]);
      let vw1 = vec4<u32>(vt[vb + tx + 64u], vt[vb + tx + 80u], vt[vb + tx + 96u], vt[vb + tx + 112u]);
      p0_0 = p0_0 + vec4<i32>(idot(pw0, vw0.x), idot(pw0, vw0.y), idot(pw0, vw0.z), idot(pw0, vw0.w));
      p0_1 = p0_1 + vec4<i32>(idot(pw0, vw1.x), idot(pw0, vw1.y), idot(pw0, vw1.z), idot(pw0, vw1.w));
      p1_0 = p1_0 + vec4<i32>(idot(pw1, vw0.x), idot(pw1, vw0.y), idot(pw1, vw0.z), idot(pw1, vw0.w));
      p1_1 = p1_1 + vec4<i32>(idot(pw1, vw1.x), idot(pw1, vw1.y), idot(pw1, vw1.z), idot(pw1, vw1.w));
      p2_0 = p2_0 + vec4<i32>(idot(pw2, vw0.x), idot(pw2, vw0.y), idot(pw2, vw0.z), idot(pw2, vw0.w));
      p2_1 = p2_1 + vec4<i32>(idot(pw2, vw1.x), idot(pw2, vw1.y), idot(pw2, vw1.z), idot(pw2, vw1.w));
      p3_0 = p3_0 + vec4<i32>(idot(pw3, vw0.x), idot(pw3, vw0.y), idot(pw3, vw0.z), idot(pw3, vw0.w));
      p3_1 = p3_1 + vec4<i32>(idot(pw3, vw1.x), idot(pw3, vw1.y), idot(pw3, vw1.z), idot(pw3, vw1.w));
    }
    o0_0 = o0_0 + vec4<f32>(p0_0);
    o0_1 = o0_1 + vec4<f32>(p0_1);
    o1_0 = o1_0 + vec4<f32>(p1_0);
    o1_1 = o1_1 + vec4<f32>(p1_1);
    o2_0 = o2_0 + vec4<f32>(p2_0);
    o2_1 = o2_1 + vec4<f32>(p2_1);
    o3_0 = o3_0 + vec4<f32>(p3_0);
    o3_1 = o3_1 + vec4<f32>(p3_1);
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
  let vs0 = vec4<f32>(vs[bh * 128u + tx], vs[bh * 128u + tx + 16u], vs[bh * 128u + tx + 32u], vs[bh * 128u + tx + 48u]);
  let vs1 = vec4<f32>(vs[bh * 128u + tx + 64u], vs[bh * 128u + tx + 80u], vs[bh * 128u + tx + 96u], vs[bh * 128u + tx + 112u]);
  if (qrow0 < dims.m) {
    // MUST: 1/127 by multiplication; O · vs[d] first, then the row factor
    let f0 = INV_P / l0;
    let ob0 = (bh * dims.m + qrow0) * 128u;
    let ov0_0 = (o0_0 * vs0) * f0;
    o[ob0 + tx] = ov0_0.x;
    o[ob0 + tx + 16u] = ov0_0.y;
    o[ob0 + tx + 32u] = ov0_0.z;
    o[ob0 + tx + 48u] = ov0_0.w;
    let ov0_1 = (o0_1 * vs1) * f0;
    o[ob0 + tx + 64u] = ov0_1.x;
    o[ob0 + tx + 80u] = ov0_1.y;
    o[ob0 + tx + 96u] = ov0_1.z;
    o[ob0 + tx + 112u] = ov0_1.w;
  }
  if (qrow1 < dims.m) {
    // MUST: 1/127 by multiplication; O · vs[d] first, then the row factor
    let f1 = INV_P / l1;
    let ob1 = (bh * dims.m + qrow1) * 128u;
    let ov1_0 = (o1_0 * vs0) * f1;
    o[ob1 + tx] = ov1_0.x;
    o[ob1 + tx + 16u] = ov1_0.y;
    o[ob1 + tx + 32u] = ov1_0.z;
    o[ob1 + tx + 48u] = ov1_0.w;
    let ov1_1 = (o1_1 * vs1) * f1;
    o[ob1 + tx + 64u] = ov1_1.x;
    o[ob1 + tx + 80u] = ov1_1.y;
    o[ob1 + tx + 96u] = ov1_1.z;
    o[ob1 + tx + 112u] = ov1_1.w;
  }
  if (qrow2 < dims.m) {
    // MUST: 1/127 by multiplication; O · vs[d] first, then the row factor
    let f2 = INV_P / l2;
    let ob2 = (bh * dims.m + qrow2) * 128u;
    let ov2_0 = (o2_0 * vs0) * f2;
    o[ob2 + tx] = ov2_0.x;
    o[ob2 + tx + 16u] = ov2_0.y;
    o[ob2 + tx + 32u] = ov2_0.z;
    o[ob2 + tx + 48u] = ov2_0.w;
    let ov2_1 = (o2_1 * vs1) * f2;
    o[ob2 + tx + 64u] = ov2_1.x;
    o[ob2 + tx + 80u] = ov2_1.y;
    o[ob2 + tx + 96u] = ov2_1.z;
    o[ob2 + tx + 112u] = ov2_1.w;
  }
  if (qrow3 < dims.m) {
    // MUST: 1/127 by multiplication; O · vs[d] first, then the row factor
    let f3 = INV_P / l3;
    let ob3 = (bh * dims.m + qrow3) * 128u;
    let ov3_0 = (o3_0 * vs0) * f3;
    o[ob3 + tx] = ov3_0.x;
    o[ob3 + tx + 16u] = ov3_0.y;
    o[ob3 + tx + 32u] = ov3_0.z;
    o[ob3 + tx + 48u] = ov3_0.w;
    let ov3_1 = (o3_1 * vs1) * f3;
    o[ob3 + tx + 64u] = ov3_1.x;
    o[ob3 + tx + 80u] = ov3_1.y;
    o[ob3 + tx + 96u] = ov3_1.z;
    o[ob3 + tx + 112u] = ov3_1.w;
  }
}
