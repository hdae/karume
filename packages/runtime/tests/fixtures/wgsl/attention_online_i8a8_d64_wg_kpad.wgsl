// karume attention_online (O = softmax((q·scale)(k·scale)ᵀ) · v with S never stored — ADR 0132; q / k per-token i8, v per-column i8, P̃ = round(127·exp(S − m)) against the running max, D = 64, 64 rows × 64-key tiles, 256 threads, vec4 tiles, workgroup memory reductions, Vᵀ rows of ceil(N/4) words)
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
// integer dot product (WGSL language extension packed_4x8_integer_dot_product); returns **the same integer** as the emulated variant
fn idot(a: u32, b: u32) -> i32 {
  return dot4I8Packed(a, b);
}
// q tile: 64 rows × 4 vec4 words, padded to 5 per row (loaded once per workgroup)
var<workgroup> qt: array<vec4<u32>, 320>;
// k tile: 64 keys × 4 vec4 words, padded to 5 per key
var<workgroup> kt: array<vec4<u32>, 320>;
// Vᵀ tile: [pack 0..15][column quad 0..15], padded to 17 per pack; word w of a column packs the keys {w, w+16, w+32, w+48}
var<workgroup> vt: array<vec4<u32>, 272>;
// P̃ tile, transposed: [word 0..15][row group 0..15] = that word of the rows ty·4..+3, padded to 17 per word
var<workgroup> pt: array<vec4<u32>, 272>;
var<workgroup> red: array<f32, 1024>;
const INV_P: f32 = 0.007874015748031496;
const NEG: f32 = -3.0e38;
@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_index) t: u32,
) {
  let bh = wid.y;
  let row0 = dims.row_offset + wid.x * 64u;
  // the 16 lanes of a row: reduced through workgroup memory
  let tx = t % 16u;
  let ty = t / 16u;
  // Vᵀ rows hold ceil(N/4) words: the producer zero-pads the last one, and its keys from N on
  // score NEG, so their P̃ is 0 and the padding never weighs in
  let n4 = (dims.n + 3u) / 4u;
  // this head's offsets: q / k in vec4 words, Vᵀ in words, the k scales in elements
  let q_bh = bh * dims.m * 4u;
  let k_bh = bh * dims.n * 4u;
  let v_bh = bh * 64u * n4;
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
  for (var w = t; w < 256u; w = w + 256u) {
    let r = w / 4u;
    let pk = w % 4u;
    qt[r * 5u + pk] = select(vec4<u32>(0u), qq[q_bh + (row0 + r) * 4u + pk], row0 + r < dims.m);
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
  var o1_0: vec4<f32> = vec4<f32>(0.0);
  var o2_0: vec4<f32> = vec4<f32>(0.0);
  var o3_0: vec4<f32> = vec4<f32>(0.0);
  var kr0: vec4<u32>;
  var vr0: vec4<u32>;
  let tiles = (dims.n + 63u) / 64u;
  for (var tile = 0u; tile < tiles; tile = tile + 1u) {
    let key0 = tile * 64u;
    // every load of the tile into registers first, then every store (the loads overlap)
    { let w = t + 0u; let r = w / 4u; let pk = w % 4u; kr0 = select(vec4<u32>(0u), kq[k_bh + (key0 + r) * 4u + pk], key0 + r < dims.n); }
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
    { let w = t + 0u; let r = w / 4u; let pk = w % 4u; kt[r * 5u + pk] = kr0; }
    { let w = t + 0u; let pk = w % 16u; let d4 = w / 16u; vt[pk * 17u + d4] = vr0; }
    workgroupBarrier();
    // ---- scores: my 4 rows × the keys {tx, tx+16, tx+32, tx+48} ----
    var acc0: vec4<i32> = vec4<i32>(0);
    var acc1: vec4<i32> = vec4<i32>(0);
    var acc2: vec4<i32> = vec4<i32>(0);
    var acc3: vec4<i32> = vec4<i32>(0);
    for (var p4 = 0u; p4 < 4u; p4 = p4 + 1u) {
      let qv0 = qt[(ty * 4u + 0u) * 5u + p4];
      let qv1 = qt[(ty * 4u + 1u) * 5u + p4];
      let qv2 = qt[(ty * 4u + 2u) * 5u + p4];
      let qv3 = qt[(ty * 4u + 3u) * 5u + p4];
      let kv0 = kt[(tx + 0u) * 5u + p4];
      let kv1 = kt[(tx + 16u) * 5u + p4];
      let kv2 = kt[(tx + 32u) * 5u + p4];
      let kv3 = kt[(tx + 48u) * 5u + p4];
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
    red[(ty * 4u + 0u) * 16u + tx] = tm0;
    red[(ty * 4u + 1u) * 16u + tx] = tm1;
    red[(ty * 4u + 2u) * 16u + tx] = tm2;
    red[(ty * 4u + 3u) * 16u + tx] = tm3;
    workgroupBarrier();
    {
      let b = (ty * 4u + 0u) * 16u;
      var mx = red[b];
      for (var i = 1u; i < 16u; i = i + 1u) { mx = max(mx, red[b + i]); }
      tm0 = mx;
    }
    {
      let b = (ty * 4u + 1u) * 16u;
      var mx = red[b];
      for (var i = 1u; i < 16u; i = i + 1u) { mx = max(mx, red[b + i]); }
      tm1 = mx;
    }
    {
      let b = (ty * 4u + 2u) * 16u;
      var mx = red[b];
      for (var i = 1u; i < 16u; i = i + 1u) { mx = max(mx, red[b + i]); }
      tm2 = mx;
    }
    {
      let b = (ty * 4u + 3u) * 16u;
      var mx = red[b];
      for (var i = 1u; i < 16u; i = i + 1u) { mx = max(mx, red[b + i]); }
      tm3 = mx;
    }
    let mn0 = max(m0, tm0);
    let alpha0 = exp(m0 - mn0);
    m0 = mn0;
    l0 = l0 * alpha0;
    o0_0 = o0_0 * alpha0;
    // MUST: exp per component (vector transcendentals are implementation-dependent)
    let pe0 = vec4<f32>(exp(sv0.x - mn0), exp(sv0.y - mn0), exp(sv0.z - mn0), exp(sv0.w - mn0));
    l0 = l0 + pe0.x + pe0.y + pe0.z + pe0.w;
    let pq0 = pack4xI8(vec4<i32>(round(pe0 * 127.0)));
    let mn1 = max(m1, tm1);
    let alpha1 = exp(m1 - mn1);
    m1 = mn1;
    l1 = l1 * alpha1;
    o1_0 = o1_0 * alpha1;
    // MUST: exp per component (vector transcendentals are implementation-dependent)
    let pe1 = vec4<f32>(exp(sv1.x - mn1), exp(sv1.y - mn1), exp(sv1.z - mn1), exp(sv1.w - mn1));
    l1 = l1 + pe1.x + pe1.y + pe1.z + pe1.w;
    let pq1 = pack4xI8(vec4<i32>(round(pe1 * 127.0)));
    let mn2 = max(m2, tm2);
    let alpha2 = exp(m2 - mn2);
    m2 = mn2;
    l2 = l2 * alpha2;
    o2_0 = o2_0 * alpha2;
    // MUST: exp per component (vector transcendentals are implementation-dependent)
    let pe2 = vec4<f32>(exp(sv2.x - mn2), exp(sv2.y - mn2), exp(sv2.z - mn2), exp(sv2.w - mn2));
    l2 = l2 + pe2.x + pe2.y + pe2.z + pe2.w;
    let pq2 = pack4xI8(vec4<i32>(round(pe2 * 127.0)));
    let mn3 = max(m3, tm3);
    let alpha3 = exp(m3 - mn3);
    m3 = mn3;
    l3 = l3 * alpha3;
    o3_0 = o3_0 * alpha3;
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
          let vv = vt[(p4 * 4u + 0u) * 17u + tx];
          p0_0 = p0_0 + vec4<i32>(idot(pw.x, vv.x), idot(pw.x, vv.y), idot(pw.x, vv.z), idot(pw.x, vv.w));
          p1_0 = p1_0 + vec4<i32>(idot(pw.y, vv.x), idot(pw.y, vv.y), idot(pw.y, vv.z), idot(pw.y, vv.w));
          p2_0 = p2_0 + vec4<i32>(idot(pw.z, vv.x), idot(pw.z, vv.y), idot(pw.z, vv.z), idot(pw.z, vv.w));
          p3_0 = p3_0 + vec4<i32>(idot(pw.w, vv.x), idot(pw.w, vv.y), idot(pw.w, vv.z), idot(pw.w, vv.w));
        }
        {
          let pw = pt[(p4 * 4u + 1u) * 17u + ty];
          let vv = vt[(p4 * 4u + 1u) * 17u + tx];
          p0_0 = p0_0 + vec4<i32>(idot(pw.x, vv.x), idot(pw.x, vv.y), idot(pw.x, vv.z), idot(pw.x, vv.w));
          p1_0 = p1_0 + vec4<i32>(idot(pw.y, vv.x), idot(pw.y, vv.y), idot(pw.y, vv.z), idot(pw.y, vv.w));
          p2_0 = p2_0 + vec4<i32>(idot(pw.z, vv.x), idot(pw.z, vv.y), idot(pw.z, vv.z), idot(pw.z, vv.w));
          p3_0 = p3_0 + vec4<i32>(idot(pw.w, vv.x), idot(pw.w, vv.y), idot(pw.w, vv.z), idot(pw.w, vv.w));
        }
        {
          let pw = pt[(p4 * 4u + 2u) * 17u + ty];
          let vv = vt[(p4 * 4u + 2u) * 17u + tx];
          p0_0 = p0_0 + vec4<i32>(idot(pw.x, vv.x), idot(pw.x, vv.y), idot(pw.x, vv.z), idot(pw.x, vv.w));
          p1_0 = p1_0 + vec4<i32>(idot(pw.y, vv.x), idot(pw.y, vv.y), idot(pw.y, vv.z), idot(pw.y, vv.w));
          p2_0 = p2_0 + vec4<i32>(idot(pw.z, vv.x), idot(pw.z, vv.y), idot(pw.z, vv.z), idot(pw.z, vv.w));
          p3_0 = p3_0 + vec4<i32>(idot(pw.w, vv.x), idot(pw.w, vv.y), idot(pw.w, vv.z), idot(pw.w, vv.w));
        }
        {
          let pw = pt[(p4 * 4u + 3u) * 17u + ty];
          let vv = vt[(p4 * 4u + 3u) * 17u + tx];
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
    workgroupBarrier(); // the next tile overwrites kt / vt / pt
  }
  red[(ty * 4u + 0u) * 16u + tx] = l0;
  red[(ty * 4u + 1u) * 16u + tx] = l1;
  red[(ty * 4u + 2u) * 16u + tx] = l2;
  red[(ty * 4u + 3u) * 16u + tx] = l3;
  workgroupBarrier();
  {
    let b = (ty * 4u + 0u) * 16u;
    // The butterfly pairing of the sg32 form, spelled out: a_i = l_i + l_{i+8}, b_i = a_i + a_{i+4},
    // c_i = b_i + b_{i+2}, l = c_0 + c_1 (MUST: the same pairing, or the two forms diverge in bits)
    let a0 = red[b] + red[b + 8u];   let a1 = red[b + 1u] + red[b + 9u];
    let a2 = red[b + 2u] + red[b + 10u]; let a3 = red[b + 3u] + red[b + 11u];
    let a4 = red[b + 4u] + red[b + 12u]; let a5 = red[b + 5u] + red[b + 13u];
    let a6 = red[b + 6u] + red[b + 14u]; let a7 = red[b + 7u] + red[b + 15u];
    let b0 = a0 + a4; let b1 = a1 + a5; let b2 = a2 + a6; let b3 = a3 + a7;
    let c0 = b0 + b2; let c1 = b1 + b3;
    l0 = c0 + c1;
  }
  {
    let b = (ty * 4u + 1u) * 16u;
    // The butterfly pairing of the sg32 form, spelled out: a_i = l_i + l_{i+8}, b_i = a_i + a_{i+4},
    // c_i = b_i + b_{i+2}, l = c_0 + c_1 (MUST: the same pairing, or the two forms diverge in bits)
    let a0 = red[b] + red[b + 8u];   let a1 = red[b + 1u] + red[b + 9u];
    let a2 = red[b + 2u] + red[b + 10u]; let a3 = red[b + 3u] + red[b + 11u];
    let a4 = red[b + 4u] + red[b + 12u]; let a5 = red[b + 5u] + red[b + 13u];
    let a6 = red[b + 6u] + red[b + 14u]; let a7 = red[b + 7u] + red[b + 15u];
    let b0 = a0 + a4; let b1 = a1 + a5; let b2 = a2 + a6; let b3 = a3 + a7;
    let c0 = b0 + b2; let c1 = b1 + b3;
    l1 = c0 + c1;
  }
  {
    let b = (ty * 4u + 2u) * 16u;
    // The butterfly pairing of the sg32 form, spelled out: a_i = l_i + l_{i+8}, b_i = a_i + a_{i+4},
    // c_i = b_i + b_{i+2}, l = c_0 + c_1 (MUST: the same pairing, or the two forms diverge in bits)
    let a0 = red[b] + red[b + 8u];   let a1 = red[b + 1u] + red[b + 9u];
    let a2 = red[b + 2u] + red[b + 10u]; let a3 = red[b + 3u] + red[b + 11u];
    let a4 = red[b + 4u] + red[b + 12u]; let a5 = red[b + 5u] + red[b + 13u];
    let a6 = red[b + 6u] + red[b + 14u]; let a7 = red[b + 7u] + red[b + 15u];
    let b0 = a0 + a4; let b1 = a1 + a5; let b2 = a2 + a6; let b3 = a3 + a7;
    let c0 = b0 + b2; let c1 = b1 + b3;
    l2 = c0 + c1;
  }
  {
    let b = (ty * 4u + 3u) * 16u;
    // The butterfly pairing of the sg32 form, spelled out: a_i = l_i + l_{i+8}, b_i = a_i + a_{i+4},
    // c_i = b_i + b_{i+2}, l = c_0 + c_1 (MUST: the same pairing, or the two forms diverge in bits)
    let a0 = red[b] + red[b + 8u];   let a1 = red[b + 1u] + red[b + 9u];
    let a2 = red[b + 2u] + red[b + 10u]; let a3 = red[b + 3u] + red[b + 11u];
    let a4 = red[b + 4u] + red[b + 12u]; let a5 = red[b + 5u] + red[b + 13u];
    let a6 = red[b + 6u] + red[b + 14u]; let a7 = red[b + 7u] + red[b + 15u];
    let b0 = a0 + a4; let b1 = a1 + a5; let b2 = a2 + a6; let b3 = a3 + a7;
    let c0 = b0 + b2; let c1 = b1 + b3;
    l3 = c0 + c1;
  }
  let vs0 = vs[bh * 16u + tx];
  if (qrow0 < dims.m) {
    // MUST: 1/127 by multiplication; O · vs[d] first, then the row factor
    let f0 = INV_P / l0;
    let ob0 = (bh * dims.m + qrow0) * 16u;
    o[ob0 + tx] = (o0_0 * vs0) * f0;
  }
  if (qrow1 < dims.m) {
    // MUST: 1/127 by multiplication; O · vs[d] first, then the row factor
    let f1 = INV_P / l1;
    let ob1 = (bh * dims.m + qrow1) * 16u;
    o[ob1 + tx] = (o1_0 * vs0) * f1;
  }
  if (qrow2 < dims.m) {
    // MUST: 1/127 by multiplication; O · vs[d] first, then the row factor
    let f2 = INV_P / l2;
    let ob2 = (bh * dims.m + qrow2) * 16u;
    o[ob2 + tx] = (o2_0 * vs0) * f2;
  }
  if (qrow3 < dims.m) {
    // MUST: 1/127 by multiplication; O · vs[d] first, then the row factor
    let f3 = INV_P / l3;
    let ob3 = (bh * dims.m + qrow3) * 16u;
    o[ob3 + tx] = (o3_0 * vs0) * f3;
  }
}
