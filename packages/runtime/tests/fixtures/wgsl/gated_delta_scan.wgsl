// karume gated_delta_scan (f32, key split 8, 32 columns per workgroup, token block 8, head dim 128; ADR 0135 decision 3)
struct Dims {
  time: u32,
  batch: u32,
  hk: u32,
  hv: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> q: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> k: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> v: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> g: array<f32>;
@group(0) @binding(5) var<storage, read> beta: array<f32>;
@group(0) @binding(6) var<storage, read_write> o: array<f32>;

const SCALE: f32 = 0.08838834764831845;
var<workgroup> qs: array<vec4<f32>, 256>;
var<workgroup> ks: array<vec4<f32>, 288>;
var<workgroup> vs: array<f32, 256>;
var<workgroup> al: array<f32, 8>;
var<workgroup> be: array<f32, 8>;
var<workgroup> part: array<vec2<f32>, 512>;

fn hsum(x: vec4<f32>) -> f32 {
  return (x.x + x.y) + (x.z + x.w);
}
fn exp_accurate(x: f32) -> f32 {
  if (x < -87.0) { return 0.0; }
  let n = round(x * 1.4426950408889634);
  let r = (x - n * 0.693145751953125) - n * 1.428606820309417e-6;
  let p = 1.0 + r * (1.0 + r * (0.5 + r * (0.16666667 + r * (0.041666668 + r * (0.008333334 + r * (0.0013888889 + r * 0.0001984127))))));
  return ldexp(p, i32(n));
}

// Staging item e of the block starting at token t0 (vec4 items: q [tb][32], k [tb+1][32], v [tb][cols/4]).
fn stage_load(e: u32, t0: u32, b: u32, h: u32, kh: u32, cb: u32) -> vec4<f32> {
  let T = dims.time;
  if (e < 256u) {
    let t = t0 + e / 32u;
    if (t >= T) { return vec4<f32>(0.0); }
    return q[((b * T + t) * dims.hk + kh) * 32u + e % 32u] * SCALE;
  }
  if (e < 544u) {
    let f = e - 256u;
    let t = t0 + f / 32u;
    if (t >= T) { return vec4<f32>(0.0); }
    return k[((b * T + t) * dims.hk + kh) * 32u + f % 32u];
  }
  if (e < 608u) {
    let f = e - 544u;
    let t = t0 + f / 8u;
    if (t >= T) { return vec4<f32>(0.0); }
    return v[((b * T + t) * dims.hv + h) * 32u + cb * 8u + f % 8u];
  }
  return vec4<f32>(0.0);
}

fn stage_store(e: u32, x: vec4<f32>) {
  if (e < 256u) {
    qs[e] = x;
  } else if (e < 544u) {
    ks[e - 256u] = x;
  } else if (e < 608u) {
    let f = e - 544u;
    let base = (f / 8u) * 32u + (f % 8u) * 4u;
    vs[base] = x.x;
    vs[base + 1u] = x.y;
    vs[base + 2u] = x.z;
    vs[base + 3u] = x.w;
  }
}

// Scalar items: e < tb -> alpha = exp(g) of token t0 + e; tb <= e < 2 tb -> beta.
fn stage_load_s(e: u32, t0: u32, b: u32, h: u32) -> f32 {
  let T = dims.time;
  if (e < 8u) {
    let t = t0 + e;
    if (t >= T) { return 1.0; }
    return exp_accurate(g[(b * T + t) * dims.hv + h]);
  }
  if (e < 16u) {
    let t = t0 + e - 8u;
    if (t >= T) { return 0.0; }
    return beta[(b * T + t) * dims.hv + h];
  }
  return 0.0;
}

fn stage_store_s(e: u32, x: f32) {
  if (e < 8u) {
    al[e] = x;
  } else if (e < 16u) {
    be[e - 8u] = x;
  }
}

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_index) lid: u32,
) {
  let T = dims.time;
  let ncb = 4u;
  let cb = wid.x % ncb;
  let bh = wid.x / ncb;
  let h = bh % dims.hv;
  let b = bh / dims.hv;
  let kh = h / (dims.hv / dims.hk);
  let jl = lid % 32u;
  let p = lid / 32u;
  let j = cb * 32u + jl;
  var st0 = vec4<f32>(0.0);
  var st1 = vec4<f32>(0.0);
  var st2 = vec4<f32>(0.0);
  var st3 = vec4<f32>(0.0);
  var pf0 = vec4<f32>(0.0);
  var pf1 = vec4<f32>(0.0);
  var pf2 = vec4<f32>(0.0);
  var pfs0 = 0.0;
  // block 0, synchronously
  pf0 = stage_load(lid + 0u, 0u, b, h, kh, cb);
  pf1 = stage_load(lid + 256u, 0u, b, h, kh, cb);
  pf2 = stage_load(lid + 512u, 0u, b, h, kh, cb);
  pfs0 = stage_load_s(lid + 0u, 0u, b, h);
  stage_store(lid + 0u, pf0);
  stage_store(lid + 256u, pf1);
  stage_store(lid + 512u, pf2);
  stage_store_s(lid + 0u, pfs0);
  workgroupBarrier();
  var kc0 = ks[p + 0u];
  var kc1 = ks[p + 8u];
  var kc2 = ks[p + 16u];
  var kc3 = ks[p + 24u];
  var a_acc = vec4<f32>(0.0);
  a_acc = a_acc + st0 * ks[p + 0u];
  a_acc = a_acc + st1 * ks[p + 8u];
  a_acc = a_acc + st2 * ks[p + 16u];
  a_acc = a_acc + st3 * ks[p + 24u];
  part[lid] = vec2<f32>(hsum(a_acc), 0.0);
  workgroupBarrier();
  var t0 = 0u;
  loop {
    if (t0 >= T) { break; }
    let t1 = t0 + 8u;
    let more = t1 < T;
    if (more) {
      pf0 = stage_load(lid + 0u, t1, b, h, kh, cb);
      pf1 = stage_load(lid + 256u, t1, b, h, kh, cb);
      pf2 = stage_load(lid + 512u, t1, b, h, kh, cb);
      pfs0 = stage_load_s(lid + 0u, t1, b, h);
    }
    if (t0 + 0u < T) {
      let t = t0 + 0u;
      let pv0 = part[0u * 256u + 0u + jl];
      let pv1 = part[0u * 256u + 32u + jl];
      let pv2 = part[0u * 256u + 64u + jl];
      let pv3 = part[0u * 256u + 96u + jl];
      let pv4 = part[0u * 256u + 128u + jl];
      let pv5 = part[0u * 256u + 160u + jl];
      let pv6 = part[0u * 256u + 192u + jl];
      let pv7 = part[0u * 256u + 224u + jl];
      let a_all = (((pv0.x + pv1.x) + (pv2.x + pv3.x)) + ((pv4.x + pv5.x) + (pv6.x + pv7.x)));
      let c_prev = (((pv0.y + pv1.y) + (pv2.y + pv3.y)) + ((pv4.y + pv5.y) + (pv6.y + pv7.y)));
      if (p == 0u && t > 0u) {
        o[((b * T + t - 1u) * dims.hv + h) * 128u + j] = c_prev;
      }
      let alpha = al[0u];
      let bet = be[0u];
      let delta = (vs[0u * 32u + jl] - alpha * a_all) * bet;
      var c_acc = vec4<f32>(0.0);
      a_acc = vec4<f32>(0.0);
      let kn0 = ks[(0u + 1u) * 32u + p + 0u];
      let qv0 = qs[0u * 32u + p + 0u];
      st0 = st0 * alpha + kc0 * delta;
      c_acc = c_acc + st0 * qv0;
      a_acc = a_acc + st0 * kn0;
      kc0 = kn0;
      let kn1 = ks[(0u + 1u) * 32u + p + 8u];
      let qv1 = qs[0u * 32u + p + 8u];
      st1 = st1 * alpha + kc1 * delta;
      c_acc = c_acc + st1 * qv1;
      a_acc = a_acc + st1 * kn1;
      kc1 = kn1;
      let kn2 = ks[(0u + 1u) * 32u + p + 16u];
      let qv2 = qs[0u * 32u + p + 16u];
      st2 = st2 * alpha + kc2 * delta;
      c_acc = c_acc + st2 * qv2;
      a_acc = a_acc + st2 * kn2;
      kc2 = kn2;
      let kn3 = ks[(0u + 1u) * 32u + p + 24u];
      let qv3 = qs[0u * 32u + p + 24u];
      st3 = st3 * alpha + kc3 * delta;
      c_acc = c_acc + st3 * qv3;
      a_acc = a_acc + st3 * kn3;
      kc3 = kn3;
      part[(1u - 0u) * 256u + lid] = vec2<f32>(hsum(a_acc), hsum(c_acc));
      workgroupBarrier();
    }
    if (t0 + 1u < T) {
      let t = t0 + 1u;
      let pv0 = part[1u * 256u + 0u + jl];
      let pv1 = part[1u * 256u + 32u + jl];
      let pv2 = part[1u * 256u + 64u + jl];
      let pv3 = part[1u * 256u + 96u + jl];
      let pv4 = part[1u * 256u + 128u + jl];
      let pv5 = part[1u * 256u + 160u + jl];
      let pv6 = part[1u * 256u + 192u + jl];
      let pv7 = part[1u * 256u + 224u + jl];
      let a_all = (((pv0.x + pv1.x) + (pv2.x + pv3.x)) + ((pv4.x + pv5.x) + (pv6.x + pv7.x)));
      let c_prev = (((pv0.y + pv1.y) + (pv2.y + pv3.y)) + ((pv4.y + pv5.y) + (pv6.y + pv7.y)));
      if (p == 0u && t > 0u) {
        o[((b * T + t - 1u) * dims.hv + h) * 128u + j] = c_prev;
      }
      let alpha = al[1u];
      let bet = be[1u];
      let delta = (vs[1u * 32u + jl] - alpha * a_all) * bet;
      var c_acc = vec4<f32>(0.0);
      a_acc = vec4<f32>(0.0);
      let kn0 = ks[(1u + 1u) * 32u + p + 0u];
      let qv0 = qs[1u * 32u + p + 0u];
      st0 = st0 * alpha + kc0 * delta;
      c_acc = c_acc + st0 * qv0;
      a_acc = a_acc + st0 * kn0;
      kc0 = kn0;
      let kn1 = ks[(1u + 1u) * 32u + p + 8u];
      let qv1 = qs[1u * 32u + p + 8u];
      st1 = st1 * alpha + kc1 * delta;
      c_acc = c_acc + st1 * qv1;
      a_acc = a_acc + st1 * kn1;
      kc1 = kn1;
      let kn2 = ks[(1u + 1u) * 32u + p + 16u];
      let qv2 = qs[1u * 32u + p + 16u];
      st2 = st2 * alpha + kc2 * delta;
      c_acc = c_acc + st2 * qv2;
      a_acc = a_acc + st2 * kn2;
      kc2 = kn2;
      let kn3 = ks[(1u + 1u) * 32u + p + 24u];
      let qv3 = qs[1u * 32u + p + 24u];
      st3 = st3 * alpha + kc3 * delta;
      c_acc = c_acc + st3 * qv3;
      a_acc = a_acc + st3 * kn3;
      kc3 = kn3;
      part[(1u - 1u) * 256u + lid] = vec2<f32>(hsum(a_acc), hsum(c_acc));
      workgroupBarrier();
    }
    if (t0 + 2u < T) {
      let t = t0 + 2u;
      let pv0 = part[0u * 256u + 0u + jl];
      let pv1 = part[0u * 256u + 32u + jl];
      let pv2 = part[0u * 256u + 64u + jl];
      let pv3 = part[0u * 256u + 96u + jl];
      let pv4 = part[0u * 256u + 128u + jl];
      let pv5 = part[0u * 256u + 160u + jl];
      let pv6 = part[0u * 256u + 192u + jl];
      let pv7 = part[0u * 256u + 224u + jl];
      let a_all = (((pv0.x + pv1.x) + (pv2.x + pv3.x)) + ((pv4.x + pv5.x) + (pv6.x + pv7.x)));
      let c_prev = (((pv0.y + pv1.y) + (pv2.y + pv3.y)) + ((pv4.y + pv5.y) + (pv6.y + pv7.y)));
      if (p == 0u && t > 0u) {
        o[((b * T + t - 1u) * dims.hv + h) * 128u + j] = c_prev;
      }
      let alpha = al[2u];
      let bet = be[2u];
      let delta = (vs[2u * 32u + jl] - alpha * a_all) * bet;
      var c_acc = vec4<f32>(0.0);
      a_acc = vec4<f32>(0.0);
      let kn0 = ks[(2u + 1u) * 32u + p + 0u];
      let qv0 = qs[2u * 32u + p + 0u];
      st0 = st0 * alpha + kc0 * delta;
      c_acc = c_acc + st0 * qv0;
      a_acc = a_acc + st0 * kn0;
      kc0 = kn0;
      let kn1 = ks[(2u + 1u) * 32u + p + 8u];
      let qv1 = qs[2u * 32u + p + 8u];
      st1 = st1 * alpha + kc1 * delta;
      c_acc = c_acc + st1 * qv1;
      a_acc = a_acc + st1 * kn1;
      kc1 = kn1;
      let kn2 = ks[(2u + 1u) * 32u + p + 16u];
      let qv2 = qs[2u * 32u + p + 16u];
      st2 = st2 * alpha + kc2 * delta;
      c_acc = c_acc + st2 * qv2;
      a_acc = a_acc + st2 * kn2;
      kc2 = kn2;
      let kn3 = ks[(2u + 1u) * 32u + p + 24u];
      let qv3 = qs[2u * 32u + p + 24u];
      st3 = st3 * alpha + kc3 * delta;
      c_acc = c_acc + st3 * qv3;
      a_acc = a_acc + st3 * kn3;
      kc3 = kn3;
      part[(1u - 0u) * 256u + lid] = vec2<f32>(hsum(a_acc), hsum(c_acc));
      workgroupBarrier();
    }
    if (t0 + 3u < T) {
      let t = t0 + 3u;
      let pv0 = part[1u * 256u + 0u + jl];
      let pv1 = part[1u * 256u + 32u + jl];
      let pv2 = part[1u * 256u + 64u + jl];
      let pv3 = part[1u * 256u + 96u + jl];
      let pv4 = part[1u * 256u + 128u + jl];
      let pv5 = part[1u * 256u + 160u + jl];
      let pv6 = part[1u * 256u + 192u + jl];
      let pv7 = part[1u * 256u + 224u + jl];
      let a_all = (((pv0.x + pv1.x) + (pv2.x + pv3.x)) + ((pv4.x + pv5.x) + (pv6.x + pv7.x)));
      let c_prev = (((pv0.y + pv1.y) + (pv2.y + pv3.y)) + ((pv4.y + pv5.y) + (pv6.y + pv7.y)));
      if (p == 0u && t > 0u) {
        o[((b * T + t - 1u) * dims.hv + h) * 128u + j] = c_prev;
      }
      let alpha = al[3u];
      let bet = be[3u];
      let delta = (vs[3u * 32u + jl] - alpha * a_all) * bet;
      var c_acc = vec4<f32>(0.0);
      a_acc = vec4<f32>(0.0);
      let kn0 = ks[(3u + 1u) * 32u + p + 0u];
      let qv0 = qs[3u * 32u + p + 0u];
      st0 = st0 * alpha + kc0 * delta;
      c_acc = c_acc + st0 * qv0;
      a_acc = a_acc + st0 * kn0;
      kc0 = kn0;
      let kn1 = ks[(3u + 1u) * 32u + p + 8u];
      let qv1 = qs[3u * 32u + p + 8u];
      st1 = st1 * alpha + kc1 * delta;
      c_acc = c_acc + st1 * qv1;
      a_acc = a_acc + st1 * kn1;
      kc1 = kn1;
      let kn2 = ks[(3u + 1u) * 32u + p + 16u];
      let qv2 = qs[3u * 32u + p + 16u];
      st2 = st2 * alpha + kc2 * delta;
      c_acc = c_acc + st2 * qv2;
      a_acc = a_acc + st2 * kn2;
      kc2 = kn2;
      let kn3 = ks[(3u + 1u) * 32u + p + 24u];
      let qv3 = qs[3u * 32u + p + 24u];
      st3 = st3 * alpha + kc3 * delta;
      c_acc = c_acc + st3 * qv3;
      a_acc = a_acc + st3 * kn3;
      kc3 = kn3;
      part[(1u - 1u) * 256u + lid] = vec2<f32>(hsum(a_acc), hsum(c_acc));
      workgroupBarrier();
    }
    if (t0 + 4u < T) {
      let t = t0 + 4u;
      let pv0 = part[0u * 256u + 0u + jl];
      let pv1 = part[0u * 256u + 32u + jl];
      let pv2 = part[0u * 256u + 64u + jl];
      let pv3 = part[0u * 256u + 96u + jl];
      let pv4 = part[0u * 256u + 128u + jl];
      let pv5 = part[0u * 256u + 160u + jl];
      let pv6 = part[0u * 256u + 192u + jl];
      let pv7 = part[0u * 256u + 224u + jl];
      let a_all = (((pv0.x + pv1.x) + (pv2.x + pv3.x)) + ((pv4.x + pv5.x) + (pv6.x + pv7.x)));
      let c_prev = (((pv0.y + pv1.y) + (pv2.y + pv3.y)) + ((pv4.y + pv5.y) + (pv6.y + pv7.y)));
      if (p == 0u && t > 0u) {
        o[((b * T + t - 1u) * dims.hv + h) * 128u + j] = c_prev;
      }
      let alpha = al[4u];
      let bet = be[4u];
      let delta = (vs[4u * 32u + jl] - alpha * a_all) * bet;
      var c_acc = vec4<f32>(0.0);
      a_acc = vec4<f32>(0.0);
      let kn0 = ks[(4u + 1u) * 32u + p + 0u];
      let qv0 = qs[4u * 32u + p + 0u];
      st0 = st0 * alpha + kc0 * delta;
      c_acc = c_acc + st0 * qv0;
      a_acc = a_acc + st0 * kn0;
      kc0 = kn0;
      let kn1 = ks[(4u + 1u) * 32u + p + 8u];
      let qv1 = qs[4u * 32u + p + 8u];
      st1 = st1 * alpha + kc1 * delta;
      c_acc = c_acc + st1 * qv1;
      a_acc = a_acc + st1 * kn1;
      kc1 = kn1;
      let kn2 = ks[(4u + 1u) * 32u + p + 16u];
      let qv2 = qs[4u * 32u + p + 16u];
      st2 = st2 * alpha + kc2 * delta;
      c_acc = c_acc + st2 * qv2;
      a_acc = a_acc + st2 * kn2;
      kc2 = kn2;
      let kn3 = ks[(4u + 1u) * 32u + p + 24u];
      let qv3 = qs[4u * 32u + p + 24u];
      st3 = st3 * alpha + kc3 * delta;
      c_acc = c_acc + st3 * qv3;
      a_acc = a_acc + st3 * kn3;
      kc3 = kn3;
      part[(1u - 0u) * 256u + lid] = vec2<f32>(hsum(a_acc), hsum(c_acc));
      workgroupBarrier();
    }
    if (t0 + 5u < T) {
      let t = t0 + 5u;
      let pv0 = part[1u * 256u + 0u + jl];
      let pv1 = part[1u * 256u + 32u + jl];
      let pv2 = part[1u * 256u + 64u + jl];
      let pv3 = part[1u * 256u + 96u + jl];
      let pv4 = part[1u * 256u + 128u + jl];
      let pv5 = part[1u * 256u + 160u + jl];
      let pv6 = part[1u * 256u + 192u + jl];
      let pv7 = part[1u * 256u + 224u + jl];
      let a_all = (((pv0.x + pv1.x) + (pv2.x + pv3.x)) + ((pv4.x + pv5.x) + (pv6.x + pv7.x)));
      let c_prev = (((pv0.y + pv1.y) + (pv2.y + pv3.y)) + ((pv4.y + pv5.y) + (pv6.y + pv7.y)));
      if (p == 0u && t > 0u) {
        o[((b * T + t - 1u) * dims.hv + h) * 128u + j] = c_prev;
      }
      let alpha = al[5u];
      let bet = be[5u];
      let delta = (vs[5u * 32u + jl] - alpha * a_all) * bet;
      var c_acc = vec4<f32>(0.0);
      a_acc = vec4<f32>(0.0);
      let kn0 = ks[(5u + 1u) * 32u + p + 0u];
      let qv0 = qs[5u * 32u + p + 0u];
      st0 = st0 * alpha + kc0 * delta;
      c_acc = c_acc + st0 * qv0;
      a_acc = a_acc + st0 * kn0;
      kc0 = kn0;
      let kn1 = ks[(5u + 1u) * 32u + p + 8u];
      let qv1 = qs[5u * 32u + p + 8u];
      st1 = st1 * alpha + kc1 * delta;
      c_acc = c_acc + st1 * qv1;
      a_acc = a_acc + st1 * kn1;
      kc1 = kn1;
      let kn2 = ks[(5u + 1u) * 32u + p + 16u];
      let qv2 = qs[5u * 32u + p + 16u];
      st2 = st2 * alpha + kc2 * delta;
      c_acc = c_acc + st2 * qv2;
      a_acc = a_acc + st2 * kn2;
      kc2 = kn2;
      let kn3 = ks[(5u + 1u) * 32u + p + 24u];
      let qv3 = qs[5u * 32u + p + 24u];
      st3 = st3 * alpha + kc3 * delta;
      c_acc = c_acc + st3 * qv3;
      a_acc = a_acc + st3 * kn3;
      kc3 = kn3;
      part[(1u - 1u) * 256u + lid] = vec2<f32>(hsum(a_acc), hsum(c_acc));
      workgroupBarrier();
    }
    if (t0 + 6u < T) {
      let t = t0 + 6u;
      let pv0 = part[0u * 256u + 0u + jl];
      let pv1 = part[0u * 256u + 32u + jl];
      let pv2 = part[0u * 256u + 64u + jl];
      let pv3 = part[0u * 256u + 96u + jl];
      let pv4 = part[0u * 256u + 128u + jl];
      let pv5 = part[0u * 256u + 160u + jl];
      let pv6 = part[0u * 256u + 192u + jl];
      let pv7 = part[0u * 256u + 224u + jl];
      let a_all = (((pv0.x + pv1.x) + (pv2.x + pv3.x)) + ((pv4.x + pv5.x) + (pv6.x + pv7.x)));
      let c_prev = (((pv0.y + pv1.y) + (pv2.y + pv3.y)) + ((pv4.y + pv5.y) + (pv6.y + pv7.y)));
      if (p == 0u && t > 0u) {
        o[((b * T + t - 1u) * dims.hv + h) * 128u + j] = c_prev;
      }
      let alpha = al[6u];
      let bet = be[6u];
      let delta = (vs[6u * 32u + jl] - alpha * a_all) * bet;
      var c_acc = vec4<f32>(0.0);
      a_acc = vec4<f32>(0.0);
      let kn0 = ks[(6u + 1u) * 32u + p + 0u];
      let qv0 = qs[6u * 32u + p + 0u];
      st0 = st0 * alpha + kc0 * delta;
      c_acc = c_acc + st0 * qv0;
      a_acc = a_acc + st0 * kn0;
      kc0 = kn0;
      let kn1 = ks[(6u + 1u) * 32u + p + 8u];
      let qv1 = qs[6u * 32u + p + 8u];
      st1 = st1 * alpha + kc1 * delta;
      c_acc = c_acc + st1 * qv1;
      a_acc = a_acc + st1 * kn1;
      kc1 = kn1;
      let kn2 = ks[(6u + 1u) * 32u + p + 16u];
      let qv2 = qs[6u * 32u + p + 16u];
      st2 = st2 * alpha + kc2 * delta;
      c_acc = c_acc + st2 * qv2;
      a_acc = a_acc + st2 * kn2;
      kc2 = kn2;
      let kn3 = ks[(6u + 1u) * 32u + p + 24u];
      let qv3 = qs[6u * 32u + p + 24u];
      st3 = st3 * alpha + kc3 * delta;
      c_acc = c_acc + st3 * qv3;
      a_acc = a_acc + st3 * kn3;
      kc3 = kn3;
      part[(1u - 0u) * 256u + lid] = vec2<f32>(hsum(a_acc), hsum(c_acc));
      workgroupBarrier();
    }
    if (t0 + 7u < T) {
      let t = t0 + 7u;
      let pv0 = part[1u * 256u + 0u + jl];
      let pv1 = part[1u * 256u + 32u + jl];
      let pv2 = part[1u * 256u + 64u + jl];
      let pv3 = part[1u * 256u + 96u + jl];
      let pv4 = part[1u * 256u + 128u + jl];
      let pv5 = part[1u * 256u + 160u + jl];
      let pv6 = part[1u * 256u + 192u + jl];
      let pv7 = part[1u * 256u + 224u + jl];
      let a_all = (((pv0.x + pv1.x) + (pv2.x + pv3.x)) + ((pv4.x + pv5.x) + (pv6.x + pv7.x)));
      let c_prev = (((pv0.y + pv1.y) + (pv2.y + pv3.y)) + ((pv4.y + pv5.y) + (pv6.y + pv7.y)));
      if (p == 0u && t > 0u) {
        o[((b * T + t - 1u) * dims.hv + h) * 128u + j] = c_prev;
      }
      let alpha = al[7u];
      let bet = be[7u];
      let delta = (vs[7u * 32u + jl] - alpha * a_all) * bet;
      var c_acc = vec4<f32>(0.0);
      a_acc = vec4<f32>(0.0);
      let kn0 = ks[(7u + 1u) * 32u + p + 0u];
      let qv0 = qs[7u * 32u + p + 0u];
      st0 = st0 * alpha + kc0 * delta;
      c_acc = c_acc + st0 * qv0;
      a_acc = a_acc + st0 * kn0;
      kc0 = kn0;
      let kn1 = ks[(7u + 1u) * 32u + p + 8u];
      let qv1 = qs[7u * 32u + p + 8u];
      st1 = st1 * alpha + kc1 * delta;
      c_acc = c_acc + st1 * qv1;
      a_acc = a_acc + st1 * kn1;
      kc1 = kn1;
      let kn2 = ks[(7u + 1u) * 32u + p + 16u];
      let qv2 = qs[7u * 32u + p + 16u];
      st2 = st2 * alpha + kc2 * delta;
      c_acc = c_acc + st2 * qv2;
      a_acc = a_acc + st2 * kn2;
      kc2 = kn2;
      let kn3 = ks[(7u + 1u) * 32u + p + 24u];
      let qv3 = qs[7u * 32u + p + 24u];
      st3 = st3 * alpha + kc3 * delta;
      c_acc = c_acc + st3 * qv3;
      a_acc = a_acc + st3 * kn3;
      kc3 = kn3;
      part[(1u - 1u) * 256u + lid] = vec2<f32>(hsum(a_acc), hsum(c_acc));
      workgroupBarrier();
    }
    if (more) {
      // the last token's barrier ordered every staging read of this block
      stage_store(lid + 0u, pf0);
      stage_store(lid + 256u, pf1);
      stage_store(lid + 512u, pf2);
      stage_store_s(lid + 0u, pfs0);
      workgroupBarrier();
    }
    t0 = t1;
  }
  if (T > 0u) {
    let par = T & 1u;
    let c_last = (((part[par * 256u + 0u + jl].y + part[par * 256u + 32u + jl].y) + (part[par * 256u + 64u + jl].y + part[par * 256u + 96u + jl].y)) + ((part[par * 256u + 128u + jl].y + part[par * 256u + 160u + jl].y) + (part[par * 256u + 192u + jl].y + part[par * 256u + 224u + jl].y)));
    if (p == 0u) {
      o[((b * T + T - 1u) * dims.hv + h) * 128u + j] = c_last;
    }
  }
}
