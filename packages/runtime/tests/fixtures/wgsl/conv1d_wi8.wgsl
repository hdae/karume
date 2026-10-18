// karume conv1d (x[B,Cin,L] * W[Cout,Cin/groups,K] + b[Cout], f32, 重み i8 格納, direct convolution)
struct Dims {
  n: u32,
  batch: u32,
  channels_in: u32,
  channels_out: u32,
  length_in: u32,
  length_out: u32,
  kernel: u32,
  stride: u32,
  padding: u32,
  dilation: u32,
  groups: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

@group(0) @binding(5) var<storage, read> wscale: array<f32>;

// decode of i8 storage: element i = f32(unpack4xI8(w[i / 4])[i % 4]) · scale
// (the flat index gives the word and the position. The scale is per output channel — ADR 0019)
fn dequant(i: u32, scale: f32) -> f32 {
  return f32(unpack4xI8(w[i >> 2u])[i & 3u]) * scale;
}

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let step = nwg.x * 256u;
  // The contract check (src/ops.ts) guarantees groups divides Cin / Cout — the division is exact.
  let in_per_group = dims.channels_in / dims.groups;
  let out_per_group = dims.channels_out / dims.groups;
  var i = gid.x;
  while (i < dims.n) {
    let plane = dims.channels_out * dims.length_out;
    let b = i / plane;
    let rest = i % plane;
    let oc = rest / dims.length_out;
    let ox = rest % dims.length_out;
    // the start position on the input side is signed (it goes negative by the padding)
    let origin = i32(ox * dims.stride) - i32(dims.padding);
    // the weight's second axis is Cin/groups — indexed by the relative number within the group
    let group_index = oc / out_per_group;
    let ic_base = group_index * in_per_group;
    // the output channel's scale is loop-invariant — not re-read per weight element (ADR 0019)
    let wscale_v = wscale[oc];
    var acc = bias[oc];
    for (var ic_rel = 0u; ic_rel < in_per_group; ic_rel = ic_rel + 1u) {
      let x_base = (b * dims.channels_in + ic_base + ic_rel) * dims.length_in;
      let w_base = (oc * in_per_group + ic_rel) * dims.kernel;
      for (var k = 0u; k < dims.kernel; k = k + 1u) {
        let ix = origin + i32(k * dims.dilation);
        // the padding region is zero-filled — skipped rather than added
        if (ix >= 0 && u32(ix) < dims.length_in) {
          acc = acc + x[x_base + u32(ix)] * dequant(w_base + k, wscale_v);
        }
      }
    }
    out[i] = acc;
    i = i + step;
  }
}
