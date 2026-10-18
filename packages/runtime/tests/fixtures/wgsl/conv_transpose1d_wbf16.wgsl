// karume conv_transpose1d (x[B,Cin,L] * W[Cin,Cout,K] + b[Cout], f32, 重み bf16 格納, gather form)
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
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

// decode of bf16 storage: element i = the lower / upper 16 bits of w[i / 2] moved to the top
// of an f32 (the parity of the flat index picks)
fn dequant(i: u32) -> f32 {
  let word = w[i >> 1u];
  return bitcast<f32>(select(word << 16u, word & 0xFFFF0000u, (i & 1u) == 1u));
}

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let step = nwg.x * 256u;
  var i = gid.x;
  while (i < dims.n) {
    let plane = dims.channels_out * dims.length_out;
    let b = i / plane;
    let rest = i % plane;
    let oc = rest / dims.length_out;
    let ox = rest % dims.length_out;
    // ox = ix*stride - padding + k  =>  ix*stride = ox + padding - k
    // the valid taps are only k ≡ shifted (mod stride) — decompose into quotient and remainder
    // once per output element
    let shifted = ox + dims.padding;
    let q = shifted / dims.stride;
    let r = shifted % dims.stride;
    // the valid taps are k = r + j*stride / ix = q - j. The range of j closes with three
    // inequalities:
    // ix <= L-1 → j >= q+1-L / ix >= 0 → j <= q / k <= K-1 → j <= (K-1-r)/stride
    let j_start = max(0, i32(q) + 1 - i32(dims.length_in));
    // r > K-1 (reachable only when stride > K) has zero valid taps. Avoids the u32 borrow of K-1-r
    // and turns it into an empty loop with j_end = -1 < j_start (j_start >= 0)
    var j_end = -1;
    if (r < dims.kernel) {
      j_end = min(i32((dims.kernel - 1u - r) / dims.stride), i32(q));
    }
    var acc = bias[oc];
    for (var ic = 0u; ic < dims.channels_in; ic = ic + 1u) {
      let x_base = (b * dims.channels_in + ic) * dims.length_in;
      // the weight is [Cin, Cout, K] — the transpose of conv1d (the first axis is the input
      // channel)
      let w_base = (ic * dims.channels_out + oc) * dims.kernel;
      // MUST: ascending j = ascending k. The tap set and the ascending (ic, k) reduction order are
      // identical to the version that scans every k: the same f32 multiply-adds apply to the same
      // operands in the same order = bit-identical (reordering changes the rounding)
      for (var j = j_start; j <= j_end; j = j + 1) {
        let ix = q - u32(j);
        let k = r + u32(j) * dims.stride;
        acc = acc + x[x_base + ix] * dequant(w_base + k);
      }
    }
    out[i] = acc;
    i = i + step;
  }
}
