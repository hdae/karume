// karume conv_transpose1d (x[B,Cin,L] * W[Cin,Cout,K] + b[Cout], f32, 重み i8 格納, gather 形)
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

@group(0) @binding(5) var<storage, read> wscale: array<f32>;

// i8 格納の展開: 要素 i = f32(unpack4xI8(w[i / 4])[i % 4]) · scale
// （平坦添字で語と位置を割る。scale は出力チャネルごと — ADR 0019）
fn dequant(i: u32, scale: f32) -> f32 {
  return f32(unpack4xI8(w[i >> 2u])[i & 3u]) * scale;
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
    let shifted = i32(ox) + i32(dims.padding);
    // 出力チャネルの scale はループ不変 — 重みの要素ごとに引き直さない（ADR 0019）
    let wscale_v = wscale[oc];
    var acc = bias[oc];
    for (var ic = 0u; ic < dims.channels_in; ic = ic + 1u) {
      let x_base = (b * dims.channels_in + ic) * dims.length_in;
      // 重みは [Cin, Cout, K] — conv1d と転置（第 1 軸が入力チャネル）
      let w_base = (ic * dims.channels_out + oc) * dims.kernel;
      for (var k = 0u; k < dims.kernel; k = k + 1u) {
        let t = shifted - i32(k);
        // stride で割り切れない位置には入力が無い（寄与ゼロ）— 加算せずに読み飛ばす
        if (t >= 0 && u32(t) % dims.stride == 0u) {
          let ix = u32(t) / dims.stride;
          if (ix < dims.length_in) {
            acc = acc + x[x_base + ix] * dequant(w_base + k, wscale_v);
          }
        }
      }
    }
    out[i] = acc;
    i = i + step;
  }
}
