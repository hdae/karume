// karume conv_transpose1d (x[B,Cin,L] * W[Cin,Cout,K] + b[Cout], f32, 重み f16 格納, gather 形)
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

// f16 格納の展開: 要素 i = unpack2x16float(w[i / 2])[i % 2]（平坦添字の偶奇で対を選ぶ）
fn dequant(i: u32) -> f32 {
  let pair = unpack2x16float(w[i >> 1u]);
  return select(pair.x, pair.y, (i & 1u) == 1u);
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
    // 有効 tap は k ≡ shifted (mod stride) だけ — 出力要素あたり 1 回だけ商と剰余へ分解する
    let shifted = ox + dims.padding;
    let q = shifted / dims.stride;
    let r = shifted % dims.stride;
    // 有効 tap は k = r + j*stride / ix = q - j。j の範囲は 3 本の不等式で閉じる:
    // ix <= L-1 → j >= q+1-L / ix >= 0 → j <= q / k <= K-1 → j <= (K-1-r)/stride
    let j_start = max(0, i32(q) + 1 - i32(dims.length_in));
    // r > K-1（stride > K のときだけ到達）は有効 tap ゼロ。K-1-r の u32 桁借りを避けて
    // j_end = -1 < j_start（j_start >= 0）で空ループへ倒す
    var j_end = -1;
    if (r < dims.kernel) {
      j_end = min(i32((dims.kernel - 1u - r) / dims.stride), i32(q));
    }
    var acc = bias[oc];
    for (var ic = 0u; ic < dims.channels_in; ic = ic + 1u) {
      let x_base = (b * dims.channels_in + ic) * dims.length_in;
      // 重みは [Cin, Cout, K] — conv1d と転置（第 1 軸が入力チャネル）
      let w_base = (ic * dims.channels_out + oc) * dims.kernel;
      // MUST: j 昇順 = k 昇順。tap 集合も (ic, k) 昇順の縮約順序も k 全数走査版と同一で、
      // 同じ被演算子に同じ f32 積和が同じ順で掛かる = ビット同一（並べ替えは丸めを変える）
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
