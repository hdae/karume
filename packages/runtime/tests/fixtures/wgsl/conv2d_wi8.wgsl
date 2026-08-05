// karume conv2d (x[B,Cin,H,W] * W[Cout,Cin/groups,Kh,Kw] + b[Cout], f32, 重み i8 格納, 直接畳み込み)
struct Dims {
  n: u32,
  batch: u32,
  channels_in: u32,
  channels_out: u32,
  height_in: u32,
  width_in: u32,
  height_out: u32,
  width_out: u32,
  kernel_h: u32,
  kernel_w: u32,
  stride_h: u32,
  stride_w: u32,
  padding_h: u32,
  padding_w: u32,
  dilation_h: u32,
  dilation_w: u32,
  groups: u32,
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
  // 契約検査（src/ops.ts）で groups は Cin / Cout を割り切る — 除算は厳密。
  let in_per_group = dims.channels_in / dims.groups;
  let out_per_group = dims.channels_out / dims.groups;
  var i = gid.x;
  while (i < dims.n) {
    let plane = dims.height_out * dims.width_out;
    let chunk = dims.channels_out * plane;
    let b = i / chunk;
    let rest = i % chunk;
    let oc = rest / plane;
    let pixel = rest % plane;
    let oy = pixel / dims.width_out;
    let ox = pixel % dims.width_out;
    // 入力側の開始位置は符号付き（padding ぶん負に出る）
    let origin_y = i32(oy * dims.stride_h) - i32(dims.padding_h);
    let origin_x = i32(ox * dims.stride_w) - i32(dims.padding_w);
    // 重みの第 2 軸は Cin/groups — グループ内の相対番号で引く
    let group_index = oc / out_per_group;
    let ic_base = group_index * in_per_group;
    // 出力チャネルの scale はループ不変 — 重みの要素ごとに引き直さない（ADR 0019）
    let wscale_v = wscale[oc];
    var acc = bias[oc];
    for (var ic_rel = 0u; ic_rel < in_per_group; ic_rel = ic_rel + 1u) {
      let x_base = (b * dims.channels_in + ic_base + ic_rel) * dims.height_in * dims.width_in;
      let w_base = (oc * in_per_group + ic_rel) * dims.kernel_h * dims.kernel_w;
      for (var kh = 0u; kh < dims.kernel_h; kh = kh + 1u) {
        let iy = origin_y + i32(kh * dims.dilation_h);
        // padding 域は 0 詰め — 加算せずに読み飛ばす
        if (iy < 0 || u32(iy) >= dims.height_in) {
          continue;
        }
        let row_base = x_base + u32(iy) * dims.width_in;
        let w_row = w_base + kh * dims.kernel_w;
        for (var kw = 0u; kw < dims.kernel_w; kw = kw + 1u) {
          let ix = origin_x + i32(kw * dims.dilation_w);
          if (ix >= 0 && u32(ix) < dims.width_in) {
            acc = acc + x[row_base + u32(ix)] * dequant(w_row + kw, wscale_v);
          }
        }
      }
    }
    out[i] = acc;
    i = i + step;
  }
}
