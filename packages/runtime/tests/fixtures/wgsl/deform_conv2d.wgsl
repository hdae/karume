// karume deform_conv2d (DCNv2: x[B,Cin,H,W] * W[Cout,Cin,Kh,Kw] + b[Cout], f32, 直接畳み込み)
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
  padding_h: u32,
  padding_w: u32,
  oob: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> offsets: array<f32>;
@group(0) @binding(4) var<storage, read> modulator: array<f32>;
@group(0) @binding(5) var<storage, read> bias: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;

// NaN のビット列判定（ADR 0020）— 指数部が全 1 かつ仮数部が非 0。比較演算に寄せると
// ドライバの畳み込みで判定ごと消える。
fn is_nan(v: f32) -> bool {
  return (bitcast<u32>(v) & 0x7fffffffu) > 0x7f800000u;
}

// 入力平面 `plane` の双線形サンプル。範囲外はゼロ埋め（4 隅個別）・NaN は伝播。
fn deform_sample(plane: u32, sy: f32, sx: f32) -> f32 {
  if (is_nan(sy) || is_nan(sx)) {
    return bitcast<f32>(dims.oob);
  }
  // 正の形の範囲判定。これを通れば floor(sy) は [-1, H-1] なので i32 変換は必ず定義される。
  let inside = sy > -1.0 && sy < f32(dims.height_in)
    && sx > -1.0 && sx < f32(dims.width_in);
  if (!inside) {
    return 0.0;
  }
  let base_y = floor(sy);
  let base_x = floor(sx);
  let y0 = i32(base_y);
  let x0 = i32(base_x);
  let ly1 = sy - base_y;
  let lx1 = sx - base_x;
  let ly0 = 1.0 - ly1;
  let lx0 = 1.0 - lx1;
  let last_y = i32(dims.height_in) - 1;
  let last_x = i32(dims.width_in) - 1;
  var v00 = 0.0;
  var v01 = 0.0;
  var v10 = 0.0;
  var v11 = 0.0;
  if (y0 >= 0) {
    let row = plane + u32(y0) * dims.width_in;
    if (x0 >= 0) {
      v00 = x[row + u32(x0)];
    }
    if (x0 + 1 <= last_x) {
      v01 = x[row + u32(x0 + 1)];
    }
  }
  if (y0 + 1 <= last_y) {
    let row = plane + u32(y0 + 1) * dims.width_in;
    if (x0 >= 0) {
      v10 = x[row + u32(x0)];
    }
    if (x0 + 1 <= last_x) {
      v11 = x[row + u32(x0 + 1)];
    }
  }
  // 重みと加算順は torchvision 逐語（w1*v1 + w2*v2 + w3*v3 + w4*v4）。
  return ly0 * lx0 * v00 + ly0 * lx1 * v01 + ly1 * lx0 * v10 + ly1 * lx1 * v11;
}

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let step = nwg.x * 256u;
  let taps = dims.kernel_h * dims.kernel_w;
  let plane_out = dims.height_out * dims.width_out;
  let plane_in = dims.height_in * dims.width_in;
  var i = gid.x;
  while (i < dims.n) {
    let chunk = dims.channels_out * plane_out;
    let b = i / chunk;
    let rest = i % chunk;
    let oc = rest / plane_out;
    let pixel = rest % plane_out;
    let oy = pixel / dims.width_out;
    let ox = pixel % dims.width_out;
    // 入力側の基準位置は符号付き（padding ぶん負に出る）。stride / dilation は 1 固定。
    let origin_y = i32(oy) - i32(dims.padding_h);
    let origin_x = i32(ox) - i32(dims.padding_w);
    // offset / mask は B と出力画素で引く（offset_groups == 1 なのでグループ軸は無い）。
    let offset_base = b * 2u * taps * plane_out + pixel;
    let mask_base = b * taps * plane_out + pixel;
    var acc = bias[oc];
    for (var ic = 0u; ic < dims.channels_in; ic = ic + 1u) {
      let x_base = (b * dims.channels_in + ic) * plane_in;
      let w_base = (oc * dims.channels_in + ic) * taps;
      for (var kh = 0u; kh < dims.kernel_h; kh = kh + 1u) {
        for (var kw = 0u; kw < dims.kernel_w; kw = kw + 1u) {
          let tap = kh * dims.kernel_w + kw;
          // 偶数チャネル = y / 奇数チャネル = x
          let sy = f32(origin_y + i32(kh)) + offsets[offset_base + 2u * tap * plane_out];
          let sx = f32(origin_x + i32(kw)) + offsets[offset_base + (2u * tap + 1u) * plane_out];
          let m = modulator[mask_base + tap * plane_out];
          acc = acc + (m * deform_sample(x_base, sy, sx)) * w[w_base + tap];
        }
      }
    }
    out[i] = acc;
    i = i + step;
  }
}
