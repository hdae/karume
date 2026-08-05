// karume conv1d (x[B,Cin,L] * W[Cout,Cin/groups,K] + b[Cout], f32, 重み f16 格納, 直接畳み込み)
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
  // 契約検査（src/ops.ts）で groups は Cin / Cout を割り切る — 除算は厳密。
  let in_per_group = dims.channels_in / dims.groups;
  let out_per_group = dims.channels_out / dims.groups;
  var i = gid.x;
  while (i < dims.n) {
    let plane = dims.channels_out * dims.length_out;
    let b = i / plane;
    let rest = i % plane;
    let oc = rest / dims.length_out;
    let ox = rest % dims.length_out;
    // 入力側の開始位置は符号付き（padding ぶん負に出る）
    let origin = i32(ox * dims.stride) - i32(dims.padding);
    // 重みの第 2 軸は Cin/groups — グループ内の相対番号で引く
    let group_index = oc / out_per_group;
    let ic_base = group_index * in_per_group;
    var acc = bias[oc];
    for (var ic_rel = 0u; ic_rel < in_per_group; ic_rel = ic_rel + 1u) {
      let x_base = (b * dims.channels_in + ic_base + ic_rel) * dims.length_in;
      let w_base = (oc * in_per_group + ic_rel) * dims.kernel;
      for (var k = 0u; k < dims.kernel; k = k + 1u) {
        let ix = origin + i32(k * dims.dilation);
        // padding 域は 0 詰め — 加算せずに読み飛ばす
        if (ix >= 0 && u32(ix) < dims.length_in) {
          acc = acc + x[x_base + u32(ix)] * dequant(w_base + k);
        }
      }
    }
    out[i] = acc;
    i = i + step;
  }
}
