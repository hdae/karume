enable subgroups, subgroup_size_control;
// karume linear gemv K subgroup32 (i8, 16 lanes/output)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
// 行頭が 16 B 整列なのは k % 16 == 0 から（適格判定が保証する）
@group(0) @binding(2) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<storage, read> wscale: array<f32>;

@compute @workgroup_size(128) @subgroup_size(32)
fn main(
  @builtin(subgroup_invocation_id) sub: u32,
  @builtin(subgroup_id) sg: u32,
  @builtin(workgroup_id) wg: vec3<u32>,
) {
  let lane = sub % 16u;
  let col = wg.x * 8u + sg * 2u + sub / 16u;
  var acc = 0.0;
  // 端の列も全レーンがshuffleへ参加する。local IDとsubgroup IDの配置を仮定しない。
  if (col < dims.n) {
    let units = dims.k / 16u;
    let row_base = col * units;
  // 出力チャネルの scale はループ不変 — 重みの要素ごとに引き直さない（ADR 0019）
  let wscale_v = wscale[col];
    for (var unit = lane; unit < units; unit += 16u) {
    let unitt = unit;
    let pwt = w[row_base + unitt];
    let xqt = wg.y * (dims.k / 4u) + unitt * 4u;
    let bt_0 = unpack4xI8(pwt.x);
    let xat_0 = x[xqt + 0u];
    acc = acc + xat_0.x * (f32(bt_0.x) * wscale_v);
    acc = acc + xat_0.y * (f32(bt_0.y) * wscale_v);
    acc = acc + xat_0.z * (f32(bt_0.z) * wscale_v);
    acc = acc + xat_0.w * (f32(bt_0.w) * wscale_v);
    let bt_1 = unpack4xI8(pwt.y);
    let xat_1 = x[xqt + 1u];
    acc = acc + xat_1.x * (f32(bt_1.x) * wscale_v);
    acc = acc + xat_1.y * (f32(bt_1.y) * wscale_v);
    acc = acc + xat_1.z * (f32(bt_1.z) * wscale_v);
    acc = acc + xat_1.w * (f32(bt_1.w) * wscale_v);
    let bt_2 = unpack4xI8(pwt.z);
    let xat_2 = x[xqt + 2u];
    acc = acc + xat_2.x * (f32(bt_2.x) * wscale_v);
    acc = acc + xat_2.y * (f32(bt_2.y) * wscale_v);
    acc = acc + xat_2.z * (f32(bt_2.z) * wscale_v);
    acc = acc + xat_2.w * (f32(bt_2.w) * wscale_v);
    let bt_3 = unpack4xI8(pwt.w);
    let xat_3 = x[xqt + 3u];
    acc = acc + xat_3.x * (f32(bt_3.x) * wscale_v);
    acc = acc + xat_3.y * (f32(bt_3.y) * wscale_v);
    acc = acc + xat_3.z * (f32(bt_3.z) * wscale_v);
    acc = acc + xat_3.w * (f32(bt_3.w) * wscale_v);
    }
  }
  for (var width = 8u; width > 0u; width /= 2u) {
    let other = subgroupShuffleXor(acc, width);
    if (lane < width) { acc = acc + other; }
  }
  if (col < dims.n && lane == 0u) {
    out[wg.y * dims.n + col] = acc + bias[col];
  }
}
