// karume upsample_bilinear2d (NCHW bilinear, align_corners=true, f32, full-write)
struct Params {
  n: u32,
  in_h: u32,
  in_w: u32,
  out_h: u32,
  out_w: u32,
  // ホストが f32 で割った (in - 1) / (out - 1)（出力長 1 の軸は 0）
  scale_h: f32,
  scale_w: f32,
  reserved: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let stride = nwg.x * 256u;
  let out_hw = params.out_h * params.out_w;
  let in_hw = params.in_h * params.in_w;
  var i = gid.x;
  while (i < params.n) {
    // N と C は補間に関与しないので 1 本の平面添字へ畳む（rank を params に載せない）。
    let ox = i % params.out_w;
    let oy = (i / params.out_w) % params.out_h;
    let plane = i / out_hw;

    let ry = params.scale_h * f32(oy);
    let y0 = u32(ry);
    var y1 = y0;
    if (y0 + 1u < params.in_h) {
      y1 = y0 + 1u;
    }
    let ly1 = ry - f32(y0);
    let ly0 = 1.0 - ly1;

    let rx = params.scale_w * f32(ox);
    let x0 = u32(rx);
    var x1 = x0;
    if (x0 + 1u < params.in_w) {
      x1 = x0 + 1u;
    }
    let lx1 = rx - f32(x0);
    let lx0 = 1.0 - lx1;

    let row0 = plane * in_hw + y0 * params.in_w;
    let row1 = plane * in_hw + y1 * params.in_w;
    out[i] = ly0 * (lx0 * x[row0 + x0] + lx1 * x[row0 + x1])
      + ly1 * (lx0 * x[row1 + x0] + lx1 * x[row1 + x1]);
    i = i + stride;
  }
}
