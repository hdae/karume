// karume transpose_tiled (f32 [batch, s, c] → [batch, c, s] through a 32×33 workgroup tile)
struct Params {
  s: u32,
  c: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

var<workgroup> tile: array<f32, 1056>;

@compute @workgroup_size(32, 8)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let c0 = wid.x * 32u;
  let s0 = wid.y * 32u;
  let plane = wid.z * params.s * params.c;
  // Load: lane x walks c (a coalesced read of one source row), y steps through the block's rows
  for (var r = lid.y; r < 32u; r = r + 8u) {
    let s = s0 + r;
    let c = c0 + lid.x;
    if (s < params.s && c < params.c) {
      tile[r * 33u + lid.x] = src[plane + s * params.c + c];
    }
  }
  workgroupBarrier();
  // Store: lane x walks s (a coalesced write of one destination row)
  for (var r = lid.y; r < 32u; r = r + 8u) {
    let c = c0 + r;
    let s = s0 + lid.x;
    if (s < params.s && c < params.c) {
      dst[plane + c * params.s + s] = tile[lid.x * 33u + r];
    }
  }
}
