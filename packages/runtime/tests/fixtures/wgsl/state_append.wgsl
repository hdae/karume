// karume state_append (今 step の k / v を state スロットへ, f32)
struct Params {
  kv_planes: u32,
  chunk_rows: u32,
  depth: u32,
  capacity: u32,
  window: u32,
}
struct Lengths {
  past: u32,
  query: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> slot: array<f32>;
@group(0) @binding(3) var<uniform> lengths: Lengths;

fn slot_row(col: u32) -> u32 {
  return col;
}

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let past = lengths.past;
  let query = lengths.query;
  // 総仕事は **B·Hkv·Q·D**（pad 行は添字空間に入らない = 書かれない）
  let total = params.kv_planes * query * params.depth;
  let stride = nwg.x * 256u;
  var i = gid.x;
  while (i < total) {
    let d = i % params.depth;
    let rest = i / params.depth;
    let row = rest % query;
    let kv_plane = rest / query;
    let src = (kv_plane * params.chunk_rows + row) * params.depth + d;
    let dst = (kv_plane * params.capacity + slot_row(past + row)) * params.depth + d;
    slot[dst] = x[src];
    i = i + stride;
  }
}
