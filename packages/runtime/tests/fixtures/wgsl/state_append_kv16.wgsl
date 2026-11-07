// karume state_append (this step's k / v into the state slot, f32, KV in f16 storage, written a word at a time)
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
@group(0) @binding(2) var<storage, read_write> slot: array<u32>;
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
  // Total work is **B·Hkv·Q·D/2** words (pad rows are not in the index space = not written). One
  // invocation owns one word, because writing half a word is a read-modify-write that races
  let words = params.depth / 2u;
  let total = params.kv_planes * query * words;
  let stride = nwg.x * 256u;
  var i = gid.x;
  while (i < total) {
    let w = i % words;
    let rest = i / words;
    let row = rest % query;
    let kv_plane = rest / query;
    let src = (kv_plane * params.chunk_rows + row) * params.depth + w * 2u;
    let dst = (kv_plane * params.capacity + slot_row(past + row)) * words + w;
    // **This pack2x16float is the only rounding point** of the slot
    slot[dst] = pack2x16float(vec2<f32>(x[src], x[src + 1u]));
    i = i + stride;
  }
}
