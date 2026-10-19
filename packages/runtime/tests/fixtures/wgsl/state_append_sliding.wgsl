// karume state_append (this step's k / v into the state slot, f32, ring write of the sliding window)
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
  return col % params.capacity;
}

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let past = lengths.past;
  let query = lengths.query;
  // Total work is **B·Hkv·Q·D** (pad rows are not in the index space = not written)
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
    // With Q > C the ring wraps and several logical rows map to the same physical row. **Only
    // the last logical row** writes (letting every row write in parallel makes the winner
    // implementation-dependent = silent nondeterminism). The modulus is **capacity**, as in
    // slot_row (cutting by window would, when C > W, silently skip rows that alias nothing)
    if (row + params.capacity >= query) {
      slot[dst] = x[src];
    }
    i = i + stride;
  }
}
