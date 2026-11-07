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
@group(0) @binding(1) var<storage, read> x: array<u32>;
@group(0) @binding(2) var<storage, read_write> slot: array<u32>;
@group(0) @binding(3) var<uniform> lengths: Lengths;

fn slot_row(col: u32) -> u32 {
  return col;
}

// x >> shift, to nearest with ties to even (shift >= 1)
fn round_shift(x: u32, shift: u32) -> u32 {
  let kept = x >> shift;
  let rest = x & ((1u << shift) - 1u);
  let half = 1u << (shift - 1u);
  return kept + u32(rest > half || (rest == half && (kept & 1u) == 1u));
}

// **The only rounding point** of the slot: f32 bits to f16 bits
fn f16_bits(f: u32) -> u32 {
  let sign = (f >> 16u) & 0x8000u;
  let magnitude = f & 0x7fffffffu;
  if (magnitude > 0x7f800000u) {
    return sign | 0x7e00u; // NaN stays a quiet NaN
  }
  if (magnitude >= 0x47800000u) {
    return sign | 0x7c00u; // from 2^16 up, infinity
  }
  if (magnitude >= 0x38800000u) {
    return sign | round_shift(magnitude - 0x38000000u, 13u); // normal: the exponent rebased from 127 to 15
  }
  if (magnitude < 0x33000000u) {
    return sign; // below 2^-25 (f32 subnormals included) rounds to zero
  }
  // Subnormal in f16: the f32 significand in units of 2^-24
  return sign | round_shift((magnitude & 0x7fffffu) | 0x800000u, 126u - (magnitude >> 23u));
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
    slot[dst] = f16_bits(x[src]) | (f16_bits(x[src + 1u]) << 16u);
    i = i + stride;
  }
}
