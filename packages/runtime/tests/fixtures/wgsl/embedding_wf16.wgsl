// karume embedding (out[..., h] = weight[index[...], h], f32 / index i32, f16 weight storage)
struct Dims {
  n: u32,
  hidden: u32,
  vocab: u32,
  oob: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> weight: array<u32>;
@group(0) @binding(2) var<storage, read> index: array<i32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

// decode of f16 storage: element i = unpack2x16float(weight[i / 2])[i % 2]
// (the parity of the flat index picks the half)
fn dequant(i: u32) -> f32 {
  let pair = unpack2x16float(weight[i >> 1u]);
  return select(pair.x, pair.y, (i & 1u) == 1u);
}

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let stride = nwg.x * 256u;
  var i = gid.x;
  while (i < dims.n) {
    let row = i / dims.hidden;
    let col = i % dims.hidden;
    let pick = index[row];
    // an index outside the contract contaminates with NaN instead of returning another row (the
    // ruling in the kernel doc)
    if (pick < 0 || u32(pick) >= dims.vocab) {
      out[i] = bitcast<f32>(dims.oob);
    } else {
      out[i] = dequant(u32(pick) * dims.hidden + col);
    }
    i = i + stride;
  }
}
