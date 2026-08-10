// karume adaLN (layer_norm -> 1 + scale -> mul -> add, f32, staged 変調)
struct Params {
  rows: u32,
  dim: u32,
  eps: f32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read> scale_vec: array<f32>;
@group(0) @binding(5) var<storage, read> one: array<f32>;
@group(0) @binding(6) var<storage, read> shift: array<f32>;
@group(0) @binding(7) var<storage, read_write> out: array<f32>;

var<workgroup> scratch: array<f32, 256>;
var<workgroup> staged: array<u32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid3: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let lid = lid3.x;
  let dim = params.dim;
  let scale = 1.0 / f32(dim);
  // NOTE: 切り上げを (dim + 255) / 256 と書かない — dim = u32::MAX で桁溢れする。
  let blocks = dim / 256u +
    select(0u, 1u, dim % 256u != 0u);
  var row = wid.x;
  while (row < params.rows) {
    let base = row * dim;

    // ① 和 → 平均
    var acc = 0.0;
    var i = lid;
    while (i < dim) {
      acc = acc + x[base + i];
      i = i + 256u;
    }
    scratch[lid] = acc;
    workgroupBarrier();
    var stride = 128u;
    while (stride > 0u) {
      if (lid < stride) {
        scratch[lid] = scratch[lid] + scratch[lid + stride];
      }
      workgroupBarrier();
      stride = stride / 2u;
    }
    let mean = scratch[0u] * scale;
    // scratch の読み終わりを揃えてから ② で上書きする
    workgroupBarrier();

    // ② 偏差平方和 → 母分散（correction = 0）
    var sq = 0.0;
    var j = lid;
    while (j < dim) {
      let d = x[base + j] - mean;
      sq = sq + d * d;
      j = j + 256u;
    }
    scratch[lid] = sq;
    workgroupBarrier();
    var stride2 = 128u;
    while (stride2 > 0u) {
      if (lid < stride2) {
        scratch[lid] = scratch[lid] + scratch[lid + stride2];
      }
      workgroupBarrier();
      stride2 = stride2 / 2u;
    }
    let inv = 1.0 / sqrt(scratch[0u] * scale + params.eps);
    // 次の行が scratch[lid] を上書きする前に scratch[0] の読み終わりを揃える
    workgroupBarrier();

    var block = 0u;
    while (block < blocks) {
      let o = block * 256u + lid;
      // ③ 変調係数 1 + scale の実体化（素の列では add ノードが storage へ書く点）
      if (o < dim) {
        staged[lid] = bitcast<u32>(scale_vec[o] + one[0u]);
      }
      workgroupBarrier();
      // ④ 正規化（素の layer_norm と同一式）→ 乗算。素の mul と同じ位置で丸める。
      if (o < dim) {
        let modulation = bitcast<f32>(staged[lid]);
        staged[lid] = bitcast<u32>(((x[base + o] - mean) * inv * weight[o] + bias[o]) * modulation);
      }
      workgroupBarrier();
      // ⑤ shift の加算（素の add ノード）
      if (o < dim) {
        out[base + o] = bitcast<f32>(staged[lid]) + shift[o];
      }
      workgroupBarrier();
      block = block + 1u;
    }
    row = row + nwg.x;
  }
}
