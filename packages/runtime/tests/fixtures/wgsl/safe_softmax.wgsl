// karume safe_softmax (last dim, f32, safe-softmax: exp(x - amax), 行 max が -inf の行は全 0)
struct Params {
  rows: u32,
  dim: u32,
  neg_inf: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid3: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let lid = lid3.x;
  let dim = params.dim;
  let neg_inf = bitcast<f32>(params.neg_inf);
  var row = wid.x;
  while (row < params.rows) {
    let base = row * dim;

    // ① 行の最大値（safe-softmax の減算項 — identity は -inf）
    var hi = neg_inf;
    var i = lid;
    while (i < dim) {
      hi = max(hi, x[base + i]);
      i = i + 256u;
    }
    scratch[lid] = hi;
    workgroupBarrier();
    var stride = 128u;
    while (stride > 0u) {
      if (lid < stride) {
        scratch[lid] = max(scratch[lid], scratch[lid + stride]);
      }
      workgroupBarrier();
      stride = stride / 2u;
    }
    let row_max = scratch[0u];
    // 全要素 -inf の行（torch のガードが 0 を返す行）— 減算項を 0 にして NaN を作らない
    let empty = row_max == neg_inf;
    let amax = select(row_max, 0.0, empty);
    // scratch の読み終わりを揃えてから ② で上書きする
    workgroupBarrier();

    // ② Σ exp(x - amax)（最大要素が exp(0) = 1 を出すので分母は必ず 1 以上）
    var acc = 0.0;
    var j = lid;
    while (j < dim) {
      acc = acc + exp(x[base + j] - amax);
      j = j + 256u;
    }
    scratch[lid] = acc;
    workgroupBarrier();
    var stride2 = 128u;
    while (stride2 > 0u) {
      if (lid < stride2) {
        scratch[lid] = scratch[lid] + scratch[lid + stride2];
      }
      workgroupBarrier();
      stride2 = stride2 / 2u;
    }
    let inv = 1.0 / scratch[0u];
    // 次の行が scratch[lid] を上書きする前に scratch[0] の読み終わりを揃える
    workgroupBarrier();

    // ③ 書き出し（exp は ② と同じ引数 — 同じ値が返るので決定的）
    var o = lid;
    while (o < dim) {
      out[base + o] = select(exp(x[base + o] - amax) * inv, 0.0, empty);
      o = o + 256u;
    }
    row = row + nwg.x;
  }
}
