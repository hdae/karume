// karume attention_stats (行ごとの m = amax(S) と inv = 1/Σexp(S - m), f32, S は 1 回読みでレジスタ保持（1 スレッド 16 要素）)
struct Params {
  rows: u32,
  dim: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> s: array<f32>;
@group(0) @binding(2) var<storage, read_write> stats: array<f32>;

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid3: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let lid = lid3.x;
  let dim = params.dim;
  let i0 = lid;
  let i1 = lid + 256u;
  let i2 = lid + 512u;
  let i3 = lid + 768u;
  let i4 = lid + 1024u;
  let i5 = lid + 1280u;
  let i6 = lid + 1536u;
  let i7 = lid + 1792u;
  let i8 = lid + 2048u;
  let i9 = lid + 2304u;
  let i10 = lid + 2560u;
  let i11 = lid + 2816u;
  let i12 = lid + 3072u;
  let i13 = lid + 3328u;
  let i14 = lid + 3584u;
  let i15 = lid + 3840u;
  var row = wid.x;
  while (row < params.rows) {
    let base = row * dim;

    // ① 行の最大値（safe-softmax の減算項）
    var hi = -3.402823466e38;
    var c0 = 0.0;
    if (i0 < dim) {
      c0 = s[base + i0];
      hi = max(hi, c0);
    }
    var c1 = 0.0;
    if (i1 < dim) {
      c1 = s[base + i1];
      hi = max(hi, c1);
    }
    var c2 = 0.0;
    if (i2 < dim) {
      c2 = s[base + i2];
      hi = max(hi, c2);
    }
    var c3 = 0.0;
    if (i3 < dim) {
      c3 = s[base + i3];
      hi = max(hi, c3);
    }
    var c4 = 0.0;
    if (i4 < dim) {
      c4 = s[base + i4];
      hi = max(hi, c4);
    }
    var c5 = 0.0;
    if (i5 < dim) {
      c5 = s[base + i5];
      hi = max(hi, c5);
    }
    var c6 = 0.0;
    if (i6 < dim) {
      c6 = s[base + i6];
      hi = max(hi, c6);
    }
    var c7 = 0.0;
    if (i7 < dim) {
      c7 = s[base + i7];
      hi = max(hi, c7);
    }
    var c8 = 0.0;
    if (i8 < dim) {
      c8 = s[base + i8];
      hi = max(hi, c8);
    }
    var c9 = 0.0;
    if (i9 < dim) {
      c9 = s[base + i9];
      hi = max(hi, c9);
    }
    var c10 = 0.0;
    if (i10 < dim) {
      c10 = s[base + i10];
      hi = max(hi, c10);
    }
    var c11 = 0.0;
    if (i11 < dim) {
      c11 = s[base + i11];
      hi = max(hi, c11);
    }
    var c12 = 0.0;
    if (i12 < dim) {
      c12 = s[base + i12];
      hi = max(hi, c12);
    }
    var c13 = 0.0;
    if (i13 < dim) {
      c13 = s[base + i13];
      hi = max(hi, c13);
    }
    var c14 = 0.0;
    if (i14 < dim) {
      c14 = s[base + i14];
      hi = max(hi, c14);
    }
    var c15 = 0.0;
    if (i15 < dim) {
      c15 = s[base + i15];
      hi = max(hi, c15);
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
    let amax = scratch[0u];
    // scratch の読み終わりを揃えてから ② で上書きする
    workgroupBarrier();

    // ② Σ exp(S - amax)（最大要素が exp(0) = 1 を出すので分母は必ず 1 以上）
    var acc = 0.0;
    if (i0 < dim) {
      acc = acc + exp(c0 - amax);
    }
    if (i1 < dim) {
      acc = acc + exp(c1 - amax);
    }
    if (i2 < dim) {
      acc = acc + exp(c2 - amax);
    }
    if (i3 < dim) {
      acc = acc + exp(c3 - amax);
    }
    if (i4 < dim) {
      acc = acc + exp(c4 - amax);
    }
    if (i5 < dim) {
      acc = acc + exp(c5 - amax);
    }
    if (i6 < dim) {
      acc = acc + exp(c6 - amax);
    }
    if (i7 < dim) {
      acc = acc + exp(c7 - amax);
    }
    if (i8 < dim) {
      acc = acc + exp(c8 - amax);
    }
    if (i9 < dim) {
      acc = acc + exp(c9 - amax);
    }
    if (i10 < dim) {
      acc = acc + exp(c10 - amax);
    }
    if (i11 < dim) {
      acc = acc + exp(c11 - amax);
    }
    if (i12 < dim) {
      acc = acc + exp(c12 - amax);
    }
    if (i13 < dim) {
      acc = acc + exp(c13 - amax);
    }
    if (i14 < dim) {
      acc = acc + exp(c14 - amax);
    }
    if (i15 < dim) {
      acc = acc + exp(c15 - amax);
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
    // MUST: 逆数はここで作る（③ で割り算に戻すと softmax のパス③と演算が変わる）
    let inv = 1.0 / scratch[0u];
    if (lid == 0u) {
      stats[row * 2u] = amax;
      stats[row * 2u + 1u] = inv;
    }
    // 次の行が scratch[lid] を上書きする前に scratch[0] の読み終わりを揃える
    workgroupBarrier();
    row = row + nwg.x;
  }
}
