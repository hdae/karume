// karume quantize_rows (行ごとの per-token symmetric i8: s = max(amax/127, tiny), q = clamp(round(x/s), ±127)) — 256 行 × 1 レーン / workgroup
struct Params {
  rows: u32,
  dim: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> xq: array<u32>;
@group(0) @binding(3) var<storage, read_write> xs: array<f32>;

fn is_nan_bits(x: f32) -> bool {
  return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u;
}

fn nan_max(a: f32, b: f32) -> f32 {
  return select(select(max(a, b), b, is_nan_bits(b)), a, is_nan_bits(a));
}

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid3: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let lid = lid3.x;
  let dim = params.dim;
  // 呼び出し側が dim % 4 == 0 を保証する（i8 ペイロードは平坦添字 4 詰め）
  let quads = dim / 4u;
  // レーン組: 幅 1 × 256 組。組の中の位置 lane と組の番号 slot
  let lane = lid % 1u;
  let slot = lid / 1u;
  var rowBase = wid.x * 256u;
  while (rowBase < params.rows) {
    let row = rowBase + slot;
    // 末尾で行が無い組は読み書きをせず barrier だけ揃える（行ループは workgroup 一様）
    let has_row = row < params.rows;
    let base = row * dim;

    // ① 行の絶対値最大（NaN はビット列判定で伝播 — 素の max は NaN を飲む）
    var acc = 0.0;
    if (has_row) {
      var i = lane;
      while (i < dim) {
        acc = nan_max(acc, abs(x[base + i]));
        i = i + 1u;
      }
    }
    scratch[lid] = acc;
    workgroupBarrier();
    var stride = 0u;
    while (stride > 0u) {
      if (lane < stride) {
        scratch[lid] = nan_max(scratch[lid], scratch[lid + stride]);
      }
      workgroupBarrier();
      stride = stride / 2u;
    }
    let amax = scratch[slot * 1u];
    // 全ゼロ行は s = tiny → q = 0 → 0·tiny = 0 で厳密。NaN は max に飲まれるので外殻で通す。
    // MUST: 127 での除算ではなく **1/127 との乗算**（乗算だけが正しく丸められる — 上の MUST）
    let s = select(max(amax * 0.007874015748031496, 1.1754943508222875e-38), amax, is_nan_bits(amax));
    if (has_row && lane == 0u) {
      xs[row] = s;
    }

    // ② 4 連続要素（quad）ごとに量子化して 1 語へ詰める
    if (has_row) {
      let qbase = row * quads;
      var q = lane;
      while (q < quads) {
        let e = base + q * 4u;
        let v = vec4<f32>(x[e], x[e + 1u], x[e + 2u], x[e + 3u]) / s;
        let r = clamp(round(v), vec4<f32>(-127.0), vec4<f32>(127.0));
        xq[qbase + q] = pack4xI8(vec4<i32>(r));
        q = q + 1u;
      }
    }

    // 次の行が scratch[lid] を上書きする前に scratch の読み終わりを揃える
    workgroupBarrier();
    rowBase = rowBase + nwg.x * 256u;
  }
}
