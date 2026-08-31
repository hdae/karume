// karume gru_scan (reverse, f32, 1 workgroup = 1 バッチ要素 / 1 lane = 1 隠れユニット)
struct Dims {
  time: u32,
  batch: u32,
  hidden: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> gi: array<f32>;
@group(0) @binding(2) var<storage, read> h0: array<f32>;
@group(0) @binding(3) var<storage, read> w_hh: array<f32>;
@group(0) @binding(4) var<storage, read> b_hh: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;

var<workgroup> h_shared: array<f32, 256>;
// 丸め障壁の中継（f32 のビット列をそのまま置く）。lane は自分の枠しか触らない。
var<workgroup> stage: array<u32, 256>;

fn sigmoid_stable(x: f32) -> f32 {
  let t = exp(-abs(x));
  return select(1.0 / (1.0 + t), t / (1.0 + t), x < 0.0);
}

fn is_nan_bits(x: f32) -> bool {
  return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u;
}

fn tanh_stable(x: f32) -> f32 {
  let lo = select(x, -9.5, x < -9.5);
  let t = select(lo, 9.5, x > 9.5);
  return select(tanh(t), x, is_nan_bits(x));
}

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid3: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let lid = lid3.x;
  let hidden = dims.hidden;
  let gates = hidden * 3u;
  let lane_used = lid < hidden;
  // W_hh は [3H, H] の行優先。ゲートは r / z / n の順に H 行ずつ並ぶ。
  let row_r = lid * hidden;
  let row_z = (hidden + lid) * hidden;
  let row_n = (hidden * 2u + lid) * hidden;
  var item = wid.x;
  while (item < dims.batch) {
    if (lane_used) {
      h_shared[lid] = h0[item * hidden + lid];
    }
    workgroupBarrier();
    var step = 0u;
    while (step < dims.time) {
      let t = dims.time - 1u - step;
      let gi_base = (t * dims.batch + item) * gates;
      var h_prev = 0.0;
      var gate_z = 0.0;
      if (lane_used) {
        h_prev = h_shared[lid];
        var acc_r = 0.0;
        var acc_z = 0.0;
        var acc_n = 0.0;
        for (var k = 0u; k < hidden; k = k + 1u) {
          let hk = h_shared[k];
          acc_r = acc_r + w_hh[row_r + k] * hk;
          acc_z = acc_z + w_hh[row_z + k] * hk;
          acc_n = acc_n + w_hh[row_n + k] * hk;
        }
        // bias は last（GEMM の epilogue と同じ）。ゲートの足し順は隠れ側が第 1 引数。
        let gate_r = sigmoid_stable((acc_r + b_hh[lid]) + gi[gi_base + lid]);
        gate_z = sigmoid_stable((acc_z + b_hh[hidden + lid]) + gi[gi_base + hidden + lid]);
        let gh_n = acc_n + b_hh[hidden * 2u + lid];
        // 丸め障壁 ①: 分解経路の mul は別 dispatch なので、fma 縮約させない
        stage[lid] = bitcast<u32>(gh_n * gate_r);
      }
      workgroupBarrier();
      var cand = 0.0;
      if (lane_used) {
        // n = tanh(i_n + h_n·r) — 入力側が第 1 引数
        cand = tanh_stable(gi[gi_base + hidden * 2u + lid] + bitcast<f32>(stage[lid]));
        // 丸め障壁 ②: h' = (h − n)·z + n の mul と add の間
        stage[lid] = bitcast<u32>((h_prev - cand) * gate_z);
      }
      workgroupBarrier();
      if (lane_used) {
        let h_next = bitcast<f32>(stage[lid]) + cand;
        h_shared[lid] = h_next;
        out[(t * dims.batch + item) * hidden + lid] = h_next;
      }
      // 次ステップの縮約が h_shared を読む前に、この書き込みを揃える
      workgroupBarrier();
      step = step + 1u;
    }
    item = item + nwg.x;
  }
}
