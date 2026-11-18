// karume expert_linear (y[t, j] = weight[index[t, j]] · x[t (, j)], f32, i4 weight storage with f16 steps, 32 cols / wg, 4 words prefetched, the rows of an expert grouped 8 / 4 / 2 / 1 to a walk)
struct Dims {
  rows: u32,
  n: u32,
  k: u32,
  picks: u32,
  shared_x: u32,
  experts: u32,
  oob: u32,
  pad: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> index: array<i32>;
// Row starts are 16 B aligned because k % 32 == 0 (guaranteed by expertLinearParams)
@group(0) @binding(3) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<storage, read> wscale: array<u32>;

// step s of an f16 scale: half s % 2 of word s / 2, widened to f32 (exact)
fn step_at(s: u32) -> f32 {
  let pair = unpack2x16float(wscale[s >> 1u]);
  return select(pair.x, pair.y, (s & 1u) == 1u);
}
// One weight word's multiply-adds onto `initial`, in the one-row form's element order and spelling
fn expert_word(pwa: vec4<u32>, xqa: u32, wsa: f32, initial: f32) -> f32 {
  var acc = initial;
    let ba_0 = unpack4xU8(pwa.x);
    let xaa_0 = x[xqa + 0u];
    let xba_0 = x[xqa + 1u];
    acc = acc + xaa_0.x * (f32(i32(ba_0.x & 0xFu) - 8) * wsa);
    acc = acc + xaa_0.y * (f32(i32(ba_0.x >> 4u) - 8) * wsa);
    acc = acc + xaa_0.z * (f32(i32(ba_0.y & 0xFu) - 8) * wsa);
    acc = acc + xaa_0.w * (f32(i32(ba_0.y >> 4u) - 8) * wsa);
    acc = acc + xba_0.x * (f32(i32(ba_0.z & 0xFu) - 8) * wsa);
    acc = acc + xba_0.y * (f32(i32(ba_0.z >> 4u) - 8) * wsa);
    acc = acc + xba_0.z * (f32(i32(ba_0.w & 0xFu) - 8) * wsa);
    acc = acc + xba_0.w * (f32(i32(ba_0.w >> 4u) - 8) * wsa);
    let ba_1 = unpack4xU8(pwa.y);
    let xaa_1 = x[xqa + 2u];
    let xba_1 = x[xqa + 3u];
    acc = acc + xaa_1.x * (f32(i32(ba_1.x & 0xFu) - 8) * wsa);
    acc = acc + xaa_1.y * (f32(i32(ba_1.x >> 4u) - 8) * wsa);
    acc = acc + xaa_1.z * (f32(i32(ba_1.y & 0xFu) - 8) * wsa);
    acc = acc + xaa_1.w * (f32(i32(ba_1.y >> 4u) - 8) * wsa);
    acc = acc + xba_1.x * (f32(i32(ba_1.z & 0xFu) - 8) * wsa);
    acc = acc + xba_1.y * (f32(i32(ba_1.z >> 4u) - 8) * wsa);
    acc = acc + xba_1.z * (f32(i32(ba_1.w & 0xFu) - 8) * wsa);
    acc = acc + xba_1.w * (f32(i32(ba_1.w >> 4u) - 8) * wsa);
    let ba_2 = unpack4xU8(pwa.z);
    let xaa_2 = x[xqa + 4u];
    let xba_2 = x[xqa + 5u];
    acc = acc + xaa_2.x * (f32(i32(ba_2.x & 0xFu) - 8) * wsa);
    acc = acc + xaa_2.y * (f32(i32(ba_2.x >> 4u) - 8) * wsa);
    acc = acc + xaa_2.z * (f32(i32(ba_2.y & 0xFu) - 8) * wsa);
    acc = acc + xaa_2.w * (f32(i32(ba_2.y >> 4u) - 8) * wsa);
    acc = acc + xba_2.x * (f32(i32(ba_2.z & 0xFu) - 8) * wsa);
    acc = acc + xba_2.y * (f32(i32(ba_2.z >> 4u) - 8) * wsa);
    acc = acc + xba_2.z * (f32(i32(ba_2.w & 0xFu) - 8) * wsa);
    acc = acc + xba_2.w * (f32(i32(ba_2.w >> 4u) - 8) * wsa);
    let ba_3 = unpack4xU8(pwa.w);
    let xaa_3 = x[xqa + 6u];
    let xba_3 = x[xqa + 7u];
    acc = acc + xaa_3.x * (f32(i32(ba_3.x & 0xFu) - 8) * wsa);
    acc = acc + xaa_3.y * (f32(i32(ba_3.x >> 4u) - 8) * wsa);
    acc = acc + xaa_3.z * (f32(i32(ba_3.y & 0xFu) - 8) * wsa);
    acc = acc + xaa_3.w * (f32(i32(ba_3.y >> 4u) - 8) * wsa);
    acc = acc + xba_3.x * (f32(i32(ba_3.z & 0xFu) - 8) * wsa);
    acc = acc + xba_3.y * (f32(i32(ba_3.z >> 4u) - 8) * wsa);
    acc = acc + xba_3.z * (f32(i32(ba_3.w & 0xFu) - 8) * wsa);
    acc = acc + xba_3.w * (f32(i32(ba_3.w >> 4u) - 8) * wsa);
  return acc;
}

@compute @workgroup_size(32)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let col = gid.x;
  // No shared memory and no barrier, so edge workgroups may return early
  if (col >= dims.n) {
    return;
  }
  let units = dims.k / 32u;
  let quads = dims.k / 4u;
  var row = gid.y;
  while (row < dims.rows) {
    let pick = index[row];
    // an id outside [0, E) breaks the planner's invariant: NaN instead of another expert's row
    if (pick < 0 || u32(pick) >= dims.experts) {
      out[row * dims.n + col] = bitcast<f32>(dims.oob);
    } else {
      let wrow = u32(pick) * dims.n + col;
      // The rows that picked this expert share the walk of its weight row, in the thread of the
      // first of them: a row that an earlier row leads is written by that row's thread
      var leads = true;
      for (var earlier = 0u; earlier < row; earlier = earlier + 1u) {
        if (index[earlier] == pick) {
          leads = false;
          break;
        }
      }
      if (leads) {
        let row_base = wrow * units;
        let scale_base = wrow * (dims.k >> 5u);
        var next = row;
        while (next < dims.rows) {
          var members: array<u32, 8>;
          var count = 0u;
          while (next < dims.rows && count < 8u) {
            if (index[next] == pick) {
              members[count] = next;
              count = count + 1u;
            }
            next = next + 1u;
          }
          var start = 0u;
          for (; count - start >= 8u; start = start + 8u) {
            let m0 = members[start + 0u];
            let m1 = members[start + 1u];
            let m2 = members[start + 2u];
            let m3 = members[start + 3u];
            let m4 = members[start + 4u];
            let m5 = members[start + 5u];
            let m6 = members[start + 6u];
            let m7 = members[start + 7u];
            let xr0 = select(m0, m0 / dims.picks, dims.shared_x == 1u) * quads;
            let xr1 = select(m1, m1 / dims.picks, dims.shared_x == 1u) * quads;
            let xr2 = select(m2, m2 / dims.picks, dims.shared_x == 1u) * quads;
            let xr3 = select(m3, m3 / dims.picks, dims.shared_x == 1u) * quads;
            let xr4 = select(m4, m4 / dims.picks, dims.shared_x == 1u) * quads;
            let xr5 = select(m5, m5 / dims.picks, dims.shared_x == 1u) * quads;
            let xr6 = select(m6, m6 / dims.picks, dims.shared_x == 1u) * quads;
            let xr7 = select(m7, m7 / dims.picks, dims.shared_x == 1u) * quads;
            var acc0 = 0.0;
            var acc1 = 0.0;
            var acc2 = 0.0;
            var acc3 = 0.0;
            var acc4 = 0.0;
            var acc5 = 0.0;
            var acc6 = 0.0;
            var acc7 = 0.0;
            var unit = 0u;
            for (; unit + 4u <= units; unit = unit + 4u) {
              let unit0 = unit + 0u;
              let pw0 = w[row_base + unit0];
              let ws0 = step_at(scale_base + ((unit0 * 32u) >> 5u));
              let xo0 = unit0 * 8u;
              let unit1 = unit + 1u;
              let pw1 = w[row_base + unit1];
              let ws1 = step_at(scale_base + ((unit1 * 32u) >> 5u));
              let xo1 = unit1 * 8u;
              let unit2 = unit + 2u;
              let pw2 = w[row_base + unit2];
              let ws2 = step_at(scale_base + ((unit2 * 32u) >> 5u));
              let xo2 = unit2 * 8u;
              let unit3 = unit + 3u;
              let pw3 = w[row_base + unit3];
              let ws3 = step_at(scale_base + ((unit3 * 32u) >> 5u));
              let xo3 = unit3 * 8u;
              acc0 = expert_word(pw0, xr0 + xo0, ws0, acc0);
              acc1 = expert_word(pw0, xr1 + xo0, ws0, acc1);
              acc2 = expert_word(pw0, xr2 + xo0, ws0, acc2);
              acc3 = expert_word(pw0, xr3 + xo0, ws0, acc3);
              acc4 = expert_word(pw0, xr4 + xo0, ws0, acc4);
              acc5 = expert_word(pw0, xr5 + xo0, ws0, acc5);
              acc6 = expert_word(pw0, xr6 + xo0, ws0, acc6);
              acc7 = expert_word(pw0, xr7 + xo0, ws0, acc7);
              acc0 = expert_word(pw1, xr0 + xo1, ws1, acc0);
              acc1 = expert_word(pw1, xr1 + xo1, ws1, acc1);
              acc2 = expert_word(pw1, xr2 + xo1, ws1, acc2);
              acc3 = expert_word(pw1, xr3 + xo1, ws1, acc3);
              acc4 = expert_word(pw1, xr4 + xo1, ws1, acc4);
              acc5 = expert_word(pw1, xr5 + xo1, ws1, acc5);
              acc6 = expert_word(pw1, xr6 + xo1, ws1, acc6);
              acc7 = expert_word(pw1, xr7 + xo1, ws1, acc7);
              acc0 = expert_word(pw2, xr0 + xo2, ws2, acc0);
              acc1 = expert_word(pw2, xr1 + xo2, ws2, acc1);
              acc2 = expert_word(pw2, xr2 + xo2, ws2, acc2);
              acc3 = expert_word(pw2, xr3 + xo2, ws2, acc3);
              acc4 = expert_word(pw2, xr4 + xo2, ws2, acc4);
              acc5 = expert_word(pw2, xr5 + xo2, ws2, acc5);
              acc6 = expert_word(pw2, xr6 + xo2, ws2, acc6);
              acc7 = expert_word(pw2, xr7 + xo2, ws2, acc7);
              acc0 = expert_word(pw3, xr0 + xo3, ws3, acc0);
              acc1 = expert_word(pw3, xr1 + xo3, ws3, acc1);
              acc2 = expert_word(pw3, xr2 + xo3, ws3, acc2);
              acc3 = expert_word(pw3, xr3 + xo3, ws3, acc3);
              acc4 = expert_word(pw3, xr4 + xo3, ws3, acc4);
              acc5 = expert_word(pw3, xr5 + xo3, ws3, acc5);
              acc6 = expert_word(pw3, xr6 + xo3, ws3, acc6);
              acc7 = expert_word(pw3, xr7 + xo3, ws3, acc7);
            }
            for (; unit < units; unit = unit + 1u) {
              let unitt = unit;
              let pwt = w[row_base + unitt];
              let wst = step_at(scale_base + ((unitt * 32u) >> 5u));
              let xot = unitt * 8u;
              acc0 = expert_word(pwt, xr0 + xot, wst, acc0);
              acc1 = expert_word(pwt, xr1 + xot, wst, acc1);
              acc2 = expert_word(pwt, xr2 + xot, wst, acc2);
              acc3 = expert_word(pwt, xr3 + xot, wst, acc3);
              acc4 = expert_word(pwt, xr4 + xot, wst, acc4);
              acc5 = expert_word(pwt, xr5 + xot, wst, acc5);
              acc6 = expert_word(pwt, xr6 + xot, wst, acc6);
              acc7 = expert_word(pwt, xr7 + xot, wst, acc7);
            }
            out[m0 * dims.n + col] = acc0;
            out[m1 * dims.n + col] = acc1;
            out[m2 * dims.n + col] = acc2;
            out[m3 * dims.n + col] = acc3;
            out[m4 * dims.n + col] = acc4;
            out[m5 * dims.n + col] = acc5;
            out[m6 * dims.n + col] = acc6;
            out[m7 * dims.n + col] = acc7;
          }
          for (; count - start >= 4u; start = start + 4u) {
            let m0 = members[start + 0u];
            let m1 = members[start + 1u];
            let m2 = members[start + 2u];
            let m3 = members[start + 3u];
            let xr0 = select(m0, m0 / dims.picks, dims.shared_x == 1u) * quads;
            let xr1 = select(m1, m1 / dims.picks, dims.shared_x == 1u) * quads;
            let xr2 = select(m2, m2 / dims.picks, dims.shared_x == 1u) * quads;
            let xr3 = select(m3, m3 / dims.picks, dims.shared_x == 1u) * quads;
            var acc0 = 0.0;
            var acc1 = 0.0;
            var acc2 = 0.0;
            var acc3 = 0.0;
            var unit = 0u;
            for (; unit + 4u <= units; unit = unit + 4u) {
              let unit0 = unit + 0u;
              let pw0 = w[row_base + unit0];
              let ws0 = step_at(scale_base + ((unit0 * 32u) >> 5u));
              let xo0 = unit0 * 8u;
              let unit1 = unit + 1u;
              let pw1 = w[row_base + unit1];
              let ws1 = step_at(scale_base + ((unit1 * 32u) >> 5u));
              let xo1 = unit1 * 8u;
              let unit2 = unit + 2u;
              let pw2 = w[row_base + unit2];
              let ws2 = step_at(scale_base + ((unit2 * 32u) >> 5u));
              let xo2 = unit2 * 8u;
              let unit3 = unit + 3u;
              let pw3 = w[row_base + unit3];
              let ws3 = step_at(scale_base + ((unit3 * 32u) >> 5u));
              let xo3 = unit3 * 8u;
              acc0 = expert_word(pw0, xr0 + xo0, ws0, acc0);
              acc1 = expert_word(pw0, xr1 + xo0, ws0, acc1);
              acc2 = expert_word(pw0, xr2 + xo0, ws0, acc2);
              acc3 = expert_word(pw0, xr3 + xo0, ws0, acc3);
              acc0 = expert_word(pw1, xr0 + xo1, ws1, acc0);
              acc1 = expert_word(pw1, xr1 + xo1, ws1, acc1);
              acc2 = expert_word(pw1, xr2 + xo1, ws1, acc2);
              acc3 = expert_word(pw1, xr3 + xo1, ws1, acc3);
              acc0 = expert_word(pw2, xr0 + xo2, ws2, acc0);
              acc1 = expert_word(pw2, xr1 + xo2, ws2, acc1);
              acc2 = expert_word(pw2, xr2 + xo2, ws2, acc2);
              acc3 = expert_word(pw2, xr3 + xo2, ws2, acc3);
              acc0 = expert_word(pw3, xr0 + xo3, ws3, acc0);
              acc1 = expert_word(pw3, xr1 + xo3, ws3, acc1);
              acc2 = expert_word(pw3, xr2 + xo3, ws3, acc2);
              acc3 = expert_word(pw3, xr3 + xo3, ws3, acc3);
            }
            for (; unit < units; unit = unit + 1u) {
              let unitt = unit;
              let pwt = w[row_base + unitt];
              let wst = step_at(scale_base + ((unitt * 32u) >> 5u));
              let xot = unitt * 8u;
              acc0 = expert_word(pwt, xr0 + xot, wst, acc0);
              acc1 = expert_word(pwt, xr1 + xot, wst, acc1);
              acc2 = expert_word(pwt, xr2 + xot, wst, acc2);
              acc3 = expert_word(pwt, xr3 + xot, wst, acc3);
            }
            out[m0 * dims.n + col] = acc0;
            out[m1 * dims.n + col] = acc1;
            out[m2 * dims.n + col] = acc2;
            out[m3 * dims.n + col] = acc3;
          }
          for (; count - start >= 2u; start = start + 2u) {
            let m0 = members[start + 0u];
            let m1 = members[start + 1u];
            let xr0 = select(m0, m0 / dims.picks, dims.shared_x == 1u) * quads;
            let xr1 = select(m1, m1 / dims.picks, dims.shared_x == 1u) * quads;
            var acc0 = 0.0;
            var acc1 = 0.0;
            var unit = 0u;
            for (; unit + 4u <= units; unit = unit + 4u) {
              let unit0 = unit + 0u;
              let pw0 = w[row_base + unit0];
              let ws0 = step_at(scale_base + ((unit0 * 32u) >> 5u));
              let xo0 = unit0 * 8u;
              let unit1 = unit + 1u;
              let pw1 = w[row_base + unit1];
              let ws1 = step_at(scale_base + ((unit1 * 32u) >> 5u));
              let xo1 = unit1 * 8u;
              let unit2 = unit + 2u;
              let pw2 = w[row_base + unit2];
              let ws2 = step_at(scale_base + ((unit2 * 32u) >> 5u));
              let xo2 = unit2 * 8u;
              let unit3 = unit + 3u;
              let pw3 = w[row_base + unit3];
              let ws3 = step_at(scale_base + ((unit3 * 32u) >> 5u));
              let xo3 = unit3 * 8u;
              acc0 = expert_word(pw0, xr0 + xo0, ws0, acc0);
              acc1 = expert_word(pw0, xr1 + xo0, ws0, acc1);
              acc0 = expert_word(pw1, xr0 + xo1, ws1, acc0);
              acc1 = expert_word(pw1, xr1 + xo1, ws1, acc1);
              acc0 = expert_word(pw2, xr0 + xo2, ws2, acc0);
              acc1 = expert_word(pw2, xr1 + xo2, ws2, acc1);
              acc0 = expert_word(pw3, xr0 + xo3, ws3, acc0);
              acc1 = expert_word(pw3, xr1 + xo3, ws3, acc1);
            }
            for (; unit < units; unit = unit + 1u) {
              let unitt = unit;
              let pwt = w[row_base + unitt];
              let wst = step_at(scale_base + ((unitt * 32u) >> 5u));
              let xot = unitt * 8u;
              acc0 = expert_word(pwt, xr0 + xot, wst, acc0);
              acc1 = expert_word(pwt, xr1 + xot, wst, acc1);
            }
            out[m0 * dims.n + col] = acc0;
            out[m1 * dims.n + col] = acc1;
          }
          for (; count - start >= 1u; start = start + 1u) {
            let m0 = members[start + 0u];
            let xr0 = select(m0, m0 / dims.picks, dims.shared_x == 1u) * quads;
            var acc0 = 0.0;
            var unit = 0u;
            for (; unit + 4u <= units; unit = unit + 4u) {
              let unit0 = unit + 0u;
              let pw0 = w[row_base + unit0];
              let ws0 = step_at(scale_base + ((unit0 * 32u) >> 5u));
              let xo0 = unit0 * 8u;
              let unit1 = unit + 1u;
              let pw1 = w[row_base + unit1];
              let ws1 = step_at(scale_base + ((unit1 * 32u) >> 5u));
              let xo1 = unit1 * 8u;
              let unit2 = unit + 2u;
              let pw2 = w[row_base + unit2];
              let ws2 = step_at(scale_base + ((unit2 * 32u) >> 5u));
              let xo2 = unit2 * 8u;
              let unit3 = unit + 3u;
              let pw3 = w[row_base + unit3];
              let ws3 = step_at(scale_base + ((unit3 * 32u) >> 5u));
              let xo3 = unit3 * 8u;
              acc0 = expert_word(pw0, xr0 + xo0, ws0, acc0);
              acc0 = expert_word(pw1, xr0 + xo1, ws1, acc0);
              acc0 = expert_word(pw2, xr0 + xo2, ws2, acc0);
              acc0 = expert_word(pw3, xr0 + xo3, ws3, acc0);
            }
            for (; unit < units; unit = unit + 1u) {
              let unitt = unit;
              let pwt = w[row_base + unitt];
              let wst = step_at(scale_base + ((unitt * 32u) >> 5u));
              let xot = unitt * 8u;
              acc0 = expert_word(pwt, xr0 + xot, wst, acc0);
            }
            out[m0 * dims.n + col] = acc0;
          }
        }
      }
    }
    row = row + nwg.y;
  }
}
