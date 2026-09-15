/**
 * states 形 attention の行統計と PV の融合（ADR 0102）。
 * 深さタイルごとに行統計を再計算するため、少数行・短い列上限だけに使う。
 */
import { IS_NAN_BITS_WGSL, NAN_MAX_WGSL } from "../codegen/numerics-wgsl.ts";
import {
  kvPlaneWgsl,
  STATE_ATTENTION_TILE_X,
  STATE_LENGTHS_STRUCT,
  STATE_PARAMS_STRUCT,
  STATE_PV_KV_LANES,
  stateEffectiveRowsWgsl,
  stateLiveWgsl,
  stateSlotRowWgsl,
  stateVariantKeyPart,
} from "./state-attention.ts";

/** readonly 形は別族を維持する。判定は静的な M と列上限だけ。 */
export const stateStatsPvEligible = (chunkRows: number, colCap: number): boolean =>
  chunkRows <= 8 && colCap <= 1024;

export const stateStatsPvKey = (sliding: boolean, gqa: boolean): string =>
  "attention_state_stats_pv:v1:f32:wg16x16" + stateVariantKeyPart(sliding, gqa);

/** binding 2 の行統計を除き、従来の並列 PV と同じ束縛番号・幾何を使う。 */
export const stateStatsPvWgsl = (sliding: boolean, gqa: boolean): string =>
  `// karume attention_state_stats_pv (states 形の O = P @ V, f32, KV 並列縮約${
    sliding ? ", sliding window" : ""
  }${gqa ? ", GQA" : ""})
${STATE_PARAMS_STRUCT}
${STATE_LENGTHS_STRUCT}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> s: array<f32>;
@group(0) @binding(3) var<storage, read> ins_v: array<f32>;
@group(0) @binding(4) var<storage, read> slot_v: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;
@group(0) @binding(6) var<uniform> lengths: Lengths;

${stateSlotRowWgsl(sliding)}

${stateLiveWgsl(sliding)}

${stateEffectiveRowsWgsl()}

${IS_NAN_BITS_WGSL}
${NAN_MAX_WGSL}

var<workgroup> scratch: array<f32, ${STATE_ATTENTION_TILE_X * STATE_PV_KV_LANES}>;

@compute @workgroup_size(${STATE_ATTENTION_TILE_X}, ${STATE_PV_KV_LANES})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let d = wid.x * ${STATE_ATTENTION_TILE_X}u + lid.x;
  let local_row = wid.y;
  let lane = lid.y;
  let z = wid.z;
  let in_depth = d < params.depth;
  let at = (z * params.chunk_rows + params.row_offset + local_row) * params.depth + d;
  // pad 行（row ≥ Q）は live を走査せず厳密 0（③ と同じ契約）。局所行は workgroup 一様なので
  // barrier の手前で返してよい
  if (local_row >= effective_rows(lengths.query)) {
    if (lane == 0u && in_depth) {
      out[at] = 0.0;
    }
    return;
  }
  let past = lengths.past;
  let live = live_columns(past, lengths.query);
  let base_col = column_base(past);
  let kv_plane = ${kvPlaneWgsl(gqa)};
  let s_row = z * params.rows_block + local_row;
  let s_base = s_row * params.col_cap;
  // 行統計は元の 256 lane と同じ添字・加算順。PV の KV lane とは別の写像。
  let flat = lid.y * 16u + lid.x;
  let neg_inf = bitcast<f32>(params.neg_inf);
  var hi = neg_inf;
  for (var i = flat; i < live; i = i + 256u) {
    hi = nan_max(hi, s[s_base + i]);
  }
  scratch[flat] = hi;
  workgroupBarrier();
  for (var step = 128u; step > 0u; step = step / 2u) {
    if (flat < step) {
      scratch[flat] = nan_max(scratch[flat], scratch[flat + step]);
    }
    workgroupBarrier();
  }
  let maximum = scratch[0u];
  workgroupBarrier();
  var sum = 0.0;
  if (maximum != neg_inf) {
    for (var i = flat; i < live; i = i + 256u) {
      sum = sum + exp(s[s_base + i] - maximum);
    }
  }
  scratch[flat] = sum;
  workgroupBarrier();
  for (var step = 128u; step > 0u; step = step / 2u) {
    if (flat < step) {
      scratch[flat] = scratch[flat] + scratch[flat + step];
    }
    workgroupBarrier();
  }
  var amax = 0.0;
  var inv = 0.0;
  if (maximum != neg_inf) {
    amax = maximum;
    inv = 1.0 / scratch[0u];
  }
  // 分母を読み終えてから scratch を PV の部分和へ使い回す。
  workgroupBarrier();
  // レーンごとの部分和（col 昇順・stride KV_LANES）。d が範囲外のレーンは走査せず 0 を寄与する
  // （barrier に参加させるため return しない）
  var acc = 0.0;
  if (in_depth) {
    for (var cl = lane; cl < live; cl = cl + ${STATE_PV_KV_LANES}u) {
      let col = base_col + cl;
      let p = exp(s[s_base + cl] - amax) * inv;
      var value = 0.0;
      if (col < past) {
        value = slot_v[(kv_plane * params.capacity + slot_row(col)) * params.depth + d];
      } else {
        value = ins_v[(kv_plane * params.chunk_rows + (col - past)) * params.depth + d];
      }
      acc = acc + p * value;
    }
  }
  scratch[lane * ${STATE_ATTENTION_TILE_X}u + lid.x] = acc;
  workgroupBarrier();
  // 固定順の木縮約（stride 8 → 4 → 2 → 1）— 決定性の根拠
  var stride = ${STATE_PV_KV_LANES / 2}u;
  while (stride > 0u) {
    if (lane < stride) {
      let mine = lane * ${STATE_ATTENTION_TILE_X}u + lid.x;
      scratch[mine] = scratch[mine] + scratch[(lane + stride) * ${STATE_ATTENTION_TILE_X}u + lid.x];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }
  if (lane == 0u && in_depth) {
    out[at] = scratch[lid.x];
  }
}
`;
