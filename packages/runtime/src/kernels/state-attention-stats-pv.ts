/**
 * Fusion of the row statistics and PV of states-form attention (ADR 0102).
 * The row statistics are recomputed per depth tile, so it is used only for few rows and a short
 * column cap.
 *
 * v2 (perf-ledger K-91): the PV loop is the unrolled scan shared with ③' (`statePvScanWgsl` — the
 * section "The unrolled KV loop" of state-attention.ts). MUST: bit-identical to v1 word for word;
 * the statistics part is v1's unchanged (the 256 flat lanes, the `nan_max` loop, the 128 → 1 trees,
 * the `maximum != neg_inf` guards and `1/Σ`). NOTE: the identity rests on the backend compiling the
 * same expression the same way; it is checked word for word against the v1 WGSL on each backend
 * (tests/gpu_state_attention_fold_test.ts).
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
  statePvScanWgsl,
  stateSlotRowWgsl,
  stateVariantKeyPart,
} from "./state-attention.ts";

/** The readonly form keeps its own family. Eligibility uses only the static M and the column cap. */
export const stateStatsPvEligible = (chunkRows: number, colCap: number): boolean =>
  chunkRows <= 8 && colCap <= 1024;

// v2: the PV loop is unrolled (bit-identical to v1; the WGSL bytes changed, and one key MUST map to
// byte-identical WGSL)
export const stateStatsPvKey = (sliding: boolean, gqa: boolean): string =>
  "attention_state_stats_pv:v2:f32:wg16x16" + stateVariantKeyPart(sliding, gqa);

/** The binding numbers and geometry of the existing parallel PV, minus binding 2 (row statistics). */
export const stateStatsPvWgsl = (sliding: boolean, gqa: boolean): string =>
  `// karume attention_state_stats_pv (states-form O = P @ V, f32, KV-parallel reduction${
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
  // Pad rows (row ≥ Q) do not scan live and are exactly 0 (the same contract as ③). The local row is
  // workgroup-uniform, so returning before the barrier is allowed
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
  // Row statistics use the same indices and summation order as the original 256 lanes. A mapping
  // separate from the PV KV lanes.
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
  // Reuse scratch for the PV partial sums only after the denominator has been read.
  workgroupBarrier();
  // Per-lane partial sum (ascending col, stride KV_LANES). A lane whose d is out of range does not
  // scan and contributes 0 (it does not return, so that it takes part in the barrier)
  var acc = 0.0;
  if (in_depth) {
${statePvScanWgsl()}
  }
  scratch[lane * ${STATE_ATTENTION_TILE_X}u + lid.x] = acc;
  workgroupBarrier();
  // Fixed-order tree reduction (stride 8 → 4 → 2 → 1) — the basis of determinism
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
