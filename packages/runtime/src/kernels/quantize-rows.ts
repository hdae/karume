/**
 * **Per-token symmetric int8 quantization** of activations (the front stage of the w8a8 execution
 * path — ADR 0025 / design in docs/research/2026-08-03-dp4a-w8a8-design.md §4.2).
 *
 * Quantizes a contiguous `[rows, dim]` layout per row (= token) and writes two outputs:
 *
 * ```
 * s        = max(rowmax(|x|) / 127, f32 tiny)      … xs[row] (per-token scale, f32)
 * q        = clamp(round(x / s), -127, +127)       … xq[row·dim/4 + …] (u32 packing 4 i8)
 * ```
 *
 * The shape matches the row-reduce family (**1 row = 1 workgroup (256), grid-stride over rows**),
 * and each row is walked twice (pass ① absmax → pass ② quantization). The row is read from global
 * memory a second time because shared memory cannot hold a whole row (the largest measured shape is
 * 32KB per row).
 *
 * MUST: **closed at ±127** (−128 is not used). The same rule as the weight side of ADR 0019: the
 * element with the largest magnitude is restored exactly and fake-quant is idempotent. Do not rely
 * on `pack4xI8Clamp` (saturates to [−128,127]) — use `pack4xI8` after an explicit `clamp(±127)`.
 * NOTE (detection limit, measured by fault injection): **the `clamp` branch itself is unreachable
 * with finite data**. Since `s = amax/127`, `|x/s| ≤ 127` holds structurally, and rounding error
 * cannot push it past 127.5. So the detector of a "lattice becomes ±128" regression is **the
 * denominator of the scale**, not the clamp: changing only the `clamp` bounds to ±128 keeps every
 * test green (making the denominator 128 turns 6 red). The clamp stays as a guard against
 * non-finite values and future changes to `s`, not as a carrier of the numeric contract.
 * MUST: NaN propagation of absmax is a **bit-pattern check** (ADR 0020). The driver's `max`
 * swallows NaN, so relying on the plain `max` gives a NaN row a finite scale and silently erases
 * the NaN of the whole row. The `max(…, tiny)` that builds `s` lets NaN through with an outer
 * `select` for the same reason.
 * MUST: the caller guarantees `dim % 4 == 0` (the i8 payload packs 4 per flat index, and this is
 * exactly the condition for a row to start on a word boundary — eligibility is decided in
 * src/runtime/recipe-builders/linear.ts and src/runtime/attention-route.ts). Here every one of the
 * `dim / 4` words is written on the assumption that there is no remainder (full-write).
 *
 * NOTE (measured 2026-08-03 on the development machine, Deno 2.9.4 + RTX 3080 Ti): WGSL `round`
 * **rounds half to even** (`round(0.5) = 0` / `round(1.5) = 2` / `round(-0.5) = -0` /
 * `round(2.5) = 2`), matching `torch.round`. The TS reference uses `roundTiesToEven`
 * (src/reference/i8a8.ts), not `Math.round` (half-up).
 *
 * ## Division precision (a design constraint found by measurement, 2026-08-03)
 *
 * WGSL f32 division is **not correctly rounded** (the spec allows 2.5 ULP; measured on the
 * development machine, `a / b` lowers to `a * rcp(b)` and differed by 1 ULP from IEEE correct
 * rounding in **55,605 of 200,000 samples**). Multiplication, addition and `fma` are correctly
 * rounded (0 cases in the same measurement). Two consequences:
 *
 * - MUST: write the scale's `amax / 127` **as the multiplication `amax * INV_127`** (`INV_127` is
 *   the f32 literal of 1/127). The development machine's compiler folds the constant divisor into
 *   the same multiplication, so the generated code does not change, but a driver that does not fold
 *   it drops to the 2.5 ULP division and `xs` diverges from the reference. **Determinism is
 *   established on the source side**.
 * - The `x / s` that builds `q` has a data-dependent divisor and cannot be replaced. So **for
 *   elements where `x/s` lies near a rounding boundary (a half-integer) the quantized value may
 *   move by ±1 step**. The atol=0 comparison of GPU and TS reference is a contract that holds only
 *   for "data far enough from the boundaries"; the test measures the margin and makes it a check
 *   (tests/gpu_i8a8_test.ts). In real models the ±1 step wobble shows up directly as a numeric
 *   difference (one reason the E2E tolerances are derived from measurements).
 */

import { IS_NAN_BITS_WGSL, NAN_MAX_WGSL } from "../codegen/numerics-wgsl.ts";
import { CodegenError } from "../codegen/errors.ts";
import { assertU32Params } from "../codegen/params.ts";

/** Width of the workgroup that folds one row (256, as in row reduce / softmax). */
export const QUANTIZE_ROWS_WORKGROUP_SIZE = 256;

/** Edge of the quantization lattice (**−128 is not used** — the MUST above). */
const QUANTIZE_ROWS_ABS_MAX = 127;

/** Smallest f32 normal (`torch.finfo(float32).tiny`). Floor of the scale of an all-zero row. */
const F32_TINY = 1.1754943508222875e-38;

/**
 * **Double-precision literal** of 1/127 (the WGSL side reads it correctly rounded to f32). The
 * source of truth for building the scale by multiplying with this constant instead of dividing
 * (the MUST above).
 */
const INV_ABS_MAX = 1 / QUANTIZE_ROWS_ABS_MAX;

export const QUANTIZE_ROWS_KEY = `quantize_rows:v1:f32>i8:pertoken:wg${QUANTIZE_ROWS_WORKGROUP_SIZE}`;

export const QUANTIZE_ROWS_WGSL: string = `// karume quantize_rows (per-row per-token symmetric i8: s = max(amax/127, tiny), q = clamp(round(x/s), ±127))
struct Params {
  rows: u32,
  dim: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> xq: array<u32>;
@group(0) @binding(3) var<storage, read_write> xs: array<f32>;

${IS_NAN_BITS_WGSL}

${NAN_MAX_WGSL}

var<workgroup> scratch: array<f32, ${QUANTIZE_ROWS_WORKGROUP_SIZE}>;

@compute @workgroup_size(${QUANTIZE_ROWS_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid3: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let lid = lid3.x;
  let dim = params.dim;
  // The caller guarantees dim % 4 == 0 (the i8 payload packs 4 per flat index)
  let quads = dim / 4u;
  var row = wid.x;
  while (row < params.rows) {
    let base = row * dim;

    // ① Absolute maximum of the row (NaN propagates by a bit-pattern check — plain max swallows it)
    var acc = 0.0;
    var i = lid;
    while (i < dim) {
      acc = nan_max(acc, abs(x[base + i]));
      i = i + ${QUANTIZE_ROWS_WORKGROUP_SIZE}u;
    }
    scratch[lid] = acc;
    workgroupBarrier();
    var stride = ${QUANTIZE_ROWS_WORKGROUP_SIZE / 2}u;
    while (stride > 0u) {
      if (lid < stride) {
        scratch[lid] = nan_max(scratch[lid], scratch[lid + stride]);
      }
      workgroupBarrier();
      stride = stride / 2u;
    }
    let amax = scratch[0u];
    // An all-zero row is exact: s = tiny → q = 0 → 0·tiny = 0. max swallows NaN, so the outer
    // select passes it.
    // MUST: **multiplication by 1/127**, not division by 127 (only multiplication is correctly
    // rounded — the MUST above)
    let s = select(max(amax * ${INV_ABS_MAX}, ${F32_TINY}), amax, is_nan_bits(amax));
    if (lid == 0u) {
      xs[row] = s;
    }

    // ② Quantize each 4 consecutive elements (quad) and pack them into one word
    let qbase = row * quads;
    var q = lid;
    while (q < quads) {
      let e = base + q * 4u;
      let v = vec4<f32>(x[e], x[e + 1u], x[e + 2u], x[e + 3u]) / s;
      let r = clamp(round(v), vec4<f32>(-${QUANTIZE_ROWS_ABS_MAX}.0), vec4<f32>(${QUANTIZE_ROWS_ABS_MAX}.0));
      xq[qbase + q] = pack4xI8(vec4<i32>(r));
      q = q + ${QUANTIZE_ROWS_WORKGROUP_SIZE}u;
    }

    // Make sure scratch[0] has been read before the next row overwrites scratch[lid]
    workgroupBarrier();
    row = row + nwg.x;
  }
}
`;

/**
 * Uniform Params (`{rows, dim}`). 16 bytes are allocated even though the content is 2 words (as in
 * softmax / reduce — allocating in 16-byte units is a project convention (the fusion pass fixes the
 * same form in ADR 0040), not a requirement of the spec).
 */
export const quantizeRowsParams = (rows: number, dim: number): Uint32Array<ArrayBuffer> => {
  assertU32Params("quantize_rows params", { rows, dim });
  if (dim < 1) {
    throw new CodegenError(`quantize_rows params: dim は正整数（${dim}）`);
  }
  if (dim % 4 !== 0) {
    throw new CodegenError(`quantize_rows params: dim は 4 の倍数（${dim}）`);
  }
  const params = new Uint32Array(4);
  params[0] = rows;
  params[1] = dim;
  return params;
};

/**
 * Workgroup assignment relative to the row length (**small-D variant** — perf-ledger P-1).
 *
 * The shape above is 1 row = 1 workgroup (256 lanes); with short rows most lanes idle (for D=128,
 * pass ① uses 128 lanes and pass ② 32 quads = 12.5% lane utilisation), and the fixed cost (an
 * 8-level tree reduce + barriers) is paid per row. In real shapes the QK front stage of fused
 * attention (D=128) is 87% of the quantized rows. So **the lane width is the smallest power of 2
 * not below pass ②'s quad count**, and the remaining lanes line up other rows in the same
 * workgroup and fold them (`rowsPerGroup = 256 / lanesPerRow`).
 *
 * The numeric contract does not change: the absmax reduction is `nan_max` (associative and
 * commutative), so changing the tree shape keeps amax bit-identical, and pass ② is per element.
 * Row lengths where one row uses all 256 lanes (quad ≥ 256 = dim ≥ 1024) keep exactly the old shape
 * (`rowsPerGroup = 1`, key and WGSL stay {@link QUANTIZE_ROWS_KEY} / {@link QUANTIZE_ROWS_WGSL} —
 * existing outputs do not move by a single byte).
 */
export type QuantizeRowsGeometry = {
  /** Lanes that fold one row (a power of 2, 1..256). */
  readonly lanesPerRow: number;
  /** Rows one workgroup lines up and folds (= 256 / lanesPerRow). */
  readonly rowsPerGroup: number;
};

export const quantizeRowsGeometry = (dim: number): QuantizeRowsGeometry => {
  if (!Number.isSafeInteger(dim) || dim < 1 || dim % 4 !== 0) {
    throw new CodegenError(`quantize_rows geometry: dim は 4 の倍数の正整数（${dim}）`);
  }
  const quads = dim / 4;
  let lanesPerRow = 1;
  while (lanesPerRow < quads && lanesPerRow < QUANTIZE_ROWS_WORKGROUP_SIZE) lanesPerRow *= 2;
  return { lanesPerRow, rowsPerGroup: QUANTIZE_ROWS_WORKGROUP_SIZE / lanesPerRow };
};

/**
 * Pipeline key per geometry. The old shape (1 row = 1 workgroup) keeps {@link QUANTIZE_ROWS_KEY}
 * as is (the name does not change, so it can be matched against past breakdowns in research), and
 * only the lined-up shape appends `:r<rows>w<width>` (the determinism contract: a different
 * geometry means different WGSL = a different key).
 */
export const quantizeRowsKey = (geometry: QuantizeRowsGeometry): string =>
  geometry.rowsPerGroup === 1
    ? QUANTIZE_ROWS_KEY
    : `${QUANTIZE_ROWS_KEY}:r${geometry.rowsPerGroup}w${geometry.lanesPerRow}`;

/** WGSL per geometry (the old shape is exactly {@link QUANTIZE_ROWS_WGSL}). */
export const quantizeRowsWgsl = (geometry: QuantizeRowsGeometry): string =>
  geometry.rowsPerGroup === 1 ? QUANTIZE_ROWS_WGSL : quantizeRowsGroupedWgsl(geometry);

/**
 * WGSL of the lined-up shape. One workgroup (256) is cut into R lane groups of width W, and group
 * `slot` handles row `rowBase + slot`. The row loop (`rowBase`) is workgroup-uniform, so the
 * barriers sit at the same places as in the old shape — a group without a row (at the tail) only
 * joins the barriers without touching memory. The tree reduce stays inside the group (width W,
 * log2(W) levels).
 */
const quantizeRowsGroupedWgsl = ({ lanesPerRow, rowsPerGroup }: QuantizeRowsGeometry): string =>
  `// karume quantize_rows (per-row per-token symmetric i8: s = max(amax/127, tiny), q = clamp(round(x/s), ±127)) — ${rowsPerGroup} rows × ${lanesPerRow} lanes / workgroup
struct Params {
  rows: u32,
  dim: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> xq: array<u32>;
@group(0) @binding(3) var<storage, read_write> xs: array<f32>;

${IS_NAN_BITS_WGSL}

${NAN_MAX_WGSL}

var<workgroup> scratch: array<f32, ${QUANTIZE_ROWS_WORKGROUP_SIZE}>;

@compute @workgroup_size(${QUANTIZE_ROWS_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid3: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let lid = lid3.x;
  let dim = params.dim;
  // The caller guarantees dim % 4 == 0 (the i8 payload packs 4 per flat index)
  let quads = dim / 4u;
  // Lane groups: width ${lanesPerRow} × ${rowsPerGroup} groups. lane = position in a group, slot = group
  let lane = lid % ${lanesPerRow}u;
  let slot = lid / ${lanesPerRow}u;
  var rowBase = wid.x * ${rowsPerGroup}u;
  while (rowBase < params.rows) {
    let row = rowBase + slot;
    // A group without a row at the tail does not read or write and only joins the barriers (the
    // row loop is workgroup-uniform)
    let has_row = row < params.rows;
    let base = row * dim;

    // ① Absolute maximum of the row (NaN propagates by a bit-pattern check — plain max swallows it)
    var acc = 0.0;
    if (has_row) {
      var i = lane;
      while (i < dim) {
        acc = nan_max(acc, abs(x[base + i]));
        i = i + ${lanesPerRow}u;
      }
    }
    scratch[lid] = acc;
    workgroupBarrier();
    var stride = ${Math.floor(lanesPerRow / 2)}u;
    while (stride > 0u) {
      if (lane < stride) {
        scratch[lid] = nan_max(scratch[lid], scratch[lid + stride]);
      }
      workgroupBarrier();
      stride = stride / 2u;
    }
    let amax = scratch[slot * ${lanesPerRow}u];
    // An all-zero row is exact: s = tiny → q = 0 → 0·tiny = 0. max swallows NaN, so the outer
    // select passes it.
    // MUST: **multiplication by 1/127**, not division by 127 (only multiplication is correctly
    // rounded — the MUST above)
    let s = select(max(amax * ${INV_ABS_MAX}, ${F32_TINY}), amax, is_nan_bits(amax));
    if (has_row && lane == 0u) {
      xs[row] = s;
    }

    // ② Quantize each 4 consecutive elements (quad) and pack them into one word
    if (has_row) {
      let qbase = row * quads;
      var q = lane;
      while (q < quads) {
        let e = base + q * 4u;
        let v = vec4<f32>(x[e], x[e + 1u], x[e + 2u], x[e + 3u]) / s;
        let r = clamp(round(v), vec4<f32>(-${QUANTIZE_ROWS_ABS_MAX}.0), vec4<f32>(${QUANTIZE_ROWS_ABS_MAX}.0));
        xq[qbase + q] = pack4xI8(vec4<i32>(r));
        q = q + ${lanesPerRow}u;
      }
    }

    // Make sure scratch has been read before the next row overwrites scratch[lid]
    workgroupBarrier();
    rowBase = rowBase + nwg.x * ${rowsPerGroup}u;
  }
}
`;
