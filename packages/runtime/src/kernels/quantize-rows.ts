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
 * `dim / 4` words is written on the assumption that there is no remainder (full-write). The one
 * exception is the padded form ({@link quantizeRowsPaddedWgsl} — the online attention's Vᵀ at
 * `N % 4 != 0`), which owns exactly the row lengths that are not multiples of 4.
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

import { gridStrideWorkgroups } from "../codegen/dispatch.ts";
import { type PrologueUnaryOp, prologueUnaryWgsl } from "../codegen/elementwise.ts";
import { IS_NAN_BITS_WGSL, NAN_MAX_WGSL } from "../codegen/numerics-wgsl.ts";
import { CodegenError } from "../codegen/errors.ts";
import { assertRowLoopBound, assertU32Params } from "../codegen/params.ts";

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

/** The per-token scale from the row's absolute maximum `amax` (the MUSTs above) — shared by every form. */
const SCALE_WGSL = `select(max(amax * ${INV_ABS_MAX}, ${F32_TINY}), amax, is_nan_bits(amax))`;

/** Rounds a vec4 of `x / s` onto the lattice closed at ±127 (the MUST above) — shared by every form. */
const latticeWgsl = (v: string): string =>
  `clamp(round(${v}), vec4<f32>(-${QUANTIZE_ROWS_ABS_MAX}.0), vec4<f32>(${QUANTIZE_ROWS_ABS_MAX}.0))`;

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
    let s = ${SCALE_WGSL};
    if (lid == 0u) {
      xs[row] = s;
    }

    // ② Quantize each 4 consecutive elements (quad) and pack them into one word
    let qbase = row * quads;
    var q = lid;
    while (q < quads) {
      let e = base + q * 4u;
      let v = vec4<f32>(x[e], x[e + 1u], x[e + 2u], x[e + 3u]) / s;
      let r = ${latticeWgsl("v")};
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
  // MUST: the in-row loops step by at most the workgroup size (the row form, the lanes of the
  // grouped forms and the channel-major form share these params).
  assertRowLoopBound("quantize_rows params", { dim }, QUANTIZE_ROWS_WORKGROUP_SIZE);
  if (dim < 1) {
    throw new CodegenError(`quantize_rows params: dim must be a positive integer (${dim})`);
  }
  if (dim % 4 !== 0) {
    throw new CodegenError(`quantize_rows params: dim must be a multiple of 4 (${dim})`);
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
    throw new CodegenError(
      `quantize_rows geometry: dim must be a positive integer multiple of 4 (${dim})`,
    );
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
 * Where the rows of the input sit. `rows` = contiguous `[rows, dim]` (every form so far). `bshd` =
 * the attention's q / k as the projections write them, token-major `[B, S, H, D]`, read so that the
 * outputs (`xq`, `xs`) come out head-major `[B·H, S]` exactly as the plain form writes them from a
 * materialised `[B, H, S, D]` permute (the attention absorbs that permute — perf-ledger F-3).
 *
 * Only the read address changes, the per-row arithmetic is the plain form's, so the outputs are
 * bit-identical to "permute → quantize_rows".
 */
export type QuantizeRowsLayout = "rows" | "bshd";

/** The text a layout adds to the lined-up form (`rows` adds nothing: its bytes stay as they were). */
const BSHD_TEXT: Readonly<
  Record<
    QuantizeRowsLayout,
    { readonly header: string; readonly params: string; readonly base: string }
  >
> = {
  rows: { header: "", params: "", base: "    let base = row * dim;" },
  bshd: {
    header: ", rows read from a token-major [B, S, H, D] input",
    params: "\n  seq: u32,\n  heads: u32,",
    base: `    // Output row (b·H + h)·S + s reads token s, head h of batch b
    let bh = row / params.seq;
    let base = ((bh / params.heads * params.seq + row % params.seq) * params.heads + bh % params.heads) * dim;`,
  },
};

/** The word count and pass ② of the lined-up form for rows of `dim % 4 == 0` (every form but the padded one). */
const QUADS_TEXT = `  // The caller guarantees dim % 4 == 0 (the i8 payload packs 4 per flat index)
  let quads = dim / 4u;`;
const WORD_TEXT = `        let e = base + q * 4u;
        let v = vec4<f32>(x[e], x[e + 1u], x[e + 2u], x[e + 3u]) / s;
        let r = ${latticeWgsl("v")};`;
/** Where word `q` of row `row` is stored: rows of `quads` words, one after the other. */
const ROW_STORE_TEXT = {
  base: "      let qbase = row * quads;",
  at: "qbase + q",
} as const;

/**
 * The text of the padded form ({@link quantizeRowsPaddedWgsl}): `ceil(dim / 4)` words per row, and
 * in the last word the elements from `dim` on read a clamped in-row element and are forced to 0
 * (`select` after the lattice, so a NaN row cannot leave a non-zero padding byte either).
 */
const PADDED_TEXT = {
  header: ", rows of any length (the last word zero-padded)",
  quads: `  // Rows of any length: ceil(dim / 4) words per row, the bytes from element dim on are 0
  let quads = (dim + 3u) / 4u;`,
  word: `        let e = base + q * 4u;
        // In the last word the elements from dim on are padding: clamped reads, bytes forced to 0
        let last = base + dim - 1u;
        let inside = vec4<u32>(0u, 1u, 2u, 3u) + q * 4u < vec4<u32>(dim);
        let v = vec4<f32>(x[e], x[min(e + 1u, last)], x[min(e + 2u, last)], x[min(e + 3u, last)]) / s;
        let r = select(vec4<f32>(0.0), ${latticeWgsl("v")}, inside);`,
} as const;

/**
 * The text of the tiled form ({@link quantizeRowsTiledWgsl}): the key order of the online attention
 * (attention-online-i8a8.ts `AttentionOnlineKeyWords` `tiled`). A row is cut into tiles of 64
 * elements, `16 · ceil(dim / 64)` words per row; word `q` packs the elements
 * `64·(q/16) + q%16 + 16·i` (i = 0..3) — the four keys one lane of the attention's score phase
 * takes — and the elements from `dim` on read a clamped in-row element and are forced to 0, as in
 * the padded form. The words of four consecutive rows are interleaved (`[row/4][word][row%4]`), so
 * the consumer reads the same word of four rows as one `vec4<u32>`.
 *
 * Each byte is the plain form's: the same scale, the same lattice of the same `x / s`; only which
 * four elements share a word, and where the word is stored, differ.
 */
const TILED_TEXT = {
  header:
    ", rows cut into 64-element tiles (word q = the elements 64·(q/16) + q%16 + 16·i), four rows interleaved",
  quads: `  // 16 words per 64-element tile; the bytes from element dim on are 0
  let quads = ((dim + 63u) / 64u) * 16u;`,
  word: `        // Word q = the elements 64·(q/16) + q%16 + 16·i; from dim on: clamped reads, bytes forced to 0
        let last = base + dim - 1u;
        let e = (q / 16u) * 64u + q % 16u;
        let keys = vec4<u32>(e, e + 16u, e + 32u, e + 48u);
        let inside = keys < vec4<u32>(dim);
        let v = vec4<f32>(x[min(base + keys.x, last)], x[min(base + keys.y, last)], x[min(base + keys.z, last)], x[min(base + keys.w, last)]) / s;
        let r = select(vec4<f32>(0.0), ${latticeWgsl("v")}, inside);`,
  // MUST: rows % 4 == 0 (quantizeRowsTiledParams) — a group of four rows shares each vec4
  base: "      let qbase = (row / 4u) * quads * 4u + row % 4u;",
  at: "qbase + q * 4u",
} as const;

/** How the lined-up form packs a row into words: the plain quads, the padded last word, or the tiled key order. */
type QuantizeRowsWords = "quads" | "padded" | "tiled";

/**
 * WGSL of the lined-up shape. One workgroup (256) is cut into R lane groups of width W, and group
 * `slot` handles row `rowBase + slot`. The row loop (`rowBase`) is workgroup-uniform, so the
 * barriers sit at the same places as in the old shape — a group without a row (at the tail) only
 * joins the barriers without touching memory. The tree reduce stays inside the group (width W,
 * log2(W) levels).
 */
const quantizeRowsGroupedWgsl = (
  { lanesPerRow, rowsPerGroup }: QuantizeRowsGeometry,
  layout: QuantizeRowsLayout = "rows",
  words: QuantizeRowsWords = "quads",
): string =>
  `// karume quantize_rows (per-row per-token symmetric i8: s = max(amax/127, tiny), q = clamp(round(x/s), ±127)) — ${rowsPerGroup} rows × ${lanesPerRow} lanes / workgroup${BSHD_TEXT[layout].header}${
    words === "padded" ? PADDED_TEXT.header : words === "tiled" ? TILED_TEXT.header : ""
  }
struct Params {
  rows: u32,
  dim: u32,${BSHD_TEXT[layout].params}
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
${words === "padded" ? PADDED_TEXT.quads : words === "tiled" ? TILED_TEXT.quads : QUADS_TEXT}
  // Lane groups: width ${lanesPerRow} × ${rowsPerGroup} groups. lane = position in a group, slot = group
  let lane = lid % ${lanesPerRow}u;
  let slot = lid / ${lanesPerRow}u;
  var rowBase = wid.x * ${rowsPerGroup}u;
  while (rowBase < params.rows) {
    let row = rowBase + slot;
    // A group without a row at the tail does not read or write and only joins the barriers (the
    // row loop is workgroup-uniform)
    let has_row = row < params.rows;
${BSHD_TEXT[layout].base}

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
    let s = ${SCALE_WGSL};
    if (has_row && lane == 0u) {
      xs[row] = s;
    }

    // ② Quantize each 4 consecutive elements (quad) and pack them into one word
    if (has_row) {
${(words === "tiled" ? TILED_TEXT : ROW_STORE_TEXT).base}
      var q = lane;
      while (q < quads) {
${words === "padded" ? PADDED_TEXT.word : words === "tiled" ? TILED_TEXT.word : WORD_TEXT}
        xq[${(words === "tiled" ? TILED_TEXT : ROW_STORE_TEXT).at}] = pack4xI8(vec4<i32>(r));
        q = q + ${lanesPerRow}u;
      }
    }

    // Make sure scratch has been read before the next row overwrites scratch[lid]
    workgroupBarrier();
    rowBase = rowBase + nwg.x * ${rowsPerGroup}u;
  }
}
`;

/**
 * The largest row length of the prologue form ({@link quantizeRowsPrologueWgsl}): 16 vec4 per lane
 * kept in registers. Longer rows keep the pair "elementwise op → {@link QUANTIZE_ROWS_WGSL}".
 */
export const QUANTIZE_ROWS_PROLOGUE_MAX_DIM = 16 * 4 * QUANTIZE_ROWS_WORKGROUP_SIZE;

/** Whether a row of `dim` elements can take the prologue form (`dim % 4 == 0`, within the bound). */
export const quantizeRowsPrologueEligible = (dim: number): boolean =>
  Number.isSafeInteger(dim) && dim >= 4 && dim % 4 === 0 && dim <= QUANTIZE_ROWS_PROLOGUE_MAX_DIM;

/** The vec4 values each lane keeps for a row of `dim` elements. */
const prologueSlots = (dim: number): number => {
  if (!quantizeRowsPrologueEligible(dim)) {
    throw new CodegenError(
      `quantize_rows prologue: dim must be a multiple of 4 in 4..${QUANTIZE_ROWS_PROLOGUE_MAX_DIM} (${dim})`,
    );
  }
  return Math.ceil(dim / 4 / QUANTIZE_ROWS_WORKGROUP_SIZE);
};

/**
 * Pipeline key of the prologue form. The slot count is baked into the WGSL (the row length itself
 * is read from Params), so rows with the same slot count share one pipeline.
 */
export const quantizeRowsPrologueKey = (op: PrologueUnaryOp, dim: number): string =>
  `${QUANTIZE_ROWS_KEY}:${op}:reg${prologueSlots(dim)}`;

/**
 * **Per-token quantization with an elementwise prologue** (perf-ledger F-4): `y = op(x)` is
 * computed once per element and quantized exactly as {@link QUANTIZE_ROWS_WGSL} quantizes a stored
 * `y`, without ever storing the f32 `y`. It is step ① of an a8 linear whose input the fusion pass
 * folded with its `op` producer (the bindings are the plain form's: Params, x, xq, xs).
 *
 * The row is held in registers: lane `lid` keeps the quads `lid + 256·j` (`j` below the slot
 * count) as named vec4 values, so `x` is read once. One row = one workgroup, grid-stride over rows,
 * as in the plain form.
 *
 * Bit identity with the pair (measured on Vulkan / NVIDIA and D3D12 / NVIDIA — research
 * 2026-10-25-speed-recon-3080ti §6, 2026-10-25-speed-recon-5070ti §8):
 * - `y` is {@link prologueUnaryWgsl}'s expression per component, the same string the elementwise
 *   kernel compiles (its MUST).
 * - amax is a `nan_max` fold (exact, associative and commutative), so the different element-to-lane
 *   assignment does not change it; the scale and the lattice are the shared {@link SCALE_WGSL} and
 *   {@link latticeWgsl}.
 * - No rounding barrier sits between `y` and the quantization: the next operations are `abs` and
 *   `y / s` (a multiplication by `rcp(s)` after lowering), so no contraction `a·b+c` spans the
 *   boundary. WGSL also allows reassociation, so this is a fact measured at adoption, not a
 *   specification guarantee (the argument for `t` in `adaln-norm.ts`); the per-environment sha256
 *   rows are the check.
 *
 * MUST: only for {@link quantizeRowsPrologueEligible} row lengths — the register count grows with
 * the slot count.
 */
export const quantizeRowsPrologueWgsl = (op: PrologueUnaryOp, dim: number): string => {
  const slots = prologueSlots(dim);
  const lanes = QUANTIZE_ROWS_WORKGROUP_SIZE;
  const valueOf = (j: number): string =>
    `vec4<f32>(${["x", "y", "z", "w"].map((c) => prologueUnaryWgsl(op, `x${j}.${c}`).value).join(", ")})`;
  // The prologue's helpers first (in the elementwise generator's order), then what the
  // quantization needs and the prologue did not bring.
  const helpers = [...prologueUnaryWgsl(op, "v").helpers];
  for (const helper of [IS_NAN_BITS_WGSL, NAN_MAX_WGSL]) {
    if (!helpers.includes(helper)) helpers.push(helper);
  }
  const loads = Array.from(
    { length: slots },
    (_, j) => `    let q${j} = lid + ${j * lanes}u;
    var y${j} = vec4<f32>(0.0);
    if (q${j} < quads) {
      let x${j} = x[base + q${j}];
      y${j} = ${valueOf(j)};
      acc = nan_max(acc, nan_max(nan_max(abs(y${j}.x), abs(y${j}.y)), nan_max(abs(y${j}.z), abs(y${j}.w))));
    }`,
  ).join("\n");
  const stores = Array.from(
    { length: slots },
    (_, j) => `    if (q${j} < quads) {
      let r${j} = ${latticeWgsl(`y${j} / s`)};
      xq[qbase + q${j}] = pack4xI8(vec4<i32>(r${j}));
    }`,
  ).join("\n");
  return `// karume quantize_rows with a ${op} prologue (per-row per-token symmetric i8 of y = ${op}(x); y is never stored) — the row in registers, ${slots} vec4 per lane
struct Params {
  rows: u32,
  dim: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> xq: array<u32>;
@group(0) @binding(3) var<storage, read_write> xs: array<f32>;

${helpers.join("\n\n")}

var<workgroup> scratch: array<f32, ${lanes}>;

@compute @workgroup_size(${lanes})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid3: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let lid = lid3.x;
  // The caller guarantees dim % 4 == 0 and dim <= ${lanes * 4 * slots} (quantizeRowsPrologueEligible)
  let quads = params.dim / 4u;
  var row = wid.x;
  while (row < params.rows) {
    let base = row * quads;

    // ① y = ${op}(x) once per element, kept in registers, and the absolute maximum of y (NaN
    // propagates by a bit-pattern check — plain max swallows it)
    var acc = 0.0;
${loads}
    scratch[lid] = acc;
    workgroupBarrier();
    var stride = ${lanes / 2}u;
    while (stride > 0u) {
      if (lid < stride) {
        scratch[lid] = nan_max(scratch[lid], scratch[lid + stride]);
      }
      workgroupBarrier();
      stride = stride / 2u;
    }
    let amax = scratch[0u];
    // MUST: **multiplication by 1/127** (the MUST above), and NaN passes through the outer select
    let s = ${SCALE_WGSL};
    if (lid == 0u) {
      xs[row] = s;
    }

    // ② Quantize the kept quads (the plain form's pass ② on y)
    let qbase = row * quads;
${stores}

    // Make sure scratch[0] has been read before the next row overwrites scratch[lid]
    workgroupBarrier();
    row = row + nwg.x;
  }
}
`;
};

/**
 * Key of the token-major read ({@link QuantizeRowsLayout} `bshd`). It always uses the lined-up
 * generator (one row per workgroup when the row needs all 256 lanes), so the form does not branch
 * on the geometry.
 */
export const quantizeRowsBshdKey = (geometry: QuantizeRowsGeometry): string =>
  `${QUANTIZE_ROWS_KEY}:r${geometry.rowsPerGroup}w${geometry.lanesPerRow}:bshd`;

/** WGSL of the token-major read (Params `{rows, dim, seq, heads}`). */
export const quantizeRowsBshdWgsl = (geometry: QuantizeRowsGeometry): string =>
  quantizeRowsGroupedWgsl(geometry, "bshd");

/**
 * Uniform Params of the token-major read: `rows` = B·H·S output rows, `dim` = D, `seq` = S, `heads`
 * = H (one 16-byte block).
 */
export const quantizeRowsBshdParams = (
  rows: number,
  dim: number,
  seq: number,
  heads: number,
): Uint32Array<ArrayBuffer> => {
  const params = quantizeRowsParams(rows, dim);
  assertU32Params("quantize_rows bshd params", { seq, heads });
  if (seq < 1 || heads < 1 || rows % (seq * heads) !== 0) {
    throw new CodegenError(
      `quantize_rows bshd params: rows (${rows}) must be a positive multiple of seq·heads (${seq}·${heads})`,
    );
  }
  params[2] = seq;
  params[3] = heads;
  return params;
};

/** Rejects a row length the padded form does not own (`dim % 4 == 0` belongs to the plain forms). */
const assertPaddedDim = (where: string, dim: number): void => {
  if (!Number.isSafeInteger(dim) || dim < 1 || dim % 4 === 0) {
    throw new CodegenError(
      `${where}: dim must be a positive integer that is not a multiple of 4 (${dim}) — a multiple of 4 takes the plain form`,
    );
  }
};

/**
 * Geometry of the **padded form** (ADR 0132 addendum 2026-10-27): rows of any length `dim` in a
 * contiguous `[rows, dim]` layout, each written as `ceil(dim / 4)` words (`xq[row · ceil(dim/4) +
 * …]`) whose bytes from element `dim` on are 0. It is the Vᵀ quantisation of the online attention
 * at `N % 4 != 0` (attention-online-i8a8.ts `AttentionOnlineKeyWords`). The scale is the amax over
 * exactly the row's `dim` elements, and every byte of a real element is the plain form's lattice
 * of the same value (the same {@link SCALE_WGSL} / {@link latticeWgsl} text) — only the row length
 * of `xq` and the zero padding are new. The geometry is the one of the row padded to
 * `4 · ceil(dim / 4)`, always in the lined-up generator (as the token-major read).
 *
 * MUST: only for `dim % 4 != 0` — the plain forms own `dim % 4 == 0`, so their keys and bytes (and
 * every output that uses them) cannot move.
 */
export const quantizeRowsPaddedGeometry = (dim: number): QuantizeRowsGeometry => {
  assertPaddedDim("quantize_rows padded geometry", dim);
  return quantizeRowsGeometry(Math.ceil(dim / 4) * 4);
};

export const quantizeRowsPaddedKey = (geometry: QuantizeRowsGeometry): string =>
  `${QUANTIZE_ROWS_KEY}:r${geometry.rowsPerGroup}w${geometry.lanesPerRow}:pad`;

/** WGSL of the padded form (Params `{rows, dim}` — {@link quantizeRowsPaddedParams}). */
export const quantizeRowsPaddedWgsl = (geometry: QuantizeRowsGeometry): string =>
  quantizeRowsGroupedWgsl(geometry, "rows", "padded");

/** Uniform Params of the padded form: `rows`, `dim` = the row length (not a multiple of 4). */
export const quantizeRowsPaddedParams = (rows: number, dim: number): Uint32Array<ArrayBuffer> => {
  assertU32Params("quantize_rows padded params", { rows, dim });
  // MUST: the in-row loops step by at most the workgroup size.
  assertRowLoopBound("quantize_rows padded params", { dim }, QUANTIZE_ROWS_WORKGROUP_SIZE);
  assertPaddedDim("quantize_rows padded params", dim);
  const params = new Uint32Array(4);
  params[0] = rows;
  params[1] = dim;
  return params;
};

/** Elements per tile of the tiled form: the online attention's key tile. */
const QUANTIZE_ROWS_TILE = 64;

/** Words per row of the tiled form: 16 per 64-element tile. */
export const quantizeRowsTiledRowWords = (dim: number): number =>
  Math.ceil(dim / QUANTIZE_ROWS_TILE) * (QUANTIZE_ROWS_TILE / 4);

/**
 * Geometry of the **tiled form** ({@link TILED_TEXT}): rows of any length `dim ≥ 1` in a contiguous
 * `[rows, dim]` layout, always in the lined-up generator, with the lanes of the row padded to
 * whole tiles.
 */
export const quantizeRowsTiledGeometry = (dim: number): QuantizeRowsGeometry => {
  if (!Number.isSafeInteger(dim) || dim < 1) {
    throw new CodegenError(`quantize_rows tiled geometry: dim must be a positive integer (${dim})`);
  }
  return quantizeRowsGeometry(quantizeRowsTiledRowWords(dim) * 4);
};

export const quantizeRowsTiledKey = (geometry: QuantizeRowsGeometry): string =>
  `${QUANTIZE_ROWS_KEY}:r${geometry.rowsPerGroup}w${geometry.lanesPerRow}:tile${QUANTIZE_ROWS_TILE}`;

/** WGSL of the tiled form (Params `{rows, dim}` — {@link quantizeRowsTiledParams}). */
export const quantizeRowsTiledWgsl = (geometry: QuantizeRowsGeometry): string =>
  quantizeRowsGroupedWgsl(geometry, "rows", "tiled");

/**
 * Uniform Params of the tiled form: `rows`, `dim` = the row length.
 * MUST: `rows % 4 == 0` — four consecutive rows share each `vec4` of the output; a last group
 * of fewer rows would leave components the consumer reads unwritten.
 */
export const quantizeRowsTiledParams = (rows: number, dim: number): Uint32Array<ArrayBuffer> => {
  assertU32Params("quantize_rows tiled params", { rows, dim });
  // MUST: the in-row loops step by at most the workgroup size.
  assertRowLoopBound("quantize_rows tiled params", { dim }, QUANTIZE_ROWS_WORKGROUP_SIZE);
  if (!Number.isSafeInteger(dim) || dim < 1 || rows % 4 !== 0) {
    throw new CodegenError(
      `quantize_rows tiled params: dim must be a positive integer and rows a multiple of 4 (rows ${rows}, dim ${dim})`,
    );
  }
  const params = new Uint32Array(4);
  params[0] = rows;
  params[1] = dim;
  return params;
};

/**
 * The channel-major form (perf-ledger V-6): the per-pixel quantisation of conv3d w8a8's activation
 * read straight from `x[C][P]` (channel-major, the conv's input as stored), writing the same
 * `xq[P][C/4]` / `xs[P]` as "transpose to `[P][C]` → quantize_rows over rows = P". The transposing
 * copy (a strided f32 read at about 100 GB/s and a 4·C·P-byte temporary) disappears.
 *
 * One workgroup = `pixels` pixels × `256 / pixels` channel lanes (consecutive lanes take
 * consecutive pixels, so the reads of one channel row coalesce). It walks blocks of pixels with a
 * grid stride. Pass ① folds `nan_max(·, abs(x))` from 0.0 per lane over the channels `lane, lane +
 * lanes, …`, then across the lanes; pass ② quantises blocks of `blockWords` words per pixel into a
 * workgroup stage and writes each pixel's words contiguously.
 *
 * Bit identity with the row form (MUST — the basis): for finite values `max` is exact,
 * associative and commutative, so the lane partials folded in any order give the row form's amax
 * (the same argument as the lined-up form); `s` and every word are built from the very same
 * {@link SCALE_WGSL} and {@link latticeWgsl} text over the same `vec4(x[c..c+3]) / s` in the same
 * channel order. Only the order of memory accesses changes, so the geometry is free to vary (it is
 * in the key). Caveat: for a pixel with a NaN, which NaN payload reaches `xs` can depend on the
 * fold order (the value is NaN either way), as in the lined-up form.
 * MUST: every barrier sits in workgroup-uniform control flow (the pixel-block loop and the word
 * blocks are bounded by uniform params; the per-pixel guard never encloses a barrier).
 */
export type QuantizeRowsChannelMajorGeometry = {
  /** Pixels per workgroup (a power of two from 4 to 64); the channel lanes are `256 / pixels`. */
  readonly pixels: number;
};

/** The geometry's derived shape (lanes per pixel, words per stage block) — rejected when invalid. */
const channelMajorShape = (
  geometry: QuantizeRowsChannelMajorGeometry,
): { readonly pixels: number; readonly lanes: number; readonly blockWords: number } => {
  const { pixels } = geometry;
  if (!Number.isInteger(Math.log2(pixels)) || pixels < 4 || pixels > 64) {
    throw new CodegenError(
      `quantize_rows channel-major: pixels per workgroup ${pixels} is not a power of 2 in [4, 64]`,
    );
  }
  const lanes = QUANTIZE_ROWS_WORKGROUP_SIZE / pixels;
  // Every lane quantises at least one word of a block (16 words = 64 channels at 64 pixels).
  return { pixels, lanes, blockWords: Math.max(16, lanes) };
};

export const quantizeRowsChannelMajorKey = (geometry: QuantizeRowsChannelMajorGeometry): string => {
  const { pixels, lanes, blockWords } = channelMajorShape(geometry);
  return `${QUANTIZE_ROWS_KEY}:cm:p${pixels}l${lanes}b${blockWords}`;
};

/** WGSL of the channel-major form (Params {@link quantizeRowsParams} with `rows` = P, `dim` = C). */
export const quantizeRowsChannelMajorWgsl = (
  geometry: QuantizeRowsChannelMajorGeometry,
): string => {
  const { pixels, lanes, blockWords } = channelMajorShape(geometry);
  const stage = pixels * blockWords;
  return `// karume quantize_rows channel-major (x[C][P] → xq[P][C/4], xs[P]; ${pixels} pixels × ${lanes} lanes, ${blockWords}-word blocks)
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

var<workgroup> partial: array<f32, ${QUANTIZE_ROWS_WORKGROUP_SIZE}>;
var<workgroup> staged: array<u32, ${stage}>;

@compute @workgroup_size(${QUANTIZE_ROWS_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid3: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let lid = lid3.x;
  let px = lid % ${pixels}u;
  let lane = lid / ${pixels}u;
  let pixelCount = params.rows;
  let dim = params.dim;
  // The caller guarantees dim % 4 == 0 (the i8 payload packs 4 per word)
  let quads = dim / 4u;
  var block = wid.x * ${pixels}u;
  while (block < pixelCount) {
    let p = block + px;
    let inside = p < pixelCount;

    // ① Absolute maximum of the pixel over its channels (NaN propagates by a bit-pattern check)
    var acc = 0.0;
    if (inside) {
      var c = lane;
      while (c < dim) {
        acc = nan_max(acc, abs(x[c * pixelCount + p]));
        c = c + ${lanes}u;
      }
    }
    partial[lid] = acc;
    workgroupBarrier();
    var amax = partial[px];
    for (var l = 1u; l < ${lanes}u; l = l + 1u) {
      amax = nan_max(amax, partial[l * ${pixels}u + px]);
    }
    // MUST: the same scale text as the row form (multiplication by 1/127, the tiny floor, NaN kept)
    let s = ${SCALE_WGSL};
    if (inside && lane == 0u) {
      xs[p] = s;
    }

    // ② Quantise ${blockWords} words per pixel at a time into the stage, then write each pixel's words
    // contiguously
    var w0 = 0u;
    while (w0 < quads) {
      var j = lane;
      while (j < ${blockWords}u) {
        let q = w0 + j;
        if (inside && q < quads) {
          let c = q * 4u;
          let v = vec4<f32>(
            x[c * pixelCount + p],
            x[(c + 1u) * pixelCount + p],
            x[(c + 2u) * pixelCount + p],
            x[(c + 3u) * pixelCount + p],
          ) / s;
          let r = ${latticeWgsl("v")};
          staged[px * ${blockWords}u + j] = pack4xI8(vec4<i32>(r));
        }
        j = j + ${lanes}u;
      }
      workgroupBarrier();
      var t = lid;
      while (t < ${stage}u) {
        let wp = block + t / ${blockWords}u;
        let wq = w0 + t % ${blockWords}u;
        if (wp < pixelCount && wq < quads) {
          xq[wp * quads + wq] = staged[t];
        }
        t = t + ${QUANTIZE_ROWS_WORKGROUP_SIZE}u;
      }
      // The stage and the partials are rewritten by the next block
      workgroupBarrier();
      w0 = w0 + ${blockWords}u;
    }
    block = block + nwg.x * ${pixels}u;
  }
}
`;
};

/** Workgroups of the channel-major form (grid stride over blocks of pixels). */
export const quantizeRowsChannelMajorWorkgroups = (
  pixelCount: number,
  geometry: QuantizeRowsChannelMajorGeometry,
  limit: number,
): number => gridStrideWorkgroups(pixelCount, channelMajorShape(geometry).pixels, limit);
