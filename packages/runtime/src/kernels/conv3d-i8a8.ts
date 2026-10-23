/**
 * The **w8a8 variant** of conv3d (an integer dot product of per-pixel i8 activations × per-channel
 * i8 weights — ADR 0133), opt-in through `SessionOptions.convCompute: "a8"` (default `"f32"`).
 *
 * The implicit GEMM has the orientation of the f32 kernel (src/kernels/conv3d.ts): `M = Cout` rows,
 * `N = Tout·Hout·Wout` output pixels (the time axis folded into N), and the row-major output
 * `[Cout][N]` is the unbatched output tensor as is. What differs is the reduction:
 *
 * ```
 * accf = 0
 * for τ in (kt, kh, kw) ascending:                       // tap-major
 *     p    = input pixel of output pixel n under τ         (out of range → acci = 0, s = 0)
 *     acci = Σ_{ci} xq[p, ci] · wq[co, ci, τ]              // exact i32 over Cin
 *     accf = fma(f32(acci), xs[p], accf)                   // one rounding per tap
 * out[co, n] = fma(accf, wscale[co], bias[co])             // + 1
 * ```
 *
 * The activation is quantised **once per input pixel over its channels** (`quantize_rows` on the
 * transposed `[T·H·W, Cin]` layout — the recipe builder does the transposed copy), so one input
 * pixel's packed channels serve every tap that reads it. The scale changes along K at every tap,
 * which is why the flush is per tap: the form of w4a8's per-group flush (src/kernels/linear-i8a8.ts)
 * with the roles swapped — there the per-row activation scale goes into the final fma, here the
 * per-channel weight scale does.
 *
 * ## Numeric contract (class E of ADR 0110 — a TS mirror at atol = 0)
 *
 * - The dot product over Cin is exact i32 addition (|acci| ≤ Cin·127·128 — {@link CONV3D_I8A8_MAX_CHANNELS_IN}),
 *   so neither the tile geometry nor the K tiles inside a tap move the integers by a bit.
 * - The roundings are `Kt·Kh·Kw + 1` fma operations per output, in ascending tap order — the mirror
 *   `referenceConv3dI8a8` (src/reference/conv3d-i8a8.ts) does the same operations in the same order.
 * - MUST: the flush and the final multiply-add are `fma` (single rounding). The spec caveat of
 *   ADR 0025 (fma's accuracy is inherited from `x*y+z`; non-fused implementations differ by 1 ULP)
 *   carries over as is.
 * - MUST: an out-of-range tap contributes exactly nothing — the B tile is zero-filled
 *   (`idot(0, x) == 0`) **and** the scale used in its flush is `0.0`, so `fma(0, 0, accf) = accf`.
 *   Reading a clamped pixel instead would mix in a real voxel, a wrong value with no exception.
 *
 * ## Layouts
 *
 * - `xq`: `[T·H·W][Cin/4]` words (four i8 channels per word, the packing of `quantize_rows`);
 *   `xscale`: `[T·H·W]`.
 * - `w`: the i8 weight **reordered to tap-major** `[Cout][Kt·Kh·Kw][Cin]` at Session construction
 *   ({@link reorderConv3dWeightTapMajor} — one byte per element, four per word, so one pack of a tap
 *   is one word). The container keeps the canonical `[Cout][Cin][Kt][Kh][Kw]`; reading it in place
 *   (four byte gathers per word) measured 2.3× slower than this layout (ADR 0133 decision 4).
 *   `wscale`: `[Cout]`.
 * - The shared tiles and the inner-product loop are the i8a8 family's single source
 *   (src/kernels/i8a8-geometry.ts): `sa` = weight rows `[pack][row]`, `sb` = pixel columns
 *   `[pack][column]`.
 *
 * ## Eligibility ({@link conv3dI8a8Eligible})
 *
 * i8-resident weight, `groups == 1`, `Cin % 4 == 0` (a pixel's channels start on a word boundary)
 * and `Cin ≤ CONV3D_I8A8_MAX_CHANNELS_IN`. Anything else runs the f32 implicit GEMM and the pipeline
 * key shows it (ADR 0058). The v4 store needs `N % 4 == 0` (four consecutive pixels of one output
 * channel are one word); otherwise the scalar store.
 */

import { tiledWorkgroups } from "../codegen/dispatch.ts";
import { CodegenError } from "../codegen/errors.ts";
import { type Conv3dDims, conv3dIgemmParams } from "./conv3d.ts";
import { CONV3D_DIMS_EXTRA, LINEAR_SCALE_BINDING } from "./gemm.ts";
import {
  assertI8a8Geometry,
  defaultI8a8Geometry,
  I8A8_PACK,
  i8a8AccumulatorInit,
  i8a8ASlots,
  i8a8BSlots,
  i8a8FillStride,
  type I8a8Geometry,
  i8a8GeometryKeyPart,
  i8a8GeometryNote,
  i8a8InnerProductLoop,
  i8a8KPacks,
  i8a8Threads,
  i8a8TileM,
  i8a8TileN,
} from "./i8a8-geometry.ts";
import { i8a8IdotWgsl, LINEAR_ACT_SCALE_BINDING, LINEAR_I8A8_MAX_K } from "./linear-i8a8.ts";
import type { WeightStorage } from "./weight-storage.ts";

/** Binding of the per-pixel activation scale (the same slot as linear's per-token scale). */
export const CONV3D_ACT_SCALE_BINDING = LINEAR_ACT_SCALE_BINDING;

/**
 * Overflow limit of the i32 reduction per tap (the reduction length is `Cin`, so the bound of
 * ADR 0025's K limit applies to `Cin` — `|acci| ≤ Cin · 127 · 128`).
 */
export const CONV3D_I8A8_MAX_CHANNELS_IN = LINEAR_I8A8_MAX_K;

/**
 * The geometry this kernel runs with. The adapter profile has no conv3d field yet (ADR 0133
 * decision 5 — a sweep with the decoder's shapes adds one), so it is linear's default.
 */
export const conv3dI8a8Geometry = (): I8a8Geometry => defaultI8a8Geometry("linear");

/**
 * The shape side of the eligibility (the part Session construction decides from the graph alone
 * when it reorders the weight — `conv3dTapMajorInitializers` in src/runtime/plan.ts).
 * MUST: every condition here is a word-boundary or overflow condition of the kernel, not a
 * preference; a node that fails one runs the f32 implicit GEMM, and the key shows it (ADR 0058).
 */
export const conv3dI8a8EligibleShape = (channelsIn: number, groups: number): boolean =>
  groups === 1 && channelsIn % I8A8_PACK === 0 && channelsIn <= CONV3D_I8A8_MAX_CHANNELS_IN;

/** Whether the node takes this path (the recipe builder's predicate — decided in one place). */
export const conv3dI8a8Eligible = (dims: Conv3dDims, weight: WeightStorage): boolean =>
  weight === "i8" && conv3dI8a8EligibleShape(dims.channelsIn, dims.groups);

/**
 * Reorders the i8 weight bytes of `rows` output channels from the canonical `[row][Cin][taps]` to
 * the tap-major `[row][taps][Cin]` the kernel reads (a permutation inside each row, so a piece of
 * a row-split upload is reordered on its own). Pure; returns a new array.
 */
export const reorderConv3dWeightTapMajor = (
  bytes: Uint8Array<ArrayBuffer>,
  rows: number,
  channelsIn: number,
  taps: number,
): Uint8Array<ArrayBuffer> => {
  const perRow = channelsIn * taps;
  if (bytes.byteLength !== rows * perRow) {
    throw new CodegenError(
      `conv3d i8a8 weight reorder: ${bytes.byteLength} bytes for ${rows} rows × ${channelsIn} channels × ${taps} taps`,
    );
  }
  const out = new Uint8Array(bytes.byteLength);
  for (let row = 0; row < rows; row += 1) {
    const base = row * perRow;
    for (let ci = 0; ci < channelsIn; ci += 1) {
      const from = base + ci * taps;
      for (let tap = 0; tap < taps; tap += 1) {
        out[base + tap * channelsIn + ci] = bytes[from + tap];
      }
    }
  }
  return out;
};

/** Whether the output is written as `vec4<f32>` (four consecutive pixels of one output channel). */
export const conv3dI8a8UsesVec4 = (pixelsOut: number): boolean => pixelsOut % I8A8_PACK === 0;

/** Pipeline key (the geometry discriminator and the dot-product variant, as linear's). */
export const conv3dI8a8Key = (
  v4: boolean,
  dp4a: boolean,
  geometry: I8a8Geometry = conv3dI8a8Geometry(),
): string => `conv3d:v1:i8a8:${i8a8GeometryKeyPart(geometry, v4)}:${dp4a ? "dp4a" : "dp4aEmu"}`;

/** Additive index offset (0 is omitted — keeps the output readable). */
const at = (offset: number): string => (offset === 0 ? "" : ` + ${offset}u`);

/**
 * Assignment of the weight tile (A side: rows = output channels) and of the activation tile
 * (B side: columns = output pixels), and the output assignment, all invariant over the loops.
 * The output assignment is hoisted because the flush at every tap looks up per-column pixel scales.
 */
const prologue = (geometry: I8a8Geometry): string => {
  const tileM = i8a8TileM(geometry);
  const tileN = i8a8TileN(geometry);
  const stride = i8a8FillStride(geometry);
  const kPacks = i8a8KPacks(geometry);
  const { regM, regN } = geometry;
  const aRows = Array.from(
    { length: i8a8ASlots(geometry) },
    (_, slot) =>
      `  let arow${slot} = ${slot === 0 ? `wid.y * ${tileM}u + ar` : `arow0 + ${slot * stride}u`};
  let abase${slot} = arow${slot} * taps * cin4;`,
  ).join("\n");
  const bCols = Array.from(
    { length: i8a8BSlots(geometry) },
    (_, slot) =>
      `  let wcol${slot} = ${slot === 0 ? `wid.x * ${tileN}u + wc` : `wcol0 + ${slot * stride}u`};`,
  ).join("\n");
  const oRows = Array.from({ length: regM }, (_, row) =>
    row === 0
      ? `  let orow0 = wid.y * ${tileM}u + lid.y * ${regM}u;`
      : `  let orow${row} = orow0 + ${row}u;`,
  ).join("\n");
  const accf = Array.from({ length: regM }, (_, row) =>
    Array.from(
      { length: regN / I8A8_PACK },
      (_, quad) => `  var accf${row}_${quad} = vec4<f32>(0.0);`,
    ).join("\n"),
  ).join("\n");
  return `  // W tile assignment (${tileM} output channels × ${kPacks} packs over ${i8a8Threads(
    geometry,
  )} threads in ${i8a8ASlots(geometry)} rounds). abase is the word number of the row's first pack
  // (tap-major [Cout][taps][Cin], four channels per word)
  let ar = tid / ${kPacks}u;
  let ap = tid % ${kPacks}u;
  let sa_at = ap * ${tileM}u + ar;
${aRows}
  // x tile assignment (${tileN} output pixels × ${kPacks} packs over ${i8a8Threads(
    geometry,
  )} threads in ${i8a8BSlots(geometry)} rounds)
  let wc = tid / ${kPacks}u;
  let wp = tid % ${kPacks}u;
  let sb_at = wp * ${tileN}u + wc;
${bCols}
  // Output assignment is **outside the loop** (the flush at every tap looks up per-column pixel
  // scales)
  let ocol = wid.x * ${tileN}u + lid.x * ${regN}u;
${oRows}
  // f32 accumulator (written only at tap boundaries — Kt·Kh·Kw roundings)
${accf}`;
};

/**
 * Tile fill inside one K tile of one tap. The A side reads one word of the tap-major weight; the
 * B side reads one word of the input pixel's packed channels.
 * Zero-filled out of range (`idot(0, x) == 0` — exact even when `Cin % tileK != 0`).
 */
const fillTiles = (geometry: I8a8Geometry): string => {
  const kPacks = i8a8KPacks(geometry);
  const stride = i8a8FillStride(geometry);
  const a = Array.from(
    { length: i8a8ASlots(geometry) },
    (_, slot) =>
      `      var av${slot} = 0u;
      if (arow${slot} < dims.m && apack < cin4) {
        av${slot} = w[abase${slot} + tap * cin4 + apack];
      }
      sa[sa_at${at(slot * stride)}] = av${slot};`,
  ).join("\n");
  const b = Array.from(
    { length: i8a8BSlots(geometry) },
    (_, slot) =>
      `      var bv${slot} = 0u;
      if (bpix${slot} >= 0 && wpack < cin4) {
        bv${slot} = xq[u32(bpix${slot}) * cin4 + wpack];
      }
      sb[sb_at${at(slot * stride)}] = bv${slot};`,
  ).join("\n");
  return `      // A side: one word of the row's pack at this tap (tap-major [Cout][taps][Cin])
      let apack = t * ${kPacks}u + ap;
${a}
      // B side: one word of the input pixel's packed channels (0 out of range — the MUST of the
      // out-of-range tap)
      let wpack = t * ${kPacks}u + wp;
${b}`;
};

/** The input pixel of each fill column under the current tap (−1 = out of range or no column). */
const columnPixels = (geometry: I8a8Geometry): string =>
  Array.from(
    { length: i8a8BSlots(geometry) },
    (_, slot) =>
      `    var bpix${slot} = -1;
    if (wcol${slot} < dims.n) {
      bpix${slot} = xpix(wcol${slot}, kt, ky, kx);
    }`,
  ).join("\n");

/**
 * Flush at the tap boundary — the only place that folds the i32 accumulator into f32, once, with
 * the input pixel's scale of each output column (0 where the tap is out of range, so the fma
 * leaves accf unchanged).
 * MUST: the multiply-add is `fma` (single rounding). MUST: do not multiply by `wscale` here — it is
 * per output channel and goes into the final fma (the roles of w4a8's flush, swapped).
 */
const flush = (geometry: I8a8Geometry): string => {
  const { regM, regN } = geometry;
  const quads = regN / I8A8_PACK;
  const scales = Array.from(
    { length: regN },
    (_, col) =>
      `    var xs${col} = 0.0;
    if (ocol${at(col)} < dims.n) {
      let pix${col} = xpix(ocol${at(col)}, kt, ky, kx);
      if (pix${col} >= 0) {
        xs${col} = xscale[u32(pix${col})];
      }
    }`,
  ).join("\n");
  const blocks = Array.from({ length: quads }, (_, quad) => {
    const lanes = Array.from(
      { length: I8A8_PACK },
      (_, lane) => `xs${quad * I8A8_PACK + lane}`,
    ).join(", ");
    const rows = Array.from(
      { length: regM },
      (_, row) =>
        `    accf${row}_${quad} = fma(vec4<f32>(acc${row}_${quad}), xsv${quad}, accf${row}_${quad});`,
    ).join("\n");
    return `    let xsv${quad} = vec4<f32>(${lanes});
${rows}`;
  }).join("\n");
  return `    // Flush to f32 exactly at the tap boundary with the input pixel's scale per column (MUST:
    // single rounding of fma; wscale is not applied until the end)
${scales}
${blocks}`;
};

/**
 * Output store — `out = fma(accf, wscale[row], bias[row])`, the per-channel scale applied here for
 * the first time. Rows and column quads are unrolled as in linear (no dynamic `acc[i]` index).
 */
const store = (geometry: I8a8Geometry, v4: boolean): string => {
  const { regM, regN } = geometry;
  const quads = regN / I8A8_PACK;
  if (v4) {
    const quadDecls = Array.from({ length: quads }, (_, quad) =>
      quad === 0 ? `  let ocq0 = ocol / ${I8A8_PACK}u;` : `  let ocq${quad} = ocq0 + ${quad}u;`,
    ).join("\n");
    const blocks = Array.from({ length: quads }, (_, quad) => {
      const rowStores = Array.from(
        { length: regM },
        (_, row) =>
          `    if (orow${row} < dims.m) {
      out[orow${row} * n4 + ocq${quad}] = fma(accf${row}_${quad}, vec4<f32>(wscale[orow${row}]), vec4<f32>(bias[orow${row}]));
    }`,
      ).join("\n");
      return `  if (ocq${quad} < n4) {
    // n % ${I8A8_PACK} == 0 and ocq${quad} < n4, so the four pixels of the word are in range (bias is
    // always f32 — ADR 0006)
    // MUST: wscale goes to the final fma (single rounding) — xs was folded at the tap boundaries
${rowStores}
  }`;
    }).join("\n");
    return `  let n4 = dims.n / ${I8A8_PACK}u;
${quadDecls}
${blocks}`;
  }
  const lanes = ["x", "y", "z", "w"] as const;
  return Array.from({ length: regM }, (_, row) => {
    const writes = Array.from({ length: regN }, (_, col) => {
      const acc = `accf${row}_${Math.floor(col / I8A8_PACK)}.${lanes[col % I8A8_PACK]}`;
      return `    if (ocol${at(col)} < dims.n) {
      out[obase + ocol${at(col)}] = fma(${acc}, ws, b);
    }`;
    }).join("\n");
    return `  if (orow${row} < dims.m) {
    let obase = orow${row} * dims.n;
    // MUST: wscale goes to the final fma (single rounding) — xs was folded at the tap boundaries
    let ws = wscale[orow${row}];
    let b = bias[orow${row}];
${writes}
  }`;
  }).join("\n");
};

export const conv3dI8a8Wgsl = (
  v4: boolean,
  dp4a: boolean,
  geometry: I8a8Geometry = conv3dI8a8Geometry(),
): string => {
  assertI8a8Geometry(geometry, "conv3d i8a8");
  const kPacks = i8a8KPacks(geometry);
  return `// karume conv3d (x[Cin,T,H,W] * W[Cout,Cin,Kt,Kh,Kw] + b[Cout], integer dot product of per-pixel i8 activations × i8 weights${
    dp4a ? "" : " (emulated)"
  }, ${i8a8GeometryNote(geometry)}${v4 ? " + vec4" : ""})
struct Dims {
  m: u32,
  n: u32,
  k: u32,
${CONV3D_DIMS_EXTRA}}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> xq: array<u32>;
@group(0) @binding(2) var<storage, read> w: array<u32>; // tap-major [Cout][Kt·Kh·Kw][Cin], four channels per word
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<${v4 ? "vec4<f32>" : "f32"}>;
@group(0) @binding(${LINEAR_SCALE_BINDING}) var<storage, read> wscale: array<f32>;
@group(0) @binding(${CONV3D_ACT_SCALE_BINDING}) var<storage, read> xscale: array<f32>;

${i8a8IdotWgsl(dp4a)}

// Input pixel of output pixel n under the tap offsets (the dilated tap minus the padding); −1 out of
// range. n = (ot·Hout + oy)·Wout + ox; the pixel is (it·Hin + iy)·Win + ix of the transposed
// [T·H·W][Cin] activation
fn xpix(n: u32, kt: i32, ky: i32, kx: i32) -> i32 {
  let plane = dims.height_out * dims.width_out;
  let pixel = n % plane;
  let it = i32((n / plane) * dims.stride_t) + kt;
  let iy = i32((pixel / dims.width_out) * dims.stride_h) + ky;
  let ix = i32((pixel % dims.width_out) * dims.stride_w) + kx;
  if (it < 0 || u32(it) >= dims.time_in || iy < 0 || u32(iy) >= dims.height_in ||
      ix < 0 || u32(ix) >= dims.width_in) {
    return -1;
  }
  return i32((u32(it) * dims.height_in + u32(iy)) * dims.width_in + u32(ix));
}

// Shared tiles are laid out [pack][row] / [pack][column] (the i8a8 family's layout)
var<workgroup> sa: array<u32, ${kPacks * i8a8TileM(geometry)}>;
var<workgroup> sb: array<u32, ${kPacks * i8a8TileN(geometry)}>;

@compute @workgroup_size(${geometry.wgX}, ${geometry.wgY})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.y * ${geometry.wgX}u + lid.x;
  // Eligibility requires Cin % 4 == 0, so a pixel's channels start on a word boundary
  let cin4 = dims.channels_in / ${I8A8_PACK}u;
  let khw = dims.kernel_h * dims.kernel_w;
  let taps = dims.kernel_t * khw;
${prologue(geometry)}
  // The loop bounds are uniform (dims is a uniform buffer) — needed for the inner
  // workgroupBarrier to meet WGSL's uniformity requirement
  let tiles = (cin4 + ${kPacks - 1}u) / ${kPacks}u;
  for (var tap = 0u; tap < taps; tap = tap + 1u) {
    // The tap's offsets into the input (dilated, minus the padding — the decomposition of
    // src/kernels/gemm.ts conv3dKDecode, with ci taken out of the flat k)
    let kt = i32((tap / khw) * dims.dilation_t) - i32(dims.padding_t);
    let ks = tap % khw;
    let ky = i32((ks / dims.kernel_w) * dims.dilation_h) - i32(dims.padding_h);
    let kx = i32((ks % dims.kernel_w) * dims.dilation_w) - i32(dims.padding_w);
${columnPixels(geometry)}
    // The i32 accumulator is reset per tap (|acci| ≤ Cin·127·128 — limit: CONV3D_I8A8_MAX_CHANNELS_IN)
${i8a8AccumulatorInit(geometry).replace(/^/gm, "  ")}
    for (var t = 0u; t < tiles; t = t + 1u) {
${fillTiles(geometry)}
      workgroupBarrier();
${i8a8InnerProductLoop(geometry).replace(/^/gm, "  ")}
      workgroupBarrier();
    }
${flush(geometry)}
  }
${store(geometry, v4)}
}
`;
};

/**
 * Uniform Dims — the same 24 words as the f32 implicit GEMM (`conv3dIgemmParams`), plus the checks
 * of this path (`Cin % 4 == 0` and the i32 bound per tap). `k` is carried but not read.
 */
export const conv3dI8a8Params = (dims: Conv3dDims): Uint32Array<ArrayBuffer> => {
  if (dims.channelsIn % I8A8_PACK !== 0) {
    throw new CodegenError(
      `conv3d i8a8 params: channels_in must be a multiple of ${I8A8_PACK} (${dims.channelsIn})`,
    );
  }
  if (dims.channelsIn > CONV3D_I8A8_MAX_CHANNELS_IN) {
    throw new CodegenError(
      `conv3d i8a8 params: channels_in=${dims.channelsIn} exceeds the i32 reduction guard ${CONV3D_I8A8_MAX_CHANNELS_IN}`,
    );
  }
  return conv3dIgemmParams(dims);
};

/** Dispatch `[ceil(N/tileN), ceil(Cout/tileM), 1]` (one workgroup = one output tile; the limit fails loudly). */
export const conv3dI8a8Workgroups = (
  dims: Conv3dDims,
  geometry: I8a8Geometry,
  limit: number,
  where: string,
): readonly [number, number, number] => [
  tiledWorkgroups(dims.timeOut * dims.heightOut * dims.widthOut, i8a8TileN(geometry), limit, where),
  tiledWorkgroups(dims.channelsOut, i8a8TileM(geometry), limit, where),
  1,
];
