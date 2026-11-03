/**
 * The **byte-plane GEMV** of linear (`linearGemvPlanes`): a decode GEMV for i2 weights (one scale per
 * output channel) with **f32 activations**, at measured shapes — the QAT E2B lm_head (262144 × 1536).
 * It computes
 *
 *     out[n] = (Σ_k q[n,k] · x[k]) · w_scale[n] + bias[n]
 *
 * with `q` the signed weight value (stored code − 2, in −2..1).
 *
 * ## Why a separate kernel (what bounds the sequential GEMV on Apple GPUs)
 *
 * The sequential GEMV gives every thread one column: a SIMD group's 32 lanes read 32 rows 384 B
 * apart. On the M2 that layout stays at 46–53 GB/s even with the decode removed (research
 * 2026-11-03-qat-integer-gemv-m2, the lm_head probe). Here `lanes` threads share one column and walk
 * its u32 words (16 codes) `lane, lane + lanes, …`, so a lane group reads a row contiguously; a lane
 * group holds `rows` columns and reads each activation entry once for all of them. Without the decode
 * that layout reaches 87 GB/s on the M2 — the read roofline — after which the decode instructions
 * bound it, so the decode is kept to a few instructions per four codes.
 *
 * ## The decode
 *
 * A word is split into four **byte planes**, `(w >> 2p) & 0x03030303` (plane `p` holds elements
 * `4b + p` in its bytes `b`); `unpack4x8unorm` turns a plane into `u / 255`, and
 *
 *     fma(255, u / 255, 1.5 · 2^23) − (1.5 · 2^23 + 2)
 *
 * gives `u − 2` exactly: near 1.5 · 2^23 consecutive f32 values are 1 apart, so adding the constant
 * rounds `255 · (u / 255)` (within 1e-6 of `u`) to the integer `u`, whether or not the multiply is
 * fused with the add, and the subtraction is exact. The four plane dot products with the regrouped
 * activations are added to the lane's accumulator.
 * NOTE: a compiler that folds the two constants (`fma(a, b, c) − d` → `fma(a, b, c − d)`) would
 * decode code 2 as about 1.2e-7 instead of 0. The M2's Metal does not (u32-equal to a `round` decode
 * — research); elsewhere the class-R check below bounds the result either way.
 *
 * ## Numerics: contract class R
 *
 * Because `q` is in −2..1, every product `q · x` is exact in f32; the kernel's roundings are the
 * additions (each lane's running sum, the dot products, the workgroup tree) and one multiply by the
 * channel scale. The reference op multiplies per element by `q · scale` and adds in ascending k — so
 * this kernel differs by the addition order and by applying the scale once (a distributive regrouping
 * whose rounding is at most the reference's per-element scale roundings). Its claim is class R of ADR
 * 0110: the f64 band is the reference op band (`GEMM_TOLERANCE`), checked with the four points.
 * Nothing pins the order inside a `dot` or across the four planes; within one compiled pipeline it is
 * fixed, so two dispatches of the same input are bit-identical. An E mirror is not offered: the
 * activations are f32, so the sum is not exact, and WGSL leaves whether `dot` / `fma` fuse to the
 * implementation.
 *
 * ## Work assignment
 *
 * A workgroup of 128 threads is `128 / lanes` lane groups of `rows` consecutive columns; the lane
 * partials meet in a workgroup tree, and lane `j < rows` of a group stores column `j`. The grid is
 * `(workgroups(n), m)`; `wg.y` selects the activation row (M = 1..8, as for the parallel GEMV).
 * Columns past `n` read the last column's words (in range) and are never stored.
 *
 * The shapes and their (lanes, rows) come from {@link PLANES_GEMV_SHAPES} (measured shapes only — ADR
 * 0098's rule).
 */

import { CodegenError } from "../codegen/errors.ts";
import { gemmParams } from "./gemm.ts";
import type { WeightStorage } from "./weight-storage.ts";

export type PlanesGemvLanes = 4 | 8 | 16 | 32;
export type PlanesGemvRows = 1 | 2 | 4 | 8;

export type PlanesGemvVariant = {
  /** Threads that share one output channel's K. */
  readonly lanes: PlanesGemvLanes;
  /** Output channels one lane group holds (each activation entry is read once for all of them). */
  readonly rows: PlanesGemvRows;
};

const WORKGROUP = 128;
/** Elements per weight word (`u32` = 16 i2 codes) = per four activation quads. */
const WORD = 16;
const MAX_M = 8;

type PlanesGemvShape = {
  readonly n: number;
  readonly k: number;
  readonly variant: PlanesGemvVariant;
};

// DECIDED: only measured shapes are eligible for the opt-in option (the parallel GEMV's rule).
// docs/decisions/0098-linear-gemv-parallel.md
const PLANES_GEMV_SHAPES: readonly PlanesGemvShape[] = [
  // E2B QAT lm_head (M2, research 2026-11-03): 2383 µs median against 2973 for the sequential GEMV.
  { n: 262144, k: 1536, variant: { lanes: 8, rows: 4 } },
];

/**
 * The **only pure function** that returns whether a linear runs as the byte-plane GEMV and with which
 * variant: i2 storage (one scale per channel), M = 1..8, a measured shape.
 */
export const planesGemvVariantFor = (
  storage: WeightStorage,
  m: number,
  n: number,
  k: number,
): PlanesGemvVariant | undefined => {
  if (storage !== "i2" || m < 1 || m > MAX_M || k <= 0 || k % WORD !== 0) return undefined;
  return PLANES_GEMV_SHAPES.find((shape) => shape.n === n && shape.k === k)?.variant;
};

const assertVariant = (variant: PlanesGemvVariant): void => {
  if (![4, 8, 16, 32].includes(variant.lanes) || ![1, 2, 4, 8].includes(variant.rows)) {
    throw new CodegenError(
      `linear_gemv_planes: invalid variant (lanes ${variant.lanes}, rows ${variant.rows})`,
    );
  }
  if (variant.rows > variant.lanes) {
    throw new CodegenError("linear_gemv_planes: rows must not exceed lanes (one writer per row)");
  }
};

export const planesGemvKey = (variant: PlanesGemvVariant): string => {
  assertVariant(variant);
  return `linear_gemv_planes:v1:wi2:l${variant.lanes}r${variant.rows}`;
};

/** Workgroups along x for `n` output channels. */
export const planesGemvWorkgroups = (n: number, variant: PlanesGemvVariant): number =>
  Math.ceil(n / ((WORKGROUP / variant.lanes) * variant.rows));

/** The uniform: `m, n, k` (the linear's layout). MUST: k is a positive multiple of 16, m in 1..8. */
export const planesGemvParams = (m: number, n: number, k: number): Uint32Array<ArrayBuffer> => {
  if (!Number.isSafeInteger(m) || m < 1 || m > MAX_M) {
    throw new CodegenError(`linear_gemv_planes params: m must be in 1..${MAX_M} (${m})`);
  }
  if (!Number.isSafeInteger(k) || k <= 0 || k % WORD !== 0) {
    throw new CodegenError(`linear_gemv_planes params: k must be a positive multiple of 16 (${k})`);
  }
  return gemmParams("linear", m, n, k);
};

const LANES = ["x", "y", "z", "w"] as const;

/** `u − 2` of plane `p` of row `r`'s word, exact (module doc). */
const decode = (r: number, p: number): string => {
  const plane = p === 0 ? `pw${r}` : `(pw${r} >> ${2 * p}u)`;
  return `(fma(vec4<f32>(255.0), unpack4x8unorm(${plane} & 0x03030303u), vec4<f32>(12582912.0)) - vec4<f32>(12582914.0))`;
};

/**
 * The WGSL. Bindings follow the linear's numbering (0 dims / 1 x / 2 w / 3 bias / 4 out / 5 wscale);
 * x is f32 quads, w the i2 words as `u32`.
 */
export const planesGemvWgsl = (variant: PlanesGemvVariant): string => {
  assertVariant(variant);
  const { lanes, rows } = variant;
  const range = (count: number): number[] => Array.from({ length: count }, (_, i) => i);
  return `// karume linear gemv planes (i2, f32 activations, ${lanes} lanes x ${rows} rows)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<storage, read> wscale: array<f32>;
var<workgroup> red: array<f32, ${rows * WORKGROUP}>;

@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let lane = lid % ${lanes}u;
  let col0 = (wg.x * ${WORKGROUP / lanes}u + lid / ${lanes}u) * ${rows}u;
  let units = dims.k / ${WORD}u;
  // Channels past n read the last channel's words (in range) and are never stored.
  let last = dims.n - 1u;
  let xbase = wg.y * units * 4u;
${range(rows)
  .map((r) => `  let base${r} = min(col0 + ${r}u, last) * units;\n  var acc${r} = 0.0;`)
  .join("\n")}
  for (var unit = lane; unit < units; unit += ${lanes}u) {
    // The step's loads come first, ahead of its arithmetic, so their latencies overlap.
${range(rows)
  .map((r) => `    let pw${r} = w[base${r} + unit];`)
  .join("\n")}
${range(4)
  .map((b) => `    let a${b} = x[xbase + unit * 4u + ${b}u];`)
  .join("\n")}
    // Plane p pairs with elements 4b + p of the word: component p of each activation quad.
${LANES.map((c, p) => `    let p${p} = vec4<f32>(a0.${c}, a1.${c}, a2.${c}, a3.${c});`).join("\n")}
${range(rows)
  .map(
    (r) =>
      `    acc${r} = acc${r} + (${range(4)
        .map((p) => `dot(${decode(r, p)}, p${p})`)
        .join(" + ")});`,
  )
  .join("\n")}
  }
${range(rows)
  .map((r) => `  red[${r * WORKGROUP}u + lid] = acc${r};`)
  .join("\n")}
  workgroupBarrier();
  for (var width = ${lanes / 2}u; width > 0u; width /= 2u) {
    if (lane < width) {
${range(rows)
  .map(
    (r) =>
      `      red[${r * WORKGROUP}u + lid] = red[${r * WORKGROUP}u + lid] + red[${r * WORKGROUP}u + lid + width];`,
  )
  .join("\n")}
    }
    workgroupBarrier();
  }
  // Lane j < rows of the group stores row j.
  let head = lid - lane;
  let col = col0 + lane;
  if (lane < ${rows}u && col < dims.n) {
    out[wg.y * dims.n + col] = red[lane * ${WORKGROUP}u + head] * wscale[col] + bias[col];
  }
}
`;
};
