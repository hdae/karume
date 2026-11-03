/**
 * The **integer GEMV** of linear (`integerGemv`): the decode linear of a QAT model whose activation
 * arrives as the packed int8 codes of a fixed SRQ (ADR 0105) and whose weights are integer codes —
 * i2 (offset 2, one scale per output channel), i4 (offset 8, one scale per group of K) or i8
 * (symmetric, one scale per output channel). The product of the two code vectors is accumulated
 * **exactly** per group (one group = the whole row for i2 / i8), and the scales are applied once per
 * group and once per output:
 *
 *     out[n] = (Σ_g (S[n,g] · w_scale[n,g])) · x_scale + bias[n]        (then the optional SRQ)
 *     S[n,g] = Σ_{k ∈ g} q[n,k] · c[k]
 *
 * where `q` is the signed weight value (stored code − offset) and `c` the activation code in
 * [−128, 127]. The groups are added in ascending order, each term and each partial sum rounded once.
 *
 * ## Why exact (the property everything else rests on)
 *
 * Every partial sum of integer products is an integer. A lane accumulates its share of one group in
 * f32, which holds it exactly while its magnitude stays at or below 2^24
 * ({@link INTEGER_GEMV_EXACT_LIMIT}); the lane partials then meet in an i32 workgroup tree. Both are
 * exact **whatever the order**, so `S` is the same in any work assignment — lanes per output, rows per
 * lane group — and the only roundings are those of the epilogue, which every assignment spells the
 * same: `S` to f32, each group's `S · w_scale`, each partial sum of the groups, the `x_scale`
 * multiply and the bias add. That is what lets the kernel reuse one activation load across several
 * output rows without an addition-order contract, and what makes it a mirror-class (E) kernel: the TS
 * reference (`src/reference/linear-gemv-integer.ts`) reproduces every output word.
 * MUST: {@link integerGemvLaneBound} ≤ {@link INTEGER_GEMV_EXACT_LIMIT} for every generated shape;
 * the params builder refuses a shape past it (a lane sum would round and become order-dependent).
 * MUST: the epilogue roundings are separated by the run-time rounding barrier
 * (`^ dims.rounding_mask`, always 0) so no compiler contracts two of them into a fused multiply-add —
 * Metal contracts per expression form (ADR 0105 addendum 4), and the mirror rounds each step.
 * MUST: `S` reaches f32 as `f32(S >> 12) · 4096 + f32(S & 0xfff)`: both parts and the product are
 * exact, so the one add is the only rounding (correctly rounded, as the mirror's `Math.fround`). A
 * bare `f32(S)` would leave the rounding of a sum past 2^24 (i8 at K = 1536) to the implementation.
 *
 * ## The decode (what bounds the kernel on Apple GPUs)
 *
 * On the M2 the parallel GEMV is bound by the per-element weight decode, not by memory: the same loop
 * without the decode streams at 79 GB/s, with a shift-mask-convert decode at 26–37 GB/s (research
 * 2026-11-03). One step covers 16 elements: one activation entry (`vec4<u32>`, 16 codes) and the
 * matching weight unit (i2: one `u32`; i4: one `vec2<u32>`; i8: one `vec4<u32>`).
 *
 * - i2 / i4: the unit is split into four **byte planes** — i2 `(w >> 2p) & 0x03030303` (plane `p`
 *   holds elements `4b + p` in its bytes `b`), i4 `w & 0x0f0f0f0f` and `(w >> 4) & 0x0f0f0f0f` of
 *   each word (the even and the odd elements) — and each plane is turned into floats by one
 *   `unpack4x8unorm` (code / 255). The four planes' dot products with the matching activation planes
 *   are summed and turned back into the exact integer by `round(255 · s)`.
 *   MUST: the rounding is per step (16 elements). There |s| ≤ 16 · 15/255 · 128 ≈ 121 (i4; ≈ 24 for
 *   i2), so the f32 error of the four dot products and of `unpack4x8unorm` itself stays below 0.05
 *   after the scaling by 255 — far inside the 0.5 that `round` absorbs, whatever the device's unpack
 *   accuracy. Rounding a longer sum would let the error grow with K.
 * - i8: `vec4<f32>(unpack4xI8(word))` is already the exact integers; each dot product (≤ 2^16) is
 *   exact, so no rounding is needed.
 *
 * ## Work assignment
 *
 * A workgroup of 128 threads is `128 / lanes` lane groups. A lane group owns `rows` consecutive
 * output channels; lane `l` walks units `l, l + lanes, …` of each group of all of them, reads each
 * activation entry **once** and adds its integer dot product with each of the `rows` weight units.
 * The zero point (i2 / i4) is folded: the lane accumulates `Σ u·c` per row and group (u = the stored
 * code) and one `Σ c` per group shared by the rows, and the epilogue forms `Σ u·c − offset · Σ c`
 * (exact in i32). The i4 groups are unrolled at generation (their count is in the key), each with its
 * own accumulators.
 *
 * The shapes and their (lanes, rows) come from {@link INTEGER_GEMV_SHAPES} (measured shapes only, as
 * for the parallel GEMV — ADR 0098).
 */

import { CodegenError } from "../codegen/errors.ts";
import { gemmParams } from "./gemm.ts";
import { staticQuantizeGuessWgsl, staticQuantizeParams } from "./static-quantize.ts";
import type { WeightStorage } from "./weight-storage.ts";

export type IntegerGemvLanes = 4 | 8 | 16 | 32;
export type IntegerGemvRows = 1 | 2 | 4 | 8;

/** The weight side: storage, and for i4 the group length and the number of groups per row. */
export type IntegerGemvWeight =
  | { readonly storage: "i2" }
  | { readonly storage: "i4"; readonly groupSize: number; readonly groups: number }
  | { readonly storage: "i8" };

export type IntegerGemvVariant = {
  readonly weight: IntegerGemvWeight;
  /** Threads that share one output channel's K. */
  readonly lanes: IntegerGemvLanes;
  /** Output channels one lane group holds (each activation entry is read once for all of them). */
  readonly rows: IntegerGemvRows;
};

const WORKGROUP = 128;
/** Elements per step = per activation entry (`vec4<u32>` = 16 codes). */
const STEP = 16;
/** The largest number of i4 groups per row the generator unrolls. */
const MAX_GROUPS = 4;

/** The largest integer f32 holds exactly together with all smaller ones (2^24). */
export const INTEGER_GEMV_EXACT_LIMIT = 2 ** 24;
const I32_LIMIT = 2 ** 31 - 1;

/** The stored-code offset folded through `Σ c` (0 = symmetric, no `Σ c`). */
const OFFSET: Readonly<Record<IntegerGemvWeight["storage"], number>> = { i2: 2, i4: 8, i8: 0 };
/** The largest magnitude of what a lane multiplies an activation code by (stored code or value). */
const MAX_FACTOR: Readonly<Record<IntegerGemvWeight["storage"], number>> = {
  i2: 3,
  i4: 15,
  i8: 128,
};
/** The weight binding's element per step: i2 16 × 2 bits, i4 16 × 4 bits, i8 16 × 8 bits. */
const UNIT_TYPE: Readonly<Record<IntegerGemvWeight["storage"], string>> = {
  i2: "u32",
  i4: "vec2<u32>",
  i8: "vec4<u32>",
};

const groupsOf = (weight: IntegerGemvWeight): number =>
  weight.storage === "i4" ? weight.groups : 1;

/** The elements of one group at reduction length `k` (the whole row for i2 / i8). */
const groupLength = (weight: IntegerGemvWeight, k: number): number =>
  weight.storage === "i4" ? weight.groupSize : k;

/**
 * The largest magnitude a lane's f32 accumulator can reach at reduction length `k`: its share of one
 * group, `ceil(steps / lanes)` steps of 16 products of at most `MAX_FACTOR · 128` (the `Σ c`
 * accumulator, at most 128 per element, is smaller).
 */
export const integerGemvLaneBound = (variant: IntegerGemvVariant, k: number): number =>
  Math.ceil(groupLength(variant.weight, k) / STEP / variant.lanes) *
  STEP *
  MAX_FACTOR[variant.weight.storage] *
  128;

/** The largest magnitude of a group's i32 sums (`Σ u·c`, `offset · Σ c` and their difference). */
const groupBound = (weight: IntegerGemvWeight, k: number): number =>
  groupLength(weight, k) * (MAX_FACTOR[weight.storage] + OFFSET[weight.storage]) * 128;

/** Why `k` does not fit the variant, or `undefined` when it does. */
const shapeProblem = (variant: IntegerGemvVariant, k: number): string | undefined => {
  const { weight } = variant;
  if (!Number.isSafeInteger(k) || k <= 0 || k % STEP !== 0) {
    return `k must be a positive multiple of ${STEP} (${k})`;
  }
  if (weight.storage === "i4" && k !== weight.groupSize * weight.groups) {
    return `k=${k} is not ${weight.groups} groups of ${weight.groupSize}`;
  }
  if (integerGemvLaneBound(variant, k) > INTEGER_GEMV_EXACT_LIMIT) {
    return `k=${k} lets a lane sum pass 2^24 (no longer exact)`;
  }
  if (groupBound(weight, k) > I32_LIMIT) return `k=${k} lets a group sum overflow i32`;
  return undefined;
};

type IntegerGemvShape = (
  | { readonly storage: "i2" | "i8" }
  | { readonly storage: "i4"; readonly groupSize: number }
) & {
  readonly n: number;
  readonly k: number;
  readonly lanes: IntegerGemvLanes;
  readonly rows: IntegerGemvRows;
};

// DECIDED: only measured shapes are eligible for the opt-in option (the parallel GEMV's rule).
// docs/decisions/0098-linear-gemv-parallel.md
const INTEGER_GEMV_SHAPES: readonly IntegerGemvShape[] = [
  // E2B QAT, the 20 KV-shared layers (M2, research 2026-11-03): gate / up and down.
  { storage: "i2", n: 12288, k: 1536, lanes: 8, rows: 4 },
  { storage: "i2", n: 1536, k: 12288, lanes: 16, rows: 4 },
  // E2B QAT i4 (M2, research 2026-11-03): o (sliding / full) and the own-KV layers' down — the
  // K ≥ 2048 shapes, where the packed parallel GEMV ran at 28–38 GB/s. The K = 1536 shapes (q / k /
  // v / gate / up) and the i8 PLE linears measured slower or equal, so they stay on the parallel GEMV.
  { storage: "i4", groupSize: 2048, n: 1536, k: 2048, lanes: 16, rows: 4 },
  { storage: "i4", groupSize: 4096, n: 1536, k: 4096, lanes: 16, rows: 2 },
  { storage: "i4", groupSize: 2048, n: 1536, k: 6144, lanes: 8, rows: 2 },
];

/**
 * The **only pure function** that returns whether a linear runs as the integer GEMV and with which
 * variant: i2 / i4 / i8 storage, M = 1..8, a measured shape within the exact bounds.
 *
 * @param groupSize The i4 group length (`undefined` for i2 / i8).
 */
export const integerGemvVariantFor = (
  storage: WeightStorage,
  groupSize: number | undefined,
  m: number,
  n: number,
  k: number,
): IntegerGemvVariant | undefined => {
  if (m < 1 || m > 8) return undefined;
  const shape = INTEGER_GEMV_SHAPES.find(
    (entry) =>
      entry.storage === storage &&
      (entry.storage === "i4" ? entry.groupSize : undefined) === groupSize &&
      entry.n === n &&
      entry.k === k,
  );
  if (shape === undefined) return undefined;
  const weight: IntegerGemvWeight =
    shape.storage === "i4"
      ? { storage: "i4", groupSize: shape.groupSize, groups: k / shape.groupSize }
      : { storage: shape.storage };
  const variant: IntegerGemvVariant = { weight, lanes: shape.lanes, rows: shape.rows };
  return weightProblem(weight) === undefined && shapeProblem(variant, k) === undefined
    ? variant
    : undefined;
};

const weightProblem = (weight: IntegerGemvWeight): string | undefined => {
  if (weight.storage !== "i4") return undefined;
  const { groupSize, groups } = weight;
  if (!Number.isInteger(Math.log2(groupSize)) || groupSize < STEP) {
    return `i4 group_size ${groupSize} is not a power of two that is at least ${STEP}`;
  }
  if (!Number.isInteger(groups) || groups < 1 || groups > MAX_GROUPS) {
    return `i4 groups ${groups} is not in 1..${MAX_GROUPS}`;
  }
  return undefined;
};

const assertVariant = (variant: IntegerGemvVariant): void => {
  if (![4, 8, 16, 32].includes(variant.lanes) || ![1, 2, 4, 8].includes(variant.rows)) {
    throw new CodegenError(
      `linear_gemv_integer: invalid variant (lanes ${variant.lanes}, rows ${variant.rows})`,
    );
  }
  if (variant.rows > variant.lanes) {
    throw new CodegenError("linear_gemv_integer: rows must not exceed lanes (one writer per row)");
  }
  const problem = weightProblem(variant.weight);
  if (problem !== undefined) throw new CodegenError(`linear_gemv_integer: ${problem}`);
};

const weightKeyPart = (weight: IntegerGemvWeight): string =>
  weight.storage === "i4" ? `wi4g${weight.groupSize}x${weight.groups}` : `w${weight.storage}`;

export const integerGemvKey = (variant: IntegerGemvVariant, quantize: boolean): string => {
  assertVariant(variant);
  return `linear_gemv_integer:v2:${weightKeyPart(variant.weight)}:l${variant.lanes}r${variant.rows}${
    quantize ? ":static-quantize:v2" : ""
  }`;
};

/** Workgroups along x for `n` output channels. */
export const integerGemvWorkgroups = (n: number, variant: IntegerGemvVariant): number =>
  Math.ceil(n / ((WORKGROUP / variant.lanes) * variant.rows));

const f32Bits = (value: number, what: string): number => {
  if (!Number.isFinite(value) || value <= 0 || Math.fround(value) !== value) {
    throw new CodegenError(
      `linear_gemv_integer: ${what} ${value} must be positive, finite and exactly representable as f32`,
    );
  }
  return new Uint32Array(Float32Array.of(value).buffer)[0];
};

/**
 * The uniform: `m, n, k, rounding_mask (0), x_scale, 3 pad words`, then — with the SRQ epilogue —
 * the 260-word output-side SRQ table ({@link staticQuantizeParams}) at word 8.
 * MUST: k fits the variant (a multiple of 16, the i4 groups, the exact bounds — module doc).
 */
export const integerGemvParams = (
  variant: IntegerGemvVariant,
  m: number,
  n: number,
  k: number,
  xScale: number,
  outputScale?: number,
): Uint32Array<ArrayBuffer> => {
  assertVariant(variant);
  if (!Number.isSafeInteger(m) || m < 1 || m > 8) {
    throw new CodegenError(`linear_gemv_integer params: m must be in 1..8 (${m})`);
  }
  const problem = shapeProblem(variant, k);
  if (problem !== undefined) throw new CodegenError(`linear_gemv_integer params: ${problem}`);
  const params = new Uint32Array(outputScale === undefined ? 8 : 268);
  params.set(gemmParams("linear", m, n, k));
  params[3] = 0;
  params[4] = f32Bits(xScale, "activation scale");
  if (outputScale !== undefined) params.set(staticQuantizeParams(m * n, outputScale), 8);
  return params;
};

const LANES = ["x", "y", "z", "w"] as const;

/** The SRQ epilogue on the output (the same table and search as the parallel GEMV's). */
const staticQuantizeEpilogue = (): string => `
fn srqWord(index: u32) -> u32 { return dims.srq[index >> 2u][index & 3u]; }
fn quantize(bits: u32) -> u32 {
  if (srqWord(258u) != 0u) { return bits; }
  let magnitude = bits & 0x7fffffffu;
  if (magnitude > 0x7f800000u) { return bits | 0x00400000u; }
  let sign = bits & 0x80000000u;
  let cap = select(127u, 128u, sign != 0u);
${staticQuantizeGuessWgsl("srqWord")}
  let guessed = srqWord(129u + min(c_guess, cap));
  if (!off_guess) { return guessed | sign; }
  var lo = 0u;
${[64, 32, 16, 8, 4, 2, 1]
  .map((step) => `  lo += select(0u, ${step}u, magnitude >= srqWord(lo + ${step}u));`)
  .join("\n")}
  lo += select(0u, 1u, magnitude >= srqWord(128u));
  let level = min(lo, cap);
  return srqWord(129u + level) | sign;
}
`;

/** `dot(unpack4x8unorm(plane), activation)` for one byte plane of a weight word. */
const planeDot = (word: string, shift: number, mask: string, activation: string): string =>
  `dot(unpack4x8unorm(${shift === 0 ? word : `(${word} >> ${shift}u)`} & ${mask}), ${activation})`;

/**
 * The activation planes of one step that pair with the weight planes: i2 `p_p = (e_p, e_{4+p},
 * e_{8+p}, e_{12+p})`; i4 the even / odd elements of each half (`e0 = (e0, e2, e4, e6)`, `o0 = (e1,
 * e3, e5, e7)`, `e1` / `o1` the same for elements 8..15). i8 pairs the entries `a_b` directly.
 */
const activationPlanes = (storage: IntegerGemvWeight["storage"]): string[] => {
  if (storage === "i2") {
    return LANES.map((c, p) => `    let p${p} = vec4<f32>(a0.${c}, a1.${c}, a2.${c}, a3.${c});`);
  }
  if (storage === "i4") {
    return [0, 1].flatMap((half) => {
      const [lo, hi] = [`a${2 * half}`, `a${2 * half + 1}`];
      return [
        `    let e${half} = vec4<f32>(${lo}.x, ${lo}.z, ${hi}.x, ${hi}.z);`,
        `    let o${half} = vec4<f32>(${lo}.y, ${lo}.w, ${hi}.y, ${hi}.w);`,
      ];
    });
  }
  return [];
};

/** One row's exact integer dot product of its weight unit `pw{r}` with the step's activations. */
const rowDot = (storage: IntegerGemvWeight["storage"], r: number): string => {
  const word = `pw${r}`;
  if (storage === "i2") {
    const planes = [0, 1, 2, 3].map((p) => planeDot(word, 2 * p, "0x03030303u", `p${p}`));
    return `round(255.0 * (${planes.join(" + ")}))`;
  }
  if (storage === "i4") {
    const planes = [0, 1].flatMap((half) => {
      const part = `${word}.${LANES[half]}`;
      return [
        planeDot(part, 0, "0x0f0f0f0fu", `e${half}`),
        planeDot(part, 4, "0x0f0f0f0fu", `o${half}`),
      ];
    });
    return `round(255.0 * (${planes.join(" + ")}))`;
  }
  return LANES.map((c, b) => `dot(vec4<f32>(unpack4xI8(${word}.${c})), a${b})`).join(" + ");
};

/**
 * One step of group `g`: activation entry `unit` (component `b` = elements 4b..4b+3, one code per
 * byte) is read once and regrouped into the planes that pair with the weight planes; then every row
 * adds its unit's exact integer dot product.
 */
const step = (variant: IntegerGemvVariant, g: number): string => {
  const { storage } = variant.weight;
  // The step's loads come first, ahead of its arithmetic, so their latencies overlap.
  const lines = Array.from(
    { length: variant.rows },
    (_, r) => `    let pw${r} = w[min(col0 + ${r}u, last) * units + unit];`,
  );
  lines.push(`    let xe = x[xbase + unit];`);
  for (let b = 0; b < 4; b++) lines.push(`    let a${b} = vec4<f32>(unpack4xI8(xe.${LANES[b]}));`);
  if (OFFSET[storage] !== 0) {
    lines.push(`    asum${g} = asum${g} + dot(a0 + a1 + a2 + a3, vec4<f32>(1.0));`);
  }
  lines.push(...activationPlanes(storage));
  for (let r = 0; r < variant.rows; r++) {
    lines.push(`    acc${r}_${g} = acc${r}_${g} + ${rowDot(storage, r)};`);
  }
  return lines.join("\n");
};

/** The unit loop of group `g` (i4: the group's baked unit range; i2 / i8: the whole row). */
const groupLoop = (variant: IntegerGemvVariant, g: number): string => {
  const { weight, lanes } = variant;
  const [begin, end] =
    weight.storage === "i4"
      ? [`${(g * weight.groupSize) / STEP}u + lane`, `${((g + 1) * weight.groupSize) / STEP}u`]
      : ["lane", "units"];
  return `  for (var unit = ${begin}; unit < ${end}; unit += ${lanes}u) {
${step(variant, g)}
  }`;
};

/**
 * The WGSL. Bindings follow the linear's numbering (0 dims / 1 x / 2 w / 3 bias / 4 out / 5 wscale);
 * x is the packed int8 activation (`vec4<u32>` = 16 codes), w the weight units (one per 16
 * elements), wscale one f32 per output channel (i4: per channel and group, row-major), and out is
 * `u32` with the SRQ epilogue. The grid is `(workgroups(n), m)`; `wg.y` selects the activation row.
 */
export const integerGemvWgsl = (variant: IntegerGemvVariant, quantize: boolean): string => {
  assertVariant(variant);
  const { weight, lanes, rows } = variant;
  const groups = groupsOf(weight);
  const offset = OFFSET[weight.storage];
  const rowSlots = rows * groups;
  const slots = rowSlots + (offset === 0 ? 0 : groups);
  const range = (count: number): number[] => Array.from({ length: count }, (_, i) => i);
  const scale = (g: number): string =>
    weight.storage === "i4" ? `wscale[col * ${groups}u + ${g}u]` : "wscale[col]";
  const sum = (g: number): string => {
    const row = `red[(lane * ${groups}u + ${g}u) * ${WORKGROUP}u + head]`;
    return offset === 0 ? row : `${row} - ${offset} * red[${(rowSlots + g) * WORKGROUP}u + head]`;
  };
  return `// karume linear gemv integer (${weightKeyPart(weight)}, ${lanes} lanes x ${rows} rows, exact integer sums${
    quantize ? ", SRQ epilogue" : ""
  })
struct Dims {
  m: u32,
  n: u32,
  k: u32,
  rounding_mask: u32,
  x_scale: f32,
  pad0: u32,
  pad1: u32,
  pad2: u32,${quantize ? "\n  srq: array<vec4<u32>, 65>," : ""}
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> x: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> w: array<${UNIT_TYPE[weight.storage]}>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<${quantize ? "u32" : "f32"}>;
@group(0) @binding(5) var<storage, read> wscale: array<f32>;
${quantize ? staticQuantizeEpilogue() : ""}
var<workgroup> red: array<i32, ${slots * WORKGROUP}>;

fn rounded(value: f32) -> f32 { return bitcast<f32>(bitcast<u32>(value) ^ dims.rounding_mask); }

// The exact integer to f32 with one correctly rounded add (both parts and the product are exact).
fn integerToF32(value: i32) -> f32 {
  return rounded(f32(value >> 12u) * 4096.0 + f32(value & 0xfff));
}

@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let lane = lid % ${lanes}u;
  let col0 = (wg.x * ${WORKGROUP / lanes}u + lid / ${lanes}u) * ${rows}u;
  let units = dims.k / ${STEP}u;
  // Channels past n read the last channel's units (in range) and are never stored.
  let last = dims.n - 1u;
  let xbase = wg.y * units;
${range(groups)
  .flatMap((g) => [
    ...range(rows).map((r) => `  var acc${r}_${g} = 0.0;`),
    ...(offset === 0 ? [] : [`  var asum${g} = 0.0;`]),
  ])
  .join("\n")}
${range(groups)
  .map((g) => groupLoop(variant, g))
  .join("\n")}
  // Every lane sum is an exact integer within 2^24, so the conversions are exact.
${range(groups)
  .flatMap((g) => [
    ...range(rows).map((r) => `  red[${(r * groups + g) * WORKGROUP}u + lid] = i32(acc${r}_${g});`),
    ...(offset === 0 ? [] : [`  red[${(rowSlots + g) * WORKGROUP}u + lid] = i32(asum${g});`]),
  ])
  .join("\n")}
  workgroupBarrier();
  for (var width = ${lanes / 2}u; width > 0u; width /= 2u) {
    if (lane < width) {
${range(slots)
  .map(
    (s) =>
      `      red[${s * WORKGROUP}u + lid] = red[${s * WORKGROUP}u + lid] + red[${s * WORKGROUP}u + lid + width];`,
  )
  .join("\n")}
    }
    workgroupBarrier();
  }
  // Lane j < rows of the group stores row j: its groups in ascending order, then the x scale.
  let head = lid - lane;
  let col = col0 + lane;
  if (lane < ${rows}u && col < dims.n) {
    // MUST: the first group's term starts the sum. \`rounded(0.0 + rounded(t))\` simplifies to two
    // adjacent XORs that cancel, and Metal then reassociates \`(S · w_scale) · x_scale\` (measured on
    // the M2: 1-ulp mismatches on every f32-output case).
    var total = rounded(integerToF32(${sum(0)}) * ${scale(0)});
${range(groups)
  .slice(1)
  .map((g) => `    total = rounded(total + rounded(integerToF32(${sum(g)}) * ${scale(g)}));`)
  .join("\n")}
    let scaled = rounded(total * dims.x_scale);
    out[wg.y * dims.n + col] = ${
      quantize
        ? "quantize(bitcast<u32>(scaled + bias[col]) ^ dims.rounding_mask)"
        : "scaled + bias[col]"
    };
  }
}
`;
};
