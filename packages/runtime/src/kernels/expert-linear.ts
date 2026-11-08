/**
 * The kernel of `expert_linear` (`y[t, j] = weight[index[t, j]] · x[t (, j)]` — ADR 0145 decision 4).
 *
 * The GEMV form of `linear` (src/kernels/linear-gemv.ts, ADR 0082) with the weight chosen per row:
 * one thread owns one output column of one pick row (`row = t · picks + j`), reads the pick's expert
 * id from `index` on the GPU, and walks that expert's weight row in ascending k. The routing never
 * goes to the host, so decode keeps one fence per run.
 *
 * ## Numeric contract
 *
 * There is no default path to be bit-identical to, so the contract is the op's own: per output, a
 * sequential f32 sum over `k` in ascending order, each element decoded as `f32(i32(u) − 8) · step`
 * (i4) or read as is (f32), with the multiply-add spelled as `linear`'s sequential GEMV spells it
 * ({@link unitMacs} — the same text). The step is f32 for `int4-sym-g` and an f16 widened to f32
 * for `int4-sym-g-f16` (exact — ADR 0069 decision 3). MUST: one thread holds the whole reduction of
 * an output, so the result does not depend on how rows are split across workgroups.
 *
 * ## Layout
 *
 * `weight` is `[E, n, k]`: expert `e`'s row `c` starts at word `(e · n + c) · (k / unit)`, and its
 * group `g` step at `(e · n + c) · (k / G) + g` (the group codec's scale is `[E, n · k / G]`, row
 * major — `groupScaleShape`). With f16 steps two share a u32 word (element `s` = half `s % 2` of
 * word `s / 2`), read with `unpack2x16float`, so no `shader-f16` feature is needed.
 * `x` is `[T, k]` (shared by the picks of a token: row `t = row / picks`) or `[T, picks, k]` (row
 * `row`), `out` is `[T, picks, n]`.
 *
 * ## Dispatch
 *
 * `[ceil(n / COLS), min(rows, limit), 1]`; the rows grid-stride along y (T · picks can exceed
 * `maxComputeWorkgroupsPerDimension`). A column past `n` returns at once (no shared memory, no
 * barrier).
 *
 * ## Out-of-range ids (ADR 0061)
 *
 * The planner admits only an index produced by a `topk` over `E` (ADR 0145 decision 2), so an id is
 * in `[0, E)` by construction. Should the invariant break, the row is written as quiet NaN instead
 * of reading another expert, as `gather` does; the CPU reference throws.
 */

import { CodegenError } from "../codegen/errors.ts";
import { assertU32Params } from "../codegen/params.ts";
import { unitMacs } from "./linear-gemv.ts";
import { i4GroupKeyPart, i4GroupShift } from "./weight-storage.ts";

/** The weight storages `expert_linear` takes (ADR 0145 decisions 2 and 3). */
export type ExpertLinearStorage = "f32" | "i4" | "i4f16";

/** Output columns per workgroup. */
export const EXPERT_LINEAR_COLS = 32;

/**
 * The binding of the i4 step table (i4 / i4f16 only; bindings 0–4 are dims / x / index / w / out).
 * The WGSL declares it from this constant, so the recipe builder and the kernel cannot drift.
 */
export const EXPERT_LINEAR_SCALE_BINDING = 5;

/** Weight words issued ahead of their multiply-adds (memory parallelism, as `linear`'s GEMV). */
export const EXPERT_LINEAR_UNROLL = 4;

/** The quiet NaN written to a row whose expert id is out of range (ADR 0061). */
export const EXPERT_LINEAR_OOB_BITS = 0x7fc00000;

/** Elements per weight word (`vec4<f32>` = 4, `vec4<u32>` of i4 = 32). */
export const expertLinearUnit = (storage: ExpertLinearStorage): number =>
  storage === "f32" ? 4 : 32;

const isI4 = (storage: ExpertLinearStorage): boolean => storage !== "f32";

const groupShift = (
  storage: ExpertLinearStorage,
  groupSize: number | undefined,
): number | undefined => i4GroupShift("expert_linear", isI4(storage) ? "i4" : "f32", groupSize);

const storageKeyPart = (storage: ExpertLinearStorage): string =>
  storage === "f32" ? "" : storage === "i4" ? ":wi4" : ":wi4h";

/** The pipeline key (codegen determinism: same key → byte-identical WGSL). */
export const expertLinearKey = (storage: ExpertLinearStorage, groupSize?: number): string =>
  `expert_linear:v1:f32:i32:c${EXPERT_LINEAR_COLS}u${EXPERT_LINEAR_UNROLL}${storageKeyPart(
    storage,
  )}${i4GroupKeyPart(groupSize)}`;

const scaleBinding = (storage: ExpertLinearStorage): string =>
  storage === "i4"
    ? `\n@group(0) @binding(${EXPERT_LINEAR_SCALE_BINDING}) var<storage, read> wscale: array<f32>;`
    : storage === "i4f16"
      ? `
@group(0) @binding(${EXPERT_LINEAR_SCALE_BINDING}) var<storage, read> wscale: array<u32>;

// step s of an f16 scale: half s % 2 of word s / 2, widened to f32 (exact)
fn step_at(s: u32) -> f32 {
  let pair = unpack2x16float(wscale[s >> 1u]);
  return select(pair.x, pair.y, (s & 1u) == 1u);
}`
      : "";

/** The reads of one word: the weight word, the group step (i4) and the first x quad. */
const wordLoads = (
  storage: ExpertLinearStorage,
  slot: string,
  unitExpr: string,
  shift: number | undefined,
): string => {
  const unit = expertLinearUnit(storage);
  const step = `scale_base + ((unit${slot} * ${unit}u) >> ${shift}u)`;
  const groupStep =
    storage === "i4"
      ? `
      let ws${slot} = wscale[${step}];`
      : storage === "i4f16"
        ? `
      let ws${slot} = step_at(${step});`
        : "";
  return `      let unit${slot} = ${unitExpr};
      let pw${slot} = w[row_base + unit${slot}];${groupStep}
      let xq${slot} = x_base + unit${slot} * ${unit / 4}u;`;
};

/**
 * The multiply-adds of one word, indented for the row loop. The i4 text is `linear`'s own
 * ({@link unitMacs}); both i4 codecs decode the same bytes, only the step's source differs.
 */
const wordMacs = (storage: ExpertLinearStorage, slot: string): string =>
  unitMacs(storage === "f32" ? "f32" : "i4", slot)
    .split("\n")
    .map((line) => `  ${line}`)
    .join("\n");

/** The WGSL of `expert_linear` (module doc). `groupSize` is required for i4 and refused for f32. */
export const expertLinearWgsl = (storage: ExpertLinearStorage, groupSize?: number): string => {
  const shift = groupShift(storage, groupSize);
  const unit = expertLinearUnit(storage);
  const slots = Array.from({ length: EXPERT_LINEAR_UNROLL }, (_, slot) => `${slot}`);
  const loads = slots.map((slot) => wordLoads(storage, slot, `unit + ${slot}u`, shift)).join("\n");
  const macs = slots.map((slot) => wordMacs(storage, slot)).join("\n");
  const scaleBase = isI4(storage)
    ? `
      let scale_base = wrow * (dims.k >> ${shift}u);`
    : "";
  const note =
    storage === "f32"
      ? ""
      : storage === "i4"
        ? ", i4 weight storage"
        : ", i4 weight storage with f16 steps";
  return `// karume expert_linear (y[t, j] = weight[index[t, j]] · x[t (, j)], f32${note}, ${EXPERT_LINEAR_COLS} cols / wg, ${EXPERT_LINEAR_UNROLL} words prefetched)
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
// Row starts are 16 B aligned because k % ${unit} == 0 (guaranteed by expertLinearParams)
@group(0) @binding(3) var<storage, read> w: array<vec4<${storage === "f32" ? "f32" : "u32"}>>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;${scaleBinding(storage)}

@compute @workgroup_size(${EXPERT_LINEAR_COLS})
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let col = gid.x;
  // No shared memory and no barrier, so edge workgroups may return early
  if (col >= dims.n) {
    return;
  }
  let units = dims.k / ${unit}u;
  let quads = dims.k / 4u;
  var row = gid.y;
  while (row < dims.rows) {
    let pick = index[row];
    // an id outside [0, E) breaks the planner's invariant: NaN instead of another expert's row
    if (pick < 0 || u32(pick) >= dims.experts) {
      out[row * dims.n + col] = bitcast<f32>(dims.oob);
    } else {
      let wrow = u32(pick) * dims.n + col;
      let row_base = wrow * units;${scaleBase}
      let x_base = select(row, row / dims.picks, dims.shared_x == 1u) * quads;
      var acc = 0.0;
      var unit = 0u;
      // Issue the prefetched words first, then the multiply-adds; words stay in ascending order,
      // so the addition order per output does not depend on the prefetch count
      for (; unit + ${EXPERT_LINEAR_UNROLL}u <= units; unit = unit + ${EXPERT_LINEAR_UNROLL}u) {
${loads}
${macs}
      }
      for (; unit < units; unit = unit + 1u) {
${wordLoads(storage, "t", "unit", shift)}
${wordMacs(storage, "t")}
      }
      out[row * dims.n + col] = acc;
    }
    row = row + nwg.y;
  }
}
`;
};

/** The shape of one `expert_linear` dispatch. */
export type ExpertLinearShape = {
  /** Tokens. */
  readonly tokens: number;
  /** Picks per token (the `topk` k). */
  readonly picks: number;
  /** Output features per expert. */
  readonly n: number;
  /** Input features. */
  readonly k: number;
  /** Experts in the weight (`E`). */
  readonly experts: number;
  /** `x` is `[T, k]`, shared by a token's picks (otherwise `[T, picks, k]`). */
  readonly sharedX: boolean;
};

/**
 * The uniform Dims (8 words = 32 bytes, the 16-byte unit of the project convention).
 *
 * MUST: `k` is a multiple of the storage's word (4 for f32, 32 for i4) and, for i4, of the group;
 * every flat index the kernel forms (weight words, steps, `x`, `out`) fits in u32 — checked here,
 * since the WGSL multiplies without a guard.
 */
export const expertLinearParams = (
  storage: ExpertLinearStorage,
  shape: ExpertLinearShape,
  groupSize?: number,
): Uint32Array<ArrayBuffer> => {
  const { tokens, picks, n, k, experts, sharedX } = shape;
  assertU32Params("expert_linear params", { tokens, picks, n, k, experts });
  const shift = groupShift(storage, groupSize);
  const unit = expertLinearUnit(storage);
  if (picks < 1 || experts < 1 || n < 1 || k < 1) {
    throw new CodegenError(
      `expert_linear params: picks ${picks}, experts ${experts}, n ${n} and k ${k} must be at least 1`,
    );
  }
  if (k % unit !== 0 || (shift !== undefined && k % (1 << shift) !== 0)) {
    throw new CodegenError(
      `expert_linear params: k ${k} is not a multiple of the word (${unit})${
        shift === undefined ? "" : ` and of the group (${1 << shift})`
      }`,
    );
  }
  const rows = tokens * picks;
  const limit = 2 ** 32;
  const largest = [
    rows * n, // out
    (sharedX ? tokens : rows) * (k / 4), // x quads
    experts * n * (k / unit), // weight words
    shift === undefined ? 0 : experts * n * (k >> shift), // steps
  ];
  if (largest.some((count) => count >= limit)) {
    throw new CodegenError(
      `expert_linear params: an index of the shape (tokens ${tokens}, picks ${picks}, n ${n}, k ${k}, experts ${experts}) exceeds u32`,
    );
  }
  const params = new Uint32Array(8);
  params[0] = rows;
  params[1] = n;
  params[2] = k;
  params[3] = picks;
  params[4] = sharedX ? 1 : 0;
  params[5] = experts;
  params[6] = EXPERT_LINEAR_OOB_BITS;
  return params;
};

/**
 * The workgroup counts: one column per thread along x, the rows along y, capped at the device's
 * per-dimension limit (the kernel grid-strides the rest).
 */
export const expertLinearDispatch = (
  rows: number,
  n: number,
  maxWorkgroupsPerDimension: number,
): readonly [number, number, number] => {
  const columns = Math.ceil(n / EXPERT_LINEAR_COLS);
  if (columns > maxWorkgroupsPerDimension) {
    throw new CodegenError(
      `expert_linear dispatch: ${columns} column workgroups exceed the device limit ${maxWorkgroupsPerDimension}`,
    );
  }
  return [columns, Math.max(1, Math.min(rows, maxWorkgroupsPerDimension)), 1];
};
