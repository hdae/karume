/**
 * The fixed f32 kernel of the **gated delta rule scan** (op `gated_delta_scan`, the value form — an
 * extended composite op, ADR 0135). The recurrence of a Gated DeltaNet layer, per batch element,
 * value head h (reading key head `⌊h/r⌋`, `r = Hv/Hk`) and token t, with the state
 * `S ∈ ℝ^{dk×dv}` (row = key dim, column = value dim) starting from zero:
 *
 * ```
 * α_t = exp(g_t)
 * S'  = α_t · S_{t−1}
 * m   = S'ᵀ k_t
 * δ   = β_t · (v_t − m)
 * S_t = S' + k_t δᵀ
 * o_t = S_tᵀ (q_t · dk^-½)
 * ```
 *
 * The kernel is the measured form **(a2)** of the spike, `seq:s8:c32:tb8:u:kreg:xexp`
 * (DECIDED: docs/decisions/0135-gated-delta-rule-scan.md decision 3; the measurements are
 * docs/research/2026-10-27-gdn-scan-spike.md). It is a **transcription** of the spike's generator
 * at that configuration, not a redesign — the arithmetic, the re-association and the loop structure
 * are the ones measured. The textual differences from the spike's WGSL are only: the header
 * comment, the dropped final-state binding and its stores (the value form returns no state — ADR
 * 0135 decision 1) together with the then-unused `sbase`, and the accurate exp's function name
 * (`exp_acc` → `exp_accurate`, now the shared {@link EXP_ACCURATE_WGSL}).
 *
 * ## Form
 *
 * A sequential token loop in **one dispatch**; one workgroup = one (batch element, value head,
 * block of {@link COLUMNS} value columns). One value column is split over {@link KEY_SPLIT} threads
 * by key slices (slice p owns the vec4 groups p, p+8, p+16, p+24 of the key axis, 16 rows): each
 * thread holds 16 state floats in registers. The re-associated per-token step (the spike's form):
 *
 * ```
 * a_t   = S_{t−1}ᵀ k_t               (computed in the previous step, from the updated state)
 * m     = α_t · a_t                  (= (α_t S_{t−1})ᵀ k_t)
 * δ     = (v_t − m) · β_t
 * S_t   = S_{t−1} · α_t + k_t δᵀ      (per slice: `st = st * alpha + kc * delta`)
 * o_t   = S_tᵀ (q_t · scale)
 * ```
 *
 * The two column dot products (a for the next token, o for this token) are reduced through
 * workgroup memory once per token, double-buffered by token parity, so **one barrier per token**.
 * k_t of the slice is kept in registers. q (scaled by `dk^-½` on load), k (the block's tokens plus
 * the next one), v, α = exp(g) and β of a block of {@link TOKEN_BLOCK} tokens are staged in
 * workgroup memory; the next block's loads are issued into registers at the start of a block and
 * stored at its end.
 *
 * MUST: **the key split is part of the numerics** (it fixes the order of the partial sums of `m` and
 * `o` — the tree sum over 8 slices of the `hsum` of 4 vec4 products) and lives in the key; it does
 * not vary by device. Columns per workgroup and the token block are geometry: another geometry MAY
 * be chosen per adapter (ADR 0115) **only after a Uint32 check on that adapter shows it
 * bit-identical** to this one (perf ledger K-93; open point 6 of ADR 0135).
 * MUST: `α = exp(g)` uses {@link EXP_ACCURATE_WGSL}, never WGSL's built-in `exp`. With the built-in,
 * this form fails the acceptance band on the long-memory cases (o rel_l2 up to 1.83× the bound —
 * ADR 0135 decision 3). The exp runs once per token at staging (no measurable time).
 * MUST: the token loop of a block is **fully unrolled with constant workgroup-memory offsets**, and
 * no inner loop has a runtime bound. A runtime-bounded loop trip costs about 60 cycles on the
 * measured GPU whatever its body (research §8 finding 2; perf ledger K-94), against a token step of
 * about 960 cycles.
 * MUST: every barrier sits in workgroup-uniform control flow (`T = dims.time` and `t0` are uniform;
 * the per-token guards `t0 + i < T` and `more` are uniform too).
 *
 * ## Full write, determinism and hazards
 *
 * - Every element of o is written once: o_t of column j by the slice-0 thread of j, at the next
 *   token's step (`c_prev`) or, for the last token, in the epilogue (`c_last`).
 * - The reduction order is fixed and there are no atomics, so the output is the same in every run on
 *   one device (in-device determinism, ADR 0110 decision 6 — measured Uint32-identical on the RTX
 *   3080 Ti, research §6.1).
 * - `g` is not checked: a positive g makes the transition expand and is outside the accurate exp's
 *   domain, so its result is unspecified (not guaranteed to be +∞). `g = −∞` gives α = 0 (the state
 *   is forgotten), `g = 0` / `−0` give α = 1 exactly, NaN in g or v propagates.
 * - The implemented subset is `dk = dv = ` {@link GATED_DELTA_SCAN_HEAD_DIM}: other head dims fail
 *   loudly in {@link gatedDeltaScanParams} (and at Session construction — runtime/plan.ts).
 */

import { EXP_ACCURATE_WGSL } from "../codegen/numerics-wgsl.ts";
import { CodegenError } from "../codegen/errors.ts";
import { GATED_DELTA_SCAN_HEAD_DIM } from "../codegen/limits.ts";
import { assertU32Params } from "../codegen/params.ts";
import { tiledWorkgroups } from "../codegen/dispatch.ts";
import { workgroupVariableBytes } from "./workgroup-storage.ts";

/** Threads per value column = key slices (part of the numerics — the MUST of the module doc). */
const KEY_SPLIT = 8;
/** Value columns per workgroup (geometry). Divides the head dim and is a multiple of 4. */
const COLUMNS = 32;
/** Tokens staged per block (geometry). Even, so the token parity of a block's first token is 0. */
const TOKEN_BLOCK = 8;
/** vec4 groups of one q / k row. */
const ROW_VECTORS = GATED_DELTA_SCAN_HEAD_DIM / 4;
/** vec4 groups one thread owns of the key axis (16 key rows). */
const SLICE_VECTORS = ROW_VECTORS / KEY_SPLIT;
/** Staged vec4 items of a block: q `[tb][32]`, k `[tb+1][32]`, v `[tb][cols/4]`. */
const Q_ITEMS = TOKEN_BLOCK * ROW_VECTORS;
const K_ITEMS = (TOKEN_BLOCK + 1) * ROW_VECTORS;
const V_ITEMS = (TOKEN_BLOCK * COLUMNS) / 4;
const VECTOR_ITEMS = Q_ITEMS + K_ITEMS + V_ITEMS;
/** Staged scalar items of a block: α `[tb]`, β `[tb]`. */
const SCALAR_ITEMS = 2 * TOKEN_BLOCK;

/**
 * Invocations per workgroup (256 = the WebGPU default `maxComputeInvocationsPerWorkgroup` and
 * `maxComputeWorkgroupSizeX`, so the kernel needs no raised limit).
 */
export const GATED_DELTA_SCAN_WORKGROUP_SIZE = COLUMNS * KEY_SPLIT;

/**
 * Bytes of `var<workgroup>` storage the kernel uses, counted as WebGPU counts it against
 * `maxComputeWorkgroupStorageSize` (13,888 — under the 16,384 default, so no raised limit is needed;
 * pinned in tests/gated_delta_scan_test.ts).
 */
export const gatedDeltaScanWorkgroupStorageBytes = (): number =>
  workgroupVariableBytes(Q_ITEMS * 16) +
  workgroupVariableBytes(K_ITEMS * 16) +
  workgroupVariableBytes(TOKEN_BLOCK * COLUMNS * 4) +
  workgroupVariableBytes(TOKEN_BLOCK * 4) +
  workgroupVariableBytes(TOKEN_BLOCK * 4) +
  workgroupVariableBytes(2 * GATED_DELTA_SCAN_WORKGROUP_SIZE * 8);

/**
 * MUST: bump the version when the WGSL changes (the pipeline cache does not look at the source).
 * The key carries the key split (numerics), the geometry and the head dim.
 */
export const GATED_DELTA_SCAN_KEY = `gated_delta_scan:v1:f32:s${KEY_SPLIT}:c${COLUMNS}:tb${TOKEN_BLOCK}:d${GATED_DELTA_SCAN_HEAD_DIM}`;

const range = (count: number): number[] => Array.from({ length: count }, (_, index) => index);

/** A balanced pairwise sum (the fixed reduction order of the key slices). */
const treeSum = (terms: readonly string[]): string => {
  if (terms.length === 1) return terms[0];
  const half = terms.length / 2;
  return `(${treeSum(terms.slice(0, half))} + ${treeSum(terms.slice(half))})`;
};

/** The vec4 group of the key axis that the slice-local `group` (0..3) of thread slice `p` owns. */
const slot = (group: number): string => `p + ${KEY_SPLIT * group}u`;

/**
 * One token of a block: `index` is the token index inside the block (a constant), `parity` its
 * parity (the half of `part` this token reads; it writes the other half).
 */
const tokenStep = (index: string, parity: string, indent: string): string => {
  const S = KEY_SPLIT;
  const C = COLUMNS;
  const NT = GATED_DELTA_SCAN_WORKGROUP_SIZE;
  const D4 = ROW_VECTORS;
  const lines: string[] = [];
  const add = (line: string): void => {
    lines.push(indent + line);
  };
  add(`let t = t0 + ${index};`);
  range(S).forEach((slice) =>
    add(`let pv${slice} = part[${parity} * ${NT}u + ${slice * C}u + jl];`),
  );
  add(`let a_all = ${treeSum(range(S).map((slice) => `pv${slice}.x`))};`);
  add(`let c_prev = ${treeSum(range(S).map((slice) => `pv${slice}.y`))};`);
  add(`if (p == 0u && t > 0u) {`);
  add(`  o[((b * T + t - 1u) * dims.hv + h) * ${GATED_DELTA_SCAN_HEAD_DIM}u + j] = c_prev;`);
  add(`}`);
  add(`let alpha = al[${index}];`);
  add(`let bet = be[${index}];`);
  add(`let delta = (vs[${index} * ${C}u + jl] - alpha * a_all) * bet;`);
  add(`var c_acc = vec4<f32>(0.0);`);
  add(`a_acc = vec4<f32>(0.0);`);
  range(SLICE_VECTORS).forEach((group) => {
    add(`let kn${group} = ks[(${index} + 1u) * ${D4}u + ${slot(group)}];`);
    add(`let qv${group} = qs[${index} * ${D4}u + ${slot(group)}];`);
    add(`st${group} = st${group} * alpha + kc${group} * delta;`);
    add(`c_acc = c_acc + st${group} * qv${group};`);
    add(`a_acc = a_acc + st${group} * kn${group};`);
    add(`kc${group} = kn${group};`);
  });
  add(`part[(1u - ${parity}) * ${NT}u + lid] = vec2<f32>(hsum(a_acc), hsum(c_acc));`);
  add(`workgroupBarrier();`);
  return lines.join("\n");
};

export const gatedDeltaScanWgsl = (): string => {
  const S = KEY_SPLIT;
  const C = COLUMNS;
  const TB = TOKEN_BLOCK;
  const D = GATED_DELTA_SCAN_HEAD_DIM;
  const D4 = ROW_VECTORS;
  const NT = GATED_DELTA_SCAN_WORKGROUP_SIZE;
  const QN = Q_ITEMS;
  const KN = K_ITEMS;
  const NV = VECTOR_ITEMS;
  const NS = SCALAR_ITEMS;
  const vectorLoads = Math.ceil(NV / NT);
  const scalarLoads = Math.ceil(NS / NT);

  const stateInit = range(SLICE_VECTORS)
    .map((group) => `  var st${group} = vec4<f32>(0.0);`)
    .join("\n");
  const keyInit = range(SLICE_VECTORS)
    .map((group) => `  var kc${group} = ks[${slot(group)}];`)
    .join("\n");
  const firstDot = range(SLICE_VECTORS)
    .map((group) => `  a_acc = a_acc + st${group} * ks[${slot(group)}];`)
    .join("\n");
  const tokens = range(TB)
    .map(
      (index) =>
        `    if (t0 + ${index}u < T) {\n${tokenStep(`${index}u`, `${index & 1}u`, "      ")}\n    }`,
    )
    .join("\n");
  const prefetchDecl = [
    ...range(vectorLoads).map((item) => `  var pf${item} = vec4<f32>(0.0);`),
    ...range(scalarLoads).map((item) => `  var pfs${item} = 0.0;`),
  ].join("\n");
  const prefetchLoad = (start: string, indent: string): string =>
    [
      ...range(vectorLoads).map(
        (item) => `${indent}pf${item} = stage_load(lid + ${NT * item}u, ${start}, b, h, kh, cb);`,
      ),
      ...range(scalarLoads).map(
        (item) => `${indent}pfs${item} = stage_load_s(lid + ${NT * item}u, ${start}, b, h);`,
      ),
    ].join("\n");
  const prefetchStore = (indent: string): string =>
    [
      ...range(vectorLoads).map((item) => `${indent}stage_store(lid + ${NT * item}u, pf${item});`),
      ...range(scalarLoads).map(
        (item) => `${indent}stage_store_s(lid + ${NT * item}u, pfs${item});`,
      ),
    ].join("\n");
  const lastPartial = treeSum(range(S).map((slice) => `part[par * ${NT}u + ${slice * C}u + jl].y`));
  const epilogue = `  if (T > 0u) {
    let par = T & 1u;
    let c_last = ${lastPartial};
    if (p == 0u) {
      o[((b * T + T - 1u) * dims.hv + h) * ${D}u + j] = c_last;
    }
  }`;

  return `// karume gated_delta_scan (f32, key split ${S}, ${C} columns per workgroup, token block ${TB}, head dim ${D}; ADR 0135 decision 3)
struct Dims {
  time: u32,
  batch: u32,
  hk: u32,
  hv: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> q: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> k: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> v: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> g: array<f32>;
@group(0) @binding(5) var<storage, read> beta: array<f32>;
@group(0) @binding(6) var<storage, read_write> o: array<f32>;

const SCALE: f32 = ${D ** -0.5};
var<workgroup> qs: array<vec4<f32>, ${QN}>;
var<workgroup> ks: array<vec4<f32>, ${KN}>;
var<workgroup> vs: array<f32, ${TB * C}>;
var<workgroup> al: array<f32, ${TB}>;
var<workgroup> be: array<f32, ${TB}>;
var<workgroup> part: array<vec2<f32>, ${2 * NT}>;

fn hsum(x: vec4<f32>) -> f32 {
  return (x.x + x.y) + (x.z + x.w);
}
${EXP_ACCURATE_WGSL}

// Staging item e of the block starting at token t0 (vec4 items: q [tb][32], k [tb+1][32], v [tb][cols/4]).
fn stage_load(e: u32, t0: u32, b: u32, h: u32, kh: u32, cb: u32) -> vec4<f32> {
  let T = dims.time;
  if (e < ${QN}u) {
    let t = t0 + e / ${D4}u;
    if (t >= T) { return vec4<f32>(0.0); }
    return q[((b * T + t) * dims.hk + kh) * ${D4}u + e % ${D4}u] * SCALE;
  }
  if (e < ${QN + KN}u) {
    let f = e - ${QN}u;
    let t = t0 + f / ${D4}u;
    if (t >= T) { return vec4<f32>(0.0); }
    return k[((b * T + t) * dims.hk + kh) * ${D4}u + f % ${D4}u];
  }
  if (e < ${NV}u) {
    let f = e - ${QN + KN}u;
    let t = t0 + f / ${C / 4}u;
    if (t >= T) { return vec4<f32>(0.0); }
    return v[((b * T + t) * dims.hv + h) * ${D4}u + cb * ${C / 4}u + f % ${C / 4}u];
  }
  return vec4<f32>(0.0);
}

fn stage_store(e: u32, x: vec4<f32>) {
  if (e < ${QN}u) {
    qs[e] = x;
  } else if (e < ${QN + KN}u) {
    ks[e - ${QN}u] = x;
  } else if (e < ${NV}u) {
    let f = e - ${QN + KN}u;
    let base = (f / ${C / 4}u) * ${C}u + (f % ${C / 4}u) * 4u;
    vs[base] = x.x;
    vs[base + 1u] = x.y;
    vs[base + 2u] = x.z;
    vs[base + 3u] = x.w;
  }
}

// Scalar items: e < tb -> alpha = exp(g) of token t0 + e; tb <= e < 2 tb -> beta.
fn stage_load_s(e: u32, t0: u32, b: u32, h: u32) -> f32 {
  let T = dims.time;
  if (e < ${TB}u) {
    let t = t0 + e;
    if (t >= T) { return 1.0; }
    return exp_accurate(g[(b * T + t) * dims.hv + h]);
  }
  if (e < ${NS}u) {
    let t = t0 + e - ${TB}u;
    if (t >= T) { return 0.0; }
    return beta[(b * T + t) * dims.hv + h];
  }
  return 0.0;
}

fn stage_store_s(e: u32, x: f32) {
  if (e < ${TB}u) {
    al[e] = x;
  } else if (e < ${NS}u) {
    be[e - ${TB}u] = x;
  }
}

@compute @workgroup_size(${NT})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_index) lid: u32,
) {
  let T = dims.time;
  let ncb = ${D / C}u;
  let cb = wid.x % ncb;
  let bh = wid.x / ncb;
  let h = bh % dims.hv;
  let b = bh / dims.hv;
  let kh = h / (dims.hv / dims.hk);
  let jl = lid % ${C}u;
  let p = lid / ${C}u;
  let j = cb * ${C}u + jl;
${stateInit}
${prefetchDecl}
  // block 0, synchronously
${prefetchLoad("0u", "  ")}
${prefetchStore("  ")}
  workgroupBarrier();
${keyInit}
  var a_acc = vec4<f32>(0.0);
${firstDot}
  part[lid] = vec2<f32>(hsum(a_acc), 0.0);
  workgroupBarrier();
  var t0 = 0u;
  loop {
    if (t0 >= T) { break; }
    let t1 = t0 + ${TB}u;
    let more = t1 < T;
    if (more) {
${prefetchLoad("t1", "      ")}
    }
${tokens}
    if (more) {
      // the last token's barrier ordered every staging read of this block
${prefetchStore("      ")}
      workgroupBarrier();
    }
    t0 = t1;
  }
${epilogue}
}
`;
};

/** Geometry of gated_delta_scan (the one input of params — shapes after binding). */
type GatedDeltaScanDims = {
  /** Tokens T. */
  readonly time: number;
  /** Batch B. */
  readonly batch: number;
  /** Key heads Hk. */
  readonly keyHeads: number;
  /** Value heads Hv (a positive multiple of Hk). */
  readonly valueHeads: number;
  /** Key dim dk. */
  readonly keyDim: number;
  /** Value dim dv. */
  readonly valueDim: number;
};

/**
 * The uniform Dims `{ time, batch, hk, hv }` (16 bytes).
 *
 * MUST: the implemented subset and the head broadcast are checked here too (redundant with the
 * Session-construction check and the shape rule, but a check that the direct kernel-call path also
 * passes). A head dim other than {@link GATED_DELTA_SCAN_HEAD_DIM} would stage the wrong rows and
 * skip or repeat value columns, and `Hv % Hk ≠ 0` would read another head's keys — both without an
 * exception.
 */
export const gatedDeltaScanParams = (dims: GatedDeltaScanDims): Uint32Array<ArrayBuffer> => {
  const { time, batch, keyHeads, valueHeads, keyDim, valueDim } = dims;
  assertU32Params("gated_delta_scan params", { time, batch, keyHeads, valueHeads });
  if (time < 1 || batch < 1) {
    throw new CodegenError(
      `gated_delta_scan params: T ${time} and B ${batch} must be positive (ADR 0135 decision 1)`,
    );
  }
  if (keyDim !== GATED_DELTA_SCAN_HEAD_DIM || valueDim !== GATED_DELTA_SCAN_HEAD_DIM) {
    throw new CodegenError(
      `gated_delta_scan params: head dims dk ${keyDim} / dv ${valueDim} are outside the GPU's implemented subset` +
        ` (dk = dv = ${GATED_DELTA_SCAN_HEAD_DIM} only — ADR 0135 decision 1)`,
    );
  }
  if (keyHeads < 1 || valueHeads < keyHeads || valueHeads % keyHeads !== 0) {
    throw new CodegenError(
      `gated_delta_scan params: Hv ${valueHeads} is not a positive multiple of Hk ${keyHeads}`,
    );
  }
  return new Uint32Array([time, batch, keyHeads, valueHeads]);
};

/**
 * Dispatch count: one workgroup per (batch element, value head, block of {@link COLUMNS} value
 * columns), on x only. MUST: exceeding the device limit fails loudly (a fallback would leave whole
 * heads unwritten — the kernel has no grid-stride).
 */
export const gatedDeltaScanWorkgroups = (
  batch: number,
  valueHeads: number,
  limit: number,
): number =>
  tiledWorkgroups(
    batch * valueHeads * GATED_DELTA_SCAN_HEAD_DIM,
    COLUMNS,
    limit,
    "gated_delta_scan",
  );
