// Real-GPU checks of the **GEMV family** of linear (1 ≤ M ≤ {@link LINEAR_GEMV_MAX_ROWS} × weights i4 / i8 —
// ADR 0082 / perf-ledger K-16 (M=1) / K-21 (row blocks for M ≥ 2)).
//
// This family exists only for speed, and **its values must not differ by a single bit from the default path
// (linear in src/kernels/gemm.ts)** (the only freedom ADR 0022 decision 3 grants is the work assignment). So three
// things are checked:
//
// 1. **Exact u32 equality with the default path** — for the same weights and the same input rows, the GEMV path's
//    output and **the first m rows** of the default GEMM do not differ by a single bit. It directly proves that the
//    K reduction order per output element is the same "sequential in ascending k" regardless of the path, and
//    extends the "bit-identical across buckets" of tests/gpu_gemm_skinny_test.ts across families.
//    MUST: if it breaks, suspect **the bit-identity claim itself**, not the variant selection (take it back to the
//    design decision).
//    MUST: run the comparison at **M = {@link GEMM_ROWS} (= limit + 1)**. Up to the limit the row-block GEMV now
//    takes over (K-21), so the former M=2 would be **another variant of the same family**, not the default path,
//    and the comparison would degrade into a self-comparison within the family. The activation `XS` is a function
//    of the flat index (row-major r·k + c), so the values of rows 0..m−1 are the same regardless of m = the first m
//    rows serve directly as the reference.
// 2. **Agreement with the CPU reference** — 1 alone cannot rule out "both paths broken by the same amount"
//    (structurally unlikely, but if the comparison somehow flowed into GEMV it would become vacuous).
// 3. **The check itself (the conditions of the branch)** — from the pipeline keys, the GEMV family comes up only
//    for 1 ≤ M ≤ limit (M=1 with the M=1 variant, M ≥ 2 with the row-block variant), while limit + 1, group 16 and
//    n%4≠0 keep the former `linear:` key.
//
// **Shape axes the row block (M ≥ 2) adds** (fragments that never run at M=1):
//   - `rows > 1` — one thread reads a weight word once and multiply-accumulates it with `rows` rows (only the
//     integer weight words and the scales are shared between rows)
//   - **multiple y tiles** (`ceil(m / rows) > 1`) — the weights are re-read per tile
//   - **a partial last y tile** (`m % rows != 0`) — out-of-range rows re-read the last row of x, and only the
//     write-back is dropped with `row < dims.m` (out-of-range reads and writes are handled separately)
//   - the y-tile shape with `rows == 1` — a different text from the M=1 variant (x read through the row start
//     `xr0` + the write-back guard); for small n it is chosen even at the limit M
//
// MUST: the shapes include **edges that the 12 production shapes never hit** (real-model decode only produces
// `n % 32 == 0` and `units % 4 == 0`, so neither the early return of the edge workgroup nor the remainder of the
// prefetch loop ever runs on a production shape):
//   - `n % 32 != 0` — the last workgroup is partial (the early return on `col >= dims.n` takes effect)
//   - `units % 4 != 0` (`units = k / 32`) — the 4-word prefetch loop leaves a remainder
//   - `units < 4` — the prefetch loop **never runs** and the reduction finishes in the remainder loop alone
// MUST: the weights **change magnitude per scale unit** (i4 = per group, i8 = per output channel).
// With the same scale for every unit, mixing up the index (i4 `(unit · 32) >> shift`, i8 `wscale[col]`) never
// shows in the values (the same trap as gpu_i4_weights_test.ts / gpu_f16_i8_weights_test.ts).
// MUST: alternate the signs of adjacent elements (mixing up the low / high nibble of the pack for i4, or the lanes
// within a word for i8, cancels out in the values with a symmetric pattern).
//
// The 2 storages hit **the same 3 viewpoints separately** (i4 = 32 elements per word × a group scale per word /
// i8 = 16 elements per word × a channel scale looked up once outside the reduction — separate branches of the
// generator, so green on one is no evidence for the other). For i8, `units` is `k / 16`, and the 3 edge
// conditions mean the same.
//
// Mutations it detects (fault injection checked at design time — 2026-08-31 / row block 2026-09-07):
// - shifting the group-scale shift by one step (`>> shift` → `>> shift+1`) → 1 fails
//   (the difference is about 3×, so the allclose of 2 fails too)
// - reordering the in-word multiply-accumulate to "high nibble first" (the set of products is the same; **only the
//   addition order** changes) → only 1 fails (the measured difference is 1 ULP = `0xc083c922` vs `0xc083c921`,
//   and the allclose of 2 lets it through). Proof that **this check means nothing unless it is exact u32 equality**.
// - putting the same reordering **only on the row-block side** (swapping the 2 low / high nibble lines of the
//   per-row MAC in `unitMacsI4`, without moving a single byte of what the M=1 variant (`row === undefined`)
//   generates) → **only 1 of the row block** fails (the i4 rows4 case, row 0 column 2: `0xc083c922` vs
//   `0xc083c921` = 1 ULP. The difference is 4.8e-7 / rel 1.2e-7, so the allclose of 2 lets it through and 3 is
//   green too). Meanwhile the 3 M=1 viewpoints, the i8 side and the 3 M=1 WGSL snapshots all stay green, and only
//   the row-block snapshots (linear_gemv_r*.wgsl) move — proof that **green at M=1 is no evidence for the row
//   block** (measured 2026-09-07).

import { test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import type { OpenedContainer } from "../src/format/container/open.ts";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import {
  defaultLinearGemvRowsVariant,
  LINEAR_GEMV_MAX_ROWS,
  linearGemvKey,
  linearGemvRowsForShape,
  linearGemvRowsKey,
} from "../src/kernels/linear-gemv.ts";
import type { WeightStorage } from "../src/kernels/weight-storage.ts";
import { linearKey } from "../src/kernels/linear.ts";
import { gemmRowsGeometry, type GeometryProfile } from "../src/kernels/geometry-profile.ts";
import { compareTensors, formatAllclose } from "../src/reference/allclose.ts";
import { GEMM_TOLERANCE } from "./helpers/op-tolerance.ts";
import { applyReferenceOp, type RefTensor, refTensor } from "../src/reference/ops.ts";
import {
  createSessionFromContainer,
  type RunInput,
  type SessionOptions,
  type Tensor,
} from "../src/runtime/executor.ts";
import type { TensorInput } from "./helpers/container-write.ts";
import { quantizeBf16 } from "./helpers/bf16.ts";
import { quantizeF16 } from "./helpers/f16.ts";
import {
  type DeclarationJson,
  f32Bytes,
  fill,
  type FilledTensor,
  GRAPH_NAME,
  openModelBytes,
} from "./helpers/model-fixture.ts";
import { quantizeI4 } from "./helpers/i4.ts";
import { quantizeI8 } from "./helpers/i8.ts";
import { deviceGeometryProfile } from "./helpers/geometry.ts";
import { GPU_AVAILABLE, SHADER_F16_AVAILABLE } from "./helpers/gpu.ts";
import { requireCensus } from "./helpers/pipeline-census.ts";

/**
 * Activations. A function of the flat index, so **the values of row r are the same regardless of m** (row-major
 * `r·k + c`) — the very condition that lets the m rows of GEMV be compared with the first m rows of the default
 * GEMM. Signs and magnitudes are spread enough to cause cancellation.
 */
const XS = (index: number): number => ((index % 11) - 5) * 0.375 + 0.125;

/**
 * The row count the bit-identity comparison runs at = **the smallest value outside the check** (limit + 1).
 * MUST: do not compare against a count at or below the limit — that is the range the GEMV family takes, so the
 * comparison would degrade into a vacuous self-comparison within the family.
 * NOTE: limit + 1 = 65 uses the geometry M64N32 (`gemmGeometryForRows` in gemm-geometry.ts), while the geometry the
 * GEMV family actually replaces for M ≤ 64 is M16N16. What this check sees directly is "GEMV == the M64N32 GEMM";
 * "== M16N16" follows transitively through bit identity across geometries (the cross-bucket check of
 * gpu_gemm_skinny_test.ts).
 */
const GEMM_ROWS = LINEAR_GEMV_MAX_ROWS + 1;

/**
 * Checks that the comparison **really ran on the default path**.
 * MUST: if the check's limit (recipe-builder) and `LINEAR_GEMV_MAX_ROWS` drift apart and the comparison flows into
 * the GEMV family, u32 equality becomes a vacuous self-comparison within the family — this stops that degradation
 * mechanically.
 */
const assertRanGemm = (name: string, keys: readonly string[]): void => {
  assert(
    !ranGemv(keys),
    `${name}: the comparison (M=${GEMM_ROWS}) ran on the GEMV family (breakdown: ${keys.join(" / ")})`,
  );
};

/** bias (in GEMV it is added once at the end, **outside** the reduction — if the order moves, it shows in the values). */
const BS = (index: number): number => ((index % 5) - 2) * 0.25;

/** Weights whose amplitude differs per group and whose adjacent elements alternate in sign (the 2 MUSTs above). */
const weightAt =
  (cols: number, groupSize: number) =>
  (index: number): number => {
    const group = Math.floor((index % cols) / groupSize);
    const row = Math.floor(index / cols);
    const base = (0.125 + (index % 11) * 0.5) * (index % 2 === 0 ? 1 : -1);
    return base * (1 + group * 0.75 + (row % 5) * 0.25);
  };

type GemvCase = {
  readonly name: string;
  readonly k: number;
  readonly n: number;
  readonly groupSize: number;
};

/**
 * Shape selection. `units = k / 32` (the number of weight words) and `n % 32` (how full the last workgroup is) are
 * independent axes, and the group length is the axis that moves the shift of the scale index.
 * MUST: `n % 4 == 0` (the check requires v4 — `buildLinear` in recipe-builders/linear.ts).
 */
const CASES: readonly GemvCase[] = [
  // A base shape without any edge (units = exactly 4; n = exactly 2 workgroups)
  { name: "divisible k128 n64 g32", k: 128, n: 64, groupSize: 32 },
  // units = 5 → 1 remainder word after the 4-word prefetch / n = 100 leaves only 4 columns in the last workgroup
  { name: "remainder units5 n100 g32", k: 160, n: 100, groupSize: 32 },
  // units = 6 (2 remainder words); group 64 = the scale changes every other word
  { name: "remainder units6 n36 g64", k: 192, n: 36, groupSize: 64 },
  // units = 8; group 128 = 1 scale per 4 words (the shape where baking the shift matters most)
  { name: "divisible units8 n68 g128", k: 256, n: 68, groupSize: 128 },
  // units = 3 < prefetch 4 → the prefetch loop never runs (the reduction finishes in the remainder loop alone)
  { name: "no prefetch units3 n4 g32", k: 96, n: 4, groupSize: 32 },
  // units = 7 (3 remainder words); n = 32 is exactly 1 workgroup
  { name: "remainder units7 n32 g32", k: 224, n: 32, groupSize: 32 },
];

/** A graph with one `linear(x, w, b)` (w is i4 + group scale). Only `m` decides the path. */
const linearI4Model = (
  testCase: GemvCase,
  m: number,
  quantized: ReturnType<typeof quantizeI4>,
  bias: FilledTensor,
): Promise<OpenedContainer> => {
  const { k, n, groupSize } = testCase;
  const declaration: DeclarationJson = {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["linear"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [m, k] }],
    outputs: ["y"],
    initializers: { w: {}, b: {} },
    values: {
      w: { dtype: "f32", shape: [n, k] },
      b: { dtype: "f32", shape: [n] },
      y: { dtype: "f32", shape: [m, n] },
    },
    nodes: [{ op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} }],
  };
  return openModelBytes(declaration, [
    {
      graph: GRAPH_NAME,
      initializer: "w",
      bytes: quantized.bytes,
      encoding: {
        codec: "int4-sym-g",
        groupSize,
        scale: { bytes: f32Bytes([...quantized.scale]), dtype: "f32" },
      },
    },
    {
      graph: GRAPH_NAME,
      initializer: "b",
      bytes: f32Bytes([...bias.data]),
      encoding: { codec: "f32" },
    },
  ]);
};

type RunResult = {
  readonly output: Tensor;
  /** The pipeline keys that actually ran in that run (census — independent of timing). */
  readonly keys: readonly string[];
};

/** Runs one assembled model on the given input and returns the output and the keys that ran. */
const runLinearOn = async (
  gpu: GpuContext,
  model: Promise<OpenedContainer>,
  x: RunInput,
  options: SessionOptions = {},
): Promise<RunResult> => {
  const session = await createSessionFromContainer(gpu, await model, GRAPH_NAME, options);
  try {
    const output = (await session.run({ x }))["y"];
    const census = requireCensus(session.diagnostics().lastRunPipelines, "linear");
    return { output, keys: census.map((row) => row.key) };
  } finally {
    await session.dispose();
  }
};

/** Runs one built model and returns the output and the keys that ran (shared by the 2 storages). */
const runLinear = (
  gpu: GpuContext,
  model: Promise<OpenedContainer>,
  m: number,
  k: number,
): Promise<RunResult> => runLinearOn(gpu, model, fill([m, k], XS));

/** The raw f32 bit pattern (looks at a 1-bit difference, -0 / NaN included). */
const bits = (tensor: Tensor): Uint32Array =>
  new Uint32Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.length);

test.skipIf(!GPU_AVAILABLE)(
  "an M=1 i4 linear does not differ by a single bit from the default path even on the GEMV family (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const testCase of CASES) {
        const { name, k, n, groupSize } = testCase;
        const weight = fill([n, k], weightAt(k, groupSize));
        const bias = fill([n], BS);
        const quantized = quantizeI4(weight.data, weight.shape, groupSize);

        const gemv = await runLinear(gpu, linearI4Model(testCase, 1, quantized, bias), 1, k);
        // The comparison is M = limit + 1 (outside the check = the default GEMM skeleton). The K reduction order per
        // output element does not depend on M (ADR 0022 decision 3 — the cross-bucket check of
        // gpu_gemm_skinny_test.ts looks at the same claim), so the first row is the reference for all GEMV outputs.
        const gemm = await runLinear(
          gpu,
          linearI4Model(testCase, GEMM_ROWS, quantized, bias),
          GEMM_ROWS,
          k,
        );
        assertRanGemm(name, gemm.keys);

        assertEquals(gemv.output.shape, [1, n], `${name}: output shape`);
        const actual = bits(gemv.output);
        const expected = bits(gemm.output);
        for (let col = 0; col < n; col += 1) {
          assert(
            actual[col] === expected[col],
            `${name}: column ${col} differs in bits from the default path (0x${actual[col].toString(16)} vs ` +
              `0x${expected[col].toString(16)} = ${gemv.output.data[col]} vs ` +
              `${gemm.output.data[col]}）`,
          );
        }

        // Rules out "both paths broken by the same amount" (if the comparison flowed into GEMV, 1 would be vacuous).
        const reference = applyReferenceOp(
          "linear",
          [
            fill([1, k], XS) as RefTensor,
            refTensor(weight.shape, quantized.values),
            bias as RefTensor,
          ],
          {},
          [1, n],
        );
        const report = compareTensors(gemv.output, reference, GEMM_TOLERANCE);
        assertEquals(report.pass, true, `${name}: ${formatAllclose(report)}`);
      }
    } finally {
      gpu.destroy();
    }
  },
);

/** The expected key of one check case (a branch of `buildLinear`). */
type DoorCase = {
  readonly name: string;
  readonly shape: GemvCase;
  readonly m: number;
  /** If true this shape goes to the GEMV family; if false it falls to the default `linear:`. */
  readonly gemv: boolean;
  /** The `linear:` key expected for a shape that falls to the default (unused for shapes that enter GEMV). */
  readonly fallback?: (profile: GeometryProfile) => string;
};

/**
 * Decides from the diagnostic keys "whether the GEMV family ran" by the family name alone.
 *
 * MUST: look at **the family-name prefix, not the presence of a specific variant key**. Looking at a variant key, a
 * shape for which the check should be closed but that ran **on another variant** would read as "the expected key
 * is missing" = the check is closed.
 */
const ranGemv = (keys: readonly string[]): boolean =>
  keys.some((key) => key.startsWith("linear_gemv:"));

/**
 * The family key expected for a shape where the check opens. M=1 is the M=1 variant (it does not move what decode
 * generates), M ≥ 2 the row-block variant, and `rows` comes from a pure function of (storage, m, n).
 */
const expectedGemvKey = (
  storage: WeightStorage,
  m: number,
  n: number,
  groupSize?: number,
): string =>
  m === 1
    ? linearGemvKey(storage, groupSize)
    : linearGemvRowsKey(storage, groupSize, defaultLinearGemvRowsVariant(storage, m, n));

/**
 * MUST: list **shapes that drop exactly one condition of the check at a time**. Dropping several at once makes it
 * impossible to tell which condition closes the check (or whether no condition is in effect at all).
 * MUST: the conditions other than the row count (group length / v4) are dropped one at a time **for both M=1 and
 * M ≥ 2**. The row block is a separate generator branch, so the check closing at M=1 is no evidence that it closes
 * at M ≥ 2.
 */
const DOOR_CASES: readonly DoorCase[] = [
  {
    name: "M=1 × i4 × g32 × v4",
    shape: { name: "", k: 128, n: 64, groupSize: 32 },
    m: 1,
    gemv: true,
  },
  {
    // The lower end satisfying the row-count condition (M ≥ 2 = the row-block variant — perf-ledger K-21)
    name: "M=2 × i4 × g32 × v4 (lower end of the row block)",
    shape: { name: "", k: 128, n: 64, groupSize: 32 },
    m: 2,
    gemv: true,
  },
  {
    // Exactly the limit (the upper end inside the check)
    name: `M=${LINEAR_GEMV_MAX_ROWS} × i4 × g32 × v4 (upper end of the row block)`,
    shape: { name: "", k: 128, n: 64, groupSize: 32 },
    m: LINEAR_GEMV_MAX_ROWS,
    gemv: true,
  },
  {
    // Drops only the row count (limit + 1 — beyond it, the default GEMM skeleton as before)
    name: `M=${GEMM_ROWS} (drops only the row-count condition)`,
    shape: { name: "", k: 128, n: 64, groupSize: 32 },
    m: GEMM_ROWS,
    gemv: false,
    fallback: (profile) =>
      linearKey("i4", true, "f32", GEMM_ROWS, 32, gemmRowsGeometry(profile, GEMM_ROWS)),
  },
  {
    // Drops only the group length (16 = a 32-element word crosses groups)
    name: "group 16 × M=1 (drops only the group-length condition)",
    shape: { name: "", k: 128, n: 64, groupSize: 16 },
    m: 1,
    gemv: false,
    fallback: (profile) => linearKey("i4", true, "f32", 1, 16, gemmRowsGeometry(profile, 1)),
  },
  {
    // Drops the same condition on the row-block side too
    name: "group 16 × M=2 (drops only the group-length condition)",
    shape: { name: "", k: 128, n: 64, groupSize: 16 },
    m: 2,
    gemv: false,
    fallback: (profile) => linearKey("i4", true, "f32", 2, 16, gemmRowsGeometry(profile, 2)),
  },
  {
    // Drops only v4 (n % 4 != 0 — to the default scalar variant)
    name: "n=33 × M=1 (drops only the v4 condition)",
    shape: { name: "", k: 128, n: 33, groupSize: 32 },
    m: 1,
    gemv: false,
    fallback: (profile) => linearKey("i4", false, "f32", 1, 32, gemmRowsGeometry(profile, 1)),
  },
  {
    // Drops the same condition on the row-block side too
    name: "n=33 × M=2 (drops only the v4 condition)",
    shape: { name: "", k: 128, n: 33, groupSize: 32 },
    m: 2,
    gemv: false,
    fallback: (profile) => linearKey("i4", false, "f32", 2, 32, gemmRowsGeometry(profile, 2)),
  },
];

test.skipIf(!GPU_AVAILABLE)(
  `the GEMV family check opens only for 1 ≤ M ≤ ${LINEAR_GEMV_MAX_ROWS} × i4 × group32 or more × v4` +
    " (real GPU / diagnostic keys)",
  async () => {
    const gpu = await acquireGpu();
    const profile = deviceGeometryProfile(gpu);
    try {
      for (const door of DOOR_CASES) {
        const { k, n, groupSize } = door.shape;
        const weight = fill([n, k], weightAt(k, groupSize));
        const bias = fill([n], BS);
        const quantized = quantizeI4(weight.data, weight.shape, groupSize);
        const { keys } = await runLinear(
          gpu,
          linearI4Model(door.shape, door.m, quantized, bias),
          door.m,
          k,
        );
        const shown = keys.join(" / ");
        assertEquals(
          ranGemv(keys),
          door.gemv,
          `${door.name}: whether the GEMV family ran differs from the expectation (breakdown of what ran: ${shown})`,
        );
        if (door.gemv) {
          // For an open shape, also check **on which variant** it opened (mixing up the M=1 variant and the
          // row-block variant is invisible from the family name alone).
          const key = expectedGemvKey("i4", door.m, n, groupSize);
          assert(
            keys.includes(key),
            `${door.name}: did not run on the expected variant key ${key} (breakdown: ${shown})`,
          );
        } else {
          const fallback = door.fallback?.(profile);
          assert(
            fallback !== undefined && keys.includes(fallback),
            `${door.name}: did not run on the default-path key ${fallback} (breakdown: ${shown})`,
          );
        }
      }
    } finally {
      gpu.destroy();
    }
  },
);

// ---------------------------------------------------------------------------
// i8 storage (lm_head — perf-ledger K-16). The same 3 viewpoints as i4, hit independently against the separate
// generator branch (16 elements per word, one scale per output channel).
// ---------------------------------------------------------------------------

type GemvI8Case = {
  readonly name: string;
  readonly k: number;
  readonly n: number;
};

/**
 * Weights whose amplitude (= scale) differs per output channel and whose adjacent elements alternate in sign.
 * MUST: vary the amplitude per channel — with the same scale for every channel, mixing up the `wscale[col]` index
 * never shows in the values (the same trap as the per-group MUST of i4).
 */
const weightAtI8 =
  (k: number) =>
  (index: number): number => {
    const row = Math.floor(index / k);
    const base = (0.125 + (index % 11) * 0.5) * (index % 2 === 0 ? 1 : -1);
    return base * (1 + (row % 7) * 0.5);
  };

/**
 * Shape selection. `units = k / 16` (the number of weight words) and `n % 32` (how full the last workgroup is) are
 * independent axes; the production shape (n 262,144 × k 1,536 = units 96, n % 32 == 0) hits no edge at all.
 * MUST: `n % 4 == 0` (the check requires v4 — `buildLinear` in recipe-builders/linear.ts).
 */
const I8_CASES: readonly GemvI8Case[] = [
  // A base shape without any edge (units = exactly 4; n = exactly 2 workgroups)
  { name: "divisible k64 n64", k: 64, n: 64 },
  // units = 5 → 1 remainder word after the 4-word prefetch / n = 100 leaves only 4 columns in the last workgroup
  { name: "remainder units5 n100", k: 80, n: 100 },
  // units = 6 (2 remainder words); n = 36 also leaves the last workgroup partial
  { name: "remainder units6 n36", k: 96, n: 36 },
  // units = 8 (divisible); n = 68 is 2 workgroups + 4 columns
  { name: "divisible units8 n68", k: 128, n: 68 },
  // units = 3 < prefetch 4 → the prefetch loop never runs (the reduction finishes in the remainder loop alone)
  { name: "no prefetch units3 n4", k: 48, n: 4 },
  // units = 7 (3 remainder words); n = 32 is exactly 1 workgroup
  { name: "remainder units7 n32", k: 112, n: 32 },
];

/** A graph with one `linear(x, w, b)` (w is i8 + a scale per output channel). Only `m` decides the path. */
const linearI8Model = (
  testCase: GemvI8Case,
  m: number,
  quantized: ReturnType<typeof quantizeI8>,
  bias: FilledTensor,
): Promise<OpenedContainer> => {
  const { k, n } = testCase;
  const declaration: DeclarationJson = {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["linear"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [m, k] }],
    outputs: ["y"],
    initializers: { w: {}, b: {} },
    values: {
      w: { dtype: "f32", shape: [n, k] },
      b: { dtype: "f32", shape: [n] },
      y: { dtype: "f32", shape: [m, n] },
    },
    nodes: [{ op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} }],
  };
  return openModelBytes(declaration, [
    {
      graph: GRAPH_NAME,
      initializer: "w",
      bytes: quantized.bytes,
      // per-channel (int8-sym) has groupSize = the row length k and groups = 1.
      encoding: {
        codec: "int8-sym",
        groupSize: k,
        scale: { bytes: f32Bytes([...quantized.scale]), dtype: "f32" },
      },
    },
    {
      graph: GRAPH_NAME,
      initializer: "b",
      bytes: f32Bytes([...bias.data]),
      encoding: { codec: "f32" },
    },
  ]);
};

test.skipIf(!GPU_AVAILABLE)(
  "an M=1 i8 linear does not differ by a single bit from the default path even on the GEMV family (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const testCase of I8_CASES) {
        const { name, k, n } = testCase;
        const weight = fill([n, k], weightAtI8(k));
        const bias = fill([n], BS);
        // The channel axis is 0 (linear weights are [n,k] — MUST ④ of ADR 0019 / 0024).
        const quantized = quantizeI8(weight.data, weight.shape, 0);

        const gemv = await runLinear(gpu, linearI8Model(testCase, 1, quantized, bias), 1, k);
        // The comparison is M = limit + 1 (outside the check = the default GEMM skeleton — same as the i4 side).
        const gemm = await runLinear(
          gpu,
          linearI8Model(testCase, GEMM_ROWS, quantized, bias),
          GEMM_ROWS,
          k,
        );
        assertRanGemm(name, gemm.keys);

        assertEquals(gemv.output.shape, [1, n], `${name}: output shape`);
        const actual = bits(gemv.output);
        const expected = bits(gemm.output);
        for (let col = 0; col < n; col += 1) {
          assert(
            actual[col] === expected[col],
            `${name}: column ${col} differs in bits from the default path (0x${actual[col].toString(16)} vs ` +
              `0x${expected[col].toString(16)} = ${gemv.output.data[col]} vs ` +
              `${gemm.output.data[col]}）`,
          );
        }

        // Rules out "both paths broken by the same amount" (if the comparison flowed into GEMV, 1 would be vacuous).
        const reference = applyReferenceOp(
          "linear",
          [
            fill([1, k], XS) as RefTensor,
            refTensor(weight.shape, quantized.values),
            bias as RefTensor,
          ],
          {},
          [1, n],
        );
        const report = compareTensors(gemv.output, reference, GEMM_TOLERANCE);
        assertEquals(report.pass, true, `${name}: ${formatAllclose(report)}`);
      }
    } finally {
      gpu.destroy();
    }
  },
);

/** The expected key of one check case (i8 side). i8 has no groups, so the axes are the 3 of M / k / n. */
type DoorI8Case = {
  readonly name: string;
  readonly shape: GemvI8Case;
  readonly m: number;
  readonly gemv: boolean;
  readonly fallback?: (profile: GeometryProfile) => string;
};

/**
 * MUST: list **shapes that drop exactly one condition of the check at a time** (the same discipline as the i4 side).
 * `k % 16 != 0` is an axis specific to i8; dropping it makes the 16-element-per-word reduction silently drop the
 * end of the row.
 * MUST: the conditions other than the row count are dropped for both M=1 and M ≥ 2 (same as the i4 side — separate
 * generator branches).
 */
const DOOR_I8_CASES: readonly DoorI8Case[] = [
  {
    name: "M=1 × i8 × k%16 × v4",
    shape: { name: "", k: 64, n: 64 },
    m: 1,
    gemv: true,
  },
  {
    // The lower end satisfying the row-count condition (M ≥ 2 = the row-block variant)
    name: "M=2 × i8 × k%16 × v4 (lower end of the row block)",
    shape: { name: "", k: 64, n: 64 },
    m: 2,
    gemv: true,
  },
  {
    // Exactly the limit (the upper end inside the check)
    name: `M=${LINEAR_GEMV_MAX_ROWS} × i8 × k%16 × v4 (upper end of the row block)`,
    shape: { name: "", k: 64, n: 64 },
    m: LINEAR_GEMV_MAX_ROWS,
    gemv: true,
  },
  {
    // Drops only the row count (limit + 1)
    name: `M=${GEMM_ROWS} (drops only the row-count condition)`,
    shape: { name: "", k: 64, n: 64 },
    m: GEMM_ROWS,
    gemv: false,
    fallback: (profile) =>
      linearKey("i8", true, "f32", GEMM_ROWS, undefined, gemmRowsGeometry(profile, GEMM_ROWS)),
  },
  {
    // Drops only the reduction step (k=68 is a multiple of 4, so v4 stays up)
    name: "k=68 × M=1 (drops only the k % 16 condition)",
    shape: { name: "", k: 68, n: 64 },
    m: 1,
    gemv: false,
    fallback: (profile) => linearKey("i8", true, "f32", 1, undefined, gemmRowsGeometry(profile, 1)),
  },
  {
    // Drops the same condition on the row-block side too
    name: "k=68 × M=2 (drops only the k % 16 condition)",
    shape: { name: "", k: 68, n: 64 },
    m: 2,
    gemv: false,
    fallback: (profile) => linearKey("i8", true, "f32", 2, undefined, gemmRowsGeometry(profile, 2)),
  },
  {
    // Drops only v4 (n % 4 != 0 — to the default scalar variant)
    name: "n=33 × M=1 (drops only the v4 condition)",
    shape: { name: "", k: 64, n: 33 },
    m: 1,
    gemv: false,
    fallback: (profile) =>
      linearKey("i8", false, "f32", 1, undefined, gemmRowsGeometry(profile, 1)),
  },
  {
    // Drops the same condition on the row-block side too
    name: "n=33 × M=2 (drops only the v4 condition)",
    shape: { name: "", k: 64, n: 33 },
    m: 2,
    gemv: false,
    fallback: (profile) =>
      linearKey("i8", false, "f32", 2, undefined, gemmRowsGeometry(profile, 2)),
  },
];

test.skipIf(!GPU_AVAILABLE)(
  `the GEMV family check opens only for 1 ≤ M ≤ ${LINEAR_GEMV_MAX_ROWS} × i8 × k%16 × v4` +
    " (real GPU / diagnostic keys)",
  async () => {
    const gpu = await acquireGpu();
    const profile = deviceGeometryProfile(gpu);
    try {
      for (const door of DOOR_I8_CASES) {
        const { k, n } = door.shape;
        const weight = fill([n, k], weightAtI8(k));
        const bias = fill([n], BS);
        const quantized = quantizeI8(weight.data, weight.shape, 0);
        const { keys } = await runLinear(
          gpu,
          linearI8Model(door.shape, door.m, quantized, bias),
          door.m,
          k,
        );
        const shown = keys.join(" / ");
        assertEquals(
          ranGemv(keys),
          door.gemv,
          `${door.name}: whether the GEMV family ran differs from the expectation (breakdown of what ran: ${shown})`,
        );
        if (door.gemv) {
          const key = expectedGemvKey("i8", door.m, n);
          assert(
            keys.includes(key),
            `${door.name}: did not run on the expected variant key ${key} (breakdown: ${shown})`,
          );
        } else {
          const fallback = door.fallback?.(profile);
          assert(
            fallback !== undefined && keys.includes(fallback),
            `${door.name}: did not run on the default-path key ${fallback} (breakdown: ${shown})`,
          );
        }
      }
    } finally {
      gpu.destroy();
    }
  },
);

// ---------------------------------------------------------------------------
// The row-block variant (M ≥ 2 — ADR 0082 addendum 5 / perf-ledger K-21). The same 3 viewpoints as the M=1 variant,
// aimed at **the 4 fragments the row block adds** (row sharing with rows > 1 / multiple y tiles / a partial last y
// tile / the y-tile shape with rows == 1). The 2 storages are separate branches, so they are hit independently.
// ---------------------------------------------------------------------------

/**
 * The shape of one row-block case. `rows` / `tiles` / `lastTileRows` are **values derived from the implementation
 * table of `linearGemvRowsForShape`**, recording which fragment the case intends to hit.
 *
 * MUST: state all 3. Writing only the shape without the intent lets the case silently slide to "only rows-1 y
 * tiles" when the selection changes, staying green while neither row sharing nor partial tiles ever run.
 */
type RowsCase = {
  readonly name: string;
  readonly k: number;
  readonly n: number;
  /** i4 only (i8 has no groups). */
  readonly groupSize?: number;
  readonly m: number;
  /** The expected row-block height (the rows one thread holds). */
  readonly rows: number;
  /** The expected number of y tiles = `ceil(m / rows)` (how many times the weights are re-read). */
  readonly tiles: number;
  /** The rows actually written back in the last y tile (a partial tile if `< rows`). */
  readonly lastTileRows: number;
  /** A long K checks the forward error bound from an independent CPU reference, not the short-reduction tolerance. */
  readonly longReference?: boolean;
};

/**
 * i4 shapes. `units = k / 32` (the remainder of the 4-word prefetch) and `n % 32` (how full the last workgroup is)
 * are the same axes as on the M=1 side, with the rows / y-tile axes added on top.
 * MUST: `n % 4 == 0` (the check requires v4).
 */
const ROWS_CASES: readonly RowsCase[] = [
  // Keeps the rounding of the normal GEMM even with a long K spanning function calls and a partial tile.
  {
    name: "long K, rows2 × partial tile (k4096 n4096 m9)",
    longReference: true,
    k: 4096,
    n: 4096,
    groupSize: 64,
    m: 9,
    rows: 2,
    tiles: 5,
    lastTileRows: 1,
  },
  // rows > 1 × multiple y tiles (the base shape where row sharing and re-reads both run; units exactly 4)
  {
    name: "rows4 × 8 tiles (k128 n2048 g32 m32)",
    k: 128,
    n: 2048,
    groupSize: 32,
    m: 32,
    rows: 4,
    tiles: 8,
    lastTileRows: 4,
  },
  // The ceiling rows (i4 caps at 32 elements per word × 8 rows = 256 — the compile-cost ceiling); units 3 < prefetch 4
  {
    name: "ceiling rows8 × 4 tiles × no prefetch (k96 n16384 g32 m32)",
    k: 96,
    n: 16384,
    groupSize: 32,
    m: 32,
    rows: 8,
    tiles: 4,
    lastTileRows: 8,
  },
  // The last y tile is partial (9 = 2×4 + 1 → only the last 1 row is written back); n % 32 = 4 makes the edge
  // workgroup partial too; units 5 (1 remainder word)
  {
    name: "partial tile rows2 × last 1 row × n%32≠0 (k160 n4100 g32 m9)",
    k: 160,
    n: 4100,
    groupSize: 32,
    m: 9,
    rows: 2,
    tiles: 5,
    lastTileRows: 1,
  },
  // Partial tile (7 = 4 + 3) × group 64 (the scale shift does not reduce to the identity); units 6 (2 remainder words)
  {
    name: "partial tile rows4 × last 3 rows × g64 (k192 n16384 g64 m7)",
    k: 192,
    n: 16384,
    groupSize: 64,
    m: 7,
    rows: 4,
    tiles: 2,
    lastTileRows: 3,
  },
  // The y-tile shape with rows 1 (for small n it is chosen even at M near the limit) — a different text from the M=1 variant
  {
    name: "rows1 × 33 tiles (k128 n64 g32 m33)",
    k: 128,
    n: 64,
    groupSize: 32,
    m: 33,
    rows: 1,
    tiles: 33,
    lastTileRows: 1,
  },
  // Exactly the upper end of the check (M = LINEAR_GEMV_MAX_ROWS); units 7 (3 remainder words)
  {
    name: "limit m64 × ceiling rows8 × 8 tiles (k224 n8192 g32)",
    k: 224,
    n: 8192,
    groupSize: 32,
    m: LINEAR_GEMV_MAX_ROWS,
    rows: 8,
    tiles: 8,
    lastTileRows: 8,
  },
];

/**
 * i8 shapes. `units = k / 16`, and the ceiling is 16 elements per word × 16 rows = 256 (up to twice the rows of i4).
 * MUST: `n % 4 == 0` (the check requires v4).
 */
const ROWS_I8_CASES: readonly RowsCase[] = [
  // Keeps the rounding of the normal GEMM even with a long K spanning function calls and a partial tile.
  {
    name: "long K, rows2 × partial tile (k4096 n4096 m9)",
    longReference: true,
    k: 4096,
    n: 4096,
    m: 9,
    rows: 2,
    tiles: 5,
    lastTileRows: 1,
  },
  // rows > 1 × multiple y tiles; units 5 (1 remainder word)
  {
    name: "rows8 × 4 tiles (k80 n4096 m32)",
    k: 80,
    n: 4096,
    m: 32,
    rows: 8,
    tiles: 4,
    lastTileRows: 8,
  },
  // The ceiling rows 16 (the point of this case is that it differs from the 8 of i4); 2 y tiles; units 3
  {
    name: "ceiling rows16 × 2 tiles × no prefetch (k48 n16384 m32)",
    k: 48,
    n: 16384,
    m: 32,
    rows: 16,
    tiles: 2,
    lastTileRows: 16,
  },
  // The last y tile is partial (9 = 2×4 + 1); n % 32 = 4; units 7 (3 remainder words)
  {
    name: "partial tile rows2 × last 1 row × n%32≠0 (k112 n4100 m9)",
    k: 112,
    n: 4100,
    m: 9,
    rows: 2,
    tiles: 5,
    lastTileRows: 1,
  },
  // The y-tile shape with rows 1
  {
    name: "rows1 × 33 tiles (k64 n64 m33)",
    k: 64,
    n: 64,
    m: 33,
    rows: 1,
    tiles: 33,
    lastTileRows: 1,
  },
  // Exactly the upper end of the check × ceiling rows 16 (4 y tiles)
  {
    name: "limit m64 × ceiling rows16 × 4 tiles (k112 n8192)",
    k: 112,
    n: 8192,
    m: LINEAR_GEMV_MAX_ROWS,
    rows: 16,
    tiles: 4,
    lastTileRows: 16,
  },
];

/**
 * Checks of one row-block case (shared by the 2 storages).
 *
 * ① The intent of the shape (rows / number of y tiles / rows of the partial tile) matches the implementation table
 * ② Exact u32 equality with **the first m rows** of the default GEMM (M = {@link GEMM_ROWS}) (every element)
 * ③ allclose with the CPU reference (rules out both paths broken by the same amount)
 * ④ It actually ran on the expected row-block variant key (only on devices with diagnostics — rules out the
 *    vacuous form where ② and ③ stay green on the default path)
 */
const checkRowsCase = async (
  gpu: GpuContext,
  storage: WeightStorage,
  rowsCase: RowsCase,
  build: (m: number) => Promise<OpenedContainer>,
  weightShape: readonly number[],
  values: Float32Array<ArrayBuffer>,
  bias: FilledTensor,
): Promise<void> => {
  const { name, k, n, m, groupSize } = rowsCase;
  // ① The intent of the shape (if the implementation table moves, the case turns red together with its aim)
  assertEquals(linearGemvRowsForShape(storage, m, n), rowsCase.rows, `${name}: row-block height`);
  const tiles = Math.ceil(m / rowsCase.rows);
  assertEquals(tiles, rowsCase.tiles, `${name}: number of y tiles`);
  assertEquals(
    m - (tiles - 1) * rowsCase.rows,
    rowsCase.lastTileRows,
    `${name}: valid rows of the last y tile`,
  );

  const gemv = await runLinear(gpu, build(m), m, k);
  const gemm = await runLinear(gpu, build(GEMM_ROWS), GEMM_ROWS, k);
  assertRanGemm(name, gemm.keys);

  // ② Exact u32 equality of every element (on the GEMM side the first m rows = the same flat-index range)
  assertEquals(gemv.output.shape, [m, n], `${name}: output shape`);
  const actual = bits(gemv.output);
  const expected = bits(gemm.output);
  for (let index = 0; index < m * n; index += 1) {
    if (actual[index] === expected[index]) continue;
    assert(
      false,
      `${name}: row ${Math.floor(index / n)} column ${index % n} differs in bits from the default path` +
        `（0x${actual[index].toString(16)} vs 0x${expected[index].toString(16)} = ` +
        `${gemv.output.data[index]} vs ${gemm.output.data[index]}）`,
    );
  }

  // ③ CPU reference. The existing tolerance for short K is used as is.
  const reference = applyReferenceOp(
    "linear",
    [fill([m, k], XS) as RefTensor, refTensor(weightShape, values), bias as RefTensor],
    {},
    [m, n],
  );
  if (rowsCase.longReference) {
    // Do not extrapolate the empirical band of K=72 to K=4096. With u=2^-23, which includes rounding to an adjacent
    // f32 value, the bound is γ(2K+2)·Σ|terms| from K multiplications, K additions, the bias and the reference's
    // final rounding. This input is finite and normal. Whether FMA is used is not pinned. Optimization regressions
    // are caught by the all-bit comparison of ②.
    const input = fill([m, k], XS).data;
    const errorFactor = (2 * k + 2) * 2 ** -23;
    const gamma = errorFactor / (1 - errorFactor);
    for (let row = 0; row < m; row++) {
      for (let col = 0; col < n; col++) {
        let magnitude = Math.abs(bias.data[col]);
        for (let inner = 0; inner < k; inner++) {
          magnitude += Math.abs(input[row * k + inner] * values[col * k + inner]);
        }
        const index = row * n + col;
        const error = Math.abs(gemv.output.data[index] - reference.data[index]);
        assert(
          error <= gamma * magnitude,
          `${name}: CPU reference ${index}: ${error} > ${gamma * magnitude}`,
        );
      }
    }
  } else {
    const report = compareTensors(gemv.output, reference, GEMM_TOLERANCE);
    assertEquals(report.pass, true, `${name}: ${formatAllclose(report)}`);
  }

  // ④ It ran on the row-block variant key (with only ② and ③ it would stay green on the default path)
  const key = linearGemvRowsKey(storage, groupSize, defaultLinearGemvRowsVariant(storage, m, n));
  assert(
    gemv.keys.includes(key),
    `${name}: did not run on the row-block key ${key} (breakdown: ${gemv.keys.join(" / ")})`,
  );
};

test.skipIf(!GPU_AVAILABLE)(
  "a row-block (M ≥ 2) i4 linear does not differ by a single bit from the default path (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const rowsCase of ROWS_CASES) {
        const { k, n } = rowsCase;
        const groupSize = rowsCase.groupSize ?? 32;
        const weight = fill([n, k], weightAt(k, groupSize));
        const bias = fill([n], BS);
        const quantized = quantizeI4(weight.data, weight.shape, groupSize);
        const shape: GemvCase = { name: rowsCase.name, k, n, groupSize };
        await checkRowsCase(
          gpu,
          "i4",
          rowsCase,
          (m) => linearI4Model(shape, m, quantized, bias),
          weight.shape,
          quantized.values,
          bias,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "a row-block (M ≥ 2) i8 linear does not differ by a single bit from the default path (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const rowsCase of ROWS_I8_CASES) {
        const { k, n } = rowsCase;
        const weight = fill([n, k], weightAtI8(k));
        const bias = fill([n], BS);
        // The channel axis is 0 (linear weights are [n,k] — MUST ④ of ADR 0019 / 0024).
        const quantized = quantizeI8(weight.data, weight.shape, 0);
        const shape: GemvI8Case = { name: rowsCase.name, k, n };
        await checkRowsCase(
          gpu,
          "i8",
          rowsCase,
          (m) => linearI8Model(shape, m, quantized, bias),
          weight.shape,
          quantized.values,
          bias,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
);

// ---------------------------------------------------------------------------
// Float storage (f16 / f32 — ADR 0082 addenda 6 and 7). The same three viewpoints as the
// quantized storages, run per storage because each storage is its own generated branch.
// The float GEMV gate admits M=1 only, so the bit-identity partner is M=2: that still runs the
// M16N16 GEMM skeleton and is not another member of the family. The shapes include N edges and
// prefetch edges real models never reach, plus the packed-f16 pair / word boundaries (f16) and
// the component / word boundaries (f32).
// ---------------------------------------------------------------------------

type FloatStorage = "f16" | "bf16" | "f32";

/** Activation of the float-storage cases (row-major, so row 0 does not depend on m). */
const floatActivation = (length: number): Float32Array<ArrayBuffer> =>
  Float32Array.from({ length }, (_, i) => Math.sin(i * 0.73) * 1.17);

/**
 * `linear(x, w, b)` with float-stored weights. f16 storage always holds f16-rounded values (the
 * CPU reference sees the same rounded values); `roundToF16` rounds f32-stored weights too.
 */
const floatLinearFixture = (
  m: number,
  n: number,
  k: number,
  storage: FloatStorage,
  roundToF16 = false,
) => {
  // Scatter sign and magnitude between neighbours so a swapped lo/hi of a pair does not cancel out.
  const raw = Float32Array.from(
    { length: n * k },
    (_, i) => (i % 2 === 0 ? 1 : -1) * (0.0013 + (i % 23) * 0.017),
  );
  const rounded = storage === "bf16" ? quantizeBf16(raw) : quantizeF16(raw);
  const weight = storage === "f16" || storage === "bf16" || roundToF16 ? rounded.values : raw;
  const bias = Float32Array.from({ length: n }, (_, i) => Math.sin(i * 0.37) * 0.13);
  const declaration: DeclarationJson = {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["linear"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [m, k] }],
    outputs: ["y"],
    initializers: { w: {}, b: {} },
    values: {
      w: { dtype: "f32", shape: [n, k] },
      b: { dtype: "f32", shape: [n] },
      y: { dtype: "f32", shape: [m, n] },
    },
    nodes: [{ op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} }],
  };
  const tensors: readonly TensorInput[] = [
    {
      graph: GRAPH_NAME,
      initializer: "w",
      bytes: storage === "f32" ? f32Bytes([...weight]) : rounded.bytes,
      encoding: { codec: storage },
    },
    {
      graph: GRAPH_NAME,
      initializer: "b",
      bytes: f32Bytes([...bias]),
      encoding: { codec: "f32" },
    },
  ];
  const input = refTensor([m, k], floatActivation(m * k));
  return {
    model: () => openModelBytes(declaration, tensors),
    input,
    reference: () =>
      applyReferenceOp("linear", [input, refTensor([n, k], weight), refTensor([n], bias)], {}, [
        m,
        n,
      ]),
  };
};

type FloatLinearFixture = ReturnType<typeof floatLinearFixture>;

const runFloatLinear = (
  gpu: GpuContext,
  data: FloatLinearFixture,
  options: SessionOptions = {},
): Promise<RunResult> => runLinearOn(gpu, data.model(), data.input, options);

/**
 * M=1 float GEMV against the default GEMM (M=2, first row) bit for bit, and against the CPU
 * reference.
 */
const checkFloatGemvBits = async (
  storage: FloatStorage,
  shapes: readonly (readonly [number, number])[],
): Promise<void> => {
  const gpu = await acquireGpu();
  const profile = deviceGeometryProfile(gpu);
  try {
    for (const [n, k] of shapes) {
      const data = floatLinearFixture(1, n, k, storage);
      const actual = await runFloatLinear(gpu, data);
      // The float gate is M=1 only. M=2 runs the same M16N16 GEMM skeleton, not a family variant.
      const expected = await runFloatLinear(gpu, floatLinearFixture(2, n, k, storage));
      assertEquals(actual.keys, [linearGemvKey(storage)]);
      assertEquals(expected.keys, [
        linearKey(storage, true, "f32", 2, undefined, gemmRowsGeometry(profile, 2)),
      ]);
      assertEquals(actual.output.shape, [1, n]);
      assertEquals(bits(actual.output), bits(expected.output).subarray(0, n), `n=${n} k=${k}`);
      const report = compareTensors(actual.output, data.reference(), GEMM_TOLERANCE);
      assert(report.pass, formatAllclose(report));
    }
  } finally {
    gpu.destroy();
  }
};

test.skipIf(!GPU_AVAILABLE)(
  "an f16 M=1 GEMV is bit-identical to the normal GEMM even on edge shapes and satisfies the CPU reference (real GPU)",
  () =>
    checkFloatGemvBits("f16", [
      [64, 128],
      [100, 40],
      [36, 48],
      [68, 56],
      [4, 8],
      [32, 24],
      [36, 136],
      [100, 264],
    ]),
);

// bf16 shares f16's packing (8 elements per 16-byte word), so the same edge shapes apply.
test.skipIf(!GPU_AVAILABLE)(
  "a bf16 M=1 GEMV is bit-identical to the normal GEMM even on edge shapes and satisfies the CPU reference (real GPU)",
  () =>
    checkFloatGemvBits("bf16", [
      [64, 128],
      [100, 40],
      [36, 48],
      [68, 56],
      [4, 8],
      [32, 24],
      [36, 136],
      [100, 264],
    ]),
);

test.skipIf(!GPU_AVAILABLE)(
  "an f32 M=1 GEMV is bit-identical to the normal GEMM even on edge shapes and satisfies the CPU reference (real GPU)",
  () =>
    checkFloatGemvBits("f32", [
      [64, 128],
      [100, 20],
      [36, 24],
      [68, 28],
      [4, 4],
      [32, 12],
      [36, 132],
      [100, 260],
    ]),
);

/** One float gate case. `gemv` marks the shape that the gate admits. */
type FloatDoorCase = {
  readonly m: number;
  readonly n: number;
  readonly k: number;
  readonly storage: FloatStorage;
  readonly gemv?: boolean;
  /** f32 storage holding f16-representable values. */
  readonly roundToF16?: boolean;
};

const checkFloatDoor = async (cases: readonly FloatDoorCase[]): Promise<void> => {
  const gpu = await acquireGpu();
  const profile = deviceGeometryProfile(gpu);
  try {
    for (const c of cases) {
      const data = floatLinearFixture(c.m, c.n, c.k, c.storage, c.roundToF16);
      const actual = await runFloatLinear(gpu, data);
      assertEquals(actual.keys, [
        c.gemv === true
          ? linearGemvKey(c.storage)
          : linearKey(
              c.storage,
              c.k % 4 === 0 && c.n % 4 === 0,
              "f32",
              c.m,
              undefined,
              gemmRowsGeometry(profile, c.m),
            ),
      ]);
      const report = compareTensors(actual.output, data.reference(), GEMM_TOLERANCE);
      assert(report.pass, formatAllclose(report));
    }
  } finally {
    gpu.destroy();
  }
};

test.skipIf(!GPU_AVAILABLE)(
  "checks the row-count, K-alignment and N-alignment conditions of the f16 GEMV and the separate variant for f32 storage (real GPU)",
  () =>
    checkFloatDoor([
      { m: 2, n: 36, k: 40, storage: "f16" },
      { m: 1, n: 36, k: 36, storage: "f16" },
      { m: 1, n: 35, k: 40, storage: "f16" },
      { m: 1, n: 36, k: 40, storage: "f32", gemv: true, roundToF16: true },
    ]),
);

test.skipIf(!GPU_AVAILABLE)(
  "the row-count, K-alignment and N-alignment conditions of the bf16 GEMV open under the same conditions as f16 (real GPU)",
  () =>
    checkFloatDoor([
      { m: 2, n: 36, k: 40, storage: "bf16" },
      { m: 1, n: 36, k: 36, storage: "bf16" },
      { m: 1, n: 35, k: 40, storage: "bf16" },
      { m: 1, n: 36, k: 40, storage: "bf16", gemv: true },
    ]),
);

test.skipIf(!GPU_AVAILABLE)(
  "the f32 GEMV selects the normal GEMM when a row-count, K-alignment or N-alignment condition is dropped (real GPU)",
  () =>
    checkFloatDoor([
      { m: 2, n: 36, k: 20, storage: "f32" },
      { m: 1, n: 36, k: 18, storage: "f32" },
      { m: 1, n: 35, k: 20, storage: "f32" },
    ]),
);

/** An f16-compute request keeps the GEMM key (checks the key only; numbers live above). */
const checkF16ComputeKeepsGemm = async (storage: FloatStorage): Promise<void> => {
  const gpu = await acquireGpu({ shaderF16: true });
  const profile = deviceGeometryProfile(gpu);
  try {
    const actual = await runFloatLinear(gpu, floatLinearFixture(1, 36, 40, storage), {
      linearCompute: "f16",
    });
    assertEquals(actual.keys, [
      linearKey(storage, true, "f16", 1, undefined, gemmRowsGeometry(profile, 1)),
    ]);
  } finally {
    gpu.destroy();
  }
};

test.skipIf(!SHADER_F16_AVAILABLE)(
  "even with f16 storage, an f16 compute request is not replaced by GEMV (real GPU)",
  () => checkF16ComputeKeepsGemm("f16"),
);

test.skipIf(!SHADER_F16_AVAILABLE)(
  "even with f32 storage, an f16 compute request is not replaced by GEMV (real GPU)",
  () => checkF16ComputeKeepsGemm("f32"),
);
