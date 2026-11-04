// Numeric comparison of real GPU execution vs the CPU reference (stage 2 of ADR 0005). Runs every M0 op on representative shapes.
// MUST: no random numbers — if a failure does not reproduce, its cause cannot be isolated.

import { test } from "vite-plus/test";
import { assert, assertEquals, assertNotEquals, assertRejects, assertThrows } from "@std/assert";
import { CodegenError } from "../src/codegen/errors.ts";
import { acquireGpu, type GpuContext, LIMIT_CAPS } from "../src/gpu/device.ts";
import { TOPK_WORKGROUP_SIZE } from "../src/kernels/topk.ts";
import { compareTensors, formatAllclose } from "../src/reference/allclose.ts";
import {
  applyReferenceOp,
  applyReferenceOpOutputs,
  type RefTensor,
  refTensor,
} from "../src/reference/ops.ts";
import { createSessionFromContainer, type Tensor } from "../src/runtime/executor.ts";
import {
  ARGMAX_CASES,
  ARGMAX_TIEBREAK_INPUT,
  ARGMAX_TIEBREAK_TORCH,
  ATTENTION_CASES,
  BINARY_CASES,
  BMM_CASES,
  BOUNDARY_CASES,
  DEFORM_CASES,
  DTYPE_CASES,
  FUSED_CASES,
  GATHER_CASES,
  GROUP_NORM_CASES,
  GRU_SCAN_CASES,
  INTEGERS,
  LAYOUT2_CASES,
  LAYOUT_CASES,
  MATH_CASES,
  MATMUL_CASES,
  type OpCase,
  POSITIVE,
  REDUCE_CASES,
  SIGNED,
  TOPK_CASES,
  UNARY_CASES,
  UPSAMPLE_CASES,
  UPSAMPLE_FRACTIONAL_CASES,
} from "./helpers/gpu_op_cases.ts";
import { BIT_IDENTICAL_OPS, opTolerance } from "./helpers/op-tolerance.ts";
import {
  type DeclarationJson,
  fill,
  GRAPH_NAME,
  openModelBytes,
  outputName,
  singleOpDeclaration,
} from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import {
  type RankMutation,
  type RankPipelineCache,
  runArgmaxDirect,
  runTopkDirect,
} from "./helpers/rank-dispatch.ts";
import { assertMutated } from "./helpers/state-dispatch.ts";

/**
 * Runs one case on the real GPU and returns **the outputs in ascending slot order** (ADR 0068 decision 1). For a
 * single-output op it is a list of length 1; only `topk` has 2.
 */
const runOutputs = async (gpu: GpuContext, testCase: OpCase): Promise<readonly Tensor[]> => {
  const graph = singleOpDeclaration(
    testCase.op,
    testCase.inputs.map((input) => input.shape),
    testCase.outShapes,
    {
      inDtypes: testCase.inputs.map((input) => input.dtype),
      outDtypes: testCase.outShapes.map(
        (_, slot) => testCase.outDtypes?.[slot] ?? testCase.inputs[0].dtype,
      ),
      attrs: testCase.attrs,
    },
  );
  const session = await createSessionFromContainer(
    gpu,
    await openModelBytes(graph, []),
    GRAPH_NAME,
  );
  try {
    const named: Record<string, Tensor> = {};
    testCase.inputs.forEach((input, index) => {
      named[`x${index}`] = input;
    });
    const outputs = await session.run(named);
    return testCase.outShapes.map((_, slot) => outputs[outputName(slot)]);
  } finally {
    await session.dispose();
  }
};

/** Runs a single-output case (multi-output ops go through {@link runOutputs}). */
const runCase = async (gpu: GpuContext, testCase: OpCase): Promise<Tensor> => {
  const outputs = await runOutputs(gpu, testCase);
  assertEquals(outputs.length, 1, `${testCase.name}: single-output case`);
  return outputs[0];
};

const checkAll = async (cases: readonly OpCase[]): Promise<void> => {
  const gpu = await acquireGpu();
  try {
    for (const testCase of cases) {
      const actual = await runOutputs(gpu, testCase);
      const expected = applyReferenceOpOutputs(
        testCase.op,
        testCase.inputs,
        testCase.attrs ?? {},
        testCase.outShapes[0],
      );
      // MUST: compare the counts first (keeps a multi-output op that skips reading one output from going green)
      assertEquals(actual.length, expected.length, `${testCase.name}: number of outputs`);
      actual.forEach((tensor, slot) => {
        const label = expected.length === 1 ? testCase.name : `${testCase.name} output ${slot}`;
        assertEquals(tensor.shape, expected[slot].shape, label);
        assertEquals(tensor.dtype, expected[slot].dtype, `${label}: dtype`);
        // f32 uses the per-op tolerance table (helpers/op-tolerance.ts — an op missing from the table fails loudly);
        // for i32 / bool, compareTensors picks exact equality by dtype (ADR 0009)
        const report = compareTensors(
          tensor,
          expected[slot],
          testCase.tolerance ?? opTolerance(testCase.op),
        );
        assertEquals(report.pass, true, `${label}: ${formatAllclose(report)}`);
      });
    }
  } finally {
    gpu.destroy();
  }
};

test.skipIf(!GPU_AVAILABLE)("11 unary ops match the CPU reference (real GPU)", () =>
  checkAll(UNARY_CASES),
);

test.skipIf(!GPU_AVAILABLE)(
  "4 binary ops match the CPU reference, broadcast included (real GPU)",
  () => checkAll(BINARY_CASES),
);

test.skipIf(!GPU_AVAILABLE)(
  "i32 / bool elementwise (mask path) matches the CPU reference exactly (real GPU)",
  () => checkAll(DTYPE_CASES),
);

test.skipIf(!GPU_AVAILABLE)(
  "the wave 3 math ops (where / comparisons / clamp / leaky_relu / log1p / cumsum / bool sum) match the CPU reference (real GPU)",
  () => checkAll(MATH_CASES),
);

/**
 * NaN propagation of leaky_relu (pins the ruling of ADR 0015 on a real GPU).
 *
 * torch gives `leaky_relu(NaN) = NaN`, but WGSL's `max` may have implementations that do not guarantee NaN
 * propagation (that relu / clamp already carry the same divergence is a known item in limitations.md).
 * MUST: allclose fails NaN on either side, so this check cannot go into checkAll
 * — it is kept as a dedicated test that looks at the raw values.
 */
test.skipIf(!GPU_AVAILABLE)(
  "leaky_relu propagates NaN (the ruling for the select form — real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      const actual = await runCase(gpu, {
        name: "leaky_relu NaN",
        op: "leaky_relu",
        inputs: [fill([4], (i) => [Number.NaN, -2, 0, 3][i])],
        outShapes: [[4]],
        attrs: { negative_slope: 0.1 },
      });
      const values = [...actual.data];
      assertEquals(
        values.map((v) => Number.isNaN(v)),
        [true, false, false, false],
      );
      // Everything but NaN takes the normal path (slope times on the negative side, pass-through on the positive side)
      assertEquals(values[1], Math.fround(-0.2));
      assertEquals(values[2], 0);
      assertEquals(values[3], 3);
    } finally {
      gpu.destroy();
    }
  },
);

/** NaN positions (first / middle / last). Indices into the elementwise input of length 4. */
const NAN_POSITIONS = [0, 2, 3] as const;
/** The plain sequence before NaN is inserted. Spans the lower side, upper side and pass-through of clamp (min=-1 / max=1). */
const NAN_BASE = [-3, -0.5, 0.5, 3] as const;

/** Builds, for one op, "3 inputs with NaN at different positions + 1 NaN-free control". */
const nanUnaryCases = (op: string, attrs?: Record<string, unknown>): readonly OpCase[] => [
  ...NAN_POSITIONS.map((position) => ({
    name: `${op} NaN@${position}`,
    op,
    inputs: [fill([4], (i) => (i === position ? Number.NaN : NAN_BASE[i]))],
    outShapes: [[4]],
    attrs,
  })),
  {
    name: `${op} without NaN (control)`,
    op,
    inputs: [fill([4], (i) => NAN_BASE[i])],
    outShapes: [[4]],
    attrs,
  },
];

/** The NaN position per row (-1 = a NaN-free control row). A reduction propagates it "wherever it is in the row". */
const NAN_REDUCE_AT = [0, 2, 3, -1] as const;
const NAN_REDUCE_ROWS = (index: number): number => {
  const row = Math.floor(index / 4);
  const col = index % 4;
  return col === NAN_REDUCE_AT[row] ? Number.NaN : (col - 1.5) * (row + 1);
};

const NAN_CASES: readonly OpCase[] = [
  ...nanUnaryCases("clamp", { min: -1, max: 1 }),
  ...nanUnaryCases("clamp_min", { min: 0 }),
  ...nanUnaryCases("relu"),
  ...(["amax", "amin"] as const).flatMap((op): readonly OpCase[] => [
    {
      name: `${op} NaN at varying positions in the row [4,4]`,
      op,
      inputs: [fill([4, 4], NAN_REDUCE_ROWS)],
      outShapes: [[4]],
      attrs: { dim: 1 },
    },
    {
      // Row length 300 > workgroup size 256 → index 299 is read in the **second pass** of one thread's scan loop
      // ([4,4] only runs the first pass, so this path is hit only here).
      // Row 1 is the NaN-free control.
      name: `${op} NaN in the second pass of the scan loop [2,300]`,
      op,
      inputs: [fill([2, 300], (i) => (i === 299 ? Number.NaN : SIGNED(i)))],
      outShapes: [[2]],
      attrs: { dim: 1 },
    },
  ]),
];

/**
 * NaN propagation of clamp / clamp_min / relu / amax / amin (pins the ruling of bit-pattern checks on a real GPU).
 *
 * For all 5 ops torch gives "if the input (for a reduction, the words being reduced) has a NaN, the result is
 * NaN". The driver's `max` / `min` swallow NaN, so only the bit-pattern check of the generated WGSL carries the
 * propagation (the mechanism is IS_NAN_BITS_WGSL in src/codegen/elementwise.ts).
 * MUST: allclose fails NaN on either side, so this cannot go into checkAll — compare by **matching the isNaN
 * pattern against the CPU reference** (matching the NaN bit patterns themselves is not required).
 */
test.skipIf(!GPU_AVAILABLE)(
  "clamp / clamp_min / relu / amax / amin propagate NaN (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const testCase of NAN_CASES) {
        const actual = await runCase(gpu, testCase);
        const expected = applyReferenceOp(
          testCase.op,
          testCase.inputs,
          testCase.attrs ?? {},
          testCase.outShapes[0],
        );
        const actualValues = [...actual.data];
        const expectedValues = [...expected.data];
        const actualNan = actualValues.map((value) => Number.isNaN(value));
        const expectedNan = expectedValues.map((value) => Number.isNaN(value));
        assertEquals(actualNan, expectedNan, `${testCase.name}: NaN positions`);
        // MUST: have control elements. An implementation that "always returns NaN" passes a position match alone.
        assertEquals(expectedNan.includes(false), true, `${testCase.name}: no control elements`);
        // The non-NaN side has values without rounding differences, so they are compared raw and exactly (NaN never
        // compares equal, so only those elements are flattened to 0 before checking)
        const finiteOnly = (values: readonly number[], isNan: readonly boolean[]): number[] =>
          values.map((value, index) => (isNan[index] ? 0 : value));
        assertEquals(
          finiteOnly(actualValues, actualNan),
          finiteOnly(expectedValues, expectedNan),
          `${testCase.name}: non-NaN values`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * Precision of log1p (the measured basis of the implementation choice — LOG1P_FN in src/codegen/elementwise.ts).
 *
 * The naive `log(1 + x)` loses significant digits to the rounding of `1 + x` for |x| ≪ 1. Here x includes the
 * f32 range where "1 + x rounds to 1", and the **relative error** against the CPU reference (`Math.log1p`) is
 * measured to stay within a few f32 ulp. The naive form gives output 0 (relative error 1) at x = 1e-8 and does not
 * pass this threshold.
 */
test.skipIf(!GPU_AVAILABLE)(
  "log1p stays within a few f32 ulp of relative error even where 1+x rounds (real GPU)",
  async () => {
    const smalls = [1e-8, -1e-8, 1e-7, 5e-7, -3e-6, 1e-4, -1e-3, 0.5, 2, 100];
    const gpu = await acquireGpu();
    let worst = 0;
    try {
      const actual = await runCase(gpu, {
        name: "log1p small",
        op: "log1p",
        inputs: [fill([smalls.length], (i) => smalls[i])],
        outShapes: [[smalls.length]],
      });
      smalls.forEach((x, index) => {
        const expected = Math.log1p(Math.fround(x));
        const relative = Math.abs((actual.data[index] - expected) / expected);
        worst = Math.max(worst, relative);
      });
    } finally {
      gpu.destroy();
    }
    // The f32 eps is 1.19e-7. Allowing a few ulp for the composition of log / division / multiplication, the cap is 8×
    // (the naive form's error at x = 1e-8 is 1.0 = 7 orders of magnitude above this threshold).
    assertEquals(worst < 1e-6, true, `worst relative error of log1p ${worst}`);
  },
);

/**
 * **Bit-identity check** of the GEMM / conv / reduction ops (op-tolerance research §7.1 = the second half of option γ).
 *
 * The inputs of the current corpus sit on a binary lattice (multiples of 0.75 / 0.125 / 0.5), and the
 * intermediate sums stay within the exact integer range of f32, so no rounding ever happens — on any IEEE-754 f32
 * conforming implementation the GPU and the f64 oracle **match bit for bit** (the lattice argument was checked
 * mechanically in research §2). The GEMM / reduction bands of the per-op tolerance table are consumed at a rate of
 * 0 on this corpus, so this check carries regression detection.
 *
 * MUST: do not loosen the tolerance if it breaks. It means "the driver computes below f32 / the generated WGSL fell
 * to a different operation" — if it breaks on a specific driver, take the op out of the check and record why
 * (the same practice as the tanh saturation canary).
 */
test.skipIf(!GPU_AVAILABLE)(
  "GEMM / conv / reduction ops are bit-identical to the f64 oracle on the current corpus (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      const cases = [
        ...MATMUL_CASES,
        ...BMM_CASES,
        ...FUSED_CASES,
        ...DEFORM_CASES,
        ...UPSAMPLE_CASES,
        ...REDUCE_CASES,
      ].filter((testCase) => BIT_IDENTICAL_OPS.has(testCase.op));
      // Blocks the form where a target op picks up no case at all (the form where a list change makes the check spin idle).
      // MUST: a lower bound on the total is not enough (dropping UPSAMPLE / DEFORM entirely still leaves a count) —
      // compare the set of ops itself, so adding to BIT_IDENTICAL_OPS is red until a case is added.
      assertEquals(
        new Set(cases.map((testCase) => testCase.op)),
        new Set(BIT_IDENTICAL_OPS),
        "the target ops of the bit-identity check disagree with the case set",
      );
      for (const testCase of cases) {
        const actual = await runOutputs(gpu, testCase);
        const expected = applyReferenceOpOutputs(
          testCase.op,
          testCase.inputs,
          testCase.attrs ?? {},
          testCase.outShapes[0],
        );
        actual.forEach((tensor, slot) => {
          const got = new Uint32Array(
            tensor.data.buffer,
            tensor.data.byteOffset,
            tensor.data.length,
          );
          const want = new Uint32Array(
            expected[slot].data.buffer,
            expected[slot].data.byteOffset,
            expected[slot].data.length,
          );
          let bad = -1;
          for (let i = 0; i < got.length; i += 1) {
            if (got[i] !== want[i]) {
              bad = i;
              break;
            }
          }
          assertEquals(
            bad,
            -1,
            `${testCase.name}: element #${bad} is 0x${(got[bad] ?? 0).toString(16)} vs 0x${(
              want[bad] ?? 0
            ).toString(16)} (the current corpus should have zero rounding — research §2)`,
          );
        });
      }
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * The **saturation range** of tanh / gelu_tanh (TANH_STABLE_WGSL in src/codegen/elementwise.ts).
 *
 * **These 2 are canaries that go "red only on Metal → green after the fix"**; on drivers that implement tanh
 * IEEE-faithfully, like Vulkan / DXC, they stay green both before and after the cutoff (the cutoff does not change
 * non-NaN values bit for bit). They turn red only on implementations that compute tanh as
 * `(exp(2x)−1)/(exp(2x)+1)` (Metal's fast-math path — measured), where for
 *   - `tanh`: |x| > 44.36
 *   - `gelu_tanh`: pre-activation x ≳ 10.05 (the inner argument √(2/π)·(x + 0.044715x³) exceeds 44.36)
 * exp(2x) overflows f32 and returns `Inf/Inf` = a silent NaN. This is the hole a real model (gemma4 E2B
 * activations measure up to 11.45) was hitting on every token.
 *
 * MUST: the input includes **both the inside and the outside of the saturation band**. Looking only inside the
 * cutoff threshold (9.5) lets "an implementation whose cutoff is not in effect" pass green as well.
 * MUST: check NaN propagation as a pair. Writing the cutoff with the `clamp` idiom turns NaN into the threshold and
 * swallows it into ±1.0 (the mechanism is IS_NAN_BITS_WGSL in src/codegen/elementwise.ts) — one test watches that.
 */
const TANH_SATURATION_INPUTS = [
  0, 5, -5, 9, -9,
  // 9.2 = the band "not cut off (< 9.5) but already 1.0 in f32 (> 9.011)" — detects a regression that lowers the
  // threshold below 9.011 and a shift of the saturation boundary itself. 9.5 is exactly the cutoff boundary.
  9.2,
  -9.2, 9.5, -9.5, 10.6, -10.6, 12, -12, 50, -50, 88, -88, 120, -120,
] as const;

test.skipIf(!GPU_AVAILABLE)(
  "tanh / gelu_tanh match the CPU reference in the saturation range (no NaN emitted — real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const op of ["tanh", "gelu_tanh"] as const) {
        const actual = await runCase(gpu, {
          name: `${op} saturation range`,
          op,
          inputs: [fill([TANH_SATURATION_INPUTS.length], (i) => TANH_SATURATION_INPUTS[i])],
          outShapes: [[TANH_SATURATION_INPUTS.length]],
        });
        const expected = applyReferenceOp(
          op,
          [fill([TANH_SATURATION_INPUTS.length], (i) => TANH_SATURATION_INPUTS[i])],
          {},
          [TANH_SATURATION_INPUTS.length],
        );
        // MUST: catch NaN first. The tolerance check below fails NaN on either side, but unless the message says
        // "which input produced NaN", it is useless for isolating the Metal case.
        TANH_SATURATION_INPUTS.forEach((x, index) => {
          assertEquals(
            Number.isNaN(actual.data[index]),
            false,
            `${op}(${x}) is NaN (an exp-based tanh implementation overflows in the middle)`,
          );
        });
        // Checked with combined absolute + relative tolerance (on the negative saturated side of gelu_tanh the
        // expectation is ±0, which relative tolerance alone cannot judge). WGSL does not guarantee tanh precision, so exact equality is not required.
        TANH_SATURATION_INPUTS.forEach((x, index) => {
          const want = expected.data[index];
          const gap = Math.abs(actual.data[index] - want);
          assertEquals(
            gap <= 1e-6 * Math.max(1, Math.abs(want)),
            true,
            `${op}(${x}): GPU ${actual.data[index]} vs CPU reference ${want}`,
          );
        });
        // MUST: pin the saturation range to **exact values** (the device-side check of the core claim of the cutoff
        // = it does not change non-NaN values bit for bit). The 1e-6 band above also lets 0.99999985, 2.5 ulp lower,
        // through, so "the cutoff does not change the values" is visible only here. tanh is ±1.0, gelu_tanh is
        // x itself on the positive side and **−0.0** on the negative side (the signed zero of 0.5·x·(1 + (−1.0))).
        // NOTE: WGSL allows tanh an absolute error of 1e-5, so a spec-conforming driver could in theory turn this
        // check red — then the bit-identical goldens should break at the same time, and this check is the canary
        // that "learns that fact first in a unit test" (do not loosen it per machine).
        TANH_SATURATION_INPUTS.forEach((x, index) => {
          const v = actual.data[index];
          if (op === "tanh" && Math.abs(x) >= 9.2) {
            assertEquals(
              Object.is(v, Math.sign(x)),
              true,
              `tanh(${x}) = ${v} (the saturation range is not exactly ±1.0)`,
            );
          }
          if (op === "gelu_tanh" && x >= 9) {
            assertEquals(
              Object.is(v, Math.fround(x)),
              true,
              `gelu_tanh(${x}) = ${v} (the positive saturation range is not exactly x)`,
            );
          }
          if (op === "gelu_tanh" && x <= -9) {
            assertEquals(
              Object.is(v, -0),
              true,
              `gelu_tanh(${x}) = ${v} (the negative saturation range is not exactly −0.0)`,
            );
          }
        });
        // MUST: do not become vacuous. Check that outside the saturation range (x = ±5) does not collapse to a constant
        // (an implementation that "returns ±1 for everything" passes the 2 checks above).
        assertNotEquals(actual.data[1], actual.data[3], `${op}: x=5 and x=9 are equal`);
      }

      // ±Inf (a hole the cutoff closed as a side effect — with a plain exp-based implementation tanh(±Inf) = Inf/Inf = NaN).
      // NaN is correct for gelu_tanh(−Inf): the 0·Inf of 0.5·(−Inf)·0.0, and torch gives NaN too
      // (F.gelu(-inf) measured on torch 2.13.0 — both approximate settings). The CPU reference agrees too, all three.
      const infInput = fill([2], (i) => (i === 0 ? Infinity : -Infinity));
      const tanhInf = await runCase(gpu, {
        name: "tanh ±Inf",
        op: "tanh",
        inputs: [infInput],
        outShapes: [[2]],
      });
      assertEquals(Object.is(tanhInf.data[0], 1), true, `tanh(+Inf) = ${tanhInf.data[0]}`);
      assertEquals(Object.is(tanhInf.data[1], -1), true, `tanh(-Inf) = ${tanhInf.data[1]}`);
      const geluInf = await runCase(gpu, {
        name: "gelu_tanh ±Inf",
        op: "gelu_tanh",
        inputs: [infInput],
        outShapes: [[2]],
      });
      assertEquals(geluInf.data[0], Infinity, `gelu_tanh(+Inf) = ${geluInf.data[0]}`);
      assertEquals(
        Number.isNaN(geluInf.data[1]),
        true,
        `gelu_tanh(-Inf) = ${geluInf.data[1]} (should be NaN, as in torch)`,
      );
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * NaN preservation of the softmax family and the empty-row guard of fused attention (v2 — unified nan_max + the
 * safe guard ported. ADR 0044 addendum 2026-08-31).
 *
 * - In the old implementation WGSL's `max` drops NaN at the spec level, so **an all-NaN row turned into the
 *   empty-row case of safe_softmax, became exactly 0, and the NaN silently vanished** (partially NaN rows have
 *   always propagated through the sum — the hole was only all-NaN rows).
 * - Fused attention gave an indeterminate `1/0` on fully masked rows (mask is −inf in every column) — v2's
 *   empty-row guard settles them to the same "output 0 row" as safe_softmax (contract violation → extension to a
 *   valid input).
 */
test.skipIf(!GPU_AVAILABLE)(
  "the softmax family does not flatten all-NaN rows to 0, and fully masked rows of fused attention are exactly 0 (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      // Row 0 = all NaN, row 1 = a normal row (also checks that the NaN row does not contaminate the neighbouring row)
      const nanRows = (i: number): number => (i < 4 ? Number.NaN : SIGNED(i));
      for (const op of ["softmax", "safe_softmax"] as const) {
        const actual = await runCase(gpu, {
          name: `${op} all-NaN row`,
          op,
          inputs: [fill([2, 4], nanRows)],
          outShapes: [[2, 4]],
          attrs: { dim: 1 },
        });
        for (let i = 0; i < 4; i += 1) {
          assertEquals(
            Number.isNaN(actual.data[i]),
            true,
            `${op}: element ${i} of the all-NaN row is ${actual.data[i]} (the NaN vanished)`,
          );
        }
        const tail = [...actual.data.slice(4)];
        assertEquals(
          tail.every(Number.isFinite),
          true,
          `${op}: the normal row was contaminated (${tail})`,
        );
        const sum = tail.reduce((a, b) => a + b, 0);
        assertEquals(Math.abs(sum - 1) < 1e-5, true, `${op}: the sum of the normal row ${sum} ≠ 1`);
      }

      // Fused attention: make row 0 of the mask −inf in every column (a fully masked row). Row 0 of the output is
      // exactly 0 and row 1 stays finite (the sweep covers matching the CPU reference's empty-row branch, so this
      // pins only that the guard fires: "exactly 0, neither indeterminate nor NaN")
      const m = 2;
      const n = 3;
      const d = 4;
      const scale = Math.fround(Math.sqrt(1 / Math.sqrt(d)));
      const masked = await runCase(gpu, {
        name: "attention fully masked row",
        op: "attention",
        inputs: [
          fill([1, 1, m, d], SIGNED),
          fill([1, 1, n, d], POSITIVE),
          fill([1, 1, n, d], SIGNED),
          fill([1, 1, m, n], (i) => (i < n ? -Infinity : 0)),
        ],
        outShapes: [[1, 1, m, d]],
        attrs: { scale },
      });
      for (let i = 0; i < d; i += 1) {
        assertEquals(
          Object.is(masked.data[i], 0),
          true,
          `attention: element ${i} of the fully masked row is ${masked.data[i]} (not exactly 0)`,
        );
      }
      assertEquals(
        [...masked.data.slice(d)].every(Number.isFinite),
        true,
        "attention: the normal row was contaminated",
      );
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * The NaN side of the canary above. `tanh` carries NaN through the outer shell of the bit-pattern check,
 * `gelu_tanh` through the factor `x` left outside the expression (different mechanisms, so each op is checked).
 */
test.skipIf(!GPU_AVAILABLE)(
  "tanh / gelu_tanh propagate NaN (the cutoff does not swallow NaN — real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const op of ["tanh", "gelu_tanh"] as const) {
        // Mixes values inside (0.5) and outside (12) the saturation band with NaN. Without a control, "always NaN" would be green.
        const inputs = [Number.NaN, 0.5, 12, Number.NaN] as const;
        const actual = await runCase(gpu, {
          name: `${op} NaN`,
          op,
          inputs: [fill([4], (i) => inputs[i])],
          outShapes: [[4]],
        });
        assertEquals(
          [...actual.data].map((value) => Number.isNaN(value)),
          [true, false, false, true],
          `${op}: NaN positions`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "matmul matches the CPU reference, tile remainders included (real GPU)",
  () => checkAll(MATMUL_CASES),
);

test.skipIf(!GPU_AVAILABLE)(
  "bmm matches the CPU reference with asymmetric shapes and tile remainders (real GPU)",
  () => checkAll(BMM_CASES),
);

test.skipIf(!GPU_AVAILABLE)(
  "fused attention matches the CPU reference (all axes distinct, remainders and D≠128 included / real GPU)",
  () => checkAll(ATTENTION_CASES),
);

test.skipIf(!GPU_AVAILABLE)(
  "gather (last dimension fixed) matches the CPU reference (real GPU)",
  () => checkAll(GATHER_CASES),
);

/**
 * Pins the ruling for out-of-range indices (src/kernels/gather.ts) on a real GPU.
 *
 * The GPU cannot throw, so it contaminates **only** the out-of-range elements with NaN (it does not quietly return
 * another element). The CPU reference on the oracle side always throws on the same input — if the two agreed,
 * "a graph that violates the contract would pass the comparison", so this pins that they are deliberately asymmetric.
 */
test.skipIf(!GPU_AVAILABLE)(
  "out-of-range gather indices contaminate with NaN on the GPU and throw in the CPU reference (real GPU)",
  async () => {
    const src = fill([2, 4], POSITIVE);
    // Mixes in one index 4 (above the limit) and one -1 (negative); the rest look up valid columns
    const index = fill([2, 3], (i) => [1, 4, 0, 2, -1, 3][i], "i32");
    const gpu = await acquireGpu();
    try {
      const actual = await runCase(gpu, {
        name: "gather oob",
        op: "gather",
        inputs: [src, index],
        outShapes: [[2, 3]],
      });
      const values = [...actual.data];
      assertEquals(
        values.map((v) => Number.isNaN(v)),
        [false, true, false, false, true, false],
      );
      // Indices within the contract look up their own row as usual (only out-of-range elements are contaminated)
      assertEquals(values[0], src.data[1]);
      assertEquals(values[3], src.data[4 + 2]);
    } finally {
      gpu.destroy();
    }
    assertThrows(() => applyReferenceOp("gather", [src, index], {}, [2, 3]));
  },
);

test.skipIf(!GPU_AVAILABLE)("3 row reduces match the CPU reference (real GPU)", () =>
  checkAll(REDUCE_CASES),
);

test.skipIf(!GPU_AVAILABLE)(
  "argmax (last dimension, rank preserved) matches the CPU reference exactly (real GPU)",
  () => checkAll(ARGMAX_CASES),
);

/**
 * Direct check of argmax tie-breaking (pins the 3 MUSTs of ADR 0068 decision 2 with **literal expected values**).
 *
 * MUST: comparison with the CPU reference (checkAll) cannot replace it. These axes (direction of tie-breaking,
 * handling of NaN, the answer for all −inf rows) go green if both sides are wrong in the same direction, so the
 * values measured on torch are placed as literals (the meaning of the table and its rows is `ARGMAX_TIEBREAK_INPUT`
 * in `helpers/gpu_op_cases.ts`). This looks only at **the GPU output** — the direction of the CPU reference on the
 * same table is bound by the check in `reference_ops_test.ts`, which also runs without an adapter.
 */
test.skipIf(!GPU_AVAILABLE)(
  "argmax tie-breaking / NaN / all −inf rows match the values measured on torch (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      const actual = await runCase(gpu, {
        name: "argmax tie-break",
        op: "argmax",
        inputs: [ARGMAX_TIEBREAK_INPUT],
        outShapes: [[ARGMAX_TIEBREAK_TORCH.length, 1]],
        outDtypes: ["i32"],
      });
      assertEquals(
        [...actual.data],
        [...ARGMAX_TIEBREAK_TORCH],
        "the GPU argmax differs from the values measured on torch",
      );
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * Pins the fixed behaviour of two-phase argmax (row length ≥ 16,384 — the "two-phase split" of src/kernels/argmax.ts)
 * with **literal expected values**. The comparison with the CPU reference (the two-phase cases of ARGMAX_CASES)
 * checks that "folding the maximum of each segment gives the maximum of the whole range", but tie-breaking
 * direction, NaN and all −inf rows are axes that go green if both sides are wrong in the same direction, so the same
 * values as the one-dispatch check (above) are placed as literals.
 *
 * Meaning of the rows (row length 20,000 = 5 segments of 4,096):
 * 0. Equal maxima in segment 1 (index 5,000) and segment 2 (index 9,000) → the smallest index 5,000
 * 1. A finite maximum in segment 0 (index 10, value 99) and a NaN in segment 3 (index 13,000) → **NaN is the maximum** 13,000
 * 2. All elements −inf → 0 (every segment's sentinel loses in the merge, and the smallest index of segment 0 remains)
 * 3. The maximum at the end of the last segment (index 19,999) → 19,999 (the scan of the remainder segment reaches the end)
 */
test.skipIf(!GPU_AVAILABLE)(
  "segment boundaries / NaN / all −inf rows of two-phase argmax follow the same rules as the one-dispatch form (real GPU)",
  async () => {
    const dim = 20000;
    const rows = 4;
    const input = fill([rows, dim], (index) => {
      const row = Math.floor(index / dim);
      const at = index % dim;
      switch (row) {
        case 0:
          return at === 5000 || at === 9000 ? 7 : -1;
        case 1:
          return at === 10 ? 99 : at === 13000 ? Number.NaN : -1;
        case 2:
          return Number.NEGATIVE_INFINITY;
        default:
          return at === 19999 ? 3 : -1;
      }
    });
    const gpu = await acquireGpu();
    try {
      const actual = await runCase(gpu, {
        name: "argmax split fixed",
        op: "argmax",
        inputs: [input],
        outShapes: [[rows, 1]],
        outDtypes: ["i32"],
      });
      assertEquals([...actual.data], [5000, 13000, 0, 19999], "the rules of two-phase argmax");
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "topk (last dimension, static k, 2 outputs: values + indices) matches the CPU reference (real GPU)",
  () => checkAll(TOPK_CASES),
);

/** A display form with NaN flattened to "NaN" (neither allclose nor assertEquals treats NaN as equal to NaN). */
const nanSafe = (values: Iterable<number>): readonly (number | "NaN")[] =>
  [...values].map((value) => (Number.isNaN(value) ? "NaN" : value));

/**
 * Equal-value ties / NaN / all −inf rows of topk (pins the fixed behaviour of ADR 0068 decision 3 with **literal
 * expected values**).
 *
 * MUST: comparison with the CPU reference (checkAll) cannot replace it. These axes (direction of tie-breaking,
 * handling of NaN, the answer for all −inf rows) go green if both sides are wrong in the same direction, so literals are used.
 *
 * **Relation to torch (measured 2026-08-17 / torch 2.13.0+cpu)**:
 *
 * - **The value column is bit-identical to torch** (`VALUES` are the values measured on torch). The multiplicity
 *   of equal values in descending order is the same too, so even rows full of duplicates match (comparing the values
 *   column of `torch.topk` against a (value descending, index ascending) implementation over 200×4 cases also
 *   matched everywhere).
 * - **The index column does not match torch** (`INDICES` is karume's rule). torch's topk does not guarantee the
 *   order of equal elements, and for `[5,5,5,5]` with k=1 it returns index **2** while `argmax` returns **0**
 *   (self-contradictory within the same repository). karume defines **smallest index first**, consistent with argmax.
 *   In the table below the rows that disagree with torch are 1 / 2 / 4 / 5 / 6 (torch gives, in order, [2,3,0,1] /
 *   [2,3,0,1] / [2,3,0,1] / [1,3,2,0] / [0,2,3,1]).
 *
 * Meaning of the rows:
 * 0. Two equal values (3.0 at index 1,2) → the smallest index first (torch agrees)
 * 1. Every element equal → index ascending (a "last maximum" implementation would give descending)
 * 2. All elements −inf → index ascending (with a finite-sentinel identity the sentinel index leaks)
 * 3. Two NaN (index 1,3) → **NaN is the maximum**; among NaN the smallest index (torch agrees)
 * 4. All elements NaN → index ascending
 * 5. −inf mixed with equal maxima → 2.0 first; −inf in index ascending
 * 6. Two groups of equal values → the 2.0 group first, index ascending within it
 */
test.skipIf(!GPU_AVAILABLE)(
  "equal-value ties / NaN / all −inf rows of topk follow the fixed behaviour (values bit-identical to torch — real GPU)",
  async () => {
    const NEG = Number.NEGATIVE_INFINITY;
    const NAN = Number.NaN;
    const rows = [
      [1, 3, 3, 2],
      [5, 5, 5, 5],
      [NEG, NEG, NEG, NEG],
      [1, NAN, 3, NAN],
      [NAN, NAN, NAN, NAN],
      [NEG, 2, NEG, 2],
      [2, 1, 2, 1],
    ];
    // The values column measured on torch (k = 4 = the last dimension = the whole row in descending order)
    const VALUES = [
      [3, 3, 2, 1],
      [5, 5, 5, 5],
      [NEG, NEG, NEG, NEG],
      [NAN, NAN, 3, 1],
      [NAN, NAN, NAN, NAN],
      [2, 2, NEG, NEG],
      [2, 2, 1, 1],
    ];
    // karume's rule (smallest index first — torch does not guarantee the order of equal values)
    const INDICES = [
      [1, 2, 3, 0],
      [0, 1, 2, 3],
      [0, 1, 2, 3],
      [1, 3, 2, 0],
      [0, 1, 2, 3],
      [1, 3, 0, 2],
      [0, 2, 1, 3],
    ];
    const input = fill([rows.length, 4], (i) => rows[Math.floor(i / 4)][i % 4]);
    const testCase: OpCase = {
      name: "topk tie-break",
      op: "topk",
      inputs: [input],
      outShapes: [
        [rows.length, 4],
        [rows.length, 4],
      ],
      outDtypes: ["f32", "i32"],
      attrs: { k: 4 },
    };
    const gpu = await acquireGpu();
    try {
      const actual = await runOutputs(gpu, testCase);
      assertEquals(nanSafe(actual[0].data), nanSafe(VALUES.flat()), "GPU topk values");
      assertEquals([...actual[1].data], INDICES.flat(), "GPU topk indices");
    } finally {
      gpu.destroy();
    }
    // Applies the same table to the CPU reference too, pinning the direction of the oracle side at the same time
    const expected = applyReferenceOpOutputs("topk", [input], { k: 4 });
    assertEquals(nanSafe(expected[0].data), nanSafe(VALUES.flat()), "CPU reference topk values");
    assertEquals([...expected[1].data], INDICES.flat(), "CPU reference topk indices");
  },
);

/**
 * topk with k=1 matches argmax (detects disagreement between the families).
 *
 * MUST: run both ops on the same input and compare them. Their kernels and keys differ, so rewriting the
 * predicate body of only one keeps shape and dtype matching while only the answers split.
 * The input contains **ties, NaN and all −inf rows** (with an input that has only unique maxima, either
 * tie-breaking discipline gives the same answer and it is no detector).
 */
test.skipIf(!GPU_AVAILABLE)(
  "the topk index with k=1 matches argmax (ties / NaN / all −inf rows included; real GPU)",
  async () => {
    const NEG = Number.NEGATIVE_INFINITY;
    const NAN = Number.NaN;
    const rows = [
      [1, 3, 3, 2],
      [5, 5, 5, 5],
      [NEG, NEG, NEG, NEG],
      [1, NAN, 3, NAN],
      [NEG, 2, NEG, 2],
    ];
    const input = fill([rows.length, 4], (i) => rows[Math.floor(i / 4)][i % 4]);
    const gpu = await acquireGpu();
    try {
      const topk = await runOutputs(gpu, {
        name: "topk k=1",
        op: "topk",
        inputs: [input],
        outShapes: [
          [rows.length, 1],
          [rows.length, 1],
        ],
        outDtypes: ["f32", "i32"],
        attrs: { k: 1 },
      });
      const argmax = await runCase(gpu, {
        name: "argmax",
        op: "argmax",
        inputs: [input],
        outShapes: [[rows.length, 1]],
        outDtypes: ["i32"],
      });
      assertEquals([...topk[1].data], [...argmax.data], "topk with k=1 disagrees with argmax");
      // The value side returns the same element as amax too (rules out matching indices with values from another element)
      assertEquals(
        nanSafe(topk[0].data),
        nanSafe([...argmax.data].map((at, row) => rows[row][at])),
        "the topk value with k=1 differs from the element its index points to",
      );
    } finally {
      gpu.destroy();
    }
  },
);

/** The **bit pattern** of an f32 tensor (±0.0 are numerically equal, so the sign bit is visible only here). */
const f32Bits = (tensor: Tensor | RefTensor): readonly number[] => {
  if (tensor.dtype !== "f32") throw new Error(`not an f32 dtype: ${tensor.dtype}`);
  return [...new Uint32Array(tensor.data.slice().buffer)];
};

/**
 * Tie rows mixing `+0.0` and `-0.0` (**the bits of the element at the chosen index come out as they are**).
 *
 * MUST: the expected values are **karume's rule** (smallest index first within an equivalence class, the value is
 * the chosen element's bits themselves), not a torch comparison. `±0.0` compare equal, so writing either sign keeps
 * the numeric comparison green (both allclose and assertEquals have `-0 === +0`) — the sign bit is visible only in
 * **the Uint32 bit pattern**. If this loosens, an implementation that folds the comparison into the `max` idiom
 * (the driver's `max(-0.0, +0.0)` may return `+0.0`), or one that normalizes the value instead of looking it up again
 * from the index, passes silently.
 *
 * NOTE: differences in NaN payload (the same NaN with different mantissa bits) are out of scope — WGSL / drivers
 * do not guarantee payload preservation, so it cannot be pinned as a rule.
 *
 * Meaning of the rows:
 * 0. A tie of all-±0.0 elements → index ascending, each value keeps its index's sign
 * 1. One finite value + a ±0.0 tie → 1.0 first, the zeros in index ascending order
 * 2. A ±0.0 tie + a negative value → the 3 zeros first (index ascending), −1.0 last
 */
test.skipIf(!GPU_AVAILABLE)(
  "topk writes the bits of the selected index even on ±0.0 ties (values pinned as bit patterns; real GPU)",
  async () => {
    const rows = [
      [-0, 0, -0, 0],
      [0, -0, 1, -0],
      [-0, 0, -1, -0],
    ];
    // Bit patterns of the values (0x80000000 = −0.0 / 0x00000000 = +0.0 / 0x3f800000 = 1.0 / 0xbf800000 = −1.0)
    const VALUE_BITS = [
      [0x80000000, 0x00000000, 0x80000000, 0x00000000],
      [0x3f800000, 0x00000000, 0x80000000, 0x80000000],
      [0x80000000, 0x00000000, 0x80000000, 0xbf800000],
    ];
    const INDICES = [
      [0, 1, 2, 3],
      [2, 0, 1, 3],
      [0, 1, 3, 2],
    ];
    const input = fill([rows.length, 4], (i) => rows[Math.floor(i / 4)][i % 4]);
    const testCase: OpCase = {
      name: "topk signed-zero tie",
      op: "topk",
      inputs: [input],
      outShapes: [
        [rows.length, 4],
        [rows.length, 4],
      ],
      outDtypes: ["f32", "i32"],
      attrs: { k: 4 },
    };
    const gpu = await acquireGpu();
    try {
      const actual = await runOutputs(gpu, testCase);
      assertEquals(f32Bits(actual[0]), VALUE_BITS.flat(), "GPU topk value bits");
      assertEquals([...actual[1].data], INDICES.flat(), "GPU topk indices");
    } finally {
      gpu.destroy();
    }
    // Applies the same table to the CPU reference too, pinning the sign on the oracle side at the same time
    const expected = applyReferenceOpOutputs("topk", [input], { k: 4 });
    assertEquals(f32Bits(expected[0]), VALUE_BITS.flat(), "CPU reference topk value bits");
    assertEquals([...expected[1].data], INDICES.flat(), "CPU reference topk indices");
  },
);

/** Builds the ranking from the CPU's ordinary numeric comparison. Does not duplicate the GPU's integer order transform. */
const rank = (data: Float32Array<ArrayBuffer>, base: number, dim: number): number[] =>
  Array.from({ length: dim }, (_, i) => i).sort((a, b) => {
    const va = data[base + a],
      vb = data[base + b];
    if (Number.isNaN(va)) return Number.isNaN(vb) ? a - b : -1;
    if (Number.isNaN(vb)) return 1;
    return va > vb ? -1 : va < vb ? 1 : a - b;
  });

test.skipIf(!GPU_AVAILABLE)(
  "argmax/topk: distinguish subnormals from zero and return the same values and indices as the CPU on short rows and split boundaries",
  async function () {
    const cases = [
      [0, 1, 2, 0x80000001],
      [0x80000002, 0x80000001, 0x80000003, 0xff800000],
      [0x80000000, 0, 1, 0x007fffff],
      [0x80000001, 0, 0x80000000, 0xff800000],
      [0, 0x80000000, 0x00800000, 0x007fffff],
      [0xff800000, 0x807fffff, 0x80800000, 0xff7fffff],
      [0, 1, 1, 0x80000000],
      [0xffa00001, 0x7fc00002, 0x7f800000, 0xffc00003],
      [0x7f800000, 0x7f7fffff, 0x7f800000, 0],
      [0x80000000, 0, 0x80000000, 0],
    ];
    const gpu = await acquireGpu();
    try {
      for (const dim of [4, 257, 16383, 16384, 16385, 262144]) {
        const rows = cases.length,
          data = new Float32Array(rows * dim).fill(-Infinity);
        const bits = new Uint32Array(data.buffer);
        // Checks smallest-index ties across lane and partial-segment boundaries.
        const positions = [0, Math.floor(dim / 3), Math.floor((dim * 2) / 3), dim - 1];
        for (let r = 0; r < rows; r++) {
          for (let i = 0; i < positions.length; i++) bits[r * dim + positions[i]] = cases[r][i];
        }
        const expected = cases.map((_, r) => rank(data, r * dim, dim)[0]);
        for (const op of ["argmax", "topk"] as const) {
          const graph = singleOpDeclaration(
            op,
            [[rows, dim]],
            op === "topk"
              ? [
                  [rows, 1],
                  [rows, 1],
                ]
              : [[rows, 1]],
            {
              outDtypes: op === "topk" ? ["f32", "i32"] : ["i32"],
              attrs: op === "topk" ? { k: 1 } : {},
            },
          );
          const session = await createSessionFromContainer(
            gpu,
            await openModelBytes(graph, []),
            GRAPH_NAME,
          );
          try {
            for (let repeat = 0; repeat < 2; repeat++) {
              const out = await session.run({ x0: { dtype: "f32", shape: [rows, dim], data } });
              const indices = op === "topk" ? out.y1 : out.y;
              assert(indices.dtype === "i32");
              assertEquals([...indices.data], expected, `${op}, dim=${dim}, repeat=${repeat}`);
              if (op === "topk") {
                assert(out.y.dtype === "f32");
                assertEquals(
                  [...new Uint32Array(out.y.data.buffer, out.y.data.byteOffset, out.y.data.length)],
                  expected.map((i, r) => bits[r * dim + i]),
                  "keeps the bit pattern of the selected value",
                );
              }
            }
          } finally {
            await session.dispose();
          }
        }
      }
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "topk: returns multiple candidates of subnormals, NaN and signed zeros in the same order and value bits as the CPU",
  async function () {
    const edgeBits = [
      0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff, 0, 0x00800000, 0x80800000, 0x7f800000,
      0xff800000, 0x7fa12345, 0xffc05678, 0x7f7fffff, 0xff7fffff, 2, 0x80000002,
    ];
    const gpu = await acquireGpu();
    try {
      for (const [dim, k] of [
        [16, 16],
        [257, 4],
        [257, 63],
        [16385, 4],
      ]) {
        const rows = 4,
          data = new Float32Array(rows * dim),
          bits = new Uint32Array(data.buffer);
        // Fixed-seed LCG: the bit patterns are identical on every run, so failures reproduce.
        let state = 123456789;
        for (let i = 0; i < bits.length; i++) {
          state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
          const row = Math.floor(i / dim);
          // Rows 0/1 are all positive / negative subnormals, row 2 boundaries and duplicates, row 3 the full bit distribution.
          bits[i] =
            row === 0
              ? state & 0x7fffff
              : row === 1
                ? (state & 0x7fffff) | 0x80000000
                : row === 2
                  ? edgeBits[i % edgeBits.length]
                  : state;
        }
        const expected = Array.from({ length: rows }, (_, r) =>
          rank(data, r * dim, dim).slice(0, k),
        );
        const session = await createSessionFromContainer(
          gpu,
          await openModelBytes(
            singleOpDeclaration(
              "topk",
              [[rows, dim]],
              [
                [rows, k],
                [rows, k],
              ],
              { outDtypes: ["f32", "i32"], attrs: { k } },
            ),
            [],
          ),
          GRAPH_NAME,
        );
        try {
          const out = await session.run({ x0: { dtype: "f32", shape: [rows, dim], data } });
          assert(out.y.dtype === "f32" && out.y1.dtype === "i32");
          assertEquals([...out.y1.data], expected.flat(), `dim=${dim}, k=${k}`);
          assertEquals(
            [...new Uint32Array(out.y.data.buffer, out.y.data.byteOffset, out.y.data.length)],
            expected.flatMap((row, r) => row.map((i) => bits[r * dim + i])),
            "keeps the value bits of every candidate",
          );
        } finally {
          await session.dispose();
        }
      }
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * The **implementation limit of k** for topk (ADR 0068 decision 3 — no degradation; fail loudly with the limit).
 *
 * MUST: check it on **a narrowed device** (`LIMIT_CAPS`). The local adapter offers 49152 bytes of workgroup storage,
 * so the k limit is 191, and there is no other way to confirm by running the k that fails on the default machine
 * (16384 → limit 63) (the same discipline as "enumeration is no evidence that it works").
 * Also check as a pair that inside the limit (k=2) is **green on the same narrowed device** — an implementation that
 * fails everything would still pass the rejection-side check alone.
 */
test.skipIf(!GPU_AVAILABLE)(
  "topk fails with the limit when k exceeds the limit derived from workgroup storage (narrowed device / real GPU)",
  async () => {
    // 8·W·(k+1) ≤ 1024 → k limit = 3
    const gpu = await acquireGpu({ [LIMIT_CAPS]: { maxComputeWorkgroupStorageSize: 1024 } });
    try {
      assertEquals(gpu.limits.maxComputeWorkgroupStorageSize, 1024);
      const input = fill([2, 8], SIGNED);
      const outputs = await runOutputs(gpu, {
        name: "topk k=2 (inside the limit)",
        op: "topk",
        inputs: [input],
        outShapes: [
          [2, 2],
          [2, 2],
        ],
        outDtypes: ["f32", "i32"],
        attrs: { k: 2 },
      });
      const expected = applyReferenceOpOutputs("topk", [input], { k: 2 });
      assertEquals([...outputs[0].data], [...expected[0].data]);
      assertEquals([...outputs[1].data], [...expected[1].data]);
      // Exceeding the limit fails rather than degrading (the diagnostics show the limit value 3)
      const graph = singleOpDeclaration(
        "topk",
        [[2, 8]],
        [
          [2, 4],
          [2, 4],
        ],
        {
          outDtypes: ["f32", "i32"],
          attrs: { k: 4 },
        },
      );
      const session = await createSessionFromContainer(
        gpu,
        await openModelBytes(graph, []),
        GRAPH_NAME,
      );
      try {
        await assertRejects(
          () => session.run({ x0: input }),
          CodegenError,
          "exceeds the implementation limit 3",
        );
      } finally {
        await session.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * Fault injection (ADR 0068 acceptance criterion ③) — proof that the argmax / topk checks **actually are detectors**.
 *
 * MUST: it cannot be fired through a Session ({@link runOutputs}). `PipelineCache` caches by **key**, so there is no
 * inlet to feed mutated WGSL (even with one, the normal pipeline would be handed out). This uses the direct dispatch
 * of `helpers/rank-dispatch.ts` with a cache keyed by **the final WGSL string** (the same style as
 * `helpers/state-dispatch.ts` for the states form).
 *
 * MUST: prepare inputs of **both families**. As the P2E measurement (`.claude/reviews/2026-08-19_b04f589/findings/
 * P2E-topk-merge-fuzz.md` §3, 18,000 cases) shows, the input families that can detect topk mutations are
 * complementary, and with only one of them a mutation is **always green**:
 *
 * - ③ (the tail comparison of the lane-local insertion) shows only with **single-lane inputs** (the top k all
 *   `i ≡ l (mod W)`) — unless the top k land on one lane, the local list does not overflow and the tail comparison
 *   never acts.
 * - ① ② (cursor advance, tie branch) show only with **spread inputs** (equal values scattered over lanes) — with an
 *   input where a single lane wins every round, the cursor-advance branch becomes harmless and no tie occurs.
 *
 * This test reproduced that complementarity (measured 2026-08-19 / RTX 3080 Ti): applying ① to the single-lane input
 * and ③ to the spread input are both green (= not detected).
 *
 * NOTE: 2 mutations were considered but **not included** (to avoid vacuous tests; same measurement as above):
 *
 * - **removing the final `workgroupBarrier()` of topk**: 0 mismatches in 20 trials at rows=256, dim=320, k=16.
 *   W=32 is exactly NVIDIA's warp width and the workgroup executes in lockstep, so the read of `won` and the
 *   overwrite of `head` in the next round do not race. P2E §4-1's "undetectable in principle by sequential
 *   simulation" was measured to hold **on a real GPU as well** on this machine.
 * - **argmax sentinel `dim` → `0`**: the sentinel affects the answer only when the row maximum is −inf (= every
 *   element is −inf), and the correct index of that row is 0, so **the mutant returns the same 0**. It cannot show
 *   in the values in principle. Instead, a mutation of **the sentinel value** (`neg_inf` → the rank key of positive
 *   zero) that fires the same MUST (no finite sentinel — src/kernels/argmax.ts) is included. That one leaks the
 *   sentinel index `dim` into the output on all-negative rows.
 */
type RankInjection = {
  readonly label: string;
  readonly mutate: RankMutation;
};

/** Rewrites the relevant part of the WGSL (a miss = the replacement target vanished because of a wording change is caught by `assertMutated`). */
const injectRank =
  (from: string, to: string): RankMutation =>
  (wgsl) => {
    const mutated = wgsl.replaceAll(from, to);
    assertMutated(wgsl, mutated, from);
    return mutated;
  };

/** One topk fault injection (including which input family it is fired with — the complementarity above). */
type TopkInjection = RankInjection & { readonly input: TopkFaultInput };

type TopkFaultInput = {
  readonly rows: number;
  readonly dim: number;
  readonly k: number;
  readonly data: Float32Array<ArrayBuffer>;
};

/**
 * Spread input (3 rounds of `W = 32`). Row 0 = every element equal, row 1 = equal blocks of 8 elements crossing
 * lane boundaries. Both "take k from the set of equal values in smallest-index order", so advancing the cursor even
 * one lane too far makes the answer jump.
 */
const TOPK_SPREAD_INPUT: TopkFaultInput = {
  rows: 2,
  dim: 96,
  k: 8,
  data: Float32Array.from(
    Array.from({ length: 2 * 96 }, (_, i) => (i < 96 ? 7.5 : -Math.floor((i - 96) / 8))),
  ),
};

/**
 * Single-lane input (the top 5 all `i ≡ 7 (mod 32)` = handled by lane 7). k=3, so lane 7's local list overflows twice
 * with 5 candidates, taking the path where the tail comparison actually discards candidates.
 */
const TOPK_SINGLE_LANE_INPUT: TopkFaultInput = {
  rows: 1,
  dim: 160,
  k: 3,
  data: Float32Array.from(
    Array.from({ length: 160 }, (_, i) => (i % 32 === 7 ? 9 - (i - 7) / 32 : -1)),
  ),
};

const TOPK_INJECTIONS: readonly TopkInjection[] = [
  // ① Make the cursor, which only the owner of the winning element advances, **advance on every lane** (the core of the merge)
  {
    label: "make the cursor-advance condition always true",
    input: TOPK_SPREAD_INPUT,
    mutate: injectRank(`won % ${TOPK_WORKGROUP_SIZE}u == lid`, "true"),
  },
  // ② Invert the tie branch of the winner predicate (smallest-index rule → largest index)
  {
    label: "invert the tie branch to ib > ia",
    input: TOPK_SPREAD_INPUT,
    mutate: injectRank("(kb == ka && ib < ia)", "(kb == ka && ib > ia)"),
  },
  // ③ Shift the tail (weakest) comparison of the lane-local insertion one place earlier (the k-th no longer gets in)
  {
    label: "shift the tail comparison of the lane-local insertion by 1",
    input: TOPK_SINGLE_LANE_INPUT,
    mutate: injectRank(
      `block + ${TOPK_SINGLE_LANE_INPUT.k - 1}u`,
      `block + ${TOPK_SINGLE_LANE_INPUT.k - 2}u`,
    ),
  },
];

/**
 * Rows for argmax fault injection. 0 = NaN mixed in (the detection line of the NaN branch) / 1 = all negative (the
 * detection line of the sentinel value — a finite sentinel leaks the sentinel index `dim`) / 2 = all −inf (the row
 * where the rank of the identity matters).
 * `dim = 8 < 256`, so **some lanes hold no element** = the sentinel enters the tree as is.
 */
const ARGMAX_FAULT_INPUT = {
  rows: 3,
  dim: 8,
  data: Float32Array.from([
    1,
    Number.NaN,
    3,
    Number.NaN,
    2,
    0,
    -1,
    5,
    -5,
    -2,
    -9,
    -2,
    -7,
    -3,
    -4,
    -8,
    ...Array.from({ length: 8 }, () => Number.NEGATIVE_INFINITY),
  ]),
};

const ARGMAX_INJECTIONS: readonly RankInjection[] = [
  // ① Make the identity **value** a finite sentinel (the sentinel index leaks outside [0, dim))
  {
    label: "make the identity of the row max a finite sentinel",
    mutate: injectRank("var best = neg_inf;", "var best = 0x80000000u;"),
  },
  // ② Invert the priority of NaN versus non-NaN (the detection line for NaN losing to a finite value).
  {
    label: "invert the NaN priority of the max comparison",
    mutate: injectRank(
      "select(ordered, 0xffffffffu, magnitude > 0x7f800000u)",
      "select(ordered, 0u, magnitude > 0x7f800000u)",
    ),
  },
];

/** Folds the 2 outputs of topk into one comparison target (rules out looking at only one). */
const topkPair = (result: {
  readonly values: Float32Array;
  readonly indices: Int32Array;
}): readonly (readonly number[])[] => [[...result.values], [...result.indices]];

const referenceTopkPair = (input: TopkFaultInput): readonly (readonly number[])[] => {
  const expected = applyReferenceOpOutputs(
    "topk",
    [refTensor([input.rows, input.dim], input.data)],
    { k: input.k },
  );
  return [[...expected[0].data], [...expected[1].data]];
};

test.skipIf(!GPU_AVAILABLE)(
  "fault injection: every argmax / topk mutant turns red in the reference comparison (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const cache: RankPipelineCache = new Map();
    try {
      // Precondition: the normal version matches the expected values (if this is red, "the mutant mismatches" is evidence of nothing)
      for (const input of [TOPK_SPREAD_INPUT, TOPK_SINGLE_LANE_INPUT]) {
        const actual = await runTopkDirect(gpu.device, input, input.data, { cache });
        assertEquals(
          topkPair(actual),
          referenceTopkPair(input),
          `the normal topk (dim=${input.dim} k=${input.k}) differs from the CPU reference`,
        );
      }
      const argmaxExpected = [
        ...applyReferenceOp(
          "argmax",
          [refTensor([ARGMAX_FAULT_INPUT.rows, ARGMAX_FAULT_INPUT.dim], ARGMAX_FAULT_INPUT.data)],
          {},
          [ARGMAX_FAULT_INPUT.rows, 1],
        ).data,
      ];
      assertEquals(
        [
          ...(await runArgmaxDirect(gpu.device, ARGMAX_FAULT_INPUT, ARGMAX_FAULT_INPUT.data, {
            cache,
          })),
        ],
        argmaxExpected,
        "the normal argmax differs from the CPU reference",
      );
      for (const injection of TOPK_INJECTIONS) {
        const input = injection.input;
        const actual = await runTopkDirect(gpu.device, input, input.data, {
          cache,
          mutate: injection.mutate,
        });
        assertNotEquals(
          topkPair(actual),
          referenceTopkPair(input),
          `topk fault injection '${injection.label}' was not detected`,
        );
      }
      for (const injection of ARGMAX_INJECTIONS) {
        const actual = await runArgmaxDirect(
          gpu.device,
          ARGMAX_FAULT_INPUT,
          ARGMAX_FAULT_INPUT.data,
          { cache, mutate: injection.mutate },
        );
        assertNotEquals(
          [...actual],
          argmaxExpected,
          `argmax fault injection '${injection.label}' was not detected`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "3 layout ops (aliasing reshape / strided permute and expand) match the CPU reference (real GPU)",
  () => checkAll(LAYOUT_CASES),
);

test.skipIf(!GPU_AVAILABLE)(
  "the second group of layout ops (slice / cat / pad / flip) matches the CPU reference (real GPU)",
  () => checkAll(LAYOUT2_CASES),
);

test.skipIf(!GPU_AVAILABLE)(
  "10 fused ops (linear / layer_norm / rms_norm / softmax / safe_softmax / embedding / masked_fill / conv1d / conv2d / conv_transpose1d) match the CPU reference (real GPU)",
  () => checkAll(FUSED_CASES),
);

test.skipIf(!GPU_AVAILABLE)(
  "group_norm (samples, groups, two chunks, rank 2) matches the CPU reference (real GPU)",
  () => checkAll(GROUP_NORM_CASES),
);

test.skipIf(!GPU_AVAILABLE)(
  "bilinear resample (upscale / downscale / scale 0) matches the CPU reference (real GPU)",
  () => checkAll(UPSAMPLE_CASES),
);

/**
 * Comparison for inexact scales (shapes where `fl((in−1)/(out−1))` does not round-trip in f32). It is **kept
 * separate from the bit-identity check** because by definition this shape does not satisfy bit identity (the MUST
 * of UPSAMPLE_FRACTIONAL_CASES in helpers/gpu_op_cases.ts). What it checks is agreement within the band of the
 * measured table, and it is the only case that makes both the GPU and the CPU reference walk a rounding sequence
 * where λ does not collapse to 0 / 1.
 */
test.skipIf(!GPU_AVAILABLE)(
  "bilinear resample matches the CPU reference within the measured band even when the scale is not exact in f32 (real GPU)",
  () => checkAll(UPSAMPLE_FRACTIONAL_CASES),
);

test.skipIf(!GPU_AVAILABLE)(
  "DCNv2 (2 shapes of k / batch 2 / asymmetric shapes) matches the CPU reference (real GPU)",
  () => checkAll(DEFORM_CASES),
);

test.skipIf(!GPU_AVAILABLE)(
  "the GRU hidden-side scan (2 directions × batch 1 / 3) matches the CPU reference (real GPU)",
  () => checkAll(GRU_SCAN_CASES),
);

/**
 * Pins **the definition itself** of `align_corners = True` on a real GPU — the first / last output row and the first
 * / last output column **match exactly** the corresponding edges of the input (λ collapses to 0 or 1, so no rounding
 * enters).
 *
 * MUST: check exact equality, not allclose. Mixing it up with the coordinate formula of `align_corners = False`
 * shifts the edges by half a pixel, but the shift can be smaller than the tolerance, and in checkAll it **stays green
 * if both sides share the same error** (both the CPU reference and the GPU are under test here). This compares
 * against the edge values of the input tensor itself, without going through the reference implementation.
 */
test.skipIf(!GPU_AVAILABLE)(
  "bilinear resample makes the output edges match the input edges exactly where (in−1)/(out−1) round-trips in f32 (align_corners — real GPU)",
  async () => {
    // H≠W / Hout≠Wout (mixing up axes shows in the edge values). All values are distinct.
    const [heightIn, widthIn] = [4, 5];
    const [heightOut, widthOut] = [9, 7];
    const input = fill([1, 1, heightIn, widthIn], (i) => 1 + i * 0.5);
    const gpu = await acquireGpu();
    try {
      const actual = await runCase(gpu, {
        name: "upsample_bilinear2d edges",
        op: "upsample_bilinear2d",
        inputs: [input],
        outShapes: [[1, 1, heightOut, widthOut]],
        attrs: { output_size: [heightOut, widthOut] },
      });
      const at = (row: number, column: number): number => actual.data[row * widthOut + column];
      const src = (row: number, column: number): number => input.data[row * widthIn + column];
      assertEquals(at(0, 0), src(0, 0), "top left");
      assertEquals(at(0, widthOut - 1), src(0, widthIn - 1), "top right");
      assertEquals(at(heightOut - 1, 0), src(heightIn - 1, 0), "bottom left");
      assertEquals(at(heightOut - 1, widthOut - 1), src(heightIn - 1, widthIn - 1), "bottom right");
      // The interior is interpolated (checking only the edges would let an identity copy pass). scale_h = 3/8, so
      // the source coordinate of output row 4 is 1.5 = the midpoint of input rows 1 and 2 — an interior point consistent with the exact edges.
      assertEquals(at(4, 0), Math.fround((src(1, 0) + src(2, 0)) / 2), "midpoint along H");
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * **The other side** of the check above (by-design inexact endpoints). For shapes where `scale = fl((in−1)/(out−1))`
 * does not round-trip in f32, the source coordinate of the last output stays at `fl(fl(1/41)·41) = 0.9999999403953552`,
 * and **the output edge does not exactly match the input edge** (a by-design contract published in
 * docs/limitations.md and the NOTE of src/kernels/upsample-bilinear2d.ts — the price of walking the same formula in
 * the same order as torch).
 *
 * MUST: shape it so that a change that "clamps the edges to make them exact" always turns red. Every other case has
 * an exact scale in f32, so adding a clamp moves no value by a single bit and stays green.
 */
test.skipIf(!GPU_AVAILABLE)(
  "bilinear resample does not make the edges exact where the scale does not round-trip in f32 (no clamping; real GPU)",
  async () => {
    const rows = 42;
    // The 2 edge values are 0 / 1 (the value range where the difference of λ = 1 − 2⁻²⁴ is representable on the f32
    // lattice — with large values the interpolation rounds back to the edge value and the inexactness cannot be observed).
    const input = fill([1, 1, 2, 1], (i) => i);
    const attrs = { output_size: [rows, 1] };
    const gpu = await acquireGpu();
    try {
      const actual = await runCase(gpu, {
        name: "upsample_bilinear2d inexact endpoint",
        op: "upsample_bilinear2d",
        inputs: [input],
        outShapes: [[1, 1, rows, 1]],
        attrs,
      });
      const last = actual.data[rows - 1];
      assertNotEquals(last, input.data[1], "the last row is clamped to the input edge");
      // λ = fl(fl(1/41)·41) (exactly the amount by which the source coordinate falls short of 1)
      assertEquals(last, Math.fround(Math.fround(1 / 41) * 41), "λ of the last row");
      // The CPU reference produces the same inexact endpoint too (a regression that clamps only one side fails too)
      const expected = applyReferenceOp("upsample_bilinear2d", [input], attrs, [1, 1, rows, 1]);
      const report = compareTensors(actual, expected, opTolerance("upsample_bilinear2d"));
      assertEquals(report.pass, true, `inexact endpoint: ${formatAllclose(report)}`);
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * Pins the ruling for out-of-range embedding indices (src/kernels/embedding.ts) on a real GPU. The same asymmetry as
 * gather — the GPU contaminates **only** the out-of-range rows with NaN, and the CPU reference of the oracle throws.
 */
test.skipIf(!GPU_AVAILABLE)(
  "out-of-range embedding indices contaminate with NaN on the GPU and throw in the CPU reference (real GPU)",
  async () => {
    const weight = fill([3, 2], POSITIVE);
    // Mixes in one index 3 (beyond the vocabulary size) and one -1 (negative); the rest look up valid rows
    const index = fill([4], (i) => [1, 3, -1, 0][i], "i32");
    const gpu = await acquireGpu();
    try {
      const actual = await runCase(gpu, {
        name: "embedding oob",
        op: "embedding",
        inputs: [weight, index],
        outShapes: [[4, 2]],
        attrs: { padding_idx: -1 },
      });
      const values = [...actual.data];
      assertEquals(
        values.map((v) => Number.isNaN(v)),
        [false, false, true, true, true, true, false, false],
      );
      // Indices within the contract look up their own row as usual (only out-of-range rows are contaminated)
      assertEquals(values[0], weight.data[2]);
      assertEquals(values[1], weight.data[3]);
      assertEquals(values[6], weight.data[0]);
    } finally {
      gpu.destroy();
    }
    assertThrows(() => applyReferenceOp("embedding", [weight, index], { padding_idx: -1 }, [4, 2]));
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "boundaries (one large tensor / row count above the workgroup limit) match the CPU reference (real GPU)",
  () => checkAll(BOUNDARY_CASES),
);

/**
 * sym_prefix_slice (ADR 0010) — cuts **the head** of a constant baked at Tmax to the runtime length.
 * Execution reuses the strided materializing copy (no new kernel), and the read stride is the contiguous stride of
 * the input (Tmax shape).
 *
 * MUST: **place only cases with T < Tmax**. At T = Tmax the error "building the stride from the output shape after
 * binding" and the correct implementation give the same answer, so that shape is no detector (the same trap as the
 * involution reorders of the wave 2 permute testing nothing).
 * MUST: shrink not only the last dimension but **leading dimensions** too. Shrinking only the last dimension is a
 * special case where the row step does not change, and a mixed-up stride does not show in the values.
 */
const prefixSliceGraph = (
  constShape: readonly number[],
  outShape: readonly (number | string)[],
  slices: readonly { dim: number; coeff: number; offset: number }[],
  dtype: "f32" | "i32",
): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["sym_prefix_slice"] },
  symbols: ["T"],
  // A binding can only be taken from a dimension position of an input shape (docs/ir-v2.md), so one dummy input
  // carries T in its plain form.
  inputs: [{ name: "bind", dtype: "f32", shape: ["T"] }],
  outputs: ["y"],
  initializers: { table: {} },
  values: {
    table: { dtype, shape: [...constShape] },
    y: { dtype, shape: [...outShape] },
  },
  nodes: [
    {
      op: "sym_prefix_slice",
      ins: ["table"],
      outs: ["y"],
      attrs: { sym: "T", slices: slices.map((slice) => ({ ...slice })) },
    },
  ],
});

test.skipIf(!GPU_AVAILABLE)(
  "sym_prefix_slice cuts the head of a Tmax constant (real GPU / T < Tmax)",
  async () => {
    const bound = 3;
    const cases = [
      {
        name: "shrinks both axes (relative-position bucket table i32 [6,5] → [3,3])",
        constShape: [6, 5],
        outShape: ["T", "T"],
        resolved: [3, 3],
        slices: [
          { dim: 0, coeff: 1, offset: 0 },
          { dim: 1, coeff: 1, offset: 0 },
        ],
        dtype: "i32" as const,
        generator: INTEGERS,
      },
      {
        name: "shrinks only a leading dimension (position table f32 [7,4] → [3,4])",
        constShape: [7, 4],
        outShape: ["T", 4],
        resolved: [3, 4],
        slices: [{ dim: 0, coeff: 1, offset: 0 }],
        dtype: "f32" as const,
        generator: SIGNED,
      },
      {
        name: "with a coefficient (f32 [16] → [2T+1] = [7])",
        constShape: [16],
        outShape: ["2T+1"],
        resolved: [7],
        slices: [{ dim: 0, coeff: 2, offset: 1 }],
        dtype: "f32" as const,
        generator: POSITIVE,
      },
      {
        name: "shrinks the middle axis of rank 4 (i32 [2,6,3,4] → [2,3,3,4])",
        constShape: [2, 6, 3, 4],
        outShape: [2, "T", 3, 4],
        resolved: [2, 3, 3, 4],
        slices: [{ dim: 1, coeff: 1, offset: 0 }],
        dtype: "i32" as const,
        generator: INTEGERS,
      },
    ];
    const gpu = await acquireGpu();
    try {
      for (const testCase of cases) {
        const table = fill(testCase.constShape, testCase.generator, testCase.dtype);
        const graph = prefixSliceGraph(
          testCase.constShape,
          testCase.outShape,
          testCase.slices,
          testCase.dtype,
        );
        const session = await createSessionFromContainer(
          gpu,
          await openModelBytes(graph, [
            {
              graph: GRAPH_NAME,
              initializer: "table",
              bytes: new Uint8Array(
                table.data.buffer,
                table.data.byteOffset,
                table.data.byteLength,
              ),
              encoding: { codec: testCase.dtype },
            },
          ]),
          GRAPH_NAME,
        );
        try {
          const outputs = await session.run({ bind: fill([bound], SIGNED) });
          // MUST: keep the expected shape as **a literal** (`resolved`). Passing the actual output's shape to
          // applyReferenceOp and comparing against the same thing passes vacuously even if the path that derives the
          // prefix length from the bindings is completely broken.
          assertEquals(
            outputs["y"].shape,
            testCase.resolved,
            `${testCase.name}: shape after binding`,
          );
          const expected = applyReferenceOp(
            "sym_prefix_slice",
            [table],
            graph.nodes[0].attrs as Record<string, unknown>,
            testCase.resolved,
          );
          // MUST: take the band from the measured table (falling to the default band lets EXACT ops through with 1e-5-level noise)
          const report = compareTensors(outputs["y"], expected, opTolerance("sym_prefix_slice"));
          assertEquals(report.pass, true, `${testCase.name}: ${formatAllclose(report)}`);
        } finally {
          await session.dispose();
        }
      }
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * Symbolic-axis cat (ADR 0046) — closes on a real GPU the shapes where the concatenation axis resolves to
 * `T+constant` / `2T`.
 *
 * What was relaxed is **only the declaration-level guard**; after binding the shapes are numeric, so the strided
 * write-copy kernel should be unchanged. This test replaces "should" with a measurement: if a path still builds the
 * write offset from the sum of static axes, it is bound to drift.
 *
 * MUST: keep the expected shape as **a literal** (`resolved`). Passing the actual output's shape to the reference
 * side too passes vacuously even if the path that derives the concatenation axis length from the bindings is
 * completely broken.
 */
test.skipIf(!GPU_AVAILABLE)(
  "symbolic-axis cat (T+constant / 2T) matches the CPU reference (real GPU)",
  async () => {
    const cases = [
      {
        name: "symbol + constant [1,T,4] + [1,3,4] dim=1 (same shape as DiT's KV concatenation)",
        inShapes: [
          [1, "T", 4],
          [1, 3, 4],
        ] as const,
        outShape: [1, "T+3", 4] as const,
        bound: 5,
        resolved: [1, 8, 4],
        dim: 1,
      },
      {
        name: "same symbol on both sides [T,3] + [T,3] dim=0",
        inShapes: [
          ["T", 3],
          ["T", 3],
        ] as const,
        outShape: ["2T", 3] as const,
        bound: 4,
        resolved: [8, 3],
        dim: 0,
      },
    ];
    const gpu = await acquireGpu();
    try {
      for (const testCase of cases) {
        const graph = singleOpDeclaration(
          "cat",
          testCase.inShapes.map((shape) => [...shape]),
          [[...testCase.outShape]],
          { symbols: ["T"], attrs: { dim: testCase.dim } },
        );
        // The binding is taken from a dimension position of an input shape (the position where 'T' appears in plain form).
        const inputs = testCase.inShapes.map((shape) =>
          fill(
            shape.map((dim) => (dim === "T" ? testCase.bound : (dim as number))),
            SIGNED,
          ),
        );
        const session = await createSessionFromContainer(
          gpu,
          await openModelBytes(graph, []),
          GRAPH_NAME,
        );
        try {
          const outputs = await session.run(
            Object.fromEntries(inputs.map((input, index) => [`x${index}`, input])),
          );
          assertEquals(
            outputs["y"].shape,
            testCase.resolved,
            `${testCase.name}: shape after binding`,
          );
          const expected = applyReferenceOp(
            "cat",
            inputs,
            { dim: testCase.dim },
            testCase.resolved,
          );
          // MUST: take the band from the measured table (falling to the default band lets EXACT ops through with 1e-5-level noise)
          const report = compareTensors(outputs["y"], expected, opTolerance("cat"));
          assertEquals(report.pass, true, `${testCase.name}: ${formatAllclose(report)}`);
        } finally {
          await session.dispose();
        }
      }
    } finally {
      gpu.destroy();
    }
  },
);
