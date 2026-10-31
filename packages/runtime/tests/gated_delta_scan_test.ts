/**
 * `gated_delta_scan` (ADR 0135, the value form) without a GPU: the CPU reference against the f64
 * oracle and upstream's f32 functions (fixtures/gated-delta-scan.safetensors), the acceptance band's
 * power against the spike's fault classes, the accurate exp's polynomial in f32 arithmetic, the
 * shape rule's symbolic length, the GPU subset check at Session construction and the kernel's
 * params, dispatch and limits.
 *
 * The GPU kernel against this reference is tests/gpu_gated_delta_scan_test.ts.
 */

import { assert, assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { CodegenError, DispatchLimitError } from "../src/codegen/errors.ts";
import { GATED_DELTA_SCAN_HEAD_DIM } from "../src/codegen/limits.ts";
import { EXP_ACCURATE_WGSL } from "../src/codegen/numerics-wgsl.ts";
import {
  GATED_DELTA_SCAN_KEY,
  GATED_DELTA_SCAN_WORKGROUP_SIZE,
  gatedDeltaScanParams,
  gatedDeltaScanWgsl,
  gatedDeltaScanWorkgroups,
  gatedDeltaScanWorkgroupStorageBytes,
} from "../src/kernels/gated-delta-scan.ts";
import { computeOutputShape, OpContractError, resolveOpContract } from "../src/ops.ts";
import {
  applyReferenceOp,
  referenceGatedDeltaScanF64,
  type RefTensor,
} from "../src/reference/ops.ts";
import { ExecutionError, planGraph, validateGraphContracts } from "../src/runtime/plan.ts";
import {
  bandShare,
  describeBand,
  errorMetrics,
  type GatedDeltaCase,
  gatedDeltaCases,
} from "./helpers/gated-delta-fixture.ts";
import { mergeGraph } from "./helpers/merged-graph.ts";
import { singleOpDeclaration } from "./helpers/model-fixture.ts";

const CASES = await gatedDeltaCases();

const caseNamed = (name: string): GatedDeltaCase => {
  const found = CASES.find((entry) => entry.name === name);
  assert(found !== undefined, `fixture case '${name}' is missing`);
  return found;
};

const ORACLES = new Map<string, Float64Array<ArrayBuffer>>();

/** The f64 CPU reference of a case (computed once — the long case is 16.8M multiply-adds). */
const oracleOf = (entry: GatedDeltaCase): Float64Array<ArrayBuffer> => {
  const cached = ORACLES.get(entry.name);
  if (cached !== undefined) return cached;
  const computed = referenceGatedDeltaScanF64(...entry.inputs).data;
  ORACLES.set(entry.name, computed);
  return computed;
};

describe("the fixture", () => {
  it("holds the cases the landing tests need (ADR 0135 decision 6)", () => {
    assertEquals(
      CASES.map((entry) => entry.name),
      ["mixed_B2_T16", "init_T64", "long_T512"],
    );
    // MUST: a long-memory case with T ≥ 512 sits in the default lane (the built-in-exp regression
    // is not caught at T = 64 — research 2026-10-27 §6).
    assert(caseNamed("long_T512").shape.T >= 512);
    // B > 1 and a head broadcast with more than one key head (the key-head fault can act only there)
    assertEquals(caseNamed("mixed_B2_T16").shape.B, 2);
    assertEquals(caseNamed("mixed_B2_T16").shape.Hk, 2);
    for (const entry of CASES) {
      assertEquals(entry.shape.dk, GATED_DELTA_SCAN_HEAD_DIM, entry.name);
      assertEquals(entry.shape.dv, GATED_DELTA_SCAN_HEAD_DIM, entry.name);
      // the band is twice the larger upstream error, per metric
      for (const metric of ["maxAbs", "relL2"] as const) {
        assertEquals(
          entry.band[metric],
          2 * Math.max(entry.upstream.chunk[metric], entry.upstream.recurrent[metric]),
          `${entry.name} ${metric}`,
        );
      }
    }
  });
});

describe("the CPU reference (f64)", () => {
  it("matches the Python f64 oracle element for element on the stored case", () => {
    const entry = caseNamed("mixed_B2_T16");
    assert(entry.stored !== undefined);
    const metrics = errorMetrics(
      oracleOf(entry),
      entry.stored.oracle,
      entry.shape.Hv,
      entry.shape.dv,
    );
    // Two f64 evaluations in different summation orders (the spike measured 4.8e-16 rel_l2), and
    // the stored oracle is an f32 head + tail pair (about 2^-48 relative).
    assert(metrics.relL2 < 1e-13, `rel_l2 ${metrics.relL2}`);
    assert(metrics.maxRelHead < 1e-13, `max_rel_head ${metrics.maxRelHead}`);
  });

  it("matches the oracle's sums and samples on every case", () => {
    for (const entry of CASES) {
      const oracle = oracleOf(entry);
      let sum = 0;
      let sumSquares = 0;
      let largest = 0;
      for (const value of oracle) {
        sum += value;
        sumSquares += value * value;
        largest = Math.max(largest, Math.abs(value));
      }
      const close = (actual: number, expected: number, scale: number, what: string): void => {
        assert(
          Math.abs(actual - expected) <= 1e-12 * scale,
          `${entry.name} ${what}: ${actual} against the oracle's ${expected}`,
        );
      };
      close(sum, entry.oracle.sum, largest * Math.sqrt(oracle.length), "sum");
      close(sumSquares, entry.oracle.sumSquares, entry.oracle.sumSquares, "sum of squares");
      for (const [index, value] of entry.oracle.samples) {
        close(oracle[index], value, largest, `element ${index}`);
      }
    }
  });

  it("is what the unified entry returns, rounded to f32 once", () => {
    const entry = caseNamed("init_T64");
    const rounded = applyReferenceOp("gated_delta_scan", entry.inputs);
    assertEquals(rounded.dtype, "f32");
    assertEquals(rounded.shape, [entry.shape.B, entry.shape.T, entry.shape.Hv, entry.shape.dv]);
    assertEquals(rounded.data, Float32Array.from(oracleOf(entry)));
  });

  it("recomputes the generator's upstream errors with the same metric definitions", () => {
    const entry = caseNamed("mixed_B2_T16");
    assert(entry.stored !== undefined);
    const oracle = oracleOf(entry);
    for (const [name, output] of [
      ["chunk", entry.stored.chunk],
      ["recurrent", entry.stored.recurrent],
    ] as const) {
      const metrics = errorMetrics(output, oracle, entry.shape.Hv, entry.shape.dv);
      const recorded = entry.upstream[name];
      for (const metric of ["maxAbs", "relL2", "maxRelHead"] as const) {
        const relative = Math.abs(metrics[metric] - recorded[metric]) / recorded[metric];
        assert(
          relative < 1e-6,
          `${name} ${metric}: ${metrics[metric]} against ${recorded[metric]}`,
        );
      }
    }
  });

  it("holds the semantics for head dims the GPU does not implement (dk = 3, dv = 2)", () => {
    // Hand-computed: one key head, one value head, k = q = e₀, g = 0 (α = 1), β = 1/2.
    // t = 0: m = 0, δ = ½·v₀, S = e₀ δᵀ, o₀ = Sᵀ q·3^-½ = ½·v₀·3^-½.
    // t = 1: m = Sᵀ e₀ = ½·v₀, δ = ½·(v₁ − ½·v₀), o₁ = (½·v₀ + δ)·3^-½.
    const tensor = (shape: readonly number[], values: readonly number[]): RefTensor => ({
      dtype: "f32",
      shape,
      data: Float32Array.from(values),
    });
    const unit = [1, 0, 0, 1, 0, 0];
    const values = [4, -2, 2, 6];
    const out = referenceGatedDeltaScanF64(
      tensor([1, 2, 1, 3], unit),
      tensor([1, 2, 1, 3], unit),
      tensor([1, 2, 1, 2], values),
      tensor([1, 2, 1], [0, 0]),
      tensor([1, 2, 1], [0.5, 0.5]),
    );
    const scale = 1 / Math.sqrt(3);
    const first = [2, -1];
    const delta = [0.5 * (2 - 2), 0.5 * (6 + 1)];
    const expected = [...first, first[0] + delta[0], first[1] + delta[1]].map((x) => x * scale);
    assertEquals(out.shape, [1, 2, 1, 2]);
    out.data.forEach((value, index) => {
      assert(Math.abs(value - expected[index]) < 1e-15, `o[${index}] = ${value}`);
    });
  });
});

type Fault = "no-decay" | "no-beta" | "head-mod" | "o-before-update";

/**
 * A second f64 recurrence with one injected fault (the five of the spike — research §5.3; the
 * missing initial state cannot act in the value form, which has no initial state). Without a fault
 * it equals the product reference bit for bit (the first test below checks that), so a fault is the
 * only difference.
 */
const faultyScan = (entry: GatedDeltaCase, fault: Fault | undefined): Float64Array<ArrayBuffer> => {
  const [q, k, v, g, beta] = entry.inputs.map((input) => input.data);
  const { B, T, Hk, Hv, dk, dv } = entry.shape;
  const ratio = Hv / Hk;
  const scale = 1 / Math.sqrt(dk);
  const out = new Float64Array(B * T * Hv * dv);
  const state = new Float64Array(dk * dv);
  const delta = new Float64Array(dv);
  for (let item = 0; item < B; item += 1) {
    for (let head = 0; head < Hv; head += 1) {
      const keyHead = fault === "head-mod" ? head % Hk : Math.floor(head / ratio);
      state.fill(0);
      for (let t = 0; t < T; t += 1) {
        const keyBase = ((item * T + t) * Hk + keyHead) * dk;
        const gate = (item * T + t) * Hv + head;
        const decay = fault === "no-decay" ? 1 : Math.exp(g[gate]);
        const strength = fault === "no-beta" ? 1 : beta[gate];
        const valueBase = gate * dv;
        delta.fill(0);
        for (let row = 0; row < dk; row += 1) {
          for (let column = 0; column < dv; column += 1) {
            const decayed = state[row * dv + column] * decay;
            state[row * dv + column] = decayed;
            delta[column] += decayed * k[keyBase + row];
          }
        }
        for (let column = 0; column < dv; column += 1) {
          delta[column] = strength * (v[valueBase + column] - delta[column]);
        }
        for (let row = 0; row < dk; row += 1) {
          const query = q[keyBase + row] * scale;
          for (let column = 0; column < dv; column += 1) {
            const before = state[row * dv + column];
            const updated = before + k[keyBase + row] * delta[column];
            state[row * dv + column] = updated;
            out[valueBase + column] += (fault === "o-before-update" ? before : updated) * query;
          }
        }
      }
    }
  }
  return out;
};

/** Whether `h % Hk` differs from `⌊h / r⌋` for some value head (else the fault cannot act). */
const headFaultActs = (entry: GatedDeltaCase): boolean => {
  const { Hk, Hv } = entry.shape;
  return Array.from({ length: Hv }, (_, head) => head).some(
    (head) => head % Hk !== Math.floor(head / (Hv / Hk)),
  );
};

describe("the acceptance band (ADR 0135 decision 6)", () => {
  it("the fault-free second recurrence is the product reference bit for bit", () => {
    for (const entry of CASES) {
      assertEquals(faultyScan(entry, undefined), oracleOf(entry), entry.name);
    }
  });

  it("rejects each of the spike's fault classes wherever the fault can act (non-tautology)", () => {
    for (const entry of CASES) {
      const oracle = oracleOf(entry);
      for (const fault of ["no-decay", "no-beta", "head-mod", "o-before-update"] as const) {
        const output = faultyScan(entry, fault);
        const label = `${entry.name} ${fault}`;
        if (fault === "head-mod" && !headFaultActs(entry)) {
          // One key head: `h % Hk` and `⌊h/r⌋` are both 0 — the fault cannot act here.
          assertEquals(output, oracle, label);
          continue;
        }
        const metrics = errorMetrics(output, oracle, entry.shape.Hv, entry.shape.dv);
        // The spike measured shares of at least 1e6 for these coarse faults (research §5.3).
        assert(
          bandShare(metrics, entry.band) > 1e3,
          `${label}: ${describeBand(metrics, entry.band)}`,
        );
      }
    }
    // the key-head fault does act somewhere
    assert(CASES.some(headFaultActs));
  });

  it("rejects an output that is not finite (an unwritten, NaN-poisoned element)", () => {
    const entry = caseNamed("init_T64");
    const output = Float32Array.from(oracleOf(entry));
    output[output.length - 1] = Number.NaN;
    const metrics = errorMetrics(output, oracleOf(entry), entry.shape.Hv, entry.shape.dv);
    assertEquals(metrics.nonFinite, 1);
    assertEquals(bandShare(metrics, entry.band), Infinity);
  });
});

/** The f32 value nearest to `x` with WGSL's `round` (half to even). */
const roundHalfEven = (x: number): number => {
  const floor = Math.floor(x);
  const fraction = x - floor;
  if (fraction !== 0.5) return Math.round(x);
  return floor % 2 === 0 ? floor : floor + 1;
};

/**
 * The accurate exp's body evaluated in f32 arithmetic (`Math.fround` after every operation, no
 * fused multiply-add) — the body's algorithm, not the GPU (a GPU may contract to fma; the GPU test
 * measures the real thing).
 */
const EXP_CONSTANTS = {
  log2e: Math.fround(1.4426950408889634),
  ln2High: Math.fround(0.693145751953125),
  ln2Low: Math.fround(1.428606820309417e-6),
  taylor: [0.0001984127, 0.0013888889, 0.008333334, 0.041666668, 0.16666667, 0.5, 1.0, 1.0].map(
    Math.fround,
  ),
} as const;

const expAccurateF32 = (x: number): number => {
  const f = Math.fround;
  if (x < -87) return 0;
  const n = roundHalfEven(f(x * EXP_CONSTANTS.log2e));
  const r = f(f(x - f(n * EXP_CONSTANTS.ln2High)) - f(n * EXP_CONSTANTS.ln2Low));
  let p = EXP_CONSTANTS.taylor[0];
  for (const coefficient of EXP_CONSTANTS.taylor.slice(1)) p = f(coefficient + f(r * p));
  return f(p * 2 ** n);
};

/** One f32 ULP at |value| (the gap to the next f32 away from zero). */
const ulp = (value: number): number => {
  const bits = new Uint32Array(Float32Array.of(Math.abs(value)).buffer);
  bits[0] += 1;
  return new Float32Array(bits.buffer)[0] - Math.fround(Math.abs(value));
};

describe("the accurate exp (EXP_ACCURATE_WGSL)", () => {
  it("is the body this file emulates (the same constants, in the same Horner order)", () => {
    for (const literal of [
      "1.4426950408889634",
      "0.693145751953125",
      "1.428606820309417e-6",
      "1.0 + r * (1.0 + r * (0.5 + r * (0.16666667 + r * (0.041666668 + r * (0.008333334 + r * (0.0013888889 + r * 0.0001984127))))))",
      "if (x < -87.0) { return 0.0; }",
      "ldexp(p, i32(n))",
    ]) {
      assert(EXP_ACCURATE_WGSL.includes(literal), literal);
    }
  });

  it("keeps at most 1 ULP and almost no bias on the long-memory range [-0.1, 0) in f32 arithmetic", () => {
    // The built-in exp measured max 3.00 / mean signed −0.337 ULP here on the GPU, this body
    // 1.00 / −0.091 (research §5.4).
    const count = 1 << 16;
    let worst = 0;
    let signed = 0;
    for (let i = 0; i < count; i += 1) {
      const x = Math.fround(-0.1 + (0.1 * (i + 0.5)) / count);
      const exact = Math.exp(x);
      const error = (expAccurateF32(x) - exact) / ulp(exact);
      worst = Math.max(worst, Math.abs(error));
      signed += error;
    }
    assert(worst <= 1, `max ${worst} ULP`);
    assert(Math.abs(signed / count) < 0.2, `mean signed ${signed / count} ULP`);
  });

  it("is exact at 0 and −0, 0 at −∞ and below −87, NaN at NaN", () => {
    assertEquals(expAccurateF32(0), 1);
    assertEquals(expAccurateF32(-0), 1);
    assertEquals(expAccurateF32(-Infinity), 0);
    assertEquals(expAccurateF32(-87.5), 0);
    assert(expAccurateF32(-86.9) > 0);
    assert(Number.isNaN(expAccurateF32(Number.NaN)));
  });
});

describe("the shape rule and the GPU subset", () => {
  const contract = resolveOpContract("gated_delta_scan");
  const shape = (ins: readonly (readonly number[])[]): number[] =>
    computeOutputShape(contract, ins, "t")[0];

  it("derives the head broadcast and keeps the semantics for any head dim", () => {
    assertEquals(
      shape([
        [2, 5, 3, 4],
        [2, 5, 3, 4],
        [2, 5, 6, 7],
        [2, 5, 6],
        [2, 5, 6],
      ]),
      [2, 5, 6, 7],
    );
    assertEquals(
      shape([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 4, 1],
        [1, 1, 4],
        [1, 1, 4],
      ]),
      [1, 1, 4, 1],
    );
  });

  it("rejects axes of length 0 after binding (a symbolic T bound to 0)", () => {
    for (const ins of [
      [
        [1, 0, 1, 4],
        [1, 0, 1, 4],
        [1, 0, 2, 4],
        [1, 0, 2],
        [1, 0, 2],
      ],
      [
        [0, 3, 1, 4],
        [0, 3, 1, 4],
        [0, 3, 2, 4],
        [0, 3, 2],
        [0, 3, 2],
      ],
      [
        [1, 3, 1, 0],
        [1, 3, 1, 0],
        [1, 3, 2, 4],
        [1, 3, 2],
        [1, 3, 2],
      ],
    ]) {
      assertThrows(() => shape(ins), OpContractError, "at least 1");
    }
  });

  const graph = (keyDim: number | string, valueDim: number | string) =>
    mergeGraph(
      singleOpDeclaration(
        "gated_delta_scan",
        [
          [1, "T", 2, keyDim],
          [1, "T", 2, keyDim],
          [1, "T", 4, valueDim],
          [1, "T", 4],
          [1, "T", 4],
        ],
        [[1, "T", 4, valueDim]],
        { symbols: ["T", ...(typeof keyDim === "string" ? [keyDim] : [])] },
      ),
    );

  it("plans a symbolic length axis through the node", () => {
    const plan = planGraph(graph(128, 128), { T: 9 });
    assertEquals(plan.nodes[0].outputs[0].shape, [1, 9, 4, 128]);
    assertEquals(plan.nodes[0].inputShapes[0], [1, 9, 2, 128]);
  });

  it("accepts dk = dv = 128 at Session construction and rejects other head dims there", () => {
    assertEquals(validateGraphContracts(graph(128, 128)), undefined);
    assertThrows(() => validateGraphContracts(graph(64, 128)), ExecutionError, "q (dk) 64");
    assertThrows(() => validateGraphContracts(graph(128, 256)), ExecutionError, "v (dv) 256");
    // a symbolic head dim is not the literal 128 either
    assertThrows(() => validateGraphContracts(graph("D", 128)), ExecutionError, '"D"');
  });
});

describe("the kernel's params, dispatch and limits", () => {
  const dims = { time: 7, batch: 2, keyHeads: 16, valueHeads: 32, keyDim: 128, valueDim: 128 };

  it("writes { T, B, Hk, Hv } and fails loudly outside the implemented subset", () => {
    assertEquals([...gatedDeltaScanParams(dims)], [7, 2, 16, 32]);
    assertEquals(gatedDeltaScanParams(dims).byteLength, 16);
    assertThrows(() => gatedDeltaScanParams({ ...dims, keyDim: 64 }), CodegenError, "dk 64");
    assertThrows(() => gatedDeltaScanParams({ ...dims, valueDim: 64 }), CodegenError, "dv 64");
    assertThrows(() => gatedDeltaScanParams({ ...dims, valueHeads: 24 }), CodegenError, "Hk 16");
    assertThrows(() => gatedDeltaScanParams({ ...dims, valueHeads: 8 }), CodegenError, "Hk 16");
    assertThrows(() => gatedDeltaScanParams({ ...dims, keyHeads: 0 }), CodegenError, "Hk 0");
    assertThrows(() => gatedDeltaScanParams({ ...dims, time: 0 }), CodegenError, "T 0");
    assertThrows(() => gatedDeltaScanParams({ ...dims, time: -1 }), CodegenError, "time");
  });

  it("dispatches one workgroup per (batch element, value head, 32 value columns) and fails past the limit", () => {
    assertEquals(gatedDeltaScanWorkgroups(1, 32, 65535), 128);
    assertEquals(gatedDeltaScanWorkgroups(2, 32, 65535), 256);
    assertEquals(gatedDeltaScanWorkgroups(16383, 1, 65535), 65532);
    assertThrows(() => gatedDeltaScanWorkgroups(16384, 1, 65535), DispatchLimitError);
  });

  it("fits the WebGPU default limits (256 invocations, 16 KiB of workgroup storage)", () => {
    assertEquals(GATED_DELTA_SCAN_WORKGROUP_SIZE, 256);
    assertEquals(gatedDeltaScanWorkgroupStorageBytes(), 13888);
    assert(gatedDeltaScanWorkgroupStorageBytes() <= 16384);
    assert(gatedDeltaScanWgsl().includes(`@workgroup_size(${GATED_DELTA_SCAN_WORKGROUP_SIZE})`));
  });

  it("uses the accurate exp, unrolls the token block and has no runtime-bounded inner loop", () => {
    const wgsl = gatedDeltaScanWgsl();
    assertEquals(GATED_DELTA_SCAN_KEY, "gated_delta_scan:v1:f32:s8:c32:tb8:d128");
    assert(wgsl.includes(EXP_ACCURATE_WGSL));
    // the only `exp` call is the accurate one, at staging (MUST of decision 3) — comments aside
    const code = wgsl.replaceAll(/\/\/.*$/gm, "");
    assertEquals([...code.matchAll(/\bexp\s*\(/g)].length, 0);
    assertEquals([...code.matchAll(/\bexp_accurate\(/g)].length, 2);
    // perf ledger K-94: the token block is unrolled and the only loop is the block loop
    assertEquals([...wgsl.matchAll(/\bfor\s*\(/g)].length, 0);
    assertEquals([...wgsl.matchAll(/\bloop\s*\{/g)].length, 1);
    for (let token = 0; token < 8; token += 1) {
      assert(wgsl.includes(`if (t0 + ${token}u < T) {`), `token ${token} of the block`);
    }
    // one barrier per token, plus block 0, the first exchange and the staging store
    assertEquals([...wgsl.matchAll(/workgroupBarrier\(\)/g)].length, 8 + 3);
    // the value form writes o only (no state binding)
    assertEquals([...wgsl.matchAll(/var<storage, read_write>/g)].length, 1);
  });
});
