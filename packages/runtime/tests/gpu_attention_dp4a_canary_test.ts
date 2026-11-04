// The canary that **decides by an actual run** the i8a8 dot-product variant of fused attention
// (src/gpu/attention-dp4a-canary.ts).
//
// Background = the Metal (Apple GPU) section of docs/known-issues.md. At first we reasoned that the
// claim "the dot4I8Packed version and the emulated version return the same integers"
// (src/kernels/linear-i8a8.ts) breaks on Apple M2 **only for fused attention**, but that was
// retracted on 2026-08-29 (both variants are bit-identical on M2 too; what was off was 1 ULP of the
// shared f32 epilogue). To keep "no automatic selection of a path that changes the numbers from
// adapter capabilities" (ADR 0058 decision 2), the variant can only be decided by **comparison
// against a known answer**, not by enumeration.
//
// ## The contract this file pins (decision rule v2)
//
// 1. On a healthy device **both the dp4a and emu arms match the known answer with atol=0**, and the
//    canary picks dp4a (= behaviour unchanged by a single bit from before).
// 2. On a device where only dp4a misses the known answer, only attention drops to emu.
// 3. When **neither arm matches exactly**, the sanity band decides — if only one arm is inside the
//    band, that arm; if both arms are inside the band and bit-identical to each other, dp4a (the
//    normal path on Apple M2: the rounding of the shared f32 epilogue only differs from the known
//    answer by a few ULP, and the variant choice is irrelevant to the numbers). The decision
//    carries the fact "it was not an exact match" in its return value.
// 4. **If both arms fall outside the band, fail loudly** (it is not a problem of how the variant is
//    chosen, so there is nothing to degrade to).
// 5. The decision runs **once per device** (Promise memoization). A Session that is not a8 pays not
//    a single dispatch. Only a decision that failed with a transient allocation failure (OOM) is
//    not kept in the memo, and runs again on the next request.
// 6. The decision shows in the diagnostics (`:dp4aEmu` in the pipeline key) (ADR 0058 decision 3).
//
// ## About fault injection
//
// The mismatch path, the in-band path and the both-mismatch path cannot be produced on a healthy
// real machine (the same situation as the shader-f16 canary of src/gpu/device.ts). So **a hook that
// deliberately breaks the generated WGSL** (`CanaryWgslPatch`) is used to hit the decision branches
// themselves on a real GPU. MUST: first make the injection taking effect a check of its own (if the
// replacement misses, it becomes "broken yet matched" = a tautological test).

import { test } from "vite-plus/test";
import { assert, assertEquals, assertRejects, assertStrictEquals } from "@std/assert";
import { f16BitsToF32, roundToF16 } from "../src/format/f16.ts";
import {
  acquireGpu,
  type GpuContext,
  GpuDeviceLostError,
  GpuFeatureError,
  RUNTIME_INTERNAL,
} from "../src/gpu/device.ts";
import { GpuOutOfMemoryError } from "../src/gpu/error-scope.ts";
import {
  type AttentionI8a8Decision,
  buildPvCase,
  buildQkCase,
  type CanaryWgslPatch,
  decideAttentionI8a8Dot,
  formatAttentionI8a8Decision,
  packScoresF16,
  probeAttentionI8a8Dot,
  SANITY_ABSOLUTE_FLOOR,
  SANITY_RELATIVE_TOLERANCE,
} from "../src/gpu/attention-dp4a-canary.ts";
import {
  attentionPvI8a8Key,
  attentionPvI8a8UsesVec4,
  attentionPvI8a8Wgsl,
  attentionQkI8a8Key,
  attentionQkI8a8UsesVec4,
  attentionQkI8a8Wgsl,
} from "../src/kernels/attention-i8a8.ts";
import {
  defaultI8a8Geometry,
  type I8a8Geometry,
  i8a8GeometryKeyPart,
  i8a8KPacks,
  i8a8TileM,
  i8a8TileN,
} from "../src/kernels/i8a8-geometry.ts";
import { createSessionFromContainer, type SessionOptions } from "../src/runtime/executor.ts";
import type { I8a8Dot, SessionDiagnostics } from "../src/runtime/session-types.ts";
import { fill, GRAPH_NAME, openGraphModel, singleOpDeclaration } from "./helpers/model-fixture.ts";
import { fakeDevice, fakeGpuContext } from "./helpers/fake-gpu.ts";
import { deviceGeometryProfile } from "./helpers/geometry.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { assertPipelineCensus } from "./helpers/pipeline-census.ts";

/**
 * The i8a8 attention geometry of the default profile (the same one the default Session passes to
 * the canary).
 */
const DEFAULT_CANARY_GEOMETRY = {
  qk: defaultI8a8Geometry("attention_qk"),
  pv: defaultI8a8Geometry("attention_pv"),
};

/**
 * The minimum margin by which `127·exp(S−m)` should stay away from a half-integer (an order of
 * magnitude above the ~1e-5 error of WGSL's `exp`).
 */
const QUANT_MARGIN = 0.3;

/**
 * How many times **the sanity band's tolerance** of the element the distance from `value` to the
 * f16 rounding boundary (the midpoint to the neighbouring f16 value) is. Above 1 means "a
 * difference inside the band does not move the f16 step" — exactly the sufficient condition for the
 * band to close for the s16 variant too (RC1-1).
 *
 * The f16 spacing is constant per binary exponent, and the boundaries are `h ± spacing/2` (only
 * when `h` is exactly a power of 2 is the spacing below halved). The subnormal range has the
 * constant spacing `2⁻²⁴`, so it is clamped from below.
 */
const f16BoundaryBandRatio = (value: number): number => {
  const magnitude = Math.abs(value);
  const step = roundToF16(magnitude);
  const exponent = Math.floor(Math.log2(step));
  const above = Math.max(2 ** (exponent - 10), 2 ** -24);
  const below = step === 2 ** exponent ? Math.max(2 ** (exponent - 11), 2 ** -24) : above;
  const distance = Math.min(
    Math.abs(magnitude - (step + above / 2)),
    Math.abs(magnitude - (step - below / 2)),
  );
  return distance / Math.max(magnitude * SANITY_RELATIVE_TOLERANCE, SANITY_ABSOLUTE_FLOOR);
};

/**
 * Breaks **only** the dp4a variant (this spelling does not appear in the emulated generated code).
 */
const BREAK_DP4A: CanaryWgslPatch = (wgsl) =>
  wgsl.replace("return dot4I8Packed(a, b);", "return dot4I8Packed(a, b) + 1;");

/** Breaks both variants (the body of `idot` starts with `  return dot` in both). */
const BREAK_BOTH: CanaryWgslPatch = (wgsl) => wgsl.replace("  return dot", "  return 1 + dot");

/**
 * Puts a relative `factor − 1` perturbation into **the ③PV epilogue only** (it affects both arms
 * equally — this spelling appears in the same form in both the dp4a and emu generated code).
 *
 * Why only ③PV: the s16 variant of ①QK rounds its output to f16 for storage, so **a perturbation
 * inside the band is absorbed by the storage rounding and does not move a single bit** (the ①QK
 * known answer of the fixed input sits exactly on the f16 grid, 24 times the band away from the
 * rounding boundary — RC1-1, the margin check below). Only ③PV, which writes O in f32 for all 3
 * variants, lets the perturbation "be observed at exactly the factor put in".
 */
const nudgePv =
  (factor: string): CanaryWgslPatch =>
  (wgsl) =>
    wgsl.replaceAll("let prow = stats[", `let prow = ${factor} * stats[`);

/**
 * A perturbation **inside** the band (both arms, identical). `1.0000001` rounds to 1 + 1 ULP in
 * f32, so multiplying by it always moves the output by 1–2 ULP (= exact match always fails), but
 * the relative ~1.2e-7 is 2 orders of magnitude below the band (1e-5).
 */
const NUDGE_BOTH = nudgePv("1.0000001");

/**
 * A perturbation **outside** the band (both arms, identical). Relative 1e-4 is an order of
 * magnitude above the band — the same level as the 9.5e-5 measured by fault injection on M2.
 */
const SKEW_BOTH = nudgePv("1.0001");

/** Only one arm (emu) stays inside the band = what a device with a truly broken dp4a looks like. */
const NUDGE_BOTH_BREAK_DP4A: CanaryWgslPatch = (wgsl) => BREAK_DP4A(NUDGE_BOTH(wgsl));

// ---------------------------------------------------------------------------
// (0) Properties of the fixed inputs (premise for the known answer to hold at atol=0 — no GPU)
// ---------------------------------------------------------------------------

test("the canary's shape fills one whole tile of the production geometry × at least 2 K tiles", () => {
  const qkGeometry = defaultI8a8Geometry("attention_qk");
  const qk = buildQkCase(qkGeometry);
  assertEquals(qk.rows, i8a8TileM(qkGeometry), "M of ①QK differs from the tile edge");
  assertEquals(qk.cols, i8a8TileN(qkGeometry), "N of ①QK differs from the tile edge");
  assert(
    qk.depth / 4 / i8a8KPacks(qkGeometry) >= 2,
    `①QK has only ${qk.depth / 4 / i8a8KPacks(qkGeometry)} K tiles`,
  );
  const pvGeometry = defaultI8a8Geometry("attention_pv");
  const pv = buildPvCase(pvGeometry);
  assertEquals(pv.rows, i8a8TileM(pvGeometry), "M of ③PV differs from the tile edge");
  assertEquals(pv.depth, i8a8TileN(pvGeometry), "D of ③PV differs from the tile edge");
  assert(
    pv.cols / 4 / i8a8KPacks(pvGeometry) >= 2,
    `③PV has only ${pv.cols / 4 / i8a8KPacks(pvGeometry)} K tiles`,
  );
});

test("the fixed S of ③PV is exactly representable in f16 and does not move a bit through a round trip (premise of the s16 variant)", () => {
  const pv = buildPvCase(DEFAULT_CANARY_GEOMETRY.pv);
  const words = packScoresF16(pv.scores);
  assertEquals(words.length, pv.scores.length / 2, "word count of the repacking");
  for (let i = 0; i < pv.scores.length; i += 1) {
    const value = pv.scores[i];
    // ① exactly f16 as seen by Math.f16round (the reference for rounding)
    assertEquals(roundToF16(value), value, `S[${i}] is not exactly f16`);
    // ② the canary's repacker (exactF16Bits) and the reference decode round-trip
    const word = words[i >> 1];
    const half = (i & 1) === 0 ? word & 0xffff : word >>> 16;
    assertEquals(f16BitsToF32(half), value, `f16 round trip of S[${i}]`);
  }
});

test("qP of ③PV stays well away from the rounding boundaries (the GPU's exp error does not move a step)", () => {
  const pv = buildPvCase(DEFAULT_CANARY_GEOMETRY.pv);
  let worst = Number.POSITIVE_INFINITY;
  const seen = new Set<number>();
  for (let row = 0; row < pv.rows; row += 1) {
    const max = pv.stats[row * 2];
    for (let i = 0; i < pv.cols; i += 1) {
      const shifted = Math.fround(pv.scores[row * pv.cols + i] - max);
      const p = Math.fround(Math.fround(Math.exp(shifted)) * 127);
      seen.add(Math.round(p));
      worst = Math.min(worst, Math.abs(p - Math.floor(p) - 0.5));
    }
  }
  assert(worst > QUANT_MARGIN, `the worst qP margin is only ${worst} (pick the fixed S again)`);
  // The steps have not collapsed into one (if they did, a mixed-up index would not show in the
  // values = the comparison becomes a tautology)
  assert(seen.size >= 8, `qP has only ${seen.size} distinct steps`);
});

test("the ①QK known answer fits the visible range of f16 storage and has not collapsed to a constant", () => {
  const qk = buildQkCase(DEFAULT_CANARY_GEOMETRY.qk);
  let distinct = 0;
  const values = new Set<number>();
  for (const value of qk.expected) {
    assert(Number.isFinite(value), "the known answer S has non-finite values");
    assert(Math.abs(value) < 65504, `the known answer S=${value} exceeds the visible range of f16`);
    if (!values.has(value)) {
      values.add(value);
      distinct += 1;
    }
  }
  assert(
    distinct > 100,
    `the known answer S has only ${distinct} distinct values (the comparison could become a tautology)`,
  );
});

test("the ①QK known answer is farther than the sanity band from the f16 rounding boundaries (premise for the band to close with the s16 variant)", () => {
  // The s16 variant compares against "the known answer rounded to f16 on the host", so if the known
  // answer is near a rounding boundary, **a difference inside the band is amplified to one f16 step
  // (55 times the band) and both arms fall outside the band**. With this check holding for every
  // element, "an f32 difference inside the band ⇒ the same f16 step" becomes a theorem for the
  // fixed input.
  const qk = buildQkCase(DEFAULT_CANARY_GEOMETRY.qk);
  let worst = Number.POSITIVE_INFINITY;
  let at = -1;
  for (let i = 0; i < qk.expected.length; i += 1) {
    const ratio = f16BoundaryBandRatio(qk.expected[i]);
    if (ratio < worst) {
      worst = ratio;
      at = i;
    }
  }
  assert(
    worst > 1,
    `the f16 boundary margin of the known answer S[${at}] = ${qk.expected[at]} is only ${worst} times the band` +
      " (pick the fixed input again — §fixed inputs of src/gpu/attention-dp4a-canary.ts)",
  );
});

test("the fault-injection replacements affect only the targeted variant (closing off an injection that misses first)", () => {
  const geometry = defaultI8a8Geometry("attention_qk");
  const dp4aWgsl = attentionQkI8a8Wgsl(true, true, "f32", geometry);
  const emuWgsl = attentionQkI8a8Wgsl(true, false, "f32", geometry);
  assert(
    BREAK_DP4A(dp4aWgsl) !== dp4aWgsl,
    "the injection has no effect on the dp4a generated code",
  );
  assertEquals(
    BREAK_DP4A(emuWgsl),
    emuWgsl,
    "the dp4a-only injection also affects the emulated generated code",
  );
  assert(BREAK_BOTH(dp4aWgsl) !== dp4aWgsl, "the both-variant injection has no effect on dp4a");
  assert(
    BREAK_BOTH(emuWgsl) !== emuWgsl,
    "the both-variant injection has no effect on the emulation",
  );
});

test("the epilogue perturbation affects all 3 ③PV variants, equally for both arms (①QK passes through)", () => {
  const pvGeometry = defaultI8a8Geometry("attention_pv");
  for (const patch of [NUDGE_BOTH, SKEW_BOTH]) {
    for (const score of ["f32", "f16"] as const) {
      for (const v4 of score === "f16" ? [true] : [true, false]) {
        for (const dp4a of [true, false]) {
          const wgsl = attentionPvI8a8Wgsl(v4, dp4a, score, pvGeometry);
          assert(patch(wgsl) !== wgsl, `no effect on ③PV v4=${v4} score=${score} dp4a=${dp4a}`);
        }
      }
    }
    // ①QK's f16 storage risks crossing a rounding boundary, so being outside the perturbation's
    // target is made a check
    const qkWgsl = attentionQkI8a8Wgsl(true, true, "f16", defaultI8a8Geometry("attention_qk"));
    assertEquals(patch(qkWgsl), qkWgsl, "the epilogue perturbation also affects ①QK");
  }
});

test("the one-line rationale of the decision carries the branch and the maximum error (the body the warning uses as is)", () => {
  const decision: AttentionI8a8Decision = {
    dot: "dp4a",
    exact: false,
    branch: "band-both-identical",
    maxAbsError: 1.9073486328125e-6,
    maxBandRatio: 0.0067,
  };
  const line = formatAttentionI8a8Decision(decision);
  assert(line.includes("bit-identical"), line);
  assert(line.includes("'dp4a'"), line);
  assert(line.includes(String(decision.maxAbsError)), line);
  assert(line.includes(String(decision.maxBandRatio)), line);
});

// ---------------------------------------------------------------------------
// (0') The per-device memo (the slot of GpuContext — fake device, no GPU needed)
// ---------------------------------------------------------------------------

/** The decision the memo slot hands out (its content is irrelevant to the slot's behaviour). */
const memoDecision: AttentionI8a8Decision = {
  dot: "dp4a",
  exact: true,
  branch: "dp4a-exact",
  maxAbsError: 0,
  maxBandRatio: 0,
};

test("memo: a decision that failed with a transient OOM is not kept, and the next request runs the canary again", async () => {
  // The canary's temporary buffers are KB-sized, but when other Sessions fill VRAM close to the
  // ceiling they can hit a non-fatal OOM. Handing out the same failure after memory frees up would
  // make a8 Sessions impossible on the same GpuContext (it is not a property of the device, so
  // running again can change the conclusion).
  const gpu = fakeGpuContext(fakeDevice());
  const oom = new GpuOutOfMemoryError("canary temporary buffer: out of memory");
  await assertRejects(() => gpu[RUNTIME_INTERNAL].attentionI8a8Dot(() => Promise.reject(oom)));
  let reruns = 0;
  const decided = await gpu[RUNTIME_INTERNAL].attentionI8a8Dot(() => {
    reruns += 1;
    return Promise.resolve(memoDecision);
  });
  assertEquals(reruns, 1, "the canary did not run again on the request after the OOM");
  assertStrictEquals(decided, memoDecision);
  // A successful decision is kept for the device's whole lifetime, as before.
  let extra = 0;
  await gpu[RUNTIME_INTERNAL].attentionI8a8Dot(() => {
    extra += 1;
    return Promise.resolve(memoDecision);
  });
  assertEquals(extra, 0, "a successful decision was dropped from the memo");
});

test("memo: a decision that failed with a numeric mismatch or device loss is kept and not run again", async () => {
  // Both are properties of the device (or the device itself is unusable), so running again would
  // only pay one submit to reach the same conclusion.
  for (const failure of [
    new GpuFeatureError("both arms fell outside the sanity band"),
    new GpuDeviceLostError("device was lost while verifying the canary"),
  ]) {
    const gpu = fakeGpuContext(fakeDevice());
    await assertRejects(() =>
      gpu[RUNTIME_INTERNAL].attentionI8a8Dot(() => Promise.reject(failure)),
    );
    let reruns = 0;
    const again = await gpu[RUNTIME_INTERNAL]
      .attentionI8a8Dot(() => {
        reruns += 1;
        return Promise.resolve(memoDecision);
      })
      .then(
        () => undefined,
        (cause: unknown) => cause,
      );
    assertEquals(reruns, 0, `the canary ran again after ${failure.name}`);
    assertStrictEquals(again, failure, "hands out a result other than the kept failure");
  }
});

test("memo: concurrent requests share one run and hand out the same failure when it fails with OOM", async () => {
  const gpu = fakeGpuContext(fakeDevice());
  const pending = Promise.withResolvers<AttentionI8a8Decision>();
  let runs = 0;
  const run = (): Promise<AttentionI8a8Decision> => {
    runs += 1;
    return pending.promise;
  };
  const first = gpu[RUNTIME_INTERNAL].attentionI8a8Dot(run);
  const second = gpu[RUNTIME_INTERNAL].attentionI8a8Dot(run);
  assertStrictEquals(first, second, "concurrent requests received different Promises");
  assertEquals(runs, 1, "concurrent requests ran the canary twice");
  const oom = new GpuOutOfMemoryError("canary temporary buffer: out of memory");
  pending.reject(oom);
  for (const waiter of [first, second]) {
    assertStrictEquals(
      await waiter.then(
        () => undefined,
        (cause: unknown) => cause,
      ),
      oom,
    );
  }
  // A request after the failure runs a new one.
  const retried = await gpu[RUNTIME_INTERNAL].attentionI8a8Dot(() => {
    runs += 1;
    return Promise.resolve(memoDecision);
  });
  assertEquals(runs, 2);
  assertStrictEquals(retried, memoDecision);
});

// ---------------------------------------------------------------------------
// (1) A healthy device (this Linux / NVIDIA)
// ---------------------------------------------------------------------------

test.skipIf(!GPU_AVAILABLE)(
  "the plain decision picks dp4a, and the branch and the exact-match flag agree (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      const decision = await decideAttentionI8a8Dot(gpu, DEFAULT_CANARY_GEOMETRY);
      // The expectation splits by platform: the reference machine (Linux / NVIDIA) gives
      // dp4a-exact, Apple M2 gives band-both-identical through the 1 ULP difference of the shared
      // epilogue (the Metal section of known-issues). Either way **the chosen arm is dp4a** — that
      // is the subject of this test. The bit exactness of the reference machine itself is pinned
      // separately by the attention parity check (atol=0).
      assertEquals(decision.dot, "dp4a", "the plain decision does not pick dp4a");
      assert(
        decision.branch === "dp4a-exact" || decision.branch === "band-both-identical",
        `the plain decision took an unexpected branch: ${decision.branch}`,
      );
      assertEquals(
        decision.exact,
        decision.branch === "dp4a-exact",
        "the exact-match flag and the branch disagree",
      );
      if (decision.exact) {
        assertEquals(decision.maxAbsError, 0, "an exact match yet an error is present");
        // On the reference machine the emulated side also lands on the known answer (= the canary's
        // silence is not "both are off by the same amount"). Not applicable on band machines — the
        // identity of the two arms is already verified by the band-both-identical branch itself.
        // MUST: even when called directly, inside the per-device errorScope section lock (the
        // probe's doc).
        const emu = await gpu[RUNTIME_INTERNAL].withScopeLock(() =>
          probeAttentionI8a8Dot(gpu, "emu", DEFAULT_CANARY_GEOMETRY),
        );
        assertEquals(emu.mismatch, undefined, "the emulated variant missed the known answer");
        assert(emu.exact, "the emulated variant does not match at atol=0");
        assertEquals(emu.variants.length, 6, "the number of variants hit is not 6");
      }
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * A non-default i8a8 attention geometry (64×64 tiles — the default is ①QK 128×64 / ③PV 64×128, and
 * the apple-metal-3 profile applies ①QK tile32x64 / ③PV tile16x128; this differs from both). Used
 * as a detector that the canary hits "the generated code of the geometry the Session passed".
 */
const PROFILE_GEOMETRY: I8a8Geometry = { regM: 8, regN: 4, wgX: 16, wgY: 8, tileK: 16 };

test.skipIf(!GPU_AVAILABLE)(
  "given the profile's geometry, the canary hits the generated code of that geometry and reaches a decision (real GPU)",
  async () => {
    const geometry = { qk: PROFILE_GEOMETRY, pv: PROFILE_GEOMETRY };
    const gpu = await acquireGpu();
    try {
      // The decision itself: even with a non-default geometry the fixed input fills one whole tile
      // and lands on an in-band decision (if both arms were outside the band, decide would fail
      // with GpuFeatureError).
      const decision = await decideAttentionI8a8Dot(gpu, geometry);
      assert(
        decision.maxBandRatio <= 1,
        `band margin ${decision.maxBandRatio} is outside the band`,
      );
      assertEquals(
        decision.exact,
        decision.branch === "dp4a-exact" || decision.branch === "emu-exact",
        "the exact-match flag and the branch disagree",
      );

      // Detector: the pipeline keys the probe built are exactly the keys of **the geometry
      // passed**, different from the default geometry's keys (reverting to an implementation that
      // reads the default geometry directly returns the default keys here and fails).
      const probe = await gpu[RUNTIME_INTERNAL].withScopeLock(() =>
        probeAttentionI8a8Dot(gpu, "dp4a", geometry),
      );
      const shapes = [
        [true, "f32"],
        [false, "f32"],
        [true, "f16"],
      ] as const;
      const expected = [
        ...shapes.map(([v4, score]) => attentionQkI8a8Key(v4, true, score, PROFILE_GEOMETRY)),
        ...shapes.map(([v4, score]) => attentionPvI8a8Key(v4, true, score, PROFILE_GEOMETRY)),
      ];
      const defaults = [
        ...shapes.map(([v4, score]) =>
          attentionQkI8a8Key(v4, true, score, DEFAULT_CANARY_GEOMETRY.qk),
        ),
        ...shapes.map(([v4, score]) =>
          attentionPvI8a8Key(v4, true, score, DEFAULT_CANARY_GEOMETRY.pv),
        ),
      ];
      assertEquals(
        probe.variants.map((variant) => variant.key),
        expected,
        "the canary did not hit with the keys of the geometry passed",
      );
      for (const [index, key] of expected.entries()) {
        const v4 = shapes[index % shapes.length][0];
        assert(
          key.includes(i8a8GeometryKeyPart(PROFILE_GEOMETRY, v4)),
          `the geometry discriminator is missing: ${key}`,
        );
        assert(key !== defaults[index], `same as the default geometry's key: ${key}`);
      }
    } finally {
      gpu.destroy();
    }
  },
);

// ---------------------------------------------------------------------------
// (2) Fault injection (the branches of the mismatch path)
// ---------------------------------------------------------------------------

test.skipIf(!GPU_AVAILABLE)(
  "when only the dp4a variant misses the known answer, attention drops to emu (real GPU, fault injection)",
  async () => {
    const gpu = await acquireGpu();
    try {
      const decision = await decideAttentionI8a8Dot(gpu, DEFAULT_CANARY_GEOMETRY, BREAK_DP4A);
      assertEquals(decision.dot, "emu");
      // The reference machine gives emu-exact; on Apple M2 the emulated side is also off by 1 ULP,
      // so emu via band-single-arm (the subject — "discard the broken dp4a and take emu" — is the
      // same either way).
      assert(
        decision.branch === "emu-exact" || decision.branch === "band-single-arm",
        `unexpected branch: ${decision.branch}`,
      );
      assertEquals(
        decision.exact,
        decision.branch === "emu-exact",
        "the exact-match flag and the branch disagree",
      );
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "when both variants miss the known answer, it fails with GpuFeatureError (nothing to degrade to, real GPU, fault injection)",
  async () => {
    const gpu = await acquireGpu();
    try {
      const error = await assertRejects(
        () => decideAttentionI8a8Dot(gpu, DEFAULT_CANARY_GEOMETRY, BREAK_BOTH),
        GpuFeatureError,
      );
      // The diagnostic goes down to "which arm missed, in which generated code, at which element"
      assert(error.message.includes("attention_qk:v3:i8a8:"), error.message);
      assert(error.message.includes("dp4a:"), error.message);
      assert(error.message.includes("emu:"), error.message);
      // It reads as "it missed after already allowing the band" (v2's order of decision shows in
      // the diagnostic)
      assert(error.message.includes("sanity band"), error.message);
    } finally {
      gpu.destroy();
    }
  },
);

// ---------------------------------------------------------------------------
// (2') Fault injection (the band branches of decision rule v2 — paths the Apple M2 data required)
// ---------------------------------------------------------------------------

test.skipIf(!GPU_AVAILABLE)(
  "when both arms stay inside the band under the same tiny epilogue perturbation, dp4a passes as is (real GPU, fault injection)",
  async () => {
    const gpu = await acquireGpu();
    try {
      const decision = await decideAttentionI8a8Dot(gpu, DEFAULT_CANARY_GEOMETRY, NUDGE_BOTH);
      // The arms are bit-identical to each other = the variant choice is irrelevant to the numbers
      // (ADR 0058 decision 2) → take the default dp4a
      assertEquals(decision.branch, "band-both-identical");
      assertEquals(decision.dot, "dp4a");
      // MUST: the fact that it was not an exact match stays in the return value (so the caller can
      // warn)
      assert(!decision.exact, "passed inside the band yet the exact-match flag is set");
      assert(decision.maxAbsError > 0, "the perturbation missed (error 0)");
      assert(decision.maxBandRatio <= 1, `band margin ${decision.maxBandRatio} exceeds the band`);
      // The perturbation is relative ~1.2e-7 = 2 orders of magnitude below the band (measured: it
      // fits without loosening the band width)
      assert(
        decision.maxBandRatio < 0.1,
        `band margin ${decision.maxBandRatio} is larger than expected`,
      );
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "even with identical arms, a perturbation outside the band gives GpuFeatureError (bit identity is no excuse, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      const error = await assertRejects(
        () => decideAttentionI8a8Dot(gpu, DEFAULT_CANARY_GEOMETRY, SKEW_BOTH),
        GpuFeatureError,
      );
      assert(error.message.includes("attention_pv:v3:i8a8:"), error.message);
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "when only one arm is inside the band, that arm is taken (what M2 + a dp4a fault looks like, real GPU, fault injection)",
  async () => {
    const gpu = await acquireGpu();
    try {
      const decision = await decideAttentionI8a8Dot(
        gpu,
        DEFAULT_CANARY_GEOMETRY,
        NUDGE_BOTH_BREAK_DP4A,
      );
      assertEquals(decision.branch, "band-single-arm");
      assertEquals(decision.dot, "emu");
      assert(!decision.exact, "passed inside the band yet the exact-match flag is set");
      assert(
        decision.maxBandRatio <= 1,
        `the band margin ${decision.maxBandRatio} of the chosen arm is outside the band`,
      );
    } finally {
      gpu.destroy();
    }
  },
);

// ---------------------------------------------------------------------------
// (3) Use from Sessions (memoized once / the slot is reflected / a non-a8 Session does not pay)
// ---------------------------------------------------------------------------

/**
 * A decision placed directly in the memo slot (to see the slot's own behaviour without running the
 * canary).
 */
const seatDecision = (dot: I8a8Dot): AttentionI8a8Decision => ({
  dot,
  exact: true,
  branch: dot === "dp4a" ? "dp4a-exact" : "emu-exact",
  maxAbsError: 0,
  maxBandRatio: 0,
});

const QUERY = (i: number): number => (((i * 3) % 29) - 14) * 0.3717 + 0.0419;
const KEY = (i: number): number => (((i * 3) % 41) - 20) * 0.2917 - 0.0173;
const VALUE = (i: number): number => (((i * 5) % 17) - 8) * 0.3119;

/** Runs a Session of one fused attention node once and returns that run's census. */
const runAttention = async (
  gpu: GpuContext,
  options: SessionOptions,
): Promise<SessionDiagnostics["lastRunPipelines"]> => {
  const [b, h, m, n, d] = [1, 2, 20, 16, 8];
  const q = fill([b, h, m, d], QUERY);
  const k = fill([b, h, n, d], KEY);
  const v = fill([b, h, n, d], VALUE);
  const graph = singleOpDeclaration("attention", [q.shape, k.shape, v.shape], [[b, h, m, d]], {
    attrs: { scale: Math.fround(Math.sqrt(1 / Math.sqrt(d))) },
  });
  const session = await createSessionFromContainer(
    gpu,
    await openGraphModel(graph),
    GRAPH_NAME,
    options,
  );
  try {
    await session.run({ x0: q, x1: k, x2: v });
    return session.diagnostics().lastRunPipelines;
  } finally {
    await session.dispose();
  }
};

test.skipIf(!GPU_AVAILABLE)(
  "the canary decides once per device, and a8 Sessions read that slot (real GPU, fault injection)",
  async () => {
    const gpu = await acquireGpu();
    try {
      // The real machine is healthy, so the injected decision is burned into the memo **first**
      // (otherwise it would be dp4a). That the Session reads this slot is observable only through
      // the key becoming `:dp4aEmu`.
      const seeded = await gpu[RUNTIME_INTERNAL].attentionI8a8Dot(() =>
        decideAttentionI8a8Dot(gpu, DEFAULT_CANARY_GEOMETRY, BREAK_DP4A),
      );
      assertEquals(seeded.dot, "emu");

      // That the slot reaches the Session (visible only in keys) is covered by the census below
      for (const _pass of [1, 2]) {
        await runAttention(gpu, { attentionCompute: "a8" });
      }

      // Even after running a 2nd one the canary runs only once (the memo works through the Promise)
      let extraRuns = 0;
      const memoized = await gpu[RUNTIME_INTERNAL].attentionI8a8Dot(() => {
        extraRuns += 1;
        return Promise.resolve(seatDecision("dp4a"));
      });
      assertEquals(extraRuns, 0, "the canary runs again per Session");
      assertEquals(memoized.dot, "emu", "the memo does not hand out the first decision");
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * **census** (ADR 0058 decision 4). That the injected slot (a decision tipped to emu with
 * `BREAK_DP4A`) reaches the Session's variant choice is **observable only through keys** (dp4a and
 * emu return the same integers, so the values do not change by a single bit).
 *
 * MUST: the source is `lastRunPipelines`, which does not depend on timing (runs on machines without
 * timing too). An empty census FAILs unconditionally (guarding with "if keys is empty, go to the
 * next shape" turns green without a single check of the variant choice running — `requireCensus` in
 * tests/helpers/pipeline-census.ts). `acquireGpu` creates a new device on every call, so the memo
 * slot is private to this test too.
 */
test.skipIf(!GPU_AVAILABLE)(
  "the injected slot (emu) reaches both ①QK and ③PV of an a8 Session (real GPU, fault injection)",
  async () => {
    const gpu = await acquireGpu();
    const profile = deviceGeometryProfile(gpu);
    try {
      const seeded = await gpu[RUNTIME_INTERNAL].attentionI8a8Dot(() =>
        decideAttentionI8a8Dot(gpu, DEFAULT_CANARY_GEOMETRY, BREAK_DP4A),
      );
      assertEquals(seeded.dot, "emu");
      for (const pass of [1, 2]) {
        const census = await runAttention(gpu, { attentionCompute: "a8" });
        const qkV4 = attentionQkI8a8UsesVec4(16);
        const pvV4 = attentionPvI8a8UsesVec4(8);
        assertPipelineCensus(
          census,
          [
            {
              label: "emu variant of ①QK",
              match: attentionQkI8a8Key(qkV4, false, "f32", profile.i8a8.attentionQk),
              atLeast: 1,
            },
            {
              label: "dp4a variant of ①QK",
              match: attentionQkI8a8Key(qkV4, true, "f32", profile.i8a8.attentionQk),
              count: 0,
            },
            {
              label: "emu variant of ③PV",
              match: attentionPvI8a8Key(pvV4, false, "f32", profile.i8a8.attentionPv),
              atLeast: 1,
            },
            {
              label: "dp4a variant of ③PV",
              match: attentionPvI8a8Key(pvV4, true, "f32", profile.i8a8.attentionPv),
              count: 0,
            },
          ],
          `run ${pass}`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "a Session whose attentionCompute is not 'a8' pays not a single canary dispatch (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      await runAttention(gpu, {});
      await runAttention(gpu, { attentionCompute: "f16" }).catch(() => undefined);
      // If the slot is still empty, the instance passed here is the first to run (= the 2 above did
      // not pay)
      let ran = 0;
      const verdict = await gpu[RUNTIME_INTERNAL].attentionI8a8Dot(() => {
        ran += 1;
        return Promise.resolve(seatDecision("emu"));
      });
      assertEquals(ran, 1, "a non-a8 Session runs the canary");
      assertEquals(verdict.dot, "emu");
    } finally {
      gpu.destroy();
    }
  },
);
