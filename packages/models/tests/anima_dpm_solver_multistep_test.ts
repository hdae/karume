// Behaviour tests of DPM++ 2M (`src/generation/dpm-solver-multistep.ts`). Pure functions that need
// neither a GPU nor assets.
//
// The expected values come from `tools/export-recipes/anima/dpmsolver_ref.py` — it plugs karume's
// sigma ladder into the **real class** `DPMSolverMultistepScheduler` of diffusers 0.39.0, feeds it
// a synthetic model output (fixed seed) and prints a golden, which is inlined here as is. To
// regenerate:
//
//     cd tools/export-recipes
//     uv run --no-sync python -m anima.dpmsolver_ref --steps 8 --elements 4   # SERIES_8
//     uv run --no-sync python -m anima.dpmsolver_ref --steps 2 --elements 3   # SERIES_2
//
// The source of truth for how the tolerance ({@link TOLERANCE}) is derived is the module's doc (the
// MUST section "not bit-identical to the reference") — the origin of the number is not written
// twice here.

import { test } from "vite-plus/test";
import { assert, assertAlmostEquals, assertEquals, assertThrows } from "@std/assert";
import { dpmSolverMultistepStep, needsUncond } from "../src/generation/dpm-solver-multistep.ts";

/**
 * The parity tolerance. Derived from the measurements in the MUST section of the module doc (about
 * 11× the worst fixture measurement 3.576e-07, about 3.4× the worst sweep measurement 1.192e-06).
 * The smallest difference an implementation bug produces is 1.36e-01 (= 4.5 orders of magnitude
 * above this threshold), so allowing rounding differences keeps the detection power.
 */
const TOLERANCE = 4e-6;

/** The same value as `pipelineConfig.scheduler.shift` in `models/karume-anima/karume.json`. */
const SHIFT = 3;

type Series = {
  readonly sigmas: readonly number[];
  readonly initial: readonly number[];
  readonly steps: readonly {
    readonly velocity: readonly number[];
    readonly x0: readonly number[];
    readonly sample: readonly number[];
  }[];
};

/**
 * `--steps 8 --elements 4` (default seed). One series walks every branch: 1st order → 2nd order ×6
 * → 1st order.
 */
const SERIES_8: Series = {
  sigmas: [
    1, 0.9545454382896423, 0.8999999761581421, 0.8333333134651184, 0.75, 0.6428571343421936, 0.5,
    0.30000001192092896, 0,
  ],
  initial: [0.6157209873199463, -1.2011234760284424, 0.2345673143863678, -0.9813034534454346],
  steps: [
    {
      velocity: [0.1726374477148056, -1.3876512050628662, -0.6250362992286682, 1.5988130569458008],
      x0: [0.4430835247039795, 0.18652772903442383, 0.8596036434173584, -2.5801165103912354],
      sample: [0.6078738570213318, -1.1380484104156494, 0.26297807693481445, -1.0539767742156982],
    },
    {
      velocity: [-0.5767762660980225, -0.8768693208694458, 0.5082118511199951, -0.7355096936225891],
      x0: [1.158432960510254, -0.3010367751121521, -0.22213321924209595, -0.35189932584762573],
      sample: [0.6393344402313232, -1.090219259262085, 0.23525743186473846, -1.0138580799102783],
    },
    {
      velocity: [-1.2072668075561523, 1.2542095184326172, 1.3473405838012695, 0.2711418569087982],
      x0: [1.7258745431900024, -2.219007968902588, -0.9773491024971008, -1.2578856945037842],
      sample: [0.734398365020752, -1.2231121063232422, 0.12603075802326202, -1.0552119016647339],
    },
    {
      velocity: [1.2115410566329956, 1.3144892454147339, 0.2535824775695801, 1.3913758993148804],
      x0: [-0.275219202041626, -2.3185198307037354, -0.08528797328472137, -2.214691638946533],
      sample: [0.5464824438095093, -1.3369770050048828, 0.14366188645362854, -1.2127362489700317],
    },
    {
      velocity: [0.402450829744339, -0.6969248056411743, -0.20336712896823883, 0.4934312701225281],
      x0: [0.24464431405067444, -0.8142833709716797, 0.29618722200393677, -1.5828096866607666],
      sample: [0.5404958128929138, -1.1548610925674438, 0.19269946217536926, -1.2204694747924805],
    },
    {
      velocity: [
        -1.5521337985992432, -0.8093506097793579, -0.22987085580825806, -1.7819559574127197,
      ],
      x0: [1.538296103477478, -0.6345642805099487, 0.34047359228134155, -0.07492637634277344],
      sample: [0.9276240468025208, -1.016262412071228, 0.2312002331018448, -0.7731197476387024],
    },
    {
      velocity: [
        0.4386770725250244, -0.9429063200950623, -0.23527345061302185, -0.5406302809715271,
      ],
      x0: [0.7082855105400085, -0.5448092222213745, 0.3488369584083557, -0.5028046369552612],
      sample: [0.6005957126617432, -0.8018047213554382, 0.2806660830974579, -0.7883514761924744],
    },
    {
      velocity: [-0.5566991567611694, 0.9431769251823425, 1.243248701095581, 0.281678169965744],
      x0: [0.7676054835319519, -1.0847578048706055, -0.09230855107307434, -0.8728549480438232],
      sample: [0.7676054835319519, -1.0847578048706055, -0.09230855107307434, -0.8728549480438232],
    },
  ],
};

/**
 * `--steps 2 --elements 3` (default seed). The minimum ladder length — both steps fall to 1st
 * order.
 */
const SERIES_2: Series = {
  sigmas: [1, 0.75, 0],
  initial: [0.6157209873199463, -1.2011234760284424, 0.2345673143863678],
  steps: [
    {
      velocity: [-0.9813034534454346, 0.1726374477148056, -1.3876512050628662],
      x0: [1.5970244407653809, -1.3737609386444092, 1.6222184896469116],
      sample: [0.8610468506813049, -1.244282841682434, 0.5814801454544067],
    },
    {
      velocity: [-0.6250362992286682, 1.5988130569458008, -0.5767762660980225],
      x0: [1.3298240900039673, -2.443392753601074, 1.0140624046325684],
      sample: [1.3298240900039673, -2.443392753601074, 1.0140624046325684],
    },
  ],
};

/**
 * Runs the series to the end and returns the **maximum absolute difference** between each step's
 * `sample` / `x0` and the golden.
 *
 * `velocityOrder` is the fault-injection point for the step order — it feeds the velocities in the
 * given order (the default is the plain 0,1,2,… order). **The state (x and the previous x0) is
 * chained plainly**, so an injected error propagates as is to the following steps.
 */
const worstDeviation = (series: Series, velocityOrder?: readonly number[]): number => {
  const sigmas = Float32Array.from(series.sigmas);
  let sample = Float32Array.from(series.initial);
  let previousX0: Float32Array<ArrayBuffer> | undefined = undefined;
  let worst = 0;
  for (let index = 0; index < series.steps.length; index += 1) {
    const fed = series.steps[velocityOrder === undefined ? index : velocityOrder[index]];
    const update = dpmSolverMultistepStep({
      sample,
      cond: Float32Array.from(fed.velocity),
      uncond: undefined,
      guidance: 1,
      sigmas,
      index,
      previousX0,
    });
    const expected = series.steps[index];
    for (let element = 0; element < expected.sample.length; element += 1) {
      worst = Math.max(worst, Math.abs(update.sample[element] - expected.sample[element]));
      worst = Math.max(worst, Math.abs(update.x0[element] - expected.x0[element]));
    }
    sample = update.sample;
    previousX0 = update.x0;
  }
  return worst;
};

test("dpmSolverMultistepStep: parity with the rig golden (steps=8, every branch of 1st/2nd/1st order)", () => {
  // It compares step by step, so "only the last one matches" does not pass (the intermediate x0 is
  // checked too).
  const sigmas = Float32Array.from(SERIES_8.sigmas);
  let sample = Float32Array.from(SERIES_8.initial);
  let previousX0: Float32Array<ArrayBuffer> | undefined = undefined;
  for (let index = 0; index < SERIES_8.steps.length; index += 1) {
    const expected = SERIES_8.steps[index];
    const update = dpmSolverMultistepStep({
      sample,
      cond: Float32Array.from(expected.velocity),
      uncond: undefined,
      guidance: 1,
      sigmas,
      index,
      previousX0,
    });
    for (let element = 0; element < expected.sample.length; element += 1) {
      assertAlmostEquals(
        update.x0[element],
        expected.x0[element],
        TOLERANCE,
        `x0 diverged from the reference (step=${index} element=${element})`,
      );
      assertAlmostEquals(
        update.sample[element],
        expected.sample[element],
        TOLERANCE,
        `the state diverged from the reference (step=${index} element=${element})`,
      );
    }
    sample = update.sample;
    previousX0 = update.x0;
  }
});

test("dpmSolverMultistepStep: the final step returns x0 as is (1st-order fallback at the terminal σ=0)", () => {
  // A consequence of `final_sigmas_type="zero"`. The coefficients become (0, −1), so it matches bit
  // for bit.
  const sigmas = Float32Array.from(SERIES_8.sigmas);
  const last = SERIES_8.steps.length - 1;
  const update = dpmSolverMultistepStep({
    sample: Float32Array.from(SERIES_8.steps[last - 1].sample),
    cond: Float32Array.from(SERIES_8.steps[last].velocity),
    uncond: undefined,
    guidance: 1,
    sigmas,
    index: last,
    previousX0: Float32Array.from(SERIES_8.steps[last - 1].x0),
  });
  assertEquals([...update.sample], [...update.x0]);
});

test("dpmSolverMultistepStep: boundary — matches the reference even at the minimum ladder length (steps=2)", () => {
  // With only two steps the 2nd-order branch is never entered (step 0 and the final step are all
  // there is). If this path broke, a test that looks only at the 2nd-order side would pass anyway.
  assert(
    worstDeviation(SERIES_2) <= TOLERANCE,
    `the steps=2 series diverged from the reference (max difference ${worstDeviation(SERIES_2)})`,
  );
});

test("dpmSolverMultistepStep: boundary — guidance=1 uses cond as the velocity as is", () => {
  // With a one-step ladder σ0=1, σ1=0 the update is exactly x0 = x − 1·v.
  const sigmas = Float32Array.from([1, 0]);
  const update = dpmSolverMultistepStep({
    sample: Float32Array.from([1, 2, 3]),
    cond: Float32Array.from([0.25, -0.5, 0.125]),
    uncond: undefined,
    guidance: 1,
    sigmas,
    index: 0,
    previousX0: undefined,
  });
  assertEquals([...update.x0], [0.75, 2.5, 2.875]);
  assertEquals([...update.sample], [0.75, 2.5, 2.875]);
});

test("dpmSolverMultistepStep: guidance≠1 uses uncond + scale·(cond − uncond) as the velocity", () => {
  const update = dpmSolverMultistepStep({
    sample: Float32Array.from([0]),
    cond: Float32Array.from([1]),
    uncond: Float32Array.from([0.5]),
    guidance: 4,
    sigmas: Float32Array.from([1, 0]),
    index: 0,
    previousX0: undefined,
  });
  // v = 0.5 + 4·(1 − 0.5) = 2.5 → x0 = 0 − 1·2.5 = −2.5
  assertEquals([...update.x0], [-2.5]);
});

test("needsUncond: only CFG=1 needs no uncond", () => {
  assertEquals(needsUncond(1), false);
  assertEquals(needsUncond(4), true);
  assertEquals(needsUncond(0), true);
});

test("dpmSolverMultistepStep: rejects a presence of uncond that disagrees with needsUncond", () => {
  const common = {
    sample: Float32Array.from([0, 0]),
    cond: Float32Array.from([1, 1]),
    sigmas: Float32Array.from([1, 0]),
    index: 0,
    previousX0: undefined,
  };
  assertThrows(
    () => dpmSolverMultistepStep({ ...common, uncond: undefined, guidance: 4 }),
    Error,
    "needs the model output of the uncond side",
  );
  assertThrows(
    () => dpmSolverMultistepStep({ ...common, uncond: Float32Array.from([2, 2]), guidance: 1 }),
    Error,
    "path is supposed to skip computing the uncond branch",
  );
});

test("dpmSolverMultistepStep: rejects a presence of the previous x0 that disagrees with whether this is step 0", () => {
  // Forbids silently falling to 1st order when the state hand-off is broken (1st and 2nd order give
  // different images).
  const sigmas = Float32Array.from(SERIES_8.sigmas);
  const common = {
    sample: Float32Array.from([0, 0, 0, 0]),
    cond: Float32Array.from([1, 1, 1, 1]),
    uncond: undefined,
    guidance: 1,
    sigmas,
  };
  assertThrows(
    () => dpmSolverMultistepStep({ ...common, index: 3, previousX0: undefined }),
    Error,
    "the presence of the previous x0 disagree",
  );
  assertThrows(
    () =>
      dpmSolverMultistepStep({
        ...common,
        index: 0,
        previousX0: Float32Array.from([0, 0, 0, 0]),
      }),
    Error,
    "the presence of the previous x0 disagree",
  );
});

test("dpmSolverMultistepStep: the accepted set of ladders and step indices", () => {
  const sample = Float32Array.from([0]);
  const cond = Float32Array.from([1]);
  const common = { sample, cond, uncond: undefined, guidance: 1, previousX0: undefined };
  assertThrows(
    () => dpmSolverMultistepStep({ ...common, sigmas: Float32Array.from([0]), index: 0 }),
    RangeError,
    "there is not a single step",
  );
  // A ladder whose end is not 0 cannot use the 1st-order fallback at the final step (it assumes
  // `final_sigmas_type="zero"`).
  assertThrows(
    () => dpmSolverMultistepStep({ ...common, sigmas: Float32Array.from([1, 0.5]), index: 0 }),
    RangeError,
    "is not 0",
  );
  assertThrows(
    () => dpmSolverMultistepStep({ ...common, sigmas: Float32Array.from([1, 0]), index: 1 }),
    RangeError,
    "is not an integer in",
  );
  assertThrows(
    () => dpmSolverMultistepStep({ ...common, sigmas: Float32Array.from([1, 0]), index: -1 }),
    RangeError,
    "is not an integer in",
  );
});

test("dpmSolverMultistepStep: fault injection — swapping the step order far exceeds the parity threshold", () => {
  // Proof of detection power. Even with the threshold (4e-6) absorbing rounding differences,
  // swapping a single pair in the order makes the maximum absolute difference jump to 2.03e+00
  // (measured), red with a margin of 5.7 orders of magnitude.
  const flipped = [0, 2, 1, 3, 4, 5, 6, 7];
  const deviation = worstDeviation(SERIES_8, flipped);
  assert(
    deviation > TOLERANCE * 1e4,
    `swapping the step order was not detected (max difference ${deviation})`,
  );
});

test("dpmSolverMultistepStep: fault injection — a wrong shift far exceeds the parity threshold", () => {
  // Confirms that a wrong ladder itself (a different shift in `sigmaSchedule`) breaks at the same
  // threshold. Every coefficient inside is derived from the ladder, so if the ladder shifts by one
  // rung the coefficients of every step shift. Measured (shift 3 → 4): maximum absolute difference
  // 1.568e-01 = 4.6 orders of magnitude above the threshold.
  const wrongLadder: Series = {
    ...SERIES_8,
    sigmas: sigmaLadder(SERIES_8.steps.length, SHIFT + 1),
  };
  const deviation = worstDeviation(wrongLadder);
  assert(
    deviation > TOLERANCE * 1e4,
    `a wrong shift was not detected (max difference ${deviation})`,
  );
});

/**
 * Builds a ladder with another shift for fault injection (a copy of `sigmaSchedule` — used only
 * inside the test).
 *
 * MUST: do not import it from the family side (`src/anima/sampler.ts`). The module under test is
 * family-independent, and if the test created that dependency direction first, a violation of the
 * rule could no longer be detected.
 */
function sigmaLadder(steps: number, shift: number): number[] {
  const f32 = Math.fround;
  const stop = 1 / steps;
  const delta = (stop - 1) / (steps - 1);
  const ladder: number[] = [];
  for (let index = 0; index < steps; index += 1) {
    const sigma = f32(index === steps - 1 ? stop : index * delta + 1);
    ladder.push(f32(f32(shift * sigma) / f32(f32((shift - 1) * sigma) + 1)));
  }
  ladder.push(0);
  return ladder;
}

test("dpmSolverMultistepStep: a factor not representable in f32 is rounded to f32 before multiplying (matches torch to the last digit)", () => {
  // The same pair as `cfgEulerStep` (combined f32 tensors on the CPU with torch 2.13.0 =
  // −0.03790003061294556). x0 = 0 − 1·v, so the result is the combined value with its sign flipped.
  // Multiplying in f64 gives 0.03789997100830078.
  const update = dpmSolverMultistepStep({
    sample: Float32Array.from([0]),
    cond: Float32Array.from([0.011]),
    uncond: Float32Array.from([0.5]),
    guidance: 1.1,
    sigmas: Float32Array.from([1, 0.5, 0]),
    index: 0,
    previousX0: undefined,
  });
  assertEquals([...update.x0], [0.03790003061294556]);
});
