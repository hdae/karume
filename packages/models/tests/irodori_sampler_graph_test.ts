/**
 * The check (real GPU) of the 2 graphs that keep Irodori's DiT loop resident on the GPU
 * (`host/sampler-graph.ts`).
 *
 * Only two things are checked:
 *
 * 1. **Numerics** — running the host-built combine / euler the way the loop stacks them matches the
 *    TS source of truth (`combineCfg` / `eulerStep`) **bit for bit**. If this breaks, the resident
 *    loop of `pipeline.ts` breaks too (= the WAV sha256 check fails), so this check isolates the
 *    cause with the smallest material.
 * 2. **Structure** — stacking as many steps as needed still produces only the batch's one
 *    `onSubmittedWorkDone`. Looking only at values, it would stay green even if the internals fell
 *    back to the run path (a fence per forward).
 *
 * No `dit` is needed (the velocity field is substituted by host-made arrays written into resident
 * tensors), so it closes without reading the 3.3GB distribution. That the whole loop actually runs
 * in one batch is pinned on the E2E side (the WAV sha256 check).
 */

import { test } from "vite-plus/test";
import { assert, assertEquals, assertThrows } from "@std/assert";
import {
  acquireGpu,
  createSessionFromContainer,
  type GpuContext,
  type IrDeclaration,
  IrError,
  openMemoryContainer,
  parseIrDeclarationValue,
  type Session,
  type Tensor,
} from "@karume/runtime";
import {
  COMBINE_INPUTS,
  COMBINE_OUTPUT,
  combineGraph,
  EULER_INPUTS,
  EULER_OUTPUT,
  eulerGraph,
} from "../src/irodori/host/sampler-graph.ts";
import { combineCfg, eulerStep } from "../src/irodori/host/sampler.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

/**
 * Smaller than the real shape (about the v4-small no-ref case), but the width is the distribution's
 * 32.
 */
const FRAMES = 12;
const LATENT_DIM = 32;
const COUNT = FRAMES * LATENT_DIM;
const BYTES = COUNT * 4;
/** The effective CFG strengths (text / speaker / caption of the v4-small distribution). */
const SCALES = [3, 5, 3] as const;
const STEPS = 4;

/** A deterministic pseudo velocity field (no random numbers — a failure would not reproduce). */
const fill = (seed: number): Float32Array<ArrayBuffer> => {
  const out = new Float32Array(COUNT);
  let state = seed;
  for (let index = 0; index < COUNT; index += 1) {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    out[index] = ((state % 20001) - 10000) / 3137;
  }
  return out;
};

const scalar = (value: number): Tensor => ({
  dtype: "f32",
  shape: [1],
  data: Float32Array.of(value),
});

/** Bit-pattern comparison (no tolerance for mixed-up rounding). */
const bits = (data: Float32Array | ArrayBuffer): readonly number[] =>
  Array.from(
    data instanceof ArrayBuffer
      ? new Uint32Array(data)
      : new Uint32Array(data.buffer, data.byteOffset, data.length),
  );

/**
 * Counts calls to `queue.onSubmittedWorkDone` (a mechanical check of the fence count). To avoid
 * adding an observation point inside the implementation, the test wraps the queue's surface
 * (the same technique as `packages/runtime/tests/gpu_resident_batch_test.ts`).
 */
const countFences = (gpu: GpuContext): { readonly count: () => number; restore: () => void } => {
  const queue = gpu.device.queue;
  const original = queue.onSubmittedWorkDone.bind(queue);
  let fences = 0;
  Object.defineProperty(queue, "onSubmittedWorkDone", {
    configurable: true,
    writable: true,
    value: () => {
      fences += 1;
      return original();
    },
  });
  return {
    count: () => fences,
    restore: () => {
      Object.defineProperty(queue, "onSubmittedWorkDone", {
        configurable: true,
        writable: true,
        value: original,
      });
    },
  };
};

/**
 * Puts a host-built declaration (0 weights) into an in-memory container and makes it a Session
 * (built the same way as `openHostGraph` of `dit-loop.ts`).
 */
const hostSession = (gpu: GpuContext, name: string, declaration: IrDeclaration): Promise<Session> =>
  createSessionFromContainer(
    gpu,
    openMemoryContainer({ graphs: { [name]: declaration }, tensors: {} }),
    name,
  );

/**
 * The material of one step (the cond velocity field / the velocity field per variant / the step
 * size).
 */
type Step = {
  readonly cond: Float32Array<ArrayBuffer>;
  readonly variants: readonly Float32Array<ArrayBuffer>[];
  readonly deltaT: number;
};

/** Builds the step sequence (`variantCount` = 0 is a step outside the CFG window). */
const plan = (variantCount: number): readonly Step[] =>
  Array.from({ length: STEPS }, (_, step) => ({
    cond: fill(1000 + step),
    variants: Array.from({ length: variantCount }, (_, k) => fill(2000 + step * 10 + k)),
    deltaT: Math.fround(-0.999 / STEPS),
  }));

/** Runs the loop with the TS source of truth (what this file compares against). */
const hostLoop = (
  initial: Float32Array<ArrayBuffer>,
  steps: readonly Step[],
): Float32Array<ArrayBuffer> => {
  let x = initial;
  for (const step of steps) {
    const variants = step.variants.map((velocity, k) => ({ scale: SCALES[k], velocity }));
    x = eulerStep(x, combineCfg(step.cond, variants), step.deltaT);
  }
  return x;
};

/**
 * Runs the loop with the resident graphs (**stacked the same way** as `runDitLoopResident` of
 * `dit-loop.ts`). Only the `dit` forward is replaced by a `write` of a host array.
 */
const residentLoop = async (
  gpu: GpuContext,
  initial: Float32Array<ArrayBuffer>,
  steps: readonly Step[],
): Promise<{ readonly x: ArrayBuffer; readonly fences: number }> => {
  const combine = await hostSession(gpu, "combine", combineGraph(FRAMES, LATENT_DIM));
  const euler = await hostSession(gpu, "euler", eulerGraph(FRAMES, LATENT_DIM));
  const xT = await gpu.createResident(BYTES, "x_t");
  const vCond = await gpu.createResident(BYTES, "v_cond");
  const vVariant = await gpu.createResident(BYTES, "v_variant");
  const accumulator = await gpu.createResident(BYTES, "cfg_acc");
  const fences = countFences(gpu);
  try {
    xT.write(initial);
    const before = fences.count();
    const batch = await gpu.beginBatch();
    try {
      for (const step of steps) {
        vCond.write(step.cond);
        for (let k = 0; k < step.variants.length; k += 1) {
          vVariant.write(step.variants[k]);
          await combine.enqueue(
            {
              [COMBINE_INPUTS.accumulator]: k === 0 ? vCond : accumulator,
              [COMBINE_INPUTS.cond]: vCond,
              [COMBINE_INPUTS.variant]: vVariant,
              [COMBINE_INPUTS.scale]: scalar(SCALES[k]),
            },
            { batch, copyOutputs: { [COMBINE_OUTPUT]: accumulator } },
          );
        }
        await euler.enqueue(
          {
            [EULER_INPUTS.x]: xT,
            [EULER_INPUTS.velocity]: step.variants.length > 0 ? accumulator : vCond,
            [EULER_INPUTS.deltaT]: scalar(step.deltaT),
          },
          { batch, copyOutputs: { [EULER_OUTPUT]: xT } },
        );
      }
      assertEquals(fences.count() - before, 0, "enqueue sets no fence at all");
    } finally {
      await batch.finish();
    }
    return { x: await xT.read(), fences: fences.count() - before };
  } finally {
    fences.restore();
    await combine.dispose();
    await euler.dispose();
    xT.dispose();
    vCond.dispose();
    vVariant.dispose();
    accumulator.dispose();
  }
};

test("a host-built declaration passes the graph-level rules (a forward reference fails at the declaration stage)", () => {
  const declaration = eulerGraph(FRAMES, LATENT_DIM);
  // Negative control: the built declaration itself passes (= the detector is not always red).
  assertEquals(parseIrDeclarationValue({ ...declaration }), declaration);
  // Swapping the node order makes `step` a forward reference. Neither the merge (declaration ×
  // supply) nor `prepareContainer` (capability + op contracts) looks at node order, so only this
  // check can catch it.
  assertThrows(
    () => parseIrDeclarationValue({ ...declaration, nodes: [...declaration.nodes].reverse() }),
    IrError,
    "forward reference",
  );
});

test.skipIf(!GPU_AVAILABLE)(
  "the resident CFG + Euler loop matches the TS source of truth bit for bit, with the batch's one fence (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      const initial = fill(7);
      const steps = plan(SCALES.length);
      // Guarding against a vacuous check: if the expectation stayed at the initial value, it would
      // be green even if the loop never had an effect.
      const expected = hostLoop(initial, steps);
      assert(
        JSON.stringify(bits(expected)) !== JSON.stringify(bits(initial)),
        "the expectation equals the initial latent (the detector would fire on nothing)",
      );

      const actual = await residentLoop(gpu, initial, steps);
      assertEquals(
        bits(actual.x),
        bits(expected),
        "the resident loop's latent differs from the TS source of truth",
      );
      assertEquals(actual.fences, 1, "onSubmittedWorkDone happens once for the whole loop");
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "a step outside the CFG window (0 variants) passes cond to Euler as is (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      const initial = fill(11);
      // The source-of-truth side goes through `combineCfg(cond, [])` (= a copy of cond). The
      // resident side never stacks a combine, so the two matching is the basis for "skipping the
      // combination outside the window is fine".
      const steps = plan(0);
      const actual = await residentLoop(gpu, initial, steps);
      assertEquals(bits(actual.x), bits(hostLoop(initial, steps)));
      assertEquals(actual.fences, 1);
    } finally {
      gpu.destroy();
    }
  },
);
