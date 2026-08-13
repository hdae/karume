/**
 * Irodori の DiT ループを GPU 常駐化する 2 グラフ（`host/sampler-graph.ts`）の門（実 GPU）。
 *
 * 見るのは 2 つだけ:
 *
 * 1. **数値** — ホストで組んだ combine / euler をループの積み方どおりに回した結果が、TS 正本
 *    （`combineCfg` / `eulerStep`）と**ビット単位で**一致すること。ここが割れれば `pipeline.ts`
 *    の常駐ループも割れる（= WAV sha256 門が落ちる）ので、原因を最小の材料で切り分ける門。
 * 2. **構造** — ステップ数ぶん積んでも `onSubmittedWorkDone` が batch の 1 本しか出ないこと。
 *    値だけを見ていると、内部が run 経路（forward ごとのフェンス）へ退避しても緑のままになる。
 *
 * `dit` は要らない（速度場はホストで作った配列を常駐テンソルへ書いて代用する）ので、配布形
 * 3.3GB を読まずに閉じる。ループ全体が実際に 1 batch で回ることは E2E 側（WAV sha256 門）が
 * 押さえる。
 */

import { assert, assertEquals } from "@std/assert";
import {
  acquireGpu,
  createSession,
  type GpuContext,
  openModel,
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

/** 実形（v4-small の no-ref ケース相当）より小さくしてあるが、幅は配布形と同じ 32。 */
const FRAMES = 12;
const LATENT_DIM = 32;
const COUNT = FRAMES * LATENT_DIM;
const BYTES = COUNT * 4;
/** 実効の CFG 強さ（配布形 v4-small の text / speaker / caption）。 */
const SCALES = [3, 5, 3] as const;
const STEPS = 4;

/** 決定的な擬似速度場（乱数は使わない — 失敗が再現しないため）。 */
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

/** ビット列比較（丸めの取り違えを許容しない）。 */
const bits = (data: Float32Array | ArrayBuffer): readonly number[] =>
  Array.from(
    data instanceof ArrayBuffer
      ? new Uint32Array(data)
      : new Uint32Array(data.buffer, data.byteOffset, data.length),
  );

/**
 * `queue.onSubmittedWorkDone` の呼び出し回数を数える（フェンス数の機械検査）。
 * 実装内部に観測点を足さずに済むよう、テスト側でキューの面をラップする
 * （`packages/runtime/tests/gpu_resident_batch_test.ts` と同じ手）。
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

/** 1 step ぶんの材料（cond の速度場 / 変種ごとの速度場 / 刻み幅）。 */
type Step = {
  readonly cond: Float32Array<ArrayBuffer>;
  readonly variants: readonly Float32Array<ArrayBuffer>[];
  readonly deltaT: number;
};

/** ステップ列を作る（`variantCount` = 0 は CFG 窓の外の step）。 */
const plan = (variantCount: number): readonly Step[] =>
  Array.from({ length: STEPS }, (_, step) => ({
    cond: fill(1000 + step),
    variants: Array.from({ length: variantCount }, (_, k) => fill(2000 + step * 10 + k)),
    deltaT: Math.fround(-0.999 / STEPS),
  }));

/** TS 正本でループを回す（このファイルが比べる相手）。 */
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
 * 常駐グラフでループを回す（`pipeline.ts` の `runDitLoopResident` と**同じ積み方**）。
 * `dit` の forward だけがホスト配列の `write` に置き換わっている。
 */
const residentLoop = async (
  gpu: GpuContext,
  initial: Float32Array<ArrayBuffer>,
  steps: readonly Step[],
): Promise<{ readonly x: ArrayBuffer; readonly fences: number }> => {
  const combine = await createSession(gpu, openModel(combineGraph(FRAMES, LATENT_DIM)));
  const euler = await createSession(gpu, openModel(eulerGraph(FRAMES, LATENT_DIM)));
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
          await combine.enqueue({
            [COMBINE_INPUTS.accumulator]: k === 0 ? vCond : accumulator,
            [COMBINE_INPUTS.cond]: vCond,
            [COMBINE_INPUTS.variant]: vVariant,
            [COMBINE_INPUTS.scale]: scalar(SCALES[k]),
          }, { batch, copyOutputs: { [COMBINE_OUTPUT]: accumulator } });
        }
        await euler.enqueue({
          [EULER_INPUTS.x]: xT,
          [EULER_INPUTS.velocity]: step.variants.length > 0 ? accumulator : vCond,
          [EULER_INPUTS.deltaT]: scalar(step.deltaT),
        }, { batch, copyOutputs: { [EULER_OUTPUT]: xT } });
      }
      assertEquals(fences.count() - before, 0, "enqueue はフェンスを 1 本も張らない");
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

Deno.test({
  name: "常駐 CFG + Euler ループは TS 正本とビット一致し、フェンスは batch の 1 本（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      const initial = fill(7);
      const steps = plan(SCALES.length);
      // 恒真化の門: 期待値が初期値のままなら、ループが 1 度も効いていなくても緑になる。
      const expected = hostLoop(initial, steps);
      assert(
        JSON.stringify(bits(expected)) !== JSON.stringify(bits(initial)),
        "期待値が初期潜在と同じ（検出器として空振る）",
      );

      const actual = await residentLoop(gpu, initial, steps);
      assertEquals(bits(actual.x), bits(expected), "常駐ループの潜在が TS 正本と違う");
      assertEquals(actual.fences, 1, "ループ全体で onSubmittedWorkDone は 1 回");
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "CFG 窓の外の step（変種 0 本）は cond をそのまま Euler へ渡す（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      const initial = fill(11);
      // 正本側は `combineCfg(cond, [])`（= cond の写し）を通る経路。常駐側は combine を
      // 1 度も積まないので、両者が一致することが「窓の外で合成を飛ばしてよい」の根拠。
      const steps = plan(0);
      const actual = await residentLoop(gpu, initial, steps);
      assertEquals(bits(actual.x), bits(hostLoop(initial, steps)));
      assertEquals(actual.fences, 1);
    } finally {
      gpu.destroy();
    }
  },
});
