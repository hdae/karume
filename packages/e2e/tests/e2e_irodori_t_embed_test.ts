// Compares the host-generated `t_embed` table with the exporter's **torch-referenced golden**.
//
// `t_embed` is not in the IR (`cos` is not in the op vocabulary — ADR 0043 / 0047 decision 4), so
// no other check looks at this formula's correctness. It is the DiT's input itself, so a mistake
// changes the whole latent in the shape "the condition points at another time at every step".
//
// The golden is `outputs/test-assets/goldens/irodori-v4-small/pipeline/t-embed.safetensors` (`t`
// `[40]` and `t_embed` `[40,512]`), written by `tools/export-recipes/irodori/pipeline_ref.py`. An
// environment without it SKIPs explicitly (ADR 0005), unless the test assets are a complete fetch of
// the pin, where its absence FAILs (ADR 0139). No GPU is needed.

import { test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { parseSafetensors } from "@karume/runtime";
import { goldenDir, testAssetsFetched } from "@karume/test-support/test-assets";
import { tSchedule } from "../../models/src/irodori/host/sampler.ts";
import { timestepEmbedding, timestepFrequencies } from "../../models/src/irodori/host/t-embed.ts";
import { readFileIfPresent } from "../../models/tests/helpers/read-if-present.ts";

const GOLDEN = new URL("pipeline/t-embed.safetensors", goldenDir("irodori-v4-small"));

/** How to fetch the golden from the test-asset dataset. */
const FETCH_HINT =
  "fetch: node tools/test-assets/fetch.ts — the assets gate accepts only a complete fetch; " +
  "--include 'goldens/irodori-v4-small/pipeline/**' with KARUME_ALLOW_NO_ASSETS=1 runs this file alone";

/**
 * The tolerance for comparing the `t_embed` table.
 *
 * Measured (a plain comparison with `atol=0`, 40 steps × 512 columns = 20,480 elements):
 * **maxAbs 3.05e-5** (step 13 / t=0.674319 / column k=24). The breakdown splits in two:
 *
 * - **255/256 of the frequency columns are within 1 ulp** (≤ 5.96e-8 = 1 ulp of the range [-1,1])
 * - **Only the single column k=24 is 3.05e-5**. The cause is the frequency table itself: the
 *   `freqs[24]` torch's f32 `exp` returns is **exactly 1 ulp above** the value via JS's
 *   `Math.exp` (measured: adding that one bit drops the whole column to 5.96e-8).
 *   `args = t·freqs[24] ≈ 284` is in the region where f32's ulp is 3.05e-5, so the frequency's
 *   1 ulp shows up directly as an absolute difference of `cos` / `sin`.
 *
 * So the structure is that **in columns with large `args`, a transcendental function's 1 ulp grows
 * into an absolute difference of order 1e-5**, and bit identity would require copying SLEEF's
 * polynomial approximation (the same judgement as anima's `timestepsProj`).
 *
 * atol 3e-4 is about 10 times the measured worst 3.05e-5. Differences from implementation bugs show
 * at least 4 orders of magnitude above this (measured by fault injection): **1.41e+0 with the
 * cos/sin halves swapped** and **2.00e+0 with the frequencies' factor of 1000 forgotten**.
 */
const T_EMBED_ATOL = 3e-4;

/**
 * The tolerance between the closed-form t schedule and the golden's `t` (upstream `linspace`).
 *
 * Measured **5.96e-8** (= the same value as `tScheduleClosedFormMaxAbs` the exporter puts in
 * meta.json). `linspace` accumulates `start + i·step` in f32, so there are points where the last
 * ulp differs from the closed form `0.999·(1 − i/40)`. t enters only `t_embed` and the step size,
 * and its effect on the latent is 3 orders of magnitude below the golden comparison's threshold
 * (the doc of `EULER_REFERENCE_ATOL` in `tools/export-recipes/irodori/pipeline_ref.py`).
 */
const T_SCHEDULE_ATOL = 1e-7;

const goldenBytes = await readFileIfPresent(GOLDEN);
// An absent golden SKIPs unless `outputs/test-assets/` holds a complete fetch of the pin (ADR 0139),
// where the dataset at the pin lacks it and `readGolden` FAILs.
const AVAILABLE = goldenBytes !== undefined || testAssetsFetched();
if (!AVAILABLE) {
  console.warn(
    `[karume] The Irodori t_embed comparison SKIPs: missing ${GOLDEN.pathname} (${FETCH_HINT})`,
  );
}

/** The golden's `[name → f32 view]` (read once at registration). */
const readGolden = () => {
  if (goldenBytes === undefined) {
    throw new Error(
      `${GOLDEN.pathname} absent, but outputs/test-assets/ holds a complete fetch of the pinned ` +
        "revision: the test-asset dataset at the pin lacks it",
    );
  }
  const file = parseSafetensors(
    goldenBytes.buffer.slice(
      goldenBytes.byteOffset,
      goldenBytes.byteOffset + goldenBytes.byteLength,
    ),
  );
  const view = (name: string): { data: Float32Array; shape: readonly number[] } => {
    const tensor = file.tensors.get(name);
    assert(tensor !== undefined, `the golden has no '${name}'`);
    assertEquals(tensor.dtype, "F32", `dtype of golden '${name}'`);
    return {
      data: new Float32Array(file.buffer, tensor.byteOffset, tensor.byteLength / 4),
      shape: tensor.shape,
    };
  };
  return { t: view("t"), table: view("t_embed") };
};

test.skipIf(!AVAILABLE)(
  "Irodori t_embed: the host formula matches the torch-referenced golden table",
  () => {
    const { t, table } = readGolden();
    const [steps, width] = [table.shape[0], table.shape[1]];
    assertEquals(t.shape, [steps], "step counts of t and t_embed");
    const frequencies = timestepFrequencies(width);
    let worst = 0;
    let where = "";
    let nonFinite = 0;
    for (let step = 0; step < steps; step += 1) {
      const row = timestepEmbedding(t.data[step], frequencies);
      assertEquals(row.length, width, `width of step ${step}`);
      for (let column = 0; column < width; column += 1) {
        const golden = table.data[step * width + column];
        if (!Number.isFinite(row[column]) || !Number.isFinite(golden)) {
          nonFinite += 1;
          continue;
        }
        const diff = Math.abs(row[column] - golden);
        if (diff > worst) {
          worst = diff;
          where = `step ${step} (t=${t.data[step]}) column ${column}`;
        }
      }
    }
    // MUST: NaN / ±Inf fail. A naive difference check turns NaN into "difference 0" and passes it.
    assertEquals(nonFinite, 0, "non-finite samples in t_embed");
    assert(
      worst <= T_EMBED_ATOL,
      `the maximum absolute difference of t_embed ${worst.toExponential(4)} exceeds atol ${T_EMBED_ATOL} (${where})`,
    );
  },
);

test.skipIf(!AVAILABLE)(
  "Irodori t schedule: the closed form matches upstream linspace to about 1 ulp",
  () => {
    const { t } = readGolden();
    const steps = t.shape[0];
    // The golden is `[:-1]` with the trailing 0 dropped, so only the same range is looked at here.
    const schedule = tSchedule(steps, 0.999);
    let worst = 0;
    for (let index = 0; index < steps; index += 1) {
      worst = Math.max(worst, Math.abs(schedule[index] - t.data[index]));
    }
    assert(
      worst <= T_SCHEDULE_ATOL,
      `the maximum absolute difference of the t schedule ${worst.toExponential(4)} exceeds atol ${T_SCHEDULE_ATOL}`,
    );
  },
);
