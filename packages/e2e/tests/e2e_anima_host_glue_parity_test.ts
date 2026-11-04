/**
 * Checks Anima's host glue (the stages TS runs outside the IR) against the Python reference
 * fixtures **bit for bit**. Four functions are covered: `sigmaSchedule` / `cfgEulerStep`
 * (`sampler.ts`) and `denormalizeLatents` / `padSequence` (`latents.ts`). No GPU is needed. The
 * fixtures are `pipeline.{json,safetensors}` of `tools/export-recipes/anima/pipeline_ref.py` (the
 * generation commands and their roles: the recipe README's "Anima host pipeline reference fixture"
 * section), read from the test-asset root `goldenDir("anima-pipeline{,-f16,-i8,-turbo-f16}")` (ADR
 * 0139 — fetched with `node tools/test-assets/fetch.ts`). In all 4 variants, each stage's inputs
 * and outputs are in the same file, so the host glue can be checked on its own without mixing in
 * the DiT's error (the DiT's raw output `noise_*` is passed as the input and the next latent is
 * compared).
 *
 * MUST: the comparison is **bit-identical** (no tolerance). All 4 functions round to f32 one
 * operation at a time with `Math.fround`, and agreeing with the reference (numpy / torch sequential
 * f32 arithmetic) down to the last bit is exactly what the recipe README records as measured.
 * Loosening it lets a mix-up of the operation order (dividing by the reciprocal of std →
 * multiplying, moving where the f32 conversion happens) be absorbed as "GPU error" by the tolerance
 * of later stages.
 *
 * NOTE (why the turbo variant has no case for the CFG combination): the turbo reference was taken
 * with guidance_scale=1.0, and `pipeline_ref.py` does **not call** the uncond DiT. So the
 * `noise_uncond_stepNNNN` keys do not exist at all, and the `uncond + scale·(cond − uncond)` branch
 * has nothing to be checked against. The turbo variant pins the CFG=1 branch (cond goes straight
 * into Euler); the CFG combination branch is pinned by the other 3 variants (guidance_scale=4). The
 * presence of the keys is checked against `needsUncond(guidance_scale)` — a fixture "with
 * guidance≠1 but no uncond" fails instead of silently flowing into the CFG=1 branch. The assets are
 * outside git, so a variant whose fixture is absent SKIPs explicitly, naming the missing path and
 * the fetch command (ADR 0005) — except when `outputs/test-assets/` holds a complete fetch of the
 * pinned dataset (`testAssetsFetched`): then its cases FAIL, since the dataset at the pin lacks the
 * fixture.
 */

import { goldenDir, TEST_ASSETS, testAssetsFetched } from "@karume/test-support/test-assets";
import { test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { parseSafetensors, type Tensor } from "@karume/runtime";
import { cfgEulerStep, sigmaSchedule } from "../../models/src/anima/sampler.ts";
import { latentStats, denormalizeLatents, padSequence } from "../../models/src/anima/latents.ts";
import { needsUncond } from "../../models/src/generation/dpm-solver-multistep.ts";
import {
  readFileIfPresent,
  readTextIfPresent,
} from "../../models/tests/helpers/read-if-present.ts";

/**
 * The 4 fixture variants (directory name = the default output of `pipeline_ref.py` / turbo's
 * `--out`).
 */
const VARIANTS = [
  "anima-pipeline",
  "anima-pipeline-f16",
  "anima-pipeline-i8",
  "anima-pipeline-turbo-f16",
] as const;

/** The command that fetches the fixtures (pasted into the SKIP message). */
const FETCH_HINT = "node tools/test-assets/fetch.ts";

/** The fields of `pipeline.json` read here. */
type FixtureMeta = {
  readonly steps: number;
  readonly ref_steps: number;
  readonly shift: number;
  readonly guidance_scale: number;
  readonly min_sequence_length: number;
};

type Fixture = {
  readonly meta: FixtureMeta;
  readonly f32: (name: string) => { data: Float32Array<ArrayBuffer>; shape: readonly number[] };
  readonly has: (name: string) => boolean;
  /** The shape regardless of dtype (to read the row count of the id column = I32). */
  readonly shape: (name: string) => readonly number[];
};

const openFixture = async (variant: string): Promise<Fixture | undefined> => {
  const dir = goldenDir(variant);
  const [metaText, bytes] = await Promise.all([
    readTextIfPresent(new URL("pipeline.json", dir)),
    readFileIfPresent(new URL("pipeline.safetensors", dir)),
  ]);
  if (metaText === undefined || bytes === undefined) return undefined;
  const meta = JSON.parse(metaText) as FixtureMeta;
  const file = parseSafetensors(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
  return {
    meta,
    has: (name) => file.tensors.has(name),
    shape: (name) => {
      const tensor = file.tensors.get(name);
      assert(tensor !== undefined, `${variant}: '${name}' is missing from the fixture`);
      return tensor.shape;
    },
    f32: (name) => {
      const tensor = file.tensors.get(name);
      assert(tensor !== undefined, `${variant}: '${name}' is missing from the fixture`);
      assertEquals(tensor.dtype, "F32", `${variant}: dtype of '${name}'`);
      return {
        data: new Float32Array(file.buffer, tensor.byteOffset, tensor.byteLength / 4),
        shape: tensor.shape,
      };
    },
  };
};

/**
 * Checks that the bit patterns agree (not `===` on the values). `-0` and `+0`, and NaN payloads,
 * are distinguished too — neither the sign of a zero padding nor a stray non-finite value counts as
 * "equal".
 */
const assertBitIdentical = (actual: Float32Array, expected: Float32Array, label: string): void => {
  assertEquals(actual.length, expected.length, `${label}: element count`);
  const got = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
  const want = new Uint32Array(expected.buffer, expected.byteOffset, expected.length);
  let mismatches = 0;
  let first = -1;
  for (let index = 0; index < got.length; index += 1) {
    if (got[index] !== want[index]) {
      mismatches += 1;
      if (first < 0) first = index;
    }
  }
  assert(
    mismatches === 0,
    `${label}: ${mismatches}/${got.length} elements differ in bits (first index ${first}: ` +
      `${actual[first]} vs reference ${expected[first]})`,
  );
};

const stepTag = (step: number): string => `step${String(step).padStart(4, "0")}`;

const FETCHED = testAssetsFetched();

for (const variant of VARIANTS) {
  const fixture = await openFixture(variant);
  const files = `${goldenDir(variant).pathname}pipeline.{json,safetensors}`;
  /** The FAIL message of a fixture absent from a complete fetch of the pin. */
  const absent =
    `${files}: absent, but outputs/test-assets/ holds a complete fetch of` +
    ` ${TEST_ASSETS.repo}@${TEST_ASSETS.revision} — the dataset at the pin lacks it`;
  if (fixture === undefined && !FETCHED) {
    console.warn(
      `[karume] ${files} is missing, so the Anima host glue parity of ${variant} SKIPs (the` +
        ` fixtures are outside git). Fetch: ${FETCH_HINT}`,
    );
  }
  const ignore = fixture === undefined && !FETCHED;
  const skip = ignore ? ` (SKIP: no ${files} — ${FETCH_HINT})` : "";

  test.skipIf(ignore)(
    `${variant}: sigmaSchedule is bit-identical to the reference sigma sequence (terminal 0 included)${skip}`,
    () => {
      assert(fixture !== undefined, absent);
      const { steps, shift } = fixture.meta;
      assertBitIdentical(sigmaSchedule(steps, shift), fixture.f32("sigmas").data, "sigmas");
    },
  );

  test.skipIf(ignore)(
    `${variant}: cfgEulerStep chained ref_steps times is bit-identical to the reference latent sequence${skip}`,
    () => {
      assert(fixture !== undefined, absent);
      const { ref_steps: refSteps, guidance_scale: guidance } = fixture.meta;
      const sigmas = fixture.f32("sigmas").data;
      // Whether the uncond keys exist is decided by guidance (see the NOTE). A disagreement means
      // the fixture or the reference's branch is broken, so the case fails instead of skipping.
      assertEquals(
        fixture.has(`noise_uncond_${stepTag(1)}`),
        needsUncond(guidance),
        `guidance_scale ${guidance} disagrees with the presence of noise_uncond`,
      );
      // Each step starts from the reference's latent of the previous step (the TS output is not
      // carried forward). Carried forward, a drift at step 1 would propagate into the later steps
      // and hide which step broke.
      let previous = fixture.f32("latents_init").data;
      for (let step = 1; step <= refSteps; step += 1) {
        const tag = stepTag(step);
        const uncond = needsUncond(guidance) ? fixture.f32(`noise_uncond_${tag}`).data : undefined;
        // Δσ is spelled as in the denoise loop (`denoiseStep` in `pipeline.ts`).
        const next = cfgEulerStep(
          previous,
          fixture.f32(`noise_cond_${tag}`).data,
          uncond,
          Math.fround(sigmas[step] - sigmas[step - 1]),
          guidance,
        );
        const expected = fixture.f32(`latents_${tag}`).data;
        assertBitIdentical(next, expected, `latents_${tag}`);
        previous = expected;
      }
    },
  );

  test.skipIf(ignore)(
    `${variant}: denormalizeLatents is bit-identical to the reference denormalised latent${skip}`,
    () => {
      assert(fixture !== undefined, absent);
      // Production uses the copy of the VAE config (latentStats), so it is passed here too. First
      // pin that the copy holds the same numbers as this variant's reference (values taken by
      // reading the VAE again for each variant).
      const { mean, std } = latentStats();
      assertBitIdentical(mean, fixture.f32("latents_mean").data, "latents_mean");
      assertBitIdentical(std, fixture.f32("latents_std").data, "latents_std");
      // The denormalisation input is the latent of the last step with a reference (the decode
      // stage of `pipeline_ref.py`).
      const last = fixture.f32(`latents_${stepTag(fixture.meta.ref_steps)}`);
      const expected = fixture.f32("latents_denorm");
      assertEquals(expected.shape, last.shape, "shape of latents_denorm");
      assertBitIdentical(
        denormalizeLatents(last.data, last.shape, mean, std),
        expected.data,
        "latents_denorm",
      );
    },
  );

  test.skipIf(ignore)(
    `${variant}: padSequence is bit-identical to the reference 512-row zero padding (positive and negative)${skip}`,
    () => {
      assert(fixture !== undefined, absent);
      const rows = fixture.meta.min_sequence_length;
      for (const prefix of ["", "neg_"]) {
        // The IR conditioner outputs the same row count T as the T5 id column. The reference is the
        // upstream conditioner's output padded to `min_sequence_length` rows, so its first T rows
        // stand in for the IR output, and the padded result is compared with the whole reference
        // (down to the sign of the zeros in the padding). T comes from the reference's id column,
        // so a mix-up of where the padding starts breaks here too.
        const targetLength = fixture.shape(`${prefix}t5_input_ids`)[1];
        const padded = fixture.f32(`${prefix}encoder_hidden_states`);
        const [batch, total, width] = padded.shape;
        assertEquals([batch, total], [1, rows], `shape of ${prefix}encoder_hidden_states`);
        assert(targetLength < rows, `${prefix}t5 length ${targetLength} leaves no padding`);
        const hidden: Tensor = {
          dtype: "f32",
          shape: [1, targetLength, width],
          data: padded.data.slice(0, targetLength * width),
        };
        assertBitIdentical(
          padSequence(hidden, rows),
          padded.data,
          `${prefix}encoder_hidden_states`,
        );
      }
    },
  );
}
