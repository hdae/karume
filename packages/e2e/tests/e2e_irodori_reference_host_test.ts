// The **parity check** of the reference audio's host preprocessing (LUFS measurement → −16
// normalisation → reflect pad).
// No GPU is needed (pure functions only), but the golden
// `outputs/test-assets/goldens/dacvae-32dim/host/` is (fetch it as FETCH_HINT below says).
//
// The golden was written by `tools/export-recipes/irodori/dacvae/host.py` **calling upstream**,
// and that script measures every time, before writing, that "the normalised waveform is
// reproduced bit-identically from the 2 recorded scalars" and that "it is bit-identical to
// upstream's `_normalize_loudness`". This check closing is therefore the second stage of "the TS
// host preprocessing produces the same waveform as upstream".
//
// ## What is compared and what is not
//
// Of the 5 golden cases, **3 go through LUFS normalisation** (`normalizeDb: -16`); `ref-plain` /
// `ref-hot` are cases of upstream's "no normalisation (peak limiting only)" path. The pipeline's
// target is fixed at −16 (`TARGET_DB` in `host/reference.ts`) and **has no** such path, so from
// these 2 only **the LUFS measurement** is compared (comparing the waveform too would make the
// check demand a branch the implementation does not have).
//
// ## Covering the branches (blocking tautology)
//
// - `ref-short` (0.4 s) — both the zero-pad measurement branch below 0.5 s **and** the
//   peak-limiting branch (a gain of 441× raises the peak to 1.84). **The only case that hits peak
//   limiting on the LUFS path**
// - `ref-odd` (144,777 samples) — the remainder branch of reflect pad **and** the block zero-pad
//   branch (it shares its first 144,000 samples with `ref-plain`, so an implementation that drops
//   the incomplete last block would give both the same LUFS — this checks directly that they do
//   not match)
// - `ref-default` — the identity branch with a pad of 0 samples
// - `ref-hot` — 3× amplitude (the LUFS measurement itself does not depend on the value range)

import { readFile } from "@karume/test-support/fs";
import { goldenDir, testAssetsFetched } from "@karume/test-support/test-assets";
import { test } from "vite-plus/test";
import { assertAlmostEquals, assertEquals } from "@std/assert";
import { parseSafetensors } from "@karume/runtime";
import { integratedLoudness } from "../../models/src/irodori/host/loudness.ts";
import { normalizeReference, reflectPadToHop } from "../../models/src/irodori/host/reference.ts";
import { readTextIfPresent } from "../../models/tests/helpers/read-if-present.ts";
import { goldenPresent } from "./helpers/irodori-assets.ts";

/** The operating values of real-weight v4-small (the same numbers `pipelineConfig` carries). */
const SAMPLE_RATE = 48000;
const HOP_LENGTH = 1920;

/**
 * The tolerance of the LUFS measurement (LU).
 *
 * Measured (the table this check logs every time; 2026-08-12, the 5 golden cases):
 *
 * | Case        | Seconds   | refDb diff | Gain rel. diff | Peak gain rel. diff | Normalised maxAbs |
 * | ----------- | --------- | ---------- | -------------- | ------------------- | ----------------- |
 * | ref-default | 7.6       | 5.131e-6   | −6.061e-7      | 0 (exact)           | 5.961e-7          |
 * | ref-plain   | 3.0       | 2.659e-6   | — (off path)   | —                   | —                 |
 * | ref-hot     | 3.0       | 4.982e-6   | — (off path)   | —                   | —                 |
 * | ref-short   | 0.4       | −2.158e-5  | 2.500e-6       | −2.457e-6           | 1.192e-7          |
 * | ref-odd     | 3.0161875 | 3.526e-6   | −3.745e-7      | 0 (exact)           | 3.576e-7          |
 *
 * The difference has **a single source**: upstream runs the K-weighting IIR in f32 and this code in
 * f64 (the MUST in `host/loudness.ts`). Block splitting and gating are integer and comparison
 * decisions, so a different implementation shows as a step of 0.01 LU or more (measured: counting
 * the blocks wrongly shifted `ref-odd` by 0.074 LU — the step the "block zero pad" test below
 * looks at).
 *
 * 2e-4 LU is **9.3 times** the measured worst 2.158e-5. As a normalisation gain it is a relative
 * 2.3e-5, smaller than 1 LSB of 16-bit PCM (3.05e-5) — a size that **vanishes once written to
 * WAV**.
 */
const REF_DB_ATOL = 2e-4;

/**
 * The relative tolerance of the gains (LUFS gain / peak gain). **8.0 times** the measured worst
 * 2.500e-6 (`ref-short`). The `refDb` difference just rides on the exponent, so it is a number of
 * the same nature as the table above.
 */
const GAIN_RTOL = 2e-5;

/**
 * The absolute tolerance of the normalised / padded waveform. **8.4 times** the measured worst
 * 5.961e-7 (`ref-default`).
 *
 * The range is ±1, so this is around −106 dBFS. What it really is: one gain's relative difference
 * of 6e-7 showing at the top of the range, sitting at 10 times f32's 1 ulp (6e-8). **A wrong
 * formula** (the target dB, the gain's sign, the peak condition) shows at O(0.1–1), the same as the
 * range.
 */
const WAVE_ATOL = 5e-6;

const GOLDEN_DIR = new URL("host/", goldenDir("dacvae-32dim"));
const FETCH_HINT =
  "fetch: node tools/test-assets/fetch.ts — the assets gate accepts only a complete fetch; " +
  "--include 'goldens/dacvae-32dim/host/**' with KARUME_ALLOW_NO_ASSETS=1 runs this file alone";

/**
 * Whether `outputs/test-assets/` holds a complete fetch of the pin (ADR 0139). Then an absent
 * `meta.json` means the dataset at the pin lacks it, so the cases FAIL instead of SKIPping.
 */
const FETCHED = testAssetsFetched();
const lostAsset = (url: URL): Error =>
  new Error(
    `${url.pathname} absent, but outputs/test-assets/ holds a complete fetch of the pinned ` +
      "revision: the test-asset dataset at the pin lacks it",
  );

/**
 * Only the fields of the golden `meta.json` this check reads (the exporter owns the shape — this is
 * the reader).
 */
type GoldenCase = {
  readonly samples: number;
  readonly seconds: number;
  readonly normalizeDb: number | null;
  readonly refDb: number;
  readonly loudnessGain: number;
  readonly peakBeforeScale: number;
  readonly peakGain: number;
  readonly paddedSamples: number;
  readonly padSamples: number;
  readonly frames: number;
};
type GoldenMeta = {
  readonly sampleRate: number;
  readonly hopLength: number;
  readonly gainFactor: number;
  readonly cases: Readonly<Record<string, GoldenCase>>;
};

const metaUrl = new URL("meta.json", GOLDEN_DIR);
const metaText = await readTextIfPresent(metaUrl);
const readMeta = (): GoldenMeta => {
  if (metaText === undefined) throw lostAsset(metaUrl);
  return JSON.parse(metaText) as GoldenMeta;
};

/** `case.<name>.safetensors` of one golden case. */
const caseUrl = (name: string): URL => new URL(`case.${name}.safetensors`, GOLDEN_DIR);

// An absent `meta.json` SKIPs unless the test assets are a complete fetch of the pin, where
// `readMeta` FAILs. A case file absent beside a present `meta.json` is an inconsistent golden set,
// so `readCase` FAILs whatever the fetch record says.
const RUNNABLE = metaText !== undefined || FETCHED;
if (!RUNNABLE) {
  console.warn(
    `[karume] The reference-audio host parity check SKIPs: missing ${metaUrl.pathname} (${FETCH_HINT})`,
  );
}

/** The golden tensors of one case (`raw` / `normalized` / `padded`). */
const readCase = async (
  name: string,
): Promise<Readonly<Record<string, Float32Array<ArrayBuffer>>>> => {
  const url = caseUrl(name);
  if (!goldenPresent(url)) {
    throw metaText === undefined
      ? lostAsset(url)
      : new Error(
          `inconsistent golden set: ${metaUrl.pathname} is present but ${url.pathname} is absent`,
        );
  }
  const bytes = await readFile(url);
  const file = parseSafetensors(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer,
  );
  const out: Record<string, Float32Array<ArrayBuffer>> = {};
  for (const key of ["raw", "normalized", "padded"]) {
    const spec = file.tensors.get(key);
    if (spec === undefined) throw new Error(`golden case.${name} has no '${key}'`);
    out[key] = new Float32Array(
      file.buffer.slice(spec.byteOffset, spec.byteOffset + spec.byteLength),
    ) as Float32Array<ArrayBuffer>;
  }
  return out;
};

const maxAbsDiff = (actual: Float32Array, expected: Float32Array, where: string): number => {
  if (actual.length !== expected.length) {
    throw new Error(`${where}: the lengths differ, ${actual.length} vs ${expected.length}`);
  }
  let worst = 0;
  for (let i = 0; i < actual.length; i += 1) {
    worst = Math.max(worst, Math.abs(actual[i] - expected[i]));
  }
  return worst;
};

// ---- Parity with the golden (the main check) -------------------------------------------

test.skipIf(!RUNNABLE)(
  "the reference audio's host preprocessing matches the golden (upstream measurement) — 5 cases",
  async () => {
    const meta = readMeta();
    assertEquals(meta.sampleRate, SAMPLE_RATE, "golden sampleRate");
    assertEquals(meta.hopLength, HOP_LENGTH, "golden hopLength");
    // The gain conversion factor (`ln(10)/20`) — assumes `host/reference.ts` holds the same number.
    assertAlmostEquals(Math.LN10 / 20, meta.gainFactor, 1e-15, "gainFactor");

    const rows: string[] = [];
    let worstRefDb = 0;
    let worstGain = 0;
    let worstWave = 0;
    for (const [name, expected] of Object.entries(meta.cases)) {
      const tensors = await readCase(name);
      assertEquals(tensors["raw"].length, expected.samples, `${name}: length of the golden's raw`);

      const refDb = integratedLoudness(tensors["raw"], SAMPLE_RATE);
      const refDbDiff = refDb - expected.refDb;
      worstRefDb = Math.max(worstRefDb, Math.abs(refDbDiff));
      if (expected.normalizeDb === null) {
        // A case of upstream's "no normalisation" path. Only the LUFS measurement is checked (the
        // module doc).
        rows.push(`  ${name.padEnd(11)} refDb ${refDbDiff.toExponential(3)} (off the LUFS path)`);
        assertAlmostEquals(refDb, expected.refDb, REF_DB_ATOL, `${name}: refDb`);
        continue;
      }
      assertEquals(expected.normalizeDb, -16, `${name}: the golden's target dB is not −16`);

      const normalized = normalizeReference(tensors["raw"], SAMPLE_RATE);
      const padded = reflectPadToHop(normalized.data, HOP_LENGTH);
      const gainDiff = (normalized.loudnessGain - expected.loudnessGain) / expected.loudnessGain;
      const peakGainDiff = (normalized.peakGain - expected.peakGain) / expected.peakGain;
      const waveDiff = maxAbsDiff(normalized.data, tensors["normalized"], `normalized of ${name}`);
      const padDiff = maxAbsDiff(padded, tensors["padded"], `padded of ${name}`);
      worstGain = Math.max(worstGain, Math.abs(gainDiff), Math.abs(peakGainDiff));
      worstWave = Math.max(worstWave, waveDiff, padDiff);
      rows.push(
        `  ${name.padEnd(11)} refDb ${refDbDiff.toExponential(3)} / gain ${gainDiff.toExponential(
          3,
        )} / peak gain ${peakGainDiff.toExponential(3)} / waveform ${waveDiff.toExponential(3)}`,
      );

      assertAlmostEquals(refDb, expected.refDb, REF_DB_ATOL, `${name}: refDb`);
      assertAlmostEquals(gainDiff, 0, GAIN_RTOL, `${name}: relative difference of the LUFS gain`);
      assertAlmostEquals(
        peakGainDiff,
        0,
        GAIN_RTOL,
        `${name}: relative difference of the peak gain`,
      );
      assertEquals(
        waveDiff < WAVE_ATOL,
        true,
        `${name}: difference of the normalised waveform ${waveDiff}`,
      );
      assertEquals(
        padDiff < WAVE_ATOL,
        true,
        `${name}: difference of the padded waveform ${padDiff}`,
      );
      // The length and the frames are integer decisions, so an exact match is required.
      assertEquals(padded.length, expected.paddedSamples, `${name}: padded length`);
      assertEquals(padded.length / HOP_LENGTH, expected.frames, `${name}: frame count`);
    }
    console.log(
      `[parity] irodori reference-audio preprocessing (tolerance: refDb ${REF_DB_ATOL} / gain ${GAIN_RTOL}` +
        ` / waveform ${WAVE_ATOL})\n${rows.join("\n")}\n  worst: refDb ${worstRefDb.toExponential(
          3,
        )} / gain ${worstGain.toExponential(3)} / waveform ${worstWave.toExponential(3)}`,
    );
  },
);

// ---- Covering the branches (blocking tautology) ------------------------------------------

test.skipIf(!RUNNABLE)("the 5 golden cases actually hit the branches they target", async () => {
  const meta = readMeta();
  const short = meta.cases["ref-short"];
  const odd = meta.cases["ref-odd"];
  const hot = meta.cases["ref-hot"];
  const base = meta.cases["ref-default"];

  // ① The zero-pad measurement branch below 0.5 s.
  assertEquals(
    short.seconds < 0.5,
    true,
    `ref-short is ${short.seconds} s — it does not hit the branch`,
  );
  // ② The peak-limiting branch. **On the LUFS path only ref-short hits it** (ref-hot is the case
  //    that hits the peak on upstream's "no normalisation" path; through −16 normalisation its gain
  //    is 0.27× and the peak drops to 0.81, so no limiting applies).
  const normalizedShort = normalizeReference((await readCase("ref-short"))["raw"], SAMPLE_RATE);
  assertEquals(normalizedShort.peakBeforeScale > 1, true, "ref-short does not reach peak > 1");
  assertEquals(normalizedShort.peakGain < 1, true, "peak limiting does not apply to ref-short");
  const normalizedHot = normalizeReference((await readCase("ref-hot"))["raw"], SAMPLE_RATE);
  assertEquals(
    normalizedHot.peakGain,
    1,
    "ref-hot hit peak limiting after −16 normalisation (the premise changed — fix the doc's explanation)",
  );
  assertEquals(
    hot.peakGain < 1,
    true,
    "the golden's ref-hot does not hit the peak-limiting branch",
  );
  // ③ The remainder branch and the identity branch of reflect pad.
  assertEquals(base.padSamples, 0, "ref-default does not hit the pad's identity branch");
  assertEquals(
    0 < odd.padSamples && odd.padSamples < HOP_LENGTH,
    true,
    `ref-odd's pad is ${odd.padSamples} samples — it does not hit the remainder branch`,
  );
});

test.skipIf(!RUNNABLE)(
  "block zero pad: the LUFS of ref-odd and ref-plain differ (the tail is not dropped)",
  async () => {
    // ref-odd is ref-plain's first 144,000 samples plus 777 samples. In an implementation that
    // drops the incomplete last block (the `torch.nn.Unfold` spelling), these 777 samples make no
    // block at all, so **the two LUFS values match exactly**. `julius.core.unfold` zero-pads and
    // makes a 28th block, so they do not match — the measured step is 0.074 LU.
    const plain = integratedLoudness((await readCase("ref-plain"))["raw"], SAMPLE_RATE);
    const odd = integratedLoudness((await readCase("ref-odd"))["raw"], SAMPLE_RATE);
    const gap = Math.abs(odd - plain);
    console.log(`[parity] block zero pad: ref-odd − ref-plain = ${(odd - plain).toFixed(6)} LU`);
    assertEquals(
      gap > 100 * REF_DB_ATOL,
      true,
      `the LUFS difference of ref-odd and ref-plain is only ${gap} — the incomplete last block is dropped`,
    );
  },
);
