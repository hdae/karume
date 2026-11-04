// Host-side unit cases of the reference-audio preprocessing (K-weighting coefficients, input
// errors, reflect pad). They need neither a GPU nor assets. The parity gate against the upstream
// golden (goldens/dacvae-32dim/host/ of the test assets) lives in
// packages/e2e/tests/e2e_irodori_reference_host_test.ts.

import { test } from "vite-plus/test";
import { assert, assertEquals, assertThrows } from "@std/assert";
import { integratedLoudness, kWeightingFilters } from "../src/irodori/host/loudness.ts";
import { ModelInputError } from "../src/errors.ts";
import { normalizeReference, reflectPadToHop } from "../src/irodori/host/reference.ts";

/** The value used with the real v4-small weights (the same number `pipelineConfig` carries). */
const SAMPLE_RATE = 48000;

// ---- K-weighting coefficients (catching a miscopied formula) ----------------------------

test("kWeightingFilters: the 48kHz coefficients match upstream's numbers (pyloudnorm's RBJ formulas)", () => {
  const [shelf, highPass] = kWeightingFilters(SAMPLE_RATE);
  // The values are the coefficients of `pyloudnorm.IIRfilter` built at 48kHz (measured in recon).
  // The coefficients are built from the formulas, so if they match, fc / Q / dB gain are not mixed
  // up.
  assertEquals(
    [...shelf.b],
    [1.5351828863637502, -2.691804030199196, 1.198426263333146],
    "high shelf numerator",
  );
  assertEquals(
    [...shelf.a],
    [1, -1.6906995865986896, 0.7325047060963897],
    "high shelf denominator",
  );
  assertEquals(
    [...highPass.b],
    [0.9950442970178917, -1.9900885940357833, 0.9950442970178917],
    "high pass numerator",
  );
  assertEquals(
    [...highPass.a],
    [1, -1.990076284018423, 0.9901009040531438],
    "high pass denominator",
  );
});

test("kWeightingFilters: changing the frequency moves the coefficients (48kHz is not baked in)", () => {
  const [shelf] = kWeightingFilters(SAMPLE_RATE);
  const [other] = kWeightingFilters(44100);
  assertEquals(
    shelf.b[0] === other.b[0],
    false,
    "the sampling rate does not affect the coefficients",
  );
});

// ---- Silence / invalid input -----------------------------------------------------

test("integratedLoudness: silence is clamped to the floor −70 (never returns −Infinity)", () => {
  assertEquals(integratedLoudness(new Float32Array(SAMPLE_RATE), SAMPLE_RATE), -70);
});

test("integratedLoudness: rejects an empty waveform and an invalid rate", () => {
  assertThrows(
    () => integratedLoudness(new Float32Array(0), SAMPLE_RATE),
    RangeError,
    "the waveform is empty",
  );
  assertThrows(
    () => integratedLoudness(Float32Array.of(1), 0),
    RangeError,
    "not a positive integer",
  );
});

test("integratedLoudness: rejects a rate so small the windows cannot be derived (positive integers are not enough)", () => {
  // The windows are derived with floor (kernel = trunc(0.4·sr) / stride = trunc(0.1·sr)), so at a
  // small rate the windows themselves vanish. Without the check, sr ≤ 9 gives stride 0 and an
  // Infinity block count, failing in the unrelated guise of "Invalid typed array length", and at
  // sr ≤ 2 the kernel is 0 too.
  assertThrows(
    () => integratedLoudness(Float32Array.of(0.5, 0.5), 9),
    RangeError,
    "the loudness window cannot be derived",
  );
  assertThrows(
    () => integratedLoudness(Float32Array.of(0.5, 0.5), 2),
    RangeError,
    "0.4 s = 0 samples / stride 0 samples",
  );
  // From 10Hz on the windows exist (the lower-bound check does not reject the valid domain).
  assertEquals(integratedLoudness(new Float32Array(20), 10), -70);
});

test("normalizeReference: rejects non-finite samples with their position (they do not slip past the gain stage)", () => {
  // Without the check, NaN slips past the peak detection because `magnitude > peakBeforeScale` is
  // always false, and +Inf gives `peakGain = 1/∞ = 0`, **crushing every sample to 0** while leaving
  // a single NaN. Neither throws; both reach the encoder as "an almost silent reference audio".
  for (const value of [NaN, Infinity, -Infinity]) {
    const samples = Float32Array.from({ length: SAMPLE_RATE }, (_, index) =>
      index === 7 ? value : 0.1,
    ) as Float32Array<ArrayBuffer>;
    assertThrows(
      () => normalizeReference(samples, SAMPLE_RATE),
      ModelInputError,
      "sample 7 of the reference audio is non-finite",
    );
  }
});

test("normalizeReference: rejects an empty waveform at the entry (it does not turn into the window check)", () => {
  // An empty waveform is the size of the wav the caller passed (ADR 0107 decision 5). Without this
  // check it would fall into `integratedLoudness`'s "the windows cannot be derived" side and be
  // indistinguishable from the check on derived values (rate, window length).
  assertThrows(
    () => normalizeReference(new Float32Array(0), SAMPLE_RATE),
    ModelInputError,
    "the reference audio has 0 samples",
  );
  // The other side: `integratedLoudness`'s `RangeError` on the premises of derived values stays
  // (the caller cannot fix it).
  const windows = assertThrows(() => integratedLoudness(Float32Array.of(0.5, 0.5), 9));
  assert(!(windows instanceof ModelInputError));
});

test("normalizeReference: finite input passes (the checks do not reject everything)", () => {
  const samples = Float32Array.from(
    { length: SAMPLE_RATE },
    (_, index) => Math.sin(index / 50) * 0.1,
  ) as Float32Array<ArrayBuffer>;
  const normalized = normalizeReference(samples, SAMPLE_RATE);
  assertEquals(normalized.data.length, samples.length);
  assertEquals(
    normalized.data.every((sample) => Number.isFinite(sample)),
    true,
  );
});

// ---- Reflect pad (a form that holds without a golden) ------------------------------

test("reflectPadToHop: the mirror does not include the edge sample (the same spelling as torch's reflect)", () => {
  const samples = Float32Array.from({ length: 5 }, (_, i) => i + 1) as Float32Array<ArrayBuffer>;
  // hop 4 → length 5 goes to 8 (pad 3). [1,2,3,4,5] → [1,2,3,4,5,4,3,2].
  assertEquals(Array.from(reflectPadToHop(samples, 4)), [1, 2, 3, 4, 5, 4, 3, 2]);
});

test("reflectPadToHop: when already a multiple, returns the input as is without even copying", () => {
  const samples = new Float32Array(8) as Float32Array<ArrayBuffer>;
  assertEquals(reflectPadToHop(samples, 4), samples);
});

test("reflectPadToHop: rejects an input shorter than hop (the mirror would jump past its own start)", () => {
  assertThrows(
    () => reflectPadToHop(new Float32Array(3) as Float32Array<ArrayBuffer>, 1920),
    ModelInputError,
    "hopLength 1920 is required",
  );
});
