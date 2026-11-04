// Parity of the vowel detector's host layer (the feature extraction and postprocessing of
// `src/vowel-detector/`) with the Python reference, plus behaviour tests. Needs neither a GPU nor
// real assets, so it always runs.
//
// The fixture `vowel-detector/parity.json` folds in, as is, the references held by upstream
// `@hdae/vowel-detector` (MIT, same author); karume regenerated none of them (the provenance is in
// `_doc` at the top of the fixture). The reference is the Python implementation
// (`training/src/vowel_detector/{features,dsp,postprocess}.py`); upstream's TS implementation is a
// port of it = **not the reference**.
//
// What is pinned is behaviour: "the same waveform yields the same 83-dimensional features as
// Python" and "the same logits yield the same `.lab` as Python". Neither how the FFT is run nor the
// internal representation of the smoothing is pinned.
//
// ## Deriving the tolerance (measured)
//
// The reference features are **stored as f32** (JSON of the values Python passed through
// `astype(np.float32)`). The fixture's reference features have absolute values ≤ 3.7179, so the f32
// ULP in that band ([2,4)) is 2.384e-7. That is, **the reference side already carries rounding of
// the order of 1 ULP** as a floor, and agreement finer than that cannot be measured in principle.
// Measured:
//
//  - the port (f64 FFT)      max abs diff 4.768e-7 = **2 ULP** (largest in the 80 log-mel columns;
//                            voiced 2.98e-8 / logEnergy 5.96e-8 / ZCR matches exactly)
//  - upstream TS (f32 FFT)   max abs diff 5.424e-6 (same fixture, same mel basis)
//
// The port is an order of magnitude better because it follows `np.fft` running in f64 (the MUST in
// `src/vowel-detector/fft.ts`). The check is therefore set at "twice 2 ULP" = {@link FEATURE_ATOL}.
// The 5.42e-6 of upstream's TS implementation does **not pass** this check — the width keeps a change
// back to an f32 FFT from slipping in silently.
//
// Postprocessing has no tolerance (**exact match** of the `.lab`). From stage 2 on it is a discrete
// segmentation, so a slight numeric drift can only either split the string or leave it identical.

import { readTextFile } from "@karume/test-support/fs";
import { test } from "vite-plus/test";
import { assert, assertEquals, assertThrows } from "@std/assert";
import { ModelInputError } from "../src/errors.ts";
import {
  DSP_WINDOW,
  extractFeatures,
  FEATURE_DIM,
  HOP,
  MEL_BINS,
  N_FFT,
  N_MELS,
  SAMPLE_RATE,
  WIN_LENGTH,
} from "../src/vowel-detector/features.ts";
import {
  FRAME_SEC,
  LIPSYNC_CLASSES,
  logitsToSegments,
  toLab,
  viterbiSmooth,
} from "../src/vowel-detector/postprocess.ts";

type Fixture = {
  readonly constants: {
    readonly sampleRate: number;
    readonly nFft: number;
    readonly winLength: number;
    readonly hop: number;
    readonly nMels: number;
    readonly featureDim: number;
    readonly dspWindow: number;
    readonly switchPenalty: number;
    readonly minDurationFrames: number;
    readonly frameSec: number;
    readonly classes: string[];
  };
  readonly melBasis: number[][];
  readonly audio: number[];
  readonly features: number[][];
  readonly logits: number[][];
  readonly lab: string;
};

const FIXTURE_PATH = new URL("./fixtures/vowel-detector/parity.json", import.meta.url);
const fixture = JSON.parse(await readTextFile(FIXTURE_PATH)) as Fixture;

/** Twice 2 ULP of the f32 reference (4.768e-7 in the |x| ≤ 3.72 band). Derived at the top of the file. */
const FEATURE_ATOL = 1e-6;

const melBasis = Float32Array.from(fixture.melBasis.flat());
const audio = Float32Array.from(fixture.audio);

// ---- The fixture itself -----------------------------------------------------

test("the fixture is not empty (a mix-up does not let every case pass through)", () => {
  // Kept first. If a misread or a failed generation leaves zero rows, the loops below run zero times
  // and a "green but checking nothing" state passes silently.
  assertEquals(fixture.melBasis.length, N_MELS);
  assertEquals(fixture.features.length, 108, "frame count of the reference features");
  assertEquals(fixture.logits.length, 54, "frame count of the reference logits");
  assert(fixture.lab.endsWith("\n"), "the reference .lab does not end with a newline");
  assert(fixture.lab.trimEnd().split("\n").length >= 2, "the reference .lab has only one line");
  for (const row of fixture.melBasis) assertEquals(row.length, MEL_BINS);
  for (const row of fixture.features) assertEquals(row.length, FEATURE_DIM);
  for (const row of fixture.logits) assertEquals(row.length, LIPSYNC_CLASSES.length);
});

test("the training-time constants match upstream's distributed config", () => {
  // The module constants copy Python's module constants. If retraining changes the geometry, this
  // fails before the features become "another plausible value".
  const { constants } = fixture;
  assertEquals(SAMPLE_RATE, constants.sampleRate);
  assertEquals(N_FFT, constants.nFft);
  assertEquals(WIN_LENGTH, constants.winLength);
  assertEquals(HOP, constants.hop);
  assertEquals(N_MELS, constants.nMels);
  assertEquals(DSP_WINDOW, constants.dspWindow);
  assertEquals(FEATURE_DIM, constants.featureDim);
  assertEquals(MEL_BINS, constants.nFft / 2 + 1);
  assertEquals(FRAME_SEC, constants.frameSec);
  assertEquals([...LIPSYNC_CLASSES], constants.classes);
  // The two smoothing constants are private to the module, so their values are pinned by the
  // behaviour tests.
  assertEquals(constants.switchPenalty, 4);
  assertEquals(constants.minDurationFrames, 2);
});

// ---- (1) Parity of the feature extraction -----------------------------------

test("feature parity: the same waveform yields the same 83 dimensions as Python", () => {
  const { data, frames } = extractFeatures(audio, melBasis);
  assertEquals(frames, fixture.features.length);
  assertEquals(data.length, frames * FEATURE_DIM);
  let worst = 0;
  let worstAt = "";
  let nonFinite = 0;
  for (let frame = 0; frame < frames; frame += 1) {
    for (let dim = 0; dim < FEATURE_DIM; dim += 1) {
      const got = data[frame * FEATURE_DIM + dim];
      const golden = fixture.features[frame][dim];
      if (!Number.isFinite(got) || !Number.isFinite(golden)) {
        nonFinite += 1;
        continue;
      }
      const diff = Math.abs(got - golden);
      if (diff > worst) {
        worst = diff;
        worstAt = `frame ${frame} / dim ${dim}`;
      }
    }
  }
  // MUST: NaN / ±Inf fail. A naive difference check turns NaN into "a difference of 0" and passes.
  assertEquals(nonFinite, 0, "non-finite samples in the features");
  assert(worst <= FEATURE_ATOL, `feature max abs diff ${worst.toExponential(3)} (${worstAt})`);
});

test("features: the frame count is floor((n − N_FFT) / HOP) + 1 (librosa's center=False)", () => {
  const at = (samples: number): number =>
    extractFeatures(new Float32Array(samples), melBasis).frames;
  assertEquals(at(N_FFT), 1, "exactly N_FFT gives 1 frame");
  assertEquals(at(N_FFT + HOP - 1), 1, "one sample short of a hop adds no frame");
  assertEquals(at(N_FFT + HOP), 2);
  assertEquals(at(SAMPLE_RATE), Math.floor((SAMPLE_RATE - N_FFT) / HOP) + 1);
});

test("features: a waveform too short for a single frame fails loudly (no NaN spread)", () => {
  // The waveform is the caller's own argument, so this is an input error (ADR 0107 decision 2 —
  // passing a longer one succeeds).
  assertThrows(
    () => extractFeatures(new Float32Array(N_FFT - 1), melBasis),
    ModelInputError,
    "the waveform is only",
  );
});

test("features: a non-finite sample fails as an input error naming its position and value (no NaN features reach the GPU)", () => {
  // A single NaN makes the mean of the per-utterance z-normalisation NaN and taints all 83
  // dimensions. The waveform is the caller's own argument, so this is an input error (ADR 0107
  // decision 2). Placing it at the end also catches a check that "looks only at the start".
  for (const value of [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]) {
    const tainted = Float32Array.from(audio);
    const at = tainted.length - 1;
    tainted[at] = value;
    assertThrows(
      () => extractFeatures(tainted, melBasis),
      ModelInputError,
      `waveform sample ${at} (${value}) is non-finite`,
    );
  }
});

test("features: a mel basis with the wrong element count is not accepted", () => {
  assertThrows(
    () => extractFeatures(audio, new Float32Array(N_MELS * MEL_BINS - 1)),
    Error,
    "the mel basis has",
  );
  assertThrows(
    () => extractFeatures(audio, new Float32Array(MEL_BINS * N_MELS + 1)),
    Error,
    "the mel basis has",
  );
  // The other side: the mel basis comes from the asset, so fixing the waveform does not fix it =
  // not an input error.
  const basis = assertThrows(() => extractFeatures(audio, new Float32Array(N_MELS * MEL_BINS - 1)));
  assert(!(basis instanceof ModelInputError));
});

test("features: ZCR counts −0 as negative, like np.signbit", () => {
  // Alternating +0 / −0 gives zero crossings with a `value < 0` implementation, and a crossing at
  // every sample boundary with `np.signbit`. The log-mel degenerates to all zeros, but column 82
  // alone splits.
  const alternating = new Float32Array(N_FFT);
  for (let index = 0; index < alternating.length; index += 1) {
    alternating[index] = index % 2 === 0 ? 0 : -0;
  }
  const { data } = extractFeatures(alternating, melBasis);
  assertEquals(data[N_MELS + 2], 1, "a crossing at every sample boundary (ZCR = 1)");

  const positive = new Float32Array(N_FFT);
  assertEquals(
    extractFeatures(positive, melBasis).data[N_MELS + 2],
    0,
    "all +0 gives zero crossings",
  );
});

// ---- (2) Parity of the postprocessing ---------------------------------------

test("postprocessing parity: the same logits yield the same .lab as Python (exact match)", () => {
  const logits = Float32Array.from(fixture.logits.flat());
  assertEquals(toLab(logitsToSegments(logits, fixture.logits.length)), fixture.lab);
});

test("postprocessing: logits whose element count does not match the frame count are not accepted", () => {
  assertThrows(() => logitsToSegments(new Float32Array(8 * 3 + 1), 3), Error, "the logits have");
  assertThrows(() => logitsToSegments(new Float32Array(0), 0), Error, "frame count");
});

test("postprocessing: a non-finite logit fails naming its coordinates (it does not turn into a .lab that is all a)", () => {
  // Without the check, NaN / +Infinity make that row all NaN in log_softmax, nearly every Viterbi
  // comparison becomes false, and a `.lab` that is **well-formed** with "the whole utterance is a" is
  // returned (the numeric fault surfaces nowhere). Pinned: that it fails, and that the coordinates
  // are named.
  const finite = Float32Array.from(fixture.logits.flat());
  // Counterpart: the same array untainted returns the reference `.lab` (the check does not always
  // fail).
  assertEquals(toLab(logitsToSegments(finite, fixture.logits.length)), fixture.lab);
  const at = 7 * LIPSYNC_CLASSES.length + 5; // frame 7 / class 5 = "N"
  for (const [poison, shown] of [
    [NaN, "NaN"],
    [Number.POSITIVE_INFINITY, "Infinity"],
  ] as const) {
    const logits = finite.slice();
    logits[at] = poison;
    assertThrows(
      () => logitsToSegments(logits, fixture.logits.length),
      Error,
      `logitsToSegments: logits frame 7 / class 5 (N) is non-finite (${shown})`,
    );
  }
});

test("Viterbi: a non-finite log posterior fails naming its coordinates", () => {
  // `viterbiSmooth` is public and callable on its own, so it is guarded here too, apart from the
  // entry of `logitsToSegments` (−Infinity looks like "an impossible class" but freezes the whole
  // lattice).
  const frames = 5;
  const logProbabilities = new Float64Array(frames * LIPSYNC_CLASSES.length);
  assertEquals(viterbiSmooth(logProbabilities, frames).length, frames, "all finite passes");
  logProbabilities[3 * LIPSYNC_CLASSES.length + 6] = Number.NEGATIVE_INFINITY;
  assertThrows(
    () => viterbiSmooth(logProbabilities, frames),
    Error,
    "viterbiSmooth: log posteriors frame 3 / class 6 (pau) is non-finite (-Infinity)",
  );
});

// ---- (3) Behaviour of the smoothing and segmentation ------------------------

/**
 * Logits with +20 on only the given class in each frame (`overrides` are individual overrides).
 * 20 is well above the switch penalty 4, so Viterbi follows the argmax.
 */
const logitsFor = (
  sequence: readonly (typeof LIPSYNC_CLASSES)[number][],
  overrides: Readonly<Record<number, Readonly<Record<string, number>>>> = {},
): Float32Array<ArrayBuffer> => {
  const classes = LIPSYNC_CLASSES.length;
  const logits = new Float32Array(sequence.length * classes);
  for (let frame = 0; frame < sequence.length; frame += 1) {
    logits[frame * classes + LIPSYNC_CLASSES.indexOf(sequence[frame])] = 20;
    for (const [label, value] of Object.entries(overrides[frame] ?? {})) {
      logits[frame * classes + LIPSYNC_CLASSES.indexOf(label as (typeof LIPSYNC_CLASSES)[number])] =
        value;
    }
  }
  return logits;
};

const labFor = (
  sequence: readonly (typeof LIPSYNC_CLASSES)[number][],
  overrides?: Readonly<Record<number, Readonly<Record<string, number>>>>,
): string => toLab(logitsToSegments(logitsFor(sequence, overrides), sequence.length));

test("Viterbi: a one-frame wobble below the switch penalty does not switch", () => {
  // Class 0 gets +3 in every frame, class 1 gets +3 only in frame 2. Switching there and back costs 8,
  // which a gain of 3 does not pay for = all 0, not the argmax (0,0,1,0,0).
  const frames = 5;
  const logProbabilities = new Float64Array(frames * LIPSYNC_CLASSES.length);
  for (let frame = 0; frame < frames; frame += 1) {
    logProbabilities[frame * LIPSYNC_CLASSES.length + (frame === 2 ? 1 : 0)] = 3;
  }
  assertEquals([...viterbiSmooth(logProbabilities, frames)], [0, 0, 0, 0, 0]);
});

test("Viterbi: a lasting lead above the penalty switches", () => {
  const frames = 5;
  const logProbabilities = new Float64Array(frames * LIPSYNC_CLASSES.length);
  for (let frame = 0; frame < frames; frame += 1) {
    logProbabilities[frame * LIPSYNC_CLASSES.length + (frame < 2 ? 0 : 1)] = 20;
  }
  assertEquals([...viterbiSmooth(logProbabilities, frames)], [0, 0, 1, 1, 1]);
});

test("Viterbi: a lattice of the wrong shape is not accepted", () => {
  assertThrows(
    () => viterbiSmooth(new Float64Array(8 * 4 - 1), 4),
    Error,
    "the log posteriors have",
  );
});

test("short-segment merge: a segment below the minimum duration is relabelled to the neighbour with the higher posterior", () => {
  // In frame 2, i stands out (Viterbi passes through i), but it lasts only one frame, so it goes to a
  // neighbour. Of the neighbouring candidates a / e, e has the higher posterior in frame 2 and wins (it
  // does not fall into taking the first on a tie).
  assertEquals(
    labFor(["a", "a", "i", "e", "e", "e"], { 2: { e: 1 } }),
    "0.0000000 0.0400000 a\n0.0400000 0.1200000 e\n",
  );
});

test("short-segment merge: pau is kept even below the minimum duration", () => {
  // A pau of one frame (20ms). Relabelling it removes the silence and the mouth does not close = a
  // perceptible degradation.
  assertEquals(
    labFor(["a", "a", "pau", "i", "i"]),
    "0.0000000 0.0400000 a\n0.0400000 0.0600000 pau\n0.0600000 0.1000000 i\n",
  );
});

test("cons absorption: followed by a vowel, the vowel segment's start moves earlier to fold it in", () => {
  assertEquals(labFor(["cons", "cons", "a", "a"]), "0.0000000 0.0800000 a\n");
});

test("cons absorption: not followed by a vowel/N, the preceding segment is extended", () => {
  assertEquals(
    labFor(["a", "a", "cons", "cons", "pau", "pau"]),
    "0.0000000 0.0800000 a\n0.0800000 0.1200000 pau\n",
  );
});

test("cons absorption: a cons between two segments of the same label joins them into one", () => {
  // Starting a new segment at the moved-up start would split a in two (although the mouth shape has
  // not changed).
  assertEquals(labFor(["a", "a", "cons", "cons", "a", "a"]), "0.0000000 0.1200000 a\n");
});

test("cons absorption: with nothing to absorb into on either side, it becomes pau", () => {
  assertEquals(labFor(["cons", "cons"]), "0.0000000 0.0400000 pau\n");
});

test("toLab: seconds have a fixed 7 decimal places, and each line ends with a newline", () => {
  assertEquals(
    toLab([
      { start: 0, end: 0.02, label: "a" },
      { start: 0.02, end: 1 / 3, label: "pau" },
    ]),
    "0.0000000 0.0200000 a\n0.0200000 0.3333333 pau\n",
  );
  assertEquals(toLab([]), "");
});
