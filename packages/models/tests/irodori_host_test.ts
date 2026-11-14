// Behaviour tests of Irodori's host layer (the pure functions of `src/irodori/host/`). Needs
// neither a GPU nor real assets.
//
// This covers the "kept on the host" part of ADR 0047 decision 4 — deciding S, segment masks, the t
// schedule, CFG combination, Euler updates, patching the reference latent, reductions and building
// token sequences. Mistakes show up as "a different sound, not an exception", so the values
// themselves are pinned.
//
// Numeric parity (the golden comparison of the `t_embed` table) needs assets, so it lives in
// `irodori_t_embed_test.ts`.

import {
  assert,
  assertEquals,
  assertNotStrictEquals,
  assertStrictEquals,
  assertThrows,
} from "@std/assert";
import { describe, it, test } from "vite-plus/test";
import { ModelInputError } from "../src/errors.ts";
import { conditionFromStateOverride, patchRequestedLatent } from "../src/irodori/conditioning.ts";
import { buildDitMask, SEGMENT_ORDER } from "../src/irodori/host/mask.ts";
import { packCaptionIds, packIds } from "../src/irodori/host/pack.ts";
import { patchReferenceLatent } from "../src/irodori/host/patch.ts";
import { prependMeanToken, rowMean } from "../src/irodori/host/pooling.ts";
import { Randn } from "../src/irodori/host/random.ts";
import {
  bankerRound,
  maxSequenceLength,
  type SampleBounds,
  sequenceLengthFromLogFrames,
  sequenceLengthFromSeconds,
} from "../src/irodori/host/round.ts";
import { combineCfg, eulerStep, tSchedule } from "../src/irodori/host/sampler.ts";
import { IrodoriTokenizer, type IrodoriTokenizerAssets } from "../src/irodori/text/tokenizer.ts";

/** The values used with the real v4-small weights (the same numbers `pipelineConfig` carries). */
const BOUNDS: SampleBounds = {
  frameRate: 25,
  minSeconds: 0.5,
  maxSeconds: 30,
  sampleRate: 48000,
  hopLength: 1920,
};

// ---- Deciding S -------------------------------------------------------------

test("bankerRound: exactly 0.5 goes to the even side (where it differs from Math.round)", () => {
  assertEquals(bankerRound(0.5), 0);
  assertEquals(bankerRound(1.5), 2);
  assertEquals(bankerRound(2.5), 2);
  assertEquals(bankerRound(3.5), 4);
  // Anything not at 0.5 goes to the nearest (same as Math.round).
  assertEquals(bankerRound(2.4), 2);
  assertEquals(bankerRound(2.6), 3);
  // The negative side also goes to even (same as Python's round).
  assertEquals(bankerRound(-2.5), -2);
  assertEquals(bankerRound(-3.5), -4);
});

test("sequenceLengthFromLogFrames: the golden 2 cases' log frames give the same S", () => {
  // The values are `duration.logFrames` of the golden
  // `goldens/irodori-v4-small/pipeline/meta.json` (full = 161 frames / no-ref = 116 frames).
  assertEquals(sequenceLengthFromLogFrames(5.087219, BOUNDS).frames, 161);
  assertEquals(sequenceLengthFromLogFrames(4.761631, BOUNDS).frames, 116);
});

test("sequenceLengthFromLogFrames: the clamp is 13 (0.5s) and 750 (30s)", () => {
  // ceil(0.5×25) = 13 / floor(30×25) = 750 — the asymmetric rounding at the ends follows upstream's
  // spelling.
  assertEquals(
    sequenceLengthFromLogFrames(0, BOUNDS).frames,
    13,
    "expm1(0)=0 goes to the lower bound",
  );
  assertEquals(
    sequenceLengthFromLogFrames(10, BOUNDS).frames,
    750,
    "expm1(10)≈22025 goes to the upper bound",
  );
});

test("sequenceLengthFromLogFrames: this path's cut length is exactly S × hopLength", () => {
  // The S that duration decided is the output length as is — the cut is effectively a no-op.
  assertEquals(sequenceLengthFromLogFrames(5.087219, BOUNDS).targetSamples, 161 * 1920);
});

test("sequenceLengthFromSeconds: seconds → samples → frames (the spelling of upstream's manual_seconds)", () => {
  assertEquals(sequenceLengthFromSeconds(1, BOUNDS), { frames: 25, targetSamples: 48000 });
  // Remainders round up (no utterance one frame short). 1.004s = 48,192 samples → 26 frames.
  assertEquals(sequenceLengthFromSeconds(1.004, BOUNDS), { frames: 26, targetSamples: 48192 });
  assertEquals(
    sequenceLengthFromSeconds(0.1, BOUNDS),
    {
      frames: 13,
      targetSamples: 24000,
    },
    "clamped to the lower bound 0.5s",
  );
  assertEquals(
    sequenceLengthFromSeconds(100, BOUNDS),
    {
      frames: 750,
      targetSamples: 1_440_000,
    },
    "clamped to the upper bound 30s",
  );
});

test("sequenceLengthFromSeconds: the 1-frame difference at a sample boundary (where it parts from the frameRate path)", () => {
  // 1.000005s × 48,000 = 48,000.24 samples → truncated to 48,000 → 25 frames.
  // Through frameRate (ceil(1.000005×25) = 26) the utterance would be 1 frame longer.
  assertEquals(sequenceLengthFromSeconds(1.000005, BOUNDS), {
    frames: 25,
    targetSamples: 48000,
  });
  // Exceeding by even 1 sample carries up (checking the rounding direction is not reversed).
  assertEquals(sequenceLengthFromSeconds(1.0000209, BOUNDS), {
    frames: 26,
    targetSamples: 48001,
  });
});

test("sequenceLengthFromSeconds: the cut length can be shorter than S × hopLength", () => {
  // 26 frames = 49,920 samples are decoded, but the returned waveform is 48,192 samples.
  const plan = sequenceLengthFromSeconds(1.004, BOUNDS);
  assertEquals(plan.frames * BOUNDS.hopLength - plan.targetSamples, 1728);
});

test("bankerRound: rejects non-finite values (a value with nowhere to round to does not pass silently)", () => {
  for (const value of [NaN, Infinity, -Infinity]) {
    assertThrows(() => bankerRound(value), RangeError, "is not a finite number");
  }
});

test("sequenceLengthFromLogFrames: rejects output whose expm1 is not finite (not hidden by the clamp)", () => {
  // When the duration graph outputs a broken value, a clamp that silently pushes it to the upper
  // bound (750) would only "produce 30 seconds of audio" and the real cause would vanish. f32 expm1
  // becomes Infinity at logFrames=1e5.
  assertThrows(() => sequenceLengthFromLogFrames(1e5, BOUNDS), Error, "is not finite");
});

test("sequenceLengthFromSeconds: rejects non-finite seconds", () => {
  for (const seconds of [NaN, Infinity]) {
    assertThrows(
      () => sequenceLengthFromSeconds(seconds, BOUNDS),
      ModelInputError,
      "durationSeconds",
    );
  }
});

test("maxSequenceLength: the reachable S is the larger of the 2 paths", () => {
  // The surface for checking at parse time that the distribution's declaration (`maxSeconds`) and
  // `ditSymMax` agree. The rounding differs per path, so looking only at the duration path (floor)
  // misses the manual-seconds path (ceil).
  assertEquals(
    maxSequenceLength(BOUNDS),
    750,
    "the real distribution's 30 seconds is 750 on both paths",
  );
  assertEquals(
    maxSequenceLength({ ...BOUNDS, maxSeconds: 30.02 }),
    751,
    "the duration path stays at 750; only the manual-seconds path comes out at 751",
  );
});

// ---- Segment masks -----------------------------------------------------------

/**
 * The same values as `tools/export-recipes/irodori/tests/test_pipeline_ref.py::TestSegmentMasks`.
 */
const CAPS = { text: 4, speaker: 5, caption: 6 };
const USED = { text: 2, speaker: 3, caption: 4 };

test("buildDitMask: the concatenation order is self → text → speaker → caption, each segment a prefix of its used length", () => {
  const mask = buildDitMask(3, USED, CAPS);
  assertEquals(mask.length, 3 + 4 + 5 + 6);
  assertEquals([...mask.slice(0, 3)], [1, 1, 1], "self is all 1 for the latent length");
  assertEquals([...mask.slice(3, 7)], [1, 1, 0, 0], "text");
  assertEquals([...mask.slice(7, 12)], [1, 1, 1, 0, 0], "speaker");
  assertEquals([...mask.slice(12)], [1, 1, 1, 1, 0, 0], "caption");
});

test("buildDitMask: an uncond variant drops only its own segment (ADR 0047 decision 1)", () => {
  const cond = buildDitMask(3, USED, CAPS);
  const offsets = { text: 3, speaker: 7, caption: 12 };
  for (const segment of SEGMENT_ORDER) {
    const got = buildDitMask(3, USED, CAPS, segment);
    const start = offsets[segment];
    const end = start + CAPS[segment];
    assertEquals(
      [...got.slice(start, end)],
      new Array(CAPS[segment]).fill(0),
      `${segment} segment`,
    );
    assertEquals([...got.slice(0, start)], [...cond.slice(0, start)], `before ${segment}`);
    assertEquals([...got.slice(end)], [...cond.slice(end)], `after ${segment}`);
  }
});

test("buildDitMask: rejects a used length beyond the declared length (a broken right pad)", () => {
  assertThrows(
    () => buildDitMask(3, { ...USED, speaker: 6 }, CAPS),
    Error,
    "speaker segment: used length 6",
  );
});

// ---- The t schedule and the CFG window -------------------------------------------

test("tSchedule: length steps+1, starts at 0.999, ends at 0, strictly decreasing", () => {
  const schedule = tSchedule(40, 0.999);
  assertEquals(schedule.length, 41);
  assertEquals(schedule[0], Math.fround(0.999));
  assertEquals(schedule[40], 0);
  for (let index = 1; index <= 40; index += 1) {
    assert(schedule[index] < schedule[index - 1], `t does not decrease (index=${index})`);
  }
});

test("tSchedule: 20 of the 40 steps fall in the CFG window [0.5, 1.0]", () => {
  // This is where cond 40 + uncond 20×(number of active conditions) = 100 / 60 forwards comes from
  // (meta.json). The multiplication side (`40 + 20 * 3 === 100`) is not placed here — a tautology
  // between literals that calls not a single line of code. The real forward count is pinned by
  // `latent.forwards` of `e2e_irodori_latent_test.ts` and the golden comparison.
  const schedule = tSchedule(40, 0.999);
  const inWindow = [...schedule.slice(0, 40)].filter((t) => t >= 0.5 && t <= 1);
  assertEquals(inWindow.length, 20);
});

// ---- CFG combination and Euler updates -----------------------------------------------

test("combineCfg: with 0 variants it is a copy of cond (not the same array)", () => {
  const cond = Float32Array.from([1, -2, 0.5]);
  const combined = combineCfg(cond, []);
  assertEquals([...combined], [1, -2, 0.5]);
  assertNotStrictEquals(
    combined,
    cond,
    "does not enter the next run still holding the Session's output buffer",
  );
});

test("combineCfg: accumulates v_cond + Σ s_k(v_cond − v_k) in order", () => {
  const cond = Float32Array.from([2]);
  const text = { scale: 3, velocity: Float32Array.from([1]) };
  const caption = { scale: 5, velocity: Float32Array.from([1.5]) };
  // 2 + 3·(2−1) = 5 → 5 + 5·(2−1.5) = 7.5
  assertEquals([...combineCfg(cond, [text, caption])], [7.5]);
});

test("combineCfg: fault injection — the reference of each difference is always v_cond (not the previous combined result)", () => {
  // Miswritten as "v = v + s(v − v_k)", the second difference would be taken against the already
  // combined value.
  const cond = Float32Array.from([2]);
  const variants = [
    { scale: 1, velocity: Float32Array.from([0]) },
    { scale: 1, velocity: Float32Array.from([0]) },
  ];
  // Right: 2 + (2−0) + (2−0) = 6 / wrong: 2 + (2−0) = 4 → 4 + (4−0) = 8
  assertEquals([...combineCfg(cond, variants)], [6]);
});

test("combineCfg: rejects mismatched lengths", () => {
  assertThrows(
    () => combineCfg(Float32Array.from([1, 2]), [{ scale: 1, velocity: Float32Array.from([1]) }]),
    Error,
    "CFG variant length 1",
  );
});

test("eulerStep: x + v·Δt (Δt is negative)", () => {
  const x = Float32Array.from([1, 2]);
  const v = Float32Array.from([0.5, -4]);
  assertEquals([...eulerStep(x, v, -0.5)], [0.75, 4]);
});

test("eulerStep: rejects a velocity field of mismatched length (the counterpart of the same test on the combineCfg side)", () => {
  // Passing a shorter one through makes reading `undefined` give NaN that propagates, and the
  // diagnosis turns up as the waveform's non-finite check. With a longer one the remainder is
  // silently dropped.
  assertThrows(
    () => eulerStep(Float32Array.from([0, 0]), Float32Array.from([1]), -0.1),
    Error,
    "Velocity field length 1",
  );
});

test("tSchedule: rejects steps that are not an integer of at least 1", () => {
  for (const steps of [0, -1, 2.5]) {
    assertThrows(() => tSchedule(steps, 0.999), RangeError, "is not an integer >= 1");
  }
});

// ---- Patching and reducing the reference latent -----------------------------------------

test("patchReferenceLatent: drops the leftover frames and reshapes", () => {
  // latentDim=2 / patchSize=4 → of 9 frames only 8 (2 tokens) remain.
  const latent = Float32Array.from({ length: 9 * 2 }, (_value, index) => index);
  const patched = patchReferenceLatent(latent, 2, 4);
  assertEquals(patched.tokens, 2);
  assertEquals(patched.width, 8);
  assertEquals(patched.data.length, 16);
  assertEquals([...patched.data.slice(0, 8)], [0, 1, 2, 3, 4, 5, 6, 7]);
  assertEquals(patched.data[15], 15, "only the 9th frame is dropped");
});

test("patchReferenceLatent: rejects a reference below the lower bound (0 tokens) and a broken shape", () => {
  assertThrows(() => patchReferenceLatent(new Float32Array(3 * 2), 2, 4), Error, "is less than");
  assertThrows(
    () => patchReferenceLatent(new Float32Array(5), 2, 4),
    Error,
    "is not a multiple of",
  );
  // The other side: this function also receives the codec encoder's output, so a breakage stays an
  // internal invariant (a function that does not know where its input came from claiming an
  // input-caused failure would turn a model-output fault into a 400 — ADR 0107 decision 2).
  const shortcut = assertThrows(() => patchReferenceLatent(new Float32Array(5), 2, 4));
  assert(!(shortcut instanceof ModelInputError));
});

test("patchReferenceLatent: rejects a reference below speaker's lower bound of 2 tokens (the export's Dim(min=2))", () => {
  // At patch width 4, 4–7 frames are 1 token. Passed through, S = 1, outside the symbolic
  // dimension's lower bound, would reach the speaker graph.
  for (const frames of [4, 7]) {
    const error = assertThrows(
      () => patchReferenceLatent(new Float32Array(frames * 2), 2, 4),
      Error,
      "is less than",
    );
    assert(
      !(error instanceof ModelInputError),
      "a breakage in the codec encoder's output stays an internal invariant",
    );
  }
  assertEquals(
    patchReferenceLatent(new Float32Array(8 * 2), 2, 4).tokens,
    2,
    "exactly at the lower bound",
  );
});

// ---- Specifying speaker directly (input-caused per ADR 0107 decision 2) -----------------------

describe("the accepted set when speaker is specified directly", () => {
  // Uses a scaled-down version of the same relation as the real v4-small weights (speakerRows−1
  // tokens + 1 mean token).
  const SHAPE = { speakerDim: 4, speakerRows: 3, latentDim: 2, speakerPatchSize: 4 } as const;

  it("stateOverride accepts only positive multiples of speakerDim within the declared length", () => {
    assertEquals(conditionFromStateOverride(new Float32Array(2 * 4), SHAPE).rows, 2);
    // Exactly the row count passes (the counterpart showing the check is not too wide).
    assertEquals(conditionFromStateOverride(new Float32Array(3 * 4), SHAPE).rows, 3);
    assertThrows(
      () => conditionFromStateOverride(new Float32Array(0), SHAPE),
      ModelInputError,
      "is not a positive multiple of",
    );
    assertThrows(
      () => conditionFromStateOverride(new Float32Array(5), SHAPE),
      ModelInputError,
      "is not a positive multiple of",
    );
    // A row count beyond the declared length fails here rather than being carried to `rightPad`.
    assertThrows(
      () => conditionFromStateOverride(new Float32Array(4 * 4), SHAPE),
      ModelInputError,
      "makes 4 rows of speaker condition, exceeding the declared length 3",
    );
  });

  it("stateOverride becomes the condition as is, without processing (no double normalization)", () => {
    const given = Float32Array.from({ length: 2 * 4 }, (_value, index) => index);
    assertStrictEquals(conditionFromStateOverride(given, SHAPE).data, given);
  });

  it("latent fails as input-caused both for a broken shape and for too many rows (the owner's diagnosis stays in cause)", () => {
    // 2 tokens + 1 mean = 3 rows is exactly the declared length.
    assertEquals(patchRequestedLatent(new Float32Array(8 * 2), SHAPE).tokens, 2);
    const shape = assertThrows(
      () => patchRequestedLatent(new Float32Array(5), SHAPE),
      ModelInputError,
      "is not a multiple of",
    );
    assert(
      shape.cause instanceof Error,
      "the diagnosis of the accepted set's owner is not thrown away",
    );
    assert(!(shape.cause instanceof ModelInputError));
    assertThrows(
      () => patchRequestedLatent(new Float32Array(3 * 2), SHAPE),
      ModelInputError,
      "is less than",
    );
    assertThrows(
      () => patchRequestedLatent(new Float32Array(12 * 2), SHAPE),
      ModelInputError,
      "makes 4 rows of speaker condition, exceeding the declared length 3",
    );
  });

  it("a latent of 4–7 frames (1 token after patching) is outside speaker's lower bound of 2 tokens and input-caused", () => {
    for (const frames of [4, 7]) {
      assertThrows(
        () => patchRequestedLatent(new Float32Array(frames * 2), SHAPE),
        ModelInputError,
        "is less than",
      );
    }
  });
});

test("rowMean: the plain per-column mean (only the all-True mask path is ever taken)", () => {
  const state = Float32Array.from([1, 10, 3, 20]);
  assertEquals([...rowMean(state, 2, 2)], [2, 15]);
});

test("prependMeanToken: the mean token goes **first** (where duration's speaker_vec comes from)", () => {
  const state = Float32Array.from([1, 10, 3, 20]);
  const prepended = prependMeanToken(state, 2, 2);
  assertEquals(prepended.length, 6);
  assertEquals([...prepended.slice(0, 2)], [2, 15], "the first is the mean");
  assertEquals([...prepended.slice(2)], [1, 10, 3, 20], "the rest is the original sequence itself");
});

// ---- Building token sequences ---------------------------------------------------

/**
 * A minimal vocabulary (it exercises only the Unigram lattice — the parity of the real vocabulary
 * is in `irodori_text_test.ts`).
 */
const tinyTokenizer = (): IrodoriTokenizer => {
  const tokens = ["<unk>", "<s>", "</s>", "<pad>", "あ", "い", "う"];
  const vocab = new Map(tokens.map((token, id) => [token, { id, score: -1 }]));
  const assets: IrodoriTokenizerAssets = {
    vocab,
    minScore: -1,
    maxTokenLength: 5,
    unkId: 0,
    byteBaseId: 100,
    bosId: 1,
    padId: 3,
    addedTokens: new Map(),
  };
  return new IrodoriTokenizer(assets);
};

/** The unsplit path (`tinyTokenizer` has no rare-token split table). */
const PLAIN = { splitRareTokens: false } as const;

test("packIds: BOS prefix + no pad (the packed sequence of the static scheme)", () => {
  assertEquals([...packIds(tinyTokenizer(), "あいう", 8, "text", PLAIN)], [1, 4, 5, 6]);
});

test("packIds: the text budget is maxLength−1 (room is left for the BOS)", () => {
  assertEquals([...packIds(tinyTokenizer(), "あいう", 3, "text", PLAIN)], [1, 4, 5]);
});

test("packIds: goes through normalization (ideographic spaces vanish — stage ① of `normalize.ts`)", () => {
  assertEquals([...packIds(tinyTokenizer(), "あ　い", 8, "text", PLAIN)], [1, 4, 5]);
});

test("packIds: rejects a sequence that is empty after normalization (no BOS-only sequence passes)", () => {
  assertThrows(
    () => packIds(tinyTokenizer(), "　 ", 8, "caption", PLAIN),
    ModelInputError,
    "caption is empty after normalisation",
  );
});

/**
 * The ids when expanded with `tinyTokenizer`'s `byteBaseId` (the expected values for
 * out-of-vocabulary characters).
 */
const byteIds = (text: string): number[] => [...new TextEncoder().encode(text)].map((b) => 100 + b);

test("packCaptionIds: applies no normalization (upstream's `_synthesize` only strips the caption)", () => {
  // Outer brackets are something `normalize_text` strips. If they were stripped on the caption
  // side, the conditioning would become something else without any exception or warning (the
  // difference between these 2 calls is exactly that boundary).
  assertEquals(
    [...packCaptionIds(tinyTokenizer(), "「あい」", 16)],
    [1, ...byteIds("「"), 4, 5, ...byteIds("」")],
  );
  assertEquals([...packIds(tinyTokenizer(), "「あい」", 16, "text", PLAIN)], [1, 4, 5]);
});

test("packCaptionIds: only leading and trailing spaces are dropped (an inner ideographic space stays)", () => {
  assertEquals([...packCaptionIds(tinyTokenizer(), " あ　い ", 16)], [1, 4, ...byteIds("　"), 5]);
});

test("packCaptionIds: rejects a caption that is empty after strip (no BOS-only caption goes into the conditions)", () => {
  assertThrows(
    () => packCaptionIds(tinyTokenizer(), " \n\t ", 8),
    Error,
    "caption is empty after strip",
  );
});

// ---- The initial-noise generator ---------------------------------------------------
//
// `host/random.ts` is **a deliberate duplicate** of `sbv2/host/random.ts` (families do not import
// each other — see the module doc). Being a duplicate, it needs a place here where one side moving
// alone does not go unnoticed. It pins the same 3 points as the sbv2 side (same seed, same sequence
// / statistics like a standard normal / no pair carried over).

test("Randn: the same seed gives the same sequence, a different seed a different one", () => {
  const a = new Randn(7).normals(64);
  const b = new Randn(7).normals(64);
  const c = new Randn(8).normals(64);
  assertEquals([...a], [...b]);
  assert(
    [...a].some((value, index) => value !== c[index]),
    "the same sequence even with a different seed",
  );
});

test("Randn: has statistics like a standard normal, and scale acts linearly", () => {
  const samples = new Randn(1).normals(20000);
  const mean = samples.reduce((sum, value) => sum + value, 0) / samples.length;
  const variance = samples.reduce((sum, value) => sum + (value - mean) ** 2, 0) / samples.length;
  assert(Math.abs(mean) < 0.05, `the mean ${mean} is too far from 0`);
  assert(Math.abs(variance - 1) < 0.05, `the variance ${variance} is too far from 1`);
  // scale only multiplies the same sequence (the sequence itself is decided by the seed). This
  // family's calls do not use scale, but a drift of the duplicate shows up only here.
  const plain = new Randn(3).normals(16);
  const scaled = new Randn(3).normals(16, 2);
  for (const [index, value] of plain.entries()) {
    assertEquals(scaled[index], Math.fround(value * 2));
  }
});

test("Randn: does not carry a Box–Muller pair over even at odd lengths", () => {
  // Drawing 5 elements at once consumes 3 pairs: [cos1, sin1, cos2, sin2, cos3]. Split into 3 + 2,
  // the first call consumes 2 whole pairs: [cos1, sin1, cos2] (sin2 is dropped), and the second
  // starts from the 3rd pair: [cos3, sin3]. An implementation that carries over would start the
  // second call with sin2, so that is the observation point.
  const single = new Randn(5).normals(5);
  const split = new Randn(5);
  const head = split.normals(3);
  const tail = split.normals(2);
  assertEquals([...head], [...single.slice(0, 3)], "the same sequence up to the first 3 elements");
  assertEquals(
    tail[0],
    single[4],
    "the second call starts from the next pair's cos (= nothing carried over)",
  );
  assert(
    tail[0] !== single[3],
    "the second call's first value picks up the sin that should have been dropped",
  );
});

// ---- ADR 0107's line (input-caused / internal invariant) --------------------------

describe("irodori's input-caused failures (ADR 0107)", () => {
  // What is pinned is the type itself. A host carrying several families can tell 400 / 500 apart
  // only by `instanceof`, and branching on message strings silently breaks the moment the wording
  // changes.
  it("non-finite durationSeconds is caught as ModelInputError", () => {
    assertThrows(() => sequenceLengthFromSeconds(NaN, BOUNDS), ModelInputError);
  });

  it("text that is empty after normalization is caught as ModelInputError", () => {
    assertThrows(() => packIds(tinyTokenizer(), "　 ", 8, "text", PLAIN), ModelInputError);
  });

  it("an out-of-range seed is caught as ModelInputError on the generator side too", () => {
    // The owner of the accepted set is the one in `request-gates.ts` (ADR 0107 decision 6). If the
    // generator kept its own family-specific copy, only this would stay a `RangeError` and the type
    // would split from the entry's.
    for (const seed of [-1, 1.5, Number.MAX_SAFE_INTEGER + 2]) {
      assertThrows(() => new Randn(seed), ModelInputError, "is not a non-negative safe integer");
    }
  });
});

describe("irodori's internal invariant breakages (the flip side of ADR 0107's line)", () => {
  it("a segment's used length exceeding the declared length is not a ModelInputError", () => {
    // A wiring breakage that the caller cannot fix by fixing the input (= the 500 kind). Making it
    // a `ModelInputError` would make the host return 400 even for failures where "fix your input"
    // is not an answer.
    const error = assertThrows(() => buildDitMask(3, { ...USED, speaker: 6 }, CAPS), Error);
    assert(
      !(error instanceof ModelInputError),
      "an internal invariant breakage is mixed into the input-caused type",
    );
  });
});
