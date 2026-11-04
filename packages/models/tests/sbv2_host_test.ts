// Behaviour tests of the SBV2 host glue (the pure functions of `src/sbv2/host/`) and the packaging
// of its outputs.
//
// The "data-dependent shape" computation between front and voice (durations → frame expansion →
// z_p), and the packaging of the outputs (WAV is the family-independent layer in `src/audio/`; the
// dump is on the example side). This lies outside the golden E2E — no graph is executed, so a
// mistake shows up only as "the audio length changes" or "it goes silent" and is not caught by a
// numeric comparison.

import { test } from "vite-plus/test";
import { assert, assertEquals, assertThrows } from "@std/assert";
import { parseSafetensors } from "@karume/runtime";
import { ModelInputError } from "../src/errors.ts";
import { InputError } from "../src/sbv2/errors.ts";
import { durationsToFrames } from "../src/sbv2/host/duration.ts";
import { buildZp } from "../src/sbv2/host/latent.ts";
import { Randn } from "../src/sbv2/host/random.ts";
import { encodeWav } from "../src/audio/wav.ts";
// Writing the dump is a **development contract** (the transport format handed to the torch
// reference), so it lives in a test-side helper. It is not on the package's public surface, but the
// packaging round trip itself is pinned here.
import { writeSafetensors } from "./helpers/safetensors-write.ts";

/**
 * Inverts `logw = log(w)` (so that tests can write the expected frame counts as values rather than
 * formulas).
 *
 * MUST: do not pass integers as `w`. `exp(fround(log(n)))` can land just above or just below n, so
 * `ceil` wobbles between n and n+1 (a property of the ceil boundary itself, not a defect of the
 * implementation).
 */
const logwFor = (frames: readonly number[]): Float32Array<ArrayBuffer> =>
  Float32Array.from(frames, Math.log);

/**
 * The maximum passed by tests whose subject is not the maximum (4096, as `maxFrames` of the real
 * distribution).
 */
const AMPLE_FRAMES = 4096;

test("durationsToFrames: exp → ceil gives the frame counts, and the expansion index is non-decreasing", () => {
  // sdp and dp are made equal so the mixing can be ignored and only the ceil behaviour is observed.
  const logw = logwFor([1.2, 1.8, 0.3]);
  const plan = durationsToFrames(logw, logw, new Float32Array([1, 1, 1]), 0.5, 1, AMPLE_FRAMES);
  assertEquals([...plan.wCeil], [2, 2, 1]);
  assertEquals(plan.totalFrames, 5);
  assertEquals([...plan.expandIdx], [0, 0, 1, 1, 2]);
});

test("durationsToFrames: a phoneme whose x_mask is 0 gets 0 frames", () => {
  const logw = logwFor([1.2, 5.4]);
  const plan = durationsToFrames(logw, logw, new Float32Array([1, 0]), 0, 1, AMPLE_FRAMES);
  assertEquals([...plan.wCeil], [2, 0]);
  assertEquals([...plan.expandIdx], [0, 0]);
});

test("durationsToFrames: length_scale scales the frame counts proportionally", () => {
  const logw = logwFor([1.2, 1.2]);
  const mask = new Float32Array([1, 1]);
  assertEquals([...durationsToFrames(logw, logw, mask, 0, 1, AMPLE_FRAMES).wCeil], [2, 2], "w=1.2");
  assertEquals([...durationsToFrames(logw, logw, mask, 0, 2, AMPLE_FRAMES).wCeil], [3, 3], "w=2.4");
});

test("durationsToFrames: fault injection — the sdp_ratio mixing takes effect", () => {
  // Pass different values for sdp and dp and check that changing the ratio changes the result. An
  // implementation that uses only one of them (the mixing forgotten) gives the same answer for any
  // ratio.
  const sdp = logwFor([9.5, 9.5]);
  const dp = logwFor([1.5, 1.5]);
  const mask = new Float32Array([1, 1]);
  assertEquals(
    [...durationsToFrames(sdp, dp, mask, 0, 1, AMPLE_FRAMES).wCeil],
    [2, 2],
    "r=0 is dp only",
  );
  assertEquals(
    [...durationsToFrames(sdp, dp, mask, 1, 1, AMPLE_FRAMES).wCeil],
    [10, 10],
    "r=1 is sdp only",
  );
  // The mixing happens on the logw (log) side, so a middle ratio gives the geometric mean
  // √(9.5·1.5) ≈ 3.775.
  const mixed = durationsToFrames(sdp, dp, mask, 0.5, 1, AMPLE_FRAMES).wCeil;
  assertEquals([...mixed], [4, 4]);
});

test("durationsToFrames: rejects a length mismatch and all-zero frames", () => {
  const logw = logwFor([1, 1]);
  // Both mean the front output is broken = an internal fault, so they stay plain `Error`
  // (500-like). The caller cannot fix them by changing text or lengthScale = they must not go to
  // the input-error branch.
  const mismatch = assertThrows(
    () => durationsToFrames(logw, logwFor([1]), new Float32Array([1, 1]), 0, 1, AMPLE_FRAMES),
    Error,
    "Length mismatch",
  );
  assert(!(mismatch instanceof InputError), "a length mismatch is not an input error");
  const empty = assertThrows(
    () => durationsToFrames(logw, logw, new Float32Array([0, 0]), 0, 1, AMPLE_FRAMES),
    Error,
    "total frame count is 0",
  );
  assert(!(empty instanceof InputError), "a total frame count of 0 is not an input error");
});

test("durationsToFrames: a total frame count above the distribution's maximum fails before the expansion index is allocated", () => {
  // For the same text, raising lengthScale grows the total frame count without bound (5 → 8). A
  // request above the maximum stops here, allocating neither the length-Ty expansion index nor the
  // downstream T×T table (on the order of 8·T² bytes).
  const logw = logwFor([1.2, 1.8, 0.3]);
  const mask = new Float32Array([1, 1, 1]);
  assertEquals(durationsToFrames(logw, logw, mask, 0.5, 1, 7).totalFrames, 5);
  // The type is `InputError` (400-like) — the request goes through if the text is split shorter or
  // lengthScale is lowered, so the caller branches differently than for a broken internal invariant
  // (plain `Error` = 500) (the classification axis of `errors.ts`).
  assertThrows(
    () => durationsToFrames(logw, logw, mask, 0.5, 2, 7),
    InputError,
    "The total frame count 8 exceeds the distribution's maximum maxFrames=7",
  );
});

test("durationsToFrames: exactly the maximum passes (the boundary is not one too far inside)", () => {
  const logw = logwFor([1.2, 1.8, 0.3]);
  const plan = durationsToFrames(logw, logw, new Float32Array([1, 1, 1]), 0.5, 2, 8);
  assertEquals(plan.totalFrames, 8);
  assertEquals(plan.expandIdx.length, 8, "an expansion index of exactly the maximum is allocated");
});

test("buildZp: expands m_p and adds noise·exp(logs_p)·scale", () => {
  // C=2 / P=2 / Ty=3. logs_p = 0, so exp is 1 and the expected values can be written by hand.
  const mP = new Float32Array([1, 2, 10, 20]);
  const logsP = new Float32Array([0, 0, 0, 0]);
  const expandIdx = Int32Array.from([0, 1, 1]);
  const noise = new Float32Array([1, 0, -1, 2, 0, -2]);
  const zP = buildZp(mP, logsP, expandIdx, 2, noise, 0.5);
  assertEquals([...zP], [1 + 0.5, 2, 2 - 0.5, 10 + 1, 20, 20 - 1]);
});

test("buildZp: fault injection — a shifted expansion index changes the values", () => {
  // An implementation that ignores the gather and slices the first Ty columns returns the same
  // answer for any expandIdx.
  const mP = new Float32Array([1, 2, 3]);
  const logsP = new Float32Array([0, 0, 0]);
  const noise = new Float32Array(3);
  const a = buildZp(mP, logsP, Int32Array.from([0, 1, 2]), 1, noise, 1);
  const b = buildZp(mP, logsP, Int32Array.from([0, 0, 2]), 1, noise, 1);
  assertEquals(a.length, b.length, "the lengths are equal (= a fault the length cannot detect)");
  assert(
    [...a].some((value, index) => value !== b[index]),
    "the values are the same even with a different expansion",
  );
});

test("buildZp: rejects mismatched shapes", () => {
  const mP = new Float32Array([1, 2, 3, 4]);
  const logsP = new Float32Array([0, 0, 0, 0]);
  const idx = Int32Array.from([0, 1]);
  assertThrows(() => buildZp(mP, logsP, idx, 3, new Float32Array(6), 1), Error, "not divisible");
  assertThrows(() => buildZp(mP, logsP, idx, 2, new Float32Array(3), 1), Error, "Noise length");
  assertThrows(
    () => buildZp(mP, logsP, Int32Array.from([0, 9]), 2, new Float32Array(4), 1),
    Error,
    "Expansion index",
  );
});

test("Randn: the same seed gives the same sequence and a different seed a different one", () => {
  const a = new Randn(7).normals(64);
  const b = new Randn(7).normals(64);
  const c = new Randn(8).normals(64);
  assertEquals([...a], [...b]);
  assert(
    [...a].some((value, index) => value !== c[index]),
    "the same sequence even with a different seed",
  );
});

test("Randn: has standard-normal statistics, and scale acts linearly", () => {
  const samples = new Randn(1).normals(20000);
  const mean = samples.reduce((sum, value) => sum + value, 0) / samples.length;
  const variance = samples.reduce((sum, value) => sum + (value - mean) ** 2, 0) / samples.length;
  assert(Math.abs(mean) < 0.05, `mean ${mean} is too far from 0`);
  assert(Math.abs(variance - 1) < 0.05, `variance ${variance} is too far from 1`);
  // scale only multiplies the same sequence (the sequence itself is determined by the seed).
  const plain = new Randn(3).normals(16);
  const scaled = new Randn(3).normals(16, 2);
  for (const [index, value] of plain.entries()) {
    assertEquals(scaled[index], Math.fround(value * 2));
  }
});

test("Randn: does not carry a Box–Muller pair over, even for odd lengths", () => {
  // Carrying it over would make the sequence depend on the call order even for the same seed, and
  // the correspondence with the dumped random sequence could no longer be read. Pinned by checking
  // that an odd length followed by the next call gives the same sequence as two even-length calls.
  const split = new Randn(5);
  const head = split.normals(3);
  const tail = split.normals(2);
  const fresh = new Randn(5);
  assertEquals([...head], [...fresh.normals(3)]);
  assertEquals([...tail], [...fresh.normals(2)]);
});

test("encodeWav: every RIFF header field follows the spec", () => {
  const wav = encodeWav(new Float32Array([0, 1, -1]), 44100);
  const view = new DataView(wav.buffer);
  const text = (offset: number): string => String.fromCharCode(...wav.subarray(offset, offset + 4));
  assertEquals(wav.byteLength, 44 + 3 * 2);
  assertEquals(text(0), "RIFF");
  assertEquals(view.getUint32(4, true), 36 + 6);
  assertEquals(text(8), "WAVE");
  assertEquals(text(12), "fmt ");
  assertEquals(view.getUint32(16, true), 16);
  assertEquals(view.getUint16(20, true), 1, "PCM");
  assertEquals(view.getUint16(22, true), 1, "mono");
  assertEquals(view.getUint32(24, true), 44100);
  assertEquals(view.getUint32(28, true), 44100 * 2, "byte rate");
  assertEquals(view.getUint16(32, true), 2, "block align");
  assertEquals(view.getUint16(34, true), 16);
  assertEquals(text(36), "data");
  assertEquals(view.getUint32(40, true), 6);
});

test("encodeWav: clips out-of-range values and rounds like floor(x+0.5)", () => {
  // ±0.5 times 32767 is exactly ±16383.5 (0.5 is exact in f32). This is **the only observation
  // point that tells the rounding rules for halves apart**: `Math.round` always rounds towards +∞
  // (+16384 / −16383). Python's built-in `round` (round half to even) gives −16384, and the three
  // wavs would differ by 1 LSB — the reference side is aligned with `floor(x+0.5)`.
  const wav = encodeWav(new Float32Array([2, -2, 0.5, -0.5]), 8000);
  const view = new DataView(wav.buffer);
  assertEquals(view.getInt16(44, true), 32767, "upper clip");
  assertEquals(view.getInt16(46, true), -32767, "lower clip");
  assertEquals(view.getInt16(48, true), 16384);
  assertEquals(view.getInt16(50, true), -16383);
});

test("encodeWav: rejects an invalid sample rate", () => {
  assertThrows(() => encodeWav(new Float32Array(1), 0), ModelInputError);
  assertThrows(() => encodeWav(new Float32Array(1), 44100.5), ModelInputError);
});

test("writeSafetensors: what is written can be read back (the dump transport round-trips)", () => {
  const bytes = writeSafetensors(
    new Map([
      ["ids", { dtype: "I32" as const, shape: [1, 3], data: Int32Array.from([7, 8, 9]) }],
      ["noise", { dtype: "F32" as const, shape: [2, 2], data: Float32Array.from([1, 2, 3, 4]) }],
    ]),
    { demo: '{"text":"あ"}' },
  );
  // The start of the data section (8 + header length) is on an 8-byte boundary (the reference
  // implementation's padding).
  const headerLength = Number(new DataView(bytes.buffer).getBigUint64(0, true));
  assertEquals((8 + headerLength) % 8, 0);

  const file = parseSafetensors(bytes.buffer.slice(0, bytes.byteLength));
  assertEquals(file.metadata.get("demo"), '{"text":"あ"}');
  const ids = file.tensors.get("ids");
  const noise = file.tensors.get("noise");
  assert(ids !== undefined && noise !== undefined);
  assertEquals(ids.shape, [1, 3]);
  assertEquals([...new Int32Array(file.buffer, ids.byteOffset, 3)], [7, 8, 9]);
  assertEquals(noise.shape, [2, 2]);
  assertEquals([...new Float32Array(file.buffer, noise.byteOffset, 4)], [1, 2, 3, 4]);
});

test("writeSafetensors: rejects a mismatch between shape and data length", () => {
  assertThrows(
    () =>
      writeSafetensors(
        new Map([["x", { dtype: "F32" as const, shape: [3], data: new Float32Array(2) }]]),
        {},
      ),
    Error,
    "element count",
  );
});
