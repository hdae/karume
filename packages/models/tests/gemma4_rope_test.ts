// Checks of the host generation of gemma4 RoPE rows (`src/gemma/rope.ts`) — needs neither a GPU nor assets.
//
// The source of truth is the TS formula (f64 → f32 on storage), which is not bit-identical to upstream transformers'
// tables (f32 throughout, SLEEF's 1 ULP) (the numeric contract at the top of rope.ts). Four things are pinned here:
//
//  ① **Tolerance parity with upstream's actual output** — the fixture is the output of `Gemma4TextRotaryEmbedding`
//     written as is in f32 by the exporter-side generator (tools/export-recipes/gemma4/tests/rope_fixture.py). The
//     tolerance is proportional to the position (the upper bound of upstream's f32 angle error), not a tautology
//     that writes the same formula twice
//  ② **Fault injection** — a position shifted by 1, swapped layer types, a wrong theta and a wrong rotaryDim fail
//     within ①'s band (proof that the check is not vacuous)
//  ③ **Structure** — non-rotated dimensions have exactly cos = 1 / sin = 0, the first and second halves are equal,
//     the pad row (position 0) is cos = 1 / sin = 0, and generating the same input twice is bit-identical
//  ④ **The shape of the graph inputs** — 4 names, shape `[1, rows, headDim]`, f32

import { readTextFile } from "@karume/test-support/fs";
import { test } from "vite-plus/test";
import { assert, assertEquals, assertThrows } from "@std/assert";
import {
  assertGemma4RopeSpec,
  GEMMA4_ROPE_LAYER_TYPES,
  gemma4RopeInputName,
  gemma4RopeInputNames,
  ropeInputs,
  gemma4RopeInverseFrequencies,
  type RopeLayerSpec,
  type RopeLayerType,
  gemma4RopeRows,
  type RopeSpec,
} from "../src/gemma/rope.ts";

type Fixture = {
  readonly positions: readonly number[];
  readonly spec: RopeSpec;
  readonly tables: Readonly<Record<RopeLayerType, { readonly cos: string; readonly sin: string }>>;
};

const FIXTURE_URL = new URL("./fixtures/gemma4-rope-upstream.json", import.meta.url);

const loadFixture = async (): Promise<Fixture> =>
  JSON.parse(await readTextFile(FIXTURE_URL)) as Fixture;

/** base64 (f32 little endian) → Float32Array. */
const decodeF32 = (base64: string): Float32Array<ArrayBuffer> => {
  const bytes = Uint8Array.from(atob(base64), (char) => char.charCodeAt(0));
  return new Float32Array(bytes.buffer as ArrayBuffer, 0, bytes.byteLength / 4);
};

/**
 * A position-proportional tolerance (per element).
 *
 * Upstream multiplies the angle `position × invFreq` in f32, so its rounding error is at most
 * `position × invFreq × 2⁻²⁴` (invFreq ≤ 1). The slopes of cos / sin are at most 1, so the difference in value is
 * bounded by the same, and `2.5e-7` covers the f32 storage rounding on both sides + SLEEF's 1 ULP. The coefficient
 * `1.2e-7` is `2⁻²³` (two roundings of the product). The measurement **gives a different number per measured range
 * even with the same formula** (all 3 inside the band): the maximum over this test's 13 fixture points is 4.77e-3
 * (P=131,071, band 1.57e-2) / the maximum over a full sweep of positions 0..131,071 is 9.44e-3 (worst point
 * P≈129,293) / a full sweep of positions 0..1,024 gives 6.9e-5.
 */
const toleranceAt = (position: number): number => 2.5e-7 + position * 1.2e-7;

const maxAbsDiffPerRow = (
  actual: Float32Array,
  expected: Float32Array,
  width: number,
): number[] => {
  assertEquals(actual.length, expected.length, "length");
  const rows = actual.length / width;
  const worst: number[] = [];
  for (let row = 0; row < rows; row += 1) {
    let max = 0;
    for (let j = 0; j < width; j += 1) {
      const at = row * width + j;
      max = Math.max(max, Math.abs(actual[at] - expected[at]));
    }
    worst.push(max);
  }
  return worst;
};

/** Comparison with the upstream fixture (does each row's worst difference fit the position-proportional band). */
const passesParity = (
  spec: RopeLayerSpec,
  positions: readonly number[],
  expected: { readonly cos: Float32Array; readonly sin: Float32Array },
): { readonly pass: boolean; readonly worst: number } => {
  const rows = gemma4RopeRows(spec, positions);
  const cosDiff = maxAbsDiffPerRow(rows.cos, expected.cos, spec.headDim);
  const sinDiff = maxAbsDiffPerRow(rows.sin, expected.sin, spec.headDim);
  let pass = true;
  let worst = 0;
  positions.forEach((position, row) => {
    const diff = Math.max(cosDiff[row], sinDiff[row]);
    worst = Math.max(worst, diff);
    // MUST: written in the form "fail unless inside the band". With `diff > tol`, a NaN difference is false and slips through.
    if (!(diff <= toleranceAt(position))) pass = false;
  });
  return { pass, worst };
};

test("matches upstream transformers' RoPE tables within a position-proportional tolerance (fixture parity)", async () => {
  const fixture = await loadFixture();
  assertGemma4RopeSpec("fixture", fixture.spec);
  for (const layerType of GEMMA4_ROPE_LAYER_TYPES) {
    const spec = fixture.spec[layerType];
    const expected = {
      cos: decodeF32(fixture.tables[layerType].cos),
      sin: decodeF32(fixture.tables[layerType].sin),
    };
    assertEquals(
      expected.cos.length,
      fixture.positions.length * spec.headDim,
      `${layerType}: row count`,
    );
    const report = passesParity(spec, fixture.positions, expected);
    assert(
      report.pass,
      `${layerType}: the difference from upstream exceeded the band (worst ${report.worst})`,
    );
    console.log(`${layerType}: worst difference from upstream ${report.worst.toExponential(2)}`);
  }
});

test("fault injection — a shifted position / swapped layer types / wrong theta / wrong rotaryDim fail parity", async () => {
  const fixture = await loadFixture();
  const full = fixture.spec.full_attention;
  const sliding = fixture.spec.sliding_attention;
  const expectedFull = {
    cos: decodeF32(fixture.tables.full_attention.cos),
    sin: decodeF32(fixture.tables.full_attention.sin),
  };
  const shifted = fixture.positions.map((position) => position + 1);
  const cases: readonly [string, () => boolean][] = [
    ["shift the position by 1", () => passesParity(full, shifted, expectedFull).pass],
    [
      "swap the layer types (build full with sliding's formula)",
      () =>
        passesParity({ ...sliding, headDim: full.headDim }, fixture.positions, expectedFull).pass,
    ],
    [
      "set theta to 1e5",
      () => passesParity({ ...full, theta: 1e5 }, fixture.positions, expectedFull).pass,
    ],
    [
      "set rotaryDim to headDim (ignore partial rotary)",
      () =>
        passesParity({ ...full, rotaryDim: full.headDim }, fixture.positions, expectedFull).pass,
    ],
  ];
  for (const [label, run] of cases) {
    assertEquals(run(), false, `${label}: the mutation passed the band`);
  }
});

test("fault injection — a row with NaN, and a row with both NaN and a large error, fail parity", async () => {
  // Judging failure with `diff > tolerance` makes a NaN difference false and passes it, and a large finite error in the
  // same row is hidden too by Math.max's NaN propagation. One fixture row is broken to see that the band lets no non-finite value through.
  const fixture = await loadFixture();
  const full = fixture.spec.full_attention;
  const row = fixture.positions.length - 1;
  const corrupt = (edit: (cos: Float32Array<ArrayBuffer>, at: number) => void) => {
    const cos = decodeF32(fixture.tables.full_attention.cos).slice();
    edit(cos, row * full.headDim);
    return { cos, sin: decodeF32(fixture.tables.full_attention.sin) };
  };
  const cases: readonly [string, { cos: Float32Array; sin: Float32Array }][] = [
    ["one cos element is NaN", corrupt((cos, at) => (cos[at] = Number.NaN))],
    [
      "NaN and a large error (+999) in the same row",
      corrupt((cos, at) => {
        cos[at] = Number.NaN;
        cos[at + 1] += 999;
      }),
    ],
  ];
  for (const [label, expected] of cases) {
    assertEquals(
      passesParity(full, fixture.positions, expected).pass,
      false,
      `${label}: a non-finite value passed the band`,
    );
  }
});

test("non-rotated dimensions have exactly cos = 1 / sin = 0, the halves are equal, and position 0 is the identity", () => {
  const spec: RopeLayerSpec = { theta: 1e6, headDim: 512, rotaryDim: 128 };
  const invFreq = gemma4RopeInverseFrequencies(spec);
  assertEquals(invFreq.length, 256);
  assertEquals(invFreq[0], 1);
  for (let i = 64; i < 256; i += 1) assert(Object.is(invFreq[i], 0), `invFreq[${i}]`);
  const rows = gemma4RopeRows(spec, [0, 7, 131071]);
  for (let row = 0; row < 3; row += 1) {
    for (let j = 0; j < 256; j += 1) {
      const at = row * 512 + j;
      assertEquals(
        rows.cos[at],
        rows.cos[at + 256],
        `row ${row} cos first half = second half (${j})`,
      );
      assertEquals(
        rows.sin[at],
        rows.sin[at + 256],
        `row ${row} sin first half = second half (${j})`,
      );
      if (j >= 64 || row === 0) {
        assert(Object.is(rows.cos[at], 1), `row ${row} cos[${j}] is exactly 1`);
        assert(Object.is(rows.sin[at], 0), `row ${row} sin[${j}] is exactly 0`);
      }
    }
  }
});

test("generating the same input twice is bit-identical (determinism)", () => {
  const spec: RopeLayerSpec = { theta: 1e4, headDim: 256, rotaryDim: 256 };
  const positions = [0, 1, 2, 511, 4096, 131071];
  const a = gemma4RopeRows(spec, positions);
  const b = gemma4RopeRows(spec, positions);
  assertEquals(new Uint32Array(a.cos.buffer), new Uint32Array(b.cos.buffer));
  assertEquals(new Uint32Array(a.sin.buffer), new Uint32Array(b.sin.buffer));
});

test("the graph inputs are 4, named by the fixed layer type × part form, shape [1, rows, headDim], f32", () => {
  const spec: RopeSpec = {
    sliding_attention: { theta: 1e4, headDim: 256, rotaryDim: 256 },
    full_attention: { theta: 1e6, headDim: 512, rotaryDim: 128 },
  };
  const positions = Int32Array.of(5, 6, 0, 0);
  const inputs = ropeInputs(spec, positions);
  assertEquals(Object.keys(inputs).sort(), gemma4RopeInputNames().sort());
  assertEquals(gemma4RopeInputName("full_attention", "sin"), "rope_full_attention_sin");
  for (const layerType of GEMMA4_ROPE_LAYER_TYPES) {
    for (const part of ["cos", "sin"] as const) {
      const tensor = inputs[gemma4RopeInputName(layerType, part)];
      assertEquals(tensor.dtype, "f32");
      assertEquals(tensor.shape, [1, 4, spec[layerType].headDim]);
      assertEquals(tensor.data.length, 4 * spec[layerType].headDim);
    }
  }
  // The pad row (position 0) is the identity
  const cos = inputs.rope_full_attention_cos.data as Float32Array;
  assert(Object.is(cos[2 * 512], 1));
});

test("range checks of the parameters — odd / out of range / non-finite fail loudly, and a negative position fails too", () => {
  const ok: RopeLayerSpec = { theta: 1e4, headDim: 256, rotaryDim: 256 };
  assertThrows(() => gemma4RopeInverseFrequencies({ ...ok, headDim: 255 }), Error, "headDim");
  assertThrows(() => gemma4RopeInverseFrequencies({ ...ok, rotaryDim: 258 }), Error, "rotaryDim");
  assertThrows(() => gemma4RopeInverseFrequencies({ ...ok, rotaryDim: 3 }), Error, "rotaryDim");
  assertThrows(() => gemma4RopeInverseFrequencies({ ...ok, theta: 0 }), Error, "theta");
  assertThrows(() => gemma4RopeInverseFrequencies({ ...ok, theta: Infinity }), Error, "theta");
  assertThrows(() => gemma4RopeRows(ok, [-1]), Error, "position");
  assertThrows(() => gemma4RopeRows(ok, [1.5]), Error, "position");
});
