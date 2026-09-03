// gemma4 の RoPE 行のホスト生成（`src/gemma/rope.ts`）の門 —— GPU も資産も要らない。
//
// 正本は TS の式（f64 → 格納時 f32）で、上流 transformers の表（全経路 f32・SLEEF の 1 ULP）とは
// ビット同一にならない（rope.ts 冒頭の数値契約）。ここで押さえるのは 4 つ:
//
//  ① **上流の実出力との許容差 parity** — fixture は exporter 側の生成器
//     （tools/export-recipes/gemma4/tests/rope_fixture.py）が `Gemma4TextRotaryEmbedding` の出力を
//     そのまま f32 で焼いたもの。許容差は位置比例（上流の f32 角度誤差の上界）で、同じ式を 2 回
//     書いた恒真にはしない
//  ② **故障注入** — 位置 1 ずらし・層種入れ替え・theta 違い・rotaryDim 違いが①の帯で落ちる
//     （門が空振りでない証明）
//  ③ **構造** — 回さない次元は cos = 1 / sin = 0 が厳密・前半と後半は同値・pad 行（位置 0）は
//     cos = 1 / sin = 0・同じ入力の 2 回生成はビット同一
//  ④ **グラフ入力の形** — 名前 4 本・shape `[1, rows, headDim]`・f32

import { assert, assertEquals, assertThrows } from "@std/assert";
import {
  assertGemma4RopeSpec,
  GEMMA4_ROPE_LAYER_TYPES,
  gemma4RopeInputName,
  gemma4RopeInputNames,
  gemma4RopeInputs,
  gemma4RopeInverseFrequencies,
  type Gemma4RopeLayerSpec,
  type Gemma4RopeLayerType,
  gemma4RopeRows,
  type Gemma4RopeSpec,
} from "../src/gemma/rope.ts";

type Fixture = {
  readonly positions: readonly number[];
  readonly spec: Gemma4RopeSpec;
  readonly tables: Readonly<
    Record<Gemma4RopeLayerType, { readonly cos: string; readonly sin: string }>
  >;
};

const FIXTURE_URL = new URL("./fixtures/gemma4-rope-upstream.json", import.meta.url);

const loadFixture = async (): Promise<Fixture> =>
  JSON.parse(await Deno.readTextFile(FIXTURE_URL)) as Fixture;

/** base64（f32 リトルエンディアン）→ Float32Array。 */
const decodeF32 = (base64: string): Float32Array<ArrayBuffer> => {
  const bytes = Uint8Array.from(atob(base64), (char) => char.charCodeAt(0));
  return new Float32Array(bytes.buffer as ArrayBuffer, 0, bytes.byteLength / 4);
};

/**
 * 位置比例の許容差（要素ごと）。
 *
 * 上流は角度 `position × invFreq` を f32 で積むので、その丸め誤差は `position × invFreq × 2⁻²⁴`
 * 以下（invFreq ≤ 1）。cos / sin の傾きは 1 以下なので値の差もその上界で押さえられ、さらに
 * 両側の f32 格納丸め + SLEEF の 1 ULP を `2.5e-7` で持つ。係数 `1.2e-7` は `2⁻²³`（積の丸め 2 回分）。
 * 実測は**同じ式でも測定範囲ごとに別の数**になる（3 つとも帯の内側）: 本テストの fixture
 * 13 点の最大が 4.77e-3（P=131,071・帯 1.57e-2）/ 位置 0..131,071 を全掃引したときの最大が
 * 9.44e-3（最大点は P≈129,293）/ 位置 0..1,024 の全掃引では 6.9e-5。
 */
const toleranceAt = (position: number): number => 2.5e-7 + position * 1.2e-7;

const maxAbsDiffPerRow = (
  actual: Float32Array,
  expected: Float32Array,
  width: number,
): number[] => {
  assertEquals(actual.length, expected.length, "長さ");
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

/** 上流 fixture との突合（行ごとの最悪差が位置比例の帯に収まるか）。 */
const passesParity = (
  spec: Gemma4RopeLayerSpec,
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
    if (diff > toleranceAt(position)) pass = false;
  });
  return { pass, worst };
};

Deno.test("上流 transformers の RoPE 表と位置比例の許容差で一致する（fixture parity）", async () => {
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
      `${layerType}: 行数`,
    );
    const report = passesParity(spec, fixture.positions, expected);
    assert(report.pass, `${layerType}: 上流との差が帯を超えた（最悪 ${report.worst}）`);
    console.log(`${layerType}: 上流との最悪差 ${report.worst.toExponential(2)}`);
  }
});

Deno.test("故障注入 — 位置ずらし / 層種入れ替え / theta / rotaryDim の誤りは parity で落ちる", async () => {
  const fixture = await loadFixture();
  const full = fixture.spec.full_attention;
  const sliding = fixture.spec.sliding_attention;
  const expectedFull = {
    cos: decodeF32(fixture.tables.full_attention.cos),
    sin: decodeF32(fixture.tables.full_attention.sin),
  };
  const shifted = fixture.positions.map((position) => position + 1);
  const cases: readonly [string, () => boolean][] = [
    ["位置を 1 ずらす", () => passesParity(full, shifted, expectedFull).pass],
    [
      "層種を入れ替える（sliding の式で full を作る）",
      () =>
        passesParity({ ...sliding, headDim: full.headDim }, fixture.positions, expectedFull).pass,
    ],
    [
      "theta を 1e5 にする",
      () => passesParity({ ...full, theta: 1e5 }, fixture.positions, expectedFull).pass,
    ],
    [
      "rotaryDim を headDim にする（partial rotary を無視）",
      () =>
        passesParity({ ...full, rotaryDim: full.headDim }, fixture.positions, expectedFull).pass,
    ],
  ];
  for (const [label, run] of cases) {
    assertEquals(run(), false, `${label}: 変異が帯を通ってしまった`);
  }
});

Deno.test("回さない次元は cos = 1 / sin = 0 が厳密で、前半と後半は同値・位置 0 は恒等", () => {
  const spec: Gemma4RopeLayerSpec = { theta: 1e6, headDim: 512, rotaryDim: 128 };
  const invFreq = gemma4RopeInverseFrequencies(spec);
  assertEquals(invFreq.length, 256);
  assertEquals(invFreq[0], 1);
  for (let i = 64; i < 256; i += 1) assert(Object.is(invFreq[i], 0), `invFreq[${i}]`);
  const rows = gemma4RopeRows(spec, [0, 7, 131071]);
  for (let row = 0; row < 3; row += 1) {
    for (let j = 0; j < 256; j += 1) {
      const at = row * 512 + j;
      assertEquals(rows.cos[at], rows.cos[at + 256], `row ${row} cos 前半 = 後半 (${j})`);
      assertEquals(rows.sin[at], rows.sin[at + 256], `row ${row} sin 前半 = 後半 (${j})`);
      if (j >= 64 || row === 0) {
        assert(Object.is(rows.cos[at], 1), `row ${row} cos[${j}] は厳密 1`);
        assert(Object.is(rows.sin[at], 0), `row ${row} sin[${j}] は厳密 0`);
      }
    }
  }
});

Deno.test("同じ入力の 2 回生成はビット同一（決定性）", () => {
  const spec: Gemma4RopeLayerSpec = { theta: 1e4, headDim: 256, rotaryDim: 256 };
  const positions = [0, 1, 2, 511, 4096, 131071];
  const a = gemma4RopeRows(spec, positions);
  const b = gemma4RopeRows(spec, positions);
  assertEquals(
    new Uint32Array(a.cos.buffer),
    new Uint32Array(b.cos.buffer),
  );
  assertEquals(
    new Uint32Array(a.sin.buffer),
    new Uint32Array(b.sin.buffer),
  );
});

Deno.test("グラフ入力は 4 本・名前は層種 × 部の固定形・shape [1, rows, headDim]・f32", () => {
  const spec: Gemma4RopeSpec = {
    sliding_attention: { theta: 1e4, headDim: 256, rotaryDim: 256 },
    full_attention: { theta: 1e6, headDim: 512, rotaryDim: 128 },
  };
  const positions = Int32Array.of(5, 6, 0, 0);
  const inputs = gemma4RopeInputs(spec, positions);
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
  // pad 行（位置 0）は恒等
  const cos = inputs.rope_full_attention_cos.data as Float32Array;
  assert(Object.is(cos[2 * 512], 1));
});

Deno.test("パラメータの値域門 — 奇数 / 範囲外 / 非有限は fail loudly・負の位置も落ちる", () => {
  const ok: Gemma4RopeLayerSpec = { theta: 1e4, headDim: 256, rotaryDim: 256 };
  assertThrows(() => gemma4RopeInverseFrequencies({ ...ok, headDim: 255 }), Error, "headDim");
  assertThrows(() => gemma4RopeInverseFrequencies({ ...ok, rotaryDim: 258 }), Error, "rotaryDim");
  assertThrows(() => gemma4RopeInverseFrequencies({ ...ok, rotaryDim: 3 }), Error, "rotaryDim");
  assertThrows(() => gemma4RopeInverseFrequencies({ ...ok, theta: 0 }), Error, "theta");
  assertThrows(() => gemma4RopeInverseFrequencies({ ...ok, theta: Infinity }), Error, "theta");
  assertThrows(() => gemma4RopeRows(ok, [-1]), Error, "位置");
  assertThrows(() => gemma4RopeRows(ok, [1.5]), Error, "位置");
});
