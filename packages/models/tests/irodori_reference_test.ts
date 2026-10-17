// Host-side unit cases of the reference-audio preprocessing (K-weighting coefficients, input
// errors, reflect pad). They need neither a GPU nor assets. The parity gate against the upstream
// golden (outputs/series/dacvae-32dim/host/) lives in
// packages/e2e/tests/e2e_irodori_reference_host_test.ts.

import { test } from "vite-plus/test";
import { assert, assertEquals, assertThrows } from "@std/assert";
import { integratedLoudness, kWeightingFilters } from "../src/irodori/host/loudness.ts";
import { ModelInputError } from "../src/errors.ts";
import { normalizeReference, reflectPadToHop } from "../src/irodori/host/reference.ts";

/** 実重み v4-small の運用値（`pipelineConfig` が運ぶ数と同じ）。 */
const SAMPLE_RATE = 48000;

// ---- K-weighting の係数（式の写し間違いの検出）----------------------------

test("kWeightingFilters: 48kHz の係数が上流（pyloudnorm の RBJ 式）の数と一致する", () => {
  const [shelf, highPass] = kWeightingFilters(SAMPLE_RATE);
  // 値は `pyloudnorm.IIRfilter` を 48kHz で組んだときの係数（recon の実測）。係数は式から
  // 作るので、ここが合っていれば fc / Q / dB ゲインの取り違えは無い。
  assertEquals(
    [...shelf.b],
    [1.5351828863637502, -2.691804030199196, 1.198426263333146],
    "high shelf の分子",
  );
  assertEquals([...shelf.a], [1, -1.6906995865986896, 0.7325047060963897], "high shelf の分母");
  assertEquals(
    [...highPass.b],
    [0.9950442970178917, -1.9900885940357833, 0.9950442970178917],
    "high pass の分子",
  );
  assertEquals([...highPass.a], [1, -1.990076284018423, 0.9901009040531438], "high pass の分母");
});

test("kWeightingFilters: 周波数を変えると係数が動く（48kHz を焼き込んでいない）", () => {
  const [shelf] = kWeightingFilters(SAMPLE_RATE);
  const [other] = kWeightingFilters(44100);
  assertEquals(shelf.b[0] === other.b[0], false, "サンプリング周波数が係数に効いていない");
});

// ---- 無音 / 異常入力 -----------------------------------------------------

test("integratedLoudness: 無音は下限 −70 に clamp される（−Infinity を返さない）", () => {
  assertEquals(integratedLoudness(new Float32Array(SAMPLE_RATE), SAMPLE_RATE), -70);
});

test("integratedLoudness: 空の波形と不正な周波数は落とす", () => {
  assertThrows(() => integratedLoudness(new Float32Array(0), SAMPLE_RATE), RangeError, "波形が空");
  assertThrows(() => integratedLoudness(Float32Array.of(1), 0), RangeError, "正の整数でない");
});

test("integratedLoudness: 窓が導出できない極小の周波数は落とす（正の整数だけでは足りない）", () => {
  // 窓は floor 導出（kernel = trunc(0.4·sr) / stride = trunc(0.1·sr)）なので、周波数が小さいと
  // 窓そのものが消える。検査が無いと sr ≤ 9 は stride 0 でブロック数が Infinity になり
  // 「Invalid typed array length」という無関係な顔で落ち、sr ≤ 2 は kernel も 0 になる。
  assertThrows(
    () => integratedLoudness(Float32Array.of(0.5, 0.5), 9),
    RangeError,
    "loudness 窓が導出できない",
  );
  assertThrows(
    () => integratedLoudness(Float32Array.of(0.5, 0.5), 2),
    RangeError,
    "0.4 秒 = 0 サンプル / stride 0 サンプル",
  );
  // 10Hz からは窓が立つ（下限検査が正常域まで落としていない）。
  assertEquals(integratedLoudness(new Float32Array(20), 10), -70);
});

test("normalizeReference: 非有限サンプルは位置付きで落とす（利得の段を素通りさせない）", () => {
  // 検査が無いと、NaN は `magnitude > peakBeforeScale` が常に false なので peak 判定を
  // すり抜け、+Inf は `peakGain = 1/∞ = 0` で**全サンプルを 0 に潰した**上で NaN 1 点だけを
  // 残す。どちらも例外にならず「ほぼ無音の参照音声」として encoder まで届く。
  for (const value of [NaN, Infinity, -Infinity]) {
    const samples = Float32Array.from({ length: SAMPLE_RATE }, (_, index) =>
      index === 7 ? value : 0.1,
    ) as Float32Array<ArrayBuffer>;
    assertThrows(
      () => normalizeReference(samples, SAMPLE_RATE),
      ModelInputError,
      "参照音声の 7 番目のサンプルが非有限",
    );
  }
});

test("normalizeReference: 空の波形は入口で落とす（窓の門に化けさせない）", () => {
  // 空波形は呼び手が渡した wav の寸法そのもの（ADR 0107 決定 5）。この検査が無いと
  // `integratedLoudness` の「窓が導出できない」側に落ち、導出値（周波数・窓長）の門と
  // 区別が付かなくなる。
  assertThrows(
    () => normalizeReference(new Float32Array(0), SAMPLE_RATE),
    ModelInputError,
    "参照音声のサンプルが 0 個",
  );
  // 逆側: 導出値の前提を見る `integratedLoudness` の `RangeError` は残す（呼び手は直せない）。
  const windows = assertThrows(() => integratedLoudness(Float32Array.of(0.5, 0.5), 9));
  assert(!(windows instanceof ModelInputError));
});

test("normalizeReference: 有限な入力は通る（検査が全部を落としていない）", () => {
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

// ---- reflect pad（golden 無しでも成り立つ形）------------------------------

test("reflectPadToHop: 鏡像は端のサンプルを含まない（torch の reflect と同じ綴り）", () => {
  const samples = Float32Array.from({ length: 5 }, (_, i) => i + 1) as Float32Array<ArrayBuffer>;
  // hop 4 → 長さ 5 は 8 へ（pad 3）。[1,2,3,4,5] → [1,2,3,4,5,4,3,2]。
  assertEquals(Array.from(reflectPadToHop(samples, 4)), [1, 2, 3, 4, 5, 4, 3, 2]);
});

test("reflectPadToHop: 既に倍数なら写しも作らず入力をそのまま返す", () => {
  const samples = new Float32Array(8) as Float32Array<ArrayBuffer>;
  assertEquals(reflectPadToHop(samples, 4), samples);
});

test("reflectPadToHop: hop 未満の入力は落とす（鏡像が自分の先頭を飛び越える）", () => {
  assertThrows(
    () => reflectPadToHop(new Float32Array(3) as Float32Array<ArrayBuffer>, 1920),
    ModelInputError,
    "hopLength 1920 が要る",
  );
});
