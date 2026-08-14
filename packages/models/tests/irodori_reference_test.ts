// 参照音声のホスト前処理（LUFS 測定 → −16 正規化 → reflect pad）の**パリティ門**。
// GPU は要らない（純関数だけ）が、golden 資産 `outputs/series/dacvae-32dim/host/` が要る。
//
// golden は `tools/exporter/dacvae_host.py` が **上流を呼んで**書いたもので、あちらは
// 「記録したスカラー 2 本で正規化後の波形がビット一致で再現できる」ことと「上流
// `_normalize_loudness` とビット一致すること」を毎回実測してから書く。したがってこの門が
// 閉じることは「TS のホスト前処理が上流と同じ波形を作る」の 2 段目にあたる。
//
// ## 何を突き合わせ、何は突き合わせないのか
//
// golden 5 ケースのうち **LUFS 正規化を通るのは 3 本**（`normalizeDb: -16`）で、`ref-plain` /
// `ref-hot` は上流の「正規化しない（peak 制限だけ）」経路のケース。パイプラインの目標値は
// −16 固定（`host/reference.ts` の `TARGET_DB`）でその経路を**持たない**ので、この 2 本からは
// **LUFS 測定値だけ**を突き合わせる（波形まで比べると、実装が持たない枝を門が要求してしまう）。
//
// ## 枝の踏み分け（恒真化の遮断）
//
// - `ref-short`（0.4 秒）— 0.5 秒未満のゼロ pad 測定枝 **と** peak 制限枝（利得 441 倍で
//   peak が 1.84 まで上がる）の両方。**LUFS 経路で peak 制限を踏む唯一のケース**
// - `ref-odd`（144,777 サンプル）— reflect pad の端数枝 **と**、ブロックのゼロ pad 枝
//   （`ref-plain` と先頭 144,000 サンプルを共有するので、末尾の欠けたブロックを捨てる実装では
//   両者の LUFS が一致してしまう — その一致が起きないことを直接見る）
// - `ref-default` — pad が 0 サンプルの恒等枝
// - `ref-hot` — 振幅 3 倍（LUFS 測定そのものが値域に依らないこと）

import { assertAlmostEquals, assertEquals, assertThrows } from "@std/assert";
import { parseSafetensors } from "@karume/runtime";
import { integratedLoudness, kWeightingFilters } from "../src/irodori/host/loudness.ts";
import { normalizeReference, reflectPadToHop } from "../src/irodori/host/reference.ts";

/** 実重み v4-small の運用値（`pipelineConfig` が運ぶ数と同じ）。 */
const SAMPLE_RATE = 48000;
const HOP_LENGTH = 1920;

/**
 * LUFS 測定値（LU）の許容差。
 *
 * 実測（この門が毎回ログに出す表。2026-08-12 / golden 5 ケース）:
 *
 * | ケース      | 秒        | refDb 差   | 利得の相対差 | peak 利得の相対差 | 正規化後 maxAbs |
 * | ----------- | --------- | ---------- | ------------ | ----------------- | --------------- |
 * | ref-default | 7.6       | 5.131e-6   | −6.061e-7    | 0（厳密）         | 5.961e-7        |
 * | ref-plain   | 3.0       | 2.659e-6   | —（経路外）  | —                 | —               |
 * | ref-hot     | 3.0       | 4.982e-6   | —（経路外）  | —                 | —               |
 * | ref-short   | 0.4       | −2.158e-5  | 2.500e-6     | −2.457e-6         | 1.192e-7        |
 * | ref-odd     | 3.0161875 | 3.526e-6   | −3.745e-7    | 0（厳密）         | 3.576e-7        |
 *
 * 差の出どころは**ただ 1 つ**、K-weighting の IIR を上流が f32 で・ここが f64 で回すこと
 * （`host/loudness.ts` の MUST）。ブロック分割もゲーティングも整数と比較の判断なので、
 * 実装が違えば 0.01 LU 以上の段差で出る（実測: ブロックの数え方を取り違えると `ref-odd` が
 * 0.074 LU ずれた — 下の「ブロックのゼロ pad」テストが見ているのはその段差）。
 *
 * 2e-4 LU は実測最悪 2.158e-5 の **9.3 倍**。正規化利得に直すと相対 2.3e-5 で、16bit PCM の
 * 1LSB（3.05e-5）より小さい — **WAV に書いた時点で消える**大きさ。
 */
const REF_DB_ATOL = 2e-4;

/**
 * 利得（LUFS 利得 / peak 利得）の相対許容差。実測最悪 2.500e-6（`ref-short`）の **8.0 倍**。
 * `refDb` の差がそのまま指数に乗るだけなので、上の表と同じ素性の数。
 */
const GAIN_RTOL = 2e-5;

/**
 * 正規化後 / pad 後の波形の絶対許容差。実測最悪 5.961e-7（`ref-default`）の **8.4 倍**。
 *
 * 値域は ±1 なので、−106dBFS 級。利得 1 本の相対差 6e-7 が値域上端に出たものが実体で、
 * f32 の 1ulp（6e-8）の 10 倍のところに居る。**式の取り違え**（目標 dB・利得の符号・peak の
 * 判定条件）は値域と同じ O(0.1〜1) で出る。
 */
const WAVE_ATOL = 5e-6;

const GOLDEN_DIR = new URL("../../../outputs/series/dacvae-32dim/host/", import.meta.url);
const GOLDEN_COMMAND =
  "cd tools/exporter && uv run --with descript-audiotools --with einops --with 'transformers==5.14.1' python dacvae_host.py";

/** golden `meta.json` のうち、この門が読む欄だけ（形は exporter が持つ — ここは読み口）。 */
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

const metaText = await Deno.readTextFile(new URL("meta.json", GOLDEN_DIR)).catch(() => undefined);
if (metaText === undefined) {
  console.warn(
    `[karume] 参照音声パリティ門を SKIP する（${GOLDEN_DIR.pathname} が要る）。生成: ${GOLDEN_COMMAND}`,
  );
}
const RUNNABLE = metaText !== undefined;

const readMeta = (): GoldenMeta => JSON.parse(metaText as string) as GoldenMeta;

/** 1 ケースぶんの golden テンソル（`raw` / `normalized` / `padded`）。 */
const readCase = async (
  name: string,
): Promise<Readonly<Record<string, Float32Array<ArrayBuffer>>>> => {
  const bytes = await Deno.readFile(new URL(`case.${name}.safetensors`, GOLDEN_DIR));
  const file = parseSafetensors(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer,
  );
  const out: Record<string, Float32Array<ArrayBuffer>> = {};
  for (const key of ["raw", "normalized", "padded"]) {
    const spec = file.tensors.get(key);
    if (spec === undefined) throw new Error(`golden case.${name} に '${key}' が無い`);
    out[key] = new Float32Array(
      file.buffer.slice(spec.byteOffset, spec.byteOffset + spec.byteLength),
    ) as Float32Array<ArrayBuffer>;
  }
  return out;
};

const maxAbsDiff = (actual: Float32Array, expected: Float32Array, where: string): number => {
  if (actual.length !== expected.length) {
    throw new Error(`${where}: 長さが ${actual.length} と ${expected.length} で違う`);
  }
  let worst = 0;
  for (let i = 0; i < actual.length; i += 1) {
    worst = Math.max(worst, Math.abs(actual[i] - expected[i]));
  }
  return worst;
};

// ---- K-weighting の係数（式の写し間違いの検出）----------------------------

Deno.test("kWeightingFilters: 48kHz の係数が上流（pyloudnorm の RBJ 式）の数と一致する", () => {
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

Deno.test("kWeightingFilters: 周波数を変えると係数が動く（48kHz を焼き込んでいない）", () => {
  const [shelf] = kWeightingFilters(SAMPLE_RATE);
  const [other] = kWeightingFilters(44100);
  assertEquals(shelf.b[0] === other.b[0], false, "サンプリング周波数が係数に効いていない");
});

// ---- 無音 / 異常入力 -----------------------------------------------------

Deno.test("integratedLoudness: 無音は下限 −70 に clamp される（−Infinity を返さない）", () => {
  assertEquals(integratedLoudness(new Float32Array(SAMPLE_RATE), SAMPLE_RATE), -70);
});

Deno.test("integratedLoudness: 空の波形と不正な周波数は落とす", () => {
  assertThrows(() => integratedLoudness(new Float32Array(0), SAMPLE_RATE), RangeError, "波形が空");
  assertThrows(() => integratedLoudness(Float32Array.of(1), 0), RangeError, "正の整数でない");
});

Deno.test("integratedLoudness: 窓が導出できない極小の周波数は落とす（正の整数だけでは足りない）", () => {
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

Deno.test("normalizeReference: 非有限サンプルは位置付きで落とす（利得の段を素通りさせない）", () => {
  // 検査が無いと、NaN は `magnitude > peakBeforeScale` が常に false なので peak 判定を
  // すり抜け、+Inf は `peakGain = 1/∞ = 0` で**全サンプルを 0 に潰した**上で NaN 1 点だけを
  // 残す。どちらも例外にならず「ほぼ無音の参照音声」として encoder まで届く。
  for (const value of [NaN, Infinity, -Infinity]) {
    const samples = Float32Array.from(
      { length: SAMPLE_RATE },
      (_, index) => (index === 7 ? value : 0.1),
    ) as Float32Array<ArrayBuffer>;
    assertThrows(
      () => normalizeReference(samples, SAMPLE_RATE),
      Error,
      "参照音声の 7 番目のサンプルが非有限",
    );
  }
});

Deno.test("normalizeReference: 有限な入力は通る（検査が全部を落としていない）", () => {
  const samples = Float32Array.from(
    { length: SAMPLE_RATE },
    (_, index) => Math.sin(index / 50) * 0.1,
  ) as Float32Array<ArrayBuffer>;
  const normalized = normalizeReference(samples, SAMPLE_RATE);
  assertEquals(normalized.data.length, samples.length);
  assertEquals(normalized.data.every((sample) => Number.isFinite(sample)), true);
});

// ---- reflect pad（golden 無しでも成り立つ形）------------------------------

Deno.test("reflectPadToHop: 鏡像は端のサンプルを含まない（torch の reflect と同じ綴り）", () => {
  const samples = Float32Array.from({ length: 5 }, (_, i) => i + 1) as Float32Array<ArrayBuffer>;
  // hop 4 → 長さ 5 は 8 へ（pad 3）。[1,2,3,4,5] → [1,2,3,4,5,4,3,2]。
  assertEquals(Array.from(reflectPadToHop(samples, 4)), [1, 2, 3, 4, 5, 4, 3, 2]);
});

Deno.test("reflectPadToHop: 既に倍数なら写しも作らず入力をそのまま返す", () => {
  const samples = new Float32Array(8) as Float32Array<ArrayBuffer>;
  assertEquals(reflectPadToHop(samples, 4), samples);
});

Deno.test("reflectPadToHop: hop 未満の入力は落とす（鏡像が自分の先頭を飛び越える）", () => {
  assertThrows(
    () => reflectPadToHop(new Float32Array(3) as Float32Array<ArrayBuffer>, 1920),
    RangeError,
    "hopLength 1920 が要る",
  );
});

// ---- golden とのパリティ（本体）-------------------------------------------

Deno.test({
  name: "参照音声のホスト前処理が golden（上流実測）と一致する — 5 ケース",
  ignore: !RUNNABLE,
  fn: async () => {
    const meta = readMeta();
    assertEquals(meta.sampleRate, SAMPLE_RATE, "golden の sampleRate");
    assertEquals(meta.hopLength, HOP_LENGTH, "golden の hopLength");
    // 利得換算係数（`ln(10)/20`）— `host/reference.ts` が同じ数を持っている前提。
    assertAlmostEquals(Math.LN10 / 20, meta.gainFactor, 1e-15, "gainFactor");

    const rows: string[] = [];
    let worstRefDb = 0;
    let worstGain = 0;
    let worstWave = 0;
    for (const [name, expected] of Object.entries(meta.cases)) {
      const tensors = await readCase(name);
      assertEquals(tensors["raw"].length, expected.samples, `${name}: golden の raw の長さ`);

      const refDb = integratedLoudness(tensors["raw"], SAMPLE_RATE);
      const refDbDiff = refDb - expected.refDb;
      worstRefDb = Math.max(worstRefDb, Math.abs(refDbDiff));
      if (expected.normalizeDb === null) {
        // 上流の「正規化しない」経路のケース。LUFS 測定値だけを見る（モジュール doc）。
        rows.push(`  ${name.padEnd(11)} refDb ${refDbDiff.toExponential(3)}（LUFS 経路外）`);
        assertAlmostEquals(refDb, expected.refDb, REF_DB_ATOL, `${name}: refDb`);
        continue;
      }
      assertEquals(expected.normalizeDb, -16, `${name}: golden の目標 dB が −16 でない`);

      const normalized = normalizeReference(tensors["raw"], SAMPLE_RATE);
      const padded = reflectPadToHop(normalized.data, HOP_LENGTH);
      const gainDiff = (normalized.loudnessGain - expected.loudnessGain) / expected.loudnessGain;
      const peakGainDiff = (normalized.peakGain - expected.peakGain) / expected.peakGain;
      const waveDiff = maxAbsDiff(normalized.data, tensors["normalized"], `${name} の normalized`);
      const padDiff = maxAbsDiff(padded, tensors["padded"], `${name} の padded`);
      worstGain = Math.max(worstGain, Math.abs(gainDiff), Math.abs(peakGainDiff));
      worstWave = Math.max(worstWave, waveDiff, padDiff);
      rows.push(
        `  ${name.padEnd(11)} refDb ${refDbDiff.toExponential(3)} / 利得 ${
          gainDiff.toExponential(3)
        } / peak 利得 ${peakGainDiff.toExponential(3)} / 波形 ${waveDiff.toExponential(3)}`,
      );

      assertAlmostEquals(refDb, expected.refDb, REF_DB_ATOL, `${name}: refDb`);
      assertAlmostEquals(gainDiff, 0, GAIN_RTOL, `${name}: LUFS 利得の相対差`);
      assertAlmostEquals(peakGainDiff, 0, GAIN_RTOL, `${name}: peak 利得の相対差`);
      assertEquals(waveDiff < WAVE_ATOL, true, `${name}: 正規化後の波形の差 ${waveDiff}`);
      assertEquals(padDiff < WAVE_ATOL, true, `${name}: pad 後の波形の差 ${padDiff}`);
      // 長さと frames は整数の判断なので完全一致を要求する。
      assertEquals(padded.length, expected.paddedSamples, `${name}: pad 後の長さ`);
      assertEquals(padded.length / HOP_LENGTH, expected.frames, `${name}: フレーム数`);
    }
    console.log(
      `[parity] irodori 参照音声の前処理（tolerance: refDb ${REF_DB_ATOL} / 利得 ${GAIN_RTOL}` +
        ` / 波形 ${WAVE_ATOL}）\n${rows.join("\n")}\n  最悪: refDb ${
          worstRefDb.toExponential(3)
        } / 利得 ${worstGain.toExponential(3)} / 波形 ${worstWave.toExponential(3)}`,
    );
  },
});

// ---- 枝の踏み分け（恒真化の遮断）------------------------------------------

Deno.test({
  name: "golden の 5 ケースが狙った枝を実際に踏んでいる",
  ignore: !RUNNABLE,
  fn: async () => {
    const meta = readMeta();
    const short = meta.cases["ref-short"];
    const odd = meta.cases["ref-odd"];
    const hot = meta.cases["ref-hot"];
    const base = meta.cases["ref-default"];

    // ① 0.5 秒未満のゼロ pad 測定枝。
    assertEquals(short.seconds < 0.5, true, `ref-short が ${short.seconds} 秒 — 枝を踏まない`);
    // ② peak 制限枝。**LUFS 経路でこれを踏むのは ref-short だけ**（ref-hot は上流の
    //    「正規化しない」経路で peak を踏むケースで、−16 正規化を通すと利得 0.27 倍で
    //    peak が 0.81 まで下がり、制限が掛からない）。
    const normalizedShort = normalizeReference(
      (await readCase("ref-short"))["raw"],
      SAMPLE_RATE,
    );
    assertEquals(normalizedShort.peakBeforeScale > 1, true, "ref-short が peak > 1 にならない");
    assertEquals(normalizedShort.peakGain < 1, true, "ref-short で peak 制限が掛かっていない");
    const normalizedHot = normalizeReference((await readCase("ref-hot"))["raw"], SAMPLE_RATE);
    assertEquals(
      normalizedHot.peakGain,
      1,
      "ref-hot が −16 正規化の後で peak 制限を踏んだ（前提が変わった — doc の説明を直す）",
    );
    assertEquals(hot.peakGain < 1, true, "golden の ref-hot が peak 制限枝を踏んでいない");
    // ③ reflect pad の端数枝と恒等枝。
    assertEquals(base.padSamples, 0, "ref-default が pad の恒等枝を踏んでいない");
    assertEquals(
      0 < odd.padSamples && odd.padSamples < HOP_LENGTH,
      true,
      `ref-odd の pad が ${odd.padSamples} サンプル — 端数枝を踏んでいない`,
    );
  },
});

Deno.test({
  name: "ブロックのゼロ pad: ref-odd と ref-plain の LUFS が一致しない（末尾を捨てていない）",
  ignore: !RUNNABLE,
  fn: async () => {
    // ref-odd は ref-plain の先頭 144,000 サンプルに 777 サンプル足したもの。末尾の欠けた
    // ブロックを捨てる実装（`torch.nn.Unfold` の綴り）では、この 777 サンプルが 1 ブロックも
    // 作らないので**両者の LUFS が厳密に一致**してしまう。`julius.core.unfold` はゼロ pad して
    // 28 ブロック目を作るので一致しない — 実測の段差は 0.074 LU。
    const plain = integratedLoudness((await readCase("ref-plain"))["raw"], SAMPLE_RATE);
    const odd = integratedLoudness((await readCase("ref-odd"))["raw"], SAMPLE_RATE);
    const gap = Math.abs(odd - plain);
    console.log(
      `[parity] ブロックのゼロ pad: ref-odd − ref-plain = ${(odd - plain).toFixed(6)} LU`,
    );
    assertEquals(
      gap > 100 * REF_DB_ATOL,
      true,
      `ref-odd と ref-plain の LUFS 差が ${gap} しかない — 末尾の欠けたブロックを捨てている`,
    );
  },
});
