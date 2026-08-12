// Irodori のホスト層（`src/irodori/host/` の純関数）の挙動テスト。GPU も実資産も要らない。
//
// ここが受け持つのは ADR 0047 決定 4 の「ホスト残置」— S の決定・区間マスク・t スケジュール・
// CFG 合成・Euler 更新・参照 latent の patch・縮約・token 列の組み立て。誤りの出方が
// 「例外ではなく別の音」なので、値そのものを固定する。
//
// 数値パリティ（`t_embed` 表の golden 突合）は資産が要るので `irodori_t_embed_test.ts`。

import { assert, assertEquals, assertNotStrictEquals, assertThrows } from "@std/assert";
import { buildDitMask, SEGMENT_ORDER } from "../src/irodori/host/mask.ts";
import { packIds } from "../src/irodori/host/pack.ts";
import { patchReferenceLatent } from "../src/irodori/host/patch.ts";
import { prependMeanToken, rowMean } from "../src/irodori/host/pooling.ts";
import { Randn } from "../src/irodori/host/random.ts";
import {
  bankerRound,
  type SampleBounds,
  sequenceLengthFromLogFrames,
  sequenceLengthFromSeconds,
} from "../src/irodori/host/round.ts";
import { combineCfg, eulerStep, tSchedule } from "../src/irodori/host/sampler.ts";
import { IrodoriTokenizer, type IrodoriTokenizerAssets } from "../src/irodori/text/tokenizer.ts";

/** 実重み v4-small の運用値（`pipelineConfig` が運ぶ数と同じ）。 */
const BOUNDS: SampleBounds = {
  frameRate: 25,
  minSeconds: 0.5,
  maxSeconds: 30,
  sampleRate: 48000,
  hopLength: 1920,
};

// ---- S の決定 -------------------------------------------------------------

Deno.test("bankerRound: 0.5 ちょうどは偶数側（Math.round との差がここに出る）", () => {
  assertEquals(bankerRound(0.5), 0);
  assertEquals(bankerRound(1.5), 2);
  assertEquals(bankerRound(2.5), 2);
  assertEquals(bankerRound(3.5), 4);
  // 0.5 でなければ最近接（Math.round と同じ）。
  assertEquals(bankerRound(2.4), 2);
  assertEquals(bankerRound(2.6), 3);
  // 負側も偶数へ倒す（Python の round と同じ）。
  assertEquals(bankerRound(-2.5), -2);
  assertEquals(bankerRound(-3.5), -4);
});

Deno.test("sequenceLengthFromLogFrames: golden 2 ケースの log frames から同じ S が出る", () => {
  // 値は `outputs/series/irodori-v4-small/pipeline/meta.json` の `duration.logFrames`
  // （full = 161 フレーム / no-ref = 116 フレーム）。
  assertEquals(sequenceLengthFromLogFrames(5.087219, BOUNDS).frames, 161);
  assertEquals(sequenceLengthFromLogFrames(4.761631, BOUNDS).frames, 116);
});

Deno.test("sequenceLengthFromLogFrames: clamp は 13（0.5s）と 750（30s）", () => {
  // ceil(0.5×25) = 13 / floor(30×25) = 750 — 端の丸め方が非対称なのは上流の綴りどおり。
  assertEquals(sequenceLengthFromLogFrames(0, BOUNDS).frames, 13, "expm1(0)=0 は下限へ");
  assertEquals(sequenceLengthFromLogFrames(10, BOUNDS).frames, 750, "expm1(10)≈22025 は上限へ");
});

Deno.test("sequenceLengthFromLogFrames: この経路の切り出し長は S × hopLength ちょうど", () => {
  // duration が決めた S はそのまま出力長 — 切り出しは実質 no-op になる。
  assertEquals(sequenceLengthFromLogFrames(5.087219, BOUNDS).targetSamples, 161 * 1920);
});

Deno.test("sequenceLengthFromSeconds: 秒 → サンプル → フレーム（上流 manual_seconds の綴り）", () => {
  assertEquals(sequenceLengthFromSeconds(1, BOUNDS), { frames: 25, targetSamples: 48000 });
  // 端数は切り上げ（1 フレーム足りない発話を作らない）。1.004s = 48,192 サンプル → 26 フレーム。
  assertEquals(sequenceLengthFromSeconds(1.004, BOUNDS), { frames: 26, targetSamples: 48192 });
  assertEquals(sequenceLengthFromSeconds(0.1, BOUNDS), {
    frames: 13,
    targetSamples: 24000,
  }, "下限 0.5s へ clamp");
  assertEquals(sequenceLengthFromSeconds(100, BOUNDS), {
    frames: 750,
    targetSamples: 1_440_000,
  }, "上限 30s へ clamp");
});

Deno.test("sequenceLengthFromSeconds: サンプル境界の 1 フレーム差（frameRate 経由との分かれ目）", () => {
  // 1.000005s × 48,000 = 48,000.24 サンプル → trunc して 48,000 → 25 フレーム。
  // frameRate 経由（ceil(1.000005×25) = 26）だと 1 フレーム長い発話になる。
  assertEquals(sequenceLengthFromSeconds(1.000005, BOUNDS), {
    frames: 25,
    targetSamples: 48000,
  });
  // 1 サンプルでも超えれば繰り上がる（切り上げの向きが逆でないことの確認）。
  assertEquals(sequenceLengthFromSeconds(1.0000209, BOUNDS), {
    frames: 26,
    targetSamples: 48001,
  });
});

Deno.test("sequenceLengthFromSeconds: 切り出し長は S × hopLength より短くなりうる", () => {
  // 26 フレーム = 49,920 サンプルぶん decode するが、返す波形は 48,192 サンプル。
  const plan = sequenceLengthFromSeconds(1.004, BOUNDS);
  assertEquals(plan.frames * BOUNDS.hopLength - plan.targetSamples, 1728);
});

// ---- 区間マスク -----------------------------------------------------------

/** exporter の `tests/test_irodori_pipeline.py::TestSegmentMasks` と同じ値。 */
const CAPS = { text: 4, speaker: 5, caption: 6 };
const USED = { text: 2, speaker: 3, caption: 4 };

Deno.test("buildDitMask: 連結順は self → text → speaker → caption・各区間は使用長 prefix", () => {
  const mask = buildDitMask(3, USED, CAPS);
  assertEquals(mask.length, 3 + 4 + 5 + 6);
  assertEquals([...mask.slice(0, 3)], [1, 1, 1], "self は latent 長ぶん全 1");
  assertEquals([...mask.slice(3, 7)], [1, 1, 0, 0], "text");
  assertEquals([...mask.slice(7, 12)], [1, 1, 1, 0, 0], "speaker");
  assertEquals([...mask.slice(12)], [1, 1, 1, 1, 0, 0], "caption");
});

Deno.test("buildDitMask: uncond 変種は自分の区間だけを落とす（ADR 0047 決定 1）", () => {
  const cond = buildDitMask(3, USED, CAPS);
  const offsets = { text: 3, speaker: 7, caption: 12 };
  for (const segment of SEGMENT_ORDER) {
    const got = buildDitMask(3, USED, CAPS, segment);
    const start = offsets[segment];
    const end = start + CAPS[segment];
    assertEquals([...got.slice(start, end)], new Array(CAPS[segment]).fill(0), `${segment} 区間`);
    assertEquals([...got.slice(0, start)], [...cond.slice(0, start)], `${segment} の前`);
    assertEquals([...got.slice(end)], [...cond.slice(end)], `${segment} の後`);
  }
});

Deno.test("buildDitMask: 使用長が宣言長を超えたら落とす（右 pad の破れ）", () => {
  assertThrows(
    () => buildDitMask(3, { ...USED, speaker: 6 }, CAPS),
    Error,
    "speaker 区間の使用長 6",
  );
});

// ---- t スケジュールと CFG の窓 -------------------------------------------

Deno.test("tSchedule: 長さ steps+1・始点 0.999・終端 0・狭義単調減少", () => {
  const schedule = tSchedule(40, 0.999);
  assertEquals(schedule.length, 41);
  assertEquals(schedule[0], Math.fround(0.999));
  assertEquals(schedule[40], 0);
  for (let index = 1; index <= 40; index += 1) {
    assert(schedule[index] < schedule[index - 1], `t が減っていない（index=${index}）`);
  }
});

Deno.test("tSchedule: CFG の窓 [0.5, 1.0] に入るのは 40 step 中 20 本", () => {
  // これが cond 40 + uncond 20×(有効条件数) = 100 / 60 forward の出どころ（meta.json）。
  const schedule = tSchedule(40, 0.999);
  const inWindow = [...schedule.slice(0, 40)].filter((t) => t >= 0.5 && t <= 1);
  assertEquals(inWindow.length, 20);
  assertEquals(40 + 20 * 3, 100, "full ケース（text/speaker/caption の 3 本）");
  assertEquals(40 + 20 * 1, 60, "no-ref ケース（text だけ）");
});

// ---- CFG 合成と Euler 更新 -----------------------------------------------

Deno.test("combineCfg: 変種 0 本なら cond の写し（同じ配列を返さない）", () => {
  const cond = Float32Array.from([1, -2, 0.5]);
  const combined = combineCfg(cond, []);
  assertEquals([...combined], [1, -2, 0.5]);
  assertNotStrictEquals(combined, cond, "Session の出力バッファを掴んだまま次の run へ入らない");
});

Deno.test("combineCfg: v_cond + Σ s_k(v_cond − v_k) を順番どおりに積む", () => {
  const cond = Float32Array.from([2]);
  const text = { scale: 3, velocity: Float32Array.from([1]) };
  const caption = { scale: 5, velocity: Float32Array.from([1.5]) };
  // 2 + 3·(2−1) = 5 → 5 + 5·(2−1.5) = 7.5
  assertEquals([...combineCfg(cond, [text, caption])], [7.5]);
});

Deno.test("combineCfg: 故障注入 — 差の基準は必ず v_cond（直前の合成結果ではない）", () => {
  // 「v = v + s(v − v_k)」と書き違えると、2 本目の差が合成済みの値との差になる。
  const cond = Float32Array.from([2]);
  const variants = [
    { scale: 1, velocity: Float32Array.from([0]) },
    { scale: 1, velocity: Float32Array.from([0]) },
  ];
  // 正: 2 + (2−0) + (2−0) = 6 ／ 誤: 2 + (2−0) = 4 → 4 + (4−0) = 8
  assertEquals([...combineCfg(cond, variants)], [6]);
});

Deno.test("combineCfg: 長さの食い違いは落とす", () => {
  assertThrows(
    () => combineCfg(Float32Array.from([1, 2]), [{ scale: 1, velocity: Float32Array.from([1]) }]),
    Error,
    "CFG 変種の長さ 1",
  );
});

Deno.test("eulerStep: x + v·Δt（Δt は負）", () => {
  const x = Float32Array.from([1, 2]);
  const v = Float32Array.from([0.5, -4]);
  assertEquals([...eulerStep(x, v, -0.5)], [0.75, 4]);
});

// ---- 参照 latent の patch と縮約 -----------------------------------------

Deno.test("patchReferenceLatent: 端数のフレームは捨てて reshape する", () => {
  // latentDim=2 / patchSize=4 → 9 フレームのうち 8 フレーム（2 トークン）だけが残る。
  const latent = Float32Array.from({ length: 9 * 2 }, (_value, index) => index);
  const patched = patchReferenceLatent(latent, 2, 4);
  assertEquals(patched.tokens, 2);
  assertEquals(patched.width, 8);
  assertEquals(patched.data.length, 16);
  assertEquals([...patched.data.slice(0, 8)], [0, 1, 2, 3, 4, 5, 6, 7]);
  assertEquals(patched.data[15], 15, "捨てられるのは 9 フレーム目だけ");
});

Deno.test("patchReferenceLatent: 1 トークンも作れない参照は落とす", () => {
  assertThrows(() => patchReferenceLatent(new Float32Array(3 * 2), 2, 4), Error, "満たない");
  assertThrows(() => patchReferenceLatent(new Float32Array(5), 2, 4), Error, "倍数でない");
});

Deno.test("rowMean: 列ごとの単純平均（マスク全 True の経路しか通らない）", () => {
  const state = Float32Array.from([1, 10, 3, 20]);
  assertEquals([...rowMean(state, 2, 2)], [2, 15]);
});

Deno.test("prependMeanToken: 平均トークンは**先頭**に入る（duration の speaker_vec の出どころ）", () => {
  const state = Float32Array.from([1, 10, 3, 20]);
  const prepended = prependMeanToken(state, 2, 2);
  assertEquals(prepended.length, 6);
  assertEquals([...prepended.slice(0, 2)], [2, 15], "先頭が平均");
  assertEquals([...prepended.slice(2)], [1, 10, 3, 20], "以降は元の系列そのもの");
});

// ---- token 列の組み立て ---------------------------------------------------

/** 最小の語彙（Unigram の格子だけを踏む — 実語彙のパリティは `irodori_text_test.ts`）。 */
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

Deno.test("packIds: BOS 前置 + pad 無し（静的方式の詰めた列）", () => {
  assertEquals([...packIds(tinyTokenizer(), "あいう", 8, "text")], [1, 4, 5, 6]);
});

Deno.test("packIds: 本文の予算は maxLength−1（BOS のぶんを空ける）", () => {
  assertEquals([...packIds(tinyTokenizer(), "あいう", 3, "text")], [1, 4, 5]);
});

Deno.test("packIds: 正規化を通す（全角空白は消える — `normalize.ts` の段 ①）", () => {
  assertEquals([...packIds(tinyTokenizer(), "あ　い", 8, "text")], [1, 4, 5]);
});

Deno.test("packIds: 正規化後に空なら落とす（BOS だけの列を通さない）", () => {
  assertThrows(
    () => packIds(tinyTokenizer(), "　 ", 8, "caption"),
    Error,
    "caption が正規化後に空",
  );
});

// ---- 初期ノイズの生成器 ---------------------------------------------------
//
// `host/random.ts` は `sbv2/host/random.ts` の**意図的な複製**（family 間で import し合わない —
// モジュール doc）。複製である以上、片側だけが動いても誰も気づかない席がここに要る。
// 縛るのは sbv2 側と同じ 3 点（同 seed 同列 / 標準正規らしい統計量 / 対を持ち越さない）。

Deno.test("Randn: 同じ seed は同じ列・違う seed は違う列", () => {
  const a = new Randn(7).normals(64);
  const b = new Randn(7).normals(64);
  const c = new Randn(8).normals(64);
  assertEquals([...a], [...b]);
  assert([...a].some((value, index) => value !== c[index]), "seed を変えても同じ列");
});

Deno.test("Randn: 標準正規らしい統計量を持ち、scale が線形に効く", () => {
  const samples = new Randn(1).normals(20000);
  const mean = samples.reduce((sum, value) => sum + value, 0) / samples.length;
  const variance = samples.reduce((sum, value) => sum + (value - mean) ** 2, 0) / samples.length;
  assert(Math.abs(mean) < 0.05, `平均 ${mean} が 0 から離れすぎ`);
  assert(Math.abs(variance - 1) < 0.05, `分散 ${variance} が 1 から離れすぎ`);
  // scale は同じ列に掛かるだけ（列そのものは seed で決まる）。この家族の呼び出しは scale を
  // 使わないが、複製の drift はここでしか出ない。
  const plain = new Randn(3).normals(16);
  const scaled = new Randn(3).normals(16, 2);
  for (const [index, value] of plain.entries()) {
    assertEquals(scaled[index], Math.fround(value * 2));
  }
});

Deno.test("Randn: 奇数長でも Box–Muller の対を持ち越さない", () => {
  // 5 要素を 1 回で引くと 3 対を消費して [cos1, sin1, cos2, sin2, cos3]。3 + 2 に割ると
  // 1 回目が 2 対を丸ごと消費して [cos1, sin1, cos2]（sin2 は捨てる）、2 回目は 3 対目から
  // [cos3, sin3]。持ち越す実装なら 2 回目の先頭が sin2 になるので、そこが観測点。
  const single = new Randn(5).normals(5);
  const split = new Randn(5);
  const head = split.normals(3);
  const tail = split.normals(2);
  assertEquals([...head], [...single.slice(0, 3)], "先頭 3 要素までは同じ列");
  assertEquals(tail[0], single[4], "2 回目は次の対の cos から始まる（= 持ち越していない）");
  assert(tail[0] !== single[3], "2 回目の先頭が捨てたはずの sin を拾っている");
});
