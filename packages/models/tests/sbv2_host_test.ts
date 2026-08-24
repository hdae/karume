// SBV2 のホストグルー（`src/sbv2/host/` の純関数）と、その成果物の梱包の挙動テスト。
//
// front と voice の間に挟まる「データ依存 shape」の計算（継続長 → フレーム展開 → z_p）と、
// 成果物の梱包（WAV は `src/audio/` のファミリ非依存層・dump は example 側）。ここは golden
// E2E の外側 — グラフを 1 つも実行しないので、誤りは「音の長さが変わる」「無音になる」形で
// しか出ず、数値突合では捕まらない。

import { assert, assertEquals, assertThrows } from "@std/assert";
import { parseSafetensors } from "@karume/runtime";
import { Sbv2InputError } from "../src/sbv2/errors.ts";
import { durationsToFrames } from "../src/sbv2/host/duration.ts";
import { buildZp } from "../src/sbv2/host/latent.ts";
import { Randn } from "../src/sbv2/host/random.ts";
import { encodeWav } from "../src/audio/wav.ts";
// dump の書き出しは**開発用の契約**（torch 参照へ渡す運搬形式）なのでテスト側ヘルパに置いて
// ある。パッケージの公開面には出ないが、梱包が閉じていること自体はここで固定する。
import { writeSafetensors } from "./helpers/safetensors-write.ts";

/**
 * `logw = log(w)` の逆算（テストが期待フレーム数を式ではなく値で書けるようにする）。
 *
 * MUST: `w` に整数を渡さない。`exp(fround(log(n)))` は n の直上・直下どちらにも落ちうるので
 * `ceil` が n か n+1 かで揺れる（ceil 境界の性質そのものであって実装の欠陥ではない）。
 */
const logwFor = (frames: readonly number[]): Float32Array => Float32Array.from(frames, Math.log);

/** 上限を主題にしないテストで渡す上限（実配布形の `maxFrames` と同じ 4096）。 */
const AMPLE_FRAMES = 4096;

Deno.test("durationsToFrames: exp → ceil でフレーム数になり、展開列は単調非減少", () => {
  // sdp と dp を同じにして混合を無視できる形にし、ceil の挙動だけを見る。
  const logw = logwFor([1.2, 1.8, 0.3]);
  const plan = durationsToFrames(logw, logw, new Float32Array([1, 1, 1]), 0.5, 1, AMPLE_FRAMES);
  assertEquals([...plan.wCeil], [2, 2, 1]);
  assertEquals(plan.totalFrames, 5);
  assertEquals([...plan.expandIdx], [0, 0, 1, 1, 2]);
});

Deno.test("durationsToFrames: x_mask が 0 の音素は 0 フレームになる", () => {
  const logw = logwFor([1.2, 5.4]);
  const plan = durationsToFrames(logw, logw, new Float32Array([1, 0]), 0, 1, AMPLE_FRAMES);
  assertEquals([...plan.wCeil], [2, 0]);
  assertEquals([...plan.expandIdx], [0, 0]);
});

Deno.test("durationsToFrames: length_scale はフレーム数に比例して効く", () => {
  const logw = logwFor([1.2, 1.2]);
  const mask = new Float32Array([1, 1]);
  assertEquals([...durationsToFrames(logw, logw, mask, 0, 1, AMPLE_FRAMES).wCeil], [2, 2], "w=1.2");
  assertEquals([...durationsToFrames(logw, logw, mask, 0, 2, AMPLE_FRAMES).wCeil], [3, 3], "w=2.4");
});

Deno.test("durationsToFrames: 故障注入 — sdp_ratio の混合が効いている", () => {
  // sdp と dp に違う値を渡し、比を変えると結果が変わることを見る。片方だけ使う実装
  // （混合を書き忘れ）は ratio を変えても同じ答えになる。
  const sdp = logwFor([9.5, 9.5]);
  const dp = logwFor([1.5, 1.5]);
  const mask = new Float32Array([1, 1]);
  assertEquals(
    [...durationsToFrames(sdp, dp, mask, 0, 1, AMPLE_FRAMES).wCeil],
    [2, 2],
    "r=0 は dp だけ",
  );
  assertEquals(
    [...durationsToFrames(sdp, dp, mask, 1, 1, AMPLE_FRAMES).wCeil],
    [10, 10],
    "r=1 は sdp だけ",
  );
  // 混合は logw（対数）側で行うので中間比は幾何平均 √(9.5·1.5) ≈ 3.775 になる。
  const mixed = durationsToFrames(sdp, dp, mask, 0.5, 1, AMPLE_FRAMES).wCeil;
  assertEquals([...mixed], [4, 4]);
});

Deno.test("durationsToFrames: 長さ不一致・全 0 フレームは落とす", () => {
  const logw = logwFor([1, 1]);
  // どちらも front 出力が壊れている＝内部異常なので、素の `Error`（500 相当）のまま。
  // 呼び手が text や lengthScale を直しても直らない = 入力起因の分岐へ流してはならない。
  const mismatch = assertThrows(
    () => durationsToFrames(logw, logwFor([1]), new Float32Array([1, 1]), 0, 1, AMPLE_FRAMES),
    Error,
    "長さ不一致",
  );
  assert(!(mismatch instanceof Sbv2InputError), "長さ不一致は入力起因ではない");
  const empty = assertThrows(
    () => durationsToFrames(logw, logw, new Float32Array([0, 0]), 0, 1, AMPLE_FRAMES),
    Error,
    "総フレーム数が 0",
  );
  assert(!(empty instanceof Sbv2InputError), "総フレーム数 0 は入力起因ではない");
});

Deno.test("durationsToFrames: 総フレーム数が配布形の上限を超えたら展開列を確保する前に落とす", () => {
  // 同じ text でも lengthScale を上げれば総フレーム数はいくらでも伸びる（5 → 8）。上限を
  // 超えた要求は、長さ Ty の展開列も下流の T×T 表（8·T² bytes 級）も確保せずにここで止まる。
  const logw = logwFor([1.2, 1.8, 0.3]);
  const mask = new Float32Array([1, 1, 1]);
  assertEquals(durationsToFrames(logw, logw, mask, 0.5, 1, 7).totalFrames, 5);
  // 型は `Sbv2InputError`（400 相当）— text を短く分けるか lengthScale を下げれば通る要求で、
  // 内部不変条件の破れ（素の `Error` = 500）とは呼び手の分岐先が違う（`errors.ts` の分類軸）。
  assertThrows(
    () => durationsToFrames(logw, logw, mask, 0.5, 2, 7),
    Sbv2InputError,
    "総フレーム数 8 が配布形の上限 maxFrames=7 を超えている",
  );
});

Deno.test("durationsToFrames: ちょうど上限は通る（境界を 1 つ内側に取っていない）", () => {
  const logw = logwFor([1.2, 1.8, 0.3]);
  const plan = durationsToFrames(logw, logw, new Float32Array([1, 1, 1]), 0.5, 2, 8);
  assertEquals(plan.totalFrames, 8);
  assertEquals(plan.expandIdx.length, 8, "上限ちょうどの展開列が確保されている");
});

Deno.test("buildZp: m_p を展開して noise·exp(logs_p)·scale を足す", () => {
  // C=2 / P=2 / Ty=3。logs_p = 0 なので exp は 1 になり、期待値を手で書ける。
  const mP = new Float32Array([1, 2, 10, 20]);
  const logsP = new Float32Array([0, 0, 0, 0]);
  const expandIdx = Int32Array.from([0, 1, 1]);
  const noise = new Float32Array([1, 0, -1, 2, 0, -2]);
  const zP = buildZp(mP, logsP, expandIdx, 2, noise, 0.5);
  assertEquals([...zP], [1 + 0.5, 2, 2 - 0.5, 10 + 1, 20, 20 - 1]);
});

Deno.test("buildZp: 故障注入 — 展開インデックスがずれると値も変わる", () => {
  // 「gather を無視して先頭 Ty 列を切り出す」実装は expandIdx を変えても同じ答えを返す。
  const mP = new Float32Array([1, 2, 3]);
  const logsP = new Float32Array([0, 0, 0]);
  const noise = new Float32Array(3);
  const a = buildZp(mP, logsP, Int32Array.from([0, 1, 2]), 1, noise, 1);
  const b = buildZp(mP, logsP, Int32Array.from([0, 0, 2]), 1, noise, 1);
  assertEquals(a.length, b.length, "長さは同じ（= 長さでは検出できない形）");
  assert([...a].some((value, index) => value !== b[index]), "展開を変えても同じ値");
});

Deno.test("buildZp: 形状の食い違いは落とす", () => {
  const mP = new Float32Array([1, 2, 3, 4]);
  const logsP = new Float32Array([0, 0, 0, 0]);
  const idx = Int32Array.from([0, 1]);
  assertThrows(() => buildZp(mP, logsP, idx, 3, new Float32Array(6), 1), Error, "割り切れない");
  assertThrows(() => buildZp(mP, logsP, idx, 2, new Float32Array(3), 1), Error, "ノイズ長");
  assertThrows(
    () => buildZp(mP, logsP, Int32Array.from([0, 9]), 2, new Float32Array(4), 1),
    Error,
    "展開インデックス",
  );
});

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
  // scale は同じ列に掛かるだけ（列そのものは seed で決まる）。
  const plain = new Randn(3).normals(16);
  const scaled = new Randn(3).normals(16, 2);
  for (const [index, value] of plain.entries()) {
    assertEquals(scaled[index], Math.fround(value * 2));
  }
});

Deno.test("Randn: 奇数長でも Box–Muller の対を持ち越さない", () => {
  // 持ち越すと「同じ seed でも呼び出し順で列が変わる」形になり、dump した乱数列との
  // 対応が読めなくなる。奇数長 → 次の呼び出し、が偶数長 2 回と同じ列になることで固定する。
  const split = new Randn(5);
  const head = split.normals(3);
  const tail = split.normals(2);
  const fresh = new Randn(5);
  assertEquals([...head], [...fresh.normals(3)]);
  assertEquals([...tail], [...fresh.normals(2)]);
});

Deno.test("encodeWav: RIFF ヘッダの各欄が仕様どおり", () => {
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
  assertEquals(view.getUint16(22, true), 1, "モノラル");
  assertEquals(view.getUint32(24, true), 44100);
  assertEquals(view.getUint32(28, true), 44100 * 2, "byte rate");
  assertEquals(view.getUint16(32, true), 2, "block align");
  assertEquals(view.getUint16(34, true), 16);
  assertEquals(text(36), "data");
  assertEquals(view.getUint32(40, true), 6);
});

Deno.test("encodeWav: 値域外はクリップし、丸めは floor(x+0.5) 相当", () => {
  // ±0.5 は 32767 倍すると丁度 ±16383.5 になる（0.5 は f32 で厳密）。ここが
  // **半端値の丸め規則を分ける唯一の観測点**で、`Math.round` は常に +∞ 方向へ倒す
  // （+16384 / −16383）。Python 組み込みの `round`（偶数丸め）だと −16384 になり、
  // 3 本の wav が 1 LSB 食い違う — 参照側は `floor(x+0.5)` で揃えてある。
  const wav = encodeWav(new Float32Array([2, -2, 0.5, -0.5]), 8000);
  const view = new DataView(wav.buffer);
  assertEquals(view.getInt16(44, true), 32767, "上限クリップ");
  assertEquals(view.getInt16(46, true), -32767, "下限クリップ");
  assertEquals(view.getInt16(48, true), 16384);
  assertEquals(view.getInt16(50, true), -16383);
});

Deno.test("encodeWav: 不正なサンプリング周波数は落とす", () => {
  assertThrows(() => encodeWav(new Float32Array(1), 0), RangeError);
  assertThrows(() => encodeWav(new Float32Array(1), 44100.5), RangeError);
});

Deno.test("writeSafetensors: 書いたものを読み戻せる（dump の運搬が閉じている）", () => {
  const bytes = writeSafetensors(
    new Map([
      ["ids", { dtype: "I32" as const, shape: [1, 3], data: Int32Array.from([7, 8, 9]) }],
      ["noise", { dtype: "F32" as const, shape: [2, 2], data: Float32Array.from([1, 2, 3, 4]) }],
    ]),
    { demo: '{"text":"あ"}' },
  );
  // データ節の開始（8 + ヘッダ長）が 8 バイト境界に乗る（参照実装の詰め方）。
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

Deno.test("writeSafetensors: shape と data 長の食い違いは落とす", () => {
  assertThrows(
    () =>
      writeSafetensors(
        new Map([["x", { dtype: "F32" as const, shape: [3], data: new Float32Array(2) }]]),
        {},
      ),
    Error,
    "要素数",
  );
});
