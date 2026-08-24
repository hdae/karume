// 母音検出のホスト層（`src/vowel-detector/` の特徴抽出と後処理）の Python 正本とのパリティ、
// および挙動テスト。GPU も実資産も要らないので常時走る。
//
// フィクスチャ `vowel-detector/parity.json` は上流 `@hdae/vowel-detector`（MIT・同一著者）が
// 持つ参照をそのまま畳んだもので、karume 側では 1 つも作り直していない（provenance は
// フィクスチャ先頭の `_doc`）。正本は Python 実装
// （`training/src/vowel_detector/{features,dsp,postprocess}.py`）で、上流の TS 実装はそこからの
// 移植 = **正ではない**。
//
// 縛るのは振る舞い: 「同じ波形を入れたら Python と同じ 83 次元特徴が出る」「同じロジットを
// 入れたら Python と同じ `.lab` が出る」。FFT の回し方も平滑化の内部表現も縛らない。
//
// ## tolerance の導出（実測）
//
// 特徴の参照は **f32 で保存**されている（Python が `astype(np.float32)` した値の JSON）。
// フィクスチャの参照特徴は絶対値 ≤ 3.7179 なので、その帯（[2,4)）の f32 ULP は 2.384e-7。
// つまり**参照側に既に 1 ULP 級の丸めが載っている**のが下限で、これより細かい一致は原理的に
// 測れない。実測:
//
//  ・移植（f64 FFT）        max abs diff 4.768e-7 = **2 ULP**（log-mel 80 列で最大。
//                           voiced 2.98e-8 / logEnergy 5.96e-8 / ZCR は完全一致）
//  ・上流 TS（f32 FFT）     max abs diff 5.424e-6（同じフィクスチャ・同じ mel 基底）
//
// 移植側が 1 桁良いのは `np.fft` が f64 で回ることに合わせたため（`src/vowel-detector/fft.ts`
// の MUST）。したがって門は「2 ULP の 2 倍」= {@link FEATURE_ATOL} に置く。上流 TS 実装の
// 5.42e-6 はこの門を**通らない** — f32 FFT へ書き戻す変更が黙って入らない幅である。
//
// 後処理は tolerance を持たない（`.lab` の**完全一致**）。段 2 以降は離散的な区間割りなので、
// 数値が僅かにずれても文字列は割れるか一致するかのどちらかにしかならない。

import { assert, assertEquals, assertThrows } from "@std/assert";
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
const fixture = JSON.parse(await Deno.readTextFile(FIXTURE_PATH)) as Fixture;

/** 参照 f32 の 2 ULP（|x| ≤ 3.72 の帯で 4.768e-7）の 2 倍。導出はファイル冒頭。 */
const FEATURE_ATOL = 1e-6;

const melBasis = Float32Array.from(fixture.melBasis.flat());
const audio = Float32Array.from(fixture.audio);

// ---- フィクスチャ本体 -------------------------------------------------------

Deno.test("フィクスチャが空でない（取り違えで全ケース素通しになっていない）", () => {
  // 1 本目に置く。読み違い / 生成失敗で行数が 0 になると、以降のループが 0 回になって
  // 「緑だが何も検証していない」状態が黙って成立する。
  assertEquals(fixture.melBasis.length, N_MELS);
  assertEquals(fixture.features.length, 108, "参照特徴のフレーム数");
  assertEquals(fixture.logits.length, 54, "参照ロジットのフレーム数");
  assert(fixture.lab.endsWith("\n"), "参照 .lab が改行で終わっていない");
  assert(fixture.lab.trimEnd().split("\n").length >= 2, "参照 .lab が 1 行しかない");
  for (const row of fixture.melBasis) assertEquals(row.length, MEL_BINS);
  for (const row of fixture.features) assertEquals(row.length, FEATURE_DIM);
  for (const row of fixture.logits) assertEquals(row.length, LIPSYNC_CLASSES.length);
});

Deno.test("学習時の定数が上流の配布 config と一致する", () => {
  // モジュール定数は Python の module 定数の写し。再学習で幾何が変わったら、特徴が
  // 「それらしい別の値」になる前にここで落ちる。
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
  // 平滑化の 2 定数はモジュール内に閉じているので、値そのものは挙動テスト側で固定する。
  assertEquals(constants.switchPenalty, 4);
  assertEquals(constants.minDurationFrames, 2);
});

// ---- ① 特徴抽出のパリティ ---------------------------------------------------

Deno.test("特徴パリティ: 同じ波形から Python と同じ 83 次元が出る", () => {
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
  // MUST: NaN / ±Inf は不合格。素朴な差分判定だと NaN が「差 0」に化けて通る。
  assertEquals(nonFinite, 0, "特徴に非有限の標本");
  assert(worst <= FEATURE_ATOL, `特徴の max abs diff ${worst.toExponential(3)}（${worstAt}）`);
});

Deno.test("特徴: フレーム数は floor((n − N_FFT) / HOP) + 1（librosa の center=False）", () => {
  const at = (samples: number): number =>
    extractFeatures(new Float32Array(samples), melBasis).frames;
  assertEquals(at(N_FFT), 1, "N_FFT ちょうどで 1 フレーム");
  assertEquals(at(N_FFT + HOP - 1), 1, "hop に 1 サンプル足りなければ増えない");
  assertEquals(at(N_FFT + HOP), 2);
  assertEquals(at(SAMPLE_RATE), Math.floor((SAMPLE_RATE - N_FFT) / HOP) + 1);
});

Deno.test("特徴: 1 フレームも取れない波形は fail loudly（NaN を撒かない）", () => {
  assertThrows(
    () => extractFeatures(new Float32Array(N_FFT - 1), melBasis),
    Error,
    "サンプルしかない",
  );
});

Deno.test("特徴: mel 基底の要素数が違えば受け付けない", () => {
  assertThrows(
    () => extractFeatures(audio, new Float32Array(N_MELS * MEL_BINS - 1)),
    Error,
    "mel 基底が",
  );
  assertThrows(
    () => extractFeatures(audio, new Float32Array(MEL_BINS * N_MELS + 1)),
    Error,
    "mel 基底が",
  );
});

Deno.test("特徴: ZCR は np.signbit と同じで −0 を負と数える", () => {
  // 交互に +0 / −0 を並べると、`value < 0` の実装では交差 0 本、`np.signbit` では
  // 全サンプル境界が交差になる。log-mel は全ゼロで縮退するが、82 列目だけが割れる。
  const alternating = new Float32Array(N_FFT);
  for (let index = 0; index < alternating.length; index += 1) {
    alternating[index] = index % 2 === 0 ? 0 : -0;
  }
  const { data } = extractFeatures(alternating, melBasis);
  assertEquals(data[N_MELS + 2], 1, "全サンプル境界が交差（ZCR = 1）");

  const positive = new Float32Array(N_FFT);
  assertEquals(extractFeatures(positive, melBasis).data[N_MELS + 2], 0, "全て +0 なら交差ゼロ");
});

// ---- ② 後処理のパリティ -----------------------------------------------------

Deno.test("後処理パリティ: 同じロジットから Python と同じ .lab が出る（完全一致）", () => {
  const logits = Float32Array.from(fixture.logits.flat());
  assertEquals(toLab(logitsToSegments(logits, fixture.logits.length)), fixture.lab);
});

Deno.test("後処理: ロジットの要素数がフレーム数と合わなければ受け付けない", () => {
  assertThrows(() => logitsToSegments(new Float32Array(8 * 3 + 1), 3), Error, "ロジットが");
  assertThrows(() => logitsToSegments(new Float32Array(0), 0), Error, "フレーム数");
});

Deno.test("後処理: 非有限ロジットは座標を添えて落とす（全区間 a の .lab に化けない）", () => {
  // 門が無いと NaN / +Infinity は log_softmax でその行を全て NaN にし、Viterbi の比較が
  // 軒並み偽になって「発話全体が a」の**書式として正当な** `.lab` が返る（数値異常が
  // どこにも表面化しない）。落ちること自体と、座標が名指しされることを縛る。
  const finite = Float32Array.from(fixture.logits.flat());
  // 対: 汚していない同じ配列は参照どおりの `.lab` を返す（常に落ちる門になっていない）。
  assertEquals(toLab(logitsToSegments(finite, fixture.logits.length)), fixture.lab);
  const at = 7 * LIPSYNC_CLASSES.length + 5; // frame 7 / class 5 = "N"
  for (const [poison, shown] of [[NaN, "NaN"], [Number.POSITIVE_INFINITY, "Infinity"]] as const) {
    const logits = finite.slice();
    logits[at] = poison;
    assertThrows(
      () => logitsToSegments(logits, fixture.logits.length),
      Error,
      `logitsToSegments: ロジット frame 7 / class 5（N）が非有限（${shown}）`,
    );
  }
});

Deno.test("Viterbi: 非有限の log 事後確率は座標を添えて落とす", () => {
  // `viterbiSmooth` は公開されていて単体で叩けるので、`logitsToSegments` の入口とは別に
  // ここでも塞ぐ（−Infinity は「あり得ないクラス」に見えて、格子全体を凍らせる）。
  const frames = 5;
  const logProbabilities = new Float64Array(frames * LIPSYNC_CLASSES.length);
  assertEquals(viterbiSmooth(logProbabilities, frames).length, frames, "全て有限なら通る");
  logProbabilities[3 * LIPSYNC_CLASSES.length + 6] = Number.NEGATIVE_INFINITY;
  assertThrows(
    () => viterbiSmooth(logProbabilities, frames),
    Error,
    "viterbiSmooth: log 事後確率 frame 3 / class 6（pau）が非有限（-Infinity）",
  );
});

// ---- ③ 平滑化と区間化の挙動 -------------------------------------------------

/**
 * 各フレームで指定クラスだけ +20 のロジット列（`overrides` は個別の上書き）。
 * 20 は切替ペナルティ 4 より十分大きいので、Viterbi は argmax に追随する。
 */
const logitsFor = (
  sequence: readonly (typeof LIPSYNC_CLASSES[number])[],
  overrides: Readonly<Record<number, Readonly<Record<string, number>>>> = {},
): Float32Array => {
  const classes = LIPSYNC_CLASSES.length;
  const logits = new Float32Array(sequence.length * classes);
  for (let frame = 0; frame < sequence.length; frame += 1) {
    logits[frame * classes + LIPSYNC_CLASSES.indexOf(sequence[frame])] = 20;
    for (const [label, value] of Object.entries(overrides[frame] ?? {})) {
      logits[frame * classes + LIPSYNC_CLASSES.indexOf(label as typeof LIPSYNC_CLASSES[number])] =
        value;
    }
  }
  return logits;
};

const labFor = (
  sequence: readonly (typeof LIPSYNC_CLASSES[number])[],
  overrides?: Readonly<Record<number, Readonly<Record<string, number>>>>,
): string => toLab(logitsToSegments(logitsFor(sequence, overrides), sequence.length));

Deno.test("Viterbi: 切替ペナルティ未満の 1 フレームの揺れでは乗り換えない", () => {
  // クラス 0 が毎フレーム +3、frame 2 だけクラス 1 が +3。乗り換えは往復で 8 の損なので
  // 3 の得では引き合わない = argmax（0,0,1,0,0）ではなく全て 0 になる。
  const frames = 5;
  const logProbabilities = new Float64Array(frames * LIPSYNC_CLASSES.length);
  for (let frame = 0; frame < frames; frame += 1) {
    logProbabilities[frame * LIPSYNC_CLASSES.length + (frame === 2 ? 1 : 0)] = 3;
  }
  assertEquals([...viterbiSmooth(logProbabilities, frames)], [0, 0, 0, 0, 0]);
});

Deno.test("Viterbi: ペナルティを上回る優位が続けば乗り換える", () => {
  const frames = 5;
  const logProbabilities = new Float64Array(frames * LIPSYNC_CLASSES.length);
  for (let frame = 0; frame < frames; frame += 1) {
    logProbabilities[frame * LIPSYNC_CLASSES.length + (frame < 2 ? 0 : 1)] = 20;
  }
  assertEquals([...viterbiSmooth(logProbabilities, frames)], [0, 0, 1, 1, 1]);
});

Deno.test("Viterbi: 格子の形が合わなければ受け付けない", () => {
  assertThrows(() => viterbiSmooth(new Float64Array(8 * 4 - 1), 4), Error, "log 事後確率が");
});

Deno.test("短区間マージ: 最小継続長未満は事後確率の高い隣へ塗り替える", () => {
  // frame 2 は i が突出（Viterbi は i を通る）が 1 フレームしかないので隣へ。隣接候補
  // a / e のうち frame 2 での事後確率が高い e が勝つ（同点で先頭を採る形に落ちない）。
  assertEquals(
    labFor(["a", "a", "i", "e", "e", "e"], { 2: { e: 1 } }),
    "0.0000000 0.0400000 a\n0.0400000 0.1200000 e\n",
  );
});

Deno.test("短区間マージ: pau は最小継続長未満でも残す", () => {
  // 1 フレーム（20ms）の pau。塗り替えると無音が消えて口が閉じない = 知覚上の劣化。
  assertEquals(
    labFor(["a", "a", "pau", "i", "i"]),
    "0.0000000 0.0400000 a\n0.0400000 0.0600000 pau\n0.0600000 0.1000000 i\n",
  );
});

Deno.test("cons 吸収: 後続が母音なら開始を前倒しして母音区間に畳む", () => {
  assertEquals(labFor(["cons", "cons", "a", "a"]), "0.0000000 0.0800000 a\n");
});

Deno.test("cons 吸収: 後続が母音/N でなければ先行区間を伸ばす", () => {
  assertEquals(
    labFor(["a", "a", "cons", "cons", "pau", "pau"]),
    "0.0000000 0.0800000 a\n0.0800000 0.1200000 pau\n",
  );
});

Deno.test("cons 吸収: 同じラベルに挟まれた cons は 1 区間へ連結する", () => {
  // 前倒しした開始で新しい区間を作ると a が 2 本に割れる（口形は変わっていないのに）。
  assertEquals(labFor(["a", "a", "cons", "cons", "a", "a"]), "0.0000000 0.1200000 a\n");
});

Deno.test("cons 吸収: 吸収先がどちらも無ければ pau に落とす", () => {
  assertEquals(labFor(["cons", "cons"]), "0.0000000 0.0400000 pau\n");
});

Deno.test("toLab: 秒は 7 桁固定で、行ごとに改行が付く", () => {
  assertEquals(
    toLab([{ start: 0, end: 0.02, label: "a" }, { start: 0.02, end: 1 / 3, label: "pau" }]),
    "0.0000000 0.0200000 a\n0.0200000 0.3333333 pau\n",
  );
  assertEquals(toLab([]), "");
});
