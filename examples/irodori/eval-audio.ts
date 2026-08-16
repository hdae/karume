/**
 * 評価用の日本語発話 4 本を **再現可能な形で**焼き直す台本（母音検出の実 GPU e2e が読む）。
 *
 *     deno task demo:eval-audio --source models/karume-irodori-v4-small
 *
 * 中身は `examples/irodori/main.ts` をテキストと seed を変えて 4 回呼び、**48kHz の生成物を
 * 16kHz へ落とす**だけ。**テキスト / seed / ケース名の正本はこのファイル**で、生成物は
 * `outputs/demo/` 直下（`rm -rf` で安全に消せる席 — docs/assets-layout.md）。置き場が
 * `examples/irodori/` なのは `examples/anima/eval-images.ts` と同じ理由で、**焼く側の家族**に
 * 属するため（読む側は母音検出）。
 *
 * MUST: `--source` は必須にする（既定を置かない）。Irodori の配布形は untracked のローカル資産で
 * 置き場が環境ごとに違い、`main.ts` は `karume.json` を持たないパスを **HF リポジトリ名**と
 * 読んで取得しに行くので、既定を書くと打ち間違いが「知らないリポジトリの取得」に化ける。
 *
 * ## なぜ 16kHz へ落とす段がここに要るのか
 *
 * 母音検出の特徴抽出（`packages/models/src/vowel-detector/features.ts`）は **16kHz 固定**で、
 * リサンプルを持たない（学習時の契約 — n_fft 512 / hop 160 が 16kHz 前提）。一方 Irodori の
 * 配布形は 48kHz なので、素材を作るこちら側で落とす。**48000 / 16000 = 3 の整数倍**なので、
 * 折り返し防止の低域通過を掛けて 3 サンプルに 1 本採るだけで済む（有理数比の多相フィルタは
 * 要らない — SBV2（44.1kHz）を素材に選ばなかったのはこの一点）。
 *
 * この段の数値そのものは仕様ではない（学習時の前処理でも配布形でもない）。**素材の同一性は
 * ファイルの sha256 が持つ**ので、フィルタの綴りを変えたら e2e の期待 `.lab` は採り直しになる。
 */

import { decodeWav, encodeWav } from "../../packages/models/mod.ts";

/** 母音検出が受ける唯一の周波数（`features.ts` の `SAMPLE_RATE`）。 */
const TARGET_RATE = 16000;
/** Irodori 配布形の周波数（`karume.json` の `pipelineConfig.sampleRate`）。実測して突き合わせる。 */
const SOURCE_RATE = 48000;

/**
 * 焼く 4 本。長さを **2 秒級 / 3 秒級 / 8 秒級 / 12 秒級**に散らしてあるのが要で、母音検出の
 * 長さバケット（グラフは長さごとに別 — GRU が時間方向へ展開される）の実測はこの散らばりを使う。
 *
 * `vowels` だけは内容も検査の材料になる — 5 母音を区切って読ませているので、出てくる `.lab` が
 * `a i u e o` の順に並ぶかどうかが**目視で**判る（数値の門とは別に、鎖が意味のある出力を
 * 出しているかを人が見るための 1 本）。
 */
const CASES: readonly { name: string; seed: number; text: string; why: string }[] = [
  {
    name: "short",
    seed: 11,
    text: "ありがとうございます。",
    why: "2 秒級（1 文・短い）",
  },
  {
    name: "vowels",
    seed: 12,
    text: "あ、い、う、え、お。",
    why: "3 秒級（5 母音を区切って読む — `.lab` の並びが目視で判る）",
  },
  {
    name: "mid",
    seed: 13,
    text: "こんにちは。今日はとてもいい天気ですね。" +
      "散歩に出かけるには、ちょうどいい季節だと思います。",
    why: "8 秒級（3 文）",
  },
  {
    name: "long",
    seed: 14,
    text: "音声から母音の並びを取り出して、口の形のタイムラインを作ります。" +
      "短い文でも長い文でも、同じ手順で処理できることを確かめています。",
    why: "12 秒級（2 文・長い）",
  },
];

/** `main.ts` へ渡す 48kHz の置き場（素材の出所が判るように残す）。 */
const sourcePath = (name: string): string => `outputs/demo/vowel-${name}-48k.wav`;
/** 母音検出が読む 16kHz の置き場。 */
const targetPath = (name: string): string => `outputs/demo/vowel-${name}.wav`;

const USAGE = "--source <Irodori 配布形のパス|HF repo>";

/** `--key value` の対だけを受ける（`main.ts` と同じ規律 — 未知キーは落とす）。 */
const args = new Map<string, string>();
for (let at = 0; at < Deno.args.length; at += 2) {
  const [key, value] = [Deno.args[at], Deno.args[at + 1]];
  if (!key.startsWith("--") || value === undefined || value.startsWith("--")) {
    throw new Error(`引数 ${key} が --key value の対になっていない（使い方: ${USAGE}）`);
  }
  if (key !== "--source") throw new Error(`未知のオプション ${key}（使い方: ${USAGE}）`);
  args.set(key.slice(2), value);
}

const source = args.get("source");
if (source === undefined) throw new Error(`--source が無い（使い方: ${USAGE}）`);

/**
 * 折り返し防止の低域通過 FIR（Blackman 窓の sinc）。
 *
 * 遮断は目標ナイキストの 95%（7.6kHz）に置く — 母音の識別に効く帯域（F1/F2 は高くても
 * 3kHz 級）より充分上で、遷移帯を 8kHz までに収める。タップ数は奇数（群遅延が整数サンプルに
 * なる＝畳み込みの中心を素直に取れる）。
 */
const lowpassTaps = (cutoffHz: number, rate: number, taps: number): Float64Array => {
  if (taps % 2 === 0) throw new Error(`タップ数 ${taps} が奇数でない`);
  const half = (taps - 1) / 2;
  const omega = (2 * Math.PI * cutoffHz) / rate;
  const filter = new Float64Array(taps);
  let sum = 0;
  for (let index = 0; index < taps; index += 1) {
    const offset = index - half;
    // sinc（offset = 0 は極限値 omega）。
    const sinc = offset === 0 ? omega : Math.sin(omega * offset) / offset;
    // Blackman 窓（阻止域 −58dB 級 — 折り返しを可聴域外まで落とす）。
    const phase = (2 * Math.PI * index) / (taps - 1);
    const window = 0.42 - 0.5 * Math.cos(phase) + 0.08 * Math.cos(2 * phase);
    const value = sinc * window;
    filter[index] = value;
    sum += value;
  }
  // 直流利得を 1 に正規化する（掛けたぶんだけ音量が変わるのを避ける）。
  for (let index = 0; index < taps; index += 1) filter[index] /= sum;
  return filter;
};

/**
 * 整数倍のダウンサンプル（低域通過 → 間引き）。窓の外は 0 として畳む（端 6ms の減衰は
 * 発話の外側なので実害が無い）。積算は f64、出力だけ f32 へ落とす。
 */
const decimate = (samples: Float32Array, factor: number, filter: Float64Array): Float32Array => {
  const half = (filter.length - 1) / 2;
  const length = Math.floor(samples.length / factor);
  const output = new Float32Array(length);
  for (let index = 0; index < length; index += 1) {
    const center = index * factor;
    let accumulator = 0;
    const from = Math.max(0, center - half);
    const to = Math.min(samples.length - 1, center + half);
    for (let at = from; at <= to; at += 1) {
      accumulator += samples[at] * filter[at - center + half];
    }
    output[index] = accumulator;
  }
  return output;
};

const sha256Hex = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
  Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");

const main = new URL("./main.ts", import.meta.url);
const factor = SOURCE_RATE / TARGET_RATE;
const filter = lowpassTaps(TARGET_RATE * 0.475, SOURCE_RATE, 193);

for (const entry of CASES) {
  console.log(`[eval-audio] ${entry.name} — ${entry.why}`);
  const { code } = await new Deno.Command(Deno.execPath(), {
    args: [
      "run",
      "-A",
      main.href,
      "--source",
      source,
      "--text",
      entry.text,
      "--seed",
      String(entry.seed),
      "--out",
      sourcePath(entry.name),
    ],
    stdout: "inherit",
    stderr: "inherit",
  }).output();
  if (code !== 0) throw new Error(`${entry.name}（seed ${entry.seed}）の生成が終了コード ${code}`);

  const decoded = decodeWav(await Deno.readFile(sourcePath(entry.name)));
  // MUST: 周波数を実測して突き合わせる。配布形が別の周波数になったら、黙って 1/3 に
  // 間引かれた別の速さの音声が素材になる。
  if (decoded.sampleRate !== SOURCE_RATE) {
    throw new Error(
      `${sourcePath(entry.name)} が ${decoded.sampleRate}Hz（${SOURCE_RATE}Hz を期待）` +
        " — 配布形の sampleRate が変わったなら SOURCE_RATE の前提ごと見直す",
    );
  }
  const resampled = decimate(decoded.data, factor, filter);
  const bytes = encodeWav(resampled, TARGET_RATE);
  await Deno.writeFile(targetPath(entry.name), bytes);
  console.log(
    `[eval-audio] ${targetPath(entry.name)}` +
      `（${(resampled.length / TARGET_RATE).toFixed(2)}s / ${resampled.length} サンプル` +
      ` / sha256 ${await sha256Hex(bytes)}）`,
  );
}

console.log(
  `[eval-audio] ${CASES.length} 本。長さが変わったら母音検出 e2e の期待 \`.lab\` と sha256 を` +
    "採り直す（packages/models/tests/e2e_vowel_detector_chain_test.ts）",
);
