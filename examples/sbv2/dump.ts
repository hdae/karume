/**
 * torch 参照突合の **dump 経路**（開発用の契約）。合成しつつ、離散入力と乱数列を
 * `dump.safetensors` へ書き出す。
 *
 *     deno task dump:sbv2 --text "こんにちは、これはテストです。"
 *     cd tools/export-recipes && uv run --group sbv2 python -m sbv2.demo reference \
 *         --dump ../../outputs/demo/sbv2-dump/dump.safetensors
 *
 * ## なぜ `main.ts` と別なのか
 *
 * dump の 11 テンソルは `sbv2.demo reference` と対になる**開発用の契約**で、パイプライン
 * 利用者のストーリー（テキストを渡すと音が返る）ではない。したがって `generate` の返り値を
 * 診断で膨らませず、面を分ける。
 *
 * ## MUST: チェーンの実装は 1 本きり
 *
 * ここが呼ぶのは `Sbv2Pipeline` と**同じ** {@link synthesizeSbv2}（`generate` はこの返り値から
 * 音だけを採る薄い包み）。中間値を観測するためにホスト計算を写経すると、dump が「配布物が
 * 実際に走らせた値」ではなくなり、torch 突合が何も証明しなくなる。そのために models 側の
 * 内部面（`src/sbv2/pipeline.ts`）を直に掴んでいる — 公開面（`mod.ts` / `sbv2.ts`）には
 * 出ていない口である。
 *
 * ## NOTE: `reference` はスタイル資産を dump から読まない
 *
 * `sbv2.demo reference` は `style_vec` / `g` を `outputs/sbv2-demo/assets.safetensors`
 * から読む（`--assets`）。その資産は既定スタイル / 既定話者で焼かれているので、
 * `--style` / `--style-weight` を既定から動かした dump をそのまま突き合わせると、
 * **Karume 側と torch 側で別のスタイルベクトルを使った比較**になる。
 */

import { analyzeWithWords } from "@hdae/yomi";
import { getDictionary } from "@hdae/yomi/loader";
import { encodeWav, toSbv2Utterance } from "../../packages/models/mod.ts";
import {
  assetOpener,
  closeSbv2State,
  openSbv2State,
  type Sbv2GenerateOptions,
  synthesizeSbv2,
} from "../../packages/models/src/sbv2/pipeline.ts";
import {
  type DumpTensor,
  writeSafetensors,
} from "../../packages/models/tests/helpers/safetensors-write.ts";
import { isLocalDist, loadLocalAssets } from "../shared/local-assets.ts";

const USAGE = "--source <パス> --text <文字列> --model <名前> --quant <名前> --style <名前>" +
  " --style-weight <数> --sdp-ratio <数> --noise-scale <数> --noise-scale-w <数>" +
  " --length-scale <数> --seed <整数> --out <ディレクトリ>";
const KNOWN = new Set([
  "source",
  "model",
  "quant",
  "text",
  "style",
  "style-weight",
  "sdp-ratio",
  "noise-scale",
  "noise-scale-w",
  "length-scale",
  "seed",
  "out",
]);
const DUMP_FILE = "dump.safetensors";
const WAV_FILE = "out.wav";

const args = new Map<string, string>();
for (let at = 0; at < Deno.args.length; at += 2) {
  const [key, value] = [Deno.args[at], Deno.args[at + 1]];
  if (!key.startsWith("--") || value === undefined || value.startsWith("--")) {
    throw new Error(`引数 ${key} が --key value の対になっていない（使い方: ${USAGE}）`);
  }
  if (!KNOWN.has(key.slice(2))) throw new Error(`未知のオプション ${key}（使い方: ${USAGE}）`);
  args.set(key.slice(2), value);
}

const number = (key: string): number | undefined => {
  const raw = args.get(key);
  if (raw === undefined) return undefined;
  const value = Number(raw);
  if (!Number.isFinite(value)) throw new Error(`--${key} ${raw} が数値でない`);
  return value;
};

const source = args.get("source") ?? "models/karume-sbv2-fn";
const model = args.get("model");
const quant = args.get("quant");
/** hub / パイプラインへ渡す選択軸（未指定の欄は manifest の既定が埋める）。 */
const selection = {
  ...(model === undefined ? {} : { model }),
  ...(quant === undefined ? {} : { quant }),
};
const text = args.get("text");
if (text === undefined || text.length === 0) throw new Error(`--text が必要（使い方: ${USAGE}）`);
const outDir = args.get("out") ?? "outputs/demo/sbv2-dump";
const rawSeed = args.get("seed");
if (rawSeed !== undefined && !/^\d+$/.test(rawSeed)) {
  throw new Error(`--seed ${rawSeed} が非負整数でない`);
}
const style = args.get("style");
const styleWeight = number("style-weight");
const sdpRatio = number("sdp-ratio");
const noiseScale = number("noise-scale");
const noiseScaleW = number("noise-scale-w");
const lengthScale = number("length-scale");
const options: Sbv2GenerateOptions = {
  seed: rawSeed === undefined ? 0 : Number(rawSeed),
  ...(style === undefined ? {} : { style }),
  ...(styleWeight === undefined ? {} : { styleWeight }),
  ...(sdpRatio === undefined ? {} : { sdpRatio }),
  ...(noiseScale === undefined ? {} : { noiseScale }),
  ...(noiseScaleW === undefined ? {} : { noiseScaleW }),
  ...(lengthScale === undefined ? {} : { lengthScale }),
};

// dump は torch 側が同じ実重み（ckpt）を読み直す前提の突合なので、ローカルの配布形しか
// 相手にしない（HF から取った資産と手元の ckpt が同じ保証が無い）。
if (!await isLocalDist(source)) {
  throw new Error(`--source ${source} に karume.json が無い（dump はローカルの配布形専用）`);
}

const started = performance.now();
// テキスト → 発話は呼び手側（`main.ts` と同じ写経 — models は解析器を持たない）。
const utterance = toSbv2Utterance(analyzeWithWords(await getDictionary(), text));
const dumpAssets = await loadLocalAssets(source, selection);
const state = await openSbv2State(dumpAssets, assetOpener(dumpAssets.assets), selection);
try {
  const { sampleRate, audio, trace } = await synthesizeSbv2(state, utterance, options);
  const { input } = trace;
  const tokens = input.inputIds.length;
  const phonemes = input.ids.phoneIds.length;
  console.log(
    `[dump] ${JSON.stringify(text)} → 音素 ${input.phones.length}（add_blank 後 ${phonemes}）` +
      ` / DeBERTa トークン ${tokens} / ${trace.bertHiddenOutput}\n` +
      `       Ty=${trace.frames} フレーム / samples=${audio.length}`,
  );

  await Deno.mkdir(outDir, { recursive: true });
  await Deno.writeFile(`${outDir}/${WAV_FILE}`, encodeWav(audio, sampleRate));

  const tensors = new Map<string, DumpTensor>([
    ["input_ids", { dtype: "I32", shape: [1, tokens], data: Int32Array.from(input.inputIds) }],
    ["attention_mask", { dtype: "I32", shape: [1, tokens], data: new Int32Array(tokens).fill(1) }],
    ["word2ph", { dtype: "I32", shape: [tokens], data: Int32Array.from(input.word2ph) }],
    ["x", { dtype: "I32", shape: [1, phonemes], data: Int32Array.from(input.ids.phoneIds) }],
    ["tone", { dtype: "I32", shape: [1, phonemes], data: Int32Array.from(input.ids.toneIds) }],
    [
      "language",
      { dtype: "I32", shape: [1, phonemes], data: Int32Array.from(input.ids.languageIds) },
    ],
    ["x_mask", { dtype: "F32", shape: [1, 1, phonemes], data: trace.xMask }],
    ["z_noise", { dtype: "F32", shape: [1, 2, phonemes], data: trace.zNoise }],
    ["w_ceil", { dtype: "I32", shape: [phonemes], data: trace.wCeil }],
    [
      "zp_noise",
      { dtype: "F32", shape: [1, trace.channels, trace.frames], data: trace.zpNoise },
    ],
    ["audio", { dtype: "F32", shape: [audio.length], data: audio }],
  ]);
  const metadata = {
    demo: JSON.stringify({
      text,
      bertText: input.bertText,
      phones: input.phones,
      tones: input.tones,
      knobs: trace.knobs,
      seed: trace.seed,
      samplingRate: sampleRate,
      hopLength: state.rules.hopLength,
      bertHiddenFromEnd: state.rules.bertHiddenFromEnd,
      bertHiddenOutput: trace.bertHiddenOutput,
      // 出所の記録（torch 参照突合は資産の構成に依存する — 差の原因を後から辿るため）。
      source,
      model: model ?? "（manifest の既定）",
      quant: quant ?? "（manifest の既定）",
      speaker: trace.speaker,
      style: trace.style,
      styleWeight: trace.styleWeight,
    }),
  };
  await Deno.writeFile(`${outDir}/${DUMP_FILE}`, writeSafetensors(tensors, metadata));
  console.log(
    `[dump] ${outDir}/${WAV_FILE} / ${outDir}/${DUMP_FILE}（${
      ((performance.now() - started) / 1000).toFixed(1)
    }s）`,
  );
} finally {
  closeSbv2State(state);
}
