/**
 * SBV2（テキスト → 音声）の 1 画面デモ。資産の出所だけが分岐で、あとは `generate` のノブ。
 *
 *     deno task demo:sbv2 --source models/karume-sbv2-fn --text "こんにちは、これはテストです。"
 *     deno task demo:sbv2 --source someone/sbv2 --model FN4 --quant f16 --style high --seed 7
 *
 * MUST: `--source` は必須（既定を置かない）。この台本が想定する FN 系列の配布リポは公開保留で
 * pin 定数を持たない（ADR 0073 決定 1）ので、既定を書くと「存在しないリポの取得」に化ける。
 * `karume.json` を持つディレクトリならローカル読み（`fromAssets`）、それ以外は HF リポジトリ名
 * として `fromPretrained`。未指定のノブは manifest の `pipelineConfig.defaults` が埋める。
 *
 * NOTE: torch 参照突合の dump（11 テンソル契約）は `dump.ts` にある — パイプライン利用者の
 * ストーリーではなく開発用の契約なので、面を分けてある。
 */

import { encodeWav, Sbv2Pipeline } from "../../packages/models/mod.ts";
import { isLocalDist, loadLocalAssets } from "../shared/local-assets.ts";

const USAGE = "--source <パス|HF repo> --text <文字列> --model <名前> --quant <名前>" +
  " --style <名前> --style-weight <数> --sdp-ratio <数> --noise-scale <数>" +
  " --noise-scale-w <数> --length-scale <数> --seed <整数> --out <パス>";
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
const DEFAULT_TEXT = "こんにちは、これはテストです。";

/** `--key value` の対だけを受ける。MUST: 次のフラグを値として食わない（黙って既定へ落ちる）。 */
const args = new Map<string, string>();
for (let at = 0; at < Deno.args.length; at += 2) {
  const [key, value] = [Deno.args[at], Deno.args[at + 1]];
  if (!key.startsWith("--") || value === undefined || value.startsWith("--")) {
    throw new Error(`引数 ${key} が --key value の対になっていない（使い方: ${USAGE}）`);
  }
  // MUST: 未知のキーは落とす。打ち間違えたノブが黙って既定値で走ると、出力の違いが
  // 「モデルの揺れ」に見える。
  if (!KNOWN.has(key.slice(2))) throw new Error(`未知のオプション ${key}（使い方: ${USAGE}）`);
  args.set(key.slice(2), value);
}

const encoder = new TextEncoder();
const number = (key: string): number | undefined => {
  const raw = args.get(key);
  if (raw === undefined) return undefined;
  const value = Number(raw);
  if (!Number.isFinite(value)) throw new Error(`--${key} ${raw} が数値でない`);
  return value;
};
const integer = (key: string): number | undefined => {
  const raw = args.get(key);
  if (raw !== undefined && !/^\d+$/.test(raw)) throw new Error(`--${key} ${raw} が非負整数でない`);
  return raw === undefined ? undefined : Number(raw);
};

const source = args.get("source");
if (source === undefined) throw new Error(`--source が要る（使い方: ${USAGE}）`);
const model = args.get("model");
const quant = args.get("quant");
/** hub / パイプラインへ渡す選択軸（未指定の欄は manifest の既定が埋める）。 */
const selection = {
  ...(model === undefined ? {} : { model }),
  ...(quant === undefined ? {} : { quant }),
};
const text = args.get("text") ?? DEFAULT_TEXT;
const style = args.get("style");
const seed = integer("seed") ?? 0;
const styleWeight = number("style-weight");
const sdpRatio = number("sdp-ratio");
const noiseScale = number("noise-scale");
const noiseScaleW = number("noise-scale-w");
const lengthScale = number("length-scale");

const openPipeline = async (): Promise<Sbv2Pipeline> => {
  if (await isLocalDist(source)) {
    return Sbv2Pipeline.fromAssets(await loadLocalAssets(source, selection), selection);
  }
  return Sbv2Pipeline.fromPretrained(source, {
    ...selection,
    onProgress: ({ phase, loaded, total }) =>
      Deno.stderr.writeSync(encoder.encode(`\r  ${phase} ${(loaded / total * 100).toFixed(1)}%  `)),
  });
};

console.log(
  `[sbv2] ${source} / model ${model ?? "（manifest の既定）"}` +
    ` / quant ${quant ?? "（manifest の既定）"} / seed ${seed}\n` +
    `       ${JSON.stringify(text)}`,
);
const started = performance.now();
await using pipeline = await openPipeline();
const audio = await pipeline.generate({
  text,
  seed,
  ...(style === undefined ? {} : { style }),
  ...(styleWeight === undefined ? {} : { styleWeight }),
  ...(sdpRatio === undefined ? {} : { sdpRatio }),
  ...(noiseScale === undefined ? {} : { noiseScale }),
  ...(noiseScaleW === undefined ? {} : { noiseScaleW }),
  ...(lengthScale === undefined ? {} : { lengthScale }),
});
const name = `sbv2-${quant ?? "default"}-${style ?? "default"}-seed${seed}.wav`;
const out = args.get("out") ?? `outputs/demo/${name}`;
// MUST: `cut > 0` で判定する。`-1`（`/` 無し = cwd 直下）を切ると 1 文字削ったディレクトリを
// 作り、`0`（絶対パスの根）を切ると空文字列で `mkdir` を呼ぶ。
const cut = out.lastIndexOf("/");
if (cut > 0) await Deno.mkdir(out.slice(0, cut), { recursive: true });
await Deno.writeFile(out, encodeWav(audio.data, audio.sampleRate));
console.log(
  `[sbv2] ${out}（${(audio.data.length / audio.sampleRate).toFixed(2)}s / ${
    ((performance.now() - started) / 1000).toFixed(1)
  }s）`,
);
