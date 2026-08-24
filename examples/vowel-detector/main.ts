/**
 * 母音検出（音声 → リップシンク用の `.lab`）の 1 画面デモ。
 *
 *     deno task demo:vowel-detector --source models/karume-vowel-detector \
 *         --audio outputs/demo/vowel-vowels.wav
 *     deno task demo:vowel-detector --source models/karume-vowel-detector \
 *         --audio voice.wav --out outputs/demo/voice.lab
 *
 * MUST: `--source` は必須（既定を置かない）。このファミリは公開配布リポを持たず pin 定数も
 * 無い（ADR 0073 決定 1）ので、既定を書くと「存在しないリポの取得」に化ける。`karume.json` を
 * 持つディレクトリならローカル読み（`fromAssets`）、それ以外は HF リポジトリ名として
 * `fromPretrained`。
 *
 * MUST: 入力 WAV は **16kHz モノラル**（パイプラインはリサンプラを持たない — 周波数が違っても
 * 落ちずに別の母音列が出る）。この台本は `decodeWav` が読んだ周波数を宣言（`sampleRate`）と
 * 突き合わせて**先に落とす**。48kHz の素材から 16kHz を作る例は
 * `examples/irodori/eval-audio.ts`（3 分の 1 間引き）にある。
 */

import { decodeWav, VowelDetectorPipeline } from "../../packages/models/mod.ts";
import { isLocalDist, loadLocalAssets } from "../shared/local-assets.ts";

const USAGE = "--audio <WAV パス> --source <パス|HF repo> --model <名前> --quant <名前>" +
  " --out <パス>";
const KNOWN = new Set(["audio", "source", "model", "quant", "out"]);

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

const audioPath = args.get("audio");
if (audioPath === undefined) throw new Error(`--audio が要る（使い方: ${USAGE}）`);
const source = args.get("source");
if (source === undefined) throw new Error(`--source が要る（使い方: ${USAGE}）`);
const model = args.get("model");
const quant = args.get("quant");
/** hub / パイプラインへ渡す選択軸（未指定の欄は manifest の既定が埋める）。 */
const selection = {
  ...(model === undefined ? {} : { model }),
  ...(quant === undefined ? {} : { quant }),
};

const encoder = new TextEncoder();

const openPipeline = async (): Promise<VowelDetectorPipeline> => {
  if (await isLocalDist(source)) {
    return VowelDetectorPipeline.fromAssets(await loadLocalAssets(source, selection), selection);
  }
  return VowelDetectorPipeline.fromPretrained(source, {
    ...selection,
    onProgress: ({ phase, loaded, total }) =>
      Deno.stderr.writeSync(encoder.encode(`\r  ${phase} ${(loaded / total * 100).toFixed(1)}%  `)),
  });
};

const wav = decodeWav(await Deno.readFile(audioPath));
console.log(
  `[vowel-detector] ${source} / model ${model ?? "（manifest の既定）"}` +
    ` / quant ${quant ?? "（manifest の既定）"}\n` +
    `                 ${audioPath}（${(wav.data.length / wav.sampleRate).toFixed(2)}s /` +
    ` ${wav.sampleRate}Hz）`,
);
const started = performance.now();
await using pipeline = await openPipeline();
if (wav.sampleRate !== pipeline.sampleRate) {
  throw new Error(
    `${audioPath} は ${wav.sampleRate}Hz — この配布形は ${pipeline.sampleRate}Hz` +
      " モノラルを要求する（リサンプルは呼び出し側の責務）",
  );
}
const { segments, lab } = await pipeline.detect(wav.data);

const out = args.get("out") ?? `outputs/demo/${audioPath.split("/").at(-1)}.lab`;
// MUST: `cut > 0` で判定する。`-1`（`/` 無し = cwd 直下）を切ると 1 文字削ったディレクトリを
// 作り、`0`（絶対パスの根）を切ると空文字列で `mkdir` を呼ぶ。
const cut = out.lastIndexOf("/");
if (cut > 0) await Deno.mkdir(out.slice(0, cut), { recursive: true });
await Deno.writeTextFile(out, lab);
console.log(lab.trimEnd());
console.log(
  `[vowel-detector] ${out}（${segments.length} 区間 /` +
    ` ${((performance.now() - started) / 1000).toFixed(1)}s）`,
);
