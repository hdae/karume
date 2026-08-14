/**
 * 母音検出（音声 → リップシンク用の `.lab`）の 1 画面デモ。
 *
 *     deno task demo:vowel-detector --audio outputs/demo/vowel-vowels.wav
 *     deno task demo:vowel-detector --audio voice.wav --out outputs/demo/voice.lab
 *
 * `--source` が `karume.json` を持つディレクトリならローカル読み（`fromAssets`）、それ以外は
 * HF リポジトリ名として `fromPretrained`。
 *
 * MUST: 入力 WAV は **16kHz モノラル**（パイプラインはリサンプラを持たない — 周波数が違っても
 * 落ちずに別の母音列が出る）。この台本は `decodeWav` が読んだ周波数を宣言（`sampleRate`）と
 * 突き合わせて**先に落とす**。48kHz の素材から 16kHz を作る例は
 * `examples/irodori/eval-audio.ts`（3 分の 1 間引き）にある。
 */

import { parseManifest, resolveFiles } from "../../packages/hub/mod.ts";
import {
  decodeWav,
  type VowelDetectorAssets,
  VowelDetectorPipeline,
} from "../../packages/models/mod.ts";

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
const source = args.get("source") ?? "models/karume-vowel-detector";
const model = args.get("model");
const quant = args.get("quant");
/** hub / パイプラインへ渡す選択軸（未指定の欄は manifest の既定が埋める）。 */
const selection = {
  ...(model === undefined ? {} : { model }),
  ...(quant === undefined ? {} : { quant }),
};

const encoder = new TextEncoder();

/** ローカルディレクトリから manifest + 資産を読む（`fetchAssets` のローカル版）。 */
const loadLocal = async (dir: string): Promise<VowelDetectorAssets> => {
  const manifest = parseManifest(await Deno.readTextFile(`${dir}/karume.json`));
  const files = resolveFiles(manifest, selection);
  const byPath = new Map<string, Uint8Array<ArrayBuffer>>();
  let assets: Record<string, Uint8Array<ArrayBuffer>> = {};
  for (const key of Object.keys(files)) {
    const { path } = files[key];
    const bytes = byPath.get(path) ?? await Deno.readFile(`${dir}/${path}`);
    byPath.set(path, bytes);
    assets = { ...assets, [key]: bytes };
  }
  return { manifest, assets };
};

const openPipeline = async (): Promise<VowelDetectorPipeline> => {
  if (await Deno.stat(`${source}/karume.json`).then(() => true, () => false)) {
    return VowelDetectorPipeline.fromAssets(await loadLocal(source), selection);
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
const parent = out.slice(0, out.lastIndexOf("/"));
if (parent !== "") await Deno.mkdir(parent, { recursive: true });
await Deno.writeTextFile(out, lab);
console.log(lab.trimEnd());
console.log(
  `[vowel-detector] ${out}（${segments.length} 区間 /` +
    ` ${((performance.now() - started) / 1000).toFixed(1)}s）`,
);
