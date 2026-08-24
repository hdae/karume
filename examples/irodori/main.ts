/**
 * Irodori（テキスト → 音声）の 1 画面デモ。資産の出所だけが分岐で、あとは `generate` のノブ。
 *
 *     deno task demo:irodori --text "こんにちは、これはテストです。"
 *     deno task demo:irodori --caption "落ち着いた女性の声で、ゆっくりと話している。" --seed 7
 *     deno task demo:irodori --ref inputs/irodori/v4-small/samples/clone_ref1.wav
 *
 * `--source` 未指定ならこの台本が {@link IRODORI_V4_SMALL_CURRENT}（このパッケージ版が検証した
 * 取得元 — ADR 0073）を渡す。`fromPretrained` 自体に既定は無いので、取得元を綴るのは常に
 * 呼び出し側。明示したときだけ、`karume.json` を持つディレクトリならローカル読み（`fromAssets`）、
 * それ以外は HF リポジトリ名として `fromPretrained`。`--ref` は参照音声（WAV — 配布形と同じ
 * 48kHz の mono/多ch PCM16 か IEEE float）で、渡すとその声質に寄る（voice cloning）。`--caption` は
 * 声質の指示文（Voice Design）。サンプラのノブ（steps / CFG）は manifest の `pipelineConfig`
 * が固定していて、実行時には `--seed` と `--seconds`（発話長の直接指定）だけが動かせる。
 */

import {
  decodeWav,
  encodeWav,
  IrodoriPipeline,
  type IrodoriSpeakerInput,
} from "../../packages/models/mod.ts";
import { IRODORI_V4_SMALL_CURRENT } from "../../packages/models/irodori.ts";
import { isLocalDist, loadLocalAssets } from "../shared/local-assets.ts";

const USAGE = "--source <パス|HF repo> --text <文字列> --caption <文字列> --ref <WAV パス>" +
  " --model <名前> --quant <名前> --seconds <数> --seed <整数> --out <パス>";
const KNOWN = new Set([
  "source",
  "model",
  "quant",
  "text",
  "caption",
  "ref",
  "seconds",
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
const integer = (key: string): number | undefined => {
  const raw = args.get(key);
  if (raw !== undefined && !/^\d+$/.test(raw)) throw new Error(`--${key} ${raw} が非負整数でない`);
  return raw === undefined ? undefined : Number(raw);
};
const number = (key: string): number | undefined => {
  const raw = args.get(key);
  if (raw === undefined) return undefined;
  const value = Number(raw);
  if (!Number.isFinite(value)) throw new Error(`--${key} ${raw} が数値でない`);
  return value;
};

const source = args.get("source");
const model = args.get("model");
const quant = args.get("quant");
/** hub / パイプラインへ渡す選択軸（未指定の欄は manifest の既定が埋める）。 */
const selection = {
  ...(model === undefined ? {} : { model }),
  ...(quant === undefined ? {} : { quant }),
};
const text = args.get("text") ?? DEFAULT_TEXT;
const caption = args.get("caption");
const ref = args.get("ref");
const seconds = number("seconds");
const seed = integer("seed") ?? 0;

/** 参照音声（任意）。周波数が配布形と違えば `generate` が fail loudly する（リサンプル無し）。 */
const speaker: IrodoriSpeakerInput | undefined = ref === undefined
  ? undefined
  : { audio: decodeWav(await Deno.readFile(ref)) };

const openPipeline = async (): Promise<IrodoriPipeline> => {
  if (source !== undefined && await isLocalDist(source)) {
    return IrodoriPipeline.fromAssets(await loadLocalAssets(source, selection), selection);
  }
  return IrodoriPipeline.fromPretrained(source ?? IRODORI_V4_SMALL_CURRENT, {
    ...selection,
    onProgress: ({ phase, loaded, total }) =>
      Deno.stderr.writeSync(encoder.encode(`\r  ${phase} ${(loaded / total * 100).toFixed(1)}%  `)),
  });
};

console.log(
  `[irodori] ${source ?? `${IRODORI_V4_SMALL_CURRENT.repo}（台本の既定 = 検証済み pin）`}` +
    ` / model ${model ?? "（manifest の既定）"}` +
    ` / quant ${quant ?? "（manifest の既定）"} / seed ${seed}` +
    `${ref === undefined ? "" : ` / 参照 ${ref}`}\n` +
    `          ${JSON.stringify(text)}` +
    `${caption === undefined ? "" : `\n          caption ${JSON.stringify(caption)}`}`,
);
const started = performance.now();
await using pipeline = await openPipeline();
const audio = await pipeline.generate({
  text,
  seed,
  ...(caption === undefined ? {} : { caption }),
  ...(speaker === undefined ? {} : { speaker }),
  ...(seconds === undefined ? {} : { durationSeconds: seconds }),
});
const name = `irodori-${quant ?? "default"}-${ref === undefined ? "no-ref" : "cloned"}` +
  `-seed${seed}.wav`;
const out = args.get("out") ?? `outputs/demo/${name}`;
// MUST: `cut > 0` で判定する。`-1`（`/` 無し = cwd 直下）を切ると 1 文字削ったディレクトリを
// 作り、`0`（絶対パスの根）を切ると空文字列で `mkdir` を呼ぶ。
const cut = out.lastIndexOf("/");
if (cut > 0) await Deno.mkdir(out.slice(0, cut), { recursive: true });
await Deno.writeFile(out, encodeWav(audio.data, audio.sampleRate));
console.log(
  `[irodori] ${out}（${(audio.data.length / audio.sampleRate).toFixed(2)}s / S ${audio.frames}` +
    ` / dit ${audio.forwards} forward / ${((performance.now() - started) / 1000).toFixed(1)}s）`,
);
