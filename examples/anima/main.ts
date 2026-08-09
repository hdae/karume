/**
 * Anima（テキスト → 画像）の 1 画面デモ。資産の出所だけが分岐で、あとは `generate` のノブ。
 *
 *     deno task demo:anima --prompt "1girl, solo, ..." --resolution 1344x768 --seed 42
 *     deno task demo:anima --source someone/anima --model anima-turbo --quant f16 --steps 8
 *
 * `--source` が `karume.json` を持つディレクトリならローカル読み（`fromAssets`）、それ以外は
 * HF リポジトリ名として `fromPretrained`。未指定のノブは manifest の `defaults` が埋める。
 */

import { parseManifest, resolveFiles } from "../../packages/hub/mod.ts";
import { type AnimaAssets, AnimaPipeline, encodePng } from "../../packages/models/mod.ts";
import { parseResolution } from "../../packages/models/anima.ts";

const USAGE = "--source <パス|HF repo> --prompt <文字列> --resolution <WxH> --model <名前>" +
  " --quant <名前> --seed <整数> --steps <整数>";
const DEFAULT_PROMPT = "1girl, solo, long hair, blue eyes, school uniform, cherry blossoms," +
  " outdoors, smile, upper body, masterpiece, best quality";

/** `--key value` の対だけを受ける。MUST: 次のフラグを値として食わない（黙って既定へ落ちる）。 */
const args = new Map<string, string>();
for (let at = 0; at < Deno.args.length; at += 2) {
  const [key, value] = [Deno.args[at], Deno.args[at + 1]];
  if (!key.startsWith("--") || value === undefined || value.startsWith("--")) {
    throw new Error(`引数 ${key} が --key value の対になっていない（使い方: ${USAGE}）`);
  }
  args.set(key.slice(2), value);
}

const encoder = new TextEncoder();
const integer = (key: string): number | undefined => {
  const raw = args.get(key);
  if (raw !== undefined && !/^\d+$/.test(raw)) throw new Error(`--${key} ${raw} が非負整数でない`);
  return raw === undefined ? undefined : Number(raw);
};

const source = args.get("source") ?? "models/anima-turbo";
const model = args.get("model");
const quant = args.get("quant");
/** hub / パイプラインへ渡す選択軸（未指定の欄は manifest の既定が埋める）。 */
const selection = {
  ...(model === undefined ? {} : { model }),
  ...(quant === undefined ? {} : { quant }),
};
const prompt = args.get("prompt") ?? DEFAULT_PROMPT;
const seed = integer("seed") ?? 42;
const steps = integer("steps");
const spelled = args.get("resolution");
const resolution = spelled === undefined ? undefined : parseResolution(spelled);

/** ローカルディレクトリから manifest + 資産を読む（`fetchAssets` のローカル版）。 */
const loadLocal = async (dir: string): Promise<AnimaAssets> => {
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

const openPipeline = async (): Promise<AnimaPipeline> => {
  if (await Deno.stat(`${source}/karume.json`).then(() => true, () => false)) {
    return AnimaPipeline.fromAssets(await loadLocal(source), selection);
  }
  return AnimaPipeline.fromPretrained(source, {
    ...selection,
    onProgress: ({ phase, loaded, total }) =>
      Deno.stderr.writeSync(encoder.encode(`\r  ${phase} ${(loaded / total * 100).toFixed(1)}%  `)),
  });
};

console.log(
  `[anima] ${source} / model ${model ?? "（manifest の既定）"}` +
    ` / quant ${quant ?? "（manifest の既定）"} / seed ${seed}`,
);
const started = performance.now();
using pipeline = await openPipeline();
const image = await pipeline.generate({
  prompt,
  seed,
  ...(steps === undefined ? {} : { steps }),
  ...(resolution === undefined ? {} : { resolution }),
});
const png = await encodePng(image.data, image.width, image.height);
const name = `anima-${quant ?? "default"}-${image.width}x${image.height}` +
  `-${steps ?? "default"}step-seed${seed}.png`;
await Deno.mkdir("outputs", { recursive: true });
await Deno.writeFile(`outputs/${name}`, png);
console.log(`[anima] outputs/${name}（${((performance.now() - started) / 1000).toFixed(1)}s）`);
