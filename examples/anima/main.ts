/**
 * Anima（テキスト → 画像）の 1 画面デモ。資産の出所だけが分岐で、あとは `generate` のノブ。
 *
 *     deno task demo:anima --prompt "1girl, solo, ..." --resolution 1344x768 --seed 42
 *     deno task demo:anima --source someone/anima --model anima-turbo --quant f16 --steps 8
 *     deno task demo:anima --source models/karume-anima --steps 32 --guidance 6
 *     deno task demo:anima --source models/karume-anima-turbo \
 *         --source-map hdae/karume-anima=models/karume-anima
 *
 * `--source` 未指定ならこの台本が {@link ANIMA_TURBO_CURRENT}（このパッケージ版が検証した
 * 取得元 — ADR 0073）を渡す。`fromPretrained` 自体に既定は無いので、取得元を綴るのは常に
 * 呼び出し側。明示したときだけ、`karume.json` を持つディレクトリなら `denoDirectory` で直に
 * 読み、それ以外は HF リポジトリ名として読む。どちらも `fromPretrained` の 1 本なので、shard
 * 分割された配布形もそのまま通る。未指定のノブは manifest の `defaults` が埋める。
 *
 * turbo 系のミラーは `text_encoder` を base リポへ**越境参照**するので、その取得元を
 * `--source-map owner/name=<パス>` で名指しする（繰り返し可。未指定で越境を踏むと hub が
 * 案内付きで落ちる — 隣接ディレクトリの推測はしない）。
 */

import { AnimaPipeline, encodePng } from "../../packages/models/mod.ts";
import { ANIMA_TURBO_CURRENT, parseResolution } from "../../packages/models/anima.ts";
import { distributionSource } from "../shared/local-source.ts";

const USAGE = "--source <パス|HF repo> --source-map <owner/name=パス> --prompt <文字列>" +
  " --resolution <WxH> --model <名前>" +
  " --quant <名前> --seed <整数> --steps <整数> --guidance <数> --negative <文字列>";
const KNOWN = new Set([
  "source",
  "source-map",
  "model",
  "quant",
  "prompt",
  "resolution",
  "seed",
  "steps",
  "guidance",
  "negative",
]);
const DEFAULT_PROMPT = "1girl, solo, long hair, blue eyes, school uniform, cherry blossoms," +
  " outdoors, smile, upper body, masterpiece, best quality";

/** `--key value` の対だけを受ける。MUST: 次のフラグを値として食わない（黙って既定へ落ちる）。 */
const args = new Map<string, string>();
/** `--source-map` だけは繰り返せる（越境先が複数ある配布形があるため）。 */
const sourceMaps: string[] = [];
for (let at = 0; at < Deno.args.length; at += 2) {
  const [key, value] = [Deno.args[at], Deno.args[at + 1]];
  if (!key.startsWith("--") || value === undefined || value.startsWith("--")) {
    throw new Error(`引数 ${key} が --key value の対になっていない（使い方: ${USAGE}）`);
  }
  // MUST: 未知のキーは落とす。打ち間違えたノブが黙って既定値で走ると、出力の違いが
  // 「モデルの揺れ」に見える。
  if (!KNOWN.has(key.slice(2))) throw new Error(`未知のオプション ${key}（使い方: ${USAGE}）`);
  if (key === "--source-map") sourceMaps.push(value);
  else args.set(key.slice(2), value);
}

const encoder = new TextEncoder();
const integer = (key: string): number | undefined => {
  const raw = args.get(key);
  if (raw !== undefined && !/^\d+$/.test(raw)) throw new Error(`--${key} ${raw} が非負整数でない`);
  return raw === undefined ? undefined : Number(raw);
};

const source = args.get("source");
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
/**
 * CFG のノブ。`guidanceScale === 1` では uncond 側を 1 度も計算しないので、そこへ
 * `negativePrompt` を渡すとパイプラインが fail loudly する — ここでは黙って落とさず、
 * 指定をそのまま通して向こうの門に判定させる（デモが理由を隠すと使い方を学べない）。
 */
const rawGuidance = args.get("guidance");
if (rawGuidance !== undefined && !Number.isFinite(Number(rawGuidance))) {
  throw new Error(`--guidance ${rawGuidance} が数値でない`);
}
const guidanceScale = rawGuidance === undefined ? undefined : Number(rawGuidance);
const negativePrompt = args.get("negative");

/** 取得元（ローカルの配布形なら `denoDirectory`・それ以外は HF リポジトリ名）。 */
const from = source === undefined
  ? ANIMA_TURBO_CURRENT
  : await distributionSource(source, sourceMaps);

console.log(
  `[anima] ${source ?? `${ANIMA_TURBO_CURRENT.repo}（台本の既定 = 検証済み pin）`}` +
    ` / model ${model ?? "（manifest の既定）"}` +
    ` / quant ${quant ?? "（manifest の既定）"} / seed ${seed}`,
);
const started = performance.now();
await using pipeline = await AnimaPipeline.fromPretrained(
  from,
  {
    ...selection,
    onProgress: ({ phase, loaded, total }) =>
      Deno.stderr.writeSync(encoder.encode(`\r  ${phase} ${(loaded / total * 100).toFixed(1)}%  `)),
  },
);
const image = await pipeline.generate({
  prompt,
  seed,
  ...(steps === undefined ? {} : { steps }),
  ...(resolution === undefined ? {} : { resolution }),
  ...(guidanceScale === undefined ? {} : { guidanceScale }),
  ...(negativePrompt === undefined ? {} : { negativePrompt }),
});
const png = await encodePng(image.data, image.width, image.height);
const name = `anima-${quant ?? "default"}-${image.width}x${image.height}` +
  `-${steps ?? "default"}step-seed${seed}.png`;
/** 既定の出力先に使うモデル名（取得元の末尾要素 — パスでも HF リポ名でも同じ規則）。 */
const sourceRef = source ?? ANIMA_TURBO_CURRENT.repo;
const sourceName = sourceRef.replace(/\/+$/, "").split("/").at(-1) ?? sourceRef;
const outDir = `outputs/examples/${sourceName}`;
await Deno.mkdir(outDir, { recursive: true });
await Deno.writeFile(`${outDir}/${name}`, png);
console.log(
  `[anima] ${outDir}/${name}（${((performance.now() - started) / 1000).toFixed(1)}s）`,
);
