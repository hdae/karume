/**
 * Wan2.1（テキスト → 動画）の最小のデモ。テキスト埋め込み資産の固定プロンプトを 1 本選んで動画を
 * 生成し、フレームを PNG の連番で書く。
 *
 *     deno task demo:wan --prompt boxing-cats --seed 42
 *     deno task demo:wan --prompt ferret --steps 20 --frames 17 --size 480x832
 *
 * 配布形（`models/karume-wan2.1`）はまだ無い（ADR 0118 段 7）ので、系列（`outputs/series/`）の krm と
 * 埋め込み資産を直接読む（`--series` / `--embeds`）。第 1 段の埋め込みは事前計算した固定プロンプトだけ
 * なので、`--prompt` は資産の名前（`boxing-cats` など）で選ぶ — 名前を間違えると選べる名前の一覧を
 * 出して落ちる。未指定のノブはパイプラインの既定（50 ステップ・guide 5.0・shift 3.0・832×480・
 * 33 フレーム）。
 */

import { encodePng } from "../../packages/models/mod.ts";
import { wanFrameToRgba, WanPipeline } from "../../packages/models/wan.ts";
import { runMain } from "../shared/run-main.ts";

const USAGE = "--prompt <名前> --negative <名前> --seed <整数> --steps <整数> --frames <整数>" +
  " --guidance <数> --shift <数> --size <WxH> --series <dir> --embeds <file> --out <dir>";
const KNOWN = new Set([
  "prompt",
  "negative",
  "seed",
  "steps",
  "frames",
  "guidance",
  "shift",
  "size",
  "series",
  "embeds",
  "out",
]);

/** `--key value` の対だけを受ける。MUST: 次のフラグを値として食わない（黙って既定へ落ちる）。 */
const args = new Map<string, string>();
for (let at = 0; at < Deno.args.length; at += 2) {
  const [key, value] = [Deno.args[at], Deno.args[at + 1]];
  if (!key.startsWith("--") || value === undefined || value.startsWith("--")) {
    throw new Error(`引数 ${key} が --key value の対になっていない（使い方: ${USAGE}）`);
  }
  // MUST: 未知のキーは落とす（打ち間違えたノブが黙って既定値で走らない）。
  if (!KNOWN.has(key.slice(2))) throw new Error(`未知のオプション ${key}（使い方: ${USAGE}）`);
  args.set(key.slice(2), value);
}

const integer = (key: string): number | undefined => {
  const raw = args.get(key);
  if (raw !== undefined && !/^\d+$/.test(raw)) throw new Error(`--${key} ${raw} が非負整数でない`);
  return raw === undefined ? undefined : Number(raw);
};
const number = (key: string): number | undefined => {
  const raw = args.get(key);
  if (raw !== undefined && !Number.isFinite(Number(raw))) {
    throw new Error(`--${key} ${raw} が数値でない`);
  }
  return raw === undefined ? undefined : Number(raw);
};

const promptName = args.get("prompt") ?? "boxing-cats";
const negativeName = args.get("negative");
const seed = integer("seed") ?? 42;
const steps = integer("steps");
const frames = integer("frames");
const guidance = number("guidance");
const shift = number("shift");
const size = args.get("size");
const sizeMatch = size === undefined ? undefined : /^(\d+)x(\d+)$/.exec(size);
if (sizeMatch === null) throw new Error(`--size ${size} が WxH の形でない`);
const series = args.get("series") ?? "outputs/series/wan2.1-t2v-1.3b-f16-dyn";
const embeds = args.get("embeds") ??
  "outputs/series/wan2.1-t2v-1.3b-text-embeds/text_embeds.safetensors";
const outRoot = args.get("out") ?? "outputs/examples/wan2.1-t2v-1.3b";

/** 部品（系列のグラフ名 = ディレクトリ名）。 */
const COMPONENTS = ["transformer", "vae_decoder_first", "vae_decoder_next"] as const;
const PART = /^model-(\d{5})-of-(\d{5})\.krm$/;

/**
 * 部品 1 本の krm を `<部品>[i]` のキーで読む（part 連番 `model-0000N-of-0000M.krm` か単一形
 * `model.krm`）。MUST: 番号の欠け・総数の食い違い・単一形との同居は落とす（どのバイト列を読むかが
 * 一意に決まらない）。
 */
const readComponent = async (
  assets: Record<string, Uint8Array<ArrayBuffer>>,
  component: string,
): Promise<void> => {
  const dir = `${series}/${component}`;
  const parts = new Map<number, string>();
  const totals = new Set<number>();
  let single = false;
  for await (const entry of Deno.readDir(dir)) {
    if (!entry.isFile) continue;
    if (entry.name === "model.krm") single = true;
    const match = PART.exec(entry.name);
    if (match === null) continue;
    parts.set(Number(match[1]), entry.name);
    totals.add(Number(match[2]));
  }
  if (single) {
    if (parts.size > 0) throw new Error(`${dir}: 単一形と part 連番が同居している`);
    assets[component] = await Deno.readFile(`${dir}/model.krm`);
    return;
  }
  const [total] = totals;
  if (totals.size !== 1 || parts.size !== total) {
    throw new Error(`${dir}: part 連番が揃っていない（${parts.size} 本・総数 [${[...totals]}]）`);
  }
  for (let index = 1; index <= total; index += 1) {
    const name = parts.get(index);
    if (name === undefined) throw new Error(`${dir}: part ${index} が無い`);
    assets[`${component}[${index - 1}]`] = await Deno.readFile(`${dir}/${name}`);
  }
};

/**
 * 台本の本体。MUST: `await using` はこの中に置く（`shared/run-main.ts` — 本体と解放が両方投げたときの
 * `SuppressedError` を展開するため）。
 */
const main = async (): Promise<void> => {
  const assets: Record<string, Uint8Array<ArrayBuffer>> = {};
  for (const component of COMPONENTS) await readComponent(assets, component);
  assets.text_embeds = await Deno.readFile(embeds);
  await using pipeline = await WanPipeline.fromAssets({ assets });

  const textOf = (name: string): string => {
    const entry = pipeline.prompts.find((candidate) => candidate.name === name);
    if (entry === undefined) {
      throw new Error(
        `プロンプト '${name}' は埋め込み資産に無い（選べる名前: ${
          pipeline.prompts.map((candidate) => `${candidate.name}（${candidate.role}）`).join(" / ")
        }）`,
      );
    }
    return entry.prompt;
  };
  const prompt = textOf(promptName);
  console.log(`[wan] ${promptName}: ${prompt.trim()}`);
  const started = performance.now();
  const encoder = new TextEncoder();
  const video = await pipeline.generate({
    prompt,
    seed,
    ...(negativeName === undefined ? {} : { negativePrompt: textOf(negativeName) }),
    ...(steps === undefined ? {} : { steps }),
    ...(frames === undefined ? {} : { frames }),
    ...(guidance === undefined ? {} : { guidance }),
    ...(shift === undefined ? {} : { shift }),
    ...(sizeMatch === undefined
      ? {}
      : { width: Number(sizeMatch[1]), height: Number(sizeMatch[2]) }),
    onEvent: (event) => {
      const elapsed = ((performance.now() - started) / 1000).toFixed(0);
      if (event.kind === "denoise-step") {
        Deno.stderr.writeSync(
          encoder.encode(`\r  denoise ${event.step}/${event.steps}（${elapsed}s）  `),
        );
      } else if (event.kind === "vae-tile") {
        Deno.stderr.writeSync(
          encoder.encode(`\r  vae tile ${event.tile}/${event.tiles}（${elapsed}s）  `),
        );
      }
    },
  });
  Deno.stderr.writeSync(encoder.encode("\n"));
  const outDir = `${outRoot}/wan-${promptName}-${video.width}x${video.height}-${video.frames}f` +
    `-${steps ?? "default"}step-seed${seed}`;
  await Deno.mkdir(outDir, { recursive: true });
  for (let frame = 0; frame < video.frames; frame += 1) {
    await Deno.writeFile(
      `${outDir}/frame-${String(frame).padStart(2, "0")}.png`,
      await encodePng(wanFrameToRgba(video, frame), video.width, video.height),
    );
  }
  console.log(
    `[wan] ${outDir}/frame-*.png（${video.frames} 枚・${
      ((performance.now() - started) / 1000).toFixed(1)
    }s）`,
  );
};

await runMain(main);
