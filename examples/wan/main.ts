/**
 * Wan2.1（テキスト → 動画）の最小のデモ。テキスト埋め込み資産の固定プロンプトを 1 本選んで動画を
 * 生成し、フレームを PNG の連番で書く。
 *
 *     deno task demo:wan --prompt boxing-cats --seed 42
 *     deno task demo:wan --prompt ferret --steps 20 --frames 17 --size 480x832
 *
 * 配布形は `fromPretrained` で読む（ADR 0118 段 7）。`--source` 未指定なら手元の配布形ミラー
 * `models/karume-wan2.1`（`dist.py --pipeline wan` が組む）を取得元ハンドル（`denoDirectory`）で読む —
 * HF の公開リポはまだ無いので、ミラーが無ければ組み立てのコマンドを出して落ちる。明示した `--source` は
 * ローカルの配布形か HF のリポ名（`owner/name`）としてそのまま読む。第 1 段の埋め込みは事前計算した
 * 固定プロンプトだけなので、`--prompt` は資産の名前（`boxing-cats` など）で選ぶ — 名前を間違えると
 * 選べる名前の一覧を出して落ちる。未指定のノブはパイプラインの既定（step 数・guidance・shift は
 * manifest の `pipelineConfig` — 50・5.0・3.0、寸法とフレーム数は 832×480・33 フレーム）。
 */

import { encodePng } from "../../packages/models/mod.ts";
import { wanFrameToRgba, WanPipeline } from "../../packages/models/wan.ts";
import { runMain } from "../shared/run-main.ts";
import { distributionSource } from "../shared/local-source.ts";
import { isLocalDist } from "../shared/local-assets.ts";

const USAGE = "--source <パス|HF repo> --prompt <名前> --negative <名前> --seed <整数>" +
  " --steps <整数> --frames <整数> --guidance <数> --shift <数> --size <WxH> --out <dir>";
const KNOWN = new Set([
  "source",
  "prompt",
  "negative",
  "seed",
  "steps",
  "frames",
  "guidance",
  "shift",
  "size",
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
const outRoot = args.get("out") ?? "outputs/examples/wan2.1-t2v-1.3b";

/** 既定の取得元（手元の配布形ミラー — HF の公開リポはまだ無い）。 */
const DEFAULT_SOURCE = "models/karume-wan2.1";
const ASSEMBLE_COMMAND = "cd tools/export-recipes && uv run python dist.py --pipeline wan";

/**
 * `--source` を `fromPretrained` の取得元へ写す。未指定は既定のミラーで、無ければ落とす（公開リポが
 * 無いので、他所へ黙って取りに行く先が無い）。
 */
const resolveSource = async () => {
  const source = args.get("source");
  if (source !== undefined) return { from: await distributionSource(source), label: source };
  if (!await isLocalDist(DEFAULT_SOURCE)) {
    throw new Error(
      `配布形ミラー ${DEFAULT_SOURCE} に karume.json が無い — ${ASSEMBLE_COMMAND} で組むか、` +
        `--source で配布形を指す（使い方: ${USAGE}）`,
    );
  }
  return { from: await distributionSource(DEFAULT_SOURCE), label: DEFAULT_SOURCE };
};

/**
 * 台本の本体。MUST: `await using` はこの中に置く（`shared/run-main.ts` — 本体と解放が両方投げたときの
 * `SuppressedError` を展開するため）。
 */
const main = async (): Promise<void> => {
  const { from, label } = await resolveSource();
  console.log(`[wan] source: ${label}`);
  await using pipeline = await WanPipeline.fromPretrained(from);

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
