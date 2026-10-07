/**
 * Wan2.1 T2V 1.3B と Wan2.2 TI2V 5B（テキスト → 動画）の最小のデモ。プロンプトを 1 本渡して動画を生成し、
 * フレームを PNG の連番で書く。
 *
 *     deno task demo:wan --prompt boxing-cats --seed 42
 *     deno task demo:wan --prompt "A red fox trots through fresh snow at sunrise." --steps 20
 *     deno task demo:wan --text-encoder precomputed --prompt ferret --frames 17 --size 480x832
 *     deno task demo:wan --quant f16 --prompt boxing-cats --seed 42
 *     deno task demo:wan --generation wan2.2 --prompt boxing-cats --seed 42
 *
 * 世代は `--generation`（`wan2.1` / `wan2.2` — 既定 `wan2.1`）。世代で違うのはパイプラインの class
 * （`WanPipeline` / `WanTi2vPipeline`）・既定の配布形ミラー・その組み立てのコマンド・既定の出力先だけで、
 * 下の取得元・テキストエンコーダ・席・ノブの扱いは両世代で共通（2.2 の manifest の `text_encoder` も同じ
 * umT5 の配布リポを越境参照する）。Wan2.2 は画像入力（I2V）をまだ受けない（ADR 0121 段 9）。
 *
 * 配布形は `fromPretrained` で読む（ADR 0118 段 7）。`--source` 未指定なら手元の配布形ミラー
 * `models/karume-wan2.1`（`dist.py --pipeline wan` が組む）/ `models/karume-wan2.2`
 * （`dist.py --pipeline wan-ti2v` が組む）を取得元ハンドル（`denoDirectory`）で読む —
 * HF の公開リポはまだ無いので、ミラーが無ければ組み立てのコマンドを出して落ちる。明示した `--source` は
 * ローカルの配布形か HF のリポ名（`owner/name`）としてそのまま読む。
 *
 * テキストエンコーダの経路は `--text-encoder`（既定 `gpu` — パイプラインの既定と同じ・ADR 0119 決定 7）:
 * `gpu` は umT5（i8）を GPU で回して任意の文字列を受け、`precomputed` は umT5 を読まずに埋め込み資産の
 * 固定プロンプトだけを受ける。`gpu` でローカルの配布形を読むとき、umT5 は別の配布リポ（Wan の manifest の
 * `text_encoder` が越境参照する）なので、その手元のミラーを `--umt5-source`（既定 `models/karume-umt5-xxl`）で
 * 指し、hub の取得元の `crossRepo` の mapping で渡す（隣のディレクトリを推測しない — hub の `local.ts`）。
 *
 * `--swap-text-encoder` は umT5 を互換の別の配布形へ差し替える（`fromPretrained` の
 * `components: { text_encoder: { source } }` — ADR 0122 決定 7）。差し替え先は手元の配布形
 * （`karume.json` を持つディレクトリ — 例: 取り込みから組んだ実験用ミラー）か、`owner/name@<40 桁の commit>`
 * の HF のリポ（`{ repo, revision }` へ写す）。素のリポ名は受けない — 文字列の取得元は main 追従になり、
 * 元の配布形の pin が消える。差し替えた席は差し替え先から読むので、元の umT5 の手元のミラー
 * （`--umt5-source`）は要らない（渡すと効かないので落とす）。`--text-encoder precomputed` との組み合わせも
 * 落とす（umT5 を読まない経路では差し替えが黙って効かない）。model / quant は差し替え先の manifest の既定。
 *
 *     deno task demo:wan --swap-text-encoder outputs/misc/local-dist/<名前> --prompt boxing-cats
 *
 * 席は `--quant`（manifest の quants のキー — 省略時は manifest の `defaultQuant`）。そのまま `fromPretrained` へ
 * 渡し（綴りの検証は hub に任せる）、出力先の名前に入れる（省略時は `default` — 他の例の CLI と同じ流儀）。
 *
 * `--prompt` / `--negative` は埋め込み資産の名前（`boxing-cats` など — その原文を渡す）か、それ以外の任意の
 * 文字列（そのまま渡す — `precomputed` では資産の集合の外なので選べる名前の一覧つきで落ちる）。未指定のノブは
 * パイプラインの既定（以下は Wan2.1 の値 — step 数・guidance・shift は manifest の `pipelineConfig` — 50・5.0・3.0、
 * 寸法とフレーム数は 832×480・33 フレーム・negative は公式の `sample_neg_prompt`）。`--frames` は 4n+1 の 5〜81（受理集合の外は
 * パイプラインが `ModelInputError` で落とす）。
 *
 * Wan2.2 の事実（`--generation wan2.2`）: step 数・guidance・shift の既定は 2.2 の manifest の `pipelineConfig` で
 * 50・5.0・5.0。寸法は 1280×704 か 704×1280（既定 1280×704）、フレーム数は 4n+1 の 5〜49（既定 33）、24 fps。
 * 席の既定は manifest の `defaultQuant` の `f16+dit8-a8-attn8-s16` で、もう 1 つの席は `f16+dit8`
 * （2.2 に `f16` の席は無い）。
 */

import {
  type DistributionSource,
  type HubRepoRef,
  parseManifest,
  resolveSelection,
} from "../../packages/hub/mod.ts";
import { denoDirectory } from "../../packages/hub/deno.ts";
import { encodePng } from "../../packages/models/mod.ts";
import { wanFrameToRgba, WanPipeline, WanTi2vPipeline } from "../../packages/models/wan.ts";
import { runMain } from "../shared/run-main.ts";
import { distributionSource } from "../shared/local-source.ts";
import { isLocalDist } from "../shared/local-assets.ts";

const USAGE =
  "--generation <wan2.1|wan2.2> --source <パス|HF repo> --umt5-source <パス> --quant <名前>" +
  " --text-encoder <gpu|precomputed> --swap-text-encoder <パス|owner/name@<commit>>" +
  " --prompt <名前|文字列> --negative <名前|文字列>" +
  " --seed <整数> --steps <整数> --frames <整数> --guidance <数> --shift <数> --size <WxH> --out <dir>";
const KNOWN = new Set([
  "generation",
  "source",
  "umt5-source",
  "quant",
  "text-encoder",
  "swap-text-encoder",
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

/**
 * 世代ごとに違う事実だけの表（パイプラインの class・既定の配布形ミラー・ミラーが無いときに出す組み立ての
 * コマンド・既定の出力先）。取得元の越境の mapping・経路・席・ノブ・出力先の名前は両世代で共通。
 * どちらの世代の組み立ても umT5 への越境参照の 5 つの `--ref-*` が要り、参照先が未公開の間は仮の SHA と
 * `--allow-placeholder-ref` を付ける（`tools/export-recipes/dist.py` の doc の組み立て例・ADR 0121 追記「段 8a の結果」で
 * 組んだコマンド）。
 */
const GENERATIONS = {
  "wan2.1": {
    pipeline: WanPipeline,
    defaultSource: "models/karume-wan2.1",
    assembleCommand: "cd tools/export-recipes && uv run python dist.py --pipeline wan" +
      " --ref-repo hdae/karume-umt5-xxl --ref-revision 0000000000000000000000000000000000000000" +
      " --ref-dist ../../models/karume-umt5-xxl --ref-model xxl --ref-role text_encoder" +
      " --allow-placeholder-ref",
    defaultOutRoot: "outputs/examples/wan2.1-t2v-1.3b",
  },
  "wan2.2": {
    pipeline: WanTi2vPipeline,
    defaultSource: "models/karume-wan2.2",
    assembleCommand: "cd tools/export-recipes && uv run python dist.py --pipeline wan-ti2v" +
      " --ref-repo hdae/karume-umt5-xxl --ref-revision 0000000000000000000000000000000000000000" +
      " --ref-dist ../../models/karume-umt5-xxl --ref-model xxl --ref-role text_encoder" +
      " --allow-placeholder-ref",
    defaultOutRoot: "outputs/examples/wan2.2-ti2v-5b",
  },
} as const;
const isGeneration = (value: string): value is keyof typeof GENERATIONS =>
  Object.hasOwn(GENERATIONS, value);
const generation = args.get("generation") ?? "wan2.1";
// MUST: 表に無い世代は落とす（`Object.hasOwn` — 打ち間違いが黙って既定の世代で走らない）。
if (!isGeneration(generation)) {
  throw new Error(`--generation ${generation} が wan2.1 / wan2.2 のどちらでもない`);
}
const GENERATION = GENERATIONS[generation];

const textEncoderArg = args.get("text-encoder");
if (textEncoderArg !== undefined && textEncoderArg !== "gpu" && textEncoderArg !== "precomputed") {
  throw new Error(`--text-encoder ${textEncoderArg} が gpu / precomputed のどちらでもない`);
}
/** 経路（未指定はパイプラインの既定と同じ `gpu` — 出力先の名前に入れるので値で持つ）。 */
const textEncoder = textEncoderArg ?? "gpu";
/** umT5 の差し替え先の綴り（未指定は差し替えない）。 */
const swapArg = args.get("swap-text-encoder");
// MUST: umT5 を読まない経路での差し替えは落とす（黙って効かないノブを残さない）。
if (swapArg !== undefined && textEncoder === "precomputed") {
  throw new Error(
    `--swap-text-encoder ${swapArg} は --text-encoder precomputed では効かない（umT5 を読まない経路）`,
  );
}
/** 席の指定（未指定は manifest の既定）。 */
const quantArg = args.get("quant");
const promptArg = args.get("prompt") ?? "boxing-cats";
const negativeArg = args.get("negative");
const seed = integer("seed") ?? 42;
const steps = integer("steps");
const frames = integer("frames");
const guidance = number("guidance");
const shift = number("shift");
const size = args.get("size");
const sizeMatch = size === undefined ? undefined : /^(\d+)x(\d+)$/.exec(size);
if (sizeMatch === null) throw new Error(`--size ${size} が WxH の形でない`);
const outRoot = args.get("out") ?? GENERATION.defaultOutRoot;

/** 既定の取得元（手元の配布形ミラー — HF の公開リポはまだ無い）。 */
const DEFAULT_SOURCE = GENERATION.defaultSource;
const ASSEMBLE_COMMAND = GENERATION.assembleCommand;
/** umT5 の配布形ミラーの既定（Wan の manifest の `text_encoder` が越境参照する先）。 */
const DEFAULT_UMT5_SOURCE = "models/karume-umt5-xxl";
const ASSEMBLE_UMT5_COMMAND = "cd tools/export-recipes && uv run python dist.py --pipeline umt5";

/**
 * ローカルの配布形の既定の選択で、umT5（`text_encoder`）の容器が越境参照する repo（自リポの容器・umT5 を
 * 持たない配布形なら undefined）。mapping のキーは manifest の宣言から引く（repo 名を写経しない）。
 */
const textEncoderRepo = async (dir: string): Promise<string | undefined> => {
  const manifest = parseManifest(await Deno.readTextFile(`${dir}/karume.json`));
  return resolveSelection(manifest).containers["text_encoder"]?.parts[0]?.repo;
};

/**
 * `--source`（と `--umt5-source`）を `fromPretrained` の取得元へ写す。未指定は既定のミラーで、無ければ落とす
 * （公開リポが無いので、他所へ黙って取りに行く先が無い）。GPU の経路でローカルの配布形の umT5 が越境参照なら、
 * 越境先のミラーを mapping で渡す。
 */
const resolveSource = async () => {
  const explicit = args.get("source");
  if (explicit === undefined && !await isLocalDist(DEFAULT_SOURCE)) {
    throw new Error(
      `配布形ミラー ${DEFAULT_SOURCE} に karume.json が無い — ${ASSEMBLE_COMMAND} で組むか、` +
        `--source で配布形を指す（使い方: ${USAGE}）`,
    );
  }
  const source = explicit ?? DEFAULT_SOURCE;
  const umt5Arg = args.get("umt5-source");
  // MUST: 効かない --umt5-source は落とす（黙って捨てると「効かないノブ」が静かに残る）。
  const ineffective = (why: string): Error =>
    new Error(`--umt5-source ${umt5Arg} は効かない（${why}）`);
  if (!await isLocalDist(source)) {
    // HF のリポ名 — 越境先は HF 取得元が宣言された (repo, revision) のまま開く。
    if (umt5Arg !== undefined) throw ineffective(`--source ${source} は HF リポ名`);
    return { from: await distributionSource(source), label: source };
  }
  // 差し替えた席は差し替え先から読み、元の越境参照を引かない — 越境先の mapping は要らない。
  const repo = textEncoder === "gpu" && swapArg === undefined
    ? await textEncoderRepo(source)
    : undefined;
  if (repo === undefined) {
    if (umt5Arg !== undefined) {
      throw ineffective(
        swapArg !== undefined
          ? `--swap-text-encoder ${swapArg} が umT5 を差し替える`
          : textEncoder === "gpu"
          ? `${source} の umT5 は越境参照でない`
          : "--text-encoder precomputed は umT5 を読まない",
      );
    }
    return { from: await distributionSource(source), label: source };
  }
  const umt5 = umt5Arg ?? DEFAULT_UMT5_SOURCE;
  if (!await isLocalDist(umt5)) {
    throw new Error(
      `umT5 の配布形 ${umt5}（${repo} の越境先）に karume.json が無い — ${ASSEMBLE_UMT5_COMMAND} で組むか、` +
        `--umt5-source で指す。umT5 を読まずに回すなら --text-encoder precomputed（使い方: ${USAGE}）`,
    );
  }
  return {
    from: await distributionSource(source, [`${repo}=${umt5}`]),
    label: `${source}（umT5: ${umt5}）`,
  };
};

/** HF の差し替え先の綴り（`owner/name@<40 桁の commit>` — revision は必須）。 */
const PINNED_REPO = /^([^@\s]+)@([0-9a-f]{40})$/;

/**
 * `--swap-text-encoder` を差し替え先の取得元へ写す（未指定は undefined）。手元の配布形は
 * `denoDirectory`、それ以外は `owner/name@<commit>` を `{ repo, revision }` にする。
 *
 * MUST: 素のリポ名（`owner/name`）は落とす — 文字列の取得元は main 追従になり（`fromPretrained` の
 * `ref` と同じ読み）、元の配布形が越境参照で pin していた umT5 の版が差し替えで黙って動く。
 */
const resolveSwap = async (): Promise<
  { readonly source: DistributionSource | HubRepoRef; readonly label: string } | undefined
> => {
  if (swapArg === undefined) return undefined;
  if (await isLocalDist(swapArg)) return { source: denoDirectory(swapArg), label: swapArg };
  const pinned = PINNED_REPO.exec(swapArg);
  if (pinned === null) {
    throw new Error(
      `--swap-text-encoder ${swapArg} が karume.json を持つディレクトリでも owner/name@<40 桁の commit> でもない` +
        `（HF のリポは commit で pin する — 素のリポ名は main 追従になり、元の配布形の pin が消える）` +
        `（使い方: ${USAGE}）`,
    );
  }
  return { source: { repo: pinned[1], revision: pinned[2] }, label: swapArg };
};

/**
 * 台本の本体。MUST: `await using` はこの中に置く（`shared/run-main.ts` — 本体と解放が両方投げたときの
 * `SuppressedError` を展開するため）。
 */
/** 文字列の sha256（小文字 16 進 — 出力先の名前に使う）。 */
const sha256Hex = async (text: string): Promise<string> =>
  Array.from(
    new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text))),
    (byte) => byte.toString(16).padStart(2, "0"),
  ).join("");

const main = async (): Promise<void> => {
  // 差し替え先の綴りの誤りを先に見せる（Wan のミラーが無い機で、ミラー不在のエラーに隠さない）。
  const swap = await resolveSwap();
  const { from, label } = await resolveSource();
  /** 出力先の名前に入れる経路（差し替えた回は元の umT5 の回と別のディレクトリにする）。 */
  const route = swap === undefined ? textEncoder : "gpu-swap";
  console.log(
    `[wan] source: ${label}・generation: ${generation}・quant: ${
      quantArg ?? "（manifest の既定）"
    }・text encoder: ${
      swap === undefined ? textEncoder : `${textEncoder}（差し替え: ${swap.label}）`
    }`,
  );
  await using pipeline = await GENERATION.pipeline.fromPretrained(from, {
    ...(quantArg === undefined ? {} : { quant: quantArg }),
    textEncoder,
    ...(swap === undefined ? {} : { components: { text_encoder: { source: swap.source } } }),
  });

  // 資産の名前ならその原文、それ以外は渡した文字列そのもの（受理はパイプラインの門が決める — 経路ごと）。
  const textOf = (value: string): string =>
    pipeline.prompts.find((candidate) => candidate.name === value)?.prompt ?? value;
  const prompt = textOf(promptArg);
  const isAssetName = pipeline.prompts.some((candidate) => candidate.name === promptArg);
  // 出力先の名前: 資産の名前はそのまま、任意の文字列は sha256 の先頭 8 桁（綴りをパスに入れない）。
  const promptLabel = isAssetName ? promptArg : `prompt-${(await sha256Hex(prompt)).slice(0, 8)}`;
  console.log(`[wan] ${promptLabel}: ${prompt.trim()}`);
  const started = performance.now();
  const encoder = new TextEncoder();
  // 実際に回した step 数（--steps 省略時は manifest の既定）。出力先の名前に入れるので、イベントが
  // 運ぶ値を取る（既定を台本に写すと配布形の既定が変わったときに名前と中身が食い違う）。
  let ranSteps: number | undefined;
  const video = await pipeline.generate({
    prompt,
    seed,
    ...(negativeArg === undefined ? {} : { negativePrompt: textOf(negativeArg) }),
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
        ranSteps = event.steps;
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
  if (ranSteps === undefined) throw new Error("denoise-step のイベントが 1 度も来なかった");
  // NOTE: guidance / shift / negative は名前に入らない — 変えて比べるときは --out で分ける。
  const outDir = `${outRoot}/wan-${quantArg ?? "default"}-${promptLabel}-${route}` +
    `-${video.width}x${video.height}-${video.frames}f-${ranSteps}step-seed${seed}`;
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
