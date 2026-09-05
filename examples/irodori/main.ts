/**
 * Irodori（テキスト → 音声）の 1 画面デモ。資産の出所だけが分岐で、あとは `generate` のノブ。
 *
 *     deno task demo:irodori --text "こんにちは、これはテストです。"
 *     deno task demo:irodori --caption "落ち着いた女性の声で、ゆっくりと話している。" --seed 7
 *     deno task demo:irodori --ref inputs/irodori/v4-small/samples/clone_ref1.wav
 *
 * `--source` 未指定ならこの台本が `IRODORI_SOURCES["irodori-v4.1-small"]`（このパッケージ版が
 * 検証した取得元 — ADR 0073 / 0092。旧版 v4 の pin `IRODORI_SOURCES["irodori-v4-small"]` も
 * 同じ表に残っている）を渡す。`fromPretrained` 自体に既定は無いので、取得元を綴るのは常に
 * 呼び出し側。明示したときだけ、`karume.json` を持つディレクトリなら `denoDirectory` で直に
 * 読み、それ以外は HF リポジトリ名として読む（どちらも `fromPretrained` の 1 本 — shard 分割
 * された配布形もそのまま通る）。越境参照を持つ配布形は `--source-map owner/name=<パス>` で
 * 越境先を名指しする（繰り返し可）。`--ref` は参照音声
 * （WAV — 配布形と同じ 48kHz の mono/多ch PCM16 か IEEE float）で、渡すとその声質に寄る
 * （voice cloning）。`--caption` は声質の指示文（Voice Design）。サンプラのノブ（steps / CFG）は
 * manifest の `pipelineConfig` が固定していて、実行時には `--seed` と `--seconds`（発話長の
 * 直接指定）だけが動かせる。
 */

import {
  decodeWav,
  encodeWav,
  IrodoriPipeline,
  type IrodoriSpeakerInput,
} from "../../packages/models/mod.ts";
import { IRODORI_SOURCES } from "../../packages/models/irodori.ts";
import { distributionSource } from "../shared/local-source.ts";
import { runMain } from "../shared/run-main.ts";

const USAGE = "--source <パス|HF repo> --source-map <owner/name=パス> --text <文字列>" +
  " --caption <文字列> --ref <WAV パス>" +
  " --model <名前> --quant <名前> --seconds <数> --seed <整数> --out <パス>";
const KNOWN = new Set([
  "source",
  "source-map",
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

// MUST: 効かないノブを黙って捨てない（`local-source.ts` が HF リポ名 + mapping で落とすのと
// 同じ線）。`--source` 未指定の取得元は焼き込み pin（`{repo, revision}`）で、越境 mapping を
// 渡す口が無い — 黙って捨てると「mapping を渡したのに既定 pin を取りに行く」が沈黙する。
if (source === undefined && sourceMaps.length > 0) {
  throw new Error(
    `--source-map は --source を明示したときだけ効く（既定の取得元は HF の pin）（使い方: ${USAGE}）`,
  );
}

/**
 * 台本の本体。
 *
 * MUST: `await using` はこの中に置く。モジュール本体で掴むと、本体と解放が両方投げたときの
 * `SuppressedError` を誰も展開できず、device 消失の理由が画面に出ない（`shared/run-main.ts`）。
 */
const main = async (): Promise<void> => {
  /** 参照音声（任意）。周波数が配布形と違えば `generate` が fail loudly する（リサンプル無し）。 */
  const speaker: IrodoriSpeakerInput | undefined = ref === undefined
    ? undefined
    : { audio: decodeWav(await Deno.readFile(ref)) };

  /** 取得元（ローカルの配布形なら `denoDirectory`・それ以外は HF リポジトリ名）。 */
  const from = source === undefined
    ? IRODORI_SOURCES["irodori-v4.1-small"]
    : await distributionSource(source, sourceMaps);

  console.log(
    `[irodori] ${
      source ?? `${IRODORI_SOURCES["irodori-v4.1-small"].repo}（台本の既定 = 検証済み pin）`
    }` +
      ` / model ${model ?? "（manifest の既定）"}` +
      ` / quant ${quant ?? "（manifest の既定）"} / seed ${seed}` +
      `${ref === undefined ? "" : ` / 参照 ${ref}`}\n` +
      `          ${JSON.stringify(text)}` +
      `${caption === undefined ? "" : `\n          caption ${JSON.stringify(caption)}`}`,
  );
  const started = performance.now();
  await using pipeline = await IrodoriPipeline.fromPretrained(
    from,
    {
      ...selection,
      onProgress: ({ phase, loaded, total }) =>
        Deno.stderr.writeSync(
          encoder.encode(`\r  ${phase} ${(loaded / total * 100).toFixed(1)}%  `),
        ),
    },
  );
  const audio = await pipeline.generate({
    text,
    seed,
    ...(caption === undefined ? {} : { caption }),
    ...(speaker === undefined ? {} : { speaker }),
    ...(seconds === undefined ? {} : { durationSeconds: seconds }),
  });
  const name = `irodori-${quant ?? "default"}-${ref === undefined ? "no-ref" : "cloned"}` +
    `-seed${seed}.wav`;
  /** 既定の出力先に使うモデル名（取得元の末尾要素 — パスでも HF リポ名でも同じ規則）。 */
  const sourceRef = source ?? IRODORI_SOURCES["irodori-v4.1-small"].repo;
  const sourceName = sourceRef.replace(/\/+$/, "").split("/").at(-1) ?? sourceRef;
  const out = args.get("out") ?? `outputs/examples/${sourceName}/${name}`;
  // MUST: `cut > 0` で判定する。`-1`（`/` 無し = cwd 直下）を切ると 1 文字削ったディレクトリを
  // 作り、`0`（絶対パスの根）を切ると空文字列で `mkdir` を呼ぶ。
  const cut = out.lastIndexOf("/");
  if (cut > 0) await Deno.mkdir(out.slice(0, cut), { recursive: true });
  await Deno.writeFile(out, encodeWav(audio.data, audio.sampleRate));
  console.log(
    `[irodori] ${out}（${(audio.data.length / audio.sampleRate).toFixed(2)}s / S ${audio.frames}` +
      ` / dit ${audio.forwards} forward / ${((performance.now() - started) / 1000).toFixed(1)}s）`,
  );
};

await runMain(main);
