/**
 * birefnet-ab — BiRefNet の α マットの後処理の A/B の道具（ADR 0125 — 裁定 4・HR だけ）。
 *
 *     deno run -A tools/birefnet-ab/main.ts commands
 *     deno run -A tools/birefnet-ab/main.ts run
 *     deno run -A tools/birefnet-ab/main.ts run --out outputs/birefnet-ab/2026-10-12 --seed 20261011
 *     deno run -A tools/birefnet-ab/main.ts run --image outputs/misc/corpus/<PNG> \
 *         --crop 96,272,832,480 --out outputs/birefnet-ab/dry-run
 *
 * `commands` は素材の生成（anima）と、この道具・検算の実行のコマンドを並べるだけ（何も実行しない）。
 * `run` は素材（`corpus.ts` — 既定の置き場は anima の出力 `outputs/examples/karume-anima/`）を画像ごとに
 * **グラフへ 1 回だけ**通して logit を保存し、`matte.ts` の手順を全部当てる（手順ごとにグラフを回し
 * 直さない）。`--image` は素材の代わりに 1 枚だけを通す（dry-run — `--crop` は補間を通さない切り出し）。
 *
 * GPU は 1 つを取り、1024² のモデル（全画像）→ 2048² のモデル（目標が 2048² ちょうどの画像だけ —
 * 擬似正解）の順に Session を 1 本ずつ張って畳む（2 本を同時に持たない）。読み込みと門は
 * パイプラインと同じ 1 本（`loadBirefnetState` / `matteLogits`）。このファイルは GPU の段だけを持ち、
 * 手順の当て方と書き出しは `render.ts`（GPU に触らない）。
 *
 * 出力（`--out` の下だけに書く）:
 *
 * - `blind/` — 利用者に見せる盲検の対（手順の名前を含まない）: `pair-<番号>-{checker,black,white}.png`
 *   （左 = A・右 = B の合成）と `pair-<番号>-zoom.png`（差の大きい 64×64 の窓の 8 倍 — 上から市松・黒・
 *   白）、回答用紙 `answer-sheet.md`（A / 同じ / B の 3 択 — 裁定 30）。
 * - `key/` — 回答の後で開く: 対応表 `mapping.json`（seed 固定 — 同じ seed で再現する）と、数値の列の
 *   表 `metrics.md` / `metrics.json`（K 列・総合列 — 記録だけで判定には使わない）。
 * - `work/` — 検算用の中間物: 画像ごとの logit（グラフ出力そのもの）と手順ごとの 8bit α
 *   （`tools/export-recipes/birefnet/matte_ab_check.py` が golden の生成器と同じ上流の関数で検算する）。
 */

import { acquireGpu, type GpuContext } from "../../packages/runtime/mod.ts";
import { denoDirectory } from "../../packages/hub/deno.ts";
import {
  type BirefnetState,
  loadBirefnetState,
  matteLogits,
} from "../../packages/models/src/birefnet/pipeline.ts";
import { decodePng } from "../../packages/runtime/tests/helpers/png-decode.ts";
import type { Rgb, Window } from "./analysis.ts";
import { animaCommand, animaFile, CORPUS } from "./corpus.ts";
import {
  alphasOf,
  logitsFile,
  type Pair,
  type Subject,
  writeAbOutputs,
  writeAlphas,
  writeLogits,
} from "./render.ts";

/** A/B の元になるモデル（グラフ 1024²）と、擬似正解のモデル（グラフ 2048²）。 */
const BASE_MODEL = "1024";
const REFERENCE_MODEL = "2048";

const DEFAULT_SOURCE = "models/karume-birefnet-hr";
const DEFAULT_IMAGES = "outputs/examples/karume-anima";
/** 盲検の割り付けの既定の seed（対応表の再現に使う — mapping.json に残す）。 */
const DEFAULT_SEED = 20261011;

const USAGE = "commands | run [--source <配布形>] [--images <dir>] [--out <dir>] [--seed <整数>]" +
  " [--image <PNG> [--crop <left>,<top>,<width>,<height>]]";

const parseArgs = (args: readonly string[]): { command: string; options: Map<string, string> } => {
  const [command, ...rest] = args;
  if (command !== "commands" && command !== "run") throw new Error(`使い方: ${USAGE}`);
  const known = new Set(["source", "images", "out", "seed", "image", "crop"]);
  const options = new Map<string, string>();
  for (let at = 0; at < rest.length; at += 2) {
    const [key, value] = [rest[at], rest[at + 1]];
    // MUST: 未知のキーと値の欠けは落とす（打ち間違いが黙って既定値で走ると、別の素材の A/B になる）。
    if (!key.startsWith("--") || value === undefined || value.startsWith("--")) {
      throw new Error(`引数 ${key} が --key value の対になっていない（使い方: ${USAGE}）`);
    }
    if (!known.has(key.slice(2))) throw new Error(`未知のオプション ${key}（使い方: ${USAGE}）`);
    options.set(key.slice(2), value);
  }
  if (options.has("crop") && !options.has("image")) {
    throw new Error("--crop は --image と組で使う（素材の切り出しは corpus.ts が持つ）");
  }
  return { command, options };
};

const parseCrop = (spelled: string): Window => {
  const parts = spelled.split(",").map(Number);
  if (parts.length !== 4 || !parts.every((value) => Number.isInteger(value) && value >= 0)) {
    throw new Error(`--crop ${spelled} が <left>,<top>,<width>,<height> の非負整数でない`);
  }
  const [left, top, width, height] = parts;
  return { left, top, width, height };
};

/** 補間を通さない切り出し（範囲の外は fail loudly）。 */
const cropRgb = (image: Rgb, window: Window, where: string): Rgb => {
  const { left, top, width, height } = window;
  if (width <= 0 || height <= 0 || left + width > image.width || top + height > image.height) {
    throw new Error(
      `${where}: 切り出し ${JSON.stringify(window)} が画像 ${image.width}×${image.height} の外`,
    );
  }
  const data = new Uint8Array(width * height * 3);
  for (let y = 0; y < height; y += 1) {
    const from = ((top + y) * image.width + left) * 3;
    data.set(image.data.subarray(from, from + width * 3), y * width * 3);
  }
  return { data, width, height };
};

const loadSubject = async (subject: Subject): Promise<Rgb> => {
  const decoded = await decodePng(await Deno.readFile(subject.path), subject.path);
  return subject.crop === undefined ? decoded : cropRgb(decoded, subject.crop, subject.path);
};

/** グラフ出力の名前（logit の safetensors のテンソル名 — e2e の実物と同じ綴り）。 */
const outputName = (state: BirefnetState): string => state.matte.graph.outputs[0];

/**
 * `body` を走らせ、成否に関わらず最後に Session を畳む（GPU は呼び手のもの）。
 *
 * MUST: 本体と解放の両方が投げたら本体の例外を消さない — `finally` の中で投げると元の失敗を上書き
 * して、最初に何が壊れたかが消える（`packages/models/src/session/dispose-steps.ts` と同じ理由）。
 * 失敗の運び方も `disposeSteps` に揃える: 1 件ならその例外をそのまま、2 件なら本体を先頭にした
 * `AggregateError`（`disposeSteps` を直接使わないのは、本体の失敗まで「後始末の失敗」と名乗るため）。
 */
const withSession = async <T>(state: BirefnetState, body: () => Promise<T>): Promise<T> => {
  let value: T;
  try {
    value = await body();
  } catch (error) {
    try {
      await state.session.dispose();
    } catch (disposeError) {
      throw new AggregateError(
        [error, disposeError],
        "birefnet-ab: 本体が失敗し、Session の解放も失敗した",
      );
    }
    throw error;
  }
  await state.session.dispose();
  return value;
};

/**
 * 1024² のモデルで画像ごとに 1 回だけグラフを回し、logit を保存して手順を全部当てる。
 * MUST: Session は最後に必ず畳む（{@link withSession}）。
 */
const runBase = async (
  gpu: GpuContext,
  source: string,
  subjects: readonly Subject[],
  work: string,
): Promise<{ readonly pairs: Pair[]; readonly graph: number }> => {
  const state = await loadBirefnetState(denoDirectory(source), { model: BASE_MODEL, gpu });
  return await withSession(state, async () => {
    const { imageWidth, imageHeight } = state.config;
    if (imageWidth !== imageHeight) {
      throw new Error(
        `モデル '${BASE_MODEL}' のグラフが正方でない（${imageWidth}×${imageHeight}）`,
      );
    }
    const pairs: Pair[] = [];
    for (const subject of subjects) {
      const image = await loadSubject(subject);
      const logits = await matteLogits(state, image);
      await writeLogits(
        work,
        logitsFile(subject.name, BASE_MODEL),
        outputName(state),
        imageWidth,
        logits,
      );
      const alphas = alphasOf(logits, imageWidth, image);
      await writeAlphas(work, subject.name, alphas);
      pairs.push({ subject, image, alphas });
      console.log(`[birefnet-ab] ${subject.name}: ${image.width}×${image.height}`);
    }
    return { pairs, graph: imageWidth };
  });
};

/**
 * 目標が 2048² ちょうどの画像だけ、2048² のモデルの σ（resize を通さない）を擬似正解として採る。
 * 対象が無ければモデルを張らない。
 */
const runReference = async (
  gpu: GpuContext,
  source: string,
  pairs: readonly Pair[],
  work: string,
): Promise<Map<string, Float64Array<ArrayBuffer>>> => {
  const truths = new Map<string, Float64Array<ArrayBuffer>>();
  const size = Number(REFERENCE_MODEL);
  const targets = pairs.filter(({ image }) => image.width === size && image.height === size);
  if (targets.length === 0) return truths;
  const state = await loadBirefnetState(denoDirectory(source), { model: REFERENCE_MODEL, gpu });
  return await withSession(state, async () => {
    const { imageWidth, imageHeight } = state.config;
    if (imageWidth !== size || imageHeight !== size) {
      throw new Error(`モデル '${REFERENCE_MODEL}' のグラフが ${imageWidth}×${imageHeight}`);
    }
    for (const { subject, image } of targets) {
      const logits = await matteLogits(state, image);
      await writeLogits(
        work,
        logitsFile(subject.name, REFERENCE_MODEL),
        outputName(state),
        size,
        logits,
      );
      const sigma = new Float64Array(logits.length);
      for (let index = 0; index < logits.length; index += 1) {
        sigma[index] = 1 / (1 + Math.exp(-logits[index]));
      }
      truths.set(subject.name, sigma);
      console.log(`[birefnet-ab] ${subject.name}: 擬似正解（モデル ${REFERENCE_MODEL}）`);
    }
    return truths;
  });
};

const run = async (options: Map<string, string>): Promise<void> => {
  const source = options.get("source") ?? DEFAULT_SOURCE;
  const out = options.get("out") ?? `outputs/birefnet-ab/${new Date().toISOString().slice(0, 10)}`;
  const spelledSeed = options.get("seed");
  if (spelledSeed !== undefined && !/^\d+$/.test(spelledSeed)) {
    throw new Error(`--seed ${spelledSeed} が非負整数でない`);
  }
  const seed = spelledSeed === undefined ? DEFAULT_SEED : Number(spelledSeed);
  const single = options.get("image");
  const crop = options.get("crop");
  const subjects: Subject[] = single === undefined
    ? CORPUS.map((entry) => ({
      name: entry.name,
      path: `${options.get("images") ?? DEFAULT_IMAGES}/${animaFile(entry)}`,
      ...(entry.crop === undefined ? {} : { crop: entry.crop }),
    }))
    : [{
      name: single.replace(/^.*\//, "").replace(/\.png$/i, ""),
      path: single,
      ...(crop === undefined ? {} : { crop: parseCrop(crop) }),
    }];
  // MUST: 素材が欠けていれば GPU を取る前に名指しで落とす（欠けた素材を黙って飛ばすと、対の数が
  // 判定規則の前提〈8 対〉から外れたまま表が出る）。
  const missing: string[] = [];
  for (const { path } of subjects) {
    try {
      await Deno.stat(path);
    } catch (cause) {
      if (!(cause instanceof Deno.errors.NotFound)) throw cause;
      missing.push(path);
    }
  }
  if (missing.length > 0) {
    throw new Error(
      `素材が無い: ${
        missing.join(" / ")
      }（生成のコマンドは \`deno run -A tools/birefnet-ab/main.ts` +
        " commands`）",
    );
  }

  const work = `${out}/work`;
  await Deno.mkdir(work, { recursive: true });
  const gpu = await acquireGpu();
  const { pairs, graph, truths } = await (async () => {
    try {
      const base = await runBase(gpu, source, subjects, work);
      return { ...base, truths: await runReference(gpu, source, base.pairs, work) };
    } finally {
      gpu.destroy();
    }
  })();
  await writeAbOutputs({ out, pairs, truths, graph, model: BASE_MODEL, seed, source });
  console.log(`[birefnet-ab] ${out}（盲検の対 ${pairs.length} 組・seed ${seed}）`);
};

const commands = (): void => {
  const lines = [
    "# 1. 素材（anima — リポの根で・GPU）",
    ...CORPUS.map(animaCommand),
    "# 2. A/B の材料（GPU）",
    "deno run -A tools/birefnet-ab/main.ts run --out outputs/birefnet-ab/<日付>",
    "# 3. 検算（CPU — golden の生成器と同じ上流の関数）",
    "cd tools/export-recipes && uv run --group birefnet python -m birefnet.matte_ab_check" +
    " ../../outputs/birefnet-ab/<日付>/work",
  ];
  console.log(lines.join("\n"));
};

const { command, options } = parseArgs(Deno.args);
if (command === "commands") commands();
else await run(options);
