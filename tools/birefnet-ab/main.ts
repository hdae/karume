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
 * パイプラインと同じ 1 本（`loadBirefnetState` / `matteLogits`）。
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
import { encodePng } from "../../packages/models/mod.ts";
import {
  type BirefnetState,
  loadBirefnetState,
  matteLogits,
} from "../../packages/models/src/birefnet/pipeline.ts";
import {
  MATTE_PROCEDURES,
  matteFromLogits,
  type MatteProcedure,
} from "../../packages/models/src/birefnet/matte.ts";
import { decodePng } from "../../packages/runtime/tests/helpers/png-decode.ts";
import { buildSafetensors } from "../../packages/runtime/tests/helpers/safetensors.ts";
import {
  type Background,
  blindAssignment,
  composite,
  contourBand,
  errorAgainst,
  type KernelDifference,
  kernelDifference,
  largestDifferenceWindow,
  type MatteError,
  type Plane,
  pseudoTruth,
  type Rgb,
  tile,
  type Window,
  zoomNearest,
} from "./analysis.ts";
import { animaCommand, animaFile, CORPUS } from "./corpus.ts";

/** 上流の手順（HR の上流の意図）と今の経路（ADR 0125）。 */
const UPSTREAM: MatteProcedure = "u8-pil-bicubic";
const CURRENT: MatteProcedure = "float-bilinear";

/** A/B の元になるモデル（グラフ 1024²）と、擬似正解のモデル（グラフ 2048²）。 */
const BASE_MODEL = "1024";
const REFERENCE_MODEL = "2048";

const DEFAULT_SOURCE = "models/karume-birefnet-hr";
const DEFAULT_IMAGES = "outputs/examples/karume-anima";
/** 盲検の割り付けの既定の seed（対応表の再現に使う — mapping.json に残す）。 */
const DEFAULT_SEED = 20261011;

/** 差の大きい窓の辺と拡大の倍率（目視で輪郭の画素の段が見える大きさ）。 */
const ZOOM_WINDOW = 64;
const ZOOM_FACTOR = 8;
/** 並べる画像の隙間（画素）。 */
const GAP = 8;

const BACKGROUNDS: readonly Background[] = ["checker", "black", "white"];

const USAGE = "commands | run [--source <配布形>] [--images <dir>] [--out <dir>] [--seed <整数>]" +
  " [--image <PNG> [--crop <left>,<top>,<width>,<height>]]";

/** 1 枚の入力（素材か dry-run の画像）。 */
type Subject = {
  readonly name: string;
  readonly path: string;
  readonly crop?: Window;
};

/** 画像 1 枚ぶんの A/B の材料。 */
type Pair = {
  readonly subject: Subject;
  readonly image: Rgb;
  readonly alphas: Readonly<Record<MatteProcedure, Plane>>;
};

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

/** RGB8 → `encodePng` の受ける RGBA（α は 255 — 合成済みの不透明な画像）。 */
const pngOf = (image: Rgb): Promise<Uint8Array<ArrayBuffer>> => {
  const rgba = new Uint8Array(image.width * image.height * 4).fill(255);
  for (let index = 0; index < image.width * image.height; index += 1) {
    rgba.set(image.data.subarray(index * 3, index * 3 + 3), index * 4);
  }
  return encodePng(rgba, image.width, image.height);
};

const u8Spec = (name: string, plane: Plane) => ({
  name,
  dtype: "U8",
  shape: [plane.height, plane.width],
  data: new Uint8Array(plane.data.buffer, plane.data.byteOffset, plane.data.byteLength),
});

const f32Spec = (name: string, shape: readonly number[], data: Float32Array<ArrayBuffer>) => ({
  name,
  dtype: "F32",
  shape,
  data: new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
});

/** グラフ出力の名前（logit の safetensors のテンソル名 — e2e の実物と同じ綴り）。 */
const outputName = (state: BirefnetState): string => state.matte.graph.outputs[0];

/**
 * 1024² のモデルで画像ごとに 1 回だけグラフを回し、logit を保存して手順を全部当てる。
 * MUST: Session は最後に必ず畳む（GPU は呼び手のもの）。
 */
const runBase = async (
  gpu: GpuContext,
  source: string,
  subjects: readonly Subject[],
  work: string,
): Promise<{ readonly pairs: Pair[]; readonly graph: number }> => {
  const state = await loadBirefnetState(denoDirectory(source), { model: BASE_MODEL, gpu });
  try {
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
      await Deno.writeFile(
        `${work}/${subject.name}.logits-${BASE_MODEL}.safetensors`,
        new Uint8Array(
          buildSafetensors([f32Spec(outputName(state), [1, 1, imageHeight, imageWidth], logits)]),
        ),
      );
      const apply = (procedure: MatteProcedure): Plane =>
        matteFromLogits(logits, imageWidth, imageHeight, image.width, image.height, procedure);
      const alphas: Record<MatteProcedure, Plane> = {
        "float-bilinear": apply("float-bilinear"),
        "u8-pil-bicubic": apply("u8-pil-bicubic"),
      };
      await Deno.writeFile(
        `${work}/${subject.name}.alpha.safetensors`,
        new Uint8Array(
          buildSafetensors(
            MATTE_PROCEDURES.map((procedure) => u8Spec(procedure, alphas[procedure])),
          ),
        ),
      );
      pairs.push({ subject, image, alphas });
      console.log(`[birefnet-ab] ${subject.name}: ${image.width}×${image.height}`);
    }
    return { pairs, graph: imageWidth };
  } finally {
    await state.session.dispose();
  }
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
  try {
    const { imageWidth, imageHeight } = state.config;
    if (imageWidth !== size || imageHeight !== size) {
      throw new Error(`モデル '${REFERENCE_MODEL}' のグラフが ${imageWidth}×${imageHeight}`);
    }
    for (const { subject, image } of targets) {
      const logits = await matteLogits(state, image);
      await Deno.writeFile(
        `${work}/${subject.name}.logits-${REFERENCE_MODEL}.safetensors`,
        new Uint8Array(
          buildSafetensors([f32Spec(outputName(state), [1, 1, imageHeight, imageWidth], logits)]),
        ),
      );
      const sigma = new Float64Array(logits.length);
      for (let index = 0; index < logits.length; index += 1) {
        sigma[index] = 1 / (1 + Math.exp(-logits[index]));
      }
      truths.set(subject.name, sigma);
      console.log(`[birefnet-ab] ${subject.name}: 擬似正解（モデル ${REFERENCE_MODEL}）`);
    }
    return truths;
  } finally {
    await state.session.dispose();
  }
};

/** 1 対ぶんの数値の列（記録だけ）。 */
type PairMetrics = {
  readonly pair: number;
  readonly name: string;
  readonly width: number;
  readonly height: number;
  readonly kernel: KernelDifference;
  readonly error?: Readonly<Record<MatteProcedure, MatteError>>;
};

const sizeLabel = (image: Rgb, graph: number): string => {
  const direction = image.width > graph && image.height > graph
    ? "拡大"
    : image.width < graph && image.height < graph
    ? "縮小"
    : "片軸";
  return `${image.width}×${image.height}（${direction}）`;
};

const formatNumber = (value: number): string =>
  value === 0 ? "0" : Math.abs(value) >= 0.01 ? value.toFixed(3) : value.toExponential(2);

const answerSheet = (pairs: readonly Pair[], graph: number): string =>
  [
    "# BiRefNet の後処理の A/B — 回答用紙（盲検・3 択）",
    "",
    "- 各対は同じ画像の α マットを 2 つの手順で作ったもの。左が A、右が B（どちらがどの手順かは伏せて" +
    "ある）。",
    "- 見る画像: `pair-<番号>-checker.png`（市松の地）・`-black.png`（黒地）・`-white.png`（白地）・" +
    `\`-zoom.png\`（2 つの差が最も大きい ${ZOOM_WINDOW}×${ZOOM_WINDOW} の窓を ${ZOOM_FACTOR} 倍に` +
    "拡大 — 上から市松・黒・白）。",
    "- 輪郭（髪・毛皮・細い物の縁）がより自然に見えるほうに `x` を付ける。差が分からなければ「同じ」。",
    "- 答え終わるまで `../key/` を開かない。",
    "",
    "| 対 | 画像 | 寸法 | A | 同じ | B |",
    "| -- | ---- | ---- | - | ---- | - |",
    ...pairs.map(({ subject, image }, index) =>
      `| ${index + 1} | ${subject.name} | ${sizeLabel(image, graph)} | [ ] | [ ] | [ ] |`
    ),
    "",
  ].join("\n");

const metricsTable = (rows: readonly PairMetrics[]): string =>
  [
    "# BiRefNet の後処理の A/B — 数値の列（記録だけ・判定には使わない — ADR 0125）",
    "",
    `- 上流 = \`${UPSTREAM}\`・今 = \`${CURRENT}\`。帯 = どちらかの手順で 0 < α < 255 の画素を 2 画素` +
    "膨張した範囲。",
    "- K 列: 帯の中の 2 手順の差（8bit の段）。総合列: 目標が 2048² の画像だけ、2048² モデルの σ を" +
    "擬似正解にした帯の中の SAD（Σ|Δα|）・MSE・Gradient error（α ∈ [0, 1]・σ = 1.4 のガウス微分）。",
    "- 2048² モデルは輪郭が鋭く、鋭いカーネルに有利に振れうる（推測）。",
    "",
    "| 対 | 画像 | 帯の画素 | 差のある画素 | 平均 \\|差\\|（段） | 最大 \\|差\\|（段） | SAD 上流 | SAD 今 |" +
    " MSE 上流 | MSE 今 | Grad 上流 | Grad 今 |",
    "| -- | ---- | -------- | ------------ | ------------------ | ------------------ | -------- | ------ |" +
    " -------- | ------ | --------- | ------- |",
    ...rows.map(({ pair, name, kernel, error }) => {
      const pick = (procedure: MatteProcedure, key: keyof MatteError): string =>
        error === undefined ? "—" : formatNumber(error[procedure][key]);
      return `| ${pair} | ${name} | ${kernel.bandPixels} | ${kernel.differing} | ` +
        `${formatNumber(kernel.meanAbs)} | ${kernel.maxAbs} | ${pick(UPSTREAM, "sad")} | ` +
        `${pick(CURRENT, "sad")} | ${pick(UPSTREAM, "mse")} | ${pick(CURRENT, "mse")} | ` +
        `${pick(UPSTREAM, "gradient")} | ${pick(CURRENT, "gradient")} |`;
    }),
    "",
  ].join("\n");

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

  const [blind, key, work] = ["blind", "key", "work"].map((name) => `${out}/${name}`);
  for (const dir of [blind, key, work]) await Deno.mkdir(dir, { recursive: true });

  const gpu = await acquireGpu();
  const { pairs, graph, truths } = await (async () => {
    try {
      const base = await runBase(gpu, source, subjects, work);
      return { ...base, truths: await runReference(gpu, source, base.pairs, work) };
    } finally {
      gpu.destroy();
    }
  })();

  const assignment = blindAssignment(pairs.length, seed);
  const mapping = [];
  const metrics: PairMetrics[] = [];
  for (const [index, { subject, image, alphas }] of pairs.entries()) {
    const number = index + 1;
    const [left, right] = assignment[index] ? [UPSTREAM, CURRENT] : [CURRENT, UPSTREAM];
    const window = largestDifferenceWindow(alphas[left], alphas[right], ZOOM_WINDOW);
    const zoomRows: Rgb[][] = [];
    for (const background of BACKGROUNDS) {
      const pair = [
        composite(image, alphas[left], background),
        composite(image, alphas[right], background),
      ];
      await Deno.writeFile(
        `${blind}/pair-${number}-${background}.png`,
        await pngOf(tile([pair], GAP)),
      );
      zoomRows.push(pair.map((half) => zoomNearest(half, window, ZOOM_FACTOR)));
    }
    await Deno.writeFile(`${blind}/pair-${number}-zoom.png`, await pngOf(tile(zoomRows, GAP)));

    const band = contourBand(alphas[UPSTREAM], alphas[CURRENT]);
    const sigma = truths.get(subject.name);
    const truth = sigma === undefined ? undefined : pseudoTruth(sigma, image.width, image.height);
    metrics.push({
      pair: number,
      name: subject.name,
      width: image.width,
      height: image.height,
      kernel: kernelDifference(alphas[UPSTREAM], alphas[CURRENT], band),
      ...(truth === undefined ? {} : {
        error: {
          "float-bilinear": errorAgainst(alphas["float-bilinear"], truth, band),
          "u8-pil-bicubic": errorAgainst(alphas["u8-pil-bicubic"], truth, band),
        },
      }),
    });
    mapping.push({ pair: number, name: subject.name, image: subject.path, left, right, window });
  }

  await Deno.writeTextFile(`${blind}/answer-sheet.md`, answerSheet(pairs, graph));
  await Deno.writeTextFile(
    `${key}/mapping.json`,
    JSON.stringify(
      {
        seed,
        source,
        procedures: { upstream: UPSTREAM, current: CURRENT },
        rule:
          `今の経路（${CURRENT}）を残すのは「今の経路」が 5 対以上かつ「上流」が 0 対のときだけ。` +
          `それ以外は上流（${UPSTREAM}）を採る（ADR 0125）。`,
        pairs: mapping,
      },
      null,
      2,
    ) + "\n",
  );
  await Deno.writeTextFile(`${key}/metrics.md`, metricsTable(metrics));
  await Deno.writeTextFile(`${key}/metrics.json`, JSON.stringify(metrics, null, 2) + "\n");
  await Deno.writeTextFile(
    `${work}/manifest.json`,
    JSON.stringify(
      {
        graph: { width: graph, height: graph },
        entries: pairs.map(({ subject, image }) => ({
          name: subject.name,
          width: image.width,
          height: image.height,
          logits: `${subject.name}.logits-${BASE_MODEL}.safetensors`,
          alpha: `${subject.name}.alpha.safetensors`,
        })),
      },
      null,
      2,
    ) + "\n",
  );
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
