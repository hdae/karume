/**
 * A/B の道具の**書き出し**（GPU の後の段）: 手順の当て方・検算用の中間物（`work/`）・盲検の対と
 * 回答用紙（`blind/`）・対応表と数値の列（`key/`）。GPU に触らないので、手元の logit から同じ
 * 書き出しを再現できる（`main.ts` は GPU の段だけを持つ）。置き場の説明は `main.ts` のモジュール doc。
 */

import { encodePng } from "../../packages/models/mod.ts";
import {
  MATTE_PROCEDURES,
  matteFromLogits,
  type MatteProcedure,
} from "../../packages/models/src/birefnet/matte.ts";
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

/** 上流の手順（HR の上流の意図）と今の経路（ADR 0125）。 */
export const UPSTREAM: MatteProcedure = "u8-pil-bicubic";
export const CURRENT: MatteProcedure = "float-bilinear";

/** 差の大きい窓の辺と拡大の倍率（目視で輪郭の画素の段が見える大きさ）。 */
const ZOOM_WINDOW = 64;
const ZOOM_FACTOR = 8;
/** 並べる画像の隙間（画素）。 */
const GAP = 8;

const BACKGROUNDS: readonly Background[] = ["checker", "black", "white"];

/** 1 枚の入力（素材か dry-run の画像）。 */
export type Subject = {
  readonly name: string;
  readonly path: string;
  readonly crop?: Window;
};

/** 画像 1 枚ぶんの A/B の材料。 */
export type Pair = {
  readonly subject: Subject;
  readonly image: Rgb;
  readonly alphas: Readonly<Record<MatteProcedure, Plane>>;
};

/** グラフ `graph`² の logit に手順を全部当てる（1 回の実行の logit から — 手順ごとに回し直さない）。 */
export const alphasOf = (
  logits: Float32Array<ArrayBuffer>,
  graph: number,
  image: Rgb,
): Record<MatteProcedure, Plane> => {
  const apply = (procedure: MatteProcedure): Plane =>
    matteFromLogits(logits, graph, graph, image.width, image.height, procedure);
  return {
    "float-bilinear": apply("float-bilinear"),
    "u8-pil-bicubic": apply("u8-pil-bicubic"),
  };
};

/** `work/` の logit のファイル名（モデル名ごと）。 */
export const logitsFile = (name: string, model: string): string =>
  `${name}.logits-${model}.safetensors`;

/** `work/` の手順ごとの α のファイル名。 */
const alphaFile = (name: string): string => `${name}.alpha.safetensors`;

/** logit（グラフ出力そのもの `[1, 1, graph, graph]` — テンソル名は出力名）を `work/` に書く。 */
export const writeLogits = async (
  work: string,
  file: string,
  tensorName: string,
  graph: number,
  logits: Float32Array<ArrayBuffer>,
): Promise<void> => {
  const data = new Uint8Array(logits.buffer, logits.byteOffset, logits.byteLength);
  await Deno.writeFile(
    `${work}/${file}`,
    new Uint8Array(
      buildSafetensors([{ name: tensorName, dtype: "F32", shape: [1, 1, graph, graph], data }]),
    ),
  );
};

/** 手順ごとの 8bit α（U8 `[H, W]` — テンソル名は手順の名前）を `work/` に書く。 */
export const writeAlphas = async (
  work: string,
  name: string,
  alphas: Readonly<Record<MatteProcedure, Plane>>,
): Promise<void> => {
  const specs = MATTE_PROCEDURES.map((procedure) => {
    const plane = alphas[procedure];
    return {
      name: procedure,
      dtype: "U8",
      shape: [plane.height, plane.width],
      data: new Uint8Array(plane.data.buffer, plane.data.byteOffset, plane.data.byteLength),
    };
  });
  await Deno.writeFile(`${work}/${alphaFile(name)}`, new Uint8Array(buildSafetensors(specs)));
};

/** RGB8 → `encodePng` の受ける RGBA（α は 255 — 合成済みの不透明な画像）。 */
const pngOf = (image: Rgb): Promise<Uint8Array<ArrayBuffer>> => {
  const rgba = new Uint8Array(image.width * image.height * 4).fill(255);
  for (let index = 0; index < image.width * image.height; index += 1) {
    rgba.set(image.data.subarray(index * 3, index * 3 + 3), index * 4);
  }
  return encodePng(rgba, image.width, image.height);
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

const json = (value: unknown): string => JSON.stringify(value, null, 2) + "\n";

/** {@link writeAbOutputs} の入力。 */
export type AbRun = {
  /** 出力の根（`blind/`・`key/`・`work/` を持つ — `work/` の logit と α は書き込み済み）。 */
  readonly out: string;
  readonly pairs: readonly Pair[];
  /** 擬似正解（画像の名前 → 2048² モデルの σ）。 */
  readonly truths: ReadonlyMap<string, Float64Array<ArrayBuffer>>;
  /** A/B の元のグラフの辺（logit は `graph × graph`）。 */
  readonly graph: number;
  /** A/B の元のモデル名（`work/` の logit のファイル名）。 */
  readonly model: string;
  readonly seed: number;
  readonly source: string;
};

/** 盲検の対・回答用紙・対応表・数値の列・検算の manifest を書く。 */
export const writeAbOutputs = async (run: AbRun): Promise<void> => {
  const { out, pairs, truths, graph, model, seed, source } = run;
  const [blind, key, work] = ["blind", "key", "work"].map((name) => `${out}/${name}`);
  for (const dir of [blind, key, work]) await Deno.mkdir(dir, { recursive: true });
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
    json({
      seed,
      source,
      procedures: { upstream: UPSTREAM, current: CURRENT },
      rule:
        `今の経路（${CURRENT}）を残すのは「今の経路」が 5 対以上かつ「上流」が 0 対のときだけ。` +
        `それ以外は上流（${UPSTREAM}）を採る（ADR 0125）。`,
      pairs: mapping,
    }),
  );
  await Deno.writeTextFile(`${key}/metrics.md`, metricsTable(metrics));
  await Deno.writeTextFile(`${key}/metrics.json`, json(metrics));
  await Deno.writeTextFile(
    `${work}/manifest.json`,
    json({
      graph: { width: graph, height: graph },
      entries: pairs.map(({ subject, image }) => ({
        name: subject.name,
        width: image.width,
        height: image.height,
        logits: logitsFile(subject.name, model),
        alpha: alphaFile(subject.name),
      })),
    }),
  );
};
