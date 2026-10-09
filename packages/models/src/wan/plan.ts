/**
 * Wan の生成要求の門 — 要求を検査して計画にする（`generate` の入口・GPU に触る前の純粋な門）。
 *
 * 経路ごとの違いはプロンプトの引き方だけで（{@link PromptGate} — `./text-stage.ts`）、世代ごとの値
 * （受理集合・フレームの範囲・既定）は世代の記述子（`./descriptor.ts`）から引数で受ける。Wan2.1 / 2.2 の
 * class が共有する（ADR 0121 決定 10）。
 *
 * MUST: 全モジュール副作用ゼロ（import 時実行・グローバル可変状態の禁止 — CLAUDE.md）。
 */

import { ModelInputError } from "../errors.ts";
import { assertAcceptableSeed } from "../request-gates.ts";
import type { WanPipelineConfig } from "./config.ts";
import type { WanGenerationDescriptor } from "./descriptor.ts";
import { isWanI2vFit, preprocessWanI2vImage, type WanI2vImageInput } from "./i2v-preprocess.ts";
import type { WanTi2vGenerateRequest } from "./ti2v-pipeline.ts";
import { WAN_UNIPC_CONFIG, type WanUniPcSchedule, wanUniPcSchedule } from "./scheduler.ts";
import type { WanTextEmbedding } from "./text-embeds.ts";
import { planWanGenerationTiles, wanSpatialCompression } from "./tile-decode.ts";
import type { WanVaeChunkLayout } from "./vae-chunks.ts";
import type { WanVaeTilePlan } from "./vae-tiles.ts";

/**
 * VAE の時間圧縮（最初の chunk は 1 フレーム・以降は 4 フレーム — `vae-chunks.ts` の取り決め）。
 *
 * NOTE: `export` は記述子の整合のテストと gpu-lab が同じ値を引くため（`mod.ts` / サブパス面には
 * 出さない — ADR 0008）。
 */
export const TEMPORAL_COMPRESSION = 4;

/**
 * 初期ノイズの seed の既定（世代に依らない）。寸法とフレーム数の既定は世代の記述子
 * （{@link WanGenerationDescriptor}）、step 数・guidance・shift の既定は manifest の `pipelineConfig`
 * （{@link WanPipelineConfig}）。
 */
const DEFAULT_SEED = 0;

/** 初期ノイズの出所（{@link planWanRequest}）。 */
type InitialNoise =
  | { readonly kind: "seed"; readonly seed: number }
  | { readonly kind: "latents"; readonly data: Float32Array<ArrayBuffer> };

/**
 * 入口の門を通った生成の要求（{@link planWanRequest} の戻り）。
 * `Text` はプロンプト 1 本の形（`"precomputed"` = 資産の埋め込みの行・`"gpu"` = umT5 の id 列）。
 */
export type WanGenerationPlan<Text = WanTextEmbedding> = WanGenerationKnobs & {
  readonly positive: Text;
  /** CFG の uncond 側（`guidance` が 1 なら undefined — uncond を回さない）。 */
  readonly negative: Text | undefined;
};

/** 計画のうちプロンプト以外（DiT と VAE の段が使う — 経路に依らない）。 */
export type WanGenerationKnobs = {
  readonly steps: number;
  readonly guidance: number;
  readonly shift: number;
  readonly frames: number;
  readonly width: number;
  readonly height: number;
  /**
   * 潜在の形 `[C, F', H/s, W/s]`（C は VAE のグラフ宣言の潜在のチャネル数・s は空間の圧縮
   * `wanSpatialCompression` — Wan2.1 は `[16, F', H/8, W/8]`）。
   */
  readonly latentShape: readonly [number, number, number, number];
  readonly initial: InitialNoise;
  /** `steps` × `shift` の UniPC の σ 列と timestep 列（denoise はこれを使い、組み直さない）。 */
  readonly schedule: WanUniPcSchedule;
  /** VAE のタイル計画（潜在 `H/s × W/s` — denoise の前に立てる）。 */
  readonly tiles: WanVaeTilePlan;
  /**
   * I2V の条件画像を前処理した VAE encoder の入力（`image` を渡した要求だけ — 出力寸法は `width` / `height` と同じ）。
   * undefined ならテキストだけの要求（encoder の段を回さない・DiT は条件づけない）。
   */
  readonly conditionImage: WanI2vImageInput | undefined;
};

/** 計画に要る VAE の chunk グラフの幾何（資産の宣言から — `wanVaeChunkLayout`）。 */
export type PlanLayout = Pick<
  WanVaeChunkLayout,
  "latentChannels" | "tile" | "sampleTile" | "sampleChannels"
>;

/** 経路ごとのプロンプトの門（{@link planWanRequest} が 1 本の検査の順で呼ぶ）。 */
export type PromptGate<Text> = {
  /** 渡された文字列（型は検査済み）を受理集合で引く / 符号化する。拒否は `ModelInputError`。 */
  readonly resolve: (text: string, what: "prompt" | "negativePrompt") => Text;
  /** `negativePrompt` を省いたときの既定（`guidance` > 1 のときだけ呼ぶ）。 */
  readonly defaultNegative: () => Text;
};

/**
 * 門が読む要求の欄（観測席と中断は段が読む — `WanGenerateRequest`）。I2V の欄（`image` / `fit`）は Wan2.2 の要求の型
 * （{@link WanTi2vGenerateRequest}）にしか無いが、門は世代に依らず 1 本で読む — Wan2.1 の class に JS の呼び手が
 * 型をすり抜けて渡した `image` も、ここで拒む。
 */
type WanPlanRequest = Pick<
  WanTi2vGenerateRequest,
  | "prompt"
  | "negativePrompt"
  | "seed"
  | "latents"
  | "steps"
  | "guidance"
  | "shift"
  | "frames"
  | "width"
  | "height"
  | "image"
  | "fit"
>;

/**
 * 条件画像が RGB8 の形（`data` が `Uint8Array` の object）であること。JS の呼び手が型をすり抜けて渡した値
 * （`null`・数・`data` の無い object）を `ModelInputError` にする — 要求の欄の型の誤りは入力起因（プロンプトの
 * `typeof` の門と同じ扱い）。寸法と長さの食い違いは前処理の `assertRgb8` が見る。
 *
 * MUST: `data` の型まで見る — `Float32Array` などの別の型の配列は長さの検査を通り、画素を uint8 と読み違えたまま
 * 黙って進む。
 */
const assertRgb8Shaped = (image: unknown): void => {
  if (
    typeof image === "object" && image !== null && "data" in image &&
    image.data instanceof Uint8Array
  ) {
    return;
  }
  const actual = image === null
    ? "null"
    : typeof image !== "object"
    ? typeof image
    : "data" in image
    ? `data が ${Object.prototype.toString.call(image.data).slice("[object ".length, -1)}`
    : "data が無い";
  throw new ModelInputError(
    `image が RGB8 の画像（data が Uint8Array・width・height の object）でない（実際: ${actual}）`,
  );
};

/**
 * 生成の要求を検査して計画にする。2 つの経路の入口が共有する検査の本体で、プロンプトの引き方だけを
 * `gate` から受ける。省いた step 数・guidance・shift は `config`（manifest の `pipelineConfig`）の
 * 既定、寸法とフレーム数は世代の記述子（`generation`）の既定で埋める。
 *
 * MUST: 入力起因の失敗（集合の外のプロンプト・受理集合の外の寸法・値域外のノブ・σ 列が壊れる
 * steps × shift の組・条件画像を受けない世代への `image` / `fit`・`image` 無しの `fit`・RGB8 の形でない値や寸法と
 * 長さの合わない RGB8）は全部ここで `ModelInputError` にする。DiT の重みを上げてから落ちる形にしない。
 * 既定だけの組は manifest の門（`parseWanPipelineConfig`）が σ 列まで見ているので、ここで σ 列が
 * 壊れるのは要求が値を渡したときだけ（= 入力起因）。`fit` の綴り違いは素の `Error`（名前の綴り違い — ADR 0107
 * 決定 3。構築オプションの `textEncoder` と同じ扱い）で、`image` の有無より先に見る（欄そのものの値の検査を欄の
 * 組合せの検査より先に — `image` 無しの `fit: "cover"` も綴り違いとして落ちる）。
 *
 * I2V（`image` を渡した要求 — 入力の形 `"ti2v"` の世代だけ）は、出力寸法を画像の縦横比と明示の `width` / `height` から
 * 受理集合の中で選び、前処理（寸法合わせ → `[-1, 1]` → patchify — `i2v-preprocess.ts`）までここで済ませる（GPU に
 * 触る前に入力の誤りを全部落とす）。`image` を渡さない要求は、画像の欄が無かったときと同じ計画になる。
 *
 * MUST: VAE のタイル計画もここで立てる — denoise の後に立てると、資産だけで判る不整合が全 step を
 * 払った後に落ちる。受理する寸法は家族 admission（`tile-decode.ts` の `assertWanVaeTilesCover`）が
 * 全数を通しているので、ここで落ちるのは admission を経ない呼び出しだけ（資産の齟齬 — 素の `Error`）。
 */
export const planWanRequest = <Text>(
  request: WanPlanRequest,
  gate: PromptGate<Text>,
  layout: PlanLayout,
  config: WanPipelineConfig,
  generation: WanGenerationDescriptor,
): WanGenerationPlan<Text> => {
  const resolve = (text: unknown, what: "prompt" | "negativePrompt"): Text => {
    if (typeof text !== "string") throw new ModelInputError(`${what} が文字列でない`);
    return gate.resolve(text, what);
  };
  const positive = resolve(request.prompt, "prompt");

  const steps = request.steps ?? config.defaults.steps;
  if (!Number.isInteger(steps) || steps < 1) {
    throw new ModelInputError(`steps ${steps} が 1 以上の整数でない`);
  }
  const guidance = request.guidance ?? config.defaults.guidance;
  // f32 で見る: CFG は `f32(guidance)` で掛ける（`wanClassifierFreeGuidance`）ので、f64 で有限でも
  // f32 で Infinity になる値（`Number.MAX_VALUE` など）は合成を NaN にする。
  if (!Number.isFinite(Math.fround(guidance)) || guidance < 1) {
    throw new ModelInputError(
      `guidance ${guidance} が 1 以上で f32 に収まる有限の数でない` +
        "（上流は 1 以下で CFG を回さないので、1 未満は効かない・CFG は f32 で掛ける）",
    );
  }
  const shift = request.shift ?? config.scheduler.shift;
  if (!Number.isFinite(shift) || shift <= 0) {
    throw new ModelInputError(`shift ${shift} が正の有限の数でない`);
  }
  let schedule: WanUniPcSchedule;
  try {
    schedule = wanUniPcSchedule(steps, shift, WAN_UNIPC_CONFIG.numTrainTimesteps);
  } catch (error) {
    // 単項の門を通った steps / shift の組で σ 列が壊れる（低水準の RangeError — scheduler.ts の
    // MUST）。既定の組は manifest の門が通しているので、ここに来るのは要求が渡した値のとき。
    if (!(error instanceof RangeError)) throw error;
    throw new ModelInputError(
      `steps ${steps} と shift ${shift} の組では UniPC の σ 列が組めない（${error.message}）`,
      { cause: error },
    );
  }

  // CFG の uncond 側（上流の `do_classifier_free_guidance = guidance_scale > 1`）。
  let negative: Text | undefined;
  if (guidance > 1) {
    negative = request.negativePrompt !== undefined
      ? resolve(request.negativePrompt, "negativePrompt")
      : gate.defaultNegative();
  } else if (request.negativePrompt !== undefined) {
    throw new ModelInputError(
      "guidance 1 では uncond 側を回さないので negativePrompt は効かない（効かせるなら guidance を 1 より大きくする）",
    );
  }

  const frames = request.frames ?? generation.defaults.frames;
  if (
    !Number.isInteger(frames) || frames < generation.minFrames ||
    frames > generation.maxFrames || (frames - 1) % TEMPORAL_COMPRESSION !== 0
  ) {
    throw new ModelInputError(
      `frames ${frames} が受理集合（4n+1 の ${generation.minFrames}〜${generation.maxFrames}）に無い`,
    );
  }
  // I2V の欄の門（ADR 0121 決定 10・11）。条件画像を受けるのは条件入力を持つ DiT（入力の形 "ti2v"）の世代だけ。
  // 値が undefined の欄は欄が無いのと同じに読む（2.2 の T2V と同じ読み方 — 画像を運ばないので、2.1 で I2V が回ると
  // いう誤った期待を作らない）。
  if (
    generation.ditInputForm !== "ti2v" && (request.image !== undefined || request.fit !== undefined)
  ) {
    throw new ModelInputError(
      "image / fit（画像からの生成）はこの世代では受けない（受けるのは Wan2.2 の WanTi2vPipeline）",
    );
  }
  if (request.fit !== undefined && !isWanI2vFit(request.fit)) {
    throw new Error(`fit '${String(request.fit)}' は 'crop' / 'stretch' のどちらでもない`);
  }
  if (request.image === undefined && request.fit !== undefined) {
    throw new ModelInputError(
      "fit は image を渡したときだけ効く（条件画像の寸法の合わせ方 — image 無しでは渡さない）",
    );
  }
  let width: number;
  let height: number;
  let conditionImage: WanI2vImageInput | undefined;
  if (request.image === undefined) {
    width = request.width ?? generation.defaults.width;
    height = request.height ?? generation.defaults.height;
    if (
      !generation.acceptedSizes.some((size) => size.width === width && size.height === height)
    ) {
      throw new ModelInputError(
        `${width}×${height} が受理集合（${
          generation.acceptedSizes.map((size) => `${size.width}×${size.height}`).join(" / ")
        }）に無い`,
      );
    }
  } else {
    assertRgb8Shaped(request.image);
    // 出力寸法の選択（省いた欄は画像の縦横比で — 明示した欄は受理集合の中）と前処理。
    conditionImage = preprocessWanI2vImage(
      request.image,
      { width: request.width, height: request.height, fit: request.fit },
      generation,
    );
    width = conditionImage.width;
    height = conditionImage.height;
  }
  const compression = wanSpatialCompression(layout, generation);
  const latentShape: [number, number, number, number] = [
    layout.latentChannels,
    (frames - 1) / TEMPORAL_COMPRESSION + 1,
    height / compression,
    width / compression,
  ];
  if (!latentShape.every(Number.isInteger)) {
    // 受理集合は空間の圧縮で割り切れる寸法だけ — 割れるなら資産の取り違え（入力起因ではない）。
    throw new Error(
      `潜在の形 [${latentShape}] が整数でない（空間の圧縮 ${compression}）`,
    );
  }

  if (request.seed !== undefined && request.latents !== undefined) {
    throw new ModelInputError("seed と latents は排他（初期ノイズの出所はどちらか 1 つ）");
  }
  let initial: InitialNoise;
  if (request.latents !== undefined) {
    const { latents } = request;
    const count = latentShape.reduce((product, dim) => product * dim, 1);
    if (!(latents instanceof Float32Array) || latents.length !== count) {
      throw new ModelInputError(
        `latents が Float32Array [${latentShape.join(", ")}]（${count} 要素）でない`,
      );
    }
    if (!latents.every(Number.isFinite)) throw new ModelInputError("latents に非有限値がある");
    initial = { kind: "latents", data: latents };
  } else {
    const seed = request.seed ?? DEFAULT_SEED;
    assertAcceptableSeed(seed);
    initial = { kind: "seed", seed };
  }

  // 重なりの式（64 px ÷ 空間の圧縮）は潜在の整数の検査の後で求める（`assertWanVaeTilesCover` と同じ順）。
  const tiles = planWanGenerationTiles(layout, latentShape[2], latentShape[3], generation);
  return {
    positive,
    negative,
    steps,
    guidance,
    shift,
    frames,
    width,
    height,
    latentShape,
    initial,
    schedule,
    tiles,
    conditionImage,
  };
};
