/**
 * GPU lab の Wan のタブ（ADR 0118 段 9 — Chrome で Wan2.1 を回す・ADR 0121 段 8 — 世代の選択で Wan2.2 TI2V 5B
 * も回す）の GPU に依らない部分。
 *
 * ここに置くのは純関数と型と世代の仕様だけ（DOM も GPU も触らない — `wan-plan_test.ts` が deno test で縛る）:
 *
 * - **世代の仕様**（{@link WanLabGeneration} — {@link WAN21_LAB} / {@link WAN22_LAB}）: 世代で違う値（記述子・DiT の
 *   寸法・1 token の画素数・VAE の最大の値・参照ケースの表・配布形の経路名）を 1 つに畳む。以下の関数は全て
 *   これを第 1 引数に取る。
 * - **limits の判定表**（{@link judgeWanLimits}）: アダプタの limits と、要求（フレーム数・寸法）が要る
 *   値を並べる。束縛上限の下限は「行ブロックで割れない最大の値」で決まる — 自己 attention のスコア S は
 *   runtime が束縛上限に収まる枚数へ等分する（`planRowBlocks` — ADR 0060）ので、上限が小さければ枚数が
 *   増えるだけで、下限は 1 行ぶんになる。
 * - **要求の組み立て**（{@link buildWanRequest}）: フォームの文字列 → `generate` の要求と、既定を埋めた
 *   解決済みの値（参照ケースの照合と記録に使う）。
 * - **所要の集計**（{@link summarizeWanTimeline}）: 生成イベントの時刻 → 段・step・VAE タイルごとの時間。
 * - **sha256 の実物**（{@link wanRgbBytes}）と参照ケースの照合（{@link wanReferenceCaseId} /
 *   {@link checkWanReference}）。
 */
import type { SessionDiagnostics } from "../../../packages/runtime/mod.ts";
import { REQUIRED_LIMIT_KEYS } from "../../../packages/runtime/src/gpu/acquire.ts";
import { MAX_SINGLE_CONTAINER_BYTES } from "../../../packages/runtime/src/format/container/limits.ts";
import { planRowBlocks } from "../../../packages/runtime/src/runtime/fusion.ts";
import type { WanGenerateRequest, WanPrompt } from "../../../packages/models/wan.ts";
import type { WanPipelineConfig } from "../../../packages/models/src/wan/config.ts";
import {
  WAN21_GENERATION,
  WAN22_TI2V_GENERATION,
  type WanGenerationDescriptor,
} from "../../../packages/models/src/wan/descriptor.ts";
import { TEMPORAL_COMPRESSION } from "../../../packages/models/src/wan/plan.ts";

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;
const F32_BYTES = 4;

/** cross-attn のキーの行数（テキスト埋め込みを詰める文脈 `encoder_hidden_states [1,512,4096]` — 2 世代とも同じ）。 */
export const WAN_TEXT_CONTEXT_ROWS = 512;

/** WebGPU の既定（requiredLimits で要求しなければ device はこの値になる）。 */
export const WEBGPU_DEFAULT_STORAGE_BINDING = 128 * MIB;
export const WEBGPU_DEFAULT_BUFFER = 256 * MIB;

/**
 * Chromium の単一 ArrayBuffer の上限（2³¹ − 2 MiB — docs/limitations.md「ブラウザ: Chromium は単一
 * ArrayBuffer を…」節）。runtime の単一形の容器の上限と同じ値を共有する。
 */
export const CHROMIUM_ARRAY_BUFFER_MAX = MAX_SINGLE_CONTAINER_BYTES;

export type WanSize = { readonly width: number; readonly height: number };

/** 世代の識別子（記録の JSON の `generation` に載る綴り）。 */
export type WanLabGenerationId = "wan2.1" | "wan2.2";

/**
 * 参照ケースの 1 組（e2e の sha 行のケース — 条件が揃った要求だけがその id を名乗る）。プロンプト・seed・
 * negative・guidance・shift の共通条件は {@link wanReferenceCaseId} が見る。
 */
export type WanReferenceCase = {
  readonly steps: number;
  readonly frames: number;
  readonly sizes: readonly WanSize[];
  /** e2e が行を持つ席（ここに無い席の組は参照ケースではない）。 */
  readonly quants: readonly string[];
  /** sha 行のケース id（e2e の綴りの写し）。 */
  readonly id: (quant: string, size: WanSize, shift: number) => string;
};

/** 世代の仕様（タブの世代の選択 1 つぶん — 世代で違う値はここにしか書かない）。 */
export type WanLabGeneration = {
  readonly id: WanLabGenerationId;
  /** 世代の選択の表示（`Wan2.1 T2V 1.3B（karume-wan2.1）`）。 */
  readonly label: string;
  /** このサーバの配布形の経路名（`server.ts` の `/models/<名前>/`）と、置き場を指定するオプション。 */
  readonly route: "wan" | "wan22";
  readonly serverOption: "--wan-source" | "--wan22-source";
  /**
   * 世代の記述子（フレーム数の選択肢・判定表の受理の門・既定の寸法とフレーム数・fps）。製品の記述子そのもの
   * （gpu-lab は製品の受理の外を選ばせない）。
   */
  readonly descriptor: WanGenerationDescriptor;
  /** DiT の head 数と FFN の中間の幅（スコアの行ブロックと FFN 中間の大きさ）。 */
  readonly ditHeads: number;
  readonly ditFfnWidth: number;
  /** 1 token の画素の辺（VAE の空間比 × unpatchify × DiT のパッチ）。 */
  readonly pixelsPerToken: number;
  /**
   * VAE の段で最大の値（f32 のバイト数と、どの値か）。タイルで固定なのでフレーム数と寸法に依らない。
   *
   * NOTE: タイル辺は配布物だけで差し替えられる（ADR 0118 決定 2）。差し替えたら読み直す値。
   */
  readonly vaeLargestValue: { readonly bytes: number; readonly what: string };
  /** 参照ケースの表（`fixtures/references/` の sha 行のケース）。 */
  readonly referenceCases: readonly WanReferenceCase[];
};

/**
 * 席名を持たない参照ケースの id の席（Wan2.1 の `e2e_wan_pipeline_test.ts` の `F16_QUANT` — 既定席が実用席へ
 * 移っても既存の行はこの席の値・ADR 0120 裁定 2026-10-04 の 4）。
 */
const WAN21_UNSEATED_QUANT = "f16";

/** Wan2.1 の参照ケースの id（`f16` 席は席名を持たない id・i8 の席は席名を先頭に置いた id — e2e の `seatCaseId`）。 */
const wan21CaseId = (base: string) => (quant: string): string =>
  quant === WAN21_UNSEATED_QUANT ? base : `${quant}-${base}`;

/**
 * Wan2.1 T2V 1.3B の仕様。
 *
 * DiT の寸法は ADR 0118 Context「調査の結論」（30 層・dim 1536・12 heads × 128・FFN 8960）。上流の 1.3B の config の
 * 値で、配布形のグラフ宣言（`gelu_* [1,S,8960]`）とも一致する（2026-10-03 に読んだ）。1 token の画素の辺は 16
 * （VAE の空間 8 × patch 2）。
 *
 * VAE の最大の値は `vae_decoder_next` の `expand_5 [768,128,2,256]` / `view_17 [4,192,256,256]` の f32（2026-10-03 に
 * 配布形のグラフ宣言から読んだ値）。first のグラフの最大（75,497,472 B）はこれより小さい。
 *
 * 参照ケースは `e2e_wan_pipeline_test.ts` の `SEED_CASE`〈2 ステップ〉と `FULL_CASES`〈50 ステップ・33 / 81
 * フレーム〉（832×480）。席は `f16` と、e2e の `SEAT_QUANTS`〈2 ステップ〉と `FULL_CASES` の実用席〈50 ステップ〉の写し。
 *
 * MUST: e2e のケースの定義を変えたらここも変える（行の値が別の条件の sha と突き合わさる）。
 */
export const WAN21_LAB: WanLabGeneration = {
  id: "wan2.1",
  label: "Wan2.1 T2V 1.3B（karume-wan2.1）",
  route: "wan",
  serverOption: "--wan-source",
  descriptor: WAN21_GENERATION,
  ditHeads: 12,
  ditFfnWidth: 8960,
  pixelsPerToken: 16,
  vaeLargestValue: { bytes: 201_326_592, what: "VAE（next）の中間 [768,128,2,256] f32" },
  referenceCases: [
    {
      steps: 2,
      frames: 33,
      sizes: [{ width: 832, height: 480 }],
      quants: [WAN21_UNSEATED_QUANT, "f16+dit8", "f16+dit8-a8-attn8-s16"],
      id: wan21CaseId("2step-seed-boxing-cats-seed42"),
    },
    {
      steps: 50,
      frames: 33,
      sizes: [{ width: 832, height: 480 }],
      quants: [WAN21_UNSEATED_QUANT, "f16+dit8-a8-attn8-s16"],
      id: wan21CaseId("50step-boxing-cats-seed42"),
    },
    {
      steps: 50,
      frames: 81,
      sizes: [{ width: 832, height: 480 }],
      quants: [WAN21_UNSEATED_QUANT, "f16+dit8-a8-attn8-s16"],
      id: wan21CaseId("50step-boxing-cats-seed42-81f"),
    },
  ],
};

/** Wan2.2 の参照ケースの id（`e2e_wan_ti2v_pipeline_test.ts` の `caseIdOf` の綴り）。 */
const wan22CaseId =
  (head: string, frames: number) => (quant: string, size: WanSize, shift: number): string =>
    `${quant}-${head}-${size.width}x${size.height}-${frames}f-shift${shift}`;

/**
 * Wan2.2 TI2V 5B の仕様。
 *
 * DiT の寸法は ADR 0121 Context「調査の結論」（dim 3072・24 heads × 128・FFN 14336）。FFN の幅は i8 の系列
 * （`outputs/series/wan2.2-ti2v-5b-i8-dyn/transformer/`）の graph 文書の FFN 中間 `[1,S,14336]`（60 本）とも一致する
 * （2026-10-06 に読んだ — 他の活性は `[1,S,3072]` / `[1,S,24,128]` 以下）。1 token の画素の辺は 32（VAE のグラフの
 * 空間比 8 × unpatchify 2 × DiT のパッチ 2 — S = 1280×704×33 で 7,920・121 フレームで 27,280）。
 *
 * VAE の最大の値は `vae_decoder_next` の `cat_25 [512,6,128,128]` の f32 = 201,326,592 B（2026-10-06 に f16 の系列
 * `outputs/series/wan2.2-ti2v-5b-f16-dyn/` の graph 文書から読んだ値）。どちらのグラフでも活性に限った最大で、重みの
 * 値はこれより小さい（宣言の最大は f32 の `decoder.mid_block.resnets.0.conv1.weight [1024,1024,3,3,3]` の 113,246,208 B・
 * 格納は f16）。first のグラフの活性の最大は `cat_23 [512,3,128,128]` の 100,663,296 B。ADR 0121 追記「段 4 の結果」の前の調査の約 207.7 MB（`[1,512,6,130,130]` —
 * 上流の詰め物つきの形）ではなく、配布する宣言の値を使う。
 *
 * I2V の VAE encoder の段（3 グラフ `vae_encoder_pre` / `attn` / `post` — ADR 0121 段 9）は判定に入れない: 宣言の最大は
 * `vae_encoder_pre` の `constant_pad_nd_1 [1,160,641,353]` の f32 = 144,814,720 B（1280×704 でも 704×1280 でも同じ値・
 * attn は `[3520,1920]` の 27,033,600 B・post は 14,745,600 B — 2026-10-10 に f16 の系列の graph 文書から読んだ値）で、
 * encoder は出力寸法〈受理集合の 2 寸法〉でしか回らないので、上の VAE の最大の値を越えない。
 *
 * 記述子は製品の記述子そのもの（受理の上限は公式の既定の 121 フレーム。DiT のグラフの S の記号の上限は 27,280〈1280×704×121
 * — ADR 0121 決定 3〉なので、この上限まではグラフの宣言の内）。
 *
 * 参照ケースは `e2e_wan_ti2v_pipeline_test.ts` の `SEED_CASES`（参照席・2 ステップ・17 フレーム・2 寸法）と
 * `FULL_CASES`（参照席と実用席・50 ステップ・33 フレーム・1280×704）と `LONG_CLIP_CASE_ID`（参照席・2 ステップ・
 * 121 フレーム・1280×704 — e2e では opt-in）。席名は e2e の helper
 * （`tests/helpers/wan-ti2v-pipeline.ts` の `WAN_TI2V_REFERENCE_QUANT` / `WAN_TI2V_PRACTICAL_QUANT`）の写し —
 * 2.1 の表と同じく文字列で持つ（helper は Deno の API を読むモジュールを引くので、ブラウザの bundle に入れない）。
 *
 * MUST: e2e のケースの定義を変えたらここも変える（行の値が別の条件の sha と突き合わさる）。
 */
export const WAN22_LAB: WanLabGeneration = {
  id: "wan2.2",
  label: "Wan2.2 TI2V 5B（karume-wan2.2）",
  route: "wan22",
  serverOption: "--wan22-source",
  descriptor: WAN22_TI2V_GENERATION,
  ditHeads: 24,
  ditFfnWidth: 14336,
  pixelsPerToken: 32,
  vaeLargestValue: { bytes: 201_326_592, what: "VAE（next）の中間 [512,6,128,128] f32" },
  referenceCases: [
    {
      steps: 2,
      frames: 17,
      sizes: [{ width: 1280, height: 704 }, { width: 704, height: 1280 }],
      quants: ["f16+dit8"],
      id: wan22CaseId("2step-boxing-cats-seed42", 17),
    },
    {
      steps: 50,
      frames: 33,
      sizes: [{ width: 1280, height: 704 }],
      quants: ["f16+dit8", "f16+dit8-a8-attn8-s16"],
      id: wan22CaseId("50step-boxing-cats-seed42", 33),
    },
    {
      steps: 2,
      frames: 121,
      sizes: [{ width: 1280, height: 704 }],
      quants: ["f16+dit8"],
      id: wan22CaseId("2step-boxing-cats-seed42", 121),
    },
  ],
};

/** 世代の選択肢（先頭がタブの既定 — Wan2.1 のタブの既定の挙動を変えない）。 */
export const WAN_LAB_GENERATIONS: readonly WanLabGeneration[] = [WAN21_LAB, WAN22_LAB];

/** 受理集合のフレーム数（4n+1 の下限〜上限 — 世代の仕様の記述子を正本にする）。 */
export const wanFrameChoices = (generation: WanLabGeneration): number[] => {
  const { minFrames, maxFrames } = generation.descriptor;
  const choices: number[] = [];
  for (let frames = minFrames; frames <= maxFrames; frames += TEMPORAL_COMPRESSION) {
    choices.push(frames);
  }
  return choices;
};

/** `832x480` → 寸法（綴りが違えば落とす — 黙って既定の寸法で回さない）。 */
export const parseWanSize = (text: string): WanSize => {
  const match = /^(\d+)x(\d+)$/.exec(text);
  if (match === null) throw Error(`寸法 ${text} が WxH の形でない`);
  return { width: Number(match[1]), height: Number(match[2]) };
};

export const wanSizeLabel = (size: WanSize): string => `${size.width}x${size.height}`;

/**
 * 寸法の選択の「画像から自動」の値（I2V — 要求に `width` / `height` を載せず、パイプラインが条件画像の縦横比で
 * 受理集合から選ぶ）。
 */
export const WAN_SIZE_FROM_IMAGE = "from-image";

/**
 * 寸法の選択の値 → 寸法。{@link WAN_SIZE_FROM_IMAGE} なら `imageSize`（呼び手が条件画像から製品の規則
 * `selectWanI2vSize` で求めた寸法 — パイプラインが選ぶのと同じ値）。画像が無いのに「画像から自動」なら落とす。
 */
export const resolveWanFormSize = (text: string, imageSize: WanSize | undefined): WanSize => {
  if (text !== WAN_SIZE_FROM_IMAGE) return parseWanSize(text);
  if (imageSize === undefined) {
    throw Error("寸法「画像から自動」には条件画像が要る（画像を選んでいない）");
  }
  return imageSize;
};

/**
 * DiT のトークン数 S（潜在フレーム数 × H/p × W/p — p は世代の 1 token の画素の辺）。世代の仕様の受理集合
 * （2.2 は 4n+1 の 5〜121 フレーム — 製品の受理集合そのまま）の外は落とす — 判定表は受理集合の中の要求についてだけ意味を持つ
 * （外の要求は `generate` が `ModelInputError` で拒む）。
 */
export const wanTokenCount = (
  generation: WanLabGeneration,
  frames: number,
  size: WanSize,
): number => {
  const { minFrames, maxFrames, acceptedSizes } = generation.descriptor;
  if (
    !Number.isInteger(frames) || frames < minFrames || frames > maxFrames ||
    (frames - 1) % TEMPORAL_COMPRESSION !== 0
  ) {
    throw new RangeError(
      `フレーム数 ${frames} が受理集合（4n+1 の ${minFrames}〜${maxFrames}）に無い`,
    );
  }
  if (!acceptedSizes.some(({ width, height }) => width === size.width && height === size.height)) {
    throw new RangeError(`寸法 ${wanSizeLabel(size)} が受理集合に無い`);
  }
  const latentFrames = (frames - 1) / TEMPORAL_COMPRESSION + 1;
  const { pixelsPerToken } = generation;
  return latentFrames * (size.height / pixelsPerToken) * (size.width / pixelsPerToken);
};

/** 行ブロックで割れない最大の値（束縛上限と maxBufferSize の下限 — {@link judgeWanLimits}）。 */
export type WanLargestValue = {
  readonly bytes: number;
  readonly what: string;
};

/**
 * 要求 1 本で最大の値。DiT の FFN 中間 `[1,S,FFN]` の f32 と VAE の最大の値の大きい方。Wan2.1 は
 * `[1,S,8960]`（S = 14,040 で 503,193,600 B・32,760 で 1,174,118,400 B — ADR 0118 段 8 の「FFN 中間 480 MiB /
 * 1.09 GiB」）で DiT の他の値は `[1,S,1536]` 以下、Wan2.2 は `[1,S,14336]`（S = 27,280 で 1,564,344,320 B — ADR 0121
 * 決定 8 の「FFN 中間 最大 1.46 GiB」）で他の値は `[1,S,3072]` 以下。どちらも FFN 中間が DiT の最大。
 *
 * NOTE: 自己 attention のスコア S は値として持たない（attention の 1 op の内側で行ブロックに割る）ので
 * ここには入らない — 下限への効き方は {@link judgeWanLimits} の行ブロックの欄。
 */
export const wanLargestValue = (
  generation: WanLabGeneration,
  frames: number,
  size: WanSize,
): WanLargestValue => {
  const tokens = wanTokenCount(generation, frames, size);
  const { ditFfnWidth, vaeLargestValue } = generation;
  const ffn = tokens * ditFfnWidth * F32_BYTES;
  return ffn >= vaeLargestValue.bytes
    ? { bytes: ffn, what: `DiT の FFN 中間 [1,${tokens},${ditFfnWidth}] f32` }
    : vaeLargestValue;
};

/** 自己 attention / cross-attn のスコアの行ブロック（runtime と同じ `planRowBlocks` で数える）。 */
export type WanRowBlocks = {
  /** 1 行（クエリ 1 行 × 全 head）のバイト数。束縛上限がこれ未満なら割れない。 */
  readonly bytesPerRow: number;
  /** 枚数（1 行も入らなければ undefined）。 */
  readonly count?: number;
  /** 1 枚の最大のバイト数（1 行も入らなければ undefined）。 */
  readonly blockBytes?: number;
};

const rowBlocks = (rows: number, bytesPerRow: number, limit: number): WanRowBlocks => {
  if (bytesPerRow > limit) return { bytesPerRow };
  const blocks = planRowBlocks(rows, bytesPerRow, limit);
  const widest = Math.max(...blocks.map((block) => block.rows));
  return { bytesPerRow, count: blocks.length, blockBytes: widest * bytesPerRow };
};

/**
 * スコアの行ブロック（自己 attention は S × S・cross-attn は S × 512 — どちらも f32 格納・H は世代の head 数
 * 〈2.1 は 12・2.2 は 24〉。実用席の `attentionScoreStorage: "f16"` で半分になる側は見ない — 保守側の f32 で数える・
 * ADR 0118 決定 6）。
 */
export const wanAttentionRowBlocks = (
  generation: WanLabGeneration,
  frames: number,
  size: WanSize,
  maxStorageBufferBindingSize: number,
): { readonly self: WanRowBlocks; readonly cross: WanRowBlocks } => {
  const tokens = wanTokenCount(generation, frames, size);
  const { ditHeads } = generation;
  return {
    self: rowBlocks(tokens, ditHeads * tokens * F32_BYTES, maxStorageBufferBindingSize),
    cross: rowBlocks(
      tokens,
      ditHeads * WAN_TEXT_CONTEXT_ROWS * F32_BYTES,
      maxStorageBufferBindingSize,
    ),
  };
};

export type WanLimitKey = (typeof REQUIRED_LIMIT_KEYS)[number];
export type WanLimits = Readonly<Pick<GPUSupportedLimits, WanLimitKey>>;

/** 判定表の 1 行。`info` は Wan 固有の下限を導いていない項目（値を見せるだけ）。 */
export type WanLimitRow = {
  readonly key: WanLimitKey;
  readonly value: number;
  readonly required?: number;
  readonly verdict: "ok" | "short" | "info";
  readonly note: string;
};

/** バイト数の表示（`503,193,600 B（479.9 MiB）`）。 */
export const formatBytes = (bytes: number): string =>
  `${bytes.toLocaleString("en-US")} B（${
    bytes >= GIB ? `${(bytes / GIB).toFixed(2)} GiB` : `${(bytes / MIB).toFixed(1)} MiB`
  }）`;

const rowBlocksNote = (name: string, blocks: WanRowBlocks): string =>
  blocks.count === undefined
    ? `${name}のスコア 1 行 ${formatBytes(blocks.bytesPerRow)} が入らない（行ブロックでも割れない）`
    : `${name}のスコアは行ブロック ${blocks.count} 枚（1 枚 ≤ ${
      formatBytes(blocks.blockBytes ?? 0)
    }）`;

/**
 * アダプタ（か取得した device）の limits を、要求（フレーム数・寸法）が要る値と並べる。
 *
 * 判定するのは 2 つだけ: `maxStorageBufferBindingSize` と `maxBufferSize` が行ブロックで割れない最大の値
 * （{@link wanLargestValue}）以上か。これは**必要条件**で、十分条件ではない（VRAM の総量・submit の時間・
 * カーネルの workgroup の形は別の門 — runtime が Session の構築と実行で fail loudly にする）。他の
 * `REQUIRED_LIMIT_KEYS` は `acquireGpu` がアダプタ値をそのまま要求する項目で、Wan 固有の下限は導いて
 * いないので `info`。
 */
export const judgeWanLimits = (
  generation: WanLabGeneration,
  limits: WanLimits,
  frames: number,
  size: WanSize,
): WanLimitRow[] => {
  const largest = wanLargestValue(generation, frames, size);
  const binding = limits.maxStorageBufferBindingSize;
  const attention = wanAttentionRowBlocks(generation, frames, size, binding);
  const bindingOk = binding >= largest.bytes && attention.self.count !== undefined &&
    attention.cross.count !== undefined;
  return REQUIRED_LIMIT_KEYS.map((key): WanLimitRow => {
    const value = limits[key];
    if (key === "maxStorageBufferBindingSize") {
      return {
        key,
        value,
        required: largest.bytes,
        verdict: bindingOk ? "ok" : "short",
        note: `最大の値 = ${largest.what} · ${rowBlocksNote("自己 attention ", attention.self)} · ${
          rowBlocksNote("cross-attn ", attention.cross)
        }`,
      };
    }
    if (key === "maxBufferSize") {
      return {
        key,
        value,
        required: largest.bytes,
        verdict: value >= largest.bytes ? "ok" : "short",
        note: "1 本の値を 1 つのバッファに置くための下限",
      };
    }
    return { key, value, verdict: "info", note: "acquireGpu がアダプタ値をそのまま要求する" };
  });
};

/** この limits で判定表が全て通る最大のフレーム数（1 本も通らなければ undefined）。 */
export const wanMaxFramesWithin = (
  generation: WanLabGeneration,
  limits: WanLimits,
  size: WanSize,
): number | undefined =>
  wanFrameChoices(generation).filter((frames) =>
    judgeWanLimits(generation, limits, frames, size).every((row) => row.verdict !== "short")
  ).at(-1);

/**
 * フォームの文字列（空欄 = 指定しない）。`prompt` / `negative` は埋め込み資産の名前（`negative` の
 * 空欄 = 資産の negative の行）。
 */
export type WanForm = {
  readonly prompt: string;
  readonly negative: string;
  readonly seed: string;
  readonly frames: string;
  readonly size: string;
  readonly steps: string;
  readonly guidance: string;
  readonly shift: string;
};

/** 既定を埋めた要求（記録と参照ケースの照合に使う — プロンプトは資産の名前）。 */
export type WanResolvedRequest = {
  readonly prompt: string;
  /** CFG の uncond 側の名前（guidance 1 なら undefined — uncond を回さない）。 */
  readonly negative?: string;
  readonly seed: number;
  readonly steps: number;
  readonly guidance: number;
  readonly shift: number;
  readonly frames: number;
  readonly width: number;
  readonly height: number;
};

/**
 * 数の欄（空欄 = undefined）。数として読めない綴りは落とす — `Number("")` の 0 や NaN を黙って
 * パイプラインへ渡さない（範囲の門はパイプラインの `ModelInputError` が持つ）。
 */
const optionalNumber = (text: string, what: string): number | undefined => {
  const trimmed = text.trim();
  if (trimmed === "") return undefined;
  const value = Number(trimmed);
  if (!Number.isFinite(value)) throw Error(`${what} ${text} が数でない`);
  return value;
};

/**
 * フォーム → `generate` の要求（`onEvent` は呼び手が足す）と解決済みの値。
 *
 * 要求には**入力された欄だけ**を載せる（空欄の既定はパイプラインが manifest の `pipelineConfig` から
 * 埋める — 既定の正本を二重に持たない）。解決済みの値は同じ `pipelineConfig`（呼び手が manifest から
 * `parseWanPipelineConfig` で読んだもの）で埋める。範囲の検査はしない（パイプラインの門の写しを持たない）。
 *
 * 寸法が「画像から自動」（{@link WAN_SIZE_FROM_IMAGE}）なら要求に `width` / `height` を載せず、解決済みの寸法は
 * `imageSize`（{@link resolveWanFormSize}）。条件画像そのもの（`image` / `fit`）は呼び手が要求に足す。
 */
export const buildWanRequest = (
  form: WanForm,
  prompts: readonly WanPrompt[],
  config: WanPipelineConfig,
  imageSize?: WanSize,
): {
  readonly request: Omit<WanGenerateRequest, "onEvent">;
  readonly resolved: WanResolvedRequest;
} => {
  const entry = (name: string, what: string): WanPrompt => {
    const found = prompts.find((candidate) => candidate.name === name);
    if (found === undefined) {
      throw Error(
        `${what} '${name}' が埋め込み資産に無い（${prompts.map((p) => p.name).join(" / ")}）`,
      );
    }
    return found;
  };
  const positive = entry(form.prompt, "プロンプト");
  const negative = form.negative === "" ? undefined : entry(form.negative, "ネガティブ");
  const seed = optionalNumber(form.seed, "seed");
  if (seed === undefined) throw Error("seed が空欄（参照ケースと照合できるように明示する）");
  const frames = optionalNumber(form.frames, "フレーム数");
  if (frames === undefined) throw Error("フレーム数が空欄");
  const size = resolveWanFormSize(form.size, imageSize);
  const sizeFromImage = form.size === WAN_SIZE_FROM_IMAGE;
  const steps = optionalNumber(form.steps, "steps");
  const guidance = optionalNumber(form.guidance, "guidance");
  const shift = optionalNumber(form.shift, "shift");
  const resolvedGuidance = guidance ?? config.defaults.guidance;
  // 省いた negative はパイプラインが資産の negative の行（1 本のとき）で埋める — 解決済みの名前も同じ規則
  const defaults = prompts.filter((candidate) => candidate.role === "negative");
  const resolvedNegative = resolvedGuidance <= 1
    ? undefined
    : negative?.name ?? (defaults.length === 1 ? defaults[0].name : undefined);
  return {
    request: {
      prompt: positive.prompt,
      ...(negative === undefined ? {} : { negativePrompt: negative.prompt }),
      seed,
      frames,
      ...(sizeFromImage ? {} : { width: size.width, height: size.height }),
      ...(steps === undefined ? {} : { steps }),
      ...(guidance === undefined ? {} : { guidance }),
      ...(shift === undefined ? {} : { shift }),
    },
    resolved: {
      prompt: positive.name,
      ...(resolvedNegative === undefined ? {} : { negative: resolvedNegative }),
      seed,
      steps: steps ?? config.defaults.steps,
      guidance: resolvedGuidance,
      shift: shift ?? config.scheduler.shift,
      frames,
      width: size.width,
      height: size.height,
    },
  };
};

/**
 * 所要を記録する段（`text_encoder` は呼び手が別に採る）。`vae_encoder` は Wan2.2 の I2V の条件画像の encode の段
 * （`image` を渡した要求だけ・最初の段 — step もタイルも持たない）。
 */
export type WanTimelineStage = "vae_encoder" | "transformer" | "vae_decoder";

/** 生成イベントの時刻（`performance.now()` の ms）。 */
export type WanTimelineMark =
  | {
    readonly kind: "stage";
    readonly component: WanTimelineStage;
    readonly at: "start" | "end";
    readonly ms: number;
  }
  | { readonly kind: "step"; readonly step: number; readonly ms: number }
  | { readonly kind: "tile"; readonly tile: number; readonly ms: number };

/**
 * 段・step・タイルごとの所要（ms）。`stepMs[0]` は DiT の段の開始から 1 step 目の完了まで（Session の
 * 構築 = 重みの転送を含む）、以降は前の step の完了から。`tileMs` も同じ（1 枚目は VAE の 2 Session と
 * cache の構築を含む）。段が途中で落ちたら、その段の `stageMs` は欠ける。
 */
export type WanTimeline = {
  readonly stageMs: Readonly<Partial<Record<WanTimelineStage, number>>>;
  readonly stepMs: readonly number[];
  readonly tileMs: readonly number[];
};

/**
 * イベントの時刻列 → {@link WanTimeline}。並びがパイプラインの取り決め（段の start → step / tile が
 * 1 から順に → 段の end）から外れたら落とす — 黙って時間を別の区間へ帰属させない。
 */
export const summarizeWanTimeline = (marks: readonly WanTimelineMark[]): WanTimeline => {
  const stageMs: Partial<Record<WanTimelineStage, number>> = {};
  const stepMs: number[] = [];
  const tileMs: number[] = [];
  let open: { readonly component: WanTimelineStage; readonly ms: number } | undefined;
  let last = 0;
  for (const mark of marks) {
    if (mark.kind === "stage") {
      if (mark.at === "start") {
        if (open !== undefined) {
          throw Error(`段 ${open.component} が閉じる前に ${mark.component} が始まった`);
        }
        open = { component: mark.component, ms: mark.ms };
        last = mark.ms;
      } else {
        if (open?.component !== mark.component) {
          throw Error(`開いていない段 ${mark.component} の end`);
        }
        stageMs[mark.component] = mark.ms - open.ms;
        open = undefined;
      }
      continue;
    }
    const [expected, list, index] = mark.kind === "step"
      ? ["transformer", stepMs, mark.step] as const
      : ["vae_decoder", tileMs, mark.tile] as const;
    if (open?.component !== expected || index !== list.length + 1) {
      throw Error(
        `${mark.kind} ${index} が段 ${expected} の ${list.length + 1} 番目として来ていない`,
      );
    }
    list.push(mark.ms - last);
    last = mark.ms;
  }
  return { stageMs, stepMs, tileMs };
};

/** 環境キーの基底から落とす商標の飾り（`Intel(R) Graphics` → `intel graphics`）。 */
const TRADEMARKS = ["(r)", "(tm)", "(c)"] as const;

/**
 * Chrome の環境キー（sha256 の環境行のキー — ADR 0106）。規則は
 * `packages/runtime/tests/helpers/environment.ts` の `environmentKey("chrome", info)` の写し: description が
 * 空でなければそれを、空なら `vendor-architecture` を基底にして小文字化・商標の飾りを落とし・英数字以外を
 * `-` に畳む。例: vendor `nvidia` + architecture `blackwell` → `chrome-nvidia-blackwell`。
 *
 * 写しを持つのは、正本のモジュールが import 時に top-level await で環境を確定する（アダプタを取り
 * `Deno.version` を読む）ので、ページの bundle にもホストのテストにも入れられないため。
 * MUST: 正本の規則を変えたらここも変える（キーが割れると、利用者が報告した sha が別の行に入る）。
 *
 * NOTE: Chrome は開発者向けフラグ（`chrome://flags/#enable-webgpu-developer-features`）を有効にすると
 * description を埋めるので、同じ機でもフラグの有無でキーが変わる（無しなら vendor-architecture）。
 */
export const chromeEnvironmentKey = (
  info: Pick<GPUAdapterInfo, "vendor" | "architecture" | "description">,
): string => {
  const base = info.description !== "" ? info.description : `${info.vendor}-${info.architecture}`;
  let slug = base.toLowerCase();
  for (const mark of TRADEMARKS) slug = slug.replaceAll(mark, "");
  slug = slug.replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
  if (slug === "") {
    throw Error(
      "GPUAdapterInfo から環境キーを作れない（description / vendor / architecture が全て空）",
    );
  }
  return `chrome-${slug}`;
};

/** 区間の列の要約（`1 回目 12.3 s · 2 回目以降 中央値 10.1 s / 最大 10.5 s（49 回）`）。 */
export const formatSpans = (spans: readonly number[]): string => {
  if (spans.length === 0) return "—";
  const seconds = (ms: number): string => `${(ms / 1000).toFixed(1)} s`;
  const head = `1 回目 ${seconds(spans[0])}`;
  const rest = [...spans.slice(1)].sort((a, b) => a - b);
  if (rest.length === 0) return head;
  const middle = rest.length >> 1;
  const median = rest.length % 2 === 1 ? rest[middle] : (rest[middle - 1] + rest[middle]) / 2;
  return `${head} · 2 回目以降 中央値 ${seconds(median)} / 最大 ${
    seconds(rest.at(-1) ?? 0)
  }（${rest.length} 回）`;
};

/**
 * sha256 の実物 — 全フレームの uint8 の RGB を連結したバイト列（`packages/models/tests/
 * e2e_wan_pipeline_test.ts` の `rgbBytes` と同じ並び — ADR 0118 決定 8）。`frames` は
 * `wanFrameToRgba` の RGBA（フレーム順）。1 枚の RGBA（`getImageData`）から I2V の条件画像の RGB8 を作るのにも使う。
 */
export const wanRgbBytes = (
  frames: readonly Uint8ClampedArray[],
  plane: number,
): Uint8Array<ArrayBuffer> => {
  const out = new Uint8Array(frames.length * plane * 3);
  for (const [frame, rgba] of frames.entries()) {
    if (rgba.length !== plane * 4) {
      throw Error(`フレーム ${frame} の RGBA ${rgba.length} 要素が ${plane} 画素 × 4 と違う`);
    }
    for (let index = 0; index < plane; index += 1) {
      out.set(rgba.subarray(index * 4, index * 4 + 3), (frame * plane + index) * 3);
    }
  }
  return out;
};

/**
 * 参照ケースの id（sha256 の環境行のキー — Wan2.1 は `fixtures/references/wan.json`・Wan2.2 は
 * `fixtures/references/wan-ti2v.json`）。要求と席が e2e の参照ケースと同じ条件のときだけ返す: 共通条件は
 * `boxing-cats`・seed 42・既定の negative・guidance と shift は manifest の既定、step 数・フレーム数・寸法・席は
 * 世代の参照ケースの表（{@link WanLabGeneration.referenceCases}）。`quant` は実際に回した席（manifest の既定へ
 * 解決した後の名前）。
 *
 * MUST: e2e のケースの定義を変えたら世代の参照ケースの表も変える（行の値が別の条件の sha と突き合わさる）。
 */
export const wanReferenceCaseId = (
  generation: WanLabGeneration,
  resolved: WanResolvedRequest,
  config: WanPipelineConfig,
  defaultNegative: string | undefined,
  quant: string,
): string | undefined => {
  const common = resolved.prompt === "boxing-cats" && resolved.seed === 42 &&
    resolved.negative === defaultNegative && defaultNegative !== undefined &&
    resolved.guidance === config.defaults.guidance && resolved.shift === config.scheduler.shift;
  if (!common) return undefined;
  const size = { width: resolved.width, height: resolved.height };
  const found = generation.referenceCases.find((candidate) =>
    candidate.steps === resolved.steps && candidate.frames === resolved.frames &&
    candidate.sizes.some(({ width, height }) => width === size.width && height === size.height) &&
    candidate.quants.includes(quant)
  );
  return found?.id(quant, size, resolved.shift);
};

/** sha256 の環境行の表（`fixtures/references/wan.json` の形 — ADR 0106）。 */
export type WanReferences = {
  readonly cases: Readonly<Record<string, Readonly<Record<string, string>>>>;
};

export type WanReferenceVerdict =
  | { readonly kind: "no-case" }
  | { readonly kind: "no-row"; readonly caseId: string; readonly key: string }
  | { readonly kind: "match"; readonly caseId: string; readonly key: string }
  | {
    readonly kind: "mismatch";
    readonly caseId: string;
    readonly key: string;
    readonly expected: string;
  };

/**
 * sha256 を環境行と突き合わせる。行が無いのは失敗ではない（その機の行はまだ作られていない — 利用者が
 * 報告した値を行にする）。別の環境キーの行とは比べない（機をまたぐビット同一は保証しない — ADR 0106）。
 */
export const checkWanReference = (
  references: WanReferences,
  caseId: string | undefined,
  key: string,
  sha256: string,
): WanReferenceVerdict => {
  if (caseId === undefined) return { kind: "no-case" };
  const rows = Object.hasOwn(references.cases, caseId) ? references.cases[caseId] : {};
  if (!Object.hasOwn(rows, key)) return { kind: "no-row", caseId, key };
  const expected = rows[key];
  return expected === sha256
    ? { kind: "match", caseId, key }
    : { kind: "mismatch", caseId, key, expected };
};

/** 参照の判定の表示（行 → 状態行・表）。 */
export const referenceVerdictText = (verdict: WanReferenceVerdict): string => {
  switch (verdict.kind) {
    case "no-case":
      return "参照ケースではない（条件が e2e のケースと違う）";
    case "no-row":
      return `${verdict.caseId} · ${verdict.key} の行が無い — この sha を報告する`;
    case "match":
      return `${verdict.caseId} · ${verdict.key} の行と一致`;
    case "mismatch":
      return `${verdict.caseId} · ${verdict.key} の行と不一致（行は ${
        verdict.expected.slice(0, 12)
      }）`;
  }
};

/** Session の診断の要約（記録に載せる欄だけ — `e2e_wan_pipeline_test.ts` の観測行と同じ欄）。 */
export type WanComponentDiagnostics = {
  readonly weightsBytes: number;
  /** slot backing の常駐 + 入力（VAE の常駐 cache〈約 0.58 GiB〉は Session の外なので入らない）。 */
  readonly backingBytes: number;
  readonly geometryProfile: string;
  readonly submitCount: number;
  readonly dispatchCount: number;
  /** 窓平均の最大（1 submit の最大時間の下界 — 計測なしの device で読める唯一の submit の時間）。 */
  readonly maxWindowMeanMs?: number;
  readonly overBudgetChunks: number;
};

/** {@link summarizeWanDiagnostics} が読む欄（`SessionDiagnostics` の部分 — テストが全欄を作らずに済む形）。 */
export type WanDiagnosticsInput = {
  readonly weights: Pick<SessionDiagnostics["weights"], "allocatedBytes">;
  readonly planBacking: Pick<SessionDiagnostics["planBacking"], "residentBytes" | "inputBytes">;
  readonly geometryProfile: string;
  readonly submit:
    & Pick<SessionDiagnostics["submit"], "submitCount" | "dispatchCount">
    & {
      readonly chunkBudget: Pick<
        SessionDiagnostics["submit"]["chunkBudget"],
        "maxWindowMeanMs" | "overBudgetChunks"
      >;
    };
};

export const summarizeWanDiagnostics = (
  diagnostics: WanDiagnosticsInput,
): WanComponentDiagnostics => {
  const { maxWindowMeanMs } = diagnostics.submit.chunkBudget;
  return {
    weightsBytes: diagnostics.weights.allocatedBytes,
    backingBytes: diagnostics.planBacking.residentBytes + diagnostics.planBacking.inputBytes,
    geometryProfile: diagnostics.geometryProfile,
    submitCount: diagnostics.submit.submitCount,
    dispatchCount: diagnostics.submit.dispatchCount,
    ...(maxWindowMeanMs === undefined ? {} : { maxWindowMeanMs }),
    overBudgetChunks: diagnostics.submit.chunkBudget.overBudgetChunks,
  };
};

/** 診断の 1 行（`transformer: 重み 2.645 GiB・backing 3.517 GiB・幾何 default・submit …`）。 */
export const formatWanDiagnostics = (component: string, summary: WanComponentDiagnostics): string =>
  `${component}: 重み ${(summary.weightsBytes / GIB).toFixed(3)} GiB・backing ${
    (summary.backingBytes / GIB).toFixed(3)
  } GiB・幾何 ${summary.geometryProfile}・submit ${
    summary.submitCount.toLocaleString("en-US")
  } 本・窓平均の最大 ${
    summary.maxWindowMeanMs?.toFixed(1) ?? "—"
  } ms・予算超過 ${summary.overBudgetChunks} 本`;
