/**
 * GPU lab の Wan のタブ（ADR 0118 段 9 — Chrome で Wan2.1 を回す）の GPU に依らない部分。
 *
 * ここに置くのは純関数と型だけ（DOM も GPU も触らない — `wan-plan_test.ts` が deno test で縛る）:
 *
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
import { WAN21_GENERATION } from "../../../packages/models/src/wan/descriptor.ts";
import { TEMPORAL_COMPRESSION } from "../../../packages/models/src/wan/plan.ts";

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;
const F32_BYTES = 4;

/** 画素 → DiT のトークンの縮尺（VAE の空間 8 × patch 2）。 */
const PIXELS_PER_TOKEN = 16;

/**
 * DiT の寸法（ADR 0118 Context「調査の結論」— 30 層・dim 1536・12 heads × 128・FFN 8960）。上流の
 * 1.3B の config の値で、配布形のグラフ宣言（`gelu_* [1,S,8960]`）とも一致する（2026-10-03 に読んだ）。
 */
export const WAN_DIT_HEADS = 12;
export const WAN_DIT_FFN_WIDTH = 8960;

/** cross-attn のキーの行数（テキスト埋め込みを詰める文脈 `encoder_hidden_states [1,512,4096]`）。 */
export const WAN_TEXT_CONTEXT_ROWS = 512;

/**
 * VAE の段で最大の値のバイト数（`vae_decoder_next` の `expand_5 [768,128,2,256]` / `view_17
 * [4,192,256,256]` の f32 — 2026-10-03 に配布形のグラフ宣言から読んだ値）。タイル（潜在 32）で固定なので
 * フレーム数と寸法に依らない。first のグラフの最大（75,497,472 B）はこれより小さい。
 *
 * NOTE: タイル辺は配布物だけで差し替えられる（ADR 0118 決定 2）。差し替えたら読み直す値。
 */
export const WAN_VAE_LARGEST_VALUE_BYTES = 201_326_592;

/** WebGPU の既定（requiredLimits で要求しなければ device はこの値になる）。 */
export const WEBGPU_DEFAULT_STORAGE_BINDING = 128 * MIB;
export const WEBGPU_DEFAULT_BUFFER = 256 * MIB;

/**
 * Chromium の単一 ArrayBuffer の上限（2³¹ − 2 MiB — docs/limitations.md「ブラウザ: Chromium は単一
 * ArrayBuffer を…」節）。runtime の単一形の容器の上限と同じ値を共有する。
 */
export const CHROMIUM_ARRAY_BUFFER_MAX = MAX_SINGLE_CONTAINER_BYTES;

export type WanSize = { readonly width: number; readonly height: number };

/** 受理集合のフレーム数（4n+1 の 5〜81 — 世代の記述子 `descriptor.ts` の受理集合を正本にする）。 */
export const wanFrameChoices = (): number[] => {
  const { minFrames, maxFrames } = WAN21_GENERATION;
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
 * DiT のトークン数 S（潜在フレーム数 × H/16 × W/16）。受理集合の外は落とす — 判定表は受理集合の中の
 * 要求についてだけ意味を持つ（外の要求は `generate` が `ModelInputError` で拒む）。
 */
export const wanTokenCount = (frames: number, size: WanSize): number => {
  const { minFrames, maxFrames, acceptedSizes } = WAN21_GENERATION;
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
  return latentFrames * (size.height / PIXELS_PER_TOKEN) * (size.width / PIXELS_PER_TOKEN);
};

/** 行ブロックで割れない最大の値（束縛上限と maxBufferSize の下限 — {@link judgeWanLimits}）。 */
export type WanLargestValue = {
  readonly bytes: number;
  readonly what: string;
};

/**
 * 要求 1 本で最大の値。DiT の FFN 中間 `[1,S,8960]` の f32（S = 14,040 で 503,193,600 B・32,760 で
 * 1,174,118,400 B — ADR 0118 段 8 の「FFN 中間 480 MiB / 1.09 GiB」）と VAE の最大の値の大きい方。
 * DiT の他の値は `[1,S,1536]` 以下なので FFN 中間より小さい。
 *
 * NOTE: 自己 attention のスコア S は値として持たない（attention の 1 op の内側で行ブロックに割る）ので
 * ここには入らない — 下限への効き方は {@link judgeWanLimits} の行ブロックの欄。
 */
export const wanLargestValue = (frames: number, size: WanSize): WanLargestValue => {
  const tokens = wanTokenCount(frames, size);
  const ffn = tokens * WAN_DIT_FFN_WIDTH * F32_BYTES;
  return ffn >= WAN_VAE_LARGEST_VALUE_BYTES
    ? { bytes: ffn, what: `DiT の FFN 中間 [1,${tokens},${WAN_DIT_FFN_WIDTH}] f32` }
    : { bytes: WAN_VAE_LARGEST_VALUE_BYTES, what: "VAE（next）の中間 [768,128,2,256] f32" };
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
 * スコアの行ブロック（自己 attention は S × S・cross-attn は S × 512 — どちらも f32 格納・H = 12。
 * 既定で `attentionScoreStorage: "f16"` は使わない — ADR 0118 決定 6）。
 */
export const wanAttentionRowBlocks = (
  frames: number,
  size: WanSize,
  maxStorageBufferBindingSize: number,
): { readonly self: WanRowBlocks; readonly cross: WanRowBlocks } => {
  const tokens = wanTokenCount(frames, size);
  return {
    self: rowBlocks(tokens, WAN_DIT_HEADS * tokens * F32_BYTES, maxStorageBufferBindingSize),
    cross: rowBlocks(
      tokens,
      WAN_DIT_HEADS * WAN_TEXT_CONTEXT_ROWS * F32_BYTES,
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
export const judgeWanLimits = (limits: WanLimits, frames: number, size: WanSize): WanLimitRow[] => {
  const largest = wanLargestValue(frames, size);
  const binding = limits.maxStorageBufferBindingSize;
  const attention = wanAttentionRowBlocks(frames, size, binding);
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
export const wanMaxFramesWithin = (limits: WanLimits, size: WanSize): number | undefined =>
  wanFrameChoices().filter((frames) =>
    judgeWanLimits(limits, frames, size).every((row) => row.verdict !== "short")
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
 */
export const buildWanRequest = (
  form: WanForm,
  prompts: readonly WanPrompt[],
  config: WanPipelineConfig,
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
  const size = parseWanSize(form.size);
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
      width: size.width,
      height: size.height,
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

/** 生成イベントの時刻（`performance.now()` の ms）。 */
export type WanTimelineMark =
  | {
    readonly kind: "stage";
    readonly component: "transformer" | "vae_decoder";
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
  readonly stageMs: Readonly<Partial<Record<"transformer" | "vae_decoder", number>>>;
  readonly stepMs: readonly number[];
  readonly tileMs: readonly number[];
};

/**
 * イベントの時刻列 → {@link WanTimeline}。並びがパイプラインの取り決め（段の start → step / tile が
 * 1 から順に → 段の end）から外れたら落とす — 黙って時間を別の区間へ帰属させない。
 */
export const summarizeWanTimeline = (marks: readonly WanTimelineMark[]): WanTimeline => {
  const stageMs: Partial<Record<"transformer" | "vae_decoder", number>> = {};
  const stepMs: number[] = [];
  const tileMs: number[] = [];
  let open: { readonly component: "transformer" | "vae_decoder"; readonly ms: number } | undefined;
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
 * `wanFrameToRgba` の RGBA（フレーム順）。
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
 * 席名を持たない参照ケースの id の席（`e2e_wan_pipeline_test.ts` の `F16_QUANT` — 既定席が実用席へ移っても
 * 既存の行はこの席の値・ADR 0120 裁定 2026-10-04 の 4）。
 */
const UNSEATED_QUANT = "f16";

/**
 * 席を名乗る行を持つ席（id は `<席>-<席名を持たない id>` — e2e の `seatCaseId`）。e2e の `SEAT_QUANTS`
 * 〈2 ステップ〉と `FULL_CASES` の実用席〈50 ステップ〉の写し。ここに無い席の組は参照ケースではない。
 */
const SEATED_CASES: Readonly<Record<string, readonly string[]>> = {
  "2step-seed-boxing-cats-seed42": ["f16+dit8", "f16+dit8-a8-attn8-s16"],
  "50step-boxing-cats-seed42": ["f16+dit8-a8-attn8-s16"],
  "50step-boxing-cats-seed42-81f": ["f16+dit8-a8-attn8-s16"],
};

/**
 * 参照ケースの id（sha256 の環境行のキー — `fixtures/references/wan.json`）。要求と席が e2e の参照ケースと
 * 同じ条件のときだけ返す（`e2e_wan_pipeline_test.ts` の `SEED_CASE`〈2 ステップ〉と `FULL_CASES`〈50
 * ステップ・33 / 81 フレーム〉: `boxing-cats`・seed 42・既定の negative・guidance と shift は manifest の
 * 既定・832×480）。`quant` は実際に回した席（manifest の既定へ解決した後の名前）— `f16` 席は席名を持たない
 * id、i8 の席は席名を先頭に置いた id。
 *
 * MUST: e2e のケースの定義を変えたらここも変える（行の値が別の条件の sha と突き合わさる）。
 */
export const wanReferenceCaseId = (
  resolved: WanResolvedRequest,
  config: WanPipelineConfig,
  defaultNegative: string | undefined,
  quant: string,
): string | undefined => {
  const common = resolved.prompt === "boxing-cats" && resolved.seed === 42 &&
    resolved.negative === defaultNegative && defaultNegative !== undefined &&
    resolved.guidance === config.defaults.guidance && resolved.shift === config.scheduler.shift &&
    resolved.width === 832 && resolved.height === 480;
  if (!common) return undefined;
  const base = resolved.steps === 2 && resolved.frames === 33
    ? "2step-seed-boxing-cats-seed42"
    : resolved.steps === 50 && resolved.frames === 33
    ? "50step-boxing-cats-seed42"
    : resolved.steps === 50 && resolved.frames === 81
    ? "50step-boxing-cats-seed42-81f"
    : undefined;
  if (base === undefined) return undefined;
  if (quant === UNSEATED_QUANT) return base;
  return SEATED_CASES[base].includes(quant) ? `${quant}-${base}` : undefined;
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
