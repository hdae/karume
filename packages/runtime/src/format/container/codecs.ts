/**
 * codec 台帳 — docs/container-v1.md §6.2 / §6.3（ADR 0108 決定 12 / 13）。
 *
 * 台帳は**リポ内の不変なデータ**である。登録名は資産に焼かれる（後から変えると移行がもう 1 回
 * 要る）ので、エントリの追加はあっても既存エントリの改変は無い。
 *
 * `packing` が bit 幅の正本で、bit 数は派生値（`blockBytes · 8 / blockElements`）。宣言側
 * （`encoding.packing`）は台帳の写しであり、読み手は**一致することを検査する**（別の版の台帳で
 * 書かれた資産を静かに受理しないための門 — §6.1）。
 *
 * `layout` は展開経路（CPU の `decodeI8` / `decodeI4` / `decodeI2`・GPU の WGSL）の種別で、
 * `ternary` は `int2-off` の値域の部分集合なので i2 経路を**そのまま共有する**（runtime の
 * 追加は 0 行 — 決定 13）。
 */

import { compareCodePoints } from "./json.ts";

/** 台帳の登録名。 */
export type CodecName =
  | "f32"
  | "f16"
  | "bf16"
  | "i32"
  | "int8-sym"
  | "int4-sym-g"
  | "int2-off"
  | "ternary";

/** 展開経路の種別（codec からの派生値 — 消費側はこれで分岐する）。 */
export type CodecLayout = "f32" | "f16" | "bf16" | "i32" | "i8" | "i4" | "i2";

export type CodecPacking = {
  /** 1 packing block が運ぶ要素数。 */
  readonly blockElements: number;
  /** 1 packing block のバイト数。 */
  readonly blockBytes: number;
  /** payload 先頭の整列要求。 */
  readonly alignBytes: number;
};

export type CodecEntry = {
  readonly layout: CodecLayout;
  readonly packing: CodecPacking;
  /**
   * scale の要否。`required` = 量子化 codec（`encoding.scale` 必須・`rowAxis` / `groupSize` も
   * 必須）、`forbidden` = 非量子化（3 欄とも書けない）。
   */
  readonly scale: "required" | "forbidden";
  /**
   * group の刻み。`channel` = per-channel（`groupSize` は行長に等しい MUST）、`group` = 行長を
   * 割る 2 冪の group 長（`MIN_GROUP_SIZE` 以上 — ADR 0069 決定 2）。非量子化は undefined。
   */
  readonly grouping: "channel" | "group" | undefined;
  /** 初版は 4 種とも zeroPoint を持てない（§6.3）。 */
  readonly zeroPoint: "allowed" | "forbidden";
  /**
   * 宣言してよい `rowAxis`（§6.1 / §6.3 の表）。行の軸 1 を読めるのは per-channel i8 だけで
   * （`conv_transpose1d`）、group codec と i2 経路は展開（`decodeI4` / `decodeI2` と WGSL）が軸 0
   * 固定。非量子化は `rowAxis` を書けないので空。
   * MUST: 合流層はこの集合の外を拒否する — 展開は宣言の軸を受け取らないので、`[N,N]` のように
   * 両軸の長さが一致する形では scale 形の突合が通り、別の軸で黙って展開される。
   */
  readonly rowAxes: readonly (0 | 1)[];
};

/** scale の dtype の受理集合（初版は f32 のみ — §6.1）。 */
export const SCALE_DTYPES = ["f32"] as const;
export type ScaleDtype = typeof SCALE_DTYPES[number];

/** group 量子化の group 長の下限（ADR 0069 決定 2 — ORT と同じ制約）。 */
export const MIN_GROUP_SIZE = 16;

const RAW = (layout: CodecLayout, blockBytes: number): CodecEntry => ({
  layout,
  packing: { blockElements: 1, blockBytes, alignBytes: 4 },
  scale: "forbidden",
  grouping: undefined,
  zeroPoint: "forbidden",
  rowAxes: [],
});

/**
 * 初版の台帳。値は現行実装からの逐語移送（§6.3）。
 *
 * NOTE: `int8-sym` の packing は **1 要素 / 1 バイト**である。u32 語に 4 要素を詰めるのは GPU 束縛
 * （`unpack4xI8`）の側の都合で、payload の粒度ではない。4 要素 / 4 バイトにすると
 * `numel % 4 == 0` という v1 に無かった制約が入り、「新しい値を 1 つも作らない」に反する。
 * f16 / bf16 の `alignBytes` が 4 なのも同じ理由（要素は 2 バイトだが `array<u32>` で束縛する）。
 */
export const CODEC_LEDGER: ReadonlyMap<CodecName, CodecEntry> = new Map<CodecName, CodecEntry>([
  ["f32", RAW("f32", 4)],
  ["f16", RAW("f16", 2)],
  ["bf16", RAW("bf16", 2)],
  ["i32", RAW("i32", 4)],
  ["int8-sym", {
    layout: "i8",
    packing: { blockElements: 1, blockBytes: 1, alignBytes: 4 },
    scale: "required",
    grouping: "channel",
    zeroPoint: "forbidden",
    rowAxes: [0, 1],
  }],
  ["int4-sym-g", {
    layout: "i4",
    packing: { blockElements: 8, blockBytes: 4, alignBytes: 4 },
    scale: "required",
    grouping: "group",
    zeroPoint: "forbidden",
    rowAxes: [0],
  }],
  ["int2-off", {
    layout: "i2",
    packing: { blockElements: 16, blockBytes: 4, alignBytes: 4 },
    scale: "required",
    grouping: "channel",
    zeroPoint: "forbidden",
    rowAxes: [0],
  }],
  ["ternary", {
    layout: "i2",
    packing: { blockElements: 16, blockBytes: 4, alignBytes: 4 },
    scale: "required",
    grouping: "channel",
    zeroPoint: "forbidden",
    rowAxes: [0],
  }],
]);

export const isCodecName = (value: unknown): value is CodecName =>
  typeof value === "string" && CODEC_LEDGER.has(value as CodecName);

/** 展開経路の種別（消費側の分岐はこれで行う — `ternary` は i2）。 */
export const codecLayout = (codec: CodecName): CodecLayout => codecEntry(codec).layout;

/** 台帳のエントリ（登録名は型で閉じているので必ず在る）。 */
export const codecEntry = (codec: CodecName): CodecEntry => {
  const entry = CODEC_LEDGER.get(codec);
  if (entry === undefined) throw new Error(`codec 台帳に '${codec}' が無い`);
  return entry;
};

/** 台帳の登録名を code point 順に並べたもの（診断用）。 */
export const CODEC_NAMES: readonly CodecName[] = [...CODEC_LEDGER.keys()].sort(compareCodePoints);

/**
 * per-channel codec の `groupSize`（= 行長）。要素数 0 の退化形（`in_features = 0` など）は行長 0 で、
 * `groupSize` は 1 以上 MUST なので 1 に丸める（group 数は {@link groupCount} が 1 に戻す）。
 */
export const perChannelGroupSize = (rowLength: number): number => Math.max(rowLength, 1);

/**
 * scale の group 数 `行長 / groupSize`（§6.1）。行長 0 の退化形は group 数 1（per-channel scale は
 * 行ごとに 1 本あり、旧配布形の `[rows, 1]` と一致する）。
 */
const groupCount = (rowLength: number, groupSize: number): number =>
  rowLength === 0 ? 1 : rowLength / groupSize;

/**
 * 量子化行の長さ = `numel / shape[rowAxis]`（行の軸を除いた残りを平坦化した長さ — conv1d
 * `[O,Cin,K]` の行軸 0 なら `Cin·K`）。行数 0 の退化形は 0（除算の NaN を作らない）。
 */
export const quantizedRowLength = (shape: readonly number[], rowAxis: number): number => {
  const rows = shape[rowAxis];
  return rows === 0 ? 0 : shape.reduce((count, dim) => count * dim, 1) / rows;
};

/**
 * companion scale の論理形 = **rank 非依存の rank 2** `[shape[rowAxis], 行長 / groupSize]`
 * （§6.1 / ADR 0069 決定 3）。per-channel codec は group 数 1 の `[rows, 1]`。
 *
 * MUST: scale の形はこの 1 本からだけ導く。合流層（scale block の長さ）・常駐プランナ（scale の
 * バイト数）・容器から Session 構築へ渡す scale 形・CPU 展開 `decodeI4` の突合が同じ形を前提に
 * しており、どれか 1 つが別の式を持つと、受理した形と展開が読む形が静かに食い違う（group scale が
 * 1 チャネル 1 値として配られる沈黙誤値）。
 * NOTE: 割り切れない組は非整数のまま返す（整除の検査は呼び手の担当 — ここで投げると呼び手ごとの
 * エラー型が混ざる）。
 */
export const groupScaleShape = (
  shape: readonly number[],
  rowAxis: number,
  groupSize: number,
): readonly [number, number] => [
  shape[rowAxis],
  groupCount(quantizedRowLength(shape, rowAxis), groupSize),
];

/**
 * payload のバイト長（§6.1 の式）。`numel % blockElements == 0` MUST — 端数の packing block は
 * 「最後の block だけ短い」を作り、行境界がバイト境界からずれる。
 */
export const payloadBytes = (codec: CodecName, numel: number, where: string): number => {
  const { blockElements, blockBytes } = codecEntry(codec).packing;
  if (numel % blockElements !== 0) {
    throw new Error(
      `${where}: 要素数 ${numel} が codec '${codec}' の packing（${blockElements} 要素 / block）で割り切れない`,
    );
  }
  return (numel / blockElements) * blockBytes;
};

/**
 * companion scale の payload バイト長（要素数 → バイト数 — {@link payloadBytes} の scale 版）。
 * scale の dtype は初版では f32 の 1 通りきり（{@link SCALE_DTYPES}）なので 1 要素 4 バイト。
 *
 * MUST: 安全整数の外は fail loudly。バイト数は確保寸法と block 長の突合に使われるので、
 * 丸まった値を通すと「宣言と現物が一致している」という突合門の主張が壊れる。
 */
export const scaleBytes = (count: number, where: string): number => {
  if (!Number.isSafeInteger(count) || count < 0) {
    throw new Error(`${where}: scale の要素数 ${count} が非負の安全整数でない`);
  }
  const bytes = count * 4;
  if (!Number.isSafeInteger(bytes)) {
    throw new Error(`${where}: scale のバイト数 ${bytes} が安全整数の外`);
  }
  return bytes;
};
