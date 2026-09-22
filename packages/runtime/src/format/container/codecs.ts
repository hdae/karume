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
  }],
  ["int4-sym-g", {
    layout: "i4",
    packing: { blockElements: 8, blockBytes: 4, alignBytes: 4 },
    scale: "required",
    grouping: "group",
    zeroPoint: "forbidden",
  }],
  ["int2-off", {
    layout: "i2",
    packing: { blockElements: 16, blockBytes: 4, alignBytes: 4 },
    scale: "required",
    grouping: "channel",
    zeroPoint: "forbidden",
  }],
  ["ternary", {
    layout: "i2",
    packing: { blockElements: 16, blockBytes: 4, alignBytes: 4 },
    scale: "required",
    grouping: "channel",
    zeroPoint: "forbidden",
  }],
]);

export const isCodecName = (value: unknown): value is CodecName =>
  typeof value === "string" && CODEC_LEDGER.has(value as CodecName);

/** 台帳のエントリ（登録名は型で閉じているので必ず在る）。 */
export const codecEntry = (codec: CodecName): CodecEntry => {
  const entry = CODEC_LEDGER.get(codec);
  if (entry === undefined) throw new Error(`codec 台帳に '${codec}' が無い`);
  return entry;
};

/** 台帳の登録名を code point 順に並べたもの（診断用）。 */
export const CODEC_NAMES: readonly CodecName[] = [...CODEC_LEDGER.keys()].sort();

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
