/**
 * Codec ledger — docs/container-v1.md §6.2 / §6.3 (ADR 0108 decisions 12 / 13).
 *
 * The ledger is **immutable data in the repository**. Registered names are baked into assets
 * (changing one later needs another migration), so entries may be added but existing entries are
 * never modified.
 *
 * `packing` is the source of truth for the bit width; the bit count is derived
 * (`blockBytes · 8 / blockElements`). The declaration side (`encoding.packing`) is a copy of the
 * ledger, and the reader **checks that they match** (the check that keeps assets written with a
 * different version of the ledger from being accepted silently — §6.1).
 *
 * `layout` is the kind of decode path (CPU `decodeI8` / `decodeI4` / `decodeI2`, GPU WGSL), and
 * `ternary` is a subset of the value range of `int2-off`, so it **shares the i2 path as is** (zero
 * lines added to the runtime — decision 13).
 *
 * `int4-sym-g-f16` has the bytes of `int4-sym-g` and an f16 step, and its own layout `i4f16` (ADR
 * 0145 decision 3): the step dtype travels in the codec name, and a consumer that switches on the
 * layout does not read f16 steps as f32 unless it decides to. Only `expert_linear` reads it.
 */

import { compareCodePoints } from "./json.ts";

/** Registered names of the ledger. */
export type CodecName =
  | "f32"
  | "f16"
  | "bf16"
  | "i32"
  | "int8-sym"
  | "int4-sym-g"
  | "int4-sym-g-f16"
  | "int2-off"
  | "ternary";

/** Kind of decode path (derived from the codec — consumers branch on it). */
export type CodecLayout = "f32" | "f16" | "bf16" | "i32" | "i8" | "i4" | "i4f16" | "i2";

export type CodecPacking = {
  /** Number of elements one packing block carries. */
  readonly blockElements: number;
  /** Bytes of one packing block. */
  readonly blockBytes: number;
  /** Alignment required at the start of the payload. */
  readonly alignBytes: number;
};

export type CodecEntry = {
  readonly layout: CodecLayout;
  readonly packing: CodecPacking;
  /**
   * Whether a scale is needed. `required` = quantized codec (`encoding.scale` required, and
   * `rowAxis` / `groupSize` too), `forbidden` = not quantized (none of the 3 fields may be written).
   */
  readonly scale: "required" | "forbidden";
  /**
   * Group granularity. `channel` = per-channel (`groupSize` MUST equal the row length), `group` =
   * a power-of-two group length that divides the row length (at least `MIN_GROUP_SIZE` — ADR 0069
   * decision 2). undefined when not quantized.
   */
  readonly grouping: "channel" | "group" | undefined;
  /** In the first edition none of the 4 codecs may have a zeroPoint (§6.3). */
  readonly zeroPoint: "allowed" | "forbidden";
  /**
   * The `rowAxis` values that may be declared (the tables of §6.1 / §6.3). Only per-channel i8 can
   * read row axis 1 (`conv_transpose1d`); for the group codec and the i2 path the decode
   * (`decodeI4` / `decodeI2` and WGSL) is fixed to axis 0. Empty when not quantized, since
   * `rowAxis` may not be written then.
   * MUST: the merge layer rejects anything outside this set — the decode does not receive the
   * declared axis, so for a shape whose two axes have the same length, like `[N,N]`, the scale-shape
   * check passes and the weight is silently decoded along the other axis.
   */
  readonly rowAxes: readonly (0 | 1)[];
  /**
   * The dtype of the companion scale, implied by the codec (`encoding.scale.dtype` MUST equal it):
   * `f32` for every quantized codec but `int4-sym-g-f16`, whose steps are f16 (ADR 0145 decision 3).
   * undefined when not quantized.
   */
  readonly scaleDtype: ScaleDtype | undefined;
};

/**
 * Accepted scale dtypes (§6.1). Each quantized codec takes exactly one of them
 * ({@link CodecEntry.scaleDtype}); f16 only for `int4-sym-g-f16` (ADR 0069 decision 3's conditions
 * for f16, bound to one op by ADR 0145 decision 3).
 */
export const SCALE_DTYPES = ["f32", "f16"] as const;
export type ScaleDtype = (typeof SCALE_DTYPES)[number];

/**
 * Lower bound of the group length of group quantization (ADR 0069 decision 2 — the same constraint
 * as ORT).
 */
export const MIN_GROUP_SIZE = 16;

const RAW = (layout: CodecLayout, blockBytes: number): CodecEntry => ({
  layout,
  packing: { blockElements: 1, blockBytes, alignBytes: 4 },
  scale: "forbidden",
  grouping: undefined,
  zeroPoint: "forbidden",
  rowAxes: [],
  scaleDtype: undefined,
});

/**
 * The first-edition ledger. The values are carried over verbatim from the implementation of the time
 * (§6.3).
 *
 * NOTE: the packing of `int8-sym` is **1 element / 1 byte**. Packing 4 elements into a u32 word is a
 * concern of the GPU binding (`unpack4xI8`), not the granularity of the payload. 4 elements / 4
 * bytes would add the constraint `numel % 4 == 0`, which v1 did not have, against "create not a
 * single new value". The `alignBytes` of 4 for f16 / bf16 has the same reason (an element is 2 bytes,
 * but it is bound as `array<u32>`).
 */
export const CODEC_LEDGER: ReadonlyMap<CodecName, CodecEntry> = new Map<CodecName, CodecEntry>([
  ["f32", RAW("f32", 4)],
  ["f16", RAW("f16", 2)],
  ["bf16", RAW("bf16", 2)],
  ["i32", RAW("i32", 4)],
  [
    "int8-sym",
    {
      layout: "i8",
      packing: { blockElements: 1, blockBytes: 1, alignBytes: 4 },
      scale: "required",
      grouping: "channel",
      zeroPoint: "forbidden",
      rowAxes: [0, 1],
      scaleDtype: "f32",
    },
  ],
  [
    "int4-sym-g",
    {
      layout: "i4",
      packing: { blockElements: 8, blockBytes: 4, alignBytes: 4 },
      scale: "required",
      grouping: "group",
      zeroPoint: "forbidden",
      rowAxes: [0],
      scaleDtype: "f32",
    },
  ],
  // ADR 0145 decision 3: the bytes, nibble order, grouping and row axis of `int4-sym-g`, one f16
  // step per group (two per u32 word: step 2i in the low half, 2i+1 in the high half).
  [
    "int4-sym-g-f16",
    {
      layout: "i4f16",
      packing: { blockElements: 8, blockBytes: 4, alignBytes: 4 },
      scale: "required",
      grouping: "group",
      zeroPoint: "forbidden",
      rowAxes: [0],
      scaleDtype: "f16",
    },
  ],
  [
    "int2-off",
    {
      layout: "i2",
      packing: { blockElements: 16, blockBytes: 4, alignBytes: 4 },
      scale: "required",
      grouping: "channel",
      zeroPoint: "forbidden",
      rowAxes: [0],
      scaleDtype: "f32",
    },
  ],
  [
    "ternary",
    {
      layout: "i2",
      packing: { blockElements: 16, blockBytes: 4, alignBytes: 4 },
      scale: "required",
      grouping: "channel",
      zeroPoint: "forbidden",
      rowAxes: [0],
      scaleDtype: "f32",
    },
  ],
]);

export const isCodecName = (value: unknown): value is CodecName =>
  typeof value === "string" && CODEC_LEDGER.has(value as CodecName);

/** Kind of decode path (consumers branch on this — `ternary` is i2). */
export const codecLayout = (codec: CodecName): CodecLayout => codecEntry(codec).layout;

/** The ledger entry (registered names are closed by the type, so it always exists). */
export const codecEntry = (codec: CodecName): CodecEntry => {
  const entry = CODEC_LEDGER.get(codec);
  if (entry === undefined) throw new Error(`codec '${codec}' is not in the codec ledger`);
  return entry;
};

/** Registered names of the ledger in code point order (for diagnostics). */
export const CODEC_NAMES: readonly CodecName[] = [...CODEC_LEDGER.keys()].sort(compareCodePoints);

/**
 * `groupSize` of a per-channel codec (= row length). A zero-element degenerate shape
 * (`in_features = 0` and the like) has row length 0, and `groupSize` MUST be at least 1, so it is
 * rounded to 1 (the group count is brought back to 1 by {@link groupCount}).
 */
export const perChannelGroupSize = (rowLength: number): number => Math.max(rowLength, 1);

/**
 * Group count of the scale, `row length / groupSize` (§6.1). A degenerate shape with row length 0
 * has group count 1 (a per-channel scale has one per row, matching the old distribution's
 * `[rows, 1]`).
 */
const groupCount = (rowLength: number, groupSize: number): number =>
  rowLength === 0 ? 1 : rowLength / groupSize;

/**
 * Length of a quantized row = `numel / shape[rowAxis]` (the flattened length of everything except
 * the row axis — `Cin·K` for conv1d `[O,Cin,K]` with row axis 0). A degenerate shape with 0 rows
 * gives 0 (no NaN from the division).
 */
export const quantizedRowLength = (shape: readonly number[], rowAxis: number): number => {
  const rows = shape[rowAxis];
  return rows === 0 ? 0 : shape.reduce((count, dim) => count * dim, 1) / rows;
};

/**
 * Logical shape of the companion scale = **rank 2 regardless of rank**
 * `[shape[rowAxis], row length / groupSize]` (§6.1 / ADR 0069 decision 3). A per-channel codec has
 * `[rows, 1]` with group count 1.
 *
 * MUST: the scale shape is derived from this one function only. The merge layer (scale block
 * length), the residency planner (scale bytes), the scale shape handed from the container to Session
 * construction and the check in the CPU decode `decodeI4` all assume the same shape; if any one of
 * them has its own formula, the accepted shape and the shape the decode reads silently disagree (a
 * silent wrong value where the group scale is handed out as one value per channel).
 * NOTE: a pair that does not divide evenly is returned as a non-integer (the divisibility check is
 * the caller's job — throwing here would mix the callers' error types).
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
 * Byte length of the payload (the formula of §6.1). `numel % blockElements == 0` MUST hold — a
 * partial packing block makes "only the last block is short", and row boundaries drift off byte
 * boundaries.
 */
export const payloadBytes = (codec: CodecName, numel: number, where: string): number => {
  const { blockElements, blockBytes } = codecEntry(codec).packing;
  if (numel % blockElements !== 0) {
    throw new Error(
      `${where}: element count ${numel} is not divisible by the packing of codec '${codec}' (${blockElements} elements / block)`,
    );
  }
  return (numel / blockElements) * blockBytes;
};

/**
 * Bytes of one scale step of a quantized codec (its {@link CodecEntry.scaleDtype}: f32 = 4, f16 =
 * 2). The bytes of a run of whole scale rows are `rows · steps per row ·` this — the length of a
 * piece of a split scale, which carries no padding of its own unless it is the last.
 */
export const scaleStepBytes = (codec: CodecName): number => {
  const dtype = codecEntry(codec).scaleDtype;
  if (dtype === undefined) throw new Error(`codec '${codec}' is not quantized, so it has no scale`);
  return dtype === "f16" ? 2 : 4;
};

/**
 * Payload byte length of the companion scale (element count → bytes — the scale version of
 * {@link payloadBytes}): 4 bytes per f32 step, and 2 per f16 step **padded to a whole u32 word**
 * (an odd step count rounds up — the kernel binds the f16 steps as `array<u32>`, two per word,
 * and `writeBuffer` takes whole words). Every scale dtype gives a multiple of 4.
 *
 * MUST: values outside safe integers fail loudly. The byte count is used for allocation sizes and
 * the block-length check, so letting a rounded value through breaks that check's claim that "the
 * declaration and the actual data match".
 */
export const scaleBytes = (codec: CodecName, count: number, where: string): number => {
  if (!Number.isSafeInteger(count) || count < 0) {
    throw new Error(`${where}: scale element count ${count} is not a non-negative safe integer`);
  }
  const bytes = Math.ceil((count * scaleStepBytes(codec)) / 4) * 4;
  if (!Number.isSafeInteger(bytes)) {
    throw new Error(`${where}: scale byte count ${bytes} is outside the safe integer range`);
  }
  return bytes;
};
