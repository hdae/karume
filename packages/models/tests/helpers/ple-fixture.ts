/**
 * A synthetic PLE index and blocks (the fixture shared by the checks of `src/gemma/ple.ts` /
 * `ple-gpu.ts`).
 *
 * It needs no real assets — the index is built as a schema 3 document and passed through
 * `parseGemma4PleIndex`, and the blocks are made with deterministic values as "a raw matrix of the
 * rows of a token range laid out without gaps" (the same shape as the container's assets — ADR 0109
 * decision 4). The values follow from `quantized` / `scaleOf`, so the expected dequantized values
 * ({@link expectedValue}) are computed entirely on the test side.
 *
 * The reader ({@link PleFixture.openBlock}) **records every call**. The branches of the policy
 * table (full read / row read / resident hit) can be observed only through "which range of which
 * block was read how many times", and it also keeps the peak of concurrent reads
 * (`peak.inFlight`) — with a source where 1 read takes 1 fd, a working limit is visible only in the
 * peak, not in the count.
 *
 * {@link pleBf16Fixture} builds the schema 5 form (a bf16 `values` table, no `scales`): its values
 * are chosen to be exact in bf16, so the expected gather ({@link expectedBf16Value}) is the chosen
 * number times the embed scale — the test never decodes bf16 bits itself.
 */

import type { AssetReader } from "@karume/runtime";
import {
  BF16_BYTES,
  FLOAT_SCHEMA,
  type Gemma4PleAssetSource,
  type Gemma4PleFloatIndex,
  type Gemma4PleQuantizedIndex,
  type Gemma4PleQuantizedStorage,
  GROUPED_SCHEMA,
  packFactor,
  parseGemma4PleIndex,
  PLE_INDEX_ASSET,
  SCALE_BYTES,
  SCHEMA,
} from "../../src/gemma/ple-index.ts";

/** The dimensions of the synthetic index. */
export type PleFixtureSpec = {
  readonly storage?: Gemma4PleQuantizedStorage;
  readonly tokens: number;
  readonly layers: number;
  readonly dim: number;
  /**
   * A power of 2 (f32 multiplication is exact — the synthetic side keeps the same property as ADR
   * 0085 decision 4).
   */
  readonly embedScale: number;
  /** The token rows one `values` block holds (default = `tokens` = 1 block). */
  readonly valueRows?: number;
  /** The token rows one `scales` block holds (default = `valueRows`). */
  readonly scaleRows?: number;
  /**
   * Values per scale of a (token, layer) row: set = a schema 4 index (`groupSize` spelled, a scales
   * row of `layers × dim / groupSize` f32 in (layer, group) order); absent = schema 3.
   */
  readonly groupSize?: number;
};

/** One range the reader received. */
export type BlockRead = {
  readonly asset: string;
  readonly offset: number;
  readonly length: number;
};

/** What both fixture forms share (the document, the blocks and the recording reader). */
export type PleFixtureBase = {
  /** The index's raw JSON (the source when driving the `parseGemma4PleIndex` check directly). */
  readonly document: Record<string, unknown>;
  /** Block name → contents. */
  readonly blocks: ReadonlyMap<string, Uint8Array<ArrayBuffer>>;
  /** The `openBlock` for `createGemma4Ple` / `createGemma4PleResident`. */
  readonly openBlock: (asset: string) => AssetReader;
  /** The number of readers opened (calls to `asset()`). */
  readonly opens: string[];
  readonly reads: BlockRead[];
  readonly peak: { inFlight: number };
};

/** A quantized fixture (schema 3 / 4). */
export type PleFixture = PleFixtureBase & { readonly index: Gemma4PleQuantizedIndex };

/** A bf16 fixture (schema 5 — {@link pleBf16Fixture}). */
export type PleBf16Fixture = PleFixtureBase & { readonly index: Gemma4PleFloatIndex };

/**
 * The i8 value is `id * 10 + layer * 2 + column` (packed storage truncates it to the storage
 * width).
 */
export const quantized = (id: number, layer: number, column: number): number =>
  id * 10 + layer * 2 + column;

/**
 * The per-row scale is `(id+1) / 2^(layer+1)`.
 *
 * MUST: **make it depend on the token too**. As a function of the layer alone the scale table would
 * repeat one token row, so swapped rows, rotation and mixed-up blocks would give "the same bytes",
 * and every claim about order would be blind to them (a shifted scale dequantizes with another
 * token's scale while the shape and dtype still match — a silent wrong value). The numerator is an
 * integer and the denominator a power of 2, so it is exact in f32 (the condition for the expected
 * dequantized values to be computed entirely on the test side).
 */
export const scaleOf = (id: number, layer: number, group = 0): number =>
  // Group 0 is the schema 3 value as is; a later group adds group / 64 to the numerator, so every
  // (token, group) pair below 64 groups has its own step (a scale read from the wrong group or the
  // row's first group is visible) and the value stays exact in f32.
  (id + 1 + group / 64) / 2 ** (layer + 1);

/** The scale group of value `column` (always 0 for a schema 3 fixture). */
export const groupOf = (spec: PleFixtureSpec, column: number): number =>
  spec.groupSize === undefined ? 0 : Math.floor(column / spec.groupSize);

/** Scales of one layer of a row (1 for schema 3). */
const groupsOf = (spec: PleFixtureSpec): number =>
  spec.groupSize === undefined ? 1 : spec.dim / spec.groupSize;

/**
 * The quantized value after truncation to the storage dtype (packed values wrap into the signed
 * range).
 */
export const storedValue = (
  storage: Gemma4PleQuantizedStorage,
  id: number,
  layer: number,
  column: number,
): number => {
  const bits = storage === "i8" ? 8 : storage === "i4" ? 4 : 2;
  const span = 1 << bits;
  const offset = span >> 1;
  return (((quantized(id, layer, column) % span) + span) % span) - offset;
};

/**
 * The two stages dequantize → embed scale (the same order and the same rounding points as
 * `writeRow`).
 */
export const expectedValue = (
  spec: PleFixtureSpec,
  id: number,
  layer: number,
  column: number,
): number =>
  Math.fround(
    storedValue(spec.storage ?? "i8", id, layer, column) *
      scaleOf(id, layer, groupOf(spec, column)),
  ) * spec.embedScale;

/** The block partition of one table (ascending `[start, stop)`, no gaps). */
const partition = (tokens: number, rows: number, prefix: string) => {
  const blocks: { asset: string; start: number; stop: number }[] = [];
  for (let start = 0; start < tokens; start += rows) {
    blocks.push({
      asset: `${prefix}.${blocks.length}`,
      start,
      stop: Math.min(start + rows, tokens),
    });
  }
  return blocks;
};

const valuesBlockBytes = (
  spec: PleFixtureSpec,
  start: number,
  stop: number,
): Uint8Array<ArrayBuffer> => {
  const storage = spec.storage ?? "i8";
  const factor = packFactor({ storage });
  const bits = 8 / factor;
  const mask = (1 << bits) - 1;
  const offset = 1 << (bits - 1);
  const rowBytes = (spec.layers * spec.dim) / factor;
  const bytes = new Uint8Array(new ArrayBuffer((stop - start) * rowBytes));
  for (let row = 0; row < stop - start; row += 1) {
    for (let layer = 0; layer < spec.layers; layer += 1) {
      const base = row * rowBytes + (layer * spec.dim) / factor;
      for (let column = 0; column < spec.dim; column += 1) {
        // i8 is a plain signed byte (`writeRow` reads it as `Int8Array` directly). packed is the
        // unsigned nibble / 2 bits lifted by `+ offset` (`writeRow` brings it back).
        const value =
          storedValue(storage, start + row, layer, column) + (factor === 1 ? 0 : offset);
        bytes[base + Math.floor(column / factor)] |= (value & mask) << (bits * (column % factor));
      }
    }
  }
  return bytes;
};

const scalesBlockBytes = (
  spec: PleFixtureSpec,
  start: number,
  stop: number,
): Uint8Array<ArrayBuffer> => {
  const groups = groupsOf(spec);
  const scales = new Float32Array((stop - start) * spec.layers * groups);
  for (let row = 0; row < stop - start; row += 1) {
    for (let layer = 0; layer < spec.layers; layer += 1) {
      for (let group = 0; group < groups; group += 1) {
        // (layer, group) order within the row (the writer's layout — `karume/ple.py`).
        scales[(row * spec.layers + layer) * groups + group] = scaleOf(start + row, layer, group);
      }
    }
  }
  return new Uint8Array(scales.buffer);
};

/** An option that delays reads (used to observe aborts and concurrent reads). */
export type PleFixtureOptions = {
  /** Delays each read by one macrotask (makes the peak of concurrent reads observable). */
  readonly defer?: boolean;
};

/** Builds the synthetic index and blocks. */
export const pleFixture = (spec: PleFixtureSpec, options: PleFixtureOptions = {}): PleFixture => {
  const storage = spec.storage ?? "i8";
  const factor = packFactor({ storage });
  const valueRows = spec.valueRows ?? spec.tokens;
  const scaleRows = spec.scaleRows ?? valueRows;
  const document: Record<string, unknown> = {
    schema: spec.groupSize === undefined ? SCHEMA : GROUPED_SCHEMA,
    storage,
    tokens: spec.tokens,
    layers: spec.layers,
    dim: spec.dim,
    ...(spec.groupSize === undefined ? {} : { groupSize: spec.groupSize }),
    embedScale: spec.embedScale,
    values: {
      rowBytes: (spec.layers * spec.dim) / factor,
      blocks: partition(spec.tokens, valueRows, "ple.values"),
    },
    scales: {
      rowBytes: spec.layers * groupsOf(spec) * SCALE_BYTES,
      blocks: partition(spec.tokens, scaleRows, "ple.scales"),
    },
  };
  const index = parseGemma4PleIndex(document);
  if (index.storage === "bf16") throw new Error("fixture: a quantized spec parsed as bf16");
  const blocks = new Map<string, Uint8Array<ArrayBuffer>>();
  for (const block of index.values.blocks) {
    blocks.set(block.asset, valuesBlockBytes(spec, block.start, block.stop));
  }
  for (const block of index.scales.blocks) {
    blocks.set(block.asset, scalesBlockBytes(spec, block.start, block.stop));
  }
  return { document, index, ...recordingReader(blocks, options) };
};

/** The recording reader over a fixture's blocks (every open, every read, the in-flight peak). */
const recordingReader = (
  blocks: ReadonlyMap<string, Uint8Array<ArrayBuffer>>,
  options: PleFixtureOptions,
): Omit<PleFixtureBase, "document"> => {
  const opens: string[] = [];
  const reads: BlockRead[] = [];
  const peak = { inFlight: 0 };
  let inFlight = 0;
  const openBlock = (asset: string): AssetReader => {
    opens.push(asset);
    const bytes = blocks.get(asset);
    if (bytes === undefined) throw new Error(`fixture: unknown block '${asset}'`);
    return {
      role: asset.startsWith("ple.values") ? "ple-values" : "ple-scales",
      length: bytes.byteLength,
      read: (offset, length) => {
        if (offset < 0 || length < 0 || offset + length > bytes.byteLength) {
          throw new Error(
            `fixture: block '${asset}': range [${offset}, ${offset + length}) is outside` +
              ` its length ${bytes.byteLength}`,
          );
        }
        reads.push({ asset, offset, length });
        inFlight += 1;
        peak.inFlight = Math.max(peak.inFlight, inFlight);
        const slice = bytes.slice(offset, offset + length);
        const pending =
          options.defer === true
            ? new Promise<Uint8Array<ArrayBuffer>>((resolve) => setTimeout(() => resolve(slice), 0))
            : Promise.resolve(slice);
        // Decrement only **after it settles** (`finally`) — decrementing before returning would
        // keep this count at 1 and stop it measuring "reads issued but not yet returned".
        return pending.finally(() => {
          inFlight -= 1;
        });
      },
    };
  };
  return { blocks, openBlock, opens, reads, peak };
};

/** The dimensions of a bf16 fixture (no scales, so no group). */
export type PleBf16FixtureSpec = Omit<PleFixtureSpec, "storage" | "scaleRows" | "groupSize">;

/**
 * The value of (token, layer, column) in a bf16 fixture: a signed integer of at most 7 bits times a
 * power of two that depends on the token and the layer — exact in bf16 (8 significant bits), so the
 * stored bits are the number itself.
 *
 * MUST: depend on the token, the layer and the column (the same reason as {@link scaleOf}: a value
 * that repeats per token would let a row read from the wrong token or block pass), and cover both
 * signs and several exponents (a reader that drops the sign or the exponent bits shows).
 */
export const bf16ValueOf = (id: number, layer: number, column: number): number =>
  (((quantized(id, layer, column) * 37) % 255) - 127) * 2 ** ((id % 5) - layer - 1);

/** The gather's expected value of a bf16 fixture (the value times the embed scale, rounded to f32). */
export const expectedBf16Value = (
  spec: PleBf16FixtureSpec,
  id: number,
  layer: number,
  column: number,
): number => Math.fround(bf16ValueOf(id, layer, column) * spec.embedScale);

/** One `values` block of a bf16 fixture: little-endian bf16 bits, token-major. */
const bf16BlockBytes = (
  spec: PleBf16FixtureSpec,
  start: number,
  stop: number,
): Uint8Array<ArrayBuffer> => {
  const rowBytes = spec.layers * spec.dim * BF16_BYTES;
  const bytes = new Uint8Array(new ArrayBuffer((stop - start) * rowBytes));
  const view = new DataView(bytes.buffer);
  const bits = new Uint32Array(1);
  const value = new Float32Array(bits.buffer);
  for (let row = 0; row < stop - start; row += 1) {
    for (let layer = 0; layer < spec.layers; layer += 1) {
      for (let column = 0; column < spec.dim; column += 1) {
        value[0] = bf16ValueOf(start + row, layer, column);
        if ((bits[0] & 0xffff) !== 0) throw new Error("fixture: a value is not exact in bf16");
        const offset = row * rowBytes + (layer * spec.dim + column) * BF16_BYTES;
        view.setUint16(offset, bits[0] >>> 16, true);
      }
    }
  }
  return bytes;
};

/** The schema 5 fixture: a bf16 `values` table cut into blocks of `valueRows` tokens, no scales. */
export const pleBf16Fixture = (
  spec: PleBf16FixtureSpec,
  options: PleFixtureOptions = {},
): PleBf16Fixture => {
  const document: Record<string, unknown> = {
    schema: FLOAT_SCHEMA,
    storage: "bf16",
    tokens: spec.tokens,
    layers: spec.layers,
    dim: spec.dim,
    embedScale: spec.embedScale,
    values: {
      rowBytes: spec.layers * spec.dim * BF16_BYTES,
      blocks: partition(spec.tokens, spec.valueRows ?? spec.tokens, "ple.values"),
    },
  };
  const index = parseGemma4PleIndex(document);
  if (index.storage !== "bf16") throw new Error("fixture: a bf16 spec parsed as quantized");
  const blocks = new Map<string, Uint8Array<ArrayBuffer>>();
  for (const block of index.values.blocks) {
    blocks.set(block.asset, bf16BlockBytes(spec, block.start, block.stop));
  }
  return { document, index, ...recordingReader(blocks, options) };
};

/** The bytes of one block the index points to (the entry for the test side to build budgets). */
export const blockBytesOf = (index: Gemma4PleQuantizedIndex, table: "values" | "scales"): number =>
  (index[table].blocks[0].stop - index[table].blocks[0].start) * index[table].rowBytes;

/**
 * Folds the fixture into **the container's component surface** ({@link Gemma4PleAssetSource})
 * (the entry for driving the checks of `readGemma4PleIndex` / `assertGemma4PleAssets`).
 *
 * `patch` breaks only the declarations: passing `undefined` makes the asset "not declared by the
 * container", and passing `{role}` / `{length}` replaces only the role / logical length.
 */
export const pleComponentOf = (
  fixture: PleFixtureBase,
  patch: Readonly<
    Record<string, { readonly role?: string; readonly length?: number } | undefined>
  > = {},
): Gemma4PleAssetSource => {
  const document = new TextEncoder().encode(JSON.stringify(fixture.document));
  const declared = new Map<string, { role: string; length: number }>();
  declared.set(PLE_INDEX_ASSET, { role: "ple-index", length: document.byteLength });
  for (const [asset, bytes] of fixture.blocks) {
    declared.set(asset, {
      role: asset.startsWith("ple.values") ? "ple-values" : "ple-scales",
      length: bytes.byteLength,
    });
  }
  for (const [asset, override] of Object.entries(patch)) {
    if (override === undefined) {
      declared.delete(asset);
      continue;
    }
    const current = declared.get(asset);
    declared.set(asset, {
      role: override.role ?? current?.role ?? "ple-values",
      length: override.length ?? current?.length ?? 0,
    });
  }
  return {
    assets: Object.fromEntries([...declared].map(([asset, entry]) => [asset, entry.role])),
    asset: (asset: string): AssetReader => {
      const entry = declared.get(asset);
      if (entry === undefined) throw new Error(`fixture: undeclared asset '${asset}'`);
      if (asset === PLE_INDEX_ASSET) {
        return {
          role: entry.role,
          length: entry.length,
          read: (offset, length) =>
            Promise.resolve(document.slice(offset, offset + length) as Uint8Array<ArrayBuffer>),
        };
      }
      const reader = fixture.openBlock(asset);
      // oxlint-disable-next-line typescript/unbound-method -- `openBlock` builds `read` as an arrow property that never touches `this`
      return { role: entry.role, length: entry.length, read: reader.read };
    },
  };
};
