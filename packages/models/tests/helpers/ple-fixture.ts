/**
 * 合成の PLE 索引と block（`src/gemma/ple.ts` / `ple-gpu.ts` の門が共有するフィクスチャ）。
 *
 * 実資産を要さない — 索引は schema 3 の文書を組み立てて `parseGemma4PleIndex` に通し、block は
 * 「token 区間ぶんの行を隙間なく並べた生の行列」を決定的な値で作る（容器の資産と同じ形 —
 * ADR 0109 決定 4）。値は `quantized` / `scaleOf` から決まるので、逆量子化の期待値
 * （{@link expectedValue}）がテスト側で閉じる。
 *
 * 読み口（{@link PleFixture.openBlock}）は**呼び出しを全部記録する**。方針表の分岐（全量読み /
 * 行読み / 常駐 hit）は「どの block の・どの区間が・何回読まれたか」でしか観測できないためで、
 * 同時発行のピーク（`peak.inFlight`）も持つ — 1 read = 1 fd を取る取得元では、上限が効いている
 * ことは回数ではなくピークでしか見えない。
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

/** 合成索引の寸法。 */
export type PleFixtureSpec = {
  readonly storage?: Gemma4PleQuantizedStorage;
  readonly tokens: number;
  readonly layers: number;
  readonly dim: number;
  /** 2 冪（f32 の乗算が厳密 — ADR 0085 決定 4 と同じ性質を合成側でも保つ）。 */
  readonly embedScale: number;
  /** `values` の block 1 本が持つ token 行数（既定 = `tokens` = 1 本）。 */
  readonly valueRows?: number;
  /** `scales` の block 1 本が持つ token 行数（既定 = `valueRows`）。 */
  readonly scaleRows?: number;
  /**
   * Values per scale of a (token, layer) row: set = a schema 4 index (`groupSize` spelled, a scales
   * row of `layers × dim / groupSize` f32 in (layer, group) order); absent = schema 3.
   */
  readonly groupSize?: number;
};

/** 読み口が受けた区間 1 本。 */
export type BlockRead = {
  readonly asset: string;
  readonly offset: number;
  readonly length: number;
};

/** What both fixture forms share (the document, the blocks and the recording reader). */
export type PleFixtureBase = {
  /** 索引の生 JSON（`parseGemma4PleIndex` の門を直接叩くときの元）。 */
  readonly document: Record<string, unknown>;
  /** block 名 → 実体。 */
  readonly blocks: ReadonlyMap<string, Uint8Array<ArrayBuffer>>;
  /** `createGemma4Ple` / `createGemma4PleResident` の `openBlock`。 */
  readonly openBlock: (asset: string) => AssetReader;
  /** 開かれた読み口の本数（`asset()` の呼び出し回数）。 */
  readonly opens: string[];
  readonly reads: BlockRead[];
  readonly peak: { inFlight: number };
};

/** A quantized fixture (schema 3 / 4). */
export type PleFixture = PleFixtureBase & { readonly index: Gemma4PleQuantizedIndex };

/** A bf16 fixture (schema 5 — {@link pleBf16Fixture}). */
export type PleBf16Fixture = PleFixtureBase & { readonly index: Gemma4PleFloatIndex };

/** i8 の値は `id * 10 + 層 * 2 + 列`（packed では格納幅で切り詰める）。 */
export const quantized = (id: number, layer: number, column: number): number =>
  id * 10 + layer * 2 + column;

/**
 * per-row scale は `(id+1) / 2^(層+1)`。
 *
 * MUST: **token にも依存させる**。層だけの関数にすると scale 表は 1 token 行ぶんの繰り返しに
 * なり、行の入れ替え・巡回・block の取り違えが「同じバイト列」になって、並びを見る主張が
 * どれも無感になる（scale がずれると形も dtype も合ったまま別 token で逆量子化する沈黙誤値）。
 * 分子は整数・分母は 2 冪なので f32 で厳密（逆量子化の期待値がテスト側で閉じる条件）。
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

/** 格納 dtype で切り詰めた後の量子化値（packed は符号付き範囲へ折り返す）。 */
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

/** 逆量子化 → embed scale の 2 段（`writeRow` と同じ順序・同じ丸め点）。 */
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

/** 表 1 本ぶんの block 分割（`[start, stop)` の昇順・隙間なし）。 */
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
        // i8 は素の符号付きバイト（`writeRow` は `Int8Array` でそのまま読む）。packed は
        // `+ offset` で持ち上げた無符号のニブル / 2bit（`writeRow` が引き戻す）。
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

/** 読みを遅らせる口（中断と同時発行の観測に使う）。 */
export type PleFixtureOptions = {
  /** 各読みを 1 マクロタスク遅らせる（同時発行のピークが観測できるようになる）。 */
  readonly defer?: boolean;
};

/** 合成の索引と block を作る。 */
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
    if (bytes === undefined) throw new Error(`fixture: 知らない block '${asset}'`);
    return {
      role: asset.startsWith("ple.values") ? "ple-values" : "ple-scales",
      length: bytes.byteLength,
      read: (offset, length) => {
        if (offset < 0 || length < 0 || offset + length > bytes.byteLength) {
          throw new Error(
            `fixture: block '${asset}' の区間 [${offset}, ${offset + length}) が` +
              ` 長さ ${bytes.byteLength} の外`,
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
        // 減らすのは**解決した後**（`finally`）— 返す前に減らすと、この数は常に 1 になり
        // 「撒いたまま返ってきていない本数」を測らなくなる。
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

/** 索引が指す block 1 本ぶんのバイト数（テスト側が予算を組み立てるための口）。 */
export const blockBytesOf = (index: Gemma4PleQuantizedIndex, table: "values" | "scales"): number =>
  (index[table].blocks[0].stop - index[table].blocks[0].start) * index[table].rowBytes;

/**
 * フィクスチャを**容器の部品の面**（{@link Gemma4PleAssetSource}）へ畳む
 * （`readGemma4PleIndex` / `assertGemma4PleAssets` の門を叩くための口）。
 *
 * `patch` は宣言だけを壊す口で、`undefined` を渡すとその資産を「容器が宣言していない」形に、
 * `{role}` / `{length}` を渡すと役割・論理長だけを差し替えた形にする。
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
      if (entry === undefined) throw new Error(`fixture: 未宣言の資産 '${asset}'`);
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
