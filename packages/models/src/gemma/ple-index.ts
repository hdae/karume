/**
 * The **index** of PLE (per-layer embeddings), and the constants that decide a row's position,
 * length and scale (ADR [0085](../../../../docs/decisions/0085-ple-host-gather.md) /
 * [0109](../../../../docs/decisions/0109-manifest-v5-container.md) decision 4).
 *
 * The index is the model container's asset `ple_index` (role `ple-index`, canonical JSON of schema
 * 3, 4 or 5), and points at `values` and `scales` as **separate block lists** (roles `ple-values` /
 * `ple-scales`) — a schema 5 index (a bf16 table) has `values` only. One block is "a raw matrix of
 * the rows of the token range `[start, stop)` laid out without gaps", with neither a safetensors
 * header nor metadata — all a reader does is translate token → (block, row offset).
 *
 * What is gathered here is "what is decided **without reading a single byte of a block**" — the
 * conversion to the accepted form, the number of elements per byte decided by the storage dtype,
 * the resident bytes of one block, the default resident budget, and the cross-check of the index
 * against the container's asset declarations ({@link assertGemma4PleAssets}). There are three
 * readers: the owner of the host gather (`./ple.ts`), the GPU residency option (`./ple-gpu.ts`) and
 * the entry check (the admission in `./pipeline.ts`) read the same values.
 *
 * MUST: this file is the only owner of {@link SCALE_BYTES} / {@link packFactor}. The GPU residency
 * option used to hold the same values independently with a comment saying "paired with `ple.ts`",
 * but synchronization by convention comments breaks silently, and the moment they diverge it
 * becomes "read another row with shape and dtype still matching" (ADR 0085 decision 5's silent
 * wrong value).
 *
 * MUST: zero module side effects (cross-cutting invariant).
 */

import type { OpenedContainer } from "@karume/runtime";
import type { ModelComponent } from "../hub/components.ts";
import { readWholeAsset } from "../hub/asset-readers.ts";
import { assertAllowedKeys, readRecord } from "../config/readers.ts";

/**
 * The component surface {@link readGemma4PleIndex} / {@link assertGemma4PleAssets} read
 * (declarations only).
 */
export type Gemma4PleAssetSource = Pick<ModelComponent, "assets" | "asset">;

/**
 * Container asset name and role of the index (the writer's canonical source is the exporter's
 * `migrate.py`).
 */
export const PLE_INDEX_ASSET = "ple_index";
const PLE_INDEX_ROLE = "ple-index";

/**
 * Roles the `values` / `scales` blocks carry (container-v1 §2.2's "name of the interpreter on the
 * models side").
 */
const TABLE_ROLE = { values: "ple-values", scales: "ple-scales" } as const;

/** Table names (also the index's field names). */
export type Gemma4PleTableName = keyof typeof TABLE_ROLE;

const TABLE_NAMES: readonly Gemma4PleTableName[] = ["values", "scales"];

/** Storage dtype of a quantized PLE table (packed integer `values`; `scales` is always f32). */
export type Gemma4PleQuantizedStorage = "i8" | "i4" | "i2";

/**
 * One block of a table (the token range `[start, stop)` and the container asset name that serves
 * its data).
 */
export type Gemma4PleBlock = {
  readonly asset: string;
  readonly start: number;
  readonly stop: number;
};

/**
 * One table of `values` / `scales` (bytes per row + an ascending, gap-free split of the token
 * range).
 */
export type Gemma4PleTable = {
  readonly rowBytes: number;
  readonly blocks: readonly Gemma4PleBlock[];
};

/** The fields every accepted index has. */
type Gemma4PleIndexShape = {
  /** Number of token rows the index holds (= `vocab_size_per_layer_input`). */
  readonly tokens: number;
  /** Number of layers (35 for E2B). */
  readonly layers: number;
  /** Dimension per layer (256 for E2B). */
  readonly dim: number;
  /** The embed scale applied after the lookup (`hidden_size_per_layer_input ** 0.5`). */
  readonly embedScale: number;
  readonly values: Gemma4PleTable;
};

/** The accepted form of a quantized `ple_index` (schema 3 or 4). */
export type Gemma4PleQuantizedIndex = Gemma4PleIndexShape & {
  /** Storage dtype. */
  readonly storage: Gemma4PleQuantizedStorage;
  /**
   * Consecutive values of one (token, layer) row that share one f32 scale. Schema 4 spells it
   * (`groupSize`); schema 3 has one scale per row, which is the group `dim` — the accepted form
   * carries that value, so every reader decodes both schemas through one formula
   * ({@link scalesPerLayer}).
   */
  readonly groupSize: number;
  readonly scales: Gemma4PleTable;
};

/**
 * The accepted form of a bf16 `ple_index` (schema 5): the values are the checkpoint's floats, so
 * there is no scale table and no group.
 */
export type Gemma4PleFloatIndex = Gemma4PleIndexShape & {
  readonly storage: "bf16";
};

/**
 * The accepted form of `ple_index` — discriminated by `storage`, so a reader cannot touch
 * `scales` without first ruling out the bf16 table that has none.
 */
export type Gemma4PleIndex = Gemma4PleQuantizedIndex | Gemma4PleFloatIndex;

/**
 * Index version with one scale per (token, layer) row.
 *
 * MUST: read **schema 3, {@link GROUPED_SCHEMA} and {@link FLOAT_SCHEMA} only**. The old sidecar (schema 1 / 2 — the form
 * pointing at file names and safetensors tables) was retired together with its distribution, and
 * implementing both would let the combination "old index × new container" pass on shape alone (the
 * same approach as ADR 0109 decision 1's major-bump rule).
 */
export const SCHEMA = 3;

/**
 * Index version with one scale per group of `groupSize` consecutive values of a (token, layer) row
 * (the writer's source of truth: `PLE_GROUPED_INDEX_SCHEMA` of the exporter's `karume/ple.py`).
 * A `scales` row is `layers × dim / groupSize` f32 in (layer, group) order. It is the form of
 * Gemma 4's q4_0 QAT checkpoints, whose 4-bit table has one step per 32 values — schema 3 cannot
 * hold those values exactly.
 */
export const GROUPED_SCHEMA = 4;

/**
 * Index version of a bf16 table (the writer's source of truth: `PLE_FLOAT_INDEX_SCHEMA` of the
 * exporter's `karume/ple.py`): `storage` `bf16`, a `values` row is `layers × dim` little-endian
 * bf16 values (2 bytes each), and there is **no** `scales` table — neither the field nor its blocks.
 * It is the original-weights form of a BF16 checkpoint (Gemma 4's regular line — ADR 0085 addendum
 * 2026-11-04).
 */
export const FLOAT_SCHEMA = 5;

/** Bytes of one per-row scale (`scales` is f32). */
export const SCALE_BYTES = 4;

/** Bytes of one bf16 value (the `values` of a schema 5 index). */
export const BF16_BYTES = 2;

/**
 * Storage dtype → elements packed per byte (one `values` row = `layers × dim / packFactor` bytes).
 */
export const packFactor = (index: Pick<Gemma4PleQuantizedIndex, "storage">): number =>
  index.storage === "i2" ? 4 : index.storage === "i4" ? 2 : 1;

/** Bytes of one layer's values in a row (the unit the GPU-resident weight takes as one row). */
export const layerValueBytes = (index: Pick<Gemma4PleIndex, "storage" | "dim">): number => {
  const { storage } = index;
  return storage === "bf16" ? index.dim * BF16_BYTES : index.dim / packFactor({ storage });
};

/**
 * Scales of one layer of a row (1 for schema 3). A `scales` row holds `layers ×` this many f32, and
 * the scale of value `column` of `layer` is element `layer × scalesPerLayer + column / groupSize`.
 */
export const scalesPerLayer = (index: Pick<Gemma4PleQuantizedIndex, "dim" | "groupSize">): number =>
  index.dim / index.groupSize;

/**
 * The tables an index points at, as `[name, table]` in layout order — `values`, then `scales` for a
 * quantized index (a bf16 index has `values` only). Every reader that walks "all the blocks" goes
 * through this one list, so a reader cannot forget that the scale table may be absent.
 */
export const gemma4PleTables = (
  index: Gemma4PleIndex,
): readonly (readonly [Gemma4PleTableName, Gemma4PleTable])[] =>
  index.storage === "bf16"
    ? [["values", index.values]]
    : [
        ["values", index.values],
        ["scales", index.scales],
      ];

// `concat` rather than array spread: a pure annotation applies to a call only, and a bundler keeps
// an unannotated spread (it runs an iterator) in every barrel import (ADR 0037 §4 addendum 2026-11-04).
const INDEX_KEYS: readonly string[] = /* @__PURE__ */ [
  "schema",
  "storage",
  "tokens",
  "layers",
  "dim",
  "embedScale",
].concat(TABLE_NAMES);
/** Schema 4 spells exactly one key more than schema 3. */
const GROUPED_INDEX_KEYS: readonly string[] = /* @__PURE__ */ INDEX_KEYS.concat("groupSize");
/** Schema 5 spells no `scales` (and no `groupSize`). */
const FLOAT_INDEX_KEYS: readonly string[] = /* @__PURE__ */ INDEX_KEYS.filter(
  (key) => key !== "scales",
);
const TABLE_KEYS: readonly string[] = ["rowBytes", "blocks"];
const BLOCK_KEYS: readonly string[] = ["asset", "start", "stop"];

const readCount = (raw: Record<string, unknown>, key: string, where: string): number => {
  const value = Object.hasOwn(raw, key) ? raw[key] : undefined;
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 1) {
    throw new Error(`${where}.${key} ${String(value)} is not a positive integer`);
  }
  return value;
};

const readOffset = (raw: Record<string, unknown>, key: string, where: string): number => {
  const value = Object.hasOwn(raw, key) ? raw[key] : undefined;
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0) {
    throw new Error(`${where}.${key} ${String(value)} is not a non-negative integer`);
  }
  return value;
};

/**
 * Converts one table to the accepted form.
 *
 * MUST: the block ranges are an **ascending split of `[0, tokens)` with neither gaps nor
 * overlaps**. Relaxing it would let through "an index with ids that cannot be read" or "an index
 * where two blocks hold the same id", and with the latter the result depends on **which row was
 * read** (a silent wrong value).
 * MUST: `rowBytes` equals the value decided by the declaration (layers / dimension / storage). If
 * it diverges, only the row-offset multiplication quietly shifts, reading another token's row with
 * shape and dtype still matching.
 */
const readTable = (
  root: Record<string, unknown>,
  name: Gemma4PleTableName,
  expectedRowBytes: number,
  tokens: number,
  where: string,
): Gemma4PleTable => {
  const at = `${where}.${name}`;
  const table = readRecord(Object.hasOwn(root, name) ? root[name] : undefined, at);
  assertAllowedKeys(table, TABLE_KEYS, at);
  const rowBytes = readCount(table, "rowBytes", at);
  if (rowBytes !== expectedRowBytes) {
    throw new Error(
      `${at}.rowBytes ${rowBytes} differs from ${expectedRowBytes}, which the declaration determines`,
    );
  }
  if (!Array.isArray(table.blocks) || table.blocks.length === 0) {
    throw new Error(`${at}.blocks is not a non-empty array`);
  }
  const blocks: Gemma4PleBlock[] = [];
  const names = new Set<string>();
  let expected = 0;
  table.blocks.forEach((entry, position) => {
    const path = `${at}.blocks[${position}]`;
    const block = readRecord(entry, path);
    assertAllowedKeys(block, BLOCK_KEYS, path);
    const asset = block.asset;
    if (typeof asset !== "string" || asset === "") {
      throw new Error(`${path}.asset is not a non-empty string`);
    }
    if (names.has(asset)) throw new Error(`${path}.asset '${asset}' is a duplicate`);
    names.add(asset);
    const start = readOffset(block, "start", path);
    const stop = readOffset(block, "stop", path);
    if (start !== expected) {
      throw new Error(
        `${path}.start ${start} does not continue from the end ${expected} of the previous block`,
      );
    }
    if (stop <= start) throw new Error(`${path}: range [${start}, ${stop}) is empty`);
    expected = stop;
    blocks.push({ asset, start, stop });
  });
  if (expected !== tokens) {
    throw new Error(
      `${at}: the blocks total ${expected} rows, which differs from tokens ${tokens}`,
    );
  }
  return { rowBytes, blocks };
};

/**
 * Converts the `ple_index` JSON to the accepted form (unknown keys, missing fields and
 * non-contiguous ranges fail loudly).
 */
export const parseGemma4PleIndex = (raw: unknown, where = PLE_INDEX_ASSET): Gemma4PleIndex => {
  const root = readRecord(raw, where);
  if (root.schema !== SCHEMA && root.schema !== GROUPED_SCHEMA && root.schema !== FLOAT_SCHEMA) {
    throw new Error(
      `${where}.schema ${String(root.schema)} is not ${SCHEMA} / ${GROUPED_SCHEMA} / ${FLOAT_SCHEMA}` +
        ` (the old sidecar index is not read — the distribution moved to container assets)`,
    );
  }
  if (root.schema === FLOAT_SCHEMA) return parseFloatIndex(root, where);
  const grouped = root.schema === GROUPED_SCHEMA;
  // MUST: `groupSize` is an unknown key of schema 3 — a schema 3 reader would apply one group's
  // scale to the whole row, so a document that mixes the two forms is refused, not read as either.
  assertAllowedKeys(root, grouped ? GROUPED_INDEX_KEYS : INDEX_KEYS, where);
  // MUST: do not read a missing field as i8 — a schema 3 writer always spells it. Falling back to a
  // default would make an index that dropped the field show up not as "storage is missing" but as a
  // rowBytes mismatch (or pass silently for an i8 block).
  // MUST: bf16 is not a schema 3 / 4 storage — those carry a scale table a bf16 table does not have.
  const storage = Object.hasOwn(root, "storage") ? root.storage : undefined;
  if (storage !== "i8" && storage !== "i4" && storage !== "i2") {
    throw new Error(`${where}.storage ${String(storage)} is not i8 / i4 / i2`);
  }
  const { tokens, layers, dim } = readShape(root, where);
  const factor = packFactor({ storage });
  if (dim % factor !== 0) {
    throw new Error(
      `${where}.dim ${dim} is not divisible by the packing factor of storage '${storage}', ${factor}`,
    );
  }
  // MUST: schema 4 without `groupSize` fails — defaulting it to `dim` would read a grouped table
  // with one scale per row (another group's step for all but the first group of every row).
  const groupSize = grouped ? readCount(root, "groupSize", where) : dim;
  if (dim % groupSize !== 0) {
    throw new Error(
      `${where}.groupSize ${groupSize} does not divide dim ${dim} (a group never crosses a` +
        ` (token, layer) row)`,
    );
  }
  return {
    storage,
    tokens,
    layers,
    dim,
    groupSize,
    embedScale: readEmbedScale(root, where),
    values: readTable(root, "values", (layers * dim) / factor, tokens, where),
    scales: readTable(
      root,
      "scales",
      layers * scalesPerLayer({ dim, groupSize }) * SCALE_BYTES,
      tokens,
      where,
    ),
  };
};

/**
 * Converts a schema 5 document (a bf16 table) to the accepted form.
 *
 * MUST: `scales` and `groupSize` are unknown keys here — a document that spells them is not a float
 * table, and reading it as one would silently drop the scales it declares.
 * MUST: `storage` is spelled and is `bf16` — the schema says "a float table", the storage says
 * which float; a reader that assumed one would read another width with the shape still matching.
 */
const parseFloatIndex = (root: Record<string, unknown>, where: string): Gemma4PleFloatIndex => {
  assertAllowedKeys(root, FLOAT_INDEX_KEYS, where);
  const storage = Object.hasOwn(root, "storage") ? root.storage : undefined;
  if (storage !== "bf16") {
    throw new Error(
      `${where}.storage ${String(storage)} is not bf16 (schema ${FLOAT_SCHEMA} is a float table)`,
    );
  }
  const { tokens, layers, dim } = readShape(root, where);
  return {
    storage,
    tokens,
    layers,
    dim,
    embedScale: readEmbedScale(root, where),
    values: readTable(root, "values", layers * dim * BF16_BYTES, tokens, where),
  };
};

/** The three dimensions every schema spells. */
const readShape = (
  root: Record<string, unknown>,
  where: string,
): { readonly tokens: number; readonly layers: number; readonly dim: number } => ({
  tokens: readCount(root, "tokens", where),
  layers: readCount(root, "layers", where),
  dim: readCount(root, "dim", where),
});

const readEmbedScale = (root: Record<string, unknown>, where: string): number => {
  const embedScale = root.embedScale;
  if (typeof embedScale !== "number" || !Number.isFinite(embedScale) || embedScale <= 0) {
    throw new Error(`${where}.embedScale ${String(embedScale)} is not a positive finite number`);
  }
  return embedScale;
};

/** Actual bytes of one block (= the logical length the container declares for that asset). */
export const gemma4PleBlockBytes = (table: Gemma4PleTable, block: Gemma4PleBlock): number =>
  (block.stop - block.start) * table.rowBytes;

/**
 * token → block index in that table.
 *
 * The index is an ascending, gap-free split (the MUST of {@link parseGemma4PleIndex}), so a binary
 * search is enough. The value range (`0 <= id < tokens`) is checked by the caller first (the gather
 * in `./ple.ts` / the enqueue in `./ple-gpu.ts`).
 */
export const gemma4PleBlockOf = (table: Gemma4PleTable, id: number): number => {
  let low = 0;
  let high = table.blocks.length - 1;
  while (low < high) {
    const middle = (low + high) >> 1;
    if (id < table.blocks[middle].stop) high = middle;
    else low = middle + 1;
  }
  return low;
};

/** All blocks the index points at (= the host RAM full residency needs). */
export const gemma4PleTotalBytes = (index: Gemma4PleIndex): number =>
  gemma4PleTables(index).reduce((total, [, table]) => total + index.tokens * table.rowBytes, 0);

/**
 * The largest single block in the index (the budget's lower bound = below it not even one block
 * fits).
 */
export const largestBlockBytes = (index: Gemma4PleIndex): number => {
  let largest = 0;
  for (const [, table] of gemma4PleTables(index)) {
    for (const block of table.blocks) {
      largest = Math.max(largest, gemma4PleBlockBytes(table, block));
    }
  }
  return largest;
};

/**
 * Number of blocks the default resident budget is derived from (the meaning of
 * {@link defaultGemma4PleResidentBytes}).
 */
const DEFAULT_RESIDENT_BLOCKS = 2;

/**
 * The default resident budget = **two of the largest blocks** (`Gemma4PleOptions.maxResidentBytes`
 * in `./ple.ts`).
 *
 * Making "two blocks" the default as a count would make the same number mean different RAM the
 * moment the block width changes with the asset generation. Basing it on the **largest** block is
 * the only choice for which any two blocks fit the budget = the meaning of "two resident blocks"
 * holds regardless of the width (ADR 0085 addendum 2026-09-02).
 *
 * NOTE: one row of a quantized index is read from the two tables `values` and `scales`, so two is
 * also exactly the number that covers "one value block + one scale block" (a bf16 index keeps two
 * value blocks).
 */
export const defaultGemma4PleResidentBytes = (index: Gemma4PleIndex): number =>
  DEFAULT_RESIDENT_BLOCKS * largestBlockBytes(index);

/**
 * The check that the index and the container's asset declarations **match exactly** (**before a
 * single byte of a block is fetched**).
 *
 * MUST: list everything in both directions. A block in the index but not in the container does not
 * fail until the turn that first reads that token range (= mid-conversation, after a GB-scale load
 * has finished). A block in the container but not in the index is the shape "runs without ever
 * reading an asset the distribution declared" and is never detected. A length mismatch **reads
 * another token's row with shape and dtype still matching** (ADR 0085 decision 5's silent wrong
 * value).
 *
 * MUST: do not stop at the first problem. In a distribution where only one side was re-baked,
 * dozens diverge at once, so do not force a round trip per fix.
 */
export const assertGemma4PleAssets = (
  where: string,
  index: Gemma4PleIndex,
  component: Gemma4PleAssetSource,
): void => {
  const problems: string[] = [];
  const indexed = new Set<string>();
  for (const [name, table] of gemma4PleTables(index)) {
    const role = TABLE_ROLE[name];
    for (const block of table.blocks) {
      indexed.add(block.asset);
      const found = Object.hasOwn(component.assets, block.asset)
        ? component.assets[block.asset]
        : undefined;
      if (found === undefined) {
        problems.push(`${block.asset}: not declared by the container`);
        continue;
      }
      if (found !== role) {
        problems.push(`${block.asset}: role '${found}' ('${role}' is needed)`);
        continue;
      }
      const want = gemma4PleBlockBytes(table, block);
      const got = component.asset(block.asset).length;
      if (got !== want) {
        problems.push(
          `${block.asset}: logical length ${got} ([${block.start}, ${block.stop}) × ${table.rowBytes} =` +
            ` ${want} is needed)`,
        );
      }
    }
  }
  for (const [name, role] of Object.entries(component.assets)) {
    if ((role === TABLE_ROLE.values || role === TABLE_ROLE.scales) && !indexed.has(name)) {
      problems.push(`${name}: an asset with role '${role}' that the index does not point at`);
    }
  }
  if (problems.length === 0) return;
  throw new Error(
    `${where}: the PLE index and the container's assets disagree (${problems.length} problems) — ` +
      problems.join(" / "),
  );
};

/**
 * The container's PLE assets **as declared** (no byte read): every asset named
 * {@link PLE_INDEX_ASSET} or carrying one of the PLE roles, as `name (role)`.
 *
 * Empty means a PLE-free container (gemma-4-12B-it has no per-layer inputs). Whether that agrees
 * with the graph is the family admission's decision (`admitGemma4Ple` in `./admission.ts`); this
 * only knows the spellings, which stay private to this module.
 */
export const declaredGemma4PleAssets = (
  component: Pick<Gemma4PleAssetSource, "assets">,
): string[] =>
  Object.entries(component.assets)
    .filter(
      ([name, role]) =>
        name === PLE_INDEX_ASSET ||
        role === PLE_INDEX_ROLE ||
        role === TABLE_ROLE.values ||
        role === TABLE_ROLE.scales,
    )
    .map(([name, role]) => `${name} (${role})`);

/**
 * Reads the index from the container's assets into the accepted form, and checks that the blocks it
 * points at exist (the family check passed **before a single byte of a weight block is fetched** —
 * the admission slot in `./pipeline.ts`).
 *
 * The index itself is an asset of a few KB, placed in a block separate from the `values` / `scales`
 * blocks (container-v1 §4.2 — only blocks that need range reads sit alone in their own part).
 */
export const readGemma4PleIndex = async (
  where: string,
  component: Gemma4PleAssetSource,
): Promise<Gemma4PleIndex> => {
  const role = Object.hasOwn(component.assets, PLE_INDEX_ASSET)
    ? component.assets[PLE_INDEX_ASSET]
    : undefined;
  if (role === undefined) {
    throw new Error(
      `${where}: the container does not declare asset '${PLE_INDEX_ASSET}'` +
        ` (declared assets: ${Object.keys(component.assets).join(" / ") || "none"})`,
    );
  }
  if (role !== PLE_INDEX_ROLE) {
    throw new Error(
      `${where}: asset '${PLE_INDEX_ASSET}' has role '${role}' ('${PLE_INDEX_ROLE}' is needed)`,
    );
  }
  const bytes = await readWholeAsset(component.asset(PLE_INDEX_ASSET));
  const at = `${where}: asset '${PLE_INDEX_ASSET}'`;
  let text: string;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch (cause) {
    throw new Error(`${at} cannot be read as UTF-8`, { cause });
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (cause) {
    throw new Error(`${at} cannot be read as JSON`, { cause });
  }
  const index = parseGemma4PleIndex(parsed, at);
  assertGemma4PleAssets(where, index, component);
  return index;
};

/**
 * Folds an opened container (the runtime's `OpenedContainer`) into a {@link Gemma4PleAssetSource}.
 *
 * Components from the fetch surface and the full-bytes surface (`ModelComponent`) already have this
 * surface, so only **callers that open a container directly** (acceptance on real assets, and the
 * bench scripts — tools/mtp-bench, removed in 0.15) need it. Writing the role copy per caller would
 * quietly multiply copies with different spellings.
 */
export const gemma4PleAssetSource = (
  opened: Pick<OpenedContainer, "model" | "asset">,
): Gemma4PleAssetSource => ({
  assets: Object.fromEntries(
    Object.keys(opened.model?.assets ?? {}).map((name) => [name, opened.asset(name).role]),
  ),
  asset: (name) => opened.asset(name),
});
