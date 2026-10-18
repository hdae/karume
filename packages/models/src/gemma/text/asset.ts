/**
 * Acceptance and interpretation of the Gemma tokenizer asset (compiled).
 *
 * The distribution is not upstream's `tokenizer.json` (32.2 MB) but a small JSON the export recipe
 * folds it into (`tools/export-recipes/_shared/gemma_tokenizer.py` — ADR 0084 decision 1).
 * **Fetching is hub's responsibility**, so this module only receives bytes and does no I/O.
 *
 * It lives **on the family side** because the acceptance schema and the special-token set differ
 * per asset (ADR 0084 decision 2). The implementation (`src/text/bpe.ts` /
 * `src/text/detokenizer.ts`) can be shared by gemma4 and EmbeddingGemma, but **the assets cannot**:
 * the merges are bit-identical but the vocabulary differs in spelling in 6,206 slots, the added
 * vocabulary is 24 vs 6,415 entries, and the post_processor differs too (ADR 0084 Context 1).
 *
 * MUST: `fromPretrained(ref)` can point at any repo, so the asset JSON is treated as **untrusted
 * external input**. The folded asset carries no upstream JSON, so the configuration the compile
 * step checked is **declared** as {@link GemmaTokenizerSpec} and exact-matched here (an asset with
 * a different declaration is a different tokenizer, so it is rejected without being read).
 */

import { assertAllowedKeys } from "../../config/readers.ts";
import { assertUniqueLines, setUnique } from "../../text/asset-gates.ts";
import { type BpeModel, createBpeModel } from "../../text/bpe.ts";

/** Version of the asset schema (unknown versions are not read). */
const ASSET_FORMAT = "karume-gemma-tokenizer/1";

/**
 * Declaration of the upstream configuration the compile step checked (spellings 1:1 with the
 * compile script).
 */
const SPEC_NORMALIZER = "replace-space-with-metaspace";
const SPEC_PRE_TOKENIZER = "split-space-merged-with-previous";
const SPEC_DECODER = "metaspace-byte-fallback-fuse";

/**
 * Asset fields (1:1 with `asset_payload` / `spec_of` of the writer `_shared/gemma_tokenizer.py`).
 * `source` is a provenance record and the reader does not interpret it.
 */
const ASSET_KEYS = [
  "format",
  "source",
  "spec",
  "vocab",
  "mergesText",
  "mergesCount",
  "byteIds",
  "addedTokens",
  "specialIds",
  "unkId",
  "bosId",
  "eosId",
] as const;
const SPEC_KEYS = ["normalizer", "preTokenizer", "decoder", "postProcessor"] as const;

/**
 * SentencePiece's space replacement character (U+2581). Used by both the normalizer and the
 * decoder.
 */
export const METASPACE = "▁";

/**
 * The post_processor equivalent (the two forms in real assets).
 *
 * - `none` — adds nothing (gemma4; only the chat function adds `<bos>` — ADR 0084 decision 5)
 * - `bos-eos` — wraps in `<bos>` … `<eos>` (EmbeddingGemma)
 */
export type GemmaPostProcessor = "none" | "bos-eos";

/** The upstream configuration the asset declares. */
export type GemmaTokenizerSpec = {
  readonly postProcessor: GemmaPostProcessor;
};

/**
 * The interpreted asset (the output of {@link parseGemmaTokenizerAsset}; tests build it directly).
 */
export type GemmaTokenizerAssets = {
  readonly spec: GemmaTokenizerSpec;
  readonly model: BpeModel;
  /** Added vocabulary (cut out leftmost-longest **before** normalization). */
  readonly addedTokens: ReadonlyMap<string, number>;
  /** ids of the added vocabulary with `special: true` (skipped when decoding). */
  readonly specialIds: ReadonlySet<number>;
  readonly unkId: number;
  readonly bosId: number;
  readonly eosId: number;
};

const asRecord = (value: unknown, label: string): Record<string, unknown> => {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${label}: オブジェクトでない`);
  }
  return value as Record<string, unknown>;
};

/**
 * Reads a field through `Object.hasOwn` (a missing field fails as "missing" — not disguised as a
 * type message).
 */
const readField = (record: Record<string, unknown>, key: string, where: string): unknown => {
  if (!Object.hasOwn(record, key)) throw new Error(`${where}.${key}: 無い`);
  return record[key];
};

const asString = (value: unknown, label: string): string => {
  if (typeof value !== "string") throw new Error(`${label}: 文字列でない`);
  return value;
};

/** The i32 limit (ids finally reach the graph as an `Int32Array` — ADR 0009). */
const MAX_ID = 2147483647;

const asId = (value: unknown, label: string): number => {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 0 || value > MAX_ID) {
    throw new Error(`${label}: トークン id が 0..${MAX_ID} の整数でない（${String(value)}）`);
  }
  return value;
};

const asArray = (value: unknown, label: string): unknown[] => {
  if (!Array.isArray(value)) throw new Error(`${label}: 配列でない`);
  return value;
};

/** Exact-matches the declared configuration (one differing field means a different tokenizer). */
const parseSpec = (raw: unknown, label: string): GemmaTokenizerSpec => {
  const spec = asRecord(raw, label);
  assertAllowedKeys(spec, SPEC_KEYS, label);
  for (const [field, want] of [
    ["normalizer", SPEC_NORMALIZER],
    ["preTokenizer", SPEC_PRE_TOKENIZER],
    ["decoder", SPEC_DECODER],
  ] as const) {
    const got = asString(readField(spec, field, label), `${label}.${field}`);
    if (got !== want) {
      throw new Error(
        `${label}.${field}: ${JSON.stringify(want)} でない（${JSON.stringify(got)}）`,
      );
    }
  }
  const post = asString(readField(spec, "postProcessor", label), `${label}.postProcessor`);
  if (post !== "none" && post !== "bos-eos") {
    throw new Error(`${label}.postProcessor: 未知の形（${JSON.stringify(post)}）`);
  }
  return { postProcessor: post };
};

/** Lines of `left right` → id pair + rank (line number). */
const parseMerges = (raw: unknown, count: unknown, label: string): [number, number, number][] => {
  const text = asString(raw, label);
  // MUST: check the count. With merges missing, BPE only "does not merge what it cannot find" and
  // **does not fail** — the split changes and the id sequence quietly becomes something else.
  const want = asId(count, `${label}Count`);
  const lines = text === "" ? [] : text.split("\n");
  if (lines.length !== want) {
    throw new Error(`${label}: 行数 ${lines.length} が申告の ${want} と違う（資産が壊れている）`);
  }
  return lines.map((line, rank) => {
    const parts = line.split(" ");
    if (parts.length !== 2) throw new Error(`${label}: 行 ${rank} が「左 右」でない`);
    return [
      asId(Number(parts[0]), `${label}[${rank}].左`),
      asId(Number(parts[1]), `${label}[${rank}].右`),
      rank,
    ];
  });
};

const parseAddedTokens = (raw: unknown, label: string): Map<string, number> => {
  const out = new Map<string, number>();
  for (const [index, entry] of asArray(raw, label).entries()) {
    const pair = asArray(entry, `${label}[${index}]`);
    if (pair.length !== 2) throw new Error(`${label}[${index}]: [綴り, id] でない`);
    // NOTE: not bounded by the vocabulary row count — added vocabulary may be numbered **outside**
    // the vocabulary table (EmbeddingGemma's `<image_soft_token>` is id 262144 / 262,144 vocabulary
    // rows).
    setUnique(
      out,
      asString(pair[0], `${label}[${index}]`),
      asId(pair[1], `${label}[${index}]`),
      label,
    );
  }
  return out;
};

/** Asset JSON (parsed) → asset tables. */
const interpretGemmaTokenizerAsset = (raw: unknown): GemmaTokenizerAssets => {
  const asset = asRecord(raw, "tokenizer");
  const format = asString(readField(asset, "format", "tokenizer"), "tokenizer.format");
  if (format !== ASSET_FORMAT) {
    throw new Error(`tokenizer.format: ${JSON.stringify(ASSET_FORMAT)} でない（${format}）`);
  }
  // Placed after the version check — a newer version with more fields fails as "wrong version", not
  // "unknown key".
  assertAllowedKeys(asset, ASSET_KEYS, "tokenizer");
  const spec = parseSpec(readField(asset, "spec", "tokenizer"), "tokenizer.spec");

  const vocabLines = asArray(readField(asset, "vocab", "tokenizer"), "tokenizer.vocab").map(
    (token, index) => asString(token, `tokenizer.vocab[${index}]`),
  );
  // Line number = id, so a duplicate is "distribution corruption that raises no exception" (the
  // earlier id becomes unreachable).
  assertUniqueLines(vocabLines, "tokenizer.vocab");

  const byteIds = asArray(readField(asset, "byteIds", "tokenizer"), "tokenizer.byteIds").map(
    (id, index) => asId(id, `tokenizer.byteIds[${index}]`),
  );
  const model = createBpeModel({
    vocab: vocabLines.map((token, id) => [id, token] as const),
    merges: parseMerges(
      readField(asset, "mergesText", "tokenizer"),
      readField(asset, "mergesCount", "tokenizer"),
      "tokenizer.mergesText",
    ),
    byteIds,
  });

  const addedTokens = parseAddedTokens(
    readField(asset, "addedTokens", "tokenizer"),
    "tokenizer.addedTokens",
  );
  // MUST: the added vocabulary is kept consistent in both directions, "spelling → id" and "id →
  // spelling". Duplicate spellings are rejected by `parseAddedTokens`, but duplicate ids and an id
  // in the vocabulary used with a different spelling raise no exception — encode returns that id
  // and decode returns the added-vocabulary spelling, so asset corruption turns into a different
  // token at inference. Numbering outside the vocabulary (EmbeddingGemma) is accepted as before.
  const addedIds = new Set<number>();
  for (const [token, id] of addedTokens) {
    if (addedIds.has(id)) {
      throw new Error(`tokenizer.addedTokens: id ${id} が 2 つの綴りに割り当てられている`);
    }
    addedIds.add(id);
    const base = model.tokenOf.get(id);
    if (base !== undefined && base !== token) {
      throw new Error(
        `tokenizer.addedTokens: id ${id} の綴り ${JSON.stringify(token)} が語彙の` +
          ` ${JSON.stringify(base)} と違う`,
      );
    }
  }
  const specialIds = new Set<number>();
  for (const [index, id] of asArray(
    readField(asset, "specialIds", "tokenizer"),
    "tokenizer.specialIds",
  ).entries()) {
    const specialId = asId(id, `tokenizer.specialIds[${index}]`);
    // Special tokens are a subset of the added vocabulary (the reference's `is_special_token` also
    // looks at the added-vocabulary side). Letting a stray id through makes "a token that should be
    // skipped appear in the text / one that should appear disappear".
    if (!addedIds.has(specialId)) {
      throw new Error(`tokenizer.specialIds[${index}]: id ${specialId} が追加語彙に無い`);
    }
    specialIds.add(specialId);
  }

  const vocabId = (field: string): number => {
    const id = asId(readField(asset, field, "tokenizer"), `tokenizer.${field}`);
    if (!model.tokenOf.has(id)) throw new Error(`tokenizer.${field}: id ${id} が語彙に無い`);
    return id;
  };
  return {
    spec,
    model,
    addedTokens,
    specialIds,
    unkId: vocabId("unkId"),
    bosId: vocabId("bosId"),
    eosId: vocabId("eosId"),
  };
};

/** Builds the asset tables from the fetched bytes (the manifest's `tokenizer`). */
export const parseGemmaTokenizerAsset = (bytes: Uint8Array<ArrayBuffer>): GemmaTokenizerAssets => {
  let text: string;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch (cause) {
    throw new Error("tokenizer: UTF-8 として読めない", { cause });
  }
  let json: unknown;
  try {
    json = JSON.parse(text);
  } catch (cause) {
    throw new Error("tokenizer: JSON として読めない", { cause });
  }
  return interpretGemmaTokenizerAsset(json);
};
