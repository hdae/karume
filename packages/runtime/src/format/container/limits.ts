/**
 * Size constants of the container (`krm` / `krg`) — docs/container-v1.md §1 / §4 / §10.
 *
 * MUST: every limit is kept as an **independent constant**, none derived from another (ADR 0108
 * decision 9). The old distribution put the RAM peak, the ArrayBuffer ceiling and the delivery
 * granularity all on the single `SHARD_BYTE_LIMIT`, so when the reason behind any one of them moved,
 * the other two moved with it (ADR 0090 decision 2).
 */

const MIB = 1024 * 1024;

/**
 * Fixed header length (§1): `[magic 4][u32 version][u64 graph descriptor length][u64 model
 * descriptor length]`.
 */
export const HEADER_BYTES = 24;

/** The container version this runtime reads (§1). */
export const CONTAINER_VERSION = 1;

/** Magic of the model container (`krm`) (4 ASCII characters — §1). */
export const MAGIC_MODEL = "KRMC";

/** Magic of the graph container (`krg`) (§1). */
export const MAGIC_GRAPH = "KRGC";

/** Alignment of a block's start and of the boundaries where parts are concatenated (§4.1 / §8). */
export const BLOCK_START_ALIGN = 64;

/**
 * Alignment of a block's length (§4.1 — binding-table rule ②). The writer **bakes the tail
 * padding**; two reader-side alignment branches remain (`alignF16Payload` in f16.ts and
 * `alignI8Payload` in i8.ts — ADR 0108 addendum 10).
 */
export const BLOCK_TAIL_ALIGN = 4;

/** Byte value of the padding (§4.1 — fixed). */
export const PAD_BYTE = 0x00;

/**
 * Upper bound of a block's length — 32 MiB **inclusive** (the three reasons of §4.1; an independent
 * constant, not derived from the part length).
 */
export const BLOCK_MAX_BYTES = 32 * MIB;

/** Part lengths the writer may choose (part 2 onward — §4.2). The default is the first, 256 MiB. */
export const PART_LENGTH_CHOICES: readonly number[] = [256 * MIB, 512 * MIB, 768 * MIB, 1024 * MIB];

/** Default part length (§4.2 — kept at 256 MiB by the step 3e measurements, ADR 0108 addendum 5). */
export const DEFAULT_PART_BYTES = PART_LENGTH_CHOICES[0];

/** Ceiling of the part length (every part, parts 0 / 1 included — §10). */
export const PART_MAX_BYTES = 1024 * MIB;

/** Upper bound of the part count (§10 — inherited from the old `MAX_SHARDS`). */
export const MAX_PARTS = 1024;

/** Upper bound of the block count of one container (const index and model index together — §10). */
export const MAX_BLOCKS = 65_536;

/** Upper bound of the graph count of one container (§10). */
export const MAX_GRAPHS = 64;

/** Upper bound of the byte length of the graph descriptor and of the model descriptor, each (§10). */
export const MAX_DESCRIPTOR_BYTES = 32 * MIB;

/**
 * Upper bound of the JSON nesting depth of a descriptor (§10 — bounds the parser's recursion depth
 * by declaration).
 */
export const MAX_JSON_DEPTH = 64;

/**
 * Limit that applies only to `openContainer({ kind: "bytes" })` (the whole-ArrayBuffer input of the
 * single form) (§10 — Chromium's single-ArrayBuffer limit; the measurement is recorded in
 * `packages/hub/src/fetch.ts`).
 */
export const MAX_SINGLE_CONTAINER_BYTES = 2_145_386_496;

/** Vocabulary of graph names (§2.1). */
export const GRAPH_NAME_PATTERN = /^[A-Za-z0-9._-]{1,64}$/;

/** Vocabulary of block ids (§0). */
export const BLOCK_ID_PATTERN = /^[A-Za-z0-9._:-]{1,64}$/;

/** Spelling of sha256 fields (§0 — 64 lowercase hex characters). */
export const SHA256_HEX_PATTERN = /^[0-9a-f]{64}$/;
