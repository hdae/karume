/**
 * コンテナ（`krm` / `krg`）の寸法定数 — docs/container-v1.md §1 / §4 / §10。
 *
 * MUST: 上限はそれぞれ**独立した定数**として持ち、一方を他方から派生させない（ADR 0108
 * 決定 9）。旧配布形は RAM ピークも ArrayBuffer の天井も配信粒度も `SHARD_BYTE_LIMIT` 1 本に
 * 載せていて、どれか 1 つの根拠が動くと他の 2 つが巻き添えで動いた（ADR 0090 決定 2）。
 */

const MIB = 1024 * 1024;

/** ヘッダの固定長（§1）: `[magic 4][u32 版][u64 グラフ記述長][u64 モデル記述長]`。 */
export const HEADER_BYTES = 24;

/** このランタイムが読むコンテナ版（§1）。 */
export const CONTAINER_VERSION = 1;

/** モデル容器（`krm`）の magic（ASCII 4 文字 — §1）。 */
export const MAGIC_MODEL = "KRMC";

/** グラフ容器（`krg`）の magic（§1）。 */
export const MAGIC_GRAPH = "KRGC";

/** block の先頭と part の連結境界が満たす整列（§4.1 / §8）。 */
export const BLOCK_START_ALIGN = 64;

/**
 * block 長が満たす整列（§4.1 — 束縛表の規則②）。末尾の詰め物は**書き手が焼く**ので、読み手側の
 * 末尾整列分岐は存在しない。
 */
export const BLOCK_TAIL_ALIGN = 4;

/** 詰め物のバイト値（§4.1 — 固定）。 */
export const PAD_BYTE = 0x00;

/** block 長の上限 — 32 MiB **以下**（§4.1 の根拠 3 点・part 長からの派生ではない独立定数）。 */
export const BLOCK_MAX_BYTES = 32 * MIB;

/** 書き手が選べる part 長（part 2 以降 — §4.2）。既定は先頭の 256 MiB。 */
export const PART_LENGTH_CHOICES: readonly number[] = [256 * MIB, 512 * MIB, 768 * MIB, 1024 * MIB];

/** part 長の既定（§4.2 — exporter の既定を動かすのは段 3 の検収後）。 */
export const DEFAULT_PART_BYTES = PART_LENGTH_CHOICES[0];

/** part 長の天井（part 0 / 1 を含む全 part — §10）。 */
export const PART_MAX_BYTES = 1024 * MIB;

/** part 件数の上限（§10 — 旧 `MAX_SHARDS` を継承）。 */
export const MAX_PARTS = 1024;

/** 1 コンテナの block 件数の上限（const 目次とモデル目次の合計 — §10）。 */
export const MAX_BLOCKS = 65_536;

/** 1 コンテナのグラフ件数の上限（§10）。 */
export const MAX_GRAPHS = 64;

/** グラフ記述 / モデル記述それぞれのバイト長の上限（§10）。 */
export const MAX_DESCRIPTOR_BYTES = 32 * MIB;

/** descriptor の JSON の入れ子深さの上限（§10 — parse の再帰深さを宣言で閉じる）。 */
export const MAX_JSON_DEPTH = 64;

/**
 * `fromContainer(bytes)`（単一形の全量 ArrayBuffer の口）だけに掛かる上限（§10 —
 * Chromium の単一 ArrayBuffer 上限。実測は `packages/hub/src/fetch.ts` に記録）。
 */
export const MAX_SINGLE_CONTAINER_BYTES = 2_145_386_496;

/** グラフ名の語彙（§2.1）。 */
export const GRAPH_NAME_PATTERN = /^[A-Za-z0-9._-]{1,64}$/;

/** block id の語彙（§0）。 */
export const BLOCK_ID_PATTERN = /^[A-Za-z0-9._:-]{1,64}$/;

/** sha256 欄の綴り（§0 — 小文字 16 進 64 文字）。 */
export const SHA256_HEX_PATTERN = /^[0-9a-f]{64}$/;
