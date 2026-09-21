/**
 * PLE sidecar の**索引**（`ple.json`）の codec と、行の位置・長さ・scale を決める定数
 * （ADR 0085 / 0097）。
 *
 * ここに集めてあるのは「バイト列を 1 つも読まずに索引だけで決まること」である — 受理形への
 * 変換、格納 dtype から決まる 1 バイトあたりの要素数、shard 1 本ぶんの常駐バイト、既定の
 * 常駐予算。読み手は 3 つで、ホスト gather の所有者（`./ple.ts`）・shard の読取りと検査
 * （`./ple-shard.ts`）・GPU 常駐席（`./ple-gpu.ts`）が同じ値を引く。
 *
 * MUST: {@link SCALE_BYTES} / {@link HEADER_LENGTH_BYTES} / {@link packFactor} の所有者は
 * この 1 本であること。以前は GPU 常駐席が「`ple.ts` と対」というコメント付きで同じ値を
 * 独立に持っていたが、規約コメントによる同期は壊れても静かで、ずれた瞬間に「形も dtype も
 * 合ったまま別の行を引く」形になる（ADR 0085 決定 5 の沈黙誤値）。
 *
 * MUST: モジュール副作用ゼロ（横断不変条件）。
 */

import { assertAllowedKeys, readRecord } from "../config/readers.ts";

/** sidecar shard 1 本の受け持つ token 範囲（`[start, stop)`）。 */
export type Gemma4PleShard = {
  /** 配布形の相対ファイル名（読み手が `./ple.ts` の `Gemma4PleOptions.openShard` へ渡す綴り）。 */
  readonly file: string;
  readonly start: number;
  readonly stop: number;
};

/** `ple.json` の受理形（書き手の正本は `gemma4/export_product.py`）。 */
export type Gemma4PleIndex = {
  /** schema 2 の packed 格納。省略は従来の schema 1 / I8（ADR 0097）。 */
  readonly storage?: "i2" | "i4";
  /** sidecar が持つ token 行数（= `vocab_size_per_layer_input`）。 */
  readonly tokens: number;
  /** 層数（E2B は 35）。 */
  readonly layers: number;
  /** 層当たりの次元（E2B は 256）。 */
  readonly dim: number;
  /** lookup 後に掛かる embed scale（`hidden_size_per_layer_input ** 0.5`）。 */
  readonly embedScale: number;
  /** token 範囲の昇順・隙間なしの分割（先頭は 0・末尾は `tokens`）。 */
  readonly shards: readonly Gemma4PleShard[];
};

/** 索引と shard メタデータの版（知らない版を黙って読まない）。 */
export const SCHEMA = 1;

const INDEX_KEYS: readonly string[] = ["schema", "tokens", "layers", "dim", "embedScale", "shards"];
const SHARD_KEYS: readonly string[] = ["file", "start", "stop"];

const readCount = (raw: Record<string, unknown>, key: string, where: string): number => {
  const value = Object.hasOwn(raw, key) ? raw[key] : undefined;
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 1) {
    throw new Error(`${where}.${key} ${String(value)} が 1 以上の整数でない`);
  }
  return value;
};

const readOffset = (raw: Record<string, unknown>, key: string, where: string): number => {
  const value = Object.hasOwn(raw, key) ? raw[key] : undefined;
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0) {
    throw new Error(`${where}.${key} ${String(value)} が 0 以上の整数でない`);
  }
  return value;
};

/**
 * `ple.json` を受理形へ落とす（未知キー・欠け・不連続な範囲は fail loudly）。
 *
 * MUST: shard の範囲は `[0, tokens)` の**隙間も重なりも無い昇順分割**であること。緩めると
 * 「引けない id がある索引」や「2 本が同じ id を持つ索引」が通り、後者は**どちらの行を
 * 引いたか**で結果が変わる（沈黙誤値）。
 */
export const parseGemma4PleIndex = (raw: unknown, where = "ple.json"): Gemma4PleIndex => {
  const root = readRecord(raw, where);
  assertAllowedKeys(root, root.schema === 2 ? [...INDEX_KEYS, "storage"] : INDEX_KEYS, where);
  if (root.schema !== SCHEMA && root.schema !== 2) {
    throw new Error(`${where}.schema ${String(root.schema)} が 1 / 2 でない`);
  }
  let storage: "i2" | "i4" | undefined;
  if (root.schema === 2) {
    const value = root.storage;
    if (value !== "i2" && value !== "i4") throw new Error(`${where}.storage は i2 / i4 が必要`);
    storage = value;
  }
  const tokens = readCount(root, "tokens", where);
  const layers = readCount(root, "layers", where);
  const dim = readCount(root, "dim", where);
  if (storage !== undefined && dim % 16 !== 0) {
    throw new Error(`${where}.dim は packed で16の倍数が必要`);
  }
  const embedScale = root.embedScale;
  if (typeof embedScale !== "number" || !Number.isFinite(embedScale) || embedScale <= 0) {
    throw new Error(`${where}.embedScale ${String(embedScale)} が正の有限数でない`);
  }
  if (!Array.isArray(root.shards) || root.shards.length === 0) {
    throw new Error(`${where}.shards が非空の配列でない`);
  }
  const shards: Gemma4PleShard[] = [];
  const files = new Set<string>();
  let expected = 0;
  root.shards.forEach((entry, position) => {
    const at = `${where}.shards[${position}]`;
    const shard = readRecord(entry, at);
    assertAllowedKeys(shard, SHARD_KEYS, at);
    const file = shard.file;
    if (typeof file !== "string" || file === "") throw new Error(`${at}.file が非空の文字列でない`);
    if (files.has(file)) throw new Error(`${at}.file '${file}' が重複している`);
    files.add(file);
    const start = readOffset(shard, "start", at);
    const stop = readOffset(shard, "stop", at);
    if (start !== expected) {
      throw new Error(`${at}.start ${start} が直前の shard の末尾 ${expected} と連続しない`);
    }
    if (stop <= start) throw new Error(`${at}: 範囲 [${start}, ${stop}) が空`);
    expected = stop;
    shards.push({ file, start, stop });
  });
  if (expected !== tokens) {
    throw new Error(`${where}: shard の合計 ${expected} 行が tokens ${tokens} と違う`);
  }
  return { tokens, layers, dim, embedScale, shards, ...(storage === undefined ? {} : { storage }) };
};

/**
 * per-row scale 1 個ぶんのバイト数（`scales` は f32 — `./ple-shard.ts` の `readResidentShard` の
 * dtype 門と対）。
 */
export const SCALE_BYTES = 4;

/** safetensors 先頭のヘッダ長欄（u64 LE）— ヘッダ 2 段読みの 1 段目の長さ。 */
export const HEADER_LENGTH_BYTES = 8;

/** 格納 dtype → 1 バイトに詰まる要素数（`values` の 1 行 = `layers × dim / packFactor` バイト）。 */
export const packFactor = (index: Gemma4PleIndex): number =>
  index.storage === "i2" ? 4 : index.storage === "i4" ? 2 : 1;

/** 既定の常駐予算を導く shard 本数（{@link defaultGemma4PleResidentBytes} の意味づけ）。 */
const DEFAULT_RESIDENT_SHARDS = 2;

/**
 * shard 1 本を常駐させたときのホスト RAM（i8 `values` + f32 `scales`）。
 *
 * 索引だけで決まる（バイト列を読む前に分かる）ので、予算の検査も LRU の追い出しも取得の完了を
 * 待たずに判定できる。
 */
export const gemma4PleShardBytes = (index: Gemma4PleIndex, shard: Gemma4PleShard): number =>
  (shard.stop - shard.start) * index.layers * (index.dim / packFactor(index) + SCALE_BYTES);

/** 索引中で最も大きい shard 1 本ぶん（予算の下限 = これを割ると 1 本も載せられない）。 */
export const largestShardBytes = (index: Gemma4PleIndex): number =>
  index.shards.reduce((largest, shard) => Math.max(largest, gemma4PleShardBytes(index, shard)), 0);

/**
 * 常駐予算の既定 = **最も大きい shard 2 本ぶん**（`./ple.ts` の
 * `Gemma4PleOptions.maxResidentBytes`）。
 *
 * 「2 本」を本数のまま既定にすると、資産世代で shard 幅が変わった瞬間に同じ数字が別の RAM を
 * 意味する（実例: shard 上限 1GiB 世代の 3 本 = 1 本 758MiB → 256MiB 世代の 9 本 = 1 本 253MiB）。
 * **最大** shard を基準に取るのは、どの 2 本を掴んでも予算に収まる = 「2 本常駐」の意味が幅に
 * 依らず保たれる唯一の取り方だからである（ADR 0085 追記 2026-09-02）。
 */
export const defaultGemma4PleResidentBytes = (index: Gemma4PleIndex): number =>
  DEFAULT_RESIDENT_SHARDS * largestShardBytes(index);
