/**
 * PLE sidecar の shard 1 本の**読み口と検査**（ADR 0085 / 0090）。
 *
 * 読み口の面（{@link Gemma4PleShardSource}）と、そこから読んだバイト列を受理するかどうかを
 * 決める門（{@link gemma4PleShardViews}）だけを持ち、キャッシュも予算も持たない — 所有者は
 * ホスト gather の `./ple.ts` と GPU 常駐席の `./ple-gpu.ts` である。
 *
 * MUST: 全量経路（{@link readResidentShard}）・行読み経路（{@link readShardLayout}）・GPU 常駐席が
 * 通る門は**この 1 実装**であること。席ごとに検査を書くと、その席で読むときだけ別形式・別世代の
 * sidecar が通り、形も dtype も合ったまま別 token の行を引く（ADR 0085 決定 5 の沈黙誤値）。
 *
 * MUST: モジュール副作用ゼロ（横断不変条件）。
 */

import {
  parseSafetensors,
  parseSafetensorsHeader,
  type SafetensorsFile,
  safetensorsHeaderLength,
  type TensorView,
} from "@karume/runtime";
import {
  type Gemma4PleIndex,
  type Gemma4PleShard,
  HEADER_LENGTH_BYTES,
  SCALE_BYTES,
  SCHEMA,
} from "./ple-index.ts";
import { readRecord } from "../config/readers.ts";

/**
 * shard 読みへ透過するノブ（`./ple.ts` の `Gemma4PleOptions.openShard` と `Gemma4Ple.gather`）。
 *
 * MUST: **best-effort** の契約である — 読み口が無視しても壊れない（無視した実装では中断が
 * 「この shard を読み終わってから」効くだけで、値も寿命も変わらない）。生成側は run の発行前に
 * 自分で `signal` を見る（`generation/sequence.ts`）ので、中断の正しさをここへ委ねていない。
 */
export type Gemma4PleReadOptions = {
  /** この読みの中断（生成 1 回ぶんの `signal` がそのまま降りてくる）。 */
  readonly signal?: AbortSignal;
};

/**
 * PLE shard 1 本の読み口（`./ple.ts` の `Gemma4PleOptions.openShard` が返す handle — ADR 0085
 * 追記 2026-09-07）。
 *
 * 全量（{@link Gemma4PleShardSource.readAll}）は必須で、区間読み（{@link Gemma4PleShardSource.range}）
 * は**任意能力**である（hub の `openAsset` がそのまま満たす）。range を持たない読み口では
 * 従来どおり「触った shard を全量読み → LRU 常駐」だけが起きる。
 *
 * 閉じる面は持たない — 支える読み口（hub の `AssetRangeReader`）が fd も handle も保持しない
 * 契約なので、呼び手に解放の責務が生えない。
 */
export type Gemma4PleShardSource = {
  /**
   * ファイル全長（配布形の宣言 size）。
   *
   * 行の位置検査と、ヘッダ 2 段読みの clamp に使う。MUST: 実体長ではなく**宣言**長であること —
   * 実体は宣言より長いことがあり（別世代の取り違え・書きかけのコピー）、実体長で検査すると
   * 配布形の外側のバイト列が黙って読める。
   */
  readonly bytes: number;
  /** 全量を読む（返す `ArrayBuffer` は view が buffer 全体を占める — 従来の読み口と同じ契約）。 */
  readonly readAll: (options?: Gemma4PleReadOptions) => Promise<ArrayBuffer>;
  /**
   * `[offset, offset + length)` だけを読む（**任意能力**）。
   *
   * `cost` は**費用の型**（hub の `AssetRangeReader` と同じ語彙）: `"seek"` = offset に依らず
   * 小さい（位置読み / 遅延 Blob の slice）・`"scan"` = offset に比例する（本文ストリームの
   * 読み飛ばし）。行読みへ倒す行数の境目がこれで変わる（`./ple.ts` の `createGemma4Ple` の
   * 方針表）。
   *
   * MUST: `length` ちょうどを返す（短い戻りは 0 埋めの行として配られる）。
   */
  readonly range?: {
    readonly cost: "seek" | "scan";
    readonly read: (
      offset: number,
      length: number,
      options?: Gemma4PleReadOptions,
    ) => Promise<ArrayBuffer>;
  };
};

/** {@link Gemma4PleShardSource.range} の実体（任意能力なので、絞った後の型を名前で持つ）。 */
export type ShardRange = NonNullable<Gemma4PleShardSource["range"]>;

/** sidecar のテンソルキーと shard メタデータのキー（綴りの正本は `gemma4/export_product.py`）。 */
const VALUES_KEY = "values";
const SCALES_KEY = "scales";
const METADATA_KEY = "karume_ple";

/** 読み込み済みの shard 1 本（整数値と層別 scale の**生の並び**）。 */
export type ResidentShard = {
  readonly start: number;
  readonly values: Int8Array<ArrayBuffer> | Uint8Array<ArrayBuffer>;
  readonly scales: Float32Array<ArrayBuffer>;
};

/**
 * 全量経路と行読み経路が共有する検査対象（`SafetensorsFile` と `SafetensorsHeader` の共通形）。
 *
 * 全量経路は buffer 付きの `SafetensorsFile`・行読み経路は buffer を持たない
 * `SafetensorsHeader` を渡すが、資産の受理可否を決めるのはこの 2 つの表だけである。
 */
type ShardTables = {
  readonly metadata: ReadonlyMap<string, string>;
  readonly tensors: ReadonlyMap<string, TensorView>;
};

const tensorView = (tables: ShardTables, name: string, where: string): TensorView => {
  const view = tables.tensors.get(name);
  if (view === undefined) throw new Error(`${where}: テンソル '${name}' が無い`);
  return view;
};

const assertShape = (
  actual: readonly number[],
  expected: readonly number[],
  where: string,
): void => {
  if (actual.length !== expected.length || actual.some((dim, axis) => dim !== expected[axis])) {
    throw new Error(`${where}: shape [${actual.join(",")}] が [${expected.join(",")}] でない`);
  }
};

/**
 * shard のメタデータが索引と同じ資産世代を名乗っていることを見る。
 *
 * MUST: 範囲まで突き合わせる — 索引だけ差し替えた組み合わせは**形も dtype も合う**まま
 * 別 token の行を引く（ADR 0085 決定 5 の沈黙誤値そのもの）。
 */
const assertShardMetadata = (
  tables: ShardTables,
  index: Gemma4PleIndex,
  shard: Gemma4PleShard,
): void => {
  const raw = tables.metadata.get(METADATA_KEY);
  if (raw === undefined) {
    throw new Error(`${shard.file}: __metadata__.${METADATA_KEY} が無い（別形式の資産）`);
  }
  const declared = readRecord(JSON.parse(raw), `${shard.file} の ${METADATA_KEY}`);
  const mismatches = (
    [
      ["schema", index.storage === undefined ? SCHEMA : 2],
      ["tokens", index.tokens],
      ["layers", index.layers],
      ["dim", index.dim],
      ["embedScale", index.embedScale],
      ["start", shard.start],
      ["stop", shard.stop],
    ] as const
  ).filter(([key, want]) => (Object.hasOwn(declared, key) ? declared[key] : undefined) !== want);
  if (index.storage !== undefined && declared.storage !== index.storage) {
    throw new Error(`${shard.file}: karume_ple.storage が索引と違う`);
  }
  if (mismatches.length > 0) {
    throw new Error(
      `${shard.file}: ${METADATA_KEY} が索引と食い違う（` +
        mismatches
          .map(([key, want]) =>
            `${key} ${String(Object.hasOwn(declared, key) ? declared[key] : undefined)} ≠ ${want}`
          )
          .join(" / ") +
        `）— 片方だけ作り直した組み合わせ`,
    );
  }
};

/**
 * shard の表（metadata + テンソル 2 本）を検査し、`values` / `scales` の view を返す。
 *
 * MUST: 全量経路（{@link readResidentShard}）と行読み経路（{@link readShardLayout}）が通るのは
 * **この 1 実装**であること。片方だけ検査を持つと、行読みのときにだけ別形式・別世代の資産が
 * 通り、形も dtype も合ったまま別 token の行を引く（ADR 0085 決定 5 の沈黙誤値）。
 */
const assertShardTables = (
  tables: ShardTables,
  index: Gemma4PleIndex,
  shard: Gemma4PleShard,
): { readonly values: TensorView; readonly scales: TensorView } => {
  assertShardMetadata(tables, index, shard);
  const rows = shard.stop - shard.start;
  const values = tensorView(tables, VALUES_KEY, shard.file);
  const dtype = index.storage === "i2" ? "I2" : index.storage === "i4" ? "I4" : "I8";
  if (values.dtype !== dtype) {
    throw new Error(
      `${shard.file}: '${VALUES_KEY}' の格納 dtype が ${values.dtype}（${dtype} でない）`,
    );
  }
  assertShape(values.shape, [rows, index.layers, index.dim], `${shard.file} の '${VALUES_KEY}'`);
  const scales = tensorView(tables, SCALES_KEY, shard.file);
  if (scales.dtype !== "F32") {
    throw new Error(`${shard.file}: '${SCALES_KEY}' の格納 dtype が ${scales.dtype}（F32 でない）`);
  }
  assertShape(scales.shape, [rows, index.layers], `${shard.file} の '${SCALES_KEY}'`);
  return { values, scales };
};

export const readResidentShard = (
  bytes: ArrayBuffer,
  index: Gemma4PleIndex,
  shard: Gemma4PleShard,
): ResidentShard => {
  const file: SafetensorsFile = parseSafetensors(bytes);
  const { values, scales } = assertShardTables(file, index, shard);
  return {
    start: shard.start,
    values: index.storage === undefined
      ? new Int8Array(file.buffer, values.byteOffset, values.byteLength)
      : new Uint8Array(file.buffer, values.byteOffset, values.byteLength),
    scales: new Float32Array(file.buffer, scales.byteOffset, scales.byteLength / SCALE_BYTES),
  };
};

/** 行読みが使う shard 内の位置（ヘッダを 1 度だけ解いた結果 — 行数に依らず小さい）。 */
export type ShardLayout = {
  /** この shard の先頭 token id（`row = id - start`）。 */
  readonly start: number;
  /** `values` のファイル先頭からの絶対 offset（1 行 = `layers × dim` バイト連続）。 */
  readonly valuesOffset: number;
  /** `scales` の同上（1 行 = `layers × 4` バイト連続）。 */
  readonly scalesOffset: number;
};

/**
 * 区間読み 1 回（範囲の検査は読み口へ渡す**前**・長さ違いは fail loudly）。
 *
 * MUST: 宣言 `bytes` の外を要求しない。読み口が短く返す実装だと消費側は 0 埋めの行を正常な値
 * として読むので、要求と戻りの長さが違えば必ず落とす。
 */
export const readRange = async (
  range: ShardRange,
  bytes: number,
  file: string,
  offset: number,
  length: number,
  options: Gemma4PleReadOptions,
): Promise<ArrayBuffer> => {
  if (offset < 0 || length < 0 || offset + length > bytes) {
    throw new Error(
      `${file}: 区間 [${offset}, ${offset + length}) が宣言 ${bytes} バイトの外`,
    );
  }
  const read = await range.read(offset, length, options);
  if (read.byteLength !== length) {
    throw new Error(
      `${file}: 区間 [${offset}, ${offset + length}) の読みが ${read.byteLength} バイトを` +
        `返した（${length} バイト要求 — 短い戻りは 0 埋めの行として配られる）`,
    );
  }
  return read;
};

/**
 * ヘッダ区間（先頭 `8 + ヘッダ長` バイト）を 2 段で読む。
 *
 * MUST: 2 段目の読み長は宣言 `bytes` で clamp する（runtime の `safetensorsHeaderLength` の
 * doc）。壊れたヘッダ長（例 1TiB）をそのまま読み長にすると確保か読みが先に落ち、
 * `SafetensorsError` の文言に到達できない。宣言長に収まらないと分かった時点で 2 段目は
 * **読まずに**戻り、8 バイトのまま `parseSafetensorsHeader` の文言で落とす（clamp した長さで
 * 読むと 253MiB 級を無駄に読むことになる）。
 */
const readHeaderPrefix = async (
  range: ShardRange,
  bytes: number,
  file: string,
  options: Gemma4PleReadOptions,
): Promise<Uint8Array<ArrayBuffer>> => {
  // 宣言長が 8 バイトに満たない shard はここで落ちる（{@link readRange} の範囲検査 — 読み口は
  // 1 度も呼ばれない）。safetensors としては「ヘッダ長すら無い」形である。
  const head = new Uint8Array(await readRange(range, bytes, file, 0, HEADER_LENGTH_BYTES, options));
  const headerLength = safetensorsHeaderLength(head);
  const dataStart = HEADER_LENGTH_BYTES + headerLength;
  if (dataStart > bytes) return head;
  const body = new Uint8Array(
    await readRange(range, bytes, file, HEADER_LENGTH_BYTES, headerLength, options),
  );
  const prefix = new Uint8Array(new ArrayBuffer(dataStart));
  prefix.set(head);
  prefix.set(body, HEADER_LENGTH_BYTES);
  return prefix;
};

/**
 * shard の表（metadata + `values` / `scales`）を検査して view を返す**共有の門**。
 *
 * MUST: GPU 常駐席（`./ple-gpu.ts`）もこの 1 実装を通す。資産世代の突合（{@link
 * assertShardMetadata}）と dtype / shape の突合を席ごとに書くと、GPU 常駐で読むときだけ
 * 別形式・別世代の sidecar が通り、形も dtype も合ったまま別 token の行を引く（ADR 0085
 * 決定 5 の沈黙誤値）。
 *
 * NOTE: `export` はこの共有のためで、`mod.ts` / サブパス面には出さない（ADR 0008）。
 */
export const gemma4PleShardViews = (
  tables: {
    readonly metadata: ReadonlyMap<string, string>;
    readonly tensors: ReadonlyMap<string, TensorView>;
  },
  index: Gemma4PleIndex,
  shard: Gemma4PleShard,
): { readonly values: TensorView; readonly scales: TensorView } =>
  assertShardTables(tables, index, shard);

/**
 * 区間読みできる読み口から safetensors のヘッダ区間だけを 2 段で読む**共有の門**
 * （{@link readHeaderPrefix} の公開名）。
 *
 * MUST: GPU 常駐席もこの 1 実装を通す — 壊れたヘッダ長の clamp（{@link readHeaderPrefix} の
 * MUST）を 2 実装持つと、片方だけが 1TiB の読みを出して `SafetensorsError` の文言に到達
 * できなくなる。
 */
export const readGemma4PleHeaderPrefix = (
  range: NonNullable<Gemma4PleShardSource["range"]>,
  bytes: number,
  file: string,
  options: Gemma4PleReadOptions = {},
): Promise<Uint8Array<ArrayBuffer>> => readHeaderPrefix(range, bytes, file, options);

/** shard のヘッダだけを解いて行の位置を得る（**shard ごとに 1 度**）。 */
export const readShardLayout = async (
  range: ShardRange,
  bytes: number,
  index: Gemma4PleIndex,
  shard: Gemma4PleShard,
  options: Gemma4PleReadOptions,
): Promise<ShardLayout> => {
  const header = parseSafetensorsHeader(
    await readHeaderPrefix(range, bytes, shard.file, options),
    bytes,
  );
  const { values, scales } = assertShardTables(header, index, shard);
  return {
    start: shard.start,
    valuesOffset: values.byteOffset,
    scalesOffset: scales.byteOffset,
  };
};
