/**
 * PLE（per-layer embeddings）の**索引**と、行の位置・長さ・scale を決める定数
 * （ADR [0085](../../../../docs/decisions/0085-ple-host-gather.md) /
 * [0109](../../../../docs/decisions/0109-manifest-v5-container.md) 決定 4）。
 *
 * 索引はモデル容器の資産 `ple_index`（役割 `ple-index`・schema 3 の正準 JSON）で、`values` と
 * `scales` を**別々の block 列**として指す（役割 `ple-values` / `ple-scales`）。block 1 本は
 * 「token 区間 `[start, stop)` ぶんの行を隙間なく並べた生の行列」で、safetensors ヘッダも
 * メタデータも持たない — 読み手がするのは token → (block, 行 offset) の翻訳だけである。
 *
 * ここに集めてあるのは「**block のバイト列を 1 つも読まずに**決まること」— 受理形への変換、
 * 格納 dtype から決まる 1 バイトあたりの要素数、block 1 本ぶんの常駐バイト、既定の常駐予算、
 * そして索引と容器の資産宣言の突合（{@link assertGemma4PleAssets}）である。読み手は 3 つで、
 * ホスト gather の所有者（`./ple.ts`）・GPU 常駐席（`./ple-gpu.ts`）・入口の門
 * （`./pipeline.ts` の admission）が同じ値を引く。
 *
 * MUST: {@link SCALE_BYTES} / {@link packFactor} の所有者はこの 1 本であること。以前は GPU
 * 常駐席が「`ple.ts` と対」というコメント付きで同じ値を独立に持っていたが、規約コメントによる
 * 同期は壊れても静かで、ずれた瞬間に「形も dtype も合ったまま別の行を引く」形になる
 * （ADR 0085 決定 5 の沈黙誤値）。
 *
 * MUST: モジュール副作用ゼロ（横断不変条件）。
 */

import type { OpenedContainer } from "@karume/runtime";
import type { ModelComponent } from "../hub/components.ts";
import { readWholeAsset } from "../hub/asset-readers.ts";
import { assertAllowedKeys, readRecord } from "../config/readers.ts";

/** {@link readGemma4PleIndex} / {@link assertGemma4PleAssets} が読む部品の面（宣言だけ）。 */
export type Gemma4PleAssetSource = Pick<ModelComponent, "assets" | "asset">;

/** 索引が載る容器の資産名と役割（書き手の正本は exporter の `migrate.py`）。 */
export const PLE_INDEX_ASSET = "ple_index";
const PLE_INDEX_ROLE = "ple-index";

/** `values` / `scales` の block が名乗る役割（container-v1 §2.2 の「models 側の解釈者名」）。 */
const TABLE_ROLE = { values: "ple-values", scales: "ple-scales" } as const;

/** 表の名前（索引の欄名でもある）。 */
export type Gemma4PleTableName = keyof typeof TABLE_ROLE;

const TABLE_NAMES: readonly Gemma4PleTableName[] = ["values", "scales"];

/** PLE の格納 dtype（`values` の詰め方 — `scales` は常に f32）。 */
export type Gemma4PleStorage = "i8" | "i4" | "i2";

/** 表 1 本の block（token 区間 `[start, stop)` と、その実体を配る容器の資産名）。 */
export type Gemma4PleBlock = {
  readonly asset: string;
  readonly start: number;
  readonly stop: number;
};

/** `values` / `scales` の 1 表ぶん（1 行のバイト数 + token 区間の昇順・隙間なし分割）。 */
export type Gemma4PleTable = {
  readonly rowBytes: number;
  readonly blocks: readonly Gemma4PleBlock[];
};

/** `ple_index`（schema 3）の受理形。 */
export type Gemma4PleIndex = {
  /** 格納 dtype（省略された索引は i8 と読む）。 */
  readonly storage: Gemma4PleStorage;
  /** 索引が持つ token 行数（= `vocab_size_per_layer_input`）。 */
  readonly tokens: number;
  /** 層数（E2B は 35）。 */
  readonly layers: number;
  /** 層当たりの次元（E2B は 256）。 */
  readonly dim: number;
  /** lookup 後に掛かる embed scale（`hidden_size_per_layer_input ** 0.5`）。 */
  readonly embedScale: number;
  readonly values: Gemma4PleTable;
  readonly scales: Gemma4PleTable;
};

/**
 * 索引の版。
 *
 * MUST: **schema 3 だけ**を読む。旧 sidecar（schema 1 / 2 — ファイル名と safetensors の表を
 * 指す形）は配布形ごと退役しており、両読みを実装すると「旧索引 × 新容器」の組み合わせが
 * 形だけ通る（ADR 0109 決定 1 の major 繰り上げ規則と同じ流儀）。
 */
export const SCHEMA = 3;

/** per-row scale 1 個ぶんのバイト数（`scales` は f32）。 */
export const SCALE_BYTES = 4;

/** 格納 dtype → 1 バイトに詰まる要素数（`values` の 1 行 = `layers × dim / packFactor` バイト）。 */
export const packFactor = (index: Pick<Gemma4PleIndex, "storage">): number =>
  index.storage === "i2" ? 4 : index.storage === "i4" ? 2 : 1;

const INDEX_KEYS: readonly string[] = [
  "schema",
  "storage",
  "tokens",
  "layers",
  "dim",
  "embedScale",
  ...TABLE_NAMES,
];
const TABLE_KEYS: readonly string[] = ["rowBytes", "blocks"];
const BLOCK_KEYS: readonly string[] = ["asset", "start", "stop"];

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
 * 表 1 本を受理形へ落とす。
 *
 * MUST: block の範囲は `[0, tokens)` の**隙間も重なりも無い昇順分割**であること。緩めると
 * 「引けない id がある索引」や「2 本が同じ id を持つ索引」が通り、後者は**どちらの行を
 * 引いたか**で結果が変わる（沈黙誤値）。
 * MUST: `rowBytes` は宣言（層数 / 次元 / 格納）から決まる値と一致すること。ここがずれると
 * 行 offset の掛け算だけが静かにずれ、形も dtype も合ったまま別 token の行を引く。
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
      `${at}.rowBytes ${rowBytes} が宣言から決まる ${expectedRowBytes} と違う`,
    );
  }
  if (!Array.isArray(table.blocks) || table.blocks.length === 0) {
    throw new Error(`${at}.blocks が非空の配列でない`);
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
      throw new Error(`${path}.asset が非空の文字列でない`);
    }
    if (names.has(asset)) throw new Error(`${path}.asset '${asset}' が重複している`);
    names.add(asset);
    const start = readOffset(block, "start", path);
    const stop = readOffset(block, "stop", path);
    if (start !== expected) {
      throw new Error(`${path}.start ${start} が直前の block の末尾 ${expected} と連続しない`);
    }
    if (stop <= start) throw new Error(`${path}: 範囲 [${start}, ${stop}) が空`);
    expected = stop;
    blocks.push({ asset, start, stop });
  });
  if (expected !== tokens) {
    throw new Error(`${at}: block の合計 ${expected} 行が tokens ${tokens} と違う`);
  }
  return { rowBytes, blocks };
};

/** `ple_index` の JSON を受理形へ落とす（未知キー・欠け・不連続な範囲は fail loudly）。 */
export const parseGemma4PleIndex = (raw: unknown, where = PLE_INDEX_ASSET): Gemma4PleIndex => {
  const root = readRecord(raw, where);
  assertAllowedKeys(root, INDEX_KEYS, where);
  if (root.schema !== SCHEMA) {
    throw new Error(
      `${where}.schema ${String(root.schema)} が ${SCHEMA} でない` +
        `（旧 sidecar の索引は読まない — 配布形は容器の資産へ移った）`,
    );
  }
  const declared = Object.hasOwn(root, "storage") ? root.storage : undefined;
  if (
    declared !== undefined && declared !== "i8" && declared !== "i4" && declared !== "i2"
  ) {
    throw new Error(`${where}.storage ${String(declared)} が i8 / i4 / i2 でない`);
  }
  const storage: Gemma4PleStorage = declared ?? "i8";
  const tokens = readCount(root, "tokens", where);
  const layers = readCount(root, "layers", where);
  const dim = readCount(root, "dim", where);
  const factor = packFactor({ storage });
  if (dim % factor !== 0) {
    throw new Error(`${where}.dim ${dim} が格納 '${storage}' の詰め数 ${factor} で割り切れない`);
  }
  const embedScale = root.embedScale;
  if (typeof embedScale !== "number" || !Number.isFinite(embedScale) || embedScale <= 0) {
    throw new Error(`${where}.embedScale ${String(embedScale)} が正の有限数でない`);
  }
  return {
    storage,
    tokens,
    layers,
    dim,
    embedScale,
    values: readTable(root, "values", layers * dim / factor, tokens, where),
    scales: readTable(root, "scales", layers * SCALE_BYTES, tokens, where),
  };
};

/** block 1 本の実体バイト数（= 容器がその資産に宣言する論理長）。 */
export const gemma4PleBlockBytes = (table: Gemma4PleTable, block: Gemma4PleBlock): number =>
  (block.stop - block.start) * table.rowBytes;

/**
 * token → その表の block 添字。
 *
 * 索引は昇順の隙間なし分割（{@link parseGemma4PleIndex} の MUST）なので二分探索でよい。
 * 値域（`0 <= id < tokens`）は呼び手が先に見る（`./ple.ts` の gather / `./ple-gpu.ts` の enqueue）。
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

/** 索引が指す block 全部ぶん（= 全量常駐に要るホスト RAM）。 */
export const gemma4PleTotalBytes = (index: Gemma4PleIndex): number =>
  index.tokens * (index.values.rowBytes + index.scales.rowBytes);

/** 索引中で最も大きい block 1 本（予算の下限 = これを割ると 1 本も載せられない）。 */
export const largestBlockBytes = (index: Gemma4PleIndex): number => {
  let largest = 0;
  for (const name of TABLE_NAMES) {
    const table = index[name];
    for (const block of table.blocks) {
      largest = Math.max(largest, gemma4PleBlockBytes(table, block));
    }
  }
  return largest;
};

/** 既定の常駐予算を導く block 本数（{@link defaultGemma4PleResidentBytes} の意味づけ）。 */
const DEFAULT_RESIDENT_BLOCKS = 2;

/**
 * 常駐予算の既定 = **最も大きい block 2 本ぶん**（`./ple.ts` の
 * `Gemma4PleOptions.maxResidentBytes`）。
 *
 * 「2 本」を本数のまま既定にすると、資産世代で block 幅が変わった瞬間に同じ数字が別の RAM を
 * 意味する。**最大** block を基準に取るのは、どの 2 本を掴んでも予算に収まる = 「2 本常駐」の
 * 意味が幅に依らず保たれる唯一の取り方だからである（ADR 0085 追記 2026-09-02 — 旧 shard 世代の
 * 同じ規則を block 単位へ引き継いだもの）。
 *
 * NOTE: 1 行は `values` と `scales` の 2 表から引くので、2 本は「値の block 1 本 + scale の
 * block 1 本」をちょうど賄う本数でもある。
 */
export const defaultGemma4PleResidentBytes = (index: Gemma4PleIndex): number =>
  DEFAULT_RESIDENT_BLOCKS * largestBlockBytes(index);

/**
 * 索引と容器の資産宣言が**ちょうど一致**することを見る門（**block のバイト列を 1 つも
 * 取る前**）。
 *
 * MUST: 両方向を全件列挙する。索引にあって容器に無い block は、その token 範囲を初めて引いた
 * ターン（= 会話の途中・GB 級のロード完了後）まで落ちない。容器にあって索引に無い block は
 * 「配布形が宣言した資産を 1 本も読まないまま動く」形で、永久に検出されない。長さの食い違いは
 * **形も dtype も合ったまま別 token の行を引く**（ADR 0085 決定 5 の沈黙誤値）。
 *
 * MUST: 1 件目で止めない。片方だけ焼き直した配布形では数十件が同時にずれるので、1 件ずつ
 * 直す往復にしない。
 */
export const assertGemma4PleAssets = (
  where: string,
  index: Gemma4PleIndex,
  component: Gemma4PleAssetSource,
): void => {
  const problems: string[] = [];
  const indexed = new Set<string>();
  for (const name of TABLE_NAMES) {
    const table = index[name];
    const role = TABLE_ROLE[name];
    for (const block of table.blocks) {
      indexed.add(block.asset);
      const found = Object.hasOwn(component.assets, block.asset)
        ? component.assets[block.asset]
        : undefined;
      if (found === undefined) {
        problems.push(`${block.asset}: 容器が宣言していない`);
        continue;
      }
      if (found !== role) {
        problems.push(`${block.asset}: 役割 '${found}'（'${role}' が要る）`);
        continue;
      }
      const want = gemma4PleBlockBytes(table, block);
      const got = component.asset(block.asset).length;
      if (got !== want) {
        problems.push(
          `${block.asset}: 論理長 ${got}（[${block.start}, ${block.stop}) × ${table.rowBytes} =` +
            ` ${want} が要る）`,
        );
      }
    }
  }
  for (const [name, role] of Object.entries(component.assets)) {
    if ((role === TABLE_ROLE.values || role === TABLE_ROLE.scales) && !indexed.has(name)) {
      problems.push(`${name}: 役割 '${role}' の資産だが索引が指していない`);
    }
  }
  if (problems.length === 0) return;
  throw new Error(
    `${where}: PLE の索引と容器の資産が食い違う（${problems.length} 件） — ` +
      problems.join(" / "),
  );
};

/**
 * 容器の資産から索引を読んで受理形にし、指し先の block が実在することまで見る
 * （**重みの block を 1 バイトも取る前**に通す家族の門 — `./pipeline.ts` の admission 席）。
 *
 * 索引そのものは数 KB の資産で、`values` / `scales` の block とは別の block に置かれる
 * （container-v1 §4.2 — 区間読みを要する block だけが専用 part に単独で載る）。
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
      `${where}: 容器が資産 '${PLE_INDEX_ASSET}' を宣言していない` +
        `（宣言されている資産: ${Object.keys(component.assets).join(" / ") || "なし"}）`,
    );
  }
  if (role !== PLE_INDEX_ROLE) {
    throw new Error(
      `${where}: 資産 '${PLE_INDEX_ASSET}' の役割が '${role}'（'${PLE_INDEX_ROLE}' が要る）`,
    );
  }
  const bytes = await readWholeAsset(component.asset(PLE_INDEX_ASSET));
  const at = `${where}: 資産 '${PLE_INDEX_ASSET}'`;
  let text: string;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch (cause) {
    throw new Error(`${at} が UTF-8 として読めない`, { cause });
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (cause) {
    throw new Error(`${at} が JSON として読めない`, { cause });
  }
  const index = parseGemma4PleIndex(parsed, at);
  assertGemma4PleAssets(where, index, component);
  return index;
};

/**
 * 開いた容器（runtime の `OpenedContainer`）を {@link Gemma4PleAssetSource} へ畳む。
 *
 * 取得面・全量面の部品（`ModelComponent`）は最初からこの面なので、要るのは**容器を直に開く
 * 側**（実資産の検収・bench の台本）だけである。役割の写し取りを呼び手ごとに書くと、綴りの
 * 違う写しが静かに増える。
 */
export const gemma4PleAssetSource = (
  opened: Pick<OpenedContainer, "model" | "asset">,
): Gemma4PleAssetSource => ({
  assets: Object.fromEntries(
    Object.keys(opened.model?.assets ?? {}).map((name) => [name, opened.asset(name).role]),
  ),
  asset: (name) => opened.asset(name),
});
