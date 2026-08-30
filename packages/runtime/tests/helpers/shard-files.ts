// 代表 path（`…/model.safetensors` や `…/model.i8.safetensors`）から**配布形の shard ファイル列**を
// 解決する（shard 仕様 v2 — ADR 0081）。テストが持つのは「ファイルの見つけ方」だけで、読み手
// （runtime）は N 本の shard を既に扱える。
//
// Python 側 `karume.shards.resolve_shards`（tools/exporter/src/karume/shards.py）の**鏡像**。
// MUST: 規則を 2 箇所で別々に育てない — 焼く側と読み返す側で連番の綴りが割れると、前回の
// 書き出しの残骸を今回の期待値で読む形が黙って通る。
//
// helpers の規律どおり `src/` は import しない（shard 1 本の型は `ModelShard` / hub の
// `StreamedAsset` と**構造互換**の形で独立に持つ）。models 側からも
// `../../runtime/tests/helpers/shard-files.ts` で使う（パッケージ跨ぎ相対 import の先例は
// `png-decode.ts` / `limit_vocabulary_test.ts`）。

/** 連番の桁数（`-NNNNN-of-NNNNN`）— Python 側 `_INDEX_DIGITS` と同値。 */
const INDEX_DIGITS = 5;

/** shard 1 本（runtime の `ModelShard` と構造互換）。 */
export type ShardFile = {
  /** 失敗とフェンスの帰属先。配布形と同じ `<コンポーネント>/<ファイル名>` の綴りで名乗る。 */
  readonly id: string;
  /** shard のバイト列。**buffer 全体を占める view MUST**（slice すると RAM ピークが倍増する）。 */
  readonly bytes: Uint8Array<ArrayBuffer>;
};

const escapeRegExp = (text: string): string => text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/**
 * この代表 path と同じコンポーネントの shard ファイル名に一致する正規表現。
 *
 * MUST: stem / suffix は escape する — 実 path にはドットもハイフンも入るので、素で埋めると
 * 無関係なファイルを拾う（glob を使わないのも同じ理由）。
 */
const sequencePattern = (name: string): RegExp => {
  const dot = name.lastIndexOf(".");
  const stem = dot <= 0 ? name : name.slice(0, dot);
  const suffix = dot <= 0 ? "" : name.slice(dot);
  return new RegExp(
    `^${escapeRegExp(stem)}-(\\d{${INDEX_DIGITS}})-of-(\\d{${INDEX_DIGITS}})${
      escapeRegExp(suffix)
    }$`,
  );
};

/** URL の最終要素（`%` エスケープを解いた実ファイル名）。 */
const baseName = (url: URL): string => decodeURIComponent(url.pathname).split("/").pop() ?? "";

/**
 * ファイルの有無。
 * MUST: NotFound 以外は伝播させる — 全 I/O エラーを「資産が無い」に丸めると、資産ルートの
 * マウント異常が SKIP に化けて、実行されていない検証が静かに緑になる。
 */
const isFile = (url: URL): boolean => {
  try {
    return Deno.statSync(url).isFile;
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return false;
    throw cause;
  }
};

/** ディレクトリの列挙（不在は空 — 中身の異常は伝播させる）。 */
const listDir = (url: URL): readonly Deno.DirEntry[] => {
  try {
    return [...Deno.readDirSync(url)];
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return [];
    throw cause;
  }
};

/**
 * コンポーネントの**代表 path** → 実在する shard ファイル列（読む順 = shard 番号順・先頭が
 * グラフ shard）。分割されていない資産と存在しない資産はどちらも `[representative]` を返す
 * （不在の診断は呼び手の既存の門が持つ — ここで先回りすると綴りが 2 つに割れる）。
 *
 * MUST: 曖昧な現場は fail loudly。単一ファイルと連番の同居・`of` の食い違い・番号の欠けや
 * はみ出しは、どれも「どのバイト列を読むか」が一意に決まらない。黙って一方を選ぶと、前回の
 * 重みを今回の期待値で突き合わせる形になる。
 */
export const resolveShards = (representative: URL): readonly URL[] => {
  const parent = new URL("./", representative);
  const pattern = sequencePattern(baseName(representative));
  const found = new Map<number, URL>();
  const totals = new Set<number>();
  for (const entry of listDir(parent)) {
    if (!entry.isFile) continue;
    const match = pattern.exec(entry.name);
    if (match === null) continue;
    found.set(Number(match[1]), new URL(encodeURIComponent(entry.name), parent));
    totals.add(Number(match[2]));
  }
  if (found.size === 0) return [representative];
  const where = decodeURIComponent(representative.pathname);
  if (isFile(representative)) {
    throw new Error(
      `${where}: 単一ファイルと shard 連番（${found.size} 本）が同居している` +
        " — 前回の書き出しの残骸を消してから読む",
    );
  }
  if (totals.size !== 1) {
    throw new Error(`${where}: shard 連番の総数が ${[...totals].sort()} と食い違っている`);
  }
  const [total] = totals;
  const files: URL[] = [];
  for (let index = 1; index <= total; index += 1) {
    const file = found.get(index);
    if (file === undefined) {
      throw new Error(`${where}: shard 連番 1..${total} のうち ${index} 本目が無い`);
    }
    files.push(file);
  }
  if (found.size !== total) {
    const surplus = [...found.keys()].filter((index) => index > total).sort((a, b) => a - b);
    throw new Error(`${where}: shard 連番 1..${total} からはみ出した番号 ${surplus} がある`);
  }
  return files;
};

/**
 * コンポーネントが実在するか（単一形 / 連番のどちらでも）。資産の有無で SKIP を決める門が
 * 「`model.safetensors` があるか」を直に見ていた席の置き換え。
 */
export const modelPresent = (representative: URL): boolean =>
  isFile(resolveShards(representative)[0]);

/**
 * shard 1 本を読む。
 * MUST: slice しない — `Deno.readFile` の返す view は buffer 全体を覆っているので、辻褄合わせの
 * コピーを挟むと 4GiB 級の資産で RAM ピークが倍増する。
 */
export const readShard = async (file: URL): Promise<ShardFile> => ({
  id: decodeURIComponent(file.pathname).split("/").slice(-2).join("/"),
  bytes: await Deno.readFile(file),
});

/**
 * shard 列を 1 本ずつ読んで流す（`PreparedModel.createSession` / `createSessionFromShards` へ
 * そのまま渡せる）。逐次に読むので、常駐するのは「いま消費している 1 本」だけ。
 */
export const streamShards = async function* (
  files: readonly URL[],
): AsyncGenerator<ShardFile, void, unknown> {
  for (const file of files) yield await readShard(file);
};

/** `size` バイトちょうど読む（短い読み返しで黙ってゼロ埋めのヘッダを作らない）。 */
const readExact = async (handle: Deno.FsFile, into: Uint8Array, where: URL): Promise<void> => {
  for (let read = 0; read < into.length;) {
    const chunk = await handle.read(into.subarray(read));
    if (chunk === null) {
      throw new Error(`${where.pathname}: ${into.length} バイトを読み切れない`);
    }
    read += chunk;
  }
};

/**
 * shard 列にあるテンソル名の**和**（safetensors のヘッダ JSON だけを読む — データ節は 1 バイトも
 * 読まない）。「宣言された companion scale が実体として在る」類の門を、単一ファイル面と同じ
 * 強さで shard 列にも掛けるための面。
 */
export const shardTensorNames = async (files: readonly URL[]): Promise<ReadonlySet<string>> => {
  const names = new Set<string>();
  for (const file of files) {
    const handle = await Deno.open(file, { read: true });
    try {
      const lengthBytes = new Uint8Array(8);
      await readExact(handle, lengthBytes, file);
      const length = Number(new DataView(lengthBytes.buffer).getBigUint64(0, true));
      const headerBytes = new Uint8Array(length);
      await readExact(handle, headerBytes, file);
      const header: unknown = JSON.parse(new TextDecoder().decode(headerBytes));
      if (typeof header !== "object" || header === null) {
        throw new Error(`${file.pathname}: safetensors ヘッダが JSON オブジェクトでない`);
      }
      for (const name of Object.keys(header)) {
        if (name !== "__metadata__") names.add(name);
      }
    } finally {
      handle.close();
    }
  }
  return names;
};
