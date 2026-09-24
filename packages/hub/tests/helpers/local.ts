/**
 * ローカル配布形（ディレクトリに置かれた `karume.json` + 資産）の合成。実 fs を使うテスト
 * （`deno_directory_test.ts`）とメモリ上のアダプターを使うテスト（`local_test.ts`）が同じ
 * 配布形を共有する — 片方だけの形に寄せると「実 fs では通らない合成」が生まれる。
 *
 * `size` / `sha256` は**実際のバイト列から導出**する（手書きの定数にすると、payload を変えた
 * ときに size 門が意味を失う）。
 */

import { MANIFEST_FILENAME } from "../../src/manifest.ts";
import type { DirectoryAdapter } from "../../src/sources/local.ts";
import { payloadFor } from "./mock.ts";

/**
 * 容器 1 本ぶんの part（container-v1 §8 の綴り）。添字 1 は**長さ 0 の part**（const が空の
 * コンテナ — ADR 0109 決定 3）で、0 バイトのファイルとして配布形に置かれるが取得も読みも
 * 起きない。
 */
export const PART_PATHS = [
  "net/model-00001-of-00003.krm",
  "net/model-00002-of-00003.krm",
  "net/model-00003-of-00003.krm",
] as const;
/** 長さ 0 の part の添字（{@link PART_PATHS} の中の 1 本）。 */
export const EMPTY_PART_INDEX = 1;
/** 選択の列に載る part（長さ 0 を除いた 2 本 — 取得も進捗もこの本数で数える）。 */
export const FETCHED_PART_PATHS: readonly string[] = PART_PATHS.filter(
  (_path, index) => index !== EMPTY_PART_INDEX,
);
/**
 * 選択の表（`helpers/selection.ts` の `selectionFiles`）で part を指すキー。添字は**宣言上の
 * part の id** なので、長さ 0 の part を飛ばしても後続の添字は詰まらない。
 */
export const PART_KEYS: readonly string[] = PART_PATHS
  .map((_path, index) => `net[${index}]`)
  .filter((_key, index) => index !== EMPTY_PART_INDEX);
export const TOKENIZER_PATH = "tokenizer/tokenizer.json";

/** コンテナのヘッダ長（container-v1 §1）。part 0 は「ヘッダ + 2 文書ちょうど」。 */
const HEADER_BYTES = 24;
/** モデル記述のバイト長（ダミー — hub は descriptor を parse しない）。 */
const MODEL_DOC_BYTES = 5;

/** 越境参照（ADR 0038 §7）— 別リポの資産を 1 本だけ持つ形。 */
export const CROSS_REPO = "someone/shared";
export const CROSS_REVISION = "c".repeat(40);
export const CROSS_PATH = "text_encoder/model.safetensors";

export const sha256Hex = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> => {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
};

type FileRefJson = {
  readonly path: string;
  readonly size: number;
  readonly sha256: string;
  readonly repo?: string;
  readonly revision?: string;
};

const fileRefJson = async (
  path: string,
  bytes: Uint8Array<ArrayBuffer>,
  cross?: { readonly repo: string; readonly revision: string },
): Promise<FileRefJson> => ({
  path,
  size: bytes.byteLength,
  sha256: await sha256Hex(bytes),
  ...(cross ?? {}),
});

/** 合成した配布形 1 つぶん（`files` は `karume.json` を含む「ディレクトリの中身」）。 */
export type LocalDist = {
  readonly files: Map<string, Uint8Array<ArrayBuffer>>;
  /** 越境先リポのディレクトリの中身（`cross` を頼んだときだけ中身が入る）。 */
  readonly crossFiles: Map<string, Uint8Array<ArrayBuffer>>;
};

/**
 * 1 モデル・容器 1 本（part 3 本・うち 1 本は長さ 0）+ tokenizer の最小配布形。`cross` を真に
 * すると、その上に越境参照の資産（別リポの `text_encoder`）が 1 本乗る。
 *
 * 長さ 0 の part も**0 バイトのファイルとして置く**（container-v1 §8）— 取得は起きないが、
 * 配布形としては実在する。
 */
export const buildLocalDist = async (
  options: { readonly cross?: boolean } = {},
): Promise<LocalDist> => {
  const parts = PART_PATHS.map((path, index) => ({
    path,
    bytes: index === EMPTY_PART_INDEX ? new Uint8Array(new ArrayBuffer(0)) : payloadFor(path),
  }));
  const tokenizer = payloadFor(TOKENIZER_PATH);
  const files = new Map<string, Uint8Array<ArrayBuffer>>(
    parts.map(({ path, bytes }) => [path, bytes]),
  );
  files.set(TOKENIZER_PATH, tokenizer);
  const crossFiles = new Map<string, Uint8Array<ArrayBuffer>>();
  // 越境先は**別の実体**（同じ path 文字列でもバイト列が違うことを踏ませる）。
  const crossBytes = payloadFor(`${CROSS_REPO}/${CROSS_PATH}`);
  if (options.cross === true) crossFiles.set(CROSS_PATH, crossBytes);

  const assets: Record<string, FileRefJson> = {
    tokenizer: await fileRefJson(TOKENIZER_PATH, tokenizer),
    ...(options.cross === true
      ? {
        text_encoder: await fileRefJson(CROSS_PATH, crossBytes, {
          repo: CROSS_REPO,
          revision: CROSS_REVISION,
        }),
      }
      : {}),
  };
  const partRefs = await Promise.all(parts.map(({ path, bytes }) => fileRefJson(path, bytes)));
  const manifest = {
    format: "karume/5",
    generator: "karume/0.13.0",
    defaultModel: "m",
    models: {
      m: {
        pipeline: "anima/1",
        weights: {
          net: {
            f16: {
              container: {
                // part 0 は「ヘッダ + グラフ記述 + モデル記述」ちょうど（container-v1 §8）。
                descriptor: {
                  graph: {
                    length: partRefs[0].size - HEADER_BYTES - MODEL_DOC_BYTES,
                    sha256: await sha256Hex(new TextEncoder().encode("graph:net/model")),
                  },
                  model: {
                    length: MODEL_DOC_BYTES,
                    sha256: await sha256Hex(new TextEncoder().encode("model:net/model")),
                  },
                },
                parts: partRefs,
              },
            },
          },
        },
        assets,
        quants: { f16: { weights: { net: "f16" }, session: {} } },
        defaultQuant: "f16",
        pipelineConfig: {},
      },
    },
  };
  files.set(MANIFEST_FILENAME, new TextEncoder().encode(JSON.stringify(manifest, undefined, 2)));
  return { files, crossFiles };
};

/** メモリ上のディレクトリ。読んだ path と、透過してきた `signal` の有無を記録する。 */
export type MemoryDirectory = {
  readonly adapter: DirectoryAdapter;
  /** `readFile` に渡された path（順序どおり）。 */
  readonly reads: string[];
  /** `readFile` に `signal` が透過してきたか（`reads` と同じ順）。 */
  readonly signals: boolean[];
};

export const memoryDirectory = (
  files: ReadonlyMap<string, Uint8Array<ArrayBuffer>>,
): MemoryDirectory => {
  const reads: string[] = [];
  const signals: boolean[] = [];
  const lookup = (path: string): Uint8Array<ArrayBuffer> => {
    const bytes = files.get(path);
    // 実 fs のアダプター（`deno.ts`）と同じ作法 — 欠損は実体のパスを名乗って落とす。
    if (bytes === undefined) throw new Error(`test-directory: ${path} を読めない`);
    return bytes;
  };
  const adapter: DirectoryAdapter = {
    readFile: (path, { signal }) => {
      reads.push(path);
      signals.push(signal !== undefined);
      try {
        // 実体を読むたびに新しい buffer が来る（tight view）— 同じ参照を配ると、共通層が
        // 「別の実体を読んだ」ことをテストが観測できなくなる。
        return Promise.resolve(new Uint8Array(lookup(path)));
      } catch (error) {
        return Promise.reject(error);
      }
    },
  };
  return { reads, signals, adapter };
};
