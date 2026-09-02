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

export const SHARD_PATHS = [
  "net/model.shard0.safetensors",
  "net/model.shard1.safetensors",
] as const;
export const TOKENIZER_PATH = "tokenizer/tokenizer.json";

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
 * 1 モデル・2 shard + tokenizer の最小配布形。`cross` を真にすると、その上に越境参照の資産
 * （別リポの `text_encoder`）が 1 本乗る。
 */
export const buildLocalDist = async (
  options: { readonly cross?: boolean } = {},
): Promise<LocalDist> => {
  const shards = SHARD_PATHS.map((path) => ({ path, bytes: payloadFor(path) }));
  const tokenizer = payloadFor(TOKENIZER_PATH);
  const files = new Map<string, Uint8Array<ArrayBuffer>>(
    shards.map(({ path, bytes }) => [path, bytes]),
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
  const manifest = {
    format: "karume/4",
    generator: "karume/0.8.0",
    defaultModel: "m",
    models: {
      m: {
        pipeline: "anima/1",
        weights: {
          net: {
            f16: {
              shards: await Promise.all(
                shards.map(({ path, bytes }) => fileRefJson(path, bytes)),
              ),
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
  options: { readonly vessel?: boolean } = {},
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
        // 実体を読むたびに新しい buffer が来る（tight view）— 同じ参照を配ると、
        // 逐次面が「手放した」ことをテストが観測できなくなる。
        return Promise.resolve(new Uint8Array(lookup(path)));
      } catch (error) {
        return Promise.reject(error);
      }
    },
    // `vessel: true` のときだけ器へ読む面を持つ（`deno.ts` と同じ契約 — 実長を返し、収まらない
    // ファイルは読まずに実長だけ返す）。既定では持たず、従来の tight view 経路を観測する
    // テストの前提を変えない。
    ...(options.vessel === true
      ? {
        readFileInto: (path, target, { signal }) => {
          reads.push(path);
          signals.push(signal !== undefined);
          try {
            const bytes = lookup(path);
            if (bytes.byteLength <= target.byteLength) target.set(bytes);
            return Promise.resolve(bytes.byteLength);
          } catch (error) {
            return Promise.reject(error);
          }
        },
      }
      : {}),
  };
  return { reads, signals, adapter };
};
