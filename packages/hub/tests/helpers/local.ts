/**
 * Synthesis of the local distribution form (`karume.json` + assets placed in a directory). The test
 * on the real fs (`deno_directory_test.ts`) and the test on an in-memory adapter (`local_test.ts`)
 * share the same distribution form — leaning it towards only one of them produces "a synthesis that
 * does not pass on the real fs".
 *
 * `size` / `sha256` are **derived from the actual bytes** (with hand-written constants the size
 * check would lose its meaning when the payload changes).
 */

import { MANIFEST_FILENAME } from "../../src/manifest.ts";
import type { DirectoryAdapter } from "../../src/sources/local.ts";
import { payloadFor } from "./mock.ts";

/**
 * The parts of one container (the spelling of container-v1 §8). Index 1 is a **zero-length part**
 * (a container with an empty const region — ADR 0109 decision 3); it is placed in the distribution
 * form as a 0-byte file but is neither fetched nor read.
 */
export const PART_PATHS = [
  "net/model-00001-of-00003.krm",
  "net/model-00002-of-00003.krm",
  "net/model-00003-of-00003.krm",
] as const;
/** The index of the zero-length part (one of {@link PART_PATHS}). */
export const EMPTY_PART_INDEX = 1;
/**
 * The parts listed in the selection (the 2 without the zero-length one — fetches and progress count
 * these).
 */
export const FETCHED_PART_PATHS: readonly string[] = PART_PATHS.filter(
  (_path, index) => index !== EMPTY_PART_INDEX,
);
/**
 * The keys that point at parts in the selection table (`selectionFiles` of `helpers/selection.ts`).
 * The index is **the declared part id**, so skipping the zero-length part does not shift the
 * indices
 * after it.
 */
export const PART_KEYS: readonly string[] = PART_PATHS.map(
  (_path, index) => `net[${index}]`,
).filter((_key, index) => index !== EMPTY_PART_INDEX);
export const TOKENIZER_PATH = "tokenizer/tokenizer.json";

/** The container header length (container-v1 §1). Part 0 is exactly "header + the 2 documents". */
const HEADER_BYTES = 24;
/** The byte length of the model descriptor (a dummy — the hub does not parse descriptors). */
const MODEL_DOC_BYTES = 5;

/**
 * A cross-repository reference (ADR 0038 §7) — the shape with a single asset from another
 * repository.
 */
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
  ...cross,
});

/**
 * One synthesized distribution form (`files` is "the directory contents", `karume.json` included).
 */
export type LocalDist = {
  readonly files: Map<string, Uint8Array<ArrayBuffer>>;
  /**
   * The directory contents of the referenced repository (filled only when `cross` is requested).
   */
  readonly crossFiles: Map<string, Uint8Array<ArrayBuffer>>;
};

/**
 * The minimal distribution form of 1 model, 1 container (3 parts, one of them zero-length) +
 * tokenizer. With `cross` set, a cross-repository asset (another repository's `text_encoder`) is
 * added on top.
 *
 * The zero-length part is **placed as a 0-byte file** too (container-v1 §8) — it is never fetched,
 * but it does exist in the distribution form.
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
  // The referenced repository holds **different content** (the same path string with different
  // bytes).
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
                // Part 0 is exactly "header + graph descriptor + model descriptor" (container-v1
                // §8).
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

/** An in-memory directory. Records the paths read and whether a `signal` was passed through. */
export type MemoryDirectory = {
  readonly adapter: DirectoryAdapter;
  /** The paths passed to `readFile` (in order). */
  readonly reads: string[];
  /** Whether a `signal` reached `readFile` (in the same order as `reads`). */
  readonly signals: boolean[];
};

export const memoryDirectory = (
  files: ReadonlyMap<string, Uint8Array<ArrayBuffer>>,
): MemoryDirectory => {
  const reads: string[] = [];
  const signals: boolean[] = [];
  const lookup = (path: string): Uint8Array<ArrayBuffer> => {
    const bytes = files.get(path);
    // The same convention as the real-fs adapter (`deno.ts`) — a missing file fails naming the
    // path.
    if (bytes === undefined) throw new Error(`test-directory: cannot read ${path}`);
    return bytes;
  };
  const adapter: DirectoryAdapter = {
    readFile: (path, { signal }) => {
      reads.push(path);
      signals.push(signal !== undefined);
      try {
        // Every read of a file returns a new buffer (tight view) — handing out the same reference
        // would hide from the test that the shared layer "read a different file".
        return Promise.resolve(new Uint8Array(lookup(path)));
      } catch (error) {
        return Promise.reject(error);
      }
    },
  };
  return { reads, signals, adapter };
};
