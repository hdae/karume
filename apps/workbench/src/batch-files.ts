/**
 * The dev server's `GET /workbench/batches`: every `*.json` file directly in
 * `inputs/workbench/batches/` under the repository root, read at request time, as
 * `{ "batches": [{ "id": …, "text": … }] }` in file-name order. The request names no path, so
 * nothing the client sends reaches the file system. A file over the cap is listed with that problem
 * in place of its text (nothing past the cap is read). Node only.
 *
 * SPEC: docs/spec/workbench.md#the-dev-server
 * SPEC: docs/spec/workbench.md#batches
 */

import { open, readdir } from "node:fs/promises";
import { join } from "node:path";

import { type BatchFile, UNTRACKED_BATCHES_DIRECTORY } from "./batch.ts";

/** The largest batch file the endpoint reads (a batch is a short list). */
export const MAX_BATCH_BYTES = 1024 * 1024;

export type BatchesRequest = { readonly method: string };

export type BatchesContext = {
  readonly repositoryRoot: string;
  readonly maxBytes: number;
};

export type BatchesResponse = {
  readonly status: number;
  readonly headers: Readonly<Record<string, string>>;
  readonly body: string;
};

/** The file's bytes, or `undefined` once they pass `maxBytes` (at most `maxBytes + 1` are read). */
const readCapped = async (
  path: string,
  maxBytes: number,
): Promise<Uint8Array<ArrayBuffer> | undefined> => {
  const file = await open(path, "r");
  try {
    const buffer = new Uint8Array(maxBytes + 1);
    let length = 0;
    while (length < buffer.byteLength) {
      const { bytesRead } = await file.read(buffer, length, buffer.byteLength - length, length);
      if (bytesRead === 0) break;
      length += bytesRead;
    }
    return length > maxBytes ? undefined : buffer.slice(0, length);
  } finally {
    await file.close();
  }
};

const isMissing = (error: unknown): boolean =>
  error instanceof Error && "code" in error && error.code === "ENOENT";

/** The untracked batch files under `repositoryRoot`, in file-name order; none when the directory is missing. */
export const untrackedBatchFiles = async (
  repositoryRoot: string,
  maxBytes: number,
): Promise<readonly BatchFile[]> => {
  const directory = join(repositoryRoot, UNTRACKED_BATCHES_DIRECTORY);
  let names: string[];
  try {
    names = (await readdir(directory, { withFileTypes: true }))
      .filter((item) => item.isFile() && item.name.endsWith(".json"))
      .map(({ name }) => name)
      .sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
  } catch (error) {
    if (isMissing(error)) return [];
    throw error;
  }
  return Promise.all(
    names.map(async (name): Promise<BatchFile> => {
      const id = name.slice(0, -".json".length);
      const bytes = await readCapped(join(directory, name), maxBytes);
      if (bytes === undefined) {
        return { id, problem: `the file is larger than ${maxBytes} bytes, so it was not read` };
      }
      try {
        return { id, text: new TextDecoder("utf-8", { fatal: true }).decode(bytes) };
      } catch {
        return { id, problem: "the file is not UTF-8 text" };
      }
    }),
  );
};

export const handleBatchesGet = async (
  request: BatchesRequest,
  context: BatchesContext,
): Promise<BatchesResponse> => {
  if (request.method !== "GET") {
    return {
      status: 405,
      headers: { Allow: "GET", "Content-Type": "text/plain; charset=utf-8" },
      body: "Only GET\n",
    };
  }
  const batches = await untrackedBatchFiles(context.repositoryRoot, context.maxBytes);
  return {
    status: 200,
    headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
    body: `${JSON.stringify({ batches })}\n`,
  };
};
