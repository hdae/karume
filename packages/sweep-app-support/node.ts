/**
 * The Node side of the two apps (`./node`): the checkout a record was made on, and the write of a
 * record under the repository root by the path rule of `./src/record-path.ts`. The workbench's dev
 * server and Node run use both, a production build of either app bakes the checkout in when its
 * config loads, and the benchmark app's `unpack.ts` writes a bundle's records here.
 *
 * SPEC: docs/spec/workbench.md#where-records-go
 */

import { execFile } from "node:child_process";
import { type FileHandle, mkdir, open, readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { promisify } from "node:util";

import { type RecordTarget, recordPath } from "./src/record-path.ts";
import { type Checkout, utcStamp } from "./src/record.ts";

const run = promisify(execFile);

const git = async (repositoryRoot: string, ...args: string[]): Promise<string> =>
  (await run("git", ["-C", repositoryRoot, ...args], { maxBuffer: 64 * 1024 * 1024 })).stdout;

/** The checkout, read from git and `packages/runtime/package.json` at the moment it is asked for. */
export const readCheckout = async (repositoryRoot: string): Promise<Checkout> => {
  const [head, status, manifest] = await Promise.all([
    git(repositoryRoot, "rev-parse", "HEAD"),
    git(repositoryRoot, "status", "--porcelain"),
    readFile(join(repositoryRoot, "packages/runtime/package.json"), "utf8"),
  ]);
  const parsed: unknown = JSON.parse(manifest);
  const version =
    typeof parsed === "object" && parsed !== null && "version" in parsed
      ? parsed.version
      : undefined;
  if (typeof version !== "string") {
    throw new Error("packages/runtime/package.json has no version");
  }
  return { commit: head.trim(), dirty: status.trim() !== "", runtimeVersion: version };
};

/** A new file at `path`, or `undefined` when one exists. */
const createExclusive = async (path: string): Promise<FileHandle | undefined> => {
  try {
    return await open(path, "wx");
  } catch (error) {
    if (error instanceof Error && "code" in error && error.code === "EEXIST") return undefined;
    throw error;
  }
};

/**
 * Writes `bytes` to a new file at the repository-relative `path` (its directory made as needed) and
 * returns true, or returns false and writes nothing when the file exists: an existing file is never
 * replaced.
 */
export const writeNewFile = async (
  repositoryRoot: string,
  path: string,
  bytes: Uint8Array,
): Promise<boolean> => {
  const absolute = join(repositoryRoot, path);
  await mkdir(dirname(absolute), { recursive: true });
  const file = await createExclusive(absolute);
  if (file === undefined) return false;
  try {
    await file.writeFile(bytes);
  } finally {
    await file.close();
  }
  return true;
};

/**
 * Writes `bytes` to a new file for `target` at `now` and returns its path relative to
 * `repositoryRoot` (with `/`). A name that is taken gets `-1`, `-2`, … after the stamp; an existing
 * file is never replaced.
 */
export const writeRecord = async (
  repositoryRoot: string,
  target: RecordTarget,
  bytes: Uint8Array,
  now: Date,
): Promise<string> => {
  const stamp = utcStamp(now);
  for (let attempt = 0; ; attempt += 1) {
    const path = recordPath(target, stamp, attempt);
    if (await writeNewFile(repositoryRoot, path, bytes)) return path;
  }
};
