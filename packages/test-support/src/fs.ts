/**
 * Deno-shaped file helpers for tests, implemented on `node:fs`.
 *
 * The Deno suite reads fixtures and local assets with `Deno.readFile` & co. These helpers keep the
 * call shape (name, argument order, `string | URL` paths, `Uint8Array<ArrayBuffer>` results) so the
 * port is a mechanical rename. Only the functions the suite uses exist here; add on demand.
 *
 * Byte results are always backed by their own `ArrayBuffer` (Node may hand out views into a shared
 * pool for small reads, which would break `new Float32Array(bytes.buffer)`-style reinterpretation).
 */

import {
  mkdirSync as nodeMkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync as nodeReadFileSync,
  readlinkSync,
  realpathSync,
  rmdirSync,
  rmSync,
  statSync as nodeStatSync,
  symlinkSync as nodeSymlinkSync,
  writeFileSync as nodeWriteFileSync,
  type Dirent,
  type Stats,
} from "node:fs";
import {
  mkdir as nodeMkdir,
  mkdtemp,
  open,
  readdir,
  readFile as nodeReadFile,
  rm,
  rmdir,
  stat as nodeStat,
  symlink as nodeSymlink,
  writeFile as nodeWriteFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

export type PathLike = string | URL;

export type FileInfo = {
  readonly isFile: boolean;
  readonly isDirectory: boolean;
  readonly isSymlink: boolean;
  readonly size: number;
  readonly mtime: Date | null;
};

export type DirEntry = {
  readonly name: string;
  readonly isFile: boolean;
  readonly isDirectory: boolean;
  readonly isSymlink: boolean;
};

export type WriteOptions = { readonly append?: boolean; readonly createNew?: boolean };
export type MkdirOptions = { readonly recursive?: boolean };
export type RemoveOptions = { readonly recursive?: boolean };
export type TempDirOptions = { readonly prefix?: string; readonly dir?: string };

const owned = (buffer: Buffer): Uint8Array<ArrayBuffer> => {
  const backing = buffer.buffer;
  if (
    backing instanceof ArrayBuffer && buffer.byteOffset === 0 &&
    backing.byteLength === buffer.byteLength
  ) {
    return new Uint8Array(backing);
  }
  const copy = new Uint8Array(new ArrayBuffer(buffer.byteLength));
  copy.set(buffer);
  return copy;
};

const errorCode = (error: unknown): string | undefined =>
  typeof error === "object" && error !== null && "code" in error && typeof error.code === "string"
    ? error.code
    : undefined;

/** `error instanceof Deno.errors.NotFound` */
export const isNotFound = (error: unknown): boolean => errorCode(error) === "ENOENT";
/** `error instanceof Deno.errors.IsADirectory` */
export const isADirectory = (error: unknown): boolean => errorCode(error) === "EISDIR";
/** `error instanceof Deno.errors.FilesystemLoop` */
export const isFilesystemLoop = (error: unknown): boolean => errorCode(error) === "ELOOP";

export type ReadOptions = { readonly signal?: AbortSignal };

export const readFile = async (path: PathLike, options?: ReadOptions): Promise<Uint8Array<ArrayBuffer>> =>
  owned(await nodeReadFile(path, { signal: options?.signal }));
export const readFileSync = (path: PathLike): Uint8Array<ArrayBuffer> => owned(nodeReadFileSync(path));
export const readTextFile = (path: PathLike, options?: ReadOptions): Promise<string> =>
  nodeReadFile(path, { encoding: "utf8", signal: options?.signal });
export const readTextFileSync = (path: PathLike): string => nodeReadFileSync(path, "utf8");

/** `createNew` fails when the file exists (Deno semantics; Node flag `wx`). */
const flag = (options: WriteOptions | undefined): "a" | "w" | "wx" =>
  options?.createNew === true ? "wx" : options?.append === true ? "a" : "w";

export const writeFile = (path: PathLike, data: Uint8Array, options?: WriteOptions): Promise<void> =>
  nodeWriteFile(path, data, { flag: flag(options) });
export const writeFileSync = (path: PathLike, data: Uint8Array, options?: WriteOptions): void =>
  nodeWriteFileSync(path, data, { flag: flag(options) });
export const writeTextFile = (path: PathLike, text: string, options?: WriteOptions): Promise<void> =>
  nodeWriteFile(path, text, { encoding: "utf8", flag: flag(options) });
export const writeTextFileSync = (path: PathLike, text: string, options?: WriteOptions): void =>
  nodeWriteFileSync(path, text, { encoding: "utf8", flag: flag(options) });

export const mkdir = async (path: PathLike, options?: MkdirOptions): Promise<void> => {
  await nodeMkdir(path, { recursive: options?.recursive === true });
};
export const mkdirSync = (path: PathLike, options?: MkdirOptions): void => {
  nodeMkdirSync(path, { recursive: options?.recursive === true });
};

/** `Deno.remove`: a file, an empty directory, or (with `recursive`) a whole tree. */
export const remove = async (path: PathLike, options?: RemoveOptions): Promise<void> => {
  if (options?.recursive === true) return rm(path, { recursive: true });
  const info = await nodeStat(path);
  if (info.isDirectory()) return rmdir(path);
  return rm(path);
};
export const removeSync = (path: PathLike, options?: RemoveOptions): void => {
  if (options?.recursive === true) return rmSync(path, { recursive: true });
  if (nodeStatSync(path).isDirectory()) return rmdirSync(path);
  rmSync(path);
};

const tempTemplate = (options: TempDirOptions | undefined): string =>
  join(options?.dir ?? tmpdir(), options?.prefix ?? "");

export const makeTempDir = (options?: TempDirOptions): Promise<string> => mkdtemp(tempTemplate(options));
export const makeTempDirSync = (options?: TempDirOptions): string => mkdtempSync(tempTemplate(options));

const fileInfo = (stats: Stats): FileInfo => ({
  isFile: stats.isFile(),
  isDirectory: stats.isDirectory(),
  isSymlink: stats.isSymbolicLink(),
  size: stats.size,
  mtime: stats.mtime,
});

/** `Deno.stat` (follows symlinks). */
export const stat = async (path: PathLike): Promise<FileInfo> => fileInfo(await nodeStat(path));
export const statSync = (path: PathLike): FileInfo => fileInfo(nodeStatSync(path));

const dirEntry = (entry: Dirent): DirEntry => ({
  name: entry.name,
  isFile: entry.isFile(),
  isDirectory: entry.isDirectory(),
  isSymlink: entry.isSymbolicLink(),
});

/** `Deno.readDir` (entries are not followed through symlinks, like Deno). */
export async function* readDir(path: PathLike): AsyncIterableIterator<DirEntry> {
  for (const entry of await readdir(path, { withFileTypes: true })) yield dirEntry(entry);
}
export const readDirSync = (path: PathLike): DirEntry[] =>
  readdirSync(path, { withFileTypes: true }).map(dirEntry);

export const symlink = (target: PathLike, path: PathLike): Promise<void> => nodeSymlink(target, path);
export const symlinkSync = (target: PathLike, path: PathLike): void => nodeSymlinkSync(target, path);
export const readLinkSync = (path: PathLike): string => readlinkSync(path, "utf8");

/**
 * Bytes `[offset, offset + length)` of a file in an owned buffer (the `Deno.open` + `seek` + `read`
 * loop of the Deno suite). Throws when the file ends before `length` bytes were read.
 */
export const readRange = async (
  path: PathLike,
  offset: number,
  length: number,
): Promise<Uint8Array<ArrayBuffer>> => {
  const handle = await open(path, "r");
  try {
    const into = new Uint8Array(new ArrayBuffer(length));
    let filled = 0;
    while (filled < length) {
      const { bytesRead } = await handle.read(into, filled, length - filled, offset + filled);
      if (bytesRead === 0) {
        throw new Error(
          `${String(path)}: could not read ${length} bytes at offset ${offset} (file ends after ${filled})`,
        );
      }
      filled += bytesRead;
    }
    return into;
  } finally {
    await handle.close();
  }
};
export const realPathSync = (path: PathLike): string => realpathSync(path, "utf8");
