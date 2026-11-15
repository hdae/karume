/**
 * Records which files a test file reads, which directories it lists and which commands it starts.
 *
 * The verification cache reuses a test file's result only while its inputs are unchanged. The code a
 * test imports is found by walking the import graph, and the kernels by `./kernel-record.ts`; the data
 * a test reads at run time — a mirror's manifest, a fixture, a golden — is visible in neither. A read
 * the cache does not know about is the dangerous kind of miss: the data changes and the old result is
 * reused. So the reads are not declared by each test; they are observed where every read passes.
 *
 * `node:fs` and `node:child_process` are wrapped in the test worker (never in package sources): the
 * functions that open a path for reading, the ones that list a directory, and the ones that start a
 * process. `syncBuiltinESMExports` makes the wrappers visible to modules that import the functions by
 * name. A wrapper records its argument and calls the original, so behaviour does not change.
 *
 * What is recorded is every attempt, including one that fails: a file that was looked for and not
 * found is an input too (its appearance changes the outcome).
 */

import childProcess from "node:child_process";
import fs from "node:fs";
import { syncBuiltinESMExports } from "node:module";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

/** What one test file took in from outside its imports, each list sorted and without duplicates. */
export type InputRecord = {
  /** Absolute paths opened for reading. */
  readonly files: readonly string[];
  /** Absolute paths of directories listed. */
  readonly directories: readonly string[];
  /** Commands started: the program followed by its arguments. */
  readonly commands: readonly (readonly string[])[];
};

type InputState = {
  installed: boolean;
  files: Set<string>;
  directories: Set<string>;
  /** Commands as JSON text (a set of arrays needs a value to compare by). */
  commands: Set<string>;
};

type InputHost = typeof globalThis & { __karumeInputRecord?: InputState };

const stateOf = (): InputState => {
  const host = globalThis as InputHost;
  host.__karumeInputRecord ??= {
    installed: false,
    files: new Set(),
    directories: new Set(),
    commands: new Set(),
  };
  return host.__karumeInputRecord;
};

/** The absolute path a `fs` path argument names (`undefined` for a file descriptor or a handle). */
export const pathOfArgument = (target: unknown): string | undefined => {
  if (typeof target === "string") return resolve(target);
  if (target instanceof URL) return target.protocol === "file:" ? fileURLToPath(target) : undefined;
  if (target instanceof Uint8Array) return resolve(Buffer.from(target).toString("utf8"));
  return undefined;
};

/** Whether `flags` of an `open` call can read (the default, `"r"`, `"r+"`, `"w+"`, `O_RDWR`, …). */
export const opensForReading = (flags: unknown): boolean => {
  if (flags === undefined || flags === null) return true;
  if (typeof flags === "string") return flags.includes("r") || flags.includes("+");
  if (typeof flags === "number") {
    const access = flags & (fs.constants.O_WRONLY | fs.constants.O_RDWR);
    return access !== fs.constants.O_WRONLY;
  }
  return true;
};

type AnyFunction = (this: unknown, ...args: unknown[]) => unknown;

/**
 * Replaces `owner[name]` with a function that calls `note` and then the original.
 *
 * MUST: the original's own properties are carried over, and its `util.promisify.custom`
 * implementation is wrapped too. `promisify(execFile)` returns that implementation, which resolves
 * with `{ stdout, stderr }`; a bare wrapper would lose it and callers would receive `stdout` alone.
 */
const wrap = (owner: object, name: string, note: (args: readonly unknown[]) => void): void => {
  const original: unknown = Reflect.get(owner, name);
  if (typeof original !== "function") return;
  const wrapped: AnyFunction = function (this: unknown, ...args: unknown[]): unknown {
    note(args);
    return Reflect.apply(original, this, args);
  };
  for (const key of Reflect.ownKeys(original)) {
    if (key === "length" || key === "name" || key === "prototype") continue;
    const descriptor = Reflect.getOwnPropertyDescriptor(original, key);
    if (descriptor === undefined) continue;
    const custom: unknown = descriptor.value;
    if (key === promisify.custom && typeof custom === "function") {
      descriptor.value = function (this: unknown, ...args: unknown[]): unknown {
        note(args);
        return Reflect.apply(custom, this, args);
      };
    }
    Reflect.defineProperty(wrapped, key, descriptor);
  }
  Reflect.set(owner, name, wrapped);
};

const textArguments = (value: unknown): string[] =>
  Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];

/**
 * Starts recording. Calling it again does nothing: a setup file may call it for every test file of a
 * worker.
 */
export const installInputRecord = (): void => {
  const state = stateOf();
  if (state.installed) return;
  state.installed = true;

  const noteFile = (target: unknown): void => {
    const path = pathOfArgument(target);
    if (path !== undefined) state.files.add(path);
  };
  const noteDirectory = (target: unknown): void => {
    const path = pathOfArgument(target);
    if (path !== undefined) state.directories.add(path);
  };
  const noteOpen = ([target, flags]: readonly unknown[]): void => {
    if (opensForReading(flags)) noteFile(target);
  };
  const noteRead = ([target]: readonly unknown[]): void => noteFile(target);
  const noteList = ([target]: readonly unknown[]): void => noteDirectory(target);

  for (const owner of [fs, fs.promises]) {
    wrap(owner, "open", noteOpen);
    wrap(owner, "readFile", noteRead);
    wrap(owner, "readdir", noteList);
    wrap(owner, "opendir", noteList);
  }
  wrap(fs, "openSync", noteOpen);
  wrap(fs, "readFileSync", noteRead);
  wrap(fs, "createReadStream", noteRead);
  wrap(fs, "readdirSync", noteList);
  wrap(fs, "opendirSync", noteList);

  // `spawn(command, args)` and its relatives; `exec(command)` passes one shell line.
  const noteCommand = ([command, args]: readonly unknown[]): void => {
    if (typeof command !== "string") return;
    state.commands.add(JSON.stringify([command, ...textArguments(args)]));
  };
  for (const name of [
    "spawn",
    "spawnSync",
    "execFile",
    "execFileSync",
    "exec",
    "execSync",
    "fork",
  ]) {
    wrap(childProcess, name, noteCommand);
  }

  syncBuiltinESMExports();
};

/** Forgets the inputs of the previous test file (called when a test file starts). */
export const beginInputRecordFile = (): void => {
  const state = stateOf();
  state.files = new Set();
  state.directories = new Set();
  state.commands = new Set();
};

/** Everything the test file took in since it began. */
export const inputsOfFile = (): InputRecord => {
  const state = stateOf();
  return {
    files: [...state.files].sort(),
    directories: [...state.directories].sort(),
    commands: [...state.commands].sort().map((text) => textArguments(JSON.parse(text))),
  };
};
