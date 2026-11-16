/**
 * The registry: every file `src/entries/<id>.ts` default-exports one {@link Entry}, and its id is
 * its file name. Adding an entry is adding a file; nothing else lists them.
 */

import { type Entry, ENTRY_ID } from "./entry.ts";

const MODULES = import.meta.glob<Entry>("./entries/*.ts", { eager: true, import: "default" });

/** The registry does not hold what it promises (a missing default export, an id off its file name). */
export class RegistryError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "RegistryError";
  }
}

/**
 * The entries of `modules` (glob path → default export) in file-name order, or a
 * {@link RegistryError} for the first file whose default export is missing or whose id is not its
 * kebab-case file name.
 */
export const entriesOf = (modules: Readonly<Record<string, Entry | undefined>>): readonly Entry[] =>
  Object.keys(modules)
    .sort()
    .map((path) => {
      const name = /([^/]+)\.ts$/.exec(path)?.[1] ?? path;
      const entry = modules[path];
      if (entry === undefined) throw new RegistryError(`${path} has no default export`);
      if (entry.id !== name || !ENTRY_ID.test(entry.id)) {
        throw new RegistryError(
          `${path} exports the id ${JSON.stringify(entry.id)}; an id is the kebab-case file name`,
        );
      }
      return entry;
    });

/** Every entry of `src/entries/`, checked (the shell and the Node run call this at start). */
export const registeredEntries = (): readonly Entry[] => entriesOf(MODULES);
