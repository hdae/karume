/**
 * The registry: every file `src/entries/<id>.ts` default-exports one {@link Entry}, and its id is
 * its file name. An entry that reads a local distribution is a file `src/entries/local/<id>.ts`:
 * that directory is the declaration, because a production build resolves its module of local
 * entries to an empty one (`local-entries.ts`). Adding an entry is adding a file; nothing else
 * lists them.
 */

import { ENTRY_ID } from "@karume/sweep-app-support/record";

import type { Entry } from "./entry.ts";
import { LOCAL_MODULES } from "./local-entries.ts";

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
 * {@link RegistryError} for the first file whose default export is missing, or whose id is not its
 * kebab-case file name or is another file's too.
 */
export const entriesOf = (modules: Readonly<Record<string, Entry | undefined>>): readonly Entry[] =>
  Object.keys(modules)
    .map((path) => ({ path, name: /([^/]+)\.ts$/.exec(path)?.[1] ?? path }))
    .sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0))
    .map(({ path, name }, index, files) => {
      const previous = files[index - 1];
      if (previous?.name === name) {
        throw new RegistryError(`${previous.path} and ${path} have the same file name`);
      }
      const entry = modules[path];
      if (entry === undefined) throw new RegistryError(`${path} has no default export`);
      if (entry.id !== name || !ENTRY_ID.test(entry.id)) {
        throw new RegistryError(
          `${path} exports the id ${JSON.stringify(entry.id)}; an id is the kebab-case file name`,
        );
      }
      return entry;
    });

/** Every entry this build holds, checked (the shell and the Node run call this at start). */
export const registeredEntries = (): readonly Entry[] =>
  entriesOf({ ...MODULES, ...LOCAL_MODULES });
