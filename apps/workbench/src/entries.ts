/**
 * The registry: every file `src/entries/<id>.ts` default-exports one {@link Entry}, and its id is
 * its file name. An entry that reads a local distribution (`readsLocalDistribution`) is a file
 * `src/entries/local/<id>.ts`. Adding an entry is adding a file; nothing else lists them.
 */

import { type Entry, ENTRY_ID } from "./entry.ts";
import { LOCAL_MODULES } from "./local-entries.ts";

const MODULES = import.meta.glob<Entry>("./entries/*.ts", { eager: true, import: "default" });

/** Where the registry's glob finds the entries that read a local distribution. */
const LOCAL_DIRECTORY = "./entries/local/";

/** The registry does not hold what it promises (a missing default export, an id off its file name). */
export class RegistryError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "RegistryError";
  }
}

/**
 * The entries of `modules` (glob path → default export) in file-name order, or a
 * {@link RegistryError} for the first file whose default export is missing, whose id is not its
 * kebab-case file name or is another file's too, or whose `readsLocalDistribution` disagrees with
 * its directory (a file under `entries/local/` carries the mark, any other file does not).
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
      const local = path.startsWith(LOCAL_DIRECTORY);
      if ((entry.readsLocalDistribution === true) !== local) {
        throw new RegistryError(
          local
            ? `${path} is under ${LOCAL_DIRECTORY} but does not declare readsLocalDistribution`
            : `${path} declares readsLocalDistribution; such an entry is a file under ${LOCAL_DIRECTORY}`,
        );
      }
      return entry;
    });

/**
 * The entries a build lists: a production build serves no distribution, so it leaves out every
 * entry that reads one; the dev server and the Node run list them all.
 */
export const listedEntries = (entries: readonly Entry[], production: boolean): readonly Entry[] =>
  production ? entries.filter((entry) => entry.readsLocalDistribution !== true) : entries;

/** Every entry this build lists, checked (the shell and the Node run call this at start). */
export const registeredEntries = (): readonly Entry[] =>
  listedEntries(entriesOf({ ...MODULES, ...LOCAL_MODULES }), import.meta.env.PROD);
