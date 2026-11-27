/**
 * Where a record lands: `outputs/workbench/<host>-<adapter name>/<entry id>_<UTC stamp>.json` under
 * the repository root. Pure: the Node side (`../node.ts`) writes by it, and a page names a download
 * by the same adapter rule.
 *
 * Nothing a client sends becomes a path segment except through two rules: the entry id must be
 * kebab-case, and the adapter directory keeps only `[a-z0-9-]`.
 *
 * SPEC: docs/spec/workbench.md#where-records-go
 */

import { ENTRY_ID, type Host } from "./record.ts";

/** The results directory, relative to the repository root (outside git: `outputs/` is ignored). */
export const RESULTS_DIRECTORY = "outputs/workbench";

const MAX_SLUG_LENGTH = 80;

/** The adapter strings of a record, as far as they are strings. */
export type AdapterNames = {
  readonly vendor?: unknown;
  readonly architecture?: unknown;
  readonly device?: unknown;
  readonly description?: unknown;
};

const slugOf = (text: string): string =>
  text
    .toLowerCase()
    .replaceAll(/[^a-z0-9]+/g, "-")
    .replaceAll(/^-+|-+$/g, "")
    .slice(0, MAX_SLUG_LENGTH)
    .replaceAll(/-+$/g, "");

const textOf = (value: unknown): string => (typeof value === "string" ? value : "");

/**
 * The directory name of a host's adapter, `<host>-<adapter name>`, by the rule of the tests'
 * environment key: the name is the `description` in a browser and the `device` under Node — Dawn
 * puts the driver version in `description`, and a directory MUST NOT change with a driver update —
 * else `vendor-architecture` (Chrome may leave the description empty), else `unknown-adapter`
 * (a candidate whose slug is empty counts as absent).
 */
export const adapterDirectory = (host: Host, adapter: AdapterNames): string => {
  const name =
    [
      textOf(host === "node" ? adapter.device : adapter.description),
      `${textOf(adapter.vendor)}-${textOf(adapter.architecture)}`,
    ]
      .map(slugOf)
      .find((slug) => slug !== "") ?? "unknown-adapter";
  return `${host}-${name}`;
};

/** What decides a record's path. */
export type RecordTarget = {
  readonly entryId: string;
  readonly host: Host;
  readonly adapter: AdapterNames;
};

/**
 * The repository-relative path (with `/`) of a record of `target` stamped `stamp` (`utcStamp` of a
 * moment): attempt 0 is `<entry id>_<stamp>.json`, attempt n adds `-n` after the stamp. Throws when
 * the entry id is not kebab-case.
 */
export const recordPath = (target: RecordTarget, stamp: string, attempt = 0): string => {
  if (!ENTRY_ID.test(target.entryId)) {
    throw new Error(`The entry id ${JSON.stringify(target.entryId)} is not kebab-case`);
  }
  const directory = `${RESULTS_DIRECTORY}/${adapterDirectory(target.host, target.adapter)}`;
  return `${directory}/${target.entryId}_${stamp}${attempt === 0 ? "" : `-${attempt}`}.json`;
};
