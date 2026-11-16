/**
 * The page's URL: the selected entry, its parameter values and the declared backend, so a link
 * reproduces a run setup. Both directions are pure; the shell reads `location.search` and writes
 * with `history.replaceState`.
 *
 * SPEC: docs/spec/workbench.md#the-shell
 */

import { type Entry, parameterText, type ParameterValues } from "../entry.ts";

/** The query key of the selected entry. */
export const ENTRY_KEY = "entry";

/** The query key of the declared backend. */
export const BACKEND_KEY = "backend";

/** Query keys that are not parameters. */
const RESERVED_KEYS: readonly string[] = [ENTRY_KEY, BACKEND_KEY];

/** The parameters a query gives, as the string map `parseParameters` reads (the last of a repeated key wins). */
export const givenOfQuery = (query: URLSearchParams): Record<string, string> =>
  Object.fromEntries([...query].filter(([name]) => !RESERVED_KEYS.includes(name)));

/** The query of `entry` at `values`, with `backend` when it is not empty. */
export const queryOf = (
  entry: Entry,
  values: ParameterValues,
  backend: string,
): URLSearchParams => {
  const query = new URLSearchParams({ [ENTRY_KEY]: entry.id });
  for (const parameter of entry.parameters) {
    query.set(parameter.name, parameterText(values[parameter.name]));
  }
  if (backend !== "") query.set(BACKEND_KEY, backend);
  return query;
};
