/**
 * The page's URL: the selected entry and its parameter values, or the selected batch, and the
 * declared backend, so a link reproduces a run setup. Both directions are pure; the shell reads
 * `location.search` and writes with `history.replaceState`.
 *
 * SPEC: docs/spec/workbench.md#the-shell
 */

import { type Entry, parameterText, type ParameterValues } from "../entry.ts";

/** The query key of the selected entry. */
export const ENTRY_KEY = "entry";

/** The query key of the selected batch. */
export const BATCH_KEY = "batch";

/** The query key of the declared backend. */
export const BACKEND_KEY = "backend";

/** What a query selects. An entry and a batch in one query is an error the shell shows. */
export type QuerySelection =
  | { readonly kind: "none" }
  | { readonly kind: "entry"; readonly id: string }
  | { readonly kind: "batch"; readonly id: string }
  | { readonly kind: "both"; readonly entry: string; readonly batch: string };

export const selectionOfQuery = (query: URLSearchParams): QuerySelection => {
  const entry = query.get(ENTRY_KEY);
  const batch = query.get(BATCH_KEY);
  if (entry !== null && batch !== null) return { kind: "both", entry, batch };
  if (entry !== null) return { kind: "entry", id: entry };
  if (batch !== null) return { kind: "batch", id: batch };
  return { kind: "none" };
};

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

/** The query of the batch `id`, with `backend` when it is not empty (the step selection is not in it). */
export const batchQueryOf = (id: string, backend: string): URLSearchParams => {
  const query = new URLSearchParams({ [BATCH_KEY]: id });
  if (backend !== "") query.set(BACKEND_KEY, backend);
  return query;
};
