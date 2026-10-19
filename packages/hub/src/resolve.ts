/**
 * model / quant → **a structured selection** (ADR 0109 decision 6). The flat "fetch key → FileRef"
 * table of `karume/4` (the spelling convention `<weights>[i]` / `<weights>.<extra>`) is retired,
 * and a selection is expressed by two fields:
 *
 * - component name → container reference ({@link ResolvedSelection.containers})
 * - asset name → FileRef ({@link ResolvedSelection.assets})
 *
 * The flat FileRef list that fetching, inventory and progress need is **derived from them** by
 * {@link selectionRefs} — there is no second, independently updated table.
 */

import { ManifestReferenceError } from "./errors.ts";
import type { ContainerRef, FileRef, Manifest, ModelEntry } from "./manifest.ts";
import { fileRefKey } from "./manifest.ts";

/**
 * The result of one selection. `model` / `quant` are **the real names after resolving the
 * defaults**, so callers can use "which model and preset were actually chosen" as is for display
 * and diagnostics.
 */
export type ResolvedSelection = {
  readonly model: string;
  readonly quant: string;
  /**
   * Component name → container (in the declaration order of the manifest's `weights`). When a
   * subset is requested with {@link ResolveOptions.weights}, only that subset.
   */
  readonly containers: Readonly<Record<string, ContainerRef>>;
  /**
   * Asset name → FileRef (**always all of them**, independent of the quant and of the weights
   * subset).
   */
  readonly assets: Readonly<Record<string, FileRef>>;
};

/**
 * The selection axes of {@link resolveSelection}. Each one left out falls back to the manifest's
 * default.
 */
export type ResolveOptions = {
  /** Model name (`defaultModel` when omitted). */
  readonly model?: string;
  /** Quant name (the model's `defaultQuant` when omitted). */
  readonly quant?: string;
  /**
   * The **subset** of weights to fetch (all of the model's weights when omitted).
   *
   * This axis handles, without splitting the distribution, a model that "runs with the main part
   * alone and can also add extra roles" (gemma4's `model` + the speculative-decoding `drafter`).
   * The containers of unused roles **do not appear** in the table, so the download, the progress
   * total and the inventory count all shrink by that role.
   *
   * MUST: an unknown weights name is a {@link ManifestReferenceError} (a typo is not silently
   * folded into "do not fetch"). Listing the same name twice is rejected too — as a subset it is a
   * spelling without meaning.
   *
   * NOTE: the table is in **declaration order** (the order of the manifest's `weights`), not the
   * order of this array — if the caller's ordering moved the fetch dispatch order, the same
   * manifest would be seen in a different order by each caller.
   * NOTE: an empty array means "fetch no weights at all (assets only)" — it is not rejected (it is
   * a defined value of a subset, and a selection with no containers is returned as requested).
   */
  readonly weights?: readonly string[];
};

/**
 * Decides the weights names to fetch in **declaration order** (all of them when omitted).
 *
 * MUST: the existence check lives here alone (parsing already guarantees that `quants` maps
 * completely, so as long as the name exists its dtype label is always found).
 */
const selectWeightNames = (
  entry: ModelEntry,
  weights: readonly string[] | undefined,
): readonly string[] => {
  const declared = Object.keys(entry.weights);
  if (weights === undefined) return declared;
  const chosen = new Set<string>();
  for (const name of weights) {
    if (!Object.hasOwn(entry.weights, name)) {
      throw new ManifestReferenceError(
        `weights '${name}' is not in the manifest (available: ${declared.join(" / ")})`,
        { available: entry.available },
      );
    }
    if (chosen.has(name)) {
      throw new ManifestReferenceError(
        `weights '${name}' was given twice (list the subset of weights to fetch without duplicates)`,
        { available: entry.available },
      );
    }
    chosen.add(name);
  }
  return declared.filter((name) => chosen.has(name));
};

/**
 * Selects the model. An unknown model name fails with **the list of what is available** (the
 * enumeration first became machine-readable in v2 — ADR 0041 decision 8).
 *
 * MUST: the default is resolved (falling back to `defaultModel`) in this one place, and **the name
 * is returned with it** — if callers wrote the same expression again, a change to how the default
 * is chosen could make "the returned model name" and "the ModelEntry actually looked up" point at
 * different models.
 */
const selectModel = (
  manifest: Manifest,
  model?: string,
): { readonly name: string; readonly entry: ModelEntry } => {
  const name = model ?? manifest.defaultModel;
  if (!Object.hasOwn(manifest.models, name)) {
    throw new ManifestReferenceError(
      `model '${name}' is not in the manifest (available: ${manifest.available.models.join(" / ")})`,
      { available: manifest.available },
    );
  }
  return { name, entry: manifest.models[name] };
};

/**
 * Selects a model and a quant and builds **the selection**. `defaultModel` / `defaultQuant` when
 * omitted. Passing {@link ResolveOptions.weights} lists only that subset (assets are always all).
 *
 * Resolvability (completeness of the weights mapping, existence of the dtype labels) is checked at
 * parse time, so the only new failure here is "a model / quant / weights name that does not exist".
 * Component names and asset names go into **separate fields**, so a name collision can no longer
 * happen (the fetch-key table of `karume/4` folded three namespaces into one, so it needed that
 * check).
 */
export const resolveSelection = (
  manifest: Manifest,
  options: ResolveOptions = {},
): ResolvedSelection => {
  const { name: model, entry } = selectModel(manifest, options.model);
  const quant = options.quant ?? entry.defaultQuant;
  if (!Object.hasOwn(entry.quants, quant)) {
    throw new ManifestReferenceError(
      `quant '${quant}' is not in the manifest (available: ${entry.available.quants.join(" / ")})`,
      { available: entry.available },
    );
  }
  const labels = entry.quants[quant].weights;
  let containers: Readonly<Record<string, ContainerRef>> = {};
  for (const name of selectWeightNames(entry, options.weights)) {
    containers = { ...containers, [name]: entry.weights[name][labels[name]].container };
  }
  // MUST: build **a new table** for assets as well (never return the parsed manifest's table
  // itself) — if someone assigned to a field of the return value, every later selection, inventory
  // count and evict would see the polluted table. It also keeps the lifetime handling of the two
  // fields the same (containers is rebuilt every time).
  return { model, quant, containers, assets: { ...entry.assets } };
};

/**
 * Folds a selection into **the flat FileRef list that fetching, inventory and progress handle**.
 * The order is declaration order (containers in `weights` order, parts by index within each, assets
 * last in their order).
 *
 * MUST: **drop zero-length parts** (ADR 0109 decision 3). Fetching a 0-byte file brings no bytes,
 * and readers do not read once `partLength` answers 0 — leaving it in the list only inflates the
 * progress total and the file count.
 * MUST: **deduplicate** by identity ({@link fileRefKey}). Two names pointing at the same file (an
 * aliased asset, an asset shared by several components) is a valid manifest, but the surface that
 * takes this list (`prefetchAssets`) rejects duplicates as a caller error. The folding happens in
 * this one place, where the table is built; writing it per surface would forget it on one side.
 */
export const selectionRefs = (selection: ResolvedSelection): readonly FileRef[] => {
  const unique = new Map<string, FileRef>();
  const put = (ref: FileRef): void => {
    const key = fileRefKey(ref);
    if (!unique.has(key)) unique.set(key, ref);
  };
  for (const name of Object.keys(selection.containers)) {
    for (const ref of selection.containers[name].parts) {
      if (ref.size > 0) put(ref);
    }
  }
  for (const name of Object.keys(selection.assets)) put(selection.assets[name]);
  return [...unique.values()];
};
