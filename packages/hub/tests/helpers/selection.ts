/**
 * Test-side tools for selections. The `karume/5` hub has no flat "fetch key → FileRef" table (a
 * selection is a structured type — ADR 0109 decision 6), so the tests build the table passed to
 * `fetchAssets` and the mock's serving table.
 *
 * MUST: do not copy paths by hand — deriving them from the manifest makes the test-side tables move
 * along when the fixture is regenerated (a copy silently leads to "the fixture was fixed but the
 * tests keep serving the old path").
 */

import type { FileRef, Manifest, ResolveOptions } from "../../mod.ts";
import { resolveSelection } from "../../mod.ts";

/**
 * Lists every FileRef the manifest declares **in declaration order with unique paths** (zero-length
 * parts are left out — there is no content to serve). Used to build the mock's serving table.
 */
export const declaredRefs = (manifest: Manifest): readonly FileRef[] => {
  const seen = new Map<string, FileRef>();
  const put = (ref: FileRef): void => {
    if (ref.size > 0 && !seen.has(ref.path)) seen.set(ref.path, ref);
  };
  for (const model of Object.keys(manifest.models)) {
    const entry = manifest.models[model];
    for (const weight of Object.keys(entry.weights)) {
      const dtypes = entry.weights[weight];
      for (const dtype of Object.keys(dtypes)) {
        for (const ref of dtypes[dtype].container.parts) put(ref);
      }
    }
    for (const name of Object.keys(entry.assets)) put(entry.assets[name]);
  }
  return [...seen.values()];
};

/**
 * Turns a selection into a "fetch key → FileRef" table. The keys are spellings for the tests to
 * read; the hub does not have them (`fetchAssets` only returns the keys of the table it was given):
 *
 * - a container part … `<component name>[<part index>]` (zero-length parts do not appear)
 * - assets … the asset name as is
 *
 * The order is the same declaration order as `selectionRefs`. **Keys that point at the same file
 * are not dropped** (two names can point at one file, as with an aliased asset — `fetchAssets`
 * dedupes fetches and progress).
 */
export const selectionFiles = (
  manifest: Manifest,
  options: ResolveOptions = {},
): Record<string, FileRef> => {
  const selection = resolveSelection(manifest, options);
  const files: Record<string, FileRef> = {};
  for (const name of Object.keys(selection.containers)) {
    selection.containers[name].parts.forEach((ref, index) => {
      if (ref.size > 0) files[`${name}[${index}]`] = ref;
    });
  }
  for (const name of Object.keys(selection.assets)) files[name] = selection.assets[name];
  return files;
};
