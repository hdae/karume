/**
 * Querying the cache inventory ({@link listCachedAssets}) and deleting **per selection**
 * ({@link evictCachedAssets}). A fine-grained surface next to `clearHubCache` (the whole
 * namespace); the unit is the same as for fetching, "one model / quant pair".
 *
 * Deletion is not just "delete the target's refs" because several selections share the same files
 * (switching the quant keeps the same tokenizer or f16 text_encoder). Deleting naively would
 * silently drop selections that were not deleted into a partial inventory, and the next start would
 * re-download just those. So **reference counting** sits in between: if another selection is
 * complete in the inventory (= all of its refs are present), the files it uses are kept.
 *
 * However, **a selection with exactly the same ref set as the target is not counted as a
 * protector** (for example `f16` and `f16-c16`, same weights, differing only in the session's
 * compute knobs). The cache granularity cannot tell them apart and "delete only one of them" is
 * impossible by definition, so counting it as a protector would give a lying answer that claims
 * "protected" while nothing is deleted. {@link EvictedAssets.alsoEvicted} names which selections
 * were dropped into a partial inventory as collateral. When the app wants to decide the protectors,
 * it names them with {@link CacheInventoryOptions.protect}.
 *
 * MUST: inventory queries are **delegated to the source** (`source.ts` (6) (7)) — the spelling of
 * cache keys belongs to the fetch layer, and if hub built them, every new version of the fetch
 * layer could produce entries "believed deleted but still there".
 * MUST: ref identity is {@link fileRefKey} (a cross-repo reference counts a same-named path of
 * another repo as a separate file).
 *
 * NOTE: the entry of the manifest itself (`karume.json`) is out of scope — it is small, keyed by
 * URL, and owned by no selection (deleting everything is `clearHubCache`).
 *
 * ## How narrowing the weights ({@link ResolveOptions.weights}) affects the counting
 *
 * A selection is a {@link ResolveOptions}, so both the target and `protect` can take "a subset of
 * the weights". It has one effect — **the ref set shrinks by what was narrowed** (the result of
 * `resolveSelection` is the input to the counting as is):
 *
 * - Query ({@link listCachedAssets}) … `cached` / `missing` count only the narrowed refs. Whether a
 *   role (gemma4's `drafter`) has been downloaded can be asked independently of the main model.
 * - Deletion ({@link evictCachedAssets}) … only the narrowed refs are deletion candidates. **The
 *   rest of the same (model, quant) is not counted as a protector** (protector candidates are
 *   "selections with a label different from the target's"; another subset of the same label is not
 *   a candidate), so for the weights, "keep the main model and delete only the drafter" can be
 *   written (but the shared assets — tokenizer / PLE — are not protected; see
 *   `docs/known-issues.md`).
 * - Collateral ({@link EvictedAssets.alsoEvicted}) … counts only **other (model, quant) pairs**;
 *   the rest of the same selection is not listed. Whether something became partial within the same
 *   label can, if needed, be checked by calling {@link listCachedAssets} again (the label stays
 *   `"<model>/<quant>"`, with no field naming the subset).
 * - Labels ({@link KeptAsset.sharedWith} / {@link EvictedAssets.alsoEvicted}) stay
 *   `"<model>/<quant>"` even when narrowed. Listing two subsets of the same label in `protect`
 *   makes **both protect** (`sharedWith` is deduplicated so the same label does not appear twice).
 */

import { HubError } from "./errors.ts";
import { crossRefOf, type FileRef, fileRefKey, type Manifest } from "./manifest.ts";
import { type ResolveOptions, resolveSelection, selectionRefs } from "./resolve.ts";
import { type LoadedManifest, pinnedSourceOf } from "./session.ts";
import { type PinnedSource, sourceForRef } from "./source.ts";

/** Options for inventory query and deletion (the same replacement points as the fetch surfaces). */
export type CacheInventoryOptions = {
  /**
   * Replacement for `CacheStorage` (for tests). When omitted, the fetch layer's default =
   * `globalThis.caches`.
   */
  readonly caches?: CacheStorage;
  /**
   * Names the protecting selections explicitly (**read only by {@link evictCachedAssets}** — an
   * inventory query has no protectors). When omitted, every selection of the manifest is a
   * candidate (except those with the same ref set as the target); when given, only this list is. In
   * both cases "only selections with all refs in the inventory actually protect".
   *
   * When given, same-set selections are **not** excluded — protecting a sibling selection by name
   * yields `kept: "shared"`. This option leaves the policy "how to treat same-set selections" to
   * the app. The target itself is ignored if listed (= the same (model, quant) is never a
   * candidate, even when narrowed with `weights`). A nonexistent model / quant / weights is a
   * `ManifestReferenceError`.
   */
  readonly protect?: readonly ResolveOptions[];
};

/**
 * The result of {@link listCachedAssets}. Together they are all refs of the selection (deduplicated
 * by `fileRefKey`).
 */
export type CachedAssets = {
  /** Refs in the inventory (in `selectionRefs` order). */
  readonly cached: readonly FileRef[];
  /** Refs not in the inventory (in `selectionRefs` order). */
  readonly missing: readonly FileRef[];
};

/** One ref whose deletion was skipped, and why. */
export type KeptAsset = {
  readonly ref: FileRef;
  /**
   * `"shared"` = a protecting selection (one with all files in the inventory) references it /
   * `"cross-repo"` = a cross-repo reference, whose file is owned by the target repo.
   *
   * By default the protector candidates are "the other selections of the same manifest whose ref
   * set differs from the target's"; with {@link CacheInventoryOptions.protect}, only that list.
   */
  readonly reason: "shared" | "cross-repo";
  /** When `reason` is `"shared"`, the labels `"<model>/<quant>"` of the protecting selections. */
  readonly sharedWith: readonly string[];
};

/**
 * The result of {@link evictCachedAssets}. Refs that were not in the inventory to begin with appear
 * in neither.
 */
export type EvictedAssets = {
  /** Refs actually removed from the inventory (in `selectionRefs` order). */
  readonly evicted: readonly FileRef[];
  /** Refs not deleted, and why (in `selectionRefs` order). */
  readonly kept: readonly KeptAsset[];
  /**
   * The labels `"<model>/<quant>"` of **selections dropped into a partial inventory as collateral**
   * of this deletion (in manifest declaration order). Among all selections of the manifest except
   * the target, those that had all refs in the inventory before the call and used at least one of
   * {@link EvictedAssets.evicted}.
   *
   * Decided by **the refs actually deleted**, not the candidates (the same contract as `evicted` —
   * only what the source reports as deleted). By default the same-set siblings of the target appear
   * here; with `protect`, every complete selection that was not protected appears. The app only
   * needs to withdraw its "downloaded" display for the selections in this list.
   */
  readonly alsoEvicted: readonly string[];
};

/** One pair of real (model, quant) names (the names after resolving the defaults). */
type Selection = { readonly model: string; readonly quant: string };

const labelOf = (selection: Selection): string => `${selection.model}/${selection.quant}`;

/**
 * Deduplicates by `fileRefKey` (the same rule and order as `fetchAssets`). Even if several keys
 * point at the same path, one file in the inventory is one file.
 */
const uniqueRefs = (refs: Iterable<FileRef>): readonly FileRef[] => {
  const unique = new Map<string, FileRef>();
  for (const ref of refs) {
    const key = fileRefKey(ref);
    if (!unique.has(key)) unique.set(key, ref);
  }
  return [...unique.values()];
};

/**
 * The ref list of one selection (zero-length parts dropped, deduplicated by {@link fileRefKey} —
 * the contract of `selectionRefs`). A nonexistent model / quant is a `ManifestReferenceError` here.
 */
const refsOf = (manifest: Manifest, selection: ResolveOptions): readonly FileRef[] =>
  selectionRefs(resolveSelection(manifest, selection));

/** Every (model, quant) pair of the manifest. */
const allSelections = (manifest: Manifest): readonly Selection[] =>
  Object.entries(manifest.models).flatMap(([model, entry]) =>
    Object.keys(entry.quants).map((quant): Selection => ({ model, quant })),
  );

/**
 * Normalises a selection to real names. The existence check belongs to {@link resolveSelection}, so
 * call this **only after going through it** (do not grow the source of truth for diagnostics to two
 * places).
 */
const namedSelection = (manifest: Manifest, selection: ResolveOptions): Selection => {
  const model = selection.model ?? manifest.defaultModel;
  if (!Object.hasOwn(manifest.models, model)) {
    throw new Error(`hub: model '${model}' が resolveSelection 通過後に引けない（不変条件破れ）`);
  }
  return { model, quant: selection.quant ?? manifest.models[model].defaultQuant };
};

/**
 * One selection and its ref list (carried as a pair through inventory matching and the protector
 * check).
 */
type SelectionRefs = { readonly selection: Selection; readonly refs: readonly FileRef[] };

/**
 * Whether two ref lists are **the same set** (order does not matter). Both went through
 * {@link uniqueRefs}, so equal counts plus one side fully contained means the sets are equal.
 */
const sameRefSet = (left: readonly FileRef[], right: readonly FileRef[]): boolean => {
  if (left.length !== right.length) return false;
  const keys = new Set(left.map(fileRefKey));
  return right.every((ref) => keys.has(fileRefKey(ref)));
};

/**
 * Normalises the `protect` list into protector candidates. It goes through `resolveSelection`
 * before normalising to real names, so a nonexistent model / quant / weights is a
 * `ManifestReferenceError` here. Duplicates are dropped, and so is the target itself (a target
 * cannot protect against itself).
 *
 * MUST: the deduplication key is **label + ref set**, not the label. A selection narrowed with
 * `weights` protects a different set under the same label, so folding by label alone would silently
 * drop one of the entries, last one wins ("files that should have been protected are gone" = a
 * mix-up with no diagnostic).
 */
const protectorsOf = (
  manifest: Manifest,
  entries: readonly ResolveOptions[],
  target: Selection,
): readonly SelectionRefs[] => {
  const byKey = new Map<string, SelectionRefs>();
  for (const entry of entries) {
    const refs = refsOf(manifest, entry);
    const selection = namedSelection(manifest, entry);
    if (selection.model === target.model && selection.quant === target.quant) continue;
    // The refs are in `selectionRefs` declaration order, so the same subset yields the same key.
    // MUST: build it with a spelling whose element boundaries are unambiguous (a `JSON.stringify`
    // array) — merely inserting a separator would give a selection whose label contains that
    // character the same key as another selection, and one would silently disappear, last one wins
    // (the manifest does not reject control characters in model / quant names, so names with NUL
    // really arrive).
    byKey.set(JSON.stringify([labelOf(selection), refs.map(fileRefKey)]), {
      selection,
      refs,
    });
  }
  return [...byKey.values()];
};

/**
 * Groups refs by origin, queries the inventory, and returns the `fileRefKey`s of those present.
 *
 * MUST: origin identity is the (repo, revision) pair of {@link crossRefOf}, not `origin.label` (a
 * display string for diagnostics) — the label is a display field whose vocabulary differs per
 * source; the same coordinates may carry different labels, and different coordinates the same
 * label.
 */
const queryInventory = async (
  source: PinnedSource,
  refs: readonly FileRef[],
): Promise<ReadonlySet<string>> => {
  const groups = new Map<string, FileRef[]>();
  for (const ref of refs) {
    const cross = crossRefOf(ref);
    // The empty string is the session's source (cross-repo coordinates always take the form
    // `<owner/name>@<sha>`).
    const origin = cross === undefined ? "" : `${cross.repo}@${cross.revision}`;
    const bucket = groups.get(origin);
    if (bucket === undefined) groups.set(origin, [ref]);
    else bucket.push(ref);
  }
  const cached = new Set<string>();
  for (const bucket of groups.values()) {
    const originSource = sourceForRef(source, bucket[0]);
    const inventory = originSource.inventory;
    if (inventory === undefined) {
      // Both built-in sources have it, so reaching here means a broken source contract (not caused
      // by user input) — fail with a bare `Error`, not a `HubError`.
      throw new Error(
        `hub: 取得元 ${originSource.origin.label} が在庫の照会を持たない（取得元契約の不変条件破れ）`,
      );
    }
    for (const key of await inventory(bucket)) cached.add(key);
  }
  return cached;
};

/**
 * Queries whether the refs of one selection "can be read without going to the network". **It
 * fetches nothing** (only an inventory query; it neither downloads what is missing nor verifies).
 *
 * Uses: showing "has this quant been downloaded", and deciding before calling `prefetchAssets`.
 * **For a source with a cache (HF)**, everything is `missing` in an environment without
 * `CacheStorage` (as the fetch layer's contract says — the cache is an optimisation, not a
 * correctness requirement, so hub does not special-case it). A local directory is read without a
 * cache, so it answers "everything passed in is in the inventory" regardless of `CacheStorage`.
 *
 * On runtimes without `Cache.keys()` (Deno 2.8 and earlier) the fetch layer throws, failing loudly
 * (a bare `Error`, not a {@link HubError}).
 */
export const listCachedAssets = async (
  loaded: LoadedManifest,
  selection: ResolveOptions = {},
  options: CacheInventoryOptions = {},
): Promise<CachedAssets> => {
  const source = pinnedSourceOf(loaded, options);
  const refs = refsOf(loaded.manifest, selection);
  const stock = await queryInventory(source, refs);
  return {
    cached: refs.filter((ref) => stock.has(fileRefKey(ref))),
    missing: refs.filter((ref) => !stock.has(fileRefKey(ref))),
  };
};

/**
 * Deletes the inventory of one selection to free space. It deletes **only as far as no other
 * selection breaks**:
 *
 * - If a protecting selection **has all its refs in the inventory**, the files it uses are kept
 *   (`kept: "shared"`). A partially cached selection does not protect — it fetches the rest the
 *   next time it is used anyway, so letting it protect would leave only "files that cannot be
 *   deleted yet cannot be used".
 * - By default the protector candidates are "the other selections of the same manifest" **minus
 *   those with the same ref set as the target**. A same-set selection (a quant with the same
 *   weights, differing only in session settings) cannot be told apart at cache granularity, and
 *   counting it as a protector would delete nothing. Passing `protect` makes only that list the
 *   candidates, without the same-set exclusion (the app decides the protection policy).
 * - Selections dropped into a partial inventory as collateral are named by `alsoEvicted`
 *   (selections that used at least one deleted ref and were complete before the call). Refetching
 *   what is missing on the next load restores them.
 * - Cross-repo references are kept (`kept: "cross-repo"`). The file is owned by the target repo,
 *   and deleting it from the referencing manifest would mean "deleting another repo's inventory for
 *   the convenience of whoever happens to reference it". To delete it, open the target repo's
 *   manifest and delete there.
 * - Refs not in the inventory to begin with do not appear in the result (nothing to delete or
 *   keep).
 *
 * A source without a cache (a local directory) is refused with a {@link HubError} — its contents
 * are the user's assets, not fetched copies, so nothing there is hub's to delete. Deleting the
 * whole namespace is `clearHubCache`.
 *
 * On runtimes without `Cache.keys()` (Deno 2.8 and earlier) the fetch layer throws, failing loudly
 * (a bare `Error`, not a {@link HubError}).
 */
export const evictCachedAssets = async (
  loaded: LoadedManifest,
  selection: ResolveOptions = {},
  options: CacheInventoryOptions = {},
): Promise<EvictedAssets> => {
  const { manifest } = loaded;
  const source = pinnedSourceOf(loaded, options);
  const targets = refsOf(manifest, selection);
  const evictRefs = source.evict;
  if (evictRefs === undefined) {
    throw new HubError(
      `evictCachedAssets: 取得元 ${source.origin.label} はキャッシュを持たない` +
        `（手元のディレクトリは取得物ではなく利用者の資産 — 消さない）`,
      { available: manifest.available },
    );
  }

  const target = namedSelection(manifest, selection);
  const others: readonly SelectionRefs[] = allSelections(manifest)
    .filter((other) => other.model !== target.model || other.quant !== target.quant)
    .map((other) => ({ selection: other, refs: refsOf(manifest, other) }));
  // Protector candidates. Same-set selections are excluded by default because cache granularity
  // cannot tell them from the target, and "delete only one" is impossible by definition — counting
  // them as protectors would always make every file `shared`, a lie that claims "protected" while
  // nothing is deleted. Proper subsets and supersets protect as before.
  const protectors =
    options.protect === undefined
      ? others.filter((other) => !sameRefSet(other.refs, targets))
      : protectorsOf(manifest, options.protect, target);
  // The inventory is queried **once over the union of all selections** (per origin). Querying per
  // selection would repeat the full enumeration of the same repo once per selection. Protectors are
  // selections of the manifest, so their refs are always in this union.
  const stock = await queryInventory(
    source,
    uniqueRefs([...targets, ...others.flatMap(({ refs }) => refs)]),
  );
  const isCached = (ref: FileRef): boolean => stock.has(fileRefKey(ref));

  const sharedWith = new Map<string, string[]>();
  for (const other of protectors) {
    if (!other.refs.every(isCached)) continue;
    const label = labelOf(other.selection);
    for (const ref of other.refs) {
      const key = fileRefKey(ref);
      const labels = sharedWith.get(key);
      if (labels === undefined) sharedWith.set(key, [label]);
      // Even if two subsets of the same label protect, the label is named once (this happens only
      // when `protect` lists entries differing in `weights` — repeating the name adds nothing for
      // the reader).
      else if (!labels.includes(label)) labels.push(label);
    }
  }

  const kept: KeptAsset[] = [];
  const evictable: FileRef[] = [];
  for (const ref of targets) {
    if (!isCached(ref)) continue;
    if (crossRefOf(ref) !== undefined) {
      kept.push({ ref, reason: "cross-repo", sharedWith: [] });
      continue;
    }
    const labels = sharedWith.get(fileRefKey(ref));
    if (labels !== undefined) {
      kept.push({ ref, reason: "shared", sharedWith: labels });
      continue;
    }
    evictable.push(ref);
  }

  // The source reports what was deleted (count 0 = someone deleted it first). The order follows the
  // target list.
  const removed = new Set((await evictRefs(evictable)).map(fileRefKey));
  const evicted = evictable.filter((ref) => removed.has(fileRefKey(ref)));
  // Collateral is also judged by **the set actually deleted** (= `evicted` itself). Counting the
  // candidates would also name the users of refs the source failed to delete as "dropped into a
  // partial inventory". Presence in the inventory is taken from the observation before the call
  // (`stock`).
  const evictedKeys = new Set(evicted.map(fileRefKey));
  const alsoEvicted = others
    .filter(
      (other) =>
        other.refs.every(isCached) && other.refs.some((ref) => evictedKeys.has(fileRefKey(ref))),
    )
    .map((other) => labelOf(other.selection));
  return { evicted, kept, alsoEvicted };
};
