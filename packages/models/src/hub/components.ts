/**
 * Internal mechanism that loads the manifest's weights components from **containers (`krm`)** into
 * Sessions (the one mechanism shared by every family's `fromPretrained` / `fromAssets` — ADR 0109
 * decisions 2 / 10, ADR 0108 decision 19).
 *
 * MUST: not exported from the barrel. It exists to keep the load paths spelled the same way; it is
 * not a surface the user touches.
 *
 * ## Order ("do not download a single byte of weights for a model that cannot run" — inherited from
 * ADR 0070 decision 5)
 *
 * 1. Warm and open **only part 0 (the descriptor)** of each component (`openContainer` — checks the
 *    two documents against the manifest's expected values and confirms **zero shortage / excess** in
 *    the merge of graph × binding table from the declarations alone). The capability check of
 *    `prepareContainer` also runs here.
 * 2. Pass the family check ({@link FamilyAdmission}). What is at hand is each component's graph
 *    declaration and the **declarations** of its assets (name → role, logical length). Assets that
 *    share no part with weight blocks (indexes, `rope_base` — the writer's convention, container-v1
 *    §4.2) may be read here too via `asset(name)` (the check of the PLE index against every asset
 *    entry does exactly that).
 * 3. Once it passes, download the weight parts (part 1 onward) into the persistent cache
 *    (`prefetchAssets` — progress, abort, 4 in parallel). Blocks are read in part order at the
 *    moment the Session is built (hub's fetch surface — what lands in host RAM is the overlap of
 *    blocks for a seek-type source, one part for a scan-type source).
 * 4. Fetch the remaining assets (tokenizer etc. — the manifest's `assets`) in full.
 *
 * ## Component-replacement option ({@link LoadContainerOptions.components} — ADR 0108 decision 19)
 *
 * Replaces a component by role → {@link ComponentSource} (the same role in another `karume/5`
 * repo). Admission runs **before a single byte of weights is fetched**: (1) the sha256 of the graph
 * descriptor matches the manifest's declaration (only components with an identical graph contract),
 * (2) zero shortage / excess in the binding table (`openContainer` reports it from the descriptor
 * alone), (3) the family check. Both shortage and excess are rejected with a full listing
 * (diffusers' `strict=False` is not adopted).
 *
 * ## Whole-bytes surface (`from*Assets`)
 *
 * The entries that build from already-fetched bytes fold into the same shape
 * ({@link ModelComponent}) via {@link assetComponentOpener}. A component key is either **the bytes
 * of a single-form `krm`** (`<role>`) or **the part list of a split form** (`<role>[i]` — in index
 * order from part 0). There are no expected values (the sha256 of the two documents) — the caller
 * vouches for the bytes at hand. To return a synchronous opener it **opens every component first**,
 * so on this surface opening the containers comes before family admission (the reverse of the
 * fetch surface) — the family's `fromAssets` entry checks for abort first.
 */

import {
  type AdmissionReport,
  type AssetReader,
  type ContainerInput,
  type EstimateOptions,
  type GpuContext,
  openContainer,
  type OpenedContainer,
  prepareContainer,
  type PreparedModel,
  type Session,
  type SessionOptions,
} from "@karume/runtime";
import {
  type AssetProgress,
  type ContainerRef,
  type DistributionSource,
  fetchAssets,
  type FetchAssetsOptions,
  type FileRef,
  type HubRepoRef,
  type LoadedManifest,
  loadManifest,
  openContainerSource,
  prefetchAssets,
  type ResolvedSelection,
  resolveSelection,
} from "@karume/hub";
import { toManifestSource } from "./repo-ref.ts";

/**
 * Something that carries a graph **declaration** (the common surface of the `PreparedModel` that
 * `prepareContainer` returns and {@link ModelComponent}).
 *
 * The check against the declaration (input/output counts, static dimensions agreeing with
 * `pipelineConfig`) should be written once for both the fetch surface and the whole-bytes surface,
 * so the check helpers accept only this surface.
 */
export type GraphOwner = {
  readonly graph: PreparedModel["graph"];
};

/** One component "before execution" = graph declaration + Session entry + the container's assets. */
export type ModelComponent = GraphOwner & {
  /** Builds one Session. Every call re-reads the blocks in part order (the stream is not reused). */
  readonly createSession: (gpu: GpuContext, options?: SessionOptions) => Promise<Session>;
  /**
   * Asset name → role as declared by the container (container-v1 §2.2 — the runtime does not
   * interpret roles). The family's way to see "this component has / has no PLE" from the
   * **declarations alone** (usable during admission).
   */
  readonly assets: Readonly<Record<string, string>>;
  /**
   * Reader for one asset (undeclared names fail loudly). Opening fetches no byte — range reads
   * happen only on `read(offset, length)` (PLE row reads and the full read of `rope_base` are such
   * reads).
   *
   * Every call returns a new reader. With an unverified source (the part list of the whole-bytes
   * surface, a local directory) a reader **holds its blocks for its lifetime** (the runtime's
   * `AssetReader`), so a family does not hold a reader longer than one read (holding it grows host
   * RAM outside the family's residency budget).
   */
  readonly asset: (name: string) => AssetReader;
  /**
   * Estimate of the demand-side byte count (`PreparedModel.estimate` itself — a pure function that
   * does not touch the GPU). Used by callers that measure "does it fit in the remaining VRAM" before
   * building a stage (anima's pre-emptive eviction of the resident DiT).
   */
  readonly estimate: (options: EstimateOptions) => AdmissionReport;
  /**
   * The largest byte length of a weight part (part 1 onward; 0 for a container whose only part is
   * part 0 = the descriptor).
   *
   * WHY: building a Session uploads one part at a time, and the implementation's staging (wgpu's
   * `queue.writeBuffer`) stays allocated until that part's fence — the transient overhead that the
   * estimate ({@link ModelComponent.estimate}) does not count is "one largest part".
   */
  readonly maxPartBytes: number;
};

/**
 * Role → component opener. On the whole-bytes surface it is "`openContainer` on the bytes at hand",
 * on the fetch surface "a lookup into what was opened and admitted"; the pipeline body assembles in
 * the same order without knowing which.
 */
export type ComponentOpener = (key: string) => ModelComponent;

/**
 * Source of a component replacement (ADR 0108 decision 19 / ADR 0109 decision 10) — **the same
 * role** in another `karume/5` repo. `model` / `quant` default to that repo's defaults when omitted.
 */
export type ComponentSource = {
  readonly source: string | HubRepoRef | DistributionSource;
  readonly model?: string;
  readonly quant?: string;
};

/**
 * Family-side admission — the hook that rejects "with this manifest / this graph, this cannot run
 * as this family" **before a single byte of the weight parts is fetched** (inherited from ADR 0070
 * decision 5).
 *
 * Called once, right after `openContainer` / `prepareContainer` (merge, shortage / excess,
 * capability) and **before** the weight prefetch; all that is at hand is each component's `IrGraph`
 * and the asset declarations ({@link ModelComponent.assets} / `asset(name).length`). The asset bytes
 * are not there yet (waiting for them would lose the position before the weight prefetch).
 *
 * MUST: the family runs **all of its checks** here and passes the return value (parsed config /
 * quant) and **the same opener** straight on to the later state construction — with two
 * implementations of the same check in the early and the late step, the moment only one of them is
 * updated it goes back to "passes early but fails later".
 */
export type FamilyAdmission<Admitted> = (open: ComponentOpener) => Admitted | Promise<Admitted>;

/** Return value of {@link loadContainerComponents}. */
export type ContainerComponents<Admitted> = {
  /** Opener for the weights components (anything outside the given `componentKeys` fails loudly). */
  readonly open: ComponentOpener;
  /** The bytes of the manifest's `assets` (tokenizer etc.) — received in full. */
  readonly assets: Record<string, Uint8Array<ArrayBuffer>>;
  /** The material that family admission ({@link FamilyAdmission}) settled. */
  readonly admitted: Admitted;
};

/** Extra options of {@link loadContainerComponents} (the fetch layer's options pass through as is). */
export type LoadContainerOptions = FetchAssetsOptions & {
  /**
   * Component-replacement option (role → source). The given roles MUST be a subset of
   * `componentKeys` (an unknown role fails loudly — a misspelling is not silently folded into
   * "replace nothing").
   */
  readonly components?: Readonly<Record<string, ComponentSource>>;
};

/** The largest byte length of a weight part (part 1 onward — {@link ModelComponent.maxPartBytes}). */
const maxWeightPartBytes = (opened: OpenedContainer): number => {
  let max = 0;
  for (let index = 1; index < opened.source.partCount; index += 1) {
    max = Math.max(max, opened.source.partLength(index));
  }
  return max;
};

/** Folds one opened container into a {@link ModelComponent} (shared by the fetch and whole-bytes surfaces). */
const containerComponent = (opened: OpenedContainer, prepared: PreparedModel): ModelComponent => ({
  graph: prepared.graph,
  createSession: (gpu, options = {}) => prepared.createContainerSession(gpu, options),
  assets: Object.fromEntries(
    Object.entries(opened.model?.assets ?? {}).map(([name, asset]) => [name, asset.role]),
  ),
  asset: (name) => opened.asset(name),
  estimate: (options) => prepared.estimate(options),
  maxPartBytes: maxWeightPartBytes(opened),
});

/** Builds an opener from the table of opened components (a role that was not opened fails loudly). */
const openerOf =
  (where: string, components: ReadonlyMap<string, ModelComponent>): ComponentOpener =>
  (key) => {
    const component = components.get(key);
    if (component === undefined) {
      throw new Error(
        `${where}: component '${key}' is not open (open components: ${[...components.keys()].join(" / ")})`,
      );
    }
    return component;
  };

/** The index part of `<role>[<index>]` (decimal integers only). */
const PART_INDEX = /^\d+$/;

/** How one component of the whole-bytes surface is read (one single-form key / a list of part keys). */
type ComponentKeyPlan =
  | { readonly kind: "bytes"; readonly key: string }
  | { readonly kind: "parts"; readonly keys: readonly string[] };

/** Total number of keys shaped `<role>[<index>]` (its difference from the contiguous count is the number of gaps). */
const indexedKeyCount = (keys: readonly string[], componentKey: string): number => {
  const prefix = `${componentKey}[`;
  let count = 0;
  for (const key of keys) {
    if (!key.startsWith(prefix) || !key.endsWith("]")) continue;
    if (PART_INDEX.test(key.slice(prefix.length, -1))) count += 1;
  }
  return count;
};

/**
 * Whether one component of the whole-bytes surface is read as "one single form" or "a split form's
 * part list".
 *
 * MUST: the indexes run contiguously from `[0]` without gaps, and a bare key is not mixed with
 * `[i]` keys. Either violation marks broken key construction; skipping it silently fails far away,
 * in another layer, as "not enough parts".
 */
const planComponentKeys = (
  where: string,
  keys: readonly string[],
  componentKey: string,
): ComponentKeyPlan => {
  const has = (key: string): boolean => keys.includes(key);
  const partKeys: string[] = [];
  for (let index = 0; has(`${componentKey}[${index}]`); index += 1) {
    partKeys.push(`${componentKey}[${index}]`);
  }
  const indexed = indexedKeyCount(keys, componentKey);
  const available = ` (keys present: ${keys.join(" / ")})`;
  if (has(componentKey)) {
    if (indexed > 0) {
      throw new Error(
        `${where}: component '${componentKey}' arrived under both the single-form key and split-form keys` +
          ` ('${componentKey}[0]' etc.) (exactly one of the two MUST be used — ` +
          `${indexed} indexed keys)${available}`,
      );
    }
    return { kind: "bytes", key: componentKey };
  }
  if (indexed === 0) {
    throw new Error(
      `${where}: component '${componentKey}' has no container` +
        ` (single form: '${componentKey}'; split form: '${componentKey}[0]' onwards in index order)${available}`,
    );
  }
  if (partKeys.length === 0) {
    throw new Error(
      `${where}: the part indexes of component '${componentKey}' do not start at [0]` +
        ` ('${componentKey}[0]' is missing / ${indexed} indexed keys)${available}`,
    );
  }
  if (indexed !== partKeys.length) {
    throw new Error(
      `${where}: the part indexes of component '${componentKey}' are not contiguous from [0]` +
        ` (${partKeys.length} contiguous / ${indexed} indexed keys)${available}`,
    );
  }
  return { kind: "parts", keys: partKeys };
};

/**
 * Component opener of the whole-bytes surface (`from*Assets`) — the one shared by every family.
 *
 * It **opens every component of `componentKeys` first** (`openContainer` is asynchronous, so the
 * only way to return a synchronous {@link ComponentOpener} is to open first and hand out after).
 * `buffer` is the family's asset accessor (each single part goes through the same check too, so
 * that the "asset missing" / "bytes do not cover the whole buffer" messages stay with the family).
 */
export const assetComponentOpener = async (
  where: string,
  assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>,
  buffer: (key: string) => ArrayBuffer,
  componentKeys: readonly string[],
): Promise<ComponentOpener> => {
  const keys = Object.keys(assets);
  const components = new Map<string, ModelComponent>();
  for (const key of componentKeys) {
    const plan = planComponentKeys(where, keys, key);
    const input: ContainerInput =
      plan.kind === "bytes"
        ? { kind: "bytes", bytes: new Uint8Array(buffer(plan.key)) }
        : { kind: "parts", parts: plan.keys.map((partKey) => new Uint8Array(buffer(partKey))) };
    const opened = await openContainer(input);
    components.set(key, containerComponent(opened, prepareContainer(opened, key)));
  }
  return openerOf(where, components);
};

/**
 * Makes **one** progress stream for the whole model (MUST — a contract visible to consumers).
 *
 * Fetching splits into "warming the descriptors + warming the weight parts + fetching the assets in
 * full", and with the replacement option the sources split too, but the `total` the caller
 * sees is the size sum of every file fetched and `loaded` is the sum received. The per-file fields
 * (`fileLoaded` / `fileTotal`), `phase` and the fetch target (`repo` / `revision`) pass through from
 * the fetch layer.
 *
 * MUST: files are matched by (fetch target repo, revision, path) (the MUST NOT on
 * `AssetProgress.path`) — cross-repo references and the replacement option put the same path from
 * different repos into one load, so folding by path alone counts two files as one and makes `total`
 * smaller than what is actually fetched.
 */
const aggregateProgress = (
  sources: readonly { readonly loaded: LoadedManifest; readonly refs: readonly FileRef[] }[],
  onProgress: ((progress: AssetProgress) => void) | undefined,
): ((progress: AssetProgress) => void) | undefined => {
  if (onProgress === undefined) return undefined;
  const sizes = new Map<string, number>();
  for (const { loaded, refs } of sources) {
    for (const ref of refs) sizes.set(progressKeyOf(loaded, ref), ref.size);
  }
  let total = 0;
  for (const size of sizes.values()) total += size;
  const received = new Map<string, number>();
  return (progress) => {
    received.set(progressKey(progress.repo, progress.revision, progress.path), progress.fileLoaded);
    let loaded = 0;
    for (const bytes of received.values()) loaded += bytes;
    onProgress({ ...progress, loaded, total });
  };
};

/** Progress lookup key (a source without a fetch target repo / revision is spelled with those fields empty). */
const progressKey = (repo: string | undefined, revision: string | undefined, path: string) =>
  JSON.stringify([repo ?? "", revision ?? "", path]);

/**
 * The progress lookup key that the fetch surface of `loaded` emits for `ref`. The fetch target is
 * decided as hub's public contract (`AssetProgress.repo` / `revision`) says — a cross-repo reference
 * (a `repo` and `revision` pair) uses the declared target, anything else the session of that
 * manifest (`LoadedManifest.repo` / `revisionSha`).
 *
 * NOTE: between references of the same manifest, this key is equal ⇔ hub's `fileRefKey` is equal
 * (the de-duplication in {@link prefetchSeats} does not disagree with hub's duplicate check). A
 * same-repo reference and a cross-repo reference fold into the same key only when the cross-repo
 * reference points at the very repo that was loaded, and hub's `loadManifest` rejects such a
 * manifest (ADR 0038 addendum 2026-10-11).
 */
const progressKeyOf = (loaded: LoadedManifest, ref: FileRef): string =>
  ref.repo !== undefined && ref.revision !== undefined
    ? progressKey(ref.repo, ref.revision, ref.path)
    : progressKey(loaded.repo, loaded.revisionSha, ref.path);

/** The source of one component (for the replacement option, another repo's manifest). */
type Seat = {
  readonly key: string;
  readonly loaded: LoadedManifest;
  readonly container: ContainerRef;
};

/** Parts with content (a zero-length part is not fetched — hub does not fetch it either). */
const nonEmptyParts = (parts: readonly FileRef[]): readonly FileRef[] =>
  parts.filter((part) => part.size > 0);

/**
 * Warms the FileRefs chosen per seat, grouped by manifest (source). The parts of one manifest go
 * into a single `prefetchAssets` call — calling it per family or per component defeats the fetch
 * layer's concurrent fetching and drops to serial downloads.
 *
 * MUST: the fetch plan carries each file only once (the per-container part lists stay as they are;
 * only the plan is de-duplicated) — a manifest allows different containers in one selection to
 * share a part, but `prefetchAssets` rejects duplicate references as a caller error, so passing the
 * concatenation as is makes the load fail although the declaration passed. Identity is judged by
 * {@link progressKeyOf}, the same as the progress denominator ({@link aggregateProgress}) — with a
 * different measure, the files fetched and the denominator disagree.
 */
const prefetchSeats = async (
  seats: readonly Seat[],
  select: (seat: Seat) => readonly FileRef[],
  options: FetchAssetsOptions,
): Promise<void> => {
  const bySource = new Map<LoadedManifest, Map<string, FileRef>>();
  for (const seat of seats) {
    const refs = bySource.get(seat.loaded) ?? new Map<string, FileRef>();
    for (const ref of select(seat)) {
      const key = progressKeyOf(seat.loaded, ref);
      if (!refs.has(key)) refs.set(key, ref);
    }
    bySource.set(seat.loaded, refs);
  }
  for (const [loaded, refs] of bySource) {
    if (refs.size > 0) await prefetchAssets(loaded, [...refs.values()], options);
  }
};

/**
 * Fetches only each component's descriptor (part 0) and runs admission (merge + capability +
 * family); once that passes, downloads the weight parts into the persistent cache and fetches the
 * remaining assets (the manifest's `assets`) in full.
 *
 * MUST: `admit` ({@link FamilyAdmission}) is a parameter that cannot be omitted — "do not download
 * a single byte of weights for a model that cannot run" is not met by the runtime's capability
 * check alone; it holds as written only once the family's checks (pipeline name / major, the
 * `pipelineConfig` schema, the graph-vs-config check, the features of the shared GPU) all sit in
 * the early step.
 *
 * MUST: the checks of the replacement option ({@link LoadContainerOptions.components}) (the role
 * exists, the graph descriptor is identical) finish before any descriptor is fetched — reading the
 * other repo's manifest is enough to decide them.
 */
export const loadContainerComponents = async <Admitted>(
  where: string,
  loaded: LoadedManifest,
  selection: ResolvedSelection,
  componentKeys: readonly string[],
  admit: FamilyAdmission<Admitted>,
  options: LoadContainerOptions = {},
): Promise<ContainerComponents<Admitted>> => {
  const { components: replacements = {}, ...streamOptions } = options;
  for (const key of Object.keys(replacements)) {
    if (!componentKeys.includes(key)) {
      throw new Error(
        `${where}: '${key}' in components is not a component of this family` +
          ` (replaceable roles: ${componentKeys.join(" / ")})`,
      );
    }
  }

  // Seats (the source of each component) — a replacement reads another repo's manifest and picks
  // the container of **the same role**.
  const seats: Seat[] = [];
  for (const key of componentKeys) {
    // MUST: look up the role table by own keys only (the cross-cutting rule "bind symbols with
    // `Object.hasOwn`") — the check above uses `Object.keys` (own), so reading an inherited property
    // by direct indexing (`Object.create` or prototype pollution) seats a value that never passed
    // the check.
    const base = Object.hasOwn(selection.containers, key) ? selection.containers[key] : undefined;
    if (base === undefined) {
      throw new Error(
        `${where}: the container of component '${key}' is not in the manifest` +
          ` (components of selection ${selection.model} / ${selection.quant}: ${Object.keys(
            selection.containers,
          ).join(" / ")})`,
      );
    }
    const replacement = Object.hasOwn(replacements, key) ? replacements[key] : undefined;
    if (replacement === undefined) {
      seats.push({ key, loaded, container: base });
      continue;
    }
    const seatWhere = `${where} components['${key}']`;
    const { onProgress: _progress, ...manifestOptions } = streamOptions;
    const other = await loadManifest(
      toManifestSource(replacement.source, seatWhere),
      manifestOptions,
    );
    const { containers } = resolveSelection(other.manifest, {
      ...(replacement.model === undefined ? {} : { model: replacement.model }),
      ...(replacement.quant === undefined ? {} : { quant: replacement.quant }),
      // An unknown role name fails in resolveSelection with `ManifestReferenceError` (listing the available components).
      weights: [key],
    });
    const container = Object.hasOwn(containers, key) ? containers[key] : undefined;
    if (container === undefined) {
      throw new Error(
        `${seatWhere}: the referenced manifest has no container for component '${key}'`,
      );
    }
    // Identity of the graph contract — judged from the two manifests' declarations alone, before a
    // single byte of weights is fetched.
    if (container.descriptor.graph.sha256 !== base.descriptor.graph.sha256) {
      throw new Error(
        `${seatWhere}: the graph descriptor does not match the manifest's declaration (only a component whose graph descriptor sha256 is` +
          ` identical can be replaced — declared ${base.descriptor.graph.sha256} / replacement ${container.descriptor.graph.sha256})`,
      );
    }
    seats.push({ key, loaded: other, container });
  }

  const aggregated = aggregateProgress(
    [
      ...seats.map((seat) => ({ loaded: seat.loaded, refs: nonEmptyParts(seat.container.parts) })),
      { loaded, refs: Object.values(selection.assets) },
    ],
    streamOptions.onProgress,
  );
  const hubOptions: FetchAssetsOptions = {
    ...streamOptions,
    ...(aggregated === undefined ? {} : { onProgress: aggregated }),
  };
  // Reads during Session construction are range reads of cached parts, so **no progress is
  // streamed** (streaming would emit `complete` again after the load finished and double-count the
  // aggregated `loaded`).
  //
  // MUST: drop `signal` too — the signal the caller passes expresses the lifetime of "this one
  // load", and carrying it into the fetch surface turns `AbortSignal.timeout(120_000)` or an
  // `abort()` on unmount into "the load succeeded but every later generation fails". `headers` /
  // `fetch` / `caches` / `onCacheError` are "fetch tools", so holding them for the whole lifetime is
  // right; only `signal` is a different kind of value.
  const { onProgress: _loadProgress, signal: _loadSignal, ...sessionOptions } = hubOptions;

  // 1. Warm and open only the descriptors (part 0). The merge (zero shortage / excess) and the
  //    capability check fail here.
  await prefetchSeats(seats, (seat) => [seat.container.parts[0]], hubOptions);
  const components = new Map<string, ModelComponent>();
  for (const seat of seats) {
    hubOptions.signal?.throwIfAborted();
    const source = openContainerSource(seat.loaded, seat.container, sessionOptions);
    const opened = await openContainer({ kind: "source", source }, seat.container.descriptor);
    // Graph name = role name (the writer's convention — container-v1 §2.1; when it is missing,
    // prepareContainer fails listing the graphs that exist).
    components.set(seat.key, containerComponent(opened, prepareContainer(opened, seat.key)));
  }
  const open = openerOf(where, components);

  // 2. Family admission — right after merge / capability, before the weight prefetch. Pass it **the
  // same opener** that is returned below (opening differently in the early step alone would check
  // different components from the ones the later step holds).
  const admitted = await admit(open);

  // 3. Download the weight parts (for every component, once per source — removes the shape where
  // a family that builds its Sessions lazily "delays the weight download until the first run, and
  // it never shows in the load progress").
  await prefetchSeats(seats, (seat) => nonEmptyParts(seat.container.parts.slice(1)), hubOptions);

  // 4. The remaining assets (tokenizer etc.) are still fetched as whole bytes.
  const assets =
    Object.keys(selection.assets).length === 0
      ? {}
      : await fetchAssets(loaded, selection.assets, hubOptions);

  return { open, assets, admitted };
};
