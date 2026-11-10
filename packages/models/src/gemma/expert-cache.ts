/**
 * Gemma's side of the exact expert cache (`PipelineOptions.expertCache` — ADR 0146): the checks of
 * the option and the prefill chunk rows it allows. The re-run protocol itself is the generation
 * layer's (`../generation/expert-cache-session.ts`).
 *
 * Every refusal here is decided by the caller's options and the admitted graph, so each is an input
 * error (ADR 0107 decision 2), raised before a weight part is fetched.
 *
 * NOTE: `export` of the checks is for the test that calls them directly (not exposed on `mod.ts` /
 * the subpath surface — ADR 0008).
 *
 * DECIDED: docs/decisions/0146-exact-expert-cache.md
 */

import { ModelInputError } from "../errors.ts";

/** The options the checks read (`PipelineOptions` satisfies it). */
type ExpertCacheOptions = {
  readonly expertCache?: {
    readonly slots: number | readonly number[];
    readonly hostBytes?: number;
    readonly depth?: number;
  };
  readonly decodeAhead?: number;
  readonly speculative?: unknown;
  readonly pleResidency?: string;
  readonly vision?: boolean;
  readonly chunkLength?: number;
};

type Dims = readonly (number | string)[];

/** The parts of a graph the checks read (an `IrGraph` satisfies it). */
type RoutedGraph = {
  readonly nodes: readonly {
    readonly op: string;
    readonly ins: readonly string[];
    readonly outs: readonly string[];
    readonly attrs: Readonly<Record<string, unknown>>;
    readonly states: Readonly<Record<string, string>>;
  }[];
  readonly values: Readonly<Record<string, { readonly shape: Dims }>>;
  readonly states: Readonly<Record<string, { readonly shape: Dims }>>;
};

/**
 * What the cache needs to know of the routed layers (the runtime's contract: one k, one E) and how
 * many there are (an array of slots has one entry per routed layer).
 */
type Routing = { readonly picks: number; readonly experts: number; readonly layers: number };

/**
 * The settled option: the slots (one count, or one per routed layer in layer order — as given), the
 * host tier's budget (absent = no tier), self-speculation's depth (absent = scheme A alone) and the
 * prefill chunk rows the pipeline runs with.
 */
export type Gemma4ExpertCache = {
  readonly slots: number | readonly number[];
  readonly hostBytes?: number;
  readonly depth?: number;
  readonly chunkLength: number;
};

/** A positive safe integer (a count of slots). */
const isCount = (value: unknown): value is number =>
  typeof value === "number" && Number.isSafeInteger(value) && value >= 1;

/**
 * The checks the options alone decide (run **before reading a single byte of the assets**), returning
 * the slots (`undefined` without the option): the number, or a frozen copy of the per-layer array
 * (its length against the routed layers is {@link resolveGemma4ExpertCache}'s, which needs the
 * graph). `hostBytes` is checked for its form here (a positive safe integer); that it holds at least
 * one expert is the runtime's check at Session construction (an expert's bytes are the container's,
 * not the graph's).
 *
 * MUST: refuse every option that would run the target through a batch or an immediate run, both of
 * which a run under the cache cannot be — the run-ahead (`decodeAhead ≥ 2`) and the GPU-resident PLE
 * enqueue the target, a drafter's verify needs the cache's self-speculation (later work), and an
 * image's bidirectional rows need an immediate run (ADR 0146 decision 1).
 */
export const assertGemma4ExpertCacheOptions = (
  where: string,
  options: ExpertCacheOptions,
): number | readonly number[] | undefined => {
  const cache: unknown = options.expertCache;
  if (cache === undefined) return undefined;
  const given: unknown =
    typeof cache === "object" && cache !== null && "slots" in cache ? cache.slots : undefined;
  let slots: number | readonly number[];
  if (Array.isArray(given)) {
    // Copied entry by entry as checked (by index, so a hole is read as undefined), so the settled
    // value is the one checked.
    const counts: number[] = [];
    for (let layer = 0; layer < given.length; layer += 1) {
      const count: unknown = given[layer];
      if (!isCount(count)) {
        throw new ModelInputError(
          `${where}: expertCache.slots[${layer}] must be a positive integer, one per routed layer in layer order (slots[${layer}] ${typeof count === "number" ? String(count) : (JSON.stringify(count) ?? String(count))})`,
        );
      }
      counts.push(count);
    }
    slots = Object.freeze(counts);
  } else if (isCount(given)) {
    slots = given;
  } else {
    throw new ModelInputError(
      `${where}: expertCache must be { slots: <a positive integer> } or { slots: [<a positive integer per routed layer>] } (slots ${String(given)})`,
    );
  }
  const hostBytes: unknown =
    typeof cache === "object" && cache !== null && "hostBytes" in cache
      ? cache.hostBytes
      : undefined;
  if (
    hostBytes !== undefined &&
    (typeof hostBytes !== "number" || !Number.isSafeInteger(hostBytes) || hostBytes < 1)
  ) {
    throw new ModelInputError(
      `${where}: expertCache.hostBytes must be a positive integer of bytes held in RAM (Number.MAX_SAFE_INTEGER holds every expert), or absent for no host tier (hostBytes ${typeof hostBytes === "number" ? String(hostBytes) : JSON.stringify(hostBytes)})`,
    );
  }
  const depth: unknown =
    typeof cache === "object" && cache !== null && "depth" in cache ? cache.depth : undefined;
  if (
    depth !== undefined &&
    (typeof depth !== "number" || !Number.isSafeInteger(depth) || depth < 1)
  ) {
    throw new ModelInputError(
      `${where}: expertCache.depth must be a positive integer of drafts per self-speculation cycle, or absent for none (depth ${typeof depth === "number" ? String(depth) : JSON.stringify(depth)})`,
    );
  }
  const refuse = (other: string, why: string): never => {
    throw new ModelInputError(
      `${where}: expertCache cannot be combined with ${other} (${why}; a run under the cache runs alone, deferred, until it misses no expert — ADR 0146)`,
    );
  };
  if ((options.decodeAhead ?? 1) >= 2) {
    refuse(`decodeAhead ${options.decodeAhead}`, "the run-ahead chains its steps in one GPU batch");
  }
  if (options.speculative !== undefined) {
    refuse(
      "speculative",
      "a drafter's verify does not run over the cache — its own self-speculation is expertCache.depth",
    );
  }
  if (options.pleResidency === "gpu") {
    refuse('pleResidency: "gpu"', "the on-GPU gather enqueues the target's decode in its batch");
  }
  if (options.vision === true) {
    refuse("vision", "an image's bidirectional rows need an immediate run");
  }
  return slots;
};

/**
 * The routed layers' top-k, expert count and number — one routed layer per distinct index the
 * `expert_linear` nodes read, as the runtime counts them (`undefined` = the graph has no
 * `expert_linear`).
 */
const routingOf = (where: string, graph: RoutedGraph): Routing | undefined => {
  const routers = new Map(
    graph.nodes.flatMap((node) =>
      node.op === "topk" && node.outs[1] !== undefined ? [[node.outs[1], node] as const] : [],
    ),
  );
  const picks = new Set<unknown>();
  const experts = new Set<unknown>();
  const indices = new Set<string>();
  for (const node of graph.nodes) {
    if (node.op !== "expert_linear") continue;
    indices.add(node.ins[1]);
    picks.add(routers.get(node.ins[1])?.attrs["k"]);
    const weight = node.ins[2];
    experts.add(Object.hasOwn(graph.values, weight) ? graph.values[weight].shape[0] : undefined);
  }
  if (picks.size === 0) return undefined;
  const [k] = picks;
  const [e] = experts;
  if (
    picks.size !== 1 ||
    experts.size !== 1 ||
    typeof k !== "number" ||
    typeof e !== "number" ||
    !Number.isSafeInteger(k) ||
    !Number.isSafeInteger(e) ||
    k < 1 ||
    e < k
  ) {
    throw new Error(
      `${where}: the expert_linear layers do not share one topk k and one expert count E` +
        ` (k ${[...picks].join(" / ")}, E ${[...experts].join(" / ")}) — the cache needs both uniform`,
    );
  }
  return { picks: k, experts: e, layers: indices.size };
};

/**
 * The sliding ring's slack (`undefined` without sliding slots): the smallest row capacity (axis 2 of
 * a slot) minus its window, the rows a deferred run may write ahead (the runtime's
 * `GenerationContext.slidingSlack`, read here from the graph before any context exists).
 */
const slidingSlackOf = (where: string, graph: RoutedGraph): number | undefined => {
  let slack: number | undefined;
  for (const node of graph.nodes) {
    if (!Object.hasOwn(node.attrs, "window")) continue;
    const window = node.attrs["window"];
    for (const slot of Object.values(node.states)) {
      const rows = Object.hasOwn(graph.states, slot) ? graph.states[slot].shape[2] : undefined;
      if (typeof window !== "number" || typeof rows !== "number") {
        throw new Error(
          `${where}: sliding slot '${slot}' (window ${String(window)}, rows ${String(rows)}) has no numeric window and row capacity — the slack the cache's chunks need cannot be read`,
        );
      }
      slack = Math.min(slack ?? rows - window, rows - window);
    }
  }
  return slack;
};

/**
 * Settles {@link Gemma4ExpertCache} from the options and the admitted graph (`undefined` without the
 * option) — called once admission has the graph, before any weight part is fetched.
 *
 * A prefill chunk under the cache has at most `min(slidingSlack, ⌊slots / k⌋, the distribution's
 * chunkLength)` rows: a chunk is a deferred run (at most the slack), and all of one chunk's picks
 * must fit in the slots at once so its re-run converges (ADR 0146 decision 4). That bound is the
 * default chunk length, and a larger `chunkLength` fails loudly. The fewest slots are 2 · k (a
 * prefill chunk has at least 2 rows), and at most E.
 *
 * With one count per routed layer, the array has exactly one entry per routed layer, each entry is
 * held to those bounds (a failure names it), and `slots` above is the **smallest** layer's — every
 * layer must hold one chunk's (and one verify's) picks at once; the messages name that layer.
 */
export const resolveGemma4ExpertCache = (
  where: string,
  graph: RoutedGraph,
  options: ExpertCacheOptions,
  distributionChunkLength: number,
): Gemma4ExpertCache | undefined => {
  const slots = assertGemma4ExpertCacheOptions(where, options);
  if (slots === undefined) return undefined;
  const routing = routingOf(where, graph);
  if (routing === undefined) {
    throw new ModelInputError(
      `${where}: expertCache needs a mixture-of-experts model (a graph with expert_linear, such as Gemma 4 26B A4B); this graph has none`,
    );
  }
  const minimum = 2 * routing.picks;
  // One count per routed layer, in layer order (a number is every layer's).
  const perLayer = typeof slots === "number" ? undefined : slots;
  if (perLayer !== undefined && perLayer.length !== routing.layers) {
    throw new ModelInputError(
      `${where}: expertCache.slots has ${perLayer.length} entries, but the model has ${routing.layers} routed layers (one entry per routed layer, in layer order)`,
    );
  }
  const counts = typeof slots === "number" ? [slots] : slots;
  for (const [layer, count] of counts.entries()) {
    const named = perLayer === undefined ? "expertCache.slots" : `expertCache.slots[${layer}]`;
    if (count < minimum) {
      throw new ModelInputError(
        `${where}: ${named} ${count} is below the minimum ${minimum}` +
          ` (2 × the routing's top-k ${routing.picks}: a prefill chunk has at least 2 rows, and all of one chunk's picks must fit in the slots)`,
      );
    }
    if (count > routing.experts) {
      throw new ModelInputError(
        `${where}: ${named} ${count} exceeds the ${routing.experts} experts of a layer`,
      );
    }
  }
  // MUST: the rows of a chunk and of a verify are bounded by the smallest layer — every layer holds
  // one run's distinct picks at once, so the layer with the fewest slots decides (the first of them
  // is named).
  const fewest = Math.min(...counts);
  const smallest = counts.indexOf(fewest);
  const slack = slidingSlackOf(where, graph);
  const perSlots = Math.floor(fewest / routing.picks);
  const bound = Math.min(slack ?? perSlots, perSlots, distributionChunkLength);
  // How the messages name the slots: the number as given, or the array by its smallest layer.
  const under =
    perLayer === undefined
      ? `expertCache { slots: ${slots} }`
      : `expertCache's per-layer slots (the smallest layer, slots[${smallest}], has ${fewest})`;
  const ratio =
    perLayer === undefined
      ? `⌊${fewest} / ${routing.picks}⌋ = ${perSlots}`
      : `⌊${fewest} / ${routing.picks}⌋ = ${perSlots} of the smallest layer, slots[${smallest}],`;
  const chunkLength = options.chunkLength ?? bound;
  if (chunkLength > bound) {
    throw new ModelInputError(
      `${where}: chunkLength ${chunkLength} exceeds ${bound}, the most rows one prefill chunk can have under ${under}` +
        ` (the smallest of the sliding slack ${slack ?? "(none)"}, ${ratio} and the distribution's chunkLength ${distributionChunkLength})`,
    );
  }
  const hostBytes = options.expertCache?.hostBytes;
  const depth = options.expertCache?.depth;
  if (depth !== undefined) {
    // The verify of `depth + 1` rows is one deferred run (the slack and the chunk rows bound it),
    // and its picks are loaded at once, pinned (`depth + 1` rows of k picks fit every layer's slots).
    const depthBound = Math.min(perSlots, slack ?? perSlots, chunkLength) - 1;
    if (depth > depthBound) {
      throw new ModelInputError(
        `${where}: expertCache.depth ${depth} exceeds ${depthBound}, the most drafts a verify of depth + 1 rows allows under ${under}` +
          ` (depth + 1 ≤ ${ratio} so that its picks fit the slots, ≤ the sliding slack ${slack ?? "(none)"} and ≤ chunkLength ${chunkLength})`,
      );
    }
  }
  return {
    slots,
    ...(hostBytes === undefined ? {} : { hostBytes }),
    ...(depth === undefined ? {} : { depth }),
    chunkLength,
  };
};
