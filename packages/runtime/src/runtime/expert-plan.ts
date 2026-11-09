/**
 * The plans of the exact expert cache (ADR 0146) that touch no GPU: which `topk` routes which
 * `expert_linear` weights (the layers), the pool sizes the residency plan counts, where every expert
 * lies in the container, the victims of a load, and the distinct picks of a routing.
 *
 * Session construction (session-build.ts), the estimate (estimate.ts) and the Session's cache
 * (expert-cache.ts) read the same plan from here, so the pool the estimate counts and the pool the
 * Session allocates cannot drift apart. MUST: bring in no import that touches the GPU (the estimate's
 * dependency would pull it in — the same rule as session-options-check.ts).
 */

import type { BlockRangeReader, InitializerSupply, SupplyBlock } from "../format/container/bind.ts";
import { scaleStepBytes } from "../format/container/codecs.ts";
import type { IrGraph } from "../format/ir.ts";
import { EXPERT_LINEAR_OP, TOPK_OP, topkK, WEIGHT_SLOTS } from "../ops.ts";
import { ExecutionError } from "./plan.ts";
import { expertCacheViolation } from "./session-options-check.ts";
import type { ExpertRouting } from "./session-types.ts";
import type { WeightResidency } from "./weight-residency.ts";

/** One `expert_linear` weight of a routed layer, with the bytes of one of its experts. */
export type ExpertWeightPlan = {
  readonly name: string;
  /** Bytes of one expert's codes (its f32 values for f32) = one slot of the codes pool. */
  readonly codesBytes: number;
  /** Bytes of one expert's group steps (0 for f32) = one slot of the steps pool. */
  readonly stepsBytes: number;
} & (
  | { readonly storage: "f32"; readonly groupSize?: undefined }
  /** The group codecs (ADR 0145 decision 3), with their group length. */
  | { readonly storage: "i4" | "i4f16"; readonly groupSize: number }
);

/** One routed layer: a `topk` whose index output the layer's `expert_linear` nodes read. */
export type ExpertLayerPlan = {
  /** The `topk`'s index value (output 1) — the key of the layer's slot table. */
  readonly index: string;
  /** The layer's expert weights, in graph order of the first `expert_linear` reading each. */
  readonly weights: readonly ExpertWeightPlan[];
};

/** The graph-side plan of an expert cache (shared by Session construction and the estimate). */
export type ExpertGraphPlan = {
  readonly slots: number;
  readonly experts: number;
  readonly picks: number;
  /** In graph order of the layers' `topk` nodes — the order of the routing's layer axis. */
  readonly layers: readonly ExpertLayerPlan[];
  /**
   * The graph with every layer's index appended to its outputs (ADR 0146 decision 3: the routing is
   * read back in the run's own command list and fence, kept alive by the same liveness rule as any
   * output).
   */
  readonly graph: IrGraph;
  /** The graph's declared outputs (what a run returns to its caller). */
  readonly declaredOutputs: readonly string[];
  /** The residency plan with every expert weight sized as its pool of `slots` experts. */
  readonly residency: ReadonlyMap<string, WeightResidency>;
  /** Bytes of one slot table (one i32 per expert). */
  readonly tableBytes: number;
};

/**
 * One expert weight's plan and its residency as a pool of `slots` experts (`[slots, out, in]` codes
 * and `[slots, out · in / G]` steps).
 *
 * MUST: one expert's codes and steps are whole u32 words — a load writes a slot with one
 * `writeBuffer`, which takes whole words only.
 */
const pooledWeight = (
  name: string,
  seat: WeightResidency | undefined,
  shape: readonly number[],
  slots: number,
): { readonly plan: ExpertWeightPlan; readonly pool: WeightResidency } => {
  const where = `options.expertCache: expert weight '${name}'`;
  const [experts, rows, columns] = shape;
  const words = (codesBytes: number, stepsBytes: number): void => {
    if (
      !Number.isSafeInteger(codesBytes) ||
      !Number.isSafeInteger(stepsBytes) ||
      codesBytes % 4 !== 0 ||
      stepsBytes % 4 !== 0
    ) {
      throw new ExecutionError(
        `${where}: one expert is ${codesBytes} bytes of codes and ${stepsBytes} bytes of steps — a slot needs whole u32 words`,
      );
    }
  };
  if (seat?.seat === "raw") {
    const codesBytes = seat.payloadBytes / experts;
    words(codesBytes, 0);
    return {
      plan: { name, storage: "f32", codesBytes, stepsBytes: 0 },
      pool: { seat: "raw", payloadBytes: slots * codesBytes },
    };
  }
  if (seat?.seat === "i4" || seat?.seat === "i4f16") {
    const codesBytes = seat.payloadBytes / experts;
    const stepsBytes =
      ((rows * columns) / seat.groupSize) *
      scaleStepBytes(seat.seat === "i4" ? "int4-sym-g" : "int4-sym-g-f16");
    words(codesBytes, stepsBytes);
    return {
      plan: { name, storage: seat.seat, groupSize: seat.groupSize, codesBytes, stepsBytes },
      pool: {
        ...seat,
        payloadBytes: slots * codesBytes,
        scaleBytes: slots * stepsBytes,
      },
    };
  }
  throw new ExecutionError(
    `${where}: planned as '${seat?.seat}' (f32 / i4 / i4f16 only — ADR 0145 decision 2)`,
  );
};

/**
 * Plans the cache over a graph (ADR 0146 decisions 1 and 2) and fails loudly on every form it does
 * not hold exactly:
 *
 * - `slots` is a positive integer from the routed `topk` k to E (a row's picks always fit, and a
 *   pool never holds more than E experts);
 * - the graph has `expert_linear`, every layer has the same k and E (the routing is one
 *   `[layers, rows, picks]` array);
 * - an expert weight is an initializer the Session owns (not borrowed, not a graph output), read
 *   only at `expert_linear`'s weight slot and under one routing — a pool holds slots, not experts,
 *   so any other reader, or a second slot table, would read another expert's bytes;
 * - one expert's codes and steps are whole u32 words (a load writes a slot with `writeBuffer`).
 *
 * The planner's producer rule of the index (ADR 0145 decision 2) is unchanged: the declarations stay
 * `[E, out, in]` and only the allocated buffers are pools.
 */
export const planExpertGraph = (
  graph: IrGraph,
  residency: ReadonlyMap<string, WeightResidency>,
  cache: { readonly slots: number },
): ExpertGraphPlan => {
  const violation = expertCacheViolation(cache);
  if (violation !== undefined) throw new ExecutionError(violation);
  const where = "options.expertCache";
  const { slots } = cache;
  const weightSlot = WEIGHT_SLOTS.get(EXPERT_LINEAR_OP);
  if (weightSlot === undefined) {
    throw new ExecutionError(`${EXPERT_LINEAR_OP} has no weight slot (broken op table)`);
  }
  // index value → its weights in graph order, and weight → the one index it is read under.
  const routed = new Map<string, string[]>();
  const routing = new Map<string, string>();
  for (const node of graph.nodes) {
    if (node.op !== EXPERT_LINEAR_OP) continue;
    const index = node.ins[1];
    const weight = node.ins[weightSlot];
    const initializer = Object.hasOwn(graph.initializers, weight)
      ? graph.initializers[weight]
      : undefined;
    if (initializer === undefined || initializer.shared === true) {
      throw new ExecutionError(
        `${where}: expert_linear weight '${weight}' is ${
          initializer === undefined ? "not an initializer" : "borrowed (shared)"
        } — the cache pools the Session's own initializers only`,
      );
    }
    const previous = routing.get(weight);
    if (previous !== undefined && previous !== index) {
      throw new ExecutionError(
        `${where}: weight '${weight}' is read under two routings ('${previous}' and '${index}') — one pool follows one slot table`,
      );
    }
    routing.set(weight, index);
    const weights = routed.get(index) ?? [];
    if (!weights.includes(weight)) weights.push(weight);
    routed.set(index, weights);
  }
  if (routed.size === 0) {
    throw new ExecutionError(`${where} needs a graph with expert_linear (this graph has none)`);
  }
  graph.nodes.forEach((node, at) => {
    node.ins.forEach((name, slot) => {
      if (routing.has(name) && (node.op !== EXPERT_LINEAR_OP || slot !== weightSlot)) {
        throw new ExecutionError(
          `${where}: expert weight '${name}' is also input ${slot} of nodes[${at}] (${node.op}) — a pool is read at expert_linear's weight slot only`,
        );
      }
    });
  });
  for (const name of graph.outputs) {
    if (routing.has(name)) {
      throw new ExecutionError(`${where}: expert weight '${name}' is a graph output`);
    }
  }

  const layers: ExpertLayerPlan[] = [];
  const pooled = new Map(residency);
  let experts: number | undefined;
  let picks: number | undefined;
  for (const [at, node] of graph.nodes.entries()) {
    const index = node.outs[1];
    const names = node.op === TOPK_OP && index !== undefined ? routed.get(index) : undefined;
    if (index === undefined || names === undefined) continue;
    const k = topkK(node.attrs, `nodes[${at}] (topk)`);
    if (picks !== undefined && k !== picks) {
      throw new ExecutionError(
        `${where}: the routed topk nodes pick ${picks} and ${k} experts — every layer needs the same k`,
      );
    }
    picks = k;
    const weights = names.map((name): ExpertWeightPlan => {
      const shape = graph.values[name].shape.map(Number);
      const count = shape[0];
      if (experts !== undefined && count !== experts) {
        throw new ExecutionError(
          `${where}: expert weights hold ${experts} and ${count} experts — every layer needs the same E`,
        );
      }
      experts = count;
      const { plan, pool } = pooledWeight(name, residency.get(name), shape, slots);
      pooled.set(name, pool);
      return plan;
    });
    layers.push({ index, weights });
  }
  // The index of every expert_linear is a topk output (the producer rule of plan.ts); a direct build
  // that skipped the contract checks lands here instead of planning a table nobody writes.
  if (experts === undefined || picks === undefined || layers.length !== routed.size) {
    throw new ExecutionError(
      `${where}: an expert_linear index is not the index output of a topk (ADR 0145 decision 2)`,
    );
  }
  if (slots < picks || slots > experts) {
    throw new ExecutionError(
      `${where}.slots ${slots} must be from the topk k ${picks} to the experts E ${experts}`,
    );
  }
  const declaredOutputs = graph.outputs;
  const indices = layers
    .map((layer) => layer.index)
    .filter((name) => !declaredOutputs.includes(name));
  return {
    slots,
    experts,
    picks,
    layers,
    graph: { ...graph, outputs: [...declaredOutputs, ...indices] },
    declaredOutputs,
    residency: pooled,
    tableBytes: experts * 4,
  };
};

/** One contiguous range of a declared block: one expert's codes or steps. */
export type ExpertByteRange = {
  readonly block: string;
  readonly offset: number;
  readonly length: number;
};

/** Where each expert of one pooled weight lies in the container (index = expert id). */
export type ExpertWeightRanges = {
  readonly codes: readonly ExpertByteRange[];
  /** Absent for f32. */
  readonly steps: readonly ExpertByteRange[] | undefined;
};

/**
 * Derives, once, where every expert of one pooled weight lies (ADR 0146 decision 5): the codes from
 * the data's blocks (one, or a piece list over the leading — expert — dimension, rule ④), the steps
 * from the scale's block or its pieces (rule ③). Expert `e` of a block over rows `[r0, r1)` starts at
 * `(e − r0) · bytes`; MUST: each block's own payload equals its rows × the bytes of one expert from
 * the declaration — no piece size is assumed, and a mismatch fails loudly.
 */
export const expertWeightRanges = (
  weight: ExpertWeightPlan,
  supply: InitializerSupply,
  experts: number,
): ExpertWeightRanges => {
  const where = `options.expertCache: weight '${weight.name}'`;
  const rangesOf = (
    blocks: readonly SupplyBlock[],
    bytes: number,
    what: string,
  ): ExpertByteRange[] => {
    const ranges = new Map<number, ExpertByteRange>();
    for (const block of blocks) {
      const [first, end] = block.rows;
      if (block.payloadBytes !== (end - first) * bytes) {
        throw new ExecutionError(
          `${where}: ${what} block '${block.id}' carries ${block.payloadBytes} bytes for rows [${first}, ${end}), not ${bytes} per expert`,
        );
      }
      for (let expert = first; expert < end; expert += 1) {
        ranges.set(expert, { block: block.id, offset: (expert - first) * bytes, length: bytes });
      }
    }
    return Array.from({ length: experts }, (_, expert) => {
      const range = ranges.get(expert);
      if (range === undefined) {
        throw new ExecutionError(`${where}: no ${what} block covers expert ${expert}`);
      }
      return range;
    });
  };
  const codes = rangesOf(supply.blocks, weight.codesBytes, "codes");
  if (weight.storage === "f32") return { codes, steps: undefined };
  const scale = supply.scale;
  if (scale === undefined) throw new ExecutionError(`${where}: ${weight.storage} but no scale`);
  return { codes, steps: rangesOf(scale.pieces ?? [scale], weight.stepsBytes, "steps") };
};

/** The container side of a Session's expert cache (passed by `PreparedModel`). */
export type ExpertSource = {
  readonly supplies: ReadonlyMap<string, InitializerSupply>;
  readonly ranges: BlockRangeReader | undefined;
};

/** Where every expert of every layer is read from (per layer, per weight of {@link ExpertGraphPlan}). */
export type ExpertStorePlan = {
  readonly ranges: readonly (readonly ExpertWeightRanges[])[];
  readonly reader: BlockRangeReader;
};

/**
 * The host store of the cache (ADR 0146 decision 5): the ranges of every expert and the container's
 * range reader. MUST: a container that cannot read ranges, or whose range reads of an expert block
 * re-read the whole block or part (`"scan"`), fails loudly — every miss would read a whole block for
 * one expert. A source that does not state its cost class passes (the runtime's `BlockSource` has
 * none).
 */
export const planExpertStore = (
  plan: ExpertGraphPlan,
  source: ExpertSource | undefined,
): ExpertStorePlan => {
  const reader = source?.ranges;
  if (source === undefined || reader === undefined) {
    throw new ExecutionError(
      "options.expertCache: the container cannot read byte ranges of its blocks (BoundContainer.ranges) — a miss could not load one expert",
    );
  }
  const ranges = plan.layers.map((layer) =>
    layer.weights.map((weight) => {
      const supply = source.supplies.get(weight.name);
      if (supply === undefined) {
        throw new ExecutionError(`options.expertCache: weight '${weight.name}' has no supply`);
      }
      const derived = expertWeightRanges(weight, supply, plan.experts);
      for (const range of [...derived.codes, ...(derived.steps ?? [])]) {
        if (reader.cost(range.block) === "scan") {
          throw new ExecutionError(
            `options.expertCache: a range read of block '${range.block}' (weight '${weight.name}') re-reads the whole block — each miss would read it again for one expert`,
          );
        }
      }
      return derived;
    }),
  );
  return { ranges, reader };
};

/** One slot's move in a load: the expert it receives, or the expert that leaves it. */
export type ExpertSlotMove = { readonly expert: number; readonly slot: number };

/** The result of {@link planExpertLoad}. */
export type ExpertLoadPlan = {
  /** The missing experts and their slots, in the order of `needed`. */
  readonly load: readonly ExpertSlotMove[];
  /** The resident experts that give their slot up, least recently used first. */
  readonly evict: readonly ExpertSlotMove[];
};

/**
 * The slot choice of one layer's load (ADR 0146 decision 6): the missing experts of `needed` take the
 * free slots first (ascending), then the slots of the least recently used resident experts that
 * `needed` does not hold (they are pinned for this load). A pure function of its arguments.
 *
 * @param table expert → slot, −1 when not resident (the host mirror of the layer's table)
 * @param recency the resident experts, least recently used first (every resident expert, once)
 * @param needed the distinct experts the routing needs; more than `slots` fails loudly
 */
export const planExpertLoad = (
  slots: number,
  table: ArrayLike<number>,
  recency: Iterable<number>,
  needed: readonly number[],
): ExpertLoadPlan => {
  if (needed.length > slots) {
    throw new ExecutionError(
      `expert load: one layer needs ${needed.length} distinct experts, more than its ${slots} slots (load the routing of fewer rows)`,
    );
  }
  const used = new Set<number>();
  for (let expert = 0; expert < table.length; expert += 1) {
    if (table[expert] >= 0) used.add(table[expert]);
  }
  const free = Array.from({ length: slots }, (_, slot) => slot).filter((slot) => !used.has(slot));
  const pinned = new Set(needed);
  const victims = [...recency].filter((expert) => !pinned.has(expert));
  const load: ExpertSlotMove[] = [];
  const evict: ExpertSlotMove[] = [];
  for (const expert of needed) {
    if (table[expert] >= 0) continue;
    const slot = free.shift();
    if (slot !== undefined) {
      load.push({ expert, slot });
      continue;
    }
    // `needed` ≤ slots, so the resident experts not needed are at least the missing ones left.
    const victim = victims.shift();
    const taken = victim === undefined ? -1 : table[victim];
    if (victim === undefined || taken < 0) {
      throw new ExecutionError(
        "expert load: no slot left for a missing expert (the recency list and the table disagree — broken bookkeeping)",
      );
    }
    evict.push({ expert: victim, slot: taken });
    load.push({ expert, slot: taken });
  }
  return { load, evict };
};

/** The distinct experts of layer `layer` in the first `rows` rows, in row then pick order. */
export const routedExperts = (routing: ExpertRouting, layer: number, rows: number): number[] => {
  const { picks } = routing;
  const base = layer * routing.rows * picks;
  return [...new Set(routing.ids.subarray(base, base + rows * picks))];
};

/**
 * Checks a routing handed back by the caller against the cache's shape: its sizes, an `Int32Array`
 * of exactly `layers × rows × picks` ids, every id an expert in `[0, experts)`.
 *
 * MUST: an id out of range fails here — written to a table, it would address another expert.
 */
export const assertRouting = (
  routing: ExpertRouting,
  shape: { readonly layers: number; readonly picks: number; readonly experts: number },
  where: string,
): void => {
  const { layers, rows, picks, ids } = routing;
  if (
    layers !== shape.layers ||
    picks !== shape.picks ||
    !Number.isSafeInteger(rows) ||
    rows < 0 ||
    !(ids instanceof Int32Array) ||
    ids.length !== layers * rows * picks
  ) {
    throw new ExecutionError(
      `${where}: the routing is not [${shape.layers}, rows, ${shape.picks}] i32 ids of this cache`,
    );
  }
  for (const id of ids) {
    if (id < 0 || id >= shape.experts) {
      throw new ExecutionError(
        `${where}: routing id ${id} is not an expert in [0, ${shape.experts})`,
      );
    }
  }
};
