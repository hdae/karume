/**
 * **The fusion hit-count check on the real distribution graphs** (no GPU needed).
 *
 * A fusion silently stops matching when the exporter changes the emission order of a single node.
 * The values stay correct (an unmatched pattern falls back to the plain op sequence) and only the
 * performance regresses, with no exception and no warning — that is why ADR 0040 §3 made
 * `lastRunFusions` permanent, and this file **gives that number something to be compared against**.
 *
 * `planFusions` is a pure function, so reading the IR from the real assets and planning is enough
 * to decide (not a single dispatch is issued). Only the container's **part 0** (header + 2 documents
 * — container-v1 §3) is read; the weight blocks are never reached (about 7GB in total).
 *
 * Shared by the per-family gates `packages/e2e/tests/e2e_<family>_fusion_counts_test.ts`: each
 * family lane owns its own counts, and this module holds only the planning and reading side they
 * have in common. It has no side effects; asset presence is decided by each test file.
 */

import { isNotFound, stat } from "@karume/test-support/fs";
import type { IrGraph } from "../../../runtime/src/format/ir.ts";
import type { AttentionDeviceFacts } from "../../../runtime/src/runtime/attention-route.ts";
import {
  FUSION_RULES,
  type FusionCounts,
  planFusions,
} from "../../../runtime/src/runtime/fusion.ts";
import { bindSymbols, countUses, planGraph } from "../../../runtime/src/runtime/plan.ts";
import {
  type ResidentWeightStorage,
  type SessionFusionKnobs,
  sessionFusionContext,
} from "../../../runtime/src/runtime/session-fusion-context.ts";
import {
  executionKnobDefaults,
  sessionOptionsViolation,
} from "../../../runtime/src/runtime/session-options-check.ts";
import type { SessionOptions } from "../../../runtime/src/runtime/session-types.ts";
import { planWeightResidency } from "../../../runtime/src/runtime/weight-residency.ts";
import { resolveParts } from "../../../runtime/tests/helpers/container-files.ts";
import {
  type ContainerManifest,
  containerPart0,
  readContainerGraph,
} from "../../../runtime/tests/helpers/container-graph.ts";
import { seriesGraph } from "../../../models/tests/helpers/series-graphs.ts";

/** Where to read a series output's container (the part 0 path and the graph name the container declares). */
export type SeriesSource = { readonly part0: URL; readonly graph: string };

/**
 * The **part 0** of a series output's container (the graph description and the binding table live
 * there — container-v1 §3) and the graph name. In an unsplit container the representative path is
 * itself part 0, so the same single line covers both forms.
 *
 * MUST: the graph name is not spelled out here but looked up in the table of `packages/models/tests/helpers/series-graphs.ts` (the same
 * single source as the guard `e2e_assets_gate_test.ts`). If each reader spelled it, rewriting only one would fail nothing.
 */
export const seriesSource = (series: string): SeriesSource => ({
  part0: resolveParts(
    new URL(`../../../../outputs/series/${series}/model.krm`, import.meta.url),
  )[0],
  graph: seriesGraph(series),
});

/**
 * Whether an asset is present.
 *
 * MUST: anything other than NotFound propagates — folding every I/O error into "not generated"
 * would turn a broken mount of the asset root (EIO and the like) into a SKIP, and a check that never
 * ran would quietly go green.
 */
export const exists = async (url: URL): Promise<boolean> => {
  try {
    await stat(url);
    return true;
  } catch (cause) {
    if (isNotFound(cause)) return false;
    throw cause;
  }
};

/**
 * Reads the graph from the **part 0** (header + 2 documents — ADR 0109 decision 3) of a
 * component's container in one distribution (the manifest directly under `root`). The weight
 * blocks are not read (fusion planning needs only the graph declaration). The part 0 of a
 * cross-repository reference is read from `mirrors` (referenced repo → local mirror — ADR 0038 §7).
 */
export const readDistributionGraph = (
  root: URL,
  manifest: ContainerManifest,
  model: string,
  component: string,
  dtype: string,
  mirrors: ReadonlyMap<string, URL>,
): Promise<IrGraph> => {
  const head = containerPart0(manifest, model, component, dtype);
  if (head.repo === undefined) {
    return readContainerGraph(new URL(head.path, root), component);
  }
  const mirror = mirrors.get(head.repo);
  if (mirror === undefined) {
    throw new Error(
      `${component}/${dtype}: no local mirror for the cross-repository target '${head.repo}'` +
        ` (known: ${[...mirrors.keys()].join(" / ")})`,
    );
  }
  return readContainerGraph(new URL(head.path, mirror), component);
};

/**
 * The Session options a check plans under: the subset of `SessionOptions` the fusion pass reads
 * (the row-block budget aside — it moves no hit count). An omitted field is the Session's default.
 * The opt-in rules and annotations fire only under their options: the linear prologue (perf-ledger
 * F-4) under `linearCompute: "a8"`, the attention's absorbed permutes (perf-ledger F-3 / F-8) on
 * the a8 attention seat, the output permute only on the online route (`attentionScoreStorage:
 * "none"` with the device facts).
 */
export type FusionCountOptions = Pick<
  SessionOptions,
  | "fuseRmsNormAdd"
  | "fuseLinearStaticQuantize"
  | "fuseGeluMulStaticQuantize"
  | "packedStaticQuantize"
  | "integerGemv"
  | "linearGemvPlanes"
  | "linearGemvReduce"
  | "linearCompute"
  | "attentionCompute"
  | "attentionScoreStorage"
  | "rmsNormReduce"
>;

/**
 * The knobs a Session constructed with `options` fixes — the same entry check and the same defaults
 * as Session construction, so a combination no Session accepts fails here instead of planning a
 * Session that cannot exist.
 */
const sessionKnobs = (options: FusionCountOptions): SessionFusionKnobs => {
  const violation = sessionOptionsViolation(options);
  if (violation !== undefined) throw new Error(`fusion-count options: ${violation}`);
  const defaults = executionKnobDefaults(options);
  return {
    fuseRmsNormAdd: options.fuseRmsNormAdd ?? false,
    fuseLinearStaticQuantize: options.fuseLinearStaticQuantize ?? false,
    fuseGeluMulStaticQuantize: options.fuseGeluMulStaticQuantize ?? false,
    packedStaticQuantize: options.packedStaticQuantize ?? false,
    integerGemv: options.integerGemv ?? false,
    linearGemvPlanes: options.linearGemvPlanes ?? false,
    linearGemvReduce: defaults.linearGemvReduce,
    linearCompute: defaults.linearCompute,
    attentionCompute: defaults.attentionCompute,
    attentionScoreStorage: defaults.attentionScoreStorage,
    rmsNormReduce: defaults.rmsNormReduce,
    rowBlockSplit: undefined,
    attentionRowBlockMaxBytes: undefined,
  };
};

/**
 * The storage each initializer is resident in, from the residency plan — the group size comes with
 * the preset (the planner derives it from the merged declaration, so it is not derived twice). Raw
 * and expanded weights are f32, and a borrowed weight's storage is the lender's (unknown here): all
 * absent.
 */
const residentWeights = (graph: IrGraph): ReadonlyMap<string, ResidentWeightStorage> =>
  new Map(
    [...planWeightResidency(graph)].flatMap(([name, weight]): [string, ResidentWeightStorage][] => {
      switch (weight.seat) {
        case "i4":
        case "i4f16":
          return [[name, { storage: weight.seat, groupSize: weight.groupSize }]];
        case "f16":
        case "bf16":
        case "i8":
        case "i2":
          return [[name, { storage: weight.seat }]];
        default:
          return [];
      }
    }),
  );

/**
 * The fusion hit counts of a graph planned as a Session with `options` would plan it (the context
 * from the executor's own `sessionFusionContext`). `attentionDevice` is the device facts of the
 * online attention route, needed only on the online seat.
 */
export const fusionCounts = (
  graph: IrGraph,
  inputShapes: Readonly<Record<string, readonly number[]>>,
  options: FusionCountOptions = {},
  attentionDevice?: AttentionDeviceFacts,
  stateShapes?: ReadonlyMap<string, readonly number[]>,
): FusionCounts =>
  planFusions(
    planGraph(graph, bindSymbols(graph, inputShapes), stateShapes).nodes,
    sessionFusionContext({
      rules: FUSION_RULES,
      knobs: sessionKnobs(options),
      useCounts: countUses(graph),
      outputNames: new Set(graph.outputs),
      // Decide against the WebGPU core default (128MiB). The row-block count does not affect the
      // hit counts, but **not taking the limit from the machine's measured value** keeps this check
      // fixed and independent of the machine.
      limits: {
        maxStorageBufferBindingSize: 128 * 1024 * 1024,
        maxComputeWorkgroupsPerDimension: 65535,
      },
      ...(attentionDevice === undefined ? {} : { attentionDevice }),
      residentWeights: residentWeights(graph),
    }),
  ).counts;

/**
 * Fusion hit counts of a decode graph with states (planned for a given physical chunk row count M).
 *
 * The slot capacity symbol is resolved to 640 — fusion is decided by the node sequence and the
 * shapes of the **values**, not by the slot capacity (using the same 640 as the e2e acceptance value
 * only makes the two easier to read side by side). Every input is `[1, M]`.
 */
export const decodeFusionCounts = (
  graph: IrGraph,
  rows: number,
  options: FusionCountOptions = {},
): FusionCounts => {
  const inputShapes = Object.fromEntries(
    graph.inputs.map((spec) => [
      spec.name,
      spec.shape.map((dim) => (typeof dim === "number" ? dim : rows)),
    ]),
  );
  const stateShapes = new Map(
    Object.entries(graph.states).map(([name, slot]) => [
      name,
      slot.shape.map((dim) => (typeof dim === "number" ? dim : 640)),
    ]),
  );
  return fusionCounts(graph, inputShapes, options, undefined, stateShapes);
};

export const NONE: FusionCounts = {
  silu: 0,
  upsample2x: 0,
  rope: 0,
  rmsNormRope: 0,
  adaln: 0,
  gatedResidual: 0,
  rmsNormAdd: 0,
  linearStaticQuantize: 0,
  geluMulStaticQuantize: 0,
  rowBlockAttention: 0,
  identityExpand: 0,
  packedStaticQuantize: 0,
  linearPrologue: 0,
  attentionPermute: 0,
};
