/**
 * The table the attention bench page shows: one row per (shape, route), plus one for the shared
 * quantization when it was timed. Pure functions over the runtime's bench record, so `vp test`
 * covers them without a GPU.
 */

import type {
  AttentionBenchRoute,
  AttentionBenchShape,
  AttentionBenchShapeResult,
} from "@karume/runtime/tune";

export type BenchTableRow = {
  readonly key: string;
  readonly shape: string;
  readonly route: "s16" | "online" | "quantization";
  /** Query rows one measured dispatch covered. */
  readonly rows?: number;
  /** How production splits the node for this route ("3 blocks × 2,640 rows"). */
  readonly split?: string;
  readonly msPerDispatch?: number;
  readonly msPer1000Rows?: number;
  readonly tops?: number;
  /** The route's (or the quantization's) time for the whole node. */
  readonly nodeMs?: number;
  /** Per-dispatch ms of each stage ("qk 12.30 · stats 1.20 · pv 10.10"). */
  readonly stages?: string;
  /** The variant that ran (the online reduce form and its workgroup storage, the s16 row window). */
  readonly variant?: string;
  /** online / s16 per 1,000 rows (on the online row). */
  readonly ratio?: number;
  /** Why the route did not run, or the first failure. */
  readonly note?: string;
};

const integer = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });

/** `27280` → `27,280`. */
export const formatCount = (value: number): string => integer.format(value);

/** Milliseconds: 2 decimals from 1 ms up, 3 significant digits below. */
export const formatMs = (value: number | undefined): string =>
  value === undefined ? "—" : value >= 1 ? value.toFixed(2) : value.toPrecision(3);

/** A shape's label: "wan22-self-121f (24 × 27,280 → 27,280, D 128)". */
export const shapeLabel = (shape: AttentionBenchShape): string =>
  `${shape.id} (${shape.heads} × ${formatCount(shape.queries)} → ${formatCount(shape.keys)}, D ${shape.depth})`;

const routeRow = (
  result: AttentionBenchShapeResult,
  name: "s16" | "online",
  route: AttentionBenchRoute,
): BenchTableRow => {
  const base = { key: `${result.shape.id}:${name}`, shape: result.shape.id, route: name } as const;
  if (!route.eligible) return { ...base, note: `not eligible: ${route.reason}` };
  const unit = name === "s16" ? "block" : "window";
  const failed = route.stages.find((stage) => stage.error !== undefined);
  const variant =
    route.reduce === undefined
      ? route.rowWindow === true
        ? "row window"
        : "whole node"
      : `${route.reduce} · ${formatCount(route.workgroupStorageBytes ?? 0)} B`;
  return {
    ...base,
    rows: route.rows,
    split: `${route.productionDispatches} ${unit}${route.productionDispatches === 1 ? "" : "s"} × ${formatCount(route.productionRows)} rows`,
    ...(route.msPerDispatch === undefined ? {} : { msPerDispatch: route.msPerDispatch }),
    ...(route.msPer1000Rows === undefined ? {} : { msPer1000Rows: route.msPer1000Rows }),
    ...(route.tops === undefined ? {} : { tops: route.tops }),
    ...(route.estimatedNodeMs === undefined ? {} : { nodeMs: route.estimatedNodeMs }),
    ...(route.stages.length > 1
      ? {
          stages: route.stages
            .map((stage) => `${stage.stage} ${formatMs(stage.msPerDispatch)}`)
            .join(" · "),
        }
      : {}),
    variant,
    ...(name === "online" && result.onlineVsS16 !== undefined ? { ratio: result.onlineVsS16 } : {}),
    ...(route.error === undefined && failed === undefined
      ? {}
      : { note: route.error ?? `${failed?.stage}: ${failed?.error}` }),
  };
};

/** The rows of every measured shape, in record order (s16, online, then quantization). */
export const benchTableRows = (results: readonly AttentionBenchShapeResult[]): BenchTableRow[] =>
  results.flatMap((result) => {
    const rows = [routeRow(result, "s16", result.s16), routeRow(result, "online", result.online)];
    const quantization = result.quantization;
    if (quantization === undefined) return rows;
    const failed = quantization.stages.find((stage) => stage.error !== undefined);
    return [
      ...rows,
      {
        key: `${result.shape.id}:quantization`,
        shape: result.shape.id,
        route: "quantization",
        ...(quantization.msPerNode === undefined ? {} : { nodeMs: quantization.msPerNode }),
        stages: quantization.stages
          .map((stage) => `${stage.stage} ${formatMs(stage.msPerDispatch)}`)
          .join(" · "),
        ...(failed === undefined ? {} : { note: `${failed.stage}: ${failed.error}` }),
      },
    ];
  });

/**
 * One row of the candidate's table (perf-ledger K-106, the tiled key axis): its ratios to `online`
 * at one shape, the paired per-dispatch times, the producer pair and the identity flag.
 */
export type CandidateTableRow = {
  readonly key: string;
  readonly shape: string;
  /** Time tiled ÷ online, the kernel alone (the median of the per-round ratios; below 1 = tiled faster). */
  readonly kernelRatio?: number;
  /** Kernel + producer per node, tiled ÷ today (needs the quantization line). */
  readonly withProducerRatio?: number;
  /** ms per dispatch of online and of the candidate (the paired minimums). */
  readonly onlineMs?: number;
  readonly tiledMs?: number;
  /** "today 0.512 · tiled 0.640 (×1.250)" — the producer pair of the quantization line. */
  readonly producer?: string;
  /** Paired rounds that gave a ratio (a round with a clamped pass gives none). */
  readonly rounds?: number;
  /** Both producers on the same Vᵀ, both kernels on their outputs: O identical word for word. */
  readonly identical?: boolean;
  /** Why the candidate did not run, a failure, or the first differing word. */
  readonly note?: string;
};

/** A ratio as `×0.951`. */
export const formatRatio = (value: number | undefined): string =>
  value === undefined ? "—" : `×${value.toFixed(3)}`;

/** The candidate's rows, one per shape that carries it, in record order. */
export const candidateRows = (results: readonly AttentionBenchShapeResult[]): CandidateTableRow[] =>
  results.flatMap((result): CandidateTableRow[] => {
    const tiled = result.onlineTiled;
    if (tiled === undefined) return [];
    const base = { key: `${result.shape.id}:onlineTiled`, shape: result.shape.id } as const;
    if (!tiled.eligible) return [{ ...base, note: `not run: ${tiled.reason}` }];
    const { kernel, identity } = tiled;
    const pair = result.quantization?.tiledVt;
    const notes = [
      kernel.error === undefined ? undefined : `kernel: ${kernel.error}`,
      pair?.error === undefined ? undefined : `producer: ${pair.error}`,
      identity.error === undefined ? undefined : `identity: ${identity.error}`,
      identity.first === undefined
        ? undefined
        : `first differing O word ${formatCount(identity.first.index)}: 0x${identity.first.a.toString(16)} vs 0x${identity.first.b.toString(16)}`,
    ].filter((note) => note !== undefined);
    return [
      {
        ...base,
        ...(result.tiledVsOnline === undefined ? {} : { kernelRatio: result.tiledVsOnline }),
        ...(result.tiledVsOnlineWithProducer === undefined
          ? {}
          : { withProducerRatio: result.tiledVsOnlineWithProducer }),
        ...(kernel.aMsPerDispatch === undefined ? {} : { onlineMs: kernel.aMsPerDispatch }),
        ...(kernel.bMsPerDispatch === undefined ? {} : { tiledMs: kernel.bMsPerDispatch }),
        ...(pair === undefined || pair.error !== undefined
          ? {}
          : {
              producer: `today ${formatMs(pair.aMsPerDispatch)} · tiled ${formatMs(pair.bMsPerDispatch)} (${formatRatio(pair.ratio)})`,
            }),
        rounds: (kernel.rounds ?? []).filter((round) => round.ratio !== undefined).length,
        identical: identity.identical,
        ...(notes.length === 0 ? {} : { note: notes.join(" · ") }),
      },
    ];
  });
