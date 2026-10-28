/**
 * The memory estimate of the BiRefNet page, before any weight is read: the matte graph's descriptor
 * (part 0 of the container, under 1 MiB) is fetched and opened, and the runtime's estimate runs on
 * it with the device's limits. The estimate runs the same intermediate-buffer planning as the
 * Session (ADR 0093 decision 5): an intermediate larger than `maxStorageBufferBindingSize` (or a
 * region larger than `maxBufferSize`) throws, naming every node. The Session plans its
 * intermediates when it first runs, so without this the failure would come on the first
 * segmentation, after the weights were read and uploaded. The manifest's `requiredLimits` cannot
 * show it either: they cover the resident part only (and both distributions declare none).
 *
 * The same route as `birefnet.Pipeline.fromPretrained` up to its admission step (manifest →
 * selection → part 0 → `openContainer` → `prepareContainer`).
 */

import {
  type DistributionSource,
  type HubRepoRef,
  loadManifest,
  openContainerSource,
  prefetchAssets,
  resolveSelection,
} from "@karume/hub";
import { type AdmissionReport, openContainer, prepareContainer } from "@karume/runtime";

import type { Selection } from "./settings";

/** The pipeline's one component; its graph has the same name (the writer's convention). */
const MATTE = "matte";

/** The device limits the estimate checks against (`GpuContext.limits` has both). */
export type EstimateLimits = {
  readonly maxStorageBufferBindingSize: number;
  readonly maxBufferSize: number;
};

/**
 * The estimate of one selection on one device. The birefnet family resolves no execution knobs
 * (its session policy accepts none, and both presets declare `session: {}`), so the estimate's
 * defaults are the Session's settings.
 */
export const estimateMatte = async (
  ref: HubRepoRef | DistributionSource,
  selection: Selection,
  limits: EstimateLimits,
  signal: AbortSignal,
): Promise<AdmissionReport> => {
  const loaded = await loadManifest(ref, { signal });
  const { containers } = resolveSelection(loaded.manifest, {
    model: selection.model,
    quant: selection.quant,
    weights: [MATTE],
  });
  const container = Object.hasOwn(containers, MATTE) ? containers[MATTE] : undefined;
  if (container === undefined) {
    throw new Error(
      `Selection ${selection.model} / ${selection.quant} has no '${MATTE}' container`,
    );
  }
  const [descriptorPart] = container.parts;
  if (descriptorPart === undefined) {
    throw new Error(
      `The '${MATTE}' container of ${selection.model} / ${selection.quant} has no parts`,
    );
  }
  // Part 0 only: the header and the graph and model descriptors (no weight block).
  await prefetchAssets(loaded, [descriptorPart], { signal });
  const source = openContainerSource(loaded, container, { signal });
  const opened = await openContainer({ kind: "source", source }, container.descriptor);
  signal.throwIfAborted();
  return prepareContainer(opened, MATTE).estimate({
    maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
    maxBufferSize: limits.maxBufferSize,
  });
};

/** What the status card shows of an estimate. */
export type EstimateSummary = {
  /** The resident weights (compressed, uncompressed and expanded). */
  readonly weightsBytes: number;
  /** The largest run shape: its input and readback buffers plus its intermediates (approximate). */
  readonly runBytes: number;
  /** Weights plus the kept intermediates (the runtime's `peakAccountedBytes`, not an upper bound). */
  readonly peakBytes: number;
  /** What the estimate does not count, as the runtime lists it. */
  readonly unaccounted: readonly string[];
};

export const summarizeEstimate = (report: AdmissionReport): EstimateSummary => ({
  weightsBytes: report.resident.weights.totalBytes,
  runBytes: Math.max(
    0,
    ...report.scenarios.map((scenario) => scenario.ioBytes + scenario.workspaceBytes),
  ),
  peakBytes: report.peakAccountedBytes,
  unaccounted: report.unaccounted,
});
