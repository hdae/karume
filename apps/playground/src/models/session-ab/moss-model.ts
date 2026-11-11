/**
 * The model of the session A/B page's MOSS-Image DiT target: the `dit` component of one preset,
 * fetched and opened through the hub on the route `moss.Pipeline.fromPretrained` takes for its
 * components (manifest → selection → part 0 → `openContainer` → `prepareContainer`, then the weight
 * parts into the hub's cache before a Session reads them), so both arms build their Sessions from
 * one prepared model. Opening builds no Session and holds no GPU memory.
 */

import {
  type AssetProgress,
  type DistributionSource,
  type HubRepoRef,
  loadManifest,
  openContainerSource,
  prefetchAssets,
  resolveSelection,
} from "@karume/hub";
import { openContainer, type PreparedModel, prepareContainer } from "@karume/runtime";

import { sha256Hex } from "@/lib/record";

import { MOSS_AB_COMPONENT } from "./moss";

/** The prepared `dit` both arms build their Sessions from, and its container's identity. */
export type MossDitModel = {
  readonly prepared: PreparedModel;
  /** sha256 of the container's part 0 (its two documents name every block's sha256). */
  readonly part0Sha256: string;
};

/**
 * Opens the `dit` component of `model` / `quant`: its descriptor first (a selection without the
 * component, or a container the runtime cannot prepare, fails before any weight byte is fetched),
 * then the weight parts, with their progress.
 */
export const openMossDitModel = async (
  ref: HubRepoRef | DistributionSource,
  model: string,
  quant: string,
  signal: AbortSignal,
  onProgress: (progress: AssetProgress) => void,
): Promise<MossDitModel> => {
  const loaded = await loadManifest(ref, { signal });
  const { containers } = resolveSelection(loaded.manifest, {
    model,
    quant,
    weights: [MOSS_AB_COMPONENT],
  });
  const container = Object.hasOwn(containers, MOSS_AB_COMPONENT)
    ? containers[MOSS_AB_COMPONENT]
    : undefined;
  if (container === undefined) {
    throw new Error(`Selection ${model} / ${quant} has no '${MOSS_AB_COMPONENT}' container`);
  }
  const [descriptor, ...weights] = container.parts;
  if (descriptor === undefined) {
    throw new Error(`The '${MOSS_AB_COMPONENT}' container of ${model} / ${quant} has no parts`);
  }
  await prefetchAssets(loaded, [descriptor], { signal });
  const source = openContainerSource(loaded, container, { signal });
  const opened = await openContainer({ kind: "source", source }, container.descriptor);
  const prepared = prepareContainer(opened, MOSS_AB_COMPONENT);
  const part0Sha256 = await sha256Hex(await opened.source.read(0, 0, opened.source.partLength(0)));
  const parts = weights.filter((part) => part.size > 0);
  if (parts.length > 0) await prefetchAssets(loaded, parts, { signal, onProgress });
  signal.throwIfAborted();
  return { prepared, part0Sha256 };
};
