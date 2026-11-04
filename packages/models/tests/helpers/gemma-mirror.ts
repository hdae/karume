/**
 * Builds gemma's full-surface input (`Assets`) and the PLE reader from a distribution mirror
 * (`karume/5`) — **the one helper the real-asset e2e tests share**.
 *
 * ## Why the mirror, not the series output
 *
 * The real-weight tests read the distribution mirrors (ADR 0139 decision 1). Only a manifest
 * carries the part-list keys (`<component>[i]`) that the full surface of `fromAssets` takes, the
 * model / quant selection axes, and the expected hashes of the two documents
 * (`container.descriptor`); a series output has no manifest. A caller that has already opened the
 * `model` container hands it to `seriesPleHandle` of `helpers/ple-series.ts` instead of opening
 * it again. The weights and the PLE are the same bake as the series the goldens were taken from,
 * so a golden assertion holds on either.
 *
 * NOTE: nothing **test-only** from hub / runtime is imported (the rule of
 * `helpers/memory-cache.ts`). This file touches only the public surface (`@karume/hub` /
 * `@karume/hub/node` / `@karume/runtime` and the public subpaths of models).
 */

import { readFile } from "@karume/test-support/fs";
import type { AssetReader, OpenedContainer } from "@karume/runtime";
import { type Assets, parsePipelineConfig } from "../../gemma.ts";
import {
  gemma4PleAssetSource,
  type Gemma4PleIndex,
  readGemma4PleIndex,
} from "../../src/gemma/ple-index.ts";
import {
  type MirrorChoice,
  mirrorHasModel,
  mirrorRoot,
  openSelectedContainer,
  readMirrorAsset,
  selectMirrorComponent,
} from "./mirror.ts";
import { readFileIfPresent } from "./read-if-present.ts";
import { SERIES_CHECKS } from "./series-checks.ts";

// The mirror resolver, the presence check and the component opener are shared with every family
// (`helpers/mirror.ts`); they stay exported here under their existing names.
export {
  MIRROR_FORMAT,
  mirrorPresent as mirrorAvailable,
  type MirrorChoice,
  openMirrorComponent,
} from "./mirror.ts";

const MODEL = "model";

const selectionOf = (mirror: URL, choice: MirrorChoice) =>
  selectMirrorComponent(mirror, MODEL, choice);

/**
 * Opens the mirror's `model` container with **range reads** (reads only part 0 and the blocks
 * touched afterwards).
 *
 * Unlike {@link gemma4MirrorAssets}, which loads every part into RAM, the PLE reader is taken from
 * here.
 */
const openModelContainer = async (mirror: URL, choice: MirrorChoice): Promise<OpenedContainer> =>
  await openSelectedContainer(await selectionOf(mirror, choice));

/**
 * The PLE index and the block reader (the input of `createGemma4Ple` / `createGemma4PleResident`).
 */
export type Gemma4PleHandle = {
  readonly index: Gemma4PleIndex;
  readonly openBlock: (asset: string) => AssetReader;
};

/**
 * Reads the PLE index from the mirror's `model` container and hands out the block reader.
 *
 * This also runs the cross-check of the index against the container's assets
 * (`readGemma4PleIndex`), so a mirror where only one of the two was re-baked fails here.
 */
export const openGemma4Ple = async (
  mirror: URL,
  choice: MirrorChoice = {},
): Promise<Gemma4PleHandle> => {
  const opened = await openModelContainer(mirror, choice);
  return {
    index: await readGemma4PleIndex(`test: ${mirror.pathname}`, gemma4PleAssetSource(opened)),
    openBlock: (asset) => opened.asset(asset),
  };
};

/**
 * Builds the input of `fromAssets` from the mirror (the `model` container's parts + the tokenizer).
 *
 * MUST: zero-length parts are listed too — the index is the part id inside the container, so
 * skipping one shifts every later part down by one and they are read as other parts.
 */
export const gemma4MirrorAssets = async (
  mirror: URL,
  choice: MirrorChoice = {},
): Promise<Assets> => {
  const { loaded, selection, container } = await selectionOf(mirror, choice);
  const modelName = choice.model ?? loaded.manifest.defaultModel;
  const tokenizer = selection.assets["tokenizer"];
  if (tokenizer === undefined) {
    throw new Error(`test: the manifest of ${mirror.pathname} has no asset 'tokenizer'`);
  }
  return {
    config: parsePipelineConfig(loaded.manifest.models[modelName].pipelineConfig),
    model: await Promise.all(container.parts.map((part) => readFile(new URL(part.path, mirror)))),
    tokenizer: await readFile(new URL(tokenizer.path, mirror)),
  };
};

/**
 * The q4_0 QAT line's distribution mirror (E2B, E4B and 12B with their QAT drafters — the only
 * distribution that ships 12B).
 */
export const GEMMA4_QAT_MIRROR: URL = mirrorRoot("karume-gemma4-qat");

/** gemma-4-12B-it's tokenizer series (`gemma4.tokenizer --model 12b`). */
export const GEMMA4_12B_TOKENIZER_SERIES = "gemma4-12b-tokenizer";

/** Whether {@link readGemma4_12bTokenizer} reads the mirror (false = the opt-in series, or nothing). */
export const GEMMA4_12B_IN_MIRROR: boolean = mirrorHasModel(GEMMA4_QAT_MIRROR, "12b");

/**
 * gemma-4-12B-it's compiled tokenizer (its own asset — its glitch-token fix drops other BPE merges than
 * E2B's and E4B's):
 * the mirror's model `12b` asset when the mirror carries it, else the series as an
 * implementation-time check under KARUME_SERIES=1 (ADR 0139 decision 6), else `undefined`.
 */
export const readGemma4_12bTokenizer = async (): Promise<Uint8Array<ArrayBuffer> | undefined> => {
  if (GEMMA4_12B_IN_MIRROR)
    return await readMirrorAsset(GEMMA4_QAT_MIRROR, "tokenizer", { model: "12b" });
  if (!SERIES_CHECKS) return undefined;
  return await readFileIfPresent(
    new URL(
      `../../../../outputs/series/${GEMMA4_12B_TOKENIZER_SERIES}/tokenizer.json`,
      import.meta.url,
    ),
  );
};
