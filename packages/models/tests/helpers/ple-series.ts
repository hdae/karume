/**
 * The PLE of an **already opened** `model` container, on the same surface as the mirror helper
 * ({@link Gemma4PleHandle}).
 *
 * The PLE is an asset of the `model` container (the `ple_index` index plus the `ple-values` /
 * `ple-scales` block roles — ADR 0109 decision 4), so whoever opened the container — through a
 * distribution manifest (`helpers/mirror.ts`) or from a container file written in the series
 * layout (the tracked packed-PLE fixtures of `gemma4_ple_packed_test.ts`) — reads the index with
 * the public parts of models (`gemma4PleAssetSource` + `readGemma4PleIndex`). Callers that build a
 * Session from the same container pass it here instead of opening it twice.
 *
 * NOTE: the default-lane real-weight e2e open the distribution mirror (ADR 0139 decision 1) and
 * hand the opened container to {@link seriesPleHandle}, or use `openGemma4Ple` of
 * `helpers/gemma-mirror.ts`; only the opt-in series checks (`KARUME_SERIES=1`) still open series
 * containers, and they do not read the PLE through this helper.
 */

import type { OpenedContainer } from "@karume/runtime";
import { gemma4PleAssetSource, readGemma4PleIndex } from "../../src/gemma/ple-index.ts";
import type { Gemma4PleHandle } from "./gemma-mirror.ts";

/**
 * The representative path of a `model` container in the series layout (the parts are numbered —
 * `resolveParts` finds them).
 */
export const SERIES_MODEL_FILE = "model.krm";

/**
 * Builds the PLE index and the block reader from an already opened container.
 *
 * `readGemma4PleIndex` checks the index and the container's assets **in both directions** (whether
 * the assets the index points to exist / whether the container's PLE assets are in the index /
 * their roles and logical lengths), so a container where only one side was rebuilt fails here.
 */
export const seriesPleHandle = async (
  opened: OpenedContainer,
  where: string,
): Promise<Gemma4PleHandle> => ({
  index: await readGemma4PleIndex(where, gemma4PleAssetSource(opened)),
  openBlock: (asset) => opened.asset(asset),
});
