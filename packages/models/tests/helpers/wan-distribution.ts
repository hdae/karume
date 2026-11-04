/**
 * The command that builds the Wan2.1 distribution mirror (`models/karume-wan2.1/`). The e2e that read the distribution
 * and the SKIP messages of the host tests point to the same command (the same form as 2.2's
 * {@link WAN_TI2V_ASSEMBLE_COMMAND} — `wan-ti2v-pipeline.ts`).
 *
 * The mirror is local-only (not published) and is still the distribution form the tests read (ADR 0139 decision 1 —
 * an unpublished family reads its local mirror built by `dist.py`).
 *
 * Zero side effects: only constants and functions that read when called.
 */

import type { OpenedContainer } from "@karume/runtime";
import { mirrorRoot, openSelectedContainer, selectMirrorComponent } from "./mirror.ts";

/**
 * The command that builds the distribution. umT5's cross-repository reference is written with a placeholder SHA
 * (`--allow-placeholder-ref`) — the e2e rooted at the local mirror pass `models/karume-umt5-xxl/` through the source's
 * `crossRepo` mapping, so the revision is not read.
 */
export const WAN_ASSEMBLE_COMMAND =
  "cd tools/export-recipes && uv run python dist.py --pipeline wan --allow-placeholder-ref";

/** The Wan2.1 mirror (the default output of `dist.py --pipeline wan`). */
export const WAN_DIST_ROOT: URL = mirrorRoot("karume-wan2.1");
/** The mirror's model name (the key of the manifest's `models`). */
export const WAN_DIST_MODEL = "t2v-1.3b";

/**
 * Opens one component of the mirror in the given storage, through the manifest (ADR 0139 decision 1): f16 through the
 * `f16` preset, i8 through the `f16+dit8` preset (every preset that stores a component in a dtype names the same
 * container). Range reads; the Session options stay the caller's — the preset only selects the container.
 *
 * MUST: the preset's declared storage is checked — a mirror whose preset stores the component differently would
 * otherwise run the check against the other container without saying so.
 */
export const openWanComponent = async (
  component: string,
  dtype: "f16" | "i8",
): Promise<OpenedContainer> => {
  const quant = dtype === "f16" ? "f16" : "f16+dit8";
  const chosen = await selectMirrorComponent(WAN_DIST_ROOT, component, {
    model: WAN_DIST_MODEL,
    quant,
  });
  const declared = chosen.loaded.manifest.models[WAN_DIST_MODEL].quants[quant].weights[component];
  if (declared !== dtype) {
    throw new Error(
      `test: preset '${quant}' of ${WAN_DIST_ROOT.pathname} stores '${component}' as ${declared}, not ${dtype}`,
    );
  }
  return await openSelectedContainer(chosen);
};
