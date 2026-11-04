/**
 * The Wan2.2 TI2V-5B distribution mirror (`models/karume-wan2.2/` — ADR 0121 step 8) and the preset names. The
 * end-to-end e2e (`e2e_wan_ti2v_pipeline_test.ts`) and the DiT e2e (`e2e_wan_ti2v_dit_test.ts`) read the same place,
 * the same preset names and the same build command.
 *
 * `dist.py --pipeline wan-ti2v` builds the distribution from the series containers (the DiT is one i8 series, the two
 * VAEs are f16) and the two text assets of the 2.1 series (ADR 0121 decision 9 — the same umT5 embedding and
 * tokenizer). The umT5 container is, as in 2.1, a cross-repository reference to another distribution repository
 * (`karume-umt5-xxl`); the `text_encoder` part of the manifest carries the repo name.
 *
 * The source of truth for a preset's bundle of options (`session`) is the quant preset of the distribution's manifest
 * (ADR 0110 decision 1) and it is not copied here — callers that need the practical preset's bundle read the manifest
 * with {@link readWanTi2vDistributionManifestText} and look up the preset's `session`.
 *
 * Zero side effects: only URL and string constants and functions that read when called (no asset is read at module
 * evaluation).
 */

import type { OpenedContainer } from "@karume/runtime";
import { openSelectedContainer, selectMirrorComponent } from "./mirror.ts";
import { readTextIfPresent } from "./read-if-present.ts";

/** The reference preset (ADR 0121 decision 2 — i8 weights, f32 compute, empty `session`). */
export const WAN_TI2V_REFERENCE_QUANT = "f16+dit8";
/** The practical preset (ADR 0121 decision 2 — the same name and bundle as 2.1's practical preset). */
export const WAN_TI2V_PRACTICAL_QUANT = "f16+dit8-a8-attn8-s16";
/**
 * The online-attention preset (ADR 0132 decision 6 — the practical preset's weights with the scores never stored,
 * `attentionScoreStorage: "none"`; a sibling of the practical preset until its A/B check makes it the default).
 */
export const WAN_TI2V_ONLINE_QUANT = "f16+dit8-a8-attn8-online";
/** The int8 VAE decoder's reference preset (ADR 0133 decision 6 — the control of the decoder's A/B). */
export const WAN_TI2V_I8_QUANT = "i8";
/** The conv8 preset (ADR 0133 decision 6 — the online bundle plus the VAE decoder on int8 activations). */
export const WAN_TI2V_CONV8_QUANT = "i8-a8-attn8-online-conv8";

/** The distribution mirror (the default output of `dist.py --pipeline wan-ti2v`). */
export const WAN_TI2V_DIST_ROOT = new URL("../../../../models/karume-wan2.2/", import.meta.url);
/** The distribution's Wan2.2 model name (the key of the manifest's models). */
export const WAN_TI2V_DIST_MODEL = "ti2v-5b";

/**
 * The command that builds the distribution. umT5's cross-repository reference is written with a placeholder SHA
 * (`--allow-placeholder-ref`) — the e2e rooted at the local mirror pass `models/karume-umt5-xxl/` through the source's
 * `crossRepo` mapping, so the revision is not read.
 */
export const WAN_TI2V_ASSEMBLE_COMMAND =
  "cd tools/export-recipes && uv run python dist.py --pipeline wan-ti2v --allow-placeholder-ref";

/**
 * Reads the text of the distribution's manifest (`undefined` when `karume.json` is missing — the source of the SKIP
 * decision). Failures other than NotFound throw (a broken distribution is not read as "missing" and SKIPped). It is not
 * parsed here but where the caller uses it — so that a broken manifest turns red only the cases that read that manifest
 * (in the DiT e2e, only the practical-preset cases; laid out the same way as the 2.1 DiT e2e).
 */
export const readWanTi2vDistributionManifestText = (): Promise<string | undefined> =>
  readTextIfPresent(new URL("karume.json", WAN_TI2V_DIST_ROOT));

/**
 * Opens one component of the mirror in the given storage, through the manifest (ADR 0139 decision 1): f16 through the
 * reference preset, i8 through the int8 preset (every preset that stores a component in a dtype names the same
 * container). Range reads; the Session options stay the caller's — the preset only selects the container.
 *
 * MUST: the preset's declared storage is checked — a mirror whose preset stores the component differently would
 * otherwise run the check against the other container without saying so.
 */
export const openWanTi2vComponent = async (
  component: string,
  dtype: "f16" | "i8",
): Promise<OpenedContainer> => {
  const quant = dtype === "f16" ? WAN_TI2V_REFERENCE_QUANT : WAN_TI2V_I8_QUANT;
  const chosen = await selectMirrorComponent(WAN_TI2V_DIST_ROOT, component, {
    model: WAN_TI2V_DIST_MODEL,
    quant,
  });
  const declared =
    chosen.loaded.manifest.models[WAN_TI2V_DIST_MODEL].quants[quant].weights[component];
  if (declared !== dtype) {
    throw new Error(
      `test: preset '${quant}' of ${WAN_TI2V_DIST_ROOT.pathname} stores '${component}' as ${declared}, not ${dtype}`,
    );
  }
  return await openSelectedContainer(chosen);
};
