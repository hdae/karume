/**
 * The MOSS-Image presets — the execution knobs of the DiT pair (`dit`, `dit_skip22`) (ADR 0138
 * decision 13, stage 5). The names follow ADR 0074's grammar (`<storage>+<component><bits>`
 * then the knobs) and are provisional until the ADR is accepted.
 *
 * | Preset                     | DiT weights | `dit` / `dit_skip22`                                                                 | `dit_text`  | VAE |
 * | -------------------------- | ----------- | ------------------------------------------------------------------------------------- | ----------- | --- |
 * | `f32+dit8` (reference)     | i8          | f32 compute (no knob)                                                                 | f32 compute | f32 |
 * | `f32+dit8-a8-attn8-online` | the same i8 | `linearCompute: "a8"`, `attentionCompute: "a8"`, `attentionScoreStorage: "none"`       | f32 compute | f32 |
 *
 * - `dit_text` runs with f32 compute in every preset (ADR 0138 decision 3 — its input is an LLM
 *   hidden state with outlier channels, and it runs twice per generation) and the VAE in f32
 *   (upstream decodes in f32), so their Sessions take no preset knob.
 * - **No default.** Every entry takes the preset explicitly: no manifest exists yet whose
 *   `defaultQuant` would own it (ADR 0138 decision 12), so a default here would be a second owner to
 *   retire — a departure from ADR 0128 decision 1 (ADR 0138 decision 13). The family's default is
 *   `f32+dit8-a8-attn8-online` (the visual review passed), applied by the manifest when it comes.
 * - The knobs reach the Sessions and the headroom estimates as one value ({@link MossPreset.dit}),
 *   so a stage's check sizes the Sessions it opens (`./dit-stage.ts` "Headroom"). The online kernel's
 *   route depends on the device (ADR 0132 decision 5), so its estimate also takes the device's
 *   attention facts (`attentionDeviceFacts` — the function the Session's own route reads).
 * - The online kernel's subgroup form needs the device's subgroups. {@link mossAcquireGpuOptions}
 *   gives the acquisition options of a preset (`subgroups: "available"` for the online preset — never
 *   required, ADR 0128 C1). Without subgroups the kernel takes its portable form (`wg`,
 *   bit-identical to `sg32` — ADR 0132 decision 5), and on a device with neither form the attention
 *   falls back to the three dispatches (the route's definition — `attentionRoute`).
 *
 * MUST: every module is side-effect free (no import-time execution, no global mutable state).
 */

import type { AcquireGpuOptions, SessionOptions } from "@karume/runtime";

import { toAcquireGpuOptions } from "../session/gpu-features.ts";

/** The knobs a preset sets on `dit` and `dit_skip22` (the fields the estimate reads too). */
export type MossDitKnobs = Pick<
  SessionOptions,
  "linearCompute" | "attentionCompute" | "attentionScoreStorage"
>;

/** The preset names (module doc). */
export const MOSS_REFERENCE_PRESET = "f32+dit8";
export const MOSS_PRACTICAL_PRESET = "f32+dit8-a8-attn8-online";

export type MossPresetName = typeof MOSS_REFERENCE_PRESET | typeof MOSS_PRACTICAL_PRESET;

/** One preset (module doc). */
export type MossPreset = {
  readonly name: MossPresetName;
  /** The Session knobs of `dit` and `dit_skip22` (empty for the reference preset). */
  readonly dit: MossDitKnobs;
};

const PRESETS: Readonly<Record<MossPresetName, MossPreset>> = Object.freeze({
  [MOSS_REFERENCE_PRESET]: Object.freeze({ name: MOSS_REFERENCE_PRESET, dit: Object.freeze({}) }),
  [MOSS_PRACTICAL_PRESET]: Object.freeze({
    name: MOSS_PRACTICAL_PRESET,
    dit: Object.freeze({
      linearCompute: "a8",
      attentionCompute: "a8",
      attentionScoreStorage: "none",
    } as const),
  }),
});

/** Every preset name, the reference preset first. */
export const MOSS_PRESET_NAMES: readonly MossPresetName[] = Object.freeze([
  MOSS_REFERENCE_PRESET,
  MOSS_PRACTICAL_PRESET,
]);

const isPresetName = (name: string): name is MossPresetName => Object.hasOwn(PRESETS, name);

/** The preset of `name`; an unknown name fails loudly with the names there are. */
export const mossPreset = (name: string): MossPreset => {
  if (!isPresetName(name)) {
    throw new Error(
      `MOSS-Image: no preset '${name}' (there are: ${MOSS_PRESET_NAMES.join(" / ")})`,
    );
  }
  return PRESETS[name];
};

/**
 * `preset`, or a named failure when a state leaves it out (module doc "No default"). The types
 * require the field; a caller outside them (JavaScript) would otherwise fail on `preset.dit` with a
 * TypeError that names no cause. MUST: every entry that reads a state's preset passes it through
 * here first — `generateMoss`, and `mossDitStageNeeds`, which `withMossDitPair` calls before it
 * opens a Session.
 */
export const requireMossPreset = (preset: MossPreset | undefined): MossPreset => {
  if (preset === undefined) {
    throw new Error("MOSS-Image: the state names no preset (moss/preset.ts — there is no default)");
  }
  return preset;
};

/**
 * The `acquireGpu` options a device for `preset` asks for (module doc): the online preset requests
 * subgroups when the adapter advertises them (the models-wide rule — `../session/gpu-features.ts`).
 */
export const mossAcquireGpuOptions = (preset: MossPreset): AcquireGpuOptions =>
  toAcquireGpuOptions(undefined, preset.dit);
