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
 * - **The default is the manifest's** `defaultQuant` (`f32+dit8-a8-attn8-online` in
 *   `karume-moss-image` — the visual review passed; ADR 0128 decision 1, ADR 0138 decision 13 and
 *   its addendum of 2026-11-01), resolved by the pipeline (`./pipeline.ts`). This table holds no
 *   default of its own (one owner): the internal entries take the preset explicitly.
 * - **The manifest names a preset; this table owns its knobs** ({@link mossPresetOf}). A quant's
 *   `session` must declare exactly the knobs of the preset of the same name: the checks that make a
 *   preset safe (the self A/B check, the census, the sha rows whose IDs start with the preset's name
 *   — ADR 0138 decision 11) were run on these knobs, so a manifest declaring other knobs under a
 *   checked name, or a preset this table does not know, fails loudly before a weight byte is
 *   fetched. A new preset is added here with its checks first (ADR 0110).
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

import type { SessionSpec } from "@karume/hub";
import type { AcquireGpuOptions, SessionOptions } from "@karume/runtime";

import { toAcquireGpuOptions } from "../session/gpu-features.ts";
import { type FamilySessionPolicy, resolveSessionOptions } from "../session/options.ts";

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
 * The keys a MOSS-Image manifest's `session` may declare (the acceptance table —
 * `../session/options.ts`): the DiT pair's three knobs ({@link MossDitKnobs}). The others were not
 * checked on this family, so a declaration of one fails before the weights are fetched.
 *
 * NOTE: `export` is for the host tests (not on a public surface).
 */
export const MOSS_SESSION_POLICY: FamilySessionPolicy = {
  linearCompute: true,
  convCompute: false,
  attentionCompute: true,
  attentionScoreStorage: true,
  linearGemvReduce: false,
  stateAttentionReduce: false,
  fuseRmsNormAdd: false,
  fuseLinearStaticQuantize: false,
  fuseGeluMulStaticQuantize: false,
  packedStaticQuantize: false,
  integerGemv: false,
};

/** The same knobs (keys and values) in both. */
const sameKnobs = (left: SessionOptions, right: SessionOptions): boolean => {
  const keys = Object.keys(left);
  return (
    keys.length === Object.keys(right).length &&
    keys.every(
      (key) =>
        Object.hasOwn(right, key) &&
        Object.is(left[key as keyof SessionOptions], right[key as keyof SessionOptions]),
    )
  );
};

/**
 * The preset a manifest's quant `name` with the declared `session` stands for (module doc "The
 * manifest names a preset"). `where` names the quant in the messages; every failure is a plain
 * `Error` (the distribution's, not the request's — ADR 0107 decision 2).
 */
export const mossPresetOf = (name: string, session: SessionSpec, where: string): MossPreset => {
  if (!isPresetName(name)) {
    throw new Error(
      `${where}: this implementation has no preset '${name}' (it reads ${MOSS_PRESET_NAMES.join(" / ")})`,
    );
  }
  const preset = PRESETS[name];
  const declared = resolveSessionOptions(MOSS_SESSION_POLICY, session, {}, where);
  if (!sameKnobs(declared, preset.dit)) {
    throw new Error(
      `${where}: the manifest declares the knobs ${JSON.stringify(declared)}, but the preset` +
        ` '${name}' is ${JSON.stringify(preset.dit)} (moss/preset.ts)`,
    );
  }
  return preset;
};

/**
 * `preset`, or a named failure when a state leaves it out (module doc "The default is the
 * manifest's" — the internal entries have none). The types
 * require the field; a caller outside them (JavaScript) would otherwise fail on `preset.dit` with a
 * TypeError that names no cause. MUST: every entry that reads a state's preset passes it through
 * here first — `generateMoss`, and `mossDitStageNeeds`, which `withMossDitPair` calls before it
 * opens a Session.
 */
export const requireMossPreset = (preset: MossPreset | undefined): MossPreset => {
  if (preset === undefined) {
    throw new Error(
      "MOSS-Image: the state names no preset (moss/preset.ts — the internal entries have no default)",
    );
  }
  return preset;
};

/**
 * The `acquireGpu` options a device for `preset` asks for (module doc): the online preset requests
 * subgroups when the adapter advertises them (the models-wide rule — `../session/gpu-features.ts`).
 */
export const mossAcquireGpuOptions = (preset: MossPreset): AcquireGpuOptions =>
  toAcquireGpuOptions(undefined, preset.dit);
