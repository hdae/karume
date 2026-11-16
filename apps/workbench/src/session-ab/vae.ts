/**
 * The session A/B's two VAE targets, their pure parts: the work of one forward (the seeded
 * feeds), the VAE Session's options as each pipeline derives them from the preset, the request and
 * the arms' options. A run is one forward of one VAE graph — the
 * Wan2.2 `vae_decoder_next` chunk graph (tile 16: the latent frame and its 32 caches in, four frames
 * and the 32 updated caches out) or the anima `vae_decoder` tile graph (latents `[1, 16, 64, 64]` →
 * a 512² tile; a 1024² image decodes 9 of them) — on fixed feeds. The arms are `sessionArm`s of
 * `@karume/runtime/tune` over one prepared model — `base` with the options the pipeline gives its
 * VAE besides its fusion rule, `rule` with the same plus that rule `models/vae_rms_silu` (the plan
 * the pipeline runs) — and every output is read back, so the A/B's identity is u32 identity of every
 * output word (NaN by class). The counts (the rule's hits, the dispatches) are the runner's: a
 * `sessionArm` reports them for every run.
 */

import type { SessionSpec } from "@karume/hub";
import type { MountName } from "@karume/local-distributions/mounts";
import { VAE_RMS_SILU_RULE, VAE_RMS_SILU_RULE_NAME } from "@karume/models/wan";
import type { SessionOptions, Tensor } from "@karume/runtime";

import type { FoundPreset } from "../model-preset.ts";
import { type GraphInput, seededUniform } from "./feeds.ts";
import { type RequestArm, type RoundsAndGain, twoArms } from "./shared.ts";

/** The two targets: the graph a run forwards once, and the seed of its feeds. */
export const VAE_AB_TARGETS = {
  "wan22-vae-next": { component: "vae_decoder_next", seed: 0x5eed },
  "anima-vae-decode": { component: "vae_decoder", seed: 0x5eed },
} as const;

/**
 * Fixed feeds of one VAE forward (every shape of a VAE graph is static): every input uniform in
 * [−1, 1) from the seeded generator, input by input in the graph's order (the latent and, on a Wan
 * chunk graph, every cache stand in for a tile's — the times and the identity do not depend on them
 * being real). Refuses an input that is not f32 or has a symbolic dimension, so a changed graph does
 * not run on feeds made for another.
 */
export const vaeFeeds = (
  graphInputs: readonly GraphInput[],
  seed: number,
): Readonly<Record<string, Tensor>> => {
  const next = seededUniform(seed);
  const feeds: Record<string, Tensor> = {};
  for (const input of graphInputs) {
    const shape = input.shape.map((dim) => {
      if (typeof dim === "number") return dim;
      throw new Error(`VAE feeds: input '${input.name}' has symbol '${dim}'`);
    });
    if (input.dtype !== "f32") {
      throw new Error(`VAE feeds: no feed for input '${input.name}' (${input.dtype})`);
    }
    const count = shape.reduce((total, dim) => total * dim, 1);
    feeds[input.name] = {
      dtype: "f32",
      shape,
      data: Float32Array.from({ length: count }, () => next() * 2 - 1),
    };
  }
  return feeds;
};

export type SessionAbVaeTarget = keyof typeof VAE_AB_TARGETS;

/** The mount each target reads its distribution from. */
export const VAE_AB_MOUNTS = {
  "wan22-vae-next": "wan22",
  "anima-vae-decode": "anima",
} as const satisfies Readonly<Record<SessionAbVaeTarget, MountName>>;

/** The rule an arm names (`models/vae_rms_silu`, the one the Wan and anima pipelines pass). */
const vaeRuleNamed = (name: string) => {
  if (name === VAE_RMS_SILU_RULE_NAME) return VAE_RMS_SILU_RULE;
  throw new Error(`No VAE rule named ${name}`);
};

/** The VAE Session's options as JSON (the knobs the two pipelines can give their VAE). */
export type VaeSessionOptions = Pick<SessionOptions, "convCompute">;

/**
 * The VAE Session's options as the target's pipeline derives them from the preset's `session`,
 * besides the fusion rule it adds (the `rule` arm adds it here): Wan's VAE takes the conv knob
 * alone (`wanVaeSessionOptions` of packages/models/src/wan/tile-decode.ts — ADR 0133 decision 6)
 * and anima's VAE stage none of it (`withStage` of packages/models/src/anima/pipeline.ts). A copy of
 * that rule: neither function is on the public surface the workbench reads.
 */
export const vaeSessionOptions = (
  target: SessionAbVaeTarget,
  session: SessionSpec,
): VaeSessionOptions =>
  target === "wan22-vae-next" && session.convCompute !== undefined
    ? { convCompute: session.convCompute }
    : {};

/** The model a target loads: the distribution's model, the preset, what it declares and stores. */
export type SessionAbVaeModel = {
  readonly model: string;
  readonly quant: string;
  /** The preset's `session` as the manifest declares it. */
  readonly session: SessionSpec;
  /** The storage of the target's component in the preset (`i8`, `f16`, ...). */
  readonly storage: string;
};

/**
 * `model` at `preset` (the one `presetOf` found): its declared knobs and the target component's
 * storage. Refuses a preset without the component.
 */
export const vaeAbModel = (
  target: SessionAbVaeTarget,
  model: string,
  preset: FoundPreset,
): SessionAbVaeModel => {
  const { component } = VAE_AB_TARGETS[target];
  const { weights, session } = preset.declaration;
  const storage = Object.hasOwn(weights, component) ? weights[component] : undefined;
  if (storage === undefined) {
    throw new Error(`Preset '${preset.name}' of model '${model}' has no '${component}'`);
  }
  return { model, quant: preset.name, session: { ...session }, storage };
};

/** What one run asked for on a VAE target, as the record keeps it. */
export type SessionAbVaeRequest = SessionAbVaeModel &
  RoundsAndGain & {
    readonly target: SessionAbVaeTarget;
    readonly mount: (typeof VAE_AB_MOUNTS)[SessionAbVaeTarget];
    readonly component: string;
    /** Always: two VAE Sessions never live together. */
    readonly openPerVisit: true;
    readonly seed: number;
    /** What every arm's Session gets from the preset ({@link vaeSessionOptions}). */
    readonly sessionOptions: VaeSessionOptions;
    /** The arms in order: the first is the base. */
    readonly arms: readonly RequestArm[];
  };

/**
 * The request of a run on a VAE target: the model, the settings this target reads (the token count
 * and the count probe are the Gemma target's, and every visit opens and closes its arm's Session
 * whatever the settings say), the options the pipeline gives the VAE besides its rule, and the two
 * arms.
 */
export const sessionAbVaeRequest = (
  target: SessionAbVaeTarget,
  model: SessionAbVaeModel,
  settings: RoundsAndGain,
): SessionAbVaeRequest => ({
  target,
  mount: VAE_AB_MOUNTS[target],
  component: VAE_AB_TARGETS[target].component,
  ...model,
  rounds: settings.rounds,
  minGain: settings.minGain,
  openPerVisit: true,
  seed: VAE_AB_TARGETS[target].seed,
  sessionOptions: vaeSessionOptions(target, model.session),
  arms: twoArms(VAE_RMS_SILU_RULE_NAME),
});

/** Each arm's Session options: the request's, plus the rules the arm names. */
export const vaeArmOptions = (
  request: SessionAbVaeRequest,
): readonly { readonly name: string; readonly options: SessionOptions }[] =>
  request.arms.map(({ name, fusionRules }) => ({
    name,
    options:
      fusionRules.length === 0
        ? { ...request.sessionOptions }
        : {
            ...request.sessionOptions,
            fusionRules: fusionRules.map(vaeRuleNamed),
          },
  }));
