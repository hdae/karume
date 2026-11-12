/**
 * The session A/B page's two VAE targets, their pure parts: the work of one forward (the seeded
 * feeds), the VAE Session's options as each pipeline derives them from the preset, the request, the
 * arms' options and which counter an arm's row reads. A run is one forward of one VAE graph — the
 * Wan2.2 `vae_decoder_next` chunk graph (tile 16: the latent frame and its 32 caches in, four frames
 * and the 32 updated caches out) or the anima `vae_decoder` tile graph (latents `[1, 16, 64, 64]` →
 * a 512² tile; a 1024² image decodes 9 of them) — on fixed feeds. The arms are `sessionArm`s of
 * `@karume/runtime/tune` over one prepared model — `base` with the options the pipeline gives its
 * VAE besides its fusion rule, `rule` with the same plus that rule `models/vae_rms_silu` (the plan
 * the pipeline runs) — and every output is read back, so the A/B's identity is u32 identity of every
 * output word (NaN by class). The counts
 * (the rules' hits, the dispatches) are the runner's: a `sessionArm` reports them for every run.
 *
 * NOTE: the feeds (between the twin markers) have a twin in
 * packages/e2e/tests/e2e_vae_session_ab_test.ts (the opt-in Node A/B). No module both may import
 * fits the gates (the e2e tests cannot import the playground, `@karume/candidate-rules` holds rules
 * only), so the two copies are kept textually parallel: a change to one MUST be made to the other
 * (tests/session-ab_vae_test.ts compares them).
 */

import type { Manifest, SessionSpec } from "@karume/hub";
import { VAE_RMS_SILU_RULE, VAE_RMS_SILU_RULE_NAME } from "@karume/models/wan";
import type { SessionOptions, Tensor } from "@karume/runtime";

import type { MountName } from "@/mounts";

import type { SessionAbSettings } from "./request";

// ---- The VAE feeds: twin begin (the same text in e2e_vae_session_ab_test.ts and the playground's session-ab/vae.ts, `export` aside) ----

/** The two targets: the graph a run forwards once, and the seed of its feeds. */
export const VAE_AB_TARGETS = {
  "wan22-vae-next": { component: "vae_decoder_next", seed: 0x5eed },
  "anima-vae-decode": { component: "vae_decoder", seed: 0x5eed },
} as const;

/** A graph input as a VAE graph declares it (every shape static). */
export type VaeInput = {
  readonly name: string;
  readonly dtype: string;
  readonly shape: readonly (number | string)[];
};

/** mulberry32: the same seed gives the same feeds on every machine and runtime. */
export const seededUniform = (seed: number): (() => number) => {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};

/**
 * Fixed feeds of one VAE forward: every input uniform in [−1, 1) from the seeded generator, input by
 * input in the graph's order (the latent and, on a Wan chunk graph, every cache stand in for a
 * tile's — the times and the identity do not depend on them being real). Refuses an input that is
 * not f32 or has a symbolic dimension, so a changed graph does not run on feeds made for another.
 */
export const vaeFeeds = (
  graphInputs: readonly VaeInput[],
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

// ---- The VAE feeds: twin end ----

export type SessionAbVaeTarget = keyof typeof VAE_AB_TARGETS;

/** The mount each target reads its distribution from. */
export const VAE_AB_MOUNTS = {
  "wan22-vae-next": "wan22",
  "anima-vae-decode": "anima",
} as const satisfies Readonly<Record<SessionAbVaeTarget, MountName>>;

export const isVaeTarget = (target: string): target is SessionAbVaeTarget =>
  Object.hasOwn(VAE_AB_TARGETS, target);

/** The rule an arm names (`models/vae_rms_silu`, the one the Wan and anima pipelines pass). */
const vaeRuleNamed = (name: string) => {
  if (name === VAE_RMS_SILU_RULE_NAME) return VAE_RMS_SILU_RULE;
  throw new Error(`No VAE rule named ${name}`);
};

/** The VAE Session's options as JSON (the knobs the two pipelines can give their VAE). */
export type VaeSessionOptions = Pick<SessionOptions, "convCompute">;

/**
 * The VAE Session's options as the target's pipeline derives them from the preset's `session`,
 * besides the fusion rule it adds (the `rule` arm adds it here): Wan's VAE takes the conv knob alone (`wanVaeSessionOptions` of
 * packages/models/src/wan/tile-decode.ts — ADR 0133 decision 6) and anima's VAE stage none of it
 * (`withStage` of packages/models/src/anima/pipeline.ts). A copy of that rule: neither function is
 * on the public surface the playground reads.
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

/** The manifest's `defaultModel` at `quant`: its declared knobs and the component's storage. */
export const vaeAbModel = (
  manifest: Manifest,
  target: SessionAbVaeTarget,
  quant: string,
): SessionAbVaeModel => {
  const model = manifest.defaultModel;
  const { component } = VAE_AB_TARGETS[target];
  const entry = Object.hasOwn(manifest.models, model) ? manifest.models[model] : undefined;
  const preset =
    entry !== undefined && Object.hasOwn(entry.quants, quant) ? entry.quants[quant] : undefined;
  if (preset === undefined) {
    throw new Error(`The manifest has no preset '${quant}' of model '${model}'`);
  }
  const storage = Object.hasOwn(preset.weights, component) ? preset.weights[component] : undefined;
  if (storage === undefined) {
    throw new Error(`Preset '${quant}' of model '${model}' has no '${component}'`);
  }
  return { model, quant, session: { ...preset.session }, storage };
};

/** What one press of Run asked for on a VAE target, as the record keeps it. */
export type SessionAbVaeRequest = SessionAbVaeModel &
  Pick<SessionAbSettings, "rounds" | "minGain" | "backend"> & {
    readonly target: SessionAbVaeTarget;
    readonly mount: (typeof VAE_AB_MOUNTS)[SessionAbVaeTarget];
    readonly component: string;
    /** Always: two VAE Sessions never live together. */
    readonly openPerVisit: true;
    readonly seed: number;
    /** What every arm's Session gets from the preset ({@link vaeSessionOptions}). */
    readonly sessionOptions: VaeSessionOptions;
    /** The arms in order: the first is the base. */
    readonly arms: readonly { readonly name: string; readonly fusionRules: readonly string[] }[];
  };

/**
 * The request of a run on a VAE target: the model, the settings this target reads (the token count
 * and the count probe are the Gemma target's, and every visit opens and closes its arm's Session
 * whatever the form says), the options the pipeline gives the VAE besides its rule, and the two arms.
 */
export const sessionAbVaeRequest = (
  target: SessionAbVaeTarget,
  model: SessionAbVaeModel,
  settings: SessionAbSettings,
): SessionAbVaeRequest => ({
  target,
  mount: VAE_AB_MOUNTS[target],
  component: VAE_AB_TARGETS[target].component,
  ...model,
  rounds: settings.rounds,
  minGain: settings.minGain,
  openPerVisit: true,
  backend: settings.backend,
  seed: VAE_AB_TARGETS[target].seed,
  sessionOptions: vaeSessionOptions(target, model.session),
  arms: [
    { name: "base", fusionRules: [] },
    { name: "rule", fusionRules: [VAE_RMS_SILU_RULE_NAME] },
  ],
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

/**
 * The counter an arm's row shows as its hits: the rule the arm adds, or for an arm that adds none
 * (the base) the first rule of the request, which reads 0 there.
 */
export const vaeRowRule = (request: SessionAbVaeRequest, arm: string): string => {
  const rules = request.arms.flatMap(({ fusionRules }) => fusionRules);
  const own = request.arms.find(({ name }) => name === arm)?.fusionRules[0];
  const rule = own ?? rules[0];
  if (rule === undefined) throw new Error("The request's arms add no rule");
  return rule;
};

/**
 * Why a VAE target does not run on `backend`, or `undefined`. MUST NOT run on Metal: the rules'
 * XOR rounding barrier is measured not to hold there (docs/known-issues.md), so their arms would
 * not be the plans they stand for.
 */
export const vaeBackendRefusal = (backend: SessionAbSettings["backend"]): string | undefined =>
  backend === "metal"
    ? "The VAE rules must not run on Metal: their rounding barrier is measured not to hold there."
    : undefined;
