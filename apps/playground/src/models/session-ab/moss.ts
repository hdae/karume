/**
 * The session A/B page's MOSS-Image DiT target, its pure parts: the work of one forward (the
 * sequence of a 1024 × 1024 request and the seeded feeds), the request and the arms' step reports.
 * A run is one forward of the `dit` component (the positive branch) on the fixed feeds, the arms are
 * `sessionArm`s of `@karume/runtime/tune` over the one prepared model — `base` with the preset's
 * knobs, `rule` with the same plus the rule `models/ew_chain` — and the output is the DiT's output
 * rows, so the A/B's identity is u32 identity of the output words (NaN by class). The counts (the
 * rule's hits, the dispatches) are the runner's: a `sessionArm` reports every run's fusion counts and dispatches
 * outside the timed span, so this target needs no count probe.
 *
 * NOTE: the feeds (between the twin markers) have a twin in
 * packages/e2e/tests/e2e_moss_session_ab_test.ts (the opt-in Node A/B). No module both may import
 * fits the gates (the e2e tests cannot import the playground, `@karume/candidate-rules` holds rules
 * only), so the two copies are kept textually parallel: a change to one MUST be made to the other
 * (tests/session-ab_moss_test.ts compares them).
 */

import type { Manifest, SessionSpec } from "@karume/hub";
import type { Tensor } from "@karume/runtime";
import type { SessionAbArm } from "@karume/runtime/tune";

import { MOSS_MOUNT } from "@/models/moss-image/settings";

import type { SessionAbSettings } from "./request";
import type { ArmStep } from "./run";

// ---- The MOSS DiT feeds: twin begin (the same text in e2e_moss_session_ab_test.ts and the playground's session-ab/moss.ts, `export` aside) ----

/**
 * The work of one forward: the sequence of a 1024 × 1024 request — 4,096 image tokens (64 × 64
 * latent cells) and the 63 text tokens of the fridge-note prompt, S = 4,159 — and the seed of the
 * feeds.
 */
export const MOSS_DIT_WORK = { imageTokens: 4096, textTokens: 63, seed: 0x5eed } as const;

/** A graph input as the DiT declares it (`S` its one symbol). */
export type DitInput = {
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
 * Fixed feeds of one DiT forward at `imageTokens + textTokens` tokens: every f32 input uniform in
 * [−1, 1) from the seeded generator, input by input in the graph's order (the latent tokens, the
 * text context, the RoPE tables and the timestep projection stand in for a real request's — the
 * times and the identity do not depend on them being real), and the bool `is_image` 1 for the image
 * tokens, which come first, and 0 for the text tokens. Refuses an input of another dtype or a
 * symbol other than `S`, so a changed graph does not run on feeds made for another.
 */
export const mossDitFeeds = (
  inputs: readonly DitInput[],
  imageTokens: number,
  textTokens: number,
  seed: number,
): Readonly<Record<string, Tensor>> => {
  const sequence = imageTokens + textTokens;
  const next = seededUniform(seed);
  const feeds: Record<string, Tensor> = {};
  for (const input of inputs) {
    const shape = input.shape.map((dim) => {
      if (typeof dim === "number") return dim;
      if (dim !== "S") throw new Error(`MOSS DiT feeds: input '${input.name}' has symbol '${dim}'`);
      return sequence;
    });
    const count = shape.reduce((total, dim) => total * dim, 1);
    if (input.dtype === "f32") {
      feeds[input.name] = {
        dtype: "f32",
        shape,
        data: Float32Array.from({ length: count }, () => next() * 2 - 1),
      };
    } else if (input.dtype === "bool" && input.name === "is_image" && count === sequence) {
      const data = new Uint32Array(count);
      data.fill(1, 0, imageTokens);
      feeds[input.name] = { dtype: "bool", shape, data };
    } else {
      throw new Error(
        `MOSS DiT feeds: no feed for input '${input.name}' (${input.dtype} ${JSON.stringify(input.shape)})`,
      );
    }
  }
  return feeds;
};

// ---- The MOSS DiT feeds: twin end ----

/** The component the target runs (its graph has the same name). */
export const MOSS_AB_COMPONENT = "dit";

/** The model the target loads: the distribution's model, the preset and the knobs it declares. */
export type SessionAbMossModel = {
  readonly model: string;
  readonly quant: string;
  /** The preset's `session` as the manifest declares it: the `dit` Session's knobs in both arms. */
  readonly session: SessionSpec;
};

/** The manifest's `defaultModel` (the distribution has one) at `quant`, with the knobs it declares. */
export const mossAbModel = (manifest: Manifest, quant: string): SessionAbMossModel => {
  const model = manifest.defaultModel;
  const entry = Object.hasOwn(manifest.models, model) ? manifest.models[model] : undefined;
  const preset =
    entry !== undefined && Object.hasOwn(entry.quants, quant) ? entry.quants[quant] : undefined;
  if (preset === undefined) {
    throw new Error(`The manifest has no preset '${quant}' of model '${model}'`);
  }
  return { model, quant, session: { ...preset.session } };
};

/** What one press of Run asked for on this target, as the record keeps it. */
export type SessionAbMossRequest = SessionAbMossModel &
  Pick<SessionAbSettings, "rounds" | "minGain" | "backend"> & {
    readonly target: "moss-dit-forward";
    readonly mount: typeof MOSS_MOUNT;
    readonly component: typeof MOSS_AB_COMPONENT;
    /** Always: two DiT Sessions never live together. */
    readonly openPerVisit: true;
    readonly imageTokens: number;
    readonly textTokens: number;
    readonly seed: number;
    /** The arms in order: the first is the base. */
    readonly arms: readonly { readonly name: string; readonly fusionRules: readonly string[] }[];
  };

/**
 * The request of a run on this target: the model, the settings this target reads (the token count
 * and the count probe are the Gemma target's, and every visit opens and closes its arm's Session
 * whatever the form says) and the fixed work, with the arms named.
 */
export const sessionAbMossRequest = (
  model: SessionAbMossModel,
  settings: SessionAbSettings,
  ruleName: string,
): SessionAbMossRequest => ({
  target: "moss-dit-forward",
  mount: MOSS_MOUNT,
  component: MOSS_AB_COMPONENT,
  ...model,
  rounds: settings.rounds,
  minGain: settings.minGain,
  openPerVisit: true,
  backend: settings.backend,
  ...MOSS_DIT_WORK,
  arms: [
    { name: "base", fusionRules: [] },
    { name: "rule", fusionRules: [ruleName] },
  ],
});

/**
 * `arm` with its steps reported, as the Gemma target's arms report theirs: `onStep` is called
 * with `open` before the arm opens, `ran` after its report has read a run's counts and `close`
 * before it closes — never inside `run`, so a status render never lands in a timed span. The arm's
 * runs and counts pass through unchanged.
 */
export const steppedArm = (arm: SessionAbArm, onStep: (step: ArmStep) => void): SessionAbArm => ({
  name: arm.name,
  open: async () => {
    onStep({ arm: arm.name, step: "open" });
    await arm.open();
  },
  run: arm.run,
  report: async () => {
    const counts = (await arm.report?.()) ?? {};
    onStep({ arm: arm.name, step: "ran" });
    return counts;
  },
  close: async () => {
    onStep({ arm: arm.name, step: "close" });
    await arm.close();
  },
});
