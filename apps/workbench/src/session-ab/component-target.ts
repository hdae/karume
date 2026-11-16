/**
 * The `session-ab` entry's three Session targets: which mount each reads, the request a run makes
 * from the manifest and the parameters, each arm's Session options and the seeded feeds of a run.
 * Pure, so `vp test` covers it.
 */

import type { Manifest } from "@karume/hub";
import type { MountName } from "@karume/local-distributions/mounts";
import { EW_CHAIN_RULE, EW_CHAIN_RULE_NAME } from "@karume/models/moss";
import type { SessionOptions, Tensor } from "@karume/runtime";

import type { GraphInput } from "./feeds.ts";
import {
  MOSS_MOUNT,
  mossAbModel,
  mossDitFeeds,
  type SessionAbMossRequest,
  sessionAbMossRequest,
} from "./moss.ts";
import { presetOf, type RoundsAndGain } from "./shared.ts";
import {
  type SessionAbVaeRequest,
  sessionAbVaeRequest,
  VAE_AB_MOUNTS,
  vaeAbModel,
  vaeArmOptions,
  vaeFeeds,
} from "./vae.ts";

/** The targets, the smallest read last (it is the default). */
export const COMPONENT_TARGETS = [
  "moss-dit-forward",
  "wan22-vae-next",
  "anima-vae-decode",
] as const;
export type ComponentTarget = (typeof COMPONENT_TARGETS)[number];

/** What one run of a Session target asked for. */
export type ComponentRequest = SessionAbMossRequest | SessionAbVaeRequest;

/** The mount a target reads its distribution from. */
export const targetMount = (target: ComponentTarget): MountName =>
  target === "moss-dit-forward" ? MOSS_MOUNT : VAE_AB_MOUNTS[target];

/**
 * The request of a run on `target`: the manifest's default model at `preset` (empty = the model's
 * default preset; another name that is not a preset of it is a `ParameterError` listing them), with
 * the settings and the target's fixed work.
 */
export const componentRequest = (
  target: ComponentTarget,
  manifest: Manifest,
  preset: string,
  settings: RoundsAndGain,
): ComponentRequest => {
  const model = manifest.defaultModel;
  const found = presetOf(manifest, model, preset);
  return target === "moss-dit-forward"
    ? sessionAbMossRequest(mossAbModel(model, found), settings, EW_CHAIN_RULE_NAME)
    : sessionAbVaeRequest(target, vaeAbModel(target, model, found), settings);
};

/**
 * Each arm's Session options: on the MOSS-Image DiT the preset's knobs, plus `models/ew_chain` for
 * an arm that names it; on a VAE the options its pipeline gives it, plus the rule the arm names.
 */
export const componentArmOptions = (
  request: ComponentRequest,
): readonly { readonly name: string; readonly options: SessionOptions }[] => {
  if (request.target !== "moss-dit-forward") return vaeArmOptions(request);
  const knobs: SessionOptions = { ...request.session };
  return request.arms.map(({ name, fusionRules }) => ({
    name,
    options: fusionRules.length === 0 ? knobs : { ...knobs, fusionRules: [EW_CHAIN_RULE] },
  }));
};

/** The fixed feeds of one forward of the request's graph (`inputs` as the graph declares them). */
export const componentFeeds = (
  request: ComponentRequest,
  inputs: readonly GraphInput[],
): Readonly<Record<string, Tensor>> =>
  request.target === "moss-dit-forward"
    ? mossDitFeeds(inputs, request.imageTokens, request.textTokens, request.seed)
    : vaeFeeds(inputs, request.seed);
