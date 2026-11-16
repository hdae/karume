/**
 * The `gemma4-session-ab` entry's parameters → the model it opens and the request it runs: the
 * model's text form `<mount>/<model>` (empty = the default model of the first mount whose mirror is
 * present), the preset, the load options (f16 KV, the expert cache of Gemma 4 26B A4B) and the
 * settings. Every value that does not fit is a `ParameterError` that says what exists.
 */

import {
  hasExpertCache,
  type ModelKey,
  parseExpertCacheSlots,
  parseExpertHostTier,
} from "@karume/gemma4-app-support/settings";
import { loadManifest, type Manifest } from "@karume/hub";
import type { SourceKind } from "@karume/local-distributions/browser";

import {
  booleanValue,
  integerValue,
  numberValue,
  type OpenDistribution,
  type OpenedDistribution,
  ParameterError,
  type ParameterValues,
  textValue,
} from "../entry.ts";
import { modelRefusal, MOUNT_LIST, parseModelKey, presentMounts } from "../gemma4/open.ts";
import { type SessionAbModel, type SessionAbRequest, sessionAbRequest } from "./gemma-request.ts";
import { presetOf } from "./shared.ts";

/**
 * The model's text form → its key, or `undefined` for the empty default. Refuses text that is not
 * `<mount>/<model>` and a mount that is not one of the three.
 */
export const parseModelText = (text: string): ModelKey | undefined =>
  text === ""
    ? undefined
    : parseModelKey(text, {
        form: `expected <mount>/<model> with a mount of ${MOUNT_LIST} (empty: the default), got "${text}"`,
        refuse: (reason) => new ParameterError("model", reason),
      });

/**
 * Refuses a model the manifest does not have (naming the models it has), and a model of the mobile
 * QAT mount its entry point does not accept (naming the ones it does).
 */
const checkModel = (key: ModelKey, manifest: Manifest): void => {
  const refusal = modelRefusal(key, manifest);
  if (refusal === undefined) return;
  throw new ParameterError(
    "model",
    refusal.kind === "unknown"
      ? `${key.mount} has no model '${key.model}'; its models are ${refusal.models.join(", ")}`
      : `gemma4QatMobile.Pipeline does not accept '${key.model}'; the models of ${key.mount} it accepts are ${refusal.accepted.join(", ")}`,
  );
};

/** The model a run loads, the distribution it is read from and its manifest. */
export type OpenedGemmaModel = {
  readonly key: ModelKey;
  readonly opened: OpenedDistribution;
  readonly manifest: Manifest;
};

/**
 * Opens the model `chosen` names, or for `undefined` the default model of the first of the three
 * mounts (in their order) whose mirror is present. A missing mirror of a named mount is the
 * `MountUnavailableError` of `openDistribution`.
 */
export const openGemmaModel = async (
  chosen: ModelKey | undefined,
  source: SourceKind,
  openDistribution: OpenDistribution,
  signal: AbortSignal,
): Promise<OpenedGemmaModel> => {
  if (chosen !== undefined) {
    const opened = await openDistribution(chosen.mount, source);
    const { manifest } = await loadManifest(opened.ref, { signal });
    checkModel(chosen, manifest);
    return { key: chosen, opened, manifest };
  }
  for await (const { mount, opened, manifest } of presentMounts(openDistribution, source, signal)) {
    const key = { mount, model: manifest.defaultModel };
    checkModel(key, manifest);
    return { key, opened, manifest };
  }
  throw new ParameterError(
    "model",
    `the default is the default model of the first of ${MOUNT_LIST} whose mirror is present, and none is`,
  );
};

/**
 * The expert cache the parameters ask for: absent when none of its three parameters is set. Refuses
 * any of them on a model without the cache, a slots or host tier text that does not parse, and a
 * host tier or a depth without slots.
 */
const expertCacheOf = (
  parameters: ParameterValues,
  key: ModelKey,
): SessionAbModel["expertCache"] => {
  const slotsText = textValue(parameters, "expertCacheSlots");
  const hostTierText = textValue(parameters, "hostTier");
  const depth = integerValue(parameters, "depth");
  const set = [
    ...(slotsText.trim() === "" ? [] : ["expertCacheSlots"]),
    ...(hostTierText.trim() === "" ? [] : ["hostTier"]),
    ...(depth === 0 ? [] : ["depth"]),
  ];
  if (!hasExpertCache(key)) {
    const [first] = set;
    if (first === undefined) return undefined;
    throw new ParameterError(
      first,
      `the expert cache is only for gemma4-qat/26b (a mixture-of-experts model), not ${key.mount}/${key.model}`,
    );
  }
  const slots = parseExpertCacheSlots(slotsText);
  if (!slots.ok) throw new ParameterError("expertCacheSlots", slots.errors.join(" "));
  const hostBytes = parseExpertHostTier(hostTierText);
  if (!hostBytes.ok) throw new ParameterError("hostTier", hostBytes.errors.join(" "));
  if (slots.value === undefined) {
    const [first] = set;
    if (first === undefined) return undefined;
    throw new ParameterError(first, "needs expertCacheSlots (it applies to the expert cache)");
  }
  return {
    slots: slots.value,
    ...(hostBytes.value === undefined ? {} : { hostBytes: hostBytes.value }),
    ...(depth === 0 ? {} : { depth }),
  };
};

/** The request a run makes of `key` in `manifest` from the parameters, with the rule's arm named. */
export const gemmaRequest = (
  parameters: ParameterValues,
  key: ModelKey,
  manifest: Manifest,
  ruleName: string,
): SessionAbRequest => {
  const expertCache = expertCacheOf(parameters, key);
  return sessionAbRequest(
    {
      mount: key.mount,
      model: key.model,
      quant: presetOf(manifest, key.model, textValue(parameters, "preset")).name,
      ...(booleanValue(parameters, "f16Kv") ? { fullStateStorage: "f16" as const } : {}),
      ...(expertCache === undefined ? {} : { expertCache }),
    },
    {
      rounds: integerValue(parameters, "rounds"),
      tokens: integerValue(parameters, "tokens"),
      minGain: numberValue(parameters, "minGain"),
      openPerVisit: booleanValue(parameters, "openPerVisit"),
      probeCounts: booleanValue(parameters, "probeCounts"),
    },
    ruleName,
  );
};
