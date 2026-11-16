/**
 * The `gemma4-session-ab` entry's parameters → the model it opens and the request it runs: the
 * model's text form `<mount>/<model>` (empty = the default model of the first mount whose mirror is
 * present), the preset, the load options (f16 KV, the expert cache of Gemma 4 26B A4B) and the
 * settings. Every value that does not fit is a `ParameterError` that says what exists.
 */

import { loadManifest, type Manifest } from "@karume/hub";
import { MountUnavailableError, type SourceKind } from "@karume/local-distributions/browser";

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
import {
  GEMMA_MOUNTS,
  hasExpertCache,
  isGemmaMount,
  isQatModel,
  type ModelKey,
  parseExpertCacheSlots,
  parseExpertHostTier,
} from "../gemma4/settings.ts";
import {
  presetOf,
  type SessionAbModel,
  type SessionAbRequest,
  sessionAbRequest,
} from "./request.ts";

const MOUNT_LIST = GEMMA_MOUNTS.join(", ");

/**
 * The model's text form → its key, or `undefined` for the empty default. Refuses text that is not
 * `<mount>/<model>` and a mount that is not one of the three.
 */
export const parseModelText = (text: string): ModelKey | undefined => {
  if (text === "") return undefined;
  const slash = text.indexOf("/");
  const mount = slash < 0 ? "" : text.slice(0, slash);
  const model = slash < 0 ? "" : text.slice(slash + 1);
  if (mount === "" || model === "") {
    throw new ParameterError(
      "model",
      `expected <mount>/<model> with a mount of ${MOUNT_LIST} (empty: the default), got "${text}"`,
    );
  }
  if (!isGemmaMount(mount)) {
    throw new ParameterError("model", `'${mount}' is not one of the mounts ${MOUNT_LIST}`);
  }
  return { mount, model };
};

/**
 * Refuses a model the manifest does not have (naming the models it has), and a model of the mobile
 * QAT mount its entry point does not accept (naming the ones it does).
 */
export const checkModel = (key: ModelKey, manifest: Manifest): void => {
  const models = Object.keys(manifest.models);
  if (!models.includes(key.model)) {
    throw new ParameterError(
      "model",
      `${key.mount} has no model '${key.model}'; its models are ${models.join(", ")}`,
    );
  }
  if (key.mount === "gemma4-qat-mobile" && !isQatModel(key.model)) {
    throw new ParameterError(
      "model",
      `gemma4QatMobile.Pipeline does not accept '${key.model}'; the models of ${key.mount} it accepts are ${models.filter(isQatModel).join(", ")}`,
    );
  }
};

/** The model a run loads, the distribution it is read from and its manifest. */
export type OpenedGemmaModel = {
  readonly key: ModelKey;
  readonly opened: OpenedDistribution;
  readonly manifest: Manifest;
};

/** `mount` opened, or `undefined` when its mirror is missing. */
const openPresent = async (
  openDistribution: OpenDistribution,
  mount: ModelKey["mount"],
  source: SourceKind,
): Promise<OpenedDistribution | undefined> => {
  try {
    return await openDistribution(mount, source);
  } catch (error) {
    // Every Gemma mount has HF-compatible coordinates, so a mount unavailable here has no mirror.
    if (error instanceof MountUnavailableError && error.mount === mount) return undefined;
    throw error;
  }
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
  for (const mount of GEMMA_MOUNTS) {
    const opened = await openPresent(openDistribution, mount, source);
    if (opened === undefined) continue;
    const { manifest } = await loadManifest(opened.ref, { signal });
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
      quant: presetOf(manifest, key.model, textValue(parameters, "preset")),
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
