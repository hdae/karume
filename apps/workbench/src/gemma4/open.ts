/**
 * Opening a Gemma 4 model for the workbench's Gemma entries: the `<mount>/<model>` text form, the
 * mounts whose mirror is present, and why a model of a mount cannot run. Each entry words its own
 * refusals (the session A/B names the `model` parameter, the decode bench names the arm).
 */

import {
  GEMMA_MOUNTS,
  type GemmaMount,
  isGemmaMount,
  isQatModel,
  type ModelKey,
} from "@karume/gemma4-app-support/settings";
import { loadManifest, type Manifest } from "@karume/hub";
import { MountUnavailableError, type SourceKind } from "@karume/local-distributions/browser";

import type { OpenDistribution, OpenedDistribution } from "../entry.ts";

/** The three mounts as a refusal lists them. */
export const MOUNT_LIST = GEMMA_MOUNTS.join(", ");

/**
 * `<mount>/<model>` → its key. Text without a mount or a model is refused with `form`, a mount that
 * is not one of the three with a reason that names them; `refuse` makes the caller's error of either.
 */
export const parseModelKey = (
  text: string,
  refusal: { readonly form: string; readonly refuse: (reason: string) => Error },
): ModelKey => {
  const slash = text.indexOf("/");
  const mount = slash < 0 ? "" : text.slice(0, slash);
  const model = slash < 0 ? "" : text.slice(slash + 1);
  if (mount === "" || model === "") throw refusal.refuse(refusal.form);
  if (!isGemmaMount(mount)) {
    throw refusal.refuse(`'${mount}' is not one of the mounts ${MOUNT_LIST}`);
  }
  return { mount, model };
};

/** `mount` opened, or `undefined` when its mirror is missing. */
const openPresent = async (
  openDistribution: OpenDistribution,
  mount: GemmaMount,
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

/** A mount whose mirror is present, opened, and its manifest. */
export type PresentMount = {
  readonly mount: GemmaMount;
  readonly opened: OpenedDistribution;
  readonly manifest: Manifest;
};

/**
 * The mounts whose mirror is present, in the order of the three: each is opened and its manifest
 * loaded when the caller reaches it, so a caller that stops at the first opens no mount after it.
 */
export async function* presentMounts(
  openDistribution: OpenDistribution,
  source: SourceKind,
  signal: AbortSignal,
): AsyncGenerator<PresentMount> {
  for (const mount of GEMMA_MOUNTS) {
    const opened = await openPresent(openDistribution, mount, source);
    if (opened === undefined) continue;
    const { manifest } = await loadManifest(opened.ref, { signal });
    yield { mount, opened, manifest };
  }
}

/** Why a model of a mount cannot run. */
export type ModelRefusal =
  /** The manifest has no such model; `models` are the ones it has. */
  | { readonly kind: "unknown"; readonly models: readonly string[] }
  /** A model of the mobile QAT mount its entry point does not accept; `accepted` are the ones it does. */
  | { readonly kind: "not-mobile"; readonly accepted: readonly string[] };

/** Why `key` cannot run on `manifest`, or `undefined` when it can. */
export const modelRefusal = (key: ModelKey, manifest: Manifest): ModelRefusal | undefined => {
  const models = Object.keys(manifest.models);
  if (!models.includes(key.model)) return { kind: "unknown", models };
  if (key.mount === "gemma4-qat-mobile" && !isQatModel(key.model)) {
    return { kind: "not-mobile", accepted: models.filter(isQatModel) };
  }
  return undefined;
};
