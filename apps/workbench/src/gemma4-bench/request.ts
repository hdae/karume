/**
 * The `gemma4-bench` entry's parameters → what it runs: the arms its `arms` text names (or, for an
 * empty text, one per Gemma 4 mount whose mirror is present), resolved against the manifests of the
 * distributions they read, and the workload. Every value that does not fit is a `ParameterError`:
 * an arm's names the arm and says what exists or why.
 *
 * An arm is `<mount>/<model>[@<preset>][:<switch>]…`, and the arms are comma-separated. None of the
 * three Gemma 4 manifests' model or preset names (`e2b`, `e4b`, `12b`, `26b`; `bf16`, `i4`,
 * `i4-fast`, `i4-gemvpar`) holds `,`, `@` or `:`, and none of the three needs quoting in a shell or
 * escaping in a URL query (unlike `+`, which a query reads as a space).
 */

import { loadManifest } from "@karume/hub";
import { MountUnavailableError, type SourceKind } from "@karume/local-distributions/browser";

import {
  choicesValue,
  choiceValue,
  integerValue,
  type OpenDistribution,
  type OpenedDistribution,
  ParameterError,
  type ParameterValues,
} from "../entry.ts";
import {
  GEMMA_MOUNTS,
  type GemmaManifests,
  type GemmaMount,
  isGemmaMount,
} from "../gemma4/settings.ts";
import {
  type ArmDraft,
  BENCH_PROMPT_IDS,
  type BenchArm,
  type BenchSettings,
  type DecodeAhead,
  defaultArmDrafts,
  resolveArm,
  SAMPLER_KINDS,
} from "./plan.ts";

/** An arm's text form, as the refusals name it. */
const ARM_FORM = "<mount>/<model>[@<preset>][:<switch>]…";

/**
 * The switches of an arm, after its model and preset: `tap-off` (a wall-clock control; the tap is on
 * otherwise), `split` (one timed pass per dispatch), `integer` (the integer GEMV), `ple-gpu` (the PLE
 * on the GPU) and `ahead2` / `ahead4` / `ahead8` (the decode run-ahead depth; one step per batch
 * otherwise).
 */
const ARM_SWITCHES = [
  "tap-off",
  "split",
  "integer",
  "ple-gpu",
  "ahead2",
  "ahead4",
  "ahead8",
] as const;
type ArmSwitch = (typeof ARM_SWITCHES)[number];

const isArmSwitch = (text: string): text is ArmSwitch => ARM_SWITCHES.some((name) => name === text);

const AHEAD_DEPTHS: Readonly<Record<"ahead2" | "ahead4" | "ahead8", DecodeAhead>> = {
  ahead2: "2",
  ahead4: "4",
  ahead8: "8",
};

const MOUNT_LIST = GEMMA_MOUNTS.join(", ");

/** One arm of the `arms` text: as written (a refusal names it so) and as a draft. */
export type NamedArm = { readonly text: string; readonly draft: ArmDraft };

/** The refusal of the arm at 1-based `position`, written `text`. */
const armError = (position: number, text: string, reason: string): ParameterError =>
  new ParameterError("arms", `arm ${position} "${text}": ${reason}`);

/** One arm's text → its draft; text that does not fit the form is refused, naming the arm. */
const parseArm = (text: string, position: number): NamedArm => {
  const [head = "", ...switchTexts] = text.split(":");
  const parts = head.split("@");
  if (parts.length > 2) {
    throw armError(position, text, `names more than one preset; an arm is ${ARM_FORM}`);
  }
  const [keyText = "", quant] = parts;
  const slash = keyText.indexOf("/");
  const mount = slash < 0 ? "" : keyText.slice(0, slash);
  const model = slash < 0 ? "" : keyText.slice(slash + 1);
  if (mount === "" || model === "") {
    throw armError(position, text, `expected ${ARM_FORM} with a mount of ${MOUNT_LIST}`);
  }
  if (!isGemmaMount(mount)) {
    throw armError(position, text, `'${mount}' is not one of the mounts ${MOUNT_LIST}`);
  }
  if (quant === "") {
    throw armError(
      position,
      text,
      "names an empty preset (without '@' the arm runs the default preset)",
    );
  }
  let tap = true;
  let split = false;
  let integerGemv = false;
  let pleOnGpu = false;
  let decodeAhead: DecodeAhead | undefined;
  const seen = new Set<ArmSwitch>();
  for (const name of switchTexts) {
    if (!isArmSwitch(name)) {
      throw armError(
        position,
        text,
        `${name === "" ? "has an empty switch" : `'${name}' is not a switch`}; the switches are ${ARM_SWITCHES.join(", ")}`,
      );
    }
    if (seen.has(name)) throw armError(position, text, `names the switch '${name}' twice`);
    seen.add(name);
    switch (name) {
      case "tap-off":
        tap = false;
        break;
      case "split":
        split = true;
        break;
      case "integer":
        integerGemv = true;
        break;
      case "ple-gpu":
        pleOnGpu = true;
        break;
      default:
        if (decodeAhead !== undefined) {
          throw armError(position, text, "names more than one decode run-ahead depth");
        }
        decodeAhead = AHEAD_DEPTHS[name];
    }
  }
  // The split needs the tap (`resolveArm` drops it from a tap-off draft): text that asks for both
  // is refused rather than run without the split.
  if (split && !tap) {
    throw armError(
      position,
      text,
      "split times every dispatch through the tap, so it cannot go with tap-off",
    );
  }
  return {
    text,
    draft: {
      key: { mount, model },
      ...(quant === undefined ? {} : { quant }),
      tap,
      ...(split ? { split: true } : {}),
      ...(integerGemv ? { integerGemv: true } : {}),
      ...(pleOnGpu ? { pleOnGpu: true } : {}),
      ...(decodeAhead === undefined ? {} : { decodeAhead }),
    },
  };
};

/**
 * The `arms` text → its arms in order, before any distribution is read; empty (or blank) text is
 * none, which {@link openBenchArms} reads as the default arms. Each arm is trimmed; an empty arm and
 * one that does not fit {@link ARM_FORM} are refused, naming the arm.
 */
export const parseArmsText = (text: string): readonly NamedArm[] => {
  if (text.trim() === "") return [];
  return text.split(",").map((item, index) => {
    const arm = item.trim();
    if (arm === "") {
      throw new ParameterError(
        "arms",
        `arm ${index + 1} is empty; the arms are ${ARM_FORM}, comma-separated`,
      );
    }
    return parseArm(arm, index + 1);
  });
};

/** The arms a run makes, and the distribution of every mount they read (each opened once). */
export type OpenedBenchArms = {
  readonly arms: readonly BenchArm[];
  readonly distributions: Partial<Readonly<Record<GemmaMount, OpenedDistribution>>>;
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

/** Every arm resolved against the manifests (`resolveArm`), the first refused one named. */
const resolveArms = (named: readonly NamedArm[], manifests: GemmaManifests): readonly BenchArm[] =>
  named.map(({ text, draft }, index) => {
    const resolution = resolveArm(draft, manifests);
    if (!resolution.ok) throw armError(index + 1, text, resolution.error);
    return resolution.arm;
  });

/**
 * The arms `text` names, each mount they read opened (`source`: how a browser reads it) and its
 * manifest loaded, then every arm resolved against the manifests. A mount whose mirror is missing
 * is refused, naming the first arm that reads it. Empty text is the default arms: one per mount of
 * the three whose mirror is present, at its default model and preset, the tap on (refused when no
 * mirror is present).
 */
export const openBenchArms = async (
  text: string,
  source: SourceKind,
  openDistribution: OpenDistribution,
  signal: AbortSignal,
): Promise<OpenedBenchArms> => {
  const named = parseArmsText(text);
  const distributions: Partial<Record<GemmaMount, OpenedDistribution>> = {};
  const manifests: GemmaManifests = {};
  const load = async (mount: GemmaMount, opened: OpenedDistribution): Promise<void> => {
    distributions[mount] = opened;
    manifests[mount] = (await loadManifest(opened.ref, { signal })).manifest;
  };
  if (named.length === 0) {
    for (const mount of GEMMA_MOUNTS) {
      const opened = await openPresent(openDistribution, mount, source);
      if (opened !== undefined) await load(mount, opened);
    }
    const drafts = defaultArmDrafts(manifests);
    if (drafts.length === 0) {
      throw new ParameterError(
        "arms",
        `the default is one arm per mount of ${MOUNT_LIST} whose mirror is present, and none is`,
      );
    }
    const defaults = drafts.map((draft) => ({
      text: `${draft.key.mount}/${draft.key.model}`,
      draft,
    }));
    return { arms: resolveArms(defaults, manifests), distributions };
  }
  for (const [index, { text: armText, draft }] of named.entries()) {
    const { mount } = draft.key;
    if (distributions[mount] !== undefined) continue;
    let opened: OpenedDistribution;
    try {
      opened = await openDistribution(mount, source);
    } catch (error) {
      if (error instanceof MountUnavailableError && error.mount === mount) {
        throw armError(index + 1, armText, error.message);
      }
      throw error;
    }
    await load(mount, opened);
  }
  return { arms: resolveArms(named, manifests), distributions };
};

/** The workload the parameters ask for; no prompt is refused. */
export const benchSettingsOf = (parameters: ParameterValues): BenchSettings => {
  const prompts = choicesValue(parameters, "prompts", BENCH_PROMPT_IDS);
  if (prompts.length === 0) throw new ParameterError("prompts", "choose at least one prompt");
  return {
    chats: integerValue(parameters, "chats"),
    newTokens: integerValue(parameters, "newTokens"),
    rounds: integerValue(parameters, "rounds"),
    sampler: choiceValue(parameters, "sampler", SAMPLER_KINDS),
    prompts,
  };
};
