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

import type { GemmaManifests, GemmaMount } from "@karume/gemma4-app-support/settings";
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
import { MOUNT_LIST, parseModelKey, presentMounts } from "../gemma4/open.ts";
import {
  type ArmDraft,
  BENCH_PROMPT_IDS,
  type BenchArm,
  type BenchSettings,
  type DecodeAhead,
  defaultArmDrafts,
  type FirstChunk,
  resolveArm,
  SAMPLER_KINDS,
} from "./plan.ts";

/** An arm's text form, as the refusals name it. */
const ARM_FORM = "<mount>/<model>[@<preset>][:<switch>]…";

/**
 * The switches of an arm, after its model and preset: `tap-off` (a wall-clock control; the tap is on
 * otherwise), `split` (one timed pass per dispatch), `host` (the tap's host mode: wall-clock marks at
 * the WebGPU calls instead of pass timing), `linear-reorder` (linear at the level `reorder` on a
 * preset that does not declare it), `integer` (the integer GEMV), `ple-gpu` (the PLE on the GPU),
 * `ahead2` / `ahead4` / `ahead8` (the decode run-ahead depth; one step per batch otherwise),
 * `no-signal` (the chats are sent without an `AbortSignal`), `sampled` (the distribution's sampler
 * with seed 0, whatever the bench's `sampler` is) and `first8` / `first16` / `first32` / `first64` /
 * `first-off` (the first chunk of a run's submits; the pipeline's own submit policy otherwise).
 */
const ARM_SWITCHES = [
  "tap-off",
  "split",
  "host",
  "linear-reorder",
  "integer",
  "ple-gpu",
  "ahead2",
  "ahead4",
  "ahead8",
  "no-signal",
  "sampled",
  "first8",
  "first16",
  "first32",
  "first64",
  "first-off",
] as const;
type ArmSwitch = (typeof ARM_SWITCHES)[number];

const isArmSwitch = (text: string): text is ArmSwitch => ARM_SWITCHES.some((name) => name === text);

const AHEAD_DEPTHS: Readonly<Record<"ahead2" | "ahead4" | "ahead8", DecodeAhead>> = {
  ahead2: "2",
  ahead4: "4",
  ahead8: "8",
};

const FIRST_CHUNKS: Readonly<
  Record<"first8" | "first16" | "first32" | "first64" | "first-off", FirstChunk>
> = {
  first8: "8",
  first16: "16",
  first32: "32",
  first64: "64",
  "first-off": "off",
};

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
  const key = parseModelKey(keyText, {
    form: `expected ${ARM_FORM} with a mount of ${MOUNT_LIST}`,
    refuse: (reason) => armError(position, text, reason),
  });
  if (quant === "") {
    throw armError(
      position,
      text,
      "names an empty preset (without '@' the arm runs the default preset)",
    );
  }
  let tap = true;
  let split = false;
  let linearReorder = false;
  let hostMarks = false;
  let noSignal = false;
  let sampled = false;
  let integerGemv = false;
  let pleOnGpu = false;
  let decodeAhead: DecodeAhead | undefined;
  let firstChunk: FirstChunk | undefined;
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
      case "linear-reorder":
        linearReorder = true;
        break;
      case "host":
        hostMarks = true;
        break;
      case "no-signal":
        noSignal = true;
        break;
      case "sampled":
        sampled = true;
        break;
      case "integer":
        integerGemv = true;
        break;
      case "ple-gpu":
        pleOnGpu = true;
        break;
      case "first8":
      case "first16":
      case "first32":
      case "first64":
      case "first-off":
        if (firstChunk !== undefined) {
          throw armError(position, text, "names more than one first chunk");
        }
        firstChunk = FIRST_CHUNKS[name];
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
  // The host mode is a mode of the tap, and one that leaves the passes alone: neither without the
  // tap nor together with the split.
  if (hostMarks && !tap) {
    throw armError(
      position,
      text,
      "host marks the WebGPU calls through the tap, so it cannot go with tap-off",
    );
  }
  if (hostMarks && split) {
    throw armError(
      position,
      text,
      "host leaves the passes as the product encodes them, so it cannot go with split",
    );
  }
  return {
    text,
    draft: {
      key,
      ...(quant === undefined ? {} : { quant }),
      tap,
      ...(split ? { split: true } : {}),
      ...(linearReorder ? { linearReorder: true } : {}),
      ...(hostMarks ? { hostMarks: true } : {}),
      ...(noSignal ? { noSignal: true } : {}),
      ...(sampled ? { sampled: true } : {}),
      ...(integerGemv ? { integerGemv: true } : {}),
      ...(pleOnGpu ? { pleOnGpu: true } : {}),
      ...(decodeAhead === undefined ? {} : { decodeAhead }),
      ...(firstChunk === undefined ? {} : { firstChunk }),
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

/** An arm with the distribution of its mount, as it was opened (once per mount). */
export type OpenedBenchArm = BenchArm & { readonly distribution: OpenedDistribution };

/**
 * Every arm resolved against the manifests (`resolveArm`), the first refused one named, with its
 * mount's distribution. `resolveArm` refuses an arm whose manifest is not loaded, and a manifest is
 * loaded only together with its distribution.
 */
const resolveArms = (
  named: readonly NamedArm[],
  manifests: GemmaManifests,
  distributions: ReadonlyMap<GemmaMount, OpenedDistribution>,
): readonly OpenedBenchArm[] =>
  named.map(({ text, draft }, index) => {
    const resolution = resolveArm(draft, manifests);
    if (!resolution.ok) throw armError(index + 1, text, resolution.error);
    const { arm } = resolution;
    const distribution = distributions.get(arm.mount);
    if (distribution === undefined) throw new Error(`No distribution opened for '${arm.mount}'`);
    return { ...arm, distribution };
  });

/**
 * The arms `text` names, each mount they read opened (`source`: how a browser reads it) and its
 * manifest loaded, then every arm resolved against the manifests and given its mount's
 * distribution (the arms of one mount share it). A mount whose mirror is missing
 * is refused, naming the first arm that reads it. Empty text is the default arms: one per mount of
 * the three whose mirror is present, at its default model and preset, the tap on (refused when no
 * mirror is present).
 */
export const openBenchArms = async (
  text: string,
  source: SourceKind,
  openDistribution: OpenDistribution,
  signal: AbortSignal,
): Promise<readonly OpenedBenchArm[]> => {
  const named = parseArmsText(text);
  const distributions = new Map<GemmaMount, OpenedDistribution>();
  const manifests: GemmaManifests = {};
  if (named.length === 0) {
    for await (const { mount, opened, manifest } of presentMounts(
      openDistribution,
      source,
      signal,
    )) {
      distributions.set(mount, opened);
      manifests[mount] = manifest;
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
    return resolveArms(defaults, manifests, distributions);
  }
  for (const [index, { text: armText, draft }] of named.entries()) {
    const { mount } = draft.key;
    if (distributions.has(mount)) continue;
    let opened: OpenedDistribution;
    try {
      opened = await openDistribution(mount, source);
    } catch (error) {
      if (error instanceof MountUnavailableError && error.mount === mount) {
        throw armError(index + 1, armText, error.message);
      }
      throw error;
    }
    distributions.set(mount, opened);
    manifests[mount] = (await loadManifest(opened.ref, { signal })).manifest;
  }
  return resolveArms(named, manifests, distributions);
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
