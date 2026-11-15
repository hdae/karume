/**
 * What a sha256 reference value was taken from (`outputs/references/`, untracked).
 *
 * A reference row holds only a sha256. When an output stops matching, the row cannot say what the
 * matching output looked like, nor whether the kernels have changed since — the two things a person
 * needs to judge the mismatch. This store keeps both, keyed by the reference value:
 *
 * - `<sha256><extension>` — the artifact of a run that matched the row (or created it);
 * - `<sha256>.json` — the case, the environment, and the WGSL texts the case dispatched on the last
 *   run that matched ({@link KernelUse}).
 *
 * It is evidence for a diagnostic, never an input to the verdict: a mismatch fails whatever the store
 * holds (ADR 0152 decision 4 turns a mismatch with a changed kernel set into a warning only for presets
 * that declare a requirement level, which do not exist yet). Deleting the directory loses the evidence
 * and nothing else; the next matching run writes it again.
 */

import { copyFileSync } from "node:fs";
import {
  isNotFound,
  mkdirSync,
  readFileSync,
  readTextFileSync,
  statSync,
  writeTextFileSync,
} from "@karume/test-support/fs";
import type { KernelUse } from "@karume/test-support/kernel-record";

/** Root of the store (`outputs/references/`). */
const DEFAULT_ROOT = new URL("../../../../outputs/references/", import.meta.url);

const SCHEMA = 1;

/** Above this size the two artifacts of a mismatch are not read to count differing bytes. */
const BYTE_COMPARISON_LIMIT = 256 * 1024 * 1024;

/** How many kernel labels a diagnostic names before it says "and N more". */
const NAMED_KERNELS = 8;

/** The record kept next to an artifact. */
export type KeptReference = {
  readonly schema: number;
  readonly sha256: string;
  readonly caseId: string;
  readonly environment: string;
  /** File name of the artifact inside the store. */
  readonly artifact: string;
  /** The WGSL texts the case dispatched on the last run that matched this value. */
  readonly kernels: readonly KernelUse[];
};

export type ReferenceStore = {
  /**
   * Keeps the artifact of a run whose output has the reference value `sha256`, and the kernels of
   * that run. The artifact is copied only when the store has none (same value, same content); the
   * record is rewritten every time, so its kernels are those of the latest matching run.
   */
  keep(entry: {
    readonly sha256: string;
    readonly caseId: string;
    readonly environment: string;
    readonly artifactUrl: URL;
    readonly kernels: readonly KernelUse[];
  }): void;
  /** The record of a reference value and the location of its artifact (`undefined` when not kept). */
  read(sha256: string): { readonly record: KeptReference; readonly artifactUrl: URL } | undefined;
};

const extensionOf = (url: URL): string => {
  const name = url.pathname.slice(url.pathname.lastIndexOf("/") + 1);
  const dot = name.lastIndexOf(".");
  return dot <= 0 ? "" : name.slice(dot);
};

const isKernelUse = (value: unknown): value is KernelUse =>
  typeof value === "object" &&
  value !== null &&
  typeof Reflect.get(value, "label") === "string" &&
  typeof Reflect.get(value, "wgslSha256") === "string";

/** Reads a record. A record of another schema or shape throws (never read as "not kept"). */
const parseRecord = (text: string, url: URL): KeptReference => {
  const parsed: unknown = JSON.parse(text);
  if (typeof parsed !== "object" || parsed === null) {
    throw new Error(`the kept reference record is not an object: ${url.pathname}`);
  }
  const field = (name: string): unknown => Reflect.get(parsed, name);
  const kernels = field("kernels");
  if (
    field("schema") !== SCHEMA ||
    typeof field("sha256") !== "string" ||
    typeof field("caseId") !== "string" ||
    typeof field("environment") !== "string" ||
    typeof field("artifact") !== "string" ||
    !Array.isArray(kernels) ||
    !kernels.every(isKernelUse)
  ) {
    throw new Error(
      `the kept reference record has an unexpected shape (schema ${SCHEMA}): ${url.pathname}`,
    );
  }
  return parsed as KeptReference;
};

/** Opens the store. `root` is injected by unit tests; runtime callers omit it. */
export const openReferenceStore = (root: URL = DEFAULT_ROOT): ReferenceStore => {
  const recordUrl = (sha256: string): URL => new URL(`${sha256}.json`, root);
  const exists = (url: URL): boolean => {
    try {
      statSync(url);
      return true;
    } catch (cause) {
      if (isNotFound(cause)) return false;
      throw cause;
    }
  };
  return {
    keep: ({ sha256, caseId, environment, artifactUrl, kernels }): void => {
      mkdirSync(root, { recursive: true });
      const artifact = `${sha256}${extensionOf(artifactUrl)}`;
      const keptUrl = new URL(artifact, root);
      if (!exists(keptUrl)) copyFileSync(artifactUrl, keptUrl);
      const record: KeptReference = {
        schema: SCHEMA,
        sha256,
        caseId,
        environment,
        artifact,
        kernels,
      };
      writeTextFileSync(recordUrl(sha256), `${JSON.stringify(record, undefined, 2)}\n`);
    },
    read: (sha256) => {
      const url = recordUrl(sha256);
      let text: string;
      try {
        text = readTextFileSync(url);
      } catch (cause) {
        if (isNotFound(cause)) return undefined;
        throw cause;
      }
      const record = parseRecord(text, url);
      return { record, artifactUrl: new URL(record.artifact, root) };
    },
  };
};

/** How the kernels of this run differ from the kept ones, by label. */
export type KernelChanges = {
  /** Labels whose WGSL text differs. */
  readonly changed: readonly string[];
  /** Labels this run dispatched and the kept run did not. */
  readonly added: readonly string[];
  /** Labels the kept run dispatched and this run did not. */
  readonly removed: readonly string[];
};

/**
 * Compares two kernel sets by label (pure function).
 *
 * A label present on both sides with any text on one side missing from the other counts as changed;
 * a label on one side only counts as added or removed.
 */
export const kernelChanges = (
  kept: readonly KernelUse[],
  current: readonly KernelUse[],
): KernelChanges => {
  const textsByLabel = (kernels: readonly KernelUse[]): Map<string, Set<string>> => {
    const byLabel = new Map<string, Set<string>>();
    for (const { label, wgslSha256 } of kernels) {
      const texts = byLabel.get(label) ?? new Set<string>();
      texts.add(wgslSha256);
      byLabel.set(label, texts);
    }
    return byLabel;
  };
  const before = textsByLabel(kept);
  const after = textsByLabel(current);
  const changed: string[] = [];
  const added: string[] = [];
  const removed: string[] = [];
  for (const [label, texts] of after) {
    const previous = before.get(label);
    if (previous === undefined) added.push(label);
    else if (previous.size !== texts.size || [...texts].some((text) => !previous.has(text))) {
      changed.push(label);
    }
  }
  for (const label of before.keys()) {
    if (!after.has(label)) removed.push(label);
  }
  return { changed: changed.sort(), added: added.sort(), removed: removed.sort() };
};

const named = (labels: readonly string[]): string =>
  labels.length <= NAMED_KERNELS
    ? labels.join(", ")
    : `${labels.slice(0, NAMED_KERNELS).join(", ")} and ${labels.length - NAMED_KERNELS} more`;

/** One line on whether the kernels changed since the reference was kept. */
const kernelLine = (kept: readonly KernelUse[], current: readonly KernelUse[]): string => {
  const { changed, added, removed } = kernelChanges(kept, current);
  if (changed.length === 0 && added.length === 0 && removed.length === 0) {
    return (
      `kernels: the same ${current.length} WGSL texts as on the last matching run — the difference ` +
      "comes from outside the WGSL (model code, weights, host-side numerics) or the output is not deterministic"
    );
  }
  const parts = [
    ...(changed.length === 0
      ? []
      : [`${changed.length} with a different WGSL text (${named(changed)})`]),
    ...(added.length === 0 ? [] : [`${added.length} new (${named(added)})`]),
    ...(removed.length === 0 ? [] : [`${removed.length} no longer dispatched (${named(removed)})`]),
  ];
  return `kernels: changed since the last matching run — ${parts.join("; ")}`;
};

/** One line comparing the bytes of the kept artifact and this run's. */
const byteLine = (keptUrl: URL, actualUrl: URL): string => {
  let keptSize: number;
  try {
    keptSize = statSync(keptUrl).size;
  } catch (cause) {
    if (isNotFound(cause)) return `kept artifact: missing from the store (${keptUrl.pathname})`;
    throw cause;
  }
  const actualSize = statSync(actualUrl).size;
  const where = `kept artifact ${keptUrl.pathname}`;
  if (keptSize !== actualSize) {
    return `${where}: ${keptSize} bytes against this run's ${actualSize}`;
  }
  if (keptSize > BYTE_COMPARISON_LIMIT) {
    return `${where}: same size (${keptSize} bytes); too large to count the differing bytes here`;
  }
  const kept = readFileSync(keptUrl);
  const actual = readFileSync(actualUrl);
  let differing = 0;
  let first = -1;
  for (let index = 0; index < kept.length; index += 1) {
    if (kept[index] === actual[index]) continue;
    if (first < 0) first = index;
    differing += 1;
  }
  return differing === 0
    ? `${where}: the files are byte-identical (the reference value is taken from something other than the file's bytes)`
    : `${where}: ${differing} of ${keptSize} bytes differ, the first at offset ${first}`;
};

/**
 * The evidence lines of a mismatch: what the store kept for the expected value, compared with this
 * run (its kernels, and its artifact when one was written).
 */
export const mismatchEvidence = (
  store: ReferenceStore,
  expected: string,
  current: { readonly kernels: readonly KernelUse[]; readonly artifactUrl?: URL | undefined },
): string[] => {
  const kept = store.read(expected);
  if (kept === undefined) {
    return [
      "nothing is kept for the reference value (the row predates the store, or outputs/references was " +
        "cleared): whether the kernels changed cannot be told — check out the commit that last matched and run the case once to keep it",
    ];
  }
  return [
    kernelLine(kept.record.kernels, current.kernels),
    ...(current.artifactUrl === undefined ? [] : [byteLine(kept.artifactUrl, current.artifactUrl)]),
  ];
};
