/**
 * The BiRefNet page's choices and what it knows before a run: the two checkpoints (one distribution
 * each), the model and preset the form settles on, how many bytes the selection reads, and whether
 * the GPU meets the limits the preset declares. Pure, so `vp test` covers it.
 */

import {
  type Manifest,
  type RequiredLimitName,
  type RequiredLimitsSpec,
  resolveSelection,
  selectionRefs,
} from "@karume/hub";

import type { MountName } from "@/mounts";

/** A checkpoint is a mount (and the key of `birefnet.SOURCES`): each is its own distribution. */
export type CheckpointId = Extract<MountName, "birefnet-hr" | "lucida">;

export type Checkpoint = {
  readonly id: CheckpointId;
  readonly label: string;
  /** What the fine-tune is for. */
  readonly note: string;
};

export const CHECKPOINTS: readonly Checkpoint[] = [
  { id: "birefnet-hr", label: "HR", note: "Photos: general subjects at high resolution." },
  { id: "lucida", label: "Lucida", note: "Illustrations, transparent objects and logos." },
];

export const isCheckpointId = (value: string): value is CheckpointId =>
  CHECKPOINTS.some(({ id }) => id === value);

/** The model (the graph's input size: `1024`, `2048`) and the preset (a quant of the manifest). */
export type Selection = { readonly model: string; readonly quant: string };

/**
 * The selection the form stands for once the manifest is known: an empty or unknown model becomes
 * `defaultModel`, an empty or unknown preset becomes that model's `defaultQuant`. A valid choice is
 * kept, so a choice survives a switch to the other checkpoint and back.
 */
export const settleSelection = (form: Selection, manifest: Manifest): Selection => {
  const model = Object.hasOwn(manifest.models, form.model) ? form.model : manifest.defaultModel;
  const entry = manifest.models[model];
  if (entry === undefined) throw new Error(`The manifest has no model '${model}'`);
  const quant = Object.hasOwn(entry.quants, form.quant) ? form.quant : entry.defaultQuant;
  return model === form.model && quant === form.quant ? form : { model, quant };
};

/**
 * The bytes a build of `selection` reads (the weight parts and the assets, each file once — the
 * hub's own fold of a selection). The same for both sources; the cached one reads them from the
 * browser's disk after the first run.
 */
export const selectionBytes = (manifest: Manifest, selection: Selection): number =>
  selectionRefs(resolveSelection(manifest, selection)).reduce((sum, { size }) => sum + size, 0);

/** One declared limit the device does not meet. */
export type LimitShortfall = {
  readonly name: RequiredLimitName;
  readonly required: number;
  readonly granted: number;
};

/**
 * The preset's `requiredLimits` against the device's limits. `undeclared`: the preset declares
 * none (an empty table counts as none). The declaration covers the resident part only (weights,
 * state), not the intermediates: a device that fits here can still fail when the Session is built.
 */
export type DeviceFit =
  | { readonly kind: "undeclared" }
  | { readonly kind: "fits"; readonly declared: readonly RequiredLimitName[] }
  | { readonly kind: "short"; readonly shortfalls: readonly LimitShortfall[] };

const isLimitName = (
  name: string,
  granted: Readonly<Record<RequiredLimitName, number>>,
): name is RequiredLimitName => Object.hasOwn(granted, name);

export const deviceFit = (
  spec: RequiredLimitsSpec | undefined,
  granted: Readonly<Record<RequiredLimitName, number>>,
): DeviceFit => {
  const declared: RequiredLimitName[] = [];
  const shortfalls: LimitShortfall[] = [];
  for (const [name, required] of Object.entries(spec ?? {})) {
    // The hub rejects unknown names when it parses the manifest; a name the device does not report
    // is a drift between hub and runtime, not a fit.
    if (!isLimitName(name, granted)) {
      throw new Error(`The device reports no limit '${name}' that the preset declares`);
    }
    if (required === undefined) continue;
    declared.push(name);
    if (granted[name] < required) shortfalls.push({ name, required, granted: granted[name] });
  }
  if (shortfalls.length > 0) return { kind: "short", shortfalls };
  return declared.length === 0 ? { kind: "undeclared" } : { kind: "fits", declared };
};

const MIB = 2 ** 20;

/** Bytes in MiB with one decimal (`452.1 MiB`). */
export const mebibytes = (bytes: number): string => `${(bytes / MIB).toFixed(1)} MiB`;
