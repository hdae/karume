/**
 * What the entries that read a local distribution (`src/entries/local/`) declare alike: how a
 * browser reads the distribution (`source`), and for the Gemma entries whether the device takes the
 * subgroup features (`subgroups`) with the acquisition that parameter decides.
 */

import type { SourceKind } from "@karume/local-distributions/browser";
import { acquireGpu, type GpuContext } from "@karume/runtime";

import type { BooleanParameter, ChoiceParameter, GpuOptions } from "./entry.ts";

/**
 * How a browser reads the distribution (`openDistribution`'s `source`): `cached` first, the default —
 * an A/B reads the weights again for every pipeline or Session it builds, and the direct route would
 * fetch them anew each time. Under Node the mirror's directory is read either way.
 */
export const SOURCE_KINDS = ["cached", "direct"] as const satisfies readonly SourceKind[];

/** The `source` parameter; `label` is the form's text (an entry may read one distribution or several). */
export const sourceParameter = (label: string): ChoiceParameter => ({
  kind: "choice",
  name: "source",
  label,
  options: SOURCE_KINDS,
  default: "cached",
});

/** The `subgroups` parameter of the Gemma entries ({@link acquireWithSubgroups}). */
export const SUBGROUPS_PARAMETER: BooleanParameter = {
  kind: "boolean",
  name: "subgroups",
  label:
    "Acquire the device with the subgroup features when it has them (off: a device without them)",
  default: true,
};

/**
 * The device the `subgroups` parameter asks for. With the subgroup features the device can run the
 * 32-lane kernel forms; without them the arms run on the device a plain `acquireGpu()` gives. Which
 * one fits can differ: the 26B model with 64 expert slots builds a second pipeline on a 12 GB GPU
 * only without them (docs/known-issues.md).
 */
export const acquireWithSubgroups = (
  gpuOptions: GpuOptions,
  subgroups: boolean,
): Promise<GpuContext> =>
  acquireGpu({ ...gpuOptions, ...(subgroups ? { subgroups: "available" as const } : {}) });
