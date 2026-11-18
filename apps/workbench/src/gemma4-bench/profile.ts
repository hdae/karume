/**
 * The geometry profile of a bench arm that carries profile switches (`PROFILE_SWITCHES` of
 * `./plan.ts`), and the GEMV choices an arm's devices run with — what the record states per arm.
 *
 * A device takes a profile through the runtime's public `acquireGpu({ geometryProfile })`, here a
 * callback that gets the adapter's information. Building that profile needs two things the
 * runtime's public surface does not export as values: the table an adapter selects and the default
 * profile's GEMV choices (a table without the field runs with them). They are the runtime's own
 * (`selectGeometryProfile`, `resolveGeometryProfile`), imported from its source as the repository's
 * tools import them, so the bench cannot disagree with the runtime on either. This module is the
 * one place the workbench does that; the bench is an entry that reads a local distribution, which a
 * production build leaves out.
 *
 * Pure: `vp test` covers it without a GPU.
 *
 * SPEC: docs/spec/workbench.md#the-decode-benchs-profile-switches
 */

import type { GeometryProfile } from "@karume/runtime";

import {
  resolveGeometryProfile,
  selectGeometryProfile,
} from "../../../../packages/runtime/src/kernels/geometry-profile.ts";
import type { ProfileSwitch } from "./plan.ts";

/** What selects a table: the adapter's vendor, architecture and description. */
export type BenchAdapter = Pick<GPUAdapterInfo, "vendor" | "architecture" | "description">;

/** The GEMV choices of linear a profile holds (`GeometryProfile.linearGemv`). */
export type GemvChoices = NonNullable<GeometryProfile["linearGemv"]>;

/** The GEMV choices an arm's devices ran with, as the bench's record states them. */
export type ArmGemvChoices = {
  /** The arm's profile switches, in the order they applied (`[]` = the profile the device resolves to). */
  readonly switches: readonly ProfileSwitch[];
  /** The id of the table the adapter selects (`default` where no embedded table matches). */
  readonly table: string;
  /** Whether that table carries GEMV choices of its own (false: it runs with the default profile's). */
  readonly tableCarriesGemv: boolean;
  /** The choices the devices ran with, the switches applied. */
  readonly linearGemv: GemvChoices;
};

/** The table `adapter` selects, the switches applied in order. */
const switched = (adapter: BenchAdapter, switches: readonly ProfileSwitch[]): GeometryProfile =>
  switches.reduce<GeometryProfile>((profile, name) => {
    switch (name) {
      case "default-gemv": {
        // Without the field a table runs with the default profile's choices
        const { linearGemv: _own, ...rest } = profile;
        return rest;
      }
      case "general-sequential": {
        const gemv = resolveGeometryProfile(profile).linearGemv;
        return {
          ...profile,
          linearGemv: { ...gemv, sequential: { variant: gemv.sequential.variant, shapes: [] } },
        };
      }
    }
  }, selectGeometryProfile(adapter));

/**
 * The profile to acquire a device of `adapter` with for an arm with `switches`: `undefined` without
 * a switch (the runtime selects the table itself), otherwise the table the adapter selects with the
 * switches applied. The tile geometry and every other field are the selected table's.
 */
export const armGeometryProfile = (
  adapter: BenchAdapter,
  switches: readonly ProfileSwitch[],
): GeometryProfile | undefined => (switches.length === 0 ? undefined : switched(adapter, switches));

/** The GEMV choices a device of `adapter` runs with for an arm with `switches`. */
export const armGemvChoices = (
  adapter: BenchAdapter,
  switches: readonly ProfileSwitch[],
): ArmGemvChoices => {
  const table = selectGeometryProfile(adapter);
  return {
    switches,
    table: table.id,
    tableCarriesGemv: table.linearGemv !== undefined,
    linearGemv: resolveGeometryProfile(switched(adapter, switches)).linearGemv,
  };
};

/**
 * One summary line for an arm's choices: the table, where the choices come from, and their size.
 * `label` is the arm as the other summary lines name it.
 */
export const gemvChoicesLine = (label: string, choices: ArmGemvChoices): string => {
  const { sequential, parallelShapes } = choices.linearGemv;
  const source =
    choices.switches.includes("default-gemv") || !choices.tableCarriesGemv
      ? "the default profile's GEMV choices"
      : "the table's own GEMV choices";
  return `${label} · table ${choices.table} · ${source}${
    choices.switches.includes("general-sequential") ? ", the sequential exceptions removed" : ""
  } · sequential c${sequential.variant.cols}u${sequential.variant.unroll} with ${sequential.shapes.length} per-shape exceptions · ${parallelShapes.length} parallel rows`;
};
