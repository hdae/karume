/**
 * The host side of the MOSS-Image DiT's time input — the `dit` container's asset `training_grid`
 * and the lookup of the graph input `timesteps_proj [1,256]` by training-grid index (ADR 0138
 * decision 11, stage 2).
 *
 * ## Why a table, not a formula
 *
 * Upstream's sampler snaps every σ to the nearest point of its 1,000-point training grid and passes
 * the timestep σ × 1000, so the DiT only ever sees one of 1,000 timesteps. Its forward computes the
 * sinusoid in float64 and casts it to f32; transcribing that in TS could split the last bit (the
 * same judgement as the RoPE base tables — ADR 0034 decision 3). The asset holds upstream's own
 * values — the training grid of `MOSSImageScheduler.set_timesteps` and, per grid point, the input
 * upstream's forward feeds its time MLP — and the host only indexes it.
 *
 * ## The asset (safetensors, role `training-grid`)
 *
 * - `sigmas` F32 `[1000]` and `timesteps` F32 `[1000]`: the training grid in upstream's descending
 *   order (index 0 is σ = 1, timestep 1000);
 * - `time_sinusoid` F32 `[1000, freq_dim]`: row i is `timesteps_proj` at `timesteps[i]`.
 *
 * The recipe's write checks the three tables bit for bit against the stage-0 host record
 * (`tools/export-recipes/moss/export_dit.py`, gate 3).
 *
 * MUST: every module is side-effect free (no import-time execution, no global mutable state).
 */

import { parseSafetensors, type SafetensorsFile } from "@karume/runtime";

/** The training grid's tables. */
export type MossTrainingGrid = {
  /** The number of grid points (1,000). */
  readonly points: number;
  /** The width of one `timesteps_proj` row (`freq_dim` — 256). */
  readonly width: number;
  /** σ per grid index (descending). */
  readonly sigmas: Float32Array<ArrayBuffer>;
  /** The timestep per grid index (σ × 1000 in f32, as upstream computes it). */
  readonly timesteps: Float32Array<ArrayBuffer>;
  /** `[points, width]` row-major. */
  readonly sinusoid: Float32Array<ArrayBuffer>;
};

const floatsOf = (
  file: SafetensorsFile,
  name: string,
  rank: number,
): { readonly data: Float32Array<ArrayBuffer>; readonly shape: readonly number[] } => {
  const view = file.tensors.get(name);
  if (view === undefined) throw new Error(`The training grid has no '${name}'`);
  if (view.dtype !== "F32") {
    throw new Error(`Training grid '${name}': storage dtype is ${view.dtype} (F32 is required)`);
  }
  if (view.shape.length !== rank) {
    throw new Error(`Training grid '${name}': rank ${view.shape.length} (${rank} expected)`);
  }
  return {
    data: new Float32Array(file.buffer, view.byteOffset, view.byteLength / 4),
    shape: view.shape,
  };
};

/**
 * Reads the payload (safetensors) of the asset `training_grid`.
 *
 * MUST: exactly the three tables, one point count for all of them, and σ strictly descending from
 * 1 (a reversed or shuffled table would index other timesteps silently).
 */
export const parseMossTrainingGrid = (buffer: ArrayBuffer): MossTrainingGrid => {
  const file = parseSafetensors(buffer);
  const names = [...file.tensors.keys()].sort();
  if (names.join(",") !== "sigmas,time_sinusoid,timesteps") {
    throw new Error(
      `The training grid holds [${names.join(", ")}] (sigmas, time_sinusoid, timesteps)`,
    );
  }
  const sigmas = floatsOf(file, "sigmas", 1);
  const timesteps = floatsOf(file, "timesteps", 1);
  const sinusoid = floatsOf(file, "time_sinusoid", 2);
  const points = sigmas.shape[0];
  if (timesteps.shape[0] !== points || sinusoid.shape[0] !== points) {
    throw new Error(
      `Training grid: ${points} sigmas, ${timesteps.shape[0]} timesteps, ${sinusoid.shape[0]} sinusoid rows`,
    );
  }
  if (
    sigmas.data[0] !== 1 ||
    sigmas.data.some((sigma, at) => at > 0 && sigma >= sigmas.data[at - 1])
  ) {
    throw new Error("Training grid: σ is not strictly descending from 1");
  }
  return {
    points,
    width: sinusoid.shape[1],
    sigmas: sigmas.data,
    timesteps: timesteps.data,
    sinusoid: sinusoid.data,
  };
};

/** The training-grid index (an integer in `0 … points − 1`, fails loudly otherwise). */
const assertGridIndex = (grid: MossTrainingGrid, index: number): void => {
  if (!Number.isInteger(index) || index < 0 || index >= grid.points) {
    throw new Error(`Training-grid index ${index} outside 0 … ${grid.points - 1}`);
  }
};

/** `timesteps_proj` (`[width]` — the graph input `[1, width]`) at training-grid index `index`. */
export const mossTimestepsProj = (
  grid: MossTrainingGrid,
  index: number,
): Float32Array<ArrayBuffer> => {
  assertGridIndex(grid, index);
  return grid.sinusoid.slice(index * grid.width, (index + 1) * grid.width);
};

/** The timestep upstream passes at training-grid index `index` (σ × 1000 in f32). */
export const mossTimestep = (grid: MossTrainingGrid, index: number): number => {
  assertGridIndex(grid, index);
  return grid.timesteps[index];
};
