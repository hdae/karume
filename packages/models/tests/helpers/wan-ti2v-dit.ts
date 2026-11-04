/**
 * What the tests of Wan2.2 TI2V-5B's DiT share (ADR 0121 stages 1 / 2): the int8 transformer of the distribution
 * mirror `models/karume-wan2.2/` (opened through its manifest — ADR 0139 decision 1), the goldens of the series
 * `wan2.2-ti2v-5b-i8-dyn` (in the test-asset root — decision 2), the table of golden cases, the golden reader and the
 * host-side condition mask.
 *
 * The GPU-free cross-check (`e2e_wan_ti2v_dit_host_test.ts`) and the real-GPU r check (`e2e_wan_ti2v_dit_test.ts`) read
 * the same table of cases — growing the table in 2 places would let a case added to only one side silently escape the
 * other side's check.
 *
 * The source of truth for the cases is `CASES` (S = 192 — stage 1's `write`) and `FULL_CASES` (full size — stage 2's
 * `write-full`) of `tools/export-recipes/wan/ti2v_export_dit.py`. **Fixed here rather than enumerated** (so that tests
 * do not silently vanish in an environment where only part of the generation was run).
 *
 * ## The host-side condition mask ({@link conditionMask})
 *
 * The T2V condition inputs (an all-false condition mask and a condition-side timestep equal to the generation side's)
 * are built by `src/wan/dit-loop.ts` (ADR 0121 stage 6). `src/wan/` has no function that builds the I2V condition mask
 * yet (stage 9), so this file holds the I2V form too. The convention is the same as the recipe's
 * `dit_patch.dit_condition_mask`: for I2V the P = H'·W' tokens of the first latent frame are 1 (bool is u32 0 / 1), for
 * T2V everything is 0. The host test pins bit equality with the golden's `input.condition_mask`.
 */

import { isNotFound, statSync } from "@karume/test-support/fs";
import { assertEquals } from "@std/assert";
import { type OpenedContainer, parseSafetensors, type SafetensorsFile } from "@karume/runtime";
import { goldenDir } from "@karume/test-support/test-assets";
import type { WanPatchGeometry, WanTokenGrid } from "../../src/wan/dit-tokens.ts";
import { parseWanRopeBase, type WanRopeBase } from "../../src/wan/dit-rope.ts";
import { mirrorPresent } from "./mirror.ts";
import { readBuffer, viewOf } from "./safetensors.ts";
import { openWanTi2vComponent, WAN_TI2V_DIST_ROOT } from "./wan-ti2v-pipeline.ts";

/** The series the goldens were exported with (their folder under `goldens/` of the test-asset root). */
export const WAN_TI2V_SERIES_NAME = "wan2.2-ti2v-5b-i8-dyn";
/** The component: the manifest's weights key, which is also the container's graph name (container-v1 §2.1). */
export const WAN_TI2V_COMPONENT = "transformer";
/** The goldens `io.<case>` / `reference.<case>` (ADR 0139 decision 2 — the series' relative paths). */
export const WAN_TI2V_GOLDEN_DIR = new URL(
  `${WAN_TI2V_COMPONENT}/`,
  goldenDir(WAN_TI2V_SERIES_NAME),
);
/** The command that fetches the goldens and corpus into the local test-asset root (ADR 0139 decision 3). */
export const TEST_ASSETS_FETCH = "node tools/test-assets/fetch.ts";

/** Whether the Wan2.2 mirror is present in a format this version reads (checked on each call). */
export const wanTi2vMirrorPresent = (): boolean => mirrorPresent(WAN_TI2V_DIST_ROOT);

/** Opens the mirror's int8 transformer with range reads (the transformer is int8 in every preset). */
export const openTi2vTransformer = (): Promise<OpenedContainer> =>
  openWanTi2vComponent(WAN_TI2V_COMPONENT, "i8");
/** The asset's name (`ROPE_BASE_ASSET` of `wan/export_dit.py` — the same slot for 5B). */
const ROPE_BASE_ASSET = "rope_base";

/** The patch of Wan2.2 TI2V-5B (`(1,2,2)`, 48 latent channels — the transformer's config). */
export const WAN22_GEOMETRY: WanPatchGeometry = {
  channels: 48,
  patchFrames: 1,
  patchHeight: 2,
  patchWidth: 2,
};

export type Ti2vForm = "t2v" | "i2v";

/**
 * One golden case (the role is band decision / acceptance — keeping the two independent is a MUST of the recipe's
 * table).
 */
export type Ti2vCase = {
  readonly name: string;
  readonly role: "band" | "accept" | "full-band" | "full-accept";
  readonly form: Ti2vForm;
};

/** The S = 192 goldens (stage 1 — deciding: T2V 3 + I2V 3; acceptance: T2V 2 + I2V 2). */
export const WAN_TI2V_CASES: readonly Ti2vCase[] = [
  { name: "band-t2v-s00192-t0999", role: "band", form: "t2v" },
  { name: "band-t2v-s00192-t0500", role: "band", form: "t2v" },
  { name: "band-t2v-s00192-t0250", role: "band", form: "t2v" },
  { name: "band-i2v-s00192-t0999", role: "band", form: "i2v" },
  { name: "band-i2v-s00192-t0750", role: "band", form: "i2v" },
  { name: "band-i2v-s00192-t0113", role: "band", form: "i2v" },
  { name: "accept-t2v-s00192-t0600", role: "accept", form: "t2v" },
  { name: "accept-t2v-s00192-t0030", role: "accept", form: "t2v" },
  { name: "accept-i2v-s00192-t0400", role: "accept", form: "i2v" },
  { name: "accept-i2v-s00192-t0900", role: "accept", form: "i2v" },
];

/**
 * The full-size S (the 2 shapes of ADR 0121's addendum "ruling 1 confirmed" — 832×480, 81 frames = 8,190 for the
 * latent `[48,21,30,52]`, and 1280×704, 33 frames = 7,920 for `[48,9,44,80]`).
 */
export type Ti2vFullTokens = 8190 | 7920;

/**
 * The full-size goldens (stage 2 — per shape, 3 deciding (T2V 3 + I2V 3 over the 2 shapes) and acceptance T2V 2 +
 * I2V 2). Ordered with the larger allocation, S = 8,190, first (the GPU test runs them in this order).
 */
export const WAN_TI2V_FULL_CASES: readonly (Ti2vCase & { readonly tokens: Ti2vFullTokens })[] = [
  { name: "full-band-t2v-s08190-t0999", role: "full-band", form: "t2v", tokens: 8190 },
  { name: "full-band-i2v-s08190-t0500", role: "full-band", form: "i2v", tokens: 8190 },
  { name: "full-band-t2v-s08190-t0113", role: "full-band", form: "t2v", tokens: 8190 },
  { name: "full-accept-t2v-s08190-t0999", role: "full-accept", form: "t2v", tokens: 8190 },
  { name: "full-accept-t2v-s08190-t0600", role: "full-accept", form: "t2v", tokens: 8190 },
  { name: "full-accept-i2v-s08190-t0900", role: "full-accept", form: "i2v", tokens: 8190 },
  { name: "full-accept-i2v-s08190-t0400", role: "full-accept", form: "i2v", tokens: 8190 },
  { name: "full-band-i2v-s07920-t0999", role: "full-band", form: "i2v", tokens: 7920 },
  { name: "full-band-t2v-s07920-t0750", role: "full-band", form: "t2v", tokens: 7920 },
  { name: "full-band-i2v-s07920-t0250", role: "full-band", form: "i2v", tokens: 7920 },
  { name: "full-accept-t2v-s07920-t0999", role: "full-accept", form: "t2v", tokens: 7920 },
  { name: "full-accept-t2v-s07920-t0300", role: "full-accept", form: "t2v", tokens: 7920 },
  { name: "full-accept-i2v-s07920-t0800", role: "full-accept", form: "i2v", tokens: 7920 },
  { name: "full-accept-i2v-s07920-t0050", role: "full-accept", form: "i2v", tokens: 7920 },
];

/** Whether the case is an acceptance case (not used to decide the band). */
export const isAccept = ({ role }: Ti2vCase): boolean =>
  role === "accept" || role === "full-accept";

/**
 * Whether the file exists (anything other than NotFound propagates — a permission error is not read as "the asset is
 * missing").
 */
export const filePresent = (url: URL): boolean => {
  try {
    return statSync(url).isFile;
  } catch (cause) {
    if (isNotFound(cause)) return false;
    throw cause;
  }
};

/** The 2 golden files (`io.<case>` and `reference.<case>`). */
export const caseFiles = (name: string): readonly URL[] => [
  new URL(`io.${name}.safetensors`, WAN_TI2V_GOLDEN_DIR),
  new URL(`reference.${name}.safetensors`, WAN_TI2V_GOLDEN_DIR),
];

export const scalarOf = (file: SafetensorsFile, key: string, where: string): number => {
  const view = viewOf(file, key, where);
  if (view.dtype !== "I32" || view.byteLength !== 4)
    throw new Error(`${where}: '${key}' is unexpected`);
  return new Int32Array(file.buffer, view.byteOffset, 1)[0];
};

/** The index of the first element whose bits differ (-1 when equal). 0 when the lengths differ. */
export const firstBitMismatch = (
  actual: Float32Array | Uint32Array,
  expected: Float32Array | Uint32Array,
): number => {
  if (actual.length !== expected.length) return 0;
  const left = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
  const right = new Uint32Array(expected.buffer, expected.byteOffset, expected.length);
  return left.findIndex((bits, index) => bits !== right[index]);
};

/** The goldens of one case (the graph inputs, upstream's latent shape, the 2 timesteps and the reference). */
export type Ti2vGolden = {
  readonly io: SafetensorsFile;
  readonly reference: SafetensorsFile;
  /** The host latent's shape `[C,F,H,W]` (no batch axis — `src/wan/dit-tokens.ts`). */
  readonly latentShape: readonly number[];
  /** The generation-side timestep. */
  readonly timestep: number;
  /**
   * The condition-side timestep (0 for I2V, the same as the generation side for T2V — `condition_timestep` of
   * `reference.<case>`).
   */
  readonly conditionTimestep: number;
};

export const loadTi2vGolden = async (name: string): Promise<Ti2vGolden> => {
  const [ioUrl, referenceUrl] = caseFiles(name);
  const io = parseSafetensors(await readBuffer(ioUrl));
  const reference = parseSafetensors(await readBuffer(referenceUrl));
  const latents = viewOf(reference, "latents", name);
  assertEquals(latents.shape.length, 5, `${name}: rank of latents`);
  assertEquals(latents.shape[0], 1, `${name}: batch of latents`);
  return {
    io,
    reference,
    latentShape: latents.shape.slice(1),
    timestep: scalarOf(reference, "timestep", name),
    conditionTimestep: scalarOf(reference, "condition_timestep", name),
  };
};

/** The golden's `input.condition_mask` (bool stored as u32 0 / 1 — `docs/ir-v2.md`). */
export const storedConditionMask = (
  golden: Ti2vGolden,
  where: string,
): Uint32Array<ArrayBuffer> => {
  const view = viewOf(golden.io, "input.condition_mask", where);
  if (view.dtype !== "U32") {
    throw new Error(`${where}: the condition mask is stored as ${view.dtype} (expected U32)`);
  }
  return new Uint32Array(golden.io.buffer, view.byteOffset, view.byteLength / 4);
};

/**
 * The host-side condition mask `[1,S,1]` (u32 0 / 1 — see the module doc). For I2V the H'·W' tokens of the first latent
 * frame are 1. The token index is `(f·H' + h)·W' + w` (`src/wan/dit-tokens.ts`), so they come first, contiguously.
 */
export const conditionMask = (grid: WanTokenGrid, form: Ti2vForm): Uint32Array<ArrayBuffer> => {
  const mask = new Uint32Array(grid.count);
  if (form === "i2v") mask.fill(1, 0, grid.rows * grid.cols);
  return mask;
};

/** Reads the container asset `rope_base` (role `rope-base`) into the raw table. */
export const readRopeBase = async (opened: OpenedContainer): Promise<WanRopeBase> => {
  const reader = opened.asset(ROPE_BASE_ASSET);
  assertEquals(reader.role, "rope-base", `role of asset '${ROPE_BASE_ASSET}'`);
  const bytes = await reader.read(0, reader.length);
  return parseWanRopeBase(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer,
  );
};
