/**
 * The **readers** shared by Irodori's real-GPU E2E checks (the distribution's assets and the
 * full-loop goldens).
 *
 * The checks themselves are split into one file per preset (the storage-dtype presets get the
 * numeric parity suite = `e2e_irodori_latent_test.ts`; the activation-quantized `i8-a8` preset gets
 * a discrimination band + key census = `e2e_irodori_ab_test.ts` — the suites differ in nature, so
 * they do not share one table). What is shared is only "what to read from where"; **what to require
 * belongs to each check**.
 *
 * NOTE: only the tests read local files (`readFile` of `@karume/test-support/fs`). The packages themselves use Web
 * standard APIs only (cross-cutting invariant), and `fromAssets` is a surface that only takes bytes.
 */

import { isNotFound, readFile, statSync } from "@karume/test-support/fs";
import { goldenDir } from "@karume/test-support/test-assets";
import { parseManifest, resolveSelection } from "@karume/hub";
import type { FileRef, Manifest, ModelEntry } from "@karume/hub";
import { parseSafetensors } from "@karume/runtime";
import { mirrorPresent, mirrorRoot } from "../../../models/tests/helpers/mirror.ts";
import { readTextIfPresent } from "../../../models/tests/helpers/read-if-present.ts";

/** The distribution mirror (`models/karume-irodori-v4-small/` — ADR 0139 decision 1). */
export const ASSETS_DIR: URL = mirrorRoot("karume-irodori-v4-small");

/** The generation command to paste as is when a case SKIPs. */
export const DIST_COMMAND = "cd tools/exporter && uv run karume dist --pipeline irodori";
/**
 * Fetches the test assets, including the full-loop goldens (`goldens/<series>/pipeline/`). The
 * assets gate accepts only a complete fetch; a partial `--include` fetch never records a revision.
 */
export const FETCH_COMMAND = "node tools/test-assets/fetch.ts";

export const MODEL = "v4-small";

/**
 * The full-loop goldens of one series root (named for its storage dtype):
 * `outputs/test-assets/goldens/<seriesRoot>/pipeline/` (ADR 0139 decision 2).
 */
export const pipelineGoldenDir = (seriesRoot: string): URL =>
  new URL("pipeline/", goldenDir(seriesRoot));

/**
 * Whether a golden file is present. MUST: only NotFound means absent — any other failure propagates,
 * so a broken root is reported as broken rather than skipped.
 */
export const goldenPresent = (url: URL): boolean => {
  try {
    statSync(url);
    return true;
  } catch (cause) {
    if (isNotFound(cause)) return false;
    throw cause;
  }
};

/**
 * The mirror's manifest, or `undefined` when the mirror is absent or unreadable by this version
 * (`mirrorPresent` — not `karume/5`, or a referenced mirror is absent), so its cases SKIP instead of
 * failing at module evaluation in `parseManifest`.
 */
export const manifestText: string | undefined = mirrorPresent(ASSETS_DIR)
  ? await readTextIfPresent(new URL("karume.json", ASSETS_DIR))
  : undefined;

/** Only the fields of the golden `meta.json` the checks read (the exporter owns the shape — this is only the reader). */
export type GoldenCase = {
  readonly text: string;
  readonly caption: string;
  readonly S: number;
  readonly forwards: number;
  readonly zAbsMax: number;
  readonly reference: { readonly frames: number } | null;
};

export type GoldenMeta = {
  readonly steps: number;
  readonly initScale: number;
  readonly cfgRange: readonly [number, number];
  readonly cfgScales: { readonly text: number; readonly speaker: number; readonly caption: number };
  readonly frameRate: number;
  readonly tEmbedDim: number;
  readonly caps: { readonly text: number; readonly speaker: number; readonly caption: number };
  readonly cases: Readonly<Record<string, GoldenCase>>;
};

export const readManifest = (): Manifest => parseManifest(manifestText as string);

export const modelEntry = (manifest: Manifest): ModelEntry => {
  if (!Object.hasOwn(manifest.models, MODEL)) {
    throw new Error(
      `the distribution has no model '${MODEL}' (available: ${manifest.available.models.join(" / ")})`,
    );
  }
  return manifest.models[MODEL];
};

/** Whether the distribution has this quant preset (an environment without it "has not rebuilt" = SKIP). */
export const hasQuantSeat = (quant: string): boolean =>
  manifestText !== undefined && Object.hasOwn(modelEntry(readManifest()).quants, quant);

/**
 * Reads the assets the distribution requires from local files (a local version of `fetchAssets`
 * — it bypasses the fetch layer).
 *
 * A component lists its container's parts as `<component>[i]` (in index order from part 0, **listing
 * zero-length parts too** — the index is the id inside the container, so skipping one shifts every
 * later part down by one).
 */
export const loadLocalAssets = async (
  manifest: Manifest,
  quant: string,
): Promise<Record<string, Uint8Array<ArrayBuffer>>> => {
  const selection = resolveSelection(manifest, { model: MODEL, quant });
  const byPath = new Map<string, Uint8Array<ArrayBuffer>>();
  let assets: Record<string, Uint8Array<ArrayBuffer>> = {};
  const read = async (key: string, ref: FileRef): Promise<void> => {
    const cached = byPath.get(ref.path);
    const bytes = cached ?? (await readFile(new URL(ref.path, ASSETS_DIR)));
    if (cached === undefined) byPath.set(ref.path, bytes);
    assets = { ...assets, [key]: bytes };
  };
  for (const name of Object.keys(selection.containers)) {
    const { parts } = selection.containers[name];
    for (const [index, ref] of parts.entries()) await read(`${name}[${index}]`, ref);
  }
  for (const name of Object.keys(selection.assets)) await read(name, selection.assets[name]);
  return assets;
};

/** Looks up an f32 tensor in one golden case (`case.<name>.safetensors`). */
export const readCase = async (
  directory: URL,
  name: string,
): Promise<(tensor: string) => Float32Array<ArrayBuffer>> => {
  const bytes = await readFile(new URL(`case.${name}.safetensors`, directory));
  const file = parseSafetensors(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
  return (tensor: string): Float32Array<ArrayBuffer> => {
    const spec = file.tensors.get(tensor);
    if (spec === undefined) throw new Error(`golden case.${name} has no '${tensor}'`);
    if (spec.dtype !== "F32") throw new Error(`golden '${tensor}' has dtype ${spec.dtype}`);
    // MUST: return a copy (all tensors of a golden share one buffer, and `initialNoise` becomes an
    // array the pipeline **overwrites in place** — a borrowed view would break the second comparison).
    return new Float32Array(
      file.buffer.slice(spec.byteOffset, spec.byteOffset + spec.byteLength),
    ) as Float32Array<ArrayBuffer>;
  };
};

/**
 * The maximum absolute difference over all elements and its position (for reporting the difference).
 *
 * MUST: NaN / ±Inf are counted and fail. `difference > maxAbs` is always false for NaN, so a naive
 * version would let **an all-NaN output pass the tolerance check as maxAbs 0** (the checks that look
 * only at this return value are in `e2e_irodori_latent_test.ts` / `e2e_irodori_ab_test.ts`). For
 * contrast, see the pixel_values comparison in `e2e_depth_anything_real_test.ts`.
 */
export const worstDifference = (
  actual: Float32Array,
  expected: Float32Array,
): { readonly maxAbs: number; readonly at: number } => {
  if (actual.length !== expected.length) {
    throw new Error(`element counts differ (actual ${actual.length} / golden ${expected.length})`);
  }
  let maxAbs = 0;
  let at = 0;
  let nonFinite = 0;
  let firstNonFinite = -1;
  for (let index = 0; index < actual.length; index += 1) {
    if (!Number.isFinite(actual[index]) || !Number.isFinite(expected[index])) {
      nonFinite += 1;
      if (firstNonFinite < 0) firstNonFinite = index;
      continue;
    }
    const difference = Math.abs(actual[index] - expected[index]);
    if (difference > maxAbs) {
      maxAbs = difference;
      at = index;
    }
  }
  if (nonFinite > 0) {
    throw new Error(
      `${nonFinite} non-finite samples (first at ${firstNonFinite}: ` +
        `actual ${actual[firstNonFinite]} / golden ${expected[firstNonFinite]})`,
    );
  }
  return { maxAbs, at };
};
