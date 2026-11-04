/**
 * The goldens of umT5-XXL's i8 series (`reference.<case>.safetensors`, written by the recipe
 * `wan/umt5_reference.py`; read from the test-asset root `goldens/umt5-xxl-i8-dyn/text_encoder/` — ADR 0139
 * decision 2) and the judgement that compares the GPU output with them (normalized ratio r and band — ADR 0119
 * decision 8 ①).
 *
 * Two e2e checks read the same goldens with the same judgement: Wan's umT5 port check (`e2e_wan_umt5_test.ts` —
 * runs the container of the mirror `karume-umt5-xxl` with the runtime directly) and the public `umt5.Encoder`
 * (`e2e_wan_umt5_test.ts` — built by `fromPretrained` over the same mirror). The band, metric and cases live only
 * here (copied into both, only one of them would move).
 *
 * The golden's contents (one file per case):
 *
 * - `input_ids` I32 `[L]` / `relative_position_buckets` I32 `[L,L]` — the inputs of the Python path (upstream's
 *   `prompt_clean` → transformers 5.14.1, upstream's bucket formula)
 * - `output.f64` F32 `[1,L,4096]` — the reference run with f64 activations too, rounded to f32 (the error baseline)
 * - `output.f32` F32 `[1,L,4096]` — the CPU f32 reference (a write-out bit-identical to upstream's eager — the
 *   normalization denominator)
 * - `output.unquantized.f64` / `output.unquantized.f32` F32 `[1,L,4096]` (**acceptance only**) — the same write-out
 *   run with unquantized weights (the baseline of the quality record — not used by the check)
 * - metadata (JSON under one key) — the prompt's original text and the string after preprocessing, L, the role and
 *   the sha256 of the container's part 0
 *
 * Zero side effects (only constants and functions — files are read when called).
 */

import { isNotFound, statSync } from "@karume/test-support/fs";
import { assert, assertEquals } from "@std/assert";
import { parseSafetensors, type SafetensorsFile } from "@karume/runtime";
import { goldenDir } from "@karume/test-support/test-assets";
import { floatsOf, readBuffer, viewOf } from "./safetensors.ts";

/**
 * The band of the normalized ratio r (ADR 0119 decision 8 ① — derived the same way as ADR 0118 decision 8). The worst
 * r of the 6 deciding cases (`band-*`) × 5, rounded up to 2 significant digits; the 4 acceptance cases (`accept-*`) are
 * not used to decide the band.
 *
 * `undefined` = not derived yet: each case prints r and the fault-injection r and **stops red**, and the last step
 * prints the band candidate and goes red (no check without detection power is made by provisionally putting in a value
 * that passes — the same treatment as the DiT's `I8_NORMALIZED_BAND`). After the GPU measurement it is written here
 * together with the measured table.
 *
 * MUST: do not change this value, the deciding cases or the metric after looking at the acceptance results (the source
 * of truth for the cases is `band_cases` / `accept_cases` of the recipe `wan/umt5_reference.py`). If an acceptance case
 * falls outside the band, investigate the cause instead of widening the band.
 * Derivation (2026-10-03, B570, the normal device, CPU layer-sequential f64 reference):
 *
 * | deciding (L) | r     |   | acceptance      | r    |
 * | ------------ | ----- | - | --------------- | ---- |
 * | band-l0008   | 8.25  |   | boxing-cats     | 2.78 |
 * | band-l0022   | 5.92  |   | ferret          | 8.87 |
 * | band-l0036   | 2.20  |   | cat-dog-baking  | 4.36 |
 * | band-l0163   | 4.93  |   | negative        | 3.74 |
 * | band-l0327   | 0.904 |   |                 |      |
 * | band-l0488   | 15.8  |   |                 |      |
 *
 * Worst 15.8 × 5 = 79 (rounded up to 2 significant digits). Fault injection: swapping the tables of layers 0 / 1 gives
 * r 5.0e+5–8.0e+5, scale × 2 on wo.12 gives 5.0e+5–9.7e+5 (4 orders of magnitude above the band). scale × 1.0001
 * (recorded only) gives 22.6 / 72.6 / 80.2 / 40.0, at the edge of the band (r is proportional to δ — a 1e-4 scale shift
 * cannot be told apart by this check; there is no floor for subtle faults).
 */
export const UMT5_NORMALIZED_BAND: number | undefined = 79;

/** The series the goldens were exported with (their folder under `goldens/` of the test-asset root). */
const GOLDEN_SERIES = "umt5-xxl-i8-dyn";
export const COMPONENT = "text_encoder";
export const GOLDEN_DIR = new URL(`${COMPONENT}/`, goldenDir(GOLDEN_SERIES));

/** The command that fetches the goldens into the local test-asset root (ADR 0139 decision 3). */
export const FETCH = "node tools/test-assets/fetch.ts";

export const GENERATE =
  "cd tools/export-recipes && uv run --group wan --inexact python -m wan.umt5_export reference";

/** The golden's keys (the same spelling as the recipe `wan/umt5_reference.py`). */
export const INPUT_IDS_KEY = "input_ids";
export const BUCKETS_KEY = "relative_position_buckets";
const OUTPUT_F64_KEY = "output.f64";
const OUTPUT_F32_KEY = "output.f32";
const OUTPUT_UNQUANTIZED_F64_KEY = "output.unquantized.f64";
const OUTPUT_UNQUANTIZED_F32_KEY = "output.unquantized.f32";
const METADATA_KEY = "karume.wan.umt5_reference";
/**
 * /2 = acceptance cases also carry the unquantized reference ({@link OUTPUT_UNQUANTIZED_F32_KEY} and the like are
 * required for acceptance).
 */
const REFERENCE_FORMAT = "karume-wan-umt5-reference/2";

export type CaseRole = "band" | "accept";

/**
 * The cases that should have been generated. **Fixed here rather than enumerated** (so that tests do not silently
 * vanish in an environment where only part of the generation was run). The source of truth is the recipe
 * `wan/umt5_reference.py` (the deciding cases are 3 single ones chosen with seed 20261003 from the accepted random
 * cases of the 10a parity fixture (L = 8 / 22 / 36) and 3 composites that chain random cases (L = 163 / 327 / 488);
 * the acceptance cases are the 4 fixed prompts). `embedding` is the tensor name of the bf16 precomputed asset
 * (acceptance only).
 */
export const CASES: readonly {
  readonly name: string;
  readonly role: CaseRole;
  readonly embedding?: string;
}[] = [
  { name: "band-l0008", role: "band" },
  { name: "band-l0022", role: "band" },
  { name: "band-l0036", role: "band" },
  { name: "band-l0163", role: "band" },
  { name: "band-l0327", role: "band" },
  { name: "band-l0488", role: "band" },
  { name: "accept-boxing-cats", role: "accept", embedding: "boxing-cats" },
  { name: "accept-ferret", role: "accept", embedding: "ferret" },
  { name: "accept-cat-dog-baking", role: "accept", embedding: "cat-dog-baking" },
  { name: "accept-negative", role: "accept", embedding: "negative" },
];

export const goldenUrl = (name: string): URL =>
  new URL(`reference.${name}.safetensors`, GOLDEN_DIR);

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

const intsOf = (file: SafetensorsFile, key: string, where: string): Int32Array<ArrayBuffer> => {
  const view = viewOf(file, key, where);
  if (view.dtype !== "I32") throw new Error(`${where}: '${key}' is ${view.dtype}`);
  return new Int32Array(file.buffer, view.byteOffset, view.byteLength / 4);
};

/** The golden's metadata (the fields of the recipe's `golden_metadata` used here). */
export type GoldenMeta = {
  readonly role: string;
  readonly prompt: string;
  /**
   * The string after upstream's preprocessing (`prompt_clean`) — the golden's `input_ids` are this string passed
   * through transformers 5.14.1. A standalone tokenizer without the preprocessing takes this string.
   */
  readonly cleaned: string;
  readonly length: number;
  readonly part0Sha256: string;
};

const parseMeta = (file: SafetensorsFile, where: string): GoldenMeta => {
  const raw = file.metadata.get(METADATA_KEY);
  if (raw === undefined || file.metadata.size !== 1) {
    throw new Error(`${where}: the metadata should be the single key ${METADATA_KEY}`);
  }
  const value: unknown = JSON.parse(raw);
  const isRecord = (item: unknown): item is Record<string, unknown> =>
    typeof item === "object" && item !== null && !Array.isArray(item);
  const record = (item: unknown, label: string): Record<string, unknown> => {
    if (!isRecord(item)) throw new Error(`${where}: metadata ${label} is not an object`);
    return item;
  };
  const meta = record(value, "root");
  const container = record(meta["container"], "container");
  const { format, role, prompt, cleaned, length } = meta;
  const part0Sha256 = container["part0Sha256"];
  if (
    format !== REFERENCE_FORMAT ||
    typeof role !== "string" ||
    typeof prompt !== "string" ||
    typeof cleaned !== "string" ||
    typeof length !== "number" ||
    typeof part0Sha256 !== "string"
  ) {
    throw new Error(`${where}: the metadata's shape is not ${REFERENCE_FORMAT}`);
  }
  return { role, prompt, cleaned, length, part0Sha256 };
};

/** The unquantized reference (acceptance only — the baseline of the quality record). */
type UnquantizedReference = {
  readonly expected: Float32Array<ArrayBuffer>;
  readonly reference: Float32Array<ArrayBuffer>;
};

/** One case's golden. */
export type Golden = {
  readonly meta: GoldenMeta;
  readonly ids: Int32Array<ArrayBuffer>;
  readonly buckets: Int32Array<ArrayBuffer>;
  readonly expected: Float32Array<ArrayBuffer>;
  readonly reference: Float32Array<ArrayBuffer>;
  readonly outputShape: readonly number[];
  /**
   * Only acceptance cases have it (fails loudly when present on a deciding case or missing on an acceptance case —
   * mirrors the recipe's `_check_unquantized`).
   */
  readonly unquantized?: UnquantizedReference;
};

export const loadGolden = async (name: string): Promise<Golden> => {
  const file = parseSafetensors(await readBuffer(goldenUrl(name)));
  const meta = parseMeta(file, name);
  const ids = intsOf(file, INPUT_IDS_KEY, name);
  assertEquals(meta.length, ids.length, `${name}: the metadata's L and the length of input_ids`);
  const accept = meta.role === "accept";
  assertEquals(
    file.tensors.has(OUTPUT_UNQUANTIZED_F32_KEY),
    accept,
    `${name}: only acceptance cases carry the unquantized reference (role ${meta.role})`,
  );
  return {
    meta,
    ids,
    buckets: intsOf(file, BUCKETS_KEY, name),
    expected: floatsOf(file, OUTPUT_F64_KEY, name),
    reference: floatsOf(file, OUTPUT_F32_KEY, name),
    outputShape: viewOf(file, OUTPUT_F64_KEY, name).shape,
    ...(accept
      ? {
          unquantized: {
            expected: floatsOf(file, OUTPUT_UNQUANTIZED_F64_KEY, name),
            reference: floatsOf(file, OUTPUT_UNQUANTIZED_F32_KEY, name),
          },
        }
      : {}),
  };
};

/** The index of the first differing element (-1 when equal, 0 when the lengths differ). */
export const firstMismatch = (actual: ArrayLike<number>, expected: ArrayLike<number>): number => {
  if (actual.length !== expected.length) return 0;
  for (let index = 0; index < expected.length; index += 1) {
    if (actual[index] !== expected[index]) return index;
  }
  return -1;
};

export type Difference = {
  readonly maxAbs: number;
  readonly maxRel: number;
  readonly referenceMaxAbs: number;
  readonly nonFinite: number;
};

export const difference = (actual: Float32Array, expected: Float32Array): Difference => {
  assertEquals(actual.length, expected.length, "element count");
  let maxAbs = 0;
  let maxRel = 0;
  let referenceMaxAbs = 0;
  let nonFinite = 0;
  for (let index = 0; index < expected.length; index += 1) {
    const got = actual[index];
    if (!Number.isFinite(got)) nonFinite += 1;
    const error = Math.abs(got - expected[index]);
    const magnitude = Math.abs(expected[index]);
    maxAbs = Math.max(maxAbs, error);
    if (magnitude > 0) maxRel = Math.max(maxRel, error / magnitude);
    referenceMaxAbs = Math.max(referenceMaxAbs, magnitude);
  }
  return { maxAbs, maxRel, referenceMaxAbs, nonFinite };
};

/**
 * The ratio (maximum absolute difference ÷ the reference's maximum absolute value — the same form as the recipe's
 * `max_ratio`).
 */
export const ratioOf = (diff: Difference): number => diff.maxAbs / diff.referenceMaxAbs;

/**
 * The normalization denominator = the ratio of the CPU f32 reference to the f64 reference. 0 (the f32 reference is
 * bit-identical to f64) cannot be divided by, so it fails loudly.
 */
export const referenceRatioOf = (golden: Golden, name: string): number => {
  const ratio = ratioOf(difference(golden.reference, golden.expected));
  if (!(ratio > 0)) {
    throw new Error(
      `${name}: the ratio of the CPU f32 reference to f64 is ${ratio} (cannot normalize)`,
    );
  }
  return ratio;
};

export const formatRatio = (label: string, diff: Difference, referenceRatio: number): string =>
  `${label}: maxAbs ${diff.maxAbs.toExponential(3)} / reference max abs ${diff.referenceMaxAbs.toFixed(
    3,
  )} (ratio ${ratioOf(diff).toExponential(2)}), r ${(ratioOf(diff) / referenceRatio).toPrecision(3)}`;

/** The band decision (not derived yet → prints r and goes red — {@link UMT5_NORMALIZED_BAND}). */
export const judgeBand = (
  band: number | undefined,
  id: string,
  ratio: number,
  faults: readonly { readonly label: string; readonly ratio: number }[],
): void => {
  if (band === undefined) {
    throw new Error(
      `${id}: the band is not derived yet (r ${ratio.toPrecision(
        3,
      )}) — write the worst deciding r × 5 into UMT5_NORMALIZED_BAND` +
        " (the last step prints the candidate)",
    );
  }
  // The fault-injection decisions come first (even when an acceptance case is outside the band, whether the band
  // catches the faults is still decided).
  for (const fault of faults) {
    assert(
      fault.ratio > band,
      `${id}: fault injection ${fault.label} has r ${fault.ratio.toPrecision(
        3,
      )}, inside the band ${band}`,
    );
  }
  assert(ratio <= band, `${id}: r ${ratio.toPrecision(3)} is outside the band ${band}`);
};
