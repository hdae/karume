/**
 * The **pure-function part** of the self A/B check (ADR 0110 decision 5). It touches neither the
 * GPU nor the file system — the real-GPU checks (`e2e_anima_ab_test.ts` / `e2e_irodori_ab_test.ts` /
 * `e2e_sbv2_ab_test.ts`) own everything up to capturing the observation-point arrays and the census,
 * and "what turns red" is gathered into this one module (so the order and wording of the verdicts
 * do not diverge between families). `e2e_ab_gate_test.ts` (core lane) pins the behaviour.
 *
 * The comparison partner is **the reference path on the same machine** (same weights, the preset
 * with an empty `session`); no torch golden is used. The reference path on the same machine is
 * deterministic, so the control side has zero noise — when none of the bundle's knobs reach the
 * Session, the practical preset's observation point **does not differ from the reference partner's
 * by a single bit**. This one point is the minimum requirement of the floor (decision 5-2).
 */

import type { GpuFeaturesSpec, SessionSpec } from "@karume/hub";
import type { SessionOptions } from "@karume/runtime";
import {
  type CensusRow,
  countDispatches,
  type KeyPredicate,
  type NumericSeat,
  SEAT_REFERENCE,
  SEAT_SIGNATURES,
} from "../../../runtime/tests/helpers/pipeline-census.ts";
import type { Comparison } from "../../../runtime/tests/helpers/results.ts";

/**
 * The band (ADR 0110 decisions 4 / 5 — a declaration, not a row per environment key; not rewritten
 * by `KARUME_REFERENCE`). The metric is the observation point's relative RMS ({@link relRms}).
 *
 * - `floor` — a lower bound set **in addition to** "differs by at least one bit" (the MUST of
 *   decision 5-2 — {@link judgeAb} always checks it regardless of the band). It is set above 0 only
 *   when measurements show it can numerically catch a bundle of which only part dropped out
 *   (otherwise the census owns per-knob fallbacks — decision 5-1).
 * - `ceiling` — the collapse bound (decision 5-3 — about 2 × the observation point's measurement).
 */
export type AbBand = { readonly floor: number; readonly ceiling: number };

/** The A/B measurement of one observation point. */
export type AbMeasurement = {
  /** ‖p − r‖₂ / ‖r‖₂ ({@link relRms}). */
  readonly relRms: number;
  readonly maxAbs: number;
  /** Whether the observation points of the practical preset and the reference partner are bit-identical (input to the floor verdict). */
  readonly identical: boolean;
};

/** The part of the manifest {@link referencePartnerOf} reads (a `Manifest` is assignable to it). */
export type QuantTable = {
  readonly models: Readonly<
    Record<
      string,
      {
        readonly quants: Readonly<
          Record<
            string,
            {
              readonly weights: Readonly<Record<string, string>>;
              readonly session: SessionSpec;
              readonly gpuFeatures?: GpuFeaturesSpec;
            }
          >
        >;
      }
    >
  >;
};

type QuantEntry = QuantTable["models"][string]["quants"][string];

/** Looks up the model / quant declaration (fails loudly when absent, listing what exists). */
export const quantOf = (manifest: QuantTable, model: string, quant: string): QuantEntry => {
  if (!Object.hasOwn(manifest.models, model)) {
    throw new Error(
      `the manifest has no model '${model}' (available: ${Object.keys(manifest.models).join(
        " / ",
      )})`,
    );
  }
  const { quants } = manifest.models[model];
  if (!Object.hasOwn(quants, quant)) {
    throw new Error(
      `model '${model}' has no quant '${quant}' (available: ${Object.keys(quants).join(" / ")})`,
    );
  }
  return quants[quant];
};

const sameWeights = (
  left: Readonly<Record<string, string>>,
  right: Readonly<Record<string, string>>,
): boolean => {
  const keys = Object.keys(left);
  return (
    keys.length === Object.keys(right).length &&
    keys.every((key) => Object.hasOwn(right, key) && right[key] === left[key])
  );
};

/** The feature names that are required (`shaderF16: false` is the same as not requiring it). */
const requiredFeatures = (features: GpuFeaturesSpec | undefined): ReadonlySet<string> =>
  new Set(
    Object.entries(features ?? {})
      .filter(([, required]) => required === true)
      .map(([name]) => name),
  );

/**
 * Derives the **reference partner** of a practical preset from the manifest (the single preset with
 * the same `weights` mapping, an empty `session`, and `gpuFeatures` requirements that are a subset
 * of the practical preset's — there must be exactly one).
 *
 * A subset, so that a practical preset needing a feature the partner does not (shader-f16 of
 * `f16-c16`) still pairs (refutation verify/A2 §4-3 — with "equal", `f16-c16 ↔ f16` has zero
 * candidates). The reverse (only the partner needs a feature) is not paired, because the partner
 * cannot be built on the device the practical preset gets.
 *
 * MUST: zero or two or more candidates throw. Zero means "no reference-path control is distributed
 * for the practical preset"; two or more means "the floor's meaning depends on which one was
 * compared" — silently picking one in either case would make the check's claim ambiguous.
 */
export const referencePartnerOf = (manifest: QuantTable, model: string, quant: string): string => {
  const practical = quantOf(manifest, model, quant);
  if (Object.keys(practical.session).length === 0) {
    throw new Error(
      `quant '${quant}' has an empty session (a reference partner) — it cannot be the practical side of an A/B`,
    );
  }
  const practicalFeatures = requiredFeatures(practical.gpuFeatures);
  const { quants } = manifest.models[model];
  const candidates = Object.keys(quants).filter((name) => {
    if (name === quant) return false;
    const candidate = quants[name];
    return (
      sameWeights(candidate.weights, practical.weights) &&
      Object.keys(candidate.session).length === 0 &&
      [...requiredFeatures(candidate.gpuFeatures)].every((feature) =>
        practicalFeatures.has(feature),
      )
    );
  });
  if (candidates.length !== 1) {
    throw new Error(
      `quant '${quant}': the reference partners (same weights, empty session, gpuFeatures ⊆ the practical preset's) number ` +
        `${candidates.length} (${candidates.join(" / ") || "none"}) — without exactly one ` +
        "the floor's control is undetermined",
    );
  }
  return candidates[0];
};

const assertSameLength = (practical: Float32Array, reference: Float32Array): void => {
  if (practical.length !== reference.length) {
    throw new Error(
      `the observation points differ in element count (practical ${practical.length} / reference ${reference.length}) — ` +
        "the same observation point was not captured from the same input",
    );
  }
};

/**
 * Relative RMS ‖p − r‖₂ / ‖r‖₂ (accumulated in f64).
 *
 * Non-finite values propagate (NaN / ±Inf elements and ‖r‖₂ = 0 yield a non-finite value) —
 * {@link judgeAb} treats them as a ceiling-side failure.
 */
export const relRms = (practical: Float32Array, reference: Float32Array): number => {
  assertSameLength(practical, reference);
  let difference = 0;
  let norm = 0;
  for (let index = 0; index < reference.length; index += 1) {
    const delta = practical[index] - reference[index];
    difference += delta * delta;
    norm += reference[index] * reference[index];
  }
  return Math.sqrt(difference) / Math.sqrt(norm);
};

/**
 * The maximum absolute difference over all elements. Returns NaN when any element is NaN (a `>`
 * comparison is always false for NaN, so a naive version lets NaN pass as 0 — see `worstDifference`
 * in `helpers/irodori-assets.ts`).
 */
export const maxAbs = (practical: Float32Array, reference: Float32Array): number => {
  assertSameLength(practical, reference);
  let worst = 0;
  for (let index = 0; index < reference.length; index += 1) {
    const difference = Math.abs(practical[index] - reference[index]);
    if (Number.isNaN(difference)) return Number.NaN;
    if (difference > worst) worst = difference;
  }
  return worst;
};

const bitsOf = (values: Float32Array): Uint32Array =>
  new Uint32Array(values.buffer, values.byteOffset, values.length);

/** The first index whose bits differ and the number of differing elements (undefined when identical). */
const bitDifference = (
  left: Float32Array,
  right: Float32Array,
): { readonly first: number; readonly count: number } | undefined => {
  assertSameLength(left, right);
  const leftBits = bitsOf(left);
  const rightBits = bitsOf(right);
  let first = -1;
  let count = 0;
  for (let index = 0; index < leftBits.length; index += 1) {
    if (leftBits[index] === rightBits[index]) continue;
    if (first < 0) first = index;
    count += 1;
  }
  return first < 0 ? undefined : { first, count };
};

/** The measurement of one observation point (the input of {@link judgeAb}). */
export const measureAb = (practical: Float32Array, reference: Float32Array): AbMeasurement => ({
  relRms: relRms(practical, reference),
  maxAbs: maxAbs(practical, reference),
  identical: bitDifference(practical, reference) === undefined,
});

/**
 * The automated check of on-device determinism (the MUST of ADR 0110 decision 6) — the observation
 * points of two independent Sessions with the same preset and the same input are bit-identical.
 */
export const assertBitIdentical = (
  first: Float32Array,
  second: Float32Array,
  where: string,
): void => {
  const difference = bitDifference(first, second);
  if (difference === undefined) return;
  const { first: at, count } = difference;
  throw new Error(
    `${where}: two runs with the same preset and the same input are not bit-identical (violates the determinism MUST — ADR 0110 decision 6)\n` +
      `  ${count} / ${first.length} elements differ. First at ${at}: ${first[at]} / ${second[at]}\n` +
      "  suspect floating-point atomics, arrival-order merges, or runtime parameters missing from the key",
  );
};

const format = (value: number): string =>
  Number.isFinite(value) ? value.toExponential(4) : String(value);

/**
 * Judges the A/B of one observation point. The red verdicts, in order: ① non-finite (ceiling side)
 * ② not a single bit differs (floor MUST) ③ band not derived yet ④ the band's floor ⑤ the band's
 * ceiling.
 *
 * MUST: ② is checked whether or not a band exists — a silent fallback of the whole bundle (not one
 * knob reaches the Session) is detectable before any band is derived, and passing it there would let
 * a run without a band hide "not reaching".
 * MUST: ③ stops red with the measurement in the message (the same approach as `MEASURED` in
 * `e2e_irodori_ab_test.ts` — no placeholder value that passes, which would add a check with no
 * detection power).
 */
export const judgeAb = (measured: AbMeasurement, band: AbBand | undefined, where: string): void => {
  const observed = describeAb(measured);
  if (!Number.isFinite(measured.relRms) || !Number.isFinite(measured.maxAbs)) {
    throw new Error(
      `${where}: ceiling-side failure — non-finite (${observed}). The observation point has NaN / Inf, or the reference partner's ` +
        "observation point is zero (‖r‖₂ = 0)",
    );
  }
  if (measured.identical) {
    throw new Error(
      `${where}: floor failure — the practical preset does not differ from the reference partner by a single bit (the bundle's knobs ` +
        "probably do not reach the Session — check the quant's session → SessionOptions → Session wiring or the eligibility checks)",
    );
  }
  if (band === undefined) {
    throw new Error(
      `${where}: band not derived yet (measured ${observed}) — derive the band from this measurement by the rule of ADR 0110 decision 5 ` +
        "and write it into the test's constants together with the derivation table",
    );
  }
  if (!(band.floor >= 0 && band.ceiling > band.floor && Number.isFinite(band.ceiling))) {
    throw new Error(
      `${where}: the band declaration is broken (floor ${band.floor} / ceiling ${band.ceiling})`,
    );
  }
  if (measured.relRms < band.floor) {
    throw new Error(
      `${where}: floor failure — ${observed} fell below the floor ${band.floor} (part of the bundle's knobs probably dropped ` +
        "and the difference shrank — a small difference is a sign of a silent fallback, not an improvement)",
    );
  }
  if (measured.relRms > band.ceiling) {
    throw new Error(
      `${where}: ceiling failure — ${observed} exceeded the ceiling ${band.ceiling} (collapse detected). Before widening ` +
        "the band, confirm with the op checks which kernel's numbers moved",
    );
  }
};

/**
 * Aborts generation once the observation point has been captured (the convention that a throw in
 * the subscriber aborts generation — the doc of `onEvent` in anima / irodori). `run` calls the
 * `capture` it receives at the observation point.
 *
 * MUST: any exception other than the abort marker is rethrown as is (a generation failure is not
 * hidden as "aborted"). A run whose generation ends without ever reaching the observation point also
 * fails (a configuration that never reaches step 1 does not go green).
 */
export const captureThenAbort = async <T>(
  where: string,
  run: (capture: (value: T) => never) => Promise<unknown>,
): Promise<T> => {
  const stop = new Error(`${where}: observation point captured, aborting generation`);
  let captured: { readonly value: T } | undefined;
  try {
    await run((value) => {
      captured = { value };
      throw stop;
    });
  } catch (error) {
    if (error !== stop) throw error;
  }
  if (captured === undefined) {
    throw new Error(`${where}: generation ended without ever reaching the observation point`);
  }
  return captured.value;
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const recordAt = (
  parent: Record<string, unknown>,
  key: string,
  where: string,
): Record<string, unknown> => {
  const child = parent[key];
  if (!Object.hasOwn(parent, key) || !isRecord(child)) {
    throw new Error(`${where}.${key}: missing or not an object`);
  }
  return child;
};

/**
 * Returns a copy of the distribution JSON (the raw content of `karume.json`) with only the `session`
 * of one preset replaced (a temporary manifest for fault injection — neither the weights nor the
 * other presets are touched). The original value is not modified.
 */
export const overrideQuantSession = (
  raw: unknown,
  model: string,
  quant: string,
  session: Readonly<Record<string, unknown>>,
): Record<string, unknown> => {
  if (!isRecord(raw)) throw new Error("manifest: not an object");
  const models = recordAt(raw, "models", "manifest");
  const entry = recordAt(models, model, "manifest.models");
  const quants = recordAt(entry, "quants", `manifest.models.${model}`);
  const seat = recordAt(quants, quant, `manifest.models.${model}.quants`);
  return {
    ...raw,
    models: {
      ...models,
      [model]: { ...entry, quants: { ...quants, [quant]: { ...seat, session: { ...session } } } },
    },
  };
};

/** The option names of the census table (`Object.keys` returns string[], so the type is taken from the table again). */
const NUMERIC_SEATS = Object.keys(SEAT_REFERENCE) as readonly NumericSeat[];

/**
 * A one-line summary of how many dispatches ran the variant of each bundle knob (options with a non-reference value) (`linearCompute=a8 ×317 / …`).
 * Kept in the result's note and the log — the "took effect" verdict belongs to `assertSeatsApplied`; this only records counts.
 */
export const describeBundleCensus = (
  rows: readonly CensusRow[],
  declared: SessionOptions,
): string =>
  NUMERIC_SEATS.flatMap((seat) => {
    const value = declared[seat];
    if (value === undefined || value === SEAT_REFERENCE[seat]) return [];
    const signatures: Readonly<Record<string, KeyPredicate>> = SEAT_SIGNATURES[seat];
    const predicate = Object.hasOwn(signatures, String(value))
      ? signatures[String(value)]
      : undefined;
    const count =
      predicate === undefined ? "no predicate" : String(countDispatches(rows, predicate));
    return [`${seat}=${String(value)} ×${count}`];
  }).join(" / ");

/**
 * One entry for the results (`comparisons` of `results.json`). It is recorded **before** the verdict
 * — a red run also leaves its measurement in the result (the material for deriving a band is needed
 * from failing runs too).
 */
export const comparisonOf = (
  observation: { readonly output: string; readonly reference: string; readonly practical: string },
  measured: AbMeasurement,
  band: AbBand | undefined,
): Comparison => ({
  ...observation,
  relRms: measured.relRms,
  maxAbs: measured.maxAbs,
  band: band === undefined ? undefined : { metric: "relRms", ...band },
});

/** The log form of a measurement (`relRMS 1.2345e-2 / maxAbs 3.1000e-1`). */
export const describeAb = (measured: AbMeasurement): string =>
  `relRMS ${format(measured.relRms)} / maxAbs ${format(measured.maxAbs)}`;
