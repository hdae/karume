/**
 * Reads the `gated_delta_scan` fixture (`fixtures/gated-delta-scan.safetensors`, written by
 * `tools/export-recipes/_shared/gated_delta_fixture.py` from upstream's two f32 functions and an
 * f64 oracle — ADR 0135 decision 6) and holds the acceptance band's arithmetic.
 *
 * The CPU reference test (gated_delta_scan_test.ts) and the GPU test
 * (gpu_gated_delta_scan_test.ts) read **the same cases and the same band** from here.
 *
 * The band (ADR 0135 decision 6, the spike's acceptance): the output is finite, and max_abs and
 * rel_l2 of o against the f64 oracle are each at most **twice the larger of the two upstream f32
 * functions' errors** against the same oracle on the same inputs. The value form exposes only o, so
 * this is the o band.
 */

import { assert } from "@std/assert";
import { readFile } from "@karume/test-support/fs";
import { parseSafetensors } from "../../src/format/safetensors.ts";
import type { RefTensor } from "../../src/reference/ops.ts";

/** The metric pair the band bounds. */
export type BandMetrics = { readonly maxAbs: number; readonly relL2: number };

export type ErrorMetrics = BandMetrics & {
  /** Max over value heads of (head max |x − ref|) / (head max |ref|). */
  readonly maxRelHead: number;
  /** Elements that are not finite (they make every metric infinite). */
  readonly nonFinite: number;
};

export type GatedDeltaShape = {
  readonly B: number;
  readonly T: number;
  readonly Hk: number;
  readonly Hv: number;
  readonly dk: number;
  readonly dv: number;
};

export type OracleFingerprint = {
  readonly sum: number;
  readonly sumSquares: number;
  /** `[flat index, exact f64 value]` of evenly spaced elements and of the largest |value|. */
  readonly samples: readonly (readonly [number, number])[];
};

/** The upstream outputs and the oracle, stored for one case only (the metric cross-check). */
export type StoredOutputs = {
  readonly chunk: Float32Array<ArrayBuffer>;
  readonly recurrent: Float32Array<ArrayBuffer>;
  /** The f64 oracle, rebuilt from its f32 head + tail pair (about 48 significant bits). */
  readonly oracle: Float64Array<ArrayBuffer>;
};

export type GatedDeltaCase = {
  readonly name: string;
  readonly shape: GatedDeltaShape;
  /** The op's inputs in slot order: q, k, v, g, beta. */
  readonly inputs: readonly [RefTensor, RefTensor, RefTensor, RefTensor, RefTensor];
  /** Upstream's f32 errors against the oracle, as the generator measured them (in f64). */
  readonly upstream: { readonly chunk: ErrorMetrics; readonly recurrent: ErrorMetrics };
  /** Twice the larger upstream error, per metric. */
  readonly band: BandMetrics;
  readonly oracle: OracleFingerprint;
  readonly stored?: StoredOutputs;
};

const FIXTURE = new URL("../fixtures/gated-delta-scan.safetensors", import.meta.url);

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const numberAt = (record: Record<string, unknown>, key: string, where: string): number => {
  const value = record[key];
  assert(typeof value === "number", `${where}.${key} is not a number`);
  return value;
};

const recordAt = (
  record: Record<string, unknown>,
  key: string,
  where: string,
): Record<string, unknown> => {
  const value = record[key];
  assert(isRecord(value), `${where}.${key} is not an object`);
  return value;
};

const metricsOf = (record: Record<string, unknown>, where: string): ErrorMetrics => ({
  maxAbs: numberAt(record, "max_abs", where),
  relL2: numberAt(record, "rel_l2", where),
  maxRelHead: numberAt(record, "max_rel_head", where),
  nonFinite: record["finite"] === true ? 0 : Number.NaN,
});

const fingerprintOf = (record: Record<string, unknown>, where: string): OracleFingerprint => {
  const samples = record["samples"];
  assert(Array.isArray(samples), `${where}.samples is not a list`);
  return {
    sum: numberAt(record, "sum", where),
    sumSquares: numberAt(record, "sum_squares", where),
    samples: samples.map((sample: unknown, index): readonly [number, number] => {
      assert(
        Array.isArray(sample) &&
          sample.length === 2 &&
          Number.isInteger(sample[0]) &&
          typeof sample[1] === "number",
        `${where}.samples[${index}] is not [index, value]`,
      );
      return [sample[0], sample[1]];
    }),
  };
};

/** Every case of the fixture, in the generator's order. */
export const gatedDeltaCases = async (): Promise<readonly GatedDeltaCase[]> => {
  const bytes = await readFile(FIXTURE);
  const file = parseSafetensors(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
  const records: unknown = JSON.parse(file.metadata.get("cases") ?? "null");
  assert(Array.isArray(records), "fixture metadata 'cases' is not a list");
  const f32 = (name: string, shape?: readonly number[]): Float32Array<ArrayBuffer> => {
    const view = file.tensors.get(name);
    assert(view?.dtype === "F32", `fixture tensor '${name}' is missing or not F32`);
    if (shape !== undefined) {
      assert(
        view.shape.length === shape.length && view.shape.every((dim, axis) => dim === shape[axis]),
        `fixture tensor '${name}' has shape [${view.shape.join(",")}], expected [${shape.join(",")}]`,
      );
    }
    return new Float32Array(file.buffer, view.byteOffset, view.byteLength / 4);
  };
  return records.map((record: unknown, index): GatedDeltaCase => {
    const where = `cases[${index}]`;
    assert(isRecord(record), `${where} is not an object`);
    const name = record["name"];
    assert(typeof name === "string", `${where}.name is not a string`);
    const shapeRecord = recordAt(record, "shape", where);
    const shape: GatedDeltaShape = {
      B: numberAt(shapeRecord, "B", where),
      T: numberAt(shapeRecord, "T", where),
      Hk: numberAt(shapeRecord, "Hk", where),
      Hv: numberAt(shapeRecord, "Hv", where),
      dk: numberAt(shapeRecord, "dk", where),
      dv: numberAt(shapeRecord, "dv", where),
    };
    const { B, T, Hk, Hv, dk, dv } = shape;
    const tensor = (slot: string, dims: readonly number[]): RefTensor => ({
      dtype: "f32",
      shape: dims,
      data: f32(`${name}.${slot}`, dims),
    });
    const upstream = recordAt(record, "upstream", where);
    const band = recordAt(record, "band", where);
    const oShape = [B, T, Hv, dv];
    const stored = record["stored_outputs"] === true;
    const oracle = (): Float64Array<ArrayBuffer> => {
      const tail = f32(`${name}.o_oracle_tail`, oShape);
      return Float64Array.from(
        f32(`${name}.o_oracle_head`, oShape),
        (head, element) => head + tail[element],
      );
    };
    return {
      name,
      shape,
      inputs: [
        tensor("q", [B, T, Hk, dk]),
        tensor("k", [B, T, Hk, dk]),
        tensor("v", [B, T, Hv, dv]),
        tensor("g", [B, T, Hv]),
        tensor("beta", [B, T, Hv]),
      ],
      upstream: {
        chunk: metricsOf(recordAt(upstream, "chunk", where), `${where}.upstream.chunk`),
        recurrent: metricsOf(recordAt(upstream, "recurrent", where), `${where}.upstream.recurrent`),
      },
      band: { maxAbs: numberAt(band, "max_abs", where), relL2: numberAt(band, "rel_l2", where) },
      oracle: fingerprintOf(recordAt(record, "oracle", where), `${where}.oracle`),
      ...(stored
        ? {
            stored: {
              chunk: f32(`${name}.o_chunk`, oShape),
              recurrent: f32(`${name}.o_recurrent`, oShape),
              oracle: oracle(),
            },
          }
        : {}),
    };
  });
};

/**
 * max_abs, rel_l2 and max_rel_head of `x` against `reference` over o `[B,T,Hv,dv]` (value head =
 * axis 2). A non-finite element makes every metric infinite (an unwritten NaN-poisoned element
 * counts as non-finite).
 */
export const errorMetrics = (
  x: ArrayLike<number>,
  reference: ArrayLike<number>,
  valueHeads: number,
  valueDim: number,
): ErrorMetrics => {
  assert(x.length === reference.length, `length ${x.length} against ${reference.length}`);
  const headError = new Float64Array(valueHeads);
  const headReference = new Float64Array(valueHeads);
  let squaredError = 0;
  let squaredReference = 0;
  let maxAbs = 0;
  let nonFinite = 0;
  for (let index = 0; index < x.length; index += 1) {
    const value = x[index];
    const expected = reference[index];
    if (!Number.isFinite(value)) {
      nonFinite += 1;
      continue;
    }
    const error = Math.abs(value - expected);
    const head = Math.floor(index / valueDim) % valueHeads;
    squaredError += error * error;
    squaredReference += expected * expected;
    maxAbs = Math.max(maxAbs, error);
    headError[head] = Math.max(headError[head], error);
    headReference[head] = Math.max(headReference[head], Math.abs(expected));
  }
  let maxRelHead = 0;
  for (let head = 0; head < valueHeads; head += 1) {
    if (headReference[head] > 0) {
      maxRelHead = Math.max(maxRelHead, headError[head] / headReference[head]);
    }
  }
  const bad = nonFinite > 0;
  return {
    maxAbs: bad ? Infinity : maxAbs,
    relL2: bad ? Infinity : Math.sqrt(squaredError / squaredReference),
    maxRelHead: bad ? Infinity : maxRelHead,
    nonFinite,
  };
};

/** The largest share of the band a result uses (> 1 = outside the band; ∞ when not finite). */
export const bandShare = (metrics: BandMetrics, band: BandMetrics): number =>
  Math.max(metrics.maxAbs / band.maxAbs, metrics.relL2 / band.relL2);

/** Spells a result against the band for a failure message. */
export const describeBand = (metrics: BandMetrics, band: BandMetrics): string =>
  `max_abs ${metrics.maxAbs.toExponential(3)} / band ${band.maxAbs.toExponential(3)}, ` +
  `rel_l2 ${metrics.relL2.toExponential(3)} / band ${band.relL2.toExponential(3)} ` +
  `(share ${bandShare(metrics, band).toFixed(3)})`;
