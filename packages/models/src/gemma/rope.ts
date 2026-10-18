/**
 * **Host-side generation** of the cos / sin rows of Gemma 4's RoPE (rotary position embedding).
 *
 * ## Why the host builds them
 *
 * The distribution used to bake four cos / sin tables for positions 0..maxPosition−1 (2 layer types
 * × cos / sin) as IR initializers, and an in-graph `embedding` (row gather) looked them up with
 * `position_ids`. A table is a pure function `f(position)` independent per position, so if the host
 * builds only the positions in the chunk and passes them as **ordinary graph inputs**, the tables
 * are needed neither in VRAM nor in the distribution, and the maximum conversation length
 * (`capacity`) is freed from the table's row count and becomes a runtime option. It fits the same
 * slot as PLE's host gather (`ple.ts`): "a derived input the host builds only for the chunk".
 *
 * ## Formula (canonical here; the exporter's `tools/export-recipes/gemma4/rope.py` is its mirror)
 *
 * For a layer type `{theta, headDim, rotaryDim}`, with `half = headDim / 2` and
 * `rotated = rotaryDim / 2`:
 *
 *     invFreq[i] = theta ** (−(2·i) / headDim)   (0 ≤ i < rotated)
 *     invFreq[i] = 0                              (rotated ≤ i < half — dimensions not rotated)
 *     angle[j]   = position × invFreq[j mod half] (0 ≤ j < headDim — both halves share the angle)
 *     cos[j] = cos(angle[j]) / sin[j] = sin(angle[j])
 *
 * The layout equals upstream's (transformers `Gemma4TextRotaryEmbedding`)
 * `emb = cat(freqs, freqs)`, and default (sliding: rotaryDim = headDim) and proportional (full:
 * rotaryDim = the partial_rotary_factor share) fold into **the same single formula**.
 * attention_scaling is 1 for both rope_types, so it is not applied.
 *
 * ## Numerics contract of regular Gemma
 *
 * - Computed in **f64**, rounded to f32 once on storage. Upstream is f32 throughout (the angle
 *   `position × invFreq` is accumulated in f32, so the angle gets coarser at larger positions — ULP
 *   ≈ 0.008 rad at 131,071), so it is **not bit-identical to upstream's table** (the same wall as
 *   ADR 0034 "Detection limits" item 2 — torch's f32 trigonometric functions also carry SLEEF's
 *   1-ULP error). This side is the mathematically exact one, and the comparison uses a tolerance
 *   proportional to the position (`tests/gemma4_rope_test.ts` — comparison with a fixture of the
 *   upstream module's real output + fault injection).
 * - Determinism: f64 addition, subtraction, multiplication and division are IEEE 754 and
 *   engine-independent. Only `Math.cos` / `Math.sin` may differ by 1 f64 ULP between engines, but
 *   the probability that it survives rounding to f32 is about 2⁻²⁹ per element (an expected 0.4
 *   elements even over a full 128K × 1,536-element table), and it is treated like the GPU-side
 *   1-ULP group (the Metal section of known-issues).
 * - Dimensions not rotated (`invFreq = 0`) have angle 0, so cos = 1 / sin = 0 come out **exactly**.
 *
 * The QAT-only entry rounds the frequency power, the reciprocal and the position product to f32 at
 * each step (ADR 0097 addendum 5). Full bit identity with upstream's trigonometric functions and
 * reductions is not guaranteed.
 *
 * MUST: pad rows (`position = 0`) are filled with the same formula as ordinary rows (cos = 1 / sin
 * = 0). The values of pad rows are meaningless by contract (ADR 0066 addendum of 2026-08-17,
 * item 6), but passing them uninitialized could let NaN into the attention of pad rows, with NaN
 * classification acting outside the structural guarantee "empty row → exact 0" (ADR 0067
 * decision 6).
 */

import type { Tensor } from "@karume/runtime";

/** Layer type names (upstream's `layer_types` spelling as is — part of the graph input names). */
export const GEMMA4_ROPE_LAYER_TYPES = ["sliding_attention", "full_attention"] as const;

export type Gemma4RopeLayerType = (typeof GEMMA4_ROPE_LAYER_TYPES)[number];

/**
 * RoPE parameters of one layer type (declared by the distribution's
 * `pipelineConfig.rope.<layerType>`).
 */
export type Gemma4RopeLayerSpec = {
  /** Frequency base (upstream `rope_theta`). */
  readonly theta: number;
  /** Head dimension `D` (= the width of the generated row). */
  readonly headDim: number;
  /**
   * Number of rotated dimensions (`headDim` for default, `2 · int(factor · headDim / 2)` for
   * proportional). Even.
   */
  readonly rotaryDim: number;
};

/** Both layer types. */
export type Gemma4RopeSpec = {
  readonly [layerType in Gemma4RopeLayerType]: Gemma4RopeLayerSpec;
};

/** The two generated parts, cos / sin. */
export const GEMMA4_ROPE_PARTS = ["cos", "sin"] as const;

export type Gemma4RopePart = (typeof GEMMA4_ROPE_PARTS)[number];

/**
 * Graph input name (**the same spelling** as the exporter's
 * `tools/export-recipes/gemma4/export_decode.py`).
 *
 * MUST: this is the only place that generates it (the distribution side's check
 * `GEMMA4_GRAPH_INPUTS` and the TS derived-input names come from the same string).
 */
export const gemma4RopeInputName = (layerType: Gemma4RopeLayerType, part: Gemma4RopePart): string =>
  `rope_${layerType}_${part}`;

/** The four graph input names (in the fixed order layer type × part). */
export const gemma4RopeInputNames = (): string[] =>
  GEMMA4_ROPE_LAYER_TYPES.flatMap((layerType) =>
    GEMMA4_ROPE_PARTS.map((part) => gemma4RopeInputName(layerType, part)),
  );

/**
 * Value-range check of the layer-type parameters.
 *
 * MUST: `rotaryDim` is even with `2 ≤ rotaryDim ≤ headDim`, and `headDim` is even. An odd value
 * breaks the "first half = second half" layout and silently builds a table different from
 * upstream's. `theta` is a positive finite value (rejecting the shape where `theta ** negative`
 * falls to 0 / Inf).
 */
const assertGemma4RopeLayerSpec = (where: string, spec: Gemma4RopeLayerSpec): void => {
  const { theta, headDim, rotaryDim } = spec;
  if (!Number.isFinite(theta) || theta <= 0) {
    throw new Error(`${where}: theta は正の有限値（${theta}）`);
  }
  if (!Number.isInteger(headDim) || headDim < 2 || headDim % 2 !== 0) {
    throw new Error(`${where}: headDim は 2 以上の偶数（${headDim}）`);
  }
  if (!Number.isInteger(rotaryDim) || rotaryDim < 2 || rotaryDim % 2 !== 0 || rotaryDim > headDim) {
    throw new Error(
      `${where}: rotaryDim は 2 以上 headDim 以下の偶数（${rotaryDim} / headDim ${headDim}）`,
    );
  }
};

export const assertGemma4RopeSpec = (where: string, spec: Gemma4RopeSpec): void => {
  for (const layerType of GEMMA4_ROPE_LAYER_TYPES) {
    assertGemma4RopeLayerSpec(`${where}.${layerType}`, spec[layerType]);
  }
};

/**
 * Inverse frequencies `invFreq[0 .. headDim/2)`. Dimensions not rotated are 0. Only QAT rounds to
 * f32 at each step.
 *
 * NOTE: `theta ** x` is `Math.pow` in f64. Upstream computes `1 / base ** t` in f32, so there is a
 * relative difference around 1e-7, but this difference is intended (the numerics contract above).
 */
const ropeInverseFrequencies = (
  spec: Gemma4RopeLayerSpec,
  qat: boolean,
): Float64Array<ArrayBuffer> => {
  assertGemma4RopeLayerSpec("gemma4 rope", spec);
  const half = spec.headDim / 2;
  const rotated = spec.rotaryDim / 2;
  const invFreq = new Float64Array(half);
  for (let i = 0; i < rotated; i += 1) {
    invFreq[i] = qat
      ? Math.fround(1 / Math.fround(Math.pow(spec.theta, Math.fround((2 * i) / spec.headDim))))
      : Math.pow(spec.theta, -(2 * i) / spec.headDim);
    if (qat && (!Number.isFinite(invFreq[i]) || invFreq[i] <= 0)) {
      throw new Error("gemma4-qat rope: f32 逆周波数が正の有限数でない");
    }
  }
  return invFreq;
};

/** Row-major `[rows, headDim]` cos / sin (f32). */
export type Gemma4RopeRows = {
  readonly cos: Float32Array<ArrayBuffer>;
  readonly sin: Float32Array<ArrayBuffer>;
};

/**
 * Positions → cos / sin rows (`[positions.length, headDim]`).
 *
 * MUST: positions are non-negative integers (u32 logical positions). Negative or non-integer values
 * leave the formula undefined, so they fail loudly.
 */
const ropeRows = (
  spec: Gemma4RopeLayerSpec,
  positions: ArrayLike<number>,
  qat: boolean,
): Gemma4RopeRows => {
  const invFreq = ropeInverseFrequencies(spec, qat);
  const width = spec.headDim;
  const half = width / 2;
  const rows = positions.length;
  const cos = new Float32Array(rows * width);
  const sin = new Float32Array(rows * width);
  for (let row = 0; row < rows; row += 1) {
    const position = positions[row];
    if (!Number.isInteger(position) || position < 0) {
      throw new Error(`gemma4 rope: 位置は非負整数（row ${row}: ${position}）`);
    }
    const base = row * width;
    for (let i = 0; i < half; i += 1) {
      const angle = qat ? Math.fround(Math.fround(position) * invFreq[i]) : position * invFreq[i];
      // The trigonometric results are rounded to f32 on assignment (first and second half share the
      // value).
      const c = Math.cos(angle);
      const s = Math.sin(angle);
      cos[base + i] = c;
      cos[base + half + i] = c;
      sin[base + i] = s;
      sin[base + half + i] = s;
    }
  }
  return { cos, sin };
};

/**
 * Positions of one physical chunk → the four graph inputs (`[1, rows, headDim]` f32).
 *
 * `positions` has one entry per physical row (pad rows are 0 — `sequence.ts` fills them by the same
 * convention as `input_ids`).
 */
const ropeInputs = (
  spec: Gemma4RopeSpec,
  positions: ArrayLike<number>,
  qat: boolean,
): Record<string, Tensor> => {
  const inputs: Record<string, Tensor> = {};
  for (const layerType of GEMMA4_ROPE_LAYER_TYPES) {
    const layer = spec[layerType];
    const rows = ropeRows(layer, positions, qat);
    const shape = [1, positions.length, layer.headDim];
    inputs[gemma4RopeInputName(layerType, "cos")] = {
      dtype: "f32",
      shape,
      data: rows.cos,
    };
    inputs[gemma4RopeInputName(layerType, "sin")] = {
      dtype: "f32",
      shape,
      data: rows.sin,
    };
  }
  return inputs;
};

/** Regular Gemma's inverse frequencies (f64). Dimensions not rotated are 0. */
export const gemma4RopeInverseFrequencies = (
  spec: Gemma4RopeLayerSpec,
): Float64Array<ArrayBuffer> => ropeInverseFrequencies(spec, false);
/** Builds cos / sin rows from regular Gemma's positions, computed in f64 and stored as f32. */
export const gemma4RopeRows = (
  spec: Gemma4RopeLayerSpec,
  positions: ArrayLike<number>,
): Gemma4RopeRows => ropeRows(spec, positions, false);
/** The four RoPE inputs for a physical chunk of regular Gemma. Pad positions are 0. */
export const gemma4RopeInputs = (
  spec: Gemma4RopeSpec,
  positions: ArrayLike<number>,
): Record<string, Tensor> => ropeInputs(spec, positions, false);
/**
 * Per-step f32 rounding, for fixed QAT only. Builds the four RoPE inputs of a physical chunk
 * including pad.
 * DECIDED: [ADR 0097 addendum 5](../../../../docs/decisions/0097-gemma4-qat-integration.md#addendum-5-2026-09-11--fixed-qat-recipe-and-how-numeric-comparisons-are-treated).
 */
export const gemma4QatRopeInputs = (
  spec: Gemma4RopeSpec,
  positions: ArrayLike<number>,
): Record<string, Tensor> => ropeInputs(spec, positions, true);
