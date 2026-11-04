// Builders for i8 storage tests (host-side f32 → per-channel int8 encoding).
//
// The encoding is **test-only** (in production the exporter encodes, on the Python side, and the
// runtime only decodes). This is the tool that builds "weights exactly representable as i8 +
// per-channel scale", and it reproduces ADR 0006's fake-quant on the host:
//   bytes  = the quantized stored bytes (passed to the GPU / load path)
//   scale  = scale per channel (carried in the distribution form as an F32 tensor of keepdim shape)
//   values = those bytes decoded back to f32 **by the production decodeI8** (the expected values
//            the CPU reference uses)
// MUST: the expected values come from decodeI8, not from the encoder. Building both sides with the
// encoder would carry an encoder bug into the expected values just the same, making the check
// tautological.

import { decodeI8 } from "../../src/format/i8.ts";

/** Smallest normal f32 (`torch.finfo(float32).tiny` — used as the lower clamp of the scale). */
export const F32_TINY = 1.1754943508222875e-38;

/** Sequence of signed 8-bit values → raw bytes (only the low 8 bits are used). */
export const i8BytesFrom = (values: readonly number[]): Uint8Array<ArrayBuffer> =>
  Uint8Array.from(values, (value) => value & 0xff);

export type QuantizedI8 = {
  /** Raw I8 bytes to put in safetensors. */
  readonly bytes: Uint8Array<ArrayBuffer>;
  /** Scale per channel (flat = channel order). */
  readonly scale: Float32Array<ArrayBuffer>;
  /** Keepdim shape of the scale tensor. */
  readonly scaleShape: readonly number[];
  /** The quantized values (fake-quant — the CPU reference uses these as the weights). */
  readonly values: Float32Array<ArrayBuffer>;
};

/**
 * Per-channel symmetric int8 fake-quant (ADR 0019).
 *
 * `scale = max(amax / 127, tiny)` / `q = round(w / scale)` **clamped to ±127**
 * (−128 is not used, so the element with the largest magnitude is restored exactly). Rounding is
 * half-away-from-zero, which differs strictly from the torch side (half-to-even), but **the
 * expected values come from decodeI8**, so this does not affect what the runtime is checked
 * against (the production encoder lives in the exporter).
 */
export const quantizeI8 = (
  values: ArrayLike<number>,
  shape: readonly number[],
  axis: number,
): QuantizedI8 => {
  const channels = shape[axis];
  const inner = shape.slice(axis + 1).reduce((count, dim) => count * dim, 1);
  const channelOf = (index: number): number => Math.floor(index / inner) % channels;

  const amax = new Float32Array(channels);
  for (let i = 0; i < values.length; i += 1) {
    const channel = channelOf(i);
    amax[channel] = Math.max(amax[channel], Math.abs(values[i]));
  }
  const scale = new Float32Array(channels);
  for (let c = 0; c < channels; c += 1) scale[c] = Math.max(Math.fround(amax[c] / 127), F32_TINY);

  const quantized: number[] = [];
  for (let i = 0; i < values.length; i += 1) {
    const ratio = values[i] / scale[channelOf(i)];
    const rounded = Math.sign(ratio) * Math.round(Math.abs(ratio));
    quantized.push(Math.max(-127, Math.min(127, rounded)));
  }
  const bytes = i8BytesFrom(quantized);
  const scaleShape = shape.map((_, index) => (index === axis ? channels : 1));
  return { bytes, scale, scaleShape, values: decodeI8(bytes, shape, scale, scaleShape) };
};
