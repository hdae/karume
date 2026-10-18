/**
 * CPU-side decode of i4 storage (group-symmetric packed 4-bit along K) (for the ineligible slots of
 * ADR 0069 / the host mirror of the GPU-side `unpack4xU8` + mask/shift + scale multiplication).
 *
 * MUST: decoding lives in this one place only (the same reason as format/i8.ts — if the load path
 * and the tests interpret i4 separately, a mix-up of the pack order 〈element 2i = low nibble / 2i+1 =
 * high, stored value u = q + 8〉 goes unnoticed). The source of truth for the pack order is the
 * exporter's `karume/emit.py: pack_int4`.
 *
 * Why it is bit-identical: the decoded value is `f32(q) · s` with **one f32 rounding** (`q ∈ [−7,7]`
 * is exact in f32 — the same argument as i8.ts). The exporter's round-trip check (bit identity of
 * `dequant(unpack(pack(w)))` — ADR 0069 decision 4 ③) uses the same formula, so the three
 * implementations TS / Python / WGSL produce the same numbers.
 */

import { groupScaleShape } from "./container/codecs.ts";

/**
 * Element count (the format layer does not depend on src/ops.ts — keeps this one line for the same
 * reason as i8.ts).
 */
const numel = (shape: readonly number[]): number => shape.reduce((count, dim) => count * dim, 1);

/** The decode contract was broken (mismatched byte length, group-shaped scale or group length). */
export class I4Error extends Error {
  override readonly name = "I4Error";
}

/**
 * Length of an i4 quantization row = the length with **the first dimension as rows and the rest
 * flattened** (`numel / shape[0]`).
 *
 * The `channel_rows` of every eligible op (exporter `karume/quantize.py`) has this form
 * (linear `[O,I]` → `I` / embedding `[V,D]` → `D` / conv1d `[O,Cin,K]` → `Cin·K`), and for
 * rank 2 weights it equals "the last dimension".
 */
const rowLength = (shape: readonly number[]): number => numel(shape) / shape[0];

/**
 * Decodes packed 4-bit bytes into f32 with a group-shaped scale.
 *
 * `scaleShape` is the rank 2 shape of {@link groupScaleShape} (row axis 0) (ADR 0069 decision 3 — a
 * different thing from i8's keepdim broadcast shape; their accepted sets do not overlap. A mix-up
 * fails loudly here — silently reading it as a broadcast would hand out the group scale as one
 * value per channel, a silent wrong value).
 *
 * MUST: nibbles are in **flat memory order** (`[O,Cin,K]` row-major = the same bytes as the flat
 * `[O, Cin·K]`). Row = first dimension, group = `groupSize` consecutive elements within a row; this
 * is the same formula as the GPU-side flat index (`arow·(k>>shift) + (ak0>>shift)`).
 */
export const decodeI4 = (
  bytes: Uint8Array<ArrayBuffer>,
  shape: readonly number[],
  scale: Float32Array<ArrayBuffer>,
  scaleShape: readonly number[],
  groupSize: number,
): Float32Array<ArrayBuffer> => {
  if (shape.length === 0) {
    throw new I4Error("i4 の重みに量子化軸が無い（rank 0）");
  }
  const count = numel(shape);
  if (count % 2 !== 0 || bytes.byteLength * 2 !== count) {
    throw new I4Error(
      `i4 ペイロード ${bytes.byteLength} バイトが shape [${shape.join(",")}] と違う` +
        `（numel / 2 = ${count / 2} バイトが要る）`,
    );
  }
  const width = rowLength(shape);
  if (!Number.isInteger(groupSize) || groupSize < 1 || width % groupSize !== 0) {
    throw new I4Error(`group_size ${groupSize} が量子化軸（行長）${width} を割り切らない`);
  }
  const expected = groupScaleShape(shape, 0, groupSize);
  const groups = expected[1];
  if (
    scaleShape.length !== expected.length ||
    scaleShape.some((dim, axis) => dim !== expected[axis])
  ) {
    throw new I4Error(
      `scale [${scaleShape.join(",")}] が重み [${shape.join(",")}] の group 形 [${expected.join(
        ",",
      )}]（group_size=${groupSize}）でない`,
    );
  }
  if (scale.length !== numel(scaleShape)) {
    throw new I4Error(
      `scale の要素数 ${scale.length} が shape [${scaleShape.join(",")}] と合わない`,
    );
  }
  const out = new Float32Array(count);
  const rows = width === 0 ? 0 : count / width;
  let i = 0;
  for (let row = 0; row < rows; row += 1) {
    const scaleBase = row * groups;
    for (let group = 0; group < groups; group += 1) {
      const s = scale[scaleBase + group];
      for (let element = 0; element < groupSize; element += 1, i += 1) {
        const byte = bytes[i >> 1];
        const u = (i & 1) === 1 ? byte >> 4 : byte & 0x0f;
        // MUST: round the product only once (the same as i8.ts — the GPU side does one f32
        // multiplication). Offset 8 is "the default when zero_point is omitted" (reservation ② of
        // ADR 0069 decision 3), paired with the pack side in emit.py.
        out[i] = Math.fround((u - 8) * s);
      }
    }
  }
  return out;
};
