// Helpers that build valid / deliberately broken safetensors binaries for the tests of the
// safetensors reader (`src/format/safetensors.ts`).
//
// Karume's distribution form moved to containers (`krm` / `krg`), so all this builds is the plain
// safetensors that **the readers of auxiliary assets** deal with (the extras of ADR 0038 §2, golden
// io). The "correct shape" the tests expect lives in one place, and each failure-case test breaks
// exactly one point of it.

export const f32Bytes = (values: ArrayLike<number>): Uint8Array<ArrayBuffer> => {
  const array = Float32Array.from(values);
  return new Uint8Array(array.buffer);
};

/** Packs the header JSON as raw text (the entry point for building broken JSON). */
export const packSafetensorsRaw = (headerText: string, data: Uint8Array): ArrayBuffer => {
  const headerBytes = new TextEncoder().encode(headerText);
  // Put the data section on an 8-byte boundary (the safetensors convention; the alignment check
  // also assumes it)
  const headerLength = headerBytes.length + ((8 - (headerBytes.length % 8)) % 8);
  const buffer = new ArrayBuffer(8 + headerLength + data.length);
  const bytes = new Uint8Array(buffer);
  new DataView(buffer).setBigUint64(0, BigInt(headerLength), true);
  bytes.set(headerBytes, 8);
  bytes.fill(0x20, 8 + headerBytes.length, 8 + headerLength);
  bytes.set(data, 8 + headerLength);
  return buffer;
};

/**
 * Packs the header table as is (unchecked, so a mismatch between declaration and data can be
 * built).
 */
export const packSafetensors = (header: Record<string, unknown>, data: Uint8Array): ArrayBuffer =>
  packSafetensorsRaw(JSON.stringify(header), data);

export type TensorSpec = {
  readonly name: string;
  readonly dtype: string;
  readonly shape: readonly number[];
  readonly data: Uint8Array;
};

/** Builds a consistent file packed without gaps. */
export const buildSafetensors = (
  tensors: readonly TensorSpec[],
  metadata?: Record<string, string>,
): ArrayBuffer => {
  const header: Record<string, unknown> = {};
  if (metadata !== undefined) header["__metadata__"] = metadata;
  let offset = 0;
  for (const tensor of tensors) {
    header[tensor.name] = {
      dtype: tensor.dtype,
      shape: tensor.shape,
      data_offsets: [offset, offset + tensor.data.length],
    };
    offset += tensor.data.length;
  }
  const data = new Uint8Array(offset);
  let cursor = 0;
  for (const tensor of tensors) {
    data.set(tensor.data, cursor);
    cursor += tensor.data.length;
  }
  return packSafetensors(header, data);
};
