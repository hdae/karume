/**
 * The tensor list — the format-neutral input of a conversion. A file format reader (safetensors,
 * GGUF, a container read as tensors in the tests) gives, per tensor name, the element type, the
 * shape and a function that reads that tensor's bytes; nothing is read until a tensor is converted.
 */

import { decodeBf16, decodeF16 } from "@karume/runtime/container";

/** The element types of a source tensor (the safetensors spelling). */
export type TensorDtype = "F32" | "F16" | "BF16";

/** One source tensor. */
export type SourceTensor = {
  readonly dtype: TensorDtype;
  readonly shape: readonly number[];
  /**
   * The tensor's bytes: little endian, in row-major order, exactly `numel × element size` long.
   * Called when the tensor is converted, once per conversion.
   */
  read(): Promise<Uint8Array<ArrayBuffer>>;
};

/** Tensor name → tensor. */
export type TensorList = ReadonlyMap<string, SourceTensor>;

const ELEMENT_BYTES: Readonly<Record<TensorDtype, number>> = { F32: 4, F16: 2, BF16: 2 };

/** The byte length of a tensor of `numel` elements of `dtype`. */
export const tensorByteLength = (dtype: TensorDtype, numel: number): number =>
  numel * ELEMENT_BYTES[dtype];

/**
 * The tensor's values as f32 — exact for every source dtype. F32 bytes at a 4-byte aligned
 * position are viewed, not copied; F16 and BF16 go through the runtime's decoders (the only place
 * that interprets them; a NaN's payload is not kept).
 */
export const tensorValues = (
  dtype: TensorDtype,
  bytes: Uint8Array<ArrayBuffer>,
): Float32Array<ArrayBuffer> => {
  switch (dtype) {
    case "F32":
      return bytes.byteOffset % 4 === 0
        ? new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)
        : new Float32Array(bytes.slice().buffer);
    case "F16":
      return decodeF16(bytes);
    case "BF16":
      return decodeBf16(bytes);
  }
};
