/**
 * safetensors **writing** (a minimal implementation for the dumps of tests and demos).
 *
 * The safetensors of `@karume/runtime` is read-only (the runtime has no use for writing). Demos
 * have to hand "the discrete inputs and the random sequence itself" to the torch reference, and we
 * want that transport to use the existing asset format rather than a new one, so it lives here. It
 * is part of neither the runtime's public surface (ADR 0008) nor that of `@karume/models`
 * (`publish.exclude` in `deno.json` drops `tests/`).
 *
 * MUST: it lives on the package side (here). Placed on the demo side, the models tests would
 * import `examples/` — a **reverse dependency** (CLAUDE.md's one-directional dependencies). The
 * demo (`examples/sbv2/dump.ts`) borrows it in the forward direction, examples → packages.
 *
 * Layout: `[u64 LE header length][header JSON][data section]`. `data_offsets` are relative to the
 * start of the data section, and the header is padded with spaces so that the data section starts
 * on an 8-byte boundary (the same as the reference implementation).
 */

const ALIGNMENT = 8;

/**
 * The element types that can be written (one to one with the semantic dtypes — bool is not used, so
 * it is absent).
 */
export type DumpTensor =
  | { readonly dtype: "F32"; readonly shape: readonly number[]; readonly data: Float32Array }
  | { readonly dtype: "I32"; readonly shape: readonly number[]; readonly data: Int32Array }
  | { readonly dtype: "I8"; readonly shape: readonly number[]; readonly data: Int8Array };

const elementCount = (shape: readonly number[]): number =>
  shape.reduce((product, dim) => product * dim, 1);

export const writeSafetensors = (
  tensors: ReadonlyMap<string, DumpTensor>,
  metadata: Readonly<Record<string, string>>,
): Uint8Array<ArrayBuffer> => {
  const header: Record<string, unknown> = { __metadata__: metadata };
  const chunks: Uint8Array[] = [];
  let offset = 0;
  for (const [name, tensor] of tensors) {
    if (elementCount(tensor.shape) !== tensor.data.length) {
      throw new Error(
        `tensor '${name}': the element count of shape [${tensor.shape}] differs from the data length ${tensor.data.length}`,
      );
    }
    const bytes = new Uint8Array(
      tensor.data.buffer,
      tensor.data.byteOffset,
      tensor.data.byteLength,
    );
    chunks.push(bytes);
    header[name] = {
      dtype: tensor.dtype,
      shape: [...tensor.shape],
      data_offsets: [offset, offset + bytes.byteLength],
    };
    offset += bytes.byteLength;
  }

  const encoder = new TextEncoder();
  const headerBytes = encoder.encode(JSON.stringify(header));
  // Puts the start of the data section (8 + header length) on an 8-byte boundary. The padding is
  // whitespace, harmless in JSON.
  const padding = (ALIGNMENT - ((headerBytes.byteLength + ALIGNMENT) % ALIGNMENT)) % ALIGNMENT;
  const headerLength = headerBytes.byteLength + padding;

  const out = new Uint8Array(ALIGNMENT + headerLength + offset);
  new DataView(out.buffer).setBigUint64(0, BigInt(headerLength), true);
  out.set(headerBytes, ALIGNMENT);
  out.fill(0x20, ALIGNMENT + headerBytes.byteLength, ALIGNMENT + headerLength);
  let cursor = ALIGNMENT + headerLength;
  for (const chunk of chunks) {
    out.set(chunk, cursor);
    cursor += chunk.byteLength;
  }
  return out;
};
