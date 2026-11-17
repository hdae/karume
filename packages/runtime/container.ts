/**
 * The format layer for a package that produces container bytes (`@karume/runtime/container`).
 *
 * A converter that serves or writes a `krm` it did not read from a file — weights encoded while a
 * Session reads them, or written out as a new container — needs a few of the format's own rules.
 * This entry exposes them; it does not copy them, so the format keeps one owner (docs/container-v1.md):
 *
 * - {@link serializeModelDescriptor} is the canonical serialization of the model descriptor (§2.2,
 *   the rules of docs/ir-v2.md "Canonical serialization"). A descriptor read from a container and
 *   serialized again gives the same bytes when the writer wrote canonically, so a block's or a
 *   part's sha256 can be replaced without moving any other byte (a sha256 is always 64 characters).
 * - {@link sha256Hex} is the digest every block and part sha256 is (Web Crypto, one call over the
 *   whole input); {@link asSha256Hex} checks a string is one (64 lowercase hex characters) and gives
 *   it the descriptor's type.
 * - {@link decodeF16} and {@link decodeBf16} are the runtime's only f16 / bf16 → f32 decoders (both
 *   exact; NaN payloads are not kept), so a converter reads 16-bit source values the way the runtime
 *   reads stored ones.
 * - The types: the model descriptor and its block records, and the merge layer's supply plan of one
 *   initializer (`OpenedContainer.graphs[name].supplies`), which says which blocks hold its data and
 *   its scale, at which rows and with how many payload bytes.
 *
 * Reading a container (the header, the two documents, the merge) stays `openContainer` of
 * `@karume/runtime`.
 *
 * This entry has no side effects and reads no WebGPU global at import, so it can be imported where
 * WebGPU does not exist.
 *
 * @module
 */

/** Canonical serialization of the model descriptor, and the sha256 of the format. */
export {
  asSha256Hex,
  serializeModelDescriptor,
  sha256Hex,
} from "./src/format/container/descriptor.ts";
/** The model descriptor, its block records and the type of a sha256 field. */
export type {
  DataBlockRecord,
  ModelDescriptor,
  Sha256Hex,
} from "./src/format/container/descriptor.ts";
/** The merge layer's supply plan of one initializer, and one block of it. */
export type { InitializerSupply, SupplyBlock } from "./src/format/container/bind.ts";
/** The f16 and bf16 → f32 decoders. */
export { decodeBf16 } from "./src/format/bf16.ts";
export { decodeF16 } from "./src/format/f16.ts";
