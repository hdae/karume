/**
 * The safetensors reader of Karume (`@karume/safetensors`).
 *
 * {@link openSafetensors} opens a file from a `Blob` (a browser `File` is one) or a ranged reader
 * by reading its 8-byte length and its header only, checks the whole tensor table, and gives every
 * tensor's dtype, shape, byte length and a function that reads its bytes with one ranged read — so
 * a file too large to hold in memory is read one tensor at a time. It does not interpret values.
 *
 * No dependency; Web standard APIs only; no side effects at import.
 *
 * ```ts ignore
 * import { openSafetensors } from "@karume/safetensors";
 *
 * // file: a browser `File`, any `Blob`, or `{ size, read(offset, length) }`.
 * const { tensors } = await openSafetensors(file);
 * const [first] = tensors.values();
 * const bytes = await first?.read();
 * ```
 *
 * @module
 */

export { openSafetensors, SafetensorsError } from "./src/read.ts";
export type {
  RangedReader,
  SafetensorsDtype,
  SafetensorsFile,
  SafetensorsSource,
  SafetensorsTensor,
} from "./src/read.ts";
