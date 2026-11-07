/**
 * **Physical storage of a full state slot** (`f32` / `f16`) — the generation part shared by
 * `state_append` and the states-form attention kernels that read a slot. Enabled by the opt-in
 * `SessionOptions.fullStateStorage`.
 *
 * The form is the f16 storage of S ({@link "./score-storage.ts"}) on a longer-lived buffer: the slot
 * is bound as **`array<u32>`** holding 2 elements per word (core WGSL, no optional feature, so it
 * composes with every compute form). The graph keeps declaring its states as f32: this is storage,
 * not the semantic type.
 *
 * ## Rounding happens once, at the append
 *
 * {@link stateSlotPackWgsl} in the f16 `state_append` is the only rounding point, and
 * `unpack2x16float` is exact. So the contract is
 *
 *     output(f16) ≡ output(f32 kernel over a slot rounded to f16 on the host)     … not one bit differs
 *
 * and the current step's rows, read from `ins`, stay f32: a token sees its own run in f32 and earlier
 * runs in f16.
 *
 * MUST NOT round with `pack2x16float`: WGSL leaves its rounding open, and D3D12 rounds toward zero, so
 * the slots of the two backends would differ by up to one f16 ulp.
 *
 * ## Only full slots
 *
 * A sliding slot's cost is fixed by its window, so halving it buys little and every sliding kernel
 * would need a second form for it.
 *
 * MUST: the output of the f32 form stays **byte-identical to before**. The key part and the loader are
 * the empty string for f32, and each read expression is the string it was. The snapshots
 * (`tests/fixtures/wgsl/*.wgsl`) and the pipeline cache depend on this.
 *
 * DECIDED: docs/decisions/0143-f16-full-state-storage.md
 */

import { CodegenError } from "../codegen/errors.ts";

/** Storage of one element of a full slot. The semantics are f32 in both (ADR 0006). */
export type StateStorage = "f32" | "f16";

/**
 * The physical storage of one slot under {@link SessionOptions.fullStateStorage}.
 *
 * MUST: **the only decision** of the per-slot storage. The recipe builders, the allocation and the
 * estimate all call it: a split between them would bind a packed slot to an f32 kernel with matching
 * shapes and no exception.
 */
export const slotStateStorage = (option: StateStorage, sliding: boolean): StateStorage =>
  option === "f16" && !sliding ? "f16" : "f32";

/** Bytes per element. MUST: the allocation and the estimate both take the size from here. */
export const stateStorageBytes = (storage: StateStorage): number => (storage === "f16" ? 2 : 4);

/**
 * Discriminator put last in the pipeline key. MUST: the empty string for f32 (existing keys do not
 * move by a single character).
 */
export const stateStorageKeyPart = (storage: StateStorage): string =>
  storage === "f16" ? ":kv16" : "";

/** Note appended to the leading WGSL comment (empty for f32 — keeps the existing bytes). */
export const stateStorageNote = (storage: StateStorage): string =>
  storage === "f16" ? ", KV in f16 storage (unpack2x16float)" : "";

/** WGSL element type of a slot binding (the flat word array for f16). */
export const stateSlotElementType = (storage: StateStorage): string =>
  storage === "f16" ? "u32" : "f32";

/**
 * Element read function of the f16 form, named `<name>_at` (takes a **flat element index**, so the
 * caller's index arithmetic does not change). Starts with the blank line that separates it from what
 * it follows (the empty string for f32).
 *
 * MUST: the word and the half are chosen by the parity of the flat index (ADR 0018's trap: the parity
 * relative to a row matches only by accident).
 */
export const stateSlotLoaderWgsl = (name: string, storage: StateStorage): string =>
  storage === "f16"
    ? `

// Decode of f16 storage: element i = unpack2x16float(${name}[i / 2])[i % 2] (unpack is exact)
fn ${name}_at(i: u32) -> f32 {
  let pair = unpack2x16float(${name}[i >> 1u]);
  return select(pair.x, pair.y, (i & 1u) == 1u);
}`
    : "";

/**
 * The rounding of the f16 `state_append`: f32 bits to f16 bits, to nearest with ties to even, the
 * same as `Math.f16round`. Integer operations only, on bits the kernel reads as `u32`, so no backend's
 * float handling (flushed subnormals, a rounding mode) reaches the slot.
 *
 * A carry out of the mantissa is the right next value, so rounding up from the largest finite value
 * gives infinity without a case of its own.
 */
export const stateSlotPackWgsl =
  (): string => `// x >> shift, to nearest with ties to even (shift >= 1)
fn round_shift(x: u32, shift: u32) -> u32 {
  let kept = x >> shift;
  let rest = x & ((1u << shift) - 1u);
  let half = 1u << (shift - 1u);
  return kept + u32(rest > half || (rest == half && (kept & 1u) == 1u));
}

// **The only rounding point** of the slot: f32 bits to f16 bits
fn f16_bits(f: u32) -> u32 {
  let sign = (f >> 16u) & 0x8000u;
  let magnitude = f & 0x7fffffffu;
  if (magnitude > 0x7f800000u) {
    return sign | 0x7e00u; // NaN stays a quiet NaN
  }
  if (magnitude >= 0x47800000u) {
    return sign | 0x7c00u; // from 2^16 up, infinity
  }
  if (magnitude >= 0x38800000u) {
    return sign | round_shift(magnitude - 0x38000000u, 13u); // normal: the exponent rebased from 127 to 15
  }
  if (magnitude < 0x33000000u) {
    return sign; // below 2^-25 (f32 subnormals included) rounds to zero
  }
  // Subnormal in f16: the f32 significand in units of 2^-24
  return sign | round_shift((magnitude & 0x7fffffu) | 0x800000u, 126u - (magnitude >> 23u));
}`;

/** Read expression of one slot element (flat element index). */
export const stateSlotRead = (name: string, index: string, storage: StateStorage): string =>
  storage === "f16" ? `${name}_at(${index})` : `${name}[${index}]`;

/** Rejects f16 storage on a form that does not have it (does not silently generate the f32 form). */
export const assertStateStorageFull = (
  op: string,
  sliding: boolean,
  storage: StateStorage,
): void => {
  if (storage === "f16" && sliding) {
    throw new CodegenError(`${op}: f16 KV storage is for full slots only (this slot is sliding)`);
  }
};

/**
 * MUST: an even `D`. One append invocation owns one word, so it writes both halves at once; with an
 * odd `D` the last element of a row would share its word with the first of the next, and writing half
 * a word is a read-modify-write that loses values silently.
 */
export const assertStateStorageDepth = (op: string, depth: number, storage: StateStorage): void => {
  if (storage === "f16" && depth % 2 !== 0) {
    throw new CodegenError(`${op}: f16 KV storage needs an even depth (D = ${depth})`);
  }
};
