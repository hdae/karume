/**
 * Weight storage variants (`w=f32` / `w=f16` / `w=bf16` / `w=i8` are common to the five fused
 * kernels; `w=i4` / `w=i2` only for the ops that have a decode path — ADR 0069 / 0097) — the one
 * generator part shared by the five fused kernels (linear / conv1d / conv2d / conv_transpose1d /
 * embedding) (ADR 0018 / 0019).
 *
 * The f16 variant binds the weight buffer as **`array<u32>`** and decodes two elements at a time
 * with `unpack2x16float` (core WGSL, no optional feature). The value at index `i` is
 * `unpack2x16float(w[i >> 1])[i & 1]`; only **the parity of the flat index** picks the half.
 *
 * The bf16 variant (ADR 0131) uses the same `array<u32>` binding and the same parity selection as
 * f16; only the decode differs: bf16 is exactly the upper 16 bits of an f32, so the lower / upper
 * half of the word is turned back into an f32 with `bitcast<f32>` (`word << 16u` /
 * `word & 0xFFFF0000u`). The conversion is always exact (no rounding, no exponent-range remapping).
 *
 * The i8 variant is also bound as `array<u32>`; `unpack4xI8` extracts four elements from one word.
 * The value is `f32(unpack4xI8(w[i >> 2])[i & 3]) * scale[output channel]` — **the scale is not
 * moved out of the reduction; it is applied per element** (ADR 0019: the `(Σ x·q)·s` form saves
 * multiplies but loses bit equality with the CPU decode). The scale is **an extra binding paired
 * with the weight** ({@link weightLoaderWgsl} emits its declaration), a flat array of one f32 per
 * output channel. The i4 / i2 variants also use the `array<u32>` binding, the per-element scale
 * multiply and the scale binding paired with the weight; they differ only in packing and in scale
 * granularity (i2 per output channel, i4 per group).
 *
 * MUST: the word and the position within it are derived from the **flat index**. Taking the parity
 * (f16) or the remainder mod 4 (i8) of the index relative to the row gives matching numbers by
 * accident only when the row length is a multiple of 2 / 4 (the row start then always falls on a
 * word boundary). Tests with odd row lengths / row lengths not a multiple of 4 are the only
 * detector (the MUST notes of tests/gpu_f16_i8_weights_test.ts).
 *
 * MUST: the output of the `w=f32` / `w=f16` variants stays **byte-identical to before**. The
 * storage discriminator in the key is the empty string for f32, and the WGSL insertion points
 * ({@link weightLoaderWgsl} / {@link weightScaleWgsl}) return the same strings as before for f32
 * and f16 respectively. The existing snapshots (tests/fixtures/wgsl/*.wgsl) and the identity of the
 * pipeline cache depend on this.
 *
 * **The quad version** (`quad: true`) is an opt-in used only by GEMM's v4 path
 * (src/kernels/gemm.ts); it decodes four elements together into a `vec4<f32>`. MUST: the default is
 * `false` — adding it unconditionally breaks the byte invariance above and moves the output of
 * conv1d / conv2d / conv_transpose1d / embedding all at once.
 */

import { CodegenError } from "../codegen/errors.ts";

/**
 * Storage dtype of a weight slot. The semantics are always f32 (compute is f32 — ADR 0006).
 *
 * NOTE: the codec layout `i4f16` (`int4-sym-g-f16`) is deliberately not a member — its f16 steps are
 * read by `expert_linear` only (ADR 0145 decision 3), and the residency planner refuses it for every
 * consumer of this type (`planWeightResidency` in runtime/weight-residency.ts).
 */
export type WeightStorage = "f32" | "f16" | "bf16" | "i8" | "i4" | "i2";

/**
 * Every variant **shared** by the five fused kernels (an enumeration so the snapshots and the
 * fallback harness cover them mechanically).
 *
 * INT2 is also linear / embedding only, so this common enumeration leaves it out (ADR 0097).
 * MUST: `i4` is not listed — the i4 execution path is **linear / embedding / conv1d's implicit GEMM
 * (groups == 1) only** (ADR 0069 decision 5 and its embedding / conv1d addenda), and each generator
 * of the remaining entry points (conv1d direct kernel / conv2d / conv_transpose1d) rejects i4.
 * Besides, the i4 variant bakes the group length into the WGSL (paired with the `g` part of the
 * key), which this enumeration of storage dtypes alone cannot express — the snapshots list the i4
 * variants explicitly with their group lengths.
 */
export const WEIGHT_STORAGES: readonly WeightStorage[] = ["f32", "f16", "bf16", "i8"];

/**
 * i4 group length → the shift baked into the WGSL (the single derivation point for the key
 * fragment and the output — shared by linear / embedding).
 *
 * MUST: one-to-one with i4 (missing for i4 / present for anything else are both wiring bugs; let
 * through silently, they become silent wrong values: "a group-32 pipeline runs on group-64
 * assets"). The declaration layer (format/ir.ts) already guarantees a power of two ≥ 16, but since
 * the shift is derived here, it is restated.
 *
 * `where` is the kernel name (the subject of the diagnostic) — so a mixed-up generator entry point
 * can be traced from the message.
 */
export const i4GroupShift = (
  where: string,
  weight: WeightStorage,
  groupSize: number | undefined,
): number | undefined => {
  if ((weight === "i4") !== (groupSize !== undefined)) {
    throw new CodegenError(
      `${where}: groupSize must be passed together with i4 weight storage (weight=${weight})`,
    );
  }
  if (groupSize === undefined) return undefined;
  const shift = Math.log2(groupSize);
  if (!Number.isInteger(shift) || groupSize < 16) {
    throw new CodegenError(
      `${where}: i4 group_size ${groupSize} is not a power of two that is at least 16`,
    );
  }
  return shift;
};

/** Key fragment of the i4 group length (`g32` — codegen determinism: same key → same WGSL). */
export const i4GroupKeyPart = (groupSize: number | undefined): string =>
  groupSize === undefined ? "" : `g${groupSize}`;

/**
 * Storage discriminator carried in the pipeline key.
 *
 * MUST: the empty string for f32. Adding a discriminator to existing keys turns the same kernel
 * into a different key, so the browser's implicit shader cache is rebuilt and every test that pins
 * a key moves at once.
 */
export const weightKeyPart = (storage: WeightStorage): string =>
  storage === "f32"
    ? ""
    : storage === "f16"
      ? ":wf16"
      : storage === "bf16"
        ? ":wbf16"
        : storage === "i8"
          ? ":wi8"
          : storage === "i2"
            ? ":wi2"
            : ":wi4";

/**
 * WGSL element type of the weight buffer (u32 for f16 / bf16, which pack two elements per word, and
 * for i8, which packs four).
 *
 * `quad` changes only f32 to `vec4<f32>` (f16 / i8 are already packed; their array type does not
 * change).
 */
export const weightArrayType = (storage: WeightStorage, quad = false): string =>
  storage === "f32" ? (quad ? "vec4<f32>" : "f32") : "u32";

/** Note appended to the WGSL header comment (empty for f32 — keeps the existing bytes). */
export const weightNote = (storage: WeightStorage): string =>
  storage === "f32"
    ? ""
    : storage === "f16"
      ? ", f16 weight storage"
      : storage === "bf16"
        ? ", bf16 weight storage"
        : storage === "i8"
          ? ", i8 weight storage"
          : storage === "i2"
            ? ", i2 weight storage"
            : ", i4 weight storage";

/**
 * The **default** name of the local variable that holds the per-channel scale in the i8 / i2
 * variants.
 *
 * {@link weightScaleWgsl} binds it once outside the reduction, and each kernel passes it as the
 * fourth argument of {@link weightRead}. A kernel that forgets the binding fails WGSL compilation
 * with "undeclared identifier" (not silently). Where one thread handles several channels (the fill
 * slots of the GEMM skeleton) one name is not enough, so another name can be passed as the fourth
 * argument of {@link weightScaleWgsl}.
 */
export const WEIGHT_SCALE_VAR = "wscale_v";

/**
 * The decode function of the f16 / bf16 / i8 / i4 / i2 variants (plus the scale binding
 * declaration for i8 / i4 / i2). Meant to be placed right after the bindings, and shaped **with one
 * blank line before and after** (for f32 it is the empty string, and only the single blank line of
 * before remains).
 *
 * `scaleBinding` is used only for i8 / i4 / i2 — pass the **number after** the output binding. f32
 * / f16 / bf16 emit no declaration, so the number is not consumed and the existing f32 / f16 output
 * does not move by a single byte.
 *
 * `quad` is the opt-in that switches to the quad decode (`dequant4`) paired with
 * {@link weightRead4}; the scalar `dequant` is then not emitted (the v4 path only reads quads).
 * MUST: the default is `false` — adding `dequant4` unconditionally moves the output of the four
 * existing ops.
 */
export const weightLoaderWgsl = (
  name: string,
  storage: WeightStorage,
  scaleBinding: number,
  quad = false,
): string => {
  if (storage === "f32") return "";
  if (storage === "f16") {
    // MUST: the quad version relies on **the flat index i being a multiple of 4** (it never
    // crosses a word, so two u32 words hold exactly four elements). It does not rely on the row
    // start falling on a word boundary, so it is not ADR 0018's parity trap — the scalar path's
    // detector (tests with row lengths not a multiple of 2) stays intact.
    return quad
      ? `
// quad decode of f16 storage: elements i..i+3 = the two words ${name}[i / 2] and ${name}[i / 2 + 1]
fn dequant4(i: u32) -> vec4<f32> {
  let lo = unpack2x16float(${name}[i >> 1u]);
  let hi = unpack2x16float(${name}[(i >> 1u) + 1u]);
  return vec4<f32>(lo.x, lo.y, hi.x, hi.y);
}
`
      : `
// decode of f16 storage: element i = unpack2x16float(${name}[i / 2])[i % 2]
// (the parity of the flat index picks the half)
fn dequant(i: u32) -> f32 {
  let pair = unpack2x16float(${name}[i >> 1u]);
  return select(pair.x, pair.y, (i & 1u) == 1u);
}
`;
  }
  if (storage === "bf16") {
    // bf16 = the upper 16 bits of an f32. The lower half of a word becomes the f32 bit pattern with
    // `<< 16u`, the upper half with `& 0xFFFF0000u` (ADR 0131). The WGSL specification leaves the
    // result of bitcasting NaN / Inf bit patterns undefined, but weights have no non-finite values
    // (the exporter rejects them). The quad version's 4-alignment premise is the same as f16's.
    return quad
      ? `
// quad decode of bf16 storage: elements i..i+3 = the two words ${name}[i / 2] and ${name}[i / 2 + 1]
// (bf16 is the upper 16 bits of an f32)
fn dequant4(i: u32) -> vec4<f32> {
  let lo = ${name}[i >> 1u];
  let hi = ${name}[(i >> 1u) + 1u];
  return bitcast<vec4<f32>>(vec4<u32>(lo << 16u, lo & 0xFFFF0000u, hi << 16u, hi & 0xFFFF0000u));
}
`
      : `
// decode of bf16 storage: element i = the lower / upper 16 bits of ${name}[i / 2] moved to the top
// of an f32 (the parity of the flat index picks)
fn dequant(i: u32) -> f32 {
  let word = ${name}[i >> 1u];
  return bitcast<f32>(select(word << 16u, word & 0xFFFF0000u, (i & 1u) == 1u));
}
`;
  }
  if (storage === "i2") {
    // Extracts four elements at a time from the low end. A quad is 4-aligned, so it stays within
    // one byte and never crosses a word boundary.
    return quad
      ? `
@group(0) @binding(${scaleBinding}) var<storage, read> wscale: array<f32>;

// INT2: u = q + 2 from the low 2 bits upward. The dequantization multiply is per element.
fn dequant4(i: u32, scale: f32) -> vec4<f32> {
  let byte = (${name}[i >> 4u] >> ((i & 15u) * 2u)) & 255u;
  let q = vec4<i32>(vec4<u32>(byte, byte >> 2u, byte >> 4u, byte >> 6u) & vec4<u32>(3u)) - vec4<i32>(2);
  return vec4<f32>(q) * scale;
}
`
      : `
@group(0) @binding(${scaleBinding}) var<storage, read> wscale: array<f32>;

// INT2: 16 elements per word. The word and the position in it come from the flat index.
fn dequant(i: u32, scale: f32) -> f32 {
  let u = (${name}[i >> 4u] >> ((i & 15u) * 2u)) & 3u;
  return f32(i32(u) - 2) * scale;
}
`;
  }
  if (storage === "i4") {
    // MUST: the nibble order is "element 2i = low / 2i+1 = high, u = q + 8" (ADR 0069 decision 4
    // — the source of truth is pack_int4 in the exporter's emit.py. A mix-up is a silent wrong
    // value with matching shape and type, so the GPU check with an asymmetric pattern is the
    // detector). The scale is per group and depends on k, so unlike i8 it cannot be hoisted as a
    // loop invariant — the caller (for linear, gemm.ts's fillBLinear on the B side; for the conv1d
    // igemm, fillAConv on the A side) reads it per quad and passes it in (a 4-aligned quad never
    // crosses a group because group_size ≥ 16).
    return quad
      ? `
@group(0) @binding(${scaleBinding}) var<storage, read> wscale: array<f32>;

// quad decode of i4 storage: elements i..i+3 = two bytes of the same word (i is a multiple of 4 —
// crosses neither a word nor a group); restored as f32(i32(u) − 8) · scale with a per-component f32
// multiply (the same rounding as the CPU decode in format/i4.ts)
fn dequant4(i: u32, scale: f32) -> vec4<f32> {
  let bytes = unpack4xU8(${name}[i >> 3u]);
  let b0 = bytes[(i >> 1u) & 3u];
  let b1 = bytes[((i >> 1u) & 3u) + 1u];
  return vec4<f32>(
    f32(i32(b0 & 0xFu) - 8),
    f32(i32(b0 >> 4u) - 8),
    f32(i32(b1 & 0xFu) - 8),
    f32(i32(b1 >> 4u) - 8),
  ) * scale;
}
`
      : `
@group(0) @binding(${scaleBinding}) var<storage, read> wscale: array<f32>;

// decode of i4 storage: element i = f32(i32(nibble) − 8) · scale
// (1 word = 8 elements. The flat index gives word i/8, byte (i/2)%4, nibble i%2 — ADR 0069)
fn dequant(i: u32, scale: f32) -> f32 {
  let byte = unpack4xU8(${name}[i >> 3u])[(i >> 1u) & 3u];
  let nibble = select(byte & 0xFu, byte >> 4u, (i & 1u) == 1u);
  return f32(i32(nibble) - 8) * scale;
}
`;
  }
  // MUST: in the quad version too the scale is **a per-component f32 multiply** (`vec4 * scalar`).
  // It is the same operation with the same rounding as the scalar path's `f32(q) * s`, and keeps
  // ADR 0019's "the scale is not moved out of the reduction".
  return quad
    ? `
@group(0) @binding(${scaleBinding}) var<storage, read> wscale: array<f32>;

// quad decode of i8 storage: elements i..i+3 = f32(unpack4xI8(${name}[i / 4])) · scale
// (4 elements in one word)
// MUST: i is a multiple of 4 (called only from the v4 path) — crossing a word scatters the 4
// elements over different rows
fn dequant4(i: u32, scale: f32) -> vec4<f32> {
  return vec4<f32>(unpack4xI8(${name}[i >> 2u])) * scale;
}
`
    : `
@group(0) @binding(${scaleBinding}) var<storage, read> wscale: array<f32>;

// decode of i8 storage: element i = f32(unpack4xI8(${name}[i / 4])[i % 4]) · scale
// (the flat index gives the word and the position. The scale is per output channel — ADR 0019)
fn dequant(i: u32, scale: f32) -> f32 {
  return f32(unpack4xI8(${name}[i >> 2u])[i & 3u]) * scale;
}
`;
};

/**
 * **Integer-lane decode** of i4 storage (w4a8 path only — {@link "./linear-i8a8.ts"}). Turns the
 * flat elements `i`..`i+3` (`i` a multiple of 4) into **a u32 holding four i8 lanes** (the same
 * byte order as `idot` / `pack4xI8`: lane `j` is element `i+j`).
 *
 * MUST: no scale is applied. The w4a8 numeric contract is "an exact i32 dot product within a group,
 * flushed to f32 only at group boundaries"; mixing in f32 here adds a rounding per element (the
 * caller applies the scale once at the group boundary). Its **role differs** from the `dequant4` of
 * {@link weightLoaderWgsl} (the f32 compute path); only the nibble order can be shared.
 *
 * MUST: the nibble order is **identical** to `dequant4` (element 2i = low / 2i+1 = high, stored
 * value `u = q + 8` — ADR 0069 decision 4; the source of truth is the exporter's
 * `karume/emit.py: pack_int4`). Keeping it inside this one file is the only structural safeguard
 * against a mix-up; on the value side the only detector is the GPU check with "an asymmetric
 * pattern that alternates the sign of adjacent elements" (the w4a8 section of
 * tests/gpu_i8a8_test.ts — with a symmetric pattern the values match even when low and high are
 * swapped).
 *
 * MUST: no dynamic index into a `vec4` (on Metal it falls back to threadgroup / local memory — the
 * Metal pitfall in ACTIVE_DESIGN). The four nibbles are extracted with 16-bit shifts only.
 */
export const i4IntegerLanesWgsl = (name: string): string => `
// integer-lane decode of i4 storage: flat elements i..i+3 (i a multiple of 4) → a u32 of four i8
// lanes. 1 word = 8 elements, so the 4 elements are the upper or lower 16 bits of the word
// ((i >> 2) & 1 picks). Restored as q = u − 8 **as an integer** (the scale is applied by the f32
// flush at the group boundary — ADR 0069 decision 4)
fn i4lanes(i: u32) -> u32 {
  let half = (${name}[i >> 3u] >> (((i >> 2u) & 1u) * 16u)) & 0xFFFFu;
  let u = vec4<u32>(half, half >> 4u, half >> 8u, half >> 12u) & vec4<u32>(0xFu);
  return pack4xI8(vec4<i32>(u) - vec4<i32>(8));
}
`;

/**
 * The binding that reads the scale of output channel `channel` **once, outside the reduction**
 * (the empty string for f32 / f16 / bf16).
 *
 * Meant to be placed at the end of a line, so it starts with a newline (for f32 / f16 it is the
 * empty string and the existing line stays as it is). `indent` is the indentation of the insertion
 * point, passed through as is.
 *
 * `variable` is the name of the local variable to bind. MUST: the default is
 * {@link WEIGHT_SCALE_VAR} — the default that keeps the output of the four one-thread-one-channel
 * kernels (conv1d / conv2d direct / conv_transpose1d / embedding) from moving by a single byte; the
 * snapshots (tests/fixtures/wgsl/) are the detector. Only the side that handles several channels
 * (the fill slots of the GEMM skeleton) passes a separate name per slot.
 *
 * NOTE: i4 is **not bound here** (stays the empty string) — the group scale depends on the
 * quantized axis (k / D), so hoisting it as channel-invariant does not hold, and where it is read
 * differs per kernel: linear reads it per quad on the fill side (gemm.ts's fillBLinear), conv1d's
 * implicit GEMM per quad in the A-side fill (gemm.ts's fillAConv), and embedding, with one thread
 * per output element, reads it per element in the kernel body (ADR 0069).
 */
export const weightScaleWgsl = (
  storage: WeightStorage,
  channel: string,
  indent: string,
  variable: string = WEIGHT_SCALE_VAR,
): string =>
  storage === "i8" || storage === "i2"
    ? `
${indent}// the output channel's scale is loop-invariant — not re-read per weight element (ADR 0019)
${indent}let ${variable} = wscale[${channel}];`
    : "";

/**
 * Read expression for one weight element (direct index for f32; the decode function for f16 / bf16
 * / i8).
 *
 * `scale` is **a scale expression** used only by the i8 / i4 / i2 variants (i8 / i2 pass the
 * output channel's {@link WEIGHT_SCALE_VAR} bound by {@link weightScaleWgsl}; i4 passes the group
 * scale the caller read). f32 / f16 / bf16 ignore it, so the f32 / f16 output stays
 * byte-identical to before.
 */
export const weightRead = (
  name: string,
  storage: WeightStorage,
  index: string,
  scale: string,
): string =>
  storage === "f32"
    ? `${name}[${index}]`
    : storage === "f16" || storage === "bf16"
      ? `dequant(${index})`
      : `dequant(${index}, ${scale})`;

/**
 * Read expression for **four** weight elements (flat indices `index`..`index+3`).
 *
 * MUST: `index` is a multiple of 4 (called only from GEMM's v4 path). f32 becomes a quad index into
 * the `vec4<f32>` array; f16 / bf16 / i8 / i4 / i2 become the `dequant4` emitted by the quad
 * version of {@link weightLoaderWgsl}.
 */
export const weightRead4 = (
  name: string,
  storage: WeightStorage,
  index: string,
  scale: string,
): string =>
  storage === "f32"
    ? `${name}[(${index}) >> 2u]`
    : storage === "f16" || storage === "bf16"
      ? `dequant4(${index})`
      : `dequant4(${index}, ${scale})`;
