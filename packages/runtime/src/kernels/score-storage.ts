/**
 * **Storage form** of the attention score S (`s=f32` / `s=f16`) — a generation part shared only by
 * the three fused-attention kernels (①QK writes it, ②row statistics / ③PV read it — ADR 0023).
 * Enabled by the opt-in `SessionOptions.attentionScoreStorage`.
 *
 * ## How it mirrors weight storage ({@link "./weight-storage.ts"}) and where it decisively differs
 *
 * The f16 variant binds S as **`array<u32>`** and moves 2 elements at a time with
 * `unpack2x16float` / `pack2x16float` (core WGSL, no optional feature — the same mechanism as the
 * f16 weights of ADR 0018). So the rule that `attentionCompute: "a8"` **does not request
 * `shader-f16`** (ADR 0030 decision 1) stays intact, and i8a8 and S f16 compose orthogonally.
 *
 * **The difference is that it has a write side.** f16 weights were read-only, so `unpack2x16float`
 * was free, but S is written by ①QK. Packing 2 elements per word imposes the constraint "one thread
 * writes the 2 elements of an even boundary at once", which only the **v4 path** of the GEMM
 * skeleton meets (one thread writes 4 consecutive columns = exactly 2 words). The scalar path writes
 * partially under `if (ocol + k < dims.n)`, which becomes a non-atomic read-modify-write of the same
 * u32 word.
 * MUST: therefore **s16 is not wired to the scalar path**. The generation entry points
 * (src/kernels/gemm.ts / src/kernels/attention-i8a8.ts) reject `!v4` with fail loudly, and the
 * eligibility check ({@link attentionScoreUsesF16}) falls back to f32 storage when not eligible.
 *
 * ## Rounding happens once, at "storage"
 *
 * `pack2x16float` is the only rounding point, and the read side's `unpack2x16float` is **exact**
 * (f16 → f32 is always representable). So the numeric contract is
 *
 *     output(s16) ≡ output(f32 with S rounded to f16 on the host)     … not one bit differs
 *
 * (a stronger contract than the f16 **tile computation** of ADR 0028 — there the rounding enters in
 * the middle of the dot product).
 * MUST: do not write an `f32 → f16 → f32` round trip in registers (the compiler removes it — a
 * finding of ADR 0028). Rounding always goes through storage.
 *
 * MUST: the output of the `s=f32` variant stays **byte-identical to before**. The storage
 * discriminator of the key is the empty string for f32, and each insertion point returns the same
 * string as before for f32. The identity of the existing snapshots (tests/fixtures/wgsl/*.wgsl) and
 * the pipeline cache depends on this.
 */

import { CodegenError } from "../codegen/errors.ts";

/** Storage form of S. The semantics are f32 for both (computation is always f32 — ADR 0006). */
export type ScoreStorage = "f32" | "f16";

/** Storage bytes per S element (f16 packs 2 elements into one word). */
export const scoreStorageBytes = (storage: ScoreStorage): number => (storage === "f16" ? 2 : 4);

/**
 * Whether the shape allows f16 storage of S.
 *
 * **The only condition is that the writer (①QK) takes the v4 path**, which comes down to
 * `D % 4 == 0 && N % 4 == 0`:
 *
 * - f32 ①QK decides v4 with `gemmUsesVec4(k = D, n = N)` (src/kernels/gemm.ts) — it needs **both**
 *   the reduction side `D % 4` and the output-column side `N % 4`.
 * - i8a8 ①QK looks only at the output-column side (`N % 4`), but the eligibility of the i8a8 stage
 *   already requires `D % 4 == 0`, so whether ① is i8a8 or fell back to f32, the condition folds
 *   into the same single one.
 *
 * The read side (②row statistics / ③PV) is safe even when not eligible (reads have no RMW), but it
 * is pointless while the writer writes f32. **Not eligible falls back silently to f32 storage** (the
 * same style as linear's `k % 4`, ADR 0030 decision 5; the fallback shows only in the diagnostic
 * pipeline keys — the family check detects it with the full-eligibility check
 * `assertAttentionSeatsComplete`. ADR 0058 addendum 2026-10-10).
 */
export const attentionScoreUsesF16 = (depth: number, cols: number): boolean =>
  depth % 4 === 0 && cols % 4 === 0;

/**
 * Storage discriminator put in the pipeline key. **A third word that does not collide with the
 * stored weights `:wf16` (ADR 0018) / computation `:c16` (ADR 0028)**.
 *
 * MUST: f32 is the empty string (existing keys do not move by a single character). MUST: the word
 * goes **at the end of the key** — the three axes can be enabled at once, so unless the order is
 * fixed in one place the same configuration gets two keys.
 */
export const scoreKeyPart = (storage: ScoreStorage): string => (storage === "f16" ? ":s16" : "");

/** Note appended to the leading WGSL comment (empty for f32 — keeps the existing bytes). */
export const scoreNote = (storage: ScoreStorage): string =>
  storage === "f16" ? ", S は f16 格納（pack2x16float）" : "";

/**
 * WGSL element type of the S buffer. f16 storage packs 2 elements into one word, so it becomes a
 * **flat u32 array** (`f32Type` passes through the existing type the f32 variant uses — it differs
 * per variant, `vec4<f32>` / `f32` / `vec4<f16>` / `f16`, so the caller passes it instead of the
 * decision being gathered here).
 */
export const scoreArrayType = (storage: ScoreStorage, f32Type: string): string =>
  storage === "f16" ? "u32" : f32Type;

/**
 * Quad read function of the f16 variant (takes a **quad index** — so the caller's index arithmetic
 * does not change). Meant to follow the bindings directly, and shaped with one blank line before and
 * after (the empty string for f32).
 */
export const scoreQuadLoaderWgsl = (name: string, storage: ScoreStorage): string =>
  storage === "f16"
    ? `
// Quad read of f16 storage: quad q = the 2 words ${name}[2q] and ${name}[2q + 1] (unpack is exact)
fn score_quad(q: u32) -> vec4<f32> {
  let w = q * 2u;
  let lo = unpack2x16float(${name}[w]);
  let hi = unpack2x16float(${name}[w + 1u]);
  return vec4<f32>(lo.x, lo.y, hi.x, hi.y);
}
`
    : "";

/**
 * Scalar read function of the f16 variant (takes a **flat element index** — the same form as the
 * `dequant` of f16 weights).
 *
 * MUST: the word and position are chosen by the parity of the flat index. Taking the parity of the
 * index relative to the row matches only by accident when the row length is a multiple of 2 (the
 * same trap as in ADR 0018).
 */
export const scoreScalarLoaderWgsl = (name: string, storage: ScoreStorage): string =>
  storage === "f16"
    ? `
// Decode of f16 storage: element i = unpack2x16float(${name}[i / 2])[i % 2]
// (the parity of the flat index picks the half of the pair)
fn score_at(i: u32) -> f32 {
  let pair = unpack2x16float(${name}[i >> 1u]);
  return select(pair.x, pair.y, (i & 1u) == 1u);
}
`
    : "";

/**
 * Quad write function of the f16 variant (takes a **quad index**). **Rounding happens here only**.
 *
 * MUST: called only from the v4 path, where one thread owns 4 consecutive columns (it writes
 * exactly 2 words exclusively).
 */
export const scoreStoreWgsl = (name: string, storage: ScoreStorage): string =>
  storage === "f16"
    ? `
// Quad write of f16 storage: **this pack2x16float is the only rounding point** (the v4 path owns
// the 2 words exclusively)
fn score_store(q: u32, value: vec4<f32>) {
  let w = q * 2u;
  ${name}[w] = pack2x16float(value.xy);
  ${name}[w + 1u] = pack2x16float(value.zw);
}
`
    : "";

/** Read expression for **one quad** of S (f32 indexes the quad directly, f16 uses the decoder). */
export const scoreReadQuad = (name: string, storage: ScoreStorage, quad: string): string =>
  storage === "f16" ? `score_quad(${quad})` : `${name}[${quad}]`;

/** Read expression for **one element** of S (flat element index). */
export const scoreReadAt = (name: string, storage: ScoreStorage, index: string): string =>
  storage === "f16" ? `score_at(${index})` : `${name}[${index}]`;

/** Write statement for **one quad** of S (including the trailing `;`). */
export const scoreStoreQuad = (
  name: string,
  storage: ScoreStorage,
  quad: string,
  value: string,
): string =>
  storage === "f16" ? `score_store(${quad}, ${value});` : `${name}[${quad}] = ${value};`;

/**
 * Rejects violations of the s16 variant's generation preconditions (does not silently return the
 * f32 path's WGSL).
 *
 * - `!v4`: partial writes on the scalar path become an RMW of the same u32 word (the MUST above).
 * - `computeF16`: `:c16` is **a different form** that holds S as `array<f16>` (ADR 0028).
 *   Silently interpreting a redundant and contradictory combination as one of them hides from the
 *   diagnostics which contract the rounding sequence follows.
 */
export const assertScoreStorageSupported = (
  op: string,
  storage: ScoreStorage,
  v4: boolean,
  computeF16 = false,
): void => {
  if (storage !== "f16") return;
  if (computeF16) {
    throw new CodegenError(
      `${op}: S の f16 格納（s16）と f16 タイル計算（c16）は同時に指定できない` +
        "（c16 は S を array<f16> で持つ別の形 — attentionScoreStorage か attentionCompute の" +
        "どちらか一方にすること）",
    );
  }
  if (!v4) {
    throw new CodegenError(
      `${op}: S の f16 格納（s16）は v4 経路専用` +
        "（スカラ経路の部分書きは同じ u32 語への read-modify-write になる）",
    );
  }
};
