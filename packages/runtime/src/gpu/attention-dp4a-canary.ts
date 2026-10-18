/**
 * A canary that checks by running whether the **i8a8 kernels** of fused attention (①QK / ③PV)
 * **return a known answer** on this device. Its verdict becomes the default of
 * `SessionState.attentionI8a8Dot`.
 *
 * ## Why only attention needs a run-time check
 *
 * "The `dot4I8Packed` version and the emulated version return the same integers"
 * (src/kernels/linear-i8a8.ts) is a claim derived from the properties of integer arithmetic, and
 * for linear it holds on real hardware, Metal included. The run-time check was put on attention
 * alone because the fused-attention i8a8 tests initially failed on an Apple M2, and we inferred
 * that "dp4a / emu disagree in the context of attention (geometry, filling the shared tiles,
 * writing out)". That inference was withdrawn after verifying the canary on real hardware on
 * 2026-08-29 — the two variants are bit-identical on the M2 too, and what failed was the comparison
 * between the GPU and the TS reference (a 1 ULP difference in the f32 epilogue both variants share
 * — §Decision rule v2 below, docs/known-issues.md).
 *
 * The check stays after the withdrawal. Its ground is not agreement between variants but
 * **comparison with a known answer**: it is the only way to confirm by running that the
 * integer-dot-product variant returns correct values on this device (no path that changes the
 * numbers is selected automatically from the adapter's list — ADR 0058 decision 2). Therefore:
 *
 * - MUST: the canary is fired in a form that fills **at least one whole tile at production
 *   geometry**. A tiny shape exercises neither the work assignment of tile filling nor the cycling
 *   over K tiles, so failures on those paths would pass straight through.
 * - MUST: the **primary verdict** of the comparison is made against the **known answer** (the TS
 *   reference src/reference/i8a8.ts). If cross-comparing the arms were the primary verdict, a
 *   device whose emu side is broken would be judged "both give the same wrong value → agree →
 *   healthy" (which arm breaks differs per device and cannot be ruled out in advance).
 *   Cross-comparison is used only to pick an arm **after both arms fell within the known answer's
 *   sanity band** (step 3 below).
 * - MUST: if both arms fall outside the sanity band, fail with {@link GpuFeatureError}. It is not a
 *   variant-selection problem but "this device's integer attention is broken", so silently picking
 *   one would hand out wrong values while still honoring ADR 0058 decision 2 (no path that changes
 *   the numbers is selected automatically from the adapter's capabilities).
 *
 * A fallback shows in the diagnostics — choosing emu puts `:dp4aEmu` in the pipeline key
 * (src/kernels/attention-i8a8.ts, ADR 0058 decision 3).
 *
 * ## Decision rule v2 — why atol=0 alone is not enough
 *
 * The first version accepted only "an arm that matched the known answer with atol=0". However, the
 * measurement on an Apple M2 (2026-08-29 / Deno 2.9.6 —
 * docs/research/2026-08-29-chatgpt-review-verification.md) shows that **dp4a and emu do not differ
 * by a single bit** (canary ①QK v4 measured 28.433290481567383 on both arms), yet they are off from
 * the known answer 28.43328857421875 by **exactly 1 ULP**. The integer stage (qP of ③PV) mismatches
 * in 0% of 62,088 elements, so what is off is not the integer dot product but **the rounding of the
 * f32 epilogue both variants share** (hypothesis: a difference in FMA contraction in naga→MSL — not
 * settled). The first version read this as "both arms missed the known answer", failed, and killed
 * `attentionCompute: "a8"` entirely on the M2.
 *
 * So the verdict is made in this order ({@link decideAttentionI8a8Dot}):
 *
 * 1. dp4a matches the known answer exactly → `"dp4a"` (a healthy machine finishes in 1 submit as
 *    before).
 * 2. emu matches the known answer exactly → `"emu"` (a device with only one arm broken).
 * 3. If neither matches exactly, judge by the **sanity band** ({@link SANITY_RELATIVE_TOLERANCE}).
 *    If only one arm is inside the band, that arm. If both are inside, check **whether the arms are
 *    bit-identical in every element of all 6 variants**: if identical, `"dp4a"` (= it has been
 *    verified mechanically that the variant choice does not affect the numbers, so ADR 0058
 *    decision 2 is satisfied); if they differ, the arm closer to the known answer (ties go to
 *    dp4a).
 * 4. Both arms outside the band → {@link GpuFeatureError} (only a truly broken device gets here).
 *
 * A verdict decided by 3 or 4 carries the fact "it was not an exact match" in the return value
 * ({@link AttentionI8a8Decision.exact}). The caller (src/runtime/session-build.ts) warns once per
 * device — it does not let it pass silently, but does not stop execution.
 *
 * ## Variants fired (the minimal representative set of the generated key axes)
 *
 * The WGSL of ①QK branches on `v4` (output columns `N % 4`) × the storage form of S, and ③PV on
 * `v4` (output columns `D % 4`) × the storage form of S. 6 combinations can actually be generated,
 * and they are fired in 1 submit:
 *
 * - ①QK: `v4/s=f32` / `scalar/s=f32` / `v4/s=f16`
 * - ③PV: `v4/s=f32` / `scalar/s=f32` / `v4/s=f16`
 *
 * `scalar × s=f16` cannot be generated (s16 is for the v4 path only —
 * src/kernels/score-storage.ts), so it is not a full product. The geometry (tile edges, work
 * assignment) is **the i8a8 attention geometry of the geometry profile the Session chose**
 * (`i8a8.attentionQk` / `i8a8.attentionPv` of src/kernels/geometry-profile.ts) used as is — so the
 * canary fires output that does not differ by a single character from the WGSL production runs;
 * with the default profile it is the geometry of `defaultI8a8Geometry` as before. **The batch axis
 * (B·H ≥ 2) is outside the canary's scope** — the base arithmetic is a generated part that does not
 * change by a single character between dp4a / emu, and existing GPU tests have detectors for it.
 *
 * ## Fixed inputs (frozen)
 *
 * Every input is made by a deterministic pure function (no random numbers — a failure would not
 * reproduce). Only the shape (M / N / D) is derived from the tile edges of the given geometry; how
 * the values are made (below) does not depend on the geometry. **Quantization is done on the host
 * and the i8 payload is uploaded directly**, so the division in `quantize_rows` (WGSL allows up to
 * 2.5 ULP) is not on the canary's path = agreement with the known answer can be claimed with
 * atol=0. Only ③PV has the GPU make `qP = round(127·exp(S−m))`, so implementation differences of
 * `exp` enter. Hence **only S values for which `127·exp(S−m)` stays at least 0.30 away from a
 * rounding boundary (a half-integer) are used** (the error of WGSL's `exp` is a few ULP = on the
 * order of 1e-5, safe by orders of magnitude). tests/gpu_attention_dp4a_canary_test.ts measures
 * this margin over every value of the fixed inputs and makes it a check.
 *
 * ①QK puts the known answer **on the f16 grid itself**. The s16 variant is compared against "the
 * known answer rounded to f16 on the host", so if the known answer sat near an f16 rounding
 * boundary, a GPU S that differs only inside the sanity band would shift the stored value by one
 * step, amplifying the difference to 1 f16 ULP (55 times the band) and putting **both arms outside
 * the band** (the old fixed inputs had a minimum margin of only 1.78e-7 relative = 1/56 of the
 * band). Two ways put it there:
 *
 * 1. **Keep the factors powers of 2** — make all 3 factors of
 *    `S = f32(acc)·((qscale·scale)·(kscale·scale))` powers of 2 ({@link QK_DEPTH} 256 gives a half
 *    scale of 0.25 / making values as `integer × 2⁻ᵉ` makes `amax·(1/127)` exactly `2⁻ᵉ`). The
 *    product never rounds, so `S = acc·2⁻ᵏ` is exact.
 * 2. **Fit the significant digits of acc into the f16 mantissa (11 bits)** — fixing everything in k
 *    except the row max to ±64 gives `acc = 64·M`, and with `|M| ≤ 2047` S is exactly an f16. The
 *    measured maximum of the design is **|M| = 519**.
 *
 * Further, **M is fixed odd** to exclude `S = 0` (0 has no definable relative margin; only the
 * band's absolute floor would remain, which loses to one f16 subnormal step). The parity of M is
 * decided by "the number of odd elements of q", so the odd elements are fixed to **exactly 2** (the
 * row max and col 1). tests/gpu_attention_dp4a_canary_test.ts measures the margin over all 8,192
 * elements of the known answer and makes it a check (measured at **24.4 times** the band — the old
 * fixed inputs were at 0.018 times).
 */

import { type I8a8Geometry, i8a8TileM, i8a8TileN } from "../kernels/i8a8-geometry.ts";
import {
  ATTENTION_PV_V_SCALE_BINDING,
  ATTENTION_QK_K_SCALE_BINDING,
  ATTENTION_QK_Q_SCALE_BINDING,
  attentionPvI8a8Key,
  attentionPvI8a8Params,
  attentionPvI8a8Wgsl,
  attentionQkI8a8Key,
  attentionQkI8a8Params,
  attentionQkI8a8Wgsl,
} from "../kernels/attention-i8a8.ts";
import { tiledWorkgroups } from "../codegen/dispatch.ts";
import type { ScoreStorage } from "../kernels/score-storage.ts";
import { f16BitsToF32, roundToF16 } from "../format/f16.ts";
import {
  quantizeRowsReference,
  referenceAttentionPvI8a8Core,
  referenceAttentionPvQuant,
  referenceAttentionQkI8a8,
} from "../reference/i8a8.ts";
import type { I8a8Dot } from "../runtime/session-types.ts";
import { type GpuContext, GpuFeatureError, RUNTIME_INTERNAL } from "./device.ts";
import {
  discardFailureScopes,
  popFailureScopes,
  pushFailureScopes,
  withPipelineScope,
} from "./error-scope.ts";
import { BUFFER_USAGE, MAP_MODE } from "./webgpu-constants.ts";

/**
 * The name used in diagnostics and error messages (this one string tells "which canary failed").
 */
const WHERE = "attention i8a8 カナリア";

/**
 * A hook to replace the generated WGSL (**for fault-injection tests only**). By default the
 * production output itself.
 */
export type CanaryWgslPatch = (wgsl: string) => string;

const IDENTITY_PATCH: CanaryWgslPatch = (wgsl) => wgsl;

/**
 * The relative width of the sanity band. The per-element tolerance is
 * `max(|known answer| · this value, {@link SANITY_ABSOLUTE_FLOOR})`.
 *
 * Derivation (M2 measurement 2026-08-29 — docs/research/2026-08-29-chatgpt-review-verification.md):
 *
 * - **What we want to accept** = the rounding difference of the f32 epilogue both variants share.
 *   Measured at 1 ULP (28.433290481567383 vs the known answer 28.43328857421875 = 6.7e-8 relative);
 *   even allowing a few ULP it is 1.2e-7 relative. It sits **2 orders of magnitude below** the
 *   band.
 * - **What we want to reject** = real integer failures. The fault injection (breaking the dp4a
 *   WGSL) measured 28.435983657836914 = 9.5e-5 relative, **1 order of magnitude above** the band.
 *
 * So 1e-5 sits at the bottom of a valley 3 orders of magnitude wide between the two. MUST NOT:
 * loosen it — widening to 1e-3 would let real damage on the order of 1 f16 storage ULP (~1e-3
 * relative) into the band and kill the detector for the s16 variants.
 *
 * The band accepts only **differences in the f32 epilogue**; in the s16 variants the storage
 * rounding amplifies "a difference inside the band" to one f16 step (55 times the band) — the
 * rounding becomes an amplifier that crosses the band. Therefore that the band also closes for s16
 * is guaranteed not by the band width but by **the margin of the fixed inputs from f16 boundaries**
 * (§Fixed inputs).
 */
export const SANITY_RELATIVE_TOLERANCE = 1e-5;

/**
 * The absolute floor of the sanity band. Judging by the relative width alone, the tolerance
 * collapses to 0 as `|known answer| → 0`, and only elements near 0 effectively stay at atol=0 (the
 * band becomes patchy per element).
 *
 * Why 1e-7: the smallest amplitude in the current fixed inputs is |O| = 6.2e-4 of ③PV (the f32 ULP
 * there is ~6e-11), so no element is governed by the floor today — the relative side alone already
 * leaves ~100 ULP of margin. It starts to matter when the known answer is rebuilt and values near 0
 * appear; even then 1e-7 sits where it "absorbs rounding differences of a few ULP but not integer
 * failures" (if ③PV's acc is off by 1, |O| moves by at least 5e-4 — the lower bound of prow·vs).
 */
export const SANITY_ABSOLUTE_FLOOR = 1e-7;

// ---------------------------------------------------------------------------
// Fixed inputs (frozen — if you move a value, redo the tests' margin measurements)
// ---------------------------------------------------------------------------

/**
 * The reduction length of ①QK. It MUST be **a power of 16** — so that the half scale
 * `√(1/√depth)` is a power of 2 (256 → 0.25) and the dequant factor never rounds (item 1 of §Fixed
 * inputs). With the default geometry that is 16 K tiles, comfortably meeting the "2 tiles or more"
 * requirement (which the test makes a check).
 */
const QK_DEPTH = 256;

/**
 * The exponent of ①QK's row scale (`qscale = 2⁻ᵉ`). **It varies per row** — so that mixing up row
 * scales shows in the values (in the old fixed inputs all 128 rows had the same row scale, and a
 * row mix-up passed straight through).
 */
const QK_ROW_EXPONENT = (row: number): number => row % 3;

/** The exponent of ①QK's column scale (`kscale = 2⁻ᶠ`). A different period from the row side. */
const QK_COL_EXPONENT = (col: number): number => col % 4;

/**
 * Where q's row max (±127) sits. Avoids col 0 (the slot that meets k's row max) and col 1 (the odd
 * slot).
 */
const QK_SPIKE_COLUMN = (row: number): number => 2 + ((row * 37) % (QK_DEPTH - 2));

/**
 * The quantized values of ①QK's q (item 2 of §Fixed inputs — `acc = 64·M` and M's oddness are
 * decided here).
 *
 * - col 0 = ±64: the slot that meets k's row max (±127). **A multiple of 64 MUST** — if 64 could
 *   not be factored out here alone, `acc` would no longer be a multiple of 64, forcing
 *   `|acc| ≤ 2047` and breaking the design.
 * - col {@link QK_SPIKE_COLUMN} = ±127: the row max (= the only slot where quantization returns
 *   127). The first odd value.
 * - col 1 = odd within ±31: the second odd value. **Exactly 2 odd values**, so M is always odd.
 * - the rest = even within ±30 (slots that do not affect M's parity).
 */
const QK_QUERY_QUANT = (row: number, col: number): number => {
  if (col === 0) return row % 2 === 0 ? 64 : -64;
  if (col === QK_SPIKE_COLUMN(row)) return row % 4 < 2 ? 127 : -127;
  if (col === 1) return ((row * 11) % 31) * 2 - 31;
  return 2 * (((row * 13 + col * 5) % 31) - 15);
};

/**
 * The quantized values of ①QK's k. Only the row max (i = 0) is ±127; the rest are **fixed to ±64**
 * (the 64 of `acc = 64·M` is factored out from here). The sign period 67 is larger than the 64
 * columns, so the sign sequences of the 64 columns are all distinct (a column mix-up shows in the
 * values).
 */
const QK_KEY_QUANT = (col: number, i: number): number => {
  if (i === 0) return col % 2 === 0 ? 127 : -127;
  return (i * 7 + col * 5) % 67 < 34 ? 64 : -64;
};

/** ①QK's q (f32, before quantization = quantized value × row scale). */
const QK_QUERY = (row: number, col: number): number =>
  QK_QUERY_QUANT(row, col) * 2 ** -QK_ROW_EXPONENT(row);

/** ①QK's k (f32, before quantization = quantized value × column scale). */
const QK_KEY = (col: number, i: number): number =>
  QK_KEY_QUANT(col, i) * 2 ** -QK_COL_EXPONENT(col);

/**
 * How far ③PV's S drops below the row max (in steps of 1/64). **All 16 are exactly representable in
 * f16**, and only j for which `127·exp(-j/64)` stays at least 0.30 away from a half-integer are
 * chosen (qP takes the 16 levels 127/121/112/102/93/86/77/68/60/52/44/34/26/18/9/1).
 * MUST: the first is 0 — so that the row max actually exists in that row (`exp(S−m) ≤ 1` is
 * guaranteed by structure).
 */
const PV_SCORE_STEPS = [0, 3, 8, 14, 20, 25, 32, 40, 48, 57, 68, 84, 102, 125, 169, 310] as const;

/** ③PV's row max. Any small value whose 1/64-step S fits exactly in f16 will do. */
const PV_ROW_MAX = 2;

/** ③PV's row `1/Σexp` (exact 1/8 steps in f32 — varies per row). */
const PV_ROW_INV = (row: number): number => (1 + (row % 5)) / 8;

/** ③PV's quantized Vᵀ values (integers within ±127, contiguous over N). */
const PV_V_QUANT = (i: number): number => ((i * 53) % 251) - 125;

/** ③PV's per-column scale of V (exact 1/8 steps in f32). */
const PV_V_SCALE = (col: number): number => 0.5 + (col % 9) * 0.125;

/** The half scale (ADR 0023's `√scale_factor`) — computed once, the same way as for real shapes. */
const halfScale = (depth: number): number => Math.fround(Math.sqrt(1 / Math.sqrt(depth)));

const fillBy = (length: number, value: (i: number) => number): Float32Array<ArrayBuffer> => {
  const data = new Float32Array(length);
  for (let i = 0; i < length; i += 1) data[i] = value(i);
  return data;
};

/**
 * The row-major 2-D version (①QK's values depend on "which row / column slot", so the indices are
 * passed separately).
 */
const fillRowsBy = (
  rows: number,
  cols: number,
  value: (row: number, col: number) => number,
): Float32Array<ArrayBuffer> => {
  const data = new Float32Array(rows * cols);
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) data[row * cols + col] = value(row, col);
  }
  return data;
};

/** ①QK's fixed inputs and known answer. */
type QkCase = {
  readonly rows: number;
  readonly cols: number;
  readonly depth: number;
  readonly scale: number;
  /**
   * The quantized q / k payloads (flat i8 of `[row, depth]` = the same order as the GPU's packing
   * by 4).
   */
  readonly queryPayload: Int8Array<ArrayBuffer>;
  readonly keyPayload: Int8Array<ArrayBuffer>;
  readonly queryScale: Float32Array<ArrayBuffer>;
  readonly keyScale: Float32Array<ArrayBuffer>;
  /** The known answer S (`[rows, cols]`). */
  readonly expected: Float32Array<ArrayBuffer>;
};

/** ③PV's fixed inputs and known answer. */
type PvCase = {
  readonly rows: number;
  /** The reduction axis N (= `k` of the params). */
  readonly cols: number;
  /** The output columns D (= `n` of the params). */
  readonly depth: number;
  readonly scores: Float32Array<ArrayBuffer>;
  /** Row statistics `[max, inv]` (the same order as the output of ② row statistics). */
  readonly stats: Float32Array<ArrayBuffer>;
  readonly valuePayload: Int8Array<ArrayBuffer>;
  readonly valueScale: Float32Array<ArrayBuffer>;
  /** The known answer O (`[rows, depth]`). */
  readonly expected: Float32Array<ArrayBuffer>;
};

/**
 * ①QK's fixed inputs (**the tile edges are derived from the given geometry** — even when a profile
 * changes the geometry, the canary keeps filling one whole tile). K is fixed at {@link QK_DEPTH};
 * with the default geometry that is 16 tiles (with 1 tile the cycling itself is not exercised, so 2
 * or more tiles is a requirement — the test makes it a check).
 */
export const buildQkCase = (geometry: I8a8Geometry): QkCase => {
  const rows = i8a8TileM(geometry);
  const cols = i8a8TileN(geometry);
  const depth = QK_DEPTH;
  const query = fillRowsBy(rows, depth, QK_QUERY);
  const key = fillRowsBy(cols, depth, QK_KEY);
  const quantizedQuery = quantizeRowsReference(query, rows, depth);
  const quantizedKey = quantizeRowsReference(key, cols, depth);
  const scale = halfScale(depth);
  return {
    rows,
    cols,
    depth,
    scale,
    queryPayload: quantizedQuery.q,
    keyPayload: quantizedKey.q,
    queryScale: quantizedQuery.scale,
    keyScale: quantizedKey.scale,
    // The reference quantizes q / k by itself (the same pure function, so it always matches the
    // payloads above)
    expected: referenceAttentionQkI8a8({
      q: query,
      k: key,
      batch: 1,
      m: rows,
      n: cols,
      d: depth,
      scale,
    }),
  };
};

/**
 * ③PV's fixed inputs (likewise — ③ can take a different geometry from ①, so it is derived
 * separately).
 */
export const buildPvCase = (geometry: I8a8Geometry): PvCase => {
  const rows = i8a8TileM(geometry);
  const depth = i8a8TileN(geometry);
  const cols = geometry.tileK * 2;
  const scores = new Float32Array(rows * cols);
  const stats = new Float32Array(rows * 2);
  for (let row = 0; row < rows; row += 1) {
    // Shift the phase per row (so a row mix-up shows in the order of qP)
    for (let i = 0; i < cols; i += 1) {
      const step = PV_SCORE_STEPS[(i + row) % PV_SCORE_STEPS.length];
      scores[row * cols + i] = PV_ROW_MAX - step / 64;
    }
    stats[row * 2] = PV_ROW_MAX;
    stats[row * 2 + 1] = PV_ROW_INV(row);
  }
  const valuePayload = new Int8Array(depth * cols);
  for (let i = 0; i < valuePayload.length; i += 1) valuePayload[i] = PV_V_QUANT(i);
  const valueScale = fillBy(depth, PV_V_SCALE);
  const rowMax = new Float32Array(rows).fill(PV_ROW_MAX);
  const rowInv = fillBy(rows, PV_ROW_INV);
  // Build the known answer by separating the integer sequence that A-tile filling produces
  // (matching the GPU's `round(127·exp(S−m))`) from the pure function after the integers are
  // received (the side where atol=0 holds).
  const quantized = referenceAttentionPvQuant(scores, rowMax, rows, cols);
  return {
    rows,
    cols,
    depth,
    scores,
    stats,
    valuePayload,
    valueScale,
    expected: referenceAttentionPvI8a8Core({
      qp: quantized,
      vq: valuePayload,
      rowInv,
      vs: valueScale,
      batch: 1,
      m: rows,
      n: cols,
      d: depth,
    }),
  };
};

// ---------------------------------------------------------------------------
// f16 storage of S (pack / read back the fixed inputs in the same order as the GPU)
// ---------------------------------------------------------------------------

/**
 * The 16-bit pattern of a **normal number exactly representable** in f16.
 *
 * MUST: do not pass a general f32 — it handles neither rounding nor subnormals nor Inf, and
 * silently drops the low 13 bits of the mantissa. The fixed S are only normal numbers in steps of
 * 1/64 with |v| ≤ 3, so re-biasing the exponent and cutting the mantissa determine it exactly (the
 * source of truth for rounding is {@link roundToF16}; this is only a repacking container).
 * The round trip (getting back through {@link f16BitsToF32}) is made a check by the test over every
 * value of the fixed inputs.
 */
const exactF16Bits = (view: DataView, value: number): number => {
  view.setFloat32(0, value);
  const bits = view.getUint32(0);
  const sign = (bits >>> 16) & 0x8000;
  const exponent = ((bits >>> 23) & 0xff) - 127 + 15;
  const mantissa = (bits & 0x7fffff) >>> 13;
  return sign | (exponent << 10) | mantissa;
};

/**
 * Packs S into the same storage form as the GPU (`pack2x16float`, 2 elements per word).
 * MUST: the low half holds the even index (the x side of `pack2x16float(vec2(x, y))`).
 */
export const packScoresF16 = (scores: Float32Array<ArrayBuffer>): Uint32Array<ArrayBuffer> => {
  const words = new Uint32Array(scores.length / 2);
  const view = new DataView(new ArrayBuffer(4));
  for (let word = 0; word < words.length; word += 1) {
    const low = exactF16Bits(view, scores[word * 2]);
    const high = exactF16Bits(view, scores[word * 2 + 1]);
    words[word] = (low | (high << 16)) >>> 0;
  }
  return words;
};

/** Turns f16-stored S back into f32 (`unpack2x16float` is exact, so no rounding happens). */
const unpackScoresF16 = (words: Uint32Array<ArrayBuffer>): Float32Array<ArrayBuffer> => {
  const values = new Float32Array(words.length * 2);
  for (let word = 0; word < words.length; word += 1) {
    values[word * 2] = f16BitsToF32(words[word] & 0xffff);
    values[word * 2 + 1] = f16BitsToF32(words[word] >>> 16);
  }
  return values;
};

// ---------------------------------------------------------------------------
// Running
// ---------------------------------------------------------------------------

const STORAGE_IN = BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_DST;
const STORAGE_OUT = BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_SRC;

/**
 * The i8a8 geometry of fused attention the canary fires (①QK / ③PV — built from
 * `i8a8.attentionQk` / `i8a8.attentionPv` of the profile the Session chose).
 */
export type AttentionI8a8CanaryGeometry = {
  readonly qk: I8a8Geometry;
  readonly pv: I8a8Geometry;
};

/** The result of comparing one variant with the known answer ({@link compareToReference}). */
export type CanaryVariantOutcome = {
  /**
   * The production pipeline key itself (the diagnostics tell which generated output it concerns).
   */
  readonly key: string;
  /** Whether it matched the known answer with atol=0. */
  readonly exact: boolean;
  /** Whether every element fell within the sanity band. */
  readonly withinBand: boolean;
  /** The maximum absolute error against the known answer. */
  readonly maxAbsError: number;
  /** The maximum band ratio (`|error| / tolerance`). At most 1 is inside the band. */
  readonly maxBandRatio: number;
  /** Diagnostic of the first mismatch (undefined on an exact match). */
  readonly mismatch: string | undefined;
  /**
   * The read-back output itself. Kept because **the bit-identity comparison between arms needs it**
   * (a few KB).
   */
  readonly output: Uint8Array<ArrayBuffer>;
};

/**
 * Full comparison with the known answer. Takes both the exact match (atol=0) and the sanity band in
 * one scan.
 *
 * MUST: write the updates of the band and the maxima in the form `!(diff <= x)` / {@link Math.max}.
 * Written as `diff > x`, a NaN silently lands "inside the band" (an invalid dispatch or an
 * uninitialized readback would be judged healthy).
 */
const compareToReference = (
  key: string,
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  output: Uint8Array<ArrayBuffer>,
): CanaryVariantOutcome => {
  let exact = true;
  let withinBand = true;
  let maxAbsError = 0;
  let maxBandRatio = 0;
  let mismatch: string | undefined;
  let bandMismatch: string | undefined;
  for (let i = 0; i < expected.length; i += 1) {
    if (actual[i] !== expected[i]) {
      exact = false;
      mismatch ??= `${key}[${i}]: 実測 ${actual[i]} / 既知解 ${expected[i]}`;
    }
    const diff = Math.abs(actual[i] - expected[i]);
    const tolerance = Math.max(
      Math.abs(expected[i]) * SANITY_RELATIVE_TOLERANCE,
      SANITY_ABSOLUTE_FLOOR,
    );
    if (!(diff <= tolerance)) {
      withinBand = false;
      bandMismatch ??= `${key}[${i}]: 実測 ${actual[i]} / 既知解 ${expected[i]}`;
    }
    maxAbsError = Math.max(maxAbsError, diff);
    maxBandRatio = Math.max(maxBandRatio, diff / tolerance);
  }
  // MUST: the diagnostic **prefers elements outside the band**. Leading with an element that is an
  // inexact match but inside the band (the M2's 1 ULP difference) would, when a failure outside the
  // band exists elsewhere, put "an unrelated tiny difference" and "band ratio 10" on the same line
  // and mislead (real harm confirmed on the M2 — the expected key of the outside-band test could
  // not be read).
  return {
    key,
    exact,
    withinBand,
    maxAbsError,
    maxBandRatio,
    mismatch: bandMismatch ?? mismatch,
    output,
  };
};

/** The run plan of one variant (pipeline already created, buffers yet to be allocated). */
type Variant = {
  /** The production pipeline key itself (so the diagnostics tell which generated output failed). */
  readonly key: string;
  readonly pipeline: GPUComputePipeline;
  /** Bindings (the params at binding 0 are added by the running side). */
  readonly bindings: readonly (readonly [number, GPUBuffer])[];
  readonly params: GPUBuffer;
  readonly output: GPUBuffer;
  readonly outputBytes: number;
  readonly workgroups: readonly [number, number, number];
  /** Compares the read-back bytes with the known answer. */
  readonly compare: (bytes: ArrayBuffer) => CanaryVariantOutcome;
};

/**
 * The result of firing one arm (either dp4a or emu) on all 6 variants ({@link
 * probeAttentionI8a8Dot}).
 */
export type CanaryProbe = {
  readonly dot: I8a8Dot;
  /** Whether **all** 6 variants matched the known answer with atol=0. */
  readonly exact: boolean;
  /** Whether **all** 6 variants fell within the sanity band. */
  readonly withinBand: boolean;
  /** The maximum absolute error across the 6 variants. */
  readonly maxAbsError: number;
  /** The maximum band ratio across the 6 variants (at most 1 is inside the band). */
  readonly maxBandRatio: number;
  /**
   * Diagnostic of the first variant that missed the known answer (undefined when {@link exact}).
   */
  readonly mismatch: string | undefined;
  /** Per-variant breakdown (in generation order, the same across arms). */
  readonly variants: readonly CanaryVariantOutcome[];
};

/**
 * Fires the 6 variants in 1 submit and returns **the comparison results of all variants**.
 *
 * MUST: do not stop midway (the first version returned at the first mismatch). The bit-identity
 * comparison between arms (step 3 of {@link decideAttentionI8a8Dot}) needs the output of every
 * variant, and the band verdict needs a scan of every element.
 * MUST: the caller calls this inside the per-device errorScope section lock
 * ({@link decideAttentionI8a8Dot}). It pushes errorScopes across `await`s, so running it bare would
 * interleave with a concurrent Session's run and misattribute failures.
 */
export const probeAttentionI8a8Dot = async (
  gpu: GpuContext,
  dot: I8a8Dot,
  geometry: AttentionI8a8CanaryGeometry,
  patch: CanaryWgslPatch = IDENTITY_PATCH,
): Promise<CanaryProbe> => {
  const device = gpu.device;
  const dp4a = dot === "dp4a";
  const { qk: qkGeometry, pv: pvGeometry } = geometry;
  const qk = buildQkCase(qkGeometry);
  const pv = buildPvCase(pvGeometry);
  const limit = gpu.limits.maxComputeWorkgroupsPerDimension;

  const pipelineOf = async (key: string, wgsl: string): Promise<GPUComputePipeline> =>
    await gpu[RUNTIME_INTERNAL].raceDeviceLost(
      withPipelineScope(device, `${WHERE} '${key}'`, () =>
        device.createComputePipeline({
          layout: "auto",
          compute: { module: device.createShaderModule({ code: patch(wgsl) }), entryPoint: "main" },
        }),
      ),
      `${WHERE} '${key}' のパイプライン生成`,
    );

  // Pipelines are created **outside** the synchronous section (creation involves awaits). **scalar
  // × s=f16 is not assembled** — on the ①QK side generation itself refuses it (s16 is for the v4
  // path only), so firing only ③PV would not make a pair.
  const shapes = [
    { v4: true, score: "f32" },
    { v4: false, score: "f32" },
    { v4: true, score: "f16" },
  ] as const;
  type Resolved = {
    readonly key: string;
    readonly pipeline: GPUComputePipeline;
    readonly score: ScoreStorage;
  };
  const qkPipelines: Resolved[] = [];
  const pvPipelines: Resolved[] = [];
  for (const { v4, score } of shapes) {
    const qkKey = attentionQkI8a8Key(v4, dp4a, score, qkGeometry);
    qkPipelines.push({
      key: qkKey,
      score,
      pipeline: await pipelineOf(qkKey, attentionQkI8a8Wgsl(v4, dp4a, score, qkGeometry)),
    });
    const pvKey = attentionPvI8a8Key(v4, dp4a, score, pvGeometry);
    pvPipelines.push({
      key: pvKey,
      score,
      pipeline: await pipelineOf(pvKey, attentionPvI8a8Wgsl(v4, dp4a, score, pvGeometry)),
    });
  }

  const buffers: GPUBuffer[] = [];
  const upload = (data: ArrayBufferView<ArrayBuffer>, usage: number): GPUBuffer => {
    const buffer = device.createBuffer({ size: data.byteLength, usage });
    buffers.push(buffer);
    device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  };
  const allocate = (byteLength: number): GPUBuffer => {
    const buffer = device.createBuffer({ size: byteLength, usage: STORAGE_OUT });
    buffers.push(buffer);
    return buffer;
  };

  try {
    // MUST: no await between the push and the issue of the pop (the ground for preventing
    // interleaving in the per-device LIFO — "Invariants of errorScope sections" in
    // src/gpu/context.ts). Allocation, binding, encoding and submit are all synchronous.
    pushFailureScopes(device);
    let variants: readonly Variant[];
    let staging: GPUBuffer;
    try {
      const uniform = BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST;
      const qkParams = upload(attentionQkI8a8Params(qk.rows, qk.cols, qk.depth, qk.scale), uniform);
      const queryPayload = upload(qk.queryPayload, STORAGE_IN);
      const keyPayload = upload(qk.keyPayload, STORAGE_IN);
      const queryScale = upload(qk.queryScale, STORAGE_IN);
      const keyScale = upload(qk.keyScale, STORAGE_IN);
      const pvParams = upload(attentionPvI8a8Params(pv.rows, pv.depth, pv.cols), uniform);
      const scoresF32 = upload(pv.scores, STORAGE_IN);
      const scoresF16 = upload(packScoresF16(pv.scores), STORAGE_IN);
      const stats = upload(pv.stats, STORAGE_IN);
      const valuePayload = upload(pv.valuePayload, STORAGE_IN);
      const valueScale = upload(pv.valueScale, STORAGE_IN);

      const qkVariants = qkPipelines.map(({ key, pipeline, score }): Variant => {
        const outputBytes = qk.rows * qk.cols * (score === "f16" ? 2 : 4);
        const output = allocate(outputBytes);
        return {
          key,
          pipeline,
          params: qkParams,
          bindings: [
            [1, queryPayload],
            [2, keyPayload],
            [3, output],
            [ATTENTION_QK_Q_SCALE_BINDING, queryScale],
            [ATTENTION_QK_K_SCALE_BINDING, keyScale],
          ],
          output,
          outputBytes,
          workgroups: [
            tiledWorkgroups(qk.cols, i8a8TileN(qkGeometry), limit, WHERE),
            tiledWorkgroups(qk.rows, i8a8TileM(qkGeometry), limit, WHERE),
            1,
          ],
          compare: (bytes) =>
            score === "f16"
              ? // s16 output ≡ the f32 variant with S rounded to f16 on the host (rounded once, at storage)
                compareToReference(
                  key,
                  unpackScoresF16(new Uint32Array(bytes)),
                  Float32Array.from(qk.expected, roundToF16),
                  new Uint8Array(bytes),
                )
              : compareToReference(
                  key,
                  new Float32Array(bytes),
                  qk.expected,
                  new Uint8Array(bytes),
                ),
        };
      });
      const pvVariants = pvPipelines.map(({ key, pipeline, score }): Variant => {
        const outputBytes = pv.rows * pv.depth * 4;
        const output = allocate(outputBytes);
        return {
          key,
          pipeline,
          params: pvParams,
          bindings: [
            [1, score === "f16" ? scoresF16 : scoresF32],
            [2, valuePayload],
            [3, stats],
            [4, output],
            [ATTENTION_PV_V_SCALE_BINDING, valueScale],
          ],
          output,
          outputBytes,
          workgroups: [
            tiledWorkgroups(pv.depth, i8a8TileN(pvGeometry), limit, WHERE),
            tiledWorkgroups(pv.rows, i8a8TileM(pvGeometry), limit, WHERE),
            1,
          ],
          // The storage form of S concerns the input side, so ③'s known answer is the same one for
          // all 3 variants (the fixed S are exactly representable in f16 = repacking does not move
          // a single bit).
          compare: (bytes) =>
            compareToReference(key, new Float32Array(bytes), pv.expected, new Uint8Array(bytes)),
        };
      });
      variants = [...qkVariants, ...pvVariants];

      staging = device.createBuffer({
        size: variants.reduce((sum, variant) => sum + variant.outputBytes, 0),
        usage: BUFFER_USAGE.COPY_DST | BUFFER_USAGE.MAP_READ,
      });
      buffers.push(staging);

      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      for (const variant of variants) {
        pass.setPipeline(variant.pipeline);
        pass.setBindGroup(
          0,
          device.createBindGroup({
            layout: variant.pipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: variant.params } },
              ...variant.bindings.map(([binding, buffer]) => ({
                binding,
                resource: { buffer },
              })),
            ],
          }),
        );
        pass.dispatchWorkgroups(...variant.workgroups);
      }
      pass.end();
      let offset = 0;
      for (const variant of variants) {
        encoder.copyBufferToBuffer(variant.output, 0, staging, offset, variant.outputBytes);
        offset += variant.outputBytes;
      }
      device.queue.submit([encoder.finish()]);
    } catch (cause) {
      // MUST: always pop the two pushed scopes and leave nothing on the stack (leftovers make later
      // validation results land in the wrong scope, and errors become permanently invisible).
      await discardFailureScopes(device);
      throw cause;
    }
    const failure = await gpu[RUNTIME_INTERNAL].raceDeviceLost(
      popFailureScopes(device, WHERE),
      `${WHERE}の検証`,
    );
    // MUST: raise validation / OOM as a loud failure, not as "differs from the known answer". An
    // invalid dispatch returns all 0, so collapsing it into a mismatch would misdiagnose "this
    // device's integer attention is broken".
    if (failure !== undefined) throw failure;

    await gpu[RUNTIME_INTERNAL].raceDeviceLost(
      staging.mapAsync(MAP_MODE.READ),
      `${WHERE}の読み戻し`,
    );
    const mapped = staging.getMappedRange().slice(0);
    staging.unmap();
    let offset = 0;
    const outcomes: CanaryVariantOutcome[] = [];
    for (const variant of variants) {
      outcomes.push(variant.compare(mapped.slice(offset, offset + variant.outputBytes)));
      offset += variant.outputBytes;
    }
    return {
      dot,
      exact: outcomes.every((outcome) => outcome.exact),
      withinBand: outcomes.every((outcome) => outcome.withinBand),
      maxAbsError: outcomes.reduce((worst, o) => Math.max(worst, o.maxAbsError), 0),
      maxBandRatio: outcomes.reduce((worst, o) => Math.max(worst, o.maxBandRatio), 0),
      // The aggregation across variants also prefers outside-the-band (the same reason as
      // compareToReference — when a variant outside the band exists, the 1 ULP difference of an
      // earlier variant inside the band is not made the representative).
      mismatch: (
        outcomes.find((outcome) => !outcome.withinBand) ??
        outcomes.find((outcome) => outcome.mismatch !== undefined)
      )?.mismatch,
      variants: outcomes,
    };
  } finally {
    for (const buffer of buffers) buffer.destroy();
  }
};

/**
 * Which branch decided {@link AttentionI8a8Decision} (decision rule v2 — §Decision rule v2 at the
 * top of the module).
 */
export type CanaryBranch =
  | "dp4a-exact"
  | "emu-exact"
  | "band-single-arm"
  | "band-both-identical"
  | "band-both-differ";

/**
 * The Japanese rationale per branch (the head of the warning message — {@link
 * formatAttentionI8a8Decision}).
 */
const BRANCH_NOTE: Record<CanaryBranch, string> = {
  "dp4a-exact": "dot4I8Packed 版が既知解と厳密一致した",
  "emu-exact": "dot4I8Packed 版だけが既知解を外し、エミュ版は厳密一致した",
  "band-single-arm": "厳密一致した腕が無く、sanity 帯に収まったのは 1 腕だけだった",
  "band-both-identical":
    "両腕とも厳密一致は外したが sanity 帯に収まり、腕同士は全 6 変種でビット同一だった",
  "band-both-differ":
    "両腕とも厳密一致は外したが sanity 帯に収まり、腕同士は相違したので既知解に近い側を採った",
};

/** The canary's verdict ({@link decideAttentionI8a8Dot}). */
export type AttentionI8a8Decision = {
  readonly dot: I8a8Dot;
  /**
   * Whether the chosen arm matched the known answer with **atol=0**. false means "inside the band
   * but not exact" — the caller does not stop execution and warns once per device (it does not pass
   * silently).
   */
  readonly exact: boolean;
  readonly branch: CanaryBranch;
  /** The chosen arm's maximum absolute error against the known answer (0 when {@link exact}). */
  readonly maxAbsError: number;
  /** The chosen arm's maximum band ratio (`|error| / tolerance` — at most 1 is inside the band). */
  readonly maxBandRatio: number;
};

/** The one-line rationale of the verdict (wording shared by the warning and the diagnostics). */
export const formatAttentionI8a8Decision = (decision: AttentionI8a8Decision): string =>
  `${BRANCH_NOTE[decision.branch]}ので '${decision.dot}' を選んだ` +
  `（既知解との最大絶対誤差 ${decision.maxAbsError} / 帯余裕 ${decision.maxBandRatio}）`;

const decide = (probe: CanaryProbe, branch: CanaryBranch): AttentionI8a8Decision => ({
  dot: probe.dot,
  exact: probe.exact,
  branch,
  maxAbsError: probe.maxAbsError,
  maxBandRatio: probe.maxBandRatio,
});

/**
 * Whether the arms are bit-identical in **every element of all 6 variants**. If even 1 byte of 1
 * variant differs, they "differ".
 *
 * The read-back bytes are compared as is (f32 and f16 storage are judged by the same rule). The
 * order of variants is the generation order and is the same across arms, so they pair by index.
 */
const armsBitIdentical = (a: CanaryProbe, b: CanaryProbe): boolean =>
  a.variants.length === b.variants.length &&
  a.variants.every((variant, index) => {
    const other = b.variants[index].output;
    return (
      variant.output.length === other.length &&
      variant.output.every((byte, at) => byte === other[at])
    );
  });

/**
 * Decides the integer-dot-product variant of fused attention by **running on this device** (once
 * per device — memoization is on the {@link GpuContext} side). The procedure is exactly steps 1 to
 * 4 of §Decision rule v2 at the top of the module:
 *
 * - dp4a matches the known answer exactly → `"dp4a"` (**the emu side is not fired on this path** —
 *   the cost paid by a healthy machine stays at 1 submit. A machine whose emu is broken is
 *   irrelevant once dp4a is chosen).
 * - only dp4a mismatches → `"emu"` (linear has its own option, so linear stays on dp4a).
 * - neither matches exactly → judge by the sanity band (only one arm inside / both inside and
 *   bit-identical / both inside and different).
 * - both arms outside the band → {@link GpuFeatureError}. It is not a matter of how the variant is
 *   chosen, so there is nothing to fall back to.
 *
 * `patch` is a hook to replace the generated output, **for fault-injection tests only** (the
 * mismatch path and the both-mismatch path cannot be produced on a healthy real machine — the same
 * situation as the shader-f16 canary in src/gpu/acquire.ts).
 */
export const decideAttentionI8a8Dot = (
  gpu: GpuContext,
  geometry: AttentionI8a8CanaryGeometry,
  patch: CanaryWgslPatch = IDENTITY_PATCH,
): Promise<AttentionI8a8Decision> =>
  gpu[RUNTIME_INTERNAL].withScopeLock(async () => {
    const dp4a = await probeAttentionI8a8Dot(gpu, "dp4a", geometry, patch);
    if (dp4a.exact) return decide(dp4a, "dp4a-exact");
    const emu = await probeAttentionI8a8Dot(gpu, "emu", geometry, patch);
    if (emu.exact) return decide(emu, "emu-exact");
    if (dp4a.withinBand !== emu.withinBand) {
      return decide(dp4a.withinBand ? dp4a : emu, "band-single-arm");
    }
    if (dp4a.withinBand) {
      // Bit-identical arms mean "the variant choice does not affect the numbers" (ADR 0058 decision
      // 2) **has been verified mechanically on this device**, so the default dp4a is taken as is.
      // The M2, whose shared epilogue rounding is only a few ULP off the known answer, comes here.
      if (armsBitIdentical(dp4a, emu)) return decide(dp4a, "band-both-identical");
      // If they differ, the variant choice moves the numbers. Inside the band, so execution does
      // not stop, but the arm closer to the known answer is taken (ties go to dp4a — there is no
      // ground for moving away from the default).
      return decide(emu.maxAbsError < dp4a.maxAbsError ? emu : dp4a, "band-both-differ");
    }
    throw new GpuFeatureError(
      `融合 attention の i8a8 カーネルが dot4I8Packed 版・エミュ版とも既知解の sanity 帯` +
        `（相対 ${SANITY_RELATIVE_TOLERANCE} / 絶対床 ${SANITY_ABSOLUTE_FLOOR}）を外した` +
        `（dp4a: ${dp4a.mismatch} 帯余裕 ${dp4a.maxBandRatio} / ` +
        `emu: ${emu.mismatch} 帯余裕 ${emu.maxBandRatio}）。` +
        "共有 f32 エピローグの丸め差（帯内）は既に許容した上で外しているので、" +
        "整数内積の変種選択の問題ではなく、この device の整数 attention 自体が信用できない — " +
        "attentionCompute を 'f32' か 'f16' にして実行すること",
    );
  });
