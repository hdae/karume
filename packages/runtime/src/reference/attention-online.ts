/**
 * The TS mirror of the online int8 attention kernel (src/kernels/attention-online-i8a8.ts — ADR
 * 0132). It walks the same 64-key tiles in the same order and performs the same f32 operations in
 * the same order (running max, `α`, the per-lane partial sums of `l` and their butterfly, the
 * rescaled f32 output, the final `O · vs · (1/127) / l`), so a structural error in the kernel — a
 * wrong lane / key mapping, a missing rescale, a tile boundary — shows up as a large difference,
 * while the only legitimate differences are the ULP-level ones of `exp` and the division (which
 * move P̃ by at most one lattice step).
 *
 * MUST: this is **not** the contract oracle of the kernel. The contract (class Q, ADR 0110) is the
 * error model against the exact attention of the dequantised inputs, held by
 * tests/gpu_attention_online_i8a8_test.ts; this mirror is the structural check beside it.
 * MUST: the integer products are exact (a JS number holds `n · 127²` exactly), the f32 operations
 * are `Math.fround` after every operation the GPU rounds.
 */

import { quantizeRowsReference, roundTiesToEven } from "./i8a8.ts";

const TILE = 64;
const LANES = 16;
const P_LATTICE_MAX = 127;
const INV_P = Math.fround(1 / P_LATTICE_MAX);
/** The identity for keys beyond N (the kernel's `NEG`). */
const NEG = -3.0e38;

export type AttentionOnlineInput = {
  /** q `[batch, m, d]` (f32, before quantisation; batch folds B·H). */
  readonly q: ArrayLike<number>;
  /** k `[batch, n, d]`. */
  readonly k: ArrayLike<number>;
  /** v `[batch, n, d]`. */
  readonly v: ArrayLike<number>;
  readonly batch: number;
  readonly m: number;
  readonly n: number;
  readonly d: number;
  /** The half scale (applied to both q and k on the dequant side). */
  readonly scale: number;
};

/** Vᵀ `[batch, d, n]` from v `[batch, n, d]` (the job of the `strided` permute on the GPU). */
export const transposeValues = (
  v: ArrayLike<number>,
  batch: number,
  n: number,
  d: number,
): Float32Array<ArrayBuffer> => {
  const out = new Float32Array(batch * d * n);
  for (let b = 0; b < batch; b += 1) {
    for (let row = 0; row < n; row += 1) {
      for (let col = 0; col < d; col += 1) {
        out[(b * d + col) * n + row] = v[(b * n + row) * d + col];
      }
    }
  }
  return out;
};

/**
 * The 16-lane butterfly sum of the kernel (`subgroupShuffleXor` with masks 8, 4, 2, 1 — every lane
 * ends with the same bits; the workgroup-memory form spells the same pairing).
 */
export const onlineLaneSum = (partials: ArrayLike<number>): number => {
  const a = Array.from({ length: 8 }, (_, i) => Math.fround(partials[i] + partials[i + 8]));
  const b = Array.from({ length: 4 }, (_, i) => Math.fround(a[i] + a[i + 4]));
  const c = [Math.fround(b[0] + b[2]), Math.fround(b[1] + b[3])];
  return Math.fround(c[0] + c[1]);
};

/** `O[batch, m, d]` of the online kernel, mirrored operation by operation. */
export const referenceAttentionOnlineI8a8 = (
  input: AttentionOnlineInput,
): Float32Array<ArrayBuffer> => {
  const { batch, m, n, d } = input;
  const scale = Math.fround(input.scale);
  const query = quantizeRowsReference(input.q, batch * m, d);
  const key = quantizeRowsReference(input.k, batch * n, d);
  const values = quantizeRowsReference(transposeValues(input.v, batch, n, d), batch * d, n);
  const out = new Float32Array(batch * m * d);
  const tiles = Math.ceil(n / TILE);
  const scores = new Float32Array(TILE);
  const weights = new Int32Array(TILE);
  const acc = new Float32Array(d);
  const lane = new Float32Array(LANES);
  for (let b = 0; b < batch; b += 1) {
    for (let row = 0; row < m; row += 1) {
      const qRow = (b * m + row) * d;
      const qsc = Math.fround(query.scale[b * m + row] * scale);
      let max = NEG;
      lane.fill(0);
      acc.fill(0);
      for (let tile = 0; tile < tiles; tile += 1) {
        const key0 = tile * TILE;
        let tileMax = NEG;
        for (let j = 0; j < TILE; j += 1) {
          const col = key0 + j;
          if (col >= n) {
            scores[j] = NEG;
            continue;
          }
          let dot = 0;
          for (let i = 0; i < d; i += 1) {
            dot += query.q[qRow + i] * key.q[(b * n + col) * d + i];
          }
          const ksc = Math.fround(key.scale[b * n + col] * scale);
          // MUST: the dequant folds (qs·scale)·(ks·scale) first (the order of ①QK and the kernel)
          scores[j] = Math.fround(Math.fround(dot) * Math.fround(qsc * ksc));
          if (scores[j] > tileMax) tileMax = scores[j];
        }
        const next = Math.max(max, tileMax);
        const alpha = Math.fround(Math.exp(Math.fround(max - next)));
        max = next;
        for (let i = 0; i < LANES; i += 1) lane[i] = Math.fround(lane[i] * alpha);
        for (let col = 0; col < d; col += 1) acc[col] = Math.fround(acc[col] * alpha);
        // Lane tx owns the keys tx, tx+16, tx+32, tx+48 of the tile and adds them in that order
        for (let tx = 0; tx < LANES; tx += 1) {
          for (let i = 0; i < 4; i += 1) {
            const j = tx + LANES * i;
            const p = Math.fround(Math.exp(Math.fround(scores[j] - next)));
            lane[tx] = Math.fround(lane[tx] + p);
            weights[j] = roundTiesToEven(Math.fround(p * P_LATTICE_MAX));
          }
        }
        for (let col = 0; col < d; col += 1) {
          let dot = 0;
          for (let j = 0; j < TILE; j += 1) {
            if (key0 + j < n) dot += weights[j] * values.q[(b * d + col) * n + key0 + j];
          }
          acc[col] = Math.fround(acc[col] + Math.fround(dot));
        }
      }
      const sum = onlineLaneSum(lane);
      const factor = Math.fround(INV_P / sum);
      for (let col = 0; col < d; col += 1) {
        out[qRow + col] = Math.fround(Math.fround(acc[col] * values.scale[b * d + col]) * factor);
      }
    }
  }
  return out;
};
