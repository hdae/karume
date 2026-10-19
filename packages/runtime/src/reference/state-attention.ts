/**
 * CPU reference of states-form attention (ADR 0067 decisions 4 / 6) and `state_append`
 * (decision 5).
 *
 * The oracle's condition is **not to share the structure** of the GPU side
 * (src/kernels/state-attention.ts):
 *
 * - The GPU uses 3 dispatches (materialize S → row statistics → non-materialized PV), while the
 *   reference is written as **three plain steps** (S → materialize P → PV). Errors in the fused
 *   steps do not cancel out between the two sides (the same discipline as the MUST of
 *   `referenceAttention`).
 * - Reductions **accumulate in f64 and round to f32 once, when stored** (not matching the GPU's
 *   sequential f32 accumulation bit for bit is correct — which is why allclose decides).
 * - The logical col → physical row mapping, the two-sided predicate and empty row → 0 are **the
 *   semantics itself**, so they are the same expressions. The comparison does not become vacuous
 *   even so (the GPU side spans 3 dispatches and buffers, the reference is one loop; only the
 *   values agree).
 * - The scan range is **not shared** with the GPU side. The reference walks the logical columns
 *   `[0, P+Q)` from the top and selects columns by the predicate alone (stopping where it falls
 *   below the window's lower bound is valid because the window predicate is monotonic in col — it
 *   gives the same set as skipping the columns outside the predicate). Borrowing the GPU's live
 *   range (`stateColumnBase` / `stateLiveColumns`) would let both sides share an error where that
 *   range misses columns by being too narrow, and the comparison would become vacuous (an error on
 *   the wide side is harmless because the predicate drops it).
 *
 * MUST: empty rows (rows where no col satisfies the predicate) and contributions outside the
 * predicate are **exactly 0** (the counterpart of a check by exact comparison, not tolerance —
 * ADR 0067 decision 6). The reference **skips without reading** the K / V of columns outside the
 * predicate (as in the formula `O[row] = Σ_{col in predicate} P[row,col] · V(col)`). The GPU also
 * adds weight 0 × V for the live-range columns outside the predicate, so the two sides agree
 * **when V is finite**. With a finite V, adding 0 × V = 0 does not change the sum, and the
 * reference's value is bit-identical to the form that adds 0 × V over the live range.
 * When V holds NaN / ±Inf, the GPU gives NaN through 0 × NaN on the rows whose live range contains
 * that column, while the reference does not — non-finite values are outside the contract
 * (docs/limitations.md "Non-finite values (NaN / ±Infinity) are not checked").
 * MUST: pad rows (`row ≥ queryLength`) **are written as exact 0** (the output is fully written, and
 * "the values carry no meaning by contract" of ADR 0066 addendum 6 was fixed at 0 — the GPU side
 * satisfies it by skipping the live scan of pad rows entirely. Empty rows ⊂ pad rows, so this
 * convention contains empty row → 0). Conversely, `state_append` writes only the first
 * `queryLength` rows (slots are outside the full-write rule).
 */

import { stateSliding } from "../kernels/state-attention.ts";
import { ReferenceOpError, type RefTensor } from "./ops.ts";

/**
 * Shape and logical lengths of the states form (symbols from ADR 0067 decision 4 — the table in
 * src/kernels/state-attention.ts).
 */
export type StateAttentionRefShape = {
  /** `B`. */
  readonly batch: number;
  /** `H` (number of q heads). */
  readonly heads: number;
  /** `Hkv` (number of k / v heads — `H % Hkv == 0`). */
  readonly kvHeads: number;
  /** `M` (physical chunk rows). */
  readonly chunkRows: number;
  /** `D`. */
  readonly depth: number;
  /** `C` (row capacity of the slot). */
  readonly capacity: number;
  /** `W` (`0` = full). */
  readonly window: number;
  /** `P` (pastLength). */
  readonly past: number;
  /** `Q` (queryLength — `1 ≤ Q ≤ M`). */
  readonly query: number;
};

export type StateAttentionRefInput = StateAttentionRefShape & {
  /** `[B,H,M,D]`. */
  readonly q: Float32Array<ArrayBuffer>;
  /** This step's k `[B,Hkv,M,D]` (valid data is the first `Q` rows). */
  readonly insK: Float32Array<ArrayBuffer>;
  /** This step's v `[B,Hkv,M,D]`. */
  readonly insV: Float32Array<ArrayBuffer>;
  /** The past k slot `[B,Hkv,C,D]`. */
  readonly slotK: Float32Array<ArrayBuffer>;
  /** The past v slot `[B,Hkv,C,D]`. */
  readonly slotV: Float32Array<ArrayBuffer>;
  /**
   * The half scale (applied to **both** the q side and the k side — this oracle detects an
   * implementation that applies it to one side only).
   */
  readonly scale: number;
};

/**
 * Range of the logical length `past` and the window `window` (both non-negative integers).
 *
 * MUST: checked before the comparison with the capacity. A negative `window` makes `stateSliding`
 * false and is silently treated as full, and a negative `past` collapses the live columns to 0 and
 * returns exact 0 — both are forms where "the oracle silently returns another number", giving a
 * wrong expected value instead of a red comparison.
 */
const assertPastAndWindow = (where: string, past: number, window: number): void => {
  if (!Number.isSafeInteger(past) || past < 0) {
    throw new ReferenceOpError(`${where}: pastLength ${past} is not a non-negative integer`);
  }
  if (!Number.isSafeInteger(window) || window < 0) {
    throw new ReferenceOpError(
      `${where}: window ${window} is not a non-negative integer (0 = full)`,
    );
  }
};

/**
 * Range of queryLength (an integer with `1 ≤ Q ≤ M`).
 *
 * MUST: check for an integer before the range comparison. Every comparison with NaN is false, so it
 * slips past the range check (the row loop runs 0 times and the output is all 0), and 1.5 returns
 * NaN through a fractional live column count — both silently return another number.
 */
const assertQuery = (query: number, chunkRows: number): void => {
  if (!Number.isSafeInteger(query) || query < 1 || query > chunkRows) {
    throw new ReferenceOpError(`queryLength ${query} is not an integer in 1..${chunkRows} (= M)`);
  }
};

/**
 * Consistency of shape and logical lengths (mix-ups fail before the comparison).
 *
 * MUST: check the dimensions for integers before the range comparison (the same reason as
 * {@link assertQuery}). A non-integer passes the array-length check too when the product of the
 * dimensions is an integer, and silently returns an output of a fractional shape (e.g. D=1.5). NaN
 * slips past because `< 1` is false, and only the later array-length check catches it by accident
 * (with a message that does not name a shape error).
 */
const assertShape = (shape: StateAttentionRefShape): void => {
  const { batch, heads, kvHeads, chunkRows, depth, capacity, window, past, query } = shape;
  assertPastAndWindow("states form", past, window);
  const dims = [batch, heads, kvHeads, chunkRows, depth, capacity];
  if (dims.some((dim) => !Number.isSafeInteger(dim) || dim < 1)) {
    throw new ReferenceOpError(
      `states form shape is not all positive integers (B=${batch} H=${heads} Hkv=${kvHeads} M=${chunkRows} D=${depth} C=${capacity})`,
    );
  }
  if (heads % kvHeads !== 0) {
    throw new ReferenceOpError(
      `H=${heads} is not divisible by Hkv=${kvHeads} (ADR 0067 decision 1)`,
    );
  }
  assertQuery(query, chunkRows);
  if (stateSliding(window)) {
    if (window > capacity) {
      throw new ReferenceOpError(`window ${window} exceeds capacity ${capacity}`);
    }
  } else if (past + query > capacity) {
    throw new ReferenceOpError(
      `In a full slot, pastLength ${past} + queryLength ${query} exceeds capacity ${capacity} (ADR 0067 decision 4 ④)`,
    );
  }
};

/** Whether a given array has the expected length (an argument mix-up fails as a shape mismatch). */
const assertLength = (name: string, data: Float32Array<ArrayBuffer>, expected: number): void => {
  if (data.length !== expected) {
    throw new ReferenceOpError(
      `${name}: element count ${data.length} does not match the expected ${expected}`,
    );
  }
};

/**
 * Logical col → physical slot row (**the same formula for reads and writes** — a copy of the GPU
 * side's `slot_row`).
 *
 * MUST: use the same function as the write of `state_append` ({@link referenceStateAppend}). If
 * the reference's read and write formulas diverged, the oracle would reproduce the same error as
 * the GPU side and the comparison would become vacuous.
 * MUST: the modulus is **the capacity `C`** (not the window `W` — the doc of `stateSlotRowWgsl` in
 * `src/kernels/state-attention.ts`. The slack of `C > W` holds "rows written ahead of the logical
 * length").
 */
const slotRow = (window: number, capacity: number, col: number): number =>
  stateSliding(window) ? col % capacity : col;

/**
 * States-form attention. The output is `[B,H,M,D]` (all M rows including pad rows).
 *
 * The predicate is **two-sided** (causal `col ≤ P + row` AND sliding
 * `col ≥ max(0, P + row − W + 1)`). Columns are selected from the logical columns `[0, P+Q)` by the
 * predicate alone, and columns outside it are skipped without reading K / V (module doc).
 * An empty row outputs exact 0 (the structural 0 when the row max is −inf — ADR 0067 decision 6).
 * Pad rows (`row ≥ query`) are exact 0 too (`out` keeps its 0 initialization — the value contract
 * in the module doc).
 */
export const referenceStateAttention = (input: StateAttentionRefInput): RefTensor => {
  assertShape(input);
  const { batch, heads, kvHeads, chunkRows, depth, capacity, window, past, query } = input;
  const repeat = heads / kvHeads;
  assertLength("q", input.q, batch * heads * chunkRows * depth);
  assertLength("insK", input.insK, batch * kvHeads * chunkRows * depth);
  assertLength("insV", input.insV, batch * kvHeads * chunkRows * depth);
  assertLength("slotK", input.slotK, batch * kvHeads * capacity * depth);
  assertLength("slotV", input.slotV, batch * kvHeads * capacity * depth);
  const scale = Math.fround(input.scale);
  // Total number of logical columns. Column numbers are kept as number (f64) — sliding's P can
  // reach the top of u32, so packing them into Int32 would wrap values ≥ 2^31 to negative.
  const columns = past + query;
  const shape = [batch, heads, chunkRows, depth];
  const out = new Float32Array(batch * heads * chunkRows * depth);
  for (let plane = 0; plane < batch * heads; plane += 1) {
    // The kv plane is an **integer division** (`z / r = b·Hkv + h/r` — the same identity as the
    // GPU's `wid.z / r`)
    const kvPlane = Math.floor(plane / repeat);
    // MUST: only valid rows run (pad rows keep their 0 initialization = the same value as the
    // exact 0 the GPU's ③ writes).
    for (let row = 0; row < query; row += 1) {
      const qBase = (plane * chunkRows + row) * depth;
      const limit = past + row;
      // Collect the columns that satisfy the predicate by walking from the top of the logical
      // columns. Skip those above the causal upper bound and stop once below the window's lower
      // bound — the window predicate `limit − col < W` is monotonic, becoming false as col gets
      // smaller, so once a column falls out every column below it does too (sliding's P can reach
      // the top of u32, so walking down to 0 would never end the reference).
      const selected: number[] = [];
      for (let col = columns - 1; col >= 0; col -= 1) {
        if (col > limit) continue;
        // The lower bound has the same form `limit − col < W` as the GPU side (`in_window`) (the
        // form safe against u32 wraparound)
        if (stateSliding(window) && limit - col >= window) break;
        selected.push(col);
      }
      // Sums accumulate in ascending logical-column order (collected in descending order).
      selected.reverse();
      // ① S (only the columns that satisfy the predicate; K is not read for the others)
      const scores = selected.map((col) => {
        const kBase =
          col < past
            ? (kvPlane * capacity + slotRow(window, capacity, col)) * depth
            : (kvPlane * chunkRows + (col - past)) * depth;
        const source = col < past ? input.slotK : input.insK;
        let acc = 0;
        for (let d = 0; d < depth; d += 1) {
          acc += input.q[qBase + d] * scale * (source[kBase + d] * scale);
        }
        return Math.fround(acc);
      });
      // ② safe-softmax (identity −inf; an empty row is all 0)
      let amax = Number.NEGATIVE_INFINITY;
      for (const score of scores) amax = Math.max(amax, score);
      if (amax === Number.NEGATIVE_INFINITY) {
        // An empty row outputs **exact 0** (out is 0-initialized, so go to the next row without
        // writing)
        continue;
      }
      let total = 0;
      for (const score of scores) total += Math.exp(score - amax);
      const weights = scores.map((score) => Math.fround(Math.exp(score - amax) / total));
      // ③ O[row, d] = Σ_{col in predicate} P[row,col] · V(col, d) (V outside the predicate is not
      // read)
      for (let d = 0; d < depth; d += 1) {
        let acc = 0;
        for (const [i, col] of selected.entries()) {
          const vBase =
            col < past
              ? (kvPlane * capacity + slotRow(window, capacity, col)) * depth
              : (kvPlane * chunkRows + (col - past)) * depth;
          const source = col < past ? input.slotV : input.insV;
          acc += weights[i] * source[vBase + d];
        }
        out[qBase + d] = Math.fround(acc);
      }
    }
  }
  return { dtype: "f32", shape, data: out };
};

export type StateAppendRefInput = {
  /** `B·Hkv` (number of slot planes). */
  readonly kvPlanes: number;
  /** `M` (row stride of the input). */
  readonly chunkRows: number;
  /** `D`. */
  readonly depth: number;
  /** `C` (row capacity of the slot). */
  readonly capacity: number;
  /** `W` (`0` = full). */
  readonly window: number;
  /** `P` (start of the logical positions written to). */
  readonly past: number;
  /** `Q` (**only this many rows are written**). */
  readonly query: number;
  /** This step's k or v `[B,Hkv,M,D]`. */
  readonly x: Float32Array<ArrayBuffer>;
  /** The slot before the write `[B,Hkv,C,D]` (**not modified** — a copy is returned). */
  readonly slot: Float32Array<ArrayBuffer>;
};

/**
 * `state_append`. Returns **a copy of the slot after the write** (`[B,Hkv,C,D]`).
 *
 * MUST: do not modify the input `slot` (in a test that reuses the same array across cases, the
 * previous case's write would leak into the next case's "before the write").
 * MUST: only the first `Q` rows are written. Not writing pad rows shows up in the values (leftovers
 * remain).
 */
export const referenceStateAppend = (input: StateAppendRefInput): RefTensor => {
  const { kvPlanes, chunkRows, depth, capacity, window, past, query } = input;
  assertPastAndWindow("state_append", past, window);
  if (kvPlanes < 1 || chunkRows < 1 || depth < 1 || capacity < 1) {
    throw new ReferenceOpError(
      `state_append shape is not positive (B·Hkv=${kvPlanes} M=${chunkRows} D=${depth} C=${capacity})`,
    );
  }
  assertQuery(query, chunkRows);
  if (stateSliding(window)) {
    if (window > capacity) {
      throw new ReferenceOpError(`window ${window} exceeds capacity ${capacity}`);
    }
  } else if (past + query > capacity) {
    throw new ReferenceOpError(
      `In a full slot, pastLength ${past} + queryLength ${query} exceeds capacity ${capacity}`,
    );
  }
  assertLength("x", input.x, kvPlanes * chunkRows * depth);
  assertLength("slot", input.slot, kvPlanes * capacity * depth);
  const out = input.slot.slice();
  for (let plane = 0; plane < kvPlanes; plane += 1) {
    for (let row = 0; row < query; row += 1) {
      const src = (plane * chunkRows + row) * depth;
      const dst = (plane * capacity + slotRow(window, capacity, past + row)) * depth;
      for (let d = 0; d < depth; d += 1) out[dst + d] = input.x[src + d];
    }
  }
  return { dtype: "f32", shape: [kvPlanes, capacity, depth], data: out };
};

/**
 * Shape and logical lengths of readonly (reads the past only — ADR 0096 decision 1). `M = Q = 1`
 * is fixed, so there are no fields for them.
 */
export type StateAttentionReadonlyRefInput = {
  readonly batch: number;
  readonly heads: number;
  readonly kvHeads: number;
  readonly depth: number;
  /** `C` (row capacity of the lender's slot). */
  readonly capacity: number;
  /** `W` (`0` = full). */
  readonly window: number;
  /** `P` (the lender's pastLength — the query is at logical position `P−1`). */
  readonly past: number;
  /** `[B,H,1,D]`. */
  readonly q: Float32Array<ArrayBuffer>;
  /** The lender's k slot `[B,Hkv,C,D]`. */
  readonly slotK: Float32Array<ArrayBuffer>;
  /** The lender's v slot `[B,Hkv,C,D]`. */
  readonly slotV: Float32Array<ArrayBuffer>;
  readonly scale: number;
};

/**
 * readonly attention (the drafter's cross-attention — ADR 0096 decision 1). The output is
 * `[B,H,1,D]`.
 *
 * One query row at position `P−1` reads the slot columns `[P − min(P, W), P)` (`[0, P)` for full).
 * There is **no** ins for the current step. The predicate is `col < P` (rejected rows = rows
 * written ahead of the logical length are not read) AND sliding `P−1−col < W`. `P = 0` is an empty
 * row → exact 0. The numeric contract (half scale, −inf identity, accumulate in f64 and round to
 * f32 once when stored) is the same as {@link referenceStateAttention}.
 */
export const referenceStateAttentionReadonly = (
  input: StateAttentionReadonlyRefInput,
): RefTensor => {
  const { batch, heads, kvHeads, depth, capacity, window, past } = input;
  assertPastAndWindow("readonly", past, window);
  if (batch < 1 || heads < 1 || kvHeads < 1 || depth < 1 || capacity < 1) {
    throw new ReferenceOpError(
      `readonly shape is not positive (B=${batch} H=${heads} Hkv=${kvHeads} D=${depth} C=${capacity})`,
    );
  }
  if (heads % kvHeads !== 0) {
    throw new ReferenceOpError(
      `H=${heads} is not divisible by Hkv=${kvHeads} (ADR 0067 decision 1)`,
    );
  }
  if (stateSliding(window) ? window > capacity : past > capacity) {
    throw new ReferenceOpError(
      `readonly: window ${window} / pastLength ${past} does not fit in capacity ${capacity}`,
    );
  }
  const repeat = heads / kvHeads;
  assertLength("q", input.q, batch * heads * depth);
  assertLength("slotK", input.slotK, batch * kvHeads * capacity * depth);
  assertLength("slotV", input.slotV, batch * kvHeads * capacity * depth);
  const scale = Math.fround(input.scale);
  const base = stateSliding(window) ? past - Math.min(past, window) : 0;
  const live = stateSliding(window) ? Math.min(past, window) : past;
  const out = new Float32Array(batch * heads * depth);
  const scores = new Float32Array(live);
  const weights = new Float32Array(live);
  for (let plane = 0; plane < batch * heads; plane += 1) {
    const kvPlane = Math.floor(plane / repeat);
    const qBase = plane * depth;
    for (let cl = 0; cl < live; cl += 1) {
      const col = base + cl;
      const inWindow = col < past && (!stateSliding(window) || past - 1 - col < window);
      if (!inWindow) {
        scores[cl] = Number.NEGATIVE_INFINITY;
        continue;
      }
      const kBase = (kvPlane * capacity + slotRow(window, capacity, col)) * depth;
      let acc = 0;
      for (let d = 0; d < depth; d += 1) {
        acc += input.q[qBase + d] * scale * (input.slotK[kBase + d] * scale);
      }
      scores[cl] = Math.fround(acc);
    }
    let amax = Number.NEGATIVE_INFINITY;
    for (let cl = 0; cl < live; cl += 1) amax = Math.max(amax, scores[cl]);
    if (amax === Number.NEGATIVE_INFINITY) continue;
    let total = 0;
    for (let cl = 0; cl < live; cl += 1) total += Math.exp(scores[cl] - amax);
    for (let cl = 0; cl < live; cl += 1) {
      weights[cl] = Math.fround(Math.exp(scores[cl] - amax) / total);
    }
    for (let d = 0; d < depth; d += 1) {
      let acc = 0;
      for (let cl = 0; cl < live; cl += 1) {
        const col = base + cl;
        acc +=
          weights[cl] *
          input.slotV[(kvPlane * capacity + slotRow(window, capacity, col)) * depth + d];
      }
      out[qBase + d] = Math.fround(acc);
    }
  }
  return { dtype: "f32", shape: [batch, heads, 1, depth], data: out };
};
