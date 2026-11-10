/**
 * The **pure part** of speculative decoding (draft acceptance, cut-off at a stop token, building
 * the verify rows). Called by the speculative path of `sequence.ts`. It is **pipeline-independent
 * and GPU-independent**, so it lives in `src/generation/` next to `sampler.ts` (DECIDED: ADR 0096
 * decisions 3 / 5 / 7, addendum (stage 3)).
 *
 * ## The shape of one cycle (ADR 0096 decisions 5 to 7)
 *
 * Let `P` be the conversation's logical length (rows the target has consumed) and `b` the frontier
 * (the confirmed token at position `P` — not yet in the KV). The drafter produces `d₁..d_k` from
 * `(b, the hidden of the row that produced b, position P)`, and verify runs the `k'+1` rows
 * `[b, d₁..d_k']` in one run. Row `j` processes token `[b, d₁..][j]` at position `P+j`, so row `j`'s
 * logits are **the prediction for position `P+j+1`**.
 *
 * Acceptance is the matching prefix: it advances only while row `j`'s sampled `t_j` equals
 * `d_{j+1}`; the `j` where it stops is the accepted count `a`, and that `t_a` is the new frontier
 * `b'`. The `a+1` tokens `[d₁..d_a, b']` are confirmed, and `b'` is always confirmed even when
 * `a = 0` (= it degenerates to one non-speculative decode step — speculation can only be slower, it
 * never changes the result).
 *
 * However, **when a confirmed token is a stop token the enumeration stops right there**.
 * Non-speculative decoding ends generation at that point, so it never looks at the logits of later
 * rows nor calls `sampler.next` — going on without stopping would produce cycles that fail on logits
 * non-speculative decoding never touches (NaN contamination from an out-of-range gather, for
 * example), and would consume extra RNG draws. In a cycle whose accepted draft was a stop token, the
 * confirmed sequence is the `a` tokens `[d₁..d_a]`.
 *
 * The draw yields the same sequence as non-speculative decoding **regardless of temperature**. The
 * per-row `sampler.next` is called with the same logits, the same `history` (the confirmed sequence
 * up to that row) and in the same order as the draw non-speculative decode makes at the same
 * position, and consumes the RNG exactly once per confirmed token (rows past a rejection are not
 * drawn), so even the RNG consumption sequence matches. At temperature > 0, "accept when the drawn
 * token equals the draft" is speculative sampling with a one-hot draft (acceptance probability = the
 * draft's probability under the target distribution; the drawn token becomes the new frontier as
 * is).
 */

import type { Sampler } from "./sampler.ts";

/** One cycle's input to the drafter (built by `sequence.ts` — meaning in the module doc above). */
export type DraftCycle = {
  /** The frontier `b` (the confirmed token at position `P` — not yet in the KV). */
  readonly token: number;
  /**
   * Final-norm hidden of the row that produced `b` (position `P−1`) (a copy taken by the sequence —
   * the caller can keep it).
   */
  readonly hidden: Float32Array<ArrayBuffer>;
  /** The logical position `P` of `b` (= the drafter's query position). */
  readonly position: number;
};

/**
 * The surface the generation loop requires of a drafter (narrow interface — DI lets a fake stand
 * in).
 *
 * The real one is gemma4's borrowing context (`src/gemma/speculative.ts`), but this surface knows
 * neither Session nor GenerationContext — a fake that forces zero acceptance or one that puts a stop
 * token in the middle of a draft can be written without a GPU (so no test-only injection point goes
 * on the public surface).
 */
export type DraftFace = {
  /** Drafts produced per run (the step count baked into the distribution — 3 for gemma4). */
  readonly steps: number;
  /** `steps` draft tokens (`d₁, d₂, …` from the front). The caller uses only the first `k'`. */
  draft(cycle: DraftCycle): Promise<Int32Array<ArrayBuffer>>;
  /** Closes the borrower (called **before** the lending context — the sequence owns the order). */
  dispose(): Promise<void>;
};

/**
 * The number of drafts used in this cycle, `k' = min(k, remaining − 1)`.
 *
 * At most `k'+1` tokens are confirmed, so the remaining budget `remaining` (≥ 1) is not exceeded.
 * With `remaining = 1`, `k' = 0` = no draft is taken and one verify row (= decode) confirms the last
 * token. No capacity term is needed (`assertBudget` already guaranteed
 * `pastLength + prompt + maxNewTokens − 1 ≤ capacity` before the run).
 */
export const planDraftLength = (k: number, remaining: number): number => Math.min(k, remaining - 1);

/**
 * The verify run's `last_row` (the indices of the rows to read — its length binds that run's R).
 *
 * `k' = 0` is `[0]` (the decode shape itself — identical down to the PreparedPlan key). `k' ≥ 1`
 * **pads** `[0..k']` **to `k+1` entries with the last index** — R is part of the PreparedPlan key,
 * so this avoids a separate shape (M=4 with R=3 / R=2) for every cycle whose `k'` shrinks at the end
 * of the budget. The logits of pad rows are not read.
 */
export const verifyRowIndices = (drafted: number, k: number): number[] => {
  if (drafted === 0) return [0];
  const rows = Array.from({ length: drafted + 1 }, (_unused, index) => index);
  while (rows.length < k + 1) rows.push(drafted);
  return rows;
};

/**
 * Takes the first `drafted` entries of the drafter's output and checks the vocabulary range.
 *
 * Drafts enter verify's `input_ids` and `history` (the same path as the prompt), so they go through
 * the same check as the prompt — an out-of-vocabulary id is an out-of-range embedding gather = NaN
 * contamination of the whole row, and the failure surfaces far from the real cause.
 */
export const takeDrafts = (
  raw: Int32Array<ArrayBuffer>,
  drafted: number,
  vocabSize: number,
): number[] => {
  if (raw.length < drafted) {
    throw new Error(`Only ${raw.length} drafts (${drafted} needed)`);
  }
  const drafts: number[] = [];
  for (let index = 0; index < drafted; index += 1) {
    const id = raw[index];
    if (!Number.isSafeInteger(id) || id < 0 || id >= vocabSize) {
      throw new Error(`draft[${index}] ${id} is outside the vocabulary 0..${vocabSize - 1}`);
    }
    drafts.push(id);
  }
  return drafts;
};

/**
 * The result of acceptance (`confirmed` = exactly the confirmed sequence this cycle delivers).
 *
 * Its length is `accepted + 1` (`[d₁..d_a, b']`). However, **when the last accepted draft is a stop
 * token** the enumeration stops there, so there is no `b'` and the length is `accepted`
 * (`[d₁..d_a]`). The stop token itself stays in the sequence (the same as the non-speculative stop
 * rule — it stays in the conversation as `pendingToken` and is prepended to the next turn's
 * prefill).
 */
export type DraftAcceptance = {
  readonly accepted: number;
  readonly confirmed: readonly number[];
};

/**
 * Accepts by matching prefix (**an enumeration with early cut-off**).
 *
 * `row(j)` is the logits of row `j` of the verify output (`drafts.length + 1` rows,
 * `j = 0..drafts.length`, are needed — on full acceptance row `drafts.length` produces `b'`).
 * `history` is this turn's confirmed sequence (including the frontier `b`), and row `j`'s draw uses
 * `history + [d₁..d_j]`. Drafts of rejected rows are not pushed onto `history` (the caller's
 * `history` is not touched — the caller extends the confirmed sequence as it delivers).
 *
 * MUST: when a confirmed token satisfies `isStop`, **stop there** (the stop section of the module
 * doc). When an accepted draft was a stop token, the next row's `sampler.next` is not called = no
 * logits beyond the point where non-speculative decoding stops are touched. When `b'` on rejection
 * is a stop token, the sequence stays as is (it returns there anyway).
 */
export const acceptDrafts = (
  row: (index: number) => Float32Array<ArrayBuffer>,
  drafts: readonly number[],
  sampler: Pick<Sampler, "next">,
  history: readonly number[],
  isStop: (token: number) => boolean,
): DraftAcceptance => {
  // A plain step does not extend the history: it passes the same history as ordinary decode to the
  // draw.
  if (drafts.length === 0) return { accepted: 0, confirmed: [sampler.next(row(0), history)] };
  const extended = [...history];
  let accepted = 0;
  for (;;) {
    const token = sampler.next(row(accepted), extended);
    if (accepted < drafts.length && token === drafts[accepted]) {
      extended.push(token);
      accepted += 1;
      if (isStop(token)) return { accepted, confirmed: drafts.slice(0, accepted) };
      continue;
    }
    return { accepted, confirmed: [...drafts.slice(0, accepted), token] };
  }
};

/**
 * Speculation counters (the `used: true` side of `GenerationStop.speculation`) — the speculative
 * loop of `sequence.ts` adds to the fields once per cycle.
 */
export type SpeculationTally = {
  cycles: number;
  draftRuns: number;
  drafted: number;
  accepted: number;
  /** Total tokens the cycles confirmed (Σ `confirmed.length`). */
  delivered: number;
  /** Index = that cycle's accepted count `a` (length `k+1`). */
  readonly acceptedHistogram: number[];
};

export const createSpeculationTally = (k: number): SpeculationTally => ({
  cycles: 0,
  draftRuns: 0,
  drafted: 0,
  accepted: 0,
  delivered: 0,
  acceptedHistogram: Array.from({ length: k + 1 }, () => 0),
});
