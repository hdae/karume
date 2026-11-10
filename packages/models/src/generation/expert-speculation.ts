/**
 * One cycle of self-speculation over the exact expert cache (ADR 0146 decision 8, scheme B) — the
 * GPU-free protocol the cache's face runs as one operation of its serialization chain
 * (`expert-cache-session.ts`), for the decode loop of `sequence.ts`.
 *
 * ## The cycle
 *
 * From the frontier b (confirmed, not in the KV) at position P, with `k'` drafts at most:
 *
 * 1. **Draft** on the cache as it is: for j = 1..k', one row (b, then d_{j−1}) at P + j − 1, a
 *    deferred run chained on the ones before it — a pick whose expert is not resident contributes 0
 *    and nothing is loaded. d_j is {@link Sampler.peek} of its logits, `ahead` the peeks of this
 *    cycle that would have consumed a draw: the draft peeks the very uniform value the exact row j − 1
 *    will draw with, so nearly the same logits pick nearly always the same token. A stop draft ends
 *    the drafting.
 * 2. `commit(0)` drops the whole draft chain (its rows are rewritten by the verify), and one load
 *    of every draft row's routing makes their picks resident at once.
 * 3. **Verify** the k' + 1 rows [b, d₁..d_k'] at P in one deferred run. Its rows before
 *    `firstMissingRow` (m) are exact: attention is causal, and no row up to them missed. While m is 0,
 *    `commit(0)`, load the verify's routing and run it again — row 0 alone is a scheme-A run, so its
 *    first missed layer moves strictly forward and it converges in at most `layers + 1` attempts (the
 *    load pins every row's picks, which fit the slots: `(depth + 1) · k ≤ slots` is the caller's).
 * 4. **Accept** on the exact rows only: `acceptDrafts` over the drafts before row m − 1 draws t_j =
 *    {@link Sampler.next} of row j with the confirmed history — the only draws of the cycle, in the
 *    order, on the logits and with the history plain decode has — and stops at a rejection or a stop
 *    token. When every exact row accepted (t_{m−1} = d_m below the verify's last row), the cycle
 *    ends at row m: its confirmed tokens are d₁..d_m, and a second load of the verify's routing
 *    readies row m's picks for the next cycle, whose row 0 is row m's input at row m's position.
 * 5. `touch` the confirmed rows (all exact, so resident) and return with the verify **pending**: the
 *    decode loop commits the rows of the tokens it delivers, so a consumer's `break` leaves the
 *    conversation at "the tokens received + one frontier" (`sequence.ts`). The recency then counts
 *    every confirmed row even when fewer are delivered; it orders evictions only.
 *
 * The output is plain decode's at every temperature: the drafts only propose, every confirmed token
 * is a draw from an exact row, and the RNG advances once per confirmed token.
 *
 * MUST: every module is side-effect free (CLAUDE.md).
 * DECIDED: docs/decisions/0146-exact-expert-cache.md
 */

import type { ExpertLoad, ExpertRouting, ExpertRun, RunInputs } from "@karume/runtime";
import { acceptDrafts } from "./speculation.ts";
import type {
  GenerationContextFace,
  GenerationSelfSpeculationCycle,
  GenerationSelfSpeculationRequest,
} from "./sequence.ts";

/** One run as the cycle issues it: deferred, chained for a draft. */
type CycleRun<C> = {
  readonly context: C;
  readonly queryLength: number;
  readonly commit: "deferred";
  readonly chain?: true;
};

/** The cache a cycle drives (the runtime's `ExpertCache`, narrowed so that a fake can stand in). */
export type ExpertCycleCache<C extends GenerationContextFace> = {
  readonly layers: number;
  run(
    inputs: RunInputs,
    bindings: undefined,
    generation: CycleRun<C>,
  ): Promise<Pick<ExpertRun, "outputs" | "routing" | "firstMissingRow">>;
  load(routing: ExpertRouting): Promise<ExpertLoad>;
  touch(routing: ExpertRouting, rows: number): void;
};

/** The counters of the cycles (the face adds them to its report). */
export type ExpertCycleTally = {
  /** Cycles whose verify reached an exact row (each confirms at least one token). */
  cycles: number;
  /** Draft runs that completed. */
  draftRuns: number;
  /** Drafts confirmed (Σ {@link GenerationSelfSpeculationCycle.accepted}). */
  acceptedDrafts: number;
  /** Verify runs issued again because their first row missed (m = 0). */
  verifyRetries: number;
  /** Cycles that ended at the verify's first missing row (m ≥ 1, every exact row accepted). */
  partialCommits: number;
};

export const createExpertCycleTally = (): ExpertCycleTally => ({
  cycles: 0,
  draftRuns: 0,
  acceptedDrafts: 0,
  verifyRetries: 0,
  partialCommits: 0,
});

const WHERE = "expert cache (self-speculation)";

/**
 * The draft rows' routings as one routing of their rows in order, so that one load pins every one
 * of their picks.
 */
const mergeRoutings = (routings: readonly ExpertRouting[]): ExpertRouting => {
  const [first] = routings;
  if (first === undefined) throw new Error(`${WHERE}: no routing to merge`);
  const { layers, picks } = first;
  if (routings.some((routing) => routing.layers !== layers || routing.picks !== picks)) {
    throw new Error(`${WHERE}: the draft routings disagree on their layers or picks`);
  }
  const rows = routings.reduce((total, routing) => total + routing.rows, 0);
  const ids = new Int32Array(layers * rows * picks);
  for (let layer = 0; layer < layers; layer += 1) {
    let at = layer * rows * picks;
    for (const routing of routings) {
      const span = routing.rows * picks;
      ids.set(routing.ids.subarray(layer * span, (layer + 1) * span), at);
      at += span;
    }
  }
  return { layers, rows, picks, ids };
};

/** A run's first missing row, checked against its real rows (a value outside them is a broken cache). */
const firstMissingOf = (run: Pick<ExpertRun, "firstMissingRow">, rows: number): number => {
  const first = run.firstMissingRow;
  if (!Number.isSafeInteger(first) || first < 0 || first > rows) {
    throw new Error(
      `${WHERE}: firstMissingRow ${first} is not in [0, ${rows}] (broken bookkeeping)`,
    );
  }
  return first;
};

/** The number of leading tokens of `confirmed` equal to the drafts at their place. */
const leadingDrafts = (confirmed: readonly number[], drafts: readonly number[]): number => {
  let count = 0;
  while (count < confirmed.length && count < drafts.length && confirmed[count] === drafts[count]) {
    count += 1;
  }
  return count;
};

/**
 * Runs one cycle (module doc) and counts it into `tally`. The verify runs at most `layers + 1`
 * times; one more would be a broken invariant and fails loudly, with nothing pending.
 */
export const runExpertCycle = async <C extends GenerationContextFace>(
  cache: ExpertCycleCache<C>,
  request: GenerationSelfSpeculationRequest<C>,
  tally: ExpertCycleTally,
): Promise<GenerationSelfSpeculationCycle> => {
  const { context, token, history, sampler, isStop } = request;
  // A cycle starts from committed rows only: a draft chained onto a pending run would sit after rows
  // the cycle does not own.
  if (context.pendingCommit !== undefined) {
    throw new Error(`${WHERE}: a cycle started with rows pending commit (broken bookkeeping)`);
  }
  const position = context.pastLength;

  // 1. Drafts on the cache as it is.
  const drafts: number[] = [];
  const routings: ExpertRouting[] = [];
  const extended = [...history];
  let ahead = 0;
  for (let step = 1; step <= request.drafts; step += 1) {
    const prepared = await request.prepare(
      [step === 1 ? token : drafts[step - 2]],
      position + step - 1,
    );
    const run = await cache.run(prepared.inputs, undefined, {
      context,
      queryLength: 1,
      commit: "deferred",
      chain: true,
    });
    tally.draftRuns += 1;
    routings.push(run.routing);
    const peeked = sampler.peek(prepared.logits(run.outputs, `draft@${step}`)(0), extended, ahead);
    ahead += peeked.draws;
    drafts.push(peeked.token);
    extended.push(peeked.token);
    if (isStop(peeked.token)) break;
  }

  // 2. Drop the drafts' rows, then make every draft row's picks resident in one load.
  if (drafts.length > 0) {
    context.commit(0);
    await cache.load(mergeRoutings(routings));
  }

  // 3. The verify, again while its first row misses.
  const ids = [token, ...drafts];
  const prepared = await request.prepare(ids, position);
  const verify: CycleRun<C> = { context, queryLength: ids.length, commit: "deferred" };
  const maxAttempts = cache.layers + 1;
  let run = await cache.run(prepared.inputs, undefined, verify);
  for (let attempt = 1; firstMissingOf(run, ids.length) === 0; attempt += 1) {
    // MUST: the missed rows are dropped before anything else, so a failure below leaves nothing
    // pending (the scheme-A rule).
    context.commit(0);
    if (attempt === maxAttempts) {
      throw new Error(
        `${WHERE}: the verify's first row still missed after ${attempt} attempts, more than one per routed layer (${cache.layers}) — broken invariant`,
      );
    }
    await cache.load(run.routing);
    tally.verifyRetries += 1;
    run = await cache.run(prepared.inputs, undefined, verify);
  }

  // 4. Acceptance on the exact rows alone: the drafts before the last exact row, so the walk draws
  // from rows 0..exact − 1 and never from a row that may have missed.
  const exact = firstMissingOf(run, ids.length);
  const row = prepared.logits(run.outputs, "verify");
  const { confirmed } = acceptDrafts(row, drafts.slice(0, exact - 1), sampler, history, isStop);
  const accepted = leadingDrafts(confirmed, drafts);
  // Every exact row accepted, and the verify has a row after them that missed: the cycle ends there.
  const partial =
    exact < ids.length && accepted === exact && !isStop(confirmed[confirmed.length - 1]);

  // 5. The confirmed rows are exact, so resident: their picks become the most recent.
  cache.touch(run.routing, confirmed.length);
  tally.cycles += 1;
  tally.acceptedDrafts += accepted;
  if (partial) {
    tally.partialCommits += 1;
    await cache.load(run.routing);
  }
  return { confirmed, drafted: drafts.length, accepted };
};
