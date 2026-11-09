/**
 * The generation face of a Session built with `SessionOptions.expertCache` — the whole-token re-run
 * of ADR 0146 (scheme A, decision 4).
 *
 * Under the cache a pick whose expert is not resident contributes 0, so a run that missed is not
 * the model's output. The face runs every run through `session.expertCache.run` deferred, and a run
 * that missed is never handed on: its rows are dropped with `commit(0)`, the experts of its routing
 * are loaded, and the same inputs run again, until a run misses nothing. Only then are that run's
 * rows committed, its picks marked recent (`touch`) and its outputs returned. To the generation loop
 * this is an ordinary immediate run ({@link GenerationSession}), so the loop needs no retry logic
 * and the sampler never sees a missed run's logits (drawing from them would consume the RNG and
 * diverge from full residency).
 *
 * ## Convergence
 *
 * A layer's routing depends only on the layers before it. In a run that missed, let l be the first
 * layer with a miss: every layer before l ran on resident experts, so l's routing is exact, and the
 * load makes l's picks resident while it pins the picks of every layer of the same routing (they
 * are not victims). The next attempt is therefore exact up to and including l, so the first missed
 * layer moves strictly forward and a run converges in at most `layers + 1` attempts — provided one
 * run's distinct picks of a layer fit the slots (the caller's chunk rows; `load` fails loudly
 * otherwise). An attempt beyond that bound is a broken invariant and fails loudly instead of
 * looping.
 *
 * ## Why the protocol is serialized here
 *
 * Two conversations can share one Session. Between this run's clean attempt and its `touch`, or
 * between its load and its re-run, another conversation's load could evict the experts this run
 * needs: `touch` would then fail ("not resident"), or the two could evict each other's experts
 * attempt after attempt. One protocol at a time on the face (run → commit(0) → load → run again →
 * commit → touch) closes both; the Session's own lock serializes only single calls.
 *
 * MUST: every module is side-effect free (CLAUDE.md).
 * DECIDED: docs/decisions/0146-exact-expert-cache.md
 */

import type {
  ExpertCacheStats,
  ExpertLoad,
  ExpertRouting,
  ExpertRun,
  GenerationContextSpec,
  RunInputs,
  RunOutputs,
  SymbolBindings,
} from "@karume/runtime";
import { createOperationChain } from "../concurrency/serial.ts";
import type { GenerationContextFace, GenerationSession } from "./sequence.ts";

/** The face's counters merged with the runtime's (what `Pipeline.expertCacheReport()` returns). */
export type ExpertCacheReport = ExpertCacheStats & {
  /** Slots per routed layer (K). */
  readonly slots: number;
  /** Generation runs whose rows were committed (each of them missed no expert). */
  readonly committedRuns: number;
  /** Runs issued again after a miss (every attempt beyond a run's first). */
  readonly retries: number;
};

/** One run as the face issues it: always deferred, never chained. */
type IssuedRun<C> = {
  readonly context: C;
  readonly queryLength: number;
  readonly commit: "deferred";
};

/**
 * The expert cache the face drives — the runtime's `ExpertCache`, narrowed so that a GPU-free fake
 * can stand in (the context is a type parameter for the reason of {@link GenerationSession}).
 */
export type ExpertCacheRuns<C extends GenerationContextFace> = {
  readonly slots: number;
  readonly layers: number;
  run(
    inputs: RunInputs,
    bindings: SymbolBindings | undefined,
    generation: IssuedRun<C> | undefined,
  ): Promise<ExpertRun>;
  load(routing: ExpertRouting): Promise<ExpertLoad>;
  touch(routing: ExpertRouting, rows: number): void;
  stats(): ExpertCacheStats;
};

/**
 * What the face needs of a Session. MUST: the real `Session` satisfies it with `GenerationContext`
 * (pinned by a type check on the test side).
 */
export type ExpertCacheTarget<C extends GenerationContextFace> = {
  createGenerationContext(spec: GenerationContextSpec): Promise<C>;
  readonly expertCache: ExpertCacheRuns<C> | undefined;
};

/** The generation argument of {@link GenerationSession.run}. */
type GenerationArgument<C extends GenerationContextFace> = Parameters<
  GenerationSession<C>["run"]
>[2];

/**
 * The re-run face: a {@link GenerationSession} (it is assignable to one) whose `run` also takes no
 * generation, plus its counters.
 */
export type ExpertCacheSession<C extends GenerationContextFace> = {
  createGenerationContext(spec: GenerationContextSpec): Promise<C>;
  run(
    inputs: RunInputs,
    bindings: SymbolBindings | undefined,
    generation?: GenerationArgument<C>,
  ): Promise<RunOutputs>;
  report(): ExpertCacheReport;
};

const WHERE = "expert cache";

/**
 * Builds the re-run face over a Session with an expert cache (module doc). A run with a generation
 * commits its rows itself, so a caller's `commit`, `chain` or `bidirectionalRows` is refused (the
 * last needs an immediate run, and a run under the cache is deferred until it misses nothing). A
 * run without a generation retries the same way and commits nothing.
 */
export const createExpertCacheSession = <C extends GenerationContextFace>(
  target: ExpertCacheTarget<C>,
): ExpertCacheSession<C> => {
  const cache = target.expertCache;
  if (cache === undefined) {
    throw new Error(
      `${WHERE}: the Session was built without SessionOptions.expertCache (there is no cache to run through)`,
    );
  }
  const maxAttempts = cache.layers + 1;
  const chain = createOperationChain();
  let committedRuns = 0;
  let retries = 0;

  /** The protocol of one run (module doc), on the face's chain. */
  const runClean = async (
    inputs: RunInputs,
    bindings: SymbolBindings | undefined,
    issued: IssuedRun<C> | undefined,
  ): Promise<RunOutputs> => {
    for (let attempt = 1; ; attempt += 1) {
      const run = await cache.run(inputs, bindings, issued);
      if (run.missing === 0) {
        if (issued !== undefined) {
          issued.context.commit(issued.queryLength);
          // The recency follows committed rows only (ADR 0146 decision 6).
          cache.touch(run.routing, issued.queryLength);
          committedRuns += 1;
        }
        return run.outputs;
      }
      // MUST: the missed run's rows are dropped before anything else, so a failure below leaves no
      // pending commit behind (the next run on the context would be refused).
      issued?.context.commit(0);
      if (attempt === maxAttempts) {
        throw new Error(
          `${WHERE}: a run still missed ${run.missing} experts after ${attempt} attempts, more than one per routed layer (${cache.layers}) — broken invariant`,
        );
      }
      await cache.load(run.routing);
      retries += 1;
    }
  };

  /** Reads the caller's generation once and refuses what the face owns. */
  const issue = (generation: GenerationArgument<C> | undefined): IssuedRun<C> | undefined => {
    if (generation === undefined) return undefined;
    // `chain` is not on the face's type; a caller that passes it anyway is refused all the same.
    const fields: Readonly<Record<string, unknown>> = generation;
    const { context, queryLength, commit, bidirectionalRows } = generation;
    const chained = fields["chain"];
    if (commit !== undefined) {
      throw new Error(
        `${WHERE}: a run cannot pass commit ${JSON.stringify(commit)} — the face commits a run's rows itself, once the run missed no expert`,
      );
    }
    if (chained !== undefined) {
      throw new Error(
        `${WHERE}: a run cannot be chained — a run that missed is run again alone, after commit(0)`,
      );
    }
    if (bidirectionalRows !== undefined) {
      throw new Error(
        `${WHERE}: a run cannot take bidirectionalRows — they need an immediate run, and a run under the cache stays deferred until it misses nothing`,
      );
    }
    return { context, queryLength, commit: "deferred" };
  };

  return {
    createGenerationContext: (spec) => target.createGenerationContext(spec),
    run: (inputs, bindings, generation) => {
      let issued: IssuedRun<C> | undefined;
      try {
        issued = issue(generation);
      } catch (cause) {
        return Promise.reject(cause);
      }
      return chain(() => runClean(inputs, bindings, issued));
    },
    report: () => ({ ...cache.stats(), slots: cache.slots, committedRuns, retries }),
  };
};
