/**
 * Host-side sampling (temperature / top-k / top-p / repetition penalty / logit bias / seed) and the
 * stop check against an EOS **set**. It is **pipeline-independent shared code**, so it lives in
 * `src/generation/` next to `greedy.ts` (ADR 0083 decision 7 — that is the placement ruling itself).
 *
 * The entry of general sampling is the raw last-row logits `[1,1,V]` (ADR 0083 decision 6).
 * Because it handles the whole vocabulary, top-k has no implementation limit, and repetition
 * penalty and full-vocabulary nucleus can be expressed. Only temperature 0 without processing also
 * returns an internal capability that checks the maximum and its smallest index selected on the GPU.
 * The stored graph's exit and the contract of general sampling do not change (ADR 0083 addendum on
 * temperature-0 decode).
 *
 * ## MUST: token-sequence parity with HF is not attainable
 *
 * The RNG is splitmix64 (`random.ts`), not torch's Philox — **the same seed does not give the same
 * sequence** (the MUST of ADR 0083 decision 7). Reference comparisons are taken at temperature 0
 * (= the greedy degenerate form); the paths where the draw runs are bound by our own fixtures and
 * distribution checks.
 *
 * ## The default of this layer is greedy
 *
 * The default is `temperature = 0` = the argmax degenerate form, and injecting the recommended
 * values (gemma-4-E2B-it = temperature 1.0 / top_k 64 / top_p 0.95) is **the distribution's
 * responsibility** (ADR 0083 decision 7 — the precedent is anima's `scheduler.type`). Keeping
 * "no sampler declared = deterministic" keeps the parity checks against the existing golden alive on
 * this path.
 *
 * ## Processing order (the same as HF's `LogitsProcessorList`)
 *
 * (1) repetition penalty → (2) logit bias → (2') suppressed tokens → (3) divide by temperature →
 * (4) top-k → (5) top-p → (6) multinomial draw. (1), (2) and (2') come **before** temperature (HF
 * splits processors and warpers into stages, and penalty / bias / `SuppressTokensLogitsProcessor`
 * are in the earlier one). Swapping the order gives a different distribution for the same knobs,
 * so this order is the contract.
 *
 * ## Suppressed tokens are the distribution's, not the request's
 *
 * (2') sets the logits of the ids the distribution declares as never generated to `-Infinity`
 * (upstream `generation_config.json` `suppress_tokens`, applied by HF `generate()` in greedy and in
 * sampling alike — gemma-4-12B-it declares `<image|>` / `<audio|>`). It is a second argument of
 * {@link createSampler}, not a {@link SamplerSpec} field: a request cannot turn it off, and the
 * set comes from the verified wiring (`GenerationWiring.suppressTokens`). An empty set (every
 * distribution that declares none) changes nothing — not a copy, not a scan.
 */

import { ModelInputError } from "../errors.ts";
import { Randu } from "./random.ts";

/** The sampling specification (omitted = greedy = the temperature-0 degenerate form). */
export type SamplerSpec = {
  /**
   * The temperature that divides the logits before softmax (0 or more, default 0).
   *
   * `0` is **greedy** (argmax): it consumes no RNG and top-k / top-p have no effect (trimming
   * candidates keeps the maximum, so the result does not change). Repetition penalty and logit bias
   * still apply at temperature 0.
   */
  readonly temperature?: number;
  /**
   * Narrows the candidates to the `topK` most probable entries (an integer of 1 or more, **no upper
   * limit**). A value above the vocabulary size is clamped to the vocabulary size (the same as HF's
   * `min(top_k, vocab)`).
   */
  readonly topK?: number;
  /**
   * Narrows to the **smallest candidate set** whose cumulative probability reaches `topP`
   * (0 < topP ≤ 1). `1` does not narrow. The top entry always remains (HF's
   * `min_tokens_to_keep = 1`).
   */
  readonly topP?: number;
  /**
   * Factor that weakens the logits of tokens already seen (a positive finite number; `1`
   * disables it). The direction is the same as HF's `RepetitionPenaltyLogitsProcessor`: **negative
   * logits are multiplied, non-negative logits are divided** (both lower the value when the factor
   * is > 1). However many times a token appears in the history, **it is applied once**.
   */
  readonly repetitionPenalty?: number;
  /**
   * Logit addition per token id (a list of `[token, bias]`; the value is a finite number, or
   * `-Infinity` = that token is banned). A token outside the vocabulary fails loudly (silently
   * ignoring it would quietly leave "a bias that does nothing").
   *
   * MUST: accept an **array of tuples**, not a `Map`. This specification crosses paths that go
   * through JSON (the distribution's declaration, host-side settings storage, logs) and worker
   * boundaries, and `JSON.stringify` flattens a `Map` to `{}`, so a round-tripped spec **silently
   * becomes an empty bias** (no exception; it shows up only as "a knob that has no effect"). An
   * array stays the same specification after a round trip.
   *
   * MUST: fail when the same token is written twice (do not fold it as last-write-wins). `Map`
   * folded it, but folding on this additive surface means "only one of the two written biases
   * applies" — the caller cannot tell whether they add or overwrite.
   */
  readonly logitBias?: readonly (readonly [token: number, bias: number])[];
  /**
   * Seed of the draw sequence (a non-negative safe integer, default 0 — the same seed gives the
   * same token sequence).
   */
  readonly seed?: number;
};

/**
 * The draw target (exclusions applied, probabilities normalized within the candidate set;
 * `tokens[i]` is chosen with `probabilities[i]`). At temperature 0 its length is 1 (argmax's
 * one-point distribution).
 *
 * The order is **descending probability when `topK` is given or `topP < 1`** (needed because top-p
 * is defined as "the cumulative sum from the top"), and ascending token id otherwise (including
 * `topP = 1` alone) — so a specification without narrowing does not sort the whole vocabulary
 * (V = 262,144) every step.
 *
 * NOTE: **the order affects which token a fixed seed picks**. The draw looks up the cumulative
 * distribution with one uniform random number (inverse CDF), so even with the same distribution a
 * different order maps the same random number to a different token (logits `[0, 1, 2]`,
 * temperature 1, seed 1: without `topK` 2, 2, 2, …; with `topK: 3` 2, 1, 0, …). As a distribution
 * it is the same, and the same specification with the same seed gives the same sequence (the
 * determinism contract is per specification).
 */
export type SamplerDistribution = {
  readonly tokens: Int32Array<ArrayBuffer>;
  readonly probabilities: Float64Array<ArrayBuffer>;
};

/** The sampler bound to one generation (it holds RNG state, so it is reused across steps). */
export type Sampler = {
  /**
   * Only for temperature 0 without processing: checks and returns the first NaN, or the maximum
   * with its smallest index, selected by the GPU.
   *
   * MUST: absent while tokens are suppressed, unless the selection is built to mask them
   * (`selectionMasksSuppressed` of {@link createSampler}): a selection over the **raw** logits
   * would return a suppressed id that is their maximum. A selection that claims the mask and still
   * returns a suppressed id fails here.
   */
  readonly greedy?: (value: number, index: number) => number;
  /**
   * Picks the next token from the last-row logits.
   *
   * `history` is **the token sequence so far** that the repetition penalty looks at (including the
   * prompt — the same as HF applying it to the whole `input_ids`). It is not read when the
   * specification does not use the penalty.
   */
  next(logits: Float32Array<ArrayBuffer>, history: readonly number[]): number;
};

const assertFiniteAtLeast = (value: number, minimum: number, where: string): void => {
  if (!Number.isFinite(value) || value < minimum) {
    throw new ModelInputError(`${where} ${value} is not a finite number ≥ ${minimum}`);
  }
};

/**
 * The allowed fields of {@link SamplerSpec}. It is a `Record` so that adding a field to the type
 * without adding it here fails the type check.
 */
const SAMPLER_SPEC_KEYS: Readonly<Record<keyof SamplerSpec, true>> = {
  temperature: true,
  topK: true,
  topP: true,
  repetitionPenalty: true,
  logitBias: true,
  seed: true,
};

/**
 * The acceptance set of the specification (the only place that fails loudly).
 *
 * MUST: do not clamp silently. Folding `temperature: -1` or `topP: 0` into the default quietly
 * leaves "a knob that has no effect", noticeable only from differences in the output.
 *
 * MUST: reject unknown keys too. A `minP` or a misspelled `top_p` arriving from JS or from restored
 * saved settings escapes TS's excess-property check, and ignoring it makes just that knob silently
 * ineffective (the same treatment as Gemma's settings in the same layer and the hub's manifest).
 */
const assertSpec = (spec: SamplerSpec): void => {
  for (const key of Object.keys(spec)) {
    if (!Object.hasOwn(SAMPLER_SPEC_KEYS, key)) {
      throw new ModelInputError(
        `Unknown key '${key}' in the sampler spec (allowed: ${Object.keys(SAMPLER_SPEC_KEYS).join(" / ")})`,
      );
    }
  }
  if (spec.temperature !== undefined) assertFiniteAtLeast(spec.temperature, 0, "temperature");
  if (spec.topK !== undefined && (!Number.isSafeInteger(spec.topK) || spec.topK < 1)) {
    throw new ModelInputError(`topK ${spec.topK} is not a positive integer`);
  }
  if (spec.topP !== undefined) {
    if (!Number.isFinite(spec.topP) || spec.topP <= 0 || spec.topP > 1) {
      throw new ModelInputError(`topP ${spec.topP} is not in the range 0 < topP ≤ 1`);
    }
  }
  if (spec.repetitionPenalty !== undefined) {
    if (!Number.isFinite(spec.repetitionPenalty) || spec.repetitionPenalty <= 0) {
      throw new ModelInputError(
        `repetitionPenalty ${spec.repetitionPenalty} is not a positive finite number`,
      );
    }
  }
  // MUST: check the shape and the types too (untyped JS / JSON input arrives here). Do not silently
  // accept a `Map` or an object through a different iteration shape, and do not let a string bias
  // concatenate through `+=` (`1 + "0"` = `"10"`).
  if (spec.logitBias !== undefined && !Array.isArray(spec.logitBias)) {
    throw new ModelInputError("logitBias is not an array of [token, bias] pairs");
  }
  const biased = new Set<number>();
  for (const [position, entry] of (spec.logitBias ?? []).entries()) {
    if (!Array.isArray(entry) || entry.length !== 2) {
      throw new ModelInputError(`logitBias[${position}] is not a two-element [token, bias] pair`);
    }
    const [token, bias] = entry;
    if (!Number.isSafeInteger(token) || token < 0) {
      throw new ModelInputError(`logitBias token ${token} is not a non-negative integer`);
    }
    if (biased.has(token)) {
      throw new ModelInputError(
        `Token ${token} appears twice in logitBias (not folded as last-wins)`,
      );
    }
    biased.add(token);
    // `-Infinity` is the idiom for "ban this token" (the same as HF's `SequenceBias`). `+Infinity`
    // and `NaN` only turn softmax into NaN, so they are rejected. Non-numbers (string, null,
    // boolean) are implicitly converted by the addition into another value, so they are rejected.
    if (typeof bias !== "number" || !(Number.isFinite(bias) || bias === Number.NEGATIVE_INFINITY)) {
      throw new ModelInputError(
        `logitBias[${token}] ${bias} is neither a finite number nor -Infinity`,
      );
    }
  }
};

/**
 * The logits with (1) repetition penalty + (2) logit bias + (2') suppressed tokens + (3)
 * temperature applied.
 *
 * A specification that applies nothing and an empty suppressed set **return the input as is** (no
 * copy of V = 262,144 values every step). The return value is read-only and must not be modified.
 */
const processLogits = (
  logits: Float32Array<ArrayBuffer>,
  spec: SamplerSpec,
  history: readonly number[],
  suppressed: readonly number[],
): Float32Array<ArrayBuffer> => {
  const temperature = spec.temperature ?? 0;
  const penalty = spec.repetitionPenalty ?? 1;
  const bias = spec.logitBias;
  const scaled = temperature !== 0 && temperature !== 1;
  if (
    penalty === 1 &&
    (bias === undefined || bias.length === 0) &&
    suppressed.length === 0 &&
    !scaled
  ) {
    return logits;
  }

  const processed = new Float32Array(logits.length);
  processed.set(logits);
  if (penalty !== 1) {
    // MUST: once however many times the token appears in the history (HF writes back the
    // **original values** it `gather`ed, so duplicates are idempotent). Walking the history naively
    // raises the factor to the power of the duplicates, which is a different knob.
    for (const token of new Set(history)) {
      // NOTE: `history` is not a caller request (`createSampler` is not on the public surface, and
      // its only caller `GenerationSequence.generate` passes the checked prompt + tokens it chose
      // itself). A failure here is broken wiring (logits width disagreeing with
      // `program.vocabSize`), so unlike `logitBias` below (a value the caller passes) it is not a
      // `ModelInputError` — ADR 0107 decision 2.
      if (!Number.isSafeInteger(token) || token < 0 || token >= processed.length) {
        throw new RangeError(
          `History token id ${token} is outside the vocabulary 0..${processed.length - 1}`,
        );
      }
      const value = processed[token];
      // HF's `RepetitionPenaltyLogitsProcessor` direction: multiply if negative, divide if not.
      processed[token] = value < 0 ? value * penalty : value / penalty;
    }
  }
  if (bias !== undefined) {
    for (const [token, amount] of bias) {
      if (token >= processed.length) {
        // The vocabulary limit is unknown when the specification is checked (`assertSpec`), so the
        // check is deferred to here. The value itself is the caller's `logitBias`, so it is
        // input-caused (ADR 0107 decision 3).
        throw new ModelInputError(
          `logitBias token ${token} is outside the vocabulary 0..${processed.length - 1}`,
        );
      }
      processed[token] += amount;
    }
  }
  // (2') after the bias, so a bias cannot bring a suppressed id back (HF runs
  // `SuppressTokensLogitsProcessor` after `SequenceBiasLogitsProcessor`; −∞ stays −∞ under the
  // temperature division that follows).
  for (const token of suppressed) {
    // The set is the verified wiring (`createGenerationProgram` checked it against `vocabSize`,
    // and the logits width against the same value), so a miss here is broken wiring, not a request
    // error (ADR 0107 decision 2).
    if (token >= processed.length) {
      throw new RangeError(
        `Suppressed token id ${token} is outside the vocabulary 0..${processed.length - 1}`,
      );
    }
    processed[token] = Number.NEGATIVE_INFINITY;
  }
  if (scaled) {
    for (let token = 0; token < processed.length; token += 1) processed[token] /= temperature;
  }
  return processed;
};

/**
 * Rejects NaN at the entry (fail loudly).
 *
 * MUST: reject it even at the cost of a scan. With NaN **every comparison is false**, so three
 * places downstream break silently — argmax passes over it and returns another token, the top-k
 * threshold rejects every later candidate once it becomes NaN, and the softmax denominator becomes
 * NaN and wipes out every probability. An out-of-range index gather contaminates a whole row with
 * NaN ([known-issues] / [limitations]), so this is a shape that really occurs.
 *
 * Temperature 0 folds this check into the search for the maximum, and top-k into the candidate
 * scan. Everything else checks before ranking.
 */
const assertNoNaN = (logits: Float32Array<ArrayBuffer>): void => {
  for (let token = 0; token < logits.length; token += 1) {
    // MUST: do not use `Number.isNaN` (it would be called V times) — NaN is the value not equal to
    // itself.
    if (logits[token] !== logits[token]) {
      throw new Error(
        `logits[${token}] is NaN (non-finite) — failing here instead of folding it into a token id`,
      );
    }
  }
};

/**
 * Whether the maximum outside `excluded` is finite (NaN never becomes the maximum: all its
 * comparisons are false).
 */
const hasFiniteMaximum = (
  logits: Float32Array<ArrayBuffer>,
  excluded: readonly number[],
): boolean => {
  const skip = new Set(excluded);
  let maximum = Number.NEGATIVE_INFINITY;
  for (let token = 0; token < logits.length; token += 1) {
    if (logits[token] > maximum && !skip.has(token)) maximum = logits[token];
  }
  return Number.isFinite(maximum);
};

/**
 * Throws a non-finite maximum classified **by its origin**.
 *
 * If the maximum of `raw` (the logits **before** processing) outside the distribution's
 * suppressed set is finite, what removed the candidates is certainly the caller's `logitBias` —
 * the only remedy is "remove the ban", so it is input-caused (ADR 0107 decision 2). If it was
 * non-finite before processing (the suppressed ids left out — the request cannot lift that mask),
 * the model output is broken, and no change to the request fixes it = it stays a plain `Error`.
 *
 * MUST: run the classification **only right before throwing** (scanning `raw` is proportional to
 * the vocabulary V, so it is not added to the ordinary per-step path).
 */
const nonFiniteMaximum = (
  message: string,
  raw: Float32Array<ArrayBuffer> | undefined,
  suppressed: readonly number[],
): Error =>
  raw !== undefined && hasFiniteMaximum(raw, suppressed)
    ? new ModelInputError(message)
    : new Error(message);

/**
 * The maximum check shared by CPU and GPU. It does not fold non-finite values into a token and
 * returns the same errors as before.
 *
 * Only a surface that can pass `raw` (a caller holding the unprocessed logits) splits a non-finite
 * maximum into input-caused and internal failure by the rule of {@link nonFiniteMaximum}.
 */
const checkedMaximum = (
  value: number,
  index: number,
  raw?: Float32Array<ArrayBuffer>,
  suppressed: readonly number[] = [],
): number => {
  if (value !== value) {
    throw new Error(
      `logits[${index}] is NaN (non-finite) — failing here instead of folding it into a token id`,
    );
  }
  if (!Number.isFinite(value)) {
    throw nonFiniteMaximum(
      `The maximum of the logits is non-finite (${value}) — failing here instead of folding it into a token id`,
      raw,
      suppressed,
    );
  }
  return index;
};

/**
 * Index of the maximum (on a tie **the one that came first** — the same tie-break as torch's
 * `argmax`).
 *
 * MUST: fail when the maximum is non-finite. A gather outside the position table or broken weights
 * make the logits non-finite, but argmax **folds that into a plausible token id**, and this is the
 * only place before it is folded.
 * NaN fails per position in the same scan, and a ±Infinity maximum is rejected at the end.
 * It does not scan the whole vocabulary again for the check, and keeps first-wins on ties and the
 * error for the first NaN.
 */
const argmax = (
  logits: Float32Array<ArrayBuffer>,
  raw: Float32Array<ArrayBuffer>,
  suppressed: readonly number[],
): number => {
  let best = 0;
  let bestValue = Number.NEGATIVE_INFINITY;
  for (let token = 0; token < logits.length; token += 1) {
    const value = logits[token];
    if (value !== value) {
      throw new Error(
        `logits[${token}] is NaN (non-finite) — failing here instead of folding it into a token id`,
      );
    }
    if (value > bestValue) {
      bestValue = value;
      best = token;
    }
  }
  return checkedMaximum(bestValue, best, raw, suppressed);
};

/**
 * Selects the top `k` token ids in **descending** order (on a tie the smaller id ranks higher — the
 * same tie-break as {@link argmax}).
 *
 * A full sort is avoided because this runs every step over V = 262,144. A **binary heap with the
 * worst element at the root** is kept at length k and only tokens better than the root replace it —
 * V scan steps plus an O(log k) sift-down only when a token is taken. Inserting into a descending
 * array of length ≤ k would make each push-out O(k) on input whose values keep improving
 * (ascending), growing to V·k.
 *
 * Ranking is settled by one comparator, `better`, shared by the heap ordering and the final sort
 * (written in two places, a drifted tie-break would still give "the top k by value", so a broken
 * order contract would pass silently).
 *
 * The steady-phase cut-off is the single comparison `logits[token] <= logits[heap[0]]` because the
 * scan runs in ascending id order and every id in the heap is smaller than the current token — on a
 * tie the current token loses the tie-break (ascending id), so `better(token, heap[0])` is
 * equivalent to `logits[token] > logits[heap[0]]`. Without this cut-off, input dense with ties (most
 * of the vocabulary at the same logit) would lose to the insertion form.
 *
 * MUST: in both the fill and the steady phase, NaN is rejected in token id order before it touches
 * the heap. A NaN at the root makes the cut-off comparison always false, and every later token
 * replaces the root (with a wrong result as well).
 * Clamping `k` to the vocabulary size is also the caller's ({@link samplerDistribution}) job and is
 * not repeated here.
 *
 * NOTE: HF's `TopKLogitsWarper` is a threshold form that drops values **below** the k-th value, so
 * it keeps every candidate equal to the k-th value and more than k remain on a tie. This one
 * narrows to **exactly k entries**, keeping the smaller id on a tie — token-sequence parity with HF
 * is not attainable anyway (the RNG differs), so it takes the side where `topK: 1` matches greedy
 * exactly.
 */
const selectTopKHeap = (logits: Float32Array<ArrayBuffer>, k: number): number[] => {
  // "left ranks above right" — descending value, ascending id on a tie.
  const better = (left: number, right: number): boolean =>
    logits[left] > logits[right] || (logits[left] === logits[right] && left < right);

  // A heap whose root is the **worst** (= the next candidate to be pushed out).
  const heap: number[] = [];
  const siftDown = (from: number): void => {
    let at = from;
    for (;;) {
      const left = at * 2 + 1;
      if (left >= heap.length) break;
      const right = left + 1;
      const worst = right < heap.length && better(heap[left], heap[right]) ? right : left;
      if (!better(heap[at], heap[worst])) break;
      const swap = heap[at];
      heap[at] = heap[worst];
      heap[worst] = swap;
      at = worst;
    }
  };

  // Check every entry in token id order and reject the first NaN regardless of candidate ranking.
  // Fill phase: push the first k entries as they are.
  let token = 0;
  for (; token < k; token += 1) {
    const value = logits[token];
    if (value !== value) {
      throw new Error(
        `logits[${token}] is NaN (non-finite) — failing here instead of folding it into a token id`,
      );
    }
    heap.push(token);
    let at = heap.length - 1;
    while (at > 0) {
      const parent = (at - 1) >> 1;
      if (!better(heap[parent], heap[at])) break;
      const swap = heap[parent];
      heap[parent] = heap[at];
      heap[at] = swap;
      at = parent;
    }
  }
  // Steady phase: only tokens better than the root replace it. The root value changes only on a
  // replacement.
  let cutoff = logits[heap[0]];
  for (; token < logits.length; token += 1) {
    const value = logits[token];
    if (value !== value) {
      throw new Error(
        `logits[${token}] is NaN (non-finite) — failing here instead of folding it into a token id`,
      );
    }
    if (value <= cutoff) continue;
    heap[0] = token;
    siftDown(0);
    cutoff = logits[heap[0]];
  }
  return heap.sort((left, right) => (better(left, right) ? -1 : 1));
};

/**
 * Stably sorts every token id by descending f32 logit (ascending id on a tie).
 *
 * Top-p alone needs the rank of the whole vocabulary. Avoiding the O(V log V) comparison sort, it
 * maps the IEEE 754 bit patterns to keys in descending numeric order and distributes them one byte
 * at a time in 4 passes. Keeping the original bits for negative numbers and inverting all bits but
 * the sign for non-negative numbers makes the unsigned ascending order of the keys the descending
 * order of the logits. NaN was already rejected at the entry.
 *
 * MUST: +0 and -0 rank equal, as in a comparison sort. It starts from ascending id order and
 * distributes each digit stably, so the tie-break and the softmax addition order do not change.
 * The top-p boundary and the probabilities stay bit-identical.
 * It keeps the input's byteOffset, and normalizing the keys does not modify the input logits.
 */
const sortAllDescending = (logits: Float32Array<ArrayBuffer>): Int32Array<ArrayBuffer> => {
  const length = logits.length;
  let order = new Int32Array(length);
  let scratch = new Int32Array(length);
  const keys = new Uint32Array(length);
  const bits = new Uint32Array(logits.buffer, logits.byteOffset, length);
  const counts = new Uint32Array(1024);
  for (let token = 0; token < length; token += 1) {
    const raw = logits[token] === 0 ? 0 : bits[token];
    const key = (raw & 0x80000000) !== 0 ? raw : raw ^ 0x7fffffff;
    keys[token] = key;
    order[token] = token;
    for (let pass = 0; pass < 4; pass += 1) counts[pass * 256 + ((key >>> (pass * 8)) & 255)]++;
  }
  for (let pass = 0; pass < 4; pass += 1) {
    const offset = pass * 256;
    let total = 0;
    for (let byte = 0; byte < 256; byte += 1) {
      const count = counts[offset + byte];
      counts[offset + byte] = total;
      total += count;
    }
    const shift = pass * 8;
    for (let index = 0; index < length; index += 1) {
      const token = order[index];
      scratch[counts[offset + ((keys[token] >>> shift) & 255)]++] = token;
    }
    const swap = order;
    order = scratch;
    scratch = swap;
  }
  return order;
};

/**
 * Processed logits → the draw target (normalized; the order contract is in
 * {@link SamplerDistribution}).
 *
 * Temperature 0 degenerates to argmax's one-point distribution (the surface that yields the same
 * token as the parity check of `greedy.ts`).
 *
 * `suppressTokens` is the distribution's suppressed set (step (2') of the module doc — the
 * verified wiring's value; omitted = none).
 *
 * NOTE: `topP` alone radix-sorts the whole vocabulary (O(V)). With top-k, the bounded heap still
 * picks the candidates and top-p is applied within them (the normalized set does not change).
 */
export const samplerDistribution = (
  logits: Float32Array<ArrayBuffer>,
  spec: SamplerSpec,
  history: readonly number[],
  suppressTokens: readonly number[] = [],
): SamplerDistribution => {
  if (logits.length < 1) throw new Error("logits is empty");
  assertSpec(spec);
  // Checked **after** processing (this catches not only the raw logits but also combinations such
  // as a `-Infinity` bias added to a `+Infinity` entry).
  const processed = processLogits(logits, spec, history, suppressTokens);
  const temperature = spec.temperature ?? 0;
  if (temperature === 0) {
    // The unprocessed `logits` (and the suppressed set, which the request cannot lift) are passed
    // to tell the **origin** of a non-finite maximum (model output / the caller's `logitBias`)
    // apart ({@link nonFiniteMaximum}).
    return {
      tokens: Int32Array.of(argmax(processed, logits, suppressTokens)),
      probabilities: Float64Array.of(1),
    };
  }

  // A topK above the vocabulary size is clamped to the vocabulary size (HF's `min(top_k, vocab)`).
  const topK = spec.topK === undefined ? undefined : Math.min(spec.topK, processed.length);
  const topP = spec.topP;
  let candidates: readonly number[] | Int32Array<ArrayBuffer>;
  if (topK !== undefined) candidates = selectTopKHeap(processed, topK);
  else {
    assertNoNaN(processed);
    if (topP !== undefined && topP < 1) candidates = sortAllDescending(processed);
    else candidates = Array.from({ length: processed.length }, (_unused, token) => token);
  }

  // Softmax normalizes within the candidate set only (the mass dropped by top-k leaves the
  // denominator — equivalent to HF setting the dropped entries to −inf and passing them to
  // `softmax`).
  let peak = Number.NEGATIVE_INFINITY;
  for (const token of candidates) {
    if (processed[token] > peak) peak = processed[token];
  }
  if (!Number.isFinite(peak)) {
    // Classify the origin by the same rule as the temperature-0 path (finite before processing =
    // the bias removed the candidates).
    throw nonFiniteMaximum(
      `The maximum candidate logit is non-finite (${peak}) — every token is banned`,
      logits,
      suppressTokens,
    );
  }
  const weights = new Float64Array(candidates.length);
  let total = 0;
  for (let index = 0; index < candidates.length; index += 1) {
    const weight = Math.exp(processed[candidates[index]] - peak);
    weights[index] = weight;
    total += weight;
  }

  // (5) top-p: keep entries **including the one at which** the cumulative sum reaches topP (the
  // descending form equivalent to the ascending cumsum of HF's `TopPLogitsWarper`). The top entry
  // always remains. `topP = 1` does not narrow, since rounding could drop the tail.
  let kept = candidates.length;
  if (topP !== undefined && topP < 1) {
    let cumulative = 0;
    for (let index = 0; index < candidates.length; index += 1) {
      cumulative += weights[index] / total;
      if (cumulative >= topP) {
        kept = index + 1;
        break;
      }
    }
  }

  const tokens = new Int32Array(kept);
  const probabilities = new Float64Array(kept);
  let keptTotal = 0;
  for (let index = 0; index < kept; index += 1) keptTotal += weights[index];
  for (let index = 0; index < kept; index += 1) {
    tokens[index] = candidates[index];
    probabilities[index] = weights[index] / keptTotal;
  }
  return { tokens, probabilities };
};

/**
 * **Snapshot at issue** of the specification (the defensive copy of the ADR 0083 addendum of
 * 2026-09-02).
 *
 * MUST: do not keep the object the caller holds. The sampler lives across steps and re-reads this
 * specification every step, so rewriting it during a run **changes the distribution mid-generation**
 * (no exception; only the determinism contract "same seed, same sequence" silently breaks).
 *
 * MUST: copy `logitBias` down to its elements — copying only the array shares the tuples inside,
 * and `bias[0][1] = -Infinity` would reach the running draw. **This is the only place the array is
 * copied** ({@link samplerDistribution} reads the spec only within one call, so it need not copy).
 *
 * NOTE: it is exported so the upper layers (`gemma.Pipeline.chat` / `gemma.ChatSession.send`) take
 * the same copy **at issue**. Those two layers promise "requests are copied at issue" in their type
 * docs, and if the copy were delayed to {@link createSampler} (= the first `next()`), rewrites in
 * between would silently take effect. The export keeps a single copy point and is not on the
 * `mod.ts` / subpath surface (ADR 0008).
 */
export const snapshotSpec = (spec: SamplerSpec): SamplerSpec => {
  // MUST: pass the acceptance set before copying (one entry for check and copy). The copy assumes
  // the element shape, so if it ran before the check a shape error would surface as a TypeError
  // rather than an input error.
  assertSpec(spec);
  return Object.freeze({
    ...spec,
    ...(spec.logitBias === undefined
      ? {}
      : {
          logitBias: Object.freeze(
            spec.logitBias.map(([token, bias]) => Object.freeze([token, bias] as const)),
          ),
        }),
  });
};

/**
 * Builds a sampler (it holds RNG state, so one generation **reuses the same object**).
 *
 * Omitting the specification gives greedy (temperature 0) — this expresses that this layer owns no
 * defaults (ADR 0083 decision 7).
 *
 * `suppressTokens` is the distribution's suppressed set (step (2') of the module doc), passed by
 * the generation surface from the verified wiring; omitted or empty = none, and the sampler is the
 * one it was without the argument. A non-empty set withdraws the GPU selection capability
 * ({@link Sampler.greedy}) unless `selectionMasksSuppressed` says that selection masks the set.
 */
export const createSampler = (
  spec: SamplerSpec = {},
  suppressTokens: readonly number[] = [],
  selectionMasksSuppressed = false,
): Sampler => {
  // Keep a **copy** of the checked specification ({@link snapshotSpec} checks, then copies — the
  // checked values and the values the draw reads stay the same).
  const frozen = snapshotSpec(spec);
  // The seed acceptance check also happens here (the draw runs in the middle of decode, so delaying
  // the generator's creation would make an invalid seed fail after a GB-scale load —
  // `assertAcceptableSeed` in `request-gates.ts` owns that acceptance set).
  const random = new Randu(frozen.seed ?? 0);
  // A copy for the same reason as the spec snapshot (the set is read every step).
  const suppressed = Object.freeze([...suppressTokens]);
  const plainGreedy =
    (suppressed.length === 0 || selectionMasksSuppressed) &&
    (frozen.temperature ?? 0) === 0 &&
    (frozen.repetitionPenalty ?? 1) === 1 &&
    (frozen.logitBias?.length ?? 0) === 0;
  const greedy = (value: number, index: number): number => {
    const token = checkedMaximum(value, index);
    if (suppressed.includes(token)) {
      throw new Error(`The GPU selection returned the suppressed token ${token}`);
    }
    return token;
  };
  return {
    ...(plainGreedy ? { greedy } : {}),
    next(logits: Float32Array<ArrayBuffer>, history: readonly number[]): number {
      const { tokens, probabilities } = samplerDistribution(logits, frozen, history, suppressed);
      // One candidate (temperature 0 / topK 1 / vocabulary 1) makes the draw trivial, so the RNG
      // is not advanced.
      if (tokens.length === 1) return tokens[0];
      const draw = random.uniform();
      let cumulative = 0;
      for (let index = 0; index < tokens.length; index += 1) {
        cumulative += probabilities[index];
        if (draw < cumulative) return tokens[index];
      }
      // Rounding can leave the cumulative sum slightly below 1 (`draw` is in [0,1), so only then
      // does control reach here).
      return tokens[tokens.length - 1];
    },
  };
};

/**
 * Check against the stop token set (ADR 0083 decision 8).
 *
 * The stop condition is a **set**, not a single EOS (gemma-4-E2B-it's `generation_config.json` has
 * `eos_token_id = [1, 106, 50]`). What is checked is **the sampling result**, at the same stage for
 * argmax and for the draw — checking on the logits side produces mismatches of the kind "a banned
 * EOS turns into a stop".
 *
 * MUST: the set comes from **the same distribution digest set** as the chat format (ADR 0084
 * decision 5). Taken from different places, one of them goes stale.
 */
export const isStopToken = (token: number, stopTokens: readonly number[]): boolean =>
  stopTokens.includes(token);
