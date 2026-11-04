/**
 * Post-processing of vowel detection — folds the graph's logits `[T20, 8]` into a `.lab` (start
 * seconds, end seconds, label).
 *
 * There are 4 steps, and the canonical spec, order included, is the Python side
 * `training/src/vowel_detector/postprocess.py` (upstream `@hdae/vowel-detector`, MIT):
 *
 * 1. log_softmax (logits → log posteriors)
 * 2. **Viterbi with a transition penalty** — a fixed cost on class switches flattens short
 *    fragments
 * 3. **short-segment merge** — repaint segments shorter than `MIN_DURATION_FRAMES` with whichever
 *    neighbouring class has the higher posterior there (`pau` is perceptually important silence and
 *    stays even when short)
 * 4. **cons absorption** — based on Japanese mora structure C+V, absorb a consonant segment into
 *    the **following** vowel/N (into the preceding segment if the following one is not a vowel/N)
 *
 * ## Not the same thing as the Viterbi in `src/text/unigram.ts` (not shared)
 *
 * That one is the shortest path through "a lattice covering a string with variable-length pieces"
 * (the set of edges changes per input); this one is a smoothing that puts a uniform transition
 * penalty on "a lattice of a fixed 8 classes × T frames". They share neither states, transitions
 * nor the scoring function, so merging the two that only share a name would add an abstraction
 * that just branches on its arguments.
 *
 * ## The constants are canonical as Python's module constants
 *
 * They are not put in the distribution's `pipelineConfig` (doing so would create a field that "can
 * declare a smoothing different from training", and there is no ground for moving that). The match
 * of the values is checked by `tests/vowel_detector_host_test.ts` against a copy of upstream's
 * `feature_config.json`.
 */

/** The 8 lip-sync classes. **The order is the id** (1:1 with the graph's output columns). */
export const LIPSYNC_CLASSES = ["a", "i", "u", "e", "o", "N", "pau", "cons"] as const;

/** Length of one frame (seconds). The graph output is a 20ms grid (input 10ms × stride 2). */
export const FRAME_SEC = 0.02;
/** Cost of a class switch (on the log-probability scale — larger is smoother). */
const SWITCH_PENALTY = 4.0;
/**
 * Segments shorter than this are absorbed into a neighbour (2 = 40ms. 3 would also erase real
 * short vowels).
 */
const MIN_DURATION_FRAMES = 2;

const CLASS_COUNT = LIPSYNC_CLASSES.length;
const classId = (label: (typeof LIPSYNC_CLASSES)[number]): number => LIPSYNC_CLASSES.indexOf(label);
const CONS_ID = /* @__PURE__ */ classId("cons");
const PAU_ID = /* @__PURE__ */ classId("pau");
/** The partners cons may be absorbed into **forward** (vowels and the moraic nasal). */
const VOWEL_OR_N: readonly number[] = /* @__PURE__ */ (["a", "i", "u", "e", "o", "N"] as const).map(
  classId,
);

/**
 * A run of the class sequence `[classId, start, end]` (start / end are in the unit of the caller's
 * scale).
 */
type ClassRun = readonly [classId: number, start: number, end: number];

/**
 * Fails with the coordinates if the lattice `[frames, 8]` contains even one non-finite value.
 *
 * MUST: do not let it through. A non-finite value raises no exception and turns into **a `.lab`
 * that is valid as a format** — NaN / `+Infinity` make all 8 elements of that row NaN in
 * log_softmax, the comparisons in `viterbiSmooth` (`score[cls] > best` / `stays`) all become
 * false, and the backpointer stays zero-initialised. The result is a single segment "the whole
 * utterance is `a`", and neither the GPU's numerical fault nor the broken asset surfaces anywhere.
 * This sides with the cross-cutting invariant "unsupported or unexpected input fails loudly (no
 * silent approximation)".
 *
 * @param where prefix of the error message (function name + which lattice).
 */
const assertFiniteGrid = (
  values: Float32Array<ArrayBuffer> | Float64Array<ArrayBuffer>,
  frames: number,
  where: string,
): void => {
  for (let frame = 0; frame < frames; frame += 1) {
    for (let cls = 0; cls < CLASS_COUNT; cls += 1) {
      const value = values[frame * CLASS_COUNT + cls];
      if (!Number.isFinite(value)) {
        throw new Error(
          `${where} frame ${frame} / class ${cls} (${LIPSYNC_CLASSES[cls]}) is non-finite (${value})`,
        );
      }
    }
  }
};

/** One line of a `.lab` (seconds). */
export type LabSegment = {
  readonly start: number;
  readonly end: number;
  readonly label: string;
};

/**
 * Smooths the class sequence with a transition-penalty Viterbi.
 *
 * A transition is one of only 2 choices: "stay (cost 0)" or "move **from the best class at that
 * point** (cost `SWITCH_PENALTY`)" — there is no all-to-all transition matrix, so it runs in
 * O(T·C).
 *
 * It is exported so checks can name it (if the spelling of the recurrence slips in one place, the
 * `.lab` only becomes "a plausible but different segmentation", and the anomaly is invisible until
 * the final string comparison).
 */
export const viterbiSmooth = (
  logProbabilities: Float64Array<ArrayBuffer>,
  frames: number,
): Int32Array<ArrayBuffer> => {
  if (logProbabilities.length !== frames * CLASS_COUNT) {
    throw new Error(
      `viterbiSmooth: the log posteriors have ${logProbabilities.length} elements` +
        ` (${frames}×${CLASS_COUNT} are required)`,
    );
  }
  if (frames < 1) throw new Error(`viterbiSmooth: frame count ${frames} is less than 1`);
  // It is a public function and can be called alone (through `logitsToSegments` the entry has
  // already checked).
  assertFiniteGrid(logProbabilities, frames, "viterbiSmooth: log posteriors");

  const score = new Float64Array(CLASS_COUNT);
  for (let cls = 0; cls < CLASS_COUNT; cls += 1) score[cls] = logProbabilities[cls];
  const backpointer = new Int32Array(frames * CLASS_COUNT);
  for (let frame = 1; frame < frames; frame += 1) {
    let best = Number.NEGATIVE_INFINITY;
    let argbest = 0;
    for (let cls = 0; cls < CLASS_COUNT; cls += 1) {
      if (score[cls] > best) {
        best = score[cls];
        argbest = cls;
      }
    }
    const switched = best - SWITCH_PENALTY;
    for (let cls = 0; cls < CLASS_COUNT; cls += 1) {
      const stays = score[cls] >= switched;
      backpointer[frame * CLASS_COUNT + cls] = stays ? cls : argbest;
      score[cls] = (stays ? score[cls] : switched) + logProbabilities[frame * CLASS_COUNT + cls];
    }
  }

  const path = new Int32Array(frames);
  let best = Number.NEGATIVE_INFINITY;
  for (let cls = 0; cls < CLASS_COUNT; cls += 1) {
    if (score[cls] > best) {
      best = score[cls];
      path[frames - 1] = cls;
    }
  }
  for (let frame = frames - 1; frame > 0; frame -= 1) {
    path[frame - 1] = backpointer[frame * CLASS_COUNT + path[frame]];
  }
  return path;
};

/** Class sequence → runs (returned on the scale multiplied by `frameSec`). */
const toRuns = (path: Int32Array<ArrayBuffer>, frameSec: number): ClassRun[] => {
  const runs: ClassRun[] = [];
  let start = 0;
  for (let frame = 1; frame <= path.length; frame += 1) {
    if (frame === path.length || path[frame] !== path[start]) {
      runs.push([path[start], start * frameSec, frame * frameSec]);
      start = frame;
    }
  }
  return runs;
};

/**
 * Repaints segments shorter than `MIN_DURATION_FRAMES` with whichever neighbouring class has the
 * higher posterior over the segment.
 *
 * The scan restarts after each repaint (a repaint changes the structure of the runs itself). This
 * repeats until convergence. `pau` stays even when short.
 */
const mergeShortRuns = (
  path: Int32Array<ArrayBuffer>,
  logProbabilities: Float64Array<ArrayBuffer>,
): Int32Array<ArrayBuffer> => {
  const merged = path.slice();
  let changed = true;
  while (changed) {
    changed = false;
    const runs = toRuns(merged, 1); // Scale 1 = the segment unit is frames
    for (let index = 0; index < runs.length; index += 1) {
      const [cls, start, end] = runs[index];
      if (end - start >= MIN_DURATION_FRAMES || cls === PAU_ID) continue;
      const candidates: number[] = [];
      if (index > 0) candidates.push(runs[index - 1][0]);
      if (index + 1 < runs.length) candidates.push(runs[index + 1][0]);
      if (candidates.length === 0) continue;
      let best = candidates[0];
      let bestScore = Number.NEGATIVE_INFINITY;
      for (const candidate of candidates) {
        let score = 0;
        for (let frame = start; frame < end; frame += 1) {
          score += logProbabilities[frame * CLASS_COUNT + candidate];
        }
        if (score > bestScore) {
          bestScore = score;
          best = candidate;
        }
      }
      merged.fill(best, start, end);
      changed = true;
      break;
    }
  }
  return merged;
};

/**
 * Absorbs cons segments into neighbouring segments, giving a sequence of 7 labels
 * (a/i/u/e/o/N/pau).
 *
 * By default it is absorbed **forward** (with mora structure C+V, the consonant appears as part of
 * the mouth shape of the next vowel). If the following segment is not a vowel/N the preceding
 * segment is extended, and in the degenerate case at the start where neither exists (the whole
 * utterance is cons) it falls to `pau`.
 */
const absorbConsonants = (runs: readonly ClassRun[]): LabSegment[] => {
  const result: LabSegment[] = [];
  let pendingStart: number | undefined;
  for (let index = 0; index < runs.length; index += 1) {
    const [cls, start, end] = runs[index];
    if (cls === CONS_ID) {
      const next = index + 1 < runs.length ? runs[index + 1][0] : undefined;
      if (next !== undefined && VOWEL_OR_N.includes(next)) {
        // Move the start of the following segment earlier (for consecutive cons only the first
        // start is remembered).
        pendingStart ??= start;
      } else if (result.length > 0) {
        const last = result[result.length - 1];
        result[result.length - 1] = { start: last.start, end, label: last.label };
      } else {
        result.push({ start, end, label: "pau" });
      }
      continue;
    }
    const segmentStart = pendingStart ?? start;
    pendingStart = undefined;
    const label = LIPSYNC_CLASSES[cls];
    const last = result.at(-1);
    if (last !== undefined && last.label === label) {
      result[result.length - 1] = { start: last.start, end, label };
    } else {
      result.push({ start: segmentStart, end, label });
    }
  }
  return result;
};

/**
 * Logits `[frames, 8]` (row-major) → `.lab` segment sequence.
 *
 * log_softmax subtracts the maximum before applying (the Python side uses a plain `logsumexp`, but
 * this form, which avoids `exp` overflow, is equivalent as log posteriors — the difference is only
 * f64 rounding, and step 2 onwards look **only at differences**).
 */
export const logitsToSegments = (
  logits: Float32Array<ArrayBuffer>,
  frames: number,
): LabSegment[] => {
  if (logits.length !== frames * CLASS_COUNT) {
    throw new Error(
      `logitsToSegments: the logits have ${logits.length} elements (${frames}×${CLASS_COUNT} are required)`,
    );
  }
  if (frames < 1) throw new Error(`logitsToSegments: frame count ${frames} is less than 1`);
  assertFiniteGrid(logits, frames, "logitsToSegments: logits");

  const logProbabilities = new Float64Array(frames * CLASS_COUNT);
  for (let frame = 0; frame < frames; frame += 1) {
    const row = frame * CLASS_COUNT;
    let peak = Number.NEGATIVE_INFINITY;
    for (let cls = 0; cls < CLASS_COUNT; cls += 1) peak = Math.max(peak, logits[row + cls]);
    let sum = 0;
    for (let cls = 0; cls < CLASS_COUNT; cls += 1) sum += Math.exp(logits[row + cls] - peak);
    const logSum = Math.log(sum) + peak;
    for (let cls = 0; cls < CLASS_COUNT; cls += 1) {
      logProbabilities[row + cls] = logits[row + cls] - logSum;
    }
  }
  const path = mergeShortRuns(viterbiSmooth(logProbabilities, frames), logProbabilities);
  return absorbConsonants(toRuns(path, FRAME_SEC));
};

/**
 * The `.lab` body (lines of `start end label` with seconds to 7 digits, each followed by a
 * newline).
 *
 * MUST: fixed at 7 digits (upstream `lab.py`'s `f"{s.start:.7f}"`). Changing the digits breaks the
 * string-comparison match with existing `.lab` consumers (lip-sync timelines).
 */
export const toLab = (segments: readonly LabSegment[]): string =>
  segments
    .map((segment) => `${segment.start.toFixed(7)} ${segment.end.toFixed(7)} ${segment.label}\n`)
    .join("");
