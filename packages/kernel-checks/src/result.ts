/**
 * The structured result every kernel check returns, and the word comparisons behind it. A check
 * does not assert: it runs every case, counts the words that fail each criterion and keeps the
 * first (or, for a tolerance band, the worst) failing word, so the node GPU tests assert on the
 * counts and the playground page shows them.
 *
 * A failure of the harness itself (a WebGPU error, a lost device, a mutation anchor that no longer
 * matches the generated WGSL) is not a result: it throws, so a broken harness never reads as a
 * zero-mismatch run. Every case's GPU work runs inside {@link scopedGpuWork} for that reason.
 */

import { assertDeviceUsable, type GpuContext } from "../../runtime/src/gpu/device.ts";
import { type NanPolicy, wordIdentity } from "../../runtime/src/tune/word-identity.ts";

/** One failing word: its index in the compared range and both values as u32 words. */
export type WordFailure = {
  readonly index: number;
  readonly expected: number;
  readonly actual: number;
  /** `first` for an identity criterion, `worst` for a tolerance band (the largest error). */
  readonly at: "first" | "worst";
  /** What else the criterion knows (the tolerance report, a count criterion's meaning). */
  readonly note?: string;
};

/** One criterion of a case: how many of its words failed it, and one failing word. */
export type Criterion = {
  readonly name: string;
  readonly words: number;
  readonly failures: number;
  /** Present exactly when `failures > 0`. */
  readonly failure?: WordFailure;
};

export type CheckCase = {
  readonly label: string;
  /** The class a case belongs to, where a check reports per class (the folded ①' classes). */
  readonly group?: string;
  readonly criteria: readonly Criterion[];
};

/** A whole-part condition (the word classes the grid met, an output left unwritten). */
export type Requirement = {
  /** Stable within a part (`census`, `full-write`…): callers look a requirement up by it, not by name. */
  readonly id: string;
  readonly name: string;
  readonly met: boolean;
  readonly detail: string;
};

/** One gate of a check (a check has one part or several). */
export type CheckPart = {
  readonly id: string;
  readonly title: string;
  readonly cases: readonly CheckCase[];
  readonly requirements: readonly Requirement[];
  /**
   * Present when the device lacks what the part needs (a WGSL language feature, a form of a kernel
   * it never takes): the reason. Such a part has no cases and MUST NOT read as a pass — it was not
   * run here.
   */
  readonly notRun?: string;
};

/** The part a device cannot run: no cases, and the reason. */
export const notRunPart = (id: string, title: string, reason: string): CheckPart => ({
  id,
  title,
  cases: [],
  requirements: [],
  notRun: reason,
});

/**
 * The number of cases each part of a check MUST return, by part id. A part with another count did
 * not run the check's grid (a part with no cases would otherwise pass with nothing compared).
 */
export type ExpectedCases = Readonly<Record<string, number>>;

/** Cancellation and per-case progress of a running check. */
export type CheckOptions = {
  /** Checked between cases; an abort rejects with the signal's reason. */
  readonly signal?: AbortSignal;
  readonly onProgress?: (progress: CheckProgress) => void;
};

export type CheckProgress = {
  readonly part: string;
  readonly done: number;
  readonly total: number;
};

/**
 * Whether the device can run the subgroup32 variants: the runtime requests `subgroups` and
 * `subgroup-size-control` only when the adapter has both and WGSL lists `subgroup_id`
 * (`acquireGpu({ subgroups: true | "available" })`), so a device without them cannot run the checks.
 */
export const subgroup32Available = (
  gpu: Pick<GpuContext, "features" | "wgslLanguageFeatures">,
): boolean =>
  gpu.features.has("subgroups") &&
  gpu.features.has("subgroup-size-control") &&
  gpu.wgslLanguageFeatures.has("subgroup_id");

/**
 * The identity criterion of two word arrays under a NaN policy (the comparison is
 * {@link wordIdentity}, shared with the runtime). A first failure past the shorter array notes the
 * two lengths.
 */
const compareWordsUnder = (
  name: string,
  actual: Uint32Array<ArrayBuffer>,
  expected: Uint32Array<ArrayBuffer>,
  nan: NanPolicy,
): Criterion => {
  const { words, failures, first } = wordIdentity(actual, expected, nan);
  if (first === undefined) return { name, words, failures };
  const lengthOnly = first.index >= Math.min(actual.length, expected.length);
  const failure: WordFailure = lengthOnly
    ? { ...first, at: "first", note: `length ${actual.length} vs ${expected.length}` }
    : { ...first, at: "first" };
  return { name, words, failures, failure };
};

/** Compares two word arrays one by one; a length difference counts each missing word. */
export const compareWords = (
  name: string,
  actual: Uint32Array<ArrayBuffer>,
  expected: Uint32Array<ArrayBuffer>,
): Criterion => compareWordsUnder(name, actual, expected, "payload");

/**
 * {@link compareWords} with NaN compared by class: where `expected` holds a NaN, `actual` must hold a
 * NaN of any payload and sign; every other word must be equal (an actual NaN against a non-NaN
 * expected word fails). For words a kernel computes, whose NaN payloads are backend-defined.
 */
export const compareWordsNanClass = (
  name: string,
  actual: Uint32Array<ArrayBuffer>,
  expected: Uint32Array<ArrayBuffer>,
): Criterion => compareWordsUnder(name, actual, expected, "class");

/** Every word of `actual` must be `expected` (a poison value a kernel must leave in place). */
export const compareFill = (
  name: string,
  actual: Uint32Array<ArrayBuffer>,
  expected: number,
): Criterion => compareWords(name, actual, new Uint32Array(actual.length).fill(expected));

/** A count that must equal `expected` (one "word": the count itself). */
export const compareCount = (
  name: string,
  actual: number | undefined,
  expected: number,
): Criterion =>
  actual === expected
    ? { name, words: 1, failures: 0 }
    : {
        name,
        words: 1,
        failures: 1,
        failure: {
          index: 0,
          expected,
          actual: actual ?? 0,
          at: "first",
          note: `expected ${expected}, got ${actual ?? "none"}`,
        },
      };

/** The u32 word of `value` rounded to f32. */
export const f32Word = (value: number): number => new Uint32Array(Float32Array.of(value).buffer)[0];

/** The u32 words of an f32 array (a view, no copy). */
export const wordsOf = (
  data: Float32Array<ArrayBuffer> | Uint32Array<ArrayBuffer> | Int32Array<ArrayBuffer>,
): Uint32Array<ArrayBuffer> => new Uint32Array(data.buffer, data.byteOffset, data.length);

/** Throws the signal's reason when the run was cancelled. */
export const throwIfAborted = (options: CheckOptions): void => options.signal?.throwIfAborted();

/**
 * Runs one case's GPU work under the validation, out-of-memory and internal error scopes and throws
 * on anything the work left behind: a caught error, an uncaptured error (one raised outside these
 * scopes while the work ran) or a lost device. Most WebGPU failures do not throw — an invalid shader,
 * a backend compile failure (DXC / FXC on D3D12) or an allocation the device cannot serve yields an
 * invalid object, the submit is dropped and the readback returns the buffers' initial contents, which
 * the two runs of a case would compare as identical (and a mutant that no longer compiles as
 * mismatching).
 *
 * MUST NOT run two of these at once on one device: error scopes are a per-device stack and the work
 * awaits between the push and the pops. The checks run their cases one after another.
 */
export const scopedGpuWork = async <T>(
  gpu: GpuContext,
  where: string,
  work: () => Promise<T>,
): Promise<T> => {
  const { device } = gpu;
  assertDeviceUsable(gpu, where);
  const uncaptured: GPUError[] = [];
  const onUncaptured = (event: GPUUncapturedErrorEvent): void => {
    uncaptured.push(event.error);
  };
  device.addEventListener("uncapturederror", onUncaptured);
  try {
    device.pushErrorScope("internal");
    device.pushErrorScope("out-of-memory");
    device.pushErrorScope("validation");
    // Called from a reaction, so a synchronous throw of the work also settles as its rejection and
    // the scopes are still popped
    const settled = await Promise.resolve()
      .then(work)
      .then(
        (value) => ({ ok: true, value }) as const,
        (error: unknown) => ({ ok: false, error }) as const,
      );
    // The three pops are issued in one synchronous section: a pop takes the top of the stack
    const caught = await Promise.all([
      device.popErrorScope(),
      device.popErrorScope(),
      device.popErrorScope(),
    ]).catch((popError: unknown) => {
      // A rejected pop (an unbalanced stack) is the harness failing; the work's own failure goes along
      if (settled.ok) throw popError;
      const reason = popError instanceof Error ? popError.message : String(popError);
      throw new Error(`${where}: popErrorScope rejected: ${reason}`, { cause: settled.error });
    });
    const error = caught.find((scoped) => scoped !== null) ?? uncaptured[0];
    if (error !== undefined) {
      // The WebGPU error is the cause; a rejection of the work (a readback of a dropped submit) follows from it
      throw new Error(`${where}: WebGPU ${error.constructor.name}: ${error.message}`, {
        cause: settled.ok ? undefined : settled.error,
      });
    }
    assertDeviceUsable(gpu, where);
    if (!settled.ok) throw settled.error;
    return settled.value;
  } finally {
    device.removeEventListener("uncapturederror", onUncaptured);
  }
};
