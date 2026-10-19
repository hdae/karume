/**
 * **Incremental** decoding for tokenizers with byte_fallback (`push(id) → settled string`).
 *
 * It is **family-independent** (the `ByteFallback` + `Fuse` decoder chain does not depend on the
 * model kind), so it lives in `src/text/`. Resolving id → spelling and handling special tokens is
 * the family's job, received here through {@link DetokenizerSource}.
 *
 * The source of truth is `ByteFallback::decode_chain` of `tokenizers`:
 *
 *   A sequence of `<0xHH>` (a **run**) is accumulated and decoded as UTF-8 when the run ends. If
 *   the whole run is valid UTF-8 it becomes one string; otherwise it becomes **one** U+FFFD **per
 *   byte** (it does not take out just the valid prefix — which is why the run has to be carried to
 *   its end).
 *
 * MUST: **accumulate nothing but the run**. A "fake streaming" that accumulates everything until
 * `finish()` and returns it then makes each iteration of ADR 0083 decision 2's `AsyncIterable`
 * meaningless (the consumer sees no progress). A test pins this directly.
 *
 * NOTE: conversely, nothing can be emitted **in the middle of** a run ("3 bytes are complete, emit
 * one character" is not possible) — if one following byte makes the whole run invalid, even the
 * valid prefix turns into replacement characters. In practice this matters rarely: characters in
 * the vocabulary (Japanese included) make no byte run; only out-of-vocabulary symbols and emoji
 * become runs, and they close within a few bytes.
 *
 * MUST: special tokens being skipped **do not cut the run**. The source of truth skips them in the
 * step that builds "id sequence → spelling sequence", and the decoder chain runs after that, so
 * bytes on both sides of a skipped token are adjacent (`[<0xE3>, <eos>, <0x81>, <0x82>]` with
 * skipping becomes the one character `あ`).
 *
 * ## The stop-string check lives in the same layer ({@link createStopStringFilter})
 *
 * Stop **strings** can only be checked on "the sequence of settled strings", and the check is
 * family-independent, so this is its place. It is a second machine with **separate state** (one
 * pending tail) from the decoding state machine, and leaves the "return only settled pieces"
 * contract of {@link StreamingDetokenizer} untouched.
 */

import { ModelInputError } from "../errors.ts";

/** The entry that resolves an id to a spelling (the family passes its knowledge in here). */
export type DetokenizerSource = {
  /** The byte value if the id is in the byte_fallback vocabulary, otherwise `undefined`. */
  readonly byteOf: (id: number) => number | undefined;
  /**
   * The id's output spelling (with the decoder chain applied up to the per-token transforms). Only
   * special tokens to be skipped return `undefined`.
   *
   * MUST: **an unknown id fails instead of returning `undefined`** (the source of truth silently
   * skips it, but here a mismatch between the assets and the id sequence is not silenced — the
   * cross-cutting fail-loudly invariant).
   */
  readonly textOf: (id: number) => string | undefined;
};

/** The `push(id) → settled string` state machine. The only state is one byte run. */
export class StreamingDetokenizer {
  readonly #source: DetokenizerSource;
  /**
   * MUST: do not let it strip the BOM (`ignoreBOM: true`). The default `TextDecoder` silently drops
   * a leading U+FEFF, but the source of truth's `String::from_utf8` does not — only an `EF BB BF`
   * run would quietly disappear.
   */
  readonly #decoder = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true });
  #pending: number[] = [];

  constructor(source: DetokenizerSource) {
    this.#source = source;
  }

  /** Advances one token. Returns only what **this push settled** (an empty string if nothing). */
  push(id: number): string {
    const text = this.#source.textOf(id);
    if (text === undefined) return "";
    const byte = this.#source.byteOf(id);
    if (byte !== undefined) {
      this.#pending.push(byte);
      return "";
    }
    return this.#flush() + text;
  }

  /** End of the sequence. Settles the carried byte run. */
  finish(): string {
    return this.#flush();
  }

  #flush(): string {
    if (this.#pending.length === 0) return "";
    const bytes = Uint8Array.from(this.#pending);
    this.#pending = [];
    try {
      return this.#decoder.decode(bytes);
    } catch {
      // A whole run that is invalid gives one replacement character per byte (the failure path of
      // the source of truth's `String::from_utf8`).
      return "�".repeat(bytes.length);
    }
  }
}

/**
 * What one {@link StopStringFilter.push} settled in the stop-**string** check.
 *
 * Once `matched` is set, `text` holds the string that push passed **only up to the stop string**
 * (the stop string itself and everything after it are dropped).
 */
export type StopStringChunk = {
  /**
   * The string that may be output (excluding the pending part; an empty string if nothing to emit).
   */
  readonly text: string;
  /** The stop string that matched (`undefined` = not stopped yet). */
  readonly matched?: string;
};

/**
 * Sequence of settled strings → the sequence cut before the stop string (a `push` / `finish` state
 * machine).
 *
 * The only state is **one pending tail**.
 */
export type StopStringFilter = {
  /**
   * Advances one settled piece (the return value of {@link StreamingDetokenizer.push} can be passed
   * as is).
   */
  push(text: string): StopStringChunk;
  /** End of the sequence. Settles the pending part (the tail when it ended without stopping). */
  finish(): string;
};

/**
 * The length of "the tail that could be a **prefix** of a stop string".
 *
 * This is the amount held back. A stop string can appear **across** token boundaries, so emitting
 * every settled string at once would mean "stopping after having emitted part of the string that
 * should have stopped it". Holding back a fixed length (longest stop string − 1 character) instead
 * would delay rendering on every turn that does not stop, so only the part that **really is a
 * prefix** is held.
 */
const partialTail = (buffer: string, stopStrings: readonly string[]): number => {
  let longest = 0;
  for (const stop of stopStrings) {
    // A whole match is checked by the caller first, so only proper prefixes
    // (`length < stop.length`) are checked here.
    for (let length = Math.min(stop.length - 1, buffer.length); length > longest; length -= 1) {
      if (buffer.endsWith(stop.slice(0, length))) {
        longest = length;
        break;
      }
    }
  }
  return longest;
};

/**
 * Incremental stop-**string** check (the upper of the two layers of ADR 0083's addendum of
 * 2026-09-02).
 *
 * Stop tokens are checked by the sequence layer (token ids are settled in a layer that knows no
 * strings), but stop **strings** can only be checked after decoding — one stop string can be split
 * over several tokens, and one token can straddle the end of a stop string and the next text. So
 * its place is here (on the sequence of settled strings), and it is **a separate hold-back** from
 * the run carry-over of `byte_fallback` ({@link StreamingDetokenizer} keeps its "return only
 * settled pieces" contract).
 *
 * MUST: empty strings and duplicates fail loudly. An empty string "always matches" = a
 * specification that cannot emit a single character, and a duplicate cannot mean more than writing
 * the same condition twice (both are signs of a caller mix-up).
 *
 * MUST: also check that it is an array and that each element is a string (untyped JS / JSON input
 * arrives). A string itself would spread into an array of characters and stop at the first
 * character, and a number element would stop at another position through `indexOf`'s implicit
 * conversion — neither throws; only the stop position changes.
 */
export const createStopStringFilter = (stopStrings: readonly string[]): StopStringFilter => {
  if (!Array.isArray(stopStrings)) {
    throw new ModelInputError("stopStrings is not an array of strings");
  }
  // MUST: copy at issue (if the array were rewritten while running, the stop condition would change
  // midway).
  const elements: readonly unknown[] = stopStrings;
  const stops: string[] = [];
  const declared = new Set<string>();
  // MUST: iterate by index — forEach / for-in skip the holes of a sparse array, and the hole of
  // `[, "END"]` would be silently dropped without passing the type check. Read by index, a hole is
  // rejected as undefined.
  for (let index = 0; index < elements.length; index++) {
    const stop = elements[index];
    if (typeof stop !== "string") {
      throw new ModelInputError(`stopStrings[${index}] ${String(stop)} is not a string`);
    }
    if (stop === "") throw new ModelInputError(`stopStrings[${index}] is an empty string`);
    if (declared.has(stop)) {
      throw new ModelInputError(`stopStrings contains ${JSON.stringify(stop)} twice`);
    }
    declared.add(stop);
    stops.push(stop);
  }

  let held = "";
  return {
    push(text: string): StopStringChunk {
      const buffer = held + text;
      // Take the **earliest** match (even if several stop strings are found at once, the one that
      // actually cut the stream is the one that appeared first). At the same position, the one
      // declared first.
      let at = -1;
      let matched: string | undefined;
      for (const stop of stops) {
        const index = buffer.indexOf(stop);
        if (index >= 0 && (at < 0 || index < at)) {
          at = index;
          matched = stop;
        }
      }
      if (matched !== undefined) {
        held = "";
        return { text: buffer.slice(0, at), matched };
      }
      const keep = partialTail(buffer, stops);
      held = buffer.slice(buffer.length - keep);
      return { text: buffer.slice(0, buffer.length - keep) };
    },
    finish(): string {
      const tail = held;
      held = "";
      return tail;
    },
  };
};

/**
 * Decodes a whole sequence at once (through the same state machine as the incremental path — no
 * second implementation).
 */
export const detokenize = (source: DetokenizerSource, ids: Iterable<number>): string => {
  const detokenizer = new StreamingDetokenizer(source);
  let out = "";
  for (const id of ids) out += detokenizer.push(id);
  return out + detokenizer.finish();
};
