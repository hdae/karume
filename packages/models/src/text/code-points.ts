/**
 * String → code-point sequence.
 *
 * It is **family-independent** (the Unigram, BPE and normalization paths all start counting on "a
 * sequence of code points", as the source of truth does), so it lives in `src/text/`. In a family
 * directory, another family using it would mean a cross-family import (a reversed dependency).
 */

import { ModelInputError } from "../errors.ts";

/**
 * Unpaired surrogates are **not accepted** — they are values Python / Rust strings cannot hold, so
 * the source of truth's output is undefined. Silently collapsing them to U+FFFD would quietly
 * change the id sequence, so they fail (input only the JS string type can express — the defence
 * most easily forgotten in a port).
 */
export const toCodePoints = (text: string): number[] => {
  const out: number[] = [];
  for (const ch of text) {
    const cp = ch.codePointAt(0) as number;
    if (cp >= 0xd800 && cp <= 0xdfff) {
      throw new Error(`対にならないサロゲート U+${cp.toString(16).toUpperCase()} が入力にある`);
    }
    out.push(cp);
  }
  return out;
};

/**
 * Checks **at the generation entry** that the caller's text passes {@link toCodePoints}.
 *
 * MUST: the source of truth for the acceptance condition stays the single {@link toCodePoints} —
 * this raises **only the thrown type**. It is not replaced wholesale because the same throw also
 * occurs while parsing vocabulary assets, where it is an asset mismatch (fixing the caller's text
 * does not fix it), so a plain `Error` is right there. Only the `encode` of families that accept
 * the caller's text goes through this entry (ADR 0107 decision 2).
 */
export const assertEncodableText = (text: string, label: string): void => {
  try {
    toCodePoints(text);
  } catch (cause) {
    throw new ModelInputError(
      `${label}: ${cause instanceof Error ? cause.message : String(cause)}`,
      { cause },
    );
  }
};
