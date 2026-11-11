/**
 * Asserts a part or a case of a shared kernel check (`@karume/kernel-checks`): every criterion of
 * every case has 0 failing words, and every requirement is met. The message names the case, the
 * criterion and the failing word as u32 hex.
 */

import { assert, assertEquals } from "@std/assert";
import type { CheckCase, CheckPart, Criterion } from "@karume/kernel-checks";
import { type NanPolicy, wordIdentity } from "../../src/tune/word-identity.ts";

const hex = (word: number): string => `0x${word.toString(16).padStart(8, "0")}`;

const failureText = (criterion: Criterion): string => {
  const failure = criterion.failure;
  if (failure === undefined) return "";
  const note = failure.note === undefined ? "" : ` (${failure.note})`;
  return ` — ${failure.at} at word ${failure.index}: expected ${hex(failure.expected)}, actual ${hex(failure.actual)}${note}`;
};

/** Asserts every criterion of one case (`prefix` heads the message — a part's id). */
export const assertCheckCase = (checkCase: CheckCase, prefix = ""): void => {
  for (const criterion of checkCase.criteria) {
    assertEquals(
      criterion.failures,
      0,
      `${prefix}${checkCase.label}: ${criterion.name}: ${criterion.failures} of ${criterion.words} words${failureText(criterion)}`,
    );
  }
};

export const assertCheckPart = (part: CheckPart): void => {
  for (const checkCase of part.cases) assertCheckCase(checkCase, `${part.id} `);
  for (const requirement of part.requirements) {
    assert(requirement.met, `${part.id}: ${requirement.name} — ${requirement.detail}`);
  }
};

/**
 * Asserts that `actual` equals `expected` word for word under the NaN policy `nan` — the comparison
 * the checks make (`wordIdentity` in src/tune/word-identity.ts), for a comparison no check makes
 * (against a host-built expectation, or between two runs of the same side).
 */
export const assertWordIdentity = (
  actual: Uint32Array<ArrayBuffer>,
  expected: Uint32Array<ArrayBuffer>,
  nan: NanPolicy,
  label: string,
): void => {
  const { words, failures, first } = wordIdentity(actual, expected, nan);
  const at =
    first === undefined
      ? ""
      : ` — first at word ${first.index}: expected ${hex(first.expected)}, actual ${hex(first.actual)}`;
  assertEquals(failures, 0, `${label}: ${failures} of ${words} words (NaN by ${nan})${at}`);
};
