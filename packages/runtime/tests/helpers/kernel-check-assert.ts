/**
 * Asserts a part or a case of a shared kernel check (`@karume/kernel-checks`): every criterion of
 * every case has 0 failing words, and every requirement is met. The message names the case, the
 * criterion and the failing word as u32 hex.
 */

import { assert, assertEquals } from "@std/assert";
import type { CheckCase, CheckPart, Criterion } from "@karume/kernel-checks";

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
