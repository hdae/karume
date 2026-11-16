/**
 * The fields of a derived profile that differ from the registered table (the dry run's answer to
 * "what would regenerating change"). The provenance fields are listed apart from the table's
 * values: a new record changes the provenance even when every geometry stays.
 */

import { literal } from "./literal.ts";

export type Difference = {
  /** `gemmRows[3].geometry.regM`, `provenance.families["i8a8-linear"].sweep`, ... */
  readonly path: string;
  /** The value as a literal, or undefined when the field is absent on that side. */
  readonly old?: string;
  readonly new?: string;
};

export type ProfileDifferences = {
  readonly fields: readonly Difference[];
  readonly provenance: readonly Difference[];
};

const IDENTIFIER = /^[A-Za-z_$][\w$]*$/;

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const member = (path: string, name: string): string =>
  IDENTIFIER.test(name)
    ? `${path === "" ? "" : `${path}.`}${name}`
    : `${path}[${JSON.stringify(name)}]`;

const walk = (path: string, old: unknown, next: unknown, out: Difference[]): void => {
  if (isRecord(old) && isRecord(next)) {
    const names = [...new Set([...Object.keys(old), ...Object.keys(next)])];
    for (const name of names) walk(member(path, name), old[name], next[name], out);
    return;
  }
  if (Array.isArray(old) && Array.isArray(next)) {
    for (let index = 0; index < Math.max(old.length, next.length); index++) {
      walk(`${path}[${index}]`, old[index], next[index], out);
    }
    return;
  }
  if (Object.is(old, next)) return;
  out.push({
    path,
    ...(old === undefined ? {} : { old: literal(old) }),
    ...(next === undefined ? {} : { new: literal(next) }),
  });
};

/** Every leaf that differs between the registered table and the derived one, in field order. */
export const profileDifferences = (registered: unknown, derived: unknown): ProfileDifferences => {
  const all: Difference[] = [];
  walk("", registered, derived, all);
  const inProvenance = (difference: Difference): boolean =>
    difference.path === "provenance" ||
    difference.path.startsWith("provenance.") ||
    difference.path.startsWith("provenance[");
  return {
    fields: all.filter((difference) => !inProvenance(difference)),
    provenance: all.filter(inProvenance),
  };
};

/** One difference as a line (`path: old -> new`, `(absent)` for a missing side). */
export const differenceLine = (difference: Difference): string =>
  `${difference.path}: ${difference.old ?? "(absent)"} -> ${difference.new ?? "(absent)"}`;
