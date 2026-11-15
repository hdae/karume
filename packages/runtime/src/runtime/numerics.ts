/**
 * The numeric requirement a Session runs under ({@link Numerics} — `SessionOptions.numerics`): how
 * far each kind of op may stand from its reference kernel, declared for the whole model and
 * overridden per op kind (ADR 0152 decision 2).
 *
 * The requirement is what a model's preset declares; which kernel meets it is the runtime's choice.
 * A level admits strategies, it does not name one: an op kind with no strategy at the declared level
 * runs its reference kernel, and declaring `reorder` for it is not an error.
 *
 * Approximations are not levels here. Each is asked for by its own Session option
 * (`linearCompute: "a8"`, `attentionScoreStorage: "f16"`, …) and is never implied by a level.
 *
 * Pure and device-free (the memory estimate reads the same functions as Session construction).
 */

import { describeInputValue } from "./plan.ts";
import type { RequirementLevel } from "./strategy-level.ts";

/** The levels a declaration can ask for (an approximation is named by its own option). */
export type NumericLevel = Exclude<RequirementLevel, "approximation">;

/** The value set of {@link NumericLevel} (a union member added without a row fails type checking). */
const NUMERIC_LEVELS: Readonly<Record<NumericLevel, true>> = { exact: true, reorder: true };

/**
 * The op kinds a declaration can address: the kinds that have a strategy below `exact`.
 * `attention` covers both forms of attention (today only the states form has such strategies).
 */
export const NUMERIC_OP_KINDS = ["linear", "attention", "rms_norm"] as const;

export type NumericOpKind = (typeof NUMERIC_OP_KINDS)[number];

/**
 * The declaration. `level` is the whole model's (omitted: `exact`); `ops` overrides it per op kind,
 * and the narrower scope wins in either direction (`level: "reorder"` with `ops.attention: "exact"`,
 * or `ops.linear: "reorder"` alone).
 *
 * - `exact`: the reference kernel's bits on this device. Not "the same on every GPU".
 * - `reorder`: the summation order may differ from the reference kernel's; the error stays inside
 *   the op's band against an f64 evaluation, and two runs of the same input on one device are
 *   bit-identical.
 */
export type Numerics = {
  readonly level?: NumericLevel;
  readonly ops?: { readonly [K in NumericOpKind]?: NumericLevel };
};

/** The level each op kind runs under, with the scopes resolved. */
export type NumericLevels = { readonly [K in NumericOpKind]: NumericLevel };

const levelViolation = (value: unknown, where: string): string | undefined =>
  typeof value === "string" && Object.hasOwn(NUMERIC_LEVELS, value)
    ? undefined
    : `${where} ${describeInputValue(value)} must be "exact" or "reorder"`;

/**
 * The violation of a `numerics` value from outside the type (`undefined` when it is acceptable).
 *
 * MUST: an unknown key fails. A misspelled op kind read as "not declared" would run the model's
 * level where the caller asked for another one, with nothing in the numbers to show it.
 * MUST: do not call the value's conversions (the rule of `sessionOptionsViolation`).
 */
export const numericsViolation = (numerics: unknown): string | undefined => {
  if (numerics === undefined) return undefined;
  if (typeof numerics !== "object" || numerics === null || Array.isArray(numerics)) {
    return `options.numerics must be an object { level?, ops? } (got ${describeInputValue(numerics)})`;
  }
  const record: Readonly<Record<string, unknown>> = { ...numerics };
  for (const key of Object.keys(record)) {
    if (key !== "level" && key !== "ops") return `options.numerics.${key} is not a field`;
  }
  if (record["level"] !== undefined) {
    const violation = levelViolation(record["level"], "options.numerics.level");
    if (violation !== undefined) return violation;
  }
  const ops = record["ops"];
  if (ops === undefined) return undefined;
  if (typeof ops !== "object" || ops === null || Array.isArray(ops)) {
    return `options.numerics.ops must be an object of op kind → level (got ${describeInputValue(ops)})`;
  }
  const kinds: Readonly<Record<string, unknown>> = { ...ops };
  const known: readonly string[] = NUMERIC_OP_KINDS;
  for (const [kind, level] of Object.entries(kinds)) {
    if (!known.includes(kind)) {
      return `options.numerics.ops.${kind} is not an op kind (${NUMERIC_OP_KINDS.join(" / ")})`;
    }
    if (level === undefined) continue;
    const violation = levelViolation(level, `options.numerics.ops.${kind}`);
    if (violation !== undefined) return violation;
  }
  return undefined;
};

/**
 * The level of each op kind under a declaration: the op kind's own, else the model's, else `exact`.
 *
 * MUST: the caller has run {@link numericsViolation} first (the values are read as checked).
 */
export const numericLevels = (numerics: Numerics | undefined): NumericLevels => {
  const model = numerics?.level ?? "exact";
  return {
    linear: numerics?.ops?.linear ?? model,
    attention: numerics?.ops?.attention ?? model,
    rms_norm: numerics?.ops?.rms_norm ?? model,
  };
};

const isRecord = (value: unknown): boolean =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const hasOnlyKeys = (value: object, allowed: readonly string[]): boolean =>
  Object.keys(value).every((key) => allowed.includes(key));

/**
 * A copy of a declaration as plain data, each field read once (Session construction copies its
 * options at the entry, so the value {@link numericsViolation} checks is the value
 * {@link numericLevels} reads — a getter cannot hand the two different values).
 *
 * A value {@link numericsViolation} rejects by its own type or by its key names is returned as it
 * is: neither changes between the two reads, and the check that follows rejects it. Field values
 * are copied as read, valid or not.
 */
export const copyNumerics = (numerics: Numerics | undefined): Numerics | undefined => {
  if (numerics === undefined || !isRecord(numerics)) return numerics;
  if (!hasOnlyKeys(numerics, ["level", "ops"])) return numerics;
  const { level, ops } = numerics;
  const model = level === undefined ? {} : { level };
  if (ops === undefined) return model;
  if (!isRecord(ops)) return { ...model, ops };
  if (!hasOnlyKeys(ops, NUMERIC_OP_KINDS)) return numerics;
  const { linear, attention, rms_norm } = ops;
  return {
    ...model,
    ops: {
      ...(linear === undefined ? {} : { linear }),
      ...(attention === undefined ? {} : { attention }),
      ...(rms_norm === undefined ? {} : { rms_norm }),
    },
  };
};
