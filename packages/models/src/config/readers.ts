/**
 * Basic readers for `pipelineConfig` (a lower layer shared across families).
 *
 * As ADR 0038 §1 states, each pipeline implementation owns the schema of `pipelineConfig`. What lives
 * here is **only the reading procedure whose skeleton is the same in every family**; **the
 * canonical declarations** — allowed key sets, defaults, accepted numeric conditions, accepted fixed
 * values — **stay in each family's `config.ts`**. The text naming which field was read (`where` /
 * `key`) and the description of the condition (`requirement` / `why`) also come from the caller, so
 * the error text is the family's responsibility too.
 *
 * MUST: maps are looked up only through `Object.hasOwn` (the cross-cutting invariant).
 * MUST: unknown keys fail loudly (if a misspelling silently fell back to the default, the
 * preprocessing the distributor intended and the actual execution would disagree unnoticed).
 */

/** Number of mean / std elements (RGB — alpha is not accepted at the entry). */
const CHANNELS = 3;

/**
 * Whether it is a plain object (arrays are **not included** — if `["a"]` silently passed as
 * `{ "0": "a" }` where it is read as a set of fields, the unknown-key check and the defaults would
 * behave with a different meaning).
 */
export const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

/** Reads as a set of fields (missing and not-an-object are fixed the same way, so one message). */
export const readRecord = (raw: unknown, where: string): Record<string, unknown> => {
  if (!isRecord(raw)) throw new Error(`${where}: missing / not an object`);
  return raw;
};

/** Acceptance condition for size / count fields (passed as the `check` of {@link readNumber}). */
export const isPositiveInteger = (value: number): boolean => Number.isInteger(value) && value > 0;

/** Fails on the first key found outside the allowed set. */
export const assertAllowedKeys = (
  value: Record<string, unknown>,
  allowed: readonly string[],
  where: string,
): void => {
  for (const key of Object.keys(value)) {
    if (!allowed.includes(key)) {
      throw new Error(`${where}: unknown key '${key}' (allowed: ${allowed.join(" / ")})`);
    }
  }
};

/** Reads a number field (the caller passes the range test `check` and the message `requirement` used on failure). */
export const readNumber = (
  raw: Record<string, unknown>,
  key: string,
  where: string,
  check: (value: number) => boolean,
  requirement: string,
): number => {
  if (!Object.hasOwn(raw, key)) throw new Error(`${where}.${key}: missing`);
  const value = raw[key];
  if (typeof value !== "number" || !check(value)) {
    throw new Error(`${where}.${key}: ${requirement} (${String(value)})`);
  }
  return value;
};

/**
 * Reads the 3 per-channel constants.
 *
 * MUST: `std` rejects 0 (the caller's `parse*PipelineConfig` passes it as `check`) — division by
 * zero throws nothing, produces `pixel_values` of `±Infinity`, and the graph emits NaN while only the
 * shape matches.
 */
export const readChannels = (
  raw: Record<string, unknown>,
  key: string,
  where: string,
  check: (value: number) => boolean,
  requirement: string,
): readonly [number, number, number] => {
  if (!Object.hasOwn(raw, key)) throw new Error(`${where}.${key}: missing`);
  const value = raw[key];
  if (!Array.isArray(value) || value.length !== CHANNELS) {
    throw new Error(
      `${where}.${key}: not an array of length ${CHANNELS} (${JSON.stringify(value)})`,
    );
  }
  for (const entry of value) {
    if (typeof entry !== "number" || !check(entry)) {
      throw new Error(`${where}.${key}: ${requirement} (${JSON.stringify(value)})`);
    }
  }
  return [value[0], value[1], value[2]];
};

/** A field whose accepted set has a single value. A misspelling and an unsupported value fail with the same message. */
export const readOnly = <T extends string>(
  raw: Record<string, unknown>,
  key: string,
  where: string,
  accepted: T,
  why: string,
): T => {
  if (!Object.hasOwn(raw, key)) throw new Error(`${where}.${key}: missing`);
  const value = raw[key];
  if (value !== accepted) {
    throw new Error(
      `${where}.${key}: this implementation supports only '${accepted}' (${String(value)}) — ${why}`,
    );
  }
  return accepted;
};
