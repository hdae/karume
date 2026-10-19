/**
 * The guard for **reading the JSON of a descriptor / IR safely**, and helpers for canonical
 * serialization — docs/container-v1.md §0 / §10, docs/ir-v2.md "Canonical serialization".
 *
 * - Reading: invalid UTF-8, invalid JSON, non-finite numbers (`1e999` is grammatically valid and
 *   rounds to Infinity), too-deep nesting and `__proto__` keys fail loudly.
 * - Writing: maps keyed by name are in **code point order** (not UTF-16 unit order). The spelling of
 *   numbers is canonical as `JSON.stringify` = ECMAScript `Number::toString` writes it.
 */

import { ContainerFormatError } from "./header.ts";
import { MAX_JSON_DEPTH } from "./limits.ts";

export type JsonObject = Readonly<Record<string, unknown>>;

export const isJsonObject = (value: unknown): value is JsonObject =>
  typeof value === "object" && value !== null && !Array.isArray(value);

/**
 * Checks a value after `JSON.parse` (each branch says why it fails loudly).
 * Depth counts the root as 0.
 */
export const assertJsonSafe = (
  value: unknown,
  path: string,
  depth: number,
  fail: (message: string) => never,
): void => {
  if (depth > MAX_JSON_DEPTH) fail(`${path}: nesting exceeded the depth limit ${MAX_JSON_DEPTH}`);
  if (typeof value === "number") {
    if (!Number.isFinite(value))
      fail(`${path}: non-finite number (including overflows such as 1e999)`);
    return;
  }
  if (value === null || typeof value === "string" || typeof value === "boolean") return;
  if (Array.isArray(value)) {
    value.forEach((item, index) => assertJsonSafe(item, `${path}[${index}]`, depth + 1, fail));
    return;
  }
  if (isJsonObject(value)) {
    for (const key of Object.keys(value)) {
      // The name whose assignment to a plain `{}` turns into setting [[Prototype]], so no own
      // property is created. Deno disabled this setter by default, but Node and browsers (the
      // target runtimes) perform it.
      if (key === "__proto__") fail(`${path}: key '__proto__' is not accepted`);
      assertJsonSafe(value[key], `${path}.${key}`, depth + 1, fail);
    }
    return;
  }
  fail(`${path}: value not representable as JSON`);
};

/** Bytes → a checked JSON value. `maxBytes` is the per-document limit (§10). */
export const decodeJsonDocument = (
  bytes: Uint8Array<ArrayBuffer>,
  path: string,
  maxBytes: number,
  fail: (message: string) => never,
): unknown => {
  if (bytes.byteLength > maxBytes) {
    fail(`${path} exceeds the limit of ${maxBytes} bytes: ${bytes.byteLength}`);
  }
  let text: string;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch (cause) {
    fail(`${path} is invalid UTF-8: ${String(cause)}`);
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (cause) {
    fail(`${path} is invalid JSON: ${String(cause)}`);
  }
  assertJsonSafe(parsed, path, 0, fail);
  return parsed;
};

/**
 * Code point order comparison (ir-v2.md "Canonical serialization"). The default of
 * `Array.prototype.sort` is UTF-16 unit order, so non-BMP characters (surrogate pairs) change the
 * order — names are all ASCII in practice, but the rule is fixed to one regardless of the writer.
 */
export const compareCodePoints = (a: string, b: string): number => {
  const ia = a[Symbol.iterator]();
  const ib = b[Symbol.iterator]();
  for (;;) {
    const na = ia.next();
    const nb = ib.next();
    if (na.done && nb.done) return 0;
    if (na.done) return -1;
    if (nb.done) return 1;
    const ca = na.value.codePointAt(0) as number;
    const cb = nb.value.codePointAt(0) as number;
    if (ca !== cb) return ca < cb ? -1 : 1;
  }
};

export const sortedByCodePoints = (keys: Iterable<string>): string[] =>
  [...keys].sort(compareCodePoints);

/**
 * Canonical form of a map keyed by name — a list of entries in code point order. Only
 * {@link canonicalJsonText} serializes it.
 *
 * MUST: the canonical order is not expressed by an object's enumeration order. JS objects enumerate
 * integer-like keys ("2" / "10") first in ascending order regardless of insertion order, so even when
 * assigned in sorted order `JSON.stringify` writes "2","10" (code point order is "10","2" — which
 * is what the Python writer writes).
 */
export class SortedEntries {
  readonly entries: readonly (readonly [string, unknown])[];
  constructor(entries: readonly (readonly [string, unknown])[]) {
    this.entries = entries;
  }

  /**
   * MUST: fails when passed to `JSON.stringify`. Spelled as is, it would yield a different JSON,
   * `{"entries":[...]}`, without an exception — only {@link canonicalJsonText} can spell the
   * canonical form.
   */
  toJSON(): never {
    throw new ContainerFormatError(
      "SortedEntries cannot be serialized with JSON.stringify (use canonicalJsonText)",
    );
  }
}

/**
 * An object's keys as a list of entries in code point order (values converted by `map` — identity
 * by default).
 *
 * MUST: the key `__proto__` is rejected on the writer side too (docs/container-v1.md §0). Writing a
 * name the reader rejects would make our own output unreadable by ourselves.
 */
export const sortedObject = <T>(
  source: Readonly<Record<string, T>>,
  map: (value: T, key: string) => unknown = (value) => value,
): SortedEntries =>
  new SortedEntries(
    sortedByCodePoints(Object.keys(source)).map((key) => {
      if (key === "__proto__") throw new ContainerFormatError("Key '__proto__' cannot be written");
      return [key, map(source[key], key)] as const;
    }),
  );

/**
 * Recursively canonicalizes a free-form JSON value (a node's `attrs` and the like) — objects get
 * their keys in code point order, arrays keep their declared order, scalars stay as they are.
 */
export const canonicalJsonValue = (value: unknown): unknown => {
  if (Array.isArray(value)) return value.map(canonicalJsonValue);
  if (isJsonObject(value)) return sortedObject(value, canonicalJsonValue);
  return value;
};

const fieldText = ([key, value]: readonly [string, unknown]): string =>
  `${JSON.stringify(key)}:${canonicalJsonText(value)}`;

/**
 * Canonical JSON serializer (no whitespace). {@link SortedEntries} is written in entry order, and
 * fixed-schema objects in the order their fields were assembled (fixed-schema field names do not look
 * like integers, so enumeration order = assembly order). It does not sort — the assembling side
 * decides the order (the same split as Python's `canonical_json`).
 *
 * Values that `JSON.stringify` rewrites silently (non-finite numbers → null, array holes and
 * undefined → null) fail loudly. Fixed-schema fields that are undefined are not written (as
 * `JSON.stringify` does — optional fields).
 */
export const canonicalJsonText = (value: unknown): string => {
  if (value instanceof SortedEntries) return `{${value.entries.map(fieldText).join(",")}}`;
  // Array.from passes holes as undefined (map skips holes, and join spells them as empty strings).
  if (Array.isArray(value)) return `[${Array.from(value, canonicalJsonText).join(",")}]`;
  if (isJsonObject(value)) {
    const fields = Object.entries(value).filter(([, field]) => field !== undefined);
    return `{${fields.map(fieldText).join(",")}}`;
  }
  if (typeof value === "number") {
    if (!Number.isFinite(value))
      throw new ContainerFormatError(`Non-finite number ${value} cannot be written`);
    return JSON.stringify(value);
  }
  if (value === null || typeof value === "string" || typeof value === "boolean") {
    return JSON.stringify(value);
  }
  throw new ContainerFormatError(`Value not representable as JSON: ${typeof value}`);
};

export const encodeJsonBytes = (value: unknown): Uint8Array<ArrayBuffer> =>
  new TextEncoder().encode(canonicalJsonText(value)) as Uint8Array<ArrayBuffer>;
