/**
 * The printer: a profile as a one-line JavaScript literal and the module that exports it. Passed
 * through the repository's formatter, the module is the table file's text from the `import type`
 * line on, so one printer serves the generated files and the check that the registered files are
 * still what the generator writes.
 */

import type { GeometryProfile } from "@karume/runtime";

/** A value the printer has no spelling for (a profile holds none: the injection check rejects them). */
export class PrinterError extends Error {
  override name = "PrinterError";
}

const IDENTIFIER = /^[A-Za-z_$][\w$]*$/;

const key = (name: string): string => (IDENTIFIER.test(name) ? name : JSON.stringify(name));

/**
 * One value as a literal on one line. Infinity, the last rule's `maxRows`, has no literal of its
 * own and is spelled `Number.POSITIVE_INFINITY` as the hand-checked tables spell it.
 */
export const literal = (value: unknown): string => {
  if (typeof value === "string") return JSON.stringify(value);
  if (typeof value === "boolean") return String(value);
  if (typeof value === "number") {
    if (value === Number.POSITIVE_INFINITY) return "Number.POSITIVE_INFINITY";
    if (!Number.isFinite(value)) throw new PrinterError(`No literal for the number ${value}`);
    return String(value);
  }
  if (Array.isArray(value)) return `[${value.map(literal).join(", ")}]`;
  if (typeof value === "object" && value !== null) {
    const entries = Object.entries(value).map(([name, entry]) => {
      if (entry === undefined) throw new PrinterError(`No literal for undefined (${name})`);
      return `${key(name)}: ${literal(entry)}`;
    });
    return entries.length === 0 ? "{}" : `{ ${entries.join(", ")} }`;
  }
  throw new PrinterError(`No literal for ${String(value)}`);
};

/** The exported constant's name: the id in upper snake case (`nvidia-ampere` → `NVIDIA_AMPERE`). */
export const constantName = (id: string): string => id.toUpperCase().replaceAll("-", "_");

/** The table module without its header, before formatting. */
export const profileModule = (profile: GeometryProfile): string =>
  `import type { GeometryProfile } from "../geometry-profile.ts";\n\nexport const ${constantName(
    profile.id,
  )}: GeometryProfile = ${literal(profile)};\n`;
