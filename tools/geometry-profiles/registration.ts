/**
 * The two edits that register a new table, worked out from the files as they stand (the command
 * prints them and does not make them: registering a table puts it into automatic selection for every
 * adapter it matches, which is the maintainer's step).
 *
 * - `packages/runtime/src/kernels/geometry-profiles/index.ts`: import the table's constant and add it
 *   to the array of `BUILTIN_GEOMETRY_PROFILES`.
 * - `packages/runtime/tests/geometry_profile_test.ts`: import it and add it to the list the test
 *   compares `BUILTIN_GEOMETRY_PROFILES` with.
 */

import { constantName } from "./literal.ts";

export const INDEX_PATH = "packages/runtime/src/kernels/geometry-profiles/index.ts";
export const PROFILE_TEST_PATH = "packages/runtime/tests/geometry_profile_test.ts";

/** A place an edit goes is not in the file (the file changed shape; edit by hand). */
export class RegistrationError extends Error {
  override name = "RegistrationError";
}

const lastLineIndex = (lines: readonly string[], pattern: RegExp): number =>
  lines.findLastIndex((line) => pattern.test(line));

/** The two edits of one file: an import after the last table import, the list extended in place. */
const fileEdits = (
  path: string,
  text: string,
  imports: RegExp,
  list: RegExp,
  importLine: string,
  constant: string,
): string[] => {
  const lines = text.split("\n");
  const importAt = lastLineIndex(lines, imports);
  const listAt = lastLineIndex(lines, list);
  if (importAt < 0 || listAt < 0) {
    throw new RegistrationError(
      `${path}: cannot find ${importAt < 0 ? "the table imports" : "the list of tables"} (register the table by hand)`,
    );
  }
  const extended = lines[listAt].replace(list, (whole, items: string) =>
    whole.replace(`[${items}]`, `[${items}, ${constant}]`),
  );
  return [
    `  ${path}`,
    `    after line ${importAt + 1}, add: ${importLine}`,
    `    line ${listAt + 1} becomes: ${extended.trim()}`,
  ];
};

/** The edits as printable lines, for the table `id` (whose constant is derived from it). */
export const registrationEdits = (id: string, indexText: string, testText: string): string[] => {
  const constant = constantName(id);
  return [
    "Registration (not made by this command; line numbers as the files stand now):",
    ...fileEdits(
      INDEX_PATH,
      indexText,
      /^import \{ [A-Z0-9_]+ \} from "\.\/[a-z0-9-]+\.ts";$/,
      /\[([A-Z0-9_, ]+)\]\.map\(freezeGeometryProfile\)/,
      `import { ${constant} } from "./${id}.ts";`,
      constant,
    ),
    ...fileEdits(
      PROFILE_TEST_PATH,
      testText,
      /^import \{ [A-Z0-9_]+ \} from "\.\.\/src\/kernels\/geometry-profiles\/[a-z0-9-]+\.ts";$/,
      /assertEquals\(BUILTIN_GEOMETRY_PROFILES, \[([A-Z0-9_, ]+)\]\);/,
      `import { ${constant} } from "../src/kernels/geometry-profiles/${id}.ts";`,
      constant,
    ),
  ];
};
