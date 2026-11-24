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

/** The 1-based line of a character offset of `text`. */
const lineAt = (text: string, offset: number): number => text.slice(0, offset).split("\n").length;

/**
 * The two edits of one file: an import after the last table import, the list extended in place.
 * `list` matches the list statement over one line or several (the formatter wraps a long one) and
 * captures its names; `render` writes the statement on one line for a list of names (the
 * formatter wraps it again).
 */
const fileEdits = (
  path: string,
  text: string,
  imports: RegExp,
  list: RegExp,
  render: (names: string) => string,
  importLine: string,
  constant: string,
): string[] => {
  const importAt = lastLineIndex(text.split("\n"), imports);
  const found = [...text.matchAll(list)].at(-1);
  if (importAt < 0 || found === undefined) {
    throw new RegistrationError(
      `${path}: cannot find ${importAt < 0 ? "the table imports" : "the list of tables"} (register the table by hand)`,
    );
  }
  const names = (found[1] ?? "")
    .split(",")
    .map((name) => name.trim())
    .filter((name) => name !== "");
  const first = lineAt(text, found.index);
  const last = lineAt(text, found.index + found[0].length);
  const lines = first === last ? `line ${first} becomes` : `lines ${first}–${last} become`;
  return [
    `  ${path}`,
    `    after line ${importAt + 1}, add: ${importLine}`,
    `    ${lines}: ${render([...names, constant].join(", "))}`,
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
      /\[([A-Z0-9_,\s]+)\]\.map\(\s*freezeGeometryProfile,?\s*\)/g,
      (names) => `[${names}].map(freezeGeometryProfile),`,
      `import { ${constant} } from "./${id}.ts";`,
      constant,
    ),
    ...fileEdits(
      PROFILE_TEST_PATH,
      testText,
      /^import \{ [A-Z0-9_]+ \} from "\.\.\/src\/kernels\/geometry-profiles\/[a-z0-9-]+\.ts";$/,
      /assertEquals\(BUILTIN_GEOMETRY_PROFILES, \[([A-Z0-9_,\s]+)\]\);/g,
      (names) => `assertEquals(BUILTIN_GEOMETRY_PROFILES, [${names}]);`,
      `import { ${constant} } from "../src/kernels/geometry-profiles/${id}.ts";`,
      constant,
    ),
  ];
};
