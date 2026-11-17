/**
 * The path of an input file as an entry names it (relative to the repository's `inputs/`), checked
 * the same way by both hosts' `openInput` before either touches the file system or the network.
 *
 * SPEC: docs/spec/workbench.md#input-files
 */

/**
 * The path's segments. Refused: an empty path or segment, `.` and `..`, a backslash and NUL — a
 * path that could leave `inputs/` or name a file ambiguously.
 */
export const inputSegments = (path: string): readonly string[] => {
  const segments = path.split("/");
  const bad = segments.find(
    (segment) =>
      segment === "" ||
      segment === "." ||
      segment === ".." ||
      segment.includes("\\") ||
      segment.includes("\0"),
  );
  if (bad !== undefined) {
    throw new Error(
      `input file '${path}': not a path inside inputs/ (relative, '/'-separated, no empty, '.' or '..' segment)`,
    );
  }
  return segments;
};
