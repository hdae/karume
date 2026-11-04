/**
 * Test surface that connects the **container files** of a series output / distribution mirror
 * (the part sequence of a `krm`) to the runtime's container surface (`openContainer`). All the
 * tests own is "how to find the files" and "how to read a range".
 *
 * The **mirror image** of the Python side's `karume.container.resolve_sequence` /
 * `container_parts` (tools/exporter/src/karume/container.py). MUST: the numbering spelling is not
 * grown separately in two places — when the writer and the reader disagree on the rule, reading the
 * leftovers of a previous write-out against this run's expected values passes silently.
 *
 * ## Part numbering (file numbers and `BlockSource` indices are off by one)
 *
 * File-name numbering **starts at 1**: `model-00001-of-000NN.krm` is **part 0** of the
 * `BlockSource` (`[header][graph descriptor][model descriptor]`). After it, `-00002-` is part 1
 * (the const region) and `-00003-` onward are parts 2.. (weights / assets). {@link fileBlockSource}
 * maps `parts[i]` to part `i` as is, so the **index order {@link resolveParts} returns MUST NOT be
 * disturbed**.
 *
 * ## How much is read
 *
 * Each `read` takes only the range with `Deno.open` + `seek` (a file is never loaded whole into
 * RAM). `verified: false` is a MUST — nobody has verified the local files, so `openContainer`
 * applies the per-block sha256 (claiming true would let tampering through — container-v1 §7).
 */

import {
  type DirEntry,
  isNotFound,
  readDirSync,
  readRange,
  statSync,
} from "@karume/test-support/fs";
import {
  type BlockSource,
  type DescriptorExpectation,
  openContainer,
  type OpenedContainer,
} from "../../mod.ts";

/** Digits of the numbering (`-NNNNN-of-NNNNN`) — equal to the Python side's `SEQUENCE_DIGITS`. */
const INDEX_DIGITS = 5;

const escapeRegExp = (text: string): string => text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/**
 * A regular expression that matches the part file names of the same container as this
 * representative path.
 *
 * MUST: escape the stem / suffix — real paths contain dots and hyphens (`model.i8.krm`), so
 * inserting them raw would pick up unrelated files (the same reason glob is not used).
 */
const sequencePattern = (name: string): RegExp => {
  const dot = name.lastIndexOf(".");
  const stem = dot <= 0 ? name : name.slice(0, dot);
  const suffix = dot <= 0 ? "" : name.slice(dot);
  return new RegExp(
    `^${escapeRegExp(stem)}-(\\d{${INDEX_DIGITS}})-of-(\\d{${INDEX_DIGITS}})${escapeRegExp(
      suffix,
    )}$`,
  );
};

/** The last segment of a URL (the real file name with `%` escapes decoded). */
const baseName = (url: URL): string => decodeURIComponent(url.pathname).split("/").pop() ?? "";

/**
 * Whether a file exists.
 * MUST: propagate everything but NotFound — rounding every I/O error to "no assets" would turn a
 * mount failure of the asset root into a SKIP, and checks that never ran would go green quietly.
 */
const isFile = (url: URL): boolean => {
  try {
    return statSync(url).isFile;
  } catch (cause) {
    if (isNotFound(cause)) return false;
    throw cause;
  }
};

/** Lists a directory (missing = empty — failures inside it propagate). */
const listDir = (url: URL): readonly DirEntry[] => {
  try {
    return readDirSync(url);
  } catch (cause) {
    if (isNotFound(cause)) return [];
    throw cause;
  }
};

/**
 * A container's **representative path** (`…/model.krm` / `…/model.i8.krm`) → the part files that
 * exist (index order = the `BlockSource` part order). The single form and a container that does
 * not exist both return `[representative]` (the caller's existing checks own the diagnosis of a
 * missing container — anticipating it here would split the wording in two).
 *
 * MUST: an ambiguous directory fails loudly. The single form next to numbered parts, disagreeing
 * `of` totals, and missing or out-of-range numbers all leave "which bytes to read" undetermined.
 * Silently picking one would compare a previous write-out against this run's expected values.
 */
export const resolveParts = (representative: URL): readonly URL[] => {
  const parent = new URL("./", representative);
  const pattern = sequencePattern(baseName(representative));
  const found = new Map<number, URL>();
  const totals = new Set<number>();
  for (const entry of listDir(parent)) {
    if (!entry.isFile) continue;
    const match = pattern.exec(entry.name);
    if (match === null) continue;
    found.set(Number(match[1]), new URL(encodeURIComponent(entry.name), parent));
    totals.add(Number(match[2]));
  }
  if (found.size === 0) return [representative];
  const where = decodeURIComponent(representative.pathname);
  if (isFile(representative)) {
    throw new Error(
      `${where}: the single form and numbered parts (${found.size} files) sit together` +
        " — delete what an earlier write left behind before reading",
    );
  }
  if (totals.size !== 1) {
    throw new Error(
      `${where}: the totals of the numbered parts disagree: ${[...totals].sort((a, b) => a - b)}`,
    );
  }
  const [total] = totals;
  const files: URL[] = [];
  for (let index = 1; index <= total; index += 1) {
    const file = found.get(index);
    if (file === undefined) {
      throw new Error(`${where}: numbered parts 1..${total}: part ${index} is missing`);
    }
    files.push(file);
  }
  if (found.size !== total) {
    // MUST: count numbers out of range both below and above. Looking only above would, in a
    // directory with a number of 0 or less such as `model-00000-of-00003.krm`, fail with an empty
    // list ("out-of-range numbers []") and an unreadable diagnosis (that it fails at all is
    // guaranteed by the count mismatch).
    const surplus = [...found.keys()]
      .filter((index) => index < 1 || index > total)
      .sort((a, b) => a - b);
    throw new Error(`${where}: numbered parts 1..${total}: out-of-range numbers ${surplus}`);
  }
  return files;
};

/**
 * Whether the container exists (single form or numbered parts). Replaces the places where the
 * checks that SKIP on missing assets looked directly at "is there a `model.krm`".
 *
 * MUST: synchronous — callers read it in a module-level `const ANY_PRESENT = …`.
 */
export const modelPresent = (representative: URL): boolean =>
  isFile(resolveParts(representative)[0]);

/**
 * Part files → `BlockSource` (range reads). `parts[i]` is part `i`.
 *
 * The part lengths are `stat`ed once here and kept (`BlockSource.partLength` is a synchronous
 * surface that `openContainer` uses to compare against the declared lengths).
 */
export const fileBlockSource = (parts: readonly URL[]): BlockSource => {
  const lengths = parts.map((part) => statSync(part).size);
  const at = (index: number): { readonly url: URL; readonly length: number } => {
    const url = parts[index];
    const length = lengths[index];
    if (url === undefined || length === undefined) {
      throw new Error(
        `test: part ${index} does not exist (the part list has ${parts.length} parts)`,
      );
    }
    return { url, length };
  };
  return {
    partCount: parts.length,
    // MUST: nobody has verified the local files (container-v1 §7).
    verified: false,
    partLength: (index) => at(index).length,
    read: async (part, offset, length) => {
      const { url, length: partBytes } = at(part);
      if (offset < 0 || length < 0 || offset + length > partBytes) {
        throw new Error(
          `test: ${url.pathname}: range [${offset}, ${
            offset + length
          }) is outside the part length ${partBytes}`,
        );
      }
      return readRange(url, offset, length);
    },
  };
};

/**
 * Opens the container of a series output / mirror from its representative path (resolve the part
 * sequence → a range-reading `BlockSource` → `openContainer`).
 *
 * `expect` is the expected value of the two documents held outside (the manifest's
 * `container.descriptor`). A series output has no manifest, so it is omitted — then nobody
 * guarantees the integrity of the two documents, but the per-block sha256 still applies through
 * the `verified: false` path.
 */
export const openSeriesContainer = async (
  representative: URL,
  expect?: DescriptorExpectation,
): Promise<OpenedContainer> =>
  // MUST: make it `async` — resolving the part sequence and `stat` fail synchronously, so as a bare
  // expression only missing numbers and mount failures would come back as a **synchronous
  // throw**. The surface makes them arrive the same way whether the caller uses `await` or
  // `.catch`.
  await openContainer(
    { kind: "source", source: fileBlockSource(resolveParts(representative)) },
    expect,
  );
