/**
 * Unpacks the files the benchmark page leaves into the workbench's records, where and as the
 * workbench would have written them, for the table generator (`tools/geometry-profiles`):
 *
 *     node apps/benchmark/unpack.ts <bundle.json>…
 *
 * Every record of every file goes to `outputs/workbench/<host>-<adapter name>/<entry id>_<UTC
 * time>.json` (the time is the record's `recordedAt`) with the text the shared record code gives
 * it. Nothing is ever overwritten: when any of those files exists already (or two records would take
 * one name), nothing is written. Beside a GEMV record it writes, it says the record's class (one
 * steady record is enough for a table, an unsteady one needs a second); beside a grid record, that
 * the generator's `--grid-records` reads it, or refuses its timing. Exit 0 when every record
 * was written, 1 for a file that cannot be
 * read or is no bundle and for a file that exists, 2 for a command line without a file.
 *
 * SPEC: docs/spec/benchmark.md#unpacking-the-file
 */

import { access, readFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { writeNewFile } from "@karume/sweep-app-support/node";

import { type Bundle, bundleRecordFiles, parseBundle, type RecordFile } from "./src/bundle.ts";
import { gemvClassText, gridRecordText } from "./src/notes.ts";

const REPOSITORY_ROOT = fileURLToPath(new URL("../../", import.meta.url));

/** What a command made of its files: the lines to print and its exit code. */
export type UnpackResult = { readonly code: 0 | 1 | 2; readonly lines: readonly string[] };

const exists = async (path: string): Promise<boolean> => {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
};

/** Why a bundle stopped short, in a few words. */
const stopText = (bundle: Bundle): string => {
  const { stop } = bundle;
  switch (stop.kind) {
    case "completed":
      return "it stopped short";
    case "cancelled":
      return "it was cancelled";
    case "deviceLost":
      return `the GPU device was lost (${stop.reason})`;
    case "error":
      return `it stopped on an error: ${stop.message}`;
  }
};

/**
 * Unpacks `files` (paths as given, read from the working directory) under `repositoryRoot`: every
 * file is read and checked, and every target checked, before anything is written.
 */
export const unpackBundles = async (
  repositoryRoot: string,
  files: readonly string[],
): Promise<UnpackResult> => {
  if (files.length === 0) {
    return { code: 2, lines: ["Name the files: node apps/benchmark/unpack.ts <bundle.json>…"] };
  }
  const read: { readonly file: string; readonly bundle: Bundle; readonly records: RecordFile[] }[] =
    [];
  for (const file of files) {
    let bundle: Bundle;
    try {
      bundle = parseBundle(await readFile(file, "utf8"));
    } catch (error) {
      return {
        code: 1,
        lines: [`${file}: ${error instanceof Error ? error.message : String(error)}`],
      };
    }
    read.push({ file, bundle, records: bundleRecordFiles(bundle) });
  }
  const targets = read.flatMap(({ records }) => records.map(({ path }) => path));
  const twice = targets.filter((path, index) => targets.indexOf(path) !== index);
  const taken: string[] = [];
  for (const path of targets) if (await exists(join(repositoryRoot, path))) taken.push(path);
  if (twice.length > 0 || taken.length > 0) {
    return {
      code: 1,
      lines: [
        ...twice.map(
          (path) => `${path}: two records would take this name (the same file given twice?)`,
        ),
        ...taken.map((path) => `${path}: exists already`),
        "Nothing was written.",
      ],
    };
  }
  const lines: string[] = [];
  for (const { file, bundle, records } of read) {
    if (!bundle.complete) {
      lines.push(`${file}: incomplete — ${stopText(bundle)}; its records hold what was measured`);
    }
    for (const [index, { path, text }] of records.entries()) {
      if (!(await writeNewFile(repositoryRoot, path, new TextEncoder().encode(text)))) {
        return { code: 1, lines: [...lines, `${path}: exists already (written meanwhile)`] };
      }
      // A GEMV record says whether one record is enough (its class); a grid record, what reads it
      const record = bundle.records[index];
      const gemv = record === undefined ? undefined : gemvClassText(record);
      const grid = record === undefined ? undefined : gridRecordText(record);
      lines.push(
        `wrote ${path}${gemv === undefined ? "" : ` (GEMV ${gemv})`}${grid === undefined ? "" : ` (grid set: ${grid})`}`,
      );
    }
  }
  return { code: 0, lines };
};

if (import.meta.main) {
  const result = await unpackBundles(REPOSITORY_ROOT, process.argv.slice(2));
  const stream = result.code === 0 ? process.stdout : process.stderr;
  stream.write(`${result.lines.join("\n")}\n`);
  process.exitCode = result.code;
}
