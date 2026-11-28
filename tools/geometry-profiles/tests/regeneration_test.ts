/**
 * The generator's derivation reproduces the registered tables, provenance included, from the records
 * they cite: for nvidia-ampere and nvidia-blackwell, the records the table's `i8a8-linear`
 * provenance cites, with the registered table as base, derive an object deep-equal to the table —
 * when they are of the format the derivation reads (`karume-geometry-sweep/4`). Records of an older
 * format are refused by it, naming the format, and nothing converts them: such a table regenerates
 * from records again once that family is swept again. The records cited today are
 * `karume-geometry-sweep/3` (before the quick look), so on a machine that has them the case checks the
 * refusal. The records live outside git (under `outputs/`); on a machine without them the case is
 * skipped explicitly with one console line naming the missing files. Each file's sha256 is checked
 * against the one the table cites.
 *
 * apple-metal-3 has no such case: every record its tile geometry's provenance cites has the format
 * `karume-geometry-sweep/2` (its GEMV choices come from GEMV sweep records, which this case does not
 * read). This ends when its tile geometry is swept again.
 *
 * apple-m5 and the shared table apple-metal-3-shared come from records of the current format only, so
 * every record any of their families cites, with the registered table as base (a shared derivation
 * for the shared table), and the GEMV sweep records the header cites (apple-m5's), whose `linearGemv`
 * is derived again rather than carried, derive an object deep-equal to the table. The GEMV
 * derivation is given the table as the generator gives it, its own choices in place: the rule of one
 * steady record keeps a choice the table holds unless a challenger beats it, so the table's choices
 * are an input of the rule, and a derivation from the default's choices alone is not what the
 * generator writes. No GPU.
 */

import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { assert, assertEquals, assertThrows } from "@std/assert";
import { BUILTIN_GEOMETRY_PROFILES } from "@karume/runtime";
import { describe, it } from "vite-plus/test";
import { REPOSITORY_ROOT } from "../format.ts";
import {
  DEFAULT_MIN_SPEEDUP,
  deriveGemvTable,
  deriveTable,
  PROFILES_DIRECTORY,
} from "../generate.ts";
import { loadRecord } from "../records.ts";

/** The geometry sweep's format the derivation reads. */
const CURRENT_FORMAT = "karume-geometry-sweep/4";

const cited = (id: string): { paths: string[]; sha256: string[] } => {
  const entry = BUILTIN_GEOMETRY_PROFILES.find((profile) => profile.id === id)?.provenance
    ?.families["i8a8-linear"];
  if (entry === undefined) throw new Error(`${id} has no i8a8-linear provenance`);
  return { paths: entry.sweep.split(", "), sha256: entry.sha256.split(", ") };
};

const formatOf = (report: unknown): unknown =>
  typeof report === "object" && report !== null && "format" in report ? report.format : undefined;

describe("regenerating a registered table from the records its i8a8-linear family cites", () => {
  for (const id of ["nvidia-ampere", "nvidia-blackwell"]) {
    const { paths, sha256 } = cited(id);
    const missing = paths.filter((path) => !existsSync(join(REPOSITORY_ROOT, path)));
    if (missing.length > 0) {
      console.log(`skipped: regenerating ${id} needs local records: ${missing.join(", ")}`);
    }
    it.skipIf(missing.length > 0)(
      `gives ${id} back, provenance included, or refuses its records by their older format`,
      () => {
        const base = BUILTIN_GEOMETRY_PROFILES.find((profile) => profile.id === id);
        assert(base !== undefined);
        const records = paths.map((path) =>
          loadRecord(readFileSync(join(REPOSITORY_ROOT, path)), path, "records"),
        );
        assertEquals(
          records.map((record) => record.sha256),
          sha256,
        );
        const derive = () =>
          deriveTable({
            id,
            minSpeedup: DEFAULT_MIN_SPEEDUP,
            records,
            attentionBench: [],
            base,
          });
        const older = records.find(({ report }) => formatOf(report) !== CURRENT_FORMAT);
        if (older === undefined) {
          assertEquals(derive().profile, base);
          return;
        }
        assertThrows(derive, Error, `${older.path}: format is not ${CURRENT_FORMAT}`);
      },
    );
  }
});

/**
 * Every record any family of `id` cites, once, in an order that keeps every family's own order. A
 * family's entry lists the records with a row of it in the order the derivation was given them,
 * so records given in such an order give every entry back; the order of first appearance does not
 * when two families cite records the other does not between ones they share. Ties go by first
 * appearance.
 */
const citedByEveryFamily = (id: string): { paths: string[]; sha256: string[] } => {
  const families = BUILTIN_GEOMETRY_PROFILES.find((profile) => profile.id === id)?.provenance
    ?.families;
  if (families === undefined) throw new Error(`${id} has no provenance`);
  const cited = new Map<string, { sha256: string; before: Set<string> }>();
  for (const entry of Object.values(families)) {
    const shas = entry.sha256.split(", ");
    const paths = entry.sweep.split(", ");
    paths.forEach((path, index) => {
      const known = cited.get(path) ?? { sha256: shas[index] ?? "", before: new Set<string>() };
      const previous = paths[index - 1];
      if (previous !== undefined) known.before.add(previous);
      cited.set(path, known);
    });
  }
  const ordered: string[] = [];
  while (ordered.length < cited.size) {
    const next = [...cited.entries()].find(
      ([path, { before }]) =>
        !ordered.includes(path) && [...before].every((earlier) => ordered.includes(earlier)),
    );
    if (next === undefined)
      throw new Error(`${id}: its families cite records in orders that conflict`);
    ordered.push(next[0]);
  }
  return { paths: ordered, sha256: ordered.map((path) => cited.get(path)?.sha256 ?? "") };
};

/**
 * The GEMV sweep records `id`'s header cites: the records it lists that its provenance does not
 * (the profile's type holds no entry for `linearGemv`, so its records are named in the header
 * alone). Empty for a table whose header lists none.
 */
const gemvCitedByHeader = (id: string): { paths: string[]; sha256: string[] } => {
  const text = readFileSync(join(REPOSITORY_ROOT, PROFILES_DIRECTORY, `${id}.ts`), "utf8");
  const header = text.slice(0, text.indexOf("import type {"));
  const inProvenance = new Set(citedByEveryFamily(id).paths);
  const listed = [...header.matchAll(/^ \* - (\S+) \(sha256 ([0-9a-f]{64}), /gm)].flatMap(
    ([, path, sha]) =>
      path === undefined || sha === undefined || inProvenance.has(path) ? [] : [{ path, sha }],
  );
  return { paths: listed.map(({ path }) => path), sha256: listed.map(({ sha }) => sha) };
};

describe("regenerating a table whose every family comes from records of the current format", () => {
  for (const { id, shared } of [
    { id: "apple-m5", shared: false },
    { id: "apple-metal-3-shared", shared: true },
  ]) {
    const { paths, sha256 } = citedByEveryFamily(id);
    const gemv = gemvCitedByHeader(id);
    const missing = [...paths, ...gemv.paths].filter(
      (path) => !existsSync(join(REPOSITORY_ROOT, path)),
    );
    if (missing.length > 0) {
      console.log(`skipped: regenerating ${id} needs local records: ${missing.join(", ")}`);
    }
    it.skipIf(missing.length > 0)(
      `gives ${id} back, provenance included, from every record its families and its header cite`,
      () => {
        const base = BUILTIN_GEOMETRY_PROFILES.find((profile) => profile.id === id);
        assert(base !== undefined);
        const records = paths.map((path) =>
          loadRecord(readFileSync(join(REPOSITORY_ROOT, path)), path, "records"),
        );
        assertEquals(
          records.map((record) => record.sha256),
          sha256,
        );
        // A table with GEMV choices names their records in its header; one without names none
        assertEquals(gemv.paths.length > 0, base.linearGemv !== undefined, gemv.paths.join(", "));
        const gemvRecords = gemv.paths.map((path) =>
          loadRecord(readFileSync(join(REPOSITORY_ROOT, path)), path, "gemv-records"),
        );
        assertEquals(
          gemvRecords.map((record) => record.sha256),
          gemv.sha256,
        );
        const tiled = deriveTable({
          id,
          minSpeedup: DEFAULT_MIN_SPEEDUP,
          records,
          attentionBench: [],
          base,
          ...(shared ? { shared: true } : {}),
        }).profile;
        // The GEMV choices are derived again from their records, not carried from the base: every
        // decision is taken again, from the table's own choices as the generator starts from them
        assertEquals(tiled.linearGemv, base.linearGemv);
        const derived =
          gemvRecords.length === 0 ? tiled : deriveGemvTable(tiled, gemvRecords).profile;
        assertEquals(derived, base);
      },
    );
  }
});
