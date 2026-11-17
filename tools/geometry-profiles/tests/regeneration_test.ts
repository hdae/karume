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
 * apple-metal-3 has no such case: every record it cites has the format `karume-geometry-sweep/2`.
 * This ends when that table is regenerated from a new sweep. No GPU.
 */

import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { assert, assertEquals, assertThrows } from "@std/assert";
import { BUILTIN_GEOMETRY_PROFILES } from "@karume/runtime";
import { describe, it } from "vite-plus/test";
import { REPOSITORY_ROOT } from "../format.ts";
import { DEFAULT_MIN_SPEEDUP, deriveTable } from "../generate.ts";
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
