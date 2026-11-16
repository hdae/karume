/**
 * The generator's derivation reproduces the registered tables, provenance included: for
 * nvidia-ampere and nvidia-blackwell, the records the table's `i8a8-linear` provenance cites
 * (format `karume-geometry-sweep/3`, the only format the derivation accepts), with the registered
 * table as base, derive an object deep-equal to the table. The records live outside git (under
 * `outputs/`); on a machine without them the case is skipped explicitly with one console line
 * naming the missing files. Each file's sha256 is checked against the one the table cites.
 *
 * apple-metal-3 has no such case: every record it cites has the old format
 * (`karume-geometry-sweep/2`), which the derivation refuses. This ends when that table is
 * regenerated from a new sweep. No GPU.
 */

import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { assert, assertEquals } from "@std/assert";
import { BUILTIN_GEOMETRY_PROFILES } from "@karume/runtime";
import { describe, it } from "vite-plus/test";
import { REPOSITORY_ROOT } from "../format.ts";
import { DEFAULT_MIN_SPEEDUP, deriveTable } from "../generate.ts";
import { loadRecord } from "../records.ts";

const cited = (id: string): { paths: string[]; sha256: string[] } => {
  const entry = BUILTIN_GEOMETRY_PROFILES.find((profile) => profile.id === id)?.provenance
    ?.families["i8a8-linear"];
  if (entry === undefined) throw new Error(`${id} has no i8a8-linear provenance`);
  return { paths: entry.sweep.split(", "), sha256: entry.sha256.split(", ") };
};

describe("regenerating a registered table from the records its i8a8-linear family cites", () => {
  for (const id of ["nvidia-ampere", "nvidia-blackwell"]) {
    const { paths, sha256 } = cited(id);
    const missing = paths.filter((path) => !existsSync(join(REPOSITORY_ROOT, path)));
    if (missing.length > 0) {
      console.log(`skipped: regenerating ${id} needs local records: ${missing.join(", ")}`);
    }
    it.skipIf(missing.length > 0)(`gives ${id} back, provenance included`, () => {
      const base = BUILTIN_GEOMETRY_PROFILES.find((profile) => profile.id === id);
      assert(base !== undefined);
      const records = paths.map((path) =>
        loadRecord(readFileSync(join(REPOSITORY_ROOT, path)), path, "records"),
      );
      assertEquals(
        records.map((record) => record.sha256),
        sha256,
      );
      const { profile } = deriveTable({
        id,
        minSpeedup: DEFAULT_MIN_SPEEDUP,
        records,
        attentionBench: [],
        base,
      });
      assertEquals(profile, base);
    });
  }
});
