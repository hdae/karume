/**
 * The printer through the repository's formatter reproduces every registered table from its
 * `import type` line on: the generator writes the same bytes as the tables in the repository for
 * an unchanged profile (the header comments above that line are older than the tool and are
 * replaced when a table is regenerated). No GPU and no records.
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { assert, assertEquals, assertThrows } from "@std/assert";
import { BUILTIN_GEOMETRY_PROFILES } from "@karume/runtime";
import { describe, it } from "vite-plus/test";
import { formatTypeScript, REPOSITORY_ROOT } from "../format.ts";
import { constantName, literal, PrinterError, profileModule } from "../literal.ts";

const PROFILES = "packages/runtime/src/kernels/geometry-profiles";

describe("the printer through the formatter", () => {
  for (const profile of BUILTIN_GEOMETRY_PROFILES) {
    it(`reproduces ${profile.id}.ts from its import line on`, () => {
      const path = `${PROFILES}/${profile.id}.ts`;
      const file = readFileSync(join(REPOSITORY_ROOT, path), "utf8");
      const body = file.slice(file.indexOf("import type {"));
      const printed = formatTypeScript(profileModule(profile), path);
      assertEquals(printed, body);
      assertEquals(
        body.includes(`export const ${constantName(profile.id)}: GeometryProfile`),
        true,
      );
    });
  }
});

describe("literal", () => {
  it("spells Infinity as Number.POSITIVE_INFINITY and quotes only keys that are not identifiers", () => {
    assertEquals(
      literal({ maxRows: Number.POSITIVE_INFINITY, "i8a8-linear": ["a"], ok: true }),
      '{ maxRows: Number.POSITIVE_INFINITY, "i8a8-linear": ["a"], ok: true }',
    );
  });

  it("refuses values a profile cannot hold", () => {
    assertThrows(() => literal({ a: undefined }), PrinterError);
    assertThrows(() => literal(Number.NaN), PrinterError);
    assertThrows(() => literal(null), PrinterError);
  });
});

describe("the printer and the GEMV choices of a table (linearGemv)", () => {
  it("prints the field of a table that carries it, and nothing for a table without it (every registered table)", () => {
    for (const profile of BUILTIN_GEOMETRY_PROFILES) {
      assertEquals(
        profileModule(profile).includes("linearGemv"),
        profile.linearGemv !== undefined,
        profile.id,
      );
    }
    // The three device tables with GEMV records carry the field, apple-m5 (no GEMV records yet)
    // and the shared table (tile geometries only) do not; one taken out of a table prints nothing of
    // it
    assertEquals(
      BUILTIN_GEOMETRY_PROFILES.map(({ id, linearGemv }) => [id, linearGemv !== undefined]),
      [
        ["apple-metal-3", true],
        ["nvidia-ampere", true],
        ["nvidia-blackwell", true],
        ["apple-m5", false],
        ["apple-metal-3-shared", false],
      ],
    );
    const registered = BUILTIN_GEOMETRY_PROFILES[0];
    assert(registered !== undefined);
    const { linearGemv: _own, ...table } = registered;
    assertEquals(profileModule(table).includes("linearGemv"), false);
    const linearGemv = {
      sequential: { variant: { cols: 32, unroll: 4 }, shapes: [] },
      rowsThreadTarget: 16384,
      parallelShapes: [
        { storage: "i4", n: 64, k: 1536, group: 32, lanes: 8, packedActivations: false },
      ],
    } as const;
    assertEquals(
      profileModule({ ...table, linearGemv }).includes(
        'linearGemv: { sequential: { variant: { cols: 32, unroll: 4 }, shapes: [] }, rowsThreadTarget: 16384, parallelShapes: [{ storage: "i4", n: 64, k: 1536, group: 32, lanes: 8, packedActivations: false }] }',
      ),
      true,
    );
  });
});
