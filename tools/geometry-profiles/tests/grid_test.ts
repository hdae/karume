/**
 * The grid-fill rule of the GEMM in the command (`--grid-records`): a dry run prints the
 * derivation's verdicts and the field difference and writes nothing; a written table carries the
 * derived field, and its header names the grid records and the rule; a table that carries the field
 * keeps it through a regeneration of the tile geometry alone; the command line refuses grid records
 * beside `--shared` and `--gemv-plan`. Each case runs on a temporary repository whose registered
 * table is built here from synthetic records of one adapter (the grid records from
 * packages/runtime/tests/helpers/grid-records.ts). The formatter is the real one. No GPU.
 */

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { assert, assertEquals, assertStringIncludes } from "@std/assert";
import { BUILTIN_GEOMETRY_PROFILES, type GeometryProfile } from "@karume/runtime";
import { NVIDIA_AMPERE } from "../../../packages/runtime/src/kernels/geometry-profiles/nvidia-ampere.ts";
import { deriveGemmGrid, deriveGeometryProfile } from "@karume/runtime/tune";
import { makeTempDir, remove } from "@karume/test-support/fs";
import { afterEach, beforeEach, describe, it } from "vite-plus/test";
import { gridRecord } from "../../../packages/runtime/tests/helpers/grid-records.ts";
import {
  familyDefaultRows,
  report,
} from "../../../packages/runtime/tests/helpers/sweep-records.ts";
import { type CommandEnvironment, main, PROFILES_DIRECTORY } from "../generate.ts";
import { ADAPTER, bytesOf, workbenchRecord } from "./helpers/records.ts";

const GRID_A = "outputs/workbench/node-test-gpu/geometry-sweep_grid-a.json";
const GRID_B = "outputs/workbench/node-test-gpu/geometry-sweep_grid-b.json";
const SWEEP = "outputs/workbench/node-test-gpu/geometry-sweep_gemm.json";
const BASE_COMMIT = "fedcba9876543210fedcba9876543210fedcba98";
const TABLE = `${PROFILES_DIRECTORY}/test-gpu.ts`;
const NVIDIA_AMPERE_ROWS = NVIDIA_AMPERE.gemmRows;

let root = "";
let stdout: string[] = [];
let stderr: string[] = [];

/**
 * The registered table: a first table of the adapter from a gemm record that adopts nothing, with
 * `nvidia-ampere`'s M buckets (the recon's grid times are ratios against those tiles).
 */
const registeredTable = (): GeometryProfile => ({
  ...deriveGeometryProfile(
    [
      {
        report: report(familyDefaultRows("gemm"), { ...ADAPTER }),
        path: "outputs/old.json",
        sha256: "c0ffee",
      },
    ],
    { id: "test-gpu", vendor: ADAPTER.vendor, architecture: ADAPTER.architecture },
  ).profile,
  gemmRows: NVIDIA_AMPERE_ROWS,
});

const environment = (registered: GeometryProfile): CommandEnvironment => ({
  repositoryRoot: root,
  registered: [...BUILTIN_GEOMETRY_PROFILES, registered],
  lastCommit: () => BASE_COMMIT,
  stdout: (text) => stdout.push(text),
  stderr: (text) => stderr.push(text),
});

const write = (path: string, bytes: Uint8Array): void => {
  mkdirSync(join(root, path, ".."), { recursive: true });
  writeFileSync(join(root, path), bytes);
};

const gridReport = (date: string): Record<string, unknown> =>
  gridRecord({ adapter: { ...ADAPTER }, date });

beforeEach(async () => {
  root = await makeTempDir({ prefix: "karume-geometry-profiles-grid-" });
  stdout = [];
  stderr = [];
  mkdirSync(join(root, PROFILES_DIRECTORY), { recursive: true });
  write(GRID_A, bytesOf(workbenchRecord("geometry-sweep", gridReport("2026-11-28T05:00:00.000Z"))));
  write(GRID_B, bytesOf(gridReport("2026-11-28T06:00:00.000Z")));
  write(SWEEP, bytesOf(report(familyDefaultRows("gemm"), { ...ADAPTER })));
});

afterEach(async () => {
  await remove(root, { recursive: true });
});

/** The field the derivation gives the registered table from the two records. */
const expectedField = (registered: GeometryProfile) =>
  deriveGemmGrid(
    [
      { report: gridReport("2026-11-28T05:00:00.000Z"), path: GRID_A, sha256: "a" },
      { report: gridReport("2026-11-28T06:00:00.000Z"), path: GRID_B, sha256: "b" },
    ],
    { table: registered },
  ).gemmGrid;

describe("--grid-records on a registered table", () => {
  it("in a dry run prints the verdicts and the field difference, and writes nothing", async () => {
    const code = await main(
      ["test-gpu", `--grid-records=${GRID_A},${GRID_B}`, "--dry-run"],
      environment(registeredTable()),
    );
    assertEquals(code, 0, stderr.join("\n"));
    assertEquals(readdirSync(join(root, PROFILES_DIRECTORY)), []);
    const text = stdout.join("\n");
    assertStringIncludes(text, "- gemmGrid (the grid-fill rule of the GEMM; 33 cases;");
    assertStringIncludes(text, "gemmGrid: written to the table");
    assertStringIncludes(text, "  gemmGrid: (absent) -> { ladder: [");
    assertEquals(text.includes("gemmRows["), false, "no M bucket differs");
  });

  it("writes the table with the derived field, and its header names the grid records and the rule", async () => {
    const registered = registeredTable();
    const field = expectedField(registered);
    assert(field !== undefined, "the records give the registered table a field");
    const code = await main(
      ["test-gpu", `--grid-records=${GRID_A},${GRID_B}`],
      environment(registered),
    );
    assertEquals(code, 0, stderr.join("\n"));
    const text = readFileSync(join(root, TABLE), "utf8");
    assertStringIncludes(
      text,
      "Derived in this generation: the grid-fill rule of the GEMM (grid records).",
    );
    assertStringIncludes(text, ` * - ${GRID_A} (sha256 `);
    assertStringIncludes(text, ` * - ${GRID_B} (sha256 `);
    assertStringIncludes(
      text,
      " * reg128x128r8x8w16, reg64x64r4x4w16, reg64x32r4x4w8, reg32x32r2x4w8, reg16x16r1x4w4,",
    );
    const module: unknown = await import(pathToFileURL(join(root, TABLE)).href);
    const table = (module as Record<string, GeometryProfile>).TEST_GPU;
    assertEquals(table.gemmGrid, field);
    assertEquals(table.gemmRows, registered.gemmRows);
  });

  it("keeps a table's field through a regeneration of the tile geometry alone", async () => {
    const registered = registeredTable();
    const field = expectedField(registered);
    assert(field !== undefined);
    const carrying: GeometryProfile = { ...registered, gemmGrid: field };
    const code = await main(["test-gpu", `--records=${SWEEP}`], environment(carrying));
    assertEquals(code, 0, stderr.join("\n"));
    const text = readFileSync(join(root, TABLE), "utf8");
    assertStringIncludes(text, "the grid-fill rule of the GEMM.");
    const module: unknown = await import(pathToFileURL(join(root, TABLE)).href);
    assertEquals((module as Record<string, GeometryProfile>).TEST_GPU.gemmGrid, field);
  });
});

describe("the command line with --grid-records", () => {
  it("refuses grid records beside --shared and --gemv-plan", async () => {
    const shared = await main(
      ["test-gpu", "--shared", `--records=${SWEEP}`, `--grid-records=${GRID_A}`],
      environment(registeredTable()),
    );
    assertEquals(shared, 2);
    assertStringIncludes(stderr.join("\n"), "drop --grid-records");
    stderr = [];
    const plan = await main(
      ["test-gpu", "--gemv-plan", `--grid-records=${GRID_A}`],
      environment(registeredTable()),
    );
    assertEquals(plan, 2);
    assertStringIncludes(stderr.join("\n"), "--grid-records");
  });
});
