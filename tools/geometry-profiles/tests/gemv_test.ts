/**
 * The GEMV choices of linear in the command (`--gemv-records`): a dry run prints the derivation's
 * decisions and the field differences and writes nothing; a written table carries the derived
 * field, its records and its rule in the header; a table that carries the field keeps it through a
 * regeneration of the tile geometry alone; one GEMV record derives nothing, and a first table
 * without tile-geometry records is refused. Each case runs on a temporary repository whose registered table is built here
 * from synthetic records of one adapter (the GEMV records from the runtime's own measurement plans —
 * packages/runtime/tests/helpers/gemv-sweep-records.ts). The formatter is the real one. No GPU.
 *
 * Faults injected: one GEMV record instead of two (nothing adopted). The carry-over case fails when the
 * tile-geometry derivation does not carry a base's `linearGemv` (checked by removing the carry).
 */

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { assert, assertEquals, assertStringIncludes } from "@std/assert";
import { BUILTIN_GEOMETRY_PROFILES, type GeometryProfile } from "@karume/runtime";
import { deriveGeometryProfile } from "@karume/runtime/tune";
import { makeTempDir, remove } from "@karume/test-support/fs";
import { afterEach, beforeEach, describe, it } from "vite-plus/test";
import {
  BASE_TIME,
  GEMV_ADAPTER,
  gemvRecord,
} from "../../../packages/runtime/tests/helpers/gemv-sweep-records.ts";
import {
  familyDefaultRows,
  report,
} from "../../../packages/runtime/tests/helpers/sweep-records.ts";
import { type CommandEnvironment, deriveGemvTable, main, PROFILES_DIRECTORY } from "../generate.ts";
import { loadRecord } from "../records.ts";
import { ADAPTER, bytesOf, workbenchRecord } from "./helpers/records.ts";

const GEMV_A = "outputs/workbench/node-test-gpu/gemv-sweep_a.json";
const GEMV_B = "outputs/workbench/node-test-gpu/gemv-sweep_b.json";
const SWEEP = "outputs/workbench/node-test-gpu/geometry-sweep_gemm.json";
const BASE_COMMIT = "fedcba9876543210fedcba9876543210fedcba98";
const TABLE = `${PROFILES_DIRECTORY}/test-gpu.ts`;

/** E2B's per_layer_model_projection: c64u8 ×1.2 over the baseline (identical), staged-l8 ×1.3 over parallel-l4. */
const PLE = "i4g32-n8960-k1536";
const TIMES = { 1: { c64u8: BASE_TIME / 1.2, "staged-l8": BASE_TIME / 1.3 } };

let root = "";
let stdout: string[] = [];
let stderr: string[] = [];

/** The registered table: a first table of the adapter from a gemm record that adopts nothing. */
const registeredTable = (): GeometryProfile =>
  deriveGeometryProfile(
    [
      {
        report: report(familyDefaultRows("gemm"), { ...ADAPTER }),
        path: "outputs/old.json",
        sha256: "c0ffee",
      },
    ],
    { id: "test-gpu", vendor: ADAPTER.vendor, architecture: ADAPTER.architecture },
  ).profile;

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

const loaded = (path: string, kind: "records" | "gemv-records") =>
  loadRecord(readFileSync(join(root, path)), path, kind);

beforeEach(async () => {
  root = await makeTempDir({ prefix: "karume-geometry-profiles-gemv-" });
  stdout = [];
  stderr = [];
  mkdirSync(join(root, PROFILES_DIRECTORY), { recursive: true });
  for (const [path, date] of [
    [GEMV_A, "2026-11-17T11:00:00.000Z"],
    [GEMV_B, "2026-11-17T12:00:00.000Z"],
  ] as const) {
    write(
      path,
      bytesOf(
        workbenchRecord(
          "gemv-sweep",
          gemvRecord(date, [{ caseId: PLE, rows: [1, 4, 8], times: TIMES }]),
        ),
      ),
    );
  }
  write(SWEEP, bytesOf(report(familyDefaultRows("gemm"), { ...ADAPTER })));
});

afterEach(async () => {
  await remove(root, { recursive: true });
});

describe("--gemv-records on a registered table", () => {
  it("in a dry run prints the decisions and the field differences, and writes nothing", async () => {
    assertEquals(GEMV_ADAPTER, ADAPTER);
    const code = await main(
      ["test-gpu", `--gemv-records=${GEMV_A},${GEMV_B}`, "--dry-run"],
      environment(registeredTable()),
    );
    assertEquals(code, 0, stderr.join("\n"));
    assertEquals(readdirSync(join(root, PROFILES_DIRECTORY)), []);
    const text = stdout.join("\n");
    assertStringIncludes(text, `  - [1] ${GEMV_A} (2026-11-17T11:00:00.000Z, set full)`);
    assertStringIncludes(
      text,
      `  - i4 n8960 k1536 (case ${PLE}): c32u4 -> c64u8 (×1.200 [1], ×1.200 [2])`,
    );
    assertStringIncludes(
      text,
      // The default's row carries a row block of 4; the records show none faster for 8 lanes
      `  - i4 g32 n8960 k1536 (case ${PLE}): parallel-l4 with row blocks of 4 -> staged-l8 (×1.300 [1], ×1.300 [2])`,
    );
    assertStringIncludes(text, "linearGemv: written to the table");
    // The registered table has no field: the differences are against the default's choices, whose
    // 26 per-shape exceptions the adopted one follows
    assertStringIncludes(
      text,
      '  linearGemv.sequential.shapes[26]: (absent) -> { storage: "i4", n: 8960, k: 1536, variant: { cols: 64, unroll: 8 } }',
    );
    assertStringIncludes(text, "].lanes: 4 -> 8");
    assertEquals(text.includes("gemmRows["), false, "no tile geometry differs");
  });

  it("writes the table with the derived field, its records and its rule in the header", async () => {
    const registered = registeredTable();
    const code = await main(
      ["test-gpu", `--gemv-records=${GEMV_A},${GEMV_B}`],
      environment(registered),
    );
    assertEquals(code, 0, stderr.join("\n"));
    assertEquals(stdout[0], `Wrote ${TABLE}`);
    const path = join(root, TABLE);
    const module: unknown = await import(pathToFileURL(path).href);
    assert(typeof module === "object" && module !== null && "TEST_GPU" in module);
    const expected = deriveGemvTable(registered, [
      loaded(GEMV_A, "gemv-records"),
      loaded(GEMV_B, "gemv-records"),
    ]);
    assert(expected.profile.linearGemv !== undefined);
    assertEquals(module.TEST_GPU, expected.profile);
    const header = readFileSync(path, "utf8").split("import type {")[0];
    assertStringIncludes(
      header,
      " * Derived in this generation: the GEMV choices of linear (GEMV sweep records).\n",
    );
    assertStringIncludes(header, ` * - ${GEMV_A} (sha256 `);
    assertStringIncludes(
      header,
      `Carried over from the registered table at commit ${BASE_COMMIT}: gemm.`,
    );
    assertStringIncludes(header, "least ×1.100 faster than that choice in every GEMV sweep record");
    assertStringIncludes(header, "spread beyond ×1.250 keeps the default");
    // No tile geometry was derived, so its rule is not stated
    assertEquals(header.includes("Adoption: a geometry replaces"), false);
    assertStringIncludes(
      header,
      " *   - i4 g32 n8960 k1536 (case i4g32-n8960-k1536): parallel-l4 with row blocks of 4 -> staged-l8",
    );
  });

  it("derives nothing from one GEMV record: its groups are one record's evidence (fault: one record)", async () => {
    const code = await main(
      ["test-gpu", "--dry-run", `--gemv-records=${GEMV_A}`],
      environment(registeredTable()),
    );
    assertEquals(code, 0, stderr.join("\n"));
    const printed = stdout.join("\n");
    assertStringIncludes(printed, "evidence at M=1 from fewer than 2 records");
    assertStringIncludes(printed, "linearGemv: absent from the table");
    assertEquals(readdirSync(join(root, PROFILES_DIRECTORY)), []);
  });

  it("refuses a first table from GEMV records alone (it needs the tile geometry's records)", async () => {
    const code = await main(
      ["other-gpu", `--gemv-records=${GEMV_A},${GEMV_B}`, `--vendor=${ADAPTER.vendor}`],
      environment(registeredTable()),
    );
    assertEquals(code, 2);
    assertStringIncludes(stderr[0] ?? "", "a first table needs --records");
  });
});

describe("a table that carries linearGemv", () => {
  it("keeps it through a regeneration of the tile geometry alone", async () => {
    const registered = registeredTable();
    const { profile: carrying } = deriveGemvTable(registered, [
      loaded(GEMV_A, "gemv-records"),
      loaded(GEMV_B, "gemv-records"),
    ]);
    assert(carrying.linearGemv !== undefined);
    const code = await main(["test-gpu", `--records=${SWEEP}`], environment(carrying));
    assertEquals(code, 0, stderr.join("\n"));
    const path = join(root, TABLE);
    const module: unknown = await import(pathToFileURL(path).href);
    assert(typeof module === "object" && module !== null && "TEST_GPU" in module);
    const written: unknown = module.TEST_GPU;
    assert(typeof written === "object" && written !== null && "linearGemv" in written);
    assertEquals(written.linearGemv, carrying.linearGemv);
    const header = readFileSync(path, "utf8").split("import type {")[0];
    assertStringIncludes(header, " * Derived in this generation: gemm.\n");
    assertStringIncludes(
      header,
      `Carried over from the registered table at commit ${BASE_COMMIT}: the GEMV choices of linear.`,
    );
  });
});
