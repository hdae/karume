/**
 * The command: what it refuses on the command line (exit 2), that a dry run writes nothing, a new
 * adapter's first table written from hand-built sweep records, and a shared table of two adapters.
 * Each case runs on a temporary repository holding copies of the two registration files and the
 * records, so no test writes into packages/runtime/src. The formatter is the real one. No GPU.
 */

import { copyFileSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { assert, assertEquals } from "@std/assert";
import { BUILTIN_GEOMETRY_PROFILES, type GeometryProfile } from "@karume/runtime";
import {
  deriveGeometryProfile,
  geometryProfileMismatch,
  geometrySweepCases,
} from "@karume/runtime/tune";
import { makeTempDir, remove } from "@karume/test-support/fs";
import { afterEach, beforeEach, describe, it } from "vite-plus/test";
import { formatTypeScript, REPOSITORY_ROOT } from "../format.ts";
import { type CommandEnvironment, main, PROFILES_DIRECTORY } from "../generate.ts";
import { profileModule } from "../literal.ts";
import { loadRecord } from "../records.ts";
import { INDEX_PATH, PROFILE_TEST_PATH } from "../registration.ts";
import {
  ADAPTER,
  bytesOf,
  CASE_ID,
  CONV2D_FIELDS_GEOMETRY,
  DEFAULT_GEOMETRY,
  FAST_GEOMETRY,
  fieldsSweepRecord,
  FORM_GEOMETRY,
  OTHER_ADAPTER,
  sweepRecord,
  workbenchRecord,
} from "./helpers/records.ts";

const RECORD_A = "outputs/workbench/browser-test-gpu/geometry-sweep_a.json";
const RECORD_B = "outputs/diag/bare-b.json";
const MATCH = [`--vendor=${ADAPTER.vendor}`, `--architecture=${ADAPTER.architecture}`];

let root = "";
let stdout: string[] = [];
let stderr: string[] = [];

const environment = (): CommandEnvironment => ({
  repositoryRoot: root,
  registered: BUILTIN_GEOMETRY_PROFILES,
  lastCommit: () => {
    throw new Error("a first table has no base commit");
  },
  stdout: (text) => stdout.push(text),
  stderr: (text) => stderr.push(text),
});

const write = (path: string, bytes: Uint8Array | string): void => {
  mkdirSync(join(root, path, ".."), { recursive: true });
  writeFileSync(join(root, path), bytes);
};

beforeEach(async () => {
  root = await makeTempDir({ prefix: "karume-geometry-profiles-" });
  stdout = [];
  stderr = [];
  for (const path of [INDEX_PATH, PROFILE_TEST_PATH]) {
    mkdirSync(join(root, path, ".."), { recursive: true });
    copyFileSync(join(REPOSITORY_ROOT, path), join(root, path));
  }
  write(RECORD_A, bytesOf(workbenchRecord("geometry-sweep", sweepRecord("2026-11-16T01:00:00Z"))));
  write(RECORD_B, bytesOf(sweepRecord("2026-11-16T02:00:00Z", 1.3)));
});

afterEach(async () => {
  await remove(root, { recursive: true });
});

describe("the command line", () => {
  it("refuses match options for a registered id (its match is the table's own)", async () => {
    const code = await main(
      ["nvidia-blackwell", `--records=${RECORD_A}`, "--vendor=nvidia"],
      environment(),
    );
    assertEquals(code, 2);
    assertEquals(stderr[0]?.includes("'nvidia-blackwell' is registered"), true);
  });

  it("refuses an id that is not registered without a match", async () => {
    const code = await main(["test-gpu", `--records=${RECORD_A}`], environment());
    assertEquals(code, 2);
    assertEquals(stderr[0]?.includes("'test-gpu' is not registered"), true);
  });

  it("refuses a record path outside the repository", async () => {
    const code = await main(["test-gpu", "--records=../outside.json", ...MATCH], environment());
    assertEquals(code, 2);
  });

  it("exits 1 with the derivation's own message when it refuses the records", async () => {
    const code = await main(
      ["test-gpu", `--records=${RECORD_A}`, "--vendor=othervendor"],
      environment(),
    );
    assertEquals(code, 1);
    // Unprefixed: the line starts with the record path, as the derivation words it.
    assertEquals(stderr.length, 1);
    assertEquals(
      stderr[0]?.startsWith(
        `${RECORD_A}: adapter testvendor / testarch / Test GPU does not match the requested`,
      ),
      true,
      stderr[0],
    );
    assertEquals(stdout, []);
  });
});

describe("a dry run", () => {
  it("writes nothing and prints the derived profile and the verdicts", async () => {
    const code = await main(
      ["test-gpu", `--records=${RECORD_A},${RECORD_B}`, ...MATCH, "--dry-run"],
      environment(),
    );
    assertEquals(code, 0, stderr.join("\n"));
    assertEquals(readdirSync(join(root, PROFILES_DIRECTORY)), ["index.ts"]);
    assertEquals(stdout.includes("Verdicts:"), true);
    assertEquals(
      stdout.some((text) => text.includes('"id": "test-gpu"')),
      true,
    );
  });
});

describe("a new adapter's first table", () => {
  it("is written from the records, prints the registration edits and makes none", async () => {
    const code = await main(
      ["test-gpu", `--records=${RECORD_A},${RECORD_B}`, ...MATCH],
      environment(),
    );
    assertEquals(code, 0, stderr.join("\n"));
    const tablePath = join(root, PROFILES_DIRECTORY, "test-gpu.ts");
    const text = readFileSync(tablePath, "utf8");
    assertEquals(
      text.split("\n")[1],
      " * Geometry profile `test-gpu` (**generated — do not edit by hand**).",
    );
    assertEquals(text.includes("Carried over"), false);
    const records = [RECORD_A, RECORD_B].map((path) =>
      loadRecord(readFileSync(join(root, path)), path, "records"),
    );
    const expected = deriveGeometryProfile(records, {
      id: "test-gpu",
      vendor: ADAPTER.vendor,
      architecture: ADAPTER.architecture,
      minSpeedup: 1.05,
    }).profile;
    // The written module exports the derived table (its only import is type-only).
    const module: unknown = await import(pathToFileURL(tablePath).href);
    assert(typeof module === "object" && module !== null && "TEST_GPU" in module);
    assertEquals(module.TEST_GPU, expected);
    assertEquals(stdout[0], `Wrote ${PROFILES_DIRECTORY}/test-gpu.ts`);
    assertEquals(
      stdout.filter((line) => line.includes('add: import { TEST_GPU } from "')).length,
      2,
    );
    assertEquals(
      stdout.some((line) =>
        line.includes(
          "become: [APPLE_METAL_3, NVIDIA_AMPERE, NVIDIA_BLACKWELL, APPLE_M5, APPLE_METAL_3_SHARED, TEST_GPU].map(freezeGeometryProfile),",
        ),
      ),
      true,
    );
    for (const path of [INDEX_PATH, PROFILE_TEST_PATH]) {
      assertEquals(
        readFileSync(join(root, path), "utf8"),
        readFileSync(join(REPOSITORY_ROOT, path), "utf8"),
      );
    }
  });
});

describe("a table that adopts a skeleton form and conv2d's counter decode (perf-ledger K-88)", () => {
  const RECORD = "outputs/workbench/browser-test-gpu/geometry-sweep_fields.json";

  beforeEach(() => {
    write(
      RECORD,
      bytesOf(workbenchRecord("geometry-sweep", fieldsSweepRecord("2026-11-18T01:00:00Z"))),
    );
  });

  it("the dry run names the adopted geometries with their fields", async () => {
    const code = await main(
      ["test-gpu", `--records=${RECORD}`, ...MATCH, "--dry-run"],
      environment(),
    );
    assertEquals(code, 0, stderr.join("\n"));
    const printed = stdout.join("\n");
    assertEquals(printed.includes("adopted reg128x128r8x8w16:tb-un ×1.200"), true, printed);
    assertEquals(
      printed.includes("adopted igemm64x128:wg16x8:tas-tb-un:dec ×1.300"),
      true,
      printed,
    );
    assertEquals(readdirSync(join(root, PROFILES_DIRECTORY)), ["index.ts"]);
  });

  it("is written with the fields in its rules, and the file is what the printer gives for the table it exports", async () => {
    const code = await main(["test-gpu", `--records=${RECORD}`, ...MATCH], environment());
    assertEquals(code, 0, stderr.join("\n"));
    const path = `${PROFILES_DIRECTORY}/test-gpu.ts`;
    const text = readFileSync(join(root, path), "utf8");
    assertEquals(
      text.includes('{ regM: 8, regN: 8, wgX: 16, wgY: 16, form: "tb-un" }'),
      true,
      text,
    );
    assertEquals(
      text.includes(
        'rows64: { regM: 8, regN: 8, wgX: 16, wgY: 8, form: "tas-tb-un", decode: "counter" }',
      ),
      true,
      text,
    );
    const module: unknown = await import(pathToFileURL(join(root, path)).href);
    assert(typeof module === "object" && module !== null && "TEST_GPU" in module);
    const table = module.TEST_GPU;
    const record = loadRecord(readFileSync(join(root, RECORD)), RECORD, "records");
    const expected = deriveGeometryProfile([record], {
      id: "test-gpu",
      vendor: ADAPTER.vendor,
      architecture: ADAPTER.architecture,
      minSpeedup: 1.05,
    }).profile;
    assertEquals(table, expected);
    assertEquals(expected.gemmRows.at(-1)?.geometry, FORM_GEOMETRY);
    assertEquals(expected.conv2d.rows64, CONV2D_FIELDS_GEOMETRY);
    // The check the core lane runs on every registered table: the file from its import line on is
    // the printer's text for the table
    assertEquals(
      text.slice(text.indexOf("import type {")),
      formatTypeScript(profileModule(expected), path),
    );
    // … and the written table stands for the kernels it runs on its adapter
    assertEquals(geometryProfileMismatch(expected, ADAPTER), undefined);
  });
});

describe("a shared table (--shared)", () => {
  const RECORD_OTHER = "outputs/workbench/browser-other-gpu/geometry-sweep_c.json";

  beforeEach(() => {
    write(
      RECORD_OTHER,
      bytesOf(
        workbenchRecord("geometry-sweep", sweepRecord("2026-11-16T03:00:00Z", 1.4, OTHER_ADAPTER)),
      ),
    );
  });

  it("refuses --shared beside a description, attention-bench or GEMV records and --gemv-plan, and without records", async () => {
    for (const extra of [
      [...MATCH, "--description=Test GPU"],
      [...MATCH, `--attention-bench=${RECORD_B}`],
      [...MATCH, `--gemv-records=${RECORD_B}`],
      ["--gemv-plan"],
    ]) {
      stderr = [];
      const code = await main(
        ["test-shared", "--shared", `--records=${RECORD_A},${RECORD_OTHER}`, ...extra],
        environment(),
      );
      assertEquals(code, 2, extra.join(" "));
      assertEquals(
        stderr[0]?.startsWith("--shared derives the tile geometry of a table matched by vendor"),
        true,
        stderr[0],
      );
    }
    stderr = [];
    assertEquals(await main(["test-shared", "--shared", ...MATCH], environment()), 2);
    assertEquals(
      await main(
        ["test-shared", "--shared", `--records=${RECORD_A}`, `--vendor=${ADAPTER.vendor}`],
        environment(),
      ),
      2,
    );
  });

  it("exits 1 with the derivation's message for the records of one adapter", async () => {
    const code = await main(
      ["test-shared", "--shared", `--records=${RECORD_A},${RECORD_B}`, ...MATCH],
      environment(),
    );
    assertEquals(code, 1);
    assertEquals(
      stderr[0]?.startsWith("A shared derivation needs the records of at least two devices"),
      true,
      stderr[0],
    );
  });

  it("is written from the records of two adapters: the header names the devices, says the table is matched without a description and lists the records per device, and the module exports the derived table", async () => {
    const code = await main(
      ["test-shared", "--shared", `--records=${RECORD_A},${RECORD_OTHER}`, ...MATCH],
      environment(),
    );
    assertEquals(code, 0, stderr.join("\n"));
    const path = `${PROFILES_DIRECTORY}/test-shared.ts`;
    const text = readFileSync(join(root, path), "utf8");
    const lines = text.split("\n");
    for (const expected of [
      " * A static table of tile geometries for the adapter match: vendor testvendor, architecture testarch.",
      " * A shared table: derived from the sweep records of 2 devices of that vendor and architecture,",
      " * Test GPU (device 0x1234) and Other GPU (device 0x1234).",
      " * It is matched without a description, so it predicts the geometries for every adapter of that",
      " * table with `node tools/geometry-profiles/generate.ts test-shared --shared --records=<path>,<path>[,…]`.",
      " * Records of Test GPU (device 0x1234):",
      ` * - ${RECORD_A} (sha256 ${loadRecord(readFileSync(join(root, RECORD_A)), RECORD_A, "records").sha256}, 2026-11-16T01:00:00Z)`,
      " * Records of Other GPU (device 0x1234):",
      " * Adoption (shared): a geometry replaces the default of a field only when, on every device, on",
    ]) {
      assertEquals(lines.includes(expected), true, `${expected}\n${text.slice(0, 3000)}`);
    }
    assertEquals(
      lines.some((line) =>
        line.startsWith(
          " * - gemmRows > 512 (linear / matmul / bmm row count > 512; Test GPU 1 cases, Other GPU 1 cases): adopted reg64x32r4x4w8 ×1.296 over the devices (Test GPU ×1.200 (×1.200–×1.200), Other GPU ×1.400 (×1.400–×1.400))",
        ),
      ),
      true,
      text,
    );
    const records = [RECORD_A, RECORD_OTHER].map((record) =>
      loadRecord(readFileSync(join(root, record)), record, "records"),
    );
    const expected = deriveGeometryProfile(records, {
      id: "test-shared",
      vendor: ADAPTER.vendor,
      architecture: ADAPTER.architecture,
      minSpeedup: 1.05,
      shared: true,
    }).profile;
    const module: unknown = await import(pathToFileURL(join(root, path)).href);
    assert(typeof module === "object" && module !== null && "TEST_SHARED" in module);
    assertEquals(module.TEST_SHARED, expected);
    assertEquals(expected.match, { vendor: ADAPTER.vendor, architecture: ADAPTER.architecture });
    assertEquals(expected.provenance?.adapters, [ADAPTER, OTHER_ADAPTER]);
    assertEquals(
      text.slice(text.indexOf("import type {")),
      formatTypeScript(profileModule(expected), path),
    );
    // A shared table predicts for an adapter of its vendor and architecture it was not measured on
    assertEquals(
      geometryProfileMismatch(expected, { ...ADAPTER, device: "", description: "" }),
      undefined,
    );
  });

  describe("regenerated on its registered table, with a family each device covers only in part", () => {
    // Each record holds one gemm case of the sweep's table, so both devices cover gemm only in part
    const loaded = () =>
      [RECORD_A, RECORD_OTHER].map((record) =>
        loadRecord(readFileSync(join(root, record)), record, "records"),
      );
    const SHARED_TARGET = {
      id: "test-shared",
      vendor: ADAPTER.vendor,
      architecture: ADAPTER.architecture,
      minSpeedup: 1.05,
      shared: true,
    };
    const registered = (): GeometryProfile =>
      deriveGeometryProfile(loaded(), SHARED_TARGET).profile;
    const onBase = (): CommandEnvironment => ({
      ...environment(),
      registered: [...BUILTIN_GEOMETRY_PROFILES, registered()],
      lastCommit: () => "fedcba9876543210fedcba9876543210fedcba98",
    });
    const missing = (): string =>
      geometrySweepCases()
        .filter(({ id, op }) => ["linear", "matmul", "bmm"].includes(op) && id !== CASE_ID)
        .map(({ id }) => `${id} (no row)`)
        .join(", ");
    const reason = (): string =>
      `(Test GPU covers family gemm only in part: no usable row of the default geometry in ${missing()}; ` +
      `Other GPU covers family gemm only in part: no usable row of the default geometry in ${missing()} ` +
      "(regenerated on a base, a shared table takes a geometry for a family only where every device measured every case of its case set))";
    const LARGEST = `- gemmRows > 512 (linear / matmul / bmm row count > 512; Test GPU 1 cases, Other GPU 1 cases): keeps the default reg128x128r8x8w16 `;

    it("keeps the default in every gemm field, and the dry run prints the reason naming each device and the cases", async () => {
      // Without a base the same records adopt a geometry in the bucket above 512 rows
      assertEquals(registered().gemmRows.at(-1)?.geometry, FAST_GEOMETRY);
      const code = await main(
        ["test-shared", "--shared", `--records=${RECORD_A},${RECORD_OTHER}`, "--dry-run"],
        onBase(),
      );
      assertEquals(code, 0, stderr.join("\n"));
      assertEquals(stdout.includes(`${LARGEST}${reason()}`), true, stdout.join("\n"));
      assertEquals(
        stdout.filter((text) => text.startsWith("- gemmRows ") && text.endsWith(reason())).length,
        7,
        stdout.join("\n"),
      );
    });

    it("writes the table with the default gemm fields, and its header states the rule and names the devices and the cases", async () => {
      const code = await main(
        ["test-shared", "--shared", `--records=${RECORD_A},${RECORD_OTHER}`],
        onBase(),
      );
      assertEquals(code, 0, stderr.join("\n"));
      const path = `${PROFILES_DIRECTORY}/test-shared.ts`;
      const lines = readFileSync(join(root, path), "utf8").split("\n");
      for (const expected of [
        " * Regenerated on the registered table, a family some device covers only in part (a case of its",
        " * case set with no usable row of the default geometry there) keeps the default in every field;",
        ` * ${LARGEST}${reason()}`,
      ]) {
        assertEquals(lines.includes(expected), true, `${expected}\n${lines.join("\n")}`);
      }
      const expected = deriveGeometryProfile(loaded(), {
        ...SHARED_TARGET,
        base: registered(),
      }).profile;
      assertEquals(expected.gemmRows.at(-1)?.geometry, DEFAULT_GEOMETRY);
      const module: unknown = await import(pathToFileURL(join(root, path)).href);
      assert(typeof module === "object" && module !== null && "TEST_SHARED" in module);
      assertEquals(module.TEST_SHARED, expected);
    });
  });
});
