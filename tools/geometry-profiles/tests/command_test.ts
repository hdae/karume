/**
 * The command: what it refuses on the command line (exit 2), that a dry run writes nothing, and a
 * new adapter's first table written from hand-built sweep records. Each case runs on a temporary
 * repository holding copies of the two registration files and the records, so no test writes into
 * packages/runtime/src. The formatter is the real one. No GPU.
 */

import { copyFileSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { assert, assertEquals } from "@std/assert";
import { BUILTIN_GEOMETRY_PROFILES } from "@karume/runtime";
import { deriveGeometryProfile } from "@karume/runtime/tune";
import { makeTempDir, remove } from "@karume/test-support/fs";
import { afterEach, beforeEach, describe, it } from "vite-plus/test";
import { REPOSITORY_ROOT } from "../format.ts";
import { type CommandEnvironment, main, PROFILES_DIRECTORY } from "../generate.ts";
import { loadRecord } from "../records.ts";
import { INDEX_PATH, PROFILE_TEST_PATH } from "../registration.ts";
import { ADAPTER, bytesOf, sweepRecord, workbenchRecord } from "./helpers/records.ts";

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
          "becomes: [APPLE_METAL_3, NVIDIA_AMPERE, NVIDIA_BLACKWELL, TEST_GPU].map(freezeGeometryProfile)",
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
