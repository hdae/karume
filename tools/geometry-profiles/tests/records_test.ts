/**
 * Loading a record: a workbench record is unwrapped to its `data` when its entry is the one the
 * option takes, a bare runtime record is passed as it is, and anything else is refused naming the
 * file; a record path must lie inside the repository. No GPU and no records on disk.
 */

import { createHash } from "node:crypto";
import { assertEquals, assertStrictEquals, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { loadRecord, RecordError, repositoryRelativePath, UsageError } from "../records.ts";
import { bytesOf, CHROME, sweepRecord, workbenchRecord } from "./helpers/records.ts";

const sha256 = (bytes: Uint8Array): string => createHash("sha256").update(bytes).digest("hex");

describe("loadRecord", () => {
  it("unwraps a workbench sweep record to its data and keeps how the run was made", () => {
    const data = sweepRecord("2026-11-16T01:00:00.000Z");
    const bytes = bytesOf(workbenchRecord("geometry-sweep", data, { dirty: true }));
    const loaded = loadRecord(bytes, "outputs/workbench/a.json", "records");
    assertEquals(loaded.report, data);
    assertEquals(loaded.sha256, sha256(bytes));
    assertEquals(loaded.path, "outputs/workbench/a.json");
    assertEquals(loaded.origin, {
      kind: "workbench",
      host: "browser",
      userAgent: CHROME,
      backend: { name: "d3d12", source: "declared" },
      checkout: { commit: "0123456789abcdef0123456789abcdef01234567", dirty: true },
    });
  });

  it("unwraps a workbench attention-bench record for --attention-bench", () => {
    const data = { format: "karume-attention-bench/2", date: "2026-11-16T02:00:00.000Z" };
    const loaded = loadRecord(
      bytesOf(workbenchRecord("attention-bench", data)),
      "b.json",
      "attention-bench",
    );
    assertEquals(loaded.report, data);
    assertStrictEquals(loaded.origin.kind, "workbench");
  });

  it("passes a bare runtime record as it is, whatever its format version", () => {
    const old = { format: "karume-geometry-sweep/2", date: "2026-10-01T00:00:00.000Z" };
    const loaded = loadRecord(bytesOf(old), "outputs/bench-browser/old.json", "records");
    assertEquals(loaded.report, old);
    assertEquals(loaded.origin, { kind: "bare" });
  });

  it("refuses a workbench record of another entry than the option takes", () => {
    const bytes = bytesOf(workbenchRecord("geometry-sweep", sweepRecord("2026-11-16T01:00:00Z")));
    const error = assertThrows(() => loadRecord(bytes, "a.json", "attention-bench"), RecordError);
    assertEquals(error.message.startsWith("a.json: "), true);
    assertEquals(error.message.includes('"geometry-sweep"'), true);
  });

  it("refuses a workbench record without data, naming the file and the outcome", () => {
    const bytes = bytesOf(workbenchRecord("geometry-sweep", undefined, { outcome: "error" }));
    const error = assertThrows(() => loadRecord(bytes, "failed.json", "records"), RecordError);
    assertEquals(error.message, 'failed.json: the workbench record has no data (outcome "error")');
  });

  it("refuses a file that is neither, naming the file", () => {
    const bench = bytesOf({ format: "karume-attention-bench/2" });
    assertThrows(() => loadRecord(bench, "bench.json", "records"), RecordError, "bench.json: ");
    assertThrows(
      () => loadRecord(new TextEncoder().encode("not json"), "x.txt", "records"),
      RecordError,
      "x.txt: not a JSON file",
    );
    assertThrows(() => loadRecord(bytesOf([1, 2]), "list.json", "records"), RecordError);
  });
});

describe("repositoryRelativePath", () => {
  it("takes a relative path from the repository root and spells an absolute one relative to it", () => {
    assertEquals(repositoryRelativePath("/repo", "outputs/a.json"), "outputs/a.json");
    assertEquals(repositoryRelativePath("/repo", "/repo/outputs/a.json"), "outputs/a.json");
  });

  it("refuses a path outside the repository", () => {
    assertThrows(() => repositoryRelativePath("/repo", "../elsewhere/a.json"), UsageError);
    assertThrows(() => repositoryRelativePath("/repo", "/elsewhere/a.json"), UsageError);
    assertThrows(() => repositoryRelativePath("/repo", "/repo"), UsageError);
  });
});
