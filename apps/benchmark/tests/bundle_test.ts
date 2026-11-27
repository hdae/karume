import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { gemvSweepResult } from "@karume/sweep-app-support/gemv-sweep/summary";
import { geometrySweepResult } from "@karume/sweep-app-support/geometry-sweep/summary";
import { recordRun, recordText, type WorkbenchRecord } from "@karume/sweep-app-support/record";
import { afterEach, beforeEach, describe, expect, it } from "vite-plus/test";

import { loadRecord, repositoryRelativePath } from "../../../tools/geometry-profiles/records.ts";
import {
  type Bundle,
  BUNDLE_FORMAT,
  BundleError,
  bundleFileName,
  bundleRecordFiles,
  bundleText,
  parseBundle,
} from "../src/bundle.ts";
import { planBenchmark } from "../src/plan.ts";
import { unpackBundles } from "../unpack.ts";
import {
  CHECKOUT,
  CONV3D_A,
  CONV3D_B,
  ENVIRONMENT,
  GEMV_A,
  GEMV_B,
  gemvReport,
  geometryReport,
} from "./helpers/fakes.ts";

const PLAN = planBenchmark(
  { gemv: true, geometry: true },
  { cases: [GEMV_A, GEMV_B, CONV3D_A, CONV3D_B] },
);

/** The two records the shared record code makes of two finished per-case runs. */
const records = async (): Promise<WorkbenchRecord[]> => {
  const signal = new AbortController().signal;
  const made: WorkbenchRecord[] = [];
  for (const sweep of PLAN.sweeps) {
    made.push(
      await recordRun(
        {
          entry: { id: sweep.entryId, parameters: sweep.parameters },
          environment: ENVIRONMENT,
          checkout: CHECKOUT,
          signal,
        },
        () =>
          Promise.resolve(
            sweep.sweep === "gemv"
              ? gemvSweepResult(sweep.plan, {
                  reports: sweep.plan.caseIds.map((caseId) => gemvReport(caseId)),
                  stop: { kind: "completed" },
                })
              : geometrySweepResult(sweep.plan, {
                  reports: sweep.plan.caseIds.map((caseId) => geometryReport(caseId)),
                  stop: { kind: "completed" },
                }),
          ),
      ),
    );
  }
  return made;
};

const bundleOf = (made: readonly WorkbenchRecord[], complete = true): Bundle => ({
  format: BUNDLE_FORMAT,
  startedAt: "2026-11-26T12:00:00.000Z",
  endedAt: "2026-11-26T13:40:00.000Z",
  complete,
  stop: complete ? { kind: "completed" } : { kind: "cancelled" },
  sweeps: ["gemv-sweep", "geometry-sweep"],
  trial: { cases: [GEMV_A, GEMV_B, CONV3D_A, CONV3D_B] },
  page: { hiddenCount: 1, hiddenSeconds: 2.5, wakeLock: "held" },
  records: made,
});

let root: string;

beforeEach(async () => {
  root = await mkdtemp(join(tmpdir(), "benchmark-unpack-"));
});

afterEach(async () => {
  await rm(root, { recursive: true, force: true });
});

describe("the bundle", () => {
  it("reads back what it wrote, its records byte for byte what the shared record code gives", async () => {
    const made = await records();
    const read = parseBundle(bundleText(bundleOf(made)));
    expect(read).toEqual(bundleOf(made));
    expect(read.records.map(recordText)).toEqual(made.map(recordText));
  });

  it("places each record where the workbench writes it, by its adapter and the time it was made", async () => {
    const made = await records();
    const files = bundleRecordFiles(parseBundle(bundleText(bundleOf(made))));
    expect(files.map(({ path }) => path)).toEqual(
      made.map(
        (record) =>
          `outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/${record.entry.id}_${record.recordedAt.slice(0, 19).replaceAll(":", "-")}.json`,
      ),
    );
  });

  it("is named by the adapter and the UTC end time, marked when incomplete", async () => {
    const made = await records();
    expect(bundleFileName(bundleOf(made))).toBe(
      "karume-benchmark_browser-nvidia-geforce-rtx-5070-ti_2026-11-26T13-40-00.json",
    );
    expect(bundleFileName(bundleOf(made, false))).toBe(
      "karume-benchmark_browser-nvidia-geforce-rtx-5070-ti_2026-11-26T13-40-00_incomplete.json",
    );
  });

  it("refuses another format, a record of another entry and a record without its envelope (faults)", async () => {
    const made = await records();
    const text = bundleText(bundleOf(made));
    expect(() => parseBundle(text.replace(BUNDLE_FORMAT, "karume-benchmark/0"))).toThrow(
      BundleError,
    );
    const [first, second] = made;
    if (first === undefined || second === undefined) throw new Error("two records");
    const other = { ...first, entry: { ...first.entry, id: "kernel-check" } };
    expect(() =>
      parseBundle(JSON.stringify({ ...bundleOf(made), records: [other, second] })),
    ).toThrow(/neither gemv-sweep nor geometry-sweep/);
    const { checkout: _checkout, ...bare } = first;
    expect(() =>
      parseBundle(JSON.stringify({ ...bundleOf(made), records: [bare, second] })),
    ).toThrow(/Record 1 of the file: its checkout/);
  });
});

describe("unpack", () => {
  const writeBundle = async (name: string, bundle: Bundle): Promise<string> => {
    const path = join(root, name);
    await writeFile(path, bundleText(bundle));
    return path;
  };

  it("writes every record as the workbench would have, and the generator's reader takes each", async () => {
    const made = await records();
    const file = await writeBundle("run.json", bundleOf(made));
    const result = await unpackBundles(root, [file]);
    expect(result.code).toBe(0);
    const files = bundleRecordFiles(bundleOf(made));
    expect(result.lines).toEqual(files.map(({ path }) => `wrote ${path}`));
    for (const [index, { path }] of files.entries()) {
      const bytes = new Uint8Array(await readFile(join(root, path)));
      const record = made[index];
      if (record === undefined) throw new Error("a record per file");
      expect(new TextDecoder().decode(bytes)).toBe(recordText(record));
      const kind = record.entry.id === "gemv-sweep" ? "gemv-records" : "records";
      const loaded = loadRecord(bytes, repositoryRelativePath(root, join(root, path)), kind);
      expect(loaded.report).toEqual(record.data);
      expect(loaded.origin).toEqual({
        kind: "workbench",
        host: "browser",
        userAgent: ENVIRONMENT.userAgent,
        checkout: { commit: CHECKOUT.commit, dirty: false },
      });
    }
  });

  it("overwrites nothing: a second unpack of the same file writes nothing and names what exists", async () => {
    const made = await records();
    const file = await writeBundle("run.json", bundleOf(made));
    expect((await unpackBundles(root, [file])).code).toBe(0);
    const [first] = bundleRecordFiles(bundleOf(made));
    if (first === undefined) throw new Error("a record");
    await writeFile(join(root, first.path), "kept\n");
    const again = await unpackBundles(root, [file]);
    expect(again.code).toBe(1);
    expect(again.lines.at(-1)).toBe("Nothing was written.");
    expect(again.lines).toContain(`${first.path}: exists already`);
    expect(await readFile(join(root, first.path), "utf8")).toBe("kept\n");
  });

  it("refuses the same file twice before writing anything", async () => {
    const made = await records();
    const file = await writeBundle("run.json", bundleOf(made));
    const result = await unpackBundles(root, [file, file]);
    expect(result.code).toBe(1);
    expect(result.lines.at(-1)).toBe("Nothing was written.");
    const [first] = bundleRecordFiles(bundleOf(made));
    await expect(readFile(join(root, first?.path ?? ""))).rejects.toThrow();
  });

  it("writes an incomplete bundle's records and says it is incomplete; a file that is no bundle and no file at all are refused", async () => {
    const made = await records();
    const file = await writeBundle("partial.json", bundleOf(made.slice(0, 1), false));
    const result = await unpackBundles(root, [file]);
    expect(result.code).toBe(0);
    expect(result.lines[0]).toBe(
      `${file}: incomplete — it was cancelled; its records hold what was measured`,
    );
    const broken = await writeBundle("broken.json", { ...bundleOf(made), records: [] });
    await writeFile(broken, "{");
    expect((await unpackBundles(root, [broken])).lines[0]).toBe(`${broken}: The file is not JSON`);
    expect((await unpackBundles(root, [])).code).toBe(2);
  });
});
