// Checks of the pure part of verify-diff (touches neither the GPU nor outputs/ — every input is
// synthetic).
//
// The value of this tool is "pointing at where the results diverge", so three aspects are pinned:
// ① how results directories are picked (newest date / --date / the family filter) ② how differences are
// found (status / artifact sha / a case only one side has) ③ an unreadable document never silently
// becomes empty.

import { assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { buildDiff, type LoadedResults, parseResultsDocument } from "../diff.ts";
import { parseArgs, single } from "../main.ts";

/**
 * Synthetic input for one results directory (the same shape as a real `results.json`, passed through the
 * boundary check).
 */
const seat = (
  environment: string,
  date: string,
  family: string,
  cases: readonly unknown[],
  checkout: unknown = { sha: "1111111111111111111111111111111111111111", dirty: false },
): LoadedResults => ({
  path: `outputs/verify/${environment}/${date}_${family}/results.json`,
  date,
  document: parseResultsDocument({
    schema: 1,
    family,
    environment: { key: environment, runtime: { name: "deno" } },
    checkout,
    startedAt: `${date}T00:00:00.000Z`,
    cases,
  }),
});

const passing = (id: string, actual?: string): Record<string, unknown> => ({
  id,
  status: "pass",
  elapsedMs: 1,
  ...(actual === undefined ? {} : { actual }),
});

describe("boundary check of results.json", () => {
  it("throws on a document whose schema is not 1", () => {
    assertThrows(
      () => parseResultsDocument({ schema: 2, family: "anima" }, "a.json"),
      Error,
      "a.json: schema is not 1 (2)",
    );
  });

  it("throws on a document missing a required field (does not fall back to an empty matrix)", () => {
    assertThrows(
      () =>
        parseResultsDocument({
          schema: 1,
          family: "anima",
          environment: { key: "deno-a" },
          checkout: null,
          startedAt: "2026-09-21T00:00:00.000Z",
        }),
      Error,
      "cases is not an array",
    );
    assertThrows(
      () =>
        parseResultsDocument({
          schema: 1,
          family: "anima",
          environment: {},
          checkout: null,
          startedAt: "2026-09-21T00:00:00.000Z",
          cases: [],
        }),
      Error,
      "environment: key is not a string",
    );
    assertThrows(
      () => seat("deno-a", "2026-09-21", "anima", [{ id: "x", status: "ok", elapsedMs: 1 }]),
      Error,
      "unknown status 'ok'",
    );
  });

  it("throws on a document with a duplicate case ID (the second outcome does not vanish silently)", () => {
    assertThrows(
      () =>
        seat("deno-a", "2026-09-21", "anima", [
          passing("base"),
          { id: "base", status: "fail", elapsedMs: 2 },
        ]),
      Error,
      "duplicate case ID 'base'",
    );
    // With different IDs both items stay (only duplicates are rejected).
    const loaded = seat("deno-a", "2026-09-21", "anima", [passing("base"), passing("extra")]);
    assertEquals(
      loaded.document.cases.map((one) => one.id),
      ["base", "extra"],
    );
  });

  it("reads measurements (does not fail when a non-finite value arrives as null)", () => {
    const loaded = seat("deno-a", "2026-09-21", "golden", [
      {
        id: "activations",
        status: "pass",
        elapsedMs: 3,
        measurements: [
          {
            output: "y",
            maxAbs: 0.00002,
            maxRel: null,
            tolerance: { atol: 0.00048828125, rtol: 0 },
            stage: "spec",
          },
        ],
      },
    ]);
    assertEquals(loaded.document.cases[0].measurements, [
      {
        output: "y",
        maxAbs: 0.00002,
        maxRel: null,
        tolerance: { atol: 0.00048828125, rtol: 0 },
        stage: "spec",
      },
    ]);
    // They also go into the matrix as is (what --json and the small measurement table read).
    const report = buildDiff([loaded]);
    assertEquals(report.families[0].rows[0].cells["deno-a"].measurements?.[0].maxRel, null);
  });

  it("reads comparisons (accepts a non-finite null and a run without a band, throws on a broken band)", () => {
    const banded = {
      output: "step1-latent",
      reference: "f16+dit8",
      practical: "f16+dit8-a8-attn8-s16",
      relRms: 0.012,
      maxAbs: 0.31,
      band: { metric: "relRms" as const, floor: 0.001, ceiling: 0.025 },
    };
    // A run without a band has no field at all (the writer's JSON.stringify drops undefined fields).
    const { band: _dropped, ...unbanded } = banded;
    const loaded = seat("deno-a", "2026-09-26", "anima-ab", [
      {
        id: "f16+dit8-a8-attn8-s16-1024",
        status: "fail",
        elapsedMs: 3,
        comparisons: [banded, { ...unbanded, output: "m_p", relRms: null }],
      },
    ]);
    assertEquals(loaded.document.cases[0].comparisons, [
      banded,
      { ...unbanded, output: "m_p", relRms: null },
    ]);
    const report = buildDiff([loaded]);
    assertEquals(report.families[0].rows[0].cells["deno-a"].comparisons?.[0].relRms, 0.012);

    const broken = (comparison: Record<string, unknown>) => () =>
      seat("deno-a", "2026-09-26", "anima-ab", [
        {
          id: "x",
          status: "pass",
          elapsedMs: 1,
          comparisons: [comparison],
        },
      ]);
    assertThrows(
      broken({ ...banded, band: { ...banded.band, metric: "maxAbs" } }),
      Error,
      "unknown metric 'maxAbs'",
    );
    assertThrows(
      broken({ ...banded, band: { metric: "relRms", floor: 0.001 } }),
      Error,
      "ceiling is not a number",
    );
    assertThrows(broken({ ...banded, practical: 3 }), Error, "practical is not a string");
    assertThrows(
      () =>
        seat("deno-a", "2026-09-26", "anima-ab", [
          {
            id: "x",
            status: "pass",
            elapsedMs: 1,
            comparisons: {},
          },
        ]),
      Error,
      "comparisons is not an array",
    );
  });
});

describe("picking results directories", () => {
  it("takes the newest date when the same family and environment have several dates", () => {
    const report = buildDiff([
      seat("deno-a", "2026-09-20", "anima", [passing("base")]),
      seat("deno-a", "2026-09-21", "anima", [passing("base"), passing("extra")]),
    ]);
    assertEquals(report.families[0].selected, [
      {
        environment: "deno-a",
        date: "2026-09-21",
        path: "outputs/verify/deno-a/2026-09-21_anima/results.json",
        checkout: { sha: "1111111111111111111111111111111111111111", dirty: false },
      },
    ]);
    assertEquals(
      report.families[0].rows.map((row) => row.id),
      ["base", "extra"],
    );
  });

  it("with --date takes only that date, and an environment without it becomes absent", () => {
    const report = buildDiff(
      [
        seat("deno-a", "2026-09-20", "anima", [passing("base")]),
        seat("deno-a", "2026-09-21", "anima", [passing("base")]),
        seat("deno-b", "2026-09-21", "anima", [passing("base")]),
      ],
      { date: "2026-09-20" },
    );
    assertEquals(report.families[0].environments, ["deno-a"]);
    assertEquals(report.families[0].selected[0].date, "2026-09-20");
    assertEquals(report.families[0].absent, ["deno-b"]);
  });

  it("warns when no results directory matches --date (no warning when one matches)", () => {
    const loaded = [seat("deno-a", "2026-09-21", "anima", [passing("base")])];
    const missed = buildDiff(loaded, { date: "2026-02-30" });
    assertEquals(missed.warnings, ["no results directory matches --date 2026-02-30"]);
    assertEquals(missed.families, []);
    assertEquals(buildDiff(loaded, { date: "2026-09-21" }).warnings, []);
  });

  it("warns with both paths when two results directories share family, environment key and date", () => {
    const first = seat("deno-a", "2026-09-21", "anima", [passing("base")]);
    // Two machines with the same GPU model report the same environment key, so a hand copy puts
    // the same results directory side by side under another path.
    const second: LoadedResults = {
      ...seat("deno-a", "2026-09-21", "anima", [{ id: "base", status: "fail", elapsedMs: 2 }]),
      path: "outputs/verify/copied/2026-09-21_anima/results.json",
    };
    const report = buildDiff([first, second]);
    assertEquals(report.warnings, [
      "two results directories with the same family, environment key and date (anima / deno-a / 2026-09-21) — only one is taken: " +
        `${first.path} / ${second.path}`,
    ]);
    // Still only one is taken (the warning is there to tell the reader "which one vanished").
    assertEquals(
      report.families[0].selected.map((one) => one.path),
      [first.path],
    );
    // With different dates the newest is taken, so no warning.
    assertEquals(
      buildDiff([
        seat("deno-a", "2026-09-20", "anima", [passing("base")]),
        seat("deno-a", "2026-09-21", "anima", [passing("base")]),
      ]).warnings,
      [],
    );
  });

  it("builds matrices only for the families given by --family, and warns on a filter that matches none", () => {
    const report = buildDiff(
      [
        seat("deno-a", "2026-09-21", "anima", [passing("base")]),
        seat("deno-a", "2026-09-21", "sbv2", [passing("jvnv")]),
      ],
      { families: ["sbv2", "irodori"] },
    );
    assertEquals(
      report.families.map((matrix) => matrix.family),
      ["sbv2"],
    );
    assertEquals(report.warnings, ["no results match --family irodori"]);
  });
});

describe("differences between environments", () => {
  it("has no differences when every environment agrees", () => {
    const report = buildDiff([
      seat("deno-a", "2026-09-21", "anima", [passing("base", "aa")]),
      seat("deno-b", "2026-09-21", "anima", [passing("base", "aa")]),
    ]);
    assertEquals(report.families[0].differences, []);
    assertEquals(report.families[0].environments, ["deno-a", "deno-b"]);
  });

  it("lists a case whose status differs", () => {
    const report = buildDiff([
      seat("deno-a", "2026-09-21", "anima", [passing("base")]),
      seat("deno-b", "2026-09-21", "anima", [{ id: "base", status: "fail", elapsedMs: 2 }]),
    ]);
    assertEquals(report.families[0].differences, [
      {
        caseId: "base",
        kind: "status",
        values: { "deno-a": "pass", "deno-b": "fail" },
      },
    ]);
  });

  it("lists a case whose artifact sha differs (not red = only listed as a difference)", () => {
    const report = buildDiff([
      seat("deno-a", "2026-09-21", "anima", [passing("base", "aa")]),
      seat("deno-b", "2026-09-21", "anima", [passing("base", "bb")]),
    ]);
    assertEquals(report.families[0].differences, [
      {
        caseId: "base",
        kind: "actual",
        values: { "deno-a": "aa", "deno-b": "bb" },
      },
    ]);
  });

  it("lists a case only one side has", () => {
    const report = buildDiff([
      seat("deno-a", "2026-09-21", "anima", [passing("base"), passing("only-a")]),
      seat("deno-b", "2026-09-21", "anima", [passing("base")]),
    ]);
    assertEquals(report.families[0].differences, [
      {
        caseId: "only-a",
        kind: "missing",
        values: { "deno-a": "pass" },
        missing: ["deno-b"],
      },
    ]);
  });
});

describe("checkout warnings", () => {
  it("warns when the sha of the selected results differs between environments", () => {
    const report = buildDiff([
      seat("deno-a", "2026-09-21", "anima", [passing("base")]),
      seat("deno-b", "2026-09-21", "anima", [passing("base")], {
        sha: "2222222222222222222222222222222222222222",
        dirty: false,
      }),
    ]);
    assertEquals(report.families[0].warnings, [
      "checkout sha differs between environments (results of different commits side by side): " +
        "deno-a=1111111 / deno-b=2222222",
    ]);
  });

  it("also warns on a dirty working tree and on results without a checkout", () => {
    const report = buildDiff([
      seat("deno-a", "2026-09-21", "anima", [passing("base")], {
        sha: "1111111111111111111111111111111111111111",
        dirty: true,
      }),
      seat("deno-b", "2026-09-21", "anima", [passing("base")], null),
    ]);
    assertEquals(report.families[0].warnings, [
      "deno-a: results taken from a checkout with a dirty working tree（1111111）",
      "deno-b: no checkout (results taken on a machine without git)",
    ]);
  });
});

describe("CLI arguments", () => {
  it("--json takes no value and does not shift the later pairs", () => {
    const args = parseArgs(["--json", "--family", "anima", "--family", "sbv2"]);
    assertEquals(args.flags.has("json"), true);
    assertEquals(args.values.get("family"), ["anima", "sbv2"]);
  });

  it("fails on an unknown option, a missing value and a key given twice", () => {
    assertThrows(() => parseArgs(["--families", "anima"]), Error, "unknown option --families");
    assertThrows(() => parseArgs(["--date"]), Error, "is not a '--key value' pair");
    assertThrows(
      () => parseArgs(["--date", "--json"]),
      Error,
      "has no value ('--json' is an option)",
    );
    assertThrows(
      () => single(parseArgs(["--date", "2026-09-20", "--date", "2026-09-21"]), "date"),
      Error,
      "--date can be given only once",
    );
  });
});
