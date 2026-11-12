// The check of the entry points of the public surface `@karume/runtime/tune` (ADR 0117 decision 2)
// — runGeometrySweep (takes a dedicated device, sweeps and returns the record) and
// deriveGeometryProfile (the table and the adoption rows from records). The entry points come from
// the public surface (../tune.ts); only the comparison counterpart (the generator internals that
// the CLI / GPU lab go through) comes from src. The GPU sections are SKIPped explicitly in an
// environment without an adapter.

import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import {
  deriveGeometryProfile,
  type GeometrySweepCaseSummary,
  type GeometrySweepOptions,
  type GeometrySweepProgress,
  type GeometrySweepReport,
  type GeometrySweepRow,
  runGeometrySweep,
} from "../tune.ts";
import {
  buildGeometryProfile,
  deriveProfile,
  parseSweepReport,
  verdictLines,
} from "../src/tune/derive.ts";
import { timestampUnitFor } from "../src/tune/sweep.ts";
import { sweepCaseEntries } from "../src/tune/cases.ts";
import { familiesOf } from "../src/tune/families.ts";
import {
  candidateSetCaseSetIds,
  candidateSetDefaultKernelsIds,
  sweepCandidateKernelsId,
} from "../src/tune/fingerprint.ts";
import { A, B, BIG, caseRows, linearCase, report } from "./helpers/sweep-records.ts";

const adapter = navigator.gpu === undefined ? null : await navigator.gpu.requestAdapter();
const timestampQuery = adapter !== null && adapter.features.has("timestamp-query");

/**
 * The checks of the fields added to the record (ADR 0117 decision 8) that do not depend on what was
 * swept: the start time is ISO 8601 and not after the end (`date`); the case-set versions and the
 * kernel fingerprints of the default table are the current runtime's values for exactly the op
 * families of the planned cases (ADR 0117 addendum 2026-11-12); the fingerprint of the measured
 * candidates equals the value re-derived from the record's rows with the generator's own function (a
 * writer-reader round trip); the wall-clock time of each case is finite and non-negative.
 */
const assertRecordFields = (swept: GeometrySweepReport): void => {
  assertEquals(new Date(swept.startedAt).toISOString(), swept.startedAt);
  assert(swept.startedAt <= swept.date, `startedAt ${swept.startedAt} > date ${swept.date}`);
  const { candidateSet, ops, cases } = swept.settings;
  const planned = familiesOf(
    sweepCaseEntries(candidateSet).filter(
      ({ id, op }) => ops.includes(op) && (cases === undefined || cases.includes(id)),
    ),
  );
  assertEquals(swept.caseSets, candidateSetCaseSetIds(candidateSet, planned));
  assertEquals(swept.defaultKernels, candidateSetDefaultKernelsIds(candidateSet, planned));
  assertEquals(Object.keys(swept.caseSets), [...planned]);
  assertEquals(swept.candidateKernels, sweepCandidateKernelsId(swept.rows, swept.dp4a));
  for (const summary of swept.cases) {
    assert(
      Number.isFinite(summary.elapsedMs) && summary.elapsedMs >= 0,
      `${summary.caseId}: elapsedMs ${summary.elapsedMs}`,
    );
  }
};

describe("deriveGeometryProfile", () => {
  const inputs = [
    {
      report: report(
        caseRows(linearCase(1024), BIG, [
          [A, { speedup: 1.2 }],
          [B, { speedup: 1.01 }],
        ]),
      ),
      path: "a.json",
      sha256: "sha-a",
    },
  ];

  it("returns the table and the adoption rows from the records, equal to the CLI / GPU lab path (deriveProfile → buildGeometryProfile)", () => {
    const options = { id: "test-gpu", vendor: "apple", architecture: "metal-3" } as const;
    const { profile, verdicts } = deriveGeometryProfile(inputs, options);
    const sources = inputs.map(({ report, path, sha256 }) =>
      parseSweepReport(report, { path, sha256 }),
    );
    const internal = deriveProfile(sources, { ...options, minSpeedup: 1.05 });
    assertEquals(profile, buildGeometryProfile(options, sources, internal));
    assertEquals(verdicts, verdictLines(internal));
    assertEquals(profile.gemmRows.at(-1)?.geometry, A);
    assertEquals(profile.provenance?.families.gemm?.sweep, "a.json");
  });

  it("minSpeedup defaults to ×1.05 (a ×1.04 win is not adopted), and values below 1 or non-finite throw", () => {
    const close = [
      {
        report: report(caseRows(linearCase(1024), BIG, [[A, { speedup: 1.04 }]])),
        path: "a.json",
        sha256: "sha-a",
      },
    ];
    const largest = (minSpeedup?: number) =>
      deriveGeometryProfile(close, {
        id: "x",
        optIn: true,
        ...(minSpeedup === undefined ? {} : { minSpeedup }),
      }).profile.gemmRows.at(-1)?.geometry;
    assertEquals(largest(), BIG);
    assertEquals(largest(1.03), A);
    for (const minSpeedup of [0.99, Number.NaN, Number.POSITIVE_INFINITY]) {
      assertThrows(() => largest(minSpeedup), Error, "minSpeedup must be a number of at least 1");
    }
  });

  it("throws a record that fails the check (a sweep timed by the wall clock) with the generator's reason as is", () => {
    const wall = [
      {
        report: {
          ...report(caseRows(linearCase(1024), BIG, [])),
          gpuTiming: { feature: false, unit: "wall", quantized: false },
        },
        path: "wall.json",
        sha256: "sha-wall",
      },
    ];
    assertThrows(
      () => deriveGeometryProfile(wall, { id: "x", optIn: true }),
      Error,
      'wall.json: gpuTiming.unit is "wall"',
    );
  });
});

describe("runGeometrySweep: options validation (before touching the GPU)", () => {
  // The check of values from outside the types (JS callers, UI strings) — what JSON.parse returns
  // goes through no type check
  const untyped = (json: string): GeometrySweepOptions => JSON.parse(json);

  it("throws on an unknown candidate set, op or case id and on rounds that is not a positive integer (does not run with the default)", async () => {
    await assertRejects(
      () => runGeometrySweep(untyped('{ "candidateSet": "quik" }')),
      Error,
      "Candidate set quik",
    );
    await assertRejects(() => runGeometrySweep(untyped('{ "ops": ["gemv"] }')), Error, "op gemv");
    await assertRejects(
      () => runGeometrySweep({ cases: ["linear-m1-n1-k1"] }),
      Error,
      "Case linear-m1-n1-k1 is not in the shape table",
    );
    for (const rounds of [0, 1.5]) {
      await assertRejects(
        () => runGeometrySweep({ rounds }),
        Error,
        "rounds must be an integer of at least 1",
      );
    }
  });

  it("throws when no case matches both the op and the case id", async () => {
    await assertRejects(
      () => runGeometrySweep({ ops: ["linear"], cases: ["conv2d-c96-512x512"] }),
      Error,
      "No case matches",
    );
  });
});

describe("timestampUnitFor", () => {
  it("is deno-raw-tick for Deno (wgpu does not convert raw ticks) and ns otherwise (Chrome / Dawn)", () => {
    assertEquals(timestampUnitFor("Deno/2.9.6"), "deno-raw-tick");
    assertEquals(
      timestampUnitFor(
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/154.0.0.0 Safari/537.36",
      ),
      "ns",
    );
  });
});

describe("runGeometrySweep: real GPU", () => {
  it.skipIf(adapter === null)(
    "with an already aborted signal makes no rows and returns a record carrying the settings and the adapter",
    async () => {
      const kinds: GeometrySweepProgress["kind"][] = [];
      const swept = await runGeometrySweep({
        candidateSet: "quick",
        ops: ["linear"],
        rounds: 1,
        signal: AbortSignal.abort(),
        onProgress: (progress) => kinds.push(progress.kind),
      });
      assertEquals(swept.format, "karume-geometry-sweep/3");
      assertEquals(kinds, ["started"]);
      assertEquals(swept.aborted, true);
      assertRecordFields(swept);
      assertEquals(swept.rows, []);
      assertEquals(swept.cases, []);
      assertEquals(swept.deviceLost, null);
      assertEquals(swept.userAgent, navigator.userAgent);
      assertEquals(swept.adapter.vendor, adapter?.info.vendor);
      assertEquals(swept.settings.candidateSet, "quick");
      assertEquals(swept.settings.quick, true);
      assertEquals(swept.settings.ops, ["linear"]);
      assertEquals(swept.settings.rounds, 1);
      assert(!("cases" in swept.settings), JSON.stringify(swept.settings));
    },
  );

  it.skipIf(adapter === null)(
    "measuring 1 case with quick puts the default row first, every row matches the default's output, and notifications come in the order started → row → case",
    async () => {
      const id = "linear-m16-n3072-k1024";
      const kinds: GeometrySweepProgress["kind"][] = [];
      const notified: GeometrySweepRow[] = [];
      const summaries: GeometrySweepCaseSummary[] = [];
      const swept = await runGeometrySweep({
        candidateSet: "quick",
        cases: [id],
        rounds: 1,
        onProgress: (progress) => {
          if (progress.kind !== "status") kinds.push(progress.kind);
          if (progress.kind === "row") notified.push(progress.row);
          if (progress.kind === "case") summaries.push(progress.summary);
        },
      });
      assertEquals(swept.settings.cases, [id]);
      // Deno reports raw ticks; every other runtime (Dawn on Node, browsers) reports nanoseconds.
      const unit = navigator.userAgent.startsWith("Deno/") ? "deno-raw-tick" : "ns";
      assertEquals(swept.gpuTiming.unit, timestampQuery ? unit : "wall");
      assert(swept.rows.length >= 2, `${swept.rows.length} rows`);
      assertEquals(swept.rows[0].isDefault, true);
      for (const row of swept.rows) {
        assertEquals(row.caseId, id);
        assertEquals(row.error, undefined, row.error);
      }
      for (const row of swept.rows.slice(1)) {
        assertEquals(row.identicalToDefault, true, row.geometry);
      }
      assertEquals(notified, swept.rows);
      assertEquals(kinds, ["started", ...swept.rows.map(() => "row" as const), "case"]);
      assertEquals(
        swept.cases.map((summary) => summary.caseId),
        [id],
      );
      assert((swept.cases[0].defaultRepeat?.driftRatio ?? 0) > 0);
      assertEquals(swept.aborted, false);
      assertRecordFields(swept);
      // The notified case and the record's cases[] hold the same values (elapsedMs included)
      assertEquals(summaries, swept.cases);
    },
  );
});

describe("runGeometrySweep: real GPU — recording an abort (aborted)", () => {
  const id = "linear-m16-n3072-k1024";

  it.skipIf(adapter === null)(
    "aborting mid-case (right after the default row) sets aborted to true, and that case has no re-measurement but has elapsedMs",
    async () => {
      const abort = new AbortController();
      const swept = await runGeometrySweep({
        candidateSet: "quick",
        cases: [id],
        rounds: 1,
        signal: abort.signal,
        onProgress: (progress) => {
          if (progress.kind === "row") abort.abort();
        },
      });
      assertEquals(swept.aborted, true);
      assertEquals(swept.rows.length, 1);
      assertEquals(swept.rows[0].isDefault, true);
      assertEquals(
        swept.cases.map((summary) => summary.caseId),
        [id],
      );
      assertEquals(swept.cases[0].defaultRepeat, undefined);
      assertRecordFields(swept);
    },
  );

  it.skipIf(adapter === null)(
    "an abort that arrives after everything is measured (at the last case's notification) does not count (aborted is false — the record is complete)",
    async () => {
      const abort = new AbortController();
      const swept = await runGeometrySweep({
        candidateSet: "quick",
        cases: [id],
        rounds: 1,
        signal: abort.signal,
        onProgress: (progress) => {
          if (progress.kind === "case") abort.abort();
        },
      });
      assert(abort.signal.aborted);
      assertEquals(swept.aborted, false);
      assert(swept.cases[0].defaultRepeat !== undefined, "the re-measurement was not finished");
      assertRecordFields(swept);
    },
  );
});

describe("runGeometrySweep: the evaluation set k85 sweeps its own case table (options validation, before touching the GPU)", () => {
  it("refuses a case of the sweep's table under k85, and an op its table does not have", async () => {
    await assertRejects(
      () => runGeometrySweep({ candidateSet: "k85", cases: ["i8a8-linear-m1024-n2048-k2048"] }),
      Error,
      "Case i8a8-linear-m1024-n2048-k2048 is not in the shape table of candidate set k85",
    );
    await assertRejects(
      () => runGeometrySweep({ candidateSet: "k85", ops: ["linear"] }),
      Error,
      "No case matches",
    );
    await assertRejects(
      () => runGeometrySweep({ cases: ["k85-i8a8-linear-m7920-n3072-k3072"] }),
      Error,
      "is not in the shape table of candidate set quick+",
    );
  });
});

describe("runGeometrySweep: the evaluation set k88 sweeps its own case table (options validation, before touching the GPU)", () => {
  it("refuses a case of the sweep's table under k88, a k88 case under another set, and an op its table does not have", async () => {
    await assertRejects(
      () => runGeometrySweep({ candidateSet: "k88", cases: ["linear-m1024-n2048-k2048"] }),
      Error,
      "Case linear-m1024-n2048-k2048 is not in the shape table of candidate set k88",
    );
    await assertRejects(
      () => runGeometrySweep({ cases: ["k88-matmul-m4096-n4096-k4096"] }),
      Error,
      "is not in the shape table of candidate set quick+",
    );
    await assertRejects(
      () => runGeometrySweep({ candidateSet: "k88", ops: ["attention"] }),
      Error,
      "No case matches",
    );
  });
});
