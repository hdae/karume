// The geometry sweep's `explore` set without a GPU (src/tune/geometry-explore.ts, the class mapping
// of src/tune/derive.ts it shares with the derivation, and what the derivation and the fingerprint
// read of an explore record):
//
// - the cases are measured class by class: the classes in the order of their first case, a class's
//   cases in the table's order (fault: the table's own order, which interleaves the gemmRows
//   buckets of linear and matmul);
// - the class of a case is the derivation's: for every case of the table, `sweepCaseSlot` names the
//   field whose verdict lists the case (fault: any other mapping, conv2d's m tile among them);
// - a candidate is dropped for its class by each of four observations in an earlier case of the
//   class — screened, failed, not identical, timed below ×1 — and by a skipped row there (faults:
//   one per reason), never by the default's row, a row of another class or a row without a ratio,
//   and a candidate timed at exactly ×1 is kept (fault: the drop at ≤ 1);
// - the derivation reads a skipped row as no observation (no failure, no mismatch, "not measured"),
//   so a geometry another record timed there is adopted (fault: a skipped row read as a failure);
//   a skipped default row and a row both skipped and screened are refused; an explore record
//   derives a table, and a set that is neither a candidate set nor explore is refused;
// - skipped rows do not enter `candidateKernels`, screened rows do (fault: a skipped row's kernel
//   counted).

import { assert, assertEquals, assertNotEquals, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { deriveGeometryProfile } from "../tune.ts";
import type { GemmGeometry } from "../src/kernels/gemm-geometry.ts";
import { GEOMETRY_PROFILE_FAMILIES } from "../src/kernels/geometry-profile.ts";
import { caseShape, SWEEP_CASES, type SweepCase } from "../src/tune/cases.ts";
import { deriveProfile, parseSweepReport, sweepCaseSlot } from "../src/tune/derive.ts";
import { sweepCandidateKernelsId } from "../src/tune/fingerprint.ts";
import { gemmCandidate } from "../src/tune/geometries.ts";
import {
  EXPLORE_KEEP_SPEEDUP,
  exploreRulesOut,
  geometryCaseClass,
  geometryExploreDropped,
  geometryExploreOrder,
} from "../src/tune/geometry-explore.ts";
import type { SweepRow } from "../src/tune/report.ts";
import {
  A,
  B,
  BIG,
  caseRows,
  familyDefaultRows,
  linearCase,
  OPTIONS,
  rejectionOf,
  report,
  row,
  source,
  verdictOf,
} from "./helpers/sweep-records.ts";

/** The case of the sweep's table with `id`. */
const tableCase = (id: string): SweepCase => {
  const found = SWEEP_CASES.find((sweepCase) => sweepCase.id === id);
  if (found === undefined) throw new Error(`no case ${id}`);
  return found;
};

describe("the explore set's order", () => {
  it("measures the cases class by class: the classes in the order of their first case, a class's cases in the order given (fault: the table's order)", () => {
    // In the table's order: gemmRows > 512, > 512, 257〜512, ≤ 16, then matmul 257〜512, ≤ 16 and
    // bmm ≤ 16 (bucket by M, written by hand)
    const given = [
      "linear-m1024-n2048-k2048",
      "linear-m4096-n2048-k2048",
      "linear-m512-n2048-k1024",
      "linear-m16-n3072-k1024",
      "matmul-m512-n2048-k1024",
      "matmul-m16-n3072-k1024",
      "bmm-b16-m16-n64-k64",
    ].map((id) => ({ id }));
    assertEquals(
      geometryExploreOrder(given).map(({ id }) => id),
      [
        "linear-m1024-n2048-k2048",
        "linear-m4096-n2048-k2048",
        "linear-m512-n2048-k1024",
        "matmul-m512-n2048-k1024",
        "linear-m16-n3072-k1024",
        "matmul-m16-n3072-k1024",
        "bmm-b16-m16-n64-k64",
      ],
    );
  });

  it("puts every case of the table once, each class's cases together", () => {
    const order = geometryExploreOrder(SWEEP_CASES);
    assertEquals(new Set(order.map(({ id }) => id)), new Set(SWEEP_CASES.map(({ id }) => id)));
    assertEquals(order.length, SWEEP_CASES.length);
    const slots = order.map(sweepCaseSlot);
    const started = new Set<string>();
    slots.forEach((slot, index) => {
      if (index > 0 && slots[index - 1] !== slot) {
        assert(!started.has(slot), `${slot} comes back at ${order[index].id}`);
      }
      started.add(slot);
    });
    // The control: the table's own order interleaves the classes, so there is something to reorder
    assertNotEquals(
      order.map(({ id }) => id),
      SWEEP_CASES.map(({ id }) => id),
    );
  });

  it("refuses an id that is no case of the explore set's table", () => {
    assertThrows(
      () => geometryExploreOrder([{ id: "k88-matmul-m4096-n4096-k4096" }]),
      Error,
      "Case k88-matmul-m4096-n4096-k4096 is not in the shape table of the explore set",
    );
  });
});

describe("the class of a case", () => {
  it("is the field whose derivation verdict lists the case, for every case of the table (one mapping)", () => {
    const rows = GEOMETRY_PROFILE_FAMILIES.flatMap(familyDefaultRows);
    const verdicts = deriveProfile([source(rows)], OPTIONS);
    for (const sweepCase of SWEEP_CASES) {
      const listing = verdicts.filter((verdict) => verdict.cases.includes(sweepCase.id));
      assertEquals(
        listing.map(({ slot }) => slot),
        [sweepCaseSlot(sweepCase)],
        sweepCase.id,
      );
    }
  });

  it("is what geometryCaseClass names by the case's id (the public form of the one mapping), and an unknown id throws", () => {
    const rows = GEOMETRY_PROFILE_FAMILIES.flatMap(familyDefaultRows);
    const verdicts = deriveProfile([source(rows)], OPTIONS);
    for (const sweepCase of SWEEP_CASES) {
      const listing = verdicts.filter((verdict) => verdict.cases.includes(sweepCase.id));
      assertEquals(
        listing.map(({ slot }) => slot),
        [geometryCaseClass(sweepCase.id)],
        sweepCase.id,
      );
    }
    assertThrows(() => geometryCaseClass("linear-m3-n3-k3"), Error, "linear-m3-n3-k3");
  });
});

describe("dropping a candidate for its class", () => {
  // Two cases of gemmRows > 512 and one of 257〜512
  const FIRST = "linear-m1024-n2048-k2048";
  const LATER = tableCase("linear-m4096-n2048-k2048");
  const OTHER_CLASS = "linear-m512-n2048-k1024";

  /** A row of `geometry` in case `caseId` with the given measurement fields. */
  const rowOf = (
    caseId: string,
    geometry: GemmGeometry,
    fields: Partial<SweepRow> = {},
  ): SweepRow => {
    const sweepCase = tableCase(caseId);
    const candidate = gemmCandidate(geometry);
    return {
      caseId,
      op: sweepCase.op,
      shape: caseShape(sweepCase),
      censusCount: 1,
      geometry: candidate.name,
      geometryParams: candidate.geometry,
      isDefault: false,
      ...fields,
    };
  };
  const timed = (speedupVsDefault: number): Partial<SweepRow> => ({
    perDispatch: 1,
    speedupVsDefault,
    identicalToDefault: true,
  });
  const G1: GemmGeometry = { regM: 4, regN: 4, wgX: 8, wgY: 8 };
  const G2: GemmGeometry = { regM: 4, regN: 4, wgX: 16, wgY: 16 };
  const droppedBy = (rows: readonly SweepRow[]): string[] => [
    ...geometryExploreDropped(LATER, rows),
  ];

  it("drops a candidate the quick look screened out in an earlier case of the class", () => {
    assertEquals(droppedBy([rowOf(FIRST, A, { screened: true, quickPerDispatch: 2 })]), [
      gemmCandidate(A).name,
    ]);
  });

  it("drops a candidate that failed in an earlier case of the class", () => {
    assertEquals(droppedBy([rowOf(FIRST, A, { error: "pipeline: no" })]), [gemmCandidate(A).name]);
  });

  it("drops a candidate whose output was not the default's in an earlier case of the class, however fast", () => {
    assertEquals(droppedBy([rowOf(FIRST, A, { ...timed(1.5), identicalToDefault: false })]), [
      gemmCandidate(A).name,
    ]);
  });

  it("drops a candidate timed below ×1 in an earlier case of the class, and keeps one timed at exactly ×1 (not slower than the default)", () => {
    assertEquals(EXPLORE_KEEP_SPEEDUP, 1);
    assertEquals(
      droppedBy([
        rowOf(FIRST, A, timed(0.999)),
        rowOf(FIRST, B, timed(1)),
        rowOf(FIRST, G1, timed(1.2)),
      ]),
      [gemmCandidate(A).name],
    );
  });

  it("drops a candidate skipped in an earlier case of the class (ruled out before it)", () => {
    assertEquals(droppedBy([rowOf(FIRST, A, { skipped: true })]), [gemmCandidate(A).name]);
  });

  it("drops nothing by the default's row, a row of another class or a row without a ratio (the default's row failed)", () => {
    const failedDefault = { ...rowOf(FIRST, BIG, { error: "out of memory" }), isDefault: true };
    assertEquals(
      droppedBy([
        failedDefault,
        rowOf(OTHER_CLASS, A, timed(0.5)),
        rowOf(FIRST, G2, { perDispatch: 1 }),
      ]),
      [],
    );
    assertEquals(exploreRulesOut(failedDefault), false);
  });
});

describe("the derivation reads an explore record", () => {
  const skipped = { ...row(linearCase(4096), gemmCandidate(B), false), skipped: true };

  it("takes a skipped row as no observation: B is not adopted on that record alone, as not measured there, and is adopted where another record timed it (fault: B skipped in linear-m4096)", () => {
    const first = source([
      ...caseRows(linearCase(1024), BIG, [[B, { speedup: 1.5 }]]),
      ...caseRows(linearCase(4096), BIG, []),
      skipped,
    ]);
    const alone = verdictOf(deriveProfile([first], OPTIONS), "gemmRows > 512");
    assertEquals(alone.outcome.kind, "default");
    assertEquals(
      rejectionOf(alone, B),
      "not measured in linear-m4096 (the explore set dropped it for the class)",
    );
    const timed = source(
      [
        ...caseRows(linearCase(1024), BIG, [[B, { speedup: 1.5 }]]),
        ...caseRows(linearCase(4096), BIG, [[B, { speedup: 1.4 }]]),
      ],
      "sweep-2.json",
    );
    const both = verdictOf(deriveProfile([first, timed], OPTIONS), "gemmRows > 512");
    assertEquals([both.outcome.kind, both.outcome.geometry], ["adopted", B]);
  });

  it("refuses a skipped default row and a row both skipped and screened", () => {
    const parse = (rows: readonly Record<string, unknown>[]) =>
      parseSweepReport(report(rows), { path: "explore.json", sha256: "sha" });
    const defaultRow = { ...row(linearCase(1024), gemmCandidate(BIG), true), skipped: true };
    assertThrows(() => parse([defaultRow]), Error, "the default row is skipped");
    assertThrows(
      () =>
        parse([
          ...caseRows(linearCase(1024), BIG, []),
          { ...row(linearCase(1024), gemmCandidate(A), false), skipped: true, screened: true },
        ]),
      Error,
      "is both skipped and screened",
    );
  });

  it("derives a table from a record of the explore set, naming the set in the provenance", () => {
    const explore = {
      ...report([...caseRows(linearCase(1024), BIG, [[A, { speedup: 1.2 }]])]),
      settings: { candidateSet: "explore", quick: false },
    };
    const { profile } = deriveGeometryProfile(
      [{ report: explore, path: "explore.json", sha256: "sha-explore" }],
      { id: "explore-gpu", optIn: true },
    );
    assertEquals(profile.gemmRows.at(-1)?.geometry, A);
    assertEquals(profile.provenance?.families.gemm?.candidateSet, "explore");
  });

  it("refuses a candidate set that is neither a candidate set nor explore", () => {
    assertThrows(
      () =>
        parseSweepReport(
          { ...report([]), settings: { candidateSet: "explor", quick: false } },
          { path: "typo.json", sha256: "sha" },
        ),
      Error,
      'settings.candidateSet is none of quick+ / quick / full / k88 / grid ("explor") nor explore',
    );
  });
});

describe("the measured-candidate fingerprint of an explore record", () => {
  it("leaves the skipped rows out and keeps the screened ones (fault: a skipped row's kernel counted)", () => {
    const sweepCase = tableCase("linear-m1024-n2048-k2048");
    const head = {
      caseId: sweepCase.id,
      op: sweepCase.op,
      shape: caseShape(sweepCase),
      censusCount: 1,
    } as const;
    const base: SweepRow[] = [
      { ...head, geometry: gemmCandidate(BIG).name, geometryParams: BIG, isDefault: true },
    ];
    const other: SweepRow = {
      ...head,
      geometry: gemmCandidate(A).name,
      geometryParams: A,
      isDefault: false,
    };
    const skipped: SweepRow = { ...other, skipped: true };
    const screened: SweepRow = { ...other, screened: true };
    assertEquals(
      sweepCandidateKernelsId([...base, skipped], false),
      sweepCandidateKernelsId(base, false),
    );
    assertNotEquals(
      sweepCandidateKernelsId([...base, screened], false),
      sweepCandidateKernelsId(base, false),
    );
  });
});
