// The GEMV derivation's rule of one steady record (src/tune/gemv-derive.ts with
// src/tune/gemv-one-record.ts), without a GPU: synthetic records of the current runtime
// (tests/helpers/gemv-sweep-records.ts) whose passes are 100 ms at the base time and whose rounds
// have a known gap, padded with cases of M = 1 alone to 20 groups so that they have a class.
//
// What this pins, rule point by rule point: ① a record's class (20 groups, 20 % beyond ×1.05) decides
// which rule reads it; ② the reference is the same group's row, its time unrepeated; ③ at M = 1 a
// gain clears margin × gap × gap, and the output carries every candidate's readings and break-even
// margin; ④ at M = 4 and 8 the derived row's kernel has a time in the same record and is not slower
// than the default's choice (no margin); ⑤ a row block clears margin × gaps at one row count and is
// not slower at any; ⑥ the largest gain, and the table's choice kept unless beaten by the bar in the
// same group (not moved by a record that did not time it); ⑦ an unsteady record adopts nothing
// alone, two adopt by the rule of two records; ⑧ a pass under 20 ms is no evidence; ⑨ several
// steady records: the decision holds in every one that times it; ⑩ a re-decision under another
// margin from the output alone, equal to a derivation at that margin; and the re-timed reference.
//
// Faults injected: 19 groups; 5 of 20 groups beyond ×1.05; a gain under its bar (×1.08 with gaps of
// 1.01); a kernel ×0.99 the default's choice at M = 4; a record without M = 8; a row block ×1.05 the
// candidate's own kernel; a challenger ×1.04 over the table's choice; the table's choice screened out;
// the table's choice slower than the default's; one record of an unsteady device; one unsteady or
// classless record against a table that moved PLE's choices (the default's written back), and the same
// record where those choices failed their checks (kept for lack of evidence); a pass of 10 ms; a
// second steady record at ×1.05; a re-timed reference ×0.9 its rounds' minimum.

import { assert, assertEquals, assertStringIncludes, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import {
  deriveLinearGemvChoices,
  type GemvSweepInput,
  type LinearGemvDecision,
  type LinearGemvDerivation,
  planGemvMeasurements,
  redecideLinearGemvChoices,
} from "../tune.ts";
import { DEFAULT_GEOMETRY_PROFILE, type GeometryProfile } from "../src/kernels/geometry-profile.ts";
import { GEMV_CASES } from "../src/tune/gemv-cases.ts";
import type { GemvSweepReport, GemvSweepRow } from "../src/tune/gemv-report.ts";
import {
  BASE_TIME,
  GEMV_ADAPTER,
  type GemvCaseSpec,
  gemvRecord,
} from "./helpers/gemv-sweep-records.ts";

/** A table of the records' adapter that carries the default's choices. */
const TABLE: GeometryProfile = {
  ...DEFAULT_GEOMETRY_PROFILE,
  id: "test-gpu",
  match: { vendor: GEMV_ADAPTER.vendor, architecture: GEMV_ADAPTER.architecture },
};

/** E2B's per_layer_model_projection: i4 g32, a parallel row at 4 lanes with a row block of 4. */
const PLE = "i4g32-n8960-k1536";

/** A pass of 100 000 dispatches: 100 ms at the base time of 1 µs. Every gap 1.002 unless set. */
const PASSES = { reps: 100_000, gap: 1.002 } as const;

const faster = (gain: number): number => BASE_TIME / gain;

/** The cases besides `used` that pad a record to `count` groups more (M = 1 alone, base times). */
const fillers = (used: readonly string[], count: number): GemvCaseSpec[] =>
  GEMV_CASES.filter(({ id }) => !used.includes(id))
    .slice(0, count)
    .map(({ id }) => ({ caseId: id }));

type RecordShape = {
  /** Groups in all, the specs' own and the padding (default 20). */
  readonly groups?: number;
  /** How many padding groups spread beyond ×1.05 (default none). */
  readonly unsteady?: number;
  readonly gap?: number;
  readonly reps?: number;
  readonly change?: (record: GemvSweepReport) => GemvSweepReport;
};

/** A record of the specs, padded with cases of M = 1 alone to its groups. */
const record = (
  path: string,
  specs: readonly GemvCaseSpec[],
  shape: RecordShape = {},
): GemvSweepInput => {
  const own = specs.reduce((sum, spec) => sum + (spec.rows ?? [1]).length, 0);
  const padding = fillers(
    specs.map(({ caseId }) => caseId),
    Math.max(0, (shape.groups ?? 20) - own),
  ).map((spec, index) => (index < (shape.unsteady ?? 0) ? { ...spec, spread: 1.2 } : spec));
  const report = gemvRecord(`2026-11-28T0${path.length % 10}:00:00.000Z`, [...specs, ...padding], {
    passes: { reps: shape.reps ?? PASSES.reps, gap: shape.gap ?? PASSES.gap },
  });
  return { report: shape.change?.(report) ?? report, path, sha256: `sha-${path}` };
};

const derive = (
  inputs: readonly GemvSweepInput[],
  table: GeometryProfile = TABLE,
  margin?: number,
): LinearGemvDerivation =>
  deriveLinearGemvChoices(inputs, { table, ...(margin === undefined ? {} : { margin }) });

const decisionOf = (
  derivation: LinearGemvDerivation,
  field: LinearGemvDecision["field"],
  caseId: string,
): LinearGemvDecision => {
  const found = derivation.decisions.find(
    (decision) => decision.field === field && decision.cases.includes(caseId),
  );
  assert(found !== undefined, `no ${field} decision of ${caseId}`);
  return found;
};

const sequentialOf = (derivation: LinearGemvDerivation) =>
  decisionOf(derivation, "sequential.shapes", PLE);
const parallelOf = (derivation: LinearGemvDerivation) =>
  decisionOf(derivation, "parallelShapes", PLE);

/** PLE at M = 1 with the times given. */
const ple1 = (times: Readonly<Record<string, number>>): GemvCaseSpec => ({
  caseId: PLE,
  times: { 1: times },
});

/** PLE's exception in a derived field, if any. */
const pleException = (derivation: LinearGemvDerivation) =>
  derivation.linearGemv?.sequential.shapes.find(
    (row) => row.storage === "i4" && row.n === 8960 && row.k === 1536,
  )?.variant;

describe("① the class of a record decides which rule reads it", () => {
  it("adopts from one record of 20 groups of which at most 20 % spread beyond ×1.05, by the rule of one steady record", () => {
    const derivation = derive([record("a.json", [ple1({ c64u8: faster(1.2) })], { unsteady: 4 })]);
    const decision = sequentialOf(derivation);
    assertEquals(
      [decision.path, decision.outcome, decision.derivedChoice],
      ["one-record", "adopted", "c64u8"],
    );
    assertEquals(pleException(derivation), { cols: 64, unroll: 8 });
    const [provenance] = derivation.provenance.records;
    assertEquals(provenance?.steady, true);
    assertEquals(provenance?.steadiness, {
      class: "steady",
      groups: 20,
      unsteadyGroups: 4,
      share: 0.2,
    });
  });

  it("reads a record of 19 groups as having no class: the rule of two records, which adopts nothing from one record (fault: 19 groups)", () => {
    const derivation = derive([record("a.json", [ple1({ c64u8: faster(1.2) })], { groups: 19 })]);
    const decision = sequentialOf(derivation);
    assertEquals([decision.path, decision.outcome], ["two-record", "kept"]);
    assertEquals(derivation.provenance.records[0]?.steadiness.class, "unclassified");
    assertEquals(derivation.linearGemv, undefined);
  });

  it("reads a record with 5 of 20 groups beyond ×1.05 as unsteady: it adopts nothing alone (fault: 25 %)", () => {
    const derivation = derive([record("a.json", [ple1({ c64u8: faster(1.2) })], { unsteady: 5 })]);
    assertEquals(derivation.provenance.records[0]?.steadiness.class, "unsteady");
    assertEquals(
      [sequentialOf(derivation).path, sequentialOf(derivation).outcome],
      ["two-record", "kept"],
    );
  });
});

describe("② and ③ at M = 1: the gain over the same group's reference, against margin × gap × gap", () => {
  it("reads the gain against the reference's row of the same group, whose time no second record repeats", () => {
    // The baseline is slow in this record's group (every other kernel ×1.3 against it): the gain
    // is against it, ×1.3 / 0.9
    const derivation = derive([record("a.json", [ple1({ c32u4: 1300, c64u8: 900 })])]);
    const decision = sequentialOf(derivation);
    assertEquals(decision.derivedChoice, "c64u8");
    assertEquals(
      decision.gains.map(({ gain }) => Number(gain.toFixed(6))),
      [Number((1300 / 900).toFixed(6))],
    );
  });

  it("refuses a gain under its bar, and carries every candidate with a time at M = 1 — its gain, both gaps, its check and its break-even margin (fault: ×1.08 with gaps of 1.01, a bar of ×1.086)", () => {
    const derivation = derive([record("a.json", [ple1({ c64u8: faster(1.08) })], { gap: 1.01 })]);
    const decision = sequentialOf(derivation);
    assertEquals([decision.outcome, decision.derivedChoice], ["kept", "c32u4"]);
    assertStringIncludes(decision.reason ?? "", "break-even");
    const [evidence] = decision.evidence ?? [];
    assert(evidence !== undefined);
    const c64u8 = evidence.candidates.find(({ candidate }) => candidate === "c64u8");
    const [reading] = c64u8?.m1 ?? [];
    assert(reading !== undefined);
    assertEquals([reading.record, reading.gap, reading.referenceGap], ["a.json", 1.01, 1.01]);
    assert(Math.abs(reading.breakEven - 1.08 / (1.01 * 1.01)) < 1e-9);
    assertEquals(c64u8?.ruledOut, undefined);
    // Every timed candidate is there, not only the nearest one
    assert(evidence.candidates.length > 10, `${evidence.candidates.length} candidates`);
    // The control: the same gain with gaps of 1.002 clears ×1.065 × 1.002²
    const control = derive([record("a.json", [ple1({ c64u8: faster(1.08) })])]);
    assertEquals(sequentialOf(control).derivedChoice, "c64u8");
  });

  it("names a failed check in the candidate's evidence, which rules it out whatever its gain", () => {
    const derivation = derive([
      record("a.json", [
        {
          ...ple1({ c64u8: faster(1.5) }),
          fields: { 1: { c64u8: { identicalToBaseline: false } } },
        },
      ]),
    ]);
    const decision = sequentialOf(derivation);
    assertEquals(decision.outcome, "kept");
    const c64u8 = decision.evidence?.[0]?.candidates.find(({ candidate }) => candidate === "c64u8");
    assertStringIncludes(c64u8?.ruledOut ?? "", "words differ");
  });
});

describe("④ and ⑤ a parallel row at the other row counts, and its row block", () => {
  /** PLE at 1, 4 and 8 rows, staged-l8 ×1.3 the default's choice at M = 1. */
  const pleRows = (
    later: Readonly<Record<string, number>> = {},
    rows = [1, 4, 8],
  ): GemvCaseSpec => ({
    caseId: PLE,
    rows,
    times: { 1: { "staged-l8": faster(1.3) }, 4: later, 8: later },
  });

  it("adopts a row whose kernel is not slower than the default's choice at M = 4 and 8 — a gain of 1 is enough, no margin", () => {
    const decision = parallelOf(derive([record("a.json", [pleRows()])]));
    assertEquals(
      [decision.path, decision.derivedChoice, decision.derivedRowBlock],
      ["one-record", "staged-l8", undefined],
    );
  });

  it("keeps the default's row when the row's kernel is ×0.99 the default's choice at M = 4 (fault)", () => {
    const decision = parallelOf(
      derive([record("a.json", [pleRows({ "staged-l8": BASE_TIME * 1.01 })])]),
    );
    assertEquals(decision.derivedChoice, "parallel-l4");
    assertStringIncludes(decision.reason ?? "", "slower than the default's choice");
  });

  it("keeps the default's row when the record that timed it at M = 1 has no time at M = 8 (fault: the record without M = 8)", () => {
    const decision = parallelOf(derive([record("a.json", [pleRows({}, [1, 4])])]));
    assertEquals(decision.derivedChoice, "parallel-l4");
    assertStringIncludes(decision.reason ?? "", "no time at M=8");
  });

  it("gives the row the row block that clears margin × gaps over its own kernel at a row count, and none at ×1.05 (fault)", () => {
    const blocked = parallelOf(
      derive([record("a.json", [pleRows({ "staged-l8r4": faster(1.2) })])]),
    );
    assertEquals([blocked.derivedChoice, blocked.derivedRowBlock], ["staged-l8", 4]);
    const near = parallelOf(derive([record("a.json", [pleRows({ "staged-l8r4": faster(1.05) })])]));
    assertEquals([near.derivedChoice, near.derivedRowBlock], ["staged-l8", undefined]);
    // The output carries the row's kernels under each row block
    const options = near.evidence?.[0]?.candidates.find(
      ({ candidate }) => candidate === "staged-l8",
    )?.rows;
    assertEquals(
      options?.map(({ rowBlock }) => rowBlock),
      [undefined, 2, 4, 8],
    );
    const four = options?.find(({ rowBlock }) => rowBlock === 4)?.at.find(({ m }) => m === 4);
    assertEquals([four?.kernel, four?.block?.reference], ["staged-l8r4", "staged-l8"]);
    assert(
      four?.block?.readings[0] !== undefined && Math.abs(four.block.readings[0].gain - 1.05) < 1e-9,
    );
  });
});

describe("⑥ the largest gain, and the table's choice", () => {
  /** A table whose exception for PLE's shape is c64u8. */
  const WITH_C64U8: GeometryProfile = {
    ...TABLE,
    linearGemv: {
      ...DEFAULT_GEOMETRY_PROFILE.linearGemv,
      sequential: {
        ...DEFAULT_GEOMETRY_PROFILE.linearGemv.sequential,
        shapes: [
          ...DEFAULT_GEOMETRY_PROFILE.linearGemv.sequential.shapes,
          { storage: "i4", n: 8960, k: 1536, variant: { cols: 64, unroll: 8 } },
        ],
      },
    },
  };

  it("takes the candidate whose gain is largest among those that qualify", () => {
    const decision = sequentialOf(
      derive([record("a.json", [ple1({ c64u8: faster(1.2), c128u8: faster(1.3) })])]),
    );
    assertEquals(decision.derivedChoice, "c128u8");
  });

  it("keeps the table's choice unless a challenger beats it by the bar in the same group (fault: ×1.04 over it; control: ×1.125)", () => {
    const kept = sequentialOf(
      derive([record("a.json", [ple1({ c64u8: faster(1.2), c128u8: faster(1.25) })])], WITH_C64U8),
    );
    assertEquals([kept.outcome, kept.derivedChoice, kept.tableChoice], ["kept", "c64u8", "c64u8"]);
    const beaten = derive(
      [record("a.json", [ple1({ c64u8: faster(1.2), c128u8: faster(1.35) })])],
      WITH_C64U8,
    );
    assertEquals(
      [sequentialOf(beaten).outcome, sequentialOf(beaten).derivedChoice],
      ["adopted", "c128u8"],
    );
    assertEquals(pleException(beaten), { cols: 128, unroll: 8 });
  });

  it("does not move the table's choice by a record that did not time it (fault: c64u8 screened out)", () => {
    const decision = sequentialOf(
      derive(
        [
          record("a.json", [
            { ...ple1({ c128u8: faster(1.5) }), fields: { 1: { c64u8: { screened: true } } } },
          ]),
        ],
        WITH_C64U8,
      ),
    );
    assertEquals([decision.outcome, decision.derivedChoice], ["kept", "c64u8"]);
    assertStringIncludes(decision.reason ?? "", "no steady record timed the table's choice");
  });

  it("goes back to the default's choice where it beats the table's by the bar (fault: c64u8 ×0.8 the default's)", () => {
    const derivation = derive([record("a.json", [ple1({ c64u8: faster(0.8) })])], WITH_C64U8);
    assertEquals(
      [sequentialOf(derivation).outcome, sequentialOf(derivation).derivedChoice],
      ["adopted", "c32u4"],
    );
    assertEquals(pleException(derivation), undefined);
  });

  it("keeps a table's choice on a shape no record measured (fault of the rule of two records alone: the field started from the default's)", () => {
    const other = GEMV_CASES.find(({ id }) => id !== PLE && id.startsWith("i8-"));
    assert(other !== undefined);
    // One group, no padding: PLE is no case of the record
    const derivation = derive(
      [record("a.json", [{ caseId: other.id }], { groups: 1 })],
      WITH_C64U8,
    );
    assertEquals(
      derivation.decisions.some(({ cases }) => cases.includes(PLE)),
      false,
    );
    assertEquals(pleException(derivation), { cols: 64, unroll: 8 });
  });
});

describe("⑦ an unsteady record", () => {
  const unsteady = (path: string) => record(path, [ple1({ c64u8: faster(1.2) })], { unsteady: 10 });

  it("adopts nothing alone (fault: one record of an unsteady device), and two adopt by the rule of two records", () => {
    assertEquals(sequentialOf(derive([unsteady("a.json")])).outcome, "kept");
    const two = sequentialOf(derive([unsteady("a.json"), unsteady("bb.json")]));
    assertEquals([two.path, two.outcome, two.derivedChoice], ["two-record", "adopted", "c64u8"]);
  });

  /**
   * A table whose choices for PLE's shape are not the default's: the exception c64u8 and the row
   * staged-l8 with a row block of 8 (the default's: c32u4 and parallel-l4 with a row block of 4).
   */
  const MOVED: GeometryProfile = {
    ...TABLE,
    linearGemv: {
      ...DEFAULT_GEOMETRY_PROFILE.linearGemv,
      sequential: {
        ...DEFAULT_GEOMETRY_PROFILE.linearGemv.sequential,
        shapes: [
          ...DEFAULT_GEOMETRY_PROFILE.linearGemv.sequential.shapes,
          { storage: "i4", n: 8960, k: 1536, variant: { cols: 64, unroll: 8 } },
        ],
      },
      parallelShapes: DEFAULT_GEOMETRY_PROFILE.linearGemv.parallelShapes.map((row) =>
        row.storage === "i4" && row.n === 8960 && row.k === 1536 && row.group === 32
          ? { ...row, lanes: 8, stagedActivations: true, rowBlock: 8 }
          : row,
      ),
    },
  };
  const pleRow = (derivation: LinearGemvDerivation) =>
    (derivation.linearGemv ?? derivation.defaults).parallelShapes.find(
      (row) => row.storage === "i4" && row.n === 8960 && row.k === 1536 && row.group === 32,
    );

  it("leaves the table's choices of a shape that fewer than two records are evidence for, and says so (fault: one unsteady record, and one of no class, wrote the default's back)", () => {
    for (const lone of [unsteady("a.json"), record("a.json", [ple1({})], { groups: 19 })]) {
      const derivation = derive([lone], MOVED);
      const sequential = sequentialOf(derivation);
      assertEquals(
        [sequential.path, sequential.outcome, sequential.derivedChoice, sequential.tableChoice],
        ["two-record", "kept", "c64u8", "c64u8"],
      );
      assertStringIncludes(sequential.reason ?? "", "so the table's choice stays");
      const parallel = parallelOf(derivation);
      assertEquals(
        [parallel.outcome, parallel.derivedChoice, parallel.derivedRowBlock],
        ["kept", "staged-l8", 8],
      );
      assertStringIncludes(parallel.reason ?? "", "so the table's choice stays");
      assertEquals(derivation.linearGemv, MOVED.linearGemv);
      assertStringIncludes(derivation.verdicts.join("\n"), "keeps c64u8 (the default's: c32u4)");
    }
  });

  it("writes the default's choices back where a record rules the table's out, though none is evidence, and says where (fault: the table's choices kept for lack of evidence)", () => {
    const faults: readonly Readonly<Record<string, Partial<GemvSweepRow>>>[] = [
      { c64u8: { identicalToBaseline: false }, "staged-l8": { withinBand: false } },
      { c64u8: { error: "Error: lost" }, "staged-l8": { error: "Error: lost" } },
    ];
    for (const fields of faults) {
      const spec = { ...ple1({ c64u8: faster(1.2) }), fields: { 1: fields } };
      for (const lone of [
        record("a.json", [spec], { unsteady: 10 }),
        record("a.json", [spec], { groups: 19 }),
      ]) {
        const derivation = derive([lone], MOVED);
        const sequential = sequentialOf(derivation);
        assertEquals(
          [sequential.path, sequential.outcome, sequential.derivedChoice, sequential.tableChoice],
          ["two-record", "kept", "c32u4", undefined],
        );
        assertStringIncludes(sequential.reason ?? "", "the table's choice c64u8 is not held (");
        assertStringIncludes(sequential.reason ?? "", "at M=1");
        const parallel = parallelOf(derivation);
        assertEquals(
          [
            parallel.outcome,
            parallel.derivedChoice,
            parallel.derivedRowBlock,
            parallel.tableChoice,
          ],
          ["kept", "parallel-l4", 4, undefined],
        );
        assertStringIncludes(parallel.reason ?? "", "the table's choice staged-l8 is not held (");
        assertEquals(pleException(derivation), undefined);
        assertEquals([pleRow(derivation)?.lanes, pleRow(derivation)?.rowBlock], [4, 4]);
      }
    }
  });

  it("still writes the default's choice where two records are evidence and keep it (control: ×1.05 in both, under ×1.10)", () => {
    const near = (path: string) => record(path, [ple1({ c64u8: faster(1.05) })], { unsteady: 10 });
    const derivation = derive([near("a.json"), near("bb.json")], MOVED);
    const sequential = sequentialOf(derivation);
    assertEquals(
      [sequential.path, sequential.outcome, sequential.derivedChoice, sequential.tableChoice],
      ["two-record", "kept", "c32u4", undefined],
    );
    assertEquals(pleException(derivation), undefined);
    // Its parallel candidates are evidence in both records too (at the default's time): the default's row
    assertEquals([pleRow(derivation)?.lanes, pleRow(derivation)?.rowBlock], [4, 4]);
  });
});

describe("⑧ a pass shorter than 20 ms", () => {
  it("is no evidence (fault: 10 000 dispatches a pass, 10 ms)", () => {
    const decision = sequentialOf(
      derive([record("a.json", [ple1({ c64u8: faster(1.5) })], { reps: 10_000 })]),
    );
    assertEquals([decision.path, decision.outcome], ["one-record", "kept"]);
    assertEquals(decision.evidence?.[0]?.candidates.length, 0);
  });
});

describe("⑨ several steady records", () => {
  it("adopt only what holds in every steady record that times it (fault: ×1.05 in the second)", () => {
    const first = record("a.json", [ple1({ c64u8: faster(1.2) })]);
    assertEquals(
      sequentialOf(derive([first, record("bb.json", [ple1({ c64u8: faster(1.05) })])])).outcome,
      "kept",
    );
    const both = sequentialOf(derive([first, record("bb.json", [ple1({ c64u8: faster(1.15) })])]));
    assertEquals([both.outcome, both.gains.length], ["adopted", 2]);
  });
});

describe("the re-timed reference", () => {
  /** PLE's baseline at M = 1 re-timed at `ratio` its rounds' minimum. */
  const retimed = (ratio: number) =>
    record("a.json", [
      {
        ...ple1({ c64u8: faster(1.2) }),
        fields: {
          1: { c32u4: { retimedPerDispatch: BASE_TIME * ratio } satisfies Partial<GemvSweepRow> },
        },
      },
    ]);

  it("makes a group no evidence when it departs from the rounds' minimum by more than the bar (fault: ×0.9; control: ×0.98)", () => {
    const departed = sequentialOf(derive([retimed(0.9)]));
    assertEquals(departed.outcome, "kept");
    assertStringIncludes(departed.reason ?? "", "re-timing departs");
    assertEquals(sequentialOf(derive([retimed(0.98)])).derivedChoice, "c64u8");
  });

  it("makes a group no evidence when the re-timed pass read no time, while a row of a record before the re-timing is read untested (fault: the marker read as no re-timing at all)", () => {
    for (const unread of ["clamped", "zero"] as const) {
      const derivation = derive([
        record("a.json", [
          {
            ...ple1({ c64u8: faster(1.2) }),
            fields: { 1: { c32u4: { retimingUnread: unread } satisfies Partial<GemvSweepRow> } },
          },
        ]),
      ]);
      const decision = sequentialOf(derivation);
      assertEquals([decision.outcome, decision.derivedChoice], ["kept", "c32u4"], unread);
      assertStringIncludes(decision.reason ?? "", "read no time");
      const reading = decision.evidence?.[0]?.candidates.find(
        ({ candidate }) => candidate === "c64u8",
      )?.m1[0];
      assertEquals(reading?.retime, "unread");
    }
    // The control: neither field (a record written before the re-timing)
    assertEquals(
      sequentialOf(derive([record("a.json", [ple1({ c64u8: faster(1.2) })])])).derivedChoice,
      "c64u8",
    );
  });

  it("refuses a row that has both a re-timed pass and the mark that it read no time", () => {
    const both = record("a.json", [
      {
        ...ple1({ c64u8: faster(1.2) }),
        fields: {
          1: {
            c32u4: {
              retimedPerDispatch: BASE_TIME,
              retimingUnread: "zero",
            } satisfies Partial<GemvSweepRow>,
          },
        },
      },
    ]);
    assertThrows(() => derive([both]), Error, "retimingUnread or retimedPerDispatch, not both");
  });
});

describe("⑩ re-deciding an output under another margin", () => {
  const records = [record("a.json", [ple1({ c64u8: faster(1.06) })])];

  it("changes what lies between the margins, from the output alone, as a derivation at that margin does (break-even ×1.056)", () => {
    const derivation = derive(records);
    assertEquals(sequentialOf(derivation).outcome, "kept");
    const { derivation: again, changed } = redecideLinearGemvChoices(derivation, 1.03);
    assertEquals(changed, [
      {
        field: "sequential.shapes",
        shape: "i4 n8960 k1536",
        cases: [PLE],
        from: { choice: "c32u4" },
        to: { choice: "c64u8" },
      },
    ]);
    assertEquals(again.rule.margin, 1.03);
    const direct = derive(records, TABLE, 1.03);
    assertEquals(again.decisions, direct.decisions);
    assertEquals(again.linearGemv, direct.linearGemv);
    assertEquals(again.verdicts, direct.verdicts);
  });

  it("changes nothing at the derivation's own margin", () => {
    const derivation = derive(records);
    const { derivation: again, changed } = redecideLinearGemvChoices(derivation, 1.065);
    assertEquals(changed, []);
    assertEquals(again, derivation);
  });
});

describe("the plan of what the records lack (planGemvMeasurements)", () => {
  /** PLE at 1, 4 and 8 rows in a record of `groups` groups. */
  const plePlan = (groups: number) =>
    planGemvMeasurements([record("a.json", [{ caseId: PLE, rows: [1, 4, 8] }], { groups })], {
      table: TABLE,
    }).missing.filter(({ caseId }) => caseId === PLE);

  it("counts one steady record as enough for every candidate it timed (fault: no class, 19 groups — then two are needed)", () => {
    assertEquals(plePlan(20), []);
    const lacking = plePlan(19);
    assertEquals(
      lacking.map(({ m }) => m),
      [1, 4, 8],
    );
  });

  /** A steady record that times PLE at M = 1 alone, staged-l8 ×1.3 the default's choice there. */
  const steadyAtOne = () =>
    record("s.json", [{ caseId: PLE, times: { 1: { "staged-l8": faster(1.3) } } }]);
  /** A record of a few groups, no padding: no class (a batch step's record). */
  const small = (path: string, specs: readonly GemvCaseSpec[]) =>
    record(path, specs, { groups: 0 });
  const ofCase = <Entry extends { readonly caseId: string }>(
    entries: readonly Entry[],
    id: string,
  ) => entries.filter(({ caseId }) => caseId === id);

  it("does not count records of no class toward a decision of the rule of one steady record, and says only a steady record fills it (fault: two classless records of PLE at M = 4 and 8 counted as enough)", () => {
    const inputs = [
      steadyAtOne(),
      small("u1.json", [{ caseId: PLE, rows: [1, 4, 8] }]),
      small("u22.json", [{ caseId: PLE, rows: [1, 4, 8] }]),
    ];
    const planning = planGemvMeasurements(inputs, { table: TABLE });
    assertEquals(ofCase(planning.missing, PLE), []);
    assertEquals(
      ofCase(planning.missingSteady, PLE).map(({ m }) => m),
      [4, 8],
    );
    // The derivation agrees: a steady record timed PLE at M = 1, so the steady records decide
    const parallel = parallelOf(derive(inputs));
    assertEquals([parallel.path, parallel.outcome], ["one-record", "kept"]);
    assertStringIncludes(parallel.reason ?? "", "staged-l8 has no time at M=4 in s.json");
  });

  it("leaves after its batch has run exactly what it said only a steady record fills", () => {
    // A case beside the steady record's own (PLE and its padding), of another sequential key
    const other = GEMV_CASES.at(-1);
    assert(other !== undefined);
    const steady = steadyAtOne();
    const before = [steady, small("u1.json", [{ caseId: other.id }])];
    const steadyCases = new Set(
      planGemvMeasurements([steady], { table: TABLE }).missingSteady.map(({ caseId }) => caseId),
    );
    assert(!steadyCases.has(other.id));
    const planned = planGemvMeasurements(before, { table: TABLE });
    const rows = ofCase(planned.missing, other.id).map(({ m }) => m);
    assert(rows.includes(1), `the plan lists ${other.id} at M=${rows.join(", ")}`);
    assertEquals(
      ofCase(planned.missingSteady, PLE).map(({ m }) => m),
      [4, 8],
    );
    // The batch runs: its steps' records, twice (the steps measure more candidates than they name)
    const after = [
      ...before,
      small("p333.json", [{ caseId: other.id, rows }]),
      small("p4444.json", [{ caseId: other.id, rows }]),
    ];
    const replanned = planGemvMeasurements(after, { table: TABLE });
    assertEquals(ofCase(replanned.missing, other.id), []);
    assertEquals(replanned.missingSteady, planned.missingSteady);
    const derivation = derive(after);
    for (const field of ["sequential.shapes", "parallelShapes"] as const) {
      const decision = decisionOf(derivation, field, other.id);
      assertEquals(decision.path, "two-record");
      assert(
        !(decision.reason ?? "").includes("so the table's choice stays"),
        `${field}: ${decision.reason ?? ""}`,
      );
    }
    assertStringIncludes(
      parallelOf(derivation).reason ?? "",
      "staged-l8 has no time at M=4 in s.json",
    );
  });
});
