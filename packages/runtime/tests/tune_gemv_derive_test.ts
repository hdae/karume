// The GEMV derivation without a GPU (src/tune/gemv-derive.ts — deriveLinearGemvChoices of
// `@karume/runtime/tune`): synthetic GEMV sweep records over the current runtime's cases and plans
// (tests/helpers/gemv-sweep-records.ts), so their fingerprints and default-profile choices are what a
// sweep of the current runtime writes.
//
// What this pins: ① a candidate is adopted only at ×1.10 or more in every group that is evidence,
// groups of two records or more, the one with the largest smallest gain among several; ② an unsteady
// group, a failed check in one record, the packed rows, a shared sequential key whose cases disagree,
// a row count of the parallel range a record did not measure and a regression at one keep the
// default; ③ groups as the unit of evidence — a stale, screened or skipped row is no evidence while the
// record's other groups are, records that each hold half of the cases derive what a complete pair
// derives, and groups that are not current are reported by cause instead of refused; ④ the refusals —
// none and the same record twice, an older format, another adapter, wall-clock or quantized timing
// (one record derives nothing); ⑤ what the field gets (the exception row replaced, added or removed;
// the parallel row replaced in place; nothing when nothing is adopted) and the provenance and lines
// that name the records; ⑥ the plan of what the records lack (planGemvMeasurements); ⑦ a regeneration
// of the tile geometry carries a base table's `linearGemv`.
//
// Faults injected: record 2 at ×1.09 (one record below the bar); a baseline spread of ×1.26 in
// record 2 (unsteady); differing words, and a reorder row outside the band, in record 2; the two cases
// of one sequential key adopting different variants; a parallel candidate ×0.95 at M = 4; record 2
// without M = 8, and records of M = 1 alone, for a parallel row; a row's kernel fingerprint replaced
// (no evidence there, the record's other cases adopt; the plan lists that candidate alone); the
// winner screened out in one record; the parallel winner skipped at M = 4 in one record; a skipped
// row that is a baseline, has a time or is screened (refused); a case's fingerprint replaced, the
// reorder choice of a summary renamed and the baseline's kernel fingerprint replaced (each a stale
// group of its cause); the formats /3, /2 and /1, the unit, the quantized flag and the adapter
// changed; a failed row and an unsteady group (missing in the plan), a screened or skipped row (not
// missing).

import { assert, assertEquals, assertStringIncludes, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import {
  deriveGeometryProfile,
  deriveLinearGemvChoices,
  type GemvSweepInput,
  type LinearGemvDecision,
  type LinearGemvDerivation,
  parseGeometryProfileJson,
  planGemvMeasurements,
  geometryProfileJson,
} from "../tune.ts";
import { DEFAULT_GEOMETRY_PROFILE, type GeometryProfile } from "../src/kernels/geometry-profile.ts";
import type { LinearGemvChoices } from "../src/kernels/linear-gemv.ts";
import { GEMV_CASES } from "../src/tune/gemv-cases.ts";
import { pickRowBlock, type RowBlockAt, rowBlockRefusal } from "../src/tune/gemv-derive.ts";
import type { GemvSweepReport } from "../src/tune/gemv-report.ts";
import {
  BASE_TIME,
  GEMV_ADAPTER,
  type GemvCaseSpec,
  gemvRecord,
} from "./helpers/gemv-sweep-records.ts";
import { familyDefaultRows, report } from "./helpers/sweep-records.ts";

/** A table of the records' adapter (its match and provenance name it). */
const TABLE: GeometryProfile = {
  ...DEFAULT_GEOMETRY_PROFILE,
  id: "test-gpu",
  match: { vendor: GEMV_ADAPTER.vendor, architecture: GEMV_ADAPTER.architecture },
};

const DEFAULTS: LinearGemvChoices = DEFAULT_GEOMETRY_PROFILE.linearGemv;

/** E2B's per_layer_model_projection: i4 g32, a parallel row at 4 lanes, no sequential exception. */
const PLE = "i4g32-n8960-k1536";
/** E2B's i8 output head: the sequential exception c16u4 and a parallel row at 16 lanes. */
const HEAD_I8 = "i8-n262144-k1536";
/** QAT E2B's o (sliding): its default row takes packed activations. */
const PACKED = "i4g2048-n1536-k2048";
/** Two cases of one sequential key (i4 n2048 k1536): g32 and g512. */
const Q_G32 = "i4g32-n2048-k1536";
const Q_G512 = "i4g512-n2048-k1536";

/** The row counts a parallel row serves that the case table has: a record must measure all three. */
const PARALLEL_ROWS = [1, 4, 8];

/** The cases of the table (the plan reads every one). */
const GEMV_CASE_COUNT = GEMV_CASES.length;

/** A time that makes a kernel `gain` times faster than {@link BASE_TIME}. */
const faster = (gain: number): number => BASE_TIME / gain;

const input = (
  path: string,
  specs: readonly GemvCaseSpec[],
  change: (record: GemvSweepReport) => unknown = (record) => record,
): GemvSweepInput => ({
  report: change(gemvRecord(`2026-11-17T0${path.length % 10}:00:00.000Z`, specs)),
  path,
  sha256: `sha-${path}`,
});

const derive = (
  first: readonly GemvCaseSpec[],
  second: readonly GemvCaseSpec[] = first,
  table: GeometryProfile = TABLE,
): LinearGemvDerivation =>
  deriveLinearGemvChoices([input("a.json", first), input("bb.json", second)], { table });

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

const at1 = (times: Readonly<Record<string, number>>): GemvCaseSpec["times"] => ({ 1: times });

describe("deriveLinearGemvChoices: the identical class (sequential.shapes)", () => {
  it("adopts a variant ×1.10 or more faster than the baseline in every record, as the shape's exception", () => {
    const derivation = derive([{ caseId: PLE, times: at1({ c64u8: faster(1.2) }) }]);
    const decision = decisionOf(derivation, "sequential.shapes", PLE);
    assertEquals(decision.outcome, "adopted");
    assertEquals([decision.defaultChoice, decision.derivedChoice], ["c32u4", "c64u8"]);
    assertEquals(
      decision.gains.map(({ record, gain }) => [record, gain.toFixed(3)]),
      [
        ["a.json", "1.200"],
        ["bb.json", "1.200"],
      ],
    );
    assertEquals(derivation.linearGemv?.sequential.shapes, [
      ...DEFAULTS.sequential.shapes,
      { storage: "i4", n: 8960, k: 1536, variant: { cols: 64, unroll: 8 } },
    ]);
    // The other fields stay the default's
    assertEquals(derivation.linearGemv?.parallelShapes, DEFAULTS.parallelShapes);
    assertEquals(derivation.linearGemv?.sequential.variant, DEFAULTS.sequential.variant);
    assertEquals(derivation.linearGemv?.rowsThreadTarget, DEFAULTS.rowsThreadTarget);
  });

  it("keeps the default when one record is below ×1.10 (fault: record 2 at ×1.09)", () => {
    const derivation = derive(
      [{ caseId: PLE, times: at1({ c64u8: faster(1.2) }) }],
      [{ caseId: PLE, times: at1({ c64u8: faster(1.09) }) }],
    );
    const decision = decisionOf(derivation, "sequential.shapes", PLE);
    assertEquals(decision.outcome, "kept");
    assertStringIncludes(decision.reason ?? "", "×1.090 < ×1.100 in [2]");
    assertEquals(decision.candidate, "c64u8");
    assertEquals(derivation.linearGemv, undefined);
  });

  it("takes, among several, the candidate whose smallest gain over the records is largest", () => {
    const derivation = derive(
      [{ caseId: PLE, times: at1({ c64u8: faster(1.5), c128u4: faster(1.3) }) }],
      [{ caseId: PLE, times: at1({ c64u8: faster(1.11), c128u4: faster(1.25) }) }],
    );
    assertEquals(decisionOf(derivation, "sequential.shapes", PLE).derivedChoice, "c128u4");
  });

  it("keeps a shape with an unsteady record (fault: baseline spread ×1.26 in record 2), and takes ×1.25 as steady", () => {
    const times = at1({ c64u8: faster(1.2) });
    const unsteady = derive([{ caseId: PLE, times }], [{ caseId: PLE, times, spread: 1.26 }]);
    const decision = decisionOf(unsteady, "sequential.shapes", PLE);
    assertEquals(decision.outcome, "kept");
    assertStringIncludes(
      decision.reason ?? "",
      "unsteady in [2] at M=1 (baseline spread ×1.260 > ×1.250)",
    );
    const bound = derive([{ caseId: PLE, times }], [{ caseId: PLE, times, spread: 1.25 }]);
    assertEquals(decisionOf(bound, "sequential.shapes", PLE).outcome, "adopted");
  });

  it("keeps the default when the candidate's words differ from the baseline's in one record (fault)", () => {
    const derivation = derive(
      [{ caseId: PLE, times: at1({ c64u8: faster(1.2) }) }],
      [
        {
          caseId: PLE,
          times: at1({ c64u8: faster(1.2) }),
          fields: { 1: { c64u8: { identicalToBaseline: false, differingWords: 3 } } },
        },
      ],
    );
    const decision = decisionOf(derivation, "sequential.shapes", PLE);
    assertEquals(decision.outcome, "kept");
    assertStringIncludes(decision.reason ?? "", "its words differ from the baseline's in [2]");
  });

  it("replaces a key's exception row, and removes it when the general variant wins", () => {
    const replaced = derive([{ caseId: HEAD_I8, times: at1({ c16u8: faster(1.2) }) }]);
    assertEquals(replaced.linearGemv?.sequential.shapes, [
      { storage: "i8", n: 262144, k: 1536, variant: { cols: 16, unroll: 8 } },
    ]);
    const removed = derive([{ caseId: HEAD_I8, times: at1({ c32u4: faster(1.2) }) }]);
    const decision = decisionOf(removed, "sequential.shapes", HEAD_I8);
    assertEquals([decision.defaultChoice, decision.derivedChoice], ["c16u4", "c32u4"]);
    assertEquals(removed.linearGemv?.sequential.shapes, []);
  });

  it("adopts on a key whose cases agree, and keeps the default when they disagree (fault: g32 c64u8, g512 c16u8)", () => {
    const agree = derive([
      { caseId: Q_G32, times: at1({ c64u8: faster(1.2) }) },
      { caseId: Q_G512, times: at1({ c64u8: faster(1.3) }) },
    ]);
    const agreed = decisionOf(agree, "sequential.shapes", Q_G32);
    assertEquals(agreed.cases, [Q_G32, Q_G512]);
    assertEquals([agreed.outcome, agreed.derivedChoice], ["adopted", "c64u8"]);
    assertEquals(agreed.gains.length, 4);
    const disagree = derive([
      { caseId: Q_G32, times: at1({ c64u8: faster(1.2) }) },
      { caseId: Q_G512, times: at1({ c16u8: faster(1.3) }) },
    ]);
    const decision = decisionOf(disagree, "sequential.shapes", Q_G32);
    assertEquals([decision.outcome, decision.derivedChoice], ["kept", "c32u4"]);
    assertStringIncludes(decision.reason ?? "", "the cases that share the key disagree");
    assertStringIncludes(decision.reason ?? "", `${Q_G32} adopts c64u8`);
    assertStringIncludes(decision.reason ?? "", `${Q_G512} adopts c16u8`);
    assertEquals(disagree.linearGemv, undefined);
  });
});

describe("deriveLinearGemvChoices: the reorder class (parallelShapes)", () => {
  it("replaces the shape's row in place with the adopted lanes and staged flag", () => {
    const derivation = derive([
      { caseId: PLE, rows: PARALLEL_ROWS, times: at1({ "staged-l8": faster(1.3) }) },
    ]);
    const decision = decisionOf(derivation, "parallelShapes", PLE);
    assertEquals(
      [decision.outcome, decision.defaultChoice, decision.derivedChoice],
      ["adopted", "parallel-l4", "staged-l8"],
    );
    const index = DEFAULTS.parallelShapes.findIndex(
      ({ storage, n, k, group }) => storage === "i4" && n === 8960 && k === 1536 && group === 32,
    );
    assertEquals(derivation.linearGemv?.parallelShapes[index], {
      storage: "i4",
      n: 8960,
      k: 1536,
      group: 32,
      lanes: 8,
      packedActivations: false,
      stagedActivations: true,
    });
    assertEquals(
      derivation.linearGemv?.parallelShapes.filter((_, at) => at !== index),
      DEFAULTS.parallelShapes.filter((_, at) => at !== index),
    );
    // The sequential side is the identical class's alone
    assertEquals(derivation.linearGemv?.sequential, DEFAULTS.sequential);
  });

  it("gains against the default's choice at reorder, not against the baseline", () => {
    // parallel-l4 (the default's) twice as fast as the baseline; staged-l8 ×1.05 over it
    const derivation = derive([
      {
        caseId: PLE,
        rows: PARALLEL_ROWS,
        times: at1({ "parallel-l4": faster(2), "staged-l8": faster(2.1) }),
      },
    ]);
    const decision = decisionOf(derivation, "parallelShapes", PLE);
    assertEquals(decision.outcome, "kept");
    assertEquals(
      decision.gains.map(({ gain }) => gain.toFixed(3)),
      ["1.050", "1.050"],
    );
  });

  it("keeps the default when the candidate is outside the band in one record (fault)", () => {
    const times = at1({ "staged-l8": faster(1.3) });
    const derivation = derive(
      [{ caseId: PLE, rows: PARALLEL_ROWS, times }],
      [
        {
          caseId: PLE,
          rows: PARALLEL_ROWS,
          times,
          fields: { 1: { "staged-l8": { withinBand: false } } },
        },
      ],
    );
    const decision = decisionOf(derivation, "parallelShapes", PLE);
    assertEquals(decision.outcome, "kept");
    assertEquals(decision.candidate, "staged-l8");
    assertStringIncludes(decision.reason ?? "", "outside the band in [2] at M=1");
  });

  it("keeps the default when the candidate is slower at another row count of the parallel range, with and without a row block (fault: ×0.95 at M = 4)", () => {
    const regressed = derive([
      {
        caseId: PLE,
        rows: PARALLEL_ROWS,
        times: {
          1: { "staged-l8": faster(1.3) },
          // Its own kernel and every row block of its lanes
          4: {
            "staged-l8": faster(0.95),
            "staged-l8r2": faster(0.95),
            "staged-l8r4": faster(0.95),
          },
        },
      },
    ]);
    const decision = decisionOf(regressed, "parallelShapes", PLE);
    assertEquals(decision.outcome, "kept");
    assertStringIncludes(decision.reason ?? "", "slower than the default's choice in [1] at M=4");
    const even = derive([
      { caseId: PLE, rows: PARALLEL_ROWS, times: { 1: { "staged-l8": faster(1.3) } } },
    ]);
    assertEquals(decisionOf(even, "parallelShapes", PLE).outcome, "adopted");
  });

  it("keeps the row when a record did not measure a row count the row serves (fault: record 2 without M = 8; quick records)", () => {
    const times = at1({ c64u8: faster(1.2), "staged-l8": faster(1.3) });
    const partial = derive(
      [{ caseId: PLE, rows: PARALLEL_ROWS, times }],
      [{ caseId: PLE, rows: [1, 4], times }],
    );
    const decision = decisionOf(partial, "parallelShapes", PLE);
    assertEquals([decision.outcome, decision.candidate], ["kept", "staged-l8"]);
    assertStringIncludes(decision.reason ?? "", "M=8 is not measured in [2]");
    // What M = 1 showed is still reported
    assertEquals(
      decision.gains.map(({ gain }) => gain.toFixed(3)),
      ["1.300", "1.300"],
    );
    const quick = derive([{ caseId: PLE, times }]);
    assertStringIncludes(
      decisionOf(quick, "parallelShapes", PLE).reason ?? "",
      "M=4 is not measured in [1]",
    );
    // The sequential exception is read at M = 1 alone, so quick records still decide it
    assertEquals(decisionOf(quick, "sequential.shapes", PLE).outcome, "adopted");
    assertEquals(quick.linearGemv?.parallelShapes, DEFAULTS.parallelShapes);
  });

  it("gives the adopted row the row block that fares best at the other row counts: the largest smallest gain over the default's choice there", () => {
    const derivation = derive([
      {
        caseId: PLE,
        rows: PARALLEL_ROWS,
        times: {
          1: { "staged-l8": faster(1.3) },
          // A cap of 2: ×1.2 and ×1.3. A cap of 4: ×1.5 and ×1.4. A cap of 8: ×1.5 (four rows at
          // M = 4) and ×2 — the largest smallest gain
          4: { "staged-l8r2": faster(1.2), "staged-l8r4": faster(1.5) },
          8: { "staged-l8r2": faster(1.3), "staged-l8r4": faster(1.4), "staged-l8r8": faster(2) },
        },
      },
    ]);
    const decision = decisionOf(derivation, "parallelShapes", PLE);
    assertEquals(
      [decision.outcome, decision.derivedChoice, decision.derivedRowBlock],
      ["adopted", "staged-l8", 8],
    );
    assertEquals(
      decision.rowGains?.map(({ m, gain }) => `M${m} ${gain.toFixed(1)}`),
      ["M4 1.5", "M4 1.5", "M8 2.0", "M8 2.0"],
    );
    const row = derivation.linearGemv?.parallelShapes.find(
      ({ n, k, group }) => n === 8960 && k === 1536 && group === 32,
    );
    assertEquals(row, {
      storage: "i4",
      n: 8960,
      k: 1536,
      group: 32,
      lanes: 8,
      packedActivations: false,
      stagedActivations: true,
      rowBlock: 8,
    });
    assertStringIncludes(derivation.verdicts.join("\n"), "-> staged-l8 with row blocks of 8");
  });

  it("takes no row block that is not faster than the candidate's own kernel at the other row counts (fault: every row block at the default's time)", () => {
    const derivation = derive([
      { caseId: PLE, rows: PARALLEL_ROWS, times: at1({ "staged-l8": faster(1.3) }) },
    ]);
    const decision = decisionOf(derivation, "parallelShapes", PLE);
    assertEquals([decision.outcome, decision.derivedRowBlock], ["adopted", undefined]);
    const row = derivation.linearGemv?.parallelShapes.find(
      ({ n, k, group }) => n === 8960 && k === 1536 && group === 32,
    );
    assertEquals(row !== undefined && "rowBlock" in row, false);
  });

  it("adopts a candidate that is slower at another row count by its own kernel when a row block of its lanes is not", () => {
    const derivation = derive([
      {
        caseId: PLE,
        rows: PARALLEL_ROWS,
        times: {
          1: { "staged-l8": faster(1.3) },
          4: { "staged-l8": faster(0.95), "staged-l8r4": faster(1.2) },
          8: { "staged-l8": faster(0.9), "staged-l8r4": faster(1.1) },
        },
      },
    ]);
    const decision = decisionOf(derivation, "parallelShapes", PLE);
    assertEquals(
      [decision.outcome, decision.derivedChoice, decision.derivedRowBlock],
      ["adopted", "staged-l8", 4],
    );
  });

  it("replaces the row block of a row whose lanes M = 1 keeps when another is slower in no group and ×1.10 faster at a row count", () => {
    const blocks = {
      4: { "staged-l4r2": faster(1.3), "staged-l4r4": faster(2) },
      8: { "staged-l4r2": faster(1.6), "staged-l4r4": faster(3), "staged-l4r8": faster(2.1) },
    };
    // The default's row of the case: 4 lanes, plain, no row block
    const defaultRow = DEFAULTS.parallelShapes.find(
      ({ n, k, group }) => n === 2048 && k === 1536 && group === 512,
    );
    assert(defaultRow !== undefined, "the default table names the case's shape");
    assertEquals(defaultRow.rowBlock, undefined);
    const derivation = derive([{ caseId: Q_G512, rows: PARALLEL_ROWS, times: blocks }]);
    const decision = decisionOf(derivation, "parallelShapes", Q_G512);
    assertEquals(
      [decision.outcome, decision.defaultChoice, decision.derivedChoice],
      ["adopted", "parallel-l4", "parallel-l4"],
    );
    // A cap of 4 (×2 and ×3) and a cap of 8 (×2 and ×2.1) tie on the smallest gain: the lower cap
    assertEquals([decision.defaultRowBlock, decision.derivedRowBlock], [undefined, 4]);
    assertEquals(
      derivation.linearGemv?.parallelShapes.find(
        ({ n, k, group }) => n === 2048 && k === 1536 && group === 512,
      ),
      { ...defaultRow, rowBlock: 4 },
    );
    assertStringIncludes(
      derivation.verdicts.join("\n"),
      "parallel-l4 -> parallel-l4 with row blocks of 4; at the other row counts: ×2.000 [1] M=4",
    );
    /** Record 2 with every row block at `gain` over the default's choice at M = 8. */
    const record2At8 = (gain: number): LinearGemvDerivation =>
      derive(
        [{ caseId: Q_G512, rows: PARALLEL_ROWS, times: blocks }],
        [
          {
            caseId: Q_G512,
            rows: PARALLEL_ROWS,
            times: {
              4: blocks[4],
              8: {
                "staged-l4r2": faster(gain),
                "staged-l4r4": faster(gain),
                "staged-l4r8": faster(gain),
              },
            },
          },
        ],
      );
    // A group below the threshold but not slower does not refuse: M = 4 is ×1.10 faster in every group
    assertEquals(decisionOf(record2At8(1.09), "parallelShapes", Q_G512).outcome, "adopted");
    // Fault: slower than the default's row in one group (×0.95 at M = 8 in record 2) keeps the row
    const slower = record2At8(0.95);
    const kept = decisionOf(slower, "parallelShapes", Q_G512);
    assertEquals(kept.outcome, "kept");
    assertEquals(slower.linearGemv, undefined);
    // The decision says why each other row block did not replace the default's
    const note = kept.notes.find((line) => line.startsWith("no other row block replaces"));
    assertStringIncludes(
      note ?? "",
      "2 rows — slower than the reference's row block in [2] at M=8 (×0.950)",
    );
    assertStringIncludes(
      note ?? "",
      "4 rows — slower than the reference's row block in [2] at M=8 (×0.950)",
    );
    // Fault: faster everywhere but ×1.10 at no row count (×1.09 in every group) keeps the row
    const near = { "staged-l4r2": faster(1.09), "staged-l4r4": faster(1.09) };
    const nowhere = derive([
      {
        caseId: Q_G512,
        rows: PARALLEL_ROWS,
        times: { 4: near, 8: { ...near, "staged-l4r8": faster(1.09) } },
      },
    ]);
    assertEquals(decisionOf(nowhere, "parallelShapes", Q_G512).outcome, "kept");
    assertEquals(nowhere.linearGemv, undefined);
    // A row whose row blocks reach the threshold in no group says nothing about them
    const unmeasured = derive([{ caseId: Q_G512, rows: PARALLEL_ROWS }]);
    assertEquals(
      decisionOf(unmeasured, "parallelShapes", Q_G512).notes.filter((line) =>
        line.startsWith("no other row block"),
      ),
      [],
    );
    // Fault: a record without M = 8 gives a row block no evidence there
    const partial = derive(
      [{ caseId: Q_G512, rows: PARALLEL_ROWS, times: blocks }],
      [{ caseId: Q_G512, rows: [1, 4], times: { 4: blocks[4] } }],
    );
    assertEquals(decisionOf(partial, "parallelShapes", Q_G512).outcome, "kept");
  });

  describe("a default row that carries a row block (E2B's per_layer_model_projection: 4 lanes, a row block of 4)", () => {
    const rowOf = (derivation: LinearGemvDerivation) =>
      derivation.linearGemv?.parallelShapes.find(
        ({ n, k, group }) => n === 8960 && k === 1536 && group === 32,
      );
    const defaultRow = (() => {
      const found = DEFAULTS.parallelShapes.find(
        ({ n, k, group }) => n === 8960 && k === 1536 && group === 32,
      );
      if (found === undefined) throw new Error("the default table has no row of the case");
      return found;
    })();

    it("is the reference: its row-blocked kernel is the default's choice at M = 4 and 8", () => {
      assertEquals(defaultRow.rowBlock, 4);
    });

    it("moves from 4 to 8 on M = 8 alone: at M = 4 both run four rows, which neither refuses nor counts (fault of the earlier rule: ×1.000 at M = 4 refused it)", () => {
      const derivation = derive([
        { caseId: PLE, rows: PARALLEL_ROWS, times: { 8: { "staged-l4r8": faster(1.2) } } },
      ]);
      const decision = decisionOf(derivation, "parallelShapes", PLE);
      assertEquals(
        [decision.outcome, decision.defaultRowBlock, decision.derivedRowBlock],
        ["adopted", 4, 8],
      );
      assertEquals(rowOf(derivation), { ...defaultRow, rowBlock: 8 });
      assertStringIncludes(
        derivation.verdicts.join("\n"),
        "parallel-l4 with row blocks of 4 -> parallel-l4 with row blocks of 8",
      );
      // The row count where both run the same kernel does not enter the comparison among several:
      // 8 rows (×1.2 at M = 8 alone) beat 2 rows (×1.15 at both); counted as ×1.000 at M = 4 they
      // would lose (fault: the shared row count read as a gain)
      const several = derive([
        {
          caseId: PLE,
          rows: PARALLEL_ROWS,
          times: {
            4: { "staged-l4r2": faster(1.15) },
            8: { "staged-l4r2": faster(1.15), "staged-l4r8": faster(1.2) },
          },
        },
      ]);
      assertEquals(decisionOf(several, "parallelShapes", PLE).derivedRowBlock, 8);
      // Fault: ×1.09 at M = 8 is below the threshold at the one row count where the two differ
      const below = derive([
        { caseId: PLE, rows: PARALLEL_ROWS, times: { 8: { "staged-l4r8": faster(1.09) } } },
      ]);
      assertEquals(decisionOf(below, "parallelShapes", PLE).outcome, "kept");
    });

    it("changes nothing when every kernel takes the default's time (fault: a row block dropped or moved at ×1.000)", () => {
      const derivation = derive([{ caseId: PLE, rows: PARALLEL_ROWS }]);
      const decision = decisionOf(derivation, "parallelShapes", PLE);
      assertEquals(
        [decision.outcome, decision.defaultRowBlock, decision.derivedRowBlock],
        ["kept", 4, 4],
      );
      assertEquals(derivation.linearGemv, undefined);
    });

    it("drops the row block only when the row's own kernel is slower in no group and ×1.10 faster at a row count", () => {
      const own = (m4: number, m8: number): LinearGemvDerivation =>
        derive([
          {
            caseId: PLE,
            rows: PARALLEL_ROWS,
            times: { 4: { "parallel-l4": faster(m4) }, 8: { "parallel-l4": faster(m8) } },
          },
        ]);
      const dropped = own(1.2, 1.0);
      assertEquals(
        [
          decisionOf(dropped, "parallelShapes", PLE).outcome,
          decisionOf(dropped, "parallelShapes", PLE).derivedRowBlock,
        ],
        ["adopted", undefined],
      );
      const row = rowOf(dropped);
      assertEquals(row !== undefined && "rowBlock" in row, false);
      // Faults: faster at M = 4 but slower at M = 8, and nowhere ×1.10
      assertEquals(decisionOf(own(1.2, 0.95), "parallelShapes", PLE).outcome, "kept");
      assertEquals(decisionOf(own(1.05, 1.05), "parallelShapes", PLE).outcome, "kept");
    });
  });

  it("gives an adopted candidate no row block that is not ×1.10 faster than the candidate without one at some row count (fault of the earlier rule: any margin attached one)", () => {
    const derivation = derive([
      {
        caseId: Q_G512,
        rows: PARALLEL_ROWS,
        times: {
          1: { "staged-l8": faster(1.3) },
          4: { "staged-l8r2": faster(1.05) },
          8: { "staged-l8r2": faster(1.05) },
        },
      },
    ]);
    const decision = decisionOf(derivation, "parallelShapes", Q_G512);
    assertEquals(
      [decision.outcome, decision.derivedChoice, decision.derivedRowBlock],
      ["adopted", "staged-l8", undefined],
    );
  });

  it("leaves a row whose default takes packed activations as it is", () => {
    const derivation = derive([{ caseId: PACKED, times: at1({ "staged-l16": faster(1.5) }) }]);
    const decision = decisionOf(derivation, "parallelShapes", PACKED);
    assertEquals([decision.outcome, decision.defaultChoice], ["kept", "packed-l32"]);
    assertStringIncludes(decision.reason ?? "", "packed activations");
    assertEquals(derivation.linearGemv, undefined);
  });

  it("writes no row for a subgroup candidate and names it in the decision's notes", () => {
    const derivation = derive([
      { caseId: PLE, rows: PARALLEL_ROWS, times: at1({ "subgroup32-l8": faster(1.4) }) },
    ]);
    const decision = decisionOf(derivation, "parallelShapes", PLE);
    assertEquals(decision.outcome, "kept");
    assertEquals(decision.notes.length, 1);
    assertStringIncludes(decision.notes[0] ?? "", "subgroup32-l8 meets the rule too");
    assertEquals(derivation.linearGemv, undefined);
  });
});

describe("the rule of a parallel row's row block (rowBlockRefusal, pickRowBlock)", () => {
  /** A row count where the row block and the reference run different kernels, with one gain a record. */
  const at = (m: number, ...gains: number[]): RowBlockAt => ({
    m,
    gains: gains.map((gain, index) => ({ label: `[${index + 1}]`, gain })),
    records: gains.length,
  });

  it("lets a row block replace the reference's when it is slower in no group and ×1.10 faster in every group of one row count", () => {
    // 4 → 8, and 8 → 4: only M = 8 is read, since both run four rows at M = 4
    assertEquals(rowBlockRefusal([at(8, 1.2, 1.15)]), undefined);
    assertEquals(rowBlockRefusal([at(8, 1.1, 1.1)]), undefined);
    // None → 4: both row counts are read; one clears the threshold, the other only holds
    assertEquals(rowBlockRefusal([at(4, 1.0, 1.02), at(8, 1.5, 1.4)]), undefined);
  });

  it("refuses one that runs the reference's kernel at every row count, or is nowhere ×1.10 faster (faults: every gain ×1.000; ×1.02 and ×1.03, which the earlier rule took)", () => {
    assertStringIncludes(
      rowBlockRefusal([]) ?? "",
      "the reference's kernel at every other row count",
    );
    assertStringIncludes(
      rowBlockRefusal([at(4, 1, 1), at(8, 1, 1)]) ?? "",
      "at no row count ×1.100 faster than the reference's row block in every group (at most ×1.000)",
    );
    assertStringIncludes(
      rowBlockRefusal([at(4, 1.02, 1.02), at(8, 1.03, 1.03)]) ?? "",
      "at most ×1.030",
    );
    // ×1.10 in one record of a row count and less in the other is not every group of it
    assertStringIncludes(rowBlockRefusal([at(8, 1.3, 1.09)]) ?? "", "at no row count");
  });

  it("refuses one that is slower in a group, however fast elsewhere (fault: ×0.99 at M = 4 beside ×2 at M = 8)", () => {
    assertStringIncludes(
      rowBlockRefusal([at(4, 1.2, 0.99), at(8, 2, 2)]) ?? "",
      "slower than the reference's row block in [2] at M=4 (×0.990)",
    );
  });

  it("refuses one without evidence from two records at a row count where it differs, or whose check failed there", () => {
    assertStringIncludes(
      rowBlockRefusal([at(4, 1.5, 1.5), { ...at(8, 1.5), missing: "unsteady in [2] at M=8" }]) ??
        "",
      "evidence at M=8 from fewer than 2 records (a parallel row serves M = 1 to 8: unsteady in [2] at M=8)",
    );
    assertEquals(
      rowBlockRefusal([{ ...at(8, 1.5, 1.5), ruledOut: "outside the band in [1] at M=8" }]),
      "outside the band in [1] at M=8",
    );
  });

  it("picks, among those that replace it, the one whose smallest gain over the row counts where it differs is largest, and none when none does", () => {
    const options = [
      { rowBlock: 2, at: [at(4, 1.3, 1.3), at(8, 1.6, 1.6)] },
      { rowBlock: 4, at: [at(4, 2, 2), at(8, 3, 1.9)] },
      { rowBlock: 8, at: [at(4, 2, 2), at(8, 1.0, 0.98)] },
    ];
    assertEquals(pickRowBlock(options)?.rowBlock, 4);
    // A tie goes to the first
    assertEquals(
      pickRowBlock([
        { rowBlock: 4, at: [at(8, 1.5, 1.5)] },
        { rowBlock: 8, at: [at(8, 1.5, 1.5)] },
      ])?.rowBlock,
      4,
    );
    assertEquals(pickRowBlock([{ rowBlock: 8, at: [at(8, 1.05, 1.05)] }]), undefined);
    assertEquals(pickRowBlock([]), undefined);
  });
});

describe("deriveLinearGemvChoices: the derivation", () => {
  it("keeps the thread target, and has no field when nothing is adopted", () => {
    const derivation = derive([{ caseId: PLE }]);
    assertEquals(derivation.linearGemv, undefined);
    assertEquals(derivation.defaults, DEFAULTS);
    const target = derivation.decisions.at(-1);
    assertEquals([target?.field, target?.outcome], ["rowsThreadTarget", "kept"]);
    assertStringIncludes(target?.reason ?? "", "only records of the full and row-blocks sets");
    const full = derive([{ caseId: PLE, rows: [1, 16] }]);
    assertStringIncludes(full.decisions.at(-1)?.reason ?? "", "no rule turns the heights into it");
  });

  it("names the records in its provenance and its lines, with the rule it used", () => {
    const derivation = derive([{ caseId: PLE, times: at1({ c64u8: faster(1.2) }) }]);
    assertEquals(
      derivation.provenance.records.map(({ path, sha256, candidateSet }) => [
        path,
        sha256,
        candidateSet,
      ]),
      [
        ["a.json", "sha-a.json", "quick"],
        ["bb.json", "sha-bb.json", "quick"],
      ],
    );
    assertEquals(derivation.provenance.adapter, GEMV_ADAPTER);
    assertEquals(derivation.rule, { minSpeedup: 1.1, steadySpread: 1.25 });
    assertEquals(derivation.verdicts[1]?.startsWith("  - [1] a.json ("), true);
    assert(
      derivation.verdicts.includes(
        `  - i4 n8960 k1536 (case ${PLE}): c32u4 -> c64u8 (×1.200 [1], ×1.200 [2])`,
      ),
      derivation.verdicts.join("\n"),
    );
  });

  it("gives a field the saved-table reader reads back", () => {
    const { linearGemv } = derive([
      { caseId: PLE, rows: PARALLEL_ROWS, times: at1({ "staged-l8": faster(1.3) }) },
    ]);
    assert(linearGemv !== undefined);
    const table = { ...TABLE, linearGemv };
    assertEquals(parseGeometryProfileJson(geometryProfileJson(table)).linearGemv, linearGemv);
  });
});

describe("deriveLinearGemvChoices: groups as the unit of evidence", () => {
  const twice = [{ caseId: PLE, times: at1({ c64u8: faster(1.2) }) }];

  it("reads a group with a stale row as no evidence for that candidate, while the record's other groups stay evidence (fault: c64u8's kernel fingerprint replaced in record 2)", () => {
    const specs: GemvCaseSpec[] = [
      { caseId: PLE, times: at1({ c64u8: faster(1.2) }) },
      { caseId: Q_G32, times: at1({ c64u8: faster(1.2) }) },
      { caseId: Q_G512, times: at1({ c64u8: faster(1.2) }) },
    ];
    const stale = specs.map((spec) =>
      spec.caseId === PLE
        ? { ...spec, fields: { 1: { c64u8: { kernel: "0000000000000000" } } } }
        : spec,
    );
    const derivation = derive(specs, stale);
    const ple = decisionOf(derivation, "sequential.shapes", PLE);
    assertEquals([ple.outcome, ple.candidate], ["kept", "c64u8"]);
    assertStringIncludes(ple.reason ?? "", "not current in [2] at M=1 (its kernel changed)");
    assertStringIncludes(ple.reason ?? "", "evidence at M=1 from fewer than 2 records");
    // The other cases of record 2 are evidence as before
    assertEquals(decisionOf(derivation, "sequential.shapes", Q_G32).outcome, "adopted");
    assertEquals(derivation.provenance.records[1]?.groups, {
      current: 3,
      unsteady: 0,
      stale: 0,
      staleRows: 1,
    });
  });

  it("derives from two records that each hold half of the cases what one complete pair derives", () => {
    const left: GemvCaseSpec[] = [
      { caseId: PLE, rows: PARALLEL_ROWS, times: at1({ c64u8: faster(1.2) }) },
      { caseId: HEAD_I8, times: at1({ c16u8: faster(1.3) }) },
    ];
    const right: GemvCaseSpec[] = [
      { caseId: Q_G32, times: at1({ c64u8: faster(1.15) }) },
      { caseId: Q_G512, times: at1({ c64u8: faster(1.25) }) },
      { caseId: PACKED, rows: PARALLEL_ROWS, times: at1({ "staged-l16": faster(1.5) }) },
    ];
    const whole = derive([...left, ...right]);
    const halves = deriveLinearGemvChoices(
      [
        input("left-1.json", left),
        input("right-1.json", right),
        input("left-2.json", left),
        input("right-2.json", right),
      ],
      { table: TABLE },
    );
    assertEquals(halves.linearGemv, whole.linearGemv);
    assert(whole.linearGemv !== undefined);
    assertEquals(
      halves.decisions.map(({ field, shape, outcome, derivedChoice }) => [
        field,
        shape,
        outcome,
        derivedChoice,
      ]),
      whole.decisions.map(({ field, shape, outcome, derivedChoice }) => [
        field,
        shape,
        outcome,
        derivedChoice,
      ]),
    );
  });

  it("never adopts a screened row and never counts it as evidence (fault: the winner screened out in record 2)", () => {
    const screened = derive(twice, [
      {
        caseId: PLE,
        times: at1({ c64u8: faster(1.2) }),
        fields: { 1: { c64u8: { screened: true } } },
      },
    ]);
    const decision = decisionOf(screened, "sequential.shapes", PLE);
    assertEquals([decision.outcome, decision.candidate], ["kept", "c64u8"]);
    assertStringIncludes(decision.reason ?? "", "screened out by the quick look in [2] at M=1");
    // Its one group of evidence is the record that timed it
    assertEquals(
      decision.gains.map(({ record }) => record),
      ["a.json"],
    );
    const both = derive([
      {
        caseId: PLE,
        times: at1({ c64u8: faster(1.2) }),
        fields: { 1: { c64u8: { screened: true } } },
      },
    ]);
    // Screened in both, c64u8 is nobody's evidence: the nearest is another candidate (at ×1.000)
    const kept = decisionOf(both, "sequential.shapes", PLE);
    assertEquals(kept.outcome, "kept");
    assert(kept.candidate !== "c64u8", kept.candidate);
    assertEquals(both.linearGemv, undefined);
  });

  it("never counts a skipped row as evidence and names why (fault: the winner of the parallel row skipped at M = 4 in record 2)", () => {
    const adopted: GemvCaseSpec = {
      caseId: PLE,
      rows: PARALLEL_ROWS,
      times: at1({ "staged-l8": faster(1.3) }),
    };
    // A run that skips the candidate at M = 4 skips the row blocks of its lanes with it
    const skipped = derive(
      [adopted],
      [
        {
          ...adopted,
          fields: {
            4: {
              "staged-l8": { skipped: true },
              "staged-l8r2": { skipped: true },
              "staged-l8r4": { skipped: true },
            },
          },
        },
      ],
    );
    const decision = decisionOf(skipped, "parallelShapes", PLE);
    assertEquals([decision.outcome, decision.candidate], ["kept", "staged-l8"]);
    assertStringIncludes(
      decision.reason ?? "",
      "skipped in [2] at M=4 (its M=1 group there left it unadoptable)",
    );
    assertStringIncludes(decision.reason ?? "", "evidence at M=4 from fewer than 2 records");
    // The control: measured at M = 4 in both records, it is adopted
    assertEquals(decisionOf(derive([adopted]), "parallelShapes", PLE).outcome, "adopted");
  });

  it("refuses a skipped row that is a baseline, has a time or is also screened", () => {
    const specs: GemvCaseSpec[] = [{ caseId: PLE, rows: PARALLEL_ROWS }];
    const mark =
      (pick: (row: GemvSweepReport["rows"][number]) => boolean, timed: boolean) =>
      (record: GemvSweepReport) => ({
        ...record,
        rows: record.rows.map((row) => {
          if (!pick(row)) return row;
          const { perDispatch, ...rest } = row;
          return { ...rest, ...(timed ? { perDispatch } : {}), skipped: true };
        }),
      });
    const message = "a skipped row is no baseline, has no time and is not screened";
    for (const change of [
      mark((row) => row.m === 4 && row.isBaseline, false),
      mark((row) => row.m === 4 && row.candidate === "staged-l8", true),
    ]) {
      assertThrows(
        () =>
          deriveLinearGemvChoices([input("a.json", specs), input("bb.json", specs, change)], {
            table: TABLE,
          }),
        Error,
        message,
      );
    }
    const both: GemvCaseSpec[] = [
      { ...specs[0], fields: { 4: { "staged-l8": { screened: true, skipped: true } } } },
    ];
    assertThrows(
      () =>
        deriveLinearGemvChoices([input("a.json", specs), input("bb.json", both)], { table: TABLE }),
      Error,
      message,
    );
  });

  it("adopts from a third record when one of three has the candidate's group unsteady", () => {
    const derivation = deriveLinearGemvChoices(
      [
        input("a.json", twice),
        input("bb.json", [{ ...twice[0], spread: 1.3 }]),
        input("ccc.json", twice),
      ],
      { table: TABLE },
    );
    const decision = decisionOf(derivation, "sequential.shapes", PLE);
    assertEquals(decision.outcome, "adopted");
    assertEquals(
      decision.gains.map(({ record }) => record),
      ["a.json", "ccc.json"],
    );
  });

  it("reports the groups that are not current, by cause, without refusing their records (faults: a case's fingerprint, a default's choice renamed, the baseline's kernel)", () => {
    const changedCase = input("bb.json", [{ caseId: PLE }, { caseId: Q_G32 }], (record) => ({
      ...record,
      cases: record.cases.map((entry) =>
        entry.id === PLE ? { ...entry, fingerprint: "0123456789abcdef" } : entry,
      ),
    }));
    const renamed = input("ccc.json", [{ caseId: PLE }], (record) => ({
      ...record,
      summaries: record.summaries.map((summary) => ({
        ...summary,
        profile: { ...summary.profile, reorder: "parallel-l8" },
      })),
    }));
    const baseline = input("dddd.json", [
      { caseId: HEAD_I8, fields: { 1: { c16u4: { kernel: "fedcba9876543210" } } } },
    ]);
    const derivation = deriveLinearGemvChoices(
      [input("a.json", [{ caseId: PLE }]), changedCase, renamed, baseline],
      { table: TABLE },
    );
    assertEquals(
      derivation.stale.map(({ record, caseId, m, cause }) => [record, caseId, m, cause]),
      [
        ["bb.json", PLE, 1, "case"],
        ["ccc.json", PLE, 1, "choices"],
        ["dddd.json", HEAD_I8, 1, "references"],
      ],
    );
    assertStringIncludes(derivation.stale[0]?.reason ?? "", "changed (its fingerprint differs)");
    assertStringIncludes(
      derivation.stale[1]?.reason ?? "",
      "is parallel-l8 in the record, parallel-l4 now",
    );
    assertStringIncludes(derivation.stale[2]?.reason ?? "", "the reference c16u4 is not current");
    assertEquals(
      derivation.provenance.records.map(({ groups }) => [groups.current, groups.stale]),
      [
        [1, 0],
        [1, 1],
        [0, 1],
        [0, 1],
      ],
    );
    assert(
      derivation.verdicts.some((line) =>
        line.includes("1 not current: the default profile's choices moved"),
      ),
      derivation.verdicts.join("\n"),
    );
  });
});

describe("deriveLinearGemvChoices: the refusals", () => {
  const specs = [{ caseId: PLE }];
  const refused = (inputs: readonly GemvSweepInput[], message: string, table = TABLE): void => {
    assertThrows(() => deriveLinearGemvChoices(inputs, { table }), Error, message);
  };

  it("derives nothing from one record (its groups are one record's evidence), and refuses the same record twice and none at all", () => {
    const one = deriveLinearGemvChoices(
      [input("a.json", [{ caseId: PLE, times: at1({ c64u8: faster(1.2) }) }])],
      { table: TABLE },
    );
    const decision = decisionOf(one, "sequential.shapes", PLE);
    assertEquals([decision.outcome, decision.candidate], ["kept", "c64u8"]);
    assertStringIncludes(decision.reason ?? "", "evidence at M=1 from fewer than 2 records");
    assertEquals(one.linearGemv, undefined);
    const record = input("a.json", specs);
    refused([record, { ...record, path: "copy.json" }], "passed twice (a.json and copy.json)");
    refused([], "No GEMV sweep record is given");
  });

  it("refuses the older formats, naming the format (faults: karume-gemv-sweep/3, /2 and /1)", () => {
    for (const format of ["karume-gemv-sweep/3", "karume-gemv-sweep/2", "karume-gemv-sweep/1"]) {
      refused(
        [input("a.json", specs), input("bb.json", specs, (record) => ({ ...record, format }))],
        `bb.json: format "${format}" is not karume-gemv-sweep/4 — a record of an older format is not read`,
      );
    }
  });

  it("refuses wall-clock or quantized timing", () => {
    for (const [change, message] of [
      [
        (record: GemvSweepReport) => ({
          ...record,
          gpuTiming: { ...record.gpuTiming, unit: "wall" },
        }),
        'gpuTiming.unit is "wall"',
      ],
      [
        (record: GemvSweepReport) => ({
          ...record,
          gpuTiming: { ...record.gpuTiming, quantized: true },
        }),
        "quantized timestamps are not read",
      ],
    ] as const) {
      refused([input("a.json", specs), input("bb.json", specs, change)], message);
    }
  });

  it("refuses records of two adapters, and records the table's match or provenance does not name", () => {
    refused(
      [
        input("a.json", specs),
        input("bb.json", specs, (record) => ({
          ...record,
          adapter: { ...record.adapter, device: "0x9999" },
        })),
      ],
      'bb.json: adapter device "0x9999" differs from a.json',
    );
    const inputs = [input("a.json", specs), input("bb.json", specs)];
    refused(inputs, "is not table 'other''s match", {
      ...TABLE,
      id: "other",
      match: { vendor: "nvidia" },
    });
    const withProvenance: GeometryProfile = {
      ...TABLE,
      provenance: { adapter: { ...GEMV_ADAPTER, description: "Another GPU" }, families: {} },
    };
    refused(inputs, "differs from table 'test-gpu''s provenance", withProvenance);
  });
});

describe("planGemvMeasurements: what the records lack", () => {
  const complete: GemvCaseSpec[] = [{ caseId: PLE, rows: PARALLEL_ROWS }];
  const plan = (inputs: readonly GemvSweepInput[]) =>
    planGemvMeasurements(inputs, { table: TABLE });
  const missingAt = (planning: ReturnType<typeof plan>, caseId: string, m: number) =>
    planning.missing.find((entry) => entry.caseId === caseId && entry.m === m)?.candidates;

  it("lists nothing for a case two current records hold, and every candidate the derivation reads for a case they do not", () => {
    const planning = plan([input("a.json", complete), input("bb.json", complete)]);
    for (const m of PARALLEL_ROWS) assertEquals(missingAt(planning, PLE, m), undefined);
    const other = missingAt(planning, Q_G32, 1);
    assert(other !== undefined);
    // At M = 1 every candidate but the references: the identical grid and the reorder forms
    assert(
      other.includes("c64u8") && other.includes("staged-l8") && other.includes("subgroup32-l8"),
    );
    assert(!other.includes("c32u4"), "the baseline is a reference");
    // At M = 4 the reorder forms only: the row-block heights feed no rule
    const four = missingAt(planning, Q_G32, 4) ?? [];
    assert(four.length > 0 && four.every((name) => !name.startsWith("c")), four.join(", "));
    assertEquals(planning.groups, GEMV_CASE_COUNT * PARALLEL_ROWS.length);
  });

  it("lists exactly the candidate whose row is stale in one record (fault: staged-l8's kernel fingerprint replaced at M = 4)", () => {
    const planning = plan([
      input("a.json", complete),
      input("bb.json", [
        { ...complete[0], fields: { 4: { "staged-l8": { kernel: "0000000000000000" } } } },
      ]),
    ]);
    assertEquals(missingAt(planning, PLE, 1), undefined);
    assertEquals(missingAt(planning, PLE, 4), ["staged-l8"]);
    assertEquals(missingAt(planning, PLE, 8), undefined);
  });

  it("counts a skipped row as looked at: the plan does not ask for it again (fault: staged-l8 skipped at M = 4 in record 2)", () => {
    const skipped = plan([
      input("a.json", complete),
      input("bb.json", [{ ...complete[0], fields: { 4: { "staged-l8": { skipped: true } } } }]),
    ]);
    assertEquals(missingAt(skipped, PLE, 4), undefined);
    // The control: without its row in record 2, it is missing there
    const absent = plan([
      input("a.json", complete),
      input("bb.json", [
        { ...complete[0], fields: { 4: { "staged-l8": { error: "Error: lost" } } } },
      ]),
    ]);
    assertEquals(missingAt(absent, PLE, 4), ["staged-l8"]);
  });

  it("counts a screened row as looked at, a failed row and an unsteady group as missing (faults: an error, a spread of ×1.3)", () => {
    const screened = plan([
      input("a.json", complete),
      input("bb.json", [{ ...complete[0], fields: { 1: { c64u8: { screened: true } } } }]),
    ]);
    assertEquals(missingAt(screened, PLE, 1), undefined);
    const failed = plan([
      input("a.json", complete),
      input("bb.json", [{ ...complete[0], fields: { 1: { c64u8: { error: "Error: lost" } } } }]),
    ]);
    assertEquals(missingAt(failed, PLE, 1), ["c64u8"]);
    const unsteady = plan([
      input("a.json", complete),
      input("bb.json", [{ ...complete[0], spread: { 8: 1.3 } }]),
    ]);
    assertEquals(missingAt(unsteady, PLE, 1), undefined);
    const eight = missingAt(unsteady, PLE, 8) ?? [];
    assert(eight.includes("parallel-l8") && eight.includes("staged-l32"), eight.join(", "));
  });
});

describe("a regeneration of the tile geometry (deriveGeometryProfile with a base)", () => {
  it("carries the base's linearGemv as it is", () => {
    const sweep = (path: string) => ({
      report: report(familyDefaultRows("gemm")),
      path,
      sha256: `sha-${path}`,
    });
    const target = { vendor: "apple", architecture: "metal-3" } as const;
    const base = deriveGeometryProfile([sweep("base.json")], { ...target, id: "test-gpu" }).profile;
    assertEquals(base.linearGemv, undefined);
    const { linearGemv } = derive([{ caseId: PLE, times: at1({ c64u8: faster(1.2) }) }]);
    assert(linearGemv !== undefined);
    const { profile } = deriveGeometryProfile([sweep("new.json")], {
      ...target,
      id: "test-gpu",
      base: { ...base, linearGemv },
    });
    assertEquals(profile.linearGemv, linearGemv);
    // Before the provenance, as the type orders the fields (the printer writes them in this order)
    const keys = Object.keys(profile);
    assertEquals(keys.indexOf("linearGemv"), keys.indexOf("provenance") - 1);
  });
});
