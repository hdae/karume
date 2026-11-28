// The GEMV derivation without a GPU (src/tune/gemv-derive.ts — deriveLinearGemvChoices of
// `@karume/runtime/tune`): synthetic GEMV sweep records over the current runtime's cases and plans
// (tests/helpers/gemv-sweep-records.ts), so their fingerprints and default-profile choices are what a
// sweep of the current runtime writes.
//
// What this pins: ① a candidate is adopted only at ×1.10 or more in every group that is evidence,
// groups of two records or more, the one with the largest smallest gain among several; ② a failed
// check in one record, the packed rows, a shared sequential key whose cases disagree, a row count of
// the parallel range a record did not measure and a regression at one keep the default; ③ groups as
// the unit of evidence — a stale, screened or skipped row is no evidence while the record's other
// groups are, records that each hold half of the cases derive what a complete pair derives, and
// groups whose case is not current are reported instead of refused; ③′ the reference a class is read
// against — the current default's kernel wherever a record timed it, whatever default the record
// was made under, and no group for that class where its row is no reference; ③″ the reference's
// time — its fastest over the records, read only when a second record repeats it within ×1.10, the
// baseline's spread unread; ④ the refusals —
// none and the same record twice, an older format, another adapter, wall-clock or quantized timing
// (one record derives nothing); ⑤ what the field gets (the exception row replaced, added or removed;
// the parallel row replaced in place; nothing when nothing is adopted) and the provenance and lines
// that name the records; ⑥ the plan of what the records lack (planGemvMeasurements); ⑦ a regeneration
// of the tile geometry carries a base table's `linearGemv`; ⑧ the candidates a table can hold, as a
// reader of one record asks it (gemvTableCandidate: not the subgroup form, a row-blocked kernel, a
// row-block height or the GEMM).
//
// Faults injected: record 2 at ×1.09 (one record below the bar); differing words, and a reorder row
// outside the band, in record 2; records made under older default choices (read against the current
// ones), with the current baseline's words differing or unchecked; the reference's row dropped,
// untimed, failed or with another kernel in record 2; differing words where the baseline failed; a
// record whose baseline stayed slow (×1.3 against it, ×1.0 against the fastest); baselines spread ×2.5;
// the baseline's fastest time ×1.25 and ×1.11 ahead of the second record's; the two cases
// of one sequential key adopting different variants; a parallel candidate ×0.95 at M = 4; record 2
// without M = 8, and records of M = 1 alone, for a parallel row; a row's kernel fingerprint replaced
// (no evidence there, the record's other cases adopt; the plan lists that candidate alone); the
// winner screened out in one record; the parallel winner skipped at M = 4 in one record; a skipped
// row that is a baseline, has a time or is screened (refused); a case's fingerprint replaced (a stale
// group), the reorder choice of a summary renamed (no fault) and the baseline's kernel fingerprint
// replaced (a current group without a usable reference); the formats /3, /2 and /1, the unit, the
// quantized flag and the adapter changed; a failed row, a group without a usable reference and a
// reference whose fastest time one record holds (missing in the plan), a screened or skipped row and
// records made under older defaults (not missing); a record of the explore set read as one that holds
// part of the candidates, whose thread target names no row-block heights and whose skipped row the
// plan still lists (faults: the earlier readers, which took any set but quick for row blocks and
// every skipped row as looked at).

import { assert, assertEquals, assertStringIncludes, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import {
  deriveGeometryProfile,
  deriveLinearGemvChoices,
  type GemvSweepInput,
  gemvTableCandidate,
  type LinearGemvDecision,
  type LinearGemvDerivation,
  parseGeometryProfileJson,
  planGemvMeasurements,
  geometryProfileJson,
} from "../tune.ts";
import { DEFAULT_GEOMETRY_PROFILE, type GeometryProfile } from "../src/kernels/geometry-profile.ts";
import type { LinearGemvChoices } from "../src/kernels/linear-gemv.ts";
import { gemvCandidates } from "../src/tune/gemv-candidates.ts";
import { GEMV_CASES, gemvCaseOf } from "../src/tune/gemv-cases.ts";
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
/** E2B's i8 output head: no sequential exception, and a parallel row at 16 lanes. */
const HEAD_I8 = "i8-n262144-k1536";
/** Regular E2B's per_layer_model_projection in bf16: the default's first sequential exception (c16u8). */
const PLE_BF16 = "bf16-n8960-k1536";
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
    // The default's exceptions of the other keys, which these records hold no case of, stay in place
    const [head, ...others] = DEFAULTS.sequential.shapes;
    assertEquals(head, { storage: "bf16", n: 8960, k: 1536, variant: { cols: 16, unroll: 8 } });
    const replaced = derive([{ caseId: PLE_BF16, times: at1({ c16u16: faster(1.2) }) }]);
    assertEquals(replaced.linearGemv?.sequential.shapes, [
      { storage: "bf16", n: 8960, k: 1536, variant: { cols: 16, unroll: 16 } },
      ...others,
    ]);
    const removed = derive([{ caseId: PLE_BF16, times: at1({ c32u4: faster(1.2) }) }]);
    const decision = decisionOf(removed, "sequential.shapes", PLE_BF16);
    assertEquals([decision.defaultChoice, decision.derivedChoice], ["c16u8", "c32u4"]);
    assertEquals(removed.linearGemv?.sequential.shapes, others);
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

/**
 * A record made under older default choices at M = 1: its summary names `exact` and `reorder` as the
 * default's, its baseline row is `exact`'s, and the current baseline is a candidate row whose words
 * were found equal to that baseline's or not (`words`), or not checked.
 */
const underOlderDefaults =
  (exact: string, reorder: string, words: boolean | "unchecked" = true) =>
  (record: GemvSweepReport): GemvSweepReport => ({
    ...record,
    summaries: record.summaries.map((summary) =>
      summary.m === 1 ? { ...summary, profile: { exact, reorder } } : summary,
    ),
    rows: record.rows.map((row) => {
      if (row.m !== 1) return row;
      if (row.candidate === exact) {
        const {
          identicalToBaseline: _words,
          differingWords: _differing,
          speedupVsBaseline: _speedup,
          ...rest
        } = row;
        return { ...rest, isBaseline: true };
      }
      if (!row.isBaseline) return row;
      return {
        ...row,
        isBaseline: false,
        ...(words === "unchecked"
          ? {}
          : { identicalToBaseline: words, differingWords: words ? 0 : 3 }),
      };
    }),
  });

/** A record made when the default ran c64u4 at `exact` and parallel-l8 at `reorder` at M = 1. */
const OLDER_DEFAULTS = underOlderDefaults("c64u4", "parallel-l8");

/** A change that lays `fields` over one row of a record (absent: the row is dropped). */
const rowChange =
  (m: number, candidate: string, fields?: Partial<GemvSweepReport["rows"][number]>) =>
  (record: GemvSweepReport): GemvSweepReport => ({
    ...record,
    rows: record.rows.flatMap((row) =>
      row.m !== m || row.candidate !== candidate
        ? [row]
        : fields === undefined
          ? []
          : [{ ...row, ...fields }],
    ),
  });

describe("deriveLinearGemvChoices: the reference a class is read against", () => {
  /** A case both classes adopt from: c64u8 ×1.2 over c32u4, staged-l8 ×1.3 over parallel-l4. */
  const specs: GemvCaseSpec[] = [
    {
      caseId: PLE,
      rows: PARALLEL_ROWS,
      // The choices of the older defaults are slow: a gain read against them would be larger
      times: at1({
        c64u8: faster(1.2),
        "staged-l8": faster(1.3),
        c64u4: faster(0.5),
        "parallel-l8": faster(0.5),
      }),
    },
  ];
  const pair = (
    first: (record: GemvSweepReport) => unknown,
    second: (record: GemvSweepReport) => unknown,
  ): LinearGemvDerivation =>
    deriveLinearGemvChoices([input("a.json", specs, first), input("bb.json", specs, second)], {
      table: TABLE,
    });

  it("reads a record made under other default choices wherever it timed the current default's kernels, against those kernels (fault of the earlier rule: the group was not current)", () => {
    const derivation = pair(OLDER_DEFAULTS, OLDER_DEFAULTS);
    const sequential = decisionOf(derivation, "sequential.shapes", PLE);
    assertEquals([sequential.outcome, sequential.derivedChoice], ["adopted", "c64u8"]);
    // Against the current baseline (c32u4), not the record's baseline (c64u4: ×2.4)
    assertEquals(
      sequential.gains.map(({ gain }) => gain.toFixed(3)),
      ["1.200", "1.200"],
    );
    const parallel = decisionOf(derivation, "parallelShapes", PLE);
    assertEquals([parallel.outcome, parallel.derivedChoice], ["adopted", "staged-l8"]);
    assertEquals(
      parallel.gains.map(({ gain }) => gain.toFixed(3)),
      ["1.300", "1.300"],
    );
    assertEquals(derivation.stale, []);
    assertEquals(
      derivation.provenance.records.map(({ groups }) => groups.withoutReference),
      [
        { exact: 0, reorder: 0 },
        { exact: 0, reorder: 0 },
      ],
    );
  });

  it("takes a record's baseline as a candidate where the default moved off it: its own words are the baseline's (faults of a stricter reading: it ruled out for no word check, or never a candidate)", () => {
    const fasterOld: GemvCaseSpec[] = [{ caseId: PLE, times: at1({ c64u4: faster(1.2) }) }];
    const derivation = deriveLinearGemvChoices(
      [input("a.json", fasterOld, OLDER_DEFAULTS), input("bb.json", fasterOld, OLDER_DEFAULTS)],
      { table: TABLE },
    );
    const decision = decisionOf(derivation, "sequential.shapes", PLE);
    assertEquals([decision.outcome, decision.derivedChoice], ["adopted", "c64u4"]);
  });

  it("reads the identical class against the choice at exact alone: a moved choice at reorder whose row is missing leaves it standing (fault: parallel-l4's row dropped in record 2)", () => {
    const moved = (record: GemvSweepReport) =>
      rowChange(1, "parallel-l4")(underOlderDefaults("c32u4", "parallel-l8")(record));
    const derivation = pair((record) => record, moved);
    assertEquals(decisionOf(derivation, "sequential.shapes", PLE).outcome, "adopted");
    const parallel = decisionOf(derivation, "parallelShapes", PLE);
    assertEquals(parallel.outcome, "kept");
    assertStringIncludes(
      parallel.reason ?? "",
      "the reference parallel-l4 is not measured in [2] at M=1",
    );
  });

  it("reads no group for the identical class whose reference is not the record's baseline and was not shown to write its words (faults: words differing, and not checked)", () => {
    for (const words of [false, "unchecked"] as const) {
      const derivation = pair(
        (record) => record,
        underOlderDefaults("c64u4", "parallel-l4", words),
      );
      const decision = decisionOf(derivation, "sequential.shapes", PLE);
      assertEquals(decision.outcome, "kept", String(words));
      assertStringIncludes(
        decision.reason ?? "",
        "the reference c32u4 is not the baseline in [2] at M=1, and its words were not shown equal to the baseline's",
      );
      assertEquals(
        decision.gains.map(({ record }) => record),
        ["a.json"],
      );
      // The reorder class compares no words with the baseline: its reference stands
      assertEquals(decisionOf(derivation, "parallelShapes", PLE).outcome, "adopted");
      assertEquals(derivation.provenance.records[1]?.groups.withoutReference, {
        exact: 1,
        reorder: 0,
      });
    }
  });

  it("reads no group for a class whose reference's row is missing, untimed, failed or not current, and leaves the other class standing (faults in record 2)", () => {
    const reorderFaults = [
      [rowChange(1, "parallel-l4"), "the reference parallel-l4 is not measured in [2] at M=1"],
      [
        rowChange(1, "parallel-l4", { perDispatch: undefined }),
        "the reference parallel-l4 has no time in [2] at M=1",
      ],
      [
        rowChange(1, "parallel-l4", { error: "Error: lost" }),
        "the reference parallel-l4 failed in [2] at M=1 (Error: lost)",
      ],
      [
        rowChange(1, "parallel-l4", { kernel: "0000000000000000" }),
        "the reference parallel-l4 is not current in [2] at M=1 (its kernel changed)",
      ],
    ] as const;
    for (const [change, reason] of reorderFaults) {
      const derivation = pair((record) => record, change);
      const parallel = decisionOf(derivation, "parallelShapes", PLE);
      assertEquals(parallel.outcome, "kept", reason);
      assertStringIncludes(parallel.reason ?? "", reason);
      assertEquals(decisionOf(derivation, "sequential.shapes", PLE).outcome, "adopted", reason);
      assertEquals(derivation.provenance.records[1]?.groups.withoutReference, {
        exact: 0,
        reorder: 1,
      });
    }
    const baseline = pair(
      (record) => record,
      rowChange(1, "c32u4", { kernel: "0000000000000000" }),
    );
    const sequential = decisionOf(baseline, "sequential.shapes", PLE);
    assertEquals(sequential.outcome, "kept");
    assertStringIncludes(
      sequential.reason ?? "",
      "the reference c32u4 is not current in [2] at M=1 (its kernel changed)",
    );
    assertEquals(decisionOf(baseline, "parallelShapes", PLE).outcome, "adopted");
    // The control: the same records without a fault adopt both
    const control = pair(
      (record) => record,
      (record) => record,
    );
    assertEquals(
      [
        decisionOf(control, "sequential.shapes", PLE).outcome,
        decisionOf(control, "parallelShapes", PLE).outcome,
      ],
      ["adopted", "adopted"],
    );
  });

  it("rules a candidate out by a failed check in a group that has no usable reference (fault: differing words where the baseline's row failed, in one of three records)", () => {
    const derivation = deriveLinearGemvChoices(
      [
        input("a.json", specs),
        input("bb.json", specs, (record) =>
          rowChange(1, "c64u8", { identicalToBaseline: false, differingWords: 3 })(
            rowChange(1, "c32u4", { error: "Error: lost" })(record),
          ),
        ),
        input("ccc.json", specs),
      ],
      { table: TABLE },
    );
    const decision = decisionOf(derivation, "sequential.shapes", PLE);
    assertEquals(decision.outcome, "kept");
    assertStringIncludes(decision.reason ?? "", "its words differ from the baseline's in [2]");
  });
});

describe("deriveLinearGemvChoices: the reference's time", () => {
  const one = (times: Readonly<Record<string, number>>, spread?: number): GemvCaseSpec[] => [
    { caseId: PLE, times: at1(times), ...(spread === undefined ? {} : { spread }) },
  ];
  const three = (
    first: readonly GemvCaseSpec[],
    second: readonly GemvCaseSpec[],
    third: readonly GemvCaseSpec[],
  ): LinearGemvDerivation =>
    deriveLinearGemvChoices(
      [input("a.json", first), input("bb.json", second), input("ccc.json", third)],
      { table: TABLE },
    );

  it("reads every gain against the reference's fastest time over the records, so a record whose reference stayed slow cannot overstate a gain (fault of the earlier rule: ×1.37 against record 1's own baseline)", () => {
    // Record 1's baseline stayed slow (1300) and c64u8 took 950 there: ×1.37 against the record's
    // own baseline, ×1.05 against the fastest, which records 2 and 3 repeat
    const derivation = three(
      one({ c32u4: 1300, c64u8: 950 }),
      one({ c64u8: faster(1.2) }),
      one({ c64u8: faster(1.2) }),
    );
    const decision = decisionOf(derivation, "sequential.shapes", PLE);
    assertEquals([decision.outcome, decision.candidate], ["kept", "c64u8"]);
    assertStringIncludes(decision.reason ?? "", "×1.053 < ×1.100 in [1]");
    assertEquals(
      decision.gains.map(({ gain }) => gain.toFixed(3)),
      ["1.053", "1.200", "1.200"],
    );
  });

  it("reads records whose baseline's rounds spread widely: the spread is not read, the repeated fastest time is (fault of the earlier rule: a spread of ×2.5 discarded both)", () => {
    const derivation = derive(
      one({ c64u8: faster(1.2) }, 2.5),
      one({ c32u4: 1050, c64u8: faster(1.2) }, 2.5),
    );
    const decision = decisionOf(derivation, "sequential.shapes", PLE);
    assertEquals(decision.outcome, "adopted");
    // Both against the fastest (1000), record 2's own slower baseline unread
    assertEquals(
      decision.gains.map(({ gain }) => gain.toFixed(3)),
      ["1.200", "1.200"],
    );
  });

  it("reads no group where the reference's fastest time is held by one record alone, and takes a second within ×1.10 as a repeat (faults: ×1.25 and ×1.11 apart; ×1.10 is a repeat)", () => {
    /** The baseline at 800 in record 1, at `second` in record 2 and at 1000 in record 3; c64u8 at 600. */
    const withSecond = (second: number): LinearGemvDerivation =>
      three(
        one({ c32u4: 800, c64u8: 600 }),
        one({ c32u4: second, c64u8: 600 }),
        one({ c32u4: 1000, c64u8: 600 }),
      );
    for (const second of [1000, 889]) {
      const decision = decisionOf(withSecond(second), "sequential.shapes", PLE);
      assertEquals(decision.outcome, "kept", String(second));
      assertStringIncludes(
        decision.reason ?? "",
        "the reference c32u4's fastest time at M=1 ([1]) is repeated within ×1.100 by no second record",
      );
      assertEquals(decision.gains, []);
    }
    const repeated = decisionOf(withSecond(880), "sequential.shapes", PLE);
    assertEquals(repeated.outcome, "adopted");
    assertEquals(
      repeated.gains.map(({ gain }) => gain.toFixed(3)),
      ["1.333", "1.333", "1.333"],
    );
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
    // The numbers of both rules: one steady record's, then two records'
    assertEquals(derivation.rule, {
      margin: 1.065,
      steadyMinGroups: 20,
      steadyMaxShare: 0.2,
      steadyGroupSpread: 1.05,
      minPassMs: 20,
      rounds: 5,
      minSpeedup: 1.1,
      referenceRepeat: 1.1,
    });
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
      stale: 0,
      withoutReference: { exact: 0, reorder: 0 },
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

  it("adopts from a third record when one of three has no usable reference for the candidate's class (fault: its baseline's row failed)", () => {
    const derivation = deriveLinearGemvChoices(
      [
        input("a.json", twice),
        input("bb.json", [{ ...twice[0], fields: { 1: { c32u4: { error: "Error: lost" } } } }]),
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

  it("reports the groups whose case is not current and counts the current groups without a usable reference, without refusing their records (faults: a case's fingerprint, the baseline's kernel; a default's choice renamed is no fault)", () => {
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
      { caseId: HEAD_I8, fields: { 1: { c32u4: { kernel: "fedcba9876543210" } } } },
    ]);
    const derivation = deriveLinearGemvChoices(
      [input("a.json", [{ caseId: PLE }]), changedCase, renamed, baseline],
      { table: TABLE },
    );
    assertEquals(
      derivation.stale.map(({ record, caseId, m }) => [record, caseId, m]),
      [["bb.json", PLE, 1]],
    );
    assertStringIncludes(derivation.stale[0]?.reason ?? "", "changed (its fingerprint differs)");
    assertEquals(
      derivation.provenance.records.map(({ groups }) => [
        groups.current,
        groups.stale,
        groups.withoutReference.exact,
        groups.withoutReference.reorder,
      ]),
      [
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 1, 0],
      ],
    );
    assert(
      derivation.verdicts.some((line) =>
        line.includes(
          "dddd.json (2026-11-17T09:00:00.000Z, set quick): 1 groups — 1 current, 0 not current (their case is not the current one); of the current, 1 without a usable reference at exact and 0 at reorder; 1 rows of current groups not current",
        ),
      ),
      derivation.verdicts.join("\n"),
    );
    // The baseline's reason is the identical class's, read per group
    assertStringIncludes(
      decisionOf(derivation, "sequential.shapes", HEAD_I8).reason ?? "",
      "the reference c32u4 is not current in [4] at M=1 (its kernel changed)",
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
    assertEquals(decision.outcome, "kept");
    // One record's fastest time of the reference is not repeated, so its groups are no evidence
    assertStringIncludes(decision.reason ?? "", "repeated within ×1.100 by no second record");
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
      provenance: { adapter: { ...GEMV_ADAPTER, device: "0x9999" }, families: {} },
    };
    refused(inputs, "differs from table 'test-gpu''s provenance \"0x9999\"", withProvenance);
  });

  it("refuses a shared table (one derived from several adapters), even when the records are of one of its adapters: a shared table carries tile geometries only", () => {
    const shared: GeometryProfile = {
      ...TABLE,
      provenance: {
        adapters: [GEMV_ADAPTER, { ...GEMV_ADAPTER, description: "Other GPU" }],
        families: {},
      },
    };
    refused(
      [input("a.json", specs), input("bb.json", specs)],
      "Table 'test-gpu' is a shared table (derived from 2 adapters): it carries tile geometries only",
      shared,
    );
  });

  it("takes records whose description differs from the table's provenance (a driver update), while records that disagree on it are still refused", () => {
    const table: GeometryProfile = {
      ...TABLE,
      provenance: {
        adapter: { ...GEMV_ADAPTER, description: "NVIDIA: 615.78.08 615.78.8.0" },
        families: {},
      },
    };
    const updated = (record: GemvSweepReport): GemvSweepReport => ({
      ...record,
      adapter: { ...record.adapter, description: "NVIDIA: 620.70.06 620.70.6.0" },
    });
    const derivation = deriveLinearGemvChoices(
      [input("a.json", specs, updated), input("bb.json", specs, updated)],
      { table },
    );
    assertEquals(derivation.provenance.adapter.description, "NVIDIA: 620.70.06 620.70.6.0");
    refused(
      [input("a.json", specs), input("bb.json", specs, updated)],
      "bb.json: adapter description",
      table,
    );
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

  it("counts a screened row as looked at, a failed row and a group without a usable reference as missing (faults: an error, the baseline's kernel)", () => {
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
    // The identical class reads against the baseline: its candidates go missing, the reorder ones not
    const noReference = plan([
      input("a.json", complete),
      input("bb.json", [
        { ...complete[0], fields: { 1: { c32u4: { kernel: "0000000000000000" } } } },
      ]),
    ]);
    const one = missingAt(noReference, PLE, 1) ?? [];
    assert(one.includes("c64u8") && !one.includes("staged-l8"), one.join(", "));
    assertEquals(missingAt(noReference, PLE, 8), undefined);
    // The baseline's spread is not read (a spread of ×1.3, an unsteady group by the earlier rule)
    const spread = plan([
      input("a.json", complete),
      input("bb.json", [{ ...complete[0], spread: { 8: 1.3 } }]),
    ]);
    assertEquals(missingAt(spread, PLE, 8), undefined);
  });

  it("lists every candidate read against a reference whose fastest time one record alone holds, so the step measures the reference again (fault: the baseline ×1.25 faster in record 1)", () => {
    const planning = plan([
      input("a.json", [{ ...complete[0], times: { 1: { c32u4: 800 } } }]),
      input("bb.json", complete),
    ]);
    // Every identical candidate, the reorder ones (read against parallel-l4) not
    const gemvCase = gemvCaseOf(PLE);
    assert(gemvCase !== undefined);
    const identical = gemvCandidates(gemvCase, 1, "full", {
      subgroups: true,
      maxComputeWorkgroupsPerDimension: 65535,
    })
      .filter((candidate) => candidate.class === "identical")
      .map(({ name }) => name);
    assertEquals(missingAt(planning, PLE, 1), identical);
    assertEquals(missingAt(planning, PLE, 4), undefined);
  });

  it("counts a record made under other default choices where it timed the current reference, and not where that reference's words were not shown equal to its baseline's (fault: differing)", () => {
    const older = plan([
      input("a.json", complete, OLDER_DEFAULTS),
      input("bb.json", complete, OLDER_DEFAULTS),
    ]);
    for (const m of PARALLEL_ROWS) assertEquals(missingAt(older, PLE, m), undefined);
    const differing = plan([
      input("a.json", complete, OLDER_DEFAULTS),
      input("bb.json", complete, underOlderDefaults("c64u4", "parallel-l8", false)),
    ]);
    const one = missingAt(differing, PLE, 1) ?? [];
    assert(one.includes("c64u8") && !one.includes("staged-l8"), one.join(", "));
  });
});

describe("a record of the explore set", () => {
  /** A record of the explore set: its candidate set and sentinels, and only the rows `keep` keeps. */
  const explore =
    (keep: (row: GemvSweepReport["rows"][number]) => boolean = () => true) =>
    (record: GemvSweepReport): GemvSweepReport => ({
      ...record,
      settings: {
        ...record.settings,
        candidateSet: "explore",
        sentinels: ["i8-n2048-k256"],
      },
      rows: record.rows.filter(keep),
    });
  /** The same rows under the full set's name (a record that holds part of the candidates). */
  const partial =
    (keep: (row: GemvSweepReport["rows"][number]) => boolean = () => true) =>
    (record: GemvSweepReport): GemvSweepReport => ({ ...record, rows: record.rows.filter(keep) });
  const specs: GemvCaseSpec[] = [
    {
      caseId: PLE,
      rows: PARALLEL_ROWS,
      times: at1({ c64u8: faster(1.2), "staged-l8": faster(1.3) }),
    },
  ];
  /** The rows an explore group of PLE would hold at M = 1: the references and a few predicted ones. */
  const predicted = (row: GemvSweepReport["rows"][number]): boolean =>
    row.m !== 1 ||
    [
      "c32u4",
      "parallel-l4",
      "c64u8",
      "c32u8",
      "c64u4",
      "staged-l8",
      "staged-l4",
      "parallel-l8",
    ].includes(row.candidate);
  const withoutThreadTarget = (derivation: LinearGemvDerivation) =>
    derivation.decisions.filter((decision) => decision.field !== "rowsThreadTarget");

  it("is read exactly as a record that holds part of the candidates: the rows it has are evidence, a candidate without a row has none (control: the same rows under the full set's name)", () => {
    const asExplore = deriveLinearGemvChoices(
      [input("a.json", specs, explore(predicted)), input("bb.json", specs, explore(predicted))],
      { table: TABLE },
    );
    const asPartial = deriveLinearGemvChoices(
      [input("a.json", specs, partial(predicted)), input("bb.json", specs, partial(predicted))],
      { table: TABLE },
    );
    assertEquals(withoutThreadTarget(asExplore), withoutThreadTarget(asPartial));
    assertEquals(decisionOf(asExplore, "sequential.shapes", PLE).derivedChoice, "c64u8");
    assertEquals(
      asExplore.provenance.records.map(({ candidateSet }) => candidateSet),
      ["explore", "explore"],
    );
    // A winner one explore record did not measure is not adopted: one record is not evidence
    const once = deriveLinearGemvChoices(
      [
        input("a.json", specs, explore(predicted)),
        input(
          "bb.json",
          specs,
          explore((row) => predicted(row) && row.candidate !== "c64u8"),
        ),
      ],
      { table: TABLE },
    );
    assertEquals(decisionOf(once, "sequential.shapes", PLE).outcome, "kept");
  });

  it("measures no row-block height: the thread target's reason says no record of the full or row-blocks set is given (fault: read as one that measured them)", () => {
    const derivation = deriveLinearGemvChoices(
      [input("a.json", specs, explore()), input("bb.json", specs, explore())],
      { table: TABLE },
    );
    const target = derivation.decisions.find(({ field }) => field === "rowsThreadTarget");
    assertStringIncludes(target?.reason ?? "", "none is given");
  });

  it("does not complete the plan with its skipped or screened rows: the explore set skips by its own rule and screens at its own margin, so the plan still lists the candidate (fault: counted as looked at)", () => {
    const plan = (change: (record: GemvSweepReport) => GemvSweepReport) =>
      planGemvMeasurements([input("a.json", specs), input("bb.json", specs, change)], {
        table: TABLE,
      });
    const missingAt = (planning: ReturnType<typeof plan>, m: number) =>
      planning.missing.find((entry) => entry.caseId === PLE && entry.m === m)?.candidates;
    const skipped: GemvCaseSpec[] = [
      { ...specs[0], fields: { 4: { "staged-l8": { skipped: true } } } },
    ];
    const skippedExplore = planGemvMeasurements(
      [input("a.json", specs), input("bb.json", skipped, explore())],
      { table: TABLE },
    );
    assertEquals(missingAt(skippedExplore, 4), ["staged-l8"]);
    // The control: the same skipped row in a record of the full set counts as looked at
    const skippedFull = planGemvMeasurements([input("a.json", specs), input("bb.json", skipped)], {
      table: TABLE,
    });
    assertEquals(missingAt(skippedFull, 4), undefined);
    // Its screened rows do not count either (it screens at a narrower margin: the full set may
    // time the row), and the candidates it did not measure are missing
    const screened: GemvCaseSpec[] = [
      { ...specs[0], fields: { 1: { c64u8: { screened: true } } } },
    ];
    const screenedExplore = planGemvMeasurements(
      [input("a.json", specs), input("bb.json", screened, explore())],
      { table: TABLE },
    );
    assertEquals(missingAt(screenedExplore, 1), ["c64u8"]);
    // The control: the same screened row in a record of the full set counts as looked at
    const screenedFull = planGemvMeasurements(
      [input("a.json", specs), input("bb.json", screened)],
      { table: TABLE },
    );
    assertEquals(missingAt(screenedFull, 1), undefined);
    const narrowed = plan(explore(predicted));
    const one = missingAt(narrowed, 1) ?? [];
    assert(one.includes("c16u1") && !one.includes("c64u8"), one.join(", "));
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

describe("gemvTableCandidate: the candidates a table can hold", () => {
  const variant = { cols: 32, unroll: 4 };
  it("takes a sequential variant of the identical class and a plain or staged parallel row of the reorder class", () => {
    assert(
      gemvTableCandidate({
        class: "identical",
        choice: { id: "gemv-m1", storage: "i4", groupSize: 32, variant },
      }),
    );
    assert(
      gemvTableCandidate({
        class: "reorder",
        choice: { id: "gemv-parallel", storage: "i4", groupSize: 32, lanes: 4, variant },
      }),
    );
    assert(
      gemvTableCandidate({
        class: "reorder",
        choice: { id: "gemv-parallel-staged", storage: "i8", lanes: 16, variant },
      }),
    );
  });

  it("does not take the subgroup form (a Session option), a row-blocked kernel (a row's form past one row), a row-block height or the GEMM (faults: each)", () => {
    assert(
      !gemvTableCandidate({
        class: "reorder",
        choice: { id: "gemv-parallel-subgroup32", storage: "i4", groupSize: 32, lanes: 8, variant },
      }),
    );
    assert(
      !gemvTableCandidate({
        class: "reorder",
        choice: { id: "gemv-parallel-staged", storage: "i8", lanes: 16, variant, parallelRows: 4 },
      }),
    );
    assert(
      !gemvTableCandidate({
        class: "identical",
        choice: { id: "gemv-rows", storage: "i8", variant, rowsVariant: { ...variant, rows: 8 } },
      }),
    );
    assert(
      !gemvTableCandidate({
        class: "identical",
        choice: { id: "gemm-f32", storage: "bf16", v4: true },
      }),
    );
  });
});
