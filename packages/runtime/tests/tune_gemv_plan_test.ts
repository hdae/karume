// The GEMV sweep without a GPU (src/tune/gemv-*.ts): the case table and its pin to the default
// profile's GEMV rows, the baseline and the candidates of each set and class, the narrowing by row
// count and candidate, the roles the derivation reads a group for, the reorder candidates a run
// carries from its M = 1 group on to M = 4 and 8, the launch a program becomes, the host oracle and
// the band reading, the fingerprints and the record's shape.
//
// Faults injected: a profile row without a case (the pin reports it); an M = 1 row screened, failed,
// outside the band or ×1.09 the default's choice (each skips the candidate at M = 4), the same in an
// unsteady group or with the default untimed (only the check decides), a reference ruled out at
// M = 1 (still measured); a kernel whose WGSL gains a
// token, and a case whose n changes (each fingerprint moves); an output off the oracle by more than
// the band, a band of zero width and a NaN output (each reads outside the band).

import { assert, assertEquals, assertNotEquals, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { gemvSweepCandidateNames, gemvSweepCases } from "../tune.ts";
import { parseStorageRoles } from "../src/gpu/pipeline-cache.ts";
import { DEFAULT_GEOMETRY_PROFILE } from "../src/kernels/geometry-profile.ts";
import {
  isFloatStorage,
  type LinearGemvChoices,
  type LinearGemvParallelLanes,
  type LinearGemvParallelRows,
} from "../src/kernels/linear-gemv.ts";
import { opTolerance } from "../src/reference/op-tolerance.ts";
import {
  GEMV_MIN_SPEEDUP,
  GEMV_STEADY_SPREAD,
  type GemvCandidate,
  type GemvGates,
  type GemvM1Group,
  type GemvM1Row,
  gemvBaseline,
  gemvCandidates,
  gemvCarriedFromM1,
  gemvGroupRole,
  gemvKernelsAt,
  gemvMeasurementPlans,
  gemvProfileChoice,
  gemvProgram,
  programWorkgroups,
} from "../src/tune/gemv-candidates.ts";
import { GEMV_LOWEST_MARGIN } from "../src/tune/gemv-rule.ts";
import {
  GEMV_CASES,
  GEMV_SWEEP_FLOAT_ROW_COUNTS,
  GEMV_SWEEP_ROW_COUNTS,
  type GemvCase,
} from "../src/tune/gemv-cases.ts";
import {
  gemvCaseFingerprint,
  gemvCaseTableId,
  gemvKernelId,
} from "../src/tune/gemv-fingerprint.ts";
import { bandAgainstOracle, gemvOracle } from "../src/tune/gemv-oracle.ts";
import { choiceOfRecord, choiceRecord, type GemvSweepRow } from "../src/tune/gemv-report.ts";
import { gemvLaunch, gemvSweepGates, gemvSweepReport } from "../src/tune/gemv-sweep.ts";
import { assertBindingRoles, WARM_START_MAX_PASSES } from "../src/tune/harness.ts";
import {
  CALIBRATION_ATTEMPTS,
  RECALIBRATE_BELOW,
  WARM_START_LOADED_MIN_NS,
  WARM_START_MAX_NS,
  WARM_START_MIN_NS,
  WARM_START_PASSES,
  WARM_START_TOLERANCE,
} from "../src/tune/measurement.ts";

const GATES: GemvGates = { subgroups: true, maxComputeWorkgroupsPerDimension: 65535 };
const NO_SUBGROUPS: GemvGates = { ...GATES, subgroups: false };

const caseOf = (id: string): GemvCase => {
  const found = GEMV_CASES.find((gemvCase) => gemvCase.id === id);
  if (found === undefined) throw new Error(`no case ${id}`);
  return found;
};

/** E2B's per_layer_model_projection: i4 g32, a profile row at 4 lanes. */
const PLE = caseOf("i4g32-n8960-k1536");
/** E2B's i8 output head: the general sequential variant and a profile row at 16 lanes. */
const HEAD_I8 = caseOf("i8-n262144-k1536");
/** Regular E2B's per_layer_model_projection in bf16: the sequential variant's per-shape exception (c16u8). */
const PLE_BF16 = caseOf("bf16-n8960-k1536");
/** 12B's output head: 262144 columns, so 32 lanes need 65536 workgroups. */
const HEAD_12B = caseOf("i4g32-n262144-k3840");

const names = (candidates: readonly GemvCandidate[]): string[] =>
  candidates.map(({ name }) => name);

/**
 * The profile rows (parallel shapes and sequential exceptions) the case table has no case of. A
 * sequential exception has no group length, so a case of its storage and shape at any group is its
 * case (`group`: `"any"`).
 */
const uncovered = (choices: LinearGemvChoices, cases: readonly GemvCase[]): string[] =>
  [
    ...choices.parallelShapes.map(({ storage, group, n, k }) => ({ storage, group, n, k })),
    ...choices.sequential.shapes.map(({ storage, n, k }) => ({
      storage,
      group: "any" as const,
      n,
      k,
    })),
  ]
    .filter(
      (row) =>
        !cases.some(
          (gemvCase) =>
            gemvCase.storage === row.storage &&
            (row.group === "any" || gemvCase.group === row.group) &&
            gemvCase.n === row.n &&
            gemvCase.k === row.k,
        ),
    )
    .map(
      ({ storage, group, n, k }) =>
        `${storage} g${group === "any" ? "*" : (group ?? "-")} ${n}×${k}`,
    );

describe("the case table", () => {
  it("has a case for every GEMV row of the default profile, parallel and sequential", () => {
    assertEquals(uncovered(DEFAULT_GEOMETRY_PROFILE.linearGemv, GEMV_CASES), []);
  });

  it("reports a profile row that has no case (fault: one row added to the profile)", () => {
    const choices: LinearGemvChoices = {
      ...DEFAULT_GEOMETRY_PROFILE.linearGemv,
      parallelShapes: [
        ...DEFAULT_GEOMETRY_PROFILE.linearGemv.parallelShapes,
        { storage: "i4", n: 4096, k: 4096, group: 64, lanes: 4, packedActivations: false },
      ],
    };
    assertEquals(uncovered(choices, GEMV_CASES), ["i4 g64 4096×4096"]);
    // A sequential exception is covered by a case of its storage and shape at any group, and by no other
    const { sequential } = DEFAULT_GEOMETRY_PROFILE.linearGemv;
    const exceptions: LinearGemvChoices = {
      ...DEFAULT_GEOMETRY_PROFILE.linearGemv,
      sequential: {
        ...sequential,
        shapes: [
          ...sequential.shapes,
          { storage: "bf16", n: 8960, k: 2560, variant: { cols: 16, unroll: 8 } },
        ],
      },
    };
    assertEquals(uncovered(exceptions, GEMV_CASES), ["bf16 g* 8960×2560"]);
  });

  it("has a case for every linear of 26B A4B's QAT MTP drafter, the shapes 12B's drafter shares named for both", () => {
    // [n, k] of the drafter's linears at i4 g32, as its checkpoint holds them ([out, in]); the
    // embedding table is the output head (n 262144, k 1024).
    const drafter26b = [
      [1024, 5632], // pre_projection
      [4096, 1024], // q (sliding)
      [1024, 4096], // o (sliding)
      [8192, 1024], // gate / up, q (full)
      [1024, 8192], // down, o (full)
      [2816, 1024], // post_projection
      [262144, 1024], // lm_head
    ] as const;
    for (const [n, k] of drafter26b) {
      const { source } = caseOf(`i4g32-n${n}-k${k}`);
      assert(source.includes("26B A4B"), source);
      // 12B's drafter has the same shapes but at its projections (its hidden width is 3840)
      assertEquals(source.includes("12B"), n !== 2816 && k !== 5632, source);
    }
  });

  it("gives every case a unique id spelled from its shape, a source and the sweep's row counts", () => {
    assertEquals(new Set(GEMV_CASES.map(({ id }) => id)).size, GEMV_CASES.length);
    for (const gemvCase of GEMV_CASES) {
      const group = gemvCase.group === undefined ? "" : `g${gemvCase.group}`;
      assertEquals(gemvCase.id, `${gemvCase.storage}${group}-n${gemvCase.n}-k${gemvCase.k}`);
      assert(gemvCase.source.length > 0, gemvCase.id);
      // A float storage has no row block: its cases stop at the parallel GEMV's range
      assertEquals(
        gemvCase.rows,
        isFloatStorage(gemvCase.storage) ? GEMV_SWEEP_FLOAT_ROW_COUNTS : GEMV_SWEEP_ROW_COUNTS,
      );
    }
    assertEquals(GEMV_SWEEP_ROW_COUNTS, [1, 4, 8, 16, 64]);
    assertEquals(GEMV_SWEEP_FLOAT_ROW_COUNTS, [1, 4, 8]);
  });

  it("holds only shapes the GEMV family takes at every row count (the selection at exact), but for a float storage past one row, which the GEMM takes", () => {
    for (const gemvCase of GEMV_CASES) {
      for (const m of gemvCase.rows) {
        const { choice } = gemvBaseline(gemvCase, m);
        assertEquals(
          choice.id,
          m === 1 ? "gemv-m1" : isFloatStorage(gemvCase.storage) ? "gemm-f32" : "gemv-rows",
          `${gemvCase.id} M=${m}`,
        );
      }
    }
  });

  it("is public as gemvSweepCases, a copy each call", () => {
    const entries = gemvSweepCases();
    assertEquals(
      entries.map(({ id }) => id),
      GEMV_CASES.map(({ id }) => id),
    );
    assert(entries[0] !== GEMV_CASES[0]);
  });
});

describe("the baseline and the profile's choices", () => {
  it("is the selection at exact: the column form at M = 1, the profile's per-shape exception included", () => {
    assertEquals(gemvBaseline(PLE, 1).name, "c32u4");
    assertEquals(gemvBaseline(PLE_BF16, 1).name, "c16u8");
  });

  it("is the row block at M ≥ 2, its height from the profile's thread target", () => {
    // 8960 columns: 8960 · ceil(4 / rows) stays under 16384 at rows 4, so the height halves to 2.
    assertEquals(gemvBaseline(PLE, 4).name, "c32u4r2");
    assertEquals(gemvBaseline(HEAD_12B, 64).name, "c32u4r8");
  });

  it("reads the parallel row of the shape at reorder, and the baseline outside the parallel range", () => {
    assertEquals(gemvProfileChoice(PLE, 1, "reorder").name, "parallel-l4");
    // A row with a row block: its own kernel at M = 1, the row-blocked one at M = 4 and 8
    const down = caseOf("i4g32-n1536-k6144");
    assertEquals(gemvProfileChoice(down, 1, "reorder").name, "staged-l32");
    assertEquals(gemvProfileChoice(down, 4, "reorder").name, "staged-l32r2");
    assertEquals(gemvProfileChoice(PLE, 4, "reorder").name, "staged-l4r4");
    assertEquals(gemvProfileChoice(PLE, 8, "reorder").name, "staged-l4r4");
    // A row without one: the kernel of M = 1 at every row count
    assertEquals(gemvProfileChoice(caseOf("i4g32-n1536-k2048"), 4, "reorder").name, "staged-l32");
    assertEquals(gemvProfileChoice(PLE, 16, "reorder").name, gemvBaseline(PLE, 16).name);
  });
});

describe("the candidates", () => {
  describe("of the quick set", () => {
    it("measure M = 1 only", () => {
      assertEquals(
        gemvMeasurementPlans(PLE, "quick", GATES).map(({ m }) => m),
        [1],
      );
    });

    it("take the identical class on the 3 × 3 grid around the baseline, without it", () => {
      const identical = gemvCandidates(PLE, 1, "quick", GATES).filter(
        (candidate) => candidate.class === "identical",
      );
      assertEquals(names(identical), [
        "c16u2",
        "c16u4",
        "c16u8",
        "c32u2",
        "c32u8",
        "c64u2",
        "c64u4",
        "c64u8",
      ]);
    });

    it("stop the grid at its edge around a baseline at the edge (c16u8)", () => {
      const identical = gemvCandidates(PLE_BF16, 1, "quick", GATES).filter(
        (candidate) => candidate.class === "identical",
      );
      assertEquals(names(identical), ["c16u4", "c16u16", "c32u4", "c32u8", "c32u16"]);
    });

    it("take the reorder class in full: every lane count, plain, staged and subgroup", () => {
      const reorder = gemvCandidates(PLE, 1, "quick", GATES).filter(
        (candidate) => candidate.class === "reorder",
      );
      const lanes = [2, 4, 8, 16, 32];
      assertEquals(names(reorder), [
        ...lanes.map((count) => `parallel-l${count}`),
        ...lanes.map((count) => `staged-l${count}`),
        ...lanes.map((count) => `subgroup32-l${count}`),
      ]);
    });
  });

  describe("of the full set", () => {
    it("measure the row counts a derivation rule reads: the parallel range of the case", () => {
      assertEquals(
        gemvMeasurementPlans(PLE, "full", GATES).map(({ m }) => m),
        [1, 4, 8],
      );
    });

    it("take the whole column × read-ahead grid at M = 1, without the baseline", () => {
      const identical = gemvCandidates(PLE, 1, "full", GATES).filter(
        (candidate) => candidate.class === "identical",
      );
      assertEquals(identical.length, 24);
      assert(!names(identical).includes("c32u4"));
    });

    it("take every row-block height up to the tallest the shape allows at M ≥ 2", () => {
      const at = (m: number): string[] =>
        names(
          gemvCandidates(PLE, m, "full", GATES).filter(({ class: kind }) => kind === "identical"),
        );
      assertEquals(at(4), ["c32u4r1", "c32u4r4"]);
      assertEquals(names(gemvCandidates(PLE, 64, "row-blocks", GATES)), [
        "c32u4r1",
        "c32u4r2",
        "c32u4r4",
      ]);
      // i8 holds 16 rows a word, i2 4.
      assertEquals(gemvCandidates(HEAD_I8, 64, "row-blocks", GATES).length, 4);
      assertEquals(gemvCandidates(caseOf("i2-n12288-k1536"), 64, "row-blocks", GATES).length, 2);
    });

    it("take the reorder class at M = 1, 4 and 8 only, with the row-blocked form at M = 4 and 8", () => {
      // Plain, staged and subgroup at 5 lane counts; then the row blocks of at most M rows the
      // kernel exists at (i4: 2 / 4 / 8 rows up to 8 lanes, 2 / 4 at 16, 2 at 32)
      const expected: Readonly<Record<number, number>> = {
        1: 15,
        4: 15 + 9,
        8: 15 + 12,
        16: 0,
        64: 0,
      };
      for (const m of [1, 4, 8, 16, 64]) {
        const reorder = gemvCandidates(PLE, m, "full", GATES).filter(
          (candidate) => candidate.class === "reorder",
        );
        assertEquals(reorder.length, expected[m], `M=${m}`);
      }
    });

    it("name a row-blocked candidate by its lanes and rows, and build the staged form with that row block", () => {
      const blocked = (m: number): string[] =>
        names(
          gemvCandidates(PLE, m, "full", GATES).filter(
            ({ choice }) => "parallelRows" in choice && choice.parallelRows !== undefined,
          ),
        );
      assertEquals(blocked(1), []);
      assertEquals(blocked(4), [
        "staged-l2r2",
        "staged-l2r4",
        "staged-l4r2",
        "staged-l4r4",
        "staged-l8r2",
        "staged-l8r4",
        "staged-l16r2",
        "staged-l16r4",
        "staged-l32r2",
      ]);
      assertEquals(
        blocked(8).filter((name) => name.endsWith("r8")),
        ["staged-l2r8", "staged-l4r8", "staged-l8r8"],
      );
      // i2 stages 16 quads a word: no row block at 32 lanes
      assert(
        !names(gemvCandidates(caseOf("i2-n12288-k1536"), 8, "full", GATES)).some((name) =>
          name.startsWith("staged-l32r"),
        ),
      );
      const [candidate] = gemvCandidates(PLE, 8, "full", GATES).filter(
        ({ name }) => name === "staged-l4r4",
      );
      assertEquals(candidate.class, "reorder");
      assertEquals(candidate.choice.id, "gemv-parallel-staged");
      const [dispatch] = gemvProgram(PLE, 8, candidate.choice).dispatches;
      assertEquals(dispatch.key, "linear_gemv_parallel:wi4g32:l4:staged-x:r4");
      // Two y tiles of four rows
      assertEquals(programWorkgroups(gemvProgram(PLE, 8, candidate.choice)), [[280, 2, 1]]);
      // The record's choice keeps the rows, and stands for the same kernel
      const recorded = choiceRecord(candidate.choice, PLE.storage);
      assertEquals("parallelRows" in recorded ? recorded.parallelRows : undefined, 4);
      assertEquals(
        gemvKernelId(PLE, 8, choiceOfRecord(recorded)),
        gemvKernelId(PLE, 8, candidate.choice),
      );
    });
  });

  describe("of the row-blocks set", () => {
    it("measure the row counts beyond the parallel range: M = 16 and 64, the identical class alone", () => {
      const plans = gemvMeasurementPlans(PLE, "row-blocks", GATES);
      assertEquals(
        plans.map(({ m }) => m),
        [16, 64],
      );
      for (const plan of plans) {
        assert(plan.candidates.every((candidate) => candidate.class === "identical"));
      }
    });

    it("and the full set together measure every row count of the case", () => {
      assertEquals(
        [
          ...gemvMeasurementPlans(PLE, "full", GATES),
          ...gemvMeasurementPlans(PLE, "row-blocks", GATES),
        ].map(({ m }) => m),
        GEMV_SWEEP_ROW_COUNTS,
      );
    });
  });

  describe("narrowed by row count and candidate", () => {
    it("keep the row counts named, and the references with the candidates named", () => {
      const plans = gemvMeasurementPlans(PLE, "full", GATES, {
        rows: [4],
        candidates: ["staged-l8", "c32u4r1"],
      });
      assertEquals(
        plans.map(({ m }) => m),
        [4],
      );
      // The default's choice at reorder (the row's row-blocked kernel at M = 4) is measured with
      // them, the baseline is the plan's own
      assertEquals(names(plans[0]?.candidates ?? []), ["c32u4r1", "staged-l8", "staged-l4r4"]);
      assertEquals(plans[0]?.baseline.name, gemvBaseline(PLE, 4).name);
    });

    it("name every candidate of the table among the public names, each once", () => {
      const all = gemvSweepCandidateNames();
      assertEquals(new Set(all).size, all.length);
      for (const gemvCase of [PLE, HEAD_I8, HEAD_12B]) {
        for (const m of gemvCase.rows) {
          for (const { name } of gemvKernelsAt(gemvCase, m).slice(1)) {
            assert(all.includes(name), `${gemvCase.id} M=${m}: ${name}`);
          }
        }
      }
    });
  });

  it("are read by the derivation for a winner at M = 1, for a regression in the reorder class up to M = 8, and not at all otherwise", () => {
    assertEquals(
      [1, 4, 8, 16, 64].map((m) => [gemvGroupRole(m, "identical"), gemvGroupRole(m, "reorder")]),
      [
        ["winner", "winner"],
        ["unread", "no-regression"],
        ["unread", "no-regression"],
        ["unread", "unread"],
        ["unread", "unread"],
      ],
    );
  });

  it("leave the subgroup form out on a device without the features", () => {
    const candidates = gemvCandidates(PLE, 1, "quick", NO_SUBGROUPS);
    assert(!candidates.some(({ choice }) => choice.id === "gemv-parallel-subgroup32"));
    assertEquals(candidates.filter((candidate) => candidate.class === "reorder").length, 10);
  });

  it("leave out a candidate whose grid exceeds the device's limit, and only that one", () => {
    const at = (limit: number): string[] =>
      names(
        gemvCandidates(HEAD_12B, 1, "quick", { ...GATES, maxComputeWorkgroupsPerDimension: limit }),
      );
    assert(!at(65535).includes("parallel-l32"));
    assert(at(65536).includes("parallel-l32"));
    assertEquals(at(65536).length - at(65535).length, 3);
  });

  it("class every kernel by its summation order, and include the profile's choice at reorder", () => {
    for (const gemvCase of [PLE, HEAD_I8, HEAD_12B]) {
      const plans = [
        ...gemvMeasurementPlans(gemvCase, "full", GATES),
        ...gemvMeasurementPlans(gemvCase, "row-blocks", GATES),
      ];
      for (const plan of plans) {
        for (const candidate of plan.candidates) {
          const sequential =
            candidate.choice.id === "gemv-m1" || candidate.choice.id === "gemv-rows";
          assertEquals(candidate.class, sequential ? "identical" : "reorder", candidate.name);
        }
        const reorder = plan.profile.reorder;
        assert(
          reorder === plan.baseline.name || names(plan.candidates).includes(reorder),
          `${gemvCase.id} M=${plan.m}: ${reorder}`,
        );
      }
    }
  });

  it("have at M = 1 a candidate of the same name for every reorder candidate at M = 4 and 8, or for a row-blocked one the plain and the staged candidate of its lanes", () => {
    for (const gates of [GATES, NO_SUBGROUPS]) {
      for (const gemvCase of GEMV_CASES) {
        // The baseline is no candidate of the list, and the default's choice at reorder may be it
        const first = [
          gemvBaseline(gemvCase, 1).name,
          ...names(gemvCandidates(gemvCase, 1, "full", gates)),
        ];
        for (const m of [4, 8]) {
          for (const candidate of gemvCandidates(gemvCase, m, "full", gates)) {
            if (candidate.class !== "reorder") continue;
            const where = `${gemvCase.id} M=${m}: ${candidate.name}`;
            const { choice } = candidate;
            const lanes = "lanes" in choice ? choice.lanes : undefined;
            if (!("parallelRows" in choice) || choice.parallelRows === undefined) {
              assert(first.includes(candidate.name), where);
              continue;
            }
            assert(first.includes(`parallel-l${lanes}`), where);
            assert(first.includes(`staged-l${lanes}`), where);
          }
        }
      }
    }
  });
});

describe("the candidates carried from M = 1 on to M = 4 and 8 (gemvCarriedFromM1)", () => {
  /** The default's choice at reorder at M = 1 and at M = 4 (PLE's parallel row). */
  const DEFAULT_REORDER = "parallel-l4";
  /**
   * The references at M = 4: PLE's baseline there and the default's choice at reorder, which is the
   * row-blocked kernel of the row (its row block is 4).
   */
  const REFERENCES = { exact: "c32u4r2", reorder: "staged-l4r4" } as const;
  const REFERENCE_TIME = 1000;

  /** A timed M = 1 row that held its check. */
  const held = (candidate: string, perDispatch: number, fields: Partial<GemvM1Row> = {}) => ({
    candidate,
    perDispatch,
    withinBand: true,
    nonFiniteOutputs: 0,
    ...fields,
  });

  /**
   * An M = 1 group: the default's choice at reorder timed at {@link REFERENCE_TIME}, then `rows`; a
   * baseline spread of 1.01 unless `"none"` leaves it out.
   */
  const groupOf = (rows: readonly GemvM1Row[], spread: number | "none" = 1.01): GemvM1Group => ({
    ...(spread === "none" ? {} : { baselineSpread: spread }),
    profile: { reorder: DEFAULT_REORDER },
    rows: [held(DEFAULT_REORDER, REFERENCE_TIME), ...rows],
  });

  const carried = (
    group: GemvM1Group | undefined,
    name = "staged-l8",
    m = 4,
    candidateClass: GemvCandidate["class"] = "reorder",
  ): boolean => gemvCarriedFromM1(m, { name, class: candidateClass }, REFERENCES, group);

  /** The time at M = 1 that makes a candidate `gain` times faster than the default's choice. */
  const at = (gain: number): number => REFERENCE_TIME / gain;

  it("carries a candidate that held its check and is ×1.03 (the lowest margin) or more faster than the default's choice in a steady M = 1 group, at M = 4 and 8", () => {
    assertEquals(GEMV_LOWEST_MARGIN, 1.03);
    for (const m of [4, 8]) {
      assertEquals(carried(groupOf([held("staged-l8", at(1.3))]), "staged-l8", m), true);
    }
    // At the lowest margin itself it is adoptable
    assertEquals(carried(groupOf([held("staged-l8", REFERENCE_TIME / GEMV_LOWEST_MARGIN)])), true);
    // Below the rule of two records' ×1.10, which a steady record's rule does not ask (fault of the
    // earlier rule, which skipped it)
    assertEquals(GEMV_MIN_SPEEDUP, 1.1);
    assertEquals(carried(groupOf([held("staged-l8", at(1.07))])), true);
  });

  it("skips a candidate the quick look screened at M = 1 (condition 1)", () => {
    const screened: GemvM1Row = { candidate: "staged-l8", screened: true };
    assertEquals(carried(groupOf([screened])), false);
  });

  it("skips a candidate that failed at M = 1, or has no time there (condition 1)", () => {
    assertEquals(carried(groupOf([{ candidate: "staged-l8", error: "Error: lost" }])), false);
    // A timing whose every round was negative: a failure row without a time
    assertEquals(
      carried(groupOf([{ candidate: "staged-l8", withinBand: true, error: "negative" }])),
      false,
    );
  });

  it("skips a candidate outside the band, with a non-finite output, or without a check at M = 1 (condition 2)", () => {
    for (const row of [
      held("staged-l8", at(1.3), { withinBand: false }),
      held("staged-l8", at(1.3), { withinBand: false, nonFiniteOutputs: 3 }),
      held("staged-l8", at(1.3), { nonFiniteOutputs: 1 }),
      { candidate: "staged-l8", perDispatch: at(1.3) },
    ]) {
      assertEquals(carried(groupOf([row])), false, JSON.stringify(row));
    }
  });

  it("skips a candidate less than ×1.03 faster than the default's choice in a steady M = 1 group (condition 3; fault: ×1.02)", () => {
    assertEquals(carried(groupOf([held("staged-l8", at(1.02))])), false);
    assertEquals(carried(groupOf([held("staged-l8", at(0.5))])), false);
    // Steady up to the bound itself
    assertEquals(carried(groupOf([held("staged-l8", at(1.02))], GEMV_STEADY_SPREAD)), false);
  });

  it("lets an unsteady M = 1 group rule out nothing by its times: only conditions 1 and 2 decide (fault: a spread of ×1.26, and none)", () => {
    for (const spread of [GEMV_STEADY_SPREAD + 0.01, "none"] as const) {
      assertEquals(carried(groupOf([held("staged-l8", at(0.5))], spread)), true, `${spread}`);
      assertEquals(
        carried(groupOf([{ candidate: "staged-l8", screened: true }], spread)),
        false,
        `${spread}`,
      );
      assertEquals(
        carried(groupOf([held("staged-l8", at(1.3), { withinBand: false })], spread)),
        false,
        `${spread}`,
      );
    }
  });

  it("carries every candidate that held its check when the gain is unknown: the default's choice has no time at M = 1", () => {
    const failedReference: GemvM1Group = {
      baselineSpread: 1.01,
      profile: { reorder: DEFAULT_REORDER },
      rows: [{ candidate: DEFAULT_REORDER, error: "Error: lost" }, held("staged-l8", at(0.5))],
    };
    assertEquals(carried(failedReference), true);
    const noReferenceRow: GemvM1Group = { ...failedReference, rows: [held("staged-l8", at(0.5))] };
    assertEquals(carried(noReferenceRow), true);
    // Conditions 1 and 2 still hold sway
    assertEquals(
      carried({ ...failedReference, rows: [{ candidate: "staged-l8", screened: true }] }),
      false,
    );
  });

  it("always measures the references at the row count: the baseline and the default's choice at reorder (fault: their M = 1 rows rule them out)", () => {
    assertEquals(REFERENCES, {
      exact: gemvBaseline(PLE, 4).name,
      reorder: gemvProfileChoice(PLE, 4, "reorder").name,
    });
    assertEquals(gemvProfileChoice(PLE, 1, "reorder").name, DEFAULT_REORDER);
    // The default's choice is its own reference at M = 1: a gain of 1
    assertEquals(carried(groupOf([]), DEFAULT_REORDER), true);
    assertEquals(
      carried(groupOf([{ candidate: REFERENCES.exact, screened: true }]), REFERENCES.exact),
      true,
    );
    // Any other candidate at a gain of 1 is skipped
    assertEquals(carried(groupOf([held("staged-l8", REFERENCE_TIME)])), false);
  });

  describe("a row-blocked candidate, which has no row at M = 1 and is read by the rows of its lanes", () => {
    /** Whether `staged-l<lanes>r<rows>` is measured at M = 4, given the M = 1 group. */
    const blocked = (
      group: GemvM1Group | undefined,
      lanes: LinearGemvParallelLanes = 8,
      rows: LinearGemvParallelRows = 4,
    ): boolean =>
      gemvCarriedFromM1(
        4,
        {
          name: `staged-l${lanes}r${rows}`,
          class: "reorder",
          choice: { lanes, parallelRows: rows },
        },
        REFERENCES,
        group,
      );
    /** The plain and the staged candidate of 8 lanes at M = 1, each at a gain over the default. */
    const lanes8 = (plain: number, staged: number): GemvM1Group =>
      groupOf([held("parallel-l8", at(plain)), held("staged-l8", at(staged))]);

    it("is measured at the lanes of the default's choice at reorder, whatever the other rows say (the row the derivation keeps)", () => {
      // 4 lanes are the default's: its own gain is 1, and the staged candidate of 4 lanes is slower
      const group = groupOf([held("staged-l4", at(0.8))]);
      assertEquals(blocked(group, 4, 2), true);
      assertEquals(blocked(group, 4, 4), true);
    });

    it("is measured when the plain or the staged candidate of its lanes is carried from M = 1", () => {
      assertEquals(blocked(lanes8(1.3, 0.9)), true);
      assertEquals(blocked(lanes8(0.9, 1.3)), true);
      assertEquals(blocked(lanes8(1.3, 1.3)), true);
    });

    it("is skipped when M = 1 leaves both candidates of its lanes unadoptable (fault: ×1.02 and ×1.02)", () => {
      assertEquals(blocked(lanes8(1.02, 1.02)), false);
      assertEquals(
        blocked(
          groupOf([{ candidate: "parallel-l8", screened: true }, held("staged-l8", at(1.02))]),
        ),
        false,
      );
      assertEquals(
        blocked(
          groupOf([
            held("parallel-l8", at(1.3), { withinBand: false }),
            { candidate: "staged-l8", error: "Error: lost" },
          ]),
        ),
        false,
      );
    });

    it("is measured when the M = 1 group has no row of a candidate of its lanes (a run narrowed to other candidates)", () => {
      assertEquals(blocked(groupOf([])), true);
      assertEquals(blocked(groupOf([held("staged-l8", at(1.02))])), true);
      assertEquals(blocked(undefined), true);
    });

    it("leaves the default's choice at reorder at M = 1 measured where the default's row carries a row block: the reference at the row count is then the row-blocked kernel (fault: its own gain of 1)", () => {
      const references = { exact: "c32u4r2", reorder: "staged-l4r4" } as const;
      assertEquals(
        gemvCarriedFromM1(4, { name: DEFAULT_REORDER, class: "reorder" }, references, groupOf([])),
        true,
      );
      // The control: another candidate at a gain of 1 is skipped
      assertEquals(
        gemvCarriedFromM1(
          4,
          { name: "staged-l4", class: "reorder" },
          references,
          groupOf([held("staged-l4", REFERENCE_TIME)]),
        ),
        false,
      );
    });

    it("is measured as a reference when it is the default's choice at reorder at the row count", () => {
      assertEquals(
        gemvCarriedFromM1(
          4,
          { name: "staged-l8r4", class: "reorder", choice: { lanes: 8, parallelRows: 4 } },
          { exact: "c32u4r2", reorder: "staged-l8r4" },
          lanes8(1.02, 1.02),
        ),
        true,
      );
    });
  });

  it("skips nothing without the run's M = 1 group, for a candidate the M = 1 group has no row of, for the identical class, or at a row count not read for a regression", () => {
    const ruledOut = groupOf([{ candidate: "staged-l8", screened: true }]);
    assertEquals(carried(undefined), true);
    assertEquals(carried(ruledOut, "parallel-l16"), true);
    assertEquals(carried(ruledOut, "staged-l8", 4, "identical"), true);
    for (const m of [1, 16, 64]) assertEquals(carried(ruledOut, "staged-l8", m), true, `M=${m}`);
    // The control: the same group skips it at M = 4
    assertEquals(carried(ruledOut), false);
  });
});

describe("the launch of a candidate's program", () => {
  it("binds the case's resources at the program's positions, which the WGSL's roles accept", () => {
    const candidates = [
      ...gemvMeasurementPlans(PLE, "full", GATES),
      ...gemvMeasurementPlans(PLE, "row-blocks", GATES),
    ].flatMap((plan) =>
      [plan.baseline, ...plan.candidates].map((candidate) => ({ m: plan.m, candidate })),
    );
    for (const { m, candidate } of candidates) {
      const launch = gemvLaunch(gemvProgram(PLE, m, candidate.choice), "out", 65535);
      assertEquals(launch.bindings, ["x", "w", "bias", "out", "wscale"]);
      assertEquals(launch.writes, ["out"]);
      assertBindingRoles(launch, parseStorageRoles(launch.wgsl));
    }
  });

  it("writes the baseline's output to the reference buffer", () => {
    const launch = gemvLaunch(gemvProgram(PLE, 1, gemvBaseline(PLE, 1).choice), "reference", 65535);
    assertEquals(launch.bindings, ["x", "w", "bias", "reference", "wscale"]);
    assertEquals(launch.writes, ["reference"]);
  });

  it("fails loudly on a grid over the limit", () => {
    const choice = gemvBaseline(HEAD_12B, 1).choice;
    assertThrows(() => gemvLaunch(gemvProgram(HEAD_12B, 1, choice), "out", 1000));
  });
});

/** A stream over fixed words. */
const words = (values: readonly number[]): (() => number) => {
  let index = 0;
  return () => {
    const value = values[index];
    index += 1;
    if (value === undefined) throw new Error("stream exhausted");
    return value >>> 0;
  };
};

const f32Bits = (value: number): number => new Uint32Array(Float32Array.of(value).buffer)[0];

describe("the host oracle", () => {
  const immediately = async (): Promise<void> => undefined;

  it("restores i4 low nibble first as u − 8, per group scale, and adds the bias last", async () => {
    // One output, k = 32 (one group): nibbles 0..7 → q −8..−1, 8..15 → 0..7, then 16 × q = 0.
    const x = Array.from({ length: 64 }, (_, index) => f32Bits(index < 32 ? 1 : (index - 32) % 4));
    const out = await gemvOracle(
      { storage: "i4", group: 32, n: 1, k: 32 },
      2,
      {
        x: words(x),
        weight: words([0x76543210, 0xfedcba98, 0x88888888, 0x88888888]),
        bias: words([f32Bits(0.25)]),
        scale: words([f32Bits(0.5)]),
      },
      immediately,
    );
    const q = [
      ...Array.from({ length: 16 }, (_, index) => index - 8),
      ...new Array<number>(16).fill(0),
    ];
    const row1 = q.reduce((sum, value, index) => sum + ((index % 4) * value) / 2, 0);
    assertEquals([...out], [-3.75, row1 + 0.25]);
  });

  it("restores i8 as the signed byte, byte 0 first", async () => {
    const out = await gemvOracle(
      { storage: "i8", n: 1, k: 16 },
      1,
      {
        x: words(Array.from({ length: 16 }, (_, index) => f32Bits(index + 1))),
        weight: words([0x80ff017f, 0, 0, 0]),
        bias: words([0]),
        scale: words([f32Bits(1)]),
      },
      immediately,
    );
    assertEquals([...out], [1 * 127 + 2 * 1 + 3 * -1 + 4 * -128]);
  });

  it("restores i2 as code − 2, the low bits first", async () => {
    const out = await gemvOracle(
      { storage: "i2", n: 1, k: 64 },
      1,
      {
        x: words(Array.from({ length: 64 }, (_, index) => f32Bits(index < 4 ? 10 ** index : 0))),
        // Codes 0, 1, 2, 3 → −2, −1, 0, 1; the rest are code 2 (0).
        weight: words([0xaaaaaae4, 0xaaaaaaaa, 0xaaaaaaaa, 0xaaaaaaaa]),
        bias: words([0]),
        scale: words([f32Bits(2)]),
      },
      immediately,
    );
    // x · q = 1 · −2 + 10 · −1 + 100 · 0 + 1000 · 1; read high bits first, the same words would give
    // 2 · (1 − 100 − 2000).
    assertEquals([...out], [2 * (-2 - 10 + 1000)]);
  });
});

describe("the band reading", () => {
  const tolerance = opTolerance("linear");
  const oracle = Float64Array.of(1, -2, 1000);

  it("is inside the band within atol + rtol · |oracle|, with the largest error and its share", () => {
    const reading = bandAgainstOracle(Float32Array.of(1 + 1e-4, -2, 1000), oracle, tolerance);
    assert(reading.withinBand);
    assert(Math.abs((reading.maxAbsError ?? 0) - 1e-4) < 1e-7);
    assert(reading.bandUse !== undefined && reading.bandUse > 0.49 && reading.bandUse < 0.51);
  });

  it("is outside the band past it (fault: an output off by 1e-3)", () => {
    const reading = bandAgainstOracle(Float32Array.of(1 + 1e-3, -2, 1000), oracle, tolerance);
    assert(!reading.withinBand);
    assert(reading.bandUse !== undefined && reading.bandUse > 1);
  });

  it("is outside a band of zero width for any error (fault: atol = rtol = 0)", () => {
    const reading = bandAgainstOracle(Float32Array.of(1 + 1e-6, -2, 1000), oracle, {
      atol: 0,
      rtol: 0,
    });
    assertEquals(reading.withinBand, false);
    assertEquals(reading.bandUse, undefined);
  });

  it("counts a NaN output and is outside the band (fault: one NaN)", () => {
    const reading = bandAgainstOracle(Float32Array.of(Number.NaN, -2, 1000), oracle, tolerance);
    assertEquals(reading.nonFiniteOutputs, 1);
    assertEquals(reading.withinBand, false);
  });
});

describe("the fingerprints", () => {
  const kernels = gemvMeasurementPlans(PLE, "quick", GATES).flatMap((plan) =>
    [plan.baseline, ...plan.candidates].map((candidate) => ({ m: plan.m, candidate })),
  );

  it("of the case table moves with a dispatch-deciding field, not with a source (fault: n changed)", () => {
    const changed = GEMV_CASES.map((gemvCase, index) =>
      index === 0 ? { ...gemvCase, n: gemvCase.n + 32 } : gemvCase,
    );
    const described = GEMV_CASES.map((gemvCase, index) =>
      index === 0 ? { ...gemvCase, source: "another description" } : gemvCase,
    );
    assertNotEquals(gemvCaseTableId(changed), gemvCaseTableId());
    assertEquals(gemvCaseTableId(described), gemvCaseTableId());
  });

  it("of a case moves with a dispatch-deciding field, not with a source or the row counts, and not with another case (fault: n changed)", () => {
    const [first, second] = GEMV_CASES;
    assertNotEquals(gemvCaseFingerprint({ ...first, n: first.n + 32 }), gemvCaseFingerprint(first));
    assertEquals(
      gemvCaseFingerprint({ ...first, source: "another description" }),
      gemvCaseFingerprint(first),
    );
    assertEquals(gemvCaseFingerprint({ ...first, rows: [1] }), gemvCaseFingerprint(first));
    assertNotEquals(gemvCaseFingerprint(second), gemvCaseFingerprint(first));
  });

  it("of a kernel moves when its text changes, not with a comment, and not with another kernel's (fault: one token added)", () => {
    const edited =
      (edit: (wgsl: string) => string) =>
      (gemvCase: GemvCase, m: number, choice: GemvCandidate["choice"]) => {
        const program = gemvProgram(gemvCase, m, choice);
        return {
          ...program,
          dispatches: program.dispatches.map((dispatch) => ({
            ...dispatch,
            wgsl: () => edit(dispatch.wgsl()),
          })),
        };
      };
    for (const { m, candidate } of kernels) {
      const base = gemvKernelId(PLE, m, candidate.choice);
      assertNotEquals(
        gemvKernelId(
          PLE,
          m,
          candidate.choice,
          edited((wgsl) => `${wgsl}\nconst extra = 1u;`),
        ),
        base,
        candidate.name,
      );
      assertEquals(
        gemvKernelId(
          PLE,
          m,
          candidate.choice,
          edited((wgsl) => `// note\n${wgsl}`),
        ),
        base,
        candidate.name,
      );
    }
    const ids = kernels.map(({ m, candidate }) => gemvKernelId(PLE, m, candidate.choice));
    assertEquals(new Set(ids).size, ids.length);
  });
});

describe("the record", () => {
  const plan = gemvMeasurementPlans(PLE, "quick", GATES)[0];
  const timed = (candidate: GemvCandidate, isBaseline: boolean): GemvSweepRow => ({
    caseId: PLE.id,
    m: 1,
    candidate: candidate.name,
    class: candidate.class,
    isBaseline,
    choice: choiceRecord(candidate.choice, PLE.storage),
    kernel: gemvKernelId(PLE, 1, candidate.choice),
    keys: [gemvProgram(PLE, 1, candidate.choice).dispatches[0]?.key ?? ""],
    rounds: [150_000, 200_000],
    perDispatch: 1000,
  });
  const rows = [timed(plan.baseline, true), ...plan.candidates.map((one) => timed(one, false))];
  const report = gemvSweepReport({
    startedAt: "2026-11-17T10:00:00.000Z",
    date: "2026-11-17T10:30:00.000Z",
    aborted: false,
    userAgent: "Node.js/24",
    adapter: { vendor: "nvidia", architecture: "ampere", device: "", description: "" },
    geometryProfile: "nvidia-ampere",
    gates: gemvSweepGates({
      features: new Set(["subgroups"]),
      wgslLanguageFeatures: new Set(["subgroup_id"]),
      limits: { maxComputeWorkgroupsPerDimension: 65535 },
    }),
    timing: { feature: true, unit: "ns" },
    candidateSet: "quick",
    caseIds: [PLE.id],
    selection: { candidates: ["c64u8"] },
    rounds: 5,
    quickLookMargin: 1.25,
    tolerance: opTolerance("linear"),
    deviceLost: null,
    cases: [PLE],
    summaries: [
      {
        caseId: PLE.id,
        m: 1,
        elapsedMs: 1,
        profile: plan.profile,
        baselineSpread: 1.004,
        screened: 0,
        skipped: 0,
      },
    ],
    rows,
  });

  it("is a karume-gemv-sweep/4 record whose cases and rows carry the fingerprints of this runtime", () => {
    assertEquals(report.format, "karume-gemv-sweep/4");
    assertEquals(
      report.cases.map(({ fingerprint }) => fingerprint),
      [gemvCaseFingerprint(PLE)],
    );
    for (const row of report.rows) {
      assertEquals(row.kernel, gemvKernelId(PLE, 1, choiceOfRecord(row.choice)), row.candidate);
    }
  });

  it("names the features that gated candidates (the subgroup form needs all three)", () => {
    assertEquals(report.gates, {
      subgroups: false,
      features: { subgroups: true, subgroupSizeControl: false, subgroupId: true },
      maxComputeWorkgroupsPerDimension: 65535,
    });
  });

  it("states its settings: the set, the narrowed cases and candidates, the quick look's margin, the band and what it does not measure", () => {
    assertEquals(report.settings.candidateSet, "quick");
    assertEquals(report.settings.cases, [PLE.id]);
    assertEquals(report.settings.candidates, ["c64u8"]);
    assertEquals(report.settings.rows, undefined);
    assertEquals(report.settings.quickLookMargin, 1.25);
    assertEquals(report.settings.tolerance, opTolerance("linear"));
    assert(report.settings.notMeasured.includes("packed-activation"));
  });

  it("states the warm start and the calibration check it timed with, and carries each summary's baseline spread", () => {
    assertEquals(report.settings.warmStart, {
      minNs: WARM_START_MIN_NS,
      loadedMinNs: WARM_START_LOADED_MIN_NS,
      maxNs: WARM_START_MAX_NS,
      maxPasses: WARM_START_MAX_PASSES,
      stablePasses: WARM_START_PASSES,
      tolerance: WARM_START_TOLERANCE,
    });
    assertEquals(report.settings.recalibrateBelow, RECALIBRATE_BELOW);
    assertEquals(report.settings.calibrationAttempts, CALIBRATION_ATTEMPTS);
    assertEquals(
      report.summaries.map(({ baselineSpread }) => baselineSpread),
      [1.004],
    );
  });

  it("lists the planned cases with their source, fingerprint and the row counts the set measured", () => {
    assertEquals(report.cases, [{ ...PLE, rows: [1], fingerprint: gemvCaseFingerprint(PLE) }]);
  });

  it("is not quantized when the rounds are not all multiples of 100 µs", () => {
    assertEquals(report.gpuTiming, { feature: true, unit: "ns", quantized: false });
  });
});

describe("the record's session and class", () => {
  const parts = (summaries: number, session?: { id: string; elapsedMs: number }) =>
    gemvSweepReport({
      startedAt: "2026-11-28T10:00:00.000Z",
      date: "2026-11-28T10:30:00.000Z",
      aborted: false,
      userAgent: "Node.js/24",
      ...(session === undefined ? {} : { session }),
      adapter: { vendor: "nvidia", architecture: "ampere", device: "", description: "" },
      geometryProfile: "default",
      gates: gemvSweepGates({
        features: new Set(),
        wgslLanguageFeatures: new Set(),
        limits: { maxComputeWorkgroupsPerDimension: 65535 },
      }),
      timing: { feature: true, unit: "ns" },
      candidateSet: "full",
      caseIds: [],
      selection: {},
      rounds: 5,
      quickLookMargin: 1.25,
      tolerance: opTolerance("linear"),
      deviceLost: null,
      cases: [],
      summaries: Array.from({ length: summaries }, (_, index) => ({
        caseId: `case-${index}`,
        m: 1,
        elapsedMs: 1,
        profile: { exact: "c32u4", reorder: "c32u4" },
        // One group in five beyond ×1.05: 20 %, the most a steady record may have
        baselineSpread: index % 5 === 0 ? 1.2 : 1.01,
        screened: 0,
        skipped: 0,
      })),
      rows: [],
    });

  it("states its class from its own summaries (fault: 19 groups, no class)", () => {
    assertEquals(parts(20).steadiness, {
      class: "steady",
      groups: 20,
      unsteadyGroups: 4,
      share: 0.2,
    });
    assertEquals(parts(19).steadiness?.class, "unclassified");
  });

  it("names its session when the sweep was given one, and has none otherwise", () => {
    assertEquals(parts(1, { id: "session-a", elapsedMs: 1234 }).session, {
      id: "session-a",
      elapsedMs: 1234,
    });
    assertEquals("session" in parts(1), false);
  });
});

describe("the device table's choices in a plan (gemvMeasurementPlans with a table)", () => {
  /** A table whose sequential variant for PLE's shape is c128u16 and whose PLE row is staged at 16 lanes. */
  const table: LinearGemvChoices = {
    ...DEFAULT_GEOMETRY_PROFILE.linearGemv,
    sequential: {
      ...DEFAULT_GEOMETRY_PROFILE.linearGemv.sequential,
      shapes: [
        ...DEFAULT_GEOMETRY_PROFILE.linearGemv.sequential.shapes,
        { storage: "i4", n: PLE.n, k: PLE.k, variant: { cols: 128, unroll: 16 } },
      ],
    },
    parallelShapes: DEFAULT_GEOMETRY_PROFILE.linearGemv.parallelShapes.map((row) =>
      row.storage === "i4" && row.group === PLE.group && row.n === PLE.n && row.k === PLE.k
        ? { ...row, lanes: 16, stagedActivations: true, rowBlock: undefined }
        : row,
    ),
  };

  it("names the table's choices at both levels, and the default's where there is no table", () => {
    const [one, four] = gemvMeasurementPlans(PLE, "full", GATES, {}, table);
    assertEquals(one.table, { exact: "c128u16", reorder: "staged-l16" });
    assertEquals(four.table.reorder, "staged-l16");
    const [plain] = gemvMeasurementPlans(PLE, "full", GATES);
    assertEquals(plain.table, plain.profile);
  });

  it("measures them in the quick set, whose small grid does not hold c128u16, and when a narrowing names other candidates (faults: both)", () => {
    const quick = gemvMeasurementPlans(PLE, "quick", GATES, {}, table)[0];
    assert(quick.candidates.some(({ name }) => name === "c128u16"));
    assert(!gemvCandidates(PLE, 1, "quick", GATES).some(({ name }) => name === "c128u16"));
    const narrowed = gemvMeasurementPlans(PLE, "full", GATES, { candidates: ["c64u8"] }, table)[0];
    assertEquals(
      narrowed.candidates.map(({ name }) => name).toSorted(),
      ["c128u16", "c64u8", gemvProfileChoice(PLE, 1, "reorder").name, "staged-l16"].toSorted(),
    );
  });
});
