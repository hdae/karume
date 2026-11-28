// The geometry sweep's `grid` set (ADR 0157 — the grid-fill rule of the GEMM): its case table,
// its candidates (the rule's ladder), its interleaved rounds and its record.
//
// 1. The case table spans the dispatch grid's workgroup count on every step of the ladder: under
//    the recon's fitted rule every step is some case's tile, 128 × 128's grid goes from 4 to 1,728
//    workgroups, and the storages are those of the presets (f32, f16, bf16, int4 at group 32) with
//    the integer cases past the GEMV family's rows.
// 2. Every case measures the ladder, its default geometry among it; the palindrome of a round times
//    every arm twice; the set takes no quick look; the field derivation refuses its records.
// 3. On a real GPU, a grid sweep's rows of a case share their reps and their rounds are aligned,
//    every tile writes the default's words, and the record carries the grid set's own keys.
//
// Fault injection (recorded 2026-11-28 on an RTX 3080 Ti, Dawn / Vulkan): with the dispatch of every
// candidate but the default given one column tile fewer than its geometry needs (`ceil(n / tileN) - 1`
// on the grid's x axis), part 3 turns red at the first candidate row, 128 × 128 on the bmm case: its
// words are not the default's.

import { assert, assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import {
  DEFAULT_GEOMETRY_PROFILE,
  type GemmGridRule,
  gemmGridStep,
  gemmGridWorkgroups,
} from "../src/kernels/geometry-profile.ts";
import { gemmTileM, gemmTileN } from "../src/kernels/gemm-geometry.ts";
import {
  type BmmCase,
  gemmCaseShape,
  GRID_CASES,
  GRID_CENSUS,
  K88_CASES,
  type LinearCase,
  SWEEP_CASES,
  sweepCaseEntries,
} from "../src/tune/cases.ts";
import { parseSweepReport } from "../src/tune/derive.ts";
import { candidateSetCaseSetIds, candidateSetDefaultKernelsIds } from "../src/tune/fingerprint.ts";
import { GEMM_GRID_LADDER, GEMM_GRID_MAX_PADDING, GEMM_GRID_MIN_K } from "../src/tune/gemm-grid.ts";
import { gemmCandidate, gridLadderCandidates } from "../src/tune/geometries.ts";
import { candidatesFor, casePlan, gridRoundOrder } from "../src/tune/harness.ts";
import { runGeometrySweep } from "../src/tune/sweep.ts";
import { planSweep } from "../src/tune/sweep.ts";

const adapter = navigator.gpu === undefined ? null : await navigator.gpu.requestAdapter();

/** The recon's fitted rule over the code's ladder (keep 128 × 128 at 256 workgroups, else 48). */
const FITTED: GemmGridRule = {
  ladder: GEMM_GRID_LADDER,
  keepLargest: { minWorkgroups: 256, minK: GEMM_GRID_MIN_K },
  minWorkgroups: 48,
  maxPadding: GEMM_GRID_MAX_PADDING,
};

const gemmCases = GRID_CASES.filter(
  (sweepCase): sweepCase is LinearCase | BmmCase =>
    sweepCase.op === "linear" || sweepCase.op === "bmm",
);

describe("the grid set's case table", () => {
  it("holds 33 linears and bmms of their own ids, apart from the sweep's and k88's tables", () => {
    assertEquals(GRID_CASES.length, 33);
    assertEquals(gemmCases.length, GRID_CASES.length);
    const ids = GRID_CASES.map(({ id }) => id);
    assertEquals(new Set(ids).size, ids.length);
    assert(ids.every((id) => id.startsWith("grid-")));
    const others = new Set([...SWEEP_CASES, ...K88_CASES].map(({ id }) => id));
    assert(ids.every((id) => !others.has(id)));
  });

  it("stores the weights as the presets do, the integer cases past the GEMV family's 64 rows", () => {
    const linears = GRID_CASES.filter(
      (sweepCase): sweepCase is LinearCase => sweepCase.op === "linear",
    );
    assertEquals([...new Set(linears.map(({ weight }) => weight))], ["f32", "f16", "bf16", "i4"]);
    for (const linear of linears) {
      if (linear.weight === "i4") {
        assertEquals(linear.groupSize, 32, linear.id);
        assert(linear.m > 64, `${linear.id} would run the GEMV family`);
      } else {
        assertEquals(linear.groupSize, undefined, linear.id);
      }
    }
    assertEquals(GRID_CASES.filter(({ op }) => op === "bmm").length, 7);
  });

  it("counts each case's nodes in a unit of the census, and lists them so", () => {
    const units: readonly string[] = Object.values(GRID_CENSUS);
    for (const sweepCase of gemmCases) {
      assert(sweepCase.censusCount > 0, sweepCase.id);
      assert(units.includes(sweepCase.censusGroup ?? ""), sweepCase.id);
    }
    assertEquals(
      sweepCaseEntries("grid").map(({ id, censusCount, censusGroup }) => [
        id,
        censusCount,
        censusGroup,
      ]),
      gemmCases.map(({ id, censusCount, censusGroup }) => [id, censusCount, censusGroup]),
    );
  });

  it("spans the grid fill: every step of the ladder is some case's tile under the fitted rule, and 128 × 128's grid runs from 4 to 1,728 workgroups", () => {
    const steps = gemmCases.map((sweepCase) => gemmGridStep(FITTED, gemmCaseShape(sweepCase)));
    assertEquals(
      [...new Set(steps)].toSorted((a, b) => a - b),
      [0, 1, 2, 3, 4],
    );
    for (const step of [0, 1, 2, 3, 4]) {
      assert(steps.filter((at) => at === step).length >= 1, `step ${step}`);
    }
    const largest = gemmCases.map((sweepCase) =>
      gemmGridWorkgroups(GEMM_GRID_LADDER[0], gemmCaseShape(sweepCase)),
    );
    assertEquals(Math.min(...largest), 4);
    assertEquals(Math.max(...largest), 1728);
    // Short-K bmm (K < 256) and padded shapes (N ≤ 192 at large M) are among them
    assert(gemmCases.some((sweepCase) => sweepCase.op === "bmm" && sweepCase.k < GEMM_GRID_MIN_K));
    assert(gemmCases.some((sweepCase) => sweepCase.n <= 192 && sweepCase.m >= 4096));
  });
});

describe("the grid set's candidates and rounds", () => {
  it("measures the ladder in every case, the case's default geometry among it", () => {
    const ladder = GEMM_GRID_LADDER.map((geometry) => gemmCandidate(geometry).name);
    assertEquals(
      gridLadderCandidates().map(({ name }) => name),
      ladder,
    );
    for (const sweepCase of GRID_CASES) {
      assertEquals(
        candidatesFor(sweepCase, "grid").map(({ name }) => name),
        ladder,
        sweepCase.id,
      );
      const defaultName = casePlan(sweepCase, Number.POSITIVE_INFINITY, false).defaultCandidate
        .name;
      assert(
        ladder.includes(defaultName),
        `${sweepCase.id}: default ${defaultName} is off the ladder`,
      );
    }
    // The default profile's three bucket tiles are on the ladder
    for (const { geometry } of DEFAULT_GEOMETRY_PROFILE.gemmRows) {
      assert(ladder.includes(gemmCandidate(geometry).name));
    }
  });

  it("times every arm twice a round, in a palindrome that turns round each round", () => {
    assertEquals(gridRoundOrder(3, 0), [0, 1, 2, 2, 1, 0]);
    assertEquals(gridRoundOrder(3, 1), [2, 1, 0, 0, 1, 2]);
    assertEquals(gridRoundOrder(3, 2), [0, 1, 2, 2, 1, 0]);
    for (const round of [0, 1]) {
      const order = gridRoundOrder(5, round);
      for (let arm = 0; arm < 5; arm += 1) {
        assertEquals(order.filter((at) => at === arm).length, 2);
      }
    }
  });

  it("takes no quick look: its plan's margin is 0, and another margin is refused", () => {
    assertEquals(planSweep({ candidateSet: "grid" }).quickLookMargin, 0);
    assertEquals(planSweep({ candidateSet: "grid", quickLookMargin: 0 }).quickLookMargin, 0);
    assertThrows(
      () => planSweep({ candidateSet: "grid", quickLookMargin: 1.25 }),
      Error,
      "takes no quick look",
    );
    assertEquals(planSweep({ candidateSet: "grid" }).cases.length, 33);
  });

  it("gives no field derivation its records", () => {
    assertThrows(
      () =>
        parseSweepReport(
          { format: "karume-geometry-sweep/4", settings: { candidateSet: "grid" } },
          { path: "grid.json", sha256: "0" },
        ),
      Error,
      "is the grid set",
    );
  });
});

describe("a grid sweep on a real GPU", () => {
  it.skipIf(adapter === null)(
    "aligns the rounds of a case's rows at one reps, finds every tile identical to the default, and records the grid set's own keys",
    async () => {
      const ids = ["grid-bmm-b16-m64-n64-k128", "grid-linear-m64-n512-k512-wbf16"];
      const swept = await runGeometrySweep({ candidateSet: "grid", cases: ids, rounds: 3 });
      assertEquals(swept.settings.candidateSet, "grid");
      assertEquals(swept.settings.quickLookMargin, 0);
      assertEquals(swept.caseSets, candidateSetCaseSetIds("grid", ["gemm"]));
      assertEquals(swept.defaultKernels, candidateSetDefaultKernelsIds("grid", ["gemm"]));
      for (const id of ids) {
        const rows = swept.rows.filter((row) => row.caseId === id);
        assertEquals(rows.length, GEMM_GRID_LADDER.length, id);
        assertEquals(rows[0].isDefault, true, id);
        assertEquals(rows.filter((row) => row.isDefault).length, 1, id);
        for (const row of rows) {
          assertEquals(row.error, undefined, `${id} ${row.geometry}`);
          assertEquals(row.reps, rows[0].reps, `${id} ${row.geometry}: one reps for every arm`);
          assertEquals(row.rounds?.length, 3, `${id} ${row.geometry}`);
          assert(row.perDispatch !== undefined && row.perDispatch > 0);
          if (!row.isDefault) {
            assertEquals(row.identicalToDefault, true, `${id} ${row.geometry}`);
            assert(row.speedupVsDefault !== undefined && row.speedupVsDefault > 0);
          }
        }
        const summary = swept.cases.find((entry) => entry.caseId === id);
        assert(summary !== undefined, id);
        assertEquals(summary.defaultRepeat, undefined, `${id}: no re-measurement`);
        assertEquals(summary.screened, 0);
      }
      // The tiles of a case cover the case's output however they cut it
      for (const row of swept.rows) {
        const [x, y] = row.workgroups ?? [0, 0];
        const sweepCase = gemmCases.find(({ id }) => id === row.caseId);
        assert(sweepCase !== undefined);
        const geometry = row.geometryParams;
        assertEquals(x, Math.ceil(sweepCase.n / gemmTileN(geometry)));
        assertEquals(y, Math.ceil(sweepCase.m / gemmTileM(geometry)));
      }
    },
  );
});
