import {
  geometryExploreOrder,
  type GeometrySweepCase,
  geometrySweepCases,
  QUICK_LOOK_MARGIN,
} from "@karume/runtime/tune";
import { describe, expect, it } from "vite-plus/test";
import {
  planSweep,
  SWEEP_CASE_IDS,
  SWEEP_OPS,
  type SweepSelection,
  sweepSelection,
} from "@karume/sweep-app-support/geometry-sweep/settings";
import { ParameterError, parseParameters } from "@karume/sweep-app-support/parameters";

import geometrySweep from "../src/entries/geometry-sweep.ts";

const CASES: readonly GeometrySweepCase[] = [
  { id: "linear-a", op: "linear" },
  { id: "conv3d-a", op: "i8a8-conv3d" },
  { id: "linear-b", op: "linear" },
  { id: "conv3d-b", op: "i8a8-conv3d" },
  { id: "bmm-a", op: "bmm" },
];

/** The entry's selection for the given parameter strings (the defaults for the rest). */
const selectionOf = (given: Readonly<Record<string, string>> = {}): SweepSelection =>
  sweepSelection(parseParameters(geometrySweep.parameters, given));

const DEFAULTS = selectionOf();

describe("the defaults", () => {
  it("sweep every case of the table in its order, with the runtime's candidate set", () => {
    const plan = planSweep(DEFAULTS, CASES);
    expect(plan.caseIds).toEqual(["linear-a", "conv3d-a", "linear-b", "conv3d-b", "bmm-a"]);
    expect(plan.ops).toEqual(["linear", "i8a8-conv3d", "bmm"]);
    expect(plan.candidateSet).toBe("quick+");
  });

  it("are quick+, every op family of the sweep's own table, no case or geometry chosen, the quick look at ×1.25, 5 rounds and 10 s idle", () => {
    expect(DEFAULTS).toEqual({
      candidateSet: "quick+",
      ops: [...new Set(geometrySweepCases().map(({ op }) => op))],
      cases: [],
      geometries: [],
      quickLookMargin: QUICK_LOOK_MARGIN,
      rounds: 5,
      idleSeconds: 10,
    });
  });
});

describe("the parameters", () => {
  it("offer every op family and every case id of the sweep's table and of the k88 table, in table order", () => {
    const tables = [...geometrySweepCases(), ...geometrySweepCases("k88")];
    expect(SWEEP_OPS).toEqual([...new Set(tables.map(({ op }) => op))]);
    expect(SWEEP_CASE_IDS).toEqual(tables.map(({ id }) => id));
  });

  it("take the geometries as comma-separated names, and refuse a quick-look margin between 0 and 1", () => {
    expect(
      selectionOf({ geometries: " tile64x64r8x4w16x8, tile32x64r4x4w16x8 ,," }).geometries,
    ).toEqual(["tile64x64r8x4w16x8", "tile32x64r4x4w16x8"]);
    expect(selectionOf({ quickLookMargin: "0" }).quickLookMargin).toBe(0);
    expect(() => selectionOf({ quickLookMargin: "0.9" })).toThrow(
      /^Parameter quickLookMargin: 0.9 is neither 0/,
    );
  });

  it("carry the geometries and the margin into the plan", () => {
    const plan = planSweep(
      { ...DEFAULTS, geometries: ["tile64x64r8x4w16x8"], quickLookMargin: 0 },
      CASES,
    );
    expect([plan.geometries, plan.quickLookMargin]).toEqual([["tile64x64r8x4w16x8"], 0]);
  });

  it("take no idle at all but no fewer seconds, and at least one round", () => {
    expect(selectionOf({ idleSeconds: "0" }).idleSeconds).toBe(0);
    for (const [name, value] of [
      ["idleSeconds", "-1"],
      ["rounds", "0"],
    ] as const) {
      expect(() => selectionOf({ [name]: value }), name).toThrow(new RegExp(`^Parameter ${name}:`));
    }
  });
});

describe("planSweep", () => {
  describe("when an op family is not chosen", () => {
    it("leaves its cases out, and its op out of the call's ops", () => {
      const plan = planSweep({ ...DEFAULTS, ops: ["linear", "bmm"] }, CASES);
      expect(plan.caseIds).toEqual(["linear-a", "linear-b", "bmm-a"]);
      expect(plan.ops).toEqual(["linear", "bmm"]);
    });

    it("does not sweep a chosen case of it", () => {
      const plan = planSweep(
        { ...DEFAULTS, ops: ["linear"], cases: ["linear-b", "conv3d-a"] },
        CASES,
      );
      expect(plan.caseIds).toEqual(["linear-b"]);
    });
  });

  describe("when an op is chosen but none of its cases is", () => {
    it("does not name that op in the call's ops", () => {
      expect(planSweep({ ...DEFAULTS, cases: ["conv3d-b"] }, CASES)).toMatchObject({
        caseIds: ["conv3d-b"],
        ops: ["i8a8-conv3d"],
      });
    });
  });

  describe("when a chosen case is not a case of the chosen set", () => {
    it("throws, naming it", () => {
      expect(() =>
        planSweep({ ...DEFAULTS, cases: ["linear-a", "k88-x", "k88-y"] }, CASES),
      ).toThrow(new ParameterError("cases", "k88-x, k88-y are not cases of the quick+ set"));
    });

    it("throws for a case of the sweep's table under k88, and for a k88 case under quick+", () => {
      const [k88Case] = geometrySweepCases("k88");
      const [sweepCase] = geometrySweepCases();
      if (k88Case === undefined || sweepCase === undefined) throw new Error("an empty case table");
      expect(() =>
        planSweep(
          selectionOf({ candidateSet: "k88", cases: sweepCase.id }),
          geometrySweepCases("k88"),
        ),
      ).toThrow(`${sweepCase.id} is not a case of the k88 set`);
      expect(() =>
        planSweep(selectionOf({ cases: k88Case.id }), geometrySweepCases("quick+")),
      ).toThrow(`${k88Case.id} is not a case of the quick+ set`);
    });
  });

  describe("when nothing is left to sweep", () => {
    it("refuses with the reason", () => {
      expect(() => planSweep({ ...DEFAULTS, ops: [] }, CASES)).toThrow("Choose at least one case");
    });
  });
});

describe("a full run of the i8a8-linear family alone (ADR 0115 addendum decision 14)", () => {
  it("with only the i8a8-linear op family under full, plans the family's cases of the sweep's own table in its order (one call per case, merged into one record)", () => {
    const runtimeCases = geometrySweepCases();
    const plan = planSweep(
      selectionOf({ candidateSet: "full", ops: "i8a8-linear" }),
      geometrySweepCases("full"),
    );
    expect(plan.ops).toEqual(["i8a8-linear"]);
    expect(plan.candidateSet).toBe("full");
    expect(plan.caseIds).toEqual(
      runtimeCases.filter(({ op }) => op === "i8a8-linear").map(({ id }) => id),
    );
  });
});

describe("the evaluation set k88", () => {
  it("with the default ops and no case chosen, plans every case of its own table", () => {
    const k88Cases = geometrySweepCases("k88");
    const plan = planSweep(selectionOf({ candidateSet: "k88" }), k88Cases);
    expect(plan.caseIds).toEqual(k88Cases.map(({ id }) => id));
    expect(plan.ops).toEqual(["linear", "matmul", "conv2d"]);
  });
});

describe("the explore set", () => {
  // In the table's order: linear M 1024 (gemmRows > 512) and 16 (≤ 16), then matmul M 4096 (> 512)
  // and 16 (≤ 16)
  const chosen = [
    "linear-m1024-n2048-k2048",
    "linear-m16-n3072-k1024",
    "matmul-m4096-n2048-k2048",
    "matmul-m16-n3072-k1024",
  ];

  it("plans the chosen cases of full's table class by class, as one runGeometrySweep call over them measures them (fault: the table's order)", () => {
    const plan = planSweep(
      selectionOf({ candidateSet: "explore", cases: chosen.join(",") }),
      geometrySweepCases("explore"),
    );
    expect(plan.candidateSet).toBe("explore");
    expect(plan.caseIds).toEqual([
      "linear-m1024-n2048-k2048",
      "matmul-m4096-n2048-k2048",
      "linear-m16-n3072-k1024",
      "matmul-m16-n3072-k1024",
    ]);
    expect(plan.caseIds).toEqual(
      geometryExploreOrder(chosen.map((id) => ({ id }))).map(({ id }) => id),
    );
  });

  it("refuses geometries, as the set chooses what it measures of full itself", () => {
    expect(() =>
      selectionOf({ candidateSet: "explore", geometries: "tile64x64r8x4w16x8k16" }),
    ).toThrow(
      new ParameterError(
        "geometries",
        "the explore set measures the candidates of full and drops them itself: leave geometries empty",
      ),
    );
  });
});
