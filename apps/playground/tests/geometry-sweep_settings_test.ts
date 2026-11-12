import {
  type GeometrySweepCandidateSet,
  type GeometrySweepCase,
  geometrySweepCases,
} from "@karume/runtime/tune";
import { describe, expect, it } from "vite-plus/test";

import {
  CANDIDATE_SETS,
  changeCandidateSet,
  initialForm,
  matchingPreset,
  planSweep,
  presetForm,
  SWEEP_PRESETS,
  type SweepForm,
  sweepOps,
} from "../src/models/geometry-sweep/settings.ts";

const CASES: readonly GeometrySweepCase[] = [
  { id: "linear-a", op: "linear" },
  { id: "conv3d-a", op: "i8a8-conv3d" },
  { id: "linear-b", op: "linear" },
  { id: "conv3d-b", op: "i8a8-conv3d" },
  { id: "bmm-a", op: "bmm" },
];

const plannedCases = (form: SweepForm, cases: readonly GeometrySweepCase[] = CASES) => {
  const planned = planSweep(form, cases);
  if (!("plan" in planned)) throw new Error(planned.error);
  return planned.plan;
};

const V15 = SWEEP_PRESETS.find(({ id }) => id === "v15-conv3d");

describe("the initial form", () => {
  it("sweeps every case of the table in its order, with the runtime's candidate set", () => {
    const plan = plannedCases(initialForm(CASES));
    expect(plan.caseIds).toEqual(["linear-a", "conv3d-a", "linear-b", "conv3d-b", "bmm-a"]);
    expect(plan.ops).toEqual(["linear", "i8a8-conv3d", "bmm"]);
    expect(plan.candidateSet).toBe("quick+");
  });
});

describe("planSweep", () => {
  describe("when an op family is not chosen", () => {
    it("leaves its ticked cases out, and its op out of the call's ops", () => {
      const form = { ...initialForm(CASES), ops: new Set(["linear", "bmm"] as const) };
      const plan = plannedCases(form);
      expect(plan.caseIds).toEqual(["linear-a", "linear-b", "bmm-a"]);
      expect(plan.ops).toEqual(["linear", "bmm"]);
    });
  });

  describe("when an op is chosen but none of its cases is ticked", () => {
    it("does not name that op in the call's ops", () => {
      const form = { ...initialForm(CASES), cases: new Set(["conv3d-b"]) };
      expect(plannedCases(form)).toMatchObject({ caseIds: ["conv3d-b"], ops: ["i8a8-conv3d"] });
    });
  });

  describe("when the form cannot run", () => {
    it.each([
      [{ rounds: "0" }, "Rounds must be an integer of at least 1"],
      [{ rounds: "2.5" }, "Rounds must be an integer of at least 1"],
      [{ rounds: "" }, "Rounds must be an integer of at least 1"],
      [{ idleSeconds: "-1" }, "Idle seconds must be an integer of at least 0"],
      [{ idleSeconds: " " }, "Idle seconds must be an integer of at least 0"],
      [{ cases: new Set<string>() }, "Choose at least one case"],
    ])("refuses %o with the reason", (change, reason) => {
      expect(planSweep({ ...initialForm(CASES), ...change }, CASES)).toEqual({ error: reason });
    });
  });

  it("accepts no idle at all", () => {
    expect(plannedCases({ ...initialForm(CASES), idleSeconds: "0" }).idleSeconds).toBe(0);
  });
});

describe("the V-15 preset", () => {
  it("plans every conv3d case of the runtime's table, full set, 5 rounds, 90 s idle", () => {
    if (V15 === undefined) throw new Error("no V-15 preset");
    const runtimeCases = geometrySweepCases();
    const plan = plannedCases(presetForm(V15, runtimeCases), runtimeCases);
    const conv3d = runtimeCases.filter(({ op }) => op === "i8a8-conv3d").map(({ id }) => id);
    expect(conv3d.length).toBeGreaterThan(0);
    expect(plan).toEqual({
      caseIds: conv3d,
      ops: ["i8a8-conv3d"],
      candidateSet: "full",
      rounds: 5,
      idleSeconds: 90,
    });
  });
});

describe("matchingPreset", () => {
  it("names the preset while the form plans exactly what the preset plans", () => {
    if (V15 === undefined) throw new Error("no V-15 preset");
    expect(matchingPreset(presetForm(V15, CASES), CASES)?.id).toBe("v15-conv3d");
  });

  it("names it too when other ops stay ticked but none of their cases is (the same plan)", () => {
    if (V15 === undefined) throw new Error("no V-15 preset");
    const form = { ...presetForm(V15, CASES), ops: new Set(sweepOps(CASES)) };
    expect(matchingPreset(form, CASES)?.id).toBe("v15-conv3d");
  });

  it.each([
    ["a case unticked", { cases: new Set(["conv3d-a"]) }],
    ["another candidate set", { candidateSet: "quick" as const }],
    ["other rounds", { rounds: "3" }],
    ["another idle time", { idleSeconds: "60" }],
  ])("names none after %s", (_label, change) => {
    if (V15 === undefined) throw new Error("no V-15 preset");
    expect(matchingPreset({ ...presetForm(V15, CASES), ...change }, CASES)).toBeUndefined();
  });

  it("names none for a form that cannot run", () => {
    expect(matchingPreset({ ...initialForm(CASES), rounds: "x" }, CASES)).toBeUndefined();
  });
});

describe("a full run of the i8a8-linear family alone (ADR 0115 addendum decision 14)", () => {
  it("with only the i8a8-linear op family ticked under full, plans the family's 13 cases of the sweep's own table in its order (one call per case, merged into one record)", () => {
    const runtimeCases = geometrySweepCases();
    const form = {
      ...changeCandidateSet(initialForm(runtimeCases), "full", geometrySweepCases),
      ops: new Set(["i8a8-linear"] as const),
    };
    const plan = plannedCases(form, runtimeCases);
    expect(plan.ops).toEqual(["i8a8-linear"]);
    expect(plan.candidateSet).toBe("full");
    expect(plan.caseIds).toEqual(
      runtimeCases.filter(({ op }) => op === "i8a8-linear").map(({ id }) => id),
    );
    expect(plan.caseIds.length).toBe(13);
    // The page offers the sets the runtime has (the evaluation set k85 is retired)
    expect([...CANDIDATE_SETS]).toEqual(["quick", "quick+", "full", "k88"]);
  });
});

describe("changeCandidateSet", () => {
  const casesOf = (set: GeometrySweepCandidateSet): readonly GeometrySweepCase[] =>
    set === "k88" ? [{ id: "k88-a", op: "linear", censusCount: 1 }] : CASES;

  it("keeps the ticks when the new set sweeps the same table", () => {
    const form = { ...initialForm(CASES), cases: new Set(["linear-a"]) };
    const changed = changeCandidateSet(form, "full", casesOf);
    expect(changed.candidateSet).toBe("full");
    expect([...changed.cases]).toEqual(["linear-a"]);
    expect(changed.ops).toBe(form.ops);
  });

  it("ticks every case and op of the new table when the set sweeps another one, and back", () => {
    const toK88 = changeCandidateSet(initialForm(CASES), "k88", casesOf);
    expect(toK88.candidateSet).toBe("k88");
    expect([...toK88.cases]).toEqual(["k88-a"]);
    expect([...toK88.ops]).toEqual(["linear"]);
    const back = changeCandidateSet(toK88, "quick+", casesOf);
    expect([...back.cases]).toEqual(CASES.map(({ id }) => id));
    expect(plannedCases(back).caseIds.length).toBe(CASES.length);
  });
});

const K88 = SWEEP_PRESETS.find(({ id }) => id === "k88-forms");

describe("the K-88 preset", () => {
  it("plans every case of the k88 table (linear, matmul, conv2d) with the k88 set, 5 rounds, 90 s idle", () => {
    if (K88 === undefined) throw new Error("no K-88 preset");
    expect(K88.label).toBe("K-88 (GEMM forms / conv2d decode)");
    const k88Cases = geometrySweepCases("k88");
    const plan = plannedCases(presetForm(K88, k88Cases), k88Cases);
    expect(plan).toEqual({
      caseIds: k88Cases.map(({ id }) => id),
      ops: ["linear", "matmul", "conv2d"],
      candidateSet: "k88",
      rounds: 5,
      idleSeconds: 90,
    });
    expect(plan.caseIds.length).toBe(19);
    expect(matchingPreset(presetForm(K88, k88Cases), k88Cases)?.id).toBe("k88-forms");
  });

  it("is not named for the same ops and set over another table", () => {
    if (K88 === undefined) throw new Error("no K-88 preset");
    const runtimeCases = geometrySweepCases();
    const form = { ...presetForm(K88, runtimeCases), candidateSet: "full" as const };
    expect(matchingPreset(form, runtimeCases)?.id).toBeUndefined();
  });

  it("changing between k88 and quick+ switches the case table each way", () => {
    const toK88 = changeCandidateSet(initialForm(geometrySweepCases()), "k88", geometrySweepCases);
    expect([...toK88.cases]).toEqual(geometrySweepCases("k88").map(({ id }) => id));
    const back = changeCandidateSet(toK88, "quick+", geometrySweepCases);
    expect([...back.cases]).toEqual(geometrySweepCases().map(({ id }) => id));
  });
});
