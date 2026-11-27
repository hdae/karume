import { gemvSentinels, gemvSweepCases, geometrySweepCases } from "@karume/runtime/tune";
import {
  GEMV_SWEEP_PARAMETERS,
  gemvSweepSelection,
  planGemvSweep,
} from "@karume/sweep-app-support/gemv-sweep/settings";
import {
  GEOMETRY_SWEEP_PARAMETERS,
  sweepSelection,
} from "@karume/sweep-app-support/geometry-sweep/settings";
import { ParameterError, parseParameters } from "@karume/sweep-app-support/parameters";
import { describe, expect, it } from "vite-plus/test";

import { planBenchmark, trialOfQuery } from "../src/plan.ts";
import { CONV3D_A, CONV3D_B, GEMV_A } from "./helpers/fakes.ts";

const BOTH = { gemv: true, geometry: true };

/** What the workbench is given for the run of the page's defaults. */
const DEFAULT_RUN = { candidateSet: "explore", idleSeconds: "0" };

describe("planBenchmark", () => {
  it("runs both sweeps at the explore set with the entries' defaults and no idle time: the parameters the workbench parses for the same run", () => {
    const plan = planBenchmark(BOTH, {});
    expect(plan.sweeps.map(({ entryId }) => entryId)).toEqual(["gemv-sweep", "geometry-sweep"]);
    const [gemv, geometry] = plan.sweeps;
    expect(gemv?.parameters).toEqual(parseParameters(GEMV_SWEEP_PARAMETERS, DEFAULT_RUN));
    expect(geometry?.parameters).toEqual(parseParameters(GEOMETRY_SWEEP_PARAMETERS, DEFAULT_RUN));
    // Same key order as the workbench's record (the text is compared, not only the values)
    expect(JSON.stringify(gemv?.parameters)).toBe(
      JSON.stringify(parseParameters(GEMV_SWEEP_PARAMETERS, DEFAULT_RUN)),
    );
    expect(gemv?.sweep === "gemv" ? gemv.plan : undefined).toEqual(
      planGemvSweep(gemvSweepSelection(parseParameters(GEMV_SWEEP_PARAMETERS, DEFAULT_RUN))),
    );
    expect(plan.idleSeconds).toBe(0);
    expect(plan.cases).toHaveLength(gemvSweepCases().length + geometrySweepCases("explore").length);
  });

  it("idles before no case by default, the geometry sweep too, whose workbench default for the explore set is 10 s", () => {
    const plan = planBenchmark(BOTH, {});
    expect(plan.idleSeconds).toBe(0);
    expect(plan.sweeps.map((sweep) => sweep.plan.idleSeconds)).toEqual([0, 0]);
    expect(plan.sweeps.map((sweep) => sweep.parameters["idleSeconds"])).toEqual(["0", 0]);
    // The workbench's own default for the same set: the page does not take it
    expect(
      sweepSelection(parseParameters(GEOMETRY_SWEEP_PARAMETERS, { candidateSet: "explore" }))
        .idleSeconds,
    ).toBe(10);
  });

  it("takes the trial's idle time for both sweeps, above the default too", () => {
    const plan = planBenchmark(BOTH, { idleSeconds: 10 });
    expect(plan.idleSeconds).toBe(10);
    expect(plan.sweeps.map((sweep) => sweep.plan.idleSeconds)).toEqual([10, 10]);
    expect(plan.sweeps.map((sweep) => sweep.parameters["idleSeconds"])).toEqual(["10", 10]);
  });

  it("names the kinds the estimate reads: the GEMV sentinels first, and the first geometry case of each class", () => {
    const plan = planBenchmark(BOTH, {});
    const gemv = plan.cases.filter(({ sweep }) => sweep === "gemv");
    const sentinels = gemvSentinels(gemvSweepCases());
    expect(gemv.slice(0, sentinels.length).every(({ kind }) => kind === "sentinel")).toBe(true);
    expect(
      gemv
        .filter(({ kind }) => kind === "sentinel")
        .map(({ caseId }) => caseId)
        .sort(),
    ).toEqual([...sentinels].sort());
    const geometry = plan.cases.filter(({ sweep }) => sweep === "geometry");
    expect(geometry.filter(({ kind }) => kind === "first-of-class")).toHaveLength(17);
    const conv3d = geometry.filter(({ caseId }) => caseId.startsWith("i8a8-conv3d-"));
    expect(conv3d.map(({ kind }) => kind)).toEqual([
      "first-of-class",
      ...conv3d.slice(1).map(() => "later"),
    ]);
  });

  it("leaves out a sweep that is not chosen", () => {
    expect(
      planBenchmark({ gemv: false, geometry: true }, {}).sweeps.map(({ sweep }) => sweep),
    ).toEqual(["geometry"]);
    expect(planBenchmark({ gemv: false, geometry: false }, {}).cases).toEqual([]);
  });

  it("narrows each sweep to the trial's cases of its table, runs no sweep it names none of, and records the idle time", () => {
    const plan = planBenchmark(BOTH, { cases: [CONV3D_B, CONV3D_A], idleSeconds: 0 });
    expect(plan.sweeps.map(({ sweep }) => sweep)).toEqual(["geometry"]);
    expect(plan.cases.map(({ caseId }) => caseId)).toEqual([CONV3D_A, CONV3D_B]);
    expect(plan.sweeps[0]?.parameters["cases"]).toEqual([CONV3D_A, CONV3D_B]);
    expect(plan.sweeps[0]?.parameters["idleSeconds"]).toBe(0);
    expect(plan.idleSeconds).toBe(0);
  });
});

describe("trialOfQuery", () => {
  it("reads cases and idleSeconds, and nothing for an empty query", () => {
    expect(trialOfQuery("")).toEqual({});
    expect(trialOfQuery(`?cases=${GEMV_A},${CONV3D_A}&idleSeconds=2`)).toEqual({
      cases: [GEMV_A, CONV3D_A],
      idleSeconds: 2,
    });
  });

  it("refuses another parameter, an unknown case, an empty list and an idle time that does not fit", () => {
    expect(() => trialOfQuery("?rounds=1")).toThrow(ParameterError);
    expect(() => trialOfQuery("?cases=no-such-case")).toThrow(/no-such-case is no case/);
    expect(() => trialOfQuery("?cases=")).toThrow(ParameterError);
    expect(() => trialOfQuery("?idleSeconds=-1")).toThrow(/idleSeconds/);
    expect(() => trialOfQuery("?idleSeconds=1.5")).toThrow(/idleSeconds/);
  });
});
