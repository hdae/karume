// The GEMV sweep and its derivation over float-stored cases (bf16 / f16 / f32), without a GPU: the
// cases' row counts, the baseline — the sequential GEMV at one row, the GEMM past it, where such a
// weight has no sequential GEMV —, the candidates, the launch without a scale buffer, the host
// oracle's restoration of each storage, the record's form of the GEMM and the derivation of a
// parallel row read against the GEMM at M = 4 and 8.
//
// Faults injected: a bf16 word read high half first, an f16 half read as bf16 (the oracle's value
// moves); a launch that binds a scale buffer (the WGSL's roles refuse it); a parallel candidate
// slower than the GEMM at M = 8 (the row is not adopted); a record whose GEMM row has no `v4` (refused).

import { assert, assertEquals, assertStringIncludes, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { deriveLinearGemvChoices, type GemvSweepInput, planGemvMeasurements } from "../tune.ts";
import { parseStorageRoles } from "../src/gpu/pipeline-cache.ts";
import { DEFAULT_GEOMETRY_PROFILE, type GeometryProfile } from "../src/kernels/geometry-profile.ts";
import {
  type GemvCandidate,
  type GemvGates,
  gemvBaseline,
  gemvCandidates,
  gemvMeasurementPlans,
  gemvProfileChoice,
  gemvProgram,
  gemvRowCountsIn,
  isGemmChoice,
} from "../src/tune/gemv-candidates.ts";
import { GEMV_CASES, type GemvCase, isFloatGemvStorage } from "../src/tune/gemv-cases.ts";
import { gemvOracle } from "../src/tune/gemv-oracle.ts";
import { choiceOfRecord, choiceRecord, type GemvSweepReport } from "../src/tune/gemv-report.ts";
import { gemvLaunch } from "../src/tune/gemv-sweep.ts";
import { assertBindingRoles, resourceWordStream } from "../src/tune/harness.ts";
import {
  BASE_TIME,
  GEMV_ADAPTER,
  type GemvCaseSpec,
  gemvRecord,
} from "./helpers/gemv-sweep-records.ts";

const GATES: GemvGates = { subgroups: true, maxComputeWorkgroupsPerDimension: 65535 };

const caseOf = (id: string): GemvCase => {
  const found = GEMV_CASES.find((gemvCase) => gemvCase.id === id);
  if (found === undefined) throw new Error(`no case ${id}`);
  return found;
};

/** Regular E2B's down of the 12288-wide MLP layers, in bf16. */
const DOWN = caseOf("bf16-n1536-k12288");
/** Regular E2B's output head in bf16: 262144 columns. */
const HEAD = caseOf("bf16-n262144-k1536");

const names = (candidates: readonly GemvCandidate[]): string[] =>
  candidates.map(({ name }) => name);

const PARALLEL_NAMES = [2, 4, 8, 16, 32].flatMap((lanes) => [
  `parallel-l${lanes}`,
  `staged-l${lanes}`,
]);

describe("the float-stored cases of the case table", () => {
  it("are the linears of regular Gemma 4 E2B and E4B in bf16, at the row counts of the parallel range", () => {
    const float = GEMV_CASES.filter(({ storage }) => isFloatGemvStorage(storage));
    assertEquals(float.length, 25);
    for (const gemvCase of float) {
      assertEquals(gemvCase.storage, "bf16");
      assertEquals(gemvCase.group, undefined);
      assertEquals(gemvCase.rows, [1, 4, 8]);
      assertEquals(gemvCase.id, `bf16-n${gemvCase.n}-k${gemvCase.k}`);
      assert(gemvCase.source.startsWith("Gemma 4 E"), gemvCase.source);
      assertEquals(gemvRowCountsIn(gemvCase, "row-blocks"), []);
    }
  });

  it("add to the table without moving a case that was there (the i2 / i4 / i8 cases keep their order and row counts)", () => {
    const integer = GEMV_CASES.filter(({ storage }) => !isFloatGemvStorage(storage));
    assertEquals(GEMV_CASES.slice(0, integer.length), integer);
    assertEquals(integer[0].id, "i4g32-n8960-k1536");
    for (const gemvCase of integer) assertEquals(gemvCase.rows, [1, 4, 8, 16, 64]);
  });
});

describe("the baseline of a float-stored case", () => {
  it("is the sequential GEMV at one row and the GEMM past it, both of the reference order", () => {
    const one = gemvBaseline(DOWN, 1);
    assertEquals([one.name, one.class, one.choice.id], ["c32u4", "identical", "gemv-m1"]);
    for (const m of [4, 8]) {
      const baseline = gemvBaseline(DOWN, m);
      assertEquals([baseline.name, baseline.class], ["gemm", "identical"]);
      assert(isGemmChoice(baseline.choice));
      assertEquals(baseline.choice.id, "gemm-f32");
      // The GEMM's program is one dispatch of the bf16 kernel of the default profile's geometry
      const program = gemvProgram(DOWN, m, baseline.choice);
      assertEquals(program.dispatches.length, 1);
      assert(program.dispatches[0].key.startsWith("linear:"), program.dispatches[0].key);
      assert(program.dispatches[0].key.includes(":wbf16"), program.dispatches[0].key);
    }
  });

  it("is what the default profile chooses at reorder too, unless the profile has a row of the shape", () => {
    for (const gemvCase of [DOWN, HEAD]) {
      const row = DEFAULT_GEOMETRY_PROFILE.linearGemv.parallelShapes.find(
        ({ storage, n, k }) => storage === "bf16" && n === gemvCase.n && k === gemvCase.k,
      );
      for (const m of gemvCase.rows) {
        const expected =
          row === undefined
            ? gemvBaseline(gemvCase, m).name
            : `${row.stagedActivations === true ? "staged" : "parallel"}-l${row.lanes}`;
        assertEquals(gemvProfileChoice(gemvCase, m, "reorder").name, expected, `M=${m}`);
      }
    }
  });
});

describe("the candidates of a float-stored case", () => {
  it("are the column forms and the parallel GEMV, plain and staged, at one row — without the subgroup form, which has no float kernel", () => {
    const candidates = gemvCandidates(DOWN, 1, "full", GATES);
    const identical = candidates.filter(({ class: kind }) => kind === "identical");
    const reorder = candidates.filter(({ class: kind }) => kind === "reorder");
    // The 5 × 5 grid of columns and read-ahead without the baseline
    assertEquals(identical.length, 24);
    assert(identical.every(({ choice }) => choice.id === "gemv-m1"));
    assertEquals(names(reorder).toSorted(), PARALLEL_NAMES.toSorted());
  });

  it("are the parallel GEMV alone past one row: a float storage has no row block", () => {
    for (const m of [4, 8]) {
      const candidates = gemvCandidates(DOWN, m, "full", GATES);
      assertEquals(names(candidates).toSorted(), PARALLEL_NAMES.toSorted(), `M=${m}`);
      assert(candidates.every((candidate) => candidate.class === "reorder"));
    }
  });

  it("leave out the lane count whose grid exceeds the device's limit on the 262144-column head", () => {
    const reorder = names(gemvCandidates(HEAD, 1, "full", GATES));
    assert(!reorder.includes("parallel-l32") && !reorder.includes("staged-l32"));
    assert(reorder.includes("parallel-l16") && reorder.includes("staged-l16"));
  });

  it("name the GEMM as both references of a row count past one when the profile has no row", () => {
    for (const plan of gemvMeasurementPlans(DOWN, "full", GATES)) {
      assertEquals(plan.profile.exact, plan.m === 1 ? "c32u4" : "gemm", `M=${plan.m}`);
      assert(!names(plan.candidates).includes(plan.baseline.name));
    }
  });
});

describe("the launch of a float-stored case's kernel", () => {
  it("binds no scale buffer, and the WGSL's roles accept it: the sequential GEMV, the GEMM and the parallel forms", () => {
    const kernels = gemvMeasurementPlans(DOWN, "full", GATES).flatMap((plan) =>
      [plan.baseline, ...plan.candidates].map((candidate) => ({ m: plan.m, candidate })),
    );
    assert(kernels.some(({ candidate }) => candidate.name === "gemm"));
    for (const { m, candidate } of kernels) {
      const launch = gemvLaunch(gemvProgram(DOWN, m, candidate.choice), "out", 65535, false);
      assertEquals(launch.bindings, ["x", "w", "bias", "out"], `${candidate.name} M=${m}`);
      assertBindingRoles(launch, parseStorageRoles(launch.wgsl));
    }
  });

  it("is refused by the WGSL's roles when it binds a scale buffer (fault: the launch of an integer case)", () => {
    const launch = gemvLaunch(gemvProgram(DOWN, 1, gemvBaseline(DOWN, 1).choice), "out", 65535);
    assertEquals(launch.bindings, ["x", "w", "bias", "out", "wscale"]);
    assertThrows(() => assertBindingRoles(launch, parseStorageRoles(launch.wgsl)));
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

/** A scale stream that must not be read: a float storage has no scale. */
const noScale = (): number => {
  throw new Error("the oracle read a scale for a float storage");
};

describe("the host oracle on a float-stored weight", () => {
  const immediately = async (): Promise<void> => undefined;
  const x = [1, 10, 100, 1000];

  it("reads an f32 word as the element itself, and no scale", async () => {
    const out = await gemvOracle(
      { storage: "f32", n: 2, k: 4 },
      1,
      {
        x: words(x.map(f32Bits)),
        weight: words([1.5, -2, 0.25, 4, 0, 0, 0, -1].map(f32Bits)),
        bias: words([f32Bits(0.5), f32Bits(-0.5)]),
        scale: noScale,
      },
      immediately,
    );
    assertEquals([...out], [1.5 - 20 + 25 + 4000 + 0.5, -1000 - 0.5]);
  });

  it("restores bf16 as the high 16 bits of an f32, the low half first (fault: read high half first)", async () => {
    // 1.5 = 0x3FC0_0000, −2 = 0xC000_0000, 0.25 = 0x3E80_0000, 4 = 0x4080_0000
    const weight = [0xc0003fc0, 0x40803e80];
    const out = await gemvOracle(
      { storage: "bf16", n: 1, k: 4 },
      1,
      { x: words(x.map(f32Bits)), weight: words(weight), bias: words([0]), scale: noScale },
      immediately,
    );
    assertEquals([...out], [1.5 - 20 + 25 + 4000]);
    // The same words read high half first would give −2 + 15 + 400 + 250
    assert(out[0] !== -2 + 15 + 400 + 250);
  });

  it("restores f16 as the half-float, the low half first, subnormals included (fault: an f16 half read as bf16)", async () => {
    // 1.5 = 0x3E00, −2 = 0xC000, the smallest subnormal 0x0001 = 2^-24, 4 = 0x4400
    const weight = [0xc0003e00, 0x44000001];
    const out = await gemvOracle(
      { storage: "f16", n: 1, k: 4 },
      1,
      { x: words(x.map(f32Bits)), weight: words(weight), bias: words([0]), scale: noScale },
      immediately,
    );
    assertEquals([...out], [1.5 - 20 + 100 * 2 ** -24 + 4000]);
    // 0x3E00 as the high half of an f32 is 0.125, not 1.5
    const asBf16 = await gemvOracle(
      { storage: "bf16", n: 1, k: 4 },
      1,
      { x: words(x.map(f32Bits)), weight: words(weight), bias: words([0]), scale: noScale },
      immediately,
    );
    assert(asBf16[0] !== out[0]);
  });
});

describe("the bf16 fill of the sweep's weights", () => {
  it("packs two finite, normal halves of magnitude in [2^-4, 1) per word, the same words for the same resource", () => {
    const next = resourceWordStream("bf16", 1);
    const again = resourceWordStream("bf16", 1);
    const value = new Float32Array(1);
    const bits = new Uint32Array(value.buffer);
    let negative = 0;
    for (let index = 0; index < 4096; index += 1) {
      const word = next();
      assertEquals(again(), word);
      for (const half of [word & 0xffff, word >>> 16]) {
        bits[0] = (half << 16) >>> 0;
        const magnitude = Math.abs(value[0]);
        assert(magnitude >= 2 ** -4 && magnitude < 1, `${half.toString(16)} = ${value[0]}`);
        if (value[0] < 0) negative += 1;
      }
    }
    assert(negative > 3000 && negative < 5200, `${negative} negative halves of 8192`);
  });
});

describe("the record's form of the GEMM", () => {
  it("writes the storage and the output form, and reads back as the choice that builds the same kernel", () => {
    const { choice } = gemvBaseline(DOWN, 4);
    const written = choiceRecord(choice, "bf16");
    assertEquals(written, { id: "gemm-f32", storage: "bf16", v4: true });
    assertEquals(choiceOfRecord(written), choice);
  });
});

/** A table of the records' adapter. */
const TABLE: GeometryProfile = {
  ...DEFAULT_GEOMETRY_PROFILE,
  id: "test-gpu",
  match: { vendor: GEMV_ADAPTER.vendor, architecture: GEMV_ADAPTER.architecture },
};

const input = (
  path: string,
  specs: readonly GemvCaseSpec[],
  change: (record: GemvSweepReport) => unknown = (record) => record,
): GemvSweepInput => ({
  report: change(gemvRecord(`2026-11-18T0${path.length % 10}:00:00.000Z`, specs)),
  path,
  sha256: `sha-${path}`,
});

/** A staged candidate of the case that is not the default profile's choice at reorder. */
const candidateOf = (gemvCase: GemvCase): { readonly name: string; readonly lanes: 8 | 16 } => {
  const lanes = gemvProfileChoice(gemvCase, 1, "reorder").name === "staged-l8" ? 16 : 8;
  return { name: `staged-l${lanes}`, lanes };
};

describe("deriveLinearGemvChoices on a float-stored case", () => {
  const { name, lanes } = candidateOf(DOWN);
  const fast = { [name]: BASE_TIME / 2 };
  const measured = (times: GemvCaseSpec["times"]): readonly GemvCaseSpec[] => [
    { caseId: DOWN.id, rows: [1, 4, 8], times },
  ];
  const decisionOf = (specs: readonly GemvCaseSpec[], second = specs) => {
    const derivation = deriveLinearGemvChoices([input("a.json", specs), input("bb.json", second)], {
      table: TABLE,
    });
    const decision = derivation.decisions.find(
      ({ field, cases }) => field === "parallelShapes" && cases.includes(DOWN.id),
    );
    assert(decision !== undefined);
    return { derivation, decision };
  };

  it("adopts a parallel row ×1.10 or more faster at one row and not slower than the default's choice at 4 and 8 rows — the GEMM, without a row —, as a bf16 row", () => {
    const { derivation, decision } = decisionOf(measured({ 1: fast, 4: fast, 8: fast }));
    assertEquals([decision.outcome, decision.derivedChoice], ["adopted", name]);
    assertEquals(decision.shape, "bf16 n1536 k12288");
    const row = derivation.linearGemv?.parallelShapes.findLast(
      ({ storage, n, k }) => storage === "bf16" && n === DOWN.n && k === DOWN.k,
    );
    assertEquals(row, {
      storage: "bf16",
      n: DOWN.n,
      k: DOWN.k,
      lanes,
      packedActivations: false,
      stagedActivations: true,
    });
  });

  it("keeps the default when the candidate is slower than the default's choice at 8 rows in one record (fault: ×0.95 there)", () => {
    const slowAtEight = { [name]: BASE_TIME / 0.95 };
    const { decision } = decisionOf(
      measured({ 1: fast, 4: fast, 8: fast }),
      measured({ 1: fast, 4: fast, 8: slowAtEight }),
    );
    assertEquals(decision.outcome, "kept");
    assertStringIncludes(decision.reason ?? "", "slower than the default's choice");
    assertStringIncludes(decision.reason ?? "", "M=8");
  });

  it("keeps the default when the records measured one row only: the row would serve 4 and 8 rows unmeasured", () => {
    const { decision } = decisionOf([{ caseId: DOWN.id, rows: [1], times: { 1: fast } }]);
    assertEquals(decision.outcome, "kept");
    assertStringIncludes(decision.reason ?? "", "evidence at M=4 from fewer than 2 records");
  });

  it("refuses a record whose GEMM row does not say its output form (fault: v4 removed)", () => {
    const broken = (record: GemvSweepReport): unknown => ({
      ...record,
      rows: record.rows.map((row) =>
        row.candidate === "gemm" ? { ...row, choice: { id: "gemm-f32", storage: "bf16" } } : row,
      ),
    });
    assertThrows(
      () =>
        deriveLinearGemvChoices(
          [input("a.json", measured({ 1: fast })), input("bb.json", measured({ 1: fast }), broken)],
          { table: TABLE },
        ),
      Error,
      "choice.v4 is not a boolean",
    );
  });

  it("plans the float cases at 1, 4 and 8 rows, and lists the parallel forms of a case no record holds", () => {
    const planning = planGemvMeasurements([
      input("a.json", [{ caseId: "i4g32-n8960-k1536", rows: [1, 4, 8] }]),
      input("bb.json", [{ caseId: "i4g32-n8960-k1536", rows: [1, 4, 8] }]),
    ]);
    for (const m of [1, 4, 8]) {
      const missing = planning.missing.find(
        (entry) => entry.caseId === DOWN.id && entry.m === m,
      )?.candidates;
      assert(missing !== undefined, `M=${m}`);
      const reference = gemvProfileChoice(DOWN, m, "reorder").name;
      assertEquals(
        missing.filter((candidate) => !candidate.startsWith("c")).toSorted(),
        PARALLEL_NAMES.filter((candidate) => candidate !== reference).toSorted(),
        `M=${m}`,
      );
    }
  });
});
