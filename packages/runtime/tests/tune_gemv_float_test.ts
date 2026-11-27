// The GEMV sweep and its derivation over float-stored cases (bf16 / f16 / f32), without a GPU: the
// cases' row counts, the baseline — the sequential GEMV at one row, the GEMM past it, where such a
// weight has no sequential GEMV —, the candidates, the launch without a scale buffer, the host
// oracle's restoration of each storage, the record's form of the GEMM and the derivation of a
// parallel row read against the GEMM at M = 4 and 8.
//
// The GEMM reference takes its tile geometry from the profile the device resolves to, so a GEMM a
// record measured under another geometry is no reference for the table derived for.
//
// Faults injected: a bf16 word read high half first, an f16 half read as bf16 (the oracle's value
// moves); a launch that binds a scale buffer (the WGSL's roles refuse it); a parallel candidate
// slower than the GEMM at M = 8 (the row is not adopted); a record whose GEMM row has no `v4`
// (refused); records measured under the default profile's GEMM derived for a table with another
// GEMM geometry (no row is adopted, the groups past one row are counted as without a usable
// reference).

import { assert, assertEquals, assertStringIncludes, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { deriveLinearGemvChoices, type GemvSweepInput, planGemvMeasurements } from "../tune.ts";
import { parseStorageRoles } from "../src/gpu/pipeline-cache.ts";
import { defaultGemmGeometry } from "../src/kernels/gemm-geometry.ts";
import { DEFAULT_GEOMETRY_PROFILE, type GeometryProfile } from "../src/kernels/geometry-profile.ts";
import {
  type GemvCandidate,
  type GemvGates,
  type GemvM1Group,
  type GemvM1Row,
  gemvBaseline,
  gemvCandidates,
  gemvCarriedFromM1,
  gemvMeasurementPlans,
  gemvProfileChoice,
  gemvProgram,
  gemvRowCountsIn,
  isGemmChoice,
} from "../src/tune/gemv-candidates.ts";
import { isFloatStorage } from "../src/kernels/linear-gemv.ts";
import { GEMV_CASES, type GemvCase } from "../src/tune/gemv-cases.ts";
import { gemvKernelId } from "../src/tune/gemv-fingerprint.ts";
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

/** Regular E2B's down of the 12288-wide MLP layers, in bf16 (the default profile has a parallel row of it). */
const DOWN = caseOf("bf16-n1536-k12288");
/** The default profile's sequential variant of {@link DOWN}'s shape (a per-shape exception). */
const DOWN_SEQUENTIAL = "c16u16";
/** The default profile's parallel row of {@link DOWN}'s shape: plain, 16 lanes. */
const DOWN_PARALLEL = "parallel-l16";
/** Regular E2B's gate / up of the 12288-wide MLP layers, in bf16: no parallel row of the default. */
const UP = caseOf("bf16-n12288-k1536");
/** The default profile's sequential variant of {@link UP}'s shape (a per-shape exception). */
const UP_SEQUENTIAL = "c16u8";
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
    const float = GEMV_CASES.filter(({ storage }) => isFloatStorage(storage));
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
    const integer = GEMV_CASES.filter(({ storage }) => !isFloatStorage(storage));
    assertEquals(GEMV_CASES.slice(0, integer.length), integer);
    assertEquals(integer[0].id, "i4g32-n8960-k1536");
    for (const gemvCase of integer) assertEquals(gemvCase.rows, [1, 4, 8, 16, 64]);
  });
});

describe("the baseline of a float-stored case", () => {
  it("is the sequential GEMV at one row and the GEMM past it, both of the reference order", () => {
    const one = gemvBaseline(DOWN, 1);
    assertEquals([one.name, one.class, one.choice.id], [DOWN_SEQUENTIAL, "identical", "gemv-m1"]);
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

  it("is what the default profile chooses at reorder too on a shape it has no parallel row of; on a shape with one, the default's choice at reorder is the row's parallel GEMV at every row count", () => {
    const rowOf = ({ n, k }: GemvCase) =>
      DEFAULT_GEOMETRY_PROFILE.linearGemv.parallelShapes.find(
        (row) => row.storage === "bf16" && row.n === n && row.k === k,
      );
    for (const gemvCase of [UP, HEAD]) {
      assertEquals(rowOf(gemvCase), undefined, gemvCase.id);
      for (const m of gemvCase.rows) {
        assertEquals(
          gemvProfileChoice(gemvCase, m, "reorder").name,
          gemvBaseline(gemvCase, m).name,
          `${gemvCase.id} M=${m}`,
        );
      }
    }
    assertEquals(rowOf(DOWN), {
      storage: "bf16",
      n: DOWN.n,
      k: DOWN.k,
      lanes: 16,
      packedActivations: false,
    });
    for (const m of DOWN.rows) {
      assertEquals(gemvProfileChoice(DOWN, m, "reorder").name, DOWN_PARALLEL, `M=${m}`);
      assertEquals(gemvProfileChoice(DOWN, m, "exact").name, gemvBaseline(DOWN, m).name, `M=${m}`);
    }
  });
});

/** A profile whose GEMM geometry at few rows is not the default profile's (one bucket for every row count). */
const ONE_BUCKET: GeometryProfile = {
  ...DEFAULT_GEOMETRY_PROFILE,
  id: "test-one-bucket",
  gemmRows: [{ maxRows: Number.POSITIVE_INFINITY, geometry: defaultGemmGeometry() }],
};

describe("the GEMM reference of a float-stored case", () => {
  it("is built with the tile geometry of the profile given — the one the measuring device resolves to", () => {
    const { choice } = gemvBaseline(DOWN, 4);
    const underDefault = gemvProgram(DOWN, 4, choice).dispatches[0];
    const underOther = gemvProgram(DOWN, 4, choice, ONE_BUCKET).dispatches[0];
    assertEquals(underDefault.key, "linear:v2:f32:reg16x16r1x4w4v4:wbf16");
    assertEquals(underOther.key, "linear:v2:f32:reg128x128r8x8w16v4:wbf16");
    // 1536 columns in tiles of 128 instead of 16
    assertEquals(underOther.grid, {
      kind: "tiled",
      axes: [
        { items: 1536, tile: 128 },
        { items: 4, tile: 128 },
        { items: 1, tile: 1 },
      ],
    });
    assert(
      gemvKernelId(DOWN, 4, choice) !==
        gemvKernelId(DOWN, 4, choice, (one, m, chosen) => gemvProgram(one, m, chosen, ONE_BUCKET)),
    );
  });

  it("leaves every GEMV kernel as it is: the sequential GEMV and the parallel forms read nothing of the profile", () => {
    const kernels = [gemvBaseline(DOWN, 1), ...gemvCandidates(DOWN, 4, "full", GATES)];
    for (const { name, choice } of kernels) {
      const m = name.startsWith("c") ? 1 : 4;
      assertEquals(
        gemvKernelId(DOWN, m, choice, (one, rows, chosen) =>
          gemvProgram(one, rows, chosen, ONE_BUCKET),
        ),
        gemvKernelId(DOWN, m, choice),
        name,
      );
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
    for (const plan of gemvMeasurementPlans(UP, "full", GATES)) {
      assertEquals(plan.profile.exact, plan.m === 1 ? UP_SEQUENTIAL : "gemm", `M=${plan.m}`);
      assertEquals(plan.profile.reorder, plan.profile.exact, `M=${plan.m}`);
      assert(!names(plan.candidates).includes(plan.baseline.name));
    }
  });
});

describe("the candidates of a float-stored case carried from M = 1 on to M = 4 and 8", () => {
  /** At M = 4 and 8 both references are the GEMM, which the M = 1 group has no row of. */
  const REFERENCES = { exact: "gemm", reorder: "gemm" } as const;
  const SEQUENTIAL = 1000;
  /** The M = 1 group: the default's choice at reorder is the sequential GEMV, timed at 1000. */
  const groupOf = (rows: readonly GemvM1Row[], spread = 1.01): GemvM1Group => ({
    baselineSpread: spread,
    profile: { reorder: "c32u4" },
    rows: [{ candidate: "c32u4", perDispatch: SEQUENTIAL }, ...rows],
  });
  const held = (candidate: string, gain: number): GemvM1Row => ({
    candidate,
    perDispatch: SEQUENTIAL / gain,
    withinBand: true,
    nonFiniteOutputs: 0,
  });
  const carried = (group: GemvM1Group | undefined, candidate: string, m: number): boolean =>
    gemvCarriedFromM1(m, { name: candidate, class: "reorder" }, REFERENCES, group);

  it("names the GEMM as both references past one row, and the sequential GEMV as the default's choice at reorder at one row", () => {
    const plans = gemvMeasurementPlans(UP, "full", GATES);
    assertEquals(
      plans.map(({ m, profile }) => [m, profile.exact, profile.reorder]),
      [
        [1, UP_SEQUENTIAL, UP_SEQUENTIAL],
        [4, "gemm", "gemm"],
        [8, "gemm", "gemm"],
      ],
    );
  });

  it("carries a parallel form ×1.03 (the lowest margin) or more faster than the sequential GEMV at one row, to be read against the GEMM (fault: ×1.02 is skipped)", () => {
    for (const m of [4, 8]) {
      assertEquals(carried(groupOf([held("parallel-l32", 5.2)]), "parallel-l32", m), true);
      assertEquals(carried(groupOf([held("parallel-l32", 1.03)]), "parallel-l32", m), true);
      assertEquals(carried(groupOf([held("parallel-l2", 1.02)]), "parallel-l2", m), false);
    }
  });

  it("skips a form the quick look screened or that left the band at one row, and carries every held form from an unsteady group", () => {
    const screened: GemvM1Row = { candidate: "staged-l2", screened: true };
    assertEquals(carried(groupOf([screened]), "staged-l2", 4), false);
    const outside: GemvM1Row = { ...held("parallel-l4", 3), withinBand: false };
    assertEquals(carried(groupOf([outside]), "parallel-l4", 4), false);
    // The sequential baseline's two timing modes: the times of an unsteady group rule nothing out
    assertEquals(carried(groupOf([held("parallel-l2", 0.5)], 2.46), "parallel-l2", 8), true);
  });

  it("always measures the GEMM past one row: it is the reference, and the M = 1 group has no row of it", () => {
    const group = groupOf([held("parallel-l32", 5.2)]);
    for (const m of [4, 8]) {
      assertEquals(
        gemvCarriedFromM1(m, { name: "gemm", class: "identical" }, REFERENCES, group),
        true,
      );
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
  profile?: GeometryProfile,
): GemvSweepInput => ({
  report: change(
    gemvRecord(
      `2026-11-18T0${path.length % 10}:00:00.000Z`,
      specs,
      profile === undefined ? {} : { profile },
    ),
  ),
  path,
  sha256: `sha-${path}`,
});

describe("deriveLinearGemvChoices on a float-stored case", () => {
  /** A staged candidate: the default profile has no row of the shape, so any lane count is one. */
  const name = "staged-l8";
  const lanes = 8;
  const fast = { [name]: BASE_TIME / 2 };
  const measured = (times: GemvCaseSpec["times"]): readonly GemvCaseSpec[] => [
    { caseId: UP.id, rows: [1, 4, 8], times },
  ];
  const decisionOf = (specs: readonly GemvCaseSpec[], second = specs) => {
    const derivation = deriveLinearGemvChoices([input("a.json", specs), input("bb.json", second)], {
      table: TABLE,
    });
    const decision = derivation.decisions.find(
      ({ field, cases }) => field === "parallelShapes" && cases.includes(UP.id),
    );
    assert(decision !== undefined);
    return { derivation, decision };
  };

  it("adopts a parallel row ×1.10 or more faster at one row and not slower than the default's choice at 4 and 8 rows — the GEMM, without a row —, as a bf16 row", () => {
    const { derivation, decision } = decisionOf(measured({ 1: fast, 4: fast, 8: fast }));
    assertEquals([decision.outcome, decision.derivedChoice], ["adopted", name]);
    assertEquals(decision.shape, "bf16 n12288 k1536");
    const row = derivation.linearGemv?.parallelShapes.findLast(
      ({ storage, n, k }) => storage === "bf16" && n === UP.n && k === UP.k,
    );
    assertEquals(row, {
      storage: "bf16",
      n: UP.n,
      k: UP.k,
      lanes,
      packedActivations: false,
      stagedActivations: true,
    });
  });

  it("gives the adopted float row no row block: a float case has no row-blocked candidate to measure or to read", () => {
    for (const m of [4, 8]) {
      for (const { choice, name: candidate } of gemvCandidates(UP, m, "full", GATES)) {
        assert(!("parallelRows" in choice) || choice.parallelRows === undefined, candidate);
      }
    }
    const { derivation, decision } = decisionOf(measured({ 1: fast, 4: fast, 8: fast }));
    assertEquals(decision.outcome, "adopted");
    assertEquals([decision.defaultRowBlock, decision.derivedRowBlock], [undefined, undefined]);
    assert(
      derivation.linearGemv?.parallelShapes.every(
        (row) => !isFloatStorage(row.storage) || row.rowBlock === undefined,
      ),
    );
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
    const { decision } = decisionOf([{ caseId: UP.id, rows: [1], times: { 1: fast } }]);
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

  describe("for a table whose GEMM geometry is not the one the records were measured under", () => {
    /** The records' adapter's table, with the one-bucket GEMM geometry. */
    const OTHER_TABLE: GeometryProfile = { ...TABLE, gemmRows: ONE_BUCKET.gemmRows };
    const specs = measured({ 1: fast, 4: fast, 8: fast });
    const deriveFor = (table: GeometryProfile, recordsUnder?: GeometryProfile) => {
      const derivation = deriveLinearGemvChoices(
        [
          input("a.json", specs, undefined, recordsUnder),
          input("bb.json", specs, undefined, recordsUnder),
        ],
        { table },
      );
      const decision = derivation.decisions.find(
        ({ field, cases }) => field === "parallelShapes" && cases.includes(UP.id),
      );
      assert(decision !== undefined);
      return { derivation, decision };
    };

    it("adopts no row and reads no group past one row against a GEMM measured under another profile, naming both profiles (fault: records of the default profile's GEMM)", () => {
      const { derivation, decision } = deriveFor(OTHER_TABLE);
      assertEquals(decision.outcome, "kept");
      assertStringIncludes(decision.reason ?? "", "evidence at M=4 from fewer than 2 records");
      assertStringIncludes(
        decision.reason ?? "",
        "the GEMM reference in [1] at M=4 was measured under the profile 'default', and it is not the GEMM kernel the table 'test-gpu' runs there",
      );
      // Past one row the GEMM is the reference of both classes (the case has no parallel row): each
      // record's groups at M = 4 and 8 have no usable reference at either level, and stay current
      assertEquals(
        derivation.provenance.records.map(({ groups }) => groups.withoutReference),
        [
          { exact: 2, reorder: 2 },
          { exact: 2, reorder: 2 },
        ],
      );
      assertEquals(derivation.stale, []);
      // The same records are evidence for the table they were measured under
      assertEquals(deriveFor(TABLE).decision.outcome, "adopted");
    });

    it("adopts the row from records measured under the table's own GEMM geometry", () => {
      const { derivation, decision } = deriveFor(OTHER_TABLE, OTHER_TABLE);
      assertEquals(decision.outcome, "adopted");
      assertEquals(derivation.stale, []);
    });

    it("says of a record without the field that it does not name its profile", () => {
      const unnamed = (record: GemvSweepReport): unknown => {
        const { geometryProfile: _dropped, ...rest } = record;
        return rest;
      };
      const derivation = deriveLinearGemvChoices(
        [input("a.json", specs, unnamed), input("bb.json", specs, unnamed)],
        { table: OTHER_TABLE },
      );
      const parallelOf = (of: typeof derivation) =>
        of.decisions.find(
          ({ field, cases }) => field === "parallelShapes" && cases.includes(UP.id),
        );
      assertStringIncludes(
        parallelOf(derivation)?.reason ?? "",
        "measured under a profile the record does not name",
      );
      // …and reads it where its GEMM is the table's: a record's kernel fingerprint decides
      const same = deriveLinearGemvChoices(
        [input("a.json", specs, unnamed), input("bb.json", specs, unnamed)],
        { table: TABLE },
      );
      assertEquals(same.stale, []);
      assertEquals(parallelOf(same)?.outcome, "adopted");
    });

    it("keeps an integer case's evidence: its kernels are the GEMV family's under any profile", () => {
      const ple = "i4g32-n8960-k1536";
      const derivation = deriveLinearGemvChoices(
        [
          input("a.json", [{ caseId: ple, rows: [1, 4, 8] }]),
          input("bb.json", [{ caseId: ple, rows: [1, 4, 8] }]),
        ],
        { table: OTHER_TABLE },
      );
      assertEquals(derivation.stale, []);
    });

    it("plans the groups past one row again, and nothing at one row", () => {
      const records = [input("a.json", specs), input("bb.json", specs)];
      const lacking = (table: GeometryProfile): number[] =>
        planGemvMeasurements(records, { table })
          .missing.filter(({ caseId }) => caseId === UP.id)
          .map(({ m }) => m);
      assertEquals(lacking(TABLE), []);
      assertEquals(lacking(OTHER_TABLE), [4, 8]);
    });
  });

  it("refuses a record whose profile id is not a string (fault: a number)", () => {
    assertThrows(
      () =>
        deriveLinearGemvChoices(
          [
            input("a.json", measured({ 1: fast })),
            input("bb.json", measured({ 1: fast }), (record) => ({
              ...record,
              geometryProfile: 3,
            })),
          ],
          { table: TABLE },
        ),
      Error,
      "geometryProfile is invalid",
    );
  });

  it("plans the float cases at 1, 4 and 8 rows, and lists the parallel forms of a case no record holds", () => {
    const planning = planGemvMeasurements([
      input("a.json", [{ caseId: "i4g32-n8960-k1536", rows: [1, 4, 8] }]),
      input("bb.json", [{ caseId: "i4g32-n8960-k1536", rows: [1, 4, 8] }]),
    ]);
    for (const m of [1, 4, 8]) {
      const missing = planning.missing.find(
        (entry) => entry.caseId === UP.id && entry.m === m,
      )?.candidates;
      assert(missing !== undefined, `M=${m}`);
      // The reference at reorder (the sequential GEMV, the GEMM) is no parallel form
      assertEquals(
        missing.filter((candidate) => !candidate.startsWith("c")).toSorted(),
        PARALLEL_NAMES.toSorted(),
        `M=${m}`,
      );
    }
  });
});
