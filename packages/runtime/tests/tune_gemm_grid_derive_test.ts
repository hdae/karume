// The derivation of the grid-fill rule's two integers from grid records (`deriveGemmGrid` —
// src/tune/gemm-grid-derive.ts), on hand-built records (tests/helpers/grid-records.ts). No GPU.
//
// 1. Replaying the recon of 2026-11-28 (its interleaved medians of the ladder on the grid cases,
//    against `nvidia-ampere`'s tiles) adopts (256, 48), the pair the recon fitted on all 125 shapes.
// 2. The rule's parts each refuse: a ladder row not identical to the default refuses its tile (and
//    so every pair that picks it there), a tile slower than today beyond the bar refuses the pairs
//    that pick it, a weighted gain below the threshold refuses every pair; one record that is not
//    steady is no evidence, two are.
// 3. Records that do not stand for the current runtime or for the table's device are refused.
//
// Fault injection (recorded 2026-11-28): with the identity check of a case's tile removed from
// `caseOutcome`, the first case of part 2 adopts a field where it must carry none.

import { assert, assertEquals, assertStringIncludes, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { DEFAULT_GEOMETRY_PROFILE, type GeometryProfile } from "../src/kernels/geometry-profile.ts";
import { NVIDIA_AMPERE } from "../src/kernels/geometry-profiles/nvidia-ampere.ts";
import { GEMM_GRID_LADDER, GEMM_GRID_MAX_PADDING, GEMM_GRID_MIN_K } from "../src/tune/gemm-grid.ts";
import { deriveGemmGrid, type GemmGridInput } from "../src/tune/gemm-grid-derive.ts";
import {
  GRID_ADAPTER,
  gridRecord,
  type GridRecordOptions,
  RECON_GRID_TIMES,
  UNSTEADY_ROUNDS,
} from "./helpers/grid-records.ts";

/**
 * `nvidia-ampere` as the table the recon measured against — its M buckets, without the grid-fill
 * rule it carries since (derived from this machine's grid records) —, its provenance's adapter the
 * records'.
 */
const FAMILIES = NVIDIA_AMPERE.provenance?.families ?? {};
const { gemmGrid: _derivedSince, ...BUCKETS_ONLY } = NVIDIA_AMPERE;

const TABLE: GeometryProfile = {
  ...BUCKETS_ONLY,
  provenance: { adapter: GRID_ADAPTER, families: FAMILIES },
};

const input = (options: GridRecordOptions = {}, name = "a"): GemmGridInput => ({
  report: gridRecord(options),
  path: `grid-${name}.json`,
  sha256: name,
});

describe("replaying the recon's grid measurements", () => {
  it("adopts (256, 48) over the code's ladder, K floor and padding bound", () => {
    const derivation = deriveGemmGrid([input()], { table: TABLE });
    assertEquals(derivation.gemmGrid, {
      ladder: GEMM_GRID_LADDER,
      keepLargest: { minWorkgroups: 256, minK: GEMM_GRID_MIN_K },
      minWorkgroups: 48,
      maxPadding: GEMM_GRID_MAX_PADDING,
    });
    assertEquals(derivation.pairs.length, 35);
    assertStringIncludes(derivation.verdicts[0], "adopted (256, 48)");
    // The shown pair is the chosen one, with a ratio per case
    assertEquals(derivation.shown?.largestMinWorkgroups, 256);
    assertEquals(derivation.shown?.cases.length, Object.keys(RECON_GRID_TIMES).length);
    const weighted = derivation.shown?.weightedTime ?? 1;
    assert(weighted < 1 / 1.05, `weighted time ${weighted}`);
  });

  it("states the record's class and that a steady record is evidence alone", () => {
    const [record] = deriveGemmGrid([input()], { table: TABLE }).records;
    assertEquals(record.steadiness.class, "steady");
    assertEquals(record.reading, "one-record");
  });
});

describe("the rule's parts", () => {
  it("refuses a tile not identical to the default in a case, and every pair that picks it there", () => {
    // depth-anything fc2: every pair picks 64 × 64 (33 workgroups of 128 × 128)
    const derivation = deriveGemmGrid(
      [
        input({
          identical: (caseId, step) =>
            !(caseId === "grid-linear-m1370-n384-k1536-wf32" && step === 1),
        }),
      ],
      { table: TABLE },
    );
    assertEquals(derivation.gemmGrid, undefined);
    assert(derivation.pairs.every((pair) => !pair.qualifies));
    assertStringIncludes(
      derivation.pairs[0].reason ?? "",
      "reg64x64r4x4w16 is not identical to the default in grid-a.json",
    );
    assertStringIncludes(derivation.verdicts[0], "carries no field");
  });

  it("refuses a pair whose tile is slower than today's beyond the bar, and keeps the pairs that do not pick it", () => {
    // Gemma 4 E2B's gate / up at N 6,144: 288 workgroups of 128 × 128 — a pair with a largest
    // threshold above 288 steps down to 64 × 64; make that tile ×1.10 slower than today's 128 × 128
    const caseId = "grid-linear-m768-n6144-k1536-wbf16";
    const times = RECON_GRID_TIMES[caseId];
    const derivation = deriveGemmGrid(
      [
        input({
          timeOf: (id, step) =>
            id === caseId && step === 1 ? times[0] * 1.1 : (RECON_GRID_TIMES[id] ?? [])[step],
        }),
      ],
      { table: TABLE },
    );
    const at = (largest: number, smaller: number) =>
      derivation.pairs.find(
        (pair) => pair.largestMinWorkgroups === largest && pair.minWorkgroups === smaller,
      );
    assertEquals(at(320, 48)?.qualifies, false);
    assertStringIncludes(
      at(320, 48)?.reason ?? "",
      `${caseId}: reg64x64r4x4w16 is slower than today's`,
    );
    assertEquals(at(256, 48)?.qualifies, true);
    assertEquals(derivation.gemmGrid?.keepLargest.minWorkgroups, 256);
  });

  it("refuses every pair when the weighted gain is below the threshold", () => {
    const derivation = deriveGemmGrid([input()], { table: TABLE, minSpeedup: 1.5 });
    assertEquals(derivation.gemmGrid, undefined);
    assertStringIncludes(
      derivation.pairs.find(
        (pair) =>
          pair.weightedTime !== undefined && pair.reason?.includes("weighted gain") === true,
      )?.reason ?? "",
      "is below ×1.500",
    );
  });

  it("reads one record that is not steady as no evidence, and two such records as evidence", () => {
    const one = deriveGemmGrid([input({ rounds: UNSTEADY_ROUNDS })], { table: TABLE });
    assertEquals(one.records[0].steadiness.class, "unsteady");
    assertEquals(one.gemmGrid, undefined);
    assertStringIncludes(one.pairs[0].reason ?? "", "is read by one record that is not steady");
    const two = deriveGemmGrid(
      [
        input({ rounds: UNSTEADY_ROUNDS }, "a"),
        input({ rounds: UNSTEADY_ROUNDS, date: "2026-11-28T01:00:00.000Z" }, "b"),
      ],
      { table: TABLE },
    );
    assertEquals(two.gemmGrid?.keepLargest.minWorkgroups, 256);
    assertEquals(two.gemmGrid?.minWorkgroups, 48);
  });

  it("compares with the table's own tile: on the default profile, today's tiles are its buckets'", () => {
    const derivation = deriveGemmGrid([input()], { table: DEFAULT_GEOMETRY_PROFILE });
    const entry = derivation.shown?.cases.find(
      ({ caseId }) => caseId === "grid-linear-m256-n512-k512-wbf16",
    );
    // M 256 is the default's 65–512 bucket (64 × 32); nvidia-ampere's is 64 × 64
    assertEquals(entry?.today, "reg64x32r4x4w8");
  });
});

describe("what the derivation refuses", () => {
  it("no record, a record given twice, records of two adapters", () => {
    assertThrows(() => deriveGemmGrid([], { table: TABLE }), Error, "No grid records");
    assertThrows(
      () =>
        deriveGemmGrid([input({}, "a"), { ...input({}, "a"), path: "copy.json" }], {
          table: TABLE,
        }),
      Error,
      "given twice",
    );
    assertThrows(
      () =>
        deriveGemmGrid(
          [input({}, "a"), input({ adapter: { ...GRID_ADAPTER, device: "0x9999" } }, "b")],
          { table: TABLE },
        ),
      Error,
      "the integers are one device's",
    );
  });

  it("a record of another set, timing, case set or runtime", () => {
    const changed = (patch: (record: Record<string, unknown>) => Record<string, unknown>) => ({
      report: patch(gridRecord()),
      path: "patched.json",
      sha256: "p",
    });
    for (const [what, patch, message] of [
      [
        "another set",
        (r: Record<string, unknown>) => ({
          ...r,
          settings: { ...(r.settings as object), candidateSet: "quick+" },
        }),
        "not grid",
      ],
      [
        "wall clock",
        (r: Record<string, unknown>) => ({
          ...r,
          gpuTiming: { feature: false, unit: "wall", quantized: false },
        }),
        "gpuTiming.unit",
      ],
      [
        "quantised",
        (r: Record<string, unknown>) => ({
          ...r,
          gpuTiming: { feature: true, unit: "ns", quantized: true },
        }),
        "quantized",
      ],
      [
        "another case set",
        (r: Record<string, unknown>) => ({ ...r, caseSets: { gemm: "0000" } }),
        "caseSets.gemm",
      ],
      [
        "other candidate kernels",
        (r: Record<string, unknown>) => ({ ...r, candidateKernels: "0000" }),
        "candidateKernels",
      ],
    ] as const) {
      assertThrows(() => deriveGemmGrid([changed(patch)], { table: TABLE }), Error, message, what);
    }
  });

  it("takes records made after a driver update (the description is the driver's under Node)", () => {
    const updated: GeometryProfile = {
      ...TABLE,
      provenance: {
        adapter: { ...GRID_ADAPTER, description: "an older driver" },
        families: FAMILIES,
      },
    };
    assertEquals(deriveGemmGrid([input()], { table: updated }).gemmGrid?.minWorkgroups, 48);
  });

  it("a table of another device or match, and a shared table", () => {
    const other: GeometryProfile = {
      ...TABLE,
      provenance: { adapter: { ...GRID_ADAPTER, device: "0x9999" }, families: FAMILIES },
    };
    assertThrows(
      () => deriveGemmGrid([input()], { table: { ...TABLE, match: { vendor: "apple" } } }),
      Error,
      "is not table 'nvidia-ampere''s match",
    );
    assertThrows(
      () => deriveGemmGrid([input()], { table: other }),
      Error,
      "a table takes the rule of its own device",
    );
    const shared: GeometryProfile = {
      ...TABLE,
      provenance: {
        families: FAMILIES,
        adapters: [
          { ...GRID_ADAPTER, description: "one" },
          { ...GRID_ADAPTER, description: "two" },
        ],
      },
    };
    assertThrows(() => deriveGemmGrid([input()], { table: shared }), Error, "shared table");
  });
});
