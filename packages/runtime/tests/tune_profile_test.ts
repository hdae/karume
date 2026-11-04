// The check of the geometry profile generator (deriveProfile / parseSweepReport /
// profileRowsSegments of src/tune/derive.ts). No GPU is used — the sweep records are small
// synthetic JSON (helpers/sweep-records.ts).
//
// What this pins: (1) rule extraction (the ratio-to-default gate, the output-match gate, winning on
// every case of the class, comparison by geometric mean, the default for a class with no case in
// the sweep) and (2) not mixing adapters. The TS rendering of the output and the CLI
// (`main.ts profile --check`) are pinned by tools/geometry-sweep/profile_test.ts.
//
// Also pinned here: (3) the injected table built by buildGeometryProfile (its fields, match,
// provenance, the kernels fingerprint it cannot derive, description and opt-in tables), and (4) the
// record-field gates of deriveGeometryProfile (ADR 0117 decision 8: caseSet / defaultKernels /
// candidateKernels; provenance.userAgent from the records, ADR 0117 addendum 2026-10-02; the
// injection gate on the returned table). The JSON round trip of a table is pinned by
// tune_profile_json_test.ts and the matching by tune_mismatch_test.ts.

import {
  assert,
  assertEquals,
  assertInstanceOf,
  assertStringIncludes,
  assertThrows,
} from "@std/assert";
import { describe, it } from "vite-plus/test";
import { deriveGeometryProfile, type GeometrySweepInput, sweepCaseSetId } from "../tune.ts";
import { gemmMTileGeometry } from "../src/kernels/gemm.ts";
import {
  assertGeometryProfile,
  DEFAULT_GEOMETRY_PROFILE,
  selectGeometryProfile,
} from "../src/kernels/geometry-profile.ts";
import {
  GEMM_ROWS_BUCKETS,
  type GemmGeometry,
  gemmGeometryForRows,
} from "../src/kernels/gemm-geometry.ts";
import { defaultI8a8Geometry, type I8a8Geometry } from "../src/kernels/i8a8-geometry.ts";
import { PROFILE_GEMM_ROWS_BOUNDS, SWEEP_CASES } from "../src/tune/cases.ts";
import { conv2dCandidate, gemmCandidate, i8a8Candidate } from "../src/tune/geometries.ts";
import {
  buildGeometryProfile,
  deriveProfile,
  geometryProfileJson,
  parseSweepReport,
  profileRowsSegments,
  type SlotVerdict,
  type SweepSource,
  verdictLines,
} from "../src/tune/derive.ts";
import {
  candidateKernelsId,
  type CasePlanner,
  geometryProfileKernelsId,
  KernelsIdError,
  sweepCandidateKernelsId,
} from "../src/tune/fingerprint.ts";
import { casePlan } from "../src/tune/harness.ts";
import {
  A,
  ADAPTER,
  B,
  BIG,
  bmmCase,
  type CaseRef,
  caseRows,
  FAST,
  linearCase,
  matmulCase,
  OPTIONS,
  rejectionOf,
  type Repeats,
  report,
  row,
  ROWS_SLOT_NAMES,
  SMALL,
  source,
  SOURCES,
  SPEC,
  verdictOf,
} from "./helpers/sweep-records.ts";

describe("deriveProfile: rule extraction", () => {
  it("a class with only geometries below --min-speedup of the default gets the geometry of the sweep's default row", () => {
    const verdicts = deriveProfile(
      [
        source([
          ...caseRows(linearCase(1024), BIG, [[A, { speedup: 1.04 }]]),
          ...caseRows(linearCase(4096), BIG, [[A, { speedup: 1.3 }]]),
        ]),
      ],
      OPTIONS,
    );
    const verdict = verdictOf(verdicts, "gemmRows > 512");
    assertEquals(verdict.outcome.kind, "default");
    assertEquals(verdict.outcome.geometry, BIG);
    assertStringIncludes(rejectionOf(verdict, A), "linear-m1024: ×1.040 < ×1.050");
  });

  it("does not adopt a geometry whose output differs from the default's, even if it is the fastest", () => {
    const verdict = verdictOf(
      deriveProfile(
        [
          source(
            caseRows(linearCase(1024), BIG, [
              [A, { speedup: 2, identical: false }],
              [B, { speedup: 1.3 }],
            ]),
          ),
        ],
        OPTIONS,
      ),
      "gemmRows > 512",
    );
    assertEquals(verdict.outcome.kind, "adopted");
    assertEquals(verdict.outcome.geometry, B);
    assertStringIncludes(rejectionOf(verdict, A), "output differs from the default");
  });

  it("does not adopt a geometry that misses the ratio-to-default gate on even one case of the class, even with the highest geometric mean", () => {
    const verdict = verdictOf(
      deriveProfile(
        [
          source([
            ...caseRows(linearCase(1024), BIG, [
              [A, { speedup: 3 }],
              [B, { speedup: 1.2 }],
            ]),
            ...caseRows(linearCase(4096), BIG, [
              [A, { speedup: 1.01 }],
              [B, { speedup: 1.2 }],
            ]),
          ]),
        ],
        OPTIONS,
      ),
      "gemmRows > 512",
    );
    assertEquals(verdict.outcome.geometry, B);
    assertStringIncludes(rejectionOf(verdict, A), "linear-m4096: ×1.010");
  });

  it("does not adopt a geometry that was not measured or failed on any case of the class", () => {
    const verdict = verdictOf(
      deriveProfile(
        [
          source([
            ...caseRows(linearCase(1024), BIG, [
              [A, { speedup: 2 }],
              [B, { speedup: 1.5 }],
            ]),
            ...caseRows(linearCase(4096), BIG, [[B, { speedup: 1.5, error: "device lost" }]]),
          ]),
        ],
        OPTIONS,
      ),
      "gemmRows > 512",
    );
    assertEquals(verdict.outcome.kind, "default");
    assertStringIncludes(rejectionOf(verdict, A), "not measured in linear-m4096");
    assertStringIncludes(rejectionOf(verdict, B), "failed in linear-m4096 (device lost)");
  });

  it("compares candidates by geometric mean (a pair for which the arithmetic mean picks the other)", () => {
    // A: (1.6, 1.06) arithmetic 1.33, geometric 1.302 / B: (1.31, 1.31) arithmetic 1.31, geometric
    // 1.31
    const verdict = verdictOf(
      deriveProfile(
        [
          source([
            ...caseRows(linearCase(1024), BIG, [
              [A, { speedup: 1.6 }],
              [B, { speedup: 1.31 }],
            ]),
            ...caseRows(linearCase(4096), BIG, [
              [A, { speedup: 1.06 }],
              [B, { speedup: 1.31 }],
            ]),
          ]),
        ],
        OPTIONS,
      ),
      "gemmRows > 512",
    );
    assertEquals(verdict.outcome.kind, "adopted");
    assertEquals(verdict.outcome.geometry, B);
    assertStringIncludes(
      rejectionOf(verdict, A),
      "geometric mean ×1.302 (below the adopted ×1.310)",
    );
  });

  it("row counts at segment boundaries ±1 land in the right segment (the first with rows <= maxRows), and conv2d splits by the default row's tileM", () => {
    // The default row is the geometry of the default segment covering M (the sweep is built with
    // the default 3-segment table) — if a segment's default drifts, generation stops
    const defaultFor = (m: number): GemmGeometry => (m <= 64 ? SMALL : m <= 512 ? A : BIG);
    const expected: readonly (readonly [number, string])[] = [
      [16, "gemmRows ≤ 16"],
      [17, "gemmRows 17〜32"],
      [32, "gemmRows 17〜32"],
      [33, "gemmRows 33〜64"],
      [64, "gemmRows 33〜64"],
      [65, "gemmRows 65〜128"],
      [128, "gemmRows 65〜128"],
      [129, "gemmRows 129〜256"],
      [256, "gemmRows 129〜256"],
      [257, "gemmRows 257〜512"],
      [512, "gemmRows 257〜512"],
      [513, "gemmRows > 512"],
    ];
    const conv = (channels: number, mTile: number): CaseRef => ({
      caseId: `conv2d-c${channels}-m${mTile}`,
      op: "conv2d",
      shape: `Cin${channels} Cout${channels} 8x8 3x3`,
    });
    const conv2dRows = (ref: CaseRef, mTile: number): Record<string, unknown>[] => [
      row(ref, conv2dCandidate(gemmMTileGeometry(mTile)), true),
      row(ref, conv2dCandidate(B), false, { speedup: 1.4, identical: true }),
    ];
    const verdicts = deriveProfile(
      [
        source([
          ...expected.flatMap(([m]) =>
            caseRows(
              linearCase(m),
              defaultFor(m),
              m === 33 || m === 64 ? [[B, { speedup: 1.2 }]] : [],
            ),
          ),
          ...conv2dRows(conv(96, 32), 32),
          ...conv2dRows(conv(192, 64), 64),
        ]),
      ],
      OPTIONS,
    );
    for (const slot of ROWS_SLOT_NAMES) {
      const cases = expected.filter(([, name]) => name === slot).map(([m]) => `linear-m${m}`);
      assertEquals(verdictOf(verdicts, slot).cases, cases.sort(), slot);
    }
    assertEquals(verdictOf(verdicts, "gemmRows 33〜64").outcome.geometry, B);
    assertEquals(verdictOf(verdicts, "conv2d.rows32").cases, ["conv2d-c96-m32"]);
    assertEquals(verdictOf(verdicts, "conv2d.rows64").cases, ["conv2d-c192-m64"]);
  });

  it("matmul / bmm rows go into the gemmRows field by the shape's M (for bmm, the M of one matrix regardless of the batch count)", () => {
    const verdicts = deriveProfile(
      [
        source([
          ...caseRows(matmulCase(16), SMALL, []),
          ...caseRows(bmmCase(600, 17), SMALL, []),
          ...caseRows(matmulCase(64), SMALL, []),
          ...caseRows(bmmCase(600, 64), SMALL, []),
          ...caseRows(matmulCase(65), A, []),
          ...caseRows(bmmCase(16, 512), A, []),
          ...caseRows(matmulCase(512), A, []),
          ...caseRows(matmulCase(4096), BIG, []),
        ]),
      ],
      OPTIONS,
    );
    assertEquals(verdictOf(verdicts, "gemmRows ≤ 16").cases, ["matmul-m16"]);
    assertEquals(verdictOf(verdicts, "gemmRows 17〜32").cases, ["bmm-b600-m17"]);
    assertEquals(verdictOf(verdicts, "gemmRows 33〜64").cases, ["bmm-b600-m64", "matmul-m64"]);
    assertEquals(verdictOf(verdicts, "gemmRows 65〜128").cases, ["matmul-m65"]);
    assertEquals(verdictOf(verdicts, "gemmRows 257〜512").cases, ["bmm-b16-m512", "matmul-m512"]);
    assertEquals(verdictOf(verdicts, "gemmRows > 512").cases, ["matmul-m4096"]);
  });

  it("matmul / bmm observations count among the field's cases too (below ×1.05 there, it is not adopted even if fast on linear)", () => {
    const verdicts = deriveProfile(
      [
        source([
          ...caseRows(linearCase(1024), BIG, [
            [A, { speedup: 1.5 }],
            [B, { speedup: 1.2 }],
          ]),
          ...caseRows(matmulCase(4096), BIG, [
            [A, { speedup: 1.02 }],
            [B, { speedup: 1.2 }],
          ]),
          ...caseRows(linearCase(512), A, [[B, { speedup: 1.5 }]]),
          ...caseRows(bmmCase(16, 512), A, [[B, { speedup: 1.04 }]]),
        ]),
      ],
      OPTIONS,
    );
    const large = verdictOf(verdicts, "gemmRows > 512");
    assertEquals(large.cases, ["linear-m1024", "matmul-m4096"]);
    assertEquals(large.outcome.kind, "adopted");
    assertEquals(large.outcome.geometry, B);
    assertStringIncludes(rejectionOf(large, A), "matmul-m4096: ×1.020 < ×1.050");
    const middle = verdictOf(verdicts, "gemmRows 257〜512");
    assertEquals(middle.outcome.kind, "default");
    assertEquals(middle.outcome.geometry, A);
    assertStringIncludes(rejectionOf(middle, B), "bmm-b16-m512: ×1.040 < ×1.050");
  });

  it("a class with no case in the sweep gets the runtime's default and keeps the reason", () => {
    const verdicts = deriveProfile([source(caseRows(linearCase(1024), BIG, []))], OPTIONS);
    const conv = verdictOf(verdicts, "conv2d.rows64");
    assertEquals(conv.outcome.geometry, gemmMTileGeometry(64));
    assert(conv.outcome.kind === "default");
    assertEquals(conv.outcome.reason, "the sweeps have no case of this class");
    assertEquals(
      verdictOf(verdicts, "i8a8.attentionPv").outcome.geometry,
      defaultI8a8Geometry("attention_pv"),
    );
  });

  it("when several sweeps measure the same case, the ratio is their geometric mean, and one mismatch prevents adoption", () => {
    const ref = linearCase(1024);
    const verdict = verdictOf(
      deriveProfile(
        [
          source(
            caseRows(ref, BIG, [
              [A, { speedup: 1.2 }],
              [B, { speedup: 1.5 }],
            ]),
            "quick.json",
          ),
          source(
            caseRows(ref, BIG, [
              [A, { speedup: 1 }],
              [B, { speedup: 1.5, identical: false }],
            ]),
            "full.json",
          ),
        ],
        OPTIONS,
      ),
      "gemmRows > 512",
    );
    assert(verdict.outcome.kind === "adopted");
    assertEquals(verdict.outcome.geometry, A);
    assertEquals(verdict.outcome.geomean.toFixed(4), Math.sqrt(1.2).toFixed(4));
    assertStringIncludes(rejectionOf(verdict, B), "output differs from the default");
  });

  it("i8a8 picks geometries including tileK per op and segment", () => {
    const tile: I8a8Geometry = { regM: 8, regN: 4, wgX: 16, wgY: 8, tileK: 16 };
    const ref: CaseRef = {
      caseId: "i8a8-attention-pv",
      op: "i8a8-attention",
      shape: "pv BH16 M64 N64 D128",
    };
    const verdicts = deriveProfile(
      [
        source([
          row(ref, i8a8Candidate(defaultI8a8Geometry("attention_pv")), true),
          row(ref, i8a8Candidate(tile), false, { speedup: 1.1, identical: true }),
        ]),
      ],
      OPTIONS,
    );
    assertEquals(verdictOf(verdicts, "i8a8.attentionPv").outcome.geometry, tile);
    assertEquals(
      verdictOf(verdicts, "i8a8.attentionQk").outcome.geometry,
      defaultI8a8Geometry("attention_qk"),
    );
  });
});

// The conv3d w8a8 class (ADR 0115 addendum 2026-10-26): one geometry for every conv3d case, chosen
// by the same rule as every other class.
describe("deriveProfile: the conv3d w8a8 class (i8a8.conv3d)", () => {
  const conv3dDefault = i8a8Candidate(defaultI8a8Geometry("conv3d"));
  const conv3dCase = (name: string): CaseRef => ({
    caseId: `i8a8-conv3d-${name}`,
    op: "i8a8-conv3d",
    shape: `Cin64 Cout64 3x3x3 in3x8x8 out1x8x8 ${name}`,
  });
  /** One conv3d case: the default row and one row per candidate (bit-equal unless stated). */
  const conv3dRows = (
    ref: CaseRef,
    candidates: readonly (readonly [I8a8Geometry, { speedup: number; identical?: boolean }])[],
  ): Record<string, unknown>[] => [
    row(ref, conv3dDefault, true),
    ...candidates.map(([geometry, measured]) =>
      row(ref, i8a8Candidate(geometry), false, { identical: true, ...measured }),
    ),
  ];
  const WIDE: I8a8Geometry = { regM: 8, regN: 4, wgX: 16, wgY: 8, tileK: 16 };
  const NARROW: I8a8Geometry = { regM: 4, regN: 4, wgX: 16, wgY: 16, tileK: 16 };
  const FASTEST: I8a8Geometry = { regM: 4, regN: 4, wgX: 8, wgY: 8, tileK: 16 };
  const MISMATCH: I8a8Geometry = { regM: 4, regN: 8, wgX: 8, wgY: 16, tileK: 16 };

  it("adopts only a geometry bit-equal and at least ×1.05 on every conv3d case, the best geometric mean among those", () => {
    const [a, b, c] = ["a", "b", "c"].map(conv3dCase);
    const verdict = verdictOf(
      deriveProfile(
        [
          source([
            ...conv3dRows(a, [
              [WIDE, { speedup: 1.5 }],
              [NARROW, { speedup: 1.3 }],
              [FASTEST, { speedup: 3 }],
              [MISMATCH, { speedup: 4 }],
            ]),
            ...conv3dRows(b, [
              [WIDE, { speedup: 1.4 }],
              [NARROW, { speedup: 1.3 }],
              [FASTEST, { speedup: 2 }],
              [MISMATCH, { speedup: 4, identical: false }],
            ]),
            // The edge guard: FASTEST is slower than ×1.05 on one case
            ...conv3dRows(c, [
              [WIDE, { speedup: 1.1 }],
              [NARROW, { speedup: 1.3 }],
              [FASTEST, { speedup: 1.04 }],
              [MISMATCH, { speedup: 4 }],
            ]),
          ]),
        ],
        OPTIONS,
      ),
      "i8a8.conv3d",
    );
    assertEquals(verdict.cases, [a.caseId, b.caseId, c.caseId]);
    assert(verdict.outcome.kind === "adopted");
    // WIDE: (1.5 · 1.4 · 1.1)^(1/3) = 1.3221 > NARROW 1.3
    assertEquals(verdict.outcome.geometry, WIDE);
    assertEquals(verdict.outcome.geomean.toFixed(4), Math.cbrt(1.5 * 1.4 * 1.1).toFixed(4));
    const reason = (geometry: I8a8Geometry): string =>
      verdict.rejected.find(({ name }) => name === i8a8Candidate(geometry).name)?.reason ?? "";
    assertStringIncludes(reason(FASTEST), `${c.caseId}: ×1.040 < ×1.050`);
    assertStringIncludes(reason(MISMATCH), `output differs from the default in ${b.caseId}`);
    assertStringIncludes(reason(NARROW), "geometric mean ×1.300 (below the adopted ×1.322)");
  });

  it("keeps the default geometry when the sweeps have no conv3d case, even when other i8a8 classes adopt one", () => {
    const linear: CaseRef = { caseId: "i8a8-linear-m64", op: "i8a8-linear", shape: "M64 N64 K64" };
    const verdicts = deriveProfile(
      [
        source([
          row(linear, i8a8Candidate(defaultI8a8Geometry("linear")), true),
          row(linear, i8a8Candidate(WIDE), false, { speedup: 1.5, identical: true }),
        ]),
      ],
      OPTIONS,
    );
    assertEquals(verdictOf(verdicts, "i8a8.linear").outcome.geometry, WIDE);
    const conv3d = verdictOf(verdicts, "i8a8.conv3d");
    assert(conv3d.outcome.kind === "default");
    assertEquals(conv3d.outcome.geometry, defaultI8a8Geometry("conv3d"));
    assertEquals(conv3d.outcome.reason, "the sweeps have no case of this class");
    assertEquals(conv3d.scope, "i8a8 conv3d (w8a8 implicit GEMM)");
  });

  it("conv3d rows decide only i8a8.conv3d (a conv3d winner does not move i8a8.linear), and the table carries it", () => {
    const ref = conv3dCase("a");
    const sources = [source(conv3dRows(ref, [[WIDE, { speedup: 1.2 }]]))];
    const verdicts = deriveProfile(sources, OPTIONS);
    assertEquals(verdictOf(verdicts, "i8a8.conv3d").outcome.geometry, WIDE);
    assertEquals(
      verdictOf(verdicts, "i8a8.linear").outcome.geometry,
      defaultI8a8Geometry("linear"),
    );
    const profile = buildGeometryProfile(
      { id: "conv3d-only", vendor: "apple", architecture: "metal-3" },
      sources,
      verdicts,
    );
    assertEquals(profile.i8a8, { ...DEFAULT_GEOMETRY_PROFILE.i8a8, conv3d: WIDE });
  });

  it("a conv3d default row that is not the runtime's conv3d default stops the generation", () => {
    const ref = conv3dCase("a");
    assertThrows(
      () =>
        deriveProfile(
          [
            source([
              row(ref, i8a8Candidate(WIDE), true),
              row(ref, conv3dDefault, false, { speedup: 1.2, identical: true }),
            ]),
          ],
          OPTIONS,
        ),
      Error,
      "i8a8.conv3d: the sweep's default row",
    );
  });
});

describe("deriveProfile: a case whose default re-measurement ratio is out of range is dropped from that sweep's inputs", () => {
  const ref = linearCase(1024);
  const wide = linearCase(4096);
  const drift = (caseId: string, driftRatio: number): Record<string, unknown> => ({
    caseId,
    defaultRepeat: { perDispatch: 1, driftRatio },
  });
  /** A sweep with the given `cases[]` (cases not given have ratio 1). */
  const sweepWith = (
    rows: readonly Record<string, unknown>[],
    name: string,
    repeats: Repeats,
  ): SweepSource =>
    parseSweepReport(report(rows, ADAPTER, repeats), { path: name, sha256: `sha-${name}` });

  it("does not count observations from an out-of-range sweep in the ratio, and judges by the same case in other sweeps", () => {
    // Without dropping, A's ratio would be √(0.5 × 1.3) ≈ ×0.806 and rejected
    const drifted = sweepWith(caseRows(ref, BIG, [[A, { speedup: 0.5 }]]), "drifted.json", {
      [ref.caseId]: drift(ref.caseId, 1.2),
    });
    const steady = source(caseRows(ref, BIG, [[A, { speedup: 1.3 }]]), "steady.json");
    const verdict = verdictOf(deriveProfile([drifted, steady], OPTIONS), "gemmRows > 512");
    assert(verdict.outcome.kind === "adopted");
    assertEquals(verdict.outcome.geometry, A);
    assertEquals(verdict.outcome.geomean, 1.3);
    assertEquals(verdict.cases, [ref.caseId]);
    assertEquals(verdict.excluded, [
      {
        path: "drifted.json",
        caseId: ref.caseId,
        reason: "default re-measurement ratio ×1.200 out of range",
      },
    ]);
  });

  it("an output mismatch or failure in a dropped sweep rejects the candidate even if it matched and succeeded in other sweeps", () => {
    // Only the ratio is dropped — the output match is the correctness gate and does not depend on
    // thermals
    const drifted = sweepWith(
      caseRows(ref, BIG, [
        [A, { speedup: 1.3, identical: false }],
        [B, { speedup: 1.3, error: "device lost" }],
      ]),
      "drifted.json",
      { [ref.caseId]: drift(ref.caseId, 1.2) },
    );
    const steady = source(
      caseRows(ref, BIG, [
        [A, { speedup: 1.3 }],
        [B, { speedup: 1.3 }],
      ]),
      "steady.json",
    );
    const verdict = verdictOf(deriveProfile([drifted, steady], OPTIONS), "gemmRows > 512");
    assertEquals(verdict.outcome.kind, "default");
    assertEquals(verdict.outcome.geometry, BIG);
    assertEquals(rejectionOf(verdict, A), `output differs from the default in ${ref.caseId}`);
    assertEquals(rejectionOf(verdict, B), `failed in ${ref.caseId} (device lost)`);
  });

  it("a case dropped in every sweep is treated as not measured, and the field stays at the default (the case stays in the case list)", () => {
    const rows = [
      ...caseRows(ref, BIG, [[A, { speedup: 1.3 }]]),
      ...caseRows(wide, BIG, [[A, { speedup: 1.3 }]]),
    ];
    const verdict = verdictOf(
      deriveProfile(
        [
          sweepWith(rows, "first.json", { [wide.caseId]: drift(wide.caseId, 1.15) }),
          sweepWith(rows, "second.json", { [wide.caseId]: drift(wide.caseId, 0.8) }),
        ],
        OPTIONS,
      ),
      "gemmRows > 512",
    );
    assertEquals(verdict.cases, [ref.caseId, wide.caseId]);
    assertEquals(verdict.outcome.kind, "default");
    assertEquals(verdict.outcome.geometry, BIG);
    assertEquals(
      rejectionOf(verdict, A),
      `not measured in ${wide.caseId} (dropped from the ratio material in every sweep)`,
    );
    assertEquals(
      verdict.excluded.map((entry) => [entry.path, entry.caseId]),
      [
        ["first.json", wide.caseId],
        ["second.json", wide.caseId],
      ],
    );
  });

  it("when every case of a class is dropped, the field is the default and keeps the reason", () => {
    const verdict = verdictOf(
      deriveProfile(
        [
          sweepWith(caseRows(ref, BIG, [[A, { speedup: 1.5 }]]), "sweep.json", {
            [ref.caseId]: drift(ref.caseId, 1.5),
          }),
        ],
        OPTIONS,
      ),
      "gemmRows > 512",
    );
    assert(verdict.outcome.kind === "default");
    assertEquals(verdict.outcome.geometry, BIG);
    assertEquals(
      verdict.outcome.reason,
      "every case of the class was dropped from the ratio material in every sweep",
    );
  });

  it("for a segment with one case dropped in every sweep, the segment stays at its default even with a faster geometry", () => {
    const only = linearCase(128);
    const rows = caseRows(only, A, [[B, { speedup: 1.5 }]]);
    const verdict = verdictOf(
      deriveProfile(
        [
          sweepWith(rows, "first.json", { [only.caseId]: drift(only.caseId, 1.15) }),
          sweepWith(rows, "second.json", { [only.caseId]: drift(only.caseId, 0.8) }),
        ],
        OPTIONS,
      ),
      "gemmRows 65〜128",
    );
    assertEquals(verdict.cases, [only.caseId]);
    assert(verdict.outcome.kind === "default");
    assertEquals(verdict.outcome.geometry, A);
    assertEquals(
      verdict.outcome.reason,
      "every case of the class was dropped from the ratio material in every sweep",
    );
    assertEquals(
      rejectionOf(verdict, B),
      `not measured in ${only.caseId} (dropped from the ratio material in every sweep)`,
    );
  });

  it("keeps both ends of the range (×0.9 / ×1.1), and drops a failed re-measurement, a missing field and a missing case record", () => {
    const cases = [900, 1100, 2048, 3072, 4096].map(linearCase);
    const built = report(
      cases.flatMap((entry) => caseRows(entry, BIG, [[A, { speedup: 1.2 }]])),
      ADAPTER,
      {
        [cases[0].caseId]: drift(cases[0].caseId, 0.9),
        [cases[1].caseId]: drift(cases[1].caseId, 1.1),
        [cases[2].caseId]: { caseId: cases[2].caseId, defaultRepeatError: "device lost" },
        [cases[3].caseId]: { caseId: cases[3].caseId },
      },
    );
    // cases[4] lacks its whole case record (the element of cases[]) — the shape of a record whose
    // sweep stopped midway
    const truncated = {
      ...built,
      cases: (built.cases as readonly Record<string, unknown>[]).filter(
        (entry) => entry.caseId !== cases[4].caseId,
      ),
    };
    const verdict = verdictOf(
      deriveProfile([parseSweepReport(truncated, { path: "sweep.json", sha256: "0" })], OPTIONS),
      "gemmRows > 512",
    );
    assertEquals(
      verdict.excluded.map(({ caseId, reason }) => [caseId, reason]),
      [
        [cases[2].caseId, "failed default re-measurement (device lost)"],
        [cases[3].caseId, "no default re-measurement"],
        [cases[4].caseId, "no default re-measurement"],
      ],
    );
  });
});

describe("deriveProfile: quantized timestamps are judged by the per-observation rounding-error bound", () => {
  const ref = linearCase(1024);
  /** 100 µs (Chrome's resolution) — every round is written as a multiple of it. */
  const QUANTUM = 100_000;
  /** A quantized sweep (`gpuTiming.quantized` true, a default re-measurement ratio of 1). */
  const quantizedSource = (rows: readonly Record<string, unknown>[], name: string): SweepSource =>
    parseSweepReport(
      {
        ...report(rows),
        gpuTiming: { feature: true, unit: "ns", quantized: true },
      },
      { path: name, sha256: `sha-${name}` },
    );
  /** Rounds whose minimum round is `ms` milliseconds (e = 100 µs ÷ min). */
  const roundsOf = (ms: number): readonly number[] => [
    ms * 10 * QUANTUM,
    ms * 10 * QUANTUM + QUANTUM,
  ];
  const lines = (verdict: SlotVerdict): string[] =>
    verdictLines([verdict]).filter((line) => line.startsWith("  - Sweep "));

  it("with only observations of E ≤ 1%, a quantized sweep yields verdicts and drops nothing", () => {
    // e(default) = 0.1 / 100 = 0.1%, e(A) = 0.1 / 60 ≈ 0.17% → E ≈ 0.27%
    const rows = caseRows(ref, BIG, [[A, { speedup: 1.3, rounds: roundsOf(60), reps: 3 }]], {
      rounds: roundsOf(100),
      reps: 5,
    });
    const verdict = verdictOf(
      deriveProfile([quantizedSource(rows, "q.json")], OPTIONS),
      "gemmRows > 512",
    );
    assert(verdict.outcome.kind === "adopted");
    assertEquals(verdict.outcome.geometry, A);
    assertEquals(verdict.outcome.geomean, 1.3);
    assertEquals(verdict.excluded, []);
  });

  it("drops only observations with E > 1% and keeps the case's other geometries (E also adds the default row's e)", () => {
    // e(default) = 0.1 / 20 = 0.5%. A: e = 0.1 / 12.5 = 0.8%, so E = 1.3% → dropped (e alone would
    // be at most 1%). B: e = 0.1 / 50 = 0.2%, so E = 0.7% → kept. Without dropping, A (×1.5) would
    // beat B (×1.2)
    const rows = caseRows(
      ref,
      BIG,
      [
        [A, { speedup: 1.5, rounds: roundsOf(12.5) }],
        [B, { speedup: 1.2, rounds: roundsOf(50) }],
      ],
      { rounds: roundsOf(20) },
    );
    const verdict = verdictOf(
      deriveProfile([quantizedSource(rows, "q.json")], OPTIONS),
      "gemmRows > 512",
    );
    assert(verdict.outcome.kind === "adopted");
    assertEquals(verdict.outcome.geometry, B);
    assertEquals(rejectionOf(verdict, A), `not measured in ${ref.caseId}`);
    const name = gemmCandidate(A).name;
    assertEquals(verdict.excluded, [
      {
        path: "q.json",
        caseId: ref.caseId,
        geometry: name,
        reason: "rounding-error bound E 1.30% (> 1%)",
      },
    ]);
    assertEquals(lines(verdict), [
      `  - Sweep q.json: ${ref.caseId} geometry ${name} dropped from the ratio material due to rounding-error bound E 1.30% (> 1%) (output matches and failures are still checked)`,
    ]);
  });

  it("keeps an observation with E of exactly 1% (the gate drops at E > 1%)", () => {
    // e(default) = 0.1 / 20 = 0.5%, A: e = 0.1 / 20 = 0.5%, so E = 1.00% → kept
    const rows = caseRows(ref, BIG, [[A, { speedup: 1.5, rounds: roundsOf(20) }]], {
      rounds: roundsOf(20),
    });
    const verdict = verdictOf(
      deriveProfile([quantizedSource(rows, "q.json")], OPTIONS),
      "gemmRows > 512",
    );
    assert(verdict.outcome.kind === "adopted");
    assertEquals(verdict.outcome.geometry, A);
    assertEquals(verdict.excluded, []);
  });

  it("an output mismatch of a dropped observation still affects the verdict (the correctness gate does not depend on rounding)", () => {
    const rows = caseRows(
      ref,
      BIG,
      [[A, { speedup: 1.5, identical: false, rounds: roundsOf(2) }]],
      { rounds: roundsOf(100) },
    );
    const steady = source(caseRows(ref, BIG, [[A, { speedup: 1.3 }]]), "steady.json");
    const verdict = verdictOf(
      deriveProfile([quantizedSource(rows, "q.json"), steady], OPTIONS),
      "gemmRows > 512",
    );
    assertEquals(verdict.outcome.kind, "default");
    assertEquals(rejectionOf(verdict, A), `output differs from the default in ${ref.caseId}`);
    assertEquals(
      verdict.excluded.map((entry) => entry.geometry),
      [gemmCandidate(A).name],
    );
  });

  it("a case whose default row has e over 1% drops every observation and appears as a single line in the verdict rows", () => {
    // e(default) = 0.1 / 5 = 2% → every geometry has E > 1% (both A and B have a small e)
    const rows = caseRows(
      ref,
      BIG,
      [
        [A, { speedup: 1.5, rounds: roundsOf(80) }],
        [B, { speedup: 1.2, rounds: roundsOf(80) }],
      ],
      { rounds: roundsOf(5) },
    );
    const verdict = verdictOf(
      deriveProfile([quantizedSource(rows, "q.json")], OPTIONS),
      "gemmRows > 512",
    );
    assert(verdict.outcome.kind === "default");
    assertEquals(verdict.outcome.geometry, BIG);
    assertEquals(
      verdict.outcome.reason,
      "every case of the class was dropped from the ratio material in every sweep",
    );
    assertEquals(verdict.excluded, [
      {
        path: "q.json",
        caseId: ref.caseId,
        reason: "rounding-error bound e 2.00% of the default row (> 1%)",
      },
    ]);
    assertEquals(lines(verdict), [
      `  - Sweep q.json: ${ref.caseId} dropped from the ratio material due to rounding-error bound e 2.00% of the default row (> 1%) (output matches and failures are still checked)`,
    ]);
  });

  it("when another sweep measured the same (case, geometry), it judges by that sweep's ratio (quick layered with full)", () => {
    // A of quick (quantized) is dropped at E > 1% — counted, it would be √(0.5 × 1.3) ≈ ×0.806 and
    // rejected
    const quick = quantizedSource(
      caseRows(ref, BIG, [[A, { speedup: 0.5, rounds: roundsOf(2) }]], { rounds: roundsOf(100) }),
      "quick.json",
    );
    const full = source(caseRows(ref, BIG, [[A, { speedup: 1.3 }]]), "full.json");
    const verdict = verdictOf(deriveProfile([quick, full], OPTIONS), "gemmRows > 512");
    assert(verdict.outcome.kind === "adopted");
    assertEquals(verdict.outcome.geometry, A);
    assertEquals(verdict.outcome.geomean, 1.3);
    assertEquals(
      verdict.excluded.map(({ path, geometry }) => [path, geometry]),
      [["quick.json", gemmCandidate(A).name]],
    );
  });

  it("drops rows without rounds or without a positive round, since no bound can be computed (the whole case for the default row)", () => {
    const wide = linearCase(4096);
    const rows = [
      ...caseRows(
        ref,
        BIG,
        [
          [A, { speedup: 1.3 }],
          [B, { speedup: 1.3, rounds: [0, 0] }],
        ],
        { rounds: roundsOf(100) },
      ),
      ...caseRows(wide, BIG, [[A, { speedup: 1.3, rounds: roundsOf(80) }]]),
    ];
    const verdict = verdictOf(
      deriveProfile([quantizedSource(rows, "q.json")], OPTIONS),
      "gemmRows > 512",
    );
    assertEquals(
      verdict.excluded.map(({ caseId, geometry, reason }) => [caseId, geometry, reason]),
      [
        [ref.caseId, gemmCandidate(A).name, "unknown rounding-error bound (no rounds)"],
        [ref.caseId, gemmCandidate(B).name, "unknown rounding-error bound (no positive round)"],
        [wide.caseId, undefined, "unknown rounding-error bound of the default row (no rounds)"],
      ],
    );
  });

  it("a failed row has no ratio, so its bound is not checked (it is not dropped) and it is rejected as a failure", () => {
    const rows = caseRows(ref, BIG, [[A, { error: "device lost" }]], { rounds: roundsOf(100) });
    const verdict = verdictOf(
      deriveProfile([quantizedSource(rows, "q.json")], OPTIONS),
      "gemmRows > 512",
    );
    assertEquals(verdict.excluded, []);
    assertEquals(rejectionOf(verdict, A), `failed in ${ref.caseId} (device lost)`);
  });

  it("a sweep that is not quantized does not look at rounds (neither a record without rounds nor a short pass is dropped)", () => {
    const rows = caseRows(ref, BIG, [
      [A, { speedup: 1.3, rounds: [1_000] }],
      [B, { speedup: 1.2 }],
    ]);
    const verdict = verdictOf(
      deriveProfile([source(rows, "plain.json")], OPTIONS),
      "gemmRows > 512",
    );
    assert(verdict.outcome.kind === "adopted");
    assertEquals(verdict.outcome.geometry, A);
    assertEquals(verdict.excluded, []);
  });
});

describe("parseSweepReport: the default's re-measurement (cases[])", () => {
  const rows = caseRows(linearCase(1024), BIG, []);
  const meta = { path: "sweep.json", sha256: "0" };

  it("fails a record without cases (it does not silently drop every case)", () => {
    const { cases: _, ...withoutCases } = report(rows);
    assertThrows(() => parseSweepReport(withoutCases, meta), Error, "cases is not an array");
  });

  it("fails when the same case has 2 records", () => {
    const entry = { caseId: "linear-m1024", defaultRepeat: { perDispatch: 1, driftRatio: 1 } };
    assertThrows(
      () => parseSweepReport({ ...report(rows), cases: [entry, entry] }, meta),
      Error,
      "cases has linear-m1024 twice",
    );
  });
});

describe("deriveProfile: description narrows the adapter down to the model", () => {
  const rows = caseRows(linearCase(1024), BIG, [[A, { speedup: 1.2 }]]);

  it("with a description given, a sweep whose adapter description differs fails", () => {
    assertThrows(
      () =>
        deriveProfile([source(rows, "m5.json", { ...ADAPTER, description: "Apple M5" })], {
          ...OPTIONS,
          description: "Test GPU",
        }),
      Error,
      "does not match the requested",
    );
    deriveProfile([source(rows)], { ...OPTIONS, description: "Test GPU" });
  });

  it("description only together with architecture, and the empty string fails (the same conditions as the runtime's gate)", () => {
    assertThrows(
      () =>
        deriveProfile([source(rows)], {
          vendor: "apple",
          description: "Test GPU",
          minSpeedup: 1.05,
        }),
      Error,
      "together with both vendor and architecture",
    );
    assertThrows(
      () => deriveProfile([source(rows)], { ...OPTIONS, description: "" }),
      Error,
      "empty string",
    );
  });

  it("sweeps must agree down to the description (no mixing of models even with --opt-in or without --description)", () => {
    const m2 = source(rows, "m2.json", { ...ADAPTER, description: "Apple M2" });
    const m5 = source(rows, "m5.json", { ...ADAPTER, description: "Apple M5" });
    const blank = source(rows, "blank.json", { ...ADAPTER, description: "" });
    for (const target of [{ optIn: true } as const, OPTIONS]) {
      assertThrows(
        () => deriveProfile([m2, m5], { ...target, minSpeedup: 1.05 }),
        Error,
        "differs from",
      );
      assertThrows(
        () => deriveProfile([m2, blank], { ...target, minSpeedup: 1.05 }),
        Error,
        "differs from",
      );
    }
    deriveProfile([blank, source(rows, "blank2.json", { ...ADAPTER, description: "" })], OPTIONS);
  });

  it("--opt-in does not match vendor / architecture (the sweeps' adapters must still agree)", () => {
    deriveProfile([source(rows)], { optIn: true, minSpeedup: 1.05 });
    assertThrows(
      () =>
        deriveProfile(
          [
            source(rows, "m2.json"),
            source(rows, "other.json", { ...ADAPTER, architecture: "metal-2" }),
          ],
          { optIn: true, minSpeedup: 1.05 },
        ),
      Error,
      "differs from",
    );
  });
});

describe("deriveProfile: consistency with the runtime's default profile", () => {
  it("the default written for a class without cases: a gemmRows bucket gets the geometry of the default bucket covering its range, every other field DEFAULT_GEOMETRY_PROFILE's (15 fields, in this order)", () => {
    const profile = DEFAULT_GEOMETRY_PROFILE;
    // The default table stays at 3 segments (≤ 64 / 65–512 / > 512 — ADR 0116 decision 3)
    assertEquals(
      profile.gemmRows.map((rule) => rule.maxRows),
      [64, 512, Infinity],
    );
    const [small, middle, large] = profile.gemmRows.map((rule) => rule.geometry);
    const expected: Readonly<Record<string, GemmGeometry | I8a8Geometry>> = {
      "gemmRows ≤ 16": small,
      "gemmRows 17〜32": small,
      "gemmRows 33〜64": small,
      "gemmRows 65〜128": middle,
      "gemmRows 129〜256": middle,
      "gemmRows 257〜512": middle,
      "gemmRows > 512": large,
      "attention.qk": profile.attention.qk,
      "attention.pv": profile.attention.pv,
      "conv2d.rows64": profile.conv2d.rows64,
      "conv2d.rows32": profile.conv2d.rows32,
      "i8a8.linear": profile.i8a8.linear,
      "i8a8.attentionQk": profile.i8a8.attentionQk,
      "i8a8.attentionPv": profile.i8a8.attentionPv,
      "i8a8.conv3d": profile.i8a8.conv3d,
    };
    const verdicts = deriveProfile([source([])], OPTIONS);
    assertEquals(
      verdicts.map((verdict) => verdict.slot),
      Object.keys(expected),
    );
    for (const verdict of verdicts) {
      assertEquals(verdict.outcome.geometry, expected[verdict.slot], verdict.slot);
    }
  });
});

describe("profileRowsSegments: validation of the gemmRows segment boundaries", () => {
  it("the generator's boundaries (PROFILE_GEMM_ROWS_BOUNDS) are 7 segments, spelling the field name and its range as a row-count range (not by index)", () => {
    const segments = profileRowsSegments(PROFILE_GEMM_ROWS_BOUNDS);
    assertEquals(
      segments.map((segment) => segment.maxRows),
      [16, 32, 64, 128, 256, 512, Infinity],
    );
    assertEquals(
      segments.map((segment) => segment.slot),
      [...ROWS_SLOT_NAMES],
    );
    assertEquals(segments[1].scope, "linear / matmul / bmm row count 17〜32");
    assertEquals(segments[6].scope, "linear / matmul / bmm row count > 512");
  });

  it("a segment's default is the geometry of the default segment covering its range (the 3 segments ≤ 64 → ≤ 64, the 3 segments in 65–512 → 65–512, > 512 → > 512)", () => {
    const [small, middle, large] = GEMM_ROWS_BUCKETS.map((rule) => rule.geometry);
    assertEquals(
      profileRowsSegments(PROFILE_GEMM_ROWS_BOUNDS).map((segment) => segment.fallback),
      [small, small, small, middle, middle, middle, large],
    );
  });

  it("the spelling and the segment defaults are derived from the given boundaries (the default boundaries themselves give the same 3 segments as the default table)", () => {
    assertEquals(
      profileRowsSegments([64, 512, Infinity]).map((segment) => [
        segment.maxRows,
        segment.fallback,
      ]),
      GEMM_ROWS_BUCKETS.map((rule) => [rule.maxRows, rule.geometry]),
    );
    assertEquals(
      profileRowsSegments([64, 100, 512, Infinity]).map((segment) => segment.slot),
      ["gemmRows ≤ 64", "gemmRows 65〜100", "gemmRows 101〜512", "gemmRows > 512"],
    );
  });

  it("a boundary set without the default boundaries (64 / 512) stops generation (a segment would straddle a default boundary)", () => {
    assertThrows(
      () => profileRowsSegments([32, 128, Infinity]),
      Error,
      "missing the default table's bounds 64 / 512",
    );
    assertThrows(
      () => profileRowsSegments([16, 32, 64, 128, 256, Infinity]),
      Error,
      "missing the default table's bounds 512",
    );
  });

  it("a boundary set that is not strictly ascending, does not end in Infinity or has non-positive-integer boundaries stops generation", () => {
    for (const bounds of [
      [64, 32, 512, Infinity],
      [64, 64, 512, Infinity],
    ]) {
      assertThrows(() => profileRowsSegments(bounds), Error, "not strictly ascending");
    }
    for (const bounds of [[64, 512], []]) {
      assertThrows(() => profileRowsSegments(bounds), Error, "the last bound is not Infinity");
    }
    for (const bounds of [
      [0, 64, 512, Infinity],
      [16.5, 64, 512, Infinity],
    ]) {
      assertThrows(() => profileRowsSegments(bounds), Error, "must be positive integers");
    }
  });
});

describe("deriveProfile: a segment with one case", () => {
  it("adopts a geometry at ×1.05 or above on the segment's one case, and keeps the segment default below it", () => {
    const verdicts = deriveProfile(
      [
        source([
          ...caseRows(linearCase(16), SMALL, [
            [A, { speedup: 1.05 }],
            [B, { speedup: 1.04 }],
          ]),
          ...caseRows(linearCase(32), SMALL, [[A, { speedup: 1.049 }]]),
        ]),
      ],
      OPTIONS,
    );
    const adopted = verdictOf(verdicts, "gemmRows ≤ 16");
    assertEquals(adopted.cases, ["linear-m16"]);
    assert(adopted.outcome.kind === "adopted");
    assertEquals(adopted.outcome.geometry, A);
    assertEquals(adopted.outcome.geomean, 1.05);
    assertStringIncludes(rejectionOf(adopted, B), "linear-m16: ×1.040 < ×1.050");
    const kept = verdictOf(verdicts, "gemmRows 17〜32");
    assertEquals(kept.cases, ["linear-m32"]);
    assertEquals(kept.outcome.kind, "default");
    assertEquals(kept.outcome.geometry, SMALL);
    assertStringIncludes(rejectionOf(kept, A), "linear-m32: ×1.049 < ×1.050");
  });
});

describe("deriveProfile: the reasons that compare two ratios print them so they read differently", () => {
  const verdictWithSpeedup = (speedup: number, minSpeedup: number = OPTIONS.minSpeedup) =>
    verdictOf(
      deriveProfile([source(caseRows(linearCase(32), SMALL, [[A, { speedup }]]))], {
        ...OPTIONS,
        minSpeedup,
      }),
      "gemmRows 17〜32",
    );

  it("prints a ratio that rounds to the minSpeedup bar at 3 decimals, and the bar, with as many decimals as tell them apart", () => {
    const verdict = verdictWithSpeedup(1.0498);
    assertEquals(verdict.outcome.kind, "default");
    assertStringIncludes(rejectionOf(verdict, A), "linear-m32: ×1.0498 < ×1.0500");
  });

  it("does the same for a bar with more than 3 decimals", () => {
    const verdict = verdictWithSpeedup(1.0501, 1.0504);
    assertEquals(verdict.outcome.kind, "default");
    assertStringIncludes(rejectionOf(verdict, A), "linear-m32: ×1.0501 < ×1.0504");
  });

  it("keeps the 3-decimal spelling for a ratio that already reads below the bar", () => {
    const verdict = verdictWithSpeedup(1.04);
    assertEquals(verdict.outcome.kind, "default");
    assertStringIncludes(rejectionOf(verdict, A), "linear-m32: ×1.040 < ×1.050");
  });

  it("prints a losing geometric mean and the adopted one with as many decimals as tell them apart", () => {
    const verdict = verdictOf(
      deriveProfile(
        [
          source(
            caseRows(linearCase(32), SMALL, [
              [A, { speedup: 1.2341 }],
              [B, { speedup: 1.2343 }],
            ]),
          ),
        ],
        OPTIONS,
      ),
      "gemmRows 17〜32",
    );
    assertStringIncludes(
      rejectionOf(verdict, A),
      "geometric mean ×1.2341 (below the adopted ×1.2343)",
    );
  });
});

describe("deriveProfile: the sweep's default row must be the current runtime's default", () => {
  // Fixtures with the ratios' basis (the default row) shifted by one — generation stops for every
  // field
  const shifted: readonly {
    readonly slot: SlotVerdict["slot"];
    readonly rows: readonly Record<string, unknown>[];
  }[] = [
    { slot: "gemmRows ≤ 16", rows: caseRows(linearCase(16), A, [[B, { speedup: 1.2 }]]) },
    { slot: "gemmRows 257〜512", rows: caseRows(linearCase(512), BIG, [[A, { speedup: 1.2 }]]) },
    {
      slot: "attention.qk",
      rows: [
        row(
          { caseId: "attention-qk", op: "attention", shape: "qk BH16 M64 N64 D128" },
          gemmCandidate(A),
          true,
        ),
      ],
    },
    {
      slot: "conv2d.rows64",
      rows: [
        row(
          { caseId: "conv2d-c192", op: "conv2d", shape: "Cin192 Cout192 8x8 3x3" },
          // tileM stays 64 (= the same class); only the geometry differs from the runtime's default
          conv2dCandidate({ regM: 4, regN: 4, wgX: 16, wgY: 16 }),
          true,
        ),
      ],
    },
    {
      slot: "i8a8.linear",
      rows: [
        row(
          { caseId: "i8a8-linear", op: "i8a8-linear", shape: "M64 N64 K64" },
          i8a8Candidate(defaultI8a8Geometry("attention_pv")),
          true,
        ),
      ],
    },
  ];
  for (const { slot, rows } of shifted) {
    it(`fails when the geometry of ${slot}'s default row differs from the runtime's default`, () => {
      const error = assertThrows(() => deriveProfile([source(rows)], OPTIONS), Error);
      assertStringIncludes(error.message, `${slot}: the sweep's default row`);
      assertStringIncludes(error.message, "the current runtime's default");
    });
  }
});

describe("deriveProfile: adapters are not mixed", () => {
  const rows = caseRows(linearCase(1024), BIG, [[A, { speedup: 1.2 }]]);

  it("fails when the sweeps' adapters (vendor / architecture) differ", () => {
    assertThrows(
      () =>
        deriveProfile(
          [
            source(rows, "m2.json"),
            source(rows, "other.json", { ...ADAPTER, architecture: "metal-2" }),
          ],
          { vendor: "apple", minSpeedup: 1.05 },
        ),
      Error,
      "differs from",
    );
  });

  it("fails when --vendor / --architecture differ from the sweep's adapter", () => {
    assertThrows(
      () => deriveProfile([source(rows)], { ...OPTIONS, vendor: "intel" }),
      Error,
      "does not match the requested",
    );
    assertThrows(
      () => deriveProfile([source(rows)], { ...OPTIONS, architecture: "metal-2" }),
      Error,
      "does not match the requested",
    );
  });

  it("fails when the same sweep (the same sha256) is passed twice", () => {
    assertThrows(() => deriveProfile([source(rows), source(rows)], OPTIONS), Error, "passed twice");
  });

  it("fails when the sweeps' adapter device differs, even if the other 3 fields are the same (a table holds only one set of the 4 fields)", () => {
    const m2 = source(rows, "m2.json", { ...ADAPTER, device: "0x0000" });
    const other = source(rows, "other.json", { ...ADAPTER, device: "0x1234" });
    const blank = source(rows, "blank.json", { ...ADAPTER, device: "" });
    for (const target of [{ optIn: true } as const, OPTIONS]) {
      for (const mixed of [other, blank]) {
        const error = assertThrows(
          () => deriveProfile([m2, mixed], { ...target, minSpeedup: 1.05 }),
          Error,
          "differs from",
        );
        assertStringIncludes(
          error.message,
          `adapter device ${JSON.stringify(mixed.adapter.device)} differs from m2.json's "0x0000"`,
        );
      }
    }
    deriveProfile([m2, source(rows, "m2-again.json", { ...ADAPTER, device: "0x0000" })], OPTIONS);
  });
});

describe("parseSweepReport", () => {
  const meta = { path: "sweep.json", sha256: "0" };

  it("the adapter's device is a required string (the empty string is accepted as a value)", () => {
    const rows = caseRows(linearCase(1024), BIG, []);
    const { device: _, ...withoutDevice } = ADAPTER;
    assertThrows(
      () => parseSweepReport(report(rows, withoutDevice), meta),
      Error,
      "adapter: device is not a string",
    );
    assertEquals(
      parseSweepReport(report(rows, { ...ADAPTER, device: "" }), meta).adapter.device,
      "",
    );
  });

  it("reads the candidate set from settings.candidateSet, from settings.quick for older records without the field, and fails if neither exists", () => {
    const base = report(caseRows(linearCase(1024), BIG, []));
    const read = (settings: unknown) => parseSweepReport({ ...base, settings }, meta).candidateSet;
    assertEquals(read({ candidateSet: "quick+", quick: false }), "quick+");
    assertEquals(read({ quick: true }), "quick");
    assertEquals(read({ quick: false }), "full");
    assertThrows(() => read({ candidateSet: "quik" }), Error, "settings.candidateSet is none of");
    assertThrows(() => read({}), Error, "neither candidateSet nor quick");
    assertThrows(() => read(undefined), Error, "settings is missing");
  });

  it("fails when a case does not have exactly one default row (the ratios' basis is undetermined)", () => {
    const rows = caseRows(linearCase(1024), BIG, [[A, { speedup: 1.2 }]]);
    assertThrows(
      () => parseSweepReport(report([...rows, rows[0]]), meta),
      Error,
      "has 2 default rows",
    );
  });

  it("accepts a sweep with quantized timestamps too, using the 100 µs resolution for the per-observation bound", () => {
    const parsed = parseSweepReport(
      {
        ...report(caseRows(linearCase(1024), BIG, [])),
        gpuTiming: { feature: true, unit: "ns", quantized: true },
      },
      meta,
    );
    assertEquals(parsed.timestampQuantum, 100_000);
    assertEquals(
      parseSweepReport(report(caseRows(linearCase(1024), BIG, [])), meta).timestampQuantum,
      0,
    );
  });

  it("fails a sweep whose gpuTiming.quantized is not a boolean, or is true with a unit other than ns", () => {
    const rows = caseRows(linearCase(1024), BIG, []);
    for (const quantized of [undefined, "true", 1]) {
      assertThrows(
        () =>
          parseSweepReport(
            {
              ...report(rows),
              gpuTiming: { feature: true, unit: "ns", quantized },
            },
            meta,
          ),
        Error,
        "gpuTiming.quantized is not a boolean",
      );
    }
    assertThrows(
      () =>
        parseSweepReport(
          {
            ...report(rows),
            gpuTiming: { feature: true, unit: "deno-raw-tick", quantized: true },
          },
          meta,
        ),
      Error,
      "the unit is deno-raw-tick",
    );
  });

  it("fails rows with negative or non-numeric values in rounds and rows whose reps is not a positive integer", () => {
    const ref = linearCase(1024);
    for (const measured of [
      { rounds: [1, -1] },
      { rounds: [Number.NaN] },
      { reps: 1.5 },
      { reps: 0 },
    ]) {
      assertThrows(
        () =>
          parseSweepReport(report(caseRows(ref, BIG, [[A, { speedup: 1.2, ...measured }]])), meta),
        Error,
        "has the wrong type",
      );
    }
  });

  it("does not use a wall-clock sweep (unit wall) as input", () => {
    assertThrows(
      () =>
        parseSweepReport(
          {
            ...report(caseRows(linearCase(1024), BIG, [])),
            gpuTiming: { feature: false, unit: "wall", quantized: false },
          },
          meta,
        ),
      Error,
      'gpuTiming.unit is "wall"',
    );
  });

  it("does not use a sweep without gpuTiming as input (the unit is unknown)", () => {
    const { gpuTiming: _, ...withoutTiming } = report(caseRows(linearCase(1024), BIG, []));
    assertThrows(() => parseSweepReport(withoutTiming, meta), Error, "gpuTiming is missing");
  });

  it("accepts sweeps measured with GPU timestamps (ns / deno-raw-tick)", () => {
    for (const unit of ["ns", "deno-raw-tick"]) {
      const parsed = parseSweepReport(
        {
          ...report(caseRows(linearCase(1024), BIG, [])),
          gpuTiming: { feature: true, unit, quantized: false },
        },
        meta,
      );
      assertEquals(parsed.rows.length, 1);
    }
  });
});

/**
 * Segment upper end → range spelling (the expected values of the 7 segments written by hand — not
 * the source of truth of the generator's spelling).
 */
const RANGES: Readonly<Record<number, string>> = {
  16: "≤ 16",
  32: "17〜32",
  64: "33〜64",
  128: "65〜128",
  256: "129〜256",
  512: "257〜512",
};
const rangeOf = (maxRows: number): string => RANGES[maxRows] ?? "> 512";

describe("buildGeometryProfile", () => {
  it("each field's value is the verdict's geometry itself and passes the runtime's gate (the gate through which acquireGpu accepts an injection)", () => {
    const verdicts = deriveProfile(SOURCES, SPEC);
    const profile = buildGeometryProfile(SPEC, SOURCES, verdicts);
    assertGeometryProfile(profile);
    assertEquals(verdictOf(verdicts, "gemmRows > 512").outcome.kind, "adopted");
    assertEquals(profile.gemmRows.at(-1)?.geometry, FAST);
    assertEquals(
      profile.gemmRows.map((rule) => rule.maxRows),
      [16, 32, 64, 128, 256, 512, Infinity],
    );
    for (const rule of profile.gemmRows) {
      const slot = verdictOf(verdicts, `gemmRows ${rangeOf(rule.maxRows)}`);
      assertEquals(rule.geometry, slot.outcome.geometry, slot.slot);
    }
    const slots = [
      ["attention.qk", profile.attention.qk],
      ["attention.pv", profile.attention.pv],
      ["conv2d.rows64", profile.conv2d.rows64],
      ["conv2d.rows32", profile.conv2d.rows32],
      ["i8a8.linear", profile.i8a8.linear],
      ["i8a8.attentionQk", profile.i8a8.attentionQk],
      ["i8a8.attentionPv", profile.i8a8.attentionPv],
      ["i8a8.conv3d", profile.i8a8.conv3d],
    ] as const;
    for (const [slot, geometry] of slots) {
      assertEquals(geometry, verdictOf(verdicts, slot).outcome.geometry, slot);
    }
  });

  it("a field with no case in the sweep gets the same value as the default table (DEFAULT_GEOMETRY_PROFILE) (a gemmRows segment gets the default segment covering its range)", () => {
    const profile = buildGeometryProfile(SPEC, SOURCES, deriveProfile(SOURCES, SPEC));
    // The geometry looked up in the runtime's default table at the segment's upper-end M = the
    // geometry of the default segment covering the segment's range (the segments refine the
    // defaults)
    for (const rule of profile.gemmRows.slice(0, -1)) {
      assertEquals(rule.geometry, gemmGeometryForRows(rule.maxRows), String(rule.maxRows));
    }
    assertEquals(profile.attention, DEFAULT_GEOMETRY_PROFILE.attention);
    assertEquals(profile.conv2d, DEFAULT_GEOMETRY_PROFILE.conv2d);
    assertEquals(profile.i8a8, DEFAULT_GEOMETRY_PROFILE.i8a8);
  });

  it("match and provenance follow the spec and the order of the sweeps; the adapter keeps its 4 fields as is; the fingerprint and the case-set version are the current runtime's values (without architecture, match has vendor only)", () => {
    const verdicts = deriveProfile(SOURCES, SPEC);
    const profile = buildGeometryProfile(SPEC, SOURCES, verdicts);
    assertEquals(profile.id, "test-gpu");
    assertEquals(profile.match, { vendor: "apple", architecture: "metal-3" });
    const { provenance: _provenance, ...table } = profile;
    assertEquals(profile.provenance, {
      sweep: "a.json, b.json",
      sha256: "sha-a, sha-b",
      date: "2026-09-29T00:00:00.000Z, 2026-09-29T00:00:00.000Z",
      candidateSet: "full, full",
      adapter: { vendor: "apple", architecture: "metal-3", device: "", description: "Test GPU" },
      kernels: geometryProfileKernelsId(table),
      caseSet: sweepCaseSetId(),
    });
    // The fingerprint is derived from the table's geometries (it differs from the default table's —
    // the adopted geometries are in it)
    assert(profile.provenance.kernels !== geometryProfileKernelsId(DEFAULT_GEOMETRY_PROFILE));
    const { architecture: _, ...vendorOnly } = SPEC;
    assertEquals(buildGeometryProfile(vendorOnly, SOURCES, verdicts).match, { vendor: "apple" });
  });
});

describe("buildGeometryProfile: a table whose fingerprint cannot be derived", () => {
  it("builds no table whose adopted geometry makes a sweep shape's dispatch exceed 65535, and throws naming the table, case, field and geometry", () => {
    // A geometry that could pass in a sweep on a device with larger limits (tileN 4 — the N =
    // 262144 of conv2d 512² gives 65536 workgroups)
    const narrow = { regM: 1, regN: 4, wgX: 1, wgY: 16 };
    const verdicts = deriveProfile(SOURCES, SPEC).map((verdict) =>
      verdict.slot === "conv2d.rows32"
        ? { ...verdict, outcome: { ...verdict.outcome, geometry: narrow } }
        : verdict,
    );
    const error = assertThrows(() => buildGeometryProfile(SPEC, SOURCES, verdicts), Error);
    assertInstanceOf(error.cause, KernelsIdError);
    for (const part of [
      "Cannot derive provenance.kernels (the kernel fingerprint) of generated table 'test-gpu'",
      "conv2d-c96-512x512",
      "conv2d.rows32",
      JSON.stringify(narrow),
      "65535",
    ]) {
      assertStringIncludes(error.message, part);
    }
  });
});

describe("buildGeometryProfile: description and injection-only", () => {
  const described = { ...SPEC, description: "Test GPU" };
  const optIn = {
    from: SPEC.from,
    id: SPEC.id,
    optIn: true,
    out: SPEC.out,
    minSpeedup: SPEC.minSpeedup,
  } as const;
  const adapter = { vendor: "apple", architecture: "metal-3", description: "Test GPU" };

  it("a table with a description given has the description in match and matches only that model", () => {
    const profile = buildGeometryProfile(described, SOURCES, deriveProfile(SOURCES, described));
    assertEquals(profile.match, {
      vendor: "apple",
      architecture: "metal-3",
      description: "Test GPU",
    });
    assertEquals(selectGeometryProfile(adapter, [profile]), profile);
    assertEquals(
      selectGeometryProfile({ ...adapter, description: "" }, [profile]),
      DEFAULT_GEOMETRY_PROFILE,
    );
  });

  it("an injection-only table has no match field at all, passes the gate, and is not selected automatically even on the same adapter", () => {
    const profile = buildGeometryProfile(optIn, SOURCES, deriveProfile(SOURCES, optIn));
    assert(!("match" in profile), JSON.stringify(profile));
    assertGeometryProfile(profile);
    assertEquals(selectGeometryProfile(adapter, [profile]), DEFAULT_GEOMETRY_PROFILE);
    assert(!("match" in JSON.parse(geometryProfileJson(profile))));
  });
});

const CHROME_153 =
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36";
const CHROME_154 =
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/154.0.0.0 Safari/537.36";

/**
 * A record of one linear M = 1024 case (A is ×1.2 faster than the default with matching output),
 * with the fields of `fields` added.
 */
const record = (fields: Record<string, unknown> = {}): Record<string, unknown> => ({
  ...report(caseRows(linearCase(1024), BIG, [[A, { speedup: 1.2 }]])),
  ...fields,
});

/**
 * Turns a sequence of records into generator inputs (the sha256 differs per record — the generator
 * rejects two with the same value).
 */
const inputs = (...records: readonly Record<string, unknown>[]): GeometrySweepInput[] =>
  records.map((value, index) => ({
    report: value,
    path: `sweep-${index}.json`,
    sha256: `sha-${index}`,
  }));

/** Options of the public deriveGeometryProfile (the helpers' OPTIONS is deriveProfile's spec). */
const RECORD_OPTIONS = { id: "test-gpu", vendor: "apple", architecture: "metal-3" } as const;

const derive = (...records: readonly Record<string, unknown>[]) =>
  deriveGeometryProfile(inputs(...records), RECORD_OPTIONS);

describe("the record's matching keys caseSet / defaultKernels (ADR 0117 decision 8)", () => {
  const current = {
    caseSet: sweepCaseSetId(),
    defaultKernels: geometryProfileKernelsId(DEFAULT_GEOMETRY_PROFILE),
  };

  it("accepts a record with the current runtime's values, giving the same table as a record without the fields (a record from before them)", () => {
    const withKeys = derive(record(current));
    const without = derive(record());
    assertEquals(withKeys.profile, without.profile);
    assertEquals(withKeys.verdicts, without.verdicts);
    assertEquals(withKeys.profile.gemmRows.at(-1)?.geometry, A);
  });

  for (const key of ["caseSet", "defaultKernels"] as const) {
    it(`stops, naming both the record's value and the current value, when ${key} differs from the current runtime's value`, () => {
      const stale = "0123456789abcdef";
      const error = assertThrows(
        () => derive(record({ ...current, [key]: stale })),
        Error,
        `sweep-0.json: ${key} ${stale} differs from the current runtime's ${current[key]}`,
      );
      assertStringIncludes(error.message, "sweep again");
    });

    it(`stops when ${key} is not a string (the matching is not skipped silently)`, () => {
      assertThrows(
        () => derive(record({ ...current, [key]: 1 })),
        Error,
        `sweep-0.json: ${key} is not a string (1)`,
      );
    });
  }

  it("stops, naming the record, even when only the second record differs", () => {
    assertThrows(
      () => derive(record(current), record({ ...current, caseSet: "ffffffffffffffff" })),
      Error,
      "sweep-1.json: caseSet ffffffffffffffff",
    );
  });

  it("aborted, startedAt and cases[].elapsedMs do not affect the generator's verdicts", () => {
    const plain = derive(record(current));
    const base = record(current);
    const cases = base.cases;
    if (!Array.isArray(cases)) throw new Error("the fixture's cases is not an array");
    const annotated = derive({
      ...base,
      startedAt: "2026-09-26T23:00:00.000Z",
      aborted: true,
      cases: cases.map((summary: Record<string, unknown>) => ({ ...summary, elapsedMs: 1234.5 })),
    });
    assertEquals(annotated.profile, plain.profile);
    assertEquals(annotated.verdicts, plain.verdicts);
  });
});

describe("the table's provenance.userAgent (ADR 0117 addendum 2026-10-02)", () => {
  it("copies the record's userAgent string as is into a one-element array", () => {
    assertEquals(derive(record({ userAgent: CHROME_154 })).profile.provenance?.userAgent, [
      CHROME_154,
    ]);
  });

  it("turns a Deno CLI record ({ deno }) into Deno/<version>, the same spelling as Deno's navigator.userAgent", () => {
    assertEquals(derive(record({ userAgent: { deno: "2.9.6" } })).profile.provenance?.userAgent, [
      "Deno/2.9.6",
    ]);
  });

  it("writes a single element when every record has the same value (directly comparable with the current browser)", () => {
    const { profile } = derive(
      record({ userAgent: CHROME_154 }),
      record({ userAgent: CHROME_154 }),
    );
    assertEquals(profile.provenance?.userAgent, [CHROME_154]);
  });

  it("when records differ, lists the values in order of appearance without duplicates (a value containing ', ' stays a separate element)", () => {
    const { profile } = derive(
      record({ userAgent: CHROME_153 }),
      record({ userAgent: CHROME_154 }),
      record({ userAgent: CHROME_153 }),
      record({ userAgent: { deno: "2.9.6" } }),
    );
    assertEquals(profile.provenance?.userAgent, [CHROME_153, CHROME_154, "Deno/2.9.6"]);
  });

  it("writes no field at all when a record without userAgent is mixed in (the browser of some records does not become the table's value)", () => {
    const { profile } = derive(record({ userAgent: CHROME_154 }), record());
    assertEquals(profile.provenance !== undefined && "userAgent" in profile.provenance, false);
  });

  it("stops, naming the record, on a userAgent that is neither a string nor { deno: string }", () => {
    for (const userAgent of [1, { deno: 2 }, { deno: "2.9.6", v8: "14" }, null]) {
      assertThrows(
        () => derive(record({ userAgent })),
        Error,
        "sweep-0.json: userAgent is neither a string nor { deno: string }",
        JSON.stringify(userAgent),
      );
    }
  });
});

describe("the record's matching key candidateKernels (the kernels of the measured candidates — ADR 0117 decision 8)", () => {
  // The fingerprint rebuilds the record's rows with the current runtime's case plan, so the cases
  // are ones in the current shape table (synthetic ids cannot be derived)
  const LINEAR: CaseRef = {
    caseId: "linear-m1024-n2048-k2048",
    op: "linear",
    shape: "M1024 N2048 K2048",
  };
  const MEASURED = [
    { caseId: LINEAR.caseId, geometryParams: BIG },
    { caseId: LINEAR.caseId, geometryParams: A },
  ];
  const measuredRecord = (fields: Record<string, unknown> = {}): Record<string, unknown> => ({
    ...report(caseRows(LINEAR, BIG, [[A, { speedup: 1.2 }]])),
    dp4a: false,
    ...fields,
  });

  it("accepts a record with the current runtime's value, giving the same table as a record without the field", () => {
    const current = sweepCandidateKernelsId(MEASURED, false);
    const withKey = derive(measuredRecord({ candidateKernels: current }));
    const without = derive(measuredRecord());
    assertEquals(withKey.profile, without.profile);
    assertEquals(withKey.profile.gemmRows.at(-1)?.geometry, A);
  });

  it("stops, naming both values, on a record baked by a runtime that differs only in the candidates' WGSL and not the default", () => {
    // The runtime that wrote the record: only the WGSL of dispatches with non-default geometries
    // differs from now (defaultKernels keeps the same value)
    const candidatesOnly: CasePlanner = (sweepCase, limit, dp4a) => {
      const plan = casePlan(sweepCase, limit, dp4a);
      const defaultGeometry = JSON.stringify(plan.defaultCandidate.geometry);
      return {
        ...plan,
        launch: (candidate) => {
          const launch = plan.launch(candidate);
          return JSON.stringify(candidate.geometry) === defaultGeometry
            ? launch
            : { ...launch, wgsl: `${launch.wgsl}\nfn older() {}` };
        },
      };
    };
    const stale = candidateKernelsId(MEASURED, false, SWEEP_CASES, candidatesOnly);
    const current = sweepCandidateKernelsId(MEASURED, false);
    const error = assertThrows(
      () =>
        derive(
          measuredRecord({
            defaultKernels: geometryProfileKernelsId(DEFAULT_GEOMETRY_PROFILE),
            candidateKernels: stale,
          }),
        ),
      Error,
      `sweep-0.json: candidateKernels ${stale} differs from the current runtime's ${current} (the record's kernels of the measured candidates`,
    );
    assertStringIncludes(error.message, "sweep again");
  });

  it("stops on a record whose candidateKernels is not a string or whose dp4a is not a boolean (the matching is not skipped silently)", () => {
    const current = sweepCandidateKernelsId(MEASURED, false);
    assertThrows(
      () => derive(measuredRecord({ candidateKernels: 1 })),
      Error,
      "sweep-0.json: candidateKernels is not a string (1)",
    );
    assertThrows(
      () => derive(measuredRecord({ candidateKernels: current, dp4a: undefined })),
      Error,
      "sweep-0.json: dp4a is not a boolean",
    );
  });
});

describe("deriveGeometryProfile passes the table it returns through the injection gate", () => {
  it("builds no table with an empty id and stops with the injection gate's wording (both a table with match and an injection-only table)", () => {
    for (const options of [{ ...RECORD_OPTIONS, id: "" }, { id: "", optIn: true } as const]) {
      assertThrows(
        () => deriveGeometryProfile(inputs(record()), options),
        Error,
        "Generated table '' fails the injection check — Geometry profile: id is empty",
      );
    }
  });
});
