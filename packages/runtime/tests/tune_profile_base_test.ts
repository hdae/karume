// Carrying op families over from an existing table (`DeriveGeometryProfileOptions.base` of
// deriveGeometryProfile — ADR 0117 addendum 2026-11-12): the records cover some families, the base
// table stands for the others. No GPU: the records are synthetic over the current runtime's real
// sweep cases (`SWEEP_CASES`, the default row from the sweep's own case plan), so their family keys
// and fingerprints are the ones a sweep of the current runtime writes.
//
// What this pins: ① a family the records cover is derived from them, a family they do not cover is
// copied from the base — its fields and its provenance entry verbatim — and a family without a base
// entry stays at the default with no entry; ② each family's entry names only the records with a row
// of it; ③ the refusals — a stale base entry (naming the family), a base whose family without an
// entry is not the default, a family the records cover only in part, a base without provenance or of
// another adapter; ④ the attention-bench block and `attentionOnline` are carried when no bench
// records are given, after their check against the current runtime.

import { assert, assertEquals, assertStringIncludes, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import {
  type AttentionBenchInput,
  deriveGeometryProfile,
  geometryProfileMismatch,
  type GeometrySweepInput,
} from "../tune.ts";
import {
  DEFAULT_GEOMETRY_PROFILE,
  type GeometryProfile,
  type GeometryProfileFamily,
} from "../src/kernels/geometry-profile.ts";
import { ATTENTION_BENCH_SHAPES } from "../src/tune/attention-shapes.ts";
import {
  caseShape,
  I8A8_LINEAR_ROWS_BOUNDS,
  SWEEP_CASES,
  type SweepCase,
} from "../src/tune/cases.ts";
import { infinityJson } from "../src/tune/derive.ts";
import { familyEntries } from "../src/tune/families.ts";
import { familyKernelsId } from "../src/tune/fingerprint.ts";
import { type GeometryCandidate, quickConv2dCandidates } from "../src/tune/geometries.ts";
import { casePlan } from "../src/tune/harness.ts";
import { BENCH_ADAPTER, benchRecord } from "./helpers/attention-bench-records.ts";
import { ADAPTER, type CaseRef, report, row } from "./helpers/sweep-records.ts";

const TARGET = { id: "test-gpu", vendor: "apple", architecture: "metal-3" } as const;

const refOf = (sweepCase: SweepCase): CaseRef => ({
  caseId: sweepCase.id,
  op: sweepCase.op,
  shape: caseShape(sweepCase),
});

/**
 * The rows of every case of `family` in `SWEEP_CASES` (or of `only`): the default row of the sweep's
 * case plan, and the candidate `faster` picks for the case (×1.2 and identical) when it picks one.
 */
const familyRows = (
  family: GeometryProfileFamily,
  options: {
    readonly faster?: (defaultCandidate: GeometryCandidate) => GeometryCandidate | undefined;
    readonly only?: (sweepCase: SweepCase) => boolean;
  } = {},
): Record<string, unknown>[] =>
  familyEntries(SWEEP_CASES, family)
    .filter((sweepCase) => options.only?.(sweepCase) ?? true)
    .flatMap((sweepCase) => {
      const { defaultCandidate } = casePlan(sweepCase, 65535, false);
      const candidate = options.faster?.(defaultCandidate);
      return [
        row(refOf(sweepCase), defaultCandidate, true),
        ...(candidate === undefined
          ? []
          : [row(refOf(sweepCase), candidate, false, { speedup: 1.2, identical: true })]),
      ];
    });

/** The first quick conv2d candidate that is not the case's default (adopted at ×1.2). */
const fasterConv2d = (defaultCandidate: GeometryCandidate): GeometryCandidate | undefined =>
  quickConv2dCandidates().find(({ name }) => name !== defaultCandidate.name);

const input = (
  path: string,
  rows: readonly Record<string, unknown>[],
  fields: Record<string, unknown> = {},
): GeometrySweepInput => ({ report: { ...report(rows), ...fields }, path, sha256: `sha-${path}` });

const derive = (
  inputs: readonly GeometrySweepInput[],
  options: {
    readonly id?: string;
    readonly attentionBench?: readonly AttentionBenchInput[];
    readonly base?: GeometryProfile;
  } = {},
) => deriveGeometryProfile(inputs, { ...TARGET, ...options });

/**
 * The base table: gemm at the default and conv2d adopting a faster geometry, from one record each
 * (attention and the i8a8 families: no case — default with no entry).
 */
const BASE_INPUTS = [
  input("base-gemm.json", familyRows("gemm")),
  input("base-conv2d.json", familyRows("conv2d", { faster: fasterConv2d })),
];

const base = (): GeometryProfile => derive(BASE_INPUTS, { id: "base-gpu" }).profile;

/** `profile`'s provenance (fails when absent). */
const provenanceOf = (profile: GeometryProfile): NonNullable<GeometryProfile["provenance"]> => {
  assert(profile.provenance !== undefined, `${profile.id} has no provenance`);
  return profile.provenance;
};

describe("deriveGeometryProfile with a base table", () => {
  it("derives the families the records cover, copies the others' fields and entries from the base verbatim, and leaves a family without a base entry at the default", () => {
    const from = base();
    assert(
      familyKernelsId(from, "conv2d") !== familyKernelsId(DEFAULT_GEOMETRY_PROFILE, "conv2d"),
      "the base's conv2d fields are not the default",
    );
    const { profile, verdicts } = derive([input("new-attention.json", familyRows("attention"))], {
      base: from,
    });
    // Copied verbatim (the base's adopted conv2d and its gemm)
    assertEquals(profile.conv2d, from.conv2d);
    assertEquals(profile.gemmRows, from.gemmRows);
    const families = provenanceOf(profile).families;
    assertEquals(families.conv2d, provenanceOf(from).families.conv2d);
    assertEquals(families.gemm, provenanceOf(from).families.gemm);
    // Derived from the new record
    assertEquals(families.attention?.sweep, "new-attention.json");
    assertEquals(profile.attention, DEFAULT_GEOMETRY_PROFILE.attention);
    // No case and no base entry: the default (i8a8.linear: the default's geometry in every bucket),
    // no entry
    const { linear, ...i8a8 } = profile.i8a8;
    const { linear: defaultLinear, ...defaultI8a8 } = DEFAULT_GEOMETRY_PROFILE.i8a8;
    assertEquals(i8a8, defaultI8a8);
    assertEquals(
      linear,
      I8A8_LINEAR_ROWS_BOUNDS.map((maxRows) => ({ maxRows, geometry: defaultLinear[0].geometry })),
    );
    for (const family of ["i8a8-linear", "i8a8-attention", "i8a8-conv3d"] as const) {
      assertEquals(families[family], undefined, family);
    }
    assertEquals(profile.id, "test-gpu");
    assertEquals(profile.match, { vendor: "apple", architecture: "metal-3" });
    assertEquals(geometryProfileMismatch(profile, ADAPTER), undefined);
    // One line per carried field, naming the base table and its sweeps
    const carried = verdicts.filter((line) => line.includes("carried over from table 'base-gpu'"));
    assertEquals(carried.length, 7 + 2);
    assert(
      carried.some(
        (line) => line.startsWith("- conv2d.rows64 (") && line.includes("sweeps base-conv2d.json"),
      ),
      carried.join("\n"),
    );
    assert(
      verdicts.some(
        (line) => line.startsWith("- attention.qk (") && line.includes("keeps the default"),
      ),
      verdicts.join("\n"),
    );
  });

  it("derives a covered family from the records even where the base has an entry for it", () => {
    const { profile } = derive([input("new-conv2d.json", familyRows("conv2d"))], { base: base() });
    assertEquals(profile.conv2d, DEFAULT_GEOMETRY_PROFILE.conv2d);
    assertEquals(provenanceOf(profile).families.conv2d?.sweep, "new-conv2d.json");
    assertEquals(provenanceOf(profile).families.gemm?.sweep, "base-gemm.json");
  });

  it("refuses a stale base entry, naming the family (sweep this family again)", () => {
    const from = base();
    const provenance = provenanceOf(from);
    const conv2d = provenance.families.conv2d;
    assert(conv2d !== undefined);
    const records = [input("new-attention.json", familyRows("attention"))];
    for (const [field, message] of [
      [
        "kernels",
        "Base table 'base-gpu', family conv2d is stale: its kernel fingerprint 0000000000000000",
      ],
      [
        "caseSet",
        "Base table 'base-gpu', family conv2d is stale: its case-set version 0000000000000000",
      ],
    ] as const) {
      const stale: GeometryProfile = {
        ...from,
        provenance: {
          ...provenance,
          families: { ...provenance.families, conv2d: { ...conv2d, [field]: "0000000000000000" } },
        },
      };
      const error = assertThrows(() => derive(records, { base: stale }), Error, message);
      assertStringIncludes(error.message, "sweep this family again");
    }
  });

  it("refuses a base whose family without an entry is not the default (a table edited by hand)", () => {
    const from = base();
    const { conv2d: _conv2d, ...others } = provenanceOf(from).families;
    const edited: GeometryProfile = {
      ...from,
      provenance: { ...provenanceOf(from), families: others },
    };
    assertThrows(
      () => derive([input("new-attention.json", familyRows("attention"))], { base: edited }),
      Error,
      "Base table 'base-gpu' has no provenance for family conv2d but its fields are not the default",
    );
  });

  it("refuses a family the records cover only in part when a base is given, naming the missing cases (without a base it derives as before)", () => {
    const [first, ...rest] = familyEntries(SWEEP_CASES, "attention");
    assert(first !== undefined && rest.length > 0);
    const partial = [
      input("partial.json", familyRows("attention", { only: (sweepCase) => sweepCase !== first })),
    ];
    assertThrows(
      () => derive(partial, { base: base() }),
      Error,
      `The records cover family attention only in part (no row of ${first.id})`,
    );
    assertEquals(derive(partial).profile.attention, DEFAULT_GEOMETRY_PROFILE.attention);
  });

  it("refuses a base without provenance, and one of another adapter", () => {
    const records = [input("new-attention.json", familyRows("attention"))];
    assertThrows(
      () => derive(records, { base: DEFAULT_GEOMETRY_PROFILE }),
      Error,
      "Base table 'default' has no provenance",
    );
    const from = base();
    const other: GeometryProfile = {
      ...from,
      provenance: {
        ...provenanceOf(from),
        adapter: { ...provenanceOf(from).adapter, description: "Other GPU" },
      },
    };
    assertThrows(
      () => derive(records, { base: other }),
      Error,
      "Base table 'base-gpu': adapter description",
    );
  });

  it("refuses a base whose vendor, architecture or device differs from the records', naming the field", () => {
    const records = [input("new-attention.json", familyRows("attention"))];
    const from = base();
    for (const field of ["vendor", "architecture", "device"] as const) {
      const other: GeometryProfile = {
        ...from,
        provenance: {
          ...provenanceOf(from),
          adapter: { ...provenanceOf(from).adapter, [field]: "other" },
        },
      };
      assertThrows(
        () => derive(records, { base: other }),
        Error,
        `Base table 'base-gpu': adapter ${field} "other" differs from the sweep records'`,
      );
    }
  });

  it("refuses a base saved in the earlier shape (provenance without per-family entries)", () => {
    const { families: _families, ...earlier } = provenanceOf(base());
    const text = infinityJson({
      ...base(),
      provenance: { ...earlier, sweep: "base.json", kernels: "0123456789abcdef" },
    });
    // What an application holds after JSON.parse and a cast of a table saved by an earlier runtime
    const saved: GeometryProfile = JSON.parse(text);
    assertThrows(
      () => derive([input("new-attention.json", familyRows("attention"))], { base: saved }),
      Error,
      "Base table 'base-gpu': its provenance has no per-family entries",
    );
  });

  it("refuses a base whose fields of a family the records do not cover cannot build the sweep shapes' kernels, naming the family", () => {
    const from = base();
    // conv2d rows32 with tileN 4: the N = 262144 of conv2d 512² becomes 65536 workgroups
    const narrow: GeometryProfile = {
      ...from,
      conv2d: { ...from.conv2d, rows32: { regM: 1, regN: 4, wgX: 1, wgY: 16 } },
    };
    const error = assertThrows(
      () => derive([input("new-attention.json", familyRows("attention"))], { base: narrow }),
      Error,
      "Base table 'base-gpu', family conv2d: its geometry cannot build the kernels of the sweep shapes",
    );
    assertStringIncludes(error.message, "sweep this family again");
  });

  it("ignores a stale base entry of a family the records cover (the family is derived from them)", () => {
    const from = base();
    const provenance = provenanceOf(from);
    const conv2d = provenance.families.conv2d;
    assert(conv2d !== undefined);
    const stale: GeometryProfile = {
      ...from,
      provenance: {
        ...provenance,
        families: { ...provenance.families, conv2d: { ...conv2d, kernels: "0000000000000000" } },
      },
    };
    const { profile } = derive([input("new-conv2d.json", familyRows("conv2d"))], { base: stale });
    assertEquals(profile.conv2d, DEFAULT_GEOMETRY_PROFILE.conv2d);
    assertEquals(provenanceOf(profile).families.conv2d?.sweep, "new-conv2d.json");
    assertEquals(geometryProfileMismatch(profile, ADAPTER), undefined);
  });
});

describe("a family's entry names only the records with a row of it", () => {
  it("joins the paths, digests, dates, candidate sets and browsers of those records alone", () => {
    const chrome = (version: number) => `Mozilla/5.0 Chrome/${version}.0.0.0`;
    const { profile } = derive([
      input("gemm.json", familyRows("gemm"), { userAgent: chrome(153) }),
      input("conv2d.json", familyRows("conv2d"), {
        userAgent: chrome(154),
        date: "2026-09-28T00:00:00.000Z",
      }),
      input("both.json", [...familyRows("gemm"), ...familyRows("conv2d")], {
        userAgent: chrome(155),
        settings: { candidateSet: "quick+" },
      }),
    ]);
    const { gemm, conv2d, attention } = provenanceOf(profile).families;
    assertEquals(gemm?.sweep, "gemm.json, both.json");
    assertEquals(gemm?.sha256, "sha-gemm.json, sha-both.json");
    assertEquals(gemm?.candidateSet, "full, quick+");
    assertEquals(gemm?.userAgent, [chrome(153), chrome(155)]);
    assertEquals(conv2d?.sweep, "conv2d.json, both.json");
    assertEquals(conv2d?.date, "2026-09-28T00:00:00.000Z, 2026-09-27T00:00:00.000Z");
    assertEquals(conv2d?.userAgent, [chrome(154), chrome(155)]);
    assertEquals(attention, undefined);
  });
});

describe("the attention-bench block carried over from the base", () => {
  const bench = (path: string) => ({ report: benchRecord(), path, sha256: `sha-${path}` });
  const withBench = (): GeometryProfile =>
    derive(BASE_INPUTS, { id: "base-gpu", attentionBench: [bench("run1.json")] }).profile;

  it("copies attentionOnline and the block verbatim when no bench records are given, with one verdict line", () => {
    const from = withBench();
    assertEquals(from.attentionOnline, { keyWords: "tiled" });
    const { profile, verdicts } = derive([input("new-attention.json", familyRows("attention"))], {
      base: from,
    });
    assertEquals(profile.attentionOnline, from.attentionOnline);
    assertEquals(provenanceOf(profile).attentionBench, provenanceOf(from).attentionBench);
    assertEquals(geometryProfileMismatch(profile, BENCH_ADAPTER), undefined);
    assert(
      verdicts.some(
        (line) =>
          line.startsWith("- attentionOnline (") &&
          line.includes("carried over from table 'base-gpu'") &&
          line.includes("run1.json"),
      ),
      verdicts.join("\n"),
    );
  });

  it("derives the field from bench records when they are given (the base's block is not read)", () => {
    const from = withBench();
    const { profile } = derive([input("new-attention.json", familyRows("attention"))], {
      base: from,
      attentionBench: [bench("run2.json")],
    });
    assertEquals(provenanceOf(profile).attentionBench?.records, "run2.json");
  });

  it("refuses a stale block (measure the attention bench again) and a base carrying attentionOnline without the block or its kernels", () => {
    const from = withBench();
    const provenance = provenanceOf(from);
    const block = provenance.attentionBench;
    assert(block !== undefined);
    const records = [input("new-attention.json", familyRows("attention"))];
    /** The base with its block changed by `change` (no block when absent). */
    const variant = (change: Partial<typeof block> | undefined): GeometryProfile => {
      const { attentionBench: _block, ...rest } = provenance;
      return {
        ...from,
        provenance:
          change === undefined ? rest : { ...rest, attentionBench: { ...block, ...change } },
      };
    };
    for (const [stale, message] of [
      [{ shapes: "0000000000000000" }, "the attention bench's shape set 0000000000000000"],
      [
        { kernels: "0000000000000000" },
        "the online attention's kernel fingerprint 0000000000000000",
      ],
    ] as const) {
      const error = assertThrows(() => derive(records, { base: variant(stale) }), Error, message);
      assertStringIncludes(error.message, "measure the attention bench again");
    }
    assertThrows(
      () => derive(records, { base: variant(undefined) }),
      Error,
      "Base table 'base-gpu' carries attentionOnline but no provenance.attentionBench",
    );
    const { kernels: _kernels, ...withoutKernels } = block;
    assertThrows(
      () =>
        derive(records, {
          base: { ...from, provenance: { ...provenance, attentionBench: withoutKernels } },
        }),
      Error,
      "carries attentionOnline but its provenance.attentionBench has no kernels",
    );
  });

  it("refuses a block naming the online attention's kernels on a base without attentionOnline", () => {
    const { attentionOnline: _field, ...from } = withBench();
    assert(provenanceOf(from).attentionBench?.kernels !== undefined);
    const error = assertThrows(
      () => derive([input("new-attention.json", familyRows("attention"))], { base: from }),
      Error,
      "Base table 'base-gpu': its provenance.attentionBench names the online attention's kernels but the table chooses no form",
    );
    assertStringIncludes(error.message, "measure the attention bench again");
  });

  it("carries a block without kernels from a base that kept the form N gives, with a verdict line saying so", () => {
    const [first] = ATTENTION_BENCH_SHAPES;
    assert(first !== undefined);
    // The first shape gains nothing: the verdict keeps the form N gives, the block has no kernels
    const kept = derive(BASE_INPUTS, {
      id: "base-gpu",
      attentionBench: [
        {
          report: benchRecord({
            timings: { [first.id]: { kernel: { a: 100_000_000, b: 100_000_000 } } },
          }),
          path: "run1.json",
          sha256: "sha-run1.json",
        },
      ],
    }).profile;
    assertEquals(kept.attentionOnline, undefined);
    assertEquals(provenanceOf(kept).attentionBench?.kernels, undefined);
    const { profile, verdicts } = derive([input("new-attention.json", familyRows("attention"))], {
      base: kept,
    });
    assertEquals(profile.attentionOnline, undefined);
    assertEquals(provenanceOf(profile).attentionBench, provenanceOf(kept).attentionBench);
    assertEquals(geometryProfileMismatch(profile, BENCH_ADAPTER), undefined);
    assert(
      verdicts.some(
        (line) =>
          line.startsWith("- attentionOnline (") &&
          line.includes("carried over from table 'base-gpu' (the form N gives") &&
          line.includes("run1.json"),
      ),
      verdicts.join("\n"),
    );
  });
});
