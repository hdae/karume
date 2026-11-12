// The second input of the profile derivation (ADR 0117 addendum 2026-11-12): attention-bench records
// decide the table's `attentionOnline` (the online attention's tiled key axis — perf-ledger K-106),
// through the public `deriveGeometryProfile` of ../tune.ts. No GPU: the records are synthetic
// (helpers/attention-bench-records.ts), their keys built by the current runtime.
//
// What this pins: ① the rule — identical on every shape of every record and ×1.05 on every shape,
// the gain being kernel + producer per node from each arm's minimum pass, records combined per shape
// by geometric mean; ② what is written — the field only when adopted, `provenance.attentionBench`
// whenever records are given, a kernel fingerprint that covers the field; ③ the refusals — a record
// that cannot stand for the current runtime on this adapter throws instead of being skipped.

import {
  assert,
  assertEquals,
  assertNotEquals,
  assertStringIncludes,
  assertThrows,
} from "@std/assert";
import { describe, it } from "vite-plus/test";
import {
  deriveGeometryProfile,
  geometryProfileJson,
  geometryProfileMismatch,
  parseGeometryProfileJson,
} from "../tune.ts";
import type { GeometryProfile } from "../src/kernels/geometry-profile.ts";
import { ATTENTION_BENCH_SHAPES } from "../src/tune/attention-shapes.ts";
import { attentionBenchShapeSetId, geometryProfileKernelsId } from "../src/tune/fingerprint.ts";
import {
  BENCH_ADAPTER,
  benchRecord,
  expectedGain,
  sweepRecord,
} from "./helpers/attention-bench-records.ts";

const SWEEPS = [
  { report: sweepRecord(), path: "sweep-a.json", sha256: "sha-sweep-a" },
  {
    report: { ...sweepRecord(), date: "2026-09-30T00:00:00.000Z" },
    path: "sweep-b.json",
    sha256: "sha-sweep-b",
  },
];
const TARGET = { id: "test-gpu", vendor: "apple", architecture: "metal-3" } as const;

const bench = (
  path: string,
  options: Parameters<typeof benchRecord>[0] = {},
): { readonly report: unknown; readonly path: string; readonly sha256: string } => ({
  report: benchRecord(options),
  path,
  sha256: `sha-${path}`,
});

const derive = (attentionBench: readonly ReturnType<typeof bench>[]) =>
  deriveGeometryProfile(SWEEPS, { ...TARGET, attentionBench });

/** The verdict lines of attentionOnline (its head and the lines under it). */
const attentionLines = (verdicts: readonly string[]): readonly string[] =>
  verdicts.slice(verdicts.findIndex((line) => line.startsWith("- attentionOnline")));

const [FIRST, SECOND] = ATTENTION_BENCH_SHAPES;

describe("deriveGeometryProfile: attention-bench records decide attentionOnline", () => {
  it("adopts the tiled key axis when every shape is identical and gains ×1.05 or more, and writes the field and provenance.attentionBench", () => {
    const records = [bench("run1.json"), bench("run2.json", { date: "2026-11-12T01:00:00.000Z" })];
    const { profile, verdicts } = derive(records);
    assertEquals(profile.attentionOnline, { keyWords: "tiled" });
    assertEquals(profile.provenance?.attentionBench, {
      records: "run1.json, run2.json",
      sha256: "sha-run1.json, sha-run2.json",
      date: "2026-11-12T00:00:00.000Z, 2026-11-12T01:00:00.000Z",
      shapes: attentionBenchShapeSetId(),
    });
    const [head] = attentionLines(verdicts);
    assertStringIncludes(head, `${ATTENTION_BENCH_SHAPES.length} shapes, 2 records`);
    assertStringIncludes(head, "adopted tiled");
    // The geometry fields are the sweeps' own (the second input touches only its field)
    const without = deriveGeometryProfile(SWEEPS, TARGET).profile;
    assertEquals(profile.gemmRows, without.gemmRows);
    assertEquals(profile.i8a8, without.i8a8);
  });

  it("fingerprints the online attention's kernels in a table that chose the field, and the table matches the current runtime", () => {
    const { profile } = derive([bench("run1.json")]);
    const without = deriveGeometryProfile(SWEEPS, TARGET).profile;
    assertEquals(profile.provenance?.kernels, geometryProfileKernelsId(profile));
    assertNotEquals(profile.provenance?.kernels, without.provenance?.kernels);
    // The field alone moves the fingerprint (id and provenance are not read)
    const { attentionOnline: _chosen, ...stripped } = profile;
    assertEquals(geometryProfileKernelsId(stripped), without.provenance?.kernels);
    assertEquals(geometryProfileMismatch(profile, BENCH_ADAPTER), undefined);
  });

  it("reads each arm's minimum pass: a stall burst in three of five rounds of today's kernel still adopts, at the gain of the minimums", () => {
    // A's passes in rounds 1–3 run at ×6 (the median of the paired rounds' ratios would claim ×6)
    const stalled = bench("stalled.json", { timings: { [FIRST.id]: { stall: 6 } } });
    const { profile, verdicts } = derive([stalled]);
    assertEquals(profile.attentionOnline, { keyWords: "tiled" });
    const line = attentionLines(verdicts).find((entry) => entry.startsWith(`  - ${FIRST.id}:`));
    assertStringIncludes(line ?? "", `×${expectedGain(FIRST).toFixed(3)}`);
    assert(expectedGain(FIRST) < 1.2, "the default timing gains about ×1.1");
  });

  it("combines the records of one shape by the geometric mean of their gains (×1.25 and ×0.97 adopt, ×1.10 and ×0.98 do not)", () => {
    const at = (b: number) => ({ timings: { [FIRST.id]: { kernel: { a: 100_000_000, b } } } });
    const fast = bench("fast.json", at(80_000_000));
    const slow = bench("slow.json", at(103_000_000));
    assertEquals(derive([fast, slow]).profile.attentionOnline, { keyWords: "tiled" });
    const mild = bench("mild.json", at(91_000_000));
    const slower = bench("slower.json", at(102_000_000));
    const { profile, verdicts } = derive([mild, slower]);
    assertEquals(profile.attentionOnline, undefined);
    assertStringIncludes(attentionLines(verdicts)[0], `keeps the form N gives (${FIRST.id}: ×1.0`);
  });

  it("keeps the form N gives when one shape gains less than ×1.05, and still records the bench in provenance", () => {
    const short = bench("short.json", {
      timings: { [SECOND.id]: { kernel: { a: 104_000_000, b: 100_000_000 } } },
    });
    const { profile, verdicts } = derive([short]);
    assertEquals(profile.attentionOnline, undefined);
    assertEquals(profile.provenance?.attentionBench?.records, "short.json");
    assertStringIncludes(
      attentionLines(verdicts)[0],
      `${SECOND.id}: ×${expectedGain(SECOND, { kernel: { a: 104_000_000, b: 100_000_000 } }).toFixed(3)}`,
    );
    assertStringIncludes(attentionLines(verdicts)[0], "< ×1.050");
  });

  it("keeps the form N gives when one shape of one record is not identical, however fast", () => {
    const records = [
      bench("run1.json"),
      bench("run2.json", {
        timings: { [SECOND.id]: { kernel: { a: 200_000_000, b: 100_000_000 }, differingWords: 3 } },
      }),
    ];
    const { profile, verdicts } = derive(records);
    assertEquals(profile.attentionOnline, undefined);
    const lines = attentionLines(verdicts);
    assertStringIncludes(lines[0], "not identical in 1 observation(s)");
    assert(
      lines.some((line) =>
        line.includes(`not identical: run2.json (${SECOND.id}): 3 differing O words`),
      ),
      lines.join("\n"),
    );
  });

  it("drops a failed pair and a quantised pass with a rounding-error bound over 1% from the ratio material (a shape left without any keeps the default)", () => {
    const failed = bench("failed.json", {
      timings: { [FIRST.id]: { kernelError: "device lost" } },
    });
    const ok = bench("ok.json");
    // The failed record's other shapes and the second record keep the verdict
    assertEquals(derive([failed, ok]).profile.attentionOnline, { keyWords: "tiled" });
    const alone = derive([failed]);
    assertEquals(alone.profile.attentionOnline, undefined);
    assertStringIncludes(
      attentionLines(alone.verdicts)[0],
      `${FIRST.id}: every observation was dropped`,
    );
    // Every pass a multiple of 100 µs (quantised): 2 ms passes give E = 0.1 / 2.2 + 0.1 / 2 > 1%,
    // 220 ms / 200 ms passes E = 0.095% (kept)
    const quantised = (kernel: { readonly a: number; readonly b: number }) =>
      Object.fromEntries(
        ATTENTION_BENCH_SHAPES.map(({ id }) => [
          id,
          { kernel, producer: { a: 100_000_000, b: 100_000_000 }, quantum: 100_000 },
        ]),
      );
    const coarse = bench("coarse.json", { timings: quantised({ a: 2_200_000, b: 2_000_000 }) });
    const coarseResult = derive([coarse]);
    assertEquals(coarseResult.profile.attentionOnline, undefined);
    assertStringIncludes(
      attentionLines(coarseResult.verdicts).join("\n"),
      "rounding-error bound E",
    );
    const fine = bench("fine.json", { timings: quantised({ a: 220_000_000, b: 200_000_000 }) });
    assertEquals(derive([fine]).profile.attentionOnline, { keyWords: "tiled" });
  });

  it("without attention-bench records the table has neither the field nor provenance.attentionBench", () => {
    const { profile, verdicts } = deriveGeometryProfile(SWEEPS, TARGET);
    assertEquals("attentionOnline" in profile, false);
    assertEquals(profile.provenance !== undefined && "attentionBench" in profile.provenance, false);
    assertEquals(
      verdicts.some((line) => line.startsWith("- attentionOnline")),
      false,
    );
  });

  it("accepts per-shape records that cover the set together (the Node bench writes one record per shape)", () => {
    const records = ATTENTION_BENCH_SHAPES.map((shape) =>
      bench(`${shape.id}.json`, { shapes: [shape] }),
    );
    assertEquals(derive(records).profile.attentionOnline, { keyWords: "tiled" });
  });
});

describe("deriveGeometryProfile: attention-bench records that cannot stand for the current runtime are refused", () => {
  const refuses = (records: readonly ReturnType<typeof bench>[], message: string): void => {
    assertThrows(() => derive(records), Error, message);
  };

  it("refuses a wall-clock record, a record without the candidate or the producer pair, and another format", () => {
    refuses([bench("wall.json", { unit: "wall" })], 'timing.unit is "wall"');
    refuses(
      [bench("plain.json", { onlineTiled: false })],
      "settings.onlineTiled and settings.quantization",
    );
    refuses(
      [
        {
          report: { format: "karume-playground-attention-bench/2", bench: benchRecord() },
          path: "page.json",
          sha256: "sha-page",
        },
      ],
      "a playground file holds the record under `bench`",
    );
  });

  it("refuses kernels or producers other than the current runtime's, and another measured window", () => {
    const [own, tiled] = ["attention_online:v1:old", "attention_online:v1:old:ktile"] as const;
    refuses(
      [bench("old.json", { timings: { [FIRST.id]: { kernelKeys: [own, tiled] } } })],
      "the kernel pair measured attention_online:v1:old",
    );
    refuses(
      [bench("old-producer.json", { timings: { [FIRST.id]: { producerKeys: ["a", "b"] } } })],
      "the producer pair measured a / b",
    );
    refuses(
      [bench("rows.json", { timings: { [FIRST.id]: { rows: 64 } } })],
      "the measured dispatch covers 64 rows",
    );
  });

  it("refuses another adapter, a record given twice, a shape outside the set or with other dimensions, and a set the records do not cover", () => {
    refuses(
      [bench("other.json", { adapter: { ...BENCH_ADAPTER, description: "Other GPU" } })],
      "adapter description",
    );
    const once = bench("once.json");
    refuses([once, { ...once, path: "again.json" }], "passed twice");
    refuses(
      [bench("foreign.json", { timings: { [FIRST.id]: { shape: { ...FIRST, id: "foreign" } } } })],
      "the shape is not in the bench's shape set",
    );
    refuses(
      [
        bench("dims.json", {
          timings: { [FIRST.id]: { shape: { ...FIRST, keys: FIRST.keys + 1 } } },
        }),
      ],
      "differ from the bench's",
    );
    refuses([bench("partial.json", { shapes: [FIRST] })], "do not cover the bench's shape set");
  });
});

describe("a table derived with attention-bench records through storage and matching", () => {
  it("survives the JSON round trip with attentionOnline and provenance.attentionBench", () => {
    const { profile } = derive([bench("run1.json")]);
    const parsed = parseGeometryProfileJson(geometryProfileJson(profile));
    assertEquals(parsed, profile);
  });

  it("mismatches when its bench shape set is not the current runtime's", () => {
    const { profile } = derive([bench("run1.json")]);
    const provenance = profile.provenance;
    assert(provenance?.attentionBench !== undefined);
    const moved: GeometryProfile = {
      ...profile,
      provenance: {
        ...provenance,
        attentionBench: { ...provenance.attentionBench, shapes: "0000000000000000" },
      },
    };
    assertStringIncludes(
      geometryProfileMismatch(moved, BENCH_ADAPTER) ?? "",
      "the attention bench's shape set differs (table 0000000000000000",
    );
  });
});
