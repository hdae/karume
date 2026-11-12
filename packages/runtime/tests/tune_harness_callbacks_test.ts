// Pins the control flow of the sweep's measurement core (sweepCase / runSweep of
// src/tune/harness.ts) on a fake device (no GPU needed).
//
// Two things are pinned:
//
// 1. **An exception from a user callback reaches the caller as is** — wherever it is thrown, in the
//    progress (each onProgress position), a row (onRow) or a case (onCase), it does not turn into a
//    measurement failure (a failed row, defaultRepeatError), onCase is not called twice, and the
//    allocated resources are returned.
// 2. **A default whose perDispatch is 0 (a timestamp difference of 0) is not a basis for ratios** —
//    the candidate rows carry no ratio to the default, there is no re-measurement of the default
//    (driftRatio is not written), the record stays readable by the generator after a JSON round
//    trip, and that case drops out of the ratio inputs. If only the re-measurement is 0, it remains
//    as a re-measurement failure.
//
// GpuContext is the real one (lock, errorScope, pipeline cache); only the device is replaced. The
// test decides the timestamp difference per pipeline the pass set last (= the pipeline key).
//
// The last three describe blocks pin the pure report helpers the sweep writes through
// (compareToDefault / roundsLookQuantized / driftOutOfRange) on a hand-written row, without the
// fake device.

import { assert, assertEquals, assertRejects, assertStrictEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { GpuContext, readAdapterInfo, type RequiredLimits } from "../src/gpu/device.ts";
import { SWEEP_CASES, type SweepCase } from "../src/tune/cases.ts";
import { deriveProfile, parseSweepReport } from "../src/tune/derive.ts";
import { familiesOf } from "../src/tune/families.ts";
import {
  candidateSetCaseSetIds,
  candidateSetDefaultKernelsIds,
  sweepCandidateKernelsId,
} from "../src/tune/fingerprint.ts";
import {
  casePlan,
  createSweepContext,
  destroySweepContext,
  runSweep,
  type SweepContext,
  type SweepHooks,
} from "../src/tune/harness.ts";
import {
  type CaseSummary,
  compareToDefault,
  driftOutOfRange,
  REPORT_FORMAT,
  roundsLookQuantized,
  type SweepRow,
} from "../src/tune/report.ts";

/**
 * A pass's timestamp difference (`key` = the key of the pipeline the pass set, `read` = which
 * readback of that key it is).
 */
type DeltaOf = (key: string, read: number) => number;

/** A buffer the fake device created (for counting destroys). */
type FakeBuffer = { readonly label: string; destroyed: boolean };

const LIMITS: RequiredLimits = {
  maxBufferSize: 2 ** 30,
  maxStorageBufferBindingSize: 2 ** 30,
  maxUniformBufferBindingSize: 65536,
  maxStorageBuffersPerShaderStage: 10,
  maxUniformBuffersPerShaderStage: 12,
  maxComputeWorkgroupStorageSize: 32768,
  maxComputeInvocationsPerWorkgroup: 1024,
  maxComputeWorkgroupSizeX: 1024,
  maxComputeWorkgroupSizeY: 1024,
  maxComputeWorkgroupSizeZ: 64,
  maxComputeWorkgroupsPerDimension: 65535,
};

/**
 * A fake device with only the surface the sweep's measurement core touches (with timestamp-query).
 * submit completes at once, a timestamp readback returns the difference `delta` decides, and an
 * output readback returns zeros.
 */
const fakeSweepDevice = (delta: DeltaOf): { device: GPUDevice; buffers: FakeBuffer[] } => {
  const buffers: FakeBuffer[] = [];
  const reads = new Map<string, number>();
  let lastKey = "";
  const createBuffer = (descriptor: GPUBufferDescriptor) => {
    const label = descriptor.label ?? "";
    const record: FakeBuffer = { label, destroyed: false };
    buffers.push(record);
    return {
      size: descriptor.size,
      destroy: (): void => {
        record.destroyed = true;
      },
      unmap: (): void => undefined,
      mapAsync: (): Promise<void> => Promise.resolve(),
      getMappedRange: (_offset = 0, length = descriptor.size): ArrayBuffer => {
        if (label !== "geometry-sweep-timestamps") return new ArrayBuffer(length);
        const read = reads.get(lastKey) ?? 0;
        reads.set(lastKey, read + 1);
        return new BigUint64Array([0n, BigInt(delta(lastKey, read))]).buffer;
      },
    };
  };
  const device = {
    lost: new Promise<GPUDeviceLostInfo>(() => {}),
    features: new Set(["timestamp-query"]),
    destroy: (): void => undefined,
    pushErrorScope: (): void => undefined,
    popErrorScope: (): Promise<GPUError | null> => Promise.resolve(null),
    createBuffer,
    createQuerySet: () => ({ destroy: (): void => undefined }),
    createShaderModule: () => ({
      getCompilationInfo: () => Promise.resolve({ messages: [] }),
    }),
    createComputePipelineAsync: (descriptor: GPUComputePipelineDescriptor) =>
      Promise.resolve({ label: descriptor.label ?? "", getBindGroupLayout: () => ({}) }),
    createBindGroup: () => ({}),
    createCommandEncoder: () => ({
      clearBuffer: (): void => undefined,
      copyBufferToBuffer: (): void => undefined,
      resolveQuerySet: (): void => undefined,
      beginComputePass: () => ({
        setPipeline: (pipeline: { readonly label: string }): void => {
          lastKey = pipeline.label;
        },
        setBindGroup: (): void => undefined,
        dispatchWorkgroups: (): void => undefined,
        end: (): void => undefined,
      }),
      finish: () => ({}),
    }),
    queue: {
      submit: (): void => undefined,
      writeBuffer: (): void => undefined,
      onSubmittedWorkDone: (): Promise<void> => Promise.resolve(),
    },
  };
  // Test-only boundary: the whole DOM GPUDevice type is not reproduced (only the surface the
  // sweep's measurement core touches)
  return { device: device as unknown as GPUDevice, buffers };
};

/** The case with the smallest resources (filling its inputs is fast even on the fake device). */
const SMALLEST: SweepCase = SWEEP_CASES.map((sweepCase) => ({
  sweepCase,
  bytes: casePlan(sweepCase, 65535, false).resources.reduce((sum, spec) => sum + spec.bytes, 0),
})).sort((left, right) => left.bytes - right.bytes)[0].sweepCase;

const DEFAULT_KEY = casePlan(SMALLEST, 65535, false).launch(
  casePlan(SMALLEST, 65535, false).defaultCandidate,
).key;

/**
 * Builds a sweep context on the fake device, runs `body`, and reports whether every buffer was
 * returned after the context was released.
 */
const withSweepContext = async (
  delta: DeltaOf,
  body: (context: SweepContext) => Promise<void>,
): Promise<{ readonly leaked: readonly string[] }> => {
  const { device, buffers } = fakeSweepDevice(delta);
  const gpu = new GpuContext(device, readAdapterInfo({}), LIMITS, new Set());
  const context = await createSweepContext(gpu, "deno-raw-tick");
  try {
    await body(context);
  } finally {
    destroySweepContext(context);
  }
  return { leaked: buffers.filter((buffer) => !buffer.destroyed).map((buffer) => buffer.label) };
};

const SETTINGS = { rounds: 2, candidateSet: "quick", timestampUnit: "deno-raw-tick" } as const;

/**
 * Sweeps the smallest case with the quick candidates (the default row first, the candidate rows
 * after it).
 */
const sweepSmallest = (context: SweepContext, hooks: SweepHooks) =>
  runSweep(context, [SMALLEST], SETTINGS, hooks);

class HookError extends Error {}

describe("an exception from a user callback reaches the caller without turning into a measurement failure", () => {
  it("without a throw, the case has the default's re-measurement (driftRatio) and onCase is called once (control)", async () => {
    const summaries: CaseSummary[] = [];
    const { leaked } = await withSweepContext(
      () => 1000,
      async (context) => {
        const result = await sweepSmallest(context, {
          onCase: (summary) => summaries.push(summary),
        });
        assertEquals(result.cases.length, 1);
        assertEquals(result.cases[0].defaultRepeat?.driftRatio, 1);
        assertEquals(result.cases[0].defaultRepeatError, undefined);
        assert(
          result.rows.every((row) => row.error === undefined),
          "there is a failed row",
        );
      },
    );
    assertEquals(summaries.length, 1);
    assertEquals(leaked, []);
  });

  const positions: readonly (readonly [string, (throwIt: () => never) => SweepHooks])[] = [
    [
      "onProgress: preparing inputs",
      (throwIt) => ({
        onProgress: (message) => (message.endsWith("preparing inputs") ? throwIt() : undefined),
      }),
    ],
    [
      "onProgress: candidate (1/n)",
      (throwIt) => ({
        onProgress: (message) => (message.includes(" (1/") ? throwIt() : undefined),
      }),
    ],
    [
      "onProgress: re-measuring the default geometry",
      (throwIt) => ({
        onProgress: (message) =>
          message.endsWith("re-measuring the default geometry") ? throwIt() : undefined,
      }),
    ],
    [
      "onRow: the second row",
      (throwIt) => {
        let seen = 0;
        return {
          onRow: () => {
            seen += 1;
            if (seen === 2) throwIt();
          },
        };
      },
    ],
    ["onCase", (throwIt) => ({ onCase: () => throwIt() })],
  ];

  for (const [position, hooksOf] of positions) {
    it(`an exception thrown at ${position} arrives as is, onCase is not called twice, and the resources are returned`, async () => {
      const thrown = new HookError(position);
      let caseCalls = 0;
      const hooks = hooksOf(() => {
        throw thrown;
      });
      const { leaked } = await withSweepContext(
        () => 1000,
        async (context) => {
          const caught = await assertRejects(() =>
            sweepSmallest(context, {
              ...hooks,
              onCase: (summary) => {
                caseCalls += 1;
                hooks.onCase?.(summary);
              },
            }),
          );
          assertStrictEquals(caught, thrown);
        },
      );
      assert(caseCalls <= 1, `onCase was called ${caseCalls} times`);
      assertEquals(leaked, [], "allocated resources were not returned");
    });
  }
});

/**
 * Turns a runSweep result into a record of the same shape as runGeometrySweep's (the fields the
 * generator reads — including the fingerprints the writer bakes in).
 */
const recordOf = (
  rows: readonly SweepRow[],
  cases: readonly CaseSummary[],
  dp4a: boolean,
  gpuTiming: { readonly unit: "ns" | "deno-raw-tick"; readonly quantized: boolean },
): Record<string, unknown> => ({
  format: REPORT_FORMAT,
  date: "2026-10-02T00:00:00.000Z",
  caseSets: candidateSetCaseSetIds(SETTINGS.candidateSet, familiesOf(rows)),
  defaultKernels: candidateSetDefaultKernelsIds(SETTINGS.candidateSet, familiesOf(rows)),
  adapter: { vendor: "apple", architecture: "metal-3", device: "", description: "Test GPU" },
  candidateKernels: sweepCandidateKernelsId(rows, dp4a),
  dp4a,
  gpuTiming: { feature: true, ...gpuTiming },
  settings: { candidateSet: SETTINGS.candidateSet, quick: true },
  cases,
  rows,
});

describe("a default whose perDispatch is 0 (a timestamp difference of 0) is not a basis for ratios", () => {
  it("the candidate rows have no ratio to the default, there is no re-measurement, the record is readable after a JSON round trip, and that case drops out of the ratio inputs", async () => {
    await withSweepContext(
      (key) => (key === DEFAULT_KEY ? 0 : 1000),
      async (context) => {
        const { rows, cases } = await sweepSmallest(context, {});
        const [base, candidate] = rows;
        assertEquals(base.isDefault, true);
        assertEquals(base.perDispatch, 0);
        assert(
          candidate !== undefined && (candidate.perDispatch ?? 0) > 0,
          "the candidate row is not positive",
        );
        assertEquals(candidate.speedupVsDefault, undefined);
        assertEquals(cases, [{ caseId: SMALLEST.id, elapsedMs: cases[0].elapsedMs }]);
        for (const gpuTiming of [
          { unit: "deno-raw-tick", quantized: false },
          { unit: "ns", quantized: true },
        ] as const) {
          // The writer is plain JSON.stringify (NaN / Infinity turn into null) — round-trip, then
          // read
          const text = JSON.stringify(recordOf(rows, cases, context.dp4a, gpuTiming));
          const source = parseSweepReport(JSON.parse(text), { path: "zero.json", sha256: "sha" });
          const verdicts = deriveProfile([source], {
            vendor: "apple",
            architecture: "metal-3",
            minSpeedup: 1.05,
          });
          const excluded = verdicts.flatMap((verdict) => verdict.excluded);
          assertEquals(
            excluded.map(({ caseId, geometry, reason }) => ({ caseId, geometry, reason })),
            [{ caseId: SMALLEST.id, geometry: undefined, reason: "no default re-measurement" }],
            gpuTiming.unit,
          );
        }
      },
    );
  });

  it("when the first measurement is positive and only the re-measurement is 0, it writes no ratio of 0 and keeps a re-measurement failure (only the case drops out)", async () => {
    await withSweepContext(
      (key, read) => (key === DEFAULT_KEY && read >= SETTINGS.rounds ? 0 : 1000),
      async (context) => {
        const { cases } = await sweepSmallest(context, {});
        assertEquals(cases[0].defaultRepeat, undefined);
        assert(
          cases[0].defaultRepeatError?.includes(
            "perDispatch of the default re-measurement is 0",
          ) === true,
          `defaultRepeatError: ${cases[0].defaultRepeatError}`,
        );
      },
    );
  });
});

/** A hand-written sweep row for the pure report helpers below. */
const ROW: SweepRow = {
  caseId: "c",
  op: "linear",
  shape: "M1 N1 K1",
  censusCount: 1,
  geometry: "g",
  geometryParams: { regM: 1, regN: 4, wgX: 4, wgY: 16 },
  isDefault: false,
  reps: 2,
  rounds: [400_000, 300_000],
  perDispatch: 150_000,
  outputSha256: "a",
};

describe("compareToDefault", () => {
  it("a row faster than the default gets a ratio above 1 and records the digest match", () => {
    const row = compareToDefault(ROW, { perDispatch: 300_000, outputSha256: "a" });
    assertEquals(row.speedupVsDefault, 2);
    assertEquals(row.identicalToDefault, true);
  });

  it("records a mismatch when the digest differs (the geometry changed the values — the mark of not adoptable)", () => {
    assertEquals(
      compareToDefault(ROW, { perDispatch: 150_000, outputSha256: "b" }).identicalToDefault,
      false,
    );
  });

  it("adds no comparison when the default row failed (so it does not read as a mismatch)", () => {
    const row = compareToDefault(ROW, {});
    assertEquals(row.speedupVsDefault, undefined);
    assertEquals(row.identicalToDefault, undefined);
  });

  it("writes no speed ratio when the default's or this row's perDispatch is 0 (a timestamp difference of 0) — the output match is written", () => {
    // A ratio that turns into 0 or infinity makes the generator, which accepts only positive
    // ratios, reject the whole record
    for (const [name, row, reference] of [
      ["the default is 0", ROW, { perDispatch: 0, outputSha256: "a" }],
      ["this row is 0", { ...ROW, perDispatch: 0 }, { perDispatch: 300_000, outputSha256: "a" }],
    ] as const) {
      const compared = compareToDefault(row, reference);
      assertEquals(compared.speedupVsDefault, undefined, name);
      assertEquals(compared.identicalToDefault, true, name);
    }
  });
});

describe("roundsLookQuantized", () => {
  it("suspects quantization when the unit is ns and every round is a multiple of 100 µs", () => {
    assertEquals(roundsLookQuantized([ROW], "ns"), true);
    assertEquals(roundsLookQuantized([{ ...ROW, rounds: [400_000, 300_001] }], "ns"), false);
  });

  it("does not judge raw ticks or the wall clock", () => {
    assertEquals(roundsLookQuantized([ROW], "deno-raw-tick"), false);
    assertEquals(roundsLookQuantized([ROW], "wall"), false);
  });
});

describe("driftOutOfRange", () => {
  it("does not warn when the default's re-measurement ratio is within 0.9–1.1 (boundaries included)", () => {
    for (const ratio of [0.9, 1, 1.1]) assertEquals(driftOutOfRange(ratio), false, String(ratio));
  });

  it("warns, as a hint to re-measure, outside the range and when the ratio is not a number", () => {
    for (const ratio of [0.89, 1.11, Number.NaN]) {
      assertEquals(driftOutOfRange(ratio), true, String(ratio));
    }
  });
});
