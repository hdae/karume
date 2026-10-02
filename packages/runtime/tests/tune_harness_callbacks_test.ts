// 掃引の計測核（src/tune/harness.ts の sweepCase / runSweep）の制御の流れを、偽の device で固定する（GPU 不要）。
//
// 固定するのは 2 点:
//
// 1. **利用者のコールバックの例外はそのまま呼び手へ届く** — 進捗（onProgress の各位置）・行（onRow）・ケース
//    （onCase）のどこで投げても、計測の失敗（失敗の行・defaultRepeatError）に化けず、onCase は 2 度呼ばれず、
//    確保した資源は返る。
// 2. **perDispatch が 0 の既定（timestamp の差が 0）は比の土台にしない** — 候補の行は既定比を持たず、既定の
//    再測定も無く（driftRatio を書かない）、記録は JSON を往復しても生成器に読め、そのケースは比の材料から外れる。
//    再測定だけが 0 なら再測定の失敗として残る。
//
// GpuContext は本物（ロック・errorScope・パイプラインキャッシュ）で、差し替えるのは device だけ。timestamp の差は
// pass が最後に set したパイプライン（= パイプラインキー）ごとに、テストが決める。

import { assert, assertEquals, assertRejects, assertStrictEquals } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { GpuContext, readAdapterInfo, type RequiredLimits } from "../src/gpu/device.ts";
import { SWEEP_CASES, type SweepCase } from "../src/tune/cases.ts";
import { deriveProfile, parseSweepReport } from "../src/tune/derive.ts";
import { sweepCandidateKernelsId } from "../src/tune/fingerprint.ts";
import {
  casePlan,
  createSweepContext,
  destroySweepContext,
  runSweep,
  type SweepContext,
  type SweepHooks,
} from "../src/tune/harness.ts";
import { type CaseSummary, REPORT_FORMAT, type SweepRow } from "../src/tune/report.ts";

/** pass の timestamp の差（`key` = pass が set したパイプラインのキー・`read` = そのキーの何度目の読み戻しか）。 */
type DeltaOf = (key: string, read: number) => number;

/** 偽の device が作ったバッファ（破棄の数え上げ用）。 */
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
 * 掃引の計測核が触る面だけを持つ偽の device（timestamp-query あり）。submit は即座に完了し、timestamp の
 * 読み戻しは `delta` が決めた差を返し、出力の読み戻しは 0 埋めを返す。
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
  // テスト専用の境界: DOM の GPUDevice 型全体は再現しない（掃引の計測核が触る面だけ）
  return { device: device as unknown as GPUDevice, buffers };
};

/** 資源の最も小さいケース（偽の device でも入力の充填が速い）。 */
const SMALLEST: SweepCase = SWEEP_CASES
  .map((sweepCase) => ({
    sweepCase,
    bytes: casePlan(sweepCase, 65535, false).resources.reduce((sum, spec) => sum + spec.bytes, 0),
  }))
  .sort((left, right) => left.bytes - right.bytes)[0].sweepCase;

const DEFAULT_KEY = casePlan(SMALLEST, 65535, false).launch(
  casePlan(SMALLEST, 65535, false).defaultCandidate,
).key;

/** 偽の device の上で掃引の文脈を作って `body` を走らせ、文脈を返した後に全てのバッファが返ったかを返す。 */
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

/** 最小のケースを quick の候補で掃引する（既定の行が先頭・候補の行が続く）。 */
const sweepSmallest = (context: SweepContext, hooks: SweepHooks) =>
  runSweep(context, [SMALLEST], SETTINGS, hooks);

class HookError extends Error {}

describe("利用者のコールバックの例外は計測の失敗に化けずに呼び手へ届く", () => {
  it("投げなければ、ケースは既定の再測定（driftRatio）を持ち、onCase は 1 度だけ呼ばれる（対照）", async () => {
    const summaries: CaseSummary[] = [];
    const { leaked } = await withSweepContext(() => 1000, async (context) => {
      const result = await sweepSmallest(context, { onCase: (summary) => summaries.push(summary) });
      assertEquals(result.cases.length, 1);
      assertEquals(result.cases[0].defaultRepeat?.driftRatio, 1);
      assertEquals(result.cases[0].defaultRepeatError, undefined);
      assert(result.rows.every((row) => row.error === undefined), "失敗の行がある");
    });
    assertEquals(summaries.length, 1);
    assertEquals(leaked, []);
  });

  const positions: readonly (readonly [string, (throwIt: () => never) => SweepHooks])[] = [
    ["onProgress: 入力を用意中", (throwIt) => ({
      onProgress: (message) => message.endsWith("入力を用意中") ? throwIt() : undefined,
    })],
    ["onProgress: 候補（1/n）", (throwIt) => ({
      onProgress: (message) => message.includes("（1/") ? throwIt() : undefined,
    })],
    ["onProgress: 既定幾何の再測定", (throwIt) => ({
      onProgress: (message) => message.endsWith("既定幾何の再測定") ? throwIt() : undefined,
    })],
    ["onRow: 2 行目", (throwIt) => {
      let seen = 0;
      return {
        onRow: () => {
          seen += 1;
          if (seen === 2) throwIt();
        },
      };
    }],
    ["onCase", (throwIt) => ({ onCase: () => throwIt() })],
  ];

  for (const [position, hooksOf] of positions) {
    it(`${position} で投げた例外はそのまま届き、onCase は 2 度呼ばれず、資源は返る`, async () => {
      const thrown = new HookError(position);
      let caseCalls = 0;
      const hooks = hooksOf(() => {
        throw thrown;
      });
      const { leaked } = await withSweepContext(() => 1000, async (context) => {
        const caught = await assertRejects(() =>
          sweepSmallest(context, {
            ...hooks,
            onCase: (summary) => {
              caseCalls += 1;
              hooks.onCase?.(summary);
            },
          })
        );
        assertStrictEquals(caught, thrown);
      });
      assert(caseCalls <= 1, `onCase が ${caseCalls} 回呼ばれた`);
      assertEquals(leaked, [], "確保した資源が返っていない");
    });
  }
});

/** runSweep の結果を、runGeometrySweep と同じ形の記録にする（生成器が読む欄 — 書き手の焼く指紋を含む）。 */
const recordOf = (
  rows: readonly SweepRow[],
  cases: readonly CaseSummary[],
  dp4a: boolean,
  gpuTiming: { readonly unit: "ns" | "deno-raw-tick"; readonly quantized: boolean },
): Record<string, unknown> => ({
  format: REPORT_FORMAT,
  date: "2026-10-02T00:00:00.000Z",
  adapter: { vendor: "apple", architecture: "metal-3", device: "", description: "Test GPU" },
  candidateKernels: sweepCandidateKernelsId(rows, dp4a),
  dp4a,
  gpuTiming: { feature: true, ...gpuTiming },
  settings: { candidateSet: SETTINGS.candidateSet, quick: true },
  cases,
  rows,
});

describe("perDispatch が 0 の既定（timestamp の差が 0）は比の土台にしない", () => {
  it("候補の行は既定比を持たず、再測定も無く、記録は JSON を往復しても読めてそのケースは比の材料から外れる", async () => {
    await withSweepContext((key) => key === DEFAULT_KEY ? 0 : 1000, async (context) => {
      const { rows, cases } = await sweepSmallest(context, {});
      const [base, candidate] = rows;
      assertEquals(base.isDefault, true);
      assertEquals(base.perDispatch, 0);
      assert(candidate !== undefined && (candidate.perDispatch ?? 0) > 0, "候補の行が正でない");
      assertEquals(candidate.speedupVsDefault, undefined);
      assertEquals(cases, [{ caseId: SMALLEST.id, elapsedMs: cases[0].elapsedMs }]);
      for (
        const gpuTiming of [
          { unit: "deno-raw-tick", quantized: false },
          { unit: "ns", quantized: true },
        ] as const
      ) {
        // 書き手は素の JSON.stringify（NaN / Infinity は null に化ける）— 往復させてから読む
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
          [{ caseId: SMALLEST.id, geometry: undefined, reason: "既定の再測定が無い" }],
          gpuTiming.unit,
        );
      }
    });
  });

  it("初回が正で再測定だけが 0 なら、比 0 を書かずに再測定の失敗として残す（ケースだけが外れる）", async () => {
    await withSweepContext(
      (key, read) => key === DEFAULT_KEY && read >= SETTINGS.rounds ? 0 : 1000,
      async (context) => {
        const { cases } = await sweepSmallest(context, {});
        assertEquals(cases[0].defaultRepeat, undefined);
        assert(
          cases[0].defaultRepeatError?.includes("既定の再測定の perDispatch が 0") === true,
          `defaultRepeatError: ${cases[0].defaultRepeatError}`,
        );
      },
    );
  });
});
