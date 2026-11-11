/**
 * The Session fusion of linear → SRQ (`fuseLinearStaticQuantize`, the rule linearStaticQuantize —
 * the runtime's src/runtime/fusion.ts) on the parallel GEMV, against the same model without the
 * option on the same backend: INT2 / INT4 / INT8 weights at M 1 / 4 / 8 (fused) and M 9 (stays
 * unfused), each Session run twice (the second run takes the derived plan). y and the later
 * consumer of x u32-identical with NaN by payload, linearStaticQuantize at 1 (0 at M 9) and 0.
 */

import type { GpuContext } from "../../../runtime/src/gpu/device.ts";
import { linearStaticQuantizeModel } from "../../../runtime/tests/helpers/linear-static-quantize-graph.ts";
import { checkFusionTwin } from "../fusion-twin.ts";
import { type CheckOptions, type CheckPart, type ExpectedCases } from "../result.ts";
import { twinCases, withTwinSessions } from "./session.ts";

export const linearStaticQuantizeInput = (m: number, k = 1536) => ({
  x: {
    dtype: "f32" as const,
    shape: [m, k],
    data: Float32Array.from({ length: m * k }, (_, i) => Math.sin(i * 0.037) * 0.75),
  },
});

export type LinearStaticQuantizeStorage = "i2" | "i4" | "i8";

export const LINEAR_STATIC_QUANTIZE_STORAGES: readonly LinearStaticQuantizeStorage[] = [
  "i2",
  "i4",
  "i8",
];

/** The output width of a storage (INT2 at 12288 columns). */
export const linearStaticQuantizeWidth = (storage: LinearStaticQuantizeStorage): number =>
  storage === "i2" ? 12288 : 256;

/** M 1 / 4 / 8 fuse; M 9 stays unfused. */
export const LINEAR_STATIC_QUANTIZE_ROWS: readonly number[] = [1, 4, 8, 9];

const PART = "fusion-linear-static-quantize";

/** The cases of {@link checkLinearStaticQuantizeTwins}: 3 storages × 4 row counts × 2 runs. */
export const LINEAR_STATIC_QUANTIZE_TWIN_CASES = { [PART]: 24 } as const satisfies ExpectedCases;

/**
 * Every storage and row count, twice per pair of Sessions: the Session with
 * `fuseLinearStaticQuantize` against the baseline without it, both on `linearGemvReduce:
 * "parallel"` (checkFusionTwin, NaN by payload) on y and the late x consumer, linearStaticQuantize
 * at 1 for M ≤ 8 and 0 for M 9 on the fused side, 0 on the baseline.
 */
export const checkLinearStaticQuantizeTwins = async (
  gpu: GpuContext,
  options: CheckOptions = {},
): Promise<CheckPart> => {
  const sink = twinCases(PART, LINEAR_STATIC_QUANTIZE_TWIN_CASES[PART], options);
  for (const storage of LINEAR_STATIC_QUANTIZE_STORAGES) {
    const n = linearStaticQuantizeWidth(storage);
    for (const m of LINEAR_STATIC_QUANTIZE_ROWS) {
      sink.next();
      const model = linearStaticQuantizeModel({ storage, n, m });
      await withTwinSessions(
        gpu,
        `linear→SRQ ${storage} M${m}`,
        {
          model,
          options: { linearGemvReduce: "parallel", fuseLinearStaticQuantize: true },
        },
        { model, options: { linearGemvReduce: "parallel" } },
        async ({ fused, plain }) => {
          const x = linearStaticQuantizeInput(m);
          // The second run uses the derived plan. The same scale reference is kept, and x lives
          // until the later neg.
          for (let repeat = 0; repeat < 2; repeat++) {
            sink.next();
            sink.add(
              await checkFusionTwin(gpu, {
                label: `${storage} M${m} run ${repeat + 1}`,
                fused,
                plain,
                feeds: x,
                counter: "linearStaticQuantize",
                hits: m <= 8 ? 1 : 0,
                nan: "payload",
              }),
            );
          }
        },
      );
    }
  }
  return {
    id: PART,
    title:
      "linearStaticQuantize: linear → SRQ fused on the parallel GEMV vs unfused (INT2 / 4 / 8)",
    cases: sink.cases,
    requirements: [],
  };
};
