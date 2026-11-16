/**
 * The attention route bench: times the two a8 attention routes (S stored in f16 over three
 * dispatches, and the online int8 kernel) at Wan2.2's shapes with the kernels, row blocks and
 * variants production would use on this host's GPU (`runAttentionRouteBench` of
 * `@karume/runtime/tune`), apart from a generation. Optionally it also times a measure-first
 * candidate against the online kernel (the tiled key axis, perf-ledger K-106).
 */

import { acquireGpu } from "@karume/runtime";
import { runAttentionRouteBench } from "@karume/runtime/tune";

import { benchShapes, SHAPE_IDS } from "../attention-bench/shapes.ts";
import { attentionBenchSummary, benchProgressText } from "../attention-bench/summary.ts";
import { booleanValue, choicesValue, type Entry, integerValue, ParameterError } from "../entry.ts";

const entry: Entry = {
  id: "attention-bench",
  title: "Attention route bench",
  description:
    "Per-kernel GPU time of the two a8 attention routes (S in f16 over three dispatches, and the online kernel) at Wan2.2 TI2V-5B's shapes, with the kernels and row blocks production would use here. Without timestamp-query the times are wall clock: compare ratios, not absolute values.",
  parameters: [
    {
      kind: "choices",
      name: "shapes",
      label: "Shapes",
      options: SHAPE_IDS,
      default: SHAPE_IDS,
    },
    {
      kind: "integer",
      name: "rowsPerDispatch",
      label: "Rows per dispatch (0: production's split; capped at each shape's query count)",
      min: 0,
      max: Number.MAX_SAFE_INTEGER,
      default: 0,
    },
    {
      kind: "integer",
      name: "rounds",
      label: "Rounds per dispatch (the min is kept)",
      min: 1,
      max: Number.MAX_SAFE_INTEGER,
      default: 5,
    },
    {
      kind: "boolean",
      name: "quantization",
      label:
        "Also time the shared quantization (q, k, Vᵀ — once per node, outside the route totals)",
      default: true,
    },
    {
      kind: "boolean",
      name: "onlineTiled",
      label:
        "Also time the candidate: the online kernel reading Vᵀ in its own key order (perf-ledger K-106), with its producer and an identity check",
      default: false,
    },
  ],
  run: async ({ parameters, gpuOptions, signal, progress }) => {
    const shapes = benchShapes(
      choicesValue(parameters, "shapes", SHAPE_IDS),
      integerValue(parameters, "rowsPerDispatch"),
    );
    if (shapes.length === 0) throw new ParameterError("shapes", "choose at least one shape");
    const adapter = await (gpuOptions.gpu ?? navigator.gpu).requestAdapter();
    if (adapter === null) throw new Error("WebGPU returned no adapter");
    // gpuTiming: true fails on an adapter without timestamp-query; without it the bench falls back
    // to the wall clock (per pass, divided by the repetitions — the submit overhead included).
    const timestamps = adapter.features.has("timestamp-query");
    const gpu = await acquireGpu({
      ...gpuOptions,
      subgroups: "available",
      ...(timestamps ? { gpuTiming: true } : {}),
    });
    let done = 0;
    try {
      // A cancel stops before the next dispatch; the report keeps the shapes finished before it.
      const report = await runAttentionRouteBench(gpu, {
        shapes,
        rounds: integerValue(parameters, "rounds"),
        quantization: booleanValue(parameters, "quantization"),
        onlineTiled: booleanValue(parameters, "onlineTiled"),
        signal,
        onProgress: (event) => {
          if (event.kind === "shape") done += 1;
          progress(benchProgressText(event, done, shapes.length));
        },
      });
      return { summary: attentionBenchSummary(report), data: report };
    } finally {
      gpu.destroy();
    }
  },
};

export default entry;
