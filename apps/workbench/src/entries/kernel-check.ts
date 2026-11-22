/**
 * The kernel checks: the identity checks of `@karume/kernel-checks` — the same cases and criteria as
 * the runtime's node GPU tests — on this host's GPU. In a browser that reaches the backends the node
 * lanes (Dawn / Vulkan) do not: D3D12 in Chrome on Windows, Metal on macOS. This entry is the manual
 * procedure of the release runbook's subgroup checkbox (all fourteen checks).
 */

import { type CheckProgress, subgroup32Available } from "@karume/kernel-checks";
import { acquireGpu, type GpuContext } from "@karume/runtime";

import { choicesValue, type Entry } from "../entry.ts";
import { KERNEL_CHECKS, type KernelCheck } from "../kernel-check/checks.ts";
import { kernelCheckResult } from "../kernel-check/summary.ts";
import {
  type CheckOutcomes,
  KERNEL_CHECK_IDS,
  subgroupShortfall,
} from "../kernel-check/verdict.ts";
import { errorText, seconds } from "../record.ts";

const unavailableReason = (gpu: GpuContext, adapterFeatures: readonly string[]): string => {
  const missing = subgroupShortfall(new Set(adapterFeatures), gpu.wgslLanguageFeatures);
  return missing.length === 0
    ? "The device was acquired without the subgroup features"
    : `Needs subgroups, subgroup-size-control and WGSL subgroup_id; this adapter lacks ${missing.join(", ")}`;
};

const progressText = (check: KernelCheck, progress: CheckProgress): string =>
  `${check.title} · ${progress.part} · ${progress.done} / ${progress.total} cases`;

const entry: Entry = {
  id: "kernel-check",
  title: "Kernel checks",
  description:
    "Word-for-word identity of the kernel rewrites the node GPU tests check (subgroup reductions, the states-attention rewrites, the integer and byte-plane GEMVs, f16 slots and scores, the expert linear, the built-in fusions, the attention band) on this backend. A check the device cannot run is not available, never a pass.",
  parameters: [
    {
      kind: "choices",
      name: "checks",
      label: "Checks",
      options: KERNEL_CHECK_IDS,
      default: KERNEL_CHECK_IDS,
    },
  ],
  run: async ({ parameters, gpuOptions, signal, progress }) => {
    const selected = choicesValue(parameters, "checks", KERNEL_CHECK_IDS);
    // The adapter's features say which subgroup feature is missing; the device's alone cannot.
    const adapter = await (gpuOptions.gpu ?? navigator.gpu).requestAdapter();
    if (adapter === null) throw new Error("WebGPU returned no adapter");
    const adapterFeatures = [...adapter.features];
    const gpu = await acquireGpu({ ...gpuOptions, subgroups: "available" });
    let outcomes: CheckOutcomes = {};
    try {
      for (const check of KERNEL_CHECKS.filter(({ id }) => selected.includes(id))) {
        // A cancelled run keeps what settled; the check it stopped in reads as not run.
        if (signal.aborted) break;
        if (check.needsSubgroups && !subgroup32Available(gpu)) {
          outcomes = {
            ...outcomes,
            [check.id]: { kind: "not-available", reason: unavailableReason(gpu, adapterFeatures) },
          };
          continue;
        }
        progress(check.title);
        const started = performance.now();
        try {
          const report = await check.run(gpu, {
            signal,
            onProgress: (state) => progress(progressText(check, state)),
          });
          outcomes = {
            ...outcomes,
            [check.id]: { kind: "done", report, seconds: seconds(performance.now() - started) },
          };
        } catch (error) {
          if (signal.aborted) break;
          outcomes = {
            ...outcomes,
            [check.id]: {
              kind: "error",
              message: errorText(error),
              seconds: seconds(performance.now() - started),
            },
          };
        }
      }
    } finally {
      gpu.destroy();
    }
    return kernelCheckResult(adapterFeatures, outcomes);
  },
};

export default entry;
