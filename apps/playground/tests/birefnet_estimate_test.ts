import type { AdmissionReport } from "@karume/runtime";
import { describe, expect, it } from "vite-plus/test";

import { summarizeEstimate } from "../src/models/birefnet/estimate.ts";

const MIB = 2 ** 20;

const report = (scenarios: AdmissionReport["scenarios"]): AdmissionReport => ({
  resident: {
    weights: {
      compressedBytes: 400 * MIB,
      uncompressedBytes: 30 * MIB,
      expandedBytes: 2 * MIB,
      totalBytes: 432 * MIB,
    },
    stateBytes: 0,
  },
  scenarios,
  planBackingBudgetBytes: 256 * MIB,
  peakAccountedBytes: 1700 * MIB,
  unaccounted: ["in-node temporaries of fused ops"],
});

describe("summarizeEstimate", () => {
  it("keeps the resident weights, the peak and what the runtime leaves uncounted", () => {
    const summary = summarizeEstimate(
      report([{ name: "prefill", ioBytes: 12 * MIB, workspaceBytes: 1200 * MIB }]),
    );
    expect(summary.weightsBytes).toBe(432 * MIB);
    expect(summary.peakBytes).toBe(1700 * MIB);
    expect(summary.unaccounted).toEqual(["in-node temporaries of fused ops"]);
  });

  it("reports the largest run shape, its buffers and intermediates together", () => {
    const summary = summarizeEstimate(
      report([
        { name: "prefill", ioBytes: 12 * MIB, workspaceBytes: 1200 * MIB },
        { name: "decode", ioBytes: 500 * MIB, workspaceBytes: 600 * MIB },
      ]),
    );
    expect(summary.runBytes).toBe(1212 * MIB);
  });

  it("reports no run bytes for a report without run shapes", () => {
    expect(summarizeEstimate(report([])).runBytes).toBe(0);
  });
});
