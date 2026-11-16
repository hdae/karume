/**
 * One run of an entry, as the browser shell and the Node run both do it: time the entry's `run`,
 * catch what it throws, and wrap the outcome in the record. The host gathers the environment and
 * the checkout before the run and saves the record after it.
 */

import type { Entry, GpuOptions, OpenDistribution, ParameterValues } from "./entry.ts";
import {
  buildRecord,
  type Checkout,
  type RecordEnvironment,
  type Settlement,
  type WorkbenchRecord,
} from "./record.ts";

export const executeEntry = async (options: {
  readonly entry: Entry;
  readonly parameters: ParameterValues;
  readonly gpuOptions: GpuOptions;
  readonly signal: AbortSignal;
  readonly progress: (text: string) => void;
  readonly openDistribution: OpenDistribution;
  readonly environment: RecordEnvironment;
  readonly checkout: Checkout;
}): Promise<WorkbenchRecord> => {
  const { entry, parameters, gpuOptions, signal, progress, openDistribution } = options;
  const started = performance.now();
  let settlement: Settlement;
  try {
    const result = await entry.run({ parameters, gpuOptions, signal, progress, openDistribution });
    settlement = { kind: "resolved", result, aborted: signal.aborted };
  } catch (error) {
    settlement = { kind: "rejected", error, aborted: signal.aborted };
  }
  const milliseconds = performance.now() - started;
  return buildRecord({
    entry: { id: entry.id, parameters },
    endedAt: new Date(),
    milliseconds,
    environment: options.environment,
    checkout: options.checkout,
    settlement,
  });
};
