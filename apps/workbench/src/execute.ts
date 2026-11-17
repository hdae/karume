/**
 * One run of an entry, as the browser shell and the Node run both do it. {@link executeEntry} times
 * the entry's `run`, catches what it throws and wraps the outcome in the record; {@link runEntry} is
 * the whole sequence around it — read the environment and the checkout, execute, serialise, save —
 * with what differs between the hosts passed in as a {@link RunHost}.
 */

import type {
  Entry,
  GpuOptions,
  OpenDistribution,
  OpenInput,
  ParameterValues,
  RunContext,
} from "./entry.ts";
import {
  buildRecord,
  type Checkout,
  type RecordBatch,
  type RecordEnvironment,
  recordText,
  type Settlement,
  type WorkbenchRecord,
} from "./record.ts";

/** The run context as the host gives it; the backend comes from the record's environment. */
export type HostContext = Omit<RunContext, "backend">;

export const executeEntry = async (options: {
  readonly entry: Entry;
  readonly context: HostContext;
  readonly environment: RecordEnvironment;
  readonly checkout: Checkout;
  /** The batch this run is a step of; absent for a single run. */
  readonly batch?: RecordBatch;
}): Promise<WorkbenchRecord> => {
  const { entry, context, environment } = options;
  const started = performance.now();
  let settlement: Settlement;
  try {
    const backend = environment.backend?.name;
    const result = await entry.run({ ...context, ...(backend === undefined ? {} : { backend }) });
    settlement = { kind: "resolved", result, aborted: context.signal.aborted };
  } catch (error) {
    settlement = { kind: "rejected", error, aborted: context.signal.aborted };
  }
  const milliseconds = performance.now() - started;
  return buildRecord({
    entry: { id: entry.id, parameters: context.parameters },
    ...(options.batch === undefined ? {} : { batch: options.batch }),
    endedAt: new Date(),
    milliseconds,
    environment,
    checkout: options.checkout,
    settlement,
  });
};

/** What a host brings to a run: where its facts come from, its GPU, its distributions and its save. */
export type RunHost<Saved> = {
  /** The record's environment, read from an adapter the host requests itself. */
  readonly readEnvironment: () => Promise<RecordEnvironment>;
  readonly readCheckout: () => Promise<Checkout>;
  readonly gpuOptions: GpuOptions;
  readonly openDistribution: OpenDistribution;
  readonly openInput: OpenInput;
  /** Saves a record's text; a host that cannot save says so in `Saved` or throws. */
  readonly save: (record: WorkbenchRecord, text: string) => Promise<Saved>;
};

/** What one run made: the record, the text that was saved, and what the save says. */
export type SingleRun<Saved> = {
  readonly record: WorkbenchRecord;
  readonly text: string;
  readonly saved: Saved;
};

/**
 * One run on `host`: the environment and the checkout are read when the run starts, the record's
 * text is {@link recordText} (the same bytes on every host), and what `host.save` throws rejects.
 */
export const runEntry = async <Saved>(
  host: RunHost<Saved>,
  run: {
    readonly entry: Entry;
    readonly parameters: ParameterValues;
    readonly signal: AbortSignal;
    readonly progress: (text: string) => void;
    /** The batch this run is a step of; absent for a single run. */
    readonly batch?: RecordBatch;
  },
): Promise<SingleRun<Saved>> => {
  const [environment, checkout] = await Promise.all([host.readEnvironment(), host.readCheckout()]);
  const record = await executeEntry({
    entry: run.entry,
    context: {
      parameters: run.parameters,
      gpuOptions: host.gpuOptions,
      signal: run.signal,
      progress: run.progress,
      openDistribution: host.openDistribution,
      openInput: host.openInput,
    },
    environment,
    checkout,
    ...(run.batch === undefined ? {} : { batch: run.batch }),
  });
  const text = recordText(record);
  return { record, text, saved: await host.save(record, text) };
};
