import { useCallback, useEffect, useRef, useState } from "react";

import type { GpuHandle } from "@/hooks/use-gpu";

/**
 * What a running task reports (all optional; the status card shows what is there). `cancellable:
 * false` marks a stage that takes no signal (a Session build, a GPU run): Cancel is disabled until
 * the next report, and `cancelNote` — required with it — says why and when a cancel would work.
 */
export type RunProgress = {
  /** What runs now: a stage, "loading", "prefill"… */
  readonly stage?: string;
  /** A counter of the current stage (denoise steps, tokens, tiles…). */
  readonly count?: { readonly done: number; readonly total: number; readonly unit: string };
  /** One more line of detail. */
  readonly note?: string;
} & (
  | { readonly cancellable?: undefined; readonly cancelNote?: never }
  | { readonly cancellable: false; readonly cancelNote: string }
);

export type RunTaskState<Result> =
  | { readonly kind: "idle" }
  | { readonly kind: "running"; readonly startedAt: number; readonly progress: RunProgress }
  | { readonly kind: "done"; readonly result: Result; readonly seconds: number }
  | { readonly kind: "cancelled" }
  | { readonly kind: "error"; readonly error: unknown };

/**
 * What a task gets: the signal to pass on (Cancel, leaving the page and re-acquiring the GPU abort
 * it) and the reporter.
 */
export type RunTaskContext = {
  readonly signal: AbortSignal;
  readonly report: (progress: RunProgress) => void;
};

/** What the status card needs of a task. */
export type RunTaskView = {
  readonly state: RunTaskState<unknown>;
  /** `performance.now()`, ticking every 250 ms while running (for the elapsed seconds). */
  readonly now: number;
  readonly cancel: () => void;
};

export type RunTask<Result> = RunTaskView & {
  readonly state: RunTaskState<Result>;
  readonly running: boolean;
  /** Runs `task` unless one is running (one run at a time). Never rejects: failures land in `state`. */
  readonly start: (task: (context: RunTaskContext) => Promise<Result>) => Promise<void>;
};

const isCancellation = (error: unknown, signal: AbortSignal): boolean =>
  signal.aborted &&
  (error === signal.reason || (error instanceof DOMException && error.name === "AbortError"));

/**
 * One cancellable run of a page (a generation, a synthesis, a chat turn): its state, the elapsed-time
 * ticker, Cancel, and an abort when the page is left. A rejection after Cancel is `cancelled` only
 * when it is the cancellation itself (the signal's reason or an `AbortError`); anything else stays an
 * error, so a real failure is never hidden behind a Cancel.
 *
 * `gpu` is the page's GPU (useGpu): every run goes through its `track`, so the device outlives it,
 * and a run is aborted when its lease ends (the page is left, or the GPU is acquired again), so a
 * stalled run does not hold up the destroy and the next acquisition until someone presses Cancel.
 */
export const useRunTask = <Result>(gpu: Pick<GpuHandle, "track" | "lease">): RunTask<Result> => {
  const { track, lease } = gpu;
  const [state, setState] = useState<RunTaskState<Result>>({ kind: "idle" });
  const [now, setNow] = useState(() => performance.now());
  const controllerRef = useRef<AbortController | undefined>(undefined);
  const running = state.kind === "running";

  useEffect(() => {
    if (!running) return undefined;
    const timer = setInterval(() => setNow(performance.now()), 250);
    return () => clearInterval(timer);
  }, [running]);
  // The end of the lease (leaving the page, re-acquiring the GPU) cancels a run, in the same commit
  // as useGpu's release: before the destroy waits for the run.
  useEffect(
    () => () =>
      controllerRef.current?.abort(
        new Error("The GPU was released (the page was left or the GPU is acquired again)"),
      ),
    [lease],
  );

  const start = useCallback(
    async (task: (context: RunTaskContext) => Promise<Result>): Promise<void> => {
      if (controllerRef.current !== undefined) return;
      const controller = new AbortController();
      controllerRef.current = controller;
      const startedAt = performance.now();
      setNow(startedAt);
      setState({ kind: "running", startedAt, progress: {} });
      const report = (progress: RunProgress): void => {
        if (controllerRef.current !== controller) return;
        setState({ kind: "running", startedAt, progress });
      };
      try {
        const result = await track(task({ signal: controller.signal, report }));
        setState({ kind: "done", result, seconds: (performance.now() - startedAt) / 1000 });
      } catch (error) {
        setState(
          isCancellation(error, controller.signal)
            ? { kind: "cancelled" }
            : { kind: "error", error },
        );
      } finally {
        controllerRef.current = undefined;
      }
    },
    [track],
  );
  const cancel = useCallback(
    () => controllerRef.current?.abort(new Error("Cancelled by the user")),
    [],
  );
  return { state, now, running, start, cancel };
};
