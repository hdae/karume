import { useCallback, useEffect, useRef, useState } from "react";

/** What a running task reports (all optional; the status card shows what is there). */
export type RunProgress = {
  /** What runs now: a stage, "loading", "prefill"… */
  readonly stage?: string;
  /** A counter of the current stage (denoise steps, tokens, tiles…). */
  readonly count?: { readonly done: number; readonly total: number; readonly unit: string };
  /** One more line of detail. */
  readonly note?: string;
};

export type RunTaskState<Result> =
  | { readonly kind: "idle" }
  | { readonly kind: "running"; readonly startedAt: number; readonly progress: RunProgress }
  | { readonly kind: "done"; readonly result: Result; readonly seconds: number }
  | { readonly kind: "cancelled" }
  | { readonly kind: "error"; readonly error: unknown };

/** What a task gets: the signal to pass on (Cancel and leaving the page abort it) and the reporter. */
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
 */
export const useRunTask = <Result>(): RunTask<Result> => {
  const [state, setState] = useState<RunTaskState<Result>>({ kind: "idle" });
  const [now, setNow] = useState(() => performance.now());
  const controllerRef = useRef<AbortController | undefined>(undefined);
  const running = state.kind === "running";

  useEffect(() => {
    if (!running) return undefined;
    const timer = setInterval(() => setNow(performance.now()), 250);
    return () => clearInterval(timer);
  }, [running]);
  // Leaving the page cancels a run.
  useEffect(() => () => controllerRef.current?.abort(new Error("The page was left")), []);

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
        const result = await task({ signal: controller.signal, report });
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
    [],
  );
  const cancel = useCallback(
    () => controllerRef.current?.abort(new Error("Cancelled by the user")),
    [],
  );
  return { state, now, running, start, cancel };
};
