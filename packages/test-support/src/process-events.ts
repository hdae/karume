/**
 * Observing escaped errors in tests.
 *
 * Deno and browsers dispatch `error` / `unhandledrejection` events on `globalThis`, and a test can
 * `preventDefault()` them to keep the runner alive while it counts them. Node reports the same
 * conditions through `process` events, and Vitest's own listeners turn any occurrence into a run
 * error. These helpers stand in for the event listeners: they take over the process event for the
 * duration of `body` (the runner's listeners are detached and restored afterwards), collect what
 * was reported, and hand the list back for the test to assert on.
 */

type ProcessEvent = "uncaughtException" | "unhandledRejection";

const capture = async <T>(
  name: ProcessEvent,
  body: () => Promise<T>,
): Promise<{ readonly result: T; readonly reported: readonly unknown[] }> => {
  // Through the generic emitter interface: the per-event overloads of `process.on` do not take
  // a union of event names.
  const emitter: NodeJS.EventEmitter = process;
  const event: string = name;
  const reported: unknown[] = [];
  const others = emitter.listeners(event);
  const own = (error: unknown): void => {
    reported.push(error);
  };
  emitter.removeAllListeners(event);
  emitter.on(event, own);
  try {
    const result = await body();
    return { result, reported };
  } finally {
    emitter.off(event, own);
    // `listeners()` is typed as Function[]; the entries came from `on` and have that shape.
    for (const listener of others) emitter.on(event, listener as (...args: unknown[]) => void);
  }
};

/** Runs `body`; exceptions that escape to the process (e.g. thrown from a microtask) are collected. */
export const captureUncaughtExceptions = <T>(
  body: () => Promise<T>,
): Promise<{ readonly result: T; readonly reported: readonly unknown[] }> =>
  capture("uncaughtException", body);

/** Runs `body`; rejections nobody handled by the next task are collected (their reasons). */
export const captureUnhandledRejections = <T>(
  body: () => Promise<T>,
): Promise<{ readonly result: T; readonly reported: readonly unknown[] }> =>
  capture("unhandledRejection", body);
