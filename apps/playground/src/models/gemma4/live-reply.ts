/**
 * The reply a running turn streams, held outside React state so a token does not re-render the page.
 *
 * The decode loop runs on the main thread, so every render during a turn lands inside the decode
 * window the page measures. Updates are therefore coalesced: `update` only records the latest value
 * and asks the scheduler for one flush (one animation frame in the page), and only the bubble that
 * subscribes (`useSyncExternalStore`) re-renders. Pure apart from the injected scheduler, so
 * `vp test` covers it.
 */

export type LiveReply = {
  /** The reply streamed so far. */
  readonly text: string;
  /** What the turn does now ("prefill 1 / 2 chunks", "decode 12 / 512 tokens"). */
  readonly stage?: string;
};

/** Runs `flush` later (once); returns a function that cancels it. */
export type FlushScheduler = (flush: () => void) => () => void;

export type LiveReplyStore = {
  readonly subscribe: (listener: () => void) => () => void;
  /** The last flushed value (the same object until the next flush, as `useSyncExternalStore` needs). */
  readonly snapshot: () => LiveReply;
  /** Records a new value; subscribers see it at the next scheduled flush. */
  readonly update: (next: LiveReply) => void;
  /** Clears the reply and tells subscribers at once (a pending flush is dropped). */
  readonly reset: () => void;
  /** Drops a pending flush (the page is left). */
  readonly dispose: () => void;
};

const EMPTY: LiveReply = { text: "" };

export const createLiveReplyStore = (schedule: FlushScheduler): LiveReplyStore => {
  const listeners = new Set<() => void>();
  let published = EMPTY;
  let pending: LiveReply | undefined;
  let cancel: (() => void) | undefined;
  const notify = (): void => {
    for (const listener of listeners) listener();
  };
  const flush = (): void => {
    cancel = undefined;
    if (pending === undefined) return;
    published = pending;
    pending = undefined;
    notify();
  };
  return {
    subscribe: (listener) => {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
    snapshot: () => published,
    update: (next) => {
      pending = next;
      cancel ??= schedule(flush);
    },
    reset: () => {
      cancel?.();
      cancel = undefined;
      pending = undefined;
      if (published === EMPTY) return;
      published = EMPTY;
      notify();
    },
    dispose: () => {
      cancel?.();
      cancel = undefined;
      pending = undefined;
    },
  };
};

/** The page's scheduler: one animation frame. */
export const frameScheduler: FlushScheduler = (flush) => {
  const handle = requestAnimationFrame(flush);
  return () => cancelAnimationFrame(handle);
};
