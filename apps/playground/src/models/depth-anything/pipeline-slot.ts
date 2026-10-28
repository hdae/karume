/**
 * Who owns the page's open pipeline: one slot that holds at most one pipeline, built for one key
 * (the settings it was built for). Ownership lives here, outside React state, so a pipeline is
 * disposed by the slot whatever React commits or batches: when the wanted key moves away from it,
 * when the page lets go (`want(undefined)`), and when a build finishes for a key that is no longer
 * wanted. Pure apart from the injected builder, so `vp test` covers it with fakes.
 */

/** What the slot can hold: something it can dispose. */
export type Disposable = { readonly dispose: () => Promise<void> };

/** The pipeline the slot holds, the key it was built for and how long the build took. */
export type SlotEntry<Key, Value> = {
  readonly key: Key;
  readonly value: Value;
  readonly loadMs: number;
};

/** The pipeline a run uses; `built` when this run built it. */
export type Obtained<Key, Value> = {
  readonly entry: SlotEntry<Key, Value>;
  readonly built: boolean;
};

export type PipelineSlot<Key, Value> = {
  /** The pipeline open for the wanted key, or `undefined`. The same object until it changes. */
  readonly current: () => SlotEntry<Key, Value> | undefined;
  /** Calls `listener` whenever {@link current} changes; returns the unsubscribe. */
  readonly subscribe: (listener: () => void) => () => void;
  /**
   * Sets the key runs are for (`undefined` = none: settings not ready, or the page let go). An open
   * pipeline built for another key is disposed at once; a build in flight for another key is
   * disposed when it finishes.
   */
  readonly want: (key: Key | undefined) => void;
  /**
   * The open pipeline for `key`, or a new one from `build(signal)`. Rejects, disposing the new
   * pipeline, when `signal` aborted or the wanted key moved away while it built. `key` MUST be the
   * wanted key, and one build at a time (the page runs one task at a time).
   */
  readonly obtain: (
    key: Key,
    build: (signal: AbortSignal) => Promise<Value>,
    signal: AbortSignal,
  ) => Promise<Obtained<Key, Value>>;
};

export const createPipelineSlot = <Key, Value extends Disposable>(options: {
  readonly sameKey: (left: Key, right: Key) => boolean;
  /** Milliseconds (`performance.now`), injected for tests. */
  readonly now: () => number;
  /**
   * Receives every dispose the slot starts on its own (useGpu's `track`, so the device outlives
   * it). A dispose inside `obtain` is part of the run that awaits it.
   */
  readonly track: (disposal: Promise<void>) => Promise<void>;
}): PipelineSlot<Key, Value> => {
  const { sameKey, now, track } = options;
  let wanted: Key | undefined;
  let entry: SlotEntry<Key, Value> | undefined;
  let building = false;
  const listeners = new Set<() => void>();

  const isWanted = (key: Key): boolean => wanted !== undefined && sameKey(wanted, key);
  const replace = (next: SlotEntry<Key, Value> | undefined): void => {
    const previous = entry;
    entry = next;
    if (previous !== undefined) void track(previous.value.dispose());
    for (const listener of listeners) listener();
  };

  return {
    current: () => entry,
    subscribe: (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    want: (key) => {
      wanted = key;
      if (entry !== undefined && !isWanted(entry.key)) replace(undefined);
    },
    obtain: async (key, build, signal) => {
      if (!isWanted(key)) throw new Error("The settings changed; run again");
      if (entry !== undefined && sameKey(entry.key, key)) return { entry, built: false };
      if (building) throw new Error("A pipeline is already being built");
      building = true;
      try {
        const started = now();
        const value = await build(signal);
        const loadMs = now() - started;
        if (signal.aborted || !isWanted(key)) {
          await value.dispose();
          signal.throwIfAborted();
          throw new Error("The settings changed while the pipeline was building; run again");
        }
        const built: SlotEntry<Key, Value> = { key, value, loadMs };
        replace(built);
        return { entry: built, built: true };
      } finally {
        building = false;
      }
    },
  };
};
