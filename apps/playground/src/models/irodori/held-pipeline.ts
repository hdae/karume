/**
 * The pipeline the page keeps between runs: reused while its key holds, replaced (the old one
 * disposed first) when the key changes. Pure over an injected slot and build, so `vp test` covers
 * the lifecycle without a GPU.
 */

/** A built pipeline and the key it was built for. */
export type Held<Key, Pipeline> = { readonly key: Key; readonly pipeline: Pipeline };

/** Where the held pipeline lives (the page: a ref for the run and the unmount, plus render state). */
export type PipelineSlot<Key, Pipeline> = {
  readonly get: () => Held<Key, Pipeline> | undefined;
  readonly set: (held: Held<Key, Pipeline> | undefined) => void;
};

export type DisposablePipeline = { readonly dispose: () => Promise<void> };

export type AcquireOptions<Key, Pipeline extends DisposablePipeline> = {
  readonly slot: PipelineSlot<Key, Pipeline>;
  readonly key: Key;
  readonly same: (held: Key, wanted: Key) => boolean;
  readonly build: () => Promise<Pipeline>;
  /**
   * Whether `key` is still the one the page runs for once the build finished (its GPU may have been
   * lost or acquired again meanwhile): a pipeline for a stale key is disposed, never held.
   */
  readonly isCurrent: (key: Key) => boolean;
  readonly signal: AbortSignal;
  /** Called before the previous pipeline is disposed (for a progress line). */
  readonly onReleasing?: () => void;
};

/**
 * The pipeline for `key`: the held one when its key is the same, else a new one. The slot is emptied
 * before the old pipeline is disposed, so a failed or cancelled build leaves nothing held. A new
 * pipeline is held only when the run was not aborted during the build and its key is still current;
 * otherwise it is disposed and the abort reason (or a stale-key error) is thrown.
 */
export const acquirePipeline = async <Key, Pipeline extends DisposablePipeline>(
  options: AcquireOptions<Key, Pipeline>,
): Promise<{ readonly pipeline: Pipeline; readonly built: boolean }> => {
  const { slot, key, signal } = options;
  const held = slot.get();
  if (held !== undefined && options.same(held.key, key)) {
    return { pipeline: held.pipeline, built: false };
  }
  slot.set(undefined);
  if (held !== undefined) {
    options.onReleasing?.();
    await held.pipeline.dispose();
  }
  signal.throwIfAborted();
  const pipeline = await options.build();
  if (signal.aborted) {
    await pipeline.dispose();
    signal.throwIfAborted();
  }
  if (!options.isCurrent(key)) {
    await pipeline.dispose();
    throw new Error(
      "The GPU changed while the pipeline was being built (device lost or acquired again), so the pipeline was released. Run again on the new GPU.",
    );
  }
  slot.set({ key, pipeline });
  return { pipeline, built: true };
};
