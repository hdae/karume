/**
 * The page's one `Pipeline`, opened on Run and kept for the next run while the source, model, preset
 * and GPU stay the same. Opening reads the manifest, the container descriptors and the tokenizer; it
 * builds no Session (every generation opens and closes its stages — the text encoder alone holds
 * about 7.4 GiB of weights on the device), so a held pipeline holds no GPU memory between runs and
 * keeping it only saves the opening.
 *
 * The held pipeline is disposed when the selection moves away from it, on **Release pipeline** and
 * when the page is left. The GPU belongs to the page: `dispose` only drops the pipeline. Disposals
 * run one after another; `release` resolves once every disposal started so far has settled and never
 * rejects — a failed dispose (a lost device) lands in `releaseError`. Every disposal goes through the
 * page GPU's `track`, so useGpu destroys the device only after it.
 */

import type { AssetProgress } from "@karume/hub";
import { Pipeline } from "@karume/models/moss";
import type { GpuContext } from "@karume/runtime";
import { useCallback, useEffect, useRef, useState } from "react";

import type { TrackGpuWork } from "@/hooks/use-gpu";
import { type PipelineSource, pretrainedRef } from "@/lib/distributions";

import { MOSS_MOUNT } from "./settings";

/** What a pipeline is opened for. */
export type PipelineSpec = {
  readonly gpu: GpuContext;
  /** Stable per choice (`usePipelineSource`), so it compares by identity. */
  readonly source: PipelineSource;
  readonly model: string;
  readonly quant: string;
};

/** What the status card shows of the held pipeline. */
export type HeldPipelineInfo = {
  readonly source: PipelineSource["kind"];
  readonly model: string;
  readonly quant: string;
  readonly openSeconds: number;
};

type Held = {
  readonly spec: PipelineSpec;
  readonly pipeline: Pipeline;
  readonly info: HeldPipelineInfo;
};

const sameSpec = (left: PipelineSpec, right: Partial<PipelineSpec>): boolean =>
  left.gpu === right.gpu &&
  left.source === right.source &&
  left.model === right.model &&
  left.quant === right.quant;

export type OpenedPipeline = {
  readonly pipeline: Pipeline;
  /** How long opening took; `undefined` when the held pipeline was reused. */
  readonly openMilliseconds: number | undefined;
};

export const useMossPipeline = (
  selection: Partial<PipelineSpec>,
  track: TrackGpuWork,
): {
  readonly held: HeldPipelineInfo | undefined;
  /** The latest failed dispose (cleared when the next Run opens or reuses a pipeline). */
  readonly releaseError: unknown;
  /**
   * The held pipeline when it was opened for `spec`, else a new one (the held one is disposed
   * first, so two never coexist).
   */
  readonly open: (
    spec: PipelineSpec,
    signal: AbortSignal,
    onProgress: (progress: AssetProgress) => void,
  ) => Promise<OpenedPipeline>;
  readonly release: () => Promise<void>;
} => {
  const heldRef = useRef<Held | undefined>(undefined);
  const [held, setHeld] = useState<HeldPipelineInfo | undefined>(undefined);
  const [releaseError, setReleaseError] = useState<unknown>(undefined);
  const mountedRef = useRef(false);
  const disposalsRef = useRef<Promise<void>>(Promise.resolve());
  // The selection as of the last commit: open() compares it with its spec once opening finishes.
  const selectionRef = useRef<Partial<PipelineSpec>>({});

  /** Queues a dispose behind the earlier ones; resolves when all have settled, never rejects. */
  const disposeQueued = useCallback(
    (pipeline: Pipeline | undefined): Promise<void> => {
      if (pipeline !== undefined) {
        disposalsRef.current = track(
          disposalsRef.current
            .then(() => pipeline.dispose())
            .catch((error: unknown) => {
              if (mountedRef.current) setReleaseError(error);
            }),
        );
      }
      return disposalsRef.current;
    },
    [track],
  );

  const release = useCallback((): Promise<void> => {
    const current = heldRef.current;
    heldRef.current = undefined;
    if (mountedRef.current) setHeld(undefined);
    return disposeQueued(current?.pipeline);
  }, [disposeQueued]);

  // Drop the held pipeline once the selection moves away from it.
  const { gpu, source, model, quant } = selection;
  useEffect(() => {
    selectionRef.current = { gpu, source, model, quant };
    const current = heldRef.current;
    if (current !== undefined && !sameSpec(current.spec, selectionRef.current)) {
      void release();
    }
  }, [gpu, source, model, quant, release]);
  // Leaving the page disposes it.
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      void release();
    };
  }, [release]);

  const open = useCallback(
    async (
      spec: PipelineSpec,
      signal: AbortSignal,
      onProgress: (progress: AssetProgress) => void,
    ): Promise<OpenedPipeline> => {
      setReleaseError(undefined);
      const current = heldRef.current;
      if (current !== undefined && sameSpec(current.spec, spec)) {
        return { pipeline: current.pipeline, openMilliseconds: undefined };
      }
      // A failed dispose of the old pipeline is shown, and does not block opening the new one.
      await release();
      const started = performance.now();
      const pipeline = await Pipeline.fromPretrained(pretrainedRef(spec.source, MOSS_MOUNT), {
        gpu: spec.gpu,
        model: spec.model,
        quant: spec.quant,
        signal,
        onProgress,
      });
      // A cancel (or leaving the page) that lands as opening finishes: drop the pipeline rather
      // than keep one nobody will release.
      if (signal.aborted) {
        await disposeQueued(pipeline);
        throw signal.reason;
      }
      // The GPU can change while opening runs (a device loss sets it undefined; the selects are
      // locked during a run). A pipeline opened on the old device is not kept.
      if (!sameSpec(spec, selectionRef.current)) {
        await disposeQueued(pipeline);
        throw new Error(
          `The GPU changed while the pipeline (${spec.model} · ${spec.quant}) was opening; it was released without running. Press Run again once the GPU is ready.`,
        );
      }
      const openMilliseconds = performance.now() - started;
      const info: HeldPipelineInfo = {
        source: spec.source.kind,
        model: spec.model,
        quant: spec.quant,
        openSeconds: openMilliseconds / 1000,
      };
      heldRef.current = { spec, pipeline, info };
      setHeld(info);
      return { pipeline, openMilliseconds };
    },
    [release, disposeQueued],
  );

  return { held, releaseError, open, release };
};
