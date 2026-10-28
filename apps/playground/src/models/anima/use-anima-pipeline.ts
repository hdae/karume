/**
 * The page's one `AnimaPipeline`, opened on Run and kept for the next run while the source, model,
 * preset and GPU stay the same. Opening reads the manifest, the tokenizers and the container
 * descriptors and, on the cached route, stores the weights in the hub's cache — a few GiB for the
 * `bf16` preset — so the page does not open one per select change (the Wan2.2 page's eager build
 * would download every preset the user passes over). Keeping it also lets the opt-in DiT residency
 * carry the transformer Session from one run to the next.
 *
 * The held pipeline is disposed when the selection moves away from it, on **Release pipeline** and
 * when the page is left. The GPU belongs to the page: `dispose` only drops the pipeline (and a
 * resident DiT). Disposals run one after another; `release` resolves once every disposal started so
 * far has settled and never rejects — a failed dispose (a lost device) lands in `releaseError`.
 * Every disposal goes through the page GPU's `track`, so useGpu destroys the device only after it.
 */

import type { AssetProgress } from "@karume/hub";
import {
  AnimaPipeline,
  type AnimaResidencyAction,
  type AnimaRunComponent,
} from "@karume/models/anima";
import type { GpuContext, SessionDiagnostics } from "@karume/runtime";
import { type RefObject, useCallback, useEffect, useRef, useState } from "react";

import type { TrackGpuWork } from "@/hooks/use-gpu";
import { type PipelineSource, pretrainedRef } from "@/lib/distributions";

import { residentAfter } from "./record";
import type { AnimaMount } from "./settings";

export type DiagnosticsSink = (
  component: AnimaRunComponent,
  diagnostics: SessionDiagnostics,
) => void;

/** What a pipeline is opened for. */
export type PipelineSpec = {
  readonly gpu: GpuContext;
  readonly source: PipelineSource;
  readonly mount: AnimaMount;
  readonly model: string;
  readonly quant: string;
};

/** What the status card shows of the held pipeline. */
export type HeldPipelineInfo = {
  readonly mount: AnimaMount;
  readonly model: string;
  readonly quant: string;
  readonly sourceKind: PipelineSource["kind"];
  readonly openSeconds: number;
  /** A DiT kept by `residency: "transformer"` sits on the device. */
  readonly residentDit: boolean;
};

type Held = {
  readonly spec: PipelineSpec;
  readonly pipeline: AnimaPipeline;
  readonly info: HeldPipelineInfo;
};

/** `source` is memoized by `usePipelineSource`, so identity tells a changed source (or cache). */
const sameSpec = (left: PipelineSpec, right: Partial<PipelineSpec>): boolean =>
  left.gpu === right.gpu &&
  left.source === right.source &&
  left.mount === right.mount &&
  left.model === right.model &&
  left.quant === right.quant;

export type OpenedPipeline = {
  readonly pipeline: AnimaPipeline;
  /** How long opening took; `undefined` when the held pipeline was reused. */
  readonly openMilliseconds: number | undefined;
  /** The pipeline held a DiT kept resident by an earlier run when it was handed out. */
  readonly residentAtStart: boolean;
  /** Feeds a `residency` event of a generation back, so the held state follows the DiT slot. */
  readonly noteResidency: (action: AnimaResidencyAction) => void;
};

export const useAnimaPipeline = (
  selection: Partial<PipelineSpec>,
  diagnostics: RefObject<DiagnosticsSink | undefined>,
  track: TrackGpuWork,
): {
  readonly held: HeldPipelineInfo | undefined;
  /** The latest failed dispose (cleared when the next Run opens or reuses a pipeline). */
  readonly releaseError: unknown;
  /**
   * The held pipeline when it was opened for `spec`, else a new one (the held one is disposed
   * first, so two never coexist on the device). `crossRepo` maps the repositories the manifest
   * references to mounts (direct route only).
   */
  readonly open: (
    spec: PipelineSpec,
    crossRepo: Readonly<Record<string, AnimaMount>>,
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
    (pipeline: AnimaPipeline | undefined): Promise<void> => {
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
  const { gpu, source, mount, model, quant } = selection;
  useEffect(() => {
    selectionRef.current = { gpu, source, mount, model, quant };
    const current = heldRef.current;
    if (current !== undefined && !sameSpec(current.spec, selectionRef.current)) {
      void release();
    }
  }, [gpu, source, mount, model, quant, release]);
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
      crossRepo: Readonly<Record<string, AnimaMount>>,
      signal: AbortSignal,
      onProgress: (progress: AssetProgress) => void,
    ): Promise<OpenedPipeline> => {
      setReleaseError(undefined);
      const handOut = (pipeline: AnimaPipeline, openMilliseconds: number | undefined) => ({
        pipeline,
        openMilliseconds,
        residentAtStart: heldRef.current?.info.residentDit ?? false,
        noteResidency: (action: AnimaResidencyAction): void => {
          const current = heldRef.current;
          if (current?.pipeline !== pipeline) return;
          const info = { ...current.info, residentDit: residentAfter(action) };
          heldRef.current = { ...current, info };
          if (mountedRef.current) setHeld(info);
        },
      });
      const current = heldRef.current;
      if (current !== undefined && sameSpec(current.spec, spec)) {
        return handOut(current.pipeline, undefined);
      }
      // A failed dispose of the old pipeline is shown, and does not block opening the new one.
      await release();
      const started = performance.now();
      const pipeline = await AnimaPipeline.fromPretrained(
        pretrainedRef(spec.source, spec.mount, crossRepo),
        {
          gpu: spec.gpu,
          model: spec.model,
          quant: spec.quant,
          signal,
          onRunDiagnostics: (component, report) => diagnostics.current?.(component, report),
          onProgress,
        },
      );
      // A cancel (or leaving the page) that lands as opening finishes: drop the pipeline rather
      // than keep one nobody will release.
      if (signal.aborted) {
        await disposeQueued(pipeline);
        throw signal.reason;
      }
      // The GPU can change while a long cached download runs (a device loss sets it undefined;
      // the selects are locked during a run). A pipeline opened on the old device is not kept.
      if (!sameSpec(spec, selectionRef.current)) {
        await disposeQueued(pipeline);
        throw new Error(
          `The GPU changed while the pipeline (${spec.model} · ${spec.quant}) was opening; it was released without running. Press Run again once the GPU is ready.`,
        );
      }
      const openMilliseconds = performance.now() - started;
      const info: HeldPipelineInfo = {
        mount: spec.mount,
        model: spec.model,
        quant: spec.quant,
        sourceKind: spec.source.kind,
        openSeconds: openMilliseconds / 1000,
        residentDit: false,
      };
      heldRef.current = { spec, pipeline, info };
      setHeld(info);
      return handOut(pipeline, openMilliseconds);
    },
    [release, disposeQueued, diagnostics],
  );

  return { held, releaseError, open, release };
};
