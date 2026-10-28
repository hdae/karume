/**
 * The page's `Pipeline`, built on the first Run for the chosen (GPU, source, model,
 * preset) and kept for later runs. Construction fetches the weights and creates the one Session (the
 * weight upload), so it waits for a Run instead of starting when the page opens; keeping it open
 * afterwards is the family's intended shape (one resident Session, many images).
 *
 * The pipeline belongs to a {@link PipelineSlot}, not to React state: the slot disposes it when the
 * settings move away, when the page is left, and when a build finishes for settings that are gone.
 * The wanted settings reach the slot in a layout effect, which runs in the same task as the commit,
 * so no build can finish between a settings change and the slot learning of it. Every dispose goes
 * through the page GPU's `track`, so the device is destroyed only after it (and the estimate it is
 * queued behind).
 */

import type { AssetProgress } from "@karume/hub";
import { Pipeline } from "@karume/models/depth-anything";
import type { GpuContext, SessionDiagnostics } from "@karume/runtime";
import {
  type RefObject,
  useCallback,
  useLayoutEffect,
  useMemo,
  useState,
  useSyncExternalStore,
} from "react";

import type { TrackGpuWork } from "@/hooks/use-gpu";
import { type PipelineSource, pretrainedRef } from "@/lib/distributions";
import { perRunSubmitCounts } from "@/lib/record";

import { createPipelineSlot, type Obtained, type SlotEntry } from "./pipeline-slot";

/** The settings a pipeline is built for. */
export type DepthPipelineKey = {
  readonly gpu: GpuContext;
  readonly source: PipelineSource;
  readonly model: string;
  readonly quant: string;
};

/** The open pipeline, the settings it was built for and its build time. */
export type OpenPipeline = SlotEntry<DepthPipelineKey, Pipeline>;

export type ObtainedPipeline = Obtained<DepthPipelineKey, Pipeline>;

export type DepthPipelineHandle = {
  /** The open pipeline (built for the current settings), or `undefined` before the first Run. */
  readonly open: OpenPipeline | undefined;
  /**
   * The open pipeline, or a new one built with `signal` (it stops the fetch, not the Session build).
   * Rejects when the settings are not ready, or change while it builds (the new pipeline is then
   * disposed, not kept).
   */
  readonly obtain: (
    signal: AbortSignal,
    onProgress: (progress: AssetProgress) => void,
  ) => Promise<ObtainedPipeline>;
};

/** Receives one run's Session diagnostics, with that run's submit and dispatch counts. */
export type DiagnosticsSink = (diagnostics: SessionDiagnostics) => void;

/** The source objects are stable until the choice changes, so identity compares them. */
const sameKey = (left: DepthPipelineKey, right: DepthPipelineKey): boolean =>
  left.gpu === right.gpu &&
  left.source === right.source &&
  left.model === right.model &&
  left.quant === right.quant;

export const useDepthPipeline = (input: {
  readonly gpu: GpuContext | undefined;
  readonly source: PipelineSource;
  readonly model: string;
  readonly quant: string;
  /** Where `onRunDiagnostics` goes; the page points it at the running estimate's record. */
  readonly diagnostics: RefObject<DiagnosticsSink | undefined>;
  /** The page GPU's `track` (stable for the page's life, so the slot keeps the first one). */
  readonly track: TrackGpuWork;
}): DepthPipelineHandle => {
  const { gpu, source, model, quant, diagnostics, track } = input;
  const [slot] = useState(() =>
    createPipelineSlot<DepthPipelineKey, Pipeline>({
      sameKey,
      now: () => performance.now(),
      // The slot drops the promise, so a failed dispose (a flush on a lost device) is logged here.
      track: (disposal) =>
        track(disposal).catch((error: unknown) => {
          console.error("Depth Anything: releasing the pipeline failed", error);
        }),
    }),
  );
  const key = useMemo<DepthPipelineKey | undefined>(
    () =>
      gpu === undefined || model === "" || quant === "" ? undefined : { gpu, source, model, quant },
    [gpu, source, model, quant],
  );

  // A pipeline built for other settings (or another GPU) is dropped at once, so two do not stay
  // resident; the next Run builds the one for the new settings. Leaving the page lets go of it.
  useLayoutEffect(() => {
    slot.want(key);
    return () => slot.want(undefined);
  }, [slot, key]);

  const open = useSyncExternalStore(slot.subscribe, slot.current);

  const obtain = useCallback(
    (signal: AbortSignal, onProgress: (progress: AssetProgress) => void) => {
      if (key === undefined)
        return Promise.reject(new Error("The GPU or the settings are not ready"));
      return slot.obtain(
        key,
        (buildSignal) => {
          // One per Session: the Session counts submits over its whole life.
          const perRun = perRunSubmitCounts();
          return Pipeline.fromPretrained(pretrainedRef(key.source, "depth-anything"), {
            gpu: key.gpu,
            model: key.model,
            quant: key.quant,
            signal: buildSignal,
            onProgress,
            onRunDiagnostics: (report) => {
              // MUST: count every report, sink or not (`?.()` would skip evaluating the argument).
              const run = perRun(report);
              diagnostics.current?.(run);
            },
          });
        },
        signal,
      );
    },
    [slot, key, diagnostics],
  );

  return { open, obtain };
};
