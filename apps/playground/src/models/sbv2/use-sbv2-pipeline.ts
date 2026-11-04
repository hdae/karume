/**
 * Builds a `Pipeline` for the chosen (source, model, preset) on the page's GPU and rebuilds it
 * when any of them changes. Construction reads the manifest, the components' descriptors and the
 * small assets, and with the cached source downloads every weight part into Cache Storage (the
 * first build of a model / preset fetches all of its weights; the direct source has no such step).
 * It creates no Session: every `generate` creates the three Sessions one after another, reading
 * their weights from the source (the cache the build filled, or plain fetch), and destroys them
 * again.
 */

import { type RunComponent, Pipeline } from "@karume/models/sbv2";
import type { GpuContext, SessionDiagnostics } from "@karume/runtime";
import { type RefObject, useEffect, useState } from "react";

import type { TrackGpuWork } from "@/hooks/use-gpu";
import { type PipelineSource, pretrainedRef } from "@/lib/distributions";

export type Sbv2PipelineState =
  | { readonly kind: "idle" }
  | { readonly kind: "loading"; readonly detail: string }
  | {
      readonly kind: "ready";
      readonly pipeline: Pipeline;
      /** The GPU the pipeline was built on (the page runs it only while this is still its GPU). */
      readonly gpu: GpuContext;
      readonly source: PipelineSource;
      readonly model: string;
      readonly quant: string;
      readonly loadSeconds: number;
    }
  | { readonly kind: "error"; readonly error: unknown };

export type DiagnosticsSink = (component: RunComponent, diagnostics: SessionDiagnostics) => void;

const MIB = 2 ** 20;

/**
 * Disposes a pipeline the page no longer uses: the one the cleanup drops, and a build that finished
 * after the cleanup aborted it. A failure (a flush on a lost device) has no page state left to show
 * it in, so it goes to the console. Resolves when the dispose has settled; never rejects.
 */
const release = (pipeline: Pipeline, track: TrackGpuWork): Promise<void> =>
  track(pipeline.dispose()).catch((error: unknown) => {
    console.error("Style-Bert-VITS2: releasing the pipeline failed", error);
  });

export const useSbv2Pipeline = (input: {
  readonly gpu: GpuContext | undefined;
  readonly source: PipelineSource;
  /** `undefined` until the manifest is known (nothing is built then). */
  readonly model: string | undefined;
  readonly quant: string | undefined;
  /** Where `onRunDiagnostics` goes; the page points it at the running synthesis. */
  readonly diagnostics: RefObject<DiagnosticsSink | undefined>;
  /** The page GPU's `track`: the build and every dispose hold the device until they settle. */
  readonly track: TrackGpuWork;
}): Sbv2PipelineState => {
  const { gpu, source, model, quant, diagnostics, track } = input;
  const [state, setState] = useState<Sbv2PipelineState>({ kind: "idle" });

  useEffect(() => {
    if (gpu === undefined || model === undefined || quant === undefined) {
      setState({ kind: "idle" });
      return undefined;
    }
    const controller = new AbortController();
    let built: Pipeline | undefined;
    const started = performance.now();
    setState({ kind: "loading", detail: `opening ${model} / ${quant} (${source.kind} source)` });

    const build = Pipeline.fromPretrained(pretrainedRef(source, "sbv2"), {
      gpu,
      model,
      quant,
      signal: controller.signal,
      onRunDiagnostics: (component, report) => diagnostics.current?.(component, report),
      onProgress: (progress) => {
        if (controller.signal.aborted) return;
        // `loaded` / `total` count every file of the build; `path` is only the latest one.
        setState({
          kind: "loading",
          detail: `fetching ${(progress.loaded / MIB).toFixed(1)} / ${(
            progress.total / MIB
          ).toFixed(1)} MiB in total · ${progress.path}`,
        });
      },
    });
    // Tracked with its dispose-on-abort, so the device outlives a build the cleanup abandoned.
    void track(
      build.then(async (pipeline) => {
        if (controller.signal.aborted) {
          await release(pipeline, track);
          return undefined;
        }
        return pipeline;
      }),
    ).then(
      (pipeline) => {
        if (pipeline === undefined) return;
        built = pipeline;
        setState({
          kind: "ready",
          pipeline,
          gpu,
          source,
          model,
          quant,
          loadSeconds: (performance.now() - started) / 1000,
        });
      },
      (error: unknown) => {
        if (!controller.signal.aborted) setState({ kind: "error", error });
      },
    );
    return () => {
      controller.abort();
      // The GPU belongs to the page: dispose drops only the pipeline, after a running synthesis.
      if (built !== undefined) void release(built, track);
    };
  }, [gpu, source, model, quant, diagnostics, track]);

  return state;
};
