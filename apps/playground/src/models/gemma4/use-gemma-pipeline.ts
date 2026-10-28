/**
 * Builds the Gemma 4 pipeline the page asked for (**Load**) on the page's GPU: `gemma.Pipeline` for
 * the `gemma4` mount, `Gemma4QatPipeline` for `gemma4-qat`. Construction reads every weight and
 * builds the Session (gigabytes), so it runs only on an explicit request, and again when the GPU is
 * re-acquired after a device loss. The previous pipeline is disposed when the request changes or the
 * page is left (dispose waits for a running turn); a dispose that fails is reported, not dropped.
 * Dropping the request (Cancel load) aborts the fetch through `fromPretrained`'s `signal`; the
 * construction after the fetch cannot be interrupted, so a pipeline that finishes after the abort is
 * disposed at once. The build and every dispose go through the page GPU's `track`, so the device is
 * destroyed only after them.
 */

import { Pipeline, type RunPhase } from "@karume/models/gemma";
import { Gemma4QatPipeline } from "@karume/models/gemma4-qat";
import type { GpuContext, SessionDiagnostics } from "@karume/runtime";
import { type RefObject, useEffect, useState } from "react";

import type { TrackGpuWork } from "@/hooks/use-gpu";
import { type PipelineSource, pretrainedRef } from "@/lib/distributions";

import { type GemmaMount, isQatModel } from "./settings";

/** What **Load** asks for (a new object per press, so pressing again rebuilds). */
export type LoadRequest = {
  readonly mount: GemmaMount;
  readonly model: string;
  readonly quant: string;
  readonly source: PipelineSource;
  /** Build the MTP drafter (`speculative` of `fromPretrained`; regular Gemma 4 only). */
  readonly speculative: boolean;
  /**
   * Pass `onRunDiagnostics`. It changes the decode path: the pipeline builds its on-GPU greedy
   * selection only without it, so with it a temperature-0 turn reads the whole logits row back.
   */
  readonly diagnostics: boolean;
};

export type GemmaPipeline = Pipeline | Gemma4QatPipeline;

export type GemmaPipelineState =
  | { readonly kind: "idle" }
  | { readonly kind: "loading"; readonly request: LoadRequest; readonly detail: string }
  | {
      readonly kind: "ready";
      readonly request: LoadRequest;
      readonly pipeline: GemmaPipeline;
      readonly loadSeconds: number;
    }
  | { readonly kind: "error"; readonly request: LoadRequest; readonly error: unknown };

/** Where `onRunDiagnostics` goes (when the request asks for it); the page folds it per Session. */
export type DiagnosticsSink = (diagnostics: SessionDiagnostics, phase: RunPhase) => void;

const MIB = 2 ** 20;

/**
 * Disposes a pipeline the page no longer uses. A failure (a flush on a lost device, say) goes to the
 * console — the only place left once the page is gone — and to `report` for the page to show.
 * Resolves when the dispose has settled; never rejects.
 */
const release = (
  pipeline: GemmaPipeline,
  report: (error: unknown) => void,
  track: TrackGpuWork,
): Promise<void> =>
  track(pipeline.dispose()).catch((error: unknown) => {
    console.error("Releasing the Gemma 4 pipeline failed", error);
    report(error);
  });

export const useGemmaPipeline = (input: {
  readonly gpu: GpuContext | undefined;
  readonly request: LoadRequest | undefined;
  readonly diagnostics: RefObject<DiagnosticsSink | undefined>;
  readonly track: TrackGpuWork;
}): { readonly state: GemmaPipelineState; readonly releaseError: unknown } => {
  const { gpu, request, diagnostics, track } = input;
  const [state, setState] = useState<GemmaPipelineState>({ kind: "idle" });
  const [releaseError, setReleaseError] = useState<unknown>(undefined);

  useEffect(() => {
    setReleaseError(undefined);
    if (gpu === undefined || request === undefined) {
      setState({ kind: "idle" });
      return undefined;
    }
    const controller = new AbortController();
    let built: GemmaPipeline | undefined;
    const started = performance.now();
    setState({
      kind: "loading",
      request,
      detail: `opening ${request.model} ${request.quant} (${request.source.kind} source)`,
    });

    const build = async (): Promise<GemmaPipeline> => {
      const ref = pretrainedRef(request.source, request.mount);
      const common = {
        gpu,
        quant: request.quant,
        signal: controller.signal,
        ...(request.diagnostics
          ? {
              onRunDiagnostics: (report: SessionDiagnostics, phase: RunPhase) =>
                diagnostics.current?.(report, phase),
            }
          : {}),
        onProgress: (progress: { path: string; loaded: number; total: number }) => {
          if (controller.signal.aborted) return;
          setState({
            kind: "loading",
            request,
            detail: `reading ${progress.path} (${(progress.loaded / MIB).toFixed(0)} / ${(
              progress.total / MIB
            ).toFixed(0)} MiB)`,
          });
        },
      };
      if (request.mount === "gemma4") {
        return await Pipeline.fromPretrained(ref, {
          ...common,
          model: request.model,
          ...(request.speculative ? { speculative: {} } : {}),
        });
      }
      if (request.speculative) throw new Error("The QAT distribution has no MTP drafter");
      const { model } = request;
      if (!isQatModel(model)) {
        throw new Error(`Gemma4QatPipeline accepts the models e2b and e4b, not '${model}'`);
      }
      return await Gemma4QatPipeline.fromPretrained(ref, { ...common, model });
    };

    // Tracked with its dispose-on-abort, so the device outlives a build the cleanup abandoned.
    void track(
      build().then(async (pipeline) => {
        if (!controller.signal.aborted) return pipeline;
        await release(pipeline, setReleaseError, track);
        return undefined;
      }),
    ).then(
      (pipeline) => {
        if (pipeline === undefined) return;
        built = pipeline;
        setState({
          kind: "ready",
          request,
          pipeline,
          loadSeconds: (performance.now() - started) / 1000,
        });
      },
      (error: unknown) => {
        if (!controller.signal.aborted) setState({ kind: "error", request, error });
      },
    );
    return () => {
      controller.abort();
      // The GPU belongs to the page; dispose drops the pipeline only (and closes any sequence it
      // handed out, after a running turn).
      if (built !== undefined) void release(built, setReleaseError, track);
    };
  }, [gpu, request, diagnostics, track]);

  return { state, releaseError };
};
