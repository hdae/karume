/**
 * Builds the Gemma 4 pipeline the page asked for (**Load**) on the page's GPU: `gemma.Pipeline` for
 * the `gemma4` and `gemma4-qat` mounts, `gemma4QatMobile.Pipeline` for `gemma4-qat-mobile`.
 * Construction reads every weight and
 * builds the Session (gigabytes), so it runs only on an explicit request, and again when the GPU is
 * re-acquired after a device loss. The previous pipeline is disposed when the request changes or the
 * page is left (dispose waits for a running turn); a dispose that fails is reported, not dropped.
 * Dropping the request (Cancel load) aborts the fetch through `fromPretrained`'s `signal`; the
 * construction after the fetch cannot be interrupted, so a pipeline that finishes after the abort is
 * disposed at once. The build and every dispose go through the page GPU's `track`, so the device is
 * destroyed only after them.
 */

import { type FromPretrainedOptions, Pipeline, type RunPhase } from "@karume/models/gemma";
import * as gemma4QatMobile from "@karume/models/gemma4-qat-mobile";
import type { GpuContext, SessionDiagnostics } from "@karume/runtime";
import { type RefObject, useEffect, useState } from "react";

import type { TrackGpuWork } from "@/hooks/use-gpu";
import { type PipelineSource, pretrainedRef } from "@/lib/distributions";

import { entryPointOf, type GemmaMount, isQatModel } from "./settings";

/** What **Load** asks for (a new object per press, so pressing again rebuilds). */
export type LoadRequest = {
  readonly mount: GemmaMount;
  readonly model: string;
  readonly quant: string;
  readonly source: PipelineSource;
  /** Build the MTP drafter (`speculative` of `fromPretrained`; the regular entry point only). */
  readonly speculative: boolean;
  /** Store the full-layer KV in f16 (`fullStateStorage` of `fromPretrained`); absent = f32. */
  readonly fullStateStorage?: "f16";
  /** Build the image embedder (`vision` of `fromPretrained`; the regular entry point only). */
  readonly vision?: boolean;
  /**
   * Pass `onRunDiagnostics`. It changes the decode path: the pipeline builds its on-GPU greedy
   * selection only without it, so with it a temperature-0 turn reads the whole logits row back.
   */
  readonly diagnostics: boolean;
};

export type GemmaPipeline = Pipeline | gemma4QatMobile.Pipeline;

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
 * Builds the pipeline a request names: `gemma.Pipeline` for the `gemma4` and `gemma4-qat` mounts,
 * `gemma4QatMobile.Pipeline` for `gemma4-qat-mobile` (no MTP drafter, models `e2b` / `e4b` only — refused
 * here, before any read). The one build path of the chat page and the decode bench, so both load a
 * model the same way. Without `onRunDiagnostics` the pipeline keeps its on-GPU greedy selection.
 * `sessionOptions` are passed to `fromPretrained` as explicit Session options (the pipeline resolves
 * them as explicit > the preset's declaration > the runtime default); the chat page passes none.
 * `pipelineOptions` are passed to `fromPretrained` as they are (the decode bench's PLE placement and
 * run-ahead depth); the chat page passes none.
 */
export const buildGemmaPipeline = async (input: {
  readonly gpu: GpuContext;
  readonly request: Omit<LoadRequest, "diagnostics">;
  readonly signal: AbortSignal;
  readonly onRunDiagnostics?: DiagnosticsSink;
  readonly onProgress?: FromPretrainedOptions["onProgress"];
  readonly sessionOptions?: Pick<
    FromPretrainedOptions,
    "integerGemv" | "packedStaticQuantize" | "linearGemvReduce" | "linearGemvPlanes"
  >;
  readonly pipelineOptions?: Pick<FromPretrainedOptions, "pleResidency" | "decodeAhead">;
}): Promise<GemmaPipeline> => {
  const { gpu, request, signal, onRunDiagnostics, onProgress, sessionOptions, pipelineOptions } =
    input;
  const ref = pretrainedRef(request.source, request.mount);
  const common = {
    ...sessionOptions,
    ...pipelineOptions,
    gpu,
    quant: request.quant,
    ...(request.fullStateStorage === undefined
      ? {}
      : { fullStateStorage: request.fullStateStorage }),
    signal,
    ...(onRunDiagnostics === undefined ? {} : { onRunDiagnostics }),
    ...(onProgress === undefined ? {} : { onProgress }),
  };
  if (entryPointOf(request.mount) === "gemma.Pipeline") {
    return await Pipeline.fromPretrained(ref, {
      ...common,
      model: request.model,
      ...(request.speculative ? { speculative: {} } : {}),
      ...(request.vision === true ? { vision: true } : {}),
    });
  }
  if (request.speculative) throw new Error("The mobile QAT distribution has no MTP drafter");
  if (request.vision === true) throw new Error("The mobile QAT distribution has no image embedder");
  const { model } = request;
  if (!isQatModel(model)) {
    throw new Error(`gemma4QatMobile.Pipeline accepts the models e2b and e4b, not '${model}'`);
  }
  return await gemma4QatMobile.Pipeline.fromPretrained(ref, { ...common, model });
};

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

    const build = (): Promise<GemmaPipeline> =>
      buildGemmaPipeline({
        gpu,
        request,
        signal: controller.signal,
        ...(request.diagnostics
          ? {
              onRunDiagnostics: (report: SessionDiagnostics, phase: RunPhase) =>
                diagnostics.current?.(report, phase),
            }
          : {}),
        onProgress: (progress) => {
          if (controller.signal.aborted) return;
          setState({
            kind: "loading",
            request,
            detail: `reading ${progress.path} (${(progress.loaded / MIB).toFixed(0)} / ${(
              progress.total / MIB
            ).toFixed(0)} MiB)`,
          });
        },
      });

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
