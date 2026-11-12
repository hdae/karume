/**
 * Builds a `Ti2vPipeline` for the chosen (source, preset, text encoder path, experimental fusion
 * rule) on the page's GPU, and rebuilds it when any of them changes (a Session takes its rules at
 * construction, so a different rule is a different pipeline). Construction creates no Session (it opens the containers
 * and reads the small assets), so building eagerly costs little and gives the page the prompt list.
 */

import type { GpuContext } from "@karume/runtime";
import { type Prompt, type RunComponent, Ti2vPipeline } from "@karume/models/wan";
import type { SessionDiagnostics } from "@karume/runtime";
import { type RefObject, useEffect, useState } from "react";

import type { TrackGpuWork } from "@/hooks/use-gpu";
import { type PipelineSource, pretrainedRef } from "@/lib/distributions";
import { experimentalRuleOptions, VAE_RMS_SILU_EXPERIMENT } from "@/lib/experimental-rule";

import type { TextEncoderPath } from "./settings";

export type PipelineState =
  | { readonly kind: "idle" }
  | { readonly kind: "loading"; readonly detail: string }
  | {
      readonly kind: "ready";
      readonly pipeline: Ti2vPipeline;
      readonly prompts: readonly Prompt[];
      readonly source: PipelineSource;
      readonly quant: string;
      readonly textEncoder: TextEncoderPath;
      /** The experimental rule the pipeline was opened with (`undefined` = none). */
      readonly fusionRule: string | undefined;
      readonly loadSeconds: number;
    }
  | {
      readonly kind: "error";
      readonly error: unknown;
      /** The inputs the failed build was for (a page tells a stale failure from the current one). */
      readonly source: PipelineSource;
      readonly quant: string;
      readonly textEncoder: TextEncoderPath;
      readonly fusionRule: string | undefined;
    };

export type DiagnosticsSink = (component: RunComponent, diagnostics: SessionDiagnostics) => void;

/**
 * Disposes a pipeline the page no longer uses: the one the cleanup drops, and a build that finished
 * after the cleanup aborted it. A failure (a flush on a lost device) has no page state left to show
 * it in, so it goes to the console. Resolves when the dispose has settled; never rejects.
 */
const release = (pipeline: Ti2vPipeline, track: TrackGpuWork): Promise<void> =>
  track(pipeline.dispose()).catch((error: unknown) => {
    console.error("Wan2.2: releasing the pipeline failed", error);
  });

export const useWanPipeline = (input: {
  readonly gpu: GpuContext | undefined;
  readonly source: PipelineSource;
  readonly quant: string | undefined;
  readonly textEncoder: TextEncoderPath;
  /** The experimental rule to open with (`experimentalRuleInUse` — `undefined` = none). */
  readonly fusionRule: string | undefined;
  /** `crossRepo` key for the umT5 mount (from the manifest); `gpu` path only. */
  readonly textEncoderRepo: string | undefined;
  /** Whether `/models/umt5/` is there; the `gpu` path fails early without it. */
  readonly umt5Directory: { readonly directory: string; readonly available: boolean } | undefined;
  /** Where `onRunDiagnostics` goes; the page points it at the running generation's record. */
  readonly diagnostics: RefObject<DiagnosticsSink | undefined>;
  /** The page GPU's `track`: the build and every dispose hold the device until they settle. */
  readonly track: TrackGpuWork;
}): PipelineState => {
  const {
    gpu,
    source,
    quant,
    textEncoder,
    fusionRule,
    textEncoderRepo,
    umt5Directory,
    diagnostics,
    track,
  } = input;
  const [state, setState] = useState<PipelineState>({ kind: "idle" });

  useEffect(() => {
    if (gpu === undefined || quant === undefined) {
      setState({ kind: "idle" });
      return undefined;
    }
    const controller = new AbortController();
    let built: Ti2vPipeline | undefined;
    const started = performance.now();
    setState({
      kind: "loading",
      detail: `opening ${quant} (${textEncoder} text encoder, ${source.kind} source, experimental rule ${fusionRule ?? "none"})`,
    });

    const build = async (): Promise<Ti2vPipeline> => {
      const needsUmt5 = textEncoder === "gpu" && textEncoderRepo !== undefined;
      if (needsUmt5 && umt5Directory?.available !== true) {
        throw new Error(
          `The GPU text encoder needs the umT5 distribution (${textEncoderRepo}), but ${
            umt5Directory?.directory ?? "its directory"
          } is missing — use the precomputed text encoder or add the distribution`,
        );
      }
      // The cached source reaches umT5 through the cross-repo reference of the manifest (the dev
      // server serves it on the same route); the direct source maps it to the umT5 mount.
      const ref = pretrainedRef(
        source,
        "wan22",
        needsUmt5 && textEncoderRepo !== undefined ? { [textEncoderRepo]: "umt5" } : {},
      );
      return await Ti2vPipeline.fromPretrained(ref, {
        gpu,
        quant,
        // MUST: name the path explicitly (the pipeline's default is "gpu").
        textEncoder,
        ...experimentalRuleOptions(VAE_RMS_SILU_EXPERIMENT, fusionRule),
        signal: controller.signal,
        onRunDiagnostics: (component, report) => diagnostics.current?.(component, report),
        onProgress: (progress) => {
          if (controller.signal.aborted) return;
          setState({
            kind: "loading",
            detail: `reading ${progress.path} (${(progress.loaded / 2 ** 20).toFixed(0)} / ${(
              progress.total /
              2 ** 20
            ).toFixed(0)} MiB)`,
          });
        },
      });
    };

    // Tracked with its dispose-on-abort, so the device outlives a build the cleanup abandoned.
    void track(
      build().then(async (pipeline) => {
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
          prompts: pipeline.prompts,
          source,
          quant,
          textEncoder,
          fusionRule,
          loadSeconds: (performance.now() - started) / 1000,
        });
      },
      (error: unknown) => {
        if (!controller.signal.aborted) {
          setState({ kind: "error", error, source, quant, textEncoder, fusionRule });
        }
      },
    );
    return () => {
      controller.abort();
      // The GPU belongs to the page; dispose only drops the pipeline (it waits for a running
      // generation first).
      if (built !== undefined) void release(built, track);
    };
  }, [
    gpu,
    source,
    quant,
    textEncoder,
    fusionRule,
    textEncoderRepo,
    umt5Directory,
    diagnostics,
    track,
  ]);

  return state;
};
