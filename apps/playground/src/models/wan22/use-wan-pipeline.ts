/**
 * Builds a `WanTi2vPipeline` for the chosen (source, preset, text encoder path) on the page's GPU,
 * and rebuilds it when any of them changes. Construction creates no Session (it opens the containers
 * and reads the small assets), so building eagerly costs little and gives the page the prompt list.
 */

import type { GpuContext } from "@karume/runtime";
import { type WanPrompt, type WanRunComponent, WanTi2vPipeline } from "@karume/models/wan";
import type { SessionDiagnostics } from "@karume/runtime";
import { type RefObject, useEffect, useState } from "react";

import { type PipelineSource, pretrainedRef } from "@/lib/distributions";

import type { TextEncoderPath } from "./settings";

export type PipelineState =
  | { readonly kind: "idle" }
  | { readonly kind: "loading"; readonly detail: string }
  | {
      readonly kind: "ready";
      readonly pipeline: WanTi2vPipeline;
      readonly prompts: readonly WanPrompt[];
      readonly source: PipelineSource;
      readonly quant: string;
      readonly textEncoder: TextEncoderPath;
      readonly loadSeconds: number;
    }
  | {
      readonly kind: "error";
      readonly error: unknown;
      /** The inputs the failed build was for (a page tells a stale failure from the current one). */
      readonly source: PipelineSource;
      readonly quant: string;
      readonly textEncoder: TextEncoderPath;
    };

export type DiagnosticsSink = (component: WanRunComponent, diagnostics: SessionDiagnostics) => void;

export const useWanPipeline = (input: {
  readonly gpu: GpuContext | undefined;
  readonly source: PipelineSource;
  readonly quant: string | undefined;
  readonly textEncoder: TextEncoderPath;
  /** `crossRepo` key for the umT5 mount (from the manifest); `gpu` path only. */
  readonly textEncoderRepo: string | undefined;
  /** Whether `/models/umt5/` is there; the `gpu` path fails early without it. */
  readonly umt5Directory: { readonly directory: string; readonly available: boolean } | undefined;
  /** Where `onRunDiagnostics` goes; the page points it at the running generation's record. */
  readonly diagnostics: RefObject<DiagnosticsSink | undefined>;
}): PipelineState => {
  const { gpu, source, quant, textEncoder, textEncoderRepo, umt5Directory, diagnostics } = input;
  const [state, setState] = useState<PipelineState>({ kind: "idle" });

  useEffect(() => {
    if (gpu === undefined || quant === undefined) {
      setState({ kind: "idle" });
      return undefined;
    }
    const controller = new AbortController();
    let built: WanTi2vPipeline | undefined;
    const started = performance.now();
    setState({
      kind: "loading",
      detail: `opening ${quant} (${textEncoder} text encoder, ${source.kind} source)`,
    });

    const build = async (): Promise<WanTi2vPipeline> => {
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
      return await WanTi2vPipeline.fromPretrained(ref, {
        gpu,
        quant,
        // MUST: name the path explicitly (the pipeline's default is "gpu").
        textEncoder,
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

    build().then(
      (pipeline) => {
        if (controller.signal.aborted) {
          void pipeline.dispose();
          return;
        }
        built = pipeline;
        setState({
          kind: "ready",
          pipeline,
          prompts: pipeline.prompts,
          source,
          quant,
          textEncoder,
          loadSeconds: (performance.now() - started) / 1000,
        });
      },
      (error: unknown) => {
        if (!controller.signal.aborted) {
          setState({ kind: "error", error, source, quant, textEncoder });
        }
      },
    );
    return () => {
      controller.abort();
      // The GPU belongs to the page; dispose only drops the pipeline (it waits for a running
      // generation first).
      void built?.dispose();
    };
  }, [gpu, source, quant, textEncoder, textEncoderRepo, umt5Directory, diagnostics]);

  return state;
};
