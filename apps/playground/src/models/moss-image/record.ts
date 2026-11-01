/**
 * The MOSS-Image page's run record: the assembly of the saved JSON from what one run observed. Pure
 * (the page passes the environment and the clock readings in), so `vp test` covers it.
 *
 * NOTE: `generate` reports no stage events and no Session diagnostics, so unlike the Anima record
 * there are no per-stage seconds, no timeline and no `diagnostics` — only the wall time of opening
 * and of `generate` as a whole.
 */

import type { RecordEnvironment } from "@/lib/environment";
import { seconds } from "@/lib/record";

import { type EffectiveRequest, MOSS_MOUNT, type MossRequestParts } from "./settings";

/** The version of the saved record; bump it when a field changes meaning or goes away. */
export const RECORD_FORMAT = "karume-playground-moss-image/1";

/** The saved record of one run (JSON; the page shows a summary of it). */
export type MossImageRecord = RecordEnvironment & {
  readonly format: typeof RECORD_FORMAT;
  readonly page: "moss-image";
  readonly distribution: {
    readonly mount: typeof MOSS_MOUNT;
    readonly directory: string;
    readonly manifestSha256: string;
  };
  /** Always plain fetch from `/models/`: the distribution is not published, so it has no `/hf/` route. */
  readonly source: { readonly kind: "direct" };
  readonly model: { readonly name: string; readonly isDefault: boolean };
  readonly preset: { readonly quant: string; readonly label: string; readonly isDefault: boolean };
  /** The fields sent to `generate` (a field left out takes the manifest's default). */
  readonly request: MossRequestParts;
  /** The values the run used once the manifest's defaults filled the rest. */
  readonly effective: EffectiveRequest;
  /**
   * `reused`: the pipeline opened by an earlier run (same model, preset and GPU) ran this one;
   * `openSeconds` is then null. Opening reads the manifest, the container descriptors and the
   * tokenizer (no Session is created).
   */
  readonly pipeline: { readonly reused: boolean; readonly openSeconds: number | null };
  /**
   * `generate` alone: the text encoder, the text refiner, the DiT and the VAE, each opened (weights
   * uploaded) and closed inside it.
   */
  readonly generateSeconds: number;
  readonly output: {
    readonly width: number;
    readonly height: number;
    /** sha256 of the RGBA bytes of `imageToRgba` (alpha is always 255). */
    readonly rgbaSha256: string;
  };
};

/** What one finished run observed (the page collects it; the clock readings are `performance.now()`). */
export type RunObservation = {
  readonly environment: RecordEnvironment;
  readonly directory: string;
  readonly manifestSha256: string;
  readonly model: string;
  readonly defaultModel: string;
  readonly quant: string;
  readonly quantLabel: string;
  readonly defaultQuant: string;
  readonly request: MossRequestParts;
  readonly effective: EffectiveRequest;
  readonly openMilliseconds: number | undefined;
  readonly generateMilliseconds: number;
  readonly output: { readonly width: number; readonly height: number; readonly sha256: string };
};

export const assembleRecord = (run: RunObservation): MossImageRecord => ({
  format: RECORD_FORMAT,
  page: "moss-image",
  ...run.environment,
  distribution: {
    mount: MOSS_MOUNT,
    directory: run.directory,
    manifestSha256: run.manifestSha256,
  },
  source: { kind: "direct" },
  model: { name: run.model, isDefault: run.model === run.defaultModel },
  preset: { quant: run.quant, label: run.quantLabel, isDefault: run.quant === run.defaultQuant },
  request: run.request,
  effective: run.effective,
  pipeline: {
    reused: run.openMilliseconds === undefined,
    openSeconds: run.openMilliseconds === undefined ? null : seconds(run.openMilliseconds),
  },
  generateSeconds: seconds(run.generateMilliseconds),
  output: { width: run.output.width, height: run.output.height, rgbaSha256: run.output.sha256 },
});
