/**
 * The QAT pipeline of `@karume/models/gemma4-qat`.
 *
 * It has its own module so that the class is declared under its public name `Pipeline` and its
 * signatures reference the QAT option types by the names users import (`./qat.ts`), with no import
 * alias in between — TypeScript and JSR print the declared and written names.
 */
import type { DistributionSource, HubRepoRef } from "@karume/hub";
import { type Assets, type Gemma4State, GemmaPipeline } from "./pipeline.ts";
import type { FromPretrainedOptions, PipelineOptions } from "./qat.ts";

/**
 * Fixed QAT E2B/E4B at the experimental stage. Full CPU/GPU bit agreement and broad quality are not
 * guaranteed (ADR 0097).
 *
 * The generation and conversation surface is shared with regular Gemma's `Pipeline`
 * (`@karume/models/gemma`, `gemma.Pipeline` through the barrel), but it accepts only its own
 * distribution, with fixed INT2/INT4/INT8 and SRQ, and no drafter.
 */
export class Pipeline extends GemmaPipeline {
  private constructor(state: Gemma4State) {
    super(state);
  }
  /**
   * Builds E2B/E4B from an explicitly given QAT distribution. When model is omitted, the manifest's
   * default is used.
   */
  static async fromPretrained(
    ref: string | HubRepoRef | DistributionSource,
    options: FromPretrainedOptions = {},
  ): Promise<Pipeline> {
    return new Pipeline(await super.loadPretrained("gemma4-qat", ref, options));
  }
  /**
   * Checks and builds from fetched fixed-QAT assets. The model is determined from the graph and the
   * PLE configuration.
   */
  static async fromAssets(input: Assets, options: PipelineOptions = {}): Promise<Pipeline> {
    return new Pipeline(await super.loadAssets("gemma4-qat", input, options));
  }
}
