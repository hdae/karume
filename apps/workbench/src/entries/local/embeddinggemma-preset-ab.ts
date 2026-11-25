/**
 * The speed of two presets of the EmbeddingGemma 2 mirror against each other, through the public
 * pipeline (`embeddinggemma.Pipeline.fromPretrained` of each preset on one device): the wall-clock
 * A/B of `@karume/runtime/tune` (`runSessionAb`, judged by `judgeSessionAb` with identity not
 * required — the presets store the weights differently, so their vectors differ by design) of one
 * `embed` of one text. The text is a prefix of a golden case of the recipe cut to exactly the
 * asked length in token ids (`Pipeline.tokenize`); one run measures one length, so a long text does
 * not heat the device for a short one.
 *
 * Before the A/B each preset is probed once — a pipeline of its own built with the diagnostics
 * callback, the text embedded twice, disposed — for the load's breakdown, the resident weights and
 * the plan backing at that length. The arms are built without the callback and both stay loaded
 * for the whole A/B (one Session resident per pipeline, as the family runs). The device is acquired
 * as the pipeline acquires its own (no feature is requested: the presets declare none).
 *
 * No verdict: it only measures. The record cannot see the GPU's clock or a power or thermal limit;
 * a reading of them belongs next to the run (the repository's GPU diagnostics scripts).
 */

import { Pipeline } from "@karume/models/embeddinggemma";
import { acquireGpu } from "@karume/runtime";

import { type AcquirePresetAbDevice, runPresetAb } from "../../embeddinggemma-preset-ab/run.ts";
import { LONGEST_CASE_TOKENS } from "../../embeddinggemma-preset-ab/text.ts";
import type { Entry } from "../../entry.ts";
import { sourceParameter } from "../../local-parameters.ts";
import { MIN_GAIN_PARAMETER } from "../../session-ab/shared.ts";

/** The device: `acquireGpu` with the host's options, each pipeline `Pipeline.fromPretrained` on it. */
const acquireDevice: AcquirePresetAbDevice = async (options) => {
  const gpu = await acquireGpu({ ...options });
  return {
    adapterInfo: gpu.adapterInfo,
    pipeline: (ref, pipelineOptions) => Pipeline.fromPretrained(ref, { gpu, ...pipelineOptions }),
    destroy: () => gpu.destroy(),
  };
};

const entry: Entry = {
  id: "embeddinggemma-preset-ab",
  title: "EmbeddingGemma preset A/B",
  description:
    "Wall-clock A/B (runSessionAb, judgeSessionAb without identity) of two presets of the EmbeddingGemma 2 mirror through the public pipeline: one embed of a golden text cut to the asked token length, both pipelines resident, each preset probed once first for its load, resident weights and plan backing. No verdict.",
  parameters: [
    {
      kind: "text",
      name: "base",
      label: "Base preset (empty: the manifest's default preset)",
      default: "",
    },
    {
      kind: "text",
      name: "compared",
      label: "Compared preset",
      default: "i8",
    },
    {
      kind: "integer",
      name: "tokens",
      label: "Token ids of the text, <bos> and <eos> included (a prefix of a golden case)",
      min: 2,
      max: LONGEST_CASE_TOKENS,
      default: 512,
    },
    {
      kind: "integer",
      name: "rounds",
      label: "Timed rounds (one untimed warm-up round comes first)",
      min: 3,
      max: 1000,
      default: 9,
    },
    MIN_GAIN_PARAMETER,
    sourceParameter("How a browser reads the distribution"),
  ],
  run: (context) => runPresetAb(context, acquireDevice),
};

export default entry;
