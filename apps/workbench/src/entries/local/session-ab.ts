/**
 * The session A/B on a Session target: the wall-clock A/B of `@karume/runtime/tune`
 * (`runSessionAb`, judged by `judgeSessionAb`) of one component graph with and without the fusion
 * rule its pipeline passes by default — the MOSS-Image DiT forward (`models/ew_chain`), the Wan2.2
 * VAE next chunk or the anima VAE tile decode (`models/vae_rms_silu`). The component of the
 * manifest's default model at the chosen preset is fetched and opened once; the two arms are
 * `sessionArm`s over it, `base` with the options the pipeline gives that Session besides its rule,
 * `rule` with the same plus the rule, and every visit opens and closes its arm's Session (two of
 * them never live together). A run is one forward on seeded feeds; the runner compares every output
 * word (u32, NaN by class) with the base's first run and keeps the counts every run reports.
 *
 * The verdict is identity; the judgement (adopt, reject, inconclusive) is a measurement and goes to
 * the summary and the data.
 */

import { loadManifest } from "@karume/hub";
import { EW_CHAIN_RULE_NAME } from "@karume/models/moss";
import { acquireGpu } from "@karume/runtime";
import { judgeSessionAb, runSessionAb, sessionArm } from "@karume/runtime/tune";

import { choiceValue, type Entry, integerValue, numberValue, textValue } from "../../entry.ts";
import {
  COMPONENT_TARGETS,
  componentArmOptions,
  componentFeeds,
  componentRequest,
  targetMount,
} from "../../session-ab/component-target.ts";
import { openComponentModel } from "../../session-ab/component-model.ts";
import { MODELS_VERSION, sessionAbMossData, sessionAbVaeData } from "../../session-ab/data.ts";
import { steppedArm } from "../../session-ab/moss.ts";
import { armStepper, fetchLine } from "../../session-ab/progress.ts";
import { SESSION_AB_DEFAULTS, SESSION_AB_LIMITS, SOURCE_KINDS } from "../../session-ab/request.ts";
import { runnerEnvironment, runPositions } from "../../session-ab/run.ts";
import { sessionAbComponentResult } from "../../session-ab/summary.ts";

const entry: Entry = {
  id: "session-ab",
  title: "Session A/B",
  description:
    "Wall-clock A/B (runSessionAb, judgeSessionAb) of one component forward with and without the fusion rule its pipeline passes by default: the MOSS-Image DiT (models/ew_chain), the Wan2.2 VAE next chunk or the anima VAE tile (models/vae_rms_silu). The verdict is output identity; the judgement is in the summary.",
  readsLocalDistribution: true,
  parameters: [
    {
      kind: "choice",
      name: "target",
      label: "Target",
      options: COMPONENT_TARGETS,
      default: "anima-vae-decode",
    },
    {
      kind: "text",
      name: "preset",
      label: "Preset (empty: the manifest's default preset of the target's model)",
      default: "",
    },
    {
      kind: "integer",
      name: "rounds",
      label: "Timed rounds (one untimed warm-up round comes first)",
      min: SESSION_AB_LIMITS.minRounds,
      max: SESSION_AB_LIMITS.maxRounds,
      default: SESSION_AB_DEFAULTS.rounds,
    },
    {
      kind: "number",
      name: "minGain",
      label: "Minimum gain (a fraction of the base's time: 0.01 = 1%)",
      min: SESSION_AB_LIMITS.minGain,
      max: SESSION_AB_LIMITS.maxGain,
      default: SESSION_AB_DEFAULTS.minGain,
    },
    {
      kind: "choice",
      name: "source",
      label: "How a browser reads the distribution",
      options: SOURCE_KINDS,
      default: "cached",
    },
  ],
  run: async ({ parameters, gpuOptions, signal, progress, openDistribution, backend }) => {
    const target = choiceValue(parameters, "target", COMPONENT_TARGETS);
    const opened = await openDistribution(
      targetMount(target),
      choiceValue(parameters, "source", SOURCE_KINDS),
    );
    const { manifest } = await loadManifest(opened.ref, { signal });
    const request = componentRequest(target, manifest, textValue(parameters, "preset"), {
      rounds: integerValue(parameters, "rounds"),
      minGain: numberValue(parameters, "minGain"),
    });
    // Subgroups when the device can run the 32-lane form (the MOSS-Image DiT's online attention
    // takes its subgroup form with them, its portable form without).
    const gpu = await acquireGpu({ ...gpuOptions, subgroups: "available" });
    try {
      const { component } = request;
      progress(`fetching and opening the ${component} component`);
      const model = await openComponentModel(
        opened.ref,
        request.model,
        request.quant,
        component,
        signal,
        (fetched) => progress(fetchLine(component, fetched)),
      );
      const feeds = componentFeeds(request, model.prepared.graph.inputs);
      const onStep = armStepper({
        positions: runPositions(request.arms.length, request.rounds, request.openPerVisit),
        opens: "Session",
        signal,
        progress,
      });
      const arms = componentArmOptions(request).map(({ name, options }) =>
        steppedArm(sessionArm({ name, prepared: model.prepared, gpu, options, feeds }), onStep),
      );
      const record = await runSessionAb({
        arms,
        rounds: request.rounds,
        openPerVisit: request.openPerVisit,
        environment: runnerEnvironment(gpu.adapterInfo, backend),
      });
      const judgements = judgeSessionAb(record, { minGain: request.minGain });
      const common = {
        distribution: opened.facts,
        container: { component, part0Sha256: model.part0Sha256 },
        modelsVersion: MODELS_VERSION,
        record,
        judgements,
      };
      return sessionAbComponentResult(
        request.target === "moss-dit-forward"
          ? sessionAbMossData({ ...common, request, ruleName: EW_CHAIN_RULE_NAME })
          : sessionAbVaeData({ ...common, request }),
      );
    } finally {
      gpu.destroy();
    }
  },
};

export default entry;
