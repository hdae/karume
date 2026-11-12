/**
 * The session A/B: the wall-clock A/B of `@karume/runtime/tune` (`runSessionAb`, `judgeSessionAb`)
 * with and without a fusion rule, on one of two targets.
 *
 * - **Gemma 4 decode**: two arms load the same model and preset — `base` with no rule, `rule` with
 *   `candidate/gelu_mul` of `@karume/candidate-rules` through the gemma pipeline's experimental
 *   `fusionRules` — and every timed run is one greedy generation from a fixed prompt whose token ids
 *   the runner compares, so identity is token identity. The arms are built without the diagnostics
 *   callback (every run is the production path); the rule's hits and the dispatches come from a
 *   count probe of each arm before the A/B.
 * - **MOSS-Image DiT forward**: the `dit` component of the chosen preset, fetched and opened once
 *   through the hub, and two `sessionArm`s over it — `base` with the preset's knobs, `rule` with the
 *   same plus `models/ew_chain` of `@karume/models/moss` (the rule the MOSS-Image pipeline passes by
 *   default) — each visit opening and closing its Session (two DiT Sessions never live together). Every timed run is one forward at the 1024 × 1024 sequence (S = 4,159) on
 *   seeded feeds whose output words the runner compares (u32, NaN by class); the hits and the
 *   dispatches are the runner's, which every run of a `sessionArm` reports.
 * - **Wan2.2 VAE next chunk** and **anima VAE decode**: the `vae_decoder_next` chunk graph of Wan2.2
 *   or the `vae_decoder` tile graph of anima at the chosen preset, opened the same way, and three
 *   `sessionArm`s over it — `base` with the options the pipeline gives its VAE, `g8` and `g4` with
 *   the same plus `candidate/vae_rms_silu_g8` / `_g4` — each visit opening and closing its Session.
 *   Every timed run is one forward on seeded feeds (the Wan chunk graph's caches included) and reads
 *   every output back; the runner compares every output's words and keeps the counts.
 *
 * The page shows where the runner is, then per arm the identity, the hits and dispatches and the
 * per-round times, and the judgement with its numbers and its reason; the record is saved as JSON.
 */

import { GELU_MUL_RULE, GELU_MUL_RULE_NAME } from "@karume/candidate-rules";
import { chatPrompt } from "@karume/models/gemma";
import { EW_CHAIN_RULE, EW_CHAIN_RULE_NAME } from "@karume/models/moss";
import type { SessionOptions } from "@karume/runtime";
import { judgeSessionAb, runSessionAb, sessionArm } from "@karume/runtime/tune";
import { memo, useMemo, useState } from "react";
import { flushSync } from "react-dom";

import { type Choice, ChoiceSelect } from "@/components/choice-select";
import { ErrorAlert } from "@/components/error-alert";
import { Field } from "@/components/field";
import { KeyValueTable } from "@/components/key-value-table";
import { MountBadges } from "@/components/mount-badges";
import { PresetSelect } from "@/components/preset-select";
import { RecordActions } from "@/components/record-actions";
import { RunStatusCard } from "@/components/run-status-card";
import { SourcePicker } from "@/components/source-picker";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useDistributionIndex } from "@/hooks/use-distribution-index";
import { type GpuOptions, useGpu } from "@/hooks/use-gpu";
import { type ManifestState, useManifest } from "@/hooks/use-manifest";
import { usePipelineSource } from "@/hooks/use-pipeline-source";
import { type RunProgress, useRunTask } from "@/hooks/use-run-task";
import { type DistributionStatus, mountStatus, pretrainedRef } from "@/lib/distributions";
import { fileStamp } from "@/lib/download";
import { recordEnvironment } from "@/lib/environment";
import { manifestPresets } from "@/lib/presets";
import { sourceRecord } from "@/lib/record";
import {
  EXPERT_CACHE_FITTED,
  EXPERT_HOST_TIER_ALL,
  GEMMA_MOUNTS,
  type GemmaManifests,
  type GemmaMount,
  gemmaModelChoices,
  hasExpertCache,
  type ModelKey,
  effectiveModelKey,
  modelKeyValue,
  parseExpertCacheSlots,
  parseExpertHostTier,
  parseExpertSpeculationDepth,
  parseModelKey,
} from "@/models/gemma4/settings";
import { buildGemmaPipeline, type GemmaPipeline } from "@/models/gemma4/use-gemma-pipeline";
import { MOSS_MOUNT } from "@/models/moss-image/settings";

import {
  MOSS_DIT_WORK,
  mossAbModel,
  mossDitFeeds,
  type SessionAbMossModel,
  type SessionAbMossRequest,
  sessionAbMossRequest,
  steppedArm,
} from "./moss";
import { openComponentModel } from "./component-model";
import {
  buildSessionAbMossPageRecord,
  buildSessionAbPageRecord,
  PACKAGE_VERSIONS,
  type SessionAbArmProbe,
  type SessionAbMossPageRecord,
  type SessionAbPageRecord,
  type SessionAbVaePageRecord,
  buildSessionAbVaePageRecord,
} from "./record";
import {
  INITIAL_SESSION_AB_FORM,
  parseSessionAbForm,
  SESSION_AB_BACKENDS,
  SESSION_AB_CAPACITY,
  SESSION_AB_LIMITS,
  SESSION_AB_PROMPT,
  type SessionAbForm,
  type SessionAbModel,
  type SessionAbRequest,
  sessionAbRequest,
} from "./request";
import {
  type ArmSpec,
  type ArmStep,
  pipelineArm,
  probeCounts,
  type ProbeStep,
  type RunPosition,
  runPositions,
} from "./run";
import {
  isVaeTarget,
  type SessionAbVaeRequest,
  type SessionAbVaeTarget,
  sessionAbVaeRequest,
  VAE_AB_MOUNTS,
  VAE_AB_TARGETS,
  vaeAbModel,
  vaeArmOptions,
  vaeBackendRefusal,
  vaeFeeds,
  vaeRowRule,
  vaeSessionOptions,
} from "./vae";
import { armRow, judgementView, recordedCounts } from "./view";

/**
 * Module constant (useGpu needs stable options): subgroups when the adapter has them (the MOSS-Image
 * DiT's online attention takes its subgroup form with them, its portable form without).
 */
const GPU_OPTIONS: GpuOptions = { subgroups: "available" };

const BACKEND_CHOICES: readonly Choice<string>[] = [
  { value: "", label: "Choose the backend…", disabled: true },
  ...SESSION_AB_BACKENDS.map((backend) => ({ value: backend, label: backend })),
];

/** The work an A/B runs. */
type SessionAbTarget = "gemma4-decode" | "moss-dit-forward" | SessionAbVaeTarget;

const TARGET_CHOICES: readonly Choice<SessionAbTarget>[] = [
  { value: "gemma4-decode", label: "Gemma 4 decode (candidate/gelu_mul)" },
  { value: "moss-dit-forward", label: "MOSS-Image DiT forward (models/ew_chain)" },
  { value: "wan22-vae-next", label: "Wan2.2 VAE next chunk (candidate/vae_rms_silu_g8 / _g4)" },
  { value: "anima-vae-decode", label: "anima VAE decode (candidate/vae_rms_silu_g8 / _g4)" },
];

const MIB = 2 ** 20;

const manifestOf = (state: ManifestState) => (state.kind === "ready" ? state.manifest : undefined);

const positionText = (position: RunPosition | undefined, arm: string): string =>
  position === undefined
    ? arm
    : `${position.round === 0 ? "warm-up round" : `round ${position.round}`} · visit ${position.visit} · ${arm}${position.timed ? " (timed)" : " (untimed)"}`;

/**
 * The status line of one arm step: where the runner is, and how many runs have ended. `opens` names
 * what an arm opens (the Gemma target's pipeline, the MOSS target's Session).
 */
const progressOf = (
  step: ArmStep,
  positions: readonly RunPosition[],
  ended: number,
  opens: string,
): RunProgress => {
  const count = { done: ended, total: positions.length, unit: "runs" };
  const next = positions[ended];
  switch (step.step) {
    case "open":
      return { stage: `opening the ${opens} · ${positionText(next, step.arm)}`, count };
    case "ran":
      return { stage: `ran · ${positionText(positions[ended - 1], step.arm)}`, count };
    case "close":
      return { stage: `closing the ${opens} of ${step.arm}`, count };
  }
};

/** The status line of one probe step, and how many probes have ended. */
const probeProgressOf = (step: ProbeStep, ended: number, total: number): RunProgress => {
  const count = { done: ended, total, unit: "probes" };
  switch (step.step) {
    case "open":
      return { stage: `probing the counts · building the pipeline of ${step.arm}`, count };
    case "run":
      return { stage: `probing the counts · one untimed generation of ${step.arm}`, count };
    case "close":
      return { stage: `probing the counts · closing the pipeline of ${step.arm}`, count };
  }
};

/**
 * The step reporter of a Session target's arms (the MOSS-Image DiT, the VAEs): the status is
 * rendered at once outside the timed spans, and the ticker is held from the step before a timed run
 * to the step after it, as on the Gemma target. A Session run takes no signal, so a cancel stops the
 * A/B at the next visit's open.
 */
const sessionStepper = (
  positions: readonly RunPosition[],
  signal: AbortSignal,
  report: (progress: RunProgress) => void,
  holdTicker: (hold: boolean) => void,
): ((step: ArmStep) => void) => {
  let ended = 0;
  return (step) => {
    if (step.step === "open") signal.throwIfAborted();
    if (step.step === "ran") ended += 1;
    holdTicker(step.step !== "close" && ended < positions.length && positions[ended].timed);
    flushSync(() => report(progressOf(step, positions, ended, "Session")));
  };
};

/** What one press of Run produced, on any target. */
type SessionAbResult =
  | {
      readonly kind: "running";
      readonly request: SessionAbRequest | SessionAbMossRequest | SessionAbVaeRequest;
    }
  | {
      readonly kind: "done";
      readonly record: SessionAbPageRecord | SessionAbMossPageRecord | SessionAbVaePageRecord;
    };

export const SessionAbPage = () => {
  const index = useDistributionIndex();
  const distributionIndex = index.kind === "ready" ? index.index : undefined;
  const statuses = useMemo(
    (): Readonly<Record<GemmaMount, DistributionStatus | undefined>> => ({
      gemma4: mountStatus(distributionIndex, "gemma4"),
      "gemma4-qat": mountStatus(distributionIndex, "gemma4-qat"),
      "gemma4-qat-mobile": mountStatus(distributionIndex, "gemma4-qat-mobile"),
    }),
    [distributionIndex],
  );
  const regularManifest = useManifest("gemma4", statuses.gemma4);
  const qatManifest = useManifest("gemma4-qat", statuses["gemma4-qat"]);
  const mobileManifest = useManifest("gemma4-qat-mobile", statuses["gemma4-qat-mobile"]);
  const manifestStates = useMemo(
    (): Readonly<Record<GemmaMount, ManifestState>> => ({
      gemma4: regularManifest,
      "gemma4-qat": qatManifest,
      "gemma4-qat-mobile": mobileManifest,
    }),
    [regularManifest, qatManifest, mobileManifest],
  );
  const manifests = useMemo<GemmaManifests>(() => {
    const loaded: GemmaManifests = {};
    for (const mount of GEMMA_MOUNTS) {
      const manifest = manifestOf(manifestStates[mount]);
      if (manifest !== undefined) loaded[mount] = manifest;
    }
    return loaded;
  }, [manifestStates]);

  // The model, the preset and the load options (the gemma4 page's).
  const modelChoices = useMemo(() => gemmaModelChoices(manifests), [manifests]);
  const [chosenModel, setChosenModel] = useState<ModelKey | undefined>(undefined);
  const modelKey = effectiveModelKey(chosenModel, manifests);
  const manifest = modelKey === undefined ? undefined : manifests[modelKey.mount];
  const presets = useMemo(
    () =>
      manifest === undefined || modelKey === undefined
        ? undefined
        : manifestPresets(manifest, modelKey.model),
    [manifest, modelKey],
  );
  const [chosenQuant, setChosenQuant] = useState<string | undefined>(undefined);
  const quant =
    presets === undefined
      ? undefined
      : chosenQuant !== undefined && presets.presets.some(({ name }) => name === chosenQuant)
        ? chosenQuant
        : presets.defaultQuant;
  const [f16Kv, setF16Kv] = useState(false);
  const expertCacheAvailable = modelKey !== undefined && hasExpertCache(modelKey);
  const [slotsText, setSlotsText] = useState("");
  const [hostTierText, setHostTierText] = useState("");
  const [depthText, setDepthText] = useState("");
  const slotsParsed = useMemo(() => parseExpertCacheSlots(slotsText), [slotsText]);
  const hostTierParsed = useMemo(() => parseExpertHostTier(hostTierText), [hostTierText]);
  const depthParsed = useMemo(() => parseExpertSpeculationDepth(depthText), [depthText]);
  const slots = expertCacheAvailable && slotsParsed.ok ? slotsParsed.value : undefined;
  const cacheErrors =
    slots === undefined
      ? expertCacheAvailable && !slotsParsed.ok
        ? slotsParsed.errors
        : []
      : [
          ...(hostTierParsed.ok ? [] : hostTierParsed.errors),
          ...(depthParsed.ok ? [] : depthParsed.errors),
        ];
  const hostBytes = slots !== undefined && hostTierParsed.ok ? hostTierParsed.value : undefined;
  const depth = slots !== undefined && depthParsed.ok ? depthParsed.value : undefined;

  const [form, setForm] = useState<SessionAbForm>(INITIAL_SESSION_AB_FORM);
  const parsed = useMemo(() => parseSessionAbForm(form), [form]);

  const gpu = useGpu(GPU_OPTIONS);
  const readyGpu = gpu.state.kind === "ready" ? gpu.state.gpu : undefined;
  const task = useRunTask<void>(gpu);
  const running = task.running;
  const [result, setResult] = useState<SessionAbResult | undefined>(undefined);

  const distribution = (() => {
    if (modelKey === undefined) return undefined;
    const state = manifestStates[modelKey.mount];
    const status = statuses[modelKey.mount];
    return state.kind === "ready" && status !== undefined
      ? { directory: status.directory, manifestSha256: state.sha256 }
      : undefined;
  })();
  const model: SessionAbModel | undefined =
    modelKey === undefined || quant === undefined
      ? undefined
      : {
          mount: modelKey.mount,
          model: modelKey.model,
          quant,
          ...(f16Kv ? { fullStateStorage: "f16" as const } : {}),
          ...(slots === undefined
            ? {}
            : {
                expertCache: {
                  slots,
                  ...(hostBytes === undefined ? {} : { hostBytes }),
                  ...(depth === undefined ? {} : { depth }),
                },
              }),
        };
  // One source for every arm and probe (the cached route by default: an A/B loads the pipeline once
  // per visit and probe, and the direct route would fetch the weights anew each time).
  const {
    source,
    hfRef,
    choose: chooseSource,
  } = usePipelineSource(modelKey === undefined ? undefined : statuses[modelKey.mount]);
  const canRun =
    readyGpu !== undefined &&
    model !== undefined &&
    distribution !== undefined &&
    parsed.ok &&
    cacheErrors.length === 0 &&
    !running;

  const run = (): void => {
    if (!canRun || readyGpu === undefined || model === undefined || !parsed.ok) return;
    const settings = parsed.value;
    const request = sessionAbRequest(model, settings, GELU_MUL_RULE_NAME);
    const environment = recordEnvironment(readyGpu);
    const facts = distribution;
    setResult({ kind: "running", request });
    void task.start(async ({ signal, report, holdTicker }) => {
      const positions = runPositions(request.arms.length, settings.rounds, settings.openPerVisit);
      let ended = 0;
      // Steps come from an arm's open, report and close, all outside the timed spans. The status is
      // rendered at once (flushSync), so no render is left to land in the run that follows, and the
      // ticker is held from the step before a timed run to the step after it.
      const onStep = (step: ArmStep): void => {
        if (step.step === "ran") ended += 1;
        holdTicker(step.step !== "close" && ended < positions.length && positions[ended].timed);
        flushSync(() => report(progressOf(step, positions, ended, "pipeline")));
      };
      const specOf = (
        name: string,
        fusionRules: SessionOptions["fusionRules"],
      ): ArmSpec<GemmaPipeline> => ({
        name,
        ...(fusionRules === undefined ? {} : { fusionRules }),
        request,
        prompt: (pipeline) =>
          chatPrompt(pipeline.tokenizer, [{ role: "user", content: request.prompt }]),
        build: (options) =>
          buildGemmaPipeline({
            gpu: readyGpu,
            request: {
              mount: model.mount,
              model: model.model,
              quant: model.quant,
              source,
              speculative: false,
              ...(model.fullStateStorage === undefined
                ? {}
                : { fullStateStorage: model.fullStateStorage }),
              ...(model.expertCache === undefined ? {} : { expertCache: model.expertCache }),
            },
            signal,
            ...(options.fusionRules === undefined
              ? {}
              : { sessionOptions: { fusionRules: options.fusionRules } }),
            ...(options.onRunDiagnostics === undefined
              ? {}
              : { onRunDiagnostics: options.onRunDiagnostics }),
          }),
        signal,
      });
      const specs = [specOf("base", undefined), specOf("rule", [GELU_MUL_RULE])];
      // The count probes come first, one arm at a time, all untimed; the arms are never built with
      // the diagnostics callback.
      const probes: SessionAbArmProbe[] = [];
      if (settings.probeCounts) {
        for (const spec of specs) {
          const counts = await probeCounts({
            ...spec,
            onStep: (step) => report(probeProgressOf(step, probes.length, specs.length)),
          });
          probes.push({ name: spec.name, counts });
        }
      }
      const arms = specs.map((spec) => pipelineArm({ ...spec, onStep }));
      const record = await runSessionAb({
        arms,
        rounds: settings.rounds,
        openPerVisit: settings.openPerVisit,
        environment: {
          adapter: readyGpu.adapterInfo,
          userAgent: navigator.userAgent,
          backend: settings.backend,
          date: new Date().toISOString(),
        },
      });
      const judgements = judgeSessionAb(record, { minGain: settings.minGain });
      if (facts === undefined) throw new Error("The distribution's facts went away");
      setResult({
        kind: "done",
        record: buildSessionAbPageRecord({
          environment,
          versions: PACKAGE_VERSIONS,
          request,
          distribution: facts,
          source: sourceRecord(source),
          ruleName: GELU_MUL_RULE_NAME,
          arms: arms.map((arm) => ({ name: arm.name, facts: arm.facts() })),
          ...(settings.probeCounts ? { probes } : {}),
          record,
          judgements,
        }),
      });
    });
  };

  // The MOSS-Image DiT target: the distribution's one model at the chosen preset, from its own source.
  const [target, setTarget] = useState<SessionAbTarget>("gemma4-decode");
  const moss = target === "moss-dit-forward";
  const mossStatus = mountStatus(distributionIndex, MOSS_MOUNT);
  const mossManifestState = useManifest(MOSS_MOUNT, mossStatus);
  const mossManifest = manifestOf(mossManifestState);
  const mossPresets = useMemo(
    () =>
      mossManifest === undefined
        ? undefined
        : manifestPresets(mossManifest, mossManifest.defaultModel),
    [mossManifest],
  );
  const [chosenMossQuant, setChosenMossQuant] = useState<string | undefined>(undefined);
  const mossQuant =
    mossPresets === undefined
      ? undefined
      : chosenMossQuant !== undefined &&
          mossPresets.presets.some(({ name }) => name === chosenMossQuant)
        ? chosenMossQuant
        : mossPresets.defaultQuant;
  const mossModel: SessionAbMossModel | undefined =
    mossManifest === undefined || mossQuant === undefined
      ? undefined
      : mossAbModel(mossManifest, mossQuant);
  const {
    source: mossSource,
    hfRef: mossHfRef,
    choose: chooseMossSource,
  } = usePipelineSource(mossStatus);
  const mossDistribution =
    mossManifestState.kind === "ready" && mossStatus !== undefined
      ? { directory: mossStatus.directory, manifestSha256: mossManifestState.sha256 }
      : undefined;
  // The token count is the Gemma target's: the Session targets parse the form without it.
  const sessionParsed = useMemo(
    () => parseSessionAbForm({ ...form, tokens: INITIAL_SESSION_AB_FORM.tokens }),
    [form],
  );
  const canRunMoss =
    readyGpu !== undefined &&
    mossModel !== undefined &&
    mossDistribution !== undefined &&
    sessionParsed.ok &&
    !running;

  const runMoss = (): void => {
    if (!canRunMoss || readyGpu === undefined || mossModel === undefined || !sessionParsed.ok) {
      return;
    }
    const request = sessionAbMossRequest(mossModel, sessionParsed.value, EW_CHAIN_RULE_NAME);
    const environment = recordEnvironment(readyGpu);
    const facts = mossDistribution;
    const source = mossSource;
    setResult({ kind: "running", request });
    void task.start(async ({ signal, report, holdTicker }) => {
      report({ stage: "fetching and opening the dit component" });
      const model = await openComponentModel(
        pretrainedRef(source, MOSS_MOUNT),
        request.model,
        request.quant,
        request.component,
        signal,
        (progress) =>
          report({
            stage: "fetching the dit component",
            count: {
              done: Math.round(progress.loaded / MIB),
              total: Math.round(progress.total / MIB),
              unit: "MiB",
            },
          }),
      );
      const feeds = mossDitFeeds(
        model.prepared.graph.inputs,
        request.imageTokens,
        request.textTokens,
        request.seed,
      );
      const positions = runPositions(request.arms.length, request.rounds, request.openPerVisit);
      const onStep = sessionStepper(positions, signal, report, holdTicker);
      const knobs: SessionOptions = { ...request.session };
      const arms = request.arms.map(({ name, fusionRules }) =>
        steppedArm(
          sessionArm({
            name,
            prepared: model.prepared,
            gpu: readyGpu,
            options: fusionRules.length === 0 ? knobs : { ...knobs, fusionRules: [EW_CHAIN_RULE] },
            feeds,
          }),
          onStep,
        ),
      );
      const record = await runSessionAb({
        arms,
        rounds: request.rounds,
        openPerVisit: request.openPerVisit,
        environment: {
          adapter: readyGpu.adapterInfo,
          userAgent: navigator.userAgent,
          backend: request.backend,
          date: new Date().toISOString(),
        },
      });
      const judgements = judgeSessionAb(record, { minGain: request.minGain });
      if (facts === undefined) throw new Error("The distribution's facts went away");
      setResult({
        kind: "done",
        record: buildSessionAbMossPageRecord({
          environment,
          versions: PACKAGE_VERSIONS,
          request,
          distribution: facts,
          source: sourceRecord(source),
          container: { component: request.component, part0Sha256: model.part0Sha256 },
          ruleName: EW_CHAIN_RULE_NAME,
          record,
          judgements,
        }),
      });
    });
  };

  // The VAE targets: the distribution's default model at the chosen preset, from the target's mount.
  // Both manifests are read, so switching targets never pairs one target with the other's manifest.
  const vae = isVaeTarget(target);
  const vaeTarget: SessionAbVaeTarget = vae ? target : "wan22-vae-next";
  const vaeMount = VAE_AB_MOUNTS[vaeTarget];
  const wanVaeStatus = mountStatus(distributionIndex, VAE_AB_MOUNTS["wan22-vae-next"]);
  const animaVaeStatus = mountStatus(distributionIndex, VAE_AB_MOUNTS["anima-vae-decode"]);
  const wanVaeManifestState = useManifest(VAE_AB_MOUNTS["wan22-vae-next"], wanVaeStatus);
  const animaVaeManifestState = useManifest(VAE_AB_MOUNTS["anima-vae-decode"], animaVaeStatus);
  const wanVae = vaeTarget === "wan22-vae-next";
  const vaeStatus = wanVae ? wanVaeStatus : animaVaeStatus;
  const vaeManifestState = wanVae ? wanVaeManifestState : animaVaeManifestState;
  const vaeManifest = manifestOf(vaeManifestState);
  const vaePresets = useMemo(
    () =>
      vaeManifest === undefined
        ? undefined
        : manifestPresets(vaeManifest, vaeManifest.defaultModel),
    [vaeManifest],
  );
  const [chosenVaeQuant, setChosenVaeQuant] = useState<string | undefined>(undefined);
  const vaeQuant =
    vaePresets === undefined
      ? undefined
      : chosenVaeQuant !== undefined &&
          vaePresets.presets.some(({ name }) => name === chosenVaeQuant)
        ? chosenVaeQuant
        : vaePresets.defaultQuant;
  const vaeModel =
    vaeManifest === undefined || vaeQuant === undefined
      ? undefined
      : vaeAbModel(vaeManifest, vaeTarget, vaeQuant);
  const {
    source: vaeSource,
    hfRef: vaeHfRef,
    choose: chooseVaeSource,
  } = usePipelineSource(vaeStatus);
  const vaeDistribution =
    vaeManifestState.kind === "ready" && vaeStatus !== undefined
      ? { directory: vaeStatus.directory, manifestSha256: vaeManifestState.sha256 }
      : undefined;
  const vaeRefusal = sessionParsed.ok ? vaeBackendRefusal(sessionParsed.value.backend) : undefined;
  const canRunVae =
    readyGpu !== undefined &&
    vaeModel !== undefined &&
    vaeDistribution !== undefined &&
    sessionParsed.ok &&
    vaeRefusal === undefined &&
    !running;

  const runVae = (): void => {
    if (!canRunVae || readyGpu === undefined || vaeModel === undefined || !sessionParsed.ok) {
      return;
    }
    const request = sessionAbVaeRequest(vaeTarget, vaeModel, sessionParsed.value);
    const environment = recordEnvironment(readyGpu);
    const facts = vaeDistribution;
    const source = vaeSource;
    setResult({ kind: "running", request });
    void task.start(async ({ signal, report, holdTicker }) => {
      report({ stage: `fetching and opening the ${request.component} component` });
      const model = await openComponentModel(
        pretrainedRef(source, request.mount),
        request.model,
        request.quant,
        request.component,
        signal,
        (progress) =>
          report({
            stage: `fetching the ${request.component} component`,
            count: {
              done: Math.round(progress.loaded / MIB),
              total: Math.round(progress.total / MIB),
              unit: "MiB",
            },
          }),
      );
      const feeds = vaeFeeds(model.prepared.graph.inputs, request.seed);
      const positions = runPositions(request.arms.length, request.rounds, request.openPerVisit);
      const onStep = sessionStepper(positions, signal, report, holdTicker);
      const arms = vaeArmOptions(request).map(({ name, options }) =>
        steppedArm(
          sessionArm({ name, prepared: model.prepared, gpu: readyGpu, options, feeds }),
          onStep,
        ),
      );
      const record = await runSessionAb({
        arms,
        rounds: request.rounds,
        openPerVisit: request.openPerVisit,
        environment: {
          adapter: readyGpu.adapterInfo,
          userAgent: navigator.userAgent,
          backend: request.backend,
          date: new Date().toISOString(),
        },
      });
      const judgements = judgeSessionAb(record, { minGain: request.minGain });
      if (facts === undefined) throw new Error("The distribution's facts went away");
      setResult({
        kind: "done",
        record: buildSessionAbVaePageRecord({
          environment,
          versions: PACKAGE_VERSIONS,
          request,
          distribution: facts,
          source: sourceRecord(source),
          container: { component: request.component, part0Sha256: model.part0Sha256 },
          record,
          judgements,
        }),
      });
    });
  };
  const sessionTarget = moss || vae;

  const missing = GEMMA_MOUNTS.flatMap((mount) => {
    const state = manifestStates[mount];
    return state.kind === "missing" ? [state.directory] : [];
  });
  const formErrors = sessionTarget
    ? sessionParsed.ok
      ? vae && vaeRefusal !== undefined
        ? [vaeRefusal]
        : []
      : sessionParsed.errors
    : parsed.ok
      ? []
      : parsed.errors;

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-2">
        <h1 className="text-2xl font-semibold">Session A/B</h1>
        <MountBadges index={index} mounts={moss ? [MOSS_MOUNT] : vae ? [vaeMount] : GEMMA_MOUNTS} />
      </div>
      {index.kind === "error" ? (
        <ErrorAlert title="Distribution index" error={index.error} />
      ) : null}
      {sessionTarget
        ? null
        : GEMMA_MOUNTS.map((mount) => {
            const state = manifestStates[mount];
            return state.kind === "error" ? (
              <ErrorAlert key={mount} title={`Manifest (${mount})`} error={state.error} />
            ) : null;
          })}
      {!sessionTarget && missing.length === GEMMA_MOUNTS.length ? (
        <ErrorAlert
          title="No Gemma 4 distribution"
          error={new Error(`The dev server has none of ${missing.join(" / ")}.`)}
        />
      ) : null}
      {moss && mossManifestState.kind === "error" ? (
        <ErrorAlert title={`Manifest (${MOSS_MOUNT})`} error={mossManifestState.error} />
      ) : null}
      {moss && mossManifestState.kind === "missing" ? (
        <ErrorAlert
          title="No MOSS-Image distribution"
          error={new Error(`The dev server has no ${mossManifestState.directory}.`)}
        />
      ) : null}
      {vae && vaeManifestState.kind === "error" ? (
        <ErrorAlert title={`Manifest (${vaeMount})`} error={vaeManifestState.error} />
      ) : null}
      {vae && vaeManifestState.kind === "missing" ? (
        <ErrorAlert
          title="No distribution for this VAE"
          error={new Error(`The dev server has no ${vaeManifestState.directory}.`)}
        />
      ) : null}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,24rem)_minmax(0,1fr)]">
        <div className="flex flex-col gap-6">
          <Card>
            <CardHeader>
              <CardTitle>Target</CardTitle>
              <CardDescription>
                The work every timed run does, and the fusion rule each arm but <code>base</code>{" "}
                adds.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Field id="session-ab-target" label="Target">
                <ChoiceSelect
                  id="session-ab-target"
                  value={target}
                  choices={TARGET_CHOICES}
                  onChange={(value) => {
                    setTarget(value);
                    setChosenVaeQuant(undefined);
                    setResult(undefined);
                  }}
                  disabled={running}
                />
              </Field>
            </CardContent>
          </Card>

          {moss ? (
            <Card>
              <CardHeader>
                <CardTitle>DiT</CardTitle>
                <CardDescription>
                  Both arms build their Sessions from the <code>dit</code> component (the positive
                  branch) of this preset, fetched and opened once, with the knobs the preset
                  declares. The arm <code>rule</code> adds the rule{" "}
                  <code>{EW_CHAIN_RULE_NAME}</code> (<code>@karume/models/moss</code>, which the
                  MOSS-Image pipeline passes by default) through{" "}
                  <code>SessionOptions.fusionRules</code>; <code>base</code> passes none.
                </CardDescription>
              </CardHeader>
              <CardContent className="flex flex-col gap-4">
                <SourcePicker
                  id="session-ab-moss-source"
                  value={mossSource.kind}
                  cachedAvailable={mossHfRef !== undefined}
                  onChange={chooseMossSource}
                  disabled={running}
                />
                {mossPresets === undefined || mossQuant === undefined ? null : (
                  <PresetSelect
                    id="session-ab-moss-preset"
                    value={mossQuant}
                    presets={mossPresets.presets}
                    defaultQuant={mossPresets.defaultQuant}
                    onChange={setChosenMossQuant}
                    disabled={running}
                  />
                )}
                <p className="text-xs text-muted-foreground">
                  Knobs:{" "}
                  {mossModel === undefined || Object.keys(mossModel.session).length === 0
                    ? "none (the reference path)"
                    : JSON.stringify(mossModel.session)}
                </p>
              </CardContent>
            </Card>
          ) : vae ? (
            <Card>
              <CardHeader>
                <CardTitle>VAE</CardTitle>
                <CardDescription>
                  Every arm builds its Sessions from the{" "}
                  <code>{VAE_AB_TARGETS[vaeTarget].component}</code> component of this preset,
                  fetched and opened once, with the options its pipeline gives the VAE. The arms{" "}
                  <code>g8</code> and <code>g4</code> add the candidate rule{" "}
                  <code>candidate/vae_rms_silu_g8</code> / <code>_g4</code> (
                  <code>@karume/candidate-rules</code>) through{" "}
                  <code>SessionOptions.fusionRules</code>; <code>base</code> passes none. Not on
                  Metal: the rules&apos; rounding barrier is measured not to hold there.
                </CardDescription>
              </CardHeader>
              <CardContent className="flex flex-col gap-4">
                <SourcePicker
                  id="session-ab-vae-source"
                  value={vaeSource.kind}
                  cachedAvailable={vaeHfRef !== undefined}
                  onChange={chooseVaeSource}
                  disabled={running}
                />
                {vaePresets === undefined || vaeQuant === undefined ? null : (
                  <PresetSelect
                    id="session-ab-vae-preset"
                    value={vaeQuant}
                    presets={vaePresets.presets}
                    defaultQuant={vaePresets.defaultQuant}
                    onChange={setChosenVaeQuant}
                    disabled={running}
                  />
                )}
                <p className="text-xs text-muted-foreground">
                  {vaeModel === undefined
                    ? "Options: —"
                    : `Storage ${vaeModel.storage} · options ${
                        Object.keys(vaeSessionOptions(vaeTarget, vaeModel.session)).length === 0
                          ? "none"
                          : JSON.stringify(vaeSessionOptions(vaeTarget, vaeModel.session))
                      }`}
                </p>
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardHeader>
                <CardTitle>Pipeline</CardTitle>
                <CardDescription>
                  Both arms load this model and preset from the same source. The arm{" "}
                  <code>rule</code> adds the candidate rule <code>{GELU_MUL_RULE_NAME}</code> (
                  <code>@karume/candidate-rules</code>) through the gemma pipeline&apos;s
                  experimental <code>fusionRules</code>; <code>base</code> passes none.
                </CardDescription>
              </CardHeader>
              <CardContent className="flex flex-col gap-4">
                <Field id="session-ab-model" label="Model">
                  <ChoiceSelect
                    id="session-ab-model"
                    value={modelKey === undefined ? "" : modelKeyValue(modelKey)}
                    choices={modelChoices}
                    onChange={(value) => {
                      setChosenModel(parseModelKey(value));
                      setChosenQuant(undefined);
                    }}
                    disabled={running || modelChoices.length === 0}
                  />
                </Field>
                <SourcePicker
                  id="session-ab-source"
                  value={source.kind}
                  cachedAvailable={hfRef !== undefined}
                  onChange={chooseSource}
                  disabled={running}
                />
                {presets === undefined || quant === undefined ? null : (
                  <PresetSelect
                    id="session-ab-preset"
                    value={quant}
                    presets={presets.presets}
                    defaultQuant={presets.defaultQuant}
                    onChange={setChosenQuant}
                    disabled={running}
                  />
                )}
                <div className="flex items-center gap-2">
                  <Switch
                    id="session-ab-f16-kv"
                    checked={f16Kv}
                    onCheckedChange={setF16Kv}
                    disabled={running}
                  />
                  <Label htmlFor="session-ab-f16-kv">f16 KV (full layers)</Label>
                </div>
                <Field
                  id="session-ab-expert-slots"
                  label="Expert cache slots"
                  hint={
                    expertCacheAvailable
                      ? `As on the Gemma 4 chat page: one count for every layer, a comma-separated list (one per routed layer) or "${EXPERT_CACHE_FITTED}"; empty = every expert resident. 64 with the host tier "${EXPERT_HOST_TIER_ALL}" runs 26B on a 16 GB GPU.`
                      : "Only for a mixture-of-experts model (Gemma 4 QAT 26B)."
                  }
                >
                  <Input
                    id="session-ab-expert-slots"
                    value={slotsText}
                    placeholder="off"
                    onChange={(event) => setSlotsText(event.target.value)}
                    disabled={running || !expertCacheAvailable}
                  />
                </Field>
                <Field
                  id="session-ab-host-tier"
                  label="Host tier (GiB)"
                  hint={`"${EXPERT_HOST_TIER_ALL}" = every expert in RAM; empty = every miss reads the distribution. Only with the slots.`}
                >
                  <Input
                    id="session-ab-host-tier"
                    value={hostTierText}
                    placeholder="off"
                    onChange={(event) => setHostTierText(event.target.value)}
                    disabled={running || slots === undefined}
                  />
                </Field>
                <Field
                  id="session-ab-depth"
                  label="Self-speculation depth"
                  hint="Empty = one run per decode token. Only with the slots."
                >
                  <Input
                    id="session-ab-depth"
                    inputMode="numeric"
                    value={depthText}
                    placeholder="off"
                    onChange={(event) => setDepthText(event.target.value)}
                    disabled={running || slots === undefined}
                  />
                </Field>
                {cacheErrors.length === 0 ? null : (
                  <p className="text-xs text-destructive">{cacheErrors.join(" ")}</p>
                )}
              </CardContent>
            </Card>
          )}

          <Card>
            <CardHeader>
              <CardTitle>A/B</CardTitle>
              <CardDescription>
                {vae
                  ? `A timed run is one forward of the ${VAE_AB_TARGETS[vaeTarget].component} graph on fixed feeds drawn from seed ${VAE_AB_TARGETS[vaeTarget].seed}, and every output's words (u32, NaN by class) are what the runner compares — on the Wan2.2 chunk graph its 32 caches too, about 0.9 GiB over the bus per run in every arm. A 1024² anima image decodes the tile forward 9 times.`
                  : moss
                    ? `A timed run is one DiT forward at the sequence of a 1024 × 1024 request (${MOSS_DIT_WORK.imageTokens} image tokens + ${MOSS_DIT_WORK.textTokens} text tokens) on fixed feeds drawn from seed ${MOSS_DIT_WORK.seed}; its output words (u32, NaN by class) are what the runner compares.`
                    : `A timed run is one greedy generation (seed 0) of the chosen token count from one fixed prompt in a sequence of capacity ${SESSION_AB_CAPACITY}; its token ids are what the runner compares.`}{" "}
                One untimed warm-up round, then every round visits the arms forward and in reverse
                {vae ? " (base, g8, g4, g4, g8, base)." : " (base, rule, rule, base)."}
              </CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-4">
              <div className="grid grid-cols-3 gap-4">
                <Field
                  id="session-ab-rounds"
                  label="Rounds"
                  hint={`≥ ${SESSION_AB_LIMITS.minRounds}`}
                >
                  <Input
                    id="session-ab-rounds"
                    inputMode="numeric"
                    value={form.rounds}
                    onChange={(event) => setForm({ ...form, rounds: event.target.value })}
                    disabled={running}
                  />
                </Field>
                <Field
                  id="session-ab-tokens"
                  label="Tokens per run"
                  hint={`${SESSION_AB_LIMITS.minTokens}–${SESSION_AB_LIMITS.maxTokens}`}
                >
                  <Input
                    id="session-ab-tokens"
                    inputMode="numeric"
                    value={sessionTarget ? "—" : form.tokens}
                    onChange={(event) => setForm({ ...form, tokens: event.target.value })}
                    disabled={running || sessionTarget}
                  />
                </Field>
                <Field id="session-ab-min-gain" label="Minimum gain" hint="0.01 = 1%">
                  <Input
                    id="session-ab-min-gain"
                    inputMode="decimal"
                    value={form.minGain}
                    onChange={(event) => setForm({ ...form, minGain: event.target.value })}
                    disabled={running}
                  />
                </Field>
              </div>
              <div className="flex items-center gap-2">
                <Switch
                  id="session-ab-open-per-visit"
                  checked={sessionTarget || form.openPerVisit}
                  onCheckedChange={(checked) => setForm({ ...form, openPerVisit: checked })}
                  disabled={running || sessionTarget}
                />
                <Label htmlFor="session-ab-open-per-visit">Open and close per visit</Label>
              </div>
              <p className="-mt-2 text-xs text-muted-foreground">
                {vae
                  ? "Always on for the VAEs: every visit builds its arm's Session, runs it once untimed and once timed, and disposes it, so two VAE Sessions never live together. Building is never timed."
                  : moss
                    ? "Always on for the DiT: every visit builds its arm's Session, runs it once untimed and once timed, and disposes it, so two DiT Sessions never live together. Building is never timed."
                    : "On: every visit builds its arm's pipeline, runs it once untimed and once timed, and disposes it (two pipelines of a large model do not fit together). Off: both pipelines are built once and stay loaded. Building is never timed."}
              </p>
              {sessionTarget ? (
                <p className="text-xs text-muted-foreground">
                  Counts: every run of a {vae ? "VAE" : "DiT"} arm reports its fusion counts and
                  dispatches to the runner (outside the timed span), so this target needs no count
                  probe.
                </p>
              ) : (
                <>
                  <div className="flex items-center gap-2">
                    <Switch
                      id="session-ab-probe-counts"
                      checked={form.probeCounts}
                      onCheckedChange={(checked) => setForm({ ...form, probeCounts: checked })}
                      disabled={running}
                    />
                    <Label htmlFor="session-ab-probe-counts">
                      Probe the counts first (one extra load and run per arm)
                    </Label>
                  </div>
                  <p className="-mt-2 text-xs text-muted-foreground">
                    Before the A/B, each arm&apos;s pipeline is built once with onRunDiagnostics,
                    runs one untimed generation of the same prompt and token count, and is disposed;
                    the page keeps the fusion counts and the planned dispatches of its first decode
                    run (its first verify run, depth + 1 rows, with a self-speculation depth). The
                    arms of the A/B are always built without the callback, whose presence alone
                    changes the decode path, so every run, timed or not, is the production path. The
                    probe&apos;s tokens are compared with nothing.
                  </p>
                </>
              )}
              <Field
                id="session-ab-backend"
                label="Backend"
                hint="WebGPU does not expose it; the record states what you choose here."
              >
                <ChoiceSelect
                  id="session-ab-backend"
                  value={form.backend}
                  choices={BACKEND_CHOICES}
                  onChange={(backend) => setForm({ ...form, backend })}
                  disabled={running}
                />
              </Field>
              {sessionTarget ? null : (
                <p className="text-xs text-muted-foreground">Prompt: {SESSION_AB_PROMPT}</p>
              )}
              {formErrors.length === 0 ? null : (
                <p className="text-xs text-destructive">{formErrors.join(" ")}</p>
              )}
              <Button
                id="session-ab-run"
                onClick={moss ? runMoss : vae ? runVae : run}
                disabled={!(moss ? canRunMoss : vae ? canRunVae : canRun)}
              >
                Run
              </Button>
            </CardContent>
          </Card>
        </div>

        <div className="flex flex-col gap-6">
          <RunStatusCard
            gpu={gpu.state}
            onReacquire={gpu.reacquire}
            task={task}
            rows={[
              [
                "Request",
                result === undefined
                  ? "—"
                  : requestText(result.kind === "running" ? result.request : result.record.request),
              ],
            ]}
          />
          {result?.kind === "done" ? <SessionAbResultView record={result.record} /> : null}
        </div>
      </div>
    </div>
  );
};

const requestText = (
  request: SessionAbRequest | SessionAbMossRequest | SessionAbVaeRequest,
): string =>
  "sessionOptions" in request
    ? [
        `${request.mount}/${request.model} ${request.quant} ${request.component} (${request.storage})`,
        `${request.rounds} rounds × one forward, ${request.arms.map(({ name }) => name).join(" / ")}`,
        "open per visit",
        request.backend,
      ].join(" · ")
    : "target" in request
      ? [
          `${request.mount}/${request.model} ${request.quant} ${request.component}`,
          `${request.rounds} rounds × one forward at S = ${request.imageTokens + request.textTokens}`,
          "open per visit",
          request.backend,
        ].join(" · ")
      : gemmaRequestText(request);

const gemmaRequestText = (request: SessionAbRequest): string =>
  [
    `${request.mount}/${request.model} ${request.quant}`,
    ...(request.fullStateStorage === undefined ? [] : ["f16 KV"]),
    ...(request.expertCache === undefined
      ? []
      : [
          `expert cache ${JSON.stringify(request.expertCache.slots)}${
            request.expertCache.hostBytes === undefined ? "" : " + host tier"
          }${request.expertCache.depth === undefined ? "" : ` depth ${request.expertCache.depth}`}`,
        ]),
    `${request.rounds} rounds × ${request.tokens} tokens`,
    request.openPerVisit ? "open per visit" : "kept open",
    request.probeCounts ? "counts probed" : "no counts",
    request.backend,
  ].join(" · ");

/** Memoized: the status ticker re-renders the page every 250 ms while an untimed step is going. */
const SessionAbResultView = memo(
  ({
    record,
  }: {
    record: SessionAbPageRecord | SessionAbMossPageRecord | SessionAbVaePageRecord;
  }) => {
    // The Session targets' counts are the runner's; a VAE arm's hits are its own rule's counter.
    const counted = "target" in record;
    const vaeRequest = "rules" in record ? record.request : undefined;
    const ruleText =
      "rules" in record ? record.rules.map(({ name }) => name).join(" / ") : record.rule.name;
    const rows = record.record.arms.map((arm, index) =>
      armRow(
        arm,
        counted ? recordedCounts(arm) : record.counts?.[index],
        "rules" in record ? vaeRowRule(record.request, arm.name) : record.rule.name,
      ),
    );
    const rounds = record.record.rounds;
    return (
      <>
        <Card>
          <CardHeader>
            <CardTitle>Arms</CardTitle>
            <CardDescription>
              Identity is against the base&apos;s first run, over every run (the warm-up and the
              untimed runs included). Hits are the counter of <code>{ruleText}</code>
              {vaeRequest === undefined ? "" : " (each arm's own rule; the base reads the first)"}
              {counted
                ? " and dispatches the dispatches of one forward, as every run of the arm reported them to the runner (a run whose counts differ from its arm's first run's is refused)."
                : " and dispatches the planned dispatches in the first decode run (the first verify run with a self-speculation depth) of the arm's count probe before the A/B, not of the timed runs: the runner's record carries no counts, and without the probe they show —."}{" "}
              Times are the timed runs of each round, visit 1 / visit 2.
            </CardDescription>
          </CardHeader>
          <CardContent className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Arm</TableHead>
                  <TableHead>Identity</TableHead>
                  <TableHead className="text-right">Hits</TableHead>
                  <TableHead className="text-right">Dispatches</TableHead>
                  {Array.from({ length: rounds }, (_, round) => (
                    <TableHead key={round} className="text-right">
                      Round {round + 1}
                    </TableHead>
                  ))}
                </TableRow>
              </TableHeader>
              <TableBody>
                {rows.map((row) => (
                  <TableRow key={row.name}>
                    <TableCell className="font-mono text-xs">{row.name}</TableCell>
                    <TableCell
                      className={`text-xs whitespace-normal ${row.identical ? "" : "text-destructive"}`}
                    >
                      {row.identity}
                    </TableCell>
                    <TableCell className="text-right font-mono text-xs">{row.hits}</TableCell>
                    <TableCell className="text-right font-mono text-xs">{row.dispatches}</TableCell>
                    {row.rounds.map((times, round) => (
                      <TableCell key={round} className="text-right font-mono text-xs">
                        {times}
                      </TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
        {record.judgements.map((judgement) => {
          const view = judgementView(judgement);
          return (
            <Card key={view.arm}>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  Judgement: {view.arm} against {view.base}
                  <Badge id={`session-ab-verdict-${view.arm}`} variant={view.tone}>
                    {view.label}
                  </Badge>
                </CardTitle>
                <CardDescription>
                  Minimum gain {(record.request.minGain * 100).toFixed(2)}%, identity required. The
                  verdict speaks for this device, backend and work only; inconclusive does not mean
                  the difference is zero.
                </CardDescription>
              </CardHeader>
              <CardContent className="flex flex-col gap-3">
                <p className="text-sm">{view.reason}</p>
                <KeyValueTable rows={view.numbers} />
              </CardContent>
            </Card>
          );
        })}
        <Card>
          <CardHeader>
            <CardTitle>Record</CardTitle>
            <CardDescription>
              The runtime&apos;s <code>{record.record.format}</code> record unchanged, inside the
              page&apos;s envelope with the request, the distribution and the judgements.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <RecordActions
              record={record}
              filename={`${
                vaeRequest !== undefined
                  ? `session-ab-vae-${vaeRequest.target}`
                  : counted
                    ? "session-ab-moss"
                    : "session-ab"
              }_${fileStamp(new Date(record.recordedAt))}.json`}
            />
          </CardContent>
        </Card>
      </>
    );
  },
);
SessionAbResultView.displayName = "SessionAbResultView";
