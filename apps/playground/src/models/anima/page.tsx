/**
 * Anima text → image: pick one of the official models or the third-party fine-tunes and a preset,
 * generate one image on the page's GPU, watch the stages and an approximate preview of each denoise
 * step, and save a record of the run (request, per-stage seconds, Session diagnostics, the sha256
 * of the RGBA output).
 */

import {
  SAMPLER_TYPES,
  SOURCES,
  type RunComponent,
  type SamplerType,
  approximatePreview,
} from "@karume/models/anima";
import type { AcquireGpuOptions } from "@karume/runtime";
import { useRef, useState } from "react";

import { type Choice, ChoiceSelect } from "@/components/choice-select";
import { ErrorAlert, thrownText } from "@/components/error-alert";
import { ExperimentalRuleSwitch } from "@/components/experimental-rule-switch";
import { Field } from "@/components/field";
import { KeyValueTable } from "@/components/key-value-table";
import { MountBadges } from "@/components/mount-badges";
import { PresetSelect } from "@/components/preset-select";
import { RecordActions } from "@/components/record-actions";
import { RgbaImage } from "@/components/rgba-image";
import { RunStatusCard } from "@/components/run-status-card";
import { SourcePicker } from "@/components/source-picker";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { useDistributionIndex } from "@/hooks/use-distribution-index";
import { useGpu } from "@/hooks/use-gpu";
import { type ManifestState, useManifest } from "@/hooks/use-manifest";
import { usePipelineSource } from "@/hooks/use-pipeline-source";
import { type RunProgress, useRunTask } from "@/hooks/use-run-task";
import { crossRepoMounts, type DistributionStatus, type RepoHolders } from "@/lib/distributions";
import { fileStamp } from "@/lib/download";
import { recordEnvironment } from "@/lib/environment";
import {
  experimentalRuleInUse,
  experimentalRuleLabel,
  runsOnMetal,
  VAE_RMS_SILU_EXPERIMENT,
} from "@/lib/experimental-rule";
import type { RgbaPixels } from "@/lib/image";
import { manifestPresets } from "@/lib/presets";
import {
  type BuildCost,
  type ComponentRecord,
  foldComponentDiagnostics,
  foldStageEvent,
  seconds,
  sha256Hex,
  sourceRecord,
  type Spread,
  type StageTimes,
  summarizeIntervals,
  totalBuildCost,
} from "@/lib/record";

import { PreviewCanvas } from "./preview-canvas";
import {
  type AnimaRecord,
  assembleRecord,
  buildsOfThisRun,
  carriedOverAfter,
  type ResidencyEvent,
  STAGE_LABELS,
  STAGES,
} from "./record";
import { SAMPLE_PROMPTS } from "./samples";
import {
  ANIMA_MOUNTS,
  type AnimaForm,
  type AnimaManifests,
  type AnimaMount,
  animaModelChoices,
  buildRequest,
  defaultModelRef,
  effectiveRequest,
  initialForm,
  type ModelRef,
  modelDefaults,
  modelKey,
  MOUNT_LABELS,
  parseModelKey,
  randomSeed,
  RESOLUTION_CHOICES,
  resolutionText,
  usesNegativePrompt,
} from "./settings";
import { type DiagnosticsSink, useAnimaPipeline } from "./use-anima-pipeline";

/**
 * Module constants (useGpu needs stable options). `shaderF16` only while a preset that computes in
 * f16 (`f16-c16`) is chosen: a feature can be requested only when the device is created, and the
 * runtime treats it as an explicit opt-in (it runs an f16 check at acquisition), so a failing f16
 * path never takes the other presets down with it.
 */
const GPU_PLAIN: Omit<AcquireGpuOptions, "onDeviceLost"> = {};
const GPU_WITH_F16: Omit<AcquireGpuOptions, "onDeviceLost"> = { shaderF16: true };

/**
 * The fine-tunes take the text encoder, the VAE and the tokenizers from the official repository;
 * on the direct route its files come from the `anima` mount (the cached route follows the
 * manifest's cross-repo reference on `/hf/`).
 */
const REPO_HOLDERS: RepoHolders<AnimaMount> = { [SOURCES["anima"].repo]: "anima" };

const MIB = 2 ** 20;

const TITLE = "Anima — text → image";

type RunResult = { readonly record: AnimaRecord; readonly image: RgbaPixels };

type LivePreview = { readonly image: RgbaPixels; readonly step: number; readonly steps: number };

const SAMPLER_CHOICE_DEFAULT = "default";
type SamplerChoice = typeof SAMPLER_CHOICE_DEFAULT | SamplerType;

export const AnimaPage = () => {
  const index = useDistributionIndex();
  const statusOf = (mount: AnimaMount): DistributionStatus | undefined =>
    index.kind === "ready" ? index.index[mount] : undefined;
  const officialManifest = useManifest("anima", statusOf("anima"));
  const extraManifest = useManifest("anima-extra", statusOf("anima-extra"));
  const manifestStates: Readonly<Record<AnimaMount, ManifestState>> = {
    anima: officialManifest,
    "anima-extra": extraManifest,
  };
  const manifests: AnimaManifests = {
    ...(officialManifest.kind === "ready" ? { anima: officialManifest.manifest } : {}),
    ...(extraManifest.kind === "ready" ? { "anima-extra": extraManifest.manifest } : {}),
  };

  const [form, setForm] = useState<AnimaForm>(() => initialForm(SAMPLE_PROMPTS[0]?.prompt ?? ""));
  const update = (patch: Partial<AnimaForm>): void =>
    setForm((current) => ({ ...current, ...patch }));

  // The model and preset the page runs, derived on every render: the form's choice while the
  // manifests offer it, else the default (so the default is preselected as soon as a manifest loads).
  const chosen: ModelRef | undefined = (() => {
    const ref = parseModelKey(form.model);
    const manifest = ref === undefined ? undefined : manifests[ref.mount];
    if (ref !== undefined && manifest !== undefined && Object.hasOwn(manifest.models, ref.model)) {
      return ref;
    }
    return defaultModelRef(manifests);
  })();
  const manifest = chosen === undefined ? undefined : manifests[chosen.mount];
  const presets =
    manifest === undefined || chosen === undefined
      ? undefined
      : manifestPresets(manifest, chosen.model);
  const quant =
    presets === undefined
      ? undefined
      : presets.presets.some(({ name }) => name === form.quant)
        ? form.quant
        : presets.defaultQuant;
  const defaults =
    manifest === undefined || chosen === undefined ? {} : modelDefaults(manifest, chosen.model);
  const needsShaderF16 =
    manifest !== undefined &&
    chosen !== undefined &&
    quant !== undefined &&
    manifest.models[chosen.model]?.quants[quant]?.gpuFeatures?.shaderF16 === true;

  const chosenStatus = chosen === undefined ? undefined : statusOf(chosen.mount);
  const { source, hfRef, choose: chooseSource } = usePipelineSource(chosenStatus);

  // Switching shader-f16 acquires the GPU again; the held pipeline (with a resident DiT) is released
  // through `track` first, so useGpu destroys the old device only after that dispose.
  const {
    state: gpuState,
    lease,
    reacquire,
    track,
  } = useGpu(needsShaderF16 ? GPU_WITH_F16 : GPU_PLAIN);
  const gpu = gpuState.kind === "ready" ? gpuState.gpu : undefined;
  const metal = runsOnMetal(navigator.userAgent, gpu?.adapterInfo.architecture ?? "");
  const fusionRule = experimentalRuleInUse(VAE_RMS_SILU_EXPERIMENT, form.experimentalRule, metal);

  const diagnosticsRef = useRef<DiagnosticsSink | undefined>(undefined);
  const { held, releaseError, open, release } = useAnimaPipeline(
    { gpu, source, mount: chosen?.mount, model: chosen?.model, quant, fusionRule },
    diagnosticsRef,
    track,
  );

  const task = useRunTask<RunResult>({ track, lease });
  const [livePreview, setLivePreview] = useState(true);
  const [preview, setPreview] = useState<LivePreview | undefined>(undefined);

  let requestError: unknown;
  try {
    buildRequest(form, defaults);
  } catch (error) {
    requestError = error;
  }
  const negativeUsed = usesNegativePrompt(form, defaults);
  const disabled = task.running || chosen === undefined;
  const canRun =
    !task.running &&
    gpu !== undefined &&
    chosen !== undefined &&
    quant !== undefined &&
    requestError === undefined;

  const run = (): void => {
    if (
      gpu === undefined ||
      chosen === undefined ||
      quant === undefined ||
      manifest === undefined ||
      presets === undefined
    ) {
      return;
    }
    const mountManifest = manifestStates[chosen.mount];
    if (mountManifest.kind !== "ready") return;
    const spec = { gpu, source, mount: chosen.mount, model: chosen.model, quant, fusionRule };
    const directory = chosenStatus?.directory ?? "";
    const quantLabel = presets.presets.find(({ name }) => name === quant)?.label ?? quant;
    const previewOn = livePreview;
    const requestForm = form;
    const requestDefaults = defaults;
    setPreview(undefined);

    void task.start(async ({ signal, report }) => {
      const parts = buildRequest(requestForm, requestDefaults);
      const effective = effectiveRequest(parts, requestDefaults);
      // The direct route maps the repositories the manifest references to mounts (the fine-tunes
      // read the official mount); the cached route follows the references on /hf/.
      const crossRepo =
        source.kind === "direct"
          ? crossRepoMounts(manifest, chosen.model, quant, REPO_HOLDERS)
          : {};
      for (const target of new Set(Object.values(crossRepo))) {
        const status = statusOf(target);
        if (status?.available !== true) {
          throw new Error(
            `${chosen.model} takes its text encoder, VAE and tokenizers from the ${target} mount, but ${
              status?.directory ?? target
            } is missing`,
          );
        }
      }

      const diagnostics = new Map<string, ComponentRecord>();
      diagnosticsRef.current = (component, diagnosticsReport) =>
        diagnostics.set(
          component,
          foldComponentDiagnostics(diagnostics.get(component), diagnosticsReport),
        );
      try {
        report({ stage: "Opening the pipeline" });
        const opened = await open(spec, crossRepo, signal, (progress) =>
          report({
            stage: "Opening the pipeline",
            count: {
              done: Math.round(progress.loaded / MIB),
              total: Math.round(progress.total / MIB),
              unit: "MiB",
            },
            note: progress.path,
          }),
        );

        let progress: RunProgress = {};
        const show = (next: RunProgress): void => {
          progress = next;
          report(progress);
        };
        let stages: StageTimes<RunComponent> = {};
        const stepMarks: number[] = [];
        const tileMarks: number[] = [];
        const residencyEvents: ResidencyEvent[] = [];
        let transformerCarriedOver = opened.residentAtStart;
        const started = performance.now();
        const image = await opened.pipeline.generate({
          ...parts,
          // A throw from onEvent is the pipeline's way to stop a generation (at a stage boundary,
          // a denoise step or a VAE tile); it rethrows the reason unwrapped.
          onEvent: (event) => {
            const now = performance.now();
            // Residency first: a cancelled generation still reports the DiT slot's fate (an
            // eviction or a release), and the held state must follow it before the throw.
            if (event.kind === "residency") {
              opened.noteResidency(event.action);
              transformerCarriedOver = carriedOverAfter(
                transformerCarriedOver,
                event.action,
                stages.transformer?.end !== undefined,
              );
              residencyEvents.push({ action: event.action, reason: event.reason });
            }
            signal.throwIfAborted();
            if (event.kind === "stage") {
              stages = foldStageEvent(stages, event.component, event.at, now);
              if (event.at === "start") show({ stage: STAGE_LABELS[event.component] });
            } else if (event.kind === "denoise-step") {
              // Step 1 again = the stage re-ran after an out-of-memory eviction: start over.
              if (event.step === 1) stepMarks.length = 0;
              stepMarks.push(now);
              show({
                ...progress,
                count: { done: event.step, total: event.steps, unit: "steps" },
                note: `sigma ${event.sigma.toFixed(4)}`,
              });
              if (previewOn) {
                const { data, shape } = event.copyLatents();
                const approximate = approximatePreview(data, shape);
                setPreview({
                  image: {
                    data: approximate.rgba,
                    width: approximate.width,
                    height: approximate.height,
                  },
                  step: event.step,
                  steps: event.steps,
                });
              }
            } else if (event.kind === "vae-tile") {
              if (event.tile === 1) tileMarks.length = 0;
              tileMarks.push(now);
              show({
                ...progress,
                count: { done: event.tile, total: event.tiles, unit: "VAE tiles" },
                note: undefined,
              });
            } else {
              show({ ...progress, note: `DiT ${event.action} (${event.reason})` });
            }
          },
        });
        const generateMilliseconds = performance.now() - started;
        report({ stage: "Hashing the RGBA output" });
        const sha256 = await sha256Hex(image.data);
        const record = assembleRecord({
          environment: recordEnvironment(gpu),
          mount: chosen.mount,
          directory,
          manifestSha256: mountManifest.sha256,
          source: sourceRecord(source),
          model: chosen.model,
          defaultModel: mountManifest.manifest.defaultModel,
          quant,
          quantLabel,
          fusionRule,
          defaultQuant: presets.defaultQuant,
          request: parts,
          effective,
          livePreview: previewOn,
          openMilliseconds: opened.openMilliseconds,
          residencyEvents,
          residentAtStart: opened.residentAtStart,
          transformerCarriedOver,
          stages,
          stepMarks,
          tileMarks,
          generateMilliseconds,
          output: { width: image.width, height: image.height, sha256 },
          diagnostics,
        });
        return { record, image: { data: image.data, width: image.width, height: image.height } };
      } finally {
        diagnosticsRef.current = undefined;
      }
    });
  };

  const result = task.state.kind === "done" ? task.state.result : undefined;

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap items-baseline gap-3">
        <h1 className="text-2xl font-semibold">{TITLE}</h1>
        <Badge id="anima-status" variant={chosen === undefined ? "outline" : "secondary"}>
          {manifestSummary(manifestStates)}
        </Badge>
      </div>
      <MountBadges index={index} mounts={[...ANIMA_MOUNTS]} />

      {index.kind === "error" ? (
        <ErrorAlert title="Distribution index" error={index.error} />
      ) : null}
      {ANIMA_MOUNTS.map((mount) => {
        const state = manifestStates[mount];
        if (state.kind === "error") {
          return <ErrorAlert key={mount} title={`Manifest (${mount})`} error={state.error} />;
        }
        if (state.kind === "missing") {
          return (
            <Alert key={mount}>
              <AlertTitle>
                The {MOUNT_LABELS[mount].toLowerCase()} distribution is missing
              </AlertTitle>
              <AlertDescription>
                The dev server has no <code>{state.directory}</code>; its models are not offered.
                {mount === "anima"
                  ? " The fine-tunes need it too on the direct route (their text encoder, VAE and tokenizers live there)."
                  : null}
              </AlertDescription>
            </Alert>
          );
        }
        return null;
      })}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <Card>
          <CardHeader>
            <CardTitle>Request</CardTitle>
            <CardDescription>
              The pipeline opens on Run and stays open for the next run with the same source, model
              and preset. Empty fields take the model's defaults from the manifest.
            </CardDescription>
          </CardHeader>
          <CardContent className="grid gap-4 sm:grid-cols-2">
            <div className="sm:col-span-2">
              <Field
                id="anima-model"
                label="Model"
                hint={
                  chosen === undefined
                    ? "Waiting for a manifest…"
                    : chosen.mount === "anima"
                      ? `Official model (${chosenStatus?.directory ?? "anima"}).`
                      : `Third-party fine-tune (${chosenStatus?.directory ?? "anima-extra"}); its text encoder, VAE and tokenizers come from the official distribution.`
                }
              >
                <ChoiceSelect
                  id="anima-model"
                  value={chosen === undefined ? "" : modelKey(chosen)}
                  choices={animaModelChoices(manifests)}
                  onChange={(model) => update({ model })}
                  disabled={disabled}
                />
              </Field>
            </div>
            <div className="flex flex-col gap-1.5 sm:col-span-2">
              <PresetSelect
                id="anima-quant"
                value={quant ?? ""}
                presets={presets?.presets ?? []}
                defaultQuant={presets?.defaultQuant ?? ""}
                onChange={(next) => update({ quant: next })}
                disabled={disabled}
              />
              {needsShaderF16 ? (
                <p
                  id="anima-quant-f16"
                  className={
                    gpuState.kind === "error"
                      ? "text-xs text-destructive"
                      : "text-xs text-muted-foreground"
                  }
                >
                  {gpuState.kind === "error"
                    ? "This preset computes in f16: acquiring the GPU with shader-f16 failed (the error is in Status). Choose another preset to acquire the GPU without it."
                    : "This preset computes in f16: the page acquires the GPU with shader-f16 while it is chosen (switching to or from it releases the pipeline and acquires the GPU again)."}
                </p>
              ) : null}
            </div>
            <div className="sm:col-span-2">
              <SourcePicker
                id="anima-source"
                value={source.kind}
                cachedAvailable={hfRef !== undefined}
                onChange={chooseSource}
                disabled={disabled}
              />
            </div>
            <div className="flex flex-col gap-2 sm:col-span-2">
              <Field id="anima-prompt" label="Prompt" hint="Danbooru-style tags work best.">
                <Textarea
                  id="anima-prompt"
                  rows={4}
                  value={form.prompt}
                  onChange={(event) => update({ prompt: event.target.value })}
                  disabled={task.running}
                />
              </Field>
              <div className="flex flex-wrap gap-2">
                {SAMPLE_PROMPTS.map((sample) => (
                  <Button
                    key={sample.id}
                    id={`anima-sample-${sample.id}`}
                    variant={form.prompt === sample.prompt ? "secondary" : "outline"}
                    size="sm"
                    onClick={() => update({ prompt: sample.prompt })}
                    disabled={task.running}
                  >
                    {sample.label}
                  </Button>
                ))}
              </div>
            </div>
            <div className="sm:col-span-2">
              <Field
                id="anima-negative"
                label="Negative prompt"
                hint={
                  negativeUsed
                    ? "Empty = the manifest's default negative prompt."
                    : "Not used at guidance 1 (the uncond side is not computed); the text is kept for a guidance other than 1."
                }
              >
                <Textarea
                  id="anima-negative"
                  rows={2}
                  value={form.negativePrompt}
                  placeholder={
                    defaults.negativePrompt === undefined
                      ? undefined
                      : `default: ${defaults.negativePrompt}`
                  }
                  onChange={(event) => update({ negativePrompt: event.target.value })}
                  disabled={task.running || !negativeUsed}
                />
              </Field>
            </div>
            <Field
              id="anima-guidance"
              label="Guidance"
              hint="1 = no CFG (one DiT pass per step, no negative prompt)."
            >
              <Input
                id="anima-guidance"
                inputMode="decimal"
                value={form.guidanceScale}
                placeholder={placeholderOf(defaults.guidanceScale)}
                onChange={(event) => update({ guidanceScale: event.target.value })}
                disabled={task.running}
              />
            </Field>
            <Field id="anima-steps" label="Steps" hint="2 to 1000.">
              <Input
                id="anima-steps"
                inputMode="numeric"
                value={form.steps}
                placeholder={placeholderOf(defaults.steps)}
                onChange={(event) => update({ steps: event.target.value })}
                disabled={task.running}
              />
            </Field>
            <div className="flex flex-col gap-2 sm:col-span-2">
              <Field
                id="anima-resolution"
                label="Resolution"
                hint="WxH (or one number for a square): each side a multiple of 16 from 512 to 2048."
              >
                <Input
                  id="anima-resolution"
                  value={form.resolution}
                  placeholder={resolutionText(defaults.resolution)}
                  onChange={(event) => update({ resolution: event.target.value })}
                  disabled={task.running}
                />
              </Field>
              <div className="flex flex-wrap gap-2">
                {RESOLUTION_CHOICES.map((choice) => (
                  <Button
                    key={choice.value}
                    variant={form.resolution === choice.value ? "secondary" : "outline"}
                    size="sm"
                    onClick={() => update({ resolution: choice.value })}
                    disabled={task.running}
                  >
                    {choice.label}
                  </Button>
                ))}
              </div>
            </div>
            <Field
              id="anima-sampler"
              label="Sampler"
              hint="Choosing one departs from the distributor's recommendation."
            >
              <ChoiceSelect<SamplerChoice>
                id="anima-sampler"
                value={form.sampler === "" ? SAMPLER_CHOICE_DEFAULT : form.sampler}
                choices={samplerChoices(defaults.sampler)}
                onChange={(next) =>
                  update({ sampler: next === SAMPLER_CHOICE_DEFAULT ? "" : next })
                }
                disabled={task.running}
              />
            </Field>
            <Field id="anima-seed" label="Seed">
              <div className="flex gap-2">
                <Input
                  id="anima-seed"
                  inputMode="numeric"
                  value={form.seed}
                  onChange={(event) => update({ seed: event.target.value })}
                  disabled={task.running}
                />
                <Button
                  id="anima-seed-random"
                  variant="outline"
                  onClick={() => update({ seed: String(randomSeed()) })}
                  disabled={task.running}
                >
                  Random
                </Button>
              </div>
            </Field>
            <div className="sm:col-span-2">
              <Field
                id="anima-residency"
                label="DiT residency"
                hint={
                  form.residency === "transformer"
                    ? "Keep the transformer Session after this run: the next run with the same pipeline skips the DiT load. The text and VAE stages then stack on the resident DiT (about +2.6 GiB at 1024² with the default preset); the pipeline evicts it when a stage would not fit."
                    : "Open and close each stage (the default): the peak is the largest single stage. A DiT kept by an earlier run is released after this one."
                }
              >
                <ToggleGroup
                  id="anima-residency"
                  variant="outline"
                  spacing={0}
                  value={[form.residency]}
                  onValueChange={(value) => {
                    const [next] = value;
                    if (next === "per-stage" || next === "transformer") update({ residency: next });
                  }}
                  disabled={task.running}
                >
                  <ToggleGroupItem value="per-stage">per-stage</ToggleGroupItem>
                  <ToggleGroupItem value="transformer">transformer</ToggleGroupItem>
                </ToggleGroup>
              </Field>
            </div>
            <Label className="flex items-center gap-2 sm:col-span-2">
              <Switch
                id="anima-live-preview"
                checked={livePreview}
                onCheckedChange={(checked) => setLivePreview(checked)}
                disabled={task.running}
              />
              Live preview (approximate, at 1/8 of the image, after every denoise step)
            </Label>
            <div className="sm:col-span-2">
              <ExperimentalRuleSwitch
                id="anima-experimental-rule"
                rule={VAE_RMS_SILU_EXPERIMENT}
                checked={form.experimentalRule}
                metal={metal}
                onCheckedChange={(checked) => update({ experimentalRule: checked })}
                disabled={disabled}
              />
            </div>
            <div className="flex gap-3 sm:col-span-2">
              <Button id="anima-run" onClick={run} disabled={!canRun}>
                Run
              </Button>
              <Button
                id="anima-cancel"
                variant="outline"
                onClick={task.cancel}
                disabled={!task.running}
              >
                Cancel
              </Button>
            </div>
            <p className="text-xs text-muted-foreground sm:col-span-2">
              Cancel takes effect at the next stage boundary, denoise step or VAE tile (a stage that
              is loading or encoding the prompt finishes first), or between the files while the
              pipeline opens.
            </p>
            {requestError !== undefined && chosen !== undefined ? (
              <p id="anima-request-error" className="text-sm text-destructive sm:col-span-2">
                {thrownText(requestError)}
              </p>
            ) : null}
          </CardContent>
        </Card>

        <RunStatusCard
          gpu={gpuState}
          onReacquire={reacquire}
          task={task}
          rows={[
            [
              "shader-f16",
              gpu === undefined
                ? "—"
                : gpu.shaderF16Enabled
                  ? "enabled (requested for this preset)"
                  : "not requested (only f16-c16 needs it)",
            ],
            [
              "Pipeline",
              held === undefined
                ? "not open (opens on Run)"
                : `${held.model} · ${held.quant} · ${held.sourceKind} · rule ${experimentalRuleLabel(held.fusionRule)} (${held.openSeconds.toFixed(1)} s to open)${
                    held.residentDit ? " · DiT resident" : ""
                  }`,
            ],
          ]}
        >
          <div className="flex flex-col gap-1.5">
            <Button
              id="anima-release"
              variant="outline"
              size="sm"
              className="self-start"
              onClick={() => void release()}
              disabled={task.running || held === undefined}
            >
              Release pipeline
            </Button>
            <p className="text-xs text-muted-foreground">
              Drops the open pipeline and a resident DiT; the next Run opens it again.
            </p>
            {releaseError === undefined ? null : (
              <ErrorAlert title="Releasing the pipeline failed" error={releaseError} />
            )}
          </div>
        </RunStatusCard>
      </div>

      {task.running && preview !== undefined ? (
        <Card>
          <CardHeader>
            <CardTitle>Live preview</CardTitle>
            <CardDescription>
              Step {preview.step} / {preview.steps} · approximate (a linear projection of the
              latent,
              {` ${preview.image.width}×${preview.image.height}`}), not the VAE decode.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col">
            <PreviewCanvas image={preview.image} alt="Approximate preview of the current step" />
          </CardContent>
        </Card>
      ) : null}

      {result === undefined ? null : (
        <Card>
          <CardHeader>
            <CardTitle>Result</CardTitle>
            <CardDescription>
              {result.record.output.width}×{result.record.output.height} ·{" "}
              {result.record.model.name} · {result.record.preset.quant} ·{" "}
              {result.record.generateSeconds} s · experimental rule:{" "}
              {experimentalRuleLabel(result.record.experimentalFusionRule)}
            </CardDescription>
          </CardHeader>
          <CardContent className="grid gap-6 xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
            <RgbaImage
              image={result.image}
              alt={`Generated image: ${result.record.request.prompt}`}
              filename={`anima_${fileStamp(new Date(result.record.recordedAt))}.png`}
            />
            <div className="flex flex-col gap-3">
              <RecordActions
                record={result.record}
                filename={`anima_${fileStamp(new Date(result.record.recordedAt))}.json`}
              />
              <RecordTable record={result.record} />
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

const placeholderOf = (value: number | undefined): string =>
  value === undefined ? "default" : `default (${value})`;

const samplerChoices = (manifestSampler: string | undefined): readonly Choice<SamplerChoice>[] => [
  {
    value: SAMPLER_CHOICE_DEFAULT,
    label:
      manifestSampler === undefined ? "manifest default" : `manifest default (${manifestSampler})`,
  },
  ...SAMPLER_TYPES.map((type) => ({ value: type, label: type })),
];

const manifestSummary = (states: Readonly<Record<AnimaMount, ManifestState>>): string => {
  const parts = ANIMA_MOUNTS.map((mount) => {
    const state = states[mount];
    if (state.kind === "ready") {
      return `${MOUNT_LABELS[mount].toLowerCase()}: ${Object.keys(state.manifest.models).length} models`;
    }
    if (state.kind === "missing") return `${MOUNT_LABELS[mount].toLowerCase()}: missing`;
    return `${MOUNT_LABELS[mount].toLowerCase()}: ${state.kind}`;
  });
  return parts.join(" · ");
};

const GIB = 2 ** 30;

const spreadText = (spread: Spread | undefined): string =>
  spread === undefined ? "—" : `${spread.min} / ${spread.median} / ${spread.max} (${spread.count})`;

const buildText = (cost: BuildCost): string =>
  [
    `supply wait ${seconds(cost.supplyWaitMs)} s`,
    `decode ${seconds(cost.decodeMs)} s`,
    `buffer create ${seconds(cost.bufferCreateMs)} s`,
    `write issue ${seconds(cost.writeBufferIssueMs)} s`,
    `upload fence ${seconds(cost.uploadFenceMs)} s`,
    `${(cost.uploadedBytes / GIB).toFixed(2)} GiB in ${cost.partCount} parts`,
  ].join(" · ");

/** First interval and the spread of the rest, as two rows. */
const intervalRows = (
  name: string,
  values: readonly number[],
): readonly (readonly [string, string])[] => {
  const summary = summarizeIntervals(values);
  if (summary === undefined) return [];
  return [
    [`${name} 1 s (from the stage start)`, String(summary.first)],
    [`${name}s 2–${values.length} s (min / median / max, n)`, spreadText(summary.rest)],
  ];
};

const RecordTable = ({ record }: { record: AnimaRecord }) => (
  <KeyValueTable
    rows={[
      [
        "Adapter",
        `${record.adapter.vendor} · ${record.adapter.architecture} · ${record.adapter.device}`,
      ],
      ["Driver", record.adapter.description],
      [
        "Source",
        record.source.kind === "cached"
          ? `cached (${record.source.repo} @ ${record.source.revision?.slice(0, 12) ?? "main"})`
          : "direct",
      ],
      ["Model", `${record.model.name}${record.model.isDefault ? " (default)" : ""}`],
      ["Preset", `${record.preset.quant}${record.preset.isDefault ? " (default)" : ""}`],
      ["Experimental rule", experimentalRuleLabel(record.experimentalFusionRule)],
      ["Resolution", record.effective.resolution ?? "—"],
      ["Steps", String(record.effective.steps ?? "—")],
      ["Guidance", String(record.effective.guidanceScale ?? "—")],
      ["Sampler", record.effective.sampler ?? "pipeline default"],
      ["Seed", String(record.request.seed)],
      ["Negative prompt", record.effective.negativePrompt ?? "not used (guidance 1)"],
      [
        "Residency",
        [
          record.residency.requested,
          ...(record.residency.residentAtStart
            ? [
                record.residency.transformerBuildCarriedOver
                  ? "DiT resident at start (carried over)"
                  : "DiT resident at start (evicted before the DiT stage)",
              ]
            : []),
          ...record.residency.events.map(({ action, reason }) => `${action} (${reason})`),
        ].join(" · "),
      ],
      [
        "Pipeline",
        record.pipeline.reused ? "reused" : `opened in ${String(record.pipeline.openSeconds)} s`,
      ],
      ...STAGES.flatMap((stage) => {
        const value = record.stageSeconds[stage];
        return value === undefined ? [] : [[`${STAGE_LABELS[stage]} s`, String(value)] as const];
      }),
      ["Generate s", String(record.generateSeconds)],
      ...intervalRows("Step", record.timeline.denoiseStepSeconds),
      ...intervalRows("VAE tile", record.timeline.vaeTileSeconds),
      [
        record.residency.transformerBuildCarriedOver
          ? "Build, this run (carried-over DiT left out)"
          : "Build, all components",
        buildText(totalBuildCost(buildsOfThisRun(record))),
      ],
      ["RGBA sha256", record.output.rgbaSha256],
      ...Object.entries(record.diagnostics).map(
        ([component, summary]) =>
          [
            component,
            `${summary.runs} runs · weights ${summary.weightsAllocatedGiB} GiB · ${summary.submitCount} submits · build${
              component === "transformer" && record.residency.transformerBuildCarriedOver
                ? " (an earlier run's, carried over)"
                : ""
            }: ${buildText(summary.build)}`,
          ] as const,
      ),
    ]}
  />
);
