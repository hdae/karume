/**
 * The Style-Bert-VITS2 (JVNV) text → speech check: pick a speaker (one model each), a preset and a
 * style, analyse Japanese text into an utterance, synthesize it on the page's GPU, play it and save
 * a record of the run (the request and the analysed utterance, the stage seconds, the Session
 * diagnostics per graph and the sha256 of the samples).
 */

import type { Manifest } from "@karume/hub";
import type { RunComponent } from "@karume/models/sbv2";
import type { AcquireGpuOptions } from "@karume/runtime";
import { useEffect, useMemo, useRef, useState } from "react";

import { AudioClip } from "@/components/audio-clip";
import { ChoiceSelect } from "@/components/choice-select";
import { ErrorAlert, thrownText } from "@/components/error-alert";
import { Field } from "@/components/field";
import { KeyValueTable } from "@/components/key-value-table";
import { MountBadges } from "@/components/mount-badges";
import { PresetSelect } from "@/components/preset-select";
import { RecordActions } from "@/components/record-actions";
import { SourcePicker } from "@/components/source-picker";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { useDistributionIndex } from "@/hooks/use-distribution-index";
import { useGpu } from "@/hooks/use-gpu";
import { useManifest } from "@/hooks/use-manifest";
import { usePipelineSource } from "@/hooks/use-pipeline-source";
import { type RunProgress, useRunTask } from "@/hooks/use-run-task";
import { useTimestampQuery } from "@/hooks/use-timestamp-query";
import { fileStamp } from "@/lib/download";
import { recordEnvironment } from "@/lib/environment";
import { manifestPresets, type Preset } from "@/lib/presets";
import {
  type BuildCost,
  type ComponentRecord,
  type ComponentTiming,
  f32Sha256,
  foldComponentDiagnostics,
  seconds,
  sha256Hex,
  tallyTiming,
  type TimingTally,
  topOpShares,
} from "@/lib/record";

import { analyzeText, ANALYZER } from "./analyzer";
import { readSbv2Config, type Sbv2ModelConfig, speakerModelChoices } from "./config";
import { buildSbv2Record, SBV2_STAGES, type Sbv2Record } from "./record";
import {
  buildRequest,
  defaultKnobs,
  formForModel,
  KNOBS,
  type ReadingSegment,
  readingSegments,
  type Sbv2Form,
} from "./request";
import { DEFAULT_SAMPLE_TEXT, SAMPLE_TEXTS } from "./samples";
import { Sbv2StatusCard } from "./status-card";
import { type ReadingDictionaryState, useReadingDictionary } from "./use-reading-dictionary";
import { type DiagnosticsSink, useSbv2Pipeline } from "./use-sbv2-pipeline";

/**
 * Module constants (useGpu needs stable options): subgroups when the adapter has them, and GPU
 * timestamps when the "GPU timing" switch is on.
 */
const UNTIMED_GPU: Omit<AcquireGpuOptions, "onDeviceLost"> = { subgroups: "available" };
const TIMED_GPU: Omit<AcquireGpuOptions, "onDeviceLost"> = {
  subgroups: "available",
  gpuTiming: true,
};

/** How many ops the summary names per timed graph. */
const TOP_OPS = 5;

const STAGE_LABELS: Readonly<Record<RunComponent, string>> = {
  text_encoder: "text encoder (DeBERTa)",
  front: "front (encoder + duration predictors)",
  voice: "voice (flow + decoder)",
};

const MIB = 2 ** 20;
const GIB = 2 ** 30;

/** Every stage of a run: `generate` takes no AbortSignal and reports nothing until a graph has run. */
const NOT_CANCELLABLE = {
  cancellable: false,
  cancelNote:
    "A synthesis cannot be cancelled (sbv2.Pipeline.generate takes no AbortSignal); it runs to its end.",
} as const satisfies RunProgress;

type Sbv2Result = {
  readonly record: Sbv2Record;
  readonly samples: Float32Array<ArrayBuffer>;
  readonly sampleRate: number;
};

/** The model's presets, speakers, styles and defaults — or why the manifest cannot be shown. */
type ModelView =
  | {
      readonly kind: "ready";
      readonly model: string;
      readonly quant: string;
      readonly presets: readonly Preset[];
      readonly defaultQuant: string;
      readonly config: Sbv2ModelConfig;
      readonly modelChoices: readonly { readonly value: string; readonly label: string }[];
    }
  | { readonly kind: "error"; readonly error: unknown };

/**
 * The chosen model and preset as they apply to `manifest`: an empty or unknown choice falls back to
 * the manifest's `defaultModel` / the model's `defaultQuant`.
 */
const modelView = (manifest: Manifest, chosenModel: string, chosenQuant: string): ModelView => {
  try {
    const model = Object.hasOwn(manifest.models, chosenModel) ? chosenModel : manifest.defaultModel;
    const { presets, defaultQuant } = manifestPresets(manifest, model);
    const quant = presets.some(({ name }) => name === chosenQuant) ? chosenQuant : defaultQuant;
    return {
      kind: "ready",
      model,
      quant,
      presets,
      defaultQuant,
      config: readSbv2Config(manifest, model),
      modelChoices: speakerModelChoices(manifest),
    };
  } catch (error) {
    return { kind: "error", error };
  }
};

type AnalysisPreview =
  | { readonly kind: "none" }
  | { readonly kind: "ready"; readonly segments: readonly ReadingSegment[]; readonly moras: number }
  | { readonly kind: "error"; readonly error: unknown };

export const Sbv2Page = () => {
  const index = useDistributionIndex();
  const status = index.kind === "ready" ? index.index["sbv2"] : undefined;
  const manifest = useManifest("sbv2", status);

  // A device setting, not a form field: switching re-acquires the GPU (which rebuilds the pipeline).
  const [gpuTiming, setGpuTiming] = useState(false);
  const timestampQuery = useTimestampQuery();
  const { state: gpuState, lease, reacquire, track } = useGpu(gpuTiming ? TIMED_GPU : UNTIMED_GPU);
  const gpu = gpuState.kind === "ready" ? gpuState.gpu : undefined;

  const dictionary = useReadingDictionary();
  const { source, hfRef, choose: chooseSource } = usePipelineSource(status);

  const [chosenModel, setChosenModel] = useState("");
  const [chosenQuant, setChosenQuant] = useState("");
  // Memoized: the form starts from `config.defaults` once, and the selects keep their choices.
  const view = useMemo(
    () =>
      manifest.kind === "ready"
        ? modelView(manifest.manifest, chosenModel, chosenQuant)
        : undefined,
    [manifest, chosenModel, chosenQuant],
  );
  const ready = view?.kind === "ready" ? view : undefined;

  const [form, setForm] = useState<Sbv2Form | undefined>(undefined);
  // Start from the distribution's defaults once the manifest is known.
  const initialDefaults = ready?.config.defaults;
  useEffect(() => {
    if (initialDefaults === undefined) return;
    setForm(
      (current) => current ?? { ...defaultKnobs(initialDefaults), text: DEFAULT_SAMPLE_TEXT.text },
    );
  }, [initialDefaults]);
  const update = (patch: Partial<Sbv2Form>): void =>
    setForm((current) => (current === undefined ? current : { ...current, ...patch }));

  const diagnosticsRef = useRef<DiagnosticsSink | undefined>(undefined);
  const pipeline = useSbv2Pipeline({
    gpu,
    source,
    model: ready?.model,
    quant: ready?.quant,
    diagnostics: diagnosticsRef,
    track,
  });
  const task = useRunTask<Sbv2Result>({ track, lease });
  const running = task.running;

  const dictionaryState = dictionary.state;
  const loadedDictionary =
    dictionaryState.kind === "ready" ? dictionaryState.dictionary : undefined;
  const text = form?.text ?? "";
  // The reading as the analyser sees the text now (the run analyses the text again and records it).
  const preview = useMemo<AnalysisPreview>(() => {
    if (loadedDictionary === undefined || text.trim() === "") return { kind: "none" };
    try {
      const utterance = analyzeText(loadedDictionary, text);
      return {
        kind: "ready",
        segments: readingSegments(utterance),
        moras: utterance.moras.length,
      };
    } catch (error) {
      return { kind: "error", error };
    }
  }, [loadedDictionary, text]);

  let requestError: unknown;
  try {
    if (form !== undefined) buildRequest(form);
  } catch (error) {
    requestError = error;
  }

  const canRun =
    !running &&
    form !== undefined &&
    requestError === undefined &&
    gpu !== undefined &&
    manifest.kind === "ready" &&
    ready !== undefined &&
    pipeline.kind === "ready" &&
    // The pipeline is the one for the current choices and GPU (not the previous one in the render
    // before the rebuild starts — after a GPU re-acquire it would hold a destroyed device).
    pipeline.gpu === gpu &&
    pipeline.model === ready.model &&
    pipeline.quant === ready.quant &&
    pipeline.source === source &&
    loadedDictionary !== undefined;

  const run = (): void => {
    if (!canRun) return;
    const request = buildRequest(form);
    const built = pipeline;
    const dictionaryNow = loadedDictionary;
    const gpuNow = gpu;
    const preset = ready.presets.find(({ name }) => name === built.quant);
    const distribution = {
      directory: status?.directory ?? "",
      manifestSha256: manifest.sha256,
    };
    const defaultQuant = ready.defaultQuant;
    void task.start(async ({ report }) => {
      const startedAt = performance.now();
      report({ stage: "analysing the text", ...NOT_CANCELLABLE });
      const utterance = analyzeText(dictionaryNow, request.text);
      const analysedAt = performance.now();

      const diagnostics = new Map<RunComponent, ComponentRecord>();
      const timings = new Map<RunComponent, TimingTally>();
      const marks: Partial<Record<RunComponent, number>> = {};
      const total = SBV2_STAGES.length;
      diagnosticsRef.current = (component, sessionDiagnostics) => {
        marks[component] = performance.now();
        diagnostics.set(
          component,
          foldComponentDiagnostics(diagnostics.get(component), sessionDiagnostics),
        );
        const tally = tallyTiming(timings.get(component), sessionDiagnostics.lastRunTiming);
        if (tally !== undefined) timings.set(component, tally);
        const done = SBV2_STAGES.indexOf(component) + 1;
        const next = SBV2_STAGES[done];
        report({
          stage: next === undefined ? "finishing" : STAGE_LABELS[next],
          count: { done, total, unit: "graphs" },
          ...NOT_CANCELLABLE,
        });
      };
      report({
        stage: STAGE_LABELS.text_encoder,
        count: { done: 0, total, unit: "graphs" },
        ...NOT_CANCELLABLE,
      });
      let audio;
      try {
        audio = await built.pipeline.generate(utterance, request.options);
      } finally {
        diagnosticsRef.current = undefined;
      }
      const synthesizedAt = performance.now();

      report({ stage: "hashing the output", ...NOT_CANCELLABLE });
      // The WAV encoder lives in the models barrel; load it only here.
      const { encodeWav } = await import("@karume/models");
      const [samplesSha256, wavSha256] = await Promise.all([
        f32Sha256(audio.data),
        sha256Hex(encodeWav(audio.data, audio.sampleRate)),
      ]);
      const record = buildSbv2Record({
        environment: recordEnvironment(gpuNow),
        distribution,
        source: built.source,
        model: built.model,
        preset: { quant: built.quant, label: preset?.label ?? built.quant, defaultQuant },
        pipelineLoadSeconds: built.loadSeconds,
        request,
        analyzer: ANALYZER,
        utterance,
        clock: {
          startedAt,
          analysedAt,
          marks,
          synthesizedAt,
          finishedAt: performance.now(),
        },
        audio: { sampleRate: audio.sampleRate, samples: audio.data.length },
        hashes: { f32Sha256: samplesSha256, wavSha256 },
        diagnostics,
        timings,
      });
      return { record, samples: audio.data, sampleRate: audio.sampleRate };
    });
  };

  const disabled = running || ready === undefined || form === undefined;
  const config = ready?.config;
  const sample = SAMPLE_TEXTS.find((entry) => entry.text === form?.text);

  const statusText =
    manifest.kind === "loading"
      ? "loading manifest"
      : manifest.kind === "missing"
        ? "distribution missing"
        : manifest.kind === "error" || view?.kind === "error"
          ? "manifest error"
          : pipeline.kind === "ready"
            ? "ready"
            : pipeline.kind === "error"
              ? "pipeline error"
              : "building the pipeline";

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap items-baseline gap-3">
        <h1 className="text-2xl font-semibold">Style-Bert-VITS2 — text → speech</h1>
        <Badge id="sbv2-status" variant={pipeline.kind === "ready" ? "secondary" : "outline"}>
          {statusText}
        </Badge>
      </div>

      {index.kind === "error" ? (
        <ErrorAlert title="Distribution index" error={index.error} />
      ) : null}
      <MountBadges index={index} mounts={["sbv2"]} />
      {manifest.kind === "missing" ? (
        <Alert>
          <AlertTitle>The Style-Bert-VITS2 distribution is missing</AlertTitle>
          <AlertDescription>
            The dev server has no <code>{manifest.directory}</code> (a directory with a{" "}
            <code>karume.json</code>). Put the distribution there and reload.
          </AlertDescription>
        </Alert>
      ) : null}
      {manifest.kind === "error" ? <ErrorAlert title="Manifest" error={manifest.error} /> : null}
      {view?.kind === "error" ? (
        <ErrorAlert title="The manifest's models" error={view.error} />
      ) : null}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <Card>
          <CardHeader>
            <CardTitle>Request</CardTitle>
            <CardDescription>
              Changing the speaker, the preset or the source rebuilds the pipeline (with the cached
              source the first build of a model / preset downloads all its weights; the GPU works
              only on Run). Every option below is passed to generate explicitly.
            </CardDescription>
          </CardHeader>
          <CardContent className="grid gap-4 sm:grid-cols-2">
            <div className="sm:col-span-2">
              <Field
                id="sbv2-model"
                label="Speaker"
                hint="One model per speaker (the manifest's models, with the speakers each holds)."
              >
                <ChoiceSelect
                  id="sbv2-model"
                  value={ready?.model ?? ""}
                  choices={ready?.modelChoices ?? []}
                  onChange={(model) => {
                    setChosenModel(model);
                    if (manifest.kind !== "ready") return;
                    const next = modelView(manifest.manifest, model, chosenQuant);
                    if (next.kind === "ready") {
                      setForm((current) =>
                        current === undefined ? current : formForModel(current, next.config),
                      );
                    }
                  }}
                  disabled={disabled}
                />
              </Field>
            </div>
            {config !== undefined && config.speakers.length > 1 ? (
              <div className="sm:col-span-2">
                <Field id="sbv2-speaker" label="Speaker name">
                  <ChoiceSelect
                    id="sbv2-speaker"
                    value={form?.speaker ?? config.defaults.speaker}
                    choices={config.speakers.map((name) => ({
                      value: name,
                      label: `${name}${name === config.defaults.speaker ? " (default)" : ""}`,
                    }))}
                    onChange={(speaker) => update({ speaker })}
                    disabled={disabled}
                  />
                </Field>
              </div>
            ) : null}
            <div className="sm:col-span-2">
              <PresetSelect
                id="sbv2-quant"
                value={ready?.quant ?? ""}
                presets={ready?.presets ?? []}
                defaultQuant={ready?.defaultQuant ?? ""}
                onChange={setChosenQuant}
                disabled={disabled}
              />
            </div>
            <div className="sm:col-span-2">
              <SourcePicker
                id="sbv2-source"
                value={source.kind}
                cachedAvailable={hfRef !== undefined}
                onChange={chooseSource}
                disabled={running}
              />
            </div>
            <div className="sm:col-span-2">
              <Field id="sbv2-sample" label="Sample text" hint="Fills the text below.">
                <ChoiceSelect
                  id="sbv2-sample"
                  value={sample?.id ?? "custom"}
                  choices={[
                    ...SAMPLE_TEXTS.map(({ id, label }) => ({ value: id, label })),
                    { value: "custom", label: "Custom text", disabled: true },
                  ]}
                  onChange={(id) => {
                    const chosen = SAMPLE_TEXTS.find((entry) => entry.id === id);
                    if (chosen !== undefined) update({ text: chosen.text });
                  }}
                  disabled={disabled}
                />
              </Field>
            </div>
            <div className="sm:col-span-2">
              <Field
                id="sbv2-text"
                label="Text (Japanese)"
                hint={`${text.length} characters · ${limitsText(config)}`}
              >
                <Textarea
                  id="sbv2-text"
                  rows={4}
                  value={text}
                  onChange={(event) => update({ text: event.target.value })}
                  disabled={disabled}
                />
              </Field>
            </div>
            <div className="sm:col-span-2">
              <ReadingPreview preview={preview} dictionary={dictionaryState} />
            </div>
            <div className="sm:col-span-2">
              <Field id="sbv2-style" label="Style">
                <ChoiceSelect
                  id="sbv2-style"
                  value={form?.style ?? ""}
                  choices={(config?.styles ?? []).map((name) => ({
                    value: name,
                    label: `${name}${name === config?.defaults.style ? " (default)" : ""}`,
                  }))}
                  onChange={(style) => update({ style })}
                  disabled={disabled}
                />
              </Field>
            </div>
            {KNOBS.map(({ key, label, min, max, step, hint }) => {
              const value = form?.[key] ?? config?.defaults[key] ?? min;
              const fallback = config?.defaults[key];
              return (
                <Field
                  key={key}
                  id={`sbv2-${key}`}
                  label={`${label}: ${value.toFixed(2)}`}
                  hint={`${hint}${fallback === undefined ? "" : ` Default ${fallback}.`}`}
                >
                  <Slider
                    id={`sbv2-${key}`}
                    min={min}
                    max={max}
                    step={step}
                    value={value}
                    onValueChange={(next) => {
                      if (typeof next === "number") update({ [key]: next });
                    }}
                    disabled={disabled}
                  />
                </Field>
              );
            })}
            <Field
              id="sbv2-seed"
              label="Seed"
              hint="The same seed and options give the same waveform (0 = the pipeline's default)."
            >
              <Input
                id="sbv2-seed"
                inputMode="numeric"
                value={form?.seed ?? ""}
                onChange={(event) => update({ seed: event.target.value })}
                disabled={disabled}
              />
            </Field>
            <div className="flex flex-col gap-2 sm:col-span-2">
              {requestError === undefined ? null : (
                <p id="sbv2-request-error" className="text-sm text-destructive">
                  {thrownText(requestError)}
                </p>
              )}
              <div className="flex flex-wrap gap-2">
                <Button id="sbv2-run" onClick={run} disabled={!canRun}>
                  Run
                </Button>
                <Button
                  id="sbv2-reset"
                  variant="outline"
                  onClick={() => {
                    if (config === undefined) return;
                    setForm((current) => ({
                      ...defaultKnobs(config.defaults),
                      text: current?.text ?? DEFAULT_SAMPLE_TEXT.text,
                    }));
                  }}
                  disabled={disabled}
                >
                  Reset options to the distribution's defaults
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        <Sbv2StatusCard
          gpu={gpuState}
          onReacquire={reacquire}
          task={task}
          rows={[
            ["Reading dictionary", dictionaryText(dictionaryState)],
            [
              "Pipeline",
              pipeline.kind === "ready"
                ? `${pipeline.model} / ${pipeline.quant} · ${pipeline.source.kind} · built in ${pipeline.loadSeconds.toFixed(2)} s`
                : pipeline.kind === "loading"
                  ? pipeline.detail
                  : pipeline.kind,
            ],
          ]}
          inputHint={`Fix the request and run again: ${limitsText(config)} Split a long text at a sentence end, or lower the length scale.`}
        >
          <div className="flex flex-col gap-1.5">
            <Label className="flex items-center gap-2">
              <Switch
                id="sbv2-gpu-timing"
                checked={gpuTiming}
                onCheckedChange={(checked) => setGpuTiming(checked)}
                disabled={
                  running ||
                  gpuState.kind === "acquiring" ||
                  timestampQuery.kind !== "ready" ||
                  !timestampQuery.available
                }
              />
              GPU timing
            </Label>
            <p id="sbv2-gpu-timing-hint" className="text-xs text-muted-foreground">
              {timestampQuery.kind === "probing" ? "Looking at the adapter… " : null}
              {timestampQuery.kind === "error"
                ? `Unavailable: the adapter could not be probed (${thrownText(timestampQuery.error)}). `
                : null}
              {timestampQuery.kind === "ready" && !timestampQuery.available
                ? "Unavailable: this adapter has no timestamp-query. "
                : null}
              Times every kernel of the three graphs on the GPU. Every dispatch becomes its own
              pass, so a timed run is slower and its seconds are not comparable with untimed
              records. Switching acquires the GPU again and rebuilds the pipeline.
            </p>
          </div>
          {pipeline.kind === "error" ? (
            <ErrorAlert title="Pipeline" error={pipeline.error} />
          ) : null}
          {dictionaryState.kind === "error" ? (
            <div className="flex flex-col gap-2">
              <ErrorAlert title="Reading dictionary" error={dictionaryState.error} />
              <Button
                id="sbv2-dictionary-retry"
                variant="outline"
                size="sm"
                className="self-start"
                onClick={dictionary.retry}
              >
                Load the dictionary again
              </Button>
            </div>
          ) : null}
        </Sbv2StatusCard>
      </div>

      {task.state.kind === "done" ? <ResultCard result={task.state.result} /> : null}
    </div>
  );
};

/** The distribution's two input limits, as the text hint and the input-error hint quote them. */
const limitsText = (config: Sbv2ModelConfig | undefined): string =>
  `one synthesis takes at most ${config?.maxTokens ?? "—"} tokens and as many phonemes (maxTokens), and at most ${
    config?.maxFrames ?? "—"
  } frames in total (maxFrames — the predicted length, which grows with the length scale; checked after the front graph has run). A request over either is rejected.`;

const dictionaryText = (state: ReadingDictionaryState): string => {
  if (state.kind === "error") return "failed";
  if (state.kind === "ready") {
    return `${ANALYZER.name} ${ANALYZER.version} · ready in ${state.seconds.toFixed(2)} s`;
  }
  const total = state.total === undefined ? "" : ` / ${(state.total / MIB).toFixed(1)}`;
  return `loading (${(state.loaded / MIB).toFixed(1)}${total} MiB)`;
};

/** The analysed reading of the text, high moras overlined (the usual pitch-accent notation). */
const ReadingPreview = ({
  preview,
  dictionary,
}: {
  preview: AnalysisPreview;
  dictionary: ReadingDictionaryState;
}) => (
  <div className="flex flex-col gap-1.5">
    <span className="text-sm font-medium">Reading</span>
    {preview.kind === "ready" ? (
      <p id="sbv2-reading" className="text-base leading-loose break-all">
        {preview.segments.map((segment, at) =>
          segment.kind === "punctuation" ? (
            <span key={at} className="px-0.5 text-muted-foreground">
              {segment.text}
            </span>
          ) : (
            <span key={at} className={segment.high ? "border-t-2 border-foreground" : undefined}>
              {segment.kana}
            </span>
          ),
        )}
      </p>
    ) : null}
    {preview.kind === "error" ? <ErrorAlert title="Analysis" error={preview.error} /> : null}
    <p className="text-xs text-muted-foreground">
      {preview.kind === "ready" ? `${preview.moras} moras · ` : null}
      {dictionary.kind === "ready"
        ? "What the analyser reads (overline = high tone). Fix a wrong reading in the text (kana)."
        : "Waiting for the reading dictionary."}
    </p>
  </div>
);

const gibibytes = (bytes: number): string => (bytes / GIB).toFixed(3);

const buildText = (cost: BuildCost): string =>
  [
    `supply wait ${seconds(cost.supplyWaitMs)} s`,
    `decode ${seconds(cost.decodeMs)} s`,
    `buffer create ${seconds(cost.bufferCreateMs)} s`,
    `write issue ${seconds(cost.writeBufferIssueMs)} s`,
    `upload fence ${seconds(cost.uploadFenceMs)} s`,
    `${gibibytes(cost.uploadedBytes)} GiB in ${cost.partCount} parts`,
  ].join(" · ");

/** The timed total beside the stage's wall seconds, then the most expensive ops with their share. */
const timingText = (timing: ComponentTiming, wall: number | undefined): string =>
  [
    `${timing.totalMs.toFixed(1)} ms timed${wall === undefined ? "" : ` (stage ${wall} s wall)`}`,
    ...topOpShares(timing, TOP_OPS).map(
      ({ op, ms, share }) => `${op} ${ms.toFixed(1)} ms ${(share * 100).toFixed(1)}%`,
    ),
    ...(timing.clampedNegativeSamples === 0
      ? []
      : [`${timing.clampedNegativeSamples} samples clamped to 0`]),
  ].join(" · ");

const ResultCard = ({ result }: { result: Sbv2Result }) => {
  const { record } = result;
  const stamp = fileStamp(new Date(record.recordedAt));
  const { request, timings, output } = record;
  return (
    <Card>
      <CardHeader>
        <CardTitle>Result</CardTitle>
        <CardDescription>
          {record.model} · {request.options.style} × {request.options.styleWeight} · seed{" "}
          {request.options.seed}
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <AudioClip
          samples={result.samples}
          sampleRate={result.sampleRate}
          filename={`sbv2_${stamp}.wav`}
        />
        <KeyValueTable
          rows={[
            [
              "Adapter",
              `${record.adapter.vendor} · ${record.adapter.architecture} · ${record.adapter.device}`,
            ],
            ["Driver", record.adapter.description],
            [
              "Device",
              `subgroups ${record.device.features.includes("subgroups")} · GPU timing ${record.device.gpuTimingEnabled}`,
            ],
            [
              "Source",
              record.source.kind === "cached"
                ? `cached (${record.source.repo} @ ${record.source.revision?.slice(0, 12) ?? "main"})`
                : "direct",
            ],
            ["Model", record.model],
            ["Preset", `${record.preset.quant}${record.preset.isDefault ? " (default)" : ""}`],
            ["Speaker", request.options.speaker],
            ["Style", `${request.options.style} × ${request.options.styleWeight}`],
            [
              "Knobs",
              `length ${request.options.lengthScale} · sdp ${request.options.sdpRatio} · noise ${request.options.noiseScale} · noise W ${request.options.noiseScaleW} · seed ${request.options.seed}`,
            ],
            ["Reading", record.analysis.reading],
            ["Analyser", `${record.analysis.analyzer} ${record.analysis.analyzerVersion}`],
            ["Duration s", String(output.durationSeconds)],
            ["Real-time factor", String(output.realTimeFactor)],
            ["Analysis s", String(timings.analysisSeconds)],
            ...SBV2_STAGES.flatMap((stage) => {
              const value = timings.stageSeconds[stage];
              return value === undefined
                ? []
                : [[`${STAGE_LABELS[stage]} s`, String(value)] as const];
            }),
            ["Synthesis s", String(timings.synthesisSeconds)],
            ["Wall s", String(timings.wallSeconds)],
            ["Pipeline build s", String(timings.pipelineLoadSeconds)],
            ["Samples", `${output.samples} @ ${output.sampleRate} Hz`],
            ["f32 sha256", output.f32Sha256],
            ["WAV sha256", output.wavSha256],
            ...SBV2_STAGES.flatMap((stage) => {
              const summary = record.diagnostics[stage];
              if (summary === undefined) return [];
              const rows: (readonly [string, string])[] = [
                [
                  stage,
                  `weights ${summary.weightsAllocatedGiB} GiB · ${summary.submitCount} submits · ${summary.dispatchCount} dispatches · build: ${buildText(summary.build)}`,
                ],
              ];
              if (summary.timing !== undefined) {
                rows.push([
                  `${stage} GPU time`,
                  timingText(summary.timing, timings.stageSeconds[stage]),
                ]);
              }
              return rows;
            }),
          ]}
        />
        <RecordActions record={record} filename={`sbv2_${stamp}.json`} />
      </CardContent>
    </Card>
  );
};
