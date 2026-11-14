/**
 * The Irodori-TTS text → speech check: pick v4-small or v4.1-small and a preset, design a voice with
 * a caption or clone one from a reference audio, run one synthesis on the page's GPU, listen to it
 * and save a record of the run (per-stage seconds, real-time factor, Session diagnostics, the sha256
 * of the f32 samples).
 */

import { SOURCES, type GenerateEvent, Pipeline, type RunComponent } from "@karume/models/irodori";
import type { AcquireGpuOptions, GpuContext, SessionDiagnostics } from "@karume/runtime";
import { useEffect, useMemo, useRef, useState } from "react";

import { AudioClip } from "@/components/audio-clip";
import { ChoiceSelect } from "@/components/choice-select";
import { ErrorAlert, thrownText } from "@/components/error-alert";
import { Field } from "@/components/field";
import { KeyValueTable } from "@/components/key-value-table";
import { MountBadges } from "@/components/mount-badges";
import { PresetSelect } from "@/components/preset-select";
import { RecordActions } from "@/components/record-actions";
import { RunStatusCard } from "@/components/run-status-card";
import { SourcePicker } from "@/components/source-picker";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Spinner } from "@/components/ui/spinner";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { useDistributionIndex } from "@/hooks/use-distribution-index";
import { useGpu } from "@/hooks/use-gpu";
import { useManifest } from "@/hooks/use-manifest";
import { usePipelineSource } from "@/hooks/use-pipeline-source";
import { useRunTask } from "@/hooks/use-run-task";
import { useTimestampQuery } from "@/hooks/use-timestamp-query";
import {
  crossRepoMounts,
  mountStatus,
  type PipelineSource,
  pretrainedRef,
  type RepoHolders,
} from "@/lib/distributions";
import { fileStamp } from "@/lib/download";
import { recordEnvironment } from "@/lib/environment";
import { manifestPresets, type Preset } from "@/lib/presets";
import {
  type ComponentRecord,
  f32Sha256,
  foldComponentDiagnostics,
  foldStageEvent,
  seconds,
  sourceRecord,
  type StageTimes,
  summarizeIntervals,
  tallyTiming,
  timingOf,
  type TimingTally,
  topOpShares,
} from "@/lib/record";

import { browserDecoders, type DecodedReference, decodeReferenceFile } from "./decode-reference";
import { acquirePipeline, type Held, type PipelineSlot } from "./held-pipeline";
import { assembleRecord, type IrodoriRecord, recordRequest, STAGE_ORDER } from "./record";
import { CAPTION_EXAMPLES } from "./samples";
import {
  buildRequest,
  changedKeyFields,
  FormError,
  type IrodoriConfigView,
  type IrodoriForm,
  type IrodoriMount,
  type IrodoriRequestParts,
  initialForm,
  LINEAR_CHOICES,
  linearComputeNote,
  MODEL_CHOICES,
  type PipelineKey,
  type PresetFacts,
  readIrodoriConfig,
  readPresetFacts,
  referenceLimits,
  resolveLinearCompute,
  samePipelineKey,
  type VoiceMode,
} from "./settings";

/** Module constants (useGpu needs stable options); GPU timestamps while "GPU timing" is on. */
const UNTIMED_GPU: Omit<AcquireGpuOptions, "onDeviceLost"> = {};
const TIMED_GPU: Omit<AcquireGpuOptions, "onDeviceLost"> = { gpuTiming: true };

/** How many ops the result names per timed component. */
const TOP_OPS = 5;

/**
 * v4.1-small references the components it shares with v4-small in v4-small's repository; on the
 * direct route those files come from the `irodori-v4` mount.
 */
const REPO_HOLDERS: RepoHolders<IrodoriMount> = {
  [SOURCES["irodori-v4-small"].repo]: "irodori-v4",
};

const STAGE_LABELS: Readonly<Record<RunComponent, string>> = {
  backbone: "Backbone (ModernBERT)",
  "text-proj": "Text projection",
  "caption-proj": "Caption projection",
  "codec-encoder": "Codec encoder (reference audio)",
  speaker: "Speaker encoder",
  duration: "Duration predictor",
  "dit-context": "DiT context K/V",
  dit: "DiT",
  "codec-decoder": "Codec decoder",
};

/** What the page knows of the chosen distribution's manifest. */
type ManifestView = {
  readonly model: string;
  readonly presets: readonly Preset[];
  readonly defaultQuant: string;
  readonly config: IrodoriConfigView;
  readonly presetFacts: Readonly<Record<string, PresetFacts>>;
};

type ReferenceState =
  | { readonly kind: "none" }
  | { readonly kind: "decoding"; readonly file: string }
  | {
      readonly kind: "ready";
      readonly file: string;
      readonly decoded: DecodedReference;
      readonly f32Sha256: string;
    }
  | { readonly kind: "error"; readonly file: string; readonly error: unknown };

type IrodoriRun = {
  readonly id: number;
  readonly record: IrodoriRecord;
  readonly samples: Float32Array<ArrayBuffer>;
  readonly sampleRate: number;
};

type IrodoriKey = PipelineKey<GpuContext, PipelineSource>;

/** The pipeline the page keeps between runs (reused while its key holds). */
type HeldPipeline = Held<IrodoriKey, Pipeline>;

const keyLabel = (key: IrodoriKey): string =>
  `${key.model} · ${key.quant} · linear ${key.linearCompute} · ${key.source.kind}`;

type DiagnosticsSink = (component: RunComponent, diagnostics: SessionDiagnostics) => void;

const mib = (bytes: number): string => (bytes / 2 ** 20).toFixed(0);

export const IrodoriPage = () => {
  const index = useDistributionIndex();
  const [form, setForm] = useState<IrodoriForm>(initialForm);
  const update = (patch: Partial<IrodoriForm>): void =>
    setForm((current) => ({ ...current, ...patch }));

  const status = index.kind === "ready" ? mountStatus(index.index, form.mount) : undefined;
  const manifest = useManifest(form.mount, status);
  const view = useMemo<
    | { readonly kind: "ready"; readonly view: ManifestView }
    | { readonly kind: "error"; readonly error: unknown }
    | undefined
  >(() => {
    if (manifest.kind !== "ready") return undefined;
    try {
      const model = manifest.manifest.defaultModel;
      const { presets, defaultQuant } = manifestPresets(manifest.manifest, model);
      return {
        kind: "ready",
        view: {
          model,
          presets,
          defaultQuant,
          config: readIrodoriConfig(manifest.manifest, model),
          presetFacts: readPresetFacts(manifest.manifest, model),
        },
      };
    } catch (error) {
      return { kind: "error", error };
    }
  }, [manifest]);
  const manifestView = view?.kind === "ready" ? view.view : undefined;

  // Preselect the manifest's default preset. A switch to the other distribution keeps the chosen
  // preset when that manifest has it too (both small distributions declare the same five).
  useEffect(() => {
    if (manifestView === undefined) return;
    setForm((current) =>
      manifestView.presets.some(({ name }) => name === current.quant)
        ? current
        : { ...current, quant: manifestView.defaultQuant },
    );
  }, [manifestView]);

  const { source, hfRef, choose: chooseSource } = usePipelineSource(status);

  // A device setting, not a form field: switching re-acquires the GPU (the next run rebuilds the
  // pipeline on the new one).
  const [gpuTiming, setGpuTiming] = useState(false);
  const timestampQuery = useTimestampQuery();
  const { state: gpuState, lease, reacquire, track } = useGpu(gpuTiming ? TIMED_GPU : UNTIMED_GPU);
  const gpu = gpuState.kind === "ready" ? gpuState.gpu : undefined;

  // The reference audio, decoded to the distribution's rate (again when the rate changes). The rate
  // to decode to is the last one a manifest gave: while the next manifest loads after a model switch
  // the rate is unknown, and the decoded reference stays as it is instead of being dropped.
  const [referenceFile, setReferenceFile] = useState<File | undefined>(undefined);
  const [reference, setReference] = useState<ReferenceState>({ kind: "none" });
  const sampleRate = manifestView?.config.sampleRate;
  const [decodeRate, setDecodeRate] = useState<number | undefined>(undefined);
  useEffect(() => {
    if (sampleRate !== undefined) setDecodeRate(sampleRate);
  }, [sampleRate]);
  useEffect(() => {
    if (referenceFile === undefined || decodeRate === undefined) {
      setReference({ kind: "none" });
      return undefined;
    }
    let disposed = false;
    const file = referenceFile.name;
    setReference({ kind: "decoding", file });
    const decode = async (): Promise<ReferenceState> => {
      const decoded = await decodeReferenceFile(referenceFile, decodeRate, await browserDecoders());
      return { kind: "ready", file, decoded, f32Sha256: await f32Sha256(decoded.data) };
    };
    decode().then(
      (next) => {
        if (!disposed) setReference(next);
      },
      (error: unknown) => {
        if (!disposed) setReference({ kind: "error", file, error });
      },
    );
    return () => {
      disposed = true;
    };
  }, [referenceFile, decodeRate]);

  // The pipeline is built by the first Run and kept until its key changes or the page is left.
  // Construction opens no Session (every stage opens and closes its own), so holding it keeps no
  // weights on the GPU. The ref is what the run and the cleanup read synchronously; the state is the
  // same value for rendering — both are written only through `slot.set`. A new GPU (a timing switch,
  // a re-acquire) and leaving the page dispose it through `track`: it was built on the old device,
  // which useGpu destroys only after that dispose (and a run on it) has settled.
  const heldRef = useRef<HeldPipeline | undefined>(undefined);
  const [held, setHeld] = useState<HeldPipeline | undefined>(undefined);
  const slot: PipelineSlot<IrodoriKey, Pipeline> = {
    get: () => heldRef.current,
    set: (next) => {
      heldRef.current = next;
      setHeld(next);
    },
  };
  // The GPU a finishing build may still hold its pipeline for (see acquirePipeline's isCurrent).
  const gpuRef = useRef<GpuContext | undefined>(undefined);
  useEffect(() => {
    gpuRef.current = gpu;
    return () => {
      gpuRef.current = undefined;
      const held = heldRef.current;
      if (held === undefined) return;
      heldRef.current = undefined;
      setHeld(undefined);
      track(held.pipeline.dispose()).catch((error: unknown) => {
        console.error("Irodori-TTS: releasing the pipeline failed", error);
      });
    };
  }, [gpu, track]);
  const diagnosticsRef = useRef<DiagnosticsSink | undefined>(undefined);
  const runIdRef = useRef(0);
  const task = useRunTask<IrodoriRun>({ track, lease });
  const { running } = task;

  let request: IrodoriRequestParts | undefined;
  let requestProblem: string | undefined;
  if (
    form.voice === "reference" &&
    (reference.kind === "decoding" ||
      (reference.kind === "ready" &&
        sampleRate !== undefined &&
        reference.decoded.sampleRate !== sampleRate))
  ) {
    requestProblem = "Decoding the reference audio…";
  } else if (form.voice === "reference" && reference.kind === "error") {
    requestProblem = "The reference audio could not be decoded (the error is above).";
  } else {
    try {
      request = buildRequest(form, reference.kind === "ready" ? reference.decoded : undefined);
    } catch (error) {
      if (!(error instanceof FormError)) throw error;
      requestProblem = error.message;
    }
  }

  // The key the next Run builds for, and how the held pipeline compares with it.
  const currentKey: IrodoriKey | undefined =
    gpu === undefined || manifestView === undefined || form.quant === ""
      ? undefined
      : {
          gpu,
          source,
          mount: form.mount,
          model: manifestView.model,
          quant: form.quant,
          linearCompute: form.linearCompute,
        };
  const pipelineRow = (() => {
    if (held === undefined) return "none open (the next Run builds it)";
    if (currentKey === undefined) return `${keyLabel(held.key)} — the next Run rebuilds it`;
    const changed = changedKeyFields(held.key, currentKey);
    return changed.length === 0
      ? `${keyLabel(held.key)} — the next Run reuses it`
      : `${keyLabel(held.key)} — the next Run rebuilds it (${changed.join(", ")} changed)`;
  })();
  const facts = manifestView?.presetFacts[form.quant];
  const linearResolved = resolveLinearCompute(form.linearCompute, facts?.declaredLinearCompute);
  const linearNote = linearComputeNote(linearResolved, facts?.ditWeights);

  const ditLoop: "host" | "resident" =
    gpu?.gpuTimingEnabled === true || form.stageEvents ? "host" : "resident";
  const canRun =
    !running &&
    gpu !== undefined &&
    manifest.kind === "ready" &&
    manifestView !== undefined &&
    status !== undefined &&
    form.quant !== "" &&
    request !== undefined;

  const run = (): void => {
    if (
      !canRun ||
      currentKey === undefined ||
      manifest.kind !== "ready" ||
      manifestView === undefined ||
      status === undefined ||
      request === undefined
    ) {
      return;
    }
    const parts = request;
    const key = currentKey;
    const linearComputeResolved = linearResolved;
    const stageEvents = form.stageEvents;
    const preset = manifestView.presets.find(({ name }) => name === key.quant);
    const referenceMeta =
      parts.speaker !== undefined && reference.kind === "ready"
        ? {
            file: reference.file,
            decoder: reference.decoded.decoder,
            f32Sha256: reference.f32Sha256,
          }
        : undefined;
    const recordedRequest = recordRequest(parts, referenceMeta, manifestView.config.steps);
    const distribution = {
      mount: key.mount,
      directory: status.directory,
      manifestSha256: manifest.sha256,
    };

    const referencing = manifest.manifest;
    const statuses = index.kind === "ready" ? index.index : undefined;

    void task.start(async ({ signal, report }) => {
      const runStart = performance.now();
      let buildStart = runStart;
      const acquired = await acquirePipeline({
        slot,
        key,
        same: samePipelineKey,
        isCurrent: (built) => gpuRef.current === built.gpu,
        signal,
        onReleasing: () => report({ stage: "releasing the previous pipeline" }),
        build: () => {
          report({ stage: "building the pipeline" });
          buildStart = performance.now();
          // The direct route maps the repositories the manifest references to the mounts that
          // hold them (v4.1-small takes what it shares with v4-small from the v4-small mount); the
          // cached route follows the references on /hf/. A self-contained manifest maps nothing.
          const crossRepo =
            key.source.kind === "direct"
              ? crossRepoMounts(referencing, key.model, key.quant, REPO_HOLDERS)
              : {};
          for (const target of new Set(Object.values(crossRepo))) {
            const held = mountStatus(statuses, target);
            if (held?.available !== true) {
              throw new Error(
                `${key.model} takes the components it shares with v4-small from the ${target} mount, but ${
                  held?.directory ?? target
                } is missing`,
              );
            }
          }
          return Pipeline.fromPretrained(pretrainedRef(key.source, key.mount, crossRepo), {
            gpu: key.gpu,
            model: key.model,
            quant: key.quant,
            ...(key.linearCompute === "preset" ? {} : { linearCompute: key.linearCompute }),
            signal,
            onRunDiagnostics: (component, diagnostics) =>
              diagnosticsRef.current?.(component, diagnostics),
            onProgress: (progress) =>
              report({
                stage: "building the pipeline",
                note: `reading ${progress.path} (${mib(progress.loaded)} / ${mib(progress.total)} MiB)`,
              }),
          });
        },
      });
      const { pipeline } = acquired;
      const built: IrodoriRecord["pipeline"] = acquired.built
        ? { built: true, loadSeconds: seconds(performance.now() - buildStart) }
        : { built: false };

      const diagnostics = new Map<string, ComponentRecord>();
      const timings = new Map<string, TimingTally>();
      let stages: StageTimes<RunComponent> = {};
      const stepMarks: number[] = [];
      diagnosticsRef.current = (component, runReport) => {
        diagnostics.set(component, foldComponentDiagnostics(diagnostics.get(component), runReport));
        const tally = tallyTiming(timings.get(component), runReport.lastRunTiming);
        if (tally !== undefined) timings.set(component, tally);
      };
      // Awaited by the pipeline; a throw aborts the generation at the next stage or step (the API's
      // way to cancel — its Session is released on the way out).
      const onEvent = (event: GenerateEvent): void => {
        signal.throwIfAborted();
        const now = performance.now();
        if (event.kind === "stage") {
          stages = foldStageEvent(stages, event.component, event.at, now);
          if (event.at === "start") report({ stage: STAGE_LABELS[event.component] });
        } else {
          stepMarks.push(now);
          report({
            stage: STAGE_LABELS.dit,
            count: { done: event.step, total: event.steps, unit: "steps" },
          });
        }
      };
      try {
        signal.throwIfAborted();
        const loop = key.gpu.gpuTimingEnabled || stageEvents ? "host" : "resident";
        if (!stageEvents) {
          report({
            stage: `generating (${loop} DiT loop, no stage events)`,
            cancellable: false,
            cancelNote:
              "Stage events are off: the generation ends on its own (the API stops only at a stage or step event, and none is subscribed).",
          });
        } else {
          report({ stage: "generating" });
        }
        const generateStart = performance.now();
        const audio = await pipeline.generate(stageEvents ? { ...parts, onEvent } : parts);
        const generateEnd = performance.now();
        report({ stage: "hashing the samples" });
        const record = await assembleRecord({
          environment: recordEnvironment(key.gpu),
          distribution,
          source: sourceRecord(key.source),
          model: key.model,
          preset: {
            quant: key.quant,
            label: preset?.label ?? key.quant,
            isDefault: key.quant === manifestView.defaultQuant,
          },
          linearCompute: key.linearCompute,
          linearComputeResolved,
          request: recordedRequest,
          ditLoop: loop,
          stageEvents,
          pipeline: built,
          diagnostics: Object.fromEntries(
            [...diagnostics].map(([component, summary]) => {
              const tally = timings.get(component);
              return [
                component,
                tally === undefined ? summary : { ...summary, timing: timingOf(tally) },
              ];
            }),
          ),
          stages,
          stepMarks,
          runStart,
          generateStart,
          generateEnd,
          audio,
        });
        // A Cancel pressed while the record was assembled still ends the run as cancelled (the
        // button was there); the held pipeline stays reusable.
        signal.throwIfAborted();
        runIdRef.current += 1;
        return {
          id: runIdRef.current,
          record,
          samples: audio.data,
          sampleRate: audio.sampleRate,
        };
      } finally {
        diagnosticsRef.current = undefined;
      }
    });
  };

  const result = task.state.kind === "done" ? task.state.result : undefined;
  const disabled = running;
  const limits = manifestView === undefined ? undefined : referenceLimits(manifestView.config);

  const manifestText = (() => {
    if (index.kind === "loading" || manifest.kind === "loading") return "loading…";
    if (manifest.kind === "missing") return `missing: ${manifest.directory}`;
    if (manifest.kind === "error" || view?.kind === "error") return "failed to load";
    if (manifestView === undefined) return "loading…";
    return `${manifestView.model}: ${manifestView.presets.length} presets, ${manifestView.config.steps} steps, ${manifestView.config.sampleRate} Hz`;
  })();

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap items-baseline gap-3">
        <h1 className="text-2xl font-semibold">Irodori-TTS — text → speech</h1>
        <Badge id="irodori-status" variant={manifestView === undefined ? "outline" : "secondary"}>
          {manifestText}
        </Badge>
      </div>

      {index.kind === "error" ? (
        <ErrorAlert title="Distribution index" error={index.error} />
      ) : null}
      <MountBadges index={index} mounts={["irodori-v4.1", "irodori-v4"]} />
      {manifest.kind === "missing" ? (
        <Alert>
          <AlertTitle>The distribution is missing</AlertTitle>
          <AlertDescription>
            The dev server has no <code>{manifest.directory}</code> (a directory with a{" "}
            <code>karume.json</code>). Put the distribution there and reload, or choose the other
            model.
          </AlertDescription>
        </Alert>
      ) : null}
      {manifest.kind === "error" ? <ErrorAlert title="Manifest" error={manifest.error} /> : null}
      {view?.kind === "error" ? <ErrorAlert title="Manifest" error={view.error} /> : null}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <Card>
          <CardHeader>
            <CardTitle>Request</CardTitle>
            <CardDescription>
              The first Run builds the pipeline; later runs reuse it until the model, preset, linear
              compute, source or GPU changes.
            </CardDescription>
          </CardHeader>
          <CardContent className="grid gap-4 sm:grid-cols-2">
            <div className="sm:col-span-2">
              <Field
                id="irodori-model"
                label="Model"
                hint={
                  status === undefined
                    ? undefined
                    : `${status.directory}${status.available ? "" : " (missing)"}`
                }
              >
                <ChoiceSelect<IrodoriMount>
                  id="irodori-model"
                  value={form.mount}
                  choices={MODEL_CHOICES}
                  onChange={(mount) => update({ mount })}
                  disabled={disabled}
                />
              </Field>
            </div>
            <div className="sm:col-span-2">
              <PresetSelect
                id="irodori-quant"
                value={form.quant}
                presets={manifestView?.presets ?? []}
                defaultQuant={manifestView?.defaultQuant ?? ""}
                onChange={(quant) => update({ quant })}
                disabled={disabled || manifestView === undefined}
              />
            </div>
            <div className="sm:col-span-2">
              <SourcePicker
                id="irodori-source"
                value={source.kind}
                cachedAvailable={hfRef !== undefined}
                onChange={chooseSource}
                disabled={disabled}
              />
            </div>
            <div className="sm:col-span-2">
              <Field
                id="irodori-linear"
                label="Linear compute"
                hint={`linearCompute of the DiT and its context projection only (the condition encoders and the codec always run as declared). a8 runs only on int8 / int4 weights; f16 is not offered (it needs a shader-f16 device, which this page does not request).${
                  facts === undefined
                    ? ""
                    : ` This preset: DiT weights '${facts.ditWeights ?? "?"}', declared ${facts.declaredLinearCompute ?? "nothing (f32)"}; the DiT is asked for ${linearResolved}.`
                }`}
              >
                <ChoiceSelect
                  id="irodori-linear"
                  value={form.linearCompute}
                  choices={LINEAR_CHOICES}
                  onChange={(linearCompute) => update({ linearCompute })}
                  disabled={disabled}
                />
                {linearNote === undefined ? null : (
                  <p
                    id="irodori-linear-note"
                    className="text-xs text-amber-600 dark:text-amber-400"
                  >
                    {linearNote}
                  </p>
                )}
              </Field>
            </div>
            <div className="sm:col-span-2">
              <Field
                id="irodori-text"
                label="Text"
                hint={`Japanese text to speak. The pipeline normalizes it; emoji in the text steer the speaking style. Only the first ${
                  manifestView === undefined ? "maxTextLen" : manifestView.config.maxTextLen - 1
                } tokens after normalization are read: the pipeline cuts the rest without an error (upstream behaviour), and the record keeps the text as typed.`}
              >
                <Textarea
                  id="irodori-text"
                  rows={3}
                  value={form.text}
                  onChange={(event) => update({ text: event.target.value })}
                  disabled={disabled}
                />
              </Field>
            </div>
            <div className="flex flex-col gap-1.5 sm:col-span-2">
              <Label className="flex items-center gap-2 font-normal">
                <Checkbox
                  id="irodori-split-rare-tokens"
                  checked={form.splitRareTokens}
                  onCheckedChange={(checked) => update({ splitRareTokens: checked })}
                  disabled={disabled}
                />
                Split rare tokens (the model's default)
              </Label>
              <p className="text-xs text-muted-foreground">
                Checked: splitRareTokens is left out, so the model decides: rare tokens of the text
                are split into smaller pieces, as upstream does. Unchecked: false, the unsplit
                encoding. It only has an effect on distributions whose tokenizer asset carries the
                split table (v4 / v4.1 small from the revision that includes it); elsewhere both
                give the same encoding. It applies to the text, never the caption.
              </p>
            </div>
            <div className="flex flex-col gap-2 sm:col-span-2">
              <Label>Voice</Label>
              <Tabs
                value={form.voice}
                onValueChange={(value) => {
                  const next: VoiceMode | undefined =
                    value === "caption"
                      ? "caption"
                      : value === "reference"
                        ? "reference"
                        : undefined;
                  if (next !== undefined) update({ voice: next });
                }}
              >
                <TabsList>
                  <TabsTrigger value="caption" disabled={disabled}>
                    Voice design (caption)
                  </TabsTrigger>
                  <TabsTrigger value="reference" disabled={disabled}>
                    Voice clone (reference audio)
                  </TabsTrigger>
                </TabsList>
                <TabsContent value="caption" className="flex flex-col gap-2 pt-2">
                  <Textarea
                    id="irodori-caption"
                    aria-label="Caption"
                    rows={2}
                    value={form.caption}
                    onChange={(event) => update({ caption: event.target.value })}
                    disabled={disabled}
                  />
                  <div className="flex flex-wrap gap-2">
                    {CAPTION_EXAMPLES.map((example) => (
                      <Button
                        key={example.id}
                        id={`irodori-example-${example.id}`}
                        variant={form.caption === example.caption ? "secondary" : "outline"}
                        size="sm"
                        onClick={() => update({ caption: example.caption, text: example.text })}
                        disabled={disabled}
                      >
                        {example.label}
                      </Button>
                    ))}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    A Japanese description of the voice. An example fills the caption and a text
                    that suits it. Empty = no voice design (the caption condition is zero and its
                    guidance is off). Only the first{" "}
                    {manifestView === undefined
                      ? "maxCaptionLen"
                      : manifestView.config.maxCaptionLen - 1}{" "}
                    tokens are read; the rest is cut without an error.
                  </p>
                </TabsContent>
                {/* Kept mounted so the file input keeps the chosen file while the other tab is open. */}
                <TabsContent value="reference" keepMounted className="flex flex-col gap-2 pt-2">
                  <Input
                    id="irodori-reference"
                    type="file"
                    accept="audio/*,.wav"
                    onChange={(event) => setReferenceFile(event.target.files?.[0])}
                    disabled={disabled}
                  />
                  <p className="text-xs text-muted-foreground">
                    The voice is cloned from this audio. A PCM-16 / float-32 WAV at{" "}
                    {sampleRate ?? "the distribution's"} Hz is read as is (decodeWav); any other
                    file is decoded and resampled by the browser (the record says which).
                    {limits === undefined || sampleRate === undefined
                      ? null
                      : ` At least ${limits.minSamples} samples (${(limits.minSamples / sampleRate).toFixed(2)} s); only the first ${(limits.maxSamples / sampleRate).toFixed(0)} s are used.`}
                  </p>
                  {reference.kind === "decoding" ? (
                    <p className="flex items-center gap-2 text-sm text-muted-foreground">
                      <Spinner /> Decoding {reference.file}…
                    </p>
                  ) : null}
                  {reference.kind === "error" ? (
                    <ErrorAlert
                      title={`Reference audio (${reference.file})`}
                      error={reference.error}
                    />
                  ) : null}
                  {reference.kind === "ready" ? (
                    <div className="flex flex-col gap-2">
                      <KeyValueTable
                        rows={[
                          ["File", reference.file],
                          [
                            "Decoder",
                            reference.decoded.decoder.kind === "decodeWav"
                              ? "decodeWav (the file's own samples)"
                              : `browser — ${reference.decoded.decoder.reason}`,
                          ],
                          [
                            "Length",
                            `${(reference.decoded.data.length / reference.decoded.sampleRate).toFixed(2)} s · ${reference.decoded.data.length} samples`,
                          ],
                          ["f32 sha256", reference.f32Sha256],
                        ]}
                      />
                      <AudioClip
                        samples={reference.decoded.data}
                        sampleRate={reference.decoded.sampleRate}
                        filename="irodori_reference.wav"
                      />
                    </div>
                  ) : null}
                </TabsContent>
              </Tabs>
            </div>
            <Field
              id="irodori-seed"
              label="Seed"
              hint="A non-negative integer (the API's default is 0)."
            >
              <Input
                id="irodori-seed"
                inputMode="numeric"
                value={form.seed}
                onChange={(event) => update({ seed: event.target.value })}
                disabled={disabled}
              />
            </Field>
            <Field
              id="irodori-duration"
              label="Duration (s)"
              hint={`Empty = the duration predictor decides. A number skips it; the pipeline clamps it to ${
                manifestView === undefined
                  ? "the distribution's range"
                  : `${manifestView.config.minSeconds}–${manifestView.config.maxSeconds} s`
              }.`}
            >
              <Input
                id="irodori-duration"
                inputMode="decimal"
                value={form.durationSeconds}
                onChange={(event) => update({ durationSeconds: event.target.value })}
                disabled={disabled}
              />
            </Field>
            <Field
              id="irodori-codec-tile"
              label="Codec tile (latent frames)"
              hint={`Empty = the pipeline's default. Frames per codec decode — memory only, the waveform does not change. An integer above ${
                manifestView === undefined ? "two halos" : 2 * manifestView.config.codecHaloFrames
              }.`}
            >
              <Input
                id="irodori-codec-tile"
                inputMode="numeric"
                value={form.codecTileFrames}
                onChange={(event) => update({ codecTileFrames: event.target.value })}
                disabled={disabled}
              />
            </Field>
            <Field
              id="irodori-steps"
              label="Steps"
              hint="Fixed by the distribution's pipelineConfig (the API has no step knob)."
            >
              <Input
                id="irodori-steps"
                value={manifestView?.config.steps ?? ""}
                readOnly
                disabled
              />
            </Field>
            <div className="flex flex-col gap-1.5 sm:col-span-2">
              <Label className="flex items-center gap-2">
                <Switch
                  id="irodori-stage-events"
                  checked={form.stageEvents}
                  onCheckedChange={(checked) => update({ stageEvents: checked })}
                  disabled={disabled}
                />
                Stage events
              </Label>
              <p className="text-xs text-muted-foreground">
                On: per-stage seconds, the step counter and Cancel (at the next stage or step). The
                DiT loop then runs on the host path (one readback per step); the waveform is
                bit-identical. Off: no onEvent, so no stage seconds, and once synthesis starts the
                run cannot be cancelled. The DiT loop is then resident (one GPU batch, the fastest
                path) unless GPU timing is on, which forces the host loop.
              </p>
            </div>
            <div className="flex flex-col gap-2 sm:col-span-2">
              <Button id="irodori-run" className="self-start" onClick={run} disabled={!canRun}>
                Run
              </Button>
              {requestProblem === undefined ? null : (
                <p id="irodori-request-problem" className="text-sm text-muted-foreground">
                  {requestProblem}
                </p>
              )}
            </div>
          </CardContent>
        </Card>

        <RunStatusCard
          gpu={gpuState}
          onReacquire={reacquire}
          task={task}
          rows={[
            ["Manifest", manifestText],
            ["Pipeline", pipelineRow],
            ["DiT loop", ditLoop],
          ]}
        >
          <div className="flex flex-col gap-1.5">
            <Label className="flex items-center gap-2">
              <Switch
                id="irodori-gpu-timing"
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
            <p className="text-xs text-muted-foreground">
              {timestampQuery.kind === "probing" ? "Looking at the adapter… " : null}
              {timestampQuery.kind === "error"
                ? `Unavailable: the adapter could not be probed (${thrownText(timestampQuery.error)}). `
                : null}
              {timestampQuery.kind === "ready" && !timestampQuery.available
                ? "Unavailable: this adapter has no timestamp-query. "
                : null}
              Times every kernel of every stage on the GPU. Every dispatch becomes its own pass and
              the DiT loop runs on the host path, so a timed run is slower and its seconds are not
              comparable with untimed records. Switching acquires the GPU again; the next run
              rebuilds the pipeline.
            </p>
          </div>
        </RunStatusCard>
      </div>

      {result === undefined ? null : (
        <Card>
          <CardHeader>
            <CardTitle>Result</CardTitle>
            <CardDescription>
              {result.record.output.seconds} s · {result.record.output.sampleRate} Hz · real-time
              factor {result.record.output.realTimeFactor ?? "—"} · {result.record.generateSeconds}{" "}
              s to generate
            </CardDescription>
          </CardHeader>
          <CardContent className="grid gap-6 xl:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
            <div className="flex flex-col gap-3">
              <AudioClip
                key={result.id}
                samples={result.samples}
                sampleRate={result.sampleRate}
                filename={`irodori_${fileStamp(new Date(result.record.recordedAt))}.wav`}
              />
              <RecordActions
                record={result.record}
                filename={`irodori_${fileStamp(new Date(result.record.recordedAt))}.json`}
              />
            </div>
            <RecordTable record={result.record} />
          </CardContent>
        </Card>
      )}
    </div>
  );
};

const voiceText = (voice: IrodoriRecord["request"]["voice"]): string =>
  voice.kind === "caption"
    ? voice.caption === ""
      ? "no voice design"
      : `caption: ${voice.caption}`
    : `reference: ${voice.file} (${voice.seconds} s, ${voice.decoder.kind}) · ${voice.f32Sha256}`;

const RecordTable = ({ record }: { record: IrodoriRecord }) => {
  const steps = summarizeIntervals(record.timeline.denoiseStepSeconds);
  const rows: (readonly [string, string])[] = [
    [
      "Adapter",
      Object.values(record.adapter)
        .filter((value) => value !== "")
        .join(" · "),
    ],
    [
      "Source",
      record.source.kind === "cached"
        ? `cached · ${record.source.repo}@${record.source.revision?.slice(0, 12) ?? "main"}`
        : "direct",
    ],
    [
      "Model / preset",
      `${record.model} · ${record.preset.quant}${record.preset.isDefault ? " (default)" : ""} · linear ${record.linearCompute} → ${record.linearComputeResolved}`,
    ],
    [
      "DiT linear dispatches",
      record.ditLinearDispatches === null
        ? "—"
        : `${record.ditLinearDispatches.i8a8} i8a8 · ${record.ditLinearDispatches.other} other (last dit run)${
            record.linearComputeResolved === "a8" && record.ditLinearDispatches.i8a8 === 0
              ? " — a8 was asked for, but no linear ran int8 (float weights take the f32 path)"
              : ""
          }`,
    ],
    ["Voice", voiceText(record.request.voice)],
    [
      "Seed / duration / steps",
      `${record.request.seed} · ${record.request.durationSeconds} · ${record.request.steps}`,
    ],
    [
      "Rare-token split",
      record.request.splitRareTokens === "default"
        ? "the model's default (on only with the table)"
        : String(record.request.splitRareTokens),
    ],
    ["DiT loop", `${record.ditLoop}${record.device.gpuTimingEnabled ? " · GPU timing on" : ""}`],
    ["Pipeline", record.pipeline.built ? `built in ${record.pipeline.loadSeconds} s` : "reused"],
    [
      "Output",
      `${record.output.seconds} s · ${record.output.samples} samples · S ${record.output.latentFrames} · ${record.output.forwards} forwards`,
    ],
    ["Generate / wall", `${record.generateSeconds} s / ${record.wallSeconds} s`],
    ["Real-time factor", String(record.output.realTimeFactor ?? "—")],
    ...STAGE_ORDER.flatMap((stage): (readonly [string, string])[] => {
      const value = record.stageSeconds[stage];
      return value === undefined ? [] : [[`· ${STAGE_LABELS[stage]}`, `${value} s`]];
    }),
    ...(steps === undefined
      ? []
      : [
          [
            "Denoise steps",
            steps.rest === undefined
              ? `first ${steps.first} s`
              : `first ${steps.first} s · rest min ${steps.rest.min} / median ${steps.rest.median} / max ${steps.rest.max} s (${steps.rest.count})`,
          ] as const,
        ]),
    ["f32 sha256", record.output.f32Sha256],
    ...Object.entries(record.diagnostics).map(([component, summary]): readonly [string, string] => [
      `Diagnostics · ${component}`,
      `${summary.runs} runs · ${summary.weightsAllocatedGiB} GiB weights · build ${summary.build.supplyWaitMs} ms supply / ${summary.build.uploadFenceMs} ms upload`,
    ]),
    ...Object.entries(record.diagnostics).flatMap(
      ([component, summary]): (readonly [string, string])[] =>
        summary.timing === undefined
          ? []
          : [
              [
                `GPU time · ${component}`,
                `${summary.timing.totalMs} ms · ${topOpShares(summary.timing, TOP_OPS)
                  .map(({ op, share }) => `${op} ${(share * 100).toFixed(0)}%`)
                  .join(" · ")}`,
              ],
            ],
    ),
  ];
  return <KeyValueTable rows={rows} />;
};
