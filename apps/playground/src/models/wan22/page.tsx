/**
 * The Wan2.2 text → video check: pick a preset and the VAE tile, run one generation on the page's
 * GPU, watch the stages, play the frames and save a record of the run (timeline, build cost and
 * pipelines per component, device facts, and GPU time per kernel when "GPU timing" is on).
 */

import { clearHubCache, type HubRepoRef, loadManifest } from "@karume/hub";
import { type Prompt, frameToRgba } from "@karume/models/wan";
import type { AcquireGpuOptions } from "@karume/runtime";
import { useEffect, useEffectEvent, useRef, useState } from "react";

import { ChoiceSelect } from "@/components/choice-select";
import { ErrorAlert, thrownText } from "@/components/error-alert";
import { ExperimentalRuleSwitch } from "@/components/experimental-rule-switch";
import { Field } from "@/components/field";
import { FramePlayer } from "@/components/frame-player";
import { KeyValueTable } from "@/components/key-value-table";
import { PresetSelect } from "@/components/preset-select";
import { RecordActions } from "@/components/record-actions";
import { GpuAlerts } from "@/components/run-status-card";
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
import { usePipelineSource } from "@/hooks/use-pipeline-source";
import { useTimestampQuery } from "@/hooks/use-timestamp-query";
import { distributionSource, type PipelineSource } from "@/lib/distributions";
import { downloadBlob, fileStamp } from "@/lib/download";
import { recordEnvironment } from "@/lib/environment";
import {
  experimentalRuleInUse,
  experimentalRuleLabel,
  runsOnMetal,
  VAE_RMS_SILU_EXPERIMENT,
} from "@/lib/experimental-rule";
import { type CacheInventory, queryCacheInventory } from "@/lib/hub-cache";
import {
  adapterText,
  attentionPipelines,
  type BuildCost,
  type ComponentRecord,
  type ComponentTiming,
  countNonFinite,
  type DeviceFacts,
  foldComponentDiagnostics,
  intervalSeconds,
  type RgbaStack,
  seconds,
  sha256Hex,
  type Spread,
  stackRgbaFrames,
  summarizeAdapter,
  summarizeIntervals,
  tallyTiming,
  timingOf,
  type TimingTally,
  topOpShares,
  totalBuildCost,
} from "@/lib/record";
import { zipStored } from "@/lib/zip";

import {
  matchingScenario,
  type ScenarioContext,
  type ScenarioId,
  SCENARIOS,
  type ScenarioSettings,
  scenarioSettings,
  ScenarioUnavailableError,
} from "./scenarios";
import {
  buildRequest,
  FRAME_CHOICES,
  initialForm,
  quickCheck,
  SIZES,
  summarizeManifest,
  type TextEncoderPath,
  type VaeTile,
  type WanForm,
  type WanManifestSummary,
} from "./settings";
import { type DiagnosticsSink, useWanPipeline } from "./use-wan-pipeline";

/**
 * Module constants (useGpu needs stable options): subgroups when the adapter has them (the
 * online-attention presets use them), and GPU timestamps when the "GPU timing" switch is on.
 */
const UNTIMED_GPU: Omit<AcquireGpuOptions, "onDeviceLost"> = { subgroups: "available" };
const TIMED_GPU: Omit<AcquireGpuOptions, "onDeviceLost"> = {
  subgroups: "available",
  gpuTiming: true,
};

/** How many ops the summary names per timed component. */
const TOP_OPS = 5;

type StageName = "vae_encoder" | "text_encoder" | "transformer" | "vae_decoder";
type StageTimes = Partial<Record<StageName, { readonly start: number; readonly end?: number }>>;

type Progress = {
  readonly startedAt: number;
  readonly stages: StageTimes;
  readonly current?: StageName;
  readonly step?: { readonly step: number; readonly steps: number };
  readonly tile?: { readonly tile: number; readonly tiles: number };
  readonly note?: string;
};

type RunState =
  | { readonly kind: "idle" }
  | { readonly kind: "running"; readonly progress: Progress }
  | { readonly kind: "cancelled" }
  | { readonly kind: "error"; readonly error: unknown };

type CacheState =
  | { readonly kind: "idle" }
  | { readonly kind: "busy"; readonly detail: string }
  | { readonly kind: "ready"; readonly inventory: CacheInventory }
  | { readonly kind: "error"; readonly error: unknown };

/**
 * The scenario control. `waiting`: "Run this scenario" applied the settings and the run starts once
 * the pipeline for them is open (applying can rebuild it).
 */
type ScenarioRunState =
  | { readonly kind: "idle" }
  | { readonly kind: "waiting"; readonly id: ScenarioId }
  | { readonly kind: "refused"; readonly message: string }
  | { readonly kind: "cancelled"; readonly id: ScenarioId; readonly reason: string };

type ManifestState =
  | { readonly kind: "loading" }
  | { readonly kind: "missing"; readonly directory: string }
  | { readonly kind: "ready"; readonly summary: WanManifestSummary; readonly sha256: string }
  | { readonly kind: "error"; readonly error: unknown };

/** The version of the saved record; bump it when a field changes meaning or goes away. */
const RECORD_FORMAT = "karume-playground-wan22/2";

/** The saved record of one run (JSON; the page shows a summary of it). */
type WanRecord = {
  readonly format: typeof RECORD_FORMAT;
  readonly page: "wan22";
  readonly recordedAt: string;
  readonly userAgent: string;
  readonly crossOriginIsolated: boolean;
  readonly adapter: ReturnType<typeof summarizeAdapter>;
  readonly device: DeviceFacts;
  readonly distribution: { readonly directory: string; readonly manifestSha256: string };
  /** `cached` = the HF-compatible route through the hub's CacheStorage; `direct` = plain fetch. */
  readonly source: { readonly kind: "direct" } | ({ readonly kind: "cached" } & HubRepoRef);
  readonly preset: { readonly quant: string; readonly label: string; readonly isDefault: boolean };
  /**
   * The experimental fusion rule the pipeline was opened with (`fusionRules`), or `null` when none
   * ran: the stage seconds and the wall time are those of that plan.
   */
  readonly experimentalFusionRule: string | null;
  /** The scenario the run's settings equal, or `custom`. */
  readonly scenario: ScenarioId | "custom";
  readonly textEncoder: TextEncoderPath;
  readonly prompt: string;
  readonly negativePrompt?: string;
  readonly promptName?: string;
  readonly vaeTile: VaeTile;
  readonly width: number;
  readonly height: number;
  readonly frames: number;
  readonly steps: number;
  readonly seed: number;
  readonly shift: number | "manifest default";
  readonly pipelineLoadSeconds: number;
  readonly stageSeconds: Partial<Record<StageName, number>>;
  /**
   * Seconds per denoise step and per VAE tile, in order. The first of each is measured from its
   * stage start, so it also holds the Session creation and the weight upload.
   */
  readonly timeline: {
    readonly denoiseStepSeconds: readonly number[];
    readonly vaeTileSeconds: readonly number[];
  };
  readonly wallSeconds: number;
  readonly output: {
    readonly fps: number;
    readonly nonFinite: number;
    /** sha256 of the RGBA bytes of all frames in order; null when non-finite values stop the conversion. */
    readonly rgbaSha256: string | null;
  };
  readonly diagnostics: Readonly<Record<string, ComponentRecord>>;
};

type RunResult = { readonly id: number; readonly record: WanRecord; readonly stack?: RgbaStack };

const STAGE_LABELS: Readonly<Record<StageName, string>> = {
  vae_encoder: "VAE encoder",
  text_encoder: "Text encoder (umT5)",
  transformer: "Transformer (DiT)",
  vae_decoder: "VAE decoder",
};

const STAGES: readonly StageName[] = ["vae_encoder", "text_encoder", "transformer", "vae_decoder"];

const nextTick = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 0));

export const Wan22Page = () => {
  const index = useDistributionIndex();
  const wan22 = index.kind === "ready" ? index.index["wan22"] : undefined;
  const umt5 = index.kind === "ready" ? index.index["umt5"] : undefined;
  // A device setting, not a form field: turning it on or off re-acquires the GPU (which rebuilds the
  // pipeline) and leaves the scenario matching alone.
  const [gpuTiming, setGpuTiming] = useState(false);
  const timestampQuery = useTimestampQuery();
  const { state: gpuState, lease, reacquire, track } = useGpu(gpuTiming ? TIMED_GPU : UNTIMED_GPU);
  const gpu = gpuState.kind === "ready" ? gpuState.gpu : undefined;
  // Not a form field either: the scenarios leave it alone (a scenario runs with the rule on or off),
  // and the record names the rule that ran. Changing it rebuilds the pipeline.
  const [experimentalRule, setExperimentalRule] = useState(false);
  const metal = runsOnMetal(navigator.userAgent, gpu?.adapterInfo.architecture ?? "");
  const fusionRule = experimentalRuleInUse(VAE_RMS_SILU_EXPERIMENT, experimentalRule, metal);

  // The manifest (once the dev server says the distribution is there).
  const [manifest, setManifest] = useState<ManifestState>({ kind: "loading" });
  useEffect(() => {
    if (wan22 === undefined) return undefined;
    if (!wan22.available) {
      setManifest({ kind: "missing", directory: wan22.directory });
      return undefined;
    }
    const controller = new AbortController();
    const load = async (): Promise<ManifestState> => {
      const loaded = await loadManifest(distributionSource("wan22"), {
        signal: controller.signal,
      });
      const response = await fetch("models/wan22/karume.json", { signal: controller.signal });
      if (!response.ok) throw new Error(`models/wan22/karume.json: HTTP ${response.status}`);
      const sha256 = await sha256Hex(new Uint8Array(await response.arrayBuffer()));
      return { kind: "ready", summary: summarizeManifest(loaded.manifest), sha256 };
    };
    load().then(
      (state) => {
        if (!controller.signal.aborted) setManifest(state);
      },
      (error: unknown) => {
        if (!controller.signal.aborted) setManifest({ kind: "error", error });
      },
    );
    return () => controller.abort();
  }, [wan22]);
  const summary = manifest.kind === "ready" ? manifest.summary : undefined;

  const [form, setForm] = useState<WanForm>(() => initialForm(""));
  const [scenarioId, setScenarioId] = useState<ScenarioId>("quick-check");
  const [scenarioRun, setScenarioRun] = useState<ScenarioRunState>({ kind: "idle" });
  /** A pending scenario run does not survive an edit by the user. */
  const cancelPendingRun = (reason: string): void =>
    setScenarioRun((current) =>
      current.kind === "waiting" ? { kind: "cancelled", id: current.id, reason } : current,
    );
  const update = (patch: Partial<WanForm>): void => {
    cancelPendingRun("the form was edited");
    setForm((current) => ({ ...current, ...patch }));
  };
  // Preselect the manifest's default preset once it is known.
  useEffect(() => {
    if (summary !== undefined) {
      setForm((current) =>
        current.quant === "" ? { ...current, quant: summary.defaultQuant } : current,
      );
    }
  }, [summary]);

  // Where the pipeline reads the distribution. "cached" needs the HF-compatible route of the mount;
  // clearing the cache bumps its generation (a pipeline built before holds readers of the deleted
  // entries).
  const {
    source: pipelineSource,
    hfRef,
    choose: setChosenSource,
    cacheCleared,
  } = usePipelineSource(wan22);

  const diagnosticsRef = useRef<DiagnosticsSink | undefined>(undefined);
  const pipeline = useWanPipeline({
    gpu,
    source: pipelineSource,
    quant: summary === undefined || form.quant === "" ? undefined : form.quant,
    textEncoder: form.textEncoder,
    fusionRule,
    textEncoderRepo: summary?.textEncoderRepo,
    umt5Directory: umt5,
    diagnostics: diagnosticsRef,
    track,
  });
  const prompts = pipeline.kind === "ready" ? pipeline.prompts : [];
  const positives = prompts.filter(({ role }) => role === "positive");
  // The text embedding asset belongs to the model (the same rows for every preset and path), so the
  // last opened pipeline's list still holds while the next one builds: the scenarios read it.
  const [assetPrompts, setAssetPrompts] = useState<readonly Prompt[]>([]);
  // Fill the prompt fields from the asset's examples once they are known.
  useEffect(() => {
    if (pipeline.kind !== "ready") return;
    setAssetPrompts(pipeline.prompts);
    const first = pipeline.prompts.find(({ role }) => role === "positive");
    if (first === undefined) return;
    setForm((current) => ({
      ...current,
      promptName: pipeline.prompts.some(({ name }) => name === current.promptName)
        ? current.promptName
        : first.name,
      prompt: current.prompt === "" ? first.prompt : current.prompt,
    }));
  }, [pipeline]);

  const [run, setRun] = useState<RunState>({ kind: "idle" });
  const [result, setResult] = useState<RunResult | undefined>(undefined);
  const [saving, setSaving] = useState<string | undefined>(undefined);
  const controllerRef = useRef<AbortController | undefined>(undefined);
  const runIdRef = useRef(0);
  const running = run.kind === "running";

  // What the hub's cache holds of the distribution (re-read after each run and after clearing).
  const [cache, setCache] = useState<CacheState>({ kind: "idle" });
  const [cacheVersion, setCacheVersion] = useState(0);
  useEffect(() => {
    if (hfRef === undefined) return undefined;
    const controller = new AbortController();
    setCache({ kind: "busy", detail: "reading the cache inventory…" });
    queryCacheInventory(hfRef, controller.signal).then(
      (inventory) => {
        if (!controller.signal.aborted) setCache({ kind: "ready", inventory });
      },
      (error: unknown) => {
        if (!controller.signal.aborted) setCache({ kind: "error", error });
      },
    );
    return () => controller.abort();
  }, [hfRef, cacheVersion]);
  const clearCache = async (): Promise<void> => {
    setCache({ kind: "busy", detail: "clearing…" });
    try {
      await clearHubCache();
    } catch (error) {
      setCache({ kind: "error", error });
      return;
    }
    cacheCleared();
    setCacheVersion((count) => count + 1);
  };

  // Elapsed-time ticker while a run is in flight.
  const [now, setNow] = useState(() => performance.now());
  useEffect(() => {
    if (!running) return undefined;
    const timer = setInterval(() => setNow(performance.now()), 250);
    return () => clearInterval(timer);
  }, [running]);
  // The end of the GPU lease (leaving the page, re-acquiring the GPU) cancels a run, in the same
  // commit as the release: the destroy then waits only for the run to stop.
  useEffect(() => () => controllerRef.current?.abort(), [lease]);

  let requestError: unknown;
  try {
    buildRequest(form, prompts);
  } catch (error) {
    requestError = error;
  }

  const scenarioContextOf = (
    manifestSummary: WanManifestSummary,
    assetRows: readonly Prompt[],
  ): ScenarioContext => ({
    defaultQuant: manifestSummary.defaultQuant,
    prompts: assetRows,
    cachedSource: hfRef !== undefined,
    // The same condition the pipeline's GPU path fails early on.
    gpuTextEncoder: manifestSummary.textEncoderRepo === undefined || umt5?.available === true,
  });
  const scenarioContext =
    summary === undefined ? undefined : scenarioContextOf(summary, assetPrompts);
  const matchedScenario =
    scenarioContext === undefined
      ? "custom"
      : matchingScenario(form, pipelineSource.kind, scenarioContext);

  const generate = async (): Promise<void> => {
    if (pipeline.kind !== "ready" || gpu === undefined || manifest.kind !== "ready") return;
    const parts = buildRequest(form, pipeline.prompts);
    // What the run uses: the open pipeline's preset, path and source, the form's request.
    const scenario = matchingScenario(
      { ...form, quant: pipeline.quant, textEncoder: pipeline.textEncoder },
      pipeline.source.kind,
      scenarioContextOf(manifest.summary, pipeline.prompts),
    );
    const controller = new AbortController();
    controllerRef.current = controller;
    const diagnostics = new Map<string, ComponentRecord>();
    const timings = new Map<string, TimingTally>();
    diagnosticsRef.current = (component, report) => {
      diagnostics.set(component, foldComponentDiagnostics(diagnostics.get(component), report));
      const tally = tallyTiming(timings.get(component), report.lastRunTiming);
      if (tally !== undefined) timings.set(component, tally);
    };
    const stepMarks: number[] = [];
    const tileMarks: number[] = [];
    const startedAt = performance.now();
    let progress: Progress = { startedAt, stages: {} };
    const show = (next: Progress): void => {
      progress = next;
      setRun({ kind: "running", progress });
    };
    // Drop the previous video before allocating the next one.
    setResult(undefined);
    show(progress);
    try {
      const video = await track(
        pipeline.pipeline.generate({
          ...parts,
          signal: controller.signal,
          onEvent: (event) => {
            const at = performance.now();
            if (event.kind === "stage") {
              const stages: StageTimes =
                event.at === "start"
                  ? { ...progress.stages, [event.component]: { start: at } }
                  : {
                      ...progress.stages,
                      [event.component]: {
                        start: progress.stages[event.component]?.start ?? at,
                        end: at,
                      },
                    };
              show({ ...progress, stages, current: event.component });
            } else if (event.kind === "denoise-step") {
              stepMarks.push(at);
              show({ ...progress, step: { step: event.step, steps: event.steps } });
            } else {
              tileMarks.push(at);
              show({ ...progress, tile: { tile: event.tile, tiles: event.tiles } });
            }
          },
        }),
      );
      const wall = performance.now() - startedAt;
      show({ ...progress, note: "converting frames to RGBA and hashing" });
      await nextTick();
      const nonFinite = countNonFinite(video.data);
      // `frameToRgba` refuses non-finite values; the record keeps the count instead.
      const stack =
        nonFinite === 0
          ? stackRgbaFrames(video.frames, video.width, video.height, (frame) =>
              frameToRgba(video, frame),
            )
          : undefined;
      const rgbaSha256 = stack === undefined ? null : await sha256Hex(stack.bytes);
      const stageSeconds: Partial<Record<StageName, number>> = {};
      for (const name of STAGES) {
        const times = progress.stages[name];
        if (times?.end !== undefined) stageSeconds[name] = seconds(times.end - times.start);
      }
      const preset = manifest.summary.quants.find(({ name }) => name === pipeline.quant);
      const { stages } = progress;
      const record: WanRecord = {
        format: RECORD_FORMAT,
        page: "wan22",
        ...recordEnvironment(gpu),
        distribution: { directory: wan22?.directory ?? "", manifestSha256: manifest.sha256 },
        source:
          pipeline.source.kind === "cached"
            ? { kind: "cached", ...pipeline.source.ref }
            : { kind: "direct" },
        preset: {
          quant: pipeline.quant,
          label: preset?.label ?? pipeline.quant,
          isDefault: pipeline.quant === manifest.summary.defaultQuant,
        },
        experimentalFusionRule: pipeline.fusionRule ?? null,
        scenario,
        textEncoder: pipeline.textEncoder,
        prompt: parts.prompt,
        ...(parts.negativePrompt === undefined ? {} : { negativePrompt: parts.negativePrompt }),
        ...(pipeline.textEncoder === "precomputed" ? { promptName: form.promptName } : {}),
        vaeTile: parts.vaeTile,
        width: video.width,
        height: video.height,
        frames: video.frames,
        steps: parts.steps,
        seed: parts.seed,
        shift: parts.shift ?? "manifest default",
        pipelineLoadSeconds: seconds(pipeline.loadSeconds * 1000),
        stageSeconds,
        timeline: {
          denoiseStepSeconds:
            stages.transformer === undefined
              ? []
              : intervalSeconds(stages.transformer.start, stepMarks),
          vaeTileSeconds:
            stages.vae_decoder === undefined
              ? []
              : intervalSeconds(stages.vae_decoder.start, tileMarks),
        },
        wallSeconds: seconds(wall),
        output: { fps: video.fps, nonFinite, rgbaSha256 },
        diagnostics: Object.fromEntries(
          [...diagnostics].map(([component, summary]) => {
            const tally = timings.get(component);
            return [
              component,
              tally === undefined ? summary : { ...summary, timing: timingOf(tally) },
            ];
          }),
        ),
      };
      runIdRef.current += 1;
      setResult({ id: runIdRef.current, record, ...(stack === undefined ? {} : { stack }) });
      setRun({ kind: "idle" });
    } catch (error) {
      // The pipeline rethrows our own cancellation reason unwrapped.
      setRun(error === controller.signal.reason ? { kind: "cancelled" } : { kind: "error", error });
    } finally {
      diagnosticsRef.current = undefined;
      controllerRef.current = undefined;
      // The run may have filled the cache (or failed partway through filling it).
      setCacheVersion((count) => count + 1);
    }
  };

  const saveFrames = async (current: RunResult): Promise<void> => {
    const { stack } = current;
    if (stack === undefined) return;
    try {
      // The PNG encoder lives in the models barrel; load it only when saving.
      const { encodePng } = await import("@karume/models");
      const entries = [];
      for (let frame = 0; frame < stack.frames; frame += 1) {
        setSaving(`encoding PNG ${frame + 1} / ${stack.frames}`);
        entries.push({
          name: `frame_${String(frame).padStart(3, "0")}.png`,
          data: await encodePng(stack.frame(frame), stack.width, stack.height),
        });
      }
      const stamp = fileStamp(new Date(current.record.recordedAt));
      downloadBlob(
        new Blob(zipStored(entries, new Date()), { type: "application/zip" }),
        `wan22_${stamp}_frames.zip`,
      );
    } finally {
      setSaving(undefined);
    }
  };

  const status = (() => {
    if (index.kind === "loading" || (wan22?.available === true && manifest.kind === "loading")) {
      return "loading manifest…";
    }
    if (manifest.kind === "missing") return `disabled: ${manifest.directory} is missing`;
    if (manifest.kind === "ready") return `ready: ${manifest.summary.quants.length} presets`;
    return "manifest failed to load";
  })();
  const disabled = manifest.kind !== "ready" || running;
  const canRun =
    !running &&
    manifest.kind === "ready" &&
    gpu !== undefined &&
    pipeline.kind === "ready" &&
    requestError === undefined;

  // Scenarios. Applying one can rebuild the pipeline, so "Run this scenario" does not call generate()
  // itself: it leaves a pending run that the effect below starts once the open pipeline is the one
  // built for the applied settings (or cancels when that build fails).
  const builtForForm = (built: {
    readonly source: PipelineSource;
    readonly quant: string;
    readonly textEncoder: TextEncoderPath;
    readonly fusionRule: string | undefined;
  }): boolean =>
    built.source === pipelineSource &&
    built.quant === form.quant &&
    built.textEncoder === form.textEncoder &&
    built.fusionRule === fusionRule;
  const pendingFailed =
    scenarioRun.kind === "waiting" && pipeline.kind === "error" && builtForForm(pipeline);
  const pendingStarts =
    scenarioRun.kind === "waiting" &&
    matchedScenario === scenarioRun.id &&
    pipeline.kind === "ready" &&
    builtForForm(pipeline) &&
    canRun;
  const settlePendingRun = useEffectEvent(() => {
    if (pendingFailed) {
      cancelPendingRun("the pipeline failed to build");
    } else if (pendingStarts) {
      setScenarioRun({ kind: "idle" });
      void generate();
    }
  });
  useEffect(() => {
    settlePendingRun();
  }, [pendingFailed, pendingStarts]);

  const scenarioDisabled = manifest.kind !== "ready" || running;
  const chosenScenario = SCENARIOS.find(({ id }) => id === scenarioId);
  const labelOf = (id: ScenarioId): string =>
    SCENARIOS.find((scenario) => scenario.id === id)?.label ?? id;
  const applyScenario = (andRun: boolean): void => {
    if (chosenScenario === undefined || scenarioContext === undefined) return;
    let settings: ScenarioSettings;
    try {
      settings = scenarioSettings(chosenScenario, scenarioContext);
    } catch (error) {
      if (!(error instanceof ScenarioUnavailableError)) throw error;
      setScenarioRun({ kind: "refused", message: error.message });
      return;
    }
    setForm(settings.form);
    setChosenSource(settings.source);
    setScenarioRun(andRun ? { kind: "waiting", id: chosenScenario.id } : { kind: "idle" });
  };

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap items-baseline gap-3">
        <h1 className="text-2xl font-semibold">Wan2.2 TI2V 5B — text → video</h1>
        <Badge id="wan22-status" variant={manifest.kind === "ready" ? "secondary" : "outline"}>
          {status}
        </Badge>
      </div>

      {index.kind === "error" ? (
        <ErrorAlert title="Distribution index" error={index.error} />
      ) : null}
      {manifest.kind === "missing" ? (
        <Alert>
          <AlertTitle>The Wan2.2 distribution is missing</AlertTitle>
          <AlertDescription>
            The dev server has no <code>{manifest.directory}</code> (a directory with a{" "}
            <code>karume.json</code>). Put the distribution there and reload.
          </AlertDescription>
        </Alert>
      ) : null}
      {manifest.kind === "error" ? <ErrorAlert title="Manifest" error={manifest.error} /> : null}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <Card>
          <CardHeader>
            <CardTitle>Request</CardTitle>
            <CardDescription>
              Changing the preset or the text encoder rebuilds the pipeline (no GPU work until Run).
            </CardDescription>
          </CardHeader>
          <CardContent className="grid gap-4 sm:grid-cols-2">
            <div className="flex flex-col gap-2 sm:col-span-2">
              <Field
                id="wan22-scenario"
                label="Scenario"
                hint={
                  <>
                    {chosenScenario?.hint} Default preset, GPU text encoder, prompt "boxing-cats",
                    seed 42, 1280×704, cached source. The form now:{" "}
                    <span id="wan22-scenario-match" className="font-medium text-foreground">
                      {matchedScenario === "custom" ? "custom" : labelOf(matchedScenario)}
                    </span>
                    .
                  </>
                }
              >
                <ChoiceSelect
                  id="wan22-scenario"
                  value={scenarioId}
                  choices={SCENARIOS.map(({ id, label }) => ({ value: id, label }))}
                  onChange={(id) => {
                    setScenarioId(id);
                    // A refusal or cancellation names the previous choice; a pending run stays.
                    setScenarioRun((current) =>
                      current.kind === "waiting" ? current : { kind: "idle" },
                    );
                  }}
                  disabled={scenarioDisabled}
                />
              </Field>
              <div className="flex flex-wrap gap-2">
                <Button
                  id="wan22-scenario-load"
                  variant="outline"
                  size="sm"
                  onClick={() => applyScenario(false)}
                  disabled={scenarioDisabled}
                >
                  Load
                </Button>
                <Button
                  id="wan22-scenario-run"
                  size="sm"
                  onClick={() => applyScenario(true)}
                  disabled={scenarioDisabled}
                >
                  Run this scenario
                </Button>
              </div>
              {scenarioRun.kind === "waiting" ? (
                <p id="wan22-scenario-state" className="text-sm text-muted-foreground">
                  Waiting for the pipeline… "{labelOf(scenarioRun.id)}" starts once the pipeline for
                  its settings is open.
                </p>
              ) : null}
              {scenarioRun.kind === "cancelled" ? (
                <p id="wan22-scenario-state" className="text-sm text-muted-foreground">
                  "{labelOf(scenarioRun.id)}" did not start: {scenarioRun.reason}.
                </p>
              ) : null}
              {scenarioRun.kind === "refused" ? (
                <p id="wan22-scenario-state" className="text-sm text-destructive">
                  {scenarioRun.message}.
                </p>
              ) : null}
            </div>
            <div className="sm:col-span-2">
              <PresetSelect
                id="wan22-quant"
                value={form.quant}
                presets={summary?.quants ?? []}
                defaultQuant={summary?.defaultQuant ?? ""}
                onChange={(quant) => update({ quant })}
                disabled={disabled}
              />
            </div>
            <div className="sm:col-span-2">
              <SourcePicker
                id="wan22-source"
                value={pipelineSource.kind}
                cachedAvailable={hfRef !== undefined}
                onChange={(next) => {
                  cancelPendingRun("the source was changed");
                  setChosenSource(next);
                }}
                disabled={disabled}
              />
            </div>
            <Field
              id="wan22-vae-tile"
              label="VAE tile"
              hint="32 needs about 9.4 GiB on the device during decode; the 10 GB class keeps 16."
            >
              <ToggleGroup
                id="wan22-vae-tile"
                variant="outline"
                spacing={0}
                value={[String(form.vaeTile)]}
                onValueChange={(value) => {
                  const [next] = value;
                  if (next === "16" || next === "32") update({ vaeTile: next === "16" ? 16 : 32 });
                }}
                disabled={disabled}
              >
                <ToggleGroupItem value="16">16</ToggleGroupItem>
                <ToggleGroupItem value="32">32</ToggleGroupItem>
              </ToggleGroup>
            </Field>
            <Field id="wan22-size" label="Size">
              <ChoiceSelect
                id="wan22-size"
                value={form.size}
                choices={SIZES}
                onChange={(size) => update({ size })}
                disabled={disabled}
              />
            </Field>
            <Field id="wan22-frames" label="Frames" hint="4n+1; the video plays at 24 fps.">
              <ChoiceSelect
                id="wan22-frames"
                value={String(form.frames)}
                choices={FRAME_CHOICES.map((frames) => ({
                  value: String(frames),
                  label: String(frames),
                }))}
                onChange={(frames) => update({ frames: Number(frames) })}
                disabled={disabled}
              />
            </Field>
            <Field
              id="wan22-steps"
              label="Steps"
              hint={
                <Button
                  id="wan22-quick"
                  variant="link"
                  size="sm"
                  className="h-auto p-0"
                  onClick={() => {
                    cancelPendingRun("the form was edited");
                    setForm(quickCheck);
                  }}
                  disabled={disabled}
                >
                  Quick 2-step check (2 steps, 17 frames)
                </Button>
              }
            >
              <Input
                id="wan22-steps"
                inputMode="numeric"
                value={form.steps}
                placeholder={String(summary?.defaultSteps ?? 50)}
                onChange={(event) => update({ steps: event.target.value })}
                disabled={disabled}
              />
            </Field>
            <Field id="wan22-seed" label="Seed">
              <Input
                id="wan22-seed"
                inputMode="numeric"
                value={form.seed}
                onChange={(event) => update({ seed: event.target.value })}
                disabled={disabled}
              />
            </Field>
            <Field id="wan22-shift" label="Shift" hint="Empty = the manifest's default.">
              <Input
                id="wan22-shift"
                inputMode="decimal"
                value={form.shift}
                placeholder={
                  summary?.defaultShift === undefined
                    ? "default"
                    : `default (${summary.defaultShift})`
                }
                onChange={(event) => update({ shift: event.target.value })}
                disabled={disabled}
              />
            </Field>
            <Field
              id="wan22-text-encoder"
              label="Text encoder"
              hint={
                form.textEncoder === "gpu"
                  ? "umT5 on the GPU (about 5.3 GiB of weights), any prompt."
                  : "The asset's precomputed embeddings: fixed prompts only, no umT5."
              }
            >
              <ChoiceSelect
                id="wan22-text-encoder"
                value={form.textEncoder}
                choices={[
                  { value: "precomputed", label: "precomputed" },
                  {
                    value: "gpu",
                    label: umt5?.available === false ? `gpu (${umt5.directory} is missing)` : "gpu",
                    disabled: umt5?.available === false,
                  },
                ]}
                onChange={(textEncoder) => update({ textEncoder })}
                disabled={disabled}
              />
            </Field>
            <div className="sm:col-span-2">
              <ExperimentalRuleSwitch
                id="wan22-experimental-rule"
                rule={VAE_RMS_SILU_EXPERIMENT}
                checked={experimentalRule}
                metal={metal}
                onCheckedChange={setExperimentalRule}
                disabled={disabled}
              />
            </div>
            <div className="flex flex-col gap-4 sm:col-span-2">
              {form.textEncoder === "precomputed" ? (
                <Field
                  id="wan22-prompt-name"
                  label="Prompt"
                  hint={positives.find(({ name }) => name === form.promptName)?.prompt}
                >
                  <ChoiceSelect
                    id="wan22-prompt-name"
                    value={form.promptName}
                    choices={positives.map(({ name }) => ({ value: name, label: name }))}
                    onChange={(promptName) => update({ promptName })}
                    disabled={disabled || positives.length === 0}
                  />
                </Field>
              ) : (
                <>
                  <Field id="wan22-prompt" label="Prompt">
                    <Textarea
                      id="wan22-prompt"
                      value={form.prompt}
                      onChange={(event) => update({ prompt: event.target.value })}
                      disabled={disabled}
                    />
                  </Field>
                  <Field
                    id="wan22-negative"
                    label="Negative prompt"
                    hint="Empty = the official default negative prompt."
                  >
                    <Textarea
                      id="wan22-negative"
                      value={form.negativePrompt}
                      onChange={(event) => update({ negativePrompt: event.target.value })}
                      disabled={disabled}
                    />
                  </Field>
                </>
              )}
            </div>
            <div className="flex gap-3 sm:col-span-2">
              <Button id="wan22-run" onClick={() => void generate()} disabled={!canRun}>
                Run
              </Button>
              <Button
                id="wan22-cancel"
                variant="outline"
                onClick={() => {
                  cancelPendingRun("cancelled by the user");
                  controllerRef.current?.abort(new Error("Cancelled by the user"));
                }}
                disabled={!running && scenarioRun.kind !== "waiting"}
              >
                Cancel
              </Button>
            </div>
            {requestError !== undefined &&
            manifest.kind === "ready" &&
            pipeline.kind === "ready" ? (
              <p className="text-sm text-destructive sm:col-span-2">{thrownText(requestError)}</p>
            ) : null}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Status</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <KeyValueTable
              rows={[
                [
                  "GPU",
                  gpuState.kind === "ready" ? adapterText(gpuState.gpu.adapterInfo) : gpuState.kind,
                ],
                [
                  "Cross-origin isolated",
                  String(typeof crossOriginIsolated === "boolean" && crossOriginIsolated),
                ],
                [
                  "Pipeline",
                  pipeline.kind === "ready"
                    ? `${pipeline.quant} · ${pipeline.textEncoder} · ${pipeline.source.kind} · rule ${experimentalRuleLabel(pipeline.fusionRule)} (${pipeline.loadSeconds.toFixed(1)} s to open)`
                    : pipeline.kind === "loading"
                      ? pipeline.detail
                      : pipeline.kind,
                ],
                ...(hfRef === undefined ? [] : [["Cache", cacheText(cache)] as const]),
              ]}
            />
            <div className="flex flex-col gap-1.5">
              <Label className="flex items-center gap-2">
                <Switch
                  id="wan22-gpu-timing"
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
              <p id="wan22-gpu-timing-hint" className="text-xs text-muted-foreground">
                {timestampQuery.kind === "probing" ? "Looking at the adapter… " : null}
                {timestampQuery.kind === "error"
                  ? `Unavailable: the adapter could not be probed (${thrownText(timestampQuery.error)}). `
                  : null}
                {timestampQuery.kind === "ready" && !timestampQuery.available
                  ? "Unavailable: this adapter has no timestamp-query. "
                  : null}
                Times every kernel of the text encoder and the DiT on the GPU. Every dispatch
                becomes its own pass, so a timed run is slower and its wall time is not comparable
                with untimed records. The VAE decode stage is not timed (each tile runs as one
                batch). Switching acquires the GPU again and rebuilds the pipeline.
              </p>
            </div>
            {hfRef === undefined ? null : (
              <div className="flex flex-col gap-2">
                <div className="flex flex-wrap gap-2">
                  <Button
                    id="cache-refresh"
                    variant="outline"
                    size="sm"
                    onClick={() => setCacheVersion((count) => count + 1)}
                    disabled={cache.kind === "busy" || running}
                  >
                    Refresh cache size
                  </Button>
                  <Button
                    id="cache-clear"
                    variant="outline"
                    size="sm"
                    onClick={() => void clearCache()}
                    disabled={cache.kind === "busy" || running || pipeline.kind === "loading"}
                  >
                    Clear cache
                  </Button>
                </div>
                <p className="text-xs text-muted-foreground">
                  Clearing deletes everything the hub stored for this origin; the next cached run
                  reads the weights over the network again.
                </p>
                {cache.kind === "error" ? <ErrorAlert title="Cache" error={cache.error} /> : null}
              </div>
            )}
            <GpuAlerts gpu={gpuState} onReacquire={reacquire} />
            {pipeline.kind === "error" ? (
              <ErrorAlert title="Pipeline" error={pipeline.error} />
            ) : null}
            {run.kind === "running" ? <RunProgress progress={run.progress} now={now} /> : null}
            {run.kind === "cancelled" ? (
              <Alert>
                <AlertTitle>Cancelled</AlertTitle>
                <AlertDescription>The run stopped at the next stage boundary.</AlertDescription>
              </Alert>
            ) : null}
            {run.kind === "error" ? (
              <ErrorAlert title="Generation failed" error={run.error} />
            ) : null}
          </CardContent>
        </Card>
      </div>

      {result === undefined ? null : (
        <Card>
          <CardHeader>
            <CardTitle>Result</CardTitle>
            <CardDescription>
              {result.record.width}×{result.record.height} · {result.record.frames} frames ·{" "}
              {result.record.output.fps} fps · {result.record.wallSeconds} s · experimental rule:{" "}
              {experimentalRuleLabel(result.record.experimentalFusionRule)}
            </CardDescription>
          </CardHeader>
          <CardContent className="grid gap-6 xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
            {result.stack === undefined ? (
              <Alert variant="destructive">
                <AlertTitle>{result.record.output.nonFinite} non-finite values</AlertTitle>
                <AlertDescription>
                  The frames are not converted to pixels (the conversion refuses NaN / Infinity).
                </AlertDescription>
              </Alert>
            ) : (
              <FramePlayer key={result.id} stack={result.stack} fps={result.record.output.fps} />
            )}
            <div className="flex flex-col gap-3">
              <RecordActions
                record={result.record}
                filename={`wan22_${fileStamp(new Date(result.record.recordedAt))}.json`}
              >
                <Button
                  id="record-zip"
                  variant="outline"
                  size="sm"
                  disabled={result.stack === undefined || saving !== undefined}
                  onClick={() => void saveFrames(result)}
                >
                  {saving ?? "Download frames (zip of PNGs)"}
                </Button>
              </RecordActions>
              <RecordTable record={result.record} />
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

const RunProgress = ({ progress, now }: { progress: Progress; now: number }) => (
  <div className="flex flex-col gap-2 text-sm">
    <div className="font-medium">
      Running — {((now - progress.startedAt) / 1000).toFixed(1)} s
      {progress.note === undefined ? "" : ` · ${progress.note}`}
    </div>
    <KeyValueTable
      rows={[
        ...STAGES.flatMap((name) => {
          const times = progress.stages[name];
          if (times === undefined) return [];
          const label =
            times.end === undefined
              ? `running ${((now - times.start) / 1000).toFixed(1)} s`
              : `${((times.end - times.start) / 1000).toFixed(2)} s`;
          return [[STAGE_LABELS[name], label] as const];
        }),
        [
          "Denoise step",
          progress.step === undefined ? "—" : `${progress.step.step} / ${progress.step.steps}`,
        ],
        [
          "VAE tile",
          progress.tile === undefined ? "—" : `${progress.tile.tile} / ${progress.tile.tiles}`,
        ],
      ]}
    />
  </div>
);

const GIB = 2 ** 30;

const gibibytes = (bytes: number): string => (bytes / GIB).toFixed(2);

const cacheText = (cache: CacheState): string => {
  if (cache.kind === "idle") return "—";
  if (cache.kind === "busy") return cache.detail;
  if (cache.kind === "error") return "failed";
  const { inventory } = cache;
  return `${gibibytes(inventory.cachedBytes)} of ${gibibytes(inventory.totalBytes)} GiB (${inventory.cachedFiles} / ${inventory.totalFiles} files, every preset)`;
};

const spreadText = (spread: Spread | undefined): string =>
  spread === undefined ? "—" : `${spread.min} / ${spread.median} / ${spread.max} (${spread.count})`;

const buildText = (cost: BuildCost): string =>
  [
    `supply wait ${seconds(cost.supplyWaitMs)} s`,
    `decode ${seconds(cost.decodeMs)} s`,
    `buffer create ${seconds(cost.bufferCreateMs)} s`,
    `write issue ${seconds(cost.writeBufferIssueMs)} s`,
    `upload fence ${seconds(cost.uploadFenceMs)} s`,
    `${gibibytes(cost.uploadedBytes)} GiB in ${cost.partCount} parts`,
  ].join(" · ");

/** First interval and the spread of the rest, as two rows. */
const intervalRows = (
  name: string,
  values: readonly number[],
): readonly (readonly [string, string])[] => {
  const summary = summarizeIntervals(values);
  if (summary === undefined) return [];
  return [
    [`${name} 1 s (with Session build)`, String(summary.first)],
    [`${name}s 2–${values.length} s (min / median / max, n)`, spreadText(summary.rest)],
  ];
};

const RecordTable = ({ record }: { record: WanRecord }) => (
  <KeyValueTable
    rows={[
      [
        "Adapter",
        `${record.adapter.vendor} · ${record.adapter.architecture} · ${record.adapter.device}`,
      ],
      ["Driver", record.adapter.description],
      [
        "Device",
        `subgroups ${record.device.features.includes("subgroups")} · lanes linear ${record.device.subgroupLanesLinear} · GPU timing ${record.device.gpuTimingEnabled}`,
      ],
      [
        "Source",
        record.source.kind === "cached"
          ? `cached (${record.source.repo} @ ${record.source.revision?.slice(0, 12) ?? "main"})`
          : "direct",
      ],
      ["Preset", `${record.preset.quant}${record.preset.isDefault ? " (default)" : ""}`],
      ["Experimental rule", experimentalRuleLabel(record.experimentalFusionRule)],
      [
        "Scenario",
        record.device.gpuTimingEnabled ? `${record.scenario} · GPU timing on` : record.scenario,
      ],
      ["VAE tile", String(record.vaeTile)],
      ["Size", `${record.width}×${record.height}`],
      ["Frames", String(record.frames)],
      ["Steps", String(record.steps)],
      ["Seed", String(record.seed)],
      ["Shift", String(record.shift)],
      ["Text encoder", record.textEncoder],
      ...STAGES.flatMap((name) => {
        const value = record.stageSeconds[name];
        return value === undefined ? [] : [[`${STAGE_LABELS[name]} s`, String(value)] as const];
      }),
      ["Wall s", String(record.wallSeconds)],
      ...intervalRows("Step", record.timeline.denoiseStepSeconds),
      ...intervalRows("VAE tile", record.timeline.vaeTileSeconds),
      [
        "Build, all components",
        buildText(totalBuildCost(Object.values(record.diagnostics).map(({ build }) => build))),
      ],
      ["Non-finite", String(record.output.nonFinite)],
      ["RGBA sha256", record.output.rgbaSha256 ?? "—"],
      ...Object.entries(record.diagnostics).map(
        ([component, summary]) =>
          [
            component,
            `${summary.runs} runs · weights ${summary.weightsAllocatedGiB} GiB · ${summary.submitCount} submits · build: ${buildText(summary.build)}`,
          ] as const,
      ),
      ...Object.entries(record.diagnostics).flatMap(([component, summary]) => {
        const attention = attentionPipelines(summary.lastRunPipelines ?? []);
        return attention.length === 0
          ? []
          : [
              [
                `${component} attention`,
                attention.map(({ key, dispatchCount }) => `${key} ×${dispatchCount}`).join(" · "),
              ] as const,
            ];
      }),
      ...Object.entries(record.diagnostics).flatMap(([component, summary]) =>
        summary.timing === undefined
          ? []
          : [
              [
                `${component} GPU time`,
                timingText(summary.timing, record.stageSeconds, component),
              ] as const,
            ],
      ),
    ]}
  />
);

/** The stage a component runs in (`vae_decoder_first_32` → `vae_decoder`). */
const stageOf = (component: string): StageName | undefined =>
  STAGES.find((stage) => component === stage || component.startsWith(`${stage}_`));

/** The timed total beside the stage's wall seconds, then the most expensive ops with their share. */
const timingText = (
  timing: ComponentTiming,
  stageSeconds: WanRecord["stageSeconds"],
  component: string,
): string => {
  const stage = stageOf(component);
  const wall = stage === undefined ? undefined : stageSeconds[stage];
  return [
    `${timing.totalMs.toFixed(1)} ms timed over ${timing.runs} runs${wall === undefined ? "" : ` (stage ${wall} s wall)`}`,
    ...topOpShares(timing, TOP_OPS).map(
      ({ op, ms, share }) => `${op} ${ms.toFixed(1)} ms ${(share * 100).toFixed(1)}%`,
    ),
    ...(timing.clampedNegativeSamples === 0
      ? []
      : [`${timing.clampedNegativeSamples} samples clamped to 0`]),
  ].join(" · ");
};
