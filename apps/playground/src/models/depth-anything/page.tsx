/**
 * The Depth Anything V2 check: pick an image, estimate its relative depth on the page's GPU, look at
 * the map (inferno or grayscale), save it as a PNG and save a record of the run (sizes, timings,
 * the sha256 of the raw f32 depth, Session diagnostics, GPU time per kernel when "GPU timing" is on).
 */

import type { AcquireGpuOptions } from "@karume/runtime";
import { useMemo, useRef, useState } from "react";

import { ChoiceSelect } from "@/components/choice-select";
import { ErrorAlert } from "@/components/error-alert";
import { Field } from "@/components/field";
import { ImageInput } from "@/components/image-input";
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
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { useDistributionIndex } from "@/hooks/use-distribution-index";
import { useGpu } from "@/hooks/use-gpu";
import { useManifest } from "@/hooks/use-manifest";
import { usePipelineSource } from "@/hooks/use-pipeline-source";
import { useRunTask } from "@/hooks/use-run-task";
import { useTimestampQuery } from "@/hooks/use-timestamp-query";
import { mountStatus } from "@/lib/distributions";
import { fileStamp } from "@/lib/download";
import { recordEnvironment } from "@/lib/environment";
import type { LoadedImage } from "@/lib/image";
import { manifestPresets, modelChoices } from "@/lib/presets";
import {
  type ComponentRecord,
  f32Sha256,
  foldComponentDiagnostics,
  rgb8Sha256,
  tallyTiming,
  type TimingTally,
} from "@/lib/record";

import { type DepthColoring, depthRange, depthToRgba } from "./colormap";
import { ESTIMATING, LOAD_START, loadProgress } from "./progress";
import { buildDepthRecord, type DepthRecord, summaryRows } from "./record";
import { type DepthForm, modelInputSize, runBlocker, settleForm } from "./settings";
import { type DiagnosticsSink, useDepthPipeline } from "./use-depth-pipeline";

/**
 * Module constants (useGpu needs stable options). Untimed is the device the pipeline would acquire
 * itself (the family's presets declare no GPU features); timed adds GPU timestamps.
 */
const UNTIMED_GPU: Omit<AcquireGpuOptions, "onDeviceLost"> = {};
const TIMED_GPU: Omit<AcquireGpuOptions, "onDeviceLost"> = { gpuTiming: true };

const MOUNT = "depth-anything";

type DepthResult = {
  readonly record: DepthRecord;
  readonly depth: Float32Array<ArrayBuffer>;
  readonly width: number;
  readonly height: number;
};

export const DepthAnythingPage = () => {
  const index = useDistributionIndex();
  const status = index.kind === "ready" ? mountStatus(index.index, MOUNT) : undefined;
  const manifest = useManifest(MOUNT, status);
  const [gpuTiming, setGpuTiming] = useState(false);
  const timestampQuery = useTimestampQuery();
  const { state: gpuState, lease, reacquire, track } = useGpu(gpuTiming ? TIMED_GPU : UNTIMED_GPU);
  const gpu = gpuState.kind === "ready" ? gpuState.gpu : undefined;
  const { source, hfRef, choose: chooseSource } = usePipelineSource(status);

  // The form as chosen; what the page uses is the form settled against the manifest (empty or
  // unknown choices become the manifest's defaults), worked out on every render.
  const [form, setForm] = useState<DepthForm>({ model: "", quant: "" });
  const settled = manifest.kind === "ready" ? settleForm(form, manifest.manifest) : undefined;
  const presets =
    manifest.kind === "ready" && settled !== undefined
      ? manifestPresets(manifest.manifest, settled.model)
      : undefined;
  const entry =
    manifest.kind === "ready" && settled !== undefined
      ? manifest.manifest.models[settled.model]
      : undefined;
  const inputSize = entry === undefined ? undefined : modelInputSize(entry.pipelineConfig);

  const diagnosticsRef = useRef<DiagnosticsSink | undefined>(undefined);
  const pipeline = useDepthPipeline({
    gpu,
    source,
    model: settled?.model ?? "",
    quant: settled?.quant ?? "",
    diagnostics: diagnosticsRef,
    track,
  });

  const [image, setImage] = useState<LoadedImage | undefined>(undefined);
  const [coloring, setColoring] = useState<DepthColoring>("inferno");
  const task = useRunTask<DepthResult>({ track, lease });
  const { running } = task;
  const result = task.state.kind === "done" ? task.state.result : undefined;
  const shown = useMemo(
    () =>
      result === undefined
        ? undefined
        : {
            data: depthToRgba(result.depth, result.width, result.height, coloring),
            width: result.width,
            height: result.height,
          },
    [result, coloring],
  );

  const blocker = runBlocker({
    gpuReady: gpu !== undefined,
    manifestReady: manifest.kind === "ready",
    formSettled: settled !== undefined,
    hasImage: image !== undefined,
    running,
  });

  const run = (): void => {
    if (gpu === undefined || manifest.kind !== "ready" || settled === undefined) return;
    if (presets === undefined || image === undefined) return;
    const { manifest: parsed, sha256: manifestSha256 } = manifest;
    const quantLabel =
      presets.presets.find(({ name }) => name === settled.quant)?.label ?? settled.quant;
    void task.start(async ({ signal, report }) => {
      const wallStart = performance.now();
      if (pipeline.open === undefined) report(LOAD_START);
      const obtained = await pipeline.obtain(signal, (progress) => report(loadProgress(progress)));
      signal.throwIfAborted();
      const { key: built, value: depthPipeline } = obtained.entry;

      let diagnostics: ComponentRecord | undefined;
      let timing: TimingTally | undefined;
      diagnosticsRef.current = (session) => {
        diagnostics = foldComponentDiagnostics(diagnostics, session);
        timing = tallyTiming(timing, session.lastRunTiming);
      };
      report(ESTIMATING);
      const estimateStart = performance.now();
      const depth = await depthPipeline.estimate(image.rgb).finally(() => {
        diagnosticsRef.current = undefined;
      });
      const estimateMs = performance.now() - estimateStart;
      // The estimate ran to its end; leaving the page meanwhile drops the result here.
      signal.throwIfAborted();

      report({ stage: "hashing the output" });
      const postStart = performance.now();
      const range = depthRange(depth.data);
      const outputSha256 = await f32Sha256(depth.data);
      const inputSha256 = await rgb8Sha256(image.rgb);
      // Cancel is offered again while hashing: it lands here.
      signal.throwIfAborted();
      const now = performance.now();
      const record = buildDepthRecord(recordEnvironment(gpu), {
        directory: status?.directory ?? "",
        manifestSha256,
        source: built.source,
        model: built.model,
        defaultModel: parsed.defaultModel,
        quant: built.quant,
        quantLabel,
        defaultQuant: presets.defaultQuant,
        image: {
          name: image.name,
          width: image.rgb.width,
          height: image.rgb.height,
          rgbSha256: inputSha256,
        },
        modelInput: inputSize,
        pipelineLoadMs: obtained.built ? obtained.entry.loadMs : undefined,
        estimateMs,
        postprocessMs: now - postStart,
        wallMs: now - wallStart,
        output: { width: depth.width, height: depth.height, range },
        depthSha256: outputSha256,
        diagnostics,
        timing,
      });
      return { record, depth: depth.data, width: depth.width, height: depth.height };
    });
  };

  const manifestText = (() => {
    if (index.kind === "loading" || (status?.available === true && manifest.kind === "loading")) {
      return "loading manifest…";
    }
    if (manifest.kind === "missing") return `disabled: ${manifest.directory} is missing`;
    if (manifest.kind === "ready") {
      return `ready: ${Object.keys(manifest.manifest.models).length} models`;
    }
    return "manifest failed to load";
  })();
  const disabled = manifest.kind !== "ready" || running;
  const stamp = result === undefined ? "" : fileStamp(new Date(result.record.recordedAt));
  const { open } = pipeline;

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap items-baseline gap-3">
        <h1 className="text-2xl font-semibold">Depth Anything V2 — depth estimation</h1>
        <Badge
          id="depth-anything-status"
          variant={manifest.kind === "ready" ? "secondary" : "outline"}
        >
          {manifestText}
        </Badge>
      </div>
      <MountBadges index={index} mounts={[MOUNT]} />

      {index.kind === "error" ? (
        <ErrorAlert title="Distribution index" error={index.error} />
      ) : null}
      {manifest.kind === "missing" ? (
        <Alert>
          <AlertTitle>The Depth Anything V2 distribution is missing</AlertTitle>
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
              The first Run builds the pipeline (fetches the weights and uploads them to the GPU);
              later runs reuse it until the model, preset, source or GPU changes.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <Field id="depth-anything-model" label="Model">
              <ChoiceSelect
                id="depth-anything-model"
                value={settled?.model ?? ""}
                choices={manifest.kind === "ready" ? modelChoices(manifest.manifest) : []}
                onChange={(model) => setForm({ model, quant: "" })}
                disabled={disabled}
              />
            </Field>
            <PresetSelect
              id="depth-anything-quant"
              value={settled?.quant ?? ""}
              presets={presets?.presets ?? []}
              defaultQuant={presets?.defaultQuant ?? ""}
              onChange={(quant) => setForm({ model: settled?.model ?? "", quant })}
              disabled={disabled}
            />
            <SourcePicker
              id="depth-anything-source"
              value={source.kind}
              cachedAvailable={hfRef !== undefined}
              onChange={chooseSource}
              disabled={disabled}
            />
            <ImageInput
              id="depth-anything-image"
              image={image}
              onImage={setImage}
              disabled={running}
            />
            <p className="text-xs text-muted-foreground">
              The image goes in at its own size: the pipeline resizes it to{" "}
              {inputSize === undefined
                ? "the model's input"
                : `${inputSize.width}×${inputSize.height}`}{" "}
              (bicubic, aspect not kept) and returns the depth at the input size.
            </p>
            <div className="flex flex-col gap-1.5">
              <Button
                id="depth-anything-run"
                className="self-start"
                onClick={run}
                disabled={blocker !== undefined}
              >
                Estimate depth
              </Button>
              <p className="text-xs text-muted-foreground">
                {blocker ??
                  "Cancel stops the manifest and weight fetch. The Session build (weight upload) and the estimate take no signal: Cancel is unavailable while either runs."}
              </p>
            </div>
          </CardContent>
        </Card>

        <RunStatusCard
          gpu={gpuState}
          onReacquire={reacquire}
          task={task}
          rows={[
            ["Manifest", manifestText],
            [
              "Pipeline",
              open === undefined
                ? "not built (the first Run builds it)"
                : `open: ${open.key.model} · ${open.key.quant} · ${open.key.source.kind} · built in ${(
                    open.loadMs / 1000
                  ).toFixed(1)} s`,
            ],
            [
              "Model input",
              inputSize === undefined ? "—" : `${inputSize.width}×${inputSize.height}`,
            ],
          ]}
        >
          <div className="flex flex-col gap-1.5">
            <Label className="flex items-center gap-2">
              <Switch
                id="depth-anything-gpu-timing"
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
              {timestampQuery.kind === "ready" && !timestampQuery.available
                ? "Unavailable: this adapter does not list timestamp-query."
                : "Times every kernel of the depth run on the GPU. Every dispatch becomes its own pass, so a timed run is slower and its times are not comparable with untimed records. Switching acquires the GPU again and drops the open pipeline."}
            </p>
          </div>
        </RunStatusCard>
      </div>

      {result === undefined || shown === undefined ? null : (
        <Card>
          <CardHeader>
            <CardTitle>Result</CardTitle>
            <CardDescription>
              Relative depth, min-max normalized for display: light = near, dark = far. It has no
              unit, so the shades of two images do not compare.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <Field id="depth-anything-coloring" label="Coloring">
              <ToggleGroup
                id="depth-anything-coloring"
                variant="outline"
                spacing={0}
                value={[coloring]}
                onValueChange={(next) => {
                  const [value] = next;
                  if (value === "inferno" || value === "grayscale") setColoring(value);
                }}
              >
                <ToggleGroupItem value="inferno">Inferno</ToggleGroupItem>
                <ToggleGroupItem value="grayscale">Grayscale</ToggleGroupItem>
              </ToggleGroup>
            </Field>
            <RgbaImage
              image={shown}
              alt={`Depth map (${coloring})`}
              filename={`depth-anything_${stamp}_${coloring}.png`}
            />
            <KeyValueTable rows={summaryRows(result.record)} />
            <RecordActions record={result.record} filename={`depth-anything_${stamp}.json`} />
          </CardContent>
        </Card>
      )}
    </div>
  );
};
