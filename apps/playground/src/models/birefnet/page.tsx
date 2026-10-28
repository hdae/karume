/**
 * The BiRefNet background-removal check: pick a checkpoint (HR for photos, Lucida for illustrations
 * and transparent objects), a model (the graph's input size) and a preset, cut out one image on the
 * page's GPU, look at the matte, the cut-out and the cut-out on white, and save a record of the run
 * (timings, the matte's sha256, the Session's diagnostics).
 *
 * The pipeline keeps one Session from construction to `dispose`, and construction reads the weights
 * (about 0.5–1 GiB), so it is built by the first Run and kept for the next images until the
 * selection, the source or the GPU changes (or the page is left). Before that, every selection is
 * estimated on the page's GPU from the graph descriptor alone (`./estimate.ts`), so a device whose
 * limits are too small for the intermediates is told before any weight is read.
 */

import { type AlphaMatte, Pipeline } from "@karume/models/birefnet";
import type { AcquireGpuOptions, GpuContext, SessionDiagnostics } from "@karume/runtime";
import { useEffect, useMemo, useRef, useState } from "react";

import { ChoiceSelect } from "@/components/choice-select";
import { ErrorAlert, thrownText } from "@/components/error-alert";
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
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { useDistributionIndex } from "@/hooks/use-distribution-index";
import { useGpu } from "@/hooks/use-gpu";
import { usePipelineSource } from "@/hooks/use-pipeline-source";
import { useRunTask } from "@/hooks/use-run-task";
import { useTimestampQuery } from "@/hooks/use-timestamp-query";
import { type PipelineSource, pretrainedRef } from "@/lib/distributions";
import { fileStamp } from "@/lib/download";
import { recordEnvironment } from "@/lib/environment";
import type { LoadedImage, RgbaPixels } from "@/lib/image";
import { manifestPresets, modelChoices } from "@/lib/presets";
import { type BuildCost, rgb8Sha256, seconds, sha256Hex, topOpShares } from "@/lib/record";

import { type EstimateSummary, estimateMatte, summarizeEstimate } from "./estimate";
import { compositeOnWhite, cutoutRgba, matteRgba } from "./pixels";
import { loadingProgress, SEGMENT_NOTE } from "./progress";
import { assembleRecord, type BirefnetRecord, type MatteReport } from "./record";
import {
  type CheckpointId,
  CHECKPOINTS,
  type DeviceFit,
  deviceFit,
  isCheckpointId,
  mebibytes,
  type Selection,
  selectionBytes,
  settleSelection,
} from "./settings";
import { useCheckpointManifest } from "./use-checkpoint-manifest";

/**
 * Module constants (useGpu needs stable options). No features: the family's own acquisition asks
 * for none (its presets declare no `gpuFeatures` and use no online attention), so the page's device
 * matches what `fromPretrained` without `gpu` would create. Timestamps only while "GPU timing" is on.
 */
const UNTIMED_GPU: Omit<AcquireGpuOptions, "onDeviceLost"> = {};
const TIMED_GPU: Omit<AcquireGpuOptions, "onDeviceLost"> = { gpuTiming: true };

/** How many ops the summary names when the run was timed. */
const TOP_OPS = 5;

/** What a pipeline was built for: a different value of any field needs another pipeline. */
type Build = {
  readonly gpu: GpuContext;
  readonly checkpoint: CheckpointId;
  readonly model: string;
  readonly quant: string;
  readonly source: PipelineSource;
};

const sameBuild = (left: Build, right: Build): boolean =>
  left.gpu === right.gpu &&
  left.checkpoint === right.checkpoint &&
  left.model === right.model &&
  left.quant === right.quant &&
  left.source === right.source;

/** The resident pipeline: what it was built for, how long it took to open, how many runs it did. */
type Held = {
  readonly build: Build;
  readonly pipeline: Pipeline;
  readonly loadMs: number;
  readonly runs: number;
};

type Views = {
  readonly matte: RgbaPixels;
  readonly cutout: RgbaPixels;
  readonly onWhite: RgbaPixels;
};

type RunResult = { readonly id: number; readonly record: BirefnetRecord; readonly views: Views };

/** The estimate of one build (it depends on every field of it: the GPU's limits, the selection, the source). */
type Estimate =
  | { readonly build: Build; readonly kind: "ready"; readonly summary: EstimateSummary }
  | { readonly build: Build; readonly kind: "error"; readonly error: unknown };

const buildText = (build: Build): string =>
  `${build.checkpoint} · ${build.model} · ${build.quant} · ${build.source.kind}`;

const fitText = (fit: DeviceFit): string => {
  if (fit.kind === "undeclared") return "the preset declares no device limits";
  if (fit.kind === "fits") return `meets the declared ${fit.declared.join(", ")}`;
  return `short: ${fit.shortfalls
    .map(({ name, required, granted }) => `${name} ${granted} < ${required}`)
    .join(", ")}`;
};

const estimateText = (summary: EstimateSummary): string =>
  `fits: peak ≈ ${mebibytes(summary.peakBytes)} (weights ${mebibytes(summary.weightsBytes)}, largest run ${mebibytes(summary.runBytes)})`;

export const BirefnetPage = () => {
  const index = useDistributionIndex();
  const [checkpoint, setCheckpoint] = useState<CheckpointId>("birefnet-hr");
  const status = index.kind === "ready" ? index.index[checkpoint] : undefined;
  const manifest = useCheckpointManifest(checkpoint, status);
  const { source, hfRef, choose: chooseSource } = usePipelineSource(status);

  // A device setting, not a form field: switching it acquires the GPU again (which drops the
  // resident pipeline, since it belongs to the old device).
  const [gpuTiming, setGpuTiming] = useState(false);
  const timestampQuery = useTimestampQuery();
  const { state: gpuState, lease, reacquire, track } = useGpu(gpuTiming ? TIMED_GPU : UNTIMED_GPU);
  const gpu = gpuState.kind === "ready" ? gpuState.gpu : undefined;

  // The form keeps the user's choice; the selection is what it stands for under the loaded manifest
  // (an empty or unknown choice is the manifest's default).
  const [form, setForm] = useState<Selection>({ model: "", quant: "" });
  const selection =
    manifest.kind === "ready" ? settleSelection(form, manifest.manifest) : undefined;
  const presets =
    manifest.kind === "ready" && selection !== undefined
      ? manifestPresets(manifest.manifest, selection.model)
      : undefined;
  const downloadBytes =
    manifest.kind === "ready" && selection !== undefined
      ? selectionBytes(manifest.manifest, selection)
      : undefined;
  const fit =
    manifest.kind === "ready" && selection !== undefined && gpu !== undefined
      ? deviceFit(
          manifest.manifest.models[selection.model].quants[selection.quant].requiredLimits,
          gpu.limits,
        )
      : undefined;

  const model = selection?.model;
  const quant = selection?.quant;
  const build = useMemo<Build | undefined>(
    () =>
      gpu === undefined || model === undefined || quant === undefined
        ? undefined
        : { gpu, checkpoint, model, quant, source },
    [gpu, checkpoint, model, quant, source],
  );

  // The resident pipeline. The ref is what the run reads and the cleanup releases; the state mirrors
  // it for the status card. `hold` writes both together.
  const heldRef = useRef<Held | undefined>(undefined);
  const [held, setHeld] = useState<Held | undefined>(undefined);
  const hold = (next: Held | undefined): void => {
    heldRef.current = next;
    setHeld(next);
  };
  // The build a finishing open may still hold its pipeline for: one that finishes after the build
  // changed (the device was lost or acquired again) belongs to a build the page no longer runs.
  const buildRef = useRef<Build | undefined>(undefined);
  // A failed release of a pipeline the page dropped (the run's own release fails the run instead).
  const [releaseError, setReleaseError] = useState<unknown>(undefined);
  // A new selection, source or GPU (and leaving the page) releases the resident pipeline: it was
  // built for the previous one. dispose waits for a segmentation in flight before tearing down, and
  // goes through `track`, so the GPU is destroyed only after it.
  useEffect(() => {
    buildRef.current = build;
    return () => {
      buildRef.current = undefined;
      const previous = heldRef.current;
      heldRef.current = undefined;
      setHeld(undefined);
      if (previous === undefined) return;
      track(previous.pipeline.dispose()).catch((error: unknown) => {
        // After the page is left the state is gone, so the console keeps it too.
        console.error("BiRefNet: releasing the pipeline failed", error);
        setReleaseError(error);
      });
    };
  }, [build, track]);

  // The estimate of the current build, from the graph descriptor alone (no weight is read). It is
  // kept with the build it belongs to, so a stale one never shows for a newer build.
  const [estimateState, setEstimateState] = useState<Estimate | undefined>(undefined);
  // "Estimate again" after a failure (a network error, say): a new attempt runs the effect again.
  const [estimateAttempt, setEstimateAttempt] = useState(0);
  useEffect(() => {
    if (build === undefined) return undefined;
    const controller = new AbortController();
    estimateMatte(
      pretrainedRef(build.source, build.checkpoint),
      { model: build.model, quant: build.quant },
      build.gpu.limits,
      controller.signal,
    ).then(
      (report) => {
        if (!controller.signal.aborted) {
          setEstimateState({ build, kind: "ready", summary: summarizeEstimate(report) });
        }
      },
      (error: unknown) => {
        if (!controller.signal.aborted) setEstimateState({ build, kind: "error", error });
      },
    );
    return () => controller.abort();
  }, [build, estimateAttempt]);
  const estimate = estimateState?.build === build ? estimateState : undefined;

  // `onRunDiagnostics` is fixed at construction; it reports into the running segmentation.
  const sinkRef = useRef<((report: SessionDiagnostics) => void) | undefined>(undefined);
  const [image, setImage] = useState<LoadedImage | undefined>(undefined);
  const task = useRunTask<RunResult>({ track, lease });
  const runIdRef = useRef(0);

  const run = (): void => {
    if (build === undefined || image === undefined || selection === undefined) return;
    if (manifest.kind !== "ready" || status === undefined) return;
    const target = build;
    const input = image;
    const chosen = selection;
    const { manifest: loadedManifest, sha256: manifestSha256 } = manifest;
    const { directory } = status;

    void task.start(async ({ signal, report }) => {
      const startedAt = performance.now();

      const open = async (previous: Held | undefined): Promise<Held> => {
        if (previous !== undefined) {
          hold(undefined);
          await previous.pipeline.dispose();
        }
        report({ stage: "loading", note: "reading the manifest" });
        const loadStart = performance.now();
        const pipeline = await Pipeline.fromPretrained(
          pretrainedRef(target.source, target.checkpoint),
          {
            gpu: target.gpu,
            model: target.model,
            quant: target.quant,
            signal,
            onRunDiagnostics: (diagnostics) => sinkRef.current?.(diagnostics),
            onProgress: (progress) => report(loadingProgress(progress)),
          },
        );
        if (signal.aborted) {
          await pipeline.dispose();
          signal.throwIfAborted();
        }
        if (buildRef.current !== target) {
          await pipeline.dispose();
          throw new Error(
            "The GPU changed while the pipeline was being built (device lost or acquired again), so the pipeline was released. Run again on the new GPU.",
          );
        }
        const opened: Held = {
          build: target,
          pipeline,
          loadMs: performance.now() - loadStart,
          runs: 0,
        };
        hold(opened);
        return opened;
      };

      const current = heldRef.current;
      const resident =
        current !== undefined && sameBuild(current.build, target) ? current : await open(current);
      const reused = resident === current;

      report({ stage: "segmenting", cancellable: false, cancelNote: SEGMENT_NOTE });
      const reports: MatteReport[] = [];
      sinkRef.current = (diagnostics) => reports.push(diagnostics);
      const segmentStart = performance.now();
      let matte: AlphaMatte;
      try {
        matte = await resident.pipeline.segment(input.rgb);
      } finally {
        sinkRef.current = undefined;
      }
      const segmentEnd = performance.now();
      const segmentRun = resident.runs + 1;
      if (heldRef.current === resident) hold({ ...resident, runs: segmentRun });

      report({ stage: "hashing and compositing" });
      const [matteSha256, rgbSha256] = await Promise.all([
        sha256Hex(matte.data),
        rgb8Sha256(input.rgb),
      ]);
      const views: Views = {
        matte: matteRgba(matte),
        cutout: cutoutRgba(input.rgb, matte),
        onWhite: compositeOnWhite(input.rgb, matte),
      };
      // Cancel is offered again after the segmentation (and leaving the page aborts): it lands here.
      signal.throwIfAborted();
      const record = assembleRecord({
        environment: recordEnvironment(target.gpu),
        checkpoint: target.checkpoint,
        directory,
        manifest: loadedManifest,
        manifestSha256,
        source: target.source,
        selection: chosen,
        image: {
          name: input.name,
          width: input.rgb.width,
          height: input.rgb.height,
          rgbSha256,
        },
        pipeline: { reused, loadMs: resident.loadMs, segmentRun },
        segmentMs: segmentEnd - segmentStart,
        wallMs: segmentEnd - startedAt,
        matte: { width: matte.width, height: matte.height, sha256: matteSha256 },
        report: reports.at(-1),
      });
      runIdRef.current += 1;
      return { id: runIdRef.current, record, views };
    });
  };

  const running = task.running;
  const disabled = running || manifest.kind !== "ready";
  const canRun =
    !running &&
    build !== undefined &&
    image !== undefined &&
    fit?.kind !== "short" &&
    estimate?.kind === "ready";
  const result = task.state.kind === "done" ? task.state.result : undefined;

  const pageStatus = (() => {
    if (index.kind === "loading" || (status?.available === true && manifest.kind === "loading")) {
      return "loading manifest…";
    }
    if (manifest.kind === "missing") return `disabled: ${manifest.directory} is missing`;
    if (manifest.kind === "ready") {
      return `ready: ${Object.keys(manifest.manifest.models).length} models`;
    }
    return "manifest failed to load";
  })();

  const chosenCheckpoint = CHECKPOINTS.find(({ id }) => id === checkpoint);

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap items-baseline gap-3">
        <h1 className="text-2xl font-semibold">BiRefNet — background removal</h1>
        <Badge id="birefnet-status" variant={manifest.kind === "ready" ? "secondary" : "outline"}>
          {pageStatus}
        </Badge>
      </div>
      <MountBadges index={index} mounts={["birefnet-hr", "lucida"]} />

      {index.kind === "error" ? (
        <ErrorAlert title="Distribution index" error={index.error} />
      ) : null}
      {manifest.kind === "missing" ? (
        <Alert>
          <AlertTitle>
            The {chosenCheckpoint?.label ?? checkpoint} distribution is missing
          </AlertTitle>
          <AlertDescription>
            The dev server has no <code>{manifest.directory}</code> (a directory with a{" "}
            <code>karume.json</code>). Put the distribution there and reload, or choose the other
            checkpoint.
          </AlertDescription>
        </Alert>
      ) : null}
      {manifest.kind === "error" ? <ErrorAlert title="Manifest" error={manifest.error} /> : null}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <Card>
          <CardHeader>
            <CardTitle>Request</CardTitle>
            <CardDescription>
              The first Run reads the weights and builds the Session; later runs on the same
              selection reuse it. Changing the checkpoint, model, preset or source releases it.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <Field id="birefnet-checkpoint" label="Checkpoint" hint={chosenCheckpoint?.note}>
              <ToggleGroup
                id="birefnet-checkpoint"
                variant="outline"
                spacing={0}
                value={[checkpoint]}
                onValueChange={(next) => {
                  const [id] = next;
                  if (id !== undefined && isCheckpointId(id)) setCheckpoint(id);
                }}
                disabled={running}
              >
                {CHECKPOINTS.map(({ id, label }) => {
                  const mount = index.kind === "ready" ? index.index[id] : undefined;
                  return (
                    <ToggleGroupItem key={id} value={id} disabled={mount?.available === false}>
                      {mount?.available === false ? `${label} (missing)` : label}
                    </ToggleGroupItem>
                  );
                })}
              </ToggleGroup>
            </Field>
            <Field
              id="birefnet-model"
              label="Model (input size)"
              hint="The image is resized to this square inside segment; the matte comes back at the image's own size."
            >
              <ChoiceSelect
                id="birefnet-model"
                value={selection?.model ?? ""}
                choices={manifest.kind === "ready" ? modelChoices(manifest.manifest) : []}
                onChange={(next) => setForm((current) => ({ ...current, model: next }))}
                disabled={disabled}
              />
            </Field>
            <PresetSelect
              id="birefnet-quant"
              value={selection?.quant ?? ""}
              presets={presets?.presets ?? []}
              defaultQuant={presets?.defaultQuant ?? ""}
              onChange={(next) => setForm((current) => ({ ...current, quant: next }))}
              disabled={disabled}
            />
            <SourcePicker
              id="birefnet-source"
              value={source.kind}
              cachedAvailable={hfRef !== undefined}
              onChange={chooseSource}
              disabled={disabled}
            />
            {source.kind === "direct" ? (
              <p className="text-xs text-muted-foreground">
                On Direct the load counter fills at once (there is nothing to download into a
                cache): the weights are read from /models/ while the Session is built, and that step
                takes no cancel.
              </p>
            ) : null}
            <ImageInput id="birefnet-image" image={image} onImage={setImage} disabled={running} />
            {fit?.kind === "short" ? (
              <Alert variant="destructive">
                <AlertTitle>This GPU does not meet the preset's declared limits</AlertTitle>
                <AlertDescription>
                  {fitText(fit)}. Run stays disabled so that no weights are read for a build that
                  would fail.
                </AlertDescription>
              </Alert>
            ) : null}
            {estimate?.kind === "error" ? (
              <div className="flex flex-col gap-2">
                <ErrorAlert title="Estimate" error={estimate.error} />
                <p className="text-xs text-muted-foreground">
                  The estimate opens the graph descriptor only (no weight is read) and checks every
                  intermediate against the device's binding and buffer limits with the planner the
                  Session uses. Run stays disabled: the Session would fail the same check on its
                  first segmentation, after the weights were read and uploaded.
                </p>
                <Button
                  id="birefnet-estimate-again"
                  variant="outline"
                  size="sm"
                  className="self-start"
                  onClick={() => {
                    setEstimateState(undefined);
                    setEstimateAttempt((count) => count + 1);
                  }}
                  disabled={running}
                >
                  Estimate again
                </Button>
              </div>
            ) : null}
            {releaseError === undefined ? null : (
              <ErrorAlert title="Releasing the previous pipeline failed" error={releaseError} />
            )}
            <div className="flex gap-3">
              <Button id="birefnet-run" onClick={run} disabled={!canRun}>
                Run
              </Button>
            </div>
            {image === undefined && manifest.kind === "ready" ? (
              <p className="text-sm text-muted-foreground">Choose an image to run.</p>
            ) : null}
          </CardContent>
        </Card>

        <RunStatusCard
          gpu={gpuState}
          onReacquire={reacquire}
          task={task}
          rows={[
            [
              "Pipeline",
              held !== undefined && build !== undefined && sameBuild(held.build, build)
                ? `resident: ${buildText(held.build)} (${seconds(held.loadMs)} s to open, ${held.runs} runs)`
                : "not built — Run reads the weights and builds the Session",
            ],
            ["Reads", downloadBytes === undefined ? "—" : mebibytes(downloadBytes)],
            ["Declared limits", fit === undefined ? "—" : fitText(fit)],
            [
              "Estimate",
              build === undefined
                ? "—"
                : estimate === undefined
                  ? "estimating from the graph descriptor…"
                  : estimate.kind === "ready"
                    ? estimateText(estimate.summary)
                    : "failed — see the Request card",
            ],
            [
              "Binding limit",
              gpu === undefined
                ? "—"
                : `${mebibytes(gpu.limits.maxStorageBufferBindingSize)} per binding · ${mebibytes(gpu.limits.maxBufferSize)} per buffer`,
            ],
          ]}
        >
          <div className="flex flex-col gap-1.5">
            <Label className="flex items-center gap-2">
              <Switch
                id="birefnet-gpu-timing"
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
            <p id="birefnet-gpu-timing-hint" className="text-xs text-muted-foreground">
              {timestampQuery.kind === "probing" ? "Looking at the adapter… " : null}
              {timestampQuery.kind === "error"
                ? `Unavailable: the adapter could not be probed (${thrownText(timestampQuery.error)}). `
                : null}
              {timestampQuery.kind === "ready" && !timestampQuery.available
                ? "Unavailable: this adapter has no timestamp-query. "
                : null}
              Times every kernel of the matte Session on the GPU. Every dispatch becomes its own
              pass, so a timed run is slower and its times are not comparable with untimed records.
              Switching acquires the GPU again and releases the resident pipeline.
            </p>
          </div>
        </RunStatusCard>
      </div>

      {result === undefined ? null : <ResultCard result={result} />}
    </div>
  );
};

const ResultCard = ({ result }: { result: RunResult }) => {
  const { record, views } = result;
  const stamp = fileStamp(new Date(record.recordedAt));
  const stem = `birefnet_${record.distribution.checkpoint}-${record.model.name}-${record.preset.quant}_${stamp}`;
  return (
    <Card>
      <CardHeader>
        <CardTitle>Result</CardTitle>
        <CardDescription>
          {record.output.width}×{record.output.height} · segment {record.timings.segmentSeconds} s ·
          wall {record.timings.wallSeconds} s
        </CardDescription>
      </CardHeader>
      <CardContent className="grid gap-6 xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        <Tabs defaultValue="cutout">
          <TabsList>
            <TabsTrigger value="cutout">Cut-out</TabsTrigger>
            <TabsTrigger value="matte">Matte</TabsTrigger>
            <TabsTrigger value="white">On white</TabsTrigger>
          </TabsList>
          <TabsContent value="cutout" className="flex flex-col gap-2">
            <RgbaImage image={views.cutout} alt="Cut-out over a checkerboard" />
            <p className="text-xs text-muted-foreground">
              The image with the matte as alpha, over a checkerboard. Not downloadable as PNG (the
              encoder writes opaque pixels only): save the matte or the cut-out on white.
            </p>
          </TabsContent>
          <TabsContent value="matte">
            <RgbaImage image={views.matte} alt="Alpha matte" filename={`${stem}_matte.png`} />
          </TabsContent>
          <TabsContent value="white">
            <RgbaImage
              image={views.onWhite}
              alt="Cut-out on white"
              filename={`${stem}_white.png`}
            />
          </TabsContent>
        </Tabs>
        <div className="flex flex-col gap-3">
          <RecordActions record={record} filename={`birefnet_${stamp}.json`} />
          <RecordTable record={record} />
        </div>
      </CardContent>
    </Card>
  );
};

const GIB = 2 ** 30;

const buildCostText = (cost: BuildCost): string =>
  [
    `supply wait ${seconds(cost.supplyWaitMs)} s`,
    `decode ${seconds(cost.decodeMs)} s`,
    `buffer create ${seconds(cost.bufferCreateMs)} s`,
    `write issue ${seconds(cost.writeBufferIssueMs)} s`,
    `upload fence ${seconds(cost.uploadFenceMs)} s`,
    `${(cost.uploadedBytes / GIB).toFixed(2)} GiB in ${cost.partCount} parts`,
  ].join(" · ");

const RecordTable = ({ record }: { record: BirefnetRecord }) => {
  const matte = record.diagnostics["matte"];
  return (
    <KeyValueTable
      rows={[
        [
          "Adapter",
          `${record.adapter.vendor} · ${record.adapter.architecture} · ${record.adapter.device}`,
        ],
        ["Driver", record.adapter.description],
        ["GPU timing", String(record.device.gpuTimingEnabled)],
        [
          "Source",
          record.source.kind === "cached"
            ? `cached (${record.source.repo} @ ${record.source.revision?.slice(0, 12) ?? "main"})`
            : "direct",
        ],
        ["Checkpoint", record.distribution.checkpoint],
        ["Model", `${record.model.name}${record.model.isDefault ? " (default)" : ""}`],
        ["Preset", `${record.preset.quant}${record.preset.isDefault ? " (default)" : ""}`],
        [
          "Image",
          `${record.request.image.name} · ${record.request.image.width}×${record.request.image.height}`,
        ],
        ["Image RGB sha256", record.request.image.rgbSha256],
        [
          "Pipeline",
          `${record.pipeline.reused ? "reused" : "built by this run"} · opened in ${record.pipeline.loadSeconds} s · segmentation ${record.pipeline.segmentRun}`,
        ],
        ["Segment s", String(record.timings.segmentSeconds)],
        ["Wall s", String(record.timings.wallSeconds)],
        ["Matte sha256", record.output.matteSha256],
        ...(matte === undefined
          ? []
          : ([
              [
                "Session",
                `weights ${matte.weightsAllocatedGiB} GiB · ${matte.submitCount} submits · ${matte.dispatchCount} dispatches · ${matte.geometryProfile}`,
              ],
              ["Build", buildCostText(matte.build)],
              ...(matte.timing === undefined
                ? []
                : [
                    [
                      "GPU time",
                      [
                        `${matte.timing.totalMs.toFixed(1)} ms timed`,
                        ...topOpShares(matte.timing, TOP_OPS).map(
                          ({ op, ms, share }) =>
                            `${op} ${ms.toFixed(1)} ms ${(share * 100).toFixed(1)}%`,
                        ),
                        ...(matte.timing.clampedNegativeSamples === 0
                          ? []
                          : [`${matte.timing.clampedNegativeSamples} samples clamped to 0`]),
                      ].join(" · "),
                    ] as const,
                  ]),
            ] as const)),
      ]}
    />
  );
};
