/**
 * The SigLIP 2 image → embedding check: pick a model, a preset and a source, embed one or two images
 * on the page's GPU, compare the two by cosine similarity and save a record (embedding statistics and
 * the sha256 of the raw f32 bytes, build and embed seconds, the vision Session's diagnostics).
 *
 * The distribution carries only the vision tower (`Pipeline` embeds images; there is no text
 * tower, so no zero-shot labels). The pipeline creates its Session — and uploads the weights — when
 * it is built, so the page builds it on the first Run and keeps it for later runs with the same
 * source, model and preset.
 */

import { Pipeline } from "@karume/models/siglip2";
import type { AcquireGpuOptions, GpuContext, SessionDiagnostics } from "@karume/runtime";
import { useEffect, useRef, useState } from "react";

import { ChoiceSelect } from "@/components/choice-select";
import { ErrorAlert, thrownText } from "@/components/error-alert";
import { Field } from "@/components/field";
import { ImageInput } from "@/components/image-input";
import { KeyValueTable } from "@/components/key-value-table";
import { PresetSelect } from "@/components/preset-select";
import { RecordActions } from "@/components/record-actions";
import { RunStatusCard } from "@/components/run-status-card";
import { SourcePicker } from "@/components/source-picker";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
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
import { useGpu } from "@/hooks/use-gpu";
import { useManifest } from "@/hooks/use-manifest";
import { usePipelineSource } from "@/hooks/use-pipeline-source";
import { useRunTask } from "@/hooks/use-run-task";
import { useTimestampQuery } from "@/hooks/use-timestamp-query";
import { mountStatus, type PipelineSource, pretrainedRef } from "@/lib/distributions";
import { downloadBlob, fileStamp } from "@/lib/download";
import { recordEnvironment } from "@/lib/environment";
import type { LoadedImage } from "@/lib/image";
import { manifestPresets, modelChoices } from "@/lib/presets";
import { f32Bytes, f32Sha256, NO_SUBMITS, rgb8Sha256, type SubmitCounts } from "@/lib/record";

import { buildStatus, comparisonText, embeddingRows, slotLabel, summaryRows } from "./format";
import { assembleRecord, type MeasuredImage, type Siglip2Record } from "./record";
import {
  canReuse,
  effectiveSelection,
  type ImageSlot,
  modelFacts,
  modelFactsText,
  type PipelineKey,
  requestImages,
  runBlocker,
  type Selection,
} from "./request";

/**
 * Module constants (useGpu needs stable options): subgroups when the device can run the 32-lane form, and GPU
 * timestamps when the "GPU timing" switch is on.
 */
const UNTIMED_GPU: Omit<AcquireGpuOptions, "onDeviceLost"> = { subgroups: "available" };
const TIMED_GPU: Omit<AcquireGpuOptions, "onDeviceLost"> = {
  subgroups: "available",
  gpuTiming: true,
};

/**
 * The vision Session's counters after its latest run, written by the pipeline's `onRunDiagnostics`
 * (every run, failed runs of the page included), read when a run starts — the baseline of that
 * run's per-run counts.
 */
type CounterCell = { latest: SubmitCounts };

/** The status note of the last embedding of a run (Cancel could only discard its result). */
const EMBED_NOTE = "embed() takes no signal: Cancel is unavailable until the last image ends.";

/** The pipeline a run built and later runs reuse while the GPU, source, model and preset match. */
type HeldPipeline = PipelineKey<GpuContext, PipelineSource> & {
  readonly pipeline: Pipeline;
  readonly counters: CounterCell;
};

type RunResult = {
  readonly record: Siglip2Record;
  readonly embeddings: readonly {
    readonly slot: ImageSlot;
    readonly values: Float32Array<ArrayBuffer>;
  }[];
};

export const Siglip2Page = () => {
  const index = useDistributionIndex();
  const status = mountStatus(index.kind === "ready" ? index.index : undefined, "siglip2");
  const manifest = useManifest("siglip2", status);
  const ready = manifest.kind === "ready" ? manifest : undefined;

  // A device setting, not a form field: switching acquires the GPU again, which drops the held
  // pipeline (it belongs to the old device).
  const [gpuTiming, setGpuTiming] = useState(false);
  const timestampQuery = useTimestampQuery();
  const { state: gpuState, lease, reacquire, track } = useGpu(gpuTiming ? TIMED_GPU : UNTIMED_GPU);
  const gpu = gpuState.kind === "ready" ? gpuState.gpu : undefined;

  const { source, hfRef, choose: chooseSource } = usePipelineSource(status);

  // The user's choices; the effective selection falls back to the manifest's defaults.
  const [chosenModel, setChosenModel] = useState<string | undefined>(undefined);
  const [chosenQuant, setChosenQuant] = useState<string | undefined>(undefined);
  const selection: Selection | undefined =
    ready === undefined ? undefined : effectiveSelection(ready.manifest, chosenModel, chosenQuant);
  const presets =
    ready === undefined || selection === undefined
      ? undefined
      : manifestPresets(ready.manifest, selection.model);
  const facts =
    ready === undefined || selection === undefined
      ? {}
      : modelFacts(ready.manifest, selection.model);

  const [imageA, setImageA] = useState<LoadedImage | undefined>(undefined);
  const [imageB, setImageB] = useState<LoadedImage | undefined>(undefined);
  // "Remove image B" bumps the key, which remounts the field and so resets its native file input
  // (the same file can be chosen again). A decode the old field started is dropped by the check.
  const [imageBKey, setImageBKey] = useState(0);
  const imageBKeyRef = useRef(0);
  const removeImageB = (): void => {
    imageBKeyRef.current += 1;
    setImageBKey(imageBKeyRef.current);
    setImageB(undefined);
  };
  const images = requestImages(imageA, imageB);

  const task = useRunTask<RunResult>({ track, lease });
  const blocker = runBlocker({
    gpuReady: gpu !== undefined,
    manifestReady: ready !== undefined,
    running: task.running,
    images,
  });

  // The held pipeline: the ref is what a run reads and replaces; the state mirrors it for the status
  // card. Both are written only by `hold`, in the same call.
  const heldRef = useRef<HeldPipeline | undefined>(undefined);
  const [held, setHeld] = useState<HeldPipeline | undefined>(undefined);
  const hold = (next: HeldPipeline | undefined): void => {
    heldRef.current = next;
    setHeld(next);
  };
  // The GPU a build may still hold its result for: a build that finishes after the GPU changed
  // (device lost, re-acquired, timing switched) belongs to a device the page no longer runs on.
  const gpuRef = useRef<GpuContext | undefined>(undefined);
  // A new GPU or leaving the page drops the held pipeline. The GPU is the page's: dispose releases
  // only the Session (after a running embedding finishes), through `track` so the old device is
  // destroyed only after it.
  useEffect(() => {
    gpuRef.current = gpu;
    return () => {
      gpuRef.current = undefined;
      const current = heldRef.current;
      heldRef.current = undefined;
      setHeld(undefined);
      if (current === undefined) return;
      track(current.pipeline.dispose()).catch((error: unknown) => {
        console.error("SigLIP 2: releasing the pipeline failed", error);
      });
    };
  }, [gpu, track]);
  // Where `onRunDiagnostics` goes: the running run's report list (the pipeline outlives a run).
  const diagnosticsSinkRef = useRef<((diagnostics: SessionDiagnostics) => void) | undefined>(
    undefined,
  );

  const run = (): void => {
    if (
      blocker !== undefined ||
      gpu === undefined ||
      ready === undefined ||
      selection === undefined ||
      presets === undefined
    ) {
      return;
    }
    const runGpu = gpu;
    const runSource = source;
    const { model, quant } = selection;
    const presetLabel = presets.presets.find(({ name }) => name === quant)?.label ?? quant;
    const directory = status?.directory ?? "";
    const runFacts = facts;
    const runImages = images;
    void task.start(async ({ signal, report }) => {
      const startedAt = performance.now();
      const current = heldRef.current;
      let active: HeldPipeline;
      let buildMs: number | undefined;
      if (
        current !== undefined &&
        canReuse(current, { gpu: runGpu, source: runSource, model, quant })
      ) {
        active = current;
      } else {
        // One pipeline at a time: so400m is 1.7 GB of weights, so the old one goes first.
        hold(undefined);
        if (current !== undefined) {
          report({ stage: "releasing the previous pipeline" });
          await current.pipeline.dispose();
        }
        report({
          stage: "fetching the weights",
          note: `reading the manifest (${model} · ${quant} · ${runSource.kind})`,
        });
        const buildStart = performance.now();
        const counters: CounterCell = { latest: NO_SUBMITS };
        // The signal reaches the fetches only: the Session build that follows (weight upload) takes
        // none (buildStatus marks it not cancellable); leaving the page during it disposes the
        // finished pipeline below.
        const built = await Pipeline.fromPretrained(pretrainedRef(runSource, "siglip2"), {
          gpu: runGpu,
          model,
          quant,
          signal,
          onRunDiagnostics: (diagnostics) => {
            counters.latest = {
              submitCount: diagnostics.submit.submitCount,
              dispatchCount: diagnostics.submit.dispatchCount,
            };
            diagnosticsSinkRef.current?.(diagnostics);
          },
          onProgress: (progress) => report(buildStatus(progress)),
        });
        buildMs = performance.now() - buildStart;
        if (signal.aborted) {
          // Cancelled during the fetch, or the page was left while the Session was being built.
          await built.dispose();
          signal.throwIfAborted();
        }
        if (gpuRef.current !== runGpu) {
          await built.dispose();
          throw new Error(
            "The GPU changed while the pipeline was being built (device lost or acquired again), so the pipeline was released. Run again on the new GPU.",
          );
        }
        active = { pipeline: built, counters, gpu: runGpu, source: runSource, model, quant };
        hold(active);
      }

      const countersBefore = active.counters.latest;
      const reports: SessionDiagnostics[] = [];
      const embedded: { slot: ImageSlot; values: Float32Array<ArrayBuffer>; ms: number }[] = [];
      diagnosticsSinkRef.current = (diagnostics) => reports.push(diagnostics);
      try {
        for (const [position, { slot, image }] of runImages.entries()) {
          // One embedding cannot be interrupted: a Cancel lands when it ends, so it is offered only
          // while another image follows (it then skips the rest).
          signal.throwIfAborted();
          const last = position === runImages.length - 1;
          report({
            stage: `embedding ${slotLabel(slot)}`,
            count: { done: position, total: runImages.length, unit: "images" },
            ...(last
              ? { cancellable: false, cancelNote: EMBED_NOTE }
              : { note: "embed() takes no signal: a Cancel takes effect when this image ends." }),
          });
          const embedStart = performance.now();
          const values = await active.pipeline.embed(image.rgb);
          embedded.push({ slot, values, ms: performance.now() - embedStart });
        }
      } finally {
        diagnosticsSinkRef.current = undefined;
      }
      // Leaving the page during the last embedding ends the run as cancelled, not with a result.
      signal.throwIfAborted();
      const wallMs = performance.now() - startedAt;

      report({ stage: "hashing" });
      const measured: MeasuredImage[] = [];
      for (const [position, { slot, image }] of runImages.entries()) {
        const output = embedded[position];
        if (output === undefined) throw new Error(`No embedding for ${slotLabel(slot)}`);
        measured.push({
          slot,
          name: image.name,
          width: image.rgb.width,
          height: image.rgb.height,
          rgbSha256: await rgb8Sha256(image.rgb),
          embedding: output.values,
          embeddingSha256: await f32Sha256(output.values),
          embedMs: output.ms,
        });
      }
      // Cancel is offered again while hashing: it lands here.
      signal.throwIfAborted();
      const record = assembleRecord({
        environment: recordEnvironment(runGpu),
        directory,
        manifestSha256: ready.sha256,
        source: runSource,
        selection,
        facts: runFacts,
        presetLabel,
        images: measured,
        buildMs,
        wallMs,
        reports,
        countersBefore,
      });
      return {
        record,
        embeddings: embedded.map(({ slot, values }) => ({ slot, values })),
      };
    });
  };

  const result = task.state.kind === "done" ? task.state.result : undefined;
  const controlsDisabled = task.running || ready === undefined;
  const statusText =
    manifest.kind === "ready"
      ? "ready"
      : manifest.kind === "missing"
        ? "distribution missing"
        : manifest.kind;

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap items-baseline gap-3">
        <h1 className="text-2xl font-semibold">SigLIP 2 — image embeddings</h1>
        <Badge id="siglip2-status" variant={manifest.kind === "ready" ? "secondary" : "outline"}>
          {statusText}
        </Badge>
      </div>

      {index.kind === "error" ? (
        <ErrorAlert title="Distribution index" error={index.error} />
      ) : null}
      {manifest.kind === "missing" ? (
        <Alert>
          <AlertTitle>The SigLIP 2 distribution is missing</AlertTitle>
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
              The distribution has the vision tower only: each image becomes its raw{" "}
              <code>pooler_output</code> vector (not normalized). With two images the run also gives
              their cosine similarity — an image-to-image measure, not SigLIP's text-image score
              (that needs the text tower).
            </CardDescription>
          </CardHeader>
          <CardContent className="grid gap-4 sm:grid-cols-2">
            {ready === undefined || selection === undefined || presets === undefined ? null : (
              <>
                <Field
                  id="siglip2-model"
                  label="Model"
                  hint={modelFactsText(facts) === "" ? undefined : modelFactsText(facts)}
                >
                  <ChoiceSelect
                    id="siglip2-model"
                    value={selection.model}
                    choices={modelChoices(ready.manifest)}
                    onChange={setChosenModel}
                    disabled={controlsDisabled}
                  />
                </Field>
                <PresetSelect
                  id="siglip2-preset"
                  value={selection.quant}
                  presets={presets.presets}
                  defaultQuant={presets.defaultQuant}
                  onChange={setChosenQuant}
                  disabled={controlsDisabled}
                />
              </>
            )}
            <div className="sm:col-span-2">
              <SourcePicker
                id="siglip2-source"
                value={source.kind}
                cachedAvailable={hfRef !== undefined}
                onChange={chooseSource}
                disabled={task.running}
              />
            </div>
            <ImageInput
              id="siglip2-image-a"
              label="Image A"
              image={imageA}
              onImage={setImageA}
              disabled={task.running}
            />
            <div className="flex flex-col gap-2">
              <ImageInput
                key={imageBKey}
                id="siglip2-image-b"
                label="Image B (optional — compared with A)"
                image={imageB}
                onImage={(decoded) => {
                  if (imageBKey === imageBKeyRef.current) setImageB(decoded);
                }}
                disabled={task.running}
              />
              {imageB === undefined ? null : (
                <Button
                  id="siglip2-image-b-remove"
                  variant="outline"
                  size="sm"
                  className="self-start"
                  onClick={removeImageB}
                  disabled={task.running}
                >
                  Remove image B
                </Button>
              )}
            </div>
            <p className="text-xs text-muted-foreground sm:col-span-2">
              Each image is stretched to the model's input size (no crop, no padding — the aspect
              ratio is not kept), as SigLIP 2's preprocessing does.
            </p>
            <div className="flex flex-col gap-1.5 sm:col-span-2">
              <Button
                id="siglip2-run"
                className="self-start"
                onClick={run}
                disabled={blocker !== undefined}
              >
                Run
              </Button>
              <p id="siglip2-run-hint" className="text-xs text-muted-foreground">
                {blocker ?? "Ready."} The first run builds the pipeline (fetches the weights, then
                builds the Session and uploads them); later runs with the same source, model and
                preset reuse it. Cancel stops the fetch at once. The Session build and each
                embedding take no signal: Cancel is unavailable during the Session build and the
                last embedding, and a Cancel during image A takes effect when it ends (image B is
                skipped).
              </p>
            </div>
          </CardContent>
        </Card>

        <RunStatusCard
          gpu={gpuState}
          onReacquire={reacquire}
          task={task}
          rows={[
            [
              "Pipeline",
              held === undefined
                ? "none (built on Run)"
                : `${held.model} · ${held.quant} · ${held.source.kind}`,
            ],
            ["Manifest sha256", ready?.sha256 ?? "—"],
          ]}
        >
          <div className="flex flex-col gap-1.5">
            <Label className="flex items-center gap-2">
              <Switch
                id="siglip2-gpu-timing"
                checked={gpuTiming}
                onCheckedChange={(checked) => setGpuTiming(checked)}
                disabled={
                  task.running ||
                  gpuState.kind === "acquiring" ||
                  timestampQuery.kind !== "ready" ||
                  !timestampQuery.available
                }
              />
              GPU timing
            </Label>
            <p id="siglip2-gpu-timing-hint" className="text-xs text-muted-foreground">
              {timestampQuery.kind === "probing" ? "Looking at the adapter… " : null}
              {timestampQuery.kind === "error"
                ? `Unavailable: the adapter could not be probed (${thrownText(timestampQuery.error)}). `
                : null}
              {timestampQuery.kind === "ready" && !timestampQuery.available
                ? "Unavailable: this adapter has no timestamp-query. "
                : null}
              Times every kernel of the vision tower on the GPU. Every dispatch becomes its own
              pass, so a timed run is slower and its embed seconds are not comparable with untimed
              records. Switching acquires the GPU again, so the next run rebuilds the pipeline.
            </p>
          </div>
        </RunStatusCard>
      </div>

      {result === undefined ? null : <ResultCard result={result} />}
    </div>
  );
};

const ResultCard = ({ result }: { result: RunResult }) => {
  const { record } = result;
  const stamp = fileStamp(new Date(record.recordedAt));
  const rows = embeddingRows(record);
  const cosine = comparisonText(record);
  return (
    <Card>
      <CardHeader>
        <CardTitle>Result</CardTitle>
        <CardDescription>
          {record.model.name} · {record.preset.quant} · {record.request.images.length}{" "}
          {record.request.images.length === 1 ? "image" : "images"} · {record.timings.wallSeconds} s
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-6">
        {cosine === undefined ? null : (
          <div className="flex flex-col gap-1">
            <span className="text-sm text-muted-foreground">Cosine similarity (A, B)</span>
            <span id="siglip2-cosine" className="font-mono text-3xl tabular-nums">
              {cosine}
            </span>
          </div>
        )}
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead />
              {record.request.images.map(({ slot }) => (
                <TableHead key={slot}>{slotLabel(slot)}</TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map(({ label, values }) => (
              <TableRow key={label}>
                <TableCell className="font-medium whitespace-nowrap text-muted-foreground">
                  {label}
                </TableCell>
                {values.map((value, column) => (
                  <TableCell
                    key={record.request.images[column]?.slot ?? column}
                    className="font-mono text-xs break-all whitespace-normal"
                  >
                    {value}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
        <RecordActions record={record} filename={`siglip2_${stamp}.json`}>
          {result.embeddings.map(({ slot, values }) => (
            <Button
              key={slot}
              id={`siglip2-embedding-${slot}`}
              variant="outline"
              size="sm"
              onClick={() =>
                downloadBlob(
                  new Blob([f32Bytes(values)], { type: "application/octet-stream" }),
                  `siglip2_${stamp}_${slot}.f32`,
                )
              }
            >
              Download {slotLabel(slot)} embedding (raw f32)
            </Button>
          ))}
        </RecordActions>
        <KeyValueTable rows={summaryRows(record)} />
      </CardContent>
    </Card>
  );
};
