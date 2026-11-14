/**
 * MOSS-Image text → image: pick a preset and a source, generate one image on the page's GPU from
 * the distribution `karume-moss-image`, and save a record of the run (request, wall times, the sha256
 * of the RGBA output). `generate` reports no stage events, so the status shows the elapsed time only.
 */

import { imageToRgba } from "@karume/models/moss";
import type { AcquireGpuOptions } from "@karume/runtime";
import { useState } from "react";

import { ErrorAlert, thrownText } from "@/components/error-alert";
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
import { Textarea } from "@/components/ui/textarea";
import { useDistributionIndex } from "@/hooks/use-distribution-index";
import { useGpu } from "@/hooks/use-gpu";
import { type ManifestState, useManifest } from "@/hooks/use-manifest";
import { usePipelineSource } from "@/hooks/use-pipeline-source";
import { useRunTask } from "@/hooks/use-run-task";
import { fileStamp } from "@/lib/download";
import { recordEnvironment } from "@/lib/environment";
import type { RgbaPixels } from "@/lib/image";
import { manifestPresets } from "@/lib/presets";
import { sha256Hex } from "@/lib/record";

import { assembleRecord, type MossImageRecord } from "./record";
import { SAMPLE_PROMPTS } from "./samples";
import {
  buildRequest,
  effectiveRequest,
  initialForm,
  modelDefaults,
  MOSS_MOUNT,
  type MossForm,
  randomSeed,
  SIZE_CHOICES,
  usesNegativePrompt,
} from "./settings";
import { useMossPipeline } from "./use-moss-pipeline";

/**
 * Module constant (useGpu needs stable options): subgroups when the device can run the 32-lane form, for the online
 * attention kernel's subgroup form (never required — without them it takes its portable form).
 */
const GPU_OPTIONS: Omit<AcquireGpuOptions, "onDeviceLost"> = { subgroups: "available" };

const MIB = 2 ** 20;

const TITLE = "MOSS-Image — text → image";

const GENERATE_NOTE =
  "generate takes no signal and reports no progress: the text encoder, the text refiner, the DiT and the VAE run in turn, and the run ends when the image is decoded.";

type RunResult = { readonly record: MossImageRecord; readonly image: RgbaPixels };

export const MossImagePage = () => {
  const index = useDistributionIndex();
  const status = index.kind === "ready" ? index.index[MOSS_MOUNT] : undefined;
  const manifestState = useManifest(MOSS_MOUNT, status);
  const { source, hfRef, choose: chooseSource } = usePipelineSource(status);
  const manifest = manifestState.kind === "ready" ? manifestState.manifest : undefined;

  const [form, setForm] = useState<MossForm>(() => initialForm(SAMPLE_PROMPTS[0]?.prompt ?? ""));
  const update = (patch: Partial<MossForm>): void =>
    setForm((current) => ({ ...current, ...patch }));

  // The distribution has one model; the page runs the manifest's `defaultModel`.
  const model = manifest?.defaultModel;
  const presets =
    manifest === undefined || model === undefined ? undefined : manifestPresets(manifest, model);
  const quant =
    presets === undefined
      ? undefined
      : presets.presets.some(({ name }) => name === form.quant)
        ? form.quant
        : presets.defaultQuant;
  const defaults =
    manifest === undefined || model === undefined ? {} : modelDefaults(manifest, model);

  const { state: gpuState, lease, reacquire, track } = useGpu(GPU_OPTIONS);
  const gpu = gpuState.kind === "ready" ? gpuState.gpu : undefined;

  const { held, releaseError, open, release } = useMossPipeline(
    { gpu, source, model, quant },
    track,
  );

  const task = useRunTask<RunResult>({ track, lease });

  let requestError: unknown;
  try {
    buildRequest(form, defaults);
  } catch (error) {
    requestError = error;
  }
  const negativeUsed = usesNegativePrompt(form, defaults);
  const disabled = task.running || manifest === undefined;
  const canRun =
    !task.running &&
    gpu !== undefined &&
    model !== undefined &&
    quant !== undefined &&
    requestError === undefined;

  const run = (): void => {
    if (
      gpu === undefined ||
      manifestState.kind !== "ready" ||
      model === undefined ||
      quant === undefined ||
      presets === undefined
    ) {
      return;
    }
    const spec = { gpu, source, model, quant };
    const directory = status?.directory ?? "";
    const quantLabel = presets.presets.find(({ name }) => name === quant)?.label ?? quant;
    const requestForm = form;
    const requestDefaults = defaults;

    void task.start(async ({ signal, report }) => {
      const parts = buildRequest(requestForm, requestDefaults);
      const effective = effectiveRequest(parts, requestDefaults);
      report({ stage: "Opening the pipeline" });
      const opened = await open(spec, signal, (progress) =>
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
      // The last point a Cancel takes effect: generate itself takes no signal.
      signal.throwIfAborted();
      report({
        stage: "Generating",
        note: "text encoder → text refiner → DiT → VAE (no per-stage progress)",
        cancellable: false,
        cancelNote: `Cancel is not available while generating: ${GENERATE_NOTE}`,
      });
      const started = performance.now();
      const image = await opened.pipeline.generate(parts);
      const generateMilliseconds = performance.now() - started;
      // The GPU's lease can end while generate runs (leaving the page, a re-acquisition): the
      // image is then dropped.
      signal.throwIfAborted();
      report({ stage: "Hashing the RGBA output" });
      const rgba = imageToRgba(image);
      const sha256 = await sha256Hex(rgba);
      const record = assembleRecord({
        environment: recordEnvironment(gpu),
        directory,
        manifestSha256: manifestState.sha256,
        source: spec.source,
        model,
        defaultModel: manifestState.manifest.defaultModel,
        quant,
        quantLabel,
        defaultQuant: presets.defaultQuant,
        request: parts,
        effective,
        openMilliseconds: opened.openMilliseconds,
        generateMilliseconds,
        output: { width: image.width, height: image.height, sha256 },
      });
      return { record, image: { data: rgba, width: image.width, height: image.height } };
    });
  };

  const result = task.state.kind === "done" ? task.state.result : undefined;

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap items-baseline gap-3">
        <h1 className="text-2xl font-semibold">{TITLE}</h1>
        <Badge id="moss-image-status" variant={manifest === undefined ? "outline" : "secondary"}>
          {manifestSummary(manifestState)}
        </Badge>
      </div>
      <MountBadges index={index} mounts={[MOSS_MOUNT]} />

      {index.kind === "error" ? (
        <ErrorAlert title="Distribution index" error={index.error} />
      ) : null}
      {manifestState.kind === "error" ? (
        <ErrorAlert title={`Manifest (${MOSS_MOUNT})`} error={manifestState.error} />
      ) : null}
      {manifestState.kind === "missing" ? (
        <Alert>
          <AlertTitle>The MOSS-Image distribution is missing</AlertTitle>
          <AlertDescription>
            The dev server has no <code>{manifestState.directory}</code>. Download{" "}
            <code>hdae/karume-moss-image</code> there, or assemble it with{" "}
            <code>tools/export-recipes/moss/distribution.py</code>.
          </AlertDescription>
        </Alert>
      ) : null}

      <Alert id="moss-image-memory">
        <AlertTitle>Slow and memory-heavy by design</AlertTitle>
        <AlertDescription>
          Every generation opens and closes its stages one after another: the text encoder (Qwen3.5,
          about 7.4 GiB of int8 weights) is uploaded, encodes the prompts and is released before the
          DiT (about 4.5 GiB) and the VAE open — about 12.3 GiB of weights read and uploaded per
          generation (on the Cached source the first run stores them in this browser and later runs
          read them from its disk; Direct reads them from <code>/models/</code> every time). At
          1024² × 50 steps a generation takes minutes on a desktop GPU; a smaller size or fewer
          steps is a quicker check. The pipeline reports no progress while it generates: the status
          shows the elapsed time only.
        </AlertDescription>
      </Alert>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <Card>
          <CardHeader>
            <CardTitle>Request</CardTitle>
            <CardDescription>
              The pipeline opens on Run and stays open for the next run with the same preset and
              source. Empty fields take the model's defaults from the manifest.
            </CardDescription>
          </CardHeader>
          <CardContent className="grid gap-4 sm:grid-cols-2">
            <div className="sm:col-span-2">
              <PresetSelect
                id="moss-image-quant"
                value={quant ?? ""}
                presets={presets?.presets ?? []}
                defaultQuant={presets?.defaultQuant ?? ""}
                onChange={(next) => update({ quant: next })}
                disabled={disabled}
              />
            </div>
            <div className="sm:col-span-2">
              <SourcePicker
                id="moss-image-source"
                value={source.kind}
                cachedAvailable={hfRef !== undefined}
                onChange={chooseSource}
                disabled={disabled}
              />
            </div>
            <div className="flex flex-col gap-2 sm:col-span-2">
              <Field
                id="moss-image-prompt"
                label="Prompt"
                hint="A descriptive sentence works best (up to 4,096 tokens)."
              >
                <Textarea
                  id="moss-image-prompt"
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
                    id={`moss-image-sample-${sample.id}`}
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
                id="moss-image-negative"
                label="Negative prompt"
                hint={
                  negativeUsed
                    ? `Empty = the manifest's default${defaults.negativePrompt === "" ? " (the empty prompt)" : ""}.`
                    : "Not used at guidance 1 (no negative branch runs); the text is kept for a guidance other than 1."
                }
              >
                <Textarea
                  id="moss-image-negative"
                  rows={2}
                  value={form.negativePrompt}
                  placeholder={
                    defaults.negativePrompt === undefined || defaults.negativePrompt === ""
                      ? undefined
                      : `default: ${defaults.negativePrompt}`
                  }
                  onChange={(event) => update({ negativePrompt: event.target.value })}
                  disabled={task.running || !negativeUsed}
                />
              </Field>
            </div>
            <Field id="moss-image-width" label="Width" hint="A positive multiple of 16.">
              <Input
                id="moss-image-width"
                inputMode="numeric"
                value={form.width}
                placeholder={placeholderOf(defaults.width)}
                onChange={(event) => update({ width: event.target.value })}
                disabled={task.running}
              />
            </Field>
            <Field id="moss-image-height" label="Height" hint="A positive multiple of 16.">
              <Input
                id="moss-image-height"
                inputMode="numeric"
                value={form.height}
                placeholder={placeholderOf(defaults.height)}
                onChange={(event) => update({ height: event.target.value })}
                disabled={task.running}
              />
            </Field>
            <div className="flex flex-wrap gap-2 sm:col-span-2">
              {SIZE_CHOICES.map(({ width, height }) => {
                const chosen = form.width === String(width) && form.height === String(height);
                return (
                  <Button
                    key={`${width}x${height}`}
                    variant={chosen ? "secondary" : "outline"}
                    size="sm"
                    onClick={() => update({ width: String(width), height: String(height) })}
                    disabled={task.running}
                  >
                    {width}×{height}
                  </Button>
                );
              })}
            </div>
            <Field
              id="moss-image-steps"
              label="Steps"
              hint="A positive integer; the schedule snaps to the 1,000-point training grid, so it may run fewer."
            >
              <Input
                id="moss-image-steps"
                inputMode="numeric"
                value={form.steps}
                placeholder={placeholderOf(defaults.steps)}
                onChange={(event) => update({ steps: event.target.value })}
                disabled={task.running}
              />
            </Field>
            <Field
              id="moss-image-guidance"
              label="Guidance"
              hint="CFG scale ≥ 0; 1 = no negative branch (one DiT pass per step)."
            >
              <Input
                id="moss-image-guidance"
                inputMode="decimal"
                value={form.guidance}
                placeholder={placeholderOf(defaults.guidance)}
                onChange={(event) => update({ guidance: event.target.value })}
                disabled={task.running}
              />
            </Field>
            <Field id="moss-image-shift" label="Shift" hint="The flow-matching shift, > 0.">
              <Input
                id="moss-image-shift"
                inputMode="decimal"
                value={form.shift}
                placeholder={placeholderOf(defaults.shift)}
                onChange={(event) => update({ shift: event.target.value })}
                disabled={task.running}
              />
            </Field>
            <Field
              id="moss-image-seed"
              label="Seed"
              hint="Karume's own noise generator: the same seed does not give upstream's image."
            >
              <div className="flex gap-2">
                <Input
                  id="moss-image-seed"
                  inputMode="numeric"
                  value={form.seed}
                  onChange={(event) => update({ seed: event.target.value })}
                  disabled={task.running}
                />
                <Button
                  id="moss-image-seed-random"
                  variant="outline"
                  onClick={() => update({ seed: String(randomSeed()) })}
                  disabled={task.running}
                >
                  Random
                </Button>
              </div>
            </Field>
            <div className="flex gap-3 sm:col-span-2">
              <Button id="moss-image-run" onClick={run} disabled={!canRun}>
                Run
              </Button>
            </div>
            <p className="text-xs text-muted-foreground sm:col-span-2">
              Cancel (in Status) works while the pipeline opens, between its files. Once generation
              starts it runs to the end: {GENERATE_NOTE}
            </p>
            {requestError !== undefined && manifest !== undefined ? (
              <p id="moss-image-request-error" className="text-sm text-destructive sm:col-span-2">
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
              "Pipeline",
              held === undefined
                ? "not open (opens on Run)"
                : `${held.model} · ${held.quant} · ${held.source} (${held.openSeconds.toFixed(1)} s to open)`,
            ],
          ]}
        >
          <div className="flex flex-col gap-1.5">
            <Button
              id="moss-image-release"
              variant="outline"
              size="sm"
              className="self-start"
              onClick={() => void release()}
              disabled={task.running || held === undefined}
            >
              Release pipeline
            </Button>
            <p className="text-xs text-muted-foreground">
              Drops the open pipeline (it holds no GPU memory between runs); the next Run opens it
              again.
            </p>
            {releaseError === undefined ? null : (
              <ErrorAlert title="Releasing the pipeline failed" error={releaseError} />
            )}
          </div>
        </RunStatusCard>
      </div>

      {result === undefined ? null : (
        <Card>
          <CardHeader>
            <CardTitle>Result</CardTitle>
            <CardDescription>
              {result.record.output.width}×{result.record.output.height} ·{" "}
              {result.record.preset.quant} · {result.record.generateSeconds} s
            </CardDescription>
          </CardHeader>
          <CardContent className="grid gap-6 xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
            <RgbaImage
              image={result.image}
              alt={`Generated image: ${result.record.request.prompt}`}
              filename={`moss-image_${fileStamp(new Date(result.record.recordedAt))}.png`}
            />
            <div className="flex flex-col gap-3">
              <RecordActions
                record={result.record}
                filename={`moss-image_${fileStamp(new Date(result.record.recordedAt))}.json`}
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

const manifestSummary = (state: ManifestState): string => {
  if (state.kind === "ready") {
    const count = Object.keys(state.manifest.models).length;
    return `${count} model${count === 1 ? "" : "s"} · default ${state.manifest.defaultModel}`;
  }
  return state.kind;
};

const RecordTable = ({ record }: { record: MossImageRecord }) => (
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
      [
        "Size",
        `${String(record.effective.width ?? "—")}×${String(record.effective.height ?? "—")}`,
      ],
      ["Steps", String(record.effective.steps ?? "—")],
      ["Guidance", String(record.effective.guidance ?? "—")],
      ["Shift", String(record.effective.shift ?? "—")],
      ["Seed", String(record.request.seed)],
      [
        "Negative prompt",
        record.effective.negativePrompt === null
          ? "not used (guidance 1)"
          : record.effective.negativePrompt === ""
            ? "(empty)"
            : record.effective.negativePrompt,
      ],
      [
        "Pipeline",
        record.pipeline.reused ? "reused" : `opened in ${String(record.pipeline.openSeconds)} s`,
      ],
      ["Generate s", String(record.generateSeconds)],
      ["RGBA sha256", record.output.rgbaSha256],
    ]}
  />
);
