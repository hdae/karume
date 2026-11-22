/**
 * The EmbeddingGemma text → sentence embedding check: a list of text rows, each with its own prompt
 * from the manifest, one dimension for the run, every row embedded on the page's GPU (one `embed`
 * call per row, in order) and the similarity matrix of the rows (the dot product of the unit
 * vectors), with a saved record (per-row statistics and the sha256 of the raw f32 bytes, build and
 * embed seconds, the model Session's diagnostics).
 *
 * A row the pipeline refuses (`ModelInputError`: an undeclared prompt, a text over the token limit,
 * a placeholder token in the text) shows the refusal as raised and the run goes on with the next
 * row; any other failure ends the run. The pipeline creates its Session — and uploads the weights —
 * when it is built, so the page builds it on the first Run and keeps it for later runs with the same
 * source, model and preset (the prompt and the dimension are per call).
 *
 * Once a pipeline is held, each row shows its token count against the model's limit `maxTokens`
 * (`tokenize` — on the host, recomputed as the row changes), marked when it is over; the record
 * keeps the count of each row as it was run. A device whose buffer limit is smaller runs fewer
 * tokens; the page does not know that limit, so such a row is refused only when it is run.
 */

import {
  mountStatus,
  type PipelineSource,
  pretrainedRef,
} from "@karume/local-distributions/browser";
import { ModelInputError, Pipeline } from "@karume/models/embeddinggemma";
import type { AcquireGpuOptions, GpuContext, SessionDiagnostics } from "@karume/runtime";
import { PlusIcon, Trash2Icon } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { ChoiceSelect } from "@/components/choice-select";
import { ErrorAlert, thrownText } from "@/components/error-alert";
import { Field } from "@/components/field";
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
import { Textarea } from "@/components/ui/textarea";
import { useDistributionIndex } from "@/hooks/use-distribution-index";
import { useGpu } from "@/hooks/use-gpu";
import { useManifest } from "@/hooks/use-manifest";
import { usePipelineSource } from "@/hooks/use-pipeline-source";
import { useRunTask } from "@/hooks/use-run-task";
import { useTimestampQuery } from "@/hooks/use-timestamp-query";
import { downloadBlob, fileStamp } from "@/lib/download";
import { recordEnvironment } from "@/lib/environment";
import { manifestPresets, modelChoices } from "@/lib/presets";
import { f32Bytes, f32Sha256, NO_SUBMITS, type SubmitCounts } from "@/lib/record";

import {
  dimensionChoices,
  effectiveDimension,
  type EmbeddingModelConfig,
  promptChoices,
  promptOfValue,
  promptValue,
  readEmbeddingConfig,
} from "./config";
import {
  buildStatus,
  firstValuesText,
  rowLabel,
  rowOutcomeText,
  similarityText,
  summaryRows,
  tokenCountView,
} from "./format";
import { assembleRecord, type EmbeddingGemmaRecord, type MeasuredRow } from "./record";
import {
  addRow,
  canReuse,
  effectiveSelection,
  type PipelineKey,
  removeRow,
  rowTokenCount,
  runBlocker,
  SAMPLE_ROWS,
  type Selection,
  type TextRow,
  updateRow,
} from "./request";

/** The mount the page reads (`@karume/local-distributions/mounts`). */
const MOUNT = "embeddinggemma-2";

/**
 * Module constants (useGpu needs stable options): subgroups when the device can run the 32-lane
 * form, and GPU timestamps when the "GPU timing" switch is on.
 */
const UNTIMED_GPU: Omit<AcquireGpuOptions, "onDeviceLost"> = { subgroups: "available" };
const TIMED_GPU: Omit<AcquireGpuOptions, "onDeviceLost"> = {
  subgroups: "available",
  gpuTiming: true,
};

/**
 * The model Session's counters after its latest run, written by the pipeline's `onRunDiagnostics`,
 * read when a run starts — the baseline of that run's per-run counts.
 */
type CounterCell = { latest: SubmitCounts };

/** The pipeline a run built and later runs reuse while the GPU, source, model and preset match. */
type HeldPipeline = PipelineKey<GpuContext, PipelineSource> & {
  readonly pipeline: Pipeline;
  readonly counters: CounterCell;
};

type RunResult = {
  readonly record: EmbeddingGemmaRecord;
  /** The vector of each embedded row (for the raw f32 downloads), by row position. */
  readonly vectors: readonly { readonly row: number; readonly values: Float32Array<ArrayBuffer> }[];
};

/** The model's `pipelineConfig` as the page reads it, or why it cannot. */
type ConfigState =
  | { readonly kind: "ready"; readonly config: EmbeddingModelConfig }
  | { readonly kind: "error"; readonly error: unknown };

export const EmbeddingGemmaPage = () => {
  const index = useDistributionIndex();
  const status = mountStatus(index.kind === "ready" ? index.index : undefined, MOUNT);
  const manifest = useManifest(MOUNT, status);
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
  const [chosenDimension, setChosenDimension] = useState<number | undefined>(undefined);
  const selection: Selection | undefined =
    ready === undefined ? undefined : effectiveSelection(ready.manifest, chosenModel, chosenQuant);
  const presets =
    ready === undefined || selection === undefined
      ? undefined
      : manifestPresets(ready.manifest, selection.model);
  let configState: ConfigState | undefined;
  if (ready !== undefined && selection !== undefined) {
    try {
      configState = { kind: "ready", config: readEmbeddingConfig(ready.manifest, selection.model) };
    } catch (error) {
      configState = { kind: "error", error };
    }
  }
  const config = configState?.kind === "ready" ? configState.config : undefined;
  const dimension = config === undefined ? undefined : effectiveDimension(config, chosenDimension);

  const [rows, setRows] = useState<readonly TextRow[]>(SAMPLE_ROWS);

  const task = useRunTask<RunResult>({ track, lease });
  const blocker = runBlocker({
    gpuReady: gpu !== undefined,
    manifestReady: ready !== undefined,
    configError: configState?.kind === "error",
    running: task.running,
    rows,
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
  // belongs to a device the page no longer runs on.
  const gpuRef = useRef<GpuContext | undefined>(undefined);
  // A new GPU or leaving the page drops the held pipeline. The GPU is the page's: dispose releases
  // only the Session (after a running call finishes), through `track` so the old device is
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
        console.error("EmbeddingGemma: releasing the pipeline failed", error);
      });
    };
  }, [gpu, track]);
  // Each row's token count from the held pipeline's tokenizer, recomputed when a row's text or
  // prompt changes. None before the first build (the page never builds a pipeline just to count)
  // and none while the held pipeline is another model's (the next Run builds the selected one).
  const selectedModel = selection?.model;
  const tokenCounts = useMemo(() => {
    const counter = held !== undefined && held.model === selectedModel ? held.pipeline : undefined;
    return rows.map((row) =>
      counter === undefined
        ? undefined
        : rowTokenCount((text, options) => counter.tokenize(text, options), row),
    );
  }, [held, rows, selectedModel]);
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
      presets === undefined ||
      dimension === undefined
    ) {
      return;
    }
    const runGpu = gpu;
    const runSource = source;
    const runDimension = dimension;
    const runRows = rows;
    const { model, quant } = selection;
    const presetLabel = presets.presets.find(({ name }) => name === quant)?.label ?? quant;
    const directory = status?.directory ?? "";
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
        // One pipeline at a time: the old one goes first.
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
        const built = await Pipeline.fromPretrained(
          pretrainedRef(runSource, MOUNT, document.baseURI),
          {
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
          },
        );
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
      const outcomes: {
        readonly tokens: number | undefined;
        readonly embedMs: number;
        readonly result:
          | { readonly kind: "embedding"; readonly values: Float32Array<ArrayBuffer> }
          | { readonly kind: "refused"; readonly error: ModelInputError };
      }[] = [];
      diagnosticsSinkRef.current = (diagnostics) => reports.push(diagnostics);
      try {
        for (const [position, row] of runRows.entries()) {
          report({
            stage: `embedding ${rowLabel(position)}`,
            count: { done: position, total: runRows.length, unit: "rows" },
            note: "embed() checks the signal before its run: a Cancel during a row takes effect when that row ends.",
          });
          // Counted before the row's embed, while the pipeline is certainly the one it runs on.
          const tokens = rowTokenCount(
            (text, options) => active.pipeline.tokenize(text, options),
            row,
          );
          const embedStart = performance.now();
          try {
            const values = await active.pipeline.embed(row.text, {
              prompt: row.prompt,
              dimension: runDimension,
              signal,
            });
            outcomes.push({
              tokens,
              embedMs: performance.now() - embedStart,
              result: { kind: "embedding", values },
            });
          } catch (error) {
            // A refused request is the row's result; anything else (the GPU, the Session, a
            // cancel) ends the run.
            if (!(error instanceof ModelInputError)) throw error;
            outcomes.push({
              tokens,
              embedMs: performance.now() - embedStart,
              result: { kind: "refused", error },
            });
          }
        }
      } finally {
        diagnosticsSinkRef.current = undefined;
      }
      // Leaving the page during the last row ends the run as cancelled, not with a result.
      signal.throwIfAborted();
      const wallMs = performance.now() - startedAt;

      report({ stage: "hashing" });
      const measured: MeasuredRow[] = [];
      for (const [position, row] of runRows.entries()) {
        const outcome = outcomes[position];
        if (outcome === undefined) throw new Error(`No result for ${rowLabel(position)}`);
        const { result } = outcome;
        measured.push({
          text: row.text,
          prompt: row.prompt,
          tokens: outcome.tokens,
          embedMs: outcome.embedMs,
          outcome:
            result.kind === "embedding"
              ? { kind: "embedding", values: result.values, sha256: await f32Sha256(result.values) }
              : {
                  kind: "refused",
                  error: { name: result.error.name, message: result.error.message },
                },
        });
      }
      signal.throwIfAborted();
      const record = assembleRecord({
        environment: recordEnvironment(runGpu),
        directory,
        manifestSha256: ready.sha256,
        source: runSource,
        selection,
        presetLabel,
        dimension: runDimension,
        rows: measured,
        buildMs,
        wallMs,
        reports,
        countersBefore,
      });
      return {
        record,
        vectors: outcomes.flatMap(({ result }, row) =>
          result.kind === "embedding" ? [{ row, values: result.values }] : [],
        ),
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
        <h1 className="text-2xl font-semibold">EmbeddingGemma — sentence embeddings</h1>
        <Badge
          id="embeddinggemma-status"
          variant={manifest.kind === "ready" ? "secondary" : "outline"}
        >
          {statusText}
        </Badge>
      </div>

      {index.kind === "error" ? (
        <ErrorAlert title="Distribution index" error={index.error} />
      ) : null}
      {manifest.kind === "missing" ? (
        <Alert>
          <AlertTitle>The EmbeddingGemma 2 distribution is missing</AlertTitle>
          <AlertDescription>
            The dev server has no <code>{manifest.directory}</code> (a directory with a{" "}
            <code>karume.json</code>). Assemble the local mirror there (the dist recipe of{" "}
            <code>tools/export-recipes/embeddinggemma2</code>) and reload.
          </AlertDescription>
        </Alert>
      ) : null}
      {manifest.kind === "error" ? <ErrorAlert title="Manifest" error={manifest.error} /> : null}
      {configState?.kind === "error" ? (
        <ErrorAlert title="pipelineConfig" error={configState.error} />
      ) : null}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <Card>
          <CardHeader>
            <CardTitle>Request</CardTitle>
            <CardDescription>
              Each row becomes one unit vector (one <code>embed</code> call, in order). The prompt's
              prefix goes before the row's text; the run compares every pair of rows by the dot
              product of their vectors, which is their cosine similarity.
            </CardDescription>
          </CardHeader>
          <CardContent className="grid gap-4 sm:grid-cols-2">
            {ready === undefined || selection === undefined || presets === undefined ? null : (
              <>
                <Field id="embeddinggemma-model" label="Model">
                  <ChoiceSelect
                    id="embeddinggemma-model"
                    value={selection.model}
                    choices={modelChoices(ready.manifest)}
                    onChange={setChosenModel}
                    disabled={controlsDisabled}
                  />
                </Field>
                <PresetSelect
                  id="embeddinggemma-preset"
                  value={selection.quant}
                  presets={presets.presets}
                  defaultQuant={presets.defaultQuant}
                  onChange={setChosenQuant}
                  disabled={controlsDisabled}
                />
              </>
            )}
            <SourcePicker
              id="embeddinggemma-source"
              value={source.kind}
              cachedAvailable={hfRef !== undefined}
              onChange={chooseSource}
              disabled={task.running}
            />
            {config === undefined || dimension === undefined ? null : (
              <Field
                id="embeddinggemma-dimension"
                label="Dimension"
                hint="One width for every row. A shorter one is the leading slice of the vector, re-normalised to unit length."
              >
                <ChoiceSelect
                  id="embeddinggemma-dimension"
                  value={String(dimension)}
                  choices={dimensionChoices(config)}
                  onChange={(value) => setChosenDimension(Number(value))}
                  disabled={controlsDisabled}
                />
              </Field>
            )}
            <div className="flex flex-col gap-3 sm:col-span-2">
              {rows.map((row, position) => {
                const count = tokenCounts[position];
                const tokens =
                  count === undefined || config === undefined
                    ? undefined
                    : tokenCountView(count, config.maxTokens);
                return (
                  <div
                    key={row.id}
                    className="flex flex-col gap-2 rounded-lg border p-3"
                    id={`embeddinggemma-row-${position}`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <Label htmlFor={`embeddinggemma-row-${position}-text`}>
                          {rowLabel(position)}
                        </Label>
                        {tokens === undefined ? null : (
                          <Badge
                            id={`embeddinggemma-row-${position}-tokens`}
                            variant={tokens.overLimit ? "destructive" : "secondary"}
                          >
                            {tokens.text}
                          </Badge>
                        )}
                      </div>
                      <Button
                        id={`embeddinggemma-row-${position}-remove`}
                        variant="ghost"
                        size="sm"
                        onClick={() => setRows((current) => removeRow(current, row.id))}
                        disabled={task.running || rows.length <= 1}
                      >
                        <Trash2Icon />
                        Remove
                      </Button>
                    </div>
                    <Textarea
                      id={`embeddinggemma-row-${position}-text`}
                      rows={2}
                      value={row.text}
                      onChange={(event) =>
                        setRows((current) =>
                          updateRow(current, row.id, { text: event.target.value }),
                        )
                      }
                      disabled={task.running}
                      aria-invalid={tokens?.overLimit === true}
                    />
                    {tokens?.overLimit === true ? (
                      <p
                        id={`embeddinggemma-row-${position}-over-limit`}
                        className="text-xs text-destructive"
                      >
                        Over the limit: the pipeline will refuse this row (nothing is truncated).
                        Shorten the text or split it.
                      </p>
                    ) : null}
                    {config === undefined ? null : (
                      <Field id={`embeddinggemma-row-${position}-prompt`} label="Prompt">
                        <ChoiceSelect
                          id={`embeddinggemma-row-${position}-prompt`}
                          value={promptValue(row.prompt)}
                          choices={promptChoices(config, row.prompt)}
                          onChange={(value) =>
                            setRows((current) =>
                              updateRow(current, row.id, { prompt: promptOfValue(value) }),
                            )
                          }
                          disabled={task.running}
                        />
                      </Field>
                    )}
                  </div>
                );
              })}
              <Button
                id="embeddinggemma-row-add"
                variant="outline"
                size="sm"
                className="self-start"
                onClick={() => setRows(addRow)}
                disabled={task.running}
              >
                <PlusIcon />
                Add row
              </Button>
              {config === undefined ? null : (
                <p className="text-xs text-muted-foreground">
                  A row may be up to {config.maxTokens} tokens, prompt prefix, &lt;bos&gt; and
                  &lt;eos&gt; included, or fewer on a device whose buffer limit is smaller (such a
                  row is refused when it is run); a longer row is refused (nothing is truncated). A
                  text holding an image, audio or video placeholder token is refused too. Each row
                  shows its count once the first Run has built the pipeline.
                </p>
              )}
            </div>
            <div className="flex flex-col gap-1.5 sm:col-span-2">
              <Button
                id="embeddinggemma-run"
                className="self-start"
                onClick={run}
                disabled={blocker !== undefined}
              >
                Run
              </Button>
              <p id="embeddinggemma-run-hint" className="text-xs text-muted-foreground">
                {blocker ?? "Ready."} The first run builds the pipeline (fetches the weights, then
                builds the Session and uploads them); later runs with the same source, model and
                preset reuse it. Cancel stops the fetch at once and is unavailable during the
                Session build. While rows are embedded, a Cancel takes effect when the current row
                ends (the rest are skipped).
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
                id="embeddinggemma-gpu-timing"
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
            <p id="embeddinggemma-gpu-timing-hint" className="text-xs text-muted-foreground">
              {timestampQuery.kind === "probing" ? "Looking at the adapter… " : null}
              {timestampQuery.kind === "error"
                ? `Unavailable: the adapter could not be probed (${thrownText(timestampQuery.error)}). `
                : null}
              {timestampQuery.kind === "ready" && !timestampQuery.available
                ? "Unavailable: this adapter has no timestamp-query. "
                : null}
              Times every kernel of the model on the GPU. Every dispatch becomes its own pass, so a
              timed run is slower and its embed seconds are not comparable with untimed records.
              Switching acquires the GPU again, so the next run rebuilds the pipeline.
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
  const labels = record.request.rows.map((_, position) => rowLabel(position));
  return (
    <Card>
      <CardHeader>
        <CardTitle>Result</CardTitle>
        <CardDescription>
          {record.model.name} · {record.preset.quant} · {record.request.dimension} dimensions ·{" "}
          {record.request.rows.length} {record.request.rows.length === 1 ? "row" : "rows"} ·{" "}
          {record.timings.wallSeconds} s
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-6">
        <div className="flex flex-col gap-2">
          <span className="text-sm text-muted-foreground">
            Cosine similarity (the dot product of the unit vectors; — for a refused row)
          </span>
          <Table id="embeddinggemma-similarity">
            <TableHeader>
              <TableRow>
                <TableHead />
                {labels.map((label) => (
                  <TableHead key={label}>{label}</TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {record.output.similarity.map((cells, row) => (
                <TableRow key={labels[row] ?? row}>
                  <TableCell className="font-medium whitespace-nowrap text-muted-foreground">
                    {labels[row]}
                  </TableCell>
                  {cells.map((value, column) => (
                    <TableCell key={labels[column] ?? column} className="font-mono tabular-nums">
                      {similarityText(value)}
                    </TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
        <Table id="embeddinggemma-rows">
          <TableHeader>
            <TableRow>
              <TableHead />
              <TableHead>Prompt</TableHead>
              <TableHead>Text</TableHead>
              <TableHead>Result</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {record.output.rows.map((output, row) => {
              const request = record.request.rows[row];
              return (
                <TableRow key={labels[row] ?? row}>
                  <TableCell className="align-top font-medium whitespace-nowrap text-muted-foreground">
                    {labels[row]}
                  </TableCell>
                  <TableCell className="align-top font-mono text-xs">
                    {request?.prompt ?? "none"}
                  </TableCell>
                  <TableCell className="align-top text-xs break-words whitespace-normal">
                    {request?.text === "" ? <i>(empty)</i> : request?.text}
                  </TableCell>
                  <TableCell className="align-top font-mono text-xs break-all whitespace-normal">
                    {output.kind === "refused" ? (
                      <span className="text-destructive">
                        {output.error.name}: {output.error.message}
                      </span>
                    ) : (
                      <div className="flex flex-col gap-1">
                        <span>{rowOutcomeText(output)}</span>
                        <span className="text-muted-foreground">{firstValuesText(output)}, …</span>
                        <span className="text-muted-foreground">sha256 {output.sha256}</span>
                      </div>
                    )}
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
        <RecordActions record={record} filename={`embeddinggemma_${stamp}.json`}>
          {result.vectors.map(({ row, values }) => (
            <Button
              key={row}
              id={`embeddinggemma-embedding-${row}`}
              variant="outline"
              size="sm"
              onClick={() =>
                downloadBlob(
                  new Blob([f32Bytes(values)], { type: "application/octet-stream" }),
                  `embeddinggemma_${stamp}_row${row + 1}.f32`,
                )
              }
            >
              Download {rowLabel(row)} (raw f32)
            </Button>
          ))}
        </RecordActions>
        <KeyValueTable rows={summaryRows(record)} />
      </CardContent>
    </Card>
  );
};
