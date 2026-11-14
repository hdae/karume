/**
 * The attention route bench: times the two a8 attention routes (S stored in f16 over three
 * dispatches, and the online int8 kernel) at Wan2.2's shapes with the production kernels on this
 * browser's GPU, and saves the record. The routes are timed on their own, apart from a generation,
 * so this page acquires its own GPU with timestamps. Optionally it also times a measure-first
 * candidate against the online kernel (the tiled key axis, perf-ledger K-106).
 */

import type { AcquireGpuOptions } from "@karume/runtime";
import {
  type AttentionBenchReport,
  type AttentionBenchShape,
  type AttentionBenchShapeResult,
  runAttentionRouteBench,
  WAN22_ATTENTION_SHAPES,
} from "@karume/runtime/tune";
import { useEffect, useRef, useState } from "react";

import { ErrorAlert } from "@/components/error-alert";
import { Field } from "@/components/field";
import { KeyValueTable } from "@/components/key-value-table";
import { RecordActions } from "@/components/record-actions";
import { GpuAlerts } from "@/components/run-status-card";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useGpu } from "@/hooks/use-gpu";
import { fileStamp } from "@/lib/download";
import { recordEnvironment } from "@/lib/environment";
import { adapterText, type DeviceFacts, summarizeAdapter } from "@/lib/record";

import {
  benchTableRows,
  candidateRows,
  formatCount,
  formatMs,
  formatRatio,
  shapeLabel,
} from "./rows";

/** The version of the saved record; bump it when a field changes meaning or goes away. */
const RECORD_FORMAT = "karume-playground-attention-bench/2";

/**
 * Module constants (useGpu needs stable options): subgroups when the device can run the 32-lane form (the online
 * kernel's sg32 form), GPU timestamps when it lists timestamp-query.
 */
const TIMED_GPU: Omit<AcquireGpuOptions, "onDeviceLost"> = {
  subgroups: "available",
  gpuTiming: true,
};
const UNTIMED_GPU: Omit<AcquireGpuOptions, "onDeviceLost"> = { subgroups: "available" };

type AttentionBenchRecord = {
  readonly format: typeof RECORD_FORMAT;
  readonly page: "attention-bench";
  readonly recordedAt: string;
  readonly userAgent: string;
  readonly crossOriginIsolated: boolean;
  readonly adapter: ReturnType<typeof summarizeAdapter>;
  readonly device: DeviceFacts;
  readonly settings: {
    readonly shapes: readonly string[];
    /** Empty = the first row block production would run per route. */
    readonly rowsPerDispatch: number | "production";
    readonly rounds: number;
    readonly quantization: boolean;
    /** The candidate `onlineTiled` (perf-ledger K-106) was timed. */
    readonly onlineTiled: boolean;
  };
  /** The runtime's record (`karume-attention-bench/2`). */
  readonly bench: AttentionBenchReport;
};

type RunState =
  | { readonly kind: "idle" }
  | {
      readonly kind: "running";
      readonly startedAt: number;
      readonly status: string;
      readonly done: readonly AttentionBenchShapeResult[];
      readonly total: number;
    }
  | { readonly kind: "error"; readonly error: unknown };

type Form = {
  readonly shapes: ReadonlySet<string>;
  /** Text of the rows input ("" = production's split). */
  readonly rows: string;
  readonly rounds: string;
  readonly quantization: boolean;
  readonly onlineTiled: boolean;
};

const INITIAL_FORM: Form = {
  shapes: new Set(WAN22_ATTENTION_SHAPES.map(({ id }) => id)),
  rows: "",
  rounds: "5",
  quantization: true,
  onlineTiled: false,
};

const SHAPE_HINTS: Readonly<Record<string, string>> = {
  "wan22-self-33f": "Self-attention at 33 frames (N 7,920)",
  "wan22-self-121f": "Self-attention at 121 frames (N 27,280)",
  "wan22-cross-121f": "Cross-attention over the text at 121 frames (N 512)",
};

const positiveInteger = (text: string): number | undefined => {
  const value = Number(text);
  return Number.isSafeInteger(value) && value >= 1 ? value : undefined;
};

/** The chosen shapes with the rows override applied (capped at each shape's query count). */
const chosenShapes = (form: Form, rows: number | undefined): AttentionBenchShape[] =>
  WAN22_ATTENTION_SHAPES.filter(({ id }) => form.shapes.has(id)).map((shape) =>
    rows === undefined ? shape : { ...shape, rowsPerDispatch: Math.min(rows, shape.queries) },
  );

/** Probes the adapter once for timestamp-query, then acquires the GPU accordingly. */
export const AttentionBenchPage = () => {
  const [timing, setTiming] = useState<
    | { readonly kind: "probing" }
    | { readonly kind: "ready"; readonly timestamps: boolean }
    | { readonly kind: "error"; readonly error: unknown }
  >({ kind: "probing" });
  useEffect(() => {
    let disposed = false;
    const gpu: GPU | undefined = navigator.gpu;
    if (gpu === undefined) {
      setTiming({ kind: "error", error: new Error("This browser has no navigator.gpu") });
      return;
    }
    gpu.requestAdapter().then(
      (adapter) => {
        if (!disposed) {
          setTiming({
            kind: "ready",
            timestamps: adapter?.features.has("timestamp-query") === true,
          });
        }
      },
      (error: unknown) => {
        if (!disposed) setTiming({ kind: "error", error });
      },
    );
    return () => {
      disposed = true;
    };
  }, []);
  if (timing.kind === "probing") return <p className="text-sm">Looking at the adapter…</p>;
  if (timing.kind === "error") return <ErrorAlert title="GPU" error={timing.error} />;
  return <BenchBody timestamps={timing.timestamps} />;
};

const BenchBody = ({ timestamps }: { timestamps: boolean }) => {
  const { state: gpuState, lease, reacquire, track } = useGpu(timestamps ? TIMED_GPU : UNTIMED_GPU);
  const gpu = gpuState.kind === "ready" ? gpuState.gpu : undefined;
  const [form, setForm] = useState<Form>(INITIAL_FORM);
  const [run, setRun] = useState<RunState>({ kind: "idle" });
  const [record, setRecord] = useState<AttentionBenchRecord | undefined>(undefined);
  const [now, setNow] = useState(() => performance.now());
  const controllerRef = useRef<AbortController | undefined>(undefined);
  const running = run.kind === "running";

  useEffect(() => {
    if (!running) return;
    const timer = setInterval(() => setNow(performance.now()), 500);
    return () => clearInterval(timer);
  }, [running]);
  // Leaving the page or re-acquiring the GPU (the lease ends) aborts a run; useGpu destroys the GPU
  // once the tracked run has settled.
  useEffect(
    () => () =>
      controllerRef.current?.abort(
        new Error("The GPU was released (the page was left or the GPU is acquired again)"),
      ),
    [lease],
  );

  const rowsOverride = form.rows.trim() === "" ? undefined : positiveInteger(form.rows);
  const rounds = positiveInteger(form.rounds);
  const shapes = chosenShapes(form, rowsOverride);
  const formError =
    form.rows.trim() !== "" && rowsOverride === undefined
      ? "Rows per dispatch must be a positive integer (or empty)"
      : rounds === undefined
        ? "Rounds must be a positive integer"
        : shapes.length === 0
          ? "Choose at least one shape"
          : undefined;
  const canRun = gpu !== undefined && !running && formError === undefined;

  const start = async () => {
    if (gpu === undefined || rounds === undefined) return;
    const controller = new AbortController();
    controllerRef.current = controller;
    const startedAt = performance.now();
    setNow(startedAt);
    setRun({ kind: "running", startedAt, status: "starting", done: [], total: shapes.length });
    try {
      const bench = await track(
        runAttentionRouteBench(gpu, {
          shapes,
          rounds,
          quantization: form.quantization,
          onlineTiled: form.onlineTiled,
          signal: controller.signal,
          onProgress: (progress) =>
            setRun((current) =>
              current.kind !== "running"
                ? current
                : progress.kind === "status"
                  ? { ...current, status: progress.message }
                  : { ...current, done: [...current.done, progress.result] },
            ),
        }),
      );
      setRecord({
        format: RECORD_FORMAT,
        page: "attention-bench",
        ...recordEnvironment(gpu),
        settings: {
          shapes: shapes.map(({ id }) => id),
          rowsPerDispatch: rowsOverride ?? "production",
          rounds,
          quantization: form.quantization,
          onlineTiled: form.onlineTiled,
        },
        bench,
      });
      setRun({ kind: "idle" });
    } catch (error) {
      setRun({ kind: "error", error });
    } finally {
      controllerRef.current = undefined;
    }
  };

  const toggleShape = (id: string, checked: boolean) =>
    setForm((current) => {
      const next = new Set(current.shapes);
      if (checked) next.add(id);
      else next.delete(id);
      return { ...current, shapes: next };
    });

  const results = run.kind === "running" ? run.done : (record?.bench.shapes ?? []);
  const unit = record?.bench.timing.unit ?? (gpu?.gpuTimingEnabled === true ? "ns" : "wall");

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap items-baseline gap-3">
        <h1 className="text-2xl font-semibold">Attention route bench</h1>
        <Badge variant={gpu === undefined ? "outline" : "secondary"}>
          {gpu === undefined
            ? gpuState.kind
            : gpu.gpuTimingEnabled
              ? "GPU timestamps"
              : "wall clock"}
        </Badge>
      </div>
      {!timestamps ? (
        <Alert>
          <AlertTitle>No timestamp-query on this adapter</AlertTitle>
          <AlertDescription>
            Times are wall-clock per pass divided by the repetitions (they include the submit
            overhead). Compare ratios, not absolute values.
          </AlertDescription>
        </Alert>
      ) : null}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <Card>
          <CardHeader>
            <CardTitle>Shapes and settings</CardTitle>
            <CardDescription>
              Wan2.2 TI2V-5B attention (24 heads, D 128). Each route is timed on one row block, the
              first one production would dispatch, unless rows per dispatch is set.
            </CardDescription>
          </CardHeader>
          <CardContent className="grid gap-4 sm:grid-cols-2">
            <div className="flex flex-col gap-2 sm:col-span-2">
              {WAN22_ATTENTION_SHAPES.map((shape) => (
                <Label key={shape.id} className="flex items-center gap-2 font-normal">
                  <Checkbox
                    id={`shape-${shape.id}`}
                    checked={form.shapes.has(shape.id)}
                    onCheckedChange={(checked) => toggleShape(shape.id, checked)}
                    disabled={running}
                  />
                  <span>
                    {SHAPE_HINTS[shape.id] ?? shape.id}{" "}
                    <span className="text-muted-foreground">{shapeLabel(shape)}</span>
                  </span>
                </Label>
              ))}
            </div>
            <Field
              id="bench-rows"
              label="Rows per dispatch"
              hint="Empty = production's split (s16: the first row block under the binding limit; online: the first window). Capped at each shape's query count; the online kernel rounds up to 64-row workgroups."
            >
              <Input
                id="bench-rows"
                inputMode="numeric"
                placeholder="production"
                value={form.rows}
                onChange={(event) => setForm({ ...form, rows: event.target.value })}
                disabled={running}
              />
            </Field>
            <Field id="bench-rounds" label="Rounds" hint="The min of the rounds is kept.">
              <Input
                id="bench-rounds"
                inputMode="numeric"
                value={form.rounds}
                onChange={(event) => setForm({ ...form, rounds: event.target.value })}
                disabled={running}
              />
            </Field>
            <Label className="flex items-center gap-2 font-normal sm:col-span-2">
              <Checkbox
                id="bench-quantization"
                checked={form.quantization}
                onCheckedChange={(checked) => setForm({ ...form, quantization: checked })}
                disabled={running}
              />
              Also time the shared quantization (q, k, Vᵀ — once per node, outside the route totals)
            </Label>
            <Label className="flex items-center gap-2 font-normal sm:col-span-2">
              <Checkbox
                id="bench-online-tiled"
                checked={form.onlineTiled}
                onCheckedChange={(checked) => setForm({ ...form, onlineTiled: checked })}
                disabled={running}
              />
              Also time the candidate: the online kernel reading Vᵀ in its own key order (tiled,
              perf-ledger K-106), in the same rounds as online, with its producer and an identity
              check
            </Label>
            <div className="flex gap-3 sm:col-span-2">
              <Button id="bench-run" onClick={() => void start()} disabled={!canRun}>
                Run
              </Button>
              <Button
                id="bench-cancel"
                variant="outline"
                onClick={() => controllerRef.current?.abort(new Error("Cancelled by the user"))}
                disabled={!running}
              >
                Cancel
              </Button>
            </div>
            {formError === undefined ? null : (
              <p className="text-sm text-destructive sm:col-span-2">{formError}</p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Status</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <KeyValueTable
              rows={[
                ["GPU", gpu === undefined ? gpuState.kind : adapterText(gpu.adapterInfo)],
                ["Timing", unit === "ns" ? "GPU timestamps (ns)" : `${unit} clock`],
                ...(gpu === undefined
                  ? []
                  : ([
                      ["Subgroup lanes linear", String(gpu.subgroupLanesLinear)],
                      [
                        "Workgroup storage",
                        `${formatCount(gpu.limits.maxComputeWorkgroupStorageSize)} B`,
                      ],
                      [
                        "Storage binding limit",
                        `${formatCount(gpu.limits.maxStorageBufferBindingSize)} B`,
                      ],
                    ] as const)),
              ]}
            />
            {run.kind === "running" ? (
              <div className="flex flex-col gap-2 text-sm">
                <div className="font-medium">
                  Running — {((now - run.startedAt) / 1000).toFixed(1)} s · {run.done.length} /{" "}
                  {run.total} shapes
                </div>
                <Progress value={(run.done.length / run.total) * 100} />
                <div className="font-mono text-xs text-muted-foreground">{run.status}</div>
              </div>
            ) : null}
            {record?.bench.aborted === true ? (
              <Alert>
                <AlertTitle>Cancelled</AlertTitle>
                <AlertDescription>
                  The record keeps the shapes finished before the cancel.
                </AlertDescription>
              </Alert>
            ) : null}
            <GpuAlerts gpu={gpuState} onReacquire={reacquire} />
            {run.kind === "error" ? <ErrorAlert title="Bench failed" error={run.error} /> : null}
            {record === undefined ? null : <VariantFacts report={record.bench} />}
          </CardContent>
        </Card>
      </div>

      {results.length === 0 ? null : (
        <Card>
          <CardHeader>
            <CardTitle>Result</CardTitle>
            <CardDescription>
              ms per dispatch from {unit === "ns" ? "GPU timestamps" : "the wall clock"}; online /
              s16 compares ms per 1,000 query rows (below 1 = online faster). Node ms = per-row time
              × the node's query rows, quantization excluded.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            {record === undefined ? null : (
              <RecordActions
                record={record}
                filename={`attention-bench_${fileStamp(new Date(record.recordedAt))}.json`}
              />
            )}
            <ResultTable results={results} />
          </CardContent>
        </Card>
      )}

      {results.some((result) => result.onlineTiled !== undefined) ? (
        <Card>
          <CardHeader>
            <CardTitle>Candidate: tiled key axis (K-106)</CardTitle>
            <CardDescription>
              Time ratios tiled ÷ online (below 1 = the candidate faster): the kernel alone is the
              median of the per-round ratios, each round timing A B B A; kernel + producer adds each
              side's Vᵀ producer per node (needs the quantization line). The adoption line is a
              kernel gain of ×1.05 on the RTX 5070 Ti, a ratio of at most ×0.952. Identical = both
              producers on the same Vᵀ and both kernels on their outputs gave the same O words.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <CandidateTable results={results} />
          </CardContent>
        </Card>
      ) : null}
    </div>
  );
};

const VariantFacts = ({ report }: { report: AttentionBenchReport }) => {
  const { variants } = report;
  const { dp4a, device } = variants;
  return (
    <KeyValueTable
      rows={[
        ["Geometry profile", variants.geometryProfile],
        [
          "dp4a",
          `${dp4a.dot}${dp4a.listed ? "" : " (WGSL feature not listed)"}${
            dp4a.canary === undefined
              ? ""
              : ` — canary ${dp4a.canary.branch}${dp4a.canary.exact ? "" : " (inexact)"}`
          }`,
        ],
        [
          "Subgroups",
          `${device.subgroups} (features ${device.features.subgroups} / size control ${device.features.subgroupSizeControl} / subgroup_id ${device.features.subgroupId} / lanes linear ${device.subgroupLanesLinear})`,
        ],
        ["Workgroup storage", `${formatCount(device.maxComputeWorkgroupStorageSize)} B`],
        ["Rounds", String(report.settings.rounds)],
      ]}
    />
  );
};

const CandidateTable = ({ results }: { results: readonly AttentionBenchShapeResult[] }) => (
  <Table>
    <TableHeader>
      <TableRow>
        <TableHead>Shape</TableHead>
        <TableHead className="text-right">tiled ÷ online (kernel)</TableHead>
        <TableHead className="text-right">tiled ÷ online (kernel + producer)</TableHead>
        <TableHead className="text-right">online ms</TableHead>
        <TableHead className="text-right">tiled ms</TableHead>
        <TableHead>Producer ms / dispatch</TableHead>
        <TableHead className="text-right">Rounds</TableHead>
        <TableHead>Identical</TableHead>
      </TableRow>
    </TableHeader>
    <TableBody>
      {candidateRows(results).map((row) => (
        <TableRow key={row.key}>
          <TableCell className="font-mono text-xs">{row.shape}</TableCell>
          <TableCell className="text-right font-mono text-xs">
            {formatRatio(row.kernelRatio)}
          </TableCell>
          <TableCell className="text-right font-mono text-xs">
            {formatRatio(row.withProducerRatio)}
          </TableCell>
          <TableCell className="text-right font-mono text-xs">{formatMs(row.onlineMs)}</TableCell>
          <TableCell className="text-right font-mono text-xs">{formatMs(row.tiledMs)}</TableCell>
          <TableCell className="text-xs">{row.producer ?? ""}</TableCell>
          <TableCell className="text-right font-mono text-xs">{row.rounds ?? ""}</TableCell>
          <TableCell className="text-xs whitespace-normal">
            {row.identical === undefined ? (
              ""
            ) : row.identical ? (
              <Badge variant="secondary">identical</Badge>
            ) : (
              <Badge variant="destructive">differs</Badge>
            )}
            {row.note === undefined ? null : <div className="text-destructive">{row.note}</div>}
          </TableCell>
        </TableRow>
      ))}
    </TableBody>
  </Table>
);

const ResultTable = ({ results }: { results: readonly AttentionBenchShapeResult[] }) => (
  <Table>
    <TableHeader>
      <TableRow>
        <TableHead>Shape</TableHead>
        <TableHead>Route</TableHead>
        <TableHead className="text-right">Rows</TableHead>
        <TableHead>Production split</TableHead>
        <TableHead className="text-right">ms / dispatch</TableHead>
        <TableHead className="text-right">ms / 1,000 rows</TableHead>
        <TableHead className="text-right">TOPS</TableHead>
        <TableHead className="text-right">Node ms</TableHead>
        <TableHead className="text-right">online / s16</TableHead>
        <TableHead>Stages · variant</TableHead>
      </TableRow>
    </TableHeader>
    <TableBody>
      {benchTableRows(results).map((row) => (
        <TableRow key={row.key}>
          <TableCell className="font-mono text-xs">{row.shape}</TableCell>
          <TableCell>{row.route}</TableCell>
          <TableCell className="text-right font-mono text-xs">
            {row.rows === undefined ? "" : formatCount(row.rows)}
          </TableCell>
          <TableCell className="text-xs">{row.split ?? ""}</TableCell>
          <TableCell className="text-right font-mono text-xs">
            {row.route === "quantization" ? "" : formatMs(row.msPerDispatch)}
          </TableCell>
          <TableCell className="text-right font-mono text-xs">
            {row.route === "quantization" ? "" : formatMs(row.msPer1000Rows)}
          </TableCell>
          <TableCell className="text-right font-mono text-xs">
            {row.tops === undefined ? "" : row.tops.toFixed(1)}
          </TableCell>
          <TableCell className="text-right font-mono text-xs">{formatMs(row.nodeMs)}</TableCell>
          <TableCell className="text-right font-mono text-xs">
            {row.ratio === undefined ? "" : `×${row.ratio.toFixed(3)}`}
          </TableCell>
          <TableCell className="text-xs whitespace-normal">
            {[row.stages, row.variant].filter((part) => part !== undefined).join(" · ")}
            {row.note === undefined ? null : <div className="text-destructive">{row.note}</div>}
          </TableCell>
        </TableRow>
      ))}
    </TableBody>
  </Table>
);
