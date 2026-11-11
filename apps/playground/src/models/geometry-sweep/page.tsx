/**
 * The geometry sweep: measures the GEMM tile geometries on this browser's GPU with
 * `runGeometrySweep` of `@karume/runtime/tune`, one case per call with idle time between cases,
 * merges the per-case records into one sweep record (the file `deriveGeometryProfile` reads), and
 * derives a table from it on the page so the verdicts are seen before the file is sent.
 *
 * `runGeometrySweep` acquires a dedicated device of its own, so this page has no `useGpu`: the run
 * joins the app's device queue through `runOnGpuQueue` instead.
 */

import {
  deriveGeometryProfile,
  type GeometrySweepCandidateSet,
  type GeometrySweepOp,
  type GeometrySweepReport,
  type GeometrySweepTimingUnit,
  geometryProfileJson,
  geometrySweepCases,
  sweepCaseSetId,
} from "@karume/runtime/tune";
import { useEffect, useRef, useState } from "react";

import { ChoiceSelect } from "@/components/choice-select";
import { ErrorAlert } from "@/components/error-alert";
import { Field } from "@/components/field";
import { KeyValueTable } from "@/components/key-value-table";
import { RecordActions } from "@/components/record-actions";
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
import { runOnGpuQueue, useGpuQueue } from "@/hooks/use-gpu";
import { adapterText, sha256Hex } from "@/lib/record";

import {
  caseSummaryRows,
  deriveTarget,
  mergeSweepReports,
  sweepFileName,
  sweepRecordText,
} from "./record";
import { foldSweepEvent, INITIAL_LIVE, runSweepPlan, type SweepLive, type SweepStop } from "./run";
import {
  CANDIDATE_SETS,
  initialForm,
  matchingPreset,
  planSweep,
  presetForm,
  SWEEP_PRESETS,
  type SweepForm,
  type SweepPlan,
  sweepOps,
} from "./settings";

const CANDIDATE_SET_HINTS: Readonly<Record<GeometrySweepCandidateSet, string>> = {
  quick: "a few geometries per family — a dry run that checks the page and the device",
  "quick+": "quick plus the geometries of the registered tables (the runtime's default)",
  full: "every geometry of the family — the set a table is derived from",
  k85: "the vec4 skeleton of the i8a8 linear on its own cases (perf-ledger K-85) — an evaluation, no table is derived",
};

/** The adapter as the browser reports it, read without creating a device. */
type AdapterProbe =
  | { readonly kind: "probing" }
  | { readonly kind: "ready"; readonly info: GPUAdapterInfo; readonly timestamps: boolean }
  | { readonly kind: "missing"; readonly reason: string };

type RunState =
  | { readonly kind: "idle" }
  | {
      readonly kind: "running";
      readonly plan: SweepPlan;
      readonly startedAt: number;
      readonly live: SweepLive;
    }
  | {
      readonly kind: "ended";
      readonly plan: SweepPlan;
      readonly seconds: number;
      readonly live: SweepLive;
      readonly stop: SweepStop;
    };

/** The merged record, the sha256 of the bytes saved, and the table derived from those bytes. */
type SweepResult =
  | {
      readonly kind: "ready";
      readonly report: GeometrySweepReport;
      readonly filename: string;
      readonly sha256: string;
      readonly derivation:
        | {
            readonly kind: "derived";
            readonly id: string;
            readonly verdicts: readonly string[];
            readonly profileJson: string;
          }
        | { readonly kind: "failed"; readonly error: unknown };
    }
  | { readonly kind: "failed"; readonly error: unknown };

/** Merges, serializes, hashes and derives (the derivation reads the very bytes that are saved). */
const buildResult = async (
  reports: readonly GeometrySweepReport[],
  plan: SweepPlan,
  stop: SweepStop,
): Promise<SweepResult> => {
  try {
    const report = mergeSweepReports(reports, {
      plannedCaseIds: plan.caseIds,
      cancelledBetweenCases: stop.kind === "cancelled" && reports.length < plan.caseIds.length,
    });
    const text = sweepRecordText(report);
    const filename = sweepFileName(report);
    const sha256 = await sha256Hex(new TextEncoder().encode(text));
    const target = deriveTarget(report.adapter);
    let derivation: Extract<SweepResult, { kind: "ready" }>["derivation"];
    try {
      const saved: unknown = JSON.parse(text);
      const { profile, verdicts } = deriveGeometryProfile(
        [{ report: saved, path: filename, sha256 }],
        target,
      );
      derivation = {
        kind: "derived",
        id: target.id,
        verdicts,
        profileJson: geometryProfileJson(profile),
      };
    } catch (error) {
      derivation = { kind: "failed", error };
    }
    return { kind: "ready", report, filename, sha256, derivation };
  } catch (error) {
    return { kind: "failed", error };
  }
};

/** ns (GPU time, or wall clock submit → completion) as ms; raw Deno ticks as they are. */
const formatPerDispatch = (value: number | undefined, unit: GeometrySweepTimingUnit | undefined) =>
  value === undefined
    ? ""
    : unit === "ns"
      ? `${(value / 1e6).toFixed(3)} ms`
      : unit === "wall"
        ? `${(value / 1e6).toFixed(3)} ms wall`
        : `${value.toFixed(0)} ${unit ?? ""}`;

const stopText = (stop: SweepStop): string => {
  switch (stop.kind) {
    case "completed":
      return "Done";
    case "cancelled":
      return "Cancelled";
    case "deviceLost":
      return `Device lost (${stop.reason}): ${stop.message}`;
    case "error":
      return "Failed";
  }
};

export const GeometrySweepPage = () => {
  const queue = useGpuQueue();
  const [cases] = useState(geometrySweepCases);
  const ops = sweepOps(cases);
  const [caseSetId] = useState(sweepCaseSetId);
  const [form, setForm] = useState<SweepForm>(() => initialForm(cases));
  const [probe, setProbe] = useState<AdapterProbe>({ kind: "probing" });
  const [run, setRun] = useState<RunState>({ kind: "idle" });
  const [result, setResult] = useState<SweepResult | undefined>(undefined);
  const [now, setNow] = useState(() => performance.now());
  const controllerRef = useRef<AbortController | undefined>(undefined);
  // The latest run (still set after it ended) and whether the page is mounted: a run's result is
  // built and shown only while both hold — a run that ended after a newer one started, or after the
  // page was left, builds nothing (the fingerprint and the derivation block the main thread).
  const latestRunRef = useRef<AbortController | undefined>(undefined);
  const mountedRef = useRef(false);
  const running = run.kind === "running";

  useEffect(() => {
    let disposed = false;
    const gpu: GPU | undefined = navigator.gpu;
    if (gpu === undefined) {
      setProbe({ kind: "missing", reason: "This browser has no navigator.gpu" });
      return;
    }
    gpu.requestAdapter().then(
      (adapter) => {
        if (disposed) return;
        setProbe(
          adapter === null
            ? { kind: "missing", reason: "requestAdapter returned no adapter" }
            : {
                kind: "ready",
                info: adapter.info,
                timestamps: adapter.features.has("timestamp-query"),
              },
        );
      },
      (error: unknown) => {
        if (!disposed) setProbe({ kind: "missing", reason: String(error) });
      },
    );
    return () => {
      disposed = true;
    };
  }, []);
  useEffect(() => {
    if (!running) return;
    const timer = setInterval(() => setNow(performance.now()), 500);
    return () => clearInterval(timer);
  }, [running]);
  // Leaving the page aborts the sweep at its next cancel point; the app's GPU queue stays held until
  // it has settled (runOnGpuQueue), so the next page acquires only after the sweep's device is gone.
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      controllerRef.current?.abort(new Error("The page was left during the sweep"));
    };
  }, []);

  const planned = planSweep(form, cases);
  const preset = matchingPreset(form, cases);
  const canRun = !running && "plan" in planned && probe.kind !== "missing";

  const start = async (plan: SweepPlan) => {
    if (controllerRef.current !== undefined) return;
    const controller = new AbortController();
    controllerRef.current = controller;
    latestRunRef.current = controller;
    const startedAt = performance.now();
    setNow(startedAt);
    setResult(undefined);
    setRun({ kind: "running", plan, startedAt, live: INITIAL_LIVE });
    // Taken synchronously in this click handler: the queue is held before any later commit.
    // runSweepPlan never rejects; the queue's previous destroy can (then nothing ran).
    const outcome = await runOnGpuQueue(queue, () =>
      runSweepPlan(plan, {
        signal: controller.signal,
        onEvent: (event) =>
          setRun((current) =>
            current.kind === "running"
              ? { ...current, live: foldSweepEvent(current.live, event) }
              : current,
          ),
      }),
    ).catch((error: unknown) => ({ reports: [], stop: { kind: "error", error } as const }));
    controllerRef.current = undefined;
    setRun((current) =>
      current.kind === "running"
        ? {
            kind: "ended",
            plan,
            seconds: (performance.now() - startedAt) / 1000,
            live: current.live,
            stop: outcome.stop,
          }
        : current,
    );
    const shown = () => mountedRef.current && latestRunRef.current === controller;
    if (outcome.reports.length === 0 || !shown()) return;
    const built = await buildResult(outcome.reports, plan, outcome.stop);
    if (shown()) setResult(built);
  };

  const toggle = <T,>(set: ReadonlySet<T>, value: T, on: boolean): Set<T> => {
    const next = new Set(set);
    if (on) next.add(value);
    else next.delete(value);
    return next;
  };
  const setOpCases = (op: GeometrySweepOp, on: boolean) =>
    setForm((current) => {
      const next = new Set(current.cases);
      for (const entry of cases) {
        if (entry.op !== op) continue;
        if (on) next.add(entry.id);
        else next.delete(entry.id);
      }
      return { ...current, cases: next };
    });

  const live = run.kind === "idle" ? undefined : run.live;
  const summaryRows =
    result?.kind === "ready"
      ? caseSummaryRows(result.report.rows, result.report.cases)
      : live === undefined
        ? []
        : caseSummaryRows(live.rows, live.cases);
  const unit = result?.kind === "ready" ? result.report.gpuTiming.unit : live?.unit;

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap items-baseline gap-3">
        <h1 className="text-2xl font-semibold">Geometry sweep</h1>
        {preset === undefined ? null : <Badge variant="secondary">{preset.label}</Badge>}
      </div>
      <Alert>
        <AlertTitle>Serve this page from the checkout that will derive the table</AlertTitle>
        <AlertDescription>
          The record carries the case set and the kernel fingerprints of the runtime this page was
          built from (caseSet, defaultKernels, candidateKernels), and deriveGeometryProfile rejects
          a record whose values differ from the deriving code. Do not edit packages/runtime of the
          serving checkout until the record has been derived: the dev server reloads edited modules.
          Keep this tab in front and nothing else on the GPU while it runs.
        </AlertDescription>
      </Alert>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <Card>
          <CardHeader>
            <CardTitle>Cases and settings</CardTitle>
            <CardDescription>
              One runGeometrySweep call per case (each acquires and destroys its own device), idle
              between cases, one merged record.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <div className="flex flex-wrap gap-2">
              {SWEEP_PRESETS.map((entry) => (
                <Button
                  key={entry.id}
                  id={`preset-${entry.id}`}
                  variant="outline"
                  size="sm"
                  title={entry.hint}
                  onClick={() => setForm(presetForm(entry, cases))}
                  disabled={running}
                >
                  {entry.label}
                </Button>
              ))}
              <Button
                id="preset-all"
                variant="ghost"
                size="sm"
                onClick={() => setForm(initialForm(cases))}
                disabled={running}
              >
                Every case (reset)
              </Button>
            </div>
            <div className="flex flex-col gap-2">
              <Label>Op families</Label>
              <div className="flex flex-wrap gap-x-4 gap-y-2">
                {ops.map((op) => (
                  <Label key={op} className="flex items-center gap-2 font-normal">
                    <Checkbox
                      id={`op-${op}`}
                      checked={form.ops.has(op)}
                      onCheckedChange={(checked) =>
                        setForm((current) => ({
                          ...current,
                          ops: toggle(current.ops, op, checked),
                        }))
                      }
                      disabled={running}
                    />
                    <span className="font-mono text-xs">{op}</span>
                  </Label>
                ))}
              </div>
            </div>
            <div className="flex max-h-96 flex-col gap-3 overflow-auto rounded-md border p-3">
              {ops
                .filter((op) => form.ops.has(op))
                .map((op) => (
                  <div key={op} className="flex flex-col gap-1.5">
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-xs font-medium">{op}</span>
                      <Button
                        variant="ghost"
                        size="xs"
                        onClick={() => setOpCases(op, true)}
                        disabled={running}
                      >
                        all
                      </Button>
                      <Button
                        variant="ghost"
                        size="xs"
                        onClick={() => setOpCases(op, false)}
                        disabled={running}
                      >
                        none
                      </Button>
                    </div>
                    {cases
                      .filter((entry) => entry.op === op)
                      .map((entry) => (
                        <Label key={entry.id} className="flex items-center gap-2 font-normal">
                          <Checkbox
                            id={`case-${entry.id}`}
                            checked={form.cases.has(entry.id)}
                            onCheckedChange={(checked) =>
                              setForm((current) => ({
                                ...current,
                                cases: toggle(current.cases, entry.id, checked),
                              }))
                            }
                            disabled={running}
                          />
                          <span className="font-mono text-xs break-all">{entry.id}</span>
                        </Label>
                      ))}
                  </div>
                ))}
            </div>
            <div className="grid gap-4 sm:grid-cols-3">
              <Field
                id="sweep-candidates"
                label="Candidate set"
                hint={CANDIDATE_SET_HINTS[form.candidateSet]}
              >
                <ChoiceSelect
                  id="sweep-candidates"
                  value={form.candidateSet}
                  choices={CANDIDATE_SETS.map((value) => ({ value, label: value }))}
                  onChange={(candidateSet) => setForm((current) => ({ ...current, candidateSet }))}
                  disabled={running}
                />
              </Field>
              <Field id="sweep-rounds" label="Rounds" hint="Per geometry; the min is kept.">
                <Input
                  id="sweep-rounds"
                  inputMode="numeric"
                  value={form.rounds}
                  onChange={(event) => {
                    const rounds = event.target.value;
                    setForm((current) => ({ ...current, rounds }));
                  }}
                  disabled={running}
                />
              </Field>
              <Field
                id="sweep-idle"
                label="Idle between cases (s)"
                hint="Cooling before every case but the first."
              >
                <Input
                  id="sweep-idle"
                  inputMode="numeric"
                  value={form.idleSeconds}
                  onChange={(event) => {
                    const idleSeconds = event.target.value;
                    setForm((current) => ({ ...current, idleSeconds }));
                  }}
                  disabled={running}
                />
              </Field>
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <Button
                id="sweep-run"
                onClick={() => {
                  if ("plan" in planned) void start(planned.plan);
                }}
                disabled={!canRun}
              >
                Run
              </Button>
              <Button
                id="sweep-cancel"
                variant="outline"
                onClick={() => controllerRef.current?.abort(new Error("Cancelled by the user"))}
                disabled={!running}
              >
                Cancel
              </Button>
              <span className="text-sm text-muted-foreground">
                {"plan" in planned
                  ? `${planned.plan.caseIds.length} case${planned.plan.caseIds.length === 1 ? "" : "s"}`
                  : null}
              </span>
            </div>
            {"error" in planned ? (
              <p className="text-sm text-destructive">{planned.error}</p>
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
                  "Adapter",
                  probe.kind === "ready"
                    ? adapterText(probe.info)
                    : probe.kind === "missing"
                      ? probe.reason
                      : "looking…",
                ],
                [
                  "timestamp-query",
                  probe.kind === "ready" ? (probe.timestamps ? "listed" : "not listed") : "—",
                ],
                ["Cross-origin isolated", String(globalThis.crossOriginIsolated)],
                ["Case set id", caseSetId],
                ...(live?.dp4a === undefined
                  ? []
                  : ([
                      ["dp4a", String(live.dp4a)],
                      ["Timing", live.unit ?? ""],
                    ] as const)),
              ]}
            />
            {probe.kind === "ready" && !probe.timestamps ? (
              <Alert variant="destructive">
                <AlertTitle>No timestamp-query on this adapter</AlertTitle>
                <AlertDescription>
                  The sweep falls back to the wall clock, and deriveGeometryProfile rejects a
                  wall-clock record.
                </AlertDescription>
              </Alert>
            ) : null}
            {run.kind === "idle" ? null : <RunProgress run={run} now={now} />}
            {run.kind === "ended" && run.stop.kind === "error" ? (
              <ErrorAlert title="Sweep failed" error={run.stop.error} />
            ) : null}
          </CardContent>
        </Card>
      </div>

      {summaryRows.length === 0 ? null : (
        <Card>
          <CardHeader>
            <CardTitle>Cases</CardTitle>
            <CardDescription>
              The default geometry's time per dispatch; drift = the default re-measured at the end
              of the case ÷ the first (outside 0.9–1.1 the derivation drops the case's ratios); the
              fastest candidate whose output matched the default bit for bit.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Case</TableHead>
                  <TableHead className="text-right">Rows</TableHead>
                  <TableHead className="text-right">Default</TableHead>
                  <TableHead className="text-right">Drift</TableHead>
                  <TableHead>Fastest (identical)</TableHead>
                  <TableHead className="text-right">Errors</TableHead>
                  <TableHead className="text-right">Mismatches</TableHead>
                  <TableHead className="text-right">Seconds</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {summaryRows.map((row) => (
                  <TableRow key={row.caseId}>
                    <TableCell className="font-mono text-xs">{row.caseId}</TableCell>
                    <TableCell className="text-right font-mono text-xs">{row.rows}</TableCell>
                    <TableCell className="text-right font-mono text-xs">
                      {formatPerDispatch(row.defaultPerDispatch, unit)}
                    </TableCell>
                    <TableCell className="text-right font-mono text-xs">
                      {row.driftRatio === undefined
                        ? (row.driftError ?? "")
                        : row.driftRatio.toFixed(3)}
                    </TableCell>
                    <TableCell className="font-mono text-xs">
                      {row.fastest === undefined
                        ? ""
                        : `${row.fastest.geometry} ×${row.fastest.speedup.toFixed(3)}`}
                    </TableCell>
                    <TableCell className="text-right font-mono text-xs">{row.errors}</TableCell>
                    <TableCell className="text-right font-mono text-xs">{row.mismatches}</TableCell>
                    <TableCell className="text-right font-mono text-xs">
                      {row.seconds === undefined ? "" : row.seconds.toFixed(0)}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}

      {result === undefined ? null : <ResultCard result={result} />}
    </div>
  );
};

const RunProgress = ({ run, now }: { run: Exclude<RunState, { kind: "idle" }>; now: number }) => {
  const { plan, live } = run;
  const total = plan.caseIds.length;
  const caseId = plan.caseIds[live.caseIndex] ?? "";
  const caseRows = live.rows.filter((row) => row.caseId === caseId).length;
  const seconds = run.kind === "running" ? (now - run.startedAt) / 1000 : run.seconds;
  const phase = live.phase;
  return (
    <div className="flex flex-col gap-2 text-sm">
      <div className="font-medium">
        {run.kind === "running" ? "Running" : stopText(run.stop)} — {seconds.toFixed(0)} s ·{" "}
        {live.cases.length} / {total} cases done
      </div>
      <Progress value={(live.cases.length / total) * 100} />
      {run.kind === "running" ? (
        <div className="font-mono text-xs text-muted-foreground">
          {phase.kind === "waiting"
            ? "waiting for the previous page's GPU device to be destroyed"
            : phase.kind === "idle"
              ? `idle before case ${live.caseIndex + 1} / ${total}: ${Math.max(0, (phase.until - now) / 1000).toFixed(0)} s left`
              : `case ${live.caseIndex + 1} / ${total} ${caseId} · ${caseRows} rows · ${phase.status ?? "acquiring the device"}`}
        </div>
      ) : null}
      {live.deviceLost === undefined ? null : (
        <Alert variant="destructive">
          <AlertTitle>Device lost ({live.deviceLost.reason})</AlertTitle>
          <AlertDescription>
            {live.deviceLost.message} — the sweep stops after this case; the record keeps the cases
            finished before.
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
};

const ResultCard = ({ result }: { result: SweepResult }) => {
  if (result.kind === "failed")
    return <ErrorAlert title="Merging the records failed" error={result.error} />;
  const { report, filename, sha256, derivation } = result;
  return (
    <Card>
      <CardHeader>
        <CardTitle>Record</CardTitle>
        <CardDescription>
          One sweep record ({report.format}) over every case that ran, in the format
          deriveGeometryProfile reads. Below, the table derived from this record alone; the
          repository's derivation adds the adapter's earlier records.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        {/* downloadJson saves the very bytes the sha256 was taken of (sweepRecordText). */}
        <RecordActions record={report} filename={filename} />
        <KeyValueTable
          rows={[
            ["File", filename],
            ["sha256", sha256],
            [
              "Adapter",
              Object.values(report.adapter)
                .filter((value) => value !== "")
                .join(" · "),
            ],
            [
              "Candidate set · rounds",
              `${report.settings.candidateSet} · ${report.settings.rounds}`,
            ],
            [
              "Cases",
              `${report.cases.length} finished of ${report.settings.cases?.length ?? 0} planned`,
            ],
            ["Rows", String(report.rows.length)],
            ["Aborted", String(report.aborted)],
            [
              "Timing",
              `${report.gpuTiming.unit}${report.gpuTiming.quantized ? " (quantized)" : ""}`,
            ],
            ["Started · ended", `${report.startedAt} · ${report.date}`],
          ]}
        />
        {derivation.kind === "failed" ? (
          <ErrorAlert title="deriveGeometryProfile rejected the record" error={derivation.error} />
        ) : (
          <div className="flex flex-col gap-2">
            <div className="text-sm font-medium">Verdicts ({derivation.id})</div>
            <pre className="max-h-96 overflow-auto rounded-md bg-muted p-3 text-xs whitespace-pre-wrap">
              {derivation.verdicts.join("\n")}
            </pre>
            <div className="text-sm font-medium">Derived table (geometryProfileJson)</div>
            <pre className="max-h-96 overflow-auto rounded-md bg-muted p-3 text-xs">
              {derivation.profileJson}
            </pre>
          </div>
        )}
      </CardContent>
    </Card>
  );
};
