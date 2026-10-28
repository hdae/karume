/**
 * The kernel checks: runs the identity checks of `@karume/kernel-checks` — the subgroup32 GEMV and
 * RMS reductions and the K-55 / K-91 states-attention rewrites — on this browser's GPU (D3D12 in
 * Chrome on Windows, Metal on macOS), the backends the node lanes (Dawn / Vulkan) do not reach. The
 * same cases and criteria as the runtime's node GPU tests; this page is the manual procedure of the
 * release runbook's subgroup checkbox.
 */

import { type CheckProgress, subgroup32Available } from "@karume/kernel-checks";
import type { GpuContext } from "@karume/runtime";
import { useEffect, useState } from "react";

import { ErrorAlert, thrownText } from "@/components/error-alert";
import { RecordActions } from "@/components/record-actions";
import { RunStatusCard } from "@/components/run-status-card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { type GpuOptions, useGpu } from "@/hooks/use-gpu";
import { type RunProgress, useRunTask } from "@/hooks/use-run-task";
import { fileStamp } from "@/lib/download";
import { recordEnvironment } from "@/lib/environment";
import { seconds } from "@/lib/record";

import { KERNEL_CHECKS, type KernelCheck } from "./checks";
import {
  type CheckOutcome,
  type CheckOutcomes,
  type CheckReport,
  countMismatches,
  foldClassRows,
  hexWord,
  kernelCheckRecord,
  type KernelCheckRecord,
  matrixRowPasses,
  mutantMismatched,
  notRunChecks,
  outcomeList,
  overallVerdict,
  type OverallVerdict,
  type PartSummary,
  productVerdictOf,
  subgroupShortfall,
  summarizePart,
  type Verdict,
  verdictOf,
  withoutOutcomes,
} from "./verdict";

/** A module constant (useGpu needs stable options): the subgroup features when the adapter has them. */
const GPU_OPTIONS: GpuOptions = { subgroups: "available" };

const VERDICT_TEXT: Readonly<Record<Verdict | OverallVerdict, string>> = {
  pass: "PASS",
  fail: "FAIL",
  error: "ERROR",
  "not-available": "not available",
  incomplete: "INCOMPLETE",
  none: "not run",
};

const VerdictBadge = ({ verdict }: { verdict: Verdict | OverallVerdict }) => (
  <Badge
    variant={
      verdict === "pass"
        ? "default"
        : verdict === "fail" || verdict === "error"
          ? "destructive"
          : "outline"
    }
  >
    {VERDICT_TEXT[verdict]}
  </Badge>
);

/** Reads the adapter's features once (the device's alone cannot say which feature is missing). */
export const KernelCheckPage = () => {
  const [probe, setProbe] = useState<
    | { readonly kind: "probing" }
    | { readonly kind: "ready"; readonly features: readonly string[] }
    | { readonly kind: "error"; readonly error: unknown }
  >({ kind: "probing" });
  useEffect(() => {
    let disposed = false;
    const host: GPU | undefined = navigator.gpu;
    if (host === undefined) {
      setProbe({ kind: "error", error: new Error("This browser has no navigator.gpu") });
      return;
    }
    host.requestAdapter().then(
      (adapter) => {
        if (!disposed) setProbe({ kind: "ready", features: [...(adapter?.features ?? [])].sort() });
      },
      (error: unknown) => {
        if (!disposed) setProbe({ kind: "error", error });
      },
    );
    return () => {
      disposed = true;
    };
  }, []);
  if (probe.kind === "probing") return <p className="text-sm">Looking at the adapter…</p>;
  if (probe.kind === "error") return <ErrorAlert title="GPU" error={probe.error} />;
  return <CheckBody adapterFeatures={probe.features} />;
};

const unavailableReason = (gpu: GpuContext, adapterFeatures: readonly string[]): string => {
  const missing = subgroupShortfall(new Set(adapterFeatures), gpu.wgslLanguageFeatures);
  return missing.length === 0
    ? "The device was acquired without the subgroup features"
    : `Needs subgroups, subgroup-size-control and WGSL subgroup_id; this browser lacks ${missing.join(", ")}`;
};

const progressOf = (check: KernelCheck, progress: CheckProgress): RunProgress => ({
  stage: `${check.title} · ${progress.part}`,
  count: { done: progress.done, total: progress.total, unit: "cases" },
});

const CheckBody = ({ adapterFeatures }: { adapterFeatures: readonly string[] }) => {
  const { state: gpuState, lease, reacquire, track } = useGpu(GPU_OPTIONS);
  const gpu = gpuState.kind === "ready" ? gpuState.gpu : undefined;
  const task = useRunTask<void>({ track, lease });
  const [results, setResults] = useState<
    | {
        readonly gpu: GpuContext;
        readonly outcomes: CheckOutcomes;
        readonly record?: KernelCheckRecord;
      }
    | undefined
  >(undefined);
  // The outcomes belong to the device they ran on: after a reacquire they are dropped, so a record
  // never stamps one device's results with another device's environment
  const shown = results !== undefined && results.gpu === gpu ? results : undefined;
  const outcomes = shown?.outcomes ?? {};
  const record = shown?.record;

  const run = (checks: readonly KernelCheck[]) => {
    if (gpu === undefined) return;
    void task.start(async ({ signal, report }) => {
      // The checks about to run read as not run until they settle again, so a cancelled re-run
      // leaves no earlier verdict behind. No record while the run is in progress: it is rebuilt from
      // the outcomes on screen when the run ends, a cancelled or aborted run included
      let next = withoutOutcomes(
        outcomes,
        checks.map(({ id }) => id),
      );
      setResults({ gpu, outcomes: next });
      const settle = (check: KernelCheck, outcome: CheckOutcome) => {
        next = { ...next, [check.id]: outcome };
        setResults({ gpu, outcomes: next });
      };
      try {
        for (const check of checks) {
          if (check.needsSubgroups && !subgroup32Available(gpu)) {
            settle(check, {
              kind: "not-available",
              reason: unavailableReason(gpu, adapterFeatures),
            });
            continue;
          }
          report({ stage: check.title });
          const started = performance.now();
          try {
            const result = await check.run(gpu, {
              signal,
              onProgress: (progress) => report(progressOf(check, progress)),
            });
            settle(check, {
              kind: "done",
              report: result,
              seconds: seconds(performance.now() - started),
            });
          } catch (error) {
            if (signal.aborted) throw error;
            settle(check, {
              kind: "error",
              message: error instanceof Error ? (error.stack ?? error.message) : thrownText(error),
              seconds: seconds(performance.now() - started),
            });
          }
        }
      } finally {
        setResults({
          gpu,
          outcomes: next,
          record: kernelCheckRecord(recordEnvironment(gpu), adapterFeatures, next),
        });
      }
    });
  };

  const ran = outcomeList(outcomes);
  const notRun = notRunChecks(outcomes);
  const overall = overallVerdict(outcomes);
  const canRun = gpu !== undefined && !task.running;

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap items-baseline gap-3">
        <h1 className="text-2xl font-semibold">Kernel checks</h1>
        <VerdictBadge verdict={overall} />
      </div>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <Card>
          <CardHeader>
            <CardTitle>Checks</CardTitle>
            <CardDescription>
              The cases of the runtime&apos;s node GPU tests, on this browser&apos;s backend. Every
              output word is compared as u32 (the RMS and GEMV references as tolerance bands). A
              check the device cannot run is &ldquo;not available&rdquo;, not a pass.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            {KERNEL_CHECKS.map((check) => (
              <div key={check.id} className="flex flex-col gap-1 border-b pb-3 last:border-0">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-medium">{check.title}</span>
                  {gpu !== undefined && check.needsSubgroups && !subgroup32Available(gpu) ? (
                    <Badge variant="outline">not available here</Badge>
                  ) : null}
                  <Button
                    id={`run-${check.id}`}
                    size="sm"
                    variant="outline"
                    className="ml-auto"
                    onClick={() => run([check])}
                    disabled={!canRun}
                  >
                    Run
                  </Button>
                </div>
                <p className="text-sm text-muted-foreground">{check.summary}</p>
              </div>
            ))}
            <Button id="run-all" onClick={() => run(KERNEL_CHECKS)} disabled={!canRun}>
              Run all
            </Button>
          </CardContent>
        </Card>

        <RunStatusCard
          gpu={gpuState}
          onReacquire={reacquire}
          task={task}
          rows={[
            ["Adapter features", adapterFeatures.join(", ")],
            ...(gpu === undefined
              ? []
              : ([
                  ["Device features", [...gpu.features].sort().join(", ")],
                  ["WGSL subgroup_id", String(gpu.wgslLanguageFeatures.has("subgroup_id"))],
                  ["Subgroup lanes linear", String(gpu.subgroupLanesLinear)],
                ] as const)),
          ]}
        />
      </div>

      {ran.length === 0 ? null : (
        <Card>
          <CardHeader>
            <CardTitle>Result</CardTitle>
            <CardDescription>
              Mismatching words are the failing words over every criterion of every case; the first
              failure names its case, criterion and word (the worst element for a tolerance band).
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-6">
            {record === undefined ? null : (
              <RecordActions
                record={record}
                filename={`kernel-check_${fileStamp(new Date(record.recordedAt))}.json`}
              />
            )}
            {notRun.length === 0 ? null : (
              <p className="text-sm text-muted-foreground">
                Not run:{" "}
                {notRun
                  .map((id) => KERNEL_CHECKS.find((check) => check.id === id)?.title ?? id)
                  .join(", ")}{" "}
                — the overall verdict stays INCOMPLETE until every check has run.
              </p>
            )}
            {ran.map(({ id, outcome }) => (
              <CheckResult
                key={id}
                check={KERNEL_CHECKS.find((check) => check.id === id)}
                outcome={outcome}
              />
            ))}
          </CardContent>
        </Card>
      )}
    </div>
  );
};

const CheckResult = ({
  check,
  outcome,
}: {
  check: KernelCheck | undefined;
  outcome: CheckOutcome;
}) => (
  <section className="flex flex-col gap-3">
    <div className="flex flex-wrap items-baseline gap-3">
      <h2 className="text-lg font-semibold">{check?.title}</h2>
      <VerdictBadge verdict={verdictOf(outcome)} />
      <ProductVerdict outcome={outcome} />
      {outcome.kind === "not-available" ? null : (
        <span className="text-sm text-muted-foreground">{outcome.seconds} s</span>
      )}
    </div>
    {outcome.kind === "not-available" ? (
      <p className="text-sm">{outcome.reason}</p>
    ) : outcome.kind === "error" ? (
      <pre className="max-h-80 overflow-auto rounded-md bg-muted p-3 text-xs whitespace-pre-wrap">
        {outcome.message}
      </pre>
    ) : (
      <ReportTables report={outcome.report} />
    )}
  </section>
);

/** The verdict without the classes kept only for K-95 (the strict verdict fails on them on D3D12). */
const ProductVerdict = ({ outcome }: { outcome: CheckOutcome }) => {
  const verdict = productVerdictOf(outcome);
  if (verdict === undefined) return null;
  return (
    <span className="flex items-baseline gap-2 text-sm text-muted-foreground">
      outside the K-95 classes <VerdictBadge verdict={verdict} />
    </span>
  );
};

const ReportTables = ({ report }: { report: CheckReport }) => {
  const parts = report.parts.map(summarizePart);
  const mismatches = countMismatches(report);
  return (
    <>
      <PartTable parts={parts} />
      {mismatches.length === 0 ? null : (
        <ul className="font-mono text-xs text-destructive">
          {mismatches.map((mismatch) => (
            <li key={mismatch}>count: {mismatch}</li>
          ))}
        </ul>
      )}
      {parts.map((part) =>
        part.firstFailure === undefined ? null : <FirstFailureLine key={part.id} part={part} />,
      )}
      {report.mutants === undefined ? null : <MutantTable report={report} />}
      {report.mutantMatrix === undefined ? null : <MutantMatrixTable report={report} />}
      {report.foldClasses === undefined ? null : <FoldClassTable report={report} />}
    </>
  );
};

const PartTable = ({ parts }: { parts: readonly PartSummary[] }) => (
  <Table>
    <TableHeader>
      <TableRow>
        <TableHead>Part</TableHead>
        <TableHead className="text-right">Cases</TableHead>
        <TableHead className="text-right">Failing cases</TableHead>
        <TableHead className="text-right">Mismatching words</TableHead>
        <TableHead>Requirements</TableHead>
      </TableRow>
    </TableHeader>
    <TableBody>
      {parts.map((part) => (
        <TableRow key={part.id}>
          <TableCell>{part.title}</TableCell>
          <TableCell className="text-right font-mono text-xs">{part.cases}</TableCell>
          <TableCell className="text-right font-mono text-xs">{part.failingCases}</TableCell>
          <TableCell className="text-right font-mono text-xs">{part.mismatchingWords}</TableCell>
          <TableCell className="text-xs whitespace-normal">
            {part.unmetRequirements.length === 0 ? (
              "all met"
            ) : (
              <ul className="text-destructive">
                {part.unmetRequirements.map(({ id, name, detail }) => (
                  <li key={id}>
                    {name} — {detail}
                  </li>
                ))}
              </ul>
            )}
          </TableCell>
        </TableRow>
      ))}
    </TableBody>
  </Table>
);

const FirstFailureLine = ({ part }: { part: PartSummary }) => {
  const first = part.firstFailure;
  if (first === undefined) return null;
  const { failure } = first;
  return (
    <p className="font-mono text-xs break-all text-destructive">
      {part.id}: {failure.at} failure in {first.case} — {first.criterion} — word {failure.index}:
      expected {hexWord(failure.expected)}, actual {hexWord(failure.actual)}
      {failure.note === undefined ? "" : ` (${failure.note})`}
    </p>
  );
};

const MutantTable = ({ report }: { report: CheckReport }) => (
  <Table>
    <TableHeader>
      <TableRow>
        <TableHead>Mutant (must mismatch)</TableHead>
        <TableHead>Kernel</TableHead>
        <TableHead>Case</TableHead>
        <TableHead className="text-right">Differing words</TableHead>
        <TableHead>Verdict</TableHead>
      </TableRow>
    </TableHeader>
    <TableBody>
      {(report.mutants ?? []).map((mutant) => (
        <TableRow key={mutant.label}>
          <TableCell className="text-xs whitespace-normal">{mutant.label}</TableCell>
          <TableCell className="font-mono text-xs">{mutant.kernel}</TableCell>
          <TableCell className="font-mono text-xs">{mutant.case}</TableCell>
          <TableCell className="text-right font-mono text-xs">
            {mutant.differingWords} / {mutant.words}
          </TableCell>
          <TableCell>
            <VerdictBadge verdict={mutantMismatched(mutant) ? "pass" : "fail"} />
          </TableCell>
        </TableRow>
      ))}
    </TableBody>
  </Table>
);

const MutantMatrixTable = ({ report }: { report: CheckReport }) => (
  <div className="flex flex-col gap-2">
    <p className="text-sm text-muted-foreground">
      Each mutant on every case of its shape: every case its code path runs on must mismatch, no
      other case may; the contraction-only cases mismatch only where the backend fuses the
      multiply-add (no requirement).
    </p>
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Mutant</TableHead>
          <TableHead>Acts when</TableHead>
          <TableHead className="text-right">Acting cases mismatched</TableHead>
          <TableHead className="text-right">Idle cases mismatched</TableHead>
          <TableHead className="text-right">Contraction-only mismatched</TableHead>
          <TableHead>Verdict</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {(report.mutantMatrix ?? []).map((row) => (
          <TableRow key={row.label}>
            <TableCell className="text-xs whitespace-normal">
              {row.label}
              {row.actingMatched.length === 0 ? null : (
                <span className="block text-destructive">
                  matched: {row.actingMatched.join("; ")}
                </span>
              )}
              {row.idleMismatched.length === 0 ? null : (
                <span className="block text-destructive">
                  idle but mismatched: {row.idleMismatched.join("; ")}
                </span>
              )}
            </TableCell>
            <TableCell className="text-xs whitespace-normal">{row.actsWhen}</TableCell>
            <TableCell className="text-right font-mono text-xs">
              {row.acting - row.actingMatched.length} / {row.acting}
            </TableCell>
            <TableCell className="text-right font-mono text-xs">
              {row.idleMismatched.length} / {row.idle}
            </TableCell>
            <TableCell className="text-right font-mono text-xs">
              {row.contractionMismatched} / {row.contraction}
            </TableCell>
            <TableCell>
              <VerdictBadge verdict={matrixRowPasses(row) ? "pass" : "fail"} />
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  </div>
);

const FoldClassTable = ({ report }: { report: CheckReport }) => (
  <div className="flex flex-col gap-2">
    <p className="text-sm text-muted-foreground">
      Folded ①′ classes: the product folds only the classes of STATE_QK_FOLD_CLASSES (bit-identical
      on Vulkan and D3D12); E4B&apos;s F = 4 is checked but not selected (it differed on D3D12 —
      perf-ledger K-95), so failing cases confined to it leave the product intact.
    </p>
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Class</TableHead>
          <TableHead className="text-right">r · D · M</TableHead>
          <TableHead className="text-right">F checked</TableHead>
          <TableHead className="text-right">F in the product</TableHead>
          <TableHead>Use</TableHead>
          <TableHead className="text-right">Failing cases</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {foldClassRows(report).map((row) => (
          <TableRow key={row.group}>
            <TableCell className="font-mono text-xs">{row.group}</TableCell>
            <TableCell className="text-right font-mono text-xs">
              {row.kvRepeat} · {row.depth} · {row.chunkRows}
            </TableCell>
            <TableCell className="text-right font-mono text-xs">{row.checkedFoldHeads}</TableCell>
            <TableCell className="text-right font-mono text-xs">{row.productFoldHeads}</TableCell>
            <TableCell className="text-xs">{row.status}</TableCell>
            <TableCell
              className={`text-right font-mono text-xs ${row.failingCases > 0 ? "text-destructive" : ""}`}
            >
              {row.failingCases} / {row.cases}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  </div>
);
