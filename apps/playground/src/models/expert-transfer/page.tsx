/**
 * The expert transfer bench: the host round trip of a fence and of a router-pick readback, and the
 * upload of a batch of Gemma 4 26B A4B experts by a staging ring or by `writeBuffer`, on this
 * browser's GPU — the costs an exact expert cache pays per miss round. Optionally it also reads
 * experts back from 26B in the hub's cache (mount `gemma4-qat`, cached source only), the host store
 * an expert cache would read a miss from.
 */

import type { AcquireGpuOptions } from "@karume/runtime";
import { useMemo, useState } from "react";

import { Field } from "@/components/field";
import { RecordActions } from "@/components/record-actions";
import { RunStatusCard } from "@/components/run-status-card";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
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
import { useRunTask } from "@/hooks/use-run-task";
import { fileStamp } from "@/lib/download";
import { type HfCoordinates, hfRepoRef } from "@/lib/distributions";
import { type RecordEnvironment, recordEnvironment } from "@/lib/environment";

import { type CacheReadResult, openCached26b, runCacheRead } from "./cache-read";
import { runTransferBench, type TransferCaseResult } from "./run";
import {
  EXPERT_BYTES,
  gigabytesPerSecond,
  parseBatches,
  ROUTER_READBACK_BYTES,
  transferCases,
} from "./stats";

/** The version of the saved record; bump it when a field changes meaning or goes away. */
const RECORD_FORMAT = "karume-playground-expert-transfer/1";

/** Module constant (useGpu needs stable options); the bench times with the host clock only. */
const GPU_OPTIONS: Omit<AcquireGpuOptions, "onDeviceLost"> = {};

/** The largest batch the form accepts (32 experts = 107 MB of staging and as much of targets). */
const MAX_BATCH = 32;

type TransferRecord = RecordEnvironment & {
  readonly format: typeof RECORD_FORMAT;
  readonly page: "expert-transfer";
  readonly settings: {
    readonly samples: number;
    readonly warmup: number;
    readonly batches: readonly number[];
    readonly cacheReadBatches: readonly number[];
    readonly expertBytes: number;
  };
  readonly cases: readonly TransferCaseResult[];
  /** Present when cached reads ran: the distribution they read and their results. */
  readonly cacheReads?: {
    readonly distribution: HfCoordinates;
    readonly model: "26b";
    readonly results: readonly CacheReadResult[];
  };
};

type Form = {
  readonly samples: string;
  readonly warmup: string;
  readonly batches: string;
  /** Experts per cached read sample; empty skips the cached reads. */
  readonly cacheBatches: string;
};

const INITIAL_FORM: Form = {
  samples: "50",
  warmup: "5",
  batches: "1, 4, 8, 16",
  cacheBatches: "1, 8",
};

/** The mount whose cached 26B the reads draw from. */
const CACHE_MOUNT = "gemma4-qat";

const count = (text: string, min: number): number | undefined => {
  const value = Number(text);
  return Number.isSafeInteger(value) && value >= min ? value : undefined;
};

const formatMs = (ms: number | undefined): string => (ms === undefined ? "" : ms.toFixed(3));

const caseLabel = (result: TransferCaseResult): string => {
  switch (result.kind) {
    case "fence-empty":
      return "Fence, empty submit";
    case "fence-copy":
      return "Fence after a 4-byte copy";
    case "readback":
      return `Readback of ${ROUTER_READBACK_BYTES} B (30 layers × 8 ids)`;
    case "upload":
      return `${result.experts} expert${result.experts === 1 ? "" : "s"} by ${result.method === "ring" ? "staging ring" : "writeBuffer"}`;
  }
};

export const ExpertTransferPage = () => {
  const gpuHandle = useGpu(GPU_OPTIONS);
  const { state: gpuState, reacquire } = gpuHandle;
  const gpu = gpuState.kind === "ready" ? gpuState.gpu : undefined;
  const task = useRunTask<TransferRecord>(gpuHandle);
  const [form, setForm] = useState<Form>(INITIAL_FORM);
  const index = useDistributionIndex();
  const hf = index.kind === "ready" ? index.index[CACHE_MOUNT]?.hf : undefined;
  const hfRef = useMemo(
    () => (hf === undefined ? undefined : hfRepoRef(hf, document.baseURI)),
    [hf],
  );

  const samples = count(form.samples, 1);
  const warmup = count(form.warmup, 0);
  const batches = parseBatches(form.batches, MAX_BATCH);
  const cacheBatches =
    form.cacheBatches.trim() === "" ? [] : parseBatches(form.cacheBatches, MAX_BATCH);
  const formError =
    samples === undefined
      ? "Samples must be a positive integer"
      : warmup === undefined
        ? "Warm-up must be a non-negative integer"
        : batches === undefined
          ? `Batches must be distinct integers from 1 to ${MAX_BATCH}, separated by commas`
          : cacheBatches === undefined
            ? `Cached reads must be empty or distinct integers from 1 to ${MAX_BATCH}`
            : cacheBatches.length > 0 && hf === undefined
              ? `Cached reads need the mount ${CACHE_MOUNT} on the HF-compatible route`
              : undefined;
  const canRun = gpu !== undefined && !task.running && formError === undefined;

  const start = () => {
    if (gpu === undefined || samples === undefined || warmup === undefined) return;
    if (batches === undefined || cacheBatches === undefined) return;
    const cases = transferCases(batches);
    void task.start(async ({ signal, report }) => {
      const results = await runTransferBench(gpu.device, {
        cases,
        samples,
        warmup,
        signal,
        onProgress: ({ caseId, done, total }) =>
          report({ stage: caseId, count: { done, total, unit: "samples" } }),
      });
      const cacheReads =
        cacheBatches.length === 0 || hf === undefined || hfRef === undefined
          ? undefined
          : {
              distribution: hf,
              model: "26b" as const,
              results: await runCacheRead({
                ...(await openCached26b(hfRef, signal)),
                batches: cacheBatches,
                samples,
                warmup,
                signal,
                onProgress: ({ caseId, done, total }) =>
                  report({ stage: caseId, count: { done, total, unit: "samples" } }),
              }),
            };
      return {
        format: RECORD_FORMAT,
        page: "expert-transfer",
        ...recordEnvironment(gpu),
        settings: {
          samples,
          warmup,
          batches,
          cacheReadBatches: cacheBatches,
          expertBytes: EXPERT_BYTES,
        },
        cases: results,
        ...(cacheReads === undefined ? {} : { cacheReads }),
      };
    });
  };

  const record = task.state.kind === "done" ? task.state.result : undefined;

  return (
    <div className="flex flex-col gap-6">
      <h1 className="text-2xl font-semibold">Expert transfer bench</h1>
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <Card>
          <CardHeader>
            <CardTitle>Settings</CardTitle>
            <CardDescription>
              Host wall time from the first call to the resolved promise. One expert is{" "}
              {EXPERT_BYTES.toLocaleString("en-US")} B (Gemma 4 26B A4B at q4_0). The staging ring
              maps its slots ahead, so its time is the memcpy, the copy and the fence; the remap is
              shown apart.
            </CardDescription>
          </CardHeader>
          <CardContent className="grid gap-4 sm:grid-cols-3">
            <Field id="transfer-samples" label="Samples" hint="Per case, after the warm-up.">
              <Input
                id="transfer-samples"
                inputMode="numeric"
                value={form.samples}
                onChange={(event) => setForm({ ...form, samples: event.target.value })}
                disabled={task.running}
              />
            </Field>
            <Field id="transfer-warmup" label="Warm-up" hint="Samples dropped first.">
              <Input
                id="transfer-warmup"
                inputMode="numeric"
                value={form.warmup}
                onChange={(event) => setForm({ ...form, warmup: event.target.value })}
                disabled={task.running}
              />
            </Field>
            <Field id="transfer-batches" label="Experts per batch" hint={`1 to ${MAX_BATCH}.`}>
              <Input
                id="transfer-batches"
                value={form.batches}
                onChange={(event) => setForm({ ...form, batches: event.target.value })}
                disabled={task.running}
              />
            </Field>
            <div className="sm:col-span-3">
              <Field
                id="transfer-cache-batches"
                label="Experts per cached read"
                hint={`Reads that many experts (4 ranges each, at random positions) from 26B in the hub's cache (${CACHE_MOUNT}); empty skips. Nothing is downloaded: the run fails if 26B is not fully cached.`}
              >
                <Input
                  id="transfer-cache-batches"
                  value={form.cacheBatches}
                  onChange={(event) => setForm({ ...form, cacheBatches: event.target.value })}
                  disabled={task.running}
                />
              </Field>
            </div>
            <div className="flex gap-3 sm:col-span-3">
              <Button id="transfer-run" onClick={start} disabled={!canRun}>
                Run
              </Button>
            </div>
            {formError === undefined ? null : (
              <p className="text-sm text-destructive sm:col-span-3">{formError}</p>
            )}
          </CardContent>
        </Card>
        <RunStatusCard gpu={gpuState} onReacquire={reacquire} task={task} />
      </div>

      {record === undefined ? null : (
        <Card>
          <CardHeader>
            <CardTitle>Result</CardTitle>
            <CardDescription>
              ms per sample over {record.settings.samples} samples; GB/s = bytes / median.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <RecordActions
              record={record}
              filename={`expert-transfer_${fileStamp(new Date(record.recordedAt))}.json`}
            />
            <ResultTable results={record.cases} />
            {record.cacheReads === undefined ? null : (
              <CacheReadTable results={record.cacheReads.results} />
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
};

const ResultTable = ({ results }: { results: readonly TransferCaseResult[] }) => (
  <Table>
    <TableHeader>
      <TableRow>
        <TableHead>Case</TableHead>
        <TableHead className="text-right">Bytes</TableHead>
        <TableHead className="text-right">min ms</TableHead>
        <TableHead className="text-right">median ms</TableHead>
        <TableHead className="text-right">p95 ms</TableHead>
        <TableHead className="text-right">max ms</TableHead>
        <TableHead className="text-right">GB/s</TableHead>
        <TableHead className="text-right">ms / expert</TableHead>
        <TableHead className="text-right">remap median ms</TableHead>
      </TableRow>
    </TableHeader>
    <TableBody>
      {results.map((result) => {
        const rate = gigabytesPerSecond(result.bytes, result.ms.median);
        return (
          <TableRow key={result.id}>
            <TableCell>{caseLabel(result)}</TableCell>
            <TableCell className="text-right font-mono text-xs">
              {result.bytes.toLocaleString("en-US")}
            </TableCell>
            <TableCell className="text-right font-mono text-xs">
              {formatMs(result.ms.min)}
            </TableCell>
            <TableCell className="text-right font-mono text-xs">
              {formatMs(result.ms.median)}
            </TableCell>
            <TableCell className="text-right font-mono text-xs">
              {formatMs(result.ms.p95)}
            </TableCell>
            <TableCell className="text-right font-mono text-xs">
              {formatMs(result.ms.max)}
            </TableCell>
            <TableCell className="text-right font-mono text-xs">
              {rate === undefined ? "" : rate.toFixed(2)}
            </TableCell>
            <TableCell className="text-right font-mono text-xs">
              {result.experts === undefined ? "" : formatMs(result.ms.median / result.experts)}
            </TableCell>
            <TableCell className="text-right font-mono text-xs">
              {formatMs(result.remapMs?.median)}
            </TableCell>
          </TableRow>
        );
      })}
    </TableBody>
  </Table>
);

const CacheReadTable = ({ results }: { results: readonly CacheReadResult[] }) => (
  <Table>
    <TableHeader>
      <TableRow>
        <TableHead>Cached read (26B)</TableHead>
        <TableHead className="text-right">Bytes</TableHead>
        <TableHead className="text-right">first min ms</TableHead>
        <TableHead className="text-right">first median ms</TableHead>
        <TableHead className="text-right">first p95 ms</TableHead>
        <TableHead className="text-right">first GB/s</TableHead>
        <TableHead className="text-right">ms / expert</TableHead>
        <TableHead className="text-right">again median ms</TableHead>
        <TableHead className="text-right">again GB/s</TableHead>
      </TableRow>
    </TableHeader>
    <TableBody>
      {results.map((result) => {
        const first = gigabytesPerSecond(result.bytes, result.ms.median);
        const again = gigabytesPerSecond(result.bytes, result.warmMs.median);
        return (
          <TableRow key={result.id}>
            <TableCell>
              {result.experts} expert{result.experts === 1 ? "" : "s"} ({result.experts * 4} ranges
              at once)
            </TableCell>
            <TableCell className="text-right font-mono text-xs">
              {result.bytes.toLocaleString("en-US")}
            </TableCell>
            <TableCell className="text-right font-mono text-xs">
              {formatMs(result.ms.min)}
            </TableCell>
            <TableCell className="text-right font-mono text-xs">
              {formatMs(result.ms.median)}
            </TableCell>
            <TableCell className="text-right font-mono text-xs">
              {formatMs(result.ms.p95)}
            </TableCell>
            <TableCell className="text-right font-mono text-xs">
              {first === undefined ? "" : first.toFixed(2)}
            </TableCell>
            <TableCell className="text-right font-mono text-xs">
              {formatMs(result.ms.median / result.experts)}
            </TableCell>
            <TableCell className="text-right font-mono text-xs">
              {formatMs(result.warmMs.median)}
            </TableCell>
            <TableCell className="text-right font-mono text-xs">
              {again === undefined ? "" : again.toFixed(2)}
            </TableCell>
          </TableRow>
        );
      })}
    </TableBody>
  </Table>
);
