/**
 * The session A/B: the wall-clock A/B of `@karume/runtime/tune` (`runSessionAb`, `judgeSessionAb`)
 * on a real Gemma 4 decode, with and without a candidate fusion rule. Two arms load the same model
 * and preset — `base` with no rule, `rule` with `candidate/gelu_mul` of `@karume/candidate-rules`
 * through the gemma pipeline's experimental `fusionRules` — and every timed run is one greedy
 * generation from a fixed prompt whose token ids the runner compares, so identity is token
 * identity. The page shows where the runner is, then per arm the identity, the rule's hits, the
 * dispatches and the per-round times, and the judgement with its numbers and its reason; the record
 * is saved as JSON.
 */

import { GELU_MUL_RULE, GELU_MUL_RULE_NAME } from "@karume/candidate-rules";
import { chatPrompt } from "@karume/models/gemma";
import type { SessionOptions } from "@karume/runtime";
import { judgeSessionAb, runSessionAb } from "@karume/runtime/tune";
import { memo, useMemo, useState } from "react";
import { flushSync } from "react-dom";

import { type Choice, ChoiceSelect } from "@/components/choice-select";
import { ErrorAlert } from "@/components/error-alert";
import { Field } from "@/components/field";
import { KeyValueTable } from "@/components/key-value-table";
import { MountBadges } from "@/components/mount-badges";
import { PresetSelect } from "@/components/preset-select";
import { RecordActions } from "@/components/record-actions";
import { RunStatusCard } from "@/components/run-status-card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
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
import { type GpuOptions, useGpu } from "@/hooks/use-gpu";
import { type ManifestState, useManifest } from "@/hooks/use-manifest";
import { type RunProgress, useRunTask } from "@/hooks/use-run-task";
import { type DistributionStatus, mountStatus, type PipelineSource } from "@/lib/distributions";
import { fileStamp } from "@/lib/download";
import { recordEnvironment } from "@/lib/environment";
import { manifestPresets } from "@/lib/presets";
import { sourceRecord } from "@/lib/record";
import {
  EXPERT_CACHE_FITTED,
  EXPERT_HOST_TIER_ALL,
  GEMMA_MOUNTS,
  type GemmaManifests,
  type GemmaMount,
  gemmaModelChoices,
  hasExpertCache,
  type ModelKey,
  effectiveModelKey,
  modelKeyValue,
  parseExpertCacheSlots,
  parseExpertHostTier,
  parseExpertSpeculationDepth,
  parseModelKey,
} from "@/models/gemma4/settings";
import { buildGemmaPipeline, type GemmaPipeline } from "@/models/gemma4/use-gemma-pipeline";

import { buildSessionAbPageRecord, PACKAGE_VERSIONS, type SessionAbPageRecord } from "./record";
import {
  INITIAL_SESSION_AB_FORM,
  parseSessionAbForm,
  SESSION_AB_BACKENDS,
  SESSION_AB_CAPACITY,
  SESSION_AB_LIMITS,
  SESSION_AB_PROMPT,
  type SessionAbForm,
  type SessionAbModel,
  type SessionAbRequest,
  sessionAbRequest,
} from "./request";
import { type ArmStep, pipelineArm, type RunPosition, runPositions } from "./run";
import { armRow, judgementView } from "./view";

/** Module constant (useGpu needs stable options): subgroups when the adapter has them. */
const GPU_OPTIONS: GpuOptions = { subgroups: "available" };

/** Every arm reads the direct route, so the source stays out of the comparison. */
const DIRECT_SOURCE: PipelineSource = { kind: "direct" };

const BACKEND_CHOICES: readonly Choice<string>[] = [
  { value: "", label: "Choose the backend…", disabled: true },
  ...SESSION_AB_BACKENDS.map((backend) => ({ value: backend, label: backend })),
];

const manifestOf = (state: ManifestState) => (state.kind === "ready" ? state.manifest : undefined);

const positionText = (position: RunPosition | undefined, arm: string): string =>
  position === undefined
    ? arm
    : `${position.round === 0 ? "warm-up round" : `round ${position.round}`} · visit ${position.visit} · ${arm}${position.timed ? " (timed)" : " (untimed)"}`;

/** The status line of one arm step: where the runner is, and how many runs have ended. */
const progressOf = (
  step: ArmStep,
  positions: readonly RunPosition[],
  ended: number,
): RunProgress => {
  const count = { done: ended, total: positions.length, unit: "runs" };
  const next = positions[ended];
  switch (step.step) {
    case "open":
      return { stage: `opening the pipeline · ${positionText(next, step.arm)}`, count };
    case "ran":
      return { stage: `ran · ${positionText(positions[ended - 1], step.arm)}`, count };
    case "close":
      return { stage: `closing the pipeline of ${step.arm}`, count };
  }
};

/** What one press of Run produced. */
type SessionAbResult =
  | { readonly kind: "running"; readonly request: SessionAbRequest }
  | { readonly kind: "done"; readonly record: SessionAbPageRecord };

export const SessionAbPage = () => {
  const index = useDistributionIndex();
  const distributionIndex = index.kind === "ready" ? index.index : undefined;
  const statuses = useMemo(
    (): Readonly<Record<GemmaMount, DistributionStatus | undefined>> => ({
      gemma4: mountStatus(distributionIndex, "gemma4"),
      "gemma4-qat": mountStatus(distributionIndex, "gemma4-qat"),
      "gemma4-qat-mobile": mountStatus(distributionIndex, "gemma4-qat-mobile"),
    }),
    [distributionIndex],
  );
  const regularManifest = useManifest("gemma4", statuses.gemma4);
  const qatManifest = useManifest("gemma4-qat", statuses["gemma4-qat"]);
  const mobileManifest = useManifest("gemma4-qat-mobile", statuses["gemma4-qat-mobile"]);
  const manifestStates = useMemo(
    (): Readonly<Record<GemmaMount, ManifestState>> => ({
      gemma4: regularManifest,
      "gemma4-qat": qatManifest,
      "gemma4-qat-mobile": mobileManifest,
    }),
    [regularManifest, qatManifest, mobileManifest],
  );
  const manifests = useMemo<GemmaManifests>(() => {
    const loaded: GemmaManifests = {};
    for (const mount of GEMMA_MOUNTS) {
      const manifest = manifestOf(manifestStates[mount]);
      if (manifest !== undefined) loaded[mount] = manifest;
    }
    return loaded;
  }, [manifestStates]);

  // The model, the preset and the load options (the gemma4 page's).
  const modelChoices = useMemo(() => gemmaModelChoices(manifests), [manifests]);
  const [chosenModel, setChosenModel] = useState<ModelKey | undefined>(undefined);
  const modelKey = effectiveModelKey(chosenModel, manifests);
  const manifest = modelKey === undefined ? undefined : manifests[modelKey.mount];
  const presets = useMemo(
    () =>
      manifest === undefined || modelKey === undefined
        ? undefined
        : manifestPresets(manifest, modelKey.model),
    [manifest, modelKey],
  );
  const [chosenQuant, setChosenQuant] = useState<string | undefined>(undefined);
  const quant =
    presets === undefined
      ? undefined
      : chosenQuant !== undefined && presets.presets.some(({ name }) => name === chosenQuant)
        ? chosenQuant
        : presets.defaultQuant;
  const [f16Kv, setF16Kv] = useState(false);
  const expertCacheAvailable = modelKey !== undefined && hasExpertCache(modelKey);
  const [slotsText, setSlotsText] = useState("");
  const [hostTierText, setHostTierText] = useState("");
  const [depthText, setDepthText] = useState("");
  const slotsParsed = useMemo(() => parseExpertCacheSlots(slotsText), [slotsText]);
  const hostTierParsed = useMemo(() => parseExpertHostTier(hostTierText), [hostTierText]);
  const depthParsed = useMemo(() => parseExpertSpeculationDepth(depthText), [depthText]);
  const slots = expertCacheAvailable && slotsParsed.ok ? slotsParsed.value : undefined;
  const cacheErrors =
    slots === undefined
      ? expertCacheAvailable && !slotsParsed.ok
        ? slotsParsed.errors
        : []
      : [
          ...(hostTierParsed.ok ? [] : hostTierParsed.errors),
          ...(depthParsed.ok ? [] : depthParsed.errors),
        ];
  const hostBytes = slots !== undefined && hostTierParsed.ok ? hostTierParsed.value : undefined;
  const depth = slots !== undefined && depthParsed.ok ? depthParsed.value : undefined;

  const [form, setForm] = useState<SessionAbForm>(INITIAL_SESSION_AB_FORM);
  const parsed = useMemo(() => parseSessionAbForm(form), [form]);

  const gpu = useGpu(GPU_OPTIONS);
  const readyGpu = gpu.state.kind === "ready" ? gpu.state.gpu : undefined;
  const task = useRunTask<void>(gpu);
  const running = task.running;
  const [result, setResult] = useState<SessionAbResult | undefined>(undefined);

  const distribution = (() => {
    if (modelKey === undefined) return undefined;
    const state = manifestStates[modelKey.mount];
    const status = statuses[modelKey.mount];
    return state.kind === "ready" && status !== undefined
      ? { directory: status.directory, manifestSha256: state.sha256 }
      : undefined;
  })();
  const model: SessionAbModel | undefined =
    modelKey === undefined || quant === undefined
      ? undefined
      : {
          mount: modelKey.mount,
          model: modelKey.model,
          quant,
          ...(f16Kv ? { fullStateStorage: "f16" as const } : {}),
          ...(slots === undefined
            ? {}
            : {
                expertCache: {
                  slots,
                  ...(hostBytes === undefined ? {} : { hostBytes }),
                  ...(depth === undefined ? {} : { depth }),
                },
              }),
        };
  const canRun =
    readyGpu !== undefined &&
    model !== undefined &&
    distribution !== undefined &&
    parsed.ok &&
    cacheErrors.length === 0 &&
    !running;

  const run = (): void => {
    if (!canRun || readyGpu === undefined || model === undefined || !parsed.ok) return;
    const settings = parsed.value;
    const request = sessionAbRequest(model, settings, GELU_MUL_RULE_NAME);
    const environment = recordEnvironment(readyGpu);
    const facts = distribution;
    setResult({ kind: "running", request });
    void task.start(async ({ signal, report, holdTicker }) => {
      const positions = runPositions(request.arms.length, settings.rounds, settings.openPerVisit);
      let ended = 0;
      // Steps come from an arm's open, report and close, all outside the timed spans. The status is
      // rendered at once (flushSync), so no render is left to land in the run that follows, and the
      // ticker is held from the step before a timed run to the step after it.
      const onStep = (step: ArmStep): void => {
        if (step.step === "ran") ended += 1;
        holdTicker(step.step !== "close" && ended < positions.length && positions[ended].timed);
        flushSync(() => report(progressOf(step, positions, ended)));
      };
      const armOf = (name: string, fusionRules: SessionOptions["fusionRules"]) =>
        pipelineArm<GemmaPipeline>({
          name,
          ...(fusionRules === undefined ? {} : { fusionRules }),
          request,
          prompt: (pipeline) =>
            chatPrompt(pipeline.tokenizer, [{ role: "user", content: request.prompt }]),
          build: (options) =>
            buildGemmaPipeline({
              gpu: readyGpu,
              request: {
                mount: model.mount,
                model: model.model,
                quant: model.quant,
                source: DIRECT_SOURCE,
                speculative: false,
                ...(model.fullStateStorage === undefined
                  ? {}
                  : { fullStateStorage: model.fullStateStorage }),
                ...(model.expertCache === undefined ? {} : { expertCache: model.expertCache }),
              },
              signal,
              ...(options.fusionRules === undefined
                ? {}
                : { sessionOptions: { fusionRules: options.fusionRules } }),
              ...(options.onRunDiagnostics === undefined
                ? {}
                : { onRunDiagnostics: options.onRunDiagnostics }),
            }),
          signal,
          onStep,
        });
      const arms = [armOf("base", undefined), armOf("rule", [GELU_MUL_RULE])];
      const record = await runSessionAb({
        arms,
        rounds: settings.rounds,
        openPerVisit: settings.openPerVisit,
        environment: {
          adapter: readyGpu.adapterInfo,
          userAgent: navigator.userAgent,
          backend: settings.backend,
          date: new Date().toISOString(),
        },
      });
      const judgements = judgeSessionAb(record, { minGain: settings.minGain });
      if (facts === undefined) throw new Error("The distribution's facts went away");
      setResult({
        kind: "done",
        record: buildSessionAbPageRecord({
          environment,
          versions: PACKAGE_VERSIONS,
          request,
          distribution: facts,
          source: sourceRecord(DIRECT_SOURCE),
          ruleName: GELU_MUL_RULE_NAME,
          arms: arms.map((arm) => ({ name: arm.name, facts: arm.facts() })),
          record,
          judgements,
        }),
      });
    });
  };

  const missing = GEMMA_MOUNTS.flatMap((mount) => {
    const state = manifestStates[mount];
    return state.kind === "missing" ? [state.directory] : [];
  });

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-2">
        <h1 className="text-2xl font-semibold">Session A/B</h1>
        <MountBadges index={index} mounts={GEMMA_MOUNTS} />
      </div>
      {index.kind === "error" ? (
        <ErrorAlert title="Distribution index" error={index.error} />
      ) : null}
      {GEMMA_MOUNTS.map((mount) => {
        const state = manifestStates[mount];
        return state.kind === "error" ? (
          <ErrorAlert key={mount} title={`Manifest (${mount})`} error={state.error} />
        ) : null;
      })}
      {missing.length === GEMMA_MOUNTS.length ? (
        <ErrorAlert
          title="No Gemma 4 distribution"
          error={new Error(`The dev server has none of ${missing.join(" / ")}.`)}
        />
      ) : null}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,24rem)_minmax(0,1fr)]">
        <div className="flex flex-col gap-6">
          <Card>
            <CardHeader>
              <CardTitle>Pipeline</CardTitle>
              <CardDescription>
                Both arms load this model and preset over the direct route (<code>/models/</code>).
                The arm <code>rule</code> adds the candidate rule <code>{GELU_MUL_RULE_NAME}</code>{" "}
                (<code>@karume/candidate-rules</code>) through the gemma pipeline&apos;s
                experimental <code>fusionRules</code>; <code>base</code> passes none.
              </CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-4">
              <Field id="session-ab-model" label="Model">
                <ChoiceSelect
                  id="session-ab-model"
                  value={modelKey === undefined ? "" : modelKeyValue(modelKey)}
                  choices={modelChoices}
                  onChange={(value) => {
                    setChosenModel(parseModelKey(value));
                    setChosenQuant(undefined);
                  }}
                  disabled={running || modelChoices.length === 0}
                />
              </Field>
              {presets === undefined || quant === undefined ? null : (
                <PresetSelect
                  id="session-ab-preset"
                  value={quant}
                  presets={presets.presets}
                  defaultQuant={presets.defaultQuant}
                  onChange={setChosenQuant}
                  disabled={running}
                />
              )}
              <div className="flex items-center gap-2">
                <Switch
                  id="session-ab-f16-kv"
                  checked={f16Kv}
                  onCheckedChange={setF16Kv}
                  disabled={running}
                />
                <Label htmlFor="session-ab-f16-kv">f16 KV (full layers)</Label>
              </div>
              <Field
                id="session-ab-expert-slots"
                label="Expert cache slots"
                hint={
                  expertCacheAvailable
                    ? `As on the Gemma 4 chat page: one count for every layer, a comma-separated list (one per routed layer) or "${EXPERT_CACHE_FITTED}"; empty = every expert resident. 64 with the host tier "${EXPERT_HOST_TIER_ALL}" runs 26B on a 16 GB GPU.`
                    : "Only for a mixture-of-experts model (Gemma 4 QAT 26B)."
                }
              >
                <Input
                  id="session-ab-expert-slots"
                  value={slotsText}
                  placeholder="off"
                  onChange={(event) => setSlotsText(event.target.value)}
                  disabled={running || !expertCacheAvailable}
                />
              </Field>
              <Field
                id="session-ab-host-tier"
                label="Host tier (GiB)"
                hint={`"${EXPERT_HOST_TIER_ALL}" = every expert in RAM; empty = every miss reads the distribution. Only with the slots.`}
              >
                <Input
                  id="session-ab-host-tier"
                  value={hostTierText}
                  placeholder="off"
                  onChange={(event) => setHostTierText(event.target.value)}
                  disabled={running || slots === undefined}
                />
              </Field>
              <Field
                id="session-ab-depth"
                label="Self-speculation depth"
                hint="Empty = one run per decode token. Only with the slots."
              >
                <Input
                  id="session-ab-depth"
                  inputMode="numeric"
                  value={depthText}
                  placeholder="off"
                  onChange={(event) => setDepthText(event.target.value)}
                  disabled={running || slots === undefined}
                />
              </Field>
              {cacheErrors.length === 0 ? null : (
                <p className="text-xs text-destructive">{cacheErrors.join(" ")}</p>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>A/B</CardTitle>
              <CardDescription>
                A timed run is one greedy generation (seed 0) of the chosen token count from one
                fixed prompt in a sequence of capacity {SESSION_AB_CAPACITY}; its token ids are what
                the runner compares. One untimed warm-up round, then every round visits the arms
                forward and in reverse (base, rule, rule, base).
              </CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-4">
              <div className="grid grid-cols-3 gap-4">
                <Field
                  id="session-ab-rounds"
                  label="Rounds"
                  hint={`≥ ${SESSION_AB_LIMITS.minRounds}`}
                >
                  <Input
                    id="session-ab-rounds"
                    inputMode="numeric"
                    value={form.rounds}
                    onChange={(event) => setForm({ ...form, rounds: event.target.value })}
                    disabled={running}
                  />
                </Field>
                <Field
                  id="session-ab-tokens"
                  label="Tokens per run"
                  hint={`${SESSION_AB_LIMITS.minTokens}–${SESSION_AB_LIMITS.maxTokens}`}
                >
                  <Input
                    id="session-ab-tokens"
                    inputMode="numeric"
                    value={form.tokens}
                    onChange={(event) => setForm({ ...form, tokens: event.target.value })}
                    disabled={running}
                  />
                </Field>
                <Field id="session-ab-min-gain" label="Minimum gain" hint="0.01 = 1%">
                  <Input
                    id="session-ab-min-gain"
                    inputMode="decimal"
                    value={form.minGain}
                    onChange={(event) => setForm({ ...form, minGain: event.target.value })}
                    disabled={running}
                  />
                </Field>
              </div>
              <div className="flex items-center gap-2">
                <Switch
                  id="session-ab-open-per-visit"
                  checked={form.openPerVisit}
                  onCheckedChange={(checked) => setForm({ ...form, openPerVisit: checked })}
                  disabled={running}
                />
                <Label htmlFor="session-ab-open-per-visit">Open and close per visit</Label>
              </div>
              <p className="-mt-2 text-xs text-muted-foreground">
                On: every visit builds its arm&apos;s pipeline, runs it once untimed and once timed,
                and disposes it (two pipelines of a large model do not fit together). Off: both
                pipelines are built once and stay loaded. Building is never timed.
              </p>
              <div className="flex items-center gap-2">
                <Switch
                  id="session-ab-read-counts"
                  checked={form.readCounts}
                  onCheckedChange={(checked) => setForm({ ...form, readCounts: checked })}
                  disabled={running}
                />
                <Label htmlFor="session-ab-read-counts">Read fusion and dispatch counts</Label>
              </div>
              <p className="-mt-2 text-xs text-muted-foreground">
                Passes onRunDiagnostics and records the fusion counts and the planned dispatches of
                each run&apos;s first decode run, or with a self-speculation depth its first verify
                run (depth + 1 rows). The callback runs inside every run, timed ones included.
                Without the expert cache it changes the decode path of both arms alike: greedy
                tokens are then picked on the CPU from the whole logits row instead of on the GPU.
                Under the expert cache decode is on the CPU sampler either way.
              </p>
              <Field
                id="session-ab-backend"
                label="Backend"
                hint="WebGPU does not expose it; the record states what you choose here."
              >
                <ChoiceSelect
                  id="session-ab-backend"
                  value={form.backend}
                  choices={BACKEND_CHOICES}
                  onChange={(backend) => setForm({ ...form, backend })}
                  disabled={running}
                />
              </Field>
              <p className="text-xs text-muted-foreground">Prompt: {SESSION_AB_PROMPT}</p>
              {parsed.ok ? null : (
                <p className="text-xs text-destructive">{parsed.errors.join(" ")}</p>
              )}
              <Button id="session-ab-run" onClick={run} disabled={!canRun}>
                Run
              </Button>
            </CardContent>
          </Card>
        </div>

        <div className="flex flex-col gap-6">
          <RunStatusCard
            gpu={gpu.state}
            onReacquire={gpu.reacquire}
            task={task}
            rows={[
              [
                "Request",
                result === undefined
                  ? "—"
                  : requestText(result.kind === "running" ? result.request : result.record.request),
              ],
            ]}
          />
          {result?.kind === "done" ? <SessionAbResultView record={result.record} /> : null}
        </div>
      </div>
    </div>
  );
};

const requestText = (request: SessionAbRequest): string =>
  [
    `${request.mount}/${request.model} ${request.quant}`,
    ...(request.fullStateStorage === undefined ? [] : ["f16 KV"]),
    ...(request.expertCache === undefined
      ? []
      : [
          `expert cache ${JSON.stringify(request.expertCache.slots)}${
            request.expertCache.hostBytes === undefined ? "" : " + host tier"
          }${request.expertCache.depth === undefined ? "" : ` depth ${request.expertCache.depth}`}`,
        ]),
    `${request.rounds} rounds × ${request.tokens} tokens`,
    request.openPerVisit ? "open per visit" : "kept open",
    request.readCounts ? "counts" : "no counts",
    request.backend,
  ].join(" · ");

/** Memoized: the status ticker re-renders the page every 250 ms while an untimed step is going. */
const SessionAbResultView = memo(({ record }: { record: SessionAbPageRecord }) => {
  const rows = record.record.arms.map((arm) => armRow(arm, record.rule.name));
  const rounds = record.record.rounds;
  return (
    <>
      <Card>
        <CardHeader>
          <CardTitle>Arms</CardTitle>
          <CardDescription>
            Identity is against the base&apos;s first run, over every run (the warm-up and the
            untimed runs included). Hits are the counter of <code>{record.rule.name}</code> in the
            arm&apos;s first decode run (its first verify run with a self-speculation depth);
            dispatches are that run&apos;s planned dispatches. Times are the timed runs of each
            round, visit 1 / visit 2.
          </CardDescription>
        </CardHeader>
        <CardContent className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Arm</TableHead>
                <TableHead>Identity</TableHead>
                <TableHead className="text-right">Hits</TableHead>
                <TableHead className="text-right">Dispatches</TableHead>
                {Array.from({ length: rounds }, (_, round) => (
                  <TableHead key={round} className="text-right">
                    Round {round + 1}
                  </TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((row) => (
                <TableRow key={row.name}>
                  <TableCell className="font-mono text-xs">{row.name}</TableCell>
                  <TableCell
                    className={`text-xs whitespace-normal ${row.identical ? "" : "text-destructive"}`}
                  >
                    {row.identity}
                  </TableCell>
                  <TableCell className="text-right font-mono text-xs">{row.hits}</TableCell>
                  <TableCell className="text-right font-mono text-xs">{row.dispatches}</TableCell>
                  {row.rounds.map((times, round) => (
                    <TableCell key={round} className="text-right font-mono text-xs">
                      {times}
                    </TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
      {record.judgements.map((judgement) => {
        const view = judgementView(judgement);
        return (
          <Card key={view.arm}>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                Judgement: {view.arm} against {view.base}
                <Badge id={`session-ab-verdict-${view.arm}`} variant={view.tone}>
                  {view.label}
                </Badge>
              </CardTitle>
              <CardDescription>
                Minimum gain {(record.request.minGain * 100).toFixed(2)}%, identity required. The
                verdict speaks for this device, backend and work only; inconclusive does not mean
                the difference is zero.
              </CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-3">
              <p className="text-sm">{view.reason}</p>
              <KeyValueTable rows={view.numbers} />
            </CardContent>
          </Card>
        );
      })}
      <Card>
        <CardHeader>
          <CardTitle>Record</CardTitle>
          <CardDescription>
            The runtime&apos;s <code>{record.record.format}</code> record unchanged, inside the
            page&apos;s envelope with the request, the distribution and the judgements.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <RecordActions
            record={record}
            filename={`session-ab_${fileStamp(new Date(record.recordedAt))}.json`}
          />
        </CardContent>
      </Card>
    </>
  );
});
SessionAbResultView.displayName = "SessionAbResultView";
