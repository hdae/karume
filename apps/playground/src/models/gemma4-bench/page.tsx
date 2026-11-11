/**
 * The Gemma 4 decode bench: A/B arms of regular Gemma 4 and its q4_0 QAT line (`gemma.Pipeline`)
 * and the mobile QAT distribution (`gemma4QatMobile.Pipeline`) — model × preset, with the GPU tap on
 * or off — run over a
 * fixed prompt set in interleaved rounds, with decode rates from the stream timings and the tap's GPU
 * busy time, dispatches, passes and submits per token, and a record of every chat. A split arm runs
 * every dispatch in its own timed pass and breaks decode GPU time down per op family and pipeline
 * key.
 *
 * The page's GPU is acquired through the tap (`tap.ts`), so karume's timing mode stays off and the
 * product's own pass structure is what is timed. One pipeline is resident at a time: each visit
 * builds its arm's pipeline and releases it before the next.
 */

import { chatPrompt } from "@karume/models/gemma";
import { PlusIcon, Trash2Icon } from "lucide-react";
import { memo, useMemo, useState } from "react";

import { ChoiceSelect } from "@/components/choice-select";
import { ErrorAlert } from "@/components/error-alert";
import { Field } from "@/components/field";
import { MountBadges } from "@/components/mount-badges";
import { PresetSelect } from "@/components/preset-select";
import { RecordActions } from "@/components/record-actions";
import { RunStatusCard } from "@/components/run-status-card";
import { SourcePicker } from "@/components/source-picker";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
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
import {
  type DistributionStatus,
  hfRepoRef,
  mountStatus,
  type PipelineSource,
  type SourceKind,
} from "@/lib/distributions";
import { fileStamp } from "@/lib/download";
import { type RecordEnvironment, recordEnvironment } from "@/lib/environment";
import { manifestPresets } from "@/lib/presets";
import { sha256Hex, type SourceRecord, sourceRecord, type Spread } from "@/lib/record";
import {
  GEMMA_MOUNTS,
  type GemmaManifests,
  type GemmaMount,
  gemmaModelChoices,
  modelKeyValue,
  parseModelKey,
} from "@/models/gemma4/settings";
import { buildGemmaPipeline, type GemmaPipeline } from "@/models/gemma4/use-gemma-pipeline";

import {
  type ArmDraft,
  armLabel,
  BENCH_PROMPTS,
  type BenchForm,
  type BenchPromptId,
  type BenchSettings,
  benchPrompt,
  DECODE_AHEAD_CHOICES,
  defaultArmDrafts,
  INITIAL_BENCH_FORM,
  parseBenchForm,
  PLE_DISTINCT_ID_THRESHOLD,
  resolveArm,
  SAMPLER_CHOICES,
  samplerOf,
} from "./plan";
import { OP_FAMILY_LABELS } from "./family";
import { type ArmDistribution, buildBenchRecord, chromeEnvironmentKey } from "./record";
import {
  type BenchPipeline,
  type BenchProgress,
  type ChatRecord,
  type RunArm,
  runDecodeBench,
  type VisitRecord,
} from "./run";
import {
  type ArmPromptSummary,
  shareText,
  type SplitSummary,
  splitSummaryText,
  summarizeRuns,
} from "./summary";
import { createGpuTap, type DeviceProbe, type GpuTap } from "./tap";

/** What one press of Run started, and what has landed since (kept when a later step fails). */
type BenchResult = {
  readonly environment: RecordEnvironment;
  readonly timestampQuery: boolean;
  readonly settings: BenchSettings;
  readonly arms: readonly RunArm[];
  readonly distributions: Partial<Record<GemmaMount, ArmDistribution>>;
  /** The source each mount was read from in this run. */
  readonly sources: Readonly<Record<GemmaMount, SourceRecord>>;
  readonly visits: readonly VisitRecord[];
  readonly chats: readonly ChatRecord[];
  readonly complete: boolean;
};

const manifestOf = (state: ManifestState) => (state.kind === "ready" ? state.manifest : undefined);

/** What the loop needs of a Gemma 4 pipeline (its own tokenizer renders the prompt). */
const benchPipelineOf = (pipeline: GemmaPipeline): BenchPipeline => ({
  encodePrompt: (messages) => chatPrompt(pipeline.tokenizer, messages),
  chat: (messages, options) => pipeline.chat(messages, options),
  dispose: () => pipeline.dispose(),
});

const progressOf = (
  progress: BenchProgress,
  arms: readonly RunArm[],
  chats: number,
): RunProgress => {
  const count = { done: progress.visit.visit, total: progress.visits, unit: "visits" };
  const where = `visit ${progress.visit.visit + 1} / ${progress.visits} (round ${progress.visit.round + 1}) · arm ${progress.visit.arm + 1} ${armLabel(arms[progress.visit.arm])}`;
  return progress.kind === "loading"
    ? { stage: `${where} · building the pipeline`, count }
    : {
        stage: `${where} · ${benchPrompt(progress.prompt).label} chat ${progress.chat + 1} / ${chats}${progress.chat === 0 ? " (cold)" : ""}`,
        count,
      };
};

export const Gemma4BenchPage = () => {
  // The tap wraps the browser's GPU once per page instance (nothing is touched until acquireGpu).
  const [tap] = useState<GpuTap | undefined>(() => {
    const gpu: GPU | undefined = navigator.gpu;
    return gpu === undefined ? undefined : createGpuTap(gpu);
  });
  if (tap === undefined) {
    return <ErrorAlert title="GPU" error={new Error("This browser has no navigator.gpu")} />;
  }
  return <BenchBody tap={tap} />;
};

const BenchBody = ({ tap }: { tap: GpuTap }) => {
  // Stable for the page's life (useGpu re-acquires when its options object changes).
  const [gpuOptions] = useState<GpuOptions>(() => ({ subgroups: "available", gpu: tap.gpu }));
  const gpu = useGpu(gpuOptions);
  const readyGpu = gpu.state.kind === "ready" ? gpu.state.gpu : undefined;
  const probe: DeviceProbe | undefined =
    readyGpu === undefined ? undefined : tap.probe(readyGpu.device);

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
  // One choice for every arm: the cached route where the arm's mount has one, the direct route
  // otherwise (a bench builds every arm's pipeline once per visit, and the direct route fetches the
  // weights anew each time).
  const [sourceKind, setSourceKind] = useState<SourceKind>("cached");
  const sources = useMemo((): Readonly<Record<GemmaMount, PipelineSource>> => {
    const sourceOf = (mount: GemmaMount): PipelineSource => {
      const hf = statuses[mount]?.hf;
      return sourceKind === "cached" && hf !== undefined
        ? { kind: "cached", ref: hfRepoRef(hf, document.baseURI), cacheGeneration: 0 }
        : { kind: "direct" };
    };
    return {
      gemma4: sourceOf("gemma4"),
      "gemma4-qat": sourceOf("gemma4-qat"),
      "gemma4-qat-mobile": sourceOf("gemma4-qat-mobile"),
    };
  }, [sourceKind, statuses]);
  const cachedAvailable = GEMMA_MOUNTS.some((mount) => statuses[mount]?.hf !== undefined);
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
  const distributions = useMemo(() => {
    const facts: Partial<Record<GemmaMount, ArmDistribution>> = {};
    for (const mount of GEMMA_MOUNTS) {
      const state = manifestStates[mount];
      const status = statuses[mount];
      if (state.kind === "ready" && status !== undefined) {
        facts[mount] = { directory: status.directory, manifestSha256: state.sha256 };
      }
    }
    return facts;
  }, [manifestStates, statuses]);
  const missing = GEMMA_MOUNTS.flatMap((mount) => {
    const state = manifestStates[mount];
    return state.kind === "missing" ? [state.directory] : [];
  });

  // The arms: the defaults until edited.
  const [armEdit, setArmEdit] = useState<readonly ArmDraft[] | undefined>(undefined);
  const drafts = useMemo(() => armEdit ?? defaultArmDrafts(manifests), [armEdit, manifests]);
  const resolutions = useMemo(
    () => drafts.map((draft) => resolveArm(draft, manifests)),
    [drafts, manifests],
  );
  const modelChoices = useMemo(() => gemmaModelChoices(manifests), [manifests]);
  const updateDraft = (id: number, patch: Partial<Omit<ArmDraft, "id">>): void =>
    setArmEdit(drafts.map((draft) => (draft.id === id ? { ...draft, ...patch } : draft)));
  const removeDraft = (id: number): void => setArmEdit(drafts.filter((draft) => draft.id !== id));
  const addDraft = (): void => {
    const last = drafts.at(-1);
    const key = last?.key ?? defaultArmDrafts(manifests)[0]?.key;
    if (key === undefined) return;
    const id = Math.max(-1, ...drafts.map((draft) => draft.id)) + 1;
    setArmEdit([
      ...drafts,
      { id, key, ...(last?.quant === undefined ? {} : { quant: last.quant }), tap: true },
    ]);
  };

  const [form, setForm] = useState<BenchForm>(INITIAL_BENCH_FORM);
  const parsed = useMemo(() => parseBenchForm(form), [form]);
  const togglePrompt = (id: BenchPromptId, checked: boolean): void =>
    setForm({
      ...form,
      prompts: checked ? [...form.prompts, id] : form.prompts.filter((prompt) => prompt !== id),
    });

  const task = useRunTask<void>(gpu);
  const running = task.running;
  const [result, setResult] = useState<BenchResult | undefined>(undefined);

  const arms = resolutions.flatMap((resolution) => (resolution.ok ? [resolution.arm] : []));
  const armErrors = resolutions.flatMap((resolution, armIndex) =>
    resolution.ok ? [] : [`Arm ${armIndex + 1}: ${resolution.error}`],
  );
  const timingRefused = readyGpu?.gpuTimingEnabled === true;
  const canRun =
    readyGpu !== undefined &&
    probe !== undefined &&
    !timingRefused &&
    parsed.ok &&
    drafts.length > 0 &&
    armErrors.length === 0 &&
    arms.every(({ mount }) => distributions[mount] !== undefined) &&
    !running;

  const run = (): void => {
    if (!canRun || !parsed.ok || readyGpu === undefined || probe === undefined) return;
    const settings = parsed.value;
    const runArms: readonly RunArm[] = arms.map((arm) => ({
      ...arm,
      sampler: samplerOf(settings.sampler, arm.declaredSampler),
    }));
    const started: BenchResult = {
      environment: recordEnvironment(readyGpu),
      timestampQuery: probe.timestamps,
      settings,
      arms: runArms,
      distributions,
      sources: {
        gemma4: sourceRecord(sources.gemma4),
        "gemma4-qat": sourceRecord(sources["gemma4-qat"]),
        "gemma4-qat-mobile": sourceRecord(sources["gemma4-qat-mobile"]),
      },
      visits: [],
      chats: [],
      complete: false,
    };
    setResult(started);
    // Updates only the result this press started (a later press replaces it).
    const update = (change: (current: BenchResult) => BenchResult): void =>
      setResult((current) => (current?.arms === runArms ? change(current) : current));
    void task.start(async ({ signal, report }) => {
      await runDecodeBench({
        arms: runArms,
        settings,
        signal,
        dependencies: {
          build: async (arm, buildSignal) =>
            benchPipelineOf(
              await buildGemmaPipeline({
                gpu: readyGpu,
                request: {
                  mount: arm.mount,
                  model: arm.model,
                  quant: arm.quant,
                  source: sources[arm.mount],
                  speculative: false,
                },
                signal: buildSignal,
                ...(arm.session === undefined ? {} : { sessionOptions: arm.session }),
                ...(arm.pipeline === undefined ? {} : { pipelineOptions: arm.pipeline }),
              }),
            ),
          probe,
          now: () => performance.now(),
          sha256: sha256Hex,
        },
        onProgress: (progress) => report(progressOf(progress, runArms, settings.chats)),
        onVisit: (visit) =>
          update((current) => ({ ...current, visits: [...current.visits, visit] })),
        onChat: (chat) => update((current) => ({ ...current, chats: [...current.chats, chat] })),
      });
      update((current) => ({ ...current, complete: true }));
    });
  };

  const record = useMemo(
    () =>
      result === undefined || result.visits.length === 0
        ? undefined
        : buildBenchRecord({
            environment: result.environment,
            timestampQuery: result.timestampQuery,
            complete: result.complete,
            settings: result.settings,
            arms: result.arms,
            distributions: result.distributions,
            sources: result.sources,
            visits: result.visits,
            chats: result.chats,
          }),
    [result],
  );
  const summary = useMemo(
    () =>
      result === undefined
        ? []
        : summarizeRuns(result.chats, result.arms.length, result.settings.prompts),
    [result],
  );
  const environmentKey =
    readyGpu === undefined
      ? undefined
      : chromeEnvironmentKey(navigator.userAgent, readyGpu.adapterInfo);

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-2">
        <h1 className="text-2xl font-semibold">Gemma 4 decode bench</h1>
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
        <Alert>
          <AlertTitle>No Gemma 4 distribution</AlertTitle>
          <AlertDescription>
            The dev server has none of{" "}
            {missing.map((directory, at) => (
              <span key={directory}>
                {at > 0 ? " / " : ""}
                <code>{directory}</code>
              </span>
            ))}
            . Put a distribution there and reload.
          </AlertDescription>
        </Alert>
      ) : null}
      {probe?.timestamps === false ? (
        <Alert id="bench-no-timestamps">
          <AlertTitle>No timestamp-query on this adapter</AlertTitle>
          <AlertDescription>
            The tap still counts passes, dispatches and submits, but GPU busy time and the wall −
            GPU gap are not measured.
          </AlertDescription>
        </Alert>
      ) : null}
      {timingRefused ? (
        <ErrorAlert
          title="GPU timing is on"
          error={
            new Error(
              "The device reports gpuTimingEnabled: the tap failed to hide timestamp-query, and karume would encode one dispatch per pass. The bench does not run on it.",
            )
          }
        />
      ) : null}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,24rem)_minmax(0,1fr)]">
        <div className="flex flex-col gap-6">
          <Card>
            <CardHeader>
              <CardTitle>Arms</CardTitle>
              <CardDescription>
                A model of either distribution at one preset. Every visit builds the arm&apos;s
                pipeline, runs its chats and releases it before the next visit, so two models are
                never resident at once. Every arm reads its distribution from the source chosen
                below (an arm whose distribution has no cached route reads it directly). Arm 1 is
                the one the summary compares the others with.
              </CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-4">
              <SourcePicker
                id="gemma4-bench-source"
                value={sourceKind === "cached" && cachedAvailable ? "cached" : "direct"}
                cachedAvailable={cachedAvailable}
                onChange={setSourceKind}
                disabled={running}
              />
              {drafts.map((draft, armIndex) => {
                const manifest = manifests[draft.key.mount];
                const presets =
                  manifest?.models[draft.key.model] === undefined
                    ? undefined
                    : manifestPresets(manifest, draft.key.model);
                const resolution = resolutions[armIndex];
                return (
                  <div key={draft.id} className="flex flex-col gap-3 rounded-lg border p-3">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">Arm {armIndex + 1}</span>
                      <Button
                        id={`bench-arm-${draft.id}-remove`}
                        variant="ghost"
                        size="xs"
                        onClick={() => removeDraft(draft.id)}
                        disabled={running || drafts.length === 1}
                      >
                        <Trash2Icon />
                        Remove
                      </Button>
                    </div>
                    <Field id={`bench-arm-${draft.id}-model`} label="Model">
                      <ChoiceSelect
                        id={`bench-arm-${draft.id}-model`}
                        value={modelKeyValue(draft.key)}
                        choices={modelChoices}
                        onChange={(value) => {
                          const key = parseModelKey(value);
                          // A new model starts at its default preset.
                          if (key !== undefined) updateDraft(draft.id, { key, quant: undefined });
                        }}
                        disabled={running || modelChoices.length === 0}
                      />
                    </Field>
                    {presets === undefined || !resolution?.ok ? null : (
                      <PresetSelect
                        id={`bench-arm-${draft.id}-preset`}
                        value={resolution.arm.quant}
                        presets={presets.presets}
                        defaultQuant={presets.defaultQuant}
                        onChange={(quant) => updateDraft(draft.id, { quant })}
                        disabled={running}
                      />
                    )}
                    <div className="flex items-center gap-2">
                      <Switch
                        id={`bench-arm-${draft.id}-tap`}
                        checked={draft.tap}
                        onCheckedChange={(checked) => updateDraft(draft.id, { tap: checked })}
                        disabled={running}
                      />
                      <Label htmlFor={`bench-arm-${draft.id}-tap`}>GPU tap</Label>
                    </div>
                    <div className="flex items-center gap-2">
                      <Switch
                        id={`bench-arm-${draft.id}-split`}
                        checked={draft.tap && draft.split === true}
                        onCheckedChange={(checked) => updateDraft(draft.id, { split: checked })}
                        disabled={running || !draft.tap}
                      />
                      <Label htmlFor={`bench-arm-${draft.id}-split`}>Split dispatches</Label>
                    </div>
                    <div className="flex items-center gap-2">
                      <Switch
                        id={`bench-arm-${draft.id}-integer`}
                        checked={draft.integerGemv === true}
                        onCheckedChange={(checked) =>
                          updateDraft(draft.id, { integerGemv: checked })
                        }
                        disabled={running}
                      />
                      <Label htmlFor={`bench-arm-${draft.id}-integer`}>Integer GEMV</Label>
                    </div>
                    <div className="flex items-center gap-2">
                      <Switch
                        id={`bench-arm-${draft.id}-planes`}
                        checked={draft.linearGemvPlanes === true}
                        onCheckedChange={(checked) =>
                          updateDraft(draft.id, { linearGemvPlanes: checked })
                        }
                        disabled={running}
                      />
                      <Label htmlFor={`bench-arm-${draft.id}-planes`}>Planes GEMV</Label>
                    </div>
                    <div className="flex items-center gap-2">
                      <Switch
                        id={`bench-arm-${draft.id}-ple-gpu`}
                        checked={draft.pleOnGpu === true}
                        onCheckedChange={(checked) => updateDraft(draft.id, { pleOnGpu: checked })}
                        disabled={running}
                      />
                      <Label htmlFor={`bench-arm-${draft.id}-ple-gpu`}>PLE on GPU</Label>
                    </div>
                    <Field id={`bench-arm-${draft.id}-ahead`} label="Decode ahead">
                      <ChoiceSelect
                        id={`bench-arm-${draft.id}-ahead`}
                        value={draft.decodeAhead ?? "1"}
                        choices={DECODE_AHEAD_CHOICES}
                        onChange={(decodeAhead) => updateDraft(draft.id, { decodeAhead })}
                        disabled={running}
                      />
                    </Field>
                    {resolution?.ok === false ? (
                      <p className="text-xs text-destructive">{resolution.error}</p>
                    ) : null}
                  </div>
                );
              })}
              <Button
                id="bench-add-arm"
                variant="outline"
                size="sm"
                className="self-start"
                onClick={addDraft}
                disabled={running || modelChoices.length === 0}
              >
                <PlusIcon />
                Add arm
              </Button>
              <p className="text-xs text-muted-foreground">
                GPU tap on: every compute pass gets a pair of pass-boundary timestamps, and passes,
                dispatches and submits are counted — in the product&apos;s own pass structure
                (karume&apos;s timing mode stays off; it would encode one dispatch per pass and add
                about 4 µs per dispatch, which biases QAT&apos;s extra dispatches). Off: a
                wall-clock control on the same device, with the tap only passing calls through.
                Compare a tap-off arm with a tap-on arm of the same model to see what the tap costs.
                Split dispatches (tap on): every dispatch runs in its own timed pass, labelled with
                its pipeline key, so decode GPU time breaks down per op family and kernel; the extra
                pass boundaries inflate the arm&apos;s absolute GPU and wall times, so read its
                shares, not its totals. Integer GEMV: the arm passes the Session option{" "}
                <code>integerGemv: true</code> (with <code>packedStaticQuantize: true</code> and{" "}
                <code>linearGemvReduce: &quot;parallel&quot;</code> where the preset does not
                declare them), so the decode linears of QAT E2B at the measured shapes (i2 gate / up
                and down, i4 o and down) run on the integer GEMV; the label shows
                &quot;(integer)&quot;. Other models have no linear at those shapes. Planes GEMV: the
                arm passes the Session option <code>linearGemvPlanes: true</code> (it needs no other
                option), so the QAT E2B lm_head (i2 262144 × 1536) runs on the byte-plane GEMV
                instead of the sequential GEMV; the label shows &quot;(planes)&quot;. Other models
                have no i2 linear at that shape. PLE on GPU: the arm passes the pipeline option{" "}
                <code>pleResidency: &quot;gpu&quot;</code>, so the per-layer embeddings are uploaded
                once at load and gathered on the GPU (QAT E2B: 1.09 GiB in one binding, which the
                device must allow); the label shows &quot;(PLE GPU)&quot;. Decode ahead: the arm
                passes <code>decodeAhead</code>, so a greedy chat runs that many decode steps per
                GPU batch, each fed the token the previous step selected on the GPU, with one fence
                and one readback per batch instead of per token; it needs PLE on GPU, and the label
                shows &quot;(ahead N)&quot;. The token ids stay those of depth 1; each chat records
                the batches it ran in its stop (<code>decodeAhead</code>).
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Workload</CardTitle>
              <CardDescription>
                Every visit runs the chosen prompts in this order, each as fresh chats.
              </CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-4">
              <div className="flex flex-col gap-2">
                {BENCH_PROMPTS.map((prompt) => (
                  <Label key={prompt.id} className="flex items-start gap-2 font-normal">
                    <Checkbox
                      id={`bench-prompt-${prompt.id}`}
                      checked={form.prompts.includes(prompt.id)}
                      onCheckedChange={(checked) => togglePrompt(prompt.id, checked)}
                      disabled={running}
                    />
                    <span>
                      {prompt.label}{" "}
                      <span className="text-muted-foreground">
                        ({prompt.distinctIds === "below" ? "fewer than" : "at least"}{" "}
                        {PLE_DISTINCT_ID_THRESHOLD} distinct ids): {prompt.text}
                      </span>
                    </span>
                  </Label>
                ))}
                <p className="text-xs text-muted-foreground">
                  A prompt with {PLE_DISTINCT_ID_THRESHOLD} or more distinct token ids makes a PLE
                  block whole-resident on the host — the condition under which the PLE row cache
                  once stopped admitting rows (perf-ledger H-41). One prompt sits on each side of
                  it; every chat records the distinct count it measured.
                </p>
              </div>
              <div className="grid grid-cols-3 gap-4">
                <Field id="bench-chats" label="Chats" hint="Per prompt per visit; chat 0 is cold.">
                  <Input
                    id="bench-chats"
                    inputMode="numeric"
                    value={form.chats}
                    onChange={(event) => setForm({ ...form, chats: event.target.value })}
                    disabled={running}
                  />
                </Field>
                <Field id="bench-new-tokens" label="New tokens" hint="maxNewTokens per chat.">
                  <Input
                    id="bench-new-tokens"
                    inputMode="numeric"
                    value={form.newTokens}
                    onChange={(event) => setForm({ ...form, newTokens: event.target.value })}
                    disabled={running}
                  />
                </Field>
                <Field id="bench-rounds" label="Rounds" hint="ABBA order.">
                  <Input
                    id="bench-rounds"
                    inputMode="numeric"
                    value={form.rounds}
                    onChange={(event) => setForm({ ...form, rounds: event.target.value })}
                    disabled={running}
                  />
                </Field>
              </div>
              <p className="-mt-2 text-xs text-muted-foreground">
                Chat 0 of each prompt in a visit is the first after the build; it is recorded and
                left out of the summary. Arms are visited in order on even rounds and in reverse on
                odd ones (A B, B A, …), so a slow drift of the machine lands on every arm alike.
              </p>
              <Field
                id="bench-sampler"
                label="Sampler"
                hint="Greedy by default, so every arm and round decodes the same way. The bench never collects Session diagnostics: with them every greedy token reads the whole logits row back, which changes the decode path."
              >
                <ChoiceSelect
                  id="bench-sampler"
                  value={form.sampler}
                  choices={SAMPLER_CHOICES}
                  onChange={(sampler) => setForm({ ...form, sampler })}
                  disabled={running}
                />
              </Field>
              {parsed.ok ? null : (
                <p className="text-xs text-destructive">{parsed.errors.join(" ")}</p>
              )}
              {armErrors.length === 0 ? null : (
                <p className="text-xs text-destructive">{armErrors.join(" ")}</p>
              )}
              <Button id="bench-run" onClick={run} disabled={!canRun}>
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
                "Timing",
                probe === undefined
                  ? "—"
                  : probe.timestamps
                    ? "pass-boundary timestamps (hidden from karume; its timing mode is off)"
                    : "counts only (no timestamp-query)",
              ],
              ["Environment key", environmentKey ?? "— (not a Chromium adapter name)"],
              [
                "Progress",
                result === undefined
                  ? "—"
                  : `${result.visits.length} visits, ${result.chats.length} chats${result.complete ? " (complete)" : ""}`,
              ],
            ]}
          />

          {summary.length === 0 || result === undefined ? null : (
            <Card>
              <CardHeader>
                <CardTitle>Summary</CardTitle>
                <CardDescription>
                  Warm chats only, over every round: median and min (for rates, median and max) —
                  this machine shows transient 5–45× slow kernels, which move a median less than a
                  mean and a min not at all. Decode excludes the prefill and the first token. Chrome
                  rounds GPU timestamps to 100 µs unless
                  chrome://flags/#enable-webgpu-developer-features is on. A split arm&apos;s GPU
                  ms/token, wall − GPU and decode rates include the cost of its extra pass
                  boundaries (one per dispatch).
                </CardDescription>
              </CardHeader>
              <CardContent>
                <SummaryTable summary={summary} arms={result.arms} />
              </CardContent>
            </Card>
          )}

          {result === undefined || !summary.some(({ split }) => split !== undefined) ? null : (
            <Card>
              <CardHeader>
                <CardTitle>Split attribution</CardTitle>
                <CardDescription>
                  Split arms only: decode GPU time per token by op family and pipeline key, each the
                  median over the warm chats of every round (a family or key median, so rows do not
                  add up to the total exactly). Every dispatch ran in its own timed pass, and each
                  extra pass boundary adds GPU time, inflating small kernels most: read the shares,
                  and compare them across arms split alike.
                </CardDescription>
              </CardHeader>
              <CardContent className="flex flex-col gap-6">
                {summary.map((entry) =>
                  entry.split === undefined ? null : (
                    <div key={`${entry.arm}-${entry.prompt}`} className="flex flex-col gap-2">
                      <p className="text-sm font-medium">
                        Arm {entry.arm + 1} · {armLabel(result.arms[entry.arm])} ·{" "}
                        {benchPrompt(entry.prompt).label} · {entry.split.chats} warm chats · GPU{" "}
                        {fixed(entry.split.totalGpuMs, 3)} ms/token
                      </p>
                      <SplitTable split={entry.split} />
                    </div>
                  ),
                )}
                <pre
                  id="split-summary"
                  className="overflow-x-auto rounded-md border p-3 font-mono text-xs"
                >
                  {splitSummaryText(summary, result.arms)}
                </pre>
              </CardContent>
            </Card>
          )}

          {record === undefined ? null : (
            <Card>
              <CardHeader>
                <CardTitle>Record</CardTitle>
                <CardDescription>
                  {record.complete ? "Complete" : "Incomplete (running, failed or cancelled)"}:{" "}
                  {record.visits.length} visits, {record.chats.length} chats (cold ones included),
                  the arms with their distributions, the settings and the summary (split arms: with
                  the per-key breakdown of every chat and its medians).
                </CardDescription>
              </CardHeader>
              <CardContent className="flex flex-col gap-4">
                <RecordActions
                  record={record}
                  filename={`gemma4-bench_${fileStamp(new Date(record.recordedAt))}.json`}
                />
                <ChatTable chats={record.chats} arms={result?.arms ?? []} />
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};

const fixed = (value: number | undefined, digits: number): string =>
  value === undefined ? "—" : value.toFixed(digits);

/** `median · min` (or `· max` for a rate, where higher is better). */
const spreadText = (
  spread: Spread | undefined,
  digits: number,
  best: "min" | "max" = "min",
): string =>
  spread === undefined ? "—" : `${spread.median.toFixed(digits)} · ${spread[best].toFixed(digits)}`;

/** Memoized: the status ticker re-renders the page while chats run. */
const SummaryTable = memo(
  ({ summary, arms }: { summary: readonly ArmPromptSummary[]; arms: readonly RunArm[] }) => (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Arm</TableHead>
          <TableHead>Prompt</TableHead>
          <TableHead className="text-right">Warm</TableHead>
          <TableHead className="text-right">Decode tok/s</TableHead>
          <TableHead className="text-right">Decode ms/token</TableHead>
          <TableHead className="text-right">vs arm 1</TableHead>
          <TableHead className="text-right">TTFT ms</TableHead>
          <TableHead className="text-right">Prefill tok/s</TableHead>
          <TableHead className="text-right">GPU ms/token</TableHead>
          <TableHead className="text-right">Wall − GPU ms</TableHead>
          <TableHead className="text-right">Dispatches · passes · submits / token</TableHead>
          <TableHead className="text-right">Outputs</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {summary.map((entry) => {
          const { spreads } = entry;
          return (
            <TableRow key={`${entry.arm}-${entry.prompt}`}>
              <TableCell className="text-xs">
                {entry.arm + 1} · {armLabel(arms[entry.arm])}
              </TableCell>
              <TableCell className="text-xs">{benchPrompt(entry.prompt).label}</TableCell>
              <TableCell className="text-right font-mono text-xs">{entry.warmChats}</TableCell>
              <TableCell className="text-right font-mono text-xs">
                {spreadText(spreads.decodeTokensPerSecond, 1, "max")}
              </TableCell>
              <TableCell className="text-right font-mono text-xs">
                {spreadText(spreads.decodeMsPerToken, 3)}
              </TableCell>
              <TableCell className="text-right font-mono text-xs">
                {entry.decodeVsFirstArm === undefined
                  ? ""
                  : `×${entry.decodeVsFirstArm.toFixed(3)}`}
              </TableCell>
              <TableCell className="text-right font-mono text-xs">
                {spreadText(spreads.ttftMs, 1)}
              </TableCell>
              <TableCell className="text-right font-mono text-xs">
                {spreadText(spreads.prefillTokensPerSecond, 0, "max")}
              </TableCell>
              <TableCell className="text-right font-mono text-xs">
                {spreadText(spreads.gpuBusyMsPerToken, 3)}
              </TableCell>
              <TableCell className="text-right font-mono text-xs">
                {spreadText(spreads.wallMinusGpuMsPerToken, 3)}
              </TableCell>
              <TableCell className="text-right font-mono text-xs">
                {spreads.dispatchesPerToken === undefined
                  ? "—"
                  : `${fixed(spreads.dispatchesPerToken.median, 1)} · ${fixed(spreads.passesPerToken?.median, 2)} · ${fixed(spreads.submitsPerToken?.median, 2)}`}
              </TableCell>
              <TableCell className="text-right font-mono text-xs">
                {entry.distinctOutputs === 1 ? "identical" : `${entry.distinctOutputs} distinct`}
              </TableCell>
            </TableRow>
          );
        })}
      </TableBody>
    </Table>
  ),
);
SummaryTable.displayName = "SummaryTable";

/** The families (bold), then every key, each GPU ms descending. Memoized like the other tables. */
const SplitTable = memo(({ split }: { split: SplitSummary }) => (
  <Table>
    <TableHeader>
      <TableRow>
        <TableHead>Family</TableHead>
        <TableHead>Pipeline key</TableHead>
        <TableHead className="text-right">GPU ms/token</TableHead>
        <TableHead className="text-right">Share</TableHead>
        <TableHead className="text-right">Dispatches/token</TableHead>
      </TableRow>
    </TableHeader>
    <TableBody>
      {split.families.map((row) => (
        <TableRow key={`family-${row.family}`} className="font-medium">
          <TableCell className="text-xs">{OP_FAMILY_LABELS[row.family]}</TableCell>
          <TableCell className="text-xs text-muted-foreground">(all keys)</TableCell>
          <TableCell className="text-right font-mono text-xs">{fixed(row.gpuMs, 3)}</TableCell>
          <TableCell className="text-right font-mono text-xs">
            {shareText(row.gpuMs, split.totalGpuMs)}
          </TableCell>
          <TableCell className="text-right font-mono text-xs">{fixed(row.dispatches, 1)}</TableCell>
        </TableRow>
      ))}
      {split.keys.map((row) => (
        <TableRow key={`key-${row.key}`}>
          <TableCell className="text-xs">{OP_FAMILY_LABELS[row.family]}</TableCell>
          <TableCell className="font-mono text-xs break-all whitespace-normal">{row.key}</TableCell>
          <TableCell className="text-right font-mono text-xs">{fixed(row.gpuMs, 3)}</TableCell>
          <TableCell className="text-right font-mono text-xs">
            {shareText(row.gpuMs, split.totalGpuMs)}
          </TableCell>
          <TableCell className="text-right font-mono text-xs">{fixed(row.dispatches, 1)}</TableCell>
        </TableRow>
      ))}
    </TableBody>
  </Table>
));
SplitTable.displayName = "SplitTable";

const ChatTable = memo(
  ({ chats, arms }: { chats: readonly ChatRecord[]; arms: readonly RunArm[] }) => (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead className="text-right">Visit</TableHead>
          <TableHead>Arm</TableHead>
          <TableHead>Prompt · chat</TableHead>
          <TableHead className="text-right">Prompt ids (distinct)</TableHead>
          <TableHead>Stop</TableHead>
          <TableHead className="text-right">Decode tok/s</TableHead>
          <TableHead className="text-right">TTFT ms</TableHead>
          <TableHead className="text-right">Prefill tok/s</TableHead>
          <TableHead className="text-right">GPU ms/token (median)</TableHead>
          <TableHead className="text-right">Wall − GPU ms</TableHead>
          <TableHead className="text-right">Dispatches / token</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {chats.map((chat) => {
          const { measurement } = chat;
          const perToken = measurement.tap?.perToken;
          return (
            <TableRow
              key={`${chat.visit}-${chat.prompt}-${chat.chat}`}
              className={chat.cold ? "text-muted-foreground" : undefined}
            >
              <TableCell className="text-right font-mono text-xs">{chat.visit + 1}</TableCell>
              <TableCell className="text-xs">
                {chat.arm + 1} · {arms[chat.arm] === undefined ? "" : armLabel(arms[chat.arm])}
              </TableCell>
              <TableCell className="text-xs">
                {benchPrompt(chat.prompt).label} · {chat.chat + 1}
                {chat.cold ? " (cold)" : ""}
              </TableCell>
              <TableCell className="text-right font-mono text-xs">
                {chat.promptTokens} ({chat.distinctPromptIds}
                {chat.meetsDistinctIdCondition ? "" : " — off its side of the threshold"})
              </TableCell>
              <TableCell className="text-xs">
                {chat.stop.reason} · {chat.stop.tokens}
              </TableCell>
              <TableCell className="text-right font-mono text-xs">
                {fixed(measurement.decodeTokensPerSecond, 1)}
              </TableCell>
              <TableCell className="text-right font-mono text-xs">
                {fixed(measurement.ttftMs, 1)}
              </TableCell>
              <TableCell className="text-right font-mono text-xs">
                {fixed(measurement.prefillTokensPerSecond, 0)}
              </TableCell>
              <TableCell className="text-right font-mono text-xs">
                {fixed(perToken?.gpuBusyMs, 3)} ({fixed(perToken?.gpuBusyMsMedian, 3)})
              </TableCell>
              <TableCell className="text-right font-mono text-xs">
                {fixed(perToken?.wallMinusGpuMs, 3)}
              </TableCell>
              <TableCell className="text-right font-mono text-xs">
                {fixed(perToken?.dispatches, 1)}
              </TableCell>
            </TableRow>
          );
        })}
      </TableBody>
    </Table>
  ),
);
ChatTable.displayName = "ChatTable";
