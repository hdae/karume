/**
 * The Gemma 4 chat: load regular Gemma 4 or its q4_0 QAT line (`gemma.Pipeline`), or a mobile QAT
 * model (`gemma4QatMobile.Pipeline`), on the page's GPU, hold one multi-turn conversation in a
 * `ChatSession` (the KV is carried between turns that close with EOS), stream each reply, and
 * save a record of the chat: settings, per-turn statistics and generated token ids, and the Session
 * diagnostics when the pipeline was loaded with them. A chat can turn on thinking and declare sample
 * tools; a tool call the model makes is run in the browser and the turn continues with its result.
 */

import {
  ChatSession,
  type ChatReply,
  type ChatSessionStream,
  type ChatStop,
  type ChatTemplateOptions,
  type ChatContentPart,
  chatImagePrompt,
  type ChatTurnOptions,
  chatPrompt,
  type Rgb8Image,
} from "@karume/models/gemma";
import type { AcquireGpuOptions } from "@karume/runtime";
import { SendIcon, SquareIcon } from "lucide-react";
import { type KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";

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
import {
  InputGroup,
  InputGroupAddon,
  InputGroupButton,
  InputGroupTextarea,
} from "@/components/ui/input-group";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { useDistributionIndex } from "@/hooks/use-distribution-index";
import { useGpu } from "@/hooks/use-gpu";
import { type ManifestState, useManifest } from "@/hooks/use-manifest";
import { usePipelineSource } from "@/hooks/use-pipeline-source";
import { useRunTask } from "@/hooks/use-run-task";
import { type DistributionStatus, mountStatus } from "@/lib/distributions";
import { fileStamp } from "@/lib/download";
import { recordEnvironment } from "@/lib/environment";
import { manifestPresets } from "@/lib/presets";
import { foldComponentDiagnostics, sourceRecord } from "@/lib/record";

import {
  buildGemmaRecord,
  type DiagnosticsRecord,
  type GemmaRecord,
  NO_RUNS,
  type RunCounts,
  stopRecordOf,
  type ToolResultRecord,
  type TurnRecord,
} from "./chat-record";
import { ChatTranscript, ImageThumbnail } from "./chat-transcript";
import { decodeImageFile } from "./image-input";
import { createLiveReplyStore, frameScheduler } from "./live-reply";
import { type GenerateObservation, type ObservedHost, observeHost } from "./observed-host";
import {
  SAMPLE_TOOLS,
  type SampleTool,
  type SampleToolName,
  sampleToolsOf,
  toolResultsOf,
} from "./sample-tools";
import { SAMPLE_MESSAGES } from "./samples";
import {
  type ChatForm,
  type ChatSettings,
  effectiveModelKey,
  GEMMA_MOUNTS,
  type GemmaManifests,
  type GemmaMount,
  gemmaModelChoices,
  chatSpeculative,
  hasDrafter,
  hasVision,
  INITIAL_CHAT_FORM,
  MOUNT_LABELS,
  type ModelKey,
  modelConfig,
  modelKeyValue,
  OVERFLOW_CHOICES,
  overflowPolicy,
  parseChatForm,
  parseModelKey,
  parseTurnForm,
  sameChatSettings,
  streamsStructure,
  type TurnForm,
  turnFormOf,
} from "./settings";
import {
  awaitingToolResults,
  type ContextUsage,
  contextUsageText,
  deriveTranscript,
  historyOf,
  notKeptNotice,
  type SentTurn,
  type TranscriptEntry,
  turnTiming,
} from "./turns";
import {
  type DiagnosticsSink,
  type GemmaPipelineState,
  type LoadRequest,
  useGemmaPipeline,
} from "./use-gemma-pipeline";

/** Module constant (useGpu needs stable options): subgroups when the adapter has them. */
const GPU_OPTIONS: Omit<AcquireGpuOptions, "onDeviceLost"> = { subgroups: "available" };

/**
 * How often the running turn updates the status card (its counter). The streamed text has its own
 * per-frame path (`LiveReplyStore`); this bounds the page re-renders a turn causes.
 */
const REPORT_INTERVAL_MS = 250;

/**
 * Tool rounds (run the calls, continue with the results) one message may take before the page stops
 * answering: a model that keeps calling would otherwise never give the turn back.
 */
const MAX_TOOL_ROUNDS = 8;

type ReadyPipeline = Extract<GemmaPipelineState, { kind: "ready" }>;

/** What **Load** / **New chat** ask the session to be built with (a new object per press). */
type ChatRequest = { readonly settings: ChatSettings };

/** The conversation of the loaded pipeline (one `ChatSession`). */
type ChatState =
  | { readonly kind: "none" }
  | {
      readonly kind: "ready";
      readonly session: ChatSession;
      readonly observed: ObservedHost;
      readonly settings: ChatSettings;
      /** The sample tools the session declares (their runners answer its calls). */
      readonly tools: readonly SampleTool[];
      /** The session's template options (every render of its history uses them). */
      readonly template: ChatTemplateOptions;
      /** The pipeline state and the request the session was built for (stale once either changes). */
      readonly pipeline: ReadyPipeline;
      readonly request: ChatRequest;
    }
  | { readonly kind: "error"; readonly error: unknown };

/** Where the running turn collects what the pipeline and the session report. */
type TurnSink = {
  readonly onGenerate: (observation: GenerateObservation) => void;
  readonly onRun: (phase: { readonly kind: keyof RunCounts }) => void;
  readonly onOverflow: (dropped: number) => void;
};

/** What the running turn collects (mutated by the callbacks while it runs). */
type TurnCollected = {
  promptTokens?: number;
  carriedPositions?: number;
  prefillAt: number[];
  chunks: number;
  firstTokenAt?: number;
  lastRunAt?: number;
  tokenIds: number[];
  runs: RunCounts;
  dropped: number;
};

/** One generation of a message: the message itself, or a continuation with tool results. */
type TurnStep =
  | { readonly kind: "message"; readonly issue: (options: ChatTurnOptions) => ChatSessionStream }
  | {
      readonly kind: "tool-results";
      readonly results: readonly ToolResultRecord[];
      readonly issue: (options: ChatTurnOptions) => ChatSessionStream;
    };

/** How a step settled: the parsed reply (when it was delivered) and the failure to show, if any. */
type StepOutcome = {
  readonly reply: ChatReply | undefined;
  readonly failure: { readonly error: unknown } | undefined;
};

const errorText = (error: unknown): string =>
  error instanceof Error ? `${error.name}: ${error.message}` : String(error);

const manifestOf = (state: ManifestState) => (state.kind === "ready" ? state.manifest : undefined);

const configOf = (manifests: GemmaManifests, key: ModelKey | undefined) => {
  const manifest = key === undefined ? undefined : manifests[key.mount];
  if (manifest === undefined || key === undefined) return undefined;
  try {
    return { ok: true as const, config: modelConfig(manifest, key.model) };
  } catch (error) {
    return { ok: false as const, error };
  }
};

const settingsText = (settings: ChatSettings): string =>
  `system prompt: ${settings.system === undefined ? "none" : "set"} · capacity ${settings.capacity} · overflow: ${settings.overflow}${
    settings.enableThinking === true ? " · thinking" : ""
  }${settings.tools === undefined ? "" : ` · tools: ${settings.tools.join(", ")}`}`;

const pipelineText = (state: GemmaPipelineState): string => {
  switch (state.kind) {
    case "idle":
      return "not loaded — press Load";
    case "loading":
      return `loading: ${state.detail}`;
    case "error":
      return "failed (below)";
    case "ready": {
      const { request } = state;
      return `${MOUNT_LABELS[request.mount]} ${request.model} · ${request.quant} · ${request.source.kind}${
        request.speculative ? " · MTP drafter" : ""
      }${request.fullStateStorage === "f16" ? " · f16 KV" : ""}${request.diagnostics ? " · diagnostics" : ""} · loaded in ${state.loadSeconds.toFixed(1)} s`;
    }
  }
};

export const Gemma4Page = () => {
  const index = useDistributionIndex();
  const distributions = index.kind === "ready" ? index.index : undefined;
  const statuses = useMemo(
    (): Readonly<Record<GemmaMount, DistributionStatus | undefined>> => ({
      gemma4: mountStatus(distributions, "gemma4"),
      "gemma4-qat": mountStatus(distributions, "gemma4-qat"),
      "gemma4-qat-mobile": mountStatus(distributions, "gemma4-qat-mobile"),
    }),
    [distributions],
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
  const missing = GEMMA_MOUNTS.flatMap((mount) => {
    const state = manifestStates[mount];
    return state.kind === "missing" ? [state.directory] : [];
  });

  // The selection: model (across the mounts), preset, source, MTP drafter, diagnostics. The derived
  // values are memoized: the page re-renders while a turn runs (the status ticker), and that host
  // work would land inside the decode window the page measures.
  const [chosenModel, setChosenModel] = useState<ModelKey | undefined>(undefined);
  const modelKey = useMemo(
    () => effectiveModelKey(chosenModel, manifests),
    [chosenModel, manifests],
  );
  const selectedManifest = modelKey === undefined ? undefined : manifests[modelKey.mount];
  const presets = useMemo(
    () =>
      selectedManifest === undefined || modelKey === undefined
        ? undefined
        : manifestPresets(selectedManifest, modelKey.model),
    [selectedManifest, modelKey],
  );
  const [chosenQuant, setChosenQuant] = useState<string | undefined>(undefined);
  const quant =
    presets === undefined
      ? undefined
      : presets.presets.some(({ name }) => name === chosenQuant) && chosenQuant !== undefined
        ? chosenQuant
        : presets.defaultQuant;
  const config = useMemo(() => configOf(manifests, modelKey), [manifests, modelKey]);
  const sourceChoice = usePipelineSource(statuses[modelKey?.mount ?? "gemma4"]);
  const [speculativeChosen, setSpeculativeChosen] = useState(false);
  const speculativeAvailable =
    selectedManifest !== undefined &&
    modelKey !== undefined &&
    hasDrafter(selectedManifest, modelKey);
  const speculative = speculativeChosen && speculativeAvailable;
  const [f16KvChosen, setF16KvChosen] = useState(false);
  const [diagnosticsChosen, setDiagnosticsChosen] = useState(false);

  // GPU and pipeline.
  const gpu = useGpu(GPU_OPTIONS);
  const readyGpu = gpu.state.kind === "ready" ? gpu.state.gpu : undefined;
  const diagnosticsSink = useRef<DiagnosticsSink | undefined>(undefined);
  const [loadRequest, setLoadRequest] = useState<LoadRequest | undefined>(undefined);
  const { state: pipeline, releaseError: pipelineReleaseError } = useGemmaPipeline({
    gpu: readyGpu,
    request: loadRequest,
    diagnostics: diagnosticsSink,
    track: gpu.track,
  });

  // The chat settings (applied when a session is built: Load and New chat).
  const [chatForm, setChatForm] = useState<ChatForm>(INITIAL_CHAT_FORM);
  // Load checks the form against the selected model's declaration; New chat against the loaded
  // pipeline (its program carries the same three numbers the session checks).
  const chatParsed = useMemo(
    () => (config?.ok === true ? parseChatForm(chatForm, config.config) : undefined),
    [chatForm, config],
  );
  const chatParsedLoaded = useMemo(
    () =>
      pipeline.kind === "ready" ? parseChatForm(chatForm, pipeline.pipeline.program) : undefined,
    [chatForm, pipeline],
  );
  const chatErrors = [
    ...new Set([
      ...(chatParsed?.ok === false ? chatParsed.errors : []),
      ...(chatParsedLoaded?.ok === false ? chatParsedLoaded.errors : []),
    ]),
  ];
  const toggleTool = (name: SampleToolName, checked: boolean): void => {
    const others = (chatForm.tools ?? []).filter((tool) => tool !== name);
    setChatForm({ ...chatForm, tools: checked ? [...others, name] : others });
  };
  const [chatRequest, setChatRequest] = useState<ChatRequest | undefined>(undefined);
  const [chat, setChat] = useState<ChatState>({ kind: "none" });
  const turnSink = useRef<TurnSink | undefined>(undefined);

  // The per-turn form, filled from the distribution's sampler until edited.
  const [turnFormEdit, setTurnFormEdit] = useState<TurnForm | undefined>(undefined);
  const distributionSampler = config?.ok === true ? config.config.sampler : undefined;
  const turnForm = useMemo(
    () => turnFormEdit ?? turnFormOf(distributionSampler),
    [turnFormEdit, distributionSampler],
  );
  const turnParsed = useMemo(() => parseTurnForm(turnForm), [turnForm]);
  const updateTurn = (patch: Partial<TurnForm>) => setTurnFormEdit({ ...turnForm, ...patch });

  // The conversation shown, and what the record keeps.
  const task = useRunTask<void>(gpu);
  const [input, setInput] = useState("");
  const [entries, setEntries] = useState<readonly TranscriptEntry[]>([]);
  const [turnRecords, setTurnRecords] = useState<readonly TurnRecord[]>([]);
  const [diagnostics, setDiagnostics] = useState<DiagnosticsRecord>({});
  /** Set after a turn the session did not keep (its message left no entry in the history). */
  const [notice, setNotice] = useState<string | undefined>(undefined);
  /** Why the page left the last tool call unanswered (the round limit, a Stop between the steps). */
  const [toolNotice, setToolNotice] = useState<string | undefined>(undefined);
  /** The message of the running turn, shown until the turn settles. */
  const [pendingText, setPendingText] = useState<string | undefined>(undefined);
  /** The images of the message being written; each attachment has an id for its Remove button. */
  const [attachments, setAttachments] = useState<
    readonly { readonly id: number; readonly image: Rgb8Image }[]
  >([]);
  const [attachError, setAttachError] = useState<unknown>(undefined);
  const nextAttachmentId = useRef(0);
  // Counts session resets: a decode that settles after one belongs to the chat it was started in.
  const attachEpoch = useRef(0);
  const [pendingImages, setPendingImages] = useState<readonly Rgb8Image[]>([]);
  const [sessionReleaseError, setSessionReleaseError] = useState<unknown>(undefined);
  const [fitError, setFitError] = useState<unknown>(undefined);
  const [live] = useState(() => createLiveReplyStore(frameScheduler));
  useEffect(() => () => live.dispose(), [live]);
  const nextEntryId = useRef(0);
  const turnCount = useRef(0);
  const diagnosticsAcc = useRef<DiagnosticsRecord>({});
  /** The session the conversation state belongs to (a turn that outlives it drops its results). */
  const liveSession = useRef<ChatSession | undefined>(undefined);

  // With diagnostics, every run of the pipeline reports here: folded per Session (the drafter's runs
  // are `draft`), and counted for the running turn.
  useEffect(() => {
    diagnosticsSink.current = (report, phase) => {
      const component = phase.kind === "draft" ? "drafter" : "target";
      diagnosticsAcc.current = {
        ...diagnosticsAcc.current,
        [component]: foldComponentDiagnostics(diagnosticsAcc.current[component], report),
      };
      turnSink.current?.onRun(phase);
    };
    return () => {
      diagnosticsSink.current = undefined;
    };
  }, []);

  // One session per (loaded pipeline, chat request). The session is disposed before a new one is
  // built and when the pipeline goes; the pipeline's own dispose also closes the sequence it handed
  // out, so the order of the two cleanups does not matter. The conversation shown belongs to the
  // session, so it is cleared whenever the session goes (Load, New chat, a GPU re-acquired after a
  // device loss rebuilds the pipeline).
  useEffect(() => {
    setEntries([]);
    setTurnRecords([]);
    attachEpoch.current += 1;
    setAttachments([]);
    setPendingImages([]);
    setDiagnostics({});
    setNotice(undefined);
    setToolNotice(undefined);
    setPendingText(undefined);
    live.reset();
    diagnosticsAcc.current = {};
    turnCount.current = 0;
    liveSession.current = undefined;
    if (
      pipeline.kind !== "ready" ||
      pipeline.request !== loadRequest ||
      chatRequest === undefined
    ) {
      setChat({ kind: "none" });
      return undefined;
    }
    const { settings } = chatRequest;
    const observed = observeHost(pipeline.pipeline, (observation) =>
      turnSink.current?.onGenerate(observation),
    );
    const tools = sampleToolsOf(settings.tools ?? []);
    const template: ChatTemplateOptions = {
      ...(tools.length === 0 ? {} : { tools: tools.map(({ declaration }) => declaration) }),
      ...(settings.enableThinking === true ? { enableThinking: true } : {}),
    };
    let session: ChatSession;
    try {
      session = new ChatSession(observed.host, {
        ...(settings.system === undefined ? {} : { system: settings.system }),
        ...chatSpeculative(pipeline.request),
        // No session maxNewTokens: every turn passes its own limit when the form has one, and a
        // turn without one is checked against the session's default reserve.
        capacity: settings.capacity,
        onOverflow: overflowPolicy(settings.overflow, (dropped) =>
          turnSink.current?.onOverflow(dropped),
        ),
        ...template,
      });
    } catch (error) {
      setChat({ kind: "error", error });
      return undefined;
    }
    liveSession.current = session;
    setChat({
      kind: "ready",
      session,
      observed,
      settings,
      tools,
      template,
      pipeline,
      request: chatRequest,
    });
    return () => {
      liveSession.current = undefined;
      gpu.track(session.dispose()).catch((error: unknown) => {
        console.error("Closing the Gemma 4 chat session failed", error);
        setSessionReleaseError(error);
      });
    };
  }, [pipeline, loadRequest, chatRequest, live, gpu.track]);

  const running = task.running;
  const canLoad =
    readyGpu !== undefined &&
    modelKey !== undefined &&
    quant !== undefined &&
    chatParsed?.ok === true &&
    !running;
  const load = (): void => {
    if (!canLoad || chatParsed?.ok !== true) return;
    setSessionReleaseError(undefined);
    setLoadRequest({
      mount: modelKey.mount,
      model: modelKey.model,
      quant,
      source: sourceChoice.source,
      speculative,
      ...(f16KvChosen ? { fullStateStorage: "f16" as const } : {}),
      ...(selectedManifest !== undefined && hasVision(selectedManifest, modelKey)
        ? { vision: true }
        : {}),
      diagnostics: diagnosticsChosen,
    });
    setChatRequest({ settings: chatParsed.value });
  };
  // Dropping the request aborts the fetch (the hook's cleanup); see the note next to the button.
  const cancelLoad = (): void => setLoadRequest(undefined);
  const canStartNewChat = chatParsedLoaded?.ok === true && !running;
  const newChat = (): void => {
    if (!canStartNewChat || chatParsedLoaded?.ok !== true) return;
    setSessionReleaseError(undefined);
    setChatRequest({ settings: chatParsedLoaded.value });
  };

  // The chat is usable only while it belongs to the current pipeline and chat request: in the render
  // right after Load / New chat (before the effects replace it) the old session is already stale.
  const readyChat =
    chat.kind === "ready" &&
    chat.pipeline === pipeline &&
    pipeline.request === loadRequest &&
    chat.request === chatRequest
      ? chat
      : undefined;
  // Derived from the session's history (the source of truth): the last turn ended on a call the page
  // did not answer, so the session takes no message until a new chat.
  const awaitingTools = useMemo(
    () => readyChat !== undefined && !running && awaitingToolResults(readyChat.session.turns),
    // turnRecords: every settled step changes the history.
    [readyChat, running, turnRecords],
  );
  // Read from the loaded pipeline: the control exists exactly when an image can be sent.
  const visionConfig = readyChat?.pipeline.pipeline.vision;
  const canSend =
    readyChat !== undefined &&
    turnParsed.ok &&
    (input.trim() !== "" || attachments.length > 0) &&
    !running &&
    !awaitingTools;
  const attachFiles = (files: FileList | null): void => {
    if (files === null) return;
    setAttachError(undefined);
    const epoch = attachEpoch.current;
    void Promise.all([...files].map(decodeImageFile)).then(
      (images) => {
        if (epoch !== attachEpoch.current) return;
        setAttachments((current) => [
          ...current,
          ...images.map((image) => ({ id: (nextAttachmentId.current += 1), image })),
        ]);
      },
      (error: unknown) => {
        if (epoch === attachEpoch.current) setAttachError(error);
      },
    );
  };

  const send = (): void => {
    if (!canSend || !turnParsed.ok) return;
    const { session, observed, settings, tools } = readyChat;
    const diagnosticsOn = readyChat.pipeline.request.diagnostics;
    const { maxNewTokens, sampler } = turnParsed.value;
    const text = input;
    const images = attachments.map(({ image }) => image);
    const message: string | readonly ChatContentPart[] =
      images.length === 0
        ? text
        : [
            ...images.map((image): ChatContentPart => ({ type: "image", image })),
            ...(text === "" ? [] : [{ type: "text", text } as const]),
          ];
    const imageRecords =
      images.length === 0 || visionConfig === undefined
        ? []
        : chatImagePrompt(
            readyChat.pipeline.pipeline.tokenizer,
            [{ role: "user", content: images.map((image) => ({ type: "image", image })) }],
            visionConfig,
          ).images.map(({ image, softTokens }) => ({
            width: image.width,
            height: image.height,
            softTokens,
          }));
    const structured = streamsStructure(settings);
    // Only this message's steps change the transcript, and a new session (which clears it) makes the
    // turn drop its results, so each step settles against the transcript the previous step left
    // (the first against the one as of the send).
    let previous = entries;
    void task.start(async ({ signal, report }) => {
      /**
       * One generation (the message, or a continuation with tool results): streamed through `live`,
       * then settled into its own record row and the transcript. `undefined` = the session went
       * away during it (nothing is added to the new one).
       */
      const runStep = async (step: TurnStep, round: number): Promise<StepOutcome | undefined> => {
        const collected: TurnCollected = {
          prefillAt: [],
          chunks: 0,
          tokenIds: [],
          runs: NO_RUNS,
          dropped: 0,
        };
        turnSink.current = {
          onGenerate: ({ promptTokens, usedBefore }) => {
            collected.promptTokens = promptTokens;
            collected.carriedPositions = usedBefore;
          },
          onRun: (phase) => {
            collected.runs = { ...collected.runs, [phase.kind]: collected.runs[phase.kind] + 1 };
            if (phase.kind !== "prefill") collected.lastRunAt = performance.now();
          },
          onOverflow: (dropped) => {
            collected.dropped += dropped;
          },
        };
        // The status card's counter is throttled; the streamed text goes through `live` (per frame).
        let lastReportAt = Number.NEGATIVE_INFINITY;
        let reply = "";
        let stage: string | undefined;
        const note =
          round === 0 ? {} : { note: `continuing after tool round ${round} / ${MAX_TOOL_ROUNDS}` };
        const show = (): void =>
          live.update({ text: reply, ...(stage === undefined ? {} : { stage }) });
        const sentAt = performance.now();
        // `send` / `sendToolResults` check the request synchronously (a throw here changes nothing
        // in the session).
        const stream = step.issue({
          ...(maxNewTokens === undefined ? {} : { maxNewTokens }),
          sampler,
          signal,
          onPrefill: ({ chunk, chunks }) => {
            collected.prefillAt.push(performance.now());
            collected.chunks = chunk;
            stage = `prefill ${chunk} / ${chunks} chunks`;
            show();
            report({
              stage: "prefill",
              count: { done: chunk, total: chunks, unit: "chunks" },
              ...note,
            });
          },
          onToken: (id) => {
            const now = performance.now();
            collected.firstTokenAt ??= now;
            collected.tokenIds.push(id);
            const done = collected.tokenIds.length;
            stage =
              maxNewTokens === undefined
                ? `decode ${done} tokens`
                : `decode ${done} / ${maxNewTokens} tokens`;
            show();
            if (now - lastReportAt >= REPORT_INTERVAL_MS) {
              lastReportAt = now;
              // Without a limit there is no total for the card's bar: the stage carries the count.
              report(
                maxNewTokens === undefined
                  ? { stage, ...note }
                  : {
                      stage: "decode",
                      count: { done, total: maxNewTokens, unit: "tokens" },
                      ...note,
                    },
              );
            }
          },
        });
        const turn = turnCount.current;
        turnCount.current += 1;
        if (step.kind === "message") {
          setInput("");
          setAttachments([]);
          setNotice(undefined);
          setToolNotice(undefined);
          setPendingText(text);
          setPendingImages(images);
        }
        live.reset();

        let failure: { readonly error: unknown } | undefined;
        try {
          for await (const piece of stream) {
            reply += piece;
            show();
          }
        } catch (error) {
          failure = { error };
        }
        const endAt = performance.now();
        // After Stop, `done` resolves `aborted` (the iteration threw the signal's reason); after a
        // failure it rejects with the iteration's error.
        let stop: ChatStop | undefined;
        try {
          stop = await stream.done;
        } catch (error) {
          failure ??= { error };
        }
        // The parsed reply settles right after `done` (both after the stream drained). It rejects
        // for a malformed closed turn (`ModelInputError`): the step then failed, though it stopped.
        let parsed: ChatReply | undefined;
        let replyFailure: { readonly error: unknown } | undefined;
        if (stop !== undefined && failure === undefined) {
          try {
            parsed = await stream.reply;
          } catch (error) {
            replyFailure = { error };
          }
        }

        // The session went away during the turn (a device loss rebuilt the pipeline): its
        // conversation is gone, so nothing is added to the new one.
        if (liveSession.current !== session) {
          if (stop === undefined && failure !== undefined) throw failure.error;
          return undefined;
        }
        const timing = turnTiming(
          {
            sentAt,
            prefillAt: collected.prefillAt,
            ...(collected.firstTokenAt === undefined
              ? {}
              : { firstTokenAt: collected.firstTokenAt }),
            ...(collected.lastRunAt === undefined ? {} : { lastRunAt: collected.lastRunAt }),
            endAt,
          },
          {
            ...(collected.promptTokens === undefined
              ? {}
              : { promptTokens: collected.promptTokens }),
            tokens: stop?.tokens ?? collected.tokenIds.length,
          },
        );
        const kvUsedAfter = observed.kvUsed();
        // The transcript is rebuilt from the session's history (the source of truth).
        const sent: SentTurn =
          step.kind === "message"
            ? {
                turn,
                text,
                // With tools or thinking the stream keeps the structure spellings, so the replies
                // line up by position instead of by the streamed text.
                reply: structured ? undefined : reply,
                userId: (nextEntryId.current += 1),
                assistantId: (nextEntryId.current += 1),
              }
            : { turn, toolResults: step.results.map(({ content }) => content) };
        const derived = deriveTranscript({
          previous,
          history: historyOf(session.turns),
          sent,
          dropped: collected.dropped,
          newId: () => (nextEntryId.current += 1),
        });
        previous = derived.entries;
        const failed = stop === undefined ? failure : replyFailure;
        const record: TurnRecord = {
          index: turn + 1,
          userText: text,
          ...(step.kind === "message" && imageRecords.length > 0 ? { images: imageRecords } : {}),
          ...(step.kind === "tool-results" ? { toolResults: step.results } : {}),
          ...(parsed?.end === "tool-call"
            ? {
                toolCalls: parsed.message.tool_calls.map((call) => ({
                  ...(call.id === undefined ? {} : { id: call.id }),
                  name: call.function.name,
                  arguments: call.function.arguments,
                })),
              }
            : {}),
          replyText: reply,
          request: { ...(maxNewTokens === undefined ? {} : { maxNewTokens }), sampler },
          outcome:
            stop !== undefined && failed === undefined
              ? { kind: "stopped", stop: stopRecordOf(stop) }
              : { kind: "failed", error: errorText(failed?.error) },
          prefill: {
            ...(collected.promptTokens === undefined
              ? {}
              : { promptTokens: collected.promptTokens }),
            ...(collected.carriedPositions === undefined
              ? {}
              : { carriedPositions: collected.carriedPositions }),
            chunks: collected.chunks,
          },
          timing,
          generatedTokenIds: collected.tokenIds,
          droppedEntries: collected.dropped,
          inHistory: derived.kept,
          ...(kvUsedAfter === undefined ? {} : { kvUsedAfter }),
          ...(diagnosticsOn ? { runs: collected.runs } : {}),
        };
        setTurnRecords((current) => [...current, record]);
        setDiagnostics(diagnosticsAcc.current);
        setEntries(derived.entries);
        setPendingText(undefined);
        setPendingImages([]);
        live.reset();
        if (!derived.kept) {
          if (step.kind === "message") {
            setNotice(notKeptNotice(turn + 1, stop));
            // The message is not in the history; put it back so it can be sent again.
            setInput((current) => (current === "" ? text : current));
            setAttachments((current) =>
              current.length > 0
                ? current
                : images.map((image) => ({ id: (nextAttachmentId.current += 1), image })),
            );
          } else {
            setToolNotice(
              "The continuation after the tool results ended before any text, so the session took the results back.",
            );
          }
        }
        // A Stop is an ordinary outcome (the partial reply stays); anything else is shown in full.
        return { reply: parsed, failure: failed };
      };

      try {
        let step: TurnStep = {
          kind: "message",
          issue: (options) => session.send(message, options),
        };
        for (let round = 0; ; round += 1) {
          const outcome = await runStep(step, round);
          if (outcome === undefined) return;
          if (outcome.failure !== undefined) throw outcome.failure.error;
          const reply = outcome.reply;
          if (reply?.end !== "tool-call") return;
          if (round >= MAX_TOOL_ROUNDS) {
            setToolNotice(
              `The model called tools for ${MAX_TOOL_ROUNDS} rounds in a row on this message, so the page stopped running them.`,
            );
            return;
          }
          // Stop pressed as the step ended: the calls stay unanswered rather than starting a
          // continuation that would only be aborted.
          if (signal.aborted) {
            setToolNotice("Stopped before the tool results were sent.");
            return;
          }
          const calls = reply.message.tool_calls;
          const results = toolResultsOf(tools, calls);
          step = {
            kind: "tool-results",
            results: results.map((result) => ({
              ...(result.tool_call_id === undefined ? {} : { toolCallId: result.tool_call_id }),
              name: result.name,
              content: result.content,
            })),
            issue: (options) => session.sendToolResults(results, options),
          };
        }
      } finally {
        turnSink.current = undefined;
      }
    });
  };

  const onInputKey = (event: KeyboardEvent<HTMLTextAreaElement>): void => {
    if (event.key !== "Enter" || event.shiftKey || event.nativeEvent.isComposing) return;
    event.preventDefault();
    send();
  };

  const contextUsage = useMemo((): ContextUsage | undefined => {
    if (readyChat === undefined || running) return undefined;
    const capacity = readyChat.settings.capacity;
    const used = readyChat.observed.kvUsed();
    if (used !== undefined) return { kind: "kv", used, capacity };
    const turns = readyChat.session.turns;
    // A history that ends on an unanswered call has no generation prompt to render (`chatPrompt`
    // refuses it); its size is known again once a new chat starts.
    if (awaitingToolResults(turns)) return undefined;
    const { tokenizer, vision } = readyChat.pipeline.pipeline;
    const tokens =
      turns.length === 0
        ? 0
        : vision === undefined
          ? chatPrompt(tokenizer, turns, readyChat.template).length
          : chatImagePrompt(tokenizer, turns, vision, readyChat.template).prompt.length;
    return { kind: "history", tokens, capacity };
    // turnRecords: a finished turn changes the KV or the history.
  }, [readyChat, running, turnRecords]);

  const record = useMemo((): GemmaRecord | undefined => {
    if (readyChat === undefined || readyGpu === undefined || turnRecords.length === 0) {
      return undefined;
    }
    const { request, loadSeconds } = readyChat.pipeline;
    const manifestState = manifestStates[request.mount];
    if (manifestState.kind !== "ready") return undefined;
    const preset = manifestPresets(manifestState.manifest, request.model).presets.find(
      ({ name }) => name === request.quant,
    );
    const status = statuses[request.mount];
    return buildGemmaRecord({
      environment: recordEnvironment(readyGpu),
      mount: request.mount,
      directory: status?.directory ?? "",
      manifestSha256: manifestState.sha256,
      source: sourceRecord(request.source),
      model: request.model,
      preset: {
        quant: request.quant,
        label: preset?.label ?? request.quant,
        isDefault: request.quant === manifestState.manifest.models[request.model]?.defaultQuant,
      },
      speculative: request.speculative,
      ...(request.fullStateStorage === undefined
        ? {}
        : { fullStateStorage: request.fullStateStorage }),
      ...(request.vision === true ? { vision: true as const } : {}),
      loadSeconds: Number(loadSeconds.toFixed(3)),
      chat: readyChat.settings,
      turns: turnRecords,
      diagnostics: request.diagnostics ? diagnostics : undefined,
    });
  }, [readyChat, readyGpu, turnRecords, diagnostics, manifestStates, statuses]);

  const chatSettingsChanged =
    readyChat !== undefined &&
    chatParsedLoaded?.ok === true &&
    !sameChatSettings(readyChat.settings, chatParsedLoaded.value);
  const selectionChanged =
    pipeline.kind === "ready" &&
    (modelKey?.mount !== pipeline.request.mount ||
      modelKey.model !== pipeline.request.model ||
      quant !== pipeline.request.quant ||
      sourceChoice.source !== pipeline.request.source ||
      speculative !== pipeline.request.speculative ||
      f16KvChosen !== (pipeline.request.fullStateStorage === "f16") ||
      diagnosticsChosen !== pipeline.request.diagnostics);
  const modelChoices = useMemo(() => gemmaModelChoices(manifests), [manifests]);

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-2">
        <h1 className="text-2xl font-semibold">Gemma 4 chat</h1>
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
            ))}{" "}
            (a directory with a <code>karume.json</code>). Put a distribution there and reload.
          </AlertDescription>
        </Alert>
      ) : null}
      {config?.ok === false ? <ErrorAlert title="pipelineConfig" error={config.error} /> : null}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,22rem)_minmax(0,1fr)]">
        <div className="flex flex-col gap-6">
          <Card>
            <CardHeader>
              <CardTitle>Model</CardTitle>
              <CardDescription>
                Load reads the weights and builds the Session; changing these takes effect on the
                next Load.
              </CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-4">
              <Field
                id="gemma4-model"
                label="Model"
                hint={
                  modelKey?.mount === "gemma4-qat-mobile"
                    ? "Mobile QAT (gemma4QatMobile.Pipeline): experimental, fixed int2/int4/int8 with SRQ, no MTP."
                    : modelKey?.mount === "gemma4-qat"
                      ? "q4_0 QAT line (gemma.Pipeline): Google's QAT checkpoints on their own int4 grid, with QAT MTP drafters."
                      : "Regular Gemma 4 (gemma.Pipeline)."
                }
              >
                <ChoiceSelect
                  id="gemma4-model"
                  value={modelKey === undefined ? "" : modelKeyValue(modelKey)}
                  choices={modelChoices}
                  onChange={(value) => setChosenModel(parseModelKey(value))}
                  disabled={running || modelChoices.length === 0}
                />
              </Field>
              {presets === undefined || quant === undefined ? null : (
                <PresetSelect
                  id="gemma4-preset"
                  value={quant}
                  presets={presets.presets}
                  defaultQuant={presets.defaultQuant}
                  onChange={setChosenQuant}
                  disabled={running}
                />
              )}
              <SourcePicker
                id="gemma4-source"
                value={sourceChoice.source.kind}
                cachedAvailable={sourceChoice.hfRef !== undefined}
                onChange={sourceChoice.choose}
                disabled={running}
              />
              <div className="flex items-center gap-2">
                <Switch
                  id="gemma4-speculative"
                  checked={speculative}
                  onCheckedChange={(checked) => setSpeculativeChosen(checked)}
                  disabled={running || !speculativeAvailable}
                />
                <Label htmlFor="gemma4-speculative">MTP drafter (speculative decoding)</Label>
              </div>
              <p className="-mt-2 text-xs text-muted-foreground">
                {speculativeAvailable
                  ? "Also reads the drafter weights; the session then speculates with the self-tuning gate. It is designed to emit the same tokens as plain decoding."
                  : "Not available for this model (no drafter weights, or the mobile QAT distribution)."}
              </p>
              <div className="flex items-center gap-2">
                <Switch
                  id="gemma4-f16-kv"
                  checked={f16KvChosen}
                  onCheckedChange={(checked) => setF16KvChosen(checked)}
                  disabled={running}
                />
                <Label htmlFor="gemma4-f16-kv">f16 KV (full layers)</Label>
              </div>
              <p className="-mt-2 text-xs text-muted-foreground">
                Halves the full-layer KV, and Fit counts it. It changes the numbers slightly:
                earlier runs are read rounded to f16. With the MTP drafter the session speculates
                always, without the self-tuning gate. Applies on the next Load.
              </p>
              <div className="flex items-center gap-2">
                <Switch
                  id="gemma4-diagnostics"
                  checked={diagnosticsChosen}
                  onCheckedChange={(checked) => setDiagnosticsChosen(checked)}
                  disabled={running}
                />
                <Label htmlFor="gemma4-diagnostics">Collect Session diagnostics</Label>
              </div>
              <p className="-mt-2 text-xs text-muted-foreground">
                Passes onRunDiagnostics: the record gets per-run counts and the Session diagnostics.
                It changes the decode path: without it the pipeline picks greedy (temperature 0)
                tokens on the GPU; with it every greedy token reads the whole logits row back and is
                picked on the CPU. Leave it off to measure decode rates.
              </p>
              {pipeline.kind === "loading" ? (
                <>
                  <Button id="gemma4-cancel-load" variant="outline" onClick={cancelLoad}>
                    Cancel load
                  </Button>
                  <p className="-mt-2 text-xs text-muted-foreground">
                    Cancel stops the download. Building the Session after the download cannot be
                    interrupted: a pipeline that finishes after Cancel is released at once.
                  </p>
                </>
              ) : (
                <Button id="gemma4-load" onClick={load} disabled={!canLoad}>
                  {pipeline.kind === "ready" ? "Load again" : "Load"}
                </Button>
              )}
              {selectionChanged ? (
                <p id="gemma4-selection-changed" className="text-xs text-muted-foreground">
                  The selection differs from the loaded model; press Load to switch (the chat is
                  cleared).
                </p>
              ) : null}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Chat</CardTitle>
              <CardDescription>
                A session is built with these; changes apply on New chat (or Load).
              </CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-4">
              <Field id="gemma4-system" label="System prompt" hint="Empty = no system message.">
                <Textarea
                  id="gemma4-system"
                  value={chatForm.system}
                  rows={3}
                  onChange={(event) => setChatForm({ ...chatForm, system: event.target.value })}
                  disabled={running}
                />
              </Field>
              <Field
                id="gemma4-capacity"
                label="Capacity"
                hint={
                  config?.ok === true
                    ? `KV positions this chat allocates. Empty = the distribution's ${config.config.capacity} (from ${config.config.chunkLength} to ${config.config.maxPosition}).`
                    : "KV positions this chat allocates."
                }
              >
                <div className="flex gap-2">
                  <Input
                    id="gemma4-capacity"
                    inputMode="numeric"
                    value={chatForm.capacity}
                    placeholder={config?.ok === true ? String(config.config.capacity) : ""}
                    onChange={(event) => setChatForm({ ...chatForm, capacity: event.target.value })}
                    disabled={running}
                  />
                  <Button
                    variant="outline"
                    disabled={running || pipeline.kind !== "ready"}
                    onClick={() => {
                      if (pipeline.kind !== "ready") return;
                      setFitError(undefined);
                      pipeline.pipeline.fitCapacity().then((capacity) => {
                        setChatForm((form) => ({ ...form, capacity: String(capacity) }));
                      }, setFitError);
                    }}
                  >
                    Fit
                  </Button>
                </div>
                {fitError === undefined ? null : (
                  <p className="text-xs text-destructive">
                    {fitError instanceof Error ? fitError.message : "Fit failed."}
                  </p>
                )}
              </Field>
              <Field
                id="gemma4-overflow"
                label="When a turn does not fit"
                hint="Checked before the turn is sent: the prompt plus max new tokens (1,024 reserved tokens when that is empty) must fit the capacity."
              >
                <ChoiceSelect
                  id="gemma4-overflow"
                  value={chatForm.overflow}
                  choices={OVERFLOW_CHOICES}
                  onChange={(overflow) => setChatForm({ ...chatForm, overflow })}
                  disabled={running}
                />
              </Field>
              <div className="flex flex-col gap-2">
                <Label className="flex items-start gap-2 font-normal">
                  <Checkbox
                    id="gemma4-thinking"
                    checked={chatForm.thinking === true}
                    onCheckedChange={(checked) => setChatForm({ ...chatForm, thinking: checked })}
                    disabled={running}
                  />
                  <span>
                    Thinking{" "}
                    <span className="text-muted-foreground">
                      (enableThinking: the model writes a thought before its answer, shown
                      collapsed)
                    </span>
                  </span>
                </Label>
              </div>
              <div className="flex flex-col gap-2">
                <p className="text-sm font-medium">Sample tools</p>
                {SAMPLE_TOOLS.map((tool) => (
                  <Label key={tool.name} className="flex items-start gap-2 font-normal">
                    <Checkbox
                      id={`gemma4-tool-${tool.name}`}
                      checked={chatForm.tools?.includes(tool.name) === true}
                      onCheckedChange={(checked) => toggleTool(tool.name, checked)}
                      disabled={running}
                    />
                    <span>
                      <code>{tool.name}</code>{" "}
                      <span className="text-muted-foreground">({tool.summary})</span>
                    </span>
                  </Label>
                ))}
                <p className="text-xs text-muted-foreground">
                  The session declares the checked tools; a call the model makes is run in this
                  browser and its result continues the same reply (up to {MAX_TOOL_ROUNDS} rounds
                  per message). Thinking and tools make the streamed text show the template's
                  markers; the transcript shows the parsed reply.
                </p>
              </div>
              {chatErrors.length > 0 ? (
                <p className="text-xs text-destructive">{chatErrors.join(" ")}</p>
              ) : null}
              <Button
                id="gemma4-new-chat"
                variant="outline"
                onClick={newChat}
                disabled={!canStartNewChat}
              >
                New chat
              </Button>
              {chatSettingsChanged ? (
                <p id="gemma4-chat-changed" className="text-xs text-muted-foreground">
                  These differ from the current chat's settings; press New chat to apply them.
                </p>
              ) : null}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Sampling</CardTitle>
              <CardDescription>Applies to each message as it is sent.</CardDescription>
            </CardHeader>
            <CardContent className="grid grid-cols-2 gap-4">
              <Field id="gemma4-temperature" label="Temperature" hint="0 = greedy.">
                <Input
                  id="gemma4-temperature"
                  inputMode="decimal"
                  value={turnForm.temperature}
                  onChange={(event) => updateTurn({ temperature: event.target.value })}
                />
              </Field>
              <Field id="gemma4-seed" label="Seed">
                <Input
                  id="gemma4-seed"
                  inputMode="numeric"
                  value={turnForm.seed}
                  onChange={(event) => updateTurn({ seed: event.target.value })}
                />
              </Field>
              <Field id="gemma4-top-k" label="Top-k" hint="Empty = no narrowing.">
                <Input
                  id="gemma4-top-k"
                  inputMode="numeric"
                  value={turnForm.topK}
                  onChange={(event) => updateTurn({ topK: event.target.value })}
                />
              </Field>
              <Field id="gemma4-top-p" label="Top-p" hint="Empty = no narrowing.">
                <Input
                  id="gemma4-top-p"
                  inputMode="decimal"
                  value={turnForm.topP}
                  onChange={(event) => updateTurn({ topP: event.target.value })}
                />
              </Field>
              <div className="col-span-2">
                <Field
                  id="gemma4-max-new-tokens"
                  label="Max new tokens"
                  hint="Empty = generate until the model stops or the context is full; 1,024 tokens are reserved when deciding whether old turns must be dropped."
                >
                  <Input
                    id="gemma4-max-new-tokens"
                    inputMode="numeric"
                    value={turnForm.maxNewTokens}
                    onChange={(event) => updateTurn({ maxNewTokens: event.target.value })}
                  />
                </Field>
              </div>
              <div className="col-span-2 flex flex-wrap gap-2">
                <Button
                  id="gemma4-sampler-default"
                  variant="outline"
                  size="sm"
                  onClick={() => setTurnFormEdit(undefined)}
                >
                  Distribution's sampler
                </Button>
                <Button
                  id="gemma4-sampler-greedy"
                  variant="outline"
                  size="sm"
                  onClick={() => updateTurn({ temperature: "0" })}
                >
                  Greedy
                </Button>
              </div>
              <p className="col-span-2 text-xs text-muted-foreground">
                {distributionSampler === undefined
                  ? "The distribution declares no sampler (the form starts greedy)."
                  : `The distribution declares temperature ${distributionSampler.temperature}, top-k ${distributionSampler.topK}, top-p ${distributionSampler.topP}.`}
              </p>
              {turnParsed.ok ? null : (
                <p className="col-span-2 text-xs text-destructive">{turnParsed.errors.join(" ")}</p>
              )}
            </CardContent>
          </Card>
        </div>

        <div className="flex flex-col gap-6">
          <Card>
            <CardHeader>
              <CardTitle>Conversation</CardTitle>
              <CardDescription>
                Enter sends, Shift+Enter starts a new line. Stop ends the reply where it is (the
                partial reply stays in the history; a turn stopped before any text leaves none).
              </CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-3">
              <ChatTranscript
                entries={entries}
                turns={turnRecords}
                running={running}
                pending={pendingText}
                pendingImages={pendingImages}
                live={live}
              />
              {notice === undefined ? null : (
                <Alert id="gemma4-turn-notice">
                  <AlertTitle>The last message is not in the history</AlertTitle>
                  <AlertDescription>{notice}</AlertDescription>
                </Alert>
              )}
              {awaitingTools ? (
                <Alert id="gemma4-tool-notice">
                  <AlertTitle>Waiting for tool results</AlertTitle>
                  <AlertDescription>
                    {toolNotice ??
                      "The last model turn ended on a tool call that was not answered (a Stop or a failure before its continuation wrote any text)."}{" "}
                    The session takes no new message while a call is open: press New chat to start a
                    new conversation.
                  </AlertDescription>
                </Alert>
              ) : null}
              <div className="flex flex-wrap gap-2">
                {SAMPLE_MESSAGES.map((sample) => (
                  <Button
                    key={sample.label}
                    variant="secondary"
                    size="xs"
                    onClick={() => setInput(sample.text)}
                    disabled={running}
                  >
                    {sample.label}
                  </Button>
                ))}
              </div>
              {visionConfig === undefined ? null : (
                <div className="flex flex-col gap-2">
                  <Field label="Attach images" id="gemma4-image">
                    <Input
                      id="gemma4-image"
                      type="file"
                      accept="image/*"
                      multiple
                      onChange={(event) => {
                        attachFiles(event.target.files);
                        // Cleared so that picking the same file again fires `change`.
                        event.target.value = "";
                      }}
                      disabled={running}
                    />
                  </Field>
                  {attachError === undefined ? null : (
                    <ErrorAlert title="The image could not be decoded" error={attachError} />
                  )}
                  {attachments.length === 0 ? null : (
                    <div id="gemma4-attachments" className="flex flex-wrap gap-2">
                      {attachments.map(({ id, image }) => (
                        <div key={id} className="flex flex-col items-start gap-1">
                          <ImageThumbnail image={image} />
                          <Button
                            variant="outline"
                            size="xs"
                            disabled={running}
                            onClick={() =>
                              setAttachments((current) =>
                                current.filter((attachment) => attachment.id !== id),
                              )
                            }
                          >
                            Remove
                          </Button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
              <InputGroup>
                <InputGroupTextarea
                  id="gemma4-input"
                  value={input}
                  rows={3}
                  placeholder={
                    readyChat === undefined ? "Load a model to start chatting." : "Message"
                  }
                  onChange={(event) => setInput(event.target.value)}
                  onKeyDown={onInputKey}
                  disabled={readyChat === undefined || running}
                />
                <InputGroupAddon align="block-end" className="justify-end">
                  {running ? (
                    <InputGroupButton
                      id="gemma4-stop"
                      variant="outline"
                      size="sm"
                      onClick={task.cancel}
                    >
                      <SquareIcon />
                      Stop
                    </InputGroupButton>
                  ) : (
                    <InputGroupButton
                      id="gemma4-send"
                      variant="default"
                      size="sm"
                      onClick={send}
                      disabled={!canSend}
                    >
                      <SendIcon />
                      Send
                    </InputGroupButton>
                  )}
                </InputGroupAddon>
              </InputGroup>
            </CardContent>
          </Card>

          <RunStatusCard
            gpu={gpu.state}
            onReacquire={gpu.reacquire}
            task={task}
            rows={[
              ["Pipeline", pipelineText(pipeline)],
              ["Chat", readyChat === undefined ? "—" : settingsText(readyChat.settings)],
              ["Context", contextUsage === undefined ? "—" : contextUsageText(contextUsage)],
            ]}
          >
            {pipeline.kind === "error" ? (
              <ErrorAlert title="Loading the pipeline failed" error={pipeline.error} />
            ) : null}
            {chat.kind === "error" ? (
              <ErrorAlert title="Building the chat session failed" error={chat.error} />
            ) : null}
            {pipelineReleaseError === undefined ? null : (
              <ErrorAlert
                title="Releasing the previous pipeline failed"
                error={pipelineReleaseError}
              />
            )}
            {sessionReleaseError === undefined ? null : (
              <ErrorAlert
                title="Closing the previous chat session failed"
                error={sessionReleaseError}
              />
            )}
          </RunStatusCard>

          {record === undefined ? null : (
            <Card>
              <CardHeader>
                <CardTitle>Record</CardTitle>
                <CardDescription>
                  {record.turns.length} turn{record.turns.length === 1 ? "" : "s"}: the settings,
                  each turn's request, stop, statistics and generated token ids
                  {record.diagnosticsCollected ? ", and the Session diagnostics" : ""}.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <RecordActions
                  record={record}
                  filename={`gemma4_${fileStamp(new Date(record.recordedAt))}.json`}
                />
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};
