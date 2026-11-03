/**
 * The bench loop: for every visit (an arm, in {@link visitOrder}) build the arm's pipeline, run
 * every chosen prompt `chats` times, and release the pipeline before the next visit is built — so
 * two models are never resident at once and every visit starts from the same cold state. Chat 0 of
 * each prompt in a visit is the cold one (the summary leaves it out).
 *
 * The loop takes its pipeline builder, probe and clock from the caller, so `vp test` drives it with
 * fakes. Every chat is a fresh `chat` call (one sequence per chat), sent with the arm's sampler
 * explicitly. The runner never asks for Session diagnostics: a pipeline built without
 * `onRunDiagnostics` keeps its on-GPU greedy selection, which is the decode path measured.
 */

import type { ChatMessage, ChatOptions, ChatStream, SamplerSpec } from "@karume/models/gemma";

import { type StopRecord, stopRecordOf } from "@/models/gemma4/chat-record";

import { type ChatMeasurement, type ChatMarks, measureChat, type TokenMark } from "./measure";
import {
  type BenchArm,
  type BenchPromptId,
  type BenchSettings,
  benchPrompt,
  meetsDistinctIdCondition,
  type Visit,
  visitOrder,
} from "./plan";

/** What the loop needs of a pipeline. */
export type BenchPipeline = {
  /** The prompt `chat` will prefill for `messages` (`chatPrompt` with the pipeline's tokenizer). */
  readonly encodePrompt: (messages: readonly ChatMessage[]) => readonly number[];
  readonly chat: (messages: readonly ChatMessage[], options: ChatOptions) => ChatStream;
  readonly dispose: () => Promise<void>;
};

/** What the loop needs of the tap's probe (`DeviceProbe`). */
export type BenchProbe = {
  enabled: boolean;
  /** `TapMode`: set per chat from the arm (a probe without the split mode leaves it out). */
  mode?: "passes" | "dispatches";
  readonly passes: () => readonly {
    readonly dispatches: number;
    readonly label?: string;
    beginNs?: bigint;
    endNs?: bigint;
  }[];
  readonly submits: () => number;
  readonly readTimes: () => Promise<void>;
  readonly reset: () => void;
};

/** An arm with the sampler its chats are sent with. */
export type RunArm = BenchArm & { readonly sampler: SamplerSpec };

export type VisitRecord = Visit & { readonly loadSeconds: number };

export type ChatRecord = {
  readonly visit: number;
  readonly round: number;
  readonly arm: number;
  readonly prompt: BenchPromptId;
  /** 0-based within the prompt and visit; 0 = cold (left out of the summary). */
  readonly chat: number;
  readonly cold: boolean;
  readonly promptTokens: number;
  readonly distinctPromptIds: number;
  /** Whether the distinct id count is on the side of the threshold its prompt is meant for. */
  readonly meetsDistinctIdCondition: boolean;
  readonly stop: StopRecord;
  /** sha256 of the generated token ids (u32 little-endian; the stop token excluded). */
  readonly tokenIdsSha256: string;
  readonly measurement: ChatMeasurement;
};

export type BenchProgress =
  | { readonly kind: "loading"; readonly visit: Visit; readonly visits: number }
  | {
      readonly kind: "chat";
      readonly visit: Visit;
      readonly visits: number;
      readonly prompt: BenchPromptId;
      readonly chat: number;
    };

export type BenchDependencies = {
  readonly build: (arm: RunArm, signal: AbortSignal) => Promise<BenchPipeline>;
  readonly probe: BenchProbe;
  readonly now: () => number;
  readonly sha256: (bytes: Uint8Array<ArrayBuffer>) => Promise<string>;
  /** Lets the page render between chats, outside the measured windows (one macrotask by default). */
  readonly pause?: () => Promise<void>;
};

const nextMacrotask = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 0));

const idBytes = (ids: readonly number[]): Uint8Array<ArrayBuffer> => {
  const words = Uint32Array.from(ids);
  return new Uint8Array(words.buffer, words.byteOffset, words.byteLength);
};

/** One chat: marks, the tap's records (when the arm taps), the measurement. */
const runChat = async (input: {
  readonly pipeline: BenchPipeline;
  readonly arm: RunArm;
  readonly prompt: BenchPromptId;
  readonly settings: BenchSettings;
  readonly signal: AbortSignal;
  readonly dependencies: BenchDependencies;
}): Promise<Omit<ChatRecord, "visit" | "round" | "arm" | "prompt" | "chat" | "cold">> => {
  const { pipeline, arm, prompt, settings, signal, dependencies } = input;
  const { probe, now } = dependencies;
  const messages: readonly ChatMessage[] = [{ role: "user", content: benchPrompt(prompt).text }];
  const promptIds = pipeline.encodePrompt(messages);
  const distinctPromptIds = new Set(promptIds).size;
  const prefillAt: number[] = [];
  const tokens: TokenMark[] = [];
  const ids: number[] = [];
  const drain = async () => {
    const sentAt = now();
    const stream = pipeline.chat(messages, {
      maxNewTokens: settings.newTokens,
      sampler: arm.sampler,
      signal,
      // The same callback work in both tap states (the probe reads 0 when off).
      onPrefill: () => {
        prefillAt.push(now());
      },
      onToken: (id) => {
        tokens.push({ at: now(), passes: probe.passes().length, submits: probe.submits() });
        ids.push(id);
      },
    });
    await stream.text();
    const endAt = now();
    return { sentAt, endAt, stop: await stream.done };
  };
  probe.reset();
  probe.mode = arm.split === true ? "dispatches" : "passes";
  probe.enabled = arm.tap;
  const { sentAt, endAt, stop } = await drain().finally(() => {
    probe.enabled = false;
  });
  if (arm.tap) await probe.readTimes();
  const marks: ChatMarks = { sentAt, prefillAt, tokens, endAt };
  return {
    promptTokens: promptIds.length,
    distinctPromptIds,
    meetsDistinctIdCondition: meetsDistinctIdCondition(benchPrompt(prompt), distinctPromptIds),
    stop: stopRecordOf(stop),
    tokenIdsSha256: await dependencies.sha256(idBytes(ids)),
    measurement: measureChat({
      marks,
      promptTokens: promptIds.length,
      tokens: stop.tokens,
      ...(arm.tap
        ? {
            tap: {
              passes: probe.passes(),
              submits: probe.submits(),
              ...(arm.split === true ? { split: true } : {}),
            },
          }
        : {}),
    }),
  };
};

/**
 * Builds the pipeline, hands it to `use`, and releases it whatever happens. A failed release after a
 * failed `use` is logged and the first failure is the one thrown.
 */
const withPipeline = async (
  build: () => Promise<BenchPipeline>,
  use: (pipeline: BenchPipeline) => Promise<void>,
): Promise<void> => {
  const pipeline = await build();
  let failure: { readonly error: unknown } | undefined;
  try {
    await use(pipeline);
  } catch (error) {
    failure = { error };
  }
  try {
    await pipeline.dispose();
  } catch (error) {
    if (failure === undefined) throw error;
    console.error("Releasing the bench pipeline failed", error);
  }
  if (failure !== undefined) throw failure.error;
};

/**
 * Runs the whole bench. `onVisit` and `onChat` get each finished visit and chat as it lands (the
 * page shows them and keeps them when a later step fails or is cancelled); the promise resolves when
 * every visit is done and rejects with the first failure or the signal's reason.
 */
export const runDecodeBench = async (input: {
  readonly arms: readonly RunArm[];
  readonly settings: BenchSettings;
  readonly signal: AbortSignal;
  readonly dependencies: BenchDependencies;
  readonly onProgress: (progress: BenchProgress) => void;
  readonly onVisit: (visit: VisitRecord) => void;
  readonly onChat: (chat: ChatRecord) => void;
}): Promise<void> => {
  const { arms, settings, signal, dependencies, onProgress, onVisit, onChat } = input;
  const pause = dependencies.pause ?? nextMacrotask;
  const visits = visitOrder(arms.length, settings.rounds);
  for (const visit of visits) {
    signal.throwIfAborted();
    const arm = arms[visit.arm];
    onProgress({ kind: "loading", visit, visits: visits.length });
    const loadStarted = dependencies.now();
    await withPipeline(
      async () => {
        // Built with the tap off: the load's own passes are not part of any chat.
        dependencies.probe.enabled = false;
        const pipeline = await dependencies.build(arm, signal);
        // A build that finished after an abort is released at once (withPipeline's dispose).
        return pipeline;
      },
      async (pipeline) => {
        signal.throwIfAborted();
        onVisit({
          ...visit,
          loadSeconds: Number(((dependencies.now() - loadStarted) / 1000).toFixed(3)),
        });
        for (const prompt of settings.prompts) {
          for (let chat = 0; chat < settings.chats; chat += 1) {
            onProgress({ kind: "chat", visit, visits: visits.length, prompt, chat });
            await pause();
            signal.throwIfAborted();
            const result = await runChat({ pipeline, arm, prompt, settings, signal, dependencies });
            onChat({
              visit: visit.visit,
              round: visit.round,
              arm: visit.arm,
              prompt,
              chat,
              cold: chat === 0,
              ...result,
            });
          }
        }
      },
    );
  }
};
