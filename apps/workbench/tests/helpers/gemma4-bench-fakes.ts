import type {
  ChatOptions,
  ChatStop,
  ChatStream,
  GenerationSpeculation,
} from "@karume/models/gemma";

import type { HostCall } from "../../src/gemma4-bench/measure.ts";
import type { BenchDependencies, BenchPipeline, BenchProbe } from "../../src/gemma4-bench/run.ts";

/** The label a pass gets in the split mode. */
export const SPLIT_LABEL = "linear_gemv_parallel:wi4g32:l32";

/**
 * What a fake run does besides streaming: `failChatAt` (0-based over the run) fails that chat,
 * `failBuild` fails every build, `onPause` runs at every pause, and `events` logs the builds, the
 * pauses and the chats in order (a caller may push its progress lines into it too).
 */
export type BenchFakeOptions = {
  readonly failChatAt?: number;
  readonly failBuild?: boolean;
  readonly onPause?: () => void;
  readonly events?: string[];
  /** The candidate selection counters every chat's stop reports (absent: none, as a pipeline without it). */
  readonly candidateSelection?: {
    readonly count: number;
    readonly steps: number;
    readonly fallbacks: number;
  };
  /**
   * The speculation every chat sent with `speculative: "always"` reports in its stop (absent:
   * none — the chats of an arm without the drafter never report it).
   */
  readonly speculation?: GenerationSpeculation;
};

/** What the fakes saw over a run. */
export type BenchWorld = {
  /** `build <model>` and `dispose <model>`, in order. */
  readonly lifecycle: string[];
  /** The most pipelines that were resident at once. */
  readonly maxResident: () => number;
  readonly sentOptions: ChatOptions[];
  /** Whether the probe was enabled when each chat was sent. */
  readonly enabledAtChat: boolean[];
};

/**
 * Fake dependencies: a clock that ticks 1 ms per reading, a `sha-<bytes>` hash, and pipelines
 * whose prompt is its text's char codes and whose `chat` streams `newTokens − 1` tokens then a
 * stop token. The probe records one pass of one dispatch per token step while enabled (labelled
 * {@link SPLIT_LABEL} in the split mode), and `readTimes` times every pass at 0.25 ms. In the host
 * mode it records no pass: a token step is one submit and one map cycle of 8 bytes, each call
 * marked with a reading of the clock, after two checks of the chat's signal when it has one.
 */
export const fakeBenchDependencies = (options: BenchFakeOptions = {}) => {
  let passes: { dispatches: number; label?: string; beginNs?: bigint; endNs?: bigint }[] = [];
  let marks: HostCall[] = [];
  let clock = 0;
  const probe: BenchProbe = {
    enabled: false,
    mode: "passes",
    passes: () => passes,
    submits: () => passes.length,
    marks: () => marks,
    readTimes: () => {
      passes = passes.map((pass, index) => ({
        ...pass,
        beginNs: BigInt(index * 1_000_000),
        endNs: BigInt(index * 1_000_000 + 250_000),
      }));
      return Promise.resolve();
    },
    reset: () => {
      passes = [];
      marks = [];
    },
  };
  const record = () => {
    if (!probe.enabled) return;
    if (probe.mode === "host") {
      for (const kind of [
        "encoder",
        "submit",
        "submitted",
        "map",
        "mapped",
        "range",
        "unmap",
      ] as const) {
        marks.push({ kind, at: (clock += 1), ...(kind === "range" ? { bytes: 8 } : {}) });
      }
      return;
    }
    passes.push({
      dispatches: 1,
      ...(probe.mode === "dispatches" ? { label: SPLIT_LABEL } : {}),
    });
  };
  const resident = new Set<string>();
  let maxResident = 0;
  const world: BenchWorld = {
    lifecycle: [],
    maxResident: () => maxResident,
    sentOptions: [],
    enabledAtChat: [],
  };
  let chats = 0;
  const dependencies: BenchDependencies = {
    probe,
    now: () => (clock += 1),
    sha256: (bytes) => Promise.resolve(`sha-${bytes.byteLength}`),
    pause: () => {
      options.events?.push("pause");
      options.onPause?.();
      return Promise.resolve();
    },
    build: (arm) => {
      options.events?.push(`build ${arm.model}`);
      if (options.failBuild === true) return Promise.reject(new Error("build failed"));
      world.lifecycle.push(`build ${arm.model}`);
      resident.add(arm.model);
      maxResident = Math.max(maxResident, resident.size);
      const pipeline: BenchPipeline = {
        encodePrompt: (messages) => {
          const [first] = messages;
          const text = typeof first?.content === "string" ? first.content : "";
          return Array.from({ length: text.length }, (_, index) => text.charCodeAt(index));
        },
        chat: (_messages, chatOptions) => {
          options.events?.push("chat");
          world.sentOptions.push(chatOptions);
          world.enabledAtChat.push(probe.enabled);
          const index = chats;
          chats += 1;
          let settle!: (stop: ChatStop) => void;
          let reject!: (error: unknown) => void;
          const done = new Promise<ChatStop>((resolve, fail) => {
            settle = resolve;
            reject = fail;
          });
          done.catch(() => {});
          const stream: ChatStream = {
            done,
            [Symbol.asyncIterator]: () => {
              throw new Error("the runner drains with text()");
            },
            text: async () => {
              if (index === options.failChatAt) {
                const error = new Error("chat failed");
                reject(error);
                throw error;
              }
              chatOptions.onPrefill?.({ chunk: 1, chunks: 1 });
              const maxNewTokens = chatOptions.maxNewTokens;
              if (maxNewTokens === undefined) throw new Error("the runner sets maxNewTokens");
              for (let token = 0; token < maxNewTokens - 1; token += 1) {
                // As the generation loop: after its yield, and right before it issues the run.
                chatOptions.signal?.throwIfAborted();
                chatOptions.signal?.throwIfAborted();
                record();
                chatOptions.onToken?.(100 + token);
              }
              chatOptions.signal?.throwIfAborted();
              chatOptions.signal?.throwIfAborted();
              record();
              settle({
                reason: "eos",
                tokens: maxNewTokens,
                token: 1,
                ...(options.candidateSelection === undefined
                  ? {}
                  : { candidateSelection: options.candidateSelection }),
                ...(options.speculation === undefined || chatOptions.speculative !== "always"
                  ? {}
                  : { speculation: options.speculation }),
              });
              return "text";
            },
          };
          return stream;
        },
        dispose: () => {
          world.lifecycle.push(`dispose ${arm.model}`);
          resident.delete(arm.model);
          return Promise.resolve();
        },
      };
      return Promise.resolve(pipeline);
    },
  };
  return { dependencies, probe, world };
};
