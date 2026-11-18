/**
 * The candidate selection on real weights — a sampled decode step that reads back the leading
 * logits of every part of the row, selected on the GPU, instead of the whole row.
 *
 * The invariant: **a pipeline with the selection generates the token ids a pipeline without it
 * generates**, for the same request and seed. The selection changes where the candidates of a draw
 * come from, not the draw: the sampler ranks, normalizes and draws on the host either way, with the
 * same uniform values. The pipeline without it is built with the test-only key
 * `WHOLE_ROW_SAMPLING`.
 *
 * Per model, both pipelines run:
 *
 * ① **chats** — an open request with the distribution's own sampler (temperature 1, top-k 64,
 *    top-p 0.95) that runs to the token limit, a short answer at another temperature and top-k, and
 *    a request whose sampler the selection does not serve (a repetition penalty). The stops must
 *    match too, and the pipeline with the selection must report that it drew from it
 *    (`done.candidateSelection`) — a selection that fell back to the whole row at every step would
 *    pass the id comparison alone.
 * ② **a two-turn sequence** — the second turn prefills with the first turn's frontier prepended, so
 *    a frontier lost or fed twice around the selection's draw shows as a different second turn.
 *
 * 12B has suppressed tokens (`<image|>` / `<audio|>`), which the draw masks on the host from two
 * more leading entries.
 *
 * ## Assets
 *
 * The QAT distribution mirror `models/karume-gemma4-qat/` (outside git; assemble it with dist.py in
 * tools/export-recipes). Without the mirror or a GPU every case SKIPs explicitly (ADR 0005).
 */
import { test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { nodeDirectory } from "@karume/hub/node";
import {
  chatPrompt,
  type ChatStop,
  chatTurn,
  Pipeline,
  type SamplerSpec,
} from "../../models/gemma.ts";
import { WHOLE_ROW_SAMPLING } from "../../models/src/gemma/pipeline.ts";
import { mirrorAvailable } from "../../models/tests/helpers/gemma-mirror.ts";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";

const QAT_ROOT = new URL("../../../models/karume-gemma4-qat/", import.meta.url);
const QAT_PRESENT = mirrorAvailable(QAT_ROOT);
if (!QAT_PRESENT) {
  console.warn(
    "[karume] models/karume-gemma4-qat/ is absent (or not karume/6), so the candidate selection " +
      "cases SKIP (assemble it with dist.py in tools/export-recipes)",
  );
}

/** The distributions' declared sampler, with a seed. */
const DECLARED: SamplerSpec = { temperature: 1, topK: 64, topP: 0.95, seed: 20261118 };

const CHATS: readonly {
  readonly text: string;
  readonly maxNewTokens: number;
  readonly sampler: SamplerSpec;
  /** Whether the selection serves this sampler. */
  readonly served: boolean;
}[] = [
  {
    text: "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    maxNewTokens: 48,
    sampler: DECLARED,
    served: true,
  },
  {
    text: "Answer with one word: what colour is the sky on a clear day?",
    maxNewTokens: 24,
    sampler: { temperature: 0.7, topK: 40, seed: 5 },
    served: true,
  },
  {
    text: "Name four fruits, separated by commas.",
    maxNewTokens: 24,
    sampler: { ...DECLARED, repetitionPenalty: 1.1 },
    served: false,
  },
];

type Observed = {
  readonly chats: readonly { readonly ids: readonly number[]; readonly stop: ChatStop }[];
  readonly turns: readonly (readonly number[])[];
};

/** Everything one pipeline produces (one load per pipeline). */
const observe = async (model: string, wholeRow: boolean): Promise<Observed> => {
  const pipeline = await Pipeline.fromPretrained(nodeDirectory(QAT_ROOT), {
    model,
    quant: "i4",
    ...(wholeRow ? { [WHOLE_ROW_SAMPLING]: true as const } : {}),
  });
  try {
    const chats = [];
    for (const chat of CHATS) {
      const ids: number[] = [];
      const stream = pipeline.chat([{ role: "user", content: chat.text }], {
        maxNewTokens: chat.maxNewTokens,
        sampler: chat.sampler,
        onToken: (id) => ids.push(id),
      });
      await stream.text();
      chats.push({ ids, stop: await stream.done });
    }
    const tokenizer = pipeline.tokenizer;
    const sequence = await pipeline.sequence();
    const turns: number[][] = [];
    try {
      for (const [prompt, maxNewTokens] of [
        [chatPrompt(tokenizer, [{ role: "user", content: "Describe a harbour at dawn." }]), 24],
        [chatTurn(tokenizer, { role: "user", content: "Now describe it at midnight." }), 16],
      ] as const) {
        const ids: number[] = [];
        const stream = sequence.generate({
          prompt,
          maxNewTokens,
          sampler: { temperature: 1.5, topK: 64, topP: 0.95, seed: 11 },
        });
        for await (const event of stream) if (event.kind === "token") ids.push(event.id);
        await stream.done;
        turns.push(ids);
      }
    } finally {
      await sequence.dispose();
    }
    return { chats, turns };
  } finally {
    await pipeline.dispose();
  }
};

/** The stop without the selection's counters (the one field the two pipelines differ in). */
const stopOf = (stop: ChatStop): Omit<ChatStop, "candidateSelection"> => {
  const { candidateSelection: _counters, ...rest } = stop;
  return rest;
};

/** The models, with the leading entries the declared sampler asks for: top-k 64 + suppressed + 1. */
for (const [model, count] of [
  ["e2b", 65],
  ["12b", 67],
] as const) {
  test.skipIf(!QAT_PRESENT || !GPU_AVAILABLE)(
    `the candidate selection: QAT ${model} sampled token ids equal the whole-row pipeline's (real GPU)`,
    async () => {
      const whole = await observe(model, true);
      const selected = await observe(model, false);
      assertEquals(
        whole.chats[0].ids.length,
        CHATS[0].maxNewTokens,
        "the open request stopped early, so it does not cover the decode steps it is meant to",
      );
      for (const [at, chat] of selected.chats.entries()) {
        const reference = whole.chats[at];
        assertEquals(chat.ids, reference.ids, `chat ${at} token ids`);
        assertEquals(stopOf(chat.stop), stopOf(reference.stop), `chat ${at} stop`);
        // The pipeline built without the selection has no counters at all.
        assertEquals(reference.stop.candidateSelection, undefined, `chat ${at}`);
        const counters = chat.stop.candidateSelection;
        assert(counters !== undefined, `chat ${at}: the pipeline did not report the selection`);
        const decodeSteps = chat.stop.tokens - 1;
        if (!CHATS[at].served) {
          assertEquals(counters, { count: 0, steps: 0, fallbacks: 0 }, `chat ${at}`);
          continue;
        }
        const expected = (CHATS[at].sampler.topK ?? 0) + (count - 65) + 1;
        assertEquals(counters.count, expected, `chat ${at}: the entries asked for`);
        assertEquals(counters.steps + counters.fallbacks, decodeSteps, `chat ${at}`);
        // A step may legitimately read the whole row (a tie the entries cannot settle); a turn
        // that mostly does is not using the selection.
        assert(
          counters.fallbacks <= 2,
          `chat ${at}: ${counters.fallbacks} of ${decodeSteps} decode steps read the whole row`,
        );
      }
      assertEquals(selected.turns, whole.turns, "the two-turn sequence's token ids");
      assert(
        new Set(selected.chats[0].ids).size > 8,
        "the open request's tokens barely vary, so the comparison says little",
      );
    },
  );
}
