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
 * ② **a two-turn sequence** at a temperature other than 1 — the second turn prefills with the first
 *    turn's frontier prepended, so a frontier lost or fed twice around the selection's draw shows as
 *    a different second turn. Each turn's stop reports its own draws from the selection.
 *
 * 12B has suppressed tokens (`<image|>` / `<audio|>`), which the draw masks on the host from two
 * more leading entries.
 *
 * ③ **the logits themselves** — what ① and ② rest on is that a candidate step computes the logits
 *    a normal run computes: the step issues the target through `enqueue` and copies the logits to a
 *    resident, a normal run issues it through `run` (the per-layer embeddings on the host) or
 *    `enqueueRead` (on the GPU). The case generates the same sampled turn both ways on one pipeline
 *    and compares every decode step's logits **byte for byte**, on the mobile QAT E2B with the
 *    per-layer embeddings on the host and on the GPU. Equal token ids would not show a logit that
 *    differs in a low bit.
 *
 * ## Assets
 *
 * The QAT distribution mirror `models/karume-gemma4-qat/` and, for ③, the mobile QAT mirror
 * `models/karume-gemma4-qat-mobile/` (outside git; assemble them with dist.py in
 * tools/export-recipes). Without its mirror or a GPU a case SKIPs explicitly (ADR 0005).
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
import {
  type FromPretrainedOptions,
  type Gemma4State,
  GemmaPipeline,
  WHOLE_ROW_SAMPLING,
} from "../../models/src/gemma/pipeline.ts";
import {
  createGenerationSequence,
  type GenerationRequest,
  type GenerationSequence,
} from "../../models/src/generation/sequence.ts";
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

const MOBILE_ROOT = new URL("../../../models/karume-gemma4-qat-mobile/", import.meta.url);
const MOBILE_PRESENT = mirrorAvailable(MOBILE_ROOT);
if (!MOBILE_PRESENT) {
  console.warn(
    "[karume] models/karume-gemma4-qat-mobile/ is absent (or not karume/6), so the logits " +
      "comparison of the candidate step SKIPs (assemble it with dist.py in tools/export-recipes)",
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
  readonly turns: readonly { readonly ids: readonly number[]; readonly stop: ChatStop }[];
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
    const turns: { ids: number[]; stop: ChatStop }[] = [];
    try {
      for (const [prompt, maxNewTokens] of [
        [chatPrompt(tokenizer, [{ role: "user", content: "Describe a harbour at dawn." }]), 24],
        [chatTurn(tokenizer, { role: "user", content: "Now describe it at midnight." }), 16],
      ] as const) {
        const ids: number[] = [];
        const stream = sequence.generate({
          prompt,
          maxNewTokens,
          sampler: TURN_SAMPLER,
        });
        for await (const event of stream) if (event.kind === "token") ids.push(event.id);
        turns.push({ ids, stop: await stream.done });
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

/** The two-turn sequence's sampler: a temperature other than 1. */
const TURN_SAMPLER: SamplerSpec = { temperature: 1.5, topK: 64, topP: 0.95, seed: 11 };

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
        // A step may legitimately read the whole row (a tie the entries cannot settle, a part
        // that holds too many of them). Which steps do depends on the logits of these prompts on
        // these weights and this device, so the count is not pinned; the bound is what they need
        // with room to spare, and a selection that stopped answering reads the whole row at most
        // of the steps.
        assert(
          counters.fallbacks <= 2,
          `chat ${at}: ${counters.fallbacks} of ${decodeSteps} decode steps read the whole row`,
        );
      }
      for (const [at, turn] of selected.turns.entries()) {
        const reference = whole.turns[at];
        assertEquals(turn.ids, reference.ids, `turn ${at} token ids`);
        assertEquals(stopOf(turn.stop), stopOf(reference.stop), `turn ${at} stop`);
        assertEquals(reference.stop.candidateSelection, undefined, `turn ${at}`);
        // The second turn draws from the selection as the first does: its counters are its own.
        const counters = turn.stop.candidateSelection;
        assert(counters !== undefined, `turn ${at}: the sequence did not report the selection`);
        assertEquals(counters.count, (TURN_SAMPLER.topK ?? 0) + (count - 65) + 1, `turn ${at}`);
        assertEquals(counters.steps + counters.fallbacks, turn.stop.tokens - 1, `turn ${at}`);
        // A temperature other than 1 adds the ties it rounds two logits into; the same reading
        // of the bound as above.
        assert(
          counters.fallbacks <= 2,
          `turn ${at}: ${counters.fallbacks} of ${turn.stop.tokens - 1} decode steps read the whole row`,
        );
      }
      assert(
        new Set(selected.chats[0].ids).size > 8,
        "the open request's tokens barely vary, so the comparison says little",
      );
    },
  );
}

/** A mobile QAT pipeline with its internal state in reach (the generation surface and the wiring). */
class StatePipeline extends GemmaPipeline {
  static async open(
    options: FromPretrainedOptions,
  ): Promise<{ readonly pipeline: StatePipeline; readonly state: Gemma4State }> {
    const state = await super.loadPretrained("gemma4-qat", nodeDirectory(MOBILE_ROOT), options);
    return { pipeline: new StatePipeline(state), state };
  }
}

/** The token ids of one turn. */
const turnIds = async (sequence: GenerationSequence, request: GenerationRequest) => {
  const ids: number[] = [];
  const stream = sequence.generate(request);
  for await (const event of stream) if (event.kind === "token") ids.push(event.id);
  await stream.done;
  return ids;
};

/** The first word at which two rows differ (`-1` = none), compared as bits. */
const firstDifference = (left: Float32Array<ArrayBuffer>, right: Float32Array<ArrayBuffer>) => {
  const a = new Uint32Array(left.buffer, left.byteOffset, left.length);
  const b = new Uint32Array(right.buffer, right.byteOffset, right.length);
  if (a.length !== b.length) return 0;
  return a.findIndex((word, at) => word !== b[at]);
};

for (const pleResidency of ["host", "gpu"] as const) {
  test.skipIf(!MOBILE_PRESENT || !GPU_AVAILABLE)(
    `the candidate selection: a candidate step's logits are a normal run's, byte for byte (per-layer embeddings on the ${pleResidency}, real GPU)`,
    async () => {
      const { pipeline, state } = await StatePipeline.open({ model: "e2b", pleResidency });
      try {
        const surface = state.greedyOutput;
        const selection = state.candidateSelection;
        assert(surface !== undefined && selection !== undefined, "the pipeline has no selection");
        const request: GenerationRequest = {
          prompt: chatPrompt(state.tokenizer, [
            { role: "user", content: "Name three rivers and the seas they reach." },
          ]),
          maxNewTokens: 9,
          sampler: { temperature: 0.8, topK: 40, seed: 7 },
        };

        // A normal run: the surface's `run` (the target's `run` with the embeddings on the host,
        // `enqueueRead` with them on the GPU). Every run's logits are kept; the decode steps are
        // the last of them.
        const runRows: Float32Array<ArrayBuffer>[] = [];
        const recording: typeof surface.session = {
          ...surface.session,
          run: async (inputs, bindings, generation) => {
            const outputs = await surface.session.run(inputs, bindings, generation);
            const logits = outputs[state.wiring.logits];
            assert(logits?.dtype === "f32", "the run has no f32 logits");
            runRows.push(logits.data.slice());
            return outputs;
          },
        };
        const runSequence = await createGenerationSequence({
          session: recording,
          program: state.wiring,
        });
        let runIds: number[];
        try {
          runIds = await turnIds(runSequence, request);
        } finally {
          await runSequence.dispose();
        }

        // A candidate step: the logits it copied to the resident, read whole before the draw.
        const stepRows: Float32Array<ArrayBuffer>[] = [];
        const stepSequence = await createGenerationSequence({
          session: surface.session,
          program: state.wiring,
          candidateSelection: {
            maxCount: selection.maxCount,
            run: (inputs, generation, draw) =>
              selection.run(inputs, generation, async (row) => {
                stepRows.push((await row.whole()).slice());
                return await draw(row);
              }),
          },
        });
        let stepIds: number[];
        try {
          stepIds = await turnIds(stepSequence, request);
        } finally {
          await stepSequence.dispose();
        }

        assertEquals(stepIds, runIds, "the two paths' token ids");
        assert(stepRows.length >= 4, `only ${stepRows.length} decode steps to compare`);
        assert(
          runRows.length > stepRows.length,
          "the normal runs have no prefill before the decode",
        );
        const decodeRows = runRows.slice(-stepRows.length);
        for (const [step, row] of stepRows.entries()) {
          assertEquals(row.length, state.wiring.vocabSize, `step ${step}: the row's length`);
          const at = firstDifference(row, decodeRows[step]);
          assertEquals(
            at,
            -1,
            `decode step ${step}: logit ${at} is ${row[at]} in the candidate step and ${decodeRows[step][at]} in the normal run`,
          );
        }
      } finally {
        await pipeline.dispose();
      }
    },
  );
}
