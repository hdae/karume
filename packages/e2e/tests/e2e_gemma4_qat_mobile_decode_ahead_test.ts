/**
 * The decode run-ahead (`PipelineOptions.decodeAhead`) on real weights — QAT Gemma 4 with the
 * GPU-resident PLE.
 *
 * The invariant: **with the run-ahead on, the generated token ids are identical to greedy decode
 * without it** (`decodeAhead: 1`). The same kernels run on the same inputs; only the batching
 * changes — each step's token comes from the previous step's on-GPU selection instead of a host
 * write after a readback, and the rows of a batch are committed together after one fence.
 *
 * Two gates, each run at depths 1 (the reference), 4 and 8:
 *
 * ① **chat turns** — an open request that runs to the token limit (the batches end on the budget,
 *    28 decode steps = 7 × 4 and 3 × 8 + 4) and a short answer that stops at the end-of-turn token
 *    inside a batch (the rows after it are computed and left uncommitted). The stop must match too,
 *    and a depth ≥ 2 must report the batches it ran (`done.decodeAhead`) — a run-ahead that silently
 *    fell back to one step per batch would pass the id comparison alone.
 * ② **a two-turn sequence** — the second turn prefills with the first turn's frontier prepended, so
 *    a commit of the wrong row count in the first turn shows as a different second turn.
 *
 * ## Assets
 *
 * The mobile QAT distribution mirror `models/karume-gemma4-qat-mobile/` (outside git; assemble it with dist.py in
 * tools/export-recipes). Without the mirror or a GPU every case SKIPs explicitly (ADR 0005).
 */
import { test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { nodeDirectory } from "@karume/hub/node";
import { chatPrompt, chatTurn, type ChatStop } from "../../models/gemma.ts";
import * as gemma4QatMobile from "../../models/gemma4-qat-mobile.ts";
import { mirrorAvailable } from "../../models/tests/helpers/gemma-mirror.ts";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";

const QAT_ROOT = new URL("../../../models/karume-gemma4-qat-mobile/", import.meta.url);
const QAT_PRESENT = mirrorAvailable(QAT_ROOT);
if (!QAT_PRESENT) {
  console.warn(
    "[karume] models/karume-gemma4-qat-mobile/ is absent (or not karume/5), so the decode run-ahead " +
      "cases SKIP (assemble it with dist.py in tools/export-recipes)",
  );
}

const DEPTHS = [1, 4, 8] as const;

const CHATS = [
  {
    text:
      "List three things that make the ocean important to the planet, with one short sentence " +
      "for each.",
    maxNewTokens: 29,
  },
  { text: "Answer with one word: what colour is the sky on a clear day?", maxNewTokens: 32 },
] as const;

type Observed = {
  readonly chats: readonly { readonly ids: readonly number[]; readonly stop: ChatStop }[];
  readonly turns: readonly (readonly number[])[];
};

/** Everything one depth produces (one pipeline load per depth). */
const observe = async (model: "e2b" | "e4b", depth: number): Promise<Observed> => {
  const pipeline = await gemma4QatMobile.Pipeline.fromPretrained(nodeDirectory(QAT_ROOT), {
    model,
    pleResidency: "gpu",
    decodeAhead: depth,
  });
  try {
    const chats = [];
    for (const chat of CHATS) {
      const ids: number[] = [];
      const stream = pipeline.chat([{ role: "user", content: chat.text }], {
        maxNewTokens: chat.maxNewTokens,
        sampler: { temperature: 0 },
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
        [
          chatPrompt(tokenizer, [
            { role: "user", content: "Describe a lighthouse in two sentences." },
          ]),
          24,
        ],
        [chatTurn(tokenizer, { role: "user", content: "Now describe it in five words." }), 16],
      ] as const) {
        const ids: number[] = [];
        const stream = sequence.generate({ prompt, maxNewTokens, sampler: { temperature: 0 } });
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

/** The stop without the run-ahead counters (the one field the depths legitimately differ in). */
const stopOf = (stop: ChatStop): Omit<ChatStop, "decodeAhead"> => {
  const { decodeAhead: _counters, ...rest } = stop;
  return rest;
};

for (const model of ["e2b"] as const) {
  test.skipIf(!QAT_PRESENT || !GPU_AVAILABLE)(
    `decodeAhead: QAT ${model} greedy token ids at depths 4 and 8 equal depth 1 (real GPU)`,
    async () => {
      const [reference, ...others] = await (async () => {
        const observed: Observed[] = [];
        for (const depth of DEPTHS) observed.push(await observe(model, depth));
        return observed;
      })();
      assertEquals(
        reference.chats[0].ids.length,
        CHATS[0].maxNewTokens,
        "the open request stopped early, so the batches do not end on the budget",
      );
      assert(
        reference.chats[1].stop.reason === "eos",
        `the short answer did not stop at the end of turn (${reference.chats[1].stop.reason})`,
      );
      for (const [index, observed] of others.entries()) {
        const depth = DEPTHS[index + 1];
        for (const [at, chat] of observed.chats.entries()) {
          assertEquals(chat.ids, reference.chats[at].ids, `depth ${depth}: chat ${at} token ids`);
          assertEquals(
            stopOf(chat.stop),
            stopOf(reference.chats[at].stop),
            `depth ${depth}: chat ${at} stop`,
          );
          assert(
            (chat.stop.decodeAhead?.batches ?? 0) > 0,
            `depth ${depth}: chat ${at} did not run the run-ahead (${JSON.stringify(chat.stop.decodeAhead)})`,
          );
        }
        assertEquals(observed.turns, reference.turns, `depth ${depth}: the two-turn sequence`);
      }
      assertEquals(
        reference.chats[0].stop.decodeAhead,
        undefined,
        "depth 1 has no run-ahead counters",
      );
    },
  );
}
