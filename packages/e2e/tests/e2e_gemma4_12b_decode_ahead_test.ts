/**
 * The decode run-ahead (`PipelineOptions.decodeAhead`) on 12B: a model without PLE, whose
 * distribution suppresses tokens (ADR 0142 addendum of 2026-11-06).
 *
 * The invariant: **with the run-ahead on, the generated token ids are identical to greedy decode
 * without it**, and the per-step GPU selection, which masks the suppressed ids, is identical to
 * the host's masked selection. The host reference is a pipeline with `onRunDiagnostics`, which
 * keeps per-step runs and reads the logits back.
 *
 * Three arms against that reference, on the reference quant and on `i4-fast` separately: depths 1,
 * 4 and 8. The first chat runs to the token limit (the batches end on the budget), the second stops
 * at the end of turn inside a batch, and a two-turn sequence shows a commit of the wrong row count
 * as a different second turn. A depth ≥ 2 must report the batches it ran: a run-ahead that silently
 * fell back to one step per batch would pass the id comparison alone.
 *
 * ## Assets
 *
 * The q4_0 QAT distribution mirror `models/karume-gemma4-qat/` (outside git). Without the mirror or
 * a GPU every case SKIPs explicitly (ADR 0005).
 */
import { test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import type { SessionDiagnostics } from "@karume/runtime";
import { nodeDirectory } from "@karume/hub/node";
import { chatPrompt, chatTurn, type ChatStop, Pipeline } from "../../models/gemma.ts";
import {
  assertRowCensus,
  censusRowOf,
  effectiveSessionOptions,
  gemmaComponentOf,
} from "../../models/tests/helpers/census-table.ts";
import { mirrorAvailable } from "../../models/tests/helpers/gemma-mirror.ts";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";

const ROOT = new URL("../../../models/karume-gemma4-qat/", import.meta.url);
const PRESENT = mirrorAvailable(ROOT);
if (!PRESENT) {
  console.warn(
    "[karume] models/karume-gemma4-qat/ is absent (or not karume/5), so the 12B decode run-ahead " +
      "cases SKIP (assemble it with dist.py in tools/export-recipes)",
  );
}

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

/** Everything one arm produces (one pipeline load per arm). */
const observe = async (
  quant: string,
  options: { readonly decodeAhead?: number; readonly hostSelection?: boolean },
): Promise<Observed> => {
  const pipeline = await Pipeline.fromPretrained(nodeDirectory(ROOT), {
    model: "12b",
    quant,
    ...(options.decodeAhead === undefined ? {} : { decodeAhead: options.decodeAhead }),
    ...(options.hostSelection === true ? { onRunDiagnostics: () => {} } : {}),
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

/** The stop without the run-ahead counters (the one field the arms legitimately differ in). */
const stopOf = (stop: ChatStop): Omit<ChatStop, "decodeAhead"> => {
  const { decodeAhead: _counters, ...rest } = stop;
  return rest;
};

const assertSameOutput = (observed: Observed, reference: Observed, where: string): void => {
  for (const [at, chat] of observed.chats.entries()) {
    assertEquals(chat.ids, reference.chats[at].ids, `${where}: chat ${at} token ids`);
    assertEquals(stopOf(chat.stop), stopOf(reference.chats[at].stop), `${where}: chat ${at} stop`);
  }
  assertEquals(observed.turns, reference.turns, `${where}: the two-turn sequence`);
};

for (const quant of ["i4", "i4-fast"] as const) {
  test.skipIf(!PRESENT || !GPU_AVAILABLE)(
    `decodeAhead: 12B ${quant} greedy token ids at depths 1, 4 and 8 equal the host selection (real GPU)`,
    async () => {
      const reference = await observe(quant, { hostSelection: true });
      assertEquals(
        reference.chats[0].ids.length,
        CHATS[0].maxNewTokens,
        "the open request stopped early, so the batches do not end on the budget",
      );
      assert(
        reference.chats[1].stop.reason === "eos",
        `the short answer did not stop at the end of turn (${reference.chats[1].stop.reason})`,
      );

      const perStep = await observe(quant, {});
      assertSameOutput(perStep, reference, "the masked GPU selection");
      assertEquals(perStep.chats[0].stop.decodeAhead, undefined, "depth 1 has no counters");

      for (const depth of [4, 8]) {
        const observed = await observe(quant, { decodeAhead: depth });
        assertSameOutput(observed, reference, `depth ${depth}`);
        for (const [at, chat] of observed.chats.entries()) {
          assert(
            (chat.stop.decodeAhead?.batches ?? 0) > 0,
            `depth ${depth}: chat ${at} did not run the run-ahead (${JSON.stringify(chat.stop.decodeAhead)})`,
          );
        }
      }
    },
  );
}

for (const [quant, session] of [
  ["i4-gemvpar", { linearGemvReduce: "parallel", stateAttentionReduce: "parallel" }],
  [
    "i4-fast",
    {
      linearGemvReduce: "parallel",
      fuseRmsNormAdd: true,
      stateAttentionReduce: "parallel-fused",
    },
  ],
] as const) {
  test.skipIf(!PRESENT || !GPU_AVAILABLE)(
    `12B ${quant}: the planned kernels equal the census table's row (real GPU)`,
    async () => {
      const row = censusRowOf("gemma4", "12b", effectiveSessionOptions("gemma4", session));
      assert(row !== undefined, `no census row for 12b ${quant}`);
      const byPhase = new Map<string, SessionDiagnostics["lastRunPipelines"][]>();
      const pipeline = await Pipeline.fromPretrained(nodeDirectory(ROOT), {
        model: "12b",
        quant,
        onRunDiagnostics: (diagnostics, phase) => {
          byPhase.set(phase.kind, [
            ...(byPhase.get(phase.kind) ?? []),
            diagnostics.lastRunPipelines,
          ]);
        },
      });
      try {
        const stream = pipeline.chat([{ role: "user", content: CHATS[1].text }], {
          maxNewTokens: 4,
          sampler: { temperature: 0 },
        });
        await stream.text();
        await stream.done;
      } finally {
        await pipeline.dispose();
      }
      assert(assertRowCensus(row, byPhase, gemmaComponentOf, quant) > 0, "nothing was compared");
    },
  );
}
