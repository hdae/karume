// Acceptance check for the real-weight Gemma 4 E2B **string in → string out** — the pass line of
// step 4.
//
// What it accepts is the wiring of `src/gemma/pipeline.ts` (`Pipeline`) and
// `src/gemma/text/chat.ts` (`chatPrompt`), with seven checks:
//
// ① **Runs to completion**: `fromAssets` → `chat([...])` runs on the real weights, and the output
//    at temperature 0 (this layer's default) matches a pinned greedy golden character for
//    character. The golden was **taken on this path itself**, so it does not prove the model
//    correct — what it proves is that "changing the wiring does not move the output" (numeric
//    correctness belongs to the cross parity of `e2e_gemma4_product_test.ts`)
// ② **The chat id sequence** matches the HF `apply_chat_template` fixture. `gemma4_chat_test.ts`
//    checks the same with the subset and the real assets; this one checks it with **the assets
//    the pipeline holds** (had it grabbed another tokenizer, the rendering would still match and
//    only the id sequence would differ)
// ③ **Streaming matches batch**: the concatenation of the streamed chunks matches the batch
//    decode of the token sequence obtained by running the same conversation on the low-level
//    surface (`sequence()`). ① stays green even if incremental decoding drops a byte run or turns
//    into "fake streaming that buffers everything until finish()", so this check is needed
// ④ **The stop set** matches the upstream `generation_config.json` declaration (`[1, 106, 50]`)
// ⑤ **Out-of-scope input fails before touching the GPU** (tools / an unknown role — `chat`
//    throws synchronously)
// ⑥ **Incremental multi-turn rendering** (`chatTurn` + `sequence()`) matches, **verbatim**, a
//    `chat()` that renders the whole conversation every turn. The turn-local contract (a check in
//    `gemma4_chat_test.ts`) is the equality of the id sequences; this check sees the same
//    equality in **the real-weight output**
// ⑦ **Two turns of `ChatSession`** produce the same strings as ⑥. ⑥ checks that "driving the
//    delta rendering by hand matches"; ⑦ checks that "leaving it to the middle layer gives the
//    same conversation" (the KV continuation condition and how the history accumulates live
//    inside the session, so this is the only check that sees them on real weights —
//    `gemma4_chat_session_test.ts` checks only the assembly, with a fake sequence)
//
// MUST: the entry point is imported from the public surface (`../gemma.ts`) — grabbing `src/...`
// directly would keep the check green even if the surface shrank (the acceptance would run
// through a path consumers cannot write).
//
// ## Assets
//
// The distribution mirror `models/karume-gemma4/` (`karume/5`). It is outside the repository, so
// where it is missing the cases **SKIP explicitly**. The seat is the regular line's `bf16` preset —
// the checkpoint's own weights with the PLE on the host (`fromAssets` takes every part of its
// container in RAM, about 9.3 GiB).
//
// NOTE: the series output (`outputs/series/gemma4-e2b-product/`) now writes `krm` too, but this
// surface cannot be built from a series — `fromAssets` takes **what the manifest carries** (the
// keys of the part list, the model / quant selection axes, the expected hashes of the two
// documents), and a series output, which has no manifest, has none of them (the same reason as
// in the module doc of `helpers/gemma-mirror.ts`). The weights and the PLE come from the same
// rebuild, so the golden assertions hold as they are (only the source differs from
// `e2e_gemma4_pretrained_test.ts`).

import { readTextFile } from "@karume/test-support/fs";
import { afterAll, beforeAll, describe, it, test } from "vite-plus/test";
import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
// MUST: the entry point is taken from the **public surface** (the `./gemma` subpath) — with an
// import consumers cannot write, the check could not notice that "the surface has shrunk"
// (grabbing `src/...` directly sees everything).
import {
  type ChatMessage,
  chatPrompt,
  ChatSession,
  type ChatStream,
  chatTurn,
  Pipeline,
  type PipelineOptions,
  type RunPhase,
} from "../../models/gemma.ts";
// The budget default is taken from the runtime's **public surface** (a copy would let the family
// side claim a stale value).
import {
  type AdmissionReport,
  DEFAULT_PLAN_BACKING_BUDGET_BYTES,
  type SessionDiagnostics,
} from "@karume/runtime";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
// The helper that builds `Assets` from the mirror (a series output has no manifest, so it cannot
// be used — the NOTE at the top).
import { gemma4MirrorAssets, mirrorAvailable } from "../../models/tests/helpers/gemma-mirror.ts";
import { allResidentPleBytesOfMirror } from "../../models/tests/helpers/ple-budget.ts";

const MIRROR_DIR = new URL("../../../models/karume-gemma4/", import.meta.url);
/**
 * The seat the goldens below were taken on: the regular line's `bf16` preset (its default and,
 * since 2026-11-04, its one preset). Named so a later default does not move the goldens.
 */
const BF16_SEAT = { quant: "bf16" } as const;

/** The build command, ready to paste when the cases SKIP. */
const GENERATE_COMMAND = "cd tools/export-recipes && uv run python dist.py --pipeline gemma4";

/**
 * The run conditions equal those of the existing gemma4 acceptance checks (the same ruling on the
 * asset generation, as is).
 */
const CHUNK_LENGTH = 768;
/**
 * The trace upper bound baked into the symbol `M` (equal to the distribution's declared
 * `maxChunkLength` — the `chunkLength` check).
 */
const MAX_CHUNK_LENGTH = 768;
const CAPACITY = 4096;
const MAX_POSITION = 131072;
/**
 * The RoPE parameters (the same values as the distribution's declaration — the table is not in
 * the assets; the host builds cos / sin). The source of truth is the upstream config, and the
 * check on spelling and range is `src/gemma/rope.ts`.
 */
const ROPE = {
  sliding_attention: { theta: 10000, headDim: 256, rotaryDim: 256 },
  full_attention: { theta: 1000000, headDim: 512, rotaryDim: 128 },
} as const;

/** `eos_token_id` of the `generation_config.json` of gemma-4-E2B-it (ADR 0083 decision 8). */
const STOP_TOKENS = [1, 106, 50];

/**
 * The acceptance cases. `prompt` is the case name in the fixture (`gemma4-chat.json`), and
 * `expected` is **the temperature-0 output taken on this path** — re-collected 2026-11-04 on the
 * RTX 3080 Ti (Node +
 * Dawn) when the gate moved from the regular `i4-fast` seat to `bf16` (a new target: the int4
 * seat leaves the regular line). `single-user` is unchanged (9 tokens); `japanese` was
 * "2023年時点で、およそ1,400万人です。" in 18 tokens on `i4-fast` (2026-08-31).
 *
 * MUST NOT: "overwrite the expected value with the run result to make it green" — when it splits,
 * first see whether the cross parity of `e2e_gemma4_product_test.ts` (the comparison with torch)
 * is green. If that one is green and only this one splits, what moved is the wiring (the chat
 * rendering / incremental decoding / the sampler default).
 */
const CASES = [
  {
    fixture: "single-user",
    maxNewTokens: 24,
    expected: "The capital of France is **Paris**.",
    // It closes the turn on its own with `<turn|>` (106) = evidence that the stop set takes effect
    // on the real output. `tokens` is the generated count **including the stop token** (the doc
    // of `GenerationStop.tokens` — the text is 8 tokens).
    stop: { reason: "eos", token: 106, tokens: 9 },
  },
  {
    fixture: "japanese",
    maxNewTokens: 24,
    // The reply to the 4th message of a multi-turn conversation (system + user + assistant + user).
    expected: "2023年時点の東京の人口は約1,400万人です。",
    stop: { reason: "eos", token: 106, tokens: 20 },
  },
] as const;

/** The second turn of ⑥ (the first turn uses the conversation of `CASES[0]` as is). */
const FOLLOW_UP: ChatMessage = { role: "user", content: "And of Japan?" };
const FOLLOW_UP_TOKENS = 24;

type ChatFixture = {
  readonly stopTokens: number[];
  readonly chat: {
    readonly name: string;
    readonly messages: ChatMessage[];
    readonly ids: number[];
  }[];
};

const fixture = JSON.parse(
  await readTextFile(
    new URL("../../models/tests/fixtures/gemma-text/gemma4-chat.json", import.meta.url),
  ),
) as ChatFixture;

const caseOf = (name: string) => {
  const found = fixture.chat.find((row) => row.name === name);
  assert(found !== undefined, `the fixture has no chat case '${name}'`);
  return found;
};

const AVAILABLE = mirrorAvailable(MIRROR_DIR);

if (!AVAILABLE) {
  console.warn(
    `[karume] the distribution mirror models/karume-gemma4/ is missing, so the Gemma 4 E2B chat ` +
      `acceptance is SKIPPED. Build it: ${GENERATE_COMMAND}`,
  );
}

const openPipeline = async (options: PipelineOptions = {}): Promise<Pipeline> =>
  await Pipeline.fromAssets(
    {
      ...(await gemma4MirrorAssets(MIRROR_DIR, BF16_SEAT)),
      // The declaration uses **the values this check pinned**, not the distribution's (even if
      // the chunk length, capacity or rope defaults move on the mirror side, it runs in the same
      // shape as when the goldens were taken).
      config: {
        chunkLength: CHUNK_LENGTH,
        maxChunkLength: MAX_CHUNK_LENGTH,
        maxPosition: MAX_POSITION,
        capacity: CAPACITY,
        rope: ROPE,
      },
    },
    {
      // The budget is derived from the index (= the whole PLE resident → no re-read even for a
      // conversation that spans ranges). Written as a constant, it would mean a different block
      // count once the block width changes with the asset generation — the helper's doc.
      maxResidentPleBytes: await allResidentPleBytesOfMirror(MIRROR_DIR, BF16_SEAT),
      ...options,
    },
  );

describe.skipIf(!AVAILABLE || !GPU_AVAILABLE)(
  "Gemma 4 E2B chat acceptance: string in → string out, streaming matches batch (real GPU)",
  () => {
    // Assigned in beforeAll; the its below run sequentially against this one pipeline.
    let pipeline: Pipeline;
    /** The stream issued **before** dispose and drained **after** it (used by the lifetime check below). */
    let issuedBeforeDispose: ChatStream | undefined;
    /** The second-turn reference ⑥ took (⑦ builds the same conversation in a session and compares). */
    let followUpAnswer: string | undefined;

    beforeAll(async () => {
      pipeline = await openPipeline();
    });
    // dispose is idempotent; this covers runs where the "after dispose" suite below never started
    // (a failing beforeAll, or a name filter that selects only the its before it).
    afterAll(async () => {
      if (pipeline !== undefined) await pipeline.dispose();
    });

    it("① chat at temperature 0 runs to completion on the real weights and matches the pinned greedy golden", async () => {
      for (const { fixture: name, maxNewTokens, expected, stop } of CASES) {
        const { messages } = caseOf(name);
        const started = performance.now();
        const stream = pipeline.chat(messages, { maxNewTokens });
        const parts: string[] = [];
        for await (const chunk of stream) parts.push(chunk);
        const stopped = await stream.done;

        console.log(
          `[e2e] gemma4 chat ${name}: ${JSON.stringify(parts.join(""))} / ` +
            `${JSON.stringify(stopped)} / ${(performance.now() - started).toFixed(0)}ms`,
        );
        assertEquals(parts.join(""), expected, `${name}: temperature-0 output`);
        assertEquals(stopped, stop, `${name}: stop reason`);
        // Chunks are "only what is settled", so no empty string flows (no fake iteration).
        assertEquals(parts.filter((part) => part === "").length, 0, `${name}: empty chunks`);
      }
    });

    it("② the chat id sequence of the assets the pipeline holds matches the fixture", () => {
      for (const { fixture: name } of CASES) {
        const { messages, ids } = caseOf(name);
        assertEquals(chatPrompt(pipeline.tokenizer, messages), ids, name);
      }
    });

    it("③ the concatenation of streamed chunks = the batch decode of the same conversation's tokens", async () => {
      for (const { fixture: name, maxNewTokens, expected, stop } of CASES) {
        const { ids } = caseOf(name);
        // The low-level surface is one conversation = one sequence (no context carried across).
        const sequence = await pipeline.sequence();
        try {
          const stream = sequence.generate({ prompt: ids, maxNewTokens });
          const tokens: number[] = [];
          for await (const event of stream) if (event.kind === "token") tokens.push(event.id);
          assertEquals(await stream.done, stop, `${name}: stop reason on the low-level surface`);
          assertEquals(
            pipeline.tokenizer.decode(tokens),
            expected,
            `${name}: batch decode (splits if incremental decoding drops a byte run)`,
          );
        } finally {
          await sequence.dispose();
        }
      }
    });

    it("④ the stop set matches the upstream generation_config.json declaration", () => {
      const ascending = (ids: readonly number[]): number[] => [...ids].sort((a, b) => a - b);
      assertEquals(ascending(pipeline.program.stopTokens), ascending(STOP_TOKENS), "program");
      assertEquals(ascending(fixture.stopTokens), ascending(STOP_TOKENS), "fixture");
    });

    it("⑤ an out-of-scope conversation fails before touching the GPU (a synchronous throw)", () => {
      assertThrows(
        () =>
          pipeline.chat([{ role: "user", content: "hi", tools: [] }] as unknown as ChatMessage[], {
            maxNewTokens: 4,
          }),
        Error,
        "out-of-scope fields",
      );
      assertThrows(
        () =>
          pipeline.chat([{ role: "tool", content: "…" }] as unknown as ChatMessage[], {
            maxNewTokens: 4,
          }),
        Error,
        "role",
      );
    });

    it("⑥ incremental multi-turn rendering matches chat()'s whole rendering verbatim", async () => {
      const { fixture: name, maxNewTokens } = CASES[0];
      const first = caseOf(name).messages;

      /** Drains one turn on the high-level surface (re-renders the whole conversation each turn). */
      const viaChat = async (messages: readonly ChatMessage[], limit: number): Promise<string> => {
        const parts: string[] = [];
        for await (const chunk of pipeline.chat(messages, { maxNewTokens: limit })) {
          parts.push(chunk);
        }
        return parts.join("");
      };

      const answer1 = await viaChat(first, maxNewTokens);
      // The second-turn reference "puts the first reply into the conversation and re-renders it all".
      const grown: ChatMessage[] = [...first, { role: "assistant", content: answer1 }, FOLLOW_UP];
      const answer2 = await viaChat(grown, FOLLOW_UP_TOKENS);
      followUpAnswer = answer2;

      // Low-level surface: one conversation = one sequence. The second turn feeds **only the
      // delta**, and `pendingToken` prepends the `<turn|>` that closes the previous turn (the
      // turn-local contract — ADR 0083 decision 4 / 0084 decision 5).
      const sequence = await pipeline.sequence();
      try {
        const turn = async (prompt: readonly number[], limit: number) => {
          const stream = sequence.generate({ prompt, maxNewTokens: limit });
          const ids: number[] = [];
          for await (const event of stream) if (event.kind === "token") ids.push(event.id);
          return { text: pipeline.tokenizer.decode(ids), stop: await stream.done };
        };

        const prompt1 = chatPrompt(pipeline.tokenizer, first);
        const first1 = await turn(prompt1, maxNewTokens);
        assertEquals(first1.text, answer1, "turn 1 (the whole rendering is the same path as chat)");
        // `used` is "the logical positions the conversation occupies" — explained by prompt +
        // generation (including the stop token).
        assertEquals(sequence.used, prompt1.length + first1.stop.tokens, "used after turn 1");

        const delta = chatTurn(pipeline.tokenizer, FOLLOW_UP);
        const second = await turn(delta, FOLLOW_UP_TOKENS);
        console.log(
          `[e2e] gemma4 incremental turn: ${JSON.stringify(second.text)} / ` +
            `delta ${delta.length} tokens / used ${sequence.used}`,
        );
        assertEquals(
          second.text,
          answer2,
          "turn 2 does not match chat()'s whole rendering verbatim (the delta rendering or the " +
            "pendingToken concatenation makes it a different conversation)",
        );
        assertEquals(
          sequence.used,
          prompt1.length + first1.stop.tokens + delta.length + second.stop.tokens,
          "used after turn 2",
        );
      } finally {
        await sequence.dispose();
      }
    });

    it("⑦ two turns of ChatSession give the same conversation as ⑥", async () => {
      const { fixture: name, maxNewTokens, expected } = CASES[0];
      const first = caseOf(name).messages;
      // This case is one user message (the session takes a string, so there is no system and no
      // past turn).
      assertEquals(first.length, 1, `${name}: only a one-message case maps onto send`);
      assert(followUpAnswer !== undefined, "⑥ did not take the second-turn reference");

      await using session = new ChatSession(pipeline, { maxNewTokens });
      // Turn 1 is the golden itself (it starts from the same whole rendering as `chat()`).
      assertEquals(await session.send(first[0].content).text(), expected, "turn 1");
      // Turn 2 is delta rendering + KV continuation. If the condition the session continues on
      // (the previous turn closed with EOS) or the way the history accumulates breaks, it departs
      // from ⑥'s reference here.
      const second = session.send(FOLLOW_UP.content, { maxNewTokens: FOLLOW_UP_TOKENS });
      assertEquals(
        await second.text(),
        followUpAnswer,
        "turn 2 (verbatim match with ⑥'s reference)",
      );
      assertEquals((await second.done).reason, "eos", "turn 2 stop reason");
      assertEquals(
        session.turns,
        [
          first[0],
          { role: "assistant", content: expected },
          FOLLOW_UP,
          { role: "assistant", content: followUpAnswer },
        ],
        'the history accumulates only "the text that was streamed"',
      );
    });

    it("⑧ a request is copied at issue (rewriting it before draining has no effect)", async () => {
      // The MUST of the type doc. If the copy were delayed into the receivers
      // (`GenerationSequence.generate` / `createSampler`), that is the body of an async generator =
      // **the first `next()`**, so a stop set or sampling spec rewritten in the window "between
      // issue and the start of draining" would silently take effect.
      // The observation rewrites them **outside the accepted set** (copied → passes / not copied →
      // fails in the receiver's re-check = a deterministic difference that does not rely on the
      // distribution).
      const { messages } = caseOf(CASES[0].fixture);
      const stopTokens = [7];
      const sampler = { temperature: 0 };
      const stream = pipeline.chat(messages, {
        maxNewTokens: CASES[0].maxNewTokens,
        stopTokens,
        sampler,
      });
      stopTokens.push(Number.MAX_SAFE_INTEGER); // outside the vocabulary (a value `generate`'s re-check rejects)
      sampler.temperature = -1; // outside `createSampler`'s accepted set
      assertEquals(await stream.text(), CASES[0].expected, "the rewrite after issue took effect");
    });

    it("⑨ chat's capacity reaches the sequence (no silent fallback to the default)", async () => {
      // Forgetting to pass it down still works with the default (`program.capacity`), so it
      // **raises no error** — pass a value outside the accepted set and see that it is passed down
      // by "it fails". It fails not at issue but at the first `next()` (`createGenerationSequence`
      // is called inside the body), so this also pins that the failure is not synchronous.
      const { messages } = caseOf(CASES[0].fixture);
      const stream = pipeline.chat(messages, {
        maxNewTokens: 4,
        capacity: CHUNK_LENGTH - 1,
      });
      await assertRejects(
        () => stream.text(),
        Error,
        `capacity ${CHUNK_LENGTH - 1} is below chunkLength ${CHUNK_LENGTH}`,
      );
    });

    it("⑩ estimateSessionMemory takes its defaults from the wiring and fails loudly on the range", () => {
      // The estimate is a pure function (`estimateGraphMemory`) but reads `gpu.limits`, so it
      // needs a real pipeline — it adds no run. That the default arguments use the distribution's
      // declaration (chunkLength / capacity) is bound by "equal to the result of an explicit call"
      // (a regression to another default splits it).
      const implicit = pipeline.estimateSessionMemory();
      const explicit = pipeline.estimateSessionMemory({
        chunkLength: CHUNK_LENGTH,
        capacity: CAPACITY,
      });
      assertEquals(implicit, explicit, "the default arguments do not use the wiring's two numbers");
      // The counterpart that keeps it from being vacuous: changing the capacity moves the state bytes.
      const half = pipeline.estimateSessionMemory({ capacity: CAPACITY / 2 });
      assert(
        half.resident.stateBytes < implicit.resident.stateBytes,
        `halving the capacity does not move the state bytes: ${half.resident.stateBytes}`,
      );
      // MUST: always pass `maxStorageBufferBindingSize` (the in-node temporaries of states-form
      // attention get their row-block count from this limit alone — without it the estimator
      // fails loudly).
      assert(implicit.peakAccountedBytes > 0, "the estimate claims 0 bytes");
      // The retention budget of the slot backing (ADR 0095) is the runtime default when the
      // pipeline options omit it.
      assertEquals(
        implicit.planBackingBudgetBytes,
        DEFAULT_PLAN_BACKING_BUDGET_BYTES,
        "the budget default is not on the report (the pipeline passes another value)",
      );

      // Three failure paths (each fails before the Session is rebuilt).
      assertThrows(
        () => pipeline.estimateSessionMemory({ capacity: CHUNK_LENGTH - 1 }),
        Error,
        `capacity ${CHUNK_LENGTH - 1} is below chunkLength ${CHUNK_LENGTH}`,
      );
      assertThrows(
        () => pipeline.estimateSessionMemory({ capacity: MAX_POSITION + 1 }),
        Error,
        `capacity ${MAX_POSITION + 1} exceeds maxPosition ${MAX_POSITION}`,
      );
      assertThrows(
        () => pipeline.estimateSessionMemory({ chunkLength: MAX_CHUNK_LENGTH + 1 }),
        Error,
        `chunkLength ${MAX_CHUNK_LENGTH + 1} exceeds the distribution's declared maxChunkLength`,
      );
      // A capacity that is not a safe integer is caught by the "not an integer ≥ 1" check (kept
      // apart from the ordering diagnostics — the same three as `createGenerationSequence` /
      // `ChatSession`, leaving no hole in the range).
      assertThrows(
        () => pipeline.estimateSessionMemory({ capacity: 1.5 }),
        Error,
        "capacity 1.5 is not a positive integer",
      );
    });

    // The its below observe the pipeline after dispose; this suite's beforeAll runs after every
    // it above (Vitest runs a file's tasks in definition order).
    describe("after dispose", () => {
      beforeAll(async () => {
        try {
          // The body of an async generator does not run until the first `next()`, so a check
          // only at issue misses "issue → dispose → start draining". Carry it across dispose
          // without draining.
          issuedBeforeDispose = pipeline.chat([{ role: "user", content: "hi" }], {
            maxNewTokens: 4,
          });
        } finally {
          await pipeline.dispose();
        }
      });

      it("a disposed pipeline accepts no generation", async () => {
        assertThrows(
          () => pipeline.chat([{ role: "user", content: "hi" }], { maxNewTokens: 4 }),
          Error,
          "after dispose",
        );
        await assertRejects(() => pipeline.sequence(), Error, "after dispose");

        // An issued but not yet iterated stream also closes with **the same pipeline wording as
        // at issue** (if it turned into the runtime's Session wording, the caller could not read
        // the real cause, that it disposed the pipeline itself).
        const issued = issuedBeforeDispose;
        assert(issued !== undefined, "no stream was issued before dispose");
        await assertRejects(
          async () => {
            for await (const _chunk of issued) {
              /* none arrives */
            }
          },
          Error,
          "gemma.Pipeline: cannot generate after dispose",
        );
      });

      it("the public program surface is only frozen numbers and stays readable after dispose", () => {
        // NOTE: that the host cache of the PLE (an asset of the `model` container) is released
        // on dispose belongs to the unit check of `gemma4_ple_test.ts` (`derivedInputs.derive`
        // cannot be reached from the public surface = as intended when the surface was narrowed).
        // What this sees is the property of the narrowed surface — the numbers the distribution
        // declared read as they are, and writes on the consumer side do not reach the stop set
        // of the generation loop.
        const program = pipeline.program;
        assertEquals(program.chunkLength, CHUNK_LENGTH);
        assertEquals(program.capacity, CAPACITY);
        assertEquals(program.maxPosition, MAX_POSITION);
        assertEquals(Object.isFrozen(program), true, "program is not frozen");
        assertEquals(Object.isFrozen(program.stopTokens), true, "stopTokens is not frozen");
        // A frozen copy, so even trying to empty the stop set fails (it does not silently stop
        // stopping at EOS).
        assertThrows(() => {
          (program.stopTokens as number[]).length = 0;
        }, TypeError);
      });
    });
  },
);

/**
 * **Pass-through** of the slot backing's retention budget (ADR 0095) — the value of the pipeline
 * options takes effect as is as the premise of `estimateSessionMemory`.
 *
 * ⑩ covers "the runtime default when omitted"; this one covers "the value when explicit". The
 * Session side (the pass-through to `createSession`) has no observation point here, so this sees
 * only the estimate side (passing the same value to both is the pipeline's one implementation,
 * and the held value exists only once, in `Gemma4State`).
 */
test.skipIf(!AVAILABLE || !GPU_AVAILABLE)(
  "planBackingBudgetBytes passes through from the pipeline options to the estimate (real GPU)",
  async () => {
    // Pick a value that differs from both the default (256 MiB) and the largest scenario (falling
    // back to either default fails).
    const budget = 4 * 1024 * 1024 * 1024;
    const reports: AdmissionReport[] = [];
    for (const diagnostics of [false, true]) {
      const pipeline = await openPipeline({
        planBackingBudgetBytes: budget,
        ...(diagnostics ? { onRunDiagnostics: () => {} } : {}),
      });
      try {
        const report = pipeline.estimateSessionMemory();
        reports.push(report);
        assertEquals(
          report.planBackingBudgetBytes,
          budget,
          "the budget does not reach the estimate",
        );
        const scenario = Math.max(
          ...report.scenarios.map((one) => one.ioBytes + one.workspaceBytes),
        );
        // The counterpart that keeps the check from passing vacuously: this budget is larger than
        // every scenario (= the budget side of the max wins).
        assert(
          scenario < budget,
          `the largest scenario ${scenario} is not below the budget ${budget}`,
        );
        if (diagnostics) {
          assertEquals(
            report.auxiliaryBytes,
            undefined,
            "no auxiliary output is allocated under diagnostics",
          );
        } else {
          assert(
            report.auxiliaryBytes !== undefined &&
              report.auxiliaryBytes >= pipeline.program.vocabSize * 4 + 8,
            "the auxiliary output's resident logits, values and indices are not in the extra accounting",
          );
        }
        assertEquals(
          report.peakAccountedBytes,
          report.resident.weights.totalBytes +
            report.resident.stateBytes +
            budget +
            (report.auxiliaryBytes ?? 0),
          "the peak does not account for the budget and the auxiliary resources",
        );
      } finally {
        await pipeline.dispose();
      }
    }
    const [greedy, observed] = reports;
    assertEquals(
      greedy.resident,
      observed.resident,
      "auxiliary resources are not mixed into the weights or the KV state",
    );
    assertEquals(greedy.scenarios, observed.scenarios, "the target's scenarios are the same");
    assertEquals(
      greedy.peakAccountedBytes - observed.peakAccountedBytes,
      greedy.auxiliaryBytes,
      "the difference between the paths equals the extra accounting exactly",
    );
  },
);

/**
 * **Pass-through (Session side)** of the slot backing's retention budget (ADR 0095) — the value of
 * the pipeline options reaches `createSession` and shows up as is in the run diagnostics of
 * generation.
 *
 * The check above sees only the estimate side (the numbers on the report), not whether it takes
 * effect on the Session. This one reads `planBacking` of each run directly from the observation
 * hook ({@link PipelineOptions.onRunDiagnostics}). It looks at a pair of two:
 *
 * ① **Budget 0** (the former capacity of 1): every run retains exactly one, and the switch from
 *    prefill → decode rebuilds (`buildCount` goes up by 1)
 * ② **The default budget**: **two** are retained, the prefill bucket shape and the decode shape,
 *    and from the second decode on nothing is rebuilt (exactly the switching cost per generated
 *    turn that went away)
 *
 * MUST: observe **the second turn**. A run "builds a backing only on a hit run" (a one-off run
 * does not pay for slot memory — ADR 0095), so each run of the first turn ends at deriving the
 * plan (a miss run), and neither retention nor switching shows at the observation point. The
 * first turn runs the same conversation only to have the two keys derived.
 */
type ObservedRun = {
  readonly kind: RunPhase["kind"];
  readonly retainedCount: number;
  readonly buildCount: number;
};

/**
 * The number of generated tokens of the observed turn (stops before EOS's 9 = the stop reason is
 * `max-tokens`).
 */
const OBSERVED_NEW_TOKENS = 4;

/** Runs the same conversation for two turns and picks up only the runs of **the second turn**. */
const observeSecondTurn = async (options: PipelineOptions): Promise<ObservedRun[]> => {
  const { messages } = caseOf("single-user");
  const runs: ObservedRun[] = [];
  const pipeline = await openPipeline({
    ...options,
    onRunDiagnostics: (diagnostics: SessionDiagnostics, phase: RunPhase) => {
      runs.push({
        kind: phase.kind,
        retainedCount: diagnostics.planBacking.retainedCount,
        buildCount: diagnostics.planBacking.buildCount,
      });
    },
  });
  try {
    for (const turn of [0, 1]) {
      const stream = pipeline.chat(messages, { maxNewTokens: OBSERVED_NEW_TOKENS });
      for await (const _chunk of stream) {
        /* the chunks are not used (the point is to run) */
      }
      assertEquals((await stream.done).reason, "max-tokens", `turn ${turn} stopped early`);
      // Turn 1 is only for deriving (miss runs). Drop it and keep only turn 2.
      if (turn === 0) runs.length = 0;
    }
    return runs;
  } finally {
    await pipeline.dispose();
  }
};

test.skipIf(!AVAILABLE || !GPU_AVAILABLE)(
  "planBackingBudgetBytes passes through to the Session (budget 0 retains 1, the default 2; real GPU)",
  async () => {
    /** Pins the shape of the observed sequence first (1 prefill + several decodes) — so the check does not pass vacuously. */
    const assertShape = (runs: readonly ObservedRun[], label: string): ObservedRun[] => {
      assertEquals(
        runs.filter((run) => run.kind === "prefill").length,
        1,
        `${label}: not exactly one prefill`,
      );
      assertEquals(runs[0]?.kind, "prefill", `${label}: the first run is not a prefill run`);
      const decodes = runs.filter((run) => run.kind === "decode");
      assert(decodes.length >= 2, `${label}: only ${decodes.length} decode runs`);
      return decodes;
    };

    // ① Budget 0 — always one retained, rebuilt at every switch.
    const zero = await observeSecondTurn({ planBackingBudgetBytes: 0 });
    const zeroDecodes = assertShape(zero, "budget 0");
    for (const run of zero) {
      assertEquals(
        run.retainedCount,
        1,
        `budget 0 does not reach the Session: ${JSON.stringify(zero)}`,
      );
    }
    assertEquals(
      zeroDecodes[0].buildCount,
      zero[0].buildCount + 1,
      `with budget 0 the prefill → decode switch does not rebuild: ${JSON.stringify(zero)}`,
    );
    for (const run of zeroDecodes.slice(1)) {
      assertEquals(
        run.buildCount,
        zeroDecodes[0].buildCount,
        `consecutive runs of the same decode shape rebuild: ${JSON.stringify(zero)}`,
      );
    }

    // ② The default budget — two retained, the prefill bucket shape and the decode shape; no rebuild in turn 2.
    const budgeted = await observeSecondTurn({});
    const budgetedDecodes = assertShape(budgeted, "default budget");
    for (const run of budgeted) {
      assertEquals(
        run.retainedCount,
        2,
        `the default budget does not retain the two shapes (it retires one at every switch): ${JSON.stringify(
          budgeted,
        )}`,
      );
    }
    for (const run of budgetedDecodes) {
      assertEquals(
        run.buildCount,
        budgeted[0].buildCount,
        `the default budget rebuilds the backing on decode: ${JSON.stringify(budgeted)}`,
      );
    }
  },
);
