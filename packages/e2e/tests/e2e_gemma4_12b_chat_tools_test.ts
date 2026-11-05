// Real-weight Gemma 4 12B function calling and thinking through `ChatSession` (ADR 0084 addendum
// 2026-11-04). The rendering, the output parse and the KV identity are pinned GPU-free by
// `models/tests/gemma4_chat_*`; this file checks that the real model, greedy, takes the path end
// to end:
//
// ① a tool call round trip: a session with one declared tool turns a question into a call of that
//    tool with the argument the question names (`"tool-call"`), and `sendToolResults` continues the
//    same model turn into an answer that uses the returned value (`"turn"`)
// ② thinking: a session with `enableThinking` closes its turn with a non-empty thought and an
//    answer, and the answer is the right one
//
// The assertions are structural (the call's name and argument, a value in the answer), not a pinned
// text: the text is free to move with the numerics, the structure is what the chat layer promises.
//
// ## Assets and GPU memory
//
// The q4_0 QAT line's mirror (`models/karume-gemma4-qat/`, model `12b`, quant `i4`); without it, or
// without a GPU, every case SKIPs explicitly. One 12B pipeline at the default capacity (about 9 GiB —
// `e2e_gemma4_12b_test.ts`'s account), built once for both cases.

import { readTextFileSync } from "@karume/test-support/fs";
import { afterAll, beforeAll, describe, it } from "vite-plus/test";
import { assert, assertEquals, assertStringIncludes } from "@std/assert";
import { type Manifest, MANIFEST_FILENAME, parseManifest } from "@karume/hub";
import { ChatSession, type ChatTool, Pipeline } from "../../models/gemma.ts";
import { mirrorDirectory, mirrorPresent, mirrorRoot } from "../../models/tests/helpers/mirror.ts";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";

const MIRROR = mirrorRoot("karume-gemma4-qat");
const CHOICE = { model: "12b", quant: "i4" } as const;
const GREEDY = { temperature: 0 } as const;

const MANIFEST: Manifest | undefined = mirrorPresent(MIRROR)
  ? parseManifest(readTextFileSync(new URL(MANIFEST_FILENAME, MIRROR)))
  : undefined;
const MODEL_PRESENT = MANIFEST !== undefined && Object.hasOwn(MANIFEST.models, CHOICE.model);

const WEATHER: ChatTool = {
  type: "function",
  function: {
    name: "get_weather",
    description: "Gets the current weather in a city.",
    parameters: {
      type: "object",
      properties: { city: { type: "string", description: "The city name, e.g. Tokyo." } },
      required: ["city"],
    },
  },
};
/** A reading no model would guess, so the answer can only have it from the tool result. */
const WEATHER_RESULT = '{"temperature_c": 13, "condition": "light rain"}';

describe.skipIf(!MODEL_PRESENT || !GPU_AVAILABLE)(
  "Gemma 4 12B function calling and thinking through ChatSession (real GPU)",
  () => {
    let pipeline: Pipeline | undefined;
    beforeAll(async () => {
      pipeline = await Pipeline.fromPretrained(mirrorDirectory(MIRROR), CHOICE);
    });
    afterAll(async () => {
      await pipeline?.dispose();
    });

    it("① a question turns into a call of the declared tool, and its result into the answer", async () => {
      assert(pipeline !== undefined, "the pipeline was not built in beforeAll");
      await using session = new ChatSession(pipeline, {
        maxNewTokens: 256,
        sampler: GREEDY,
        tools: [WEATHER],
      });
      const asking = session.send("What is the weather like in Kyoto right now?");
      const callText = await asking.text();
      const call = await asking.reply;
      console.log(`[e2e] gemma4 12b tool call: ${JSON.stringify(callText)}`);
      assert(call.end === "tool-call", `the turn ended '${call.end}', not on a tool call`);
      assertEquals(call.message.tool_calls.length, 1, "one call");
      const [called] = call.message.tool_calls;
      assertEquals(called.function.name, "get_weather");
      assertEquals(called.function.arguments, { city: "Kyoto" });

      const answering = session.sendToolResults([
        { role: "tool", tool_call_id: called.id, content: WEATHER_RESULT },
      ]);
      const answerText = await answering.text();
      const answer = await answering.reply;
      console.log(`[e2e] gemma4 12b answer: ${JSON.stringify(answerText)}`);
      assert(answer.end === "turn", `the continued turn ended '${answer.end}'`);
      // 12B opens the continued turn with an empty thought channel even with thinking off; the
      // parser takes it out of the content.
      const content = answer.message.content ?? "";
      assert(!content.includes("<|channel>"), `the content keeps the channel: ${content}`);
      assertStringIncludes(content, "13");
    });

    it("② with thinking on, the turn closes with a thought and the right answer", async () => {
      assert(pipeline !== undefined, "the pipeline was not built in beforeAll");
      await using session = new ChatSession(pipeline, {
        maxNewTokens: 2048,
        sampler: GREEDY,
        enableThinking: true,
      });
      const turn = session.send("What is 17 multiplied by 23? Answer with the number.");
      await turn.text();
      const reply = await turn.reply;
      assert(reply.end === "turn", `the turn ended '${reply.end}'`);
      const thinking = reply.message.thinking ?? "";
      console.log(
        `[e2e] gemma4 12b thinking: ${thinking.length} chars / answer ${JSON.stringify(reply.message.content)}`,
      );
      assert(thinking.trim() !== "", "the thought is empty");
      assertStringIncludes(reply.message.content ?? "", "391");
    });
  },
);
