/**
 * Acceptance of the fixed QAT's public entry points and their wiring to the shared chat layer. Not
 * a general quality check (ADR 0097).
 */
import { afterAll, beforeAll, describe, it, test } from "vite-plus/test";
import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import { gemma4MirrorAssets, mirrorAvailable } from "../../models/tests/helpers/gemma-mirror.ts";
import { nodeDirectory } from "@karume/hub/node";
import {
  type Assets,
  ChatSession,
  type ChatSessionHost,
  Pipeline,
} from "../../models/gemma4-qat-mobile.ts";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";

const root = new URL("../../../models/karume-gemma4-qat-mobile/", import.meta.url);
// A machine without the mirror, or with an old major left in place, SKIPs explicitly (the
// helper's MUST — ADR 0109 decision 9).
const available = mirrorAvailable(root);
if (!available) {
  console.warn(
    "[karume] models/karume-gemma4-qat-mobile/ is absent (or not karume/5), so the mobile QAT real-weight checks SKIP. Assemble it with dist.py --pipeline gemma4-qat-mobile in tools/export-recipes.",
  );
}
for (const model of ["e2b", "e4b"] as const) {
  describe.skipIf(!available || !GPU_AVAILABLE)(
    `gemma4-qat-mobile ${model}: public entry, KV continuation, recovery after abort, release (real GPU)`,
    () => {
      let abortOnDecode: AbortController | undefined;
      const states: { contextCount: number; residentBytes: number }[] = [];
      let pipeline: Pipeline;
      let sequenceCount = 0;
      let host: ChatSessionHost;
      beforeAll(async () => {
        pipeline = await Pipeline.fromPretrained(nodeDirectory(root), {
          model,
          maxResidentPleBytes: 0,
          onRunDiagnostics: (d, phase) => {
            states.push(d.stateBacking);
            if (phase.kind === "decode" && abortOnDecode) {
              const controller = abortOnDecode;
              abortOnDecode = undefined;
              controller.abort(new Error("QAT test abort"));
            }
          },
        });
        host = {
          tokenizer: pipeline.tokenizer,
          program: pipeline.program,
          defaultSampler: pipeline.defaultSampler,
          sequence: async () => {
            sequenceCount++;
            return await pipeline.sequence();
          },
        };
      });
      // Teardown, preceded by the check that closed the former test body after its steps: a
      // disposed pipeline refuses new sequences. The second dispose is the former finally block.
      afterAll(async () => {
        if (pipeline === undefined) return;
        try {
          await pipeline.dispose();
          await assertRejects(() => pipeline.sequence(), Error, "dispose");
        } finally {
          await pipeline.dispose();
        }
      });

      it("after EOS it continues the KV and gives the same reply as re-rendering the whole history", async () => {
        const chat = new ChatSession(host, {
          maxNewTokens: 8,
          sampler: { temperature: 0 },
        });
        try {
          const first = chat.send("What is 17 + 28? Reply with only the number.");
          // The official CPU run of the same short prompt also gives 45/EOS (research's
          // e2b/e4b-cpu-broad-reference).
          assertEquals(await first.text(), "45");
          assertEquals((await first.done).reason, "eos");
          const second = chat.send("What is 2 + 2? Reply with only the number.");
          const text = await second.text();
          assertEquals(text, "4");
          assertEquals((await second.done).reason, "eos");
          assertEquals(sequenceCount, 1);
          assertEquals(chat.turns.length, 4);
          const messages = chat.turns.slice(0, -1);
          await chat.dispose();
          assertEquals(
            await pipeline.chat(messages, { maxNewTokens: 8, sampler: { temperature: 0 } }).text(),
            text,
          );
        } finally {
          await chat.dispose();
        }
      });
      it("still generates after a decode abort and an early exit from iteration", async () => {
        const chat = new ChatSession(host, {
          maxNewTokens: 32,
          sampler: { temperature: 0 },
        });
        try {
          const controller = new AbortController();
          abortOnDecode = controller;
          const stream = chat.send("Write a short story about a fox.", {
            signal: controller.signal,
          });
          await assertRejects(() => stream.text(), Error, "QAT test abort");
          assertEquals((await stream.done).reason, "aborted");
          assertEquals(
            await chat
              .send("What is 2 + 2? Reply with only the number.", { maxNewTokens: 8 })
              .text(),
            "4",
          );
        } finally {
          await chat.dispose();
        }
        const early = pipeline.chat(
          [
            {
              role: "user",
              content: "Write a short story about a fox.",
            },
          ],
          { maxNewTokens: 32, sampler: { temperature: 0 } },
        );
        for await (const chunk of early) {
          assert(chunk.length > 0);
          break;
        }
        await early.done;
        assertEquals(
          await pipeline
            .chat(
              [
                {
                  role: "user",
                  content: "What is 2 + 2? Reply with only the number.",
                },
              ],
              { maxNewTokens: 8, sampler: { temperature: 0 } },
            )
            .text(),
          "4",
        );
      });
      it("a capacity override reaches the estimate, and values outside the declaration are rejected", () => {
        // The numbers are not literals (re-baking the distribution moves chunkLength / capacity).
        const { chunkLength, capacity, maxPosition } = pipeline.program;
        const smaller = Math.floor(capacity / 2);
        assert(smaller >= chunkLength, `capacity ${capacity} cannot be halved`);
        const implicit = pipeline.estimateSessionMemory();
        assertEquals(
          implicit,
          pipeline.estimateSessionMemory({ chunkLength, capacity }),
          "the default arguments do not use the distribution's declaration",
        );
        // The counterpart that shows this is not a tautology: less capacity means fewer state
        // bytes.
        assert(
          pipeline.estimateSessionMemory({ capacity: smaller }).resident.stateBytes <
            implicit.resident.stateBytes,
          "reducing the capacity does not move the state bytes",
        );
        assertThrows(
          () => pipeline.estimateSessionMemory({ capacity: chunkLength - 1 }),
          Error,
          `capacity ${chunkLength - 1} is below chunkLength ${chunkLength}`,
        );
        assertThrows(
          () => pipeline.estimateSessionMemory({ capacity: maxPosition + 1 }),
          Error,
          `capacity ${maxPosition + 1} exceeds maxPosition ${maxPosition}`,
        );
      });
      it("chat's capacity reaches the sequence (no silent fallback to the default)", async () => {
        // Forgetting to pass it down still works with the default, so it **does not throw** — the
        // check passes a value outside the accepted set and sees it fail, which shows it reached
        // the sequence. The failure comes at the first `next()`, not at issue.
        const { chunkLength } = pipeline.program;
        const stream = pipeline.chat([{ role: "user", content: "hi" }], {
          maxNewTokens: 4,
          capacity: chunkLength - 1,
        });
        await assertRejects(
          () => stream.text(),
          Error,
          `capacity ${chunkLength - 1} is below chunkLength ${chunkLength}`,
        );
      });
      it("re-creating a conversation of the same capacity does not accumulate state", () => {
        assert(states.length > 0);
        // contextCount is a running total, not the live count. The release is checked through the
        // residentBytes of the same capacity.
        assert(states.at(-1)!.contextCount > states[0].contextCount);
        assertEquals(new Set(states.map((s) => s.residentBytes)).size, 1);
      });
    },
  );
}

/**
 * The surface that builds from already-fetched bytes (the other public entry point of ADR 0097
 * addendum 6).
 *
 * Opening it at the same time as the `fromPretrained` Session would keep the weights resident
 * twice, so it is a separate test run in order (Deno's tests run serially by default).
 */
const qatAssets = (model: "e2b" | "e4b"): Promise<Assets> =>
  // The PLE lives inside the `model` container too (ADR 0109 decision 4), so only the parts and
  // the tokenizer are passed.
  gemma4MirrorAssets(root, { model });

for (const model of ["e2b", "e4b"] as const) {
  test.skipIf(!available || !GPU_AVAILABLE)(
    `gemma4-qat-mobile ${model}: fromAssets goes through the same build and reply (real GPU)`,
    async () => {
      const pipeline = await Pipeline.fromAssets(await qatAssets(model), {
        maxResidentPleBytes: 0,
      });
      try {
        // On this surface the model name is identified from the graph and the PLE layout, not
        // the manifest, so the check is that the same short prompt gives the same reply as
        // fromPretrained.
        assertEquals(
          await pipeline
            .chat(
              [
                {
                  role: "user",
                  content: "What is 17 + 28? Reply with only the number.",
                },
              ],
              { maxNewTokens: 8, sampler: { temperature: 0 } },
            )
            .text(),
          "45",
        );
      } finally {
        await pipeline.dispose();
      }
    },
  );
}
