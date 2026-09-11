/** 固定 QAT の公開入口と共通会話層の結線を検収する。品質全般の検査ではない（ADR 0097）。 */
import { assert, assertEquals, assertRejects } from "@std/assert";
import { denoDirectory } from "@karume/hub/deno";
import { Gemma4ChatSession, Gemma4QatPipeline } from "../gemma4-qat.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const root = new URL("../../../models/karume-gemma4-qat/", import.meta.url);
const exists = (): boolean => {
  try {
    return Deno.statSync(new URL("karume.json", root)).isFile;
  } catch (error) {
    if (error instanceof Deno.errors.NotFound) return false;
    throw error;
  }
};
const available = exists();
if (!available) {
  console.warn(
    "[karume] models/karume-gemma4-qat/ が無いため QAT 実重み検査を SKIP。tools/export-recipes の dist.py --pipeline gemma4-qat で作成する。",
  );
}
for (const model of ["e2b", "e4b"] as const) {
  Deno.test({
    name: `gemma4-qat ${model}: 公開入口・KV継続・中断復帰・解放（実GPU）`,
    ignore: !available || !GPU_AVAILABLE,
    fn: async (t) => {
      let abortOnDecode: AbortController | undefined;
      const states: { contextCount: number; residentBytes: number }[] = [];
      const pipeline = await Gemma4QatPipeline.fromPretrained(denoDirectory(root), {
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
      try {
        let sequenceCount = 0;
        const host = {
          tokenizer: pipeline.tokenizer,
          program: pipeline.program,
          defaultSampler: pipeline.defaultSampler,
          sequence: async () => {
            sequenceCount++;
            return await pipeline.sequence();
          },
        };
        await t.step("EOS後はKVを継続し、全履歴の再描画と同じ応答を出す", async () => {
          const chat = new Gemma4ChatSession(host, {
            maxNewTokens: 8,
            sampler: { temperature: 0 },
          });
          try {
            const first = chat.send("What is 17 + 28? Reply with only the number.");
            // 公式CPUの同じ短文でも45/EOS（researchのe2b/e4b-cpu-broad-reference）。
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
              await pipeline.chat(messages, { maxNewTokens: 8, sampler: { temperature: 0 } })
                .text(),
              text,
            );
          } finally {
            await chat.dispose();
          }
        });
        await t.step("decode中断と反復の早期終了の後も生成できる", async () => {
          const chat = new Gemma4ChatSession(host, {
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
              await chat.send("What is 2 + 2? Reply with only the number.", { maxNewTokens: 8 })
                .text(),
              "4",
            );
          } finally {
            await chat.dispose();
          }
          const early = pipeline.chat([{
            role: "user",
            content: "Write a short story about a fox.",
          }], { maxNewTokens: 32, sampler: { temperature: 0 } });
          for await (const chunk of early) {
            assert(chunk.length > 0);
            break;
          }
          await early.done;
          assertEquals(
            await pipeline.chat([{
              role: "user",
              content: "What is 2 + 2? Reply with only the number.",
            }], { maxNewTokens: 8, sampler: { temperature: 0 } }).text(),
            "4",
          );
        });
        await t.step("同じ容量の会話を作り直してもstateが累積しない", () => {
          assert(states.length > 0);
          // contextCount は現存数でなく累計。解放は同容量の residentBytes で検査する。
          assert(states.at(-1)!.contextCount > states[0].contextCount);
          assertEquals(new Set(states.map((s) => s.residentBytes)).size, 1);
        });
        await pipeline.dispose();
        await assertRejects(() => pipeline.sequence(), Error, "dispose");
      } finally {
        await pipeline.dispose();
      }
    },
  });
}
