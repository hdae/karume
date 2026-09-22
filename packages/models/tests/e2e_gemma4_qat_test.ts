/** 固定 QAT の公開入口と共通会話層の結線を検収する。品質全般の検査ではない（ADR 0097）。 */
import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import { gemma4MirrorAssets, mirrorAvailable } from "./helpers/gemma-mirror.ts";
import { denoDirectory } from "@karume/hub/deno";
import { type Gemma4Assets, Gemma4ChatSession, Gemma4QatPipeline } from "../gemma4-qat.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const root = new URL("../../../models/karume-gemma4-qat/", import.meta.url);
// 無い機も旧 major が残っている機も明示 SKIP（helper の MUST — ADR 0109 決定 9）。
const available = mirrorAvailable(root);
if (!available) {
  console.warn(
    "[karume] models/karume-gemma4-qat/ が無い（か karume/5 でない）ため QAT 実重み検査を SKIP。tools/export-recipes の dist.py --pipeline gemma4-qat で作成する。",
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
        await t.step("capacityの上書きが見積りへ降り、宣言の外は拒否する", () => {
          // 数はリテラルで持たない（配布形を焼き直すと chunkLength / capacity が動く）。
          const { chunkLength, capacity, maxPosition } = pipeline.program;
          const smaller = Math.floor(capacity / 2);
          assert(smaller >= chunkLength, `capacity ${capacity} が半分に割れない`);
          const implicit = pipeline.estimateSessionMemory();
          assertEquals(
            implicit,
            pipeline.estimateSessionMemory({ chunkLength, capacity }),
            "既定引数が配布形の宣言を使っていない",
          );
          // 恒真でないことの対: 容量を減らせば state のバイト数は減る。
          assert(
            pipeline.estimateSessionMemory({ capacity: smaller }).resident.stateBytes <
              implicit.resident.stateBytes,
            "容量を減らしても state のバイト数が動かない",
          );
          assertThrows(
            () => pipeline.estimateSessionMemory({ capacity: chunkLength - 1 }),
            Error,
            `capacity ${chunkLength - 1} が chunkLength ${chunkLength} を下回る`,
          );
          assertThrows(
            () => pipeline.estimateSessionMemory({ capacity: maxPosition + 1 }),
            Error,
            `capacity ${maxPosition + 1} が maxPosition ${maxPosition} を超えた`,
          );
        });
        await t.step("chatのcapacityはsequenceへ降りる（既定へ黙って縮退しない）", async () => {
          // 降ろし忘れは既定でも動くので**例外にならない** — 受理集合の外を渡して落ちることで
          // 降りていることを見る。落ちるのは発行時ではなく最初の `next()`。
          const { chunkLength } = pipeline.program;
          const stream = pipeline.chat([{ role: "user", content: "hi" }], {
            maxNewTokens: 4,
            capacity: chunkLength - 1,
          });
          await assertRejects(
            () => stream.text(),
            Error,
            `capacity ${chunkLength - 1} が chunkLength ${chunkLength} を下回る`,
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

/**
 * 取得済みバイト列から組む面（ADR 0097 追記 6 の公開入口のもう一方）。
 *
 * `fromPretrained` の Session と同時に張ると重みが 2 重に常駐するので、別テストにして
 * 順に走らせる（Deno のテストは既定で直列）。
 */
const qatAssets = (model: "e2b" | "e4b"): Promise<Gemma4Assets> =>
  // PLE も `model` 容器の中に在る（ADR 0109 決定 4）ので、渡すのは part 列と tokenizer だけ。
  gemma4MirrorAssets(root, { model });

for (const model of ["e2b", "e4b"] as const) {
  Deno.test({
    name: `gemma4-qat ${model}: fromAssets も同じ構築と応答を通る（実GPU）`,
    ignore: !available || !GPU_AVAILABLE,
    fn: async () => {
      const pipeline = await Gemma4QatPipeline.fromAssets(await qatAssets(model), {
        maxResidentPleBytes: 0,
      });
      try {
        // model 名は manifest ではなくグラフと PLE の構成から判別される面なので、
        // 同じ短文で fromPretrained と同じ応答になることを見る。
        assertEquals(
          await pipeline.chat([{
            role: "user",
            content: "What is 17 + 28? Reply with only the number.",
          }], { maxNewTokens: 8, sampler: { temperature: 0 } }).text(),
          "45",
        );
      } finally {
        await pipeline.dispose();
      }
    },
  });
}
