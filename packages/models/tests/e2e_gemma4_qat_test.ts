/** 固定 QAT の公開入口と共通会話層の結線を検収する。品質全般の検査ではない（ADR 0097）。 */
import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import { parseManifest, resolveFiles } from "@karume/hub";
import { denoDirectory } from "@karume/hub/deno";
import {
  type Gemma4Assets,
  Gemma4ChatSession,
  Gemma4QatPipeline,
  parseGemma4PipelineConfig,
} from "../gemma4-qat.ts";
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
            `capacity ${chunkLength - 1} が chunkLength ${chunkLength} 未満`,
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
const qatAssets = async (model: "e2b" | "e4b"): Promise<Gemma4Assets> => {
  // `parseManifest` は**生のテキスト**を受ける（上限バイト数を自分で見るため）。
  const manifest = parseManifest(await Deno.readTextFile(new URL("karume.json", root)));
  const files = resolveFiles(manifest, { model, weights: ["model"] });
  const readRef = async (key: string): Promise<Uint8Array<ArrayBuffer>> => {
    const ref = files[key];
    if (ref === undefined) {
      throw new Error(
        `test: 資産 '${key}' が配布形の表に無い（${Object.keys(files).join(" / ")}）`,
      );
    }
    return await Deno.readFile(new URL(ref.path, root));
  };
  return {
    config: parseGemma4PipelineConfig(manifest.models[model].pipelineConfig),
    // 並びは manifest の宣言順（先頭がグラフ shard）。
    model: await Promise.all(
      Object.keys(files).filter((key) => key.startsWith("model[")).map(readRef),
    ),
    tokenizer: await readRef("tokenizer"),
    pleIndex: await readRef("ple_index"),
    openPleShard: (file) => {
      const ref = files[file];
      if (ref === undefined) {
        throw new Error(`test: PLE shard '${file}' が配布形の表に無い`);
      }
      return Promise.resolve({
        bytes: ref.size,
        // NOTE: `Deno.readFile` が返す配列は tight（offset 0・buffer 長 = ファイル長）。
        readAll: async () => (await Deno.readFile(new URL(ref.path, root))).buffer,
      });
    },
  };
};

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
