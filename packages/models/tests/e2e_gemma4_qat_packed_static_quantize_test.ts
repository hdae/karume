/**
 * packed int8 活性（ADR 0105）の e2e 門 — QAT E2B の実配布重みで、席 on / off の
 * **greedy な token id 列が完全一致**することと、対付けの本数（decode 210 / prefill 0）を固定する。
 *
 * 単体の u32 一致門（runtime の `gpu_packed_static_quantize_test.ts`）はカーネル 1 本ぶんしか
 * 見ない。実グラフでは 210 本の SRQ と 275 本の融合エピローグが混ざるので、対付けが 1 本でも
 * 外れた形（生産側だけ packed・消費側だけ packed）はここでしか出ない。
 */
import { assertEquals } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { denoDirectory } from "@karume/hub/deno";
import { acquireGpu, type SessionDiagnostics } from "@karume/runtime";
import { gemma4ChatPrompt } from "../gemma.ts";
import { Gemma4QatPipeline } from "../gemma4-qat.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { mirrorAvailable } from "./helpers/gemma-mirror.ts";

const root = new URL("../../../models/karume-gemma4-qat/", import.meta.url);
const available = mirrorAvailable(root);
if (!available) {
  console.warn(
    `[karume] ${root.pathname} が無い（か karume/5 でない）ため packed 活性の e2e 検査を SKIP`,
  );
}

/** QAT E2B decode 計画で packed になる SRQ の本数（assets_fusion_counts_test.ts と同じ値）。 */
const DECODE_PACKED = 70;

describe({
  name: "gemma4-qat: packed int8 活性（ADR 0105・実GPU）",
  ignore: !available || !GPU_AVAILABLE,
  fn: () => {
    it("席 on / off の 64 token greedy id 列が完全一致し、decode 計画だけが 70 本を掴む", async () => {
      const outputs: number[][] = [];
      const packedByPhase: Record<string, number[]> = {};
      for (const packedStaticQuantize of [false, true]) {
        const gpu = await acquireGpu();
        try {
          const phases: Record<string, number[]> = {};
          const pipeline = await Gemma4QatPipeline.fromPretrained(denoDirectory(root), {
            gpu,
            model: "e2b",
            chunkLength: 32,
            maxResidentPleBytes: 0,
            packedStaticQuantize,
            onRunDiagnostics: (d: SessionDiagnostics, phase) => {
              (phases[phase.kind] ??= []).push(d.lastRunFusions?.packedStaticQuantize ?? -1);
            },
          });
          try {
            const prompt = gemma4ChatPrompt(pipeline.tokenizer, [{
              role: "user",
              content: "Write a numbered list of twenty tips for learning a new language.",
            }]);
            const sequence = await pipeline.sequence({ capacity: 256 });
            try {
              const stream = sequence.generate({
                prompt,
                maxNewTokens: 64,
                sampler: { temperature: 0 },
              });
              const ids: number[] = [];
              for await (const event of stream) {
                if (event.kind === "token") ids.push(event.id);
              }
              // 先に停止理由を見る（早期停止だと id 列の一致比較が長さで割れる）。
              const stop = await stream.done;
              assertEquals(stop.reason, "max-tokens");
              assertEquals(ids.length, 64);
              outputs.push(ids);
            } finally {
              await sequence.dispose();
            }
          } finally {
            await pipeline.dispose();
          }
          const label = packedStaticQuantize ? "on" : "off";
          for (const [kind, counts] of Object.entries(phases)) {
            packedByPhase[`${label}:${kind}`] = [...new Set(counts)];
          }
        } finally {
          gpu.destroy();
        }
      }
      assertEquals(outputs[0], outputs[1], "席 on / off の greedy id 列");
      assertEquals(packedByPhase["off:decode"], [0], "席 off の decode 計画");
      assertEquals(packedByPhase["off:prefill"], [0], "席 off の prefill 計画");
      assertEquals(packedByPhase["on:decode"], [DECODE_PACKED], "席 on の decode 計画");
      assertEquals(packedByPhase["on:prefill"], [0], "席 on の prefill 計画");
    });
  },
});
