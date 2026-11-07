/**
 * `PipelineOptions.fullStateStorage` on the real 12B: a short greedy chat runs on f16 full slots, and
 * the estimate reports those slots halved.
 *
 * This is a smoke check of the wiring on real weights, not a quality measurement: the numbers on the
 * small graphs are pinned word for word by `gpu_full_state_storage_test.ts`, and the kernels by
 * `gpu_state_kv16_test.ts`.
 *
 * ## Assets
 *
 * The q4_0 QAT distribution mirror `models/karume-gemma4-qat/` (outside git). Without the mirror or
 * a GPU the case SKIPs explicitly (ADR 0005).
 */
import { test } from "vite-plus/test";
import { assert, assertEquals, assertMatch } from "@std/assert";
import type { SessionDiagnostics } from "@karume/runtime";
import { nodeDirectory } from "@karume/hub/node";
import { Pipeline } from "../../models/gemma.ts";
import { mirrorAvailable } from "../../models/tests/helpers/gemma-mirror.ts";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import { SEAT_SIGNATURES } from "../../runtime/tests/helpers/pipeline-census.ts";

const ROOT = new URL("../../../models/karume-gemma4-qat/", import.meta.url);
const PRESENT = mirrorAvailable(ROOT);
if (!PRESENT) {
  console.warn(
    "[karume] models/karume-gemma4-qat/ is absent (or not karume/5), so the 12B f16 slot case " +
      "SKIPs (assemble it with dist.py in tools/export-recipes)",
  );
}

/** 12B has 8 full layers, each with a k and a v slot `[1,1,C,512]`. */
const FULL_SLOTS = 16;
const CAPACITY = 4096;

test.skipIf(!PRESENT || !GPU_AVAILABLE)(
  "fullStateStorage f16: 12B i4-fast runs a greedy chat, and its estimate halves the full slots (real GPU)",
  async () => {
    const open = async (
      options: Parameters<typeof Pipeline.fromPretrained>[1],
    ): Promise<Pipeline> =>
      await Pipeline.fromPretrained(nodeDirectory(ROOT), {
        model: "12b",
        quant: "i4-fast",
        ...options,
      });

    const reference = await open({});
    const f32Bytes = reference.estimateSessionMemory({ capacity: CAPACITY }).resident.stateBytes;
    await reference.dispose();

    const pipelines: SessionDiagnostics["lastRunPipelines"][] = [];
    const pipeline = await open({
      fullStateStorage: "f16",
      onRunDiagnostics: (diagnostics) => pipelines.push(diagnostics.lastRunPipelines),
    });
    try {
      assertEquals(
        f32Bytes - pipeline.estimateSessionMemory({ capacity: CAPACITY }).resident.stateBytes,
        FULL_SLOTS * (1 * 1 * CAPACITY * 512 * 2),
      );
      const stream = pipeline.chat(
        [{ role: "user", content: "Answer with one word: what colour is the sky on a clear day?" }],
        { maxNewTokens: 16, sampler: { temperature: 0 } },
      );
      // A slot read as anything but its stored words would not come back with this answer.
      assertMatch(await stream.text(), /blue/i);
      assertEquals((await stream.done).reason, "eos");
      const isF16 = SEAT_SIGNATURES.fullStateStorage.f16;
      const keys = pipelines.flatMap((rows) => (rows ?? []).map((row) => row.key));
      assert(keys.some(isF16), "no f16 key ran");
      assert(
        !keys.some((key) => isF16(key) && key.includes(":sliding")),
        "an f16 key ran on a sliding slot",
      );
    } finally {
      await pipeline.dispose();
    }
  },
);
