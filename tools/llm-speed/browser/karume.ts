import { acquireGpu, type LinearGemvReduce } from "../../../packages/runtime/mod.ts";
import { localDirectory } from "../../../packages/hub/mod.ts";
import { gemma4ChatPrompt, Gemma4Pipeline } from "../../../packages/models/gemma.ts";
import { Gemma4QatPipeline } from "../../../packages/models/gemma4-qat.ts";
import { generationTimer } from "../../../examples/shared/generation-timing.ts";
import type { ModelKind } from "./config.ts";
import type { EngineHandle, Fixture } from "./runner.ts";

export const loadKarume = async (
  kind: ModelKind,
  fixture: Fixture,
  linearGemvReduce: LinearGemvReduce = "sequential",
): Promise<EngineHandle> => {
  const gpu = await acquireGpu();
  try {
    const manifestResponse = await fetch(`/models/${kind}/karume.json`);
    if (!manifestResponse.ok) {
      throw Error(
        `Karume distribution HTTP ${manifestResponse.status}. Check the server model directory.`,
      );
    }
    const manifestSha256 = Array.from(
      new Uint8Array(await crypto.subtle.digest("SHA-256", await manifestResponse.arrayBuffer())),
      (v) => v.toString(16).padStart(2, "0"),
    ).join("");
    const source = localDirectory({
      readFile: async (path, options) => {
        const r = await fetch(`/models/${kind}/${path}`, options);
        if (!r.ok) throw Error(`Model HTTP ${r.status}: ${path}`);
        return new Uint8Array(await r.arrayBuffer());
      },
      readFileRange: async (path, offset, length, options) => {
        const r = await fetch(`/models/${kind}/${path}`, {
          ...options,
          headers: { Range: `bytes=${offset}-${offset + length - 1}` },
        });
        if (r.status !== 206) throw Error(`Model range HTTP ${r.status}: ${path}`);
        return new Uint8Array(await r.arrayBuffer());
      },
    }, { label: `browser-speed-${kind}` });
    const common = { gpu, model: "e2b", chunkLength: 64, linearGemvReduce } as const;
    const pipeline = kind === "normal"
      ? await Gemma4Pipeline.fromPretrained(source, common)
      : await Gemma4QatPipeline.fromPretrained(source, common);
    try {
      for (const c of fixture.cases) {
        const ids = gemma4ChatPrompt(pipeline.tokenizer, [{ role: "user", content: c.prompt }]);
        if (JSON.stringify(ids) !== JSON.stringify(c.inputIds)) {
          throw Error("Chat template differs from fixed fixture");
        }
      }
      await gpu.device.queue.onSubmittedWorkDone();
    } catch (error) {
      await pipeline.dispose();
      throw error;
    }
    return {
      metadata: {
        manifestSha256,
        compute: "f32",
        linearGemvReduce,
        weights: kind === "normal"
          ? "Karume packed i4 / i8"
          : "Karume fixed int2 / int4 / int8 + SRQ",
        capacity: fixture.capacity,
        chunkLength: 64,
        pleBudget: "default-two-shards",
      },
      generate: async (ids) => {
        await gpu.device.queue.onSubmittedWorkDone();
        const timer = generationTimer();
        const sequence = await pipeline.sequence({ capacity: fixture.capacity });
        await using _sequence = { [Symbol.asyncDispose]: () => sequence.dispose() };
        const stream = sequence.generate({
          prompt: ids,
          maxNewTokens: fixture.maxNewTokens,
          stopTokens: fixture.stopTokens,
          sampler: { temperature: 0 },
        });
        const tokenIds: number[] = [];
        for await (const event of stream) {
          if (event.kind === "token") {
            timer.onToken();
            tokenIds.push(event.id);
          }
        }
        const stop = await stream.done;
        const timing = timer.finish();
        return {
          ...timing,
          tokenIds,
          stopToken: stop.reason === "eos" || stop.reason === "stop-token" ? stop.token : null,
          text: pipeline.tokenizer.decode(tokenIds),
        };
      },
      dispose: async () => {
        try {
          await pipeline.dispose();
        } finally {
          gpu.destroy();
        }
      },
    };
  } catch (error) {
    gpu.destroy();
    throw error;
  }
};
