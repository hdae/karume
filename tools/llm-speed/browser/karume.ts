import {
  acquireGpu,
  DEFAULT_SUBMIT_POLICY,
  type LinearGemvReduce,
  type StateAttentionReduce,
} from "../../../packages/runtime/mod.ts";
import { localDirectory, parseManifest } from "../../../packages/hub/mod.ts";
import {
  GEMMA4_CHUNK_BUCKETS,
  gemma4ChatPrompt,
  Gemma4Pipeline,
} from "../../../packages/models/gemma.ts";
import { Gemma4QatPipeline } from "../../../packages/models/gemma4-qat.ts";
import { generationTimer } from "../../../examples/shared/generation-timing.ts";
import type { ModelKind, NormalizationMode, PrefillBuckets } from "./config.ts";
import type { EngineHandle, Fixture } from "./runner.ts";

export const loadKarume = async (
  kind: ModelKind,
  fixture: Fixture,
  linearGemvReduce?: LinearGemvReduce,
  prefillBuckets: PrefillBuckets = "default",
  normalization: NormalizationMode = "reference",
  stateAttentionReduce: StateAttentionReduce = "parallel",
  fuseLinearStaticQuantize = false,
  packedStaticQuantize = false,
): Promise<EngineHandle> => {
  const subgroup = normalization === "subgroup32";
  const fuseRmsNormAdd = normalization === "fused" || subgroup;
  const rmsNormReduce = subgroup ? "subgroup32" : "workgroup";
  const gpu = await acquireGpu({
    subgroups: subgroup || linearGemvReduce === "parallel-subgroup32",
  });
  try {
    const manifestResponse = await fetch(`/models/${kind}/karume.json`);
    if (!manifestResponse.ok) {
      throw Error(
        `Karume distribution HTTP ${manifestResponse.status}. Check the server model directory.`,
      );
    }
    const manifestBytes = await manifestResponse.arrayBuffer();
    const manifest = parseManifest(new TextDecoder().decode(manifestBytes));
    const model = manifest.models.e2b;
    if (model === undefined) throw Error("The distribution has no E2B model");
    const quant = model.defaultQuant;
    const effectiveReduce = linearGemvReduce ?? model.quants[quant].session.linearGemvReduce ??
      "sequential";
    const manifestSha256 = Array.from(
      new Uint8Array(await crypto.subtle.digest("SHA-256", manifestBytes)),
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
    const chunkBuckets = prefillBuckets === "default"
      ? GEMMA4_CHUNK_BUCKETS.filter((n) => n < 64)
      : prefillBuckets === "sparse"
      ? [4, 8, 16, 32, 48]
      : [4, 8, 16, 24, 32, 40, 48, 56];
    const submitPolicy = normalization === "reference" ? DEFAULT_SUBMIT_POLICY : {
      ...DEFAULT_SUBMIT_POLICY,
      maxChunkSize: 768,
    };
    const common = {
      stateAttentionReduce,
      fuseLinearStaticQuantize,
      packedStaticQuantize,
      fuseRmsNormAdd,
      rmsNormReduce,
      submitPolicy,
      gpu,
      model: "e2b",
      quant,
      chunkLength: 64,
      chunkBuckets,
      ...(linearGemvReduce === undefined ? {} : { linearGemvReduce }),
    } as const;
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
        enabledFeatures: [...gpu.features],
        wgslLanguageFeatures: [...gpu.wgslLanguageFeatures],
        compute: "f32",
        quant,
        linearGemvReduce: effectiveReduce,
        linearGemvReduceOverride: linearGemvReduce ?? null,
        weights: kind === "normal"
          ? "Karume packed i4 / i8"
          : "Karume fixed int2 / int4 / int8 + SRQ",
        capacity: fixture.capacity,
        chunkLength: 64,
        normalization,
        stateAttentionReduce,
        fuseLinearStaticQuantize,
        packedStaticQuantize,
        fuseRmsNormAdd,
        rmsNormReduce,
        submitMaxChunkSize: submitPolicy.maxChunkSize,
        prefillBuckets,
        chunkBuckets,
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
