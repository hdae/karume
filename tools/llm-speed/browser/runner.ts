import type { LinearGemvReduce, StateAttentionReduce } from "../../../packages/runtime/mod.ts";
import {
  generationTimer,
  type GenerationTiming,
} from "../../../examples/shared/generation-timing.ts";
import {
  type Engine,
  type ModelKind,
  type NormalizationMode,
  onnxModels,
  type PrefillBuckets,
  versions,
} from "./config.ts";
import { loadKarume } from "./karume.ts";
export type Fixture = {
  cases: { case: string; prompt: string; inputIds: number[] }[];
  stopTokens: number[];
  maxNewTokens: number;
  capacity: number;
};
export type RunResult = GenerationTiming & {
  tokenIds: number[];
  stopToken: number | null;
  text: string;
};
export type EngineHandle = {
  metadata: object;
  generate: (ids: number[]) => Promise<RunResult>;
  dispose: () => Promise<void>;
};

type OnnxTensor = { data: BigInt64Array<ArrayBuffer> };
type TransformerModel = {
  sessions: Record<string, unknown>;
  generate: (options: object) => Promise<OnnxTensor>;
  dispose: () => Promise<void>;
};
type Transformers = {
  env: {
    allowRemoteModels: boolean;
    allowLocalModels: boolean;
    localModelPath: string;
    backends: {
      onnx: {
        versions: { web: string };
        wasm: { numThreads: number; wasmPaths: { mjs: string; wasm: string } };
        webgpu: { device?: GPUDevice; powerPreference?: string };
      };
    };
  };
  Tensor: new (dtype: string, data: BigInt64Array<ArrayBuffer>, dims: number[]) => OnnxTensor;
  AutoTokenizer: {
    from_pretrained: (
      path: string,
      options: object,
    ) => Promise<{ decode: (ids: number[], options?: object) => string }>;
  };
  Gemma4ForCausalLM: {
    from_pretrained: (path: string, options: object) => Promise<TransformerModel>;
  };
};

const loadTransformers = async (
  kind: ModelKind,
  fixture: Fixture,
  local: boolean,
  status: (s: string) => void,
): Promise<EngineHandle> => {
  const moduleUrl = `/vendor/transformers.js?v=${versions.transformers}`;
  // 固定版の外部ライブラリ境界。内部 API の型をプロジェクト全体へ漏らさない。
  const imported: unknown = await import(moduleUrl);
  if (
    typeof imported !== "object" || imported === null || !("Gemma4ForCausalLM" in imported) ||
    !("env" in imported) || !("Tensor" in imported) || !("AutoTokenizer" in imported)
  ) throw Error("Unsupported Transformers.js module");
  const lib = imported as Transformers;
  const { env, Tensor } = lib;
  env.allowRemoteModels = !local;
  env.allowLocalModels = local;
  env.localModelPath = "/onnx/";
  env.backends.onnx.wasm.numThreads = 1;
  env.backends.onnx.webgpu.powerPreference = "high-performance";
  env.backends.onnx.wasm.wasmPaths = {
    mjs: `${location.origin}/vendor/ort-wasm-simd-threaded.asyncify.mjs?v=${versions.onnxruntime}`,
    wasm:
      `${location.origin}/vendor/ort-wasm-simd-threaded.asyncify.wasm?v=${versions.onnxruntime}`,
  };
  if (env.backends.onnx.versions.web !== versions.onnxruntime) {
    throw Error("ONNX Runtime version mismatch");
  }
  const spec = onnxModels[kind];
  const modelPath = local ? kind : spec.repo;
  const options = { revision: spec.revision };
  const tokenizer = await lib.AutoTokenizer.from_pretrained(modelPath, options);
  const model = await lib.Gemma4ForCausalLM.from_pretrained(modelPath, {
    ...options,
    device: "webgpu",
    dtype: spec.dtype,
    progress_callback: (value: unknown) => {
      if (
        typeof value === "object" && value !== null && "file" in value && "status" in value &&
        value.status === "done"
      ) status(`Loaded ${String(value.file)}`);
      if (
        typeof value === "object" && value !== null && "file" in value && "progress" in value &&
        typeof value.progress === "number"
      ) status(`Loading ${String(value.file)}: ${value.progress.toFixed(0)}%`);
    },
  });
  try {
    if (Object.keys(model.sessions).sort().join(",") !== "decoder_model_merged,embed_tokens") {
      throw Error("Expected only text decoder and embeddings");
    }
    const device = env.backends.onnx.webgpu.device;
    if (!device || !device.features.has("shader-f16")) {
      throw Error("ONNX Runtime requires shader-f16");
    }
    await device.queue.onSubmittedWorkDone();
  } catch (error) {
    await model.dispose();
    throw error;
  }
  return {
    metadata: {
      ...versions,
      ...spec,
      local,
      compute: "f16",
      sessions: Object.keys(model.sessions),
      wasmThreads: 1,
      cache: "dynamic KV",
      textOnly: true,
    },
    generate: async (ids) => {
      await env.backends.onnx.webgpu.device?.queue.onSubmittedWorkDone();
      const input_ids = new Tensor("int64", BigInt64Array.from(ids, BigInt), [1, ids.length]);
      const attention_mask = new Tensor("int64", new BigInt64Array(ids.length).fill(1n), [
        1,
        ids.length,
      ]);
      let prompt = true;
      const tokenIds: number[] = [];
      let stopToken: number | null = null;
      const timer = generationTimer();
      const output = await model.generate({
        input_ids,
        attention_mask,
        max_new_tokens: fixture.maxNewTokens,
        do_sample: false,
        eos_token_id: fixture.stopTokens,
        streamer: {
          put: (batch: bigint[][]): void => {
            if (batch.length !== 1) throw Error("Expected batch size one");
            if (prompt) {
              if (JSON.stringify(batch[0].map(Number)) !== JSON.stringify(ids)) {
                throw Error("Streamer prompt mismatch");
              }
              prompt = false;
              return;
            }
            for (const value of batch[0]) {
              const id = Number(value);
              if (fixture.stopTokens.includes(id)) {
                stopToken = id;
                continue;
              }
              timer.onToken();
              tokenIds.push(id);
            }
          },
          end: (): void => {},
        },
      });
      const timing = timer.finish();
      const generated = Array.from(output.data, Number).slice(ids.length);
      if (
        JSON.stringify(generated) !==
          JSON.stringify([...tokenIds, ...(stopToken === null ? [] : [stopToken])])
      ) throw Error("Streamer and generated sequence differ");
      return {
        ...timing,
        tokenIds,
        stopToken,
        text: tokenizer.decode(tokenIds, { skip_special_tokens: true }),
      };
    },
    dispose: async () => {
      await model.dispose();
      env.backends.onnx.webgpu.device?.destroy();
    },
  };
};

export const runBenchmark = async (
  engine: Engine,
  kind: ModelKind,
  localOnnx: boolean,
  status: (s: string) => void,
  linearGemvReduce?: LinearGemvReduce,
  prefillBuckets: PrefillBuckets = "default",
  normalization: NormalizationMode = "reference",
  stateAttentionReduce: StateAttentionReduce = "parallel",
  fuseLinearStaticQuantize = false,
): Promise<object> => {
  const response = await fetch("/cases.json");
  if (!response.ok) throw Error(`Fixture HTTP ${response.status}`);
  const fixture: Fixture = await response.json();
  const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter || adapter.info.isFallbackAdapter) {
    throw Error("A hardware WebGPU adapter is required");
  }
  if (engine === "transformers" && !adapter.features.has("shader-f16")) {
    throw Error("This Chrome GPU adapter lacks shader-f16; the selected ONNX models require it.");
  }
  const adapterInfo = {
    vendor: adapter.info.vendor,
    architecture: adapter.info.architecture,
    description: adapter.info.description,
    device: adapter.info.device,
    isFallbackAdapter: adapter.info.isFallbackAdapter,
    features: [...adapter.features],
  };
  status(`${kind} / ${engine}: loading`);
  const started = performance.now();
  const handle = engine === "karume"
    ? await loadKarume(
      kind,
      fixture,
      linearGemvReduce,
      prefillBuckets,
      normalization,
      stateAttentionReduce,
      fuseLinearStaticQuantize,
    )
    : await loadTransformers(kind, fixture, localOnnx, status);
  const loadSeconds = (performance.now() - started) / 1000;
  try {
    const cases = [];
    for (const [index, c] of fixture.cases.entries()) {
      status(`${kind} / ${engine}: ${c.case} first`);
      const first = await handle.generate(c.inputIds);
      status(`${kind} / ${engine}: ${c.case} warmup`);
      const warmup = await handle.generate(c.inputIds);
      const measured = [];
      for (let i = 0; i < 3; i++) {
        status(`${kind} / ${engine}: ${c.case} repeat ${i + 1}/3`);
        measured.push(await handle.generate(c.inputIds));
      }
      const deterministic = [warmup, ...measured].every((r) =>
        JSON.stringify(r.tokenIds) === JSON.stringify(first.tokenIds) &&
        r.stopToken === first.stopToken
      );
      cases.push({
        ...c,
        firstInProcess: index === 0,
        first,
        warmups: [warmup],
        measured,
        deterministic,
      });
      if (!deterministic) {
        status(`WARNING: ${c.case} tokens differed across repetitions (retained in JSON)`);
      }
    }
    return {
      format: "karume-browser-speed/1",
      engine,
      model: kind,
      userAgent: navigator.userAgent,
      platform: navigator.platform,
      adapter: adapterInfo,
      date: new Date().toISOString(),
      loadSeconds,
      ...handle.metadata,
      sampler: { temperature: 0 },
      maxNewTokens: fixture.maxNewTokens,
      textDecodingTimed: false,
      fixture,
      cases,
    };
  } finally {
    await handle.dispose();
  }
};
