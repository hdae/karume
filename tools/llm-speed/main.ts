import { assert, assertEquals } from "@std/assert";
import { acquireGpu, prepareModel } from "../../packages/runtime/mod.ts";
import { denoDirectory } from "../../packages/hub/deno.ts";
import { gemma4ChatPrompt, Gemma4Pipeline } from "../../packages/models/gemma.ts";
import { Gemma4QatPipeline } from "../../packages/models/gemma4-qat.ts";
import { gemma4StopTokens } from "../../packages/models/src/gemma/text/chat.ts";
import {
  readShard,
  resolveShards,
  streamShards,
} from "../../packages/runtime/tests/helpers/shard-files.ts";
import { createLlmTokenizer, record } from "../../examples/shared/llm-tokenizer.ts";
import { localFileUrl } from "../../examples/shared/llm-source.ts";
import { inspectLlmGraph, LlmSequence } from "../../examples/shared/llm-generate.ts";
import { generationTimer, type GenerationTiming } from "../../examples/shared/generation-timing.ts";

type Case = { case: string; prompt: string; inputIds: number[] };
type Inputs = {
  capacity: number;
  maxNewTokens: number;
  stopTokens: number[];
  cases: Case[];
  tokenizerSha256: string;
};
type Result = GenerationTiming & {
  tokenIds: number[];
  stopToken: number | null;
  stopReason: string;
  promptTokens: number;
};
type Profile =
  | {
    name: string;
    family: "gemma4" | "gemma4-qat";
    model: "e2b" | "e4b";
    source: string;
    tokenizer: string;
  }
  | { name: string; family: "minicpm5" | "qwen3"; source: string; tokenizer: string };
const profileFrom = (raw: unknown): Profile => {
  const root = record(raw, "inputs");
  assert(typeof root.model === "string");
  const p = record(root.profile, "profile");
  assert(typeof p.checkpoint === "string");
  const tokenizer = `${p.checkpoint.replace(/\/+$/, "")}/tokenizer.json`;
  const family = p.family;
  if (family === "gemma4" || family === "gemma4-qat") {
    assert(typeof p.distribution === "string" && (p.model === "e2b" || p.model === "e4b"));
    return { name: root.model, family, model: p.model, source: p.distribution, tokenizer };
  }
  assert(family === "minicpm5" || family === "qwen3");
  assert(typeof p.series === "string");
  return { name: root.model, family, source: p.series, tokenizer };
};
const ids = (raw: unknown): number[] => {
  assert(
    Array.isArray(raw) &&
      raw.every((v) => typeof v === "number" && Number.isSafeInteger(v) && v >= 0),
  );
  return raw;
};
const input = (raw: unknown): Inputs => {
  const root = record(raw, "input");
  assertEquals(root.format, "karume-llm-speed-input/2");
  assert(root.capacity === 128, "この比較の容量は128です");
  assert(
    typeof root.maxNewTokens === "number" && Number.isSafeInteger(root.maxNewTokens) &&
      root.maxNewTokens > 0,
  );
  const data = root;
  assert(typeof data.tokenizerSha256 === "string");
  assert(Array.isArray(data.cases));
  const cases = data.cases.map((raw) => {
    const c = record(raw, "case");
    assert(
      typeof c.case === "string" && /^[a-z0-9-]+$/.test(c.case) && typeof c.prompt === "string",
    );
    return { case: c.case, prompt: c.prompt, inputIds: ids(c.inputIds) };
  });
  assert(cases.length > 0 && new Set(cases.map((c) => c.case)).size === cases.length);
  for (const c of cases) {
    assert(c.inputIds.length > 0 && c.inputIds.length + root.maxNewTokens - 1 <= root.capacity);
  }
  return {
    capacity: root.capacity,
    maxNewTokens: root.maxNewTokens,
    stopTokens: ids(data.stopTokens),
    cases,
    tokenizerSha256: data.tokenizerSha256,
  };
};
const save = async (path: string, value: unknown): Promise<void> => {
  await Deno.writeTextFile(path, JSON.stringify(value, null, 2) + "\n", { createNew: true });
};
const digest = async (path: string | URL): Promise<string> =>
  Array.from(
    new Uint8Array(await crypto.subtle.digest("SHA-256", await Deno.readFile(path))),
    (b) => b.toString(16).padStart(2, "0"),
  ).join("");

const main = async (): Promise<void> => {
  if (Deno.args.length === 1 && Deno.args[0] === "--help") {
    console.log(
      "deno run -A tools/llm-speed/main.ts --inputs <inputs.json> --out <新規ディレクトリ> [--repeats 3] [--warmups 1]",
    );
    return;
  }
  const args = new Map<string, string>();
  for (let i = 0; i < Deno.args.length; i += 2) {
    const key = Deno.args[i], value = Deno.args[i + 1];
    assert(
      ["--inputs", "--out", "--repeats", "--warmups"].includes(key) && value !== undefined &&
        !value.startsWith("--") && !args.has(key),
      `不正な引数 ${key}`,
    );
    args.set(key, value);
  }
  const inputsPath = args.get("--inputs"), out = args.get("--out");
  assert(inputsPath && out, "--inputs と --out が必要です");
  const positive = (key: string, fallback: number): number => {
    const raw = args.get(key) ?? String(fallback);
    assert(
      /^[1-9]\d*$/.test(raw) && Number.isSafeInteger(Number(raw)),
      `${key} は正の安全整数が必要です`,
    );
    return Number(raw);
  };
  const repeats = positive("--repeats", 3), warmupCount = positive("--warmups", 1);
  const raw: unknown = JSON.parse(await Deno.readTextFile(inputsPath));
  const profile = profileFrom(raw), name = profile.name;
  const data = input(raw);
  assertEquals(await digest(profile.tokenizer), data.tokenizerSha256);
  await Deno.mkdir(out);
  const gpu = await acquireGpu();
  using _gpu = { [Symbol.dispose]: () => gpu.destroy() };
  const metadata = {
    format: "karume-llm-speed-result/1",
    engine: "deno-webgpu",
    model: name,
    source: profile.source,
    deno: Deno.version,
    platform: Deno.build,
    adapter: {
      vendor: gpu.adapterInfo.vendor,
      architecture: gpu.adapterInfo.architecture,
      device: gpu.adapterInfo.device,
      description: gpu.adapterInfo.description,
    },
    capacity: data.capacity,
    chunkLength: 64,
    maxNewTokens: data.maxNewTokens,
    sampler: { temperature: 0 },
    placement: "packed-weights",
    pleBudget: profile.family.startsWith("gemma4") ? "default-two-shards" : undefined,
    compute: "f32",
    textDecodingTimed: false,
    inputSha256: await digest(inputsPath),
    scriptSha256: await digest(new URL(import.meta.url)),
  };
  const runCases = async (
    generate: (ids: readonly number[]) => Promise<Result>,
    loaded: number,
  ): Promise<void> => {
    const cases = [];
    for (const [caseIndex, c] of data.cases.entries()) {
      const first = await generate(c.inputIds);
      await save(`${out}/${c.case}-first.json`, first);
      console.log(c.case, "first", first.ttftMs, first.decodeTokensPerSecond);
      const warmups: Result[] = [], measured: Result[] = [];
      for (let i = 0; i < warmupCount; i++) {
        const r = await generate(c.inputIds);
        warmups.push(r);
        await save(`${out}/${c.case}-warmup-${i}.json`, r);
      }
      for (let i = 0; i < repeats; i++) {
        const r = await generate(c.inputIds);
        measured.push(r);
        await save(`${out}/${c.case}-measured-${i}.json`, r);
        console.log(c.case, i, r.ttftMs, r.decodeTokensPerSecond);
      }
      for (const r of [...warmups, ...measured]) {
        assertEquals(r.tokenIds, first.tokenIds);
        assertEquals(r.stopToken, first.stopToken);
      }
      cases.push({ case: c.case, first, firstInProcess: caseIndex === 0, warmups, measured });
    }
    await save(`${out}/summary.json`, { ...metadata, loadSeconds: loaded, cases });
  };
  const started = performance.now();
  if (profile.family === "gemma4" || profile.family === "gemma4-qat") {
    const common = { gpu, model: profile.model, chunkLength: 64 };
    await using pipeline = profile.family === "gemma4"
      ? await Gemma4Pipeline.fromPretrained(denoDirectory(profile.source), common)
      : await Gemma4QatPipeline.fromPretrained(denoDirectory(profile.source), {
        ...common,
        model: profile.model,
      });
    await gpu.device.queue.onSubmittedWorkDone();
    const loaded = (performance.now() - started) / 1000;
    assertEquals(
      [...gemma4StopTokens(pipeline.tokenizer)].sort((a, b) => a - b),
      [...data.stopTokens].sort((a, b) => a - b),
    );
    for (const c of data.cases) {
      assertEquals(
        gemma4ChatPrompt(pipeline.tokenizer, [{ role: "user", content: c.prompt }]),
        c.inputIds,
      );
    }
    await runCases(async (prompt) => {
      await gpu.device.queue.onSubmittedWorkDone();
      const timer = generationTimer();
      const sequence = await pipeline.sequence({ capacity: data.capacity });
      await using _sequence = { [Symbol.asyncDispose]: () => sequence.dispose() };
      const stream = sequence.generate({
        prompt,
        maxNewTokens: data.maxNewTokens,
        stopTokens: data.stopTokens,
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
      return {
        ...timer.finish(),
        tokenIds,
        stopToken: stop.reason === "eos" || stop.reason === "stop-token" ? stop.token : null,
        stopReason: stop.reason,
        promptTokens: prompt.length,
      };
    }, loaded);
  } else {
    const tokenizer = createLlmTokenizer(
      profile.family,
      JSON.parse(await Deno.readTextFile(profile.tokenizer)),
      JSON.parse(
        await Deno.readTextFile(new URL("../../examples/shared/llm-unicode.json", import.meta.url)),
      ),
    );
    assertEquals(
      [...tokenizer.stopTokens].sort((a, b) => a - b),
      [...data.stopTokens].sort((a, b) => a - b),
    );
    for (const c of data.cases) assertEquals(tokenizer.chat(c.prompt), c.inputIds);
    const shards = resolveShards(localFileUrl(`${profile.source}/model.safetensors`));
    const prepared = prepareModel(await readShard(shards[0]));
    const graph = inspectLlmGraph(profile.family, prepared.graph);
    const session = await prepared.createSession(gpu, streamShards(shards.slice(1)));
    await using _session = { [Symbol.asyncDispose]: () => session.dispose() };
    await gpu.device.queue.onSubmittedWorkDone();
    const loaded = (performance.now() - started) / 1000;
    await runCases(async (prompt) => {
      await gpu.device.queue.onSubmittedWorkDone();
      const timer = generationTimer();
      await using sequence = new LlmSequence(session, graph);
      const tokenIds: number[] = [];
      let stopToken: number | null = null;
      for await (const token of sequence.stream(prompt, data.maxNewTokens, data.stopTokens)) {
        if (data.stopTokens.includes(token)) {
          stopToken = token;
          continue;
        }
        timer.onToken();
        tokenIds.push(token);
      }
      return {
        ...timer.finish(),
        tokenIds,
        stopToken,
        stopReason: stopToken === null ? "max-tokens" : "eos",
        promptTokens: prompt.length,
      };
    }, loaded);
  }
};
if (import.meta.main) await main();
