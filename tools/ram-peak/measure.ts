/**
 * ロード時のホスト RAM ピーク計測（メモリ管理波 Phase B — 時点実測の道具・使い捨て）。
 *
 *     # パイプライン面（fromPretrained → 最小生成 1 回）
 *     deno run -A tools/ram-peak/measure.ts --family anima --source models/karume-anima \
 *         --model anima-turbo-v1.1 --quant f16 --steps 2 --size 512
 *     deno run -A tools/ram-peak/measure.ts --family gemma4 --source models/karume-gemma4-e2b
 *     # コンポーネント面（createSessionFromShards 直叩き・1 コンポーネントだけ・刻みノブ付き）
 *     deno run -A tools/ram-peak/measure.ts --mode component --source models/karume-anima \
 *         --model anima-turbo-v1.1 --quant f16 --component transformer --fence-bytes 134217728
 *
 * MUST: 1 構成 = 1 プロセス。ピークはプロセス終端で読む（Linux は `/proc/self/status` の
 * VmHWM = 高水位標・Mac は無いので `Deno.memoryUsage().rss` の 50ms サンプリング最大値のみ）。
 * anima は Session を生成時に張る（`withSession`）ので、最小の生成 1 回まで回してから測る。
 * gemma4 は `fromPretrained` で常駐 Session を組むので、組めた時点で測る。
 * 出力は JSON 1 行（研究記録の表はこれを集計する）。
 */

import { denoDirectory } from "../../packages/hub/deno.ts";
import { AnimaPipeline } from "../../packages/models/mod.ts";
import { Gemma4Pipeline } from "../../packages/models/gemma.ts";
import {
  acquireGpu,
  createSessionFromShards,
  type ModelShard,
  type SessionBuildStats,
} from "../../packages/runtime/mod.ts";
// 実験専用の非公開ノブ（mod.ts からは輸出しない — テストと同じく src を直接読む）。
import { UPLOAD_FENCE_BYTES } from "../../packages/runtime/src/runtime/session-types.ts";

const args = new Map<string, string>();
for (let at = 0; at < Deno.args.length; at += 2) {
  const [key, value] = [Deno.args[at], Deno.args[at + 1]];
  if (!key.startsWith("--") || value === undefined) throw new Error(`引数 ${key} が対でない`);
  args.set(key.slice(2), value);
}
const mode = args.get("mode") ?? "pipeline";
const family = args.get("family") ?? "anima";
const source = args.get("source");
if (source === undefined) throw new Error("--source <配布形ディレクトリ> は必須");
const model = args.get("model");
const quant = args.get("quant");
const component = args.get("component") ?? "transformer";
const fenceBytesRaw = args.get("fence-bytes");
const fenceBytes = fenceBytesRaw === undefined ? undefined : Number(fenceBytesRaw);
const steps = Number(args.get("steps") ?? "2");
// 診断: shard 境界で明示 GC（`deno run --v8-flags=--expose-gc` が前提・無ければ no-op）。
const explicitGc = args.get("gc") === "true";
// コンポーネント面の読み手で器を使い回す（hub 逐次面の器の再利用と同じ形 — 新旧の A/B 用）。
const reuseVessel = args.get("vessel") === "true";
const size = Number(args.get("size") ?? "512");

const mib = (bytes: number): number => Math.round(bytes / 1048576);
const vmHwm = async (): Promise<number | undefined> => {
  if (Deno.build.os !== "linux") return undefined;
  const row = (await Deno.readTextFile("/proc/self/status")).split("\n").find((line) =>
    line.startsWith("VmHWM:")
  );
  return row === undefined ? undefined : Number(row.replace(/[^0-9]/g, "")) * 1024;
};

/** manifest から (model, component, quant) の shard 列を読み、1 本ずつ流す（hub と同じ順）。 */
type ManifestShard = { readonly path: string; readonly size: number };
const componentShards = async (): Promise<
  { readonly model: string; readonly quant: string; readonly shards: readonly ManifestShard[] }
> => {
  const manifest = JSON.parse(await Deno.readTextFile(`${source}/karume.json`));
  const modelName = model ?? manifest.defaultModel;
  const entry = manifest.models[modelName];
  const quantName = quant ?? entry.defaultQuant;
  const weights = entry.weights[component]?.[entry.quants[quantName].weights[component]];
  if (weights === undefined) {
    throw new Error(`${modelName} / ${quantName} に component '${component}' が無い`);
  }
  return { model: modelName, quant: quantName, shards: weights.shards };
};
async function* streamShards(shards: readonly ManifestShard[]): AsyncGenerator<ModelShard> {
  const largest = shards.reduce((max, shard) => Math.max(max, shard.size), 0);
  const vessel = reuseVessel ? new Uint8Array(new ArrayBuffer(largest)) : undefined;
  for (const shard of shards) {
    if (explicitGc) (globalThis as { gc?: () => void }).gc?.();
    if (vessel === undefined) {
      const bytes = await Deno.readFile(`${source}/${shard.path}`);
      yield { id: shard.path, bytes: bytes as Uint8Array<ArrayBuffer> };
      continue;
    }
    const file = await Deno.open(`${source}/${shard.path}`);
    try {
      let filled = 0;
      while (filled < shard.size) {
        const read = await file.read(vessel.subarray(filled, shard.size));
        if (read === null) throw new Error(`${shard.path} が宣言 size より短い`);
        filled += read;
      }
    } finally {
      file.close();
    }
    yield { id: shard.path, bytes: new Uint8Array(vessel.buffer, 0, shard.size) };
  }
}

let rssMax = 0;
const sampler = setInterval(() => {
  rssMax = Math.max(rssMax, Deno.memoryUsage().rss);
}, 50);
const rssBaseline = Deno.memoryUsage().rss;
const builds: Record<string, SessionBuildStats> = {};
const selection = {
  ...(model === undefined ? {} : { model }),
  ...(quant === undefined ? {} : { quant }),
};
const started = performance.now();
let loadMs = 0;
let runMs = 0;
let resolved: { model: string | null; quant: string | null } = {
  model: model ?? null,
  quant: quant ?? null,
};
if (mode === "component") {
  const target = await componentShards();
  resolved = { model: target.model, quant: target.quant };
  const gpu = await acquireGpu();
  try {
    const session = await createSessionFromShards(gpu, streamShards(target.shards), {
      ...(fenceBytes === undefined ? {} : { [UPLOAD_FENCE_BYTES]: fenceBytes }),
    });
    loadMs = performance.now() - started;
    builds[component] = session.diagnostics().buildStats;
    await session.dispose();
  } finally {
    gpu.destroy();
  }
} else if (family === "anima") {
  const pipeline = await AnimaPipeline.fromPretrained(denoDirectory(source), {
    ...selection,
    onRunDiagnostics: (name, diagnostics) => {
      builds[name] = diagnostics.buildStats;
    },
  });
  loadMs = performance.now() - started;
  const runStarted = performance.now();
  await pipeline.generate({
    prompt: "1girl, solo, upper body",
    steps,
    resolution: { width: size, height: size },
    seed: 1,
  });
  runMs = performance.now() - runStarted;
  await pipeline.dispose();
} else if (family === "gemma4") {
  const pipeline = await Gemma4Pipeline.fromPretrained(denoDirectory(source), selection);
  loadMs = performance.now() - started;
  await pipeline.dispose();
} else {
  throw new Error(`--family ${family} は未対応（anima | gemma4）`);
}
clearInterval(sampler);
rssMax = Math.max(rssMax, Deno.memoryUsage().rss);

const peak = await vmHwm();
console.log(JSON.stringify({
  mode,
  family: mode === "component" ? null : family,
  component: mode === "component" ? component : null,
  fenceBytes: fenceBytes ?? null,
  explicitGc,
  reuseVessel,
  source,
  ...resolved,
  os: Deno.build.os,
  vmHwmMiB: peak === undefined ? null : mib(peak),
  rssMaxMiB: mib(rssMax),
  rssBaselineMiB: mib(rssBaseline),
  loadMs: Math.round(loadMs),
  runMs: Math.round(runMs),
  builds: Object.fromEntries(
    Object.entries(builds).map(([name, stats]) => [name, {
      shardCount: stats.shardCount,
      uploadedMiB: mib(stats.uploadedBytes),
      shardWaitMs: Math.round(stats.shardWaitMs),
      decodeMs: Math.round(stats.decodeMs),
      bufferCreateMs: Math.round(stats.bufferCreateMs),
      writeBufferIssueMs: Math.round(stats.writeBufferIssueMs),
      uploadFenceMs: Math.round(stats.uploadFenceMs),
    }]),
  ),
}));
