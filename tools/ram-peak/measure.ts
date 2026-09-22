/**
 * ロード時のホスト RAM ピーク計測（メモリ管理波 Phase B — 時点実測の道具・使い捨て）。
 *
 *     # パイプライン面（fromPretrained → 最小生成 1 回）
 *     deno run -A tools/ram-peak/measure.ts --family anima --source models/karume-anima \
 *         --model anima-turbo-v1.1 --quant f16 --steps 2 --size 512
 *     deno run -A tools/ram-peak/measure.ts --family gemma4 --source models/karume-gemma4
 *     # 部品面（容器 1 本を直に開いて Session にする・1 部品だけ）
 *     deno run -A tools/ram-peak/measure.ts --mode component --source models/karume-anima \
 *         --model anima-turbo-v1.1 --quant f16 --component transformer
 *
 * MUST: 1 構成 = 1 プロセス。ピークはプロセス終端で読む（Linux は `/proc/self/status` の
 * VmHWM = 高水位標・Mac は無いので `Deno.memoryUsage().rss` の 50ms サンプリング最大値のみ）。
 * anima は Session を生成時に張る（`withSession`）ので、最小の生成 1 回まで回してから測る。
 * gemma4 は `fromPretrained` で常駐 Session を組むので、組めた時点で測る。
 * 出力は JSON 1 行（研究記録の表はこれを集計する）。
 */

import { denoDirectory } from "../../packages/hub/deno.ts";
import { loadManifest, openContainerSource, resolveSelection } from "../../packages/hub/mod.ts";
import { AnimaPipeline } from "../../packages/models/mod.ts";
import { Gemma4Pipeline } from "../../packages/models/gemma.ts";
import {
  acquireGpu,
  createSessionFromContainer,
  openContainer,
  type SessionBuildStats,
} from "../../packages/runtime/mod.ts";

/**
 * 受けるオプション。MUST: 未知のキーは落とす — `--stpes 4` が黙って既定 2 で走ると、
 * 研究記録の 1 行が「何を測ったか」を偽る（出力に残る条件と実際の条件が食い違う）。
 */
const KNOWN: ReadonlySet<string> = new Set([
  "mode",
  "family",
  "source",
  "model",
  "quant",
  "component",
  "steps",
  "size",
  "gc",
]);

const args = new Map<string, string>();
for (let at = 0; at < Deno.args.length; at += 2) {
  const [key, value] = [Deno.args[at], Deno.args[at + 1]];
  if (!key.startsWith("--") || value === undefined) throw new Error(`引数 ${key} が対でない`);
  const name = key.slice(2);
  if (!KNOWN.has(name)) {
    throw new Error(
      `未知のオプション ${key}（既知: ${[...KNOWN].map((known) => `--${known}`).join(" ")}）`,
    );
  }
  args.set(name, value);
}
const mode = args.get("mode") ?? "pipeline";
const family = args.get("family") ?? "anima";
const source = args.get("source");
if (source === undefined) throw new Error("--source <配布形ディレクトリ> は必須");
const model = args.get("model");
const quant = args.get("quant");
const component = args.get("component") ?? "transformer";
const steps = Number(args.get("steps") ?? "2");
// 診断: Session を組む直前に明示 GC（`deno run --v8-flags=--expose-gc` が前提・無ければ no-op）。
const explicitGc = args.get("gc") === "true";
const size = Number(args.get("size") ?? "512");

const mib = (bytes: number): number => Math.round(bytes / 1048576);
const vmHwm = async (): Promise<number | undefined> => {
  if (Deno.build.os !== "linux") return undefined;
  const row = (await Deno.readTextFile("/proc/self/status")).split("\n").find((line) =>
    line.startsWith("VmHWM:")
  );
  return row === undefined ? undefined : Number(row.replace(/[^0-9]/g, "")) * 1024;
};

/**
 * manifest から (model, component, quant) の容器 1 本を開く（取得面と**同じ読み口** —
 * `openContainerSource` は開くだけでは 1 バイトも読まず、block は Session を組む間に part 順で
 * 読まれる）。
 *
 * MUST: 選択の外れは既知一覧つきで落とす。model / quant は `resolveSelection` が
 * `ManifestReferenceError` に一覧を添えるので、ここで綴り直さない（同じ判定を 2 実装持たない）。
 */
const openComponentContainer = async (): Promise<
  {
    readonly model: string;
    readonly quant: string;
    readonly opened: Awaited<ReturnType<typeof openContainer>>;
  }
> => {
  const loaded = await loadManifest(denoDirectory(source));
  const selected = resolveSelection(loaded.manifest, {
    ...(model === undefined ? {} : { model }),
    ...(quant === undefined ? {} : { quant }),
  });
  const container = selected.containers[component];
  if (container === undefined) {
    throw new Error(
      `${selected.model} / ${selected.quant} に component '${component}' が無い` +
        `（既知: ${Object.keys(selected.containers).join(" / ")}）`,
    );
  }
  const opened = await openContainer(
    { kind: "source", source: openContainerSource(loaded, container) },
    container.descriptor,
  );
  return { model: selected.model, quant: selected.quant, opened };
};

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
  const target = await openComponentContainer();
  resolved = { model: target.model, quant: target.quant };
  const gpu = await acquireGpu();
  try {
    if (explicitGc) (globalThis as { gc?: () => void }).gc?.();
    // グラフ名 = 部品名（書き手の規約 — ADR 0109 決定 8）。
    const session = await createSessionFromContainer(gpu, target.opened, component);
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
  explicitGc,
  // 測定条件（出力 1 行から構成が復元できることが冒頭 doc の名乗り）。anima の生成でしか
  // 効かないノブなので、他の経路では null を書いて「与えていない」と読めるようにする。
  steps: mode === "pipeline" && family === "anima" ? steps : null,
  size: mode === "pipeline" && family === "anima" ? size : null,
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
