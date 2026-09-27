/**
 * anima の段ごとの op 別 GPU 時間を Deno で採る CLI（確認ページ `browser/` の双子 — perf-ledger K-70）。
 *
 * ページと同じ記録器（`record.ts`）と集計（`timing.ts`）で、ページの「JSON を保存」と**同じ形**
 * （`karume-anima-residency-browser/2`）の JSON を書く。違いは `userAgent` が
 * `{ deno: Deno.version.deno }` になること、bundle が無いので `bundleSha256` が無いこと、ダミー確保を
 * しないこと、`gpuTiming.unit` が `"deno-raw-tick"` になること（Deno は timestamp を ns へ換算
 * しない — docs/known-issues.md「Intel Arc B570」節。B570 では 1 tick = 52.08 ns）。
 *
 * pipeline は `residency: "transformer"` で 1 度だけ組み、`--count` 回 generate する（1 回目が常駐
 * DiT を作るので、2 回目以降が常駐の効いた定常の形）。アダプタが `timestamp-query` を列挙すれば
 * `acquireGpu({ gpuTiming: true })` で取り、列挙しなければ計測なし（`gpuTiming.enabled = false`）で
 * 段の壁時計と dispatch 本数だけ残す。
 *
 * 使い方（リポ直下から・GPU が学習で使われていないことを先に `outputs/diag/gpu-busy.zsh` で確認）:
 *
 *   deno run -A tools/anima-residency/profile.ts --resolution 512x512 --count 2 --quant f16
 *
 * フラグ: `--source`（既定 models/karume-anima）`--quant`（既定 = 配布形の defaultQuant）
 * `--resolution`（既定 1024x1024）`--count`（既定 2）`--seed`（既定 42）`--prompt`（既定 = ページと
 * 同じ）`--out`（既定 outputs/bench/karume/<日付>_metal-recon）`--date`（`--out` 既定の日付・既定は
 * 今日のローカル日付）。
 */
import { acquireGpu } from "../../packages/runtime/mod.ts";
import { parseManifest } from "../../packages/hub/mod.ts";
import { denoDirectory } from "../../packages/hub/deno.ts";
import {
  type AnimaGenerateEvent,
  AnimaPipeline,
  parseResolution,
} from "../../packages/models/anima.ts";
import { encodePng } from "../../packages/models/mod.ts";
import { readCheckout } from "./browser/server.ts";
import {
  createGenerateRecorder,
  DEFAULT_PROMPT,
  type GenerateRecorder,
  type PipelineLoad,
  type Report,
  REPORT_FORMAT,
  type Row,
} from "./record.ts";
import { topEntries } from "./timing.ts";

const TIMESTAMP_QUERY = "timestamp-query";

const FLAG_DEFAULTS: Readonly<Record<string, string | undefined>> = {
  source: "models/karume-anima",
  quant: undefined,
  resolution: "1024x1024",
  count: "2",
  seed: "42",
  prompt: DEFAULT_PROMPT,
  out: undefined,
  date: undefined,
};

/** `--key value` の対だけを受ける（未知のキーは落とす — 綴り違いが既定で走って記録と食い違わない）。 */
const parseFlags = (argv: readonly string[]): Record<string, string | undefined> => {
  const flags = { ...FLAG_DEFAULTS };
  for (let at = 0; at < argv.length; at += 2) {
    const key = argv[at].startsWith("--") ? argv[at].slice(2) : undefined;
    const value = argv[at + 1];
    if (
      key === undefined || !Object.hasOwn(FLAG_DEFAULTS, key) || value === undefined ||
      value.startsWith("--")
    ) {
      throw new Error(`引数を読めない: ${argv.slice(at, at + 2).join(" ")}`);
    }
    flags[key] = value;
  }
  return flags;
};

const positiveInteger = (name: string, spelled: string | undefined, min: number): number => {
  const value = Number(spelled);
  if (!Number.isInteger(value) || value < min) {
    throw new Error(`--${name} は ${min} 以上の整数（${spelled}）`);
  }
  return value;
};

const sha256Hex = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
  Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");

const describeError = (error: unknown): { name: string; message: string } =>
  error instanceof Error
    ? { name: error.name, message: error.message }
    : { name: "Error", message: String(error) };

/**
 * ファイル名に載せるアダプタの名前（小文字の英数とハイフン）。architecture が空なら description を
 * 使う — Deno（wgpu）は vendor を PCI ID の 10 進（Intel = 32902）で返し architecture を空にするので、
 * vendor-architecture では機体が読めない。
 */
const adapterSlug = (info: GPUAdapterInfo): string => {
  const parts = info.architecture === ""
    ? [info.description === "" ? info.vendor : info.description]
    : [info.vendor, info.architecture];
  const slug = parts.join("-").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  return slug === "" ? "unknown-adapter" : slug;
};

const localDate = (): string => {
  const now = new Date();
  const pad = (value: number): string => String(value).padStart(2, "0");
  return `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`;
};

const ms = (value: number): string => value.toFixed(1).padStart(10);

/** generate 1 回の要約（段の壁 / GPU と DiT 段の上位 5 キー）。 */
const printRow = (row: Row, count: number, unit: Report["gpuTiming"]["unit"]): void => {
  const residency = row.residency.map(({ action, reason }) => `${action}/${reason}`).join(", ");
  console.log(
    `\n[profile] generate ${row.index}/${count} · quant ${row.quant} · 壁 ${
      Math.round(row.wallMs)
    } ms · sha ${row.pngSha256?.slice(0, 12) ?? "—"} · residency [${residency}]`,
  );
  if (row.error !== undefined) {
    console.log(`[profile] 失敗: ${row.error.name}: ${row.error.message}`);
  }
  const gpuLabel = unit === "ns" ? "GPU ms" : "GPU ms(raw)";
  console.log(`  ${"stage".padEnd(18)}${"wall ms".padStart(10)}${gpuLabel.padStart(12)}  runs`);
  for (const stage of row.stages) {
    const wall = stage.endMs === undefined
      ? "(未完了)".padStart(10)
      : ms(stage.endMs - stage.startMs);
    const gpu = stage.gpu === undefined
      ? "—".padStart(12)
      : ms(stage.gpu.totalNs / 1e6).padStart(12);
    console.log(`  ${stage.component.padEnd(18)}${wall}${gpu}  ${stage.gpu?.runs ?? "—"}`);
  }
  const dit = row.stages.findLast((stage) => stage.component === "transformer");
  if (dit?.gpu === undefined) return;
  console.log(`  DiT 上位 5 キー（${gpuLabel}・dispatch・段に占める %）:`);
  for (const entry of topEntries(dit.gpu, 5)) {
    console.log(
      `    ${ms(entry.ns / 1e6)}  ${String(entry.dispatchCount).padStart(7)}  ${
        (entry.share * 100).toFixed(1).padStart(5)
      }%  ${entry.key}`,
    );
  }
  if (unit === "deno-raw-tick" && dit.endMs !== undefined && dit.gpu.totalNs > 0) {
    // 単位の目安: GPU が律速の DiT 段では「壁 ÷ raw 合計」が 1 tick の ns に近い（B570 は 52.08）。
    const ratio = (dit.endMs - dit.startMs) * 1e6 / dit.gpu.totalNs;
    console.log(`  （Deno の raw tick — DiT 段の壁 ÷ GPU raw 合計 = ${ratio.toFixed(2)}）`);
  }
};

const main = async (): Promise<void> => {
  const flags = parseFlags(Deno.args);
  const sourceRoot = await Deno.realPath(flags.source ?? "models/karume-anima");
  const manifestBytes = await Deno.readFile(`${sourceRoot}/karume.json`);
  const manifest = parseManifest(new TextDecoder().decode(manifestBytes));
  const model = manifest.models[manifest.defaultModel];
  if (model === undefined) {
    throw new Error(`defaultModel ${manifest.defaultModel} が models に無い`);
  }
  const quantName = flags.quant ?? model.defaultQuant;
  const quant = model.quants[quantName];
  if (quant === undefined) {
    throw new Error(
      `--quant ${quantName} が ${manifest.defaultModel} に無い（${
        Object.keys(model.quants).join(" / ")
      }）`,
    );
  }
  const resolution = parseResolution(flags.resolution ?? "");
  const count = positiveInteger("count", flags.count, 1);
  const seed = positiveInteger("seed", flags.seed, 0);
  const prompt = flags.prompt ?? DEFAULT_PROMPT;
  const outDir = flags.out ?? `outputs/bench/karume/${flags.date ?? localDate()}_metal-recon`;

  const adapter = await navigator.gpu.requestAdapter();
  if (adapter === null) throw new Error("WebGPU アダプタが無い");
  const timestampFeature = adapter.features.has(TIMESTAMP_QUERY);
  let deviceLost: Report["deviceLost"] = null;
  const gpu = await acquireGpu({
    ...(timestampFeature ? { gpuTiming: true } : {}),
    // 共有 GPU には pipeline が feature を足せないので、quant の宣言（shader-f16）はここで要求する。
    ...(quant.gpuFeatures?.shaderF16 === true ? { shaderF16: true } : {}),
    onDeviceLost: (info) => {
      deviceLost = { reason: info.reason, message: info.message };
    },
  });
  const rows: Row[] = [];
  const pipelineLoads: PipelineLoad[] = [];
  const unit = "deno-raw-tick";
  try {
    let recorder: GenerateRecorder | undefined;
    const loadStarted = performance.now();
    const pipeline = await AnimaPipeline.fromPretrained(denoDirectory(sourceRoot), {
      gpu,
      residency: "transformer",
      quant: quantName,
      onRunDiagnostics: (component, diagnostics) => {
        if (recorder === undefined) {
          throw new Error(`${component} の run が generate の外で終わった`);
        }
        recorder.onRun(component, diagnostics);
      },
    });
    pipelineLoads.push({
      at: new Date().toISOString(),
      ms: performance.now() - loadStarted,
      quant: quantName,
      gpuTiming: gpu.gpuTimingEnabled,
    });
    console.log(
      `[profile] ${gpu.adapterInfo.vendor} ${gpu.adapterInfo.architecture} · quant ${quantName} · ${resolution.width}x${resolution.height} · GPU 時間 ${
        gpu.gpuTimingEnabled ? "採る" : `採らない（${TIMESTAMP_QUERY} 無し）`
      } · pipeline 構築 ${(pipelineLoads[0].ms / 1000).toFixed(2)} s`,
    );
    try {
      for (let index = 1; index <= count; index += 1) {
        const current = createGenerateRecorder(() => performance.now());
        recorder = current;
        const request = { prompt, resolution, seed };
        const base = {
          index,
          quant: quantName,
          residencyRequested: "transformer" as const,
          request,
          dummyBytesHeld: 0,
        };
        let row: Row;
        try {
          const image = await pipeline.generate({
            ...request,
            residency: "transformer",
            onEvent: (event: AnimaGenerateEvent) => current.onEvent(event),
          });
          const wallMs = current.elapsedMs();
          const png = await encodePng(image.data, image.width, image.height);
          row = { ...base, wallMs, ...current.finish(), pngSha256: await sha256Hex(png) };
        } catch (error) {
          row = {
            ...base,
            wallMs: current.elapsedMs(),
            ...current.finish(),
            error: describeError(error),
          };
        } finally {
          recorder = undefined;
        }
        rows.push(row);
        printRow(row, count, unit);
        if (row.error !== undefined) break;
      }
    } finally {
      await pipeline.dispose();
    }
  } finally {
    const info = gpu.adapterInfo;
    const report: Report = {
      format: REPORT_FORMAT,
      date: new Date().toISOString(),
      userAgent: { deno: Deno.version.deno },
      adapter: {
        vendor: info.vendor,
        architecture: info.architecture,
        device: info.device,
        description: info.description,
      },
      ...await readCheckout().then(
        ({ revision, dirty }) => ({ checkout: revision, checkoutDirty: dirty }),
      ),
      source: sourceRoot.slice(sourceRoot.lastIndexOf("/") + 1),
      manifestSha256: await sha256Hex(manifestBytes),
      defaultModel: manifest.defaultModel,
      quant: quantName,
      gpuTiming: { enabled: gpu.gpuTimingEnabled, feature: timestampFeature, unit },
      pipelineResidency: "transformer",
      pipelineLoads,
      dummies: { heldBytes: 0, buffers: 0, holds: [] },
      deviceLost,
      rows,
    };
    gpu.destroy();
    await Deno.mkdir(outDir, { recursive: true });
    const stamp = report.date.replaceAll(":", "-");
    const path = `${outDir}/anima-profile-${
      adapterSlug(info)
    }-${quantName}-${resolution.width}x${resolution.height}-${stamp}.json`;
    await Deno.writeTextFile(path, `${JSON.stringify(report, null, 2)}\n`);
    console.log(`\n[profile] ${path}`);
  }
  if (rows.some((row) => row.error !== undefined)) Deno.exit(1);
};

if (import.meta.main) await main();
