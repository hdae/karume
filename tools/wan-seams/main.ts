/**
 * Wan2.2 TI2V 5B の DiT のパスの切れ目（ホストの計算・入力の上げ・出力の読み戻し）が 1 step に占める割合を Deno で
 * 測る CLI（perf-ledger H-37 段 1 — 確認ページ `tools/gpu-lab` の Wan のタブの「切れ目の下見」と切れ目の列の双子）。
 *
 * フレーム数ごとに 2 つを採る:
 *
 * 1. **下見**（`tools/gpu-lab/browser/wan-seam-probe.ts` の `probeWanSeams`）: 生成とは別に、1 step のホストの処理を
 *    製品の関数で実寸のまま測り、DiT 1 パスと同じ本数・バイト数の上げと読み戻しを素の WebGPU で測って、1 step
 *    （CFG 2 パス）の切れ目を見積もる。
 * 2. **生成の中の区間**: テキストだけ（T2V）で `generate` を回し、タブと同じ印（`wan-plan.ts` の `wanEventMark` /
 *    `wanRunMark`）を採って `summarizeWanTimeline` / `summarizeWanSeams` で集計する。生成の後に transformer の
 *    `planBacking.inputBytes` と下見の入力の合計を照合する（`checkWanSeamInputBytes` — 食い違いは出力で目立たせる）。
 *
 * GPU はタブと同じ方針で取る（`acquireGpu` が requiredLimits にアダプタ値を要求する・`gpuTiming` は要求しない —
 * Wan の家族 admission は計測の device を拒む）。pipeline は 1 度だけ組み（`WanTi2vPipeline.fromPretrained`・席は
 * 解決した名前を明示・テキストエンコーダの経路はパイプラインの既定 `gpu`）、フレーム数ごとに generate する。umT5 は
 * Wan の manifest の `text_encoder` が越境参照する repo を、手元のミラー `models/karume-umt5-xxl` へ結ぶ
 * （`examples/wan/main.ts` の開き方と同じ）。動画の画素は書かない。
 *
 * 使い方（リポ直下から・GPU が空いていることは呼び手が先に確かめる）:
 *
 *   deno run -A tools/wan-seams/main.ts --frames 33,121 --steps 5
 *
 * フラグ: `--source`（既定 models/karume-wan2.2）`--quant`（既定 = 配布形の defaultQuant）`--frames`（既定 33,121 —
 * カンマ区切り）`--resolution`（既定 1280x704）`--steps`（既定 5 — step 1 は Session の構築を含むので、区間の中央値は
 * step 2 以降）`--seed`（既定 42）`--prompt`（既定 boxing-cats = タブの既定 — 埋め込み資産の名前で、その原文を渡す。
 * 名前に無ければ落とす）`--out`（既定 outputs/bench/karume-wan2.2/<今日のローカル日付>_h37-seams）。未知のフラグは
 * 落とす。
 *
 * 書くもの: `<out>/wan-seams-<quant>-<WxH>-<日時>.json`（形式 `karume-wan-seams/1`）。
 */
import { acquireGpu, type SessionDiagnostics } from "../../packages/runtime/mod.ts";
import {
  type DistributionSource,
  parseManifest,
  resolveSelection,
} from "../../packages/hub/mod.ts";
import { denoDirectory } from "../../packages/hub/deno.ts";
import { type WanRunComponent, WanTi2vPipeline } from "../../packages/models/wan.ts";
import { runMain } from "../../examples/shared/run-main.ts";
import { readCheckout } from "../_shared/checkout.ts";
import {
  medianOf,
  parseWanSize,
  summarizeWanDiagnostics,
  summarizeWanSeams,
  summarizeWanTimeline,
  type WanComponentDiagnostics,
  wanEventMark,
  wanRunMark,
  type WanSeams,
  type WanTimeline,
  type WanTimelineMark,
} from "../gpu-lab/browser/wan-plan.ts";
import {
  checkWanSeamInputBytes,
  formatWanSeamProbe,
  probeWanSeams,
  type WanSeamInputCheck,
  type WanSeamProbe,
  wanSeamSizes,
} from "../gpu-lab/browser/wan-seam-probe.ts";

const REPORT_FORMAT = "karume-wan-seams/1";

/** umT5 の手元のミラー（Wan の manifest の `text_encoder` の越境先）。 */
const UMT5_SOURCE = "models/karume-umt5-xxl";

const FLAG_DEFAULTS: Readonly<Record<string, string | undefined>> = {
  source: "models/karume-wan2.2",
  quant: undefined,
  frames: "33,121",
  resolution: "1280x704",
  steps: "5",
  seed: "42",
  prompt: "boxing-cats",
  out: undefined,
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

const integerAtLeast = (name: string, spelled: string | undefined, min: number): number => {
  const value = Number(spelled);
  if (spelled === undefined || !/^\d+$/.test(spelled) || value < min) {
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

const localDate = (): string => {
  const now = new Date();
  const pad = (value: number): string => String(value).padStart(2, "0");
  return `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`;
};

/** フレーム数 1 本の記録（JSON の `rows[]`）。 */
type Row = {
  readonly frames: number;
  readonly width: number;
  readonly height: number;
  readonly steps: number;
  readonly seed: number;
  readonly prompt: string;
  readonly probe: WanSeamProbe;
  readonly wallMs: number;
  readonly timeline?: WanTimeline;
  readonly seams?: WanSeams;
  readonly seamInputCheck?: WanSeamInputCheck;
  readonly diagnostics: Partial<Record<WanRunComponent, WanComponentDiagnostics>>;
  readonly error?: { readonly name: string; readonly message: string };
};

const fixed = (value: number | undefined): string => value?.toFixed(1) ?? "—";

/** 1 行の要約（区間の中央値は step 2 以降・見積もりは T2V）。 */
const printRow = (row: Row): void => {
  const rest = row.timeline?.stepMs.slice(1) ?? [];
  const stepMedian = rest.length === 0 ? undefined : medianOf(rest);
  const stageMs = row.timeline?.stageMs.transformer;
  const median = row.seams?.median;
  console.log(
    `[wan-seams] ${row.frames}f ${row.width}x${row.height} · 壁 ${
      (row.wallMs / 1000).toFixed(1)
    } s · transformer 段 ${
      stageMs === undefined ? "—" : (stageMs / 1000).toFixed(1)
    } s · step 中央値 ${fixed(stepMedian)} ms（step 2 以降）`,
  );
  console.log(
    `  区間の中央値 segA ${fixed(median?.segAMs)} · segB ${fixed(median?.segBMs)} · segC ${
      fixed(median?.segCMs)
    } ms（run ${row.seams?.runsPerStep ?? "—"} 本 / step）· VAE 貼り合わせ ${
      fixed(row.seams?.vaeAssembleMs)
    } ms`,
  );
  const estimate = row.probe.estimate.perStepT2vMs;
  console.log(
    `  下見の見積もり T2V ${estimate.toFixed(1)} ms${
      stepMedian === undefined
        ? ""
        : `（step 中央値の ${(estimate / stepMedian * 100).toFixed(2)}%）`
    }`,
  );
  const check = row.seamInputCheck;
  if (check !== undefined) {
    console.log(
      `  入力の照合 ${check.matches ? "一致" : "!!! 不一致 !!!"}（Session ${
        check.planInputBytes.toLocaleString("en-US")
      } B · 下見の寸法 ${
        check.expectedBytes.toLocaleString("en-US")
      } B · 許容 ${check.toleranceBytes} B）`,
    );
  }
  if (row.error !== undefined) {
    console.log(`  失敗: ${row.error.name}: ${row.error.message}`);
  }
};

/**
 * 取得元（ローカルの配布形）。umT5 が越境参照なら、その repo（manifest の宣言から引く — 名前を写経しない）を
 * 手元のミラーへ結ぶ。ミラーが無ければ取得の前に落とす。
 */
const openSource = async (
  sourceRoot: string,
  manifest: ReturnType<typeof parseManifest>,
): Promise<DistributionSource> => {
  const repo = resolveSelection(manifest).containers["text_encoder"]?.parts[0]?.repo;
  if (repo === undefined) return denoDirectory(sourceRoot);
  try {
    await Deno.stat(`${UMT5_SOURCE}/karume.json`);
  } catch (error) {
    throw new Error(`umT5 の配布形 ${UMT5_SOURCE}（${repo} の越境先）に karume.json が無い`, {
      cause: error,
    });
  }
  return denoDirectory(sourceRoot, { crossRepo: { [repo]: denoDirectory(UMT5_SOURCE) } });
};

const main = async (): Promise<void> => {
  const flags = parseFlags(Deno.args);
  const sourceRoot = await Deno.realPath(flags.source ?? "models/karume-wan2.2");
  const manifestBytes = await Deno.readFile(`${sourceRoot}/karume.json`);
  const manifest = parseManifest(new TextDecoder().decode(manifestBytes));
  const model = manifest.models[manifest.defaultModel];
  if (model === undefined) {
    throw new Error(`defaultModel ${manifest.defaultModel} が models に無い`);
  }
  const quant = flags.quant ?? model.defaultQuant;
  if (!Object.hasOwn(model.quants, quant)) {
    throw new Error(
      `--quant ${quant} が ${manifest.defaultModel} に無い（${
        Object.keys(model.quants).join(" / ")
      }）`,
    );
  }
  const frameCounts = (flags.frames ?? "").split(",").map((spelled) =>
    integerAtLeast("frames", spelled.trim(), 1)
  );
  const size = parseWanSize(flags.resolution ?? "");
  const steps = integerAtLeast("steps", flags.steps, 1);
  const seed = integerAtLeast("seed", flags.seed, 0);
  const promptArg = flags.prompt ?? "";
  // 寸法の綴りは GPU を取る前に落とす（下見の寸法の門 — 4n+1・32 の倍数）。
  for (const frames of frameCounts) wanSeamSizes({ frames, ...size });
  const outDir = flags.out ?? `outputs/bench/karume-wan2.2/${localDate()}_h37-seams`;
  const source = await openSource(sourceRoot, manifest);
  // 記録の材料は GPU を取る前に読む（finally の中で読むと、その失敗が生成の失敗を上書きする）。
  const checkout = await readCheckout();
  const manifestSha256 = await sha256Hex(manifestBytes);

  let deviceLost: { readonly reason: string; readonly message: string } | null = null;
  const gpu = await acquireGpu({
    onDeviceLost: (info) => {
      deviceLost = { reason: info.reason, message: info.message };
    },
  });
  const rows: Row[] = [];
  let loadMs: number | undefined;
  try {
    let recording:
      | {
        readonly marks: WanTimelineMark[];
        readonly diagnostics: Map<WanRunComponent, SessionDiagnostics>;
      }
      | undefined;
    const loadStarted = performance.now();
    const pipeline = await WanTi2vPipeline.fromPretrained(source, {
      gpu,
      quant,
      onRunDiagnostics: (component, diagnostics) => {
        // 時刻は先頭で採る（タブと同じ — 下の検査や記録の時間を区間へ入れない）。
        const ms = performance.now();
        if (recording === undefined) {
          throw new Error(`${component} の run が generate の外で終わった`);
        }
        const mark = wanRunMark(component, ms);
        if (mark !== undefined) recording.marks.push(mark);
        recording.diagnostics.set(component, diagnostics);
      },
    });
    loadMs = performance.now() - loadStarted;
    console.log(
      `[wan-seams] ${
        gpu.adapterInfo.description === "" ? gpu.adapterInfo.vendor : gpu.adapterInfo.description
      } · quant ${quant} · ${size.width}x${size.height} · ${steps} step · pipeline 構築 ${
        (loadMs / 1000).toFixed(2)
      } s`,
    );
    try {
      const prompt = pipeline.prompts.find((candidate) => candidate.name === promptArg)?.prompt;
      if (prompt === undefined) {
        // 綴り違いの名前を文字どおりのプロンプトとして回すと、記録の名前と符号化した文が食い違う
        throw new Error(
          `--prompt ${promptArg} が埋め込み資産の名前に無い（${
            pipeline.prompts.map((candidate) => candidate.name).join(" / ")
          }）`,
        );
      }
      const encoder = new TextEncoder();
      for (const frames of frameCounts) {
        console.log(`[wan-seams] ${frames}f の下見を測定中`);
        const probe = await probeWanSeams(gpu.device, { frames, ...size });
        console.log(`[wan-seams] 下見 ${formatWanSeamProbe(probe)}`);
        const marks: WanTimelineMark[] = [];
        const current = { marks, diagnostics: new Map<WanRunComponent, SessionDiagnostics>() };
        recording = current;
        const base = {
          frames,
          width: size.width,
          height: size.height,
          steps,
          seed,
          prompt: promptArg,
          probe,
        };
        const summaries = (): Row["diagnostics"] =>
          Object.fromEntries(
            [...current.diagnostics].map((
              [component, value],
            ) => [component, summarizeWanDiagnostics(value)]),
          );
        const started = performance.now();
        let row: Row;
        try {
          await pipeline.generate({
            prompt,
            seed,
            frames,
            width: size.width,
            height: size.height,
            steps,
            onEvent: (event) => {
              const mark = wanEventMark(event, performance.now());
              if (mark !== undefined) current.marks.push(mark);
              if (event.kind === "denoise-step") {
                Deno.stderr.writeSync(encoder.encode(`\r  denoise ${event.step}/${event.steps}  `));
              } else if (event.kind === "vae-tile") {
                Deno.stderr.writeSync(
                  encoder.encode(`\r  vae tile ${event.tile}/${event.tiles}  `),
                );
              }
            },
          });
          const wallMs = performance.now() - started;
          Deno.stderr.writeSync(encoder.encode("\n"));
          const transformer = current.diagnostics.get("transformer");
          if (transformer === undefined) throw new Error("transformer の run の診断が来なかった");
          row = {
            ...base,
            wallMs,
            timeline: summarizeWanTimeline(current.marks),
            seams: summarizeWanSeams(current.marks),
            seamInputCheck: checkWanSeamInputBytes(
              probe.sizes,
              transformer.planBacking.inputBytes,
            ),
            diagnostics: summaries(),
          };
        } catch (error) {
          // 失敗した生成の行では、並びの崩れで生成そのものの失敗を上書きしない — 内訳なしで残す。
          const attempt = <T>(summarize: () => T): T | undefined => {
            try {
              return summarize();
            } catch {
              return undefined;
            }
          };
          const timeline = attempt(() => summarizeWanTimeline(current.marks));
          const seams = attempt(() => summarizeWanSeams(current.marks));
          row = {
            ...base,
            wallMs: performance.now() - started,
            ...(timeline === undefined ? {} : { timeline }),
            ...(seams === undefined ? {} : { seams }),
            diagnostics: summaries(),
            error: describeError(error),
          };
        } finally {
          recording = undefined;
        }
        rows.push(row);
        printRow(row);
        if (row.error !== undefined) break;
      }
    } finally {
      await pipeline.dispose();
    }
  } finally {
    const info = gpu.adapterInfo;
    const date = new Date().toISOString();
    const report = {
      format: REPORT_FORMAT,
      date,
      environment: {
        deno: Deno.version.deno,
        adapter: {
          vendor: info.vendor,
          architecture: info.architecture,
          device: info.device,
          description: info.description,
        },
      },
      checkout: checkout.revision,
      checkoutDirty: checkout.dirty,
      source: sourceRoot.slice(sourceRoot.lastIndexOf("/") + 1),
      manifestSha256,
      defaultModel: manifest.defaultModel,
      quant,
      textEncoder: "gpu",
      ...(loadMs === undefined ? {} : { loadMs }),
      deviceLost,
      rows,
    };
    gpu.destroy();
    await Deno.mkdir(outDir, { recursive: true });
    const path = `${outDir}/wan-seams-${quant}-${size.width}x${size.height}-${
      date.replaceAll(":", "-")
    }.json`;
    await Deno.writeTextFile(path, `${JSON.stringify(report, null, 2)}\n`);
    console.log(`[wan-seams] ${path}`);
  }
  if (rows.some((row) => row.error !== undefined || row.seamInputCheck?.matches === false)) {
    Deno.exit(1);
  }
};

if (import.meta.main) await runMain(main);
