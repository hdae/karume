/**
 * gemma の実行オプション（フラグ）を**単体で**相対比較する（Deno・同一 device 系列・ABBA）。
 *
 *     deno run -A tools/flag-bench/main.ts --source models/karume-gemma4-qat --family qat \
 *         --model e2b --quant i4 \
 *         --set ref='{"stateAttentionReduce":"sequential"}' \
 *         --set gemvpar='{"stateAttentionReduce":"sequential","linearGemvReduce":"parallel"}' \
 *         --out outputs/bench/karume-gemma4-qat/<日付>_flags/e2b-wall.jsonl
 *
 * stdout は最後の **summary JSON 1 行**だけで、進捗はすべて stderr に出る（`tools/mtp-bench` の流儀）。
 * `--out` には 1 走行 1 行 + 最終 summary 行の JSONL を書く（既存ファイルは上書きしない）。
 *
 * ## 何をどう測るか
 *
 * - set（`--set` 1 本 = pipeline へ渡す上書きの組）ごとに **fresh な GpuContext + pipeline** を組み、
 *   `cases.json`（`tools/llm-speed/browser/` の EN / JA 固定 prompt）の 2 本 × {暖機 1 + 計測 2} を
 *   greedy・毎回新しい sequence で回して畳む。これを ABBA（S0..Sk → Sk..S0）× `--rounds` 回す。
 *   set ごとに device から組み直すのは、フラグが Session 構築時に固定される静的ノブで、同じ
 *   pipeline の中では切り替えられないため。
 * - 主指標は `--gpu-timing` の GPU 時間（decode ms/step）。Deno の decode 壁時計は約 10 ms/token の
 *   フェンス床を含む（`docs/research/2026-09-19-qat-speed-recon.md` §3.2）ので、GPU 側で縮んだ
 *   ぶんが壁では床に隠れる。壁時計は副指標。
 * - Deno の GPU 時間は ns ではなく device の tick で返る（`docs/known-issues.md` の Arc B570 節 —
 *   B570 は 1 tick = 52.0833 ns で絶対値が約 52 分の 1）。周期は device の定数なので基準比 % は
 *   そのまま読めるが、ms の絶対値は period ≠ 1 の GPU では読まない。
 *
 * ## 読み方の注意
 *
 * - `--gpu-timing` の走行と付けない走行は**別プロセス・別ファイル**にし、数値を混ぜない。計測が
 *   有効な device は 1 dispatch = 1 pass に開くので壁が伸び、さらに観測席（`onRunDiagnostics`）を
 *   渡すと pipeline は GPU 上の greedy 出力経路を組まない（`pipeline.ts` の `#build`）— 計測 ON の
 *   壁は製品経路の壁ではない。観測席は計測 ON のときだけ渡す（OFF の走行は製品と同じ経路）。
 * - 基準は最初の `--set`。gemma 家族の `stateAttentionReduce` 既定は `"parallel"` なので、参照経路を
 *   基準にしたいときは基準の set に `"sequential"` を**明示**する。
 */

import { gemma4ChatPrompt, Gemma4Pipeline } from "../../packages/models/gemma.ts";
import type { Gemma4RunPhase } from "../../packages/models/gemma.ts";
import { Gemma4QatPipeline } from "../../packages/models/gemma4-qat.ts";
import { gemma4StopTokens } from "../../packages/models/src/gemma/text/chat.ts";
import { resolveGemmaSessionOptions } from "../../packages/models/src/gemma/session-options.ts";
import { GEMMA4_STATE_ATTENTION_REDUCE } from "../../packages/models/src/gemma/pipeline.ts";
import { denoDirectory } from "../../packages/hub/deno.ts";
import { MANIFEST_FILENAME, parseManifest } from "../../packages/hub/mod.ts";
import { acquireGpu } from "../../packages/runtime/mod.ts";
import type { GpuContext, SessionDiagnostics } from "../../packages/runtime/mod.ts";
import { record } from "../../examples/shared/llm-tokenizer.ts";
import { runMain } from "../../examples/shared/run-main.ts";
import { type BenchArgs, type FlagSet, needsSubgroups, parseArgs } from "./args.ts";
import {
  abbaOrder,
  addRunTiming,
  emptyRunGpuTally,
  gpuRecord,
  type RunGpuTally,
  type RunRow,
  summarizeSets,
  type Visit,
} from "./summary.ts";

const encoder = new TextEncoder();
const note = (text: string): void => {
  Deno.stderr.writeSync(encoder.encode(text));
};

/** 暖機 1 + 計測 2（visit × prompt ごと）。 */
const MEASURED_REPS = 2;

/** 流用する固定 prompt（新しい fixture は作らない — ブラウザ計測と同じ入力で比べられるように）。 */
const CASES_URL = new URL("../llm-speed/browser/cases.json", import.meta.url);

type Case = {
  readonly name: string;
  readonly prompt: string;
  readonly inputIds: readonly number[];
};
type Cases = { readonly cases: readonly Case[]; readonly stopTokens: readonly number[] };

const tokenIds = (raw: unknown, label: string): number[] => {
  if (
    !Array.isArray(raw) ||
    !raw.every((value): value is number =>
      typeof value === "number" && Number.isSafeInteger(value) && value >= 0
    )
  ) throw new Error(`${label} が非負整数の配列でない`);
  return raw;
};

/** `cases.json`（`unknown` 境界）の読み。使うのは case / prompt / inputIds / stopTokens だけ。 */
const parseCases = (raw: unknown): Cases => {
  const root = record(raw, "cases.json");
  if (!Array.isArray(root.cases) || root.cases.length === 0) {
    throw new Error("cases.json の cases が空でない配列でない");
  }
  const cases = root.cases.map((item, index): Case => {
    const one = record(item, `cases.json cases[${index}]`);
    if (typeof one.case !== "string" || typeof one.prompt !== "string") {
      throw new Error(`cases.json cases[${index}] の case / prompt が文字列でない`);
    }
    return {
      name: one.case,
      prompt: one.prompt,
      inputIds: tokenIds(one.inputIds, `cases.json ${one.case}.inputIds`),
    };
  });
  return { cases, stopTokens: tokenIds(root.stopTokens, "cases.json stopTokens") };
};

const sha256Hex = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
  [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");

const sameIds = (left: readonly number[], right: readonly number[]): boolean =>
  left.length === right.length && left.every((id, index) => id === right[index]);

/**
 * `--out` の親ディレクトリが在ることを確かめ、ファイルを**新規に**作る（測る前に落とす）。
 *
 * 既存ファイルへ追記しないのは、1 ファイル = 1 回の起動 = 1 つの `gpuTiming` 条件に保つため —
 * 計測 ON と OFF の行が 1 ファイルに混ざると、後から集計する側が混ぜてしまえる。
 */
const createOut = (path: string): void => {
  const cut = path.lastIndexOf("/");
  if (cut > 0) {
    const parent = path.slice(0, cut);
    let stat: Deno.FileInfo;
    try {
      stat = Deno.statSync(parent);
    } catch (error) {
      throw new Error(`--out ${path} の親ディレクトリ ${parent} が無い（先に作ること）`, {
        cause: error,
      });
    }
    if (!stat.isDirectory) throw new Error(`--out ${path} の親 ${parent} がディレクトリでない`);
  }
  try {
    Deno.writeTextFileSync(path, "", { createNew: true });
  } catch (error) {
    throw new Error(`--out ${path} を新規に作れない（既存ファイルは上書きしない）`, {
      cause: error,
    });
  }
};

const appendLine = (path: string, value: unknown): void => {
  Deno.writeTextFileSync(path, `${JSON.stringify(value)}\n`, { append: true });
};

/** set の device 要件（計測と subgroup — どちらも device 作成時にしか要求できない）。 */
const deviceOptions = (args: BenchArgs, set: FlagSet): Parameters<typeof acquireGpu>[0] => ({
  ...(args.gpuTiming ? { gpuTiming: true } : {}),
  ...(needsSubgroups(set.overrides) ? { subgroups: true } : {}),
});

const main = async (): Promise<void> => {
  const args = parseArgs(Deno.args);
  createOut(args.out);

  const fixture = parseCases(JSON.parse(await Deno.readTextFile(CASES_URL)));

  // 実効 quant と session 宣言は配布形の manifest から読んで記録する（同定は本文の SHA-256）。
  const manifestText = await Deno.readTextFile(`${args.source}/${MANIFEST_FILENAME}`);
  const manifest = parseManifest(manifestText);
  const entry = Object.hasOwn(manifest.models, args.model)
    ? manifest.models[args.model]
    : undefined;
  if (entry === undefined) {
    throw new Error(`${args.source}: manifest にモデル ${args.model} が無い`);
  }
  const quantEntry = Object.hasOwn(entry.quants, args.quant) ? entry.quants[args.quant] : undefined;
  if (quantEntry === undefined) {
    throw new Error(
      `${args.source}: ${args.model} に quant ${args.quant} が無い（既知: ${
        Object.keys(entry.quants).join(" / ")
      }）`,
    );
  }
  const declaredSession = quantEntry.session;

  // 上書きと宣言の合成は pipeline と**同じ関数**で解く（組合せの誤りはモデルを読む前に落ちる）。
  // stateAttentionReduce は quant 宣言の語彙に無い家族既定なので、同じ既定定数で補う。
  const resolved = new Map(args.sets.map((set) => [
    set.label,
    {
      ...resolveGemmaSessionOptions(
        declaredSession,
        set.overrides,
        `flag-bench --set ${set.label}`,
      ),
      stateAttentionReduce: set.overrides.stateAttentionReduce ?? GEMMA4_STATE_ATTENTION_REDUCE,
    },
  ]));

  // device 要件（subgroup・timestamp-query）を**モデルを読む前に**全て確かめる。持たない
  // アダプタで後ろの set だけが落ちると、前の set を読み込んだ分の時間が無駄になる。
  let adapter: GPUAdapterInfo | undefined;
  for (const set of args.sets) {
    const gpu = await acquireGpu(deviceOptions(args, set));
    try {
      adapter ??= gpu.adapterInfo;
    } finally {
      gpu.destroy();
    }
  }
  if (adapter === undefined) throw new Error("flag-bench: device を 1 度も取れていない");

  const visits = abbaOrder(args.sets.length, args.rounds);
  note(
    `[flag-bench] ${args.family}/${args.model}/${args.quant} · ${args.sets.length} set × ` +
      `${visits.length / args.sets.length} visit · ${fixture.cases.length} prompt × ` +
      `(暖機 1 + 計測 ${MEASURED_REPS}) · new-tokens ${args.newTokens} · ` +
      `gpuTiming ${args.gpuTiming ? "ON" : "OFF"}\n`,
  );

  /** 今走っている generate の GPU 器（計測 ON の走行中だけ入る）。 */
  let current: RunGpuTally | undefined;
  const observeRun = (diagnostics: SessionDiagnostics, phase: Gemma4RunPhase): void => {
    const tally = current;
    if (tally === undefined) {
      throw new Error(`[flag-bench] 走行の外で ${phase.kind} run の観測が届いた`);
    }
    if (phase.kind !== "prefill" && phase.kind !== "decode") {
      throw new Error(`[flag-bench] 非投機の走行に ${phase.kind} run が出た`);
    }
    const stats = diagnostics.lastRunTiming;
    if (stats === undefined) {
      throw new Error("[flag-bench] --gpu-timing を付けたが lastRunTiming が空（device が非対応）");
    }
    addRunTiming(tally, phase.kind, stats);
  };

  const load = (gpu: GpuContext, set: FlagSet): Promise<Gemma4Pipeline | Gemma4QatPipeline> => {
    const options = {
      gpu,
      model: args.model,
      quant: args.quant,
      ...set.overrides,
      ...(args.gpuTiming ? { onRunDiagnostics: observeRun } : {}),
    };
    return args.family === "normal"
      ? Gemma4Pipeline.fromPretrained(denoDirectory(args.source), options)
      : Gemma4QatPipeline.fromPretrained(denoDirectory(args.source), options);
  };

  const rows: RunRow[] = [];
  const features = new Map<string, readonly string[]>();

  const runVisit = async ({ visit, round, setIndex }: Visit): Promise<void> => {
    const set = args.sets[setIndex];
    const gpu = await acquireGpu(deviceOptions(args, set));
    // 解放は宣言の逆順 — pipeline を畳んでから device を壊す（flush-before-destroy）。
    using _gpu = { [Symbol.dispose]: (): void => gpu.destroy() };
    features.set(set.label, [...gpu.features].toSorted());
    const started = performance.now();
    await using pipeline = await load(gpu, set);
    await gpu.device.queue.onSubmittedWorkDone();
    note(
      `[flag-bench] visit ${visit}/${visits.length} (round ${round}) ${set.label}: ready ` +
        `${((performance.now() - started) / 1000).toFixed(1)} s\n`,
    );
    // 固定 prompt の id 列と停止集合が tokenizer と食い違ったら測らない（fixture の陳腐化）。
    const stopTokens = gemma4StopTokens(pipeline.tokenizer).toSorted((a, b) => a - b);
    if (!sameIds(stopTokens, fixture.stopTokens.toSorted((a, b) => a - b))) {
      throw new Error(`cases.json の stopTokens が tokenizer の停止集合 ${stopTokens} と違う`);
    }
    for (const one of fixture.cases) {
      const encoded = gemma4ChatPrompt(pipeline.tokenizer, [{ role: "user", content: one.prompt }]);
      if (!sameIds(encoded, one.inputIds)) {
        throw new Error(`cases.json ${one.name} の inputIds が chat テンプレートの出力と違う`);
      }
    }
    for (const one of fixture.cases) {
      for (let rep = 0; rep <= MEASURED_REPS; rep++) {
        await gpu.device.queue.onSubmittedWorkDone();
        const sequence = await pipeline.sequence({ capacity: args.capacity });
        const tally = args.gpuTiming ? emptyRunGpuTally() : undefined;
        const ids: number[] = [];
        let firstAt = Number.NaN;
        let stopReason: string;
        let startedAt: number;
        let endedAt: number;
        try {
          current = tally;
          // 時計は sequence（KV の確保）の後から — TTFT は prefill と先頭の抽選だけを数える。
          startedAt = performance.now();
          const stream = sequence.generate({
            prompt: one.inputIds,
            maxNewTokens: args.newTokens,
            stopTokens: fixture.stopTokens,
            sampler: { temperature: 0 },
          });
          for await (const event of stream) {
            if (event.kind !== "token") continue;
            if (ids.length === 0) firstAt = performance.now();
            ids.push(event.id);
          }
          stopReason = (await stream.done).reason;
          endedAt = performance.now();
        } finally {
          current = undefined;
          await sequence.dispose();
        }
        if (ids.length < 2) {
          throw new Error(
            `[flag-bench] ${set.label} ${one.name}: 配送 ${ids.length} token（${stopReason}）で ` +
              "decode ms/token の分母が無い",
          );
        }
        const row: RunRow = {
          type: "run",
          set: set.label,
          visit,
          round,
          prompt: one.name,
          rep,
          warmup: rep === 0,
          gpuTiming: args.gpuTiming,
          promptTokens: one.inputIds.length,
          delivered: ids.length,
          stopReason,
          ttftMs: firstAt - startedAt,
          decodeMsPerToken: (endedAt - firstAt) / (ids.length - 1),
          tokensSha256: await sha256Hex(encoder.encode(JSON.stringify(ids))),
          ...(tally === undefined ? {} : { gpu: gpuRecord(tally) }),
        };
        rows.push(row);
        appendLine(args.out, row);
        const gpuDecode = row.gpu?.decode;
        note(
          `[flag-bench]   ${set.label} ${one.name} ${rep === 0 ? "warmup" : `#${rep}`}: ` +
            `${row.delivered} tok · TTFT ${row.ttftMs.toFixed(1)} ms · ` +
            `decode ${row.decodeMsPerToken.toFixed(2)} ms/tok` +
            (gpuDecode === undefined
              ? ""
              : ` · GPU decode ${gpuDecode.msPerRun.toFixed(3)} ms/step · ` +
                `${gpuDecode.dispatchesPerRun.toFixed(1)} dispatch/step`) +
            ` · ${row.tokensSha256.slice(0, 12)}\n`,
        );
      }
    }
  };

  for (const visit of visits) await runVisit(visit);

  const perSet = summarizeSets(rows, args.sets.map((set) => set.label));
  const summary = {
    type: "summary",
    tool: "flag-bench",
    gpuTiming: args.gpuTiming,
    host: {
      os: Deno.build.os,
      arch: Deno.build.arch,
      deno: Deno.version.deno,
      adapter: {
        vendor: adapter.vendor,
        architecture: adapter.architecture,
        device: adapter.device,
        description: adapter.description,
      },
    },
    config: {
      source: args.source,
      family: args.family,
      model: args.model,
      quant: args.quant,
      manifestSha256: await sha256Hex(encoder.encode(manifestText)),
      // quant が宣言する実行ノブ（上書きの無い set はこれ + 家族既定で走る）。
      quantSession: declaredSession,
      rounds: args.rounds,
      newTokens: args.newTokens,
      capacity: args.capacity,
      sampler: { temperature: 0 },
      prompts: fixture.cases.map((one) => ({ case: one.name, tokens: one.inputIds.length })),
      measuredRepsPerPrompt: MEASURED_REPS,
      out: args.out,
    },
    reference: args.sets[0].label,
    sets: perSet.map((one, index) => {
      const set = args.sets[index];
      return {
        ...one,
        overrides: set.overrides,
        // quant 宣言 + 上書きを pipeline と同じ関数で解いた 4 欄 + stateAttentionReduce。
        resolvedSession: resolved.get(set.label),
        deviceFeatures: features.get(set.label),
      };
    }),
  };
  appendLine(args.out, summary);
  console.log(JSON.stringify(summary));

  note(`[flag-bench] 要約（中央値・基準 ${args.sets[0].label}・Δ は基準比 %）:\n`);
  for (const one of perSet) {
    const delta = (value: number | undefined): string =>
      value === undefined ? "" : ` (${value >= 0 ? "+" : ""}${value.toFixed(1)}%)`;
    note(
      `[flag-bench]   ${one.label}: decode ${one.decodeMsPerToken.toFixed(2)} ms/tok` +
        delta(one.deltaPercent.decodeMsPerToken) +
        ` · TTFT ${one.ttftMs.toFixed(1)} ms${delta(one.deltaPercent.ttftMs)}` +
        (one.gpuDecodeMsPerStep === undefined
          ? ""
          : ` · GPU decode ${one.gpuDecodeMsPerStep.toFixed(3)} ms/step` +
            delta(one.deltaPercent.gpuDecodeMsPerStep) +
            ` · ${one.gpuDispatchesPerStep?.toFixed(1)} dispatch/step`) +
        ` · 列 visit 間 ${one.tokensIdenticalAcrossVisits ? "同一" : "不一致"}` +
        ` / 基準と ${one.tokensMatchReference ? "同一" : "不一致"}\n`,
    );
  }
};

await runMain(main);
