/**
 * RAM ピーク harness のランナー — **1 構成 1 プロセス**で cold → warm → local を各 N 回回し、
 * 生の JSON 行と中央値の表を書き出す（ADR 0108 段階分解表 段 2 の検収②③）。
 *
 *     deno run -A tools/ram-peak/matrix.ts --source models/karume-gemma4 --family gemma4 \
 *         --model gemma4-e2b --quant i4 --repeat 3
 *
 * 出力は `outputs/ram-peak/<日付>_<系列>/` に 3 本:
 *
 * - `results.jsonl` … `measure.ts` の出力 1 行 = 1 run（生の記録。研究記録はここから集計する）
 * - `summary.md` … 取得元 3 通りの**中央値**の表（人が読む側）
 * - `cache/` … cold / warm が使うキャッシュ置き場（消して安全）
 *
 * ## 回す順序が契約
 *
 * 1 反復の中で **cold → warm → local** の順に回す。warm は「直前の cold が温めた同じキャッシュ」
 * でなければ warm ではないので、cold と warm を別の反復に散らさない。
 *
 * ## 検収③の読み方
 *
 * `digest` 欄は `全体 (payload)` の 2 数で出す。descriptor 2 文書の突合は cold でも warm でも
 * 必ず掛かる（container-v1 §7 の①）ので、**0 を要求してよいのは payload 側**（資産の全量検証・
 * block ごとの digest）だけである。warm の payload が 0 でない行は `⚠` を付ける — 判定そのものは
 * 人が読む（段階分解表の検収③）。計数の射程は `probes.ts` の `countDigests` の NOTE を読むこと
 * （取得層の cold の part 逐次ハッシュは純 TS 実装なので**この計数には出ない**）。
 */

import { FAMILIES, isFamilyName } from "./families.ts";
import type { MeasureReport, MeasureState } from "./measure.ts";
import { renderSummary } from "./summary.ts";

const KNOWN: ReadonlySet<string> = new Set([
  "source",
  "family",
  "mode",
  "component",
  "model",
  "quant",
  "weights",
  "repeat",
  "states",
  "steps",
  "size",
  "max-new-tokens",
  "out-dir",
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

const source = args.get("source");
if (source === undefined) throw new Error("--source <配布形ディレクトリ> は必須");
const family = args.get("family") ?? "anima";
if (!isFamilyName(family)) {
  throw new Error(`--family ${family} は未対応（${FAMILIES.join(" | ")}）`);
}
const repeat = Number(args.get("repeat") ?? "3");
if (!Number.isSafeInteger(repeat) || repeat < 1) {
  throw new Error(`--repeat は 1 以上の整数（受け取った値: ${args.get("repeat")}）`);
}
const ALL_STATES: readonly MeasureState[] = ["cold", "warm", "local"];
const states: readonly MeasureState[] = (args.get("states") ?? "cold,warm,local")
  .split(",")
  .map((raw) => {
    const state = raw.trim();
    if (!(ALL_STATES as readonly string[]).includes(state)) {
      throw new Error(`--states の ${state} は未対応（${ALL_STATES.join(" | ")}）`);
    }
    return state as MeasureState;
  });

const mode = args.get("mode") ?? "pipeline";
const label = mode === "component" ? `${family}-${args.get("component") ?? "transformer"}` : family;
const outDir = args.get("out-dir") ??
  `outputs/ram-peak/${new Date().toISOString().slice(0, 10)}_${label}`;
const cacheDir = `${outDir}/cache`;
await Deno.mkdir(outDir, { recursive: true });

/** 1 構成ぶんの起動引数（`measure.ts` の CLI をそのまま呼ぶ）。 */
const measureUrl = new URL("./measure.ts", import.meta.url).href;
const commandArgs = (state: MeasureState): string[] => [
  "run",
  "-A",
  measureUrl,
  "--mode",
  mode,
  "--state",
  state,
  "--family",
  family,
  "--source",
  source,
  ...(state === "local" ? [] : ["--cache-dir", cacheDir]),
  ...(args.get("component") === undefined ? [] : ["--component", args.get("component") as string]),
  ...(args.get("model") === undefined ? [] : ["--model", args.get("model") as string]),
  ...(args.get("quant") === undefined ? [] : ["--quant", args.get("quant") as string]),
  ...(args.get("weights") === undefined ? [] : ["--weights", args.get("weights") as string]),
  ...(args.get("steps") === undefined ? [] : ["--steps", args.get("steps") as string]),
  ...(args.get("size") === undefined ? [] : ["--size", args.get("size") as string]),
  ...(args.get("max-new-tokens") === undefined
    ? []
    : ["--max-new-tokens", args.get("max-new-tokens") as string]),
];

const decoder = new TextDecoder();
const resultsPath = `${outDir}/results.jsonl`;
const results: MeasureReport[] = [];

/** 1 構成を別プロセスで測る。**落ちたら止める** — 欠けた行のまま表を作らない。 */
const runOnce = async (state: MeasureState, round: number): Promise<MeasureReport> => {
  console.error(`[ram-peak] ${label} / ${state} / ${round + 1} 回目`);
  const command = new Deno.Command(Deno.execPath(), {
    args: commandArgs(state),
    stdout: "piped",
    stderr: "inherit",
  });
  const { code, stdout } = await command.output();
  const text = decoder.decode(stdout).trim();
  if (code !== 0) {
    throw new Error(`ram-peak: ${state} の計測が終了コード ${code} で落ちた（stderr を見ること）`);
  }
  const report = JSON.parse(text) as MeasureReport;
  await Deno.writeTextFile(resultsPath, `${JSON.stringify(report)}\n`, { append: true });
  return report;
};

for (let round = 0; round < repeat; round += 1) {
  // MUST: 1 反復の中で cold → warm の順（warm は直前の cold が温めたキャッシュを読む）。
  for (const state of states) results.push(await runOnce(state, round));
}

const summary = renderSummary(results, { label, source, repeat, states, resultsPath });
await Deno.writeTextFile(`${outDir}/summary.md`, summary);
console.log(summary);
