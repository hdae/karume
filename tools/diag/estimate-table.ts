// 容器 1 本の見積り（`PreparedModel.estimate`）の `workspaceBytes` と `peakAccountedBytes` を 1 行で出す診断の台本 —
// research の見積りの表（例 docs/research/2026-10-11-fused-attention-estimate-before-after.md）を手元で再計算するため。
// GPU にも重みにも触れない（グラフ宣言だけを開く）。
//
// 使い方:
//   deno run -A tools/diag/estimate-table.ts <容器の代表 path（…/model.krm — part 連番は自動で解く）> [--graph <名前>]
//     [--bind <記号>=<値> ...]
//     [--binding-limit <B>] [--buffer-limit <B>] [--budget <B>] [--compute f32|f16|a8] [--score f32|f16]
//
// `--binding-limit` / `--buffer-limit` は `EstimateOptions.maxStorageBufferBindingSize` / `maxBufferSize`、`--budget` は
// `attentionRowBlockMaxBytes`、`--compute` / `--score` は `attentionCompute` / `attentionScoreStorage`。省略した欄は
// 見積りに渡さない（runtime の既定で数える）。容器に graph が 1 本だけなら `--graph` は省ける。

import { type EstimateOptions, prepareContainer } from "../../packages/runtime/mod.ts";
import { openSeriesContainer } from "../../packages/runtime/tests/helpers/container-files.ts";

type Mutable<T> = { -readonly [K in keyof T]: T[K] };

const integerOf = (flag: string, text: string | undefined): number => {
  const value = Number(text);
  if (text === undefined || !Number.isSafeInteger(value)) {
    throw new Error(`${flag} には整数を渡す（${text ?? "値なし"}）`);
  }
  return value;
};

const main = async (args: readonly string[]): Promise<void> => {
  const [path, ...rest] = args;
  if (path === undefined || path.startsWith("--")) {
    throw new Error("最初の引数に容器の代表 path（…/model.krm）を渡す");
  }
  let graph: string | undefined;
  const bindings: Record<string, number> = {};
  const options: Mutable<EstimateOptions> = {};
  for (let index = 0; index < rest.length; index += 2) {
    const flag = rest[index];
    const value = rest[index + 1];
    if (flag === "--graph") graph = value;
    else if (flag === "--bind") {
      const [symbol, size] = value?.split("=") ?? [];
      if (symbol === undefined || symbol === "") {
        throw new Error(`--bind は <記号>=<値>（${value}）`);
      }
      bindings[symbol] = integerOf("--bind", size);
    } else if (flag === "--binding-limit") {
      options.maxStorageBufferBindingSize = integerOf(flag, value);
    } else if (flag === "--buffer-limit") options.maxBufferSize = integerOf(flag, value);
    else if (flag === "--budget") options.attentionRowBlockMaxBytes = integerOf(flag, value);
    else if (flag === "--compute" && (value === "f32" || value === "f16" || value === "a8")) {
      options.attentionCompute = value;
    } else if (flag === "--score" && (value === "f32" || value === "f16")) {
      options.attentionScoreStorage = value;
    } else throw new Error(`未知の引数 ${flag} ${value ?? ""}`);
  }
  const opened = await openSeriesContainer(new URL(path, `file://${Deno.cwd()}/`));
  const names = Object.keys(opened.graphs);
  const name = graph ?? (names.length === 1 ? names[0] : undefined);
  if (name === undefined) throw new Error(`--graph で選ぶ（在るのは ${names.join(" / ")}）`);
  if (Object.keys(bindings).length > 0) options.bindings = bindings;
  const report = prepareContainer(opened, name).estimate(options);
  for (const scenario of report.scenarios) {
    console.log(
      `${scenario.name}: workspaceBytes=${scenario.workspaceBytes} peakAccountedBytes=${report.peakAccountedBytes}`,
    );
  }
};

if (import.meta.main) await main(Deno.args);
