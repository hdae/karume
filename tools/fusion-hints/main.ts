/**
 * 融合候補ヒントの列挙（Fusion 半自動発見 1 段目・GPU 不要）。
 *
 *     # 配布形（karume.json のある根 — 既定モデル・家族の既定シナリオ）
 *     deno run -A tools/fusion-hints/main.ts enumerate --source models/karume-anima \
 *         --out outputs/bench/karume-anima/2026-09-03_fusion-hints
 *     # 系列出力（先頭 shard を木から探す）・シナリオは明示
 *     deno run -A tools/fusion-hints/main.ts enumerate --source outputs/series/gemma4-e2b-decode \
 *         --scenario decode=M:1,C:640 --max-window 9
 *
 * 出力は `<out>/candidates.jsonl`（1 行 1 レコード）と `<out>/candidates.md`（上位表）。
 * `--out` を省くと Markdown を標準出力へ書く。
 *
 * 資産の解決とシナリオ語彙は tools/opbench（静的 census）と同じ 1 本を使う — 記号束縛が
 * 2 つの道具で別値だと、形状に依存する量（census 加重と候補本数）を跨いで読めない。
 */

import { parseInductorArgs, runInductor } from "./inductor.ts";
import { directoryUrl, readIrGraph, resolveAsset } from "../_shared/assets.ts";
import {
  assertBindingKeys,
  assertPlainBindingKeys,
  defaultScenarios,
  parseScenarios,
  type Scenario,
} from "../_shared/scenario.ts";
import {
  type GraphInput,
  reportScenario,
  type ScenarioReport,
  type SourceReport,
  toJsonl,
  toMarkdown,
} from "./enumerate.ts";

const USAGE = `使い方: deno run -A tools/fusion-hints/main.ts enumerate --source <dir> [オプション]
      （Inductor との突合は \`inductor --out <dir> --candidates <candidates.jsonl>…\` — 詳細は README）
  --source <dir>       配布形（karume.json あり）か outputs/series の系列ディレクトリ
  --out <dir>          candidates.jsonl / candidates.md の書き出し先（省略時は Markdown を標準出力へ）
  --max-window <n>     連続窓の最大長（既定 9）
  --top <n>            Markdown の 1 グラフあたりの行数（既定 10・jsonl は常に全行）
  --model <name>       配布形の model（既定 = manifest の defaultModel）
  --quant <name>       配布形の quant（既定 = defaultQuant）/ 系列は格納 dtype グループ名
  --family <name>      家族名の明示（既定 = manifest の pipeline / ディレクトリ名から推定）
  --scenario <名前>=<記号>:<値>[,…]   記号次元の束縛（繰り返し可・既定は家族ごとの表）
  --no-fusion          融合を切った計画に掛ける（既知のヒット数との答え合わせ用）`;

type Options = {
  readonly source: string;
  readonly out?: string;
  readonly model?: string;
  readonly quant?: string;
  readonly family?: string;
  readonly maxWindow: number;
  readonly top: number;
  readonly fused: boolean;
  /** `--scenario` で与えたシナリオ（空なら家族の既定シナリオ）。 */
  readonly scenarios: readonly Scenario[];
};

/**
 * 数値引数は正の安全整数だけ受ける（打ち間違いを fail loudly）。
 *
 * 素の `Number` で受けると壊れ方が黙る: `--top abc` は NaN → `slice(0, NaN)` で候補表が 0 行の
 * まま「候補なし」とも書かれない Markdown が出る。
 */
const positiveInteger = (raw: string, where: string): number => {
  const value = Number(raw);
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`${where} の値 '${raw}' が正の整数でない。\n${USAGE}`);
  }
  return value;
};

/**
 * 廃止したフラグ。**互換シムは置かない**（未公開ツール — 旧綴りを黙って読み替えると、
 * 2 つの道具で束縛が別値のままの表がまた出る）。理由を名指して落とす。
 */
const REMOVED_FLAGS: Readonly<Record<string, string>> = {
  "--bind": "全 component 共通の 1 記号束縛",
  "--default-symbol": "残りの記号を 1 値で埋める既定",
};

/**
 * 値を取るオプションの全て。
 *
 * MUST: 未知の `--` は捨てずに落とす。`--scenarios`（複数形）のような 1 文字違いを黙って
 * 捨てると、明示したはずの束縛が効かないまま**家族の既定シナリオ**で走った表が出る
 * （出力にはその既定が `binding_source: default` として載るので、読み手が気付ける保証も無い）。
 */
const VALUE_FLAGS: readonly string[] = [
  "source",
  "out",
  "max-window",
  "top",
  "model",
  "quant",
  "family",
  "scenario",
];

/** 引数解析（enumerate_test.ts が廃止フラグと未知オプションの拒否を見るので export する）。 */
export const parseArgs = (argv: readonly string[]): Options => {
  if (argv[0] !== "enumerate") throw new Error(`未知のサブコマンド '${argv[0] ?? ""}'。\n${USAGE}`);
  const values = new Map<string, string[]>();
  let fused = true;
  for (let at = 1; at < argv.length;) {
    const key = argv[at];
    if (!key.startsWith("--")) throw new Error(`引数 '${key}' がオプションでない。\n${USAGE}`);
    if (key === "--no-fusion") {
      fused = false;
      at += 1;
      continue;
    }
    if (Object.hasOwn(REMOVED_FLAGS, key)) {
      throw new Error(
        `${key}（${REMOVED_FLAGS[key]}）は廃止した — opbench と同じシナリオ語彙へ統一した。` +
          " --scenario <名前>=<記号>:<値>[,…] で与える" +
          `（component ごとに別の値を入れるときは <component>.<記号>:<値>）。\n${USAGE}`,
      );
    }
    if (!VALUE_FLAGS.includes(key.slice(2))) {
      throw new Error(`未知のオプション '${key}'。\n${USAGE}`);
    }
    const value = argv[at + 1];
    if (value === undefined) throw new Error(`引数 ${key} が対でない。\n${USAGE}`);
    // 値の書き忘れ（`--source --out dir`）を「次のオプション名を値として受ける」形で飲まない。
    // 値に `--` 始まりを取るオプションは 1 つも無い。
    if (value.startsWith("--")) {
      throw new Error(`引数 ${key} の値が無い（'${value}' はオプション）。\n${USAGE}`);
    }
    const name = key.slice(2);
    values.set(name, [...(values.get(name) ?? []), value]);
    at += 2;
  }
  const single = (name: string): string | undefined => {
    const found = values.get(name);
    if (found === undefined) return undefined;
    if (found.length > 1) throw new Error(`--${name} は 1 度しか指定できない。\n${USAGE}`);
    return found[0];
  };
  const source = single("source");
  if (source === undefined) throw new Error(`--source は必須。\n${USAGE}`);
  const maxWindow = single("max-window");
  const top = single("top");
  const window = maxWindow === undefined ? 9 : positiveInteger(maxWindow, "--max-window");
  // 融合は 2 ノード以上の窓でしか成立しない — 1 は候補ゼロの表を黙って出すので落とす。
  if (window < 2) throw new Error(`--max-window は 2 以上（'${maxWindow}'）。\n${USAGE}`);
  return {
    source,
    out: single("out"),
    model: single("model"),
    quant: single("quant"),
    family: single("family"),
    maxWindow: window,
    top: top === undefined ? 10 : positiveInteger(top, "--top"),
    fused,
    scenarios: parseScenarios(values.get("scenario") ?? []),
  };
};

/** 1 資産ぶんの候補表を組む（読めなかったコンポーネントは理由つきで残す）。 */
const buildReport = async (options: Options): Promise<SourceReport> => {
  const root = directoryUrl(options.source);
  const asset = await resolveAsset(root, options.model, options.quant, options.family);
  const scenarios: readonly Scenario[] = options.scenarios.length === 0
    ? defaultScenarios(asset.family)
    : options.scenarios;
  const components = asset.components.map((target) => target.component);
  for (const scenario of scenarios) assertBindingKeys(scenario, components);

  const graphs: GraphInput[] = [];
  const skipped: { name: string; reason: string }[] = [];
  for (const target of asset.components) {
    const name = `${asset.model}/${target.component}`;
    try {
      graphs.push({
        component: target.component,
        graph: name,
        // percent encode を解いてから資産根ぶんを落とす（報告の `path` 欄は人が読む path で、
        // `%20` の混じった綴りは手で開けない）。根は末尾 `/` なので境界に escape は跨がらない。
        path: decodeURIComponent(target.graphShard.pathname)
          .slice(decodeURIComponent(root.pathname).length),
        ir: await readIrGraph(target.graphShard),
      });
    } catch (cause) {
      // 読めない 1 本で全体を止めない。ただし黙って落とさず、理由を報告に残す
      // （IR を載せない shard を先頭と取り違えた形はここに出る）。記号の未束縛はここではなく
      // reportScenario が打ち切る — 既定値で計画を進めないため。
      const reason = cause instanceof Error ? cause.message : String(cause);
      skipped.push({ name, reason });
      console.error(`[fusion-hints] ${name}: ${reason}`);
    }
  }

  // 修飾なしキーの誤綴りはグラフを読むまで判らない。読めなかったコンポーネントがあると
  // 和集合が欠けるので、その回は検査を見送る（欠けた 1 本だけが宣言していた記号を誤って
  // 落とさないため — 読めなかったこと自体は上の stderr と `skipped` に残っている）。
  if (skipped.length === 0) {
    const irs = graphs.map(({ ir }) => ir);
    for (const scenario of scenarios) assertPlainBindingKeys(scenario, irs);
  }

  const reports: ScenarioReport[] = scenarios.map((scenario) =>
    reportScenario(scenario, graphs, { maxWindow: options.maxWindow, fused: options.fused })
  );
  return {
    source: options.source,
    family: asset.family,
    model: asset.model,
    quant: asset.quant,
    max_window: options.maxWindow,
    fused: options.fused,
    scenarios: reports,
    skipped,
  };
};

// MUST: CLI の本体は `import.meta.main` の内側だけで走らせる（横断の不変条件「全モジュール
// 副作用ゼロ = import 時実行の禁止」— 型や関数をここから import した瞬間に引数解析と
// ディレクトリ書き出しが走るのを防ぐ）。
if (import.meta.main && Deno.args[0] === "inductor") {
  // Inductor の融合決定と候補表の突合（CUDA venv の python に委ねる — inductor.ts）。
  await runInductor(parseInductorArgs(Deno.args.slice(1)));
} else if (import.meta.main) {
  const options = parseArgs(Deno.args);
  const report = await buildReport(options);
  const markdown = toMarkdown(report, options.top);
  if (options.out === undefined) {
    console.log(markdown);
  } else {
    const out = directoryUrl(options.out);
    await Deno.mkdir(out, { recursive: true });
    await Deno.writeTextFile(new URL("candidates.jsonl", out), toJsonl(report));
    await Deno.writeTextFile(new URL("candidates.md", out), markdown);
    console.log(JSON.stringify({
      source: report.source,
      family: report.family,
      model: report.model,
      quant: report.quant,
      scenarios: report.scenarios.map((scenario) => ({
        scenario: scenario.scenario,
        graphs: scenario.graphs.length,
        windows: scenario.graphs.reduce((sum, graph) => sum + graph.window_count, 0),
      })),
      out: options.out,
    }));
  }
}
