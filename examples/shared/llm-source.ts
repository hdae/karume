/**
 * 実験 LLM のローカル系列だけを選ぶ。公開配布形はまだ無いので、取得・再変換は行わない。
 *
 * 選ぶのは**容器（krm）に変換済みの系列だけ**。`<name>-<quant>-<日付>-probe` の綴りで残って
 * いる系列は旧 shard 形の研究記録で、容器としては読めない（変換しない — 実測は
 * `docs/research/`）。それを既定で拾っていた頃は、デモが `model.krm` の素の `NotFound` で
 * 落ちて、利用者には「資産が壊れている」としか見えなかった。今は旧形を名指して落とす。
 */
import type { LlmFamily } from "./llm-tokenizer.ts";

export const llmProfile = (family: LlmFamily): {
  name: string;
  tokenizer: string;
  vocabSize: number;
  layers: number;
  kvHeads: number;
} =>
  family === "qwen3"
    ? {
      name: "qwen3-06b",
      tokenizer: "inputs/qwen3/Qwen3-0.6B/tokenizer.json",
      vocabSize: 151936,
      layers: 28,
      kvHeads: 8,
    }
    : {
      name: "minicpm5-2b",
      tokenizer: "inputs/minicpm5/MiniCPM5-2B/tokenizer.json",
      vocabSize: 130560,
      layers: 42,
      kvHeads: 2,
    };

export const LLM_QUANTS = ["gptq-i4", "i8", "i4", "f16", "f32"] as const;

/** ファイル名を URL の query / fragment として解釈させない。 */
export const localFileUrl = (path: string): URL => {
  const url = new URL("file:///");
  const absolute = path.startsWith("/") ? path : `${Deno.cwd()}/${path}`;
  url.pathname = absolute.split("/").map(encodeURIComponent).join("/");
  return url;
};

export const selectLlmSource = async (
  family: LlmFamily,
  source?: string,
  quant?: string,
  root = "outputs/series",
): Promise<string> => {
  if (quant !== undefined && !LLM_QUANTS.some((known) => known === quant)) {
    throw new Error(`未対応の --quant ${quant}`);
  }
  if (source !== undefined) {
    if (quant !== undefined) {
      throw new Error("--source と --quant は排他（系列ディレクトリで量子化を選ぶ）");
    }
    return source;
  }
  let entries: Deno.DirEntry[];
  try {
    entries = [];
    for await (const entry of Deno.readDir(root)) entries.push(entry);
  } catch (error) {
    if (!(error instanceof Deno.errors.NotFound)) throw error;
    entries = [];
  }
  const name = llmProfile(family).name;
  const candidates = entries.filter((entry) => entry.isDirectory || entry.isSymlink).map((
    entry,
  ) => entry.name);
  for (const selected of quant === undefined ? LLM_QUANTS : [quant]) {
    const canonical = `${name}-${selected}`;
    if (candidates.includes(canonical)) return `${root}/${canonical}`;
  }
  // 旧 shard 形のまま残る研究記録（`<name>[-<quant>]-YYYY-MM-DD-probe`）。容器ではないので
  // 選ばないが、**在ることは診断に出す** — 「資産があるのに読めない」を利用者が自力で
  // 切り分けられないまま NotFound だけを見る形にしない。
  const legacy = candidates
    .filter((entry) => new RegExp(`^${name}(-[a-z0-9-]+)?-\\d{4}-\\d{2}-\\d{2}-probe$`).test(entry))
    .sort();
  const wanted = quant === undefined ? LLM_QUANTS.join(" / ") : quant;
  throw new Error(
    `${name} の容器（krm）に変換済みの系列が ${root} に無い（探した量子化: ${wanted}）。` +
      (legacy.length === 0
        ? ""
        : `旧 shard 形の系列は在りますが容器ではないので読めません: ${legacy.join(", ")}` +
          "（研究記録 — docs/research/ 参照。変換しません）。") +
      "--source <変換済み系列ディレクトリ> で指定してください。公式の未変換重みは読めません。",
  );
};
