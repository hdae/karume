/** 実験 LLM のローカル系列だけを選ぶ。公開配布形はまだ無いので、取得・再変換は行わない。 */
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
  for (const selected of quant === undefined ? LLM_QUANTS : [quant]) {
    const canonical = `${name}-${selected}`;
    const candidates = entries.filter((entry) => entry.isDirectory || entry.isSymlink).map((
      entry,
    ) => entry.name);
    if (candidates.includes(canonical)) return `${root}/${canonical}`;
    const dated = new RegExp(
      `^${name}${selected === "f32" ? "" : `-${selected}`}-\\d{4}-\\d{2}-\\d{2}-probe$`,
    );
    const found = candidates.filter((entry) => dated.test(entry)).sort();
    if (found.length > 1) {
      throw new Error(`${name} の ${selected} が複数ある: ${found.join(", ")}（--source で指定）`);
    }
    if (found.length === 1) return `${root}/${found[0]}`;
  }
  throw new Error(
    `${name} の変換済みモデルが ${root} に無い（--source <系列ディレクトリ> で指定）。公式の未変換重みは読めません。`,
  );
};
