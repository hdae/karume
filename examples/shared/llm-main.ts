/** MiniCPM5 / Qwen3 の単発チャット。--completion はテンプレートを付けない文章継続。 */
import { acquireGpu, prepareModel } from "../../packages/runtime/mod.ts";
import {
  readShard,
  resolveShards,
  streamShards,
} from "../../packages/runtime/tests/helpers/shard-files.ts";
import { createLlmTokenizer, type LlmFamily } from "./llm-tokenizer.ts";
import { LLM_QUANTS, llmProfile, localFileUrl, selectLlmSource } from "./llm-source.ts";
import { checkLlmRequest, inspectLlmGraph, streamLlm } from "./llm-generate.ts";

export const runLlmCli = async (family: LlmFamily, argv = Deno.args): Promise<void> => {
  const profile = llmProfile(family);
  const usage = `deno task demo:${family} [--prompt <文字列>] [--system <文字列>]\n` +
    `  [--source <変換済み系列ディレクトリ> | --quant <${LLM_QUANTS.join("|")}>]\n` +
    `  [--tokenizer <公式 tokenizer.json>] [--max-new-tokens <整数>] [--completion] [--json]\n` +
    "ローカルモデルを優先します。未指定の prompt は標準入力から読みます（Ctrl+D で確定）。\n" +
    "単発生成・greedy・非 thinking。容量は 128 token。--json は EOS を含む ID 列も出力します。";
  if (argv.length === 1 && (argv[0] === "--help" || argv[0] === "-h")) {
    console.log(usage);
    return;
  }
  const args = new Map<string, string>();
  const flags = new Set<string>();
  for (let at = 0; at < argv.length; at++) {
    const key = argv[at];
    if (["--completion", "--json"].includes(key)) {
      if (flags.has(key)) throw new Error(`${key} が重複しています`);
      flags.add(key);
      continue;
    }
    if (
      !["--prompt", "--system", "--source", "--quant", "--tokenizer", "--max-new-tokens"].includes(
        key,
      )
    ) throw new Error(`未知の引数 ${key}\n${usage}`);
    const value = argv[++at];
    if (value === undefined || value.startsWith("--") || args.has(key)) {
      throw new Error(`${key} の値が無い、または重複しています\n${usage}`);
    }
    args.set(key, value);
  }
  if (flags.has("--completion") && args.has("--system")) {
    throw new Error("--completion では --system を使えません");
  }
  const rawMax = args.get("--max-new-tokens") ?? "64";
  if (!/^[1-9]\d*$/.test(rawMax) || !Number.isSafeInteger(Number(rawMax))) {
    throw new Error("--max-new-tokens は 1 以上の安全整数が必要です");
  }
  const maxNewTokens = Number(rawMax);
  const source = await selectLlmSource(family, args.get("--source"), args.get("--quant"));
  const tokenizerPath = args.get("--tokenizer") ?? profile.tokenizer;
  const tokenizer = createLlmTokenizer(
    family,
    JSON.parse(await Deno.readTextFile(tokenizerPath)),
    JSON.parse(await Deno.readTextFile(new URL("./llm-unicode.json", import.meta.url))),
  );
  const encoder = new TextEncoder();
  const note = (text: string): void => {
    Deno.stderr.writeSync(encoder.encode(text));
  };
  let prompt = args.get("--prompt");
  if (prompt === undefined) {
    if (Deno.stdin.isTerminal()) note("入力文を入力し、Ctrl+D で確定してください。\n");
    prompt = await new Response(Deno.stdin.readable).text();
    // シェルの echo / here-document が付ける改行 1 個だけを除く。
    prompt = prompt.replace(/\r?\n$/, "");
  }
  if (prompt.length === 0) throw new Error("入力文が空です（--prompt または標準入力が必要です）");
  const ids = flags.has("--completion")
    ? tokenizer.encode(prompt, true)
    : tokenizer.chat(prompt, args.get("--system"));
  const shards = resolveShards(localFileUrl(`${source.replace(/\/+$/, "")}/model.safetensors`));
  const prepared = prepareModel(await readShard(shards[0]));
  const graph = inspectLlmGraph(family, prepared.graph);
  checkLlmRequest(ids, maxNewTokens, graph);
  note(
    `[${profile.name}] ${source}\n  tokenizer: ${tokenizerPath}\n  prompt ${ids.length} token / max-new-tokens ${maxNewTokens} / greedy\n`,
  );
  const abort = new AbortController();
  const interrupt = (): void =>
    abort.abort(new DOMException("Ctrl+C で中断しました", "AbortError"));
  Deno.addSignalListener("SIGINT", interrupt);
  try {
    const gpu = await acquireGpu();
    using _destroy = { [Symbol.dispose]: () => gpu.destroy() };
    const loaded = performance.now();
    const session = await prepared.createSession(gpu, streamShards(shards.slice(1)));
    await using _release = { [Symbol.asyncDispose]: () => session.dispose() };
    note(`  loaded ${((performance.now() - loaded) / 1000).toFixed(2)}s\n`);
    const started = performance.now();
    const decoder = tokenizer.decoder();
    const tokens: number[] = [];
    let text = "";
    let stop = "length";
    const write = (chunk: string): void => {
      text += chunk;
      if (!flags.has("--json")) Deno.stdout.writeSync(encoder.encode(chunk));
    };
    for await (
      const token of streamLlm(
        session,
        graph,
        ids,
        maxNewTokens,
        tokenizer.stopTokens,
        abort.signal,
      )
    ) {
      tokens.push(token);
      if (tokenizer.stopTokens.includes(token)) {
        stop = "eos";
        continue;
      }
      write(decoder.push(token));
    }
    write(decoder.finish());
    const elapsedMs = performance.now() - started;
    if (flags.has("--json")) {
      console.log(
        JSON.stringify({
          model: profile.name,
          source,
          promptTokens: ids,
          tokens,
          text,
          stop,
          elapsedMs,
        }),
      );
    } else {
      Deno.stdout.writeSync(encoder.encode("\n"));
      note(
        `  ${tokens.length} token（EOS を含む）/ ${
          (elapsedMs / 1000).toFixed(2)
        }s / stop=${stop}\n`,
      );
    }
  } finally {
    Deno.removeSignalListener("SIGINT", interrupt);
  }
};
