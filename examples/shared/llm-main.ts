/** MiniCPM5 / Qwen3 の対話 CLI。--prompt は単発、--completion は文章継続。 */
import { acquireGpu, prepareModel } from "../../packages/runtime/mod.ts";
import {
  readShard,
  resolveShards,
  streamShards,
} from "../../packages/runtime/tests/helpers/shard-files.ts";
import { createLlmTokenizer, type LlmFamily } from "./llm-tokenizer.ts";
import { LLM_QUANTS, llmProfile, localFileUrl, selectLlmSource } from "./llm-source.ts";
import {
  checkLlmRequest,
  inspectLlmGraph,
  LlmCapacityError,
  LlmSequence,
  streamLlm,
} from "./llm-generate.ts";
import { LlmChat, readLlmLines } from "./llm-chat.ts";
import { formatGenerationTiming, generationTimer } from "./generation-timing.ts";

export const runLlmCli = async (family: LlmFamily, argv = Deno.args): Promise<void> => {
  const profile = llmProfile(family);
  const usage = `deno task demo:${family} [--prompt <文字列>] [--system <文字列>]\n` +
    `  [--source <変換済み系列ディレクトリ> | --quant <${LLM_QUANTS.join("|")}>]\n` +
    `  [--tokenizer <公式 tokenizer.json>] [--max-new-tokens <整数>] [--completion] [--json] [--no-warmup]\n` +
    "ローカルモデルを優先します。--prompt を省略すると行ごとの対話になります。\n" +
    "/reset で履歴消去、/exit・Ctrl+D で終了、生成中の Ctrl+C でターン中断。\n" +
    "greedy・非 thinking・容量 128 token。古い発話を対で削除し、system は保持します。\n" +
    "--completion は stdin を EOF まで読む単発継続。--json は各回答の ID 列も出力します。";
  if (argv.length === 1 && (argv[0] === "--help" || argv[0] === "-h")) {
    console.log(usage);
    return;
  }
  const args = new Map<string, string>();
  const flags = new Set<string>();
  for (let at = 0; at < argv.length; at++) {
    const key = argv[at];
    if (["--completion", "--json", "--no-warmup"].includes(key)) {
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
  const interactive = prompt === undefined && !flags.has("--completion");
  if (prompt === undefined && !interactive) {
    if (Deno.stdin.isTerminal()) note("入力文を入力し、Ctrl+D で確定してください。\n");
    prompt = await new Response(Deno.stdin.readable).text();
    // シェルの echo / here-document が付ける改行 1 個だけを除く。
    prompt = prompt.replace(/\r?\n$/, "");
  }
  if (prompt === "") throw new Error("入力文が空です（--prompt または標準入力が必要です）");
  const ids = prompt === undefined
    ? undefined
    : flags.has("--completion")
    ? tokenizer.encode(prompt, true)
    : tokenizer.chat(prompt, args.get("--system"));
  const shards = resolveShards(localFileUrl(`${source.replace(/\/+$/, "")}/model.safetensors`));
  const prepared = prepareModel(await readShard(shards[0]));
  const graph = inspectLlmGraph(family, prepared.graph);
  if (ids !== undefined) checkLlmRequest(ids, maxNewTokens, graph);
  note(
    `[${profile.name}] ${source}\n  tokenizer: ${tokenizerPath}\n  ${
      ids === undefined ? "対話" : `prompt ${ids.length} token`
    } / max-new-tokens ${maxNewTokens} / greedy\n`,
  );
  let turn: AbortController | undefined;
  const interrupt = (): void => {
    if (turn === undefined) Deno.exit(130);
    turn.abort(new DOMException("Ctrl+C で中断しました", "AbortError"));
  };
  Deno.addSignalListener("SIGINT", interrupt);
  try {
    const gpu = await acquireGpu();
    using _destroy = { [Symbol.dispose]: () => gpu.destroy() };
    const loaded = performance.now();
    const session = await prepared.createSession(gpu, streamShards(shards.slice(1)));
    await using _release = { [Symbol.asyncDispose]: () => session.dispose() };
    note(`  loaded ${((performance.now() - loaded) / 1000).toFixed(2)}s\n`);
    if (!flags.has("--no-warmup")) {
      note("  ウォームアップ中（別の会話・最大4 token）\n");
      const started = performance.now();
      turn = new AbortController();
      let tokens = 0;
      try {
        for await (
          const token of streamLlm(
            session,
            graph,
            tokenizer.encode("Hello", true),
            4,
            tokenizer.stopTokens,
            turn.signal,
          )
        ) if (!tokenizer.stopTokens.includes(token)) tokens += 1;
      } finally {
        turn = undefined;
      }
      note(
        `  warmup ${
          ((performance.now() - started) / 1000).toFixed(2)
        }s / ${tokens} token（会話履歴に含めません）\n`,
      );
    } else {
      note("  ウォームアップなし\n");
    }
    if (interactive) {
      await using sequence = new LlmSequence(session, graph);
      const chat = new LlmChat(sequence, tokenizer, graph, maxNewTokens, args.get("--system"));
      const json = flags.has("--json");
      const write = (text: string): void => {
        Deno.stdout.writeSync(encoder.encode(text));
      };
      const status = json ? note : write;
      const ready = (): void => {
        if (!json) write("> ");
      };
      status(
        `[${family}] ready / capacity 128 / greedy / max-new-tokens ${maxNewTokens}\n` +
          "  /reset で会話を捨てる・/exit か Ctrl+D で終わる・生成中の Ctrl+C はそのターンを中断\n\n",
      );
      ready();
      for await (const raw of readLlmLines(Deno.stdin.readable)) {
        const line = raw.trim();
        if (line === "") {
          ready();
          continue;
        }
        if (line === "/exit" || line === "/quit") break;
        if (line === "/reset") {
          await chat.reset();
          status("(reset)\n");
          ready();
          continue;
        }
        turn = new AbortController();
        const timer = generationTimer();
        try {
          const result = await chat.send(line, (chunk) => {
            if (!json) write(chunk);
          }, {
            signal: turn.signal,
            onToken: timer.onToken,
            onOverflow: (count) => status(`\n  [容量超過: 古い ${count} 組の発話を外して再構成]\n`),
            onPrefill: ({ chunk, chunks }) => {
              if (chunks > 1) note(`  prefill ${chunk}/${chunks}\n`);
            },
          });
          const timing = timer.finish();
          if (json) {
            write(JSON.stringify({ model: profile.name, source, ...result, ...timing }) + "\n");
          } else {
            status(
              `\n  [${result.stop} · ${result.tokens.length} tok · ${
                formatGenerationTiming(timing)
              }` +
                ` · 会話 ${chat.turns.length * 2} 発話 · KV 再利用 ${result.reusedTokens} token]\n`,
            );
          }
        } catch (error) {
          if (!(error instanceof LlmCapacityError)) throw error;
          note(`\n  [入り切らない: ${error.message}]\n`);
        } finally {
          turn = undefined;
        }
        ready();
      }
      status("\nbye\n");
      return;
    }
    if (ids === undefined) throw new Error("単発生成の入力がありません");
    turn = new AbortController();
    const timer = generationTimer();
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
        turn.signal,
      )
    ) {
      tokens.push(token);
      if (tokenizer.stopTokens.includes(token)) {
        stop = "eos";
        continue;
      }
      timer.onToken();
      write(decoder.push(token));
    }
    write(decoder.finish());
    const timing = timer.finish();
    if (flags.has("--json")) {
      console.log(
        JSON.stringify({
          model: profile.name,
          source,
          promptTokens: ids,
          tokens,
          text,
          stop,
          ...timing,
        }),
      );
    } else {
      Deno.stdout.writeSync(encoder.encode("\n"));
      note(
        `  ${tokens.length} token（EOS を含む）/ ${
          formatGenerationTiming(timing)
        } / stop=${stop}\n`,
      );
    }
  } finally {
    Deno.removeSignalListener("SIGINT", interrupt);
  }
};
