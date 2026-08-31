/**
 * Gemma 4 E2B（文字列 → 文字列）の対話デモ — **低レベル面（`sequence`）の写経見本**。
 *
 *     deno task demo:gemma4
 *     deno task demo:gemma4 --system "Answer in one short sentence." --max-new-tokens 128
 *     deno task demo:gemma4 --source models/karume-gemma4-e2b --temperature 0
 *     deno task demo:gemma4 --repo someone/karume-gemma4-e2b@1a2b3c4 --seed 7
 *
 * 1 行 = 1 発話。`/reset` で会話を捨て、`/exit`（または Ctrl+D）で終わる。生成中の Ctrl+C は
 * **そのターンだけ**を中断する（生成していないときに押すとプロセスごと終わる）。
 *
 * 取得元は 2 つに割れている（`fromPretrained` 自体に既定は無いので、綴るのは常に呼び出し側）:
 * `--source` は**手元の配布形ディレクトリ**（`karume.json` を持つ）を `denoDirectory` で直に読み、
 * `--repo owner/name[@revision]` は HF から取る。gemma4 は公開配布リポをまだ持たないので pin
 * 定数（`*_CURRENT` — ADR 0073）も無く、既定はローカルミラーの綴り {@link DEFAULT_SOURCE}。
 *
 * ## なぜ `chat` ではなく `sequence` なのか
 *
 * `Gemma4Pipeline.chat` は **1 ターン = 1 sequence** で、過去 turn は毎回 prompt へ描き直される
 * （会話が伸びるほど prefill が O(n²)）。多ターンの対話は `sequence()` を 1 本持ち回り、新しい
 * turn の**差分だけ**を流すのが正で、この台本はその写経見本である:
 *
 * - 初回 = `gemma4ChatPrompt`（`<bos>` 込みの全体・末尾は生成プロンプト）
 * - 2 ターン目以降 = `gemma4ChatTurn`（その turn の差分だけ — 過去は context の KV にある）
 * - 前 turn を閉じる `<turn|>` は**自分で足さない**。生成が出したその 1 token は sequence の
 *   未 commit frontier（`pendingToken`）として残っていて、次の `generate` が prompt の先頭へ
 *   自動で連結する（ADR 0083 決定 4）。二重に描いても例外は 1 つも出ず、turn の区切りだけが
 *   静かに 2 つになる。
 * - 差分を継げるのは**生成が停止 token で閉じた直後だけ**（`gemma4ChatTurn` の MUST）。
 *   max-tokens や中断で打ち切ったターンの後ろは model turn が閉じていないので、この台本は
 *   sequence を捨てて履歴から組み直す（{@link releaseSequence}）。
 *
 * ## 会話の切り詰めはホストの責務
 *
 * 容量（`program.capacity`）と位置表（`program.maxPosition`）を超えた要求は
 * `GenerationCapacityError` で落ちる。何を捨てるかはアプリの意味論なので、ここが「古い turn から
 * 落として新しい sequence で組み直す」を実装する（ADR 0083 決定 10）。判断に要る実値は例外の
 * 欄が運ぶので、文言を読み解く必要は無い。
 */

import {
  gemma4ChatPrompt,
  gemma4ChatTurn,
  Gemma4Pipeline,
  GenerationCapacityError,
} from "../../packages/models/gemma.ts";
import type {
  Gemma4ChatMessage,
  GenerationSequence,
  GenerationStop,
  GenerationStream,
  SamplerSpec,
} from "../../packages/models/gemma.ts";
import { denoDirectory } from "../../packages/hub/deno.ts";

const USAGE = "--source <配布形のパス> | --repo <owner/name[@revision]>" +
  " --system <文字列> --max-new-tokens <整数> --temperature <数> --top-k <整数>" +
  " --top-p <数> --seed <整数>";
const KNOWN = new Set([
  "source",
  "repo",
  "system",
  "max-new-tokens",
  "temperature",
  "top-k",
  "top-p",
  "seed",
]);

/** 取得元の既定（`dist.py --pipeline gemma4` が組むローカルミラー — `docs/assets-layout.md`）。 */
const DEFAULT_SOURCE = "models/karume-gemma4-e2b";

/** 1 ターンで生成する token 数の上限（停止 token は含まれない）。 */
const DEFAULT_MAX_NEW_TOKENS = 256;

/**
 * PLE sidecar の常駐本数（ADR 0085 決定 3 の LRU）。
 *
 * 既定は 2 本だが、この台本は**対話の応答時間**を優先して 3 本（= E2B の全 shard）を載せる。
 * shard は token 範囲で割ってあり、1 文の中でも id は語彙全体へ散るので、常駐が shard 本数を
 * 下回ると 1 ターンの間に 795MB の読み直しが何度も走る。代償はホスト RAM 約 2.3GiB で、
 * 絞りたければこの定数を下げる（読み直しが増えるだけで、値も token 列も変わらない）。
 */
const RESIDENT_PLE_SHARDS = 3;

/** `--key value` の対だけを受ける。MUST: 次のフラグを値として食わない（黙って既定へ落ちる）。 */
const args = new Map<string, string>();
for (let at = 0; at < Deno.args.length; at += 2) {
  const [key, value] = [Deno.args[at], Deno.args[at + 1]];
  if (!key.startsWith("--") || value === undefined || value.startsWith("--")) {
    throw new Error(`引数 ${key} が --key value の対になっていない（使い方: ${USAGE}）`);
  }
  // MUST: 未知のキーは落とす。打ち間違えたノブが黙って既定値で走ると、出力の違いが
  // 「モデルの揺れ」に見える。
  if (!KNOWN.has(key.slice(2))) throw new Error(`未知のオプション ${key}（使い方: ${USAGE}）`);
  args.set(key.slice(2), value);
}

const integer = (key: string): number | undefined => {
  const raw = args.get(key);
  if (raw !== undefined && !/^\d+$/.test(raw)) throw new Error(`--${key} ${raw} が非負整数でない`);
  return raw === undefined ? undefined : Number(raw);
};
const number = (key: string): number | undefined => {
  const raw = args.get(key);
  if (raw === undefined) return undefined;
  const value = Number(raw);
  if (!Number.isFinite(value)) throw new Error(`--${key} ${raw} が数値でない`);
  return value;
};

const temperature = number("temperature");
const topK = integer("top-k");
const topP = number("top-p");
const seed = integer("seed");

const sourceDir = args.get("source");
const repoRef = args.get("repo");
// MUST: 両方は受けない（どちらを読んだか台本の外から見えない取得は fail loudly）。
if (sourceDir !== undefined && repoRef !== undefined) {
  throw new Error(`--source と --repo は排他（使い方: ${USAGE}）`);
}
const system = args.get("system");
const maxNewTokens = integer("max-new-tokens") ?? DEFAULT_MAX_NEW_TOKENS;

/**
 * `owner/name` か `owner/name@revision`。`@` が無ければ revision は付けない — hub が「main は
 * 付け替えられる」警告を出す側で、ここで `"main"` を捏造すると警告が消える。
 */
const parseRepo = (spelled: string): { repo: string; revision?: string } => {
  const cut = spelled.lastIndexOf("@");
  if (cut < 0) return { repo: spelled };
  return { repo: spelled.slice(0, cut), revision: spelled.slice(cut + 1) };
};

const encoder = new TextEncoder();
const write = (text: string): void => {
  Deno.stdout.writeSync(encoder.encode(text));
};
const note = (text: string): void => {
  Deno.stderr.writeSync(encoder.encode(text));
};

/** stdin を行で汲む（tty でもパイプでも同じ — EOF = Ctrl+D で終わる）。 */
async function* readLines(): AsyncGenerator<string> {
  const decoder = new TextDecoder();
  let buffer = "";
  for await (const chunk of Deno.stdin.readable) {
    buffer += decoder.decode(chunk, { stream: true });
    for (;;) {
      const cut = buffer.indexOf("\n");
      if (cut < 0) break;
      yield buffer.slice(0, cut);
      buffer = buffer.slice(cut + 1);
    }
  }
  if (buffer.length > 0) yield buffer;
}

const describeStop = (stop: GenerationStop): string =>
  stop.reason === "eos" ? `eos(${stop.token})` : stop.reason;

/** 1 行ぶんの進捗表示を消す（次の出力が食い込まないように空白で塗ってから戻す）。 */
const clearLine = (): void => note(`\r${" ".repeat(40)}\r`);

const started = performance.now();
note(`[gemma4] ${sourceDir ?? repoRef ?? DEFAULT_SOURCE} を読み込む\n`);
await using pipeline = await Gemma4Pipeline.fromPretrained(
  repoRef === undefined ? denoDirectory(sourceDir ?? DEFAULT_SOURCE) : parseRepo(repoRef),
  {
    residentPleShards: RESIDENT_PLE_SHARDS,
    onProgress: ({ phase, loaded, total }) =>
      note(`\r  ${phase} ${(loaded / total * 100).toFixed(1)}%  `),
  },
);
clearLine();

/**
 * 抽選の指定。**低レベル面は配布形を知らない**ので、`generate` へ渡さなければ低層の既定
 * （温度 0 = greedy）で走る（`Gemma4Pipeline.defaultSampler` の MUST）。既定は配布形が宣言した
 * 推奨値で、CLI のフラグはその上に重ねる（`--temperature 0` だけを渡せば top-k / top-p は
 * 推奨値のまま残るが、温度 0 では候補を削っても最大値が残るので結果は greedy に畳まれる）。
 */
const overrides = {
  ...(temperature === undefined ? {} : { temperature }),
  ...(topK === undefined ? {} : { topK }),
  ...(topP === undefined ? {} : { topP }),
  ...(seed === undefined ? {} : { seed }),
};
const sampler: SamplerSpec | undefined = Object.keys(overrides).length === 0
  ? pipeline.defaultSampler
  : { ...pipeline.defaultSampler, ...overrides };

const { capacity, maxPosition, chunkLength } = pipeline.program;
write(
  `[gemma4] ready（${((performance.now() - started) / 1000).toFixed(1)}s）` +
    ` / capacity ${capacity} / maxPosition ${maxPosition} / chunk ${chunkLength}\n` +
    `         sampler ${
      sampler === undefined ? "greedy（配布形の宣言なし）" : JSON.stringify(sampler)
    }` +
    ` / max-new-tokens ${maxNewTokens}\n` +
    `         /reset で会話を捨てる・/exit か Ctrl+D で終わる・生成中の Ctrl+C はそのターンを中断\n\n`,
);

/**
 * 会話の履歴は**この台本が持つ**（sequence の可変状態は context と `pendingToken` の 2 つだけで、
 * transcript は持たない）。切り詰めも組み直しもここでしかできない。
 */
const history: Gemma4ChatMessage[] = [];
if (system !== undefined) history.push({ role: "system", content: system });
/** 落とせない先頭（system 発話）の件数。 */
const floor = history.length;

/** 現在の sequence（`undefined` = 次のターンで作り直す）。 */
let sequence: GenerationSequence | undefined;
/** その sequence の KV が持っている履歴の件数（0 = 空 = 全体を描き直す）。 */
let committed = 0;

/**
 * 1 ターンの締め。tok/s は `stop.tokens`（停止 token も 1 個 = 抽選 1 回 = run 1 回）から書く —
 * 出力文字列を符号化し直すと、byte_fallback や停止 token のぶんだけ数がずれる。
 */
const report = (stop: GenerationStop, at: number): void => {
  const elapsed = (performance.now() - at) / 1000;
  write(
    `\n  [${describeStop(stop)} · ${stop.tokens} tok · ${elapsed.toFixed(1)}s · ` +
      `${(stop.tokens / elapsed).toFixed(1)} tok/s · 会話 ${sequence?.used ?? 0}/${capacity}]\n`,
  );
};

/** context を返して組み直しに備える（KV と履歴の対応が切れたときは必ずここを通る）。 */
const releaseSequence = async (): Promise<void> => {
  const held = sequence;
  sequence = undefined;
  committed = 0;
  await held?.dispose();
};

/**
 * 古い turn を先頭から落とす（system は残す）。落とせるものが無ければ `false`。
 *
 * 2 件ずつ落とすのは user / assistant の対を単位にするため。奇数個しか残っていなければ
 * 残り 1 件（= 今のターンの user 発話）は必ず残す。
 */
const dropOldestTurn = (): boolean => {
  const droppable = history.length - floor - 1;
  if (droppable <= 0) return false;
  history.splice(floor, Math.min(2, droppable));
  return true;
};

/**
 * Ctrl+C は「今のターンを止める」。生成していないときはプロセスごと終わる。
 *
 * 中断は `AbortSignal` が正で、`signal.reason` は包まれずそのまま throw されるので、下の
 * `error === controller.signal.reason` が「自分が止めた」の判定になる（ADR 0083 決定 5）。
 */
let turn: AbortController | undefined;
const onInterrupt = (): void => {
  if (turn === undefined) {
    write("\n");
    Deno.exit(130);
  }
  turn.abort(new Error("interrupted"));
};
Deno.addSignalListener("SIGINT", onInterrupt);

write("> ");
for await (const raw of readLines()) {
  const line = raw.trim();
  if (line === "") {
    write("> ");
    continue;
  }
  if (line === "/exit" || line === "/quit") break;
  if (line === "/reset") {
    await releaseSequence();
    history.length = floor;
    write("(reset)\n> ");
    continue;
  }

  history.push({ role: "user", content: line });

  // 容量で落ちたら履歴を切り詰めて同じターンを撃ち直す（成功か fail loudly で必ず抜ける）。
  for (;;) {
    if (sequence === undefined) {
      sequence = await pipeline.sequence();
      committed = 0;
    }
    // 差分を継ぐのは「直前の発話 1 件だけが未 commit」のときに限る。ずれたまま差分を流すと
    // 会話が静かに欠けるので、写経の前提そのものを門にしておく。
    if (committed !== 0 && committed !== history.length - 1) {
      throw new Error(`内部不整合: KV は ${committed} 件だが履歴は ${history.length} 件`);
    }
    const prompt = committed === 0
      ? gemma4ChatPrompt(pipeline.tokenizer, history)
      : gemma4ChatTurn(pipeline.tokenizer, history[history.length - 1]);

    const controller = new AbortController();
    turn = controller;
    const turnStarted = performance.now();
    const detokenizer = pipeline.tokenizer.createDetokenizer();
    let stream: GenerationStream | undefined;
    let reply = "";
    let prefilled = false;
    try {
      stream = sequence.generate({
        prompt,
        maxNewTokens,
        ...(sampler === undefined ? {} : { sampler }),
        signal: controller.signal,
      });
      for await (const event of stream) {
        if (event.kind === "prefill") {
          // chunk が 1 本で終わる prompt（= chunk 長以下）では出さない — 進捗にならない。
          if (event.chunks > 1) {
            note(`\r  prefill ${event.chunk}/${event.chunks}  `);
            prefilled = true;
          }
          continue;
        }
        if (prefilled) {
          clearLine();
          prefilled = false;
        }
        // MUST: 逐次復号器が**確定させたぶん**だけを書く（byte_fallback の途中は次の token まで
        // 持ち越される — 自分で `decode` を呼び直すと途中のバイト列が U+FFFD になる）。
        const text = detokenizer.push(event.id);
        if (text !== "") {
          reply += text;
          write(text);
        }
      }
      const tail = detokenizer.finish();
      if (tail !== "") {
        reply += tail;
        write(tail);
      }
      const stop = await stream.done;
      report(stop, turnStarted);
      // 生成が停止 token で閉じたターンだけ、次のターンを差分で継げる。
      if (stop.reason === "eos") {
        history.push({ role: "assistant", content: reply });
        committed = history.length;
      } else {
        // max-tokens / break は model turn を閉じていない = 差分の前提が無い。答えは履歴へ
        // 残し、次のターンは新しい sequence へ全体を描き直す。
        if (reply !== "") history.push({ role: "assistant", content: reply });
        await releaseSequence();
      }
      break;
    } catch (error) {
      if (error instanceof GenerationCapacityError) {
        // 予算の検査は run の**前**なので KV は 1 行も進んでいない。ただし切り詰めた履歴は
        // 先頭から描き直すことになるので、context は返して組み直す。
        await releaseSequence();
        if (!dropOldestTurn()) {
          // 落とせるものが無い = この 1 発話だけで入り切らない。黙って縮めない（切り詰めの
          // 判断はホストの責務だが、判断材料が尽きたことは利用者に見せる）。
          history.pop();
          write(
            `\n  [入り切らない: ${error.constraint} 上限 ${error.limit} に対し` +
              ` 既存 ${error.pastLength} + prompt ${error.promptLength}` +
              `（この長さなら maxNewTokens ≤ ${error.maxNewTokens}）— 発話を短くするか /reset]\n`,
          );
          break;
        }
        write(
          `\n  [容量超過（${error.constraint} 上限 ${error.limit}）— 古い turn を落として` +
            `再構成（この会話で通る maxNewTokens は ${error.maxNewTokens} だった）]\n`,
        );
        continue;
      }
      if (error === controller.signal.reason) {
        // 中断でも「成功した run のぶんだけ」会話は進んでいる。done は reject ではなく
        // `aborted` で settle するので、生成できた token 数はそのまま読める。
        if (prefilled) clearLine();
        if (stream !== undefined) report(await stream.done, turnStarted);
        if (reply !== "") history.push({ role: "assistant", content: reply });
        else history.pop();
        // prompt が途中まで KV へ入った可能性がある（中断は prefill の chunk 境界でも起きる）。
        // 履歴との対応を再構成する術は無いので、context は捨てる。
        await releaseSequence();
        break;
      }
      throw error;
    } finally {
      turn = undefined;
    }
  }

  write("> ");
}

Deno.removeSignalListener("SIGINT", onInterrupt);
await releaseSequence();
write("\nbye\n");
