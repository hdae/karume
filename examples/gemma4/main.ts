/**
 * Gemma 4 E2B（文字列 → 文字列）の対話デモ — **`Gemma4ChatSession` の写経見本**。
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
 * ## なぜ `chat` ではなく `Gemma4ChatSession` なのか
 *
 * `Gemma4Pipeline.chat` は **1 ターン = 1 sequence** で、過去 turn は毎回 prompt へ描き直される
 * （会話が伸びるほど prefill が O(n²)）。多ターンの対話は `Gemma4ChatSession` が 1 本の
 * sequence を持ち回り、新しい turn の**差分だけ**を流す — 会話の履歴も、KV を継げるかの判定
 * （前ターンが EOS で閉じたか）も、容量が足りないときの切り詰めもセッションの中にある。
 * この台本が書くのは「普通の chat」そのもので、token id も `<bos>` も未 commit frontier も
 * 出てこない:
 *
 *     const session = new Gemma4ChatSession(pipeline, { system, maxNewTokens });
 *     for await (const chunk of session.send(line)) write(chunk);
 *
 * 自分で 1 段下（`sequence()` + `gemma4ChatPrompt` / `gemma4ChatTurn`）を回す形は
 * `packages/models/src/gemma/chat-session.ts` が正本で、そこに写経の中身が全部ある。
 *
 * ## 会話の切り詰めは**注入できる**
 *
 * 容量（`program.capacity`）と位置表（`program.maxPosition`）に入らないターンは、送る前に
 * `onOverflow` へ回る。既定は `dropOldestTurns`（最古の user / assistant の対を落とす・system は
 * 残す）で、この台本は「落としたことを画面に出す」ためだけに包んでいる。落とせるものが尽きたら
 * `GenerationCapacityError` で落ちる — 判断に要る実値は例外の欄が運ぶので、文言を読み解く必要は
 * 無い（ADR 0083 決定 10 の改訂）。
 */

import {
  dropOldestTurns,
  Gemma4ChatSession,
  Gemma4Pipeline,
  GenerationCapacityError,
} from "../../packages/models/gemma.ts";
import type {
  Gemma4ChatSessionOptions,
  Gemma4ChatStop,
  SamplerSpec,
} from "../../packages/models/gemma.ts";
import { denoDirectory } from "../../packages/hub/deno.ts";
import type { AssetProgress } from "../../packages/hub/mod.ts";

const USAGE = "--source <配布形のパス> | --repo <owner/name[@revision]>" +
  " --system <文字列> --max-new-tokens <整数> --temperature <数> --top-k <整数>" +
  " --top-p <数> --seed <整数> --max-resident-ple-bytes <整数>";
const KNOWN = new Set([
  "source",
  "repo",
  "system",
  "max-new-tokens",
  "temperature",
  "top-k",
  "top-p",
  "seed",
  "max-resident-ple-bytes",
]);

/** 取得元の既定（`dist.py --pipeline gemma4` が組むローカルミラー — `docs/assets-layout.md`）。 */
const DEFAULT_SOURCE = "models/karume-gemma4-e2b";

/** 1 ターンで生成する token 数の上限（停止 token は含まれない）。 */
const DEFAULT_MAX_NEW_TOKENS = 256;

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

/**
 * PLE sidecar の常駐上限（バイト・ADR 0085 決定 3 の LRU）。
 *
 * shard は token 範囲で割ってあり、1 文の中でも id は語彙全体へ散るので、常駐が薄いと 1 ターン
 * の間に shard の読み直しが何度も走る（**対話の応答時間**に直に効く）。厚くすればそのぶんホスト
 * RAM を抱える。読み直しが増えても値も token 列も変わらないので、ここは純粋に「RAM と応答時間の
 * 交換レート」を選ぶノブである。
 *
 * 省略時はライブラリの既定（= 最大 shard 2 本ぶん）。全量を載せたいならバイト数で渡す（E2B の
 * 現行世代は shard 1 本 ≈253MiB × 9 本 ≈2.2GiB）。**本数ではなくバイト**なのは、shard 幅が資産
 * 世代で変わるため「N 本」が世代ごとに違う RAM を意味するからである。
 */
const maxResidentPleBytes = integer("max-resident-ple-bytes");

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

const describeStop = (stop: Gemma4ChatStop): string =>
  stop.reason === "eos" || stop.reason === "stop-token"
    ? `${stop.reason}(${stop.token})`
    : stop.reason;

/** 上書きで描く 1 行の幅（短い行が前の行の尻を残さないよう、ここまで空白で埋める）。 */
const LINE_WIDTH = 76;

/** 1 行ぶんの進捗表示を消す（次の出力が食い込まないように空白で塗ってから戻す）。 */
const clearLine = (): void => note(`\r${" ".repeat(LINE_WIDTH)}\r`);

const percent = (part: number, whole: number): string => (part / whole * 100).toFixed(1);
const mib = (bytes: number): string => (bytes / 1024 / 1024).toFixed(0);

/** 取得したファイルに現れた順の通し番号を振る（`AssetProgress` は本数を運ばない — 下の doc）。 */
const fileOrder = new Map<string, number>();

/**
 * 取得の進捗を**ファイル 1 本 = 1 行**で描く。
 *
 * `AssetProgress` は「今のファイル」（`path` / `fileLoaded` / `fileTotal`）と「全体」
 * （`loaded` / `total`）の 2 組を運ぶので、両方を 1 行に出す。1 本終わるたびに改行して行を残す
 * ので、同じ位置を上書きし続ける形（全体の % だけを描く）と違い「進んでいるのか同じ所を
 * 繰り返しているのか」が読める — gemma4 の配布形は重み shard だけで 7 本ある。
 *
 * 通し番号は**この台本が数える**。hub は「何本目 / 全何本」を運ばない（取得は相に分かれていて、
 * 共通層の 1 イベントからは全体の本数を名乗れない）ので、分母は出さずに全体の % とバイト数で
 * 残りを示す。
 */
const showProgress = (
  { phase, path, loaded, total, fileLoaded, fileTotal }: AssetProgress,
): void => {
  const ordinal = fileOrder.get(path) ?? fileOrder.size + 1;
  fileOrder.set(path, ordinal);
  const name = path.slice(path.lastIndexOf("/") + 1);
  const line = `  [${ordinal}] ${name} ${percent(fileLoaded, fileTotal)}%` +
    ` · 全体 ${percent(loaded, total)}% (${mib(loaded)}/${mib(total)} MiB)`;
  note(`\r${line.padEnd(LINE_WIDTH)}${phase === "complete" ? "\n" : ""}`);
};

const started = performance.now();
note(`[gemma4] ${sourceDir ?? repoRef ?? DEFAULT_SOURCE} を読み込む\n`);
await using pipeline = await Gemma4Pipeline.fromPretrained(
  repoRef === undefined ? denoDirectory(sourceDir ?? DEFAULT_SOURCE) : parseRepo(repoRef),
  {
    ...(maxResidentPleBytes === undefined ? {} : { maxResidentPleBytes }),
    onProgress: showProgress,
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
 * セッションの設定（`/reset` は同じ設定で組み直す — 会話を捨てるとはそういうこと）。
 *
 * `onOverflow` を包んでいるのは**画面に出すため**だけで、切り詰めそのものは既定の
 * `dropOldestTurns` に任せる（最古の user / assistant の対を落とし、system は残す）。
 */
const sessionOptions: Gemma4ChatSessionOptions = {
  ...(system === undefined ? {} : { system }),
  maxNewTokens,
  ...(sampler === undefined ? {} : { sampler }),
  onOverflow: (context) => {
    write(
      `\n  [容量超過（上限 ${context.capacity} に対し ${context.needed} 要る）— ` +
        `古い turn を落として再構成]\n`,
    );
    return dropOldestTurns(context);
  },
};
let session = new Gemma4ChatSession(pipeline, sessionOptions);

/**
 * 1 ターンの締め。tok/s は `stop.tokens`（停止 token も 1 個 = 抽選 1 回 = run 1 回）から書く —
 * 出力文字列を符号化し直すと、byte_fallback や停止 token のぶんだけ数がずれる。
 */
const report = (stop: Gemma4ChatStop, at: number): void => {
  const elapsed = (performance.now() - at) / 1000;
  write(
    `\n  [${describeStop(stop)} · ${stop.tokens} tok · ${elapsed.toFixed(1)}s · ` +
      `${(stop.tokens / elapsed).toFixed(1)} tok/s · 会話 ${session.turns.length} 発話]\n`,
  );
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
    // 会話を捨てる = セッションを畳んで同じ設定で組み直す（履歴も KV も持っているのは
    // セッションなので、捨てる口を別に持たない）。
    await session.dispose();
    session = new Gemma4ChatSession(pipeline, sessionOptions);
    write("(reset)\n> ");
    continue;
  }

  const controller = new AbortController();
  turn = controller;
  const turnStarted = performance.now();
  // 発行した stream は必ず汲み切るか break で閉じる（ターンの締めは列の終端で走る）。
  const stream = session.send(line, { signal: controller.signal });
  try {
    // 片は逐次復号器が**確定させたぶん**だけで、連結すると全体の decode と一致する。
    for await (const chunk of stream) write(chunk);
    report(await stream.done, turnStarted);
  } catch (error) {
    if (error instanceof GenerationCapacityError) {
      // 溢れ処理で落とせるものが尽きた = この 1 発話だけで入り切らない。判断に要る実値は
      // 例外の欄が運ぶ（文言を読み解かない）。
      write(
        `\n  [入り切らない: ${error.constraint} 上限 ${error.limit} に対し` +
          ` 既存 ${error.pastLength} + prompt ${error.promptLength}` +
          `（この長さなら maxNewTokens ≤ ${error.maxNewTokens}）— 発話を短くするか /reset]\n`,
      );
    } else if (error === controller.signal.reason) {
      // 中断でも「成功した run のぶんだけ」会話は進んでいる。done は reject ではなく
      // `aborted` で settle するので、生成できた token 数はそのまま読める。
      report(await stream.done, turnStarted);
    } else {
      throw error;
    }
  } finally {
    turn = undefined;
  }

  write("> ");
}

Deno.removeSignalListener("SIGINT", onInterrupt);
await session.dispose();
write("\nbye\n");
