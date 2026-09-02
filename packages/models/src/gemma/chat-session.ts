/**
 * `Gemma4ChatSession` — **普通の多ターン chat** を持ち回るための中間層。
 *
 * 面は 3 層になる。低い方から:
 *
 * 1. `Gemma4Pipeline.sequence()`（ADR 0083 決定 1〜4）— token id と KV の寿命を自分で回す
 * 2. **ここ** — 会話（`Gemma4ChatMessage` の並び）と KV の対応を持ち、溢れたら注入された
 *    ポリシーで切り詰める
 * 3. `Gemma4Pipeline.chat()` — 1 ターン = 1 sequence（過去 turn は毎回描き直す）
 *
 * 3 だけだと会話が伸びるほど prefill が O(n²) になり、避けようとすると 1 の世界（token id /
 * `<bos>` / 未 commit frontier）へ一気に落ちる。この層が持つのは**その落差だけ**で、生成そのもの
 * は 1 に、描画と復号と停止文字列は `chat()` と同じ実装（{@link decodeChatChunks} /
 * {@link chatStreamOf}）にそのまま乗る — `send` が返す `Gemma4ChatStream` は `chat()` の返り値と
 * 同じ型・同じ契約である。
 *
 * ## KV を継ぐ条件
 *
 * 直前のターンが**配布形の EOS で閉じた**（`GenerationStop.reason === "eos"`）ときだけ、次の
 * ターンは差分（{@link gemma4ChatTurn}）で継ぐ。閉じ札は生成が出して sequence が未 commit
 * frontier（`pendingToken`）に持っており、次の `generate` が prompt の先頭へ前置する（ADR 0083
 * 決定 4）。max-tokens / 停止文字列 / 中断で打ち切ったターンの後ろは model turn が閉じていない
 * ので、sequence を捨てて履歴から描き直す。
 *
 * NOTE: 配布形の EOS 集合には `<eos>` も `<|tool_response>` も居る（`gemma4StopTokens`）。
 * `<turn|>` 以外で閉じたターンを継ぐと綴りは上流の template と 1 token だけ食い違うが、会話が
 * 欠けることはない（差分の先頭は常に「閉じ札の直後」である）。
 *
 * ## 溢れ処理は**注入可能**（ADR 0083 決定 10 の改訂）
 *
 * 低レベル面は今も「切り詰めはホストの責務」で、`GenerationCapacityError` を投げて終わる。この
 * 層は既定のポリシー（{@link dropOldestTurns} = 最古の user / assistant の対を落とす・system は
 * 残す）を持ち、{@link Gemma4ChatSessionOptions.onOverflow} で丸ごと差し替えられる。ホストが
 * throw する関数を渡せば従来どおりホスト側で扱える。
 *
 * MUST: 全モジュール副作用ゼロ（import 時実行・グローバル可変状態の禁止 — CLAUDE.md）。
 */

import type { GenerationProgram } from "../generation/program.ts";
import type { SamplerSpec } from "../generation/sampler.ts";
import {
  type GenerationCapacityDetail,
  GenerationCapacityError,
  type GenerationSequence,
  type GenerationStream,
} from "../generation/sequence.ts";
import { createStopStringFilter } from "../text/detokenizer.ts";
import type { Gemma4DefaultSampler } from "./config.ts";
import {
  chatStreamOf,
  decodeChatChunks,
  type Gemma4ChatStop,
  type Gemma4ChatStream,
} from "./pipeline.ts";
import { type Gemma4ChatMessage, gemma4ChatPrompt, gemma4ChatTurn } from "./text/chat.ts";
import type { GemmaTokenizer } from "./text/tokenizer.ts";

/**
 * セッションがパイプラインから読む面（{@link Gemma4Pipeline} がそのまま満たす）。
 *
 * 面を絞ってあるのは、この層が**公開面より内側を 1 つも使っていない**ことを型で示すため
 * （消費者が同じものを自分で書ける = 写経見本としての価値がここにある）。テストが実 GPU 無しで
 * 会話の組み立てだけを見られるのも同じ性質の帰結である。
 */
export type Gemma4ChatSessionHost = {
  readonly tokenizer: GemmaTokenizer;
  readonly program: GenerationProgram;
  readonly defaultSampler: Gemma4DefaultSampler | undefined;
  sequence(): Promise<GenerationSequence>;
};

/**
 * 溢れ処理に渡る文脈（「今の会話」と「あとどれだけ足りないか」）。
 *
 * `capacity` / `needed` は**同じ物差し**（会話が占める論理位置の数）で、`needed > capacity` が
 * 呼ばれた理由である。`capacity` は配布形の 2 つの上限のうち**先に効いた方**（state 容量と位置表
 * の小さい方）で、どちらでも打つ手は同じ（会話を短くする）。
 */
export type Gemma4ChatOverflow = {
  /** 現在の履歴（system 発話を宣言していれば先頭がそれ）。 */
  readonly turns: readonly Gemma4ChatMessage[];
  /** セッションが宣言した system 発話（`undefined` = 宣言なし）。 */
  readonly system: string | undefined;
  /** この会話が使える論理位置の上限。 */
  readonly capacity: number;
  /** このターンを通すのに要る論理位置の数（prompt + `maxNewTokens` − 1）。 */
  readonly needed: number;
};

/**
 * 溢れたときに**新しい履歴**を返す関数（既定は {@link dropOldestTurns}）。
 *
 * MUST: 返す履歴は渡された `turns` より**短い**こと。同じ長さ（= 打つ手が無い）を返せば
 * セッションが `GenerationCapacityError` で落とす — 縮まない再試行は無限ループにしかならず、
 * 「会話が入り切らない」という事実を握り潰す形でもある。throw すればそのまま呼び手へ届く。
 */
export type Gemma4ChatOverflowPolicy = (
  context: Gemma4ChatOverflow,
) => readonly Gemma4ChatMessage[] | Promise<readonly Gemma4ChatMessage[]>;

/** セッション 1 本の設定（ターンごとに変える指定は {@link Gemma4ChatTurnOptions}）。 */
export type Gemma4ChatSessionOptions = {
  /** 会話の先頭に置く system 発話（省略時は置かない）。 */
  readonly system?: string;
  /** 1 ターンで生成する token 数の上限（ターンごとに上書きできる）。 */
  readonly maxNewTokens: number;
  /**
   * このセッションの sampling 指定。省略時は配布形の宣言
   * （{@link Gemma4ChatSessionHost.defaultSampler}）で、それも無ければ温度 0（greedy）。
   */
  readonly sampler?: SamplerSpec;
  /** 容量が足りないときに履歴を作り直す関数（既定 {@link dropOldestTurns}）。 */
  readonly onOverflow?: Gemma4ChatOverflowPolicy;
};

/**
 * 1 ターンだけ効かせる指定（{@link Gemma4Pipeline.chat} の要求と同じ語彙・意味も同じ）。
 *
 * MUST: 中身は {@link Gemma4ChatSession.send} が**発行時に写す**（ADR 0083 追記 2026-09-02）—
 * 返った列を汲み始めた後に書き換えても、走行中のターンには効かない。
 */
export type Gemma4ChatTurnOptions = {
  /** 省略時はセッションの {@link Gemma4ChatSessionOptions.maxNewTokens}。 */
  readonly maxNewTokens?: number;
  /** このターンだけ効かせる追加の停止 token（配布形の EOS 集合との和集合）。 */
  readonly stopTokens?: readonly number[];
  /** このターンだけ効かせる停止文字列（復号後の本文で判定する）。 */
  readonly stopStrings?: readonly string[];
  /** 省略時はセッションの sampler。 */
  readonly sampler?: SamplerSpec;
  /** 中断（`signal.reason` をそのまま throw する — ADR 0083 決定 5）。 */
  readonly signal?: AbortSignal;
};

/**
 * 既定の溢れ処理 — **最古の user / assistant の対を落とす**（system 発話は残す）。
 *
 * 2 件ずつ落とすのは、片方だけ落とすと「答えだけが残った会話」「問いだけが残った会話」になり、
 * モデルに見せる文脈としては壊れているため。落とせるものが無ければ `turns` をそのまま返す
 * （= セッションが `GenerationCapacityError` で落とす — この 1 発話だけで入り切らない）。
 */
export const dropOldestTurns: Gemma4ChatOverflowPolicy = (
  { turns, system },
): readonly Gemma4ChatMessage[] => {
  const floor = system === undefined ? 0 : 1;
  // 末尾の 1 件（= 今答えようとしている発話）は落とさない。
  const droppable = turns.length - floor - 1;
  if (droppable <= 0) return turns;
  return [...turns.slice(0, floor), ...turns.slice(floor + Math.min(2, droppable))];
};

/**
 * このターンが上限を踏むか（踏むなら実値を返す）。
 *
 * `GenerationSequence` の予算検査（`sequence.ts` の `assertBudget`）と**同じ 2 式**を、run の直前
 * ではなく **prompt を組んだ直後**に見る。切り詰めは「例外が出てから」ではなく「送る前」に決めら
 * れる方が自然で、そのために `GenerationSequence.used` が公開されている（ADR 0083 追記
 * 2026-08-31）。
 *
 * NOTE: 欄の割り振りは sequence 層と 1 だけずれる — この層の `pastLength` は `used`（未 commit
 * frontier 込み）で、`promptLength` はこの層が渡す prompt の長さである。合計も
 * `maxNewTokens`（= 今なら通る上限）も sequence 層の欄と一致する。
 */
const overflowOf = (
  program: GenerationProgram,
  used: number,
  promptLength: number,
  maxNewTokens: number,
): GenerationCapacityDetail | undefined => {
  const detail = (
    constraint: GenerationCapacityDetail["constraint"],
    limit: number,
  ): GenerationCapacityDetail => ({
    constraint,
    pastLength: used,
    promptLength,
    requestedNewTokens: maxNewTokens,
    limit,
    maxNewTokens: limit - used - promptLength + 1,
  });
  const peak = used + promptLength + maxNewTokens - 1;
  // 順序は sequence 層と同じ（両方踏むターンで同じ constraint を名乗る）。
  if (peak - 1 >= program.maxPosition) return detail("maxPosition", program.maxPosition);
  if (peak > program.capacity) return detail("capacity", program.capacity);
  return undefined;
};

/** 会話が占める論理位置の数（{@link Gemma4ChatOverflow.needed}）。 */
const neededOf = (detail: GenerationCapacityDetail): number =>
  detail.pastLength + detail.promptLength + detail.requestedNewTokens - 1;

/**
 * 1 本の会話（履歴 + KV）を持つセッション。
 *
 * ```ts
 * await using pipeline = await Gemma4Pipeline.fromPretrained(ref);
 * await using session = new Gemma4ChatSession(pipeline, { maxNewTokens: 256 });
 * for await (const chunk of session.send("Name a color.")) Deno.stdout.write(encode(chunk));
 * console.log(await session.send("Another one.").text());
 * ```
 *
 * MUST: 同時に走らせられるターンは 1 本だけ（2 本目の `send` は**同期に** throw する）。
 * 履歴は 1 本の会話として順に積まれるので、2 本の生成が同じ履歴を押すと「答えの無い問い」を
 * 挟んだ会話が KV へ入る — 例外にならない取り違えなので、口の側で塞ぐ。
 *
 * MUST: 発行した stream は**汲み切るか `break` で閉じる**。ターンの締め（履歴への追記・KV を
 * 継ぐかの判定・次の `send` の受付）は列の終端で走るので、汲まずに捨てた stream はセッションを
 * そのターンのまま止める。
 */
export class Gemma4ChatSession {
  readonly #host: Gemma4ChatSessionHost;
  readonly #system: string | undefined;
  readonly #maxNewTokens: number;
  readonly #sampler: SamplerSpec | undefined;
  readonly #onOverflow: Gemma4ChatOverflowPolicy;
  /** 会話の履歴（この層の唯一の可変状態 — sequence は transcript を持たない）。 */
  #turns: Gemma4ChatMessage[];
  /** 現在の sequence（`undefined` = 次のターンで作り直す）。 */
  #sequence: GenerationSequence | undefined;
  /** その sequence の KV に入っている履歴の件数（0 = 空 = 全体を描き直す）。 */
  #committed = 0;
  /** ターンが走っている（発行済みで、まだ列が終端していない）。 */
  #busy = false;
  /** dispose の 1 本。**undefined でないことが「dispose 済み」**（派生状態を別に持たない）。 */
  #disposal: Promise<void> | undefined;

  constructor(host: Gemma4ChatSessionHost, options: Gemma4ChatSessionOptions) {
    this.#host = host;
    this.#system = options.system;
    this.#maxNewTokens = options.maxNewTokens;
    this.#sampler = options.sampler;
    this.#onOverflow = options.onOverflow ?? dropOldestTurns;
    this.#turns = options.system === undefined ? [] : [{ role: "system", content: options.system }];
  }

  /**
   * これまでの会話（system 発話を宣言していれば先頭がそれ）。
   *
   * 凍結**コピー**を返す — 内部の配列をそのまま出すと、消費側の `push` / `splice` が KV との
   * 対応（{@link Gemma4ChatSession} の「KV を継ぐ条件」）を静かに壊す。
   */
  get turns(): readonly Gemma4ChatMessage[] {
    return Object.freeze([...this.#turns]);
  }

  /**
   * 発話を 1 つ送り、**確定した文字列片**を流す（{@link Gemma4Pipeline.chat} と同じ列・同じ
   * 停止理由・同じ `text()`）。
   *
   * 前のターンが EOS で閉じていれば KV を継いで**差分だけ**を流し、そうでなければ sequence を
   * 作り直して履歴を描き直す。容量が足りないターンは**送る前に**溢れ処理へ回る
   * （{@link Gemma4ChatSessionOptions.onOverflow}）。
   */
  send(text: string, options: Gemma4ChatTurnOptions = {}): Gemma4ChatStream {
    if (this.#disposal !== undefined) {
      throw new Error("Gemma4ChatSession: dispose 済みでは生成できない");
    }
    if (this.#busy) {
      throw new Error(
        "Gemma4ChatSession: 前のターンがまだ終わっていない" +
          "（1 セッション = 1 生成 — 発行した stream は汲み切るか break で閉じる）",
      );
    }
    // MUST: 要求は**発行時に写す**（ADR 0083 追記 2026-09-02）。本体（async generator）は最初の
    // `next()` まで走らないので、ここで読み切らないと「汲み始めた時点の値」で走る。
    const maxNewTokens = options.maxNewTokens ?? this.#maxNewTokens;
    const sampler = options.sampler ?? this.#sampler ?? this.#host.defaultSampler;
    const stopTokens = options.stopTokens === undefined ? undefined : [...options.stopTokens];
    const signal = options.signal;
    // 停止文字列の状態機械もここで作る（指定の検査と複製がその中で済む = 受理集合が同期に落ちる）。
    const stopStrings = createStopStringFilter(options.stopStrings ?? []);
    const detokenizer = this.#host.tokenizer.createDetokenizer();

    const asked: Gemma4ChatMessage = { role: "user", content: text };
    this.#turns.push(asked);
    this.#busy = true;

    let settle!: (stop: Gemma4ChatStop) => void;
    let fail!: (error: unknown) => void;
    const done = new Promise<Gemma4ChatStop>((resolve, reject) => {
      settle = resolve;
      fail = reject;
    });
    // 二次的な通知路なので、読まれなくても unhandled rejection にしない。
    done.catch(() => {});

    const prepare = this.#prepare.bind(this);
    const finish = this.#finish.bind(this);
    const chunks = async function* (): AsyncGenerator<string, void, undefined> {
      let stream: GenerationStream | undefined;
      let failure: { readonly error: unknown } | undefined;
      /** 一致した停止文字列（この層の停止理由）。 */
      let matched: string | undefined;
      /** このターンで流した本文（履歴へ積む assistant の中身）。 */
      let reply = "";
      try {
        const { sequence, prompt } = await prepare(maxNewTokens);
        stream = sequence.generate({
          prompt,
          maxNewTokens,
          ...(stopTokens === undefined ? {} : { stopTokens }),
          ...(sampler === undefined ? {} : { sampler }),
          ...(signal === undefined ? {} : { signal }),
        });
        const parts = decodeChatChunks(stream, detokenizer, stopStrings);
        try {
          for (;;) {
            const step = await parts.next();
            if (step.done === true) {
              matched = step.value;
              break;
            }
            reply += step.value;
            yield step.value;
          }
        } finally {
          // MUST: 消費側の `break` を内側へ伝える（`yield*` の委譲と同じ形）。伝えないと
          // イベント列の `finally` が走らず、sequence が走行中のまま残る。
          await parts.return(undefined);
        }
      } catch (error) {
        failure = { error };
        // MUST: 包まずそのまま投げる（ADR 0083 決定 5 — 消費側が `error === signal.reason` で
        // 自分の中断を識別できる）。
        throw error;
      } finally {
        let stop: Gemma4ChatStop | undefined;
        if (stream === undefined) {
          if (failure !== undefined) fail(failure.error);
          else settle({ reason: "closed", tokens: 0 });
        } else {
          try {
            const inner = await stream.done;
            // 停止文字列だけはこの層の判定なので理由を差し替える（`tokens` は内側の数をそのまま
            // 使う = この層で数え直さない）。
            stop = matched === undefined
              ? inner
              : { reason: "stop-string", stopString: matched, tokens: inner.tokens };
            settle(stop);
          } catch (error) {
            fail(error);
          }
        }
        await finish(asked, reply, stop);
      }
    };

    return chatStreamOf(chunks(), done);
  }

  /**
   * 解放する。走行中のターンの後に sequence を畳む（`dispose` は sequence 自身の直列化鎖に
   * 載るので、flush-before-destroy はそちらが持つ）。2 度目以降も同じ完了を返す。
   *
   * MUST: パイプラインは畳まない（所有者は渡した側）。
   */
  dispose(): Promise<void> {
    this.#disposal ??= this.#releaseSequence();
    return this.#disposal;
  }

  /** `await using` 対応（Explicit Resource Management）— {@link dispose} の別名。 */
  [Symbol.asyncDispose](): Promise<void> {
    return this.dispose();
  }

  /**
   * このターンの sequence と prompt を用意する（容量が足りなければ溢れ処理を回してから）。
   *
   * MUST: 再試行は履歴が**縮んだ**ときだけ続く（`#shrink` の門）ので、
   * 試行回数は入口の発話数を超えない — 下の門はその不変条件が破れたことを名指しで落とす席で、
   * 成り立っている限り踏まない。
   */
  async #prepare(
    maxNewTokens: number,
  ): Promise<{ readonly sequence: GenerationSequence; readonly prompt: number[] }> {
    const attempts = this.#turns.length;
    for (let attempt = 0;; attempt += 1) {
      if (this.#sequence === undefined) {
        this.#sequence = await this.#host.sequence();
        this.#committed = 0;
      }
      const sequence = this.#sequence;
      // 差分を継ぐのは「直前の発話 1 件だけが未 commit」のときに限る。ずれたまま差分を流すと
      // 会話が静かに欠けるので、継承の前提そのものを門にしておく。
      if (this.#committed !== 0 && this.#committed !== this.#turns.length - 1) {
        throw new Error(
          `Gemma4ChatSession 内部不整合: KV は ${this.#committed} 件だが履歴は ` +
            `${this.#turns.length} 件`,
        );
      }
      const prompt = this.#committed === 0
        ? gemma4ChatPrompt(this.#host.tokenizer, this.#turns)
        : gemma4ChatTurn(this.#host.tokenizer, this.#turns[this.#turns.length - 1]);
      const detail = overflowOf(this.#host.program, sequence.used, prompt.length, maxNewTokens);
      if (detail === undefined) return { sequence, prompt };
      if (attempt >= attempts) {
        throw new Error(
          `Gemma4ChatSession 内部不整合: 溢れ処理が ${attempts} 回で収束しなかった`,
        );
      }
      await this.#shrink(detail);
    }
  }

  /** 溢れ処理を 1 回回す（履歴を差し替え、KV との対応が切れた sequence を捨てる）。 */
  async #shrink(detail: GenerationCapacityDetail): Promise<void> {
    const before = this.#turns.length;
    const needed = neededOf(detail);
    const next = await this.#onOverflow({
      turns: Object.freeze([...this.#turns]),
      system: this.#system,
      capacity: detail.limit,
      needed,
    });
    if (next.length >= before) {
      throw new GenerationCapacityError(
        `会話が入り切らない: ${detail.constraint} 上限 ${detail.limit} に対し ${needed} 要る` +
          `（溢れ処理は履歴 ${before} 件を縮めなかった — 発話を短くするか maxNewTokens を` +
          ` 下げる。この長さなら maxNewTokens ≤ ${detail.maxNewTokens}）`,
        detail,
      );
    }
    this.#turns = [...next];
    // 切り詰めた履歴は先頭から描き直すことになるので、context は返して組み直す。
    await this.#releaseSequence();
  }

  /** ターンの締め（履歴への追記・KV を継ぐかの判定・次の `send` の受付）。 */
  async #finish(
    asked: Gemma4ChatMessage,
    reply: string,
    stop: Gemma4ChatStop | undefined,
  ): Promise<void> {
    try {
      if (stop?.reason === "eos") {
        // 配布形の EOS で閉じた = model turn が閉じている。本文が空でも席は積む（KV には空の
        // model turn が入っているので、積まないと描き直しが KV と別の会話になる）。
        this.#turns.push({ role: "assistant", content: reply });
        this.#committed = this.#turns.length;
        return;
      }
      // max-tokens / 停止文字列 / 中断 / 失敗は model turn を閉じていない = 差分の前提が無い。
      if (reply !== "") this.#turns.push({ role: "assistant", content: reply });
      // 1 文字も出なかったターンは**無かったことにする**（答えの無い問いを履歴へ残さない）。
      else if (this.#turns.at(-1) === asked) this.#turns.pop();
      await this.#releaseSequence();
    } finally {
      this.#busy = false;
    }
  }

  /** context を返して組み直しに備える（KV と履歴の対応が切れたときは必ずここを通る）。 */
  async #releaseSequence(): Promise<void> {
    const held = this.#sequence;
    this.#sequence = undefined;
    this.#committed = 0;
    await held?.dispose();
  }
}
