/**
 * chat 1 ターンぶんの**変換と後始末**（{@link Gemma4Pipeline.chat} と `Gemma4ChatSession.send` が
 * 共有する部品）— 生成イベント列 → 確定した文字列片、停止理由の組み立て、ターンの畳み方、
 * そして実行 1 回ごとの観測席の仕立て。
 *
 * ここは Session も直列化鎖も GPU も**所有しない**（所有権と順番待ちは `./pipeline.ts`）。
 * 受け取ったイベント列を文字列にし、後始末の順序を 1 本に保つだけなので、停止文字列の契約も
 * 観測席の呼び出し規則も実 GPU 無しで縛れる（`tests/gemma4_chat_test.ts`）。
 *
 * MUST: 全モジュール副作用ゼロ（import 時実行・グローバル可変状態の禁止 — CLAUDE.md）。
 */

import type {
  GenerationEvent,
  GenerationRunPhase,
  GenerationSpeculation,
  GenerationStop,
  GenerationStream,
} from "../generation/sequence.ts";
import type { StopStringFilter, StreamingDetokenizer } from "../text/detokenizer.ts";

/**
 * prefill の進捗 1 通ぶん（`chunk / chunks` がそのまま進捗）。
 *
 * `chunk` は**commit 済み**の chunk 数（1 始まり）で、`GenerationEvent` の `prefill` と同じ意味・
 * 同じ数である（この層は文字列の面なのでイベント型そのものを出さない）。
 */
export type Gemma4PrefillProgress = {
  readonly chunk: number;
  readonly chunks: number;
};

/**
 * 観測席（{@link Gemma4PipelineOptions.onRunDiagnostics}）が受ける「その 1 通がどの run か」。
 *
 * 席が受けるのは **run 1 本につき 1 通**で、この値はその run が何だったかを言う（順番の勘定
 * ではない）。番号はすべて 1 始まり — `prefill` の `chunk` / `chunks` は `GenerationEvent` の
 * `prefill` と同じ数（commit 済み chunk 数）、`decode` の `step` は**そのターンの** decode run の
 * 番号、`draft` / `verify` の `cycle` は投機の cycle 番号（同じ cycle の 2 本は同じ番号を名乗る）。
 *
 * MUST: 受け手は通知の回数ではなくこの値で分岐する — 複数 chunk に割れた prompt では
 * 「1 通目だけが prefill」が成り立たない。
 * MUST: 生成面の `GenerationRunPhase` **そのもの**である（写した型を持たない — 枝が片方だけ
 * 増えたときに型検査が通り続ける）。この層が足すのは「どちらの Session の診断を引くか」だけ。
 */
export type Gemma4RunPhase = GenerationRunPhase;

/**
 * chat 1 ターンの停止理由。
 *
 * sequence 層の理由（`eos` / `stop-token` / `max-tokens` / `aborted` / `closed`）に、この層でしか
 * 判定できない 1 つ（{@link Gemma4ChatOptions.stopStrings} の一致）を足したもの。`tokens` の
 * 意味は sequence 層と同じ（そのターンが生成した token 数 — 停止 token も 1 個）で、
 * `stop-string` では**停止文字列を含む片を出した token まで**が数に入る。
 */
export type Gemma4ChatStop =
  | GenerationStop
  | {
    readonly reason: "stop-string";
    /** 一致した停止文字列（出力には含まれない）。 */
    readonly stopString: string;
    readonly tokens: number;
    /**
     * 投機の勘定（`GenerationStop.speculation` をそのまま写したもの — 投機 sequence の
     * ターンだけ載る）。
     *
     * MUST: 写す。この枝は理由を差し替えるために object を組み直すので、写さないと
     * 「停止文字列で閉じたターンだけ勘定が消える」形になる（例外にならない欠落）。
     */
    readonly speculation?: GenerationSpeculation;
  };

/**
 * 文字列片の列（`for await` で汲む）+ 停止理由 + 一括で受け取る口。
 *
 * 片は逐次復号器が**確定させたぶん**だけで（ADR 0084 決定 4）、byte_fallback の途中は次の
 * token まで持ち越される。連結すると `decode(全 token id)` と一致する（停止文字列で切った
 * ターンだけは、その手前までになる）。
 *
 * MUST: **1 つのストリームは 1 通りにしか消費できない** — 反復（`for await`）と
 * {@link Gemma4ChatStream.text} の併用も、2 度の反復も、同期に throw する。生成は 1 度しか
 * 走らないので、2 通り目には「残り」しか流れない（先に汲んだ側だけが本文を持つ）— 例外に
 * ならない取り違えなので、口の側で塞ぐ。
 *
 * MUST: `done` は**二次的な**通知路である（`GenerationStream.done` と同じ規律）— 失敗は
 * iterable 側が throw するのが一次で、`done` は同じ例外で reject するだけ。
 */
export type Gemma4ChatStream = AsyncIterable<string> & {
  readonly done: Promise<Gemma4ChatStop>;
  /**
   * 汲み切って連結した 1 本の文字列（逐次表示が要らない呼び手の口）。
   *
   * 反復と同じ列を同じ順で汲むだけなので、`text()` の結果は「片を全部連結したもの」と一致する。
   * 停止理由が要るなら {@link Gemma4ChatStream.done} を併せて読む（`text()` の後でよい）。
   */
  text(): Promise<string>;
};

/**
 * 生成イベント → **確定した文字列片**（復号 → 停止文字列の判定）。返り値は一致した停止文字列
 * （`undefined` = 止まらずに列が終わった）。
 *
 * MUST: 停止文字列で止めるときは `return` で抜ける — `for await` の脱出はイベント列の
 * `return()` を呼ぶので、sequence は**中断（`break`）と同じ後始末**で畳まれる。畳み方を自前で
 * 書くと、KV の committed 整合（未 commit frontier 1 token）が 2 実装に分かれる。
 *
 * NOTE: 停止 token（sequence 層）と違い、停止文字列は復号の**後**でしか判定できない — 1 つの
 * 停止文字列が複数 token に割れることも、1 つの token が停止文字列の末尾と次の本文をまたぐ
 * こともあるため。だから席が 2 層に分かれる（ADR 0083 追記 2026-09-02）。
 *
 * NOTE: barrel（`mod.ts` / `./gemma`）には出さない**内部の口**である（公開の入口は
 * {@link Gemma4Pipeline.chat} だけ）。export してあるのは、この単位なら停止文字列の契約を
 * 実 GPU 無しで縛れるため（`tests/gemma4_chat_test.ts`）。
 */
export const decodeChatChunks = async function* (
  events: AsyncIterable<GenerationEvent>,
  detokenizer: StreamingDetokenizer,
  stopStrings: StopStringFilter,
  onPrefill?: (progress: Gemma4PrefillProgress) => void,
  onToken?: (id: number) => void,
): AsyncGenerator<string, string | undefined, undefined> {
  for await (const event of events) {
    if (event.kind === "prefill") {
      // 文字列の面には prefill の片が無い（本文はまだ 1 文字も出ていない）ので、進捗だけを
      // 観測席へ渡す。例外は握らない（fail loudly — 呼び手のコールバックの誤りを飲まない）。
      onPrefill?.({ chunk: event.chunk, chunks: event.chunks });
      continue;
    }
    onToken?.(event.id);
    const chunk = stopStrings.push(detokenizer.push(event.id));
    if (chunk.text !== "") yield chunk.text;
    if (chunk.matched !== undefined) return chunk.matched;
  }
  // 復号器の持ち越し（byte_fallback の run）を確定させたぶんも判定へ通す — 停止文字列の最後の
  // 1 文字がその run の中に居ることがある。
  const tail = stopStrings.push(detokenizer.finish());
  if (tail.text !== "") yield tail.text;
  if (tail.matched !== undefined) return tail.matched;
  // 止まらずに終わったターンは、接頭辞として保留していたぶんを最後に流す（1 文字も落とさない —
  // 保留は判定のための遅延であって、出力の切り詰めではない）。
  const held = stopStrings.finish();
  if (held !== "") yield held;
  return undefined;
};

/**
 * 片の generator + 停止理由 → 公開の {@link Gemma4ChatStream}（**1 通りにしか消費できない**口）。
 *
 * MUST: 反復と {@link Gemma4ChatStream.text} は**同じ generator**を汲む（別経路を作らない）—
 * 生成は 1 度しか走らないので、一括の口が独自のループを持つと「どちらで読んだかで結果が違う」
 * 形が書けてしまう。2 通り目は静かに空を返すだけで例外にならないので、口の側で塞ぐ。
 *
 * NOTE: {@link decodeChatChunks} と同じく barrel には出さない内部の口である。
 */
export const chatStreamOf = (
  chunks: AsyncGenerator<string, void, undefined>,
  done: Promise<Gemma4ChatStop>,
): Gemma4ChatStream => {
  let claimed: "反復" | "text()" | undefined;
  const claim = (how: "反復" | "text()"): void => {
    if (claimed !== undefined) {
      throw new Error(
        `Gemma4ChatStream: 1 つのストリームは 1 通りにしか消費できない` +
          `（${claimed} で消費済み — ${how} は同じ生成をもう一度読もうとしている）`,
      );
    }
    claimed = how;
  };
  return {
    [Symbol.asyncIterator]: (): AsyncGenerator<string, void, undefined> => {
      claim("反復");
      return chunks;
    },
    done,
    // async にしない — 併用の検査は**同期に**落とす（返り値を await するまで気づけない形に
    // しない。`generate` の寿命検査と同じ規律）。
    text: (): Promise<string> => {
      claim("text()");
      return joinChunks(chunks);
    },
  };
};

/**
 * 観測席（{@link Gemma4PipelineOptions.onRunDiagnostics}）を生成面の `onRun` hook に仕立てる。
 *
 * 席が pipeline 層にあるのは、`GenerationSequence` が**パイプライン非依存**だからである
 * （Session も診断も知らない — ADR 0083）。生成面は run 1 本につき 1 回、その run の出力を
 * 読み終えた**同期区間**でこの hook を呼ぶので、この層がするのは「どちらの Session の診断を
 * 引くか」を `phase.kind` で決めることだけである。
 *
 * かつてはイベント列（`GenerationEvent`）を包んで run 数を**導出**していた（`withRunDiagnostics`）。
 * 導出は 2 つの例外を抱えていた — prefill 直後の最初の token は run を伴わない・停止 token を
 * 引いた最後の decode run は列に出ない（`done` から補っていた）— うえ、投機では 1 verify run が
 * 複数 token を出すので導出そのものが成り立たない。run の発行元が直接名乗る形（ADR 0083 追記
 * 〈hook〉）にすると、どちらの例外も消える。
 *
 * MUST: 観測席が無ければ `undefined` を返す（hook を渡さない = 生成面が 1 回も呼ばない）。
 * MUST: `draft` は**借り手**（drafter Session）の診断を引く。貸し手のものを渡すと、draft run の
 * 診断として「その前の verify run」の値が届く（例外にならない取り違え）。
 *
 * NOTE: 診断の型を型引数にしてあるのは、この関数が診断の**中身を 1 つも読まない**（席へ素通し
 * するだけ）ことを型で示すためで、同時に呼び出し規則の門（`gemma4_chat_test.ts`）が実 Session
 * 無しで書ける。{@link Gemma4State} は `SessionDiagnostics` でそのまま満たす。
 * NOTE: `export` は門を直接叩くテストのため（`mod.ts` / サブパス面には出さない — ADR 0008）。
 */
export const runDiagnosticsHook = <D>(
  state: {
    readonly session: { diagnostics: () => D };
    readonly drafter?: { readonly session: { diagnostics: () => D } };
    readonly onRunDiagnostics?: (diagnostics: D, phase: Gemma4RunPhase) => void;
  },
): ((phase: Gemma4RunPhase) => void) | undefined => {
  const listener = state.onRunDiagnostics;
  if (listener === undefined) return undefined;
  return (phase: Gemma4RunPhase): void => {
    if (phase.kind !== "draft") {
      listener(state.session.diagnostics(), phase);
      return;
    }
    const drafter = state.drafter;
    // draft run は drafter Session でしか起きない（居なければ簿記の破れ — 黙って貸し手の
    // 診断を渡すと、別の run の値が draft の名前で積算される）。
    if (drafter === undefined) {
      throw new Error(
        "Gemma4Pipeline: drafter が居ないのに draft run の観測が届いた",
      );
    }
    listener(drafter.session.diagnostics(), phase);
  };
};

/**
 * 停止文字列で閉じたターンの停止理由（`chat` と `Gemma4ChatSession.send` が共有する 1 本）。
 *
 * 理由と綴りはこの層の判定だが、`tokens` と `speculation` は**内側の値をそのまま写す**
 * （この層で数え直さない）。写す欄が増えたときに片方の入口だけ古いまま残るのを防ぐため、
 * 組み立てを 1 本にしてある。
 */
export const stopStringOf = (
  stopString: string,
  inner: GenerationStop,
): Gemma4ChatStop => ({
  reason: "stop-string",
  stopString,
  tokens: inner.tokens,
  ...(inner.speculation === undefined ? {} : { speculation: inner.speculation }),
});

/**
 * ターンの後始末 1 本（`chat` と `Gemma4ChatSession.send` が共有する）。
 *
 * MUST: `release` は**無条件に**呼ぶ。`cleanup`（sequence の返却・セッションの締め）が投げたら
 * 席を返さない形にすると、直列化鎖は前段の決着を得られないまま以後の `chat` / `dispose` を
 * 永久に待つ — 例外 1 つで二度と動かないパイプラインになる（device 消失時に `context.dispose`
 * が `flush` の失敗を伝播させる経路が実在する）。順序は flush-before-destroy のまま
 * 「`cleanup` → `release`」である。
 *
 * MUST: 本体（`failure`）も失敗しているときは**両方**運ぶ。呼び手の `finally` から呼ぶので、
 * ここで投げる例外は本体の例外を置き換える — 包まずに `AggregateError` へ 2 本とも載せる
 * （`errors[0]` が本体・`errors[1]` が後始末。中断の識別 `error === signal.reason` は
 * `errors[0]` に残る）。
 *
 * NOTE: 関数に切り出してあるのは、呼び手の `finally` に制御フロー文を置かないため
 * （`no-unsafe-finally` が禁ずるのは「元の例外を黙って捨てる」形で、ここは捨てずに畳んでいる）。
 */
export const closeChatTurn = async (
  where: string,
  failure: { readonly error: unknown } | undefined,
  cleanup: () => Promise<void>,
  release?: () => void,
): Promise<void> => {
  try {
    await cleanup();
  } catch (error) {
    if (failure === undefined) throw error;
    throw new AggregateError(
      [failure.error, error],
      `${where}: ターン本体と後始末の両方が失敗した`,
    );
  } finally {
    release?.();
  }
};

/** 後始末とリース返却が終わってから、iterable と同じ成否を done へ通知する。 */
export const completeChatTurn = async (options: {
  readonly where: string;
  readonly stream?: GenerationStream;
  readonly matched?: string;
  readonly failure?: { readonly error: unknown };
  readonly cleanup: (stop: Gemma4ChatStop | undefined) => Promise<void>;
  readonly release?: () => void;
  readonly settle: (stop: Gemma4ChatStop) => void;
  readonly fail: (error: unknown) => void;
}): Promise<void> => {
  let failure = options.failure;
  let stop: Gemma4ChatStop | undefined;
  try {
    const inner = options.stream === undefined
      ? { reason: "closed", tokens: 0 } satisfies Gemma4ChatStop
      : await options.stream.done;
    // 中断は iterable が reason を投げ、done は aborted を返す既存契約を維持する。
    if (failure === undefined || inner.reason === "aborted") {
      stop = options.matched === undefined ? inner : stopStringOf(options.matched, inner);
    }
  } catch (error) {
    failure ??= { error };
  }
  try {
    await closeChatTurn(
      options.where,
      failure,
      () => options.cleanup(stop),
      options.release,
    );
  } catch (error) {
    options.fail(error);
    throw error;
  }
  if (stop === undefined && failure !== undefined) {
    options.fail(failure.error);
    throw failure.error;
  }
  options.settle(stop ?? { reason: "closed", tokens: 0 });
};

/** 片を汲み切って連結する（{@link Gemma4ChatStream.text} の本体）。 */
const joinChunks = async (chunks: AsyncIterable<string>): Promise<string> => {
  let text = "";
  for await (const chunk of chunks) text += chunk;
  return text;
};
