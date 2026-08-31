/**
 * gemma4 の chat フォーマット — **ファミリ固有の純関数 1 本**（ADR 0084 決定 5）。
 *
 * 汎用テンプレートレンダラは作らない。上流の `chat_template.jinja` は 386 行 / 18,569B ある
 * が、**素の会話だけなら綴りは 5 個・分岐は 3 本**で、Jinja 相当の評価器を持ち込むことは
 * 「ランタイム依存は Web 標準のみ」（CLAUDE.md 横断不変条件）と「投機的な一般化をしない」の
 * 両方に反する。前例は ADR 0079（テキスト解析は呼び手の責務・変換関数 1 本）。
 *
 * ## 綴りは Gemma 3 系ではない
 *
 * `<|turn>` / `<turn|>` / `<|channel>` / `<|think|>` が正本で（`tokenizer_config.json` の
 * `sot_token` / `eot_token` / `soc_token` / `think_token`）、**`<start_of_turn>` はこの語彙に
 * 無い**（ADR 0084 Context 3）。消費者に手書きさせない理由がここにある。
 *
 * ## MUST: `<bos>` の所有者はここ
 *
 * `GemmaTokenizer.encode` は `<bos>` を付けない（gemma4 の post_processor の実測と一致）。
 * 付けるのは {@link renderGemma4Chat} だけで、分けないと chat 導入時に double-BOS になる
 * （ADR 0084 決定 5 の MUST）。上流も同じ形である — template が `bos_token` を描画し、
 * `apply_chat_template` は `add_special_tokens=False` で符号化する。
 *
 * ## MUST: 射程外は fail loudly
 *
 * 初版の射程は「素の会話」だけ（ADR 0084 決定 5・6.3）。tools / thinking / tool_call /
 * 画像・音声パート / 未知 role は**黙って無視しない** — 無視すると「tool を渡したのに
 * 使われない」が例外なしで通る。射程を広げるのは実需が出てからで、そのときは Python 側の
 * 電池（`tools/export-recipes/gemma4/chat.py` の docstring）にケースを足すところから始まる。
 *
 * MUST: 全モジュール副作用ゼロ（表もインスタンスもモジュールスコープで組み立てない）。
 */

import type { GemmaTokenizer } from "./tokenizer.ts";

/** 初版の射程に入る role（`assistant` は描画時に `model` へ写像される）。 */
export type Gemma4ChatRole = "system" | "developer" | "user" | "assistant";

/**
 * 会話の 1 発話。
 *
 * MUST: 欄は 2 つだけ。`tools` / `reasoning` / `tool_calls` のような射程外の欄は
 * {@link renderGemma4Chat} が**許可リストで**落とす（型で防げるのは TS の呼び手だけで、
 * JSON から来た値は素通りする）。
 */
export type Gemma4ChatMessage = {
  readonly role: Gemma4ChatRole;
  readonly content: string;
};

/** chat の綴り（正本は `tokenizer_config.json` の `*_token` 欄）。 */
const START_OF_TURN = "<|turn>";
const END_OF_TURN = "<turn|>";
const BOS = "<bos>";
const START_OF_CHANNEL = "<|channel>";
const END_OF_CHANNEL = "<channel|>";
const TOOL_RESPONSE = "<|tool_response>";

/** 生成プロンプトが開く turn の role（`assistant` の写像先）。 */
const MODEL_ROLE = "model";

/** 先頭に置かれたときだけ `system` ブロックへ落ちる role。 */
const SYSTEM_ROLES: readonly string[] = ["system", "developer"];

const ROLES: readonly string[] = ["system", "developer", "user", "assistant"];

/** メッセージが持ってよい欄（許可リスト — Python 側の `ALLOWED_MESSAGE_KEYS` と対）。 */
const MESSAGE_KEYS: readonly string[] = ["role", "content"];

/**
 * Python の `str.isspace()` が真になるコードポイント。
 *
 * Jinja の `trim` フィルタは `str.strip()` そのもので、JS の `String.prototype.trim()` とは
 * 2 点だけ違う: Python は `\x1c`–`\x1f` と `\x85` を空白と見なし、U+FEFF は見なさない。
 * exotic な差だが、出るのは**例外ではなく別の id 列**なので集合を明示して写す。
 */
const isPythonSpace = (code: number): boolean =>
  (code >= 0x09 && code <= 0x0d) || // \t \n \v \f \r
  (code >= 0x1c && code <= 0x20) || // ファイル / グループ / レコード / 単位区切り + 空白
  code === 0x85 || code === 0xa0 || code === 0x1680 ||
  (code >= 0x2000 && code <= 0x200a) ||
  code === 0x2028 || code === 0x2029 || code === 0x202f || code === 0x205f || code === 0x3000;

/** Jinja の `trim`（= `str.strip()`）。 */
const trim = (text: string): string => {
  let start = 0;
  let end = text.length;
  while (start < end && isPythonSpace(text.charCodeAt(start))) start += 1;
  while (end > start && isPythonSpace(text.charCodeAt(end - 1))) end -= 1;
  return text.slice(start, end);
};

/**
 * 素の会話であることを確かめる（射程外は fail loudly）。
 *
 * MUST: 空の会話も落とす — 上流の `apply_chat_template` が `ValueError` で拒否するので、
 * 受理集合を上流より広げない。
 */
const assertPlainConversation = (messages: readonly Gemma4ChatMessage[]): void => {
  if (messages.length === 0) {
    throw new Error("gemma4ChatPrompt: 会話が空（上流の apply_chat_template も拒否する）");
  }
  messages.forEach((message, index) => {
    const where = `gemma4ChatPrompt: messages[${index}]`;
    if (typeof message !== "object" || message === null) {
      throw new Error(`${where} がオブジェクトでない`);
    }
    const extra = Object.keys(message).filter((key) => !MESSAGE_KEYS.includes(key));
    if (extra.length > 0) {
      throw new Error(
        `${where}: 射程外の欄 ${extra.join(" / ")}` +
          `（tools / reasoning / tool_calls などは初版では拒否する — ADR 0084 決定 5。` +
          `黙って無視すると「渡したのに効かない」が例外なしで通る）`,
      );
    }
    if (!ROLES.includes(message.role)) {
      throw new Error(
        `${where}.role '${String(message.role)}' が ${ROLES.join(" / ")} のどれでもない` +
          `（'model' は template の出力側の綴りで、入力の role ではない）`,
      );
    }
    if (typeof message.content !== "string") {
      throw new Error(
        `${where}.content が文字列でない` +
          `（画像 / 音声パートの配列は初版では拒否する — ADR 0084 決定 5）`,
      );
    }
    // 上流は model turn の本文にだけ `strip_thinking` を掛ける（thinking チャネルを本文から
    // 巻き戻す）。ここは掛けないので、綴りが本文に居ると**黙って別の文字列**になる。
    if (
      message.role === "assistant" &&
      (message.content.includes(START_OF_CHANNEL) || message.content.includes(END_OF_CHANNEL))
    ) {
      throw new Error(
        `${where}.content に thinking チャネルの綴り（${START_OF_CHANNEL} / ` +
          `${END_OF_CHANNEL}）がある（初版の射程外 — ADR 0084 決定 5）`,
      );
    }
  });
};

/**
 * 会話 → 上流 `apply_chat_template(add_generation_prompt=True)` と同じ文字列。
 *
 * 踏む分岐は 3 本だけ（実測 — ADR 0084 決定 5 / research §6）:
 *
 * 1. **先頭が `system` / `developer`** なら `<|turn>system\n…<turn|>\n` の system ブロックへ
 *    落ちる。`developer` でも綴りは `system` になる（role をそのまま出さない唯一の位置）—
 *    2 番目以降の `developer` は `<|turn>developer\n` である。
 * 2. `assistant` は `model` へ写像され、**直前の role も `assistant`** なら turn を開き直さ
 *    ない（連続 assistant は 1 つの model turn へ畳まれる。閉じる側も対称に飛ばす）。
 * 3. 末尾に生成プロンプト `<|turn>model\n` を必ず置く（この関数は生成の入口専用で、
 *    「学習用に閉じた会話を描く」形は持たない）。
 *
 * NOTE: 公開面には出さない（ADR 0008 の薄い面 — 消費者の入口は
 * {@link gemma4ChatPrompt}）。`export` はフィクスチャ突合で「描画」と「符号化」を別々に
 * 見るため — 割れたときにどちらの段かが読み手に伝わる。
 */
export const renderGemma4Chat = (messages: readonly Gemma4ChatMessage[]): string => {
  assertPlainConversation(messages);
  let out = BOS;
  let index = 0;
  if (SYSTEM_ROLES.includes(messages[0].role)) {
    out += `${START_OF_TURN}system\n${trim(messages[0].content)}${END_OF_TURN}\n`;
    index = 1;
  }
  // 継続の判定は**写像前**の role で行う（`model` へ写した後だと、system ブロックを挟んだ
  // 直後の assistant が「継続」に見える）。
  let previousRole: Gemma4ChatRole | undefined;
  for (; index < messages.length; index += 1) {
    const message = messages[index];
    const role = message.role === "assistant" ? MODEL_ROLE : message.role;
    if (!(role === MODEL_ROLE && previousRole === "assistant")) {
      out += `${START_OF_TURN}${role}\n`;
    }
    out += trim(message.content);
    const nextRole = messages[index + 1]?.role;
    if (!(role === MODEL_ROLE && nextRole === "assistant")) out += `${END_OF_TURN}\n`;
    previousRole = message.role;
  }
  return `${out}${START_OF_TURN}${MODEL_ROLE}\n`;
};

/**
 * 会話 → token id 列（`<bos>` 込み・末尾は生成プロンプト）。
 *
 * 上流と同じ 2 段（描画 → `add_special_tokens=False` の符号化）で組む。chat の綴りは追加
 * 語彙なので、`encode` の leftmost-longest の切り出しがそのまま 1 トークンへ落とす。
 *
 * 合格線（段 4）は HF `apply_chat_template` の出力との **id 列一致**で、フィクスチャは
 * `tools/export-recipes/gemma4/chat.py` が上流から独立に採る（ADR 0084 決定 7）。
 */
export const gemma4ChatPrompt = (
  tokenizer: GemmaTokenizer,
  messages: readonly Gemma4ChatMessage[],
): number[] => tokenizer.encode(renderGemma4Chat(messages));

/**
 * 停止 token の集合（ADR 0083 決定 8）— **トークナイザ資産から導出する**。
 *
 * gemma-4-E2B-it の `generation_config.json` は `eos_token_id = [1, 106, 50]`（`<eos>` /
 * `<turn|>` / `<|tool_response>`）を宣言する。数を焼き写さず綴りから引くのは、chat 形式と
 * EOS 集合が**同じ配布 digest set** から来る必要があるためである（ADR 0084 決定 5 — 別々の
 * 場所から拾うと片方だけ古くなる）。資産が入れ替われば id も一緒に動く。
 *
 * MUST: 綴りが欠けている資産は fail loudly。集合が痩せたまま通すと「`<turn|>` で止まらず
 * 次の turn を自分で書き始める」形になり、例外は 1 つも出ない。
 */
export const gemma4StopTokens = (tokenizer: GemmaTokenizer): number[] => {
  const stops = [tokenizer.eosId];
  for (const spelling of [END_OF_TURN, TOOL_RESPONSE]) {
    const id = tokenizer.addedTokenId(spelling);
    if (id === undefined) {
      throw new Error(
        `gemma4StopTokens: 停止 token '${spelling}' がトークナイザ資産の追加語彙に無い` +
          `（chat 形式と EOS 集合は同じ digest set から来る — ADR 0084 決定 5）`,
      );
    }
    stops.push(id);
  }
  return stops;
};
