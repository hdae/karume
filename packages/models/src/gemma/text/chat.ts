/**
 * The gemma4 chat format — **one family-specific pure function** (ADR 0084 decision 5).
 *
 * No generic template renderer is built. The upstream `chat_template.jinja` is 386 lines /
 * 18,569 B, but **plain conversation needs only five spellings and three branches**, and bringing
 * in a Jinja-equivalent evaluator would violate both "runtime dependencies are Web standards only"
 * (CLAUDE.md cross-cutting invariants) and "no speculative generalization". Precedent: ADR 0079
 * (text analysis is the caller's job; one conversion function).
 *
 * ## The spelling is not Gemma 3's
 *
 * `<|turn>` / `<turn|>` / `<|channel>` / `<|think|>` are the source of truth (`sot_token` /
 * `eot_token` / `soc_token` / `think_token` in `tokenizer_config.json`), and **`<start_of_turn>` is
 * not in this vocabulary** (ADR 0084 Context 3). That is why consumers are not asked to write it by
 * hand.
 *
 * ## MUST: this module owns `<bos>`
 *
 * `GemmaTokenizer.encode` does not add `<bos>` (matching the measured gemma4 post_processor). Only
 * {@link renderGemma4Chat} adds it; without the split, introducing chat would produce a double BOS
 * (the MUST of ADR 0084 decision 5). Upstream has the same shape — the template renders
 * `bos_token`, and `apply_chat_template` encodes with `add_special_tokens=False`.
 *
 * ## MUST: out of scope fails loudly
 *
 * The first scope is "plain conversation" only (ADR 0084 decision 5; draft §6.3). tools / thinking /
 * tool_call / image and audio parts / unknown roles are **not silently ignored** — ignoring them
 * would let "I passed tools and they were not used" through without an exception. The scope widens
 * once a real need appears, and that starts by adding cases to the Python-side test battery (the
 * docstring of `tools/export-recipes/gemma4/chat.py`).
 *
 * MUST: every module is side-effect free (no table or instance is built at module scope).
 */

import { ModelInputError } from "../../errors.ts";
import type { GemmaTokenizer } from "./tokenizer.ts";

/** The roles in the first scope (`assistant` is mapped to `model` when rendered). */
export type Gemma4ChatRole = "system" | "developer" | "user" | "assistant";

/**
 * One message of a conversation.
 *
 * MUST: only two fields. Out-of-scope fields such as `tools` / `reasoning` / `tool_calls` are
 * rejected by {@link renderGemma4Chat} **with an allow list** (the type protects only TS callers; a
 * value that came from JSON passes straight through).
 */
export type Gemma4ChatMessage = {
  readonly role: Gemma4ChatRole;
  readonly content: string;
};

/** The chat spellings (source of truth: the `*_token` fields of `tokenizer_config.json`). */
const START_OF_TURN = "<|turn>";
const END_OF_TURN = "<turn|>";
const BOS = "<bos>";
const START_OF_CHANNEL = "<|channel>";
const END_OF_CHANNEL = "<channel|>";
const TOOL_RESPONSE = "<|tool_response>";

/** The role of the turn the generation prompt opens (what `assistant` maps to). */
const MODEL_ROLE = "model";

/** Roles that fall into the `system` block only when placed first. */
const SYSTEM_ROLES: readonly string[] = ["system", "developer"];

const ROLES: readonly string[] = ["system", "developer", "user", "assistant"];

/** The fields a message may carry (allow list — pairs with `ALLOWED_MESSAGE_KEYS` in Python). */
const MESSAGE_KEYS: readonly string[] = ["role", "content"];

/**
 * The code points for which Python's `str.isspace()` is true.
 *
 * Jinja's `trim` filter is `str.strip()` itself, which differs from JS's `String.prototype.trim()` in
 * exactly two points: Python treats `\x1c`–`\x1f` and `\x85` as whitespace, and does not treat
 * U+FEFF as whitespace. The difference is exotic, but what comes out is **a different id sequence,
 * not an exception**, so the set is ported explicitly.
 */
const isPythonSpace = (code: number): boolean =>
  (code >= 0x09 && code <= 0x0d) || // \t \n \v \f \r
  (code >= 0x1c && code <= 0x20) || // file / group / record / unit separators + space
  code === 0x85 ||
  code === 0xa0 ||
  code === 0x1680 ||
  (code >= 0x2000 && code <= 0x200a) ||
  code === 0x2028 ||
  code === 0x2029 ||
  code === 0x202f ||
  code === 0x205f ||
  code === 0x3000;

/** Jinja's `trim` (= `str.strip()`). */
const trim = (text: string): string => {
  let start = 0;
  let end = text.length;
  while (start < end && isPythonSpace(text.charCodeAt(start))) start += 1;
  while (end > start && isPythonSpace(text.charCodeAt(end - 1))) end -= 1;
  return text.slice(start, end);
};

/**
 * Checks that one message is within the scope of plain conversation (out of scope fails loudly).
 *
 * `where` is the location prefixed to the diagnostic (it differs per caller — `… messages[i]` for
 * a conversation).
 */
const assertPlainMessage = (message: Gemma4ChatMessage, where: string): void => {
  if (typeof message !== "object" || message === null) {
    throw new ModelInputError(`${where} がオブジェクトでない`);
  }
  const extra = Object.keys(message).filter((key) => !MESSAGE_KEYS.includes(key));
  if (extra.length > 0) {
    throw new ModelInputError(
      `${where}: 射程外の欄 ${extra.join(" / ")}` +
        `（tools / reasoning / tool_calls などは初版では拒否する — ADR 0084 決定 5。` +
        `黙って無視すると「渡したのに効かない」が例外なしで通る）`,
    );
  }
  if (!ROLES.includes(message.role)) {
    throw new ModelInputError(
      `${where}.role '${String(message.role)}' が ${ROLES.join(" / ")} のどれでもない` +
        `（'model' は template の出力側の綴りで、入力の role ではない）`,
    );
  }
  if (typeof message.content !== "string") {
    throw new ModelInputError(
      `${where}.content が文字列でない` +
        `（画像 / 音声パートの配列は初版では拒否する — ADR 0084 決定 5）`,
    );
  }
  // Upstream applies `strip_thinking` only to the body of a model turn (it rewinds the thinking
  // channel out of the body). It is not applied here, so a spelling in the body would **silently
  // become a different string**.
  if (
    message.role === "assistant" &&
    (message.content.includes(START_OF_CHANNEL) || message.content.includes(END_OF_CHANNEL))
  ) {
    throw new ModelInputError(
      `${where}.content に thinking チャネルの綴り（${START_OF_CHANNEL} / ` +
        `${END_OF_CHANNEL}）がある（初版の射程外 — ADR 0084 決定 5）`,
    );
  }
};

/**
 * Checks that this is a plain conversation (out of scope fails loudly).
 *
 * MUST: also reject an empty conversation — upstream `apply_chat_template` rejects it with a
 * `ValueError`, so the accepted set is not made wider than upstream's.
 */
const assertPlainConversation = (messages: readonly Gemma4ChatMessage[]): void => {
  if (messages.length === 0) {
    throw new ModelInputError(
      "gemma4ChatPrompt: 会話が空（上流の apply_chat_template も拒否する）",
    );
  }
  messages.forEach((message, index) => {
    assertPlainMessage(message, `gemma4ChatPrompt: messages[${index}]`);
  });
};

/**
 * Conversation → the same string as upstream `apply_chat_template(add_generation_prompt=True)`.
 *
 * Only three branches are taken (measured — ADR 0084 decision 5 / the 2026-08-31 generation API
 * draft §6):
 *
 * 1. If **the first message is `system` / `developer`**, it falls into the system block
 *    `<|turn>system\n…<turn|>\n`. Even `developer` is spelled `system` (the only position where the
 *    role is not emitted as is) — a second or later `developer` is `<|turn>developer\n`.
 * 2. `assistant` is mapped to `model`, and if **the previous role is also `assistant`** the turn is
 *    not reopened (consecutive assistants fold into one model turn; the closing side is skipped
 *    symmetrically).
 * 3. The generation prompt `<|turn>model\n` is always placed at the end (this function is only the
 *    entry to generation; it has no form for "rendering a closed conversation for training").
 *
 * NOTE: not exposed on the public surface (the thin surface of ADR 0008 — the consumer's entry is
 * {@link gemma4ChatPrompt}). `export` is so that the fixture comparison looks at "rendering" and
 * "encoding" separately — when they diverge, the reader learns which step it was.
 */
export const renderGemma4Chat = (messages: readonly Gemma4ChatMessage[]): string => {
  assertPlainConversation(messages);
  let out = BOS;
  let index = 0;
  if (SYSTEM_ROLES.includes(messages[0].role)) {
    out += `${START_OF_TURN}system\n${trim(messages[0].content)}${END_OF_TURN}\n`;
    index = 1;
  }
  // Continuation is decided on the role **before mapping** (after mapping to `model`, an assistant
  // right after the system block would look like a "continuation").
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
 * Conversation → token ids (`<bos>` included, ending with the generation prompt).
 *
 * Built in the same two steps as upstream (render → encode with `add_special_tokens=False`). The
 * chat spellings are added vocabulary, so the leftmost-longest split of `encode` turns each into one
 * token as is.
 *
 * The pass line (stage 4) is an **id-sequence match** with the output of HF `apply_chat_template`;
 * the fixtures are taken independently from upstream by `tools/export-recipes/gemma4/chat.py`
 * (ADR 0084 decision 7).
 */
export const gemma4ChatPrompt = (
  tokenizer: GemmaTokenizer,
  messages: readonly Gemma4ChatMessage[],
): number[] => tokenizer.encode(renderGemma4Chat(messages));

/**
 * One new message → **the delta rendered as the continuation of the conversation** (containing
 * neither `<bos>` nor past turns).
 *
 * ## MUST: do not include the `<turn|>` that closes the previous turn (the turn-local contract)
 *
 * The `<turn|>` that closes the previous model turn is a spelling generation emitted and that
 * `GenerationSequence` holds as the uncommitted frontier (`pendingToken`); **the next `generate`
 * prepends it to the prompt automatically** (ADR 0083 decision 4). This renders from **after** it,
 * starting with the newline that follows `<turn|>`. Rendering it twice would make two turn
 * separators, and not a single exception would occur.
 *
 * So the identity that holds is (`⧺` is token-sequence concatenation):
 *
 * ```text
 * gemma4ChatPrompt(whole conversation)
 *   = gemma4ChatPrompt(up to the first turn) ⧺ generated body ⧺ [<turn|>]
 *     ⧺ gemma4ChatTurn(next message) ⧺ …
 * ```
 *
 * The check in `gemma4_chat_test.ts` verifies this identity over arbitrary splits (it breaks there
 * if the template's turn-locality breaks — the source of truth that keeps consumers from writing a
 * guess such as "strip the leading `<bos>`").
 *
 * MUST: usable only **right after generation closed with `<turn|>`**. A turn cut off by max-tokens or
 * `break` has an unclosed model turn, so the right continuation is further generation
 * (`prompt: []`), not a delta.
 *
 * MUST: reject `assistant`. The model turn is the slot generation fills, and the upstream template
 * also folds consecutive assistants into one turn (branch 2 of `renderGemma4Chat`), so "appending an
 * assistant after a closed model turn" cannot be rendered as a delta. To insert one into a
 * conversation, re-render the whole of it with {@link gemma4ChatPrompt}.
 */
export const renderGemma4ChatTurn = (message: Gemma4ChatMessage): string => {
  assertPlainMessage(message, "gemma4ChatTurn: message");
  if (message.role === "assistant") {
    throw new ModelInputError(
      `gemma4ChatTurn: role 'assistant' は差分にできない` +
        `（model turn は生成が埋める席で、template も連続 assistant を 1 つへ畳む — ` +
        `会話へ差し込むなら gemma4ChatPrompt で全体を描き直す）`,
    );
  }
  // The leading newline is the one character "right after the `<turn|>` that closed the previous
  // turn" (the `${END_OF_TURN}\n` of `renderGemma4Chat`). `<turn|>` itself is prepended by
  // pendingToken, so it is not here.
  return (
    `\n${START_OF_TURN}${message.role}\n${trim(message.content)}${END_OF_TURN}\n` +
    `${START_OF_TURN}${MODEL_ROLE}\n`
  );
};

/**
 * One new message → **a delta token sequence that can be passed to the next `generate` as is** (the
 * doc of {@link renderGemma4ChatTurn} is the source of truth for the turn-local contract — the
 * `<turn|>` that closes the previous turn is not included).
 *
 * This is the entry for running a multi-turn conversation with `Gemma4Pipeline.sequence()`. Past
 * turns are in the context's KV, so there is no need to re-render the whole conversation every turn
 * (= an O(n²) prefill).
 */
export const gemma4ChatTurn = (tokenizer: GemmaTokenizer, message: Gemma4ChatMessage): number[] =>
  tokenizer.encode(renderGemma4ChatTurn(message));

/**
 * The set of stop tokens (ADR 0083 decision 8) — **derived from the tokenizer asset**.
 *
 * The `generation_config.json` of gemma-4-E2B-it declares `eos_token_id = [1, 106, 50]` (`<eos>` /
 * `<turn|>` / `<|tool_response>`). The ids are looked up from their spellings rather than hard-coded
 * because the chat format and the EOS set must come from **the same distribution digest set**
 * (ADR 0084 decision 5 — picked up from separate places, one of them goes stale). When the asset is
 * replaced, the ids move with it.
 *
 * MUST: an asset missing a spelling fails loudly. Passing a thinned set would end up "not stopping
 * at `<turn|>` and starting to write the next turn itself", and not a single exception would occur.
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
