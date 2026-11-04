/**
 * The gemma4 output parser — one generated model turn → `{ role, content?, thinking?,
 * tool_calls? }`, equal to upstream `tokenizer.parse_response(text, prefix=prompt)` (ADR 0084
 * addendum 2026-11-04).
 *
 * Upstream's parser is generic (`transformers/utils/chat_parsing/response_parser.py`, a streaming
 * region machine driven by the snapshot's `response_template`); this module is **the same machine
 * specialised to that one template** (identical in the E2B, E4B and 12B snapshots — checked by the
 * recipe's pytest):
 *
 * - `content` — the implicit region, closed by `<turn|>` / `<|tool_response>` / `<eos>`;
 *   Python-stripped;
 * - `thinking` — opened by `<|channel>thought\n`, closed by `<channel|>`; stripped;
 * - `tool_calls` — opened by `<|tool_call>call:NAME` (`NAME` = `\w+`, Unicode), closed by
 *   `<tool_call|>`; the body is JSON with `<|"|>` string delimiters and unquoted keys; repeats.
 *
 * The prefix (the prompt) is cut past its last `<|turn>model\n` or `<tool_response|>` and read
 * first, so a prefilled `<|channel>thought\n` (after a tool result with thinking on) or 12B's empty
 * thought channel is seen as upstream sees it. A non-empty prompt with neither anchor fails loudly
 * (every prompt of `chatPrompt` / `chatTurn` / `chatToolResults` ends past one; upstream would
 * read the whole prompt as part of the answer).
 *
 * ## MUST: the input is one closed model turn — failures are loud
 *
 * Where upstream raises (a body that is not JSON after its dialect transforms, the reserved
 * `\x01` / `\x02`), this throws {@link ModelInputError} too. It also throws where upstream returns
 * data that would hide a failure (each case is in the `responses` list of the fixtures, labelled
 * `ts-rejects`, with upstream's result beside it):
 *
 * - **no closing token** — the turn was cut (max tokens, a stop string, an interruption); upstream
 *   returns the cut text as if it were the answer, or an unterminated call whose JSON happens to
 *   be complete as if it were a call to run;
 * - **a region still open at the end** (a thought or a call without its closer — the stop token
 *   then lands inside it) — upstream returns the partial region;
 * - **text after the closing token** — upstream keeps only the last segment;
 * - **content or a thought in two non-empty segments** — upstream's fields do not repeat, so it
 *   keeps only the last one (an empty or whitespace-only later segment is ignored here, as an empty
 *   first one is: upstream would let it erase the earlier text);
 * - **arguments that are not an object** — upstream returns them, but the template cannot render
 *   them back (it raises on a non-mapping);
 * - **numbers a double cannot hold** — `NaN` (not JSON), a literal beyond the double range
 *   (`1e309`, a 400-digit integer: `Infinity` in JS, upstream's `inf` or a huge `int`), and an
 *   integer literal beyond 2^53 (upstream keeps it exact, JS would round it — a wrong id would
 *   reach the caller);
 * - **an argument key that is not a word** (written quoted in the output) — the template writes
 *   keys bare and `\w+` could not read it back;
 * - **a structure spelling left in the content** (`<|tool_call>` without `call:NAME`, a stray
 *   `<tool_call|>` / `<|channel>` / `<channel|>` / `<tool_response|>` / `<|tool>` / `<tool|>` /
 *   `<|"|>`, or a `<|turn>` that starts a turn of its own) — upstream returns it as text, so a
 *   malformed call would pass as an answer and a re-render would write a fake turn.
 *
 * NOTE: inside a thought the spellings stay text, as upstream returns them: a call written in the
 * thought channel is part of the thought, not a call (the model has not made it).
 *
 * NOTE: the integer check reads each number's source text through `JSON.parse`'s reviver context
 * (`context.source`, the "JSON.parse source text access" feature — measured on Node 24 and Deno 2.9
 * (V8 15.0); shipped in V8 11.4 / Chrome 114 and Firefox 135; Safari not checked here). Where a
 * runtime does not pass it, every integer-valued number beyond 2^53 is refused, float literals
 * (`1e20`) included — loud, never rounded.
 *
 * MUST: every module is side-effect free (no table or instance is built at module scope).
 */

import { ModelInputError } from "../../errors.ts";
import { pythonStrip } from "../../text/python-strip.ts";
import {
  assertReadableArguments,
  type ChatJsonObject,
  type ChatToolCall,
  wordCharacterClass,
} from "./chat.ts";
import type { Tokenizer } from "./tokenizer.ts";

/**
 * One parsed model turn (upstream's keys — `thinking`, not the `reasoning` the template reads; an
 * empty field is absent, as upstream drops it).
 */
export type ChatResponse = {
  readonly role: "assistant";
  readonly content?: string;
  readonly thinking?: string;
  readonly tool_calls?: readonly ParsedToolCall[];
};

/** A parsed tool call (`type` is always `"function"`, `arguments` always an object). */
export type ParsedToolCall = ChatToolCall & {
  readonly type: "function";
  readonly function: { readonly name: string; readonly arguments: ChatJsonObject };
};

/** Where the assistant message starts in a prompt (`start_anchor` of the response template). */
const START_ANCHORS: readonly string[] = ["<|turn>model\n", "<tool_response|>"];
/** The closers of the implicit content region (`fields.content.close`). */
const CONTENT_CLOSERS: readonly string[] = ["<turn|>", "<|tool_response>", "<eos>"];
const THOUGHT_OPEN = "<|channel>thought\n";
const THOUGHT_CLOSE = "<channel|>";
const CALL_CLOSE = "<tool_call|>";
/** Spellings that mean structure — left in the content, they mark a malformed turn. */
const STRUCTURE_SPELLINGS: readonly string[] = [
  "<|tool_call>",
  "<tool_call|>",
  "<|channel>",
  "<channel|>",
  "<|turn>",
  "<tool_response|>",
  "<|tool>",
  "<tool|>",
  '<|"|>',
];
const QUOTE = '<|"|>';

/**
 * The prompt cut past its last start anchor. An empty prompt is upstream's explicit opt-out; a
 * non-empty one without an anchor is a caller error (upstream would read it all as the answer).
 */
const pastLastAnchor = (prefix: string, where: string): string => {
  let end = -1;
  for (const anchor of START_ANCHORS) {
    const at = prefix.lastIndexOf(anchor);
    if (at >= 0) end = Math.max(end, at + anchor.length);
  }
  if (end < 0 && prefix !== "") {
    throw new ModelInputError(
      `${where}: the prompt has no ${START_ANCHORS.map((anchor) => JSON.stringify(anchor)).join(" / ")} ` +
        `(pass the prompt the output followed — chatPrompt / chatTurn / chatToolResults — or none)`,
    );
  }
  return end < 0 ? prefix : prefix.slice(end);
};

/**
 * The part of a prompt's tail the output continues inside: `<|channel>thought\n` when the prompt
 * opened a thought channel it did not close (after a tool result with thinking on), `""` otherwise.
 * A record of a turn that could not be parsed starts with it, so a re-render's `strip_thinking`
 * still finds the thought.
 *
 * NOTE: internal (not on the public surface); exported for `../chat-session.ts`.
 */
export const openPrefillOf = (prompt: string): string =>
  pastLastAnchor(prompt, "gemma.ChatSession").endsWith(THOUGHT_OPEN) ? THOUGHT_OPEN : "";

/**
 * Refuses a number a double cannot hold exactly where it matters ({@link parseArguments}): a
 * non-finite value, and an integer literal beyond 2^53 (`source` = its JSON text, when the runtime
 * passes it — the module NOTE).
 */
const assertHeldNumber = (
  value: number,
  source: string | undefined,
  name: string,
  where: string,
): void => {
  if (!Number.isFinite(value)) {
    throw new ModelInputError(
      `${where}: the call ${name} has the number ${source ?? String(value)}, which a double ` +
        `cannot hold (it parses to ${value})`,
    );
  }
  const integer = source === undefined ? Number.isInteger(value) : /^-?\d+$/.test(source);
  if (integer && !Number.isSafeInteger(value)) {
    throw new ModelInputError(
      `${where}: the call ${name} has the integer ${source ?? String(value)}, beyond 2^53 — JS ` +
        `would round it (upstream keeps it exact)${source === undefined ? "; this runtime does not pass the number's source text, so any integer-valued number beyond 2^53 is refused" : ""}`,
    );
  }
};

/** One delimiter found in the buffer. */
type Delimiter = {
  readonly kind: "thought" | "call" | "close";
  readonly start: number;
  readonly end: number;
  /** The function name (`call` only). */
  readonly name?: string;
};

/** The earliest delimiter the content region watches (the openers and its closers). */
const nextInContent = (buffer: string, from: number): Delimiter | undefined => {
  const found: Delimiter[] = [];
  const thought = buffer.indexOf(THOUGHT_OPEN, from);
  if (thought >= 0)
    found.push({ kind: "thought", start: thought, end: thought + THOUGHT_OPEN.length });
  const call = new RegExp(`<\\|tool_call>call:(${wordCharacterClass()}+)`, "gu");
  call.lastIndex = from;
  const opened = call.exec(buffer);
  if (opened !== null) {
    found.push({
      kind: "call",
      start: opened.index,
      end: opened.index + opened[0].length,
      name: opened[1],
    });
  }
  for (const closer of CONTENT_CLOSERS) {
    const at = buffer.indexOf(closer, from);
    if (at >= 0) found.push({ kind: "close", start: at, end: at + closer.length });
  }
  // No two of these spellings can start at the same position (they differ within their first
  // three characters), so the earliest is unique.
  return found.reduce<Delimiter | undefined>(
    (best, next) => (best === undefined || next.start < best.start ? next : best),
    undefined,
  );
};

/**
 * A call body → its arguments, as upstream's `json` content parser with `string_delims` and
 * `unquoted_keys`: each `<|"|>…<|"|>` string is set aside, a word key right after `{` or `,` is
 * quoted, the strings come back as JSON strings, and the result is parsed as JSON.
 */
const parseArguments = (name: string, body: string, where: string): ChatJsonObject => {
  if (body.includes("\x01") || body.includes("\x02")) {
    throw new ModelInputError(
      `${where}: the call ${name} contains the reserved characters \\x01 / \\x02 (upstream refuses them too)`,
    );
  }
  const strings: string[] = [];
  const delimited = new RegExp(
    `${QUOTE.replaceAll("|", "\\|")}([\\s\\S]*?)${QUOTE.replaceAll("|", "\\|")}`,
    "g",
  );
  let working = body.replace(delimited, (_match, text: string) => {
    strings.push(text);
    return `\x01${strings.length - 1}\x02`;
  });
  working = working.replace(new RegExp(`(?<=[{,])(${wordCharacterClass()}+):`, "gu"), '"$1":');
  // Back from the placeholders (the reserved characters occur nowhere else — checked above).
  const [head, ...rest] = working.split("\x01");
  working =
    head +
    rest
      .map((piece) => {
        const close = piece.indexOf("\x02");
        return JSON.stringify(strings[Number(piece.slice(0, close))]) + piece.slice(close + 1);
      })
      .join("");
  let value: unknown;
  const numbers: { readonly value: number; readonly source: string | undefined }[] = [];
  try {
    value = JSON.parse(working, (_key, raw: unknown, context?: { readonly source?: string }) => {
      if (typeof raw === "number") numbers.push({ value: raw, source: context?.source });
      return raw;
    });
  } catch (error) {
    throw new ModelInputError(
      `${where}: the arguments of the call ${name} are not JSON after the dialect transforms ` +
        `(${JSON.stringify(body)} → ${JSON.stringify(working)}: ${(error as Error).message})`,
    );
  }
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new ModelInputError(
      `${where}: the arguments of the call ${name} are not an object (${JSON.stringify(value)} — ` +
        `the template cannot render them back)`,
    );
  }
  for (const number of numbers) assertHeldNumber(number.value, number.source, name, where);
  assertReadableArguments(value, `${where}: the arguments of the call ${name}`);
  return value as ChatJsonObject;
};

/**
 * Generated text (special spellings included, **ending with its stop token**) + the prompt it
 * followed → the parsed turn ({@link ChatResponse}).
 *
 * NOTE: not exposed on the public surface — the consumer's entry is {@link parseChatResponse},
 * which decodes token ids with their special spellings (the text a `ChatStream` yields drops them).
 * `export` is so that the fixture comparison can pass upstream's strings as they are.
 */
export const parseGemma4ChatResponse = (text: string, prefix: string): ChatResponse => {
  const where = "gemma.parseChatResponse";
  const prefill = pastLastAnchor(prefix, where);
  const buffer = `${prefill}${text}`;
  let region: "content" | "thought" | "call" = "content";
  /** Where the open explicit region started (to say whether the prompt's prefill opened it). */
  let regionStart = 0;
  let opened = false;
  let body = "";
  let callName = "";
  let content: string | undefined;
  let thinking: string | undefined;
  const calls: ParsedToolCall[] = [];

  const closeContent = (): void => {
    if (!opened) return;
    const value = pythonStrip(body);
    // An empty (whitespace-only) segment is no text, wherever it sits — the earlier text stays.
    if (value === "") {
      content ??= "";
      return;
    }
    if (content !== undefined && content !== "") {
      throw new ModelInputError(
        `${where}: the content comes in more than one segment (${JSON.stringify(content)}, then ` +
          `${JSON.stringify(value)}) — upstream would keep only the last`,
      );
    }
    const spelled = STRUCTURE_SPELLINGS.find((spelling) => value.includes(spelling));
    if (spelled !== undefined) {
      throw new ModelInputError(
        `${where}: the content contains ${spelled} outside its region (a malformed tool call or ` +
          `thought channel — upstream would return it as text): ${JSON.stringify(value)}`,
      );
    }
    content = value;
  };

  let at = 0;
  for (;;) {
    const found: Delimiter | undefined =
      region === "content"
        ? nextInContent(buffer, at)
        : ((): Delimiter | undefined => {
            const closer = region === "thought" ? THOUGHT_CLOSE : CALL_CLOSE;
            const index = buffer.indexOf(closer, at);
            return index < 0
              ? undefined
              : { kind: "close", start: index, end: index + closer.length };
          })();
    if (found === undefined) {
      if (region !== "content") {
        const origin =
          regionStart < prefill.length ? "opened by the prompt's prefill" : "opened in the output";
        throw new ModelInputError(
          `${where}: the ${region === "thought" ? "thought channel" : `tool call ${callName}`} ` +
            `(${origin}) is not closed (the turn was cut — upstream would return the partial ` +
            `region): ${JSON.stringify(text)}`,
        );
      }
      throw new ModelInputError(
        `${where}: the model turn is not closed by ${CONTENT_CLOSERS.join(" / ")} (a turn cut by ` +
          `max tokens, a stop string or an interruption — upstream would return the cut text): ` +
          `${JSON.stringify(text)}`,
      );
    }
    const between = buffer.slice(at, found.start);
    if (between !== "") {
      opened = true;
      body += between;
    }
    at = found.end;
    if (region === "content" && found.kind !== "close") {
      closeContent();
      region = found.kind;
      regionStart = found.start;
      opened = true;
      body = "";
      callName = found.name ?? "";
      continue;
    }
    if (region === "content") {
      closeContent();
      if (at !== buffer.length) {
        throw new ModelInputError(
          `${where}: text follows the closing token (${JSON.stringify(buffer.slice(at))} — one ` +
            `model turn ends at its first stop token; upstream would keep only the last segment)`,
        );
      }
      break;
    }
    if (region === "thought") {
      const value = pythonStrip(body);
      if (value === "") thinking ??= "";
      else if (thinking !== undefined && thinking !== "") {
        throw new ModelInputError(
          `${where}: two thought channels (${JSON.stringify(thinking)}, then ` +
            `${JSON.stringify(value)}) — upstream would keep only the last`,
        );
      } else thinking = value;
    } else {
      calls.push({
        type: "function",
        function: { name: callName, arguments: parseArguments(callName, body, where) },
      });
    }
    region = "content";
    opened = false;
    body = "";
  }
  return {
    role: "assistant",
    ...(content === undefined || content === "" ? {} : { content }),
    ...(thinking === undefined || thinking === "" ? {} : { thinking }),
    ...(calls.length === 0 ? {} : { tool_calls: calls }),
  };
};

/**
 * Generated token ids → **the parsed model turn** (upstream `parse_response(ids, prefix=ids)`).
 *
 * - `output`: the turn's generated ids **including its stop token** (the ids `onToken` saw, then
 *   `done.token` of an `eos` stop — `<turn|>`, `<|tool_response>` or `<eos>`);
 * - `prompt`: the prompt those ids followed (the whole `chatPrompt`, or the delta of `chatTurn` /
 *   `chatToolResults` — only the part after its last `<|turn>model\n` / `<tool_response|>` is read).
 *
 * Both are decoded **with** their special spellings (`<|channel>`, `<|tool_call>`, `<|"|>` are
 * special tokens, which the text of a `ChatStream` drops). Failures are loud (the doc of this module
 * lists them): a turn cut before its stop token cannot be parsed.
 */
export const parseChatResponse = (
  tokenizer: Tokenizer,
  output: Iterable<number>,
  prompt: Iterable<number>,
): ChatResponse =>
  parseGemma4ChatResponse(
    tokenizer.decode(output, { skipSpecialTokens: false }),
    tokenizer.decode(prompt, { skipSpecialTokens: false }),
  );
