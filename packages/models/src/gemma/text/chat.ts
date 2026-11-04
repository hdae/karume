/**
 * The gemma4 chat format — **family-specific pure functions** (ADR 0084 decision 5 and its addendum
 * of 2026-11-04).
 *
 * No generic template renderer is built. The upstream `chat_template.jinja` is 386 lines /
 * 18,569 B; bringing in a Jinja-equivalent evaluator would violate both "runtime dependencies are
 * Web standards only" (CLAUDE.md cross-cutting invariants) and "no speculative generalization".
 * Instead the functions below **transliterate the template's control flow** — the system block, the
 * turn loop with its continuation rules, the tool declaration formatters (`format_parameters` /
 * `format_function_declaration` / `format_argument`), tool calls, tool results, the thinking
 * channel and `strip_thinking`, and the generation prompt. The source of truth is the pinned
 * template (E2B / E4B: byte-identical files; 12B: the same text plus one generation-prompt
 * branch); every behaviour is pinned by fixtures harvested from upstream `apply_chat_template`
 * (`tools/export-recipes/gemma4/chat.py`, ADR 0084 decision 7).
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
 * `Tokenizer.encode` does not add `<bos>` (matching the measured gemma4 post_processor). Only
 * {@link renderGemma4Chat} adds it; without the split, introducing chat would produce a double BOS
 * (the MUST of ADR 0084 decision 5). Upstream has the same shape — the template renders
 * `bos_token`, and `apply_chat_template` encodes with `add_special_tokens=False`.
 *
 * ## MUST: what the template cannot take, or takes wrongly, fails loudly
 *
 * Messages, tools and options are checked **with allow lists** (the types protect only TS callers;
 * a value that came from JSON passes straight through). Out of scope: the legacy `tool_responses`
 * field and content-part arrays (text / image / audio / video). Rejected although upstream renders
 * them, because upstream's output would be wrong without an exception: results whose function name
 * the template resolves by accident (several calls without ids, a contradicting `name`, an unmatched
 * id), an orphan tool result (dropped), a tool call without results (a bare `<|tool_response>`), a
 * tool name the output parser cannot read back, and the `thinking` field (not read by the template).
 * The evidence for each is the `rejectedChat` list of the fixtures.
 *
 * MUST: every module is side-effect free (no table or instance is built at module scope).
 */

import { ModelInputError } from "../../errors.ts";
import { pythonStrip } from "../../text/python-strip.ts";
import { type ChatDialect, DEFAULT_CHAT_DIALECT } from "./chat-dialect.ts";
import type { Tokenizer } from "./tokenizer.ts";

/** A JSON value as the template's `format_argument` reads it (objects are plain objects). */
export type ChatJsonValue =
  | string
  | number
  | boolean
  | null
  | readonly ChatJsonValue[]
  | { readonly [key: string]: ChatJsonValue };

/** A JSON object (tool-call arguments, a parameter schema). */
export type ChatJsonObject = { readonly [key: string]: ChatJsonValue };

/**
 * One tool call of an assistant message (the OpenAI shape the template reads).
 *
 * MUST: `arguments` is an **object** (or `null` = no arguments). OpenAI's JSON-string form is
 * refused by the template itself — parse it before passing the message.
 */
export type ChatToolCall = {
  /** Matched by a result's `tool_call_id` (the template resolves the function name with it). */
  readonly id?: string;
  readonly type?: "function";
  readonly function: {
    /** A word (`\w+`, Unicode) — the output parser reads a call's name with that pattern. */
    readonly name: string;
    readonly arguments: ChatJsonObject | null;
  };
};

/**
 * One function declaration (`tools` — the OpenAI shape the template reads).
 *
 * The template renders only part of a JSON Schema: `description`, `type`, `enum` (strings),
 * `items` (arrays), `nullable`, `properties` and `required` (objects). Other keywords are dropped by
 * the template itself, not by this module (nested `items` keep every key, through
 * `format_argument`).
 */
export type ChatTool = {
  readonly type: "function";
  readonly function: {
    /** A word (`\w+`, Unicode) — the output parser reads a call's name with that pattern. */
    readonly name: string;
    readonly description?: string;
    readonly parameters?: ChatJsonObject | null;
    /** A response declaration (the template renders its `description`, and `type` when `object`). */
    readonly response?: ChatJsonObject | null;
  };
};

/** The roles (`assistant` is rendered as `model`; `tool` results continue the model turn). */
export type ChatRole = "system" | "developer" | "user" | "assistant" | "tool";

/** A text message (a leading `system` / `developer` goes into the system block). */
export type ChatTextMessage = {
  readonly role: "system" | "developer" | "user";
  readonly content: string;
};

/**
 * An assistant message (a model turn, or part of one).
 *
 * The thought is `reasoning`, or `reasoning_content` when `reasoning` is empty (the template's
 * `reasoning or reasoning_content`); it is rendered only after the last user message (or, with
 * `preserveThinking`, on a message with tool calls). `content` goes through `strip_thinking` (a
 * spelled thought channel is removed) and is written **after** the tool results of this message.
 */
export type ChatAssistantMessage = {
  readonly role: "assistant";
  /**
   * The text (`""` for a message that only calls tools). Untyped input may also pass `null` or
   * omit it (OpenAI's shape next to `tool_calls`) — rendered as empty, as upstream does.
   */
  readonly content: string;
  readonly reasoning?: string;
  readonly reasoning_content?: string;
  readonly tool_calls?: readonly ChatToolCall[];
};

/**
 * One tool result (OpenAI's role `tool`). It follows the assistant message whose call it answers
 * (several results follow it consecutively).
 *
 * The function name comes from the call whose `id` equals `tool_call_id`; without `tool_call_id` it
 * is `name`, or the name of the only call without an id. An object `content` is not accepted (the
 * template reads an object as a content-part array and raises) — serialize it to a string.
 */
export type ChatToolMessage = {
  readonly role: "tool";
  /**
   * The result as text (`response:NAME{value:<|"|>…<|"|>}`). Untyped input may also pass a number,
   * a boolean or `null`, which the template writes as that JSON value (`{value:42}`).
   */
  readonly content: string;
  readonly tool_call_id?: string;
  readonly name?: string;
};

/** One message of a conversation (the HF / OpenAI shape — field names as upstream spells them). */
export type ChatMessage = ChatTextMessage | ChatAssistantMessage | ChatToolMessage;

/**
 * The template options (upstream's `tools` / `enable_thinking` / `preserve_thinking` keyword
 * arguments of `apply_chat_template`).
 *
 * MUST: one conversation uses the same options for its whole life — `tools` and `enableThinking`
 * shape the system block at the start, and `enableThinking` the tail of every generation prompt.
 */
export type ChatTemplateOptions = {
  /** Function declarations, rendered into the system block in list order. */
  readonly tools?: readonly ChatTool[];
  /** Thinking mode (`<|think|>` in the system block; default off). */
  readonly enableThinking?: boolean;
  /** Keep the thought of tool-calling assistant messages before the last user message. */
  readonly preserveThinking?: boolean;
};

/** The chat spellings (source of truth: `tokenizer_config.json` / `chat_template.jinja`). */
const START_OF_TURN = "<|turn>";
const END_OF_TURN = "<turn|>";
const BOS = "<bos>";
const START_OF_CHANNEL = "<|channel>";
const END_OF_CHANNEL = "<channel|>";
const THOUGHT_CHANNEL = `${START_OF_CHANNEL}thought\n`;
const THINK = "<|think|>";
const START_OF_TOOL = "<|tool>";
const END_OF_TOOL = "<tool|>";
const START_OF_TOOL_CALL = "<|tool_call>";
const END_OF_TOOL_CALL = "<tool_call|>";
const TOOL_RESPONSE = "<|tool_response>";
const END_OF_TOOL_RESPONSE = "<tool_response|>";
/** The string delimiter of the declaration and argument syntax. */
const QUOTE = '<|"|>';

/** The role of the turn the generation prompt opens (what `assistant` maps to). */
const MODEL_ROLE = "model";

/** Roles that fall into the `system` block only when placed first. */
const SYSTEM_ROLES: readonly string[] = ["system", "developer"];

const ROLES: readonly string[] = ["system", "developer", "user", "assistant", "tool"];

/** Role → the fields a message may carry (allow list — pairs with `TOOL_ROLE_KEYS` in Python). */
const MESSAGE_KEYS: Readonly<Record<ChatRole, readonly string[]>> = {
  system: ["role", "content"],
  developer: ["role", "content"],
  user: ["role", "content"],
  assistant: ["role", "content", "reasoning", "reasoning_content", "tool_calls"],
  tool: ["role", "content", "tool_call_id", "name"],
};

/** The keys the template treats as schema keywords when `filter_keys` is on. */
const STANDARD_SCHEMA_KEYS: readonly string[] = [
  "description",
  "type",
  "properties",
  "required",
  "nullable",
];

const OPTION_KEYS: readonly string[] = ["tools", "enableThinking", "preserveThinking"];

/**
 * A word character as the output parser's `\w` reads it (Python's `regex` module: Unicode TR#18 —
 * measured equal to this class on every code point). A new instance per call: a `RegExp` at module
 * scope would be module state.
 */
export const wordCharacterClass = (): string =>
  "[\\p{Alphabetic}\\p{M}\\p{Nd}\\p{Pc}\\p{Join_Control}]";

// ---- JSON values (Python semantics where the template applies them) ------------------------

const isPlainObject = (value: unknown): value is Readonly<Record<string, unknown>> => {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
  const prototype: unknown = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
};

/**
 * Checks that `value` is a JSON value the template can render (finite numbers, plain objects,
 * arrays). `undefined`, functions, class instances and non-finite numbers fail loudly (Python has
 * none of them in JSON, so there is no upstream rendering to match).
 */
const assertJsonValue = (value: unknown, where: string): void => {
  if (value === null || typeof value === "string" || typeof value === "boolean") return;
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new ModelInputError(`${where} is not a finite number`);
    return;
  }
  if (Array.isArray(value)) {
    value.forEach((item, index) => assertJsonValue(item, `${where}[${index}]`));
    return;
  }
  if (!isPlainObject(value)) {
    throw new ModelInputError(`${where} is not a JSON value (${typeof value})`);
  }
  for (const key of Object.keys(value)) assertJsonValue(value[key], `${where}.${key}`);
};

/** Python's truthiness of a JSON value (Jinja's `if`) — empty strings, arrays, objects are false. */
const truthy = (value: unknown): boolean => {
  if (value === undefined || value === null) return false;
  if (typeof value === "boolean") return value;
  if (typeof value === "number") return value !== 0;
  if (typeof value === "string") return value.length > 0;
  if (Array.isArray(value)) return value.length > 0;
  return isPlainObject(value) ? Object.keys(value).length > 0 : true;
};

/** Python's string order (by code point, not by UTF-16 unit). */
const compareCodePoints = (left: string, right: string): number => {
  let i = 0;
  let j = 0;
  for (;;) {
    if (i >= left.length) return j >= right.length ? 0 : -1;
    if (j >= right.length) return 1;
    const a = left.codePointAt(i) as number;
    const b = right.codePointAt(j) as number;
    if (a !== b) return a - b;
    i += a > 0xffff ? 2 : 1;
    j += b > 0xffff ? 2 : 1;
  }
};

/**
 * Jinja's `dictsort` — entries sorted by key, **case-insensitively** (its default), stable for keys
 * equal after lower-casing.
 */
const dictsort = <T>(object: Readonly<Record<string, T>>): [string, T][] =>
  Object.keys(object)
    .map((key): [string, T] => [key, object[key]])
    .sort(([left], [right]) => compareCodePoints(left.toLowerCase(), right.toLowerCase()));

/**
 * A number as Python's `str()` prints it. A JS number does not carry int vs float: an integer value
 * prints as Python's `int` would (`2`, the shape a JSON integer takes in Python), any other value
 * as `repr(float)` (shortest round-trip digits; exponent form below 1e-4).
 */
export const pythonNumber = (value: number): string => {
  if (Number.isInteger(value)) return BigInt(value).toString();
  const [mantissa, exponentText] = value.toExponential().split("e");
  const negative = mantissa.startsWith("-");
  const digits = mantissa.replace("-", "").replace(".", "");
  const exponent = Number(exponentText);
  const point = exponent + 1;
  let body: string;
  if (point <= -4 || point > 16) {
    const fraction = digits.length > 1 ? `.${digits.slice(1)}` : "";
    const sign = exponent < 0 ? "-" : "+";
    body = `${digits[0]}${fraction}e${sign}${String(Math.abs(exponent)).padStart(2, "0")}`;
  } else if (point <= 0) {
    body = `0.${"0".repeat(-point)}${digits}`;
  } else if (point >= digits.length) {
    body = `${digits}${"0".repeat(point - digits.length)}.0`;
  } else {
    body = `${digits.slice(0, point)}.${digits.slice(point)}`;
  }
  return negative ? `-${body}` : body;
};

/**
 * The template's `format_argument` (strings are delimited by `<|"|>`; with `escapeKeys`, object
 * keys too). The value is already checked by {@link assertJsonValue}.
 */
const formatArgument = (value: ChatJsonValue, escapeKeys = true): string => {
  if (value === null) return "null";
  if (typeof value === "string") return `${QUOTE}${value}${QUOTE}`;
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") return pythonNumber(value);
  if (Array.isArray(value)) {
    return `[${value.map((item: ChatJsonValue) => formatArgument(item, escapeKeys)).join(",")}]`;
  }
  const entries = dictsort(value as ChatJsonObject).map(
    ([key, item]) =>
      `${escapeKeys ? `${QUOTE}${key}${QUOTE}` : key}:${formatArgument(item, escapeKeys)}`,
  );
  return `{${entries.join(",")}}`;
};

// ---- tool declarations ------------------------------------------------------------------------

/** `value | upper` where the template prints it (absent = the empty string). */
const upperType = (value: unknown, where: string): string => {
  if (value === undefined) return "";
  if (typeof value !== "string") {
    throw new ModelInputError(
      `${where}.type is not a string (the template prints it with Python's str() — a list would ` +
        `render as a Python repr; use one type and \`nullable\`)`,
    );
  }
  return value.toUpperCase();
};

/** A string the template prints with `{{ … }}` (only where it tests truthiness first). */
const printedString = (value: unknown, where: string): string => {
  if (typeof value !== "string") {
    throw new ModelInputError(
      `${where} is not a string (the template prints it with Python's str())`,
    );
  }
  return value;
};

/** A `required` list (`<|"|>name<|"|>` each, comma-separated). */
const formatRequired = (value: unknown, where: string): string => {
  if (!Array.isArray(value) || !value.every((item) => typeof item === "string")) {
    throw new ModelInputError(`${where} is not an array of strings`);
  }
  return value.map((item: string) => `${QUOTE}${item}${QUOTE}`).join(",");
};

/**
 * The template's `format_parameters` (one property spec per key, keys in `dictsort` order).
 *
 * A property spec that is not an object renders as an empty one (`key:{type:<|"|><|"|>}`) — Jinja
 * looks up every field of a non-mapping as undefined.
 */
const formatParameters = (
  properties: Readonly<Record<string, ChatJsonValue>>,
  filterKeys: boolean,
  where: string,
): string => {
  let out = "";
  let first = true;
  for (const [key, raw] of dictsort(properties)) {
    if (filterKeys && STANDARD_SCHEMA_KEYS.includes(key)) continue;
    if (!first) out += ",";
    first = false;
    const at = `${where}.${key}`;
    const value: Readonly<Record<string, ChatJsonValue>> = isPlainObject(raw)
      ? (raw as ChatJsonObject)
      : {};
    out += `${key}:{`;
    let comma = false;
    const separate = (): void => {
      if (comma) out += ",";
      comma = true;
    };
    if (truthy(value.description)) {
      out += `description:${QUOTE}${printedString(value.description, `${at}.description`)}${QUOTE}`;
      comma = true;
    }
    const type = upperType(value.type, at);
    if (type === "STRING") {
      if (truthy(value.enum)) {
        separate();
        out += `enum:${formatArgument(value.enum)}`;
      }
    } else if (type === "ARRAY") {
      const items = value.items;
      if (isPlainObject(items) && truthy(items)) {
        separate();
        out += `items:{${formatItems(items as ChatJsonObject, `${at}.items`)}}`;
      }
    }
    if (truthy(value.nullable)) {
      separate();
      out += "nullable:true";
    }
    if (type === "OBJECT") {
      separate();
      out += isPlainObject(value.properties)
        ? `properties:{${formatParameters(value.properties as ChatJsonObject, false, `${at}.properties`)}}`
        : `properties:{${formatParameters(value, true, at)}}`;
      if (truthy(value.required)) {
        separate();
        out += `required:[${formatRequired(value.required, `${at}.required`)}]`;
      }
    }
    separate();
    out += `type:${QUOTE}${type}${QUOTE}}`;
  }
  return out;
};

/** The `items` block of an array property (each key in `dictsort` order, `null` values skipped). */
const formatItems = (items: ChatJsonObject, where: string): string => {
  const parts: string[] = [];
  for (const [key, value] of dictsort(items)) {
    if (value === null) continue;
    if (key === "properties") {
      const inner = isPlainObject(value)
        ? formatParameters(value as ChatJsonObject, false, `${where}.properties`)
        : "";
      parts.push(`properties:{${inner}}`);
    } else if (key === "required") {
      parts.push(`required:[${formatRequired(value, `${where}.required`)}]`);
    } else if (key === "type") {
      if (typeof value === "string") parts.push(`type:${formatArgument(value.toUpperCase())}`);
      else if (Array.isArray(value) && value.every((item) => typeof item === "string")) {
        parts.push(`type:${formatArgument(value.map((item) => (item as string).toUpperCase()))}`);
      } else {
        throw new ModelInputError(`${where}.type is neither a string nor an array of strings`);
      }
    } else {
      parts.push(`${key}:${formatArgument(value)}`);
    }
  }
  return parts.join(",");
};

/** Checks a function or call name ({@link ChatTool} — the parser reads it back with `\w+`). */
const assertToolName = (name: unknown, where: string): string => {
  if (typeof name !== "string" || !new RegExp(`^${wordCharacterClass()}+$`, "u").test(name)) {
    throw new ModelInputError(
      `${where} ${JSON.stringify(name)} is not a word (\\w+): the output parser reads a call's ` +
        `name with \\w+, so a tool named otherwise could be declared but never called back`,
    );
  }
  return name;
};

/** The template's `format_function_declaration` (then `| trim`). */
const formatFunctionDeclaration = (tool: unknown, where: string): string => {
  if (!isPlainObject(tool)) throw new ModelInputError(`${where} is not an object`);
  assertKeys(tool, ["type", "function"], where);
  if (tool.type !== "function") {
    throw new ModelInputError(`${where}.type is not 'function' (${JSON.stringify(tool.type)})`);
  }
  const fn = tool.function;
  if (!isPlainObject(fn)) throw new ModelInputError(`${where}.function is not an object`);
  assertKeys(fn, ["name", "description", "parameters", "response"], `${where}.function`);
  const name = assertToolName(fn.name, `${where}.function.name`);
  if (fn.description !== undefined && typeof fn.description !== "string") {
    throw new ModelInputError(`${where}.function.description is not a string`);
  }
  for (const key of ["parameters", "response"] as const) {
    const value = fn[key];
    if (value !== undefined && value !== null && !isPlainObject(value)) {
      throw new ModelInputError(`${where}.function.${key} is not an object`);
    }
    assertJsonValue(value ?? null, `${where}.function.${key}`);
  }
  let out = `declaration:${name}{description:${QUOTE}${fn.description ?? ""}${QUOTE}`;
  const params = (fn.parameters ?? null) as ChatJsonObject | null;
  if (params !== null && truthy(params)) {
    const at = `${where}.function.parameters`;
    out += ",parameters:{";
    if (truthy(params.properties)) {
      if (!isPlainObject(params.properties)) {
        throw new ModelInputError(`${at}.properties is not an object`);
      }
      out += `properties:{${formatParameters(params.properties as ChatJsonObject, false, `${at}.properties`)}},`;
    }
    if (truthy(params.required)) {
      out += `required:[${formatRequired(params.required, `${at}.required`)}],`;
    }
    if (truthy(params.type)) out += `type:${QUOTE}${upperType(params.type, at)}${QUOTE}}`;
  }
  if (Object.hasOwn(fn, "response")) {
    const at = `${where}.function.response`;
    const response = (fn.response ?? {}) as ChatJsonObject;
    out += ",response:{";
    if (truthy(response.description)) {
      out += `description:${QUOTE}${printedString(response.description, `${at}.description`)}${QUOTE},`;
    }
    const type = upperType(response.type, at);
    if (type === "OBJECT") out += `type:${QUOTE}${type}${QUOTE}}`;
  }
  return pythonStrip(`${out}}`);
};

// ---- messages ---------------------------------------------------------------------------------

/** Unknown keys fail loudly (silently ignoring them = "passed but has no effect"). */
const assertKeys = (
  object: Readonly<Record<string, unknown>>,
  allowed: readonly string[],
  where: string,
): void => {
  const extra = Object.keys(object).filter((key) => !allowed.includes(key));
  if (extra.length > 0) {
    throw new ModelInputError(
      `${where}: out-of-scope fields ${extra.join(" / ")}` +
        ` (silently ignoring them would let "passed but has no effect" through without an exception)`,
    );
  }
};

/** Checks one tool call of an assistant message. */
const assertToolCall = (call: unknown, where: string): ChatToolCall => {
  if (!isPlainObject(call)) throw new ModelInputError(`${where} is not an object`);
  assertKeys(call, ["id", "type", "function"], where);
  if (call.id !== undefined && typeof call.id !== "string") {
    throw new ModelInputError(`${where}.id is not a string`);
  }
  if (call.type !== undefined && call.type !== "function") {
    throw new ModelInputError(`${where}.type is not 'function' (${JSON.stringify(call.type)})`);
  }
  const fn = call.function;
  if (!isPlainObject(fn)) throw new ModelInputError(`${where}.function is not an object`);
  assertKeys(fn, ["name", "arguments"], `${where}.function`);
  assertToolName(fn.name, `${where}.function.name`);
  if (typeof fn.arguments === "string") {
    throw new ModelInputError(
      `${where}.function.arguments is a string — it must be a JSON object (the template refuses ` +
        `a string too: deserialize the arguments before passing them)`,
    );
  }
  if (!Object.hasOwn(fn, "arguments") || (fn.arguments !== null && !isPlainObject(fn.arguments))) {
    throw new ModelInputError(`${where}.function.arguments is neither an object nor null`);
  }
  assertJsonValue(fn.arguments, `${where}.function.arguments`);
  return call as ChatToolCall;
};

/** Checks one message against its role's allow list (structure across messages is checked later). */
const assertMessage = (message: unknown, where: string): ChatMessage => {
  if (!isPlainObject(message)) throw new ModelInputError(`${where} is not an object`);
  const role = message.role;
  if (typeof role !== "string" || !ROLES.includes(role)) {
    throw new ModelInputError(
      `${where}.role '${String(role)}' is none of ${ROLES.join(" / ")}` +
        ` ('model' is the template's output-side spelling, not an input role)`,
    );
  }
  if (Object.hasOwn(message, "tool_responses")) {
    throw new ModelInputError(
      `${where}: the legacy field tool_responses is out of scope — send each result as a role ` +
        `'tool' message after the assistant message with tool_calls (ADR 0084 addendum 2026-11-04)`,
    );
  }
  if (Object.hasOwn(message, "thinking")) {
    throw new ModelInputError(
      `${where}: the template does not read 'thinking' — a thought goes in 'reasoning' ` +
        `(parseChatResponse returns it as 'thinking'; passing that back unchanged would drop it)`,
    );
  }
  assertKeys(message, MESSAGE_KEYS[role as ChatRole], `${where} (role '${role}')`);
  const content = message.content;
  if (Array.isArray(content)) {
    throw new ModelInputError(
      `${where}.content is an array — content parts (text / image / audio / video) are out of ` +
        `scope (ADR 0084 decision 5)`,
    );
  }
  if (role === "assistant") {
    if (content !== undefined && content !== null && typeof content !== "string") {
      throw new ModelInputError(`${where}.content is neither a string nor null`);
    }
    for (const key of ["reasoning", "reasoning_content"] as const) {
      if (message[key] !== undefined && typeof message[key] !== "string") {
        throw new ModelInputError(`${where}.${key} is not a string`);
      }
    }
    const calls = message.tool_calls;
    if (calls !== undefined) {
      if (!Array.isArray(calls)) throw new ModelInputError(`${where}.tool_calls is not an array`);
      calls.forEach((call, index) => assertToolCall(call, `${where}.tool_calls[${index}]`));
    }
  } else if (role === "tool") {
    if (
      content !== null &&
      typeof content !== "string" &&
      typeof content !== "boolean" &&
      typeof content !== "number"
    ) {
      throw new ModelInputError(
        `${where}.content is not a string, number, boolean or null (the template reads an ` +
          `object as a content-part array and raises — serialize it to a string)`,
      );
    }
    assertJsonValue(content, `${where}.content`);
    for (const key of ["tool_call_id", "name"] as const) {
      if (message[key] !== undefined && typeof message[key] !== "string") {
        throw new ModelInputError(`${where}.${key} is not a string`);
      }
    }
  } else if (typeof content !== "string") {
    throw new ModelInputError(`${where}.content is not a string`);
  }
  return message as ChatMessage;
};

/** A content that untyped input may leave null or absent (an assistant message) → its text. */
const textOf = (content: unknown): string => (typeof content === "string" ? content : "");

/** The tool calls of an assistant message (an empty list counts as none, as in the template). */
const callsOf = (message: ChatMessage | undefined): readonly ChatToolCall[] =>
  message?.role === "assistant" ? (message.tool_calls ?? []) : [];

/**
 * The function name of one tool result — the template's resolution, accepted only where it is
 * unambiguous.
 *
 * Upstream: `name or 'unknown'`, then overridden by **every** call whose `id` equals the result's
 * `tool_call_id` — and an absent id equals an absent id (`None == None`), so with several calls
 * without ids every result gets the last call's name. Accepted here: a `tool_call_id` that matches
 * exactly one call (and agrees with `name` if given); without `tool_call_id`, the only call without
 * an id (agreeing with `name`), or `name` when every call has an id.
 */
const resolveToolName = (
  calls: readonly ChatToolCall[],
  result: ChatToolMessage,
  where: string,
): string => {
  const id = result.tool_call_id;
  const matched = calls.filter((call) => call.id === id);
  let name: string | undefined;
  if (id !== undefined) {
    if (matched.length !== 1) {
      throw new ModelInputError(
        `${where}.tool_call_id '${id}' matches ${matched.length} of the calls (exactly one is ` +
          `needed — upstream would fall back to 'name' or the spelling 'unknown')`,
      );
    }
    name = matched[0].function.name;
  } else if (matched.length > 1) {
    throw new ModelInputError(
      `${where} has no tool_call_id and ${matched.length} calls have no id — the template would ` +
        `give every such result the last call's name (give the calls ids and the results tool_call_id)`,
    );
  } else {
    name = matched[0]?.function.name ?? result.name;
  }
  if (name === undefined) {
    throw new ModelInputError(
      `${where} names no function (no tool_call_id, no name, and every call has an id — the ` +
        `template would write 'unknown')`,
    );
  }
  if (result.name !== undefined && result.name !== name) {
    throw new ModelInputError(
      `${where}.name '${result.name}' contradicts the call it answers ('${name}' — the template ` +
        `would write '${name}')`,
    );
  }
  return name;
};

/** Checks the template options (allow list and types). */
const assertOptions = (options: unknown, where: string): Required<ChatTemplateOptions> => {
  if (!isPlainObject(options)) throw new ModelInputError(`${where}: the options are not an object`);
  assertKeys(options, OPTION_KEYS, `${where}: options`);
  for (const key of ["enableThinking", "preserveThinking"] as const) {
    if (options[key] !== undefined && typeof options[key] !== "boolean") {
      throw new ModelInputError(`${where}: options.${key} is not a boolean`);
    }
  }
  const tools = options.tools ?? [];
  if (!Array.isArray(tools)) throw new ModelInputError(`${where}: options.tools is not an array`);
  return {
    tools: tools as readonly ChatTool[],
    enableThinking: options.enableThinking === true,
    preserveThinking: options.preserveThinking === true,
  };
};

/**
 * Checks template options as {@link renderGemma4Chat} does, declarations included — for an owner
 * that fixes them before its first render (`ChatSession`'s constructor fails synchronously).
 *
 * NOTE: internal (not on the public surface); exported for `../chat-session.ts`.
 */
export const assertChatTemplateOptions = (options: ChatTemplateOptions, where: string): void => {
  const { tools } = assertOptions(options, where);
  tools.forEach((tool, index) =>
    formatFunctionDeclaration(tool, `${where}: options.tools[${index}]`),
  );
};

/**
 * Checks the conversation as a whole (each message, then the structure the template needs).
 *
 * MUST: also reject an empty conversation — upstream `apply_chat_template` rejects it with a
 * `ValueError`, so the accepted set is not made wider than upstream's.
 */
const assertConversation = (messages: readonly unknown[], where: string): void => {
  if (!Array.isArray(messages) || messages.length === 0) {
    throw new ModelInputError(
      `${where}: the conversation is empty (the upstream apply_chat_template rejects it too)`,
    );
  }
  const checked = messages.map((message, index) =>
    assertMessage(message, `${where}: messages[${index}]`),
  );
  let owner: ChatMessage | undefined;
  checked.forEach((message, index) => {
    const at = `${where}: messages[${index}]`;
    if (message.role === "tool") {
      if (owner === undefined) {
        throw new ModelInputError(
          `${at} (role 'tool') follows no assistant message with tool_calls (the template ` +
            `drops such a result without a trace)`,
        );
      }
      resolveToolName(callsOf(owner), message, at);
      return;
    }
    owner = callsOf(message).length > 0 ? message : undefined;
    if (owner !== undefined && checked[index + 1]?.role !== "tool") {
      throw new ModelInputError(
        `${at} has tool_calls but no role 'tool' result follows (the template would leave a bare ` +
          `${TOOL_RESPONSE} — send the results, or render the conversation without the calls)`,
      );
    }
    if (
      owner !== undefined &&
      stripThinking(textOf(message.content)) !== "" &&
      checked.slice(index + 1).every((next) => next.role === "tool")
    ) {
      throw new ModelInputError(
        `${at} has both content and tool_calls, and its results end the conversation: the ` +
          `template writes the content after the results and closes the turn, leaving no ` +
          `generation prompt (record content written before the calls as its own assistant ` +
          `message in front of the tool-calling one)`,
      );
    }
  });
};

// ---- rendering --------------------------------------------------------------------------------

/**
 * What the generation prompt appends after `<|turn>model\n` (only when it opens a new model turn).
 *
 * `"gemma4-unified"` (gemma-4-12B-it) opens and closes an empty thought channel there when thinking
 * is off (`chat_template.jinja` at 43c9d014, lines 384-386). The E2B / E4B template appends
 * nothing, and neither does 12B's with thinking on.
 */
const generationPromptTail = (dialect: ChatDialect, enableThinking: boolean): string =>
  dialect === "gemma4-unified" && !enableThinking ? `${THOUGHT_CHANNEL}${END_OF_CHANNEL}` : "";

/** The template's `strip_thinking` (a spelled thought channel is removed, then the trim). */
const stripThinking = (text: string): string => {
  let result = "";
  for (const part of text.split(END_OF_CHANNEL)) {
    result += part.includes(START_OF_CHANNEL) ? part.split(START_OF_CHANNEL)[0] : part;
  }
  return pythonStrip(result);
};

/** One tool call as the template writes it (argument keys unescaped, in `dictsort` order). */
const formatToolCall = (call: ChatToolCall): string => {
  const args = call.function.arguments;
  const body =
    args === null
      ? ""
      : dictsort(args)
          .map(([key, value]) => `${key}:${formatArgument(value, false)}`)
          .join(",");
  return `${START_OF_TOOL_CALL}call:${call.function.name}{${body}}${END_OF_TOOL_CALL}`;
};

/** One tool result block (`format_tool_response_block` with a non-mapping response). */
const formatToolResult = (name: string, result: ChatToolMessage): string =>
  `${TOOL_RESPONSE}response:${name}{value:${formatArgument(result.content, false)}}${END_OF_TOOL_RESPONSE}`;

/**
 * Conversation → the same string as upstream `apply_chat_template(add_generation_prompt=True)`
 * with the same options.
 *
 * The branches of the template, in its order (`chat_template.jinja` of E2B at 4adf1574; 12B's
 * line numbers are one higher after line 234):
 *
 * 1. The system block (lines 190-216) opens when thinking is on, tools are given, or **the first
 *    message is `system` / `developer`**: `<|turn>system\n`, `<|think|>\n` when thinking, the first
 *    message's trimmed text, each declaration as `<|tool>…<tool|>`, `<turn|>\n`. Even `developer`
 *    is spelled `system` there — a later `developer` is `<|turn>developer\n`.
 * 2. Each non-tool message (lines 227-377): `assistant` is mapped to `model`, and a model turn is
 *    **not reopened** when the previous non-tool message was also `assistant`. Then the thought
 *    (`<|channel>thought\n…\n<channel|>`, gated as {@link ChatAssistantMessage} says), the tool
 *    calls, the results that follow (consumed here, in order), the content (`strip_thinking` for a
 *    model turn, a trim otherwise). The turn closes with `<turn|>\n` unless it continues into the
 *    next assistant message, or it ends the conversation on its results with no content (the model
 *    turn stays open for generation).
 * 3. The generation prompt (lines 380-386): `<|turn>model\n` (12B: + the empty thought channel when
 *    thinking is off) — unless the last message was a tool result, after which the open model turn
 *    continues (`<|channel>thought\n` when thinking is on, nothing otherwise).
 *
 * NOTE: not exposed on the public surface (the thin surface of ADR 0008 — the consumer's entry is
 * {@link chatPrompt}). `export` is so that the fixture comparison looks at "rendering" and
 * "encoding" separately — when they diverge, the reader learns which step it was.
 */
export const renderGemma4Chat = (
  messages: readonly ChatMessage[],
  dialect: ChatDialect = DEFAULT_CHAT_DIALECT,
  options: ChatTemplateOptions = {},
): string => {
  const where = "gemma.chatPrompt";
  const { tools, enableThinking, preserveThinking } = assertOptions(options, where);
  assertConversation(messages, where);
  const declarations = tools.map((tool, index) =>
    formatFunctionDeclaration(tool, `${where}: options.tools[${index}]`),
  );
  let out = BOS;
  /** The template's `ns.prev_message_type` (only the two tool states are read at the end). */
  let previousType: "tool_call" | "tool_response" | undefined;
  let loop = messages;
  const systemFirst = SYSTEM_ROLES.includes(messages[0].role);
  if (enableThinking || declarations.length > 0 || systemFirst) {
    out += `${START_OF_TURN}system\n`;
    if (enableThinking) out += `${THINK}\n`;
    if (systemFirst) {
      out += pythonStrip((messages[0] as ChatTextMessage).content);
      loop = messages.slice(1);
    }
    for (const declaration of declarations) out += `${START_OF_TOOL}${declaration}${END_OF_TOOL}`;
    out += `${END_OF_TURN}\n`;
  }
  const lastUser = loop.findLastIndex((message) => message.role === "user");
  // Continuation is decided on the role **before mapping** (after mapping to `model`, an assistant
  // right after the system block would look like a "continuation").
  let previousRole: ChatRole | undefined;
  loop.forEach((message, index) => {
    if (message.role === "tool") return;
    previousType = undefined;
    const role = message.role === "assistant" ? MODEL_ROLE : message.role;
    if (!(role === MODEL_ROLE && previousRole === "assistant")) out += `${START_OF_TURN}${role}\n`;
    const calls = callsOf(message);
    if (message.role === "assistant") {
      const thought = message.reasoning || message.reasoning_content;
      const gate = index > lastUser || (preserveThinking && calls.length > 0);
      if (thought !== undefined && thought !== "" && gate) {
        out += `${THOUGHT_CHANNEL}${thought}\n${END_OF_CHANNEL}`;
      }
    }
    for (const call of calls) out += formatToolCall(call);
    if (calls.length > 0) previousType = "tool_call";
    let answered = false;
    if (calls.length > 0) {
      for (let next = index + 1; loop[next]?.role === "tool"; next += 1) {
        const result = loop[next] as ChatToolMessage;
        out += formatToolResult(resolveToolName(calls, result, where), result);
        answered = true;
        previousType = "tool_response";
      }
    }
    // Untyped input may carry a null or absent assistant content (OpenAI's shape) — empty.
    const content: unknown = message.content;
    const captured =
      typeof content !== "string"
        ? ""
        : role === MODEL_ROLE
          ? stripThinking(content)
          : pythonStrip(content);
    out += captured;
    const nextRole = loop.slice(index + 1).find((next) => next.role !== "tool")?.role;
    const continues =
      role === MODEL_ROLE && nextRole === "assistant" && (calls.length === 0 || answered);
    // `previousType === "tool_call" && !answered` (a bare `<|tool_response>`) cannot happen here:
    // `assertConversation` rejects a call without results.
    if (!continues && !(answered && pythonStrip(captured) === "" && nextRole === undefined)) {
      out += `${END_OF_TURN}\n`;
    }
    previousRole = message.role;
  });
  if (previousType === "tool_response") return enableThinking ? `${out}${THOUGHT_CHANNEL}` : out;
  return `${out}${START_OF_TURN}${MODEL_ROLE}\n${generationPromptTail(dialect, enableThinking)}`;
};

/**
 * Conversation → token ids (`<bos>` included, ending with the generation prompt).
 *
 * Built in the same two steps as upstream (render → encode with `add_special_tokens=False`). The
 * chat spellings are added vocabulary, so the leftmost-longest split of `encode` turns each into one
 * token as is.
 *
 * The pass line is an **id-sequence match** with the output of HF `apply_chat_template`; the
 * fixtures are taken independently from upstream by `tools/export-recipes/gemma4/chat.py` (ADR 0084
 * decision 7).
 *
 * The dialect is the tokenizer's ({@link Tokenizer.chatDialect} — the distribution declares it next
 * to the tokenizer it ships), so a 12B pipeline's tokenizer renders the 12B template.
 */
export const chatPrompt = (
  tokenizer: Tokenizer,
  messages: readonly ChatMessage[],
  options: ChatTemplateOptions = {},
): number[] => tokenizer.encode(renderGemma4Chat(messages, tokenizer.chatDialect, options));

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
 * So the identity that holds is (`⧺` is token-sequence concatenation; the same `options` all
 * along):
 *
 * ```text
 * chatPrompt(whole conversation)
 *   = chatPrompt(up to the first turn) ⧺ generated body ⧺ [<turn|>]
 *     ⧺ chatTurn(next message) ⧺ …
 * ```
 *
 * A turn that stopped at a tool call continues with {@link renderGemma4ToolResults} instead (its
 * pending token is `<|tool_response>`); the identity over a whole tool conversation is in the doc of
 * `ChatSession` (`../chat-session.ts`). The check in `gemma4_chat_test.ts` verifies the identity over
 * arbitrary splits (it breaks there if the template's turn-locality breaks — the source of truth that
 * keeps consumers from writing a guess such as "strip the leading `<bos>`").
 *
 * MUST: usable only **right after generation closed with `<turn|>`**. A turn cut off by max-tokens or
 * `break` has an unclosed model turn, so the right continuation is further generation
 * (`prompt: []`), not a delta.
 *
 * MUST: reject `assistant` and `tool`. The model turn is the slot generation fills, and the upstream
 * template also folds consecutive assistants into one turn, so "appending an assistant after a closed
 * model turn" cannot be rendered as a delta (re-render the whole conversation with {@link chatPrompt}
 * to insert one); a tool result continues the open model turn ({@link renderGemma4ToolResults}).
 *
 * NOTE: in the `"gemma4-unified"` dialect with thinking off the identity above holds only up to the
 * empty thought channel. Every generation prompt that opens a model turn — the first one and each
 * delta's — ends with `<|channel>thought\n<channel|>`, so a conversation carried in the KV keeps
 * that channel at the start of every past model turn, while the upstream template re-rendering the
 * same history writes past model turns without it (`strip_thinking` + no reasoning field). Both are
 * the template's own spellings; only a whole re-render (`chatPrompt`) matches `apply_chat_template`
 * token for token.
 */
export const renderGemma4ChatTurn = (
  message: ChatMessage,
  dialect: ChatDialect = DEFAULT_CHAT_DIALECT,
  options: ChatTemplateOptions = {},
): string => {
  const where = "gemma.chatTurn";
  const { enableThinking } = assertOptions(options, where);
  const checked = assertMessage(message, `${where}: message`);
  if (checked.role === "assistant") {
    throw new ModelInputError(
      `${where}: role 'assistant' cannot be a delta` +
        ` (the model turn is the slot generation fills, and the template folds consecutive assistant turns into one — ` +
        `to insert one into the conversation, re-render the whole conversation with gemma.chatPrompt)`,
    );
  }
  if (checked.role === "tool") {
    throw new ModelInputError(
      `${where}: role 'tool' cannot be a delta of chatTurn (a result continues the open model ` +
        `turn after ${TOOL_RESPONSE} — use gemma.chatToolResults)`,
    );
  }
  // The leading newline is the one character "right after the `<turn|>` that closed the previous
  // turn" (the `${END_OF_TURN}\n` of `renderGemma4Chat`). `<turn|>` itself is prepended by
  // pendingToken, so it is not here.
  return (
    `\n${START_OF_TURN}${checked.role}\n${pythonStrip(checked.content)}${END_OF_TURN}\n` +
    `${START_OF_TURN}${MODEL_ROLE}\n${generationPromptTail(dialect, enableThinking)}`
  );
};

/**
 * One new message → **a delta token sequence that can be passed to the next `generate` as is** (the
 * doc of {@link renderGemma4ChatTurn} is the source of truth for the turn-local contract — the
 * `<turn|>` that closes the previous turn is not included).
 *
 * This is the entry for running a multi-turn conversation with `Pipeline.sequence()`. Past
 * turns are in the context's KV, so there is no need to re-render the whole conversation every turn
 * (= an O(n²) prefill).
 */
export const chatTurn = (
  tokenizer: Tokenizer,
  message: ChatMessage,
  options: ChatTemplateOptions = {},
): number[] => tokenizer.encode(renderGemma4ChatTurn(message, tokenizer.chatDialect, options));

/**
 * Tool results → **the delta that continues the model turn after a tool call** (the counterpart of
 * {@link renderGemma4ChatTurn} for a turn that stopped at `<|tool_response>`).
 *
 * Generation stops at `<|tool_response>` (it is in the EOS set — {@link gemma4StopTokens}), and the
 * sequence holds it as the pending token and prepends it to the next `generate`, as it does for
 * `<turn|>`. So the delta starts **right after** it: `response:NAME{value:…}<tool_response|>` for
 * the first result, `<|tool_response>response:…<tool_response|>` for each further one, then the
 * generation-prompt tail of a continued model turn — `<|channel>thought\n` with thinking on,
 * nothing with it off (the same in both dialects: the template's prompt after a result opens no new
 * turn).
 *
 * `toolCalls` are the calls of the turn that stopped (they resolve each result's function name).
 *
 * MUST: the tool-calling message carries no content. The template writes an assistant message's
 * content after its results, so content the model wrote before its calls is recorded as its own
 * assistant message in front of the tool-calling one (what `ChatSession` does); a tool-calling
 * message with content would break the identity with the whole render.
 */
export const renderGemma4ToolResults = (
  toolCalls: readonly ChatToolCall[],
  results: readonly ChatToolMessage[],
  options: ChatTemplateOptions = {},
): string => {
  const where = "gemma.chatToolResults";
  const { enableThinking } = assertOptions(options, where);
  if (!Array.isArray(toolCalls) || toolCalls.length === 0) {
    throw new ModelInputError(`${where}: there are no tool calls to answer`);
  }
  toolCalls.forEach((call, index) => assertToolCall(call, `${where}: toolCalls[${index}]`));
  if (!Array.isArray(results) || results.length === 0) {
    throw new ModelInputError(`${where}: there are no results (at least one role 'tool' message)`);
  }
  const blocks = results.map((result, index) => {
    const at = `${where}: results[${index}]`;
    const checked = assertMessage(result, at);
    if (checked.role !== "tool") {
      throw new ModelInputError(`${at}.role is '${checked.role}', not 'tool'`);
    }
    return formatToolResult(resolveToolName(toolCalls, checked, at), checked);
  });
  return `${blocks.join("").slice(TOOL_RESPONSE.length)}${enableThinking ? THOUGHT_CHANNEL : ""}`;
};

/**
 * Tool results → **the delta tokens that continue the model turn after a tool call** (the doc of
 * {@link renderGemma4ToolResults} is the source of truth — the `<|tool_response>` that stopped
 * generation is not included).
 */
export const chatToolResults = (
  tokenizer: Tokenizer,
  toolCalls: readonly ChatToolCall[],
  results: readonly ChatToolMessage[],
  options: ChatTemplateOptions = {},
): number[] => tokenizer.encode(renderGemma4ToolResults(toolCalls, results, options));

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
export const gemma4StopTokens = (tokenizer: Tokenizer): number[] => {
  const stops = [tokenizer.eosId];
  for (const spelling of [END_OF_TURN, TOOL_RESPONSE]) {
    const id = tokenizer.addedTokenId(spelling);
    if (id === undefined) {
      throw new Error(
        `gemma4StopTokens: stop token '${spelling}' is not in the tokenizer asset's added vocabulary` +
          ` (the chat format and the EOS set come from the same digest set — ADR 0084 decision 5)`,
      );
    }
    stops.push(id);
  }
  return stops;
};
