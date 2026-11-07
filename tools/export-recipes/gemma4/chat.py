"""Harvests the parity fixture of the gemma4 E2B chat format (ADR 0084 decisions 5 and 7).

    uv run --with 'transformers==5.14.1' python -m gemma4.chat
    uv run --with 'transformers==5.14.1' python -m gemma4.chat --model 12b

One output (**git-tracked**):

    packages/models/tests/fixtures/gemma-text/gemma4-chat.json
    packages/models/tests/fixtures/gemma-text/gemma4-12b-chat.json   (--model 12b)

12B renders the `gemma4-unified` dialect ({@link CHAT_DIALECTS}): its template is read from the
pinned hub snapshot (`gemma4.sources`), and the fixture is harvested from its own
`apply_chat_template`, the same way as E2B's.

The TS implementation is one pure function in `packages/models/src/gemma/text/chat.ts`
(`gemma.chatPrompt`), and no general template renderer is built (ADR 0084 decision 5 — one would
go against both "runtime dependencies are Web standards only" and "no speculative
generalization"). This script holds only its **expectations**.

MUST: the expectations are harvested by calling upstream `apply_chat_template` **independently**
(ADR 0084 decision 7) — not one line goes through an in-house renderer or the compile script's
folding. Sharing them makes the parity check tautological (the worked example in the ADR 0048
addendum).

## Scope: five batteries (ADR 0084 decision 5 and its addendum of 2026-11-04, ADR 0144)

The fixture carries up to six lists, each harvested from upstream on its own:

- `chat` ({@link CASES}) — plain conversations: roles `system` / `developer` / `user` /
  `assistant`, string content, no template options. Unchanged since the first scope; its gate
  ({@link assert_plain_conversation}) still rejects every other field.
- `toolChat` ({@link TOOL_CASES}) — function declarations (`tools`, the OpenAI shape the template
  reads), assistant `tool_calls`, role `tool` results, thinking (`enable_thinking`, `reasoning` /
  `reasoning_content`, `preserve_thinking`, `strip_thinking` of assistant content). Gate:
  {@link assert_tool_conversation}.
- `rejectedChat` ({@link REJECTED_CASES}) — inputs the TS side rejects although upstream renders
  them (or raises), with upstream's outcome recorded as evidence for the rejection.
- `responses` ({@link RESPONSE_CASES}) — model output strings and what upstream
  `tokenizer.parse_response(text, prefix=...)` makes of them (or the error it raises), the
  expectations of the TS output parser.
- `imageChat` / `rejectedImageChat` ({@link IMAGE_CASES} / {@link REJECTED_IMAGE_CASES} — 12B only)
  — user messages with text and image parts, through upstream's **processor**
  (`Gemma4UnifiedProcessor.apply_chat_template`), which renders the template and expands
  `<|image|>` to `<|image>`, N × `<|image|>`, `<image|>`. An image is named by its size: the soft
  tokens depend on nothing else, and the pixels are upstream's zero image.

Still out of scope (the TS side rejects them, and no case is harvested): the legacy
`tool_responses` field on assistant messages, and `<|audio|>` / `<|video|>` parts. **Nothing is
silently ignored** — ignoring `tool_responses` would let "I passed the results and they were not
used" through without an exception.

## Scope of plain conversations (the branches measured)

The spellings are not the Gemma 3 ones (ADR 0084 Context 3 — `<start_of_turn>` is **not in this
vocabulary**). Only three branches are actually taken, and {@link CASES} takes each one:

1. A leading `system` / `developer` falls into **the system block `<|turn>system\\n`**
   (`developer` is spelled `system` there too — a later `developer` is `<|turn>developer\\n`)
2. `assistant` maps to `model`, and **when the previous non-tool role is also `assistant`, the
   turn is not reopened** (consecutive assistant messages fold into one model turn)
3. The generation prompt `<|turn>model\\n` is always emitted at the end

The template emits `<bos>` = **the host's job** (the MUST of ADR 0084 decision 5 — `encode` does
not add it; without the split, introducing chat would give a double BOS).

## The EOS set is harvested from the same bundle (ADR 0083 decision 8 / 0084 decision 5)

The fixture carries `eos_token_id` of `generation_config.json` and the spellings of those ids.
The TS side derives the same set **from the added vocabulary of the tokenizer asset**, so this is
what "the derivation matches the upstream declaration" is checked against. Picking up the chat
format and the EOS set from different places lets one of them go stale, so the provenance record
({@link _source}) writes the digests of the four files as one bundle.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from _shared.gemma_tokenizer import (
    CompiledTokenizer,
    asset_subset,
    compile_tokenizer,
    file_digest,
    spec_of,
    write_json,
)
from _shared.paths import REPO_ROOT
from gemma4 import sources

#: The chat dialects — the closed set of `pipelineConfig.chatDialect` (the TS side's
#: `gemma.chatPrompt` renders each one; an absent key reads as {@link GEMMA4_CHAT_DIALECT}).
#: `gemma4` is E2B / E4B's `chat_template.jinja` (byte-identical in the two snapshots);
#: `gemma4-unified` is 12B's, which differs from it in exactly one place that renders anything —
#: see {@link GENERATION_PROMPT_SUFFIX} (the pytest renders every case through 12B's own template).
#: The tool and thinking parts are the same text in both.
GEMMA4_CHAT_DIALECT = "gemma4"
GEMMA4_UNIFIED_CHAT_DIALECT = "gemma4-unified"

#: Dialect → what the generation prompt appends after `<|turn>model\n` with thinking off. 12B's
#: template (lines 380-386 at the pinned revision) opens and closes an empty thought channel there
#: when `enable_thinking` is false — its default — so a host that omitted it would hand the model a
#: prompt it never saw in training. The suffix rides only on a generation prompt that opens a new
#: model turn: after a tool result the prompt continues the open model turn and is the same in
#: both dialects (`<|channel>thought\n` with thinking on, nothing with it off).
GENERATION_PROMPT_SUFFIX: Mapping[str, str] = {
    GEMMA4_CHAT_DIALECT: "",
    GEMMA4_UNIFIED_CHAT_DIALECT: "<|channel>thought\n<channel|>",
}

#: Model → its chat dialect. Key set = `gemma4.card.GEMMA4_UPSTREAM` (tests).
CHAT_DIALECTS: Mapping[str, str] = {
    "e2b": GEMMA4_CHAT_DIALECT,
    "e4b": GEMMA4_CHAT_DIALECT,
    "12b": GEMMA4_UNIFIED_CHAT_DIALECT,
}

#: The four files bundled into the provenance record (it records that the chat format and the EOS
#: set come from **the same digest set**).
SOURCE_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
)

#: The parity fixture (git-tracked — read by `gemma4_chat_test.ts` on the Deno side).
FIXTURE_PATH = (
    REPO_ROOT / "packages" / "models" / "tests" / "fixtures" / "gemma-text" / "gemma4-chat.json"
)

#: Model → its parity fixture (git-tracked; the TS chat tests read it). E4B is rendered in E2B's
#: dialect and has no fixture of its own; 12B's sits next to E2B's.
FIXTURE_PATHS: Mapping[str, Path] = {
    "e2b": FIXTURE_PATH,
    "12b": FIXTURE_PATH.with_name("gemma4-12b-chat.json"),
}

#: The scope of the first version (ADR 0084 decision 5). The TS side rejects a role not listed
#: here, failing loudly.
PLAIN_ROLES = ("system", "developer", "user", "assistant")

#: The fields a message may carry. It is an **allow list** so that an out-of-scope field such as
#: `tools` / `reasoning` / `tool_calls` does not slip through as "given but without effect".
ALLOWED_MESSAGE_KEYS = frozenset({"role", "content"})


class OutOfScopeChatError(ValueError):
    """Input outside the scope of the first version (plain conversations)."""


@dataclass(frozen=True)
class ChatCase:
    """A chat case (`why` states, in one line, the boundary where a naive port fails)."""

    name: str
    why: str
    messages: tuple[Mapping[str, Any], ...]


def assert_plain_conversation(messages: Sequence[Mapping[str, Any]]) -> None:
    """Checks that a conversation is plain (out-of-scope input is not harvested).

    MUST: reject where upstream rejects — `apply_chat_template` rejects an empty conversation with
    `ValueError`, so this rejects it too (the TS side's accepted set is not wider than upstream's).
    """
    if not messages:
        raise OutOfScopeChatError(
            "the conversation is empty (upstream apply_chat_template rejects it too)"
        )
    for index, message in enumerate(messages):
        extra = sorted(set(message) - ALLOWED_MESSAGE_KEYS)
        if extra:
            raise OutOfScopeChatError(
                f"messages[{index}]: out-of-scope fields {extra}"
                " (tools / reasoning / tool_calls and the like are rejected in the first version"
                " — ADR 0084 decision 5)"
            )
        role = message.get("role")
        if role not in PLAIN_ROLES:
            raise OutOfScopeChatError(
                f"messages[{index}].role {role!r} is none of {list(PLAIN_ROLES)}"
            )
        if not isinstance(message.get("content"), str):
            raise OutOfScopeChatError(
                f"messages[{index}].content is not a string"
                " (arrays of image / audio parts are rejected in the first version)"
            )


#: The plain-conversation battery. It takes **each branch and boundary once** (the same shape is
#: not harvested twice).
CASES: tuple[ChatCase, ...] = (
    ChatCase(
        "single-user",
        "the minimal form — `<bos>` + user turn + generation prompt",
        ({"role": "user", "content": "What is the capital of France?"},),
    ),
    ChatCase(
        "system-first",
        "a leading system falls into the system block, and its content is trimmed",
        (
            {"role": "system", "content": "  You are a terse assistant.  "},
            {"role": "user", "content": "Hi"},
        ),
    ),
    ChatCase(
        "developer-first",
        "a leading developer is **spelled system** too"
        " (the one position where the role is not emitted as is)",
        (
            {"role": "developer", "content": "Follow the house style."},
            {"role": "user", "content": "Hi"},
        ),
    ),
    ChatCase(
        "developer-late",
        "a later developer is `<|turn>developer` (asymmetric with the leading one)",
        (
            {"role": "user", "content": "Hi"},
            {"role": "developer", "content": "Be brief."},
        ),
    ),
    ChatCase(
        "round-trip",
        "a user → assistant → user round trip (assistant maps to `model`)",
        (
            {"role": "user", "content": "Who wrote Hamlet?"},
            {"role": "assistant", "content": "William Shakespeare."},
            {"role": "user", "content": "When?"},
        ),
    ),
    ChatCase(
        "assistant-last",
        "a trailing assistant still closes its turn, and the generation prompt opens a **new**"
        " model turn",
        (
            {"role": "user", "content": "Who wrote Hamlet?"},
            {"role": "assistant", "content": "William Shakespeare."},
        ),
    ),
    ChatCase(
        "assistant-twice",
        "consecutive assistant messages fold into one model turn"
        " (the turn is neither reopened nor closed)",
        (
            {"role": "user", "content": "Continue."},
            {"role": "assistant", "content": "First half."},
            {"role": "assistant", "content": "Second half."},
        ),
    ),
    ChatCase(
        "assistant-first",
        "a conversation that starts with assistant (no previous role, so it is not a continuation)",
        ({"role": "assistant", "content": "I already answered."},),
    ),
    ChatCase(
        "user-twice",
        "consecutive user messages do not fold (only the model turn checks for a continuation)",
        (
            {"role": "user", "content": "one"},
            {"role": "user", "content": "two"},
        ),
    ),
    ChatCase(
        "system-only",
        "a system-only conversation"
        " (the loop never runs and goes straight to the generation prompt)",
        ({"role": "system", "content": "You are a calculator."},),
    ),
    ChatCase(
        "empty-content",
        "empty content (the turn opens and closes at once — the empty string is not dropped)",
        ({"role": "user", "content": ""},),
    ),
    ChatCase(
        "whitespace-content",
        "trim is Python's `str.strip()` — it drops the full-width space U+3000, newlines and tabs"
        " too",
        ({"role": "user", "content": "\n\t hi 　\n"},),
    ),
    ChatCase(
        "assistant-whitespace",
        "the assistant side is trimmed too"
        " (`strip_thinking` equals a trim when no thinking spelling is present)",
        (
            {"role": "user", "content": "A?"},
            {"role": "assistant", "content": "   B.   "},
            {"role": "user", "content": "C?"},
        ),
    ),
    ChatCase(
        "japanese",
        "a Japanese multi-turn conversation"
        " (no spaces, so one pre-token spans the whole turn body)",
        (
            {"role": "system", "content": "あなたは簡潔に答える助手です。"},
            {"role": "user", "content": "日本の首都はどこですか？"},
            {"role": "assistant", "content": "東京です。"},
            {"role": "user", "content": "その人口は？"},
        ),
    ),
    ChatCase(
        "special-in-content",
        "chat spellings inside the body are **not escaped**"
        " (they are split out as added vocabulary)",
        ({"role": "user", "content": "abc<|turn>def"},),
    ),
    ChatCase(
        "multiline-content",
        "content with newlines (turns are delimited by `<turn|>`, not by newlines)",
        ({"role": "user", "content": "line one\nline two\n\nline four"},),
    ),
)

# ---------------------------------------------------------------------------
# tools, tool calls, tool results and thinking (ADR 0084 addendum 2026-11-04)
# ---------------------------------------------------------------------------

#: The keyword arguments a tool-conversation case passes to `apply_chat_template` (upstream's
#: spellings; the TS options object spells them `tools` / `enableThinking` / `preserveThinking`).
TEMPLATE_OPTION_KEYS = frozenset({"tools", "enable_thinking", "preserve_thinking"})

#: Role → the fields a message of that role may carry in the widened scope (an allow list, for the
#: same reason as {@link ALLOWED_MESSAGE_KEYS}: a field outside it would pass as "given but without
#: effect"). These are the fields the template reads for that role: `reasoning` /
#: `reasoning_content` / `tool_calls` on an assistant message, `tool_call_id` / `name` on a tool
#: result (it resolves the function name with them).
TOOL_ROLE_KEYS: Mapping[str, frozenset[str]] = {
    "system": frozenset({"role", "content"}),
    "developer": frozenset({"role", "content"}),
    "user": frozenset({"role", "content"}),
    "assistant": frozenset(
        {"role", "content", "reasoning", "reasoning_content", "tool_calls", "refusal"}
    ),
    "tool": frozenset({"role", "content", "tool_call_id", "name"}),
}


@dataclass(frozen=True)
class ToolChatCase:
    """A rendering case with template options (`why` names the branch it pins, in one line)."""

    name: str
    why: str
    messages: tuple[Mapping[str, Any], ...]
    #: `apply_chat_template` keyword arguments ({@link TEMPLATE_OPTION_KEYS}).
    options: Mapping[str, Any]


def assert_tool_conversation(
    messages: Sequence[Mapping[str, Any]], options: Mapping[str, Any]
) -> None:
    """Checks that a case is inside the widened scope (out-of-scope input is not harvested).

    MUST: reject the legacy `tool_responses` field and content-part arrays — both stay out of scope
    on the TS side, so a case using them would ship an expectation nothing may produce.
    """
    if not messages:
        raise OutOfScopeChatError(
            "the conversation is empty (upstream apply_chat_template rejects it)"
        )
    extra_options = sorted(set(options) - TEMPLATE_OPTION_KEYS)
    if extra_options:
        raise OutOfScopeChatError(f"unknown template options {extra_options}")
    for index, message in enumerate(messages):
        role = message.get("role")
        allowed = TOOL_ROLE_KEYS.get(role) if isinstance(role, str) else None
        if allowed is None:
            raise OutOfScopeChatError(
                f"messages[{index}].role {role!r} is none of {sorted(TOOL_ROLE_KEYS)}"
            )
        extra = sorted(set(message) - allowed)
        if extra:
            raise OutOfScopeChatError(
                f"messages[{index}]: out-of-scope fields {extra} for role {role!r}"
                " (`tool_responses` and content parts stay out of scope)"
            )
        content = message.get("content")
        if isinstance(content, list):
            raise OutOfScopeChatError(
                f"messages[{index}].content is an array (content parts stay out of scope)"
            )
        if role in ("system", "developer", "user") and not isinstance(content, str):
            raise OutOfScopeChatError(f"messages[{index}].content is not a string")


#: One function declaration per shape the template's `format_function_declaration` reads.
WEATHER_TOOL: Mapping[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city name"},
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit",
                },
            },
            "required": ["location"],
        },
    },
}
TIME_TOOL: Mapping[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Current local time.",
        "parameters": {"type": "object", "properties": {"timezone": {"type": "string"}}},
    },
}
SEARCH_TOOL: Mapping[str, Any] = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the web.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "description": "Max results"},
                "safe": {"type": "boolean"},
                "score": {"type": "number", "nullable": True},
            },
            "required": ["query"],
        },
        "response": {"type": "object", "description": "The results."},
    },
}
#: No `parameters` at all (the block is skipped) and an empty `properties` (falsy — only `type`).
PING_TOOL: Mapping[str, Any] = {
    "type": "function",
    "function": {"name": "ping", "description": "Liveness check."},
}
NOOP_TOOL: Mapping[str, Any] = {
    "type": "function",
    "function": {
        "name": "noop",
        "description": "",
        "parameters": {"type": "object", "properties": {}},
    },
}
#: Every branch of `format_parameters`: case-insensitive key order, arrays (string items, object
#: items with their own properties and required, a type list, nested arrays rendered through
#: `format_argument`, empty items), nested objects (with properties, or without — the
#: `filter_keys` branch), nullable, an upper-case type, a property without a type, and a key that a
#: JSON-ish reader would need to escape (rendered bare).
TRIP_TOOL: Mapping[str, Any] = {
    "type": "function",
    "function": {
        "name": "plan_trip",
        "description": "Plan a trip.",
        "parameters": {
            "type": "object",
            "properties": {
                "Zeta": {"type": "string"},
                "alpha": {"type": "integer"},
                "Beta": {"type": "boolean"},
                "stops": {
                    "type": "array",
                    "description": "Cities in order",
                    "items": {"type": "string"},
                },
                "legs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "from": {"type": "string"},
                            "to": {"type": "string"},
                            "days": {"type": "integer"},
                        },
                        "required": ["from", "to"],
                    },
                },
                "tags": {
                    "type": "array",
                    "items": {
                        "type": ["string", "null"],
                        "enum": ["a", "b"],
                        "description": "A tag",
                    },
                },
                "matrix": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                },
                "empty_items": {"type": "array", "items": {}},
                "budget": {
                    "type": "object",
                    "description": "Budget limits",
                    "properties": {
                        "max": {"type": "number"},
                        "currency": {"type": "string", "enum": ["EUR", "JPY"]},
                    },
                    "required": ["max"],
                },
                "extras": {"type": "object", "additionalProperties": {"type": "string"}},
                "flexible": {"type": "boolean", "nullable": True, "description": "Dates can move"},
                "max-results": {"type": "integer"},
                "kind": {"type": "STRING", "enum": ["x"]},
                "anything": {"description": "No type at all"},
            },
            "required": ["stops", "budget"],
        },
    },
}
#: OpenAI fields the template ignores and that mean nothing to the model: no `type` on the tool,
#: `strict` on the function (the TS side accepts and ignores them).
OPENAI_TOOL: Mapping[str, Any] = {
    "function": {
        "name": "lookup",
        "description": "Look a word up.",
        "parameters": {"type": "object", "properties": {"word": {"type": "string"}}},
        "strict": True,
    },
}


def _declaration(**function: Any) -> dict[str, Any]:
    """A one-off declaration for the rejected battery."""
    return {"type": "function", "function": {"name": "odd", "description": "Odd.", **function}}


def _call(name: str, arguments: Any, call_id: str | None = None) -> dict[str, Any]:
    """One OpenAI-shape tool call (arguments as an object — the template rejects a string)."""
    call: dict[str, Any] = {"type": "function", "function": {"name": name, "arguments": arguments}}
    if call_id is not None:
        call["id"] = call_id
    return call


_ASK_WEATHER: Mapping[str, Any] = {"role": "user", "content": "What is the weather in Paris?"}
_CALL_WEATHER: Mapping[str, Any] = {
    "role": "assistant",
    "tool_calls": [_call("get_weather", {"location": "Paris"})],
}
_WEATHER_RESULT: Mapping[str, Any] = {
    "role": "tool",
    "name": "get_weather",
    "content": "18 degrees and sunny",
}

#: The widened battery. Every case is harvested for E2B and 12B from each model's own template.
TOOL_CASES: tuple[ToolChatCase, ...] = (
    ToolChatCase(
        "tools-one",
        "tools alone open the system block (no system message); one declaration",
        (_ASK_WEATHER,),
        {"tools": [WEATHER_TOOL]},
    ),
    ToolChatCase(
        "tools-several-with-system",
        "system text first, then the declarations in list order; a response declaration, no "
        "parameters, empty properties and an empty description",
        (
            {"role": "system", "content": "  You are a helpful assistant.  "},
            {"role": "user", "content": "Hi"},
        ),
        {"tools": [WEATHER_TOOL, TIME_TOOL, SEARCH_TOOL, PING_TOOL, NOOP_TOOL]},
    ),
    ToolChatCase(
        "tools-schema-branches",
        "every branch of format_parameters",
        ({"role": "user", "content": "Plan something."},),
        {"tools": [TRIP_TOOL]},
    ),
    ToolChatCase(
        "tools-developer-first",
        "a leading developer message also lands in the system block before the declarations",
        ({"role": "developer", "content": "Prefer metric units."}, _ASK_WEATHER),
        {"tools": [TIME_TOOL]},
    ),
    ToolChatCase(
        "tool-call-then-result",
        "the result continues the open model turn: no <turn|>, no generation prompt (thinking off)",
        (_ASK_WEATHER, _CALL_WEATHER, _WEATHER_RESULT),
        {"tools": [WEATHER_TOOL]},
    ),
    ToolChatCase(
        "tool-call-then-answer",
        "the answer after the result folds into the same model turn, which then closes",
        (
            _ASK_WEATHER,
            _CALL_WEATHER,
            _WEATHER_RESULT,
            {"role": "assistant", "content": "It is 18 degrees and sunny in Paris."},
            {"role": "user", "content": "Thanks!"},
        ),
        {"tools": [WEATHER_TOOL]},
    ),
    ToolChatCase(
        "tool-calls-parallel-ids",
        "two calls with ids; results resolved by tool_call_id out of order; string and float "
        "results; content null",
        (
            _ASK_WEATHER,
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    _call("get_weather", {"location": "Paris", "unit": "celsius"}, "call_1"),
                    _call("get_time", {"timezone": "Europe/Paris"}, "call_2"),
                ],
            },
            {"role": "tool", "tool_call_id": "call_2", "content": "14:05 (CEST)"},
            {"role": "tool", "tool_call_id": "call_1", "content": 18.5},
        ),
        {"tools": [WEATHER_TOOL, TIME_TOOL]},
    ),
    ToolChatCase(
        "tool-call-arguments-nested",
        "argument values of every JSON kind, nested, keys sorted case-insensitively, strings with "
        "quotes / newlines / tabs / a chat spelling, numbers int / negative / fraction / exponent",
        (
            {"role": "user", "content": "Plan a trip."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    _call(
                        "plan_trip",
                        {
                            "stops": ["Paris", "Lyon"],
                            "legs": [{"from": "Paris", "to": "Lyon", "days": 2}],
                            "budget": {"max": 1200.5, "currency": "EUR"},
                            "flexible": False,
                            "Zeta": "",
                            "alpha": -3,
                            "notes": 'Say "hi"\nthen\tleave <|turn> it',
                            "matrix": [[0.25, 1e-05], []],
                            "extras": {},
                            "anything": None,
                        },
                    )
                ],
            },
            {"role": "tool", "name": "plan_trip", "content": "Booked.\nRef: A1"},
            {"role": "assistant", "content": "Your trip is booked."},
        ),
        {"tools": [TRIP_TOOL]},
    ),
    ToolChatCase(
        "tool-result-kinds",
        "result content of every kind (string with quotes, a JSON-looking string, integer, "
        "boolean, null, empty string); a result named by `name` alone when every call has an id",
        (
            {"role": "user", "content": "Check everything."},
            {
                "role": "assistant",
                "tool_calls": [
                    _call("search", {"query": "karume"}, "a"),
                    _call("get_time", {"timezone": "UTC"}, "b"),
                    _call("ping", None, "c"),
                ],
            },
            {"role": "tool", "tool_call_id": "a", "content": 'Found "karume".'},
            {"role": "tool", "tool_call_id": "b", "content": '{"utc": "09:05"}'},
            {"role": "tool", "tool_call_id": "c", "content": 42},
            {"role": "tool", "name": "ping", "content": False},
            {"role": "tool", "name": "search", "content": None},
            {"role": "tool", "name": "get_time", "content": ""},
        ),
        {"tools": [SEARCH_TOOL, TIME_TOOL, PING_TOOL]},
    ),
    ToolChatCase(
        "tool-call-single-without-ids",
        "one call and no ids: each result resolves to that call's name (with or without `name`)",
        (
            _ASK_WEATHER,
            _CALL_WEATHER,
            {"role": "tool", "content": "18 degrees"},
            {"role": "tool", "name": "get_weather", "content": "and sunny"},
        ),
        {"tools": [WEATHER_TOOL]},
    ),
    ToolChatCase(
        "tool-call-with-content",
        "content on a tool-calling message is written after the results, and closes the turn",
        (
            _ASK_WEATHER,
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [_call("get_weather", {"location": "Paris"})],
            },
            _WEATHER_RESULT,
            {"role": "user", "content": "And tomorrow?"},
        ),
        {"tools": [WEATHER_TOOL]},
    ),
    ToolChatCase(
        "tool-multi-step",
        "a second call after a result continues the same model turn; ends on the second result",
        (
            _ASK_WEATHER,
            _CALL_WEATHER,
            _WEATHER_RESULT,
            {"role": "assistant", "tool_calls": [_call("get_time", {"timezone": "Europe/Paris"})]},
            {"role": "tool", "name": "get_time", "content": "14:05"},
        ),
        {"tools": [WEATHER_TOOL, TIME_TOOL]},
    ),
    ToolChatCase(
        "tool-call-arguments-null",
        "arguments null renders an empty body (a string would raise)",
        (
            {"role": "user", "content": "Are you there?"},
            {"role": "assistant", "tool_calls": [_call("ping", None)]},
            {"role": "tool", "name": "ping", "content": "pong"},
            {"role": "assistant", "content": "Yes."},
        ),
        {"tools": [PING_TOOL]},
    ),
    ToolChatCase(
        "thinking-on-user-only",
        "enable_thinking alone opens the system block with <|think|>",
        ({"role": "user", "content": "What is 2+2?"},),
        {"enable_thinking": True},
    ),
    ToolChatCase(
        "thinking-on-system-only",
        "<|think|> comes before the system text; the generation prompt is the plain one",
        ({"role": "system", "content": "Be brief."},),
        {"enable_thinking": True},
    ),
    ToolChatCase(
        "thinking-on-system-tools",
        "<|think|>, then the system text, then the declarations",
        ({"role": "system", "content": "Be brief."}, _ASK_WEATHER),
        {"enable_thinking": True, "tools": [WEATHER_TOOL]},
    ),
    ToolChatCase(
        "thinking-on-after-tool-result",
        "after a result with thinking on, the generation prompt reopens the thought channel",
        (
            _ASK_WEATHER,
            {
                "role": "assistant",
                "reasoning": "I should call the weather tool.",
                "tool_calls": [_call("get_weather", {"location": "Paris"})],
            },
            _WEATHER_RESULT,
        ),
        {"enable_thinking": True, "tools": [WEATHER_TOOL]},
    ),
    ToolChatCase(
        "thinking-gate",
        "reasoning is kept only after the last user message (reasoning / reasoning_content)",
        (
            {"role": "user", "content": "First question?"},
            {"role": "assistant", "reasoning": "R1 dropped", "content": "A1"},
            {"role": "user", "content": "What is the weather in Paris?"},
            {
                "role": "assistant",
                "reasoning_content": "R2 kept",
                "tool_calls": [_call("get_weather", {"location": "Paris"})],
            },
            _WEATHER_RESULT,
            {"role": "assistant", "reasoning": "R3 kept", "content": "It is sunny."},
        ),
        {"enable_thinking": True, "tools": [WEATHER_TOOL]},
    ),
    ToolChatCase(
        "reasoning-precedence",
        "`reasoning or reasoning_content`: an empty reasoning falls through; the text is not "
        "trimmed; the gate ignores enable_thinking (off here)",
        (
            {"role": "user", "content": "Go."},
            {
                "role": "assistant",
                "reasoning": "",
                "reasoning_content": "  from reasoning_content \n",
                "content": "A",
            },
            {
                "role": "assistant",
                "reasoning": "from reasoning",
                "reasoning_content": "ignored",
                "content": "B",
            },
        ),
        {},
    ),
    ToolChatCase(
        "preserve-thinking",
        "preserve_thinking keeps the reasoning of a tool-calling message before the last user "
        "message; a message without tool calls still loses it",
        (
            _ASK_WEATHER,
            {
                "role": "assistant",
                "reasoning": "kept: it has tool calls",
                "tool_calls": [_call("get_weather", {"location": "Paris"})],
            },
            _WEATHER_RESULT,
            {"role": "assistant", "reasoning": "dropped: no tool calls", "content": "Sunny."},
            {"role": "user", "content": "Thanks."},
        ),
        {"enable_thinking": True, "preserve_thinking": True, "tools": [WEATHER_TOOL]},
    ),
    ToolChatCase(
        "preserve-thinking-off",
        "the same conversation without preserve_thinking: both reasonings are dropped",
        (
            _ASK_WEATHER,
            {
                "role": "assistant",
                "reasoning": "kept: it has tool calls",
                "tool_calls": [_call("get_weather", {"location": "Paris"})],
            },
            _WEATHER_RESULT,
            {"role": "assistant", "reasoning": "dropped: no tool calls", "content": "Sunny."},
            {"role": "user", "content": "Thanks."},
        ),
        {"enable_thinking": True, "tools": [WEATHER_TOOL]},
    ),
    ToolChatCase(
        "strip-thinking",
        "strip_thinking on assistant content: a closed channel, a stray <channel|>, an "
        "unterminated <|channel> — then the trim",
        (
            {"role": "user", "content": "One."},
            {"role": "assistant", "content": "<|channel>thought\nhidden<channel|>Visible answer."},
            {"role": "user", "content": "Two."},
            {
                "role": "assistant",
                "content": (
                    " Before<|channel>thought\nX<channel|> after <channel|>tail<|channel>cut "
                ),
            },
            {"role": "user", "content": "Three."},
        ),
        {},
    ),
    ToolChatCase(
        "assistant-content-null",
        "an assistant message with null content and no tool calls is an empty model turn",
        ({"role": "user", "content": "Say nothing."}, {"role": "assistant", "content": None}),
        {},
    ),
    ToolChatCase(
        "tool-call-japanese",
        "non-ASCII argument and result values (word characters in a key)",
        (
            {"role": "user", "content": "「こんにちは」を英訳して"},
            {"role": "assistant", "tool_calls": [_call("translate", {"原文": "こんにちは"})]},
            {"role": "tool", "name": "translate", "content": "Hello"},
            {"role": "assistant", "content": "Hello です。"},
        ),
        {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "translate",
                        "description": "日本語を英語へ",
                        "parameters": {
                            "type": "object",
                            "properties": {"原文": {"type": "string"}},
                            "required": ["原文"],
                        },
                    },
                }
            ]
        },
    ),
    ToolChatCase(
        "openai-shaped-fields",
        "fields the template ignores and that mean nothing to the model: no tool type, strict, a "
        "call index, refusal / tool_calls / reasoning null, a result without content, an empty "
        "result name next to a matching tool_call_id",
        (
            {"role": "user", "content": "Define 'karume'."},
            {"role": "assistant", "content": "Let me see.", "tool_calls": None, "refusal": None},
            {
                "role": "assistant",
                "content": None,
                "refusal": None,
                "reasoning": None,
                "reasoning_content": "look it up",
                "tool_calls": [
                    {**_call("lookup", {"word": "karume"}, "c1"), "index": 0},
                    {**_call("lookup", {"word": "light"}, "c2"), "index": 1},
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "name": ""},
            {"role": "tool", "tool_call_id": "c2", "content": "not heavy"},
        ),
        {"tools": [OPENAI_TOOL]},
    ),
)


@dataclass(frozen=True)
class RejectedCase:
    """An input the TS side rejects (`why` says why), harvested to record what upstream does."""

    name: str
    why: str
    messages: tuple[Mapping[str, Any], ...]
    options: Mapping[str, Any]


#: Inputs the TS renderer rejects with `ModelInputError`. Upstream's outcome (the rendered string,
#: or the error it raises) is recorded beside each, so the rejection rests on observed behaviour.
REJECTED_CASES: tuple[RejectedCase, ...] = (
    RejectedCase(
        "several-calls-without-ids",
        "with no ids the template matches every result to the LAST call (None == None), whatever "
        "its `name` says",
        (
            _ASK_WEATHER,
            {
                "role": "assistant",
                "tool_calls": [
                    _call("get_weather", {"location": "Paris"}),
                    _call("get_time", {"timezone": "Europe/Paris"}),
                ],
            },
            {"role": "tool", "name": "get_weather", "content": "18 degrees"},
            {"role": "tool", "name": "get_time", "content": "14:05"},
        ),
        {"tools": [WEATHER_TOOL, TIME_TOOL]},
    ),
    RejectedCase(
        "result-name-contradicts-call",
        "the single call without an id overrides the result's own `name`",
        (
            _ASK_WEATHER,
            _CALL_WEATHER,
            {"role": "tool", "name": "get_time", "content": "14:05"},
        ),
        {"tools": [WEATHER_TOOL, TIME_TOOL]},
    ),
    RejectedCase(
        "result-id-matches-no-call",
        "an unmatched tool_call_id falls back to `name`, then to the spelling 'unknown'",
        (
            _ASK_WEATHER,
            {
                "role": "assistant",
                "tool_calls": [_call("get_weather", {"location": "Paris"}, "call_1")],
            },
            {"role": "tool", "tool_call_id": "call_9", "content": "18 degrees"},
        ),
        {"tools": [WEATHER_TOOL]},
    ),
    RejectedCase(
        "orphan-tool-result",
        "a tool message that follows no tool-calling message is dropped without a trace",
        (_ASK_WEATHER, {"role": "tool", "name": "get_weather", "content": "18 degrees"}),
        {"tools": [WEATHER_TOOL]},
    ),
    RejectedCase(
        "tool-calls-without-results",
        "a tool call not followed by a result leaves a bare <|tool_response> and an unclosed turn",
        (_ASK_WEATHER, _CALL_WEATHER, {"role": "user", "content": "Never mind."}),
        {"tools": [WEATHER_TOOL]},
    ),
    RejectedCase(
        "tool-calls-awaiting-results",
        "a conversation ending on a tool call has no generation prompt (the model would write "
        "the result itself)",
        (_ASK_WEATHER, _CALL_WEATHER),
        {"tools": [WEATHER_TOOL]},
    ),
    RejectedCase(
        "tool-call-content-at-end",
        "content on a tool-calling message whose results end the conversation: the template "
        "writes it after the results and closes the turn, leaving no generation prompt",
        (
            _ASK_WEATHER,
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [_call("get_weather", {"location": "Paris"})],
            },
            _WEATHER_RESULT,
        ),
        {"tools": [WEATHER_TOOL]},
    ),
    RejectedCase(
        "parameters-without-type",
        "parameters without a top-level type: the template never closes 'parameters:{'",
        (_ASK_WEATHER,),
        {"tools": [_declaration(parameters={"properties": {"x": {"type": "string"}}})]},
    ),
    RejectedCase(
        "response-not-an-object",
        "a response whose type is not object: the template never closes 'response:{'",
        (_ASK_WEATHER,),
        {"tools": [_declaration(response={"type": "string", "description": "Text"})]},
    ),
    RejectedCase(
        "response-null",
        "a null response leaves 'response:{' open too",
        (_ASK_WEATHER,),
        {"tools": [_declaration(response=None)]},
    ),
    RejectedCase(
        "description-null",
        "a null function description is printed as Python's None",
        (_ASK_WEATHER,),
        {"tools": [_declaration(description=None)]},
    ),
    RejectedCase(
        "function-return",
        "transformers' `return` schema is not rendered (the template reads `response`)",
        (_ASK_WEATHER,),
        {"tools": [_declaration(**{"return": {"type": "string"}})]},
    ),
    RejectedCase(
        "speaker-name",
        "a speaker name on a user message is not rendered",
        ({"role": "user", "content": "hi", "name": "bob"},),
        {},
    ),
    RejectedCase(
        "refusal-text",
        "a refusal text on an assistant message is not rendered",
        (
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "refusal": "I cannot."},
            {"role": "user", "content": "ok"},
        ),
        {},
    ),
    RejectedCase(
        "argument-string-with-delimiter",
        'a string holding <|"|> renders like two arguments (parse_response reads it back so)',
        (
            _ASK_WEATHER,
            {
                "role": "assistant",
                "tool_calls": [_call("get_weather", {"location": 'a<|"|>,b:<|"|>c'})],
            },
            _WEATHER_RESULT,
        ),
        {"tools": [WEATHER_TOOL]},
    ),
    RejectedCase(
        "argument-key-not-a-word",
        "argument keys are written bare; one outside \\w+ cannot be read back",
        (
            _ASK_WEATHER,
            {"role": "assistant", "tool_calls": [_call("get_weather", {"my-key": 1})]},
            _WEATHER_RESULT,
        ),
        {"tools": [WEATHER_TOOL]},
    ),
    RejectedCase(
        "result-name-of-no-call",
        "with every call holding an id, a result's `name` is written even if no call has it",
        (
            _ASK_WEATHER,
            {
                "role": "assistant",
                "tool_calls": [_call("get_weather", {"location": "Paris"}, "call_1")],
            },
            {"role": "tool", "name": "get_time", "content": "14:05"},
        ),
        {"tools": [WEATHER_TOOL, TIME_TOOL]},
    ),
    RejectedCase(
        "empty-result-name",
        "an empty name is no name (`name or 'unknown'`): with every call holding an id, the "
        "template writes 'unknown'",
        (
            _ASK_WEATHER,
            {
                "role": "assistant",
                "tool_calls": [_call("get_weather", {"location": "Paris"}, "call_1")],
            },
            {"role": "tool", "name": "", "content": "18 degrees"},
        ),
        {"tools": [WEATHER_TOOL]},
    ),
    RejectedCase(
        "arguments-as-string",
        "OpenAI's JSON-string arguments are refused by the template itself",
        (
            _ASK_WEATHER,
            {
                "role": "assistant",
                "tool_calls": [_call("get_weather", '{"location": "Paris"}')],
            },
            _WEATHER_RESULT,
        ),
        {"tools": [WEATHER_TOOL]},
    ),
    RejectedCase(
        "legacy-tool-responses",
        "the legacy `tool_responses` field stays out of scope",
        (
            _ASK_WEATHER,
            {
                "role": "assistant",
                "tool_calls": [_call("get_weather", {"location": "Paris"})],
                "tool_responses": [{"name": "get_weather", "response": {"temp": 18}}],
            },
        ),
        {"tools": [WEATHER_TOOL]},
    ),
    RejectedCase(
        "content-parts",
        "chatPrompt takes no content-part array, text parts included: only the image-capable "
        "render does (imageChat)",
        ({"role": "user", "content": [{"type": "text", "text": "hi"}]},),
        {},
    ),
    RejectedCase(
        "tool-result-object",
        "an object result is read as a content-part array (Jinja's `sequence` test is true for a "
        "dict) and raises — serialize it to a string",
        (
            _ASK_WEATHER,
            _CALL_WEATHER,
            {"role": "tool", "name": "get_weather", "content": {"temp": 18}},
        ),
        {"tools": [WEATHER_TOOL]},
    ),
    RejectedCase(
        "tool-name-not-a-word",
        "a declared name outside \\w+ renders, but the output parser reads a call's name with "
        "\\w+, so the tool could never be called back",
        (_ASK_WEATHER,),
        {
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "get-weather", "description": "Hyphenated."},
                }
            ]
        },
    ),
    RejectedCase(
        "thinking-field",
        "`thinking` (parse_response's output key) is not read by the template — passing a parsed "
        "message back as is would drop the thought silently",
        (_ASK_WEATHER, {"role": "assistant", "thinking": "hmm", "content": "Sunny."}),
        {},
    ),
)


@dataclass(frozen=True)
class ResponseCase:
    """One model output and the conversation whose generation prompt it follows."""

    name: str
    why: str
    #: The conversation rendered (by upstream, with `add_generation_prompt`) into the `prefix`.
    messages: tuple[Mapping[str, Any], ...]
    options: Mapping[str, Any]
    #: The generated text, special spellings included, ending with its stop token.
    text: str
    #: `parse` — TS returns upstream's result; `upstream-raises` — upstream raises and TS throws;
    #: `ts-rejects` — upstream returns a value but TS throws (the reason is in `why`);
    #: `blank-segment` — TS returns upstream's result for {@link normalized} (the text without a
    #: whitespace-only content segment or an empty thought channel, which upstream lets erase the
    #: earlier text).
    expect: str
    #: `blank-segment` only: the same output with the blank segment removed.
    normalized: str | None = None
    #: A literal prompt in place of the rendered conversation (a caller error case).
    raw_prefix: str | None = None


#: The `expect` labels of {@link ResponseCase}.
RESPONSE_EXPECTATIONS = frozenset({"parse", "upstream-raises", "ts-rejects", "blank-segment"})

_WEATHER_PROMPT: tuple[Mapping[str, Any], ...] = (_ASK_WEATHER,)
_AFTER_RESULT: tuple[Mapping[str, Any], ...] = (_ASK_WEATHER, _CALL_WEATHER, _WEATHER_RESULT)
_TOOLS: Mapping[str, Any] = {"tools": [WEATHER_TOOL, TIME_TOOL]}
_THINKING_TOOLS: Mapping[str, Any] = {"tools": [WEATHER_TOOL, TIME_TOOL], "enable_thinking": True}
_Q = '<|"|>'

#: The output-parser battery (`tokenizer.parse_response` with the snapshot's `response_template`).
RESPONSE_CASES: tuple[ResponseCase, ...] = (
    ResponseCase(
        "content-turn",
        "content closed by <turn|>",
        _WEATHER_PROMPT,
        _TOOLS,
        "It is sunny.<turn|>",
        "parse",
    ),
    ResponseCase(
        "content-eos", "content closed by <eos>", _WEATHER_PROMPT, _TOOLS, "Sunny.<eos>", "parse"
    ),
    ResponseCase(
        "content-tool-response",
        "content closed by <|tool_response> without a call",
        _WEATHER_PROMPT,
        _TOOLS,
        "Sunny.<|tool_response>",
        "parse",
    ),
    ResponseCase(
        "content-stripped",
        "content is stripped (Python str.strip)",
        _WEATHER_PROMPT,
        _TOOLS,
        "\n  Sunny.\u3000\n<turn|>",
        "parse",
    ),
    ResponseCase(
        "empty-turn",
        "only the stop token: no content key",
        _WEATHER_PROMPT,
        _TOOLS,
        "<turn|>",
        "parse",
    ),
    ResponseCase(
        "one-call",
        "one tool call, closed by <|tool_response>",
        _WEATHER_PROMPT,
        _TOOLS,
        f"<|tool_call>call:get_weather{{location:{_Q}Paris{_Q}}}<tool_call|><|tool_response>",
        "parse",
    ),
    ResponseCase(
        "several-calls",
        "calls back to back accumulate in order",
        _WEATHER_PROMPT,
        _TOOLS,
        f"<|tool_call>call:get_weather{{location:{_Q}Paris{_Q},unit:{_Q}celsius{_Q}}}<tool_call|>"
        f"<|tool_call>call:get_time{{timezone:{_Q}Europe/Paris{_Q}}}<tool_call|><|tool_response>",
        "parse",
    ),
    ResponseCase(
        "thinking-content",
        "a thought channel then content (both stripped)",
        _WEATHER_PROMPT,
        _THINKING_TOOLS,
        "<|channel>thought\nThe user wants the weather.\n<channel|>It is sunny.<turn|>",
        "parse",
    ),
    ResponseCase(
        "thinking-call",
        "a thought channel then a call",
        _WEATHER_PROMPT,
        _THINKING_TOOLS,
        f"<|channel>thought\nI need the tool.<channel|><|tool_call>call:get_weather{{location:"
        f"{_Q}Paris{_Q}}}<tool_call|><|tool_response>",
        "parse",
    ),
    ResponseCase(
        "content-then-call",
        "content written before a call is kept",
        _WEATHER_PROMPT,
        _TOOLS,
        f"Let me check.<|tool_call>call:get_weather{{location:{_Q}Paris{_Q}}}<tool_call|>"
        "<|tool_response>",
        "parse",
    ),
    ResponseCase(
        "empty-thought",
        "an empty thought channel leaves no thinking key",
        _WEATHER_PROMPT,
        _TOOLS,
        "<|channel>thought\n<channel|>Sunny.<turn|>",
        "parse",
    ),
    ResponseCase(
        "nested-arguments",
        "objects, arrays, booleans, null, fractions, negative numbers and exponents",
        _WEATHER_PROMPT,
        _TOOLS,
        f"<|tool_call>call:plan_trip{{budget:{{currency:{_Q}EUR{_Q},max:1200.5}},flexible:false,"
        f"legs:[{{days:2,from:{_Q}Paris{_Q},to:{_Q}Lyon{_Q}}}],notes:null,"
        f"stops:[{_Q}Paris{_Q},{_Q}Lyon{_Q}],values:[1,-2.5,1e-05,3]}}<tool_call|><|tool_response>",
        "parse",
    ),
    ResponseCase(
        "string-special-characters",
        "string values keep quotes, newlines, braces, commas, colons and backslashes",
        _WEATHER_PROMPT,
        _TOOLS,
        f'<|tool_call>call:note{{text:{_Q}He said "hi"\nthen left: {{a,b}}, done{_Q},'
        f"path:{_Q}C:\\dir\\x{_Q}}}<tool_call|><|tool_response>",
        "parse",
    ),
    ResponseCase(
        "empty-arguments",
        "an empty object body",
        _WEATHER_PROMPT,
        _TOOLS,
        "<|tool_call>call:ping{}<tool_call|><|tool_response>",
        "parse",
    ),
    ResponseCase(
        "unicode-name-and-key",
        "\\w is Unicode: a non-ASCII function name and key",
        _WEATHER_PROMPT,
        _TOOLS,
        f"<|tool_call>call:翻訳{{原文:{_Q}こんにちは{_Q}}}<tool_call|><|tool_response>",
        "parse",
    ),
    ResponseCase(
        "call-closed-by-turn",
        "a call followed by <turn|> instead of <|tool_response> still parses",
        _WEATHER_PROMPT,
        _TOOLS,
        "<|tool_call>call:ping{}<tool_call|><turn|>",
        "parse",
    ),
    ResponseCase(
        "after-result-content",
        "after a result the anchor is <tool_response|>",
        _AFTER_RESULT,
        _TOOLS,
        "It is 18 degrees and sunny.<turn|>",
        "parse",
    ),
    ResponseCase(
        "after-result-thinking",
        "after a result with thinking on, the prompt opened the channel the output closes",
        _AFTER_RESULT,
        _THINKING_TOOLS,
        "Got the weather.\n<channel|>It is 18 degrees.<turn|>",
        "parse",
    ),
    ResponseCase(
        "after-result-second-call",
        "a second call after a result",
        _AFTER_RESULT,
        _TOOLS,
        f"<|tool_call>call:get_time{{timezone:{_Q}Europe/Paris{_Q}}}<tool_call|><|tool_response>",
        "parse",
    ),
    ResponseCase(
        "unterminated-string",
        "a call cut inside a string: the JSON does not parse",
        _WEATHER_PROMPT,
        _TOOLS,
        f"<|tool_call>call:get_weather{{location:{_Q}Par",
        "upstream-raises",
    ),
    ResponseCase(
        "bare-string-value",
        "an undelimited string value is not JSON",
        _WEATHER_PROMPT,
        _TOOLS,
        "<|tool_call>call:get_weather{location:Paris}<tool_call|><|tool_response>",
        "upstream-raises",
    ),
    ResponseCase(
        "stop-token-inside-call",
        "<|tool_response> before <tool_call|> lands in the JSON body",
        _WEATHER_PROMPT,
        _TOOLS,
        "<|tool_call>call:ping{}<|tool_response>",
        "upstream-raises",
    ),
    ResponseCase(
        "empty-call-body",
        "no body at all",
        _WEATHER_PROMPT,
        _TOOLS,
        "<|tool_call>call:ping<tool_call|><|tool_response>",
        "upstream-raises",
    ),
    ResponseCase(
        "space-before-key",
        "only a key right after { or , is quoted",
        _WEATHER_PROMPT,
        _TOOLS,
        "<|tool_call>call:ping{ a:1}<tool_call|><|tool_response>",
        "upstream-raises",
    ),
    ResponseCase(
        "sentinel-character",
        "the reserved \\x01 / \\x02 characters are refused",
        _WEATHER_PROMPT,
        _TOOLS,
        f"<|tool_call>call:ping{{a:{_Q}x\x01y{_Q}}}<tool_call|><|tool_response>",
        "upstream-raises",
    ),
    ResponseCase(
        "hyphenated-name",
        "the name stops at the first non-word character, the rest breaks the JSON",
        _WEATHER_PROMPT,
        _TOOLS,
        f"<|tool_call>call:get-weather{{location:{_Q}Paris{_Q}}}<tool_call|><|tool_response>",
        "upstream-raises",
    ),
    ResponseCase(
        "unterminated-call",
        "TS rejects: no <tool_call|> and no stop token — upstream returns the call as if complete",
        _WEATHER_PROMPT,
        _TOOLS,
        f"<|tool_call>call:get_weather{{location:{_Q}Paris{_Q}}}",
        "ts-rejects",
    ),
    ResponseCase(
        "unterminated-thought",
        "TS rejects: the thought channel never closes — upstream returns a partial thought",
        _WEATHER_PROMPT,
        _THINKING_TOOLS,
        "<|channel>thought\nStill thinking",
        "ts-rejects",
    ),
    ResponseCase(
        "stop-token-inside-thought",
        "TS rejects: the stop token arrives inside the thought — upstream keeps it in the thought",
        _WEATHER_PROMPT,
        _THINKING_TOOLS,
        "<|channel>thought\nHmm<turn|>",
        "ts-rejects",
    ),
    ResponseCase(
        "no-stop-token",
        "TS rejects: the turn is not closed (cut by max tokens) — upstream returns the content",
        _WEATHER_PROMPT,
        _TOOLS,
        "It is sun",
        "ts-rejects",
    ),
    ResponseCase(
        "text-after-stop-token",
        "TS rejects: text after the stop token — upstream keeps only the last segment",
        _WEATHER_PROMPT,
        _TOOLS,
        "First.<turn|>Second.",
        "ts-rejects",
    ),
    ResponseCase(
        "content-overwritten",
        "TS rejects: content in two segments — upstream keeps only the last",
        _WEATHER_PROMPT,
        _THINKING_TOOLS,
        "First part.<|channel>thought\nT<channel|>Second part.<turn|>",
        "ts-rejects",
    ),
    ResponseCase(
        "thought-overwritten",
        "TS rejects: two thought channels — upstream keeps only the last",
        _WEATHER_PROMPT,
        _THINKING_TOOLS,
        "<|channel>thought\nA<channel|><|channel>thought\nB<channel|>C<turn|>",
        "ts-rejects",
    ),
    ResponseCase(
        "arguments-not-an-object",
        "TS rejects: a list body — upstream returns it; the template cannot render it back",
        _WEATHER_PROMPT,
        _TOOLS,
        "<|tool_call>call:ping[1,2]<tool_call|><|tool_response>",
        "ts-rejects",
    ),
    ResponseCase(
        "non-finite-number",
        "TS rejects: NaN is not JSON — Python's json.loads accepts it",
        _WEATHER_PROMPT,
        _TOOLS,
        "<|tool_call>call:ping{a:NaN}<tool_call|><|tool_response>",
        "ts-rejects",
    ),
    ResponseCase(
        "call-without-name",
        "TS rejects: an opener without call:NAME — upstream returns the spelling as content",
        _WEATHER_PROMPT,
        _TOOLS,
        f"<|tool_call>{{location:{_Q}Paris{_Q}}}<tool_call|><|tool_response>",
        "ts-rejects",
    ),
    ResponseCase(
        "float-forms",
        "1e5 / -0.0 / 2.0 are Python floats; in JS they are the numbers 100000 / -0 / 2",
        _WEATHER_PROMPT,
        _TOOLS,
        "<|tool_call>call:ping{a:1e5,b:-0.0,c:2.0,d:0.5}<tool_call|><|tool_response>",
        "parse",
    ),
    ResponseCase(
        "integer-like-keys",
        "integer-like keys (bare or quoted) — equal as objects; JS lists them first",
        _WEATHER_PROMPT,
        _TOOLS,
        f"<|tool_call>call:ping{{b:1,2:2,a:3,{_Q}10{_Q}:4}}<tool_call|><|tool_response>",
        "parse",
    ),
    ResponseCase(
        "call-inside-thought",
        "a call spelled inside the thought channel is thought text, not a call (upstream too)",
        _WEATHER_PROMPT,
        _THINKING_TOOLS,
        "<|channel>thought\nI will <|tool_call>call:ping{}<tool_call|><channel|>ok<turn|>",
        "parse",
    ),
    ResponseCase(
        "blank-after-call",
        "a newline between a call and the stop token: upstream lets it erase the content before",
        _WEATHER_PROMPT,
        _TOOLS,
        "pre<|tool_call>call:ping{}<tool_call|>\n<|tool_response>",
        "blank-segment",
        normalized="pre<|tool_call>call:ping{}<tool_call|><|tool_response>",
    ),
    ResponseCase(
        "blank-second-thought",
        "an empty second thought channel: upstream lets it erase the first",
        _WEATHER_PROMPT,
        _THINKING_TOOLS,
        "<|channel>thought\nA<channel|><|channel>thought\n<channel|>B<turn|>",
        "blank-segment",
        normalized="<|channel>thought\nA<channel|>B<turn|>",
    ),
    ResponseCase(
        "integer-beyond-2-53",
        "TS rejects: 2^53 + 1 — upstream keeps the exact int, JS would round it",
        _WEATHER_PROMPT,
        _TOOLS,
        "<|tool_call>call:ping{n:9007199254740993}<tool_call|><|tool_response>",
        "ts-rejects",
    ),
    ResponseCase(
        "integer-of-400-digits",
        "TS rejects: a 400-digit integer — upstream keeps the int, JS reads Infinity",
        _WEATHER_PROMPT,
        _TOOLS,
        "<|tool_call>call:ping{n:" + "9" * 400 + "}<tool_call|><|tool_response>",
        "ts-rejects",
    ),
    ResponseCase(
        "float-beyond-double",
        "TS rejects: 1e309 — upstream reads inf, JS Infinity",
        _WEATHER_PROMPT,
        _TOOLS,
        "<|tool_call>call:ping{n:1e309}<tool_call|><|tool_response>",
        "ts-rejects",
    ),
    ResponseCase(
        "quoted-key-not-a-word",
        "TS rejects: a quoted key outside \\w+ — the template would write it bare, unreadable",
        _WEATHER_PROMPT,
        _TOOLS,
        f"<|tool_call>call:ping{{{_Q}my-key{_Q}:1}}<tool_call|><|tool_response>",
        "ts-rejects",
    ),
    ResponseCase(
        "next-turn-in-content",
        "TS rejects: a hallucinated <|turn> in the content (a re-render would write a fake turn)",
        _WEATHER_PROMPT,
        _TOOLS,
        "answer<|turn>user\nhi<turn|>",
        "ts-rejects",
    ),
    ResponseCase(
        "result-closer-in-content",
        "TS rejects: a stray <tool_response|> in the content",
        _WEATHER_PROMPT,
        _TOOLS,
        "answer<tool_response|><turn|>",
        "ts-rejects",
    ),
    ResponseCase(
        "delimiter-in-content",
        'TS rejects: a string delimiter <|"|> in the content',
        _WEATHER_PROMPT,
        _TOOLS,
        f"say {_Q}x{_Q}<turn|>",
        "ts-rejects",
    ),
    ResponseCase(
        "prompt-without-anchor",
        "TS rejects: a prompt with neither start anchor — upstream reads it all as the answer",
        _WEATHER_PROMPT,
        _TOOLS,
        "<|tool_call>call:ping{}<tool_call|><|tool_response>",
        "ts-rejects",
        raw_prefix="<bos><|turn>user\nHello",
    ),
)


def _image_part(width: int, height: int) -> dict[str, Any]:
    """An image part as a case states it: the size is all a render reads."""
    return {"type": "image", "width": width, "height": height}


def _text_part(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


@dataclass(frozen=True)
class ImageChatCase:
    """A conversation with image parts (`why` names what it pins, in one line)."""

    name: str
    why: str
    messages: tuple[Mapping[str, Any], ...]


#: The image battery: the boundaries of the expansion. The sizes span the rule for N without
#: repeating it: a square, the corpus's aspect, a tall one, and one far below the budget's area.
IMAGE_CASES: tuple[ImageChatCase, ...] = (
    ImageChatCase(
        "image-only",
        "the expansion alone between the turn's opening and closing",
        ({"role": "user", "content": [_image_part(832, 480)]},),
    ),
    ImageChatCase(
        "image-then-text",
        "the shape of the goldens: the text follows <image|> directly, with no separator",
        (
            {
                "role": "user",
                "content": [_image_part(832, 480), _text_part("Describe this image.")],
            },
        ),
    ),
    ImageChatCase(
        "text-then-image",
        "the text before <|image> keeps its tokens",
        (
            {
                "role": "user",
                "content": [_text_part("What is in this picture?"), _image_part(640, 640)],
            },
        ),
    ),
    ImageChatCase(
        "text-around-two-images",
        "two images with text before, between and after, each part trimmed on its own",
        (
            {
                "role": "user",
                "content": [
                    _text_part("  Compare the first  "),
                    _image_part(100, 100),
                    _text_part(" with the second: "),
                    _image_part(300, 900),
                    _text_part("\nWhich one is brighter?\n"),
                ],
            },
        ),
    ),
    ImageChatCase(
        "two-images-adjacent",
        "<image|> then <|image> with nothing between them",
        ({"role": "user", "content": [_image_part(64, 64), _image_part(2000, 500)]},),
    ),
    ImageChatCase(
        "text-parts-only",
        "text parts without an image render as their concatenation",
        ({"role": "user", "content": [_text_part("one"), _text_part(" two ")]},),
    ),
    ImageChatCase(
        "images-in-a-conversation",
        "a system block and a past exchange around messages with images",
        (
            {"role": "system", "content": "You describe images."},
            {"role": "user", "content": [_image_part(832, 480), _text_part("And this?")]},
            {"role": "assistant", "content": "A cat."},
            {"role": "user", "content": [_text_part("Now this one:"), _image_part(480, 832)]},
        ),
    ),
    ImageChatCase(
        "extreme-aspect",
        "a side that rounds to 0: the processor's own rule for it",
        ({"role": "user", "content": [_image_part(4000, 10), _text_part("Read it.")]},),
    ),
)

#: What the TS side refuses in a conversation with images. Upstream's processor counts `<|image|>`
#: against the images passed; the markers it does not count are recorded the same way.
REJECTED_IMAGE_CASES: tuple[ImageChatCase, ...] = (
    ImageChatCase(
        "text-part-soft-token",
        "a text part spelling <|image|>: one marker more than images",
        (
            {
                "role": "user",
                "content": [_text_part("look <|image|> here"), _image_part(64, 64)],
            },
        ),
    ),
    ImageChatCase(
        "text-part-begin-marker",
        "a text part spelling <|image>: a block that opens with no image behind it",
        ({"role": "user", "content": [_text_part("a <|image> b"), _image_part(64, 64)]},),
    ),
    ImageChatCase(
        "text-part-end-marker",
        "a text part spelling <image|>: a block that closes with no image in front of it",
        ({"role": "user", "content": [_text_part("a <image|> b"), _image_part(64, 64)]},),
    ),
    ImageChatCase(
        "string-message-soft-token",
        "a string message spelling <|image|> next to a message with an image",
        (
            {"role": "user", "content": "see <|image|>"},
            {"role": "user", "content": [_image_part(64, 64)]},
        ),
    ),
)


def _with_pixels(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The case's messages as upstream's processor takes them: each image part holds a PIL image."""
    from PIL import Image  # lazy import (needed only when harvesting)

    def part(item: Mapping[str, Any]) -> dict[str, Any]:
        if item["type"] != "image":
            return dict(item)
        return {"type": "image", "image": Image.new("RGB", (item["width"], item["height"]))}

    return [
        {
            **message,
            "content": (
                [part(item) for item in message["content"]]
                if isinstance(message["content"], (list, tuple))
                else message["content"]
            ),
        }
        for message in messages
    ]


def harvest_image_cases(processor: Any, cases: Sequence[ImageChatCase]) -> list[dict[str, Any]]:
    """The expectations of a conversation with images, from upstream's processor (ADR 0084 decision
    7: not one line goes through an in-house renderer)."""
    rows: list[dict[str, Any]] = []
    for case in cases:
        messages = _with_pixels(case.messages)
        rendered = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True
        )
        rows.append(
            {
                "name": case.name,
                "why": case.why,
                "messages": [dict(message) for message in case.messages],
                "rendered": rendered,
                "ids": [int(token) for token in inputs["input_ids"][0]],
            }
        )
    return rows


def harvest_rejected_image_cases(
    processor: Any, cases: Sequence[ImageChatCase]
) -> list[dict[str, Any]]:
    """What upstream's processor does with each input the TS side refuses: `ids` (it accepts it) or
    `error` (it raises). The outcome itself is the record."""
    rows: list[dict[str, Any]] = []
    for case in cases:
        row: dict[str, Any] = {
            "name": case.name,
            "why": case.why,
            "messages": [dict(message) for message in case.messages],
        }
        try:
            inputs = processor.apply_chat_template(
                _with_pixels(case.messages),
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
            )
            row["ids"] = [int(token) for token in inputs["input_ids"][0]]
        except ValueError as error:
            row["error"] = _upstream_error(error)
        rows.append(row)
    return rows


def vision_tokens(processor: Any) -> dict[str, int]:
    """The three ids and the budget, as upstream's processor states them (what `pipelineConfig.vision`
    declares for the TS side)."""
    tokenizer = processor.tokenizer
    return {
        "softTokenBudget": int(processor.image_processor.max_soft_tokens),
        "imageTokenId": int(tokenizer.convert_tokens_to_ids(processor.image_token)),
        "beginImageTokenId": int(tokenizer.convert_tokens_to_ids(processor.boi_token)),
        "endImageTokenId": int(tokenizer.convert_tokens_to_ids(processor.eoi_token)),
    }


def _source(checkpoint_dir: Path, label: str) -> dict[str, Any]:
    """The provenance record (the digests of the four files that bundle the chat format and the EOS
    set — ADR 0084 decision 5).

    `label` names the checkpoint the way a git-tracked record may (`gemma4.sources.tokenizer_label`
    — a repository path, or `<repo>@<revision>` for a hub snapshot outside the repository).
    """
    return {
        "dir": label,
        "files": {name: file_digest(checkpoint_dir / name) for name in SOURCE_FILES},
    }


def stop_tokens(checkpoint_dir: Path) -> list[int]:
    """`eos_token_id` of `generation_config.json` (read as a set, even when declared singly)."""
    raw = json.loads((checkpoint_dir / "generation_config.json").read_text(encoding="utf-8"))
    declared = raw.get("eos_token_id")
    ids = [declared] if isinstance(declared, int) else declared
    if not isinstance(ids, list) or not ids or not all(isinstance(one, int) for one in ids):
        raise ValueError(
            f"eos_token_id of generation_config.json is not an integer / integer list: {declared!r}"
        )
    return list(ids)


def harvest_cases(tokenizer: Any, cases: Sequence[ChatCase]) -> list[dict[str, Any]]:
    """Harvests the expectations from upstream `apply_chat_template` (not one line goes through an
    in-house renderer).

    MUST: harvest **both** the rendered string and the id sequence. With only one of them, a
    mismatch does not tell the reader whether rendering or encoding broke (the TS side is built in
    the same two steps).
    MUST: check here that `ids` equals the rendered result encoded with `add_special_tokens=False`
    — it is the very premise the TS implementation (render → `encode`) stands on, so when it
    breaks, the harvesting side fails.
    """
    rows: list[dict[str, Any]] = []
    for case in cases:
        assert_plain_conversation(case.messages)
        messages = [dict(message) for message in case.messages]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        ids = list(
            tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)[
                "input_ids"
            ]
        )
        direct = list(tokenizer(rendered, add_special_tokens=False)["input_ids"])
        if ids != direct:
            raise AssertionError(
                f"{case.name}: apply_chat_template(tokenize=True) does not match"
                " rendering then encoding with add_special_tokens=False"
                " (the premise of the TS two-step build is broken)"
            )
        rows.append(
            {
                "name": case.name,
                "why": case.why,
                "messages": messages,
                "rendered": rendered,
                "ids": ids,
            }
        )
    return rows


def _upstream_error(error: Exception) -> str:
    """How a fixture records an upstream exception: its type and the first line of its message."""
    return f"{type(error).__name__}: {str(error).splitlines()[0] if str(error) else ''}"


def harvest_tool_cases(tokenizer: Any, cases: Sequence[ToolChatCase]) -> list[dict[str, Any]]:
    """The widened battery's expectations from upstream `apply_chat_template` (same two halves and
    the same render → encode check as {@link harvest_cases}; the options ride as keyword
    arguments)."""
    rows: list[dict[str, Any]] = []
    for case in cases:
        assert_tool_conversation(case.messages, case.options)
        messages = [dict(message) for message in case.messages]
        options = dict(case.options)
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **options
        )
        ids = list(
            tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, **options
            )["input_ids"]
        )
        direct = list(tokenizer(rendered, add_special_tokens=False)["input_ids"])
        if ids != direct:
            raise AssertionError(
                f"{case.name}: apply_chat_template(tokenize=True) differs from rendering then"
                " encoding with add_special_tokens=False (the TS two-step premise broke)"
            )
        rows.append(
            {
                "name": case.name,
                "why": case.why,
                "messages": messages,
                "options": options,
                "rendered": rendered,
                "ids": ids,
            }
        )
    return rows


def harvest_rejected_cases(tokenizer: Any, cases: Sequence[RejectedCase]) -> list[dict[str, Any]]:
    """What upstream does with each input the TS side rejects: `rendered` (the template accepts
    it) or `error` (it raises). No gate runs here — these cases are outside the scope on purpose."""
    rows: list[dict[str, Any]] = []
    for case in cases:
        messages = [dict(message) for message in case.messages]
        options = dict(case.options)
        row: dict[str, Any] = {
            "name": case.name,
            "why": case.why,
            "messages": messages,
            "options": options,
        }
        try:
            row["rendered"] = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **options
            )
        except Exception as error:  # the outcome itself is the record
            row["error"] = _upstream_error(error)
        rows.append(row)
    return rows


def harvest_responses(tokenizer: Any, cases: Sequence[ResponseCase]) -> list[dict[str, Any]]:
    """Upstream `parse_response(text, prefix=...)` for each output (ADR 0084 decision 7 — the
    parser is upstream's, called independently).

    The prefix is upstream's own rendering of the case's conversation with the generation prompt,
    so the parser sees the prefill it would see in use (`<|turn>model\\n`, 12B's empty thought, the
    thought channel reopened after a result).

    MUST: each case's `expect` label is checked against what upstream did — `upstream-raises` must
    raise and the others must not, and a `blank-segment` output must read differently from its
    normalized form — so a label cannot drift from the behaviour it documents. For `ts-rejects`
    and `blank-segment` the upstream result is kept as a Python repr (documentation only — it can
    hold values JSON cannot, such as NaN or inf).
    """
    rows: list[dict[str, Any]] = []
    for case in cases:
        if case.expect not in RESPONSE_EXPECTATIONS:
            raise AssertionError(f"{case.name}: unknown expect label {case.expect!r}")
        assert_tool_conversation(case.messages, case.options)
        prefix = (
            case.raw_prefix
            if case.raw_prefix is not None
            else tokenizer.apply_chat_template(
                [dict(message) for message in case.messages],
                tokenize=False,
                add_generation_prompt=True,
                **dict(case.options),
            )
        )
        if (case.expect == "blank-segment") != (case.normalized is not None):
            raise AssertionError(
                f"{case.name}: `normalized` goes with the blank-segment label only"
            )
        row: dict[str, Any] = {
            "name": case.name,
            "why": case.why,
            "prefix": prefix,
            "text": case.text,
            "expect": case.expect,
        }
        try:
            parsed = tokenizer.parse_response(case.text, prefix=prefix)
        except ValueError as error:
            if case.expect != "upstream-raises":
                raise AssertionError(
                    f"{case.name}: upstream raised although the case expects {case.expect!r}"
                    f" ({_upstream_error(error)})"
                ) from error
            row["error"] = _upstream_error(error)
        else:
            if case.expect == "upstream-raises":
                raise AssertionError(
                    f"{case.name}: upstream returned {parsed!r} instead of raising"
                )
            if case.expect == "parse":
                row["parsed"] = parsed
            elif case.expect == "blank-segment":
                assert case.normalized is not None
                normalized = tokenizer.parse_response(case.normalized, prefix=prefix)
                if normalized == parsed:
                    raise AssertionError(
                        f"{case.name}: upstream reads the blank segment as nothing already"
                    )
                row["normalized"] = case.normalized
                row["parsed"] = normalized
                row["upstream"] = repr(parsed)
            else:
                row["upstream"] = repr(parsed)
        rows.append(row)
    return rows


def build_chat_fixture(
    *,
    tokenizer: Any,
    compiled: CompiledTokenizer,
    cases: Sequence[ChatCase],
    stops: Sequence[int],
    source: Mapping[str, Any],
    tool_cases: Sequence[ToolChatCase] = (),
    rejected_cases: Sequence[RejectedCase] = (),
    response_cases: Sequence[ResponseCase] = (),
    processor: Any = None,
    image_cases: Sequence[ImageChatCase] = (),
    rejected_image_cases: Sequence[ImageChatCase] = (),
) -> dict[str, Any]:
    """Builds one chat fixture (expectations from upstream, the subset from the compiled asset).

    The vocabulary and merges subset is cut by the same single function as the encoding fixtures
    ({@link _shared.gemma_tokenizer.asset_subset}) — no second implementation of the trimming.
    If something is missing, the TS-side comparison **fails**, so a mistake does not fall on the
    unsafe side.
    """
    chat = harvest_cases(tokenizer, cases)
    tool_chat = harvest_tool_cases(tokenizer, tool_cases)
    rejected = harvest_rejected_cases(tokenizer, rejected_cases)
    responses = harvest_responses(tokenizer, response_cases)
    image_chat = harvest_image_cases(processor, image_cases)
    rejected_images = harvest_rejected_image_cases(processor, rejected_image_cases)
    required = {token_id for row in [*chat, *tool_chat, *image_chat] for token_id in row["ids"]}
    # The EOS set holds ids that never appear in a body (`<|tool_response>`), so they are kept
    # explicitly — the TS side derives this set from the added vocabulary and compares it with
    # `stopTokens` here.
    required.update(stops)
    required.add(compiled.bos_id)
    if image_cases:
        # All three markers, whether or not a case's ids hold them (the refusal rows spell them).
        required.update(
            token_id
            for key, token_id in vision_tokens(processor).items()
            if key != "softTokenBudget"
        )
    added_by_id = {token_id: token for token, token_id in compiled.added_tokens}
    # The response texts join the subset too: the TS session tests encode model outputs with
    # their special spellings (a spelling missing from the subset would encode as bytes).
    texts = [row["rendered"] for row in [*chat, *tool_chat, *image_chat]]
    texts.extend(row["prefix"] + row["text"] for row in responses)
    texts.extend(
        part["text"]
        for row in rejected_images
        for message in row["messages"]
        for part in (message["content"] if isinstance(message["content"], (list, tuple)) else [])
        if part["type"] == "text"
    )
    fixture: dict[str, Any] = {
        "source": dict(source),
        "spec": spec_of(compiled),
        "asset": asset_subset(compiled, texts=texts, required_ids=required),
        "stopTokens": list(stops),
        "stopTokenSpellings": [added_by_id.get(token_id) for token_id in stops],
        "chat": chat,
    }
    if tool_cases:
        fixture["toolChat"] = tool_chat
    if rejected_cases:
        fixture["rejectedChat"] = rejected
    if response_cases:
        fixture["responses"] = responses
    if image_cases:
        fixture["visionTokens"] = vision_tokens(processor)
        fixture["imageChat"] = image_chat
    if rejected_image_cases:
        fixture["rejectedImageChat"] = rejected_images
    return fixture


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Harvest a Gemma 4 chat parity fixture")
    # The models with a fixture of their own (E4B is rendered in E2B's dialect).
    parser.add_argument("--model", choices=tuple(FIXTURE_PATHS), default="e2b")
    model = parser.parse_args(argv).model
    # The tokenizer source, not the weights' (12B's QAT weights ship the `tokenizer.json` from
    # before Google's glitch-token fixes — `gemma4.sources`).
    checkpoint_dir = sources.tokenizer_dir(model)
    fixture_path = FIXTURE_PATHS[model]
    tokenizer_json = checkpoint_dir / "tokenizer.json"
    for name in SOURCE_FILES:
        if not (checkpoint_dir / name).is_file():
            raise SystemExit(f"[karume] upstream {name} is missing: {checkpoint_dir / name}")
    pins = sources.HUB_PINS.get(model)
    if pins is not None:
        sources.assert_snapshot_pin(checkpoint_dir, pins.tokenizer)
    from transformers import AutoProcessor, AutoTokenizer  # lazy import (only when harvesting)

    compiled = compile_tokenizer(json.loads(tokenizer_json.read_text(encoding="utf-8")))
    fixture = build_chat_fixture(
        tokenizer=AutoTokenizer.from_pretrained(str(checkpoint_dir)),
        compiled=compiled,
        cases=CASES,
        stops=stop_tokens(checkpoint_dir),
        source=_source(checkpoint_dir, sources.tokenizer_label(model)),
        tool_cases=TOOL_CASES,
        rejected_cases=REJECTED_CASES,
        response_cases=RESPONSE_CASES,
        # 12B is the one model of the family with a TS image path (ADR 0144).
        **(
            {
                "processor": AutoProcessor.from_pretrained(str(checkpoint_dir)),
                "image_cases": IMAGE_CASES,
                "rejected_image_cases": REJECTED_IMAGE_CASES,
            }
            if model == "12b"
            else {}
        ),
    )
    written = write_json(fixture_path, fixture)
    print(
        f"[karume] fixture {fixture_path} ({written / 1e3:.0f} KB, chat {len(fixture['chat'])}"
        f" / toolChat {len(fixture['toolChat'])} / rejectedChat {len(fixture['rejectedChat'])} /"
        f" responses {len(fixture['responses'])} cases, vocabulary"
        f" {len(fixture['asset']['vocab'])}, merges {len(fixture['asset']['merges'])}, stops"
        f" {fixture['stopTokens']})"
    )
    print(
        f"[karume] then format it from the repository root: `vp fmt"
        f" {fixture_path.relative_to(REPO_ROOT)}`"
    )


if __name__ == "__main__":
    main()
