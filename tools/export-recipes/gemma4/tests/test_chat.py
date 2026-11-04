"""Script-level contracts of the chat fixture script (`gemma4/chat.py`) — the part that needs no
real assets.

The real assets (the 32 MB `tokenizer.json` + `chat_template.jinja`) exist only locally, so this
runs on **a synthetic asset + a synthetic template** (`tests/gemma_synthetic.py`). It pins only
the rules whose breakage would turn into a **false PASS**:

- the scope check ({@link chat.assert_plain_conversation}) **actually rejects** tools / thinking /
  tool_call / image parts / unknown roles / an empty conversation (ADR 0084 decisions 5 and 6.3).
  Ignoring them silently would let "I passed a tool and it was not used" through without an
  exception
- every case of the shipped battery ({@link chat.CASES}) is inside the scope, and no name repeats
- **not one line of an in-house renderer** is on the expectation path (ADR 0084 decision 7 —
  otherwise the comparison becomes tautological). This is checked as "what the injected tokenizer
  returned is stored as is"
- a mismatch between `apply_chat_template(tokenize=True)` and "render → encode with
  `add_special_tokens=False`" fails at harvest time (the very premise of the TS two-step build)
- `eos_token_id` of `generation_config.json` is read as a set (a singular declaration is accepted
  too)

Only the cases that need transformers SKIP through `importorskip` (the two-job layout of ADR
0065).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from gemma_synthetic import tokenizer_json

from _shared.gemma_tokenizer import compile_tokenizer
from gemma4 import chat, sources

#: A synthetic template (it has only the three branches of plain conversations — not a stand-in
#: for the real template, but a target that shows "the injected renderer's output is stored as
#: is").
SYNTHETIC_TEMPLATE = (
    "{{- bos_token -}}"
    "{%- for m in messages -%}{{- '<' + m['role'] + '>' + m['content'] -}}{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<model>' -}}{%- endif -%}"
)


def _hf_tokenizer(template: str = SYNTHETIC_TEMPLATE) -> Any:
    """An HF tokenizer of the synthetic asset + synthetic template (the harvest path goes through
    the same interface as the real one)."""
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_str(json.dumps(tokenizer_json())),
        chat_template=template,
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
    )


class TestScopeGate:
    def test_the_shipped_battery_is_in_scope(self) -> None:
        """Every case of the shipped battery is inside the scope (no shape that fails before the
        harvest)."""
        for case in chat.CASES:
            chat.assert_plain_conversation(case.messages)

    def test_the_case_names_are_unique(self) -> None:
        """A repeated name would hide, in the TS-side comparison, which of the cases failed."""
        names = [case.name for case in chat.CASES]

        assert sorted(names) == sorted(set(names))

    def test_an_empty_conversation_is_rejected(self) -> None:
        """Upstream `apply_chat_template` rejects it too (the accepted set is not wider than
        upstream's)."""
        with pytest.raises(chat.OutOfScopeChatError, match="empty"):
            chat.assert_plain_conversation([])

    @pytest.mark.parametrize(
        "extra",
        [
            {"tools": [{"name": "search"}]},
            {"reasoning": "…"},
            {"reasoning_content": "…"},
            {"tool_calls": [{"function": {"name": "f", "arguments": {}}}]},
            {"tool_responses": [{"name": "f", "response": {}}]},
            {"thinking": "…"},
        ],
    )
    def test_an_out_of_scope_field_is_rejected(self, extra: dict[str, Any]) -> None:
        """MUST: not ignored silently (ignoring it would let "given but without effect" through
        without an exception)."""
        with pytest.raises(chat.OutOfScopeChatError, match="out-of-scope fields"):
            chat.assert_plain_conversation([{"role": "user", "content": "hi", **extra}])

    @pytest.mark.parametrize("role", ["tool", "model", "function", ""])
    def test_an_unknown_role_is_rejected(self, role: str) -> None:
        """`model` is the template's spelling on the **output side**, not an input role."""
        with pytest.raises(chat.OutOfScopeChatError, match="role"):
            chat.assert_plain_conversation([{"role": role, "content": "hi"}])

    @pytest.mark.parametrize(
        "content",
        [
            [{"type": "text", "text": "hi"}],
            [{"type": "image_url", "image_url": {"url": "…"}}],
            None,
            42,
        ],
    )
    def test_a_non_string_content_is_rejected(self, content: Any) -> None:
        """Arrays of image / audio parts are outside the first version's scope (the `<|image|>`
        spelling is not brought in)."""
        with pytest.raises(chat.OutOfScopeChatError, match="content"):
            chat.assert_plain_conversation([{"role": "user", "content": content}])


class TestStopTokens:
    def test_a_declared_set_is_read_as_is(self, tmp_path: Path) -> None:
        (tmp_path / "generation_config.json").write_text(
            json.dumps({"eos_token_id": [1, 106, 50]}), encoding="utf-8"
        )

        assert chat.stop_tokens(tmp_path) == [1, 106, 50]

    def test_a_singular_declaration_is_read_as_a_set(self, tmp_path: Path) -> None:
        """The stop condition is a **set**, not a single EOS (ADR 0083 decision 8) — a singular
        declaration is accepted as a set too."""
        (tmp_path / "generation_config.json").write_text(
            json.dumps({"eos_token_id": 1}), encoding="utf-8"
        )

        assert chat.stop_tokens(tmp_path) == [1]

    @pytest.mark.parametrize("declared", [None, [], "1", [1, "106"]])
    def test_a_malformed_declaration_is_rejected(self, tmp_path: Path, declared: Any) -> None:
        (tmp_path / "generation_config.json").write_text(
            json.dumps({"eos_token_id": declared}), encoding="utf-8"
        )

        with pytest.raises(ValueError, match="eos_token_id"):
            chat.stop_tokens(tmp_path)


class TestHarvest:
    def test_expectations_come_from_the_injected_tokenizer(self) -> None:
        """No in-house renderer on the expectation path (ADR 0084 decision 7).

        The synthetic template's spellings (`<user>` / `<model>`) differ **on purpose** from the
        real template's (`<|turn>` / `<turn|>`) — if the script rendered on its own, the
        difference would show in the expectations as is.
        """
        [row] = chat.harvest_cases(
            _hf_tokenizer(),
            [chat.ChatCase("one", "minimal form", ({"role": "user", "content": "abc"},))],
        )

        assert row["rendered"] == "<bos><user>abc<model>"
        assert row["name"] == "one"
        assert row["messages"] == [{"role": "user", "content": "abc"}]

    def test_the_ids_are_the_encoding_of_the_rendered_text(self) -> None:
        """`ids` is "render → encode with `add_special_tokens=False`" (the premise of the TS
        two-step build)."""
        tokenizer = _hf_tokenizer()

        [row] = chat.harvest_cases(
            tokenizer,
            [chat.ChatCase("one", "minimal form", ({"role": "user", "content": "abc"},))],
        )

        assert row["ids"] == list(tokenizer(row["rendered"], add_special_tokens=False)["input_ids"])

    def test_the_scope_gate_runs_on_every_harvested_case(self) -> None:
        """An out-of-scope case does **not enter** the harvest (a rejected example does not ship
        as an expectation)."""
        with pytest.raises(chat.OutOfScopeChatError):
            chat.harvest_cases(
                _hf_tokenizer(),
                [chat.ChatCase("bad", "out of scope", ({"role": "tool", "content": "x"},))],
            )

    def test_a_tokenizer_whose_two_paths_disagree_is_rejected(self) -> None:
        """MUST: fail at harvest time — shipped with the mismatch, the TS side would see "the
        rendering matches but the ids differ", and the reader could not tell which step broke."""

        class Disagreeing:
            def apply_chat_template(
                self, messages: Any, *, tokenize: bool, add_generation_prompt: bool
            ) -> Any:
                return "<bos>x" if not tokenize else {"input_ids": [7, 7, 7]}

            def __call__(self, text: str, *, add_special_tokens: bool) -> Any:
                return {"input_ids": [1, 2]}

        with pytest.raises(AssertionError, match="does not match"):
            chat.harvest_cases(
                Disagreeing(),
                [chat.ChatCase("one", "minimal form", ({"role": "user", "content": "abc"},))],
            )


class TestFixture:
    def test_the_subset_covers_every_harvested_id(self) -> None:
        """A subset that falls short fails the TS-side comparison (a trimming mistake does not
        fall on the unsafe side)."""
        raw = tokenizer_json()
        compiled = compile_tokenizer(raw)

        fixture = chat.build_chat_fixture(
            tokenizer=_hf_tokenizer(),
            compiled=compiled,
            cases=[chat.ChatCase("one", "minimal form", ({"role": "user", "content": "abc"},))],
            stops=[compiled.eos_id],
            source={},
        )

        kept = {token_id for token_id, _ in fixture["asset"]["vocab"]}
        kept.update(token_id for _, token_id in fixture["asset"]["addedTokens"])
        assert set(fixture["chat"][0]["ids"]) <= kept

    def test_the_stop_tokens_ride_the_same_fixture(self) -> None:
        """The chat format and the EOS set come from the same bundle (ADR 0084 decision 5) — the
        spellings ride along too."""
        compiled = compile_tokenizer(tokenizer_json())
        pad_id = compiled.vocab.index("<pad>")
        case = chat.ChatCase("one", "minimal form", ({"role": "user", "content": "abc"},))

        def build(stops: list[int]) -> dict[str, Any]:
            return chat.build_chat_fixture(
                tokenizer=_hf_tokenizer(),
                compiled=compiled,
                cases=[case],
                stops=stops,
                source={},
            )

        def added(fixture: dict[str, Any]) -> list[str]:
            return [row[0] for row in fixture["asset"]["addedTokens"]]

        with_pad = build([compiled.eos_id, pad_id])
        without_pad = build([compiled.eos_id])

        assert with_pad["stopTokens"] == [compiled.eos_id, pad_id]
        assert with_pad["stopTokenSpellings"] == ["<eos>", "<pad>"]
        # `<pad>` never appears in a body, so it stays only because it is in `stops` (without it,
        # it drops out of the subset = the TS side could not check its derivation).
        assert "<pad>" in added(with_pad)
        assert "<pad>" not in added(without_pad)


def _twelve_b_tokenizer() -> Any:
    """12B's tokenizer from its pinned tokenizer snapshot (present on a machine that fetched it; the
    cases SKIP explicitly without it). The directory is resolved here, not at import: the hub
    cache's location comes from huggingface_hub, which arrives with transformers."""
    pytest.importorskip("transformers")
    twelve_b_dir = sources.tokenizer_dir("12b")
    if not (twelve_b_dir / "chat_template.jinja").is_file():
        pytest.skip(f"12B's snapshot is not in the hub cache ({twelve_b_dir})")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(twelve_b_dir), local_files_only=True)


class TestDialects:
    def test_every_model_has_a_dialect_with_a_generation_suffix(self) -> None:
        assert set(chat.CHAT_DIALECTS.values()) <= set(chat.GENERATION_PROMPT_SUFFIX)
        assert chat.GENERATION_PROMPT_SUFFIX[chat.GEMMA4_CHAT_DIALECT] == ""
        assert chat.CHAT_DIALECTS["12b"] == chat.GEMMA4_UNIFIED_CHAT_DIALECT

    def test_the_12b_fixture_sits_next_to_e2b_s(self) -> None:
        assert chat.FIXTURE_PATHS["e2b"] == chat.FIXTURE_PATH
        assert chat.FIXTURE_PATHS["12b"].parent == chat.FIXTURE_PATH.parent

    def test_12b_renders_e2b_s_text_plus_the_empty_thought_channel(self) -> None:
        """The `gemma4-unified` dialect is E2B's rendering with the generation suffix — checked on
        every shipped case against 12B's own `apply_chat_template` and E2B's tracked fixture
        (rendered by E2B's template when it was harvested)."""
        tokenizer = _twelve_b_tokenizer()
        e2b = json.loads(chat.FIXTURE_PATH.read_text(encoding="utf-8"))["chat"]
        suffix = chat.GENERATION_PROMPT_SUFFIX[chat.GEMMA4_UNIFIED_CHAT_DIALECT]

        assert [row["name"] for row in e2b] == [case.name for case in chat.CASES]
        for row in e2b:
            rendered = tokenizer.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=True
            )
            assert rendered == row["rendered"] + suffix, row["name"]

    def test_the_12b_fixture_harvests_through_its_own_template(self) -> None:
        """The harvest path the 12B fixture takes (rendered text and ids agree for every case)."""
        tokenizer = _twelve_b_tokenizer()

        rows = chat.harvest_cases(tokenizer, chat.CASES)

        suffix = chat.GENERATION_PROMPT_SUFFIX[chat.GEMMA4_UNIFIED_CHAT_DIALECT]
        assert all(row["rendered"].endswith("<|turn>model\n" + suffix) for row in rows)


# ---- the widened scope: tools, tool calls, tool results, thinking, output parsing -------------

#: A synthetic template that writes the options it receives (it renders nothing like the real one;
#: it shows that the harvest passes the case's options through and records what comes back).
SYNTHETIC_TOOL_TEMPLATE = (
    "{{- bos_token -}}"
    "{%- if enable_thinking -%}<think>{%- endif -%}"
    "{%- if preserve_thinking -%}<keep>{%- endif -%}"
    "{%- for t in tools or [] -%}{{- '<tool:' + t['function']['name'] + '>' -}}{%- endfor -%}"
    "{%- for m in messages -%}{{- '<' + m['role'] + '>' -}}{{- m['content'] or '' -}}{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<model>' -}}{%- endif -%}"
)


class TestToolScopeGate:
    def test_the_tool_battery_is_in_scope(self) -> None:
        for case in chat.TOOL_CASES:
            chat.assert_tool_conversation(case.messages, case.options)

    def test_the_response_conversations_are_in_scope(self) -> None:
        for case in chat.RESPONSE_CASES:
            chat.assert_tool_conversation(case.messages, case.options)

    def test_names_are_unique_within_each_battery(self) -> None:
        """A duplicated name would hide which case failed on the TS side."""
        for battery in (chat.TOOL_CASES, chat.REJECTED_CASES, chat.RESPONSE_CASES):
            names = [case.name for case in battery]
            assert sorted(names) == sorted(set(names))

    def test_every_response_case_has_a_known_label(self) -> None:
        assert {case.expect for case in chat.RESPONSE_CASES} == chat.RESPONSE_EXPECTATIONS

    @pytest.mark.parametrize(
        ("message", "match"),
        [
            (
                {"role": "assistant", "tool_responses": [{"name": "f", "response": {}}]},
                "out-of-scope fields",
            ),
            ({"role": "assistant", "thinking": "…"}, "out-of-scope fields"),
            ({"role": "user", "content": "hi", "reasoning": "…"}, "out-of-scope fields"),
            ({"role": "tool", "content": [{"type": "text", "text": "x"}]}, "array"),
            ({"role": "user", "content": [{"type": "image_url"}]}, "array"),
            ({"role": "user", "content": None}, "not a string"),
            ({"role": "model", "content": "hi"}, "role"),
        ],
    )
    def test_out_of_scope_messages_are_rejected(self, message: dict[str, Any], match: str) -> None:
        with pytest.raises(chat.OutOfScopeChatError, match=match):
            chat.assert_tool_conversation([message], {})

    def test_an_unknown_option_is_rejected(self) -> None:
        with pytest.raises(chat.OutOfScopeChatError, match="unknown template options"):
            chat.assert_tool_conversation([{"role": "user", "content": "hi"}], {"thinking": True})

    def test_an_empty_conversation_is_rejected(self) -> None:
        with pytest.raises(chat.OutOfScopeChatError, match="empty"):
            chat.assert_tool_conversation([], {})


class TestToolHarvest:
    def test_the_options_reach_the_injected_template(self) -> None:
        """The expectation is what the injected tokenizer rendered with the case's options."""
        case = chat.ToolChatCase(
            "one",
            "options pass through",
            ({"role": "user", "content": "abc"},),
            {"tools": [{"type": "function", "function": {"name": "f"}}], "enable_thinking": True},
        )

        [row] = chat.harvest_tool_cases(_hf_tokenizer(SYNTHETIC_TOOL_TEMPLATE), [case])

        assert row["rendered"] == "<bos><think><tool:f><user>abc<model>"
        assert row["options"] == dict(case.options)

    def test_the_ids_are_the_encoding_of_the_rendered_text(self) -> None:
        tokenizer = _hf_tokenizer(SYNTHETIC_TOOL_TEMPLATE)
        case = chat.ToolChatCase("one", "ids", ({"role": "user", "content": "abc"},), {})

        [row] = chat.harvest_tool_cases(tokenizer, [case])

        assert row["ids"] == list(tokenizer(row["rendered"], add_special_tokens=False)["input_ids"])

    def test_the_gate_runs_on_every_tool_case(self) -> None:
        bad = chat.ToolChatCase(
            "bad", "legacy field", ({"role": "assistant", "tool_responses": []},), {}
        )
        with pytest.raises(chat.OutOfScopeChatError):
            chat.harvest_tool_cases(_hf_tokenizer(SYNTHETIC_TOOL_TEMPLATE), [bad])


class _FakeParser:
    """A tokenizer whose template and parser are scripted (the label checks need no assets)."""

    def apply_chat_template(
        self, messages: Any, *, tokenize: bool, add_generation_prompt: bool, **options: Any
    ) -> Any:
        if options.get("raise_please"):
            raise ValueError("template refused\nsecond line")
        return "PFX"

    def parse_response(self, text: str, *, prefix: str) -> Any:
        if text == "bad":
            raise ValueError("cannot parse\nmore detail")
        return {"role": "assistant", "content": f"{prefix}:{text}"}


def _response(text: str, expect: str) -> chat.ResponseCase:
    return chat.ResponseCase("r", "why", ({"role": "user", "content": "q"},), {}, text, expect)


class TestResponseHarvest:
    def test_a_parsed_case_records_upstream_s_result_and_prefix(self) -> None:
        [row] = chat.harvest_responses(_FakeParser(), [_response("ok", "parse")])

        assert row["prefix"] == "PFX"
        assert row["parsed"] == {"role": "assistant", "content": "PFX:ok"}
        assert "error" not in row

    def test_an_upstream_error_is_recorded_by_type_and_first_line(self) -> None:
        [row] = chat.harvest_responses(_FakeParser(), [_response("bad", "upstream-raises")])

        assert row["error"] == "ValueError: cannot parse"

    def test_a_ts_rejection_keeps_upstream_s_value_as_a_repr(self) -> None:
        [row] = chat.harvest_responses(_FakeParser(), [_response("ok", "ts-rejects")])

        assert row["upstream"] == repr({"role": "assistant", "content": "PFX:ok"})
        assert "parsed" not in row

    @pytest.mark.parametrize(
        ("text", "expect"), [("bad", "parse"), ("bad", "ts-rejects"), ("ok", "upstream-raises")]
    )
    def test_a_label_that_contradicts_upstream_fails_the_harvest(
        self, text: str, expect: str
    ) -> None:
        """MUST: the label documents observed behaviour — a mismatch stops the harvest."""
        with pytest.raises(AssertionError):
            chat.harvest_responses(_FakeParser(), [_response(text, expect)])

    def test_an_unknown_label_fails_the_harvest(self) -> None:
        with pytest.raises(AssertionError, match="unknown expect label"):
            chat.harvest_responses(_FakeParser(), [_response("ok", "maybe")])


class TestRejectedHarvest:
    def test_upstream_s_rendering_or_error_is_recorded(self) -> None:
        rendered, raised = chat.harvest_rejected_cases(
            _FakeParser(),
            [
                chat.RejectedCase("a", "renders", ({"role": "user", "content": "x"},), {}),
                chat.RejectedCase(
                    "b", "raises", ({"role": "user", "content": "x"},), {"raise_please": True}
                ),
            ],
        )

        assert rendered["rendered"] == "PFX" and "error" not in rendered
        assert raised["error"] == "ValueError: template refused" and "rendered" not in raised


class TestToolFixture:
    def test_the_new_batteries_ride_the_fixture_and_its_subset(self) -> None:
        compiled = compile_tokenizer(tokenizer_json())
        tokenizer = _hf_tokenizer(SYNTHETIC_TOOL_TEMPLATE)
        tokenizer.parse_response = lambda text, *, prefix: {"role": "assistant", "content": text}

        fixture = chat.build_chat_fixture(
            tokenizer=tokenizer,
            compiled=compiled,
            cases=[chat.ChatCase("one", "plain", ({"role": "user", "content": "abc"},))],
            stops=[compiled.eos_id],
            source={},
            tool_cases=[chat.ToolChatCase("t", "tool", ({"role": "user", "content": "abc"},), {})],
            rejected_cases=[
                chat.RejectedCase("x", "rejected", ({"role": "user", "content": "abc"},), {})
            ],
            response_cases=[_response("<pad>", "parse")],
        )

        assert [row["name"] for row in fixture["toolChat"]] == ["t"]
        assert [row["name"] for row in fixture["rejectedChat"]] == ["x"]
        assert [row["name"] for row in fixture["responses"]] == ["r"]
        # The response text joins the subset (the TS session tests encode model outputs).
        assert "<pad>" in [row[0] for row in fixture["asset"]["addedTokens"]]


def _model_tokenizer(model: str) -> Any:
    """Model `model`'s tokenizer from its tokenizer directory (SKIPs without the files)."""
    pytest.importorskip("transformers")
    directory = sources.tokenizer_dir(model)
    if not (directory / "chat_template.jinja").is_file():
        pytest.skip(f"{model}'s tokenizer files are not present ({directory})")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(directory), local_files_only=True)


class TestToolDialects:
    def test_12b_renders_the_tool_battery_as_e2b_plus_the_suffix_when_a_turn_opens(self) -> None:
        """The two templates share their tool and thinking parts: 12B adds the empty thought only
        where the generation prompt opens a new model turn with thinking off (never after a tool
        result — that prompt continues the open turn)."""
        tokenizer = _twelve_b_tokenizer()
        rows = json.loads(chat.FIXTURE_PATH.read_text(encoding="utf-8"))["toolChat"]
        suffix = chat.GENERATION_PROMPT_SUFFIX[chat.GEMMA4_UNIFIED_CHAT_DIALECT]

        assert [row["name"] for row in rows] == [case.name for case in chat.TOOL_CASES]
        for row in rows:
            opens = row["rendered"].endswith("<|turn>model\n")
            expected = row["rendered"] + (
                suffix if opens and not row["options"].get("enable_thinking") else ""
            )
            rendered = tokenizer.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=True, **row["options"]
            )
            assert rendered == expected, row["name"]

    def test_e4b_renders_and_encodes_the_tool_battery_as_e2b(self) -> None:
        """E4B has no chat fixture of its own: its template is E2B's bytes, and on this battery
        its tokenizer gives E2B's ids too (checked against the tracked E2B fixture)."""
        tokenizer = _model_tokenizer("e4b")
        rows = json.loads(chat.FIXTURE_PATH.read_text(encoding="utf-8"))["toolChat"]

        for row in rows:
            ids = tokenizer.apply_chat_template(
                row["messages"], tokenize=True, add_generation_prompt=True, **row["options"]
            )["input_ids"]
            assert list(ids) == row["ids"], row["name"]

    def test_the_three_models_share_one_response_template(self) -> None:
        """One TS parser serves every model only while the snapshots declare the same parser."""
        e2b = _model_tokenizer("e2b").response_template

        assert e2b == _model_tokenizer("e4b").response_template
        assert e2b == _twelve_b_tokenizer().response_template


class TestBlankSegmentHarvest:
    def test_the_normalized_output_s_reading_is_the_expectation(self) -> None:
        case = chat.ResponseCase(
            "b", "why", ({"role": "user", "content": "q"},), {}, "x \n", "blank-segment", "x"
        )

        [row] = chat.harvest_responses(_FakeParser(), [case])

        assert row["parsed"] == {"role": "assistant", "content": "PFX:x"}
        assert row["upstream"] == repr({"role": "assistant", "content": "PFX:x \n"})
        assert row["normalized"] == "x"

    def test_a_normalized_form_upstream_reads_the_same_fails_the_harvest(self) -> None:
        """MUST: the case documents a real divergence, not a no-op."""
        case = chat.ResponseCase(
            "b", "why", ({"role": "user", "content": "q"},), {}, "x", "blank-segment", "x"
        )
        with pytest.raises(AssertionError, match="already"):
            chat.harvest_responses(_FakeParser(), [case])

    def test_normalized_goes_with_its_label_only(self) -> None:
        case = chat.ResponseCase(
            "b", "why", ({"role": "user", "content": "q"},), {}, "x", "parse", "y"
        )
        with pytest.raises(AssertionError, match="blank-segment label only"):
            chat.harvest_responses(_FakeParser(), [case])

    def test_a_raw_prefix_replaces_the_rendered_prompt(self) -> None:
        case = chat.ResponseCase(
            "r", "why", ({"role": "user", "content": "q"},), {}, "x", "parse", raw_prefix="RAW"
        )

        [row] = chat.harvest_responses(_FakeParser(), [case])

        assert row["prefix"] == "RAW"
        assert row["parsed"] == {"role": "assistant", "content": "RAW:x"}
