"""chat フィクスチャ台本（`gemma4/chat.py`）の台本レベルの約束事（実資産不要分）。

実資産（32MB の `tokenizer.json` + `chat_template.jinja`）はローカルにしか無いので、ここは
**合成資産 + 合成 template**（`tests/gemma_synthetic.py`）で回す。固定するのは、壊れると
**偽 PASS** になる側の規律だけ:

- 射程の門（{@link chat.assert_plain_conversation}）が tools / thinking / tool_call /
  画像パート / 未知 role / 空の会話を**実際に落とす**こと（ADR 0084 決定 5・6.3）。黙って
  無視すると「tool を渡したのに使われない」が例外なしで通る
- 出荷する電池（{@link chat.CASES}）が 1 件残らず射程の内側にあり、名前が重複しないこと
- 期待値の経路に**自前のレンダラを 1 行も通さない**こと（ADR 0084 決定 7 — 通すと突合が
  恒真化する）。ここは「注入した tokenizer が返したものがそのまま載る」で見る
- `apply_chat_template(tokenize=True)` と「描画 → `add_special_tokens=False` の符号化」の
  食い違いを採取時に落とすこと（TS 側の 2 段構成の前提そのもの）
- `generation_config.json` の `eos_token_id` を集合として読むこと（単数宣言も受ける）

transformers を要するケースだけ `importorskip` で SKIP する（ADR 0065 の 2 job 構成）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from gemma_synthetic import tokenizer_json

from _shared.gemma_tokenizer import compile_tokenizer
from gemma4 import chat

#: 合成 template（素の会話の 3 分岐だけを持つ — 実 template の代役ではなく「注入された
#: レンダラの出力がそのまま載る」ことを見るための的）。
SYNTHETIC_TEMPLATE = (
    "{{- bos_token -}}"
    "{%- for m in messages -%}{{- '<' + m['role'] + '>' + m['content'] -}}{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<model>' -}}{%- endif -%}"
)


def _hf_tokenizer(template: str = SYNTHETIC_TEMPLATE) -> Any:
    """合成資産 + 合成 template の HF tokenizer（採取経路を実物と同じ口で通す）。"""
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
        """出荷する電池が 1 件残らず射程の内側にある（採取の前に落ちる形を作らない）。"""
        for case in chat.CASES:
            chat.assert_plain_conversation(case.messages)

    def test_the_case_names_are_unique(self) -> None:
        """名前が重複すると TS 側の突合で「どちらのケースが落ちたか」が読めなくなる。"""
        names = [case.name for case in chat.CASES]

        assert sorted(names) == sorted(set(names))

    def test_an_empty_conversation_is_rejected(self) -> None:
        """上流の `apply_chat_template` も拒否する（受理集合を上流より広げない）。"""
        with pytest.raises(chat.OutOfScopeChatError, match="空"):
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
        """MUST: 黙って無視しない（無視すると「渡したのに効かない」が例外なしで通る）。"""
        with pytest.raises(chat.OutOfScopeChatError, match="射程外の欄"):
            chat.assert_plain_conversation([{"role": "user", "content": "hi", **extra}])

    @pytest.mark.parametrize("role", ["tool", "model", "function", ""])
    def test_an_unknown_role_is_rejected(self, role: str) -> None:
        """`model` は template の**出力側**の綴りで、入力の role ではない。"""
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
        """画像 / 音声パートの配列は初版の射程外（`<|image|>` の綴りを持ち込まない）。"""
        with pytest.raises(chat.OutOfScopeChatError, match="content"):
            chat.assert_plain_conversation([{"role": "user", "content": content}])


class TestStopTokens:
    def test_a_declared_set_is_read_as_is(self, tmp_path: Path) -> None:
        (tmp_path / "generation_config.json").write_text(
            json.dumps({"eos_token_id": [1, 106, 50]}), encoding="utf-8"
        )

        assert chat.stop_tokens(tmp_path) == [1, 106, 50]

    def test_a_singular_declaration_is_read_as_a_set(self, tmp_path: Path) -> None:
        """停止条件は単数の EOS ではなく**集合**（ADR 0083 決定 8）— 単数宣言も集合で受ける。"""
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
        """期待値の経路に自前のレンダラを通さない（ADR 0084 決定 7）。

        合成 template の綴り（`<user>` / `<model>`）は実 template の綴り（`<|turn>` /
        `<turn|>`）と**わざと**違えてある — 台本が自分で描画していれば、この差がそのまま
        期待値に出る。
        """
        [row] = chat.harvest_cases(
            _hf_tokenizer(),
            [chat.ChatCase("one", "最小形", ({"role": "user", "content": "abc"},))],
        )

        assert row["rendered"] == "<bos><user>abc<model>"
        assert row["name"] == "one"
        assert row["messages"] == [{"role": "user", "content": "abc"}]

    def test_the_ids_are_the_encoding_of_the_rendered_text(self) -> None:
        """`ids` は「描画 → `add_special_tokens=False` の符号化」（TS 側の 2 段構成の前提）。"""
        tokenizer = _hf_tokenizer()

        [row] = chat.harvest_cases(
            tokenizer,
            [chat.ChatCase("one", "最小形", ({"role": "user", "content": "abc"},))],
        )

        assert row["ids"] == list(tokenizer(row["rendered"], add_special_tokens=False)["input_ids"])

    def test_the_scope_gate_runs_on_every_harvested_case(self) -> None:
        """射程外のケースは採取に**入らない**（拒否例が期待値として配布物へ載らない）。"""
        with pytest.raises(chat.OutOfScopeChatError):
            chat.harvest_cases(
                _hf_tokenizer(),
                [chat.ChatCase("bad", "射程外", ({"role": "tool", "content": "x"},))],
            )

    def test_a_tokenizer_whose_two_paths_disagree_is_rejected(self) -> None:
        """MUST: 採取時に落とす — 食い違ったまま配ると TS 側は「描画は合うのに id が違う」に
        なり、どちらの段が壊れているのか読み手に伝わらない。"""

        class Disagreeing:
            def apply_chat_template(
                self, messages: Any, *, tokenize: bool, add_generation_prompt: bool
            ) -> Any:
                return "<bos>x" if not tokenize else {"input_ids": [7, 7, 7]}

            def __call__(self, text: str, *, add_special_tokens: bool) -> Any:
                return {"input_ids": [1, 2]}

        with pytest.raises(AssertionError, match="一致しない"):
            chat.harvest_cases(
                Disagreeing(),
                [chat.ChatCase("one", "最小形", ({"role": "user", "content": "abc"},))],
            )


class TestFixture:
    def test_the_subset_covers_every_harvested_id(self) -> None:
        """部分集合が足りなければ TS 側の突合が落ちる（絞り方の誤りは危険側に倒れない）。"""
        raw = tokenizer_json()
        compiled = compile_tokenizer(raw)

        fixture = chat.build_chat_fixture(
            tokenizer=_hf_tokenizer(),
            compiled=compiled,
            cases=[chat.ChatCase("one", "最小形", ({"role": "user", "content": "abc"},))],
            stops=[compiled.eos_id],
            source={},
        )

        kept = {token_id for token_id, _ in fixture["asset"]["vocab"]}
        kept.update(token_id for _, token_id in fixture["asset"]["addedTokens"])
        assert set(fixture["chat"][0]["ids"]) <= kept

    def test_the_stop_tokens_ride_the_same_fixture(self) -> None:
        """chat 形式と EOS 集合は同じ束から来る（ADR 0084 決定 5）— 綴りまで載せる。"""
        compiled = compile_tokenizer(tokenizer_json())
        pad_id = compiled.vocab.index("<pad>")
        case = chat.ChatCase("one", "最小形", ({"role": "user", "content": "abc"},))

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
        # `<pad>` は本文に 1 度も出ない綴りなので、`stops` に居るからこそ残る（居なければ
        # 部分集合から落ちる = TS 側が導出を突き合わせられない）。
        assert "<pad>" in added(with_pad)
        assert "<pad>" not in added(without_pad)
