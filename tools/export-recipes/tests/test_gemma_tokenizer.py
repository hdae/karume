"""Gemma 系トークナイザの compile（`_shared/gemma_tokenizer.py`）— 受理集合と畳み方。

固定するのは、**通してしまうと実行時まで表面化しない**側だけ:

- 受理集合の外を **compile 時に落とす**こと（ADR 0084 決定 1 — 実行時に散らさないのが
  compile-to-asset を採る理由そのもの）。normalizer / pre_tokenizer / decoder /
  model のフラグは 1 欄でも違えば別のトークナイザで、通せば id 列だけが静かに変わる
- byteIds を **256 本明示**で引き、欠落を落とすこと（`base + byte` の連番を仮定しない）
- merge の対の重複・語彙に無い側を落とすこと（上流は HashMap なので後勝ちで沈黙する）
- 部分集合の切り出し（{@link subset_keys}）が、走査中に現れうる対を覆うこと

実資産（32MB の `tokenizer.json` × 2）はローカルにしか無いので、ここは**合成の最小資産**で
回す。合成資産は `tokenizers.Tokenizer.from_str` にそのまま食わせられる形なので、期待値の
採取経路（{@link build_fixture}）まで同じ台本で通る。
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from gemma_synthetic import PIECES
from gemma_synthetic import tokenizer_json as _tokenizer_json

from _shared.gemma_tokenizer import (
    ASSET_FORMAT,
    POST_BOS_EOS,
    POST_NONE,
    UnsupportedTokenizerError,
    asset_payload,
    build_fixture,
    byte_token,
    compile_tokenizer,
    subset_keys,
)


class TestAcceptance:
    def test_the_real_shape_compiles(self) -> None:
        compiled = compile_tokenizer(_tokenizer_json())

        assert compiled.post_processor == POST_NONE
        assert compiled.unk_id == 3
        assert len(compiled.byte_ids) == 256
        assert compiled.merges == [(260, 261), (264, 262), (263, 260), (266, 261)]

    def test_a_wrapping_post_processor_is_recognised(self) -> None:
        """EmbeddingGemma 側の形（`<bos>` … `<eos>`）— gemma4 と同じ実装で受ける。"""
        wrapped = _tokenizer_json(
            post_processor={
                "type": "TemplateProcessing",
                "single": [
                    {"SpecialToken": {"id": "<bos>", "type_id": 0}},
                    {"Sequence": {"id": "A", "type_id": 0}},
                    {"SpecialToken": {"id": "<eos>", "type_id": 0}},
                ],
                "pair": [],
                "special_tokens": {},
            }
        )

        assert compile_tokenizer(wrapped).post_processor == POST_BOS_EOS

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("model.type", "Unigram"),
            ("model.dropout", 0.1),
            ("model.byte_fallback", False),
            ("model.fuse_unk", False),
            ("model.ignore_merges", True),
            ("model.continuing_subword_prefix", "##"),
            ("model.end_of_word_suffix", "</w>"),
            ("model.unk_token", "<pad>"),
            ("truncation", {"max_length": 512}),
            ("padding", {"strategy": "BatchLongest"}),
        ],
    )
    def test_a_changed_flag_is_rejected(self, field: str, value: Any) -> None:
        with pytest.raises(UnsupportedTokenizerError):
            compile_tokenizer(_tokenizer_json(**{field: value}))

    def test_a_different_normalizer_is_rejected(self) -> None:
        """metaspace の入り方が変われば id 列が丸ごと変わる（例外にはならない）。"""
        with pytest.raises(UnsupportedTokenizerError):
            compile_tokenizer(
                _tokenizer_json(
                    normalizer={"type": "Replace", "pattern": {"String": " "}, "content": "_"}
                )
            )

    def test_a_different_pre_tokenizer_is_rejected(self) -> None:
        with pytest.raises(UnsupportedTokenizerError):
            compile_tokenizer(_tokenizer_json(pre_tokenizer={"type": "Metaspace"}))

    def test_a_reordered_decoder_chain_is_rejected(self) -> None:
        """`ByteFallback` と `Replace` の順が入れ替わると byte run の畳み方が変わる。"""
        reordered = {
            "type": "Sequence",
            "decoders": [
                {"type": "ByteFallback"},
                {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
                {"type": "Fuse"},
            ],
        }
        with pytest.raises(UnsupportedTokenizerError):
            compile_tokenizer(_tokenizer_json(decoder=reordered))

    def test_an_unknown_post_processor_is_rejected(self) -> None:
        with pytest.raises(UnsupportedTokenizerError):
            compile_tokenizer(
                _tokenizer_json(
                    post_processor={
                        "type": "TemplateProcessing",
                        "single": [{"SpecialToken": {"id": "<bos>", "type_id": 0}}],
                        "pair": [],
                        "special_tokens": {},
                    }
                )
            )

    def test_an_added_token_flag_that_changes_splitting_is_rejected(self) -> None:
        raw = _tokenizer_json()
        raw["added_tokens"][0]["lstrip"] = True
        with pytest.raises(UnsupportedTokenizerError):
            compile_tokenizer(raw)


class TestByteFallbackTable:
    def test_a_missing_byte_token_is_rejected(self) -> None:
        """`base + byte` の連番は**この資産の実測事実**であって schema の保証ではない。"""
        raw = _tokenizer_json()
        removed = raw["model"]["vocab"].pop(byte_token(0x7F))
        raw["model"]["vocab"]["<0x7f>"] = removed  # 小文字綴りは上流の綴りではない

        with pytest.raises(UnsupportedTokenizerError, match="0x7F"):
            compile_tokenizer(raw)

    def test_a_byte_shaped_token_beyond_the_256_is_rejected(self) -> None:
        """decode の byte 判定（id 集合）と上流の判定（綴りの形）が食い違う形を塞ぐ。"""
        raw = _tokenizer_json()
        raw["model"]["vocab"]["<0xZZ>"] = len(raw["model"]["vocab"])

        with pytest.raises(UnsupportedTokenizerError, match="256"):
            compile_tokenizer(raw)


class TestMerges:
    def test_a_duplicated_pair_is_rejected(self) -> None:
        """上流は HashMap なので重複対は後勝ちで沈黙し、分割規則だけが別物になる。"""
        raw = _tokenizer_json()
        raw["model"]["merges"].append(["a", "b"])

        with pytest.raises(UnsupportedTokenizerError, match="重複"):
            compile_tokenizer(raw)

    def test_a_pair_whose_join_is_not_in_the_vocabulary_is_rejected(self) -> None:
        raw = _tokenizer_json()
        raw["model"]["merges"].append(["c", "a"])

        with pytest.raises(UnsupportedTokenizerError, match="語彙に無い"):
            compile_tokenizer(raw)

    def test_the_old_string_form_is_rejected(self) -> None:
        raw = _tokenizer_json()
        raw["model"]["merges"] = ["a b"]

        with pytest.raises(UnsupportedTokenizerError):
            compile_tokenizer(raw)


class TestAssetPayload:
    def test_the_payload_carries_the_shape_the_runtime_matches_on(self) -> None:
        compiled = compile_tokenizer(_tokenizer_json())
        payload = asset_payload(compiled, source={"path": "synthetic"})

        assert payload["format"] == ASSET_FORMAT
        assert payload["spec"]["postProcessor"] == POST_NONE
        assert payload["mergesCount"] == len(compiled.merges)
        assert payload["mergesText"].count("\n") == len(compiled.merges) - 1
        assert payload["vocab"][payload["byteIds"][0xFF]] == "<0xFF>"

    def test_the_vocabulary_is_written_in_id_order(self) -> None:
        compiled = compile_tokenizer(_tokenizer_json())
        payload = asset_payload(compiled, source={})

        assert payload["vocab"][0] == "<pad>"
        assert payload["vocab"][-1] == PIECES[-1]


class TestSubsetKeys:
    def test_every_substring_up_to_the_limit_is_covered(self) -> None:
        keys = subset_keys(["abc"], max_token_chars=3)

        assert keys == {"a", "b", "c", "ab", "bc", "abc"}

    def test_spaces_are_folded_to_metaspace_before_slicing(self) -> None:
        """記号は**正規化後**の文字列の部分列になる — 生のまま切ると `▁a` を取り落とす。"""
        assert "▁a" in subset_keys([" a"], max_token_chars=2)

    def test_longer_substrings_than_the_limit_are_not_needed(self) -> None:
        keys = subset_keys(["abcd"], max_token_chars=2)

        assert "abc" not in keys
        assert "cd" in keys


class TestFixture:
    def test_expectations_come_from_the_upstream_tokenizer(self, tmp_path: Any) -> None:
        """期待値の経路に compile を通さない（通すと突合が恒真化する — ADR 0084 決定 7）。"""
        # 上流参照の `tokenizers` は家族グループ（transformers）の推移依存で、既定 sync の CI には
        # 無い — 他の家族依存テストと同じく SKIP で守る（`test_optional_group_imports.py` の作法）。
        pytest.importorskip("tokenizers")
        from _shared.gemma_tokenizer import EncodeCase

        raw = _tokenizer_json()
        path = tmp_path / "tokenizer.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        compiled = compile_tokenizer(raw)

        fixture = build_fixture(
            tokenizer_json=path,
            compiled=compiled,
            encode_cases=[EncodeCase("abc", "merge が 2 段掛かる", "abc")],
            decode_cases=[],
            source={},
        )

        [case] = fixture["encode"]
        # "abc" は ("a","b") → ("ab","c") の 2 段で 1 トークンへ畳まれる。
        assert case["ids"] == [compiled.vocab.index("abc")]
        # 部分集合には走査中に現れる記号と対が揃っている。
        assert [row for row in fixture["asset"]["merges"] if row[2] == 0]
        assert fixture["decode"][0]["name"] == "roundtrip-abc"
        assert fixture["decode"][0]["text"] == "abc"
