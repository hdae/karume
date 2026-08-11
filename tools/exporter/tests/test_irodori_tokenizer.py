"""`irodori_tokenizer.py` の台本レベルの約束事（実資産不要分）。

実資産の emit は手動（`export_irodori.py` と同じ規律）。ここで固定するのは、壊れると
**沈黙の id 列不一致**になる側の規律だけ:

- 上流 `tokenizer.json` の形の門（model 種別 / unk_id / byte_fallback / normalizer /
  pre_tokenizer / 追加語彙のフラグ / バイトトークンの id 起点 / 行区切りの前提）
- `bosId` / `padId` がチェックポイント config と追加語彙の**両方**で一致すること
- byte_fallback の期待 id 列が UTF-8 から独立に計算されていること
- `normalize_text` の規則被覆の実測が、ケースを削れば実際に落ちること
- NFKC 差分表が恒等エントリを含まず、2 回作れば同じであること
"""

from __future__ import annotations

import copy
import unicodedata

import pytest

import irodori_tokenizer as tok

TEXT_CONFIG = {"bos_token_id": 1, "pad_token_id": 3}


#: 実物と同じ先頭 15 本（バイトトークンの id 起点が 15 になる並び）。
SPECIALS = (
    "<unk>",
    "<s>",
    "</s>",
    "<pad>",
    "<sep>",
    "<mask>",
    "<cls>",
    "<|system|>",
    "<|assistant|>",
    "<|user|>",
    "<|available_tools|>",
    "<|tool_calls|>",
    "<|tool_results|>",
    "<|code|>",
    "<|file|>",
)


def _raw(**overrides) -> dict:
    """上流 `tokenizer.json` の最小形（語彙 = 特殊 15 本 + バイト 256 本 + 通常 2 本）。"""
    vocab: list[list] = [[content, 0.0] for content in SPECIALS]
    vocab += [[f"<0x{value:02X}>", 0.0] for value in range(256)]
    vocab += [["▁こんにちは", -1.5], ["世界", -2.5]]
    raw = {
        "added_tokens": [
            {
                "id": index,
                "content": content,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
            for index, content in enumerate(SPECIALS)
        ],
        "normalizer": None,
        "pre_tokenizer": dict(tok.EXPECTED_PRE_TOKENIZER),
        "model": {"type": "Unigram", "unk_id": 0, "byte_fallback": True, "vocab": vocab},
    }
    raw.update(overrides)
    return raw


class TestUpstreamShape:
    """MUST: 「前提どおりのはず」を主張のままにしない — 外れると id 列だけが静かに割れる。"""

    def test_the_expected_shape_reports_the_byte_base_id(self):
        assert tok.check_upstream_shape(_raw()) == {
            "vocab": 273,
            "addedTokens": 15,
            "byteBaseId": tok.EXPECTED_BYTE_BASE_ID,
        }

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            (lambda raw: raw["model"].update(type="BPE"), "Unigram"),
            (lambda raw: raw["model"].update(unk_id=2), "unk_id"),
            (lambda raw: raw["model"].update(byte_fallback=False), "byte_fallback"),
            (lambda raw: raw.update(normalizer={"type": "Precompiled"}), "normalizer"),
            (lambda raw: raw["pre_tokenizer"].update(prepend_scheme="always"), "pre_tokenizer"),
            (lambda raw: raw["added_tokens"][0].update(normalized=True), "フラグ"),
            (lambda raw: raw["model"]["vocab"].append(["改\n行", 0.0]), "改行"),
            (lambda raw: raw["model"]["vocab"].append(["空 白", 0.0]), "空白"),
            (lambda raw: raw["model"]["vocab"].insert(0, ["<extra>", 0.0]), "バイト値"),
            (lambda raw: raw["model"]["vocab"].__delitem__(20), "256 本"),
        ],
    )
    def test_a_broken_shape_fails_loudly(self, mutate, message):
        raw = _raw()
        mutate(raw)

        with pytest.raises(SystemExit, match=message):
            tok.check_upstream_shape(raw)


class TestAsset:
    def test_the_vocab_text_has_one_line_per_score(self):
        asset = tok.build_asset(_raw(), TEXT_CONFIG)

        lines = asset["vocabText"].split("\n")
        assert lines == [
            *SPECIALS,
            *(f"<0x{value:02X}>" for value in range(256)),
            "▁こんにちは",
            "世界",
        ]
        assert len(asset["scores"]) == len(lines)
        assert asset["addedTokens"][:2] == [["<unk>", 0], ["<s>", 1]]
        assert [entry[1] for entry in asset["addedTokens"]] == sorted(
            entry[1] for entry in asset["addedTokens"]
        )

    def test_a_bos_id_that_disagrees_with_the_added_vocab_fails_loudly(self):
        """MUST: config だけを信じない — 別 id を前置する形は値としては通ってしまう。"""
        with pytest.raises(SystemExit, match="bos"):
            tok.build_asset(_raw(), {"bos_token_id": 5, "pad_token_id": 3})

    def test_a_pad_id_that_disagrees_with_the_added_vocab_fails_loudly(self):
        with pytest.raises(SystemExit, match="pad"):
            tok.build_asset(_raw(), {"bos_token_id": 1, "pad_token_id": 7})


class TestByteFallbackIds:
    """MUST: 期待値は語彙を見ずに UTF-8 から作る（恒真化の遮断）。"""

    def test_a_four_byte_emoji_expands_to_four_ids(self):
        ids = tok.byte_fallback_ids("\U0001f442")

        assert ids == [tok.EXPECTED_BYTE_BASE_ID + byte for byte in (0xF0, 0x9F, 0x91, 0x82)]

    def test_ascii_stays_one_id_per_byte(self):
        assert tok.byte_fallback_ids("ab") == [
            tok.EXPECTED_BYTE_BASE_ID + 0x61,
            tok.EXPECTED_BYTE_BASE_ID + 0x62,
        ]


class TestRuleCoverage:
    """MUST: 「規則を全部踏んでいる」を主張のままにしない（golden の穴の検出器）。"""

    @pytest.fixture(scope="class")
    @staticmethod
    def cases():
        pytest.importorskip("irodori_tts")
        from irodori_tts.text_normalization import normalize_text

        return tok.build_normalize_cases(normalize_text)

    def test_the_full_case_list_covers_every_rule(self, cases):
        coverage = tok.check_rule_coverage(cases)

        assert coverage["rules"] == 14
        assert coverage["brackets"] >= 3

    def test_dropping_the_case_that_fires_a_rule_fails_loudly(self, cases):
        thinned = [case for case in cases if case["name"] != "wave-dash"]

        with pytest.raises(SystemExit, match="発火"):
            tok.check_rule_coverage(thinned)

    def test_a_case_list_without_brackets_fails_loudly(self, cases):
        thinned = [case for case in cases if not case["why"].startswith("括弧剥がし")]

        with pytest.raises(SystemExit, match="被覆が薄い"):
            tok.check_rule_coverage(thinned)

    def test_the_normalized_field_is_not_stripped(self, cases):
        """MUST: strip はパイプライン段の責務（先頭空白ケースを潰さない）。"""
        by_name = {case["name"]: case for case in cases}
        assert by_name["ideographic-space"]["normalized"] == "前後ろ"
        # 生テキストを写していない（正規化が実際に効いている）ことも 1 本見ておく。
        assert by_name["question-bang"]["normalized"] == "本当?そうだ!"


class TestNfkcDiffTable:
    @pytest.fixture(scope="class")
    @staticmethod
    def table():
        return tok.nfkc_diff_table()

    def test_every_entry_is_a_real_difference(self, table):
        for key, value in table.items():
            source = chr(int(key))
            assert value != source
            assert unicodedata.normalize("NFKC", source) == value

    def test_known_compatibility_characters_are_present(self, table):
        # 全角 A / 半角カナ ｶ / 組文字 ㈱ — どれも NFKC で形が変わる代表。
        assert table[str(0xFF21)] == "A"
        assert table[str(0xFF76)] == "カ"
        assert table[str(0x3231)] == "(株)"

    def test_identity_codepoints_are_absent(self, table):
        for cp in (0x0041, 0x3042, 0x4E16, 0x1F600):
            assert str(cp) not in table

    def test_the_table_is_deterministic(self, table):
        assert tok.nfkc_diff_table() == table
        assert list(tok.nfkc_diff_table()) == list(table)


class TestCli:
    def test_the_default_out_dir_is_derived_from_the_weight_directory(self, tmp_path):
        out = tok.default_out_dir(tmp_path / "v4-small")

        assert out.name == "tokenizer"
        assert out.parent.name == "irodori-v4-small"

    def test_the_case_tables_have_unique_names(self):
        """名前が重なると golden の突合表が静かに 1 件消える。"""
        for table in (tok.ENCODE_CASES, tok.NORMALIZE_CASES):
            names = [name for name, _why, _text in table]
            assert len(names) == len(set(names))
        batch = [name for name, _why, _text, _key in tok.BATCH_CASES]
        assert len(batch) == len(set(batch))

    def test_the_expected_pre_tokenizer_is_not_mutated_by_callers(self):
        """`_raw` が `dict(...)` で渡す前提（共有参照だと fault injection が漏れる）。"""
        snapshot = copy.deepcopy(tok.EXPECTED_PRE_TOKENIZER)
        raw = _raw()
        raw["pre_tokenizer"]["prepend_scheme"] = "always"

        assert snapshot == tok.EXPECTED_PRE_TOKENIZER
