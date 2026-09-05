"""`irodori/tokenizer_ref.py` の台本レベルの約束事（実資産不要分）。

実資産の emit は手動（`irodori/export.py` と同じ規律）。ここで固定するのは、壊れると
**沈黙の id 列不一致**になる側の規律だけ:

- 上流 `tokenizer.json` の形の門（model 種別 / unk_id / byte_fallback / normalizer /
  pre_tokenizer / 追加語彙のフラグ / バイトトークンの id 起点 / 行区切りの前提）
- `bosId` / `padId` がチェックポイント config と追加語彙の**両方**で一致すること
- byte_fallback の期待 id 列が UTF-8 から独立に計算されていること
- `build_encode_cases` / `build_batch_cases` / `TokenizerPair.encode` の fail-loudly 門
  （語彙内外の絵文字・2 経路の id 列一致・マスクの有効長・truncation の発火）
- ケース表の本文に「目に見えない文字」が生で現れていないこと
- `normalize_text` の規則被覆の実測が、ケースを削れば実際に落ちること
- NFKC 差分表が恒等エントリを含まず、2 回作れば同じであること
- パリティ用フィクスチャの語彙**部分集合**が格子の一致集合を覆っていること（抽出規則の
  故障注入 + `minScore` / `maxTokenLength` を全体から採っていること）
- caption ケースが strip だけを通り、正規化に**感受**すること（無感になれば門が落ちる）
"""

from __future__ import annotations

import ast
import copy
import json
import unicodedata
from pathlib import Path

import pytest
import torch

from irodori import tokenizer_ref as tok

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


#: どのケースにも現れない、語彙**全体**の最小スコアかつ最長のトークン。`minScore` /
#: `maxTokenLength` を部分集合から導く実装（= 未知ノードの重みと探索幅が変わる）を、
#: 全体から採る正しい実装と見分けるための素材。
UNREACHABLE_TOKEN = "宇宙のはて" * 5

#: 半角空白を挟んだケース 1 件（Metaspace の置換を通して初めて `▁こんにちは` に一致する）。
PARITY_CASES = [
    {"name": "greeting", "why": "", "raw": " こんにちは世界", "normalized": " こんにちは世界"}
]


def _raw_with_unreachable_token() -> dict:
    raw = _raw()
    raw["model"]["vocab"].append([UNREACHABLE_TOKEN, -9.5])
    return raw


class _FakePair:
    """`build_caption_cases` が触る面（`wrapper.batch_encode`）だけの身代わり。

    id はコードポイント（`[BOS, *本文]` を右 pad）で、語彙も Viterbi も持たない — ここで
    見たいのは「caption 側に strip しか掛かっていないか」だけ。
    """

    class _Wrapper:
        def batch_encode(self, texts, max_length: int):
            body = [ord(char) for char in texts[0]][: max_length - 1]
            ids = torch.full((1, max_length), 3, dtype=torch.int64)
            mask = torch.zeros((1, max_length), dtype=torch.bool)
            ids[0, : len(body) + 1] = torch.tensor([1, *body], dtype=torch.int64)
            mask[0, : len(body) + 1] = True
            return ids, mask

    def __init__(self) -> None:
        self.wrapper = _FakePair._Wrapper()


#: `build_caption_cases` が読む config の最小形。
CAPTION_MODEL_CONFIG = {"max_caption_len": 32}


def _fake_normalize(body: str) -> str:
    """`normalize_text` のうち caption ケースに効く段だけの身代わり。

    実物は上流にあり実資産を要する（`TestRuleCoverage` が importorskip でそちらを見る）。
    ここで要るのは「正規化を通すと id 列が動く」という性質だけなので、記号削除・外側括弧の
    剥がし・NFKC の 3 段を写す。
    """
    stripped = body.strip()
    for char in "①;　":
        stripped = stripped.replace(char, "")
    stripped = stripped.removeprefix("「").removesuffix("」")
    return unicodedata.normalize("NFKC", stripped)


class TestCaptionCases:
    """MUST: caption 側は strip だけ（上流 `_synthesize` は `normalize_text` を掛けない）。"""

    def test_the_packed_ids_come_from_the_stripped_text(self):
        cases = tok.build_caption_cases(_FakePair(), _fake_normalize, CAPTION_MODEL_CONFIG)
        by_name = {case["name"]: case for case in cases}
        brackets = by_name["caption-brackets"]

        assert brackets["stripped"] == "「若く元気な女性の声」"
        assert brackets["idsPacked"] == [1, *(ord(char) for char in brackets["stripped"])]
        assert brackets["normalizedIdsPacked"] == [1, *(ord(char) for char in "若く元気な女性の声")]
        assert brackets["normalizeSensitive"] is True

    def test_a_table_that_no_longer_reacts_to_normalization_fails_loudly(self):
        """恒真化の遮断: 正規化が恒等になれば 2 経路が一致して門の検出力が消える。"""
        with pytest.raises(SystemExit, match="正規化に感受する caption ケース"):
            tok.build_caption_cases(_FakePair(), lambda body: body, CAPTION_MODEL_CONFIG)


class _FakeRawTokenizer:
    """`tokenizers.Tokenizer` のうち `build_encode_cases` が触る面（`get_vocab`）だけ。"""

    def __init__(self, vocab) -> None:
        self._vocab = tuple(vocab)

    def get_vocab(self) -> dict[str, int]:
        return {content: index for index, content in enumerate(self._vocab)}


class _FakeEncodePair:
    """`build_encode_cases` が触る面（`raw.get_vocab` / `encode`）だけの身代わり。

    語彙に在る文字は 1 id（コードポイント）、無い文字は byte_fallback へ展開する — 実物の
    Unigram が持つ**分岐**だけを写し、Viterbi は持たない。`overrides` は特定ケースの id 列を
    差し替える口で、byte_fallback の実測門への故障注入に使う。
    """

    def __init__(self, *, vocab=(tok.IN_VOCAB_EMOJI,), overrides=None) -> None:
        self._vocab = set(vocab)
        self.raw = _FakeRawTokenizer(sorted(self._vocab))
        self._overrides = dict(overrides or {})

    def encode(self, text: str, where: str) -> list[int]:
        if where in self._overrides:
            return list(self._overrides[where])
        ids: list[int] = []
        for char in text:
            if char in self._vocab:
                ids.append(ord(char))
            else:
                ids.extend(tok.byte_fallback_ids(char))
        return ids


class TestEncodeCases:
    """MUST: b / c ケースが「語彙内 / 語彙外」であることを台本が実測してから golden を採る。"""

    def test_the_byte_fallback_case_expands_to_the_utf8_bytes(self):
        cases = tok.build_encode_cases(_FakeEncodePair(), lambda text: text)

        by_name = {case["name"]: case for case in cases}
        assert by_name["emoji-byte-fallback"]["ids"] == tok.byte_fallback_ids(
            tok.OUT_OF_VOCAB_EMOJI
        )
        assert len(by_name["emoji-byte-fallback"]["ids"]) == 4
        # 語彙内の側は 1 本のまま（b と c が同じ検査になっていない）。
        assert by_name["emoji-in-vocab"]["ids"].count(ord(tok.IN_VOCAB_EMOJI)) == 1
        assert [case["name"] for case in cases] == [name for name, _why, _text in tok.ENCODE_CASES]

    def test_an_in_vocab_emoji_that_is_absent_from_the_vocab_fails_loudly(self):
        with pytest.raises(SystemExit, match="語彙に無い"):
            tok.build_encode_cases(_FakeEncodePair(vocab=()), lambda text: text)

    def test_an_out_of_vocab_emoji_that_is_present_in_the_vocab_fails_loudly(self):
        pair = _FakeEncodePair(vocab=(tok.IN_VOCAB_EMOJI, tok.OUT_OF_VOCAB_EMOJI))

        with pytest.raises(SystemExit, match="byte_fallback を踏まない"):
            tok.build_encode_cases(pair, lambda text: text)

    def test_a_case_that_did_not_take_the_byte_fallback_path_fails_loudly(self):
        """恒真化の遮断: 期待値は語彙を見ずに UTF-8 から作るので、id 列がずれれば落ちる。"""
        pair = _FakeEncodePair(overrides={"emoji-byte-fallback": [7]})

        with pytest.raises(SystemExit, match="byte_fallback になっていない"):
            tok.build_encode_cases(pair, lambda text: text)


class _FakeEncoding:
    """`tokenizers` 側の戻り（`.ids` を持つ）だけの身代わり。"""

    def __init__(self, ids) -> None:
        self.ids = list(ids)


class _FakeRawEncoder:
    def __init__(self, ids) -> None:
        self._ids = list(ids)

    def encode(self, text: str, add_special_tokens: bool) -> _FakeEncoding:
        return _FakeEncoding(self._ids)


class _FakeHfEncoder:
    def __init__(self, ids) -> None:
        self._ids = list(ids)

    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        return list(self._ids)


def _encode_pair(raw_ids, hf_ids) -> tok.TokenizerPair:
    """`TokenizerPair` を上流 import 抜きで組む（`__init__` は実資産の clone を要する）。"""
    pair = object.__new__(tok.TokenizerPair)
    pair.raw = _FakeRawEncoder(raw_ids)
    pair.hf = _FakeHfEncoder(hf_ids)
    return pair


class TestTokenizerPairEncode:
    """MUST: golden は 2 経路（tokenizers 生 / transformers 経由）の一致を見てから採る。"""

    def test_two_paths_that_agree_return_that_id_list(self):
        assert tok.TokenizerPair.encode(_encode_pair([4, 5, 6], [4, 5, 6]), "あ", "ja-plain") == [
            4,
            5,
            6,
        ]

    def test_two_paths_that_disagree_fail_loudly(self):
        pair = _encode_pair([4, 5, 6], [4, 5])

        with pytest.raises(SystemExit, match="tokenizers と transformers"):
            tok.TokenizerPair.encode(pair, "あ", "ja-plain")


class _FakeBatchPair(_FakePair):
    """`build_batch_cases` が触る面（`wrapper.batch_encode` + `encode`）の身代わり。

    id は 1 文字 1 本なので、`encode` の長さと wrapper の有効長が**別々の経路**から同じ値に
    なる — 有効長の門が実装の写経にならない。
    """

    def encode(self, text: str, where: str) -> list[int]:
        return [ord(char) for char in text]


class _FakeShortMaskPair(_FakeBatchPair):
    """マスクの有効長だけを 1 本短く返す（有効長の門への故障注入）。"""

    class _Wrapper(_FakePair._Wrapper):
        def batch_encode(self, texts, max_length: int):
            ids, mask = super().batch_encode(texts, max_length)
            mask[0, int(mask[0].sum()) - 1] = False
            return ids, mask

    def __init__(self) -> None:
        super().__init__()
        self.wrapper = _FakeShortMaskPair._Wrapper()


#: `build_batch_cases` が読む config の最小形（`max_text_len` は長文ケースが必ず溢れる幅）。
BATCH_MODEL_CONFIG = {"max_text_len": 64, "max_caption_len": 128}


class TestBatchCases:
    """MUST: 切り詰めの順序（body を切ってから BOS）を golden に必ず現す。"""

    def test_the_truncating_case_fills_the_window_and_keeps_the_bos_first(self):
        cases = tok.build_batch_cases(_FakeBatchPair(), lambda text: text, BATCH_MODEL_CONFIG)

        by_name = {case["name"]: case for case in cases}
        truncated = by_name["batch-truncated"]
        assert truncated["maxLength"] == 64
        assert sum(truncated["mask"]) == 64
        assert truncated["idsPadded"][0] == 1
        # 溢れない側は BOS + 本文ちょうど（pad は mask の外）。
        short = by_name["batch-short"]
        assert sum(short["mask"]) == len(short["normalized"]) + 1
        assert by_name["batch-caption"]["maxLength"] == 128

    def test_a_mask_whose_effective_length_disagrees_with_the_body_fails_loudly(self):
        with pytest.raises(SystemExit, match="マスクの有効長"):
            tok.build_batch_cases(_FakeShortMaskPair(), lambda text: text, BATCH_MODEL_CONFIG)

    def test_a_case_list_where_truncation_never_fires_fails_loudly(self):
        roomy = {"max_text_len": 4096, "max_caption_len": 4096}

        with pytest.raises(SystemExit, match="truncation が発火するケース"):
            tok.build_batch_cases(_FakeBatchPair(), lambda text: text, roomy)


class TestParityFixture:
    """語彙の**部分集合**でも TS 側の Viterbi が全語彙と同じ格子を張ることの実測。"""

    def test_the_lattice_text_is_the_metaspace_replaced_normalized_text(self):
        assert tok.lattice_texts(PARITY_CASES, [], []) == ["▁こんにちは世界"]

    def test_the_subset_holds_every_token_the_lattice_can_reach(self):
        raw = _raw_with_unreachable_token()

        subset = tok.vocab_subset(
            raw["model"]["vocab"],
            tok.lattice_texts(PARITY_CASES, [], []),
            {content: index for index, content in enumerate(SPECIALS)},
        )

        tokens = {row[0] for row in subset}
        assert "▁こんにちは" in tokens
        assert "世界" in tokens
        # ケースに現れないトークンは載らない（部分集合であることそのもの）。
        assert UNREACHABLE_TOKEN not in tokens
        # バイト 256 本 + 追加語彙は無条件で載る。
        assert len([row for row in subset if row[0].startswith("<0x")]) == 256
        assert set(SPECIALS) <= tokens
        # id 昇順（emit のバイト決定性の前提）。
        assert [row[1] for row in subset] == sorted(row[1] for row in subset)

    def test_skipping_the_metaspace_replacement_loses_a_token(self):
        """MUST: 抽出は ▁ 置換**後**の文字列で回す — 素の normalized で回すと格子が痩せる。"""
        raw = _raw_with_unreachable_token()

        subset = tok.vocab_subset(raw["model"]["vocab"], [PARITY_CASES[0]["normalized"]], {})

        assert "▁こんにちは" not in {row[0] for row in subset}

    def test_a_subset_that_lost_a_reachable_token_fails_loudly(self):
        """抽出とは**逆向き**の検査なので、抽出の取りこぼしをここで捕まえられる。"""
        raw = _raw_with_unreachable_token()
        texts = tok.lattice_texts(PARITY_CASES, [], [])
        full = tok.vocab_subset(raw["model"]["vocab"], texts, {})
        thinned = [row for row in full if row[0] != "世界"]

        assert tok.check_subset_covers_lattice(raw["model"]["vocab"], texts, full) == 2
        with pytest.raises(SystemExit, match="覆えていない"):
            tok.check_subset_covers_lattice(raw["model"]["vocab"], texts, thinned)

    def test_the_global_min_score_and_width_are_not_the_subset_values(self):
        """MUST: 未知ノードの重みと探索幅は語彙**全体**から採る（部分集合から導かない）。"""
        raw = _raw_with_unreachable_token()

        fixture = tok.build_parity_fixture(
            tok.build_asset(raw, TEXT_CONFIG), raw["model"]["vocab"], PARITY_CASES, [], [], [], {}
        )

        assert fixture["asset"]["minScore"] == -9.5
        assert min(row[2] for row in fixture["asset"]["vocab"]) == -2.5
        assert fixture["asset"]["maxTokenLength"] == len(UNREACHABLE_TOKEN)
        assert max(len(row[0]) for row in fixture["asset"]["vocab"]) < len(UNREACHABLE_TOKEN)

    def test_the_fixture_is_deterministic(self):
        raw = _raw_with_unreachable_token()
        asset = tok.build_asset(raw, TEXT_CONFIG)

        def build() -> str:
            fixture = tok.build_parity_fixture(
                asset, raw["model"]["vocab"], PARITY_CASES, [], [], [], {"65": "A"}
            )
            return json.dumps(fixture, ensure_ascii=False)

        assert build() == build()


#: 本文の literal を採る対象のケース表（どれも `(名前, 理由, 本文, ...)` の並び）。
CASE_TABLE_NAMES = ("ENCODE_CASES", "BATCH_CASES", "CAPTION_CASES", "NORMALIZE_CASES")


def _case_body_literals() -> dict[str, str]:
    """ケース表の**本文**を、値ではなく台本に書かれた綴りのまま採る。

    値を見ても「生の U+3099」と「`\\u3099` と書いた U+3099」は区別できない（それが
    この MUST の要点）ので、`ast` で本文の位置を採り、ソースの当該範囲を読む。
    """
    source = Path(tok.__file__).read_text(encoding="utf-8")
    literals: dict[str, str] = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        else:
            continue
        if name not in CASE_TABLE_NAMES:
            continue
        for case in node.value.elts:
            literals[case.elts[0].value] = ast.get_source_segment(source, case.elts[2])
    return literals


def _invisible_codepoints(text: str) -> list[str]:
    """`text` に**生で**現れる目に見えない文字（結合文字 Mn / 書式文字 Cf / U+3000）。"""
    return [
        f"U+{ord(char):04X}"
        for char in text
        if char == "\u3000" or unicodedata.category(char) in ("Mn", "Cf")
    ]


class TestCaseTableLiterals:
    r"""MUST（`ENCODE_CASES` の docstring）: 目に見えない文字は必ず `\uXXXX` で書く。

    生で置くと、エディタの NFC 正規化 1 回で `か + U+3099` が合成済み `が` へ化け、
    trailing-whitespace 除去 1 回で末尾の U+3000 が消える。**どちらも golden を焼き直せば
    緑のまま通る**（ケースが検査対象を失ったことが誰にも見えない）ので、表の literal を
    静的に見る門をここに置く。判定は本文だけ — why 欄は `U+3099` を説明文として持つ。
    """

    def test_no_case_body_is_written_with_a_raw_invisible_character(self):
        literals = _case_body_literals()

        # 表 4 本ぶんが実際に採れている（走査対象が空になったら門が消える）。
        assert len(literals) == sum(
            len(table)
            for table in (
                tok.ENCODE_CASES,
                tok.BATCH_CASES,
                tok.CAPTION_CASES,
                tok.NORMALIZE_CASES,
            )
        )
        assert {name: _invisible_codepoints(text) for name, text in literals.items()} == {
            name: [] for name in literals
        }

    def test_the_escaped_cases_still_carry_the_characters_they_name(self):
        """恒真化の遮断: 表記を変えただけで値は 1 ビットも動いていない。"""
        by_name = {name: text for name, _why, text in tok.ENCODE_CASES}
        dakuten = by_name["combining-dakuten"]
        # 合成済み `が` ではなく `か` + 結合文字のまま（ケースが検査対象を保っている）。
        assert "\u304c" not in dakuten
        composed = "\u304c\u3063\u3053\u3044\u3044\u3058\u3083\u3093\u3051\u3093"
        assert unicodedata.normalize("NFKC", dakuten) == composed
        caption = {name: text for name, _why, text in tok.CAPTION_CASES}
        assert caption["caption-surrounding-space"].count("\u3000") == 3


class TestCli:
    def test_the_default_out_dir_is_derived_from_the_weight_directory(self, tmp_path):
        out = tok.default_out_dir(tmp_path / "v4-small")

        assert out.name == "tokenizer"
        assert out.parent.name == "irodori-v4-small"

    def test_the_case_tables_have_unique_names(self):
        """名前が重なると golden の突合表が静かに 1 件消える。"""
        for table in (tok.ENCODE_CASES, tok.NORMALIZE_CASES, tok.CAPTION_CASES):
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
