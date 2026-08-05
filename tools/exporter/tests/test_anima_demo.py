"""`anima_demo.py` / `karume.anima_text` の約束事。

この層が壊れると **id 列だけが静かに別物**になる（shape は合ったまま画が変わる）。ここで
固定するのは 2 系統:

- **commit 済みフィクスチャに対する再検査**（HF 資産があれば走る・無ければ SKIP）
  `AutoTokenizer`（正本）と Python 参照実装（= TS 実装の鏡像）の両方が、フィクスチャの
  id 列を再現すること。**再生成しなくても上流の tokenizer.json の変化に気づける**のがここ
  の存在意義で、emit 側の門（`build_cases`）とは走る時点が違う。
- **資産に依らない部分**（常に走る）— 焼いた表の引き方、追加語彙の切り出し、pre-token 走査、
  台本とフィクスチャのケース集合が一致していること。

網羅検査（全 1,112,064 コードポイント × 11 文脈の pre-token 走査、正規化畳み込みの全 cp +
乱択 200,000）は**再生成時に必ず走る門**（`anima_demo.build_assets`）で、数分かかるので
pytest には置かない。
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any

import pytest

import anima_demo
import anima_pipeline
from karume import anima_text as at

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_PATH = (
    REPO_ROOT / "packages" / "runtime" / "tests" / "fixtures" / "anima-text" / "parity.json"
)
FIXTURE: dict[str, Any] = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

TABLES = at.SpmTables.from_json(FIXTURE["t5"]["normalizer"])
CLASSES: dict[str, at.Ranges] = FIXTURE["qwen"]["classes"]
CASE_FOLD: dict[int, int] = {cp: target for cp, target in FIXTURE["qwen"]["caseFold"]}
NFC_SEGMENTS: at.Ranges = FIXTURE["qwen"]["nfcSegments"]


class TestFixtureShape:
    def test_the_fixture_is_not_empty(self):
        """取り違え / 生成失敗で cases が空になると、TS 側のループが 0 回になる。"""
        assert len(FIXTURE["cases"]) == len(anima_demo.PROMPT_CASES)
        assert FIXTURE["maxLength"] == anima_demo.MAX_LENGTH

    def test_the_committed_cases_match_the_script(self):
        """フィクスチャが台本より古びていないこと（id と本文の両方を見る）。"""
        want = [(name, text) for name, text, _ in anima_demo.PROMPT_CASES]
        got = [(case["id"], case["text"]) for case in FIXTURE["cases"]]

        assert got == want

    def test_case_ids_are_unique(self):
        ids = [case["id"] for case in FIXTURE["cases"]]

        assert len(set(ids)) == len(ids)

    def test_the_t5_search_width_comes_from_the_whole_vocabulary(self):
        """MUST: `minScore` / `maxTokenLength` は語彙**全体**の値（部分集合から導かない）。

        部分集合（実測 124 件）から計算すると未知ノードの重みと探索幅が変わり、別の分割に
        なる。全体由来なら部分集合の最小 / 最長を必ず超える。
        """
        subset = FIXTURE["t5"]["vocab"]

        assert FIXTURE["t5"]["minScore"] <= min(score for _, _, score in subset)
        assert FIXTURE["t5"]["maxTokenLength"] >= max(len(token) for token, _, _ in subset)

    def test_the_pipeline_prompts_are_covered_by_the_cases(self):
        """`anima_pipeline` の 2 本は Deno 側のクロスチェックが語彙部分集合で引く対象。

        ここが外れると、あちらは「語彙に無いトークン」という読み取りにくい形で落ちる。
        """
        texts = {case["text"] for case in FIXTURE["cases"]}

        assert anima_pipeline.PROMPT in texts
        assert anima_pipeline.NEGATIVE_PROMPT in texts


class TestNormalizerTables:
    """焼いた表の引き方（Precompiled の最短接頭辞勝ち）。"""

    def test_every_case_reproduces_the_recorded_normalization(self):
        for case in FIXTURE["cases"]:
            assert at.normalize_with_tables(TABLES, case["text"]) == case["t5Normalized"], case[
                "id"
            ]

    def test_the_shortest_matching_prefix_wins_and_the_rest_is_dropped(self):
        # A + U+0301 は規則があるので Á。もう 1 つ足しても Á のまま（3 文字目は捨てられる）。
        assert at.normalize_with_tables(TABLES, "Á") == "Á"
        assert at.normalize_with_tables(TABLES, "Á́") == "Á"
        # 3cp 規則 Ấ ではなく 2cp 接頭辞 Â が勝つ（最長一致で実装するとここで外れる）。
        assert at.normalize_with_tables(TABLES, "Ấ") == "Â"

    def test_clusters_of_six_bytes_or_more_skip_the_whole_replacement(self):
        text = "Ấ́"  # 1 + 2 + 2 + 2 = 7 バイト

        assert at.normalize_with_tables(TABLES, text) == text

    def test_a_cluster_of_exactly_six_bytes_still_skips_the_whole_replacement(self):
        """境界そのもの（`< 6` を `<= 6` にすると結合文字が黙って消える）。

        U+00A0(2) + U+0300(2) + U+0300(2) = ちょうど 6 バイト。上の 7 バイトのケースだけでは
        この 1 文字の書き換えが素通りする（TS 側も同様 — 故障注入で実測）。
        """
        text = "\u00a0\u0300\u0300"

        assert at.normalize_with_tables(TABLES, text) == " \u0300\u0300"

    def test_crlf_is_one_cluster_but_two_newlines_are_two(self):
        assert at.normalize_with_tables(TABLES, "\r\n") == " "
        assert at.normalize_with_tables(TABLES, "\n\n") == "  "


class TestRanges:
    def test_adjacent_codepoints_fold_into_one_interval(self):
        assert at.to_ranges([5, 6, 7, 10]) == [[5, 7], [10, 10]]

    def test_lookup_matches_the_source_set(self):
        source = {1, 2, 3, 9, 0x10FFFF}
        ranges = at.to_ranges(source)

        for cp in [0, 1, 2, 3, 4, 8, 9, 10, 0x10FFFE, 0x10FFFF]:
            assert at.in_ranges(ranges, cp) == (cp in source), hex(cp)


class TestAddedTokens:
    def test_leftmost_longest(self):
        added = ["<|endoftext|>", "<|end"]

        assert at.split_added_tokens("a<|endoftext|>b", added) == [
            ("a", False),
            ("<|endoftext|>", True),
            ("b", False),
        ]

    def test_a_partial_match_is_not_split_out(self):
        """`<|endoftext|x` は途中まで一致するだけ — 切り出すと id 列が別物になる。"""
        assert at.split_added_tokens("<|endoftext|x", ["<|endoftext|>"]) == [
            ("<|endoftext|x", False)
        ]

    def test_an_empty_added_token_is_rejected(self):
        """空文字は全位置で一致して走査が進まない（無限ループの芽）。"""
        with pytest.raises(ValueError, match="空文字"):
            at.split_added_tokens("abc", ["", "<|a"])


class TestNfcSegments:
    """正本の NFC と素の `unicodedata` が割れる領域（分節表）。"""

    def test_the_recorded_cases_are_reproduced_by_the_segment_table(self):
        """フィクスチャの `[入力, 正本の出力]` を分節表で再現できる（TS 実装と同じ対）。"""
        cases = FIXTURE["qwen"]["nfcCases"]

        assert len(cases) >= 200
        for text, want in cases:
            assert at.nfc_with_segments(text, NFC_SEGMENTS) == want, [hex(ord(c)) for c in text]

    def test_the_plain_nfc_disagrees_with_the_source_of_truth(self):
        """実測: 正本は U+089A を結合クラス 0 と見なすので並べ替えない。

        分節表から該当区間を落とすと素の NFC と同じ（= 別の id 列）に戻る。
        """
        text = "\u0818\u089a"

        assert unicodedata.normalize("NFC", text) == "\u089a\u0818"
        assert at.nfc_with_segments(text, NFC_SEGMENTS) == text

        dropped = [r for r in NFC_SEGMENTS if not r[0] <= 0x089A <= r[1]]

        assert at.nfc_with_segments(text, dropped) == "\u089a\u0818"


class TestByteEncoder:
    def test_the_table_is_a_bijection_over_256_bytes(self):
        table = at.bytes_to_unicode()

        assert len(table) == 256
        assert len(set(table.values())) == 256

    def test_no_mapped_character_is_whitespace(self):
        """merges の鍵は `"左 右"` の 1 本の文字列 — 空白が混ざると分割が壊れる。"""
        table = at.bytes_to_unicode()

        assert not any(ch.isspace() for ch in table.values())


class TestQwenPreTokenize:
    def test_digits_split_one_by_one(self):
        """`\\p{N}` には `+` が無い（`2024` は 4 トークンになる）。"""
        assert at.qwen_pre_tokenize("2024", CLASSES, CASE_FOLD) == ["2", "0", "2", "4"]

    def test_a_leading_space_joins_the_following_word(self):
        assert at.qwen_pre_tokenize("a cat", CLASSES, CASE_FOLD) == ["a", " cat"]

    def test_trailing_spaces_keep_the_last_one_for_the_next_word(self):
        """`\\s+(?!\\S)` — 非空白が続くなら最後の 1 文字を次へ残す。"""
        assert at.qwen_pre_tokenize("a   b", CLASSES, CASE_FOLD) == ["a", "  ", " b"]

    def test_the_scan_never_stalls(self):
        """進めない位置があれば例外（黙って 1 文字進める実装にしない）。"""
        for text in ("", " ", "\r\n", "\u00a0", "\U0001f408", "'", "'s", "'S"):
            at.qwen_pre_tokenize(text, CLASSES, CASE_FOLD)

    def test_the_case_fold_table_is_not_ascii_only(self):
        """実測: 正本の `(?i:)` は simple case folding — U+017F（ſ）が s と同一視される。

        ASCII の大小反転で済ませると `it\u017fs` の `'\u017f` が切り出されず 1 断片になり、
        正本の `'\u017f` + `s` と切れ目が変わる。表から U+017F を落とすとここが落ちる。
        """
        assert CASE_FOLD[0x017F] == ord("s")
        assert CASE_FOLD[ord("S")] == ord("s")
        assert at.qwen_pre_tokenize("it'\u017fs", CLASSES, CASE_FOLD) == ["it", "'\u017f", "s"]

        # 表を ASCII だけに削ると（= 移植で最も自然な思い違い）切れ目が変わる。
        ascii_only = {cp: t for cp, t in CASE_FOLD.items() if cp < 0x80}

        assert at.qwen_pre_tokenize("it'\u017fs", CLASSES, ascii_only) == ["it", "'\u017fs"]


# ---- 実 tokenizer.json があるときだけ走る突合 --------------------------------

REPO = anima_demo.DEFAULT_REPO


def _cached(filename: str) -> Path | None:
    """HF キャッシュにあるファイル。無ければ None（**ダウンロードしない**）。"""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None
    try:
        return Path(hf_hub_download(REPO, filename, local_files_only=True))
    except Exception:
        return None


QWEN_JSON = _cached("tokenizer/tokenizer.json")
T5_JSON = _cached("t5_tokenizer/tokenizer.json")
needs_tokenizers = pytest.mark.skipif(
    QWEN_JSON is None or T5_JSON is None,
    reason=f"{REPO} の tokenizer.json が HF キャッシュに無い（再生成: README の anima_demo.py）",
)


@pytest.fixture(scope="module")
def built() -> dict[str, Any]:
    """フィクスチャの表 + 実 tokenizer.json の語彙で参照実装を組む。

    表そのものは**再生成しない**（全コードポイント走査で数分かかる）。ここが見たいのは
    「commit 済みの表と実語彙で正本の id 列が再現できるか」で、表の畳み込みの同値は
    `anima_demo.build_assets` が emit のたびに検査する。
    """
    assert QWEN_JSON is not None and T5_JSON is not None
    source = at.AnimaTokenizerSource(QWEN_JSON, T5_JSON)
    qwen_raw, t5_raw = source.qwen_json(), source.t5_json()
    return {
        "qwen": {"addedTokens": FIXTURE["qwen"]["addedTokens"]},
        "t5": {"unkId": t5_raw["model"]["unk_id"], "eosId": FIXTURE["t5"]["eosId"]},
        "classes": CLASSES,
        "caseFold": CASE_FOLD,
        "nfcSegments": NFC_SEGMENTS,
        "tables": TABLES,
        "qwenVocab": qwen_raw["model"]["vocab"],
        "qwenMerges": [(left, right) for left, right in qwen_raw["model"]["merges"]],
        "t5Vocab": [(token, float(score)) for token, score in t5_raw["model"]["vocab"]],
    }


@needs_tokenizers
class TestReferenceParity:
    def test_the_reference_implementation_reproduces_the_fixture(self, built):
        """TS 実装の鏡像がフィクスチャの id 列を出す（TS が写経すべき対象の確認）。"""
        qwen_ref, t5_ref = anima_demo.references(
            {**built, "t5": {**built["t5"], "addedTokens": FIXTURE["t5"]["addedTokens"]}}
        )
        for case in FIXTURE["cases"]:
            assert qwen_ref.encode(case["text"], anima_demo.MAX_LENGTH) == case["qwenIds"], case[
                "id"
            ]
            assert t5_ref.encode(case["text"], anima_demo.MAX_LENGTH) == case["t5Ids"], case["id"]

    def test_the_fixture_still_matches_the_upstream_tokenizers(self):
        """正本（AutoTokenizer）が動いていないこと。動いたらフィクスチャの再生成が要る。"""
        transformers = pytest.importorskip("transformers")
        # NOTE: `local_files_only=True` は使えない — `special_tokens_map.json` のような
        # **任意**ファイルの解決に失敗して OSError になる（実測。キャッシュに tokenizer.json も
        # tokenizer_config.json も在るのに構成できない）。ネットワークが無い環境は SKIP。
        try:
            truth = {
                "qwen": transformers.AutoTokenizer.from_pretrained(REPO, subfolder="tokenizer"),
                "t5": transformers.AutoTokenizer.from_pretrained(REPO, subfolder="t5_tokenizer"),
            }
        except OSError as cause:
            pytest.skip(f"AutoTokenizer を構成できない（オフライン環境）: {cause}")

        for case in FIXTURE["cases"]:
            assert anima_demo.reference_ids(truth["qwen"], case["text"]) == case["qwenIds"], case[
                "id"
            ]
            assert anima_demo.reference_ids(truth["t5"], case["text"]) == case["t5Ids"], case["id"]
