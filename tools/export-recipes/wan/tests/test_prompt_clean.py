"""`wan/prompt_clean.py` の約束事 — 前処理の表の焼き方・BADNESS_RE の翻訳・鏡像と上流の一致。

ftfy / regex / diffusers が無い環境（`wan` グループを同期していない）では SKIP。全コードポイントの
掃引と 20 万本の乱択は `wan.umt5_tokenizer` の生成が回す（ここは同じ規則を小さく叩く）。
"""

from __future__ import annotations

import dataclasses
import html
import json
import re

import pytest

from wan import prompt_clean as pc


@pytest.fixture(scope="module")
def tables() -> pc.PromptCleanTables:
    pytest.importorskip("ftfy")
    pytest.importorskip("regex")
    return pc.build_tables()


@pytest.fixture(scope="module")
def compiled(tables: pc.PromptCleanTables) -> pc.CompiledTables:
    return pc.CompiledTables(tables)


@pytest.fixture(scope="module")
def truth():
    pytest.importorskip("diffusers")
    return pc.upstream_prompt_clean()


#: BMP の全コードポイント（サロゲートを除く）— 単体テストの掃引の幅。
BMP = [cp for cp in range(0x10000) if not 0xD800 <= cp <= 0xDFFF]


class TestBadnessTranslation:
    def test_the_literal_alphabet_is_the_449_characters_of_the_recon(self, tables):
        """調査 §3.3 の「字面の非 ASCII 449 文字」（C1 の 32 文字を含む）。"""
        literals = pc.badness_literals()

        assert len(literals) == 449
        assert set(range(0x80, 0xA0)) <= set(literals)

    def test_the_python_emission_matches_badness_re_on_every_probe(self, tables):
        """中間表現への畳み込みの同値（JS 版はエスケープの綴りだけが違う）。"""
        from ftfy.badness import BADNESS_RE

        emitted = re.compile(pc.emit_python(tables.badness))
        for _, template in pc.BADNESS_PROBES:
            texts = [template.replace("{}", chr(cp)) for cp in BMP]

            assert pc.badness_bits(emitted, texts) == pc.badness_bits(BADNESS_RE, texts), template

    def test_the_python_emission_matches_badness_re_on_the_full_pool(self, tables):
        from ftfy.badness import BADNESS_RE

        emitted = re.compile(pc.emit_python(tables.badness))
        pool = pc.expand(pc.fuzz_pools(tables)["full"])
        texts = list(pc.fuzz_strings(pool, seed=11, count=5_000, max_length=40))
        bits = pc.badness_bits(BADNESS_RE, texts)

        assert pc.badness_bits(emitted, texts) == bits
        assert bits.count("1") > 100, "池が判定を叩いていない"

    def test_the_js_source_is_ascii_made_of_escapes_classes_and_alternation(self, tables):
        """JS 側は `\\u{…}` の字面と範囲だけを見る（`\\w` や `.` をエンジンに解釈させない）。"""
        source = tables.to_json()["badness"]
        skeleton = re.sub(r"\\u\{[0-9A-F]+\}", "", source)

        assert source.isascii()
        assert set(skeleton) <= set("[]-|?^")

    def test_dot_becomes_everything_but_newline(self, tables):
        """JS の `.` は `\\r` と U+2028 / 2029 も除くので、Python の `.` を集合で書き下す。"""
        dots = [
            term
            for branch in tables.badness
            for term in branch
            if isinstance(term, pc.CharClass) and term.ranges == ((0, 9), (11, 0x10FFFF))
        ]

        assert len(dots) == 1


class TestTables:
    def test_baking_twice_gives_the_same_bytes(self, tables):
        again = pc.build_tables()

        assert json.dumps(again.to_json(), ensure_ascii=False) == json.dumps(
            tables.to_json(), ensure_ascii=False
        )

    def test_table_outputs_never_make_what_the_entrance_rejects(self, tables):
        """入口で 1 度見れば各周でも no-op、の根拠（表の出力に C1・未割り当てが無い）。"""
        pc.check_table_ranges(tables)
        broken = dataclasses.replace(tables, width={**tables.width, 0xFF21: "\x85"})

        with pytest.raises(ValueError, match="C1"):
            pc.check_table_ranges(broken)

    def test_the_quote_sets_are_disjoint_from_their_replacements(self, tables):
        broken = dataclasses.replace(tables, single_quotes=[[0x22, 0x22], *tables.single_quotes])

        with pytest.raises(ValueError, match="引用符"):
            pc.check_table_ranges(broken)

    def test_the_strip_and_collapse_sets_are_isspace_and_regex_whitespace(self, tables):
        assert len(pc.expand(tables.strip_space)) == 29
        assert len(pc.expand(tables.collapse_space)) == 25
        assert set(pc.expand(tables.collapse_space)) < set(pc.expand(tables.strip_space))

    def test_the_upstream_shape_is_pinned(self, tables):
        """表に落とせない正規表現の形・既定の設定を字面で縛る（版を上げて変われば焼かない）。"""
        pc.check_upstream_shape()


class TestEntityGates:
    def test_the_charref_probe_hits_exactly_where_html_unescape_matches(self, tables):
        """TS が持つ存在判定の形と `html._charref` の一致（全 BMP × 4 文脈）。"""
        probe = re.compile(pc.HTML_CHARREF_PROBE)
        charref = html._charref  # type: ignore[attr-defined]
        for prefix in ("&", "&#", "&#x", "&#X"):
            for cp in BMP:
                text = f"a{prefix}{chr(cp)}z"

                assert bool(probe.search(text)) == bool(charref.search(text)), (prefix, hex(cp))

    def test_text_that_passes_both_gates_is_left_alone_by_both_unescapes(self, tables):
        """門を通った入力では ftfy の unescape_html も html.unescape も何もしない。

        拒んだ処理が no-op になる、の根拠。
        """
        from ftfy.fixes import unescape_html

        ftfy_entity = re.compile(pc.FTFY_ENTITY_PATTERN)
        probe = re.compile(pc.HTML_CHARREF_PROBE)
        for template in ("&{}", "&{};", "&#{}", "&{}amp;", "x&{} &lt"):
            for cp in BMP:
                text = template.replace("{}", chr(cp))
                if ftfy_entity.search(text) or probe.search(text):
                    continue

                assert html.unescape(text) == text, ascii(text)
                assert unescape_html(text) == text, ascii(text)


class TestMirror:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("色调艳丽，过曝", "色调艳丽,过曝"),
            ("\u201cquoted\u201d it\u2019s", '"quoted" it\'s'),
            ("\u0149", "'n"),
            ("\x1b[\u0661mred", "red"),
            ("a\r\nb\rc\u2028d", "a b c d"),
            ("a &\rb", "a & b"),
        ],
    )
    def test_deterministic_fixes_match_upstream(self, compiled, truth, text, expected):
        assert truth(text) == expected
        assert pc.clean(compiled, text) == expected

    @pytest.mark.parametrize(
        ("text", "reason"),
        [
            ("caf\u00c3\u00a9", "mojibake"),
            ("a\u0300A\u0300", "mojibake"),
            ("salt &amp; pepper", "entity"),
            ("R&D", "entity"),
            ("\uff06amp;", "entity"),
            ("a\x85b", "c1"),
            ("a\u0378b", "unassigned"),
        ],
    )
    def test_heuristic_inputs_are_rejected_with_their_reason(self, compiled, text, reason):
        with pytest.raises(pc.PromptRejectedError) as raised:
            pc.clean(compiled, text)

        assert raised.value.reason == reason

    def test_the_segment_boundary_at_one_million_code_points_blocks_composition(
        self, compiled, truth
    ):
        """ftfy の区切り（1,000,000 コードポイント）をまたぐ結合文字は合成されない（実測）。"""
        split = " " * 999_999 + "e\u0301"
        joined = " " * 999_998 + "e\u0301"

        assert truth(split) == "e\u0301" == pc.clean(compiled, split)
        assert truth(joined) == "\u00e9" == pc.clean(compiled, joined)

    @pytest.mark.parametrize("pool", ["safe", "full", "entity"])
    def test_the_mirror_matches_upstream_on_a_small_fuzz(self, tables, compiled, truth, pool):
        run = pc.FuzzRun(pool, pool, seed=21, count=2_000)
        result = pc.verify_fuzz(compiled, truth, run, pc.expand(pc.fuzz_pools(tables)[pool]))

        assert sum(result["outcomes"].values()) == 2_000
        assert result["maxRounds"] <= pc.MAX_FIX_ROUNDS

    def test_the_safe_pool_is_never_rejected(self, tables, compiled, truth):
        run = pc.FuzzRun("safe", "safe", seed=22, count=5_000)
        result = pc.verify_fuzz(compiled, truth, run, pc.expand(pc.fuzz_pools(tables)["safe"]))

        assert result["outcomes"] == {"accepted": 5_000}

    def test_a_record_fails_loudly_when_mirror_and_upstream_disagree(self, compiled):
        with pytest.raises(ValueError, match="鏡像が上流"):
            pc.record(compiled, lambda _: "something else", "cat")

    def test_a_rejected_record_carries_the_reason(self, compiled):
        assert pc.record(compiled, lambda _: "unused", "R&D") == "Rentity"


class TestFuzzMachinery:
    def test_xorshift32_follows_the_13_17_5_recurrence(self):
        """seed 1 の先頭は手で追える（1 → 8193 → 8193 → 8193 ^ 262176 = 270369）。"""
        rng = pc.XorShift32(1)

        assert rng.next() == 270369
        assert rng.next() == 67634689

    def test_a_zero_seed_is_refused(self):
        with pytest.raises(ValueError):
            pc.XorShift32(0)

    def test_fuzz_strings_are_reproducible(self):
        pool = list(range(0x61, 0x7B))
        first = list(pc.fuzz_strings(pool, seed=3, count=50, max_length=8))

        assert first == list(pc.fuzz_strings(pool, seed=3, count=50, max_length=8))
        assert all(1 <= len(text) <= 8 for text in first)

    def test_digests_hash_each_record_followed_by_nul(self):
        import hashlib

        assert pc.digest(["A", "Rc1"]) == hashlib.sha256(b"A\x00Rc1\x00").hexdigest()
        assert pc.digest([]) == hashlib.sha256(b"").hexdigest()
        assert pc.chunked_digests(["a", "b", "c"], 2)[1] == pc.digest(["c"])[:16]
