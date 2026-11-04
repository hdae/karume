"""The contract of `wan/prompt_clean.py` — how the preprocessing tables are baked, the translation
of BADNESS_RE, and the agreement between the mirror and upstream.

SKIPs in an environment without ftfy / regex / diffusers (the `wan` group not synced). The sweep of
every code point and the 200,000 random samples are run by the generation in `wan.umt5_tokenizer`
(this file exercises the same rules on a small scale).
"""

from __future__ import annotations

import dataclasses
import hashlib
import html
import json
import re
from pathlib import Path

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


#: Every BMP code point (surrogates excluded) — the sweep range of the unit tests.
BMP = [cp for cp in range(0x10000) if not 0xD800 <= cp <= 0xDFFF]


class TestBadnessTranslation:
    def test_the_literal_alphabet_is_the_449_characters_of_the_recon(self, tables):
        """The "449 literal non-ASCII characters" of survey §3.3 (including the 32 C1
        characters)."""
        literals = pc.badness_literals()

        assert len(literals) == 449
        assert set(range(0x80, 0xA0)) <= set(literals)

    def test_the_python_emission_matches_badness_re_on_every_probe(self, tables):
        """Equivalence of the folding into the intermediate representation (the JS version differs
        only in how the escapes are spelled)."""
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
        assert bits.count("1") > 100, "the pool does not exercise the matcher"

    def test_the_js_source_is_ascii_made_of_escapes_classes_and_alternation(self, tables):
        """The JS side sees only `\\u{…}` literals and ranges (`\\w` and `.` are never left to the
        engine to interpret)."""
        source = tables.to_json()["badness"]
        skeleton = re.sub(r"\\u\{[0-9A-F]+\}", "", source)

        assert source.isascii()
        assert set(skeleton) <= set("[]-|?^")

    def test_dot_becomes_everything_but_newline(self, tables):
        """JS `.` also excludes `\\r` and U+2028 / 2029, so Python's `.` is written out as a set."""
        dots = [
            term
            for branch in tables.badness
            for term in branch
            if isinstance(term, pc.CharClass) and term.ranges == ((0, 9), (11, 0x10FFFF))
        ]

        assert len(dots) == 1


#: The TS mirror that pins the same sha256 (from the repository root).
TS_PROMPT_CLEAN = (
    Path(__file__).resolve().parents[4] / "packages/models/src/wan/text/prompt-clean.ts"
)


class TestTheBadnessPin:
    def test_the_writer_pins_the_value_the_ts_reader_pins(self):
        """One value in two languages — a bump on one side only would ship an asset the reader
        refuses (or a writer check that guards nothing)."""
        pinned = re.findall(
            r'^const FTFY_BADNESS_SHA256 = "([0-9a-f]{64})";$',
            TS_PROMPT_CLEAN.read_text(encoding="utf-8"),
            re.MULTILINE,
        )

        assert pinned == [pc.FTFY_BADNESS_SHA256]

    def test_the_writer_refuses_a_translation_other_than_the_pinned_one(self, tables):
        """A changed translation fails at the writer, not first as a red TS test."""
        changed = dataclasses.replace(tables, badness=tables.badness[:-1])

        with pytest.raises(ValueError, match="not the pinned"):
            changed.to_json()

    def test_the_current_translation_is_the_pinned_one(self, tables):
        source = tables.to_json()["badness"]

        assert hashlib.sha256(source.encode("utf-8")).hexdigest() == pc.FTFY_BADNESS_SHA256


class TestTables:
    def test_baking_twice_gives_the_same_bytes(self, tables):
        again = pc.build_tables()

        assert json.dumps(again.to_json(), ensure_ascii=False) == json.dumps(
            tables.to_json(), ensure_ascii=False
        )

    def test_table_outputs_never_make_what_the_entrance_rejects(self, tables):
        """The basis for "checked once at the entrance, a no-op on every round": the table outputs
        contain no C1 and no unassigned code points."""
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
        """Binds the regex shapes that cannot become tables and the default settings literally (if
        a version upgrade changes them, nothing is baked)."""
        pc.check_upstream_shape()


class TestEntityGates:
    def test_the_charref_probe_hits_exactly_where_html_unescape_matches(self, tables):
        """The presence check the TS side has agrees with `html._charref` (all of the BMP × 4
        contexts)."""
        probe = re.compile(pc.HTML_CHARREF_PROBE)
        charref = html._charref  # type: ignore[attr-defined]
        for prefix in ("&", "&#", "&#x", "&#X"):
            for cp in BMP:
                text = f"a{prefix}{chr(cp)}z"

                assert bool(probe.search(text)) == bool(charref.search(text)), (prefix, hex(cp))

    def test_text_that_passes_both_gates_is_left_alone_by_both_unescapes(self, tables):
        """For an input that passed the checks, neither ftfy's unescape_html nor html.unescape does
        anything.

        The basis for the rejected processing being a no-op.
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
        """A combining character that straddles ftfy's boundary (1,000,000 code points) is not
        composed (measured)."""
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
        """The start for seed 1 can be traced by hand (1 → 8193 → 8193 → 8193 ^ 262176 = 270369)."""
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
