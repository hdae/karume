"""py_skeleton: a comment-only or docstring-only change keeps the skeleton; any change to code or
string constants changes it. In strings mode the constants are lifted out and listed, and a changed
constant that held no Japanese is a code change."""

import pytest
from py_skeleton import LITERAL_MARK, LiteralChange, analyze, literal_changes, skeleton_of


class TestSkeletonOf:
    def test_same_when_only_comments_and_whitespace_differ(self):
        before = "# 日本語の注記\nx = 1  # 補足\n"
        after = "# English note\nx   =   1\n"
        assert skeleton_of("a.py", before) == skeleton_of("a.py", after)

    def test_same_when_only_docstrings_differ(self):
        def source(note: str) -> str:
            return (
                f'"""{note}"""\n\n\ndef f():\n    """{note}"""\n    return 1\n\n\n'
                f'class C:\n    """{note}"""\n\n    x = 1\n'
            )

        assert skeleton_of("d.py", source("注記")) == skeleton_of("d.py", source("note"))

    def test_a_function_whose_body_is_only_a_docstring_keeps_its_skeleton(self):
        assert skeleton_of("d.py", 'def f():\n    """注記"""\n') == skeleton_of(
            "d.py", 'def f():\n    """note"""\n'
        )

    def test_differs_when_the_code_changes(self):
        assert skeleton_of("a.py", "x = 1\n") != skeleton_of("a.py", "x = 2\n")

    def test_differs_when_a_string_constant_changes(self):
        before = 'raise ValueError("日本語の文言")\n'
        after = 'raise ValueError("English message")\n'
        assert skeleton_of("e.py", before) != skeleton_of("e.py", after)

    def test_a_string_statement_that_is_not_first_is_not_a_docstring(self):
        before = 'def f():\n    x = 1\n    "日本語"\n'
        after = 'def f():\n    x = 1\n    "English"\n'
        assert skeleton_of("s.py", before) != skeleton_of("s.py", after)

    def test_ignores_a_formatter_wrap(self):
        before = 'f(\n    a,\n    "x",\n)\n'
        after = 'f(a, "x")\n'
        assert skeleton_of("w.py", before) == skeleton_of("w.py", after)

    def test_raises_when_the_source_does_not_parse(self):
        with pytest.raises(SyntaxError, match=r"cannot parse bad\.py"):
            skeleton_of("bad.py", "def = \n")


class TestAnalyzeInStringsMode:
    @staticmethod
    def strings(source: str):
        return analyze("s.py", source, "strings")

    def test_lifts_string_constants_in_source_order_and_drops_docstrings(self):
        source = '"""doc"""\nraise ValueError("日本語の文言")\nk = "key"\n'
        result = self.strings(source)
        assert result.literals == ("日本語の文言", "key")
        assert repr(LITERAL_MARK) in result.text
        assert "doc" not in result.text

    def test_lifts_an_fstring_as_one_cell_and_keeps_its_expressions_as_code(self):
        def source(note: str) -> str:
            return f'm = f"{note} {{count}} 個"\n'

        before = self.strings(source("合計"))
        after = self.strings(source("total"))
        assert before.text == after.text
        assert before.literals == ("合計 ${} 個",)
        assert self.strings('m = f"合計 {count + 1} 個"\n').text != before.text

    def test_keeps_a_format_spec_as_code(self):
        a = self.strings('m = f"{x:.2f} 個"\n')
        b = self.strings('m = f"{x:.3f} 個"\n')
        assert a.text != b.text
        assert a.literals == ("${} 個",)

    def test_bytes_and_other_constants_stay_in_the_skeleton(self):
        assert self.strings('x = b"ab"\n').text != self.strings('x = b"cd"\n').text
        assert self.strings("x = 1\n").text != self.strings("x = 2\n").text

    def test_lists_the_constants_that_changed_and_whether_the_old_text_held_japanese(self):
        before = self.strings('f("日本語", "same", "english")\n')
        after = self.strings('f("Japanese", "same", "English")\n')
        assert literal_changes(before, after) == [
            LiteralChange(index=0, before="日本語", after="Japanese", japanese_before=True),
            LiteralChange(index=2, before="english", after="English", japanese_before=False),
        ]

    def test_refuses_to_align_constants_when_the_skeletons_differ(self):
        with pytest.raises(ValueError, match="the skeletons differ"):
            literal_changes(self.strings('f("a")\n'), self.strings('g("a")\n'))
        with pytest.raises(ValueError, match="the skeletons differ"):
            literal_changes(self.strings('f("a")\n'), self.strings('f("a", "b")\n'))
