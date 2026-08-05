"""次元文法の適合テスト。

正本は実装ではなく `packages/runtime/tests/fixtures/dim-grammar.json`（リポジトリ直下）で、TS 実装
（packages/runtime/src/format/dims.ts）と本実装は同じ表で検証する。表をコピーせず読み込むのは、
コピーが増えた瞬間に「片側だけ通る文法」が生まれるため。
"""

from __future__ import annotations

import pytest
from conftest import DIM_GRAMMAR, DIM_GRAMMAR_PATH

from karume.dims import (
    DimError,
    DimExpr,
    eval_dim,
    format_dim,
    is_symbol_name,
    parse_dim,
    try_parse_dim,
)


class TestFixtureTable:
    def test_the_shared_case_table_exists_and_has_all_three_sections(self):
        assert DIM_GRAMMAR_PATH.exists(), f"適合ケース表が無い: {DIM_GRAMMAR_PATH}"
        assert {"valid", "invalid", "eval"} <= set(DIM_GRAMMAR)


class TestCanonicalSpellings:
    """valid 節: 正準表記は分解でき、format との往復が文字列同一になる。"""

    @pytest.mark.parametrize("case", DIM_GRAMMAR["valid"], ids=lambda c: c["text"])
    def test_parses_into_the_declared_parts(self, case):
        assert parse_dim(case["text"]) == DimExpr(
            coeff=case["coeff"], sym=case["sym"], offset=case["offset"]
        )

    @pytest.mark.parametrize("case", DIM_GRAMMAR["valid"], ids=lambda c: c["text"])
    def test_roundtrips_through_format(self, case):
        assert format_dim(parse_dim(case["text"])) == case["text"]


class TestNonCanonicalSpellings:
    """invalid 節: 非正準表記は判定でも分解でも受理しない。"""

    @pytest.mark.parametrize("text", DIM_GRAMMAR["invalid"], ids=repr)
    def test_try_parse_returns_none(self, text):
        assert try_parse_dim(text) is None

    @pytest.mark.parametrize("text", DIM_GRAMMAR["invalid"], ids=repr)
    def test_parse_fails_loudly(self, text):
        with pytest.raises(DimError):
            parse_dim(text)


class TestBindingEvaluation:
    """eval 節: 束縛表の当て方（未束縛・非整数・負値は既定値で進めない）。"""

    @pytest.mark.parametrize("case", DIM_GRAMMAR["eval"], ids=lambda c: c["expr"])
    def test_evaluates_or_fails_as_the_table_declares(self, case):
        expr = parse_dim(case["expr"])
        if case.get("throws"):
            with pytest.raises(DimError):
                eval_dim(expr, case["bindings"])
        else:
            assert eval_dim(expr, case["bindings"]) == case["value"]


class TestSymbolNames:
    @pytest.mark.parametrize("name", ["T", "_s", "seq_len", "T2"])
    def test_identifier_shaped_names_are_symbols(self, name):
        assert is_symbol_name(name)

    @pytest.mark.parametrize("name", ["2T", "T+1", "", "T-1", "T "])
    def test_names_outside_the_grammar_are_rejected(self, name):
        assert not is_symbol_name(name)


class TestFormatGuards:
    """format_dim は非正準な部品を組み立てさせない（書ける表記を 1 通りに保つ）。"""

    @pytest.mark.parametrize(
        "expr",
        [
            DimExpr(coeff=0, sym="T", offset=0),
            DimExpr(coeff=1, sym="2T", offset=0),
            DimExpr(coeff=1, sym="T", offset=-1),
        ],
        ids=["coeff0", "bad-symbol", "negative-offset"],
    )
    def test_invalid_parts_fail_loudly(self, expr):
        with pytest.raises(DimError):
            format_dim(expr)
