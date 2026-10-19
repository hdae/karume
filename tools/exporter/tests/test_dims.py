"""Conformance tests of the dimension grammar.

The source of truth is not the implementation but
`packages/runtime/tests/fixtures/dim-grammar.json` (from the repository root); the TS
implementation (packages/runtime/src/format/dims.ts) and this one are checked against the same
table. The table is loaded rather than copied because the moment a copy exists, a "grammar that
passes on one side only" appears.
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
        assert DIM_GRAMMAR_PATH.exists(), f"conformance case table is missing: {DIM_GRAMMAR_PATH}"
        assert {"valid", "invalid", "eval"} <= set(DIM_GRAMMAR)


class TestCanonicalSpellings:
    """valid section: a canonical spelling parses, and the round trip through format is
    string-identical."""

    @pytest.mark.parametrize("case", DIM_GRAMMAR["valid"], ids=lambda c: c["text"])
    def test_parses_into_the_declared_parts(self, case):
        assert parse_dim(case["text"]) == DimExpr(
            coeff=case["coeff"], sym=case["sym"], offset=case["offset"]
        )

    @pytest.mark.parametrize("case", DIM_GRAMMAR["valid"], ids=lambda c: c["text"])
    def test_roundtrips_through_format(self, case):
        assert format_dim(parse_dim(case["text"])) == case["text"]


class TestNonCanonicalSpellings:
    """invalid section: a non-canonical spelling is accepted neither by the check nor by the
    parse."""

    @pytest.mark.parametrize("text", DIM_GRAMMAR["invalid"], ids=repr)
    def test_try_parse_returns_none(self, text):
        assert try_parse_dim(text) is None

    @pytest.mark.parametrize("text", DIM_GRAMMAR["invalid"], ids=repr)
    def test_parse_fails_loudly(self, text):
        with pytest.raises(DimError):
            parse_dim(text)


class TestBindingEvaluation:
    """eval section: how the bindings apply (an unbound, non-integer or negative value does not
    proceed with a default)."""

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
    """format_dim does not assemble non-canonical parts (keeps exactly one writable spelling)."""

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
