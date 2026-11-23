"""The pure part of `embeddinggemma2/measure_quant.py` (what needs no real weights).

The measurement itself is manual (the real weights on CPU). What is pinned here is the side whose
breakage makes the table silently lie:

- the retrieval pairs are the corpus' adjacent query → document entries, and a malformed corpus
  fails instead of pairing the wrong texts
- the Matryoshka slice is re-normalised, and the short / long split is at T 2,048
- the retrieval metrics: own top-1, agreement with the reference, the rank inversion rate
- the size arithmetic of bf16 / i8 / i4
- the rounding reaches exactly the intended weight group and leaves the other bit for bit
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from embeddinggemma2 import export
from embeddinggemma2 import measure_quant as mq
from embeddinggemma2.calib_texts import CALIB_TEXTS


class TestQueryDocumentPairs:
    def test_a_query_followed_by_a_document_is_one_pair(self) -> None:
        entries = [
            ("query", "q"),
            ("document", "d"),
            (None, "bare"),
            ("query", "q2"),
            ("document", "d2"),
        ]

        assert mq.query_document_pairs(entries) == ((0, 1), (3, 4))

    def test_the_corpus_has_twenty_pairs_and_eight_bare_texts(self) -> None:
        pairs = mq.query_document_pairs(CALIB_TEXTS)

        assert len(pairs) == 20
        assert sum(kind is None for kind, _body in CALIB_TEXTS) == 8

    def test_a_query_without_its_document_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not followed by a document"):
            mq.query_document_pairs([("query", "q"), (None, "bare")])

    def test_a_document_without_its_query_is_refused(self) -> None:
        with pytest.raises(ValueError, match="does not follow a query"):
            mq.query_document_pairs([(None, "bare"), ("document", "d")])


class TestCorpusTexts:
    def test_golden_cases_come_first_then_the_corpus_with_unique_names(self) -> None:
        texts = mq.corpus_texts()
        names = [text.name for text in texts]

        assert len(texts) == len(export.GOLDEN_CASES) + len(CALIB_TEXTS) == 58
        assert names[: len(export.GOLDEN_CASES)] == [name for name, _k, _b in export.GOLDEN_CASES]
        assert len(set(names)) == len(names)

    def test_the_pairs_point_at_a_query_and_its_document(self) -> None:
        texts = mq.corpus_texts()

        for query, document in mq.retrieval_pairs():
            assert texts[query].prompt == export.QUERY
            assert texts[document].prompt == export.DOCUMENT
            assert document == query + 1


class TestMatryoshka:
    def test_the_leading_slice_is_renormalised(self) -> None:
        vectors = torch.tensor([[3.0, 4.0, 12.0]])

        sliced = mq.matryoshka(vectors, 2)

        assert torch.allclose(sliced, torch.tensor([[0.6, 0.8]], dtype=torch.float64))

    def test_a_width_beyond_the_vector_is_refused(self) -> None:
        with pytest.raises(ValueError, match="outside"):
            mq.matryoshka(torch.ones(1, 4), 8)


class TestCosineSummary:
    def test_long_starts_at_2048_tokens(self) -> None:
        cosines = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float64)

        summary = mq.cosine_summary(cosines, [2047, 2048, 12])

        assert summary["short"] == {
            "texts": 2,
            "min": pytest.approx(0.7),
            "mean": pytest.approx(0.8),
        }
        assert summary["long"] == {
            "texts": 1,
            "min": pytest.approx(0.8),
            "mean": pytest.approx(0.8),
        }
        assert summary["all"]["min"] == pytest.approx(0.7)

    def test_an_empty_group_fails_loudly(self) -> None:
        with pytest.raises(ValueError, match="no long text"):
            mq.cosine_summary(torch.ones(2, dtype=torch.float64), [10, 20])


class TestRetrievalMetrics:
    REFERENCE = torch.tensor([[0.9, 0.5, 0.1], [0.2, 0.8, 0.3], [0.1, 0.4, 0.7]])

    def test_the_reference_against_itself(self) -> None:
        assert mq.own_top1(self.REFERENCE) == 3
        assert mq.top1_agreement(self.REFERENCE, self.REFERENCE) == 3
        assert mq.rank_inversion_rate(self.REFERENCE, self.REFERENCE) == 0.0

    def test_one_swapped_pair_is_one_inversion_among_nine_triples(self) -> None:
        """3 queries × C(3, 2) document pairs = 9 triples; swapping documents 0 and 1 of query 0
        flips exactly one of them and moves its top-1."""
        swapped = self.REFERENCE.clone()
        swapped[0] = torch.tensor([0.5, 0.9, 0.1])

        assert mq.rank_inversion_rate(swapped, self.REFERENCE) == pytest.approx(1 / 9)
        assert mq.top1_agreement(swapped, self.REFERENCE) == 2
        assert mq.own_top1(swapped) == 2

    def test_a_monotone_change_of_the_scores_inverts_nothing(self) -> None:
        assert mq.rank_inversion_rate(self.REFERENCE * 0.5 - 0.1, self.REFERENCE) == 0.0


class TestSizeArithmetic:
    STATS = mq.GroupStats(tensors=1, elements=64, rows=2)

    def test_bf16_is_sixteen_bits_per_element(self) -> None:
        assert mq.projected_bits("bf16", self.STATS) == 16 * 64

    def test_i8_adds_one_f32_scale_per_row(self) -> None:
        assert mq.projected_bits("i8", self.STATS) == 8 * 64 + 32 * 2

    def test_i4_adds_one_f32_scale_per_group_of_32(self) -> None:
        assert mq.projected_bits("i4", self.STATS) == 4 * 64 + 32 * 2

    def test_i4_refuses_a_partial_group(self) -> None:
        with pytest.raises(ValueError, match="groups of 32"):
            mq.projected_bits("i4", mq.GroupStats(tensors=1, elements=48, rows=1))

    def test_the_total_adds_the_f32_remainder(self) -> None:
        config = mq.Config("x", linear="i4", table="i8")

        size = mq.size_projection(config, self.STATS, self.STATS, remainder_elements=10)

        assert size["bytes"] == (4 * 64 + 32 * 2 + 8 * 64 + 32 * 2 + 32 * 10) // 8


class TinyTower(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 32)
        self.fc = nn.Linear(32, 4, bias=False)


class TestApplyWeights:
    @pytest.mark.parametrize("config", mq.CONFIGS, ids=lambda config: config.name)
    def test_exactly_the_intended_groups_are_rounded(self, config: mq.Config) -> None:
        torch.manual_seed(0)
        model = TinyTower()
        before = {name: tensor.clone() for name, tensor in model.state_dict().items()}

        rounded = mq.apply_weights(model, config, frozenset({"fc"}))

        expected = mq.expected_rounded(config, ["fc.weight"], ["embed_tokens.weight"])
        mq.assert_rounded(rounded, expected, config.name)
        for name, tensor in model.state_dict().items():
            assert torch.equal(tensor, before[name]) == (name not in expected)

    def test_a_missed_weight_is_reported(self) -> None:
        with pytest.raises(AssertionError, match="missing"):
            mq.assert_rounded(["fc.weight"], {"fc.weight", "embed_tokens.weight"}, "x")

    def test_the_a8_configuration_is_i8_all_plus_activations(self) -> None:
        by_name = {config.name: config for config in mq.CONFIGS}

        assert by_name["i8-all+a8"].act_quant
        assert (by_name["i8-all+a8"].linear, by_name["i8-all+a8"].table) == ("i8", "i8")
        assert not any(config.act_quant for name, config in by_name.items() if name != "i8-all+a8")
