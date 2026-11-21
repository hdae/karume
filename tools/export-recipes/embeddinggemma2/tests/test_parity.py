"""Script-level contracts of the opt-in parity script (`embeddinggemma2/parity.py`) — the part that
needs neither the snapshot nor sentence-transformers.

- the sentence-transformers check refuses any release but the one the snapshot declares
- the comparison reads both sides as f32 vectors, and the summary keeps the worst case
- the pinned and unpinned read lists do not overlap, and the pinned one covers the export's
- the bf16 embedding scale is `sqrt(512)` rounded as bf16 (22.625)
"""

from __future__ import annotations

import pytest
import torch

from embeddinggemma2 import export, parity, sources


class TestSentenceTransformersVersion:
    def test_the_declared_release_passes(self) -> None:
        parity.assert_sentence_transformers_version(parity.REQUIRED_SENTENCE_TRANSFORMERS)

    @pytest.mark.parametrize("version", ["5.1.0", "6.0.0", "6.2.0"])
    def test_any_other_release_is_refused(self, version: str) -> None:
        with pytest.raises(parity.ParityScriptError, match=version):
            parity.assert_sentence_transformers_version(version)


class TestCompare:
    def test_identical_vectors(self) -> None:
        vector = torch.tensor([[0.6, 0.8]])

        assert parity.compare(vector, vector) == {"cosine": pytest.approx(1.0), "max_abs": 0.0}

    def test_a_bf16_side_is_read_as_f32(self) -> None:
        actual = torch.tensor([0.6, 0.8], dtype=torch.bfloat16)
        expected = torch.tensor([0.6, 0.8])

        result = parity.compare(actual, expected)

        assert result["max_abs"] == pytest.approx(float((actual.float() - expected).abs().max()))
        assert result["cosine"] == pytest.approx(1.0, abs=1e-4)

    def test_orthogonal_vectors(self) -> None:
        result = parity.compare(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]))

        assert result == {"cosine": 0.0, "max_abs": 1.0}

    def test_the_summary_keeps_the_worst_case(self) -> None:
        summary = parity.summarize(
            {
                "a": {"cosine": 0.99, "max_abs": 1e-3},
                "b": {"cosine": 0.999, "max_abs": 2e-3},
            }
        )

        assert summary == {"min_cosine": 0.99, "max_abs": 2e-3}


class TestReads:
    def test_the_pinned_reads_cover_the_export_and_are_pinned(self) -> None:
        assert set(export.READ_FILES) <= set(parity.PINNED_READS)
        assert set(parity.PINNED_READS) <= set(sources.PIN.sha256)

    def test_the_unpinned_reads_are_outside_the_pin(self) -> None:
        assert not set(parity.UNPINNED_READS) & set(sources.PIN.sha256)


def test_the_bf16_embedding_scale_is_sqrt_512_rounded() -> None:
    assert parity.BF16_EMBED_SCALE == 22.625
