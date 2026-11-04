"""The QAT distribution plan (`gemma4_qat.distribution.qat_plan`) — unit tests of one assembly
round.

The real checkpoint (2.4GiB) is not used. The input reaching the plan is a few-KB **valid minimal
series** ({@link gemma4_qat.tests.series_fixture}), and only the cases that check a rejection use
variants of it.

The core is "compare at assembly time the same facts held by separate scripts" — the fixed writer
(the `reference.json` `export.py` writes) / the packed PLE (`ple.py`) / the tokenizer asset / the
upstream config can move independently, so a mismatch can only be caught **before the
distribution is laid out**.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from gemma4_qat.config import (
    DEFAULT_CAPACITY,
    DEFAULT_CHUNK_LENGTH,
    MAX_CHUNK_LENGTH,
    REFERENCE_SCHEMA,
    SOURCES,
)
from gemma4_qat.distribution import qat_plan
from gemma4_qat.tests import series_fixture as fixture
from karume.dist import DistError


def _series(root: Path, model: str = "e2b", **overrides: Any) -> Path:
    fixture.write_series(root / f"gemma4-qat-{model}-product", model, **overrides)
    return root


class TestQatPlanDeclaration:
    def test_it_declares_the_runtime_knobs_and_the_traced_bound(self, tmp_path: Path) -> None:
        """The default conversation capacity and chunk length are the regular Gemma's, and the
        trace bound comes from the provenance."""
        plan = qat_plan(_series(tmp_path), "e2b")
        config = plan.pipeline_config

        assert config["capacity"] == DEFAULT_CAPACITY
        assert config["chunkLength"] == DEFAULT_CHUNK_LENGTH
        assert config["maxChunkLength"] == MAX_CHUNK_LENGTH
        assert config["maxPosition"] == fixture.MAX_POSITION

    def test_it_derives_the_sampler_and_rope_from_the_upstream_declarations(
        self, tmp_path: Path
    ) -> None:
        """Fields that are not copied by hand (they come from upstream `generation_config.json` /
        `config.json`)."""
        config = qat_plan(_series(tmp_path), "e2b").pipeline_config

        assert config["sampler"]["topK"] == fixture.GENERATION_CONFIG["top_k"]
        assert set(config["rope"]) == set(fixture.ROPE_HEAD_DIMS)

    def test_it_names_the_measured_default_quant(self, tmp_path: Path) -> None:
        plan = qat_plan(_series(tmp_path), "e2b")

        assert plan.pipeline == "gemma4-qat/1"
        assert plan.default_quant == "i4-fast"


class TestQatPlanGate:
    """Break the rejection conditions of `qat_plan` one at a time (there is exactly one valid
    series in every variant)."""

    @pytest.mark.parametrize(
        "fault",
        ["schema", "family", "model", "maxChunkLength", "maxSelectedRows"],
    )
    def test_it_refuses_a_reference_from_another_generation(
        self, tmp_path: Path, fault: str
    ) -> None:
        broken = fixture.reference_record("e2b")
        broken[fault] = "e4b" if fault == "model" else 1
        with pytest.raises(DistError, match="reference"):
            qat_plan(_series(tmp_path, reference=broken), "e2b")

    def test_it_names_the_old_sidecar_generation_by_the_field_it_carries(
        self, tmp_path: Path
    ) -> None:
        """MUST: the old sidecar generation (with `pleShards`) fails **by name**.

        If only the field name moved while the schema stayed, an old record would pass the
        version check and fail only as "`pleBlocks` (= None) differs from the actual one" —
        nowhere would it be spelled out that what is really missing is **the field itself** (two
        local series actually stopped in this shape).
        """
        broken = fixture.reference_record("e2b")
        broken["schema"] = REFERENCE_SCHEMA - 1
        broken["pleShards"] = broken.pop("pleBlocks")

        with pytest.raises(DistError, match="old sidecar generation") as raised:
            qat_plan(_series(tmp_path, reference=broken), "e2b")

        assert "pleShards" in str(raised.value) and "pleBlocks" in str(raised.value)

    def test_a_record_of_the_current_generation_passes_that_gate(self, tmp_path: Path) -> None:
        """The counterpart (not vacuous): the same record passes once it is back to schema 3 +
        `pleBlocks`."""
        assert qat_plan(_series(tmp_path), "e2b").pipeline == "gemma4-qat/1"

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("fixedWeights", 4),
            ("storageCounts", {"i2": 1, "i4": 2, "i8": 1}),
            ("pleBlocks", 2),
        ],
    )
    def test_it_reconciles_the_recorded_counts_with_the_series_itself(
        self, tmp_path: Path, field: str, value: Any
    ) -> None:
        """The record says "how many were checked" — if it differs from the actual ones, the run
        that wrote it and the series being distributed are different things."""
        broken = fixture.reference_record("e2b", **{field: value})
        with pytest.raises(DistError, match="differ from the actual ones"):
            qat_plan(_series(tmp_path, reference=broken), "e2b")

    @pytest.mark.parametrize("file", ["model.safetensors", "config.json", "tokenizer.json"])
    def test_it_refuses_a_series_exported_from_another_upstream_commit(
        self, tmp_path: Path, file: str
    ) -> None:
        """The card names the pinned commits; a series from other bytes must not be published
        under their names (the tokenizer's included — the mobile repositories ship another
        `tokenizer.json`)."""
        broken = fixture.reference_record("e2b")
        broken["checkpoint"]["files"][file]["sha256"] = "0" * 64

        with pytest.raises(
            DistError, match=f"not exported from .*@{SOURCES['e2b'].pins.weights.revision}"
        ):
            qat_plan(_series(tmp_path, reference=broken), "e2b")

    def test_it_refuses_a_record_without_the_checkpoint_fingerprint(self, tmp_path: Path) -> None:
        broken = fixture.reference_record("e2b")
        del broken["checkpoint"]

        with pytest.raises(DistError, match=r"model\.safetensors"):
            qat_plan(_series(tmp_path, reference=broken), "e2b")

    def test_it_refuses_the_other_models_checkpoint(self, tmp_path: Path) -> None:
        """E4B's fingerprint in an E2B series: the pins are per model, not one shared set."""
        broken = fixture.reference_record("e2b", checkpoint=fixture.pinned_checkpoint("e4b"))

        with pytest.raises(DistError, match=r"config\.json, model\.safetensors"):
            qat_plan(_series(tmp_path, reference=broken), "e2b")

    def test_it_refuses_a_ple_built_for_another_vocabulary(self, tmp_path: Path) -> None:
        with pytest.raises(DistError, match="vocabulary sizes"):
            qat_plan(_series(tmp_path, ple_rows=fixture.VOCAB + 1), "e2b")

    def test_it_refuses_a_ple_stored_with_the_other_models_bit_width(self, tmp_path: Path) -> None:
        """E2B is INT4 and E4B is INT2 (`PLE_BITS`) — a mixed-up sidecar fails on the storage
        spelling."""
        with pytest.raises(DistError, match="storage"):
            qat_plan(_series(tmp_path, ple_bits=2), "e2b")

    def test_it_refuses_a_container_missing_one_of_the_mixed_storages(self, tmp_path: Path) -> None:
        """A container without the full fixed mix (i2 + i4 + i8) does not pass as the QAT
        distribution."""
        with pytest.raises(DistError, match="i8"):
            qat_plan(_series(tmp_path, container_kwargs={"mlp_bits": 4}), "e2b")

    def test_it_refuses_a_capacity_beyond_the_model_limit(self, tmp_path: Path) -> None:
        """The default capacity (4096) × an upstream with a smaller position limit — never build
        a shape that fails only in long conversations."""
        text_config = {**fixture.TEXT_CONFIG, "max_position_embeddings": DEFAULT_CAPACITY - 1}
        with pytest.raises(DistError, match="position limit"):
            qat_plan(_series(tmp_path, text_config=text_config), "e2b")

    def test_it_refuses_a_graph_whose_hidden_width_disagrees_with_the_config(
        self, tmp_path: Path
    ) -> None:
        """The two exits have the same shape down to the row axis, so only comparing the widths
        can catch a mix-up."""
        with pytest.raises(DistError, match="hidden"):
            qat_plan(
                _series(tmp_path, container_kwargs={"hidden_size": fixture.HIDDEN + 16}), "e2b"
            )
