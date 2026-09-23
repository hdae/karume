"""QAT の配布計画（`gemma4_qat.distribution.qat_plan`）— 組み立て 1 周ぶんの単体テスト。

実チェックポイント（2.4GiB）は使わない。計画へ届く入力は数 KB の**正当な最小の系列**
（{@link gemma4_qat.tests.series_fixture}）で、門に落とされることを見るケースだけがその変種になる。

核は「別々の台本が持つ同じ事実を、組み立て時に突き合わせる」— 固定 writer（`export.py` が
書く `reference.json`）/ packed PLE（`ple.py`）/ トークナイザ資産 / 上流 config は独立に動ける
ので、噛み合っていないことは**配布形を並べる前**にしか落とせない。
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
)
from gemma4_qat.distribution import qat_plan
from gemma4_qat.tests import series_fixture as fixture
from karume.dist import DistError


def _series(root: Path, model: str = "e2b", **overrides: Any) -> Path:
    fixture.write_series(root / f"gemma4-qat-{model}-product", model, **overrides)
    return root


class TestQatPlanDeclaration:
    def test_it_declares_the_runtime_knobs_and_the_traced_bound(self, tmp_path: Path) -> None:
        """既定の会話容量と chunk 長は通常 Gemma と同じで、trace 上限は provenance 由来。"""
        plan = qat_plan(_series(tmp_path), "e2b")
        config = plan.pipeline_config

        assert config["capacity"] == DEFAULT_CAPACITY
        assert config["chunkLength"] == DEFAULT_CHUNK_LENGTH
        assert config["maxChunkLength"] == MAX_CHUNK_LENGTH
        assert config["maxPosition"] == fixture.MAX_POSITION

    def test_it_derives_the_sampler_and_rope_from_the_upstream_declarations(
        self, tmp_path: Path
    ) -> None:
        """写経しない欄（上流 `generation_config.json` / `config.json` 由来）。"""
        config = qat_plan(_series(tmp_path), "e2b").pipeline_config

        assert config["sampler"]["topK"] == fixture.GENERATION_CONFIG["top_k"]
        assert set(config["rope"]) == set(fixture.ROPE_HEAD_DIMS)

    def test_it_names_the_measured_default_quant(self, tmp_path: Path) -> None:
        plan = qat_plan(_series(tmp_path), "e2b")

        assert plan.pipeline == "gemma4-qat/1"
        assert plan.default_quant == "i4-fast"


class TestQatPlanGate:
    """`qat_plan` の拒否条件を 1 つずつ壊す（正当な系列はどの版でも 1 つだけ）。"""

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
        """MUST: 旧 sidecar 世代（`pleShards` を持つ）は**名指しで**落ちる。

        schema を据え置いたまま欄名だけ動かすと、古い記録は版の門を素通りして
        「`pleBlocks`（= None）が現物と違う」でだけ落ちる — 実際に足りないのが**欄そのもの**
        であることがどこにも綴られない（手元の 2 系列が実際にこの形で止まっていた）。
        """
        broken = fixture.reference_record("e2b")
        broken["schema"] = REFERENCE_SCHEMA - 1
        broken["pleShards"] = broken.pop("pleBlocks")

        with pytest.raises(DistError, match="旧 sidecar 世代") as raised:
            qat_plan(_series(tmp_path, reference=broken), "e2b")

        assert "pleShards" in str(raised.value) and "pleBlocks" in str(raised.value)

    def test_a_record_of_the_current_generation_passes_that_gate(self, tmp_path: Path) -> None:
        """対（恒真でない）: 同じ記録を schema 3 + `pleBlocks` に戻せば通る。"""
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
        """記録は「何本を照合したか」— 現物と違えば、書いた run と配る系列が別物である。"""
        broken = fixture.reference_record("e2b", **{field: value})
        with pytest.raises(DistError, match="現物と違う"):
            qat_plan(_series(tmp_path, reference=broken), "e2b")

    def test_it_refuses_a_ple_built_for_another_vocabulary(self, tmp_path: Path) -> None:
        with pytest.raises(DistError, match="語彙数"):
            qat_plan(_series(tmp_path, ple_rows=fixture.VOCAB + 1), "e2b")

    def test_it_refuses_a_ple_stored_with_the_other_models_bit_width(self, tmp_path: Path) -> None:
        """E2B は INT4・E4B は INT2（`PLE_BITS`）— 取り違えた sidecar は格納の綴りで落ちる。"""
        with pytest.raises(DistError, match="storage"):
            qat_plan(_series(tmp_path, ple_bits=2), "e2b")

    def test_it_refuses_a_container_missing_one_of_the_mixed_storages(self, tmp_path: Path) -> None:
        """固定混成（i2 + i4 + i8）が揃わない容器は QAT の配布形として通さない。"""
        with pytest.raises(DistError, match="i8"):
            qat_plan(_series(tmp_path, container_kwargs={"mlp_bits": 4}), "e2b")

    def test_it_refuses_a_capacity_beyond_the_model_limit(self, tmp_path: Path) -> None:
        """既定容量（4096）× 位置上限の小さい上流 — 長い会話でだけ落ちる形を焼かない。"""
        text_config = {**fixture.TEXT_CONFIG, "max_position_embeddings": DEFAULT_CAPACITY - 1}
        with pytest.raises(DistError, match="位置上限"):
            qat_plan(_series(tmp_path, text_config=text_config), "e2b")

    def test_it_refuses_a_graph_whose_hidden_width_disagrees_with_the_config(
        self, tmp_path: Path
    ) -> None:
        """出口 2 本は行軸まで同型なので、取り違えを落とせるのは幅の突合だけ。"""
        with pytest.raises(DistError, match="hidden"):
            qat_plan(
                _series(tmp_path, container_kwargs={"hidden_size": fixture.HIDDEN + 16}), "e2b"
            )
