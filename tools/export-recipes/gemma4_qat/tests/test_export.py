"""変換が**読まなかった**上流テンソルの記録（`gemma4_qat.export.upstream_unused`）。

text 専用の変換なので、上流 checkpoint の大半（vision / audio）と KV cache の SRQ scale は
読まない。未対応を暗黙にしないための記録なので、数えるのは実チェックポイントの綴りに対して
成立していること — ここでは同じ綴りを持つ合成ヘッダで見る（実物は 2.4GiB で開かない）。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from gemma4_qat.export import upstream_unused


def _checkpoint(directory: Path, names: list[str], file: str = "model.safetensors") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    save_file({name: torch.zeros(1) for name in names}, str(directory / file))
    return directory


class TestUpstreamUnused:
    def test_it_counts_each_untouched_group_from_the_header(self, tmp_path: Path) -> None:
        directory = _checkpoint(
            tmp_path,
            [
                "model.language_model.layers.0.self_attn.k_cache_scale",
                "model.language_model.layers.0.self_attn.v_cache_scale",
                "model.language_model.layers.0.self_attn.q_proj.weight",
                "model.vision_tower.encoder.layer.0.weight",
                "model.embed_vision.weight",
                "model.audio_tower.encoder.layer.0.weight",
            ],
        )

        assert upstream_unused(directory) == {
            "kvCacheScales": 2,
            "vision": 2,
            "audio": 1,
        }

    def test_it_spans_every_shard_of_a_split_checkpoint(self, tmp_path: Path) -> None:
        directory = _checkpoint(
            tmp_path, ["model.embed_vision.weight"], "model-00001-of-00002.safetensors"
        )
        _checkpoint(directory, ["model.audio_tower.weight"], "model-00002-of-00002.safetensors")

        assert upstream_unused(directory)["vision"] == 1
        assert upstream_unused(directory)["audio"] == 1

    def test_it_refuses_a_directory_without_a_checkpoint(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="safetensors"):
            upstream_unused(tmp_path)
