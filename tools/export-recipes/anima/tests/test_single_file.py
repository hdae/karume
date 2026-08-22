"""単一ファイル checkpoint → diffusers レイアウトの組み直し（`anima.single_file`）。

実物（4GB）は使わない — ここで観測したいのは**鍵の付け替えと、書く前の突合が実際に落ちるか**
だけで、どちらも数テンソルの偽 checkpoint で再現できる。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from anima.single_file import _assert_covers, _link, _strip_prefix


def _state(prefix: str) -> dict[str, torch.Tensor]:
    return {
        f"{prefix}blocks.0.weight": torch.zeros(2, 3),
        f"{prefix}llm_adapter.blocks.0.weight": torch.zeros(4),
    }


class TestStripPrefix:
    """civitai 側の包み方は 2 通り（`net.` と `model.diffusion_model.`）— どちらも受ける。"""

    @pytest.mark.parametrize("prefix", ["net.", "model.diffusion_model."])
    def test_it_normalizes_a_known_prefix_to_net(self, prefix: str) -> None:
        stripped = _strip_prefix(_state(prefix))

        assert sorted(stripped) == ["net.blocks.0.weight", "net.llm_adapter.blocks.0.weight"]

    def test_it_refuses_an_unknown_prefix_with_what_it_saw(self) -> None:
        """知らない包み方を黙って素通しすると、変換表が何も当たらないまま空を書く。"""
        with pytest.raises(SystemExit, match="知らない鍵の前置"):
            _strip_prefix(_state("diffusion_model."))

    def test_it_refuses_a_mixed_checkpoint(self) -> None:
        """一部だけ前置が違う checkpoint は、剥がす基準を決められない。"""
        mixed = {**_state("net."), "model.diffusion_model.blocks.1.weight": torch.zeros(1)}

        with pytest.raises(SystemExit, match="知らない鍵の前置"):
            _strip_prefix(mixed)


class TestAssertCovers:
    """変換表が上流で動いた日に「一部だけ移った重み」を書かないための門。"""

    @pytest.fixture
    def reference(self, tmp_path: Path) -> Path:
        path = tmp_path / "reference.safetensors"
        save_file({"a": torch.zeros(2, 3), "b": torch.zeros(4)}, path)
        return path

    def test_it_passes_when_the_keys_and_shapes_match(self, reference: Path) -> None:
        _assert_covers({"a": torch.zeros(2, 3), "b": torch.zeros(4)}, reference, "t")

    def test_it_stops_when_a_key_is_missing(self, reference: Path) -> None:
        with pytest.raises(SystemExit, match="足りない 1 件"):
            _assert_covers({"a": torch.zeros(2, 3)}, reference, "t")

    def test_it_stops_when_an_unexpected_key_is_left_over(self, reference: Path) -> None:
        """余る鍵は「表が別のモデル向けに当たった」兆候なので、これも落とす。"""
        state = {"a": torch.zeros(2, 3), "b": torch.zeros(4), "c": torch.zeros(1)}

        with pytest.raises(SystemExit, match="余る 1 件"):
            _assert_covers(state, reference, "t")

    def test_it_stops_when_a_shape_differs(self, reference: Path) -> None:
        """鍵名が揃っていても形が違えば別アーキテクチャ — ロードは通るが絵が壊れる。"""
        with pytest.raises(SystemExit, match="shape が base と違う"):
            _assert_covers({"a": torch.zeros(3, 2), "b": torch.zeros(4)}, reference, "t")


class TestLink:
    """再実行で古い向き先が残らない（組み直しは冪等）。"""

    def test_it_replaces_an_existing_link(self, tmp_path: Path) -> None:
        first = tmp_path / "first"
        second = tmp_path / "second"
        first.mkdir()
        second.mkdir()
        link = tmp_path / "link"

        _link(first, link)
        _link(second, link)

        assert link.resolve() == second.resolve()

    def test_it_replaces_a_real_directory_left_behind(self, tmp_path: Path) -> None:
        target = tmp_path / "target"
        target.mkdir()
        link = tmp_path / "link"
        link.mkdir()

        _link(target, link)

        assert link.is_symlink()
        assert link.resolve() == target.resolve()
