"""一本道（export → 書き出し → 検証）の原子性。検証を通るまで最終パスに触れない。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
from torch import nn

from karume import pipeline
from karume.emit import write_model
from karume.pipeline import export_module, export_to_file
from karume.verify import ContainerError


class Biased(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.arange(4, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bias


EXAMPLE = (torch.randn(2, 4),)

#: 既に配布できている成果物（再エクスポートの失敗で失われてはいけないバイト列）。
SENTINEL = b"already-shipped"


def break_write(monkeypatch: pytest.MonkeyPatch) -> None:
    """書き出しが途中まで進んでから落ちる（ディスク溢れ・I/O 故障の形）。"""

    def half_written(path, graph, tensors, **kwargs):
        Path(path).write_bytes(b"half")
        raise OSError("書き出し中に落ちた")

    monkeypatch.setattr(pipeline, "write_model", half_written)


def break_verify(monkeypatch: pytest.MonkeyPatch) -> None:
    """書けたファイルが IR v1 の規則を満たさない（= 配布したら実行できない）。"""

    def reject(path):
        raise ContainerError("読めないファイル")

    monkeypatch.setattr(pipeline, "verify_model", reject)


class TestAtomicReplacement:
    @pytest.mark.parametrize(
        ("inject", "error"), [(break_write, OSError), (break_verify, ContainerError)]
    )
    def test_a_failed_export_leaves_the_existing_artifact_untouched(
        self, tmp_path, monkeypatch, inject, error
    ):
        final = tmp_path / "model.safetensors"
        final.write_bytes(SENTINEL)
        inject(monkeypatch)

        with pytest.raises(error):
            export_to_file(Biased(), EXAMPLE, final)

        assert final.read_bytes() == SENTINEL
        assert [entry.name for entry in tmp_path.iterdir()] == [final.name]

    @pytest.mark.parametrize(
        ("inject", "error"), [(break_write, OSError), (break_verify, ContainerError)]
    )
    def test_a_failed_export_creates_nothing_when_there_is_no_previous_artifact(
        self, tmp_path, monkeypatch, inject, error
    ):
        final = tmp_path / "model.safetensors"
        inject(monkeypatch)

        with pytest.raises(error):
            export_to_file(Biased(), EXAMPLE, final)

        assert list(tmp_path.iterdir()) == []

    def test_the_delivered_bytes_are_the_same_as_a_direct_write(self, tmp_path):
        """段の追加で配布物のバイト列は変わらない（一時ファイル経由でも中身は同一）。"""
        staged = tmp_path / "staged.safetensors"
        direct = tmp_path / "direct.safetensors"

        export_to_file(Biased(), EXAMPLE, staged)
        graph, tensors = export_module(Biased(), EXAMPLE)
        write_model(direct, graph, tensors)

        assert hashlib.sha256(staged.read_bytes()).hexdigest() == (
            hashlib.sha256(direct.read_bytes()).hexdigest()
        )
