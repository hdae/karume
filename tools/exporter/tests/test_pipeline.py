"""一本道（export → 書き出し → 検証）の原子性。検証を通るまで最終パスに触れない。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
from torch import nn

from karume import emit, pipeline, verify
from karume.emit import write_model
from karume.pipeline import export_module, export_to_file
from karume.shards import resolve_shards
from karume.verify import ContainerError, verify_shards


class Biased(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.arange(4, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bias


class TwoWeights(nn.Module):
    """格納テンソルを 2 本持つ（f32 4 要素 = 16 バイトずつ）— shard 分割を踏むための最小形。"""

    def __init__(self) -> None:
        super().__init__()
        self.left = nn.Parameter(torch.arange(4, dtype=torch.float32))
        self.right = nn.Parameter(torch.arange(4, dtype=torch.float32) * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x + self.left) * self.right


EXAMPLE = (torch.randn(2, 4),)
TWO_EXAMPLE = (torch.randn(2, 4),)

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

    def reject(paths):
        raise ContainerError("読めないファイル")

    monkeypatch.setattr(pipeline, "verify_shards", reject)


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
        staged = tmp_path / "staged" / "model.safetensors"
        direct = tmp_path / "direct" / "model.safetensors"

        export_to_file(Biased(), EXAMPLE, staged)
        graph, tensors = export_module(Biased(), EXAMPLE)
        write_model(direct, graph, tensors)

        # 突合は shard 列の全要素（据え替えは 1 本ずつ os.replace で進む）。
        assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in resolve_shards(staged)] == [
            hashlib.sha256(p.read_bytes()).hexdigest() for p in resolve_shards(direct)
        ]


class TestShardedPublication:
    """コンポーネントは**連番の shard 列**として据わる（ADR 0081 — 常時分割）。

    weight shard の本数は合成の小テンソルへ人工的に上限を下げて増やす — `export_to_file` は
    分割の可否を選べない（配布形の不変条件）ので、書き手が引く定数を差し替える。
    """

    def tiny_limit(self, monkeypatch: pytest.MonkeyPatch, limit: int) -> None:
        """`emit` が呼び出しのたびに引くモジュール定数を下げる。"""
        monkeypatch.setattr(emit, "SHARD_BYTE_LIMIT", limit)

    def test_it_publishes_the_numbered_sequence(self, tmp_path, monkeypatch):
        # `TwoWeights` の格納テンソルは 16 バイト × 2 本。上限 16 で weight shard は 1 本ずつ。
        self.tiny_limit(monkeypatch, 16)
        final = tmp_path / "model.safetensors"

        export_to_file(TwoWeights(), TWO_EXAMPLE, final)

        assert sorted(entry.name for entry in tmp_path.iterdir()) == [
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
        ]

    def test_the_published_shards_pass_the_full_verification(self, tmp_path, monkeypatch):
        self.tiny_limit(monkeypatch, 16)
        final = tmp_path / "model.safetensors"

        graph = export_to_file(TwoWeights(), TWO_EXAMPLE, final)

        written = resolve_shards(final)
        assert len(written) == 3
        assert verify_shards(written).to_dict() == graph.to_dict()

    def test_a_failed_export_leaves_no_shard_behind(self, tmp_path, monkeypatch):
        """分割の途中まで書けた一時ファイルも残さない（`.partial` の連番ごと捨てる）。"""
        self.tiny_limit(monkeypatch, 16)
        break_verify(monkeypatch)

        with pytest.raises(ContainerError):
            export_to_file(TwoWeights(), TWO_EXAMPLE, tmp_path / "model.safetensors")

        assert list(tmp_path.iterdir()) == []

    def test_it_clears_the_shards_of_a_previous_export(self, tmp_path, monkeypatch):
        """本数が変わった再 export の残骸は残さない（同居は組み立てが拒否する形）。"""
        final = tmp_path / "model.safetensors"
        export_to_file(TwoWeights(), TWO_EXAMPLE, final)
        assert len(resolve_shards(final)) == 2

        self.tiny_limit(monkeypatch, 16)
        export_to_file(TwoWeights(), TWO_EXAMPLE, final)

        assert not final.exists()
        assert len(resolve_shards(final)) == 3
        assert len(list(tmp_path.iterdir())) == 3

    def test_it_clears_shards_of_a_previous_split(self, tmp_path, monkeypatch):
        """本数が減る再 export でも、前回の連番は置き去りにしない。"""
        final = tmp_path / "model.safetensors"
        self.tiny_limit(monkeypatch, 16)
        export_to_file(TwoWeights(), TWO_EXAMPLE, final)

        monkeypatch.undo()
        export_to_file(TwoWeights(), TWO_EXAMPLE, final)

        assert sorted(entry.name for entry in tmp_path.iterdir()) == [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ]

    def test_the_cli_verifies_the_component_from_its_representative_path(
        self, tmp_path, monkeypatch, capsys
    ):
        """`karume verify <代表 path>` は連番へ解決してまとめて検証する。

        shard 1 本だけを単体で検証しても「グラフが無い」としか言えないので、CLI の引数は
        本数に依らず**コンポーネントの代表 path**で通る形にしてある。
        """
        final = tmp_path / "model.safetensors"
        self.tiny_limit(monkeypatch, 16)
        export_to_file(TwoWeights(), TWO_EXAMPLE, final)

        verify.main([str(final)])

        assert "shards=3" in capsys.readouterr().out
