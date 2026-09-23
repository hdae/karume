"""一本道（export → 格納変換 → 書き出し → 読み直して検証 → 据え替え）の原子性。

検証を通るまで最終 path に触れない — 途中で落ちた回は、前回の配布物を 1 バイトも変えない。
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from karume import publish, verify
from karume.container import ContainerFormatError, Provenance, container_parts
from karume.ir import IrNode, IrValue
from karume.ops import OpContractError
from karume.pipeline import export_module, export_to_file, publish_model
from karume.verify import ContainerError, verify_container

PROVENANCE = Provenance(license="apache-2.0", writer="karume-test")


class Biased(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.arange(4, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bias


class TwoWeights(nn.Module):
    """格納テンソルを 2 本持つ（f32 4 要素 = 16 バイトずつ）— part 分割を踏むための最小形。"""

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

#: `TwoWeights` の重み 1 本ぶん（f32 4 要素）。part 長をここへ下げると 1 part = 1 重み。
ONE_WEIGHT_BYTES = 16


def break_write(monkeypatch: pytest.MonkeyPatch) -> None:
    """書き出しが途中まで進んでから落ちる（ディスク溢れ・I/O 故障の形）。"""

    def half_written(path, *args, **kwargs):
        Path(path).write_bytes(b"half")
        raise OSError("書き出し中に落ちた")

    monkeypatch.setattr(publish, "write_model_container", half_written)


def break_verify(monkeypatch: pytest.MonkeyPatch) -> None:
    """書けたファイルが読み直せない（= 配布したら実行できない）。"""

    def reject(paths):
        raise ContainerFormatError("読めないファイル")

    monkeypatch.setattr(publish, "read_container", reject)


def published(module: nn.Module, example, path: Path, **kwargs):
    """`export_module` で変換してから据える（part 長の差し込みは publish 側の席）。"""
    kwargs.setdefault("graph_name", "series")
    return publish_model(path, *export_module(module, example), provenance=PROVENANCE, **kwargs)


class TestAtomicReplacement:
    @pytest.mark.parametrize(
        ("inject", "error"), [(break_write, OSError), (break_verify, ContainerFormatError)]
    )
    def test_a_failed_export_leaves_the_existing_artifact_untouched(
        self, tmp_path, monkeypatch, inject, error
    ):
        final = tmp_path / "model.krm"
        final.write_bytes(SENTINEL)
        inject(monkeypatch)

        with pytest.raises(error):
            export_to_file(Biased(), EXAMPLE, final, provenance=PROVENANCE, graph_name="biased")

        assert final.read_bytes() == SENTINEL
        assert [entry.name for entry in tmp_path.iterdir()] == [final.name]

    @pytest.mark.parametrize(
        ("inject", "error"), [(break_write, OSError), (break_verify, ContainerFormatError)]
    )
    def test_a_failed_export_creates_nothing_when_there_is_no_previous_artifact(
        self, tmp_path, monkeypatch, inject, error
    ):
        final = tmp_path / "model.krm"
        inject(monkeypatch)

        with pytest.raises(error):
            export_to_file(Biased(), EXAMPLE, final, provenance=PROVENANCE, graph_name="biased")

        assert list(tmp_path.iterdir()) == []

    def test_the_delivered_bytes_do_not_depend_on_the_staging_path(self, tmp_path):
        """一時ファイル経由でも配布物のバイト列は変わらない（名前は中身に入らない）。"""
        left = tmp_path / "series" / "model.krm"
        right = tmp_path / "series2" / "model.krm"

        export_to_file(Biased(), EXAMPLE, left, provenance=PROVENANCE, graph_name="series")
        export_to_file(Biased(), EXAMPLE, right, provenance=PROVENANCE, graph_name="series")

        assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in container_parts(left)] == [
            hashlib.sha256(p.read_bytes()).hexdigest() for p in container_parts(right)
        ]

    def test_the_output_name_carries_the_container_suffix(self, tmp_path):
        """呼び手が旧拡張子を渡しても据わるのは `.krm`（配布形は 1 つ — container-v1 §8）。"""
        export_to_file(
            Biased(),
            EXAMPLE,
            tmp_path / "model.f32.safetensors",
            provenance=PROVENANCE,
            graph_name="biased",
        )

        assert all(entry.suffix == ".krm" for entry in tmp_path.iterdir())


class TestPartitionedPublication:
    """コンテナは**連番の part 列**として据わる（container-v1 §8 — HF の公式配布は分割形）。

    重みの part 本数は合成の小テンソルに対して part 長を下げて増やす。
    """

    def test_it_publishes_the_numbered_sequence(self, tmp_path):
        # 重みは 16 バイト × 2 本。part 長 16 で重み part は 1 本ずつ（+ part 0 と part 1）。
        final = tmp_path / "model.krm"

        published(TwoWeights(), TWO_EXAMPLE, final, _part_bytes=ONE_WEIGHT_BYTES)

        assert sorted(entry.name for entry in tmp_path.iterdir()) == [
            "model-00001-of-00004.krm",
            "model-00002-of-00004.krm",
            "model-00003-of-00004.krm",
            "model-00004-of-00004.krm",
        ]

    def test_the_published_parts_pass_the_full_verification(self, tmp_path):
        final = tmp_path / "model.krm"

        graph = published(TwoWeights(), TWO_EXAMPLE, final, _part_bytes=ONE_WEIGHT_BYTES)

        written = container_parts(final)
        assert len(written) == 4
        # グラフ名は呼び手が名乗る（`published` が渡す部品名）— 親ディレクトリ名の既定は
        # 作業席の名前を拾うので退役した（`pipeline._assert_graph_name`）。
        bound = verify_container(written, blocks=True).graphs["series"]
        assert sorted(bound.supplies) == sorted(
            graph.initializers[name].tensor for name in graph.initializers
        )

    def test_a_failed_export_leaves_no_part_behind(self, tmp_path, monkeypatch):
        """分割の途中まで書けた一時ファイルも残さない（`.partial` の連番ごと捨てる）。"""
        break_verify(monkeypatch)

        with pytest.raises(ContainerFormatError):
            published(TwoWeights(), TWO_EXAMPLE, tmp_path / "model.krm")

        assert list(tmp_path.iterdir()) == []

    def test_it_clears_the_parts_of_a_previous_export(self, tmp_path):
        """本数が変わった再 export の残骸は残さない（同居は組み立てが拒否する形）。"""
        final = tmp_path / "model.krm"
        published(TwoWeights(), TWO_EXAMPLE, final)
        assert len(container_parts(final)) == 3

        published(TwoWeights(), TWO_EXAMPLE, final, _part_bytes=ONE_WEIGHT_BYTES)

        assert not final.exists()
        assert len(container_parts(final)) == 4
        assert len(list(tmp_path.iterdir())) == 4

    def test_it_clears_the_parts_of_a_previous_split(self, tmp_path):
        """本数が減る再 export でも、前回の連番は置き去りにしない。"""
        final = tmp_path / "model.krm"
        published(TwoWeights(), TWO_EXAMPLE, final, _part_bytes=ONE_WEIGHT_BYTES)

        published(TwoWeights(), TWO_EXAMPLE, final)

        assert sorted(entry.name for entry in tmp_path.iterdir()) == [
            "model-00001-of-00003.krm",
            "model-00002-of-00003.krm",
            "model-00003-of-00003.krm",
        ]

    def test_it_clears_the_retired_safetensors_of_a_previous_export(self, tmp_path):
        """同じ部品の旧配布形（`.safetensors` の連番）も据わった後に消す。

        残すと系列ディレクトリに「前の形と今の形」が同居し、どちらを配るかが決まらない。
        """
        for name in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
            (tmp_path / name).write_bytes(SENTINEL)
        (tmp_path / "io.input.safetensors").write_bytes(SENTINEL)

        published(TwoWeights(), TWO_EXAMPLE, tmp_path / "model.krm")

        remaining = sorted(entry.name for entry in tmp_path.iterdir())
        assert "io.input.safetensors" in remaining
        assert not any(
            name.endswith(".safetensors") and name.startswith("model") for name in remaining
        )

    def test_the_cli_verifies_the_container_from_its_representative_path(self, tmp_path, capsys):
        """`karume verify <代表 path>` は連番へ解決してまとめて検証する。

        part 1 本だけを単体で検証しても「2 文書が無い」としか言えないので、CLI の引数は
        本数に依らず**コンテナの代表 path**で通る形にしてある。
        """
        final = tmp_path / "model.krm"
        published(TwoWeights(), TWO_EXAMPLE, final, _part_bytes=ONE_WEIGHT_BYTES)

        verify.main([str(final)])

        assert "parts=4" in capsys.readouterr().out


class TestTheIrAcceptanceGate:
    """書き出しの**前**に IR の受理規則（ランタイム支援 + op 契約）を掛ける。

    台本は `to_states_form` のようなグラフ手術を挟むので、変換段の検査だけでは「書けるが
    ランタイムが読めない」容器を止められない — 語彙外の op を 1 本足したグラフは、krm を
    1 バイトも据えずに落ちる MUST（`verify.py` のモジュール doc が掲げる目的の書き出し側）。
    """

    def _graph_with_an_unknown_op(self):
        graph, tensors = export_module(Biased(), EXAMPLE)
        out = "karume_test_unknown"
        graph.values[out] = IrValue(dtype="f32", shape=list(graph.values[graph.outputs[0]].shape))
        graph.nodes.append(
            IrNode(op="totally_unknown_op", ins=[graph.outputs[0]], outs=[out], attrs={})
        )
        graph.outputs.append(out)
        return graph, tensors

    def test_an_op_outside_the_vocabulary_places_nothing(self, tmp_path: Path) -> None:
        graph, tensors = self._graph_with_an_unknown_op()

        with pytest.raises(ContainerError, match="非対応 op"):
            publish_model(
                tmp_path / "model.krm",
                graph,
                tensors,
                provenance=PROVENANCE,
                graph_name="series",
            )

        assert list(tmp_path.iterdir()) == []

    def test_a_contract_violation_places_nothing(self, tmp_path: Path) -> None:
        """語彙の中でも**契約**（入力本数）が破れていれば据えない。"""
        graph, tensors = export_module(Biased(), EXAMPLE)
        graph.nodes[-1] = replace(graph.nodes[-1], ins=graph.nodes[-1].ins[:1])

        with pytest.raises(OpContractError):
            publish_model(
                tmp_path / "model.krm",
                graph,
                tensors,
                provenance=PROVENANCE,
                graph_name="series",
            )

        assert list(tmp_path.iterdir()) == []

    def test_the_same_graph_without_the_extra_op_is_published(self, tmp_path: Path) -> None:
        """対照 — 門が「常に落ちる」のではないことの裏側。"""
        graph, tensors = export_module(Biased(), EXAMPLE)

        publish_model(
            tmp_path / "model.krm", graph, tensors, provenance=PROVENANCE, graph_name="series"
        )

        assert container_parts(tmp_path / "model.krm")


class TestTheGraphNameIsNamedByTheCaller:
    """グラフ名に既定は無い（`provenance` と同じ扱い）。

    既定を親ディレクトリ名にしていた頃は、作業席（`<部品>.staging/`）へ書く全 recipe が
    `<部品>.staging` というグラフ名の容器を焼いた — 語彙に `.` が入るので fail loudly もせず、
    ランタイムが部品名で引いた時点で初めて「コンテナにグラフが無い」になる。
    """

    def test_the_container_carries_the_name_the_caller_gave(self, tmp_path: Path) -> None:
        staging = tmp_path / "dit.staging"
        staging.mkdir()

        published(Biased(), EXAMPLE, staging / "model.krm", graph_name="dit")

        bound = verify_container(container_parts(staging / "model.krm")).graphs

        assert sorted(bound) == ["dit"]

    def test_a_name_outside_the_vocabulary_fails_loudly(self, tmp_path: Path) -> None:
        with pytest.raises(ContainerFormatError, match="語彙"):
            published(Biased(), EXAMPLE, tmp_path / "model.krm", graph_name="dit/1")

        assert list(tmp_path.iterdir()) == []
