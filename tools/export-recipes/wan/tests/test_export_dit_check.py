"""`wan/export_dit.py` の `--check`（書き手で部品ディレクトリを系列の中の一時ディレクトリへ
書き直し、既存の系列と全ファイルの sha256 で突き合わせる口 — ADR 0121 決定 10）の約束事。

守るのは「Wan2.1 と共有する書き手を変えたとき、1.3B の容器と golden が 1 バイトも動いていないこと」
を確かめる口そのもの:

- 書き手が変わっていなければ全ファイル（容器の全 part・`io.*`・`reference.*`）が一致し、系列は
  1 バイトも変わらない（一時ディレクトリも残らない）
- 書き手の出力が変われば、違うファイルを名指して止まる（故障注入: 1 ケースの seed を変える →
  そのケースの 2 ファイルだけが名指しされる）
- `--no-full` は実寸のケースのファイルだけを照合から外す（外したファイルが既存に無ければ名指す）
- 照合の相手が無ければ、上流を読む前に止まる

合成モデル（`test_dit_patch.TINY_DIT` — 乱数初期化の小さな `WanTransformer3DModel`）だけで回す。
1.3B の実物の照合は `python -m wan.export_dit --dtype i8 --no-full --check` など（runbook）。
"""

from __future__ import annotations

import dataclasses
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from wan import export_dit
from wan.tests.test_dit_patch import TINY_DIT, TINY_LATENT

#: 合成の patch（TINY_DIT の patch_size）。
PATCH = (1, 2, 2)

#: 合成の系列のケース（i8 系列が採る役割 2 つ + 実寸の役割 1 つ — 実寸の格子は
#: {@link export_dit.I8_FULL_LATENTS} を合成の格子へ差し替えて選ばせる）。
CASES = (
    export_dit.CaseSpec("band", TINY_LATENT, 999, 5, 1),
    export_dit.CaseSpec("accept", TINY_LATENT, 500, 3, 2),
    export_dit.CaseSpec("full-band", TINY_LATENT, 250, 4, 3, blocks=False),
)


def _tiny(_model_name: str = "") -> nn.Module:
    """毎回同じ重みの合成 DiT（`export_dit.load_transformer` の差し替え — 書き直しの回も
    同じ値）。"""
    transformer_wan = pytest.importorskip("diffusers.models.transformers.transformer_wan")
    torch.manual_seed(20261004)
    return transformer_wan.WanTransformer3DModel(**TINY_DIT).to(torch.float32).eval()


def _point_at_tiny(patch: pytest.MonkeyPatch, root: Path) -> None:
    patch.setattr(export_dit, "CASES", CASES)
    patch.setattr(export_dit, "I8_FULL_LATENTS", (TINY_LATENT,))
    patch.setattr(export_dit, "SERIES", root / "f16")
    patch.setattr(export_dit, "I8_SERIES", root / "i8")
    patch.setattr(export_dit, "PROBE_SERIES", root / "probe")
    patch.setattr(export_dit, "load_transformer", _tiny)
    patch.setattr(export_dit, "TEXT_DIM", TINY_DIT["text_dim"])
    patch.setattr(export_dit, "pad_text_embeds", lambda embeds: embeds.unsqueeze(0))


def _digests(directory: Path) -> dict[str, str]:
    return export_dit.file_digests(directory)


@pytest.fixture(scope="module")
def written(tmp_path_factory) -> Iterator[Path]:
    """合成の i8 系列（実寸の役割を含む全ケース）を書いた系列の根。"""
    root = tmp_path_factory.mktemp("export-dit-check")
    with pytest.MonkeyPatch.context() as patch:
        _point_at_tiny(patch, root)
        assert export_dit.main(["--dtype", "i8"]) == 0
    yield root / "i8"


@pytest.fixture
def series(written: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """テストごとの写し（壊す・消すテストが他のテストの系列に触らない）。"""
    copy = tmp_path / "i8"
    shutil.copytree(written, copy)
    _point_at_tiny(monkeypatch, tmp_path)
    return copy


def _check(*flags: str) -> dict[str, Any]:
    """i8 系列の `--check` を回して要約を返す（引数は `main` の検査を通す）。"""
    return export_dit.emit(_namespace("--dtype", "i8", "--check", *flags))


def _namespace(*argv: str) -> Any:
    """`main` と同じ引数の検査を通した `emit` の引数（要約を戻りで受けるため）。"""
    import argparse

    captured: dict[str, argparse.Namespace] = {}

    def capture(args: argparse.Namespace) -> dict[str, Any]:
        captured["args"] = args
        return {}

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(export_dit, "emit", capture)
        assert export_dit.main(list(argv)) == 0
    return captured["args"]


class TestAnUnchangedWriter:
    def test_every_file_matches_and_the_series_is_untouched(self, series: Path) -> None:
        before = _digests(series / export_dit.TARGET)

        summary = _check()

        names = [spec.name(PATCH) for spec in CASES]
        golden = {file for name in names for file in export_dit.case_file_names(name)}
        assert summary["check"]["skipped_existing"] == []
        assert golden <= set(summary["check"]["identical_files"])
        assert any(name.startswith("model-") for name in summary["check"]["identical_files"])
        assert set(summary["check"]["identical_files"]) == set(before)
        assert _digests(series / export_dit.TARGET) == before
        assert sorted(path.name for path in series.iterdir()) == [export_dit.TARGET]

    def test_no_full_skips_exactly_the_full_size_case_files(self, series: Path) -> None:
        summary = _check("--no-full")

        full = set(export_dit.case_file_names(CASES[2].name(PATCH)))
        assert set(summary["check"]["skipped_existing"]) == full
        assert not full & set(summary["check"]["identical_files"])


class TestAChangedWriter:
    def test_a_changed_case_names_exactly_its_two_files(
        self, series: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """故障注入: 受入れの 1 ケースの seed を変える（書き手の出力がそのケースだけ動く）。"""
        changed = dataclasses.replace(CASES[1], seed=CASES[1].seed + 1)
        monkeypatch.setattr(export_dit, "CASES", (CASES[0], changed, CASES[2]))
        before = _digests(series / export_dit.TARGET)

        with pytest.raises(export_dit.SeriesMismatchError) as caught:
            _check()

        message = str(caught.value)
        io_name, reference_name = export_dit.case_file_names(CASES[1].name(PATCH))
        assert io_name in message
        assert reference_name in message
        assert export_dit.case_file_names(CASES[0].name(PATCH))[0] not in message
        assert _digests(series / export_dit.TARGET) == before
        assert sorted(path.name for path in series.iterdir()) == [export_dit.TARGET]

    def test_a_skipped_file_missing_from_the_series_is_named(self, series: Path) -> None:
        _, reference_name = export_dit.case_file_names(CASES[2].name(PATCH))
        (series / export_dit.TARGET / reference_name).unlink()

        with pytest.raises(export_dit.SeriesMismatchError, match=reference_name):
            _check("--no-full")


class TestNoExistingSeries:
    def test_it_stops_before_reading_the_upstream(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _point_at_tiny(monkeypatch, tmp_path)

        def never(_model_name: str) -> nn.Module:
            raise AssertionError("照合の相手が無いのに上流を読んだ")

        monkeypatch.setattr(export_dit, "load_transformer", never)

        with pytest.raises(export_dit.SeriesMismatchError, match="照合の相手"):
            _check()


class TestSeriesMismatches:
    def _directory(self, root: Path, files: dict[str, bytes]) -> Path:
        root.mkdir()
        for name, payload in files.items():
            (root / name).write_bytes(payload)
        return root

    def test_each_kind_of_difference_is_named(self, tmp_path: Path) -> None:
        written = self._directory(tmp_path / "w", {"a": b"1", "b": b"2", "extra": b"x"})
        existing = self._directory(tmp_path / "e", {"a": b"1", "b": b"3", "gone": b"y"})

        assert export_dit.series_mismatches(written, existing) == {
            "b": "sha256 が違う",
            "extra": "既存に無い",
            "gone": "書いた側に無い",
        }

    def test_skipped_files_must_exist_only_in_the_series(self, tmp_path: Path) -> None:
        written = self._directory(tmp_path / "w", {"a": b"1", "both": b"z"})
        existing = self._directory(tmp_path / "e", {"a": b"1", "skip": b"s", "both": b"z"})

        assert export_dit.series_mismatches(
            written, existing, absent={"skip", "both", "nowhere"}
        ) == {
            "both": "照合から外したケースのファイルを書いた",
            "nowhere": "照合から外したケースのファイルが既存に無い",
        }

    def test_a_directory_inside_the_component_fails_loudly(self, tmp_path: Path) -> None:
        written = self._directory(tmp_path / "w", {"a": b"1"})
        (written / "nested").mkdir()

        with pytest.raises(export_dit.SeriesMismatchError, match="ファイルでない項目"):
            export_dit.file_digests(written)


class TestTheFlags:
    def test_check_and_verify_do_not_mix(self) -> None:
        with pytest.raises(SystemExit):
            export_dit.main(["--verify", "--check"])
