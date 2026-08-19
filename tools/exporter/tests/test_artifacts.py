"""成果物の transactional な公開（`karume.artifacts`）の単体テスト。

観測する不変条件は 1 つ — **final は据わるか、前のままか、その 2 つしかない**。中途の形が
正規 path に現れる経路が本当に無いことは、各段で故障を注入して確かめる（書き込みの途中で
落ちる / 検証門が拒否する / rename が I/O 故障する）。

被験体は数バイトのテキスト木で足りる — この層が知っているのは path と rename だけで、中身の
形式には 1 つも触れない。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from karume.artifacts import (
    STAGING_SUFFIX,
    SUPERSEDED_SUFFIX,
    ArtifactSwapError,
    staged_publication,
    swap_into_place,
)


def _tree(root: Path, name: str, files: dict[str, str]) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    for filename, body in files.items():
        (directory / filename).write_text(body, encoding="utf-8")
    return directory


def _snapshot(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): path.read_text(encoding="utf-8")
        for path in root.rglob("*")
        if path.is_file()
    }


def _siblings(final: Path) -> list[str]:
    """final の隣に残った席（作業席・退避席）— 成功でも失敗でも空。"""
    return sorted(path.name for path in final.parent.iterdir() if path.name != final.name)


class TestStagedPublication:
    """作業席 → 本体（書き込み + 検証門）→ 据え替えの 1 本道。"""

    def test_a_successful_body_lands_the_staged_tree_at_the_final_path(
        self, tmp_path: Path
    ) -> None:
        """据え替えは**丸ごと** — 旧にだけあったファイルは引き継がない。"""
        final = _tree(tmp_path, "final", {"model": "old", "stale": "old"})

        with staged_publication(final) as staging:
            staging.mkdir()
            (staging / "model").write_text("new", encoding="utf-8")

        assert _snapshot(final) == {"model": "new"}
        assert _siblings(final) == []

    def test_a_failure_while_writing_leaves_the_final_artifact_untouched(
        self, tmp_path: Path
    ) -> None:
        """書き込みの途中で落ちても、正規 path は 1 バイトも変わらない。"""
        final = _tree(tmp_path, "final", {"model": "old"})
        before = _snapshot(final)

        with (
            pytest.raises(OSError, match="書き込みの途中で落ちた"),
            staged_publication(final) as staging,
        ):
            staging.mkdir()
            (staging / "model").write_text("half", encoding="utf-8")
            raise OSError("書き込みの途中で落ちた")

        assert _snapshot(final) == before
        assert _siblings(final) == []

    def test_a_rejecting_verification_hook_leaves_the_final_artifact_untouched(
        self, tmp_path: Path
    ) -> None:
        """門は本体の中 — 全部書き終えてから拒否しても、据え替えごと起きない。"""
        final = _tree(tmp_path, "final", {"model": "old"})
        before = _snapshot(final)

        def verify(tree: Path) -> None:
            raise AssertionError(f"{tree.name} の検証に失敗した")

        with (
            pytest.raises(AssertionError, match="検証に失敗した"),
            staged_publication(final) as staging,
        ):
            staging.mkdir()
            (staging / "model").write_text("new", encoding="utf-8")
            verify(staging)

        assert _snapshot(final) == before
        assert _siblings(final) == []

    def test_a_working_directory_left_by_an_interrupted_run_is_discarded(
        self, tmp_path: Path
    ) -> None:
        """中断が残した席は踏み直さない（書き手が計画していないファイルが混ざらない）。"""
        final = _tree(tmp_path, "final", {"model": "old"})
        _tree(tmp_path, "final" + STAGING_SUFFIX, {"interrupted": "半端"})

        with staged_publication(final) as staging:
            staging.mkdir()
            (staging / "model").write_text("new", encoding="utf-8")

        assert _snapshot(final) == {"model": "new"}
        assert _siblings(final) == []

    def test_an_artifact_left_only_in_the_superseded_slot_is_restored(self, tmp_path: Path) -> None:
        """前回が rename 2 回の**間**で落ちた形 — 退避席の last-known-good を捨てずに戻す。

        本体が落ちても正規 path には last-known-good が戻っている（捨てていれば「final も
        退避席も無い」= 手元の成果物が完全消失した状態で次が止まる）。
        """
        superseded = _tree(tmp_path, "final" + SUPERSEDED_SUFFIX, {"model": "old"})
        final = tmp_path / "final"

        with (
            pytest.raises(OSError, match="書き込みの途中で落ちた"),
            staged_publication(final) as staging,
        ):
            staging.mkdir()
            raise OSError("書き込みの途中で落ちた")

        assert _snapshot(final) == {"model": "old"}
        assert not superseded.exists()

    def test_a_single_file_artifact_is_swapped_with_one_atomic_rename(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """1 ファイルの成果物は退避席を経由しない — final が不在になる瞬間そのものが無い。"""
        final = tmp_path / "model.safetensors"
        final.write_text("old", encoding="utf-8")
        renames: list[tuple[str, str]] = []
        real = os.replace

        def recording(src: Path, dst: Path) -> None:
            renames.append((Path(src).name, Path(dst).name))
            real(src, dst)

        monkeypatch.setattr("karume.artifacts.os.replace", recording)

        with staged_publication(final) as staged:
            staged.write_text("new", encoding="utf-8")

        assert final.read_text(encoding="utf-8") == "new"
        assert renames == [(final.name + STAGING_SUFFIX, final.name)]
        assert _siblings(final) == []


class TestSwapIntoPlace:
    """既に埋まった席を渡す呼び手向けの据え替え（rename の I/O 故障を注入する）。"""

    def _fail_replace_from(
        self, monkeypatch: pytest.MonkeyPatch, source: Path, message: str
    ) -> None:
        """`source` を移す `os.replace` だけを I/O 故障にする（他は本物を通す）。"""
        real = os.replace

        def failing(src: Path, dst: Path) -> None:
            if Path(src) == source:
                raise OSError(message)
            real(src, dst)

        monkeypatch.setattr("karume.artifacts.os.replace", failing)

    def test_a_failing_promotion_puts_the_previous_artifact_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """昇格が落ちても final は**完全な旧資産のまま**（不在にも混成にもならない）。"""
        final = _tree(tmp_path, "final", {"model": "old"})
        staging = _tree(tmp_path, "staging", {"model": "new"})
        self._fail_replace_from(monkeypatch, staging, "昇格に失敗")

        with pytest.raises(ArtifactSwapError, match="据え替え") as failure:
            swap_into_place(staging, final)

        # 原因（I/O 故障）は連鎖で残す — 据え替えの失敗と書き出しの失敗を取り違えない。
        assert isinstance(failure.value.__cause__, OSError)
        assert _snapshot(final) == {"model": "old"}
        # 席は呼び手のもの（消さない）が、退避席は残さない。
        assert _siblings(final) == ["staging"]

    def test_a_failing_evacuation_leaves_the_previous_artifact_untouched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """退避が落ちても、os.replace は原子的なので final は無傷のまま残る。"""
        final = _tree(tmp_path, "final", {"model": "old"})
        staging = _tree(tmp_path, "staging", {"model": "new"})
        self._fail_replace_from(monkeypatch, final, "退避に失敗")

        with pytest.raises(ArtifactSwapError, match="退避") as failure:
            swap_into_place(staging, final)

        assert isinstance(failure.value.__cause__, OSError)
        assert _snapshot(final) == {"model": "old"}
        assert _siblings(final) == ["staging"]
