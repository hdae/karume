"""配布形を据える 3 段（`karume.publish`）— 書く → 読み直して検証 → 据え替え。

ここが固定するのは**据え替えの規律**そのもの（原子性・後始末・読み直し検証の検出力）で、
容器のバイト列の規則は `test_container.py`、export の一本道は `test_pipeline.py` の担当。

故障注入は「書き出しの途中で落とす」「据え替えの rename を落とす」「渡した束縛だけを壊す」の
3 系統。どれも**呼び出しが書いていない現物**（前回の成果物）が巻き添えにならないことまで見る。
"""

from __future__ import annotations

import os
from collections.abc import Buffer, Iterator, Mapping
from dataclasses import replace
from pathlib import Path

import pytest
from ir_fixtures import FIXTURE_PROVENANCE, fixture_spec

from karume import publish
from karume.emit import StoredModel, stored_model
from karume.publish import PublishError, publish_container

#: 前回の成果物の目印（`krg` の中身は読まないので、バイト列は何でもよい）。
SENTINEL = b"previous"

GRAPH_NAME = "publish"


def material(storage: str = "i8") -> StoredModel:
    """合成の部品 1 つを `publish_container` へ渡せる 3 点へ落とす。"""
    graph, tensors, scales, overrides = fixture_spec("publish", storage)
    return stored_model(
        graph,
        tensors,
        weight_dtype=storage,
        weight_scales=scales,
        weight_dtype_overrides=overrides,
    )


def publish_material(
    directory: Path, stored: StoredModel, *, graph_path: Path | None = None
) -> publish.PublishResult:
    return publish_container(
        directory / "model.krm",
        stored.graph,
        stored.tensors,
        stored.bindings,
        graph_name=GRAPH_NAME,
        provenance=FIXTURE_PROVENANCE,
        graph_path=graph_path,
    )


class _Exploding(Mapping[str, Buffer]):
    """引かれた瞬間に落ちるテンソルの口（書き出しの途中で落ちる形を作る）。"""

    def __init__(self, inner: Mapping[str, Buffer]) -> None:
        self._inner = inner

    def __getitem__(self, key: str) -> Buffer:
        raise OSError(f"実体を読めない: {key}")

    def __iter__(self) -> Iterator[str]:
        return iter(self._inner)

    def __len__(self) -> int:
        return len(self._inner)


class TestTheGraphFileIsPlacedAtomically:
    """`krg` も `krm` と同じ 3 段（一時名 → `os.replace`）を通る。

    最終名へ直接書くと、途中で落ちた回に**切り詰められた `krg`** が最終名に残る（プロセスの
    強制終了では例外経路の後始末すら走らない）。
    """

    def test_a_failed_swap_leaves_the_previous_graph_file_untouched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        graph_path = tmp_path / "model.krg"
        graph_path.write_bytes(SENTINEL)
        real = os.replace

        def failing(src: object, dst: object) -> None:
            if Path(str(dst)) == graph_path:
                raise OSError("据え替えに失敗した")
            real(src, dst)  # type: ignore[arg-type]

        monkeypatch.setattr(publish.os, "replace", failing)

        with pytest.raises(OSError, match="据え替えに失敗した"):
            publish_material(tmp_path, material(), graph_path=graph_path)

        # 最終名へ直接書いていれば、ここで前回のバイト列は既に上書きされている。
        assert graph_path.read_bytes() == SENTINEL
        # 一時名（`.partial`）の残骸も残さない。
        assert sorted(entry.name for entry in tmp_path.iterdir()) == ["model.krg"]

    def test_a_successful_publish_replaces_the_graph_file(self, tmp_path: Path) -> None:
        """対照 — 通った回は新しい `krg` が据わる（上の門が「常に消さない」ではない）。"""
        graph_path = tmp_path / "model.krg"
        graph_path.write_bytes(SENTINEL)

        result = publish_material(tmp_path, material(), graph_path=graph_path)

        assert result.graph == graph_path
        assert graph_path.read_bytes() != SENTINEL


class TestTheCleanupTouchesOnlyWhatItWrote:
    """例外経路が消すのは**この呼び出しが据えた現物**だけ。

    `graph_path` を無条件に消すと、書き出しが `krg` を抜く前に落ちた回に前回の `krg` が
    道連れになる — 自分が触っていないファイルを消すのは後始末ではなく破壊。
    """

    def test_a_failure_before_the_graph_is_extracted_keeps_the_previous_graph_file(
        self, tmp_path: Path
    ) -> None:
        graph_path = tmp_path / "model.krg"
        graph_path.write_bytes(SENTINEL)
        stored = material()

        with pytest.raises(OSError, match="実体を読めない"):
            publish_container(
                tmp_path / "model.krm",
                stored.graph,
                _Exploding(stored.tensors),
                stored.bindings,
                graph_name=GRAPH_NAME,
                provenance=FIXTURE_PROVENANCE,
                graph_path=graph_path,
            )

        assert graph_path.read_bytes() == SENTINEL
        assert sorted(entry.name for entry in tmp_path.iterdir()) == ["model.krg"]


class TestTheScaleAgreement:
    """渡した束縛と書いた容器の供給が **scale の有無**で食い違ったら落とす。

    片方だけ `None` を素通しすると、「宣言 i4 / 実体 i8」の相方（scale を持つはずの席が
    scale 無しで据わる）がこの門を抜ける。被験体は**渡す束縛表**だけを壊した組で、書いた容器
    そのものは正しい — 読み直し検証以外に検出器が無い形。
    """

    def _quantized_key(self, stored: StoredModel) -> str:
        keys = sorted(
            key for key, encoding in stored.bindings.items() if encoding.scale_key is not None
        )
        assert keys, "i8 の合成に量子化席が 1 つも無い"
        return keys[0]

    def test_a_binding_without_the_scale_key_fails_loudly(self, tmp_path: Path) -> None:
        stored = material("i8")
        key = self._quantized_key(stored)
        read_back, bound = _published(tmp_path, stored)
        doctored = {**stored.bindings, key: replace(stored.bindings[key], scale_key=None)}

        with pytest.raises(PublishError, match="scale"):
            publish._assert_payloads_match(read_back, bound, doctored, stored.tensors)

    def test_the_matching_binding_passes(self, tmp_path: Path) -> None:
        """対照 — 束縛が合っていれば通る（上の門が「常に落ちる」ではない）。"""
        stored = material("i8")
        read_back, bound = _published(tmp_path, stored)

        checked = publish._assert_payloads_match(read_back, bound, stored.bindings, stored.tensors)

        assert checked > len(bound.supplies)


def _published(tmp_path: Path, stored: StoredModel):
    """合成の部品を据え、読み直した容器と合流結果を返す（門の単体試験の土台）。"""
    result = publish_material(tmp_path, stored)
    read_back = publish.read_container(list(result.parts))
    return read_back, publish.bind_graphs(read_back.graph, read_back.model)[GRAPH_NAME]
