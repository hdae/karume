"""据えた容器を格納のまま読み戻す recipe 側の読み手（`_shared.container_read`）。

ここが固定するのは 2 点だけ:

- **開いた容器を使い回せる**（`open_container` → 各読み口）。代表 path を渡すたびに
  `verify_container` が descriptor の parse と合流をやり直すので、block ごとに読む呼び手
  （gemma4 の PLE 検収門）はその繰り返しを block 数ぶん払っていた。
- **診断はリポの語彙**（`ContainerReadError`）。`KeyError` は `repr` が二重引用符で包まれて
  読みにくく、呼び手の `except` が「辞書の引き損ない」と区別できない。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from container_series import write_component
from ir_fixtures import ir_container

from _shared.container_read import (
    ContainerReadError,
    open_container,
    read_asset,
    read_asset_declarations,
    read_layouts,
    read_stored,
)
from karume.container import AssetInput

ASSET_NAME = "rope_base"
ASSET_ROLE = "rope-base"
ASSET_PAYLOAD = bytes(range(64))


@pytest.fixture
def component(tmp_path: Path) -> Path:
    """資産を 1 本同梱した正当なコンポーネント（代表 path）。"""
    path = tmp_path / "series" / "dit" / "model.krm"
    write_component(
        path,
        ir_container(
            mark="reader",
            assets={ASSET_NAME: AssetInput(ASSET_ROLE, len(ASSET_PAYLOAD), ASSET_PAYLOAD)},
        ),
    )
    return path


class TestTheOpenedContainerIsReusable:
    """`open_container` の戻りを渡した読みは、代表 path を渡した読みと同じ答えを返す。"""

    def test_the_asset_reads_the_same_bytes_from_either_entry(self, component: Path) -> None:
        opened = open_container(component)

        assert read_asset(opened, ASSET_NAME) == read_asset(component, ASSET_NAME)
        assert read_asset(opened, ASSET_NAME) == ASSET_PAYLOAD

    def test_the_declarations_and_layouts_agree_from_either_entry(self, component: Path) -> None:
        opened = open_container(component)

        assert read_asset_declarations(opened) == read_asset_declarations(component)
        assert read_layouts(opened) == read_layouts(component)
        assert read_stored(opened).keys() == read_stored(component).keys()

    def test_reusing_the_opened_container_opens_nothing_again(
        self, component: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """故障注入 — 開き直す実装へ戻すと、ここで `verify_container` がもう一度呼ばれる。"""
        opened = open_container(component)
        from _shared import container_read

        def forbidden(*args: object, **kwargs: object) -> None:
            raise AssertionError("開いた容器を渡したのに verify_container を回した")

        monkeypatch.setattr(container_read, "verify_container", forbidden)

        assert read_asset(opened, ASSET_NAME) == ASSET_PAYLOAD
        assert read_asset_declarations(opened)[ASSET_NAME] == (ASSET_ROLE, len(ASSET_PAYLOAD))


class TestTheDiagnosticsUseTheRepositoryVocabulary:
    def test_a_missing_asset_fails_with_the_reader_error(self, component: Path) -> None:
        with pytest.raises(ContainerReadError, match="資産 'absent' が無い"):
            read_asset(component, "absent")

    def test_the_error_is_not_a_key_error(self, component: Path) -> None:
        """`KeyError` のままだと呼び手の `except KeyError` が辞書の引き損ないと区別できない。"""
        assert not issubclass(ContainerReadError, KeyError)
