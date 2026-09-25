"""パッケージ `karume` の初期化が torch を読まず、公開面の解決先が eager import と同じであること。

`karume/__init__` は公開面を初回参照で解決する（PEP 562）。主張はどれも「どのモジュールが
既に import 済みか」に依存するので、テストセッションの `sys.modules` を使わず、毎回新しい
インタプリタで確かめる。
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

#: 公開名 → 定義元モジュール（eager import だった頃の `from <module> import <name>` の写し）。
#: 公開面の名前集合と各名前の解決先の仕様で、`karume._EXPORTS` の写しではない。
EXPECTED_ORIGINS: dict[str, str] = {
    "FixedQuantizedWeight": "karume.emit",
    "IrGraph": "karume.ir",
    "Provenance": "karume.container",
    "convert": "karume.convert",
    "curated_decompositions": "karume.convert",
    "export_module": "karume.pipeline",
    "export_to_file": "karume.pipeline",
    "normalize_graph": "karume.normalize",
    "parse_ir_graph": "karume.verify",
    "publish_model": "karume.pipeline",
    "stored_model": "karume.emit",
    "verify_container": "karume.verify",
}

#: torch を要らないモジュール。配布・カード層（recipes の dist ドライバ）はこの連鎖だけで組み、
#: 移行 CLI（`karume migrate`）はこれに旧形の読み手・公開の 3 段・PLE / limits を足した連鎖で
#: 組む。CLI の入口も dispatch を遅延しているので、import しただけでは torch を読まない。
TORCH_FREE_MODULES: tuple[str, ...] = (
    "karume",
    "karume.dist",
    "karume.modelcard",
    "karume.container",
    "karume.verify",
    "karume.migrate",
    "karume.publish",
    "karume.legacy",
    "karume.ple",
    "karume.limits",
    "karume.artifacts",
    "karume.cli",
)

#: 定義元が torch を要らない公開名（`EXPECTED_ORIGINS` のうち ir / container / verify の分）。
TORCH_FREE_EXPORTS: tuple[str, ...] = tuple(
    name
    for name, module_name in EXPECTED_ORIGINS.items()
    if module_name in {"karume.ir", "karume.container", "karume.verify"}
)


def run_python(source: str) -> str:
    completed = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


class TestImportingTorchFreeModules:
    @pytest.mark.parametrize("module_name", TORCH_FREE_MODULES)
    def test_torch_is_not_loaded(self, module_name: str) -> None:
        loaded = run_python(
            f"import {module_name}, sys\n"
            "print(sorted(m for m in ('torch', 'torchvision') if m in sys.modules))"
        )

        assert loaded.strip() == "[]"

    @pytest.mark.parametrize("name", TORCH_FREE_EXPORTS)
    def test_touching_a_torch_free_export_does_not_load_torch(self, name: str) -> None:
        """同じオブジェクトでも torch 依存のモジュール経由で解決すると、触れるだけで torch が乗る。

        定義元を指す `_EXPORTS` の行がずれたときに赤になる（解決先の同一性だけでは区別できない）。
        """
        loaded = run_python(f"import karume, sys\nkarume.{name}\nprint('torch' in sys.modules)")

        assert loaded.strip() == "False"

    def test_touching_an_export_that_needs_torch_loads_it(self) -> None:
        """遅延は「読まない」ではなく「必要になったら読む」— 上の門が恒真でないことの対照。"""
        loaded = run_python("import karume, sys\nkarume.convert\nprint('torch' in sys.modules)")

        assert loaded.strip() == "True"


#: 名前ごとに「公開面の値が定義元モジュールの同名属性と同一か」を JSON で返す台本。
#: `{prelude}` で先に import しておくモジュールを差し替え、解決の順序を変える。
#: `dir` は解決より前に採る（解決後はどの名前もモジュールの辞書に載っていて区別できない）。
RESOLUTION_PROBE = """
import importlib, json
{prelude}
import karume
origins = json.loads({origins!r})
report = {{"dir": sorted(set(karume.__all__) - set(dir(karume)))}}
for name, module_name in origins.items():
    report[name] = getattr(karume, name) is getattr(importlib.import_module(module_name), name)
from karume import convert
report["from karume import convert"] = convert is importlib.import_module("karume.convert").convert
report["__all__"] = sorted(karume.__all__)
print(json.dumps(report))
"""


class TestThePublicSurfaceResolvesLikeTheEagerImport:
    @pytest.mark.parametrize(
        "prelude",
        [
            pytest.param("", id="package-first"),
            # `karume.convert` は公開関数 `convert` と同名のサブモジュール。先に読まれると import
            # システムが親へ同名の属性を張るので、公開名がモジュールに化けないかをここで見る。
            pytest.param(
                "import " + ", ".join(sorted(set(EXPECTED_ORIGINS.values()))),
                id="submodules-first",
            ),
        ],
    )
    def test_every_name_is_the_object_its_origin_module_defines(self, prelude: str) -> None:
        report = json.loads(
            run_python(
                RESOLUTION_PROBE.format(prelude=prelude, origins=json.dumps(EXPECTED_ORIGINS))
            )
        )

        assert report.pop("__all__") == sorted(EXPECTED_ORIGINS)
        assert report.pop("dir") == []
        mismatched = [name for name, same in report.items() if not same]
        assert mismatched == []

    def test_a_non_exported_submodule_stays_reachable_as_an_attribute(self) -> None:
        """公開名と同名でないサブモジュールは、import 後に親の属性として張られたままである。

        ガード（公開名と同名のモジュール属性だけ捨てる）を広げすぎると `karume.dist` が
        AttributeError になる — その変異をここで赤にする。
        """
        reachable = run_python(
            "import sys, karume.dist, karume.modelcard\n"
            "print(karume.dist is sys.modules['karume.dist'],"
            " karume.modelcard is sys.modules['karume.modelcard'])"
        )

        assert reachable.split() == ["True", "True"]

    def test_an_unknown_name_raises_attribute_error(self) -> None:
        import karume

        with pytest.raises(AttributeError, match="no_such_export"):
            _ = karume.no_such_export
