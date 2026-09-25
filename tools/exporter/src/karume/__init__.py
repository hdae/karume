"""Karume exporter — torch.export 済みモデルを IR（docs/ir-v2.md）とコンテナ（`krm`）へ落とす。

公開面は `__all__` の 12 件が正本（列挙とここの説明は 1:1 — 機械門は
`tests/test_architecture_boundary.py` の TestThePublicSurface）:
IR の型（`IrGraph`）・変換（`convert` /
`curated_decompositions` / `normalize_graph`）・格納変換（`FixedQuantizedWeight` /
`stored_model`）・出所（`Provenance`）・検証（`parse_ir_graph` / `verify_container`）・
一本道（`export_module` / `export_to_file` / `publish_model`）。

配布形を作る経路は `publish_model` の 1 本を通す（格納変換 → 書く → 読み直して検証 →
据え替え — `pipeline.publish_model` と `publish.publish_container` の docstring）。

公開面は初回参照で解決する（PEP 562）。`karume.dist` / `karume.modelcard` / `karume.container`
のような torch 不要のモジュールは、パッケージの初期化で torch（と torchvision）を読まない。
eager に import すると `import karume.dist` だけで torch の起動コストが丸ごと乗る
（`tests/test_package_init.py` が固定する）。
"""

# 別名は `_` 付き — 公開面の名前空間（属性と `dir()`）に実装の道具を載せない。
import sys as _sys
from importlib import import_module as _import_module
from types import ModuleType as _ModuleType

#: 公開名 → 定義元モジュール。
_EXPORTS: dict[str, str] = {
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

__all__ = [
    "FixedQuantizedWeight",
    "IrGraph",
    "Provenance",
    "convert",
    "curated_decompositions",
    "export_module",
    "export_to_file",
    "normalize_graph",
    "parse_ir_graph",
    "publish_model",
    "stored_model",
    "verify_container",
]


def __getattr__(name: str) -> object:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})


class _Package(_ModuleType):
    def __setattr__(self, name: str, value: object) -> None:
        # サブモジュールの初回 import は親パッケージへ同名の属性を張る。公開関数 `convert` は
        # サブモジュール `karume.convert` と同名なので、張らせると `karume.convert` /
        # `from karume import convert` がモジュールに化ける（`karume.pipeline` を先に import
        # するだけで起きる）。公開名は常に `__getattr__` の解決先を指す MUST。
        if name in _EXPORTS and isinstance(value, _ModuleType):
            return
        super().__setattr__(name, value)


_sys.modules[__name__].__class__ = _Package
