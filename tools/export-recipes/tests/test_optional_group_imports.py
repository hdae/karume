"""依存グループ側のパッケージを **module 直下で import しない**（リポ全体の作法の門）。

`[dependency-groups]` の `dev` 以外（anima / sbv2 / siglip2 / birefnet / …）は**既定の
`uv sync` に入らない**（pyproject の同 NOTE）。それらを module 直下で import すると、その
モジュールを**読むだけ**で `ModuleNotFoundError` になり、グループを同期していない環境では
**pytest が collection ごと落ちる**。

MUST: この門が要る理由は「破っても開発環境では見えない」こと。日常的にグループを同期して
いる手元では全部通るので、壊したことに気づくのは**既定 sync の誰か**が回した時になる
（2026-08-22 に実際に踏んだ — `anima/single_file.py` が `diffusers` と `huggingface_hub` を
module 直下で import し、テストが 1 本 collection error になった）。9 つの recipe すべてが
関数内 import で揃っていた慣習を、ここで機械の検査に変える。

使う側の作法は「要る関数の中で import する」（`anima/export.py` / `anima/patch.py` /
`anima/lora.py` / `anima/pipeline_ref.py` などが実例）。テストから触るなら
`pytest.importorskip` で守る（`anima/tests/test_patch.py` などが実例）。
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

#: recipe の親（`tools/export-recipes/`）。
ROOT = Path(__file__).resolve().parent.parent

#: 常に入っている依存（core `karume` の dependencies）と、この検査の対象外グループ。
#: `dev` は開発ツールで、テスト実行時には必ず入っている。
ALWAYS_SYNCED_GROUPS = frozenset({"dev"})


def _import_names(requirement: str) -> str:
    """依存の綴り（配布名）を import 名へ寄せる（`huggingface-hub` → `huggingface_hub`）。

    版指定を落として `-` を `_` にするだけ。配布名と import 名が一致しない依存が入ったら
    ここが取りこぼすが、**取りこぼしは検査が緩む方向**なので偽陽性で騒がない側に倒している。
    """
    name = requirement
    for separator in ("==", ">=", "<=", "~=", "!=", ">", "<", "["):
        name = name.split(separator, 1)[0]
    return name.strip().replace("-", "_")


def _gated_modules() -> frozenset[str]:
    """既定 sync に入らないグループが持ち込む import 名の集合（pyproject から導出）。

    表を第 2 の場所に持たない — グループを足した日にこの門が自動で追随する。
    """
    groups = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    gated: set[str] = set()
    for name, requirements in groups["dependency-groups"].items():
        if name in ALWAYS_SYNCED_GROUPS:
            continue
        gated.update(_import_names(requirement) for requirement in requirements)
    return frozenset(gated)


def _recipe_sources() -> list[Path]:
    """recipe 本体の `.py`（テストと `__pycache__` は除く）。"""
    return sorted(
        path
        for path in ROOT.glob("*/**/*.py")
        if "tests" not in path.parts and "__pycache__" not in path.parts
    )


def _module_level_imports(source: Path) -> set[str]:
    """module 直下（関数・クラスの中ではない）で import している最上位モジュール名。"""
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module is not None:
            names.add(node.module.split(".", 1)[0])
    return names


class TestOptionalGroupImports:
    def test_the_gated_set_is_not_empty(self) -> None:
        """導出が空になったら（pyproject の形が変わったら）この門は恒真になる。"""
        gated = _gated_modules()

        assert "diffusers" in gated
        assert "huggingface_hub" in gated
        assert "transformers" in gated

    def test_it_finds_the_recipe_sources(self) -> None:
        """走査が 0 本なら、やはり恒真になる。"""
        sources = _recipe_sources()

        assert len(sources) > 50
        assert any(path.name == "single_file.py" for path in sources)

    def test_no_recipe_imports_a_gated_module_at_module_level(self) -> None:
        gated = _gated_modules()
        offenders = {
            str(source.relative_to(ROOT)): sorted(_module_level_imports(source) & gated)
            for source in _recipe_sources()
            if _module_level_imports(source) & gated
        }

        assert offenders == {}, (
            "既定 sync に入らない依存を module 直下で import している — 要る関数の中へ移す"
            f"（グループ非同期の環境では読むだけで落ちる）: {offenders}"
        )
