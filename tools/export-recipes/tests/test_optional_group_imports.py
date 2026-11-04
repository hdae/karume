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
import functools
import importlib.metadata
import re
import tomllib
from pathlib import Path

#: recipe の親（`tools/export-recipes/`）。
ROOT = Path(__file__).resolve().parent.parent

#: 常に入っている依存（core `karume` の dependencies）と、この検査の対象外グループ。
#: `dev` は開発ツールで、テスト実行時には必ず入っている。
ALWAYS_SYNCED_GROUPS = frozenset({"dev"})

#: core（PyPI `karume`）の pyproject（`[project] dependencies` = 常に同期される依存）。
CORE_PYPROJECT = ROOT.parent / "exporter" / "pyproject.toml"

#: グループの**宣言**には現れないが既定 sync に入らない import 名（推移依存）。
#:
#: `tokenizers` は `transformers` の依存として入るだけなので、グループの宣言だけを見る
#: 導出には現れない — それを module 直下へ書くと、この門は緑のまま既定 sync の環境が
#: collection error になる（`tests/test_gemma_tokenizer.py` の `importorskip` は人手で
#: この穴を埋めている側）。
EXTRA_GATED = frozenset({"tokenizers"})


def _canonical_distribution(name: str) -> str:
    """The normalized distribution name of PEP 503 (`Pillow` / `huggingface_hub` →
    `pillow` / `huggingface-hub`), so a requirement and the installed metadata compare equal."""
    return re.sub(r"[-_.]+", "-", name).lower()


@functools.cache
def _installed_import_names() -> dict[str, frozenset[str]]:
    """Installed distribution (canonical name) → the top-level import names it provides.

    Taken from the package metadata (`importlib.metadata.packages_distributions`), not from a hand
    table — a table would be a second place to keep in step with the groups.
    """
    names: dict[str, set[str]] = {}
    for module, distributions in importlib.metadata.packages_distributions().items():
        for distribution in distributions:
            names.setdefault(_canonical_distribution(distribution), set()).add(module)
    return {distribution: frozenset(modules) for distribution, modules in names.items()}


def _import_names(requirement: str) -> frozenset[str]:
    """A requirement (a distribution name with a version) → its import names (`pillow` → `PIL`).

    An installed distribution answers from its metadata. A distribution that is not installed
    falls back to its name with `-` → `_` (`huggingface-hub` → `huggingface_hub`): in that
    environment a module-level import of it fails at collection anyway, so the gate matters where
    the group is synced — exactly where the metadata is present.
    """
    name = requirement
    for separator in ("==", ">=", "<=", "~=", "!=", ">", "<", "["):
        name = name.split(separator, 1)[0]
    name = name.strip()
    installed = _installed_import_names().get(_canonical_distribution(name))
    return installed if installed else frozenset({name.replace("-", "_")})


def _always_synced_modules() -> frozenset[str]:
    """core の `[project] dependencies`（グループと無関係に必ず同期される import 名）。

    グループが宣言していても、core の基本依存にある名前は gated ではない — `torchvision` は
    `anima` / `siglip2-preprocess` グループが宣言しているが、`torchvision::deform_conv2d` の
    op 登録のために core が常に引く（`tools/exporter/pyproject.toml`）。除かないと、正当な
    module 直下 `import torchvision` をこの門が誤って落とす。
    """
    core = tomllib.loads(CORE_PYPROJECT.read_text(encoding="utf-8"))
    return frozenset(
        module
        for requirement in core["project"]["dependencies"]
        for module in _import_names(requirement)
    )


def _gated_modules() -> frozenset[str]:
    """既定 sync に入らないグループが持ち込む import 名の集合（pyproject から導出）。

    表を第 2 の場所に持たない — グループを足した日にこの門が自動で追随する。宣言だけでは
    足りないぶん（推移依存 = {@link EXTRA_GATED}）と、宣言されていても常に同期されるぶん
    （core の基本依存）を、それぞれ足し引きする。
    """
    groups = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    gated: set[str] = set(EXTRA_GATED)
    for name, requirements in groups["dependency-groups"].items():
        if name in ALWAYS_SYNCED_GROUPS:
            continue
        for requirement in requirements:
            gated.update(_import_names(requirement))
    return frozenset(gated - _always_synced_modules())


def _recipe_sources() -> list[Path]:
    """recipe 本体の `.py`（テストと `__pycache__` は除く）。

    ルート直下（`dist.py` / `conftest.py`）も対象 — とくに `conftest.py` は pytest が必ず
    読むので、そこに gated import が入れば門が守るはずの collection error がそのまま起きる。
    """
    return sorted(
        path
        for path in {*ROOT.glob("*.py"), *ROOT.glob("*/**/*.py")}
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

    def test_the_gated_set_covers_the_transitive_dependency(self) -> None:
        """宣言に無い推移依存（`transformers` 経由の `tokenizers`）も gated 側。"""
        assert "tokenizers" in _gated_modules()

    def test_the_gated_set_holds_import_names_not_distribution_names(self) -> None:
        """`pillow` is imported as `PIL` — gating the distribution spelling would let a
        module-level `import PIL` through (pillow is installed as a torchvision dependency, so its
        metadata is present even in the default sync)."""
        gated = _gated_modules()

        assert "PIL" in gated
        assert "pillow" not in gated

    def test_a_core_dependency_is_never_gated(self) -> None:
        """`torchvision` はグループが宣言していても core が常に引く（誤検出の側を閉じる）。"""
        assert "torchvision" not in _gated_modules()

    def test_it_finds_the_recipe_sources(self) -> None:
        """走査が 0 本なら、やはり恒真になる。"""
        sources = _recipe_sources()

        assert len(sources) > 50
        assert any(path.name == "single_file.py" for path in sources)
        # ルート直下の 2 本（`conftest.py` は pytest が必ず読む）を取りこぼさない。
        assert any(path.name == "dist.py" for path in sources)
        assert any(path.name == "conftest.py" for path in sources)

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
