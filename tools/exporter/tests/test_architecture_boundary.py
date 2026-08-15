"""core → recipe の逆流を機械で止める門（ADR 0065 決定 3）。

exporter は「汎用 exporter core（PyPI 配布物）」と「モデル別 recipe（リポ専用）」へ段階的に
分離する途中で、依存方向は **recipe → core の一方向だけ**が許される。規約として書くだけでは
段が進むたびに逆流が忍び込むので、ここで import を機械検査する（同 ADR: 境界は規約でなく
テストが守る）。

検査は 2 本:

1. core が **recipe 側モジュール**（`karume.patch_*` — 今は 1 本も無いが再発防止で残す門 —
   と `dist` / `modelcard` / `cli` / `paths`）を import しない
2. core が **上流モデル系パッケージ**（`style_bert_vits2` / `transformers` / `diffusers` /
   `torchvision`）を import しない — core wheel に上流実装の provenance 義務を引き込まない
   ための門（ADR 0065 決定 7）

走査は文字列 grep でなく `ast` — 関数内の遅延 import（`karume.cli` の
`from karume import dist` のような形）が段階移行の逃げ道にならないようにする。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

#: 検査対象パッケージのソース置き場。
KARUME_ROOT = Path(__file__).resolve().parents[1] / "karume"

#: 汎用 exporter core として残す集合（ADR 0065 決定 1 の一覧）。
#:
#: MUST: **明示リストで持つ**。`karume/` を舐めて対象を決めると、新しいモジュールが増えたとき
#: 「core なのか recipe なのか」の判断を素通りして黙って gate に入る（または入らない）。
#: NOTE: このリスト外の `karume.*`（`dist` / `modelcard` / `cli` / `paths`）は**境界検査の
#: 対象外**。`dist` / `modelcard` から family 知識は 1 つも残らず出た（ADR 0065 段 3+4 完了）が、
#: `dist` が `karume.paths`（repo topology 依存）を import しているので、ここへ足すのは段 5 の
#: paths 決着の後（`paths` が recipe 側の関心事へ回る — 同 ADR Consequences）。
CORE_MODULES: tuple[str, ...] = (
    "__init__",
    "ir",
    "ops",
    "dims",
    "extents",
    "shapes",
    "convert",
    "normalize",
    "quantize",
    "act_quant",
    "emit",
    "verify",
    "pipeline",
    "goldens",
    "custom_ops",
    # 段 2 で Anima の patch 層から回収したモデル非依存の export 検証ヘルパ。
    "rope",
)

#: recipe 側（core から import してはならない）モジュール名の接頭辞。
#:
#: NOTE: `karume/patch_*.py` は**もう 1 本も無い**（ADR 0065 段 3+4 で全 family が
#: `tools/export-recipes/<family>/patch.py` へ出た）。それでも禁止リストに残すのは**再発防止**
#: — 上流モデル由来のパッチ層を core へ書き戻す最短の道がこの綴りで、消すと「気づいたら
#: `karume/patch_foo.py` が生えていた」を止める門が無くなる。恒真化していないことは
#: {@link TestTheBoundaryCheckItself} の合成故障注入が示す。
RECIPE_MODULE_PREFIXES = ("patch_",)

#: recipe 側（core から import してはならない）モジュール名。
RECIPE_MODULES = frozenset({"cli", "dist", "modelcard", "paths"})

#: core wheel に持ち込まない上流モデル系パッケージ。
UPSTREAM_MODEL_PACKAGES = frozenset(
    {"diffusers", "style_bert_vits2", "torchvision", "transformers"}
)

#: 検査 2 の一時除外。**検査を弱めるのではなく、既知の違反を明示して残す**。
#:
#: NOTE: `convert` は `torch.ops.torchvision.*`（deform_conv2d）のハンドラキーを引くために
#: torchvision を基本依存として素で import する（ADR 0055 決定 7）。`goldens` は同 op の
#: torch 突合に `torchvision.ops.deform_conv2d` を呼ぶ。どちらも「上流モデル実装のコピー」では
#: ないが、core wheel の依存として妥当かは ADR 0065 段 6（packaging / provenance）の裁定事項。
UPSTREAM_CHECK_EXEMPT = frozenset({"convert", "goldens"})

#: 検査 1 の一時除外。同じく既知の違反を明示して残す。
#:
#: NOTE: `goldens` は生成先（リポ直下 `packages/runtime/tests/fixtures/golden/`）を
#: `karume.paths.REPO_ROOT` から引く。`paths` は repo topology 依存で recipe 側の関心事へ回る
#: 予定（ADR 0065 Consequences）なので、goldens が Path を受け取る形へ変わるまでの除外。
RECIPE_CHECK_EXEMPT = frozenset({"goldens"})

RECIPE_GATED_MODULES = tuple(name for name in CORE_MODULES if name not in RECIPE_CHECK_EXEMPT)
UPSTREAM_GATED_MODULES = tuple(name for name in CORE_MODULES if name not in UPSTREAM_CHECK_EXEMPT)


def core_module(module_name: str) -> Path:
    """core モジュール名 → ソースの場所。

    検査を **path 越し**にしてあるのは、下の故障注入が合成ソースを**同じ走査コード**へ
    掛けられるようにするため（被験体を実在モジュールに預けると、その family が recipe 側へ
    出た瞬間に門が黙って恒真化する — {@link TestTheBoundaryCheckItself}）。
    """
    return KARUME_ROOT / f"{module_name}.py"


def imported_modules(path: Path) -> list[tuple[int, str]]:
    """`path` のソースが import する完全修飾モジュール名を行番号付きで全部採る。

    `ast.walk` なので関数内・条件内の遅延 import も拾う。`from karume import dist` の形は
    「`karume`」と「`karume.dist`」の両方を返す（後者が実体としての依存で、シンボル名か
    サブモジュール名かは AST からは区別できない — 禁止名と衝突したら違反として扱う）。
    """
    label = _label(path)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=label)
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_import_from_base(node, label)
            if base:
                found.append((node.lineno, base))
            found.extend(
                (node.lineno, f"{base}.{alias.name}" if base else alias.name)
                for alias in node.names
            )
    return found


def _label(path: Path) -> str:
    """違反報告の見出し。走査対象は `karume` 直下のモジュールとして読む（相対 import の起点）。"""
    return f"karume/{path.name}"


def _resolve_import_from_base(node: ast.ImportFrom, label: str) -> str:
    """`from ... import` の起点を完全修飾名にする（相対 import を絶対化する）。"""
    if node.level == 0:
        return node.module or ""
    # core モジュールは全て `karume` 直下なので、相対 1 段の起点は `karume`。
    # 2 段以上はパッケージの外へ出る形で、core には現れない（現れたら fail loudly）。
    if node.level > 1:
        raise ValueError(f"{label}:{node.lineno}: パッケージ外への相対 import")
    return f"karume.{node.module}" if node.module else "karume"


def recipe_imports(path: Path) -> list[str]:
    """recipe 側モジュールへの import を「行:import 名」の一覧で返す（無ければ空）。"""
    violations = []
    for lineno, dotted in imported_modules(path):
        parts = dotted.split(".")
        if parts[0] != "karume" or len(parts) < 2:
            continue
        submodule = parts[1]
        if submodule.startswith(RECIPE_MODULE_PREFIXES) or submodule in RECIPE_MODULES:
            violations.append(f"{_label(path)}:{lineno} -> {dotted}")
    return violations


def upstream_imports(path: Path) -> list[str]:
    """上流モデル系パッケージへの import を「行:import 名」の一覧で返す（無ければ空）。"""
    return [
        f"{_label(path)}:{lineno} -> {dotted}"
        for lineno, dotted in imported_modules(path)
        if dotted.split(".")[0] in UPSTREAM_MODEL_PACKAGES
    ]


class TestCoreDoesNotImportRecipeModules:
    @pytest.mark.parametrize("module_name", RECIPE_GATED_MODULES)
    def test_a_core_module_imports_no_recipe_module(self, module_name: str) -> None:
        """core は `patch_*` / family 知識を焼いたモジュールへ依存しない（依存は recipe → core）。

        逆流すると wheel の物理境界（`karume/` ディレクトリ全体）に family 知識が入る。
        """
        assert recipe_imports(core_module(module_name)) == []


class TestCoreDoesNotImportUpstreamModelPackages:
    @pytest.mark.parametrize("module_name", UPSTREAM_GATED_MODULES)
    def test_a_core_module_imports_no_upstream_model_package(self, module_name: str) -> None:
        """core wheel は上流モデル実装（とその provenance 義務）を引き込まない。"""
        assert upstream_imports(core_module(module_name)) == []


class TestTheBoundaryCheckItself:
    """検査が本当に違反を検出できるかの故障注入（恒真化の門）。

    core 集合が緑なのは「違反が無いから」であって「検査が何も見ていないから」ではない、を
    **合成ソース**で示す。被験体を実在の違反モジュールに預けない理由は、ADR 0065 段 4 が
    その形を 2 便続けて壊したから — 指し先の family が recipe 側へ出るたびに被験体が枯れ、
    最後の 1 本が出た時点では**差し替え先が 1 つも残らない**（最終便で `karume/patch_*.py` は
    1 本も無くなった）。門が黙って恒真化する形をここで閉じる。

    走査は本物と同じ {@link imported_modules}（`karume/` の core モジュールに掛かるのと
    1 バイトも違わない経路）— 合成なのは**被験体だけ**で、検査そのものは共有する。
    """

    @staticmethod
    def _violator(tmp_path: Path, name: str, source: str) -> Path:
        """`karume/<name>.py` のつもりの合成ソースを置く（走査は path しか見ない）。"""
        path = tmp_path / f"{name}.py"
        path.write_text(source, encoding="utf-8")
        return path

    def test_it_catches_a_lazy_upstream_import(self, tmp_path: Path) -> None:
        """関数の中に隠した上流 import も見えている（段階移行の逃げ道にしない）。"""
        path = self._violator(
            tmp_path,
            "sneaky",
            "def build():\n    from transformers import AutoModel\n\n    return AutoModel\n",
        )

        assert upstream_imports(path) == [
            "karume/sneaky.py:2 -> transformers",
            "karume/sneaky.py:2 -> transformers.AutoModel",
        ]

    def test_it_stays_quiet_on_a_module_that_imports_nothing_forbidden(
        self, tmp_path: Path
    ) -> None:
        """検出側が「何にでも反応する」のではないことの対（上の主張の裏側）。"""
        path = self._violator(tmp_path, "clean", "import json\n\nfrom karume.ir import IrGraph\n")

        assert upstream_imports(path) == []
        assert recipe_imports(path) == []

    def test_it_catches_a_lazy_recipe_import(self, tmp_path: Path) -> None:
        """`from karume import dist` の形も見る（シンボル名かサブモジュール名かは区別しない）。"""
        path = self._violator(
            tmp_path, "leaky", "def run(argv):\n    from karume import dist\n\n    return dist\n"
        )

        assert recipe_imports(path) == ["karume/leaky.py:2 -> karume.dist"]

    def test_it_catches_a_relative_recipe_import(self, tmp_path: Path) -> None:
        """相対 1 段も絶対化して見る（`from . import dist` で門をすり抜けさせない）。"""
        path = self._violator(tmp_path, "relative", "from . import modelcard\n")

        assert recipe_imports(path) == ["karume/relative.py:1 -> karume.modelcard"]

    def test_it_catches_a_patch_module_by_prefix(self, tmp_path: Path) -> None:
        """接頭辞の門（{@link RECIPE_MODULE_PREFIXES}）は名指しの表と独立に効く。

        `karume/patch_*.py` は 1 本も残っていないので、この枝を踏む実在モジュールはもう無い
        （= 名簿だけ残して検査が恒真、になりうる席）。**再発防止の門はここで踏み続ける** —
        上流モデル由来のパッチ層が core へ書き戻される最短の道がこの綴り。
        """
        path = self._violator(
            tmp_path, "regressed", "from karume.patch_newmodel import apply_all_patches\n"
        )

        assert recipe_imports(path) == [
            "karume/regressed.py:1 -> karume.patch_newmodel",
            "karume/regressed.py:1 -> karume.patch_newmodel.apply_all_patches",
        ]
