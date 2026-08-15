"""core → recipe の逆流を機械で止める門（ADR 0065 決定 3）。

exporter は「汎用 exporter core（PyPI 配布物）」と「モデル別 recipe（リポ専用）」へ段階的に
分離する途中で、依存方向は **recipe → core の一方向だけ**が許される。規約として書くだけでは
段が進むたびに逆流が忍び込むので、ここで import を機械検査する（同 ADR: 境界は規約でなく
テストが守る）。

検査は 2 本:

1. core が **recipe 側モジュール**（`karume.patch_*` / `anima_text` / `dist` / `modelcard` /
   `cli` / `paths`）を import しない
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
#: NOTE: このリスト外の `karume.*`（`patch_*` / `anima_text` / `dist` / `modelcard` / `cli` /
#: `paths` / `lora` / `resolution`）は **境界検査の対象外** — 段階移行中で、recipe 側へ出る
#: ものと core に昇格するものが未分離だから。段が進むたびにここへ足していく（ADR 0065 段階）。
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
    # 段 2 で patch_anima から回収したモデル非依存の export 検証ヘルパ。
    "rope",
)

#: recipe 側（core から import してはならない）モジュール名の接頭辞。
RECIPE_MODULE_PREFIXES = ("patch_",)

#: recipe 側（core から import してはならない）モジュール名。
RECIPE_MODULES = frozenset({"anima_text", "cli", "dist", "modelcard", "paths"})

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


def imported_modules(module_name: str) -> list[tuple[int, str]]:
    """`karume.<module_name>` が import する完全修飾モジュール名を行番号付きで全部採る。

    `ast.walk` なので関数内・条件内の遅延 import も拾う。`from karume import dist` の形は
    「`karume`」と「`karume.dist`」の両方を返す（後者が実体としての依存で、シンボル名か
    サブモジュール名かは AST からは区別できない — 禁止名と衝突したら違反として扱う）。
    """
    source = (KARUME_ROOT / f"{module_name}.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename=f"karume/{module_name}.py")
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_import_from_base(node, module_name)
            if base:
                found.append((node.lineno, base))
            found.extend(
                (node.lineno, f"{base}.{alias.name}" if base else alias.name)
                for alias in node.names
            )
    return found


def _resolve_import_from_base(node: ast.ImportFrom, module_name: str) -> str:
    """`from ... import` の起点を完全修飾名にする（相対 import を絶対化する）。"""
    if node.level == 0:
        return node.module or ""
    # core モジュールは全て `karume` 直下なので、相対 1 段の起点は `karume`。
    # 2 段以上はパッケージの外へ出る形で、core には現れない（現れたら fail loudly）。
    if node.level > 1:
        raise ValueError(f"karume/{module_name}.py:{node.lineno}: パッケージ外への相対 import")
    return f"karume.{node.module}" if node.module else "karume"


def recipe_imports(module_name: str) -> list[str]:
    """recipe 側モジュールへの import を「行:import 名」の一覧で返す（無ければ空）。"""
    violations = []
    for lineno, dotted in imported_modules(module_name):
        parts = dotted.split(".")
        if parts[0] != "karume" or len(parts) < 2:
            continue
        submodule = parts[1]
        if submodule.startswith(RECIPE_MODULE_PREFIXES) or submodule in RECIPE_MODULES:
            violations.append(f"karume/{module_name}.py:{lineno} -> {dotted}")
    return violations


def upstream_imports(module_name: str) -> list[str]:
    """上流モデル系パッケージへの import を「行:import 名」の一覧で返す（無ければ空）。"""
    return [
        f"karume/{module_name}.py:{lineno} -> {dotted}"
        for lineno, dotted in imported_modules(module_name)
        if dotted.split(".")[0] in UPSTREAM_MODEL_PACKAGES
    ]


class TestCoreDoesNotImportRecipeModules:
    @pytest.mark.parametrize("module_name", RECIPE_GATED_MODULES)
    def test_a_core_module_imports_no_recipe_module(self, module_name: str) -> None:
        """core は `patch_*` / family 知識を焼いたモジュールへ依存しない（依存は recipe → core）。

        逆流すると wheel の物理境界（`karume/` ディレクトリ全体）に family 知識が入る。
        """
        assert recipe_imports(module_name) == []


class TestCoreDoesNotImportUpstreamModelPackages:
    @pytest.mark.parametrize("module_name", UPSTREAM_GATED_MODULES)
    def test_a_core_module_imports_no_upstream_model_package(self, module_name: str) -> None:
        """core wheel は上流モデル実装（とその provenance 義務）を引き込まない。"""
        assert upstream_imports(module_name) == []


class TestTheBoundaryCheckItself:
    """検査が本当に違反を検出できるかの故障注入（恒真化の門）。

    core 集合が緑なのは「違反が無いから」であって「検査が何も見ていないから」ではない、を
    実在の違反モジュールで示す。ここで挙げる 2 本は recipe 側なので core 集合には入れない。
    """

    def test_it_catches_the_lazy_upstream_import_in_patch_anima(self) -> None:
        """`patch_anima` は関数内で diffusers を import する — 遅延 import も見えている。"""
        violations = upstream_imports("patch_anima")

        assert violations, "patch_anima の diffusers import を検出できていない"
        assert all("diffusers" in violation for violation in violations)

    def test_it_catches_the_lazy_recipe_import_in_cli(self) -> None:
        """`cli` は関数内で `from karume import dist` する — `from X import <submodule>` も見る。"""
        violations = recipe_imports("cli")

        assert any(violation.endswith("-> karume.dist") for violation in violations)
