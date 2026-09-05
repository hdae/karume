"""core → recipe の逆流を機械で止める門（ADR 0065 決定 3）。

exporter は「汎用 exporter core（PyPI 配布物）」と「モデル別 recipe（リポ専用）」へ段階的に
分離する途中で、依存方向は **recipe → core の一方向だけ**が許される。規約として書くだけでは
段が進むたびに逆流が忍び込むので、ここで import を機械検査する（同 ADR: 境界は規約でなく
テストが守る）。

検査は 2 本:

1. core が **recipe 側モジュール**（`karume.patch_*` / `karume.paths` — どちらも今は 1 本も
   無いが再発防止で残す門）を import しない
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

#: 検査対象パッケージのソース置き場（`src/` layout — ADR 0065 決定 5）。
KARUME_ROOT = Path(__file__).resolve().parents[1] / "src" / "karume"

#: 汎用 exporter core として残す集合（ADR 0065 決定 1 の一覧）。
#:
#: MUST: **明示リストで持つ**。`src/karume/` を舐めて対象を決めると、新しいモジュールが増えた
#: とき「core なのか recipe なのか」の判断を素通りして黙って gate に入る（または入らない）。
#: NOTE: `dist` / `modelcard` / `cli` も段 5 で門に入った — family 知識（段 3+4）に続いて
#: repo topology（`paths`）も出たので、core wheel の中身は全部この 1 リストで見る。
CORE_MODULES: tuple[str, ...] = (
    "__init__",
    "ir",
    "ops",
    "dims",
    "extents",
    "shapes",
    "convert",
    "aten_handlers",
    "normalize",
    "quantize",
    # 測定専用の丸め方式（格納経路を持たない — quant_methods モジュール docstring）。
    # 対象選択は quantize と共有で、family 知識は持たない。
    "quant_methods",
    # 校正付き丸め（GPTQ / AWQ）。対象選択は quantize と共有・family 知識は持たない。
    "quant_calib",
    "act_quant",
    "emit",
    # 配布コンテナの shard 分割規則（ADR 0070 決定 1）。path とバイト数しか知らない。
    "shards",
    # quant の `requiredLimits` 導出（ADR 0038 §7）。寸法と WebGPU の既定値しか知らない。
    "limits",
    "verify",
    # 容器の詰め替え（ADR 0081 の移行経路）。バイト列と宣言しか知らない。
    "repack",
    "pipeline",
    "goldens",
    "golden_models",
    "custom_ops",
    # 段 2 で Anima の patch 層から回収したモデル非依存の export 検証ヘルパ。
    "rope",
    # states 形への IR 手術（ADR 0067）。decode 台本が使うがモデル知識は持たない。
    "states",
    # 成果物公開の原語（ADR 0052 の staging → 検証 → swap）。path しか知らない。
    "artifacts",
    # 段 5 で純化した組み立てエンジンとカード描画・CLI（family 知識も repo topology も無い）。
    "dist",
    "modelcard",
    "cli",
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
#:
#: NOTE: `karume/paths.py` も**もう無い**（ADR 0065 段 5 で `tools/export-recipes/_shared/`
#: へ出た）。`patch_*` と同じ**再発防止の名簿**としてここに残す — repo topology（リポの
#: `models/` / `outputs/` の綴り）を core wheel へ書き戻す最短の道がこの綴りで、消すと
#: 「気づいたら `karume/paths.py` が生えていた」を止める門が無くなる。恒真化していないことは
#: {@link TestTheBoundaryCheckItself} の合成故障注入が示す。
#: 逆に `dist` / `modelcard` / `cli` はここから降りた — family 知識も repo topology も無い
#: core の一部になったので、**禁止側ではなく検査される側**（{@link CORE_MODULES}）にいる。
RECIPE_MODULES = frozenset({"paths"})

#: core wheel に持ち込まない上流モデル系パッケージ。
UPSTREAM_MODEL_PACKAGES = frozenset(
    {"diffusers", "style_bert_vits2", "torchvision", "transformers"}
)

#: 検査 2 の一時除外 — **モジュール → 許可する上流パッケージ**の写像。
#:
#: MUST: モジュール単位の全免除にしない。免除の根拠は 1 パッケージぶんしか無いので、集合で
#: 外すと「免除された 2 本に `transformers` を書けば門が 1 件も報告しない」形になり、core wheel
#: へ上流モデル実装の provenance 義務が入る最短の道が開く。
#:
#: NOTE: `aten_handlers` は `torch.ops.torchvision.*`（deform_conv2d）のハンドラキーを引くために
#: torchvision を基本依存として素で import する（ADR 0055 決定 7）。`golden_models` は同 op の
#: torch 突合に `torchvision.ops.deform_conv2d` を呼ぶ。どちらも「上流モデル実装のコピー」では
#: ないが、core wheel の依存として妥当かは ADR 0065 段 6（packaging / provenance）の裁定事項。
UPSTREAM_CHECK_EXEMPT: dict[str, frozenset[str]] = {
    "aten_handlers": frozenset({"torchvision"}),
    "golden_models": frozenset({"torchvision"}),
}

#: 検査 1 の一時除外。**空** — core は 1 本残らず検査 1 に掛かる（ADR 0065 段 5 完了）。
#:
#: NOTE: 検査 2 の {@link UPSTREAM_CHECK_EXEMPT} と形が違う（あちらは「許可するパッケージ」を
#: 持つ写像）のは、検査 1 に「一部だけ許す」という単位が無いため — 禁止側は `karume.<名前>` の
#: 名簿そのもので、部分免除は「そのモジュール名の import を許す」= 全免除と同じになる。
#: 除外を足すのは「既知の違反を明示して残す」ときだけで、緑にするための逃げ道にはしない。
RECIPE_CHECK_EXEMPT: frozenset[str] = frozenset()

RECIPE_GATED_MODULES = tuple(name for name in CORE_MODULES if name not in RECIPE_CHECK_EXEMPT)


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


def upstream_imports(path: Path, allowed: frozenset[str] = frozenset()) -> list[str]:
    """上流モデル系パッケージへの import を「行:import 名」の一覧で返す（無ければ空）。

    `allowed` はそのモジュールに限って許すパッケージ（{@link UPSTREAM_CHECK_EXEMPT}）。
    既定は空なので、免除表に載っていないモジュールは全パッケージが違反になる。
    """
    gated = UPSTREAM_MODEL_PACKAGES - allowed
    return [
        f"{_label(path)}:{lineno} -> {dotted}"
        for lineno, dotted in imported_modules(path)
        if dotted.split(".")[0] in gated
    ]


class TestTheModuleRoster:
    """名簿（{@link CORE_MODULES}）が現物を 1 本残らず名指ししている。

    名簿は手書きのまま（新しいモジュールを足すときに「core なのか recipe なのか」の判断を
    素通りさせない MUST）で、ここが見るのは**一致だけ**。書き忘れた新モジュールは検査 2 の
    parametrize から丸ごと落ちるので、上流モデル系 import を黙って通す席になる。
    """

    def test_the_roster_names_every_module_on_disk(self) -> None:
        on_disk = {path.stem for path in KARUME_ROOT.glob("*.py")}
        roster = set(CORE_MODULES)

        assert on_disk == roster, (
            f"名簿に無い: {sorted(on_disk - roster)} / 名簿だけにある: {sorted(roster - on_disk)}"
        )

    def test_the_package_has_no_subpackage_outside_the_scan(self) -> None:
        """走査は `src/karume` 直下だけ — サブパッケージができた日に穴を知らせる。

        `core_module` が `KARUME_ROOT / f"{name}.py"` を返す以上、`karume/xxx/yyy.py` は
        名簿にも検査にも載らない（`__pycache__` は成果物なので除く）。
        """
        subdirectories = sorted(
            path.name
            for path in KARUME_ROOT.iterdir()
            if path.is_dir() and path.name != "__pycache__"
        )

        assert subdirectories == []


class TestThePublicSurface:
    """パッケージの公開面（`karume.__all__`）が実在し、配布形の 1 本道が辿れる。

    公開面の正本は `__all__` 1 つ（docstring の列挙はその説明）— 2 つに割れると、docstring
    だけを読んだ書き手が「書き出し → 検証 → 据え替え」の 3 段を自分で綴り直す。
    """

    def test_every_exported_name_resolves(self) -> None:
        import karume

        missing = [name for name in karume.__all__ if not hasattr(karume, name)]

        assert missing == []

    def test_the_one_way_to_publish_is_exported(self) -> None:
        """`publish_model` が公開面に居る（配布形を作る経路はこの 1 本 — pipeline の MUST）。"""
        import karume

        assert "publish_model" in karume.__all__

    def test_the_export_list_is_sorted(self) -> None:
        """未ソートを足したら赤（並びの主張ではなく、追記の位置を機械が決める検出器）。"""
        import karume

        assert list(karume.__all__) == sorted(karume.__all__)


class TestCoreDoesNotImportRecipeModules:
    @pytest.mark.parametrize("module_name", RECIPE_GATED_MODULES)
    def test_a_core_module_imports_no_recipe_module(self, module_name: str) -> None:
        """core は `patch_*` / family 知識を焼いたモジュールへ依存しない（依存は recipe → core）。

        逆流すると wheel の物理境界（`karume/` ディレクトリ全体）に family 知識が入る。
        """
        assert recipe_imports(core_module(module_name)) == []


class TestCoreDoesNotImportUpstreamModelPackages:
    @pytest.mark.parametrize("module_name", CORE_MODULES)
    def test_a_core_module_imports_no_upstream_model_package(self, module_name: str) -> None:
        """core wheel は上流モデル実装（とその provenance 義務）を引き込まない。

        免除された 2 本も**検査に掛かる**（許可するのは torchvision 1 つだけ）— 除外リストで
        丸ごと外すと、その 2 本が上流モデル実装の入口として開いたままになる。
        """
        allowed = UPSTREAM_CHECK_EXEMPT.get(module_name, frozenset())

        assert upstream_imports(core_module(module_name), allowed) == []


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

    def test_an_exemption_covers_only_the_package_it_names(self, tmp_path: Path) -> None:
        """免除は「そのモジュール × そのパッケージ」— 免除された席でも他の上流は違反。

        免除をモジュール単位の集合に戻すとここが緑のまま素通りする（= core wheel へ上流
        モデル実装を持ち込む最短の道が開く）。
        """
        path = self._violator(
            tmp_path,
            "exempted",
            "import torchvision\n\nfrom transformers import AutoModel\n",
        )

        assert upstream_imports(path, frozenset({"torchvision"})) == [
            "karume/exempted.py:3 -> transformers",
            "karume/exempted.py:3 -> transformers.AutoModel",
        ]

    def test_it_stays_quiet_on_a_module_that_imports_nothing_forbidden(
        self, tmp_path: Path
    ) -> None:
        """検出側が「何にでも反応する」のではないことの対（上の主張の裏側）。"""
        path = self._violator(tmp_path, "clean", "import json\n\nfrom karume.ir import IrGraph\n")

        assert upstream_imports(path) == []
        assert recipe_imports(path) == []

    def test_it_catches_a_lazy_recipe_import(self, tmp_path: Path) -> None:
        """`from karume import paths` の形も見る（シンボル名かサブモジュール名かは区別しない）。"""
        path = self._violator(
            tmp_path, "leaky", "def run(argv):\n    from karume import paths\n\n    return paths\n"
        )

        assert recipe_imports(path) == ["karume/leaky.py:2 -> karume.paths"]

    def test_it_catches_a_relative_recipe_import(self, tmp_path: Path) -> None:
        """相対 1 段も絶対化して見る（`from . import paths` で門をすり抜けさせない）。"""
        path = self._violator(tmp_path, "relative", "from . import paths\n")

        assert recipe_imports(path) == ["karume/relative.py:1 -> karume.paths"]

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
