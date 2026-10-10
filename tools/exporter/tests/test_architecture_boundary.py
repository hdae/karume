"""core → recipe の逆流を機械で止める門（ADR 0065 決定 3）。

exporter は「汎用 exporter core（PyPI 配布物）」と「モデル別 recipe（リポ専用）」へ段階的に
分離する途中で、依存方向は **recipe → core の一方向だけ**が許される。規約として書くだけでは
段が進むたびに逆流が忍び込むので、ここで import を機械検査する（同 ADR: 境界は規約でなく
テストが守る）。

検査は 2 本:

1. core が **recipe 側モジュール**を import しない — recipe の現行の名前空間（`wan` /
   `_shared` などのトップレベルパッケージ、`dist` / `migrate_series` などの直下のモジュール、
   `container_series` / `gemma_synthetic` などの `tests/` 直下の共有テストヘルパ。
   `tools/export-recipes/` の配置から導出）と、旧名の `karume.patch_*` / `karume.paths`
   （どちらも今は 1 本も無いが再発防止で残す門）の両方
2. core が **上流モデル系パッケージ**（`style_bert_vits2` / `transformers` / `diffusers` /
   `torchvision`）を import しない — core wheel に上流実装の provenance 義務を引き込まない
   ための門（ADR 0065 決定 7）

走査は文字列 grep でなく `ast` — 関数内の遅延 import（`karume.cli` の
`from karume import dist` のような形）が段階移行の逃げ道にならないようにする。
`ast` で見るのは静的な import 文だけ（`importlib.import_module` や `__import__` は対象外）。
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
    # quant の `requiredLimits` 導出（ADR 0038 §7）。寸法と WebGPU の既定値しか知らない。
    "limits",
    "verify",
    # コンテナ形式（krm / krg）の書き手と自己検査用の読み手（ADR 0108）。
    # バイト列・宣言・codec 台帳しか知らない。
    "container",
    # 配布形を据える 3 段（書く → 読み直して検証 → 据え替え）。path とバイト列しか知らない。
    "publish",
    # PLE を容器の資産へ組む純関数（ADR 0085 / 0109 決定 4）。行バイト数と block 上限だけ。
    "ple",
    # 旧配布形（safetensors 方言）の**読み取り専用**の層（container-v1 §12）。移行 CLI 専用で、
    # 知っているのはバイト列と宣言だけ。
    "legacy",
    # 旧配布形 → コンテナ形式の移行 CLI（container-v1 §12）。旧形式を読むのはここだけで、
    # 知っているのは宣言と生バイト（family 知識も repo topology も持たない）。
    "migrate",
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

#: recipe 側の置き場（ADR 0065 決定 2 — wheel の外・リポ専用）。
RECIPES_ROOT = Path(__file__).resolve().parents[2] / "export-recipes"

#: recipe 側の共有テストヘルパの置き場（置き場の直下からの相対）。recipe の conftest が
#: このディレクトリを sys.path に張るので、中身は `container_series` のようなトップレベル名で
#: import される。
RECIPE_TEST_HELPERS_DIR = "tests"

#: 置き場の直下と {@link RECIPE_TEST_HELPERS_DIR} の直下にあるが、recipe 側の名前空間では
#: ないもの — テストの置き場そのもの（`tests` — sys.path に張られるのは中身で、`tests` という
#: 名前では import されない）と pytest の設定（`conftest.py` — pytest が読むもので、名前で
#: import されない）。
NON_RECIPE_NAMES = frozenset({"conftest", "tests"})

#: pytest が収集するテストモジュールの接頭辞。収集されるもので、他から名前で import されない。
TEST_MODULE_PREFIX = "test_"


def recipe_packages(root: Path) -> frozenset[str]:
    """recipe の置き場の直下から、recipe のトップレベル名（パッケージとモジュール）を導出する。

    recipe は `karume.*` でなく `wan` / `_shared` のようなトップレベルパッケージとして
    import される（`python -m <family>.<mod>`）ので、旧名の名簿だけでは今の綴りの逆流を
    止められない。名簿を手書きにすると family を足すたびに更新漏れの穴が開くので、配置から
    導出する。拾うのは識別子として import できる名前のうち次の 3 種:

    - 直下に `.py` を持つディレクトリ — `__init__.py` を要件にしないのは、namespace package
      として置かれた family も import は通るため
    - 直下の `.py` モジュール（`dist` / `migrate_series` など）— 起動スクリプトだが、recipe 側の
      テストが `import dist` / `from dist import main` とトップレベル名で import している。
      同じ綴りを core が書けば recipe への依存になる
    - `tests/` 直下の `.py` モジュールのうちテストモジュール（`test_*`）でないもの
      （`container_series` / `gemma_synthetic` など）— recipe 側のテストが
      `from container_series import write_component` とトップレベル名で import している。
      理由は直下のモジュールと同じ

    MUST: 置き場が無い環境では fail loudly にする。黙って空集合を返すと、この枝の検査が
    恒真のまま緑になる（境界はテストが守る — ADR 0065 決定 3）。`tests/` が無いことは
    許す — 共有ヘルパが 1 本も無ければ、それを import する綴りも無い。
    """
    if not root.is_dir():
        raise FileNotFoundError(f"recipe の置き場が無い: {root}（リポの外では検査 1 を回せない）")
    packages = (path.name for path in root.iterdir() if path.is_dir() and any(path.glob("*.py")))
    modules = (path.stem for path in root.glob("*.py"))
    test_helpers = (
        path.stem
        for path in (root / RECIPE_TEST_HELPERS_DIR).glob("*.py")
        if not path.stem.startswith(TEST_MODULE_PREFIX)
    )
    return frozenset(
        name
        for name in (*packages, *modules, *test_helpers)
        if name.isidentifier() and name not in NON_RECIPE_NAMES
    )


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
    """recipe 側モジュールへの import を「行:import 名」の一覧で返す（無ければ空）。

    禁止するトップレベル名はここで初めて配置から導出する — モジュールの読み込み時に導出すると、
    置き場が無い環境で検査 1 と無関係な検査（上流パッケージ・移行経路）まで collection error で
    道連れになる。fail loudly の範囲を検査 1 に閉じる。
    """
    recipe_names = recipe_packages(RECIPES_ROOT)
    violations = []
    for lineno, dotted in imported_modules(path):
        parts = dotted.split(".")
        # 判定はトップレベル名（`parts[0]`）で行う — recipe 側の `dist.py` と同名の
        # `karume.dist` を誤検出しない。
        if parts[0] in recipe_names:
            violations.append(f"{_label(path)}:{lineno} -> {dotted}")
            continue
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

    def test_it_catches_a_lazy_import_of_a_recipe_family(self, tmp_path: Path) -> None:
        """recipe の現行の名前空間（トップレベルの family パッケージ）への遅延 import も見る。

        関数内の import は core 単体の pytest では ImportError にならず、wheel の利用者が
        その関数を呼んだ時点で初めて壊れる — ここで止めないと境界違反が配布物まで届く。
        """
        path = self._violator(tmp_path, "familiar", "def run():\n    from wan import umt5_patch\n")

        assert recipe_imports(path) == [
            "karume/familiar.py:2 -> wan",
            "karume/familiar.py:2 -> wan.umt5_patch",
        ]

    def test_it_catches_a_lazy_import_of_the_shared_recipe_layer(self, tmp_path: Path) -> None:
        """`_shared`（repo topology を持つ recipe 共有層）への遅延 import も違反。"""
        path = self._violator(
            tmp_path, "topology", "def run():\n    from _shared.paths import REPO_ROOT\n"
        )

        assert recipe_imports(path) == [
            "karume/topology.py:2 -> _shared.paths",
            "karume/topology.py:2 -> _shared.paths.REPO_ROOT",
        ]

    def test_it_catches_a_lazy_import_of_a_top_level_recipe_module(self, tmp_path: Path) -> None:
        """直下のスクリプト（`dist.py`）をトップレベル名で import する形も違反。"""
        path = self._violator(tmp_path, "launcher", "def run():\n    import dist\n")

        assert recipe_imports(path) == ["karume/launcher.py:2 -> dist"]

    def test_it_catches_a_lazy_import_of_a_shared_recipe_test_helper(self, tmp_path: Path) -> None:
        """recipe 側の共有テストヘルパ（`tests/container_series.py`）をトップレベル名で
        import する形も違反。"""
        path = self._violator(
            tmp_path, "helper", "def run():\n    from container_series import write_component\n"
        )

        assert recipe_imports(path) == [
            "karume/helper.py:2 -> container_series",
            "karume/helper.py:2 -> container_series.write_component",
        ]

    def test_it_stays_quiet_on_a_core_module_sharing_a_recipe_script_name(
        self, tmp_path: Path
    ) -> None:
        """判定はトップレベル名で行う — recipe 側の `dist.py` と同名の `karume.dist` は core。"""
        path = self._violator(tmp_path, "assembler", "from karume.dist import PIPELINES\n")

        assert recipe_imports(path) == []


class TestTheRecipeNamespaceIsDerivedFromDisk:
    """禁止するトップレベル名を `tools/export-recipes/` 直下の配置から導出している。

    名簿を手書きにすると family を足すたびに更新漏れの穴が開く。導出が配置を読み損ねて
    空になると、検査 1 の新しい枝が黙って恒真化する。
    """

    def test_it_includes_the_recipe_families_and_the_shared_layer(self) -> None:
        assert {"wan", "_shared"} <= recipe_packages(RECIPES_ROOT)

    def test_it_includes_the_top_level_recipe_modules(self) -> None:
        """直下のスクリプトも禁止側 — recipe 側のテストが `import dist` の綴りで使っている。"""
        assert {"dist", "migrate_series"} <= recipe_packages(RECIPES_ROOT)

    def test_it_includes_the_shared_recipe_test_helpers(self) -> None:
        """`tests/` 直下の共有ヘルパも禁止側 — recipe 側のテストが
        `from container_series import write_component` の綴りで使っている。"""
        assert {"container_series", "gemma_synthetic"} <= recipe_packages(RECIPES_ROOT)

    def test_it_excludes_the_entries_that_are_not_recipe_packages(self) -> None:
        """意図して除外した名前（recipe 側のテストと pytest の設定）は禁止集合に入らない。"""
        assert recipe_packages(RECIPES_ROOT).isdisjoint({"tests", "conftest"})

    def test_it_takes_only_the_helpers_from_the_test_directory(self, tmp_path: Path) -> None:
        """`tests/` 直下からは共有ヘルパだけを拾い、テストモジュールと pytest の設定は拾わない。

        実物の `tests/` には今 `conftest.py` が無いので、除外の両方を合成した置き場で踏む。
        """
        helpers = tmp_path / "export-recipes" / "tests"
        helpers.mkdir(parents=True)
        for name in ("shared_helper", "test_something", "conftest"):
            (helpers / f"{name}.py").write_text("")

        assert recipe_packages(tmp_path / "export-recipes") == {"shared_helper"}

    def test_it_fails_loudly_without_the_recipe_root(self, tmp_path: Path) -> None:
        """置き場が無いときに空集合を返すと、検査 1 の新しい枝が黙って恒真化する。"""
        with pytest.raises(FileNotFoundError):
            recipe_packages(tmp_path / "export-recipes")


#: 旧配布形を読むだけの経路（移行 CLI）が到達してよい core モジュールの起点。
MIGRATION_ENTRY = "migrate"

#: 移行 CLI の import グラフへ引き込まない重い依存。
#:
#: MUST: `torch` を入れない。`karume migrate` が読むのは旧 shard のバイト列と宣言だけで、
#: 格納変換（torch 依存）は 1 度も通らない — 束縛表の写像（`container_bindings`）を torch 側の
#: モジュールへ置くと、移行の import グラフが 1 GB 級の依存を丸ごと引く（`dist` が
#: `karume.verify` の import を関数内へ遅延していたのと同じ理由）。
MIGRATION_FORBIDDEN = frozenset({"torch"})


def karume_closure(entry: str) -> dict[str, list[str]]:
    """`karume.<entry>` から辿れる core モジュール → その import 先（完全修飾名）。

    走査は検査 1 / 2 と**同じ {@link imported_modules}**（`ast` なので関数内の遅延 import も
    拾う）。`karume` 直下のモジュールだけを辿り、外部パッケージは葉として記録する。
    """
    reached: dict[str, list[str]] = {}
    pending = [entry]
    while pending:
        name = pending.pop()
        if name in reached:
            continue
        dotted = [item for _, item in imported_modules(core_module(name))]
        reached[name] = dotted
        for item in dotted:
            parts = item.split(".")
            if parts[0] == "karume" and len(parts) >= 2 and parts[1] in CORE_MODULES:
                pending.append(parts[1])
    return reached


class TestTheMigrationPathStaysIndependentOfTorch:
    """移行 CLI の import グラフに torch が現れない（F-12 の再発防止）。

    NOTE: 実行時の門（`import karume.migrate` の後の `sys.modules`）は
    `tests/test_package_init.py` の `TORCH_FREE_MODULES` が持つ — パッケージの `__init__` が
    公開面を初回参照で解決する（PEP 562）ので、サブモジュールの import だけでは torch が
    載らない。こちらは**モジュールの依存方向**を `ast` で見る（関数内の遅延 import も拾う）。
    2 本は互いの穴を埋める: `ast` は動的 import と外部パッケージ経由の torch を見ず、実行時の
    門は呼ばれていない関数内の import を見ない。
    """

    def test_no_module_reachable_from_migrate_imports_torch(self) -> None:
        closure = karume_closure(MIGRATION_ENTRY)
        offenders = sorted(
            f"karume/{name}.py -> {item}"
            for name, imports in closure.items()
            for item in imports
            if item.split(".")[0] in MIGRATION_FORBIDDEN
        )

        assert offenders == []

    def test_the_closure_really_walks_the_graph(self) -> None:
        """恒真化の門 — 起点だけでなく、そこから辿った先まで名簿に載っている。"""
        closure = karume_closure(MIGRATION_ENTRY)

        assert MIGRATION_ENTRY in closure
        # 移行は容器・旧形の読み手・公開の 3 段・合流を必ず通る。
        assert {"container", "legacy", "publish", "verify", "ple", "dist"} <= set(closure)

    def test_the_closure_catches_a_module_that_reaches_torch(self) -> None:
        """検出側が本当に torch を見つける（`emit` を起点にすれば必ず出る）。"""
        closure = karume_closure("emit")
        offenders = [
            item
            for imports in closure.values()
            for item in imports
            if item.split(".")[0] in MIGRATION_FORBIDDEN
        ]

        assert offenders
