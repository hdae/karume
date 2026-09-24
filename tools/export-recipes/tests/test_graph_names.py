"""容器のグラフ名 = 配布形の部品名（= `karume.json` の weights のキー）の機械門（全 family 横断）。

ランタイムはグラフを**名前で**引く（`prepareContainer(opened, <weights キー>)` —
container-v1 §2.1）。移行 CLI も同じキーで焼く（`karume.migrate._convert_unit` が
`graph_name=unit.component`）ので、書き手が別の綴りを名乗ると「移行済みミラー」と「再 export
した系列」が**別物**になる。

守るのは 2 つ:

1. `graph_name=` に渡すのは**部品名の定数**であって、置き場の path から導いた名前ではない。
   ディレクトリ名は部品名と一致しない: 系列直下に容器を置く family（siglip2 / birefnet /
   depth_anything / vowel_detector / gemma4 / minicpm5 / embeddinggemma / gemma4_qat）では
   ディレクトリ名が**系列名**（`siglip2-so400m-patch14-384`）になり、irodori は
   `caption-proj` / `decoder` に対してキーが `caption_proj` / `codec_decoder`、deberta は
   `full-24layer` に対してキーが `text_encoder`（SBV2 の消費側の席）である。
2. 台本が名乗る綴りの集合が、その family の配布計画（`<family>/distribution.py` の weights）
   と一致する。

この門が要るのは、書き手（export の一本道）がグラフ名を weights のキーと突き合わせないから。
現物の容器との突合は `karume dist`（{@link karume.dist.assert_weight_components_verified}）が
組み立ての前に掛けるが、そこへ届くのは実重みで export を回した後である — ここは綴りの割れを
ソースの段で落とす。

ソースを AST で読むのは、実行して確かめるには実重み（数 GB）か family ごとの tiny 模型が
要るため（`tests/test_staged_publication.py` と同じ理由）。実物を書いて読み直す側の対は
`vowel_detector/tests/test_export.py` の {@link
vowel_detector.tests.test_export.TestGraphNameIsThePartName} が 1 本持つ。
"""

from __future__ import annotations

import ast
import importlib
from collections.abc import Mapping
from pathlib import Path

import pytest

RECIPES_ROOT = Path(__file__).resolve().parent.parent


#: (台本, emit 関数, `graph_name=` の式, その式が取りうる値を持つモジュール定数)。
#:
#: 式と値の欄が別なのは、ループ変数越しに渡す台本があるため（anima は `target`、irodori は
#: `IRODORI_SERIES_ROLES[target]`）— 値の欄には**モジュール直下の定数名**だけを書き、
#: {@link _declared_values} が `str` / 並び / 表のどれでも綴りの集合へ畳む。
#:
#: MUST: 実在 → 表 の側（{@link TestEveryCallSiteIsListed}）と両建て。
ENTRIES: tuple[tuple[str, str, str, str], ...] = (
    ("anima/export.py", "emit_target", "target", "TARGETS"),
    ("deberta/export.py", "export_variant", "GRAPH_NAME", "GRAPH_NAME"),
    ("embeddinggemma/export.py", "export_series", "GRAPH_NAME", "GRAPH_NAME"),
    ("gemma4/export.py", "export_series", "GEMMA4_ROLE", "GEMMA4_ROLE"),
    ("gemma4/export_decode.py", "export_series", "GEMMA4_ROLE", "GEMMA4_ROLE"),
    ("gemma4/export_drafter.py", "export_series", "GEMMA4_DRAFTER_ROLE", "GEMMA4_DRAFTER_ROLE"),
    ("gemma4/export_product.py", "export_series", "GEMMA4_ROLE", "GEMMA4_ROLE"),
    ("gemma4_qat/export.py", "export_qat", "GEMMA4_ROLE", "GEMMA4_ROLE"),
    (
        "irodori/dacvae/export.py",
        "export_series",
        "IRODORI_CODEC_ROLES[target]",
        "IRODORI_CODEC_ROLES",
    ),
    ("irodori/export.py", "export_series", "IRODORI_SERIES_ROLES[target]", "IRODORI_SERIES_ROLES"),
    ("minicpm5/export.py", "export_series", "GRAPH_NAME", "GRAPH_NAME"),
    ("minicpm5/export_decode.py", "export_series", "one_shot.GRAPH_NAME", "GRAPH_NAME"),
    ("sbv2/export.py", "export_dec", "TARGET_DEC", "TARGET_DEC"),
    ("sbv2/export.py", "export_dp", "TARGET_DP", "TARGET_DP"),
    ("sbv2/export.py", "export_flow", "TARGET_FLOW", "TARGET_FLOW"),
    ("sbv2/export.py", "export_front", "TARGET_FRONT", "TARGET_FRONT"),
    ("sbv2/export.py", "export_voice", "TARGET_VOICE", "TARGET_VOICE"),
    ("siglip2/export.py", "export_series", "SIGLIP2_ROLE", "SIGLIP2_ROLE"),
    ("birefnet/export.py", "export_series", "BIREFNET_ROLE", "BIREFNET_ROLE"),
    ("depth_anything/export.py", "export_series", "DEPTH_ANYTHING_ROLE", "DEPTH_ANYTHING_ROLE"),
    (
        "vowel_detector/export.py",
        "export_series",
        "VOWEL_DETECTOR_GRAPH_ROLE",
        "VOWEL_DETECTOR_GRAPH_ROLE",
    ),
)

#: `minicpm5/export_decode.py` のように、値の定数が**別モジュール**に在る台本の引き先。
VALUE_MODULES: Mapping[str, str] = {"minicpm5/export_decode.py": "minicpm5.export"}


def _module_name(script: str) -> str:
    """台本の相対 path → import 名（`irodori/dacvae/export.py` → `irodori.dacvae.export`）。"""
    return script.removesuffix(".py").replace("/", ".")


def _function(script: str, name: str) -> ast.FunctionDef:
    source = (RECIPES_ROOT / script).read_text(encoding="utf-8")
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{script}: {name} がモジュール直下に無い（台本の綴りが動いた）")


def _graph_name_arguments(node: ast.AST) -> list[ast.expr]:
    """その木の中で `graph_name=` に渡している式を全部。"""
    return [
        keyword.value
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        for keyword in call.keywords
        if keyword.arg == "graph_name"
    ]


def _derived_from_a_path(expression: ast.expr) -> bool:
    """式が `<なにか>.name`（= 置き場の path から導いた名前）を含むか。

    見るのは属性の綴りだけ（`Path.name` も `Path.parent.name` も同じ形）— `one_shot.GRAPH_NAME`
    のような module 越しの定数参照は属性名が違うので当たらない。
    """
    return any(
        isinstance(node, ast.Attribute) and node.attr == "name" for node in ast.walk(expression)
    )


def _declared_values(script: str, constant: str) -> set[str]:
    """台本が `graph_name=` に渡しうる綴りの集合（定数を実体から引く）。"""
    module = importlib.import_module(VALUE_MODULES.get(script, _module_name(script)))
    value = getattr(module, constant)
    if isinstance(value, str):
        return {value}
    if isinstance(value, Mapping):
        return set(value.values())
    return set(value)


def _scanned_call_sites() -> set[tuple[str, str]]:
    """`graph_name=` を渡している関数を、recipes の**全 Python ソース**から列挙する（実在 → 表）。

    MUST: 走査は `rglob("*.py")` + {@link ast.walk} — 名前（`export*.py`）でも置き場
    （`<family>/…`）でもモジュール直下でも絞らない。絞ると「門が名乗る範囲」が実装より広く
    なり、リポ直下の書き手（`migrate_series.py`）やクラスメソッドの呼び出し口が**表に載らない
    まま緑**で通る（実測で 1 本取りこぼしていた）。
    """
    found: set[tuple[str, str]] = set()
    for path in sorted(RECIPES_ROOT.rglob("*.py")):
        if "tests" in path.parts or "__pycache__" in path.parts or ".venv" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and _graph_name_arguments(
                node
            ):
                found.add((str(path.relative_to(RECIPES_ROOT)), node.name))
    return found


#: 表に載せない呼び手（`graph_name` を**受け取って素通しする**内部ヘルパ）。名乗るのは呼び出し
#: 元の emit 関数なので、ここで綴りを決めているわけではない。
FORWARDERS: frozenset[tuple[str, str]] = frozenset(
    {
        ("gemma4/export_decode.py", "_write_container"),
        ("minicpm5/export_decode.py", "_write_container"),
    }
)

#: 表に載せない呼び手（綴りを**family の表から引く**移行ドライバ）。{@link ENTRIES} は
#: 「台本が定数を名乗る」形しか綴れないが、このドライバは系列ディレクトリ → 部品名の写像を
#: 走らせる側なので、綴りの正しさは `tests/test_migrate_series.py` の family 表の門が見る。
TABLE_DRIVEN: frozenset[tuple[str, str]] = frozenset({("migrate_series.py", "migrate_series")})


@pytest.mark.parametrize(
    ("script", "function", "expression", "constant"), ENTRIES, ids=lambda value: str(value)
)
class TestTheEmitScriptNamesThePartName:
    def test_it_passes_the_declared_constant(
        self, script: str, function: str, expression: str, constant: str
    ) -> None:
        """`graph_name=` の式が表の綴りと逐語で一致すること（式が動いたらここで落ちる）。"""
        passed = {
            ast.unparse(argument) for argument in _graph_name_arguments(_function(script, function))
        }

        assert passed == {expression}, (
            f"{script}:{function} が graph_name へ渡しているのは {sorted(passed)} —"
            f" 表の綴りは {expression}"
        )

    def test_it_does_not_derive_the_name_from_a_directory(
        self, script: str, function: str, expression: str, constant: str
    ) -> None:
        """置き場の path から導いた名前は部品名ではない（この門の存在理由）。"""
        derived = [
            ast.unparse(argument)
            for argument in _graph_name_arguments(_function(script, function))
            if _derived_from_a_path(argument)
        ]

        assert derived == [], (
            f"{script}:{function} が graph_name を置き場の名前 {derived} から導いている —"
            " ディレクトリ名は部品名（karume.json の weights のキー）と一致しない"
        )


class TestTheNamesAreTheWeightsKeys:
    """台本が名乗る綴りの集合 ↔ 配布計画の weights のキー。

    どちらの向きも見る: キーに無い綴りを焼くと**ランタイムが引けない容器**になり、キーが
    余ると**その部品だけ焼き手が居ない**（= 配布形が組めない）。
    """

    @staticmethod
    def _named(family: str) -> set[str]:
        names: set[str] = set()
        for script, function, _expression, constant in ENTRIES:
            if script.split("/")[0] == family:
                assert _graph_name_arguments(_function(script, function))
                names |= _declared_values(script, constant)
        assert names, f"{family}: 表に台本が 1 本も無い"
        return names

    @staticmethod
    def _weights(family: str) -> set[str]:
        distribution = importlib.import_module(f"{family}.distribution")
        table = getattr(distribution, f"{family.upper()}_WEIGHTS")
        return set(table)

    @pytest.mark.parametrize(
        "family", ["anima", "irodori", "gemma4", "siglip2", "birefnet", "depth_anything"]
    )
    def test_the_scripts_cover_exactly_the_weights_keys(self, family: str) -> None:
        assert self._named(family) == self._weights(family)

    def test_vowel_detector_covers_exactly_its_weights_keys(self) -> None:
        """`VOWEL_DETECTOR_WEIGHTS` は family 名から導く綴りにならないので名指しで見る。"""
        from vowel_detector.distribution import VOWEL_DETECTOR_WEIGHTS

        assert self._named("vowel_detector") == set(VOWEL_DETECTOR_WEIGHTS)

    def test_sbv2_covers_the_distributed_seats_and_nothing_else(self) -> None:
        """SBV2 は `dp` / `flow` / `dec` が golden 検証専用（配布形に載らない）。

        `text_encoder` を焼くのは別台本（`deberta/export.py`）なので、SBV2 側の台本が
        名乗るのは残りの 2 席だけ。
        """
        from sbv2.distribution import SBV2_FRONT_ROLE, SBV2_VOICE_ROLE, SBV2_WEIGHTS

        named = self._named("sbv2")

        assert named & set(SBV2_WEIGHTS) == {SBV2_FRONT_ROLE, SBV2_VOICE_ROLE}
        assert named - set(SBV2_WEIGHTS) == {"dp", "flow", "dec"}

    def test_deberta_names_the_seat_its_consumer_declares(self) -> None:
        """produce 側（deberta）は consume 側（SBV2）を import しないので、写しを突き合わせる。"""
        from sbv2.distribution import SBV2_TEXT_ENCODER_COMPONENT, SBV2_WEIGHTS

        assert self._named("deberta") == {SBV2_TEXT_ENCODER_COMPONENT}
        assert SBV2_TEXT_ENCODER_COMPONENT in SBV2_WEIGHTS

    def test_gemma4_qat_names_the_seat_its_own_plan_declares(self) -> None:
        """QAT は別 family の配布計画を持つが、部品名は通常 Gemma と同じ 1 語を共有する。"""
        from gemma4.distribution import GEMMA4_ROLE

        assert self._named("gemma4_qat") == {GEMMA4_ROLE}

    @pytest.mark.parametrize("family", ["embeddinggemma", "minicpm5"])
    def test_the_solo_families_name_the_single_part_convention(self, family: str) -> None:
        """配布形を組まない 2 家族（`distribution.py` を持たない）の綴りを固定する。

        組む日が来たときに据わっている容器がそのまま使えるよう、部品 1 つの系列の慣例
        （`model` — gemma4 / gemma4_qat と同じ）で名乗っておく。
        """
        assert self._named(family) == {"model"}


class TestEveryCallSiteIsListed:
    """実在 → 表 の逆方向（表 → 実在 だけだと、載せ忘れが永久に沈黙する）。"""

    def test_the_scan_finds_the_call_sites(self) -> None:
        """走査が 0 本なら、この門は恒真になる。"""
        assert len(_scanned_call_sites()) >= len(ENTRIES)

    def test_no_call_site_is_missing_from_the_table(self) -> None:
        listed = {(script, function) for script, function, _expression, _constant in ENTRIES}
        missing = sorted(_scanned_call_sites() - listed - FORWARDERS - TABLE_DRIVEN)

        assert missing == [], (
            f"graph_name を渡しているのに綴りの門が掛かっていない台本がある: {missing} —"
            " ENTRIES へ足す（表に無い台本は置き場の名前を名乗る形に戻せる）"
        )

    def test_the_scan_reaches_a_writer_that_is_not_an_export_script(self) -> None:
        """走査が `export*.py` / `<family>/…` に絞られていないこと（F-4 の取りこぼし）。

        移行ドライバはリポ直下の `migrate_series.py` に在り、名前も置き場も台本の形から
        外れている。ここが落ちる形に戻すと、表に載らない書き手が永久に沈黙する。
        """
        assert _scanned_call_sites() >= TABLE_DRIVEN

    def test_every_table_driven_writer_really_reads_a_table(self) -> None:
        """表引きの口も**実在**と形を見る（消えた綴りが除外表に残ると載せ忘れが隠れる）。"""
        for script, function in sorted(TABLE_DRIVEN):
            arguments = _graph_name_arguments(_function(script, function))
            assert arguments, f"{script}:{function} が graph_name を渡していない"
            assert not any(_derived_from_a_path(argument) for argument in arguments), (
                f"{script}:{function} が graph_name を置き場の名前から導いている"
            )

    def test_every_listed_script_exists(self) -> None:
        for script, _function, _expression, _constant in ENTRIES:
            assert (RECIPES_ROOT / script).is_file(), script

    def test_every_forwarder_really_forwards(self) -> None:
        """素通しの口も**実在**を見る（消えた綴りが除外表に残ると、載せ忘れが隠れる）。"""
        for script, function in sorted(FORWARDERS):
            node = _function(script, function)
            assert [ast.unparse(a) for a in _graph_name_arguments(node)] == ["graph_name"]
            assert any(argument.arg == "graph_name" for argument in node.args.kwonlyargs)


#: リポジトリの根（`RECIPES_ROOT` は `tools/export-recipes/`）。
REPO_ROOT = RECIPES_ROOT.parent.parent

#: 仕様の正本（`docs/container-v1.md`）。
DOCS_ROOT = REPO_ROOT / "docs"

#: 「グラフ名 = weights のキー」の規則の本文を持つ仕様節（見出し行そのもの）。
CONTAINER_SPEC_SECTION = "### 2.1 グラフ記述"

#: 移行 CLI の契約。グラフ名の既定（親ディレクトリ名）を持ち、規則は §2.1 を参照する。
MIGRATE_SPEC_SECTION = "## 12. 移行 CLI の契約"

#: グラフ名の話の指し先として**使ってはいけない**綴り。
#:
#: 決定 3 は `container` 欄（descriptor の期待値 + part の FileRef 列）、決定 8 は移行 CLI の
#: リポ丸ごとモードの話で、どちらも**グラフ名を述べていない**。片方だけ縛ると、同じ種類の
#: 誤指しがもう片方の綴りで復活する（実際に決定 3 を潰した後、決定 8 で 6 箇所に再発した）。
WRONG_POINTERS = ("ADR 0109 決定 3", "ADR 0109 決定 8")

#: 誤指しを走査する根（`docs/` は ADR 自身が住む場所なので外す）。
POINTER_SCAN_ROOTS = ("packages", "tools", "examples")

#: 走査する綴り（読み手の TS と書き手の Python の両方 — 誤指しは両側で起きる）。
POINTER_SCAN_SUFFIXES = (".py", ".ts")

#: 走査から外すディレクトリ名（生成物・依存の取り込み先）。
POINTER_SCAN_SKIP = frozenset({"__pycache__", ".venv", "node_modules"})


def _spec_section(document: Path, heading: str) -> str:
    """見出し行 `heading`（`## …` / `### …`）から、同じか浅い次の見出しまでを返す。

    見出しが無ければ落ちる。
    """
    lines = document.read_text(encoding="utf-8").split("\n")
    start = next(index for index, line in enumerate(lines) if line == heading)
    level = len(heading) - len(heading.lstrip("#"))
    rest = lines[start + 1 :]
    stop = next(
        (
            index
            for index, line in enumerate(rest)
            if line.startswith("#") and 0 < len(line) - len(line.lstrip("#")) <= level
        ),
        len(rest),
    )
    return "\n".join(rest[:stop])


def _scanned_sources() -> list[Path]:
    """誤指しを走査するソース一覧（この門自身は除く）。"""
    return sorted(
        path
        for root in POINTER_SCAN_ROOTS
        for suffix in POINTER_SCAN_SUFFIXES
        for path in (REPO_ROOT / root).rglob(f"*{suffix}")
        if POINTER_SCAN_SKIP.isdisjoint(path.parts) and path != Path(__file__).resolve()
    )


class TestThePointerNamesTheDocumentThatCarriesTheClaim:
    """「グラフ名 = weights のキー」の指し先が、その主張を実際に持つ文書であること。

    指し先が外れていても**どこも赤くならない**（読み手が ADR を開いて食い違いに当たるだけ）
    ので、機械で縛る。実際に 25 箇所以上が `ADR 0109 決定 3` を指していたが、あの決定が
    述べているのは `container` 欄（descriptor の期待値 + part の FileRef 列）であって
    グラフ名ではない。
    """

    def test_the_container_spec_section_states_the_rule(self) -> None:
        """`container-v1 §2.1` が「容器のグラフ名 = `weights` のキー MUST」を持つ。"""
        section = _spec_section(DOCS_ROOT / "container-v1.md", CONTAINER_SPEC_SECTION)

        assert "**グラフ名の規則**" in section
        assert (
            "容器のグラフ名 = 配布形の部品名 = `karume.json` の `weights` のキー** MUST" in section
        )

    def test_the_migrate_spec_section_defers_to_the_rule(self) -> None:
        """`container-v1 §12` は CLI の既定（親ディレクトリ名）を持ち、規則は §2.1 を指す。"""
        section = _spec_section(DOCS_ROOT / "container-v1.md", MIGRATE_SPEC_SECTION)

        assert "グラフ名の既定は親ディレクトリ名" in section
        assert "規則は §2.1 のとおり weights のキー MUST" in section

    def test_the_old_pointers_do_not_state_the_rule(self) -> None:
        """対（非恒真）: 誤りの指し先 2 つが住む節には「グラフ名」という語が 1 度も出ない。"""
        decision = _spec_section(
            DOCS_ROOT / "decisions" / "0109-manifest-v5-container.md", "## Decision"
        )

        assert "### 3. `container` 欄" in decision
        assert "### 8. 段 2 の書き手は移行 CLI の**リポ丸ごとモード**" in decision
        assert "グラフ名" not in decision

    def test_no_source_cites_an_old_pointer_for_the_naming_rule(self) -> None:
        """グラフ名 / 部品名を綴る行が {@link WRONG_POINTERS} を指していないこと。

        走査は読み手（TS）と書き手（Python）の両方 — 誤指しは recipe だけでなく runtime /
        models / hub のテストや tools 側の注釈でも起きる（実際そうなった）。

        NOTE: この門自身のソースは走査から外す — 除外しないと、上の 2 本が綴る「なぜ
        その指し先が誤りか」の説明文が自分の網に掛かる（門は他人の綴りを見る道具）。
        """
        offenders = [
            f"{path.relative_to(REPO_ROOT)}:{number} ({pointer})"
            for path in _scanned_sources()
            for number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), start=1)
            for pointer in WRONG_POINTERS
            if pointer in line and ("グラフ名" in line or "部品名" in line)
        ]

        assert offenders == [], (
            f"グラフ名 / 部品名の指し先が {' / '.join(WRONG_POINTERS)} になっている:"
            f" {offenders} — 正本は container-v1"
            f" §{CONTAINER_SPEC_SECTION.split()[1]}"
        )

    def test_the_scan_really_reaches_both_languages(self) -> None:
        """対（走査が空・片言語だけに縮退していないこと）。"""
        scanned = _scanned_sources()
        suffixes = {path.suffix for path in scanned}
        roots = {path.relative_to(REPO_ROOT).parts[0] for path in scanned}

        assert suffixes == set(POINTER_SCAN_SUFFIXES), suffixes
        assert roots == set(POINTER_SCAN_ROOTS), roots
        assert Path(__file__).resolve() not in scanned

    def test_the_deberta_note_points_at_the_gate_that_really_runs(self) -> None:
        """deberta の写しが名指しする突合の門が**実在**すること（F-2 の取り違え）。"""
        source = (RECIPES_ROOT / "deberta" / "export.py").read_text(encoding="utf-8")
        gate = TestTheNamesAreTheWeightsKeys.test_deberta_names_the_seat_its_consumer_declares

        assert f"tests/{Path(__file__).name}::{TestTheNamesAreTheWeightsKeys.__name__}" in source
        assert gate.__name__ in source


class TestTheGateItselfCanFail:
    """門が「何も見ずに緑」へ退化していないこと（述語が恒真・走査が空の 2 つを潰す）。"""

    def test_it_names_a_graph_name_taken_from_a_directory(self) -> None:
        source = (
            "def export_series(out_dir):\n"
            "    export_to_file(module, args, out_dir / MODEL_FILE, graph_name=out_dir.name)\n"
        )
        node = next(iter(ast.parse(source).body))
        assert isinstance(node, ast.FunctionDef)
        arguments = _graph_name_arguments(node)

        assert [ast.unparse(a) for a in arguments] == ["out_dir.name"]
        assert _derived_from_a_path(arguments[0])

    def test_it_passes_a_plain_constant(self) -> None:
        """対（定数を名乗る形は落ちない）— これが無いと上の主張が恒真になりうる。"""
        source = "def export_series(out_dir):\n    f(path, graph_name=SIGLIP2_ROLE)\n"
        node = next(iter(ast.parse(source).body))
        assert isinstance(node, ast.FunctionDef)

        assert not _derived_from_a_path(_graph_name_arguments(node)[0])

    def test_a_staging_seat_name_is_also_caught(self) -> None:
        """作業席（`<部品>.staging`）から導く形も同じ述語が捕まえる。"""
        source = "def export_series(staged):\n    f(path, graph_name=staged.parent.name)\n"
        node = next(iter(ast.parse(source).body))
        assert isinstance(node, ast.FunctionDef)

        assert _derived_from_a_path(_graph_name_arguments(node)[0])
