"""emit 台本の公開規律 — 「門より前に final へ置かない」の機械門（全 family 横断）。

decode 系列の 2 台本が `karume.artifacts.staged_publication` で守っている不変条件を、
1-shot 系列の台本にも同じ形で掛ける。守るのは 1 つだけ:

    生成物を書く呼び（`export_to_file` / `_write_io` / `save_file` …）が、全て
    作業席の `with` の**内側**にあること。

外側に 1 本でも漏れると、落ちた実走が「検収門を通れる資産」を final に残す — io golden は
同じ壊れたラッパから採るので互いに整合し、TS 側の突合は**緑になる**（規律の綴りは
`_shared/decode_series._publish`）。この門が要るのは、その状態が「落ちた実走の後始末を忘れた」
ようには見えず、**次の検収まで沈黙する**から。

ソースを AST で読むのは、実行して確かめるには実重み（数 GB）か family ごとの tiny 模型が要る
ため。ここが見るのは「呼びの位置関係」だけで、据え替えそのものの規律（退避 → 昇格 →
失敗時の戻し）は core の `tools/exporter/tests/` が実測で持つ。

"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

RECIPES_ROOT = Path(__file__).resolve().parent.parent

#: 作業席を開く呼びの名前（family 内のラッパ経由も認める — `sbv2._staged_target` は
#: `staged_publication` を包んで出所記録まで同じ席へ書く）。
STAGING_CALLS = frozenset({"staged_publication", "_staged_target"})

#: 生成物を書く呼び。1 つでも席の外に出た瞬間、その台本は「門より前に final へ置く」形に戻る。
WRITER_CALLS = frozenset(
    {
        "export_to_file",
        "_write_container",
        "_write_io",
        "_write_mirror_io",
        "_write_greedy",
        "_write_export_provenance",
        "_write_lora_provenance",
        "_write_calib_provenance",
        "save_file",
        # gemma4 の product 台本が使う 2 本（PLE の shard 書き出しと出所記録）。名前が
        # 表に無いと、その台本を表へ載せても「書いている場所」を 1 つも見ないまま緑になる。
        "write_ple_shards",
        "write_record",
    }
)

#: (台本, emit を行う関数)。1-shot 台本 + 手本の decode 台本。
#:
#: MUST: 実在する台本が 1 本残らずここに載っていること（表 → 実在の
#: {@link TestTheGateItselfCanFail.test_every_listed_script_exists} と、実在 → 表 の
#: {@link TestEveryStagingScriptIsListed} の両方向）。片方向だけだと**載せ忘れが永久に沈黙する**。
EMIT_ENTRIES: tuple[tuple[str, str], ...] = (
    ("irodori/export.py", "export_series"),
    ("gemma4/export.py", "export_series"),
    ("gemma4/export_decode.py", "export_series"),
    ("gemma4/export_product.py", "export_series"),
    ("minicpm5/export.py", "export_series"),
    ("minicpm5/export_decode.py", "export_series"),
    ("embeddinggemma/export.py", "export_series"),
    ("deberta/export.py", "export_variant"),
    ("anima/export.py", "emit_target"),
    ("sbv2/export.py", "export_dp"),
    ("sbv2/export.py", "export_front"),
    ("sbv2/export.py", "export_flow"),
    ("sbv2/export.py", "export_dec"),
    ("sbv2/export.py", "export_voice"),
    ("siglip2/export.py", "export_series"),
    ("birefnet/export.py", "export_series"),
    ("depth_anything/export.py", "export_series"),
    ("vowel_detector/export.py", "export_series"),
    ("irodori/dacvae/export.py", "export_series"),
)


def _called_name(node: ast.AST) -> str | None:
    """呼びの名前（`f(...)` / `mod.f(...)` のどちらも末尾の綴りで引く）。"""
    if not isinstance(node, ast.Call):
        return None
    target = node.func
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return None


def _emit_function(source: str, name: str) -> ast.FunctionDef:
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} がモジュール直下に無い（台本の綴りが動いた）")


def _staged_bodies(function: ast.FunctionDef) -> list[list[ast.stmt]]:
    """作業席を開く `with` の本体（for の中に入っていても拾う）。"""
    return [
        node.body
        for node in ast.walk(function)
        if isinstance(node, ast.With)
        and any(_called_name(item.context_expr) in STAGING_CALLS for item in node.items)
    ]


def _writer_calls(nodes: list[ast.stmt]) -> set[tuple[str, int]]:
    return {
        (name, call.lineno)
        for statement in nodes
        for call in ast.walk(statement)
        if (name := _called_name(call)) in WRITER_CALLS
    }


@pytest.mark.parametrize(("script", "function"), EMIT_ENTRIES, ids=lambda value: str(value))
class TestEmitPublishesThroughAStagedSeat:
    @staticmethod
    def _function(script: str, function: str) -> ast.FunctionDef:
        return _emit_function((RECIPES_ROOT / script).read_text(encoding="utf-8"), function)

    def test_it_opens_a_staged_seat(self, script: str, function: str) -> None:
        assert _staged_bodies(self._function(script, function)), (
            f"{script}:{function} が作業席を開いていない — 生成物が final へ直接書かれる"
        )

    def test_every_write_happens_inside_that_seat(self, script: str, function: str) -> None:
        """席の外に残った書き込みを、名前と行番号で名指しする。"""
        node = self._function(script, function)
        inside = set().union(*(_writer_calls(body) for body in _staged_bodies(node)))
        outside = _writer_calls(node.body) - inside

        assert not outside, (
            f"{script}:{function} が作業席の外で書いている {sorted(outside)} —"
            "門より前に final へ置くと、落ちた実走が検収門を通れる資産を残す"
        )


class TestTheGateItselfCanFail:
    """門が「何も見ずに緑」へ退化していないこと（対象表が空・述語が恒真の 2 つを潰す）。"""

    def test_it_names_a_writer_left_outside_the_seat(self) -> None:
        source = (
            "def export_series(out_dir):\n"
            "    export_to_file(module, args, out_dir / MODEL_FILE)\n"
            "    with staged_publication(out_dir) as staged:\n"
            "        _write_io(module, graph, cases, staged)\n"
        )
        node = _emit_function(source, "export_series")
        inside = set().union(*(_writer_calls(body) for body in _staged_bodies(node)))

        assert _writer_calls(node.body) - inside == {("export_to_file", 2)}

    def test_it_sees_no_seat_when_there_is_none(self) -> None:
        source = "def export_series(out_dir):\n    export_to_file(module, args, out_dir)\n"

        assert _staged_bodies(_emit_function(source, "export_series")) == []

    def test_every_listed_script_exists(self) -> None:
        """対象表の綴りが動いたら（ファイル名変更・移動）ここで落ちる。"""
        for script, _function in EMIT_ENTRIES:
            assert (RECIPES_ROOT / script).is_file(), script


#: `_staged_target` の**定義そのもの**（ラッパは中で `staged_publication` を開くので、
#: 走査すると「席を開くモジュール直下関数」として拾われる）。
STAGING_WRAPPER_DEFINITIONS = frozenset({"_staged_target"})


def _staging_functions() -> set[tuple[str, str]]:
    """作業席を開くモジュール直下関数を、ソース走査で列挙する（実在 → 表 の側）。

    走査対象を `export*.py` に限るのは、テスト用の疑似 `with` を偽陽性にしないため
    （`sbv2/tests/test_export.py` は綴りが `test_` 始まりなので拾われない）。
    """
    found: set[tuple[str, str]] = set()
    for path in sorted(RECIPES_ROOT.glob("*/**/export*.py")):
        if "tests" in path.parts or "__pycache__" in path.parts:
            continue
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name in STAGING_WRAPPER_DEFINITIONS or not _staged_bodies(node):
                continue
            found.add((str(path.relative_to(RECIPES_ROOT)), node.name))
    return found


class TestEveryStagingScriptIsListed:
    """実在 → 表 の逆方向（表 → 実在 だけだと、載せ忘れが永久に沈黙する）。"""

    def test_the_scan_finds_the_scripts(self) -> None:
        """走査が 0 本なら、この門は恒真になる。"""
        assert len(_staging_functions()) >= 18

    def test_no_staging_script_is_missing_from_the_table(self) -> None:
        missing = sorted(_staging_functions() - set(EMIT_ENTRIES))

        assert missing == [], (
            f"作業席を開いているのに公開規律の門が掛かっていない台本がある: {missing} —"
            " EMIT_ENTRIES へ足す（表に無い台本は「門より前に final へ置く」形に戻せる）"
        )
