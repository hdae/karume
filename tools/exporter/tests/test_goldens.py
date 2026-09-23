"""tiny golden fixtures の生成と、後段（Deno E2E）が依存する約束事の固定。

約束事は 3 つ: ①契約表の全 op を全部どれかのモデルが踏む ②レイアウトとテンソルキーが
README どおり ③固定 seed で再生成しても同じバイト列になる。
"""

from __future__ import annotations

import itertools
import os
import re
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from karume.container import container_parts
from karume.dims import eval_dim, parse_dim
from karume.goldens import (
    GOLDEN_SPECS,
    GOLDEN_T,
    INPUT_PREFIX,
    IO_FILE,
    MODEL_FILE,
    OUTPUT_PREFIX,
    generate_all,
    generate_golden,
)
from karume.ir import IrGraph
from karume.ops import EMITTABLE_OPS
from karume.verify import verify_container

#: コミット済み golden の置き場。**リポの綴りはテスト側が持つ** — 生成側（`karume.goldens`）は
#: 置き場を引数で受けるだけで repo topology を知らない（ADR 0065 Consequences）。
GOLDEN_ROOT = (
    Path(__file__).resolve().parents[3] / "packages" / "runtime" / "tests" / "fixtures" / "golden"
)

#: 後段テストが読む配布物なので、リポジトリに置ける大きさに留める。
MAX_FILE_BYTES = 16 * 1024

#: 意味論 dtype → io.safetensors の格納 torch dtype（ADR 0009 の境界正規化）。
IO_ENCODING = {"f32": torch.float32, "i32": torch.int32, "bool": torch.uint32}


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    """一時ディレクトリへ全 golden を再生成する（リポジトリの成果物には触らない）。"""
    root = tmp_path_factory.mktemp("golden")
    return root, generate_all(root)


def _io_tensors(root, name) -> dict[str, torch.Tensor]:
    with safe_open(str(root / name / IO_FILE), framework="pt") as handle:
        # safe_open は Mapping ではないので keys() が唯一の列挙手段。
        return {key: handle.get_tensor(key) for key in handle.keys()}  # noqa: SIM118


def _bindings(graph: IrGraph, io: dict[str, torch.Tensor]) -> dict[str, int]:
    """入力 shape の次元位置からシンボルを束縛する（要素数からの逆算はしない）。"""
    bindings: dict[str, int] = {}
    for spec in graph.inputs:
        actual = io[f"{INPUT_PREFIX}{spec.name}"].shape
        for index, dim in enumerate(spec.shape):
            if not isinstance(dim, str):
                continue
            expr = parse_dim(dim)
            if expr.coeff == 1 and expr.offset == 0:
                bindings[expr.sym] = int(actual[index])
    return bindings


#: README の golden 台帳の目印（この行の後ろの最初の表が「モデル名 | 記号次元 | IR ops」）。
COVERAGE_MARKER = "Current models and coverage:"
#: coverage 表の 1 行（見出し行と区切り行は名前を持たないので落ちる）。
COVERAGE_ROW = re.compile(r"^\| `([a-z0-9_]+)`\s+\|")


def _readme_coverage_names() -> list[str]:
    """README の coverage 表に並ぶモデル名（Module 表など他の表と混ざらないよう位置で切る）。"""
    text = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    _, marker, after = text.partition(COVERAGE_MARKER)
    assert marker, f"README に '{COVERAGE_MARKER}' が無い（表の位置を決められない）"
    rows = itertools.takewhile(
        lambda line: line.startswith("|"),
        itertools.dropwhile(lambda line: not line.startswith("|"), after.splitlines()),
    )
    return [match.group(1) for match in map(COVERAGE_ROW.match, rows) if match]


class TestCoverage:
    def test_the_readme_ledger_lists_exactly_the_generated_models(self):
        """台帳と現物のズレを門にする（op を足したら golden を足す契約の可視面 — ADR 0005）。

        表は recipe 作者の一次資料なので、人手更新に任せると op 追加のたびに同じ経路で
        遅れる（実際に ADR 0017 の 2 本ぶん遅れていた）。
        """
        assert set(_readme_coverage_names()) == {spec.name for spec in GOLDEN_SPECS}

    def test_every_contract_op_is_exercised_by_some_model(self, generated):
        _, graphs = generated
        covered = {op for graph in graphs.values() for op in graph.required_ops}

        assert covered == set(EMITTABLE_OPS)

    def test_a_model_carries_a_symbolic_input_dimension(self, generated):
        _, graphs = generated

        assert any(graph.symbols for graph in graphs.values())

    def test_a_model_exercises_right_aligned_broadcast(self, generated):
        """rank の違う被演算子を持つ binary が最低 1 本ある。"""
        _, graphs = generated
        ranks = [
            {len(_declared_shape(graph, name)) for name in node.ins}
            for graph in graphs.values()
            for node in graph.nodes
            if node.op in {"add", "sub", "mul", "div"}
        ]

        assert any(len(distinct) > 1 for distinct in ranks)

    def test_a_model_carries_a_non_f32_output(self, generated):
        """readback の非 f32 経路を E2E に載せる（ADR 0009）— f32 固定だと踏まれない。"""
        _, graphs = generated
        out_dtypes = {
            graph.values[name].dtype for graph in graphs.values() for name in graph.outputs
        }

        assert out_dtypes - {"f32"}

    def test_a_model_exercises_the_i64_input_boundary(self, generated):
        """torch 既定の整数 i64 は入力宣言で i32 へ正規化される（ADR 0009）。"""
        _, graphs = generated
        in_dtypes = {spec.dtype for graph in graphs.values() for spec in graph.inputs}

        assert "i32" in in_dtypes

    def test_the_bmm_model_uses_four_distinct_axis_lengths(self, generated):
        """B / M / K / N が全て違う長さであること（ACTIVE_DESIGN の Pitfalls）。

        バッチ stride を隣の次元の積で組む誤りは、2 軸が同じ長さの形では数値に出ない。
        golden を縮めて対称な形にした瞬間にここが落ちる。
        """
        _, graphs = generated
        bmms = [
            (_declared_shape(graph, node.ins[0]), _declared_shape(graph, node.ins[1]))
            for graph in graphs.values()
            for node in graph.nodes
            if node.op == "bmm"
        ]

        assert bmms
        assert any(
            len({a[0], a[1], a[2], b[2]}) == 4 and len({a[1] * a[2], a[2] * b[2], a[1] * b[2]}) == 3
            for a, b in bmms
        )

    def test_the_attention_model_uses_five_distinct_axis_lengths(self, generated):
        """融合 attention の golden は B / H / M / N / D が全て違う長さ（ADR 0023）。

        カーネルは B と H を 1 本のバッチ軸へ畳むので、B=1（実測形）や B==H では軸の
        取り違えが値に出ない（設計 recon §4.6 の検出限界 ①）。golden を縮めて対称な形に
        した瞬間にここが落ちる。
        """
        _, graphs = generated
        attentions = [
            [_declared_shape(graph, name) for name in node.ins]
            for graph in graphs.values()
            for node in graph.nodes
            if node.op == "attention"
        ]

        assert attentions, "attention を踏む golden が無い"
        assert any(len({q[0], q[1], q[2], k[2], q[3]}) == 5 for q, k, _ in attentions), (
            f"軸が 5 種類の長さになっていない: {attentions}"
        )

    def test_the_decomposed_attention_models_are_untouched(self, generated):
        """SDPA 保存はターゲット別（ADR 0023）— 既存 golden は分解形のまま。

        `attention_block` は `torch.bmm` + `torch.softmax` の手書き分解形、`masked_scores` は
        そもそも SDPA を通っていない。ここが動いたら保存の適用範囲が漏れている。
        """
        _, graphs = generated

        for name in ("attention_block", "masked_scores"):
            ops = {node.op for node in graphs[name].nodes}
            assert "attention" not in ops, f"{name} に attention が漏れている"
        assert "bmm" in {node.op for node in graphs["attention_block"].nodes}
        assert "softmax" in {node.op for node in graphs["masked_scores"].nodes}

    def test_every_gather_index_is_i32_from_both_sources(self, generated):
        """gather の添字スロットは必ず意味論 i32（スロット別 dtype 契約 — ADR 0012）。

        出どころは 2 通り: グラフ入力（gather_last_dim）と、記号依存の部分木を Tmax で焼いた
        **i32 initializer の prefix**（symbolic_table — ADR 0010）。両方が golden に在ることを
        要求する — 片方だけだと i32 添字の経路の一方が無検証のまま残る。
        """
        _, graphs = generated
        index_slots = [
            (model, graph, node.ins[1])
            for model, graph in graphs.items()
            for node in graph.nodes
            if node.op == "gather"
        ]

        assert index_slots
        sources = set()
        for model, graph, name in index_slots:
            declared = [spec for spec in graph.inputs if spec.name == name]
            dtype = declared[0].dtype if declared else graph.values[name].dtype
            assert dtype == "i32", f"{model}: gather の添字 '{name}' が i32 でない"
            sources.add("input" if declared else "folded")
        assert sources == {"input", "folded"}

    def test_a_model_runs_matmul_over_initializer_weights(self, generated):
        _, graphs = generated
        matmuls = [
            node
            for graph in graphs.values()
            for node in graph.nodes
            if node.op == "matmul" and node.ins[1] in graph.initializers
        ]

        assert matmuls


def _declared_shape(graph: IrGraph, name: str) -> list:
    for spec in graph.inputs:
        if spec.name == name:
            return spec.shape
    return graph.values[name].shape


#: コミット済み golden に `krm` の現物が在るか（skip 条件）。
#:
#: MUST: 探す綴りは**連番**（`model-NNNNN-of-NNNNN.krm`）— 代表 path `model.krm` 自身は
#: 書かれない（{@link _model_parts} の docstring）ので、`model.krm*` で探すと 1 本も一致せず
#: 「krm が置かれた時点で自動で戻る」が永久に成立しない（= 突合の永久無効化）。
COMMITTED_CONTAINERS = sorted(
    GOLDEN_ROOT.glob(f"*/{Path(MODEL_FILE).stem}-*{Path(MODEL_FILE).suffix}")
)


def _model_parts(root: Path, name: str) -> tuple[Path, ...]:
    """モデルの現物（コンテナの part 列 — part 0 から）。

    配布形は常に連番へ分割されるので（container-v1 §8）、代表 path `model.krm` 自身は
    書かれない — 現物を数えるのは `container_parts` の役目。
    """
    return container_parts(root / name / MODEL_FILE)


class TestLayout:
    @pytest.mark.parametrize("spec", GOLDEN_SPECS, ids=lambda s: s.name)
    def test_each_model_directory_has_the_parts_and_the_io(self, generated, spec):
        root, _ = generated

        parts = _model_parts(root, spec.name)
        assert all(part.is_file() for part in parts)
        assert (root / spec.name / IO_FILE).is_file()
        # part 0（2 文書）+ part 1（const 領域）は**常に**並ぶ（container-v1 §8）ので、
        # 重みを 1 本も持たない golden でも 2 本以上になり、代表 path 自身は書かれない。
        assert len(parts) >= 2
        assert not (root / spec.name / MODEL_FILE).exists()

    @pytest.mark.parametrize("spec", GOLDEN_SPECS, ids=lambda s: s.name)
    def test_each_model_passes_the_full_verification(self, generated, spec):
        root, _ = generated

        verify_container(_model_parts(root, spec.name), blocks=True)

    @pytest.mark.parametrize("spec", GOLDEN_SPECS, ids=lambda s: s.name)
    def test_io_keys_follow_the_naming_convention(self, generated, spec):
        root, graphs = generated
        graph = graphs[spec.name]

        io = _io_tensors(root, spec.name)

        assert set(io) == {f"{INPUT_PREFIX}{s.name}" for s in graph.inputs} | {
            f"{OUTPUT_PREFIX}{index}" for index in range(len(graph.outputs))
        }

    @pytest.mark.parametrize("spec", GOLDEN_SPECS, ids=lambda s: s.name)
    def test_io_dtypes_use_the_runtime_representation(self, generated, spec):
        """io は意味論 dtype の実表現で書く（i64 → int32 / bool → uint32 の 0/1）。

        後段の Deno E2E はこの対応で view を張るので、揃っていないと golden が読めない。
        """
        root, graphs = generated
        graph = graphs[spec.name]
        io = _io_tensors(root, spec.name)

        for declared in graph.inputs:
            assert io[f"{INPUT_PREFIX}{declared.name}"].dtype is IO_ENCODING[declared.dtype]
        for index, name in enumerate(graph.outputs):
            assert io[f"{OUTPUT_PREFIX}{index}"].dtype is IO_ENCODING[graph.values[name].dtype]

    @pytest.mark.parametrize("spec", GOLDEN_SPECS, ids=lambda s: s.name)
    def test_io_shapes_agree_with_the_declarations(self, generated, spec):
        """宣言 shape を束縛で解決したものが実テンソルと一致する（レイアウトの意味検査）。"""
        root, graphs = generated
        graph = graphs[spec.name]
        io = _io_tensors(root, spec.name)
        bindings = _bindings(graph, io)

        for index, name in enumerate(graph.outputs):
            declared = [
                eval_dim(parse_dim(dim), bindings) if isinstance(dim, str) else dim
                for dim in _declared_shape(graph, name)
            ]
            assert declared == list(io[f"{OUTPUT_PREFIX}{index}"].shape)

    @pytest.mark.parametrize("spec", GOLDEN_SPECS, ids=lambda s: s.name)
    def test_symbolic_models_are_bound_to_the_golden_length(self, generated, spec):
        root, graphs = generated
        graph = graphs[spec.name]

        bindings = _bindings(graph, _io_tensors(root, spec.name))

        assert set(bindings) == set(graph.symbols)
        assert all(value == GOLDEN_T for value in bindings.values())

    @pytest.mark.parametrize("spec", GOLDEN_SPECS, ids=lambda s: s.name)
    def test_fixtures_stay_small(self, generated, spec):
        root, _ = generated

        for path in (*_model_parts(root, spec.name), root / spec.name / IO_FILE):
            assert path.stat().st_size < MAX_FILE_BYTES


class TestDeterminism:
    """固定 seed の再生成はバイト単位で同一（codegen 決定性と同じ規律）。

    ただし io（torch CPU の期待出力）のバイト一致は**参照環境専用** — oneDNN が CPU の ISA で
    gemm / conv の kernel を出し分けるため、計算結果は最終 bit がマシン依存になる（CI 初回
    実測 2026-08-16: GitHub runner で activations / conv2d_block の 2 spec だけ ±1〜2 ulp）。
    model（グラフ + 固定 seed の重み — torch の CPU RNG はクロスマシンで決定的）は
    どの環境でもバイト一致を要求する。sha256 参照門が参照環境専用なのと同じ性格
    （docs/limitations.md）で、緩めるのではなく環境で検査を分ける。
    """

    @pytest.mark.parametrize("spec", GOLDEN_SPECS, ids=lambda s: s.name)
    def test_two_runs_produce_the_same_model_bytes(self, generated, tmp_path, spec):
        """同じ seed から 2 度生成してバイト一致（決定性そのものを被験体にする）。

        置き場が違ってもバイトは動かない（ファイル名は中身に入らない）。
        """
        root, _ = generated
        generate_golden(spec, tmp_path)

        assert [path.read_bytes() for path in _model_parts(root, spec.name)] == [
            path.read_bytes() for path in _model_parts(tmp_path, spec.name)
        ]

    @pytest.mark.skipif(
        not COMMITTED_CONTAINERS,
        reason="コミット済み golden がまだ旧配布形（safetensors）— 再生成は TS 側の読み手と"
        "同じ段で行う（段 3a の契約外）。krm が置かれた時点でこの門は自動で戻る",
    )
    @pytest.mark.parametrize("spec", GOLDEN_SPECS, ids=lambda s: s.name)
    def test_regeneration_matches_the_committed_model(self, generated, spec):
        root, _ = generated

        generated_parts = _model_parts(root, spec.name)
        committed = _model_parts(GOLDEN_ROOT, spec.name)
        assert [path.name for path in committed] == [path.name for path in generated_parts], (
            f"生成物が未コミット: {GOLDEN_ROOT / spec.name}"
        )
        assert [path.read_bytes() for path in committed] == [
            path.read_bytes() for path in generated_parts
        ]

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="io の torch CPU 出力は最終 bit がマシン依存（oneDNN の ISA 別 kernel）— "
        "バイト一致は参照環境専用",
    )
    @pytest.mark.parametrize("spec", GOLDEN_SPECS, ids=lambda s: s.name)
    def test_regeneration_matches_the_committed_reference_io(self, generated, spec):
        root, _ = generated

        committed = GOLDEN_ROOT / spec.name / IO_FILE
        assert committed.is_file(), f"生成物が未コミット: {committed}"
        assert committed.read_bytes() == (root / spec.name / IO_FILE).read_bytes()


class TestTheCommittedComparisonCanComeBack:
    """skip 条件が「置いた瞬間に戻る」綴りであることを、合成の置き場で確かめる。

    MUST: 条件は**連番**（`model-NNNNN-of-NNNNN.krm`）で探す。代表 path `model.krm` は
    書かれない仕様なので、`model.krm*` で探すと 1 本も一致せず、golden を `krm` へ再生成した
    後も突合が skip のまま沈黙する（既存テストの実質的な無効化）。
    """

    @staticmethod
    def _committed(root: Path) -> list[Path]:
        """`test_goldens` の skip 条件と**同じ 1 本**の綴りで探す。"""
        return sorted(root.glob(f"*/{Path(MODEL_FILE).stem}-*{Path(MODEL_FILE).suffix}"))

    def test_a_directory_of_committed_parts_is_found(self, tmp_path: Path) -> None:
        root = tmp_path / "golden"
        (root / "mlp").mkdir(parents=True)
        (root / "mlp" / "model-00001-of-00002.krm").write_bytes(b"0")
        (root / "mlp" / "model-00002-of-00002.krm").write_bytes(b"1")

        assert [path.name for path in self._committed(root)] == [
            "model-00001-of-00002.krm",
            "model-00002-of-00002.krm",
        ]

    def test_a_directory_of_legacy_shards_is_not_found(self, tmp_path: Path) -> None:
        """対照 — 旧配布形しか無い間は skip のまま（今のリポジトリの状態）。"""
        root = tmp_path / "golden"
        (root / "mlp").mkdir(parents=True)
        (root / "mlp" / "model-00001-of-00002.safetensors").write_bytes(b"0")

        assert self._committed(root) == []
