"""tiny golden fixtures の生成と、後段（Deno E2E）が依存する約束事の固定。

約束事は 3 つ: ①契約表の全 op を全部どれかのモデルが踏む ②レイアウトとテンソルキーが
README どおり ③固定 seed で再生成しても同じバイト列になる。
"""

from __future__ import annotations

import pytest
import torch
from safetensors import safe_open

from karume.dims import eval_dim, parse_dim
from karume.goldens import (
    GOLDEN_ROOT,
    GOLDEN_SPECS,
    GOLDEN_T,
    INPUT_PREFIX,
    IO_FILE,
    MODEL_FILE,
    OUTPUT_PREFIX,
    generate_all,
)
from karume.ir import IrGraph
from karume.ops import EMITTABLE_OPS
from karume.verify import verify_model

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


class TestCoverage:
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


class TestLayout:
    @pytest.mark.parametrize("spec", GOLDEN_SPECS, ids=lambda s: s.name)
    def test_each_model_directory_has_both_files(self, generated, spec):
        root, _ = generated

        assert (root / spec.name / MODEL_FILE).is_file()
        assert (root / spec.name / IO_FILE).is_file()

    @pytest.mark.parametrize("spec", GOLDEN_SPECS, ids=lambda s: s.name)
    def test_each_model_passes_the_full_verification(self, generated, spec):
        root, _ = generated

        verify_model(root / spec.name / MODEL_FILE)

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

        for name in (MODEL_FILE, IO_FILE):
            assert (root / spec.name / name).stat().st_size < MAX_FILE_BYTES


class TestDeterminism:
    """固定 seed の再生成はバイト単位で同一（codegen 決定性と同じ規律）。"""

    @pytest.mark.parametrize("spec", GOLDEN_SPECS, ids=lambda s: s.name)
    def test_regeneration_matches_the_committed_fixture(self, generated, spec):
        root, _ = generated

        for name in (MODEL_FILE, IO_FILE):
            committed = GOLDEN_ROOT / spec.name / name
            assert committed.is_file(), f"生成物が未コミット: {committed}"
            assert committed.read_bytes() == (root / spec.name / name).read_bytes()
