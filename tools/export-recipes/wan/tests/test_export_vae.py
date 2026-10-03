"""`wan/export_vae.py` の約束事 — chunk グラフ 2 種の IR が決定 2 の取り決めを満たすこと（実重み）。

export は潜在タイル 8 で回す（グラフの構造はタイル辺に依らず、宣言 shape だけが変わる）。
実寸（タイル 32）の要約は `python -m wan.export_vae` が出す（README の VAE 節）。

見るもの:

- op の集合（pad が無い・SiLU の分解と保存した attention を含む既知の集合だけ）
- 値の rank（4 以下。rank 5 は conv3d の重みだけ — rank 6 の reshape もここで落ちる）
- cache の入出力（名前・形・順が決定 2 の表どおりで、出力の cache k が cache 入力 k の cat
  から作られている）
- 検査関数（{@link wan.export_vae.assert_chunk_graph}）自身が、順の取り違えと語彙外の op を
  落とすこと
"""

from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION
from karume.ir import IrGraph, IrNode, IrValue
from karume.pipeline import export_module
from wan import export_vae, vae_patch

TILE = 8


@pytest.fixture(scope="module")
def vae(wan_snapshot):
    pytest.importorskip("diffusers")
    from diffusers import AutoencoderKLWan

    return AutoencoderKLWan.from_pretrained(
        wan_snapshot, subfolder="vae", torch_dtype=torch.float32
    ).eval()


@pytest.fixture(scope="module")
def exported(vae) -> dict[bool, tuple[IrGraph, vae_patch.WanVaeChunkDecoder]]:
    """first / next の IR（キーは `first`）。"""
    graphs = {}
    for first in (True, False):
        module = vae_patch.WanVaeChunkDecoder(vae, first=first).eval()
        with torch.no_grad():
            graph, _ = export_module(
                module,
                export_vae.example_inputs(module, TILE),
                preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
            )
        graphs[first] = (graph, module)
    return graphs


class TestChunkGraphs:
    @pytest.mark.parametrize("first", [True, False])
    def test_the_graph_keeps_the_contract_of_the_adr(self, exported, first: bool):
        graph, module = exported[first]

        export_vae.assert_chunk_graph(graph, module, TILE)

    @pytest.mark.parametrize("first", [True, False])
    def test_there_is_no_pad_and_only_conv3d_weights_are_rank5(self, exported, first: bool):
        graph, _ = exported[first]
        rank5 = {name for name, value in graph.values.items() if len(value.shape) >= 5}
        weights = {node.ins[1] for node in graph.nodes if node.op == "conv3d"}

        assert "pad" not in {node.op for node in graph.nodes}
        assert rank5 == weights
        assert rank5 <= set(graph.initializers)
        assert all(len(graph.values[name].shape) == 5 for name in rank5)

    def test_both_graphs_use_the_same_op_set(self, exported):
        first_ops = set(exported[True][0].required_ops)
        next_ops = set(exported[False][0].required_ops)

        assert first_ops == next_ops == export_vae.EXPECTED_OPS

    def test_only_next_runs_the_two_time_convs(self, exported):
        """conv3d は post-quant 1・shortcut 1・cache 付き 30（first）/ 32（next）。"""
        counts = {
            first: Counter(node.op for node in exported[first][0].nodes) for first in exported
        }

        assert counts[True]["conv3d"] == 32
        assert counts[False]["conv3d"] == 34
        assert counts[True]["cat"] == 30
        assert counts[False]["cat"] == 32

    def test_next_frames_are_four_per_chunk(self, exported):
        first_graph, _ = exported[True]
        next_graph, _ = exported[False]

        assert first_graph.values[first_graph.outputs[0]].shape == [3, 1, 8 * TILE, 8 * TILE]
        assert next_graph.values[next_graph.outputs[0]].shape == [3, 4, 8 * TILE, 8 * TILE]


class TestTheCheckItself:
    """検査が緩むと形の揃う cache どうしの入れ替えが素通りする — 故障注入で赤を確かめる。"""

    def test_two_swapped_cache_outputs_are_caught(self, exported):
        graph, module = exported[False]
        broken = copy.deepcopy(graph)
        # cache_01 と cache_02 は同じ形 [384, 2, t, t] — 形の検査だけでは見分けられない。
        broken.outputs[2], broken.outputs[3] = broken.outputs[3], broken.outputs[2]

        with pytest.raises(export_vae.ChunkGraphError, match="順の取り違え"):
            export_vae.assert_chunk_graph(broken, module, TILE)

    def test_an_op_outside_the_set_is_caught(self, exported):
        graph, module = exported[True]
        broken = copy.deepcopy(graph)
        broken.nodes.append(IrNode(op="pad", ins=[graph.outputs[0]], outs=["padded"], attrs={}))

        with pytest.raises(export_vae.ChunkGraphError, match="pad"):
            export_vae.assert_chunk_graph(broken, module, TILE)

    def test_a_rank6_value_is_caught(self, exported):
        graph, module = exported[True]
        broken = copy.deepcopy(graph)
        name = next(iter(name for name in broken.values if name not in broken.initializers))
        broken.values[name] = IrValue(dtype="f32", shape=[1, 1, 1, 1, 1, 1])

        with pytest.raises(export_vae.ChunkGraphError, match="rank 6"):
            export_vae.assert_chunk_graph(broken, module, TILE)


class TestFixtureCases:
    def test_band_and_accept_differ_in_latents_and_chunk_boundaries(self):
        band, accept, long = export_vae.FIXTURE_CASES

        assert (band.role, accept.role) == ("band", "accept")
        assert band.seed != accept.seed
        assert band.chunks != accept.chunks
        assert band.chunks == 9  # 33 フレーム = 最初の到達目標の 1 タイル分（ADR 0118）
        # 81 フレーム（ADR 0118 段 8）の受入れ: 1 + 4·20 = 81。帯は band のまま（受入れ側）。
        assert (long.name, long.role, long.chunks) == ("long", "accept", 21)
        assert long.seed not in {band.seed, accept.seed}


class TestTheSetIsPublishedTogether:
    """first / next / フィクスチャは一組で据わる — 途中で落ちた実走が新旧の混ざった組を残さない。

    export と参照の decode は差し替えて、置き場の規律だけを見る（実重みは読まない）。
    """

    OLD = "tile32-old"

    @pytest.fixture
    def series(self, tmp_path, monkeypatch):
        """旧世代の組が据わった系列と、書き込みを差し替えた export_vae。"""
        root = tmp_path / "series"
        for target in export_vae.TARGETS:
            (root / target).mkdir(parents=True)
            (root / target / export_vae.MODEL_FILE).write_text(self.OLD)
        for case in export_vae.FIXTURE_CASES:
            export_vae.fixture_path(root, case).write_text(self.OLD)

        class Decoder:
            def __init__(self, _vae, first: bool) -> None:
                self.first = first

            def eval(self):
                return self

        def export_to_file(_module, _inputs, path, *, graph_name, **_kwargs):
            path.write_text(f"{graph_name}-new")
            return SimpleNamespace(
                nodes=[],
                inputs=[],
                outputs=["frame"],
                initializers={},
                values={"frame": SimpleNamespace(shape=[3, 1, 8, 8])},
            )

        def save_file(_tensors, path, metadata):
            Path(path).write_text(f"{metadata['seed']}-new")

        monkeypatch.setattr(
            export_vae,
            "vae_patch",
            SimpleNamespace(
                WanVaeChunkDecoder=Decoder,
                reference_decode_unclamped=lambda _vae, latents: torch.zeros(1, 3, 1, 8, 8),
            ),
        )
        monkeypatch.setattr(export_vae, "example_inputs", lambda _module, _tile: ())
        monkeypatch.setattr(export_vae, "export_to_file", export_to_file)
        monkeypatch.setattr(export_vae, "assert_chunk_graph", lambda _graph, _module, _tile: None)
        monkeypatch.setattr(
            export_vae,
            "storage_breakdown",
            lambda _graph: SimpleNamespace(
                compressed_tensors=0, compressed_bytes=0, plain_tensors=0, plain_bytes=0
            ),
        )
        monkeypatch.setattr(export_vae, "container_parts", lambda path: [path])
        monkeypatch.setattr(
            export_vae,
            "fixture_latents",
            lambda _vae, case, tile: torch.zeros(1, 16, 1, tile, tile),
        )
        monkeypatch.setattr(export_vae, "save_file", save_file)
        return root

    def _contents(self, root: Path) -> list[str]:
        graphs = [
            (root / target / export_vae.MODEL_FILE).read_text() for target in export_vae.TARGETS
        ]
        fixtures = [
            export_vae.fixture_path(root, case).read_text() for case in export_vae.FIXTURE_CASES
        ]
        return graphs + fixtures

    def _emit(self, root: Path, targets=export_vae.TARGETS) -> dict:
        return export_vae.emit_targets(targets, None, 16, root, None, fixtures=True)

    def test_a_complete_run_replaces_every_member(self, series):
        self._emit(series)

        assert self._contents(series) == [
            *(f"{target}-new" for target in export_vae.TARGETS),
            *(f"{case.seed}-new" for case in export_vae.FIXTURE_CASES),
        ]
        assert sorted(path.name for path in series.iterdir()) == sorted(
            [*export_vae.TARGETS]
            + [export_vae.fixture_path(series, case).name for case in export_vae.FIXTURE_CASES]
        )

    def test_a_failure_in_the_second_graph_leaves_the_first_untouched(self, series, monkeypatch):
        """next の export で落ちると、先に書いた first も据わらない（m6 の再現の形）。"""
        written = export_vae.export_to_file

        def failing(module, inputs, path, *, graph_name, **kwargs):
            if graph_name == export_vae.TARGET_NEXT:
                raise RuntimeError("next の export で落ちる（故障注入）")
            return written(module, inputs, path, graph_name=graph_name, **kwargs)

        monkeypatch.setattr(export_vae, "export_to_file", failing)
        with pytest.raises(RuntimeError, match="故障注入"):
            self._emit(series)

        assert self._contents(series) == [self.OLD] * 5
        assert not [path for path in series.iterdir() if path.name.endswith(".staging")]

    def test_a_failure_in_a_fixture_leaves_both_graphs_untouched(self, series, monkeypatch):
        """フィクスチャの 2 本目で落ちると、検査を通ったグラフ 2 本も据わらない。"""
        written = export_vae.save_file

        def failing(tensors, path, metadata):
            if metadata["seed"] == str(export_vae.FIXTURE_CASES[1].seed):
                raise OSError("フィクスチャの書き込みで落ちる（故障注入）")
            written(tensors, path, metadata)

        monkeypatch.setattr(export_vae, "save_file", failing)
        with pytest.raises(OSError, match="故障注入"):
            self._emit(series)

        assert self._contents(series) == [self.OLD] * 5

    def test_a_duplicated_target_is_refused_before_anything_is_written(self, series):
        with pytest.raises(export_vae.ChunkGraphError, match="重複"):
            self._emit(series, (export_vae.TARGET_FIRST, export_vae.TARGET_FIRST))

        assert self._contents(series) == [self.OLD] * 5
