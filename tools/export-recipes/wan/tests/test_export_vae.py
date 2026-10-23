"""`wan/export_vae.py` の約束事 — chunk グラフ 2 種の IR が決定 2 の取り決めを満たすこと。

Wan2.1 は実重み（潜在タイル 8）、Wan2.2 は合成の VAE（conftest の `ti2v_synthetic_vae`・潜在
タイル 4 — 5B と同じ構造なので op とノードの並びは同じで、宣言 shape だけが変わる）で export する
（グラフの構造はタイル辺に依らない）。実寸（2.1 はタイル 32・2.2 はタイル 16）の要約は
`python -m wan.export_vae [--model ti2v-5b]` が出す（README の VAE 節）。

見るもの:

- op の集合（pad が無い・SiLU の分解と保存した attention を含む既知の集合だけ）
- 値の rank（4 以下。rank 5 は conv3d の重みだけ — rank 6 の reshape もここで落ちる）
- cache の入出力（名前・形・順が決定 2 の表どおりで、出力の cache k が cache 入力 k の cat
  から作られている）
- 検査関数（{@link wan.export_vae.assert_chunk_graph}）自身が、順の取り違えと語彙外の op を
  落とすこと
- `--dtype i8`（ADR 0133 決定 7）: 合成の 2.2 VAE の i8 の書き出しで、畳み込みの重みだけが
  `int8-sym`（出力チャネルごとの scale）になり、グラフの構造と他の格納は f16 の回と同じで、
  フィクスチャを書かないこと・i8 系列を持たないモデルは拒むこと
- `--shared`（ADR 0134 決定 1）: 合成の 2.2 VAE の f16 / i8 の組（タイル 4）を貸し手に、
  タイル 8 の組 `<ターゲット>_8` を同じ系列の根へ書き、initializer が全部貸し手の同じキー・形・
  格納の共有宣言で、重みのバイトを持たず、chunk グラフの取り決めがタイル 8 で成り立ち、貸し手の
  組がバイト同一のままであること・貸し手の無い系列 / フィクスチャ / 別の格納・別のタイルの
  貸し手 / `--tile` 無しを拒むこと
"""

from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch

from _shared.paths import SERIES_ROOT
from karume.container import Provenance, codec_entry, container_parts
from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION
from karume.ir import IrGraph, IrInput, IrNode, IrStorage, IrValue
from karume.ops import WEIGHT_SLOTS
from karume.pipeline import export_module
from karume.quantize import round_weights_to_f16
from karume.verify import BoundGraph, ir_graph_from_container, verify_container
from wan import export_vae, vae_patch
from wan.sources import SOURCES

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
        return export_vae.emit_targets(
            targets, None, 16, root, None, cases=export_vae.FIXTURE_CASES, patch_size=None
        )

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


# ---- Wan2.2（合成の VAE — ADR 0121 段 4）-----------------------------------------------

#: 合成の 2.2 VAE の潜在タイル。
TI2V_TILE = 4


@pytest.fixture(scope="module")
def ti2v_exported(ti2v_synthetic_vae) -> dict[bool, tuple[IrGraph, vae_patch.WanVaeChunkDecoder]]:
    """合成の 2.2 VAE の first / next の IR（キーは `first`）。"""
    graphs = {}
    for first in (True, False):
        module = vae_patch.WanVaeChunkDecoder(ti2v_synthetic_vae, first=first).eval()
        with torch.no_grad():
            graph, _ = export_module(
                module,
                export_vae.example_inputs(module, TI2V_TILE),
                preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
            )
        graphs[first] = (graph, module)
    return graphs


class TestTi2vChunkGraphs:
    @pytest.mark.parametrize("first", [True, False])
    def test_the_graph_keeps_the_contract_of_the_adr(self, ti2v_exported, first: bool):
        graph, module = ti2v_exported[first]

        export_vae.assert_chunk_graph(graph, module, TI2V_TILE)

    @pytest.mark.parametrize(("first", "frames", "caches"), [(True, 1, 30), (False, 4, 32)])
    def test_a_48_channel_latent_in_and_12_channel_frames_out(
        self, ti2v_exported, first: bool, frames: int, caches: int
    ):
        """潜在 48 ch・フレームは patchify 空間の 12 ch（unpatchify はホスト — 決定 6）。"""
        graph, _ = ti2v_exported[first]
        side = 8 * TI2V_TILE

        assert graph.inputs[0].name == export_vae.LATENT_INPUT
        assert graph.inputs[0].shape == [48, 1, TI2V_TILE, TI2V_TILE]
        assert len(graph.inputs) == 1 + caches
        assert graph.values[graph.outputs[0]].shape == [12, frames, side, side]

    def test_both_graphs_use_the_same_op_set(self, ti2v_exported):
        """ショートカットは reshape / expand / permute / slice / add だけ — 2.1 と同じ op の
        集合。"""
        first_ops = set(ti2v_exported[True][0].required_ops)
        next_ops = set(ti2v_exported[False][0].required_ops)

        assert first_ops == next_ops == export_vae.EXPECTED_OPS

    def test_only_next_runs_the_two_time_convs(self, ti2v_exported):
        """conv3d は post-quant 1・shortcut 2（up2 / up3 の Res0）・cache 付き 30（first）/
        32（next）。"""
        counts = {
            first: Counter(node.op for node in ti2v_exported[first][0].nodes)
            for first in ti2v_exported
        }

        assert counts[True]["conv3d"] == 33
        assert counts[False]["conv3d"] == 35
        assert counts[True]["cat"] == 30
        assert counts[False]["cat"] == 32

    def test_three_channel_frames_are_caught_in_a_ti2v_graph(self, ti2v_exported):
        """検査が落ちうること: フレームのチャネルはモジュールから導く（2.1 の 3 を literal で
        持たない）。"""
        graph, module = ti2v_exported[False]
        broken = copy.deepcopy(graph)
        side = 8 * TI2V_TILE
        broken.values[broken.outputs[0]] = IrValue(dtype="f32", shape=[3, 4, side, side])

        with pytest.raises(export_vae.ChunkGraphError, match="フレームの形"):
            export_vae.assert_chunk_graph(broken, module, TI2V_TILE)

    def test_a_16_channel_latent_is_caught_in_a_ti2v_graph(self, ti2v_exported):
        graph, module = ti2v_exported[True]
        broken = copy.deepcopy(graph)
        broken.inputs[0] = IrInput(
            name=export_vae.LATENT_INPUT, dtype="f32", shape=[16, 1, TI2V_TILE, TI2V_TILE]
        )

        with pytest.raises(export_vae.ChunkGraphError, match="入力"):
            export_vae.assert_chunk_graph(broken, module, TI2V_TILE)


class TestVaeSeries:
    def test_the_wan21_row_is_the_existing_series(self):
        assert export_vae.VAE_SERIES["t2v-1.3b"] == export_vae.VaeSeries(
            "wan2.1-t2v-1.3b-f16-dyn", 32, None, export_vae.FIXTURE_CASES
        )

    def test_the_ti2v_row(self):
        """ADR 0121 決定 6 / 7: 系列 `wan2.2-ti2v-5b-f16-dyn`・潜在タイル 16・patch 2。"""
        series = export_vae.VAE_SERIES["ti2v-5b"]

        assert (series.series, series.tile, series.patch_size) == (
            "wan2.2-ti2v-5b-f16-dyn",
            16,
            2,
        )
        assert series.cases == export_vae.TI2V_FIXTURE_CASES

    def test_ti2v_band_and_accept_differ_in_latents_and_chunk_boundaries(self):
        band, accept, long = export_vae.TI2V_FIXTURE_CASES

        assert (band.name, band.role, band.chunks) == ("band", "band", 9)
        assert (accept.name, accept.role) == ("accept", "accept")
        assert band.seed != accept.seed
        assert band.chunks != accept.chunks
        assert (long.name, long.role, long.chunks) == ("long", "accept", 21)
        assert long.seed not in {band.seed, accept.seed}

    def test_every_model_has_a_pinned_source(self):
        assert set(export_vae.VAE_SERIES) <= set(SOURCES)

    def test_only_ti2v_has_an_i8_series_and_it_is_the_dit_series(self):
        """ADR 0133 決定 7: Wan2.2 の VAE の i8 は DiT の i8 容器と同じ系列の根へ書く（部品
        ディレクトリが違うので `transformer/` と重ならない）。"""
        assert export_vae.VAE_SERIES["t2v-1.3b"].i8_series is None
        assert export_vae.VAE_SERIES["ti2v-5b"].i8_series == "wan2.2-ti2v-5b-i8-dyn"
        assert export_vae.TI2V_I8_SERIES_NAME == "wan2.2-ti2v-5b-i8-dyn"


class TestSeriesConfigGate:
    """上流 config の `patch_size` と空間の圧縮が系列の前提と合わなければ書かない。"""

    @staticmethod
    def _vae(patch_size: int | None, scale_factor_spatial: int) -> SimpleNamespace:
        return SimpleNamespace(
            config=SimpleNamespace(patch_size=patch_size, scale_factor_spatial=scale_factor_spatial)
        )

    def test_the_configs_of_both_generations_pass(self):
        export_vae.assert_series_config(self._vae(None, 8), export_vae.VAE_SERIES["t2v-1.3b"])
        export_vae.assert_series_config(self._vae(2, 16), export_vae.VAE_SERIES["ti2v-5b"])

    def test_a_vae_of_the_other_generation_is_refused(self):
        with pytest.raises(export_vae.ChunkGraphError, match="patch_size"):
            export_vae.assert_series_config(self._vae(2, 16), export_vae.VAE_SERIES["t2v-1.3b"])

    def test_a_spatial_compression_off_the_graph_ratio_is_refused(self):
        """2.2 で圧縮が 8 のままだと、ホストの重なりとフレームの辺の前提が崩れる。"""
        with pytest.raises(export_vae.ChunkGraphError, match="scale_factor_spatial"):
            export_vae.assert_series_config(self._vae(2, 8), export_vae.VAE_SERIES["ti2v-5b"])


@pytest.fixture(scope="module")
def written(ti2v_synthetic_vae, tmp_path_factory) -> tuple[dict, Path, object]:
    """合成の 2.2 VAE で書いた要約・系列の根・書いた VAE（`load_vae(round_f16=True)` と同じく
    f16 へ丸めた写し）。"""
    vae = copy.deepcopy(ti2v_synthetic_vae)
    round_weights_to_f16(vae)
    root = tmp_path_factory.mktemp("ti2v") / "series"
    source = Provenance(license="apache-2.0", upstream_revision="0" * 40)
    summary = export_vae.emit_targets(
        export_vae.TARGETS,
        vae,
        TI2V_TILE,
        root,
        source,
        cases=export_vae.TI2V_FIXTURE_CASES,
        patch_size=2,
    )
    return summary, root, vae


class TestTi2vWriter:
    """合成の 2.2 VAE で書き手を通しで回す（容器 2 本 + フィクスチャ 3 本を一組で据える）。"""

    def test_the_summary_records_the_ti2v_graphs(self, written):
        summary, _, _ = written
        side = 8 * TI2V_TILE

        first, following = summary["graphs"]
        assert (first["caches"], following["caches"]) == (30, 32)
        assert first["frame_shape"] == [12, 1, side, side]
        assert following["frame_shape"] == [12, 4, side, side]
        for graph in (first, following):
            assert set(graph["op_counts"]) == export_vae.EXPECTED_OPS
            assert [shape for _, shape in graph["cache_outputs"]] == [
                shape for _, shape in graph["inputs"][1:]
            ]

    def test_the_fixtures_hold_the_reference_before_unpatchify(self, written):
        from safetensors import safe_open

        _, root, vae = written
        side = 8 * TI2V_TILE
        for case in export_vae.TI2V_FIXTURE_CASES:
            with safe_open(str(export_vae.fixture_path(root, case)), framework="pt") as fixture:
                metadata = fixture.metadata()
                latents = fixture.get_tensor("latents")
                frames = fixture.get_tensor("frames")

            assert metadata == {
                "seed": str(case.seed),
                "chunks": str(case.chunks),
                "tile": str(TI2V_TILE),
                "role": case.role,
                "weights": "f16-rounded",
                "reference": export_vae.FIXTURE_REFERENCE_BEFORE_UNPATCHIFY,
            }
            assert latents.shape == (48, case.chunks, TI2V_TILE, TI2V_TILE)
            assert frames.shape == (12, 1 + 4 * (case.chunks - 1), side, side)
            with torch.no_grad():
                want = vae_patch.reference_decode_unclamped(vae, latents[None])
            assert torch.equal(frames, want[0])

    def test_the_whole_set_is_published(self, written):
        _, root, _ = written

        assert sorted(path.name for path in root.iterdir()) == sorted(
            [*export_vae.TARGETS]
            + [export_vae.fixture_path(root, case).name for case in export_vae.TI2V_FIXTURE_CASES]
        )
        for target in export_vae.TARGETS:
            assert container_parts(root / target / export_vae.MODEL_FILE)


class TestCommandLine:
    """`--model` が系列・潜在タイル・ケース・patch の倍率を一括で選ぶ（実重みは読まない）。"""

    #: `fake_quant_i8` の身代わりが返す scale 台帳（中身は見ない — 渡り先だけを見る）。
    SCALES: ClassVar[dict[str, torch.Tensor]] = {"decoder.conv_in.weight": torch.ones(1)}

    @pytest.fixture
    def storage(self) -> dict:
        """書き手へ渡った格納（`weight_dtype` / `weight_scales`）と、i8 へ丸めた VAE。"""
        return {}

    @pytest.fixture
    def emitted(self, monkeypatch, storage) -> dict:
        captured: dict = {}

        def emit_targets(
            targets, vae, tile, out_root, source, *, cases, patch_size, weight_dtype, weight_scales
        ):
            captured.update(tile=tile, out=out_root, cases=cases, patch_size=patch_size)
            storage.update(dtype=weight_dtype, scales=weight_scales)
            return {}

        def fake_quant_i8(vae):
            storage["quantized"] = vae
            return self.SCALES

        monkeypatch.setattr(export_vae, "provenance", lambda model: model)
        monkeypatch.setattr(export_vae, "load_vae", lambda model, *, round_f16: model)
        monkeypatch.setattr(export_vae, "fake_quant_i8", fake_quant_i8)
        monkeypatch.setattr(export_vae, "assert_series_config", lambda _vae, _series: None)
        monkeypatch.setattr(export_vae, "weight_summary", lambda _vae: {})
        monkeypatch.setattr(export_vae, "emit_targets", emit_targets)
        return captured

    def test_the_default_model_writes_the_wan21_series(self, emitted, capsys):
        export_vae.main([])

        assert emitted == {
            "tile": 32,
            "out": SERIES_ROOT / "wan2.1-t2v-1.3b-f16-dyn",
            "cases": export_vae.FIXTURE_CASES,
            "patch_size": None,
        }

    def test_the_ti2v_model_writes_its_own_series(self, emitted, capsys):
        export_vae.main(["--model", "ti2v-5b"])

        assert emitted == {
            "tile": 16,
            "out": SERIES_ROOT / "wan2.2-ti2v-5b-f16-dyn",
            "cases": export_vae.TI2V_FIXTURE_CASES,
            "patch_size": 2,
        }

    def test_an_explicit_out_and_no_fixtures(self, emitted, tmp_path, capsys):
        export_vae.main(["--model", "ti2v-5b", "--out", str(tmp_path), "--no-fixtures"])

        assert (emitted["out"], emitted["cases"], emitted["tile"]) == (tmp_path, (), 16)

    def test_the_default_storage_is_f16_without_quantizing(self, emitted, storage, capsys):
        export_vae.main(["--model", "ti2v-5b"])

        assert storage == {"dtype": "f16", "scales": None}

    def test_i8_writes_the_ti2v_i8_series_without_fixtures(self, emitted, storage, capsys):
        """ADR 0133 決定 7: i8 の回はモデルの i8 系列の根へ・フィクスチャ無し・台帳を渡す。"""
        export_vae.main(["--model", "ti2v-5b", "--dtype", "i8"])

        assert emitted == {
            "tile": 16,
            "out": SERIES_ROOT / "wan2.2-ti2v-5b-i8-dyn",
            "cases": (),
            "patch_size": 2,
        }
        assert storage == {"dtype": "i8", "scales": self.SCALES, "quantized": "ti2v-5b"}

    def test_i8_on_a_model_without_an_i8_series_is_refused(self, emitted, storage, capsys):
        with pytest.raises(SystemExit):
            export_vae.main(["--model", "t2v-1.3b", "--dtype", "i8"])

        assert "i8 系列を持たない" in capsys.readouterr().err
        assert (emitted, storage) == ({}, {})


@pytest.fixture(scope="module")
def written_i8(ti2v_synthetic_vae, tmp_path_factory) -> tuple[dict, Path]:
    """合成の 2.2 VAE を `main` の i8 の回と同じ順（f16 へ丸め → i8 へ丸め）で書いた要約と
    系列の根。"""
    vae = copy.deepcopy(ti2v_synthetic_vae)
    round_weights_to_f16(vae)
    scales = export_vae.fake_quant_i8(vae)
    root = tmp_path_factory.mktemp("ti2v-i8") / "series"
    source = Provenance(license="apache-2.0", upstream_revision="0" * 40)
    summary = export_vae.emit_targets(
        export_vae.TARGETS,
        vae,
        TI2V_TILE,
        root,
        source,
        cases=(),
        patch_size=2,
        weight_dtype="i8",
        weight_scales=scales,
    )
    return summary, root


def _container(root: Path, target: str) -> tuple[IrGraph, BoundGraph]:
    """据わった容器のグラフ（IR）と束縛表（格納の正本）。"""
    verified = verify_container(container_parts(root / target / export_vae.MODEL_FILE))
    return ir_graph_from_container(verified.read, target), verified.graphs[target]


def _weight_slots(graph: IrGraph) -> dict[str, str]:
    """重みスロットに入る initializer → 消費する op。"""
    return {
        node.ins[WEIGHT_SLOTS[node.op]]: node.op
        for node in graph.nodes
        if node.op in WEIGHT_SLOTS and node.ins[WEIGHT_SLOTS[node.op]] in graph.initializers
    }


class TestTi2vI8Writer:
    """`--dtype i8` の書き手（ADR 0133 決定 7）— 合成の 2.2 VAE の f16 の回（{@link written}）と
    比べる。"""

    @pytest.mark.parametrize("target", export_vae.TARGETS)
    def test_every_convolution_weight_is_int8_with_a_scale_per_output_channel(
        self, written_i8, target: str
    ):
        _, root = written_i8
        graph, bound = _container(root, target)
        slots = _weight_slots(graph)

        # 空振りしない: decoder の conv3d と、up sampler / mid の attention の conv2d がある。
        assert {"conv3d", "conv2d"} <= set(slots.values())
        assert set(slots.values()) <= {"conv3d", "conv2d", "linear"}
        for name in slots:
            supply = bound.supplies[name]
            out_channels = int(graph.values[name].shape[0])
            assert supply.encoding.codec == "int8-sym", name
            assert supply.encoding.row_axis == 0, name
            assert supply.scale is not None, name
            assert supply.scale.payload_bytes == 4 * out_channels, name

    @pytest.mark.parametrize("target", export_vae.TARGETS)
    def test_nothing_else_changes_from_the_f16_graph(self, written, written_i8, target: str):
        """ノード・入出力・値の宣言は f16 の回と同じで、格納が変わるのは重みスロットの
        initializer（f16 → int8-sym）だけ（bias・norm は f16 の回と同じ格納のまま）。"""
        _, f16_root, _ = written
        _, i8_root = written_i8
        f16_graph, f16_bound = _container(f16_root, target)
        i8_graph, i8_bound = _container(i8_root, target)
        slots = set(_weight_slots(i8_graph))

        assert i8_graph.nodes == f16_graph.nodes
        assert (i8_graph.inputs, i8_graph.outputs) == (f16_graph.inputs, f16_graph.outputs)
        assert i8_graph.values == f16_graph.values
        assert set(i8_graph.initializers) == set(f16_graph.initializers)
        codec = {name: supply.encoding.codec for name, supply in f16_bound.supplies.items()}
        assert set(i8_bound.supplies) == set(codec)
        for name, supply in i8_bound.supplies.items():
            expected = "int8-sym" if name in slots else codec[name]
            assert supply.encoding.codec == expected, name
        assert {codec[name] for name in slots} == {"f16"}
        assert {codec_entry(codec[name]).layout for name in set(codec) - slots} == {"f32"}

    def test_no_fixture_is_written_and_the_summary_names_the_storage(self, written, written_i8):
        summary, root = written_i8

        assert sorted(path.name for path in root.iterdir()) == sorted(export_vae.TARGETS)
        assert "fixtures" not in summary
        assert summary["dtype"] == "i8"
        assert written[0]["dtype"] == "f16"

    def test_a_fixture_with_i8_weights_is_refused_before_anything_is_written(self, tmp_path):
        """フィクスチャのメタは f16-rounded の参照を名乗る — i8 の重みで作った参照は据えない。"""
        root = tmp_path / "series"

        with pytest.raises(export_vae.ChunkGraphError, match="フィクスチャを書かない"):
            export_vae.emit_targets(
                export_vae.TARGETS,
                None,
                TI2V_TILE,
                root,
                None,
                cases=export_vae.TI2V_FIXTURE_CASES,
                patch_size=2,
                weight_dtype="i8",
                weight_scales={},
            )

        assert not root.exists()


class TestTi2vVerify:
    """`--verify` の 2.2 の経路（unpatchify・モデルの受け渡し・config の門）を合成の VAE で通す。

    実寸（5B・タイル 16・9 chunk）は
    `python -m wan.export_vae --model ti2v-5b --verify --chunks 9`。
    ここは経路だけを見る — 最終形の差は記録の値で、帯は `test_vae_patch_ti2v.py` が固定する。
    """

    CHUNKS = 3

    @pytest.fixture
    def verified(self, ti2v_synthetic_vae, monkeypatch, capsys) -> tuple[dict, list]:
        loaded: list = []

        def load_vae(model, *, round_f16):
            loaded.append((model, round_f16))
            return ti2v_synthetic_vae

        monkeypatch.setattr(export_vae, "load_vae", load_vae)
        argv = ["--model", "ti2v-5b", "--verify", "--tile", str(TI2V_TILE)]
        export_vae.main([*argv, "--chunks", str(self.CHUNKS)])
        return json.loads(capsys.readouterr().out), loaded

    def test_the_unrounded_ti2v_weights_are_read(self, verified):
        _, loaded = verified

        assert loaded == [("ti2v-5b", False)]

    def test_the_reference_is_compared_after_unpatchify(self, verified):
        summary, _ = verified
        side = 8 * TI2V_TILE

        assert summary["frames"] == [12, 1 + 4 * (self.CHUNKS - 1), side, side]
        assert summary["clamped_reference_equals_decode"] is True
        assert summary["normalized_cache_bit_exact"] is True


# ---- `--shared`（ADR 0134 決定 1 — 重みのバイトを持たない別タイルの組）-----------------------

#: 共有の組のタイル（貸し手の {@link TI2V_TILE} と別の辺 — 実寸の 16 → 32 と同じ 2 倍）。
SHARED_TILE = 8

_SOURCE = Provenance(license="apache-2.0", upstream_revision="0" * 40)


@pytest.fixture(scope="module")
def written_shared(
    ti2v_synthetic_vae, written, written_i8, tmp_path_factory
) -> dict[str, tuple[dict, Path, Path]]:
    """格納ラベル → (要約, 系列の根, 書く前の貸し手の系列の根)。f16 / i8 の貸し手の組
    （{@link written} / {@link written_i8}）の写しへ、共有の組をタイル {@link SHARED_TILE} で書く。

    VAE は丸めない（`main` の `--shared` と同じ — バイトを書かないので丸めも量子化も要らない）。
    """
    import shutil

    out = {}
    for dtype, lender_root in (("f16", written[1]), ("i8", written_i8[1])):
        root = tmp_path_factory.mktemp(f"shared-{dtype}") / "series"
        shutil.copytree(lender_root, root)
        summary = export_vae.emit_shared_targets(
            export_vae.TARGETS,
            ti2v_synthetic_vae,
            SHARED_TILE,
            root,
            _SOURCE,
            lender_tile=TI2V_TILE,
            weight_dtype=dtype,
        )
        out[dtype] = (summary, root, lender_root)
    return out


def _shared_name(target: str) -> str:
    return export_vae.shared_target(target, SHARED_TILE)


@pytest.mark.parametrize("dtype", export_vae.WEIGHT_DTYPES)
class TestSharedWriter:
    """合成の 2.2 VAE の 16 の組（ここではタイル 4）を貸し手に、タイル 8 の組を書く。"""

    @pytest.mark.parametrize("target", export_vae.TARGETS)
    def test_every_initializer_borrows_a_lender_tensor_of_the_same_key_and_shape(
        self, written_shared, dtype: str, target: str
    ):
        _, root, _ = written_shared[dtype]
        graph, _ = _container(root, _shared_name(target))
        lender, lender_bound = _container(root, target)

        assert graph.initializers
        assert all(init.is_shared for init in graph.initializers.values())
        # IR v2 では initializer の名前がテンソルキー — 共有宣言の名前が貸し手の実体を指す。
        assert set(graph.initializers) <= set(lender_bound.supplies)
        for name in graph.initializers:
            assert graph.values[name] == lender.values[name], name

    @pytest.mark.parametrize("target", export_vae.TARGETS)
    def test_no_weight_byte_is_written(self, written_shared, dtype: str, target: str):
        """束縛表が空で、part 0（2 文書）の外は 0 バイト。"""
        _, root, _ = written_shared[dtype]
        _, bound = _container(root, _shared_name(target))
        parts = container_parts(root / _shared_name(target) / export_vae.MODEL_FILE)

        assert bound.supplies == {}
        assert parts[0].stat().st_size > 0
        assert [part.stat().st_size for part in parts[1:]] == [0] * (len(parts) - 1)

    @pytest.mark.parametrize("target", export_vae.TARGETS)
    def test_the_chunk_graph_contract_holds_at_the_shared_tile(
        self, written_shared, ti2v_synthetic_vae, dtype: str, target: str
    ):
        _, root, _ = written_shared[dtype]
        graph, _ = _container(root, _shared_name(target))
        module = vae_patch.WanVaeChunkDecoder(
            ti2v_synthetic_vae, first=target == export_vae.TARGET_FIRST
        ).eval()

        export_vae.assert_chunk_graph(graph, module, SHARED_TILE)
        assert graph.inputs[0].shape == [48, 1, SHARED_TILE, SHARED_TILE]

    def test_the_summary_names_the_lender_and_the_tile(self, written_shared, dtype: str):
        summary, root, _ = written_shared[dtype]

        assert (summary["series"], summary["dtype"], summary["tile"]) == (
            str(root),
            dtype,
            SHARED_TILE,
        )
        assert summary["lender"] == {"tile": TI2V_TILE, "targets": list(export_vae.TARGETS)}
        assert "fixtures" not in summary
        for graph, target in zip(summary["graphs"], export_vae.TARGETS, strict=True):
            assert graph["target"] == _shared_name(target) == f"{target}_{SHARED_TILE}"
            assert graph["tile"] == SHARED_TILE
            assert graph["lender"] == str(root / target / export_vae.MODEL_FILE)
            assert (graph["compressed_bytes"], graph["plain_bytes"]) == (0, 0)

    def test_only_the_shared_pair_is_added_and_the_lenders_are_unchanged(
        self, written_shared, dtype: str
    ):
        """フィクスチャは書かない・貸し手の組はバイト同一（ADR 0134 決定 7 の 1）。"""
        _, root, lender_root = written_shared[dtype]

        before = sorted(path.name for path in lender_root.iterdir())
        assert sorted(path.name for path in root.iterdir()) == sorted(
            [*before, *(_shared_name(target) for target in export_vae.TARGETS)]
        )
        for target in export_vae.TARGETS:
            lent = container_parts(lender_root / target / export_vae.MODEL_FILE)
            kept = container_parts(root / target / export_vae.MODEL_FILE)
            assert [part.read_bytes() for part in kept] == [part.read_bytes() for part in lent]

    def test_each_declaration_carries_the_lender_storage(
        self, written_shared, ti2v_synthetic_vae, dtype: str
    ):
        """共有宣言の格納は貸し手の束縛表の格納（容器には残らないので、置き換えの関数で見る）。
        i8 の貸し手では畳み込みの重みが i8・それ以外は f32。"""
        _, root, _ = written_shared[dtype]
        target = export_vae.TARGET_NEXT
        path = root / target / export_vae.MODEL_FILE
        lender = export_vae.read_lender(path, target, TI2V_TILE, dtype)
        module = vae_patch.WanVaeChunkDecoder(ti2v_synthetic_vae, first=False).eval()
        with torch.no_grad():
            traced, _ = export_module(
                module,
                export_vae.example_inputs(module, SHARED_TILE),
                preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
            )

        shared = export_vae.share_lender_weights(traced, lender, path)

        storage = {init.shared.tensor: init.storage for init in shared.initializers.values()}
        assert storage == {
            key: IrStorage(dtype=init.storage.dtype) for key, init in lender.initializers.items()
        }
        assert {value.dtype for value in storage.values()} == {dtype, "f32"}


class TestSharedRefusals:
    """共有の組の門 — 外れたら何も据えない。"""

    def test_no_lender_is_refused(self, ti2v_synthetic_vae, tmp_path):
        root = tmp_path / "series"

        with pytest.raises(export_vae.ChunkGraphError, match=r"貸し手の容器 .* が無い"):
            export_vae.emit_shared_targets(
                export_vae.TARGETS,
                ti2v_synthetic_vae,
                SHARED_TILE,
                root,
                _SOURCE,
                lender_tile=TI2V_TILE,
            )
        assert not root.exists() or not list(root.iterdir())

    def test_fixtures_are_refused(self, ti2v_synthetic_vae, tmp_path):
        with pytest.raises(export_vae.ChunkGraphError, match="フィクスチャを書かない"):
            export_vae.emit_shared_targets(
                export_vae.TARGETS,
                ti2v_synthetic_vae,
                SHARED_TILE,
                tmp_path,
                _SOURCE,
                lender_tile=TI2V_TILE,
                cases=export_vae.TI2V_FIXTURE_CASES,
            )
        assert list(tmp_path.iterdir()) == []

    def test_a_lender_of_the_other_storage_is_refused(self, written, ti2v_synthetic_vae, tmp_path):
        """f16 系列の根の貸し手を i8 の組として借りる取り違え。"""
        import shutil

        root = tmp_path / "series"
        shutil.copytree(written[1], root)
        with pytest.raises(
            export_vae.ChunkGraphError, match=r"圧縮格納 \['f16'\] が i8 の系列でない"
        ):
            export_vae.emit_shared_targets(
                export_vae.TARGETS,
                ti2v_synthetic_vae,
                SHARED_TILE,
                root,
                _SOURCE,
                lender_tile=TI2V_TILE,
                weight_dtype="i8",
            )
        assert not [path for path in root.iterdir() if path.name.endswith(f"_{SHARED_TILE}")]

    def test_a_lender_of_another_tile_is_refused(self, written, ti2v_synthetic_vae, tmp_path):
        """貸し手のタイルは系列の表のタイル（別のタイルで書いた組は貸し手にしない）。"""
        import shutil

        root = tmp_path / "series"
        shutil.copytree(written[1], root)
        with pytest.raises(export_vae.ChunkGraphError, match="タイル 16 の 'latent' でない"):
            export_vae.emit_shared_targets(
                export_vae.TARGETS,
                ti2v_synthetic_vae,
                SHARED_TILE,
                root,
                _SOURCE,
                lender_tile=16,
            )


class TestSharedCommandLine:
    """`--shared` の CLI（実重みは読まない — 書き手は差し替える）。"""

    @pytest.fixture
    def emitted(self, monkeypatch) -> dict:
        captured: dict = {}

        def emit_shared_targets(targets, vae, tile, out_root, source, *, lender_tile, weight_dtype):
            captured.update(
                targets=targets, tile=tile, out=out_root, lender=lender_tile, dtype=weight_dtype
            )
            return {}

        def load_vae(model, *, round_f16):
            captured["round_f16"] = round_f16
            return model

        def refuse(*_args, **_kwargs):
            raise AssertionError("--shared は既定の書き手・量子化を通らない")

        monkeypatch.setattr(export_vae, "provenance", lambda model: model)
        monkeypatch.setattr(export_vae, "load_vae", load_vae)
        monkeypatch.setattr(export_vae, "assert_series_config", lambda _vae, _series: None)
        monkeypatch.setattr(export_vae, "fake_quant_i8", refuse)
        monkeypatch.setattr(export_vae, "emit_targets", refuse)
        monkeypatch.setattr(export_vae, "emit_shared_targets", emit_shared_targets)
        return captured

    def test_f16_writes_the_32_pair_beside_the_f16_lender(self, emitted, capsys):
        export_vae.main(["--model", "ti2v-5b", "--tile", "32", "--shared"])

        assert emitted == {
            "targets": export_vae.TARGETS,
            "tile": 32,
            "out": SERIES_ROOT / "wan2.2-ti2v-5b-f16-dyn",
            "lender": 16,
            "dtype": "f16",
            "round_f16": False,
        }

    def test_i8_writes_the_32_pair_beside_the_i8_lender_without_quantizing(self, emitted, capsys):
        export_vae.main(["--model", "ti2v-5b", "--dtype", "i8", "--tile", "32", "--shared"])

        assert (emitted["out"], emitted["dtype"], emitted["lender"]) == (
            SERIES_ROOT / "wan2.2-ti2v-5b-i8-dyn",
            "i8",
            16,
        )

    def test_shared_without_a_tile_is_a_parser_error(self, emitted, capsys):
        with pytest.raises(SystemExit):
            export_vae.main(["--model", "ti2v-5b", "--shared"])

        assert "--tile を要する" in capsys.readouterr().err
        assert emitted == {}
