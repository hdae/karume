"""`wan/export_vae_encoder.py` の約束事 — encoder の 3 グラフの IR と golden の書き手
（ADR 0121 段 9a）。

グラフは合成の 2.2 VAE（conftest の `ti2v_synthetic_vae` — 5B と同じ構造でチャネルだけを縮めた
もの）で export する（グラフの構造は寸法とチャネル数に依らず、宣言 shape のチャネルだけが変わる）。
実寸の要約は `python -m wan.export_vae_encoder` が出す。

見るもの:

- 各グラフの op の集合・記号（pre / post は `[h, w]`・attn は `[S]`）・入出力の宣言 shape・値の
  rank（4 以下。rank 5 は conv3d の重みだけ）・conv / pad / attention の本数
- 検査関数（{@link wan.export_vae_encoder.assert_encoder_graph}）自身が、語彙外の op・rank 6・
  記号の違い・入力の形の違いを落とすこと
- golden のケースの表（3 寸法 × 決定用 2 枚 + 受入れ 1 枚・stretch は前処理だけ）
- 書き手が 3 グラフと golden を一組で据えること（合成の VAE と合成の画像で）・来歴の sha256 が
  違う画像では 1 バイトも書かないこと
- 系列に据わった golden（git 追跡外 — 無ければ SKIP）の中身が、今の Pillow で作り直した前処理の鎖と
  正規化にビット一致すること（版の更新で黙ってずれた golden を赤にする）
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from _shared.paths import SERIES_ROOT
from karume.container import Provenance
from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION
from karume.ir import IrGraph, IrNode, IrValue
from karume.pipeline import export_module
from karume.quantize import round_weights_to_f16
from wan import export_vae_encoder, i2v_preprocess_ref, vae_encoder_patch
from wan.export_vae_encoder import (
    TARGET_ATTENTION,
    TARGET_POST,
    TARGET_PRE,
    TARGETS,
    EncoderGraphError,
    GoldenImage,
)

#: 合成の VAE の mid のチャネル（`base_dim 8 × dim_mult[-1] 4`）。
SYNTHETIC_HIDDEN = 32


@pytest.fixture(scope="module")
def exported(ti2v_synthetic_vae) -> dict[str, IrGraph]:
    graphs = {}
    for target in TARGETS:
        module = export_vae_encoder.graph_module(target, ti2v_synthetic_vae)
        with torch.no_grad():
            graph, _ = export_module(
                module,
                export_vae_encoder.example_inputs(target, ti2v_synthetic_vae),
                dynamic_shapes=export_vae_encoder.dynamic_shapes(target),
                symbol_names=export_vae_encoder.symbol_names(target),
                preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
            )
        graphs[target] = graph
    return graphs


class TestEncoderGraphs:
    @pytest.mark.parametrize("target", TARGETS)
    def test_the_graph_keeps_the_contract(self, exported, ti2v_synthetic_vae, target: str):
        export_vae_encoder.assert_encoder_graph(exported[target], target, ti2v_synthetic_vae)

    @pytest.mark.parametrize("target", TARGETS)
    def test_the_op_set_is_exactly_the_expected_one(self, exported, target: str):
        assert set(exported[target].required_ops) == export_vae_encoder.EXPECTED_OPS[target]

    def test_the_symbols_and_shapes_are_the_declared_ones(self, exported):
        """記号は潜在の h, w（pre の入力は 8h / 8w）と系列長 S — 積 h·w はどの値にも現れない。"""
        io = {
            target: (
                graph.symbols,
                [(entry.name, entry.shape) for entry in graph.inputs],
                graph.values[graph.outputs[0]].shape,
            )
            for target, graph in exported.items()
        }

        assert io == {
            TARGET_PRE: (["h", "w"], [("image", [12, 1, "8h", "8w"])], [32, 1, "h", "w"]),
            TARGET_ATTENTION: (["S"], [("tokens", [32, "S"])], [32, "S"]),
            TARGET_POST: (["h", "w"], [("hidden", [32, 1, "h", "w"])], [48, 1, "h", "w"]),
        }

    def test_only_conv3d_weights_are_rank5(self, exported):
        for graph in exported.values():
            rank5 = {name for name, value in graph.values.items() if len(value.shape) >= 5}
            weights = {node.ins[1] for node in graph.nodes if node.op == "conv3d"}

            assert rank5 == weights
            assert rank5 <= set(graph.initializers)

    def test_the_node_counts_follow_the_structure(self, exported):
        """pre: conv3d = conv_in 1 + down の resnets 16 + 1×1×1 のショートカット 2 + mid 2・
        down の Conv2d 3（pad は各 2 本）・時間倍率 2 のショートカット 2 本が cat / slice を持つ。
        post: conv3d = mid 2 + conv_out + quant_conv・mu の切り出し 1。"""
        counts = {
            target: Counter(node.op for node in graph.nodes) for target, graph in exported.items()
        }

        assert (counts[TARGET_PRE]["conv3d"], counts[TARGET_PRE]["conv2d"]) == (21, 3)
        assert (counts[TARGET_PRE]["pad"], counts[TARGET_PRE]["cat"]) == (6, 2)
        assert (counts[TARGET_ATTENTION]["attention"], counts[TARGET_ATTENTION]["linear"]) == (1, 2)
        assert (counts[TARGET_POST]["conv3d"], counts[TARGET_POST]["slice"]) == (4, 1)


class TestTheCheckItself:
    """検査が緩むと書き直しの漏れや記号の取り違えが素通りする — 故障注入で赤を確かめる。"""

    def test_an_op_outside_the_set_is_caught(self, exported, ti2v_synthetic_vae):
        broken = copy.deepcopy(exported[TARGET_POST])
        broken.nodes.append(IrNode(op="pad", ins=[broken.outputs[0]], outs=["padded"], attrs={}))

        with pytest.raises(EncoderGraphError, match="pad"):
            export_vae_encoder.assert_encoder_graph(broken, TARGET_POST, ti2v_synthetic_vae)

    def test_a_rank6_value_is_caught(self, exported, ti2v_synthetic_vae):
        broken = copy.deepcopy(exported[TARGET_PRE])
        name = next(name for name in broken.values if name not in broken.initializers)
        broken.values[name] = IrValue(dtype="f32", shape=[1, 1, 1, 1, 1, 1])

        with pytest.raises(EncoderGraphError, match="rank 6"):
            export_vae_encoder.assert_encoder_graph(broken, TARGET_PRE, ti2v_synthetic_vae)

    def test_swapped_symbols_are_caught(self, exported, ti2v_synthetic_vae):
        broken = copy.deepcopy(exported[TARGET_POST])
        broken.symbols = ["w", "h"]

        with pytest.raises(EncoderGraphError, match="記号"):
            export_vae_encoder.assert_encoder_graph(broken, TARGET_POST, ti2v_synthetic_vae)

    def test_an_input_of_another_shape_is_caught(self, exported, ti2v_synthetic_vae):
        broken = copy.deepcopy(exported[TARGET_ATTENTION])
        broken.inputs[0] = dataclasses.replace(broken.inputs[0], shape=[SYNTHETIC_HIDDEN, 3520])

        with pytest.raises(EncoderGraphError, match="入力"):
            export_vae_encoder.assert_encoder_graph(broken, TARGET_ATTENTION, ti2v_synthetic_vae)


class TestGoldenCases:
    def test_each_size_has_two_band_images_and_one_accept_image(self):
        crops = [case for case in export_vae_encoder.GOLDEN_CASES if case.fit == "crop"]
        by_size: dict[tuple[int, int], list[tuple[str, str]]] = {}
        for case in crops:
            by_size.setdefault((case.width, case.height), []).append((case.image, case.role))

        assert by_size == {
            size: [("boxing-cats", "band"), ("cat-dog-baking", "band"), ("ferret", "accept")]
            for size in ((1280, 704), (704, 1280), (256, 160))
        }

    def test_the_stretch_cases_hold_only_the_preprocessing(self):
        stretches = [case for case in export_vae_encoder.GOLDEN_CASES if case.fit == "stretch"]

        assert [(case.image, case.width, case.height, case.role) for case in stretches] == [
            (image, 1280, 704, "host") for image in ("boxing-cats", "cat-dog-baking", "ferret")
        ]
        assert not any(case.encodes for case in stretches)

    def test_the_names_are_unique_and_the_sizes_are_multiples_of_16(self):
        cases = export_vae_encoder.GOLDEN_CASES

        assert len({case.name for case in cases}) == len(cases) == 12
        assert all(case.width % 16 == 0 and case.height % 16 == 0 for case in cases)


# ---- 書き手（合成の VAE と合成の画像）-------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def synthetic_images(tmp_path, monkeypatch) -> tuple[GoldenImage, ...]:
    """テスト画像の代わりの合成の 832×480 PNG（sha256 を来歴として差し替える）。"""
    from PIL import Image

    directory = tmp_path / "inputs"
    directory.mkdir()
    rng = np.random.default_rng(0)
    images = []
    for image in export_vae_encoder.IMAGES:
        path = directory / f"{image.name}-832x480.png"
        Image.fromarray(rng.integers(0, 256, size=(480, 832, 3), dtype=np.uint8)).save(path)
        images.append(GoldenImage(image.name, _sha256(path)))
    monkeypatch.setattr(export_vae_encoder, "IMAGE_DIR", directory)
    monkeypatch.setattr(export_vae_encoder, "IMAGES", tuple(images))
    return tuple(images)


@pytest.fixture(scope="module")
def rounded_vae(ti2v_synthetic_vae):
    """f16 表現可能値へ丸めた合成の VAE（f16 席の書き出しの前提 — fake-quant）。"""
    vae = copy.deepcopy(ti2v_synthetic_vae)
    round_weights_to_f16(vae)
    return vae


SOURCE = Provenance(license="apache-2.0")


class TestTheSeriesIsPublishedTogether:
    def test_three_graphs_and_every_golden_are_written(
        self, rounded_vae, synthetic_images, tmp_path
    ):
        out_root = tmp_path / "series"

        summary = export_vae_encoder.emit_series(
            TARGETS, rounded_vae, out_root, SOURCE, cases=export_vae_encoder.GOLDEN_CASES
        )

        assert [graph["target"] for graph in summary["graphs"]] == list(TARGETS)
        assert sorted(path.name for path in out_root.iterdir()) == sorted(
            [*TARGETS]
            + [f"vae_encoder.{case.name}.safetensors" for case in export_vae_encoder.GOLDEN_CASES]
        )
        assert all(
            golden["eager_ratio"] < 1e-5 for golden in summary["goldens"] if golden["fit"] == "crop"
        )

    def test_a_golden_holds_the_chain_from_the_source_to_the_latent(
        self, rounded_vae, synthetic_images, tmp_path
    ):
        from safetensors import safe_open

        case = next(
            case
            for case in export_vae_encoder.GOLDEN_CASES
            if (case.image, case.width, case.fit) == ("ferret", 704, "crop")
        )
        out_root = tmp_path / "series"
        export_vae_encoder.emit_series((), rounded_vae, out_root, SOURCE, cases=(case,))

        with safe_open(export_vae_encoder.golden_path(out_root, case), "pt") as golden:
            metadata = golden.metadata()
            tensors = {name: golden.get_tensor(name) for name in golden.keys()}  # noqa: SIM118

        assert {name: (tensor.dtype, tuple(tensor.shape)) for name, tensor in tensors.items()} == {
            "source": (torch.uint8, (480, 832, 3)),
            "rgb8": (torch.uint8, (1280, 704, 3)),
            "encoder_input": (torch.float32, (12, 1, 640, 352)),
            "mu": (torch.float32, (48, 1, 80, 44)),
            "latent": (torch.float32, (48, 1, 80, 44)),
        }
        assert metadata["image_sha256"] == synthetic_images[2].sha256
        assert (metadata["role"], metadata["fit"], metadata["weights"]) == (
            "accept",
            "crop",
            "f16-rounded",
        )
        assert metadata["version_pillow"] == export_vae_encoder.library_versions()["pillow"]

    def test_an_image_with_another_sha256_writes_nothing(
        self, rounded_vae, synthetic_images, tmp_path, monkeypatch
    ):
        stale = (GoldenImage(synthetic_images[0].name, "0" * 64), *synthetic_images[1:])
        monkeypatch.setattr(export_vae_encoder, "IMAGES", stale)
        out_root = tmp_path / "series"

        with pytest.raises(EncoderGraphError, match="sha256"):
            export_vae_encoder.emit_series(
                TARGETS, rounded_vae, out_root, SOURCE, cases=export_vae_encoder.GOLDEN_CASES
            )

        assert not out_root.exists() or not any(out_root.iterdir())

    def test_a_failure_in_a_golden_leaves_the_graphs_unwritten(
        self, rounded_vae, synthetic_images, tmp_path, monkeypatch
    ):
        def fails(*args, **kwargs):
            raise RuntimeError("golden の途中で落ちる")

        monkeypatch.setattr(export_vae_encoder, "golden_tensors", fails)
        out_root = tmp_path / "series"

        with pytest.raises(RuntimeError, match="golden"):
            export_vae_encoder.emit_series(
                TARGETS, rounded_vae, out_root, SOURCE, cases=export_vae_encoder.GOLDEN_CASES[:1]
            )

        assert not any(out_root.iterdir())

    def test_a_duplicated_target_is_refused_before_anything_is_written(self, rounded_vae, tmp_path):
        out_root = tmp_path / "series"

        with pytest.raises(EncoderGraphError, match="重複"):
            export_vae_encoder.emit_series(
                (TARGET_PRE, TARGET_PRE), rounded_vae, out_root, SOURCE, cases=()
            )

        assert not out_root.exists()


# ---- 系列に据わった golden（git 追跡外 — 無ければ SKIP）---------------------------------


@pytest.fixture(scope="module")
def series_goldens() -> dict[str, Path]:
    root = SERIES_ROOT / export_vae_encoder.SERIES_NAME
    paths = {
        case.name: export_vae_encoder.golden_path(root, case)
        for case in export_vae_encoder.GOLDEN_CASES
    }
    present = [name for name, path in paths.items() if path.is_file()]
    if not present:
        pytest.skip(f"{root} に encoder の golden が無い（生成: python -m wan.export_vae_encoder）")
    missing = sorted(set(paths) - set(present))
    assert not missing, (
        f"golden が一部だけある（欠け: {missing}）— 一組で据える書き手の外で消された"
    )
    return paths


@pytest.fixture(scope="module")
def latent_config(wan22_snapshot) -> SimpleNamespace:
    document = json.loads((wan22_snapshot / "vae" / "config.json").read_text(encoding="utf-8"))
    return SimpleNamespace(
        config=SimpleNamespace(
            z_dim=document["z_dim"],
            latents_mean=document["latents_mean"],
            latents_std=document["latents_std"],
            patch_size=document["patch_size"],
        )
    )


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view(torch.int32)


class TestTheGoldensOnDisk:
    @pytest.mark.parametrize("case", export_vae_encoder.GOLDEN_CASES, ids=lambda case: case.name)
    def test_the_golden_is_the_preprocessing_chain_of_todays_libraries(
        self, series_goldens, latent_config, case
    ):
        from PIL import Image
        from safetensors import safe_open

        with safe_open(series_goldens[case.name], "pt") as golden:
            metadata = golden.metadata()
            tensors = {name: golden.get_tensor(name) for name in golden.keys()}  # noqa: SIM118
        images = {image.name: image.sha256 for image in export_vae_encoder.IMAGES}

        assert metadata["image_sha256"] == images[case.image]
        assert (metadata["fit"], metadata["role"]) == (case.fit, case.role)
        source = Image.fromarray(tensors["source"].numpy())
        resized = i2v_preprocess_ref.preprocess(source, case.width, case.height, case.fit)
        assert torch.equal(tensors["rgb8"], i2v_preprocess_ref.rgb8(resized))
        if not case.encodes:
            assert set(tensors) == {"source", "rgb8"}
            return
        sample = i2v_preprocess_ref.to_signed_unit(resized)
        patch_size = latent_config.config.patch_size
        assert _bits(tensors["encoder_input"]).equal(
            _bits(i2v_preprocess_ref.encoder_input(sample, patch_size))
        )
        normalized = vae_encoder_patch.normalize_condition(latent_config, tensors["mu"][None])
        assert _bits(tensors["latent"]).equal(_bits(normalized[0]))
        assert torch.isfinite(tensors["mu"]).all()
