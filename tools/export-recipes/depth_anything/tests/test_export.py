"""`depth_anything/export.py` の台本レベルの約束事（実重み不要分）。

実重みを要する部分（IR の中身・パッチ前後の同値）は台本の実行と
`depth_anything/tests/test_patch.py` が受け持つ。ここで固定するのは、**実重みが無くても壊れうる**
約束:

- 系列の綴り（`outputs/series/<モデル名>/`）が `--model-dir` に追随する
- golden 4 ケースの性格（正規化済み・互いに違う・正方）
- sanity（{@link depth_anything.export._sanity}）が**恒真でない** — 一様な出力・入力非依存の
  出力・幾何を追えていない出力を実際に落とす
- 実画像の sanity（{@link depth_anything.export._real_sanity}）も同じく恒真でない — 近い領域と
  遠い領域が並ぶ / 逆転する出力を実際に落とす
- 前処理の正本の同定（{@link depth_anything.export.check_processor}）— **resample 3 =
  BICUBIC** をはじめ、1 欄ずらすだけで落ちること
- golden io の書き出し（{@link depth_anything.export._write_io}）— **tiny な合成重み**を
  実際に export して、キー規約と「出力は深度マップ 1 本」の門を通す（実重み不要）
- `--verify` が emit しないこと（同一プロセスではクラス属性の差し替えが参照を汚染する）
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from torch import nn

from depth_anything import export as da
from karume.paths import SERIES_ROOT
from karume.pipeline import export_to_file

#: 合成 golden の解像度（実重みの 518 は重いので、性格だけを見るここでは小さく取る）。
SMALL = 28

#: tiny な合成モデルの解像度（export を 1 本通すだけなので最小で足りる）。
TINY_SIZE = 6

#: tiny な合成重みへ流す 2 ケース（`_write_io` は名前と本数しか見ない）。
TINY_CASES = (
    ("disc", torch.linspace(-1.0, 1.0, 3 * TINY_SIZE * TINY_SIZE).reshape(1, 3, TINY_SIZE, -1)),
    ("ramp", torch.linspace(1.0, -1.0, 3 * TINY_SIZE * TINY_SIZE).reshape(1, 3, TINY_SIZE, -1)),
)


class TinyDepth(nn.Module):
    """{@link depth_anything.export.DepthMap} の最小の骨格（`[B,3,S,S] → [B,S,S]`）。"""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.conv(pixel_values).squeeze(1)


class TwoOutputDepth(TinyDepth):
    """出力が 2 本ある形（`_write_io` が拒否することの確認用）。"""

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        depth = super().forward(pixel_values)
        return depth, depth * 2.0


@pytest.fixture
def exported(tmp_path):
    """tiny なラッパを 1 本 export して `(wrapper, graph, out_dir)` を返す。"""
    torch.manual_seed(0)
    wrapper = TinyDepth()
    graph = export_to_file(wrapper, (TINY_CASES[0][1],), tmp_path / da.MODEL_FILE, symbol_names=())
    return wrapper, graph, tmp_path


def _depth_from(plane: torch.Tensor) -> torch.Tensor:
    """`[H, W]` の平面を深度マップ `[1, H, W]`（非負）に均す。"""
    return (plane - plane.min()).unsqueeze(0)


def _depths(ramp_weight: float, other_weight: float) -> dict[str, torch.Tensor]:
    """`ramp` だけが対角ランプに強く相関する 4 ケース分の合成出力。

    `other_weight` を上げると相関の最大が `ramp` から移る（sanity の判別が実際に効くことを
    示すための軸）。
    """
    plane = da.ramp_plane(SMALL)
    generator = torch.Generator().manual_seed(3)
    noise = torch.rand((4, SMALL, SMALL), generator=generator)
    return {
        "disc": _depth_from(noise[0]),
        "ramp": _depth_from(ramp_weight * plane + noise[1]),
        "checker": _depth_from(other_weight * plane + noise[2]),
        "noise": _depth_from(noise[3]),
    }


def _real_depths(near: float, far: float) -> dict[str, torch.Tensor]:
    """実画像 4 ケース分の合成深度（近側の矩形に `near`・遠側に `far` を置いた地図）。

    `near` と `far` を入れ替えると判別が裏返り、揃えると一様になる — sanity が恒真でないことは
    その 2 軸で見る。ケースごとに背景の定数をずらしてあるのは、`_assert_distinct`（同一の
    地図が並ぶのは入力が届いていない形）に引っかからないようにするため。
    """
    depths: dict[str, torch.Tensor] = {}
    for index, (name, regions) in enumerate(sorted(da.REAL_REGIONS.items())):
        depth = torch.full((1, SMALL, SMALL), 0.01 * (index + 1))
        for label, top, bottom, left, right in regions:
            value = near if label == regions[0][0] else far
            depth[
                :,
                int(top * SMALL) : int(bottom * SMALL),
                int(left * SMALL) : int(right * SMALL),
            ] = value
        depths[name] = depth
    return depths


class TestSeriesLayout:
    def test_the_series_name_is_the_model_directory_name(self) -> None:
        """系列名はモデルのディレクトリ名（小文字）— 解像度は軸ではないので綴らない。"""
        assert da.default_out_dir(da.DEFAULT_MODEL_DIR) == (
            SERIES_ROOT / "depth-anything-v2-small-hf"
        )

    def test_each_model_gets_its_own_series(self) -> None:
        """サイズ軸（Small / Base / Large）は綴りで分かれる — 同じ席だと先の資産が消える。"""
        names = {
            (da.MODELS_ROOT / f"Depth-Anything-V2-{size}-hf").name: da.default_out_dir(
                da.MODELS_ROOT / f"Depth-Anything-V2-{size}-hf"
            )
            for size in ("Small", "Base", "Large")
        }

        assert len(set(names.values())) == 3
        assert names["Depth-Anything-V2-Base-hf"] == SERIES_ROOT / "depth-anything-v2-base-hf"

    def test_the_graph_input_name_matches_the_wrapper_signature(self) -> None:
        """`INPUT_NAME` はラッパの引数名そのもの（export 後の突合が意味を持つ前提）。"""
        parameters = list(inspect.signature(da.DepthMap.forward).parameters)
        assert parameters == ["self", da.INPUT_NAME]


class TestCases:
    def test_the_four_cases_are_normalized_images_of_the_requested_size(self) -> None:
        """4 ケースとも `[1, 3, S, S]` で、逆正規化すると `[0, 1]` の画像に戻る。"""
        mean = torch.tensor(da.IMAGENET_MEAN).reshape(1, 3, 1, 1)
        std = torch.tensor(da.IMAGENET_STD).reshape(1, 3, 1, 1)
        cases = da.build_cases(SMALL)

        assert [name for name, _ in cases] == ["disc", "ramp", "checker", "noise"]
        for name, pixel_values in cases:
            assert tuple(pixel_values.shape) == (1, 3, SMALL, SMALL), name
            raw = pixel_values * std + mean
            assert float(raw.min()) >= -1e-6, name
            assert float(raw.max()) <= 1.0 + 1e-6, name

    def test_the_cases_differ_from_each_other(self) -> None:
        """同じ画像が 2 枚混じると、出力の相違検査（sanity）が恒真になる。"""
        images = [pixel_values for _name, pixel_values in da.build_cases(SMALL)]
        for index, left in enumerate(images):
            for right in images[index + 1 :]:
                assert not torch.equal(left, right)

    def test_the_noise_case_does_not_depend_on_the_global_seed(self) -> None:
        """`noise` は専用 Generator で作る（グローバル seed で golden が動かない）。"""
        torch.manual_seed(1)
        first = da.build_cases(SMALL)[3][1]
        torch.manual_seed(2)
        assert torch.equal(first, da.build_cases(SMALL)[3][1])


class TestSanity:
    def test_a_geometry_following_output_passes(self) -> None:
        report = da._sanity(_depths(ramp_weight=4.0, other_weight=0.0))

        assert report["ramp_correlation"]["ramp"] > report["ramp_correlation"]["checker"]
        assert set(report["depth_range"]) == {"disc", "ramp", "checker", "noise"}

    def test_identical_outputs_are_rejected(self) -> None:
        """入力が効いていない（2 ケースの出力が同一）形を落とす。"""
        depths = _depths(ramp_weight=4.0, other_weight=0.0)
        depths["checker"] = depths["noise"]

        with pytest.raises(AssertionError, match="入力が効いていない"):
            da._sanity(depths)

    def test_a_uniform_output_is_rejected(self) -> None:
        """一様に潰れた出力を落とす（min == max）。"""
        depths = _depths(ramp_weight=4.0, other_weight=0.0)
        depths["disc"] = torch.full((1, SMALL, SMALL), 2.5)

        with pytest.raises(AssertionError, match="一様"):
            da._sanity(depths)

    def test_a_negative_output_is_rejected(self) -> None:
        """head 末尾の ReLU と矛盾する負値を落とす（head の配線違い）。"""
        depths = _depths(ramp_weight=4.0, other_weight=0.0)
        depths["noise"] = depths["noise"] - 1.0

        with pytest.raises(AssertionError, match="ReLU"):
            da._sanity(depths)

    def test_an_output_that_does_not_follow_the_geometry_is_rejected(self) -> None:
        """単調な手掛かりの無いケースの方が強く相関する形を落とす（判別が効いている証拠）。"""
        with pytest.raises(AssertionError, match="幾何"):
            da._sanity(_depths(ramp_weight=0.0, other_weight=4.0))


class TestWriteIo:
    """tiny な合成重みでの golden 書き出し（実重み無しで通る唯一の export 経路）。"""

    def test_writes_one_file_per_case_with_the_declared_keys(self, exported) -> None:
        wrapper, graph, out_dir = exported

        written, depths = da._write_io(wrapper, graph, TINY_CASES, out_dir)

        assert written == [f"{da.IO_PREFIX}{name}{da.IO_SUFFIX}" for name, _ in TINY_CASES]
        tensors = load_file(str(out_dir / written[0]))
        assert set(tensors) == {f"{da.INPUT_PREFIX}{da.INPUT_NAME}", f"{da.OUTPUT_PREFIX}0"}
        assert tuple(tensors[f"{da.INPUT_PREFIX}{da.INPUT_NAME}"].shape) == (
            1,
            3,
            TINY_SIZE,
            TINY_SIZE,
        )
        assert tuple(depths["disc"].shape) == (1, TINY_SIZE, TINY_SIZE)

    def test_more_than_one_graph_output_fails_loudly(self, tmp_path) -> None:
        """MUST: 出力は深度マップ 1 本 — 2 本目が生えると io の位置規約が黙ってずれる。"""
        torch.manual_seed(0)
        wrapper = TwoOutputDepth()
        graph = export_to_file(
            wrapper, (TINY_CASES[0][1],), tmp_path / da.MODEL_FILE, symbol_names=()
        )

        with pytest.raises(AssertionError, match="深度マップは 1 本"):
            da._write_io(wrapper, graph, TINY_CASES, tmp_path)


class TestVerifyCli:
    def test_verify_does_not_emit(self, monkeypatch) -> None:
        """MUST: 同一プロセスで emit と併用しない（クラス差し替えが参照を汚染する）。"""
        seen: list[str] = []
        monkeypatch.setattr(
            da,
            "verify_patches",
            lambda _dir: (
                seen.append("verify")
                or [{"stage": "layout", "claim": "bit-exact", "bit_exact": True, "maxdiff": {}}]
            ),
        )
        monkeypatch.setattr(
            da, "export_series", lambda *_a, **_kw: pytest.fail("--verify で emit された")
        )

        da.main(["--verify"])

        assert seen == ["verify"]

    def test_without_verify_it_emits(self, monkeypatch) -> None:
        seen: list[str] = []
        monkeypatch.setattr(
            da, "export_series", lambda *_a, **_kw: seen.append("emit") or {"dir": "x"}
        )
        monkeypatch.setattr(
            da, "verify_patches", lambda *_a: pytest.fail("emit で --verify が走った")
        )

        da.main([])

        assert seen == ["emit"]

    def test_the_output_dir_follows_the_model_dir(self, monkeypatch) -> None:
        """MUST: `--out` 未指定なら系列はサイズ軸に追随する（固定だと上書きになる）。"""
        seen: list[Path] = []
        monkeypatch.setattr(
            da, "export_series", lambda _dir, out, **_kw: seen.append(out) or {"dir": str(out)}
        )

        da.main(["--model-dir", "/tmp/Depth-Anything-V2-Large-hf"])

        assert seen == [SERIES_ROOT / "depth-anything-v2-large-hf"]

    @pytest.mark.parametrize(("argv", "expected"), [([], False), (["--real-images"], True)])
    def test_real_images_reaches_the_series(self, monkeypatch, argv, expected) -> None:
        """`--real-images` が emit まで届く（届かないと実画像 golden が黙って書かれない）。"""
        seen: list[bool] = []
        monkeypatch.setattr(
            da,
            "export_series",
            lambda _dir, _out, real_images=False: seen.append(real_images) or {"dir": "x"},
        )

        da.main(argv)

        assert seen == [expected]


class TestRealRegions:
    """実画像の判別（構図から言える近い領域 > 遠い領域）。"""

    def test_every_real_case_has_a_near_far_pair(self) -> None:
        # 対が欠けたケースは `_real_sanity` で KeyError になる前に、ここで名指しで落とす。
        assert set(da.REAL_REGIONS) == {name for name, _file, _why in da.REAL_CASES}
        for name, regions in da.REAL_REGIONS.items():
            assert len(regions) == 2, name
            for _label, top, bottom, left, right in regions:
                assert 0.0 <= top < bottom <= 1.0 and 0.0 <= left < right <= 1.0, name

    def test_the_near_region_is_lower_in_the_frame_than_the_far_region(self) -> None:
        # 構図の主張そのもの（手前は画面下方にある）。矩形を書き換えて主張が裏返ると落ちる。
        for name, (
            (_near, near_top, _nb, _nl, _nr),
            (_far, far_top, *_rest),
        ) in da.REAL_REGIONS.items():
            assert near_top > far_top, name

    def test_a_near_far_ordering_passes(self) -> None:
        report = da._real_sanity(_real_depths(near=3.0, far=0.5))

        assert set(report) == set(da.REAL_REGIONS)
        assert report["photo-corridor"]["floor"] > report["photo-corridor"]["vanishing-point"]

    def test_a_flipped_ordering_is_rejected(self) -> None:
        with pytest.raises(AssertionError, match="遠近"):
            da._real_sanity(_real_depths(near=0.5, far=3.0))

    def test_a_uniform_output_is_rejected(self) -> None:
        # 一様な地図は「近側 <= 遠側」になり、恒真な sanity ならここが素通りする。
        with pytest.raises(AssertionError):
            da._real_sanity(_real_depths(near=1.0, far=1.0))

    def test_an_empty_region_fails_loudly(self) -> None:
        tiny = torch.zeros((1, 4, 4))
        with pytest.raises(AssertionError, match="空になった"):
            da._region_mean(tiny, ("sliver", 0.10, 0.12, 0.0, 1.0))


def _processor(**overrides):
    """既定を上書きした `DPTImageProcessor`（transformers が無い環境では skip）。"""
    pytest.importorskip("torchvision")
    transformers = pytest.importorskip("transformers")
    processor = transformers.DPTImageProcessor(
        resample=da.EXPECTED_RESAMPLE,
        image_mean=list(da.IMAGENET_MEAN),
        image_std=list(da.IMAGENET_STD),
        rescale_factor=da.EXPECTED_RESCALE,
        size={"height": 518, "width": 518},
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
    )
    for key, value in overrides.items():
        setattr(processor, key, value)
    return processor


class TestCheckProcessor:
    def test_the_expected_shape_passes(self) -> None:
        shape = da.check_processor(_processor())

        assert shape["resample"] == 3
        assert shape["ensure_multiple_of"] == 14

    def test_bilinear_is_rejected(self) -> None:
        # 一番の勘所。`resample` は PIL の定数で 2 = BILINEAR / 3 = BICUBIC で、SigLIP2 の
        # 2 を当てると TS 側は「参照と別のフィルタ」に合わせたまま緑になる。
        with pytest.raises(SystemExit, match="resample"):
            da.check_processor(_processor(resample=2))

    @pytest.mark.parametrize(
        ("key", "value", "match"),
        [
            ("image_mean", [0.5, 0.5, 0.5], "mean/std"),
            ("image_std", [0.5, 0.5, 0.5], "mean/std"),
            ("rescale_factor", 1.0, "rescale_factor"),
            ("do_resize", False, "do_resize"),
            ("do_rescale", False, "do_rescale"),
            ("do_normalize", False, "do_normalize"),
            ("do_center_crop", True, "do_center_crop"),
            ("do_pad", True, "do_pad"),
        ],
    )
    def test_each_constant_is_gated(self, key, value, match) -> None:
        with pytest.raises(SystemExit, match=match):
            da.check_processor(_processor(**{key: value}))

    def test_the_pil_backend_is_rejected(self) -> None:
        pytest.importorskip("torchvision")
        transformers = pytest.importorskip("transformers")
        with pytest.raises(SystemExit, match="TorchvisionBackend"):
            da.check_processor(transformers.DPTImageProcessorPil())


class TestRealCases:
    def test_a_missing_image_stops_before_the_heavy_imports(self, tmp_path) -> None:
        """MUST: `--real-images` は明示の意思表示 — 3 枚で黙って書かない。"""
        with pytest.raises(SystemExit, match="demo:eval-images"):
            da.build_real_cases(da.DEFAULT_MODEL_DIR, 518, tmp_path)
