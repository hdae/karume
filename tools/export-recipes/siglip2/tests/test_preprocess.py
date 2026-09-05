"""`siglip2/preprocess.py` の台本レベルの約束事（実重み不要分）。

フィクスチャの emit は手動（`irodori/tokenizer_ref.py` と同じ規律）。ここで固定するのは、壊れると
**沈黙のパリティ不一致**になる側の規律だけ:

- `resample` の門が **2（PIL BILINEAR）以外を拒む**こと — 3（BICUBIC）を素通しすると、
  TS 側は「参照と別のフィルタ」に合わせて緑になる（実測で最大 47/255 ずれる形）
- 前処理定数（mean / std / rescale）と `do_*` の門が、1 欄ずらすだけで落ちること
- `check_sibling_configs` が `size` **以外**の差を捕まえること（この層が 224 / 384 の両方に
  効く、という主張の中身）
- `reference` の「保存する中間は実経路そのもの」門が、resize を差し替えたら落ちること
- ケースが互いに違う画像で、縮小 / 拡大 / 同寸 / 軸混在 / 1×1 を実際に覆っていること
  （ケースを削れば落ちる形で書く）
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from siglip2 import preprocess as pre


def _write_config(root, name: str, **overrides) -> None:
    config = {
        "do_normalize": True,
        "do_rescale": True,
        "do_resize": True,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "image_processor_type": "SiglipImageProcessor",
        "rescale_factor": 0.00392156862745098,
        "resample": 2,
        "size": {"height": 224, "width": 224},
    }
    config.update(overrides)
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "preprocessor_config.json").write_text(json.dumps(config), encoding="utf-8")


class TestCases:
    def test_cases_cover_the_geometries_that_have_their_own_branch(self) -> None:
        cases = pre.build_cases()
        shapes = {
            case["name"]: (case["image"].shape[0], case["image"].shape[1], *case["out"])
            for case in cases
        }
        assert len(shapes) == len(cases), "ケース名が重複している"
        # 縮小 / 拡大 / 同寸 / 軸混在 / 1×1 を、名前ではなく寸法から数える。
        shrink = [s for s in shapes.values() if s[0] > s[2] and s[1] > s[3]]
        grow = [s for s in shapes.values() if s[0] < s[2] and s[1] < s[3]]
        same = [s for s in shapes.values() if s[0] == s[2] and s[1] == s[3]]
        mixed = [s for s in shapes.values() if (s[0] > s[2]) != (s[1] > s[3])]
        single = [s for s in shapes.values() if s[0] == 1 and s[1] == 1]
        assert len(shrink) >= 3 and grow and same and mixed and single

    def test_images_are_pairwise_distinct(self) -> None:
        # 同じ画像を 2 度使うと、その 2 ケースは同じ 1 本の検査に縮む。
        blobs = [case["image"].tobytes() for case in pre.build_cases()]
        assert len(set(blobs)) == len(blobs)

    def test_images_are_rgb8_of_the_declared_shape(self) -> None:
        for case in pre.build_cases():
            image = case["image"]
            assert image.dtype == np.uint8
            assert image.ndim == 3 and image.shape[2] == 3

    def test_cases_are_reproducible(self) -> None:
        # 乱数種を固定しているので、2 回呼べば同じ画素が出る（出なければフィクスチャは
        # 再生成のたびに別物になり、diff が読めなくなる）。
        first = [case["image"].tobytes() for case in pre.build_cases()]
        second = [case["image"].tobytes() for case in pre.build_cases()]
        assert first == second


class TestSiblingConfigs:
    def test_size_may_differ(self, tmp_path) -> None:
        _write_config(tmp_path, "base", size={"height": 224, "width": 224})
        _write_config(tmp_path, "so400m", size={"height": 384, "width": 384})
        sizes = pre.check_sibling_configs(tmp_path)
        assert sizes == {
            "base": {"height": 224, "width": 224},
            "so400m": {"height": 384, "width": 384},
        }

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("resample", 3),
            ("image_mean", [0.485, 0.456, 0.406]),
            ("rescale_factor", 1.0),
            ("do_normalize", False),
        ],
    )
    def test_other_fields_may_not_differ(self, tmp_path, key, value) -> None:
        _write_config(tmp_path, "base")
        _write_config(tmp_path, "so400m", **{key: value})
        with pytest.raises(SystemExit, match=key):
            pre.check_sibling_configs(tmp_path)

    def test_no_checkpoint_is_an_error(self, tmp_path) -> None:
        with pytest.raises(SystemExit):
            pre.check_sibling_configs(tmp_path)


def _processor(**overrides):
    """既定を上書きした `SiglipImageProcessor`（transformers が無い環境では skip）。"""
    pytest.importorskip("torchvision")
    transformers = pytest.importorskip("transformers")
    processor = transformers.SiglipImageProcessor(
        resample=pre.EXPECTED_RESAMPLE,
        image_mean=list(pre.EXPECTED_MEAN),
        image_std=list(pre.EXPECTED_STD),
        rescale_factor=pre.EXPECTED_RESCALE,
        size={"height": 8, "width": 8},
    )
    for key, value in overrides.items():
        setattr(processor, key, value)
    return processor


class TestProcessorShape:
    def test_the_expected_shape_passes(self) -> None:
        shape = pre.check_processor_shape(_processor())
        assert shape["resample"] == 2

    def test_bicubic_is_rejected(self) -> None:
        # 一番の勘所。`resample` は PIL の定数で 2 = BILINEAR / 3 = BICUBIC で、クラス属性の
        # 既定は BICUBIC。config の上書きを読み落とした瞬間にここで止まる。
        with pytest.raises(SystemExit, match="resample"):
            pre.check_processor_shape(_processor(resample=3))

    @pytest.mark.parametrize(
        ("key", "value", "match"),
        [
            ("image_mean", [0.485, 0.456, 0.406], "mean/std"),
            ("image_std", [0.229, 0.224, 0.225], "mean/std"),
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
            pre.check_processor_shape(_processor(**{key: value}))

    def test_the_pil_backend_is_rejected(self) -> None:
        transformers = pytest.importorskip("transformers")
        pytest.importorskip("torchvision")
        with pytest.raises(SystemExit, match="TorchvisionBackend"):
            pre.check_processor_shape(transformers.SiglipImageProcessorPil())


class TestReference:
    IMAGE = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)

    def test_the_stored_intermediate_is_on_the_real_path(self) -> None:
        processor = _processor()
        resized, pixel_values = pre.reference(processor, self.IMAGE, 4, 6)
        assert resized.shape == (4, 6, 3)
        assert resized.dtype == np.uint8
        assert pixel_values.shape == (3, 4, 6)

    def test_an_unfused_upstream_fails_loudly(self, monkeypatch) -> None:
        # 上流が rescale と normalize の**畳み方**を変えた形（先に 1/255 を掛けてから
        # `[0,1]` 尺度の mean/std を引く）を注入する。値は肉眼では同じだが f32 の丸めが
        # 1 回増えるので、台本が焼く融合形との差がここで出る — 出なければ、TS 側は
        # 「上流が変わったのに緑のまま」になる。
        backend = pytest.importorskip("transformers.image_processing_backends")

        def unfused(self, images, do_rescale, rescale_factor, do_normalize, image_mean, image_std):
            scaled = images.to(dtype=torch.float32) * rescale_factor
            mean = torch.tensor(image_mean).view(1, 3, 1, 1)
            std = torch.tensor(image_std).view(1, 3, 1, 1)
            return (scaled - mean) / std

        monkeypatch.setattr(backend.TorchvisionBackend, "rescale_and_normalize", unfused)
        with pytest.raises(SystemExit, match="pixel_values"):
            pre.reference(_processor(), self.IMAGE, 4, 6)


class TestCommittedFixture:
    """git 追跡のフィクスチャと `build_cases()` の対応（再生成漏れを赤にする）。

    Python 側は `build_cases()` の幾何被覆を、Deno 側（`packages/models/tests/
    image_preprocess_test.ts`）は committed JSON を検査する — 突き合わせが無いと、**両者が
    別物になっても両方緑**のまま回り続ける（片方だけ編集して再生成し忘れた形）。
    """

    @staticmethod
    def _fixture() -> dict:
        # git 追跡ファイルなので、無ければ skip ではなく失敗させる。
        return json.loads(pre.DEFAULT_FIXTURE_PATH.read_text(encoding="utf-8"))

    def test_the_case_names_match_in_order(self) -> None:
        fixture = self._fixture()

        assert [case["name"] for case in fixture["cases"]] == [
            case["name"] for case in pre.build_cases()
        ]

    def test_every_case_carries_the_geometry_it_was_built_from(self) -> None:
        """入出力の寸法がずれると、Deno 側は別の縮尺で参照と突き合わせる。"""
        fixture = self._fixture()

        for baked, case in zip(fixture["cases"], pre.build_cases(), strict=True):
            image = case["image"]
            out_height, out_width = case["out"]
            assert (baked["height"], baked["width"]) == image.shape[:2], baked["name"]
            assert (baked["outHeight"], baked["outWidth"]) == (out_height, out_width), baked["name"]

    def test_the_baked_constants_are_the_expected_ones(self) -> None:
        constants = self._fixture()["constants"]

        assert tuple(constants["imageMean"]) == pre.EXPECTED_MEAN
        assert tuple(constants["imageStd"]) == pre.EXPECTED_STD
        assert constants["rescaleFactor"] == pre.EXPECTED_RESCALE
