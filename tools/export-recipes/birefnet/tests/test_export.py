"""`birefnet/export.py` の台本レベルの約束事（実重み不要分）。

実重みの emit は手動（既存 `deberta/export.py` / `siglip2/export.py` のテストと同じ規律）。
ここで固定するのは、壊れると**偽 PASS** になる側の規律だけ:

- グラフ出力が**最終段のマット 1 本**であること（multi-scale の中間予測が混ざったら io の
  位置規約が黙ってずれる = 学習モードのグラフを推論用として書き出した形）
- 系列の綴りが**モデル名と解像度の両方**に追随すること（片方でも欠けると別解像度の資産が
  同じ席へ黙って上書きされる）
- 解像度の刻み（64 の倍数）が**入口で**落ちること（途中の reshape エラーにしない）
- `_sanity` が顕著物体の分離を**順序**で見ること（値域も同一入力の一致も恒真）
- 実画像 golden（`--real-images`）が**欠けを黙って許さない**こと、元画像の sha256 を
  `__metadata__` に載せること、判別を**前景比の順序**で見ること
- `--verify` が emit しないこと（同一プロセスでは参照が汚染される）
- 系列名の綴りと正規化定数が**配布 recipe（`birefnet.distribution`）と一致**すること
  （上流に機械可読な前処理 config が無く両側が宣言を持つので、独立に動くと golden と
  利用者の前処理がずれる）
- 格納 dtype の系列（ADR 0113）: `--dtype` が系列を分けること・`--verify` と排他なこと・
  丸めが `load_model` の直後で `patch.apply` の前にあること（後に動かすと ⑦ の α / β が動く —
  故障注入）・品質計測の器と無損失の門が恒真でないこと
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn

from _shared.paths import SERIES_ROOT
from _shared.upstream import UpstreamProvenanceError
from birefnet import export as bn
from karume.container import Provenance
from karume.pipeline import export_to_file

#: tiny な export に焼く出所（値は突合されない — 台本の出所の導出は `_shared.upstream` の席）。
TINY_PROVENANCE = Provenance(license="fixture")

#: tiny な合成モデルの解像度（`disc_mask` が 2×2 の円内を持つ最小の形）。
TINY_SIZE = 8

CASES = (
    ("disc", torch.linspace(-1.0, 1.0, 3 * TINY_SIZE * TINY_SIZE).reshape(1, 3, TINY_SIZE, -1)),
    ("ramp", torch.linspace(1.0, -1.0, 3 * TINY_SIZE * TINY_SIZE).reshape(1, 3, TINY_SIZE, -1)),
)


class TinyMatte(nn.Module):
    """`MatteLogits` の最小の骨格（`[B,3,S,S] → [B,1,S,S]`・引数名まで同じ）。"""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.conv(pixel_values)


class TwoOutputMatte(TinyMatte):
    """出力が 2 本ある形（`_write_io` が拒否することの確認用）。"""

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        matte = super().forward(pixel_values)
        return matte, matte * 2.0


class ScaledPreds(nn.Module):
    """`scaled_preds`（list）を返す BiRefNet 側の骨格。要素数だけを外から決める。"""

    def __init__(self, outputs: int) -> None:
        super().__init__()
        self.outputs = outputs
        self.conv = nn.Conv2d(3, 1, kernel_size=1)

    def forward(self, pixel_values: torch.Tensor) -> list[torch.Tensor]:
        matte = self.conv(pixel_values)
        return [matte * float(index + 1) for index in range(self.outputs)]


def _disc_matte(inside: float, outside: float) -> torch.Tensor:
    """円内 / 円外をそれぞれ定数で塗ったマット（分離の順序だけを動かす）。"""
    mask = bn.disc_mask(TINY_SIZE)
    return torch.where(mask, torch.tensor(inside), torch.tensor(outside)).reshape(
        1, 1, TINY_SIZE, TINY_SIZE
    )


def _mattes(disc: torch.Tensor) -> dict[str, torch.Tensor]:
    """`disc` 以外は互いに違う適当な値で埋めた 4 ケース分。"""
    return {
        "disc": disc,
        "ramp": torch.full((1, 1, TINY_SIZE, TINY_SIZE), 0.25),
        "checker": torch.full((1, 1, TINY_SIZE, TINY_SIZE), -0.5),
        "noise": torch.full((1, 1, TINY_SIZE, TINY_SIZE), 1.5),
    }


def _ratio_matte(ratio: float, jitter: int) -> torch.Tensor:
    """前景比がちょうど `ratio` になるマット。

    `jitter` は背景値だけをケースごとにずらす（前景比を変えずに**テンソルとしては別物**に
    する — `_sanity` の「全ケースが互いに違う」検査を先に踏んで判別まで届かなくなるのを避ける）。
    """
    pixels = TINY_SIZE * TINY_SIZE
    flat = torch.full((pixels,), -1.0 - 0.01 * jitter)
    flat[: round(ratio * pixels)] = 1.0
    return flat.reshape(1, 1, TINY_SIZE, TINY_SIZE)


def _real_mattes(
    person: tuple[float, float], scene: tuple[float, float]
) -> dict[str, torch.Tensor]:
    """実画像 4 ケース分のマット（群ごとに前景比を指定する）。"""
    ratios = dict(
        zip((*bn.REAL_PERSON_CASES, *bn.REAL_SCENE_CASES), (*person, *scene), strict=True)
    )
    return {
        name: _ratio_matte(ratio, jitter=index + 1)
        for index, (name, ratio) in enumerate(ratios.items())
    }


def _real_array(index: int) -> torch.Tensor:
    """tiny な実画像 1 枚の画素（`[S, S, 3]` の u8）。ケースとチャネルで値をずらす。"""
    array = torch.zeros((TINY_SIZE, TINY_SIZE, 3), dtype=torch.uint8)
    for channel in range(3):
        array[:, :, channel] = 10 * index + 30 * channel + 7
    return array


def _write_real_images(root: Path) -> dict[str, bytes]:
    """`REAL_CASES` の綴りで tiny な PNG を書き、ケース名 → 生バイトを返す。"""
    from PIL import Image

    written: dict[str, bytes] = {}
    for index, (name, file_name, _why) in enumerate(bn.REAL_CASES):
        path = root / file_name
        Image.fromarray(_real_array(index).numpy()).save(path)
        written[name] = path.read_bytes()
    return written


@pytest.fixture
def exported(tmp_path):
    """tiny なラッパを 1 本 export して `(wrapper, graph, out_dir)` を返す。"""
    torch.manual_seed(0)
    wrapper = TinyMatte()
    graph = export_to_file(
        wrapper,
        (CASES[0][1],),
        tmp_path / bn.MODEL_FILE,
        provenance=TINY_PROVENANCE,
        graph_name="tiny",
        symbol_names=(),
    )
    return wrapper, graph, tmp_path


class TestSeriesLayout:
    def test_the_default_output_dir_is_a_series(self):
        """系列出力は `outputs/series/` 配下（配布形の `models/` ではない — _shared.paths）。"""
        assert bn.default_out_dir(bn.DEFAULT_MODEL_DIR, bn.DEFAULT_RESOLUTION).parent == SERIES_ROOT

    def test_each_resolution_gets_its_own_series(self):
        """MUST: 解像度も綴りへ — 焼く定数が変わるので、同じ席に置くと先の資産が消える。"""
        low = bn.default_out_dir(bn.DEFAULT_MODEL_DIR, 1024)
        high = bn.default_out_dir(bn.DEFAULT_MODEL_DIR, 2048)

        assert low != high
        assert low.name.endswith("-1024") and high.name.endswith("-2048")

    def test_each_model_gets_its_own_series(self):
        other = bn.MODELS_ROOT / "BiRefNet_lite"

        assert bn.default_out_dir(other, 1024) != bn.default_out_dir(bn.DEFAULT_MODEL_DIR, 1024)

    def test_the_distribution_looks_for_the_same_series_name(self):
        """MUST: 系列名の綴りが**書く側と読む側で一致**する。

        配布 recipe は「checkpoint → 上流リポ名 → 系列ディレクトリ名 + 解像度」で系列を探す。
        式が片方だけ動くと、組み立ては「系列が無い」で落ちる（それ自体は安全）が、**別の
        モデルの系列を掴む**綴りにずれた場合は誰も気づけない。
        """
        from birefnet.card import BIREFNET_UPSTREAM
        from birefnet.distribution import BIREFNET_MODELS, birefnet_series_name

        for checkpoint, repo in BIREFNET_UPSTREAM.items():
            model_dir = bn.MODELS_ROOT / repo.split("/", 1)[1]
            for model in BIREFNET_MODELS:
                assert bn.default_out_dir(model_dir, int(model)).name == birefnet_series_name(
                    checkpoint, model
                )


class TestResolution:
    @pytest.mark.parametrize("resolution", [64, 256, 1024, 2048])
    def test_multiples_of_the_step_are_accepted(self, resolution: int):
        bn.assert_resolution(resolution)

    @pytest.mark.parametrize("resolution", [0, -1024, 224, 1000, 1056])
    def test_anything_else_fails_loudly(self, resolution: int):
        """MUST: 入口で落とす — 通すと Swin の途中で形が合わなくなるだけで理由が残らない。"""
        with pytest.raises(SystemExit, match="の倍数でない"):
            bn.assert_resolution(resolution)


class TestMatteLogits:
    def test_it_returns_the_only_prediction(self):
        torch.manual_seed(0)
        wrapper = bn.MatteLogits(ScaledPreds(1)).eval()
        pixel_values = CASES[0][1]

        with torch.no_grad():
            assert torch.equal(wrapper(pixel_values), wrapper.model(pixel_values)[0])

    def test_more_than_one_prediction_fails_loudly(self):
        """MUST: 中間予測が付くのは学習モード — 黙って `[-1]` を採らない。"""
        wrapper = bn.MatteLogits(ScaledPreds(4)).eval()

        with pytest.raises(ValueError, match="学習モードのグラフは書き出さない"):
            wrapper(CASES[0][1])


class TestPreprocessing:
    def test_normalize_matches_the_reference_transform(self):
        """MUST: 正規化は handler.py の逐語 — torchvision の実装を独立オラクルに使う。"""
        from torchvision import transforms

        torch.manual_seed(1)
        image = torch.rand(1, 3, TINY_SIZE, TINY_SIZE)
        expected = transforms.Normalize(bn.IMAGENET_MEAN, bn.IMAGENET_STD)(image[0]).unsqueeze(0)

        assert torch.equal(bn.normalize_image(image), expected)

    def test_the_statistics_are_per_channel(self):
        """3 チャネルが同じ定数だと、前処理の順序違いが値に出ない。"""
        assert len(set(bn.IMAGENET_MEAN)) == 3
        assert len(set(bn.IMAGENET_STD)) == 3

    def test_the_distribution_declares_the_same_constants(self):
        """MUST: 配布形が宣言する数と、golden を焼く数が一致する。

        BiRefNet 系の上流には `preprocessor_config.json` に当たる機械可読な出どころが無く、
        正規化定数は同梱 `handler.py` の中にしか書かれていない。そのため台本（ここ）と
        組み立て（`birefnet.distribution`）の両方が宣言を持たざるを得ない — 2 表が独立に動くと、
        **golden は片方の統計で焼かれ、利用者はもう片方で前処理する**形が黙って作れる。
        """
        from birefnet.distribution import BIREFNET_IMAGE_MEAN, BIREFNET_IMAGE_STD

        assert BIREFNET_IMAGE_MEAN == bn.IMAGENET_MEAN
        assert BIREFNET_IMAGE_STD == bn.IMAGENET_STD


class TestGoldenCases:
    def test_cases_have_the_configured_shape(self):
        for name, pixel_values in bn.build_cases(TINY_SIZE):
            assert tuple(pixel_values.shape) == (1, 3, TINY_SIZE, TINY_SIZE), name
            assert pixel_values.dtype is torch.float32, name

    def test_every_case_is_a_different_image(self):
        """MUST: 同じ画像を 2 度使わない — 出力どうしの差分を見る sanity が恒真になる。"""
        cases = bn.build_cases(TINY_SIZE)

        for index, (name, pixel_values) in enumerate(cases):
            for other_name, other in cases[index + 1 :]:
                assert not torch.equal(pixel_values, other), f"{name} と {other_name} が同一"

    def test_the_disc_case_actually_has_a_disc(self):
        """判別の土台 — 円内と円外が同じ色なら {@link bn._sanity} は意味を失う。"""
        cases = dict(bn.build_cases(TINY_SIZE))
        image = cases[bn.DISC_CASE][0]
        mask = bn.disc_mask(TINY_SIZE)

        assert bool(mask.any()) and not bool(mask.all())
        for channel in range(3):
            plane = image[channel]
            assert float(plane[mask].std()) == 0.0
            assert float(plane[mask].mean()) != float(plane[~mask].mean())

    def test_the_sanity_case_is_one_of_the_cases(self):
        assert bn.DISC_CASE in {name for name, _ in bn.build_cases(TINY_SIZE)}


class TestWriteIo:
    def test_writes_one_file_per_case_with_the_declared_keys(self, exported):
        wrapper, graph, out_dir = exported

        written, mattes = bn._write_io(wrapper, graph, CASES, out_dir)

        assert written == [f"{bn.IO_PREFIX}{name}{bn.IO_SUFFIX}" for name, _ in CASES]
        tensors = load_file(str(out_dir / written[0]))
        assert set(tensors) == {f"{bn.INPUT_PREFIX}{bn.INPUT_NAME}", f"{bn.OUTPUT_PREFIX}0"}
        assert tuple(mattes["disc"].shape) == (1, 1, TINY_SIZE, TINY_SIZE)

    def test_more_than_one_graph_output_fails_loudly(self, tmp_path):
        """MUST: 出力はマット 1 本 — 2 本目が生えると io の位置規約が黙ってずれる。"""
        torch.manual_seed(0)
        wrapper = TwoOutputMatte()
        graph = export_to_file(
            wrapper,
            (CASES[0][1],),
            tmp_path / bn.MODEL_FILE,
            provenance=TINY_PROVENANCE,
            graph_name="tiny",
            symbol_names=(),
        )

        with pytest.raises(AssertionError, match="マットは 1 本"):
            bn._write_io(wrapper, graph, CASES, tmp_path)

    def test_metadata_is_written_only_for_the_cases_that_have_it(self, exported):
        """実画像ケースだけが `__metadata__`（元画像の同定）を持つ。"""
        wrapper, graph, out_dir = exported
        metadata = {CASES[0][0]: {bn.SOURCE_IMAGE_KEY: "photo.png", bn.SOURCE_SHA256_KEY: "ab"}}

        written, _ = bn._write_io(wrapper, graph, CASES, out_dir, metadata)

        with safe_open(str(out_dir / written[0]), framework="pt") as handle:
            assert handle.metadata() == metadata[CASES[0][0]]
        with safe_open(str(out_dir / written[1]), framework="pt") as handle:
            assert handle.metadata() is None


class TestSanity:
    def test_passes_when_the_disc_is_brighter_than_its_surroundings(self):
        result = bn._sanity(_mattes(_disc_matte(inside=3.0, outside=-4.0)))

        assert result["disc_logit_mean"] == {"inside": 3.0, "outside": -4.0}
        covered = float(bn.disc_mask(TINY_SIZE).to(torch.float32).mean())
        assert result["foreground_ratio"][bn.DISC_CASE] == pytest.approx(covered)

    def test_fails_loudly_when_the_order_is_inverted(self):
        """MUST: 顕著物体が背景より暗いなら、セグメンテーションとして壊れている。"""
        with pytest.raises(AssertionError, match="顕著物体を分離できていない"):
            bn._sanity(_mattes(_disc_matte(inside=-4.0, outside=3.0)))

    def test_fails_loudly_when_the_matte_is_uniform(self):
        """一様に潰れた出力（円内と円外が同値）も落とす。"""
        with pytest.raises(AssertionError, match="顕著物体を分離できていない"):
            bn._sanity(_mattes(_disc_matte(inside=0.5, outside=0.5)))

    def test_fails_loudly_when_two_cases_share_an_output(self):
        """MUST: 入力が届いていない形（同じ出力が並ぶ）を落とす。"""
        mattes = _mattes(_disc_matte(inside=3.0, outside=-4.0))
        mattes["checker"] = mattes["ramp"].clone()

        with pytest.raises(AssertionError, match="入力が効いていない"):
            bn._sanity(mattes)

    def test_it_reports_the_shape_mismatch_instead_of_broadcasting(self):
        """円の形と出力の形が食い違ったら黙って broadcast させない（H ≠ W のマット）。"""
        mattes = _mattes(torch.zeros(1, 1, 2, TINY_SIZE))

        with pytest.raises(AssertionError, match="円の形と違う"):
            bn._sanity(mattes)


class TestRealImageCases:
    def test_the_two_groups_partition_the_cases(self):
        """MUST: 群の綴りがケース名から外れたら落とす（判別が黙って別の 2 枚を見る）。"""
        names = {name for name, _file, _why in bn.REAL_CASES}

        assert set(bn.REAL_PERSON_CASES).isdisjoint(bn.REAL_SCENE_CASES)
        assert set(bn.REAL_PERSON_CASES) | set(bn.REAL_SCENE_CASES) == names

    def test_every_case_reads_a_different_file(self):
        files = [file for _name, file, _why in bn.REAL_CASES]

        assert len(set(files)) == len(files)

    def test_it_normalizes_with_the_handler_statistics(self, tmp_path):
        """前処理は handler.py の逐語 — 値を独立に組み直して突き合わせる。"""
        raw = _write_real_images(tmp_path)

        cases = bn.build_real_cases(TINY_SIZE, tmp_path)

        assert [name for name, _pixels, _md in cases] == list(raw)
        mean = torch.tensor(bn.IMAGENET_MEAN).reshape(3, 1, 1)
        std = torch.tensor(bn.IMAGENET_STD).reshape(3, 1, 1)
        for index, (name, pixel_values, _md) in enumerate(cases):
            assert tuple(pixel_values.shape) == (1, 3, TINY_SIZE, TINY_SIZE), name
            assert pixel_values.dtype is torch.float32, name
            # 期待値は書いた画素から組み直す（`ToTensor` / `Normalize` を写経せず「planar へ
            # 並べ替えて 255 で割り、統計を引く」だけを別経路で置く）。
            planar = _real_array(index).permute(2, 0, 1).to(torch.float32)
            assert torch.equal(pixel_values[0], ((planar / 255.0) - mean) / std), name

    def test_it_records_the_source_image_and_its_digest(self, tmp_path):
        """MUST: 焼き直した画像で golden を採り直し忘れた環境を、突合の前に落とすための欄。"""
        raw = _write_real_images(tmp_path)

        cases = bn.build_real_cases(TINY_SIZE, tmp_path)

        files = {name: file for name, file, _why in bn.REAL_CASES}
        for name, _pixel_values, metadata in cases:
            assert metadata[bn.SOURCE_IMAGE_KEY] == files[name]
            assert metadata[bn.SOURCE_SHA256_KEY] == hashlib.sha256(raw[name]).hexdigest()

    def test_a_missing_image_fails_loudly(self, tmp_path):
        """MUST: 黙って 3 枚で書かない（`--real-images` は明示の意思表示）。"""
        _write_real_images(tmp_path)
        (tmp_path / bn.REAL_CASES[-1][1]).unlink()

        with pytest.raises(SystemExit, match=r"実画像 .* が無い"):
            bn.build_real_cases(TINY_SIZE, tmp_path)


class TestRealSanity:
    def test_synthetic_only_emits_have_no_real_verdict(self):
        """既定の emit（合成 4 ケース）でも sanity は通る（実画像は追加の群）。"""
        assert "real_foreground" not in bn._sanity(_mattes(_disc_matte(inside=3.0, outside=-4.0)))

    def test_it_passes_when_the_salient_cases_have_more_foreground(self):
        mattes = {
            **_mattes(_disc_matte(inside=3.0, outside=-4.0)),
            **_real_mattes(person=(0.5, 0.125), scene=(0.0, 0.0625)),
        }

        result = bn._sanity(mattes)

        assert result["real_foreground"][bn.REAL_PERSON_CASES[0]] == 0.5
        assert result["real_foreground"][bn.REAL_SCENE_CASES[0]] == 0.0

    def test_it_fails_loudly_when_a_scene_has_more_foreground(self):
        """MUST: 顕著物体の無い 1 枚が人物より広いなら、マットは意味を捉えていない。"""
        mattes = {
            **_mattes(_disc_matte(inside=3.0, outside=-4.0)),
            **_real_mattes(person=(0.5, 0.0625), scene=(0.0, 0.25)),
        }

        with pytest.raises(AssertionError, match="実画像の前景比の順序が逆"):
            bn._sanity(mattes)

    def test_it_fails_loudly_when_every_case_covers_the_same_area(self):
        """MUST: 面積が入力に依存しなくなった出力を落とす（値は違っても順序が並ぶ）。

        値まで同一なら手前の「全ケースが互いに違う」検査が先に落とすので、ここは**面積だけ**を
        揃える（前景比を見る側の穴が残っていないことの確認）。
        """
        mattes = {
            **_mattes(_disc_matte(inside=3.0, outside=-4.0)),
            **_real_mattes(person=(0.5, 0.5), scene=(0.5, 0.5)),
        }

        with pytest.raises(AssertionError, match="実画像の前景比の順序が逆"):
            bn._sanity(mattes)


class TestVerifyCli:
    def test_verify_does_not_emit(self, monkeypatch):
        """MUST: 同一プロセスで emit と併用しない（クラス差し替えが参照を汚染する）。"""
        seen: list[str] = []
        monkeypatch.setattr(
            bn,
            "verify_patches",
            lambda _dir, _resolution: (
                seen.append("verify")
                or [{"stage": "layout", "claim": "bit-exact", "bit_exact": True, "maxdiff": {}}]
            ),
        )
        monkeypatch.setattr(
            bn, "export_series", lambda *_a, **_kw: pytest.fail("--verify で emit された")
        )

        bn.main(["--verify"])

        assert seen == ["verify"]

    def test_without_verify_it_emits(self, monkeypatch):
        seen: list[str] = []
        monkeypatch.setattr(
            bn, "export_series", lambda *_a, **_kw: seen.append("emit") or {"dir": "x"}
        )
        monkeypatch.setattr(
            bn, "verify_patches", lambda *_a: pytest.fail("emit で --verify が走った")
        )

        bn.main([])

        assert seen == ["emit"]

    def test_the_output_dir_follows_the_model_dir_and_the_resolution(self, monkeypatch):
        """MUST: `--out` 未指定なら系列は両軸に追随する（固定だと上書きになる）。"""
        seen: list[tuple[Path, int]] = []
        monkeypatch.setattr(
            bn,
            "export_series",
            lambda _dir, out, resolution, **_kw: (
                seen.append((out, resolution)) or {"dir": str(out)}
            ),
        )

        bn.main(["--model-dir", "/tmp/BiRefNet_lite", "--resolution", "512"])

        assert seen == [(SERIES_ROOT / "birefnet-lite-512", 512)]

    @pytest.mark.parametrize(("argv", "expected"), [([], False), (["--real-images"], True)])
    def test_real_images_is_opt_in(self, monkeypatch, argv: list[str], expected: bool):
        """実画像 golden は明示の意思表示でだけ書く（既定は合成 4 ケース）。"""
        seen: list[bool] = []
        monkeypatch.setattr(
            bn,
            "export_series",
            lambda _dir, out, _resolution, real_images, **_kw: (
                seen.append(real_images) or {"dir": str(out)}
            ),
        )

        bn.main(argv)

        assert seen == [expected]


class _RecordingMatte(nn.Module):
    """呼び出しの段（参照 / 段 1 / 段 2 / 段 3）を記録するだけの骨格。

    出力は入力から決まる（段によらず同一）ので、`verify_patches` の 1 段目のビット一致
    assert は通る — ここで見たいのは**順序**だけ。
    """

    def __init__(self, calls: list[str]) -> None:
        super().__init__()
        self.calls = calls
        self.stage = "reference"

    def forward(self, pixel_values: torch.Tensor) -> list[torch.Tensor]:
        if not self.calls or self.calls[-1] != self.stage:
            self.calls.append(self.stage)
        return [pixel_values.mean(dim=1, keepdim=True)]


class TestVerifyOrder:
    """`verify_patches` の順序不変条件（参照 → 段 1 → 段 2 → 段 3）。"""

    RESOLUTION = 64

    def test_a_patched_process_cannot_take_the_reference(self, monkeypatch):
        """MUST: パッチ適用済みのプロセスでは参照を採らない（差 0 の恒真化）。"""
        monkeypatch.setattr(bn.patch, "patches_applied", lambda: True)

        with pytest.raises(SystemExit, match="恒真化"):
            bn.verify_patches(Path("/nonexistent"), self.RESOLUTION)

    def test_the_reference_is_taken_once_before_any_patch(self, monkeypatch):
        """段ごとに参照を採り直す退行（2 段目の参照がパッチ後の値になる）を落とす。"""
        calls: list[str] = []
        recorder = _RecordingMatte(calls)
        monkeypatch.setattr(bn.patch, "patches_applied", lambda: False)
        monkeypatch.setattr(bn, "load_model", lambda _dir: recorder)

        def _apply_layout(model: nn.Module) -> dict[str, int]:
            calls.append("apply_layout")
            model.stage = "layout"
            return {}

        def _apply_modules(model: nn.Module) -> dict[str, int]:
            calls.append("apply_modules")
            model.stage = "modules"
            return {}

        def _apply_tail(model: nn.Module) -> dict[str, int]:
            calls.append("apply_tail")
            model.stage = "tail"
            return {}

        monkeypatch.setattr(bn.patch, "apply_layout_patches", _apply_layout)
        monkeypatch.setattr(bn.patch, "apply_module_patches", _apply_modules)
        monkeypatch.setattr(bn.patch, "apply_tail_patches", _apply_tail)
        monkeypatch.setattr(
            bn.patch, "prepare", lambda _wrapper, _sample: calls.append("prepare") or None
        )

        entries = bn.verify_patches(Path("/nonexistent"), self.RESOLUTION)

        assert calls == [
            "reference",
            "apply_layout",
            "prepare",
            "layout",
            "apply_modules",
            "modules",
            "apply_tail",
            "tail",
        ]
        assert [entry["stage"] for entry in entries] == ["layout", "modules", "tail"]


class TestProvenance:
    """容器の出所は `--model-dir` の実物から導く（`_shared.upstream.snapshot_provenance`）。"""

    def test_a_checkpoint_without_the_download_record_is_refused_before_the_weights(
        self, monkeypatch, tmp_path
    ) -> None:
        """出所を名乗れない checkpoint は、重みを読む前に落ちる（何も据わらない）。"""
        monkeypatch.setattr(
            bn, "load_wrapper", lambda *_a: pytest.fail("出所の検査より先に重みを読んだ")
        )
        out_dir = tmp_path / "series"

        with pytest.raises(UpstreamProvenanceError, match="が無い"):
            bn.export_series(tmp_path / "BiRefNet_HR", out_dir, bn.DEFAULT_RESOLUTION)

        assert not out_dir.exists()


# ---- 格納 dtype の系列（ADR 0113）----------------------------------------------


class TestWeightDtypeSeries:
    def test_the_f32_series_keeps_its_existing_name(self):
        """f32 は接尾なし — 既存系列（配布済みの資産と e2e の網）の綴りを動かさない。"""
        assert bn.default_out_dir(bn.DEFAULT_MODEL_DIR, 1024, "f32") == bn.default_out_dir(
            bn.DEFAULT_MODEL_DIR, 1024
        )
        assert bn.default_out_dir(bn.DEFAULT_MODEL_DIR, 1024).name == "birefnet-hr-1024"

    def test_each_storage_dtype_gets_its_own_series(self):
        """MUST: 同居させると f32 系列の網（系列ごとの tolerance）が圧縮資産へ黙って掛かる。"""
        plain = bn.default_out_dir(bn.DEFAULT_MODEL_DIR, 1024, "f32")
        half = bn.default_out_dir(bn.DEFAULT_MODEL_DIR, 1024, "f16")

        assert plain != half
        assert half.name == "birefnet-hr-1024-f16"

    def test_the_distribution_offers_exactly_the_dtypes_the_recipe_writes(self):
        """書き手（台本）と読み手（配布 recipe）の dtype 表が独立に動くと、焼けない席を配る。"""
        from birefnet.distribution import BIREFNET_WEIGHT_DTYPES

        assert bn.WEIGHT_DTYPES == BIREFNET_WEIGHT_DTYPES

    def test_the_distribution_looks_for_the_same_series_name_per_dtype(self):
        """MUST: 系列名の綴りが dtype の接尾まで書く側と読む側で一致する。"""
        from birefnet.card import BIREFNET_UPSTREAM
        from birefnet.distribution import BIREFNET_MODELS, birefnet_series_name

        for checkpoint, repo in BIREFNET_UPSTREAM.items():
            model_dir = bn.MODELS_ROOT / repo.split("/", 1)[1]
            for model in BIREFNET_MODELS:
                for dtype in bn.WEIGHT_DTYPES:
                    assert bn.default_out_dir(
                        model_dir, int(model), dtype
                    ).name == birefnet_series_name(checkpoint, model, dtype)


def _record_emit(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Path, str]]:
    """`export_series` を差し替え、渡った `(out_dir, dtype)` を積む。"""
    seen: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        bn,
        "export_series",
        lambda _dir, out, _resolution, real_images, dtype: (
            seen.append((out, dtype)) or {"dir": str(out)}
        ),
    )
    return seen


class TestDtypeCli:
    def test_the_default_is_the_plain_f32_series(self, monkeypatch):
        seen = _record_emit(monkeypatch)

        bn.main([])

        assert seen == [(SERIES_ROOT / "birefnet-hr-1024", "f32")]

    def test_f16_is_written_to_its_own_series(self, monkeypatch):
        seen = _record_emit(monkeypatch)

        bn.main(["--dtype", "f16", "--resolution", "2048"])

        assert seen == [(SERIES_ROOT / "birefnet-hr-2048-f16", "f16")]

    @pytest.mark.parametrize("dtype", ["f32", "f16"])
    def test_dtype_and_verify_are_exclusive(self, monkeypatch, dtype: str):
        """MUST: `--verify` の参照はパッチ前の f32 eager — 明示の `--dtype` は f32 でも拒む
        （ADR 0027 決定 3 と同じ構造理由）。どちらの経路も走らない。
        """
        monkeypatch.setattr(bn, "verify_patches", lambda *_a: pytest.fail("--verify が走った"))
        monkeypatch.setattr(bn, "export_series", lambda *_a, **_kw: pytest.fail("emit された"))

        with pytest.raises(SystemExit):
            bn.main(["--verify", "--dtype", dtype])

    def test_i8_is_not_offered_yet(self, monkeypatch):
        """i8 は BiRefNet 固有の i8 計画と一緒に段 2 で足す — 写しの i8 は黙って壊れる。"""
        monkeypatch.setattr(bn, "export_series", lambda *_a, **_kw: pytest.fail("emit された"))

        with pytest.raises(SystemExit):
            bn.main(["--dtype", "i8"])


class TinyBatchNormMatte(nn.Module):
    """`load_model` が返す形の骨格（list を返す）に BatchNorm を 1 本挟んだもの。

    ⑦（`BatchNorm2d` → `ChannelAffine`）の α / β が丸めの位置でどう動くかを見る最小形。
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(4)
        self.head = nn.Conv2d(4, 1, kernel_size=1)

    def forward(self, pixel_values: torch.Tensor) -> list[torch.Tensor]:
        return [self.head(self.norm(self.conv(pixel_values)))]


def _f16_checkpoint_model() -> TinyBatchNormMatte:
    """HR 相当（全ての重みと BatchNorm 統計が f16 で表せる値）の eval モデル。"""
    torch.manual_seed(7)
    model = TinyBatchNormMatte().eval()
    with torch.no_grad():
        model.norm.running_mean.uniform_(-0.5, 0.5)
        model.norm.running_var.uniform_(0.3, 3.0)
        model.norm.weight.uniform_(0.5, 1.5)
        model.norm.bias.uniform_(-0.2, 0.2)
        for tensor in (*model.parameters(), model.norm.running_mean, model.norm.running_var):
            tensor.copy_(tensor.to(torch.float16).to(torch.float32))
    return model


def _affine(model: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """⑦ で生えた `ChannelAffine` の α / β。"""
    affines = [module for module in model.modules() if isinstance(module, bn.patch.ChannelAffine)]
    assert len(affines) == 1
    return affines[0].alpha.clone(), affines[0].beta.clone()


def _f32_checkpoint_model() -> TinyBatchNormMatte:
    """Lucida 相当（重みと BatchNorm 統計が f16 で表せない f32 の値）の eval モデル。

    seed を固定するので、基準用と丸め用の 2 回の読み込みが同じ重みを返す。
    """
    torch.manual_seed(11)
    model = TinyBatchNormMatte().eval()
    with torch.no_grad():
        model.norm.running_mean.uniform_(-0.5, 0.5)
        model.norm.running_var.uniform_(0.3, 3.0)
        model.norm.weight.uniform_(0.5, 1.5)
        model.norm.bias.uniform_(-0.2, 0.2)
    return model


@pytest.fixture
def tiny_load(monkeypatch: pytest.MonkeyPatch) -> list[str | tuple[str, str]]:
    """`load_wrapper` の外部（重みの読み込み・パッチ・焼き）を tiny へ差し替え、呼び順を積む。

    パッチは ⑦⑧（`apply_module_patches` — 任意のモジュールへ当たる段）だけを当てる。①〜⑥ は
    上流の動的モジュールのクラス属性を差し替えるので tiny には当たらない。丸めは
    `("round", dtype)` で積む（基準と丸め側の取り違えを呼び順で見分けるため）。
    """
    calls: list[str | tuple[str, str]] = []

    def _load(_dir: Path) -> nn.Module:
        calls.append("load")
        return _f16_checkpoint_model()

    real_round = bn.round_to_storage

    def _round(model: nn.Module, dtype: str):
        calls.append(("round", dtype))
        return real_round(model, dtype)

    def _apply(model: nn.Module) -> dict[str, int]:
        calls.append("apply")
        return bn.patch.apply_module_patches(model)

    def _prepare(wrapper: nn.Module, sample: torch.Tensor) -> torch.Tensor:
        calls.append("prepare")
        with torch.no_grad():
            return wrapper(sample)

    monkeypatch.setattr(bn, "load_model", _load)
    monkeypatch.setattr(bn, "round_to_storage", _round)
    monkeypatch.setattr(bn.patch, "apply", _apply)
    monkeypatch.setattr(bn.patch, "prepare", _prepare)
    return calls


class TestRoundingPoint:
    """丸めは `load_model` の直後・`patch.apply` の前（モジュール docstring の MUST）。"""

    def test_the_weights_are_rounded_between_the_load_and_the_patches(self, tiny_load):
        bn.load_wrapper(Path("/nonexistent"), TINY_SIZE, "f16")

        assert tiny_load == ["load", ("round", "f16"), "apply", "prepare"]

    def test_f32_does_not_round(self, tiny_load):
        _wrapper, rounding = bn.load_wrapper(Path("/nonexistent"), TINY_SIZE, "f32")

        assert rounding is None

    def test_an_f16_checkpoint_comes_out_bit_identical(self, tiny_load):
        """HR 相当（checkpoint が f16）: 丸めは恒等で、α / β も出力も f32 系列とビット一致する。"""
        plain, _ = bn.load_wrapper(Path("/nonexistent"), TINY_SIZE, "f32")
        half, rounding = bn.load_wrapper(Path("/nonexistent"), TINY_SIZE, "f16")
        pixel_values = CASES[0][1]

        assert rounding is not None and rounding.parameters > 0
        for got, expected in zip(_affine(half), _affine(plain), strict=True):
            assert bn._bit_exact(got, expected)
        with torch.no_grad():
            assert bn._bit_exact(half(pixel_values), plain(pixel_values))

    def test_rounding_after_the_patches_moves_alpha_and_beta(self):
        """故障注入: apply の**後**に丸めると、f32 格納の派生定数 α / β まで f16 の値へ動く。

        f16 で表せる BatchNorm 統計から導いた α = w·rsqrt(var+eps) は一般に f16 で表せないので、
        後から丸めると値が動く — 格納は f32 のままなので VRAM は減らず数値だけが動く形。
        HR 相当の checkpoint ではこの差が品質の門（無損失 = ビット一致）で落ちることも見る。
        """
        before = _f16_checkpoint_model()
        bn.round_to_storage(before, "f16")
        bn.patch.apply_module_patches(before)
        after = _f16_checkpoint_model()
        bn.patch.apply_module_patches(after)
        bn.round_weights_to_f16(after)

        alpha_before, beta_before = _affine(before)
        alpha_after, beta_after = _affine(after)
        assert not bn._bit_exact(alpha_before, alpha_after)
        assert not bn._bit_exact(beta_before, beta_after)

        pixel_values = CASES[0][1]
        with torch.no_grad():
            reference = {"disc": before(pixel_values)[0]}
            misplaced = {"disc": after(pixel_values)[0]}
        quality = bn.measure_quality("f16", {"F16": 5}, reference, misplaced)
        with pytest.raises(AssertionError, match="ビット一致しない"):
            bn.assert_quality_gate(quality)

    def test_an_unknown_dtype_is_refused_before_touching_the_weights(self):
        model = _f16_checkpoint_model()
        snapshot = {name: tensor.clone() for name, tensor in model.state_dict().items()}

        with pytest.raises(SystemExit, match="書き出せない"):
            bn.round_to_storage(model, "i8")
        for name, tensor in model.state_dict().items():
            assert torch.equal(tensor, snapshot[name]), name


class TestCheckpointCensus:
    def test_it_counts_the_stored_dtypes_from_the_header(self, tmp_path):
        save_file(
            {
                "a": torch.zeros(2, dtype=torch.float16),
                "b": torch.zeros(3, dtype=torch.float16),
                "c": torch.zeros(1, dtype=torch.int64),
            },
            str(tmp_path / bn.CHECKPOINT_FILE),
        )

        assert bn.checkpoint_dtypes(tmp_path) == {"F16": 2, "I64": 1}

    def test_a_missing_checkpoint_fails_loudly(self, tmp_path):
        with pytest.raises(SystemExit, match="census"):
            bn.checkpoint_dtypes(tmp_path)

    @pytest.mark.parametrize(
        ("census", "lossless"),
        [
            ({"F16": 687, "I64": 67}, True),
            ({"F32": 687, "I64": 67}, False),
            ({"F16": 1, "F32": 1}, False),
            ({"F16": 1, "BF16": 1}, False),
            # 知らない綴りは浮動小数とみなす（無損失の主張を立てない側へ倒す）。
            ({"F16": 1, "F8_E8M0": 1}, False),
            ({"I64": 3}, False),
            ({}, False),
        ],
    )
    def test_f16_is_lossless_only_when_every_float_is_f16(self, census, lossless):
        assert bn.lossless_under("f16", census) is lossless

    def test_f32_never_rounds(self):
        assert bn.lossless_under("f32", {"F32": 1})

    def test_another_compressed_dtype_has_no_lossless_verdict(self):
        """「全て F16 なら無損失」は f16 だけの判定。

        段 2 の i8 へ流用すると、HR の i8 系列にビット一致を要求する門になる。
        """
        with pytest.raises(ValueError, match="i8"):
            bn.lossless_under("i8", {"F16": 687, "I64": 67})

    @pytest.mark.parametrize("checkpoint", ["hr", "lucida"])
    def test_the_card_names_the_same_storage_as_the_real_header(self, checkpoint: str):
        """カードの `stored_dtype`（manifest に無い事実）と、export の門が読むヘッダの一致。

        実重みの無い機では SKIP（`inputs/` はリポ管理外）。食い違うと、カードと NOTICE が
        「無損失」と名乗る席を品質の門は有損失として扱う（またはその逆）。
        """
        from birefnet.card import BIREFNET_CHECKPOINTS

        entry = BIREFNET_CHECKPOINTS[checkpoint]
        model_dir = bn.MODELS_ROOT / entry.repo.split("/", 1)[1]
        if not (model_dir / bn.CHECKPOINT_FILE).is_file():
            pytest.skip(f"{model_dir} に実重みが無い")

        census = bn.checkpoint_dtypes(model_dir)

        assert {name for name in census if name not in bn._NON_FLOAT_DTYPES} == {
            entry.stored_dtype.upper()
        }
        assert bn.lossless_under("f16", census) is (entry.stored_dtype == "f16")


def _logits(*values: float) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).reshape(1, 1, 1, len(values))


class TestQualityMeasure:
    def test_identical_outputs_measure_zero(self):
        reference = {"disc": _logits(-3.0, 0.5, 4.0)}

        quality = bn.measure_quality("f16", {"F32": 1}, reference, {"disc": reference["disc"]})

        case = quality["cases"]["disc"]
        assert case["bit_exact"] is True
        assert case["max_abs"] == 0.0
        assert case["mask_mismatch"] == 0.0
        assert case["alpha_max_lsb"] == 0

    def test_it_measures_the_logit_mask_and_alpha_differences(self):
        """符号の反転（境界画素）と、8 bit α の段数の動きを別々に数える。"""
        reference = {"disc": _logits(-0.01, 2.0, 0.0, 5.0)}
        rounded = {"disc": _logits(0.01, 2.0, 0.0, 5.5)}

        case = bn.measure_quality("f16", {"F32": 1}, reference, rounded)["cases"]["disc"]

        assert case["bit_exact"] is False
        assert case["max_abs"] == pytest.approx(0.5)
        # 4 画素中 1 画素（-0.01 → 0.01）だけマスクが反転する。
        assert case["mask_mismatch"] == pytest.approx(0.25)
        # 8 bit α: 反転した画素は 127 → 128 の 1 段、5.0 → 5.5 は 254 → 255 の 1 段。
        assert case["alpha_max_lsb"] == 1
        assert case["alpha_lsb_changed"] == pytest.approx(0.5)
        assert case["rel_rms"] > 0.0

    def test_signed_zero_is_not_bit_exact(self):
        """`torch.equal` は `0.0 == -0.0` を真にする — 無損失の門はビット列で見る。"""
        case = bn.measure_quality(
            "f16", {"F16": 1}, {"disc": _logits(0.0, 1.0)}, {"disc": _logits(-0.0, 1.0)}
        )["cases"]["disc"]

        assert case["max_abs"] == 0.0
        assert case["bit_exact"] is False

    def test_the_worst_row_takes_the_maximum_over_the_cases(self):
        reference = {"disc": _logits(1.0, 2.0), "ramp": _logits(-1.0, 3.0)}
        rounded = {"disc": _logits(1.0, 2.25), "ramp": _logits(-1.0, 3.0)}

        quality = bn.measure_quality("f16", {"F32": 1}, reference, rounded)

        assert quality["worst"]["max_abs"] == pytest.approx(0.25)
        assert quality["worst"]["bit_exact"] is False
        assert quality["cases"]["ramp"]["bit_exact"] is True

    def test_the_case_sets_must_match(self):
        """片方だけのケースを黙って落とすと、落ちたケースの劣化が記録から消える。"""
        with pytest.raises(AssertionError, match="ケースが食い違う"):
            bn.measure_quality("f16", {"F32": 1}, {"disc": _logits(1.0)}, {"ramp": _logits(1.0)})

    @pytest.mark.parametrize(
        ("census", "gate"), [({"F16": 1, "I64": 1}, "bit-exact"), ({"F32": 1}, "recorded")]
    )
    def test_the_expectation_comes_from_the_checkpoint_not_from_the_result(self, census, gate):
        """期待（無損失か）は上流の格納から引く — 丸めた結果から導くと門が恒真になる。"""
        reference = {"disc": _logits(1.0)}

        quality = bn.measure_quality("f16", census, reference, {"disc": _logits(1.5)})

        assert quality["gate"] == gate
        assert quality["checkpoint_dtypes"] == census


class TestQualityGate:
    def test_a_lossless_series_passes_when_every_case_is_bit_exact(self):
        reference = {"disc": _logits(1.0, -2.0), "ramp": _logits(0.5)}

        bn.assert_quality_gate(bn.measure_quality("f16", {"F16": 1}, reference, dict(reference)))

    def test_a_lossless_series_fails_on_any_moved_case(self):
        """HR 相当で 1 ケースでも動いたら公開しない（丸めの位置の退行を捕まえる）。"""
        reference = {"disc": _logits(1.0, -2.0), "ramp": _logits(0.5)}
        rounded = {"disc": _logits(1.0, -2.0), "ramp": _logits(0.5000001)}

        with pytest.raises(AssertionError, match=r"ビット一致しない: \['ramp'\]"):
            bn.assert_quality_gate(bn.measure_quality("f16", {"F16": 1}, reference, rounded))

    def test_the_lossy_limit_is_half_an_8bit_alpha_step(self):
        """暫定線は 8 bit α の 0.5 段（深掘り E1 §6 の kill 基準・ADR 0113 決定 5）。"""
        assert pytest.approx(0.5 / 255) == bn.ALPHA_MAE_LIMIT

    def test_a_lossy_series_below_the_alpha_limit_passes(self):
        """logit 0 の近傍（α の傾き 1/4）で 0.004 動かす — α は約 1e-3 動く（線の約半分）。"""
        reference = {"disc": _logits(0.0, 0.0), "ramp": _logits(0.0)}
        rounded = {"disc": _logits(0.004, 0.004), "ramp": _logits(0.0)}
        quality = bn.measure_quality("f16", {"F32": 1}, reference, rounded)
        assert 0.0 < quality["worst"]["alpha_mae"] < bn.ALPHA_MAE_LIMIT

        bn.assert_quality_gate(quality)

    def test_a_lossy_series_over_the_alpha_limit_fails(self):
        """同じ形で 0.02 動かす — α は約 5e-3 動き、線（≈ 1.96e-3）を超える。"""
        reference = {"disc": _logits(0.0, 0.0), "ramp": _logits(0.0)}
        rounded = {"disc": _logits(0.02, 0.02), "ramp": _logits(0.0)}
        quality = bn.measure_quality("f16", {"F32": 1}, reference, rounded)
        assert quality["worst"]["alpha_mae"] > bn.ALPHA_MAE_LIMIT

        with pytest.raises(AssertionError, match=r"暫定線 .*: \['disc'\]"):
            bn.assert_quality_gate(quality)

    def test_a_lossy_series_that_did_not_move_at_all_fails(self):
        """対照: 有損失のはずの丸めで 1 ビットも動かないのは、器が同じ重みを見ている形。"""
        reference = {"disc": _logits(1.0, -2.0)}

        with pytest.raises(AssertionError, match="恒真"):
            bn.assert_quality_gate(
                bn.measure_quality("f16", {"F32": 1}, reference, dict(reference))
            )


class TestExportSeriesQuality:
    """`export_series` の配線: 基準を丸めの前に採り、品質の門を公開の前に掛ける。

    重い段（実重みの読み込み・Swin 向けのパッチ）は tiny へ差し替える（{@link tiny_load}）。
    """

    #: `export_series` は入口で解像度の刻み（64）を見るので、tiny でも 64² で回す。
    SIZE = 64

    @pytest.fixture
    def wired(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tiny_load):
        """出所・census・実画像を tiny に差し替えた `export_series` の実走環境。"""
        monkeypatch.setattr(bn, "upstream_provenance", lambda _dir: TINY_PROVENANCE)
        monkeypatch.setattr(bn, "checkpoint_dtypes", lambda _dir: {"F16": 5, "I64": 1})
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        _write_real_images(corpus)
        # tiny は `disc` の円も実画像の前景も分離できない（判別の sanity は実物の領分）ので、
        # 順序の検査は外す（実画像の順序検査が**呼ばれること**は個別のテストが見る）。
        monkeypatch.setattr(bn, "_sanity", lambda mattes: {"cases": sorted(mattes)})
        monkeypatch.setattr(bn, "_real_sanity", lambda ratios: dict(ratios))
        return tmp_path, corpus

    def test_f16_writes_the_quality_record_next_to_the_golden(self, wired):
        tmp_path, corpus = wired
        out_dir = tmp_path / "series-f16"

        summary = bn.export_series(
            Path("/nonexistent"), out_dir, self.SIZE, corpus_root=corpus, dtype="f16"
        )

        quality = json.loads((out_dir / bn.QUALITY_FILE).read_text(encoding="utf-8"))
        # 品質計測は `--real-images` 無しでも合成 4 + 実画像 4 の 8 ケース。
        expected = sorted(
            [name for name, _ in bn.build_cases(self.SIZE)]
            + [name for name, _file, _why in bn.REAL_CASES]
        )
        assert sorted(quality["cases"]) == expected
        assert quality["gate"] == "bit-exact"
        assert quality["worst"]["bit_exact"] is True
        # golden には実画像を入れない（`real_images` 無し）。
        assert all(not name.startswith(f"{bn.IO_PREFIX}photo-") for name in summary["io"])
        assert summary["dtype"] == "f16"
        assert summary["quality"]["gate"] == "bit-exact"
        assert summary["storage"]["compressed_tensors"] > 0

    def test_the_reference_is_taken_before_the_weights_are_rounded(self, wired, tiny_load):
        tmp_path, corpus = wired

        bn.export_series(
            Path("/nonexistent"),
            tmp_path / "series-f16",
            self.SIZE,
            corpus_root=corpus,
            dtype="f16",
        )

        # 1 本目（基準）は丸めを通らない f32 のラッパ、2 本目が丸める側。
        assert tiny_load == [
            "load",
            ("round", "f32"),
            "apply",
            "prepare",
            "load",
            ("round", "f16"),
            "apply",
            "prepare",
        ]

    def test_a_failed_quality_gate_publishes_nothing(self, wired, monkeypatch):
        """MUST: 門は公開の前 — 落ちた実走が「検収門を通れる資産」を残さない。"""
        tmp_path, corpus = wired
        # 故障注入: 基準だけ別の重みで採る（HR 相当なのに差が出る形）。
        real_reference = bn.reference_outputs
        monkeypatch.setattr(
            bn,
            "reference_outputs",
            lambda *args: {name: out + 1.0 for name, out in real_reference(*args).items()},
        )
        out_dir = tmp_path / "series-f16"

        with pytest.raises(AssertionError, match="ビット一致しない"):
            bn.export_series(
                Path("/nonexistent"), out_dir, self.SIZE, corpus_root=corpus, dtype="f16"
            )

        assert not out_dir.exists()

    def test_a_lossy_checkpoint_is_recorded_and_published(self, wired, monkeypatch):
        """有損失の配線: census が F32・重みが f16 で表せない tiny（Lucida 相当）。

        門は `recorded`（ビット一致を期待しない）で、差は線の内側なので公開され、
        `quality.json` にビット不一致が残る。
        """
        tmp_path, corpus = wired
        monkeypatch.setattr(bn, "checkpoint_dtypes", lambda _dir: {"F32": 5, "I64": 1})
        monkeypatch.setattr(bn, "load_model", lambda _dir: _f32_checkpoint_model())
        out_dir = tmp_path / "series-f16"

        summary = bn.export_series(
            Path("/nonexistent"), out_dir, self.SIZE, corpus_root=corpus, dtype="f16"
        )

        quality = json.loads((out_dir / bn.QUALITY_FILE).read_text(encoding="utf-8"))
        assert quality["gate"] == "recorded"
        assert quality["lossless_expected"] is False
        assert quality["worst"]["bit_exact"] is False
        assert 0.0 < quality["worst"]["alpha_mae"] < bn.ALPHA_MAE_LIMIT
        assert summary["quality"]["gate"] == "recorded"

    def test_the_real_images_are_checked_without_real_images(self, wired, monkeypatch):
        """圧縮系列は `--real-images` 無しでも、丸めた実画像 4 枚の出力へ前景比の順序を掛ける。"""
        tmp_path, corpus = wired
        seen: list[set[str]] = []

        def _record(ratios):
            seen.append(set(ratios))
            return dict(ratios)

        monkeypatch.setattr(bn, "_real_sanity", _record)

        out_dir = tmp_path / "series-f16"

        summary = bn.export_series(
            Path("/nonexistent"), out_dir, self.SIZE, corpus_root=corpus, dtype="f16"
        )

        assert seen == [{name for name, _file, _why in bn.REAL_CASES}]
        assert "real_foreground" in summary["sanity"]

    def test_a_broken_real_image_order_publishes_nothing(self, wired, monkeypatch):
        tmp_path, corpus = wired

        def _broken(_ratios):
            raise AssertionError("実画像の前景比の順序が逆")

        monkeypatch.setattr(bn, "_real_sanity", _broken)
        out_dir = tmp_path / "series-f16"

        with pytest.raises(AssertionError, match="順序が逆"):
            bn.export_series(
                Path("/nonexistent"), out_dir, self.SIZE, corpus_root=corpus, dtype="f16"
            )

        assert not out_dir.exists()

    def test_f32_writes_no_quality_record(self, wired):
        tmp_path, corpus = wired
        out_dir = tmp_path / "series"

        summary = bn.export_series(Path("/nonexistent"), out_dir, self.SIZE, corpus_root=corpus)

        assert not (out_dir / bn.QUALITY_FILE).exists()
        assert summary["quality"] is None
        assert summary["storage"]["compressed_tensors"] == 0
