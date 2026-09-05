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
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn

from _shared.paths import SERIES_ROOT
from birefnet import export as bn
from karume.pipeline import export_to_file

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
    graph = export_to_file(wrapper, (CASES[0][1],), tmp_path / bn.MODEL_FILE, symbol_names=())
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

        配布 recipe は「モデル名 → 上流リポ名 → 系列ディレクトリ名」で系列を探す。式が
        片方だけ動くと、組み立ては「系列が無い」で落ちる（それ自体は安全）が、**別の
        モデルの系列を掴む**綴りにずれた場合は誰も気づけない。
        """
        from birefnet.card import BIREFNET_UPSTREAM
        from birefnet.distribution import BIREFNET_RESOLUTION, birefnet_series_name

        for model, repo in BIREFNET_UPSTREAM.items():
            model_dir = bn.MODELS_ROOT / repo.split("/", 1)[1]
            assert bn.default_out_dir(model_dir, BIREFNET_RESOLUTION).name == birefnet_series_name(
                model
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
        graph = export_to_file(wrapper, (CASES[0][1],), tmp_path / bn.MODEL_FILE, symbol_names=())

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
            lambda _dir, out, _resolution, real_images: (
                seen.append(real_images) or {"dir": str(out)}
            ),
        )

        bn.main(argv)

        assert seen == [expected]


class _RecordingMatte(nn.Module):
    """呼び出しの段（参照 / 段 1 / 段 2）を記録するだけの骨格。

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
    """`verify_patches` の順序不変条件（参照 → 段 1 → 段 2）。"""

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

        monkeypatch.setattr(bn.patch, "apply_layout_patches", _apply_layout)
        monkeypatch.setattr(bn.patch, "apply_module_patches", _apply_modules)
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
        ]
        assert [entry["stage"] for entry in entries] == ["layout", "modules"]
