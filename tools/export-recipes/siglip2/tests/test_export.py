"""`siglip2/export.py` の台本レベルの約束事（実重み不要分）。

実重みの emit は手動（既存 `deberta/export.py` / `embeddinggemma/export.py` のテストと同じ
規律）。ここで固定するのは、壊れると**偽 PASS** になる側の規律だけ:

- グラフ出力が **pooler_output 1 本**であること（2 本目が生えたら io の位置規約が黙ってずれる）
- golden の合成画像が**互いに違う**こと（同じ画像を 2 度使うと cosine 順序の sanity が恒真）
- `_sanity` が cosine の**順序**を見ること（ノルムも「同一入力どうしの cosine」も恒真）
- `--verify` が emit しないこと（同一プロセスでは参照が汚染される）
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn

from _shared.paths import SERIES_ROOT
from karume.pipeline import export_to_file
from siglip2 import export as sg


class TinyVisionPooler(nn.Module):
    """`VisionPooler` の最小の骨格（`[B,3,S,S] → [B,H]`・引数名まで同じ）。"""

    def __init__(self) -> None:
        super().__init__()
        self.patch = nn.Conv2d(3, 4, kernel_size=2, stride=2)
        self.head = nn.Linear(4, 4)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        patches = self.patch(pixel_values).flatten(2).transpose(1, 2)
        return self.head(torch.sum(patches, dim=1))


class TwoOutputPooler(TinyVisionPooler):
    """出力が 2 本ある形（`_write_io` が拒否することの確認用）。"""

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = super().forward(pixel_values)
        return pooled, pooled * 2.0


class Config:
    """`SiglipVisionConfig` のうち `build_cases` が見る欄だけ。"""

    num_channels = 3
    image_size = 32
    patch_size = 4


CASES = (
    ("case0", torch.linspace(-1.0, 1.0, 3 * 8 * 8).reshape(1, 3, 8, 8)),
    ("case1", torch.linspace(1.0, -1.0, 3 * 8 * 8).reshape(1, 3, 8, 8)),
)

#: `_sanity` の合成側を通す最小の出力（実画像側だけを動かすための土台）。
SYNTHETIC_POOLED = {
    "ramp": torch.tensor([[1.0, 0.0]]),
    "ramp-dim": torch.tensor([[0.9, 0.1]]),
    "checker": torch.tensor([[0.0, 1.0]]),
    "noise": torch.tensor([[0.5, 0.5]]),
}


def _real_pooled(street: list[float]) -> dict[str, torch.Tensor]:
    """実画像 4 ケースの出力（`photo-street` だけを動かす — 人物側が近いか風景側が近いか）。"""
    return {
        "photo-portrait": torch.tensor([[1.0, 0.0]]),
        "photo-street": torch.tensor([street]),
        "photo-landscape": torch.tensor([[0.0, 1.0]]),
        "photo-corridor": torch.tensor([[0.1, 0.99]]),
    }


@pytest.fixture
def exported(tmp_path):
    """tiny なラッパを 1 本 export して `(wrapper, graph, out_dir)` を返す。"""
    torch.manual_seed(0)
    wrapper = TinyVisionPooler()
    graph = export_to_file(wrapper, (CASES[0][1],), tmp_path / sg.MODEL_FILE, symbol_names=())
    return wrapper, graph, tmp_path


class TestSeriesLayout:
    def test_the_default_output_dir_is_a_series(self):
        """系列出力は `outputs/series/` 配下（配布形の `models/` ではない — _shared.paths）。"""
        assert sg.default_out_dir(sg.DEFAULT_MODEL_DIR).parent == SERIES_ROOT

    def test_each_model_gets_its_own_series(self):
        """MUST: モデルごとに別系列 — 綴りを共有すると先の資産が黙って上書きされる。"""
        other = sg.MODELS_ROOT / "siglip2-so400m-patch14-384"

        assert sg.default_out_dir(other) != sg.default_out_dir(sg.DEFAULT_MODEL_DIR)
        assert sg.default_out_dir(other).name == other.name


class TestWriteIo:
    def test_writes_one_file_per_case_with_the_declared_keys(self, exported):
        wrapper, graph, out_dir = exported

        written, pooled = sg._write_io(wrapper, graph, CASES, out_dir)

        assert written == [f"{sg.IO_PREFIX}{name}{sg.IO_SUFFIX}" for name, _ in CASES]
        tensors = load_file(str(out_dir / written[0]))
        assert set(tensors) == {f"{sg.INPUT_PREFIX}{sg.INPUT_NAME}", f"{sg.OUTPUT_PREFIX}0"}
        assert tuple(pooled["case0"].shape) == (1, 4)

    def test_more_than_one_graph_output_fails_loudly(self, tmp_path):
        """MUST: 出力は pooler_output 1 本 — 2 本目が生えると io の位置規約が黙ってずれる。"""
        torch.manual_seed(0)
        wrapper = TwoOutputPooler()
        graph = export_to_file(wrapper, (CASES[0][1],), tmp_path / sg.MODEL_FILE, symbol_names=())

        with pytest.raises(AssertionError, match="pooler_output は 1 本"):
            sg._write_io(wrapper, graph, CASES, tmp_path)


class TestGoldenCases:
    def test_cases_are_normalized_pixel_values_of_the_configured_shape(self):
        cases = sg.build_cases(Config())

        for name, pixel_values in cases:
            assert tuple(pixel_values.shape) == (1, 3, 32, 32), name
            assert pixel_values.dtype is torch.float32, name
            assert float(pixel_values.abs().max()) <= 1.0, name

    def test_every_case_is_a_different_image(self):
        """MUST: 同じ画像を 2 度使わない — cosine 順序の sanity が恒真になる。"""
        cases = sg.build_cases(Config())

        for index, (name, pixel_values) in enumerate(cases):
            for other_name, other in cases[index + 1 :]:
                assert not torch.equal(pixel_values, other), f"{name} と {other_name} が同一"

    def test_the_sanity_pairs_name_existing_cases(self):
        names = {name for name, _ in sg.build_cases(Config())}

        assert set(sg.NEAR_PAIR) | set(sg.FAR_PAIR) <= names


class TestRealCases:
    def test_every_case_names_a_distinct_image(self):
        """MUST: ケース名もファイル名も重複しない（同じ画像を 2 度使うと判別が恒真化する）。"""
        names = [name for name, _file, _why in sg.REAL_CASES]
        files = [file for _name, file, _why in sg.REAL_CASES]

        assert len(set(names)) == len(names)
        assert len(set(files)) == len(files)

    def test_case_names_do_not_collide_with_the_synthetic_ones(self):
        """golden は 1 ディレクトリに同居するので、綴りが衝突すると片方が黙って消える。"""
        synthetic = {name for name, _ in sg.build_cases(Config())}

        assert synthetic.isdisjoint({name for name, _file, _why in sg.REAL_CASES})

    def test_the_discrimination_groups_name_existing_cases(self):
        names = {name for name, _file, _why in sg.REAL_CASES}

        assert set(sg.REAL_PERSON_CASES) | set(sg.REAL_SCENE_CASES) <= names

    def test_a_missing_image_fails_loudly_before_anything_heavy(self, tmp_path):
        """MUST: 欠けを黙って飛ばさない（golden が 3 枚で書かれると e2e 側が欠けに気づく）。"""
        with pytest.raises(SystemExit, match="demo:eval-images"):
            sg.build_real_cases(tmp_path, Config(), tmp_path)


class TestIoMetadata:
    def test_real_cases_carry_the_source_image_digest(self, exported):
        """実画像 golden は元画像を同定できる（焼き直して採り直さない事故を落とすため）。"""
        wrapper, graph, out_dir = exported
        digest = "f" * 64
        metadata = {"case0": {sg.SOURCE_IMAGE_KEY: "a.png", sg.SOURCE_SHA256_KEY: digest}}

        written, _pooled = sg._write_io(wrapper, graph, CASES, out_dir, metadata)

        with safe_open(str(out_dir / written[0]), framework="pt") as opened:
            assert opened.metadata() == {sg.SOURCE_IMAGE_KEY: "a.png", sg.SOURCE_SHA256_KEY: digest}
        with safe_open(str(out_dir / written[1]), framework="pt") as opened:
            assert opened.metadata() is None


class TestSanity:
    def test_passes_when_the_near_pair_is_closer(self):
        pooled = {
            "ramp": torch.tensor([[1.0, 0.0]]),
            "ramp-dim": torch.tensor([[0.9, 0.1]]),
            "checker": torch.tensor([[0.0, 1.0]]),
            "noise": torch.tensor([[0.5, 0.5]]),
        }

        result = sg._sanity(pooled)

        assert result["cosine"][f"{sg.NEAR_PAIR[0]}×{sg.NEAR_PAIR[1]}"] > 0.9

    def test_fails_loudly_when_the_order_is_inverted(self):
        """構造の近い対が遠い対より遠いなら、埋め込みとして壊れている。"""
        pooled = {
            "ramp": torch.tensor([[1.0, 0.0]]),
            "ramp-dim": torch.tensor([[0.0, 1.0]]),
            "checker": torch.tensor([[0.9, 0.1]]),
            "noise": torch.tensor([[0.5, 0.5]]),
        }

        with pytest.raises(AssertionError, match="cosine の順序が構造と逆"):
            sg._sanity(pooled)

    def test_real_images_are_judged_only_when_they_are_present(self):
        """合成 4 ケースだけの emit（既定）でも sanity は通る（実画像は追加の群）。"""
        assert "real_cosine" not in sg._sanity(dict(SYNTHETIC_POOLED))

    def test_real_images_pass_when_the_two_people_are_closest(self):
        pooled = {**SYNTHETIC_POOLED, **_real_pooled(street=[0.99, 0.1])}

        result = sg._sanity(pooled)

        assert result["real_cosine"][f"{sg.REAL_PERSON_CASES[0]}×{sg.REAL_PERSON_CASES[1]}"] > 0.9

    def test_real_images_fail_loudly_when_a_scene_is_closer(self):
        """MUST: 人物どうしより人物×風景が近ければ、埋め込みは意味を捉えていない。"""
        pooled = {**SYNTHETIC_POOLED, **_real_pooled(street=[0.1, 0.99])}

        with pytest.raises(AssertionError, match="実画像の cosine の順序が逆"):
            sg._sanity(pooled)

    def test_fails_loudly_when_every_case_collapses_to_one_point(self):
        """MUST: 1 点へ潰れた埋め込みも落とす（近い対と遠い対が同値になる）。"""
        vector = torch.tensor([[0.6, 0.8]])
        pooled = dict.fromkeys(("ramp", "ramp-dim", "checker", "noise"), vector)

        with pytest.raises(AssertionError, match="cosine の順序が構造と逆"):
            sg._sanity(pooled)


class TestVerifyCli:
    def test_verify_does_not_emit(self, monkeypatch):
        """MUST: 同一プロセスで emit と併用しない（クラス差し替えが参照を汚染する）。"""
        seen: list[str] = []
        monkeypatch.setattr(
            sg,
            "verify_patches",
            lambda _dir: (
                seen.append("verify")
                or [
                    {"stage": "shape-folds", "claim": "bit-exact", "bit_exact": True, "maxdiff": {}}
                ]
            ),
        )
        monkeypatch.setattr(
            sg, "export_series", lambda *_a, **_kw: pytest.fail("--verify で emit された")
        )

        sg.main(["--verify"])

        assert seen == ["verify"]

    def test_without_verify_it_emits(self, monkeypatch):
        seen: list[str] = []
        monkeypatch.setattr(
            sg, "export_series", lambda *_a, **_kw: seen.append("emit") or {"dir": "x"}
        )
        monkeypatch.setattr(
            sg, "verify_patches", lambda _dir: pytest.fail("emit で --verify が走った")
        )

        sg.main([])

        assert seen == ["emit"]

    def test_the_output_dir_follows_the_model_dir(self, monkeypatch):
        """MUST: `--out` 未指定なら系列は `--model-dir` に追随する（固定だと上書きになる）。"""
        seen: list[Path] = []
        monkeypatch.setattr(
            sg, "export_series", lambda _dir, out, **_kw: seen.append(out) or {"dir": str(out)}
        )

        sg.main(["--model-dir", "/tmp/siglip2-so400m-patch14-384"])

        assert seen == [SERIES_ROOT / "siglip2-so400m-patch14-384"]
