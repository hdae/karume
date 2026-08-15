"""`anima/pipeline_ref.py` のうち**実重みが要らない**部分の約束事。

ここで固定するのは、壊れても実行時に例外が出ず**数だけが静かにずれる**側だけ:

- sigma スケジュールの形（shift が本当に効いている / 終端 0 / 単調減少 / f32）
- 逆正規化のチャネル軸（rank5 の軸 1 — 取り違えても要素数が合うので shape 検査を素通りする）
- プロンプト定数が conditioner の受理集合（`Dim("Ttgt", min=2)`）から外れないこと

実重みを要する emit / フィクスチャ生成は手動（README 参照）。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from anima import pipeline_ref as anima_pipeline


class TestSigmaSchedule:
    def test_shape_and_terminal_zero(self):
        """`step()` が `sigmas[i+1]` を読むので終端 0 が 1 つ余分に要る。"""
        sigmas = anima_pipeline.sigma_schedule(32, anima_pipeline.SHIFT)

        assert sigmas.dtype == np.float32
        assert len(sigmas) == 33
        assert sigmas[0] == pytest.approx(1.0)
        assert sigmas[-1] == 0.0

    def test_strictly_decreasing(self):
        sigmas = anima_pipeline.sigma_schedule(32, anima_pipeline.SHIFT)

        assert np.all(np.diff(sigmas) < 0)

    def test_shift_bends_the_schedule_upward(self):
        """shift>1 は sigma を**押し上げる**（`shift·s/(1+(shift−1)s) ≥ s`）。

        乗除を取り違える / shift を分母だけに掛ける類の誤りは、この向きが反転して落ちる。
        端点（s=1 と終端 0）は shift に依らず不動なので内側だけを見る。
        """
        plain = anima_pipeline.sigma_schedule(32, 1.0)
        shifted = anima_pipeline.sigma_schedule(32, 3.0)

        assert np.all(shifted[1:-1] > plain[1:-1])
        assert shifted[0] == plain[0]

    def test_two_steps_is_the_smallest_usable_schedule(self):
        """`linspace` は num=1 で 0 除算になる。下限が 2 であることを踏んでおく。"""
        sigmas = anima_pipeline.sigma_schedule(2, anima_pipeline.SHIFT)

        assert len(sigmas) == 3
        assert np.all(np.isfinite(sigmas))


class _FakeVaeConfig:
    def __init__(self, mean: list[float], std: list[float]) -> None:
        self.latents_mean = mean
        self.latents_std = std
        self.z_dim = len(mean)


class _FakeVae:
    def __init__(self, mean: list[float], std: list[float]) -> None:
        self.config = _FakeVaeConfig(mean, std)


class TestDenormalizeLatents:
    def test_channel_axis_is_one(self):
        """MUST: per-channel 定数はチャネル軸（rank5 の軸 1）に当たる。

        軸を取り違えても要素数は合う（broadcast が通ってしまう）ので、**チャネルごとに
        違う値**を配って初めて検出できる。
        """
        vae = _FakeVae([0.0, 10.0], [1.0, 1.0])
        latents = torch.zeros(1, 2, 3, 3)

        out = anima_pipeline.denormalize_latents(vae, latents)

        assert list(out.shape) == [1, 2, 1, 3, 3]
        assert torch.all(out[:, 0] == 0.0)
        assert torch.all(out[:, 1] == 10.0)

    def test_std_scales_before_the_mean_is_added(self):
        """`z/(1/std) + mean` — mean を先に足す実装は倍率が mean にも掛かってずれる。"""
        vae = _FakeVae([1.0], [4.0])
        latents = torch.full((1, 1, 1, 1), 2.0)

        out = anima_pipeline.denormalize_latents(vae, latents)

        assert out.item() == pytest.approx(9.0)


class TestPromptConstants:
    def test_neither_prompt_is_empty(self):
        """空プロンプトの T5 id 列は長さ 1 になり conditioner の `Dim(min=2)` から外れる。"""
        assert anima_pipeline.PROMPT.strip() != ""
        assert anima_pipeline.NEGATIVE_PROMPT.strip() != ""


class TestComponentDtype:
    """系列ごとのコンポーネント別格納 dtype（ADR 0019 — 資産系列と 1 対 1）。

    ずれると「フィクスチャの参照だけ別のモデルの数」になり、通しチェーンの差が
    量子化誤差と実装誤差の合成に化ける（tolerance を緩める方向にしか効かない）。
    """

    LABELS = ("text_encoder", "text_conditioner", "transformer", "vae")

    @pytest.mark.parametrize("label", LABELS)
    def test_f32_series_rounds_nothing(self, label):
        assert anima_pipeline.component_dtype("f32", label) == "f32"

    @pytest.mark.parametrize("label", LABELS)
    def test_f16_series_rounds_every_component(self, label):
        assert anima_pipeline.component_dtype("f16", label) == "f16"

    def test_i8_series_quantizes_only_the_transformer(self):
        """資産は `outputs/series/anima-i8/transformer` + `outputs/series/anima-f16/` の他 3 つ。"""
        assert anima_pipeline.component_dtype("i8", "transformer") == "i8"
        for label in ("text_encoder", "text_conditioner", "vae"):
            assert anima_pipeline.component_dtype("i8", label) == "f16", label

    def test_every_series_has_its_own_output_root(self):
        roots = anima_pipeline.DEFAULT_OUTS

        assert set(roots) == set(anima_pipeline.COMPONENT_DTYPES)
        assert len(set(roots.values())) == len(roots)
