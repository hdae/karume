"""`minicpm5/sweep_w4.py` の量子化式（ADR 0069 決定 3）の振る舞い。

実重み・波 E 資産・transformers はどれも要らない**純数式のテスト**。sweep 本体（模型を読んで
測る側）は手動実行で、ここで固定するのは「壊れると測定値が黙って別物になる」側の性質だけ:

- 対称が冪等で、group の amax 要素を厳密復元し、`q` が −8 へ落ちないこと（格納形の前提そのもの）
- 全ゼロ group が 0 のまま戻ること（下限 clamp 経路 — 素の式なら NaN）
- scale が **group ごとに**分かれること（大 amax の group が小 amax の group を潰さない）
- 整除違反が fail loudly（端数 group を作った形の品質を測らない）
- 非対称が min / max を復元し `u ∈ [1,15]` に収まること、縮退 group が定数のまま戻ること
- 非対称が有利な分布で実際に対称より誤差が小さいこと（**恒真でない**ことの裏取り）
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import nn

from karume.quantize import QuantizeError
from minicpm5 import export as one_shot
from minicpm5 import sweep_w4 as sweep


def relative_error(weight: torch.Tensor, fq: torch.Tensor) -> float:
    """`‖w − fq‖₂ / ‖w‖₂`（sweep が族別に採るのと同じ量）。"""
    return float(torch.linalg.vector_norm(fq - weight) / torch.linalg.vector_norm(weight))


class TestSymmetricFakeQuant:
    def test_reapplying_it_is_bit_identical(self):
        """MUST: 冪等（ADR 0069 決定 3）— 崩れると丸め済みの重みが再量子化のたびに動く。"""
        torch.manual_seed(0)
        weight = torch.randn(5, 128)

        once = sweep.fake_quant_symmetric(weight, 32)
        twice = sweep.fake_quant_symmetric(once, 32)

        assert torch.equal(once, twice)

    def test_the_extreme_of_each_group_is_restored_exactly(self):
        """amax 要素が `q = ±7` に乗って `q·s` で厳密に戻る（冪等の理由そのもの）。"""
        weight = torch.tensor([[0.5, -3.0, 1.25, 0.75, -0.5, 2.0, 8.0, -1.0]])

        fq = sweep.fake_quant_symmetric(weight, 4)

        # group 0 の amax は -3.0（1 番目）、group 1 の amax は 8.0（2 番目）。
        assert float(fq[0, 1]) == -3.0
        assert float(fq[0, 6]) == 8.0

    def test_an_all_zero_group_stays_zero(self):
        """下限 clamp が無いと `amax / 7 = 0` で割って NaN が出る（全ゼロは実在する）。"""
        weight = torch.zeros(2, 16)

        fq = sweep.fake_quant_symmetric(weight, 8)

        assert torch.equal(fq, torch.zeros(2, 16))

    def test_the_levels_never_reach_minus_eight(self):
        """MUST: `q ∈ [−7, +7]`。−8 を許すと amax 要素の厳密復元（= 冪等）が崩れる。"""
        torch.manual_seed(1)
        weight = torch.randn(3, 64)

        q, _ = sweep.symmetric_components(weight, 16)

        assert int(q.min()) >= -sweep.INT4_MAX
        assert int(q.max()) <= sweep.INT4_MAX
        # 恒真化の防止: 両端が実際に踏まれている（`clamp` の値域だけを見ていない）。
        assert int(q.min()) == -sweep.INT4_MAX
        assert int(q.max()) == sweep.INT4_MAX


#: 振幅が 4 桁違う 2 group（per-tensor / per-channel の scale なら小さい側が全部 0 に潰れる）。
LOUD_GROUP = [100.0, -80.0, 60.0, -40.0]
QUIET_GROUP = [0.01, -0.008, 0.006, -0.004]


class TestGroupBoundaries:
    """group ごとに scale が分かれること — 「K 方向 group」の存在意義そのもの。"""

    def test_each_group_gets_its_own_scale(self):
        weight = torch.tensor([LOUD_GROUP + QUIET_GROUP])

        _, scale = sweep.symmetric_components(weight, 4)

        assert scale.shape == (1, 2, 1)
        assert float(scale[0, 0, 0]) == pytest.approx(100.0 / sweep.INT4_MAX)
        assert float(scale[0, 1, 0]) == pytest.approx(0.01 / sweep.INT4_MAX)

    def test_the_quiet_group_survives_the_loud_one(self):
        """大 amax group の誤差が小 amax group へ漏れない（漏れると静かに 0 へ潰れる）。"""
        weight = torch.tensor([LOUD_GROUP + QUIET_GROUP])

        fq = sweep.fake_quant_symmetric(weight, 4)

        quiet = fq[0, 4:]
        assert bool((quiet != 0).all())
        # 量子化誤差の上界は s/2 = amax/14（group 内の最大振幅で決まる）。
        assert torch.allclose(quiet, torch.tensor(QUIET_GROUP), atol=0.01 / 14)


class TestDivisibility:
    def test_a_group_size_that_does_not_divide_the_in_axis_fails_loudly(self):
        """MUST: 端数 group は格納できない形（ADR 0069 決定 2）— その品質を測らない。"""
        weight = torch.zeros(2, 100)

        with pytest.raises(QuantizeError, match="割り切れない"):
            sweep.fake_quant_symmetric(weight, 32)


class TestAsymmetricFakeQuant:
    def test_the_group_extremes_are_restored(self):
        """`u = 1` が min に、`u = 15` が max に乗る（非対称の値域の使い切り）。"""
        weight = torch.tensor([[2.0, 3.5, 9.0, 4.25, -6.0, -1.0, 0.5, -2.5]])

        fq = sweep.fake_quant_asymmetric(weight, 4)

        assert float(fq[0, 0]) == pytest.approx(2.0)  # group 0 の min
        assert float(fq[0, 2]) == pytest.approx(9.0)  # group 0 の max
        assert float(fq[0, 4]) == pytest.approx(-6.0)  # group 1 の min
        assert float(fq[0, 6]) == pytest.approx(0.5)  # group 1 の max

    def test_the_levels_stay_in_one_to_fifteen(self):
        """0 は未使用の 15 準位（対称と同じ値域 — pack 形式は zero-point の有無で変わらない）。"""
        torch.manual_seed(2)
        weight = torch.randn(3, 64) * 5.0 + 2.0

        u, _, _ = sweep.asymmetric_components(weight, 16)

        assert int(u.min()) == sweep.UINT4_MIN
        assert int(u.max()) == sweep.UINT4_MAX

    def test_a_degenerate_group_stays_constant(self):
        """MUST: `max == min` は対称式へ落とす。min-max 式だと定数 group が ~0 へ潰れる。"""
        weight = torch.tensor([[3.5, 3.5, 3.5, 3.5, -2.0, -2.0, -2.0, -2.0]])

        fq = sweep.fake_quant_asymmetric(weight, 4)

        assert torch.allclose(fq, weight)

    def test_a_degenerate_zero_group_stays_zero(self):
        """縮退かつ全ゼロ（`|c|/7` も下限 clamp へ落ちる経路）。"""
        fq = sweep.fake_quant_asymmetric(torch.zeros(2, 8), 4)

        assert torch.equal(fq, torch.zeros(2, 8))

    def test_it_beats_the_symmetric_form_on_a_one_sided_group(self):
        """非対称の測定列が意味を持つ条件 — 片側に寄った group では実際に誤差が小さい。

        対称は `[−amax, amax]` に準位を張るので、全正の group では準位の半分が空振りする。
        ここが緑にならない実装は「非対称の上界」を測れていない（sweep の列が恒真になる）。
        """
        torch.manual_seed(3)
        weight = torch.rand(4, 64) + 10.0

        symmetric = relative_error(weight, sweep.fake_quant_symmetric(weight, 32))
        asymmetric = relative_error(weight, sweep.fake_quant_asymmetric(weight, 32))

        assert asymmetric < symmetric / 10


# ---- 校正付き丸め（波 J-2）--------------------------------------------------


class TinyAttention(nn.Module):
    """`q_proj` / `o_proj` の 2 本（族名は {@link sweep.LINEAR_FAMILIES} の綴りに合わせる）。"""

    def __init__(self, features: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(features, features, bias=False)
        self.o_proj = nn.Linear(features, features, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.o_proj(self.q_proj(hidden))


#: 全層へ同じ値で渡す kwargs（Llama の mask / position_ids 相当の代役）。
TINY_SHIFT = 0.0


class TinyLayer(nn.Module):
    """decoder layer 相当の stage（kwargs の効きは `shift` の**加算**で観測する）。"""

    def __init__(self, features: int) -> None:
        super().__init__()
        self.self_attn = TinyAttention(features)

    def forward(self, hidden: torch.Tensor, shift: float = 0.0) -> torch.Tensor:
        return self.self_attn(hidden + shift)


@dataclass(frozen=True)
class TinyInnerOutput:
    """`BaseModelOutputWithPast` の代役（`last_hidden_state` だけを持つ）。"""

    last_hidden_state: torch.Tensor


class TinyInner(nn.Module):
    """`LlamaModel` 相当（`.layers` / `.embed_tokens` / `.norm`）。

    `forward` は上流と同じ「embedding → 層を順に → 最終 norm」で、返り値も同じ席
    （`last_hidden_state`）— {@link sweep.assert_stage_split} の突合先がこれ。
    """

    def __init__(self, vocab: int, features: int, layers: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, features)
        self.layers = nn.ModuleList(TinyLayer(features) for _ in range(layers))
        self.norm = nn.LayerNorm(features)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> TinyInnerOutput:
        """kwargs は**全層で同一**（Llama の形 — `calibrate_stages` が stage 間で不変に運ぶ前提）。

        層どうしの違いは各 `TinyLayer` の重み（乱数初期化）が持つので、段落としは kwargs に
        依らず検出できる。
        """
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden, shift=TINY_SHIFT)
        return TinyInnerOutput(self.norm(hidden))


class TinyCausal(nn.Module):
    """`LlamaForCausalLM` 相当（`.model` が `TinyInner`）。"""

    def __init__(self, vocab: int, features: int, layers: int) -> None:
        super().__init__()
        self.model = TinyInner(vocab, features, layers)


class TinyWrapper(nn.Module):
    """`CausalLmWrapper` 相当 — `sweep.decoder_stages` が辿る `model.model.layers` の形。"""

    def __init__(self, vocab: int = 8, features: int = 32, layers: int = 2) -> None:
        super().__init__()
        self.model = TinyCausal(vocab, features, layers)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = one_shot.additive_causal_mask(int(input_ids.shape[1]))
        return self.model.model(
            input_ids=input_ids, attention_mask=mask, use_cache=False
        ).last_hidden_state


class SilentWrapper(TinyWrapper):
    """先頭 stage を**呼ばない** forward（Catcher の fail loudly を踏ませる形）。"""

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.model.embed_tokens(input_ids)


def tiny_inputs(count: int = 3) -> tuple[torch.Tensor, ...]:
    """長さの違う id 列（校正入力の代役 — tokenizer は通さない）。"""
    return tuple(
        torch.arange(2 + index, dtype=torch.int64).unsqueeze(0) % 8 for index in range(count)
    )


def tiny_rig(wrapper: TinyWrapper, inputs: tuple[torch.Tensor, ...]) -> sweep.CalibRig:
    stages = sweep.decoder_stages(wrapper)
    scan, counts = sweep.calib_targets(stages)
    return sweep.CalibRig(
        stages=stages,
        scan=scan,
        counts=counts,
        batches=sweep.capture_stage_batches(wrapper, stages[0][1], inputs),
    )


class TestCalibConfigs:
    def test_every_calib_config_is_selectable(self):
        """`--only` の選択肢（= `ALL_CONFIGS`）に校正グリッドが載る。"""
        names = [config.name for config in sweep.ALL_CONFIGS]

        assert all(config.name in names for config in sweep.CALIB_CONFIGS)

    def test_selecting_a_calib_config_keeps_the_baseline_first(self):
        """MUST: 部分再実行でも baseline の sanity 門を先に通す（既存の流儀）。"""
        chosen = sweep.select_configs(["gptq-rtn"])

        assert [config.name for config in chosen] == [sweep.BASELINE_NAME, "gptq-rtn"]

    def test_the_group_size_is_pinned_for_every_calib_config(self):
        """校正は格子を変えない — g 軸を振らないのは方式グリッドと同文。"""
        assert all(
            config.grid.group_size == sweep.CALIB_GROUP_SIZE for config in sweep.CALIB_CONFIGS
        )

    def test_no_calib_config_fits_its_table_on_a_subsample_by_default(self):
        """既定は全量 fit（部分標本は `--kmeans-shared-stride` を渡したときだけ）。"""
        assert all(config.grid.fit_stride == 1 for config in sweep.CALIB_CONFIGS)

    def test_the_label_names_both_the_method_and_the_grid(self):
        """表の「方式」列は方式と格納グリッドの組（片方だけだと行が読めない）。"""
        labels = {config.name: config.label for config in sweep.CALIB_CONFIGS}

        assert labels["gptq-nf4"] == "gptq/nf4"
        assert labels["awq-gptq-rtn"] == "awq+gptq/rtn"

    def test_the_group_scale_grids_project_five_bits(self):
        """4bit + group 32 ごとの f32 scale = 5.0 bpw（方式グリッドと同じ格納）。"""
        counts = sweep.TargetCounts(modules=2, channels=8, elements=3200)

        assert sweep.group_scale_bits(counts) / counts.elements == pytest.approx(5.0)

    def test_the_calibrated_codebook_costs_one_table_per_layer(self):
        """core の `kmeans_shared` は**層内**の表（全体 1 枚の方式グリッドとは式が違う）。"""
        counts = sweep.TargetCounts(modules=2, channels=8, elements=3200)

        difference = sweep.layer_table_bits(counts) - sweep.group_scale_bits(counts)

        assert difference == sweep.CODEBOOK_BITS * counts.modules


class TestCalibScan:
    def test_it_lists_every_decoder_linear_with_a_model_wide_fqn(self):
        """台帳のキーを `Int4Report` と同じ FQN 空間へ揃えるための接頭辞（core の契約）。"""
        wrapper = TinyWrapper(layers=2)

        scan, counts = sweep.calib_targets(sweep.decoder_stages(wrapper))

        assert sorted(scan) == [
            "model.model.layers.0.self_attn.o_proj.weight",
            "model.model.layers.0.self_attn.q_proj.weight",
            "model.model.layers.1.self_attn.o_proj.weight",
            "model.model.layers.1.self_attn.q_proj.weight",
        ]
        assert counts.modules == 4

    def test_the_vocabulary_table_is_not_in_the_scan(self):
        """`embed_tokens` / `lm_head` は stage の外 — 校正の対象集合に入らない。"""
        scan, _ = sweep.calib_targets(sweep.decoder_stages(TinyWrapper()))

        assert not any(sweep.EMBED_TOKENS in fqn or sweep.LM_HEAD in fqn for fqn in scan)


class TestCatcher:
    def test_it_captures_one_batch_per_calibration_text(self):
        wrapper = TinyWrapper()
        inputs = tiny_inputs(3)

        batches = sweep.capture_stage_batches(wrapper, wrapper.model.model.layers[0], inputs)

        assert len(batches) == 3
        assert [int(args[0].shape[1]) for args, _kwargs in batches] == [2, 3, 4]

    def test_it_captures_the_keyword_arguments_of_the_call(self):
        """kwargs（mask / position_ids …）を捨てると stage の再実行が別の数になる。"""
        wrapper = TinyWrapper()

        batches = sweep.capture_stage_batches(
            wrapper, wrapper.model.model.layers[0], tiny_inputs(1)
        )

        assert batches[0][1] == {"shift": 0.0}

    def test_the_hook_is_removed_even_though_the_forward_was_aborted(self):
        """番兵で打ち切っても後始末は済む（残ると以後の forward が全部落ちる）。"""
        wrapper = TinyWrapper()
        first = wrapper.model.model.layers[0]

        sweep.capture_stage_batches(wrapper, first, tiny_inputs(1))

        assert not first._forward_pre_hooks
        with torch.no_grad():
            wrapper(tiny_inputs(1)[0])

    def test_a_forward_that_never_reaches_the_first_stage_fails_loudly(self):
        """MUST: 素通りを黙って通すと「校正入力ゼロ」の診断まで見えない。"""
        wrapper = SilentWrapper()

        with pytest.raises(AssertionError, match="完走"):
            sweep.capture_stage_batches(wrapper, wrapper.model.model.layers[0], tiny_inputs(1))


class TestCalibrationRun:
    def test_it_rounds_every_scanned_linear(self):
        """校正の丸めが走査の全本へ届く（届かない層があれば門が落ちる）。"""
        torch.manual_seed(0)
        wrapper = TinyWrapper()
        rig = tiny_rig(wrapper, tiny_inputs(2))
        before = {fqn: weight.detach().clone() for fqn, weight in rig.scan.items()}

        report = sweep.apply_calib(sweep.CALIB_CONFIGS[0], rig, shared_stride=0)
        sweep.assert_calib_covers_scan(report, rig.scan, "gptq-rtn")

        assert report.modules == len(rig.scan)
        assert all(not torch.equal(rig.scan[fqn], value) for fqn, value in before.items())

    def test_every_calib_config_runs_end_to_end(self):
        """5 本とも core の駆動へ通る（方式と格納グリッドの組が全部生きている）。"""
        for config in sweep.CALIB_CONFIGS:
            torch.manual_seed(1)
            wrapper = TinyWrapper()
            rig = tiny_rig(wrapper, tiny_inputs(2))

            report = sweep.apply_calib(config, rig, shared_stride=0)
            sweep.assert_calib_covers_scan(report, rig.scan, config.name)

            assert report.method == config.method
            assert report.grid == config.grid.kind

    def test_a_subsampled_table_is_reported(self):
        """MUST: `fit_stride` を使ったら出力へ明記（`describe` が拾う）。"""
        torch.manual_seed(2)
        wrapper = TinyWrapper()
        rig = tiny_rig(wrapper, tiny_inputs(2))
        kmeans = next(config for config in sweep.CALIB_CONFIGS if config.name == "gptq-kmeans")

        report = sweep.apply_calib(kmeans, rig, shared_stride=4)

        assert "fit_stride 4" in report.describe()

    def test_a_scan_the_calibration_missed_fails_loudly(self):
        """恒真化の遮断 — 走査に無い / 丸め漏れの層があれば門が落ちる。"""
        torch.manual_seed(3)
        wrapper = TinyWrapper()
        rig = tiny_rig(wrapper, tiny_inputs(2))
        report = sweep.apply_calib(sweep.CALIB_CONFIGS[0], rig, shared_stride=0)
        widened = {**rig.scan, "model.model.layers.9.self_attn.q_proj.weight": torch.zeros(1)}

        with pytest.raises(AssertionError, match="一致しない"):
            sweep.assert_calib_covers_scan(report, widened, "gptq-rtn")


class TestStageSplit:
    """分解一致門（{@link sweep.assert_stage_split}）— 校正が「本物と同じ活性」を見る前提。"""

    def test_the_sequential_stages_reproduce_the_model_bit_for_bit(self):
        torch.manual_seed(4)
        wrapper = TinyWrapper()
        inputs = tiny_inputs(1)
        stages = sweep.decoder_stages(wrapper)
        batches = sweep.capture_stage_batches(wrapper, stages[0][1], inputs)

        sweep.assert_stage_split(wrapper, inputs[0], batches[0], stages)

    def test_a_dropped_stage_is_caught(self):
        """段が 1 枚落ちても数値は普通に出る — 一致門だけがそれを落とす。"""
        torch.manual_seed(5)
        wrapper = TinyWrapper()
        inputs = tiny_inputs(1)
        stages = sweep.decoder_stages(wrapper)
        batches = sweep.capture_stage_batches(wrapper, stages[0][1], inputs)

        with pytest.raises(AssertionError, match="ビット一致しない"):
            sweep.assert_stage_split(wrapper, inputs[0], batches[0], stages[:-1])
