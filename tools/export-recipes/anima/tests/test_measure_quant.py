"""`anima/measure_quant.py` の**実重みが要らない**部分の約束事。

実測そのものは手動（7.29GiB の DiT を構成ぶん丸めて回す）。ここで固定するのは、壊れると
実行時に例外が出ず**測定値が静かに嘘になる**側だけ:

- 10 構成の記号・順序・基準が**据え置き**であること（ADR 0030 / ACTIVE_DESIGN が構成名を
  名指しで参照する — 外部参照の安定）
- `--w4-screen` が**方式を積み重ねない**こと（戻さずに当てると「NF4 の上の MXFP4」を測る
  ことになり、方式比較そのものが成立しない）
- 対象集合の素性 — group 数を「丸めが group を割る軸」と同じ軸で数えること・g32 で割り
  切れない重みを除外して**一覧に載せる**こと・除外が丸めに実際に効くこと
- 非 linear の量子化可能型が現れたら fail loudly（linear 限定 = i4 の実行経路と一致、が
  この台本の前提）
- サイズ試算の式 — g32 の group scale 込みで rtn / nf4 が 5.0 bpw・mxfp4 が 4.25 bpw・
  kmeans:shared が表 1 枚ぶん上乗せ（表に載る数字の出どころ）
- `kmeans:shared` の fit_stride の導出（予算内は全量・大規模は group 長と互いに素な奇数）
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from anima import measure_quant as mq
from karume.quantize import iter_quant_targets


class TinyDit(nn.Module):
    """g32 で割り切れる linear 2 本 + 割り切れない 1 本（= `patch_embed.proj` の縮小形）。"""

    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = nn.Linear(68, 8, bias=False)
        self.attn = nn.Linear(64, 32, bias=False)
        self.ff = nn.Linear(32, 64, bias=False)


def _method(name: str) -> mq.W4Method:
    return next(method for method in mq.W4_METHODS if method.name == name)


def _targets(**overrides: int) -> mq.W4Targets:
    """既定は「32 要素 = 1 group・1 本」の最小形（除外なし）。"""
    base = {"modules": 1, "elements": 32, "groups": 1}
    return mq.W4Targets(excluded={}, **{**base, **overrides})


class TestExistingConfigsAreFrozen:
    """MUST: 10 構成の記号・順序・基準は据え置き（ADR 0030 が構成名を名指しで参照する）。"""

    def test_the_ten_configs_keep_their_names_and_order(self):
        assert mq.CONFIG_NAMES == (
            "f32",
            "w8",
            "w8a8",
            "w8a8-qk",
            "w8a8-pv",
            "w8a8-qkpv",
            "w8-qkpv",
            "w8a8-qkpv-s16",
            "w8a8-s16",
            "w8a8-qkpv-statsq",
        )

    def test_the_baseline_is_still_w8a8(self):
        assert mq.BASELINE == "w8a8"

    def test_the_w4_screen_does_not_share_the_stacking_table(self):
        """方式スクリーニングは別経路 — 10 構成の表を 1 つも借りない。"""
        assert set(mq.W4_METHOD_NAMES).isdisjoint(mq.CONFIG_NAMES)


class TestTargetScan:
    def test_it_counts_groups_on_the_axis_the_rounding_splits(self):
        """group 数は `channel_rows` の平坦形（= 丸めが group を割る軸）から数える。"""
        targets = mq.scan_w4_targets(TinyDit())

        # attn `[32,64]` … 32 行 × (64/32) group / ff `[64,32]` … 64 行 × 1 group。
        assert (targets.modules, targets.elements) == (2, 32 * 64 + 64 * 32)
        assert targets.groups == 32 * 2 + 64

    def test_an_indivisible_quantization_axis_is_excluded_and_listed(self):
        """MUST: 端数 group は格納できない形（ADR 0069 決定 2）— 黙って全体を諦めない。"""
        targets = mq.scan_w4_targets(TinyDit())

        assert targets.excluded == {"patch_embed.weight": 68}
        assert not targets.include("patch_embed")
        assert targets.include("attn")

    def test_a_non_linear_quantizable_weight_fails_loudly(self):
        """conv / embedding が増えたら「linear 限定 = 配布対応形」の前提が崩れる。"""

        class WithConv(TinyDit):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(4, 8, 2)

        with pytest.raises(AssertionError, match="非 linear"):
            mq.scan_w4_targets(WithConv())

    def test_a_fully_indivisible_model_fails_loudly(self):
        class Odd(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(30, 2, bias=False)

        with pytest.raises(AssertionError, match="1 本も無い"):
            mq.scan_w4_targets(Odd())

    def test_the_include_predicate_matches_the_root_module_spelling(self):
        """述語は**モジュール FQN** で呼ばれる（根は空文字列 → 重み FQN は `weight`）。"""
        targets = mq.scan_w4_targets(nn.Linear(64, 8, bias=False))

        assert targets.modules == 1
        assert targets.include("")


class TestExclusionReachesTheRounding:
    """除外は「一覧に載る」だけでなく**丸めに効く**こと（載っただけでは沈黙誤値が残る）。"""

    def test_the_excluded_weight_is_untouched_by_every_method(self):
        torch.manual_seed(0)
        for method in mq.W4_METHODS:
            model = TinyDit()
            targets = mq.scan_w4_targets(model)
            skipped = model.patch_embed.weight.detach().clone()
            rounded = model.attn.weight.detach().clone()

            report = method.apply(model, targets.include, 1)

            assert report.modules == targets.modules == 2
            assert torch.equal(model.patch_embed.weight, skipped)
            # 対象側は実際に動いている（= 述語が「全部落とす」向きに壊れていない）。
            assert not torch.equal(model.attn.weight, rounded)

    def test_without_the_predicate_the_indivisible_weight_would_fail_loudly(self):
        """恒真化の防止 — 除外しなければ core が端数 group で落ちる（上のテストが効く条件）。"""
        model = TinyDit()

        with pytest.raises(Exception, match="割り切れない"):
            _method("rtn-i4-g32").apply(model, lambda _name: True, 1)


class TestMethodsAreNotStacked:
    """MUST: 構成ごとに pristine へ戻す（戻さないと方式間の比較が成立しない）。"""

    def _pristine(self, model: nn.Module, targets: mq.W4Targets):
        weights = {
            fqn: weight
            for fqn, weight, _axis in iter_quant_targets(model, mq.W4_OP_TYPES)
            if fqn not in targets.excluded
        }
        return weights, {fqn: weight.detach().clone() for fqn, weight in weights.items()}

    def test_restore_brings_back_the_original_weights_bit_for_bit(self):
        torch.manual_seed(0)
        model = TinyDit()
        targets = mq.scan_w4_targets(model)
        weights, pristine = self._pristine(model, targets)

        _method("nf4").apply(model, targets.include, 1)
        mq.restore_w4(weights, pristine)

        assert all(torch.equal(weights[fqn], pristine[fqn]) for fqn in pristine)

    def test_a_restored_run_matches_a_fresh_one(self):
        """別方式の後に戻して当てた rtn が、素の重みへ当てた rtn とビット一致する。"""
        torch.manual_seed(0)
        model = TinyDit()
        targets = mq.scan_w4_targets(model)
        weights, pristine = self._pristine(model, targets)

        _method("mxfp4").apply(model, targets.include, 1)
        mq.restore_w4(weights, pristine)
        _method("rtn-i4-g32").apply(model, targets.include, 1)
        after_restore = weights["attn.weight"].detach().clone()

        mq.restore_w4(weights, pristine)
        _method("rtn-i4-g32").apply(model, targets.include, 1)

        assert torch.equal(after_restore, weights["attn.weight"])

    def test_stacking_the_two_methods_would_have_changed_the_answer(self):
        """恒真化の防止 — 戻さずに重ねた場合は実際に別の値になる。"""
        torch.manual_seed(0)
        model = TinyDit()
        targets = mq.scan_w4_targets(model)
        weights, pristine = self._pristine(model, targets)

        _method("mxfp4").apply(model, targets.include, 1)
        _method("rtn-i4-g32").apply(model, targets.include, 1)
        stacked = weights["attn.weight"].detach().clone()

        mq.restore_w4(weights, pristine)
        _method("rtn-i4-g32").apply(model, targets.include, 1)

        assert not torch.equal(stacked, weights["attn.weight"])


class TestSizeProjection:
    """表に載る bpw の出どころ（`docs/ir-v1.md` の i4 格納形と OCP MX の逐語）。"""

    @pytest.mark.parametrize("name", ["rtn-i4-g32", "nf4"])
    def test_group_absmax_methods_land_on_five_bits(self, name):
        """4bit ペイロード + group 32 ごとの F32 scale = 4 + 32/32 = 5.0 bpw。"""
        projection = mq.w4_size_projection(_method(name), _targets(elements=3200, groups=100))

        assert projection["bitsPerWeight"] == pytest.approx(5.0)

    def test_mxfp4_lands_on_four_and_a_quarter_bits(self):
        """共有 scale が E8M0（8bit）なので 4 + 8/32 = 4.25 bpw。"""
        projection = mq.w4_size_projection(_method("mxfp4"), _targets(elements=3200, groups=100))

        assert projection["bitsPerWeight"] == pytest.approx(4.25)

    def test_the_shared_codebook_costs_one_table_for_the_whole_model(self):
        """表は 1 枚だが group scale が要る（= i4 と同じ 5.0 bpw + 表 1 枚ぶん）。"""
        targets = _targets(modules=10, elements=3200, groups=100)

        projection = mq.w4_size_projection(_method("kmeans:shared"), targets)

        assert projection["bitsPerWeight"] == pytest.approx(5.0 + 32 * 16 / 3200)

    def test_the_ratio_is_the_bpw_against_f32(self):
        """比は「f32 格納に対する何 %」— MiB の 2 系列と同じ数から出ていること。"""
        projection = mq.w4_size_projection(_method("nf4"), _targets(elements=3200, groups=100))

        assert projection["ratio"] == pytest.approx(5.0 / 32.0)
        assert projection["projectedMiB"] == pytest.approx(projection["f32MiB"] * 5.0 / 32.0)

    def test_one_mebibyte_is_the_binary_one(self):
        """`MiB` 表記どおり 2^20（10^6 と混ぜると表の数字が 5% ずれる）。"""
        assert mq.MIB == 2**20


class TestKmeansFitStride:
    def test_a_model_inside_the_budget_fits_on_the_full_set(self):
        assert mq.w4_fit_stride(mq.W4_KMEANS_FIT_BUDGET) == 1

    def test_the_sample_stays_inside_the_budget(self):
        elements = 1_956_249_600  # Anima DiT の実測（linear 453 本・除外 1 本のあと）

        stride = mq.w4_fit_stride(elements)

        assert elements // stride <= mq.W4_KMEANS_FIT_BUDGET

    def test_the_stride_is_coprime_with_the_group_size(self):
        """MUST: 偶数 stride は group 内の偶数 lane しか踏まない（group 長は 2 冪）。"""
        for elements in (mq.W4_KMEANS_FIT_BUDGET * 2, mq.W4_KMEANS_FIT_BUDGET * 97 + 1):
            assert mq.w4_fit_stride(elements) % 2 == 1


class TestMethodTable:
    def test_the_screened_winners_are_the_four_methods(self):
        assert mq.W4_METHOD_NAMES == ("rtn-i4-g32", "nf4", "mxfp4", "kmeans:shared")

    def test_the_group_size_is_pinned_for_every_method(self):
        """方式比較の軸は「格子の張り方」だけ — group 長は動かさない（ADR 0069 追記 5）。"""
        assert mq.W4_GROUP_SIZE == 32

    def test_selecting_a_subset_keeps_the_declaration_order(self):
        chosen = mq.select_w4_methods(["mxfp4", "rtn-i4-g32"])

        assert [method.name for method in chosen] == ["rtn-i4-g32", "mxfp4"]

    def test_an_empty_selection_runs_everything(self):
        assert mq.select_w4_methods([]) == mq.W4_METHODS


def _payload(**overrides: object) -> dict[str, object]:
    entry = {
        "method": "nf4",
        "quantReport": "nf4: modules 453 / 1,956,249,600 elements",
        "elapsed": 612.0,
        "latentRelRms": [1.5e-3, 2.5e-3],
        "psnrF32": 31.25,
        "psnrUint8": 31.3,
        "imageRelRms": 4.2e-2,
        "uint8MaxDiff": 41.0,
        "moved": True,
        "size": {
            "modules": 453,
            "elements": 3200,
            "groups": 100,
            "bitsPerWeight": 5.0,
            "projectedMiB": 1.9,
            "f32MiB": 12.2,
            "ratio": 0.15625,
            "formula": "4·N + 32·G",
        },
    }
    base = {
        "generated": "2026-08-19 12:00:00",
        "script": "measure_quant.py --w4-screen",
        "torch": "2.9.0",
        "prompt": "1girl",
        "steps": 2,
        "resolution": 512,
        "guidanceScale": 1.0,
        "seed": 20260802,
        "groupSize": 32,
        "kmeansFitStride": 99,
        "kmeansFitSamplesApprox": 19_760_096,
        "targets": {"opTypes": ["Linear"], "modules": 453, "elements": 3200, "groups": 100},
        "excluded": [{"weight": "patch_embed.proj.weight", "quantAxis": 68}],
        "imageNote": "image_*.png",
        # 基準行は品質列を持たない（自分との比較は情報が無く、PSNR ∞ は JSON でも不正）。
        "configs": {
            "f32": {"method": "f32", "quantReport": "丸めなし（基準）", "elapsed": 9.0},
            "nf4": entry,
        },
    }
    return {**base, **overrides}


class TestReport:
    """表は payload だけから作る（JSON と markdown が割れないため）。"""

    def test_it_carries_the_size_formula_and_the_quality_row(self):
        report = mq.w4_report_markdown(_payload())

        assert "| `nf4` | 5.000 | 1.9 | 12.2 | 15.6% | 4·N + 32·G |" in report
        assert "| `nf4` | 31.25 | 31.30 | 4.2000e-02 | 41 | 612s |" in report

    def test_the_excluded_weight_is_named_in_the_report(self):
        """MUST: 除外を出力へ明記（黙って全体を諦めない）。"""
        report = mq.w4_report_markdown(_payload())

        assert "`patch_embed.proj.weight`（軸 68）" in report

    def test_a_sampled_codebook_fit_is_stated(self):
        """MUST: fit_stride を使ったら明記する（core docstring の MUST）。"""
        report = mq.w4_report_markdown(_payload())

        assert "1/99 の等間隔部分標本" in report
        assert "適用は全量" in report

    def test_a_full_fit_says_nothing_about_sampling(self):
        report = mq.w4_report_markdown(_payload(kmeansFitStride=1))

        assert "部分標本" not in report

    def test_a_config_that_did_not_move_is_marked_up(self):
        payload = _payload()
        payload["configs"]["nf4"] = {**payload["configs"]["nf4"], "moved": False}

        assert "**NG（素通り）**" in mq.w4_report_markdown(payload)


# ---- 校正付き丸め（波 J-2）--------------------------------------------------
#
# 実重み無しで固定できるのは「対象の割り方」と「捕まえ方」と「門」だけ。GPTQ の数値そのものは
# core（`karume.quant_calib`）のテストが持つ。


class TinyCalibBlock(nn.Module):
    """`CosmosTransformerBlock` の代役 — **引数の名前と並び**を写す（Catcher の対応表の根拠）。"""

    def __init__(self, features: int) -> None:
        super().__init__()
        self.attn1 = nn.Linear(features, features, bias=False)
        self.ff = nn.Linear(features, features, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        embedded_timestep: torch.Tensor,
        temb: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        extra_pos_emb: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        controlnet_residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.attn1(hidden_states) + encoder_hidden_states.mean(dim=1)[:, None, :]
        hidden = hidden + self.ff(hidden) + embedded_timestep[:, None, :]
        return hidden_states + hidden


class TinyCalibDit(nn.Module):
    """`CosmosTransformer3DModel` の代役（block を**全て位置引数で**呼ぶ形も写す）。"""

    def __init__(self, features: int = 32, blocks: int = 2) -> None:
        super().__init__()
        self.features = features
        self.transformer_blocks = nn.ModuleList(TinyCalibBlock(features) for _ in range(blocks))

    def forward(
        self,
        hidden: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        embedded_timestep: torch.Tensor,
        temb: torch.Tensor,
    ) -> torch.Tensor:
        for block in self.transformer_blocks:
            hidden = block(
                hidden, encoder_hidden_states, embedded_timestep, temb, None, None, None, None
            )
        return hidden


def tiny_denoise(model: TinyCalibDit, steps: int) -> dict[str, torch.Tensor]:
    """`reference_steps` と同じ形の参照ループ（1 step = DiT 1 forward — CFG=1）。"""
    torch.manual_seed(11)
    x = torch.randn(1, 4, model.features)
    encoder = torch.randn(1, 5, model.features)
    embedded = torch.randn(1, model.features)
    temb = torch.randn(1, model.features)
    out: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for index in range(steps):
            x = x + model(x, encoder, embedded, temb)
            out[f"latents_step{index + 1:04d}"] = x
    return out


def _calib_rig(model: TinyCalibDit, steps: int = 2) -> mq.CalibRig:
    stages = mq.calib_stages(model)
    scan, targets = mq.scan_calib_targets(stages)
    _result, batches = mq.capture_stage_batches(model, lambda: tiny_denoise(model, steps), steps)
    return mq.CalibRig(stages=stages, scan=scan, targets=targets, batches=batches)


class TestCalibConfigTable:
    def test_the_three_calibrated_configs_are_selectable(self):
        assert mq.CALIB_NAMES == ("gptq-rtn", "gptq-nf4", "gptq-kmeans")

    def test_selecting_a_subset_keeps_the_declaration_order(self):
        chosen = mq.select_calib(["gptq-kmeans", "gptq-rtn"])

        assert [config.name for config in chosen] == ["gptq-rtn", "gptq-kmeans"]

    def test_an_empty_selection_runs_every_calibrated_config(self):
        assert mq.select_calib([]) == mq.CALIB_CONFIGS

    def test_selecting_only_a_method_leaves_no_calibrated_config(self):
        """`--w4-only nf4` は方式だけ — 校正付き構成を巻き込まない。"""
        assert mq.select_calib(["nf4"]) == ()

    def test_selecting_only_a_calibrated_config_leaves_no_method(self):
        assert mq.select_w4_methods(["gptq-rtn"]) == ()

    def test_the_group_size_is_pinned_for_every_calibrated_config(self):
        assert all(config.grid.group_size == mq.W4_GROUP_SIZE for config in mq.CALIB_CONFIGS)

    def test_no_calibrated_config_fits_its_table_on_a_subsample(self):
        """校正の表は**層ごと**に張るので全量 fit で作業領域が収まる（stride は要らない）。"""
        assert all(config.grid.fit_stride == 1 for config in mq.CALIB_CONFIGS)

    def test_the_calibrated_grids_project_to_the_same_bpw_as_the_methods(self):
        """校正は格子を 1 バイトも変えない — rtn / nf4 は方式側と同じ 5.0 bpw。"""
        for name in ("gptq-rtn", "gptq-nf4"):
            config = next(item for item in mq.CALIB_CONFIGS if item.name == name)
            assert mq.w4_size_projection(config, _targets())["bitsPerWeight"] == 5.0

    def test_the_calibrated_codebook_costs_one_table_per_layer(self):
        config = next(item for item in mq.CALIB_CONFIGS if item.name == "gptq-kmeans")
        targets = _targets(modules=3, elements=3200, groups=100)

        projection = mq.w4_size_projection(config, targets)

        tables = mq.W4_CODEBOOK_ENTRY_BITS * mq.DEFAULT_CODEBOOK_LEVELS * 3
        assert projection["bitsPerWeight"] == pytest.approx(5.0 + tables / 3200)


class TestCalibTargets:
    def test_the_stage_prefix_carries_the_block_index(self):
        """局所 FQN（`attn1.weight`）へ足すだけでモデル内 FQN へ戻る形（core の `StageSpec`）。"""
        stages = mq.calib_stages(TinyCalibDit())

        assert [prefix for prefix, _stage in stages] == [
            "transformer_blocks.0",
            "transformer_blocks.1",
        ]

    def test_the_scan_keys_are_model_wide_fqns(self):
        scan, targets = mq.scan_calib_targets(mq.calib_stages(TinyCalibDit()))

        assert sorted(scan)[:2] == [
            "transformer_blocks.0.attn1.weight",
            "transformer_blocks.0.ff.weight",
        ]
        assert targets.modules == 4
        assert targets.groups == 4 * 32 * (32 // mq.W4_GROUP_SIZE)

    def test_a_non_linear_quantizable_weight_is_fail_loudly(self):
        """linear 限定 = i4 の実行経路と一致、がこの列の前提（`scan_w4_targets` と同文）。"""
        stages = (("transformer_blocks.0", nn.Conv1d(32, 8, 3, bias=False)),)

        with pytest.raises(AssertionError, match="非 linear"):
            mq.scan_calib_targets(stages)

    def test_a_weight_off_the_group_grid_is_fail_loudly(self):
        """MUST: 除外して進まない — stage 単位の駆動では過不足一致門が張れなくなる。"""
        stages = (("transformer_blocks.0", nn.Linear(12, 4, bias=False)),)

        with pytest.raises(AssertionError, match="割り切れない"):
            mq.scan_calib_targets(stages)

    def test_stages_without_a_linear_are_fail_loudly(self):
        with pytest.raises(AssertionError, match="1 本も無い"):
            mq.scan_calib_targets((("transformer_blocks.0", nn.LayerNorm(8)),))


class TestCatcher:
    def test_it_takes_one_batch_per_step(self):
        model = TinyCalibDit()

        _result, batches = mq.capture_stage_batches(model, lambda: tiny_denoise(model, 3), 3)

        assert len(batches) == 3

    def test_the_positional_arguments_are_named_from_the_block_signature(self):
        """写した名前の並びを持たない — 上流が並べ替えても綴りが黙って入れ替わらない。"""
        model = TinyCalibDit()

        _result, batches = mq.capture_stage_batches(model, lambda: tiny_denoise(model, 1), 1)

        _args, kwargs = batches[0]
        assert list(kwargs) == [
            "encoder_hidden_states",
            "embedded_timestep",
            "temb",
            "image_rotary_emb",
            "extra_pos_emb",
            "attention_mask",
            "controlnet_residual",
        ]

    def test_the_limit_caps_the_batches_without_cutting_the_reference(self):
        """基準 f32 の周回そのものは最後まで回す（latent は全 step ぶん要る）。"""
        model = TinyCalibDit()

        result, batches = mq.capture_stage_batches(model, lambda: tiny_denoise(model, 4), 2)

        assert len(batches) == 2
        assert len(result) == 4

    def test_the_hook_is_removed_afterwards(self):
        model = TinyCalibDit()

        mq.capture_stage_batches(model, lambda: tiny_denoise(model, 1), 1)

        assert not model.transformer_blocks[0]._forward_pre_hooks

    def test_a_reference_that_never_reaches_the_blocks_is_fail_loudly(self):
        with pytest.raises(AssertionError, match="1 step も"):
            mq.capture_stage_batches(TinyCalibDit(), dict, 2)


class TestCalibrationRun:
    def test_every_calibrated_config_rounds_every_scanned_linear(self):
        for config in mq.CALIB_CONFIGS:
            torch.manual_seed(13)
            model = TinyCalibDit()
            rig = _calib_rig(model)
            before = {fqn: weight.detach().clone() for fqn, weight in rig.scan.items()}

            report = mq.apply_calib(config, rig)

            assert report.method == config.method
            assert report.grid == config.grid.kind
            assert report.modules == len(rig.scan)
            assert all(not torch.equal(rig.scan[fqn], value) for fqn, value in before.items())

    def test_a_scan_the_calibration_missed_is_fail_loudly(self):
        """MUST: 丸め漏れは PSNR が**良い側**へ出る — 門が唯一の検出手段。"""
        torch.manual_seed(17)
        model = TinyCalibDit()
        rig = _calib_rig(model)
        report = mq.apply_calib(mq.CALIB_CONFIGS[0], rig)

        with pytest.raises(AssertionError, match="一致しない"):
            mq.assert_calib_covers_scan(
                report, {**rig.scan, "transformer_blocks.9.ff.weight": torch.zeros(1)}, "gptq-rtn"
            )


#: 校正列の対象素性（`run_w4_screen` が JSON へ載せる形の代役）。
CALIB_META = {
    "name": mq.CALIB_TARGET,
    "stages": 28,
    "modules": 280,
    "elements": 1_900_000_000,
    "groups": 59_375_000,
    "batches": 10,
}


class TestCalibReport:
    def test_the_calibrated_target_set_is_spelled_out(self):
        """MUST: 4 方式と対象集合が違うことを表に明記（bpw も品質も直接は比較できない）。"""
        report = mq.w4_report_markdown(_payload(calibTargets=CALIB_META))

        assert mq.CALIB_TARGET in report
        assert "280 本" in report

    def test_a_run_without_calibrated_configs_says_nothing_about_them(self):
        assert mq.CALIB_TARGET not in mq.w4_report_markdown(_payload(calibTargets=None))


def _stub_run() -> mq.DitRun:
    """実重み無しで `main()` を通すための最小 `DitRun`（数値は誰も読まない）。"""
    return mq.DitRun(latents={}, linear_stats={}, attn_stats={}, diagnostics={}, attention_nodes=[])


def _stub_the_measurement(monkeypatch: pytest.MonkeyPatch, *, failed: list[str]) -> None:
    """`main()` の実測段を全て差し替え、集計だけを与えられた判定に固定する。"""
    monkeypatch.setattr(mq, "run_dit", lambda _args: _stub_run())
    monkeypatch.setattr(mq, "write_layer_csv", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mq, "decode_all", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(mq, "save_images", lambda *_args, **_kwargs: "image_*.png")
    monkeypatch.setattr(
        mq,
        "build_summary",
        lambda *_args, **_kwargs: {
            "pt": {},
            "configs": {},
            "gates": {label: "… → NG" for label in failed} or {"どれか": "OK"},
            "failed": failed,
            "inject": None,
        },
    )
    monkeypatch.setattr(mq, "build_report", lambda *_args, **_kwargs: "")


class TestTheQualityGateReachesTheExitCode:
    """ADR 0019 の品質ゲート①を埋める台本なので、赤は印字だけで終わらない。

    `--w4-screen` 経路（`main_w4`）は最初から `AssertionError` を投げる — 同じ台本の 2 経路で
    強さが割れていると、自動実行では 10 構成側の赤だけが見落とされる。
    """

    def test_a_red_gate_ends_the_run_with_an_error(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_the_measurement(monkeypatch, failed=["`w8a8` の SDPA 発火計数"])
        monkeypatch.setattr(
            "sys.argv", ["measure_quant.py", "--out", str(tmp_path), "--no-latents"]
        )

        with pytest.raises(AssertionError, match="品質ゲートが赤"):
            mq.main()

    def test_an_injected_failure_is_allowed_to_be_red(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`--inject` は検出力の実証で、赤にすること自体が目的の経路（塞がない）。"""
        _stub_the_measurement(monkeypatch, failed=["`w8a8` の SDPA 発火計数"])
        monkeypatch.setattr(
            "sys.argv",
            [
                "measure_quant.py",
                "--out",
                str(tmp_path),
                "--no-latents",
                "--inject",
                "drop-attn-quant",
            ],
        )

        mq.main()

    def test_all_green_ends_normally(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_the_measurement(monkeypatch, failed=[])
        monkeypatch.setattr(
            "sys.argv", ["measure_quant.py", "--out", str(tmp_path), "--no-latents"]
        )

        mq.main()
