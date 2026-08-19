"""`irodori/measure_quant.py` の pure 部分（実重み不要分）。

実測そのものは手動（full-loop は torch CPU で数十分かかる）。ここで固定するのは、壊れると
**測定値が静かに嘘になる**側だけ:

- 指標（SNR / LSD / relRMS）の境界値 — 一致で inf / 0、参照が全ゼロ、スケール倍の上界
- 構成表 — `i8-all` が全役割を覆い、`i8-mixed` が duration だけ外し、直交分解が 1 役割ずつ、
  `w8a8` が `i8-all` と同じ重みで活性だけ足す
- 活性シムの適用 — DiT の適格 `nn.Linear` に掛かること・**0 本は fail loudly**
- 役割 → モジュールの束ね方 — 全役割で `export_series` が丸める 9 本と**同じ集合**になること
  （食い違うと「配布系列の f16 と同じもの」を測っているつもりで別物を測る）
- 直交性の門 — z / 波形が「変わるべきところだけ変わる」の**両側**が実際に落ちること
"""

from __future__ import annotations

import math
import wave

import pytest
import torch
from torch import nn

from irodori import export as ex
from irodori import measure_quant as mq
from irodori import pipeline_ref as ip
from karume.quantize import QUANT_MODULE_TYPES


class TestMetrics:
    def test_snr_is_infinite_for_a_bit_identical_signal(self):
        signal = torch.tensor([0.5, -0.25, 0.125])

        assert mq.snr_db(signal, signal) == math.inf

    def test_snr_matches_the_closed_form(self):
        reference = torch.ones(4)
        value = reference + 0.1

        assert mq.snr_db(value, reference) == pytest.approx(10.0 * math.log10(1.0 / 0.01))

    def test_snr_of_a_zero_reference_is_minus_infinite(self):
        """全ゼロ参照は SNR が定義できない — 0 で割って NaN を返さない。"""
        assert mq.snr_db(torch.ones(4), torch.zeros(4)) == -math.inf

    def test_rel_rms_is_zero_for_a_zero_reference(self):
        assert mq.rel_rms(torch.ones(4), torch.zeros(4)) == 0.0

    def test_rel_rms_matches_the_closed_form(self):
        reference = torch.tensor([3.0, 4.0])

        assert mq.rel_rms(reference * 2.0, reference) == pytest.approx(1.0)

    def test_lsd_is_zero_for_a_bit_identical_signal(self):
        torch.manual_seed(0)
        signal = torch.randn(mq.STFT_N_FFT * 4)

        assert mq.log_spectral_distance(signal, signal) == 0.0

    def test_lsd_of_a_scaled_copy_stays_under_the_scale_factor(self):
        """2 倍にした信号の LSD は 20log10(2) が上界（床 clamp は縮める方向にしか効かない）。"""
        torch.manual_seed(0)
        reference = torch.randn(mq.STFT_N_FFT * 4)

        distance = mq.log_spectral_distance(reference * 2.0, reference)

        assert 0.0 < distance <= 20.0 * math.log10(2.0) + 1e-6


class TestConfigTable:
    def test_the_base_is_the_only_unrounded_config(self):
        unrounded = [name for name, recipe in mq.RECIPES.items() if recipe.weight is None]

        assert unrounded == [mq.BASE_CONFIG]

    def test_i8_all_covers_every_role(self):
        assert mq.RECIPES["i8-all"].roles == mq.ROLES

    def test_i8_mixed_leaves_duration_alone(self):
        """混成表の候補（ADR 0050 決定 6 の (i) 案）— 外すのは duration **だけ**。"""
        mixed = set(mq.RECIPES["i8-mixed"].roles)

        assert mixed == set(mq.ROLES) - {ex.TARGET_DURATION}

    def test_the_decomposition_takes_one_role_at_a_time(self):
        legs = {name: recipe for name, recipe in mq.DIAGNOSTICS.items()}

        assert len(legs) == 5
        assert all(len(recipe.roles) == 1 for recipe in legs.values())
        assert {recipe.roles[0] for recipe in legs.values()} == set(mq.DECOMPOSED_ROLES)

    def test_duration_and_codec_are_outside_the_latent_path(self):
        """MUST: 直交性の門の期待値そのもの（duration は S だけ・codec は波形だけ）。"""
        assert set(mq.ROLES) - {ex.TARGET_DURATION, mq.ROLE_CODEC} == mq.LATENT_ROLES

    def test_w8a8_rounds_the_same_weights_as_the_weight_only_config(self):
        """配布形の 2 席は**同じ i8 バイトを共有する** — 違いは活性側だけ。"""
        weight_only = mq.RECIPES[mq.WEIGHT_ONLY_BASE]
        act = mq.RECIPES["w8a8"]

        assert (act.weight, act.roles) == (weight_only.weight, weight_only.roles)
        assert act.act_quant and not weight_only.act_quant

    def test_only_w8a8_carries_the_activation_sim(self):
        """活性シムを持つ構成が増えたら、素通り検出の比較相手も増やす必要がある。"""
        with_sim = [name for name, recipe in mq.RECIPES.items() if recipe.act_quant]

        assert with_sim == ["w8a8"]


class TinyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(2, 2, bias=False)


def _modules() -> dict[str, nn.Module]:
    return {
        name: TinyModule()
        for name in (
            ex.TARGET_BACKBONE,
            ex.TARGET_TEXT_PROJ,
            ex.TARGET_CAPTION_PROJ,
            ex.TARGET_SPEAKER,
            ex.TARGET_DURATION,
            ex.TARGET_DIT,
            "speaker_norm",
            "text_norm",
            "caption_norm",
        )
    }


class TestRoleModules:
    """役割 → 丸め対象の束ね方（`export_series` / `emit` が丸める 9 本との集合一致が肝）。"""

    def test_all_roles_hand_the_same_nine_modules_as_the_export(self):
        modules = _modules()

        picked = mq.role_modules(modules, mq.ROLES)

        assert set(picked) == set(modules)

    def test_the_codec_role_touches_nothing_on_the_latent_side(self):
        """`i8-codec-only` は latent 側を 1 本も丸めない（z のビット一致が門になる）。"""
        assert mq.role_modules(_modules(), (mq.ROLE_CODEC,)) == {}

    def test_each_norm_rides_with_the_graph_that_holds_it(self):
        """MUST: norm は**消費側**の役割へ（所在は ADR 0048 決定 1）。"""
        modules = _modules()

        assert sorted(mq.role_modules(modules, (ex.TARGET_DURATION,))) == [
            ex.TARGET_DURATION,
            "text_norm",
        ]
        assert sorted(mq.role_modules(modules, (ex.TARGET_SPEAKER,))) == [
            ex.TARGET_SPEAKER,
            "speaker_norm",
        ]
        assert sorted(mq.role_modules(modules, (ex.TARGET_CAPTION_PROJ,))) == [
            ex.TARGET_CAPTION_PROJ,
            "caption_norm",
        ]
        assert sorted(mq.role_modules(modules, (ex.TARGET_BACKBONE,))) == [ex.TARGET_BACKBONE]


class _DitLike(nn.Module):
    """適格 `nn.Linear`（`in_features % 4 == 0`）を 2 本持つ DiT 代役。"""

    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(4, 8)
        self.out = nn.Linear(8, 4)


class TestActQuantSim:
    """MUST: 掛かった本数を出す（0 本のまま回すと `i8-all` と同じ数が w8a8 の名前で載る）。"""

    def test_it_hooks_every_eligible_linear_in_the_dit(self):
        modules = {ex.TARGET_DIT: _DitLike()}

        handles, attached = mq.attach_dit_act_quant(mq.RECIPES["w8a8"], modules)

        assert attached == 2
        mq.act_quant.detach_act_quant(handles)

    def test_it_actually_rounds_the_activations(self):
        """フックの効き目そのもの（掛けた後の出力が `quantize_rows` を通した入力と一致する）。"""
        modules = {ex.TARGET_DIT: _DitLike()}
        linear = modules[ex.TARGET_DIT].qkv
        x = torch.tensor([[1.0, -0.3, 0.02, 0.7]])
        handles, _ = mq.attach_dit_act_quant(mq.RECIPES["w8a8"], modules)

        with torch.no_grad():
            hooked = linear(x)
            mq.act_quant.detach_act_quant(handles)
            expected = linear(mq.act_quant.quantize_rows(x))

        assert torch.equal(hooked, expected)

    def test_zero_eligible_linears_is_fail_loudly(self):
        """適格 0 本で素通りさせない — 品質は「良い」側に出るので黙ると誤読しかされない。"""
        modules = {ex.TARGET_DIT: TinyModule()}  # Linear(2, 2) は k % 4 != 0

        with pytest.raises(SystemExit, match="0 本"):
            mq.attach_dit_act_quant(mq.RECIPES["w8a8"], modules)

    def test_a_weight_only_config_hooks_nothing(self):
        handles, attached = mq.attach_dit_act_quant(mq.RECIPES[mq.WEIGHT_ONLY_BASE], {})

        assert (handles, attached) == ([], 0)


def _entry(*, z_changed: bool, wav_changed: bool) -> dict[str, object]:
    return {"vsBase": {"zBitEqual": not z_changed, "wavBitEqual": not wav_changed}}


def _gates(config: str, *, z_changed: bool, wav_changed: bool) -> list[str]:
    entries = {"full": _entry(z_changed=z_changed, wav_changed=wav_changed)}
    gates = mq.run_gates(config, mq.RECIPES[config], {}, entries, None)
    return gates["failures"]


class TestOrthogonalityGate:
    """MUST: 期待は**両側**（変わる / 変わらない）— 片側だけだと素通りか scope 漏れが素通りする。"""

    def test_a_dit_only_config_that_did_not_move_the_latent_is_red(self):
        failures = _gates("i8-dit-only", z_changed=False, wav_changed=False)

        assert any("z" in message for message in failures)

    def test_a_dit_only_config_that_moved_both_is_green(self):
        assert _gates("i8-dit-only", z_changed=True, wav_changed=True) == []

    def test_a_duration_only_config_that_moved_the_latent_is_red(self):
        """duration は S にしか効かない — z が動いたら scope が漏れている。"""
        failures = _gates("i8-duration-only", z_changed=True, wav_changed=True)

        assert len(failures) == 2

    def test_a_duration_only_config_that_moved_nothing_is_green(self):
        assert _gates("i8-duration-only", z_changed=False, wav_changed=False) == []

    def test_a_speaker_only_config_is_expected_to_move_nothing_without_a_reference(self):
        """speaker の期待は**ケース条件**（参照なしはゼロ供給短絡で encoder が走らない —
        ADR 0048。実測 2026-08-12: no-ref はビット一致・full は変化）。"""
        entries = {
            "full": _entry(z_changed=True, wav_changed=True),
            "no-ref": _entry(z_changed=False, wav_changed=False),
        }
        gates = mq.run_gates("i8-speaker-only", mq.RECIPES["i8-speaker-only"], {}, entries, None)

        assert gates["failures"] == []

    def test_a_speaker_only_config_that_moved_the_no_ref_case_is_red(self):
        """参照なしのケースで z が動いたら、ゼロ短絡が壊れたか scope が漏れている。"""
        entries = {"no-ref": _entry(z_changed=True, wav_changed=True)}
        gates = mq.run_gates("i8-speaker-only", mq.RECIPES["i8-speaker-only"], {}, entries, None)

        assert len(gates["failures"]) == 2

    def test_a_codec_only_config_must_move_the_waveform_but_not_the_latent(self):
        assert _gates("i8-codec-only", z_changed=False, wav_changed=True) == []
        assert _gates("i8-codec-only", z_changed=False, wav_changed=False) != []
        assert _gates("i8-codec-only", z_changed=True, wav_changed=True) != []


def _act_gates(*, moved_from_weight_only: bool) -> list[str]:
    entry = _entry(z_changed=True, wav_changed=True)
    entry["vsWeightOnly"] = {"zBitEqual": not moved_from_weight_only}
    gates = mq.run_gates("w8a8", mq.RECIPES["w8a8"], {}, {"full": entry}, None)
    return gates["failures"]


class TestActQuantPassThroughGate:
    """MUST: 素通り検出は**重みだけの構成との差**（基準との差は重みの丸めで説明が付く）。"""

    def test_a_latent_that_moved_off_the_weight_only_config_is_green(self):
        assert _act_gates(moved_from_weight_only=True) == []

    def test_a_latent_bit_identical_to_the_weight_only_config_is_red(self):
        failures = _act_gates(moved_from_weight_only=False)

        assert len(failures) == 1
        assert mq.WEIGHT_ONLY_BASE in failures[0]


class TestGoldenGate:
    """MUST: 基準 `f32` は full-loop golden とバイト一致（外れたら測定値は全部意味を失う）。"""

    def _failures(self, latent: torch.Tensor, golden: torch.Tensor) -> list[str]:
        gates = mq.run_gates(
            mq.BASE_CONFIG, mq.RECIPES[mq.BASE_CONFIG], {"full": latent}, {}, {"full": golden}
        )
        return gates["failures"]

    def test_a_bit_identical_latent_is_green(self):
        golden = torch.tensor([[[0.25, -0.5]]])

        assert self._failures(golden.clone(), golden) == []

    def test_a_latent_off_by_one_ulp_is_red(self):
        """近似では通さない — 1ulp 動いていれば経路が別物になっている。"""
        golden = torch.tensor([[[0.25, -0.5]]])
        drifted = golden + torch.tensor([[[0.0, 2.0**-23]]])

        assert self._failures(drifted, golden) != []


class TestWriteWav:
    def test_it_writes_mono_pcm16_at_the_given_rate(self, tmp_path):
        path = tmp_path / "case.wav"

        mq.write_wav(path, torch.tensor([0.0, 1.0, -1.0, 0.5]), 48000)

        with wave.open(str(path), "rb") as handle:
            assert handle.getnchannels() == 1
            assert handle.getsampwidth() == 2
            assert handle.getframerate() == 48000
            assert handle.getnframes() == 4

    def test_it_clips_instead_of_wrapping_around(self):
        """MUST: 飽和は clamp（int16 へそのまま落とすと +1.5 が負の値へ巻き付く）。"""
        clipped = torch.tensor([2.0, -2.0]).clamp(-1.0, 1.0)
        pcm = torch.floor(clipped * 32767.0 + 0.5).to(torch.int16)

        assert pcm.tolist() == [32767, -32767]


def _payload(name: str, predicted: int, *, versus: dict[str, float] | None) -> dict[str, object]:
    entry: dict[str, object] = {
        "S": 161,
        "predictedS": predicted,
        "forwards": 200,
        "predictedSeconds": predicted / 25,
        "elapsed": 12.0,
    }
    if versus is not None:
        entry["vsBase"] = versus
    return {"cases": {"full": entry}, "gates": {"failures": []}}


class TestReportTables:
    """S ドリフトの表は**この台本の主成果**（混成表の裁定がこの 1 枚に乗る）。"""

    def _collected(self, drifted: int) -> dict[str, dict[str, object]]:
        return {
            mq.BASE_CONFIG: _payload(mq.BASE_CONFIG, 161, versus=None),
            "i8-all": _payload(
                "i8-all",
                drifted,
                versus={
                    "zRelRms": 1.5e-2,
                    "zMaxAbsDiff": 3.0e-2,
                    "wavSnrDb": 12.5,
                    "wavLsdDb": 3.2,
                },
            ),
        }

    def test_a_matching_prediction_reads_as_agreement(self):
        table = mq.drift_table(self._collected(161))

        assert "| i8-all | full | 161 | 161 | 一致 |" in table

    def test_a_drifting_prediction_is_signed(self):
        table = mq.drift_table(self._collected(159))

        assert "| i8-all | full | 159 | 161 | -2 |" in table

    def test_the_base_row_says_it_is_the_base(self):
        table = mq.quality_table(self._collected(161))

        assert "| f32 | full | 0（基準）" in table
        assert "| i8-all | full | 1.5000e-02 | 3.0000e-02 | 12.50 | 3.20 |" in table

    def test_the_report_lists_only_red_configs(self):
        collected = self._collected(161)
        collected["i8-all"]["gates"] = {"failures": ["full: z が「変わる」期待に反した"]}

        report = mq.build_report(collected)

        assert sorted(report["failures"]) == ["i8-all"]


class TestFramesOverride:
    """`irodori.pipeline_ref.run_case` の注入口（S を外から固定する）— 既定は素の予測。"""

    def test_the_golden_path_keeps_the_predicted_length(self):
        signature = ip.run_case.__kwdefaults__

        assert signature["frames_override"] is None


class _W4Like(nn.Module):
    """g32 に載る重みと載らない重みを混ぜた木（対象の割り方と除外の検査用）。

    - `aligned` … linear の in 軸 64 = g32 が 2 本
    - `ragged` … linear の in 軸 12（g32 で割り切れない → 除外）
    - `conv` … `Cin·K = 8·4 = 32`（受容野を平坦化した軸で 1 本ぶん）
    """

    def __init__(self) -> None:
        super().__init__()
        self.aligned = nn.Linear(64, 4, bias=False)
        self.ragged = nn.Linear(12, 4, bias=False)
        self.conv = nn.Conv1d(8, 4, 4, bias=False)


class TestW4ConfigTable:
    """w4 構成表（ADR 0069 追記 5 — 勝者 4 方式 × 全役割 + RTN の混成）。"""

    def test_every_w4_config_carries_a_method(self):
        assert all(recipe.method is not None for recipe in mq.W4_CONFIGS.values())

    def test_no_storage_dtype_config_carries_a_method(self):
        """f16 / i8 側に方式が付くと `irodori.export.fake_quant` を素通りする。"""
        others = {**mq.CONFIGS, **mq.DIAGNOSTICS}

        assert all(recipe.method is None for recipe in others.values())

    def test_the_methods_are_the_four_screening_winners(self):
        """スクリーニングで残った 4 種そのもの（fp4 と kmeans の他 2 粒度は落選）。"""
        names = {recipe.method.name for recipe in mq.W4_CONFIGS.values()}

        assert names == {"rtn-i4-g32", "nf4", "mxfp4", "kmeans:shared"}

    def test_the_all_role_configs_cover_every_role(self):
        covering = [name for name, recipe in mq.W4_CONFIGS.items() if recipe.roles == mq.ROLES]

        assert covering == ["i4-all", "nf4-all", "mxfp4-all", "kmeans-shared-all"]

    def test_the_w4_mixed_config_shares_the_axis_with_the_i8_one(self):
        """MUST: 混成の軸は i8 と同じ（別々に書くと 2 つの混成が黙って別物になる）。"""
        assert mq.RECIPES["i4-mixed"].roles == mq.RECIPES["i8-mixed"].roles
        assert set(mq.RECIPES["i4-mixed"].roles) == set(mq.ROLES) - {ex.TARGET_DURATION}

    def test_the_weight_spelling_is_the_method_name(self):
        """`weight` は丸めの綴り — 方式名と二重管理にしない。"""
        assert mq.RECIPES["nf4-all"].weight == "nf4"

    def test_the_op_types_stay_the_five_i8_kinds(self):
        """狭めると「非 linear まで丸めた品質」を測っているつもりで linear だけ測る。"""
        assert mq.W4_OP_TYPES == QUANT_MODULE_TYPES

    def test_the_group_size_is_pinned_to_32(self):
        """方式比較で g を同時に振らない（ADR 0069 追記 5 の 3）。"""
        assert mq.W4_GROUP_SIZE == 32

    def test_the_linear_config_is_the_shippable_form(self):
        """`i4-linear` = RTN × Linear のみ（今日の配布対応形 — ADR 0069 決定 5）。"""
        recipe = mq.RECIPES["i4-linear"]

        assert recipe.method is mq.W4_RTN
        assert recipe.op_types == (nn.Linear,)
        # codec に linear は 1 本も無いので roles から外す（恒等 — 対象 0 の役割を回さない）。
        assert set(recipe.roles) == set(mq.ROLES) - {mq.ROLE_CODEC}

    def test_the_all_role_configs_keep_the_wide_op_types(self):
        """既定の op 種は広い 5 種のまま（i4-linear の追加で既存構成が狭まっていない）。"""
        for name in ("i4-all", "i4-mixed", "nf4-all", "mxfp4-all", "kmeans-shared-all"):
            assert mq.RECIPES[name].op_types == mq.W4_OP_TYPES, name


class TestScanTargets:
    """対象の割り方（g32 に載る側 / 割り切れず外す側）— 除外は必ず一覧で出る。"""

    def test_it_splits_the_aligned_and_the_ragged(self):
        scan = mq.scan_targets(_W4Like(), mq.W4_OP_TYPES)

        assert scan.counts == mq.TargetCounts(modules=2, channels=8, elements=64 * 4 + 8 * 4 * 4)
        assert [item.fqn for item in scan.excluded] == ["ragged.weight"]
        assert scan.excluded[0].axis_length == 12

    def test_the_linear_only_set_leaves_the_conv_out(self):
        scan = mq.scan_targets(_W4Like(), (nn.Linear,))

        assert scan.counts == mq.TargetCounts(modules=1, channels=4, elements=64 * 4)

    def test_the_group_count_follows_the_element_count(self):
        counts = mq.TargetCounts(modules=2, channels=8, elements=384)

        assert counts.groups == 384 // 32

    def test_the_conv_axis_is_the_flattened_receptive_field(self):
        """group 軸は各 op の「in 軸」一般化 — conv は `Cin·K`（ADR 0069 追記 5 の 1）。"""
        weight = nn.Conv1d(8, 4, 4, bias=False).weight

        assert mq.group_axis_length(weight, 0) == 8 * 4

    def test_the_transposed_conv_axis_is_the_layout_axis_one(self):
        """`ConvTranspose1d` は重みが `[Cin, Cout, K]` なのでチャネル軸が 1。"""
        weight = nn.ConvTranspose1d(4, 8, 8, bias=False).weight

        assert mq.group_axis_length(weight, 1) == 4 * 8

    def test_the_include_predicate_drops_exactly_the_excluded_modules(self):
        scan = mq.scan_targets(_W4Like(), mq.W4_OP_TYPES)

        include = mq.aligned_include(scan.excluded)

        assert not include("ragged")
        assert include("aligned") and include("conv")


class TestApplyWeightQuant:
    """丸めの当て方（w4 は方式ごとの core 呼び・f16 / i8 は配布経路と同じ関数）。"""

    def test_it_rounds_the_aligned_weights_and_leaves_the_ragged_alone(self):
        torch.manual_seed(0)
        module = _W4Like()
        before = {name: tensor.detach().clone() for name, tensor in module.named_parameters()}

        reports = mq.apply_weight_quant(
            mq.w4_recipe(mq.W4_RTN, (ex.TARGET_DIT,)), {ex.TARGET_DIT: module}
        )

        assert not torch.equal(module.aligned.weight, before["aligned.weight"])
        assert not torch.equal(module.conv.weight, before["conv.weight"])
        assert torch.equal(module.ragged.weight, before["ragged.weight"])
        assert "g32 非整列 1 本を除外" in reports[ex.TARGET_DIT]

    def test_a_role_without_target_types_stays_f32(self):
        """norm 系は w4 の対象型を持たない — 役割単位の「0 本」で落とさない側。"""
        reports = mq.apply_weight_quant(
            mq.w4_recipe(mq.W4_RTN, (ex.TARGET_DIT,)), {"text_norm": nn.LayerNorm(8)}
        )

        assert reports["text_norm"] == "格納 f32 のまま（w4 の対象型を持たない）"

    def test_a_role_whose_targets_all_fall_off_the_grid_is_fail_loudly(self):
        """全部 g32 で外れた役割を黙って f32 で回すと「丸めたつもりの構成」を測る。"""
        module = nn.Linear(12, 4, bias=False)

        with pytest.raises(SystemExit, match="g32"):
            mq.apply_weight_quant(
                mq.w4_recipe(mq.W4_RTN, (ex.TARGET_DIT,)), {ex.TARGET_DIT: module}
            )

    def test_a_storage_dtype_recipe_goes_through_the_export_helper(self):
        """f16 / i8 は配布経路と同じ `irodori.export.fake_quant` のまま。"""
        reports = mq.apply_weight_quant(
            mq.Recipe("f16", (ex.TARGET_DIT,)), {ex.TARGET_DIT: TinyModule()}
        )

        assert "f16" in reports[ex.TARGET_DIT]

    def test_a_linear_only_recipe_leaves_the_conv_untouched(self):
        """`i4-linear` の実体 — op_types が丸めの側にも効く（scan だけ狭めても意味が無い）。"""
        torch.manual_seed(0)
        module = _W4Like()
        before = {name: tensor.detach().clone() for name, tensor in module.named_parameters()}

        mq.apply_weight_quant(
            mq.w4_recipe(mq.W4_RTN, (ex.TARGET_DIT,), op_types=(nn.Linear,)),
            {ex.TARGET_DIT: module},
        )

        assert not torch.equal(module.aligned.weight, before["aligned.weight"])
        assert torch.equal(module.conv.weight, before["conv.weight"])
        assert torch.equal(module.ragged.weight, before["ragged.weight"])

    def test_a_method_that_rounds_a_different_set_than_the_scan_is_fail_loudly(self):
        """数えた対象と丸めた対象が割れたら落とす（op_types の焼き込み退行の検出器）。"""
        wide = mq.W4Method(
            "rtn-i4-g32",
            # 構成の op_types を無視して常に広い 5 種で丸める（退行の再現）。
            lambda model, _op_types, include, _stride: mq.fake_quant_int4(
                model, mq.W4_GROUP_SIZE, include=include, op_types=mq.W4_OP_TYPES
            ),
            lambda counts, _tables: 4.0 * counts.elements,
            "検査用",
        )

        with pytest.raises(SystemExit, match="丸めた本数"):
            mq.apply_weight_quant(
                mq.w4_recipe(wide, (ex.TARGET_DIT,), op_types=(nn.Linear,)),
                {ex.TARGET_DIT: _W4Like()},
            )

    def test_the_shared_codebook_takes_the_fit_stride(self):
        """表の fit だけ部分標本（適用は全量）— dit 役割の全量 fit は実メモリを超える。"""
        torch.manual_seed(0)
        module = _W4Like()
        recipe = mq.w4_recipe(mq.W4_KMEANS_SHARED, (ex.TARGET_DIT,))

        reports = mq.apply_weight_quant(recipe, {ex.TARGET_DIT: module}, 3)

        assert "表の fit は 1/3 部分標本" in reports[ex.TARGET_DIT]

    def test_a_fit_stride_on_a_method_without_a_table_is_fail_loudly(self):
        """MUST: 効かない構成へ渡されたら落とす（黙って無視すると数値の読み方が変わる）。"""
        with pytest.raises(SystemExit, match="fit"):
            mq.check_fit_stride(mq.RECIPES["i4-all"], 11)

    def test_the_default_fit_stride_passes_every_config(self):
        for recipe in mq.RECIPES.values():
            mq.check_fit_stride(recipe, 1)

    def test_the_rounding_is_idempotent(self):
        """MUST: 冪等（±7 に閉じた対称 — 再適用でビット不変・ADR 0069 決定 3）。"""
        torch.manual_seed(0)
        module = _W4Like()
        recipe = mq.w4_recipe(mq.W4_RTN, (ex.TARGET_DIT,))
        mq.apply_weight_quant(recipe, {ex.TARGET_DIT: module})
        once = module.aligned.weight.detach().clone()

        mq.apply_weight_quant(recipe, {ex.TARGET_DIT: module})

        assert torch.equal(module.aligned.weight, once)


class TestSizeProjection:
    """サイズ試算（**式による投影**・実測ではない）— 品質と同じ表に載せる列。"""

    COUNTS = mq.TargetCounts(modules=2, channels=8, elements=384)

    def test_a_f32_scale_method_projects_to_five_bits_per_weight(self):
        assert mq.project_size(mq.W4_RTN, self.COUNTS, 1).bits_per_weight == 5.0
        assert mq.project_size(mq.W4_NF4, self.COUNTS, 1).bits_per_weight == 5.0

    def test_the_power_of_two_scale_saves_the_scale_bits(self):
        """MXFP4 の scale は E8M0（1 バイト）— g32 なら 4.25 bpw。"""
        assert mq.project_size(mq.W4_MXFP4, self.COUNTS, 1).bits_per_weight == 4.25

    def test_the_shared_codebook_costs_one_table_per_artifact(self):
        projected = mq.project_size(mq.W4_KMEANS_SHARED, self.COUNTS, 3)

        assert projected.bits == 5.0 * 384 + mq.CODEBOOK_BITS * 3

    def test_a_table_free_method_ignores_the_artifact_count(self):
        assert (
            mq.project_size(mq.W4_RTN, self.COUNTS, 1).bits
            == mq.project_size(mq.W4_RTN, self.COUNTS, 9).bits
        )

    def test_the_f32_side_is_the_denominator_of_the_ratio(self):
        assert mq.project_size(mq.W4_RTN, self.COUNTS, 1).f32_mib == 384 * 4 / 1024**2


def _w4_config(excluded: list[dict[str, object]]) -> dict[str, object]:
    return {
        "cases": {},
        "gates": {"failures": []},
        "w4": {
            "method": "rtn-i4-g32",
            "excluded": excluded,
            "size": {
                "formula": "4bit + g32 f32 scale = 5.0 bpw",
                "modules": 2,
                "elements": 384,
                "bitsPerWeight": 5.0,
                "projectedMiB": 2.0,
                "f32MiB": 8.0,
            },
        },
    }


class TestW4Tables:
    """w4 の 2 出力（サイズ試算の表・除外一覧）— 除外を黙らせない。"""

    def test_the_size_table_carries_the_formula_and_the_ratio(self):
        table = mq.size_table({"i4-all": _w4_config([])})

        assert table.splitlines()[-1] == (
            "| i4-all | rtn-i4-g32 | 2 | 384 | 4bit + g32 f32 scale = 5.0 bpw"
            " | 5.000 | 2.0 | 8.0 | 0.250 |"
        )

    def test_a_config_without_w4_gets_no_row(self):
        table = mq.size_table({"i8-all": {"cases": {}, "gates": {"failures": []}}})

        assert len(table.splitlines()) == 2

    def test_the_exclusions_are_listed_per_weight(self):
        excluded = [
            {
                "role": "speaker",
                "fqn": "encoder.blocks.0.mlp.w2.weight",
                "axisLength": 1996,
                "elements": 1532928,
            }
        ]

        lines = mq.excluded_lines({"i4-all": _w4_config(excluded)})

        assert lines[0] == "- i4-all: g32 非整列 1 本を除外"
        assert "speaker / encoder.blocks.0.mlp.w2.weight 軸長 1996" in lines[1]

    def test_no_exclusion_still_says_so(self):
        """「除外なし」も出す — 行が無いのは「w4 でない」と区別が付かない。"""
        assert mq.excluded_lines({"i4-all": _w4_config([])}) == [
            "- i4-all: g32 非整列による除外なし"
        ]

    def test_a_config_without_w4_is_silent(self):
        assert mq.excluded_lines({"i8-all": {"cases": {}, "gates": {"failures": []}}}) == []
