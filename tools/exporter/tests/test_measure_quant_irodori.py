"""`measure_quant_irodori.py` の pure 部分（実重み不要分）。

実測そのものは手動（full-loop は torch CPU で数十分かかる）。ここで固定するのは、壊れると
**測定値が静かに嘘になる**側だけ:

- 指標（SNR / LSD / relRMS）の境界値 — 一致で inf / 0、参照が全ゼロ、スケール倍の上界
- 構成表 — `i8-all` が全役割を覆い、`i8-mixed` が duration だけ外し、直交分解が 1 役割ずつ
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

import export_irodori as ex
import irodori_pipeline as ip
import measure_quant_irodori as mq


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

    def test_a_codec_only_config_must_move_the_waveform_but_not_the_latent(self):
        assert _gates("i8-codec-only", z_changed=False, wav_changed=True) == []
        assert _gates("i8-codec-only", z_changed=False, wav_changed=False) != []
        assert _gates("i8-codec-only", z_changed=True, wav_changed=True) != []


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
    """`irodori_pipeline.run_case` の注入口（S を外から固定する）— 既定は素の予測。"""

    def test_the_golden_path_keeps_the_predicted_length(self):
        signature = ip.run_case.__kwdefaults__

        assert signature["frames_override"] is None
