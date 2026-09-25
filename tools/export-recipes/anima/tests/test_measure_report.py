"""`anima/measure_report.py` の `build_summary` — ADR 0019 の品質ゲート①の判定の約束事。

`failed` は `measure_quant.main` の終了コードへ届く（`test_measure_quant.py` はそこから先を
スタブで見る）ので、ここでは門の述語そのものを合成の `DitRun` / latent で固定する:

- P̃ 行 peak/rms の median は上限以下が緑・超えれば赤
- latent 比の門は linear 活性を量子化する構成だけが対象（しない構成は `w8` 比の参考値）
- SDPA の発火計数は期待値と一致して緑・期待値 0 の構成は門を立てない
- S f16 丸めは全呼び出しで発火して緑・試行 0 回は赤（恒真化の防止）
- attention を触る構成が基準と同じ latent なら赤（素通し）
"""

from __future__ import annotations

import argparse

import torch

from anima import measure_quant as mq
from anima import measure_report as mr

STEPS = 2
BASELINE = "w8a8"
#: 門の対象（linear 活性を量子化する + S f16 格納）。
GATED = "w8a8-qkpv-s16"
#: 門の対象外（linear 活性 f32 — `w8` 比の参考値だけが出る）。
REFERENCE_ONLY = "w8-qkpv"
CONFIG_NAMES = ("f32", "w8", BASELINE, GATED, REFERENCE_ONLY)
ATTN_CONFIGS = (GATED, REFERENCE_ONLY)
#: f32 からの相対誤差（`w8` と基準）。構成の誤差はこれの倍率で与える。
W8_ERROR = 0.005
BASELINE_ERROR = 0.01


def _diagnostic(
    *,
    linear_act_i8: bool,
    attn_calls: int = 0,
    attn_calls_expected: int = 0,
    score_f16: bool = False,
    s16_fired_calls: int = 0,
    s16_rounded_calls: int = 0,
) -> dict[str, object]:
    return {
        "linear_act_i8": linear_act_i8,
        "attn_calls": attn_calls,
        "attn_calls_expected": attn_calls_expected,
        "score_f16": score_f16,
        "s16_fired_calls": s16_fired_calls,
        "s16_rounded_calls": s16_rounded_calls,
        "s16_elements": 1024,
        "s16_abs_max": 12.5,
    }


def _summarize(
    *,
    pt_value: float = 4.0,
    gated_ratio: float = 1.2,
    reference_only_ratio: float = 1.2,
    attn_calls: int = 8,
    attn_calls_expected: int = 8,
    s16_fired_calls: int = 4,
    s16_rounded_calls: int = 4,
) -> mr.Summary:
    """既定は全門が緑の合成ケース。引数 1 つを動かすと、その門だけが動く。"""
    reference = torch.linspace(1.0, 2.0, 16)
    errors = {
        "f32": 0.0,
        "w8": W8_ERROR,
        BASELINE: BASELINE_ERROR,
        GATED: BASELINE_ERROR * gated_ratio,
        REFERENCE_ONLY: W8_ERROR * reference_only_ratio,
    }
    latents = {
        f"{name}/latents_step{step:04d}": reference * (1.0 + error)
        for name, error in errors.items()
        for step in range(1, STEPS + 1)
    }
    images = {name: torch.zeros(1, 3, 2, 2) for name in CONFIG_NAMES}

    pt_stat = mq.AttnStat()
    mq.hist_add(pt_stat.pt_hist, torch.full((64,), pt_value))
    pt_stat.pt_rows, pt_stat.pt_max, pt_stat.pt_sum = 64, pt_value, 64 * pt_value

    diagnostics = {
        "f32": _diagnostic(linear_act_i8=False),
        "w8": _diagnostic(linear_act_i8=False),
        BASELINE: _diagnostic(linear_act_i8=True, attn_calls=8, attn_calls_expected=8),
        GATED: _diagnostic(
            linear_act_i8=True,
            attn_calls=attn_calls,
            attn_calls_expected=attn_calls_expected,
            score_f16=True,
            s16_fired_calls=s16_fired_calls,
            s16_rounded_calls=s16_rounded_calls,
        ),
        REFERENCE_ONLY: _diagnostic(linear_act_i8=False, attn_calls=8, attn_calls_expected=8),
    }
    run = mq.DitRun(
        latents=latents,
        linear_stats={},
        attn_stats={BASELINE: {"transformer_blocks.0.attn1": pt_stat}},
        diagnostics=diagnostics,
        attention_nodes=["transformer_blocks.0.attn1"],
    )
    return mr.build_summary(
        argparse.Namespace(steps=STEPS, inject=None),
        run,
        latents,
        images,
        config_names=CONFIG_NAMES,
        attn_configs=ATTN_CONFIGS,
        baseline=BASELINE,
        pt_hist_bins=mq.PT_HIST_BINS,
        rel_rms=mq.rel_rms,
        psnr=mq.psnr,
        hist_quantile=mq.hist_quantile,
    )


def _only_failure(summary: mr.Summary) -> str:
    assert len(summary["failed"]) == 1, summary["failed"]
    return summary["failed"][0]


class TestAllGatesGreen:
    def test_a_run_inside_every_gate_fails_nothing(self):
        summary = _summarize()

        assert summary["failed"] == []
        assert all("NG" not in verdict for verdict in summary["gates"].values())


class TestPtMedianGate:
    def test_a_median_below_the_ceiling_passes(self):
        summary = _summarize(pt_value=4.0)

        assert summary["pt"]["median"] <= mr.PT_MEDIAN_GATE
        assert summary["failed"] == []

    def test_a_median_above_the_ceiling_fails(self):
        summary = _summarize(pt_value=32.0)

        assert summary["pt"]["median"] > mr.PT_MEDIAN_GATE
        assert "median" in _only_failure(summary)


class TestLatentRatioGate:
    def test_a_linear_act_config_within_the_ratio_passes(self):
        summary = _summarize(gated_ratio=1.2)

        assert summary["configs"][GATED]["latent_ratio_max"] <= mr.LATENT_RATIO_GATE
        assert summary["failed"] == []

    def test_a_linear_act_config_beyond_the_ratio_fails(self):
        summary = _summarize(gated_ratio=2.0)

        failure = _only_failure(summary)
        assert f"`{GATED}` の latent relRMS" in failure

    def test_a_config_without_linear_act_is_only_reported_against_w8(self):
        """(g) 相当は門の対象外 — 劣化が大きくても赤にせず、`w8` 比の参考値として出す。"""
        summary = _summarize(reference_only_ratio=10.0)

        assert summary["failed"] == []
        reference = summary["gates"][f"`{REFERENCE_ONLY}` の latent relRMS（参考・門の対象外）"]
        assert "`w8` 比 最大 10.000×" in reference


class TestSdpaCallCountGate:
    def test_a_count_matching_the_expectation_passes(self):
        summary = _summarize(attn_calls=8, attn_calls_expected=8)

        assert f"`{GATED}` の SDPA 発火計数" in summary["gates"]
        assert summary["failed"] == []

    def test_a_count_off_the_expectation_fails(self):
        summary = _summarize(attn_calls=7, attn_calls_expected=8)

        assert _only_failure(summary) == f"`{GATED}` の SDPA 発火計数"

    def test_a_config_without_an_expectation_raises_no_gate(self):
        summary = _summarize()

        assert "`f32` の SDPA 発火計数" not in summary["gates"]


class TestScoreF16Gate:
    def test_rounding_that_fires_on_every_call_passes(self):
        summary = _summarize(s16_fired_calls=4, s16_rounded_calls=4)

        assert f"`{GATED}` の S f16 丸めが全呼び出しで発火" in summary["gates"]
        assert summary["failed"] == []

    def test_rounding_that_stays_identity_on_some_call_fails(self):
        summary = _summarize(s16_fired_calls=3, s16_rounded_calls=4)

        assert _only_failure(summary) == f"`{GATED}` の S f16 丸めが全呼び出しで発火"

    def test_no_rounding_attempt_at_all_fails(self):
        """試行 0 回は `0 == 0` で恒真になる — 丸めが一度も呼ばれない構成を緑にしない。"""
        summary = _summarize(s16_fired_calls=0, s16_rounded_calls=0)

        assert _only_failure(summary) == f"`{GATED}` の S f16 丸めが全呼び出しで発火"


class TestPassthroughGate:
    def test_an_attention_config_identical_to_the_baseline_fails(self):
        """ratio 1.0 = latent が基準と同一（量子化が素通しになっている）。"""
        summary = _summarize(gated_ratio=1.0)

        assert _only_failure(summary) == f"`{GATED}` が `{BASELINE}` と異なる"
        assert summary["gates"][f"`{GATED}` が `{BASELINE}` と異なる"] == "NG（素通し）"
