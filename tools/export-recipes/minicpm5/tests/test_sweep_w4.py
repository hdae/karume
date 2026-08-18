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

import pytest
import torch

from karume.quantize import QuantizeError
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
