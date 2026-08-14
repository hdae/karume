"""`nn.GRU` → `karume::gru_scan` の差し替え層の約束事（ADR 0056 決定 7）。

主張は 1 つきり: **差し替えは `nn.GRU` とビット一致する**。ここが崩れると「グラフは焼けるが
数値が別物」になり、golden の期待値も差し替え経路から採ると*両方一致して両方間違っている*
状態が緑になる（`karume/custom_ops.py` の docstring が名指しする罠）。

踏むべき形:

- **長さを振る**（T=1 も含む）— 逆方向の走査境界と層間の連結順の誤りは、短い T でしか
  現れないもの / 長い T でしか現れないものの両方がある。
- **多層**（層間の `cat` の並びが `nn.GRU` の出力レイアウトと一致していること）。
- **バッチ > 1**（`h0` の batch 軸と時間軸の取り違え）。
- **対応外の形は fail loudly**（黙って別の経路を選ぶと数値が静かに変わる）。
"""

from __future__ import annotations

import re

import pytest
import torch
from torch import nn

from karume.patch_vowel_detector import assert_supported, gru_forward

#: 母音認識 CRNN と同じ形（双方向 2 層・batch_first）を小さくしたもの。
IN_FEATURES = 4
HIDDEN = 5


def _gru(
    *,
    layers: int = 2,
    batch_first: bool = True,
    bidirectional: bool = True,
    bias: bool = True,
    dropout: float = 0.0,
    seed: int = 0,
) -> nn.GRU:
    torch.manual_seed(seed)
    return nn.GRU(
        IN_FEATURES,
        HIDDEN,
        num_layers=layers,
        batch_first=batch_first,
        bidirectional=bidirectional,
        bias=bias,
        dropout=dropout,
    ).eval()


class TestEquivalence:
    @pytest.mark.parametrize("length", [1, 2, 3, 16, 137])
    def test_it_is_bit_identical_to_nn_gru(self, length: int) -> None:
        """MUST: 差 0 ではなく `torch.equal`（`0.0 == -0.0` は差 0 でもビットが違う）。"""
        gru = _gru()
        x = torch.randn(1, length, IN_FEATURES)

        with torch.no_grad():
            expected, _ = gru(x)
            got = gru_forward(gru, x)

        assert torch.equal(got, expected), float((got - expected).abs().max())

    @pytest.mark.parametrize("layers", [1, 2, 3])
    def test_every_layer_count_is_bit_identical(self, layers: int) -> None:
        """層間は「両方向の連結」を次の層へ渡す — 並びが逆だと値が静かに変わる。"""
        gru = _gru(layers=layers)
        x = torch.randn(1, 9, IN_FEATURES)

        with torch.no_grad():
            expected, _ = gru(x)
            got = gru_forward(gru, x)

        assert torch.equal(got, expected)

    def test_a_batch_of_more_than_one_is_bit_identical(self) -> None:
        """`h0` は `[N, H]`。batch 軸と時間軸を取り違えると N=1 では気づけない。"""
        gru = _gru()
        x = torch.randn(3, 7, IN_FEATURES)

        with torch.no_grad():
            expected, _ = gru(x)
            got = gru_forward(gru, x)

        assert torch.equal(got, expected)

    def test_the_forward_half_comes_first(self) -> None:
        """出力の前半が順方向・後半が逆方向（`cat` の並びが `nn.GRU` と同じ）。

        `nn.GRU` との一致だけだと「両方向とも同じ式で回してしまった」形が
        （重みが違うので）落ちるとは限らない。逆方向だけを単方向 GRU として作り直し、
        後半がその**時間反転**であることを直接見る。
        """
        gru = _gru(layers=1)
        x = torch.randn(1, 6, IN_FEATURES)
        reverse = nn.GRU(IN_FEATURES, HIDDEN, batch_first=True).eval()
        reverse.load_state_dict(
            {
                "weight_ih_l0": gru.weight_ih_l0_reverse,
                "weight_hh_l0": gru.weight_hh_l0_reverse,
                "bias_ih_l0": gru.bias_ih_l0_reverse,
                "bias_hh_l0": gru.bias_hh_l0_reverse,
            }
        )

        with torch.no_grad():
            got = gru_forward(gru, x)
            backward, _ = reverse(x.flip(1))

        assert torch.equal(got[..., HIDDEN:], backward.flip(1))


class TestSupportedShapes:
    def test_it_refuses_a_module_that_is_not_a_gru(self) -> None:
        with pytest.raises(TypeError, match=re.escape("nn.GRU でない")):
            assert_supported(nn.LSTM(IN_FEATURES, HIDDEN))  # type: ignore[arg-type]

    def test_it_refuses_time_first_input(self) -> None:
        with pytest.raises(ValueError, match="batch_first=False"):
            assert_supported(_gru(batch_first=False))

    def test_it_refuses_a_unidirectional_gru(self) -> None:
        with pytest.raises(ValueError, match="bidirectional=False"):
            assert_supported(_gru(bidirectional=False))

    def test_it_refuses_a_bias_free_gru(self) -> None:
        """`gru_scan` の契約は bias 必須（ADR 0056 決定 6）。"""
        with pytest.raises(ValueError, match="bias=False"):
            assert_supported(_gru(bias=False))

    def test_it_refuses_dropout(self) -> None:
        with pytest.raises(ValueError, match="dropout"):
            assert_supported(_gru(layers=2, dropout=0.5))

    def test_the_forward_path_runs_the_same_gate(self) -> None:
        """{@link gru_forward} 自身が検査を通す（呼び手が忘れても対応外は落ちる）。"""
        with pytest.raises(ValueError, match="bidirectional=False"):
            gru_forward(_gru(bidirectional=False), torch.randn(1, 4, IN_FEATURES))
