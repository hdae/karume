"""Contract of the `nn.GRU` → `karume::gru_scan` replacement layer (ADR 0056 decision 7).

There is exactly one claim: **the replacement is bit-identical to `nn.GRU`**. If it breaks, "the
graph bakes but the numbers are something else", and if the golden expected values were also taken
from the replacement path, the state *both agree and both are wrong* would go green (the trap the
`karume/custom_ops.py` docstring names).

Shapes to exercise:

- **Varying lengths** (T=1 included) — errors in the reverse scan boundary and in the
  concatenation order between layers include some that appear only at short T and some only at
  long T.
- **Multiple layers** (the order of `cat` between layers matches the output layout of `nn.GRU`).
- **Batch > 1** (mixing up `h0`'s batch axis and the time axis).
- **Unsupported shapes fail loudly** (silently choosing another path changes the numbers quietly).
"""

from __future__ import annotations

import re

import pytest
import torch
from torch import nn

from vowel_detector.patch import assert_supported, gru_forward

#: The same shape as the vowel-recognition CRNN (bidirectional, 2 layers, batch_first), scaled down.
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
        """MUST: `torch.equal`, not difference 0 (`0.0 == -0.0` has difference 0 but different
        bits)."""
        gru = _gru()
        x = torch.randn(1, length, IN_FEATURES)

        with torch.no_grad():
            expected, _ = gru(x)
            got = gru_forward(gru, x)

        assert torch.equal(got, expected), float((got - expected).abs().max())

    @pytest.mark.parametrize("layers", [1, 2, 3])
    def test_every_layer_count_is_bit_identical(self, layers: int) -> None:
        """Between layers, "the concatenation of both directions" goes to the next layer — a
        reversed order would change the values quietly."""
        gru = _gru(layers=layers)
        x = torch.randn(1, 9, IN_FEATURES)

        with torch.no_grad():
            expected, _ = gru(x)
            got = gru_forward(gru, x)

        assert torch.equal(got, expected)

    def test_a_batch_of_more_than_one_is_bit_identical(self) -> None:
        """`h0` is `[N, H]`. A mix-up of the batch axis and the time axis goes unnoticed at N=1."""
        gru = _gru()
        x = torch.randn(3, 7, IN_FEATURES)

        with torch.no_grad():
            expected, _ = gru(x)
            got = gru_forward(gru, x)

        assert torch.equal(got, expected)

    def test_the_forward_half_comes_first(self) -> None:
        """The first half of the output is the forward direction and the second half the reverse
        (the order of `cat` is the same as `nn.GRU`).

        Agreement with `nn.GRU` alone does not necessarily catch "both directions were run with the
        same equation" (the weights differ). The reverse direction alone is rebuilt as a
        unidirectional GRU, and the second half is checked directly to be its **time reversal**.
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
        with pytest.raises(TypeError, match=re.escape("not an nn.GRU")):
            assert_supported(nn.LSTM(IN_FEATURES, HIDDEN))  # type: ignore[arg-type]

    def test_it_refuses_time_first_input(self) -> None:
        with pytest.raises(ValueError, match="batch_first=False"):
            assert_supported(_gru(batch_first=False))

    def test_it_refuses_a_unidirectional_gru(self) -> None:
        with pytest.raises(ValueError, match="bidirectional=False"):
            assert_supported(_gru(bidirectional=False))

    def test_it_refuses_a_bias_free_gru(self) -> None:
        """The `gru_scan` contract requires bias (ADR 0056 decision 6)."""
        with pytest.raises(ValueError, match="bias=False"):
            assert_supported(_gru(bias=False))

    def test_it_refuses_dropout(self) -> None:
        with pytest.raises(ValueError, match="dropout"):
            assert_supported(_gru(layers=2, dropout=0.5))

    def test_the_forward_path_runs_the_same_gate(self) -> None:
        """{@link gru_forward} itself runs the check (unsupported shapes fail even if the caller
        forgets it)."""
        with pytest.raises(ValueError, match="bidirectional=False"):
            gru_forward(_gru(bidirectional=False), torch.randn(1, 4, IN_FEATURES))
