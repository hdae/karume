import math

import pytest
import torch
from scoring import accuracy, token_nll, windows


@pytest.mark.parametrize("tokens", [2, 3, 63, 64, 65, 127, 128, 129, 191, 192, 193, 2048])
def test_windows_score_each_next_token_once(tokens):
    actual = [
        start + i
        for start, end, first in windows(tokens, 128, 64)
        for i in range(first, end - start)
    ]
    assert actual == list(range(1, tokens))


def test_next_token_loss_uses_previous_row_and_excludes_context():
    probabilities = torch.tensor([[[0.75, 0.25], [0.8, 0.2], [0.01, 0.99]]])
    logits = probabilities.log()
    ids = [0, 1, 0]
    assert token_nll(logits, ids, 1) == pytest.approx([-math.log(0.25), -math.log(0.8)], abs=1e-7)
    assert token_nll(logits, ids, 2) == pytest.approx([-math.log(0.8)], abs=1e-7)


def test_invalid_empty_and_nonfinite_scores_are_rejected():
    with pytest.raises(ValueError):
        list(windows(128, 128, 128))
    with pytest.raises(ValueError):
        list(windows(1, 128, 64))
    with pytest.raises(ValueError):
        token_nll(torch.zeros(1, 2, 3), [0, 1], 2)
    with pytest.raises(ValueError):
        token_nll(torch.full((1, 2, 3), float("nan")), [0, 1], 1)
    assert accuracy(32, 64)["accuracy"] == 0.5
    assert accuracy(32, 64)["wilson95"] == pytest.approx([0.3810209474394701, 0.6189790525605299])
